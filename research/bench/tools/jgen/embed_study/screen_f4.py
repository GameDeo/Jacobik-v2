"""Full-census F-4 frozen-coordinate label screen (one arm per fresh process)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]

from tools.jgen import inputs  # noqa: E402
from tools.jgen.embed_study.analyze_census import truth_flags  # noqa: E402
from tools.jgen.embed_study.measure_embeds import _ligand_metrics, _reference_by_rdkit  # noqa: E402
from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol  # noqa: E402
from jacobik.builder.embed import ez_variants  # noqa: E402


class GFNFailureLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []
        self.retry_messages = []

    def emit(self, record):
        if "GFN2 embed refinement failed" in record.getMessage():
            self.messages.append(record.getMessage())
        if "ETKDG" in record.getMessage() and ("retry succeeded" in record.getMessage() or "retries exhausted" in record.getMessage()):
            self.retry_messages.append(record.getMessage())


def _work(arg):
    code, source_rows = arg
    from rdkit import Chem
    from jacobik.builder.engine import _smiles_to_conformer, pop_embed_bond_check_events, _GFN2_CACHE, _GFN2_BOND_CHECK_CACHE

    try:
        shard, dataset = inputs.frame_shards()[code]
        mol = inputs.load_mol(code, shard, dataset)
        if mol is None:
            raise ValueError("parent geometry unavailable")
        groups = _build_ligand_groups(mol, mol.smiles)
        ligands = ligand_inputs_from_mol(mol, groups)
    except Exception as exc:
        return [{"code": code, "ligand_idx": int(r["ligand_idx"]), "status": "failed",
                 "error": f"{type(exc).__name__}: {exc}", "bad": None, "seconds": 0.0,
                 "guard_events": [], "gfn2_failures": []} for r in source_rows]

    logger = logging.getLogger("jacobik.builder.engine")
    retry_logger = logging.getLogger("jacobik.builder.embed")
    handler = GFNFailureLog()
    logger.addHandler(handler)
    retry_logger.addHandler(handler)
    out = []
    try:
        for row in source_rows:
            ix = int(row["ligand_idx"])
            rec = {"code": code, "ligand_idx": ix, "status": "failed", "bad": None,
                   "error": None, "seconds": 0.0, "guard_events": [], "gfn2_failures": [],
                   "etkdg_retry": False, "etkdg_retry_messages": []}
            t0 = time.perf_counter()
            try:
                if ix >= len(ligands):
                    raise IndexError("ligand index unavailable")
                lig = ligands[ix]
                if row.get("status") != "ok":
                    raise ValueError("source census ligand was already an embed failure")
                base = Chem.MolFromSmiles(lig.smiles)
                if base is None:
                    raise ValueError("RDKit could not parse ligand SMILES")
                full = Chem.AddHs(base)
                ref = _reference_by_rdkit(mol, groups[ix], full)
                bend_pivots = set()
                try:
                    from jacobik.ik.angle_bends import bend_specs
                    xyz_to_rdkit = {int(x): int(r) for r, x in groups[ix]["rdkit_to_xyz"].items()}
                    bend_pivots = {xyz_to_rdkit[int(sp[0])] for sp in bend_specs(mol, mol.ligands[ix])
                                   if int(sp[0]) in xyz_to_rdkit}
                except Exception:
                    pass
                variants = ez_variants(lig.smiles, lig.donors) or [[]]
                bad_variants, errors = [], []
                for ez in variants:
                    _GFN2_CACHE.clear()
                    _GFN2_BOND_CHECK_CACHE.clear()
                    try:
                        rd, xyz, _has_h = _smiles_to_conformer(
                            lig.smiles, embed_seed=int(row.get("embed_seed", 42)), donors=lig.donors,
                            chelate_embed=True, embed="gfn2", ez=ez)
                        metrics = _ligand_metrics(lig.smiles, lig.donors, xyz, ref, bend_pivots)
                        bad_variants.append(any(truth_flags(metrics).values()))
                    except Exception as exc:
                        errors.append(f"{type(exc).__name__}: {exc}")
                    finally:
                        rec["guard_events"].extend(pop_embed_bond_check_events())
                if errors:
                    rec["error"] = "; ".join(errors)
                    rec["status"] = "failed"
                elif bad_variants:
                    rec["bad"] = bool(all(bad_variants))
                    rec["status"] = "ok"
                else:
                    rec["error"] = "no E/Z variants produced a label"
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["seconds"] = time.perf_counter() - t0
            rec["gfn2_failures"] = list(handler.messages)
            rec["etkdg_retry_messages"] = list(handler.retry_messages)
            rec["etkdg_retry"] = bool(handler.retry_messages)
            handler.messages.clear()
            handler.retry_messages.clear()
            out.append(rec)
    finally:
        logger.removeHandler(handler)
        retry_logger.removeHandler(handler)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--arm", choices=("production", "guard", "li_cap", "combined"), required=True)
    args = ap.parse_args()
    if not 1 <= args.workers <= 3:
        ap.error("F-4 caps local CPU concurrency at 3")
    # Flags are import-time constants; each arm must be launched in a fresh interpreter.
    expected = {"production": {"embed_bond_check": False, "embed_li_cap": False},
                "guard": {"embed_bond_check": True, "embed_li_cap": False},
                "li_cap": {"embed_bond_check": False, "embed_li_cap": True},
                "combined": {"embed_bond_check": True, "embed_li_cap": True}}[args.arm]
    from jacobik.core import profile
    from jacobik.builder.engine import EMBED_BOND_CHECK, EMBED_LI_CAP
    actual = {"embed_bond_check": EMBED_BOND_CHECK, "embed_li_cap": EMBED_LI_CAP}
    if actual != expected:
        raise SystemExit(f"wrong import-time flags for {args.arm}: expected {expected}, got {actual}")
    parents = [json.loads(line) for line in args.census.read_text().splitlines() if line.strip()]
    jobs = []
    for p in parents:
        unique = {}
        for row in p.get("ligands", []):
            unique.setdefault(int(row["ligand_idx"]), row)
        jobs.append((p["code"], list(unique.values())))
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs = [pool.submit(_work, job) for job in jobs]
        for f in as_completed(fs):
            records.extend(f.result())
    records.sort(key=lambda r: (r["code"], r["ligand_idx"]))
    summary = {"arm": args.arm, "flags": actual, "fingerprint": profile.snapshot()["fingerprint"],
               "n_ligands": len(records), "n_scoreable": sum(r["status"] == "ok" for r in records),
               "n_failed": sum(r["status"] == "failed" for r in records),
               "mean_seconds": float(np.mean([r["seconds"] for r in records])) if records else None,
               "median_seconds": float(np.median([r["seconds"] for r in records])) if records else None,
               "max_seconds": float(np.max([r["seconds"] for r in records])) if records else None,
               "guard_trigger_count": sum(bool(e.get("triggered")) for r in records for e in r["guard_events"]),
               "fallbacks": dict(Counter(e.get("fallback_used", "unspecified") for r in records
                                          for e in r["guard_events"] if e.get("triggered"))),
               "records": records, "workers": args.workers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
