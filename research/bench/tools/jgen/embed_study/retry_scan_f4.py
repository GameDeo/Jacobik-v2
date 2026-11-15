"""Count deterministic ETKDGv3 retry-path uses over the F-2 census."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]
from tools.jgen import inputs  # noqa: E402
from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol  # noqa: E402
from jacobik.builder.embed import ez_variants  # noqa: E402


class RetryLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if "ETKDG" in msg and ("retry succeeded" in msg or "retries exhausted" in msg):
            self.messages.append(msg)


def work(job):
    code, rows = job
    from jacobik.builder.engine import _smiles_to_conformer
    from rdkit import Chem
    loggers = [logging.getLogger("jacobik.builder.engine"), logging.getLogger("jacobik.builder.embed")]
    handler = RetryLog()
    for logger in loggers:
        logger.addHandler(handler)
    result = []
    try:
        shard, dataset = inputs.frame_shards()[code]
        mol = inputs.load_mol(code, shard, dataset)
        if mol is None:
            raise ValueError("parent geometry unavailable")
        groups = _build_ligand_groups(mol, mol.smiles)
        ligands = ligand_inputs_from_mol(mol, groups)
        for row in rows:
            ix = int(row["ligand_idx"])
            rec = dict(code=code, ligand_idx=ix, cohort=row.get("cohort"), status="ok", retried=False,
                       messages=[], error=None)
            try:
                if row.get("status") != "ok" or ix >= len(ligands):
                    raise ValueError("source ligand unavailable")
                lig = ligands[ix]
                for ez in (ez_variants(lig.smiles, lig.donors) or [[]]):
                    handler.messages.clear()
                    _smiles_to_conformer(lig.smiles, embed="mmff", embed_seed=int(row.get("embed_seed", 42)),
                                         donors=lig.donors, chelate_embed=True, ez=ez)
                    rec["messages"].extend(handler.messages)
                    handler.messages.clear()
            except Exception as exc:
                rec["status"] = "failed"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["messages"].extend(handler.messages)
                handler.messages.clear()
            rec["retried"] = bool(rec["messages"])
            result.append(rec)
    except Exception as exc:
        result = [dict(code=code, ligand_idx=int(r["ligand_idx"]), status="failed", retried=False,
                       messages=[], error=f"{type(exc).__name__}: {exc}") for r in rows]
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if not 1 <= a.workers <= 3:
        ap.error("F-4 caps local CPU concurrency at 3")
    jobs = []
    for line in a.census.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        unique = {}
        for row in p.get("ligands", []):
            unique.setdefault(int(row["ligand_idx"]), row)
        jobs.append((p["code"], list(unique.values())))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs = [pool.submit(work, j) for j in jobs]
        for f in as_completed(fs):
            rows.extend(f.result())
    rows.sort(key=lambda r: (r["code"], r["ligand_idx"]))
    summary = dict(n_ligands=len(rows), n_retry=sum(r["retried"] for r in rows),
                   n_failed=sum(r["status"] == "failed" for r in rows),
                   retry_ligands=[f"{r['code']}:{r['ligand_idx']}" for r in rows if r["retried"]], rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
