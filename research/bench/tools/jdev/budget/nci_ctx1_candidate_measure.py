"""MACE and strict-contact measurements for the held-out NCI-CTX1 solves."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from nci_ctx1_cohort import DEFAULT_SCAN, _load_cases
from nci_hb_context import SUPPORT_MAX_HA, features, select_high_confidence

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COHORT = DEFAULT_SCAN / "cohort"
DEFAULT_SOLVES = DEFAULT_COHORT / "pools/nci_ctx1_surface_v1"
KJ_PER_EV = 96.4853321233


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _rows(cohort: Path, splits: list[str], codes: list[str] | None, limit: int) -> list[tuple[str, dict]]:
    selection = json.loads((cohort / "selection.json").read_text())
    rows = [(split, row) for split in splits for row in selection["selected_rows"][split]
            if not codes or row["code"] in codes]
    return rows[:limit] if limit else rows


def measure_mace(jobs, solves: Path, out: Path, dtype: str, device: str) -> list[dict]:
    import torch
    from jacobik.core.energy import MACEPolarEvaluator
    model = ROOT / "research/bench/tools/jdev/MACE-POLAR-1-S.model"
    ev = MACEPolarEvaluator(model="polar-1-s", device=device, dtype=dtype,
                            model_path=model, threads=1)
    ev.load()
    records, errors = [], []
    target_root = out / f"mace_polar1s_{dtype}_top32"
    for i, (split, row) in enumerate(jobs, 1):
        code = row["code"]
        pool_path = solves / split / f"{code}_s0.npz"
        target = target_root / split / f"{code}_s0.npz"
        if target.exists():
            z = np.load(target, allow_pickle=False)
            records.append({"split": split, "code": code,
                            "n": len(z["dE_kJ_mol"]), "best_dE": float(np.min(z["dE_kJ_mol"]))})
            continue
        try:
            z = np.load(pool_path, allow_pickle=False)
            if "error" in z.files:
                raise RuntimeError(str(z["error"]))
            symbols = [str(x) for x in z["ref_symbols"]]
            X = np.asarray(z["X_raw"], dtype=np.float64)
            ref_X = np.asarray(z["ref_X"], dtype=np.float64)
            if sorted(symbols) != sorted(str(x) for x in z["symbols"]):
                raise ValueError("raw and solver coordinate compositions differ")
            order = np.argsort(-np.asarray(z["score"], dtype=np.float64), kind="stable")[:32]
            jobs_mace = [(symbols, X[j], int(z["charge"]), int(z["mult"])) for j in order]
            jobs_mace.append((symbols, ref_X, int(z["charge"]), int(z["mult"])))
            E = ev.energies(jobs_mace)
            dE = (np.asarray(E[:-1]) - float(E[-1])) * KJ_PER_EV
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, code=code, split=split, pool_indices=order.astype(np.int32),
                                dE_kJ_mol=dE, pool_energy_eV=np.asarray(E[:-1]),
                                reference_energy_eV=float(E[-1]),
                                solver_score=np.asarray(z["score"])[order],
                                pool_sha256=sha256(pool_path), model_sha256=sha256(model),
                                model="MACE-POLAR-1-S", device=str(ev.device), dtype=dtype,
                                gpu_name=torch.cuda.get_device_name(0) if ev.on_cuda else "cpu")
            rec = {"split": split, "code": code, "n": len(dE), "best_dE": float(np.min(dE)),
                   "top_score_dE": float(dE[0]), "best_mace_rank": int(np.argmin(dE))}
            records.append(rec)
            print(f"MACE {i}/{len(jobs)} {split} {code}: {len(dE)} poses, best {rec['best_dE']:.2f} kJ/mol", flush=True)
        except Exception as exc:
            errors.append({"split": split, "code": code, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            print(f"MACE {i}/{len(jobs)} {split} {code}: ERROR {type(exc).__name__}: {str(exc)[:250]}", flush=True)
    summary = {"schema": "nci_ctx1_candidate_mace_v1", "model": "MACE-POLAR-1-S",
               "device": ev.describe(), "dtype": dtype,
               "reference": "same-composition MACE E(candidate pose)-E(original TMQM reference), kJ/mol",
               "pool_rule": "top 32 by candidate solver score", "records": records, "errors": errors}
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / f"summary_{'_'.join(sorted(set(s for s, _ in jobs)))}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if errors:
        raise RuntimeError(f"MACE failures: {errors}")
    return records


def measure_contacts(jobs, scan: Path, cohort: Path, solves: Path, mace_dir: Path, out: Path) -> dict:
    selection = json.loads((cohort / "selection.json").read_text())
    chosen = {split: [row for s, row in jobs if s == split] for split in sorted({s for s, _ in jobs})}
    cases = _load_cases(scan, chosen)
    native = json.loads((cohort / "strict_hb_opportunities_v1/native_positive_contacts.json").read_text())
    native_by_code = defaultdict(set)
    for item in native:
        native_by_code[item["code"]].add((int(item["donor"]), int(item["hydrogen"]), int(item["acceptor"])))
    total = defaultdict(int)
    parent_summaries, errors = [], []
    for i, (split, row) in enumerate(jobs, 1):
        code = row["code"]
        pool_path = solves / split / f"{code}_s0.npz"
        mace_path = mace_dir / split / f"{code}_s0.npz"
        target = out / "opportunities" / split / f"{code}_s0.npz"
        try:
            if target.exists():
                zold = np.load(target, allow_pickle=False)
                if "summary" in zold.files:
                    old = json.loads(str(zold["summary"].item()))
                    parent_summaries.append(old)
                    total[(split, "poses")] += int(old["pool_poses"])
                    total[(split, "strict_contacts")] += int(old["strict_contacts"])
                    total[(split, "strict_pose_hits")] += int(old["poses_with_strict_contact"])
                    total[(split, "native_exact_hits")] += int(old["native_exact_pair_hits"])
                    continue
                zold.close()
                target.unlink()
            pool = np.load(pool_path, allow_pickle=False)
            mz = np.load(mace_path, allow_pickle=False)
            X = np.asarray(pool["X_raw"], dtype=np.float64)
            indices = np.asarray(mz["pool_indices"], dtype=np.int64)
            mol = cases[code].tmqm_mol
            from nci_hb_chemistry import from_training
            chemistry = from_training(mol)
            exact_native = native_by_code[code]
            flat = []
            strict_pairs_by_pose = []
            for rank, pool_index in enumerate(indices):
                fts = features(mol, X[pool_index], chemistry=chemistry)
                strict = select_high_confidence(fts)
                strict_keys = {(x.donor, x.hydrogen, x.acceptor) for x in strict}
                strict_pairs_by_pose.append(strict_keys)
                native_exact = strict_keys & exact_native
                total[(split, "poses")] += 1
                total[(split, "strict_contacts")] += len(strict)
                total[(split, "strict_pose_hits")] += bool(strict)
                total[(split, "native_exact_hits")] += len(native_exact)
                for x in fts:
                    if x.r_ha > SUPPORT_MAX_HA:
                        continue
                    flat.append((rank, x, (x.donor, x.hydrogen, x.acceptor) in strict_keys,
                                 (x.donor, x.hydrogen, x.acceptor) in exact_native))
            target.parent.mkdir(parents=True, exist_ok=True)
            summary = {"code": code, "split": split, "dataset": row["dataset"],
                       "pool_poses": len(indices), "opportunities": len(flat),
                       "strict_contacts": int(sum(len(x) for x in strict_pairs_by_pose)),
                       "poses_with_strict_contact": int(sum(bool(x) for x in strict_pairs_by_pose)),
                       "native_exact_pair_hits": int(sum(len(x & exact_native) for x in strict_pairs_by_pose))}
            np.savez_compressed(
                target, code=code, split=split, dataset=row["dataset"], seed=0,
                pool_sha256=sha256(pool_path), mace_sha256=sha256(mace_path),
                chemistry_version="hb_ctx1", support_max_ha=SUPPORT_MAX_HA,
                pool_rank=np.asarray([rank for rank, _, _, _ in flat], dtype=np.int16),
                donor=np.asarray([x.donor for _, x, _, _ in flat], dtype=np.int32),
                hydrogen=np.asarray([x.hydrogen for _, x, _, _ in flat], dtype=np.int32),
                acceptor=np.asarray([x.acceptor for _, x, _, _ in flat], dtype=np.int32),
                key_levels=np.asarray([json.dumps(x.keys, separators=(",", ":")) for _, x, _, _ in flat], dtype="U512"),
                r_ha=np.asarray([x.r_ha for _, x, _, _ in flat], dtype=np.float32),
                cos_dha=np.asarray([x.cos_dha for _, x, _, _ in flat], dtype=np.float32),
                cos_acceptor=np.asarray([x.cos_acceptor for _, x, _, _ in flat], dtype=np.float32),
                acceptor_axis_defined=np.asarray([x.acceptor_axis_defined for _, x, _, _ in flat], dtype=np.bool_),
                strict_selected=np.asarray([strict for _, _, strict, _ in flat], dtype=np.bool_),
                native_exact_pair=np.asarray([native_match for _, _, _, native_match in flat], dtype=np.bool_),
                pool_index=indices.astype(np.int16), dE_kJ_mol=np.asarray(mz["dE_kJ_mol"], dtype=np.float64),
                summary=json.dumps(summary),
            )
            parent_summaries.append(summary)
            print(f"contacts {i}/{len(jobs)} {split} {code}: strict={summary['strict_contacts']}, "
                  f"exact native pair hits={summary['native_exact_pair_hits']}", flush=True)
        except Exception as exc:
            errors.append({"split": split, "code": code, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            print(f"contacts {i}/{len(jobs)} {split} {code}: ERROR {type(exc).__name__}: {str(exc)[:250]}", flush=True)
    summary = {"schema": "nci_ctx1_candidate_contacts_v1", "scope": "strict contacts on top MACE-measured candidate poses",
               "support_radius_A": SUPPORT_MAX_HA, "parents": parent_summaries, "errors": errors,
               "aggregate": {k[0] + ":" + k[1]: int(v) for k, v in total.items()},
               "out": str(out)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if errors:
        raise RuntimeError(f"contact extraction failures: {errors}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("mace", "contacts"))
    p.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--solves", type=Path, default=DEFAULT_SOLVES)
    p.add_argument("--mace-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=("calibration", "validation"), default=("validation",))
    p.add_argument("--codes", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    a = p.parse_args()
    scan, cohort, solves = a.scan.resolve(), a.cohort.resolve(), a.solves.resolve()
    jobs = _rows(cohort, a.splits, a.codes, a.limit)
    if a.stage == "mace":
        out = (a.out or solves).resolve()
        measure_mace(jobs, solves, out, a.dtype, a.device)
    else:
        mace_dir = (a.mace_dir or solves / f"mace_polar1s_{a.dtype}_top32").resolve()
        out = (a.out or solves / "strict_hb_opportunities_v1").resolve()
        measure_contacts(jobs, scan, cohort, solves, mace_dir, out)


if __name__ == "__main__":
    main()
