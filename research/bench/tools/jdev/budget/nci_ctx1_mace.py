"""MACE reference energies for the frozen NCI-CTX1 Q6 baseline cohort.

Research-only measurement: reads completed u1 + repulsive-Q6 pools and writes
small per-parent records under /tmp. It does not change the scorer or profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COHORT = Path("/tmp/nci_ctx1_native_partial/cohort")
MODEL = ROOT / "research/bench/tools/jdev/MACE-POLAR-1-S.model"
KJ_PER_EV = 96.4853321233


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=("calibration", "validation"),
                   default=("calibration", "validation"))
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    a = p.parse_args()
    if a.top_k < 1:
        p.error("--top-k must be at least 1")
    cohort = a.cohort.resolve()
    out = (a.out or cohort / "mace_polar1s_float64_top32").resolve()
    selection_path = cohort / "selection.json"
    selection = json.loads(selection_path.read_text())

    from jacobik.core.energy import MACEPolarEvaluator
    import torch

    ev = MACEPolarEvaluator(model="polar-1-s", device=a.device, dtype=a.dtype,
                            model_path=MODEL, threads=1)
    ev.load()
    jobs = [(split, row) for split in a.splits
            for row in selection["selected_rows"][split]]
    if a.limit:
        jobs = jobs[:a.limit]
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"device": ev.describe(), "model_sha256": sha256(MODEL),
                      "selection_sha256": sha256(selection_path), "parents": len(jobs),
                      "top_k": a.top_k, "output": str(out)}, indent=2), flush=True)

    completed, failures = 0, []
    for i, (split, row) in enumerate(jobs, 1):
        code = row["code"]
        pool_path = cohort / "pools/q6_u1rep" / split / f"{code}_s0.npz"
        target = out / split / f"{code}_s0.npz"
        if target.exists():
            completed += 1
            continue
        try:
            z = np.load(pool_path, allow_pickle=False)
            if "error" in z.files:
                raise RuntimeError(str(z["error"]))
            symbols = [str(x) for x in z["ref_symbols"]]
            X = np.asarray(z["X_raw"], dtype=np.float64)
            ref_X = np.asarray(z["ref_X"], dtype=np.float64)
            q, mult = int(z["charge"]), int(z["mult"])
            if X.ndim != 3 or X.shape[1:] != ref_X.shape or len(symbols) != len(ref_X):
                raise ValueError(f"coordinate/composition mismatch: pool={X.shape}, ref={ref_X.shape}, symbols={len(symbols)}")
            if sorted(symbols) != sorted(str(x) for x in z["symbols"]):
                raise ValueError("engine and raw-order pool compositions differ")
            scores = np.asarray(z["score"], dtype=np.float64)
            order = np.argsort(-scores, kind="stable")[:min(a.top_k, len(scores))]
            structures = [(symbols, X[j], q, mult) for j in order]
            structures.append((symbols, ref_X, q, mult))
            energies = ev.energies(structures)
            ref_ev = float(energies[-1])
            pool_ev = np.asarray(energies[:-1], dtype=np.float64)
            dE = (pool_ev - ref_ev) * KJ_PER_EV
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, code=code, split=split, seed=0,
                                pool_path=str(pool_path), pool_sha256=sha256(pool_path),
                                model_sha256=sha256(MODEL), selection_sha256=sha256(selection_path),
                                model="MACE-POLAR-1-S", device=str(ev.device), dtype=a.dtype,
                                symbols=np.asarray(symbols), charge=q, mult=mult,
                                pool_indices=order.astype(np.int32), solver_score=scores[order],
                                pool_energy_eV=pool_ev, reference_energy_eV=ref_ev,
                                dE_kJ_mol=dE, gpu_name=torch.cuda.get_device_name(0) if ev.on_cuda else "cpu")
            completed += 1
            print(f"{i}/{len(jobs)} {split} {code}: {len(order)} poses, best dE={float(np.min(dE)):.2f} kJ/mol; {ev.describe()}", flush=True)
        except Exception as exc:
            failures.append({"split": split, "code": code,
                             "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            print(f"{i}/{len(jobs)} {split} {code}: ERROR {type(exc).__name__}: {str(exc)[:250]}", flush=True)

    summary = {"schema": "nci_ctx1_mace_measurements_v1", "model": "MACE-POLAR-1-S",
               "model_sha256": sha256(MODEL), "selection_sha256": sha256(selection_path),
               "pool_rule": f"top {a.top_k} by descending frozen u1+repulsive-Q6 solver score",
               "reference": "same-composition MACE E(pool pose) - E(original TMQM reference), kJ/mol",
               "dtype": a.dtype, "device": str(ev.device), "device_summary": ev.describe(),
               "completed": completed, "failures": failures, "output": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": completed, "failures": len(failures),
                      "summary": str(out / "summary.json"), "device": ev.describe()}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
