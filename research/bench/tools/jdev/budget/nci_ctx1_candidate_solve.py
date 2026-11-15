"""Run parent-disjoint NCI-CTX1 candidate solves in an isolated process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from nci_ctx1_cohort import DEFAULT_SCAN, _engine_to_tmqm, _load_cases
from nci_ctx1_model import install_for_process

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COHORT = DEFAULT_SCAN / "cohort"
DEFAULT_OUT = DEFAULT_COHORT / "pools/nci_ctx1_surface_v1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--model", type=Path, default=DEFAULT_COHORT / "nci_ctx1_surface_model_v1")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--splits", nargs="+", choices=("calibration", "validation"), default=("validation",))
    p.add_argument("--codes", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="device for the differentiable IK solve (default: cpu)")
    a = p.parse_args()
    scan, cohort, out = a.scan.resolve(), a.cohort.resolve(), a.out.resolve()
    selection_path = cohort / "selection.json"
    selection = json.loads(selection_path.read_text())
    todo = [(s, row) for s in a.splits for row in selection["selected_rows"][s]
            if not a.codes or row["code"] in a.codes]
    if a.limit:
        todo = todo[:a.limit]
    selected = {s: [row for split, row in todo if split == s] for s in a.splits}
    if not todo:
        raise SystemExit("no selected parents matched the requested split/codes")

    from jacobik import ComplexBuilder
    from jacobik.core import profile
    if profile.ACTIVE_NAME != "u1" or profile.get("q6_model") != "repulsive" or not profile.get("nci"):
        raise RuntimeError(f"candidate must be u1 + repulsive Q6 + research NCI; got {profile.snapshot()}")

    install_for_process(a.model)
    original_build_shell = ComplexBuilder._build_shell
    def build_shell_with_rdkit(self, *args, **kwargs):
        mol, rdmols, donor_idx_full = original_build_shell(self, *args, **kwargs)
        mol._nci_ctx1_rdmols = rdmols
        return mol, rdmols, donor_idx_full
    ComplexBuilder._build_shell = build_shell_with_rdkit

    from tools.jdev.start import starting_geometry
    from tools.jdev.budget.common import stats_from_db
    cases = _load_cases(scan, selected)
    stats = stats_from_db(ROOT / "data/refit/final/constraints_2026-10-18.json")
    out.mkdir(parents=True, exist_ok=True)
    profile_record = json.dumps(profile.snapshot(), sort_keys=True, default=str)
    failures = []
    for i, (split, row) in enumerate(todo, 1):
        code = row["code"]
        target = out / split / f"{code}_s{a.seed}.npz"
        if target.exists():
            previous = np.load(target, allow_pickle=False)
            if "error" not in previous.files:
                print(f"{i}/{len(todo)} {split} {code}: exists; skip", flush=True)
                continue
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            case = cases[code]
            t0 = time.perf_counter()
            with starting_geometry(case, "etkdg"):
                result = case.make_builder().solve(seed=a.seed, stats=stats, angle_bends=True, device=a.device)
            elapsed = time.perf_counter() - t0
            confs = result.conformers
            X_engine = np.asarray([[atom.position for atom in c.complex.atoms] for c in confs], dtype=np.float64)
            X_raw = np.stack([_engine_to_tmqm(case, c.complex, X) for c, X in zip(confs, X_engine)])
            symbols = np.asarray([atom.symbol for atom in confs[0].complex.atoms])
            if len(symbols) != len(case.ref_symbols):
                raise ValueError(f"{code}: solved/reference atom-count mismatch")
            np.savez_compressed(target, X=X_engine, X_raw=X_raw, symbols=symbols,
                                score=np.asarray([c.score for c in confs], dtype=np.float64),
                                ref_X=case.ref_X, ref_symbols=np.asarray(case.ref_symbols),
                                charge=case.charge, mult=case.mult, seed=a.seed,
                                t_solve=elapsed,
                                profile=profile_record, start="etkdg", device=a.device,
                                model=str(a.model.resolve()))
            print(f"{i}/{len(todo)} {split} {code}: {len(confs)} conformers, "
                  f"NCI={profile.get('nci')}, elapsed={elapsed:.1f}s", flush=True)
        except Exception as exc:
            failures.append({"split": split, "code": code,
                             "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            np.savez_compressed(target, error=failures[-1]["error"])
            print(f"{i}/{len(todo)} {split} {code}: ERROR {type(exc).__name__}: {str(exc)[:250]}", flush=True)
    summary = {"schema": "nci_ctx1_candidate_solve_v1", "selection_sha256": __import__("hashlib").sha256(selection_path.read_bytes()).hexdigest(),
               "model": str(a.model.resolve()), "profile": profile.snapshot(),
               "seed": a.seed, "device": a.device, "parents": len(todo), "failures": failures,
               "output": str(out)}
    (out / f"summary_{'_'.join(a.splits)}_s{a.seed}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"parents": len(todo), "failures": len(failures),
                      "summary": str(out / f"summary_{'_'.join(a.splits)}_s{a.seed}.json")}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
