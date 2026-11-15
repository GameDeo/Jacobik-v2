"""Golden outputs: the no-result-change check for refactors (2026-10-17 cleanup, kept as a regression tool).

Records, under the legacy profile and under u1 (each in its own process: the profile is read at import):
  * RESCORE  the ranking score and per-term breakdown of fixed pools (stored conformers re-scored by a fresh scorer);
  * SOLVE    full CPU solves (seed 11, the pools.py call) of a few representative complexes: conformers and scores.
`check` recomputes everything and compares: scores / terms to 1e-6 (nats), coordinates to 1e-5 A.

    python dev/tools/jdev/golden.py record          # -> data/golden/<profile>.npz
    python dev/tools/jdev/golden.py check           # exit 1 on any difference
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_this_repo = Path(__file__).resolve().parents[3]
from tools.repo import BUDGET_DB_SUPPORT, main_checkout

REPO = main_checkout(_this_repo)
#: data root (JACOBIK_DATA_ROOT, default <repo>/data): a git worktree has no data/ of its own
DATA = Path(os.environ.get("JACOBIK_DATA_ROOT", REPO / "data"))
#: reference files (JACOBIK_GOLDEN_DIR overrides; used to compare two source trees in one environment)
OUTD = Path(os.environ["JACOBIK_GOLDEN_DIR"]) if os.environ.get("JACOBIK_GOLDEN_DIR") else (REPO / "research/data/golden" if (REPO / "research/data/golden").exists() else DATA / "golden")
PROFILES = ("production", "u1")
#: pools re-scored (set, code): the arm whose pools are re-scored depends on the profile
RESCORE = ["OJIMIF", "HUYNEU", "APEBIJ", "IYONIW", "CBTYCO", "KIQYEQ", "SOKVET", "PEWBOL", "XAQRUC", "DEVFUG", "COPYIP"]
POOL_ARM = {"production": "etkdg_u1_rep", "u1": "etkdg_u1_rep"}
#: complexes solved: small Pt, Zr, Co H-bond, group-13 surrogate, E/Z, haptic, lanthanide aqua
SOLVE = ["OJIMIF", "HUYNEU", "CBTYCO", "KIQYEQ", "SOKVET", "YAVFEG", "aqua_La3_8H2O"]
TOL_SCORE, TOL_X = 1e-6, 1e-5
# References were recorded with this partition layout. One whole-batch worker
# changes batched floating-point kernels and can lead iRprop to different rows.
# Keep thread count and the configured partition count separate.
GOLDEN_LAYOUT = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                     NUMEXPR_NUM_THREADS="1", MKL_THREADING_LAYER="SEQUENTIAL",
                     JACOBIK_NUM_WORKERS="8", JACOBIK_WORKER_MODE="thread")


def golden_environment(base, workers=8):
    if workers < 1:
        raise ValueError("golden worker count must be positive")
    return dict(base, **{**GOLDEN_LAYOUT, "JACOBIK_NUM_WORKERS": str(workers)})



def _compute(profile: str) -> dict:
    sys.path.insert(0, str(_this_repo / "src"))
    sys.path.insert(0, str(_this_repo))
    sys.path.insert(0, str(_this_repo / "research/bench"))
    sys.path.insert(0, str(_this_repo / "dev/tools/jdev/budget"))
    from pools import OUT, load_stats, rank_terms, set_cases, setup

    setup()
    import torch
    torch.set_num_threads(1)
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry

    st = load_stats()
    names, set_of = {}, {}
    for s in ("dev32", "n48"):
        for c, n in set_cases(s):
            names[c], set_of[c] = n, s
    out = {}
    for code in RESCORE:
        s = set_of[code]
        for seed in (11, 12, 13):
            f = OUT / "pools" / POOL_ARM[profile] / s / f"{code}_s{seed}.npz"
            z = np.load(f, allow_pickle=False)
            if "error" in z.files:
                continue
            r = get_case(names[code]).make_builder().solve(seed=seed, stats=st, angle_bends=True, n_conformers=1,
                                                           n_ik_steps=2, device="cpu")
            X = torch.tensor(z["X"], dtype=torch.float32)
            with torch.no_grad():
                tot, bd = rank_terms(r.scorer, X, torch.as_tensor(z["asg"], dtype=torch.long))
            out[f"rescore|{code}|{seed}|total"] = tot.numpy()
            for q, v in bd.items():
                out[f"rescore|{code}|{seed}|{q}"] = v.numpy()
    for code in SOLVE:
        case = get_case(names.get(code, code))
        with starting_geometry(case, "etkdg"):
            r = case.make_builder().solve(seed=11, stats=st, angle_bends=True, device="cpu")
        out[f"solve|{code}|X"] = np.array([[a.position for a in c.complex.atoms] for c in r.conformers])
        out[f"solve|{code}|score"] = np.array([c.score for c in r.conformers])
    return out


def _run(profile: str, mode: str, workers: int = 8) -> None:
    """Child process: compute under `profile` and record or check."""
    from threadpoolctl import threadpool_limits
    layout = golden_environment({}, workers)
    os.environ.update(layout)
    print(f"{profile}: pinned CPU layout {layout}", flush=True)
    with threadpool_limits(limits=1):
        got = _compute(profile)
    f = OUTD / f"{profile}.npz"
    if mode == "record":
        OUTD.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **{k.replace("|", "__"): v for k, v in got.items()})
        print(f"{profile}: recorded {len(got)} arrays -> {f}")
        return
    ref = {k.replace("__", "|"): v for k, v in np.load(f).items()}
    bad = []
    for k in sorted(set(ref) | set(got)):
        if k not in ref or k not in got:
            bad.append(f"{k}: {'missing now' if k not in got else 'new'}")
            continue
        a, b = ref[k], got[k]
        tol = TOL_X if k.endswith("|X") else TOL_SCORE
        if a.shape != b.shape:
            bad.append(f"{k}: shape {a.shape} -> {b.shape}")
        elif not np.allclose(a, b, atol=tol, rtol=0, equal_nan=True):
            bad.append(f"{k}: max |diff| {float(np.nanmax(np.abs(a - b))):.3g} (tol {tol})")
    print(f"{profile}: {len(ref)} arrays, {len(bad)} differ" + "".join(f"\n  {x}" for x in bad[:40]))
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["record", "check"])
    p.add_argument("--profile", default=None, help="internal: run one profile in this process")
    p.add_argument("--workers", type=int, default=8,
                   help="Recorded partition maximum (8 for shared references); override only for a different reference layout")
    a = p.parse_args()
    if a.workers < 1:
        p.error("--workers must be positive")
    if a.profile:
        _run(a.profile, a.mode, a.workers)
        return
    env = {k: v for k, v in os.environ.items() if k != "JACOBIK_PROFILE_OVERRIDES"}
    budget_out = (REPO / "research/archive/data/energy_scale_refit"
                  if (REPO / "research/archive/data/energy_scale_refit").exists()
                  else DATA / "energy_scale_refit")
    env.setdefault("JACOBIK_BUDGET_OUT", str(budget_out))
    budget_db = REPO / BUDGET_DB_SUPPORT
    env.setdefault("JACOBIK_BUDGET_DB", str(budget_db))
    # OMP_NUM_THREADS=1: the GFN2 (xtb) free-ligand embed is OpenMP-threaded; under CPU load its reductions change
    # order and the solve can return a different conformer set (2026-10-17: YAVFEG, HUYNEU); one thread is exact.
    env = golden_environment(env, a.workers)
    env.update(PYTHONIOENCODING="utf-8",
               PYTHONPATH=str(_this_repo / "src") + os.pathsep + str(_this_repo))
    rc = 0
    for prof in PROFILES:
        r = subprocess.run([sys.executable, __file__, a.mode, "--profile", prof, "--workers", str(a.workers)], env=dict(env, JACOBIK_PROFILE=prof))
        rc |= r.returncode
    print("golden:", "OK" if rc == 0 else "DIFFERENCES")
    sys.exit(rc)


if __name__ == "__main__":
    main()
