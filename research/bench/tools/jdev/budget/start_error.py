"""How far the solver moves each feature (plan U3, 2026-10-16): the scale a continuation schedule has to bridge.

For every row of a solve (CPU path, JointIKSolver.trace_hook): the scorer's features (Q1 M-D distance, Q2 D-M-D angle,
S_orient angles) at step 0 and at the last step, and their change |f_end - f_0|, also in units of the thermal width
(scorer.THERMAL_SD / THERMAL_Q1_BY_DONOR). The continuation start s0 of plan U3 is the median of that change: at the
start the objective must be smooth on the scale the solver still has to travel.

    python tools/jdev/budget/start_error.py --sets dev32 --seed 11 [--limit 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pools import OUT, load_stats, set_cases, setup  # noqa: E402


def _feats(sc, X):
    """{quantity: (B, n) array} of Q1 (A), Q2 (rad), SO (rad) at X (B, N, 3)."""
    import torch
    from oracle_targets import _so_features
    ge = sc._ge
    x = torch.as_tensor(X, dtype=torch.float32)
    out = {"Q1": ge.q1_distances(x).numpy(), "Q2": ge.q2_dmd_angles(x).numpy()}
    M = x[:, ge.metal_pos_idx, :]
    so = [torch.stack([torch.acos(_so_features(x[b:b + 1], M[b:b + 1], g).reshape(-1).clamp(-1, 1))
                       for b in range(len(x))]) for g in getattr(sc, "_so", [])]
    if so:
        out["SO"] = torch.cat(so, 1).numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["dev32"])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    setup()
    import jacobik.ik.scorer as S
    from jacobik.ik.ik_solver import JointIKSolver
    from tools.jdev.cases import get_case
    st = load_stats()
    rows = []
    for s in a.sets:
        cases = list(set_cases(s))[: a.limit or None]
        for code, name in cases:
            rec = {}

            def hook(t, d, rec=rec):
                if t == 0:
                    rec["X0"] = d["X"].cpu().numpy()
                al = d["alive"]
                rec["Xt"] = d["X"].cpu().numpy()
                rec["alive"] = np.arange(len(rec["X0"])) if al is None else al.cpu().numpy()

            JointIKSolver.trace_hook = staticmethod(hook)
            try:
                r = get_case(name).make_builder().solve(seed=a.seed, stats=st, angle_bends=True, device="cpu")
            except Exception as exc:
                print(f"{code}: skipped ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
                continue
            finally:
                JointIKSolver.trace_hook = None
            if "X0" not in rec:
                continue
            sc = r.scorer
            f0, f1 = _feats(sc, rec["X0"][rec["alive"]]), _feats(sc, rec["Xt"])
            sd = {"Q1": sc._q1_thermal_sd.numpy()[None], "Q2": S.THERMAL_SD["Q2"], "SO": S.THERMAL_SD["SO"]}
            for q in f0:
                d = np.abs(f1[q] - f0[q])
                rows.append(dict(set=s, case=code, quantity=q, med=float(np.median(d)), p90=float(np.percentile(d, 90)),
                                 med_sigma=float(np.median(d / sd[q])), n=d.size))
            print(f"{code}: rows {len(rec['Xt'])} of {len(rec['X0'])}", flush=True)
    df = pd.DataFrame(rows)
    out = OUT / "start_error"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"start_error_s{a.seed}.csv", index=False)
    unit = {"Q1": ("A", 1.0), "Q2": ("deg", 180 / np.pi), "SO": ("deg", 180 / np.pi)}
    print("\nmedian over complexes of the per-complex median |f_end - f_start| (and p90), in units and in thermal sd")
    for q, g in df.groupby("quantity"):
        u, k = unit[q]
        print(f"{q}: {g.med.median() * k:.3f} {u} (p90 {g.p90.median() * k:.3f}); {g.med_sigma.median():.1f} sigma_T; "
              f"complexes {len(g)}")


if __name__ == "__main__":
    main()
