"""Which array ranks the native top-1, and what separates `score` from `rank` (CPU only).

Over all successful control pools (effective: raw + S2.1 overlay, see poolio.py):
  identity_err   max |rank * Z - sum(terms_Q1..Q6)| per pool (float32 rounding expected, ~1e-5)
  argmax_same    share of pools where argmax(score) == argmax(rank)
  score_minus_rank   max |score - rank| per pool; corr(score, rank)
  top1_dE_score / top1_dE_rank   means over pools of dE at each array's argmax
  rho_score, rho_rank            Spearman with -dE over rows (medians over pools with >= 3 rows)
Also, for the S2b regenerated units (s2b_eval_state_v1/pools/*.npz, which store bend_prior_solver per row):
  max |(score - rank) - bend_prior_solver| per unit, and corr(score - rank, bend_prior_solver).

The native top-1 of a pool is argmax(score): orchestrator `top1_idx = int(np.argmax(scores))`
(nci_v2_orchestrator.py:746) with `scores` saved as `score` (nci_v2_orchestrator.py:654).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d10_score_rank_check.py \
        --root <.../frame_v1> [--s2b <.../s2b_eval_state_v1>]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poolio import Repairs, control_tasks, load_effective  # noqa: E402
from d02_tail_terms import spearman  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--s2b", default="")
    a = ap.parse_args()
    pools = os.path.join(a.root, "pools")
    rep = Repairs(a.root)
    rows = []
    for code, seed, _ in control_tasks(os.path.join(a.root, "manifest.json")):
        p = load_effective(pools, rep, code, seed)
        if p is None:
            continue
        s, r, de = p["score"], p["rank"], p["dE"]
        rows.append(dict(
            identity=float(np.max(np.abs(r * p["Z"] - p["terms"].sum(1)))),
            same=int(np.argmax(s) == np.argmax(r)),
            smr=float(np.max(np.abs(s - r))),
            rs=spearman(s, -de) if len(s) >= 3 else np.nan,
            rr=spearman(r, -de) if len(s) >= 3 else np.nan,
            ts=float(de[np.argmax(s)]), tr=float(de[np.argmax(r)]), fl=float(de.min()), n=len(s)))
    if not rows:
        print("no pools")
        return
    f = lambda k: np.array([x[k] for x in rows], float)
    print("pools %d" % len(rows))
    print("identity max |rank*Z - sum terms| over pools: %.3g" % f("identity").max())
    print("argmax(score)==argmax(rank) in %.3f of pools" % f("same").mean())
    print("max |score-rank| per pool: median %.3f, p95 %.3f" % (np.median(f("smr")), np.percentile(f("smr"), 95)))
    print("top1 dE mean at argmax(score) %.1f ; at argmax(rank) %.1f ; floor mean %.1f" % (
        f("ts").mean(), f("tr").mean(), f("fl").mean()))
    ok = f("n") >= 3
    print("median rho(score,-dE) %.3f ; median rho(rank,-dE) %.3f (pools with >=3 rows: %d)" % (
        np.nanmedian(f("rs")[ok]), np.nanmedian(f("rr")[ok]), ok.sum()))
    if a.s2b:
        files = sorted(glob.glob(os.path.join(a.s2b, "pools", "*_control.npz")))
        dmax, corr = [], []
        for fp in files:
            z = np.load(fp, allow_pickle=True)
            if "error" in z.files or "bend_prior_solver" not in z.files:
                continue
            d = np.asarray(z["score"], float) - np.asarray(z["rank"], float)
            bp = np.asarray(z["bend_prior_solver"], float)
            dmax.append(float(np.max(np.abs(d - bp))))
            if d.std() > 0 and bp.std() > 0:
                corr.append(float(np.corrcoef(d, bp)[0, 1]))
        print("S2b units %d: max |(score-rank) - bend_prior_solver| over units %.3g (median unit %.3g)" % (
            len(dmax), max(dmax), np.median(dmax)))
        print("S2b units: min corr(score-rank, bend_prior_solver) %.5f" % min(corr))


if __name__ == "__main__":
    main()
