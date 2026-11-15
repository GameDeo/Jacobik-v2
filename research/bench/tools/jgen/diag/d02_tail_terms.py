"""Per-pool top-1 versus floor attribution, on the effective pools (CPU only).

For each pool of the given parents (effective = raw + S2.1 overlay, see poolio.py):
  native top-1 row  = argmax of the stored `score` (the array the orchestrator uses for top1_idx,
                      nci_v2_orchestrator.py:746); `rank_top` = argmax of the stored `rank`
  floor row         = argmin of dE
  dE, asg, coord_intact for the top-1 and the floor
  Q-terms (already times Z) of top-1 and floor, and Z
  rho_score = Spearman(score, -dE),  rho_rank = Spearman(rank, -dE) over rows
  identity: rank * Z - sum(terms) max abs (should be ~0), reported once per pool

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d02_tail_terms.py \
        --pools <.../pools> --root <.../frame_v1> --codes DAMVAQ,...,ZELTAP
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poolio import Repairs, load_effective  # noqa: E402


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--codes", required=True)
    a = ap.parse_args()
    rep = Repairs(a.root)
    for code in a.codes.split(","):
        for seed in (11, 12, 13):
            p = load_effective(a.pools, rep, code, seed)
            if p is None:
                print(code, seed, "error pool")
                continue
            s, r, de = p["score"], p["rank"], p["dE"]
            top, flo, rtop = int(np.argmax(s)), int(np.argmin(de)), int(np.argmax(r))
            ci = p["coord_intact"]
            ci = np.zeros(len(s), bool) if ci is None else ci
            ident = float(np.max(np.abs(r * p["Z"] - p["terms"].sum(1))))
            print("== %s s%d n=%d src=%s Z=%.0f | top1 %d dE=%.1f asg=%d intact=%s | rank-top %d dE=%.1f | floor %d dE=%.1f asg=%d intact=%s" % (
                code, seed, len(s), p["source"], p["Z"], top, de[top], p["asg"][top], bool(ci[top]),
                rtop, de[rtop], flo, de[flo], p["asg"][flo], bool(ci[flo])))
            print("   rho(score,-dE)=%.3f rho(rank,-dE)=%.3f ; max|rank*Z - sum(terms)|=%.2g" % (
                spearman(s, -de), spearman(r, -de), ident))
            print("   top1 Q1..Q6:", np.round(p["terms"][top], 2).tolist())
            print("   floor Q1..Q6:", np.round(p["terms"][flo], 2).tolist())


if __name__ == "__main__":
    main()
