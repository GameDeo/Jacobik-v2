"""Assignment-class structure and electronic-energy ensemble statistics of the control pools (CPU only).

Per successful control pool (calibration and motion_validation splits):
  n_rows            stored rows (post window and dedup)
  n_asg             distinct assignment indices present (asg from the NPZ)
  floor_asg_share   1 if the floor row's assignment equals the native top-1's assignment, else 0
  classes_within_20 number of assignment classes whose best row lies within 20 kJ/mol of the pool floor
  neff_T            effective number of rows, exp-entropy of Boltzmann weights over MACE electronic energies,
                    kT = R*T = 2.479 kJ/mol at 298.15 K; p_i ~ exp(-dE_i / kT); N_eff = exp(H), H = -sum p ln p.
                    This is an ELECTRONIC-energy ensemble over stored rows only: no vibrational, entropic or
                    solvent terms, no degeneracy beyond the stored rows, no coverage of unsampled basins.
Also: the share of rows with a class different from the top-1's, pooled over all pools.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d08_classes_and_ensemble.py \
        --root <.../frame_v1>   (effective pools, poolio.py)
"""
import argparse
import json
import os

import numpy as np

KT = 0.00831446261815324 * 298.15  # kJ/mol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poolio import Repairs, control_tasks, load_effective
    pools = os.path.join(a.root, "pools")
    rep = Repairs(a.root)
    rows_out = []
    for code, seed, rec in control_tasks(os.path.join(a.root, "manifest.json")):
        p = load_effective(pools, rep, code, seed)
        if p is None:
            continue
        de = p["dE"]
        asg = p["asg"]
        score = p["score"]
        top, flo = int(np.argmax(score)), int(np.argmin(de))
        classes = np.unique(asg)
        best = {c: float(de[asg == c].min()) for c in classes}
        w = np.exp(-(de - de.min()) / KT)
        p = w / w.sum()
        H = -np.sum(p[p > 0] * np.log(p[p > 0]))
        rows_out.append(dict(code=code, seed=seed, split=rec["split"], n_rows=len(de),
                             n_asg=len(classes), floor_asg_share=int(asg[flo] == asg[top]),
                             classes_within_20=int(sum(1 for c in classes if best[c] <= de.min() + 20.0)),
                             neff=float(np.exp(H)), frac_rows_other_class_than_top1=float(np.mean(asg != asg[top]))))
    n = len(rows_out)
    if not n:
        print("no pools")
        return
    pools_multi = [r for r in rows_out if r["n_asg"] > 1]
    print("pools %d; pools with >1 assignment class %d (%.3f)" % (n, len(pools_multi), len(pools_multi) / n))
    print("median rows %.1f; median classes %.1f (max %d)" % (
        np.median([r["n_rows"] for r in rows_out]), np.median([r["n_asg"] for r in rows_out]),
        max(r["n_asg"] for r in rows_out)))
    print("floor row in same class as native top-1: %.3f (over multi-class pools %.3f)" % (
        np.mean([r["floor_asg_share"] for r in rows_out]),
        np.mean([r["floor_asg_share"] for r in pools_multi]) if pools_multi else float("nan")))
    print("median classes within 20 kJ/mol of floor (multi-class pools): %.2f" % (
        np.median([r["classes_within_20"] for r in pools_multi]) if pools_multi else float("nan")))
    print("median N_eff over pools (electronic, stored rows): %.2f ; over pools with >=3 rows: %.2f" % (
        np.median([r["neff"] for r in rows_out]),
        np.median([r["neff"] for r in rows_out if r["n_rows"] >= 3])))
    print("share of all rows whose class differs from its pool's top-1: %.3f" % (
        np.mean([r["frac_rows_other_class_than_top1"] for r in rows_out])))


if __name__ == "__main__":
    main()
