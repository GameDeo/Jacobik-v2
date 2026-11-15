"""Per-stage wall time against size and symmetry, from stored pool summaries (CPU only).

For every successful control pool: the stage_* fields of the pool JSON, the number of stored rows (n_confs,
the post-dedup output), the heavy-atom count of the reference, and num_automorphisms from the NPZ.
Reports, per split: mean and p95 of each stage, the share of wall time in each stage, and the Spearman
correlation of stage_dedup_s with num_automorphisms and with n_confs (dedup cost is expected to grow with both;
see src/jacobik/ik/dedup.py select() and the BLOCK_MAX_MAPS branch).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d06_stage_timing.py \
        --pools <.../pools> --manifest <.../manifest.json>
"""
import argparse
import json
import os

import numpy as np

STAGES = ["stage_embed_s", "stage_seed_rows_s", "stage_solver_s", "stage_dedup_s", "stage_mace_s", "stage_other_s"]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    m = json.load(open(a.manifest))
    recs = {"calibration": [], "motion_validation": [], "surface_fit": []}
    for key, rec in m["tasks"].items():
        if rec.get("status") != "success" or rec.get("split") not in recs:
            continue
        base = os.path.join(a.pools, "%s_%d_control" % (rec["code"], rec["seed"]))
        try:
            s = json.load(open(base + ".json"))
            z = np.load(base + ".npz", allow_pickle=True)
        except (FileNotFoundError, OSError):
            continue
        if "error" in z.files:
            continue
        na = int(z["num_automorphisms"]) if "num_automorphisms" in z.files else -1
        nh = int(sum(1 for x in z["ref_symbols"] if str(x) != "H"))
        row = {k: float(s.get(k) or 0.0) for k in STAGES + ["wall_s"]}
        row.update(code=rec["code"], seed=rec["seed"], n_confs=int(s.get("n_confs") or 0), n_auto=na, n_heavy=nh)
        recs[rec["split"]].append(row)

    for split, rows in recs.items():
        if not rows:
            continue
        W = np.array([r["wall_s"] for r in rows])
        print("== %s: pools %d" % (split, len(rows)))
        for k in STAGES + ["wall_s"]:
            v = np.array([r[k] for r in rows])
            share = "" if k == "wall_s" else " share %.3f" % (v.sum() / max(W.sum(), 1e-9))
            print("  %-18s mean %6.2f p95 %6.2f%s" % (k, v.mean(), np.percentile(v, 95), share))
        dd = [r["stage_dedup_s"] for r in rows]
        print("  rho(stage_dedup_s, n_auto) = %.3f ; rho(stage_dedup_s, n_confs) = %.3f" % (
            spearman(dd, [r["n_auto"] for r in rows]), spearman(dd, [r["n_confs"] for r in rows])))
        big = sorted(rows, key=lambda r: -r["stage_dedup_s"])[:5]
        print("  top dedup pools:", [(r["code"], r["seed"], round(r["stage_dedup_s"], 1), r["n_confs"], r["n_auto"])
                                     for r in big])


if __name__ == "__main__":
    main()
