"""Upper bound for a coordination-integrity filter on ranking (CPU only, evaluation-only label).

Uses the REFERENCE-BASED stored label coord_intact (jacobik.benchmark.coordination.coordination_integrity). This is an
evaluation label and is NOT available in production; a production filter must use a reference-free site test
(see research/archive/docs/roadmap/roadmap_generation_v1.md, section 1.4). The script measures how much top-1 energy would change if ranking
were restricted to intact rows (argmax of score among intact rows), using the stored MACE energies:

  top1_dE          = dE of argmax(score) over all rows                      (the incumbent)
  top1_intact_dE   = dE of argmax(score) over intact rows, if any intact row exists
  pools with top1 non-intact and at least one intact row; their mean change

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d09_intact_filter_oracle.py \
        --root <.../frame_v1>   (effective pools: raw + S2.1 overlays, poolio.py)
"""
import argparse
import json
import os

import numpy as np


def main():
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poolio import Repairs, control_tasks, load_effective
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    pools = os.path.join(a.root, "pools")
    rep = Repairs(a.root)
    base_t, filt_t, n_nonintact, n_fixable, changes, n_skip = [], [], 0, 0, [], 0
    for code, seed, _ in control_tasks(os.path.join(a.root, "manifest.json")):
        p = load_effective(pools, rep, code, seed)
        if p is None or p["coord_intact"] is None:
            n_skip += 1
            continue
        de, score, ci = p["dE"], p["score"], p["coord_intact"]
        top = int(np.argmax(score))
        base_t.append(de[top])
        if not ci[top]:
            n_nonintact += 1
            if ci.any():
                n_fixable += 1
                idx = np.where(ci)[0]
                j = idx[np.argmax(score[idx])]
                changes.append(de[j] - de[top])
                filt_t.append(de[j])
                continue
        filt_t.append(de[top])
    base_t, filt_t = np.array(base_t), np.array(filt_t)
    print(json.dumps(dict(pools=len(base_t), skipped_no_coord=n_skip, top1_mean=round(float(base_t.mean()), 1),
                          filtered_top1_mean=round(float(filt_t.mean()), 1),
                          top1_nonintact=n_nonintact, nonintact_with_intact_row=n_fixable,
                          mean_change_on_fixable=round(float(np.mean(changes)), 1) if changes else None,
                          median_change_on_fixable=round(float(np.median(changes)), 1) if changes else None)))


if __name__ == "__main__":
    main()
