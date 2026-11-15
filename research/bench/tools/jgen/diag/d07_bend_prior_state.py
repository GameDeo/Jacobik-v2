"""Is the tail top-1 penalised by the solver's own bend prior? Uses the S2b per-pose state (CPU only).

Reads data/refit/frame_v1/s2b_eval_state_v1/pools/<CODE>_<seed>_control.npz (state_schema_version,
thetas_flat, bend_columns, bend_k, bend_prior_solver). For the native top-1 (argmax score from the frame_v1
pool) and the energy floor row, prints:
  bend_prior_solver  : stored solver bend-prior value (nats, as used in ranking)
  bend_cols_abs_max  : largest |theta| over the bend columns (rad), i.e. how far bend DOFs have moved
Compare with the angle RMS from d04. Rows are matched by position: the S2b pool stores the same rows as the
frame_v1 pool (the S2b report states the pool identity); the script checks the row count and the score.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d07_bend_prior_state.py \
        --root /mnt/e/Dev/Projects/jacobik_2026-10-02_full/data/refit/frame_v1 --codes SOBHAS,ZELTAP,HOXJUZ
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--codes", required=True)
    a = ap.parse_args()
    s2b = os.path.join(a.root, "s2b_eval_state_v1", "pools")
    for code in a.codes.split(","):
        for seed in (11, 12, 13):
            p = os.path.join(s2b, "%s_%d_control.npz" % (code, seed))
            m_path = os.path.join(s2b, "%s_%d_control.mace64.json" % (code, seed))
            if not (os.path.exists(p) and os.path.exists(m_path)):
                print(code, seed, "no S2b pool")
                continue
            z = np.load(p, allow_pickle=True)
            m = json.load(open(m_path))
            de = np.asarray(m["energies_kj"], float) - float(m["ref_energy_kj"])
            score = np.asarray(z["score"], float)
            top, flo = int(np.argmax(score)), int(np.argmin(de))
            th = np.asarray(z["thetas_flat"], float)
            bc = np.asarray(z["bend_columns"], int)
            bp = np.asarray(z["bend_prior_solver"], float)
            # bend_columns indexes the flattened theta vector; the bend DOFs are the columns listed there
            cols = bc[:, 0] if bc.ndim == 2 else bc
            cols = cols[cols < th.shape[1]]
            def bmax(i):
                return float(np.max(np.abs(th[i, cols]))) if len(cols) else 0.0
            print(json.dumps(dict(code=code, seed=seed, n_rows=int(th.shape[0]), n_bend_cols=int(len(cols)),
                                  top_dE=round(float(de[top]), 1), floor_dE=round(float(de[flo]), 1),
                                  top_bend_prior=round(float(bp[top]), 4), floor_bend_prior=round(float(bp[flo]), 4),
                                  top_bend_abs_max_rad=round(bmax(top), 3), floor_bend_abs_max_rad=round(bmax(flo), 3),
                                  rho_bendprior_dE=round(float(np.corrcoef(
                                      np.argsort(np.argsort(bp)), np.argsort(np.argsort(de)))[0, 1]), 3)
                                  if th.shape[0] > 2 and bp.std() > 0 and de.std() > 0 else None)))


if __name__ == "__main__":
    main()
