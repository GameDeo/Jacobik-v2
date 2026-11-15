"""IRAQUO hydrogen mapping, every row of every seed (CPU only, stored data).

The orchestrator maps the pool once from conformer 0 (research/archive/code/refit/nci_v2_orchestrator.py:709,
build_atom_mapping(..., X_gen=X[0])) and applies gen_to_ref_idx to every row (mapped_confs = X[:, gen_to_ref]).
For each carbon bonded to hydrogens in the reference (distance < 1.25 A), and for each such reference hydrogen h:
  mapped_dist  = |Y[h] - Y[c]|, Y = X[row][gen_to_ref]   (reference order; what the graph check sees)
  raw_nearest  = the generated hydrogen nearest to the generated carbon, raw frame, and its distance
  bad          = mapped_dist > 1.25 A
Also prints: mapping_ambiguous, num_automorphisms, and whether gen_to_ref_idx is identical across seeds.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d11_iraquo_table.py \
        --pools <.../pools>
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--code", default="IRAQUO")
    a = ap.parse_args()
    maps = {}
    clean_pairs = {}
    for seed in (11, 12, 13):
        z = np.load(os.path.join(a.pools, "%s_%d_control.npz" % (a.code, seed)), allow_pickle=True)
        X = np.asarray(z["X"], float)
        g = np.asarray(z["gen_to_ref_idx"], int)
        R = np.asarray(z["ref_X"], float)
        sym = [str(s) for s in z["ref_symbols"]]
        maps[seed] = g
        print("seed %d rows %d atoms %d mapping_ambiguous=%s num_automorphisms=%d" % (
            seed, X.shape[0], X.shape[1], bool(z["mapping_ambiguous"]), int(z["num_automorphisms"])))
        hsall = [j for j in range(len(sym)) if sym[j] == "H"]
        for c in [j for j in range(len(sym)) if sym[j] == "C" and any(
                np.linalg.norm(R[h] - R[j]) < 1.25 for h in hsall)]:
            for h in [h for h in hsall if np.linalg.norm(R[h] - R[c]) < 1.25]:
                bad_rows, mdist, raw = 0, [], []
                for r in range(X.shape[0]):
                    Y = X[r][g]
                    dm = float(np.linalg.norm(Y[h] - Y[c]))
                    mdist.append(round(dm, 2))
                    bad_rows += dm > 1.25
                    Xg = X[r]
                    gc = int(g[c])
                    gen_h = [int(g[j]) for j in hsall]
                    k = min(gen_h, key=lambda q: np.linalg.norm(Xg[q] - Xg[gc]))
                    raw.append(round(float(np.linalg.norm(Xg[k] - Xg[gc])), 2))
                if bad_rows == 0:
                    clean_pairs[seed] = clean_pairs.get(seed, 0) + 1
                    continue
                print("  BAD carbon ref %s%d  ref H %s%d : bad rows %d/%d ; mapped dist per row %s ; "
                      "raw nearest-H dist per row %s" % (sym[c], c, sym[h], h, bad_rows, X.shape[0],
                                                       mdist, raw))
        print("  clean carbon-H pairs in this seed: %d" % clean_pairs.get(seed, 0))
    same = all(np.array_equal(maps[11], maps[s]) for s in (12, 13))
    print("gen_to_ref_idx identical across seeds: %s" % same)


if __name__ == "__main__":
    main()
