"""Angle distortion, intra-ligand contacts and the largest-bond outlier, relative to the reference (CPU only).

Uses the same reference-order mapping as d03 (X[:, gen_to_ref_idx]) and the same covalent bond rule.
Per pool row:
  angle_rms   = RMS over bonded triples (i-j-k, both ends non-metal) of |theta_gen - theta_ref| (deg)
  clash_min   = minimum distance (A) over non-metal pairs that are not 1-2 or 1-3 neighbours
  n_clash     = count of such pairs with distance < 0.8 * (r_i + r_j) (covalent-radius sum)
Also prints, for a named pool, the bond with the largest reference-relative deviation (NEPNEF check).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d04_angles_clashes.py \
        --pools <.../pools> --codes DAMVAQ,...,ZELTAP --bond-report NEPNEF
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d03_bond_distortion import COV, NONMETAL, spearman  # noqa: E402  (same directory)


def graph(sym, Xr):
    nm = [i for i, s in enumerate(sym) if s in NONMETAL]
    adj = {i: set() for i in nm}
    for ii, i in enumerate(nm):
        for j in nm[ii + 1:]:
            if np.linalg.norm(Xr[i] - Xr[j]) < 1.25 * (COV[sym[i]] + COV[sym[j]]) + 0.25:
                adj[i].add(j)
                adj[j].add(i)
    return nm, adj


def analyse(pools, code, seed, bond_report=False):
    base = os.path.join(pools, "%s_%d_control" % (code, seed))
    z = np.load(base + ".npz", allow_pickle=True)
    if "error" in z.files:
        return None
    m = json.load(open(base + ".mace64.json"))
    sym = [str(s) for s in z["ref_symbols"]]
    Xref = np.asarray(z["ref_X"], float)
    X = np.asarray(z["X"], float)[:, np.asarray(z["gen_to_ref_idx"], int), :]
    nm, adj = graph(sym, Xref)
    triples = []
    for j in nm:
        nb = sorted(adj[j])
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                triples.append((nb[a], j, nb[b]))
    # 1-2 and 1-3 exclusions
    excl = set()
    for i in nm:
        for j in adj[i]:
            excl.add((min(i, j), max(i, j)))
            for k in adj[j]:
                if k != i:
                    excl.add((min(i, k), max(i, k)))
    pairs = [(i, j) for ai, i in enumerate(nm) for j in nm[ai + 1:] if (i, j) not in excl]
    pa = np.array(pairs, int) if pairs else np.zeros((0, 2), int)
    rsum = np.array([COV[sym[i]] + COV[sym[j]] for i, j in pairs]) if pairs else np.zeros(0)
    ta = np.array(triples, int) if triples else np.zeros((0, 3), int)

    def ang(Y):
        u = Y[ta[:, 0]] - Y[ta[:, 1]]
        v = Y[ta[:, 2]] - Y[ta[:, 1]]
        c = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    a_ref = ang(Xref)
    rows = []
    for Y in X:
        a_dev = ang(Y) - a_ref if len(ta) else np.zeros(1)
        d = np.linalg.norm(Y[pa[:, 0]] - Y[pa[:, 1]], axis=1) if len(pa) else np.zeros(1)
        rows.append((float(np.sqrt(np.mean(a_dev ** 2))), float(d.min()) if len(pa) else float("nan"),
                     int((d < 0.8 * rsum).sum()) if len(pa) else 0))
    rows = np.array(rows)
    de = np.asarray(m["energies_kj"], float) - float(m["ref_energy_kj"])
    score = np.asarray(z["score"], float)
    top, flo = int(np.argmax(score)), int(np.argmin(de))
    out = dict(code=code, seed=seed, n=len(de),
               rho_angle_dE=round(spearman(rows[:, 0], de), 3),
               rho_clashmin_dE=round(spearman(-rows[:, 1], de), 3),
               top_angle_rms=round(rows[top, 0], 2), floor_angle_rms=round(rows[flo, 0], 2),
               top_clash_min=round(rows[top, 1], 2), floor_clash_min=round(rows[flo, 1], 2),
               top_n_clash=int(rows[top, 2]), floor_n_clash=int(rows[flo, 2]),
               top_dE=round(de[top], 1), floor_dE=round(de[flo], 1))
    if bond_report:
        bonds = [(i, j) for i in nm for j in adj[i] if i < j]
        dev = []
        for i, j in bonds:
            dr = np.linalg.norm(Xref[i] - Xref[j])
            dg = np.linalg.norm(X[0][i] - X[0][j])
            dev.append((abs(dg - dr), sym[i], sym[j], round(dr, 3), round(dg, 3), i, j))
        dev.sort(reverse=True)
        out["largest_bond_dev_row0"] = [[round(x[0], 3), x[1], x[2], x[3], x[4], x[5], x[6]] for x in dev[:4]]
        out["n_bonds"] = len(bonds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--bond-report", default="")
    a = ap.parse_args()
    for code in a.codes.split(","):
        for seed in (11, 12, 13):
            r = analyse(a.pools, code, seed, bond_report=(code == a.bond_report and seed == 11))
            if r is not None:
                print(json.dumps(r))


if __name__ == "__main__":
    main()
