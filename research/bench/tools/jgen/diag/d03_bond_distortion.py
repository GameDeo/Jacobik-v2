"""Does ligand covalent distortion (relative to the reference) explain pool energies? CPU only.

For each pool row, in reference atom order (X[:, gen_to_ref_idx], the same mapping the orchestrator uses
in research/archive/code/refit/nci_v2_orchestrator.py ~L716 when it calls coordination_integrity):
  bond_rms  = RMS over covalent non-metal bonds of |d_gen - d_ref|           (A)
  bond_max  = max over the same bonds                                         (A)
  donor_rms = RMS over metal-to-nonmetal contacts with d_ref <= 1.15*(rM+rX)  (A)
Bonds: non-metal pairs with d_ref < 1.25*(r_i + r_j) + 0.25 A (covalent radii below).
Sanity: the metal-donor RMS for the reference-mapped floor row should be small for intact poses.

Outputs per pool: Spearman(bond_rms, dE) and Spearman(score, dE) over rows, top1/floor bond_rms,
and the number of rows with bond_max > 0.2 A. dE = energies_kj - ref_energy_kj (mace64.json).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d03_bond_distortion.py \
        --pools <.../pools> --codes DAMVAQ,...   [--all]   # --all summarises every pool
"""
import argparse
import glob
import json
import os

import numpy as np

COV = {"H": .31, "B": .84, "C": .76, "N": .71, "O": .66, "F": .57, "Si": 1.11, "P": 1.07, "S": 1.05,
       "Cl": 1.02, "Se": 1.20, "Br": 1.20, "I": 1.39, "As": 1.19}
NONMETAL = set(COV)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pair_sets(sym, Xr):
    nm = [i for i, s in enumerate(sym) if s in NONMETAL]
    mt = [i for i, s in enumerate(sym) if s not in NONMETAL]
    bonds = []
    for ii, i in enumerate(nm):
        for j in nm[ii + 1:]:
            d = np.linalg.norm(Xr[i] - Xr[j])
            if d < 1.25 * (COV[sym[i]] + COV[sym[j]]) + 0.25:
                bonds.append((i, j))
    donors = []
    for m in mt:
        for i in nm:
            d = np.linalg.norm(Xr[i] - Xr[m])
            rm = 1.4 if sym[m] not in COV else COV[sym[m]]
            if d <= 1.15 * (rm + COV[sym[i]]) + 1e-9 and sym[i] != "H":
                donors.append((m, i))
    return bonds, donors


def row_metrics(X_ref_order, Xref, sym):
    bonds, donors = pair_sets(sym, Xref)
    bi = np.array(bonds, int)
    di = np.array(donors, int) if donors else np.zeros((0, 2), int)
    out = []
    for X in X_ref_order:
        db = np.linalg.norm(X[bi[:, 0]] - X[bi[:, 1]], axis=1) - np.linalg.norm(Xref[bi[:, 0]] - Xref[bi[:, 1]], axis=1)
        dd = (np.linalg.norm(X[di[:, 0]] - X[di[:, 1]], axis=1)
              - np.linalg.norm(Xref[di[:, 0]] - Xref[di[:, 1]], axis=1)) if len(di) else np.zeros(1)
        out.append((float(np.sqrt(np.mean(db ** 2))), float(np.max(np.abs(db))),
                    float(np.sqrt(np.mean(dd ** 2)))))
    return np.array(out)


def analyse(pool_dir, code, seed):
    base = os.path.join(pool_dir, "%s_%d_control" % (code, seed))
    z = np.load(base + ".npz", allow_pickle=True)
    if "error" in z.files:
        return None
    m = json.load(open(base + ".mace64.json"))
    sym = [str(s) for s in z["ref_symbols"]]
    Xref = np.asarray(z["ref_X"], float)
    X = np.asarray(z["X"], float)
    g2r = np.asarray(z["gen_to_ref_idx"], int)
    Xr = X[:, g2r, :]
    met = row_metrics(Xr, Xref, sym)
    de = np.asarray(m["energies_kj"], float) - float(m["ref_energy_kj"])
    score = np.asarray(z["score"], float)
    top = int(np.argmax(score))
    flo = int(np.argmin(de))
    return dict(code=code, seed=seed, n=len(de),
                rho_bond_dE=spearman(met[:, 0], de), rho_bondmax_dE=spearman(met[:, 1], de),
                rho_score_dE=spearman(score, -de),
                top_bond_rms=round(met[top, 0], 3), floor_bond_rms=round(met[flo, 0], 3),
                top_donor_rms=round(met[top, 2], 3), floor_donor_rms=round(met[flo, 2], 3),
                top_dE=round(de[top], 1), floor_dE=round(de[flo], 1),
                rows_bondmax_gt_0p2=int((met[:, 1] > 0.2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--codes", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        files = sorted(glob.glob(os.path.join(a.pools, "*_control.npz")))
        pairs = [(os.path.basename(f).split("_")[0], int(os.path.basename(f).split("_")[1])) for f in files]
    else:
        pairs = [(c, s) for c in a.codes.split(",") for s in (11, 12, 13)]
    res = []
    for code, seed in pairs:
        try:
            r = analyse(a.pools, code, seed)
        except FileNotFoundError:
            r = None
        if r is None:
            continue
        res.append(r)
        if not a.all:
            print(json.dumps(r))
    if a.all and res:
        rb = np.array([r["rho_bond_dE"] for r in res], float)
        rs = np.array([r["rho_score_dE"] for r in res], float)
        print("pools %d; median rho(bond_rms,dE)=%.3f; median rho(score,-dE)=%.3f" % (
            len(res), np.nanmedian(rb), np.nanmedian(rs)))
        print("pools where top1 bond_rms > floor bond_rms: %d" % sum(1 for r in res if r["top_bond_rms"] > r["floor_bond_rms"]))


if __name__ == "__main__":
    main()
