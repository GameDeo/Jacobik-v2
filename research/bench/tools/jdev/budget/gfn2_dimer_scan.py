"""Plan 3 N1 (2026-10-17 c): per-element exchange-repulsion parameters for the repulsive Q6, from GFN2-xTB dimers.

The repulsive Q6 sums U_ij(r) = A_ij exp(-b_ij r) (shifted to zero force at the contact distance) over atom pairs, so
its parameters must be PAIR-ADDITIVE: a head-on scan of CH4 against CH4 measures C...C plus the H...C and H...H pairs
around it, which the scorer counts separately. Here the parameters are fitted jointly over all intermolecular atom
pairs of many dimers:

  E_int(config) = E_AB - E_A - E_B  ~  sum over a in A, b in B of [ sqrt(A_a A_b) exp(-(b_a + b_b) / 2 r_ab)
                                                                    - sqrt(C6_a C6_b) / (r_ab^6 + (a_d r_m,ab)^6) ]

(Buckingham combination rule: one A_i, b_i per element; a Becke-Johnson-damped C6 dispersion per element, shared
damping a_d). The dispersion is in the model because near the contact distance GFN2 dimers are net attractive: fitted
without it, the repulsion parameters absorb the dispersion and degenerate (first fit, 10-17: R^2 0.71, B / Br / I / Se
A ~ 1e8). Only the repulsion enters the repulsive Q6; the C6 are kept for the N1b contingency. Monomers carry only
non-polar H (no H-bond donors: those pairs belong to the NCI term), rigid, GFN2-optimised once. Configurations: random
orientations, the two monomers placed along a random direction so that their closest atom pair sits at s x r_m (r_m =
Alvarez vdW sum), s from 0.70 to 1.15. Electrostatics is not modelled (random orientations average it): it is the
fit's noise, reported per monomer pair.

    python tools/jdev/budget/gfn2_dimer_scan.py scan [--orient 4] [--workers 10]
    python tools/jdev/budget/gfn2_dimer_scan.py fit
Output: data/refit/q6rep/{dimers.json, params.json, report.txt}.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from tools.jdev.budget.xtb_stage import run_xtb  # noqa: E402

OUT = REPO / "data" / "refit" / "q6rep"
HARTREE_KJ = 2625.4996
#: Alvarez 2013 (Dalton Trans. 42, 8617) van der Waals radii, A (CSD contact statistics)
ALVAREZ = {"H": 1.20, "B": 1.91, "C": 1.77, "N": 1.66, "O": 1.50, "F": 1.46, "Si": 2.19, "P": 1.90, "S": 1.89,
           "Cl": 1.82, "Se": 1.82, "Br": 1.86, "I": 2.04}
#: probes: only non-polar H; sp3 / sp2 / sp variants of C, N, O
MONOMERS = {"methane": "C", "ethylene": "C=C", "benzene": "c1ccccc1", "trimethylamine": "CN(C)C",
            "pyridine": "c1ccncc1", "acetonitrile": "CC#N", "dimethylether": "COC", "acetone": "CC(C)=O",
            "fluoromethane": "CF", "silane": "[SiH4]", "trimethylphosphine": "CP(C)C", "dimethylsulfide": "CSC",
            "chloromethane": "CCl", "dimethylselenide": "C[Se]C", "bromomethane": "CBr", "iodomethane": "CI",
            "trimethylborane": "CB(C)C"}
SCALES = (0.70, 0.76, 0.82, 0.88, 0.94, 1.00, 1.08, 1.15)
E_MIN, E_MAX = -30.0, 400.0  # kJ/mol: fit window (below the hard overlap)
B_BOUNDS = (2.5, 6.0)  # 1/A: exchange decay lengths of first- to fifth-row atoms


def monomer(name, smi):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=7)
    sy = [a.GetSymbol() for a in m.GetAtoms()]
    X = m.GetConformer().GetPositions()
    E, _, Y = run_xtb(sy, X, 0, 1, "opt")
    Y = Y - Y.mean(0)
    return dict(name=name, sy=sy, X=Y.tolist(), E=E)


def rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def place(A, B, syA, syB, u, s):
    """B translated along u so that min over pairs of r_ab / r_m(ab) equals s (bisection on the shift)."""
    rm = np.array([[ALVAREZ[x] + ALVAREZ[y] for y in syB] for x in syA])
    lo, hi = 0.0, 30.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        D = np.linalg.norm(A[:, None] - (B + t * u)[None], axis=-1)
        if (D / rm).min() < s:
            lo = t
        else:
            hi = t
    return B + hi * u


def scan(a):
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        mons = list(ex.map(lambda kv: monomer(*kv), MONOMERS.items()))
    rng = np.random.default_rng(0)
    jobs = []
    for i, j in itertools.combinations_with_replacement(range(len(mons)), 2):
        for o in range(a.orient):
            RA, RB, u = rot(rng), rot(rng), rng.normal(size=3)
            u /= np.linalg.norm(u)
            for s in SCALES:
                jobs.append((i, j, o, s, RA, RB, u))

    def one(job):
        i, j, o, s, RA, RB, u = job
        A = np.array(mons[i]["X"]) @ RA.T
        B = place(A, np.array(mons[j]["X"]) @ RB.T, mons[i]["sy"], mons[j]["sy"], u, s)
        E = run_xtb(mons[i]["sy"] + mons[j]["sy"], np.vstack([A, B]), 0, 1, "sp")
        if E is None:
            return None
        return dict(a=i, b=j, orient=o, s=s, XA=A.round(5).tolist(), XB=B.round(5).tolist(),
                    E_int=(E - mons[i]["E"] - mons[j]["E"]) * HARTREE_KJ)

    print(f"{len(mons)} monomers, {len(jobs)} dimer single points ...", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(one, jobs) if r is not None]
    (OUT / "dimers.json").write_text(json.dumps(dict(monomers=mons, dimers=res)))
    print(f"wrote {OUT / 'dimers.json'} ({len(res)} dimers)")


def pair_lists(d, els):
    """Per dimer: (element index a, element index b, r) of every intermolecular pair."""
    ei = {e: k for k, e in enumerate(els)}
    mons = d["monomers"]
    out = []
    for r in d["dimers"]:
        A, B = np.array(r["XA"]), np.array(r["XB"])
        D = np.linalg.norm(A[:, None] - B[None], axis=-1)
        ea = np.array([ei[x] for x in mons[r["a"]]["sy"]])
        eb = np.array([ei[x] for x in mons[r["b"]]["sy"]])
        rm = np.array([[ALVAREZ[x] + ALVAREZ[y] for y in mons[r["b"]]["sy"]] for x in mons[r["a"]]["sy"]])
        out.append((np.repeat(ea, len(eb)), np.tile(eb, len(ea)), D.ravel(), rm.ravel()))
    return out


def model(p, pl, n_el, part="all"):
    """Interaction energy (kJ/mol) per dimer; part 'rep' = the repulsion only."""
    lnA, b, lnC6, ad = p[:n_el], p[n_el:2 * n_el], p[2 * n_el:3 * n_el], p[-1]
    out = []
    for i, j, r, rm in pl:
        e = np.exp(0.5 * (lnA[i] + lnA[j]) - 0.5 * (b[i] + b[j]) * r).sum()
        if part == "all":
            e -= (np.exp(0.5 * (lnC6[i] + lnC6[j])) / (r ** 6 + (ad * rm) ** 6)).sum()
        out.append(e)
    return np.array(out)


def fit(a):
    from scipy.optimize import least_squares

    d = json.loads((OUT / "dimers.json").read_text())
    els = sorted({x for m in d["monomers"] for x in m["sy"]})
    n = len(els)
    E = np.array([r["E_int"] for r in d["dimers"]])
    keep = (E >= E_MIN) & (E <= E_MAX)
    pl_all = pair_lists(d, els)
    pl = [pl_all[k] for k in np.flatnonzero(keep)]
    Ek = E[keep]
    p0 = np.r_[np.full(n, math.log(2.0e4)), np.full(n, 3.6), np.full(n, math.log(3000.0)), 0.6]
    lo = np.r_[np.full(n, 0.0), np.full(n, B_BOUNDS[0]), np.full(n, 0.0), 0.3]
    hi = np.r_[np.full(n, 25.0), np.full(n, B_BOUNDS[1]), np.full(n, 15.0), 1.2]

    def res(p):
        # asinh(E / 5 kJ/mol): linear near 0 (flank, dispersion), logarithmic for large E (core)
        return np.arcsinh(model(p, pl, n) / 5.0) - np.arcsinh(Ek / 5.0)

    sol = least_squares(res, p0, bounds=(lo, hi), method="trf", loss="soft_l1", f_scale=0.3)
    pred = model(sol.x, pl_all, n)
    lnA, b, lnC6 = sol.x[:n], sol.x[n:2 * n], sol.x[2 * n:3 * n]
    params = {e: dict(A=round(float(math.exp(lnA[k])), 1), b=round(float(b[k]), 4), R=ALVAREZ[e],
                      C6=round(float(math.exp(lnC6[k])), 1)) for k, e in enumerate(els)}
    params["_damping_a"] = round(float(sol.x[-1]), 4)
    lines = [f"GFN2-xTB dimers: {len(E)} configurations, {keep.sum()} in the fit window [{E_MIN}, {E_MAX}] kJ/mol",
             "per-element parameters (A kJ/mol, b 1/A, Alvarez R A):"]
    lines += [f"  {e:2s} A {v['A']:12.1f}  b {v['b']:.3f}  R {v['R']:.2f}  C6 {v['C6']:9.1f} kJ A^6/mol"
              for e, v in params.items() if not e.startswith("_")]
    lines.append(f"  BJ damping a {params['_damping_a']}")
    from scipy.stats import spearmanr

    rel = (pred[keep] - Ek) / (np.abs(Ek) + 5.0)
    ss = lambda x: ((x - x.mean()) ** 2).sum()
    rp = Ek >= 5.0
    lines.append(f"fit: R^2 {1 - ((pred[keep] - Ek) ** 2).sum() / ss(Ek):.3f}, median |error| / (|E| + 5) "
                 f"{np.median(np.abs(rel)):.2f}; repulsive configurations (E >= 5, n {rp.sum()}): Spearman "
                 f"{spearmanr(pred[keep][rp], Ek[rp])[0]:.3f}, median pred / E {np.median(pred[keep][rp] / Ek[rp]):.2f}")
    rep = model(sol.x, pl_all, n, "rep")
    sc = np.array([r["s"] for r in d["dimers"]])
    lines.append("by contact scale s (closest pair at s x r_m): median E_int / model / repulsion part (kJ/mol)")
    lines += [f"  s {v:.2f}  {np.median(E[sc == v]):7.2f} / {np.median(pred[sc == v]):7.2f} / "
              f"{np.median(rep[sc == v]):6.2f}" for v in sorted(set(sc.tolist()))]
    # combined pairs vs the existing head-on values (q6pmf.GFN2_CORES): A exp(-b r) at 2.4 / 2.7 / 3.0 A
    sys.path.insert(0, str(REPO / "src"))
    from research.archive.code.interactions.q6pmf import GFN2_CORES
    lines.append("pair energies (kJ/mol) at 2.4 / 2.7 / 3.0 A: joint fit vs head-on (q6pmf.GFN2_CORES)")
    for name, (A0, b0) in GFN2_CORES.items():
        x, y = name.replace("Hp", "H").split("_")
        if x not in params or y not in params:
            continue
        A1 = math.sqrt(params[x]["A"] * params[y]["A"])
        b1 = 0.5 * (params[x]["b"] + params[y]["b"])
        f = lambda A_, b_: " / ".join(f"{A_ * math.exp(-b_ * r):6.2f}" for r in (2.4, 2.7, 3.0))
        lines.append(f"  {name:5s} joint {f(A1, b1)}   head-on {f(A0, b0)}")
    # residual by monomer pair (median relative error, fit window)
    names = [m["name"] for m in d["monomers"]]
    by = {}
    for k, r in zip(np.flatnonzero(keep), rel):
        dd = d["dimers"][k]
        by.setdefault(f"{names[dd['a']]}-{names[dd['b']]}", []).append(r)
    worst = sorted(by.items(), key=lambda kv: -abs(np.median(kv[1])))[:12]
    lines.append("largest median relative residuals by monomer pair (electrostatics / dispersion not in the model):")
    lines += [f"  {k:36s} {np.median(v):+.2f} (n {len(v)})" for k, v in worst]
    txt = "\n".join(lines)
    (OUT / "params.json").write_text(json.dumps(params, indent=1))
    (OUT / "report.txt").write_text(txt)
    print(txt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["scan", "fit"])
    p.add_argument("--orient", type=int, default=12)
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    {"scan": scan, "fit": fit}[a.stage](a)


if __name__ == "__main__":
    main()
