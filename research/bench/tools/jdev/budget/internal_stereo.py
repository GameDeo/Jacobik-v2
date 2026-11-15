"""Ligand-internal configuration of the solver's structures vs the crystal (2026-10-16, owner question: is "isomer
choice" also cis/trans or ring stereo INSIDE the ligands?).

The solver never changes bond lengths / angles / ring torsions of the free-ligand embed (only torsions about rotatable
bonds, rigid-body moves, a few bends). So the embed fixes, for good:
  * the relative configuration of every sp3 stereocentre in a ligand (SMILES from tmQM carry no stereo tags; ETKDG
    picks one at random, and binding can create new ones, e.g. a coordinated N with three different substituents),
  * E/Z of double bonds that are not rotatable joints,
  * ring puckers (non-aromatic rings: chair / boat / envelope, macrocycle folds from the macro pool).
If those differ from the crystal, the crystal's structure is unreachable and its energy is lost before the solve.

Per pool (seed 11 by default) it compares the #1 (and the floor) with the crystal mapped onto the solver's atom order
(oracle_targets.crystal_in_engine_order): stereocentre signs (relative: a whole-molecule mirror image is the same
energy, so the minority of mismatches counts), cis/trans class of double bonds, and per ring the largest ring-torsion
difference and whether the torsion sign pattern differs.

    python tools/jdev/budget/internal_stereo.py [--sets dev32 n48] [--arms etkdg full_oracle] [--detail FIGHOT TAPWUG]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pools import OUT, check_db, load_stats, set_cases, setup  # noqa: E402

_RCOV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, Se=1.20, Br=1.20,
             Te=1.38, I=1.39, As=1.19, Ge=1.20, Sn=1.39, Al=1.21, Ga=1.22, Sb=1.39, Bi=1.48)


def _dih(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def analyse(sy, X, Xc, metal):
    """dict of internal-configuration comparisons between X (solver) and Xc (crystal, same atom order)."""
    import networkx as nx
    n = len(sy)
    r = np.array([_RCOV.get(s, 1.5) for s in sy])
    D = np.sqrt(((Xc[:, None] - Xc[None]) ** 2).sum(-1))
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n) if i != metal])
    for i in range(n):
        for j in range(i + 1, n):
            if metal not in (i, j) and D[i, j] < 1.2 * (r[i] + r[j]):
                G.add_edge(i, j)
    heavy = lambda i: sy[i] != "H"  # noqa: E731
    out = {}
    # stereocentres: heavy atoms with >= 3 heavy ligand neighbours, pyramidal in the crystal
    signs = []
    for i in G.nodes:
        if not heavy(i):
            continue
        nb = sorted(j for j in G.neighbors(i) if heavy(j))
        if len(nb) < 3:
            continue
        a, b, c = (Xc[nb[k]] - Xc[i] for k in range(3))
        vol_c = np.dot(a / np.linalg.norm(a), np.cross(b / np.linalg.norm(b), c / np.linalg.norm(c)))
        if abs(vol_c) < 0.25 or len(set(sy[j] for j in nb)) == 1 and len(nb) == 4 and False:
            continue
        a2, b2, c2 = (X[nb[k]] - X[i] for k in range(3))
        vol = np.dot(a2 / np.linalg.norm(a2), np.cross(b2 / np.linalg.norm(b2), c2 / np.linalg.norm(c2)))
        signs.append(np.sign(vol) == np.sign(vol_c))
    signs = np.array(signs, bool)
    mism = int(min((~signs).sum(), signs.sum())) if len(signs) else 0
    out.update(n_stereo=len(signs), stereo_mismatch=mism)
    # double bonds outside rings between atoms with >= 2 heavy neighbours each
    cyc_edges = {frozenset(e) for cyc in nx.cycle_basis(G) for e in zip(cyc, cyc[1:] + cyc[:1])}
    nez, ez_bad = 0, 0
    for i, j in G.edges:
        if not (heavy(i) and heavy(j)) or frozenset((i, j)) in cyc_edges:
            continue
        if D[i, j] > 1.40 or sy[i] not in "CN" or sy[j] not in "CN":
            continue
        ni = [k for k in G.neighbors(i) if k != j and heavy(k)]
        nj = [k for k in G.neighbors(j) if k != i and heavy(k)]
        if not ni or not nj:
            continue
        a, d = min(ni), min(nj)
        pc, px = _dih(Xc[a], Xc[i], Xc[j], Xc[d]), _dih(X[a], X[i], X[j], X[d])
        if abs(abs(pc) - 90) < 30:  # twisted in the crystal: no clean E/Z
            continue
        nez += 1
        ez_bad += int((abs(pc) < 90) != (abs(px) < 90))
    out.update(n_ez=nez, ez_mismatch=ez_bad)
    # non-aromatic rings: largest ring-torsion difference, sign-pattern flip
    worst, flipped, nr = 0.0, 0, 0
    for cyc in nx.cycle_basis(G):
        if len(cyc) < 5 or not all(heavy(i) for i in cyc):
            continue
        tc = [_dih(*(Xc[cyc[(k + m) % len(cyc)]] for m in range(4))) for k in range(len(cyc))]
        if max(abs(t) for t in tc) < 25:  # planar (aromatic-like) ring
            continue
        tx = [_dih(*(X[cyc[(k + m) % len(cyc)]] for m in range(4))) for k in range(len(cyc))]
        dif = [abs((a - b + 180) % 360 - 180) for a, b in zip(tc, tx)]
        mirror = [abs((-a - b + 180) % 360 - 180) for a, b in zip(tc, tx)]
        nr += 1
        worst = max(worst, min(max(dif), max(mirror)))
        big = [k for k in range(len(tc)) if abs(tc[k]) > 25]
        flipped += int(any(np.sign(tc[k]) != np.sign(tx[k]) for k in big) and
                       any(np.sign(-tc[k]) != np.sign(tx[k]) for k in big))
    out.update(n_rings=nr, ring_tors_maxdiff=worst, ring_pucker_diff=flipped)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["dev32", "n48"])
    p.add_argument("--arms", nargs="+", default=["etkdg", "full_oracle"])
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    setup()
    import torch  # noqa: F401
    from oracle_targets import crystal_in_engine_order
    from tools.jdev.cases import get_case
    check_db()
    st = load_stats()
    rows = []
    for s in a.sets:
        names = dict(set_cases(s))
        for code, name in names.items():
            try:
                case = get_case(name)
                r = case.make_builder().solve(seed=a.seed, stats=st, angle_bends=True, n_conformers=1, n_ik_steps=2,
                                              device="cpu")
                mol = r.conformers[0].complex
                metal = int(r.scorer._ge.metal_pos_idx)
            except Exception as exc:
                print(f"{code}: skipped ({type(exc).__name__})", flush=True)
                continue
            sy = [at.symbol for at in mol.atoms]
            for arm in a.arms:
                f = OUT / "pools" / arm / s / f"{code}_s{a.seed}.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=False)
                if "error" in z.files:
                    continue
                e = np.array(json.loads(f.with_suffix(".mace.json").read_text())["dE"])
                for tag, i in (("pick", int(np.argmax(z["score"]))), ("floor", int(np.argmin(e)))):
                    X = z["X"][i]
                    Xc, ok = crystal_in_engine_order(case, mol, X)
                    res = analyse(sy, X, Xc, metal)
                    rows.append(dict(set=s, case=code, arm=arm, which=tag, dE=e[i], **res))
            print(f"{code}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "internal_stereo.csv", index=False)
    pd.set_option("display.width", 220)
    P = []
    pk = df[df.which == "pick"]
    for arm, g in pk.groupby("arm"):
        P.append(f"\n#### {arm} #1 picks: {len(g)}")
        P.append(f"with a stereocentre mismatch: {(g.stereo_mismatch > 0).sum()} (of {(g.n_stereo > 0).sum()} with "
                 f"stereocentres); E/Z mismatch: {(g.ez_mismatch > 0).sum()} (of {(g.n_ez > 0).sum()}); ring pucker "
                 f"differs: {(g.ring_pucker_diff > 0).sum()} (of {(g.n_rings > 0).sum()} with non-planar rings)")
        bad = g[(g.stereo_mismatch > 0) | (g.ez_mismatch > 0) | (g.ring_pucker_diff > 0)]
        good = g.drop(bad.index)
        P.append(f"#1 dE median: with any internal mismatch {bad.dE.median():.0f} (n {len(bad)}), without "
                 f"{good.dE.median():.0f} (n {len(good)})")
        P.append(bad.sort_values("dE", ascending=False)[["case", "dE", "n_stereo", "stereo_mismatch", "n_ez",
                                                          "ez_mismatch", "n_rings", "ring_pucker_diff",
                                                          "ring_tors_maxdiff"]].round(0).to_string(index=False))
    txt = "\n".join(P)
    (OUT / "internal_stereo.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
