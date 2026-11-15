"""Plan 3 N2a (2026-10-17 c): which noncovalent interactions do crystals have that our #1 picks miss?

Per complex, counts of five interaction types (geometric criteria, pairs Q6 calls class 5 or a ring / H not bonded to
the partner) in a structure:
  hbond     polar H (bonded to N, O, F) ... acceptor (N, O, F, S, Cl, Br, I), H...A <= 2.5 A, D-H...A >= 120 deg
            excluding D/A both metal-bound (shared with stats/nci.py)
  agostic   metal ... C-H (the C not a donor, the H not a metal 1-3 atom), M...H <= 2.4 A
  pipi      two planar 5- / 6-rings (sp2 atoms, RMS out-of-plane < 0.1 A) on different ligands or >= 5 bonds apart,
            centroid distance <= 4.4 A, inter-plane angle <= 30 deg
  chpi      non-polar H ... ring centroid <= 3.0 A, centroid->H within 30 deg of the ring normal, H not on the ring or
            its substituent atoms
  xbond     C-X (X = Cl, Br, I) ... acceptor (N, O, S, halide), X...A <= Alvarez vdW sum, C-X...A >= 150 deg
For dev80: the crystal (tmQM reference) and the #1 pick of a pool arm (seed 11); for a training sample: the crystal.
Admission rule (plan): a type enters NCI v1 if >= 5 dev80 crystals have it and the #1 misses it in >= half of those,
or >= 5 % of training complexes have it.

    python tools/jdev/budget/nci_census.py [--arm etkdg_u1_os] [--train 3000]
Output: <OUT>/nci_census.csv (dev80 rows) and nci_census.txt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "research/bench/tools/jdev/budget"))
sys.path.insert(0, str(REPO))
TYPES = ("hbond", "agostic", "pipi", "chpi", "xbond")
ACC = {"N", "O", "F", "S", "Cl", "Br", "I"}
ALV = {"H": 1.20, "B": 1.91, "C": 1.77, "N": 1.66, "O": 1.50, "F": 1.46, "Si": 2.19, "P": 1.90, "S": 1.89, "Cl": 1.82,
       "Se": 1.82, "Br": 1.86, "I": 2.04}


def _graph(mol):
    from jacobik.topology.graph import get_ligand_neighbors

    return get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)


def _rings(mol, nb, P, pos):
    """Planar 5- / 6-rings of sp2-like atoms (<= 3 ligand neighbours): (atom idx list, centroid, unit normal)."""
    out, seen = [], set()
    nodes = [a.idx for a in mol.atoms if a.idx != mol.metal_idx and a.symbol in ("C", "N", "O", "S", "B", "P")]
    sym = {a.idx: a.symbol for a in mol.atoms}

    def dfs(start, cur, path):
        for v in nb.get(cur, ()):
            if v == start and len(path) in (5, 6):
                key = frozenset(path)
                if key not in seen:
                    seen.add(key)
                    yield list(path)
            elif v not in path and len(path) < 6 and v > start and sym.get(v) in ("C", "N", "O", "S", "B", "P"):
                yield from dfs(start, v, path + [v])

    for s in nodes:
        for ring in dfs(s, s, [s]):
            if any(len(nb.get(u, ())) > 3 for u in ring):
                continue
            Q = P[[pos[u] for u in ring]]
            c = Q.mean(0)
            _, sv, vt = np.linalg.svd(Q - c)
            if np.sqrt((sv[-1] ** 2) / len(ring)) < 0.1:
                out.append((ring, c, vt[-1]))
    return out


def census(mol, X=None):
    """{type: count} for mol at coordinates X (mol order; default the mol's own positions)."""
    from jacobik.stats import keys as kb
    from research.archive.code.interactions.nci import hbond_candidates
    from research.archive.code.interactions.q6pmf import ligand_separation
    from jacobik.topology.graph import donor_indices_of

    atoms = list(mol.atoms)
    pos = {a.idx: i for i, a in enumerate(atoms)}
    P = np.asarray(X if X is not None else [a.position for a in atoms], dtype=float)
    sym = {a.idx: a.symbol for a in atoms}
    nb = _graph(mol)
    sep = ligand_separation(mol, max_depth=4)
    far = lambda i, j: j not in sep.get(i, {})  # > 4 ligand bonds apart or no ligand path (as stats/nci.py)
    d = lambda i, j: float(np.linalg.norm(P[pos[i]] - P[pos[j]]))
    ang = lambda a, b, c: float(np.degrees(np.arccos(np.clip(
        np.dot(P[pos[a]] - P[pos[b]], P[pos[c]] - P[pos[b]]) /
        max(np.linalg.norm(P[pos[a]] - P[pos[b]]) * np.linalg.norm(P[pos[c]] - P[pos[b]]), 1e-9), -1, 1))))
    polar = kb.q6_polar_h(mol, nb)
    out = dict.fromkeys(TYPES, 0)
    lig = [a.idx for a in atoms if a.idx != mol.metal_idx]
    acc = [i for i in lig if sym[i] in ACC]
    for dn, h, a, _ in hbond_candidates(mol, nb, sep):
        if d(h, a) <= 2.5 and ang(dn, h, a) >= 120:
            out["hbond"] += 1
    don = set(donor_indices_of(mol))
    m = mol.metal_idx
    for h in lig:
        if sym[h] != "H" or h in polar:
            continue
        neighbours = nb.get(h, ())
        c = neighbours[0] if neighbours else None
        if c is None or sym.get(c) != "C" or c in don or any(x in don for x in nb.get(c, ())):
            continue
        if d(m, h) <= 2.4:
            out["agostic"] += 1
    rings = _rings(mol, nb, P, pos)
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            ri, rj = rings[i], rings[j]
            if set(ri[0]) & set(rj[0]) or not all(far(u, v) for u in ri[0] for v in rj[0]):
                continue
            cc = np.linalg.norm(ri[1] - rj[1])
            th = np.degrees(np.arccos(min(1.0, abs(float(np.dot(ri[2], rj[2]))))))
            if cc <= 4.4 and th <= 30:
                out["pipi"] += 1
    for ring, c, n in rings:
        near = set(ring) | {x for u in ring for x in nb.get(u, ())}
        for h in lig:
            if sym[h] != "H" or h in polar or h in near or any(x in near for x in nb.get(h, ())):
                continue
            v = P[pos[h]] - c
            r = np.linalg.norm(v)
            if r <= 3.0 and np.degrees(np.arccos(min(1.0, abs(float(np.dot(v / r, n)))))) <= 30:
                out["chpi"] += 1
    for x in lig:
        if sym[x] not in ("Cl", "Br", "I"):
            continue
        cs = [c for c in nb.get(x, ()) if sym.get(c) == "C"]
        if not cs:
            continue
        for a in acc:
            if a != x and far(x, a) and d(x, a) <= ALV[sym[x]] + ALV.get(sym[a], 1.6) and ang(cs[0], x, a) >= 150:
                out["xbond"] += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="etkdg_u1_os")
    p.add_argument("--train", type=int, default=3000)
    a = p.parse_args()
    from common import pool_path, read_pool
    from pools import OUT, load_stats, set_cases, setup

    setup()
    import pandas as pd
    from tools.jdev.cases import get_case

    st = load_stats()
    rows = []
    for s in ("dev32", "n48"):
        for code, name in set_cases(s):
            f = pool_path(a.arm, s, code, 11, OUT)
            if not f.exists():
                continue
            z = read_pool(f)
            if z is None:
                continue
            try:
                case = get_case(name)
                cry = census(case.tmqm_mol)
                r = case.make_builder().solve(seed=11, stats=st, angle_bends=True, n_conformers=1, n_ik_steps=2,
                                              device="cpu")
                pick = census(r.conformers[0].complex, z["X"][int(np.argmax(z["score"]))])
            except Exception as e:
                print(code, "ERR", e)
                continue
            rows.append(dict(code=code, **{f"cry_{t}": cry[t] for t in TYPES}, **{f"pick_{t}": pick[t] for t in TYPES}))
            print(code, {t: (cry[t], pick[t]) for t in TYPES if cry[t] or pick[t]}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "nci_census.csv", index=False)
    from fit.parser import TMQMParser
    from fit.pipeline.extract import benchmark_codes
    excl = benchmark_codes()
    tr, n = {t: 0 for t in TYPES}, 0
    for mol in TMQMParser(REPO / "data" / "tmQM").stream_split("X1"):
        if n >= a.train:
            break
        if mol.csd_code in excl:
            continue
        try:
            c = census(mol)
        except Exception:
            continue
        n += 1
        for t in TYPES:
            tr[t] += c[t] > 0
    lines = [f"NCI census: dev80 {len(df)} complexes (crystal vs {a.arm} #1, seed 11); training sample {n} (tmQM X1)",
             "type     crystals with it | picks missing (fewer than crystal) | picks with extra | training share"]
    for t in TYPES:
        has = df[f"cry_{t}"] > 0
        miss = (df[f"pick_{t}"] < df[f"cry_{t}"]) & has
        extra = df[f"pick_{t}"] > df[f"cry_{t}"]
        admit = (has.sum() >= 5 and miss.sum() >= 0.5 * has.sum()) or tr[t] / max(n, 1) >= 0.05
        lines.append(f"{t:8s} {has.sum():3d}  ({', '.join(df.code[has].head(8))})  | {miss.sum():3d} | {extra.sum():3d} "
                     f"| {tr[t] / max(n, 1):6.1%}   -> {'ADMIT' if admit else 'not admitted'}")
    txt = "\n".join(lines)
    (OUT / "nci_census.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
