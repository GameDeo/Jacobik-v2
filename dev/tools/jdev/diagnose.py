"""Localise why one solved complex is far above its reference (tmQM cases).

Reports, for the best-scoring conformer:
  1. donors: M-D solved vs crystal
  2. energy map: MACE per-atom energies (local part), each H folded into its heavy parent, mapped atom-for-atom
     onto the crystal via the benchmark mapping; top heavy atoms by (solved - crystal) excess, with element,
     ligand, whether it is a donor / junction atom, and its bond angles solved vs crystal
  3. close contacts in the solved structure (non-bonded heavy-heavy < 2.6 A, H-H < 1.7 A, H-heavy < 2.0 A)
MACE only MEASURES here. Note: MACE-POLAR's long-range electrostatics are not per-atom, so the atom sum does not
equal dE_sp; the map says WHERE local strain sits, not the whole budget.
"""
from __future__ import annotations

import itertools

import numpy as np


def _parents(sy, X):
    """H index -> nearest heavy atom index"""
    heavy = [i for i, s in enumerate(sy) if s != "H"]
    out = {}
    for i, s in enumerate(sy):
        if s == "H":
            out[i] = min(heavy, key=lambda j: np.linalg.norm(X[i] - X[j]))
    return out


def _fold(sy, X, e):
    """per-heavy-atom energy with bonded H folded in"""
    f = {i: e[i] for i, s in enumerate(sy) if s != "H"}
    for h, p in _parents(sy, X).items():
        f[p] += e[h]
    return f


def _angles(X, center, nbrs):
    return {tuple(sorted((a, b))): float(np.degrees(np.arccos(np.clip(
        np.dot((X[a] - X[center]) / np.linalg.norm(X[a] - X[center]), (X[b] - X[center]) / np.linalg.norm(X[b] - X[center])), -1, 1))))
        for a, b in itertools.combinations(nbrs, 2)}


def diagnose(case, best, sy, X, ev, top: int = 8) -> dict:
    from jacobik.core.energy import EV_TO_KJ_MOL as K
    from ase import Atoms

    m = case.tmqm_mol
    rows = {a.idx: i for i, a in enumerate(m.atoms)}
    # heavy-atom map solved row -> reference row (+ ligand index, donor flag)
    p2r, lig_of, donors = {}, {}, set()
    for li, (lig, lin, lg) in enumerate(zip(sorted(best.complex.ligands, key=lambda l: l.ligand_id), case.ligand_inputs,
                                            case.ligand_groups)):
        for rd, x1 in lg["rdkit_to_xyz"].items():
            if rd < len(lig.atom_indices):
                p = lig.atom_indices[rd] - 1
                if sy[p] != "H":
                    p2r[p] = rows[x1]; lig_of[p] = li
        for g in lin.donors:
            for dd in g:
                donors.add(lig.atom_indices[dd] - 1)
    p2r[0] = case.ref_metal_row

    calc = ev._get_calculator()

    def per_atom(s, P):
        a = Atoms(symbols=s, positions=P, pbc=False)
        a.info.update(charge=int(case.charge), spin=int(case.mult), external_field=[0.0, 0.0, 0.0]); a.calc = calc
        a.get_potential_energy()
        return np.asarray(a.get_potential_energies()) * K

    es, er = per_atom(sy, X), per_atom(case.ref_symbols, case.ref_X)
    fs, fr = _fold(sy, X, es), _fold(case.ref_symbols, case.ref_X, er)
    # heavy neighbours (by distance) in the solved structure, metal excluded
    COV = {"H": .31, "B": .84, "C": .76, "N": .71, "O": .66, "F": .57, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.2, "I": 1.39}
    def nbrs(P, s_, i):
        return [j for j in range(len(s_)) if j != i and j != 0 and s_[j] != "H" and s_[i] in COV and s_[j] in COV
                and np.linalg.norm(P[i] - P[j]) < 1.2 * (COV[s_[i]] + COV[s_[j]])]
    rows_out = []
    for p, r in p2r.items():
        if p == 0:
            continue
        exc = fs[p] - fr[r]
        nb = nbrs(X, sy, p)
        ang_s = _angles(X, p, nb + ([0] if p in donors else []))
        # same angles in the crystal via the mapping
        inv = {pp: rr for pp, rr in p2r.items()}
        ang_r = {}
        for (a, b) in ang_s:
            if a in inv and b in inv:
                ra, rb = inv[a], inv[b]
                ang_r[(a, b)] = _angles(case.ref_X, r, [ra, rb])[tuple(sorted((ra, rb)))]
        worst = sorted(((abs(ang_s[k] - ang_r[k]), k) for k in ang_r), reverse=True)[:2]
        rows_out.append(dict(atom=f"{sy[p]}{p}", ligand=lig_of.get(p), donor=p in donors, n_heavy_nbrs=len(nb),
                             excess_kJ=float(exc),
                             worst_angles=[f"{sy[k[0]]}{k[0]}-{sy[p]}{p}-{sy[k[1]]}{k[1]}: {ang_s[k]:.0f} vs {ang_r[k]:.0f}"
                                           for _, k in worst]))
    rows_out.sort(key=lambda d: -d["excess_kJ"])
    per_lig = {}
    for d in rows_out:
        per_lig[d["ligand"]] = per_lig.get(d["ligand"], 0.0) + d["excess_kJ"]
    # contacts
    contacts = []
    import networkx as nx
    G = nx.Graph()
    for i in range(1, len(sy)):
        for j in range(i + 1, len(sy)):
            if sy[i] in COV and sy[j] in COV and np.linalg.norm(X[i] - X[j]) < 1.2 * (COV[sy[i]] + COV[sy[j]]):
                G.add_edge(i, j)
    near = {i: nx.single_source_shortest_path_length(G, i, cutoff=3) if i in G else {} for i in range(1, len(sy))}
    for i in range(1, len(sy)):
        for j in range(i + 1, len(sy)):
            if j in near[i]:
                continue  # within 3 bonds: not a contact
            dd = float(np.linalg.norm(X[i] - X[j]))
            lim = 1.7 if sy[i] == sy[j] == "H" else (2.0 if "H" in (sy[i], sy[j]) else 2.6)
            if dd < lim:
                contacts.append(f"{sy[i]}{i}..{sy[j]}{j} {dd:.2f}")
    return {"ligand_smiles": [li.smiles for li in case.ligand_inputs], "top_atoms": rows_out[:top], "excess_by_ligand": {str(k): round(v, 1) for k, v in per_lig.items()},
            "local_excess_total": float(sum(d["excess_kJ"] for d in rows_out)), "close_contacts": contacts[:15],
            "n_close_contacts": len(contacts)}
