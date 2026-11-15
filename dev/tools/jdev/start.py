"""Starting-geometry modes for a solve.

  etkdg            the normal engine path (ETKDG embed of each free ligand). Production behaviour.
  full_oracle      DIAGNOSTIC: every ligand starts from its own crystal internal geometry
                   (heavy atoms via the benchmark atom mapping, H placed by parent atom).
  bodies_oracle    DIAGNOSTIC (complement of junction_oracle): each rigid body gets its CRYSTAL internal
                   geometry but is placed onto the ETKDG body (heavy-atom Kabsch), so the angles BETWEEN rigid
                   bodies stay ETKDG. Groups with < 3 heavy atoms keep their ETKDG positions.
  junction_oracle  DIAGNOSTIC: each rigid body keeps its ETKDG internal geometry but is placed where it
                   sits in the crystal (heavy-atom Kabsch), so the angles BETWEEN rigid bodies are crystal.
                   Groups with < 3 heavy atoms (CH3, NH2, a lone B) keep crystal positions.

Oracles use the reference structure and exist only to measure what a better starting geometry could
buy. They must never be used by, or tuned into, the solver itself.
"""
from __future__ import annotations

import contextlib

import numpy as np

MODES = ("etkdg", "full_oracle", "junction_oracle", "bodies_oracle")


def _crystal_ligand_coords(case) -> dict:
    """smiles -> list (one per ligand instance, in input order) of (n_atoms_with_H, 3) crystal coordinates in RDKit
    (engine) atom order. Per instance: identical ligands (3x CO, 2x Cp) used to collapse onto one crystal copy
    (fixed 2026-09-29), which broke the junction/bodies oracles for any complex with repeated ligands."""
    from rdkit import Chem

    m = case.tmqm_mol
    rows = {a.idx: i for i, a in enumerate(m.atoms)}
    X = case.ref_X
    Hrows = [i for i, a in enumerate(m.atoms) if a.symbol == "H"]
    used, out = set(), {}
    for lg, li in zip(case.ligand_groups, case.ligand_inputs):
        mol = Chem.AddHs(Chem.MolFromSmiles(li.smiles))
        P = np.zeros((mol.GetNumAtoms(), 3))
        r2x = lg["rdkit_to_xyz"]
        for k, x1 in r2x.items():
            P[k] = X[rows[x1]]
        for k in r2x:
            hs = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetAtomicNum() == 1]
            cand = sorted((h for h in Hrows if h not in used and np.linalg.norm(X[h] - P[k]) < 1.25),
                          key=lambda h: np.linalg.norm(X[h] - P[k]))
            if len(cand) < len(hs):
                raise RuntimeError(f"{case.name}: could not map H atoms of ligand {li.smiles}")
            for hi, hr in zip(hs, cand):
                P[hi] = X[hr]
                used.add(hr)
        out.setdefault(li.smiles, []).append(P)
    return out


def _junction_ligand_coords(case, crystal: dict, swap: bool = False) -> dict:
    """ETKDG rigid bodies placed on their crystal positions (swap=True: crystal bodies placed on the
    ETKDG positions, i.e. bodies_oracle)."""
    from jacobik.topology.decompose import decompose_ligand

    b = case.make_builder()
    mol, _, _ = b._build_shell(False, False)
    out = {}
    seen = {}
    for lig, li in zip(sorted(mol.ligands, key=lambda l: l.ligand_id), case.ligand_inputs):
        decompose_ligand(mol, lig)
        ai = list(lig.atom_indices)
        E = np.array([mol.atoms[i - 1].position for i in ai])
        n_seen = seen.get(li.smiles, 0); seen[li.smiles] = n_seen + 1
        P = crystal[li.smiles][n_seen]
        if swap:
            E, P = P, E          # move crystal bodies onto the ETKDG frame
        H = P.copy()
        sy = [mol.atoms[i - 1].symbol for i in ai]
        for body in lig.decomposition.rigid_bodies:
            rd = [ai.index(i) for i in body if i in ai]
            hv = [r for r in rd if sy[r] != "H"]
            if len(hv) < 3:
                continue
            ca, cb = E[hv].mean(0), P[hv].mean(0)
            U, _, Vt = np.linalg.svd((E[hv] - ca).T @ (P[hv] - cb))
            R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            H[rd] = (E[rd] - ca) @ R + cb
        out.setdefault(li.smiles, []).append(H)
    return out


@contextlib.contextmanager
def starting_geometry(case, mode: str):
    """Temporarily replace the engine's per-ligand embedding for `case`."""
    if mode == "etkdg":
        yield
        return
    if mode not in MODES:
        raise ValueError(f"unknown start mode {mode!r}; choose from {MODES}")
    if case.tmqm_mol is None:
        raise ValueError(f"{mode} needs a tmQM case with a crystal reference")
    import jacobik.builder.engine as eng
    from rdkit import Chem

    coords = _crystal_ligand_coords(case)
    if mode == "junction_oracle":
        coords = _junction_ligand_coords(case, coords)
    elif mode == "bodies_oracle":
        coords = _junction_ligand_coords(case, coords, swap=True)
    orig = eng._smiles_to_conformer
    served = {}

    def patched(smiles, fast_embed=False, **kw):
        if smiles not in coords:
            return orig(smiles, fast_embed=fast_embed, **kw)
        k = served.get(smiles, 0); served[smiles] = k + 1
        P = coords[smiles][k % len(coords[smiles])]      # instances in ligand order
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        conf = Chem.Conformer(m.GetNumAtoms())
        for i, p in enumerate(P):
            conf.SetAtomPosition(i, p.tolist())
        m.AddConformer(conf, assignId=True)
        return m, P.copy(), "[H]" in smiles

    eng._smiles_to_conformer = patched
    try:
        yield
    finally:
        eng._smiles_to_conformer = orig
