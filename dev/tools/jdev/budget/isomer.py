"""The reference's isomer as an explicit donor -> site assignment (ARCHITECTURE.md, isomer-matched track, 2026-10-16).

tmQM structures are GFN2 minima of the crystallised isomer, which need not be the lowest or the most common one.
To measure precision within the reference's isomer, the solver is given that isomer explicitly: every donor group is
pinned to the polyhedron site its crystal direction occupies (LigandInput.sites; the engine then skips the
enumeration and solves that one assignment).

Site matching: the best PROPER rotation of the ideal polyhedron onto the crystal donor directions (centroids for
haptic groups) with the optimal bijection (Hungarian on cosines), so the reference enantiomer is matched, not its
mirror image. Seeded from every (donor pair -> site pair) frame alignment, refined by Kabsch until the bijection is
stable.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _frame(u, v):
    e1 = u / np.linalg.norm(u)
    e2 = v - (v @ e1) * e1
    n = np.linalg.norm(e2)
    if n < 1e-6:
        return None
    e2 = e2 / n
    return np.stack([e1, e2, np.cross(e1, e2)], 1)


def _kabsch(P, Q):
    """Proper rotation R minimising |R P_i - Q_i|."""
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def match_sites(dirs: np.ndarray, sites: np.ndarray):
    """(assignment donor -> site, mean angular deviation in degrees) for unit donor directions `dirs` (n, 3)."""
    n = len(dirs)
    assert len(sites) == n
    if n <= 2:  # linear / one site: every bijection is one isomer up to rotation
        asg = np.arange(n)
        ang = np.degrees(np.arccos(np.clip(abs(float(dirs[0] @ dirs[-1])), -1, 1))) if n == 2 else 0.0
        return asg, float(ang / 2)
    best = (None, -np.inf)
    # anchor donor 0 and the donor most orthogonal to it
    a = 0
    b = int(np.argmin(np.abs(dirs @ dirs[a]) + np.eye(n)[a] * 9))
    Fd = _frame(dirs[a], dirs[b])
    cab = float(dirs[a] @ dirs[b])
    for i in range(n):
        for j in range(n):
            if i == j or abs(float(sites[i] @ sites[j]) - cab) > 0.6:
                continue
            Fs = _frame(sites[i], sites[j])
            if Fd is None or Fs is None:
                continue
            R = Fs @ Fd.T
            for _ in range(10):
                C = (dirs @ R.T) @ sites.T
                r, c = linear_sum_assignment(-C)
                asg = c[np.argsort(r)]
                R2 = _kabsch(dirs, sites[asg])
                if np.allclose(R2, R, atol=1e-6):
                    break
                R = R2
            tot = float(((dirs @ R.T) * sites[asg]).sum())
            if tot > best[1]:
                best = (asg.copy(), tot, R)
    asg, tot = best[0], best[1]
    if asg is None:
        return None, np.nan
    R = best[2]
    ang = np.degrees(np.arccos(np.clip(((dirs @ R.T) * sites[asg]).sum(1), -1, 1)))
    return asg, float(ang.mean())


def reference_sites(case):
    """Per ligand input, the site of each donor group in the reference structure; (None, reason) when the geometry
    has no fixed site set or the donor groups do not fill it."""
    from jacobik.builder.engine import geometry_sites
    from jacobik.ik.targets import IDEAL_GEOMETRIES

    m = case.tmqm_mol
    geom = m.coord_geom
    if geom not in IDEAL_GEOMETRIES:
        return None, f"geometry {geom!r} has no site set"
    sites = geometry_sites(geom)
    rows = {a.idx: i for i, a in enumerate(m.atoms)}
    M = case.ref_X[case.ref_metal_row]
    dirs, owner = [], []
    for li, (lin, lg) in enumerate(zip(case.ligand_inputs, case.ligand_groups)):
        for gi, grp in enumerate(lin.donors):
            P = np.array([case.ref_X[rows[lg["rdkit_to_xyz"][r]]] for r in grp])
            v = P.mean(0) - M
            dirs.append(v / np.linalg.norm(v))
            owner.append((li, gi))
    if len(dirs) != len(sites):
        return None, f"{len(dirs)} donor groups for {len(sites)} sites"
    asg, dev = match_sites(np.array(dirs), sites)
    if asg is None:
        return None, "no site match"
    out = [[None] * len(lin.donors) for lin in case.ligand_inputs]
    for (li, gi), s in zip(owner, asg):
        out[li][gi] = int(s)
    return out, dev


def isomer_builder(case):
    """(make_builder for the reference isomer, mean site deviation deg) or (None, reason)."""
    from jacobik import ComplexBuilder, LigandInput

    s, dev = reference_sites(case)
    if s is None:
        return None, dev
    m = case.tmqm_mol
    ligs = [LigandInput(lin.smiles, donors=lin.donors, sites=ss) for lin, ss in zip(case.ligand_inputs, s)]
    return (lambda: ComplexBuilder(metal=m.metal_symbol, geometry=m.coord_geom, charge=m.charge, spin=m.spin_state,
                                   ligands=ligs)), dev
