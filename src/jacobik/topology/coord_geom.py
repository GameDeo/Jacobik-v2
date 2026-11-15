"""
coord_geom.py
=============
Coordination geometry classifier for TMComplex objects.
"""

from __future__ import annotations

import itertools as _it
import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from jacobik.core.types import TMComplex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dmd_angles(metal_pos: np.ndarray, donor_positions: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise D–M–D angles (in degrees) for a set of donor
    positions relative to the metal.

    Parameters
    ----------
    metal_pos      : (3,) metal Cartesian position
    donor_positions: (n, 3) donor effective positions (atom or centroid)

    Returns
    -------
    angles : (k,) array of angles in degrees, where k = n*(n-1)/2.
             Sorted descending.
    """
    n = len(donor_positions)
    vecs = donor_positions - metal_pos  # (n, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid zero-division for degenerate positions
    norms = np.where(norms < 1e-8, 1.0, norms)
    unit = vecs / norms  # (n, 3)

    angles: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_theta = float(np.clip(np.dot(unit[i], unit[j]), -1.0, 1.0))
            angles.append(np.degrees(np.arccos(cos_theta)))

    return np.array(sorted(angles, reverse=True), dtype=float)


def _tau5(alpha: float, beta: float) -> float:
    """τ₅ = (α − β) / 60  (Addison & Reedijk 1984)."""
    return (alpha - beta) / 60.0


# ---------------------------------------------------------------------------
# CN-specific classifiers
# ---------------------------------------------------------------------------


def _cn2() -> str:
    return "linear"


def _cn5(angles: np.ndarray) -> str:
    alpha, beta = float(angles[0]), float(angles[1])
    tau = _tau5(alpha, beta)
    tau = max(0.0, min(1.0, tau))
    if tau > 0.60:
        return "trig_bipyr"
    return "square_pyr"


def _cn7() -> str:
    return "pent_bipyr"


def _cn8() -> str:
    return "square_anti"


def _cn9() -> str:
    return "tricap_pris"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify(mol: "TMComplex") -> str:
    """
    Classify the coordination geometry of a TMComplex and return the label.

    Parameters
    ----------
    mol : TMComplex
        Must have mol.coord_sites populated and mol.metal_idx set.

    Returns
    -------
    str
        One of: linear, t_shaped, trigonal, square_plan, see_saw, tetrahedral,
        square_pyr, trig_bipyr, octahedral, trig_pris, pent_bipyr, square_anti,
        tricap_pris, other.  Returns "other" on any failure
        (empty coord_sites, degenerate geometry, numerical error) so that
        the pipeline never raises here.
    """
    try:
        return _classify(mol)
    except Exception as exc:
        log.debug("%s: geometry classification failed: %s", getattr(mol, "csd_code", "?"), exc)
        return "other"


#: Nearest-ideal-shape labelling, frozen 2026-10-16 (geometry labels, unrelated to scorer shaping); read by stats.mapping.
SHAPE_LABELS = True

#: Candidate ideal polyhedra per coordination number (CN 7-9 have one ideal each in targets.IDEAL_GEOMETRIES).
SHAPES_BY_CN = {3: ("trigonal", "t_shaped"), 4: ("tetrahedral", "square_plan", "see_saw"),
                5: ("square_pyr", "trig_bipyr"), 6: ("octahedral", "trig_pris")}
_PERMS: dict = {}


def shape_deviations(directions: np.ndarray) -> dict:
    """{shape: deviation} for the ideal polyhedra of this coordination number.

    deviation = min over donor-to-vertex assignments and rotations of the mean squared distance between the donor unit
    vectors (metal -> donor or haptic centroid) and the ideal unit vectors (a continuous shape measure on directions;
    0 = perfect). Exact: every assignment is tried (CN <= 6: at most 720), each with its optimal rotation (Kabsch),
    batched in numpy.
    """
    from jacobik.ik.targets import IDEAL_GEOMETRIES

    u = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    n = len(u)
    if n not in _PERMS:
        _PERMS[n] = np.array(list(_it.permutations(range(n))))
    P = _PERMS[n]
    out = {}
    for g in SHAPES_BY_CN.get(n, ()):
        V = IDEAL_GEOMETRIES[g][P]                        # (p, n, 3) ideal vertices in each assignment order
        H = np.einsum("pni,nj->pij", V, u)                # cross-covariance per assignment
        Uu, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.einsum("pij,pjk->pik", Uu, Vt)))
        S[:, -1] *= d                                     # proper rotations only
        out[g] = float((2.0 - 2.0 * S.sum(axis=1) / n).min())  # mean |R v - u|^2 = 2 - 2 tr(R H)/n
    return out


def best_fit_assignment(directions: np.ndarray, geom: str, n_starts: int = 48, seed: int = 0):
    """Vertex index of the ideal polyhedron ``geom`` for every site direction (best-fit assignment: minimum mean
    squared distance between the unit directions and the rotated ideal unit vectors), or None when ``geom`` has no
    ideal of this size. CN <= 6: exact (every assignment, each with its Kabsch rotation, as shape_deviations).
    CN 7-9: alternating optimal assignment (Hungarian) / optimal rotation from ``n_starts`` deterministic random
    starts (the full permutation search is 5040-362880 assignments)."""
    from jacobik.ik.targets import IDEAL_GEOMETRIES

    if geom not in IDEAL_GEOMETRIES:
        return None
    V0 = np.asarray(IDEAL_GEOMETRIES[geom], dtype=float)
    V0 = V0 / np.linalg.norm(V0, axis=1, keepdims=True)
    u = np.asarray(directions, dtype=float)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    n = len(u)
    if len(V0) != n:
        return None
    if n <= 6:
        if n not in _PERMS:
            _PERMS[n] = np.array(list(_it.permutations(range(n))))
        P = _PERMS[n]
        V = V0[P]
        H = np.einsum("pni,nj->pij", V, u)
        Uu, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.einsum("pij,pjk->pik", Uu, Vt)))
        S[:, -1] *= d
        return [int(x) for x in P[int(np.argmax(S.sum(axis=1)))]]
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.transform import Rotation

    rots = Rotation.random(n_starts, random_state=seed).as_matrix()
    best, best_val = None, -np.inf
    for R in rots:
        perm = None
        for _ in range(20):
            Vr = V0 @ R.T                                   # rotated ideal vertices
            _, new = linear_sum_assignment(-(u @ Vr.T))     # site i -> vertex new[i]
            if perm is not None and np.array_equal(new, perm):
                break
            perm = new
            H = V0[perm].T @ u                              # Kabsch: R maps ideal onto the sites
            Uu, S, Vt = np.linalg.svd(H)
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ Uu.T))])
            R = Vt.T @ D @ Uu.T
        val = float(np.sum(u * (V0[perm] @ R.T)))
        if val > best_val + 1e-12:
            best_val, best = val, [int(x) for x in perm]
    return best


def ideal_pair_angles(geom: str, assignment) -> Optional[np.ndarray]:
    """(n, n) ideal angles (degrees) between the vertices assigned to each pair of sites; None if unavailable."""
    from jacobik.ik.targets import IDEAL_GEOMETRIES

    if assignment is None or geom not in IDEAL_GEOMETRIES:
        return None
    V = np.asarray(IDEAL_GEOMETRIES[geom], dtype=float)[list(assignment)]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(V @ V.T, -1.0, 1.0)))


#: CN5 keeps the tau5 rule: tau5 reads the trans structure directly, while a nearest-shape label depends on the ideal
#: square pyramid (90 deg apex), and real square pyramids open to ~100-104 deg with chelating / bulky ligands (Q2's job),
#: which puts them nearer the ideal trigonal bipyramid. tmQM tau5 values are not bimodal, so the 0.6 cut is convention.
TAU5_CUT = 0.60


def classify_with_margin(mol: "TMComplex"):
    """(label, runner-up label, margin). CN3/4/6: runner-up deviation - best deviation (nearest ideal shape).
    CN5: |tau5 - TAU5_CUT| (tau5 rule). Runner-up None when the CN has one shape."""
    cn = len(mol.coord_sites)
    if cn not in SHAPES_BY_CN:
        return classify(mol), None, None
    metal_pos = np.asarray(mol.get_atom(mol.metal_idx).position, dtype=float)
    pos = np.array([np.asarray(s.position, dtype=float) for s in mol.coord_sites])
    if cn == 5:
        a = _dmd_angles(metal_pos, pos)
        tau = max(0.0, min(1.0, float(_tau5(float(a[0]), float(a[1])))))
        lab = "trig_bipyr" if tau > TAU5_CUT else "square_pyr"
        return lab, ("square_pyr" if lab == "trig_bipyr" else "trig_bipyr"), abs(tau - TAU5_CUT)
    dirs = pos - metal_pos
    dev = shape_deviations(dirs)
    order = sorted(dev, key=dev.get)
    return order[0], order[1], dev[order[1]] - dev[order[0]]


def _classify(mol: "TMComplex") -> str:
    cn = len(mol.coord_sites)
    if cn in SHAPES_BY_CN and cn != 5:
        return classify_with_margin(mol)[0]

    if cn <= 1:
        return "other"
    if cn == 2:
        return _cn2()
    if cn >= 10:
        return "other"

    # Build donor position matrix
    metal_pos = np.asarray(mol.get_atom(mol.metal_idx).position, dtype=float)
    donor_positions = np.array(
        [np.asarray(s.position, dtype=float) for s in mol.coord_sites]
    )  # (cn, 3)

    angles = _dmd_angles(metal_pos, donor_positions)

    if cn == 5:
        return _cn5(angles)
    if cn == 7:
        return _cn7()
    if cn == 8:
        return _cn8()
    if cn == 9:
        return _cn9()

    return "other"
