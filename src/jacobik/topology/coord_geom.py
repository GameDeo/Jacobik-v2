"""
coord_geom.py
=============
Coordination geometry classifier for TMComplex objects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from jacobik.core.types import TMComplex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public label vocabulary (mirrors DESIGN.md §3.2)
# ---------------------------------------------------------------------------

GEOM_LABELS = frozenset(
    {
        "linear",
        "t_shaped",
        "trigonal",
        "square_plan",
        "see_saw",
        "tetrahedral",
        "square_pyr",
        "trig_bipyr",
        "octahedral",
        "trig_pris",
        "pent_bipyr",
        "square_anti",
        "other",
    }
)

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


def _tau4(alpha: float, beta: float) -> float:
    """τ₄ = (360 − α − β) / 141  (Yang et al. 2007)."""
    return (360.0 - alpha - beta) / 141.0


def _tau5(alpha: float, beta: float) -> float:
    """τ₅ = (α − β) / 60  (Addison & Reedijk 1984)."""
    return (alpha - beta) / 60.0


def _tau3(alpha: float, beta: float) -> float:
    """τ₃ = (β − α) / 60  (Okuniewski et al. 2016); α ≥ β."""
    # Note: for CN=3, α is the *largest* angle, β the second largest.
    # The formula gives 0 for T-shape (α≈180, β≈90) and 1 for trigonal (α≈β≈120).
    return (beta - alpha) / 60.0


# ---------------------------------------------------------------------------
# CN-specific classifiers
# ---------------------------------------------------------------------------


def _cn2() -> str:
    return "linear"


def _cn3(angles: np.ndarray) -> str:
    alpha, beta = float(angles[0]), float(angles[1])
    tau = _tau3(alpha, beta)
    # See exp.md for detailed rationale
    tau_mapped = 1.0 + tau  # = 1 + (β−α)/60; T-shape→0, trigonal→1
    if tau_mapped < 0.40:
        return "t_shaped"
    return "trigonal"


def _cn4(angles: np.ndarray) -> str:
    alpha, beta = float(angles[0]), float(angles[1])
    tau = _tau4(alpha, beta)
    tau = max(0.0, min(1.0, tau))  # clamp numerical noise

    # See exp.md for detailed rationale
    if alpha < 155.0:
        return "tetrahedral" if tau > 0.65 else "square_plan"

    if tau < 0.25:
        return "square_plan"
    if tau > 0.75:
        return "tetrahedral"
    return "see_saw"


def _cn5(angles: np.ndarray) -> str:
    alpha, beta = float(angles[0]), float(angles[1])
    tau = _tau5(alpha, beta)
    tau = max(0.0, min(1.0, tau))
    if tau > 0.60:
        return "trig_bipyr"
    return "square_pyr"


def _cn6(angles: np.ndarray) -> str:
    # Count angles > 150° as a proxy for trans pairs.
    # Ideal OC-6: three ~180° pairs → count = 3.
    # Ideal TP-6: largest angles ~120°              → count = 0.
    n_trans = int(np.sum(angles > 150.0))
    if n_trans >= 2:
        return "octahedral"
    if n_trans == 0:
        return "trig_pris"
    # n_trans == 1: one trans pair — distorted, classify by the largest angle
    return "octahedral" if float(angles[0]) > 160.0 else "trig_pris"


def _cn7() -> str:
    return "pent_bipyr"


def _cn8() -> str:
    return "square_anti"


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
        One of the GEOM_LABELS strings.  Returns "other" on any failure
        (empty coord_sites, degenerate geometry, numerical error) so that
        the pipeline never raises here.

    """
    try:
        return _classify(mol)
    except Exception as exc:
        log.debug("%s: geometry classification failed: %s", getattr(mol, "csd_code", "?"), exc)
        return "other"


def _classify(mol: "TMComplex") -> str:
    cn = len(mol.coord_sites)

    if cn <= 1:
        return "other"
    if cn == 2:
        return _cn2()
    if cn >= 9:
        return "other"

    # Build donor position matrix
    metal_pos = np.asarray(mol.get_atom(mol.metal_idx).position, dtype=float)
    donor_positions = np.array(
        [np.asarray(s.position, dtype=float) for s in mol.coord_sites]
    )  # (cn, 3)

    angles = _dmd_angles(metal_pos, donor_positions)

    if cn == 3:
        return _cn3(angles)
    if cn == 4:
        return _cn4(angles)
    if cn == 5:
        return _cn5(angles)
    if cn == 6:
        return _cn6(angles)
    if cn == 7:
        return _cn7()
    if cn == 8:
        return _cn8()

    return "other"  # unreachable given guard above, but satisfies type checker
