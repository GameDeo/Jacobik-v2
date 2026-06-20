"""
coordination.py
===============
Coordination sphere construction for the IK engine.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    # Avoid hard import at module level; only needed for type hints
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT2 = math.sqrt(2.0)
_ROOT3 = math.sqrt(3.0)

# Tilt angle for the square antiprism: arccos(1/√3)
_SATP_TILT = math.acos(1.0 / math.sqrt(3.0))  # ≈ 54.7356°
_SATP_COS = math.cos(_SATP_TILT)  # = 1/√3 ≈ 0.5774
_SATP_SIN = math.sin(_SATP_TILT)  # = √(2/3) ≈ 0.8165


# ---------------------------------------------------------------------------
# Ideal geometry unit vectors
# ---------------------------------------------------------------------------

#: Maps geometry label → (CN, 3) array of unit vectors, one row per site.
#: All vectors are unit-length and point from the metal toward the donor.
IDEAL_GEOMETRIES: Dict[str, np.ndarray] = {}

# Linear  CN=2
IDEAL_GEOMETRIES["linear"] = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)

# Trigonal planar  CN=3
# Equilateral triangle in xy-plane; first vertex along +x
IDEAL_GEOMETRIES["trigonal"] = np.array(
    [
        [1.0, 0.0, 0.0],
        [-0.5, _ROOT3 / 2.0, 0.0],
        [-0.5, -_ROOT3 / 2.0, 0.0],
    ],
    dtype=float,
)

# T-shaped  CN=3
# ±x (the crossbar), +y (the stem); all in xy-plane
IDEAL_GEOMETRIES["t_shaped"] = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

# Square planar  CN=4
IDEAL_GEOMETRIES["square_plan"] = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)

# Tetrahedral  CN=4
# Four vertices of a regular tetrahedron; one vertex along (1,1,1)/√3
_TET_A = 1.0 / _ROOT3
_TET_B = -1.0 / _ROOT3
IDEAL_GEOMETRIES["tetrahedral"] = np.array(
    [
        [_TET_A, _TET_A, _TET_A],
        [_TET_A, _TET_B, _TET_B],
        [_TET_B, _TET_A, _TET_B],
        [_TET_B, _TET_B, _TET_A],
    ],
    dtype=float,
)

# See-saw  CN=4
# One equatorial pair (±x) + one axial (+z) + one tipped equatorial (−y)
# This is midway between square-planar and tetrahedral; τ₄ ≈ 0.5
# Standard convention: two axial sites (±z) and two equatorial at 120° apart
IDEAL_GEOMETRIES["see_saw"] = np.array(
    [
        [0.0, 0.0, 1.0],  # axial +z
        [0.0, 0.0, -1.0],  # axial −z
        [1.0, 0.0, 0.0],  # equatorial +x
        [-0.5, _ROOT3 / 2.0, 0.0],  # equatorial at 120°
    ],
    dtype=float,
)

# Square pyramidal  CN=5
# Four basal (±x, ±y) + one apical (+z)
IDEAL_GEOMETRIES["square_pyr"] = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# Trigonal bipyramidal  CN=5
# Equatorial triangle in xy-plane (first along +x) + ±z axial
IDEAL_GEOMETRIES["trig_bipyr"] = np.array(
    [
        [1.0, 0.0, 0.0],  # equatorial
        [-0.5, _ROOT3 / 2.0, 0.0],  # equatorial
        [-0.5, -_ROOT3 / 2.0, 0.0],  # equatorial
        [0.0, 0.0, 1.0],  # axial +z
        [0.0, 0.0, -1.0],  # axial −z
    ],
    dtype=float,
)

# Octahedral  CN=6
IDEAL_GEOMETRIES["octahedral"] = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)

# Trigonal prismatic  CN=6
# Two staggered equilateral triangles, eclipsed (not staggered like antiprismatic)
# Upper triangle at z=+1/√2, lower at z=−1/√2, both at r=1/√2 from z-axis
_TP_Z = 1.0 / _ROOT2
_TP_R = 1.0 / _ROOT2
IDEAL_GEOMETRIES["trig_pris"] = np.array(
    [
        [_TP_R, 0.0, _TP_Z],
        [-_TP_R / 2, _TP_R * _ROOT3 / 2, _TP_Z],
        [-_TP_R / 2, -_TP_R * _ROOT3 / 2, _TP_Z],
        [_TP_R, 0.0, -_TP_Z],
        [-_TP_R / 2, _TP_R * _ROOT3 / 2, -_TP_Z],
        [-_TP_R / 2, -_TP_R * _ROOT3 / 2, -_TP_Z],
    ],
    dtype=float,
)

# Pentagonal bipyramidal  CN=7
# Pentagon in xy-plane (first vertex along +x, 72° spacing) + ±z axial
_PENT_ANGLES = np.deg2rad([0.0, 72.0, 144.0, 216.0, 288.0])
IDEAL_GEOMETRIES["pent_bipyr"] = np.vstack(
    [
        np.column_stack([np.cos(_PENT_ANGLES), np.sin(_PENT_ANGLES), np.zeros(5)]),
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    ]
).astype(float)

# Square antiprismatic  CN=8 — ideal D₄d geometry
# Upper ring: 4 vectors at tilt arccos(1/√3) from z, azimuths 0°,90°,180°,270°
# Lower ring: 4 vectors at same tilt below xy, azimuths 45°,135°,225°,315°
# Tilt ensures the M–D–M angle between upper and lower neighbours is ideal.
_UP_AZ = np.deg2rad([0.0, 90.0, 180.0, 270.0])
_LO_AZ = np.deg2rad([45.0, 135.0, 225.0, 315.0])
IDEAL_GEOMETRIES["square_anti"] = np.vstack(
    [
        np.column_stack(
            [
                _SATP_SIN * np.cos(_UP_AZ),
                _SATP_SIN * np.sin(_UP_AZ),
                np.full(4, _SATP_COS),
            ]
        ),
        np.column_stack(
            [
                _SATP_SIN * np.cos(_LO_AZ),
                _SATP_SIN * np.sin(_LO_AZ),
                np.full(4, -_SATP_COS),
            ]
        ),
    ]
).astype(float)

# Freeze all arrays
for _k, _v in IDEAL_GEOMETRIES.items():
    _v.flags.writeable = False


# ---------------------------------------------------------------------------
# Chemistry cost (stub)
# ---------------------------------------------------------------------------


def _chemistry_cost(
    donor_subclass: str,
    site_idx: int,
    geometry_label: str,
) -> float:
    """
    Chemistry mismatch penalty for assigning donor_subclass to site_idx
    in the given coordination geometry.


    Parameters
    ----------
    donor_subclass : canonical donor subclass string from donor_hierarchy
    site_idx       : index into IDEAL_GEOMETRIES[geometry_label]
    geometry_label : coordination geometry string

    Returns
    -------
    cost : float ≥ 0; higher = worse assignment

    """
    return 0.0


# ---------------------------------------------------------------------------
# Hungarian assignment
# ---------------------------------------------------------------------------


def hungarian_assign(
    donor_directions: np.ndarray,
    ideal_directions: np.ndarray,
    donor_subclasses: Optional[List[str]] = None,
    geometry_label: Optional[str] = None,
) -> np.ndarray:
    """
    Assign donors to coordination sites via the Hungarian algorithm.


    Parameters
    ----------
    donor_directions : (n_donors, 3) unit vectors pointing from the metal
                       toward each donor in the initial/reference geometry.
                       Need not be unit — normalised internally.
    ideal_directions : (n_sites, 3) unit vectors from IDEAL_GEOMETRIES.
                       n_sites must equal n_donors (one-to-one assignment).
    donor_subclasses : optional list of n_donors subclass strings; passed
                       to _chemistry_cost.  If None, chemistry cost is skipped.
    geometry_label   : coordination geometry label; passed to _chemistry_cost.

    Returns
    -------
    assignment : (n_donors,) int array — assignment[i] = j means donor i is
                 assigned to site j.

    """
    n_donors = donor_directions.shape[0]
    n_sites = ideal_directions.shape[0]

    # Normalise donor directions
    norms = np.linalg.norm(donor_directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    d_hat = donor_directions / norms  # (n_donors, 3)

    # Angular cost: arccos of dot product between donor and site unit vectors
    dots = d_hat @ ideal_directions.T  # (n_donors, n_sites)
    dots = np.clip(dots, -1.0, 1.0)
    C_ang = np.arccos(dots)  # (n_donors, n_sites)

    # Chemistry cost (currently zero)
    C_chem = np.zeros((n_donors, n_sites), dtype=float)
    if donor_subclasses is not None and geometry_label is not None:
        for i, sub in enumerate(donor_subclasses):
            for j in range(n_sites):
                C_chem[i, j] = _chemistry_cost(sub, j, geometry_label)

    C = C_ang + C_chem  # (n_donors, n_sites)

    row_ind, col_ind = linear_sum_assignment(C)
    # row_ind will be [0, 1, ..., n_donors-1] for a square matrix;
    # col_ind[i] is the site assigned to donor i
    assignment = np.empty(n_donors, dtype=int)
    assignment[row_ind] = col_ind

    return assignment
