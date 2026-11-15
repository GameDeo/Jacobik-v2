"""
targets.py
==========
Ideal coordination polyhedra, shape tolerances and donor-to-site assignment.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)

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

#: Shape-constraint tolerance per ideal polyhedron, in the units of the
#: solver's shape statistic (power-mean, tau = 2, of per-donor squared
#: unit-vector deviation from the best-fit rotated ideal polyhedron, with
#: the log(n)/tau soft-max offset removed so a perfect polyhedron is 0).
#:
#: Each value is the 90th percentile of this statistic over 11,642 non-haptic
#: single-metal tmQM complexes fit to the ideal polyhedron of their coordination
#: number (donor-to-site assignment optimised). Polyhedra with <100 samples
#: (t_shaped, tricap_pris) use the pooled p90 instead.
SHAPE_TOL_P90: Dict[str, float] = {
    "linear": 0.006,
    "square_plan": 0.017,
    "octahedral": 0.047,
    "square_pyr": 0.055,
    "trig_bipyr": 0.068,
    "trigonal": 0.070,
    "square_anti": 0.070,
    "tetrahedral": 0.083,
    "pent_bipyr": 0.100,
    "trig_pris": 0.120,
    "see_saw": 0.142,
}

#: Pooled p90 over all 13 polyhedra (11,642 structures); used for geometries
#: without their own entry (t_shaped, tricap_pris, 'other').
SHAPE_TOL_POOLED: float = 0.060


def shape_tolerance(geometry: Optional[str]) -> float:
    """Data-derived shape tolerance for `geometry` (see SHAPE_TOL_P90)."""
    return SHAPE_TOL_P90.get(geometry or "", SHAPE_TOL_POOLED)


#: Geometry label -> (CN, 3) read-only array of unit vectors from the metal toward
#: each ideal donor site.
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
# Two axial sites (±z) and two equatorial sites 120° apart.
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
# Two eclipsed equilateral triangles at z=±1/√2, r=1/√2 from the z-axis
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

# Tricapped trigonal prismatic  CN=9 -- ideal D3h geometry
# Two eclipsed equilateral triangles (the trigonal-prism "prism" sites) at
# polar angle arccos(sqrt(3/7)) from +/-z, plus three "cap" sites in the
# equatorial (z=0) plane, azimuthally staggered 60 deg from the prism
# triangles so each cap sits over the centre of a rectangular prism face.
# Polar angle fixed by making the prism's rectangular faces squares: cos(theta) = sqrt(3/7).
_TTP_COS = math.sqrt(3.0 / 7.0)  # ~0.6547 -> theta ~49.11 deg
_TTP_SIN = math.sqrt(4.0 / 7.0)  # ~0.7559
_TTP_PRISM_AZ = np.deg2rad([0.0, 120.0, 240.0])
_TTP_CAP_AZ = np.deg2rad([60.0, 180.0, 300.0])
IDEAL_GEOMETRIES["tricap_pris"] = np.vstack(
    [
        np.column_stack(
            [
                _TTP_SIN * np.cos(_TTP_PRISM_AZ),
                _TTP_SIN * np.sin(_TTP_PRISM_AZ),
                np.full(3, _TTP_COS),
            ]
        ),  # top prism triangle
        np.column_stack(
            [
                _TTP_SIN * np.cos(_TTP_PRISM_AZ),
                _TTP_SIN * np.sin(_TTP_PRISM_AZ),
                np.full(3, -_TTP_COS),
            ]
        ),  # bottom prism triangle (eclipsed with top, not staggered)
        np.column_stack(
            [
                np.cos(_TTP_CAP_AZ),
                np.sin(_TTP_CAP_AZ),
                np.zeros(3),
            ]
        ),  # equatorial caps
    ]
).astype(float)

# Freeze all arrays
for _k, _v in IDEAL_GEOMETRIES.items():
    _v.flags.writeable = False


# ---------------------------------------------------------------------------
# Chelate bite-angle repair
# ---------------------------------------------------------------------------
#
# A linear (Hungarian) cost cannot express "these two donors of a short chelate
# must land on angularly close sites", so a bounded greedy repair pass runs
# after the base solve: for each short-chelate donor pair whose assigned sites
# exceed a plausible bite angle, it looks for the best swap with another donor
# that brings the pair within range at minimal added angular cost.

#: Site-site angle (deg) above which a short-chelate pair is treated as
#: mis-assigned (real bite angles cluster ~70-95 deg); deliberately generous.
CHELATE_MAX_BITE_DEG: float = 115.0


def _repair_chelate_assignment(
    assignment: np.ndarray,
    ideal_directions: np.ndarray,
    C_ang: np.ndarray,
    chelate_pairs: List[Tuple[int, int]],
    max_bite_deg: float = CHELATE_MAX_BITE_DEG,
    max_passes: int = 4,
) -> np.ndarray:
    """
    Greedy post-hoc repair for chelate pairs assigned to implausibly
    far-apart sites.  Works on a copy of ``assignment`` and returns it.

    Parameters
    ----------
    assignment       : (n_donors,) int array from the base Hungarian solve;
                       assignment[i] = site index for donor i.
    ideal_directions : (n_sites, 3) unit vectors, IDEAL_GEOMETRIES[geometry].
    C_ang            : (n_donors, n_sites) angular cost matrix used for the
                       base solve (radians) -- reused so repair swaps are
                       judged on the same cost the base solve optimised.
    chelate_pairs    : list of (donor_i, donor_j) index pairs known (from
                       ligand connectivity) to be a short chelate link.
    max_bite_deg     : threshold above which a chelate pair's assigned
                       site-site angle is flagged as implausible.
    max_passes       : safety cap on repair iterations.

    Returns
    -------
    assignment : (n_donors,) int array, repaired in place where beneficial.
    """
    if not chelate_pairs:
        return assignment

    n_sites = ideal_directions.shape[0]
    if n_sites < 2:
        return assignment

    assignment = assignment.copy()

    # Precompute all pairwise site-site angles once (n_sites, n_sites), deg.
    dots = np.clip(ideal_directions @ ideal_directions.T, -1.0, 1.0)
    site_angles_deg = np.degrees(np.arccos(dots))

    for _pass in range(max_passes):
        made_swap = False
        for di, dj in chelate_pairs:
            si, sj = int(assignment[di]), int(assignment[dj])
            if si == sj:
                continue
            bite = site_angles_deg[si, sj]
            if bite <= max_bite_deg:
                continue

            # Search for the best swap of di's site with some other donor
            # dk's site (dk != di, dj) that brings (di, dj) within the
            # plausible bite range, minimising the resulting increase in
            # total angular cost among di, dj, dk.
            best_gain = None
            best_dk = None
            base_cost = C_ang[di, si] + C_ang[dj, sj]
            for dk in range(len(assignment)):
                if dk == di or dk == dj:
                    continue
                sk = int(assignment[dk])
                new_bite = site_angles_deg[sk, sj]
                if new_bite > max_bite_deg:
                    continue  # swap wouldn't actually fix the chelate
                new_cost = C_ang[di, sk] + C_ang[dk, si] + C_ang[dj, sj]
                old_cost = base_cost + C_ang[dk, sk]
                delta = new_cost - old_cost
                if best_gain is None or delta < best_gain:
                    best_gain = delta
                    best_dk = dk

            if best_dk is not None:
                sk = int(assignment[best_dk])
                log.debug(
                    "hungarian_assign: chelate repair swapped donor %d (site %d) "
                    "<-> donor %d (site %d) to fix bite angle %.1f deg "
                    "(pair with donor %d) -- cost delta %.4f rad",
                    di, si, best_dk, sk, bite, dj, best_gain,
                )
                assignment[di], assignment[best_dk] = sk, si
                made_swap = True

        if not made_swap:
            break

    return assignment


# ---------------------------------------------------------------------------
# Hungarian assignment
# ---------------------------------------------------------------------------


def hungarian_assign(
    donor_directions: np.ndarray,
    ideal_directions: np.ndarray,
    chelate_pairs: Optional[List[Tuple[int, int]]] = None,
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
    chelate_pairs    : optional list of (donor_i, donor_j) index pairs known,
                       from ligand connectivity, to be a short (<=4 bond)
                       chelate link.  When given, a greedy repair pass (see
                       _repair_chelate_assignment) fixes pairs assigned to
                       sites whose angle exceeds CHELATE_MAX_BITE_DEG.

    Returns
    -------
    assignment : (n_donors,) int array — assignment[i] = j means donor i is
                 assigned to site j.
    """
    n_donors = donor_directions.shape[0]

    # Normalise donor directions
    norms = np.linalg.norm(donor_directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    d_hat = donor_directions / norms  # (n_donors, 3)

    # Angular cost: arccos of dot product between donor and site unit vectors
    dots = d_hat @ ideal_directions.T  # (n_donors, n_sites)
    dots = np.clip(dots, -1.0, 1.0)
    C_ang = np.arccos(dots)  # (n_donors, n_sites)

    row_ind, col_ind = linear_sum_assignment(C_ang)
    # row_ind will be [0, 1, ..., n_donors-1] for a square matrix;
    # col_ind[i] is the site assigned to donor i
    assignment = np.empty(n_donors, dtype=int)
    assignment[row_ind] = col_ind

    if chelate_pairs:
        assignment = _repair_chelate_assignment(
            assignment, ideal_directions, C_ang, chelate_pairs
        )

    return assignment
