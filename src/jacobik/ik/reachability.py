"""Hop-distance reachability of donor-to-site assignments.

Two donors ``h`` ligand-graph bonds apart can only straddle a limited range of
site-site angles at the metal. These helpers encode that bound, check whole
assignments against it, and repair a baseline assignment that violates it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

#: (max_hops_inclusive, max_bite_deg). Two donors h bonds apart close a chelate
#: ring of h+2 members including the metal: rings up to 6 members (h <= 4) span at
#: most ~115 deg, a 7-membered ring (h = 5) ~150 deg. From h >= 6 a trans
#: arrangement is possible (larger ring, or a pair linked through another chelate
#: ring), so no upper bound applies.
REACHABILITY_BREAKPOINTS: Tuple[Tuple[int, float], ...] = (
    (4, 115.0),
    (5, 150.0),
)

#: Pairs at most MIN_BITE_HOPS apart also cannot be squeezed below this angle.
MIN_BITE_DEG_SHORT: float = 55.0
MIN_BITE_HOPS: int = 4


def max_reachable_bite_deg(hops: Optional[int]) -> float:
    """Largest site-site angle two donors `hops` bonds apart can straddle;
    None or negative (unknown / different ligands) is unconstrained (180)."""
    if hops is None or hops < 0:
        return 180.0
    for max_hops, bound in REACHABILITY_BREAKPOINTS:
        if hops <= max_hops:
            return bound
    return 180.0


def is_pair_reachable(hops: Optional[int], site_angle_deg: float) -> bool:
    """Whether donors `hops` bonds apart can sit at `site_angle_deg`."""
    if hops is None or hops < 0:
        return True
    if site_angle_deg > max_reachable_bite_deg(hops) + 1e-6:
        return False
    if hops <= MIN_BITE_HOPS and site_angle_deg < MIN_BITE_DEG_SHORT - 1e-6:
        return False
    return True


# ---------------------------------------------------------------------------
# Cost / geometry helpers
# ---------------------------------------------------------------------------


def site_angles_deg(ideal_directions: np.ndarray) -> np.ndarray:
    """(n_sites, n_sites) pairwise angles between vertex directions, degrees."""
    dots = np.clip(ideal_directions @ ideal_directions.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def angular_cost_matrix(donor_directions: np.ndarray, ideal_directions: np.ndarray) -> np.ndarray:
    """(n_donors, n_sites) angle in degrees between each donor direction and each site."""
    norms = np.linalg.norm(donor_directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    d_hat = donor_directions / norms
    dots = np.clip(d_hat @ ideal_directions.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def assignment_is_reachable(
    assignment: np.ndarray,
    site_angles: np.ndarray,
    hop_matrix: Optional[np.ndarray],
) -> bool:
    """Check every donor pair against the hop-distance bound. Negative
    hop_matrix entries (or hop_matrix None) mean unconstrained."""
    if hop_matrix is None:
        return True
    n = len(assignment)
    for i in range(n):
        for j in range(i + 1, n):
            hops = hop_matrix[i, j]
            hops = None if hops is None or hops < 0 else int(hops)
            if hops is None:
                continue
            angle = site_angles[assignment[i], assignment[j]]
            if not is_pair_reachable(hops, angle):
                return False
    return True


def repair_unreachable_assignment(
    assignment: np.ndarray,
    C_ang_deg: np.ndarray,
    site_angles: np.ndarray,
    hop_matrix: Optional[np.ndarray],
    node_budget: int = 200_000,
) -> Tuple[np.ndarray, bool]:
    """Repair a baseline assignment that violates pairwise reachability.

    Returns (assignment, repaired). A reachable assignment is returned
    unchanged. Otherwise a depth-first branch-and-bound finds the reachable
    complete assignment with the lowest total angular cost `C_ang_deg`; if none
    exists within `node_budget` nodes the original is returned (repaired=False).
    """
    if hop_matrix is None or assignment_is_reachable(assignment, site_angles, hop_matrix):
        return assignment, False

    n = len(assignment)
    # Place the most-constrained donors (most known hop partners) first.
    known = [(int((hop_matrix[i] >= 0).sum()), i) for i in range(n)]
    order = [i for _, i in sorted(known, key=lambda t: -t[0])]

    best_cost = float("inf")
    best: Optional[List[int]] = None
    cur = [-1] * n
    used = [False] * n
    nodes = 0

    def rec(depth: int, cost: float) -> None:
        nonlocal best_cost, best, nodes
        if nodes > node_budget or cost >= best_cost:
            return
        if depth == n:
            best_cost, best = cost, list(cur)
            return
        d = order[depth]
        for site in np.argsort(C_ang_deg[d]):
            site = int(site)
            if used[site]:
                continue
            ok = True
            for e in order[:depth]:
                h = int(hop_matrix[d, e])
                if h >= 0 and not is_pair_reachable(h, site_angles[site, cur[e]]):
                    ok = False
                    break
            if not ok:
                continue
            nodes += 1
            used[site] = True
            cur[d] = site
            rec(depth + 1, cost + float(C_ang_deg[d, site]))
            used[site] = False
            cur[d] = -1

    rec(0, 0.0)
    if best is None:
        return assignment, False
    return np.asarray(best, dtype=int), True
