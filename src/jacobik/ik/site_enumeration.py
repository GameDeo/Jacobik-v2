"""Proactive enumeration of donor-to-site assignments for a fixed geometry.

Enumerates every donor-to-site bijection that satisfies the hop-distance
reachability bound (jacobik.ik.reachability) by backtracking, then removes
duplicates: two assignments are the same structure if some combination of a
permutation of chemically equivalent donors and a symmetry of the ideal
polyhedron (site-symmetry group, reflections included) maps one onto the other.
Both the raw search and the dedup pass are bounded by SiteEnumerationConfig.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from jacobik.ik.reachability import is_pair_reachable, site_angles_deg
from jacobik.ik.targets import IDEAL_GEOMETRIES

#: Tolerance (degrees) for treating two site-site angles as equal.
DEFAULT_TOL_DEG: float = 1e-3

_AUTOMORPHISM_CACHE: Dict[Tuple[str, float], List[Tuple[int, ...]]] = {}


def _angle_signature(angles_row: np.ndarray, self_idx: int, tol_deg: float) -> Tuple[int, ...]:
    """Permutation-invariant fingerprint of a site: its sorted, quantised angles to all others."""
    others = np.delete(angles_row, self_idx)
    quantised = np.round(others / tol_deg).astype(np.int64)
    return tuple(sorted(quantised.tolist()))


def _find_automorphisms(angles_deg: np.ndarray, tol_deg: float) -> List[Tuple[int, ...]]:
    """All permutations sigma with angles[sigma[i], sigma[j]] == angles[i, j], by
    backtracking; signatures only prune candidates, the pairwise check decides."""
    n = angles_deg.shape[0]
    signatures = [_angle_signature(angles_deg[i], i, tol_deg) for i in range(n)]
    candidates_by_slot = [[j for j in range(n) if signatures[j] == signatures[i]] for i in range(n)]

    results: List[Tuple[int, ...]] = []
    perm = [-1] * n
    used = [False] * n

    def consistent(slot: int, candidate: int) -> bool:
        for filled_slot in range(slot):
            if (
                abs(angles_deg[slot, filled_slot] - angles_deg[candidate, perm[filled_slot]])
                > tol_deg
            ):
                return False
        return True

    def backtrack(slot: int) -> None:
        if slot == n:
            results.append(tuple(perm))
            return
        for candidate in candidates_by_slot[slot]:
            if used[candidate] or not consistent(slot, candidate):
                continue
            perm[slot] = candidate
            used[candidate] = True
            backtrack(slot + 1)
            used[candidate] = False
            perm[slot] = -1

    backtrack(0)
    return results


def site_symmetry_group(
    geometry_label: str, tol_deg: float = DEFAULT_TOL_DEG
) -> List[Tuple[int, ...]]:
    """Site permutations preserving IDEAL_GEOMETRIES[geometry_label] (proper and
    improper), identity first. Cached per (label, tol_deg)."""
    if geometry_label not in IDEAL_GEOMETRIES:
        raise KeyError(
            f"Unknown geometry label {geometry_label!r}; known labels: {sorted(IDEAL_GEOMETRIES)}"
        )
    cache_key = (geometry_label, tol_deg)
    if cache_key not in _AUTOMORPHISM_CACHE:
        directions = IDEAL_GEOMETRIES[geometry_label]
        automorphisms = _find_automorphisms(site_angles_deg(directions), tol_deg)
        identity = tuple(range(directions.shape[0]))
        automorphisms.sort(key=lambda p: p != identity)
        _AUTOMORPHISM_CACHE[cache_key] = automorphisms
    return _AUTOMORPHISM_CACHE[cache_key]


def proper_site_symmetry_group(
    geometry_label: str, tol_deg: float = DEFAULT_TOL_DEG
) -> List[Tuple[int, ...]]:
    """The site permutations of site_symmetry_group that a proper rotation realises (identity first).

    A reflection of the ideal polyhedron maps an assignment onto its mirror image. That is the same structure only
    if the ligands' embedded conformations can reach their own mirror images by torsions, which a fixed 3D embed
    need not allow (e.g. two identical arms on a tetrahedral B or C: their order around the centre is fixed).
    For a planar or linear site set every permutation is realised by a rotation (a reflection through the plane
    fixes every site), so the whole group is returned."""
    key = ("proper", geometry_label, tol_deg)
    if key not in _AUTOMORPHISM_CACHE:
        D = np.asarray(IDEAL_GEOMETRIES[geometry_label], dtype=float)
        full = site_symmetry_group(geometry_label, tol_deg)
        if np.linalg.matrix_rank(D, tol=1e-6) < 3:
            out = list(full)
        else:
            out = []
            for perm in full:
                U, _, Vt = np.linalg.svd(D.T @ D[list(perm)])
                if np.linalg.det(U @ Vt) > 0:
                    out.append(perm)
        _AUTOMORPHISM_CACHE[key] = out
    return _AUTOMORPHISM_CACHE[key]


@dataclass
class SiteEnumerationConfig:
    """Budgets for enumeration.

    max_candidates : cap on complete reachable assignments collected (None = no cap).
    time_budget_s  : wall-clock cap in seconds for each of the search and the dedup
        pass (None = no cap).
    """

    max_candidates: Optional[int] = 50_000
    time_budget_s: Optional[float] = 30.0


@dataclass
class EnumerationResult:
    """Raw (not yet deduplicated) reachable bijections; `truncated` if a budget stopped the search."""

    assignments: List[np.ndarray] = field(default_factory=list)
    truncated: bool = False


def _hop(hop_matrix: Optional[np.ndarray], i: int, j: int) -> Optional[int]:
    """Hop distance between donors i and j, or None if unknown (matrix None or entry < 0)."""
    if hop_matrix is None:
        return None
    hops = hop_matrix[i, j]
    return None if hops is None or hops < 0 else int(hops)


def _next_donor(
    remaining: Sequence[int], placed: Sequence[int], hop_matrix: Optional[np.ndarray]
) -> int:
    """Most-constrained-first: the remaining donor with the smallest known hop
    distance to a placed donor (to any other donor when none is placed yet).
    Ties and unconstrained donors fall back to lowest index."""
    if hop_matrix is None:
        return min(remaining)

    reference = placed if placed else [d for d in range(hop_matrix.shape[0]) if d not in remaining]
    if not reference:
        reference = list(range(hop_matrix.shape[0]))

    def key(donor: int) -> Tuple[bool, int, int]:
        best = None
        for other in reference:
            if other == donor:
                continue
            h = _hop(hop_matrix, donor, other)
            if h is not None and (best is None or h < best):
                best = h
        return (best is None, best if best is not None else 0, donor)

    return min(remaining, key=key)


def enumerate_assignments(
    n_donors: int,
    geometry_label: str,
    hop_matrix: Optional[np.ndarray] = None,
    *,
    config: Optional[SiteEnumerationConfig] = None,
) -> EnumerationResult:
    """Backtrack over all donor-to-site bijections (n_donors must equal the
    geometry's n_sites), pruning any partial placement whose newly placed donor
    violates is_pair_reachable against an already placed donor. hop_matrix[i, j]
    is the ligand-graph hop distance, negative for unknown; None means
    unconstrained."""
    if geometry_label not in IDEAL_GEOMETRIES:
        raise KeyError(
            f"Unknown geometry label {geometry_label!r}; known labels: {sorted(IDEAL_GEOMETRIES)}"
        )
    ideal_directions = IDEAL_GEOMETRIES[geometry_label]
    n_sites = ideal_directions.shape[0]
    if n_donors != n_sites:
        raise ValueError(
            f"n_donors ({n_donors}) must equal geometry {geometry_label!r}'s "
            f"n_sites ({n_sites}) -- enumerate_assignments only produces bijections."
        )
    if hop_matrix is not None and hop_matrix.shape != (n_donors, n_donors):
        raise ValueError(f"hop_matrix shape {hop_matrix.shape} != ({n_donors}, {n_donors})")

    cfg = config or SiteEnumerationConfig()
    site_angles = site_angles_deg(ideal_directions)

    assignments: List[np.ndarray] = []
    truncated = False
    start = time.perf_counter()

    def budget_exceeded() -> bool:
        if cfg.max_candidates is not None and len(assignments) >= cfg.max_candidates:
            return True
        if cfg.time_budget_s is not None and (time.perf_counter() - start) > cfg.time_budget_s:
            return True
        return False

    def consistent(donor: int, site: int, placement: Dict[int, int]) -> bool:
        for other_donor, other_site in placement.items():
            h = _hop(hop_matrix, donor, other_donor)
            if h is None:
                continue
            if not is_pair_reachable(h, site_angles[site, other_site]):
                return False
        return True

    def backtrack(placement: Dict[int, int], used_sites: set, remaining: List[int]) -> None:
        nonlocal truncated
        if budget_exceeded():
            truncated = True
            return
        if not remaining:
            assignments.append(np.array([placement[d] for d in range(n_donors)], dtype=int))
            return
        donor = _next_donor(remaining, list(placement.keys()), hop_matrix)
        new_remaining = [d for d in remaining if d != donor]
        for site in range(n_sites):
            if site in used_sites or not consistent(donor, site, placement):
                continue
            placement[donor] = site
            used_sites.add(site)
            backtrack(placement, used_sites, new_remaining)
            del placement[donor]
            used_sites.discard(site)
            if budget_exceeded():
                truncated = True
                return

    backtrack({}, set(), list(range(n_donors)))
    return EnumerationResult(assignments=assignments, truncated=truncated)


def dedup_by_donor_equivalence(
    assignments: Sequence[np.ndarray],
    donor_equivalence_groups: Sequence[Sequence[int]],
) -> List[np.ndarray]:
    """Keep the first assignment of each class under permutation of donors within
    each equivalence group (groups smaller than 2 are ignored)."""
    groups = [list(g) for g in donor_equivalence_groups if len(g) >= 2]
    if not groups:
        return list(assignments)
    grouped_donors = {d for g in groups for d in g}
    ungrouped = [d for d in range(len(assignments[0])) if d not in grouped_donors] if assignments else []

    # Canonical signature: sorted sites per group plus exact site of ungrouped donors.
    seen: set = set()
    kept: List[np.ndarray] = []
    for cand in assignments:
        key = tuple(tuple(sorted(int(cand[d]) for d in g)) for g in groups) + tuple(
            int(cand[d]) for d in ungrouped
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(cand)
    return kept


def _donor_relabeling_group(
    n_donors: int, donor_equivalence_groups: Sequence[Sequence[int]]
) -> List[Tuple[int, ...]]:
    """All donor-index permutations obtained by permuting each equivalence group's
    members among themselves (identity included)."""
    groups = [list(g) for g in donor_equivalence_groups if len(g) >= 2]
    if not groups:
        return [tuple(range(n_donors))]
    per_group_perms = [list(itertools.permutations(g)) for g in groups]
    relabelings = []
    for combo in itertools.product(*per_group_perms):
        relabel = list(range(n_donors))
        for g, perm in zip(groups, combo):
            for src, dst in zip(g, perm):
                relabel[src] = dst
        relabelings.append(tuple(relabel))
    return relabelings


@dataclass
class DedupResult:
    """Deduplicated assignments; `truncated` if time_budget_s cut the scan short."""

    assignments: List[np.ndarray] = field(default_factory=list)
    truncated: bool = False


def dedup_combined(
    assignments: Sequence[np.ndarray],
    donor_equivalence_groups: Sequence[Sequence[int]],
    geometry_label: str,
    *,
    tol_deg: float = DEFAULT_TOL_DEG,
    config: Optional[SiteEnumerationConfig] = None,
) -> DedupResult:
    """Keep the first assignment of each orbit under the combined action of donor
    relabelings and site symmetries. The two must be applied together: deduping
    by one and then the other can miss pairs related only by a combination of
    both. A cheap donor-equivalence pass first shrinks the input before the
    pairwise scan (the relabeling group can be as large as n_donors!). The scan
    honours config.time_budget_s."""
    if not assignments:
        return DedupResult(assignments=[], truncated=False)

    pre_reduced = dedup_by_donor_equivalence(assignments, donor_equivalence_groups)

    n_donors = len(pre_reduced[0])
    relabelings = _donor_relabeling_group(n_donors, donor_equivalence_groups)
    site_group = site_symmetry_group(geometry_label, tol_deg=tol_deg)

    relabelings_arr = np.asarray(relabelings)  # (R, n_donors)
    site_group_arr = np.asarray(site_group)  # (S, n_donors)

    def all_transforms(a: np.ndarray) -> np.ndarray:
        relabeled = a[relabelings_arr]  # (R, n_donors)
        transformed = site_group_arr[:, relabeled]  # (S, R, n_donors)
        return transformed.reshape(-1, n_donors)  # (S * R, n_donors)

    # Pairwise scan, capped by time_budget_s.
    start = time.perf_counter()
    truncated = False
    kept: List[np.ndarray] = []
    kept_arr = np.empty((0, n_donors), dtype=np.asarray(pre_reduced[0]).dtype)
    for cand in pre_reduced:
        if (
            config is not None
            and config.time_budget_s is not None
            and (time.perf_counter() - start) > config.time_budget_s
        ):
            truncated = True
            break
        if kept_arr.shape[0] > 0:
            transforms = all_transforms(np.asarray(cand))  # (S * R, n_donors)
            is_dup = bool(
                (transforms[:, None, :] == kept_arr[None, :, :]).all(axis=-1).any()
            )
            if is_dup:
                continue
        kept.append(cand)
        kept_arr = np.vstack([kept_arr, np.asarray(cand)[None, :]])
    return DedupResult(assignments=kept, truncated=truncated)


def dedup_exact(
    assignments: Sequence[np.ndarray],
    instance_groups: Sequence[Sequence[Sequence[int]]],
    geometry_label: str,
    *,
    tol_deg: float = DEFAULT_TOL_DEG,
    config: Optional[SiteEnumerationConfig] = None,
) -> DedupResult:
    """Keep the first assignment of each orbit under the EXACT symmetries of a fixed embed: proper rotations of the
    polyhedron and swaps of whole identical ligand instances.

    instance_groups: one entry per set of identical ligands (same SMILES, >= 2 instances); each entry lists the
    instances, each instance as its donor indices in a fixed per-ligand order. Swapping instances is handled by a
    canonical key (the instances' site tuples, sorted), so it costs O(n) per assignment for any number of identical
    ligands (no permutation enumeration). Two assignments are the same iff some proper site rotation maps one onto
    the other's canonical key."""
    if not assignments:
        return DedupResult(assignments=[], truncated=False)
    n = len(assignments[0])
    grouped = {d for g in instance_groups for inst in g for d in inst}
    free = [d for d in range(n) if d not in grouped]
    groups = [[list(inst) for inst in g] for g in instance_groups]

    def key(a) -> tuple:
        return tuple(tuple(sorted(tuple(int(a[d]) for d in inst) for inst in g)) for g in groups) + tuple(
            int(a[d]) for d in free)

    rot = np.asarray(proper_site_symmetry_group(geometry_label, tol_deg=tol_deg))  # (S, n_sites)
    start = time.perf_counter()
    seen: set = set()
    kept: List[np.ndarray] = []
    truncated = False
    for cand in assignments:
        k0 = key(cand)
        if k0 in seen:
            continue
        if config is not None and config.time_budget_s is not None and (time.perf_counter() - start) > config.time_budget_s:
            truncated = True
            break
        imgs = rot[:, np.asarray(cand)]  # (S, n): site g(a[d]) for every rotation g
        if any(key(im) in seen for im in imgs[1:]):
            seen.add(k0)
            continue
        seen.add(k0)
        kept.append(cand)
    return DedupResult(assignments=kept, truncated=truncated)


def topological_class_keys(
    assignments: Sequence[np.ndarray],
    donor_equivalence_groups: Sequence[Sequence[int]],
    geometry_label: str,
    tol_deg: float = DEFAULT_TOL_DEG,
) -> List[tuple]:
    """Class key of each assignment under the LEGACY (graph-level) equivalence: permutations of topologically
    equivalent donors x all site symmetries, reflections included. Assignments with the same key are the same
    structure up to the embed's fixed arrangement; dedup_exact keeps them apart so that the one the embed can reach
    is present, and the solver shares one row budget among them. O(|site group|) per assignment."""
    groups = [sorted(g) for g in donor_equivalence_groups if len(g) >= 2]
    n = len(assignments[0]) if len(assignments) else 0
    grouped = {d for g in groups for d in g}
    free = [d for d in range(n) if d not in grouped]
    sym = np.asarray(site_symmetry_group(geometry_label, tol_deg=tol_deg))
    keys = []
    for a in assignments:
        imgs = sym[:, np.asarray(a)]
        keys.append(min(tuple(tuple(sorted(int(im[d]) for d in g)) for g in groups) + tuple(int(im[d]) for d in free)
                        for im in imgs))
    return keys


@dataclass
class UniqueEnumerationResult:
    """Result of enumerate_unique_assignments.

    assignments     : symmetry-unique reachable (n_donors,) bijections.
    truncated       : the raw search hit a SiteEnumerationConfig budget, so the set may be incomplete.
    dedup_truncated : the dedup pass hit time_budget_s before checking every raw candidate.
    n_raw           : number of reachable bijections found before dedup.
    """

    assignments: List[np.ndarray] = field(default_factory=list)
    truncated: bool = False
    dedup_truncated: bool = False
    n_raw: int = 0


def enumerate_unique_assignments(
    n_donors: int,
    geometry_label: str,
    hop_matrix: Optional[np.ndarray] = None,
    *,
    donor_equivalence_groups: Sequence[Sequence[int]] = (),
    config: Optional[SiteEnumerationConfig] = None,
    instance_groups: Optional[Sequence[Sequence[Sequence[int]]]] = None,
) -> UniqueEnumerationResult:
    """enumerate_assignments followed by a dedup pass; `config` applies to both.

    instance_groups=None: legacy dedup_combined (donor-equivalence relabelings x all site symmetries, reflections
    included; can merge assignments that a fixed embed cannot interconvert). Otherwise dedup_exact."""
    raw = enumerate_assignments(n_donors, geometry_label, hop_matrix, config=config)
    if instance_groups is None:
        deduped = dedup_combined(raw.assignments, donor_equivalence_groups, geometry_label, config=config)
    else:
        deduped = dedup_exact(raw.assignments, instance_groups, geometry_label, config=config)
    return UniqueEnumerationResult(
        assignments=deduped.assignments,
        truncated=raw.truncated,
        dedup_truncated=deduped.truncated,
        n_raw=len(raw.assignments),
    )
