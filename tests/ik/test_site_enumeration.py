"""
Unit tests for ik/site_enumeration.py.

Phase 1 (site-symmetry automorphism groups) -- deliberately NOT "assert
against memorized chemistry-textbook point-group orders":
`site_symmetry_group()`'s own docstring proves that Gram-matrix
(pairwise-angle) invariance is both necessary and sufficient for a
permutation to correspond to an actual orthogonal transform of the
polyhedron, so every test below verifies that definition directly, two
independent ways, rather than trusting a hand-recalled table:

  1. `test_matches_naive_bruteforce_for_every_geometry` -- a deliberately
     dumb, unoptimised itertools.permutations scan (no signature pruning,
     shares no code with the module under test) must find EXACTLY the
     same permutation set the module returns, for every geometry label,
     including the largest (tricap_pris, CN9, 9!=362880 permutations --
     slow-ish (~3s) but a real independent oracle beats a fast fake one).
  2. `test_every_automorphism_is_geometrically_realizable` -- for every
     permutation returned, reconstructs the orthogonal matrix Q via SVD
     (no proper-rotation restriction, since reflections are valid
     automorphisms here -- see module docstring) and checks Q is
     orthogonal AND Q actually carries the ideal directions onto the
     permuted ones. This is the "does a real 3D symmetry operation exist"
     check the docstring's necessity/sufficiency argument claims always
     succeeds, verified computationally rather than only by proof.

Two special-case geometries get an exact expected order asserted from
pure logic (not chemistry recall, not the module's own output): trigonal
(3 sites, all mutually 120 deg apart -- the angle matrix is literally
constant off the diagonal, so it is invariant under EVERY permutation of
3 elements by construction, forcing |group| == 3! == 6 with no geometric
argument needed at all) and linear (2 sites, a single 180 deg angle,
trivially symmetric, forcing |group| == 2! == 2).

The group-closure test (`test_is_actually_a_group`) is a pure algebra
check independent of chemistry: identity present, closed under
composition, closed under inverse.

A note on the feasibility writeup's own guess ("octahedral should recover
a rotation group of order 24, tetrahedral order 12"): those are the
PROPER-rotation-only point-group orders (O, T). This module finds every
permutation realizable by the polyhedron's FULL point group (proper +
improper -- reflections preserve pairwise angles too), and for
octahedral/tetrahedral that representation is faithful, so the true
values are 48 (Oh) and 24 (Td, isomorphic to S4) -- see
`test_full_point_group_orders_not_rotation_only_subgroup` below, which
pins down these two values explicitly so the discrepancy from the
writeup's guess is caught by CI rather than silently reintroduced later.

Phase 2 (backtracking bijection enumerator + two-stage dedup) -- see
ik/site_enumeration.py's module docstring, "Findings from standalone
validation against EDTA" section, for the full story behind two of these
tests: (1) REACHABILITY_BREAKPOINTS reused unchanged from
assignment_search.py can make a real, well-known ligand's correct
topology geometrically UNREACHABLE (n_raw=0), not merely under-explored
-- `test_edta_as_is_reachability_bounds_give_zero_candidates` locks in
that this is still true, so a change to those shared bounds doesn't
silently alter behaviour here without notice; (2) sequential
`dedup_by_donor_equivalence` -> `dedup_by_site_equivalence` is NOT
sufficient for correctness -- `test_dedup_combined_catches_what_
sequential_dedup_misses` reproduces the exact counterexample
`dedup_combined`'s docstring describes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from jacobik.ik.site_enumeration import (
    DEFAULT_TOL_DEG,
    SiteEnumerationConfig,
    dedup_by_donor_equivalence,
    dedup_by_site_equivalence,
    dedup_combined,
    enumerate_assignments,
    enumerate_unique_assignments,
    site_symmetry_group,
    site_symmetry_group_order,
)
from jacobik.ik.targets import IDEAL_GEOMETRIES


def _naive_automorphisms(angles_deg: np.ndarray, tol_deg: float) -> set:
    """Independent, deliberately unoptimised oracle: literally try every
    permutation, no pruning, no shared code with site_enumeration.py."""
    n = angles_deg.shape[0]
    found = set()
    for perm in itertools.permutations(range(n)):
        p = np.array(perm)
        if np.max(np.abs(angles_deg[np.ix_(p, p)] - angles_deg)) <= tol_deg:
            found.add(perm)
    return found


def _orthogonal_procrustes(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best orthogonal (proper OR improper) Q with mobile @ Q ~= target.
    Deliberately does NOT force det(Q) = +1 (unlike
    jacobik.benchmark.geometry.kabsch) -- reflections are valid site
    automorphisms here, see module docstring."""
    u, _, vt = np.linalg.svd(mobile.T @ target)
    return u @ vt


@pytest.mark.parametrize("geometry_label", sorted(IDEAL_GEOMETRIES.keys()))
def test_matches_naive_bruteforce_for_every_geometry(geometry_label):
    directions = IDEAL_GEOMETRIES[geometry_label]
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1.0, 1.0)))
    expected = _naive_automorphisms(angles, DEFAULT_TOL_DEG)
    actual = set(site_symmetry_group(geometry_label))
    assert actual == expected


@pytest.mark.parametrize("geometry_label", sorted(IDEAL_GEOMETRIES.keys()))
def test_every_automorphism_is_geometrically_realizable(geometry_label):
    directions = IDEAL_GEOMETRIES[geometry_label]
    for perm in site_symmetry_group(geometry_label):
        permuted = directions[list(perm)]
        q = _orthogonal_procrustes(directions, permuted)
        assert q @ q.T == pytest.approx(np.eye(3), abs=1e-8)
        assert directions @ q == pytest.approx(permuted, abs=1e-6)


@pytest.mark.parametrize("geometry_label", sorted(IDEAL_GEOMETRIES.keys()))
def test_is_actually_a_group(geometry_label):
    n = IDEAL_GEOMETRIES[geometry_label].shape[0]
    group = site_symmetry_group(geometry_label)
    group_set = set(group)
    identity = tuple(range(n))
    assert identity in group_set

    def compose(a, b):
        return tuple(a[b[i]] for i in range(n))

    def inverse(a):
        inv = [0] * n
        for i, v in enumerate(a):
            inv[v] = i
        return tuple(inv)

    for a in group:
        assert inverse(a) in group_set
        for b in group:
            assert compose(a, b) in group_set


def test_trigonal_group_is_full_symmetric_group_by_construction():
    """No geometric argument needed: all three trigonal-planar pairwise
    angles are exactly 120 deg, so the angle matrix is literally constant
    off the diagonal and therefore invariant under EVERY permutation of
    3 elements -- |group| must equal 3! = 6."""
    assert site_symmetry_group_order("trigonal") == 6
    assert set(site_symmetry_group("trigonal")) == set(itertools.permutations(range(3)))


def test_linear_group_is_both_permutations_of_two_sites():
    """A single pairwise angle (180 deg) is trivially invariant under
    both permutations of 2 elements -- |group| must equal 2! = 2."""
    assert site_symmetry_group_order("linear") == 2
    assert set(site_symmetry_group("linear")) == {(0, 1), (1, 0)}


def test_full_point_group_orders_not_rotation_only_subgroup():
    """Pins down the two values the feasibility writeup guessed wrong
    (it expected the proper-rotation-only O/T orders, 24/12) so a future
    change that accidentally restricts this module to proper rotations
    is caught here rather than silently shipped. See module docstring
    for why the full (proper + improper) point group is the correct
    target: reflections preserve pairwise angles exactly as rotations
    do, and both octahedral's and tetrahedral's permutation
    representations are faithful (no non-identity point-group element
    fixes every vertex), so the realized permutation group equals the
    full point group: Oh (order 48) and Td (order 24, isomorphic to
    S4) respectively."""
    assert site_symmetry_group_order("octahedral") == 48
    assert site_symmetry_group_order("tetrahedral") == 24


def test_unknown_geometry_label_raises_keyerror():
    with pytest.raises(KeyError):
        site_symmetry_group("not_a_real_geometry")


def test_result_is_cached_between_calls():
    first = site_symmetry_group("octahedral")
    second = site_symmetry_group("octahedral")
    assert first is second


def test_all_geometry_labels_produce_a_nonempty_group():
    for geometry_label in IDEAL_GEOMETRIES:
        group = site_symmetry_group(geometry_label)
        assert len(group) >= 1
        n = IDEAL_GEOMETRIES[geometry_label].shape[0]
        assert tuple(range(n)) in group


# ===========================================================================
# Phase 2: backtracking bijection enumerator + two-stage dedup
# ===========================================================================


def test_enumerate_assignments_unconstrained_gives_all_bijections():
    """No hop_matrix at all -> every n_sites! bijection is reachability-
    valid (there's nothing to violate) -- tetrahedral (4! = 24) chosen
    over a larger geometry so this stays fast."""
    result = enumerate_assignments(4, "tetrahedral", hop_matrix=None)
    assert not result.truncated
    assert len(result.assignments) == 24
    keys = {tuple(int(x) for x in a) for a in result.assignments}
    assert len(keys) == 24  # all distinct


def test_enumerate_assignments_prunes_by_reachability():
    """A donor pair with hop=3 (octahedral) can only be cis (90 deg) --
    REACHABILITY_BREAKPOINTS' hop<=4 bound is [55, 115] deg, and
    octahedral pairs are only ever 90 or 180. Exhaustively verifies every
    returned assignment actually respects this, and that the count
    matches the exact combinatorial prediction: of the 6*5=30 ordered
    site pairs donor 0 and donor 1 could land on, exactly 12*2=24 (the
    12 unordered 90-deg site pairs, each orderable 2 ways) are cis, so
    720 * 24/30 = 576 of the 720 total bijections should survive."""
    hop = np.full((6, 6), -1, dtype=int)
    hop[0, 1] = hop[1, 0] = 3
    result = enumerate_assignments(6, "octahedral", hop_matrix=hop)
    assert not result.truncated
    assert len(result.assignments) == 576
    directions = IDEAL_GEOMETRIES["octahedral"]
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1.0, 1.0)))
    for a in result.assignments:
        assert angles[a[0], a[1]] == pytest.approx(90.0, abs=1e-6)


def test_enumerate_assignments_rejects_mismatched_n_donors():
    with pytest.raises(ValueError):
        enumerate_assignments(5, "octahedral")  # octahedral has 6 sites


def test_enumerate_assignments_rejects_mismatched_hop_matrix_shape():
    with pytest.raises(ValueError):
        enumerate_assignments(6, "octahedral", hop_matrix=np.zeros((5, 5), dtype=int))


def test_enumerate_assignments_unknown_geometry_raises_keyerror():
    with pytest.raises(KeyError):
        enumerate_assignments(6, "not_a_real_geometry")


def test_enumerate_assignments_respects_max_candidates_cap():
    """A tiny cap on a large, weakly-constrained search space (octahedral,
    no hop_matrix -> 720 possible) should stop early and report
    truncated=True, with exactly max_candidates assignments collected --
    the cap is checked at every node, not just on completing a search, so
    it should never overshoot."""
    cfg = SiteEnumerationConfig(max_candidates=10, time_budget_s=None)
    result = enumerate_assignments(6, "octahedral", hop_matrix=None, config=cfg)
    assert result.truncated
    assert len(result.assignments) == 10


def test_enumerate_assignments_time_budget_can_truncate():
    """An effectively-zero time budget should truncate almost
    immediately, regardless of max_candidates."""
    cfg = SiteEnumerationConfig(max_candidates=None, time_budget_s=0.0)
    result = enumerate_assignments(6, "octahedral", hop_matrix=None, config=cfg)
    assert result.truncated


def test_dedup_by_donor_equivalence_collapses_interchangeable_groups():
    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array([1, 0, 2, 3, 4, 5])  # donors {0,1} swapped sites -- same class under [[0,1]]
    c = np.array([0, 1, 2, 3, 5, 4])  # donors {4,5} swapped -- different class
    deduped = dedup_by_donor_equivalence([a, b, c], [[0, 1]])
    assert len(deduped) == 2
    assert deduped[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert deduped[1].tolist() == [0, 1, 2, 3, 5, 4]


def test_dedup_by_donor_equivalence_ignores_singleton_groups():
    """A "group" of size < 2 has nothing to relabel -- should be a no-op,
    not an error."""
    a = np.array([0, 1, 2])
    b = np.array([1, 0, 2])
    deduped = dedup_by_donor_equivalence([a, b], [[0]])
    assert len(deduped) == 2
    assert deduped[0].tolist() == a.tolist()
    assert deduped[1].tolist() == b.tolist()


def test_dedup_by_site_equivalence_collapses_automorphism_related_assignments():
    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array(
        [0, 1, 2, 3, 5, 4]
    )  # = sigma(a) for sigma = swap(4, 5), a real octahedral automorphism
    deduped = dedup_by_site_equivalence([a, b], "octahedral")
    assert len(deduped) == 1
    assert deduped[0].tolist() == [0, 1, 2, 3, 4, 5]


def test_dedup_combined_catches_what_sequential_dedup_misses():
    """Regression test for the exact counterexample dedup_combined's own
    docstring describes (derived from the EDTA validation case, donor
    order [N1, N2, O1, O1b, O2, O2b], octahedral,
    donor_equivalence_groups=[[N1,N2],[O1,O1b,O2,O2b]]):
    assignment A and assignment B are NOT related by donor-equivalence
    relabeling alone, and NOT related by site symmetry alone, but ARE the
    same physical structure once both moves are combined. Sequential
    dedup must therefore report 2 survivors; dedup_combined must report
    1."""
    groups = [[0, 1], [2, 3, 4, 5]]
    a = np.array([0, 2, 3, 4, 1, 5])  # N1=0, N2=2, O1=3, O1b=4, O2=1, O2b=5
    b = np.array([0, 4, 2, 5, 1, 3])  # N1=0, N2=4, O1=2, O1b=5, O2=1, O2b=3

    # Sanity: confirm the premise -- neither individual dedup merges them.
    assert len(dedup_by_donor_equivalence([a, b], groups)) == 2
    assert len(dedup_by_site_equivalence([a, b], "octahedral")) == 2
    after_donor = dedup_by_donor_equivalence([a, b], groups)
    assert len(dedup_by_site_equivalence(after_donor, "octahedral")) == 2

    # The fix: dedup_combined finds the combination that connects them.
    combined = dedup_combined([a, b], groups, "octahedral")
    assert len(combined.assignments) == 1
    assert not combined.truncated


def test_dedup_combined_pre_reduction_handles_homoleptic_cn8_quickly():
    """Regression test for the 2.1.0 performance fix: an unconstrained CN8
    case (square_anti) with all 8 donors in ONE donor-equivalence group is
    the synthetic worst case that motivated it -- before the fix,
    dedup_combined's pairwise scan had to try up to 8! = 40,320 donor
    relabelings per comparison (since `|donor relabeling group|`, not
    `|site symmetry group|` <=48, is what dominates the worst-case cost)
    and took ~64s on the default-capped 5,000-candidate raw list. The
    cheap dedup_by_donor_equivalence pre-reduction collapses all of those
    to a single donor-relabeling orbit in O(n) time before the expensive
    scan ever runs (see dedup_combined's "Implementation" docstring
    section), so this should now complete in well under a second."""
    import time as _time

    n = 8
    donor_groups = [list(range(n))]
    raw = enumerate_assignments(n, "square_anti", hop_matrix=None)
    assert len(raw.assignments) > 1000  # confirm this really is the large-input case

    t0 = _time.perf_counter()
    result = dedup_combined(raw.assignments, donor_groups, "square_anti")
    elapsed = _time.perf_counter() - t0

    assert len(result.assignments) == 1
    assert not result.truncated
    assert elapsed < 5.0, f"dedup_combined took {elapsed:.2f}s -- pre-reduction regression?"


def test_dedup_combined_time_budget_truncates_and_is_reported():
    """dedup_combined's own time_budget_s (mitigation 2 in its
    "Implementation" docstring section) must actually be honoured and
    reported, not just accepted and ignored -- this is the gap the 2.1.0
    fix closed (previously only enumerate_assignments had a budget; the
    dedup step had none at all, see the module docstring's "Combinatorial
    blowup / performance" section). Uses an already-negative budget so
    the very first budget check trips immediately, making this
    deterministic and fast regardless of real machine speed."""
    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array([0, 1, 2, 3, 5, 4])
    cfg = SiteEnumerationConfig(time_budget_s=-1.0)

    result = dedup_combined([a, b], [], "octahedral", config=cfg)
    assert result.truncated
    assert result.assignments == []  # cut off before the first candidate was even kept


def test_enumerate_unique_assignments_forwards_config_to_dedup_combined():
    """`enumerate_unique_assignments` must forward `config` to
    `dedup_combined` too, not just to `enumerate_assignments` -- this is
    the exact gap the 2.1.0 fix closed (see UniqueEnumerationResult's
    `dedup_truncated` field docstring). Both stages start their own fresh
    `time.monotonic()` timer from the same `config`, so an
    already-negative `time_budget_s` trips BOTH independently (this test
    isn't trying to isolate "only dedup truncates" -- that would need two
    different budgets, which the shared `config` parameter deliberately
    doesn't support -- just that the field exists and is wired through at
    all, which `dedup_truncated` defaulting to False everywhere before
    2.1.0 could never have caught)."""
    cfg = SiteEnumerationConfig(time_budget_s=-1.0)
    result = enumerate_unique_assignments(
        6, "octahedral", hop_matrix=None, donor_equivalence_groups=[[0, 1, 2, 3, 4, 5]], config=cfg
    )
    assert result.truncated  # raw enumerate_assignments also tripped the same budget
    assert result.n_raw == 0
    # dedup_combined short-circuits on empty input before ever checking its own
    # budget (nothing to truncate), so dedup_truncated correctly stays False here;
    # the mechanism itself is covered directly (non-empty input) by
    # test_dedup_combined_time_budget_truncates_and_is_reported above.
    assert not result.dedup_truncated


def _edta_hop_matrix(cross_arm_hop: int) -> np.ndarray:
    """EDTA's heavy-atom hop distances, donor order
    [N1, N2, O1, O1b, O2, O2b] -- see the module docstring / notebook for
    the full derivation from EDTA's real connectivity (2 five-membered
    glycinate chelate rings per nitrogen + 1 ethylenediamine backbone
    ring = 5 rings total, each donor pair 3 hops apart; same-nitrogen
    O-O pairs and N-to-cross-arm-O pairs 6 hops apart; cross-arm O-O
    pairs 9 hops apart). `cross_arm_hop` overrides only the 4 true
    N-to-cross-arm-O pairs (see the two tests below -- the real 6-hop
    distance there makes the geometry unsatisfiable under
    REACHABILITY_BREAKPOINTS as-is; 11 is the accurate stand-in used to
    demonstrate the enumerator+dedup are correct given an accurate
    reachability model)."""
    n = 6
    N1, N2, O1, O1b, O2, O2b = 0, 1, 2, 3, 4, 5
    hop = np.full((n, n), -1, dtype=int)

    def set_hop(i, j, h):
        hop[i, j] = h
        hop[j, i] = h

    set_hop(N1, N2, 3)
    set_hop(N1, O1, 3)
    set_hop(N1, O1b, 3)
    set_hop(N2, O2, 3)
    set_hop(N2, O2b, 3)
    set_hop(O1, O1b, 6)
    set_hop(O2, O2b, 6)
    set_hop(N1, O2, cross_arm_hop)
    set_hop(N1, O2b, cross_arm_hop)
    set_hop(N2, O1, cross_arm_hop)
    set_hop(N2, O1b, cross_arm_hop)
    set_hop(O1, O2, 9)
    set_hop(O1, O2b, 9)
    set_hop(O1b, O2, 9)
    set_hop(O1b, O2b, 9)
    return hop


def test_fe_h2o_6_collapses_to_exactly_one_candidate():
    """Six independent water ligands (no covalent bond between different
    waters -> hop_matrix is entirely "no known path") that are all
    chemically identical (one donor-equivalence group of all six). This
    is the feasibility writeup's phase-2 acceptance target for Fe(H2O)6:
    despite 720 raw reachability-valid bijections (nothing constrains
    them -- there's no known hop distance between any pair), they are
    all physically the same structure and must collapse to exactly 1."""
    hop = np.full((6, 6), -1, dtype=int)
    result = enumerate_unique_assignments(
        6, "octahedral", hop, donor_equivalence_groups=[[0, 1, 2, 3, 4, 5]]
    )
    assert result.n_raw == 720
    assert not result.truncated
    assert len(result.assignments) == 1


def test_edta_as_is_reachability_bounds_give_zero_candidates():
    """Regression-locks finding 1 from the module docstring's "Findings
    from standalone validation" section: with EDTA's real heavy-atom hop
    distances and assignment_search.py's REACHABILITY_BREAKPOINTS reused
    completely unchanged, donor N1 has no permitted trans partner at all
    (every pair within 6 hops of it is barred from 180 deg, and
    octahedral pairs are only ever 90 or 180), so the search space is
    provably empty. This is a real limitation of reusing those bounds
    unchanged for PROACTIVE enumeration (see the module docstring) -- not
    an enumerator bug, and this test exists so a future change to either
    the bounds or this module doesn't silently alter that fact without
    someone noticing."""
    hop = _edta_hop_matrix(cross_arm_hop=6)  # as-is: cross pairs at their real 6-hop distance
    result = enumerate_unique_assignments(
        6, "octahedral", hop, donor_equivalence_groups=[[0, 1], [2, 3, 4, 5]]
    )
    assert result.n_raw == 0
    assert len(result.assignments) == 0


def test_edta_with_accurate_reachability_collapses_to_exactly_one_candidate():
    """Same EDTA topology as above, but the 4 true N-to-cross-arm-O pairs
    are given a hop value (11) that maps to the fully-unconstrained
    (>10 -> 180 deg) REACHABILITY_BREAKPOINTS bucket instead of the
    misleadingly-tight generic 6-hop bucket (see finding 1 -- hop COUNT
    alone can't tell "6 bonds through a rigid ring" (O1-O1b, still
    correctly cis-restricted here) apart from "6 bonds through this
    specific molecule's genuinely flexible cross-arm tether" (N1-O2 and
    friends)). This is the feasibility writeup's phase-2 acceptance
    target for EDTA: collapses to exactly 1 unique candidate, and that
    candidate matches the well-documented real M-EDTA octahedral
    structure (both N cis to each other, each N trans to a cross-arm O,
    the two leftover O's trans to each other)."""
    hop = _edta_hop_matrix(cross_arm_hop=11)
    result = enumerate_unique_assignments(
        6, "octahedral", hop, donor_equivalence_groups=[[0, 1], [2, 3, 4, 5]]
    )
    assert result.n_raw == 192
    assert not result.truncated
    assert len(result.assignments) == 1

    directions = IDEAL_GEOMETRIES["octahedral"]
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1.0, 1.0)))
    a = result.assignments[0]
    N1, N2, O1, O1b, O2, O2b = 0, 1, 2, 3, 4, 5
    assert angles[a[N1], a[N2]] == pytest.approx(90.0, abs=1e-6)  # N,N cis
    # exactly one of {O2, O2b} is trans to N1, exactly one of {O1, O1b} is trans to N2
    n1_trans = [
        d for d in (O1, O1b, O2, O2b) if angles[a[N1], a[d]] == pytest.approx(180.0, abs=1e-6)
    ]
    n2_trans = [
        d for d in (O1, O1b, O2, O2b) if angles[a[N2], a[d]] == pytest.approx(180.0, abs=1e-6)
    ]
    assert n1_trans == [O2] or n1_trans == [O2b]
    assert n2_trans == [O1] or n2_trans == [O1b]
