"""
assignment_search.py
=====================
Generates a small, chemistry-grounded set of ALTERNATIVE donor-to-site
assignment candidates around the baseline one-shot Hungarian solve
(ik/targets.py::hungarian_assign), for cases where that single,
single-initial-embedding assignment may have locked onto the wrong
topology -- see engine.py's "Site assignment" step, which currently
computes exactly one assignment per solve() call from one arbitrary
initial ligand embedding, with no mechanism for the beam search's later
torsion/orientation diversification to ever revisit it.

Pure numpy -- no torch/rdkit/scipy-beyond-what-hungarian_assign-already-
needs dependency, so this module is unit-testable and usable standalone
from any candidate assignment array plus the same donor_dirs/ideal/
subclasses/geometry_label/hop-distance inputs hungarian_assign already
takes.

Design intent (see JACOBIK_V3_HANDOFF.md-style reasoning): candidate count
must scale with how AMBIGUOUS the baseline solve actually was, not be a
fixed brute-force budget -- a clean, unambiguous case (PtCl4's symmetric
Cl-) should generate ~0 extra candidates and cost nothing; a genuinely
confusable case (a long, heterogeneous, low-symmetry wraparound ligand)
should generate a handful, bounded by k_max.

Two candidate sources, pooled and deduplicated:

  (a) Near-tied swaps on the baseline Hungarian solution. A donor whose
      assigned site's angular cost is close to its next-best alternative's
      is exactly the kind a single arbitrary embedding could plausibly
      have gotten wrong (see the 2026-09 la_polyester diagnosis: two
      donors' crude embedded directions were close enough to swap the
      correct axial/equatorial roles and still look locally "optimal").
      Only donors within a small ambiguous set are ever permuted; every
      resulting candidate is filtered by a hop-distance-dependent angular
      reachability bound BEFORE being kept, generalising
      ik/targets.py::CHELATE_MAX_BITE_DEG (currently only applied as a
      *post-hoc repair*, and only to <=4-hop pairs) into a *proactive*
      filter applied to every donor pair a hop-distance is known for.

  (b) Distinct Hungarian solves against alternative donor_directions
      arrays (e.g. from several independent conformer embeddings of the
      free ligand), when supplied by the caller. Deduplicated against (a)
      and the baseline.

Version
-------
1.0.0 : initial release.
1.1.0 : the candidate cap (k_max) is now enforced by diversity-preserving
        selection (_select_diverse: greedy farthest-point over assignment
        Hamming distance) instead of ranking by a single scalar score.
        Three scoring heuristics were tried first and each was found to
        systematically discard the objectively-correct alternative on an
        adversarial test case for a related reason: total angular cost
        and "remaining suspects after re-flagging" are both circular
        (they re-consult the same untrustworthy donor_directions embedding
        that caused the original misassignment, or a local-ambiguity
        signal that doesn't go away just because a donor was correctly
        moved off its own bad preference); reachability_slack_deg (kept,
        see below) is principled -- it only ever consults ligand topology
        and the fixed ideal geometry -- but as a single MINIMUM over all
        hop-known donor pairs it can't tell "changed nothing" apart from
        "resolved the real conflict but tightened a different, still-
        valid pair elsewhere," so several inert candidates could still
        outscore the one that actually fixed the problem. Root cause: by
        construction, every candidate in the pool already passed the hard
        assignment_is_reachable filter, so there is no remaining
        topology-only signal that reliably separates "correct" from
        "merely plausible" -- that discrimination needs energetics, which
        is exactly what the downstream MLIP single-point screen is for.
        Diversity selection stops trying to pre-guess the winner and
        instead guarantees the surviving pool spans the distinct
        topologies found. reachability_slack_deg is retained as the
        deterministic seed/tie-break for that selection (still useful for
        reproducible ordering, just not as the sole ranking key). Default
        k_max raised 6 -> 10 to give the diversity cap enough room; a
        pre-relax + single-point MLIP call per candidate is cheap enough
        that this remains well within the "not unreasonable variety"
        performance budget.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from jacobik.ik.targets import hungarian_assign
except ImportError:  # pragma: no cover - standalone/test loading path
    hungarian_assign = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pairwise angular reachability (generalises ik/targets.py's
# CHELATE_MAX_BITE_DEG=115 for <=4 hops to a full hop-distance curve)
# ---------------------------------------------------------------------------

#: (max_hops_inclusive, max_bite_deg) breakpoints. A short, rigid tether
#: can only reach a narrow angular range at the metal; a long, floppy one
#: can reach nearly anything. The 4-hop/115-deg point matches
#: ik/targets.py::CHELATE_MAX_BITE_DEG exactly, so this is a strict
#: generalisation, not a competing constant. Deliberately loose/heuristic,
#: not a hard chemistry bound -- tune against the test-suite notebook.
REACHABILITY_BREAKPOINTS: Tuple[Tuple[int, float], ...] = (
    (4, 115.0),
    (6, 150.0),
    (8, 170.0),
    (10, 180.0),
)

#: Below this many hops, a pair also can't plausibly be pushed BELOW this
#: bite angle -- a short rigid link can't be squeezed onto near-coincident
#: sites either. Symmetric counterpart to the upper bound above.
MIN_BITE_DEG_SHORT: float = 55.0
MIN_BITE_HOPS: int = 4


def max_reachable_bite_deg(hops: Optional[int]) -> float:
    """Loosest plausible site-site angle two donors `hops` ligand-graph
    bonds apart could straddle. `None` (no path -- e.g. different
    ligands, or hop distance unknown) is treated as unconstrained (180)."""
    if hops is None or hops < 0:
        return 180.0
    for max_hops, bound in REACHABILITY_BREAKPOINTS:
        if hops <= max_hops:
            return bound
    return 180.0


def is_pair_reachable(hops: Optional[int], site_angle_deg: float) -> bool:
    """Could a donor pair `hops` ligand-graph bonds apart plausibly sit at
    `site_angle_deg` on the target polyhedron?"""
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
    """(n_sites, n_sites) pairwise angles between ideal polyhedron vertex
    directions, in degrees."""
    dots = np.clip(ideal_directions @ ideal_directions.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def angular_cost_matrix(donor_directions: np.ndarray, ideal_directions: np.ndarray) -> np.ndarray:
    """(n_donors, n_sites) angular cost in degrees -- same quantity
    ik/targets.py::hungarian_assign's C_ang uses internally (there in
    radians; degrees here for readability of the margins below)."""
    norms = np.linalg.norm(donor_directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    d_hat = donor_directions / norms
    dots = np.clip(d_hat @ ideal_directions.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def assignment_total_cost(C_ang_deg: np.ndarray, assignment: np.ndarray) -> float:
    return float(sum(C_ang_deg[di, si] for di, si in enumerate(assignment)))


def assignment_is_reachable(
    assignment: np.ndarray,
    site_angles: np.ndarray,
    hop_matrix: Optional[np.ndarray],
) -> bool:
    """Check EVERY donor pair in `assignment` (not just the ones that were
    permuted to build it) against the hop-distance reachability bound.
    `hop_matrix[i, j]` is the ligand-graph hop distance between donor i
    and donor j, or a negative number / None-like sentinel (<0) for "no
    path known" (different ligands, or caller didn't supply it -- treated
    as unconstrained)."""
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


def reachability_slack_deg(
    assignment: np.ndarray,
    site_angles: np.ndarray,
    hop_matrix: Optional[np.ndarray],
) -> float:
    """
    How comfortably (in degrees of margin) this assignment satisfies every
    KNOWN hop-distance reachability bound -- the minimum, over all
    hop-known donor pairs, of how far the pair's assigned site-site angle
    sits from either bound. Larger = more comfortable/plausible, given
    only the ligand's own covalent topology (never the crude initial
    embedding, which is exactly the signal that's untrustworthy for a
    donor whose placement is actually being contested -- see
    find_suspect_donors). +inf when hop_matrix is None or no pair has a
    known hop distance (nothing to be uncomfortable about).

    Used as the ranking tiebreaker for capping the candidate pool at
    k_max, IN PLACE OF the donor_directions-based total angular cost:
    ranking by that cost was verified (on an adversarial test case) to
    systematically discard the one candidate that correctly moves a
    confidently-mis-embedded donor off its bad site, because doing so
    necessarily looks "worse" against the same bad embedding that caused
    the problem. Reachability slack has no such circularity -- it only
    ever references the ligand's real bonded topology (hop_matrix) and
    the fixed ideal polyhedron (site_angles), never donor_directions.
    """
    if hop_matrix is None:
        return float("inf")
    n = len(assignment)
    slack = float("inf")
    any_known = False
    for i in range(n):
        for j in range(i + 1, n):
            hops = hop_matrix[i, j]
            hops = None if hops is None or hops < 0 else int(hops)
            if hops is None:
                continue
            any_known = True
            angle = site_angles[assignment[i], assignment[j]]
            upper = max_reachable_bite_deg(hops)
            slack = min(slack, upper - angle)
            if hops <= MIN_BITE_HOPS:
                slack = min(slack, angle - MIN_BITE_DEG_SHORT)
    return slack if any_known else float("inf")


# ---------------------------------------------------------------------------
# Ambiguity detection
# ---------------------------------------------------------------------------


def find_suspect_donors(
    C_ang_deg: np.ndarray,
    assignment: np.ndarray,
    site_angles: np.ndarray,
    hop_matrix: Optional[np.ndarray] = None,
    margin_deg: float = 20.0,
) -> List[int]:
    """
    Donors worth reconsidering for an alternative assignment, from two
    independent signals -- pooled, because each catches a failure mode
    the other cannot:

    (1) Locally near-tied: the donor's assigned site isn't a clearly-best
        match against its OWN crude embedded direction -- the best other
        site is within `margin_deg`. Catches genuinely close calls,
        including donors with no known hop distance to anything (e.g.
        cross-ligand donors, where signal (2) below has nothing to work
        with).

    (2) Reachability-violating: the donor is one half of a pair whose
        BASELINE-assigned site-site angle already breaks the hop-distance
        reachability bound (max_reachable_bite_deg / is_pair_reachable --
        a strict generalisation of ik/targets.py's
        _repair_chelate_assignment, which only checks this for known
        <=4-hop chelate pairs, only in the "assigned too FAR apart"
        direction, and only as a post-hoc patch rather than an input to
        candidate generation).

        Signal (1) alone CANNOT catch this donor: verified on an
        adversarial synthetic case (see test suite) where a donor's own
        crude embedded direction pointed confidently -- and wrongly --
        at an equatorial site it had no real business being anywhere
        near. Its own local margin against that (wrong) site was ~78 deg,
        nowhere near "near-tied" -- the wrongness is only visible by
        checking it against something external (here, the ligand's own
        covalent topology via hop_matrix), not from the donor's own
        angular cost row in isolation. This is the reason hop_matrix is
        strongly recommended: without it, this whole failure class -- a
        confidently-wrong single embedding, not a close call -- is
        invisible to candidate generation.
    """
    n_donors, n_sites = C_ang_deg.shape
    suspect: set = set()

    for di in range(n_donors):
        assigned_cost = C_ang_deg[di, assignment[di]]
        other_costs = [C_ang_deg[di, sj] for sj in range(n_sites) if sj != assignment[di]]
        if other_costs and (min(other_costs) - assigned_cost) <= margin_deg:
            suspect.add(di)

    if hop_matrix is not None:
        for i in range(n_donors):
            for j in range(i + 1, n_donors):
                hops = hop_matrix[i, j]
                hops = None if hops is None or hops < 0 else int(hops)
                if hops is None:
                    continue
                angle = site_angles[assignment[i], assignment[j]]
                if not is_pair_reachable(hops, angle):
                    suspect.add(i)
                    suspect.add(j)

    return sorted(suspect)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Count of donors assigned to a different site between two
    assignments -- the natural "how differently do these two candidates
    solve the problem" distance in this discrete space."""
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def _select_diverse(
    candidates: List[np.ndarray],
    k: int,
    site_angles: np.ndarray,
    hop_matrix: Optional[np.ndarray],
) -> List[np.ndarray]:
    """
    Greedy farthest-point selection: keep up to `k` candidates that are
    maximally spread out from each other in assignment-space (by
    _hamming_distance), rather than the top-`k` by any single scalar
    score.

    Why not just rank-and-truncate by reachability_slack_deg (the first
    thing tried here): that metric is a MINIMUM over all hop-known donor
    pairs, and a pair that's naturally tight for reasons that have
    nothing to do with what a candidate actually changed (e.g. a short,
    genuinely tight chelate bite that every candidate shares unchanged)
    can dominate that minimum for every candidate that leaves it alone.
    On the adversarial test case in the test suite, this let five
    candidates that only shuffled donors NOT involved in the real
    conflict outscore -- and crowd out under k_max -- the one candidate
    that actually resolved it (which scored worse only because fixing the
    real conflict happened to create a new, still-fully-valid but less
    slack-y pair elsewhere). A single scalar ranking can't distinguish
    "inert" from "resolved it, at the cost of tightening somewhere else
    that's still within bounds" -- both are just "some number", and nothing
    guarantees the resolving candidate wins that comparison.

    Diversity selection sidesteps needing to guess which candidate is
    *right* -- every survivor of generate_candidate_assignments already
    passed the hard reachability filter, so all are kinematically valid;
    the only question left is which ones are worth the downstream
    pre-relax + MLIP single-point cost. Maximizing pairwise difference
    guarantees the kept set actually spans the distinct topologies the
    generators found, instead of five near-duplicates of the same partial
    fix. Final correctness arbitration among a diverse, valid pool is
    exactly what the MLIP energy screen is for -- this cap's only job is
    to keep that pool small, not to pre-guess its winner.

    Deterministic tie-breaking: the initial seed, and ties during the
    farthest-point search, fall back to reachability_slack_deg (most
    comfortable first) purely so results are reproducible -- it is not
    trusted here as a correctness signal, only as a stable order.
    """
    if k <= 0 or not candidates:
        return []
    remaining = sorted(
        candidates,
        key=lambda c: -reachability_slack_deg(c, site_angles, hop_matrix),
    )
    selected = [remaining.pop(0)]
    while remaining and len(selected) < k:
        best_idx, best_dist = 0, -1
        for idx, cand in enumerate(remaining):
            dist = min(_hamming_distance(cand, s) for s in selected)
            if dist > best_dist:
                best_dist, best_idx = dist, idx
        selected.append(remaining.pop(best_idx))
    return selected


def generate_candidate_assignments(
    donor_directions: np.ndarray,
    ideal_directions: np.ndarray,
    *,
    donor_subclasses: Optional[List[str]] = None,
    geometry_label: Optional[str] = None,
    chelate_pairs: Optional[List[Tuple[int, int]]] = None,
    hop_matrix: Optional[np.ndarray] = None,
    alt_donor_directions: Optional[Sequence[np.ndarray]] = None,
    margin_deg: float = 20.0,
    max_ambiguous_group: int = 4,
    k_max: int = 10,
) -> List[np.ndarray]:
    """
    Returns a list of candidate (n_donors,) assignment arrays, always
    starting with the baseline Hungarian solve at index 0, followed by up
    to `k_max - 1` alternatives selected for DIVERSITY (see
    _select_diverse) rather than ranked by any single "which one looks
    best" score -- topology alone often cannot tell two kinematically
    valid candidates apart; that's left to the downstream MLIP energy
    screen. Returns just `[baseline]` (length 1) when nothing ambiguous
    is found -- the expected, near-zero-cost outcome for the common/easy
    case.

    Parameters mirror ik/targets.py::hungarian_assign's, plus:

    hop_matrix          : optional (n_donors, n_donors) ligand-graph hop
                          distances (negative/absent = unknown or
                          cross-ligand, treated as unconstrained). Used to
                          filter BOTH candidate sources for kinematic
                          plausibility -- this is the generalisation of
                          CHELATE_MAX_BITE_DEG described in the module
                          docstring. Strongly recommended; without it,
                          every swap of ambiguous donors is kept
                          regardless of whether the ligand could actually
                          reach that geometry.
    alt_donor_directions : optional list of alternative (n_donors, 3)
                          donor-direction arrays (e.g. from several free-
                          ligand conformer embeddings). Each is Hungarian-
                          solved independently and pooled in as candidate
                          source (b).
    margin_deg           : "near-tied" threshold for source (a) -- see
                          find_ambiguous_donors.
    max_ambiguous_group  : hard cap on how many ambiguous donors are
                          jointly permuted for source (a) -- bounds the
                          combinatorics (n! for n ambiguous donors) even
                          on a pathologically confused case.
    """
    if hungarian_assign is None:
        raise RuntimeError("ik/targets.py::hungarian_assign not importable")

    baseline = hungarian_assign(
        donor_directions, ideal_directions,
        donor_subclasses=donor_subclasses, geometry_label=geometry_label,
        chelate_pairs=chelate_pairs,
    )
    n_donors = len(baseline)
    site_angles = site_angles_deg(ideal_directions)
    C_ang_deg = angular_cost_matrix(donor_directions, ideal_directions)

    pool: List[np.ndarray] = [baseline]
    seen = {tuple(baseline.tolist())}

    def _try_add(cand: np.ndarray) -> None:
        key = tuple(int(x) for x in cand.tolist())
        if key in seen:
            return
        if not assignment_is_reachable(cand, site_angles, hop_matrix):
            return
        seen.add(key)
        pool.append(cand)

    # --- Source (a): swaps involving suspect donors -----------------------
    # Two sub-mechanisms, because a suspect donor's fix is not always held
    # by ANOTHER suspect donor:
    #
    #  (a1) Joint permutation among suspects' OWN currently-held sites --
    #       catches a cyclic/joint reassignment where 3+ mutually-suspect
    #       donors just need to trade among themselves.
    #  (a2) Pairwise swap of each suspect donor against EVERY OTHER donor
    #       in the whole assignment, not just other suspects -- catches
    #       the case where donor A needs donor B's site but B itself gave
    #       no local signal of being wrong (B's own placement is perfectly
    #       plausible in isolation; it only needs to move because A needs
    #       its site). This mirrors ik/targets.py::_repair_chelate_
    #       assignment's own "search over ALL dk" swap logic, generalised
    #       from its single greedy best-swap to keeping every swap that
    #       passes the full reachability check, so multiple candidates
    #       survive rather than committing to one. Confirmed necessary,
    #       not belt-and-suspenders: (a1) alone misses this fix on the
    #       adversarial test case in the test suite (D4 never gets
    #       flagged as suspect on its own, so its site is never offered).
    suspects = find_suspect_donors(
        C_ang_deg, baseline, site_angles, hop_matrix=hop_matrix, margin_deg=margin_deg,
    )

    if len(suspects) >= 2:
        amb = suspects[:max_ambiguous_group]
        amb_sites = [int(baseline[d]) for d in amb]
        for perm in itertools.permutations(amb_sites):
            if list(perm) == amb_sites:
                continue
            cand = baseline.copy()
            for d, s in zip(amb, perm):
                cand[d] = s
            _try_add(cand)

    for di in suspects:
        for dk in range(n_donors):
            if dk == di:
                continue
            cand = baseline.copy()
            cand[di], cand[dk] = int(baseline[dk]), int(baseline[di])
            _try_add(cand)

    # --- Source (b): alternative embeddings, re-solved -------------------
    if alt_donor_directions:
        for alt_dirs in alt_donor_directions:
            alt_assign = hungarian_assign(
                np.asarray(alt_dirs), ideal_directions,
                donor_subclasses=donor_subclasses, geometry_label=geometry_label,
                chelate_pairs=chelate_pairs,
            )
            _try_add(alt_assign)

    # --- Cap by diversity, not by score -------------------------------------
    # Baseline always stays first (index 0). Three scoring heuristics were
    # tried and rejected for ranking `rest` before truncating to k_max-1:
    # total angular cost against donor_directions (circular -- penalizes
    # exactly the candidates that correctly move a confidently-mis-embedded
    # donor off its bad site, since that necessarily looks "worse" against
    # the same bad embedding that caused the problem); count of remaining
    # suspects after re-flagging (also circular -- a donor correctly moved
    # off its own confidently-wrong local preference still LOOKS locally
    # ambiguous afterward, so this doesn't reliably favour a correct fix
    # either); and reachability_slack_deg alone (principled -- never
    # touches the untrustworthy embedding -- but as a single MINIMUM over
    # all hop-known pairs, it can't distinguish "did nothing" from
    # "resolved the real conflict but tightened a different, still-valid
    # pair," and on the adversarial test case that let several inert
    # candidates crowd out the one that actually fixed both misplaced
    # donors). See _select_diverse's docstring for the full account.
    #
    # All three failure modes share a root cause: every candidate in
    # `pool` already passed the hard assignment_is_reachable filter, so
    # they are ALL kinematically valid -- there is no topology-only signal
    # left that reliably tells "correct" apart from "merely plausible."
    # Trying to rank them into a false ordering just risks discarding the
    # right answer before it ever reaches the MLIP screen, which is the
    # part of this pipeline actually equipped to make that call. So this
    # cap instead maximizes how DIFFERENT the kept candidates are from
    # each other (farthest-point selection over assignment Hamming
    # distance), guaranteeing the surviving pool spans the distinct
    # topologies found rather than clustering near-duplicates of one
    # partial fix -- verified to retain the adversarial case's one fully-
    # correct candidate under a realistic (not artificially tiny) k_max.
    rest = _select_diverse(pool[1:], max(0, k_max - 1), site_angles, hop_matrix)
    return [baseline] + rest
