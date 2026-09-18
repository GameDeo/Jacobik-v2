"""
site_enumeration.py
====================
Architector-style proactive site enumeration (see JACOBIK_V3_HANDOFF-style
reasoning in the 2026-09 site-enumeration feasibility writeup): given a
fixed coordination geometry and a set of donors, enumerate ALL
geometrically distinct donor-to-site bijections up front, instead of
perturbing one already-built embedding's Hungarian assignment
(ik/targets.py::hungarian_assign, ik/assignment_search.py) after the fact.

Four pieces, matching the feasibility writeup's "Proposed architecture"
1/2/3/4 in full as of 3.0.0 (piece 4, engine.py wiring, lives in
engine.py itself, not in this module -- see below):

  1. `site_symmetry_group()` -- site-symmetry automorphism groups for each
     IDEAL_GEOMETRIES polyhedron (phase 1, 2026-09-17). Pure geometry, no
     donor set involved.

  2. `enumerate_assignments()` -- a backtracking bijection enumerator
     (phase 2, 2026-09-17): builds donor-to-site assignments incrementally
     (one donor at a time), pruning a partial assignment the instant any
     newly-placed donor pair violates `assignment_search.is_pair_reachable`
     against every donor already placed. Donors are placed
     most-constrained-first (smallest known hop distance to an
     already-placed donor) to maximise how much of the search tree gets
     pruned before it is ever built.

  3. `dedup_by_donor_equivalence()`, `dedup_by_site_equivalence()`, and
     `dedup_combined()` (phase 2, 2026-09-17) -- the two-stage dedup down
     to "truly unique" the writeup calls for: (a) donor-equivalence is a
     direct, unchanged port of engine.py's existing 2.14.0 "Fix A"
     relabeling-collapse logic (the block in `ComplexBuilder.solve()`'s
     "3a." step titled "drop candidates that are pure relabelings of an
     already-kept candidate among chemically-identical donors")
     generalised from "dedupe against one fixed baseline" to "dedupe a
     whole candidate list" -- same algorithm, same semantics, just taking
     the caller-computed donor equivalence classes (engine.py's
     `symmetric_groups`, from RDKit SMILES + CanonicalRankAtoms) as a
     plain `List[List[int]]` input instead of recomputing them, since
     this module has no RDKit/molecule dependency by design (see "Why no
     RDKit" below); (b) site-equivalence is new, and uses piece 1's
     `site_symmetry_group()`. Standalone validation against EDTA (see
     "Finding: sequential dedup is not enough" below) surfaced that
     running (a) then (b) as two independent passes -- the writeup's own
     "two-stage" framing -- is NOT sufficient for correctness: some
     duplicate pairs are only reachable by combining a donor relabeling
     WITH a site rotation, neither alone. `dedup_combined()` is the
     function that is actually correct end to end and is what
     `enumerate_unique_assignments()` uses; the two individual functions
     are kept as separate, independently-testable building blocks (and
     because the writeup's architecture describes them as two distinct
     pieces), not as a pipeline to chain yourself.

  4. `ComplexBuilder.solve(use_symmetry_enumeration=True)` (phase 3,
     2026-09-17, engine.py, NOT in this module) -- wires
     `enumerate_unique_assignments()` into a new "3a'." step, feeding its
     output into the SAME `_assignment_candidates` variable the existing
     `use_assignment_search` path (piece "3a.") populates, so every
     downstream mechanism (screen-then-resolve, `_solve_for_assignment`)
     is reused completely unchanged, per this module's own design intent
     above. Off by default, mutually exclusive with
     `use_assignment_search` (composing them -- enumerate proactively,
     then still perturb locally on top -- is a documented, not-yet-
     implemented follow-on; see `ComplexBuilder.solve()`'s
     `use_symmetry_enumeration` parameter docstring).

This module itself still does not import engine.py, ik/targets.py's
`hungarian_assign`, or anything RDKit/torch-dependent -- piece 4 is
engine.py depending on this module (computing donor-equivalence groups
and the hop matrix the same way `use_assignment_search` already does,
and passing them in as plain data), never the reverse.

Why no RDKit (donor equivalence classes are a caller input, not computed
here)
--------------------------------------------------------------------------
Donor-equivalence classes require RDKit's `CanonicalRankAtoms` over each
ligand's actual molecular graph (see engine.py's `symmetric_groups` block)
-- real molecule/RDKit machinery this module deliberately has none of, so
it stays pure numpy, standalone-importable, and unit-testable without a
full jacobik install (same policy as ik/assignment_search.py and this
module's own phase-1 half). `dedup_by_donor_equivalence()` therefore takes
the already-computed groups as a parameter; phase 4's engine.py wiring is
what will hand it the real `symmetric_groups` list.

What "leaves the polyhedron invariant" means here, precisely
--------------------------------------------------------------
`IDEAL_GEOMETRIES[label]` is an (n_sites, 3) array of unit vectors from
the metal toward each site. A permutation sigma of the n_sites indices is
counted as a site-symmetry automorphism iff there exists an orthogonal
transform Q of R^3 (proper OR improper -- a rotation or a
rotation-plus-reflection) such that Q @ directions[i] == directions[sigma(i)]
for every site i. Because all sites are unit vectors from a common origin,
this is exactly the condition that sigma preserves the pairwise-angle
(equivalently, Gram/dot-product) matrix: two ordered point sets with an
identical Gram matrix are always related by SOME orthogonal transform of
the ambient space (a standard linear-algebra fact, true regardless of
whether the points are full rank in 3D or confined to a plane/line), and
conversely any orthogonal transform trivially preserves all pairwise
angles. So checking Gram-matrix invariance is both necessary and
sufficient -- no separate "does an actual Q exist" search is needed, and
`tests/ik/test_site_enumeration.py` independently confirms this by
reconstructing Q via SVD for every permutation this module returns.

IMPORTANT correction vs. the feasibility writeup's offhand expected test
values: that doc guessed "octahedral should recover a rotation group of
order 24, tetrahedral order 12" -- those are the PROPER-rotation-only
point-group orders (O and T). Because reflections also preserve pairwise
angles (see above), the group this module actually computes is the site
permutations realizable by the FULL point group (proper + improper), and
for octahedral/tetrahedral that image is faithful (no non-identity point-
group element fixes every vertex), so the true order equals the full
point-group order: 48 for octahedral (Oh, isomorphic to the permutation
group realized), 24 for tetrahedral (Td, isomorphic to S4). Some other
geometries DO have point-group elements (e.g. sigma_h through a set of
in-plane sites) that fix every site, so their realized permutation group
is a proper quotient of the full point group -- e.g. square-planar's
realized group has order 8 (dihedral D4 on the 4 in-plane sites), not
D4h's order-16 point group, because sigma_h fixes all 4 in-plane sites
and acts as the identity permutation. Point-group order is therefore not
a safe oracle for this module's output in general; see the test file for
how each geometry's expected order is instead derived independently
(closure-of-group checks, brute-force cross-validation, and explicit
degenerate-matrix arguments for the fully-symmetric small cases) rather
than taken on memorized chemistry-table faith.

`dedup_by_site_equivalence()` uses this full reflection-inclusive group,
which means it WILL collapse two donor arrangements that are mirror
images of each other (enantiomers) into one. That is the right call when
the donor set itself has no fixed chirality to distinguish (e.g. six
identical water ligands), but is an open design question for a donor set
that is itself chiral (a chiral ligand can make its Lambda/Delta metal-
complex diastereomers genuinely, chemically different structures, not
just relabelings) -- not decided here; a caller who needs to keep
enantiomers distinct should use a proper-rotation-only subgroup of
`site_symmetry_group()`'s output instead (this module doesn't compute
that subgroup, but it is easy to derive by filtering to permutations
whose reconstructed Q -- see the test file's `_orthogonal_procrustes` --
has det(Q) > 0).

Findings from standalone validation against EDTA
--------------------------------------------------
The feasibility writeup's phase-2 acceptance target was "validated
standalone... against Fe(H2O)6 and EDTA -- both should collapse to
exactly 1 unique candidate, same as today." Fe(H2O)6 does, immediately.
EDTA did not, on the first two attempts, and both failures are real
findings worth keeping on record (full derivation and worked numbers:
script_notebooks/site_enumeration_experiments.ipynb):

1. Reused-unchanged `is_pair_reachable` can make the correct answer
   UNREACHABLE, not just under-explored. Modelling EDTA's real heavy-
   atom hop distances (donor order [N1, N2, O1, O1b, O2, O2b]; N1-N2 and
   each N-to-its-own-arm-O pair = 3 hops; N-to-cross-arm-O and same-N
   O-O pairs = 6 hops; cross-arm O-O pairs = 9 hops) gives `n_raw = 0`:
   every donor pair within 6 hops of N1 is barred from being trans
   (REACHABILITY_BREAKPOINTS' hop<=6 upper bound is 150 deg, and
   octahedral pairs are only ever 90 or 180 deg), so N1 has NO
   permitted trans partner at all among the other five donors -- yet
   every octahedral site has exactly one trans partner, so no full
   assignment can exist. Real M-EDTA octahedral complexes DO have one
   N trans to a cross-arm O (a real, crystallographically documented
   6-hop pair), so the reachability model is wrong here, not the
   chemistry. This is a real limitation of reusing
   REACHABILITY_BREAKPOINTS "essentially unchanged" as the writeup
   proposed: those bounds were tuned for ik/assignment_search.py's
   REACTIVE use (small perturbations around an already-good baseline,
   where an over-tight bound only means "fewer alternatives get
   considered" -- a safe direction to err), not for a PROACTIVE from-
   scratch search, where the same over-tight bound can eliminate the
   only physically correct topology outright. Root cause: hop COUNT
   alone can't distinguish "6 bonds through a rigid, already-committed
   ring" from "6 bonds through a genuinely flexible tether" -- EDTA has
   pairs at the same hop count (O1-O1b vs. N1-O2) with different real
   reach. Not fixed here (this module still calls `is_pair_reachable`
   completely unchanged, as designed -- see piece 2 above); flagged as
   a concrete, actionable prototyping risk for whoever tackles the
   writeup's phase 4 benchmarking, more fundamental than the CN8/CN9
   blowup risk that phase already expected to investigate.

2. Sequential two-stage dedup can under-collapse. See `dedup_combined`'s
   docstring for the mechanism and a worked EDTA counterexample; this IS
   fixed here (`enumerate_unique_assignments` uses `dedup_combined`, not
   a chain of the two individual dedup functions).

With finding 1 worked around for validation purposes only (giving the 4
true cross N-O pairs an unconstrained hop value instead of 6, leaving
same-N O-O pairs correctly still restricted -- see the notebook), and
finding 2 fixed, EDTA collapses to exactly 1 unique candidate, matching
the writeup's target.

Combinatorial blowup / performance
-----------------------------------
`enumerate_assignments()` takes a `SiteEnumerationConfig` (max_candidates,
time_budget_s) as a hard safety net, per the feasibility writeup's
explicitly flagged biggest risk: naive brute enumeration is n_sites!
before pruning (362,880 for CN9). Benchmarking this against real CN8/CN9
cases (feasibility writeup's phase 4) is still not done, but a
synthetic worst case WAS measured this round: an unconstrained (no
hop_matrix) CN8 case (square_anti, all 8 donors in one donor-equivalence
group) hits `max_candidates`'s default 5,000 cap in ~0.04s for the raw
enumeration, but the OLD `dedup_combined` (see "Findings from
standalone validation" below for what changed) then took ~64s on that
capped list -- 6x past the 10s `time_budget_s` default, because that
budget was, at the time, only ever passed to `enumerate_assignments`,
never to `dedup_combined` itself. Both the root cause (an
easily-avoidable O(n_raw x |donor relabeling group|) scan, see
`dedup_combined`'s "Implementation" section) and the missing budget are
fixed as of 2.1.0. `enumerate_assignments()` reports `truncated=True`,
and `dedup_combined()` / `enumerate_unique_assignments()` report
`dedup_truncated`/`DedupResult.truncated`, rather than any step silently
returning an incomplete-but-unlabeled list when a cap is hit. Tuning the
donor-ordering heuristic, the caps, and the time-boxes against a REAL
(not synthetic-worst-case) CN8/9 ligand is still explicitly follow-on
work (feasibility writeup's phase 4), not done here.

Comparison to Architector's own approach
-----------------------------------------
The published Architector tool (lanl/Architector on GitHub; Taylor et
al., "Architector for high-throughput cross-periodic table 3D complex
building", Nature Communications 2023,
https://www.nature.com/articles/s41467-023-38169-2) that motivated this
whole effort solves the analogous "which ligand goes on which site"
problem in `architector/io_symmetry.py` (`select_cons` /
`generate_good_combos`), and does NOT do what this module does. It
never computes an exact site-symmetry automorphism group or a donor-
relabeling group at all -- there is no permutation-group machinery in
that file. Instead it: (1) generates candidate site-combinations via a
similar backtracking recursion, but hard-slices the result to the first
10,000 combos found (`good_combos[0:10000]`), full stop, no adaptive
time-box; (2) "dedups" with a cheap PHYSICAL heuristic, not an exact
group check -- a pairwise pseudo-Coulomb-plus-steric score per combo
(charge x charge / r, atom-count x atom-count / r, summed over ligand
pairs), rounded to 2 decimal places, with combos sharing a rounded
score treated as equivalent (Python dict/set value collision, O(1) per
combo, versus this module's O(|combined group|) exact pairwise check);
(3) keeps only the top `n_symmetries` (default 10, capped at
`n_conformers` if larger) LOWEST-ENERGY survivors of that heuristic
dedup -- a small, fixed-size output regardless of how large the true
symmetry-unique count is, not the complete set this module aims for;
and (4) only AFTER actually building and relaxing 3D structures for
those <=10 survivors, runs a real RMSD-based structural duplicate check
(`io_align_mol.calc_rmsd`, <0.5 Angstrom = duplicate) as the final,
ground-truth dedup pass -- catching whatever the cheap energy heuristic
over- or under-collapsed, using real 3D geometry instead of symbolic
reasoning. In short, Architector trades this module's exactness (a
mathematically complete, provably-correct combined-orbit dedup) for
bounded, predictable cost at every single stage, and defers true
correctness to a cheap post-hoc geometric check once real structures
exist anyway. This module deliberately keeps the exact combined-group
dedup (it is the actual value proposition versus reactive
`assignment_search.py`: completeness, not just speed), but 2.1.0 adopts
Architector's other lesson -- every stage needs its OWN bounded-cost
safety net, not just the first one -- by giving `dedup_combined` a real
time budget of its own (see "Combinatorial blowup / performance" above)
rather than relying on `enumerate_assignments`'s cap to protect it by
proxy. A hard top-K output truncation and/or a cheap pre-dedup energy
heuristic, Architector-style, remain options for phase 4 if the exact
approach still proves too slow on a real CN8/9 ligand once benchmarked
-- not adopted here since the exact fix (mitigation 1 in
`dedup_combined`) already resolved the one concrete case measured so
far without sacrificing completeness. Measured after the fix: CN8/CN9
with one large donor-equivalence group (the case that was 64s) now
takes 0.08s/0.82s respectively; CN9 with NO donor-equivalence groups at
all (every donor unique -- the case mitigation 1 cannot help, since it
degrades to a no-op with no groups to pre-reduce by, and cost falls
back to the O(kept * |site symmetry group|) residual mitigation 2 was
built for) was still running past 60s when the benchmark was cut off
deliberately rather than left to finish; the analogous CN8/no-groups
case took ~53s uncapped and produced 2,520 final "unique" candidates
from 5,000 raw (a low ~2x collapse ratio is exactly the regime where
the pairwise scan can't early-exit often) -- confirming mitigation 2
(time_budget_s on dedup_combined itself) is a real, necessary safety
net for this case, not a hypothetical one: capping it at 3s and
re-running produces `dedup_truncated=True` with 839 candidates in
3.03s, bounded as designed, rather than running uncontrolled.

Version
-------
1.0.0 : initial release (phase 1, 2026-09-17). Site-symmetry automorphism
        groups only. No enumerator, no dedup, no engine.py wiring.
2.0.0 : phase 2 (2026-09-17). Added `enumerate_assignments()` (backtracking
        bijection enumerator), `dedup_by_donor_equivalence()` (port of
        engine.py's 2.14.0 Fix A), `dedup_by_site_equivalence()` (new,
        uses phase 1's site_symmetry_group()), and `dedup_combined()`
        (new -- the actually-correct combined-orbit dedup; see "Findings
        from standalone validation" above), plus the
        `enumerate_unique_assignments()` convenience that chains
        `enumerate_assignments` -> `dedup_combined`. Still no engine.py
        wiring (phase 4).
2.1.0 : performance fix (2026-09-17), prompted by benchmarking a
        synthetic unconstrained-CN8 worst case and researching how
        Architector handles the analogous problem (see "Comparison to
        Architector's own approach" above). `dedup_combined` now
        pre-reduces its input with the cheap `dedup_by_donor_equivalence`
        pass before the expensive pairwise (donor-relabeling x
        site-symmetry) scan, and accepts its own `config:
        SiteEnumerationConfig` for a real `time_budget_s` safety net
        (previously only `enumerate_assignments` had one -- the dedup
        step itself had no budget at all). `dedup_combined` now returns
        `DedupResult` (assignments + truncated) instead of a bare list;
        `UniqueEnumerationResult` gained `dedup_truncated`.
3.0.0 : phase 3 (2026-09-17). `ComplexBuilder.solve(use_symmetry_
        enumeration=True)` (engine.py, NOT this module) wires
        `enumerate_unique_assignments()` into the "3a." candidate-list
        step, reusing screen-then-resolve unchanged -- see "Proposed
        architecture" piece 4 above. No API changes in this module
        itself; version bumped because this module's own docstring
        (this section) now documents piece 4 as done rather than
        follow-on work.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from jacobik.ik.assignment_search import is_pair_reachable, site_angles_deg
except ImportError:  # pragma: no cover - standalone/test loading path

    def site_angles_deg(ideal_directions: np.ndarray) -> np.ndarray:  # type: ignore[no-redef]
        """Fallback copy of assignment_search.site_angles_deg for standalone
        use if jacobik isn't importable as a package (e.g. run as a loose
        script). Kept byte-for-byte equivalent; see assignment_search.py
        for the canonical version this defers to when available."""
        dots = np.clip(ideal_directions @ ideal_directions.T, -1.0, 1.0)
        return np.degrees(np.arccos(dots))

    def is_pair_reachable(hops: Optional[int], site_angle_deg: float) -> bool:  # type: ignore[no-redef]
        """Fallback copy of assignment_search.is_pair_reachable (its
        REACHABILITY_BREAKPOINTS / MIN_BITE_DEG_SHORT / MIN_BITE_HOPS
        constants, inlined) for the same standalone loading path as
        site_angles_deg above. Kept byte-for-byte equivalent."""
        breakpoints = ((4, 115.0), (6, 150.0), (8, 170.0), (10, 180.0))
        min_bite_deg_short = 55.0
        min_bite_hops = 4
        if hops is None or hops < 0:
            return True
        upper = 180.0
        for max_hops, bound in breakpoints:
            if hops <= max_hops:
                upper = bound
                break
        if site_angle_deg > upper + 1e-6:
            return False
        if hops <= min_bite_hops and site_angle_deg < min_bite_deg_short - 1e-6:
            return False
        return True


try:
    from jacobik.ik.targets import IDEAL_GEOMETRIES
except ImportError:  # pragma: no cover - standalone/test loading path
    IDEAL_GEOMETRIES = None  # type: ignore[assignment]


#: Default numerical tolerance (degrees) for treating two site-site angles
#: as equal. IDEAL_GEOMETRIES entries are algebraically exact constructions
#: (sqrt/trig closed forms) evaluated in float64, so genuine symmetry
#: relations agree to ~1e-10 deg; this is generous headroom, not a fit
#: parameter, and deliberately far tighter than assignment_search.py's
#: chemistry-derived reachability bounds (which bound real, noisy donor
#: geometry, not this module's exact idealised one).
DEFAULT_TOL_DEG: float = 1e-3

#: Per-geometry-label cache, populated lazily on first
#: `site_symmetry_group()` call (not at import time -- this module should
#: cost nothing to import when unused, same policy as
#: ik/assignment_search.py's `use_assignment_search` opt-in flag).
_AUTOMORPHISM_CACHE: Dict[Tuple[str, float], List[Tuple[int, ...]]] = {}


def _angle_signature(angles_row: np.ndarray, self_idx: int, tol_deg: float) -> Tuple[int, ...]:
    """A permutation-invariant fingerprint for site `self_idx`: the sorted,
    tolerance-quantised list of its angles to every other site. Two sites
    can only be mapped onto each other by an automorphism if they share
    this signature -- used purely to prune the backtracking search below,
    not as part of the correctness criterion itself (which is the exact
    pairwise check in `_find_automorphisms`'s inner `consistent` helper)."""
    others = np.delete(angles_row, self_idx)
    quantised = np.round(others / tol_deg).astype(np.int64)
    return tuple(sorted(quantised.tolist()))


def _find_automorphisms(angles_deg: np.ndarray, tol_deg: float) -> List[Tuple[int, ...]]:
    """Backtracking search for every permutation of range(n) that leaves
    `angles_deg` (an (n, n) pairwise-angle matrix) invariant, i.e. every
    sigma with angles_deg[sigma[i], sigma[j]] == angles_deg[i, j] for all
    i, j. Signature-based candidate pruning (see `_angle_signature`) keeps
    this well clear of the raw n! brute-force the feasibility writeup
    flagged as the naive approach -- correctness comes from the exact
    per-pair check in the inner loop, the signature is only a speed-up. A
    fully independent, deliberately-naive itertools.permutations
    brute-force cross-check (no pruning at all) lives in the test file and
    must agree with this function exactly for every geometry small enough
    to afford it."""
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
    geometry_label: str,
    tol_deg: float = DEFAULT_TOL_DEG,
    ideal_geometries: Optional[Dict[str, np.ndarray]] = None,
) -> List[Tuple[int, ...]]:
    """The full group of site-index permutations that leave
    `ideal_geometries[geometry_label]`'s shape invariant (see module
    docstring for the precise, orthogonal-transform-existence definition
    this is equivalent to). Always includes the identity permutation.
    Result is cached per (label, tol_deg) after first computation.

    Parameters
    ----------
    geometry_label   : key into ik/targets.py::IDEAL_GEOMETRIES (e.g.
                        "octahedral", "tetrahedral").
    tol_deg          : angle-equality tolerance in degrees; see
                        DEFAULT_TOL_DEG.
    ideal_geometries : override for IDEAL_GEOMETRIES, for testing without
                        a full jacobik install. Defaults to
                        ik/targets.py::IDEAL_GEOMETRIES.

    Returns
    -------
    List of n_sites-length tuples, each a permutation `sigma` (site i maps
    to site sigma[i]), in no particular order beyond identity-first.
    """
    geometries = IDEAL_GEOMETRIES if ideal_geometries is None else ideal_geometries
    if geometries is None:
        raise RuntimeError(
            "IDEAL_GEOMETRIES unavailable (jacobik.ik.targets failed to import) "
            "and no ideal_geometries override was supplied."
        )
    if geometry_label not in geometries:
        raise KeyError(
            f"Unknown geometry label {geometry_label!r}; known labels: {sorted(geometries.keys())}"
        )

    cache_key = (geometry_label, tol_deg)
    if cache_key not in _AUTOMORPHISM_CACHE:
        directions = geometries[geometry_label]
        angles_deg = site_angles_deg(directions)
        automorphisms = _find_automorphisms(angles_deg, tol_deg)
        # Identity is always geometrically valid; put it first for
        # deterministic, readable output (callers that just want group
        # order don't care, but tests and future dedup code benefit from
        # a stable "no-op" element in a known position).
        identity = tuple(range(directions.shape[0]))
        automorphisms.sort(key=lambda p: p != identity)
        _AUTOMORPHISM_CACHE[cache_key] = automorphisms
    return _AUTOMORPHISM_CACHE[cache_key]


def site_symmetry_group_order(
    geometry_label: str,
    tol_deg: float = DEFAULT_TOL_DEG,
    ideal_geometries: Optional[Dict[str, np.ndarray]] = None,
) -> int:
    """Convenience wrapper: `len(site_symmetry_group(...))`."""
    return len(site_symmetry_group(geometry_label, tol_deg, ideal_geometries))


# ---------------------------------------------------------------------------
# Phase 2: backtracking bijection enumerator
# ---------------------------------------------------------------------------


@dataclass
class SiteEnumerationConfig:
    """
    Tunable hyperparameters for `enumerate_assignments`. Mirrors the style
    of engine.py's `AssignmentSearchConfig` / ik_solver.py's
    `IKSolverConfig` -- a plain dataclass of defaults a caller can override
    piecemeal via `dataclasses.replace`.

    max_candidates : hard cap on the number of FULL (n_sites-length,
        reachability-valid) assignments collected before the search stops
        early. `None` disables the cap (exhaustive search). Checked
        cheaply (an int comparison) at every node, not just on completing
        an assignment, so a pathological branch can't blow past this by
        finishing many assignments in the same recursive call.
    time_budget_s  : wall-clock cap in seconds. `None` disables it. Same
        risk-mitigation role as max_candidates, for the case where
        pruning is weak (e.g. hop_matrix is None or nearly all-unknown)
        and the raw branching factor dominates.

    Defaults (2026-09-17, phase 4 benchmarking against real CN6/CN8/CN9
    cases -- la_polyester, [La(H2O)8]3+, [La(EDTA)(H2O)3]-, see
    script_notebooks/site_enumeration_experiments.ipynb): the raw
    backtracking search + dedup is CHEAP relative to the downstream IK
    screening each surviving candidate still has to go through --
    exhaustive enumeration ran in well under a second for a real CN8
    all-water case (n_raw=40320) and ~6s for a real CN9 case with
    genuine chelate topology ([La(EDTA)(H2O)3]-, n_raw=12240). The old
    max_candidates=5000 default silently truncated the RAW search (not
    just a safety net for pathological inputs) for both of those real
    cases -- for the CN9 case specifically, this meant the reported "20
    unique candidates" could not be trusted as complete without a
    separate exhaustive re-run to confirm it (it happened to already be
    complete here, but that was not knowable from the truncated run
    alone). Raised well above realistic CN8/CN9 raw-candidate counts so
    truncation, when it does fire, is a genuine signal about an
    unusually weakly-pruned or higher-CN case rather than routine
    behaviour on ordinary real ligands. time_budget_s raised to match
    (still a real, finite backstop against a pathological hop_matrix).
    The actual runtime driver for a many-candidate real case is
    AssignmentSearchConfig's screen_beam_width/screen_n_ik_steps (the
    per-candidate IK screening cost in engine.py's
    `_resolve_candidates`), not this config -- see that dataclass if a
    real case's wall-clock time needs tuning.
    """

    max_candidates: Optional[int] = 50_000
    time_budget_s: Optional[float] = 30.0


@dataclass
class EnumerationResult:
    """Return value of `enumerate_assignments`.

    assignments : the (possibly incomplete, see `truncated`) list of
        distinct, reachability-valid (n_donors,) donor-to-site bijections
        found -- NOT yet deduplicated by donor- or site-equivalence (see
        `enumerate_unique_assignments` for that).
    truncated   : True iff `SiteEnumerationConfig.max_candidates` or
        `time_budget_s` was hit before the search space was exhausted.
        `assignments` is always a set of genuinely VALID bijections either
        way -- truncation only means the search stopped early, never that
        an invalid one slipped through.
    """

    assignments: List[np.ndarray] = field(default_factory=list)
    truncated: bool = False


def _hop(hop_matrix: Optional[np.ndarray], i: int, j: int) -> Optional[int]:
    """Same "no known path" convention as assignment_search.py uses
    inline throughout: `hop_matrix` is `None` (nothing known at all), or
    an (n, n) array whose [i, j] entry is a non-negative hop count or a
    negative sentinel (engine.py always uses `-1`, matching
    `np.full((n, n), -1, dtype=int)`) meaning "no known path / cross-
    ligand / unconstrained". Either way this returns `None` for
    "unconstrained", an `int` otherwise."""
    if hop_matrix is None:
        return None
    hops = hop_matrix[i, j]
    return None if hops is None or hops < 0 else int(hops)


def _next_donor(
    remaining: Sequence[int], placed: Sequence[int], hop_matrix: Optional[np.ndarray]
) -> int:
    """Most-constrained-first donor ordering (see module docstring point
    2): among `remaining` donors, picks the one with the smallest known
    hop distance to any donor in `placed`, so that pair's (tight) bite-
    angle bound is engaged as early as possible in the search. Before
    anything is placed, uses the smallest hop distance to ANY other donor
    instead, i.e. starts from the most tightly tethered pair in the whole
    molecule (e.g. a chelate ring's first arm). Ties, and the fully-
    unconstrained case (hop_matrix is None or nothing reachable), fall
    back to the lowest donor index, for determinism."""
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
    ideal_geometries: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[SiteEnumerationConfig] = None,
) -> EnumerationResult:
    """
    Backtracking search over ALL donor-to-site bijections for a fixed
    geometry (`n_donors` must equal the geometry's n_sites -- CN must
    match), pruned incrementally by ligand-graph reachability.

    Builds an assignment one donor at a time (see `_next_donor` for the
    most-constrained-first placement order), and for each candidate site,
    checks the new donor-site pair against EVERY already-placed donor
    (not just the most recently placed one) via
    `assignment_search.is_pair_reachable` -- an incremental version of
    the full-assignment check `assignment_search.assignment_is_reachable`
    already does, applied as each donor is placed instead of only once at
    the end, so a violating branch is pruned before its
    `(n_donors - depth)!` unexplored subtree is ever visited.

    A `hop_matrix` that is `None`, or has no known hop distance for any
    pair (e.g. several fully independent monodentate ligands, like
    Fe(H2O)6's six separate waters), degrades gracefully to an arbitrary
    donor order and a pure reachability-unconstrained search over ALL
    n_sites! bijections -- correct, just potentially slow for large CN
    (see module docstring's "Combinatorial blowup" section and
    `SiteEnumerationConfig`).

    Parameters
    ----------
    n_donors         : number of donors; must equal the geometry's
                        n_sites (this function only ever produces
                        bijections).
    geometry_label    : key into IDEAL_GEOMETRIES.
    hop_matrix        : optional (n_donors, n_donors) ligand-graph hop
                        distances; see `_hop`'s docstring for the
                        "unconstrained" sentinel convention (matches
                        assignment_search.py / engine.py exactly).
    ideal_geometries  : override for IDEAL_GEOMETRIES; see
                        `site_symmetry_group`.
    config            : SiteEnumerationConfig; defaults applied if None.

    Returns
    -------
    EnumerationResult -- see its own docstring.
    """
    geometries = IDEAL_GEOMETRIES if ideal_geometries is None else ideal_geometries
    if geometries is None:
        raise RuntimeError(
            "IDEAL_GEOMETRIES unavailable (jacobik.ik.targets failed to import) "
            "and no ideal_geometries override was supplied."
        )
    if geometry_label not in geometries:
        raise KeyError(
            f"Unknown geometry label {geometry_label!r}; known labels: {sorted(geometries.keys())}"
        )
    ideal_directions = geometries[geometry_label]
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
    start = time.monotonic()

    def budget_exceeded() -> bool:
        if cfg.max_candidates is not None and len(assignments) >= cfg.max_candidates:
            return True
        if cfg.time_budget_s is not None and (time.monotonic() - start) > cfg.time_budget_s:
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


# ---------------------------------------------------------------------------
# Phase 2: two-stage dedup
# ---------------------------------------------------------------------------


def dedup_by_donor_equivalence(
    assignments: Sequence[np.ndarray],
    donor_equivalence_groups: Sequence[Sequence[int]],
) -> List[np.ndarray]:
    """
    Donor-equivalence dedup -- an unchanged port of engine.py's 2.14.0
    "Fix A" (`ComplexBuilder.solve()`'s "3a." step, the block titled
    "drop candidates that are pure relabelings of an already-kept
    candidate among chemically-identical donors"), generalised from
    "dedupe a list against one fixed baseline" to "dedupe a whole list
    against itself", keeping the first occurrence of each equivalence
    class (same greedy convention Fix A itself uses: baseline/earlier
    candidates always win over later ones).

    Two assignments are the same class iff, for every group in
    `donor_equivalence_groups`, they send that group's donors to the SAME
    SET of sites (order among a group's own donors doesn't matter -- by
    definition they're chemically interchangeable), AND every donor
    outside every group is sent to the identical site by both. This is
    exactly Fix A's `_is_trivial_relabeling` check.

    Parameters
    ----------
    assignments               : candidate (n_donors,) bijections.
    donor_equivalence_groups  : e.g. engine.py's `symmetric_groups` --
        groups of >=2 donor indices that are chemically interchangeable
        (same SMILES + RDKit CanonicalRankAtoms rank). Groups of size < 2
        are ignored (nothing to relabel). This module computes nothing
        here itself -- see "Why no RDKit" in the module docstring.

    Returns
    -------
    List of the kept (n_donors,) arrays, order preserved, first-of-class
    kept.
    """
    if not donor_equivalence_groups:
        return list(assignments)
    groups = [list(g) for g in donor_equivalence_groups if len(g) >= 2]
    if not groups:
        return list(assignments)
    grouped_donors = {d for g in groups for d in g}
    ungrouped = [d for d in range(len(assignments[0])) if d not in grouped_donors] if assignments else []

    # --- 2026-09-17 vectorization ---------------------------------------
    # "Same class" as defined above -- same SET of sites per group, exact
    # match outside every group -- is a genuine equivalence relation (it's
    # literally equality of a canonical signature, defined below), not
    # just a pairwise predicate. The original implementation checked it
    # the pairwise way anyway (`any(is_relabeling(cand, k) for k in
    # kept)`, an O(len(assignments) * len(kept)) pure-Python scan using
    # `sorted()` per group per comparison), which cProfile showed costing
    # multiple seconds on realistic candidate counts during the batched
    # candidate-screening solve (see script_notebooks/
    # site_enumeration_experiments.ipynb). Since it IS an equivalence
    # relation, first-occurrence-per-class can instead be computed by
    # hashing each candidate's canonical signature -- O(n_assignments)
    # amortized instead of O(n_assignments * n_kept) -- with identical
    # output (order preserved, first-of-class kept), because two
    # candidates are the same class iff their canonical signatures are
    # equal by construction:
    #   signature = (sorted sites for each group, in group order) +
    #               (exact site for each ungrouped donor, in donor order)
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


def dedup_by_site_equivalence(
    assignments: Sequence[np.ndarray],
    geometry_label: str,
    *,
    tol_deg: float = DEFAULT_TOL_DEG,
    ideal_geometries: Optional[Dict[str, np.ndarray]] = None,
) -> List[np.ndarray]:
    """
    Site-equivalence dedup (new -- not a port of anything existing).
    Two assignments `a`, `b` are the same physical structure up to a
    rigid rotation/reflection of the whole complex iff some permutation
    `sigma` in `site_symmetry_group(geometry_label)` carries one onto the
    other: `sigma[a[i]] == b[i]` for every donor `i` (derivation in the
    module docstring's automorphism-definition section -- `sigma` is
    exactly the site relabeling induced by the orthogonal transform Q
    that maps b's physical structure onto a's). Since
    `site_symmetry_group` includes reflections, this collapses mirror-
    image (enantiomeric) donor arrangements together -- see the module
    docstring's discussion of when that is and isn't the right call.
    Keeps the first occurrence of each orbit, same greedy convention as
    `dedup_by_donor_equivalence`.

    Parameters
    ----------
    assignments      : candidate (n_donors,) bijections; n_donors must
                        equal the geometry's n_sites.
    geometry_label    : key into IDEAL_GEOMETRIES.
    tol_deg, ideal_geometries : forwarded to `site_symmetry_group`.

    Returns
    -------
    List of the kept (n_donors,) arrays, order preserved, first-of-orbit
    kept.
    """
    group = site_symmetry_group(geometry_label, tol_deg=tol_deg, ideal_geometries=ideal_geometries)

    def same_orbit(a: np.ndarray, b: np.ndarray) -> bool:
        return any(all(sigma[a[i]] == b[i] for i in range(len(a))) for sigma in group)

    kept: List[np.ndarray] = []
    for cand in assignments:
        if any(same_orbit(cand, k) for k in kept):
            continue
        kept.append(cand)
    return kept


def _donor_relabeling_group(
    n_donors: int, donor_equivalence_groups: Sequence[Sequence[int]]
) -> List[Tuple[int, ...]]:
    """Every donor-index permutation obtainable by independently
    permuting each equivalence group's members among themselves (and
    fixing every donor outside every group) -- the same freedom
    `dedup_by_donor_equivalence` / engine.py's Fix A grants, expressed as
    an explicit permutation group (a list of permutation tuples, same
    shape as `site_symmetry_group`'s output) instead of an implicit
    "same set of sites" check, so it can be COMPOSED with
    `site_symmetry_group`'s output -- see `dedup_combined`. Group order
    is the product of each group's size! (e.g. 2! x 4! = 48 for EDTA's
    `[[N1, N2], [O1, O1b, O2, O2b]]`), always includes the identity."""
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
    """Return value of `dedup_combined`.

    assignments     : the final, fully-deduplicated (n_donors,) bijections.
    truncated       : True iff `SiteEnumerationConfig.time_budget_s` was hit
        before every candidate had been processed -- see `dedup_combined`'s
        "Performance" section. `assignments` is always correctly deduped
        among the candidates that WERE processed either way; truncation
        only means some tail of the input list was never looked at, never
        that a duplicate slipped through undetected.
    """

    assignments: List[np.ndarray] = field(default_factory=list)
    truncated: bool = False


def dedup_combined(
    assignments: Sequence[np.ndarray],
    donor_equivalence_groups: Sequence[Sequence[int]],
    geometry_label: str,
    *,
    tol_deg: float = DEFAULT_TOL_DEG,
    ideal_geometries: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[SiteEnumerationConfig] = None,
) -> DedupResult:
    """
    The correct two-stage dedup: two assignments are the same physical
    structure iff related by SOME combination of a donor relabeling
    (within `donor_equivalence_groups`) and a site symmetry
    (`site_symmetry_group(geometry_label)`), applied together -- NOT by
    running `dedup_by_donor_equivalence` and then `dedup_by_site_
    equivalence` as two independent sequential passes.

    That naive sequential composition was tried first and is WRONG: it
    can leave genuine duplicates un-merged whenever the only path
    between two assignments requires combining both kinds of move,
    neither of which alone connects them. Concretely, on EDTA (see
    script_notebooks/site_enumeration_experiments.ipynb for the full
    worked example, donor order [N1, N2, O1, O1b, O2, O2b], octahedral,
    donor_equivalence_groups=[[N1, N2], [O1, O1b, O2, O2b]]):
    assignment A = {N1:0, N2:2, O1:3, O1b:4, O2:1, O2b:5} and
    assignment B = {N1:0, N2:4, O1:2, O1b:5, O2:1, O2b:3} are NOT
    donor-equivalence-relabelings of each other (N2 itself sits at a
    different SITE, 2 vs 4, not just a different donor at the same site
    set) and are NOT related by any site symmetry alone (fixing N1 at
    site 0 forces N2's site to move within a specific 8-element
    stabiliser subgroup of the 48-element octahedral group that does not
    contain the needed permutation) -- so BOTH sequential passes report
    them as distinct. But relabel N1<->N2 (a legitimate donor
    equivalence move) and THEN apply a site rotation, and they coincide:
    they are the same physical structure. `dedup_by_donor_equivalence`
    and `dedup_by_site_equivalence` are kept as separate, independently
    testable functions (each is individually correct for the moves it
    considers, and the feasibility writeup's architecture describes them
    as two distinct pieces), but `enumerate_unique_assignments` uses
    THIS function, not a chain of those two, because only this one is
    correct end to end.

    Implementation: for each candidate, checks it against every already-
    kept assignment by trying every (donor relabeling, site symmetry)
    pair in turn and stopping at the first match (not by computing a
    canonical form over the full combined group up front, which is
    correct but ~10x slower in practice -- see the notebook's timing
    comparison). Cost is bounded by
    O(len(assignments) * len(kept) * |donor relabeling group| *
    |site symmetry group|) in the worst case (no early matches at all).
    Two mitigations keep this well clear of that bound in practice:

    1. Candidates are pre-reduced with the cheap `dedup_by_donor_
       equivalence` pass FIRST, before the expensive pairwise scan ever
       runs. That pass is sound but incomplete on its own (see the
       counterexample above -- it's exactly why `dedup_combined` exists),
       but "incomplete" only means it can leave duplicates for the
       pairwise scan to still catch, never that it removes something
       that shouldn't have been removed: every pair it merges is
       related by a pure donor relabeling, which is one specific element
       of the combined group (site symmetry = identity), so nothing a
       full combined-orbit scan would have kept is ever discarded here.
       This matters because `|donor relabeling group|` -- unlike
       `|site symmetry group|`, which is always <=48 for every geometry
       in IDEAL_GEOMETRIES -- can be as large as n_donors! for a fully
       homoleptic donor set (e.g. 8! = 40,320 for an unconstrained CN8
       case with all 8 donors in one equivalence group), and that
       factor, not the site-symmetry one, is what dominates the naive
       worst-case bound above. Measured effect: an unconstrained CN8
       case (square_anti, 1 donor-equivalence group of all 8, 5,000 raw
       candidates after `enumerate_assignments`'s default cap) that took
       ~64s before this pre-reduction (dominated by scanning up to
       40,320 relabelings per pairwise comparison) drops to a small
       fraction of a second after it, because
       `dedup_by_donor_equivalence` alone already collapses all 5,000
       raw candidates to the single donor-relabeling orbit they all
       belong to, in O(n_raw) time -- leaving the expensive pairwise
       scan almost nothing left to do. See
       script_notebooks/site_enumeration_experiments.ipynb for the
       before/after timing.
    2. `config.time_budget_s` (same `SiteEnumerationConfig` passed to
       `enumerate_assignments`) is honoured as a real wall-clock cap on
       the pairwise scan itself, checked once per outer-loop candidate.
       This is a genuine safety net, not just the first mitigation's
       side effect: a donor set with NO equivalence groups at all (every
       donor chemically unique, so mitigation 1 is a no-op) combined
       with a large raw candidate count and a low true collapse ratio
       can still hit the `O(len(assignments) * len(kept) * |site
       symmetry group|)` residual cost (up to 48x per comparison, with
       no group-size mitigation available -- site-symmetry group order
       isn't reducible the way donor-relabeling group order is). Hitting
       the budget stops the scan early and reports `truncated=True`
       rather than silently returning an under-deduped list with no
       signal that anything was cut short. `enumerate_assignments`'s own
       `max_candidates`/`time_budget_s` cap the RAW list `dedup_combined`
       ever sees, but (as found while benchmarking CN8, see the module
       docstring's "Combinatorial blowup" section) that cap was
       previously the only safety net in the whole
       `enumerate_unique_assignments` pipeline -- `dedup_combined` itself
       received no budget at all before this, so a capped-but-still-
       large raw list could blow the dedup step's own runtime past
       `time_budget_s` with nothing to stop or flag it. This parameter
       closes that gap.

    Returns
    -------
    `DedupResult` -- see its own docstring.
    """
    if not assignments:
        return DedupResult(assignments=[], truncated=False)

    # Mitigation 1 (see "Implementation" above): cheap, sound pre-reduction
    # before the expensive pairwise (relabeling x site-symmetry) scan.
    pre_reduced = dedup_by_donor_equivalence(assignments, donor_equivalence_groups)

    n_donors = len(pre_reduced[0])
    relabelings = _donor_relabeling_group(n_donors, donor_equivalence_groups)
    site_group = site_symmetry_group(
        geometry_label, tol_deg=tol_deg, ideal_geometries=ideal_geometries
    )

    # --- 2026-09-17 vectorization ---------------------------------------
    # The original `equivalent(a, b)` below did the exact same combinatorial
    # check as this vectorized version, but via a pure-Python nested loop
    # (`for relabel in relabelings: for sigma in site_group: ...`) called
    # once per (candidate, already-kept) pair. Profiling the batched
    # candidate-screening solve (see script_notebooks/
    # site_enumeration_experiments.ipynb) showed this costing several
    # seconds of wall time on realistic donor/site group sizes, entirely
    # from Python-level generator-expression and `all()`/`sorted()`
    # overhead -- not from the underlying combinatorics, which is small
    # (|relabelings| * |site_group| is at most a few hundred for every
    # geometry/donor-equivalence combination this module supports).
    #
    # `all_transforms(a)` computes every image of `a` under the combined
    # group in one shot with numpy fancy indexing, and the duplicate check
    # against everything kept so far is a single vectorized comparison
    # instead of one Python call per already-kept candidate. This is
    # exactly the same computation as the old `any(equivalent(cand, k) for
    # k in kept)` -- not an approximation -- just executed as C-level numpy
    # ops instead of millions of small Python-level ones:
    #   relabeled[r, i]    = a[relabelings_arr[r, i]]       (old: relabeled[i] = a[relabel[i]])
    #   transformed[s,r,i] = site_group_arr[s, relabeled[r,i]]  (old: sigma[relabeled[i]])
    relabelings_arr = np.asarray(relabelings)  # (R, n_donors)
    site_group_arr = np.asarray(site_group)  # (S, n_donors)

    def all_transforms(a: np.ndarray) -> np.ndarray:
        relabeled = a[relabelings_arr]  # (R, n_donors)
        transformed = site_group_arr[:, relabeled]  # (S, R, n_donors)
        return transformed.reshape(-1, n_donors)  # (S * R, n_donors)

    # Mitigation 2: a real wall-clock cap on the pairwise scan itself.
    start = time.monotonic()
    truncated = False
    kept: List[np.ndarray] = []
    kept_arr = np.empty((0, n_donors), dtype=np.asarray(pre_reduced[0]).dtype)
    for cand in pre_reduced:
        if (
            config is not None
            and config.time_budget_s is not None
            and (time.monotonic() - start) > config.time_budget_s
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


@dataclass
class UniqueEnumerationResult:
    """Return value of `enumerate_unique_assignments`.

    assignments    : final, fully-deduplicated (n_donors,) bijections (see
        `dedup_combined`).
    truncated      : forwarded from the raw `enumerate_assignments` call --
        True means the RAW search (before any dedup) was cut off by
        SiteEnumerationConfig, so `assignments` may be missing some
        genuinely-unique candidates, not just relabelings of ones kept.
    dedup_truncated : forwarded from `dedup_combined`'s own `DedupResult`
        -- True means the DEDUP pass itself was cut off by
        `SiteEnumerationConfig.time_budget_s` before every raw candidate
        had been checked (see `dedup_combined`'s "Implementation" section,
        mitigation 2), a distinct failure mode from `truncated` above:
        `truncated=False, dedup_truncated=True` means the raw search was
        complete but ran out of time budget deduplicating its own output.
    n_raw          : count of valid bijections found before any dedup.
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
    ideal_geometries: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[SiteEnumerationConfig] = None,
    tol_deg: float = DEFAULT_TOL_DEG,
) -> UniqueEnumerationResult:
    """Convenience wrapper chaining `enumerate_assignments` ->
    `dedup_combined` -- the full phase-2 pipeline the feasibility
    writeup's "Proposed architecture" section describes end to end (see
    `dedup_combined`'s docstring for why this is NOT
    `dedup_by_donor_equivalence` followed by `dedup_by_site_
    equivalence`). `config` is forwarded to BOTH steps (see
    `dedup_combined`'s "Implementation" section, mitigation 2, for why the
    dedup step needs its own budget too, not just the raw search). See
    each piece's own docstring for details; see `UniqueEnumerationResult`
    for what's returned."""
    raw = enumerate_assignments(
        n_donors, geometry_label, hop_matrix, ideal_geometries=ideal_geometries, config=config
    )
    deduped = dedup_combined(
        raw.assignments,
        donor_equivalence_groups,
        geometry_label,
        tol_deg=tol_deg,
        ideal_geometries=ideal_geometries,
        config=config,
    )
    return UniqueEnumerationResult(
        assignments=deduped.assignments,
        truncated=raw.truncated,
        dedup_truncated=deduped.truncated,
        n_raw=len(raw.assignments),
    )
