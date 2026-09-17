#!/usr/bin/env python3
"""
ik_solver.py 2.5.0: mode-committed Q3 initialization.

See module changelog (added below) for the full rationale. Short version:
symmetric donor groups whose Q3 target is genuinely multi-modal (real
chemistry -- e.g. carboxylate O binding modes) get pulled toward whichever
mode each member is nearest to; sym_angle_weight (2.4.0) then ties them
toward their MEAN angle, which for two well-separated modes can land in
the low-density valley between them. Fix: before the T-step loop starts,
for each symmetric group with >=2 Q3-eligible members, evaluate the
INITIAL pose's Q3 angle for each member against every GMM component of
its own fitted distribution, sum log-density across the group per
candidate component, and commit the whole group to whichever single
component that sum favours. That per-trial, per-group choice is passed
into every scorer.score() call in the trajectory via
q3_mode_override (scorer.py 2.8.0), collapsing each group's Q3 term to a
single well-behaved Gaussian shared by construction.
"""
import sys

PATH = "src/jacobik/ik/ik_solver.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

steps_done = []


def do_replace(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"FATAL: pattern for step [{label}] not found", file=sys.stderr)
        sys.exit(1)
    if n != count:
        print(f"FATAL: pattern for step [{label}] found {n} times, expected {count}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, count)
    steps_done.append(label)


# ---------------------------------------------------------------------------
# K0: version-history docstring
# ---------------------------------------------------------------------------
OLD_TAIL = '''        members) -- backward compatible, zero effect otherwise.
"""'''
NEW_TAIL = '''        members) -- backward compatible, zero effect otherwise.
2.5.0 : Mode-committed Q3 initialization -- solve() gains q3_mode_commit
        (default True).

        Root cause this addresses: with Q3 genuinely live (2.7.0 refit +
        the stats.py _is_empty_dist bugfix, same session), some donor
        subclasses have a REAL multi-modal fitted GMM -- confirmed on
        Co/octahedral O_carboxylate: 5 well-separated cos-space modes
        (means -0.638/-0.486/-0.439/-0.028/+0.045, i.e. roughly 87-130
        deg), reflecting genuinely different carboxylate binding modes
        pooled across the CSD (monodentate/bridging/chelating etc.), not
        a fitting artifact. EDTA's four carboxylate O are one RDKit
        canonical-rank symmetric group (all four -CH2COO- arms are
        graph-equivalent), so each independently settles near whichever
        mode it's nearest to, and the 2.4.0 sym_angle_weight restraint
        then pulls them toward their MEAN angle -- for two well-separated
        modes, the mean lands in the low-density valley BETWEEN them,
        which is worse than either mode alone and fights each donor's own
        strong pull toward its nearest mode. Observed effect: EDTA's
        n_converged collapsed (29 -> 1-2/32) and was essentially
        insensitive to sym_angle_weight (0.0/1.0/3.0 all similarly bad),
        consistent with the trajectory oscillating between competing
        modes rather than settling into a plateau ("converged" here means
        score-plateaued, not merely score-above-threshold).

        Fix: rather than letting each symmetric donor drift to its own
        mode and then averaging, COMMIT the whole group to one mode up
        front. Once per solve() call (not once per IK step -- the choice
        is made from the initial pose and held fixed for the whole
        trajectory, mirroring how theta_inits itself is fixed per trial),
        for each symmetric group with >=2 Q3-eligible members: evaluate
        the group's INITIAL Q3 angle (from theta_inits, before any
        gradient steps) in cos-space against every component of the
        group's own fitted Q3 GMM(s), sum log-density across the group's
        members for each candidate component k, and pick
        argmax_k(sum). This is exactly "which single mode best explains
        where this group already started", using the DB's own fitted
        weights and variances -- no new statistical assumption, just a
        hard per-group, per-trial commitment instead of independent
        per-donor soft draws from the same mixture. The resulting
        (B, n_q3) component-index tensor (-1 for non-grouped/ungrouped
        terms) is passed as q3_mode_override to every scorer.score() call
        for the rest of that solve() (scorer.py 2.8.0), which collapses
        the mixture to that single Gaussian for those terms -- turning a
        multi-modal tug-of-war into a well-behaved, shared unimodal
        target by construction. Because the choice is made per TRIAL
        (batch index), different trials in the same beam can still commit
        to different modes -- diversity across the beam is preserved,
        only the within-trial inconsistency is removed.

        Opt-in via q3_mode_commit (default True): only computed/applied
        when symmetric_groups is supplied and has a qualifying group (same
        gating as sym_angle_weight); q3_mode_commit=False restores
        pre-2.5.0 behaviour byte-identically (q3_mode_override stays None,
        scorer.py's new code path is a no-op). Not yet empirically
        validated against a live solve at the time of writing (implemented
        from static constraints.json inspection + code-path tracing only,
        since Claude cannot run torch in the environment this was
        developed in) -- see the test-suite notebook's shape-ALM A/B
        section for the intended comparison against q3_mode_commit=False
        and the sym_angle_weight variants (2.4.0).
"""'''
do_replace("K0-version-history", OLD_TAIL, NEW_TAIL)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
