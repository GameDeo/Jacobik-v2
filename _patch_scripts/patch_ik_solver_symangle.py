#!/usr/bin/env python3
"""
ik_solver.py 2.4.0: extend the RDKit-canonical-rank symmetry-tying
restraint (2.3.0) to also tie Q3 (donor approach angle) between
topologically-equivalent donors, not just M-D bond length.

Root cause this addresses (diagnosed via constraints.json backup/live
comparison, not yet confirmed against a live solve trace): EVERY one of
the 2956 old Q3 entries in the live constraints.json was type="gmm"
misfiled into what ik/scorer.py's _vmm_fields() treats as a VMM slot --
i.e. Q3 was NEVER actually a clean native VMM fit in production. It was
always going through _vmm_fields()'s "synthesise kappa from vars" fallback
shim, which computes kappa = 1/(var_deg^2 * (pi/180)^2) and then clamps to
_KAPPA_SYNTH_MAX = 10.0. For both of EDTA's donor types (N_amine,
O_carboxylate) the raw fitted variances were tight enough that EVERY
component's synthesised kappa saturated that cap -- so in practice Q3 was
an almost-toothless, uniformly-broad (~18 deg effective sigma) constraint
for the whole time it's been in production, not the differentiated,
sometimes-sharp preference it was designed to be.

The Q2/Q3 GMM-cos refit (this session, earlier) replaced those stale
misfiled entries with a proper native fit for the first time. Q3 now
genuinely discriminates between approach angles -- e.g. the new
N_amine/Co/octahedral entry has real structure (a moderately tight
dominant mode plus a much tighter secondary mode), instead of the old
uniformly-flat kappa=10 approximation. That's a big part of why the CO/TBP
orientation fix worked. But it also means two topologically-equivalent
donors (e.g. EDTA's two amine N, or its four carboxylate O) are now each
independently pulled by a real, sometimes-sharp Q3 preference for the
first time, and NOTHING currently ties their approach-angle outcomes
together -- the existing symmetric_groups restraint only ties M-D bond
length. That is a plausible, evidence-consistent mechanism for the
"uneven N anchoring / carboxylate bond-length spread" the user reported
after the refit: one donor settles near a good Q3 mode, its
chemically-identical partner settles near a different (locally reachable)
one, and the Q1/shape terms have to compromise differently for each to
reconcile the overall CShM target, showing up as bond-length spread.

This patch adds a second, independently-weighted soft restraint
(sym_angle_weight, default 1.0 -- a starting value, not empirically fit,
same caveat as shape_rho/shape_tol/sym_weight) that pulls each symmetric
group's Q3-eligible members' approach angles toward their group mean,
mirroring the existing bond-length restraint's pattern exactly. Backward
compatible: sym_angle_weight only has any effect when symmetric_groups is
supplied (2.3.0+ shape-ALM path) and at least one group has >=2
Q3-eligible members; both are opt-in the same way sym_weight already is.
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
# I0: version-history docstring
# ---------------------------------------------------------------------------
OLD_HIST_TAIL = """        Backward compatible: when `ideal_directions` is None (not passed
        by the caller), solve() falls back byte-identically to the old
        fixed-schedule anchor. anchor_floor_frac is accepted but ignored
        under the new path (the dual multiplier already re-inflates
        whenever the constraint is violated again -- no floor needed).
\"\"\""""
NEW_HIST_TAIL = """        Backward compatible: when `ideal_directions` is None (not passed
        by the caller), solve() falls back byte-identically to the old
        fixed-schedule anchor. anchor_floor_frac is accepted but ignored
        under the new path (the dual multiplier already re-inflates
        whenever the constraint is violated again -- no floor needed).
2.4.0 : Symmetry-tying restraint (2.3.0) extended to also tie Q3 (donor
        approach angle) between topologically-equivalent donors, not just
        M-D bond length.

        Root cause this addresses: EVERY Q3 entry in the live
        constraints.json (all 2956 of them) turned out to be type="gmm"
        misfiled into what ik/scorer.py's old _vmm_fields() treated as a
        VMM slot -- Q3 was never actually a clean native VMM fit in
        production, it was always going through a "synthesise kappa from
        vars, then clamp to _KAPPA_SYNTH_MAX=10" fallback shim. For both
        of EDTA's donor types the raw fitted variances were tight enough
        that every component saturated that clamp, so Q3 was in practice
        an almost-toothless, uniformly ~18 deg-sigma constraint for its
        entire history, not the differentiated preference it was designed
        to be. The Q2/Q3 GMM-cos refit (stats/fitter.py, same session)
        replaced those stale entries with a real native fit for the first
        time -- which is a big part of why the CO/TBP orientation fix
        worked, but it also means two chemically-identical donors (e.g.
        EDTA's two amine N, or its four carboxylate O) are now each
        independently pulled by a real, sometimes sharp Q3 preference for
        the first time, with nothing tying their outcomes together -- the
        symmetric_groups restraint only ever tied M-D distance. Plausible,
        evidence-consistent mechanism for the "uneven N anchoring /
        carboxylate bond-length spread" reported after that refit: one
        donor settles near a good Q3 mode, its symmetric partner settles
        near a different locally-reachable one, and the Q1/shape terms
        compromise differently for each to reconcile the overall CShM
        target.

        Fix: a second, independently-weighted soft restraint
        (sym_angle_weight, default 1.0 -- starting value, not empirically
        fit, same caveat as shape_rho/shape_tol/sym_weight) pulls each
        symmetric group's Q3-eligible members' approach angles toward
        their group mean, mirroring the existing bond-length restraint's
        pattern exactly (mean-squared deviation from the group mean,
        averaged over active groups). Deliberately a SEPARATE weight
        rather than reusing sym_weight: bond-length deviations (Å²) and
        approach-angle deviations (rad²) have different natural scales,
        and conflating them risks exactly the kind of cross-quantity
        imbalance this whole change is meant to fix, not another instance
        of it. Opt-in the same way sym_weight already is (only active when
        symmetric_groups is supplied and a group has >=2 Q3-eligible
        members) -- backward compatible, zero effect otherwise.
\"\"\""""
do_replace("I0-version-history", OLD_HIST_TAIL, NEW_HIST_TAIL)

# ---------------------------------------------------------------------------
# I1: typing import -- add Dict
# ---------------------------------------------------------------------------
do_replace(
    "I1-typing-import",
    "from typing import TYPE_CHECKING, List, Optional, Tuple",
    "from typing import TYPE_CHECKING, Dict, List, Optional, Tuple",
)

# ---------------------------------------------------------------------------
# I2: __init__ signature -- add sym_angle_weight
# ---------------------------------------------------------------------------
do_replace(
    "I2-init-signature",
    "        sym_weight: float = 2.0,\n        device: torch.device = torch.device(\"cpu\"),\n    ) -> None:",
    "        sym_weight: float = 2.0,\n        sym_angle_weight: float = 1.0,\n        device: torch.device = torch.device(\"cpu\"),\n    ) -> None:",
)

# ---------------------------------------------------------------------------
# I3: __init__ body -- store sym_angle_weight
# ---------------------------------------------------------------------------
do_replace(
    "I3-init-body",
    "        self.sym_weight = sym_weight\n        self.device = device",
    "        self.sym_weight = sym_weight\n        self.sym_angle_weight = sym_angle_weight\n        self.device = device",
)

# ---------------------------------------------------------------------------
# I4: class docstring -- document sym_angle_weight right after sym_weight
# ---------------------------------------------------------------------------
OLD_DOC = """    sym_weight   : fixed weight on the symmetry-tying restraint that pulls
                   topologically-equivalent donors' M-D distances toward
                   their group mean (2.3.0+; see engine.py's RDKit
                   canonical-rank symmetric-donor detection). This is a soft
                   chemical prior (analogous to crystallography's SIMU/SAME
                   restraints), not a hard constraint, so it uses a fixed
                   weight rather than the dual-ascent machinery above.
                   (default 2.0)
    device       : torch device
    \"\"\""""
NEW_DOC = """    sym_weight   : fixed weight on the symmetry-tying restraint that pulls
                   topologically-equivalent donors' M-D distances toward
                   their group mean (2.3.0+; see engine.py's RDKit
                   canonical-rank symmetric-donor detection). This is a soft
                   chemical prior (analogous to crystallography's SIMU/SAME
                   restraints), not a hard constraint, so it uses a fixed
                   weight rather than the dual-ascent machinery above.
                   (default 2.0)
    sym_angle_weight : fixed weight on the Q3 approach-angle symmetry-tying
                   restraint (2.4.0+; same symmetric_groups as sym_weight
                   above, but pulls each group's Q3-eligible members'
                   approach angles toward their group mean instead of their
                   M-D distances). Deliberately separate from sym_weight --
                   see the 2.4.0 changelog entry for why the two shouldn't
                   share one weight. Starting default, not empirically fit.
                   (default 1.0)
    device       : torch device
    \"\"\""""
do_replace("I4-docstring", OLD_DOC, NEW_DOC)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
