#!/usr/bin/env python3
"""
engine.py 2.7.0: thread the new sym_angle_weight (ik_solver.py 2.4.0)
through ComplexBuilder.solve(), mirroring how sym_weight/shape_rho/
shape_tol are already threaded through. See ik_solver.py's 2.4.0
changelog for the full rationale (Q3 approach-angle symmetry tie, fixing
the "uneven N anchoring / carboxylate spread" EDTA regression).

Note: engine.py's own version marker is currently 2.6.0 in the module
docstring; this bumps it to 2.7.0.
"""
import sys

PATH = "src/jacobik/engine.py"

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
# EN0: version-history docstring
# ---------------------------------------------------------------------------
OLD_TAIL = '''         donors (e.g. EDTA's two amine N atoms differing by ~0.15 A with
         nothing previously tying them together). Backward compatible:
         use_shape_alm=False restores the pre-2.6.0 behaviour byte-
         identically. See ik/ik_solver.py JointIKSolver 2.3.0 changelog
         for the full technical rationale.
"""'''
NEW_TAIL = '''         donors (e.g. EDTA's two amine N atoms differing by ~0.15 A with
         nothing previously tying them together). Backward compatible:
         use_shape_alm=False restores the pre-2.6.0 behaviour byte-
         identically. See ik/ik_solver.py JointIKSolver 2.3.0 changelog
         for the full technical rationale.
2.7.0 : solve() gains sym_angle_weight, threaded through to JointIKSolver
         (ik_solver.py 2.4.0). Extends the 2.6.0 symmetry-tying restraint
         (M-D bond length only) to also tie Q3 (donor approach angle)
         between topologically-equivalent donors. Root cause: the entire
         live Q3 constraint table turned out to have been running through
         a broken score-time fallback (kappa clamped to 10, i.e. an
         almost-toothless ~18 deg-sigma approximation) for its whole
         history, independent of and prior to the periodicity/model-family
         issue the same-session Q2/Q3 GMM-cos refit (stats/fitter.py) set
         out to fix. Once that refit gave Q3 a real, properly fit
         distribution for the first time, two chemically-identical donors
         (e.g. EDTA's two amine N) could for the first time be pulled
         toward different modes of a genuinely differentiated Q3
         preference, with nothing tying their approach-angle outcomes
         together the way sym_weight already ties M-D distance -- a
         plausible mechanism for the "uneven N anchoring / carboxylate
         spread" reported after that refit. See ik/ik_solver.py's
         JointIKSolver 2.4.0 changelog for the full rationale.
"""'''
do_replace("EN0-version-history", OLD_TAIL, NEW_TAIL)

# ---------------------------------------------------------------------------
# EN1: solve() signature -- add sym_angle_weight
# ---------------------------------------------------------------------------
do_replace(
    "EN1-solve-signature",
    "        sym_weight: float = 2.0,\n        diversify_orientation: bool = True,",
    "        sym_weight: float = 2.0,\n        sym_angle_weight: float = 1.0,\n        diversify_orientation: bool = True,",
)

# ---------------------------------------------------------------------------
# EN2: solve() docstring -- document sym_angle_weight
# ---------------------------------------------------------------------------
do_replace(
    "EN2-solve-docstring",
    "        sym_weight       : fixed weight on the symmetry-tying restraint\n"
    "                           (only used when use_shape_alm). Default 2.0.\n"
    "        diversify_orientation : bool, default True (BUGFIX 2.5.1 -- was False).",
    "        sym_weight       : fixed weight on the symmetry-tying restraint\n"
    "                           (only used when use_shape_alm). Default 2.0.\n"
    "        sym_angle_weight : fixed weight on the Q3 approach-angle symmetry-\n"
    "                           tying restraint (only used when use_shape_alm).\n"
    "                           Same symmetric donor groups as sym_weight, but\n"
    "                           ties approach angle instead of M-D distance --\n"
    "                           deliberately a separate weight since the two\n"
    "                           quantities have different natural scales (Å² vs\n"
    "                           rad²). See ik/ik_solver.py's JointIKSolver 2.4.0\n"
    "                           changelog. Starting default, not empirically\n"
    "                           fit. Default 1.0.\n"
    "        diversify_orientation : bool, default True (BUGFIX 2.5.1 -- was False).",
)

# ---------------------------------------------------------------------------
# EN3: JointIKSolver(...) construction -- pass sym_angle_weight
# ---------------------------------------------------------------------------
do_replace(
    "EN3-jointiksolver-call",
    "            shape_rho=shape_rho,\n            shape_tol=shape_tol,\n            sym_weight=sym_weight,\n            device=_device,",
    "            shape_rho=shape_rho,\n            shape_tol=shape_tol,\n            sym_weight=sym_weight,\n            sym_angle_weight=sym_angle_weight,\n            device=_device,",
)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
