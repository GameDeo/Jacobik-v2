#!/usr/bin/env python3
"""
engine.py 2.8.0: thread q3_mode_commit (ik_solver.py 2.5.0) through
ComplexBuilder.solve(), mirroring sym_angle_weight's threading (2.7.0).
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
OLD_TAIL = '''         spread" reported after that refit. See ik/ik_solver.py's
         JointIKSolver 2.4.0 changelog for the full rationale.
"""'''
NEW_TAIL = '''         spread" reported after that refit. See ik/ik_solver.py's
         JointIKSolver 2.4.0 changelog for the full rationale.
2.8.0 : solve() gains q3_mode_commit, threaded through to JointIKSolver.solve()
         (ik_solver.py 2.5.0). Addresses a gap left by 2.7.0: with Q3 finally
         live, sym_angle_weight's mean-tie can land a symmetric group between
         two genuinely separate GMM modes (confirmed multi-modal for
         O_carboxylate on Co/octahedral -- 5 modes, ~87-130 deg) rather than
         at either one, which matches the observed EDTA regression (bond
         spread ~5x worse, n_converged collapsed 29->1-2, insensitive to
         sym_angle_weight). q3_mode_commit hard-selects one mode per
         symmetric group per trial from the initial pose instead of tying
         toward a mean. See ik/ik_solver.py's JointIKSolver 2.5.0 changelog
         for the full rationale.
"""'''
do_replace("EN0-version-history", OLD_TAIL, NEW_TAIL)

# ---------------------------------------------------------------------------
# EN1: solve() signature -- add q3_mode_commit
# ---------------------------------------------------------------------------
do_replace(
    "EN1-solve-signature",
    "        sym_angle_weight: float = 1.0,\n        diversify_orientation: bool = True,",
    "        sym_angle_weight: float = 1.0,\n        q3_mode_commit: bool = True,\n        diversify_orientation: bool = True,",
)

# ---------------------------------------------------------------------------
# EN2: solve() docstring -- document q3_mode_commit
# ---------------------------------------------------------------------------
do_replace(
    "EN2-solve-docstring",
    '        sym_angle_weight : fixed weight on the Q3 approach-angle symmetry-\n'
    '                           tying restraint (only used when use_shape_alm).\n'
    '                           Same symmetric donor groups as sym_weight, but\n'
    '                           ties approach angle instead of M-D distance --\n'
    '                           deliberately a separate weight since the two\n'
    '                           quantities have different natural scales (Å² vs\n'
    '                           rad²). See ik/ik_solver.py\'s JointIKSolver 2.4.0\n'
    '                           changelog. Starting default, not empirically\n'
    '                           fit. Default 1.0.\n'
    '        diversify_orientation : bool, default True (BUGFIX 2.5.1 -- was False).',
    '        sym_angle_weight : fixed weight on the Q3 approach-angle symmetry-\n'
    '                           tying restraint (only used when use_shape_alm).\n'
    '                           Same symmetric donor groups as sym_weight, but\n'
    '                           ties approach angle instead of M-D distance --\n'
    '                           deliberately a separate weight since the two\n'
    '                           quantities have different natural scales (Å² vs\n'
    '                           rad²). See ik/ik_solver.py\'s JointIKSolver 2.4.0\n'
    '                           changelog. Starting default, not empirically\n'
    '                           fit. Default 1.0.\n'
    '        q3_mode_commit   : bool, default True (2.8.0). When a symmetric\n'
    '                           donor group\'s Q3 target is genuinely multi-\n'
    '                           modal, commit the whole group to a single GMM\n'
    '                           component (chosen from the initial pose) for\n'
    '                           the whole solve, instead of letting\n'
    '                           sym_angle_weight tie members toward a mean\n'
    '                           angle that can fall between two modes. See\n'
    '                           ik/ik_solver.py\'s JointIKSolver 2.5.0 changelog.\n'
    '                           False restores pre-2.8.0 behaviour byte-\n'
    '                           identically -- useful for A/B comparison.\n'
    '        diversify_orientation : bool, default True (BUGFIX 2.5.1 -- was False).',
)

# ---------------------------------------------------------------------------
# EN3: joint_solver.solve(...) call site -- pass q3_mode_commit
# ---------------------------------------------------------------------------
do_replace(
    "EN3-joint-solver-solve-call",
    "            ideal_directions=ideal_directions_full_t if use_shape_alm else None,\n"
    "            symmetric_groups=symmetric_groups if use_shape_alm else None,\n"
    "        )",
    "            ideal_directions=ideal_directions_full_t if use_shape_alm else None,\n"
    "            symmetric_groups=symmetric_groups if use_shape_alm else None,\n"
    "            q3_mode_commit=q3_mode_commit,\n"
    "        )",
)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
