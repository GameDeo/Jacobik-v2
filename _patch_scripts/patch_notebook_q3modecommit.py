#!/usr/bin/env python3
"""
Add a q3_mode_commit=False A/B row to section 12's shape-ALM comparison, so
the new 2.5.0/2.8.0 mode-committed Q3 initialization can be directly
compared against having it off (mirrors the sym_angle_weight=0.0 pattern).
Since q3_mode_commit defaults to True, the existing "NEW shape-ALM
(default)" row already exercises it -- this adds only the OFF variant plus
one at a reduced-multimodality-sensitivity control isn't needed here.
Also refreshes "Known open issues" with the real EDTA diagnosis (Q3
multimodality, not the sym_angle_weight fix from 2.4.0/2.7.0, which the
user's own re-run showed had negligible effect) and the separate Q1
charge-clamp fix (2.9.0).
"""
import json

PATH = "script_notebooks/jacobik_test_suite.ipynb"

with open(PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def src_of(i):
    return "".join(cells[i]["source"])


ab_idx = None
for i, c in enumerate(cells):
    if "_shape_ab_variants" in src_of(i):
        ab_idx = i
        break
assert ab_idx is not None, "shape-ALM A/B variants cell not found"

OLD_VARIANTS = '''    "NEW shape-ALM, sym_angle_weight=3.0": dict(use_shape_alm=True, sym_angle_weight=3.0),
}'''
assert OLD_VARIANTS in src_of(ab_idx), "variants dict text not found verbatim"

NEW_VARIANTS = '''    "NEW shape-ALM, sym_angle_weight=3.0": dict(use_shape_alm=True, sym_angle_weight=3.0),
    # 2.5.0/2.8.0: mode-committed Q3 initialization. Root cause found AFTER
    # the sym_angle_weight rows above shipped: sym_angle_weight ties
    # symmetric donors toward their MEAN Q3 angle, but O_carboxylate's
    # fitted Q3 GMM (Co/octahedral) is genuinely 5-modal (~87-130 deg,
    # confirmed against the live constraints.json) -- for EDTA's 4
    # carboxylate O (one RDKit-symmetric group), the mean of two
    # well-separated modes lands in the low-density valley BETWEEN them,
    # which explains why sym_angle_weight=0/1/3 all looked similarly bad in
    # the first re-run (this row's default sibling above, "NEW shape-ALM
    # (default)", now has q3_mode_commit=True baked in, since it defaults
    # True in engine.py -- this row isolates the effect by turning it off).
    "NEW shape-ALM, q3_mode_commit=False (mode-commit OFF)": dict(
        use_shape_alm=True, q3_mode_commit=False
    ),
}'''
new_src = src_of(ab_idx).replace(OLD_VARIANTS, NEW_VARIANTS, 1)
assert new_src != src_of(ab_idx)
cells[ab_idx]["source"] = new_src.splitlines(keepends=True)

# ---------------------------------------------------------------------------
# Refresh "Known open issues" cell.
# ---------------------------------------------------------------------------
issues_idx = None
for i, c in enumerate(cells):
    if "## Known open issues" in src_of(i):
        issues_idx = i
        break
assert issues_idx is not None

OLD_ISSUES = "".join(cells[issues_idx]["source"])
MARKER = (
    "- **Audit finding (2.7.0, not yet addressed)**: Q2's key construction\n"
)
assert MARKER in OLD_ISSUES, "expected marker line not found"

NEW_LINE = (
    "- **2.5.0/2.8.0 update -- the actual EDTA fix (supersedes the\n"
    "  2.4.0/2.7.0 sym_angle_weight note below, which the user's own\n"
    "  re-run showed had negligible effect on bond_spread_A)**: traced the\n"
    "  regression to a live query against constraints.json --\n"
    "  Q3(O_carboxylate, Co, octahedral) is a REAL 5-component GMM (cos-\n"
    "  space modes -0.638/-0.486/-0.439/-0.028/+0.045, i.e. ~87-130 deg),\n"
    "  reflecting genuinely different carboxylate binding modes pooled\n"
    "  from the CSD, not a fitting artifact. EDTA's 4 carboxylate O are\n"
    "  one RDKit-symmetric group, so each independently settles near\n"
    "  whichever mode it's nearest to, and sym_angle_weight's mean-tie can\n"
    "  land the group in the low-density valley BETWEEN two well-\n"
    "  separated modes -- worse than either mode alone, and consistent\n"
    "  with n_converged collapsing (29 -> 1-2/32) and being insensitive to\n"
    "  sym_angle_weight's value (0/1/3 all similarly bad): 'converged'\n"
    "  means score-plateaued, and a tug-of-war between modes can oscillate\n"
    "  rather than plateau. Fix: q3_mode_commit (default True) hard-\n"
    "  selects ONE GMM component per symmetric group per trial from the\n"
    "  INITIAL pose, instead of letting members drift and then averaging.\n"
    "  See ik_solver.py's 2.5.0 changelog for the full mechanism. NOT YET\n"
    "  VALIDATED against a live solve at the time of writing (implemented\n"
    "  from static constraints.json inspection + code-path tracing only,\n"
    "  since Claude cannot run torch here) -- run this section's new\n"
    "  `q3_mode_commit=False` row against its default-True sibling on\n"
    "  EDTA to confirm before trusting it.\n"
    "- **2.9.0 fix (separate, Q1-only)**: query-time Q1 keys used the\n"
    "  complex's raw, unclamped charge, but the live constraints.json only\n"
    "  contains charge in {-1, 0, 1} for every metal. Any |charge|>1 case\n"
    "  (e.g. [Co(NH3)6]3+, charge=+3) could never match any Q1 candidate\n"
    "  that still conditions on charge -- confirmed via a live trace:\n"
    "  co_nh3_6_octahedral resolved at fallback level 6/10 (Co, N_donor),\n"
    "  losing donor-subclass AND geometry specificity together, despite\n"
    "  being an extremely well-represented CSD case. Now clamped to the\n"
    "  nearest available bucket (scorer.py's _clamp_q1_charge()); verified\n"
    "  against the live DB post-fix: co_nh3_6_octahedral now resolves at\n"
    "  level 0/10 (exact match, n=958), pt_cl4 at level 0/9 (was 5/9),\n"
    "  zn_cl4 at level 2/9 (was 5/9). Unrelated to the EDTA/Q3 fix above;\n"
    "  mentioned here only because it was found via the same diagnostic\n"
    "  trace the user posted.\n"
)
new_issues = OLD_ISSUES.replace(MARKER, NEW_LINE + MARKER, 1)
assert new_issues != OLD_ISSUES
cells[issues_idx]["source"] = new_issues.splitlines(keepends=True)

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print(f"Updated A/B variants cell at {ab_idx}, issues cell at {issues_idx}.")
