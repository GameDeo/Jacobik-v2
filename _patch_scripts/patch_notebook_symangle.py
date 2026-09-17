#!/usr/bin/env python3
"""
Add sym_angle_weight A/B variants to the existing shape-ALM A/B test
section (12) so the new 2.4.0/2.7.0 Q3 approach-angle symmetry tie can be
directly compared against having it off, on EDTA specifically (the case
it's meant to fix). Also refreshes the "Known open issues" cell.
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

OLD_VARIANTS = '''_shape_ab_variants = {
    "OLD anchor (use_shape_alm=False)": dict(use_shape_alm=False),
    "NEW shape-ALM (default)": dict(use_shape_alm=True),
    "NEW shape-ALM, shape_rho=20": dict(use_shape_alm=True, shape_rho=20.0),
}'''
assert OLD_VARIANTS in src_of(ab_idx), "variants dict text not found verbatim"

NEW_VARIANTS = '''_shape_ab_variants = {
    "OLD anchor (use_shape_alm=False)": dict(use_shape_alm=False),
    "NEW shape-ALM (default)": dict(use_shape_alm=True),
    "NEW shape-ALM, shape_rho=20": dict(use_shape_alm=True, shape_rho=20.0),
    # 2.4.0/2.7.0: Q3 approach-angle symmetry tie (see ik_solver.py's 2.4.0
    # changelog) -- added to address the EDTA "uneven N anchoring /
    # carboxylate spread" regression reported after the Q2/Q3 GMM-cos
    # refit. sym_angle_weight=0.0 reproduces pre-2.4.0 behaviour exactly
    # (angle tie off) so it's a clean A/B against the "NEW shape-ALM
    # (default)" row above, which now has sym_angle_weight=1.0 baked in.
    "NEW shape-ALM, sym_angle_weight=0.0 (angle tie OFF)": dict(
        use_shape_alm=True, sym_angle_weight=0.0
    ),
    "NEW shape-ALM, sym_angle_weight=3.0": dict(use_shape_alm=True, sym_angle_weight=3.0),
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
    "- **2.4.0/2.7.0 update**: added a second symmetry-tying restraint\n"
    "  (`sym_angle_weight`, default 1.0) that ties Q3 approach angle\n"
    "  between topologically-equivalent donors, alongside the existing\n"
    "  M-D distance tie. Root cause: EVERY Q3 entry in the live\n"
    "  constraints.json turned out to have been running through a broken\n"
    "  score-time fallback (kappa clamped to 10) for its whole history --\n"
    "  independent of, and prior to, the periodicity issue the Q2/Q3\n"
    "  refit above fixed. Once that refit gave Q3 real teeth for the\n"
    "  first time, nothing stopped two chemically-identical donors (e.g.\n"
    "  EDTA's two amine N) from landing near different Q3 modes -- a\n"
    "  plausible mechanism for the uneven-N-anchoring/carboxylate-spread\n"
    "  regression reported after that refit. Run section 12's new\n"
    "  `sym_angle_weight=0.0` vs default rows on EDTA to confirm this\n"
    "  actually helps before trusting it -- NOT YET VALIDATED against a\n"
    "  live solve (implemented from static constraints.json inspection\n"
    "  only, since Claude cannot run torch here). `sym_angle_weight=1.0`\n"
    "  is a starting default, not empirically fit.\n"
)
new_issues = OLD_ISSUES.replace(MARKER, NEW_LINE + MARKER, 1)
assert new_issues != OLD_ISSUES
cells[issues_idx]["source"] = new_issues.splitlines(keepends=True)

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print(f"Updated A/B variants cell at {ab_idx}, issues cell at {issues_idx}.")
