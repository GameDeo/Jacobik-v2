#!/usr/bin/env python3
"""
Add a "Shape-ALM A/B test (2.6.0)" section to jacobik_test_suite.ipynb,
testing both PtCl4 (shape collapse) and EDTA (bond-length spread) under
use_shape_alm=False (old, pre-2.6.0 anchor) vs True (new, default).
Also fixes a pre-existing duplicate "## 11." heading and refreshes the
"Known open issues" cell.
"""
import json

PATH = "script_notebooks/jacobik_test_suite.ipynb"

with open(PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def src_of(i):
    return "".join(cells[i]["source"])


# Sanity: verify the cells we're about to touch are what we expect.
assert "## 11. Q2 aggregation A/B test" in src_of(28), src_of(28)[:80]
assert "## 11. Debugging playground" in src_of(30), src_of(30)[:80]
assert "CASE_NAME = \"co_edta_octahedral\"" in src_of(31), src_of(31)[:80]
assert "## Known open issues" in src_of(32), src_of(32)[:80]

MD_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 12. Shape-ALM A/B test (2.6.0)\n",
        "\n",
        "Tests the `use_shape_alm` mechanism added in `engine.py` 2.6.0 /\n",
        "`ik_solver.py` 2.3.0 (Augmented-Lagrangian shape constraint via a\n",
        "per-step-recomputed optimal rotation + a symmetry-tying restraint on\n",
        "topologically-equivalent donors), which **replaces** the old\n",
        "fixed-absolute-target anchor by default (`use_shape_alm=True`).\n",
        "\n",
        "Runs both reported-broken cases -- PtCl4 (shape collapse) and EDTA\n",
        "(bond-length spread between its two chemically-equivalent amine N\n",
        "donors) -- under the OLD mechanism (`use_shape_alm=False`, byte-\n",
        "identical to pre-2.6.0) and the NEW one (default), side by side.\n",
        "\n",
        "See the changelog entries in `engine.py` (2.6.0) and\n",
        "`ik/ik_solver.py` (`JointIKSolver` 2.3.0) for the full rationale.\n",
        "`shape_rho`/`shape_tol`/`sym_weight` are starting defaults, not\n",
        "empirically fit -- if shapes still fail here, that's the first thing\n",
        "worth sweeping.",
    ],
}

CODE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "_shape_ab_cases = [\n",
        '    CASES_BY_NAME["pt_cl4_square_planar"],\n',
        '    CASES_BY_NAME["co_edta_octahedral"],\n',
        "]\n",
        "\n",
        "_shape_ab_variants = {\n",
        '    "OLD anchor (use_shape_alm=False)": dict(use_shape_alm=False),\n',
        '    "NEW shape-ALM (default)": dict(use_shape_alm=True),\n',
        '    "NEW shape-ALM, shape_rho=20": dict(use_shape_alm=True, shape_rho=20.0),\n',
        "}\n",
        "\n",
        "_shape_ab_rows = []\n",
        "for _case in _shape_ab_cases:\n",
        "    for _label, _kwargs in _shape_ab_variants.items():\n",
        "        _result, _elapsed = run_case(_case, **_kwargs)\n",
        "        _checks = run_all_checks(_result, _case)\n",
        "        _mol = _result.best.complex\n",
        "        _donor_idxs, _dists = donor_metal_distances(_mol)\n",
        "        _shape_ab_rows.append({\n",
        '        "case": _case["name"],\n',
        '        "variant": _label,\n',
        '        "shape_pass": _checks["shape"]["pass"],\n',
        '        "classified_as": _checks["shape"]["actual"],\n',
        '        "bond_spread_A": _checks["bond_spread"]["spread_A"],\n',
        '        "dangling_arm_pass": _checks["dangling_arm"]["pass"],\n',
        '        "best_score": _result.best.score,\n',
        '        "n_converged": len(_result.converged),\n',
        "        })\n",
        "    plot_dmd_angles(_result, _case)\n",
        "\n",
        "_shape_ab_df = pd.DataFrame(_shape_ab_rows).set_index([\"case\", \"variant\"])\n",
        "_shape_ab_df",
    ],
}

# Insert the two new cells right after index 29 (the Q2-aggregation A/B
# test's code cell), i.e. before the old "## 11. Debugging playground".
insert_at = 30
cells[insert_at:insert_at] = [MD_CELL, CODE_CELL]

# Fix the pre-existing (unrelated) duplicate "## 11." heading on the
# debugging-playground section, now shifted to insert_at + 2.
dbg_idx = insert_at + 2
assert "## 11. Debugging playground" in src_of(dbg_idx), src_of(dbg_idx)
cells[dbg_idx]["source"] = [
    s.replace("## 11. Debugging playground", "## 13. Debugging playground")
    for s in cells[dbg_idx]["source"]
]

# Refresh the trailing "Known open issues" cell to reflect 2.6.0.
issues_idx = insert_at + 4
assert "## Known open issues" in src_of(issues_idx), src_of(issues_idx)
cells[issues_idx]["source"] = [
    "## Known open issues (context for anything that fails above)\n",
    "\n",
    "- **2.6.0 update**: the old fixed-absolute-target anchor (which went\n",
    "  stale as soon as the ligand's pose or torsions moved from its initial\n",
    "  Kabsch fit) is now replaced by default with an Augmented-Lagrangian\n",
    "  shape constraint recomputed every step, plus a symmetry-tying\n",
    "  restraint on topologically-equivalent donors. Run section 12 above\n",
    "  first on any new failure -- if `use_shape_alm=False` (old) also\n",
    "  fails the same way, the bug predates this change and isn't the shape/\n",
    "  anchor mechanism; if only the new path fails, `shape_rho`/`shape_tol`/\n",
    "  `sym_weight` are the first knobs to sweep (they're starting defaults,\n",
    "  not empirically fit).\n",
    "- Q2's arithmetic-mean pairwise-angle aggregation still dilutes a single\n",
    "  badly-wrong D-M-D pair across all pairs (documented in scorer.py's own\n",
    "  changelog). The `q2_aggregation=\"power_mean\"` experimental alternative\n",
    "  (section 11) is now largely superseded by the shape-ALM term for\n",
    "  *overall polyhedron shape* -- Q2 itself still governs the finer-grained\n",
    "  angular statistical preference within a shape, so the dilution issue\n",
    "  there is not fully closed by 2.6.0.\n",
    "- The Q2/Q3 cos(angle)-space GMM refit (both quantities are bounded\n",
    "  [0,180] deg via acos, not truly circular, so VMM is the wrong model\n",
    "  family) is still outstanding -- not yet implemented.\n",
    "- `anchor_floor_frac` is now only honoured on the OLD (`use_shape_alm=\n",
    "  False`) fallback path; the new dual multiplier makes a manual floor\n",
    "  unnecessary on the default path.\n",
    "- Haptic (multi-atom) donor groups are skipped by the new symmetry-tying\n",
    "  restraint's equivalence detection -- untested for that case.",
]

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print(f"Inserted 2 cells at index {insert_at}. New cell count: {len(cells)}")
