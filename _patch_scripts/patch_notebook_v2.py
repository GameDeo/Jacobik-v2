#!/usr/bin/env python3
"""
Update jacobik_test_suite.ipynb's "Known open issues" cell: the Q2/Q3
cos(angle)-space GMM refit landed (was previously listed as outstanding),
and record the newly-audited Q2 cis/trans key-pooling finding as a
separate, deferred item.
"""
import json

PATH = "script_notebooks/jacobik_test_suite.ipynb"

with open(PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def src_of(i):
    return "".join(cells[i]["source"])


issues_idx = None
for i, c in enumerate(cells):
    if "## Known open issues" in src_of(i):
        issues_idx = i
        break
assert issues_idx is not None, "Known open issues cell not found"

OLD_STALE_LINE = (
    "- The Q2/Q3 cos(angle)-space GMM refit (both quantities are bounded\n"
    "  [0,180] deg via acos, not truly circular, so VMM is the wrong model\n"
    "  family) is still outstanding -- not yet implemented.\n"
)
assert OLD_STALE_LINE in "".join(cells[issues_idx]["source"]), "stale Q2/Q3 line not found verbatim"

NEW_LINES = (
    "- **2.7.0 update**: Q2 (D-M-D angle) and Q3 (donor approach angle) are\n"
    "  refit as GMM-on-cos(angle) instead of VMM (both are bounded [0,180]\n"
    "  deg via arccos and never wrap to 360, so they were never circular --\n"
    "  see stats/fitter.py's fit_gmm_cosangle() docstring and ik/scorer.py's\n"
    "  2.7.0 changelog entry for the full rationale). Confirmed post-refit:\n"
    "  e.g. Q3 for C_carbonyl/Mo/other now has a dominant component at\n"
    "  ~175 deg with cos-space var ~1.8e-5 (sigma ~1.5 deg) and weight 0.72\n"
    "  -- much sharper than the old VMM fit allowed near a hard boundary --\n"
    "  which should directly help the \"ligands not pointing straight\"\n"
    "  symptom for linear/near-linear approach donors like CO. If TBP-vs-\n"
    "  square-pyramidal ranking or ligand-orientation issues persist after\n"
    "  this, the remaining known lever is folding the shape-deviation\n"
    "  signal (`solve_trace[\"anchor_loss\"]` under `use_shape_alm=True`,\n"
    "  section 12) into final conformer ranking -- currently ranking uses\n"
    "  only the Q1-Q6 statistical score, excluding shape fidelity entirely.\n"
    "- **Audit finding (2.7.0, not yet addressed)**: Q2's key construction\n"
    "  (`_graph_dist_class` in stats/key_builder.py) pools ALL cross-ligand\n"
    "  donor pairs under one `\"different_ligand\"` key regardless of their\n"
    "  actual cis/trans relationship -- unlike Q1, which has a proper\n"
    "  `trans_partner`-aware key (`_trans_partners()`, based on measured\n"
    "  angle > 140 deg). Confirmed identical at both fit-time (key_builder.\n"
    "  py) and score-time (ik/scorer.py's Q2 construction block reuses the\n"
    "  same `_graph_dist_class` call). Consequence: for any homoleptic\n"
    "  monodentate complex (PtCl4, Fe(CO)5, Co(NH3)6, ...), every pairwise\n"
    "  D-M-D angle shares ONE fitted distribution -- a pooled mixture of\n"
    "  cis-like and trans-like observations -- so Q2 can only shape the\n"
    "  aggregate angle histogram, not enforce which specific pairs should\n"
    "  be cis vs trans. This is a real, significant limitation (part of\n"
    "  why the shape-ALM/CShM term in 2.3.0 was needed in the first place)\n"
    "  but a separate, larger fix than the model-family refit above --\n"
    "  would need Hungarian-assignment info threaded into Scorer.__init__\n"
    "  for score-time-aware keys, plus new fit-time key logic. Deferred.\n"
)

old_full = "".join(cells[issues_idx]["source"])
new_full = old_full.replace(OLD_STALE_LINE, NEW_LINES, 1)
assert new_full != old_full

# Re-split into a source list the same way the rest of the notebook uses
# (one string per line, each ending in \n except possibly the last).
lines = new_full.splitlines(keepends=True)
cells[issues_idx]["source"] = lines

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print(f"Updated 'Known open issues' cell at index {issues_idx}.")
