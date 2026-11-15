# F-4 evaluation contract

**Declared:** 2026-10-28, before F-4 results.

## Cohort and inputs

Use the existing F-2 census (`census_f2.jsonl`): all 683 scoreable source ligand instances across the cal/tail cohort and surface-fit sample, plus any failed source attempts retained in the raw census (the known ZOLQIC failure makes 684 unique `(parent, ligand)` attempts). Count ligands and unique parents with at least one kappa donor whose formal charge is negative. Tail is reported separately but is a subset of cal; do not double-count it in the union. Record any missing or failed input rows. Do not use `val`, reserve, or locked parents.

## Embed-level screen

Compare four arms on the full census, with identical ligand graphs, donor indices, E/Z variants, coordinates/seed, and cache-cold execution per ligand-arm:

1. Production (`embed_bond_check=false`, `embed_li_cap=false`).
2. Stretch-only guard (`embed_bond_check=true`, `embed_li_cap=false`).
3. Li counter-ion refinement (`embed_bond_check=false`, `embed_li_cap=true`).
4. Combined (`embed_bond_check=true`, `embed_li_cap=true`).

Score each arm using F-2's symmetry-aware frozen-coordinate binary label and fixed thresholds. Report the full per-ligand transition table relative to production; repairs are baseline bad → arm good, regressions are baseline good → arm bad. Include named cases QEZVEY, OHECUZ, FIMCAH, BOCRAN, LOPCAW and EBAYIQ; failed refinements/embeds; guard fallback used; and wall time per attempted ligand. The guard triggers only for a bonded ligand-atom stretch where `d_after - d_before > 0.25 Å`, or where `d_after > 1.25 * (r_cov,i + r_cov,j)` while `d_before <=` that threshold. Shortenings do not trigger. The second refinement constrains offending bond lengths to their pre-refinement values alongside donor-pair constraints; if it fails or remains invalid, return the pre-refinement geometry.

**Embed-screen gate:** an arm proceeds to step 4 only when it repairs at least one baseline-bad ligand and regresses no more than 1% of baseline-good ligands it touches. The denominator is the count of baseline-good ligands for which that arm returns a scoreable label. An arm that fails this gate stops, and the failure is reported.

## Solve-level screen (only arms passing the embed gate)

Before any full solve, estimate runtime on two footprint parents. Run unrelaxed production solves on footprint parents in both cal and tail at seed 11, with an arm and production control paired on the same machine; add seeds 12–13 only if the estimate fits the assigned compute budget. Use runkit metrics. No GFN2 relaxation is part of this screen.

Primary outcome: paired top-1 `ΔE_unrel` (arm minus control), mean and median over parents, with parent-clustered bootstrap 95% CI. Secondary outcomes: floor `ΔE_unrel`, obscene rate (>200 kJ/mol), worst slot, and top-1 RMSD to reference. List every parent with top-1 `ΔE_unrel` worsening by more than 25 kJ/mol, with the ligand-level cause.

**Solve-screen gate:** the paired mean's CI upper bound must be < +2 kJ/mol, and the arm must either improve floor `ΔE_unrel` or obscene rate with a CI excluding zero, or repair a named failure without a guard breach. Otherwise it fails.

## Robustness and package checks

Count and list ETKDGv3 failures reaching the deterministic retry path on cal, tail and the surface-fit sample. For ZOLQIC, check the bench runner and runkit trope runner handle an empty failed ensemble and continue; inspect the A4 tool by reading its source on the `jgen/a-census` worktree only. If runkit or bench code is changed, run their direct checks and the applicable trope suite. For the default-off package, run both profile golden checks and the runkit trope suite, recording fingerprints and any deviations.

Stereo is design-only: cross-reference the 23 unassigned-centre ligands with the 14 F-2 stereo/E-Z mismatch ligands, count binding-induced donor stereocentres as specified in the directive, and recommend implementation timing. Do not implement stereo enumeration.

## Decision and reporting

Keep both experiment flags off by default. A passing arm may be proposed for owner review; no default or fingerprint change is authorized by this contract. If no arm passes, package the flags off, deterministic retries, and any needed empty-result handling fixes. Record spec issues separately in `F_EMBED_REPORT.md`, including any mismatch between the prescribed check and what the code can measure without changing the metric.
