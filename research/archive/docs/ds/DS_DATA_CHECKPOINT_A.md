# DS data package — Checkpoint A, 2026-11-10

Implemented the two opt-in changes and the confidence measurement framework
on data/ds-cleanup. The branch carries the current stereo production source
and the retained D2 pipeline. Default fingerprint is 5b56969afb9c.

Integration note: this branch starts from investigation commit 2e1ca9f and
therefore includes the earlier D2 pipeline/runner work. Four stereo fold-in
commits were carried over to align the production source. This is not a
minimal diff from master; review the DS commits and their D2 dependencies
before owner integration. No shared branch was merged or pushed.

## Donor correction

The source path now supports H-aware graph alignment, local terminal O/S
identity reconciliation and preservation of source-supported κ donors after
false C–X face removal. It preserves distinct declared donor sites. Ambiguity
and weak declarations produce review events.

The final census completes 138/138 source tasks across 46 sources. All 11
watched Pd parents replace their 22 problematic primary acetate O choices
at 2.633–2.709 A with local O atoms at 2.017–2.095 A. The baseline extraction
matches all 44 retained Q1 rows for those parents. All 36 available default-off
source signatures match their recorded baseline. Exact audited donor-target
matches improve from 17/34 to 29/34, with no previously matched target lost
and no newly introduced non-audited atom across the 34 fixtures.

The source pilot also restores DUCPAT's two dropped O donors. LUQVOK's input
sites stay κ-C,C,O,O. Final source evidence and the correction ledger are under
research/data/audit/2026-11-10_DS/donor_reconcile_final in the main checkout.
The detailed donor report is research/bench/jgen_campaign/ds/DONOR_RECONCILE_REPORT.md.

Some audited targets retain a weak secondary O alongside a primary O.
Reconciliation and the earlier overlay can disagree on κ versus κ² for those
cases (AGAPEG and ILEPOE). BADQIH and JAYCEV need broader ring-mapping review;
NUHPEL has a false η3 assignment outside this rule's scope.
These need an explicit eligibility decision before a full corrected build.
Weak-declaration events alone are insufficient grounds for excluding a fit
observation. Automatic observation quarantine has not been enabled.

## Sampling and confidence

Fresh Q1–Q5 and QO rows carry observation IDs based on source identity,
original key and occurrence before filtering. Measured values are excluded
from the ID. Ancestor selection uses a bounded hash-priority heap for each
bucket, preserving the existing cap. Output order is canonical. The legacy
path and the Q6 raw reservoir retain their previous sampling algorithms.
Stable mode rejects missing or duplicate IDs. Build metadata records the
identity and sampling versions.

The replay annotates retained raw Q1 rows with IDs before any exclusion,
using the production sampling code. It verifies all 250 watched buckets
whose eligible input rows remain identical when the charge filter changes.
**Zero buckets change sample membership**, compared with 73 under the
legacy sampler. This diagnostic backfill is explicit. New stable DB builds
require fresh extraction-time IDs.

The confidence module implements independent-parent clustering, nested
one-parent/1%/2%/5% random damage designs, bounded source-coordinate
perturbations, source-supported observation replacements, independent Q1
refits, paired three-seed severe-failure assessment, floor changes, Wilson
intervals and censored tested tolerance. A severe event on two seeds can
combine an energy loss on one seed with broken coordination on another.
Empty evidence is rejected; execution failures block acceptance and prevent
reporting a censored pass. Sparse Q1 keys require their real
ancestor refits and are rejected by the independent-fit helper.

The fit pilot exactly reproduces the earlier 71-parent near-O refit.
For this support size there are 96 trial definitions, since the one-parent
and 1% doses coincide. Structural failure rates require the four DB builds,
fixed-pool scoring and selected generation confirmations in Checkpoint B.
The framework currently consumes frozen scored-trial evidence through
tools.run diagnostic. Full quantity-specific stress orchestration remains
part of that next checkpoint.

The retained census covers 612,336 Q1 observations from 125,044 source
parent codes, representing 108,980 independent parents after LnQM sibling
clustering. Its 3,021 carboxylate observations above 2.4 A are review
triggers, including legitimate long metal–O contacts. The complete raw
corpus sweep is reserved for the approved extraction/build job.

## Validation and next audit

Focused combined tests passed 56 cases with two slow checks skipped.
The revised donor/profile suite passed 39 cases; the final confidence,
sampling and census suite passed six cases. The CPU build script passes
bash syntax validation. Workstation fingerprint and golden checks passed:
5b56969afb9c, zero differences across 490 arrays. The golden check took
508.3 seconds.

Complete pytest coverage: 54 files, 683 cases. The original run recorded
663 passed, 15 failed and five skipped. After the separately recorded
stereo assertion repair/recheck below, the final disposition is **664 passed,
14 failed, five skipped**, with **zero new unresolved failures** against the
prior investigation. All 14 unresolved failures are the same CUDA-capture
numeric parity cases in test_finish_phase. The whole test suite therefore
remains **failed**; these failures were neither hidden nor relaxed.

The current stereo fold-in exposed a stale replay test assertion: scores
and coordinates matched exactly, but NumPy's flat `equal_nan` comparison
cannot handle nested `variant_traces` dictionaries. The same failure was
reproduced at production commit 54c19f4 in a detached control checkout.
The assertion now traverses nested traces and retains exact numeric
comparisons; both MELLUN and AHIGOQ pass (38.83 seconds). Raw suite evidence
and the separate production-control/recheck JUnit files are preserved.

Evidence index, relative to main-checkout
research/data/audit/2026-11-10_DS/:

- checkpoint_A/protocol.json (SHA256
  49196049d3934507d9749d63378bc96c3a3b31b5fc9c0b1e31053d0c44a2b55a)
- donor_reconcile_final/correction_ledger.json and census_summary.json
- checkpoint_A/stable_q1_replay.json
- checkpoint_A/retained_q1_census.json and confidence_fit_pilot.json
- checkpoint_A/ws_golden/ws_20261110-203316.json
- checkpoint_A/pytest/combined_result.json (untouched original test results)
- checkpoint_A/pytest/final_result.json (explicit targeted recheck disposition)
- checkpoint_A/current_stereo_trace_control.xml and stereo_trace_recheck.xml

The frozen protocol is checkpoint_A/protocol.json under the main
investigation evidence directory. It includes MIGTEE and EFOXII, selected
from the smallest historical absolute D deltas, alongside the six losses
and three gains. DS_DATA_CONTRACT.md records the test contract.

tools/d2/ds_data_build.sbatch prepares two rounds of paired 64-worker CPU
builds for D0/D-data/D-sample/D-both. D0 checks reproduction against recorded
D2. The measured prior two-build time was 11:44. The new four-build estimate
is 24–32 minutes including additional identity/sampling work, with a
40-minute cap. It requires a clean git checkout and fresh scratch output.
Submission requires the owner's per-job go for regular QOS.

Checkpoint A stops for audit of the correction ledger, unresolved source
eligibility, tests and build estimate. Billed compute and Colab spend are zero.
