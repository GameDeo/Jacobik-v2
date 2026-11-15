# G-3d progress

## 2026-11-01 — Step 1

Owner-approved merge fast-forwarded jgen/b-solver from 498226d to master
347496f; no merge commit was needed. Canonical replay regeneration unchanged.
Default production fingerprint: 273f2ccbf892. xtb resolves in the pixi environment.
Shared-reference golden check, default worker layout: production 245/245,
u1 245/245, zero differences, exit 0. No golden data written.
Finish-phase file run once under runkit CodexB GPU lock, deterministic algorithms
and CUBLAS_WORKSPACE_CONFIG=:4096:8: 31 passed in 525.30 s (wall 525.843 s).
No skips or failures. Local RTX 2060 SUPER, driver 595.99.02, driver CUDA 13.2.
Logs: shared research/data/jgen/campaign/B/g3d/{golden,finish_phase_tests}.log.
Colab spend: zero. No step-0 or panel results generated.

## 2026-11-01 — Contract and step-0 audit stop

Contract/provenance committed before diagnostics: f1276b3. Conditional arm
identities frozen as the exhaustive family indexed by the measured policy;
no arm selected. Base verification commit 5ba445a.

Step 0 started with the original O1 identity manifest (coordinates/energies
not reused), CPU seed 11, OMP/MKL/OpenBLAS threads 1 and native one-worker
layout. HEDJOP failed: fixed row [None, 1, 148] unavailable at switch.
No substitution or retry. Complete-cohort rule returns
incomplete_no_policy_selection: release, sigma source and cap remain unset.
The contract explicitly refers incomplete cohorts to audit. The runner
continued into FEKKAJ after recording HEDJOP, then was terminated by exact
PID 1297716 (SIGTERM, exit 143) to honor the directive's stop point.
FEKKAJ interrupted; NUWFAM/QEZVEY/DAMVAQ/USAPOU/NEPNEF/AJAQAD not attempted.
XOJHIN diagnostic not started. All eight expected parents remain in the ledger.
No clean24, cal or tail runs; no comparative energies or gate conclusions.
No replay/embed fallback message in the diagnostic log. Colab spend zero.
Evidence: g/G3D_STEP0_STOP.json plus shared g3d/step0.json and step0.log.
Await auditor decision on unavailable fixed identity; no changes to scientific
rules, row replacement, or policy inference from incomplete inputs.

## 2026-11-01 — Amendment 1, row selection, embed-fallback stop

Applied audit_jgen_G3d_step0_2026-11-01.md. New row selection used only returned
scores, seed 11, CPU default worker layout, production 273f2ccbf892, 250 steps.
Manifest: g/G3D_ROWS_273f2ccbf892.json, with per-row keys/scores and sorted
step-200 chunk digests. Seven parents returned six rows each (42 total),
including all six policy parents (36/36); NEPNEF failed and has shortfall six.
No old/new key correspondence recorded. Selection wall for successful parents:
HEDJOP 12.208 s, FEKKAJ 61.630 s, NUWFAM 34.172 s, QEZVEY 77.645 s,
DAMVAQ 12.851 s, USAPOU 26.245 s, AJAQAD 10.249 s.

capture_switch now records missing or changed keys per row, preserving other
rows. It retains the default worker layout and maps chunk-local alive indices
to the global trial keys. Policy accounting implements >=30/36 and >=3 per
policy parent; denominators exported. Focused tests: 5 passed in 0.15 s,
including row-level failure continuation and chunk offset restoration.
No production solver arithmetic changed. The old stop record is retained.

STOP: NEPNEF selection triggered FallbackWatch: 1 embed fell back to MMFF
("GFN2 embed refinement failed"). Directive section 4 lists embed fallback
as a stop point; amendment 1 did not waive it for NEPNEF. The selection loop
recorded this failure and continued to AJAQAD, preserving the full batch ledger.
No NEPNEF row was accepted, no retry or tuning. No resumed step-0 measurement,
XOJHIN diagnostic, clean24 or cal/tail run started. Colab spend zero.
Selection log: shared research/data/jgen/campaign/B/g3d/row_selection.log;
manifest retains the precise fallback exception and traceback. Await audit.

## 2026-11-01 — Amendment 2 executed; convergence-cap audit referral

Committed NEPNEF exemption as contract amendment 2 (13ec8c9), then diagnostic
ledger/progress handling (3cb2e97). Five focused tests passed in 0.04 s.
Resumed with committed new-base row manifest; no row reselection or retry.
HEDJOP completed in 417.689 s. To use the allocated CPU capacity, remaining
parents were distributed over three diagnostic processes plus an idle
supervisor, keeping fixed-row capture's default worker layout. Raw scheduler,
logs and completed parent records are preserved in shared g3d/.

STOP at directive step 3's contract audit referral: every one of HEDJOP's
16 registered release-set/sigma alternatives has a selected row censored at
50 finish steps. Thermal all-release first-convergence observations are
[45,38,39,35,48,null]. All-release and shape+symmetry have one censored row
for each sigma source; every other alternative has six. The registered
censor-aware cap function returns None for all 16 alternatives (asserted).
Remaining parents cannot remove this witness. No cap from a converged subset,
no longer continuation, and no claim that the unknown cap exceeds the budget.
No release set or sigma source selected from partial policy data.

Complete full diagnostic records: HEDJOP, NUWFAM, DAMVAQ (18 policy rows).
FEKKAJ/QEZVEY/AJAQAD each captured six rows and finished fixed-row variants;
native timing interrupted at referral. USAPOU not attempted. NEPNEF remains
known-invalid with embed_fallback=true, zero accepted rows, amendment-2
exemption. No other fallback reported. XOJHIN deferred at the cap stop.
The incomplete-policy threshold is not treated as a failed scientific gate:
execution stopped earlier on the independently established cap referral.

Terminated supervisor PID 1311601 and workers 1308927/1311630/1311631 by
SIGTERM; verified no such PID remains. No clean24/cal/tail or GPU evaluation.
Colab spend zero. Report G3D_REPORT.md; exact row-level witness, full ledger,
completed native timing and artifact SHA256s in g/G3D_CAP_STOP.json. Partial
merged full records: shared g3d/step0_partial_at_cap_stop.json. Prior accepted
stop records retained. Await auditor ruling; no policy or optimizer chosen.

## 2026-11-01 — Amendment 3: complete 50-step policy measurements

Contract amendment 3 committed before resuming: f92e23a. Instruments and
component checkpointing: 84487c0, switch-digest/COD trace checks: 37afb2e.
Six focused tests passed, including the new censored nearest-rank p95 rule.
Reused full HEDJOP/NUWFAM/DAMVAQ records. Interrupted FEKKAJ/QEZVEY/AJAQAD
numeric measurements had not been serialized, so missing records were
regenerated; new component caches retain every completed chunk and trace.
USAPOU measured; NEPNEF invalid-input embed_fallback=true, exempt as amended.
XOJHIN separate R2s diagnostic completed with 250 COD geometry/closure records.

All six policy parents yield six rows: 36/36, completeness PASS. No nonexempt
fallback. Existing single-release rules select symmetry only: median gains
shape 0.0439101, symmetry 0.0551353, projection 0.0426060; class changes are
2/36, 1/36, 2/36 respectively. Shape/projection FAIL the <=5% class-change
condition; symmetry PASS. Joint selected-set class-change share is 1/36.
Curvature cost fraction 14.5639393 versus limit 0.1: cost gate FAIL, choose
thermal widths. No discretionary release/sigma selection or retuning.

Full 50-step results: shared g3d/step0_50_complete.json. Corrected only its
reused-parent metadata list (mutable-dict aliasing in the scheduler); the
original bytes are preserved as step0_50_complete_before_metadata_correction.json.
All scientific measurements unchanged. Resume elapsed 3226.521 s; Colab 0.
Decision frozen in g/G3D_POLICY_50.json before the 144-step continuation.
The old 50-step cap field is superseded by amendment 3, not treated as a gate.

## 2026-11-01 — Amendment 3 convergence result: incompatible cap, STOP

Selected policy frozen at 6027475: release symmetry only, thermal widths.
Ran the 144-step convergence-only continuation from each policy row's native
step-200 state, preserving the arm budget of 250. All six parents and all
36 rows completed; no missing keys, assignment changes or nonexempt fallback.
All default-worker chunk digests matched the committed row manifest. All
first-50 convergence observations matched the prior measurements.

22/36 rows converged; 14/36 remain right-censored at 144. Parent censor counts:
HEDJOP 6/6, USAPOU 6/6, NUWFAM 1/6, QEZVEY 0/6, DAMVAQ 1/6, AJAQAD 0/6.
The ceil(0.95*36)=35th order statistic is censored (only 22 observed values),
so the cap exceeds 144. Admissible-cap gate FAIL. No finite/subset-only p95,
no correction, extension, release-set change or tuning. Per amendment 3 and
the directive, stop and report; clean24/cal/tail not started.

Continuation wall 159.091 s; missing 50-step records resumed in 3226.521 s,
with completed HEDJOP/NUWFAM/DAMVAQ records reused. Separate XOJHIN R2s COD
trace completed (250 geometry/closure observations). NEPNEF remains exempt
invalid input with embed_fallback=true and no accepted row. Colab spend zero;
no GPU evaluation or VMs. All continuation jobs exited normally; no owned
compute job remains. Owner-requested 20-minute/completion waits were used.

Step-count bias reported plainly: the step-200 switch is more settled than
plateau switching near 100, so the measured cap may be optimistic; no correction.
Full report G3D_REPORT.md; row-level result, exact decisions, descriptive
parent-clustered CIs, case summaries and raw SHA256s in g/G3D_STEP0_FINAL.json.
Previous accepted-stop artifacts remain intact. Await auditor checkpoint.

## 2026-11-02 — Amendment 4, independent-arm screen setup

Addendum 3 accepted: FA/FR+ recorded failed at step 0; admissible-cap gate
remains failed, without revisiting criterion/horizon/quantile. Clean24 will
contain C1/R2s/IA only, resolved thermal widths and unchanged 250-step budget.
R2s is whole-run iRprop-; IA switches iRprop+ at 100. No adaptive cap used.

Runtime now permits cap omission only for whole-run/fixed-switch modes.
Adaptive finish still requires a cap. Five light finish-phase tests pass,
including exact update/state parity across all 250 steps against the legacy
cap-bearing configuration for R2s and IA. Canonical replay regeneration
unchanged; default fingerprint 273f2ccbf892. No production default arithmetic
changed; no full golden/test-file repeat. Exact flags/fingerprints frozen in
g/G3D_ARMS_AMEND4.json: C1 273f2ccbf892, R2s b2f758e72507, IA 4ad46b7b7722.

Before any GPU result, found bench runner missing finish_phase in both the
per-arm module setter and the no-control-reuse predicate. Fixed both so real
phase arms run in both solver modules, including reset on C1. Added regression
checks. Filter accounting now contains three arms and preserves FA/FR+ as
step-0 failures. No scientific gate relaxed. Colab spend remains zero.

### 2026-11-02 — First paired GPU profile

AJAQAD, seeds 11–13, C1/R2s/IA in one invocation under CodexB runkit GPU lock.
All nine records status ok; expected fingerprints; all replay chunks mode
replay, 250 steps, zero fallback. Parent wall 41.0 s; job wall 61.988 s.
Solver+assembly per arm (batched seeds, count once): C1 4.43 s, R2s 5.67 s,
IA 4.95 s. Full-pool xz export and distortion diagnostics retained. No energy
advance gate evaluated on this performance profile. A larger second parent
QEZVEY is profiled before the clean24 estimate and panel start.

### 2026-11-02 — Clean24 run frozen and wall estimate

Second profile QEZVEY: parent 582.8 s; job 595.630 s. Solver+assembly C1
106.59 s, R2s 181.02 s, IA 167.83 s (count batched seeds once). All nine
records ok, correct fingerprints, replay mode, 250 steps, zero fallback.
GPU batches 4608 rows; 1729 unique structures evaluated in batched MACE.
Together with AJAQAD (41.0 s), arithmetic-mean planning estimate is 7485.6 s
(2.08 h) for 24 parents, or 6861.8 s (1.91 h) for remaining 22. High parent
heterogeneity disclosed; no energy outcomes used for timing/parent selection.

Reuse both paired profile runs as two fixed clean24 parents; each already
contains C1/R2s/IA in the same invocation, seeds 11–13, full xz exports and
distortion records. Verified flags/fingerprints, same GPU/machine, identical
bench runner SHA, 250 steps and zero fallbacks. Originals preserved; copied
artifacts into clean24 collection, original manifest paths retained. This
avoids rerunning completed units. G3D_CLEAN24_INPUTS.json freezes the exact
24-parent cohort, reused record hashes, remaining 22 and estimate before the
panel-scale run. All outcomes will be analysed once together. No gate tuned.

### 2026-11-02 — Clean24 complete; no arm advances, STOP

Collection frozen before panel-scale run at a46b9da. Reused paired AJAQAD and
QEZVEY records; remaining 22-parent job completed normally, wall 922.841 s.
Total across both profile jobs and remaining-panel job: 1580.459 s (26.34 min),
versus conservative 2.08 h planning estimate. Same RTX 2060 SUPER, all arms
paired in each invocation, seeds 11–13, full-pool xz exports, GPU lock held,
nvidia-smi captured before every start. All jobs ended and released the lock.
Colab spend zero; no VMs. Completion waits used 20-minute timers as requested.

216/216 records valid; 72 finite paired units and 24 complete parents per
candidate. Zero missing/failure/fallback units; every recorded replay chunk
mode replay. All returned rows have finite angle and coverage geometry:
C1 2510, R2s 2507, IA 2402. Actual replay steps show some early stops;
nominal budget stays 250. Fixed-budget counterfactual wall was not measured,
and no rescue timing run was started after the failed filter.

Mechanical advance predicates, full fixed cohort:
- R2s: median paired top-1 -11.535444 kJ/mol PASS; mean floor -25.541356 PASS;
  angle p95 89.495565° vs C1 35.170910° (+54.324656°): distortion FAIL.
- IA: median top-1 -9.208721 PASS; mean floor -25.969371 PASS;
  angle p95 58.925079° (+23.754170°): distortion FAIL.
C1+10° threshold is 45.170910°. Advanced arms: none. This is the directive
step-4 stop point. No criterion change, rerun or parent exclusion rescues it.
FA/FR+ remain failed at step 0, not filter losers. Cal/tail not started and
step-5 wall estimate/gates are unexecuted. No production choice or fold-in.

Mean top-1 deltas with parent-clustered 95% CIs (10,000, seed 11):
R2s -30.087409 [-60.750744,-7.109019]; IA -29.000592 [-53.876097,-10.119727].
Energy improvements do not override the distortion guard. Full sensitivity,
obscene/class/coverage diagnostics, per-seed NUWFAM and XOJHIN tables, wall
accounting and limitations are in G3D_REPORT.md. FEKKAJ/NEPNEF are outside
clean24; their step-0 records/exemption remain in G3D_STEP0_FINAL.json.
Full pool parent coverage is 15/24 in all arms; automorphism enumeration was
capped on 21/72 units per arm and is disclosed. Cause of step-0 cap failure
remains unknown. Exact predicates in G3D_FILTER_RESULT.json; secondary metrics,
case tables and all 216 raw-record SHA256s in G3D_SCREEN_FINAL.json.
Checkpoint: commit and await audit. Later L4 confirmation still required for
any future fold-in. Production defaults unchanged, fingerprint 273f2ccbf892.
