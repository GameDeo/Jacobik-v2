# DS evaluation contracts, Steps 5, 6 and 8 — version 2 (2026-11-11)

Version 1 (`DS_STEP5_6_8_CONTRACT.md`) is unchanged. This file adds to it. Items marked **post-hoc** were written after the Step 5a/5b pilot results were open, and say so.
Items marked **prospective** were written before any Step 6 or Step 8 generation result exists. Locked sets are not read; eligibility uses only `locked_exclusions.json`.

## A. Amendments to Step 5 (post-hoc, after the pilot)

**A1. Missing replacement (post-hoc).** In `observed_input_error`, a parent that the uncorrected extraction (E_off) did not produce for the stressed quantity is replaced by no rows
(the parent was never observed under the old rules, so reverting it means it contributes nothing). Under v1 the three LIPDEU trials with XOCDEC (replicate 8, doses 32, 63, 157)
failed and blocked LIPDEU QO. Under v2 they run, and the result is 0.0 kJ/mol, not severe. The v1 verdict ("blocked, three failed trials") is kept in `DS_STEP5_RESULTS.md`;
the v2 verdict ("not blocked") replaces it for qualification. This amendment was chosen after seeing the failure and does not change any selected structure; it is a bookkeeping rule.

**A2. Geometry scenario runs (prospective for the geometry trials).** `geometry_0.02A` is registered as in v1 (LUQVOK Q1 96 trials, LIPDEU QO 128 trials, seed 11) with the executor
of `42a5b20` (re-extraction of the affected parents from coordinates perturbed by at most 0.02 A, exact null at 0 A). Any geometry trial missing or failed blocks the key, as for the other scenarios.

**A3. Ranking read-out (post-hoc).** `stress_rank.py` reports Spearman, top-10 overlap, rank of the clean top-1 and top-10 energy change over the pool. It is informational:
it adds no pass/fail rule. It showed no adverse top-10 energy change in 1,335 seed-trials and displaced the clean top-1 in 6 of 759 LIPDEU seed-trials (never to higher energy).
The pools are 32 rows per seed; they cannot test whether a weight shift moves the winner between modes (see C).

## B. Candidate identity (prospective)
The candidate DB (**DF**) is `D-final` rebuilt from the retained E_assign tables at the gate branch tip `data/ds-gate` (master `54c19f4` + `data/ds-cleanup`), with its `.meta.json`
recording the code commit. The rebuild must equal the pipeline `D-final` key by key (`--check-reproduction --target-db`), else the difference is reported before anything else runs.
Runtime identity: `gate_pm.DF_FLAGS` (the d2 profile plus `donor_identity_reconcile`, `donor_assignment`, `stable_ancestor_sampling`, `structure_sanity`). The D3 harness refuses a DB whose
meta lacks a full 40-hex `code_commit`, flags and input hashes; the pipeline DBs have `code_commit: unknown` (they were built from a snapshot made before the commit-recording fix), which is
why DF is rebuilt rather than used as is.

## C. Step 6: generation and the existing P/S start (prospective)
- **Job:** one A100 node, every arm in the same job, one claim queue per phase, seeds 11-13, per-parent cap 5 min (a timeout is recorded, never retried).
- **Parents:** LUQVOK, SESWIZ, LIPDEU, YABYEJ, LOPJEF, ZAQNUE, HOXJUZ, USAPOU, UMUBAF, MIGTEE, EFOXII (development) and JEGPUJ, SIVBUV (diagnostic supplement, reported separately, no weight in the candidate rule).
  UMUBAF, MIGTEE and EFOXII get their pools from this job (new common-target pools; arms C and DF are generated together, so no saved pool is reused for them).
- **Phases:** `s6_C` (production DB, current start); `s6_DF` (DF, current start = arm D, and DF + P/S = arm DS); four damaged-trial phases, each with D and DS on the case's parent only.
- **Damaged trials (selection rule fixed here):** per stressed key and mechanism (`observed_input_error`, `parent_removal`), the registered trial with the lowest minimum-over-seeds Spearman in the pilot ranking read-out.
  The entry at the resolved key is replaced by the trial's refitted entry (`make_damaged_dbs.py`; pooled descendants are not re-derived, the same substitution the fixed-pool scorer makes).
  LUQVOK Q1: `observed_input_error` trial `da2ae96...` (two components to one), `parent_removal` `56836b4...` (two to three); LIPDEU QO: `568b2b2...` (Spearman -0.20), `800e9a5...` (three to two).
- **Metrics and the P/S candidate rule:** as v1 (top-1 dE, floor, coordination breaks, coverage, diversity, runtime; parent-clustered bootstrap). P/S stays a candidate only if, on DF, the upper 95% bound of the paired top-1 dE
  difference (DS minus D) is below +5 kJ/mol, its new-break rate is not higher, and no development case regresses by more than 25 kJ/mol on two seeds.
  The same rule is applied to the damaged phases separately and reported; a failure there is reported, and is not by itself a reason to drop P/S.
- **Attribution of a failure** (start, optimization, pruning, final ranking): from the recorded pool, pre- and post-pruning counts and identities in the run records; no new instrumentation in this job.

## D. Step 7 decision rule (prospective)
A solver or fit change is implemented only if one of these is demonstrated: a severe fixed-pool, geometry or generation trial; a Step 6 failure that the table in the plan attributes to one mechanism; or a gate failure
traced to one mechanism. None is demonstrated at v2. Hence, by this rule: no Step 7 implementation is made; F7 (Q5 single-component mode flips), F8 (component-count instability), the broad LIPDEU QO fallback and the
SESWIZ ranking-driven seed spread stay **unresolved, documented leads**, each with a minimal next experiment (D-line of `DS_STEP7_DECISION.md`). Step 6 or the gate can change this; if so, one mechanism-specific
implementation and one baseline are compared at a time.

## E. Step 8: the qualification gate (prospective)
- **Arms:** C = production (DB `constraints_2026-11-01`, no flags), D = DF. S (P alone on production) and DS (DF + P) are added **only if** Step 6's P/S candidate rule passes on DF; otherwise omitted.
  The arm names C, D, S, DS keep `tools.jgen.d3.analyse` usable unchanged (D here is DF).
- **Stages:** stage 1 = cal + tail minus `locked_exclusions.json` (120 parents; primary 118), seeds 11-13; stage 2 = the 110 `surface_fit` eta2 holdout parents, seeds 11-13. Stage 3 (promotion reserve) is not run here; it needs a stage 1 candidate, the auditor's code list and a separate owner go.
- **Targets:** identical audited targets for both arms: `--target-overlay` (`target_overlays_v1.json`, sha `9a9c820b...`) and `--gate-eligibility` (`gate_eligibility_v1.json`) enter the control identity. Primary = valid + repaired strata; the 23 ambiguous and 3 unrepaired
  parents stay visible as their own strata and are reported missing, not passed. Failed and missing parents stay in every denominator.
- **Analysis roots:** two per stage from the same records: `primary` (frozen eligibility, `in_corrected_primary`: 109 and 95 parents) decides; `all` (120 and 110) is reported beside it. (Found 2026-11-11 before any gate result: the D3 analysis did not read the eligibility file, which `tools.run` only hashes into the control identity; `gate_pm manifest --strata` now applies it.)
- **Control:** C is recorded fresh in the same job (no A100 record of production `5b56969afb9c` with this DB exists); cache not used.
- **Gates:** those of `contract_D3_2026-11-09.md` (energy, guard without the artefact list and catastrophic count, curvature, eta2 holdout, wall <= 1.3x), and, from Step 5, no severe registered trial. The artefact-list parents stay reported both ways.
  Historical D2 outcomes (the failed D3 gate) are kept separate and are not pooled with this run.
- **A failed gate stays failed.** The D3 ablation order is not run automatically; any ablation needs its own owner go.

## F. Jobs requiring the owner's go (none submitted)
| # | Job | Node, QOS | Cap | Est. billed |
|---|---|---|---|---|
| J1 | Rebuild DF at the gate tip with reproduction check, then the geometry stress trials (LUQVOK Q1, LIPDEU QO) | CPU, regular, 1 node | 45 min | ~0.6 node-h |
| J2 | Step 6 (generation, P/S, damaged trials) | GPU, regular, 1 node (4 A100) | 40 min | ~0.5 node-h |
| J3 | Step 8 gate (stage 1 and 2; with-P variant if Step 6 passes) | GPU, regular, 1 node (4 A100) | 60 min | ~0.6-1.0 node-h |
J1 must finish first (J2 and J3 read DF). J3 reads J2's candidate-rule verdict for the with-P choice. After J3: curvature gate on the workstation (CPU), then the owner decides fold-in.
