# DS evaluation contracts — Steps 5, 6 and 8 (DRAFT for owner/auditor freeze), 2026-11-11

Written before any result exists. Amendments after the freeze are new versions, never edits. Locked sets are not read: eligibility below uses only
`locked_exclusions.json` (reserve and motion-validation codes) and the frozen gate eligibility.

## Eligibility check (done)
None of the development cases (LUQVOK, SESWIZ, LIPDEU, YABYEJ, LOPJEF, ZAQNUE, HOXJUZ, USAPOU, UMUBAF, MIGTEE, EFOXII) nor JEGPUJ/SIVBUV is in the
locked reserve or motion-validation exclusion list. JEGPUJ is in the `tail` regression panel (development evidence); SIVBUV is a named D3 case. Both may
serve as the Step 6 placement/collapse diagnostic supplement; they carry no pass/fail weight.

## Step 5 — fixed-pool attribution and fit sensitivity
- **Arms:** C (production 2026-11-01), D2 (recorded), D0, D-data, D-sample, D-both (Step 4). Only the DB varies; generator and runtime settings are the D3 ones.
- **Pools:** saved D3 pools (seeds 11-13) for targets whose atom/site/assignment signature is unchanged (`cross_score._signature`). Repaired targets
  (UMUBAF and any case whose corrected input changes the signature) get new common-target pools generated once with all arms in the same job.
- **Evaluator:** the recorded MACE row energies of the pool, identical across arms; statistical scores without bend prior, identical function for every arm.
  Reproduction gate: native C/D scores must reproduce the recorded top/floor scores after the bend prior (tolerance 5e-4) before any attribution is read.
- **Stress design:** per resolved key and target, `registry.register` with scenarios `observed_input_error`, `parent_removal` (and `geometry_0.02A`, which the driver
  refuses until coherent re-extraction exists and records as failed), doses 1 parent and ceil(1%, 2%, 5%), 32 replicates, 64 parent bootstraps, seed 11.
  Pilot scope: LUQVOK Q1 and LIPDEU `QO:(2, 4)`; expand only after the measured cost (see `DS_STRESS_PILOT.md`).
- **Broken coordination:** `explicit_boolean_per_seed_row.v1`; a row is broken when any metal-donor Q1 distance exceeds 1.25 x the source distance. Only a break that
  is absent in the clean selection counts.
- **Pass/fail:** severe = adverse top-1 change > 25 kJ/mol or a new break on >= 2 of seeds 11-13. A DB is blocked from qualification if any registered trial is severe at
  the first tested dose, or any registered trial is missing/failed. Single-seed events, Wilson 95% intervals and the first failing fraction are reported; no failure at the maximum
  tested dose is stated as censoring, not as robustness. Parent-clustered bootstrap intervals for every arm difference.
- **Outputs:** per case/arm: selected and floor energy, rank of the C-selected row, dependent-key attribution (marginal one-key swaps with the interaction remainder), dose-response tables.

## Step 6 — existing S/P start candidate
- **Arms (not crossed with every DB):** current initialization vs `start_kabsch` + `start_kabsch_pivot` (other settings fixed) on (a) the selected corrected DB and (b) preselected
  damaged trials from Step 5 (one per mechanism: observed error, removal). Matched compute: same seed budget and wall-clock budget; actual compute recorded.
- **Cases:** the development cases above plus the diagnostic supplement JEGPUJ and SIVBUV (placement/collapse; reported separately, no gate weight).
- **Metrics:** top-1 dE vs reference, available energy floor, coordination (new breaks), coverage, diversity, runtime. Parent-clustered bootstrap CIs.
- **Candidate rule:** P/S stays a candidate only if, on the corrected DB, the upper 95% bound of the paired top-1 dE difference (P/S minus current) is below +5 kJ/mol, the
  new-break rate is not higher, and no development case regresses by more than 25 kJ/mol on two seeds. Otherwise it is recorded as not qualifying; it is not tuned further here.
- **Attribution for failures:** record initial targets/poses, assignments, source mode, candidate identities and pre/post-pruning trajectory to locate the loss (start, optimization,
  pruning, final ranking).

## Step 8 — corrected qualification gate
- Identical audited targets for both arms; control identity is target-aware (`--target-overlay`, `--gate-eligibility` hashes). Primary = valid + repaired strata; ambiguous and
  unrepaired strata stay visible; failed or missing parents stay in every denominator.
- **The 26 reference-incompatible targets are exactly the 23 ambiguous and 3 unrepaired parents** (stage 1: 8 ambiguous + 3 unrepaired; stage 2: 15 ambiguous);
  every valid and repaired parent (the corrected primary set, 100 + 9 and 70 + 25) is reference-compatible. No regeneration is needed for the primary comparison; the
  26 stay visible as their own strata and are reported missing, not passed, until a decision on their targets exists.
- Frozen energy, guard, catastrophic, coordination/coverage, diversity, runtime and failure-accounting requirements are those of the D3 contract; historical D/DS outcomes stay separate.
  Severe Step 5 failures block this step. Locked qualification/reserve access follows auditor custody; the owner decides fold-in.
