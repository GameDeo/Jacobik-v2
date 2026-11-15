# DS Step 5 (part): fixed-pool DB comparison and stress cost, 2026-11-11

Only the DB varies; generator/runtime are the D3 ones. Pools: saved D3 `D`-arm pools (seeds 11-13, generated under the recorded D2). Control = D2
(= D0, reproduced exactly). Arms are the Step 4 DBs (`DS_STEP4_RESULTS.md`). Evaluator: recorded MACE row energies, statistical scores without bend prior,
argmax per seed, `confidence.evaluate` (severe = > 25 kJ/mol adverse top-1 or a new break on >= 2 seeds). Scripts: `fixed_pool_db_compare.py`,
`fixed_pool_attribute.py`; seed pools of different size are padded with rows that cannot be selected (`stress_trial._score`).

## Selected-row energy change vs D2 (kJ/mol, seeds 11 / 12 / 13; negative = lower energy = better)
| case | pool rows | D-sample | D-data | D-both | D-final |
|---|---:|---|---|---|---|
| LUQVOK | 32 | 0 / 0 / 0 | +14.5 / -131.8 / -129.0 | -40.3 / -131.8 / -129.0 | -40.3 / -131.8 / -129.0 |
| SESWIZ | 28 | 0 | 0 | 0 | 0 |
| LIPDEU | 32 | 0 / -33.1 / 0 | 0 / -33.1 / 0 | 0 / -33.1 / 0 | 0 / -33.1 / 0 |
| YABYEJ | 16 | 0 | 0 | 0 | 0 |
| LOPJEF | 112 | 0 | -49.1 / -68.9 / 0 | -49.1 / -68.9 / 0 | -49.1 / -68.9 / 0 |
| ZAQNUE | 16 | 0 | 0 | 0 | 0 |
| HOXJUZ | 15 | 0 | **+18.5 / +96.0 / +15.2** | **+18.5 / +96.0 / +15.2** | **+18.5 / +96.0 / +15.2** |
| USAPOU | 35 | 0 | 0 | 0 | 0 |
| UMUBAF | 32 | 0 | 0 | 0 / 0 / +5.3 | 0 / 0 / +5.3 |

No row is severe under the registered rule; no new broken coordination anywhere. Three arm-rows have a single-seed event (HOXJUZ seed 12, +96). UMUBAF's
pool is valid by atom/site signature but the repaired-target pool is still to be generated (Step 6). MIGTEE and EFOXII have no saved pool.

- The LUQVOK improvement (the erroneous 2.66 A Pd-O component gone) and LOPJEF are caused by the extraction change, not the sampler (D-sample 0).
- **LIPDEU's -33 kJ/mol on seed 12 appears already under D-sample (sampling change only, no extraction change):** the broad QO fallback entry moves with a
  resample alone, not with the data fix.
- **HOXJUZ regresses in every arm that carries the extraction change.** Attribution (`attr_HOXJUZ_Dfinal.json`): one looked-up key of 34 (20 differ between the DBs)
  explains all three seeds, `Q1:('Ni', 1, 0, 'S_thiocarbonyl', 'square_plan', 'N_donor')`. It has 6 direct observations, identical in both extractions; the entry is
  `adapted` (sparse), so the change comes through its ancestors. D2: means (2.155, 2.445) var (0.0012, 0.149), weights (0.93, 0.07) - a broad second component. DS:
  means (2.148, 2.223) var (0.0008, 0.0022), weights (0.85, 0.15) - narrow. The source Ni-N distance of that donor is 2.11 A. D2's selected rows have that donor at
  2.92-2.94 A (broken against the 1.25 x source rule, but already so in the clean selection, so not a counted new break) and are 18-96 kJ/mol lower in MACE energy;
  DS selects rows with the donor at 2.30-2.86 A. The trade is therefore between lower MACE energy with a stretched donor and a tighter donor with higher energy; the
  registered energy metric reads it as a regression. Not tuned; reported.

## Stress design cost (workstation, stable sampler, D-final base)
- Null refit reproduces the base DB entry exactly for LUQVOK `Q1` (136 s) and LIPDEU `QO:(2, 4)` (28 s). Both need `JACOBIK_PROFILE=d2` plus the arm's overrides in the
  environment before any jacobik import (otherwise the legacy sampler is used silently); `stress_run.py` sets them.
- Q1 trial about 100 s (expansion 42 s + closure fit); chain-restricted expansion does not help for LUQVOK because the chain's top ancestor feeds all 608,220 rows
  (verified identical, no speed-up). QO trial about 30 s. Fixed-pool scoring 0.3 s. Peak RSS 2.3-3.9 GB per worker.
- Executable trials per key: observed_input_error + parent_removal, doses {1, ceil 1%, 2%, 5%}, 32 replicates: LUQVOK Q1 (99 parents) 192 trials = 5.3 core-hours;
  LIPDEU QO (3,128 parents) 256 trials = 2.1 core-hours. The `geometry_0.02A` scenario (96 and 128 further registered trials) still has no executor and is recorded as failed.

## Seed dependence of the fixed-pool outcome (`fixed_pool_seed_spread.py`)
Gap = selected-row MACE energy minus the pool's lowest energy, per seed (kJ/mol). Reference D2 vs D-final.
| case | D2 gaps (s11/s12/s13) | D-final gaps | spread D2 -> D-final |
|---|---|---|---|
| LUQVOK | 40 / 132 / 129 | 0 / 0 / 0 | 91 -> 0 |
| SESWIZ | 189 / 0 / 195 | 189 / 0 / 195 | 195 -> 195 (DB-independent) |
| LIPDEU | 38 / 74 / 44 | 38 / 41 / 44 | 36 -> 6 |
| YABYEJ | 0 / 3 / 11 | 0 / 3 / 11 | 11 -> 11 |
| LOPJEF | 82 / 69 / 48 | 33 / 0 / 48 | 34 -> 48 |
| ZAQNUE | 20 / 0 / 5 | 20 / 0 / 5 | 20 -> 20 |
| HOXJUZ | 4 / 16 / 14 | 23 / 112 / 29 | 11 -> 89 (see below) |
| USAPOU | 0 / 0 / 0 | 0 / 0 / 0 | 0 -> 0 |
| UMUBAF | 0 / 1 / 0 | 0 / 1 / 5 | 1 -> 5 |
Mean gap 41.3 -> 29.5 kJ/mol (median 14.2 -> 4.7); mean seed spread 44.5 -> 41.6; cases with all three seeds at the pool floor 1 -> 2 of 9.
The DB improves the typical selection but barely changes seed dependence; what remains is mostly pool content and ranking (SESWIZ: seeds 11 and 13 select a row ~190
kJ/mol above the floor with 11 lower-energy rows in the pool, identical under both DBs).

HOXJUZ: in all three pools the lowest-energy rows are broken (donor beyond 1.25 x source). In seeds 11 and 13 both DBs select a broken row (the nearest properly
coordinated row is thousands of kJ/mol higher); only seed 12's pool contains a usable properly coordinated row, which D-final selects (gap 0 to the valid floor) while D2
selected a broken row 96 kJ/mol lower. The "+96" is the price of the coordination, and the seed jump is whether the generator produced that basin in the seed's pool.

## Workstation check (2026-11-11, commit fa1f585 tree)
`tools.check ws`: fingerprint 5b56969afb9c PASS; golden 0/490 arrays differ; pytest 701 passed, 14 failed, 5 skipped in 1,425 s. The 14 failures are all `tests/test_finish_phase.py::test_phase_cuda_capture[...]`, the historical CUDA parity failures (same count as the earlier 664 passed / 14 failed baseline; they are not touched by this branch). The level reports FAIL because of them; it is stated as such, not weakened.

## Stress pilot, refit on Perlmutter + scoring on the saved pools (2026-11-11)
Job 61154103 (regular, 1 CPU node, 14 min wall of the 30 min cap, rc 0, commit `466445e`, output `$SCRATCH/jacobik/runs/ds-stress-20261111-2019`). Base = D-final.
Both null refits reproduce the base DB entry exactly (LUQVOK Q1 153 s, LIPDEU `QO:(2, 4)` 30 s). Scenarios `observed_input_error` (affected parents' rows replaced by the same
parents' rows of the uncorrected extraction E_off) and `parent_removal`; doses 1 parent / ceil 1%, 2%, 5%; 32 replicates; seed 11. Geometry trials not yet run (executor `42a5b20`).
Refit cost under 48 workers per case: Q1 mean 162 s (peak 4.1 GB), QO 33 s (2.4 GB).

| key | trials | scored ok | severe | single-seed | any energy change | interval95 (0 of 32 per cell) |
|---|---:|---:|---:|---:|---:|---|
| LUQVOK Q1 (99 parents) | 192 | 192 | 0 | 0 | 0 | [0, 0.107] each cell; censored at 5.05% |
| LIPDEU QO (3,128 parents) | 256 | 253 | 0 | 0 | 2 (0.0 kJ, selection tie-break only) | [0, 0.107] each cell |

**Failed trials (3 of 448), stated as failed.** LIPDEU `observed_input_error` replicate 8 at doses 32, 63, 157 raises "replacement evidence missing an affected parent": the same
parent XOCDEC has QO rows in the corrected extraction but none in the uncorrected one. Under the registered rule these block LIPDEU QO (`candidate_blocked: true`, reconciliation
incomplete). Run as a labelled amendment (a parent the uncorrected extraction did not produce is replaced by no rows), the three trials give 0.0 / 0.0 / 0.0 kJ/mol, not severe.
The amendment needs to be written into the contract (new version) before the blocked flag can be cleared.

**No severe outcome and no energy change in 445 scored trials: the fixed-pool selection is insensitive to this damage, at doses up to 5% of the key's support parents. This is a
tested sensitivity on three seeds of one pool per case, not a robustness guarantee.**

### F8: the fit itself is not stable even where the selection is
The refitted mixtures move a lot while the selected row does not:
- LUQVOK Q1: reverting 1 parent to its uncorrected rows turns the two-component fit (means 2.0115 / 2.0747, weights 0.22 / 0.78) into one broad component (mean 2.0665,
  var 0.0042) in 3 of 32 trials; 7 of 32 at 2 parents; 15 of 32 (47%) at 5 parents. Parent removal keeps two components (one trial each at n=2 and n=5 gives three).
- LIPDEU QO: the three-component fit loses a component in 2 of 32 observed-error trials at 32 parents, 1 of 32 at 157, and 1 of 32 removal trials at each of 32 and 63.
The component count is chosen by model selection and flips under small data changes. The pool does not expose it here, but a pool with rows between the modes would. Step 7 candidate
(component selection stability / a minimum-separation or fixed-k option), not changed here.

### Ranking-level read-out of the pilot (2026-11-11; `stress_rank.py`)
The pilot's severity rule looks only at the selected row. This read-out compares the whole fixed pool (three seed pools of 32 rows each) under the refitted entry with the clean
scores: Spearman rank correlation, top-10 overlap, rank of the clean top-1, and the change in mean energy of the top-10 (positive = adverse). Same trials, no new refits.
- **LUQVOK Q1:** parent removal leaves the ranking essentially unchanged (Spearman >= 0.999). Observed-input error moves it where the fit collapses to one component: min
  Spearman 0.94 at n=1 and 2, 0.83 at n=5; top-10 overlap down to 0.6; the clean top-1 is never displaced. Top-10 mean energy never rises; it falls by up to 15 kJ/mol.
- **LIPDEU QO:** the pool is nearly flat under this key (score range 0.1-0.2 per seed, top1-top2 margin 0.002-0.011), so the ranking is easily reordered: in a few
  observed-input trials at 32 and 63 parents Spearman goes to -0.20 and the clean top-1 drops to rank 10. Top-10 mean energy again never rises (down to -39 kJ/mol).
- **Correction to the earlier wording:** "the selected structure does not change" holds for LUQVOK only. In LIPDEU the selected row does change in some trials; what holds is
  that no trial selects a higher-energy or newly broken structure.
- **Limits:** each pool is 32 rows per seed, three seeds, one case per key. A pool with rows at both modes (Step 6) is the test of whether a weight shift lets a different
  mode dominate; this pool is too narrow to say.

### Geometry-perturbation trials (J1, job 61158374, scored locally 2026-11-11; contract v2 A2)
`geometry_0.02A`: the affected parents are re-extracted from coordinates perturbed by at most 0.02 A, the entry refitted (pipeline DF code, exact null at 0 A), then scored on the same saved pools.
LUQVOK Q1: 96 trials; LIPDEU QO: 128 trials; all scored, **0 failed, 0 severe**; per dose (32 trials each, doses 0.03-5% of the key's support parents) the 95% upper bound on the severe rate is 10.7% (Wilson, 0 of 32).
No key is blocked. As for the other scenarios, this is a tested sensitivity on three seeds of one pool per case, not a robustness guarantee. With this, 669 of 672 registered trials are scored (445 of 448 pilot, 224 of 224 geometry) (the 3 XOCDEC trials are the A1 case); none is severe.
Local files: `research/data/ds_proto/pipeline_20261111/stress/geometry/*.records.summary.json`.
