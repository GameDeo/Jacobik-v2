# Locked and held-out sets (custody: auditor) — drawn 2026-10-27

This file states which data is held out, for whom, and the rules for using it. **Partition membership is not in the repo.** The auditor holds it in `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/locked/locked_sets_2026-10-27.json` (sha256 `550e50c06ee75edd4d777ca2dd8acf3f797a42762d660f30e28c12373abba2f2`). Development and training code gets only the exclusion list `locked_exclusions.json`, which is in this folder.

## Source
- Frame: `data/refit/frame_v1/frame.json` (also accessible via symlink `research/data/refit_nci_v2/scaled_v1/frame.json`), sha256 `a3c34f75…7c03`. That is the same `frame_sha256` that every bench record carries. The relevant field is `main_allocation.splits`:
  - `surface_fit`: 1,200 groups;
  - `calibration`: 120;
  - `motion_validation`: 120;
  - `reserve`: 240.
- **Eligible reserve groups:** the 240 reserve groups minus those that contain a repair replacement code (20), a manifest task code (0), or a `src/fit/data/benchmark_holdout_codes.json` code (0). That leaves **220 groups**, the same count as `roadmap_generation_v1.md` B4.
- **Draw:** shuffle within each stratum (seed `20261027`), interleave the strata, then take a disjoint prefix partition.

## Partitions (disjoint)
| Set | Groups | Owner | Use |
|---|---:|---|---|
| `nn_locked_confirmation` | 60 | NN plan §4.2 and §7 | The locked confirmation, opened **once**, after the checkpoint, DB, code, thresholds and budget are frozen. If a confirmation fails and the result informs a revision, the set becomes development evidence. |
| `jgen_promotion_reserve` | 80 | jgen (roadmap B4) | The promotion panel for jgen solver changes, run **once** per promotion decision after the `cal` gate passes. It needs its own control-vs-control repeat (roadmap §1.4.4). |
| `paper_final_test` | 80 | paper | The final held-out benchmark for the paper, run **once**, on the final production code, by the method in `research/paper/BENCHMARKS.md`. |

Stratum coverage: NN 23 strata, jgen 18, paper 17. The strata are imbalanced in size, so small strata land in at most one set.

## Status of the other splits
- **`calibration` (`cal`):** development data; tuning is allowed.
- **`tail`:** a regression panel of 10 parents (`research/bench/tools/jgen/bench/panels.py`). **Note:** 5 of its 10 parents come from `motion_validation` (IFUROQ, SOBHAS, XEYTUT, YUCZUS, ZELTAP). It has been studied heavily and is development evidence.
- **`motion_validation` (`val`):** **no longer unseen.** The tail and failure diagnostics have read it (roadmap, "Split rule"). Do not tune on it. jgen may evaluate on it only once, as a confirmation after `cal`. It must never be reported as a held-out result in the paper.
- **`surface_fit`:** the NCI v2 refit training split. It is fine for jgen development, but anything the statistics DB was fitted on is not held out for score evaluation.
- **`src/fit/data/benchmark_holdout_codes.json`** (N48, RING10, DEV32): never fitted on. It is used as development benchmarks, so it is not a clean test set either.

## Caveat: the statistics DB has seen these parents (found 2026-10-27)
The production database (`data/db/constraints*.npz`, fitted by `research/archive/code/refit_top_legacy/` and `src/fit/`) is fitted on **all of tmQM/LnQM except `src/fit/data/benchmark_holdout_codes.json`**. So every reserve parent's geometry is part of the fitting data of the score's statistics tables (Q1–Q6, QO).
- For the NN locked confirmation and for jgen promotion this is acceptable: they test the solver and the correction on unseen *solves*, with the DB held fixed.
- For the **paper's held-out claim**, it is not enough. Before `paper_final_test` is opened, either:
  1. refit the DB with all three locked partitions added to the exclusions. The fitting pipeline reproduces the DB bit-identically, so this is cheap, but it gives a new DB and a new production fingerprint; it is best done together with a fold-in; or
  2. state plainly in the paper that the test parents contributed to the statistics, and add an external test set that never entered tmQM.

  **Owner's decision (2026-10-27): option 1, done at the final end-to-end refit before the release build.** The locked partitions are added to the refit exclusions then. Until then the effect is judged negligible: these are statistical tables over about 86k complexes, not a neural network.

## Rules
1. No agent reads the auditor's partition file. Code that builds training data or tunes anything **must exclude** every code in `locked_exclusions.json` (all reserve codes plus all `val` codes) and assert that it did.
2. Opening a locked set needs the owner's go-ahead. The auditor records the date, the code commit, the fingerprint and the result in this file under "Openings".
3. Once a set has informed a decision, it is spent for that purpose. Draw replacements only from data never studied, and log the draw here.

## Openings
(none yet)
