# Package H-3 checkpoint report

Date: 2026-10-31  
Branch: `jgen/h-patch`  
Status: **Checkpoint H-3 reached; evaluation gates failed.** No fold-in or promotion.

## Execution

Compared production control against one combined arm on the same NVIDIA L4 VM (`jgH3`, machine `72098c84986f`), with seeds 11–13. The frozen manifest was `research/data/refit_nci_v2/scaled_v1/manifest.json` (SHA-256 `5c3e5897f099eac36eb52bf14b67ecd8fd299e3819c83c8d0d5daa0d0f0ce450`). The control fingerprint was `6cc586e17ab1`; combined was `d587e9503943`. Both ran with `finish_phase=None`.

- Control: production defaults.
- Combined: `q6_donor_pairs={"hh":0.95,"ho":0.90,"oo":0.80}`, `angle_tail_quadratic=true`, `embed_haptic_li_cap=true`.
- The Colab bundle was `f876447404aec04254f2340be8389f25ec74e19f`. The record field `code_git_commit=e89aea2b87b86eba17903dde7980c8496240917e` is the synthetic commit created by `cj.py` after unpacking the archive; `bundle_commit` records the source revision.
- Colab spend: 37.95 compute units total; final balance 492.84; the audit VM used for a replay attempt was never provisioned (zero active assignments, no additional VM spend).
- Full-pool exports were fetched before VM shutdown. There are 780 records (768 valid; 12 NEPNEF failures), 29,128 XYZ files, and 109,465,299 bytes across the H-3 output tree.

## Primary endpoint and gates

Paired delta is combined minus control top-1 MACE energy; positive values are worse. Bootstrap resamples parents and keeps each parent’s three seeds, 10,000 resamples, RNG seed 11.

| Panel | Valid pairs | Mean delta (kJ/mol) | Median | 95% parent-clustered CI | Primary gate |
|---|---:|---:|---:|---:|---|
| Cal | 357 (119 parents) | +1.17 | -0.01 | [-8.39, +13.22] | **FAIL**; upper bound must be < +2 |
| Tail | 27 (9 parents) | +1.42 | +0.92 | [-10.97, +15.82] | **FAIL**; upper bound must be < +2 |

The intervals do not establish that the mean effect is harmful. They fail the pre-registered non-worsening bar because each upper bound exceeds +2 kJ/mol.

## Pre-registered guards

- **More than +25 kJ/mol worse on at least two seeds — FAIL.** Cal: LOPJEF `(+32.07, -2.42, +31.71)` and NUWFAM `(+1634.96, -40.15, +135.02)`. Tail: IFUROQ `(+40.79, +44.61, +49.29)`. Each tuple is seeds 11, 12, 13. Their Q1–Q6 top-1 term breakdowns for both arms are in [`H3_GUARD_TERMS.csv`](../../../../bench/jgen_campaign/archive/H3_GUARD_TERMS.csv). Terms were recomputed with the CPU scorer from the exported top-1 XYZ; the audit CSV retains the recomputed-vs-recorded score residual (maximum absolute residual 0.0192 score units), so the attribution values are reviewable with that limitation visible.
- **FEKKAJ and NUWFAM must improve — FAIL.** FEKKAJ changes were `(-301.54, -321.87, -249.20)` kJ/mol; mean `-290.87` (improved). NUWFAM changes were `(+1634.96, -40.15, +135.02)`; mean `+576.61` (not improved).
- **XOJHIN, reported separately.** Changes were `(-6.17, +26.29, -17.80)`; mean `+0.77` kJ/mol.
- **Top-1 >200 kJ/mol rate — cal PASS, tail FAIL.** Cal: control 50/357 (14.01%), combined 49/357 (13.73%); transitions both 45, control-only 5, combined-only 4, neither 303. Tail: control 21/27 (77.78%), combined 24/27 (88.89%); transitions both 21, control-only 0, combined-only 3, neither 3.
- **Wall-time ratio — PASS.** Per-parent solve times (the record repeats each multi-seed parent time on its three seed rows): cal 527.65 s control / 527.19 s combined, ratio 0.9991; tail 62.90 / 64.62 s, ratio 1.0273; pooled 590.55 / 591.81 s, ratio 1.0021. All are below 1.10.

The combined arm does not isolate the H-1 score changes from the H-2 embed cap. This campaign supports conclusions about the combined arm only; it does not establish per-fix causal effects. No ablation arm was added to the pre-registered comparison.

## Secondary results and failures

- **Floor ΔE:** cal mean paired delta `-0.59` kJ/mol, median `-0.0004`, 95% CI `[-2.53, +1.12]`; tail mean `+8.68`, median `+5.07`, CI `[+2.54, +15.45]`.
- **Symmetry-aware full-pool 1 Å coverage:** cal top-1 hits were 104/357 in both arms; floor hits 123 control vs 122 combined; any-pool hits 230 control vs 236 combined. Tail top-1 hits were 3/27 in both arms; floor hits 1 vs 2; any-pool hits 6/27 in both arms.
- **Score vs negative MACE energy, per returned pool:** median Spearman rho was cal 0.653 control / 0.661 combined and tail 0.679 / 0.679. Means were cal 0.594 / 0.589 and tail 0.467 / 0.504. All code/seed values and row counts are in `research/data/jgen/campaign/H/h3/analysis.json`.
- **Failures:** all 12 failed records are the pre-registered NEPNEF invalid-input exclusion (three seeds × two arms × two panels). The recorded reason is `GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)`. No other unit failed. See [`H3_FAILURE_LEDGER.csv`](../../../../bench/jgen_campaign/archive/H3_FAILURE_LEDGER.csv) and the raw records under `research/data/jgen/campaign/H/h3/{cal,tail}/{control,combined}/records/`.
- Per-parent top-1/floor summaries, coverage counts, and score-energy summaries are in [`H3_PARENT_SUMMARY.csv`](../../../../bench/jgen_campaign/archive/H3_PARENT_SUMMARY.csv); the complete paired rows and full pool correlations are in `research/data/jgen/campaign/H/h3/analysis.json`.

## Final audit

- **Full CPU test suite:** `pytest -q tests` — **390 passed, 20 skipped, 3 xfailed, 1 failed** in 681.54 s. Failure: `tests/test_embedding_validity.py::test_actual_and_cached_fallback_rejected_without_retry_or_geometry_change`; `EmbeddingValidity.check()` did not raise for the cached fallback. The failure is reported without modifying the implementation.
- **Fresh golden record:** `python research/bench/tools/jdev/golden.py record`, with `JACOBIK_GOLDEN_DIR=research/data/jgen/campaign/H/h3/final_audit/golden_20261031`. Both `production.npz` and `u1.npz` were recorded with 245 arrays each. The existing golden directory was not overwritten. Fingerprint at the worktree default: `6cc586e17ab1`.
- **CUDA replay-parity panel:** not run at the owner's direction (“I don't care about the replay parity. Please just finish up and report.”). A VM request was interrupted during local bundle construction; Colab reported zero active assignments and no added spend.

## Artifacts

- Contract: [`H3_CONTRACT.md`](H3_CONTRACT.md)
- Analysis: `research/data/jgen/campaign/H/h3/analysis.json`
- Parent summary: [`H3_PARENT_SUMMARY.csv`](../../../../bench/jgen_campaign/archive/H3_PARENT_SUMMARY.csv)
- Guard term breakdown: [`H3_GUARD_TERMS.csv`](../../../../bench/jgen_campaign/archive/H3_GUARD_TERMS.csv)
- Reproduction utility: [`h3_guard_score_terms.py`](../tools/jgen/diag/h3_guard_score_terms.py)
- Failure ledger: [`H3_FAILURE_LEDGER.csv`](../../../../bench/jgen_campaign/archive/H3_FAILURE_LEDGER.csv)
- Fresh golden files: `research/data/jgen/campaign/H/h3/final_audit/golden_20261031/`

**Stop at Checkpoint H-3 for audit and owner review.**
