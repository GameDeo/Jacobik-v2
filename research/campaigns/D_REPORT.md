# D-B score-side checkpoint — 2026-11-06 audit follow-up

Branch `jgen/d-score`, based on master `74b4d47`. D-B work stops at this checkpoint. The branch is not merged.

## Flags and implementation

- `q6_physical_all` (default `false`) opens physical class-5 Buckingham repulsion for all non-bonded pairs except ligand 1–2 and 1–3. `q6_weight_14` is the 1–4 physical-energy multiplier (default `0.0`); zero masks those pairs out of Q6 cleanly, and positive values scale energy without moving the onset. Donor–donor onset scaling remains hh 0.95, ho 0.90, oo 0.80. Polar H…acceptor pairs use the physical repulsion under this flag.
- `q7_bound_centre` (default `false`) adds a score Q7 term for table-covered pivots even with the production energy-calibrated bend solver. It shifts the embed-angle center toward the crystal table, capped by `q7_shift_max` (default 10°); the free zone is `max(2°, thermal SD)`. `stats_from_db()` now preserves the DB path so `load_db()` can read the adjacent `.bends.json` sidecar. The production fingerprint remains `5f74b14b57ed` with defaults.
- The provisional Q7 table, written using `src/fit/bends.py`, contains 1,683 observations, 138 supported entries and 0 failed parents from the 120 cal/tail frames after locked exclusions. It is in the gitignored campaign data path `research/data/jgen/campaign/D/q7_bound_angles.bends.json`.
- `build_bound_angle_table_from_frame(frame, load_complex)` is now exposed in `src/fit/bends.py`; it reads the `surface_fit` split, deduplicates parents, and returns a table plus per-parent extraction report. D-A can wire this fit-frame builder into `build_db.py` and hash the sidecar in metadata.
- Tests: `pytest -q tests/test_fit_bound_bends.py tests/test_q6_donor_pairs.py` → 15 passed after the audit follow-up. Default golden check from the prior checkpoint: production 245 arrays, 0 differ; u1 245 arrays, 0 differ. The default fingerprint was rechecked after the Q6 flag correction and remains `5f74b14b57ed`.

## Q6 crystal-frame diagnostic — reported, not a gate

After the owner amendment, `q6_weight_14=0.0` was used and the diagnostic was rerun on all 120 cal/tail crystals, with no failures (LOPCAW used the existing heuristic loader fallback). The Q6 increase had mean +2.15 kT, median +1.47, p95 +5.28 and maximum +16.15 kT. There were 1,696 physical pairs above 0.1 kT and 104 above 1 kT. Ten parents exceeded ΔQ6=5 kT:

| Parent | ΔQ6 kT | Main recorded cause |
|---|---:|---|
| NEPNEF | 16.15 | Close polar O…H/acceptor contacts (already excluded from solver evaluation after its GFN2 failure) |
| XUKKIA | 12.65 | Two very short class-4 P…N 1–5 contacts; additional class-5 contacts |
| HOYBED | 10.37 | Many class-4 1–5 and class-5 close contacts |
| QERSUB | 9.12 | Two close polar Cl…H contacts |
| MIDQEY | 9.10 | Multiple close class-4 S…H 1–5 contacts |
| NOYTUS | 5.50 | Close class-5 C…C contact; additional weaker 1–5 contacts |
| VAVBIE | 5.27 | Close polar O…H contacts and other class-5 contacts |
| YABYEJ | 5.18 | Numerous moderate class-4 1–5 and class-5 contacts |
| SILROY | 5.09 | Cluster of close class-5 C…H contacts |
| EDOBUV | 5.06 | Several close class-5 Cl…P / Cl…C and C…H contacts |

This diagnostic is reported, not gated, per the 2026-11-06 owner amendment. Pair records and classifications are in `research/data/jgen/campaign/D/q6_crystal_weight14_zero.jsonl`.

## Q7 A4b direction check

On 4,584 covered parent-seed pivot angles from 119 usable parents, relaxed top-1 motion agreed with the direction toward the provisional crystal means 97.88% of the time. Mean gap closed was 42.18%; median gap closed was 93.99%. NEPNEF was recorded as a failure because GFN2 embed refinement failed; its MMFF fallback shell was not used. Full observations are in `research/data/jgen/campaign/D/q7_a4b_check.json`.

## F1 far-field scan — completed; Q2 continuation indicated

The corrected atom-order mapping passed local CPU checks on QEZVEY, VAVBIE, USAPOU and AZUSIA, including builder/reference symbols, coordinate round-trip and stored pool-frame shape. One L4 rerun on `jgD2` returned all four parents, 62 scans total, all status `ok`, fingerprint `5f74b14b57ed`.

Applying the registered coherent-mismatch rule to the scan endpoints beyond the fitted envelope: Q2 shows a consistent positive MACE-minus-Q2-penalty residual in all four parents. Parent-median residuals are about +331, +347, +302 and +258 kT (QEZVEY, VAVBIE, USAPOU, AZUSIA respectively); Q1 endpoint residual signs are mixed. S_orient is also usually too soft at high-energy tilts, but its residual size is much more class-dependent (parent medians about +80 to +1,025 kT). The rule therefore indicates a Q2 `far_field_form` continuation; it does not support a blanket continuation for Q1 or S_orient. The F1 scan artifacts are in `research/data/jgen/campaign/D/f1_v3/d_f1_v3/`.

The registered rule is triggered for Q2 and not triggered for Q1 or S_orient. This checkpoint records that far-field decision; it does not claim a fitted or enabled `far_field_form`. F1 records the scanned term profile, not all score components needed to allocate MACE residual jointly across Q2 and co-varying terms without double-counting. A versioned Q2 continuation table requires the registered full-component joint fit and cross-check on rescored pools before it can be produced.

## Fixed-row counterfactual filter

Results use 115 parents with at least one common seed containing two or more finite matched rows; four one-row pools were excluded (MIGTEE, MIVXAR, QAPFAS, UYOVUA). CIs are 2,000 parent-cluster bootstrap draws. Regret is in kJ/mol; lenient count is the per-parent mean of seed counts, with the number of parents having at least one lenient row in parentheses.

| Arm | Spearman ρ (95% CI) | Top-1 regret kJ/mol (95% CI) | Lenient rows (affected parents) |
|---|---:|---:|---:|
| Control | 0.567 [0.497, 0.631] | 18.14 [13.82, 23.20] | 1.113 (33) |
| Q6 all (1–4 weight 0) | 0.573 [0.501, 0.637] | 17.46 [13.11, 22.16] | 1.122 (not recomputed) |
| Q7 bound centre | 0.565 [0.489, 0.628] | 18.66 [13.84, 23.88] | 1.055 (30) |
| Q6 + Q7 (1–4 weight 0) | 0.572 [0.503, 0.636] | 17.85 [13.33, 23.04] | 1.070 (31) |

Paired on-minus-control differences (95% CI):

| Arm | Δρ | Δ regret kJ/mol | Δ lenient rows |
|---|---:|---:|---:|
| Q6 all (1–4 weight 0) | +0.00632 [−0.0101, +0.0230] | −0.68 [−1.89, +0.33] | +0.009 [−0.038, +0.067] |
| Q7 bound centre | −0.00235 [−0.0298, +0.0195] | +0.52 [−1.36, +2.53] | −0.058 [−0.151, +0.020] |
| Q6 + Q7 (1–4 weight 0) | +0.00473 [−0.0257, +0.0301] | −0.29 [−2.35, +1.68] | −0.043 [−0.145, +0.055] |

No flag meets the registered rule for dropping a flag because both ρ and regret do not worsen with CIs excluding zero. This fixed-row screen does not override the Q6 crystal diagnostic or the Q2 F1 continuation trigger. Contracts were amended before rerun; summaries are in `research/data/jgen/campaign/D/counterfactual/summary.json` and `q7_summary.json`.

The D-A typing flags/tables are not included: D-A still has work in progress on `jgen/t-typing` and no provisional table was available in this D-B worktree. The directive’s cross-session comparison must therefore be completed after D-A supplies its tables; the arms reported here are score-side only.

## Colab spend and status

`jgD2` started and stopped by this session. The L4 rerun added 1.78 shared units from the pre-setup balance 437.36, below the owner-approved 5-unit addition. No D VM remains active. No local GPU job was started. D-B checkpoint is ready for audit; D2/D3 and any merge are out of scope.
