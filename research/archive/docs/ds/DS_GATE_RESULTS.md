# DS qualification gate (Step 8) results (2026-11-12)

Job J3 = Perlmutter 61160804 (GPU node, `regular`, 19 min 34 s, about 0.33 node-h), snapshot `ds-51cf9ad`, arms **C** (production, DB `d87b81a5...`) and **D** (DF, `be4e0571...`), seeds 11-13, per-parent cap 5 min.
No P arms (Step 6's P/S rule failed, `DS_STEP6_RESULTS.md`). Targets: audited overlay (`9a9c820b...`) and frozen eligibility for both arms. Analysis: `tools.jgen.d3.analyse`, run once per stage and root, unchanged; curvature gate run on the workstation CPU
(80 of 80 cases, both arms). `primary` root (valid + repaired targets: 109 and 95 parents) decides; the `all` root (120 and 110) is reported beside it. Local: `research/data/ds_proto/gate/{stage1,stage2}/{primary,all}/analysis.json`.
Contract: `contract_D3_2026-11-09.md` gates plus `DS_STEP5_6_8_CONTRACT_v2.md` E. **A failed gate stays failed.**

## Verdict
**Stage 1 (calibration + tail): PASSES** on both roots; D is the stage-1 candidate. **Stage 2 (eta2 holdout): FAILS the registered rule** on both roots, because parents did not complete (below), not because D is worse.
The DB is therefore **not qualified** by this run. Stage 3 (promotion reserve) is not run (needs the auditor's list and a separate go).

## Stage 1 (D minus C, kJ/mol; negative = lower energy = better)
| | primary (109 parents) | all (120) |
|---|---|---|
| top-1 | -9.27 [-18.31, -1.58] (n 107) | -8.08 [-16.32, -0.97] (n 116) |
| floor | -7.03 [-15.62, +0.43] | -5.58 [-13.45, +1.21] |
| hit within 0.5 A | +0.049 [0.003, 0.099] | +0.05 [0.00, 0.09] |
| min RMSD (A) | -0.044 [-0.080, -0.010] | |
| guard, without artefact list | 8 trips vs 11 mirror, null 1: pass | 9 vs 11: pass |
| catastrophic (>200 kJ/mol) | 9 vs control 21: pass | 9 vs 21 |
| wall ratio | 0.957 [0.853, 1.058]: pass | 0.966 |
| curvature (median ratio / rank agreement) | D 1.316 / 0.863 vs C 1.332 / 0.864: pass (band +-0.10 / -0.03) | same |
| failures | UYOVUA (GFN2 embedding fallback, both arms): symmetric, pass | + NEPNEF, QEZVEY, symmetric |
All seven stage-1 gates (top-1, guard, catastrophic, wall, failures, energy, curvature) true on both roots. Guard trip parents (D worse than C by more than 25 kJ/mol on two or more seeds): FETTAZ, FUTSOD, HOHZAG, LOPJEF, XUKKIA, YABYEJ, ZAFPOP, ZAQNUE
(plus MIDQEY, HUNGOP on the lists that include them). YABYEJ, LOPJEF, ZAQNUE are the three already seen in Step 6 and in the D2 package. Large improvements: VOBGUR (C ~185-231 -> D ~3-5), HEDJOP (~300-335 -> ~50-90), USAPOU (~200-240 -> ~94-132), VAVBIE, DAMVAQ, FEKKAJ.
Named supplement cases: JEGPUJ D-C +13 / -36 / -3 kJ/mol, SIVBUV +4 / +3 / +2 (reported, not tuned); MIDQEY (cal artefact) D-C +31 / +33 / +5.

## Stage 2 (110 holdout parents; 95 primary)
| | primary | all |
|---|---|---|
| top-1 | -24.19 [-43.96, -7.14] (n 91) | -21.23 [-39.08, -5.28] (n 105) |
| floor | -11.43 [-22.09, -1.48] | -11.10 [-21.37, -1.65] |
| guard | 9 trips vs 25 mirror: pass | 12 vs 29: pass |
| catastrophic | 34 vs control 52: pass | 48 vs 66 |
| wall ratio | 0.835 [0.74, 0.94]: pass | 0.887 |
| **holdout gate** | **False** | **False** |
| **failures gate** | **False** | **False** |

Why the two gates are false (from `tools/jgen/d3/analyse.py`): the stage-2 failures gate is `no parent failed` (any failed parent in either arm), and the holdout gate needs the edge-on row counts of both arms, which the harness
withholds (None) if any record in either arm is not ok. Four primary parents (five in `all`) did not complete:
- ADINOR, LADJAB: `KeyError: 'v'` in **both** arms (the production control fails too: a generator/bench defect on these parents, independent of the DB).
- UZOREI: 300 s cap exceeded in both arms.
- EPUPEL: **control** timed out (300 s), D completed (250 / 324 / 223 kJ/mol): an asymmetric result in D's favour, not a D failure.
- IWOLEN (`all` root only): `atom_mapping_failed` in both arms.
Timeouts are recorded and not retried (rule). Failed and missing parents stay in the denominators. No D-specific failure occurs in stage 2.

**Post-hoc, informational, not a gate and not used to change the verdict:** summing edge-on rows over the parents complete in both arms, primary root: C 3, D 2 (91 parents); stage 1 primary: C 322, D 296 (107 parents).
On this reading D's edge-on count is not higher than the control's, and the stage-2 top-1 upper bound (-7.14) is far below the +2 limit; the registered gate is still False.

## What this means, plainly
- On the 109 calibration/tail parents (primary), D is better than production by about 9 kJ/mol on top-1 with a CI that excludes zero, with no loss on coordination, wall time, catastrophic rate, guard or curvature.
- On the holdout, D is better by about 24 kJ/mol, but the registered holdout and failures gates cannot pass while the control or both arms leave parents incomplete. Re-evaluating them requires a decision, not an automatic ablation.
- Options for the owner (none is run): (a) accept the registered result as a fail of the strict stage-2 rule and treat the incomplete parents as a harness limitation; (b) re-run only the five incomplete parents with a longer cap or after fixing the `KeyError: 'v'` defect, as a labelled addendum (contradicts "never retried"; needs your go);
  (c) fold in on stage 1 plus the stage-2 evidence read as above (your call, since the registered stage-2 rule fails).
- Leads unchanged (`DS_STEP7_DECISION.md`): YABYEJ / LOPJEF / ZAQNUE worse than production (search/pool), LIPDEU QO sensitivity, F7, F8.

## Follow-up checks (2026-11-12, after the verdict above; post-hoc, informational, nothing here changes a gate)

### 1. The three regressions: how large, and search failure or scoring failure?
D minus C top-1, kJ/mol, seeds 11 / 12 / 13 (stage 1, identical to the Step 6 records for the same parent, seed and arm, so J2 and J3 reproduce each other exactly):
YABYEJ +52.9 / +55.3 / +44.2 (mean +51); LOPJEF +8.5 / +29.8 / +28.9 (+22); ZAQNUE -5.7 / +44.5 / +26.8 (+22). Largest stage-1 loss overall is HOHZAG (+143, mean); then YABYEJ.
Test (`research/bench/jgen_campaign/ds/gate_cross_rescore.py`; fixed-pool scorer, saved J3 pools, rows of the EZ/assignment variant that has all three seed pools): each arm's pool is re-scored under its own DB (validation) and the other arm's DB.
Validation: the saved selected row is reproduced in 17 of 18 pool-seeds (ZAQNUE C seed 11 differs: 126.9 vs 133.2 kJ/mol). Flags stay those of the pool's own arm, as in the earlier fixed-pool comparisons.
- **Under D's own DB, C's top row scores higher than D's own top row in all 9 parent-seeds**: YABYEJ -0.155 vs -0.750 (about 0.6 score units; same under C's DB), ZAQNUE 0.570 / 0.651 / 0.544 vs 0.412 / 0.190 / 0.393, LOPJEF 0.366 / 0.381 / 0.388 vs 0.339 / 0.323 / 0.361 (small).
  So D's scorer prefers the lower-energy basin C finds, and D's generation did not reach it: **a search failure, not a DB scoring failure**, for YABYEJ and ZAQNUE (D's pool floor is 27-52 kJ/mol above C's). For LOPJEF D's pool does contain lower-energy rows than C's (floor 81-89 vs 91-98), ranked 2nd-6th by both scorers, with small score gaps: a ranking/score-surface limit plus a modest search shortfall.
- P/S starts do not rescue them (Step 6: YABYEJ DS-D -0.6 / +2.3 / +11.7). Not a reason to change the DB; a solver/search lead (`DS_STEP7_DECISION.md`).
- Caveat: C's rows are scored with the C-arm typing flags and D's DB (the earlier fixed-pool convention); a D-flags rescoring of C's rows was not possible because the rows are keyed by C's typing.

### 2. Correctness of the comparison
- The C and D phases used identical `tools.run panel` command lines in both stages, differing only in output paths, arms spec (flags) and the DB; both carry `--target-overlay` (sha `9a9c820b...`) and `--gate-eligibility`. The audited targets therefore apply to production as well.
- 666 parent-pairs ok in both arms (both stages): reference energy, reference coordinates and symbols identical between arms (0 mismatches).
- Independent recomputation of the stage-1 primary headline from the raw records (own code, own bootstrap): 107 parents, mean D-C top-1 -9.27, CI [-18.28, -1.68]; harness -9.27 [-18.31, -1.58]. Sign convention checked (top-1 dE is energy above the reference; lower is better).
- Large gains are real structures: VAVBIE shell RMSD 1.5 A -> 0.16 A, USAPOU 0.8 A -> 0.3 A heavy-atom RMSD, VOBGUR 185-231 -> 3-5 kJ/mol at the same shell RMSD (0.15 A), HEDJOP 300-335 -> 50-92.

### 3. What the average hides
Stage 1 primary (107 parents): median change -0.3 kJ/mol; 11 parents gain more than 25 kJ/mol (sum -1,207), 6 lose more than 25 (sum +309), 90 are within +-25 (sum -93). Without the five largest gains the mean is -1.7 kJ/mol.
Stage 2 primary (91 complete parents): median -2.2; 25 gains over 25 (sum -3,000), 8 losses over 25 (sum +747; AYARIB +180, BADQIH +124, XIKWEU +112), 58 within +-25 (sum +52); without the top five gains the mean is -8.2.
So the improvement is real but concentrated: a minority of parents improve a lot, most are unchanged, and a smaller minority get worse.
