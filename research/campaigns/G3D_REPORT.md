# G-3d — clean24 checkpoint, 2026-11-02

**STOP: neither R2s nor IA advances.** Both pass the registered energy
predicates and fail the maximum-angle-deviation p95 guard. No gate was
retuned or rerun after failure. Cal/tail were not started. FA and FR+
remain failures at step 0, not filter losers. No production choice.

## Filter, unchanged predicates

24 fixed parents × seeds 11–13 × C1/R2s/IA: **216/216 valid records**;
72/72 finite paired units per candidate, 24/24 complete parents. No
failures or missing units; no replay or embed fallback. Every returned
row has finite angle diagnostics and reference-coverage geometry.

| Arm | Median paired top-1 ΔE | Mean paired floor ΔE | Angle p95 | Angle p95 − C1 | Advance |
|---|---:|---:|---:|---:|---|
| R2s | -11.535 | -25.541 | 89.496° | 54.325° | FAIL: angle guard |
| IA | -9.209 | -25.969 | 58.925° | 23.754° | FAIL: angle guard |

C1 pooled-row angle p95 is 35.171°; admissible candidate p95 is at most 45.171°.
Energy units are kJ/mol, candidate minus paired C1; nonpositive median
top-1 and nonpositive mean floor pass. The angle statistic is the p95
over all returned rows of each row’s maximum deviation from its Q2
target means. Counts: C1 2510 rows, R2s 2507, IA 2402. Both angle
excesses exceed 10°. This guard alone determines both failures.

## Parent-clustered energy summaries

Means with 95% percentile CIs: 10,000 parent resamples, RNG seed 11;
average the three paired seed units within each parent first. These
are screen summaries, not the unrun cal/tail acceptance gates.

| Arm | Mean top-1 ΔE [95% CI] | Mean floor ΔE [95% CI] |
|---|---|---|
| R2s | -30.087 [-60.751, -7.109] | -25.541 [-51.043, -6.651] |
| IA | -29.001 [-53.876, -10.120] | -25.969 [-47.627, -10.089] |

The energy improvements do not override the failed distortion guard.

Sensitivity excluding FEKKAJ and NUWFAM (FEKKAJ is not in clean24, so
this excludes NUWFAM only): 23 parents, 69 finite pairs per candidate.

| Arm | Median top-1 ΔE | Mean top-1 ΔE [95% CI] | Mean floor ΔE [95% CI] |
|---|---:|---|---|
| R2s | -11.844 | -32.932 [-64.026, -9.709] | -26.658 [-52.433, -6.919] |
| IA | -9.582 | -30.302 [-55.995, -10.734] | -26.168 [-48.392, -9.703] |

No parent was dropped from the registered advance decision.

## Secondary diagnostics

| Arm | Obscene rows (>200 kJ/mol) | Class-changed rows | Pool coverage units /72 | Parent coverage /24 | Solver+assembly wall |
|---|---:|---:|---:|---:|---:|
| C1 | 1281/2510 (51.04%) | 151/2510 | 45 | 15 | 258.04 s |
| R2s | 1110/2507 (44.28%) | 426/2507 | 45 | 15 | 398.93 s |
| IA | 1158/2402 (48.21%) | 256/2402 | 42 | 15 | 386.42 s |

Coverage is full-pool symmetry-aware heavy-atom RMSD <=1 Å, including the
metal; parent coverage is any hit over the three declared seeds, not
top-1/floor-only coverage. Unknown class labels and automorphism-cap
counts, paired top-1 score and paired obscene-fraction CIs are retained
in G3D_SCREEN_FINAL.json. Automorphism enumeration was capped on 21/72
units in every arm. No unknown class labels occurred. Solver/assembly
wall counts batched seeds once
per parent, excluding shared preparation/MACE. Wall ratios are descriptive
here, not additional filter predicates. Nominal budget is 250 steps; some
batches stopped early. Recorded step-count sets are shown below. A
counterfactual fixed-budget wall was not measured; no rescue timing run
was started after the failed filter.

R2s: recorded replay step-count set [56, 128, 224, 250]; solver/assembly ratio versus C1 1.546; paired obscene-fraction change -9.061 pp [95% CI -19.040, -1.485].
IA: recorded replay step-count set [120, 176, 184, 248, 250]; solver/assembly ratio versus C1 1.498; paired obscene-fraction change -6.055 pp [95% CI -11.648, -1.501].

## Separate case tables

FEKKAJ and NEPNEF are not members of the frozen clean24 manifest. Their
step-0 records remain in G3D_STEP0_FINAL.json. NEPNEF remains known invalid
input, embed_fallback=true with no accepted rows, and exempt under
amendment 2. No reserve/val/locked sets were evaluated.

### NUWFAM

| Seed | C1 top-1 / floor | R2s top-1 / floor | IA top-1 / floor |
|---|---|---|---|
| 11 | 131.409 / 131.409 | 230.298 / 128.780 | 169.799 / 143.331 |
| 12 | 155.402 / 155.402 | 230.193 / 166.277 | 189.865 / 129.775 |
| 13 | 231.291 / 171.133 | 163.625 / 163.341 | 161.259 / 120.664 |

### XOJHIN

| Seed | C1 top-1 / floor | R2s top-1 / floor | IA top-1 / floor |
|---|---|---|---|
| 11 | 187.382 / 185.405 | 178.687 / 178.687 | 156.982 / 156.982 |
| 12 | 187.266 / 184.692 | 181.491 / 181.491 | 178.430 / 159.771 |
| 13 | 199.300 / 185.608 | 176.166 / 176.166 | 150.529 / 150.471 |

NUWFAM is retained in the main decision and shown in exclusion sensitivity.
XOJHIN remains included as a search challenge; no score-hole exception
is inferred and it is not dropped. Embed/E-Z variant keys remain in the
full record/export row keys.

## Execution, provenance and checks

Amendment 4 and runtime/runner fixes: 5bc9fef. FA/FR+ retain the failed
step-0 cap gate (14/36 censored at 144); criterion, horizon and quantile
were not revisited. Their absence is not a missing filter result. The cause
of the step-0 cap failure remains unknown; size is not established as its cause.

R2s/IA omit an unused cap only in whole-run/fixed-switch modes. Exact
250-step update/state parity against legacy capped modes passed; adaptive
finish still requires a cap. Five light phase tests and two runner
regression checks passed. The runner now sets/resets FINISH_PHASE in both
solver modules and prevents active phases reusing C1 results. Canonical
replay regeneration unchanged; production default fingerprint unchanged.
Production arithmetic at flags-off is unchanged; previous full golden
245/245 per profile and full finish-phase file (31 passes) were not repeated.

Frozen fingerprints: C1 273f2ccbf892; R2s b2f758e72507; IA 4ad46b7b7722.
Every arm had a 250-step budget. R2s and IA used the resolved thermal widths;
IA fixed
switch 100, R2s score-only throughout. Both release all aids as registered.
Local RTX 2060 SUPER only, seeds 11–13, each candidate in the same bench
invocation as C1. GPU lock held; nvidia-smi captured before every start.
All recorded replay chunks mode replay; zero fallback. No VMs/Colab; spend 0.

Paired performance profiles AJAQAD and QEZVEY were reused as two fixed
clean24 parents, with record hashes/flags/runner SHA/machine/GPU checked
before the remaining 22-parent run. G3D_CLEAN24_INPUTS.json freezes this
collection before panel-scale results. Original exports are preserved.

Planning estimate was 7485.6 s (~2.08 h) based on the mean of the two
profiles. Remaining 22-parent job actually took 922.841 s. Including
both profile jobs, total GPU-job wall was 1580.459 s (26.34 min).
Large profile heterogeneity made the planning estimate conservative.
Full-pool xz XYZ exports, coverage manifests, energies, scores and keys
are preserved in shared research/data/jgen/campaign/B/g3d/. Completion
waits used the owner-requested 20-minute timer or earlier completion.

G3D_FILTER_RESULT.json holds the exact mechanical decision and denominators;
G3D_SCREEN_FINAL.json adds secondary metrics, case tables and all 216
record SHA256s. No relaxation, rescue rerun, tuning, production change or
fold-in proposal. Step 5 did not start because no arm advanced; its 8-hour
estimate gate and cal/tail gates are not evaluated. L4 confirmation remains
required before any future fold-in. All GPU jobs finished; lock released.
Checkpoint: stop and await audit.
