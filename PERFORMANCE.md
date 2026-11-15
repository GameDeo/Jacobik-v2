# Jacobik Performance Report (`v1.0a`)

Production benchmark record for Jacobik `v1.0a` (P2R fold-in onto `master`), compiled from the fresh-seed paired Perlmutter fold-in gate (`research/campaigns/p2r/FOLDIN_GATE_CONTRACT.md`) and associated DS database curvature verification (`research/archive/docs/ds/DS_GATE_RESULTS.md`).

All primary metrics compare **Production `v1.0a` (`P`, fingerprint `a47283387172`)** against the preceding **Control baseline (`C`, DS fold-in `d5e2f4113497`)** paired per parent across 3 fresh seeds (`31, 32, 33`) with 10,000-replicate parent-clustered bootstrap 95% confidence intervals (`[lo, hi]`).

---

## 1. Provenance & Evaluation Scope

| Field | Value |
|---|---|
| Release / Version | `v1.0a` (P2R fold-in, `foldin/p2r` merged into `master`) |
| Gate code state | `jgen/p2r-gate` commit `e84865b84d27` (pre-refactor; mathematically equivalent to current `master`) |
| Production fingerprint (`P`) | `a47283387172` (`out_mirror`, `ring_states`, `ring_states_five`, `config_states`, `config_near` enabled; `config_remote=False`) |
| Control fingerprint (`C`) | `d5e2f4113497` |
| Statistics database | `data/db/constraints_2026-11-12.npz` (`sha256: 62c58d5382b776fce0e3d938c5e7212eedc377bf6099eb432bef2210266f326e`, 419,644 entries; shared by both arms) |
| Solve compute | 1 Perlmutter GPU node (4x NVIDIA A100, Slurm job `61284799`), 2 GPUs per arm, 300 s per-parent timeout |
| Relaxation & energy scoring | 1 Perlmutter CPU node (`post-61289501`, `debug` QOS): GFN2-xTB (`crude` = 200 cycles max; `normal` = tighter GFN2 convergence) + CPU MACE-POLAR-1-S (float64) |
| Seeds & panel | Seeds `31, 32, 33`; 120 unlocked `cal+tail` parents (`5` locked `tail` parents dropped from 125 `vm_split`: `IFUROQ`, `SOBHAS`, `XEYTUT`, `YUCZUS`, `ZELTAP`) |
| Solves completed | **`117` parents / `351` units** in **`all`** (`113` `cal_nontail` / 339 units + `4` `tail` / 12 units); `0` GFN2 embed fallbacks on solved units |
| Primary eligible cohort | **`108` parents / `324` units** (`in_corrected_primary` from `research/campaigns/ds/gate_eligibility_v1.json`) |
| Shared failures (3 parents) | `NEPNEF` (`cal`+`tail`) & `UYOVUA` (`cal`): GFN2 embed fallback failure in both arms; `QEZVEY` (`cal`): 300 s timeout in both arms |

---

## 2. Executive Verdict & Registered Gates (`6 / 6 PASS`)

All six pre-registered contract gates passed on both `primary` (`n=108` parents / `324` units) and `all` (`n=117` parents / `351` units):

| Gate | Criterion (`P - C`) | `primary` (`n=108p / 324u`): `C -> P` | Diff `[95% CI]` | `all` (`n=117p / 351u`): `C -> P` | Diff `[95% CI]` | Verdict |
|---|---|---|---|---|---|---|
| `1_cmiss_crude` | `hi < 0` | `0.2901 -> 0.2006` | `-0.0895 [-0.1421, -0.0432]` | `0.2991 -> 0.2051` | `-0.0940 [-0.1453, -0.0484]` | **PASS** |
| `2_cmiss_normal` | `hi <= +0.02` | `0.2253 -> 0.1481` | `-0.0772 [-0.1296, -0.0278]` | `0.2279 -> 0.1538` | `-0.0741 [-0.1225, -0.0313]` | **PASS** |
| `3_top1` (`C-TOP1`) | `lo >= -0.03` | `0.4074 -> 0.4105` | `+0.0031 [-0.0216, +0.0247]` | `0.4017 -> 0.4103` | `+0.0085 [-0.0171, +0.0342]` | **PASS** |
| `3_top1` (unrel med $\Delta E$) | `hi <= +2.0` kJ/mol | med diff `0.0` kJ/mol | `0.0 [0.0, 0.0]` | med diff `0.0` kJ/mol | `0.0 [0.0, 0.0]` | **PASS** |
| `4_failures` | `failed_P_solved_C == []` | `[]` (0 regressions) | `failed_C_solved_P: []` | `[]` (0 regressions) | `failed_both: []` | **PASS** |
| `5_wall` | `ratio <= 1.30x` | `1931.59 s -> 2154.44 s` | **`1.1154x`** | `2038.57 s -> 2257.26 s` | **`1.1073x`** | **PASS** |
| `6_garbage` | `hi <= +0.02` | `0.1110 -> 0.0733` | `-0.0376 [-0.0715, -0.0109]` | `0.1267 -> 0.0827` | `-0.0438 [-0.0772, -0.0152]` | **PASS** |

*Top per-parent wall-time ratios (`P / C` on `primary`):* `DOHTOM` `2.77x`, `POBCEQ` `2.77x`, `UMUBAF` `2.00x`, `FUTSOD` `1.91x`, `ZAQNUE` `1.84x`, `DAMVAQ` `1.76x`, `YABYEJ` `1.74x`, `LIPDEU` `1.60x`.

---

## 3. Energy Quality: Unrelaxed Top-1, Floor, Ranking Loss & Tails

All energies are MACE-POLAR-1-S single points (float64) relative to the GFN2-relaxed crystal reference geometry ($\Delta E = E_{\text{row}} - E_{\text{ref,relaxed}}$, kJ/mol).

### 3.1 `primary` (`n=108` parents / `324` units) and `all` (`n=117` / `351` u)

| Metric | `primary` `P` (`v1.0a`) | `primary` `C` | `primary` Diff `P - C` `[CI]` | `all` `P` (`v1.0a`) | `all` `C` |
|---|---|---|---|---|---|
| Unrel Top-1 $\Delta E$ mean (kJ/mol) | **58.89** `[49.25, 69.62]` | 62.44 | **-3.55** `[-6.77, -0.96]` | **61.93** | 66.08 |
| Unrel Top-1 $\Delta E$ median (kJ/mol) | **42.41** `[38.60, 51.44]` | 43.18 | **0.00** | **43.82** | 47.30 |
| Unrel Floor $\Delta E$ mean (kJ/mol) | **44.90** `[35.70, 55.40]` | 48.61 | **-3.71** `[-7.19, -0.97]` | **47.74** | — |
| Unrel Floor $\Delta E$ median (kJ/mol) | **29.15** `[23.40, 37.93]` | 30.09 | — | **31.38** | — |
| Unrel ranking loss (`top1 - floor`, kJ/mol) | **13.99** (med `6.60`) | 13.83 (med `6.60`) | — | — | — |
| `top1_is_floor` | **0.367** | 0.352 | — | — | — |
| `top1_le0` ($\Delta E \le 0$ kJ/mol) | **0.028** | 0.025 | — | — | — |
| `top1_le5` ($\Delta E \le 5$ kJ/mol) | **0.068** | 0.071 | — | — | — |
| `floor_le0` | **0.071** | 0.068 | — | — | — |
| `floor_le5` | **0.117** | 0.120 | — | — | — |
| Obscene rate ($\Delta E_{\text{top1}} > 200$ kJ/mol) | **0.037** | 0.040 | — | — | — |

### 3.2 Stratum Breakdown (`cal_nontail` vs `tail`, `v1.0a`)

| Stratum | Units | Top-1 $\Delta E$ mean (med) | Obscene rate (`> 200` kJ/mol) |
|---|---|---|---|
| `cal_nontail` (`P` vs `C`) | 339 u (`113p`) | **57.62** (med `42.94`) | **0.021** vs `0.024` |
| `tail` (`P` vs `C`) | 12 u (`4p`) | **183.94** (med `191.37`) | `0.500` vs `0.500` |

### 3.3 Post-Relaxation Energy Windows (`a4c` evaluation, `cal` `n=113` / 339 u and `tail` `n=4` / 12 u)

| Relaxation Level | `cal` `top1 <= 5` kJ/mol | `cal` `top1 <= 0` kJ/mol | `tail` `top1 <= 5` kJ/mol | `tail` `top1 <= 0` kJ/mol |
|---|---|---|---|---|
| **Crude** (GFN2 200 cycles) | **0.4395** | **0.2773** | `0.3333` | `0.0000` |
| **Normal** (GFN2 normal) | **0.5752** | **0.3894** | `0.5000` | `0.4167` |

---

## 4. Geometric Accuracy & Coordination Sphere (`primary`, `n=108p / 324u`)

### 4.1 Heavy-Atom RMSD vs Reference Geometry (A)

| Metric | `P` (`v1.0a`) | `C` | Diff `P - C` `[CI]` |
|---|---|---|---|
| Unrelaxed `min_rmsd` mean (median) | **0.695** (`0.561`) | 0.832 (`0.635`) | **-0.137** `[-0.213, -0.074]` |
| Unrelaxed `top1_rmsd` mean (median) | **1.714** (`1.650`) | 1.708 | — |
| `top1_within_1A` | **0.278** | 0.281 | — |
| `floor_within_1A` | **0.343** | 0.340 | — |
| `full_pool_within_1A` | **0.812** | 0.744 | **+0.068** |
| Relaxed `min_rmsd_refrel` mean (median) | **0.808** (`0.476`) | 0.927 | **-0.116** |
| Relaxed `top1_rmsd_refrel` mean | **1.694** | 1.689 | — |

### 4.2 Coordination Sphere Mean Absolute Error (M-D Distance & D-M-D Angle)

| Stage / Selection | M-D MAE mean (median), A (`P` vs `C`) | D-M-D Angle MAE mean (median), deg (`P` vs `C`) |
|---|---|---|
| **Unrelaxed Top-1** | **0.0765** (`0.0459`) vs `0.0769` | **7.53** (`3.59`) vs `7.61` |
| **Unrelaxed Floor** | **0.0753** (`0.0418`) | **6.48** (`3.59`) |
| **Crude Top-1** | **0.0404** (`0.0157`) vs `0.0430` | **6.27** (`2.93`) vs `6.45` |
| **Normal Top-1** | **0.0336** (`0.0111`) vs `0.0365` | **5.54** (`2.26`) vs `5.74` |

---

## 5. Coverage Metrics (`A4c` Class-Aware & `A4b` Structural)

### 5.1 `A4c` Class-Aware Basin Coverage (`cal` `n=113` / 339 u and `tail` `n=4` / 12 u)

A4c measures whether the generator hits the relaxed reference configuration class and basin (counting enantiomers explicitly).

| Metric | `cal` Crude `P` vs `C` | `cal` Normal `P` vs `C` | `tail` Crude `P` vs `C` | `tail` Normal `P` vs `C` |
|---|---|---|---|---|
| `C-MISS` | **0.1947** vs `0.2920` | **0.1416** vs `0.2183` | `0.5000` vs `0.5000` | `0.5000` vs `0.5000` |
| `miss_mirror_completed` | **0.1799** vs `0.2448` | **0.1357** vs `0.1770` | — | — |
| `miss_not_offered` | **0.0354** vs `0.0944` | **0.0324** vs `0.0914` | — | — |
| `miss_offered` | **0.1593** vs `0.1976` | **0.1091** vs `0.1268` | — | — |
| `C-TOP1` | **0.4159** vs `0.4071` | **0.4572** vs `0.4543` | `0.2500` vs `0.2500` | `0.2500` vs `0.2500` |
| `C-RANK` | **0.3894** vs `0.3009` | **0.4012** vs `0.3274` | — | — |
| `strict_any_new` | **0.8761** vs `0.7699` | **0.9115** vs `0.8407` | — | — |
| `strict_all_new` | **0.7257** vs `0.6460` | **0.8053** vs `0.7257` | — | — |

### 5.2 `A4b` Structural U/R Buckets, Strain & Threshold Sensitivity (Crude, `cal` `n=117` / 351 solved u)

*(Note: `a4b/summary.json` aggregates all `117` solved parents / `351` units under its `"cal"` summary key.)*

| Metric | `P` (`v1.0a`) | `C` |
|---|---|---|
| `U-TOP1` / `U-POOL` / `U-NONE` | **0.2849 / 0.5385 / 0.1766** | `0.2877 / 0.4644 / 0.2479` |
| `R-TOP1` / `R-RANK` / `R-MISS` | **0.4701 / 0.2279 / 0.3020** | `0.4530 / 0.1909 / 0.3561` |
| `strain_top1` mean (median), kJ/mol | **48.77** (`38.50`) | `49.55` (`39.23`) |
| `offset_top1` mean (median), kJ/mol | **12.90** (`6.90`) | `16.31` (`8.24`) |
| `strain_share` | **0.877** | `0.861` |
| `contact_change_top1` | **0.1897** | `0.1839` |
| `top1_inversion` | **0.0199** | `0.0199` |
| `pair_inversion_rate` (`n=67`) | **0.0943** | `0.0952` |
| Sensitivity `hit0.3_eq2` `R-MISS` | **0.3533** | `0.4131` |
| Sensitivity `hit0.3_eq10` `R-MISS` | **0.2108** | `0.2621` |
| Sensitivity `hit0.5_eq5` `R-MISS` | **0.2365** | `0.3191` |
| Sensitivity `hit0.5_eq10` `R-MISS` | **0.1823** | `0.2450` |

---

## 6. Pool Size, Energy Window, Rank Correlation & Curvature

### 6.1 Pool Coverage, Score Span & Correlation (`primary`, `n=108p / 324u`)

| Metric | `P` (`v1.0a`) | `C` | Diff `P - C` |
|---|---|---|---|
| `n_rows` mean (median) | **62.98** (`40.0`) | 37.65 (`27.5`) | **+25.33** |
| Energy window `e_span` mean (median), kJ/mol | **626.02** (`109.67`) | — | — |
| `e_q90_minus_min` mean (median), kJ/mol | **276.10** (`86.57`) | — | — |
| `n_within_5kj` / `10kj` / `20kj` | **3.61 / 6.01 / 11.73** | `2.45 / 4.05 / 7.58` | — |
| `share_within_10kj` | **0.224** | `0.233` | — |
| `score_span` (nats) | **2.03** | `2.06` | — |
| `spearman_rho_unrel` mean (median) | **0.569** (`0.661`) | `0.601` (`0.698`) | — |
| `pearson_r_unrel` mean (median) | **0.597** (`0.657`) | `0.620` (`0.691`) | — |

### 6.2 Score Curvature Calibration (DS DB Benchmark, Unchanged Q1–Q7 Potential)

Because the statistics database (`constraints_2026-11-12.npz`) and Q1–Q7 potential terms did not change in P2R, local score curvature matches the DS fold-in gate (`research/archive/docs/ds/DS_GATE_RESULTS.md`, 80/80 `dev32+n48` complexes):

- **DS Curvature (`D` vs `C`, 80/80 `dev32+n48`):** `1.316` / `0.863` (control `1.332` / `0.864`).

---

## 7. Isomer, Mirror, Ring-State & Class-Basin Resolution

### 7.1 Pool Composition (`primary`, `n=108p / 324u`)

| Metric | `P` (`v1.0a`) | `C` |
|---|---|---|
| `n_classes_pool` | **3.37** | `2.05` |
| `has_multiclass` | **0.167** | `0.111` |
| `n_mirror_rows` mean (median) | **25.27** (`15.33`) | `0.00` |
| `has_mirror_rows` | **0.944** | `0.000` |
| `mirror_is_top1` | **0.0093** | — |
| `mirror_is_floor` | **0.0062** | — |
| `n_ring_rows` | **4.44** | — |
| `n_stereo_prod` | **6.00** | `3.10` |

### 7.2 Class-Basin Relaxation Analysis (`class/summary.json`, 28-Unit Subset, Seed 31, Normal Level)

| Metric | `P` (`v1.0a`) | `C` |
|---|---|---|
| `classes_per_unit` | **5.00** | `3.68` |
| `rows_per_class` | **30.72** | `20.52` |
| `basins_per_class` | **20.27** | `13.48` |
| `collapse_to_lowest` | **0.215** | `0.241` |
| `within5_of_class_min` | **0.479** | `0.461` |
| `best_ranked_in_lowest` | **0.256** | `0.223` |
| `rho_within_class` | **0.010** | `0.001` |

---

## 8. Relaxation Settle & Solver Runtime Telemetry

### 8.1 Relaxation Behaviour & Shift (`primary`, `n=108p / 324u`)

| Metric | `P` (`v1.0a`) | `C` | Diff `P - C` `[CI]` |
|---|---|---|---|
| `normal_floor_offset_kj` (and crude offset) mean (med), kJ/mol | **3.19** (`0.51`) `[-0.95, +7.79]` | `6.76` (`1.36`) | **-3.58** `[-6.69, -1.14]` |
| `crude_cycles_mean` | **11.83** | `12.07` | — |
| `normal_cycles_mean` | **51.46** | `52.31` | — |
| `normal_level_shift_A` | **0.546** | `0.556` | — |

### 8.2 Normal-Level Relaxation Throughput (`normal_{C,P}/relax_level.json`, `all` `n=117` parents)

| Arm | `n_refs` | `n_rows` | `n_ref_failed` | `n_row_failed` | `xtb_wall_s` | `wall_s` |
|---|---|---|---|---|---|---|
| `P` (`v1.0a`) | 117 | **1,554** | 0 | **0** | 363.21 s | 403.64 s |
| `C` | 117 | 1,467 | 0 | 1 | 344.65 s | 384.15 s |

### 8.3 Solver Execution & Replay Telemetry (`primary`, `n=108p / 324u`, + GPU OOM logs)

| Metric | `P` (`v1.0a`) | `C` |
|---|---|---|
| `wall_solve_s` mean (median) | **12.98 s** (`6.92 s`) | `11.01 s` (`6.80 s`) |
| `parent_wall_s` mean (median) | **19.95 s** (`9.95 s`) | `17.89 s` (`10.17 s`) |
| `replay_chunks_n` | **1.00** | `1.31` |
| `replay_fallbacks` | **0.0** | `0.0` |
| `infeasible_replaced` (`primary` / `all`) | **0.0 / 0.051** | `0.0 / 0.017` |
| GPU `oom_retries` (`g0 + g1` in `{P,C}_g{0,1}/run_summary.json`) | **237** (`117 + 120`) | `313` (`147 + 166`) |

---

## 9. Regression Check Parity Note (`v1.0a` CPU vs GPU `out_mirror`)

Following the P2R merge, CPU assembly (`src/jacobik/builder/assemble.py`) was unified with GPU so `out_mirror` enantiomer reflection executes on CPU whenever `mirror_possible` is active (`batched = ctx.device.type == "cuda" or BATCHED_ASSEMBLY or mirror_possible`), and fresh `v1.0a` CPU golden arrays (`production.npz`, `u1.npz`) were recorded for fingerprint `a47283387172`.
