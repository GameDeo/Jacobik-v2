# C-2 coverage report — 2026-11-03

**Stopped after Phase 2 under the registered rule: no arm qualifies.** K, U and pool_window_fill were built; ring_conf_pool was omitted. No flag was selected and S1 is not recommended. No L4 confirmation was launched.

## Provenance

Worktree `jgen-c`, branch `jgen/c-coverage`, base `b0134a2`. Production fingerprint remains `5f74b14b57ed`. Screen producer revision `b9ce924`; subsequent revisions correct measurement scope and analysis/accounting without changing the running arm algorithms. All gates use unrelaxed MACE energies and crystal geometry. No post-generation relaxation was performed.

Population: 120 unique cal/tail parents after the locked exclusion list. The five validation-derived tail codes were excluded before loading inputs. Primary statistics exclude FEKKAJ and NEPNEF; both remain in tables and the failure ledger. Seeds 11–13. Every arm and its control ran in the same invocation on the local RTX 2060 SUPER, NVIDIA driver 595.99.02 / CUDA driver API 13.2. The local environment reports PyTorch 2.14.1 / CUDA runtime 12.9.

The registered contract and settings precede arm runs. Bootstrap: 10,000 parent-cluster resamples, seed 11; seeds are averaged within parent before resampling. Undefined energies/correlations are not imputed; paired sample counts and failures are reported. Coverage failures stay in the denominator.

Measured screen wall, including MACE/export/startup: 190.68 minutes. Colab spend so far: 0.00 units. No production defaults, protected DBs or goldens were changed.

| Arm | Flags / budget | Fingerprint |
|---|---|---|
| C | production; three_w16 | `5f74b14b57ed` |
| K | start_kabsch=true; three_w16 | `38dcb8218b73` |
| U | beam_union=true; three_w16 | `0a7b8896ac1f` |
| T | pool_window_fill=true; three_w16 | `8843b345cc49` |
| S1 | production; single_w48 (three independent replicates) | `5f74b14b57ed` |

## Phase 0 placement

119/120 successful parents; 195,456 initial rows across all valid native E/Z variants. 101/120 parents (84.17%) have median row-maximum offset >0.3 Å. Pooled median row maximum: 3.15 → 2.81 Å (10.96% reduction). Build start_kabsch by the parent-fraction condition; the alternative 30% reduction condition fails.

Per-parent/ligand-class/overall donor-offset, M–D-distance and M–D-error quantiles are in C2_PHASE0.json. K retains orientation diversification, pairing the Kabsch random draws in the diagnostic. Its actual rerun matches the paired-fit median. Placement counts all candidate initialization slots before class-budget selection; the tiny-pool trace separately reports actual class-budget rows.

The initial helper captured only the first native E/Z variant. A scope audit identified nine multi-variant parents, which were remeasured in C and K. There are 133 native parent/variant configurations, 132 valid (NEPNEF fails GFN2). The corrected build rule remains 101/120. Original diagnostic records are retained. The Phase 2 runner used native enumeration from its first invocation.

## Phase 0 tiny pools

Counts are seeds 11/12/13. All seven tiny-pool parents have one native E/Z variant. All their solved rows are finite and embedding-valid; production has no additional row-level embedding-validity rejection. Filters: class-budget split, finite scores/geometry, global 200-nat window, best-first symmetric 0.5-Å dedup, and output cap.

| Parent | Solved | In window | Distinct before window | Distinct in window | Returned | Classification |
|---|---|---|---|---|---|---|
| QAPFAS | 128/128/128 | 128/128/128 | 1/1/1 | 1/1/1 | 1/1/1 | a_sampled_symmetry_equivalent_starts |
| UYOVUA | 128/128/128 | 128/128/128 | 1/1/1 | 1/1/1 | 1/1/1 | a_prime_frozen_topology |
| MIGTEE | 128/128/128 | 128/128/128 | 1/1/1 | 1/1/1 | 1/1/1 | a_sampled_symmetry_equivalent_starts |
| MIVXAR | 128/128/128 | 128/128/128 | 1/1/1 | 1/1/1 | 1/1/1 | a_prime_frozen_topology |
| ZAZREZ | 128/128/128 | 128/128/128 | 1/2/2 | 1/2/2 | 1/2/2 | a_sampled_symmetry_equivalent_starts |
| SIVBUV | 128/128/128 | 128/128/128 | 2/2/2 | 2/2/2 | 2/2/2 | c_solver_collapse |
| HOXJUZ | 128/128/128 | 128/125/128 | 6/12/9 | 6/10/9 | 6/10/9 | b_window_loss |

QAPFAS/MIGTEE/ZAZREZ have nearly symmetry-equivalent sampled starts despite many formal torsions; this is provisional sampled-space class (a), not proof of a globally small conformational space. UYOVUA and MIVXAR are (a-prime), with large rings frozen by topology. SIVBUV has diverse starts that collapse in IK: class (c), unresolved. HOXJUZ seed 12 loses two distinct rows to the window: class (b), motivating pool_window_fill.

The five preselected normal controls are AJAQAD, AWONOR, AYOGUR, AZUSIA and BIBFUN. Their full stage counts, score spreads and pairwise RMSD distributions are retained. Pair distributions use all pairs up to 4096, otherwise a seed-11 uniform sample; float64 reductions avoid cancellation on nearly identical rows.

## Phase 0 rings

Internal nonaromatic 5–8 rings dominate on 2/9 (DOHTOM, HOHZAG), below the required four. FEKKAJ is 48.79%, not rounded up. ring_conf_pool is omitted. Across the population, 27 parents and 33 ligands contain qualifying internal rings. Chelate domination on UYOVUA alone is below the clarified ≥5/9 stop threshold; the earlier any-parent stop was an auditor specification ambiguity.

| Parent | Best seed / row | Shell RMSD Å | Internal 5–8 share | Chelate share | Aromatic/other share |
|---|---|---:|---:|---:|---:|
| BABNAV | 12 / 13 | 0.09 | 0.00% | 0.00% | 85.43% |
| CIXYUD | 13 / 19 | 0.12 | 0.00% | 0.00% | 89.49% |
| DOHTOM | 11 / 30 | 0.11 | 87.33% | 2.51% | 1.26% |
| DUHHIX | 11 / 13 | 0.16 | 0.00% | 0.68% | 96.75% |
| EFOXII | 13 / 24 | 0.08 | 0.00% | 0.00% | 84.63% |
| FEKKAJ | 11 / 75 | 0.09 | 48.79% | 0.00% | 38.51% |
| HOHZAG | 11 / 0 | 0.18 | 68.51% | 3.17% | 29.16% |
| UYOVUA | 12 / 0 | 0.12 | 0.00% | 84.98% | 85.88% |
| ZAFPOP | 12 / 14 | 0.00 | 4.41% | 0.00% | 71.02% |

Ring-type shares overlap and are nonadditive. CIXYUD/DUHHIX symmetry caps are retained. UYOVUA’s 13-ring is outside the macrocycle path; production FK confirms two base torsions and no macrocycle joints. Three ligands have nonaromatic ≥9 rings outside that path: LOPCAW (16), MIVXAR (16), UYOVUA (13). Each has four input donors and fails the all-single-bond test. Actual path status comes from production decomposition, including its small-donor-ring fallback. Eligibility and rigidity counts use the crystal-input topology; the separate UYOVUA check uses actual production FK. No topology change was made.

Aromatic rings with share >20% use the stored A4b pool atom map, the selected row’s embed variant for attachment identification, and physical initial conformers across all native variants in that seed’s pool. The initial free-embed mapping/first-variant calculation was corrected before opening Phase 2 comparisons.

| Parent | Ring share | Ligand / joint | Row−crystal rad | Closest initial seed rad | ≤0.2 rad? |
|---|---:|---|---:|---:|---|
| BABNAV | 34.62% | 1 / 3 | -2.80 | 0.77 | False |
| BABNAV | 31.31% | 1 / 1 | 0.19 | 0.06 | True |
| BABNAV | 31.31% | 1 / 2 | 2.07 | 0.00 | True |
| CIXYUD | 35.02% | 0 / 0 | -2.49 | 0.01 | True |
| DUHHIX | 23.24% | 2 / 0 | 2.50 | 0.01 | True |
| DUHHIX | 26.44% | 4 / 0 | -0.25 | 0.04 | True |
| EFOXII | 30.02% | none | — | — | no attachment DOF |
| ZAFPOP | 52.67% | 0 / 0 | -0.38 | 0.55 | False |

## Flags, runner and validation

- K: start_kabsch fits each row after torsion/bend seeding, preserving orientation diversification.
- U: beam_union ports the C-1 target-aware beam, rebuilt per seed/candidate/sample; unions it with the free-ligand beam. Target share 0.5, wrapped duplicate tolerance 1e-6 rad, group batch cap 16; all registered and read at import. Duplicates advance within the same ranked beam; remaining unique seeds precede unavoidable padding.
- T: pool_window_fill retains finite distinct rows beyond the score window up to the existing cap. Ranking, top-1 and solved-row budget are unchanged. This addresses window loss, not SIVBUV collapse.
- S1: runner seed budget single_w48, with W/seed_budget and row-steps in every record. Each independent W48 seed is compared with C’s W16 seed-11–13 union. Three wide seeds provide noise replicates.

All flags default off; default fingerprint stays 5f74b14b57ed. Three-parent --cpu-identity checks match exported coordinates, scores and MACE energies; the full-precision NPZ supplement matches coordinates/scores bit-for-bit against b0134a2. Nine focused tests pass. HOXJUZ seed 11 full-200-step replay/eager fast FK, replay/eager reference FK and CPU package-loop parity pass with zero score differences. No replayed source path changed, so solver_replay.py is unchanged.

Full CPU suite: 374 passed, 85 skipped, 3 xfailed, 2 missing-worktree-input failures. The multiseed test passes with the shared-DB override; the unrelated reader-path failure remains recorded. CUDA tests were excluded from that concurrent suite to preserve one GPU job.

**Golden check FAILED:** 8/245 arrays in each production/u1 profile differ under one-worker CPU execution; all 231 rescore arrays match. The unchanged b0134a2 engine reproduces the same eight solve-array differences (OJIMIF, CBTYCO, SOKVET, YAVFEG) under that CPU layout. No reference was changed.

## Phase 2 results

Values are primary-population means over parent seed means; differences are arm minus C, with parent-clustered 95% CIs. Coverage differences are percentage points. Spearman is score vs energy (ideal correlation is negative); undefined values remain null. For S1, C means refer to the three-seed union.

### K

| Metric | C | Arm | Paired difference [95% CI] | N parents |
|---|---:|---:|---:|---:|
| Floor ΔE kJ/mol | 61.30 | 59.68 | -1.62 [-3.79, 0.21] | 118 |
| Top-1 ΔE kJ/mol | 75.32 | 73.90 | -1.42 [-4.74, 2.15] | 118 |
| Minimum heavy RMSD Å | 0.92 | 0.91 | -0.01 [-0.04, 0.01] | 118 |
| Coverage ≤1 Å | 68.36 | 67.51 | -0.85 [-3.95, 2.26] | 118 |
| Coverage ≤0.5 Å | 32.49 | 33.90 | 1.41 [-2.26, 4.80] | 118 |
| Minimum shell RMSD Å | 0.14 | 0.14 | -0.00 [-0.01, 0.00] | 118 |
| Shell ≤0.3 Å | 92.09 | 92.37 | 0.28 [-1.41, 1.69] | 118 |
| Shell ≤0.5 Å | 97.18 | 97.46 | 0.28 [-1.69, 2.54] | 118 |
| Distinct rows | 32.45 | 32.85 | 0.40 [0.08, 0.73] | 118 |
| Spearman ρ | -0.56 | -0.56 | 0.00 [-0.02, 0.03] | 114 |

Wall ratio: 0.95x [0.92, 0.98]. Guard: trips 2, mirror 7, arm rotation null 1, C rotation null 0. Equal-compute prerequisite: True. Failures: NEPNEF.

Trip parents: OMAGAK, SESWIZ. Mirror parents: AWONOR, BOHWUT, JEGPUJ, QUTWOT, UMUBAF, XACMOE, XUKKIA.

### U

| Metric | C | Arm | Paired difference [95% CI] | N parents |
|---|---:|---:|---:|---:|
| Floor ΔE kJ/mol | 61.30 | 59.16 | -2.14 [-5.88, 0.35] | 118 |
| Top-1 ΔE kJ/mol | 75.32 | 76.43 | 1.11 [-2.34, 4.57] | 118 |
| Minimum heavy RMSD Å | 0.92 | 0.88 | -0.04 [-0.08, -0.01] | 118 |
| Coverage ≤1 Å | 68.36 | 69.21 | 0.85 [-2.54, 4.52] | 118 |
| Coverage ≤0.5 Å | 32.49 | 34.18 | 1.69 [-2.26, 5.93] | 118 |
| Minimum shell RMSD Å | 0.14 | 0.14 | -0.01 [-0.01, -0.00] | 118 |
| Shell ≤0.3 Å | 92.09 | 93.50 | 1.41 [0.00, 2.82] | 118 |
| Shell ≤0.5 Å | 97.18 | 97.74 | 0.56 [-0.85, 2.54] | 118 |
| Distinct rows | 32.45 | 33.53 | 1.08 [0.60, 1.61] | 118 |
| Spearman ρ | -0.56 | -0.56 | -0.01 [-0.04, 0.03] | 114 |

Wall ratio: 1.05x [1.01, 1.10]. Guard: trips 5, mirror 5, arm rotation null 1, C rotation null 0. Equal-compute prerequisite: True. Failures: NEPNEF.

Trip parents: LUQVOK, NUHPEL, NUWFAM, UMUBAF, YOSXIP. Mirror parents: AWONOR, BOHWUT, DAMVAQ, HEDJOP, QUTWOT.

### T

| Metric | C | Arm | Paired difference [95% CI] | N parents |
|---|---:|---:|---:|---:|
| Floor ΔE kJ/mol | 61.30 | 61.30 | 0.00 [0.00, 0.00] | 118 |
| Top-1 ΔE kJ/mol | 75.32 | 75.32 | 0.00 [0.00, 0.00] | 118 |
| Minimum heavy RMSD Å | 0.92 | 0.92 | -0.00 [-0.00, 0.00] | 118 |
| Coverage ≤1 Å | 68.36 | 68.36 | 0.00 [0.00, 0.00] | 118 |
| Coverage ≤0.5 Å | 32.49 | 32.49 | 0.00 [0.00, 0.00] | 118 |
| Minimum shell RMSD Å | 0.14 | 0.14 | -0.00 [-0.00, 0.00] | 118 |
| Shell ≤0.3 Å | 92.09 | 92.09 | 0.00 [0.00, 0.00] | 118 |
| Shell ≤0.5 Å | 97.18 | 97.18 | 0.00 [0.00, 0.00] | 118 |
| Distinct rows | 32.45 | 35.33 | 2.88 [0.87, 6.09] | 118 |
| Spearman ρ | -0.56 | -0.57 | -0.01 [-0.02, -0.00] | 114 |

Wall ratio: 0.96x [0.92, 0.98]. Guard: trips 0, mirror 0, arm rotation null 0, C rotation null 0. Equal-compute prerequisite: True. Failures: NEPNEF.

Trip parents: none. Mirror parents: none.

### S1

| Metric | C | Arm | Paired difference [95% CI] | N parents |
|---|---:|---:|---:|---:|
| Floor ΔE kJ/mol | 57.11 | 57.11 | -0.00 [-1.48, 1.32] | 118 |
| Top-1 ΔE kJ/mol | 73.70 | 73.80 | 0.10 [-3.57, 4.67] | 118 |
| Minimum heavy RMSD Å | 0.85 | 0.83 | -0.02 [-0.05, 0.01] | 118 |
| Coverage ≤1 Å | 72.88 | 73.45 | 0.56 [-2.54, 3.67] | 118 |
| Coverage ≤0.5 Å | 40.68 | 39.83 | -0.85 [-5.08, 3.39] | 118 |
| Minimum shell RMSD Å | 0.13 | 0.13 | 0.00 [-0.01, 0.01] | 118 |
| Shell ≤0.3 Å | 94.07 | 92.94 | -1.13 [-2.54, 0.00] | 118 |
| Shell ≤0.5 Å | 97.46 | 98.31 | 0.85 [-1.13, 3.39] | 118 |
| Distinct rows | 56.47 | 83.36 | 26.89 [20.02, 34.62] | 118 |
| Spearman ρ | -0.56 | -0.63 | -0.07 [-0.10, -0.03] | 115 |

Wall ratio: 0.99x [0.82, 1.21]. Guard: trips 2, mirror 3, arm rotation null 0, C rotation null 0. Equal-compute prerequisite: True. Failures: NEPNEF.

Trip parents: SESWIZ, XEVDAE. Mirror parents: AWONOR, UMUBAF, XACMOE.

| Arm | Floor/coverage | Top-1 | Guard | Wall | Selected |
|---|---|---|---|---|---|
| K | FAIL | FAIL | PASS | PASS | False |
| U | FAIL | FAIL | FAIL | PASS | False |
| T | FAIL | PASS | PASS | PASS | False |
| S1 | FAIL | FAIL | PASS | PASS | False |

Selection outcome: no qualifying arm.

End-to-end wall includes prepare, seed generation, solve and assembly, excluding MACE/export. Shared preparation is charged equally. Every independent S1 seed pays full preparation, including native E/Z preparation; only the three-replicate seed/solve total is divided by three. Raw timing/row-step ledgers are preserved; finalise.py reconstructs metadata after both record writers finish. The three-parent pilot projected 93 minutes including MACE/export; the actual 190.68 minutes was about twice that estimate. Large-pool/symmetry-heavy parents and the S1 resource retries were not represented adequately by the pilot.

## Arm-specific measurements

K full-placement rerun: pooled row-max median 2.8071 Å (versus 3.1526 Å C), 101/120 parents still above 0.3 Å.

U primary-population beam wall: 141.88 s; 9.74% of its charged generator wall. Beam timing includes shared and target-aware work, rebuilding per seed.

S1 row-step checks: 119 matched, NEPNEF unavailable due embedding failure, zero mismatches. Every primary parent matches exactly. S1 did not meet selection rules; no seed-budget recommendation.



| Tiny parent | C rows (11/12/13) | T rows (11/12/13) |
|---|---|---|
| QAPFAS | 1 / 1 / 1 | 1 / 1 / 1 |
| UYOVUA | 1 / 1 / 1 | 1 / 1 / 1 |
| MIGTEE | 1 / 1 / 1 | 1 / 1 / 1 |
| MIVXAR | 1 / 1 / 1 | 1 / 1 / 1 |
| ZAZREZ | 1 / 2 / 2 | 1 / 2 / 2 |
| SIVBUV | 2 / 2 / 2 | 2 / 2 / 2 |
| HOXJUZ | 6 / 10 / 9 | 6 / 12 / 9 |

## Named cases (Phase 2)

Each cell is top-1 / floor unrelaxed ΔE in kJ/mol. Phase 3 was not run; these are local-screen measurements.

| Parent | Seed | C | K | U | T | S1 |
|---|---:|---:|---:|---:|---:|---:|
| BABNAV | 11 | 36.15 / 27.68 | 35.36 / 28.22 | 31.64 / 22.57 | 36.15 / 27.68 | 32.67 / 26.74 |
| BABNAV | 12 | 39.22 / 26.54 | 32.66 / 30.15 | 37.54 / 26.94 | 39.22 / 26.54 | 37.23 / 27.14 |
| BABNAV | 13 | 36.19 / 25.95 | 24.67 / 24.67 | 24.58 / 24.58 | 36.19 / 25.95 | 37.67 / 26.64 |
| CIXYUD | 11 | 77.45 / 61.96 | 67.15 / 61.06 | 86.93 / 51.58 | 77.45 / 61.96 | 52.43 / 47.19 |
| CIXYUD | 12 | 73.93 / 59.34 | 85.63 / 60.05 | 74.68 / 64.77 | 73.93 / 59.34 | 73.40 / 53.05 |
| CIXYUD | 13 | 64.43 / 58.17 | 60.07 / 52.76 | 59.76 / 59.76 | 64.43 / 58.17 | 82.89 / 62.52 |
| DOHTOM | 11 | 126.80 / 113.32 | 128.26 / 113.59 | 112.26 / 111.76 | 126.80 / 113.32 | 118.63 / 101.68 |
| DOHTOM | 12 | 125.23 / 111.36 | 99.25 / 99.25 | 114.30 / 112.33 | 125.23 / 111.36 | 118.26 / 103.54 |
| DOHTOM | 13 | 112.46 / 107.40 | 108.09 / 108.09 | 116.58 / 111.22 | 112.46 / 107.40 | 124.49 / 99.52 |
| DUHHIX | 11 | 140.48 / 140.48 | 154.90 / 154.90 | 156.83 / 149.32 | 140.48 / 140.48 | 161.36 / 140.43 |
| DUHHIX | 12 | 167.16 / 148.86 | 157.67 / 156.30 | 162.56 / 150.66 | 167.16 / 148.86 | 172.70 / 139.68 |
| DUHHIX | 13 | 145.44 / 145.44 | 183.44 / 152.61 | 157.38 / 150.77 | 145.44 / 145.44 | 152.87 / 143.15 |
| EFOXII | 11 | 16.73 / 16.25 | 16.73 / 16.25 | 16.73 / 16.25 | 16.73 / 16.25 | 16.67 / 14.19 |
| EFOXII | 12 | 17.52 / 14.79 | 17.52 / 14.79 | 17.52 / 14.79 | 17.52 / 14.79 | 17.16 / 13.53 |
| EFOXII | 13 | 17.06 / 14.39 | 17.06 / 14.39 | 17.06 / 14.39 | 17.06 / 14.39 | 17.67 / 14.39 |
| HOHZAG | 11 | 104.85 / 104.85 | 104.73 / 104.73 | 105.54 / 105.54 | 104.85 / 104.85 | 105.51 / 105.51 |
| HOHZAG | 12 | 105.45 / 105.45 | 107.06 / 107.06 | 106.50 / 106.50 | 105.45 / 105.45 | 105.21 / 105.21 |
| HOHZAG | 13 | 105.00 / 105.00 | 105.68 / 105.68 | 105.92 / 105.92 | 105.00 / 105.00 | 105.33 / 105.33 |
| UYOVUA | 11 | 68.87 / 68.87 | 68.82 / 68.82 | 68.82 / 68.82 | 68.87 / 68.87 | 68.81 / 68.81 |
| UYOVUA | 12 | 68.86 / 68.86 | 68.86 / 68.86 | 68.85 / 68.85 | 68.86 / 68.86 | 68.87 / 68.87 |
| UYOVUA | 13 | 68.82 / 68.82 | 68.87 / 68.87 | 68.87 / 68.87 | 68.82 / 68.82 | 68.81 / 68.81 |
| ZAFPOP | 11 | 40.90 / 23.70 | 29.46 / 23.02 | 31.15 / 19.12 | 40.90 / 23.70 | 39.36 / 19.59 |
| ZAFPOP | 12 | 40.17 / 18.65 | 42.71 / 22.48 | 35.10 / 20.57 | 40.17 / 18.65 | 41.59 / 22.31 |
| ZAFPOP | 13 | 34.51 / 23.58 | 30.22 / 19.08 | 39.92 / 22.30 | 34.51 / 23.58 | 42.31 / 20.39 |
| QAPFAS | 11 | 10.30 / 10.30 | 10.18 / 10.18 | 10.13 / 8.96 | 10.30 / 10.30 | 10.08 / 10.08 |
| QAPFAS | 12 | 9.54 / 9.54 | 10.16 / 10.16 | 10.23 / 6.66 | 9.54 / 9.54 | 10.21 / 10.21 |
| QAPFAS | 13 | 10.05 / 10.05 | 9.57 / 9.57 | 10.15 / 8.47 | 10.05 / 10.05 | 10.15 / 10.15 |
| SIVBUV | 11 | 65.21 / 45.07 | 66.10 / 66.10 | 66.32 / 60.20 | 65.21 / 45.07 | 64.75 / 64.75 |
| SIVBUV | 12 | 66.41 / 66.41 | 64.71 / 64.71 | 65.77 / 43.50 | 66.41 / 66.41 | 65.42 / 65.42 |
| SIVBUV | 13 | 66.47 / 66.47 | 66.25 / 66.25 | 65.26 / 60.45 | 66.47 / 66.47 | 65.51 / 65.51 |
| ZAZREZ | 11 | 69.28 / 69.28 | 69.24 / 67.53 | 69.43 / 69.43 | 69.28 / 69.28 | 69.32 / 69.32 |
| ZAZREZ | 12 | 69.26 / 68.32 | 69.39 / 67.57 | 69.26 / 68.77 | 69.26 / 68.32 | 69.26 / 68.80 |
| ZAZREZ | 13 | 69.30 / 68.62 | 69.34 / 67.41 | 69.43 / 69.43 | 69.30 / 68.62 | 69.34 / 68.48 |
| HOXJUZ | 11 | 282.54 / 274.84 | 284.97 / 268.70 | 277.55 / 276.61 | 282.54 / 274.84 | 278.60 / 268.79 |
| HOXJUZ | 12 | 279.16 / 267.33 | 285.56 / 268.35 | 282.46 / 268.77 | 279.16 / 267.33 | 284.70 / 271.75 |
| HOXJUZ | 13 | 281.88 / 268.32 | 286.37 / 269.65 | 280.32 / 270.28 | 281.88 / 268.32 | 285.42 / 274.65 |
| XACMOE | 11 | 154.52 / 114.59 | 97.99 / 97.99 | 148.59 / 111.49 | 154.52 / 114.59 | 141.70 / 104.84 |
| XACMOE | 12 | 165.63 / 109.65 | 126.31 / 126.31 | 111.41 / 111.41 | 165.63 / 109.65 | 134.30 / 93.66 |
| XACMOE | 13 | 107.71 / 107.71 | 114.89 / 114.89 | 147.52 / 109.89 | 107.71 / 107.71 | 135.09 / 111.11 |
| JEGPUJ | 11 | 140.07 / 135.65 | 45.29 / 45.29 | 145.05 / 145.05 | 140.07 / 135.65 | 113.40 / 113.40 |
| JEGPUJ | 12 | 140.86 / 140.86 | 57.47 / 57.14 | 128.71 / 128.71 | 140.86 / 140.86 | 126.72 / 126.72 |
| JEGPUJ | 13 | 152.68 / 136.32 | 79.62 / 79.62 | 122.30 / 122.30 | 152.68 / 136.32 | 117.19 / 117.19 |
| NEPNEF | 11 | failed | failed | failed | failed | failed |
| NEPNEF | 12 | failed | failed | failed | failed | failed |
| NEPNEF | 13 | failed | failed | failed | failed | failed |
| CAXTAX | 11 | 71.00 / 58.22 | 79.18 / 45.79 | 91.72 / 56.42 | 71.00 / 58.22 | 82.27 / 54.83 |
| CAXTAX | 12 | 89.91 / 59.28 | 80.28 / 37.48 | 70.96 / 44.37 | 89.91 / 59.28 | 95.49 / 43.78 |
| CAXTAX | 13 | 71.60 / 52.58 | 72.39 / 34.79 | 62.50 / 51.95 | 71.60 / 52.58 | 66.90 / 52.65 |
| FELSAU | 11 | 34.96 / -1.97 | 20.64 / 4.87 | 43.08 / 1.92 | 34.96 / -1.97 | 18.97 / -6.15 |
| FELSAU | 12 | 44.55 / 1.10 | 18.86 / 3.96 | 43.57 / -1.75 | 44.55 / 1.10 | 25.93 / 0.95 |
| FELSAU | 13 | 38.36 / 1.59 | 19.18 / -2.01 | 37.07 / 3.04 | 38.36 / 1.59 | 14.91 / -7.02 |

## Failures and confirmation

NEPNEF failed real GFN2 embedding for all three seeds in every arm (15 failed records total), retained in tables and excluded from primary statistics as registered. S1 emitted two graph-capture fallback warnings on FEKKAJ and two on QEZVEY; subsequent retries completed in replay mode. Warning counters repeat in each seed record, so events are counted once per parent/arm. No union-beam fallback occurred, no enrichment failed, and all 1,800 records are present. The test-path failures and one-worker golden differences are retained above. The screen took 190.68 minutes, versus the 93-minute pilot projection.

Phase 3 gates, L4 strata and L4 named cases: not run, because no arm qualified in Phase 2. All final named-case values above are local-screen values. No VM was started or owned; C-2 spend is 0 units. Final cj.py credits log: balance 476.36 units, usage rate 0.00/hr, active assignments 0. Shared-account spend since its historical START_BALANCE is not C-2 spend.

Machine-readable evidence: c2/C2_PHASE0.json, C2_PHASE1.json, C2_PHASE2.json. Raw pools, timing ledgers, initialization arrays, failure logs and per-parent results are in shared research/data/jgen/campaign/C2/.
