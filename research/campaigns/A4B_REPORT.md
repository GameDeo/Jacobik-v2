# A4b production baseline

## Provenance

- Commit: `fd79db9e415e`; fingerprint(s): `273f2ccbf892`.
- Constraints DB SHA-256: `d87b81a5d6297e0251a881b871f477678ede79fc2bce694b72c4dba1c5fe10c4`; meta SHA-256: `66558e6e2910cb5255f6aea01d37976fd17a81c54b29f7d4617412e0f2700709`.
- VM: NVIDIA L4; seeds 11, 12, 13; panels cal and tail; arm control; production flags/defaults.
- Unique parents attempted: 125; units attempted by panel: cal 360, tail 30; ok/failed: cal 357/3, tail 27/3.
- Colab spend: 2.520 units for the successful full run; 5.270 units for this task including smoke and the discarded attempt. Aggregate VM-active time: 80.4 min; elapsed full-run wall: 43.5 min; solve wall sum: 1765.1 s; xTB relaxation wall sum: 3899.3 s; CUDA MACE wall: 45.4 s.

## Q1: coverage and relaxed ranking

### cal

| Bucket | Share (mean, 95% parent-bootstrap CI; parent median) |
|---|---:|
| U-TOP1 | 0.32 [0.25, 0.40], median 0.00 |
| U-POOL | 0.33 [0.26, 0.41], median 0.00 |
| U-NONE | 0.34 [0.26, 0.43], median 0.00 |
| R-TOP1 | 0.43 [0.36, 0.52], median 0.33 |
| R-RANK | 0.19 [0.13, 0.25], median 0.00 |
| R-MISS | 0.38 [0.30, 0.46], median 0.00 |

Seed consistency counts per parent (number of seeds in each bucket):

| Bucket | 3/3 | 2/3 | 1/3 | 0/3 |
|---|---:|---:|---:|---:|
| U-TOP1 | 28 | 10 | 11 | 70 |
| U-POOL | 27 | 10 | 18 | 64 |
| U-NONE | 34 | 9 | 3 | 73 |
| R-TOP1 | 38 | 14 | 13 | 54 |
| R-RANK | 11 | 7 | 20 | 81 |
| R-MISS | 35 | 11 | 8 | 65 |

### tail

| Bucket | Share (mean, 95% parent-bootstrap CI; parent median) |
|---|---:|
| U-TOP1 | 0.11 [0.00, 0.33], median 0.00 |
| U-POOL | 0.11 [0.00, 0.33], median 0.00 |
| U-NONE | 0.78 [0.44, 1.00], median 1.00 |
| R-TOP1 | 0.15 [0.00, 0.37], median 0.00 |
| R-RANK | 0.04 [0.00, 0.11], median 0.00 |
| R-MISS | 0.81 [0.56, 1.00], median 1.00 |

Seed consistency counts per parent (number of seeds in each bucket):

| Bucket | 3/3 | 2/3 | 1/3 | 0/3 |
|---|---:|---:|---:|---:|
| U-TOP1 | 1 | 0 | 0 | 8 |
| U-POOL | 1 | 0 | 0 | 8 |
| U-NONE | 7 | 0 | 0 | 2 |
| R-TOP1 | 1 | 0 | 1 | 7 |
| R-RANK | 0 | 0 | 1 | 8 |
| R-MISS | 7 | 0 | 1 | 1 |

## Q2: score inversions

### cal

- Obscene share: 0.11 [0.06, 0.17], median 0.00.
- Top-1 inversion share: 0.03 [0.01, 0.05], median 0.00.
- Pair inversion rate: 0.12 [0.09, 0.16], median 0.06.
- Ranking loss: 16.94 [12.45, 21.94], median 6.01 kJ/mol.

Top-1 inversion units:

- HEDJOP seed 13
- MIDQEY seed 11
- MIDQEY seed 12
- MIDQEY seed 13
- NUWFAM seed 11
- QUTWOT seed 11
- VAVBIE seed 12
- VAVBIE seed 13
- XEVDAE seed 12

20 largest unit ranking losses:

| Code | Seed | Ranking loss | Top-1 dE |
|---|---:|---:|---:|
| QUTWOT | 11 | 196.83 | 268.00 |
| MIDQEY | 11 | 183.42 | 234.69 |
| MIDQEY | 13 | 172.05 | 207.56 |
| HEDJOP | 13 | 167.83 | 399.13 |
| MIDQEY | 12 | 156.07 | 206.44 |
| XEVDAE | 12 | 154.44 | 266.90 |
| VAVBIE | 13 | 124.91 | 269.87 |
| NUWFAM | 11 | 124.80 | 247.93 |
| VAVBIE | 12 | 109.64 | 234.75 |
| XEYSUP | 13 | 91.59 | 104.21 |
| XEYSUP | 12 | 88.67 | 104.14 |
| XEYSUP | 11 | 87.04 | 104.73 |
| HEDJOP | 11 | 85.37 | 338.39 |
| MEJYIN | 13 | 83.48 | 133.05 |
| FEXWUC | 13 | 79.57 | 111.58 |
| QUTWOT | 13 | 78.80 | 155.26 |
| FEXWUC | 11 | 76.63 | 107.23 |
| XUKKIA | 11 | 74.17 | 15.98 |
| NUWFAM | 13 | 68.95 | 232.33 |
| RUJCUW | 12 | 68.80 | 90.78 |

### tail

- Obscene share: 0.85 [0.63, 1.00], median 1.00.
- Top-1 inversion share: 0.15 [0.00, 0.37], median 0.00.
- Pair inversion rate: 0.30 [0.10, 0.49], median 0.31.
- Ranking loss: 140.63 [5.04, 397.44], median 5.91 kJ/mol.

Top-1 inversion units:

- SOBHAS seed 11
- SOBHAS seed 12
- SOBHAS seed 13
- YUCZUS seed 12

20 largest unit ranking losses:

| Code | Seed | Ranking loss | Top-1 dE |
|---|---:|---:|---:|
| SOBHAS | 13 | 1198.93 | 1616.95 |
| SOBHAS | 11 | 1158.61 | 1605.36 |
| SOBHAS | 12 | 1084.50 | 1573.38 |
| YUCZUS | 12 | 119.56 | 249.72 |
| YUCZUS | 13 | 80.97 | 218.45 |
| ZELTAP | 12 | 21.11 | 114.57 |
| JEGPUJ | 11 | 20.79 | 484.98 |
| JEGPUJ | 12 | 20.53 | 476.09 |
| JEGPUJ | 13 | 20.04 | 483.50 |
| USAPOU | 11 | 17.74 | 410.29 |
| DAMVAQ | 12 | 14.55 | 454.07 |
| YUCZUS | 11 | 13.60 | 142.71 |
| HOXJUZ | 11 | 11.12 | 279.14 |
| ZELTAP | 11 | 9.44 | 112.28 |
| ZELTAP | 13 | 5.07 | 111.11 |
| HOXJUZ | 13 | 0.54 | 269.00 |
| DAMVAQ | 11 | 0.00 | 444.98 |
| DAMVAQ | 13 | 0.00 | 415.06 |
| HOXJUZ | 12 | 0.00 | 268.89 |
| IFUROQ | 11 | 0.00 | 264.00 |

## Q3: strain and basin offset

### cal

- Top-1 strain: 70.46 [57.14, 85.40], median 43.95 kJ/mol.
- Top-1 basin offset: 20.01 [13.54, 27.16], median 8.52 kJ/mol.
- Top-1 strain share: 0.85 [0.77, 0.96], median 0.84.
- Top-1 contact changes: 95 units.

### tail

- Top-1 strain: 378.64 [169.91, 708.00], median 207.63 kJ/mol.
- Top-1 basin offset: 81.28 [37.77, 128.98], median 47.55 kJ/mol.
- Top-1 strain share: 0.73 [0.60, 0.86], median 0.71.
- Top-1 contact changes: 18 units.

## Sensitivity

### cal

| Hit radius / equivalent offset | R-TOP1 share | R-RANK share | R-MISS share |
|---|---:|---:|---:|
| hit0.3_eq2 | 0.35 [0.28, 0.43], median 0.00 | 0.21 [0.15, 0.27], median 0.00 | 0.44 [0.36, 0.52], median 0.33 |
| hit0.3_eq5 | 0.43 [0.36, 0.52], median 0.33 | 0.19 [0.13, 0.25], median 0.00 | 0.38 [0.30, 0.46], median 0.00 |
| hit0.3_eq10 | 0.54 [0.46, 0.62], median 0.67 | 0.18 [0.12, 0.23], median 0.00 | 0.28 [0.21, 0.36], median 0.00 |
| hit0.5_eq2 | 0.39 [0.31, 0.47], median 0.00 | 0.24 [0.17, 0.30], median 0.00 | 0.38 [0.30, 0.46], median 0.00 |
| hit0.5_eq5 | 0.46 [0.38, 0.54], median 0.33 | 0.21 [0.15, 0.27], median 0.00 | 0.33 [0.25, 0.41], median 0.00 |
| hit0.5_eq10 | 0.56 [0.48, 0.64], median 0.67 | 0.19 [0.13, 0.25], median 0.00 | 0.25 [0.18, 0.32], median 0.00 |

### tail

| Hit radius / equivalent offset | R-TOP1 share | R-RANK share | R-MISS share |
|---|---:|---:|---:|
| hit0.3_eq2 | 0.04 [0.00, 0.11], median 0.00 | 0.00 [0.00, 0.00], median 0.00 | 0.96 [0.89, 1.00], median 1.00 |
| hit0.3_eq5 | 0.15 [0.00, 0.37], median 0.00 | 0.04 [0.00, 0.11], median 0.00 | 0.81 [0.56, 1.00], median 1.00 |
| hit0.3_eq10 | 0.15 [0.00, 0.37], median 0.00 | 0.04 [0.00, 0.11], median 0.00 | 0.81 [0.56, 1.00], median 1.00 |
| hit0.5_eq2 | 0.11 [0.00, 0.33], median 0.00 | 0.00 [0.00, 0.00], median 0.00 | 0.89 [0.67, 1.00], median 1.00 |
| hit0.5_eq5 | 0.22 [0.00, 0.56], median 0.00 | 0.00 [0.00, 0.00], median 0.00 | 0.78 [0.44, 1.00], median 1.00 |
| hit0.5_eq10 | 0.22 [0.00, 0.56], median 0.00 | 0.00 [0.00, 0.00], median 0.00 | 0.78 [0.44, 1.00], median 1.00 |

## Failures and deviations

- cal: `{"reference_relax_failed": 0, "relax_capped": 9, "relax_failed": 0, "unit_failure_reasons": {"RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)": 3}, "unit_status": {"failed": 3, "ok": 357}}`; fallback exclusions: `["NEPNEF"]`.
- tail: `{"reference_relax_failed": 0, "relax_capped": 0, "relax_failed": 0, "unit_failure_reasons": {"RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)": 3}, "unit_status": {"failed": 3, "ok": 27}}`; fallback exclusions: `["NEPNEF"]`.
- Embed fallback parents: NEPNEF.
- NEPNEF: 3 failed of 3 attempted seeds (11,12,13); no valid pool rows/manifests were produced, per Amendment 2.
- A first full-run attempt was discarded before fetch after a controller exception; it consumed 2.120 units and is not included in these measurements. The successful retry used the same committed code.
- Scheduling deviation: each VM began the CPU xTB relaxation batch after its complete GPU solve batch, so xTB did not overlap the GPU solves as §3 Step 3 specified. CUDA MACE evaluation also began after each VM's solves.
- U-bucket disagreements with bench coverage fields: 0; see `coverage_disagreements.json`.
- Charge/multiplicity lookup failures: none.
- Missing or incomplete records: see `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/A4b/missing_units.json`.
