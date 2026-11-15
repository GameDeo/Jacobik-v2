# Stereo panel report — 2026-11-09

**Deviations and missing data:** login21 A100 in resumable bursts, not the contracted two-GPU shared job; QEZVEY excluded; 239 OOM retries in the last job summary. Contrary to the job summary’s zero lane failures, NEPNEF and UYOVUA have failed records for all three seeds in all arms (GFN2 fallback); their structures are invalid. NEPNEF is excluded from primary statistics by the original contract, but the new hard check explicitly covers every parent and seed. No reruns. All arms of each parent shared a GPU. The job summary covers a resumed subset and its memory maxima below cover only those 112 units.

**Verdicts:** AS **fails to qualify on the available evidence**: observed paired thresholds pass, but the hard check requires every parent/seed and 9 units (QEZVEY, NEPNEF, UYOVUA) have no valid comparison. ASP **fails**: returned classes exceed K=8 (maximum 32); the recorded audit also cannot establish retention of every eligible pre-output row. Fresh seeds require a separate owner go.

Paired estimates use 116 valid parents, seed means and parent-cluster bootstrap (10,000 resamples, seed 11). Deltas are arm minus production A; dE in kJ/mol and coverage as fractions. Full metrics and named rows are in summary.json.

| Arm | floor Δ [95% CI] | top-1 Δ [95% CI] | ≤1 Å Δ [95% CI] | ≤0.5 Å Δ [95% CI] | rows Δ [95% CI] | wall ratio |
|---|---:|---:|---:|---:|---:|---:|
| C | 0.0060 [0.0000, 0.0181] | 0.0000 [0.0000, 0.0000] | 0.0029 [0.0000, 0.0086] | 0.0000 [0.0000, 0.0000] | 0.0115 [0.0000, 0.0345] | 0.973 |
| A | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.000 |
| AS | -3.2993 [-8.0356, -0.3434] | -2.6614 [-5.9684, -0.3726] | 0.0374 [0.0086, 0.0747] | 0.0000 [0.0000, 0.0000] | 4.6408 [1.4509, 8.8996] | 1.204 |
| ASP | -2.7438 [-7.5310, 0.3720] | -2.7491 [-6.0689, -0.3726] | -0.0316 [-0.0833, 0.0201] | -0.0489 [-0.0805, -0.0230] | -6.8132 [-9.5833, -3.8502] | 1.218 |

| Arm | duplicate fraction | mean rows | mean classes | bytes |
|---|---:|---:|---:|---:|
| C | 0.02886 | 31.36 | 3.69 | 17410050 |
| A | 0.01292 | 31.35 | 3.70 | 17405126 |
| AS | 0.01126 | 35.99 | 4.39 | 19261889 |
| ASP | 0.00843 | 24.54 | 4.53 | 14653203 |

Parent-clustered CIs for duplicate fraction, classes, rows, stereo presence and wall ratio are in `additional_parent_clustered_metrics` in the summary. Memory was recorded per lane across arms, not per arm: on the final resumed 112 units, peak RSS 2,903,334,912 bytes, peak GPU reserved 12,152,995,840 and allocated 5,897,692,160 bytes. Earlier bursts are not represented by these maxima.

Hard check: 0 violations among 348 valid paired units; 9 units lack valid pairs. Stereo-bearing: 32 represented (one of the 33 excluded); A has 26 present and AS/ASP 29. Named absent set (all three seeds agree):
- DAMVAQ: A [False, False, False]; AS [False, False, False]; ASP [False, False, False].
- FETTAZ: A [False, False, False]; AS [True, True, True]; ASP [True, True, True].
- FUTSOD: A [False, False, False]; AS [False, False, False]; ASP [False, False, False].
- XACMOE: A [False, False, False]; AS [True, True, True]; ASP [True, True, True].
- ZAZREZ: A [False, False, False]; AS [False, False, False]; ASP [False, False, False].
- LISVOC: A [False, False, False]; AS [True, True, True]; ASP [True, True, True].

The A-relative analysis extends `analyse_i.py` with selectable control and the per-unit hard check. `summary.json` retains every metric and failure list; paired wall ratios are sums of standalone-estimated time, not end-to-end job wall.

## QEZVEY follow-up (separate experiment, 2026-11-09)

QEZVEY remains excluded from the contract metrics. An isolated comparison checkout `jgen/q-stereo-check` at `709956f` combined the stereo superset code with the default-off, one-start dedup experiment. The A/AS seed 11 run on login33 A100 timed out after its 480 s parent budget and returned no valid arm records; MPS stopped cleanly. Its three-seed estimate was 1,090 s, but the observed first-seed wall implies more than the 20-minute total limit, so seeds 12–13 were not run. The attempted QEZVEY result is not a contract pass or failure for the observed panel arms. Run and timeout records: `$SCRATCH/jacobik/runs/qperf-stereo-709956f-s11/`; details in `jgen/q-perf/Q_PERF_PROGRESS.md`.
