# Wrap-up checkpoint — 2026-11-09 (in progress)

This report is incomplete; no D3 job or fold-in has been run. Do not treat the items below as passed until their stated validation is complete.

## S — stereo panel

See `jgen-st/STEREO_PANEL_REPORT.md` and `research/data/jgen/stereo/panel_login/`. The branch check on `e8f1d4a` passed fingerprint and golden 0/490; pytest 589 passed, 14 known 2060 CUDA-capture failures, 3 skipped (overall FAIL). Subsequent screen-only flag fix `513107a` has a focused passing test. Supplemental PM checkpoint on archived snapshot: fingerprint passed, golden 0/490; pytest was interrupted at owner direction (partial: 455 passed, 18 failed, 2 skipped; four D3 harness failures attributable to the archive lacking Git metadata, 14 CUDA equality failures). Overall check FAIL/incomplete; do not treat it as a completed pytest suite. The required all-parent hard check is incomplete: QEZVEY was excluded as planned, and NEPNEF and UYOVUA have failed records in all arms and seeds despite the lane-level summary reporting no failures. AS meets observed thresholds but cannot be declared qualified under the full contract; ASP fails the observed per-class K limit.

## D — D2 checkpoint

`jgen/d2-build` through `e6d84a4`: final workstation checkpoint at `fc721de` passed fingerprint `6b47c3fb45a9` and golden 0/490; pytest 608 passed, 14 known CUDA-capture failures, 5 skipped (overall FAIL). Later commits only add diagnostics and reports. The 60-pool rescore reproduced all original aggregate metrics exactly, with one logged xTB retry at 1000 K after a charge -4 SCF failure. `D2_RESCORE_EXPLANATION.md` and `D2_RESCORE_KEYS.md` identify moved terms and concrete lookup fits. AJIXUN and AFOWIB Q5 per-key contributions reconcile with `rank_terms`; a targeted ablation reproduced the six recorded D2 Q5 fits and demonstrated that the charge filter changes all six, with locked exclusions held fixed. Per-key score effects of the filter and independent T1d typing effects remain unmeasured. `D2_CLEANUP_LISTS.md`, `D2_LOCAL_DATA_INVENTORY.md`, and `D2_SCRATCH_INVENTORY.md` provide classified metadata, citations and proposed actions, with stated scan/citation limitations; no removal.

## M — MPS

Intermediate `infra/fastcheck` commits `b861aaf`, `d6c4ea4`, `bd769d5` change MPS lane environment and provide `tools/check/mps_four_gpu_test.sbatch`. Focused tests passed. Single-GPU MPS smoke on login23 using snapshot `bd769d5`: AJAQAD and AWONOR both succeeded, 52 s wall, 53.8 W mean / 100.2 W max, two OOM retries each; daemon stopped cleanly with no leftover PIDs. Output `$SCRATCH/jacobik/runs/mps-login-bd769d5/`. A second smoke on production-baseline test-only branch `infra/mps-test` `56207e6` passed on login33: fingerprint `6b47c3fb45a9`, both AJAQAD/AWONOR records ok, 36 s wall, two OOM retries each, no GFN2 fallback logged, daemon stopped with no leftovers. The earlier snapshot uses the older branch fingerprint noted below. Four-GPU node validation remains pending. The test requires explicit owner approval for one debug QOS 4-A100 node, up to 10 minutes; do not submit without it. Prepared snapshot: `$SCRATCH/jacobik/ext/mps-test-1f33319` (COMMIT `56207e6`), script SHA-256 `9de7197b9171ab871efa06767d2e96ef2b4e1d6b6151103aaa9ea19707b43b40`. After explicit owner go for this job and debug QOS, exact command on Perlmutter: `cd "$SCRATCH/jacobik/ext/mps-test-1f33319" && export JK_SRC="$PWD" && sbatch tools/check/mps_four_gpu_test.sbatch`. The script has 1 node, 4 A100, and a 10-minute ceiling; it has not been submitted.

## R — runner

Intermediate `b861aaf` and `d6c4ea4`, plus packed-control follow-up: failed canaries reject; timeout handling, retry and slow-parent admission are implemented. Packed control comparisons and storage stage one parent at a time. Production-baseline test-only `infra/mps-test` `56207e6`: 25 focused runner/node-verifier tests passed. PM checkpoint fingerprint passed, golden 0/490; pytest 585 passed, 18 failed, 9 skipped. Four D3-harness failures were caused by the `git archive` snapshot lacking `.git`; all 66 D3-harness tests passed when rerun in a real Git checkout. The remaining 14 are CUDA eager-versus-replay strict-equality failures; they were not diagnosed further at owner direction. Overall PM check remains FAIL.

## I — cached isomer control

See stereo branch report (`jgen/i-stereo` `513107a`); `infra/fastcheck` now applies arm overrides and supplies defaults for unspecified overrides, while the screen applies registered flags. Focused runner tests passed 24/24, live screen flag test passed. The integration needs a fresh full check. A two-parent dev run restored both A records and its canary passed 2/2; cached run 157 s vs initial run 193 s. AS-only records have null control_union_distinct; the restored A records retain this field.

## A — A4 analysis

See `jgen-cov/COV_PROGRESS.md` (commits `6e9c3a1`, `a39a574`, `973a47a`). Both arms and analyses produced byte-identical outputs on the gate data. At 16 workers control v1/v2 took 60.56/55.19 s and chair v1/v2 52.52/39.52 s; control v1 took 52.09 s at 23 workers. Workstation checkpoint fingerprint passed and golden 0/490; pytest 608 passed, 14 known CUDA-capture failures, 3 skipped. Overall check reported FAIL for those failures.

## Q — QEZVEY

`jgen/q-perf` through `3e03469`: the default-off one-start experimental arm completed QEZVEY seed 11 in 114.6 s versus 472.0 s paired default on login33 A100, MPS. Both had 192 rows, identical row keys and paired scores, and identical top-1 geometry/energy. Two OOM retries per arm. Seven fixed golden SOLVE cases with the experimental flag matched all 14 coordinate/score arrays within standard tolerances. The flag can change merge metadata or retain additional duplicate rows and has not passed a panel quality gate. Default workstation check passed fingerprint and 0/490 golden; pytest 581 passed, 14 CUDA-capture failures, 3 skipped (overall FAIL). A separate A/AS stereo seed-11 attempt on `jgen/q-stereo-check` `709956f` timed out at 480 s; all three seeds would exceed the 20-minute cap, so QEZVEY remains excluded. Default-branch workstation checkpoint through `9e18ee8`: fingerprint passed, golden 0/490; pytest 581 passed, 14 known 2060 CUDA-capture failures, 3 skipped (overall FAIL). See `Q_PERF_PROGRESS.md`.

## E — D3 estimate

Not run: M's four-GPU validation has not passed.

## Baseline discrepancy

`infra/fastcheck` at its inherited baseline has fingerprint `5f74b14b57ed`, not the directive's `6b47c3fb45a9`; its workstation check cannot establish the latter without integration with the approved production baseline. No branches have been merged.
