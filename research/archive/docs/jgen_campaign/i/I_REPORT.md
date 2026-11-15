# Campaign I final report — 2026-11-08

## 1. Merge and validation

| Check | Result |
|---|---|
| Pinned master | 235c176 merged; both profile registries retained |
| Experiment-branch default fingerprint | 5f74b14b57ed |
| Isomer unit tests | 16 passed |
| Merged recorded-pool exactness | 13/13 capped, 2/2 sampled uncapped, all seeds 11–13; 105 duplicate rows recovered |
| Perlmutter golden, unchanged defaults | 0/490 unexpected array differences, production + u1 |
| pm-quick pytest | FAIL: 570 passed, 18 failed, 9 skipped |
| pm-quick dev suite | PASS: 39 parents × 2 arms, zero failed/timeouts/not-run |
| Full CPU-only pytest, experiment branch | PASS: 574 passed, zero failed, 23 skipped; 857.4 s |

Four D3 pytest failures require Git metadata absent from the prescribed archive snapshot; all four pass locally with Git metadata. Fourteen CUDA eager/replay equality failures (`test_finish_phase.py::test_phase_cuda_capture`, 14 of 18 parameter sets, eager vs replay scores differ at about 1e-6) are **pre-existing on the login A100, not from this package**: the same 14 tests, in one 4.6-minute burst each on an idle A100, fail identically on master `235c176` (`~/repos/jacobik`, read-only) and on the isomer snapshot (14 failed, 4 passed in both; 278 s and 274 s; run dirs `$SCRATCH/jacobik/runs/iso-cudatest-{master,snap}`). Passing in both: IA-AHIGOQ, IA-PIDXOT, R2s-AHIGOQ, R2s-PIDXOT. Solver and generated replay files equal pinned master. Full pm-quick took 1208.2 s (8.2 s beyond the 20-minute target); its GPU dev phase took 301.8 s. GPU dedup from master is retained, with its ≥1000-map GPU policy also applied to the screen’s engine assembly; no solver/replay edits.

## 2. Development (seeds 11–13)

QEZVEY: **timeout (symmetry cost)**. Two merged-code login-A100 attempts timed out. Auditor verified ≥4000 heavy-atom automorphisms and single-thread brute-force symmetric RMSD in assembly/control-union/M2/coverage; the scaling correction is deferred beyond this package. No further dev retries. Verdicts stand on 119 panel parents; paired primary metrics use 116 valid parents after the two contract exclusions and failed UYOVUA.

| Arm | Route | Verdict |
|---|---|---|
| A | Correctness | PASS |
| AS | Coverage | PASS |
| ASP | Output contract | FAIL: floor CI upper +2.1457 kJ/mol exceeds +2; not rerun on fresh seeds |

## 3. Fresh check (seeds 31–33)

The only contract amendment, committed before seeing fresh results, assigned QEZVEY its own burst after the three shards, with runtime-only exclusion if it timed out. **Fresh QEZVEY completed all three seeds in 584 s, so it is included.** No chemistry or criteria were tuned on fresh seeds. C/A/AS ran paired in each parent invocation on the same shared login A100.

| Run | Parents | Parent-seed items | Wall (s) |
|---|---:|---:|
| Shard 0 | 40 | 120 | 326 |
| Shard 1 | 40 | 120 | 300 |
| Shard 2 | 39 | 117 | 373 |
| QEZVEY separate | 1 | 3 | 584 |
| CIFZEZ allocation retry | 1 (seed 33 only) | 1 | 63 |
| LOPCAW allocation retry | 1 (seeds 31,32 only) | 2 | 64 |

The three shards completed all scheduler items, but nine arm records (three parent-seed items) failed MACE allocation: CIFZEZ seed 33 and LOPCAW seeds 31–32. These items alone were retried once in an isolated lane, all three arms paired. Original packs remain preserved. A launch race briefly overlapped a CIFZEZ retry with the other agent; our lane was terminated by PID, and that interrupted attempt was discarded entirely. All accepted retry records succeeded. NEPNEF and UYOVUA failed GFN2 embedding in every arm; their invalid structures are not scored as valid results and were not retried. Primary N=117: exclude FEKKAJ/NEPNEF per contract, plus failed UYOVUA.

## 4. Contract analysis

All intervals below are 95% parent-clustered bootstrap CIs (10,000 resamples, fixed bootstrap seed 11); energy differences are arm minus paired control, kJ/mol. Hit rates are percentage-point differences.

| Stage / arm | Floor ΔE [CI] | Top-1 ΔE [CI] | Hit ≤1 Å pp [CI] | Hit ≤0.5 Å pp [CI] | Wall ratio |
|---|---|---|---|---|---|
| Dev / A | -0.006 [-0.018, +0.000] | +0.000 [+0.000, +0.000] | -0.287 [-0.862, +0.000] | +0.000 [+0.000, +0.000] | 1.041 |
| Dev / AS | -0.636 [-2.923, +1.310] | -0.723 [-3.273, +1.778] | +1.724 [-0.862, +4.885] | +0.000 [+0.000, +0.000] | 1.073 |
| Dev / ASP | +0.014 [-2.392, +2.146] | -0.723 [-3.273, +1.778] | -4.310 [-8.908, +0.287] | -4.598 [-7.759, -2.011] | 1.039 |
| Fresh / A | +0.011 [+0.000, +0.032] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.134 |
| Fresh / AS | -0.301 [-2.267, +1.612] | -1.082 [-2.972, +0.546] | +1.994 [-0.285, +4.843] | +0.000 [+0.000, +0.000] | 1.140 |

| Fresh arm | Distinct rows Δ [CI] | Pool ρ Δ [CI] | Matched pool ρ Δ [CI] |
|---|---|---|---|
| A | -0.014 [-0.043, +0.000] | +0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.002] |
| AS | +0.701 [-1.462, +3.174] | -0.011 [-0.040, +0.009] | -0.009 [-0.037, +0.011] |

| Fresh arm | Duplicate fraction Δ [CI] | Classes Δ [CI] | Returned rows Δ [CI] | Per-parent wall ratio [CI] |
|---|---|---|---|---|
| A | -0.013 [-0.030, -0.002] | +0.011 [+0.000, +0.031] | -0.014 [-0.043, +0.000] | +1.066 [+1.006, +1.169] |
| AS | -0.023 [-0.044, -0.007] | +0.288 [-0.006, +0.664] | +0.701 [-1.462, +3.174] | +1.104 [+1.015, +1.224] |

Duplicate fractions above use equally weighted parent differences; pooled row-weighted fractions below are descriptive. Solve wall ratios include cached work restored to standalone cost by the screen; MACE/post-processing are shared, so these are not independent end-to-end arm timings. The A100 is shared; no cross-platform wall comparison is used. Peak runner RSS was 18.7/18.7/19.1 GB across shards (seven lanes), 8.1 GB for QEZVEY (three lanes), and 2.6 GB for each isolated retry; per-arm memory CIs are unavailable from shared-lane telemetry.

| Fresh arm | Capped duplicate fraction C → arm | Uncapped C → arm | M3 stereo-bearing parents present C → arm | Absent set C → arm | Mean classes | Mean rows | XYZ storage bytes (C / arm) |
|---|---|---|---|---|---|---|---|
| A | 10.934% → 0.000% | 1.408% → 1.408% | 26 → 26 of 32 | 0 → 0 | 3.826 | 32.860 | 18396573 / 18389960 |
| AS | 10.934% → 0.000% | 1.408% → 0.000% | 26 → 29 of 32 | 0 → 3 | 4.103 | 33.575 | 18396573 / 18640916 |

| Fresh arm / guard panel | Trips | Mirrors | Control rotation null | Verdict |
|---|---:|---:|---:|---|
| A / with_artifacts | 0 | 0 | 1 | PASS |
| A / without_artifacts | 0 | 0 | 1 | PASS |
| AS / with_artifacts | 1 | 3 | 1 | PASS |
| AS / without_artifacts | 0 | 3 | 1 | PASS |

The binding guard excludes NUHPEL, MIDQEY, UMUBAF, YUCZUS, NUWFAM and CAFWIQ. Full-panel results are also shown. Fresh A exactness has 312 bit-identical uncapped parent-seed units and zero violations. Offline merged gate also passed all 13 capped parents.

| Fresh arm | Route | Verdict and binding evidence |
|---|---|---|
| A | Correctness | PASS: exactness passes, floor/top-1 CI uppers +0.0321 / 0 < +2; no uncapped duplicate increase |
| AS | Coverage | PASS: absent stereo 0→3, hit1 CI lower −0.285 pp > −3; floor/top-1 CI uppers +1.6123/+0.5463 < +2; solve wall 1.140 ≤ 1.5; guard passes |
| ASP | Output contract | FAIL development; not measured on fresh seeds |

### Named cases

M3 presence below is per fresh seed 31/32/33 (1=present). Values are C → AS. DAMVAQ cage junction is the declared out-of-Scope-B miss; FUTSOD is the declared cost-ceiling miss (2/32 classes). FETTAZ, XACMOE and LISVOC recover all three seeds. ZAZREZ is achiral, not a true absent stereo case. BOCRAN/CIFZEZ triple bonds have zero perceived Scope-B elements.

| Case | M3 presence C → AS | Mean returned rows C / A / AS | Capped duplicate rows C / A / AS |
|---|---|---|---|
| DAMVAQ | 000 → 000 | 16.0 / 16.0 / 40.0 | 0 / 0 / 0 |
| FETTAZ | 000 → 111 | 80.0 / 80.0 / 159.3 | 0 / 0 / 0 |
| FUTSOD | 000 → 000 | 16.0 / 16.0 / 32.0 | 0 / 0 / 0 |
| XACMOE | 000 → 111 | 17.0 / 17.0 / 26.7 | 0 / 0 / 0 |
| LISVOC | 000 → 111 | 7.3 / 7.3 / 17.0 | 0 / 0 / 0 |
| ZAZREZ | 000 → 000 | 4.3 / 4.3 / 4.3 | 0 / 0 / 0 |
| BOCRAN | 111 → 111 | 16.0 / 16.0 / 16.0 | 0 / 0 / 0 |
| CIFZEZ | 111 → 111 | 32.0 / 32.0 / 32.0 | 0 / 0 / 0 |
| CIXYUD | 111 → 111 | 32.0 / 32.0 / 16.0 | 0 / 0 / 0 |
| DUHHIX | 111 → 111 | 64.0 / 64.0 / 64.0 | 0 / 0 / 0 |
| EDOBUV | 111 → 111 | 32.0 / 32.0 / 32.0 | 0 / 0 / 0 |
| GOTWET | 111 → 111 | 64.0 / 64.0 / 64.0 | 0 / 0 / 0 |
| HEDJOP | 111 → 111 | 32.0 / 32.0 / 32.0 | 0 / 0 / 0 |
| NOYTUS | 111 → 111 | 25.3 / 23.7 / 23.7 | 12 / 0 / 0 |
| NUCKUS | 111 → 111 | 16.0 / 16.0 / 16.0 | 0 / 0 / 0 |
| NUHPEL | 111 → 111 | 64.0 / 64.0 / 64.0 | 3 / 0 / 0 |
| QEZVEY | 111 → 111 | 192.0 / 192.0 / 192.0 | 56 / 0 / 0 |
| SILROY | 111 → 111 | 16.0 / 16.0 / 16.0 | 17 / 0 / 0 |
| XAQCAT | 111 → 111 | 48.0 / 48.0 / 48.0 | 115 / 0 / 0 |
| YOVNEC | 111 → 111 | 32.0 / 32.0 / 32.0 | 10 / 0 / 0 |

The 13 capped parents are CIFZEZ, CIXYUD, DUHHIX, EDOBUV, GOTWET, HEDJOP, NOYTUS, NUCKUS, NUHPEL, QEZVEY, SILROY, XAQCAT, YOVNEC. Detailed per-case seed energies and Scope-B labels are retained in I_FRESH_SUMMARY.json and raw records.

## 5. Fold-in preparation

Candidate branch `jgen/i-foldin`, source-default commit `a8ba96e` (from experiment results commit `e936b29`). Proposed defaults: `dedup_factored=True`, `stereo_candidates=True` (scope B); `out_per_class=False` (ASP failed). No merge, no master movement, no push, no golden re-record.

| Item | Result |
|---|---|
| Candidate fingerprint | **5b56969afb9c** (experiment branch and master: 5f74b14b57ed). The `finish_phase=null` (old C1) identity is now `0af54f021eb6` (was `273f2ccbf892`) |
| Workstation golden vs current references (research/data/golden, 8-worker layout) | **8 of 490 arrays differ, in 2 solves per profile, identical in `production` and `u1`; the other 482 arrays (all rescores and 10 of 12 solves) are unchanged** |
| Pytest (`tools.check ws`, 3 shards, 1079 s) | 576 passed, 18 failed, 3 skipped. The 18 are listed below; none is an isomer test. The experiment branch passed 574/0/23 CPU-only |
| Fingerprint stage of `tools.check` | fails by design: the check hard-codes 5f74b14b57ed and must be updated by the auditor at fold-in |

Golden differences (both profiles):

| Complex | Reference | Candidate | Reading |
|---|---|---|---|
| SOKVET solve (X, score) | 128 rows | 64 rows | exactly half: symmetry-equivalent duplicates removed by the factored output dedup (intended: fewer duplicates). Inferred from the count and from the exactness gate (105 duplicate rows recovered on capped parents); the rows were not diffed individually |
| CBTYCO solve (X, score) | 30 rows | 34 rows | four added rows: stereo candidate classes added by scope B (intended: added stereo classes). Same caveat |

Both golden complexes pass through the output window; the rescore groups, all other solves (aqua_La3_8H2O, YAVFEG, KIQYEQ, OJIMIF, HUYNEU) and every shared-prefix array are unchanged within tolerance. The auditor re-records after owner approval; the two shape changes are the only expected deltas, so a re-record must show exactly these.

The 18 pytest failures on the candidate branch:

| Group | Count | Cause |
|---|---|---|
| `test_finish_phase::test_phase_cuda_capture` | 14 | the same 14 that fail on master on the A100 (section 1); this workstation run used the local CUDA device because the shard runner does not hide it |
| `test_d3_harness` (identity, effective free zone, placeholder, registered-arm rejection) | 4 | **new on the candidate branch**: the D3 harness (`tools/jgen/d3/arms.py`) hard-codes production fingerprint 5f74b14b57ed for its dry-run arm C, which the flipped defaults change to 5b56969afb9c. On the experiment branch and in the Perlmutter snapshot they failed for lack of git metadata, as stated above. At fold-in the D3 constant (and its draft contract) must move to the new fingerprint, an auditor decision; not edited here |

Test edits on the candidate branch (in the working tree, committed with this report): `test_profile_hook.py` (old-C1 fingerprint to `0af54f021eb6`), `test_typing_t1.py` (default fingerprint to `5b56969afb9c`), `test_isomer_space.py` (autouse fixture restoring `DEDUP_FACTORED`, so tests that toggle it do not leak). All three are consequences of the flipped defaults, nothing else changed.

User settings: `dedup_rmsd` controls the heavy-atom symmetry-aware duplicate threshold in angstrom (default 0.5); lowering it retains more similar conformers. `stereo_cost_ceiling` limits stereo-class work relative to the baseline solve (default 1.5), with deterministic uniform sampling when the class count exceeds the cap, so a crystal class may be missed. `out_per_class_k` sets the maximum retained rows per eligible class (default 8) when the optional `out_per_class` feature is enabled; it stays off because ASP failed.

Evidence: tracked I_FRESH_SUMMARY.json and unchanged I_CONTRACT.md criteria plus the authorized section 7 amendment; worktree research/data/jgen/campaign/I/{finish_exactness.json,finish_pmquick/,finish_cpu_tests/,dev_11_13_summary.json,fresh_31_33_summary.json,fresh_31_33/}; golden and pytest reports `research/data/golden_foldin*`, `research/data/pytest_foldin/` (worktree, gitignored); Perlmutter runs `$SCRATCH/jacobik/runs/isomer-*`. All workload ran through tools.run or the mandated check levels. No billed jobs, Colab or intentional local GPU use; billed spend zero.
