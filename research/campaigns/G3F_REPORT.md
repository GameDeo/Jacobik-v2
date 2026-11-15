# G-3f checkpoint — measurements

## Noise calibration: fresh C1 minus G-3e C1

| Panel | Complete/declared primary parents | Top-1 mean [95% CI], kJ/mol | Floor mean [95% CI] | Obscene fraction, pp | Guard trips |
|---|---:|---|---|---|---:|
| cal | 118/118 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |
| tail | 9/9 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |

N0=0; names: []. Full-cal L4 guard threshold=2.
Primary excludes FEKKAJ/NEPNEF; NUWFAM retained. All three seeds averaged
within each parent, then 10,000 percentile parent bootstrap resamples,
seed 11. C1 replicate deltas are exactly zero on all primary seed units;
largest absolute parent mean and seed delta are zero on both panels.
This measurement does not support the anticipated nonzero guard noise.
G-3e remains failed as registered. Full finite denominators and
NUWFAM-exclusion/all-included sensitivities are retained in noise JSON.

## Per-step CUDA-event profile and bounded fix

| Stage / parent / arm | B | Score/FK | Gradients | Aids | Optimizer | Other device/stream | Host/replay residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| before / AWONOR / C1 | 768 | 6.8949 | 13.8269 | 0.3123 | 0.0742 | 1.1608 | 0.2318 |
| before / AWONOR / IA-k | 768 | 6.8197 | 27.0372 | 0.4767 | 0.3153 | 1.1553 | 0.2317 |
| before / CAXTAX / C1 | 2688 | 10.9799 | 17.0958 | 0.2065 | 0.2302 | 1.5882 | 0.1876 |
| before / CAXTAX / IA-k | 2688 | 11.1880 | 34.4028 | 0.4400 | 0.7863 | 1.5528 | 0.2557 |
| after / AWONOR / C1 | 768 | 7.0603 | 14.1929 | 0.3110 | 0.0771 | 1.0814 | 0.2324 |
| after / AWONOR / IA-k | 768 | 7.0101 | 27.6172 | 0.4762 | 0.2271 | 1.0923 | 0.1734 |
| after / CAXTAX / C1 | 2688 | 11.2235 | 17.5661 | 0.2069 | 0.2930 | 1.6885 | 0.1804 |
| after / CAXTAX / IA-k | 2688 | 11.2027 | 34.5030 | 0.4395 | 0.5604 | 1.4591 | 0.2043 |

All timings are mean milliseconds per replay step, 248 samples per solve;
warm-up/capture steps excluded. Other includes eigendecomposition,
snapshot, projection, dual/history maintenance and applying/wrapping
steps. External CUDA events are inserted in a research-only compiled
method; the expressions and their order are preserved. A synchronize
reads timestamps each step, so residual host overhead includes the
observer cost. Confirmation uses uninstrumented solver code.

IA-k takes a separate score-only autograd pass for the remaining-gain stop and row freezing, then the aided gradient pass. Removing/changing it would change numeric decisions and is not admitted.

Skip unread Adam proposal only when neither phase uses Adam. Moment/count state preserved; all used arithmetic unchanged.

CPU identity: 9/9 units (AJAQAD/AWONOR/CAXTAX × seeds 11–13), one thread, no culling, 250 steps.
Returned coordinates, row keys, scores, native final DOFs and CPU
rescores compare exactly, with before/after NPZ arrays retained.
Replay/phase tests, canonical regeneration and default golden results
are recorded in the verification manifest. IA-k fingerprint remains
5f74b14b57ed, production 273f2ccbf892; goldens never written.

Initial verification without Torch deterministic mode: 21 passed, 14
CUDA exact-parity failures. This configuration did not match the bench.
With torch.use_deterministic_algorithms(True) and the same cuBLAS
workspace setting as the bench, all 18 CUDA parity cases passed.
Both logs are retained; no solver tolerance or numeric rule changed.
Production/u1 golden checks each passed 245/245. The entire bounded
profiling/fix/verification stage took less than 31 minutes.

The first profiling attempt is retained but rejected: it copied stale
module globals and measured only eager warm-up rather than replay.
Live globals and asserted replay-event drains fixed the instrument.
Those invalid measurements are not in the profile table or any gate.

## Paired L4 cal confirmation

118/118 primary parents, 354/354 paired seed units. Unrelaxed MACE
polar-1-s float64 energies; candidate minus within-VM C1. Energy
units kJ/mol; obscene-fraction change in percentage points.

| Metric | Mean IA-k − C1 [95% parent CI] |
|---|---|
| top1_dE | -10.914 [-18.719, -4.640] |
| floor_dE | -9.510 [-16.717, -3.949] |
| _obscene | -1.449 [-3.881, 0.653] |

| Gate | Result | Evidence |
|---|---|---|
| primary | PASS | Top-1 CI upper <2 kJ/mol |
| floor | PASS | Floor CI upper <0 |
| regression | FAIL | 3 parents; threshold 2 |
| wall | PASS | End-to-end ratio 1.218; maximum 1.3 |

Primary complete: True; timing complete: True.
Regression names: ['CAXTAX', 'FELSAU', 'NUWFAM']

End-to-end wall counts prefetch, prepare and seed setup plus solver and
assembly once per three-seed parent, excluding MACE, diagnostics and
exports. The full shared preparation cost is allocated identically to
both arms, representing each arm standalone. E/Z preparation is tagged
per arm. This is distinct from the measured shared paired-job wall.

| Arm | End-to-end seconds | Solver/assembly seconds | Timed parents |
|---|---:|---:|---:|
| C1 | 1228.391 | 820.540 | 120/120 |
| IA-k | 1495.948 | 1123.110 | 120/120 |

| Control stratum | Parents | Top-1 mean [95% CI] |
|---|---:|---|
| control_not_obscene | 107 | -4.835 [-8.309, -1.415] |
| control_obscene | 11 | -70.043 [-131.816, -18.831] |

Control strata use the three-seed parent mean at 200 kJ/mol. The
regression guard still covers all declared primary parents.

| Sensitivity | Top-1 mean [95% CI] | Floor mean [95% CI] |
|---|---|---|
| all_included | -10.794 [-18.645, -4.513] | -9.468 [-16.596, -4.058] |
| excluding_NUWFAM | -11.094 [-18.873, -4.786] | -9.327 [-16.444, -3.821] |

### AWONOR

| Seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| 11 | 32.540 / 32.540 | 52.918 / 52.918 |
| 12 | 75.239 / 72.848 | 32.921 / 32.921 |
| 13 | 32.174 / 32.174 | 56.665 / 56.665 |

### FELSAU

| Seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| 11 | 1.262 / 1.262 | 39.214 / -1.966 |
| 12 | 8.825 / -3.508 | 35.721 / 1.098 |
| 13 | 0.010 / 0.010 | 38.364 / 1.589 |

### CAXTAX

| Seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| 11 | 78.000 / 49.790 | 69.083 / 57.530 |
| 12 | 51.971 / 50.726 | 84.909 / 59.694 |
| 13 | 54.272 / 46.973 | 98.229 / 57.674 |

### NUWFAM

| Seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| 11 | 130.194 / 130.194 | 198.794 / 113.950 |
| 12 | 154.855 / 154.855 | 189.865 / 123.921 |
| 13 | 236.144 / 164.547 | 163.034 / 118.906 |

### XOJHIN

| Seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| 11 | 187.195 / 187.195 | 165.293 / 165.293 |
| 12 | 197.949 / 184.636 | 160.237 / 160.237 |
| 13 | 198.343 / 185.355 | 177.558 / 168.379 |

### FEKKAJ

| Seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| 11 | 266.474 / 172.211 | 223.280 / 172.310 |
| 12 | 179.556 / 171.980 | 232.898 / 153.722 |
| 13 | 210.920 / 166.433 | 210.824 / 171.082 |

### HEDJOP

| Seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| 11 | 317.076 / 257.161 | 345.487 / 304.041 |
| 12 | 318.760 / 286.115 | 314.463 / 268.833 |
| 13 | 385.165 / 228.509 | 301.354 / 258.318 |

## Failures, provenance and spend

L4 failure ledger: 6 rows (full entries and record hashes in G3F_L4_RESULT.json).
- {"vm": "jgB2", "arm": "C1", "code": "NEPNEF", "seed": 11, "status": "failed", "reason": "RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)", "embed_fallback": true}
- {"vm": "jgB2", "arm": "C1", "code": "NEPNEF", "seed": 12, "status": "failed", "reason": "RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)", "embed_fallback": true}
- {"vm": "jgB2", "arm": "C1", "code": "NEPNEF", "seed": 13, "status": "failed", "reason": "RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)", "embed_fallback": true}
- {"vm": "jgB2", "arm": "IA-k", "code": "NEPNEF", "seed": 11, "status": "failed", "reason": "RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)", "embed_fallback": true}
- {"vm": "jgB2", "arm": "IA-k", "code": "NEPNEF", "seed": 12, "status": "failed", "reason": "RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)", "embed_fallback": true}
- {"vm": "jgB2", "arm": "IA-k", "code": "NEPNEF", "seed": 13, "status": "failed", "reason": "RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)", "embed_fallback": true}

VM paired-job walls, seconds: {'jgB1': 1719.514773027, 'jgB2': 1603.6586593799998}
Estimate before launch: 7.454 units, below the 10-unit launch threshold.
Credits: 478.58 before, 476.36 after; 2.22 units spent (hard cap 12).
Owned VMs down: ['jgB1', 'jgB2']; final usage rate zero, active assignments zero.
L4 driver 580.82.07, reported CUDA 13.0, Torch 2.14.1. Local 2060
driver 595.99.02, reported CUDA 13.2. Complete runtime records retained.

G3F_CONTRACT.md committed 9389a57; measured N0 and L4 partition/cost
plan committed 2446e4d before launch. Both L4 VMs use the single ref
recorded in g3f/colab_ref.txt and COLAB_BUNDLE.json. Flags/manifests
and input/model hashes are frozen in G3F_MANIFEST.json; record hashes
and within-VM pairing retained. Driver/CUDA/hardware in nvidia-smi.txt
per VM; credits before/after and fetch/count/down journal retained.
Local CPU smoke passed on AJAQAD seed 11 before any VM launch.
No L4 tail, no reserve/locked evaluation, no numeric optimizer/score
changes. No production defaults or goldens changed. All four gates
are reported without rescue tuning. No production recommendation.
Section 5 checkpoint: commit, confirm VMs down and credits logged, stop.

### Local measured job walls

| Job | Seconds |
|---|---:|
| C1 replicate cal | 1817.001 |
| C1 replicate tail | 242.644 |
| Valid event profile before | 92.405 |
| Valid event profile after | 90.268 |
| Nine-unit CPU identity | 861.700 |

Profile/CPU verification jobs overlapped; these durations are not
summed as elapsed stage time. G3F_VERIFICATION.json records a conservative
1831.43 s upper bound for the entire bounded stage. G3F_EXECUTION.json
verifies the 720 records, 714 exports, paired initial-state digests,
single VM source ref and input/model hashes; full runtime/credit hashes
are retained there.
