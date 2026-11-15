# G-3f progress

## 2026-11-02 — contract frozen before measurements

Read G-3e audit and G-3f directive. G-3e failure stands. Frozen cohorts, arm
flags/fingerprints, baseline record hashes and N0-based guard rule committed
before any fresh result. Next: fresh local C1 cal/tail, bounded event profile,
then CPU VM-driver smoke and cost-estimated paired L4 cal. No fold-in.

## 2026-11-02 — portable driver CPU smoke

AJAQAD seed 11, both C1 and IA-k, CPU, passed in 70.820 s including evaluator
startup. Records have expected fingerprints, full-pool exports and measured
preparation/solve/assembly timing annotations. Three-phase shared preparation
is allocated fully and equally to both arms; E/Z per-arm preparation is tagged
separately. No production source changes. Initial credits 478.58, no active VMs.
Local C1 replicate continues. CUDA-event instrument anchors compile; actual
profile waits for the replicate to release the local GPU. No Colab started.

## 2026-11-02 — C1 replicate complete; N0 committed before L4

Fresh separate C1 invocations completed cal then tail, seeds 11–13. Primary
cal 118/118 complete parents, tail 9/9. Replicate-minus-G-3e top-1, floor and
obscene fraction are exactly zero for every primary seed unit: means/CIs zero.
N0=0, no cal guard trips; tail count=0, no trips. Largest absolute primary
parent mean/seed delta both zero. This measured run does not support the
assumed nonzero false-trip rate. G-3e verdict stands. Known invalid NEPNEF is
explicitly invalid in both comparison views. Frozen result G3F_NOISE_RESULT.json.

Full-cal L4 guard threshold now resolved by the precommitted rule: N0+2=2
parents (118 declared primary). No threshold adjustment after L4 results.
Cost-balanced partitions, using G-3e paired parent wall, are frozen in
G3F_L4_PLAN.json. Two L4s, safety factor two at 2060 speed, plus 40 min setup
and 10 min fetch/teardown per VM, at planning rate 2 units/hour: 7.454 units.
Estimated walls 6704.8/6712.4 s; below 10-unit launch threshold. Hard task cap
12 units, controller begins conservative abort at 10 units or 2.25 h, with
remote jobs limited to 3600 s. No VM started yet. Profile began after local
replicate ended; time budget two hours. CPU smoke already passed.

## 2026-11-02 — bounded profile complete; bit-identical fix admitted

Valid CUDA-event profile: AWONOR B=768; CAXTAX B=2688; all seeds together,
250 steps, 248 replay steps measured per arm. IA-k gradients cost ~27/34 ms
per step vs C1 ~14/17 ms: separate score-only backward for gain/row stopping
plus aided backward dominates. Changing that arithmetic is not admitted.
Small allowed fix: skip the unused Adam proposal when both phases are iRprop;
keep all moment/count state and used operations unchanged. Optimizer mean
AWONOR 0.315->0.227 ms, CAXTAX 0.786->0.560 ms. Whole-step improvement is small
and noisy, not a claimed removal of the full overhead. Before/after phase
summary G3F_PROFILE_RESULT.json, raw events retained.

First profile attempt rejected in full: copied globals froze the phase flag,
and replay drains were absent. Research instrument fixed to use live module
globals and asserted drains; valid measurements are profile_before_retry1
and profile_after. No invalid profile enters results or gate decisions.

CPU identity passed 9/9: AJAQAD/AWONOR/CAXTAX × seeds 11–13, 250 steps,
OMP=1, n_workers=1, no culling. Exact returned scores, coordinates, keys,
DOFs and CPU rescores; numeric NPZ evidence retained. Initial test launch
without deterministic Torch mode failed 14 CUDA exact-parity cases (21 other
tests passed). With the benchmark's actual deterministic mode enabled,
all 18 CUDA capture/parity cases pass. Both logs retained and disclosed.
Canonical replay regeneration unchanged. Goldens production/u1 each245/245,
read-only checks; production fingerprint273f2ccbf892 and IA-k5f74b14b57ed
unchanged. Verification evidence/hashes and bounded elapsed stage time are
in G3F_VERIFICATION.json. Profile/fix stage ends here, within two hours.

Next: paired L4 cal, both VMs from this commit, frozen N0=0 threshold2 and
7.454-unit estimate. CPU driver smoke passed before launch. No new flags or
production defaults changed; no tuning after G-3e guard failures.

Pre-launch controller correction: while the bundle was still packing, noticed
cj.push uses one local <bundle>.parts directory. Two concurrent VM uploads
would race there. Stopped only the owned local controller PID before any
up_start/new-VM call; no VM had been created and no units spent. Preserved
the partial archive under an interrupted filename. Controller now gives each
VM its own local bundle filename/chunk directory. Both VMs will use the same
new committed ref; solver/driver/verification code is unchanged by this fix.

## 2026-11-02 — L4 complete, §5 checkpoint

Both VMs used commit06566b5d8d26210897d10a476381fc65827ac593. Cost-balanced
split59/61 parents. jgB1 354/354 records and354 exports; jgB2 366/366 records,
360 exports and six expected NEPNEF failures marked embed_fallback=true.
Total720/720 records,714 valid full-pool xz exports. Both fetched/counted
before down. Both VMs confirmed down, active assignments0, usage rate0.
Credits478.58->476.36: 2.22 units versus7.454 estimate and hardcap12.
Paired job walls1719.515 and1603.659s; lifecycle2422.154s including packing,
provisioning, transfers and teardown. L4 driver580.82.07, CUDA13.0,
Torch2.14.1. No nonexempt embedding/replay fallback. No L4 tail.

Primary cal118/118 complete parents,354/354 paired units. IA-k minus C1:
- Mean top-1 -10.914[-18.719,-4.640] kJ/mol: primary PASS.
- Mean floor -9.510[-16.717,-3.949] kJ/mol: floor PASS.
- Regression guard FAIL: 3 parents (CAXTAX,FELSAU,NUWFAM), limitN0+2=2.
- End-to-end wall ratio1.217811<=1.3: wall PASS. C1 1228.391s,
  IA-k1495.948s; solver/assembly alone820.540/1123.110s.
Control-not-obscene107 parents top-1 -4.835[-8.309,-1.415]; obscene11 parents
-70.043[-131.816,-18.831]. No failed parent dropped to rescue any gate.
Named per-seed top-1/floor tables and all-included/NUWFAM sensitivities in
G3F_REPORT.md; full data and bootstrap details in G3F_L4_RESULT.json.

Validated all record counts and initial-state pairing, arm metadata,
within-VM machine identity, source bundle ref, DB/model/production hashes.
G3F_EXECUTION.json freezes counts/runtime/credit artifact hashes. Default
production273f2ccbf892 and IA-k5f74b14b57ed remain unchanged; no golden writes.
Read-only golden checks245/245 each and nine exact CPU identity units passed;
18 deterministic CUDA parity cases passed after correcting initial test
settings, with both initial failures and corrected evidence preserved.
Small bit-identical dead-Adam fix only; numeric score/update decisions not
changed. No recommendation or production fold-in. All owned jobs finished.
Commit report and STOP at directive§5 for audit.
