# G-3f frozen contract — 2026-11-02

Authority: directive_jgen_G3f_2026-11-02.md, after accepted G-3e audit
(audit_jgen_G3e_2026-11-02.md). Branch jgen/b-solver from 5dc37dc.
G-3e remains FAILED as registered. No production default, fingerprint or golden
changes. No new optimizer arms, settings, score arithmetic or tuned thresholds.
This contract and G3F_MANIFEST.json are committed before any G-3f result.

## Frozen inputs and identity

G3F_MANIFEST.json contains exact cal (120) and tail (10) parent lists, hashes
of newline-separated sorted codes, source manifest/DB/meta/model hashes,
seeds 11–13, both arm flags/fingerprints, and SHA256 of every G-3e C1 baseline
record. DB constraints_2026-11-01.json plus meta; no DB override.
C1 production 273f2ccbf892; IA-k 5f74b14b57ed, switch 100, symmetry-only
release, shape/projection retained, thermal widths, budget 250. Full-pool xz.
No val/reserve/locked panel. Tail is explicitly development evidence with
known val overlap; run only the directive's local replicate, never L4 tail.

Primary excludes FEKKAJ and NEPNEF, retains NUWFAM and XOJHIN. Report all-included
finite subsets and sensitivity without NUWFAM, with declared denominators.
Known-invalid NEPNEF embed failures are exempt, ledger embed_fallback=true;
never count a known cached fallback as valid. Any other embed fallback or any
replay fallback stops and is reported. Missing/failed units remain failures;
loops continue other complexes, no substitutions or rescue reruns.

## 3.1: one fresh C1 replicate, local RTX 2060 SUPER

One fresh C1 invocation on cal, then one on tail, all seeds together, same
bench settings as G-3e C1; no IA-k rerun. Independent processes prevent loss
of embed-failure provenance across panel transitions. Compare replicate minus
frozen G-3e C1 per parent/seed. Means and 95% percentile parent bootstrap CIs:
10,000 resamples, RNG seed 11, average all three paired seed units per parent
before resampling. Report top-1, floor, obscene pooled-row fraction (>200
kJ/mol), per-parent means and seed deltas, largest absolute parent mean and
largest absolute seed delta. Incomplete parent means do not enter bootstrap;
finite subset and complete/declared denominators explicit; no full-panel pass
inferred from incomplete data.

N0 is the number of primary cal parents with replicate-minus-baseline top-1
>25 kJ/mol on at least two of three seeds. Count all primary cal parents, as
the directive specifies the G-3e guard; report their names and seed deltas.
Same count/names for tail, descriptive only. Commit measured N0 and resulting
L4 threshold before L4 results or launch. The threshold rule is fixed now:
ceil(N0 * n_L4_primary_declared / 118) + 2, where the planned L4 primary count
is 118; thus full cal uses N0+2. Failures never shrink the declared count or
relax this threshold. Any incomplete cohort is disclosed and cannot establish
a complete confirmation. This calibration does not revise the G-3e verdict.

## 3.2: bounded cost instrument

At most two hours from profiling start, then end this stage regardless of
optimization outcome. Profile C1 and IA-k on CAXTAX and AWONOR, seeds 11–13
batched, local 2060, no concurrent local GPU jobs. Record B, steps, CUDA-event
phase timings for score/kinematics and gradient(s), optimizer update, aids,
and replay/eigendecomposition/host overhead. Diagnostic instrumentation is
separately labelled, preserves expression order, never becomes a new arm.
Disclose event/synchronization overhead and avoid treating profiling wall as
confirmation wall. Report before/after, or before plus unchanged/no-fix when
no admissible fix exists. Do not change numerics to save time.

A code fix is admitted only after CPU rescore identity with OMP_NUM_THREADS=1
on three parents x three seeds, replay parity, canonical replay regeneration,
unchanged default production fingerprint/goldens. It must leave IA-k outputs
bit-identical; its fingerprint must remain 5f74b14b57ed. No fix means retain
existing production source and report that evidence checks were not required.
No full golden rerun solely for research-only timing/report code.

## 3.3: paired L4 cal confirmation

Use own-worktree cj.py, VMs jgB1/jgB2, at most two VMs from one committed ref.
Cost-balanced split frozen before launch from existing parent timing costs;
all 120 cal parents exactly once across the two VMs, seeds 11–13. C1 and
IA-k in one bench invocation on each VM; same GPU L4 and export/flags.
First local CPU smoke of the VM driver, one cal parent/one seed.
Estimate wall and units including provisioning, fetching and conservative
margin; stop before launch if estimate >10 units. Hard aggregate task cap
12 units, log credits before and after; never stop another agent's VM.
Fetch outputs, verify expected records/exports (including failure ledger),
then down each owned VM immediately after completion or error. Stop owned
VMs before cap exhaustion; preserve/report partial results if necessary.

L4 metrics/bootstrap/cohorts same as above, candidate minus within-VM C1.
Four gates, all reported PASS/FAIL without tuning:
1. Primary cal mean top-1 energy delta CI upper <2 kJ/mol.
2. Cal mean pool-floor delta CI upper <0.
3. Primary parents tripping >25 kJ/mol on >=2 seeds <= calibrated threshold.
4. Aggregate end-to-end prepare+solve+assembly wall IA-k / C1 <=1.3,
   excluding MACE and exports. Measure parent preparation/prefetch/seed setup
   separately, allocate the full shared preparation cost identically to each
   arm (standalone equivalent), add each arm's measured solve+assembly.
   E/Z per-arm preparation is counted per arm, shared prefetch equally.
   Count batched three-seed parent wall once. Report solver-only ratio too,
   and actual entire job wall separately; do not substitute a step estimate.

Report control-obscene/not strata using three-seed control parent mean
threshold 200 kJ/mol; strict guard remains on all primary parents. Named
per-seed top-1/floor tables: AWONOR, FELSAU, CAXTAX, NUWFAM, XOJHIN, FEKKAJ,
HEDJOP. Report invalid/missing/nonfinite cases, timings, commits/fingerprints,
platform/driver/CUDA, source hashes and credits. No recommendation or fold-in.

## Checkpoint and resources

Local CPU <=5 processes, one GPU job under runkit gpu_lock('CodexB'),
nvidia-smi before launch. Long 20-minute completion-aware waits as requested
by owner; VM completion/fetch/shutdown handled promptly by controller.
At section 5: report committed G3F_REPORT.md, N0, four L4 gates, units spent,
all owned VMs down; STOP for audit. No production default/golden changes.
