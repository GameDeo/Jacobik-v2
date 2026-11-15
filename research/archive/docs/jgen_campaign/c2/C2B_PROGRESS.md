# C-2b progress

## 2026-11-03 — development opened
Read C-2b, C-2 + amendment, audit C-2/T1c §1, PLAN status/§7a-C,
and D plan §3. Continue clean branch jgen/c-coverage from d1c6cb2.
Development uses only cal/tail minus locked exclusions, seeds 11–13.
Confirmation seeds 21–23 are prohibited until the committed contract freeze is audited.
No relaxation. Primary exclusions FEKKAJ/NEPNEF; guard artefacts fixed before
new data: NUHPEL, MIDQEY, UMUBAF, YUCZUS, NUWFAM, CAFWIQ.
At most four finalists, declared routes A–D; wall ≤1.3x and guard required.
10,000 parent-cluster bootstrap resamples, seed 11; per-parent seed means;
undefined values not imputed, failures retained in coverage denominators.
Local GPU available (nvidia-smi checked), xtb resolves. CPU cap five.
No Colab in development; spend 0 units. No golden writes authorized.

Initial code inspection: FK rotates about the original donor-atom centroid
before internal torsions. Per-row Kabsch computes translation about the seeded
donor-group centroid. These pivots can differ. Diagnose this separately from
the documented post-fit full-pi orientation draw before changing code.

Development instrumentation attempt 1 failed on an assumed Ligand.smiles
attribute; stopped own PID and retained placement_attempt1. Corrected the
trace only, no algorithm adjustment. Placement retry uses the same cases.
Synthetic pivot regression passes; focused coverage tests 10/10.

Before GPU pilot: fixed stratified cases QEZVEY (large pools/symmetry and
previous S1 resource retries), HOXJUZ (window loss), JEGPUJ (K responder),
XACMOE (swings), BABNAV (aromatic). Arms C/K/P/F/FU/S1, seeds 11–13.
This is a pilot, not a panel-scale run. Pilot wall will supply the estimate
before any broader run. Estimated pilot 10–20 minutes from C-2 per-parent
wall and heavy-case representation; no full-panel timing claim yet.

2026-11-03 interim diagnostics: S1 size-matched rho (20 draws, seed 11),
115 defined parent pairs: -0.0691 [-0.1006,-0.0397]. Existing T exports:
72 pools gain rows, no added row beats C's energy floor; added rows are
within 1 A in 26 pools. Compressed XYZ + manifests storage 16,242,753 ->
17,131,100 bytes (1.0547x). These are seen-data reanalyses, no gates.

NEPNEF actual xTB trace: first ligand + one Li cap is charge -6, UHF 0;
GFN2 returns code 1, no xtbopt.xyz, geometry optimisation fails after SCC
nonconvergence. Second ligand succeeds. Generator itself returns six
conformers in a two-step CPU degrade smoke; the bench intentionally rejects
the fallback as invalid. Thus the batch failure is not an escaping generator
exception. No charge/loader change here; retain this issue for D0 embed.

Canonical replay numerical body is unchanged: changed paths are initialisation
only. Its generated source checksum is refreshed for the worker-layout doc fix. Twelve focused coverage/route/statistics tests pass.

Before panel-wide placement census: same five stratified parents (including
QEZVEY, the large/symmetry-heavy S1 resource case) completed the CPU placement
pilot. Per-parent wall 67.33, 67.69, 70.49, 18.28, 24.06 s at three workers,
including cold worker imports. Conservative direct mean extrapolation:
49.57 * 120 / 3 = 1,983 s (33.0 min); allow 45 min for E/Z variants and
resource contention. This census builds starts only, never solves or evaluates
MACE. Reuse the five completed pilot records and measure the other 115 parents
at three CPU workers, all native E/Z variants and seeds 11–13. Paired C/K/P/F
fits share torsion seeds and RNG draws, with actual FK used for distances.
GPU pilot remains running; no panel-scale GPU development run is planned.

NEPNEF's source SMILES itself has Zn+14 and N([O-])[O-] groups; the builder
preserves two charge -7 ligands. The bad high-charge free-ligand input is
upstream of search. Carry source-adapter/typing evidence to D, without changing
charges in the generator or excluding the failure from its registered tables.

Follow-up named-case run declared before KU results: C/K/P/KU, seeds 11–13,
JEGPUJ/SIVBUV/FELSAU/CAXTAX/UYOVUA/MIVXAR/LOPCAW, after the current GPU pilot
releases the lock. KU combines existing K+U with no extra settings changes;
it separates that combination from F/FU's reduced orientation exploration.
Expected 3–6 minutes from completed small-parent pilot timings. No fresh seeds.

Golden layout probe: configured maximum workers=8 passes production 245/245.
OJIMIF retains exact X/scores when those partitions execute serially, so the
partition shape, rather than concurrency scheduling, causes its one-worker
mismatch. Checker pins configured workers, thread mode, Torch/BLAS/OpenMP=1.
`--workers` exposes the recorded partition setting (default 8) for alternate
reference layouts; inherited JACOBIK_NUM_WORKERS no longer silently changes it.
No golden record operation or reference write was run.

Validation: focused C-2b/profile tests 24 passed, with the DB-dependent Kabsch
integration test skipped because a worktree has no data/db. Reran that one
with JACOBIK_TEST_DB pointing to the shared production DB: 1 passed. Full
read-only golden check: production 245/245 and u1 245/245, zero differences.
No full-suite or full replay panel repeated at this development checkpoint;
those remain a pre-fold-in package gate. Actual CUDA generation is exercised
by paired pilot arms and canonical replay numerical body is unchanged (source-checksum metadata refreshed).

Owner timing caution received during development: device/host heavily
contested (100% CPU utilisation from concurrent refit/re-extraction). All
speed and wall estimates in this log are observations under contention,
not intrinsic speed measurements. Paired invocation is preserved, but does
not eliminate changing host load between serial arms. Do not infer speedups
from development ratios or use them as clean timing evidence. Keep the
registered confirmation wall rule; record its load context and report any
timing limitation, without changing a numerical gate after results.

Panel-wide placement complete: 120 parent files, 133 native configurations,
195,456 initial rows; NEPNEF is the only failed configuration. Pooled median
row-max offset C 3.1526, K 2.8071, P 0.3057, F 0.2486 A. Among 119 successful
parents, median >0.3 A on C/K 101, P 45, F 39. At least one row maximum
changes by >0.3 A under K->P in 102 parents. These are geometry measurements,
not speed or MACE improvements. Full native-variant scope matches C-2 counts.

Confirmation runner hardening before freeze: default invocation permits only
seen seeds. The explicit post-audit path requires exact ordered seeds 21–23,
all frozen arms/settings and full population, unchanged frame/exclusion hashes,
production DB + sidecar, MACE model and source hashes. An isolated unit test
proves an unfrozen contract is rejected before any arms or results are written.
This validation test does not generate/evaluate fresh-seed structures.

2026-11-03 development resource abort: stopped own pilot PID 79709 by SIGTERM
at elapsed process time 2,488.13 s (41.47 min). Two QEZVEY graph-capture OOM
warnings had fallen back to eager execution. Under the owner's explicit
contention caution, further recovery time was not a useful speed measurement.
Preserved 54 completed records. Marked 18 unexported QEZVEY and 18 unreached
XACMOE records as controller stubs, null energies, explicit failure reasons;
no physical/numerical outcome is invented. timing.json was not flushed, so
no finalise/ratio reconstruction is attempted for this incomplete invocation.
Controller metadata and abort_failure_ledger.json retain the event. No QEZVEY
rerun. S1 has complete seen panel evidence in C-2; no fresh confirmation run
has occurred. The declared C/K/P/KU seven-case follow-up began only after the
old process exited and nvidia-smi confirmed release of the GPU.

Owner monitoring preference: for a long unattended run expected to finish
without intervention, set a long periodic check (e.g. 20 minutes) or wait for
completion events, rather than polling every couple of minutes. Short checks
are reserved for errors/checkpoints/dependent immediate work. Phase 2 must
follow that preference; no Phase 2 run is launched at this checkpoint.

## 2026-11-03 — CONTRACT FREEZE, STOP FOR AUDIT

Declared finalists: K / route B / 38dcb8218b73; P (corrected per-row fit) /
route D / e2a5cdba943f; KU / route B / 2285ca6881d1; S1 W48 / route C /
5f74b14b57ed (budget identity separate). Four finalists, eight configurations
assessed. F/FU not advanced; U no standalone slot; T remains experimental
output option, default off. Exact input/model/source hashes and all settings
are in C2B_FINALISTS.json; routes/guard/statistics in C2B_CONTRACT.md.

Completed follow-up: 84/84 records valid; per-parent row-steps equal across
C/K/P/KU; no pivot/replay/solver-OOM fallback. KU returns 11/4/6 SIVBUV rows
versus 2/2/2, with seed-13 floor 44.8 versus 66.5, and JEGPUJ top-1 56/45/51.
P alone leaves SIVBUV collapse and loses some K advantage on JEGPUJ; CAXTAX
P floor losses retained. Compact development JSON includes every declared
pilot/follow-up record, all 36 abort stubs, full placement quantiles, inherited
metrics and unchanged reference/input hashes. No aggregate from the aborted
pilot is used as a gate. Timings remain contextual under heavy contention.

Final focused checks: 27 passed (explicit shared production DB), zero skipped.
Both golden profiles: 245/245 arrays match; references unchanged. Canonical
replay numerical body unchanged, generated source checksum refreshed for the
CPU-layout doc correction. Default X/scores match b0134a2 on three parents;
production fingerprint remains 5f74b14b57ed. DB/sidecar and golden hashes all
match. Full suite/replay panel remains a pre-fold-in package check, not repeated.

All owned generation/enrichment jobs finished or explicitly stopped and logged.
No VM started/owned; Colab spend 0. No production fold-in, merge or push. No
seeds 21–23 generated/evaluated. Stop here; do not start the confirmation until
the contract-freeze audit is cleared. Use long completion-driven/scheduled
checks for a later unattended confirmation, per the owner preference.

## 2026-11-03 — confirmation authorized and launched once

Owner cleared the audit addendum after freeze 5a19ef8. Exact C/K/P/KU/S1,
seeds 21–23, full frozen population, one invocation; no reruns. Frozen source
and DB/sidecar hashes matched before launch; confirmation directory absent.
Start context: no GPU compute process, load average 2.20/2.76/9.63, roughly
11 GiB available RAM. One single-core Architector stock-sentinel process was
visible (PID111576); no full Architector or T1d run visible. Record it rather
than claiming an entirely idle host. Automatic load/RAM/nvidia-smi snapshots
at start, every1200 s and completion. New heavy load will be logged, not used
to stop/restart. Monitoring wrapper does not alter any frozen source and has
no resume or retry path. Audit interpretation rule accepted: K/KU qualification
without P supports an intentional perturbation, not shipping the pivot bug;
no adoption implementation or fold-in in this confirmation task.

Startup verification: driver accepted the frozen preflight and logged exactly
120 parents / five arms / seeds21–23. Launcher PID112838, single driver
PID112848, start UTC2026-11-04T01:58:09 (local2026-11-03). Sampled start load
1.79/2.51/9.04, available RAM11.57 GiB. Frozen sources unchanged. Monitor source
and authorization audit copied to confirmation/provenance; no generator edit.
Descriptive reporting implementation prepared without opening energy outcomes;
synthetic pairing/undefined/failure-denominator test passed (1 test). P-vs-K
and KU-vs-K will use the same parent clustering but no adoption gate or route
switch. No standalone L4 replicate or adoption is part of this once-only task.

20-minute load event (UTC2026-11-04T02:18:10): T1d re-extraction started after
the quiet launch. Snapshot shows PIDs116294/116295/116296 running
`fit.typing_reextract run --mode t1d_promote --workers3`, sampled aggregate
CPU99.8%, load17.81/18.06/15.07, available RAM6.89 GiB. Per audit condition,
recorded the interference and continued the same invocation unchanged; no
stop/restart/rerun. At the scheduled check,54 records per arm had been written.
No energy outcomes inspected or algorithm/settings changed. Wall gate remains
mechanical under the frozen rule, with this explicit timing-validity caveat.

Owner then SIGSTOPped the T1d processes and requested notification after C-2b
finishes so they can SIGCONT them. Captured an immediate owner_paused_t1d load
snapshot and process-state event. Same confirmation continues, no settings or
invocation changes. Do not resume/kill those processes; notify owner only after
both generation and postprocessing/reporting finish.

Postprocessing orchestration attempt1 failed before executing any analysis:
this Python build has no os.pidfd_open. Retained the log and failed launcher
marker. No generation interruption or rerun. Auxiliary helper now waits for
the completion marker at1200-second intervals and will execute the same frozen
finalise/enrich/four-route-analysis commands once. Frozen code remains unchanged.

## 2026-11-03 — FINAL CONFIRMATION CHECKPOINT, STOP

One invocation completed successfully (driver PID112848, return0), all1,800
records, seeds21–23. Each arm357 valid /3 NEPNEF failures; no other failure
parent. Process elapsed199.555 min including MACE/export/startup. Four S1 graph
fallback warnings retained (FEKKAJ/QEZVEY); no pivot fallback. Full raw load
journal at start/every~20 min/end, plus owner-pause event. T1d appeared at the
20-minute sample (CPU99.8%) then owner SIGSTOPped parent/workers at~37 min;
later snapshots show the lower load and additional Architector workers.
No stop/restart/load correction/rerun of confirmation. Timings remain contextual.

Routes: K(B) PASS, P(D) PASS, KU(B) PASS, S1(C) FAIL (floor NI only).
Floor/top1 differences [95% parent CI], kJ/mol:
K -0.997[-3.239,0.905] / -2.170[-5.131,0.349];
P -1.876[-5.112,0.425] / -3.220[-7.421,0.159];
KU -2.062[-6.242,1.081] / -3.299[-8.835,1.035];
S1 +0.959[-0.963,2.578] / -0.691[-3.432,1.663].
Observed wall ratios K0.9475/P0.9513/KU1.0517/S1 1.0008; no intrinsic speed
claim. Guard trips/mirrors/null with ->without artefacts:
K3/7/1 ->3/6/1; P0/5/2 ->0/4/2; KU3/7/1 ->2/6/1;
S1 4/3/3 ->3/2/3. Every guard passes, C null1.

Requested paired descriptive tables added (not routes): P-K top1
-1.050[-5.069,2.273], KU-K -1.130[-6.498,2.817]; KU-K distinct
+0.966[0.455,1.571]. No mean energy superiority over K established.
S1 matched rho improves -0.0695[-0.1084,-0.0301], but does not rescue failed
floor non-inferiority. P qualifies as correct start_kabsch; all search adoption
still waits for D3 joint evaluation. No untested combination, fold-in or push.

Postprocessing: exact frozen finalise/enrich/four-route CLI analyses completed;
descriptive reporter tested on synthetic pairing/undefined/failure cases (1pass).
Auxiliary pidfd wait failed before postprocessing; preserved and replaced by
portable20-minute wait, not a generation rerun. 119 S1 compute checks pass,
NEPNEF unavailable, zero mismatches; ordinary-arm budgets match. No enrichment
failures. All frozen source and original DB/sidecar/golden hashes unchanged.
Complete result C2B_CONFIRMATION.json, tables and named cases C2B_REPORT.md.

Generation ended UTC05:17:41, postprocessing06:13:50 (local2026-11-03).
All owned C-2b compute finished, no VM/Colab spend (0 units). T1d remains under
owner control: no SIGCONT or kill was sent. Notify owner they can SIGCONT now.
Commit final report and stop for audit; no reruns or additional experiments.
