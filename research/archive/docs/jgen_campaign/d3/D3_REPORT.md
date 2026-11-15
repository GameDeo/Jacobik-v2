# D3-H §4 checkpoint — 2026-11-04

**Harness built; checkpoint has a FAILED L4 smoke. Stop for audit. No D3 arm or D3 result.**
Branch `jgen/d3-harness`, base `74b4d47`. All production files and C-2 code unchanged; production fingerprint
`5f74b14b57ed`. No local GPU, locked partition read, reserve opening, production DB write, merge or push.
The draft is not ready to freeze until the smoke failure and methodology notes below are reviewed.

## Files and commands

- `tools/jgen/d3/arms.py`: C / D / S=P / DS / Cprime definitions and identities; JSON in
  `research/bench/jgen_campaign/D3_ARMS.json`. Exact P overrides `start_kabsch=true`, `start_kabsch_pivot=true`,
  fingerprint `e2a5cdba943f`. Control DB/meta/model identities anchored to C-2b finalist hashes.
- `d3/run.py`, `panels.py`: stage1, stage2, stage3, registered ablations, estimates, Colab lifecycle/fetch verification.
- `d3/worker.py`: one invocation per VM, parent first, all arms consecutively, isolated subprocesses for import-time
  flags and DB state; unrelaxed MACE-POLAR-1-S float64, full-pool xz, symmetric heavy coverage, shell/eta2 metrics,
  row-step ledgers, generation and process walls, load and nvidia-smi logs.
- `d3/analyse.py`, `comparison.py`: fixed gates, parent-seed means, 10,000 parent bootstrap resamples/seed 11,
  C-2b metrics/guard/artifact views/matched rho, catastrophes, DS-vs-S, interaction, candidate tie-break and named cases.
- `d3/curvature.py`: CPU wrapper using cached u1 points/directions/energy curvature and current DB/flags.
- `tests/test_d3_harness.py`: synthetic gates and refusal/lifecycle tests.
- `D3_CONTRACT.md`: copied draft, **DRAFT, not frozen**, known values filled and D2 brackets retained.
- `D3_COMMANDS.md`: complete environment, commands per stage, analysis, ablations, metadata schema and handoff.

The VM bench itself assumes one DB and builds score-dependent state once. The harness therefore isolates each arm
in a fresh subprocess, within a single parent-paired VM worker invocation. This costs startup/preparation time but
prevents an arm reusing another DB's scorer, typing or prepared shell. Preparation and solve assembly use the C-2b
wall allocation; full process wall is separate. Panel sizes: stage1 120 (primary 118), stage2 110, stage3 80 supplied
by the auditor only. Stage2 reads the existing A/B/C+D draws (seed 31), never redraws them.

Real launches currently refuse the draft/unresolved D2 config. Frozen real stage1 automatically computes D/DS
curvature before VM launch. Stage3 checks supplied code-file hash/path, OWNER_GO, qualified non-dryrun stage1,
unchanged identities, and foldin_check; a fresh execution directory prevents reopening. No gate threshold CLI switches.
Ablations require a failed real stage1 and complete reproducible group recipes, in the registered order.

## Verification

38 synthetic tests pass (39 including C-2 reporting compatibility). They cover strict energy bounds, inclusive curvature/wall bounds, holdout count/energy
edges, catastrophic >200 threshold, artifact-aware imported guard, stage3's measured C-vs-Cprime null, tie-breaks,
interaction CI, failed coverage denominators, complete synthetic 2x2 analysis, DRYRUN nonqualification, DB/meta and
identity refusals, bad code hash/missing OWNER_GO/unqualified stage1/forbidden locked path, transferred counts/hashes,
fetch-before-down on errors and refusal to stop an existing VM. Python compilation and diff whitespace checks pass.
Focused C-2 reporting compatibility is recorded in the verification log. No full production suite/golden rerun is
needed for a bench-only package with no production edits.

## Exact C-2b reproduction

Read-only links: shared `D3/dryrun/reproduction_s/{C,S}` point to confirmation `{C,P}`; no confirmation file modified.
Executed `python -m tools.jgen.d3.analyse stage1 --root <.../dryrun/reproduction_s> --reproduce-c2b`.
Every metric/CI, guard, wall ratio/CI, failure list and per-parent value compares exactly equal to
`C2b/confirmation/analysis_P.json`, not just rounded values:

- Floor −1.8759358137902558 [−5.112065781287915, 0.42485091083256843].
- Top-1 −3.2199856931742246 [−7.421390056973558, 0.15863978985623245].
- Guard without artifacts: trips 0 / mirrors 4 / null 2; both artifact views preserved.
- C-2b route D: passes. D3 energy: **fails**, since neither improvement CI excludes zero; top-1 noninferiority,
  guard and wall individually pass. Curvature is absent for this historical reproduction and cannot pass by default.
- Candidate is null; reproduction can never authorize a locked opening.

Evidence: `D3/dryrun/reproduction_s/{analysis,equality_check}.json`.

## Curvature baseline

Current production DB + meta, default flags, CPU only, 80/80 cases and 15,055 saved directions; zero failures:

| View | Median energy/kT : score ratio | Median within-point Spearman |
|---|---:|---:|
| Combined (gate) | 1.3315397846869683 | 0.8639999488508491 |
| Crystal | 1.501382083939354 | 0.8512096774193549 |
| Pool | 1.2959918130007786 | 0.8699985466551534 |

Combined gate passes. The crystal ratio alone is above 1.40 and is reported, not hidden. The draft explicitly fixes
the scalar reduction for auditor review. Inherited `forces.py` term scope is Q1/Q2/Q3+Q4/Q5/Q6; **it omits Q7**.
That inherited scope must be reviewed before D2 freezes a package containing Q7 changes. There is no CLI scope switch.
An initial pass preceded explicit GFN2-fallback instrumentation; a second complete pass with xtb/EmbeddingValidity
checks reproduced every reported number exactly and reported zero invalid embeddings. No energy recalculation or
GPU use. Original cache files are read-only; outputs are in `dryrun/curvature` and `dryrun/curvature_guarded`.

## Authorized L4 smoke — FAILED

Only one VM was created (`jgD3s`, L4), from committed harness `f89deaf1787572a57e35b4d4ff9b7e82f3eb1cce`.
Before that, a CLI preflight typo (`list` instead of `sessions`) failed without creating a VM; the failure directory
is retained. The actual attempt uploaded the bundle, completed setup, verified xtb/GPU/production fingerprint,
and uploaded exact production DB/meta. It then stalled in Colab exec during the plan upload, before worker launch.

- Requested: 3 parents (CAXTAX, FELSAU, JEGPUJ), seed 11, C + DS with production DB substitute; DRYRUN only.
- **0 / 6 expected records; no worker output file manifest. No GPU generation, MACE record or pool export.**
- At the 12-minute total run budget the attempt was interrupted by PID. The existing finally path attempted
  `cj.py get`, counted zero locally against six expected, recorded fetch failure, then called `cj.py down`.
- VM created 2026-11-04 08:33:49.311035 UTC; down confirmed 08:44:06.703771 UTC; 617.392735 s VM lifetime.
- Estimated spend **0.325846 units at the planning 1.9 units/hour** (<0.5). Account balances/rates aggregate concurrent
  agents, so this is an estimate, not an isolated billing measurement. Shutdown balance 466.17 units, shared spend
  64.62 since the helper's 530.79 start. A subsequent status check says session not found.
- Failure-only local analysis runs with all six observations missing, retains coverage denominators, fails gates
  and returns no candidate. **This does not validate the successful GPU/export/enrichment path end to end.**
- Evidence: `dryrun/smoke/{lifecycle.json,smoke_summary.json,plan.json}`, `dryrun/smoke_launch.log`,
  `dryrun/smoke/failure_analysis/analysis.json`.

No retry. Fixes from the failed attempt: a SIGALRM budget now interrupts blocked uploads as well as polling;
control DB/meta already present in the verified bundle are no longer uploaded redundantly. Synthetic lifecycle
checks pass after these fixes; they have not received another L4 smoke. Provisioning failures check for an assignment
created before the CLI failed and clean up only the owned, previously absent session.

## Cost estimate

C-2b C/P observed mean per-parent/arm/seed wall: 3.912327849419389 s, contested local 2060. Conservative unmeasured
L4 speed factor 1.0; add 20 s startup per parent/arm and 10 min VM setup; planning price 1.9 units/hour:

| Stage | Parents × arms × seeds | Hours (one VM) | Units |
|---|---|---:|---:|
| 1 | 120 × 4 × 3 | 4.398264 | 8.356702 |
| 2 | 110 × 2 × 3 | 2.106149 | 4.001683 |
| 3 | 80 × 3 × 3 | 2.282466 | 4.336685 |
| Total | | 8.786879 | 16.695070 |

These are planning assumptions, not a measured L4 speedup or an approved production-run budget. CPU curvature
preparation is additional local time. Upload/network stalls are not reliably predicted by those wall estimates.
Evidence: shared `D3/dryrun/estimate.json`.

## Open handoff items / audit decisions

1. D2 DB/meta, surviving D flags and D/DS fingerprints; reproducible metadata schema and per-group ablated DB recipes.
2. Auditor freeze of contract/config; final run cap; I qualification (no isomer arm silently added).
3. Review scalar curvature reduction and inherited omission of Q7; baseline strata above show why the choice matters.
4. Failed smoke means successful worker/export/enrichment GPU path remains **unvalidated**. Any new smoke requires
   a new audit/owner directive; none is started here.
5. Stage1 candidate is provisional; final fold-in additionally needs stage2 holdout and once-only stage3 gates.
   Named cases outside the development panel (e.g. aqua_La3) are listed missing, not silently generated outside it.

**§4: commit, report, stop.**
