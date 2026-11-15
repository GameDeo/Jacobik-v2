# D3-H audit amendments — 2026-11-04 (checkpoint)

Historical checkpoint. The Q7 measurement blocker below was resolved by the 2026-11-06 audit §4 ruling;
see `D3_Q7_RULING_REPORT.md` for the implemented scope and identity requirements.

Audit: `.claude/audit_jgen_D3H_2026-11-04.md`, accepted checkpoint f6070d3. Implementation snapshot fb127e7.
No real D3 arm, locked run, local GPU use or production change. Contract remains DRAFT pending D2/freeze.

## Persistent arm workers (amendment 2)

`worker.py` now starts one persistent subprocess per arm/VM and drives parents in strict lockstep. Each arm loads its
own import-time flags, DB and float64 MACE model once. The geometry-dependent scorer is rebuilt once per parent;
a scorer cannot be shared across different molecules. Parent acknowledgements carry PID and model/DB load counts;
transient CUDA allocations are released between parents. Startup and parent wall are recorded separately.

A failed parent or failed worker produces failure records and does not stop other arms/parents. Workers close by PID
at completion; no per-parent respawn or fork occurs. Tests cover four-arm ordering, continuation after failure, and
one subprocess PID serving multiple parent requests. The smoke covers C and DS as authorized, not a four-arm D3 run.

## Curvature (amendment 3)

The wrapper appends Q7 to `forces.py`'s measured term list on the same aggregation-normalized score scale, computes
both Q1–Q6 and Q1–Q7 scopes from the same score evaluations, and gates only the latter combined view. The registered
control-relative rule is implemented: if the Q7-inclusive control fails median 1.15–1.40 or rank ≥0.80, an arm must
have median within ±0.10 of control and rank ≥ control−0.03. Strata remain descriptive. Legacy-scope, missing control, or unmeasured active-Q7
summaries fail closed. Stage 1 prepares/validates C as well as D/DS curvature. Both scopes were recomputed on CPU for
all 80 u1 cases / 15,055 directions, with GFN2 validity guards and zero failures.

Both currently give combined median 1.3315397846869683, rank 0.8639999488508491; crystal median 1.501382083939354,
rank 0.8512096774193549; pool median 1.2959918130007786, rank 0.8699985466551534. All Q7-minus-legacy curvature
entries are zero. Investigation found that production `bend_penalty="energy"` puts the active Q7 prior in
`JointIKSolver._bend_prior(thetas)`, while `Scorer._q7` is only populated by the alternate Cartesian angle penalty.
Thus merely appending the breakdown key cannot observe the current FK prior. The gate now fails closed unless a
summary affirmatively covers the active FK prior; this Cartesian-only baseline cannot qualify. A clarification was requested while
independent worker/smoke work continued. This limitation is not a measured absence of a physical Q7 effect.

## P fallback (amendment 4)

D/DS retain the original full gate and top-1 tie-break (D wins ties). If neither qualifies, standalone P (arm S) is
eligible on its already-qualified C-2b correctness route iff the stage1 top-1 interaction upper CI <+2 and S-vs-C top-1
upper CI <+2. The floor interaction remains reported. Missing contrasts, DRYRUN and reproduction cannot authorize
fallback. Output identifies candidate `S`, display label `P`, and separate `p_fallback.qualified`; stage3 accepts that
registered fallback and retains its own reserve gates. A selected DS includes P. Registered D ablations remain
available after D/DS fail even when P qualifies alone.

## Tests and reproduction

55 focused tests passed before the smoke; final focused count is 56 including C-2 reporting compatibility. New tests cover Q7 arithmetic,
absolute/relative curvature edges, missing/legacy scope refusal, both strict P bounds, D/DS priority, P fallback through
analysis and locked preconditions, persistent worker PID/order/failure continuation, and embedded plan paths.
The amended analysis still reproduces every C-2b P-vs-C metric/CI, guard, wall value, per-parent value and failure list
exactly. No historical record was modified. Evidence: shared `D3/dryrun/amendments_tests.log` and
`amendments_reproduction_check.json`.

## Amendment-1 smoke (one owner-approved attempt)

The plan is embedded in the bundle before provisioning, avoiding the small-plan Colab exec/upload path that stalled
last time. The pre-existing SIGALRM now covers a 12-minute VM lifetime including uploads/setup, with fetch/count/down
in finally; local bundle construction is before that charged interval. Verified DB/meta already in the bundle are
not uploaded again. The only requested VM is jgD3s (L4); plan is three parents CAXTAX/FELSAU/JEGPUJ, seed 11, C+DS,
production DB substitute, every record DRYRUN. No retry is authorized or performed.

**PASS.** One attempt from fb127e7; no retry or second VM. All six records are successful and labelled DRYRUN in
both the record and arm identity. Six xz pools parsed locally after fetch: **266 rows**, matching row manifests,
energy arrays and symmetric-heavy coverage arrays; all energies finite. Shell/eta2 metrics and row-step ledgers are
present; all six replay-fallback counts are zero. The stage1 analysis completed once and returns no candidate.
This is harness validation, never evidence for adopting a search/score arm.

Fetch verification passed **6/6 records and 40/40 expected files**, every SHA256 matching, before down. Worker C PID
4179 and DS PID 4285 each served all three parents, with one DB/model load. Ready GPU snapshot: 630 MiB total,
308 MiB per process. Maximum observed active-process Torch allocation was 4,559,579,648 bytes (4.25 GiB); idle model
allocation 84,804,608 bytes per worker. No memory pressure in this two-arm smoke; four-arm residency was not exercised
on a GPU (four-arm lockstep was tested synthetically). No 2+2 fallback was needed here.

Created **2026-11-04 09:08:15.804514 UTC**, down confirmed **09:19:07.262063 UTC**: 651.457549 s VM lifetime.
Estimated spend **0.343825 units at the planning 1.9 units/h** (<0.5). Account billing is shared with other agents;
isolated actual spend is unavailable. Shutdown shared balance 464.83 (65.96 since helper START_BALANCE=530.79).
Subsequent `colab status -s jgD3s`: session not found. The prior upload-budget fix remained in force.

Evidence under shared `D3/dryrun/smoke_amendment1/`: lifecycle, smoke_summary, export_check, launch plan and
`d3_output/{completion,fetch_check,manifest,analysis}.json`, worker_events.jsonl and per-arm logs. The launch log is
`dryrun/smoke_amendment1_launch.log`. No real D3 result was written.

## Revised cost estimate

Estimator still uses the C-2b contested-2060 C/P generator mean 3.912327849419389 s/parent/arm/seed and an unmeasured,
conservative L4 factor 1.0. Persistent startup charged **once per arm/VM**, 23.570386 s (maximum of 20 s assumption and
observed smoke startup); measured mean parent process/MACE/export/enrichment overhead **3.213724 s/parent/arm**.
Add 10 min setup per VM, planning price 1.9 units/h. The overhead estimate is a three-parent pilot, not a population
speed claim; CPU curvature time is additional. One VM per stage:

| Stage | Parents × arms × seeds | Hours | Units |
|---|---|---:|---:|
| 1 | 120 × 4 × 3 | 2.186284 | 4.153939 |
| 2 | 110 × 2 × 3 | 1.093416 | 2.077490 |
| 3 | 80 × 3 × 3 | 1.183023 | 2.247743 |
| Total | | 4.462722 | 8.479172 |

Previous estimate: 8.786879 h / 16.695070 units. Startup is no longer charged 940 times. A second VM partitions parents
and adds its own setup/startup; this estimate does not assume free parallelization. Real D3 cap from audit: 20 units,
conditional on D2, measurement resolution and auditor freeze. Evidence: `dryrun/estimate_persistent_after_smoke.json`.

## Open measurement definition / checkpoint stop

Amendments 1, 2 and 4 are validated. Amendment 3's numerical scope machinery, both-scope recomputation and registered
absolute/relative decision rule are implemented and tested, **but its active-FK-prior measurement is unresolved**.
The existing cache supplies Cartesian points/displacements, not the bend coordinates on which the active prior is
defined. Substituting an angle penalty, reconstructing bend coordinates, or adding bend directions would each change
the measurement definition; none was silently substituted. A clarification is pending. Both analysis and real stage1
preflight now refuse a summary without affirmative active-FK-prior coverage; the Cartesian-only values above cannot
qualify a D arm. Diagnostic: `dryrun/curvature_q7/measurement_diagnostic.json`.

D2 DB/meta/flags/fingerprints and ablated-DB recipes remain placeholders; isomer admission remains conditional on its
separate audit. Contract remains **DRAFT, not frozen**. No production/src or C-2 change; fingerprint unchanged.
No local GPU, locked-set opening, merge or push. Commit/report/stop here; no further smoke is authorized.
