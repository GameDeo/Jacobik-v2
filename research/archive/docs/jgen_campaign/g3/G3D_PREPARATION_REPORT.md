# G-3d preparation — 2026-10-30

Implementation only, jgen-b, after H-R acceptance. Directive: H-next, Round 3,
CodexB. No master merge/fold-in, no panel run, no G-3d evaluation or policy
selection. Base fingerprint remains `6cc586e17ab1`; Colab spend zero.

## Runnable diagnostic tool

`research/bench/tools/jgen/g3d_diagnostics.py`, CPU only. It never implicitly
selects a panel. Smoke mode enforces one parent and samples two surviving rows
per native solver/E/Z call, switch after 104 updates, eight finish steps,
112 native total steps. Full diagnostics default to switch=200, finish=50,
total=250 and require an explicit O1 row-identity manifest. Regenerate those
identities on the approved folded-in base, rather than importing old scores,
energies or coordinates. Missing/culled/changed-assignment keys are reported as
failures, never replaced; every parent gets a ledger entry and the batch continues.

One-parent smoke command (from jgen-b with the AGENTS environment):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.jgen.g3d_diagnostics --parents XOJHIN --smoke \
  --out research/data/jgen/campaign/B/g/g3d_preparation/smoke_XOJHIN_final.json
```

Measurements and implementation:

- Aid activity at the switch: fraction of selected rows with nonzero gradient
  and per-row aid/score gradient-norm ratio. Projection activity is the norm
  removed by the actual bidentate rotation projection, not an invented scalar
  penalty. Score-gradient zero denominators are explicit nulls.
- Release ablations: same captured switch rows, fresh optimizer state, frozen
  native switch duals/ideals/Q3 commitments, current score + bend prior and
  native closure treatment. Keep-all, each aid alone, all aids, and the three
  two-aid subsets are stored so the chosen joint set can be reported without
  another search. Both thermal and curvature widths are exercised for every
  subset. Per row: score gain, MACE single-point change, class transition.
- Class changes: geometry label and graph-level assignment class before/after
  finish via existing classify/best_fit_assignment/topological_class_keys.
  Rows are retained and their new classes recorded; no class-based filtering.
- Thermal sigma mapping: Q1=translation in Å, Q2=rotation/bend in radians,
  Q5=ordinary/macrocycle torsion in radians, from THERMAL_SD. Curvature uses
  the implemented four fixed seed-11 Hutchinson probes in nats; nonpositive/
  nonfinite diagonal estimates retain thermal widths and are reported.
- Timing: per fixed-row continuation wall and all per-step curvature blocks,
  plus paired no-cull **native thermal/curvature whole-loop** timing at identical
  search states. The actual search-state digests at the switch must match.
  Charge max(sum of HVP-block wall over every native search/finish iteration,
  paired whole-loop wall overhead), so extra create_graph work is represented.
  The native finish timing spans the first switched step through the last
  optimizer update; assembly/MACE are excluded. A switch-only or extrapolated
  cost cannot select curvature; the projected cost is separately labelled.
- Convergence: per-row first 0.1 kT criterion check, distribution, censored count,
  p95 rounded up to a multiple of eight only when every selected row converged.
  First checks are one-based (a row meeting the criterion initially records 1).
  No cap is inferred from the converged subset alone.
- Delta/flip trace: native in-loop score-only stiffness-derived iRprop- probe,
  per-step and last-20-update aggregates. Translation, rotation, ordinary
  torsion, **macrocycle ring torsion**, bend are distinct. At-min denominator
  is live row-DOF updates; flips require nonzero previous effective/current
  gradients, with raw counts/opportunities retained. Frozen rows are excluded.
  Adam's unused delta state is not presented as an Adam step bound.

A new optional **CPU diagnostic observer** in JointIKSolver supplies live
score/aid graphs and exact switch multipliers before native backward. It is
None by default and never changes solver arithmetic. Diagnostic consumers
retain the graph and do not mutate it. The tool checks its frozen-surface
reconstruction against native score/shape/symmetry values at the switch
(tolerance 2e-4). Canonical replay regeneration changes only source hashes;
no generated update or CUDA graph arithmetic changed. No solver policy flag
or production fingerprint was changed.

Policy proposals require the complete eight-parent O1 identity cohort; extra
XOJHIN diagnostics cannot enter those pooled decisions. FEKKAJ/NEPNEF are kept
as separate ledger/diagnostic cases, excluded from policy judgement. The tool
applies the fixed single-release median-score/class-change rule, measures the
10% curvature cost gate, then uses the matching joint-release/sigma continuation
for the cap. Incomplete inputs and smoke mode never select a release set,
sigma source or cap. Real policy outputs remain proposals for audit.

## Draft contract

`research/bench/jgen_campaign/g/G3D_CONTRACT_DRAFT.md` remains **untracked and
uncommitted**. It includes C1/R2s/FA/FR+/IA, the clean24 source/hash, advance and
angle-distortion rules from G-3c, cal then tail seeds 11–13, parent bootstrap
(10,000, seed 11), energy/wall/parent guards, full-pool coverage, FA preference,
2060 spot check and resource accounting. Fold-in fingerprint, step-0 choices
and actual arm fingerprints remain pending. FEKKAJ/NEPNEF/XOJHIN are separate;
XOJHIN is treated as a search failure following the final audit correction,
not silently excluded as a score hole. The draft explicitly leaves final
cohort acceptance to the post-fold-in contract audit.

## Verification

`tests/test_finish_phase.py` ran **once to completion**, under CodexB's GPU
lock after nvidia-smi, deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`: **31 passed in 488.36 s**. Measured pytest
process wall **489.622973 s**. CUDA available; no skipped tests. This verifies
the accepted phase implementation; the test file and phase arithmetic were
not changed in this preparation. Logs and timing JSON are under
`research/data/jgen/campaign/B/g/g3d_preparation/finish_phase_tests.*`.

Three small diagnostic accounting tests pass: distinct macrocycle columns,
censor-aware cap, delta/flip denominators excluding frozen rows, and refusal
to select policy from incomplete/smoke inputs. Compilation, canonical replay
regeneration and git diff --check pass; default fingerprint asserted unchanged.

The one-parent smoke gate and its exact counts/timing are recorded below.
No cal/tail/clean24 evaluation or real eight-parent diagnostic run was started.

### One-parent CPU smoke result

**PASS**, XOJHIN seed 11, 126.922297 s measured parent diagnostic wall,
zero parent failures. One solver/variant snapshot, two fixed rows, **16 paths**
(eight release subsets × thermal/curvature), with two score/MACE/class records
per path. Native switch-surface reconstruction errors: score=0, shape=0,
symmetry=0. Native iRprop probe recorded 112 steps; macrocycle ring torsions
have nonzero counts (25,600 live row-DOF updates in the last-20 aggregate),
separate from ordinary torsions. Paired native thermal/curvature timing each
recorded one call with **identical switch-state digests**.

The intentionally short eight-step all-release thermal finish left both fixed
rows censored. This is reported, with no cap inferred. Smoke metadata leaves
release/sigma/cap unset. No energy or optimizer comparison is concluded from
these correctness outputs. An earlier 34.7 s smoke exercised the initial six
paths before the full subset/native-cost additions; the final smoke above
supersedes it and covers all final execution paths. Only XOJHIN was solved by
the diagnostic tool in this preparation.

Artifacts: `research/data/jgen/campaign/B/g/g3d_preparation/` in jgen-b:
`smoke_XOJHIN_final.json/.log` (final smoke), `smoke_XOJHIN.json/.log` (earlier
smoke), and `finish_phase_tests.json/.log`. The final smoke's derived policy
metadata was refreshed without rerunning a solve after tightening the
complete-cohort refusal guard. Baseline fingerprint `6cc586e17ab1`.

Stopped after implementation, the one-parent CPU gate, draft contract and
one complete finish-phase test-file run. No panel run, no real step-0 choices,
no H-3/G-3d execution, and no fold-in. The contract draft is intentionally
uncommitted for the post-fold-in fingerprint and audit.
