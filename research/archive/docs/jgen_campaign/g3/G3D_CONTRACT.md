# G-3d frozen evaluation contract — 2026-11-01

Base production 273f2ccbf892, master 347496f fast-forwarded into jgen/b-solver;
verification commit 5ba445a. Golden 245/245 for each profile and 31 finish-phase tests pass.
G3D_PROVENANCE.json records the full profile snapshot, DB/meta/model/manifest/row-identity
SHA256 hashes, platform, and exhaustive precomputed per-arm flags and fingerprints.
No step-0 or panel result exists at commit time.

The arm family is frozen before step 0: select the sole entry matching its mechanically
derived release set, sigma source and cap; no other config is permitted. Caps are multiples
of eight from 8 through 144 (250 budget minus bend-end 100). C1 uses 250 steps. R2s/IA
release all aids; FA/FR+ use the measured set. Record selected flags/fingerprints before
any panel run. Native plateau parameters remain unchanged. Failure to resolve a decision
by these rules stops for audit. No tuning or discretionary family selection is allowed.

## Step 0, after fold-in

CPU, seed 11: fixed 48 O1 row identities across HEDJOP, FEKKAJ, NUWFAM, QEZVEY,
DAMVAQ, USAPOU, NEPNEF, AJAQAD, regenerated with the patched base. Do not reuse
old coordinates, scores or energies. Missing/culled/unbuildable rows are ledger
failures, never substituted; NEPNEF remains an invalid-input diagnostic.
Record switch-state digests and per-E/Z scorer identity. XOJHIN seed 11 is a
separate diagnostic of COD folding under R2s, never a tuned optimizer setting.

Measure aid activity and aid/score gradient-norm ratio at step 200; release
each aid alone and all together on the identical switch rows for 50 fresh
Adam finish steps, with a keep-all finish control; report score gain, MACE
change and assignment/geometry-class transitions. Diagnostic continuations
freeze switch duals/ideals/Q3 commitments, disclosed separately from native
state-triggered arm execution. No optimizer evaluation is inferred from them.

Release an aid iff its single-release median score gain is >= 0 and class
changes are <= 5% of rows. Report the joint release-set class changes too.
Thermal mapping: translation=THERMAL_SD[Q1] Å; rotation/bend=THERMAL_SD[Q2] rad;
ordinary and macrocycle torsion=THERMAL_SD[Q5] rad. Compare with per-row diagonal
score curvature in nats (four fixed seed-11 Rademacher probes), report invalid
curvatures and thermal fallback. Use curvature only if its **measured whole
native-loop estimator cost**, including every search iteration where it is
currently evaluated, is <=10% of finish wall. Otherwise use thermal widths.
A switch-only timing or projection cannot establish this gate.

Finish steps are 0.14 sigma; stop at predicted remaining gain
0.5 sum((gradient_in_nats * sigma)^2) < 0.1 kT. Cap is p95 of observed steps to
convergence, rounded up to a multiple of 8. Unconverged rows are censored; do
not replace them by the budget or select a cap from the converged subset.
Report whether the derived cap fits the unchanged total 250-step budget and
bend-end constraint; an incompatible cap goes to the auditor, not tuning.

Per DOF class report delta-at-min share and sign-flip rate in the last 20
live updates, counts/denominators, and time traces. Classes: translation,
rotation, ordinary torsion, **macrocycle ring torsion**, bend. Distinguish
actual iRprop step bounds from the unused delta state maintained during Adam.
For XOJHIN, report COD torsions/fold and closure alongside delta/flip traces;
this is a search failure, not a score-direction exception. Solver maximises.

Policy is selected mechanically by the rules above. Incomplete cohorts, censored convergence preventing a cap, incompatible caps, or ambiguous timing refer to the auditor and stop execution. No cap is selected from converged rows alone. FEKKAJ and NEPNEF are excluded from policy decisions and retained as diagnostics, as in the preparation implementation.

## Arms, total at most 250 steps each

| Arm | Search | Finish |
|---|---|---|
| C1 | Patched production Adam with aids, 250 steps | None; finish_phase off |
| R2s | In-graph iRprop-, score only from step 0, no culling | Same throughout; delta0=.14 sigma, max=sigma, min=.001 sigma |
| FA | Production Adam with aids | Fresh Adam, .14 sigma per DOF |
| FR+ | Production Adam with aids | Fresh iRprop+, stiffness-derived bounds |
| IA | In-graph iRprop+ with aids through step 100 | iRprop+ on the score; reset delta from sigma at switch |

FA/FR+ switch per-row by aided plateau, never before bend release ends and
no later than 250-cap. IA explicitly switches at 100. R2s/IA release all aids
as prescribed; FA/FR+ release set comes from step 0. Search iRprop bounds are
the implemented production-lr mapping: delta0=lr, max=lr/.14, min=.001*lr/.14,
with existing per-DOF search scales. Do not tune these after outcomes.
Keep the bend prior and native seam/closure treatment. All arms run in-graph,
including R2s and IA. Verify arm switching actually updates finish_phase in
both solver modules and never triggers control-reuse shortcuts for active
finish/scorer/embed changes. This preparation does not run those arms on panels.

## Clean24 filter, then cal and tail, seeds 11–13

Frozen clean24 source:
`.claude/panel_G3b2_clean24_2026-10-30.txt`, SHA256
`a5851b60bc1e2be1a9c3c416750c056366297c3ac56039a766385a9bd89ef120`.
Keep all 24 fixed parents; failure ledger stays in the denominator. FEKKAJ and
NUWFAM are separate hole/embed checks for the filter as in G-3c; report both
included and excluded. NEPNEF is invalid and never silently replaced.

Advance iff paired seed-unit median top-1 MACE delta vs C1 <=0 and paired mean
floor delta <=0. An arm whose pooled-row maximum-angle-deviation p95 exceeds
C1 by >10 degrees does not advance. At most three candidates: retain FA,
then IA and R2s when eligible, then FR+ if a slot remains. No FR-/SR/legacy R2
arms. State finite-pair and parent denominators; do not tune a failed gate.

Run C1 and advanced candidates on cal, then tail. Primary judgement excludes
FEKKAJ (known score hole) and NEPNEF (invalid input); report all numbers included
also. **XOJHIN is separately reported and remains a search challenge**, following
the audit's final correction withdrawing its score-hole diagnosis. This draft
does not silently exclude it from judgement. NUWFAM is included on the patched
base and is also reported separately and in an exclusion sensitivity table.
These cohort details must be accepted with the final post-fold-in contract.

## Metrics, gates and choice

Primary: paired unrelaxed top-1 MACE polar-1-s float64 energy difference vs C1,
same reference composition/order, highest returned score selects top-1. Report
means and 95% parent-clustered percentile bootstrap CIs, 10,000 resamples,
RNG seed 11; average the declared seed units within each parent before
resampling parents. No unit bootstrap or relaxed-energy substitution.

Secondary: pool floor, score, obscene fraction (>200 kJ/mol), assignment/
geometry-class changes, maximum Q2 angle deviation distributions, steps used,
full-pool 1 Å symmetry-aware heavy-atom reference coverage, solver/assembly wall
with early stop and the corresponding fixed-budget wall. Parent coverage is
any hit over declared seeds, not top-1/floor-only coverage. Include failed/
missing cases in declared denominators and report finite paired subsets.

Pass on cal and tail: primary CI upper <2 kJ/mol, and improvement in floor
or obscene fraction with a paired parent-clustered CI excluding zero.
Guards: no parent worsens >25 kJ/mol on >=2 of 3 seeds; solver wall <=1.3x C1.
Report angle/class/obscene measures even for arms that fail. FEKKAJ, NEPNEF,
XOJHIN get separate per-seed tables; NUWFAM's embed-variant rows remain visible.

FA is preferred for simplicity if it passes. An iRprop arm can win over FA
only if it improves mean top-1 or floor by >=3 kJ/mol with a paired CI excluding
zero. No production choice is inferred when FA fails; present the registered
gates and evidence to the owner. No post-hoc rescue by dropping a parent.

## Resources, reproducibility and stop

Pair every candidate with C1 in the same bench invocation (--arms-spec), same seeds,
on the local RTX 2060 SUPER under runkit CodexB GPU lock; nvidia-smi before every GPU start.
At most five local CPU processes. No Colab; spend zero. These are screening/selection
results. Any fold-in additionally requires a later paired L4 confirmation.
CPU identity checks use OMP_NUM_THREADS=1; GPU repeat noise is not an identity test.
Full-pool XYZ exports use --xyz-compression xz. Profile 1–2 cases before panel runs.
Before cal/tail, profile one cal parent per arm, record total wall estimate, and stop
if it exceeds 8 h. Log progress per parent and preserve failure denominators.
Stop immediately on replay fallbacks or GFN2 embed refinement failed.

Stop at G-3d checkpoint: report step-0 rules, filter/advance table, cal/tail
CIs and guards, separate cases, cost and wall accounting, later L4 confirmation requirement, failures,
resource spend and fingerprint. Commit results and wait; owner chooses production.

## Amendment 1 — 2026-11-01, auditor ruling before resumed measurements

Authority: .claude/audit_jgen_G3d_step0_2026-11-01.md. Supersedes the old
fixed-key and complete-cohort requirements. Re-derive rows on production
273f2ccbf892: per parent seed 11, CPU default worker layout, production defaults,
top six delivered rows by returned score alone. Record keys, scores and
switch-state digest in G3D_ROWS_273f2ccbf892.json. Commit the manifest and
capture_switch row-level fix before any resumed step-0 measurements. No
old-to-new key correspondence is meaningful or reported.

Parents returning fewer than six contribute their available rows; ledger the
shortfall. Missing selected keys at step 200 are row-level failures; continue
other rows. Policy selection requires at least 30 of 36 rows across the six
policy parents (excluding FEKKAJ and NEPNEF), with at least three per parent.
Below that threshold stop and report; above it use the existing rules with
actual denominators. All release/sigma/cap rules, panel gates, 8-hour estimate
stop and later L4 requirement remain unchanged. G3D_STEP0_STOP.json is retained.

## Amendment 2 — 2026-11-01, NEPNEF fallback exemption

Per Addendum 1 of .claude/audit_jgen_G3d_step0_2026-11-01.md, NEPNEF is
exempt from the embed-fallback stop in step 0, clean24, cal and tail because
it is known invalid input and is already excluded from policy and primary
judgement. Record embed_fallback=true for its rows in the ledger, report it
in its separate table, and continue. An embed fallback on every other parent
still stops execution and requires a report. All other rules are unchanged.

## Amendment 3 — 2026-11-01, convergence diagnostic horizon and censoring

Per Addendum 2 of .claude/audit_jgen_G3d_step0_2026-11-01.md, finish the
50-step step-0 measurements, reusing completed same-base/code parent records;
select release set and sigma source by the existing rules. For only that
selected configuration, rerun the finish from the identical step-200 switch
state of every policy row for 144 steps. This is a convergence-only diagnostic;
arm budgets remain 250 steps. Keep rows censored at 144 in the denominator,
ranked above all observed convergence steps. Take the ceil(0.95*n)-th smallest
value across all policy rows, then round up to a multiple of eight. If that
order statistic is censored, the cap exceeds 144: stop and report; otherwise
continue to clean24 when the rounded cap is <=144. Never use the converged
subset alone. Report that a step-200 state is more settled than a plateau
switch near 100, so the measured cap may be optimistic for early switches;
this is descriptive and receives no correction. No other rules change.

## Amendment 4 — 2026-11-02, continue the independent optimizer arms

Per Addendum 3 of .claude/audit_jgen_G3d_step0_2026-11-01.md, the admissible-cap
gate remains failed and is not revisited: no change to criterion, horizon or
quantile. FA and FR+ have no contract-derived configuration and are recorded
as failed at step 0, not as filter losers; neither enters clean24. Run clean24
with C1, R2s and IA, seeds 11–13, all at the unchanged 250-step budget and the
resolved thermal widths. R2s is iRprop- score-only throughout; IA is iRprop+
with aids through step 100, then all aids released at the fixed switch.
Neither mode uses an adaptive cap. The runtime accepts cap omission only for
whole-run or fixed-switch modes; adaptive finish configurations still require
an explicit cap. All other arithmetic remains unchanged. C1 uses production
Adam with aids. Apply the existing advance predicates and retain IA then R2s
when eligible; FA/FR+ remain step-0 failures. Continue to step 5 with whatever
advances. All other gates, stop points, pairing, NEPNEF exemption and later
L4 confirmation requirement remain unchanged. Size is not established as the
cause of the cap failure; its cause remains unknown. Thermal widths and exact
arm flags/fingerprints are frozen in G3D_ARMS_AMEND4.json before GPU results.

## Amendment 5 — 2026-11-02, G-3e IA-k and independent twist diagnostic

Per Addendum 4 of .claude/audit_jgen_G3d_step0_2026-11-01.md, the first
G-3d clean24 failures remain on record and are not replaced or rerun under
that contract. G-3e uses C1 plus one candidate IA-k: registered IA, switch 100,
release symmetry only; shape/projection stay on through step 250. Both arms
use 250-step budgets, seeds 11–13, same paired bench invocation on the local
2060; IA-k uses resolved thermal widths. Exact flags/fingerprint are frozen
in G3E_ARMS_AMEND5.json before results. No new arithmetic is needed.
Clean24 cohort, advance predicates, angle guard, step-5 gates/stops, full-pool
exports and later L4 confirmation requirement remain unchanged. IA-k advances
to step 5 iff the existing predicates pass; otherwise stop and report.

Parallel CPU-only R2s twist diagnostic is descriptive, not an arm. Ten rows
are frozen in G3E_TWIST_ROWS.json: highest-score >45-degree octahedral-start
row per AZUSIA/USAPOU/HEDJOP seed, plus the highest-score row over seeds from
the next parent by count. Compare DOF score gradients to the unit's R2s top-1.
The nearest untwisted same-assignment partner uses the frozen C1/R2s/IA unit
union (R2s-only HEDJOP pools lack partners), heavy-atom fixed-order Kabsch RMSD,
same embed variant, <=45 degrees and unchanged class. Use full production FK,
including all ring/bend DOFs, to recover the exported coordinates; require
RMSD <=1e-5 A and score agreement <=1e-4 nats or ledger the recovery failure.
Interpolate DOFs linearly at 21 fixed points after globally aligning the
untwisted pose; choose nearest equivalent angular branches. Report total
and per-term Q1–Q7 scores, bend/seam contributions, score-gradient norms,
barrier from twisted score to path minimum, endpoint ordering, and CPU MACE
at both endpoints and the score's path minimum. No score optimization or
new arm. Keep starting-shell pair keys fixed; report any score-reproduction
limitations and distinguish a path barrier from proof of a full-space local
maximum. R2s remains an off-by-default built flag, reserved for the hypothesis
layer; its first screen and this diagnostic remain preserved. Future geometry
exploration would require current-shape pair keys or a geometry-agnostic score;
that is not implemented here. No post-hoc adjustment to the cap or angle gates.

### Amendment 5 CPU instrument clarification — owner-approved 2026-11-02

The frozen GPU exports omit native solver DOFs. Original coordinate-only
recovery attempts are retained as failures: some reconstruct geometry but
not the DOF bend prior, so they cannot establish the requested score gradients.
The owner explicitly chose "Regenerate CPU diagnostic rows". The replacement
CPU instrument is separately labelled; it does not replace any G-3d/G-3e GPU
result or turn into a new arm. Freeze the same ten parent/seed units and CPU
selection rules in G3E_CPU_TWIST_PROTOCOL.json before new CPU results. Capture
native R2s returned states plus CPU C1 reference states, one CPU worker and no
culling (GPU-like row retention), 250 steps. Select the highest-score >45-degree
octahedral-start R2s row and compare to that CPU unit's top-1; nearest eligible
same-assignment untwisted partner is from the CPU C1/R2s union. Ledger absent
rows/partners without replacement. Persist native DOFs/references/commitments,
verify score/geometry, and use the same 21-point fixed-key path and three CPU
MACE evaluations. Report both native score units and summed nats, with the
explicit scorer normalization factor. No changes to IA-k or its gates.
