# Checkpoint H-3b final report — 2026-10-31

**Status: stop for audit.** The frozen combined candidate passes the primary calibration gate, and the redesigned cap passes alone. Donor alone does not establish benefit under the frozen CI criterion. The original quadratic-tail design was dropped after consistent local regressions. Therefore the directive's full goal of establishing every surviving flag beneficial is **not met**. Numerical success also does not resolve whether the cap redesign satisfies the directive's strict modeling constraints; the substantive concerns are recorded below.

Work was local on `meep`, RTX 2060 SUPER, in `jgen-h` / `jgen/h-patch`. Colab spend **0**. No merge, promotion, production-default change, or confirmation-driven retuning. The auditor's `h3x/abl_2060/run.out` reached `ALLDONE` before GPU use. Development and evaluation used cal/tail only.

Implementation is `63d2bb2`; the frozen four-arm contract and complete evaluated source are `49814f8237ecf9a36f3b706ddf4502cd16665e2a`. Subsequent changes are reporting/diagnostic fixes only. The earlier donor-only contract was withdrawn when the owner requested further cap work.

## Confirmation design and integrity

[Contract](../../../../bench/jgen_campaign/archive/H3B_CONTRACT.md), [exact arms](../../../../bench/jgen_campaign/archive/H3B_ARMS.json), and [launcher](../../../../bench/jgen_campaign/archive/run_h3b_confirmation.py) were committed before opening confirmation results. Control, cap, donor, and combined ran in fresh processes, serially on the same local GPU, seeds 11/12/13, normal CUDA math, replay enabled. Cal has 119 valid parents / 357 paired units; tail has 9 / 27. CIs are 95% parent-clustered percentile bootstrap, retaining all three seeds per sampled parent, 10,000 resamples, RNG 11. Negative candidate-minus-control energy is better.

All eight phases completed with exit 0 and final `ALLDONE`: **1,560 records, 1,536 valid, 24 failures**. Every failure is the preregistered NEPNEF invalid-input/GFN2-fallback exclusion, repeated across four arms, two panels, three seeds. No unexpected failure, missing pair, metadata violation, nonfinite pool energy, or replay fallback occurred. Failure records were retained; none was silently repaired or excluded after inspection. See [failure ledger](../../../../bench/jgen_campaign/archive/H3B_FAILURE_LEDGER.csv) and [integrity JSON](../../data/jgen/campaign/H/h3b/final/integrity.json).

The full confirmation took **5,822.4 seconds (97.04 minutes)** end to end across phases; this excludes development and verification. All reference MACE energies match exactly across arms (maximum span 0). Full provenance hashes are in integrity JSON. Control fingerprint is `6cc586e17ab1`; cap `2ef36ca01e9c`; donor `3f75c9525a4e`; combined `7019c3ab7270`.

## Primary results

Mean top-1 MACE energy difference, kJ/mol:

| Arm | Cal mean [95% CI] | Cal benefit criterion | Tail mean [95% CI] |
|---|---:|---|---:|
| combined | -4.148 [-10.024, -0.402] | PASS | -5.515 [-16.878, +0.599] |
| cap | -1.877 [-4.549, -0.128] | PASS | -8.513 [-25.553, +0.024] |
| donor | -2.815 [-8.164, +0.228] | FAIL / benefit unproven | -0.016 [-0.592, +0.653] |

The combined primary requires upper cal CI <0 and passes. Cap alone also passes. Donor alone's CI crosses zero: this is inconclusive standalone evidence, not proof it is harmful. **Every arm passes the parent guard on both panels:** no parent is >+25 kJ/mol worse on at least two of three seeds. Tail CI width is report-only under the auditor-corrected contract.

The predeclared cal sensitivity excluding FEKKAJ is combined −1.777 [−3.911, −0.108], cap −1.893 [−4.589, −0.127], donor −0.433 [−1.372, +0.428]. Thus the combined and cap conclusions are not solely FEKKAJ's large win. These sensitivity results do not replace the primary.

## What changed to repair the cap and collapse

All changes below are gated by the existing `embed_haptic_li_cap` flag; its production default remains false. They are general code rules, with no CSD-code dispatch or per-complex parameter. Their chemical generality nevertheless needs audit.

1. **SO factorization (`ik/scorer.py`).** For a haptic face with at least two H sharing a pivot, select the existing fitted face-mean orientation table instead of independent per-substituent tables. Preserve the total face weight/normalization. This removes the particular independent marginal reward that favored coincident H. It is an aggregate face statistic, **not a full joint H–H distribution**. This is a model/factorization change, even though it reuses a fitted table and adds no summed score term. Face-mean alone did not eliminate collapse. Wall alone removed severe collapse in development but gave worse energies than the retained combination; the final panels do not isolate whether face-mean is necessary.

2. **Measure the existing geminal wall in the current face frame (`ik/fk.py`, `ik/ik_solver.py::_bend_prior`).** Retain every independent lean/swing column and its rotation axis. Detect exactly two same-element branches on a shared haptic pivot/face. Use current XYZ to form a metal-facing frame: for eta2, project the metal direction perpendicular to the face bond; for larger faces, orient a normal from the first three face vertices toward the metal, then form radial/tangent axes at the pivot. Sort the two branches by transverse coordinate `s`; use unwrapped polar angles `atan2(l_left, -s_left)` and `atan2(l_right, s_right)` as the two existing lean-wall coordinates. Both branches occupying the same side make one angle exceed 90° in magnitude, exposing a crossing that the old scalar/column measurement could miss. Existing stiffness, free widths, wall slots and normalization are retained. Undefined/collinear polar geometry falls back to the old wall with finite gradients. Device LongTensor indices avoid graph-capture fallback. Every caller supplies XYZ, and replay was regenerated using the canonical generator.

3. **Cap eligibility and placement (`engine.py::_gfn2_refine`).** Perceive aromaticity/conjugation on a copy so Kekule inputs receive consistent scope decisions. Reject locally neutral face-plus-neighbor motifs, including an anionic carbon adjacent to a compensating positive center. Skip aromatic monoanions with no H directly bound to the donor face; retain face-H and multiply charged components. For heteroatom-localized negative charge, use the existing sigma/kappa Li placement at 1.9 Å instead of pi-centroid placement. These fix local-charge/placement defects and narrow scope where capped starts regressed.

4. **No centroid restraint retained.** Carbon pi-cap Li remains free during GFN2. The Li-position diagnostics and restrained/normal-placement/full-allyl experiments are retained in the development log; those alternatives alone did not solve the failure. NUWFAM reference C1 has an angle sum about 332.15°, matching the roughly 332° capped pyramidalization: pyramidalization itself is not evidence of a defect. The original NUWFAM eta2 cap begins asymmetrically: Li–C1/C2 1.37/2.53 Å and Li–H 1.06 Å; after GFN2 Li–C1/C2 are 1.92/2.37 Å. A full-allyl constrained attempt held Li 1.893 Å above the face (Li–C distances 2.28/1.96/2.26 Å), yet still produced 12–13° geminal angles and a roughly 1,922 kJ/mol seed-11 top-1. Direct solver-coordinate capture showed lean columns near ±57°, swings only about ±2°: the free lean/reference geometry drove the collision. Li migration was worth testing, but did not justify replacing the final repair with a blanket centroid-normal restraint. These tests do not prove every possible restraint design fails.

### Modeling and generalizability questions requiring audit

- The wall adds no new summed score slot, but changing its coordinate and coupling changes the effective prior. It also adds pair detection, frame construction and sorting logic. I cannot certify compliance with the directive's literal “must not add complexity” restriction merely because existing slots/parameters were reused.
- The 57° eta2 / 40° eta3 widths came from the old asin(bend) coordinate. Reusing them for the new atan2 coordinate has **not** been recalibrated against the full training distribution. Thirty-four development references had zero additional wall penalty; this is limited reproduction evidence, **not proof that no refit is needed**. The earlier log's “no class refit was needed” statement is withdrawn as too strong.
- Same-element branches are a broader category than chemically equivalent geminal H. The opposite-side rule, larger-face frame, branch sorting and behavior on chemically inequivalent pairs need review before promotion. Retaining all DOFs does not mean the accessible low-penalty geometry is unchanged.
- Face-mean is a separate modeling choice. The collapse can be prevented by geometric wall handling in development without proving the face-mean substitution necessary. No final factorial ablation isolates these two pieces.
- The aromatic face-H scope is an **empirical design restriction**, not an established universal chemical law. SOBHAS/EBAYIQ/XEVDAE are returned toward control by **withholding their caps**, not by repairing their capped geometry. Better aggregate results do not establish correctness on every anionic haptic donor.
- No training refit, new parameter fit, or production promotion was performed. These caveats remain open for the joint audit; the report does not label the design universally validated.

## Starting-SO-mode restriction proposal

The owner suggested retaining each substituent's starting SO mixture mode. A read-only diagnostic reconstructed the old NUWFAM starts by E/Z variant, assignment and trial, then inspected the original independent-H model (before the new wall/face-mean). Old outputs have no initial-state hashes, so reconstruction is qualified. This is **not a locked-component optimization ablation**, and endpoints cannot rule out transient switches.

Of 96 pool rows, 82 started with both H in the same MAP component. Of 17 collapsed rows (<15° H–C–H), 15 started in the same component; all 17 ended in component pair (0,0), and six had the same MAP pair at start and end. The crystal reference is also (0,0), with posterior probabilities about .974/.965 and a valid 108.63° H–C–H angle. The modes have bend means −.389/+ .113, weights .797/.203, and bend sigmas .141/.344: they do not provide distinct geometric side labels.

Consequently a statistical mode lock is not guaranteed to supply the missing side information and could trap an inappropriate initial assignment in other cases. It might still alter trajectories; that has not been tested. A geometric side constraint is a different intervention from freezing a mixture component. See [reproducer](../tools/jgen/diag/h3b_start_mode_check.py) and [raw analysis](../../data/jgen/campaign/H/h3b/start_mode_check.json). No production change followed this diagnostic.

## Named outcomes and geometry

Combined-minus-control top-1 kJ/mol, seeds 11/12/13:

| Parent | Seed 11 | Seed 12 | Seed 13 | Mean |
|---|---:|---:|---:|---:|
| FEKKAJ | -278.64 | -322.74 | -250.61 | -284.00 |
| NUWFAM | +40.75 | -130.35 | -15.37 | -34.99 |
| XOJHIN | -3.57 | -0.28 | -2.59 | -2.14 |
| QUTWOT | -67.91 | -127.36 | -62.38 | -85.88 |
| EPUWAQ | -54.99 | -44.34 | -48.47 | -49.27 |
| FEXWUC | +1.36 | -2.30 | +1.69 | +0.25 |

Cap alone retains NUWFAM (mean −34.82), QUTWOT (−124.36) and EPUWAQ (−49.29) gains. NUWFAM still loses about +41 on one seed; that failure is reported rather than hidden by its average. FEXWUC's original tail-associated improvement is **sacrificed**, not credited to this package. All parent/seed results, including the remaining named movers, are in [parent summary](../../../../bench/jgen_campaign/archive/H3B_PARENT_SUMMARY.csv) and [paired analysis CSV](../../data/jgen/campaign/H/h3b/final/analysis.csv).

Final full-pool minimum geminal angles across all three seeds:

| Parent | Control | Cap | Donor | Combined |
|---|---:|---:|---:|---:|
| NUWFAM | 59.66° | 87.06° | 58.03° | 89.53° |
| QUTWOT | 82.94° | 90.36° | 81.79° | 90.41° |

There are zero <15° measurements in every final arm for these parents. NUWFAM has 96 rows per arm; QUTWOT has 48 rows with two measured pairs (96 measurements). This supports removal of the observed severe collapse on these pools, not a guarantee against every geometric failure. [Per-seed geometry](../../data/jgen/campaign/H/h3b/final/geminal_geometry.json).

## Secondary endpoints

Floor differences, kJ/mol, with the same clustered CI:

| Arm | Cal | Tail |
|---|---:|---:|
| combined | -1.409 [-2.889, -0.186] | +1.408 [-15.963, +18.268] |
| cap | -1.071 [-2.304, -0.145] | -5.962 [-17.794, +0.007] |
| donor | -0.394 [-1.253, +0.344] | +6.448 [-1.010, +18.625] |

Other frozen secondary endpoints (coverage numbers are hits / valid parent-seed units; rho is the descriptive median per-pool Spearman correlation of score with negative MACE energy, not a pooled inferential estimate):

| Panel / arm | >200 kJ/mol top-1 | Top-1 / floor / full-pool 1 Å hits | Median rho | Solve-wall ratio vs control |
|---|---:|---|---:|---:|
| cal / control | 42/357 | 115 / 119 / 234 of 357 | 0.638 | 1.000 |
| cal / combined | 37/357 | 116 / 121 / 233 of 357 | 0.664 | 1.051 |
| cal / cap | 37/357 | 115 / 121 / 236 of 357 | 0.644 | 0.997 |
| cal / donor | 40/357 | 116 / 119 / 232 of 357 | 0.662 | 1.041 |
| tail / control | 24/27 | 3 / 1 / 6 of 27 | 0.570 | 1.000 |
| tail / combined | 24/27 | 3 / 2 / 6 of 27 | 0.575 | 0.999 |
| tail / cap | 24/27 | 3 / 1 / 6 of 27 | 0.570 | 0.997 |
| tail / donor | 24/27 | 3 / 2 / 6 of 27 | 0.575 | 0.995 |

Solve wall is deduplicated per parent because each seed record repeats the multi-seed solve time. Total phase wall above includes more than solve time. Complete per-pool correlations, >200 transitions, paired rows and sensitivity outputs are in [analysis JSON](../../data/jgen/campaign/H/h3b/final/analysis.json). These secondaries do not replace a failed benefit criterion.

## Flag decisions and failure classification

| Item | Diagnosis / evidence | Final disposition |
|---|---|---|
| Donor pairs | Crystal-derived scales unchanged; large FEKKAJ win, no repeated-loss guard violation. Local standalone CI crosses zero. | Kept as is in evaluated candidate, **standalone benefit unproven**. hh .95 / ho .90 / oo .80. |
| Geminal collapse | Degenerate per-H feature and existing wall measurement allow invalid same-side geometry; geometric intervention removes severe collapse in tested pools. | Implementation/model defect addressed by redesigned cap path; philosophical admissibility remains open. |
| BOHWUT / neutral motifs | Local charge cancellation was missed. | General local-neutral rejection, no per-complex exception. |
| UMUBAF | Localized heteroatom charge received inappropriate pi placement. | Existing kappa placement used. |
| SOBHAS / EBAYIQ / XEVDAE | Capped starts can regress; centroid constraints alone did not establish a general repair. | Empirical aromatic/no-face-H scope restriction; not a repaired capped geometry. |
| Quadratic tail | KIXVUJ +9/+43/+55; WEYJIT +22/+9/+193 local losses. Correctly mapped WEYJIT crystal receives about +372 Q2 even under best candidate assignment. Fixed empirical-scale hardening can penalize valid geometry and perturb the search. | Original design dropped; no threshold tuned to these outcomes. A differently calibrated future design is not ruled out. |
| CIFZEZ / IFUROQ platform changes; mixed HEDJOP/LOPJEF and near ties | Some parent behavior changes between L4 and 2060; endpoints alone do not identify all as pure noise. | Report paired local results, avoid cross-platform attribution or blanket noise dismissal. |
| FEXWUC | Tail gave some real large improvements alongside its regressions elsewhere. | Known win sacrificed with dropped tail. |

Hypotheses H1–H16, Li positions relative to the face, failed attempts and their predictions/results are retained in [dated development log](H3B_LOG.md), attempt patches beside it, and raw `h3b/` outputs. [Development top-1 score terms](../../../../bench/jgen_campaign/archive/H3B_DEV_TOP1_TERMS.csv) provide the control/combined decomposition for NUWFAM, QUTWOT, BOHWUT, UMUBAF and SOBHAS. Full diagnostic CSVs contain 960 rows per arm, no failures; known-prior reconstruction residual is <2.7e-6 for NUWFAM and <4.3e-7 for QUTWOT. This decomposition is exploratory development evidence, not a new confirmation arm.

## Verification, failures and corrections

- Stable final implementation full CPU suite: **410 passed, 20 skipped, 3 xfailed**, 726.94 s. Production and u1 golden: **245 arrays each, zero differences**. Canonical replay regeneration matches checked-in source exactly. Defaults retain fingerprint `6cc586e17ab1`.
- The embedding-validity suite failure was logger pollution: XECSON audit `get_case -> tools.jdev.env.use_src` sets the jacobik logger to ERROR; the multiseed worker also changes it. A test autouse fixture now saves/restores logger level using `setLevel`. This changes test isolation, not production generation.
- Deterministic CUDA + `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 200 steps, actual FAST_FK on FEKKAJ/NUWFAM/QUTWOT: identical initial theta/reference/scorer hashes, raw eager-repeat and eager/replay error **0**, identical returned rows, 198 graph steps replayed, no error. Frozen deterministic gate passes without relaxing tolerance.
- **Ordinary-CUDA strict parity FAILS.** Max eager-repeat vs eager/replay raw errors: FEKKAJ .04108 vs .26943; NUWFAM .04829 vs .04679; QUTWOT .006971 vs .006339. Backend variation exists, but FEKKAJ graph difference exceeds its measured eager-repeat difference; it is not justified to call all differences the same noise. Deterministic equality supports graph semantics under that backend only. Confirmation used normal math consistently across arms. Both failures and deterministic passes are retained.
- Earlier full-suite output collected during source changes had four missing-attribute errors plus the logger failure; it was invalid as a final verification and was replaced by the stable rerun above. Earlier graph fallback caused by Python-list face indexing was fixed with device tensors; final eight-phase fallback count is zero.
- An early crystal diagnostic used `gen_to_ref_idx` in the wrong direction. Despite its name, it required `np.argsort` for the intended mapping. Element order was checked after correction. Earlier large KIXVUJ/FEXWUC reference-penalty conclusions are withdrawn; corrected KIXVUJ is about −.004, FEXWUC 0, while WEYJIT remains about +372.
- The first pool-term extraction attempted nonexistent `q2_angles`; that diagnostic failed (zero rows / 15 errors). The helper now uses `q2_dmd_angles` and installs embedding-validity checks; the successful replacement outputs are separately retained. A broad heavy-branch experiment was stopped by its own PID after failing its hypothesis; partial results remain. No failed attempt was discarded as though it passed.
- The old claim that a zero-penalty 34-parent check proved no class refit necessary, and the broad characterization of ordinary parity differences as just noise, are superseded by the explicit qualifications here.

Verification artifacts: [suite](../../data/jgen/campaign/H/h3b/final_full_suite.out), [golden](../../data/jgen/campaign/H/h3b/final_golden.out), [deterministic replay](../../data/jgen/campaign/H/h3b/replay_deterministic.json), [ordinary replay/repeat](../../data/jgen/campaign/H/h3b/replay_nondeterministic_noise.json). Tests cover side crossing, pair swap and rigid rotation, eta2/eta3, degenerate gradients, scope/aromaticity/local charge, placement and SO parity. The final report commit adds diagnostics/documentation only; no additional model testing or tuning follows confirmation.

## Audit handoff

The numerical package primary passes; cap alone passes; donor-alone benefit and strict modeling/generalizability requirements remain unresolved. All-goals status is **false**. The final branch is offered for audit rather than promotion. Review the face-mean necessity, wall coordinate/calibration and scope, aromatic eligibility rule, and ordinary-CUDA parity limitation together. Stop at **Checkpoint H-3b**; wait for audit, with no merge or default changes.
