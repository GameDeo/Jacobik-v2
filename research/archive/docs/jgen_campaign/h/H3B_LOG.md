# H-3b development log

## 2026-10-31 — local start

Directive: `.claude/directive_jgen_H3b_iterate_2026-10-31.md`; baseline
7e972f0, branch jgen/h-patch. Local CPU (at most three workers) and RTX
2060 SUPER only; no Colab. No GPU work before auditor ALLDONE.

### H1: capped embed changes the lean wall's effective reference

Prediction before testing: capped NUWFAM substituents start substantially out
of the face plane. The existing wall constrains beta relative to that embed,
so its nominal 40-degree window may include the face-normal crossing where
two substituents can exchange sides. Reconstruct the CPU initial frame and
compare projected substituent directions with exported collapsed rows before
changing the mechanism. Preserve all independent DOFs and the existing prior.

### Auditor ablation: completed non-cap arms

Paired local deltas (seeds 11/12/13, kJ/mol):

| Parent | Donor alone | Tail alone | Combined |
|---|---|---|---|
| CIFZEZ | -4.80 / +0.34 / -0.14 | -35.84 / -1.37 / +0.93 | -39.05 / -1.37 / -0.28 |
| KIXVUJ | -9.76 / -0.27 / -5.05 | +8.50 / +43.27 / +54.95 | +16.11 / +42.90 / +8.84 |
| HEDJOP | -5.06 / -13.07 / +50.57 | -47.59 / +16.94 / +26.96 | +49.74 / -13.49 / +25.44 |
| WEYJIT | -26.89 / -6.87 / -3.35 | +21.74 / +9.16 / +193.26 | -12.01 / +7.46 / -0.99 |
| IFUROQ | 0 / 0 / 0 | -10.13 / -2.56 / -2.51 | -10.13 / -2.56 / -2.51 |
| FEXWUC | +1.36 / -2.30 / +1.69 | -69.52 / +15.30 / -63.83 | -69.51 / +15.23 / -63.83 |

Tail alone also worsens KIXVUJ floors +6.03/+43.27/+32.96; these are
optimization/geometry effects, not simply top-1 tie selection. IFUROQ's L4
loss does not reproduce locally; platform dependence prevents transferring
individual parent effect sizes. No parameter change based on these outcomes.

### H2: cap direction and unconstrained migration (owner suggestion)

Raw xTB capture confirms Li is unconstrained. More decisively, NUWFAM is
represented by an eta2 face (C1/C2), not eta3. The eta2 placement uses an
in-plane centroid-minus-outside vector: initial Li–C1/C2 distances are
1.37/2.53 A, and Li–H is only 1.06 A. After refinement distances are
1.92/2.37 A. Thus the intended face cap begins as a strongly asymmetric local
contact. Cp placement is already roughly symmetric (2.24 A initially).

Prediction before testing: constructing the eta2 normal from its bond and a
bonded outside heavy neighbour, then using the existing xTB distance
constraints to preserve the cap-to-face distances, will avoid the asymmetric
local cap and reduce C1 pyramidalisation and collapsed H rows. General rule,
no parent-specific treatment; keep all bend DOFs and existing score unchanged.
First test NUWFAM seeds 11–13 cap alone against auditor's local control/cap.

### Test isolation

Reproduced the H-3 suite failure with just multiseed GPU-fast-path test (CPU
forced) followed by embedding-validity tests: 1 failed, 2 passed. The worker
sets logger `jacobik` to ERROR and the test did not restore it. Added logger
level restoration through its existing monkeypatch fixture; retest pending.

H2 result: normal placement + constrained face alone **fails**. NUWFAM min
H–C1–H angles remain 6.6 degrees (1/1/2 rows below 15 degrees); seed 12 top-1
is 2029 kJ/mol. Thus Li placement is relevant but does not close the existing
bend-frame loophole. The score and DOFs are unchanged.

H3 prediction: eta2 geminal substituents must share their covalent plane,
instead of each defining a different plane with the face bond. Use the
bond and difference of geminal substituent positions to define that plane;
each still retains its own lean and swing. The existing sideways prior then
prices convergence of their azimuths. Gate this frame correction with the
experimental cap flag, preserving production defaults. Retest NUWFAM.

### Tail flag: fundamental calibration flaw; drop

CPU scorer evaluation of the reference crystal (builder-order mapping from
saved control pool) gives Q2 on/off: KIXVUJ -54.64/-1.99, WEYJIT
-124.99/-1.33, FEXWUC -650.78/-3.00. Hardening an empirical Student-t mixture
at two fitted scale units catastrophically penalizes observed crystal
geometries. Combined with the independently reproduced KIXVUJ floor losses
and WEYJIT +193 kJ/mol top-1 loss, this is a fundamental flaw in this tail
rule, not a tie flip. Do not tune the threshold on panel outcomes. Drop
angle_tail_quadratic from the candidate package; retain the off-by-default
implementation for audit. Its FEXWUC wins cannot justify these penalties.

H3 first run was not a valid test of the shared-plane correction: inspection
found duplicated axis construction in `BendPlan.apply` and `axes_pivots`.
The first patch changed only the diagnostic/metric path. Removed the duplicate
construction so FK uses `axes_pivots` directly, regenerated replay solver, and
repeated as cap_frame2. The invalid cap_frame run is retained, not interpreted.

Isolation retest: the same ordered three tests now all pass (16.54 s).
Cap-placement and independent-DOF unit checks: 3 passed.

cap_frame2 failed all seeds before solving: composed FK merged bend plans
without the new plane-reference tensor. Added that tensor to the existing
merge, added a composed-FK equivalence test, and reran as cap_frame3. Failures
remain in the ledger and were not converted into geometry results.

H3 (valid composed run) reduces collapse but does not eliminate it: pool
minimum geminal angles 37.6/13.7/13.7 degrees. Top-1 is still worse
295.6/270.9/252.6 kJ/mol. The capped geminal plane starts tilted by about
27 degrees; a further free 57-degree eta2 lean can approach the normal.
H4 prediction: reference the **existing** lean wall to the shared covalent
plane (add the initial signed lean to the bend coordinate), rather than
implicitly treating the pyramidal embed as planar. This retains the same
wall widths/stiffnesses, independent columns and score terms. Test NUWFAM
before accepting any cap change. No parameter fitted to energies.

H4 result: **failed**. Pool min geminal angles 49.3/3.7/3.0 degrees;
top-1 342.2/270.9/2004.6 kJ/mol. Do not accept any cap fix. The shared
plane plus shifted soft wall does not reliably prevent convergence of two
independent substituents onto the same scalar orientation mode. Hardening
walls or adding a geminal angle penalty is outside the permitted fix design;
a safe replacement needs a separately validated, nondegenerate orientation
feature and potentially its crystal refit, rather than another ad hoc local
constraint. After this fair attempt, drop the cap from this package. Preserve
the failed patch and its unit tests as H3B_CAP_ATTEMPT.patch / _TEST.py;
restore production source. All pool exports remain under h3b. This loses the
EPUWAQ cap win; report that cost explicitly rather than accepting collapse.

Tail crystal check corrected to evaluate all candidate-specific Q2 tables:
best Q2 off/on KIXVUJ -1.45/-102.92, WEYJIT -1.33/-124.99, FEXWUC
-2.03/-41.89. The conclusion remains; initial single-table values are
superseded. Drop tail; retain donor-pair dict as the only candidate flag.

Next development test prediction: donor alone preserves FEKKAJ's repulsion
win while avoiding the tail and cap mechanisms; check all prescribed movers,
cap parents and ten neutral sentinels with paired local controls before
freezing the final contract. No parameter changes.

Development set frozen to 34 unique parents. Neutral sentinels are the first
ten cal codes with all three H-3 absolute seed deltas <2: BABNAV, BIBFUN,
CIXYUD, FASPOH, GOTWET, HOYBED, JAYCEV, LOPCAW, MUBREJ, MUFFEA.
Include FEKKAJ, all six non-cap movers, and all 17 prescribed cap parents.
Local ablation timings predict roughly 10–20 minutes for both development
arms and under two hours for paired cal+tail confirmation (comfortably below
the ten-hour cap); refine the estimate from this development run before launch.

## Owner steering — continue the cap

Owner explicitly asks for a stronger attempt on the cap. The earlier drop is
withdrawn. H3B_CONTRACT donor-only design is **superseded**, and no confirmation
has started. Keep completed development controls; do not launch donor-only
confirmation. Restore the failed attempt for diagnosis, trace actual returned
solver coordinates and reference frames, and pursue a correct general fix.

### H5: cap the conjugated anion, not its truncated donor label

NUWFAM ligand SMILES `[CH2-]C(C)=CC[B-](aryl)3` declares only donor atoms
[0,1]. RDKit marks bonds 0–1 and 1–3 conjugated, but 1–2 and 3–4
unconjugated. The actual delocalized allyl component is therefore [0,1,3].
Prediction before run: extend each anionic haptic cap group over its RDKit
conjugated component, then hold Li above that full face using existing
constraints. This prevents the counter-ion from imposing a local sigma
carbanion geometry. Revert all bend-frame/wall changes for this test; original
DOFs and scoring. No parent-specific code or new parameter.

H5: C1 angle sum improves to 341.69 degrees; Li stays 1.893 A above the
full allyl face with distances 2.28/1.96/2.26 A. Nevertheless min returned
H–C1–H angles remain 12–13 degrees and seed 11 top-1 is 1922 kJ/mol.
Direct solver-coordinate capture (no inverse fitting) identifies the motion:
collapsed rows have lean columns about +/-57 degrees (at the eta2 wall),
swings only about +/-2 degrees. Ordinary backbone bends are small. Thus
independent free lean, rather than swing, drives collapse; the wall is active
but its reference/width admits the collision. Retest shared-plane + signed
wall offset on this corrected allyl embed, recording the actual offsets and
reference vectors before choosing the next design.

### H6: fix pose-dependent normal sign in the shared-frame wall

The shared-wall trace has identical static offsets (-20.9 degrees), but the
runtime eta2 plane's sign is aligned toward the metal **after** ligand rigid
rotation. That sign can reverse between candidate poses, so the stored
signed offset does not consistently refer to the runtime axis. Prediction:
orient the geminal shared plane by stable atom order, and compute the initial
wall offset in that same frame. Then it rotates with the ligand and does not
flip toward the metal. Preserve independent DOFs and existing wall; no new
term, width, stiffness or fitted parameter. Retest full NUWFAM pools/seeds.

The earlier full-suite attempt is invalid as a final gate: while it was
running, restored experimental source caused late imports to mix old BendPlan
and new composed-FK classes (four missing-attribute failures). Record all five
failures (including the original warning isolation failure), and rerun only
once final source is stable. Production golden had already passed 245/245
for both profiles before these changes; final changes still need its gate.

Correction: crystal atom mapping in the pool manifest needs inversion
(argsort) to get builder order. Prior tail crystal-penalty conclusions are
withdrawn; corrected best-assignment deltas are KIXVUJ -0.004, WEYJIT -372.26,
FEXWUC 0.0. Tail remains under investigation (WEYJIT needs geometry/key
inspection). Correct NUWFAM C1 crystal angle sum is 332.15 degrees, matching
original capped pyramidalisation; do not treat pyramidal C1 as a defect.

H6 eliminates severe coincidence (pool min angles 56.7/56.9/56.4) but top-1
216/526/339 remains bad. The existing wall must measure the actual geometric
lean, not a rotation from a covalent-plane embed: its ranges were derived
from metal-facing crystal leans. H7 prediction: for exchangeable geminal
eta2 substituents, measure that existing wall on their **unwrapped polar
angles in the metal/face plane**, assigning the unordered pair to opposite
sides by transverse order. The same two wall slots, stiffnesses, widths and
independent DOFs remain; no added term or prior. The atan2 measurement sees a
side crossing that sine alone cannot. Check crystal reproduction explicitly
and test NUWFAM before any other development run.

H7 closes the severe geometry defect: min H–C1–H angles 85.2/87.1/80.9,
no near-coincident H. Top-1 still 315/342/256 on the redesigned Li embed.
H8 prediction: use the new geometric wall with the **original** Li embed,
which matches the crystal C1 pyramidalisation and retained NUWFAM's floor
wins. Normal restraint/conjugated extension altered valid geometry and may
have discarded the original cap benefit; isolate these effects. Preserve
Li-placement attempt separately, restore engine cap baseline, change only
the existing wall's geometric measurement. Test NUWFAM and six cap movers.

H8: original cap + fully geometric wall still reshuffles optimization,
NUWFAM top-1 deltas +84/+51/+101 although floors improve -64/-57/-6.
Original cap top-1 rows have valid opposite transverse sides and polar leans
only 23–34 degrees. Thus replacing every wall coordinate changes valid
search dynamics unnecessarily. H9 prediction: preserve the original wall
coordinate on the correct side; unwrap it to the polar angle only upon a
side crossing. This is the minimal closure of the existing wall's loophole,
leaving its old action intact everywhere that has not crossed. No new term,
prior, parameter, DOF merge or freezing. Test NUWFAM before larger runs.

Isolation root cause confirmed with an earlier polluter: XECSON audit
`test_fk_joints_well_defined_at_linear_atoms` invokes `get_case` ->
`tools.jdev.env.use_src`, which sets jacobik logger ERROR. That single test
followed by validity tests reproduces 1 failure/2 passes in 12.20 s. The
worker test was another polluter, not the earliest one. Restore jacobik logger
level via a small autouse test fixture (`setLevel`, which also clears logging
caches), rather than treating only one caller. Ordered retest pending.

H9 side-crossing-only guard prevents the worst geometry but fails energy
guard (NUWFAM 439/226/376 vs control 195/286/225). Preserve wall attempts
and revert them from active source. They do not meet the intended benefit.

### H10: use the existing joint face statistic for geminal faces

The scorer already falls back to fitted face-mean S_orient tables whenever
per-substituent tables are unavailable. Independent geminal marginals permit
both branches to select the same mixture mode. Prediction: choose that
existing face-mean mechanism when a face has multiple substituents on one
pivot (under the cap flag). This reduces scoring complexity, preserves all
DOFs, adds no term/prior/refit, and retains the same total face weight. Use
original cap and lean wall to isolate it. NUWFAM seeds 11–13 first.

H10 face-mean alone: NUWFAM top-1 182/212/518; floors 174/169/145;
severe pool collapse remains (min 5/20/13 degrees). H11 prediction: combine
the existing face-mean statistic with the actual geometric wall measurement.
The former fixes independent-mode preference; the latter prevents its
remaining side ambiguity. Both use existing score/prior mechanisms, keep
DOFs, and introduce no fitted parameters. Original cap embed retained.

H11 promising: NUWFAM top-1 230.31/157.57/208.88 vs local controls
194.55/286.23/225.02 (mean delta -36.34); floors 127.31/157.57/187.34
vs 194.55/217.34/187.94. Pool min H–C1–H 91.47/90.18/88.36 degrees;
all severe collapse eliminated. Seed 11 still +35.76; no repeated >25 guard.
Keep both existing-mechanism corrections provisionally. Next prediction:
the joint statistic also addresses multi-branch haptic face ambiguity on
other cap movers; test the frozen 34-parent dev set (cap alone), without
altering scales or thresholds. Estimated ~10 minutes from completed control.
No full panels until dev gates and stable-source verification.

H11 broad eligibility fails on development BOHWUT (+85/+23/+59) and EPUWAQ
(+5/+61/+98), losing the original cap win. Stopped this superseded arm by
its own PID 862920 (SIGINT intercepted by RDKit, then SIGTERM; exit 143).
Partial records are retained, not counted as complete. The joint fallback
must apply to **geminal H**, whose exchangeable independent motions cause
the defect; heavy ring branches are already kinematically connected and
must retain per-sub scoring. Narrowed by this chemical/topological rule,
not by parent identity. NUWFAM eligibility is unchanged; EPUWAQ's heavy
branches return to original scoring. Investigate SOBHAS separately: it has
three eta2 carboxylate faces and a Cp* face, no geminal-H face, so its cap
regression has a different cause. Next diagnostic decomposes top-1 MACE
energy into isolated-ligand strain and interaction residual on saved pools.

MACE strain decomposition (eV converted by 96.4853321233; corrected raw
helper units before interpreting): SOBHAS cap-minus-control total deltas
+238.7/+560.6/+44.7 comprise ligand-strain +9.3/+4.0/+3.4 and interaction
+229.4/+556.6/+41.3. Thus this is mostly packing/selection, not a gross
covalent embed failure. BOHWUT strain deltas +15.8/-83.9/+28.0; EBAYIQ
+5.8/+12.5/+18.5. Reference/control rerun SOBHAS agrees within .002 kJ/mol.

### H12: cap only charge components that need an ionic surrogate

Prediction fixed before run: aromatic monoanions already have a delocalized,
valence-complete pi geometry; a counter-ion is unnecessary there and
perturbs packing. Skip haptic Li caps when every donor-face atom is aromatic
and its whole conjugated component has net formal charge -1. Preserve caps
for nonaromatic localized anions and multiply charged pi components (e.g.
EPUWAQ's fused dianion). No parent labels or thresholds; derive component by
RDKit conjugation and formal charges. Keep original placement, wall +
geminal-H joint statistic. Test NUWFAM, EPUWAQ, QUTWOT, BOHWUT and SOBHAS;
then the fixed development set if promising. This does not yet change the
size of any capped group, so isolates scope from the earlier placement rule.

H12 intended scope was not active on actual benchmark embeds: incoming
RDKit molecules have cleared aromatic flags (Kekule form), so direct
GetIsAromatic misses them. SOBHAS remains +239/+557/+45; this is a scope
implementation defect, not evidence about the proposed rule. Restore
aromatic perception on a copy before deciding scope (next isolated test).

H13 first: QUTWOT has a true eta3 allyl face, but the geometric wall repair
only covered eta2. Joint scoring without its corresponding wall gave
QUTWOT +48/-122/+96 (guard fail). Extend that same wall measurement to
larger faces using their oriented plane normal and in-plane radial/tangent
at the pivot. Same DOFs, slots, widths and stiffnesses. Predict removal of
the eta3 version of the degeneracy and recovery of QUTWOT; test QUTWOT alone
before combining the aromatic-perception correction.

H13 success: QUTWOT top-1 paired deltas -145.87/-158.36/-69.69, recovering
the cap benefit. Its crystal geometric wall is exactly zero. Eta2/eta3
side-crossing tests: 5 passed. Keep the general face-size repair.

H14: correct H12's implementation by perceiving aromaticity/conjugation on
a molecule copy before applying its fixed scope rule. Unit tests now cover
both aromatic and cleared Kekule input forms, preserving localized and
multiply charged caps. Predict SOBHAS and EBAYIQ return to control while
EPUWAQ retains its benefit and NUWFAM/QUTWOT keep their anion caps. Test the
seven cap movers plus QUTWOT; no parameter adjustments or parent conditions.

H14 confirms skipping aromatic monoanions is too broad: it removes Cp–H
caps needed for NUWFAM (+156/+55/+281). EPUWAQ retains -48/-48/-51;
EBAYIQ approximately returns to control. Refine the fixed scope rule to
skip only aromatic monoanions **without face-bound hydrogens**. The H-bearing
faces still need their substituent geometry represented; fully substituted
aromatic monoanions are the class whose cap perturbations failed here.
This is a structural class rule, not a parent label or numeric energy cutoff.

H14 also exposed eta3 replay fallback: Python face-list indexing caused an
unpinned CPU-to-CUDA index copy during capture. Store face indices as device
LongTensors during FK setup and index_select during the solve. No math
change; required replay gate still pending. Keep all fallback records.

H15 prediction: refined scope preserves NUWFAM/QUTWOT gains, removes the
fully substituted aromatic monoanion regressions, and keeps EPUWAQ's dianion
win. Test all 34 development parents under stable source, with parent guards,
geometry and explicit replay-fallback counts. Estimated 10 minutes from the
completed same-PC control; no full panel yet.

H15 completed: 102/102 dev records valid, zero replay fallbacks, no repeated
>25 guards. Biased development-set mean -5.35, clustered CI [-14.39,+0.77]
(exploratory, not confirmation). NUWFAM +35/-129/-16; EPUWAQ -55/-44/-49;
QUTWOT -146/-158/-70. SOBHAS/EBAYIQ/XEVDAE approximately control. Remaining
small consistent losers: HOCCUA +15/16/14, UMUBAF +22/9/0; BOHWUT floors
still regress. Do not claim all cases are solved.

Stable c7a7123 verification: full CPU suite **406 passed,20 skipped,3 xfailed**
in 745.56 s. Both production/u1 golden **245 arrays,0 differ**. Initial
200-step CUDA stress parity fails (max raw relative score FEKKAJ .0374,
NUWFAM .867,QUTWOT .00427); record and diagnose identical starts and GPU
nondeterminism before interpreting this as a replay implementation defect.

### H16: complete the existing local-charge and cap-placement semantics

BOHWUT's negative carbon adjoins P+: the donor is a locally neutral ylide.
The existing production kappa rule excludes local zwitterions, but the new
haptic path checked only the face charge. Apply the same local-neighbour
charge rule to haptic caps. Predict BOHWUT returns to control (no new score
changes). HOCCUA's fully substituted Cp face has no H, but its conjugated
phenyl substituent does; H15 counted the whole component's H rather than
**face-bound H**. Correct that scope implementation to the declared rule.
For heteroatom-localized charge (UMUBAF's thiolate), reuse the existing kappa
lone-pair placement rather than put Li above a C–S edge. No new parameter,
score term or complex identifier. Predict improved heteroanion embedding;
test BOHWUT/HOCCUA/UMUBAF and preserve all outcomes.

H16 results: BOHWUT -1.00/-.002/+.56 (noise-scale, previous floor defect
removed); HOCCUA approximately zero; UMUBAF -18.62/-19.49/+4.80, replacing
its consistent loss with a mean gain. Keep the corrected local-charge gate,
face-H scope and existing kappa placement for heteroatom charge.

Added a finite-gradient fallback to the original wall for an undefined
face/polar angle (degenerate or axial geometry), preserving generator
robustness. This changes no defined-face math. Final local-charge/sigma
placement tests include neutral ylide and CO, plus thiolate placement at the
existing 1.9-A kappa distance. Numerical/geometry tests cover side exchange,
rigid rotation, independent columns, eta2/eta3 and degenerate faces.

Tail status: drop the original fixed-two-scale quadratic design. Corrected
crystal evaluation leaves WEYJIT Q2 worsened by 372.26 even under its best
candidate assignment, while local tail alone gives +21.7/+9.2/+193.3 and
KIXVUJ +8.5/+43.3/+55.0. This is a calibration flaw in hardening empirical
t-mixture tails, not pure tie noise. FEXWUC's tail gains are sacrificed;
no threshold is tuned on these outcomes. Donor scales remain the original
crystal-derived hh .95 / ho .90 / oo .80.

Replay diagnosis on final flags, 200 steps, actual FAST_FK path: enabling
deterministic CUDA algorithms + CUBLAS_WORKSPACE_CONFIG=:4096:8 gives
**exact equality** on FEKKAJ/NUWFAM/QUTWOT. Initial theta/reference/scorer
hashes match, eager-repeat and eager-replay max raw relative differences
are all 0; returned rows/scores identical; 198 steps replayed, no errors.
This validates graph semantics without changing the 1e-6 tolerance. Retain
the failed ordinary-math stress test and run an ordinary eager-repeat noise
comparison to classify its differences. Production confirmation will keep
its normal CUDA backend, paired for every arm on the same PC.

Next: combined donor+cap development arm on the fixed 34 parents (same
controls), predicted to retain FEKKAJ repulsion benefit and the repaired cap
wins without a repeated >25 guard. Estimated 10 minutes; no new tuning
parameters. Final source 63d2bb2 is fixed during renewed full CPU/golden
checks (new local-charge and finite-gradient cases justify this rerun).

Final 63d2bb2 checks: **410 passed,20 skipped,3 xfailed** in 726.94 s;
production and u1 golden **245/245,0 differ**; canonical replay regeneration
matches checked-in output exactly. Final 34-parent crystal-wall check gives
zero additional wall penalty for every parent, so no class refit was needed.

Ordinary CUDA repeated eager solves have identical starts but differing
raw endpoints (NUWFAM max .0483 vs eager/replay .0468; QUTWOT .00697 vs
.00634). Deterministic CUDA is exactly equal for all three units. Preserve
ordinary strict-parity failures; classify the observed level as backend
numerical/run variation, not proof of a graph-semantic change.

Combined development final source: 102/102 valid, zero replay fallbacks,
no repeated >25 guards; biased-dev mean -13.79, clustered CI [-33.24,-1.24].
FEKKAJ -279/-323/-251, NUWFAM +35/-130/-16, QUTWOT -67/-127/-64,
EPUWAQ -55/-44/-49. No confirmation outcomes opened yet.

Completed control/cap/combined pilots sum to 545.0/543.9/516.9 seconds across
34 parents. Four arms x 130 parent-panel units gives ~2.3 hours at the
control pilot rate, with a conservative 2–4 hour allowance; below 10 h.
The frozen four-arm contract/launcher and analyzer zero-bound smoke check
are ready. Start full cal+tail confirmation only after this contract commit.


## 2026-10-31 — Checkpoint H-3b, final confirmation and audit handoff

Frozen source/contract 49814f8 (implementation 63d2bb2), four serial arms,
local RTX 2060 SUPER. All phases exit 0 / ALLDONE: 1,560 records, 1,536 valid,
24 preregistered NEPNEF failures; no unexpected failures, missing pairs,
metadata violations, nonfinite energies or replay fallbacks. Wall 5,822.4 s
(97.04 min), excluding development/verification. Colab spend 0.

Cal primary combined −4.148 [−10.024, −0.402]; cap −1.877 [−4.549, −0.128];
donor −2.815 [−8.164, +0.228] kJ/mol, parent-clustered CI. All candidate
cal/tail repeated-loss guards pass. Combined/cap benefit criteria PASS;
donor standalone benefit remains unproven. Package primary passes but full
directive goal NOT MET. Tail means/CIs, named seeds, floor, coverage, score
correlations, timing and complete failures are in H3B_REPORT.md and raw final/.
No confirmation-driven tuning. FEXWUC tail win remains sacrificed.

Owner's mode-lock proposal: read-only reconstruction of old NUWFAM starts
by variant/assignment/trial (no old initial-state hashes). 17/96 collapsed
rows; 15/17 began with the same MAP component, all ended (0,0), six retained
the same start/end pair. Valid crystal also (0,0), HCH 108.63°, posterior
.974/.965. These components are not geometric side labels. This does NOT
constitute a locked-component optimization ablation; transient switches are
unobserved, so no categorical claim it cannot help. Reproducer and raw JSON
retained. No production change followed this diagnostic.

Audit qualifications supersede earlier overly strong wording: face-mean is
an aggregate statistic, not a full joint distribution; independent DOFs are
retained, but the effective wall prior and SO factorization change. Existing
asin-derived wall widths were reused with atan2 without training-wide
calibration. Zero extra penalty on 34 references does not prove no refit
needed. Pair detection/sorting/frame logic adds complexity; literal directive
compliance is unresolved. Aromatic face-H scope avoids some capped failures
by withholding caps, rather than repairing those geometries.

Ordinary strict CUDA parity remains FAILED. FEKKAJ eager-repeat raw error
.04108 versus eager/replay .26943, exceeding its measured repeat variation;
NUWFAM .04829/.04679, QUTWOT .006971/.006339. Thus the earlier blanket noise
characterization is withdrawn. Deterministic CUDA parity remains exactly 0
on all three units, identical returned rows, 198 graph steps, no errors.

The pool-term diagnostic initially used nonexistent q2_angles (0 rows,
15 errors). Corrected to q2_dmd_angles and installed embedding-validity
checks. Successful replacement: 960 rows per arm, zero failures; top-1
score terms are archived in H3B_DEV_TOP1_TERMS.csv. Diagnostic fixes and
reporting only after frozen confirmation; model code unchanged.

Final report H3B_REPORT.md, parent CSV, failure ledger and reproducer prepared.
STOP at Checkpoint H-3b. No merge, promotion or default change; wait for audit.
