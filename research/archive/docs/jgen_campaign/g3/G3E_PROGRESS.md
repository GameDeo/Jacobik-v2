# G-3e progress

## 2026-11-02 — Amendment 5 and paired screen setup

Owner/auditor Addendum 4 implemented as amendment 5, committed 32de184 before
any G-3e result. IA-k uses existing finish_phase: iRprop+ search, switch 100,
release symmetry only; keep shape/projection through budget 250. Thermal
widths unchanged. No new arithmetic. Fingerprint IA-k 5f74b14b57ed; C1
273f2ccbf892. First G-3d screen and failures preserved, not replaced.

Fresh paired AJAQAD profile (C1+IA-k, seeds 11–13, same GPU invocation) passed;
all six records valid, replay mode, no fallback. Parent wall 18.9 s; job
32.418 s. Fresh profile reused as one fixed clean24 parent, original artifacts
preserved. Frozen remaining 23 parents, hashes and 20–30 min planning estimate
in g/G3E_CLEAN24_INPUTS.json before the panel-scale start. Estimate uses the
previous same-cohort three-arm cost scaled to two arms, with contention margin.
No energy outcome used to pick cohort or tune flags. Unchanged advance/angle
predicates implemented in g3e_filter.py; no prior candidate is rescued.

Parallel CPU twist diagnostic: 10 identities frozen in g/G3E_TWIST_ROWS.json,
highest-score >45-degree octahedral-start rows on the specified three parents/
seeds, plus OMAGAK (next parent by count). Same-assignment untwisted partners
use the frozen C1/R2s/IA unit union; R2s-only HEDJOP lacks such partners.
Nearest is fixed-order heavy-atom Kabsch RMSD. Full production FK coordinate
recovery must meet 1e-5 A and 1e-4 score reproduction; failures ledgered without
substitution. All choices frozen before gradient/path results. No score change
or extra arm. R2s remains available, off, for the hypothesis-layer work.

Local RTX 2060 SUPER only, runkit CodexB GPU lock and nvidia-smi before starts;
full-pool xz exports, same-invocation C1 pairing, 250 steps. No Colab; spend 0.
Long waits use completion or 20-minute timers as requested.

## 2026-11-02 — Clean24 decision: IA-k advances

144/144 valid records, 72/72 finite paired units, 24 complete parents. No
fallback or missing/failure units. IA-k registered predicates:
- median top-1 delta -4.441607 kJ/mol: PASS;
- mean floor delta -19.203932 kJ/mol: PASS;
- pooled-row max-Q2-deviation p95 37.600735° vs C1 35.170910° (+2.429825°),
  below the 45.170910° limit: PASS.
Advanced arm IA-k, fingerprint 5f74b14b57ed. No predicate/guard change.
Mean top-1 -20.593958 [parent CI -43.680578,-3.240694], mean floor
-19.203932 [-39.903646,-4.830668] (10,000 parent resamples, seed 11).
Exact decision in g/G3E_FILTER_RESULT.json. Cal/tail performance profile and
whole-step estimate follow before bulk execution; stop if estimate >8 h.

Original GPU-row DOF inversion attempts failed score/state verification and
remain in shared g3e/twist/. Owner explicitly approved regeneration of CPU
rows. Protocol frozen at 4912980 in g/G3E_CPU_TWIST_PROTOCOL.json before new
CPU results. Same ten units, native captured DOFs/references/commitments,
separately labelled CPU diagnostic; no original screen row is substituted.

## 2026-11-02 — Recovery after owner-reported PC crash

No task processes survived. Completed clean24 collection remains intact
(72 C1 + 72 IA-k records); advance at f7aff4b unchanged. The first step-5
cal profile had started but wrote no records or completion metadata; retain
its log and restart the paired profile into profile_cal_restart1. Bulk cal
and tail had not started. GPU checked after reboot: same 2060 SUPER/driver,
no compute process. No crash cause inferred or scientific failure tuned away.

Native CPU pilot AZUSIA seed 11 survived: complete C1/R2s captures, valid NPZ
arrays and selected CPU row IDs. It took 74.134 s and will not be re-solved.
Pilot native selection/state hashes frozen in G3E_CPU_NATIVE_SELECTIONS.json
before path measurements. Native CPU regeneration is separate from the
original GPU-row recovery failures. Captured references/DOFs/Q3 commitments
persist before gradients; same ten units and prior protocol remain fixed.

## 2026-11-02 — Step-5 estimate frozen; start cal then tail

Restarted AJAQAD cal profile passed: parent 35.0 s, full cold-start job
111.476 s. Both C1/IA-k, seeds 11–13, same invocation, no fallback. Reuse
these fresh same-stage records as the first cal parent; clean24 screening
records are not reused for cal/tail judgement. Cal contains 120 parents,
tail 10; five parents overlap and each panel is still measured separately.

Frozen estimate in g/G3E_STEP5_PLAN.json: 2 × max(fresh cal profile parent wall,
whole G-3e clean24 job wall /24) ×130 parents, plus two cold-start overheads.
Estimate 3.040578 h, below the 8 h stop limit: START. Safety factor two covers
heterogeneous parent costs and CPU diagnostic contention. Shared full-pool xz
exports retained. No cal energy outcomes opened to select a configuration.

Serial GPU controller resumes only complete paired parents; preserves partial
interrupted artifacts before rerunning that whole paired parent. It refuses
automatic retry after a directive fallback stop. Cal then tail, C1+IA-k in
each invocation. All per-arm flags reset before the next panel's metadata.
No new score/solver arithmetic. CPU diagnostics overlap within the 5-process
allocation. Native CPU checkpoints now write/rename atomically on disk.

## 2026-11-02 — Native CPU selection complete

All ten fixed units produced a native R2s twisted row and same-assignment
untwisted CPU partner; no case substituted. First pilot retained, remaining
nine generation units completed in 383.355 s across three CPU workers; all
worker exits 0. Native selection IDs and numeric-state hashes frozen in
G3E_CPU_NATIVE_SELECTIONS.json before remaining path measurements.

Native AZUSIA seed-11 pilot path succeeded. Captured-state reconstruction
RMSD 4.755e-6 A and returned-score error 3.992e-8 meet the strict verification
limits. Original GPU-row coordinate-only recovery failures remain separately
preserved; their gradients/barriers were not accepted. Native paths use actual
captured DOFs, bend prior and Q3 commitments, with fixed starting-shell keys.
Scores/gradients reported in native units and summed nats via explicit Z.
Atomic CPU state/path checkpoints and preserved completed/failure records
support crash recovery; no finished scientific failure is silently retried.

## 2026-11-02 — CPU twist diagnostic results retained while GPU runs

Native gradients verified for all 10 fixed units, actual score errors <=1e-4
and reconstructed coordinates <=1e-5 A. Seven complete prescribed 21-point
paths plus all requested MACE points; all seven have interior score minima.
Three USAPOU common-reference paths fail endpoint reproduction (1.995–2.300 A)
because the native endpoints use different sampled macrocycle references.
Own-reference global-rotation checks are ~5e-15–8e-15 A, so the rotation
implementation is verified. No alternative partners or paths substituted.
USAPOU gradients and both endpoint MACE values are still complete; no path
minimum or barrier is fabricated. Path failures remain on record.

Six native twisted states meet the existing remaining-gain criterion;
USAPOU's three and OMAGAK do not. Straight-path barriers for the seven valid
paths span 11.27–1207.17 summed nats, with explicit normalization factors and
Q1–Q7/bend/seam contributions. Q1/Q6 dominate several; Q5 dominates HEDJOP;
Q3 dominates AZUSIA seed 13. CPU MACE favors the selected twisted endpoint
over its proximity-selected untwisted partner in 8/10 pairs. Thus these data
do not establish that all twists are spurious maxima; paths are not optimized
transition paths and no full-dimensional Hessian certificate was computed.
No score change or new arm. R2s remains off and reserved for hypothesis work.

Separate report G3E_R2S_TWIST_REPORT.md, full native summary/hashes in
g/G3E_TWIST_RESULT.json and inspected score-path PNG. Original GPU inversion
failures are preserved separately. Cal/tail execution continues; no interim
cal energy judgement was used to tune anything. Colab spend remains zero.

## 2026-11-02 — G-3e cal/tail checkpoint: acceptance FAILED

The active GPU controller survived the latest owner interruption; no duplicate
job was started. Cal and tail completed with 720/720 and 60/60 raw records.
Primary cohorts have all 354/354 and 27/27 paired seed units (118 and nine
parents). Cal job 3663.881 s, tail 484.705 s, controller 4150.399 s; fresh
cal pilot 111.476 s additionally. No nonexempt embed or replay fallback stop.

IA-k minus C1 top-1 means and parent-clustered 95% CIs, kJ/mol:
cal -12.092 [-20.276, -5.442]; tail -49.409 [-135.032, 24.781].
Floor: cal -10.403 [-17.783, -4.659]; tail -61.563 [-138.636, 5.195].
Cal primary and floor gates pass. Tail primary fails; its obscene-fraction
improvement passes. Both panels FAIL the parent regression guard and solver
wall guard: wall ratios 1.473 and 1.369 versus maximum 1.3.
Cal regressions: AWONOR, CAXTAX, FELSAU, NUWFAM; tail: SOBHAS, YUCZUS.
No reruns or tuning follow the failures. No production arm choice inferred.

NEPNEF cal has six failed records. Tail's six raw records said ok because
the same-process engine embed cache retained cal's known invalid fallback,
while the newly installed validity handler lost the prior failure keys.
This instrument limitation is disclosed, not silently accepted: the analysis
view marks these six invalid_embed and embed_fallback=true, retains raw
energies as invalid diagnostics, and excludes them from finite valid metrics.
Raw exports remain untouched. Primary exclusions were fixed beforehand.
Other tail/cal overlap parents had no recorded embedding failures in cal.

Checks: all 780 raw record identities/arm flags/fingerprints verified; the
six named-selector cal pilot records match their pre-bulk frozen hashes.
All records agree on DB SHA d87b81a5d6297e0251a881b871f477678ede79fc2bce694b72c4dba1c5fe10c4
and MACE model SHA e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510,
CUDA device and RTX 2060 SUPER. Full-pool xz files: cal 714, tail 60 (six
tail NEPNEF exports retained but invalid). Default fingerprint rechecked:
273f2ccbf892. Analysis compiled/run; JSON completeness assertions and git
diff whitespace check pass. Prior full golden/replay/tests remain on record;
no production arithmetic changed during this analysis. Fixed-budget wall is
a disclosed row-step scaling estimate, not a measured rerun.

Main report G3E_REPORT.md, complete result g/G3E_STEP5_RESULT.json. Separate
CPU report remains G3E_R2S_TWIST_REPORT.md: ten gradients, seven verified paths,
three USAPOU common-reference failures; no universal spurious-maxima claim.
All owned GPU/CPU jobs finished. No Colab; spend zero. L4 confirmation remains
a later owner step before any fold-in. Commit and STOP for directive step-6 audit.
