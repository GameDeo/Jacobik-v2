# G-3e checkpoint — IA-k cal/tail and CPU twist diagnostic

**The registered acceptance package does not pass.** No tuning or rescue rerun follows these outcomes.
The owner/auditor chooses the next action. Production defaults remain
unchanged; L4 confirmation is still required before any fold-in.

## IA-k energy results

Candidate minus paired C1, kJ/mol; means and 95% parent-clustered CIs
(10,000 percentile resamples, seed 11). Parent means use all three
declared paired seed units. Primary excludes FEKKAJ and NEPNEF only.

| Panel | Primary parents / declared | Mean top-1 ΔE [95% CI] | Mean floor ΔE [95% CI] | Obscene-fraction change, pp [95% CI] |
|---|---:|---|---|---|
| cal | 118/118 | -12.092 [-20.276, -5.442] | -10.403 [-17.783, -4.659] | -1.690 [-4.132, 0.455] |
| tail | 9/9 | -49.409 [-135.032, 24.781] | -61.563 [-138.636, 5.195] | -7.923 [-17.439, -0.773] |

## Registered gates

| Panel | Primary CI upper <2 | Floor CI upper <0 | Obscene CI upper <0 | No >25 regression on >=2 seeds | Wall <=1.3× | Overall |
|---|---|---|---|---|---|---|
| cal | PASS | PASS | FAIL | FAIL | FAIL (1.473×) | FAIL |
| tail | FAIL | FAIL | PASS | FAIL | FAIL (1.369×) | FAIL |

Secondary improvement requires floor **or** obscene-fraction CI excluding
zero in the improving direction. Wall uses measured solver+assembly
counted once per parent, excluding shared preparation/MACE. Incomplete
data cannot establish a full-panel pass; failed/missing units remain
in the declared ledger. No failure is removed to improve a gate.

The contract regression guard is reported on all primary parents. The
plan’s control-nonobscene stratum is also disclosed separately; it does
not replace or waive a failed contract guard. Control strata below use
the three-seed parent mean at the 200 kJ/mol obscene threshold.

### Regression failures

**cal:**
- AWONOR: deltas [62.962978954426944, 20.829525783658028, 35.58558173943311]; control-mean nonobscene=True.
- CAXTAX: deltas [-9.089891108684242, 32.937990167178214, 43.95759086869657]; control-mean nonobscene=True.
- FELSAU: deltas [28.16646318230778, 26.8933152416721, 38.3537270585075]; control-mean nonobscene=True.
- NUWFAM: deltas [38.38992173690349, 34.46335527393967, -70.03183478768915]; control-mean nonobscene=True.
**tail:**
- SOBHAS: deltas [152.45625967253, 16.00747159589082, 54.47892677783966]; control-mean nonobscene=False.
- YUCZUS: deltas [140.94855584669858, 9.032180356793106, 70.12402019649744]; control-mean nonobscene=False.

### Control quality strata

| Panel / control stratum | Complete parents | Mean top-1 ΔE [95% CI] |
|---|---:|---|
| cal / control_seed_mean_not_obscene | 108 | -6.318 [-11.012, -2.129] |
| cal / control_seed_mean_obscene | 10 | -74.451 [-143.516, -19.273] |
| tail / control_seed_mean_not_obscene | 1 | -0.282 [-0.282, -0.282] |
| tail / control_seed_mean_obscene | 8 | -55.550 [-150.906, 27.913] |

## Denominators and secondary diagnostics

| Panel / arm | Records | Valid units | Returned rows | Angle p95 | Full-pool hits, units / parents | Solver+assembly s | Fixed-budget wall estimate s |
|---|---:|---:|---:|---:|---|---:|---:|
| cal / C1 | 720/720 total | 357/360 | 11981 | 27.588° | 234/360 units; 82/120 parents | 912.30 | 912.30 |
| cal / IA-k | 720/720 total | 357/360 | 11859 | 30.032° | 245/360 units; 86/120 parents | 1343.49 | 1371.98 |
| tail / C1 | 60/60 total | 27/30 | 1477 | 29.110° | 6/30 units; 2/10 parents | 133.75 | 133.75 |
| tail / IA-k | 60/60 total | 27/30 | 1417 | 32.983° | 6/30 units; 2/10 parents | 183.11 | 183.11 |

Fixed-budget wall is explicitly a **linear row-step scaling estimate**,
not a separate measured run, and is not used in gates. It scales measured
solver+assembly by recorded B×T / B×used-steps per parent; assembly/setup
does not scale with steps, so this approximation can overestimate.
Raw recorded step counts, row-step totals, class changes, obscene rows,
score/coverage/class paired CIs and capped-automorphism counts are in
G3E_STEP5_RESULT.json. Parent coverage is any full-pool 1 Å heavy-atom
symmetry-aware reference hit over the three seeds.

### Failure ledger

cal: 6 failed/partial/missing records of 720; full ledger retained in JSON.
- RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
tail: 6 failed/partial/missing records of 60; full ledger retained in JSON.
- Known invalid NEPNEF; same-process cached fallback after cal; raw status ok retained on disk

## Sensitivity and separate cases

| Panel / cohort | Mean top-1 ΔE [95% CI] | Mean floor ΔE [95% CI] |
|---|---|---|
| cal / all included | -11.960 [-20.171, -5.343] | -10.333 [-17.723, -4.649] |
| cal / primary excluding NUWFAM | -12.203 [-20.461, -5.533] | -10.309 [-17.598, -4.624] |
| tail / all included | -49.409 [-135.032, 24.781] | -61.563 [-138.636, 5.195] |
| tail / primary excluding NUWFAM | -49.409 [-135.032, 24.781] | -61.563 [-138.636, 5.195] |

### FEKKAJ

| Panel / seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| cal / 11 | 266.295 / 172.252 | 223.851 / 172.621 |
| cal / 12 | 179.561 / 171.888 | 232.698 / 160.394 |
| cal / 13 | 211.106 / 166.250 | 211.417 / 171.097 |

### NEPNEF

| Panel / seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| cal / 11 | failed (embed fallback exempt) | failed (embed fallback exempt) |
| cal / 12 | failed (embed fallback exempt) | failed (embed fallback exempt) |
| cal / 13 | failed (embed fallback exempt) | failed (embed fallback exempt) |
| tail / 11 | 1961.148 / 1955.320 (invalid cached embed) | 1956.520 / 1956.520 (invalid cached embed) |
| tail / 12 | 1990.905 / 1928.971 (invalid cached embed) | 2005.790 / 1945.233 (invalid cached embed) |
| tail / 13 | 1971.146 / 1948.242 (invalid cached embed) | 1991.542 / 1938.594 (invalid cached embed) |

### XOJHIN

| Panel / seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| cal / 11 | 187.382 / 185.405 | 175.550 / 175.550 |
| cal / 12 | 187.266 / 184.692 | 209.330 / 169.617 |
| cal / 13 | 199.300 / 185.608 | 160.341 / 160.341 |

### NUWFAM

| Panel / seed | C1 top-1 / floor | IA-k top-1 / floor |
|---|---|---|
| cal / 11 | 131.409 / 131.409 | 169.799 / 143.331 |
| cal / 12 | 155.402 / 155.402 | 189.865 / 129.775 |
| cal / 13 | 231.291 / 171.133 | 161.259 / 120.664 |

FEKKAJ and NEPNEF exclusions were fixed before results; NEPNEF fallback
remains exempt and its failed rows are explicit.
Tail NEPNEF raw records say ok because the engine embedding cache survived
the cal-to-tail transition while the validity handler was new. Cal logged
its known fallback. The analysis conservatively marks all six tail NEPNEF
records invalid_embed and embed_fallback=true; raw files and diagnostic
energy values are retained unchanged. These rows contribute no valid
energy/coverage metrics. This disclosed instrumentation limitation does
not affect primary judgement, which excludes NEPNEF in advance.
NUWFAM remains in the
primary cohort; the exclusion sensitivity does not rescue any gate.
XOJHIN remains a search challenge and is not dropped.

## CPU R2s diagnostic

See [the separate CPU report](G3E_R2S_TWIST_REPORT.md): native gradients
for all ten units, seven verified fixed-key score paths and three
retained USAPOU common-reference failures. Six selected states meet the
existing remaining-gain criterion; four do not. Barriers are path-specific,
not full-dimensional maximum certificates. MACE favors the twisted endpoint
in 8/10 selected pairs; the data do not establish all twists as spurious.
Native CPU regeneration was explicitly approved after original GPU-export
DOF recovery failed verification. Both sets remain on record. No score change.

## Execution and checkpoint

Amendment 5 frozen at 32de184, IA-k fingerprint 5f74b14b57ed, C1 production
273f2ccbf892. IA-k fixed switch 100 releases symmetry only, retaining
shape/projection through budget 250. Same bench invocation as C1, same
local RTX 2060 SUPER, seeds 11–13, full-pool xz export. No new arithmetic
was needed for IA-k. G-3d failure and G-3e clean24 results are retained.

Clean24 passed all unchanged advance predicates; full result at f7aff4b.
After the owner-reported PC crash, complete units were reused. The
interrupted cal profile had no records and its log was preserved. Fresh
post-reboot paired profile and frozen 3.040578-hour estimate preceded
bulk cal/tail execution; limit 8 h. Same-stage profile AJAQAD is reused
within cal. Screen data were not reused as cal/tail judgement. Five
parents overlap cal and tail and are measured separately in both panels.

Native CPU DOFs/references/commitments use atomic checkpoints; original
failed coordinate-only inversions are preserved. Run controllers never
retry a directive stop automatically. All input/model/flag/record hashes
are retained in the frozen manifests and result JSON. Parent bootstrap
rules and cohort/gate definitions were not adjusted after outcomes.

No val/reserve/locked sets, relaxation development loops, embed spot fixes,
score changes, production fold-in or Colab usage. Colab spend zero. GPU
lock held, nvidia-smi before each start, all owned jobs finished. L4
confirmation remains a later requirement before any fold-in. Checkpoint:
commit and stop for the auditor; no automatic production selection.

### Recorded job wall times

| Job | Wall seconds | Exit |
|---|---:|---|
| cal/job_attempt1/job.json | 3663.881 | 0 |
| clean24/job/job.json | 963.870 | 0 |
| profile_cal_restart1/job/job.json | 111.476 | 0 |
| profile_clean24/job/job.json | 32.418 | 0 |
| tail/job_attempt1/job.json | 484.705 | 0 |
