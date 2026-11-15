# C-2b development and confirmation report — 2026-11-03

**Once-only confirmation completed after audit clearance.** Development freeze: 5a19ef8. Four finalists were declared:
K (B), P (D), KU (B), S1 (C). Confirmation seeds 21–23 were run once; results follow the development record below.
Code, ledger, machine-readable settings and development evidence are committed
with the contract. The audit addendum cleared confirmation; no production adoption or rerun occurred.
No relaxation, no production fold-in, no Colab use or spend.

Timing caveat (owner): the host/device was heavily contested by concurrent
refit/re-extraction, with CPU utilisation at 100%. Development walls and
ratios are observations under that load, not intrinsic speed measurements.
Pairing does not remove time-varying contention. No speedup claim is made;
confirmation retains its registered wall rule and must report load context.

## Mechanisms

K's seeded-conformer fit has a real pivot bug. FK applies the rigid pose
before internal torsions, rotating about the original donor-atom centroid c.
Kabsch fits the seeded donor-group centroid p, returning translation Qm-p.
Its rotation therefore displaces the seeded centroid unless p=c or R=I.
The flagged correction adds (I-R)(p-c) to the translation. It also handles
unequally sized haptic donor groups: their group centroid differs from the
atom-weighted FK pivot. The regression test fails without compensation and
passes with it. All defaults remain unchanged.

P enables that correction with K's existing orientation draws. F additionally
sets well-posed orientation diversity to zero. Rank-deficient orientation
freedom remains sampled. FU combines F with the existing union beam at share
0.5. These are general rules; none branches on a parent identifier.

Perfect donor placement does not guarantee lower energy: it ignores the
scorer's strain minima and can remove useful orientation exploration.
Development results and dropped leads will be reported without hiding this.

## Carried evidence, reanalysed under the new definitions

S1 size-matched score-vs-energy correlation: delta rho -0.0691,
parent-clustered 95% CI [-0.1006,-0.0397], 115 defined parents. Fixed 20 draws,
seed 11; subsample the larger pool, leave undefined correlations undefined.
The C pool for this correlation is the raw concatenation of its three
exported pools (the C-2 union definition); distinct-row counts separately use
symmetric deduplication. Each wide replicate is matched to that union before
averaging within parent. The inherited top-1 CI still fails non-inferiority:
+0.10 [-3.57,+4.67] kJ/mol. This is not a qualification claim.

After artefact exclusion, inherited K has 2 guard trips, 6 mirrors and null 1;
its targeted count criterion passes, but top-1 CI upper +2.148 misses +2.
U has 2 trips / 5 mirrors / null 1 after excluding artefacts, but no route-A
average gain and top-1 CI upper +4.567. These are seen-data readings only.

T adds rows in 72 pools. None beats the corresponding C energy floor; 26
pools have an added row within 1 A. Compressed XYZ plus manifests cost
16,242,753 -> 17,131,100 bytes, 1.0547x. C-2 generator wall was 0.96x
[0.92,0.98], with no additional IK work introduced; this contested timing is not evidence of a speedup. T stays a
user output-diversity option, default off; no coverage or adoption claim.

## Other diagnoses

UYOVUA's 13-ring contains one 1.5-order edge. MIVXAR and LOPCAW also contain
1.5 and double edges. The large-ring all-single test rejects them, and the
small-ring fallback rejects any aromatic-band edge. Production FK therefore
has zero ring joints. A mixed rigid-segment macrocycle path remains open;
this package does not alter bond interpretation or claim to repair it.
LOPCAW additionally logs a SMILES VF2-alignment failure and heuristic fallback.

NEPNEF: actual xTB input for the failing capped ligand has charge -6, UHF 0.
xTB returns 1, no xtbopt.xyz, with geometry optimisation/SCC nonconvergence.
The second ligand succeeds. The generator returns six conformers in a
two-step CPU smoke, while the benchmark intentionally rejects the invalid
fallback. This is not an exception escaping the generator. Charge/loader
investigation belongs to D; no per-complex embed repair is introduced.

## Validation and infrastructure

Default fingerprint remains 5f74b14b57ed. Full-precision coordinates and scores
are bit-identical to unchanged b0134a2 on BABNAV/MIVXAR/UYOVUA, seed 11,
one worker and no culling. Canonical replay numerical body is unchanged:
only initialisation paths change. Regeneration refreshes the source checksum
after correcting the CPU worker-layout documentation. Focused tests cover the before/after pivot
bug, strict route thresholds, deterministic size matching and layout pinning.

Golden layout: production 245/245 arrays match with configured worker count
8, thread worker mode, and single-thread BLAS/OpenMP. The existing one-worker
run on unchanged b0134a2 failed eight arrays. OJIMIF with the same eight-worker
partitions executed serially also matches X/scores bit-for-bit, isolating the
partition/batch shape from scheduling. The checker now pins this layout,
including inherited-environment overrides; reference files are not changed.

## Panel-wide placement scope

All 120 parent files are present: 133 native E/Z configurations, 132 valid,
195,456 initial rows. The only failure is NEPNEF. These counts agree with the
corrected C-2 scope. Candidate initialisation slots are counted before the
class-budget selection, as in C-2; this is not a count of returned pool rows.

| Placement | Pooled median row-max donor offset (A) | Successful parents with median >0.3 A |
|---|---:|---:|
| C | 3.1526 | 101/119 |
| K | 2.8071 | 101/119 |
| P, seeded fit + original-pivot compensation | 0.3057 | 45/119 |
| F, P with well-posed diversity zero | 0.2486 | 39/119 |

K->P changes at least one row's maximum offset by >0.3 A in 102 parents.
Paired fits reuse the same torsion seeds, bend seeds and orientation draws;
actual FK measures the resulting coordinates. F still consumes the same RNG
draws, preserving later ligands' random streams. Its rank-deficient rotations
are retained. Actual-engine traces independently agree with the reconstruction on seven
single-native-variant parents: 84 quantile vectors, maximum difference 0 A.
LOPCAW is excluded from this reconstruction cross-check because its native
variant scope differs from the original unfixed trace.

Remaining P offsets are not all a fitting bug: internal donor-cloud mismatch
cannot be removed by a rigid fit, and the retained well-posed orientation draw
can deliberately rotate donors away from the best fit. JEGPUJ's first-seed
median remains 3.74 A even with F. A donor docking solve is not introduced.

## Search versus score and the open leads

The inherited U changes the average floor by -2.14 kJ/mol but top-1 by +1.11;
thus the floor-to-top-1 gap grows by about 3.26 kJ/mol. It improves geometric
coverage without consistently ranking the energy improvement first. This is
consistent with the D-package rationale; no scorer or DB change is made here.
The clean KU combination tests seed choice and starting pose together without
also removing well-posed orientation exploration.

On JEGPUJ, C top-1 is 140/141/153 kJ/mol, K 45/57/80, P 83/101/97 and
F 115/95/139 (seeds 11/12/13, rounded). Correcting placement removes part of
K's accidental search advantage, although P still improves this case versus
C. F makes placement tighter yet weakens this energy gain further. These
counterexamples are retained; donor offset is not an adoption metric.

SIVBUV has essentially zero starting donor-target error with P, despite the
C-2 trace showing diverse starts collapsing to two IK rows. A better initial
pose alone is not evidence that its solver-collapse problem is repaired.
The named-case follow-up reports the actual returned rows and energies.

The mixed-bond macrocycle path and donor-only docking remain open. No new
ring flag, per-parent embed repair, new exclusion, scorer rule or DB patch
is hidden in this package. Qualified search still requires the registered
D3 joint gate before production adoption.

## Failure and limitation ledger

- NEPNEF embedding fallback remains invalid for benchmarking. Its source
  SMILES already encodes Zn+14 and neutral N([O-])[O-] groups, yielding two
  charge -7 ligands. The one-Li capped xTB input is charge -6. Preserve this
  upstream source/typing evidence for D; do not change charge to rescue a run.
- QEZVEY in the stratified pilot logged two CUDA graph step-recording OOM
  warnings (steps 1 and 0) and continued eagerly. The controller stopped the
  seen-seed development invocation after 2,488 process seconds (41.47 min)
  under owner-reported severe contention. It retained 54 complete records;
  18 QEZVEY records were not exported and 18 XACMOE records were not reached.
  All 36 are explicitly marked unavailable/not run, with null energies and
  controller provenance. No QEZVEY rerun was made. The raw partial invocation,
  its warnings and abort ledger remain; no wall ratio or efficacy aggregate is
  inferred from this incomplete run. S1's complete C-2 panel evidence remains
  the numerical basis for its proposed confirmation route.
- The initial placement-instrumentation attempt assumed Ligand.smiles and
  failed. Its partial records were retained in placement_attempt1; the fixed
  measurement retry uses the same variants/cases. No numerical variant was
  changed to hide this tooling failure.
- The pilot exceeded the initial 10–20-minute estimate. Current host contention
  and large-pool recovery prevent an intrinsic speed conclusion. This does not
  justify changing the frozen confirmation gates or rerunning a bad outcome.
- The DB-dependent test initially skipped in the data-less worktree; an explicit
  shared-production-DB rerun passed. The broad repository suite is not repeated
  at this development checkpoint; it remains a pre-fold-in package gate.

## Completed paired follow-up

All 84 records are valid (7 parents x 4 arms x 3 seen seeds), with identical
IK row-steps across arms per parent. No pivot correction, replay or solver-OOM
fallback was observed. These cases were selected for diagnosis; their averages
and CIs are not estimates for the full primary population. Timing is contested.

Cells are top-1/floor kJ/mol for seeds 11;12;13. Every energy is unrelaxed.

| Parent | C | K | P | KU |
|---|---|---|---|---|
| JEGPUJ | 140.1/135.7; 140.9/140.9; 152.7/136.3 | 45.3/45.3; 57.5/57.1; 79.6/79.6 | 83.1/83.1; 100.7/100.7; 96.6/96.6 | 55.8/55.8; 45.4/45.4; 50.9/50.9 |
| SIVBUV | 65.2/45.1; 66.4/66.4; 66.5/66.5 | 66.1/66.1; 64.7/64.7; 66.2/66.2 | 64.7/49.4; 65.9/65.9; 64.5/64.5 | 63.4/45.5; 62.3/62.3; 65.8/44.8 |
| FELSAU | 35.0/-2.0; 44.6/1.1; 38.4/1.6 | 20.6/4.9; 18.9/4.0; 19.2/-2.0 | 10.5/2.3; 20.9/-1.0; 11.6/-7.4 | 16.6/5.3; 19.3/4.8; 20.4/5.0 |
| CAXTAX | 71.0/58.2; 89.9/59.3; 71.6/52.6 | 79.2/45.8; 80.3/37.5; 72.4/34.8 | 81.7/61.9; 73.0/58.0; 76.4/67.3 | 71.6/46.9; 70.4/40.0; 76.3/48.5 |
| UYOVUA | 68.9/68.9; 68.9/68.9; 68.8/68.8 | 68.8/68.8; 68.9/68.9; 68.9/68.9 | 68.9/68.9; 68.9/68.9; 68.9/68.9 | 68.8/68.8; 68.9/68.9; 68.8/68.8 |
| MIVXAR | 62.0/62.0; 62.2/62.2; 62.1/62.1 | 62.2/62.2; 62.1/62.1; 62.1/62.1 | 62.2/62.2; 62.0/62.0; 62.1/62.1 | 62.2/62.2; 62.2/62.2; 62.2/62.2 |
| LOPCAW | 135.3/129.4; 134.8/129.5; 135.2/129.5 | 134.0/129.6; 134.0/129.5; 133.8/129.6 | 133.3/129.5; 133.3/129.5; 133.6/129.5 | 133.4/129.9; 134.1/129.6; 133.5/130.0 |

| Returned rows (11/12/13) | C | K | P | KU |
|---|---|---|---|---|
| JEGPUJ | 8/8/8 | 16/16/16 | 2/2/2 | 16/16/16 |
| SIVBUV | 2/2/2 | 1/2/3 | 2/2/2 | 11/4/6 |
| FELSAU | 16/16/16 | 16/16/16 | 16/16/16 | 16/16/16 |
| CAXTAX | 112/112/112 | 112/112/112 | 112/112/112 | 112/112/112 |
| UYOVUA | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |
| MIVXAR | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |
| LOPCAW | 2/2/2 | 2/2/2 | 2/2/2 | 2/2/2 |

KU mitigates SIVBUV collapse: 11/4/6 rows versus C 2/2/2, with seed-13 floor
44.8 versus 66.5. P alone still returns two rows despite essentially perfect
initial donor placement. The union seeds plus K reach additional distinct conformers; top-1
does not always pick the lowest-energy one, leaving the D/ranking question.

FELSAU improves mostly at top-1; its low-energy floor was already present.
P further lowers top-1 and finds a lower floor on seed 13. CAXTAX is mixed:
K and KU lower the floor but do not always rank it first; P docks better yet
raises the floor in two seeds. No embed or scorer change explains these swings:
the changed starts/seeds yield different solved rows and ranking choices.

UYOVUA/MIVXAR/LOPCAW retain 1/1/2-row pools and near-unchanged floors.
They still have no macrocycle joints. Existing LOPCAW heuristic alignment
and topology-infeasible Hungarian-assignment replacement warnings remain.

XACMOE new-P energy results were not reached before the pilot abort. Its
placement trace improves, but that is not an energy result. In complete C-2
data its C floors are 115/110/108 while top-1 is 155/166/108; U floors are
111/111/110 while top-1 is 149/111/148. Much of the arm/seed swing is ranking
among similar floors. K reaches a better floor on seed 11 but loses it on
other seeds. Retain that mixed search/ranking evidence for D.

## Freeze disposition and provenance

Eight configurations were assessed, including carried controls/options and
dropped F/FU. The final four use the routes and exact fingerprints in
c2/C2B_CONTRACT.md and C2B_FINALISTS.json. At the freeze checkpoint no variant was yet declared
qualified. The registered primary confirmation is once-only, full population, seeds
21–23, paired control, with the registered artefact guard and a 10,000-resample parent-clustered
bootstrap. Qualified search must then pass the D3 joint 2x2 before fold-in.

Producer metadata: the initial pilot records d1c6cb2 with the working-tree
development implementation; the follow-up records 6808f28 plus preflight-only
working-tree hardening. Numerical solver/initialisation paths did not change
between those runs and this freeze. Exact freeze source, DB/sidecar and MACE
hashes are recorded in C2B_FINALISTS.json. The generated replay diff refreshes
only the source checksum after CPU-layout documentation, not executable logic.

Compact development evidence is in c2/C2B_DEVELOPMENT.json; raw data, partial
pilot, controller abort and logs remain under the shared C2b campaign directory.
Default CPU X/scores match b0134a2, all reference/input hashes are unchanged,
and both read-only golden profiles match 245/245 arrays. No production default
changed; production fingerprint remains 5f74b14b57ed. No Colab VM or spend.

## Once-only confirmation — completed

One invocation, full frozen population, C/K/P/KU/S1, seeds 21–23. No rerun, relaxation, L4 replicate or production fold-in. Differences are arm minus C; energies kJ/mol, coverage percentage points. Per-parent seed means, 10,000 parent-cluster resamples, seed 11, percentile 95% CIs. Undefined values are not imputed; failures remain in coverage/distinct denominators. Gates use full precision; displayed values are rounded. Coverage arm/control columns are percentages and differences are percentage points; size-matched rho is reported as a paired contrast only.

**Qualified:** K, P, KU. P qualifies as the correct start_kabsch implementation; all adoption still awaits the D3 joint gate.

| Arm / route | Floor difference [CI] | Top-1 difference [CI] | Wall ratio [CI] | Frozen gates | Qualified |
|---|---|---|---|---|---|
| K / B | -0.997 [-3.239, 0.905] | -2.170 [-5.131, 0.349] | 0.947 [0.912, 0.980] | wall PASS, guard PASS, top1 PASS, compute PASS, floor PASS, targeted PASS | True |
| P / D | -1.876 [-5.112, 0.425] | -3.220 [-7.421, 0.159] | 0.951 [0.915, 0.985] | wall PASS, guard PASS, top1 PASS, compute PASS, floor PASS, correctness_proof PASS | True |
| KU / B | -2.062 [-6.242, 1.081] | -3.299 [-8.835, 1.035] | 1.052 [1.004, 1.104] | wall PASS, guard PASS, top1 PASS, compute PASS, floor PASS, targeted PASS | True |
| S1 / C | 0.959 [-0.963, 2.578] | -0.691 [-3.432, 1.663] | 1.001 [0.817, 1.233] | wall PASS, guard PASS, top1 PASS, compute PASS, floor FAIL, coverage PASS, ensemble PASS | False |

### Guard counts

Decision uses the registered artefact-excluded version. Artefacts stay in all other metrics. Each cell lists trips / mirrors / arm null; C rotation null also reported.

| Arm | With artefacts | Without artefacts | C null (with / without) | Decision guard |
|---|---|---|---|---|
| K | 3 / 7 / 1 | 3 / 6 / 1 | 1 / 1 | True |
| P | 0 / 5 / 2 | 0 / 4 / 2 | 1 / 1 | True |
| KU | 3 / 7 / 1 | 2 / 6 / 1 | 1 / 1 | True |
| S1 | 4 / 3 / 3 | 3 / 2 / 3 | 1 / 1 | True |

### K: all registered metrics

| Metric | C | K | Difference [95% CI] | N parents |
|---|---:|---:|---:|---:|
| Floor delta-E (kJ/mol) | 60.670 | 59.673 | -0.997 [-3.239, 0.905] | 118 |
| Top-1 delta-E (kJ/mol) | 76.053 | 73.883 | -2.170 [-5.131, 0.349] | 118 |
| Coverage <=1 A (pp) | 68.644 | 69.492 | 0.847 [-1.130, 2.825] | 118 |
| Coverage <=0.5 A (pp) | 35.593 | 35.028 | -0.565 [-4.237, 3.390] | 118 |
| Minimum heavy RMSD (A) | 0.901 | 0.893 | -0.008 [-0.032, 0.016] | 118 |
| Minimum shell RMSD (A) | 0.136 | 0.139 | 0.003 [-0.003, 0.011] | 118 |
| Shell <=0.3 A (pp) | 92.938 | 92.090 | -0.847 [-2.260, 0.282] | 118 |
| Shell <=0.5 A (pp) | 98.305 | 97.740 | -0.565 [-1.977, 0.565] | 118 |
| Distinct rows | 32.486 | 32.884 | 0.398 [-0.023, 0.850] | 118 |
| Score-energy rho | -0.559 | -0.563 | -0.004 [-0.037, 0.028] | 113 |
| Size-matched rho | not reported | not reported | -0.002 [-0.035, 0.031] | 113 |

Artefact-excluded trips: GOTWET, OMAGAK, WEYJIT; mirrors: AWONOR, JEGPUJ, QEZVEY, USAPOU, XOJHIN, XUKKIA.

### P: all registered metrics

| Metric | C | P | Difference [95% CI] | N parents |
|---|---:|---:|---:|---:|
| Floor delta-E (kJ/mol) | 60.670 | 58.794 | -1.876 [-5.112, 0.425] | 118 |
| Top-1 delta-E (kJ/mol) | 76.053 | 72.833 | -3.220 [-7.421, 0.159] | 118 |
| Coverage <=1 A (pp) | 68.644 | 69.209 | 0.565 [-2.260, 3.390] | 118 |
| Coverage <=0.5 A (pp) | 35.593 | 32.768 | -2.825 [-6.497, 0.565] | 118 |
| Minimum heavy RMSD (A) | 0.901 | 0.906 | 0.005 [-0.014, 0.026] | 118 |
| Minimum shell RMSD (A) | 0.136 | 0.140 | 0.004 [-0.004, 0.014] | 118 |
| Shell <=0.3 A (pp) | 92.938 | 93.503 | 0.565 [-2.260, 3.390] | 118 |
| Shell <=0.5 A (pp) | 98.305 | 98.023 | -0.282 [-1.412, 0.565] | 118 |
| Distinct rows | 32.486 | 32.534 | 0.048 [-0.376, 0.511] | 118 |
| Score-energy rho | -0.573 | -0.564 | 0.009 [-0.025, 0.047] | 112 |
| Size-matched rho | not reported | not reported | 0.005 [-0.029, 0.044] | 112 |

Artefact-excluded trips: none; mirrors: AWONOR, HEDJOP, JEGPUJ, SESWIZ.

### KU: all registered metrics

| Metric | C | KU | Difference [95% CI] | N parents |
|---|---:|---:|---:|---:|
| Floor delta-E (kJ/mol) | 60.670 | 58.609 | -2.062 [-6.242, 1.081] | 118 |
| Top-1 delta-E (kJ/mol) | 76.053 | 72.753 | -3.299 [-8.835, 1.035] | 118 |
| Coverage <=1 A (pp) | 68.644 | 70.621 | 1.977 [-1.130, 5.650] | 118 |
| Coverage <=0.5 A (pp) | 35.593 | 33.051 | -2.542 [-7.062, 1.695] | 118 |
| Minimum heavy RMSD (A) | 0.901 | 0.864 | -0.037 [-0.082, 0.002] | 118 |
| Minimum shell RMSD (A) | 0.136 | 0.138 | 0.002 [-0.005, 0.011] | 118 |
| Shell <=0.3 A (pp) | 92.938 | 92.090 | -0.847 [-2.825, 0.847] | 118 |
| Shell <=0.5 A (pp) | 98.305 | 98.305 | 0.000 [-0.847, 0.847] | 118 |
| Distinct rows | 32.486 | 33.850 | 1.364 [0.777, 2.054] | 118 |
| Score-energy rho | -0.563 | -0.562 | 0.000 [-0.034, 0.035] | 114 |
| Size-matched rho | not reported | not reported | 0.005 [-0.028, 0.039] | 114 |

Artefact-excluded trips: CAQLEP, EDOBUV; mirrors: AWONOR, DAMVAQ, HEDJOP, JEGPUJ, SESWIZ, XUKKIA.

### S1: all registered metrics

| Metric | C | S1 | Difference [95% CI] | N parents |
|---|---:|---:|---:|---:|
| Floor delta-E (kJ/mol) | 56.498 | 57.457 | 0.959 [-0.963, 2.578] | 118 |
| Top-1 delta-E (kJ/mol) | 74.644 | 73.953 | -0.691 [-3.432, 1.663] | 118 |
| Coverage <=1 A (pp) | 72.881 | 73.446 | 0.565 [-2.825, 3.955] | 118 |
| Coverage <=0.5 A (pp) | 43.220 | 39.266 | -3.955 [-7.062, -0.847] | 118 |
| Minimum heavy RMSD (A) | 0.830 | 0.825 | -0.005 [-0.025, 0.014] | 118 |
| Minimum shell RMSD (A) | 0.123 | 0.126 | 0.003 [-0.002, 0.008] | 118 |
| Shell <=0.3 A (pp) | 95.763 | 95.198 | -0.565 [-2.542, 1.130] | 118 |
| Shell <=0.5 A (pp) | 99.153 | 98.588 | -0.565 [-1.412, 0.000] | 118 |
| Distinct rows | 56.508 | 83.280 | 26.771 [20.065, 34.110] | 118 |
| Score-energy rho | -0.574 | -0.640 | -0.067 [-0.105, -0.029] | 114 |
| Size-matched rho | not reported | not reported | -0.069 [-0.108, -0.030] | 114 |

Artefact-excluded trips: QUTWOT, USAPOU, XUKKIA; mirrors: QEZVEY, XOJHIN.

### P versus K and KU versus K — descriptive only

These paired contrasts do not define an adoption route or rescue a failed route. Same primary population, paired seeds and bootstrap; differences are first arm minus K. Size matching uses the fixed 20 draws / seed 11 procedure on corresponding per-seed pools.

#### P_vs_K

| Metric | Difference [95% CI] | N parents |
|---|---:|---:|
| Floor delta-E (kJ/mol) | -0.879 [-3.784, 1.391] | 118 |
| Top-1 delta-E (kJ/mol) | -1.050 [-5.069, 2.273] | 118 |
| Coverage <=1 A (pp) | -0.282 [-3.390, 2.542] | 118 |
| Coverage <=0.5 A (pp) | -2.260 [-5.932, 1.130] | 118 |
| Minimum heavy RMSD (A) | 0.014 [-0.014, 0.041] | 118 |
| Minimum shell RMSD (A) | 0.001 [-0.005, 0.006] | 118 |
| Shell <=0.3 A (pp) | 1.412 [-1.412, 4.237] | 118 |
| Shell <=0.5 A (pp) | 0.282 [0.000, 0.847] | 118 |
| Distinct rows | -0.350 [-0.918, 0.209] | 118 |
| Score-energy rho | 0.005 [-0.019, 0.032] | 113 |
| Size-matched rho | 0.004 [-0.020, 0.030] | 113 |

Observed generator wall ratio versus K: 1.004 [0.998, 1.010].
Descriptive >25 kJ/mol on >=2/3 seeds (with_artifacts): worse 4; better 4.
Descriptive >25 kJ/mol on >=2/3 seeds (without_artifacts): worse 3; better 3.
#### KU_vs_K

| Metric | Difference [95% CI] | N parents |
|---|---:|---:|
| Floor delta-E (kJ/mol) | -1.065 [-4.539, 1.180] | 118 |
| Top-1 delta-E (kJ/mol) | -1.130 [-6.498, 2.817] | 118 |
| Coverage <=1 A (pp) | 1.130 [-1.977, 4.520] | 118 |
| Coverage <=0.5 A (pp) | -1.977 [-5.085, 1.130] | 118 |
| Minimum heavy RMSD (A) | -0.028 [-0.072, 0.007] | 118 |
| Minimum shell RMSD (A) | -0.001 [-0.006, 0.004] | 118 |
| Shell <=0.3 A (pp) | 0.000 [-1.977, 1.695] | 118 |
| Shell <=0.5 A (pp) | 0.565 [0.000, 1.695] | 118 |
| Distinct rows | 0.966 [0.455, 1.571] | 118 |
| Score-energy rho | -0.001 [-0.027, 0.024] | 114 |
| Size-matched rho | 0.002 [-0.023, 0.027] | 114 |

Observed generator wall ratio versus K: 1.110 [1.083, 1.141].
Descriptive >25 kJ/mol on >=2/3 seeds (with_artifacts): worse 4; better 6.
Descriptive >25 kJ/mol on >=2/3 seeds (without_artifacts): worse 4; better 5.

### Named cases

Each cell is top-1 / floor kJ/mol; failed results remain explicit.

| Parent | Seed | C | K | P | KU | S1 |
|---|---:|---|---|---|---|---|
| BABNAV | 21 | 30.287 / 29.706 | 32.707 / 26.610 | 33.842 / 28.795 | 29.978 / 27.136 | 35.517 / 28.426 |
| BABNAV | 22 | 27.425 / 27.425 | 33.621 / 29.600 | 35.246 / 29.588 | 31.731 / 25.567 | 40.510 / 24.764 |
| BABNAV | 23 | 32.889 / 27.421 | 31.879 / 23.505 | 27.342 / 27.342 | 33.915 / 31.706 | 33.620 / 31.178 |
| CIXYUD | 21 | 73.430 / 59.709 | 81.898 / 60.691 | 83.755 / 53.000 | 88.754 / 52.636 | 69.570 / 62.555 |
| CIXYUD | 22 | 73.465 / 65.220 | 87.547 / 69.662 | 66.427 / 59.601 | 79.504 / 57.523 | 45.404 / 45.404 |
| CIXYUD | 23 | 81.769 / 54.270 | 72.983 / 56.163 | 68.924 / 57.457 | 65.314 / 65.314 | 55.738 / 55.738 |
| DOHTOM | 21 | 112.811 / 106.660 | 111.894 / 110.306 | 124.400 / 106.814 | 119.878 / 105.011 | 92.322 / 92.322 |
| DOHTOM | 22 | 121.850 / 111.237 | 138.901 / 111.095 | 133.267 / 106.680 | 128.144 / 106.935 | 120.305 / 105.660 |
| DOHTOM | 23 | 106.329 / 102.908 | 118.399 / 113.474 | 112.399 / 112.399 | 129.117 / 110.973 | 116.737 / 103.434 |
| DUHHIX | 21 | 150.573 / 150.573 | 182.136 / 158.160 | 170.211 / 143.771 | 146.335 / 146.335 | 146.928 / 146.928 |
| DUHHIX | 22 | 164.498 / 163.025 | 169.993 / 145.501 | 155.671 / 155.671 | 162.008 / 152.680 | 154.008 / 140.078 |
| DUHHIX | 23 | 161.488 / 158.299 | 177.325 / 155.604 | 142.570 / 142.570 | 149.591 / 147.442 | 139.762 / 139.762 |
| EFOXII | 21 | 17.575 / 14.968 | 17.575 / 14.968 | 17.575 / 14.968 | 17.575 / 14.968 | 17.257 / 13.443 |
| EFOXII | 22 | 18.103 / 14.721 | 18.103 / 14.721 | 18.103 / 14.721 | 18.103 / 14.721 | 17.105 / 14.173 |
| EFOXII | 23 | 17.493 / 14.705 | 17.493 / 14.705 | 17.493 / 14.705 | 17.493 / 14.705 | 16.921 / 14.761 |
| HOHZAG | 21 | 106.131 / 106.131 | 107.021 / 107.021 | 104.527 / 104.527 | 105.539 / 105.539 | 104.136 / 104.136 |
| HOHZAG | 22 | 106.054 / 106.054 | 105.106 / 105.106 | 105.558 / 105.558 | 105.776 / 105.776 | 105.301 / 105.301 |
| HOHZAG | 23 | 105.287 / 105.287 | 105.637 / 105.637 | 104.428 / 104.428 | 105.935 / 105.935 | 104.504 / 104.504 |
| UYOVUA | 21 | 68.782 / 68.782 | 68.794 / 68.794 | 68.827 / 68.827 | 68.807 / 68.807 | 68.823 / 68.823 |
| UYOVUA | 22 | 68.815 / 68.815 | 68.817 / 68.817 | 68.836 / 68.836 | 68.895 / 68.895 | 68.871 / 68.871 |
| UYOVUA | 23 | 68.803 / 68.803 | 68.921 / 68.921 | 68.800 / 68.800 | 68.841 / 68.841 | 68.860 / 68.860 |
| ZAFPOP | 21 | 41.535 / 20.810 | 42.886 / 22.791 | 41.952 / 21.705 | 31.436 / 23.016 | 41.260 / 21.946 |
| ZAFPOP | 22 | 41.236 / 20.996 | 25.510 / 21.255 | 36.141 / 22.835 | 35.023 / 20.366 | 41.881 / 18.888 |
| ZAFPOP | 23 | 34.123 / 21.177 | 24.194 / 24.194 | 40.099 / 19.849 | 35.327 / 20.680 | 36.717 / 22.277 |
| QAPFAS | 21 | 10.023 / 10.023 | 10.152 / 10.152 | 9.999 / 9.999 | 10.118 / 6.815 | 10.177 / 10.177 |
| QAPFAS | 22 | 10.239 / 10.239 | 10.187 / 10.187 | 10.220 / 10.220 | 10.279 / 6.676 | 10.276 / 10.276 |
| QAPFAS | 23 | 9.571 / 9.571 | 9.469 / 9.469 | 10.153 / 10.153 | 10.232 / 7.556 | 9.599 / 9.599 |
| SIVBUV | 21 | 66.393 / 66.393 | 65.813 / 65.813 | 64.692 / 64.692 | 62.130 / 57.689 | 64.343 / 64.343 |
| SIVBUV | 22 | 66.199 / 66.199 | 66.610 / 66.610 | 64.766 / 64.766 | 63.025 / 63.025 | 65.065 / 65.065 |
| SIVBUV | 23 | 65.924 / 65.924 | 65.811 / 65.811 | 65.088 / 65.088 | 66.353 / 56.000 | 64.526 / 64.526 |
| ZAZREZ | 21 | 69.274 / 68.785 | 67.844 / 67.844 | 68.813 / 68.813 | 69.443 / 67.574 | 69.323 / 68.664 |
| ZAZREZ | 22 | 69.290 / 68.460 | 67.841 / 67.605 | 68.785 / 68.785 | 69.370 / 67.647 | 69.281 / 68.749 |
| ZAZREZ | 23 | 69.355 / 69.355 | 69.336 / 67.602 | 68.951 / 68.951 | 67.853 / 67.663 | 69.352 / 69.352 |
| HOXJUZ | 21 | 281.841 / 267.902 | 283.475 / 269.403 | 279.046 / 267.696 | 280.994 / 266.617 | 286.232 / 269.068 |
| HOXJUZ | 22 | 287.785 / 266.696 | 283.805 / 270.482 | 279.997 / 270.356 | 272.065 / 268.974 | 278.511 / 270.966 |
| HOXJUZ | 23 | 272.725 / 268.564 | 278.335 / 270.353 | 278.446 / 268.810 | 280.468 / 266.942 | 281.315 / 268.624 |
| XACMOE | 21 | 111.783 / 111.783 | 113.475 / 113.475 | 122.976 / 122.976 | 110.120 / 110.120 | 133.629 / 99.735 |
| XACMOE | 22 | 111.715 / 111.715 | 109.237 / 109.237 | 116.532 / 116.532 | 113.498 / 113.498 | 128.453 / 113.832 |
| XACMOE | 23 | 151.551 / 113.010 | 116.022 / 116.022 | 106.664 / 106.664 | 135.405 / 115.168 | 131.175 / 110.839 |
| JEGPUJ | 21 | 125.412 / 120.942 | 55.821 / 55.821 | 96.994 / 96.994 | 36.762 / 36.762 | 93.212 / 93.212 |
| JEGPUJ | 22 | 136.805 / 101.912 | 45.215 / 45.215 | 122.419 / 122.419 | 71.757 / 71.757 | 114.410 / 114.410 |
| JEGPUJ | 23 | 148.459 / 148.459 | 44.850 / 44.850 | 90.743 / 90.743 | 81.779 / 74.198 | 147.399 / 111.501 |
| NEPNEF | 21 | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) |
| NEPNEF | 22 | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) |
| NEPNEF | 23 | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) | FAILED: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback) |
| CAXTAX | 21 | 71.121 / 58.377 | 79.624 / 48.431 | 67.076 / 67.076 | 68.637 / 40.092 | 88.550 / 58.416 |
| CAXTAX | 22 | 81.310 / 50.958 | 74.285 / 49.453 | 78.807 / 67.653 | 59.988 / 39.118 | 87.445 / 48.972 |
| CAXTAX | 23 | 64.728 / 46.617 | 67.674 / 47.368 | 66.487 / 65.353 | 73.248 / 43.361 | 78.681 / 42.080 |
| FELSAU | 21 | 32.963 / -4.313 | 22.031 / 1.404 | 12.245 / -3.712 | 13.078 / 0.667 | 41.499 / -7.880 |
| FELSAU | 22 | 39.899 / 5.522 | 18.363 / 7.582 | 11.191 / -5.420 | 16.089 / 9.870 | 33.784 / -3.727 |
| FELSAU | 23 | 31.397 / 0.473 | 14.568 / -1.251 | 24.293 / 3.216 | 19.438 / 14.824 | 29.500 / -4.814 |
| MIVXAR | 21 | 62.158 / 62.158 | 62.053 / 62.053 | 62.075 / 62.075 | 62.086 / 62.086 | 62.073 / 62.073 |
| MIVXAR | 22 | 62.131 / 62.131 | 62.024 / 62.024 | 62.184 / 62.184 | 62.095 / 62.095 | 62.117 / 62.117 |
| MIVXAR | 23 | 62.115 / 62.115 | 62.233 / 62.233 | 62.214 / 62.214 | 61.975 / 61.975 | 62.162 / 62.162 |
| LOPCAW | 21 | 135.225 / 129.494 | 134.134 / 129.343 | 133.490 / 129.654 | 133.548 / 129.949 | 133.428 / 129.769 |
| LOPCAW | 22 | 135.100 / 129.420 | 134.166 / 129.589 | 133.620 / 129.526 | 134.310 / 129.976 | 133.438 / 129.665 |
| LOPCAW | 23 | 134.909 / 129.538 | 133.911 / 129.491 | 133.540 / 129.517 | 133.657 / 129.467 | 133.448 / 129.467 |

### Load context and execution

Start, approximately 20-minute snapshots and completion are preserved in load_context.jsonl and per-snapshot nvidia-smi files. Samples cannot rule out load changes between them; wall ratios describe the observed environment. Other jobs were recorded, never used to stop or rerun confirmation.

| UTC | Elapsed min | Load 1 / 5 / 15 | CPU % | Available RAM GiB | Swap used GiB |
|---|---:|---|---:|---:|---:|
| 2026-11-04T01:58:09.678974+00:00 | 0.0 | 1.79 / 2.51 / 9.04 | 12.8 | 11.57 | 1.23 |
| 2026-11-04T02:18:10.856639+00:00 | 20.0 | 17.81 / 18.06 / 15.07 | 99.8 | 6.89 | 1.23 |
| 2026-11-04T02:35:28.062681+00:00 | 37.3 | 3.09 / 11.31 / 14.22 | 11.2 | 8.90 | 0.96 |
| 2026-11-04T02:38:11.992058+00:00 | 40.1 | 1.37 / 7.14 / 12.16 | 11.4 | 8.86 | 0.96 |
| 2026-11-04T02:58:13.123459+00:00 | 60.1 | 4.15 / 4.45 / 6.59 | 35.6 | 6.83 | 0.95 |
| 2026-11-04T03:18:14.343884+00:00 | 80.1 | 4.66 / 4.53 / 5.04 | 35.2 | 7.23 | 0.94 |
| 2026-11-04T03:38:15.476109+00:00 | 100.1 | 4.46 / 4.51 / 4.62 | 39.1 | 6.19 | 0.93 |
| 2026-11-04T03:58:16.577096+00:00 | 120.1 | 4.65 / 4.36 / 4.43 | 45.2 | 7.22 | 0.93 |
| 2026-11-04T04:18:17.718117+00:00 | 140.2 | 4.86 / 5.20 / 4.92 | 37.5 | 6.31 | 0.93 |
| 2026-11-04T04:38:18.831944+00:00 | 160.2 | 4.73 / 4.53 / 4.61 | 43.7 | 6.36 | 0.93 |
| 2026-11-04T04:58:19.954170+00:00 | 180.2 | 4.94 / 4.69 / 4.59 | 44.3 | 6.08 | 0.87 |
| 2026-11-04T05:17:41.880258+00:00 | 199.6 | 5.08 / 4.82 / 4.69 | 42.3 | 9.20 | 0.87 |

Other extraction/Architector jobs observed at sampled times:

| PID | First / last observed UTC | Maximum sampled process CPU % | Command |
|---:|---|---:|---|
| 111576 | 2026-11-04T01:58:09.678974+00:00 / 2026-11-04T01:58:09.678974+00:00 | 99.7 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/checks/stock_sentinel/D21/input.json |
| 116296 | 2026-11-04T02:18:10.856639+00:00 / 2026-11-04T02:18:10.856639+00:00 | 404.3 | python -m fit.typing_reextract run --mode t1d_promote --workers 3 |
| 116295 | 2026-11-04T02:18:10.856639+00:00 / 2026-11-04T02:18:10.856639+00:00 | 386.1 | python -m fit.typing_reextract run --mode t1d_promote --workers 3 |
| 116294 | 2026-11-04T02:18:10.856639+00:00 / 2026-11-04T02:18:10.856639+00:00 | 258.6 | python -m fit.typing_reextract run --mode t1d_promote --workers 3 |
| 135420 | 2026-11-04T02:58:13.123459+00:00 / 2026-11-04T02:58:13.123459+00:00 | 100.5 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/A04/native/seed_11/input.json |
| 135471 | 2026-11-04T02:58:13.123459+00:00 / 2026-11-04T02:58:13.123459+00:00 | 100.4 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/A04/native/seed_12/input.json |
| 135478 | 2026-11-04T02:58:13.123459+00:00 / 2026-11-04T02:58:13.123459+00:00 | 100.4 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/A04/native/seed_13/input.json |
| 139375 | 2026-11-04T03:18:14.343884+00:00 / 2026-11-04T03:18:14.343884+00:00 | 99.8 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/D15/native/seed_11/input.json |
| 139425 | 2026-11-04T03:18:14.343884+00:00 / 2026-11-04T03:18:14.343884+00:00 | 99.8 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/D15/native/seed_12/input.json |
| 139471 | 2026-11-04T03:18:14.343884+00:00 / 2026-11-04T03:18:14.343884+00:00 | 99.8 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/D15/native/seed_13/input.json |
| 143067 | 2026-11-04T03:38:15.476109+00:00 / 2026-11-04T03:38:15.476109+00:00 | 99.5 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/D21/native/seed_13/input.json |
| 143816 | 2026-11-04T03:38:15.476109+00:00 / 2026-11-04T03:38:15.476109+00:00 | 99.5 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/D22/native/seed_13/input.json |
| 144432 | 2026-11-04T03:38:15.476109+00:00 / 2026-11-04T03:38:15.476109+00:00 | 99.4 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/baseline/D23/native/seed_11/input.json |
| 148730 | 2026-11-04T03:58:16.577096+00:00 / 2026-11-04T03:58:16.577096+00:00 | 99.7 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/S02/native/seed_11/input.json |
| 148750 | 2026-11-04T03:58:16.577096+00:00 / 2026-11-04T03:58:16.577096+00:00 | 99.6 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/S02/native/seed_12/input.json |
| 148759 | 2026-11-04T03:58:16.577096+00:00 / 2026-11-04T03:58:16.577096+00:00 | 99.6 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/S02/native/seed_13/input.json |
| 154688 | 2026-11-04T04:18:17.718117+00:00 / 2026-11-04T04:18:17.718117+00:00 | 100.3 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/B02/native/seed_13/input.json |
| 154021 | 2026-11-04T04:18:17.718117+00:00 / 2026-11-04T04:18:17.718117+00:00 | 99.4 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/B02/native/seed_11/input.json |
| 154244 | 2026-11-04T04:18:17.718117+00:00 / 2026-11-04T04:18:17.718117+00:00 | 99.3 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/B02/native/seed_12/input.json |
| 158375 | 2026-11-04T04:38:18.831944+00:00 / 2026-11-04T04:38:18.831944+00:00 | 100.6 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/B04/native/seed_13/input.json |
| 159592 | 2026-11-04T04:38:18.831944+00:00 / 2026-11-04T04:38:18.831944+00:00 | 100.6 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/B05/native/seed_11/input.json |
| 159735 | 2026-11-04T04:38:18.831944+00:00 / 2026-11-04T04:38:18.831944+00:00 | 100.6 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/B05/native/seed_12/input.json |
| 165042 | 2026-11-04T04:58:19.954170+00:00 / 2026-11-04T05:17:41.880258+00:00 | 99.9 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/D02/native/seed_12/input.json |
| 166051 | 2026-11-04T04:58:19.954170+00:00 / 2026-11-04T04:58:19.954170+00:00 | 99.5 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/D02/native/seed_13/input.json |
| 166081 | 2026-11-04T04:58:19.954170+00:00 / 2026-11-04T04:58:19.954170+00:00 | 99.5 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/D03/native/seed_12/input.json |
| 169843 | 2026-11-04T05:17:41.880258+00:00 / 2026-11-04T05:17:41.880258+00:00 | 99.8 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/D04/native/seed_11/input.json |
| 169902 | 2026-11-04T05:17:41.880258+00:00 / 2026-11-04T05:17:41.880258+00:00 | 98.8 | /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/.pixi/envs/default/bin/python /mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/arch/research/bench/benchmarks/scripts/run_jobs.py --worker /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/architector/run_20261103/structures/challenge/D04/native/seed_12/input.json |

T1d extraction appeared after launch. The run continued without interruption, as the audit required. The host was not quiet throughout; performance ratios have this material limitation. The frozen numerical wall gate was still applied exactly, without a load correction or rerun.

2026-11-04T02:35:28.149084+00:00: owner reports SIGSTOP of T1d to release CPU for confirmation. same confirmation continues; do not SIGCONT here; tell owner after generation and analysis complete.

Driver return code: 0. Observed process wall (including MACE/export/startup): 199.555 min. Colab spend: 0 units.

Record status: {"C": {"failed": 3, "ok": 357}, "K": {"failed": 3, "ok": 357}, "KU": {"failed": 3, "ok": 357}, "P": {"failed": 3, "ok": 357}, "S1": {"failed": 3, "ok": 357}}.
Equal-compute checks: 119 pass, 0 fail, 1 unavailable.

Parent/arm warning totals (not tripled over copied seed metadata): `{"C": {"feasibility_check_failed": 0, "infeasible_kept": 0, "infeasible_replaced": 2, "pivot_correction_fallback": 0, "polish_failure": 0, "replay_fallback": 0}, "K": {"feasibility_check_failed": 0, "infeasible_kept": 0, "infeasible_replaced": 2, "pivot_correction_fallback": 0, "polish_failure": 0, "replay_fallback": 0}, "KU": {"feasibility_check_failed": 0, "infeasible_kept": 0, "infeasible_replaced": 2, "pivot_correction_fallback": 0, "polish_failure": 0, "replay_fallback": 0}, "P": {"feasibility_check_failed": 0, "infeasible_kept": 0, "infeasible_replaced": 2, "pivot_correction_fallback": 0, "polish_failure": 0, "replay_fallback": 0}, "S1": {"feasibility_check_failed": 0, "infeasible_kept": 0, "infeasible_replaced": 2, "pivot_correction_fallback": 0, "polish_failure": 0, "replay_fallback": 4}}`.

Failure ledger (full reasons also in C2B_CONFIRMATION.json):

- C NEPNEF seed 21: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- C NEPNEF seed 22: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- C NEPNEF seed 23: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- K NEPNEF seed 21: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- K NEPNEF seed 22: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- K NEPNEF seed 23: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- P NEPNEF seed 21: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- P NEPNEF seed 22: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- P NEPNEF seed 23: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- KU NEPNEF seed 21: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- KU NEPNEF seed 22: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- KU NEPNEF seed 23: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- S1 NEPNEF seed 21: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- S1 NEPNEF seed 22: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)
- S1 NEPNEF seed 23: RuntimeError: GFN2 embedding fallback for NEPNEF; all generated structures invalid (including cached fallback)

### Final integrity and interpretation

K (B), P (D) and KU (B) qualify under their declared routes. This does not
establish a statistically significant panel-average energy gain: the floor
and top-1 improvement CIs still include zero. P meets the correctness route
with non-inferiority and zero artefact-excluded guard trips. No source bug is
promoted; P is the correct start_kabsch implementation per the audit. Qualified
search still enters the D3 joint gate, with no untested P+U blend substituted.

S1 fails only floor non-inferiority: +0.959 [-0.963,+2.578] kJ/mol, upper
bound above +2. Its stronger matched rho (-0.069 [-0.108,-0.030]) and extra
rows do not rescue that failed gate. No seed-budget adoption recommendation.

Descriptive top-1 P-minus-K is -1.050 [-5.069,+2.273] kJ/mol and KU-minus-K
-1.130 [-6.498,+2.817]; neither establishes an average energy advantage over
K. KU adds +0.966 [0.455,1.571] distinct rows versus K. These comparisons
remain descriptive and did not change any route or threshold.

Exactly 1,800 records: 357 valid + 3 known NEPNEF failures in each arm. All
record fingerprints, flags, producer commit and DB/model identities match.
119 S1 equal-compute checks pass, one unavailable (NEPNEF), none fail;
ordinary-arm row-steps match per parent. No enrichment failure. Four S1 graph
capture fallbacks (FEKKAJ/QEZVEY) were retained; no pivot correction fallback.
Frozen sources, original production DB/sidecar and golden hashes are unchanged.
No additional generation, relaxation, route rescue, production default change or L4 run.

One auxiliary completion-wait launcher failed because os.pidfd_open is absent
in this Python build. That happened before any postprocessing, and its log and
marker were preserved. The replacement used 20-minute completion-marker waits
and ran the frozen postprocessing chain once. Generation was never restarted.

Generation completed UTC 2026-11-04T05:17:41; postprocessing finished
06:13:50. The owner-paused T1d processes were not resumed or killed by C-2b.
All C-2b compute is finished; the owner can SIGCONT them. Stop for audit.
