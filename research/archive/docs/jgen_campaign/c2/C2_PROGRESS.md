# C-2 progress

2026-11-02: Read AGENTS, directive, PLAN status/search coverage, A4t audit and
addenda, roadmap search definitions. Created jgen-c / jgen/c-coverage from
b0134a2. Verified production fingerprint 5f74b14b57ed and xtb resolution.
Registered Phase 0 contract before measurement. No flags built; no production
defaults changed. Colab spend: 0 units; no VMs started.

The locked exclusion list removes validation-derived tail members; population
selection must apply it explicitly because the legacy panel selector does not.

2026-11-02 checkpoint: §0.3 measured first. Two-parent pilot 4.77 s; estimated
full ring census about one minute; 120 eligible parent records completed in
41.54 s summed measurement time, 0 failed records. Internal 5–8 rings carry
>50% on 2/9 (DOHTOM, HOHZAG), below the four-parent build threshold. UYOVUA
chelate union carries 84.98%, triggering the directive's stop rule. Separate
production prepare/FK verifies its nonaromatic 13-member ring is outside
the macrocycle path (0 macro joints, 2 ordinary torsions, B=128, W=16).
No IK steps, flags, GPU jobs or VM launches. Colab spend remains 0 units.

Placement and tiny-pool traces are explicitly not run due to this stop;
Phase 0 is partial. Gate results are not-run, not failed. Two metric tests
passed. Initial verification import error corrected; LOPCAW topology fallback
warning retained in the report. See C2_REPORT.md and c2/C2_PHASE0.json.

2026-11-02 Amendment 1: stop revoked (majority means >=5/9). Placement pilot 24.47 s / two parents; full census estimated six minutes at four workers. Tiny-pool pilot QAPFAS 83.56 s, UYOVUA 19.31 s including cold startup; remaining ten parents estimated nine minutes on local 2060. No Colab.

2026-11-02 tiny-pool fix definition, registered BEFORE coding: HOXJUZ seed 12
has 128 finite, embedding-valid solved rows; the 200-nat window removes three
rows and reduces the exact distinct set from 12 to 10, below its existing
16-row output cap. This is class (b). Build `pool_window_fill` (default false):
keep the best-first, finite, symmetry-distinct rows up to the existing output
cap even when some fall outside the 200-nat window. Top-1, solved-row count,
ranking and the output cap are unchanged. No per-complex exception and no
new numeric target. The same existing cap bounds output on every parent.
This tests the observed window loss; it does not claim to repair the separate
SIVBUV solver collapse (class c), which is retained as an unresolved finding.

2026-11-02 Phase 1: K/U/T implemented default off; P omitted. Phase 2 pilot (AJAQAD, BABNAV, JEGPUJ; C,K,U,T,S1; seeds 11-13) completed all 45 records. Actual 139.00 s incl startup, MACE and export. Estimated full screen 93 minutes. Per-arm generator estimates C 12.65, K 11.55, U 12.53, T 12.36 minutes; S1 8.82 minutes per wide-seed experiment (three replicates for noise). Row-steps match; CPU identity exact in exported coordinates, scores and energies on BABNAV/MIVXAR/UYOVUA (one seed, --cpu-identity). 17 profile/coverage tests pass. HOXJUZ seed 11 full-200-step replay/eager/CPU-package parity passes, zero score differences, no fallback. K full offset rerun and read-only golden check overlap CPU work; no replayed source path changed, so regeneration not required. Initial multiseed indentation error fixed before successful pilot. All Colab spend 0.

2026-11-02 validation update: K rerun complete (119 valid, NEPNEF failed), row-max median 2.7506815195 A, exactly the Phase 0 paired-fit value. Full CPU suite (GPU hidden to preserve single GPU job): 374 passed, 85 skipped, 3 xfailed, 2 missing-input-path failures. Multiseed failure rerun passes with shared JACOBIK_TEST_DB; reader test path failure retained. Golden check FAILED: 8/245 arrays per profile (four solve parents) differ under one-worker setting. Unchanged b0134a2 engine reproduces exactly the same eight differences under this setting; no goldens changed and no flag/default changes made to chase references. Full paired screen started from b9ce924, with read-only metric enrichment overlapping GPU work.

2026-11-02: full-precision NPZ coordinates/scores bit-identical to b0134a2 on BABNAV/MIVXAR/UYOVUA, one seed, one worker/no cull. Supplemental geometry-only check, no new MACE evaluation. Nine focused tests pass. Before opening Phase 2 comparisons, corrected accounting for native E/Z S1: each independent wide-seed experiment pays full own preparation, plus one-third of the three-replicate seed/solve cost. Raw timing ledger is preserved; finalise.py applies this after both record writers finish and checks compute matching. Portable Phase 0 population copied to C2_PANEL.json for conditional L4 use; membership/frame hashes checked, no VMs started.

2026-11-02 initialization audit: corrected first-native-E/Z-variant limitation in Phase 0. Nine multi-variant parents, total 133 native parent/variant configurations (132 valid, NEPNEF invalid), now covered in C/K. 195456 valid initial rows per census. Pooled medians corrected to 3.1525564194 / 2.8071253300 A, reduction 10.957%; parent gate still 101/120. Stored A4b pool mapping + all-variant physical initial conformers corrected aromatic mode hits. All tiny-pool parents have one variant; no trace changes. Original files retained in *_first_variant. Full screen has used native enumeration throughout. FEKKAJ completed with recorded graph-capture OOM/eager fallbacks in 588.3 s; no screen settings changed.

2026-11-03 FINAL checkpoint: all 1,800 Phase 2 records present (357 ok + 3 NEPNEF failures per arm). Native E/Z and all scoped measurements complete. 119 S1 compute matches, zero mismatches, NEPNEF unavailable. Full wall 11,440.95 s / 190.68 min, versus 93-min pilot estimate. Gates (primary/top1/guard/wall): K FAIL/FAIL/PASS/PASS; U FAIL/FAIL/FAIL/PASS; T FAIL/PASS/PASS/PASS; S1 FAIL/FAIL/PASS/PASS. No arm qualifies; stop after Phase 2 per directive. No flags selected, no S1 recommendation, no L4 confirmation. K floor -1.62 [-3.79,0.21], U -2.14 [-5.88,0.35], T 0 [0,0], S1 -0.003 [-1.48,1.32] vs equal-compute C union. S1 resource events: two graph-capture fallback warnings each on FEKKAJ/QEZVEY, successful retries in replay mode; retained. No union fallback or enrichment failure. Final credits log: 476.36 balance, 0.00/hr, active assignments 0. C-2 units spent 0; no VM started or owned. Default fingerprint 5f74b14b57ed and both shared golden hashes unchanged. Nine focused tests pass; earlier full-suite/path and one-worker golden failures remain in report. Await audit; no production fold-in.
