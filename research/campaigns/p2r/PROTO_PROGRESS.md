# P2R Step 4 prototypes: dev loop progress (branch `jgen/p2r-proto`)

Dev loop (no billed time):
- solves on a free Perlmutter login A100 through `jk_run.sh ... panel`, paired arms on the same node type;
- relaxation + A4c with `rerun post --select a4c --mace-device cuda` on a login node (24 workers, about 5 min per arm);
- `tools.jgen.a4c.paired` for the paired comparison (parent-clustered bootstrap; unsolved units count as misses).

Dev set D1 (cal only, seeds 11-13, 42 parents): the 13 "reference basin not reached" parents, 14 partial-miss parents,
10 all-hit guards, 5 not-offered parents (ZOJDEH, YOSXIP, LIPDEU, WEYJIT, NOYTUS). Selected from the baseline buckets, so
D1 numbers are optimistic for the arms by construction; the gate is the full cal + tail panel with fresh seeds.

## Prototypes (all default off; fingerprint d5e2f4113497 unchanged at the defaults)
| Flag | What | Status |
|---|---|---|
| `beam_core_share` | share of beam width kept for distinct mode combinations of core torsions (pivot within bonding distance of a donor) | D1 done (below) |
| `out_mirror` | enantiomer completion: reflect the kept rows when no ligand carries a tetrahedral tag; the reflected row keeps the source score (score mirror asymmetry measured: max 0.019, mean -0.0006 over 64 YOSXIP rows; a final version rescores) | D1 done: clear gain |
| `ring_states` | per-ligand ring-pucker state library in the one batch (near six-rings: chairs, twist-boats; only ring atoms + H move, donors fixed; coupled by summed MMFF energy, <= 8 states; rows half equal, half Boltzmann; -dE/(RT Z) in the score; remote rings lowest pucker; state 0 = embed) | full panel done: replaces embed_ring_alt |
| `config_states` | Step B: every configuration class (E/Z combinations, stereo classes) in ONE joint IK as row blocks: per-class prep, per-block cull, per-row donor-symmetry ties, per-class assembly, the same merge; `max_rows` splits whole classes over batches | equivalence done (below); panel run `p2r-full-cfg` |
| `config_near` | stereo classes from the capped product of the open elements (near: binding-created or <= 2 bonds from a donor; remote ring centres kept in the product), mirror pairs collapsed only with `out_mirror` and no tag, cap x2, extras whose prepared configuration repeats production / its reflection / another extra dropped | 5-parent check done; panel run `p2r-full-cfg` |
| `config_remote` | remote acyclic elements derived after the solve: best 4 rows per class inverted locally (isometric), 60-step repair IK, rescored, added as a derived class | synthetic checks only: no cal/tail parent has a derivable remote element |
| `embed_ring_alt` | ring-pucker candidates: production embed plus a chair embed (`ring_chair`, from `jgen/c-chair`), solved separately, pools merged by score; nothing is forced (owner 2026-11-12: no all-chair rule; the per-ring state library of the design supersedes this stopgap, which offers only one all-chair alternative) | D2 done: gain on ring parents |

## Results
### Step B: configuration classes in one batch (`config_states`, `config_near`, `config_remote`)
Why a batch: the plan's stereo scaling rule (near elements as a capped product in the batch, remote elements derived
after the solve, one enantiomer solved and reflected, output by class) needs every class's rows in one joint IK, so
that the selection sees all classes at once.
- **Equivalence, batched vs sequential classes** (engine, seed 11, local 2060): per class, the same best score and the
  same row count on BOTYUH, USAPOU, XACMOE (exact), FUTSOD, TEDPOI (≤ 7e-6), CIXYUD (0.002, sign-flip amplification
  of float noise). The one real defect: donor-symmetry ties are per class (a stereo class can break a donor
  equivalence: USAPOU class 1 has other symmetric groups than class 0), and the batch had applied class 0's ties to
  every row (2023 of 2048 rows changed, up to 86 score units). Fixed with per-row group weights
  (`JointIKSolver.sym_rows`, ported into the replay loop). Bench route (classes via the engine) with `out_mirror` +
  `ring_states` + `ring_states_five`: FUTSOD, JUPPOZ, LIPDEU, TEDPOI, USAPOU identical to the sequential bench.
- **Batch wall, engine** (local 2060, seed 11; sequential -> batched): BOTYUH 5 -> 2 s, FUTSOD 34 -> 16 s, USAPOU 27 -> 20 s,
  TEDPOI 5 -> 2 s, CIXYUD 9 -> 7 s, XACMOE 43 -> 33 s.
- **`config_near`, 5 parents** (seed 11, with `out_mirror`): BOTYUH, LISVOC, PANBOY, USAPOU keep the same top-1 and the
  same number of distinct configurations with one class fewer. The dropped extra repeats production or its reflection;
  wall drops 23-50 %. FUTSOD (6 open centres, all remote and in rings, so not derivable): 8 extras instead of 4 (cap
  x2), top 0.347 -> 0.395, wall x1.3.
- **Census of the 15 stereo-extra parents (cal + tail):** at most 2 near elements each, except FUTSOD. No tagged
  centres. No parent has a remote acyclic element, so `config_remote` changes nothing on cal + tail.
- **`config_remote`, synthetic Pt(L)Cl3 checks only:** the inversion is an exact isometry and inverts the centre (unit
  test). The repair starts at the inverted rows: joint columns are absolute dihedrals, so the start is the measured
  dihedrals, not zero, and the FK identity is checked per call. Results:
  - `NC(C)CCC(C)CC`: a derived row is the best of its diastereomer, 1.292 vs 1.261 for the solved class;
  - `NC(C)CC/C=C/C(C)CC`: derived rows sit 0.11-0.15 below the solved class (60 repair steps are short).

  As a replacement for solving remote classes it is not validated.
- **Panel arm cfg** = mrs5 flags + `config_states` + `config_near`, code 2143bf1. Run `p2r-full-cfg-{b,a}`, both
  halves on login28 (login15's GPU was taken by another user). Posts `p2r-full-cfg-*-post-cfg`. Contract rule, 351 units
  / 117 parents.
  - **vs fresh control:** C-MISS 0.322 -> 0.245, d -0.077 [-0.120, -0.040]; C-TOP1 0.407 -> 0.413, d +0.006
    [-0.020, +0.028]. Untouched units 0 flips. Gains 28 (reflected 19, ring state 7, production rows 2); losses
    NUWFAM s13 (as in every run).
  - **vs mrs6 (the same flags, classes sequential):** C-MISS 0.239 = 0.239, d 0.000 [-0.020, +0.023]; top-1 identical.
    - Gains: FUTSOD s12, s13 (8 extras instead of 4) and ZOJDEH s11.
    - Losses: LIPDEU 3/3. In mrs6 its hit was a reflection of a row of the extra class x0. That class realises the same
      P configuration as production, from another embed, so `config_near` dropped it as a repeat. The repeat was a
      second independent embed and search of the same configuration, and only that search reached the reference basin.
      Dropping repeats saves wall but costs this coverage. The d 0.000 against mrs6 is therefore two mechanisms of
      opposite sign inside one flag: cap x2 gives +2 (FUTSOD), the repeat drop gives -3 (LIPDEU). Recommendation: take
      the repeat drop out of `config_near` (keep repeats as a second independent search) rather than give it its own
      switch. Not re-run (minimal confirmations).
  - Whether a panel parent fell back to per-class solves (another layout, or an OOM split) cannot be read from the
    lane logs, which do not capture the jacobik logger. Not seen in the 10 local parent checks.
  - **Wall vs mrs6, same node only (half b, login28):** median x0.98, total x1.06. Half a (login28 vs mrs6's login15):
    median x0.99, total x1.02. The slowest parents are E/Z (CIXYUD, YABYEJ, LOPCAW x1.6-1.7): this code solved one seed
    per batch, so the bench lost its seed packing. 4b90e57 packs seeds and classes.
  - **Wall rerun with seed packing** (`p2r-full-cfg2-{b,a}`, 4b90e57, login28). Top-1 and floor dE are identical to cfg on every solved unit except
    JUPPOZ and LIPDEU s11-13: float noise, at most 1e-4 kJ/mol, same basins. QEZVEY times out as in every run. So the
    cfg coverage numbers hold for the packed code.: vs mrs6, half b median x0.96, total x1.00; half a median x0.94, total
    x0.89. vs fresh control: total x1.07 on both halves. The batch pays for the ring states and the extra FUTSOD
    classes, and Step B makes the class solves no slower than sequential. It does not deliver a large speed-up on
    this panel: only 24 of 120 parents have more than one class. PENHEY, OMAGAK and WECHOE ran x3 on unchanged code
    paths (single-class parents): shared-node noise, which also bounds how far these walls can be read.
- **The remaining 95 misses of arm cfg (86 C-MISS plus 9 unsolved; contract rule):**
  - **65: right class, another basin.** The best relaxed same-class row is +5-15 kJ/mol (34), +15-50 (28) or > +50
    (3) above the relaxed reference. The nearest same-class core RMSD has median 0.62 A [IQR 0.47, 0.79].
  - **21: class absent from the relaxed rows.** 18 of them are contact-set differences: the generated rows touch the
    metal with another atom set than the reference (GOTWET, HOXJUZ, JUCBIV, LIPDEU, DAMVAQ, HEDJOP, NUWFAM). Only 2 are
    configuration differences (XACMOE s11 a centre, OMAGAK s13 E/Z); EBAYIQ s12 has donor sets that differ.
  - **9 unsolved:** NEPNEF and UYOVUA fail (embed), QEZVEY times out. The same in every arm.
  - **Threshold sensitivity (not a proposal; the contract stays):** at +15 kJ/mol or 0.5 A core, 40 of the 86 would
    hit. At +5 / 0.5 A, 18 would.
- **Reachability (`a4c.reach`) on every same-class-miss parent** (36 parents: the earlier 6 plus 30 now, run
  `p2r-reach2`, login28). 8 multi-class parents (FETTAZ, FUTSOD, HOYBED, NUWFAM, XACMOE, YABYEJ, YOSXIP, ZOJDEH) are
  not probed: `prepare` refuses class loops. The probe fits production starts to the reference, solves from there, and
  compares with the pool:
  - **Holds the basin (≤ 0.35 A) but scores it far below the pool:** FEXWUC (core 0.05; -1.56 vs 1.34), HOHZAG
    (0.24; -5.23 vs 0.99), CAXTAX (0.33; -7.6 vs 1.10), OMAGAK (0.32; 0.00 vs 0.11). Score surface.
  - **Holds it and outscores the pool's same-class rows:** XEVDAE (0.26; 0.546 vs 0.411). The only clear search miss.
  - **Drifts 0.4-0.6 A, scoring at or below the pool:** DUHHIX, TIKZOF, XOJHIN, EFOXII, WIBHOH, MUBREJ, VAVBIE,
    NUCKUS, BABNAV, JUPDON, BOHWUT. BOHWUT outscores the pool's same-class rows (0.49 vs 0.32) and may be a partial
    search miss.
  - **Drifts ≥ 0.7 A:** DOHTOM, ZEFXAM, CAQLEP, EZIBUL, NUHPEL, LEPFUJ, ZAQNUE, CIXYUD.
  - **Leaves the class:** CIFZEZ, HUNGOP, MEJYIN, and GOTWET (fit 1.0 A, which says the embed cannot represent the
    reference).

  Clean fits (≤ 0.1 A heavy: EFOXII, DUHHIX, FEXWUC, TIKZOF, VAVBIE, XEVDAE, HUNGOP) read 6 score, 1 search. Fits of
  0.2-0.6 A carry the embed's bonds, angles and rings, so drift there is "score or embed internals".
- **Reading:** Step B batches the classes and is equivalent by construction. The class axis it serves is almost
  exhausted on cal + tail: 2 configuration misses remain, plus the LIPDEU drop. The remaining misses are within-class
  basins the score does not favour (the score track: Q-term fit, NN correction), contact-set or binding-mode
  differences (18 units, to inspect: generator, reference or metric), and 9 unsolved parents (embed failures, cost).
- **Not done:** macrocycle ligands inside the batch. No multi-class cal/tail parent has a macrocycle, so this is
  untested.
### Bench defect: stereo-extra parents were solved with only their LAST stereo class (fixed f699d25)
`multiseed.prepare` stops the engine with an exception at the first `_solve_for_assignments` call. But
`engine._solve_stereo_variants` catches exceptions per class, so every class overwrote the capture and the bench solved
only the last class. This affected 15 of the 120 cal+tail parents, in the baseline and in every arm. On those parents
the production (input-tagged) class was never solved. Every earlier `tools.run`/bench number on stereo-extra parents
carries this defect, including the Step 2 baseline 61204770 and the results below that use it. The fix: the bench
enumerates the classes itself, with engine semantics (same per-class fixes, cross-class dedup with the production
class first, output window), seed-batched per class. Checked against engine `solve()` on BOTYUH, USAPOU and FUTSOD:
same classes, same per-class best scores and row counts. `prepare()` now refuses a second capture.

### Current reference: all three flags on the corrected bench (19d79fa) vs the fresh control, same nodes
Arm mrs5 = `out_mirror` + `ring_states` + `ring_states_five`, run `p2r-full-mrs6-{a,b}` on login15/28 (the control's
own nodes), posts `p2r-full-mrs6-*-post-mrs5`. Two fixes went in before this run. First, at equal score a reflection
now ranks after its source in every class merge (engine and bench; production order unchanged, ws golden 0/490).
Second, the bench sends a ring_states arm through `solve_seeds` only on parents with a saturated ring it can change.
- **Contract rule (unsolved = miss), 351 units / 117 parents:** C-MISS 0.322 -> 0.245, d -0.077 [-0.123, -0.037];
  C-TOP1 0.407 -> 0.413, d +0.006 [-0.020, +0.028]. Untouched units (15) 0 flips.
- miss -> hit 28:
  - reflected 18: LIPDEU 3/3, NOYTUS 3/3, LISVOC 3/3, PANBOY 3/3, JUPDON 2/3, HOYBED, WEYJIT, XOJHIN, YOVNEC;
  - ring state 7: FEKKAJ 3/3, EZIBUL 2/3, DOHTOM s13, HOHZAG s13;
  - production-path rows 3: XACMOE s12, YOSXIP s11, s12. Stereo-extra parents; not yet traced.
- hit -> miss 1: NUWFAM s13 (relax-set selection, as in every run).
- Top-1 lost: SILROY 3/3. It still hits, but ring-state rows outrank the reference basin (C-RANK). Gained: DOHTOM
  s12 and s13, EZIBUL s12, FEKKAJ s11, XACMOE s12.
- **Solve wall, same nodes, 116 parents (QEZVEY times out at 300 s in both):** median per-parent x1.06, total x1.13,
  p90 x1.51. Outliers are ring parents: DOHTOM 20 -> 128 s (x6.3), SILROY 39 -> 88 s, ZAQNUE, FEXWUC, DAMVAQ,
  HOHZAG, FEKKAJ x1.7-2.0. The previous run's x1.21 median / x1.30 total was mostly the bench route: mirror-only
  parents re-ran the prep per arm (ZAZREZ 27 -> 105 s, MEBHEK 7 -> 133 s).
- The 9 "production-row" gains of the re-baseline below were mostly mislabelled: on LISVOC, PANBOY and HOYBED the hitting
  row was a reflection ranked before its equal-score source, so the attribution called it the source's
  production row. With the tie fixed they count as reflected.

### Re-baseline on the fixed bench (f699d25): `out_mirror` + `ring_states` vs a fresh control
Runs `p2r-rb-{ctl,mrs}-{a,b}` on login15/28/34/35 (all A100-PCIE-40GB), posts `p2r-rb-*-post-*`, 351 units / 117
parents, the same unsolved parents in both arms (NEPNEF and UYOVUA fail, QEZVEY times out), counted as misses.
- **mrs vs fresh control:** C-MISS 0.322 -> 0.251, d -0.071 [-0.117, -0.031]; C-TOP1 0.407 -> 0.419, d +0.011
  [-0.017, +0.040]. Untouched units (15) 0 flips.
- miss -> hit 27: reflected 11 (LIPDEU 3/3, NOYTUS 3/3, JUPDON 2/3, WEYJIT, XOJHIN, YOVNEC), ring state 7 (FEKKAJ
  3/3, EZIBUL 2/3, DOHTOM s13, HOHZAG s13), and 9 where the hitting row is a production-path row: LISVOC 3/3, PANBOY
  3/3, HOYBED s12, XACMOE s12, YOSXIP s12. These parents are touched, because they carry flag rows; the flag rows
  change the dedup and the output window around the production rows. hit -> miss 2: JUPPOZ s13, NUWFAM s13.
- Top-1 lost: SILROY 3/3, JUPPOZ s11 and s13. Gained: BOTYUH s11 and s12, DOHTOM s12 and s13, EZIBUL s12, FEKKAJ
  s11, HOYBED s12, XACMOE s12, YOSXIP s12.
- **The fix itself, old control vs fresh control:** misses 0.333 vs 0.322. SILROY now solves (it timed out before)
  and hits 3/3. Top-1 is 0.425 old vs 0.407 new, all of it from BOTYUH, QUGXUN and USAPOU, which lose top-1 on all
  seeds now that every class is solved. The production class ranks first (engine semantics), and its best row is not
  the reference basin, while the old bench's last class held it. Whether the reference structure sits in a
  non-production stereo class is a question for the stereo fold-in gate. Earlier gate evidence on these parents
  measured the last class, not production.
- The mirror's share of the gains held up after the fix (11 reflected gains vs 14 before).
- Pools from 9333e12 on carry the class labels. On USAPOU s11 the lowest-energy row is in a non-production class
  (unrelaxed MACE 0.0), while the best production-class row is +51.8 kJ/mol. BOTYUH and QUGXUN top-1 are in the
  production class (their top-1 losses are within-class).
- SILROY is at the 180 s parent cap with mrs: 174.5 s wall, 2 MPS OOM retries, solve 34.8 s vs 12.9 s in the
  control. The ring-state cost on this parent needs a look before any gate.

### `ring_states_five` (b46a878) on top: mrs5 = out_mirror + ring_states + ring_states_five vs the fresh control
Runs `p2r-full-mrs5-{a,b}` on login34/35, code 9333e12. Bench and solver code differ from the control snapshot only
in default-off paths and the pool-manifest descriptors; production rows are unchanged (ws golden 0/490).
- Five-ring states appeared on only 2 parents (JUPPOZ, XEVDAE), 3/3 seeds each.
- mrs5 vs mrs, solved units (348): JUPPOZ s13 miss -> hit (ring state + reflection); no other flip; top-1 identical.
- SILROY timed out on all 3 seeds in mrs5 (parent cap 180 s; 174.5 s in mrs), so under the contract rule mrs5 is
  d -0.066 [-0.114, -0.023] vs mrs d -0.071 [-0.117, -0.031]. That is the SILROY timeout, not the five-ring states (SILROY has no saturated five-ring; checked).
- Verdict: five-ring states are correct and cheap where they apply, but they rarely apply on cal + tail. Ship them
  with ring_states; their gain is one unit.

### Full panel: `out_mirror` + `ring_states` vs the Step 2 baseline (superseded by the re-baseline; commit 0bfb8af)
Run `p2r-full-mrs2-{a,b}` on login34 + login35 (A100-PCIE-40GB), posts `p2r-full-mrs2-{a,b}-post-mrs`, control =
baseline 61204770. One batch per parent: the ring-pucker states are rows of the same solve (no second solve).
- **Contract rule (unsolved = miss), 351 units / 117 parents:** C-MISS 0.333 -> 0.268, d -0.066 [-0.108, -0.028];
  C-TOP1 0.425 -> 0.436, d +0.011 [+0.000, +0.026].
- No unit unsolved in the arm. SILROY, a timeout in the baseline, now solves (173 s; 3 gains). Failures shared
  with the baseline: NEPNEF, UYOVUA (failed), QEZVEY (timeout).
- miss -> hit 24 units, all from the flags: reflected 14 (LIPDEU 3/3, NOYTUS 3/3, JUPDON 2/3, YOSXIP 2/3, HOYBED,
  WEYJIT, XOJHIN, YOVNEC); ring state 7 (FEKKAJ 3/3, EZIBUL 2/3, DOHTOM s13, HOHZAG s13); SILROY 3/3 hit by
  reflected ring-state rows. Production-path rows 0. Untouched units (15) 0 flips.
- hit -> miss 1: NUWFAM s13 (relax-set selection, as before). No top-1 lost; top-1 gained DOHTOM s12, s13,
  EZIBUL s12, FEKKAJ s11.
- Solve wall: x0.81 total, x0.96 median per parent (separate solves, node variance). HEDJOP 82 -> 84 s, XAQCAT
  113 -> 109 s, CIFZEZ 57 -> 65 s. Ring parents (the second solve is gone): FEKKAJ 77 -> 55 s, FUTSOD 47 -> 35 s,
  DOHTOM 31 -> 58 s, DUHHIX 36 -> 53 s, NUHPEL 24 -> 44 s.
- Unrelaxed quality: median energy released on relaxation 42.9 -> 42.9 kJ/mol.
- Against the embed_ring_alt arm (mr2), ring_states loses HOHZAG s11-12 and DOHTOM s11. It gains EZIBUL 2/3,
  SILROY 3/3 and four top-1s. Not yet explained: ring_alt chairified every saturated six-ring of the embed, so its
  rows have a different start for all rings at once. FUTSOD stays a miss on all seeds in every arm. Its top-1 is
  not worse, so the state energies prevent the axial-chair regression that forcing caused.
- Intermediate run `p2r-full-mrs` (8091a6f): d -0.068 [-0.111, -0.031] but C-TOP1 -0.003. The 7 top-1 losses
  (BOTYUH 3/3, FELSAU, JUPPOZ, QUGXUN, USAPOU) were rescored reflections 0.0006-0.024 above their equal-energy
  source, from the score's Q5 mirror asymmetry (open item below). Fixed: reflected rows keep the source's score and
  rank after it. The same run lost HOHZAG: an equal-free-ligand-energy merge dropped the metal-axial chair, and the
  ring donor moved. Fixed: donors pinned, merge removed. Remote rings now start in their lowest pucker.

### Full panel: `out_mirror` (fixed) + `embed_ring_alt` vs the Step 2 baseline (cal + tail, 120 parents, seeds 11-13)
Rerun `p2r-full-mr2-{a,b}` on login34 + login35 (A100-PCIE-40GB, same type as the baseline solve), posts
`p2r-full-mr2-{a,b}-post-mr` (MACE on GPU); control = baseline solve + post 61204770 (same production code). The D1/D2 dev
parents come from this panel: a confirmation on the source panel, not an independent test (fresh seeds 31-33 at the gate).
- **Contract rule (unsolved = miss), 351 units / 117 parents:** C-MISS 0.333 -> 0.274, d -0.060 [-0.103, -0.023];
  C-TOP1 0.425 -> 0.430, d +0.006 [+0.000, +0.017].
- Units solved in both (345 / 115): C-MISS 0.322 -> 0.261, d -0.061 [-0.104, -0.023].
- miss -> hit 22 units, each attributed from the hitting row's `row_key`: **reflected only 14** (LIPDEU 3/3, NOYTUS 3/3,
  JUPDON 2/3, YOSXIP 2/3, HOYBED, WEYJIT, XOJHIN, YOVNEC), **chair embed only 8** (FEKKAJ 3/3, HOHZAG 3/3, DOHTOM 2/3),
  production-path row 0.
- Noise: the arm is a separate solve from the control, yet over all 345 solved units no gain came from a
  production-path row and one unit was lost. (Secondary: on the 15 units / 8 parents where neither flag added a row,
  0 miss/hit and 0 top-1 flips; too few to stand alone.)
- hit -> miss 1: NUWFAM s13. Checked on the rerun pool: the control's hit row (row 27) is arm row 54 (same score and
  energy; the pool doubled to 64 rows by reflections), and the <= 5-row A4c relax set took rows 0, 1, 4, 6, 43, not 54.
  A relax-set selection effect of the measurement, not a lost structure.
- Timeouts: CIFZEZ and XAQCAT now solve (57 -> 107 s, 113 -> 135 s); **HEDJOP still times out** (82 s baseline, > 180 s
  in the arm; all-miss in the control, so no hit lost). Heartbeats show it still in the solve stage at 141 s, not in
  the post-solve dedup or MACE, so the XAQCAT cause was not assumed. Isolation (local 2060, seed 11, one flag per
  arm, separate solves): control 31.6 s (solve 18.6 s, 32 rows); `out_mirror` alone 59.8 s (solve 46.4 s, 62 rows);
  `embed_ring_alt` alone 31.4 s (no ring changed, no second solve). Cause: `out_mirror`. The mirror test runs inside
  the solve's assembly (hence the "solve" heartbeat) and costs about 28 s per seed on this parent; on a shared login
  A100 lane (baseline 82 s) that, plus MACE on twice the rows, passes the 180 s budget. Same fix as XAQCAT: test the
  reflections inside the production dedup pass (one-batch design). SILROY, QEZVEY time out in both;
  NEPNEF, UYOVUA fail in both.
- Solve wall: x0.90 in total over the 117 parents with a wall in both (excludes HEDJOP, whose arm wall is missing:
  its ratio is at least x2.2, 82 -> > 180 s) (median per parent x0.99; separate solves on other
  nodes, so this is within node variance, not a speed-up); largest per-parent increases DAMVAQ 10 -> 33 s, DOHTOM
  31 -> 79 s, NUHPEL 24 -> 54 s, FUTSOD 47 -> 94 s, DUHHIX 36 -> 69 s (ring parents: second solve).
- Unrelaxed quality: median energy released on relaxation 42.5 -> 42.4 kJ/mol (no change).
- Compared with the first (buggy) run, three gains are gone: CAXTAX s12, OMAGAK s12, TIKZOF s13. Their hitting rows were
  reflections that the production dedup counts as duplicates of other (non-source) kept rows (checked with
  `StructureDeduper.any_dup` on the first-run pools: CAXTAX row 7 = row 14, OMAGAK row 1 = row 2, TIKZOF rows 1 and 38
  = rows 39 and 71), so those gains were relax-set selection effects, not new basins.

### First full-panel run (superseded: mirror kept duplicate reflections)
C-MISS 0.333 -> 0.279 (d -0.054 [-0.103, -0.011]) under the contract rule, 25 gains / 1 loss, new timeouts CIFZEZ, XAQCAT,
HEDJOP. NUWFAM s13 loss: the control's hit row (row 27) is in the arm pool (row 54 and its reflection 55, same energy
and score); the larger pool changed the <= 5-row A4c relax set, which no longer took it. XAQCAT cost (local 2060): the
mirror dedup over 2n rows took 17-20 s per seed on a large automorphism group, plus MACE on reflected rows (80 -> 142 s).

### D1 `beam_core_share` 0.5 vs control (paired, 123 units, 41 parents)
- C-MISS 0.585 -> 0.537, d -0.049 [-0.122, +0.008]; C-TOP1 0.276 -> 0.268, d -0.008 [-0.041, +0.016].
- miss -> hit 11 units, hit -> miss 5, all on partial-miss parents (seed-level noise) plus NOYTUS (3/3).
- None of the 13 wrong-basin parents moved. Reading: diversity of core modes in the beam does not reach the missing
  basins; not pursued further as is.
- Unrelaxed quality: median energy released on relaxation 55.8 -> 57.2 kJ/mol (no change).

### D1 `out_mirror` vs control (paired, 123 units, 41 parents; same solves, mirror rows added)
- C-MISS 0.585 -> 0.472, d -0.114 [-0.195, -0.041]; C-TOP1 unchanged (0.276; reflected rows keep their source score).
- miss -> hit 14 units (LIPDEU 3/3, NOYTUS 3/3, YOSXIP 2/3, WEYJIT s13, CAXTAX, HOYBED, TIKZOF, XOJHIN, YOVNEC); hit -> miss 0.
- ZOJDEH (enantiomer-only on 128/128 starts) is not recovered: its reflected rows are the class but not at hit level.
- Released energy median 55.8 -> 53.2 kJ/mol. Post reused 141 of 561 relaxed rows from the control post (X0_sha).

### Smoke (local 2060, seed 11) `embed_ring_alt` (unrelaxed MACE dE to the reference, kJ/mol)
| Parent | top-1 control -> ring_alt | floor control -> ring_alt | rows |
|---|---|---|---|
| FEKKAJ | 155.7 -> 137.8 | 155.7 -> 59.5 | 144 -> 288 |
| HOHZAG | 247.7 -> 220.4 | 247.7 -> 220.4 | 8 -> 20 |
| ZAQNUE | 127.5 -> 127.5 | 127.5 -> 84.7 | 16 -> 32 |

### D2 `embed_ring_alt` vs D1 control (paired, 123 units, 41 parents; same node type, separate solves)
- C-MISS 0.585 -> 0.520, d -0.065 [-0.146, +0.000]; C-TOP1 0.276 -> 0.293, d +0.016 [+0.000, +0.049].
- miss -> hit 8 units: FEKKAJ 3/3, HOHZAG 3/3, DOHTOM 2/3; hit -> miss 0. ZAQNUE and EZIBUL improve in energy but
  not to hit level.

### Donor-adjacent rotor misses: lineage and reachability (DUHHIX, CIXYUD, CAQLEP, EFOXII, NUCKUS, BABNAV; seed 11-13)
- Lineage (`a4c.lineage_class`, all starts of every pass): no start of any outcome (kept, merged, beyond_cap) is the
  reference class within 0.44 A core RMSD of the unrelaxed reference; most are 0.6-1.1 A. Culling and allocation are
  not the cause.
- Reachability (`a4c.reach`, seed 11: production starts fitted to the reference, fit 0.01-0.31 A heavy RMSD, then the
  production solve with culling off):

| Parent | solved from the reference: core RMSD (A) | best score vs pool top-1 |
|---|---|---|
| DUHHIX | 0.41 (6/16 in class; other site assignments leave it) | 0.274 vs 0.305 |
| BABNAV | 0.57-0.67 | 0.702 vs 0.705 |
| NUCKUS | 0.60-0.77 | -0.412 vs -0.447 |
| EFOXII | 0.82-0.87 (the pool's basin) | 2.064 vs 2.065 |
| CAQLEP | 0.86-1.11 | -0.138 vs 0.361 |
| CIXYUD | leaves the class (2/16 in class, 2.4-3.6 A) | -0.735 vs 0.670 |

- Reading: for 5 of 6 the score has no maximum at the reference basin; ascent from the reference moves 0.6-1.1 A away
  (EFOXII onto the pool's own basin). This group is a score-surface limit, not search: no search or allocation change
  reaches it. DUHHIX holds near the basin (0.41 A) but scores it below the pool's top-1. Belongs to the score track
  (NN correction: environments), not to jgen search.
- Caveat: the fit inherits the embed's bonds, angles and rings, so a drift can also come from embed internals rather
  than the score; only EFOXII and DUHHIX have clean fits (<= 0.05 A heavy). Clean evidence of a score-surface limit is
  therefore EFOXII (and DUHHIX's misranking); for the other four the reading is "score surface or embed internals".

## Fixes made on the way (research tooling)
- Bench arm packing reused the control's beam seeds for any arm without `start_kabsch`/`beam_union`: a beam flag had no
  effect in a packed run. Fixed for `beam_core_share` (armpack and bench run).
- Bench records stamped the fingerprint from module state left by the previous arm (96/123 records of D1 carried the
  mirror arm's fingerprint; solves unaffected). Fixed; the D1 records were restamped (`STAMP_FIX.txt` in the run).
- `relax_run` resume matched rows by (code, seed, row index) only; another arm can hold another structure there. Rows are
  now reused only when the unrelaxed geometry hash matches (`X0_sha`).
- `a4c.retro` read only `*_control_*` pool manifests (every non-control arm failed with StopIteration).
- `out_mirror` first version kept reflections that duplicate another kept row (the mirror was tested only for
  self-coincidence, not against the other rows; on achiral or symmetric parents a reflection is often another row).
  This doubled pools and changed the <= 5-row relax sets. Fixed with `StructureDeduper.any_dup` (the same pair decision
  as the production dedup `select`, per regime; unit-tested against `select`). The first full-panel run used the buggy
  version and is superseded by the rerun above.
- `rerun post --mace-device cuda`: the CPU MACE pool on a login node took > 25 min per arm and was killed by the
  timeout (three posts lost, about 25 min); on the login A100 a post takes about 5 min.

## Open items
- **Q5 reflection variant (score track, small):** the score is not mirror-invariant, and the difference is all Q5.
  Measured per term on seed 11, source vs reflection: Q1, Q2, S_orient (reported in the "Q3" slot; Q4 is 0 in
  production) and Q6 change by 0.00000. Q5 changes by up to 0.055 on CIFZEZ (mean 0.029) and 0.017 on BOTYUH.
  Cause: reflection flips the sign of every dihedral. The fitted Q5 von Mises mixtures are not symmetric under
  phi -> -phi, while the torsion keys (atom types) carry no handedness. Possible fix: symmetrise the Q5 mixtures in
  the fit (each component mirrored at -mean, weight split). That is a DB change, so it goes in src/fit with a
  fold-in, and it is not done. Until then reflected rows keep their source's score: the energy is identical, and
  rescoring let this error decide top-1 (7 units on the full panel).

Deferred (owner): UYOVUA Li-cap regression.

### 2026-11-13: are the same-class gaps real? (check on the 65 "right class, wrong basin" misses, arm cfg)
- Relaxation is complete: all 65 best same-class rows relaxed with status ok and converged in 6-36 cycles (cap 200).
  All references converged in 1-2 cycles, so the crystal references are already GFN2 minima.
- The gaps are not a MACE-on-GFN2 artefact. The GFN2 offset (row E_h minus ref E_h) is larger than the MACE offset:
  median +21.2 vs +14.0 kJ/mol. Band +15-50 (MACE): 27/27 are also > +5 at GFN2 (median +29.7). Band +5-15: 28/34 are
  also > +5 at GFN2 (6 <= 5; FETTAZ s12 -19.9 and EFOXII s11 -2.1 are lower at GFN2).
- These are distinct conformers of one configuration, not near-duplicates. 29/65 have core RMSD > 1 A (chelate or
  macrocycle folds, donor-arm orientation: GOTWET 2.8, ZEFXAM 2.9, NUCKUS 2.6, CAQLEP 2.0). All 65 keep the donor set
  and bond topology after relaxation (A4c donors_same, topology_same). The old A4b `contact_change` flag (19/65) only
  means a non-donor heavy atom crossed the loose rcov+0.6 A window around the metal between the row's unrelaxed and
  relaxed geometry; it is not a donor or class change.
- The > +50 rows are DUHHIX (+98 to +105 MACE, +110 to +136 GFN2) on all three seeds.
