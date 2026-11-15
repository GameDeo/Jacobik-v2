# Track F: embedding study

**Status:** F-3 checkpoint (F-2 accepted). The F-1 label and detector results are superseded by the F-2 symmetry-aware frozen-coordinate analysis. F-3 adds a binary transition re-score, the ZOLQIC embed fallback, and an off-by-default bond guard.

## Scope and execution

The study uses the predeclared F-1 contract and the F-2 contract in `research/bench/tools/jgen/embed_study/`. Cohorts exclude `val`, reserve, and locked parents. tmQM access was limited to SMILES, ligand graph, and geometry. All study code is under `research/bench/tools/jgen/embed_study/`; there are no `src/` edits. CPU-only, at most three concurrent processes, no GPU.

The corrected census contains 391 parent records, 702 ligand-variant rows (701 successful), and 683 source ligand instances: 384 cal/tail and 299 valid surface-fit sample ligands. ZOLQIC is the one failed production embed. Results are stored in the ignored `research/data/jgen/campaign/F/` outputs (`census_f2.jsonl`, `analysis_f2.json`, `arms_f2.json`, `score_f2.json`).

## F-2 fixed-coordinate label

Atom mapping now searches ligand graph automorphisms preserving atom/bond properties and specified stereochemistry, selects the lowest heavy-atom RMS after rigid alignment, and assigns equivalent hydrogens locally to minimize bond/angle residuals. The selected mapping is used throughout metric calculation. Seven variant rows across six ligands reached the 4,096-automorphism cap; those rows are flagged in the raw census, so their mapping can be incomplete in unusually symmetric cases.

The primary label includes covalent bond lengths, non-DOF angles (hydrogen angles reported separately), trigonal `sp2` planarity and stereo/E/Z after production variant enumeration. Thresholds are 0.15 Å, 15°, 20° and mismatch respectively. Donor-pair distances are a separate chelate-bite diagnostic and are dropped from this embed-only score because no torsion fit is available. Ring pucker/torsion and non-planar-center impropers are not frozen coordinates.

The corrected reference-bad count is **30/683 (4.4%)**: 21/384 cal/tail (5.5%) and 9/299 surface-fit (3.0%). Component counts are bond 5, non-DOF angle 12, `sp2` planarity 0, and stereo 14 ligands (one overlap; 30 unique). Thus the remaining bad share is driven mainly by stereo/E/Z and non-DOF angle discrepancies, with a smaller bond component; planarity contributes none. Donor-pair shifts no longer inflate the label.

| Class | Bond median / p95 (Å) | Heavy-angle median / p95 (°) | H-angle median / p95 (°) | `sp2` planarity median / p95 (°) |
|---|---:|---:|---:|---:|
| Monodentate kappa | 0.0018 / 0.0333 | 0.649 / 6.241 | 0.285 / 1.995 | 0.279 / 2.117 |
| Multidentate kappa | 0.0032 / 0.0334 | 0.742 / 4.937 | 0.384 / 1.948 | 0.356 / 2.293 |
| Haptic | 0.0040 / 0.0482 | 0.762 / 5.684 | 0.437 / 1.784 | 0.757 / 4.762 |

The 36–50° improper p95 values were **not symmetry swaps**: symmetry-aware p95 values are 50.66° (monodentate), 37.23° (multidentate), and 36.04° (haptic), essentially unchanged from identity mapping. Of 179 improper residuals over 20°, 173 are at `sp3` centers and only six at `sp2` centers. The large tail is therefore mostly a statistic on centers where a planar improper is not an appropriate frozen coordinate; the `sp2`-only planarity p95 is at most 4.76°.

### Re-scored reference-free detectors

| Detector | TP | FP | FN | TN | Sensitivity [parent bootstrap 95% CI] | Specificity [parent bootstrap 95% CI] | Precision | Binary AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (b) empirical coordinate ranges | 28 | 358 | 2 | 295 | 0.933 [0.828, 1.000] | 0.452 [0.406, 0.493] | 0.073 | 0.693 |
| (c1) bonded-distance anomalies | 2 | 0 | 28 | 653 | 0.067 [0.000, 0.172] | 1.000 [1.000, 1.000] | 1.000 | 0.533 |
| (c2) nonbonded clashes | 0 | 0 | 30 | 653 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | — | 0.500 |
| (c3) lost stereo/E/Z | 0 | 0 | 30 | 653 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | — | 0.500 |
| (c) graph consistency union | 2 | 0 | 28 | 653 | 0.067 [0.000, 0.172] | 1.000 [1.000, 1.000] | 1.000 | 0.533 |
| (b) ∪ (c) | 28 | 358 | 2 | 295 | 0.933 [0.828, 1.000] | 0.452 [0.406, 0.493] | 0.073 | 0.693 |

The range detector recovers most corrected-label cases but has low precision and specificity on this rare-positive cohort. Graph consistency is specific but misses most bad labels. AUROC is based on binary predictions and adds no ranking information.

## Named cases

### QEZVEY: method failure at GFN2 refinement

The input is `CC(C)(C)C(=[N-])C=C([NH-])C(C)(C)C`, net charge −2. It parses with valid valence, and the bound ligand graph has a 1–4 C–C single bond of 1.530 Å. No unambiguous resonance correction applies to this nonconjugated tert-butyl-to-imine bond. Production-path re-embedding gives 1.485 Å after ETKDG, 1.532 Å after MMFF, 1.544 Å after chelate-aware ETKDG/MMFF with two caps and one donor pair, then **2.105 Å after GFN2 refinement**; production returns that GFN2 geometry. The bad bond is therefore a method failure at GFN2 refinement, not an input-artifact correction. Donor restraint/capping does not create the elongation.

### ZOLQIC: production solve raises

A full CPU `ComplexBuilder.solve` was run at production fingerprint `615ae97d2350`, with the frozen constraints DB supplied. It raises after about 10 seconds: `ValueError: Bad Conformer Id`. The route is `_build_shell` → `_build_complex_shell` → `_smiles_to_conformer`, at `AllChem.MMFFOptimizeMolecule(mol_h)` after ETKDG left no conformer. The caller receives the exception and no `EnsembleResult`; this violates the generator's graceful-degradation requirement. No source change was made under this study.

### NEPNEF and resonance mis-writes

The production-path corrected-SMILES re-embeds confirm the same input-artifact class in four parents: NEPNEF, BOHWUT, KENMAU and SUXVEN. The pattern is neutral `N([O-])[O-]`, corrected to `[N+](=O)[O-]`. The cohort scan finds **9 eligible malformed nitro centers across 4 ligand instances and 4 parents**: seven centers in the cal/tail cohort and two in the surface-fit sample. Re-embedding removes the large N–O error in all four ligand instances. Other nitrate, C–O, P–O, S–O, and N–O graph motifs were counted as candidate resonance groups, but their input forms are not unambiguous from this scan alone and are not counted as normalization targets.

The design-only normalization proposal is a SMILES preprocessing rule for unambiguous resonance mis-writes, including `N([O-])[O-]` → `[N+](=O)[O-]`, gated by element, valence, charge and local-bond checks. Scope is 9 centers / 4 ligand instances / 4 parents for this cohort scan. It is **not implemented**.

### NEPNEF shell trace

The bounded trace did not reproduce a 0.56 Å Zn–donor event. `_build_shell` places parsed donor groups at 2.50 Å from Zn but places non-donor carbon 15 at 0.484 Å. Across 512 Kabsch-initialized candidate rows, the closest parsed donor is 1.146 Å; none of the first 200 IK score evaluations puts a parsed donor below 1.0 Å. The observed overlap is already in shell placement and involves carbon, not a donor. Record the 0.56 Å report for the A-track donor-distance census and Gemini's `start_kabsch`; stop this trace here.

## Alternative-method arms

Research-side implementations live in `research/bench/tools/jgen/embed_study/alternatives.py`; the comparison runner and paired scorer are in the same directory. Arms use seed 42 and matched inputs, clear the in-process GFN2 cache before each ligand/arm timing, and run with three CPU workers. The 28-ligand failure set contains the named cases, uncorrected malformed-nitro ligands and corrected-label bad ligands from the 20 worst cal/tail parents. The control is 40 corrected-label-good, scoreable cal/tail ligands. Failed embeds receive the predeclared score penalty of 10 normalized units.

| Arm | Failure-set failure rate | Failure-set median s/ligand | Good-control failure rate | Good-control median s/ligand | Failure-set paired Δ [95% parent-bootstrap CI] | Good-control paired Δ CI | Combined time ratio | Wins? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Production (control) | 1/28 (3.6%) | 0.599 | 0/40 (0%) | 0.125 | — | — | 1.000 | — |
| Fixed-metal bound-state GFN2 | 5/28 (17.9%) | 1.346 | 8/40 (20.0%) | 0.367 | +0.00428 [+0.00036, +0.02483] | [+0.00927, +0.09416] | 2.894 | No |
| Donor-restrained GFN2 | 3/28 (10.7%) | 0.589 | 2/40 (5.0%) | 0.086 | 0.00000 [0.00000, +0.00045] | [0.00000, 0.00000] | 0.768 | No |
| GFN-FF preopt → GFN2 | 3/28 (10.7%) | 0.658 | 0/40 (0%) | 0.112 | +0.00077 [−0.00117, +0.00241] | [−0.00342, +0.00128] | 0.860 | No |
| Li⁺ at anionic-donor H-cap site | 3/28 (10.7%) | 0.904 | 0/40 (0%) | 0.134 | −0.00574 [−0.01402, 0.00000] | [0.00000, 0.00000] | 1.126 | No |

Δ is arm minus production in median normalized frozen-coordinate error; the time ratio uses the combined failure and control evaluation set, as frozen in `F2_CONTRACT.md`. Li⁺ has the best point estimate, but the failure-set CI reaches zero, so it does not meet the strict winner rule. Bound-state misses on failure-set error, control quality and time. Donor-restrained and GFN-FF do not establish a negative failure-set CI. **No alternative wins.**

## Strain share and limits

Strain share is dropped from F. The A-track reachability fit measures unreachable strain directly; the tracks will cross-check on shared parents later. No strain rebuild was run, and neither a compute nor engineering budget was spent on it in F-2.

The bounded NEPNEF trace stopped at the requested stage; the original 0.56 Å donor observation remains unreproduced. Seven highly symmetric variant rows reached the automorphism cap. ZOLQIC's full production solve raises as described above. Alternative failure-set sample sizes are 28; good-control sample sizes are 40.

## Checkpoint F-3 — fixes and direct re-score

This checkpoint follows the contract committed as `119fc8a`, after merge `8a248e3` of master `8dbe29c`. The re-score below reads the existing F-2 arm records only; no alternative-arm embeddings were rerun.

### Binary label transitions in the F-2 arm records

The fixed 28-ligand failure set has 23 production-label-bad ligands, four production-label-good ligands, and ZOLQIC, which has no label because production embedding failed. A repair is counted only when a production-bad ligand becomes label-good. A control regression is counted only when a production-good control becomes label-bad.

| Arm | Failure ligands repaired (of 23 baseline-bad) | Repaired ligands | Good controls becoming bad (of 40) | Failed failure rows | Failed control rows |
|---|---:|---|---:|---:|---:|
| Production control | 0 | — | 0 | 1 | 0 |
| Fixed-metal bound-state GFN2 | 4 | FIMCAH:0, IYUKOC:0, YABYEJ:1, ZAFPOP:0 | 13 | 5 | 8 |
| Donor-restrained GFN2 | 1 | IYUKOC:0 | 0 | 3 | 2 |
| GFN-FF preopt → GFN2 | 3 | CUGXOT:0, FIMCAH:0, IYUKOC:0 | 0 | 3 | 0 |
| Li⁺ donor cap | 6 | FIMCAH:0, IYUKOC:0, QEZVEY:2, VUPMEY:0, YABYEJ:1, ZAFPOP:0 | 0 | 3 | 0 |

The 13 bound-state control regressions are CIFZEZ:2, CIXYUD:1, EFOXII:2, FASPOH:3, LUQVOK:0 and :3, NUHPEL:2, PIWHAF:4, RIQDEB:1, XOJHIN:1 and :2, XUHMIY:0, and YISQUM:2. These transition counts complement the F-2 per-ligand median-error rule; the earlier “none wins” remains limited to that continuous metric.

Named failure-set ligand labels from the existing arm records (`good`, `bad`, `fail`, and `n/a` = unchanged/not applicable arm):

| Parent:ligand | Production | Bound-state | Donor-restraint | GFN-FF | Li⁺ cap |
|---|---|---|---|---|---|
| QEZVEY:2 | bad | bad | bad | bad | good |
| HEDJOP:0 | good | bad | good | good | n/a (good) |
| BOHWUT:2 | good | bad | fail | good | n/a (good) |
| KENMAU:2 | good | bad | good | good | good |
| NEPNEF:0 | bad | bad | n/a (bad) | fail | fail |
| NEPNEF:1 | bad | bad | n/a (bad) | fail | fail |
| SUXVEN:1 | good | good | n/a (good) | good | good |
| ZOLQIC:1 | fail | fail | fail | fail | fail |

JEGPUJ and WEYJIT were not in the fixed failure/control sets and therefore have no F-2 alternative-arm labels.

### ZOLQIC no-raise behavior

The original ETKDGv3 call remains first and unchanged. On failure, `_smiles_to_conformer` now tries `useRandomCoords=True` with seeds 43–47 (for the production seed 42). ZOLQIC fails all six attempts. The solve catches that ligand-embed failure and returns `EnsembleResult(conformers=[], best=None, failure="ETKDG failed ...")`; no exception escapes. The E/Z aggregation path also returns an empty failed ensemble if all variants fail.

Both production and `u1` golden checks pass: 245 arrays per profile, zero differences (the golden tolerance is 1e-5 Å for coordinates and 1e-6 for scores). The Runkit trope suite was rerun against the saved `8dbe29c` control: **9/9 parents bit-identical**, zero outside-footprint violations. ZOLQIC is not one of those nine suite parents, so its behavior is covered by the direct full-default `ComplexBuilder.solve` reproduction above. The default profile fingerprint remains `615ae97d2350`.

### `embed_bond_check` flag

Registered in `profile.FIELDS` and read at `jacobik.engine` import. The default is `False`; the default fingerprint remains `615ae97d2350`. When enabled, after GFN2 refinement the guard rejects the refined geometry if any bonded distance changes by more than 0.25 Å or exceeds 1.25 × the RDKit covalent-radius sum. It returns the pre-GFN2 MMFF geometry and records the offending bonds; it never rejects a ligand. A targeted QEZVEY check caught the 1.544 → 2.105 Å C–C change and returned the 1.544 Å pre-refinement geometry.

Footprint scan used the existing 701 successful F-2 variant rows (same seed, production chelate route); no new parent set was selected. It had zero scan errors and found 9 triggering ligand variants / 12 offending bond records in 8 unique parents:

| Cohort | Parents scanned | Trigger parents | Trigger parent codes |
|---|---:|---:|---|
| Calibration panel (includes tail subset) | 120 | 6 | BOCRAN, EBAYIQ, LOPCAW, NUWFAM, QEZVEY, ZAFPOP |
| Tail subset (overlaps calibration) | 5 | 0 | — |
| Surface-fit sample | 271 | 2 | FIMCAH, OHECUZ |
| Unique union | 391 | 8 | BOCRAN, EBAYIQ, FIMCAH, LOPCAW, NUWFAM, OHECUZ, QEZVEY, ZAFPOP |

The footprint’s paired screen used one full production solve per arm and parent with seed 11, flag off/on, same host (`meep`), CPU-only, and the Runkit top-1 strain / `ΔE_unrel` calculation. Every paired solve returned an ensemble. EBAYIQ's GFN2 relaxation exited with `rc=1` in both arms, so its strain is unavailable; its unrelaxed energy remains measurable.

| Trigger parent | Top-1 strain control → guard (Δ), kJ/mol | `ΔE_unrel` control → guard (Δ), kJ/mol |
|---|---:|---:|
| BOCRAN | 31.23 → 295.88 (+264.65) | 57.65 → 321.08 (+263.43) |
| EBAYIQ | unavailable → unavailable (xTB relax rc=1 in both) | 117.91 → 209.26 (+91.36) |
| FIMCAH | 425.26 → 343.83 (−81.43) | 425.78 → 326.12 (−99.66) |
| LOPCAW | 164.24 → 572.56 (+408.31) | 163.36 → 570.57 (+407.21) |
| NUWFAM | 154.15 → 154.15 (0.00) | 186.36 → 186.36 (0.00) |
| OHECUZ | 385.26 → 97.35 (−287.91) | 400.90 → 115.59 (−285.31) |
| QEZVEY | 375.44 → 259.57 (−115.87) | 372.61 → 292.16 (−80.46) |
| ZAFPOP | 17.04 → 17.04 (0.00) | 37.44 → 37.44 (0.00) |

Across the seven parents with paired strain, median Δstrain is 0.00 kJ/mol (range −287.91 to +408.31); across all eight, median ΔE_unrel is 0.00 kJ/mol (range −285.31 to +407.21). The top-1 screen is mixed and parent-specific: the guard can improve or worsen these top-1 metrics substantially. The zero medians do not imply no tail movement.

### Unassigned tetrahedral stereo design

Counted RDKit-unassigned tetrahedral centers per unique source ligand SMILES. The tail is a 5-parent subset of the 120-parent calibration panel and overlaps those calibration counts.

| Cohort | Ligands | Ligands with ≥1 unassigned center | Distribution `k: ligand count` | Naive sum of `2^k` variants |
|---|---:|---:|---|---:|
| Calibration panel | 384 | 11 | 0:373, 1:5, 2:4, 6:2 | 527 |
| Tail subset (overlapping) | 12 | 2 | 0:10, 2:1, 6:1 | 78 |
| Surface-fit sample | 299 | 12 | 0:287, 1:8, 2:4 | 319 |
| Unique calibration + sample | 683 | 23 | 0:660, 1:13, 2:8, 6:2 | 846 |

Per-ligand counts for every nonzero case (`k` unassigned centers; `2^k` raw assignments; all other listed-cohort ligands have `k=0`):

| Cohort | Ligand: `k` (`2^k`) |
|---|---|
| Calibration | BOTYUH:2 `1 (2)`; DAMVAQ:1 `6 (64)`; FUTSOD:0 `6 (64)`; JUPPOZ:0 `1 (2)`; LISVOC:2 `2 (4)`; LIPDEU:1 `2 (4)`; PANBOY:0 `2 (4)`; QUGXUN:1 `1 (2)`; USAPOU:0 `2 (4)`; YOSXIP:0 `1 (2)`; ZOJDEH:5 `1 (2)` |
| Surface-fit sample | AVUVET:1 `1 (2)`; EZISIR:0 `1 (2)`; DOLVOR:0 `1 (2)`; GOFDAG:2 `2 (4)`; KATRII:0 `1 (2)`; KOPSEN:0 `2 (4)`; LUZLID:0 `1 (2)`; OLAJIU:0 `1 (2)`; RALVAD:0 `1 (2)`; TAXJOS:0 `2 (4)`; TUXSAG:1 `1 (2)`; ZUXCIE:2 `2 (4)` |
| Tail subset (overlaps calibration) | DAMVAQ:1 `6 (64)`; USAPOU:0 `2 (4)` |

No ligand exceeds six unassigned centers in these cohorts. Proposed design only: fully enumerate through 8 centers (at most 256 assignments); above that, prioritize assignments covering the metal-donor stereocenters and use a stable-hash, stratified sample capped at 256 assignments. No stereo enumeration was implemented.

## Checkpoint F-4 — package and embed screens (2026-10-28)

### Spec issues and implementation

- The F-3 audit expected deterministic retry behavior after ETKDGv3 failure, but `chelate_embed.embed` still called `AllChem.ETKDG()` (random seed −1). F-4 replaces that last fallback with five seeded ETKDGv3 random-coordinate attempts (`seed+1` through `seed+5`). Exhaustion returns `None` so the engine can use its plain route; it does not leave an empty conformer for later code to dereference. This is an additional default-path correction, limited to the retry path.
- The raw F-2 census has 684 unique `(parent, ligand)` attempts, not 683: 683 source ligands are scoreable and ZOLQIC is the retained failed attempt. The F-4 contract was amended before the label screen to state both denominators.
- `embed_bond_check` now triggers only on the specified stretches. It retries once from the MMFF geometry with the offending bond lengths constrained alongside the donor-pair constraints, then returns MMFF if that retry fails or still violates the guard. It records `bond_constrained_gfn2`, `pre_refinement_after_constrained_violation`, or the constrained-retry failure reason. It never rejects a ligand.
- Added `embed_li_cap` (default `False`), read at engine import and included with the guard flag in the in-process GFN2 cache key. It adds one temporary Li per negatively charged singleton κ donor, at 1.9 Å along the open-valence direction, uses `q_ligand + n_Li` and the electron-count-derived `uhf`, leaves donor constraints unchanged, and strips Li before Kabsch alignment. The bound-surrogate route explicitly bypasses both experiment flags.

### Footprint and embed-level screen

Before any solve, the anionic κ-donor footprint is 226 ligands / 111 parents in cal, 5 / 4 in tail, and 166 / 161 in the surface-fit sample. Tail is contained in cal. The unique union is **392 ligands / 272 parents**. There are 20 κ donor atoms with charge −2 or lower, in 20 ligands / 16 parents: AZUSIA:0,3; BOLGIU:0,1; BUXJIO:3; CAXTAX:2; FEKKAJ:4 (−3); FUTSOD:1; JAYCEV:2; NUCKUS:0; SESWIZ:1,2; VOBGUR:1,2; XEYSUP:1; EGETOA:2; EKABIE:2; FEZPOS:3; OHECUZ:4; TAHWIK:2. Each receives one Li.

The frozen-coordinate label screen used all 684 attempts (683 scoreable; ZOLQIC failed in every arm), three CPU workers, and seed 42. Repairs / regressions were counted against the 653 baseline-good ligands attempted by each arm. The gate permits at most 6 regressions (1%); all arms have zero regressions and pass Step 3.

| Arm | Bad → good | Good → bad | Regressions / 653 | Mean / median seconds per ligand | Max seconds |
|---|---:|---:|---:|---:|---:|
| Production | 0 | 0 | 0% | 0.878 / 0.186 | 60.52 |
| Stretch guard | 3 | 0 | 0% | 0.736 / 0.138 | 51.33 |
| Li cap | 8 | 0 | 0% | 0.743 / 0.134 | 50.21 |
| Guard + Li cap | 8 | 0 | 0% | 0.750 / 0.136 | 51.01 |

The stretch-only scan matches the F-3 expectation exactly: the five trigger parents are **FIMCAH, NUWFAM, OHECUZ, QEZVEY and ZAFPOP**; BOCRAN, LOPCAW and EBAYIQ drop out. All five guard fallbacks use constrained GFN2. In the combined arm, Li changes the GFN2 trajectories: only NEPNEF and NUWFAM trigger. NEPNEF ligand variants 0 and 1 hit `FileNotFoundError` on the constrained retry and returned MMFF; NUWFAM uses constrained GFN2.

Named ligand labels (good/bad): QEZVEY:2 stays bad in every arm; OHECUZ:4 changes bad→good under guard and Li alone, but remains bad with the combined arm; FIMCAH:0 changes bad→good under guard, Li and combined; BOCRAN ligands 0 and 1 and LOPCAW:0 stay good; EBAYIQ:0 stays bad. Other repaired source ligands are ZAFPOP:0 (guard and all arms with Li), IYUKOC:0, VOBGUR:0, VUPMEY:0 and YABYEJ:1 (Li and combined).

Li-cap GFN2 failed for FIKTUP:0 in both Li arms (`xtbopt.xyz` was absent); the production code retained MMFF and the label remained scoreable. ZOLQIC is the only failed label/embed row in all four arms. The complete 684-row transition table is in the ignored artifact `research/data/jgen/campaign/F/screen_f4_transitions.json`.

ETKDGv3 retry-path use was found twice: MEBHEK:0 in cal (deterministic random-coordinate retry succeeded at seed 45) and QUGFUV:2 in the surface-fit sample (seeds 43–47 exhausted in chelate embedding, then the plain route succeeded). No retry was used by the five-parent tail subset. ZOLQIC:1 remains the failed source attempt.

### Stereo design

The 23 ligands with unassigned free-ligand centres overlap **11 of the 14** F-2 stereo/E-Z mismatch ligands. A graph-rank count finds 14 N donor atoms with three distinct ligand substituents that become stereogenic when the metal is the fourth substituent; they occur in 10 ligands across 9 parents. No P, S or As examples were found. Recommendation: implement this later as a coverage feature; the overlap is material, but enumeration and macrocycle feasibility need a separate design and gate. No stereo code was added.

### Solve-level and robustness screens

Step 4 used full production solves, seed 11, on one Colab L4 for every arm and its paired control. The two-parent estimate covered QEZVEY and NUWFAM for control and each arm (solve time 14–39 s per parent, plus MACE); the full control union took about 12 minutes. Seeds 12–13 were not added. No GFN2 relaxation was used in this screen. All parents returned finite top-1 energies: control 112/112, guard 3/3, Li cap 111/111, combined 112/112. The arms were paired against the same-machine control over their predeclared cal/tail footprints.

Paired differences below are arm minus control. CIs use 10,000 parent-cluster bootstrap resamples (seed 20261028). Negative values favor the arm for each metric.

| Arm | Parents | Top-1 Δ mean (median), 95% CI for mean | Floor Δ mean, 95% CI | Obscene-rate Δ mean, 95% CI | Arm top-1 RMSD mean / median (Å) | Worst-slot energy mean / median (kJ/mol) |
|---|---:|---|---|---|---:|---:|
| Stretch guard | 3 | −39.54 (−1.01), [−117.54, −0.08] | −51.05, [−152.80, +0.0002] | −0.167, [−0.500, 0.000] | 2.89 / 2.77 | 779.52 / 495.67 |
| Li cap | 111 | −21.37 (−0.21), [−38.58, −9.74] | −22.88, [−37.99, −12.62] | −0.0284, [−0.0490, −0.0103] | 1.73 / 1.60 | 559.45 / 185.16 |
| Guard + Li cap | 112 | −21.19 (−0.10), [−38.38, −9.84] | −22.83, [−37.66, −12.69] | −0.0280, [−0.0480, −0.0099] | 1.74 / 1.63 | 567.82 / 187.53 |

The paired top-1 median bootstrap CIs are guard [−117.54, −0.08], Li [−5.53, −0.0002], and combined [−5.53, −0.0001] kJ/mol. Guard does not meet the secondary-improvement condition: its floor CI includes zero and its obscene-rate CI reaches zero; it also repaired no Step-4 named failure. Li and combined meet the numerical conditions: their paired-mean upper bounds are below +2 kJ/mol, and both floor and obscene-rate CIs exclude zero. Each has three parents with top-1 worsening above 25 kJ/mol:

- **DUHHIX:** +48.69 kJ/mol; Li-cap donor candidates are ligand indices 0 and 1 (`[Cl-]`) and 4 (an anionic azo-enolate N donor).
- **WEYJIT:** +27.92 kJ/mol; ligand indices 0–3 are `[C-]#[O+]` carbonyl donors.
- **ZOJDEH:** +30.11 kJ/mol; ligand indices 0–4 are `[C-]#[O+]` carbonyl donors.

The solve record reports the Li-cap ligand candidates, but not the exact ligand assignments of the global top-1 conformers, so these are candidate causes rather than isolated causal attributions. The guard had no >25 kJ/mol top-1 regressions.

**Spec issue — gate interpretation:** the predeclared decision rule says an arm passes when it meets the paired-mean condition and improves floor or obscene rate, *or* repairs a named failure without a guard breach. It does not say whether a >25 kJ/mol top-1 regression is a global veto or only bars the named-failure route. Under the literal numeric route, Li and combined pass; under a global-veto reading, both fail because of DUHHIX, WEYJIT and ZOJDEH. I am leaving fold-in eligibility unresolved for the F-4 audit rather than changing the gate after results.

The bench robustness batch ran ZOLQIC and SUQWEG together: ZOLQIC was recorded failed (`KeyError: 'v'` in setup/prepare after the empty embed result), SUQWEG completed, and the bench runner finished with exit 0. The two-parent runkit trope batch likewise recorded ZOLQIC as `RuntimeError: atom mapping failed`, recorded SUQWEG as `ok`, and exited 0 after both records were written. The A4 tool was inspected by code only at the absolute `jgen/a-census` worktree path, with no run or edit: `tools.jgen.a4.run` consumes `*_s*.npz` pools through `Watcher`; `analyze_parent` assumes at least one pool (`seeds[0]`). A solve that emits no pool file therefore produces no A4 failure record; the A4 runner does not accept an empty `EnsembleResult` as input.

### Merge note

Production defaults remain `embed_bond_check=False` and `embed_li_cap=False`, with fingerprint `615ae97d2350`. Default behavior changes only for the never-raise ZOLQIC path and deterministic ETKDG retries, including the missed chelate fallback fixed in F-4. No experiment default changed. The F-4 report, full solve outputs and paired analyses are in the ignored `research/data/jgen/campaign/F/` artifacts. Production and `u1` golden checks both pass (245 arrays each, zero differences). The final default-off trope comparison against the Package F source passed 9/9 parents bit-identically, with no outside-footprint violations and fingerprint `615ae97d2350` in both arms. The full L4 output counts were checked before VM shutdown: 112 control, 3 guard, 111 Li cap, 112 combined, two successful parents per estimate, nine records per trope arm, and two records in each ZOLQIC batch check.

The numeric Step-4 conditions are met by Li cap and combined, but the contract leaves unclear whether any >25 kJ/mol top-1 worsening is a global veto or only blocks the named-failure route. The three such cases are listed above. Keep both flags off; no fold-in is proposed until the F-4 audit resolves the gate interpretation. Files touched include `src/jacobik/engine.py`, `src/jacobik/profile.py`, `src/jacobik/chelate_embed.py`, and the F-4 screen, analysis, and contract tools under `research/bench/tools/jgen/embed_study/`. Final checks: `tests/test_profile_hook.py` (9 passed); `py_compile` for modified F-4 code/tools; `git diff --check`; golden checks for both profiles; F-4 full solve/label screens; bench and runkit ZOLQIC batch continuation; and the 9-parent Package F trope comparison.

## F-5 — corrected Li donor rule and fold-in proposal (2026-10-29)

### Frozen rule and donor-class changes

The F-5 decision contract was committed before the corrected arm was evaluated in `F5_CONTRACT.md`. A Li⁺ cap is applied only when the κ donor's formal charge plus the formal charges of its directly bonded atoms is below zero. The audit's gate ruling is applied: a >25 kJ/mol breach is not a global veto, but every repeated breach needs an assignment-based cause, and none may be a donor-rule defect. The cross-platform check uses the mean paired top-1 ΔE_unrel on the same named parents, each averaged over seeds; its 2060 sign check is descriptive, with no bootstrap gate.

| Donor environment | Donors removed from cap | Local charge | Effect |
|---|---:|---:|---|
| CO-like `[C-]#[O+]` | 28 | 0 | Excluded |
| Isocyanide-like `[C-]#[N+]` | 1 | 0 | Excluded |
| Si− adjacent to two N+ atoms (HOYBED) | 1 | +1 | Excluded |

The previous 111-parent footprint contracts to 102 parents and 196 ligand instances (from 226). Eight parents were present only because of CO-like donors: NEQCIV, NOYTUS, PIWHAF, QUTWOT, WEYJIT, XAQCAT, XUHMIY and ZOJDEH. HOYBED is the ninth removed parent, due to the locally positive Si donor. BOCRAN contains the isocyanide donor whose cap status changes, but remains in the footprint through another locally anionic donor. The corrected footprint's tail overlap is DAMVAQ, JEGPUJ, NEPNEF and USAPOU.

### Frozen-coordinate embed screen

The corrected arm was run on the same 684 attempts as F-4, on CPU with three workers. It returned 683 scoreable labels; ZOLQIC remains the one failed source attempt. Against production, the corrected arm repaired 8 of the 30 baseline-bad ligands and regressed 0 of 653 baseline-good ligands. Repairs: FIMCAH:0, IYUKOC:0, OHECUZ:4, QEZVEY:2, VOBGUR:0, VUPMEY:0, YABYEJ:1 and ZAFPOP:0. The embed-screen gate passes.

### Paired unrelaxed solve screen

Control and corrected cap were run on the same Colab L4 for the same 102 parents, seeds 11–13. All 102 parents returned all three seeds in both arms. CIs use 10,000 parent-cluster bootstrap replicates; pooled outcomes first average the three paired seed deltas within each parent.

| Pooled metric | Mean Δ (cap − control) | Median | Parent-bootstrap 95% CI for mean |
|---|---:|---:|---:|
| Top-1 ΔE_unrel (kJ/mol) | −26.04 | −11.88 | [−43.12, −14.62] |
| Floor ΔE_unrel (kJ/mol) | −25.70 | −9.61 | [−41.33, −15.30] |
| Obscene rate | −0.0327 | 0.0000 | [−0.0530, −0.0151] |

Top-1 mean deltas were negative in each seed: seed 11 −25.22 [−44.28, −13.04], seed 12 −23.48 [−38.53, −13.00], and seed 13 −29.42 [−47.47, −16.59] kJ/mol. Floor and obscene-rate CIs also excluded zero in the improving direction for each seed.

Two parents were worse by more than 25 kJ/mol on at least two of three seeds. Their causes come from the top-1 assignment actually used:

- **CAQLEP:** +38.04 (seed 12) and +26.37 (seed 13) kJ/mol. The used top-1 assignment placed the `[CH3-]` donor in ligand 1 at site 3. Its local charge is −1, so this is a genuine anionic methyl cap under the corrected rule.
- **DUHHIX:** +48.69, +70.98 and +51.55 kJ/mol (seeds 11–13). The used assignments include two chloride ligands and the azo-enolate ligand's two donor groups; the cap-bearing ligands are assigned to the coordination sites in each top-1 conformer. Their local donor charges are negative. This is a real assignment-sensitive regression, not a neutral zwitterionic donor admitted by the rule.

No repeated breach is caused by the corrected donor rule.

### RTX 2060 spot check

The three named parents QEZVEY, DUHHIX and NEPNEF were run with control and cap on the local RTX 2060 SUPER, seeds 11–13. Per-parent values average the three paired seed deltas.

| Parent | L4 mean top-1 Δ (kJ/mol) | RTX 2060 mean top-1 Δ (kJ/mol) |
|---|---:|---:|
| QEZVEY | −201.00 | −139.45 |
| DUHHIX | +57.07 | +16.13 |
| NEPNEF | −704.65 | −705.67 |
| **Mean of named parents** | **−282.86** | **−276.33** |

The aggregate sign is negative on both platforms, satisfying the frozen spot-check rule. Each named parent also has the same individual sign on both: QEZVEY and NEPNEF improve; DUHHIX worsens. Local GPU: NVIDIA GeForce RTX 2060 SUPER. The default control fingerprint is `615ae97d2350`; the experiment override fingerprint is `6cc586e17ab1`.

### Decision and proposal

F-5 passes its fold-in rule: the pooled top-1 mean CI upper bound is below zero; floor and obscene-rate CIs exclude zero in the improving direction; both repeated breaches have stated causes from the actual top-1 assignment and neither is a donor-rule defect; and the L4 win retains the same aggregate direction on the named-parent RTX 2060 check.

**Proposal to the owner:** approve folding the corrected `embed_li_cap` rule into production. The default remains off pending approval, and the production fingerprint remains `615ae97d2350`. The experiment fingerprint observed with the flag enabled is `6cc586e17ab1`; confirm the final fingerprint when folding in. After approval, run the production and `u1` goldens into `research/data/golden_<newfp>/`, and one same-machine GFN2 relaxation check on at most 20 parents whose top-1 changes by more than 20 kJ/mol, as the audit specifies. No relaxation was used in this F-5 unrelaxed screen.

### Spec and execution notes

- The F-4 audit resolved the earlier breach-veto ambiguity; F-5 used that ruling.
- “Parents in the old footprint only because of CO-like donors” was interpreted as old-footprint parents with no locally anionic donor remaining. This yields 8 parents; HOYBED is a separate ninth removal.
- “Same direction” for the few-parent RTX 2060 check was otherwise underspecified. The F-5 contract amendment defines it as the sign of the mean paired top-1 delta over the same named parents and seeds; the local sample is descriptive, not a CI gate.
- The first VM was launched from the stale main-checkout Colab helper, whose fingerprint check required `485e943af036`; it was stopped after the check failed (0.22 compute units). The second, correct worktree helper passed `615ae97d2350`.
- The first L4 control launch recorded 102 policy failures because `locked_exclusions.json` is untracked and absent from the Colab bundle; no parent was evaluated. The existing mask was pushed to the VM, and the full control was rerun successfully. This setup failure remains reported separately from the valid paired run.
- Valid Colab `jgF5` usage: start balance 508.39, final balance 506.05, **2.34 units** spent; VM is down. Before shutdown, the fetched outputs were checked: 102 control parents × 3 seeds, 102 cap parents × 3 seeds, 0 solve failures in either valid arm. The failed first control artifact is also retained.
