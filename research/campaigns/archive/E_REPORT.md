# Track E, Package E-1: Q6 donor-pair repulsion and DAMVAQ embedding review

Branch `jgen/e-score` (from master `7d3e4a5`). Checkpoint E-1: no fold-in, no panel runs, no Colab.
Outputs: `research/data/jgen/campaign/E/` (main checkout, gitignored).

## Log

- 2026-10-29: Part A implemented (flags `q6_donor_pairs`, `q6_class4_physical`), light gates run. Part B analysis run (read-only).

## Part A: Q6 flags

### Implementation
- `profile.FIELDS`: `q6_donor_pairs` (module `jacobik.ik.geometry`, attr `Q6_DONOR_PAIR_SCALE`, default None = off) and `q6_class4_physical` (module `jacobik.stats.q6`, attr `Q6_CLASS4_PHYSICAL`, default False). Both read at import.
- `ik/geometry.py`: with `q6_donor_pairs = s`, kappa-kappa donor pairs on different ligands (ligands = connected components of the ligand graph, metal removed) join the Q6 pair list. Haptic face atoms and same-ligand pairs stay excluded. Pairs whose elements lack `rep_pair_params` are left out and logged (`log.info`). Per-pair scale in `ge.q6_donor_scale`, count in `ge.n_q6_donor`.
- `stats/q6.py`: `rep_pair_params(e1, e2, scale=1.0)` moves the onset to `scale * r_m` (A and b unchanged, r_lin recomputed). `build_runtime` assigns the class-5 physical repulsion to donor-extension pairs, and to class-4 non-polar pairs when `q6_class4_physical` is on. Polar-H...acceptor pairs are unchanged.
- Beam (`ik/beam.py`, not edited): the partial-Q6 path takes the pair list and wall parameters from the engine call site, so it would have applied the wall (not the repulsion) to donor-extension pairs. Fixed at the call site in `engine.py` (not the embed): with the flag on, the beam receives only the wall pairs (`ge._q6_wall_idx`). With the flag off the arguments are unchanged (`_q6_wall_idx is None`).
- Scorer: `_q6_pmf` path (repulsive model, Q6P tables) carries the repulsion. Without Q6P tables the scorer uses the wall for every pair, including donor-extension pairs. Production has the tables, so this does not arise in production; it is a stated limitation.

### Gates
- Flags off, fingerprint: `6cc586e17ab1`. Flags on (both): `fc89379a621b`.
- `tests/test_profile_hook.py`: 9 passed.
- `golden.py check` (flags off): production 245 arrays, 0 differ; u1 245 arrays, 0 differ. `golden: OK`. Log: `research/data/jgen/campaign/E/golden_check_flags_off.log`.
- `tests/test_q6_donor_pairs.py` (new, 6 tests): shifted onset is zero at `d >= s*r_m` and positive inside (float64 parameters); `scale=1` equals the default; softer at contact than the full onset; flag off adds no pairs; flag on adds cross-ligand kappa-kappa pairs only (FEKKAJ: 6014 -> 6024 pairs, 10 donor-extension pairs, scale 0.7); class-4 flag keeps the pair list and adds physical rows (2-ligand fixture, FEKKAJ). All 6 pass.
- FEKKAJ pair list with the flag on: 10 donor-extension pairs (Cl-Cl, Cl-P, Cl-C, P-C, C-C among the five donors). Cl2-Cl3 is one of them.

### Crystal check, cal (120 parents), `q6_donor_pairs = 0.7`
Script `research/data/jgen/campaign/E/q6_donor_crystal_check.py`, output `q6_donor_crystal_s07.json`. Uses the solver's own pair list (`ge.q6_donor_scale`) and the runtime `rep_energy`.

| quantity | value |
|---|---|
| parents with donor-extension pairs | 105 |
| pairs | 675 (same count as the auditor's kappa-kappa set) |
| penalty median / p90 / max (kT) | 0.000 / 0.000 / 1.75 |
| fraction of pairs > 0.2 kT | 0.3 % |
| per-parent median (max) | 0.000 (0.036) |

Gate (median <= 0.2 kT): **pass**. Top 5 (pair, parent, penalty, d, d/(s r_m)):

| parent | pair | pen (kT) | d (A) | d/(s r_m) |
|---|---|---|---|---|
| XUHMIY | C52-C53 | 1.75 | 2.29 | 0.92 |
| DOHTOM | C4-N51 | 0.34 | 2.28 | 0.95 |
| QUTWOT | C4-C18 | 0.04 | 2.45 | 0.99 |
| ZOQFOA | O11-O12 | 0.00 | 2.80 | 1.33 |
| ZOQFOA | N12-O14 | 0.00 | 2.96 | 1.34 |

For reference the auditor's unscaled form gave a crystal median of 1.8 kT on the same pairs.

### Crystal check, class-4 (`q6_class4_physical`)
Script `q6_class4_crystal_check.py`, output `q6_class4_crystal.json`. Pair definition as the auditor's rule 2.
- Per-parent sum, 120 parents: median 1.45 kT, p90 5.09, max 21.55 (XUKKIA). **This reproduces the auditor's rule-2 numbers exactly** (validates the check).
- Per pair (17 284 pairs): median 0.000, p90 0.000, max 10.80 (XUKKIA P4-N5, 2.25 A).
- No gate was specified for class 4. The class-4 physical form gives the crystal an offset of about 1.5 kT per parent (the auditor's finding; not changed here).

### FEKKAJ CPU solve, seed 11, `device="cpu"`
Script `solve_one.py`, pinned to the constraints DB `research/archive/data/refit/final/constraints_2026-10-17.json` (the DB the golden check and the pools folder use; the 2026-10-18 DB is refused by the pools folder). No "GFN2 embed refinement failed" in any log (checked by the script, exit code 0 on all six solves).

| arm | top-1 score | Cl-Ru-Cl (deg) | Cl-Cl (A) | n_conf |
|---|---|---|---|---|
| flag off | -0.1634 | 51.8 | 2.02 | 144 |
| `q6_donor_pairs=0.7` | -0.3178 | 65.1 | 2.50 | 144 |
| crystal | - | 154.0 | 4.47 | - |

The flag moves the top-1 Cl-Cl apart (2.02 -> 2.50 A) and lowers its score by 0.155 nats (the new penalty term). Neither arm is near the crystal Cl-Ru-Cl geometry. **The flag does not fix FEKKAJ in this single seed.** Reported as is.

### Two further cal parents (largest kappa-kappa pair count, 15 each; ZOJDEH and SIVBUV)

| parent | arm | top-1 score | metal | Cl-M-Cl (deg) | Cl-Cl (A) | n_conf |
|---|---|---|---|---|---|---|
| ZOJDEH | off | 0.6414 | Cr | no Cl | - | 4 |
| ZOJDEH | 0.7 | 0.6398 | Cr | no Cl | - | 4 |
| SIVBUV | off | 0.6821 | Nb | 96.5 | 3.51 | 5 |
| SIVBUV | 0.7 | 0.6822 | Nb | 96.5 | 3.51 | 4 |

Nothing breaks; scores move by at most 0.0016 nats. (Crystal SIVBUV Cl-Nb-Cl 94.1 deg.)

### Spec issues (Part A)
1. Beam partial-Q6: the directive asked to check whether it reads the same pair list. It does, so the donor-extension pairs would have acquired walls there. Fixed at the engine call site (see Implementation). This is outside `beam.py`, but it is a call-site change in `engine.py`.
2. `_q6_wall_idx` is an empty tensor if every Q6 pair were a donor-extension pair. Not reachable in practice (ordinary pairs always exist), noted only.
3. The donor-extension flag needs the repulsive scorer (Q6P tables). Without them the wall path scores donor pairs; not handled (production unaffected).
4. The solve DB: the directive gives no DB. The 2026-10-18 default is refused by the pools folder's recorded DB, so the 2026-10-17 DB was used (same as the golden check).

## Part B: DAMVAQ embedding review (read-only)

Scripts in `research/data/jgen/campaign/E/`: `damvaq_loader.py`, `damvaq_geom.py`, `damvaq_rdkit.py`, `damvaq_stereo.py`, `damvaq_real_path.py`, `damvaq_gfn2_stereo.py`, `smiles_stereo_scan.py` (+ `.json`), `unlisted_contacts_scan.py` (+ `.json`). Owner's top-1: `campaign/B/g/DAMVAQ_G3a2_P2_s11_top1.xyz` (builder order; mapped to crystal frame by graph isomorphism of the ligand graph, Rh by element; exact).

### 1. Loader donor set, SMILES, neighbours, hybridisation, M-C distances
- SMILES (tmQM, no stereo markers): `C1CCC(P(C2CCCCC2)(C2CCCCC2)->[Rh+3]2<-[CH-]3C4C5CC3[CH-]->2C5C2C3CC5C(C52)C34)CC1`
- Loader donors (`mol.coord_sites`): P (crystal frame 2, 1-based), C11 and C16 (1-based). Three kappa sites. The cage ligand's SMILES donors are atoms 0 and 2 of `[CH-]1C2[CH-]C3C(C2)C1C1C2CC4C1C4C32`, both CH-, degree 2 (heavy), 5-membered rings only (RDKit: SP3).

| atom (1-based) | role | neighbours (non-metal, incl. H) crystal / top-1 | Rh distance crystal / top-1 (A) |
|---|---|---|---|
| P2 | loader donor | 3 / 3 | 2.28 / 2.30 |
| C11 | loader donor (CH-) | 3 / 3 | 2.04 / 2.39 |
| C16 | loader donor (CH-) | 3 / 3 | 2.04 / 2.08 |
| C3 | cyclopropane, NOT a donor | 4 (3 C + H) / 4 | **2.36** / **5.34** |
| C9 | cyclopropane, NOT a donor | 4 (3 C + H) / 4 | **2.36** / **5.30** |

- The cyclopropane carbons (1-based 3, 9; RDKit ring size 3, SP3) are 5-bonded in the crystal (3 C + H + Rh at 2.36 A). The owner's pentacoordinate-carbon observation is confirmed.
- They are not donors because the BO-file Rh bonds are 0.227 and 0.225, below the 0.5 threshold, and the SMILES has no dative bond to them. The donor set is the SMILES dative set mapped to the crystal (`stats/mapping.py` around line 620: `xyz_donor_indices` come from `smi_donor_idx`; raw BO only remaps). The loader therefore gives no donor for them.
- By code reading (`stats/mapping.py`, `raw_metal_bond_exclusion_pairs`): the Rh...C3 and Rh...C9 bonds are still metal bonds that are not H, so their pairs are excluded from Q6. Neither a bonding term nor a repulsive term acts on that contact.

### 2. Owner's face reading
- Confirmed in part: the embedded cage is a **different diastereomer** from the crystal (see 4). The cyclopropane is 5.3 A from Rh in the top-1, against 2.36 A in the crystal.
- **Not confirmed by the plane test as stated.** Plane through P2, C11, C16: crystal Rh at +1.26 A, cyclopropane centroid at +3.27 A; top-1 Rh at +0.08 A (the metal has collapsed into the donor plane), cyclopropane centroid at +3.38 A. The cyclopropane is on the metal side of the donor plane in both. The dot-product test (cyclopropane minus cage centroid, against Rh minus cage centroid) is also positive in both (5.69 crystal, 3.68 top-1). So these tests do not show the cyclopropane on the opposite face. The distinguishing feature is the metal position and the 5.3 A separation, plus the stereo difference below.

### 3. Where the face is chosen
- Free-ligand path (`engine._smiles_to_conformer`, production profile): `bound_surrogate=True` takes the branch first (the cage is embedded with the donor capped, ETKDG with `randomSeed = embed_seed or 42`, then the mapping back onto the ligand as written; the function returns there, so the `chelate_embed` branch is not reached for this cage). Then GFN2 refinement (`_gfn2_refine`). The stereo is fixed at the ETKDG step.
- **The tmQM SMILES carries no stereo markers.** The cage has 10 unassigned tetrahedral centres (RDKit `FindPotentialStereo`). RDKit ETKDG picks one diastereomer per seed.
- Measured on the production path (`damvaq_real_path.py`, ETKDG level, stereo from 3D, compared with the crystal's 3D stereo of the same cage): seed None (= 42, the default) **differs** from crystal; seeds 0, 1, 2, 6, 43 **differ**; seeds 3, 4, 5, 7 **match**. Plain ETKDG without the surrogate (`damvaq_stereo.py`) gives seeds 2, 43, 44, 45 matching, so the match depends on the path. The solver's default embed is therefore the wrong cage stereo for DAMVAQ; the seeds that match are 4 of the 8 seeds of `solve_multi_embed` (0-7).
- GFN2 refinement keeps the embed stereo (`damvaq_gfn2_stereo.py`: seed None still differs, seed 3 still matches). The face is determined at embed time and not corrected later.
- `solve_multi_embed` (8 seeds 0-7) does exist and contains matching seeds (3, 4, 5, 7), so the ensemble path can reach the crystal stereo. Whether the assignment then selects that conformer is not tested here.

### 4. Pentacoordinate C, multi-centre and sigma donors
- SMILES parsing: a 5-bonded C cannot be written in the SMILES with its metal bond, since the dative bond is not there. The cyclopropane CH carbons in the cage SMILES (atoms 10-12) already have 3 heavy neighbours plus H, so they are 4-bonded without the metal.
- Donor classification (`topology/donor_classifier.py`) works from the SMILES dative set; the raw Rh-C BO (0.23) is not used to admit a donor, so the 5-bonded carbon never reaches the classifier.
- GFN2 sees the metal only through the donor-held geometry; a non-donor Rh contact does not enter the embed.
- Q1/Q3 keys: a donor carbon with 5 bonds has no key of its own; for generation it would be a user-specified donor (eta-2 haptic group), and the stats path never produces one.
- Scan (cal + tail, crystal geometry, `unlisted_contacts_scan.py`): metal contacts to atoms not in the donor set (C/N/O/P/S/Cl at < 2.6 A): cal 12 parents (15 contacts), of which 3 are carbons with 2+ heavy neighbours; tail 3 parents (5 contacts), 2 such carbons. Examples: DAMVAQ (C3, C9 at 2.36), JUPDON (C36 at 2.59). **sigma/eta-2 C-C donors that the loader drops: 2 of 130 parents (both DAMVAQ-type, one JUPDON).** No cal/tail parent has a 5-bonded donor carbon in the donor set.

### 5. Stereo over cal and tail (`smiles_stereo_scan.py`)
- cal: 0 of 120 SMILES carry any stereo marker (`@`, `/`, `\`). 75 of 120 fail my metal-placeholder RDKit parse, so the scan is partial. Of the 45 that parse, 16 (36 %) have unassigned tetrahedral centres.
- tail (10 parents in the A4 tail list): 0 with stereo markers; 3 with unassigned centres (DAMVAQ, HOXJUZ, IFUROQ); 3 fail to parse.
- The stereo problem is dataset-wide, not DAMVAQ alone. Whether a stereo-unspecified donor ligand is face-sensitive for a given parent is not measured here.

### 6. Ranked general proposals (coverage gain / cost)

Coverage: cal + tail counts above. The stereo class (36 % of parseable cal SMILES) is far larger than the sigma/eta-2 C-C class (2 of 130). Cost and risk are rough estimates, not measurements.

1. **Stereo-aware free-ligand embed (seed/diastereomer scan with a donor-convergence filter).** For ligands with unassigned tetrahedral centres, embed K seeds (ETKDG, then the existing GFN2 cache), keep those whose donor atoms converge to one point, and pick the one that best matches a donor-convergence score. Cost: K ETKDG plus GFN2 per ligand (GFN2 is cached per SMILES, so the cost is bounded). Risk: more embed time; a donor-convergence score that is not the crystal stereo. Coverage: largest of the options. Test: first measure, for the cal stereo-unspecified parents, how often the crystal stereo appears among seeds 0-7 on the real path (DAMVAQ: 4 of 8 seeds match); then run the scan on 1-3 parents (E-2 CPU), not a panel.
2. **Diastereomer enumeration at embed time** (RDKit `EnumerateStereoisomers` over unassigned centres, pruned by ring-fusion feasibility and embed success, filtered by donor convergence). Cost: up to 2^k embeds (k = 10 here = 1024, so pruning is essential). Risk: chemically impossible isomers, and the cost grows fast with rigid cages. Test: DAMVAQ and the 16 cal parents, counting isomers that survive the filter and whether the crystal isomer is among them. This is the general version of proposal 1; try proposal 1 first.
3. **Donor-face choice as an isomer dimension in the assignment candidates** (stereo-at-binding). Each embed diastereomer becomes its own candidate in the assignment search, with its own Q1-Q5 tables. Cost: high (candidate count multiplies; Q-tables per isomer). Risk: score comparability across isomers. Only worth it if 1-2 leave residual failures.
4. **sigma/eta-2 C-C donor specification plus a generalised cap** (the Li-cap idea extended to neutral sigma donors: a point cap at the C-C midpoint during GFN2 refinement, and an eta-2 haptic group for cyclopropane C-C pairs so the loader keeps them). Coverage: 2 of 130 parents (DAMVAQ, JUPDON). Cost: moderate (haptic path exists for eta-2; a new cap; the donor set changes the fitted tables, so a stats rebuild). Risk: cap placement, and the Q-tables for new donor classes. Low coverage; defer.
5. **Hypervalent / 5-bonded C in SMILES-to-embed.** The SMILES cannot express the metal bond, so this is not a SMILES fix; it reduces to proposal 4 for the donor-specification side. No cal/tail donor carbon is 5-bonded in the donor set. Lowest priority.

Recommended order: 1 (measure first), then 2 only if 1 leaves failures, then 4 if the sigma-C class grows in a broader panel. The Q6 donor-pair flag (Part A) is independent of these.

## Next
- E-1 checkpoint: this report, commit on `jgen/e-score`, no fold-in, no panel runs. The auditor schedules any L4 cal run after review.

---

# Package E-1b: score-energy equivalence (angle tails, donor-pair onset)

Continues `jgen/e-score` from d3c83d0. Constraints DB: `constraints_2026-10-17.json` (golden). CPU only, no panel run, no fold-in. Outputs: `research/data/jgen/campaign/E/eb/`.

## Log
- 2026-10-29: Step 0 (mechanism) run; Steps 1-2 implemented; Step 3.1 run; Step 3.2 re-solve and rescoring run; Step 4 run. 2026-10-30 the golden check was re-run after the scorer edit (below).

## Step 0: mechanism (verified, not refuted)
Script `eb/step0_q2.py`, output `eb/step0_q2.json`. FEKKAJ, seed 11, control solve; candidate Q2 tables read from `scorer._q2_cand`.
- Cl0-Cl1 Q2 key, assignment 14 (the collapsed cis-dichloride): type `tmix_cos`, **nu = 2.0**, one component at **88.9 deg**, scale 0.0861 (cos space).
- Log-density, actual table vs the same table made Gaussian at the same scale:
  - at 89 deg: +1.413 (actual) vs +1.533 (Gaussian)
  - at 52 deg: **-3.418** (actual) vs **-22.509** (Gaussian)
  - cost of the 52 deg miss: 4.8 nats (actual) vs 24.0 nats (Gaussian). The heavy-tail reading holds for this key.
- Assignment 0 (trans) Cl-Cl key: nu = 10, two components (168 deg, 158 deg).

Census (`eb/census_q2.py`, `eb/census_q2.json`). **Approximate**: keys are the crystal donor classes (`kb.q2_pair_keys(mol)`), not each candidate's (candidate assignments are not stored in the pool files).
- 120 cal parents, 996 Q2 pairs. **100 % are `tmix_cos`.** nu histogram: 2 (174 pairs), 4 (258), 10 (564). No Gaussian Q2 key occurs in cal.
- Top-1 rows of the rerun_F seed-11 pools (frame order via cKDTree on `ref`): 882 pairs; **134 (15.2 %) lie more than 3 scale units from their nearest component**.
- Q3 / S_orient census: not done in this package. S_orient is a multivariate **Gaussian** mixture (`stats/orientation.py`, `covs`), with no Student-t, so the tail flag does not touch it. Q2 is the only Student-t path (`_cos_mix_logp`).

## Step 1: `angle_tail_quadratic` (flag, default False)
- `profile.FIELDS["angle_tail_quadratic"]` -> `ik/scorer.py` `ANGLE_TAIL_QUADRATIC`, read at import. `ANGLE_TAIL_Z0 = 2.0` (fixed).
- In `_cos_mix_logp`, for finite-nu components: with a = |z| (standardised cos-space residual), a <= 2 keeps the t log-density; a > 2 is `f0 + s0 (a-2) - (a-2)^2/2`, with f0 and s0 the t value and slope at z0 (C1 at z0; curvature -1 in z units, i.e. -1/s^2 in obs units). Gaussian components are untouched.
- Checked numerically (`tests/test_angle_tail.py`, 6 tests pass): identical inside |z| <= 2; slope-continuous at z0; steeper beyond.
- Replay: `solver_replay.py` does not call `_cos_mix_logp` (no grep hit), so the replay path ignores this flag. **Not regenerated here (per directive); the flag must not be used with CUDA-graph replay until it is.**

## Step 2: donor-pair onset per class (dict flag)
- `q6_donor_pairs` now takes a dict `{"hh": s, "ho": s, "oo": s}` (float still accepted = every class). Classes: `hh` both halide (F, Cl, Br, I), `ho` one halide, `oo` neither (`ik/geometry.py: q6_donor_class`). Fingerprint: one per value.
- Table (`eb/step2_table.py`, `eb/step2_table.json`), cal crystal geometries, 105 parents, 675 pairs (hh 49, ho 185, oo 441; all three classes >= 30):

| class | s | median (kT) | p95 (kT) |
|---|---|---|---|
| hh | 0.70-0.85 | 0.000 | 0.000 |
| hh | **0.95** (chosen) | 0.035 | 0.470 |
| hh | 1.00 | 0.291 | 1.010 (fails p95) |
| ho | 0.85 | 0.000 | 0.114 |
| ho | **0.90** (chosen) | 0.000 | 0.731 |
| ho | 0.95 | 0.088 | 1.498 (fails) |
| oo | 0.80 | 0.000 | 0.236 |
| oo | **0.80** (chosen; 0.85 gives p95 1.118) | 0.000 | 0.236 |

  Rule: largest s in {0.70..1.00} with median <= 0.2 and p95 <= 1.0. Chosen: **hh 0.95, ho 0.90, oo 0.80** (oo: 0.85 fails p95 at 1.118, so 0.80). No extra class was added (none clearly distinct at >= 30 pairs).

## Step 3.1: curvature test (acceptance, passed)
`eb/step3_curvature.py`, `eb/step3_curvature.json`. First 20 cal parents (codes_cal order), 181 kappa pairs. Second derivative of the Q2 log-density w.r.t. the crystal D-M-D angle (`_floored` variances, 10-17 tables), flag off vs on.
- Pairs with every component within |z| <= 2 at the crystal angle: **135**; value and curvature identical (max abs diff **0.0**).
- Pairs with some component beyond z0: **46 (25 %)**; curvature changes as designed (median -79.6 -> -101.8 per rad^2).
- Donor-pair flag, class-5 repulsion curvature wrt distance (20 parents, dict values): hh 6 pairs, 4 active, median active curvature 21 kT/A^2 (max 27.7); ho 37 pairs, 17 active (max 45.7); oo 104 pairs, 10 active (max 66.8). Inactive pairs (d >= s r_m) have zero curvature.

## Step 3.2: nats against energy (acceptance: NOT passed)
**Validation gate failed for 20 of 30 parents.**
- Stored pool scores (A/a4/rerun_F) do not reproduce under the current scorer (e.g. AJAQAD top-1 stored 0.479; candidate scores -159 nats / Z = -5.3). The pool was produced under another configuration. Pool files also store no assignments. Directive fallback applied: **re-solve** the first 30 cal parents (`eb/step3_control.py`, seed 11, CPU, full solve, top-32 conformers).
- Units: `rank_terms` total equals the conformer score (already divided by Z). My first comparison divided by Z twice; corrected.
- Reproduction (`rank_terms` total vs `conf.score`, all 32 rows): **exact (<1e-4) for 10 parents**: AJAQAD, BABNAV, BUXJIO, CEXPOM, CIXYUD, DAMVAQ, EDOBUV, EFOXII, FASPOH, FEKKAJ. The other 20 mismatch on some rows (up to 1.65 score units).
- Likely cause (code reading, `engine.py` ~1218-1258): `solve` keeps conformers from every E/Z enumeration variant that passed, but returns the best variant's scorer. Conformers from other variants cannot be rescored by it. Not confirmed by a direct test.
- Analysis (`eb/step3_analysis.py`, `eb/step3_results.json`) therefore uses the **10 reproducing parents and only their reproducing rows** (3778 pairs; 405 gross pairs). Nats = score x Z. y = -(E_i - E_j)/RT (score is a log-likelihood). MACE: CPU evaluator, reproduces the stored pool energies up to a constant 22.72 kJ/mol (so Delta E is identical).

| arm | Spearman (all pairs) | Theil-Sen slope (all) | Spearman (gross) | Theil-Sen slope (gross) |
|---|---|---|---|---|
| control | 0.226 | 0.025 | -0.086 | -0.003 |
| tail | 0.409 | 0.067 | 0.856 | 0.082 |
| donor (dict) | 0.484 | 0.138 | 0.939 | 0.440 |
| both | 0.482 | 0.138 | 0.935 | 0.526 |

- All-pairs rho is **not worse** than control for any arm (all three improve).
- **Gross subset is one parent**: all 405 gross pairs come from FEKKAJ (the other nine parents have zero gross pairs). The parent-clustered bootstrap therefore has one cluster: the replicates collapse to a point (tail diff 0.942, donor 1.025, both 1.021; CIs of zero width). **The clustered CI test cannot be evaluated on this data, so no flag passes Step 3.3.**
- Bootstrap: 10,000 requested for the gross subset (about 6,500 valid replicates; the rest have no gross pair); all-pairs bootstrap 2,000 replicates: tail diff 0.18 [-0.01, 0.51], donor 0.26 [0.01, 0.61], both 0.26 [0.01, 0.61]. The all-pairs intervals include 0 for the tail arm.
- Conclusion: the rho gain on the gross subset is real for FEKKAJ and the all-pairs ordering improves on these 10 parents, but the directive's pass rule is not met (validation gate failed; single-cluster gross subset). **Neither flag is promoted.**

## Step 4: FEKKAJ and spot solves
Script `eb/step4_run.py` (variant A = production input; C = Ru(II) Fischer carbyne as `fekkaj_clean/run.py`). Full CPU solves, seeds 11-13. Score terms in nats = score x Z (Z = 29 for FEKKAJ).

| arm | seed | asg | Cl-Ru-Cl (deg) | Cl-Cl (A) | score | nats | Q1 | Q2 | Q3 | Q5 | Q6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 11 | 14 | 51.8 | 2.02 | -0.163 | -4.7 | 6.4 | 9.7 | 14.0 | -16.9 | -17.8 |
| control | 12 | 14 | 51.5 | 2.02 | -0.203 | -5.9 | 7.0 | 9.9 | 14.7 | -15.4 | -22.0 |
| control | 13 | 14 | 52.8 | 2.06 | +0.110 | +3.2 | 6.9 | 10.4 | 16.3 | -14.9 | -15.5 |
| tail | 11 | 14 | 96.1 | 3.43 | -0.388 | -11.2 | 4.2 | 12.1 | 11.9 | -15.5 | -23.9 |
| tail | 12 | 0 | 159.6 | 4.54 | -0.264 | -7.7 | 2.9 | 13.1 | 14.8 | -14.9 | -23.6 |
| tail | 13 | 14 | 73.5 | 2.77 | -0.045 | -1.3 | 6.1 | 8.5 | 16.1 | -14.6 | -17.4 |
| donor 0.95/0.90/0.80 | 11 | 14 | 84.8 | 3.14 | -0.411 | -11.9 | 2.9 | 13.2 | 15.0 | -17.8 | -25.2 |
| donor | 12 | 0 | 159.2 | 4.55 | -0.288 | -8.4 | 2.4 | 13.7 | 14.8 | -14.9 | -24.3 |
| donor | 13 | 14 | 82.9 | 3.10 | -0.135 | -3.9 | 5.0 | 12.0 | 15.8 | -14.8 | -21.9 |
| both | 11 | 14 | 94.9 | 3.41 | -0.416 | -12.1 | 3.9 | 12.7 | 11.7 | -15.5 | -24.9 |
| both | 12 | 0 | 159.3 | 4.55 | -0.293 | -8.5 | 2.4 | 13.8 | 14.8 | -15.0 | -24.4 |
| both | 13 | 14 | 82.8 | 3.10 | -0.135 | -3.9 | 5.0 | 12.0 | 15.8 | -14.8 | -21.9 |

Reading: the tail and donor arms move seed 11 off the collapsed cis-dichloride (51.8 deg) to 85-96 deg, and seed 13 to 73-83 deg. Seed 12 is trans (159 deg) in every arm. No arm gives a clean cis ~89 deg structure on all seeds. Scores are per-solve top-1 values, so arms are not compared directly on score.

Ru(II) variant C control (`fekkaj_C_ctrl_s*.json`): seed 11 trans (168 deg, 4.59 A, score -0.420); seed 12 **55.0 deg** (2.15 A, -0.291); seed 13 **52.6 deg** (2.06 A, +0.051). Ru(II) also collapses on two of three seeds, so the "search difference" reading is only partly supported (the collapsed-type top-1 appears under Ru(II) too).

Cal parents with the most pairs beyond z0 (`eb/beyond_z0_count.py`: YISQUM 15/15, SIVBUV 14/15, AYOGUR 13/15), seed 11, control vs both (tail + donor dict):

| parent | arm | score | Cl-M-Cl (deg) | Cl-Cl (A) |
|---|---|---|---|---|
| YISQUM (Re) | control | 1.0741 | 90.8 | 3.31 |
| YISQUM (Re) | both | 1.0412 | 91.5 | 3.34 |
| SIVBUV (Nb) | control | 0.6821 | 96.5 | 3.51 |
| SIVBUV (Nb) | both | 0.6666 | 96.3 | 3.50 |
| AYOGUR (Ir) | control | 1.4217 | - | - (no Cl) |
| AYOGUR (Ir) | both | 1.4187 | - | - |

Nothing broke; scores move by 0.003 to 0.03. The both-arm top-1 changes conformer in YISQUM and SIVBUV only slightly.

## Spec issues (E-1b)
1. Stored pool scores do not reproduce under the current scorer, and the pool files hold no assignments. The directive's rescoring gate cannot be met on the pools; the re-solve fallback was used and its gate also fails on 20/30 parents (E/Z-variant scorer limitation, to be confirmed).
2. The gross-distortion subset is a single parent (FEKKAJ), so the parent-clustered CI test is not evaluable. The directive's pass rule needs a broader gross set (e.g. a larger validated parent list, or the E/Z-fixed rescoring).
3. The replay path does not use `_cos_mix_logp` (tail flag not applied under replay). Not regenerated.
4. `solve` returns one scorer for conformers from several E/Z variants (`engine.py` ~1250); any per-conformer rescoring is wrong for non-best variants.
5. `rank_terms` returns the score (already divided by Z); the directive's "nats" needs x Z. The analysis now applies Z.
6. The Q2 census keys are the crystal's, not each candidate's (candidate tables are not stored).
7. The donor-pair flag changes the scored set for every parent with cross-ligand kappa pairs; the flag value is a dict now (fingerprint per value).

## Verification
- Golden (flags off, after the scorer and geometry edits): production 245/245 and u1 245/245, `golden: OK`. Log `eb/golden_check_e1b.log`.
- Unit tests: `tests/test_profile_hook.py` + `tests/test_q6_donor_pairs.py` + `tests/test_angle_tail.py`: 21 passed.

## Next
- No fold-in. Checkpoint E-1b: commit, this report. Neither `angle_tail_quadratic` nor the donor dict is promoted.
