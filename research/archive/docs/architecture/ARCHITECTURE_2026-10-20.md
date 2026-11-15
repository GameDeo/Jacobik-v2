# Jacobik: architecture, objective, evidence workflow, plan (updated 2026-10-20)

The single reference document. It replaces GLOSSARY, METHODOLOGY, SCORE_SCALE, ENSEMBLES and math.md (all archived in
`archive/docs_2026-10-17.zip`). History: `CHANGELOG.md` (full entries from 2026-10-16, condensed before that).

Contents: 1 purpose, rules, evidence workflow | 2 terminology | 3 adopted pipeline | 4 the u1 objective term by term |
5 profiles and fields | 6 experimental modules | 7 refit pipeline and DB entries | 8 benchmarking | 9 scoreboard and plan 3 |
10 ledger of discarded variants | 11 future tests and open items

Conventions: "A" = angstrom, energies in kJ/mol unless "nats" or "kT" (a score unit, see 4), RT = 2.479 kJ/mol (298.15 K).
File paths are relative to the repo root; `src/jacobik/...` is the package.

## Current model and baseline names (2026-10-26 post-restructure)

Following the October 2026 repository restructure and consolidation, the production configuration
has all resolved parameters hardcoded directly in production code:
- `nci = False` (after full-surface Step 4 calibration confirmed $\gamma_{\text{shared}} = 0.0$ on the real surface).
- `deterministic_embed = True` (GFN2-xTB embed on 1 OpenMP thread for exact bit-identical reproducibility).
- `ring_fold_hinges = False` (validated fold-hinge removal: Colab tail $e = -335.6$ kJ/mol, cal $e = -13.0$ kJ/mol, 0 parent losses $> 10$ kJ/mol).
- `q6_model = "repulsive"` (clean runtime in `src/jacobik/stats/q6.py`).
- `solve_floor = "thermal"` (measured harmonic thermal widths at 298 K).
- `score_agg = rank_agg = "sum"` (summed log-likelihood divided by per-complex feature norm $Z$).
- `bend_release = [0.1, 100]` (Package F fold-in: schedule on angle_bend step size slashing catastrophic tail).
- Production baseline fingerprint: **`5f74b14b57ed`** (IA-k finish phase, 2026-11-02; superseding the eta2 face-plane + haptic-cap fold-in `273f2ccbf892`).

The production database is the lossless float64 binary container `data/db/constraints.npz` (24.25 MB,
an 81.5% size reduction from the historical 134 MB JSON) with input provenance tracked in
`data/db/constraints.meta.json`.

All 37 historical runtime flags and the legacy profile branch were removed and archived with complete
evidence tables in `research/archive/flags/FLAGS.md`. `src/jacobik/profile.py` is maintained as an
empty, documented experiment hook for future research investigations.

| Label | Precise meaning |
|---|---|
| **Production Baseline** | Resolved production configuration (`5f74b14b57ed`: IA-k `finish_phase` on top of `273f2ccbf892`, with `q6_donor_pairs`, `eta2_face_plane`, `eta2_group_bend` and `embed_haptic_li_cap`): `nci=False`, `deterministic_embed=True`, `ring_fold_hinges=False`, `bend_release=[0.1, 100]`, `embed_li_cap=True`, repulsive Q6, thermal floor, summed aggregation. |
| **`u1_pure_rep`** | Historical basis of the production configuration (polar-H wells and NCI off). |
| **Q6** | Purely repulsive steric contact exchange potential (`src/jacobik/stats/q6.py`). Historical attractive modes and PMF code archived to `research/interactions/q6pmf.py`. |
| **NCI** | Non-covalent interaction research code, archived to `research/interactions/nci_v2/` and `research/interactions/nci.py`. |

---

## 1. Purpose, rules, evidence workflow

### 1.1 Purpose
Jacobik generates conformers of a transition-metal or lanthanide complex from a description only: metal, charge, spin,
coordination geometry label, and one SMILES per ligand with its donor atoms. It is an inverse-kinematics (IK) solver:
each free ligand is embedded once (no metal), then moved as rigid bodies plus torsions (and a few bond-angle bends) until
a **statistical-potential score** is maximal. The score is a sum of log-likelihoods of geometric features (M-D distances,
D-M-D angles, donor orientation, torsions, non-bonded contacts) against distributions fitted to ~10^5 crystal structures
(tmQM, LnQM). No crystal coordinates of the target enter a solve.

The benchmark question is: how close, in MACE single-point energy, is the best-scoring conformer to the reference
structure (a GFN2 minimum from tmQM)? Targets (METHODOLOGY, 2026-10-15): #1 |dE| median < 100 kJ/mol and no case above
200; ensemble rho > 0.6; solve time < 20 s. Current numbers: section 9.

### 1.2 Rules
1. **No force field, relaxation or MLIP acts on a complex during a solve.** Physics enters only as fitted parameters
   (GFN2 dimer scans for the repulsive Q6; GFN2 Hessians for thermal widths; MACE fits for the bend prior), measured
   offline. MACE and GFN2 are used to MEASURE results, never to steer the solver.
2. **Every result-changing setting is a profile field** (`src/jacobik/profile.py`, section 5). Discarded variants are
   deleted, not left behind a switch.
3. **The protected DB `data/db/constraints_2026-10-18.json` is never overwritten.** New entries go in a new
   file.
4. **Bug fixes that change results go through the evidence workflow** (1.3), like any other change.
5. **MACE float64 for every comparison** (float32 energies are not reproducible, CHANGELOG 2026-10-16 e).
6. **One GPU job at a time.** Memory is 16 GB; a refit and a GPU job should not overlap.
7. `u1` is the default profile. The held-out `heldout28` and `ring10` runs remain a separate, not-yet-completed evaluation.

### 1.3 Evidence workflow
Why it exists: several changes that were correct on their own terms (Q6 onset wall, WCA core, Q6 PMF, GFN2-Hessian torsion
widths, summed instead of mean terms) made the benchmark worse, and the explanation came afterwards. The protocol makes
the outcome predictable before it is run.

1. **Prediction in the CHANGELOG first:** which budget line (section 8.3) the change attacks, direction and size, the
   mechanism, the falsifier.
2. **Ceiling check:** is that line, on the target set, larger than the effect you need? (budget tools, or an oracle.)
3. **Offline check:** curvature (`forces.py`) for a reshaped term, pressure/pull-off for a sum-over-pairs term, then the
   **re-rank gate** (`gate.py`): re-scores fixed pools, zero solver noise, minutes. FAIL = stop. A change that
   re-ranks the same conformers worse has no mechanism by which steering the solver would make it better. A gate that
   shows no change means the change can act only through generation; say so in the prediction.
4. **Dev80 arm:** dev32 + N48, seeds 11-13, MACE float64, paired test per set (`bench.paired_test`). If the minimum
   detectable effect (MDE) exceeds the predicted effect the experiment cannot answer: enlarge the set, do not read medians.
5. **Held-out once (S5):** heldout28 and ring10 are small (MDE ~70 / ~190 kJ/mol on the mean): use them for regressions
   and catastrophes, not for 10 kJ/mol decisions.
6. **Report every budget line**, not only #1: pool floor, regret, top-1, rho_lo, basin share, the paired test with CI and
   better/worse counts. A wrong prediction is a finding about the model of the problem and is recorded as such.

### 1.4 Principles behind the workflow (the evidence is in section 8.3)
- **The score is an ordinal surrogate, not an energy.** It is a sum of log-likelihoods of marginals fitted to optimised
  structures; correlated marginals count shared information more than once. One nat of one term is not worth one nat of
  another. A change that is "more correct" for one term changes its weight against every other term, which is why such
  changes move the benchmark unpredictably.
- **"Matches the data" is not "matches the energy".** A table fitted to tmQM tells where structures sit, not what it costs
  to be elsewhere (section 2: position, well weights, local stiffness, tail are four separate properties).
- **Noise first.** Identical code and DB on CUDA move set medians by up to 29 kJ/mol; every comparison is paired per
  complex against that noise.
- **Cheapest decisive test first.** Offline gates before solver arms.

---

## 2. Terminology

### 2.1 The complex and the solver
| term | meaning |
|---|---|
| **donor group / site** | donor group: the atom(s) of a ligand bound to the metal (one atom for kappa, several for haptic eta-n). Site: a vertex of the ideal coordination polyhedron. The donor point of a haptic group is the centroid of its bound atoms. |
| **assignment / isomer class** | donor-group -> site map, up to proper rotations of the polyhedron and swaps of identical ligands. Mirror images are different classes. "Isomer" in this document always means this coordination isomer. |
| **internal configuration** | stereocentres, E/Z, ring puckers inside a ligand, fixed by the embed. A hidden isomer axis (1 pick in 5 carries one the crystal lacks). Not called "isomer". |
| **embed** | the one-time free-ligand geometry (RDKit ETKDG, MMFF, GFN2-xTB `--opt crude`). After it all bond lengths and bond angles are frozen, except optional bends. |
| **rigid body / junction / joint** | the ligand is cut at rotatable bonds into rigid bodies; a junction is an atom where bodies meet (its angles are frozen); a joint is a torsion DOF. |
| **bend** | an optional bond-angle DOF on a chelate backbone or haptic face, priced by an energy-calibrated prior (4.8). |
| **row** | one trial in the batch (candidate assignment x coordination sample x torsion seed). |
| **beam** | per ligand, the top W=16 torsion-mode seeds. |
| **cull** | successive halving of rows at step 0.3 T (only when B >= 256). |
| **profile** | a named, frozen set of values for every result-changing field (section 5). |
| **plain / etkdg start** | the normal solver: each ligand starts from its own free-ligand embed. |
| **templated start** | (not in src) ETKDG with extra donor-donor distance bounds from cis/trans labels; superseded by the chelate-aware embed. |
| **hinge joints, reach phase** | (not in src) experimental bond-angle hinge / pure-IK pre-step; reach phase got stuck in local minima. |
| **conformer ensemble** | what `solve()` returns: unweighted, score-ranked snapshots, one per 0.5 A RMSD cell. |
| **Boltzmann ensemble** | basins weighted by exp(-dG/kT). Not produced; preconditions in 11. |

### 2.2 Score and statistics
| term | meaning |
|---|---|
| **Q1 ... Q6, Q7, QO** | the quantities: Q1 M-D distance; Q2 D-M-D angle; Q3 donor approach angle; Q4 pyramidalisation; Q5 torsion; Q6 non-bonded contacts; Q7 legacy bend penalty; **QO = S_orient** joint donor orientation (replaces Q3 + Q4 when the DB has QO). Extra score components: **QV** (bond valence), **NCI**. |
| **key / ladder** | a feature is labelled by a key (e.g. metal, spin, OS, donor subclass, geometry, trans partner for Q1); a lookup ladder walks from the most specific key to coarser ones and takes the first fitted entry with n >= 30. |
| **mixture** | each key's distribution is a mixture: Gaussian (Q1, Q4), Student-t in cos space (Q2, Q3), von Mises (Q5), full-covariance Gaussian (QO). |
| **nat** | unit of log-likelihood. In the summed objective (u1) every term is in nats. |
| **Z** | one constant per complex (number of internal features). The u1 score is (sum of terms)/Z. It changes no argmax but keeps the Adam step on the old scale. |
| **score** | the solver's statistical objective (higher is better). Crystal VUCVUN scored -0.75 in the old mean scale. |
| **solve objective / ranking score** | two functions of one `Scorer`. In legacy they differ in floors, aggregation, Q6 and Q1 tail; in u1 they are the same function (4, 5). |
| **well weights, mode means** | mixture structure vs component positions. |

### 2.3 Words that mean two things (use the left column)
| use | not to be confused with |
|---|---|
| **pool floor** = lowest-energy conformer the solver returned | **variance floor** = minimum width a table component is scored with |
| **ring fold / pucker** = conformation of a ring (macrocycle folds from the macro pool; frozen puckers of small rings) | **rotor folding** (`q5_fold`) = torsion statistics folded by a rotor's symmetry (a methyl's 3 wells as one) |
| **isomer** = coordination isomer (assignment) | **internal configuration** = stereo/E-Z/puckers inside a ligand |
| **strain** = single point minus the UNCONSTRAINED relaxation | **internals line** = constrained-relaxation share of the floor (8.3) |
| **output window** (`out_window`, 200 summed nats) | **CREST window** (6 kcal/mol on relaxed energies) |
| **between-conformer scale** (kT per nat from finite score vs energy differences) | **curvature ratio** (local energy curvature / score curvature) |
| **position / well weights / local stiffness / tail** (four properties of a term) | **weight / aggregation** (how a term combines with others; it changes every term's effective stiffness) |

### 2.4 Measurement and benchmark terms
| term | meaning |
|---|---|
| **dE / dE_sp** | MACE-POLAR-1 single point of the structure minus that of the reference (same atoms, charge, multiplicity), kJ/mol. Negative values happen. |
| **#1** | the best-scoring conformer; "#1 error" = its dE. |
| **pool, pool floor, regret** | pool: every conformer a solve returned for one (complex, seed). Floor: its lowest dE. Regret: dE(#1) - floor (what ranking loses). |
| **top-1** | share of pools whose #1 is within 1 kcal/mol (4.184 kJ/mol) of the pool floor. |
| **rho, rho_lo** | Spearman of score vs dE over the pool; rho_lo restricted to conformers within 50 kJ/mol of the floor. |
| **basin** | one local minimum of the true surface inside an isomer class; operationally: loose GFN2 relaxation, same class, relaxed energies within 1 kJ/mol and heavy-atom RMSD < 0.25 A. Relaxations that change the covalent graph are excluded. |
| **snapshot** | any point inside a basin; its SP energy = basin energy + strain. Every returned conformer is a snapshot. |
| **dE_basin** | relaxed structure minus relaxed reference ("is it the right structure?"). |
| **thermal scale** | (3N-6) kT/2 above the minimum for an N-atom molecule (~310 kJ/mol for 85 atoms). A 100-200 kJ/mol dE is snapshot-sized; 1000+ is not. |
| **oracle** | any diagnostic using the reference structure ("what if this part were perfect?"); never part of the solver. **full oracle** (`--start full_oracle`): ligands start with crystal internals. **junction oracle** (`--start junction_oracle`): rigid pieces placed as in the crystal, only inter-piece angles corrected. **reachability fit**: can the DOFs express the crystal at all. **crystal-positioned statistics** (`--targets crystal`): table modes translated onto the crystal values. |
| **Track I / Track O** | I (isomer-matched, `pools.py --isomer ref`): solver gets the reference's assignment, so dE measures precision. O (open): solver chooses its isomer. |
| **seed noise** | spread of a case's dE over seeds; differences below ~2x this are not resolvable on one run. |
| **dev32, N48, dev80, heldout28, ring10, panel** | benchmark sets, 8.1. |
| **arm** | a named solver configuration run over a set (its pools live in `<OUT>/pools/<arm>/`). **u1c** = contact Q6; **u1c+rep** (`etkdg_u1_rep`) = repulsive Q6 before the OS fix; **u1c+rep+os** (`etkdg_u1_os`) = corrected OS, the current base. |
| **guards / golden / baseline** | guards: fast hang-proof pytest checks; golden: stored re-scores and solves to detect unintended result changes (8.6); baseline: stored panel results. |
| **device / sandbox** | laptop checkout (source of truth) vs the cloud copy. Numbers count only when fingerprints match (`python -m tools.jdev fingerprint`). |

---

## 3. The adopted pipeline: embed -> beam -> IK solve -> dedup -> rank

Entry point: `jacobik.engine.ComplexBuilder(metal, geometry, charge, spin, ligands=[LigandInput(smiles, donors)]).solve(
constraints_db=..., seed=..., device="cpu"|"cuda", angle_bends=True, ...)` returns an `EnsembleResult`. Benchmarks always
use `angle_bends=True`; it is now the u1 API default (legacy remains False). No DB ships inside the package: pass `constraints_db` or `stats=`.

| step | what happens | key files |
|---|---|---|
| input / OS | inputs per S-notation below; oxidation state deduced from charge and ligand charges (code 99 = unknown) | `core/oxidation.py`, `core/types.py` |
| donor typing | donor subclass hierarchy (`O_water` < `O_neutral_sigma` < `O` < `element_any`), geometry classification | `topology/donor_classifier.py`, `donor_hierarchy.py`, `coord_geom.py` |
| embed | per-ligand RDKit ETKDGv3 (seed 42) + MMFF94, chelate-aware bounds, GFN2 refinement | `engine.py` (`_smiles_to_conformer`, `_gfn2_refine`), `chelate_embed.py` |
| assignment | donor -> site: Hungarian baseline, chelate repair, reachability, full symmetry enumeration | `ik/site_enumeration.py`, `ik/reachability.py`, `ik/targets.py` |
| kinematics | rigid bodies, torsion joints, macrocycle joints, ring joints, bends | `topology/decompose.py`, `ik/fk.py`, `ik/fk_compose.py`, `ik/angle_bends.py` |
| beam / init | torsion seeds from Q5 modes; coordination targets from Q1; Kabsch placement | `ik/beam.py`, `engine.py` |
| IK solve | batched Adam, augmented-Lagrangian shape/symmetry constraints, cull | `ik/ik_solver.py`, `ik/solver_replay.py` (CUDA fast path; generated by `tools/jdev/build_replay_solver.py`) |
| score | Q1-Q6, QO, bend prior, optional NCI / QV | `ik/scorer.py`, `ik/kernels.py`, `ik/geometry.py`, `stats/*` |
| dedup | symmetry-aware heavy-atom RMSD, 0.5 A | `ik/dedup.py` |
| rank / output | `ranking_score`, output score window | `ik/scorer.py`, `engine.py` |

### 3.1 Inputs
Metal, total charge q, spin multiplicity, one of 13 geometry labels (linear; trigonal, t_shaped; square_plan,
tetrahedral, see_saw; square_pyr, trig_bipyr; octahedral, trig_pris; pent_bipyr; square_anti; tricap_pris), and per ligand a
SMILES with donor groups (a single atom index for kappa, several for haptic). The donor-group count must equal the CN.
Benchmark complexes take these from the tmQM record; the benchmark never passes an oxidation state.

**Oxidation state:** OS = q - sum of ligand charges. Legacy uses formal charge minus
radical electrons on non-haptic atoms, with all-carbon haptic groups assigned -1 for sizes 3/5 and 0 for 2/4/6.
The adopted u1 `oxidation_model="training"` instead counts whole-ligand formal charge once, independent of haptic size
and RDKit radicals, with the extraction's neutral-hydride and donor-aware nitro corrections. Anionic donors must be
written explicitly. It uses no target metal label or coordinates. Unassigned or outside [-4, OS_max] = 99 (unknown);
OS_max is 4 for lanthanides, the group number for listed metals, 8 otherwise. The deployed Q1 ladder is `m_sub_early`;
state-keyed levels are skipped when OS is unknown. Agreement, gate and MACE64 arm are complete; adopted 10-18 (11.1).

### 3.2 Embed
ETKDGv3 seed 42, MMFF94 (a silent no-op when MMFF lacks parameters, e.g. boron), then `embed="gfn2"` optimises the ligand
with GFN2-xTB `--opt crude`, holding every donor-donor distance at the embed's value (`mmff`, `etkdg` stop earlier).
- **Chelate-aware bounds** (ligands with >= 2 kappa donors): 2.4 A <= d_ij <= r_i + r_j (typical M-D lengths: C 2.00,
  N/O 2.05, P 2.25, S 2.40, Se 2.50, Cl 2.35, Br 2.50, I 2.70, else 2.2 A); for donors closing a 4-6 ring also
  d_ij <= sqrt(r_i^2 + r_j^2 - 2 r_i r_j cos 100 deg). Best of 8 ETKDG candidates when the first violates them; MMFF with the
  bounds as flat-bottom restraints (k = 100 kcal/mol/A^2).
- **Anion caps:** a donor written as a formal anion with a lone pair (`[C-]=O`, `[N-]`, `[O-]`, `[S-]`) is neutralised and
  H-capped for embed/MMFF, then uncapped (otherwise MMFF types an acyl carbon as isocyanide-like and straightens it).
  Group-13 "ate" centres are not capped by default; `bound_surrogate` (u1) embeds group-13 `D=[X+]` donors in their bound,
  H-capped form.
- **Chelate E/Z** (`chelate_ez`): a non-ring double bond on the shortest path between two kappa donors, closing a ring of
  <= 7 atoms with the metal, gets the ring's required configuration. `ez_enumerate` (u1) solves every E/Z combination the
  SMILES and the chelate rule leave open. `carbene_ylide`: heteroatom-stabilised carbenes embedded as ylide/iminium.
- **Determinism:** xtb is OpenMP-threaded and non-deterministic under CPU load; `deterministic_embed` pins one thread.
- A failed GFN2 call silently falls back to MMFF (11.4).

### 3.3 Donor-to-site assignment
1. Baseline: Hungarian on arccos(unit donor direction . site vector), ligands translated so the donor centroid sits 2.5 A
   from M. Chelate repair swaps (pairs <= 4 hops apart and > 115 deg apart).
2. Reachability: a same-ligand donor pair at hop count h on sites with angle g is reachable if h <= 4: 55-115 deg;
   h = 5: <= 150 deg; h >= 6: any. An unreachable baseline is replaced by the minimum-cost reachable assignment
   (branch and bound, 200 000 nodes).
3. Enumeration (default on, when n_donors = CN and the geometry has fixed sites): every reachable bijection, up to
   50 000 or 30 s. **Embed-aware dedup:** assignments are merged only if related by a proper rotation of the polyhedron
   (octahedron 24 of 48, tetrahedron 12 of 24) and/or swaps of whole identical ligand instances. Graph-level symmetry is
   NOT used: swapping two equivalent donors inside one ligand mirrors their arrangement in the frozen embed, which torsions
   cannot always undo (B(pz)2, C(pz)2; VUCVUN). Assignments the legacy dedup would merge form one class and share one
   row budget (`CLASS_BUDGET = "split"`).
4. All surviving candidates are solved together in one batch, ranked together.

### 3.4 Kinematics (what moves)
- **Rotatable bond:** non-ring, bond order 0.95-1.05, both atoms degree > 1 (H counted). Rigid bodies = components after
  cutting them. Root = a donor-free body. Joints in BFS order; joint angles are absolute dihedrals.
- Per ligand DOFs: 3 translation, 3 axis-angle, torsions, plus: branch-bend joints at a kappa donor with >= 2 branches;
  **macrocycle joints** for rings >= 8 atoms of single bonds with >= 2 donors (ring opened at a seam; closure by a
  Jacobian correction each step and 4 damped Gauss-Newton iterations after; folds drawn from a 64-embed ETKDG pool,
  diversified at ring-dihedral RMS >= 0.6 rad); **ring joints** for small donor rings (`ring_joints`, `ring_open_adj`).
- **Q5 / FK alignment:** Q5 rows are looked up for the same bond as the FK joint and shifted into the joint's dihedral
  convention (an old bug made 49 of 52 joints read another bond's statistics).
- **Bends** (`angle_bends=True`): two DOFs (in-plane, out-of-plane) per bend pivot on the shortest donor-donor paths of a
  ligand's acyclic bonds, haptic faces (`haptic_bends`, lean priced by S_orient rows up to face size `haptic_free_lean_max`,
  wall on lean `haptic_lean_wall`, all face sizes with `haptic_lean_all`), ring-fold hinges. Bends preserve bond lengths;
  composition is a rigid kinematic chain (`BendPlan`).

### 3.5 Beam and initialisation
- **Beam:** per ligand, each partial seed is expanded with every Q5 mode mean (weights > 1e-7), scored by cumulative
  log von Mises at the means, top W = 16 kept. `beam_mode_score = density` (legacy) ranks by density at the mean;
  `mass` (by mode weight) is a profile field, tested, not adopted. The beam does not see the metal or other ligands (11.3).
- **Coordination targets:** M = 8 M-D lengths per donor drawn from the Q1 mixture, T = l * site vector. Used only for
  placement, not scored.
- **Batch:** B = n_cand x 8 x 16 rows; base DOFs from a Kabsch fit of the donor points to the targets, with random
  orientation diversity; branch bends start at current + U(-pi, pi).

### 3.6 Optimisation
Adam, T = 200 steps, lr 0.1, beta (0.9, 0.999), all ligands and rows jointly, on L = -S + C_shape + C_sym.
- **Shape constraint:** best rotation R* (Horn, no gradient) taking assigned ideal vectors onto the donors;
  x_s = |R* e_pi(s) - d_s|^2, Delta = (1/tau)(LSE(tau x_s) - log n), tau = 2, v = max(0, Delta - Delta_tol) with
  Delta_tol the 90th percentile over crystals per geometry (linear 0.006, square_plan 0.017, octahedral 0.047, square_pyr
  0.055, trig_bipyr 0.068, trigonal 0.070, square_anti 0.070, tetrahedral 0.083, pent_bipyr 0.100, trig_pris 0.120, see_saw
  0.142, other 0.060). Augmented Lagrangian mu v + rho v^2/2, rho = 8, mu0 = 10, dual ascent, mu frozen once max v - min v
  < 0.01 over 20 steps.
- **Symmetry ties** for equivalent donors (distance variance mu0 = 2, approach-angle variance mu0 = 1, same freezing).
- **Bidentate null space:** gradient of base rotation projected off the donor-donor axis. **Q3 mode commitment** (only
  without S_orient).
- **Convergence/snapshot:** a row is converged if its score varied < 0.02 over 8 steps; no early stop; each row keeps its
  best-scoring step (score minus 1000 * seam residuals).
- **Cull** (B >= 256): at step 0.3 T, keep max(0.35 B, 16) rows by s_b - 30 (s_(1) - s_(k))/k * rank_b(basin signature)
  (successive halving with niches; the signature is candidate index + nearest Q5 mode of each backbone torsion).
- **Thread race fix:** each thread chunk scores with its own scorer copy.

### 3.7 Dedup, ranking, output
Macrocycle seams are closed exactly. Every row is re-scored by `Scorer.ranking_score`. The returned conformers are
deduplicated: the best-ranked member of each group within 0.5 A symmetry-aware heavy-atom (+metal) RMSD
(automorphisms incl. coordination bonds; done with matrix products and a closed-form rotation fit) is kept. The **output
window** drops conformers more than 200 summed nats below the best (`out_window`). The highest-scoring conformer is the
prediction. Default: the top max(16 n_cand, 8) rows are assembled; benchmarks keep all returned conformers.

---

## 4. The u1 objective term by term

Profile `u1` (section 5) uses ONE function for the solve and the ranking:

    S = [ Q1 + Q2 + S_orient + Q5 + Q6 + bend prior (+ NCI) (+ QV) ] / Z       every term a sum of nats, weight 1

Z is one constant per complex (the number of internal features). Shape constraints, symmetry ties and macrocycle closure
are constraints, not score. "kT" below equals nats. Widths are floored at the 298 K **thermal widths** (4.9).

### 4.1 Which term owns which distance
Each pair distance above its 0.1 % onset is priced by exactly one term; the physical core below the onset is shared.

| geometric quantity | owner | notes |
|---|---|---|
| M-D distance (kappa donor; haptic centroid) | Q1 (+ QV coupling, off) | trans-partner key from the assignment |
| D-M-D angle, every site pair | Q2 | cos space; key includes the site-pair class |
| donor orientation: angles among {M, substituents} at a kappa donor; haptic tilt/bend per substituent | S_orient (QO) | replaces Q3 + Q4 |
| torsions about rotatable bonds, incl. the sterics of pairs 4 bonds apart ("class 4", 1-5) | Q5 | 1-5 repulsion not added (tested, not kept, 6.4) |
| non-bonded pairs >= 5 bonds apart or on different ligands, non-polar ("class 5") | Q6 repulsion | physical form out to the vdW contact distance |
| polar-H ... acceptor (class 5) | Q6 contact wells now; NCI "hb" when `nci` is on | wells removed from Q6 when NCI is on |
| metal pairs (class m) | Q1 / Q2 / S_orient; Q6 core below the onset only | m13 (metal...X, X bonded to a donor) core optional |
| bond angles inside the ligand | bend prior | energy-calibrated, MACE-fitted |
| ring-ring stacking, C-H...pi | NCI "pp", "chp" (off) | |

### 4.2 Q1: metal-donor distance
d_s = |D_s - M|. Key (metal, spin, OS, subclass, geometry, trans-partner subclass); the trans partner is the site with
cos angle < -0.766 (> 140 deg), else "none" (always none for CN <= 2, linear, other). At solve time one Q1 table per
candidate assignment, gathered per row. Mixture of Gaussians in A, K by BIC. Ladder (`m_sub_early`, deployed DB sidecar): (m, spin, OS, sub,
geom, trans) -> (m, spin, OS, sub, geom) -> (m, spin, OS, sub) -> (m, sub) -> (m, spin, OS, bkt) -> (m, bkt) -> (m, spin, OS, fam) -> (m, OS, fam) -> (m, fam) -> (pg, sub) -> (pg, bkt) -> (pg, fam) ->
(any, fam) -> (fam) -> element_any; pg = period group (3d, 4d, 5d, Ln); state levels skipped when OS = 99.
Thermal floor per donor element (A): C 0.041, P 0.053, eta 0.057, Cl 0.058, O 0.060, I 0.060, Br 0.065, S 0.069, N 0.071,
else 0.059. u1: no tail cut (`q1_tail_sigma = 0`), one-sided wall beyond the table's support (`q1_wall`).

### 4.3 Q2: D-M-D angle
c_st = cos angle(D_s, M, D_t) for all site pairs. Key (geometry, distance class, metal, subA, subB); distance class:
different_ligand; else chelate_short (path <= 4), chelate_long (<= 8), same_remote. The 2026-10 refit adds the site-pair
class (cis/trans/...) from the best-fit assignment (the solver uses each candidate's own); haptic faces are their own
family. Student-t mixture (shared nu in {2, 4, 10}) in cos space (no Jacobian). Ladder: (geom, dist, m, subA, subB) ->
bkt -> fam -> (geom, dist, pg, famA, famB) -> (geom, dist, famA, famB) -> (geom, dist) -> (CN, dist) -> (geom) -> (CN) ->
(any). Thermal floor 4.9 deg (converted to cos at each component mean: Var = sin^2 s^2 + cos^2 s^4/2).

### 4.4 S_orient (QO)
For a kappa donor D with k in {1,2,3} substituents (H included): u_M = unit(M - D), u_i = unit(X_i - D), sorted by
u_M . u_i; g = (u_M . u_(i), u_(i) . u_(j)) in [-1,1]^d, d = k + C(k,2) = 1, 3, 6 (the Gram matrix of the unit bond
vectors). Chelate-ring size rho = smallest ring through the donor (0 none; >= 7 pooled). Full-covariance GMM, K <= 4 by
BIC (n >= 15K), covariance >= 0.05^2 I, components with weight < 0.05 pruned, donors with an intramolecular H...acceptor
< 2.3 A excluded from the fit (the H-bond sets their angles). Haptic faces (3-8 heavy atoms) have class k = 10 + size with
features g = (tilt = n . unit(M - c), bend = mean over substituents of unit(X - A) . n) and the same ladder;
`haptic_per_sub` fits per substituent. Ladder: (k, rho, sub, m, geom, anion) down to (k). Thermal floor 4.2 deg (Loewner
floor C >= diag(floor^2)); haptic features are not floored. Legacy divides log p by d per donor; u1 sums log p per donor.
Anion bucket = number of other anionic sites in the shell, capped at 2.

### 4.5 Q5: torsions
For each rotatable bond j-k, flanks i, l = heaviest heavy neighbours; signed IUPAC dihedral. Key (type_j, type_k, type_i,
type_l, ring context [chelate = bond on a shortest path <= 8 between two donors of one ligand]). Types sp3 / double /
aromatic; aryl ortho count in the key. Data are folded by rotor symmetry at extraction (CH3/tBu 3-fold, planar
symmetric aryl 2-fold). von Mises mixture, kappa by Best-Fisher, K by BIC, components without a density dip merged
(one component per well); broad components (kappa < 4) are the off-well tail, dropped from the scored density when they
hold < half the weight and real wells exist. Thermal cap kappa <= 1/sigma_T^2 = 17.3 (13.8 deg); 74 % of Q5 wells are
sharper than that. Q5 is the term closest to an energy: ~1 kT per nat between wells.

### 4.6 Q6: repulsive (profile u1: `q6_model = pmf`, `q6_pmf_wells = repulsive`)
Class-5 non-polar pairs (>= 5 bonds apart, other ligands, joined through the metal):

    U(r) = A [ e^(-b r) - e^(-b r_m) + b e^(-b r_m) (r - r_m) ] / RT     for r < r_m;  0 at and beyond r_m
    log p = -U, summed over pairs, / Z

- Shifted force: value and slope are zero at r_m (smooth for Adam). Linear continuation above 50 kT (so deep overlaps keep a
  gradient).
- r_m = R_i + R_j, Alvarez 2013 vdW radii (H 1.20, B 1.91, C 1.77, N 1.66, O 1.50, F 1.46, Si 2.19, P 1.90, S 1.89, Cl 1.82,
  Se 1.82, Br 1.86, I 2.04 A; "Hp" uses H).
- A, b per element (`q6pmf.REP_PARAMS`), Buckingham combination A_ij = sqrt(A_i A_j), b_ij = (b_i + b_j)/2. Fitted jointly to
  GFN2-xTB dimer interaction energies summed over all atom pairs (17 probes without polar H, 14 688 dimers at 0.70-1.15 x r_m,
  repulsion plus damped C6; R^2 0.94, Spearman 0.93 on repulsive configurations; b at its 2.5/A bound for B, N, O, P, S, Si).
  Tool `research/bench/tools/jdev/budget/gfn2_dimer_scan.py`, outputs `data/refit/q6rep/`. Pairs without parameters: the contact core.
- All other pair classes keep the **contact form**: the GFN2 exponential core below each key's 0.1 % onset r0 (w(r) = w(r0)
  + kappa(exp(b(r0 - r)) - 1); b, kappa from head-on GFN2 scans `GFN2_CORES` for 8 pairs, else generic b = 4.0 with H, 3.2
  otherwise, kappa = 1), 0 at and above r0. Class 4 (1-5) = core only; class m = core only; polar-H...acceptor pairs also
  keep the attractive part of the data curve (the H-bond well, tapered between 5 and 6 A; two D-H...A angle classes lin
  >= 120 deg / bent, prior 0.25 / 0.75) until NCI replaces it.
- Physics finding: at 0.88-1.15 x r_m GFN2 dimers are net attractive; the repulsion alone is 7 -> 0.5 kJ/mol. The repulsive
  form is mild in the flank by construction (~0.8 kT per near-crystal complex vs 9.2 for walls).
- Legacy still scores the **5th-percentile walls** (`q6_model = wall`): half-Gaussian beyond mu = Q_0.05 of the key's
  non-bonded distances with width from the 5 %-1 % spread; solved with a soft-min (tau 2) in legacy, summed in u1-like
  settings. Needs only `Q6:` entries.

### 4.7 Haptic lean wall and bends
Haptic bends are priced by S_orient rows (tilt/bend) up to face size 6; `haptic_lean_all` (u1) applies the lean wall to every face
size (legacy: eta2/eta3 only).

### 4.8 Bend prior
l_bend = -s sum_k k_k max(0, |beta_k| - 2 deg)^2 / (2 RT), summed over bend DOFs, s = 1. k_k (kJ/mol/rad^2): in-plane sp2 1300,
in-plane sp3 1800, out-of-plane sp2 310, out-of-plane sp3 1100, from harmonic fits to MACE-POLAR-1 single points on 110
bend DOFs of 12 free chelating ligands (-12 to 12 deg). An offline constant. `bend_per_z` (u1) divides it by Z (under legacy
it is added raw to a score that is already /Z, i.e. weighted Z times its log-probability). The legacy flat-bottom Q7
(free 2 deg, width 5 deg) still exists as `bend_penalty="q7"`.

### 4.9 Thermal widths (why the floor, why these values)
A fitted component can be far narrower than any one molecule's motion: a cluster of near-identical minima fits a needle (sd
0.002-0.02 A), which makes the score far stiffer than the energy. The physical lower bound is the width one molecule has
at 298 K: sigma_T^2 = kT g^T H^+ g (g = gradient of the feature, H^+ = pseudo-inverse of the GFN2 Hessian, other coordinates
relaxing). Measured once offline on 300 tmQM training complexes (100 per metal row): `refit/thermal_widths.py sample | hess |
sigma | fit` -> `data/refit/final/thermal_widths.json`; stored as constants `scorer.THERMAL_SD` / `THERMAL_Q1_BY_DONOR`
(Q1 0.0585 A, Q2 4.9 deg, S_orient 4.2 deg, Q4 4.2 deg, Q5 13.8 deg). Metal and CN move widths < 10 %, donor element by
1.7x (M-C stiffest). About half of the Q1/Q2 components are narrower than thermal. Nothing is computed in a solve.

### 4.10 What the objective is and is not
Measured curvature (energy / score, by term; 1 = energy-like): see 8.4. Before u1: ranking Q1 7-17x too stiff, solve Q1 4.5x
too soft; the single linear "kT per nat" numbers (Q5 0.85, S_orient 0.10, Q6 0.07, Q2 0.04, Q1 0.02) are misleading because
the relation is strongly non-linear (soft core, steep tail for Gaussian terms). u1 with thermal floors and the repulsive Q6:
every term 1.0-1.5 (pool). Re-weighting terms to a linear scale made ranking worse (top-1 0.47 -> 0.32) because it mutes
the tails that veto broken conformers: **re-weighting is not a lever.**

---

## 5. Production configuration and the experiment hook

In the restructured repository, all 37 runtime switches have been retired and their resolved production
values hardcoded directly across `src/jacobik/`. The retired flag switches and their evidentiary basis
are archived in `research/archive/flags/FLAGS.md`.

`src/jacobik/profile.py` serves as an empty, documented hook where future experiment flags get added.
When no experimental overrides are set, it produces the production baseline fingerprint **`5f74b14b57ed`** (IA-k finish phase; then **`273f2ccbf892`**, eta2 face-plane + haptic-cap fold-in; then **`3f75c9525a4e`**, Package H donor-pair fold-in; previously **`6cc586e17ab1`**, Li-cap fold-in, superseding Package F `615ae97d2350`).
Setting an experimental flag or override via `JACOBIK_PROFILE_OVERRIDES` modifies the fingerprint,
guaranteeing provenance tracking in logs and artifacts.

### 5.1 Hardcoded production values

| Setting | Module location | Production value | Rationale |
|---|---|---|---|
| `score_agg` | `ik.scorer.SCORE_AGG` | **"sum"** | Total log-likelihood divided by feature norm $Z$ |
| `rank_agg` | `ik.scorer.RANK_AGG` | **"sum"** | Unified solve/ranking objective |
| `solve_floor` | `ik.scorer.GMM_FLOOR_MODE` | **"thermal"** | 298 K harmonic thermal variance floors |
| `rank_floor_fit` | `ik.ik_solver._RANK_FLOOR` | **False** | Rank with solve floors (u1 standard) |
| `bend_release` | `ik.ik_solver.BEND_RELEASE` | **[0.1, 100]** | Schedule on angle_bend step size (slashes catastrophic tail) |
| `q1_tail_sigma` | `ik.scorer.Q1_TAIL_SIGMA` | **0.0** | Keep fitted table components without broad truncation |
| `q1_wall` | `ik.scorer.Q1_WALL` | **True** | One-sided quadratic barrier beyond table support |
| `q5_fold` | `ik.scorer.Q5_FOLD_MODE` | **"auto"** | Rotor symmetry folding from DB metadata |
| `q6_model` | `ik.scorer.Q6_MODEL` | **"repulsive"** | Repulsive exchange contact walls (`stats/q6.py`) |
| `nci` | `ik.scorer` | **False** | NCI calibration confirmed $\gamma_{\text{shared}} = 0.0$ on full surface |
| `bvs` | `ik.scorer` | **False** | Bond-valence coupling dead path removed |
| `ring_fold_hinges` | `ik.angle_bends.RING_FOLD_HINGES` | **False** | Validated removal (tail $-336$ kJ/mol, cal $-13$ kJ/mol) |
| `deterministic_embed`| `engine.DETERMINISTIC_EMBED` | **True** | Single-threaded xTB GFN2 embed for bit-reproducibility |
| `oxidation_model` | `core.oxidation.OXIDATION_MODEL` | **"training"** | Training conventions with explicit formal charges |
| `angle_bends` | `engine.DEFAULT_ANGLE_BENDS` | **True** | Chelate angle-bend flexibility active by default |
| `bend_per_z` | `ik.ik_solver._BEND_PER_Z` | **True** | Ligand bend prior divided by feature norm $Z$ |
| `bound_surrogate` | `chelate_embed.BOUND_SURROGATE` | **True** | Group-13 surrogate capping for rigid backbones |
| `ez_enumerate` | `chelate_embed.EZ_ENUMERATE` | **True** | Enumerate unresolved E/Z stereocenters |
| `carbene_ylide` | `chelate_embed.CARBENE_YLIDE` | **True** | Treat neutral carbenes as ylides |

The old controls `q6_agg`, `q6_pmf_wells`, `q6_rep_class4`, and `q6_m13` were removed in the 2026-10-18 cleanup.
Do not pass them through profile overrides. The historical Q6PMF modes `all`, `hbond`, and `contact` are retired.
| `local_plane_deg` | `ik.angle_bends.LOCAL_PLANE_DEG` | 20.0 | 20.0 | planarity tolerance for bend typing |
| `beam_mode_score` | `ik.beam.BEAM_MODE_SCORE` | density | density | beam ranks torsion modes by density at the mean, or by mass (weight) |
| `out_dedup` | `ik.dedup.OUT_DEDUP` | true | true | symmetry-aware RMSD dedup |
| `dedup_rmsd` | `ik.dedup.DEDUP_RMSD` | 0.5 | 0.5 | dedup threshold (A) |
| `out_window` | `engine.OUT_WINDOW` | 200.0 | 200.0 | output score window (summed nats); 0 = off |

### 5.1 Retired switches
The following `JACOBIK_*` variables raise an error (each used to set the same-named field, upper-case): SCORE_AGG, RANK_AGG,
GMM_FLOOR (-> `solve_floor`), RANK_FLOOR (-> `rank_floor_fit`), Q1_TAIL_SIGMA, Q1_WALL, Q5_FOLD, Q6_MODEL, Q6_AGG,
Q6_PMF_WELLS, HAPTIC_PER_SUB, EMBED, CHELATE_EZ, CARBENE_YLIDE, ETA2_CAP, BRANCH_BENDS, BRANCH_PRIOR, RING_OPEN_ADJ,
RING_JOINTS, DONOR_BENDS, HAPTIC_BENDS, HAPTIC_FREE_LEAN_MAX, HAPTIC_LEAN_WALL, LOCAL_PLANE_DEG, OUT_DEDUP, DEDUP_RMSD,
OUT_WINDOW. Removed variants (no field): Q6_WCA_N, Q6_WCA_T, SHAPE_CORE, SHAPE_TAIL, SHAPE_TAIL_FORM, SHAPE_TERMS, Q5_TEMPER,
RING_LEAN, RING_KEEP_FOLDS. Frozen labels (no switch): CARBON_STRUCTURAL, RESONANCE_LABELS, HAPTIC_V2, SHAPE_LABELS,
Q6_METAL_EXCL, Q6_THROUGH_METAL. List: `profile.RETIRED_ENV`.

### 5.2 Module constants that change results but are not fields (yet)
| constant | file | value | role |
|---|---|---|---|
| `CLASS_BUDGET`, `EARLY_CULL_AT` | `engine.py` | "split", 0.05 | how variants of one assignment class share the row budget |
| `_PLATEAU_WINDOW`, `_PLATEAU_ATOL` | `ik/ik_solver.py` | 8, 0.02 | convergence test |
| `_LSE_TEMP` | `ik/ik_solver.py` | 2.0 | power-mean temperature of the shape deviation |
| `_CULL_MIN_ROWS`, `_CULL_NICHE`, `_CULL_MIN_BATCH` | `ik/ik_solver.py` | 16, 30.0, 256 | row culling |
| `_Q6_LSE_TEMP` | `ik/scorer.py` | 2.0 | soft-min temperature of Q6 (legacy) |
| `_GMM_VAR_MIN` | `ik/scorer.py` | Q1 0.04, Q4 4.00, Q2/Q3 0.0025 | variance floors under `solve_floor = scorer` |
| `SHAPE_TOL_P90` | `ik/targets.py` | 0.006 ... 0.142 | shape tolerance per polyhedron |
| `THERMAL_SD`, `THERMAL_Q1_BY_DONOR` | `ik/scorer.py` | 4.9 | measured thermal widths |
| `REP_PARAMS`, `ALVAREZ_RADII`, `GFN2_CORES` | `stats/q6pmf.py` | 4.6 | repulsion parameters |
| beam width / Q1 samples / steps / lr | `solve()` args | 16 / 8 / 200 / 0.1 | |

---

## 6. Experimental modules and status

Experimental entries live in the same DB as adopted ones but are read only when their profile field is on.

| module | field | what | status |
|---|---|---|---|
| **NCI** `stats/nci.py` | `nci` | statistical term for noncovalent attractions: types "hb" (polar H...N/O/F/S/Cl/Br/I; features H...A distance, D-H...A angle), "pp" (two sp2 5-/6-rings; centroid distance, plane angle), "chp" (non-polar H...sp2 ring; H...centroid distance, angle to ring normal). Form: s(x) = ln g(x) * taper(r), g = crystal density / random-placement density (volume element r^2 sin angle), normalised to g = 1 in a shell beyond the interaction range, pseudo-count 5, ln g clipped to +-4, cos^2 taper over the window. Ladder: hb (donor el, acceptor el) -> (*, acceptor el) -> (*, *); pp, chp pooled. Candidates by topology only (sp2 5-/6-rings = ring atoms with <= 3 ligand neighbours); class-5 pairs only ("far" = > 4 ligand bonds or no ligand path). Approximately 0-50 candidates per complex; multiplicity and overlapping physics still require energetic calibration. Own score component "NCI", summed, /Z. Replaces Q6's polar-H wells only when hb is enabled; pi-only selections retain them. Needs refit `ncihist` and NCI entries. | built, default off. **Earlier census admitted** hb (5.5 % of training), pi-pi (6 dev80 crystals, 4 missed), CH-pi (11, 8 missed; 29.6 %); agostic (2 crystals, 2.2 %) and halogen bond (0) not admitted. **Eligibility fixed and tables regenerated 2026-10-18:** cis D-M-A pairs (donor atom and acceptor both bonded to the metal) peak at ~2.5 A / 88-112 deg in ('hb','*','Cl'/'Br'/'F') and ('hb','N','N'); now excluded from the shared H-bond candidate generator (Q2 prices that geometry). Corrected full-corpus regeneration and updated census are complete (11.2): 13 candidate tables, NCI still off. Independent energetic calibration completed (11.2); no component advances. Physical-width and balanced-fit hypotheses remain open. |
| **BVS** `stats/bvs.py` | `bvs` | bond-valence coupling QV: V = sum over sigma M-D bonds of exp((R0 - R)/0.37) ~ OS. Reference ratio log p_obs(x) - log p_indep(x) = -c/2 (x - mu)^2, c = max(0, 1/s_obs^2 - 1/s_indep^2), x = V/OS (V/median V without OS). p_indep from within-key bond shuffles (3). R0 ladder (metal, el) -> (pg, el) -> (el) (444 entries); V levels (metal, OS) -> (pg, OS) -> (metal) (124, 21 with c = 0). One term per complex, sigma-only complexes. Strength (nats at 10 % deviation): lanthanides 1.8-2.6, Ti(IV) 0.87, Zr(IV) 0.52, Ni/Zn/Pd(II) 0.3, Cu(I) 0.14, Ta(V) 0.12, Pt/Re/Fe/Ru ~0. | built, default off. **Gate FAIL** +0.49 [+0.02, +1.18] (better/worse 0/2; 11 of 237 pools change pick). The term itself is calibrated (crystal QV mean -0.31 nats vs -0.35 for a correct reference ratio); the failure is pool-limited and R0 per (metal, element) pools geometries/spin states (Ni(II) square-planar vs octahedral) and pi-acceptor C donors (EBUXAE). Kept, not dropped. Future test in 11. |
| **m13** `q6_m13` | `q6_m13` | metal 1-3 contacts (M...X with X bonded to a donor) in Q6: wall fitted on the 1-3 distances, or core rows in the pmf | built, default off; arm u1+lean+m13 not adopted (6.1 ledger) |
| **class-4 repulsion** | `q6_rep_class4` | physical repulsion also on 1-5 pairs | measured in N1, not kept: gate -1.32 vs -1.71 (core-only), curvature marginally better, both within noise; simpler form without double counting wins |
| **mass beam** | `beam_mode_score` | beam by mode mass instead of density | tested (arm u1+mass), not adopted; real need is metal-aware seeds (S3) |
| **q6 modes `all`/`hbond`/`contact`** | `q6_pmf_wells` | earlier PMF forms | all lost (6.1); modes still in code; deletion pending (CHANGELOG 10-17) |
| **donor_bends, branch_prior, eta2_cap** | same | off in both profiles | no benefit shown |

---

## 7. Refit pipeline and DB entry types

One CLI: `python refit/run_refit.py <stage...> [--workers N] [--force] [--install]`. Stages skip themselves when their
outputs exist. Config keys: `root`, `tm_dir` (`data/tmQM`), `ln_dir` (`data/refit/source/lnQM`), `refit_dir` (`data/refit`),
`final_dir` (`final`), shards of 4000, splits X1-X3 + LNf, held-out 1 in 10 by CSD-code hash.

| stage | does | output (`data/refit/`) |
|---|---|---|
| `audit` | oxidation-state label audit (no geometry) | `reports/os_audit.json` |
| `clean` | contact cleaning with data-derived thresholds (through-bond angle, second-sphere valley) + sharding; runs the LnQM export if missing | `clean/`, `reports/contacts/` |
| `extract` | ONE pass over each shard: Q1-Q6, QO, QG/QP/QB observations, Q6 histograms (`q6hist`), m13 histograms (`q6hist_m13`), NCI histograms (`ncihist`), merged per dataset; **benchmark complexes removed** (`refit/benchmark_holdout_codes.json`: DEV32 + N48 screen + RING10) | `observations/` |
| `select` | held-out selection: Q1 ladder variant x pooling relevance r in (4, 10, 25), Q2 r, S_orient hold-out; Q1 chosen among state-keeping variants (`q1_require_state`) | `reports/holdout/selection.json`, `pool_Q1.csv`, `pool_Q2.csv` |
| `fit` | the full fit, ONE DB with every entry (below); writes `constraints.json` + `.meta.json` + `fit_summary.json` | `<final_dir>/` |
| `check` | audits: far Q1 components, Q6 walls, entries per quantity | `reports/check.json` |
| `install` | (only with `--install`) copy the DB into `src/jacobik/` with a backup | |

Separate and expensive: `refit/thermal_widths.py` (GFN2 Hessian widths, 4.9); `refit/export_lnqm.py` (lnqm.h5 -> tmQM layout).
The sidecar `.meta.json` must stay next to the DB: the runtime reads the Q1 variant and model versions (Q5 folding on/off)
from it.

### 7.1 Fitting statistics (what `fit` computes)
- **Evidence weights:** the m rows of one complex in a key get weight 1/m, so each complex counts once; n_eff = sum of weights.
  Keys with < 30 rows are not fitted alone (Q3-Q6). Every row is also copied to each coarser ladder key (cap 500 rows per
  (ancestor, L0 bucket), reservoir with a fixed seed). Q6 has no ancestors.
- **1-D mixtures (Q1, Q3, Q4, Q2 in cos space):** weighted EM for K = 1..Kmax; a K whose fit has a component with
  n_eff * pi < 10 is not selectable, K <= n_eff/10; K by BIC (p = 3K - 1, +1 for nu); variance clamp 0.002 A / 0.002 cos / 0.2 deg
  (precision clamp, not added). Q2 uses Student-t (nu in {2, 4, 10} by likelihood).
- **Partial pooling (Q1, Q2):** keys with >= 50 complexes are fitted on their own; every other key is the MAP adaptation
  (Reynolds 2000) of its parent: alpha_k = n_k/(n_k + r), mean/variance/weights shrink toward the parent. r and the Q1
  ladder variant are chosen by held-out log-likelihood per complex. Pooled entries are valid at any n.
- **Q5:** weighted EM for von Mises, kappa = A_1^-1(R_bar), well merging as 4.5.
- **QO:** full-covariance GMM, 4.4.
- **Q6 walls (legacy):** mu = Q_0.05 evidence-weighted non-bonded distance quantile, sigma from the 5 %-1 % spread (floor 0.05 A); metal pairs also
  pooled to (period group, element). Ladder (A, B) -> (pg, X); fallback mu = r_A + r_B, sigma 0.25 (metal pair mu x 1.18,
  sigma 0.12). (The 0.1 % onset form mu = Q_0.001 was tried and lost; the production DB is the "q6p5" 5th-percentile one.)
- **Q6P:** uncapped 0.02 A histograms to 8 A per (key, bond-separation class, polar-H angle class) -> PMF tables (N_MIN 3000
  pairs per key, 25 counts per 0.06 A window), onset r0 = 0.1 % quantile.
- **Held-out check for QO:** crc32(code) mod 10 = 0 complexes.

### 7.2 DB entry types
| prefix | content | read when |
|---|---|---|
| `Q1:` ... `Q5:` | mixtures per key (Q2/Q3 in cos; Q5 von Mises) | always |
| `QO:` | S_orient full-covariance mixtures | `orientation_model = "auto"` and present |
| `Q6:` | wall parameters (mu, sigma) | `q6_model = wall`; also fallback for pairs without a pmf table |
| `Q6P:` | repulsive-Q6 core tables and legacy wall fallback data | current `q6_model = repulsive` |
| `QV:`, `QVR:` | bond-valence V levels and R0 | `bvs` |
| `NCI:` | NCI tables | `nci` |
| `QG:`, `QP:` | geometry and partition priors | **read by nothing at runtime** (11.5) |
| `QB:` | candidate donor observations; extracted but not fitted | no runtime table |

Default DB: **`data/db/constraints_2026-11-01.json`**, with its adjacent `.meta.json` (eta2 S_orient keys refit on 2026-10-18; every other key equals `constraints_2026-10-18.json`).
The 2026-10-18 comparison confirms all 445,868 keys and equality to 1e-9 outside 50 disabled QV/QVR entries.
The historical predecessor `constraints_refit_q6p5.json` is retained as an earlier baseline. Its three add-on DBs,
the v6 evaluation DB and `final_verify/` are archived under `data/_archive/`; there is no runtime `extends` handling.
Pool DB migration provenance is retained in `data/energy_scale_refit/pools_db_migration_2026-10-18.json`.
Retired refit patch scripts are in `archive/refit_patch_scripts_2026-10-17.zip`.

---

## 8. Benchmarking

### 8.1 Sets
| set | what | size |
|---|---|---|
| dev32 | development tmQM sample, `tools/q6_ablation/dev_tmqm/` | 32 |
| N48 | the 48-complex screen sample (`tests/regression/fixtures/n48/`) | 48 |
| **dev80** | dev32 + N48; **75-76 valid complexes x 3 seeds** (the working set for every arm) | 80 |
| heldout28 | random tmQM sample (2026-09-23) minus the panel; never used for diagnosis or tuning. MDE ~70 kJ/mol on the mean | 28 |
| ring10 | macrocycle/ring cases (`tests/regression/fixtures/ring10/`). MDE ~190 | 10 |
| panel | 24-case accuracy/speed panel (`tests/regression/run_panel.py`, `baseline.json`); MDE ~5 | 24 |
| training exclusion | DEV32 + N48 screen + RING10 codes are removed from every fit (`refit/benchmark_holdout_codes.json`) | |

### 8.2 Tools
Run from repo root in the pixi env with `MKL_THREADING_LAYER=SEQUENTIAL` and `PYTHONPATH=.`. Pools are
`<OUT>/pools/<arm>/<set>/<code>_s<seed>.npz` (+ `.mace64.json`); `OUT` = `data/energy_scale_refit` (`JACOBIK_BUDGET_OUT`), pools' DB =
`JACOBIK_BUDGET_DB` (recorded in `<OUT>/pools_db.txt`; a mismatching DB is refused).

    python research/bench/tools/jdev/budget/pools.py solve --sets dev32 n48 --seeds 11 12 13 --start etkdg [--isomer ref] [--tag NAME]
    python research/bench/tools/jdev/budget/pools.py mace --dtype float64 [--dirs ARM ...]          # GPU, MACE-POLAR-1-S
    python research/bench/tools/jdev/budget/tracks.py --arms u1c u1c+rep --energy mace64             # paired arm comparison + problem table
    python research/bench/tools/jdev/budget/gate.py --sets dev32 n48 --start ARM --energy mace64 --override jacobik.stats.q6pmf:PMF_WELLS=repulsive
    python research/bench/tools/jdev/budget/forces.py compute --sets dev32 n48 --top 4 ; ... report   # curvature
    python -m tools.jdev case NAME / diagnose NAME / audit NAME [--relax 300]            # one complex

- **pools.py:** one solve per (complex, seed) under the active profile; stores every returned conformer and per-term sums, Z, features.
  `--start full_oracle|junction_oracle` oracles; `--isomer ref` Track I; `--targets crystal` crystal-positioned statistics.
- **gate.py:** re-scores stored pools with a changed scorer (`--override ATTR=VALUE` for `jacobik.ik.scorer` attributes or
  `module:ATTR`; `--profile`). One scorer per (complex, arm), reused across seeds. Each pool checks that the base re-score
  equals the stored ranking score (logged if > 1e-4; the base must reproduce the stored #1). Paired per complex with a
  cluster-bootstrap CI. It does not measure effects on generation.
- **tracks.py:** `ARMS` maps names to pool folders (O etkdg, I etkdg_isoref, O+int full_oracle, u1 etkdg_u1, u1c
  etkdg_u1_contact, u1c+rep etkdg_u1_rep, ...). Per pool: #1 dE, floor, regret; per case seed medians; paired tests
  (`bench.paired_test`, cases resampled with seeds kept together).
- **Other tools** (all in `research/bench/tools/jdev/budget/`; full table in `tools/jdev/README.md`, inventory in `tools/INVENTORY.md`): `forces.py`
  (MACE curvature vs each term), `q6_diag.py`, `q6_dists.py` (Q6 diagnostics), `bvs_check.py`, `gfn2_dimer_scan.py`,
  `nci_census.py`, `oracle_targets.py`, `isomer.py`, `internal_stereo.py`, `export_xyz.py` (structures to `data/inspect/<CODE>/`),
  `ensemble_stats.py`, `validity.py`, `start_error.py`, `analyze.py`, `decomp.py`, `xtb_stage.py`, `xtb_dof.py`, `rmsd_stage.py`,
  `coord_audit.py`, `coord_report.py`.
- **Measurement settings:** MACE-POLAR-1-S (`tools/jdev/MACE-POLAR-1-S.model`) in float64; float32 only for quick looks. GFN2-xTB
  (`xtb`, CPU) for relaxations. A CUDA solve is a different float path than a CPU one (pool folders carry no separate suffix; compare only within one
  device/path). The panel path (`run_panel.py`) uses persistent loky processes, MACE energy-only mode, float32 default.
- **CUDA:** the fast path (recorded step, one-pass kinematics) makes a solve 2-4 s on an RTX 2060; `JACOBIK_GPU_REPLAY=0` / `JACOBIK_GPU_FAST_FK=0`
  restore eager. All seeds of a complex can share one pass (`multiseed.py`).
  The panel exposes `--seeds-in-one-pass`; budget `pools.py` currently calls separate solves. One GPU job does not
  require serial seeds. Before using the batched harness for a new budget arm, verify current u1 E/Z enumeration,
  per-seed results and pool/scorer metadata, with the same float path. Keep an in-flight arm's execution unchanged.
  Shared embedding matters: IWUFIS spends minutes in the eight-candidate ETKDG fallback for four E/Z combinations,
  already present in the prior u1 arm; it is not evidence of a new oxidation-mode slowdown.

### 8.3 Error budget (METHODOLOGY 2026-10-15 study, kept)
For each complex dE(#1) = **pool floor** (lowest dE the solver returned) + **ranking regret** (#1 minus floor). The floor,
by sequential constrained GFN2 relaxations (`xtb_dof.py`), splits into: torsions / peripheral placement (relax with
internals and polyhedron frozen), coordination sphere (relax with internals frozen), frozen ligand internals (full
relaxation), and basin (full relaxation minus the relaxed reference).

| line | moved by | not moved by |
|---|---|---|
| ranking regret | ranking score, term weights, term shapes (the gate measures it exactly) | anything in generation |
| basin | search (assignments, beam, torsion modes), ranking | precision of statistics |
| coordination sphere | Q1/Q2 targets and the solver's reach (bite angles) | Q5/Q6 shapes |
| torsions / periphery | Q5, Q6, S_orient, the optimiser | Q1/Q2 |
| frozen internals | the embed (ETKDG / GFN2 free ligand), bends | every score term |

Measured (v6 DB, 348 pools, 116 complexes; shares of summed SP dE, cluster-bootstrap CI): torsions/periphery 33 % [26-42],
coordination sphere 26 % [19-32], frozen internals 34 % [28-39], wrong basin 8 % [6-10]. Median #1 = floor + regret:
N48 70 = 62 + 6, dev32 74 = 66 + 5, heldout28 104 = 82 + 8, ring10 154 = 142 + 5. With crystal internals (oracle) the floor and #1 roughly halve.
Problem set (10-16 b): internals 61 % of the open #1, crystal-positioned statistics 10 % (only together with crystal internals),
residual with everything at the crystal 36 %. A local GFN2 relaxation of the #1 (breaks rule 1; measurement only) takes the median from 82 to
5 kJ/mol: the benchmark measures how close the pick is to the right structure. 64 % of picks relax into the reference basin (<= 10 kJ/mol);
heavy-atom RMSD median 0.97 A (some picks are the mirror image, which an energy metric cannot see).
**Yardstick ceiling:** MACE vs GFN2 single points within a pool: rho 0.92, rho_lo 0.71; their lowest conformer agrees within 1 kcal/mol in only 65 % of pools.

**Noise:** identical code and DB on CUDA between two runs: per-case |dE| moved by a median 9 kJ/mol (p90 80), ~1 case in 6 crossed +-10 either way,
set medians moved up to 29 (ring10 192 -> 163), whole-ensemble rho up to 0.07. Always paired.

**Coordination sphere of the picks (116):** half of M-D distances > 0.05 A off (p90 0.20); where off, Q1 prefers the crystal value in only ~half
and 70-79 % use multimodal pooled tables (wrong mode wins). D-M-D 5-30 deg off for 31 % of pairs; Q2 prefers the solved angle in 72 %
(moderate distortions are not penalised). Crystal internals barely reduce these errors, so they are statistics and reach problems.
Large angle differences are mostly swapped identical donors (27 left after automorphism; 2 permutation-invariant).

### 8.4 Curvature (E1) and the energy scale of terms
`forces.py`: central differences along the solver's soft DOFs (per-ligand translations, rotations, rotatable bonds) at the crystal and the top-4
pool conformers, 80 complexes (15,055 directions). Ratio = energy curvature (kT) / score curvature (1 = energy-like; < 1 = score too stiff; > 1 = too soft).

| term | legacy ranking | legacy solve | u1 walls | u1c (contact) | u1c+rep (N1) | **u1+nci (pool / crystal)** |
|---|---|---|---|---|---|---|
| Q1 | 0.06-0.15 (stiff) | ~4.5x soft (0.2 A floor) | 1.33 | 1.43 | 1.36 | **1.39 / 1.49** |
| Q2 | 0.25 | 0.5-0.65 | 0.93 | 1.28 | 1.00 | **0.99 / 1.96** |
| S_orient | 0.8-1.0 | 0.3-0.45 | 0.87 | (Q3) 1.41 | (Q3) 1.03 | **(Q3) 1.01 / 1.04** |
| Q5 | ~0.85 between wells; 0.2 on Q5-dominated directions | 0.25 | 0.87 | 1.82 | 1.09 | **1.11 / 0.86** |
| Q6 | ~3x soft | ~0 | 0.83 | 0.72 | 1.45 | **1.43 / 1.56** |
| NCI | n/a | n/a | n/a | n/a | n/a | **1.30 / 1.02** |
Spearman of stiff directions 0.6-0.7 (legacy), 0.87 (u1c+rep), **0.867 pool / 0.835 crystal (u1+nci)**. Joint NNLS curvature scales (kT per unit score curvature, pool top-K): Q1 0.84, Q2 0.56, S_orient 0.43, Q5 0.48, Q6 1.14, NCI 0.43. Energy-scale lessons: Q1/Q2 narrow table components (needles) make the legacy ranking too stiff;
the legacy solve's Q1 is too soft because of its 0.2 A floor; "the cores are too soft" holds only for the solve objective's Q1; Q5 acts on torsion
directions alone and the legacy over-sharp wells (74 % sharper than thermal) are what the thermal cap corrects; $C^2$ bicubic Hermite interpolation with thermal-compliance sharpness regularization and matched-baseline energetic calibration (`c_hb = 0.2583`, `S_CLIP = 8.0`) places NCI's dominant curvature ratio at **1.30** on pool conformers and **1.02** at the crystal.

### 8.5 Ensembles and metrics by level
| level | metric | yardstick | v6 / refit (dev80) | **u1+nci (dev80, 228 pools)** |
|---|---|---|---|---|
| isomer | rho_isomer: class-best score vs class energy (pools with >= 3 classes) | MACE SP / GFN2 relaxed | SP 0.60 / 0.59; relaxed 0.48 / 0.20 | **SP 0.764** |
| within isomer | rho_within: score vs SP inside each class (classes with >= 4 members) | MACE SP | 0.68 / 0.68 | **0.685** |
| basin | rho_basin: score vs relaxed energy over basins; pick in the lowest basin (1 kcal/mol); reference basin present | GFN2 relaxed | 0.22 / 0.15; 47 % / 41 %; 88 % / 89 % | inherited (fixed pool) |
| strain | strain of the #1 (SP - relaxed) | GFN2 | median 103 / 88 kJ/mol | inherited (fixed pool) |
| whole pool | rho_all, rho_lo | MACE SP | 0.68 / 0.69; 0.45 / 0.47 | **0.704; 0.505** |
| validity | share of returned conformers > 1000 kJ/mol | MACE SP | refit 10.6 % (legacy 9.9 %, u1 walls 2.2 %) | **1.25 %** |
Facts: tmQM references are GFN2 minima of the crystallised isomer, not necessarily the lowest. 16 % of v6 pools (7 % refit) contain a basin > 10 kJ/mol
below the relaxed reference (largest PEWBOL -90). The #1's own basin sits a median +2.8 (v6) / +5.6 (refit) kJ/mol above the reference basin; strain
(103 / 88) dominates dE. Forcing the reference isomer (Track I) leaves the #1 median unchanged (+0.1): isomer choice is not the hidden variable.
60-80 % of returned conformers relax to distinct basins (0.5 A dedup merges few snapshots of one basin). A Boltzmann/basin mode needs rho_basin well above the
yardstick noise (today 0.15-0.22), the score's curvature matched to the energy's, and a CREST-like 6 kcal/mol window on RELAXED energies. Until then `solve()`
stays a conformer ensemble.

### 8.6 Golden regression (the no-result-change check)
`python -m tools.jdev.golden record` (`research/bench/tools/jdev/golden.py`) writes `research/data/golden/<profile>.npz` for legacy-2026-10 and u1 (each in its own process; the profile is read at import):
RESCORE of stored pools (`etkdg` for legacy, `etkdg_u1_rep` for u1; 11 complexes x 3 seeds: ranking score and per-term breakdown) and SOLVE (seed 11,
CPU: OJIMIF, HUYNEU, CBTYCO, KIQYEQ, SOKVET, YAVFEG, aqua_La3_8H2O). `python -m tools.jdev.golden check` recomputes and compares scores/terms to 1e-6 and
coordinates to 1e-5 A; exit 1 on any difference. Pitfalls: (a) xtb must be on PATH (without it every solve silently uses the MMFF embed and golden fails);
(b) record/check on a quiet machine: GFN2 embeds are non-deterministic under CPU load; golden runs with `OMP_NUM_THREADS=1`. Run it after every refactor and every
change meant not to alter results.

### 8.7 Tests
`PYTHONPATH=. JACOBIK_TEST_DB=<db> pytest tests -q --ignore=tests/regression`, under legacy and under u1 (`JACOBIK_PROFILE=u1`). `tests/regression/run_panel.py`
is the panel runner. Dev workflow: `python -m tools.jdev guards|panel|case|diagnose|audit|fingerprint`.

---

## 9. Scoreboard and plan 3

### 9.1 Scoreboard (dev80; 75-76 valid complexes x 3 seeds, varying by comparison; MACE kJ/mol)
| metric | legacy (10-16, f32) | u1 walls (f32) | u1c: contact Q6 (f64) | u1c+rep: N1 (f64) | u1_pure_rep: no Q6 wells (f64) | **u1+nci: adopted main (f64)** | target |
|---|---|---|---|---|---|---|---|
| #1 median / mean | 74.8 / 114.7 | 66.3 / 95.3 | 72.0 / 101.7 | 65.1 / 88.0 | 64.3 / 87.0 | **64.3 / 86.3** | ~55-60 / 65-75 |
| floor median / mean | 58.3 / 89.6 | 59.8 / 80.9 | 60.0 / 81.4 | 54.6 / 73.1 | 53.2 / 72.5 | **53.2 / 72.5** | ~45-50 / 55-65 |
| regret median / mean | 7.2 / 25.1 | 6.0 / 14.4 | 7.8 / 20.4 | 3.6 / 14.9 | 3.8 / 14.6 | **3.8 / 13.9** | <= 5 / <= 10 |
| top-1 (within 1 kcal of floor) | 0.400 | 0.427 | 0.397 (gate) | 0.526 | 0.518 | **0.518** (valid arm pools) | >= 0.47 |
| #1 >= 100 / 200 / 300 | 25 / 13 / 6 | 23 / 10 / 3 | 24 / 11 / 4 | 21 / 10 / 1 | 22 / 10 / 1 | **22 / 10 / 1** (case seed medians) | <= 16 / 5 / 1 |
| H-bond subset (12 cases) med / mean | n/a | PEWBOL 428 | 106.3 / 142.3 (PEWBOL 337) | 80.6 / 107.6 (PEWBOL 256) | 53.0 / 101.5 (PEWBOL 282) | **44.6 / 88.4** (PEWBOL 256, YIYXEJ 10.4) | PEWBOL <= 220, subset -15 vs walls |
| curvature (E1) | see 8.4 | | Q5 1.82 | every term 1.0-1.45 | every term 0.99-1.43 | **every term 0.99-1.43 (NCI 1.30 pool / 1.02 cryst)** | 0.7-1.5; Q5/Q6 <= 1.3 |
| near-crystal Q6 / NCI per complex | n/a | walls 9.2 kT | ~0 | 0.82 kT mean, p90 3.3 | Q6 2.74 kT (p90 6.72) | **Q6 2.74 kT (p90 6.72); NCI 0.65 kT (p90 1.94)** | <= 10 kT |
| seed spread (worst - median) | n/a | 1 complex > 50 | 6 complexes > 50 | 4 complexes > 50 | 3 complexes > 50 | **3 complexes > 50** | 0 complexes > 50 |
| conformers > 1000 kJ/mol | 9.9 % | 2.2 % | 1.76 % | 1.26 % | 1.25 % | **1.25 %** (mean pool fraction) | <= 2 % |
| solve time mean | 2.7 s | 6.8 s | not recorded | not recorded | not recorded | not recorded | <= 8 s |
("not recorded": no number is held for these in the documents or changelog; measure them when next needed.)
Paired comparisons on `dev80` (228 valid pools / 76 complexes, `data/_archive/refit_reports_2026-10-20/n2_comprehensive_benchmark_2026-10-18/comprehensive_benchmark_dev80.json`):
- **`u1+nci` vs `u1c+rep`:** `#1` set median `65.15 -> 64.29 kJ/mol` (`-0.86 kJ/mol`), paired mean `-1.64 [-4.87, +0.37] kJ/mol` (`3` better / `2` worse `> 10 kJ/mol`), floor `-0.63 [-3.03, +0.88]`, regret `-1.01 [-2.94, +0.31]`; H-bond subset median `80.57 -> 44.64 kJ/mol` (`-35.93 kJ/mol`), mean `107.62 -> 88.38 kJ/mol` (`-19.24 kJ/mol`).
- **`u1+nci` vs `u1_pure_rep` (purely repulsive Q6 without Q6PMF polar-H wells):** `#1` paired mean `-0.69 [-1.78, +0.01] kJ/mol` across all 76 complexes (`-13.07 kJ/mol` on the H-bond subset, `-4.36 kJ/mol` on the 12-case targeted panel), **`2` better / `0` worse (`> 10 kJ/mol`)** — rescuing `PEWBOL` (`281.9 -> 256.0` median, `-30.51 kJ/mol` mean; seed 12 `314.9 -> 239.6 kJ/mol`) and `YIYXEJ` (`31.4 -> 10.4` median, `-21.97 kJ/mol` mean) with **zero regressions** across all 76 complexes.
- **`u1+nci` vs `u1c+rep+os` (which retained legacy Q6PMF 1D polar-H wells):** `#1` median diff `0.0`, mean `+0.37 [0.00, +0.89] kJ/mol` (MDE `0.66`, no detectable change), confirming that the 2D $C^2$ regularized/calibrated `NCI` term cleanly replaces the deprecated `Q6PMF` data-region attractive wells while reducing complexes with seed spread `> 50 kJ/mol` from `4` to `3`.

### 9.2 Stages: targets, results, status
Optimistic gains (#1 mean vs the base each arm is measured on): N1 -6 to -12; S2 -1 to -4 (revised -0.5 to -2); N2 -2 to -5 (H-bond subset -10 to -20);
N3 -1 to -4; S3 -5 to -10; S4 -15 to -25 (the only stage expected to move the median, -8 to -12).

| stage | what | status |
|---|---|---|
| U0-U3 | profiles, thermal widths, u1 objective, u1 arm | done (u1 walls: #1 66.3 / 95.3 vs legacy 74.8 / 114.7, f32) |
| S1 contact Q6 | GFN2 core below each key's onset + polar-H wells | arm missed its rule (+6.3 [-3.4, +17.6], regret +6.0 [+2.0, +10.6]) but fixed the H-bond cases (PEWBOL 428 -> 256); superseded by N1 + N2 |
| S2 offline | bond-valence coupling checks 1, 2, 4 | coupling 1.53x overall: lanthanides 4-5x, early metals ~2x, Ni/Zn/Cd ~1.7, Pd 1.29, Cu 1.32; none for Fe, Ru, Re, Pt, Rh, Ir, Os, W; flags the Ta stretches (XAQRUC -0.80, VUJYEG -0.68), misses KIQYEQ, YIYXEJ, OVODUZ. Check 3 (haptic) not done |
| **N1 repulsive Q6** | Historical arm: physical nonpolar Q6 repulsion with the legacy polar-H reward behavior. The later production Q6 removed those data-region attractions and added the separate NCI term. | **Adopted 10-17; superseded by N2 on 10-18.** Historical dev80 #1 mean 87.98 vs 101.72 for `u1c`. See the current baseline table above for `u1_pure_rep` and `u1+nci_hb`. |
| S2 bond valence | R0/V tables, QV term | built; **gate FAIL** +0.49 [+0.02, +1.18]; no arm; kept off (6) |
| **OS consistency** | query ligand charges match extraction conventions | **ADOPTED 10-18**, owner confirmed keep. Dev agreement 67/80 -> 80/80; usable sampled tmQM 6140/7353 -> 7353/7353. Gate valid picks unchanged. MACE64 arm #1 -2.01 [-5.14,-0.14], MDE 3.69 (limited precision); floor -0.63 [-3.03,+0.88], regret -1.38 [-3.22,-0.19]. YIYXEJ seed-median 86.6 -> 10.4, KIQYEQ 290.3 -> 222.3. New base `etkdg_u1_os`; legacy frozen |
| **N2 NCI v2 ($C^2$ + regularized + calibrated)** | 2D $C^2$ bicubic Hermite surface (`A1`) + thermal-compliance Hessian regularizer (`A2`) + `S_CLIP=8.0` + C-H directionality/sibling softmax (`C3+C4`) + $5\text{ \AA}$ `pp` cutoff (`C5`) + matched-baseline energetic calibration (`c_hb=0.2583`, `c_pp=0.3694`, `c_chp=0.0`) | **ADOPTED 10-18 as main solver method (`u1` default).** Unseen `v3` validation (`hb`): 14 better / 1 worse vs Pure-N1 ($\Delta\text{RMSE} = +0.0604\text{ kJ/mol}$, 95% CI `[+0.0263, +0.0987]`, Spearman `0.416 -> 0.633`, regret `0.572 -> 0.395 kJ/mol`). On `dev80` (228 pools): `#1` `64.3 / 86.3 kJ/mol` (`-1.64 kJ/mol` mean vs `u1c+rep`; `-0.69 kJ/mol` mean, `2` better / `0` worse vs `u1_pure_rep`; H-bond subset `44.6 / 88.4 kJ/mol`, `-13.1 kJ/mol` mean vs `u1_pure_rep`; `PEWBOL 256.0`, `YIYXEJ 10.4`). E1 curvature ratio `1.30` (pool) / `1.02` (crystal) |
| N3 | further NCI types | nothing admitted now (agostic, halogen bond not admitted); revisit if the census changes |
| S3 search coverage | measure reachability (start u1 from legacy good structures for YIYXEJ, OVODUZ, JEFJUZ, XIPLUD, FIGHOT) and distinct torsion arrangements per seed; fixes (own arms): context-aware beam (partial score sees metal and placed ligands, adopted Q6), unshared torsion rows per seed and isomer | not started. Target: 0 complexes with seed spread > 50, floor mean -5 to -10 |
| S4 ligand internals without relaxation | internals decomposition (`oracle_targets.py`, `internal_stereo.py`), offline internal-error metric; candidate A generalised bound-state (H-capped) embed, B ring pucker / macrocycle fold enumeration | not started. Target: #1 -15 to -25 mean, median -8 to -12, wrong configurations <= 8/80 |
| S5 held-out + cleanup | default switch to `u1` with purely repulsive Q6 + $C^2$ calibrated NCI; deprecate Q6PMF attractive wells & unused Q6 flags; archive superseded reports | **default switch & Q6PMF/report cleanup DONE 10-18**; heldout28 / ring10 evaluation remains open |

Order of work (GPU one job at a time; CPU/code alongside): OS/Co-EDTA completed -> N2 fix + census + gate + arm
against `etkdg_u1_os` -> S2 revision (if wanted) -> S3 -> S4 -> S5.

### 9.3 Earlier plan history (cleanup plan, 2026-10-17)
P0 safety net (golden), P1 rename (`src_next` -> `src`, etc.), P2 source cleanup, P3 tools inventory (`tools/INVENTORY.md`, 46 files in
`archive/code_2026-10-17.zip`), P4 one refit pipeline, P5 data archive (manifest first; zip then delete: v6-era `energy_scale`, `q6_ablation`, abandoned arms'
pools), P6 docs (this document), P7 final check (tests under both profiles, golden, gate base re-score with 0 mismatches, GPU smoke solve).

---

## 10. Ledger of discarded variants (one line each, with why)

### 10.1 Steric (Q6) forms
- **0.1 % onset walls** (mu = Q_0.001): +52 kJ/mol worse in arms 10-14 and 10-16; describes the data better than the 5th-percentile wall but charges ~7x fewer nats for a real clash.
- **WCA core on the walls** (`q6_model = wca`): gate FAIL +8.0 [+0.2, +18.2] (10-15) and arm worse (+3.8 median, 8 better / 11 worse); a statistical wall reshaped is not physics.
- **PMF "all"**: rewards compaction (BIDHEB: Q6 -70 kT over 819 pairs, acyl O folded onto Pd, dE 2458).
- **PMF "hbond"** (wells for H-bonds only, every other pair max(w, 0)): outward pressure, +37.8 kT per near-crystal complex; gate +8.6 [-0.5, +23] n.s., leaning worse. Mechanism: w = -ln(p_key/p_ref) with p_ref pooled over keys is +-1 kT composition noise; clamping makes it one-sided and summed over ~1500 pairs.
- **PMF "contact"** as its own mode: cut the flank between the onset and the bulk of each distribution, score 1.7x soft along torsions, Q5/Q6 exchange rate ~2.5x off, no gradient before a clash (XAQRUC); superseded by "repulsive" (the contact form remains for classes 4, m, Hp).
- **Class-4 repulsion** (`q6_rep_class4`): gate -1.32 vs -1.71, curvature within noise; double counting with Q5. Not kept.
- **Q6 data region by reference ratio against the model's own predictions (iterative Boltzmann inversion)**: proposed 10-17 b, never built; superseded by the Q6/NCI split.
- **Summed walls in a mean-aggregated solve** (`Q6_AGG = sum`): regret -14.5 [-29, -3], top-1 up, #1 n.s.; made moot by `score_agg = sum` in u1.
- **Soft-min Q6** (legacy): a contact barely counts below ~4 nats; kept only in legacy.

### 10.2 Score shape and weighting
- **SHAPE tail caps / energy-like shaping of Q1, Q2, S_orient** (`SHAPE_CORE/TAIL/TERMS`): gate FAIL; capping the Gaussian tail removes the veto of broken conformers.
- **Linear and pairwise re-weighting of the ranking** (per-term kT-per-nat weights): top-1 0.47 -> 0.32 (linear); pairwise stable but top-1 0.43, CI includes 0. Re-weighting is not a lever.
- **GFN2-Hessian torsion widths** (Q5 widths from Hessians): cost; replaced by the single thermal cap.
- **Q5 temper** (`Q5_TEMPER`): removed.
- **True fitted widths in the solve**: made the solve worse (09-27: picks 158/186/1029 vs 138): needle components give a rugged landscape. Replaced by thermal widths, which are physical.
- **Mean aggregation + 0.2 A floor** (legacy): kept only in legacy; u1 supersedes.
- **"Gaussian cores are too soft"** as a general statement: true for the legacy solve objective's Q1 only; the ranking score's Q1/Q2 were too stiff.
- **"Q5 carries no curvature"**: misread NNLS weight.
- **Quantity weights for Q1-Q5**: act as additive constants (log w inside the LSE), no effect; only Q6 and S_orient are truly weighted.

### 10.3 Embed and kinematics
- **Closest reachable pose oracle**: poses clash.
- **Hinge joints / reach phase / templated start**: stuck in local minima, superseded by the chelate-aware embed.
- **Ring-substituent leans, ring lean / keep-folds** (`RING_LEAN`, `RING_KEEP_FOLDS`): no benefit, removed 10-16.
- **Graph-level symmetry dedup of assignments**: merged assignments the frozen embed cannot interconvert (VUCVUN); replaced by embed-aware dedup.
- **Angle bends everywhere / donor bends**: worsen typical chelates; `donor_bends` off.
- **Joint-mode COD fold changes**: a ring at fixed bond lengths and angles has no continuous torsion freedom; folds come from the macro pool.

### 10.4 Interpretation errors
- **"Isomer choice of the reference is why changes look neutral"**: forcing it leaves the median unchanged.
- **"Most returned conformers are snapshots of one basin"**: 60-80 % relax to distinct basins.
- **METHODOLOGY 4.6 refit priority 1 (Q1/Q2 key resolution)**: moving the modes does not help the set; width/mixture structure is the open part.
- **"The coordination sphere dominates the floor (~48 %)"**: an early 12-pick estimate; the full sample gives ~26 %.

### 10.5 Experiments run and not adopted
- **Beam by mass** (arm u1+mass), **m13 core** (arm u1+lean+m13): no gain.
- **S2 bond valence**: gate FAIL (kept as a module).
- **Local relaxation of the #1**: large measured benefit (82 -> 5) but breaks rule 1; not adopted.

---

## 11. Future tests and open items

### 11.1 Oxidation-state deduction (adopted in u1, 2026-10-18)
Legacy radical/haptic inference mismatched 13/80 dev complexes, including neutral carbenes counted as -2 ligands.
The training-convention mode in 3.1 fixes all 13: **67/80 -> 80/80**. On the same cached inputs of a cleaned tmQM
hash sample, **6,140/7,353 -> 7,353/7,353** usable mapped structures, 1,213 fixed and zero regressed. Unknown labels,
benchmark exclusions and mapping failures are separately counted; this is sampled tmQM agreement, not energy validation.
Existing extraction labels remain unchanged. Both profiles pass 294 tests, with 3 expected failures each; u1 tested
with the candidate override. Compatibility golden passes 245 arrays/profile with zero differences before adoption.

Gate: 240 base re-scores, zero mismatches; all picks on valid inputs unchanged. Predicted ranking gain was missed.
The full dev80 generation/MACE64 arm (`etkdg_u1_os` versus `etkdg_u1_rep`) completed: #1 mean -2.01 [-5.14,-0.14],
MDE 3.69; floor inconclusive, regret -1.38 [-3.22,-0.19]. Changed cases -15.34 mean; unchanged controls +0.01.
Adopted for correct deduction with a modest favorable energy signal and no material #1 regression; effect-size
precision is limited. Legacy stays frozen. Adopted-default focused checks: 30 passed. Golden: 237/245 unchanged;
8 expected KIQYEQ arrays changed and accepted after inspection, old baseline archived with verified read-back.
No second full recomputation; legacy baseline unchanged. See `data/_archive/refit_reports_2026-10-20/golden_u1_adoption_2026-10-18.json`.
Co-EDTA was a separate API default issue: angle DOFs repair max Co-D 2.56 -> 1.98 A without weakening the guard (3).

### 11.2 NCI (N2) — Completed and Adopted (2026-10-18)
**Phase 1–3 Continuation, Calibration & Adoption (2026-10-18):**
1. **Smoothness (`A1`) & Thermal-Compliance Regularization (`A2`):** Replaced piecewise-bilinear grid interpolation in `jacobik.stats.nci` with a $C^2$ bicubic Hermite spline (`_bicubic_hermite_eval`) with exact Neumann zero-slope boundary conditions at folded angular boundaries ($\theta \in \{0^\circ, 180^\circ\}$ for `hb`; $\theta \in \{0^\circ, 90^\circ\}$ for `pp`/`chp`) and a $C^2$ quintic radial taper (`1 - 10x^3 + 15x^4 - 6x^5`), verified by `tests/test_nci_c2_surface.py`. Added MAP/penalized-Poisson 2D surface fitting with one-sided Hessian compliance ceilings ($\sigma_{T,r} = 0.191 / 0.262 / 0.351\text{ \AA}$, $\sigma_{T,\theta} = 14.57^\circ / 10.21^\circ / 8.94^\circ$ for `hb`/`pp`/`chp`, $\gamma = 1.0$, $\lambda_{\text{smooth}} = 0.05$) and aligned `S_CLIP = 8.0` with `W_DATA_CLIP = 8.0`.
2. **Descriptor & Multiplicity Corrections (`C3`, `C4`, `C5`):** Added C–H bond-to-centroid cosine-squared directionality (`C3`) and carbon-centered sibling softmax aggregation (`C4`, $\tau = 1.0$) for `chp`, plus a $5.0\text{ \AA}$ ring-centroid cutoff (`C5`) for `pp`. Kept `polar_acc` (`Hp...acceptor`) pairs on their short-range onset core (`w_core` below $r_0 \approx 1.66\text{ \AA}$) in `q6pmf` rather than non-polar `H` Buckingham repulsion ($r_m = 2.70\text{ \AA}$), avoiding false steric walls at H-bond distances ($1.8\text{ \AA}$).
3. **Matched-Baseline Calibration & Prospective `v3` Validation:** Calibrated per-type energetic amplitudes against matched within-parent MACE64 energy differences over the Pure-N1 baseline (`Q6PMF` polar-H wells OFF) on the leak-free `2026-10-18_training_calibration_v2` split (`c_hb = 0.2583`, `c_pp = 0.3694`, `c_chp = 0.0`). On unseen `v3` validation parents (`hb`, 16 complexes / 15 parents): `14` better, `1` worse, `1` tied vs Pure-N1 ($\Delta\text{RMSE} = +0.0604\text{ kJ/mol}$, 95% CI `[+0.0263, +0.0987]`, Spearman `0.416 -> 0.633`, top-pick regret `0.572 -> 0.395 kJ/mol`, force-direction agreement `85.5% -> 89.0%`).
4. **Comprehensive `dev80` Benchmark & Adoption:** On `dev80` (228 valid pools / 76 complexes, `data/_archive/refit_reports_2026-10-20/n2_comprehensive_benchmark_2026-10-18/comprehensive_benchmark_dev80.json`), `u1+nci` (`nci=True, nci_types=("hb",)`) achieves `#1` median/mean **`64.3 / 86.3 kJ/mol`** (`-0.86 / -1.64 kJ/mol` vs `u1c+rep`; `-0.69 kJ/mol` mean with **`2` better / `0` worse** vs `u1_pure_rep`; H-bond subset `44.6 / 88.4 kJ/mol`, rescuing `PEWBOL` to `256.0` and `YIYXEJ` to `10.4`), E1 dominant curvature ratio **`1.30` (pool) / `1.02` (crystal)**, stiff-direction Spearman **`0.867`**, and ensemble `rho_isomer = 0.764`, `rho_all = 0.704`, `rho_lo = 0.505`, validity `1.25%`. Adopted as the main solver method (`DEFAULT_PROFILE = "u1"`, `constraints_2026-10-18.json`).

After N2, follow 9.2. S4 internals remains the largest expected median lever; S2 is optional tail protection.
If an effect's ceiling is below the experiment's MDE, improve the design or record the limit before another arm.

### 11.3 S3 search coverage
Context-aware beam (partial score sees the metal and placed ligands, uses the repulsive Q6; the beam currently ranks by Q5 alone and a partial
Q6 pre-screen was fixed to engage in 09-28 but keeps walls); unshared torsion rows per seed and isomer; macrocycle folds (FIGHOT); needle-mode bug
(34 joints in 16/80 complexes never seeded on the dominant mode). D3 beam by probability mass was tested (no gain alone).
Metal 1-3 contact the score does not see produces the remaining obscene members (TAPWUG).

### 11.4 Silent fallbacks and strict mode
- A failed GFN2 embed is cached as MMFF for the process (`engine._gfn2_refine`), nothing records `embed_used`; fix: no caching on failure, record
  `embed_used`. Without xtb on PATH every solve silently uses MMFF.
- `engine._order_sites_like_donor_groups` only warns on a site/donor mismatch.
- Scorer stats-lookup failures become hard-coded broad priors with a WARNING (Q1 2.1 A, Q5 flat, ...); `stats._fallback_q6` / `_resolve` fall back silently;
  the benchmark input mapper falls back to element-only matching (the QOSGIP mode).
- Proposal: count fallbacks per solve and expose them; a `strict` setting raises.
- Smaller (CHANGELOG 10-17 d): `"[H" in smiles` also matches `[Hg] / [Hf] / [Ho]`; `refit_lib.merge_shards` Q6 re-cap over-represents small shards (legacy
  walls only); `ik_solver._SC_MODE` is an import-time copy; eta7/eta8 lean free zones gated off; Q1/Q3/Q4 padded mixture components are not made inert (Q2/Q5 are).

### 11.5 Unused: QG and QP
The geometry and partition priors are fitted by the refit but read by nothing at runtime. Decide: use or drop.

### 11.6 Cleanup debris
- `tests/regression/run_panel.py` has a stale `script_notebooks/MACE-POLAR-1-M.model` path in its line 34 docstring;
  the evaluator uses `$JACOBIK_MACE_MODEL_PATH` or the configured model path.
  `notebooks/archive/01_benchmark_2026-09-23.ipynb` has the same string (archive; ignore).
- `refit/run_refit.py` and `refit/refit_lib.py` are the maintained general refit pipeline. `research/refit/nci_refit.py`
  remains the targeted NCI extraction/fit tool. The one-off 10-19 NCI/HB/Q6 scripts are archived as listed in
  `archive/EXPERIMENTS_2026-10-20.md`; `doe_aqua.py` remains used by the refit notebook.
- GLOSSARY.md was removed on 2026-10-18 after reviewing sections 2, 4 and 9 against the changelog; its archive copy remains.
- README test count and any `tests_next`/`src_next` strings in notebooks and DB meta text.

### 11.7 S2 revision (when wanted)
R0 and levels keyed by coordination number and spin; pi-acceptor C donors excluded (and haptic donors: check 3); then gate and arm. Expected value is tail protection
for early metals and lanthanides only.

### 11.8 S4 and the internals line
Internals are the largest single line of the floor (34-61 %). Candidate A: bound-state (H-capped) embed generalised beyond group 13; candidate B: ring pucker /
macrocycle fold enumeration; D6: stereocentres/E-Z/ring folds (rule: user-specified stereo is kept as given; only unspecified centres/bonds are enumerated,
e.g. RDKit `EnumerateStereoisomers(onlyUnassigned=True)`; tmQM SMILES carry none, so the benchmark exercises the enumeration).

### 11.9 S5, held-out and default switch
Default profile switched to `u1` with purely repulsive `Q6` and $C^2$ calibrated `NCI` (`nci=True, nci_types=("hb",)`), unused `Q6PMF` attractive modes (`all`, `hbond`, `contact`) and obsolete `Q6` flags (`q6_agg`, `q6_pmf_wells`, `q6_rep_class4`, `q6_m13`) deprecated/cleaned up, and DB pointer updated to `constraints_2026-10-18.json` on 2026-10-18. Remaining S5 item: run `heldout28` and `ring10` once with the adopted model.

### 11.10 N1b contingency
Physical pairwise dispersion (D3/D4-type C6/r^6, Becke-Johnson damping), making Q6 a full van der Waals term. Only if a curvature check or arm shows missing contact
stiffness; owner decision first (closest step to a force field).

### 11.11 Other measured gaps
- Q1/Q2 table components narrower than thermal (about half): floored in u1, but the refit could impose a minimum component width.
- Metal 1-3 walls (M-D-X contacts): m13 module exists, not adopted.
- Boltzmann/basin mode (design: no 0.5 A collapse inside a basin, basin labels on output): after rho_basin > noise.
- Q2 at the crystal (1.87) and Q1 (1.4-1.5) are still soft in curvature; not Q6's to fix.
- KIQYEQ (Cu-B detached; a group-13 surrogate case) and the OS-deduction cases (11.1) are the standing losses; OVODUZ, FIGHOT, COPYIP likewise.
- Seed spread and expansion signature (stretched picks 6.7 % -> 9.2 %) to monitor with every arm.
- Inspect structures early: export problem cases with `export_xyz.py` to `data/inspect/<CODE>/` and look at them before theorising.

---

## Appendix A: the statistics and the maths, in brief
Notation: M metal at the origin; D_s donor point of site s; X_{s,i} substituents; e_j ideal site vectors; pi donor->site assignment; B batch rows.
- Score weights for Q1-Q5 in the legacy mean form act as additive constants (log w in the LSE); only Q6 and S_orient are truly weighted.
- Legacy per-quantity mean: S = sum_Q mean_t LSE_k[log w_tk + log f_tk(x_t)]. Gaussian log f = -0.5 log 2pi - 0.5 log v - (x - mu)^2/(2v), v = max(sigma^2, v_min); Q5: kappa cos(phi - mu)
  - log 2pi - log I0e(kappa) - kappa. Legacy Q6 soft-min: l6 = -(1/tau) log( mean_p exp(-tau h(d_p)) ), h = -(d - mu)^2/(2 sigma^2) for d <= mu, tau = 2.
- u1: S = (sum over every feature of log p, floored at the thermal widths)/Z, with Z one constant per complex.
- Evidence-weighted EM, MAP pooling, BIC: 7.1.
- Training data (`refit`): tmQM X1-X3 and LnQM; contacts cleaned with data-derived thresholds: through-bond angle theta* (minimum of the smoothed angle histogram between
  125 deg and peak - 10 deg, accepted if <= half the smaller peak, else 150 deg; 5 deg bins, sigma 1 bin) and a second-sphere valley per (metal class, element) (0.05 A bins, sigma 3 bins,
  accepted if <= half the smaller peak and >= 30 contacts lie beyond it).
- Q3 (legacy): alpha_s = angle(M - D_s, mean(X_s) - D_s), scored as cos; Q4: mean of all angles among {M, X_i} at the donor, degrees. Haptic Q3: arccos(n . unit(F_mean - M)),
  n the smallest right-singular vector of the centred face coordinates.
- Q6 pairs: every atom pair (metal and H included) more than 3 bonds apart; the bond graph includes H and one M-D edge per donor.
- Q7 legacy bend penalty: 0.5 (max(0, |beta - beta0| - 2 deg)/5 deg)^2.
- Constants: beam W 16, Q1 samples M 8, steps T 200, lr 0.1; fit n_min 30, K_max GMM 4 (Q1, Q4) / 6 (Q2, Q3), VMM 6; ancestor cap 500; QO n_min 50, K_max 4, cov floor 0.05^2, weight floor 0.05,
  H-bond cut 2.3 A, ring cap 7; embed D_min 2.4 A, max bite 100 deg, small ring <= 6, candidates 8, restraint k 100; enumeration 50 000 / 30 s; cull at 0.3, keep 0.35, niche 30, min rows 16, min batch 256;
  macrocycle closure damping 1e-4, step clip 0.5 rad, final GN iterations 4, seam weight 1000.
