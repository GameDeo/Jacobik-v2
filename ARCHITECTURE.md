# Jacobik: architecture

How the generator works today: the pipeline, the score, the statistics database and the pipeline that fits it, and the way
settings, fingerprints and regression references are managed. It describes what exists; the history is in `CHANGELOG.md`, and the
earlier, much longer design document (with the ledger of discarded variants and the evidence workflow) is archived at
`research/archive/docs/architecture/ARCHITECTURE_2026-10-20.md`.

Conventions: distances in angstrom (A), energies in kJ/mol unless "nats" (the unit of log-likelihood, the score), RT = 2.479
kJ/mol at 298.15 K. Paths are relative to the repository root.

Contents: 1 purpose and guarantees | 2 pipeline | 3 the score | 4 statistics database | 5 fitting pipeline | 6 settings,
fingerprint, golden references | 7 evaluation framework | 8 code layout | 9 limits | Appendix: the statistics in brief

---

## 1. Purpose and guarantees

Jacobik generates conformers of a transition-metal or lanthanide complex from a description only: metal, charge, spin, a
coordination-geometry label and, per ligand, a SMILES string with its donor atoms. Each free ligand is embedded once (without
the metal) and then moved as rigid bodies plus torsions, ring states and a few bond-angle bends until a **statistical-potential
score** is maximal. The score is a sum of log-likelihoods of geometric features against distributions fitted to crystal
structures (tmQM, lnQM). No crystal coordinates of the target enter a solve.

Guarantees the design keeps:
- **No force field, relaxation or machine-learned potential acts on a complex during a solve.** Physics enters as fitted
  parameters measured offline (GFN2 dimer scans for the repulsive contact term, GFN2 Hessians for thermal widths, MACE fits for
  the bend prior). GFN2 and MACE are used to *measure* results, never to steer the solver.
- **The generator never fails the caller.** It degrades (a failed step logs and falls back) rather than raising out of
  `solve()` for chemistry it cannot handle.
- **Deterministic given the seeds** (`seed`, `embed_seed`); the GFN2 embed runs on one thread for this reason.
- **No per-complex rules.** Every rule is general and its effect is measured on panels.
- **Every result-changing setting is a registered profile field** (section 6), so a result can be tied to the exact settings.

## 2. Pipeline

Entry point: `jacobik.ComplexBuilder(metal, geometry, charge, spin, ligands=[LigandInput(smiles, donors)]).solve(...)` returns an
`EnsembleResult` (`src/jacobik/builder/result.py`).

| Step | What happens | Main files |
|---|---|---|
| 1 Typing | Oxidation state from charge and ligand charges; donor subclass hierarchy; geometry classification | `core/chemistry.py`, `topology/donor_classifier.py`, `topology/donor_rules.py`, `topology/coord_geom.py` |
| 2 Configuration enumeration | Donor-to-site assignments up to symmetry; stereocentres (given, unspecified, created by binding) and E/Z bonds; ring-pucker states | `ik/site_enumeration.py`, `ik/stereo.py`, `topology/rings.py`, `builder/embed.py` |
| 3 Embed | Each free ligand once: ETKDG, MMFF, GFN2 refinement under chelate-aware distance bounds | `builder/embed.py`, `builder/engine.py` |
| 4 Beam and IK solve | Torsion seeds from the statistics, then one batched joint solve of every configuration block | `ik/beam.py`, `ik/ik_solver.py`, `ik/solver_replay.py` |
| 5 Scoring | Sum of log-likelihood terms, divided by a per-complex constant | `ik/scorer.py`, `ik/kernels.py`, `stats/*` |
| 6 Dedup | Symmetry-aware heavy-atom RMSD | `ik/dedup.py` |
| 7 Output by class | Rows grouped into configuration classes, ranked inside each | `builder/engine.py`, `builder/result.py`, `builder/assemble.py` |

### 2.1 Typing and inputs
Inputs: metal, total charge, spin, one of 13 geometry labels (`linear`, `trigonal`, `t_shaped`, `square_plan`, `tetrahedral`,
`see_saw`, `square_pyr`, `trig_bipyr`, `octahedral`, `trig_pris`, `pent_bipyr`, `square_anti`, `tricap_pris`) or `other`, and per
ligand a SMILES with donor groups (one atom index for a kappa donor, several for a haptic one). The number of donor groups must
equal the coordination number of the geometry. The oxidation state is the charge minus the ligand charges, counted from the
explicit formal charges (anionic donors must be written explicitly); an unassigned state is "unknown" and the state-keyed
statistics levels are skipped. A donor's subclass (for example `O_water` below `O_neutral_sigma` below `O`) selects the
statistics through the lookup ladder of section 4.

### 2.2 Configurations
A **class** is a configuration that does not interconvert thermally: the donor-to-site arrangement, the stereocentres (including
those created by binding) and the E/Z double bonds. Enumeration:
1. **Assignments.** Every reachable bijection of donor groups to sites (Hungarian baseline, chelate repair, reachability by hop
   count and angle), merged only by proper rotations of the polyhedron and swaps of whole identical ligands. Graph-level symmetry
   is not used: swapping two equivalent donors inside one ligand mirrors their arrangement in the frozen embed, which torsions
   cannot always undo.
2. **Stereo** (`stereo_candidates`, `config_states`, `config_near`). Unspecified and binding-created stereocentres and E/Z bonds
   form a capped product of the elements near the metal (within two bonds of a donor or created by binding); each class is a row
   block of one joint solve with its own preparation, cull and selection. Remote acyclic elements are an optional extension
   (`config_remote`, off).
3. **Ring states** (`ring_states`, `ring_states_five`). Saturated six-rings (chairs, twist-boats) and five-rings (envelopes,
   twists) near a donor are offered as a library of states in the same batch; each state carries its MMFF energy, rows are split
   over states by Boltzmann weight and the score includes the state's population in nats.
4. **Mirror completion** (`out_mirror`). Enantiomers are completed by reflecting output rows when no ligand carries a
   tetrahedral tag; reflected rows keep their source score and are marked.

### 2.3 Embed
Per free ligand: ETKDGv3 (seed 42 unless `embed_seed`), MMFF94, then GFN2-xTB `--opt crude` with every donor-donor distance held at
the embed's value. Chelate-aware bounds (donor-donor distance limits from typical M-D lengths and ring closure) are applied to
multidentate ligands; anionic donors are neutralised and H-capped during embed and uncapped afterwards (Li caps for some anionic
and haptic cases); chelate E/Z bonds get the configuration their ring requires, and unresolved E/Z are enumerated. After the embed
all bond lengths and bond angles are frozen except the optional bends. `xtb` must be on `PATH`: a failed GFN2 call falls back to
MMFF and logs "GFN2 embed refinement failed"; structures built that way must not be used.

### 2.4 Kinematics and solve
- **Degrees of freedom.** The ligand is cut at rotatable bonds (non-ring, bond order about 1, both atoms with more than one
  neighbour) into rigid bodies; each ligand has 3 translations, 3 rotations and its torsions, plus branch bends at kappa donors,
  macrocycle joints for large rings with two or more donors (closed by a Jacobian correction each step), ring joints for small donor
  rings and bond-angle bends (`angle_bends`: in-plane and out-of-plane on chelate backbones and haptic faces, priced by an
  energy-calibrated prior).
- **Beam and start.** Per ligand the top 16 torsion seeds from the torsion statistics; coordination targets drawn from the
  metal-donor distribution; Kabsch placement; batch size = candidates x 8 coordination samples x 16 seeds.
- **Optimiser.** Adam (200 steps, learning rate 0.1) on -score plus an augmented-Lagrangian shape constraint that keeps the
  donors on the chosen polyhedron (tolerance per geometry measured on crystals) and symmetry ties for equivalent donors;
  successive halving of rows with niches when the batch is large. The production finish phase is an iRprop+ schedule with
  thermal step sizes (`finish_phase`). On GPU the loop runs through a CUDA-graph replay generated from the Python solver
  (`ik/solver_replay.py`, built by `dev/tools/jdev/build_replay_solver.py`; never edit the output by hand).

### 2.5 Dedup and output
Every row is rescored by the ranking score, which is the same function as the solve objective. Output deduplication keeps the
best row of each group within 0.5 A symmetry-aware heavy-atom (plus metal) RMSD (`dedup_rmsd`, with the factored symmetry test
`dedup_factored`), and drops rows more than 200 nats below the best (`out_window`). `EnsembleResult` groups the rows by class:
`classes()`, `by_class()`, `summary()`, `write_xyz(by_class=True)`, `to_records()`. `delta_nats` is a score gap, not an energy.

## 3. The score

One function serves both as solve objective and ranking score:

    S = [ Q1 + Q2 + S_orient + Q5 + Q6 + bend prior ] / Z

Every term is a sum of log-likelihoods (nats) with weight 1; Z is one constant per complex (the number of internal features). It
changes no argmax; it keeps the optimiser's step scale. Each pair distance or angle is priced by exactly one term.

| Term | Prices | Form |
|---|---|---|
| **Q1** | metal-donor distance (haptic: to the centroid) | mixture of Gaussians per key; key = metal, spin, oxidation state, donor subclass, geometry, trans-partner subclass; one-sided wall beyond the table's support |
| **Q2** | donor-metal-donor angle, every site pair | Student-t mixture in cosine space; key adds a distance class (different ligand, short or long chelate) |
| **S_orient** (QO) | orientation of a donor's substituents relative to the metal; haptic tilt and bend | full-covariance Gaussian mixture on the Gram features of the unit bond vectors; replaces the older separate approach-angle and pyramidalisation terms |
| **Q5** | ligand torsions | von Mises mixture per rotatable bond key, folded by rotor symmetry |
| **Q6** | non-bonded contacts (five or more bonds apart, or between ligands) | physical shifted-force repulsion `A (e^(-b r) - ...)` out to the van-der-Waals contact distance, parameters fitted to GFN2 dimer scans; a contact core for pairs without parameters; 1-4 weight 0 |
| **Bend prior** | bond-angle bends | harmonic prior from MACE single points on chelating ligands, divided by Z |

**Thermal widths.** A fitted component can be narrower than any single molecule's motion (a cluster of near-identical minima fits
a needle), which would make the score far stiffer than the energy. Every width is floored at its 298 K thermal width
(sigma_T^2 = kT g^T H^+ g from GFN2 Hessians on training crystals; for example Q1 0.0585 A, Q2 4.9 deg, S_orient 4.2 deg,
Q5 13.8 deg). The constants are fixed offline; nothing is computed in a solve.

**What the score is and is not.** It is an ordinal surrogate: a sum of marginal log-likelihoods that counts correlated
information more than once. Terms are within a factor of about 1.0-1.5 of energy-like curvature inside a basin, but the pool-level
relation between score and energy is weak (rank correlation about 0.7 over all rows, about 0.5 among low-energy rows on the
development panels). Re-weighting terms is not a lever: it mutes the tails that veto broken conformers.

## 4. The statistics database

One file holds every entry: `constraints_2026-11-12.npz` (shipped packaged at `src/jacobik/data/constraints_2026-11-12.npz`
alongside `shannon_radii.json` and used automatically by default unless overridden; about 28 MB, 419,644 entries). It is a lossless container with
its metadata embedded (`_meta`: provenance, code commit, profile snapshot, input hashes, the Q5 and Q1 variants the runtime must
use; `_tables`: for example the crystal bend table). `jacobik.stats.stats.load_db` reads the container; a legacy json DB needs a
`<stem>.meta.json` sidecar next to it, or every parent silently runs another Q5 model.

| Prefix | Content | Read when |
|---|---|---|
| `Q1:` ... `Q5:` | mixtures per key | always |
| `QO:` | S_orient mixtures | `orientation_model = "auto"` and present |
| `Q6P:` | repulsive-contact core tables | `q6_model = "repulsive"` (production) |
| `QV:`, `QVR:`, `NCI:` | bond-valence and non-covalent tables | reserved in container; runtime research modules archived under `research/archive/code/` |
| `QG:`, `QP:`, `QB:` | geometry and partition priors, candidate donors | reserved for a planned coordination-hypothesis layer; nothing reads them at runtime |

**Lookup ladder.** A feature is labelled by a key; the lookup walks from the most specific key to coarser ones (for Q1: metal,
spin, oxidation state, donor subclass, geometry, trans partner, then dropping fields, then period group and donor family, then
the element) and takes the first fitted entry with enough evidence. Q1 and Q2 keys with fewer than 50 complexes are the MAP
adaptation (Reynolds 2000) of their parent, so pooled entries are valid at any n.

## 5. The fitting pipeline

`python -m fit.build_db` is the one pipeline from raw tmQM and lnQM files to the container (`src/fit/build_db.py`, stages in
`src/fit/pipeline/`):

| Stage | Does |
|---|---|
| `extract` | raw xyz to observation tables (Q1-Q6, QO) and contact histograms, one task per shard of 400 complexes |
| `census`, `filter` | charge and oxidation-state census; whole-parent exclusions (implausible charge, locked parents) |
| `expand` | ancestor expansion per quantity (each complex counts once; cap 500 rows per ancestor and bucket) |
| `fit` | Q1-Q6, QO, Q6P: weighted EM mixtures (BIC for K, evidence weights 1/m per complex), partial pooling, von Mises fits, thermal floors |
| `prune`, `ref_ll`, `bends` | optional sub-thermal merge (off), reference log-likelihoods, crystal bend table from the frozen fit frame |
| `assemble` | container with entries, `_meta` and `_tables`; json and lossless npz |
| `reproduce`, `rescore` | optional entry-by-entry comparison with a target DB; 60-pool sanity check |

`--profile NAME` and `--flags FILE.json` set the profile for the process and every worker, and the resolved snapshot is stored
in the DB; results do not depend on the worker count. Inputs are `data/ref` (clean three-column tmQM and lnQM shards, reading only
`CSD_code`, `SMILES` and geometry), the locked-exclusion list and the frozen fit frame (`data/refit/frame_v1`); each is hashed
into the metadata. Typing and exclusion rules that change the DB (haptic bond gating and label canonicalisation, donor
reconciliation and assignment, oxidation-state and structure sanity filters, exclusion of locked parents, physical Q6 for all
pairs) live in this pipeline, so a DB can always be rebuilt from its metadata.

## 6. Settings, fingerprint, golden references

`src/jacobik/core/profile.py` is the single place for result-changing settings.
- `PRODUCTION_CONFIG` holds the production values. `FIELDS` registers every experiment flag as (module, attribute, default,
  description); a flag is read by its module at import. `JACOBIK_PROFILE` selects a named profile and `JACOBIK_PROFILE_OVERRIDES`
  (JSON) overrides registered fields. Overriding a hardcoded production setting raises on purpose (it would change the fingerprint
  but not the solver). Retired `JACOBIK_*` switches raise too.
- The **fingerprint** is `sha1(PRODUCTION_CONFIG)[:12]`; `jacobik.profile.snapshot()` reports it with the active overrides. The
  production fingerprint on `master` is `2c87e2940af2` (donor-assignment flag consolidation over the `a47283387172` P2R baseline validated in `PERFORMANCE.md`).
  Flag history and the evidence behind each fold-in: `CHANGELOG.md` and the header of `src/jacobik/core/profile.py`.
- **Golden references** (`research/data/golden/`) are arrays recorded for the production and `u1` profiles on a fixed set of
  complexes. `python -m tools.check ws` (fingerprint, parallel golden check, sharded pytest) must show zero differing arrays for
  any change that claims to leave behaviour unchanged. A golden is re-recorded only at an owner-approved fold-in, and the old one
  is kept in `research/archive/data/2026-11_golden_history/`.

## 7. Evaluation framework

- **Panels and locked sets.** Development uses the `tail` and `cal` panels. The validation panel, the reserve panel and all locked
  sets are never used for tuning (manifest: `research/bench/benchmarks/LOCKED_SETS.md`).
- **Metrics.** Quality is measured after the solve with MACE-POLAR-1 single points (float64) against the reference geometry:
  top-1 energy gap, miss rate of reference basins (a class-aware metric that counts enantiomers as distinct and judges the top-1 by
  energy over any class), pool rank correlation, failures and wall time. Comparisons are paired per parent with
  parent-clustered bootstrap intervals.
- **Runner.** `python -m tools.run` is the one parent runner (lanes, MPS packing, control cache, telemetry); `python -m tools.check`
  has three levels (`ws`, `pm-quick`, `pm-full`). Benchmark tooling lives in `research/bench/tools/jgen`; contracts and inputs of
  finished campaigns in `research/campaigns/`, archived reports in `research/archive/docs/`.
- **Perlmutter** is the default compute site for anything beyond a quick check; see `AGENTS.md` for the rules.

## 8. Code layout

| Path | Holds |
|---|---|
| `src/jacobik/__init__.py` | Public API: `main`, `ComplexBuilder`, `ConformerResult`, `EnsembleResult`, `LigandInput`, `generate`, `geometry_sites`, `check_environment`, `default_db_path`, lazy `profile` |
| `src/jacobik/builder/` | `ComplexBuilder` orchestration (`engine.py`), `LigandInput`, `EnsembleResult`, conformer containers (`result.py`), `assemble.py`, `configs.py`, `ligands.py`, `embed.py`, `atom_map.py`, `assignment.py`, `beam.py`, `shell.py`, `cli.py` |
| `src/jacobik/core/` | Profile and fingerprint (`profile.py`), chemistry/oxidation/radii (`chemistry.py`), energy constants (`energy.py`), environment/xtb checks (`env.py`), math/geometry primitives (`geometry.py`), lossless NPZ/JSON DB container (`container.py`), shared types (`types.py`) |
| `src/jacobik/topology/` | Donor classification (`donor_classifier.py`, `donor_rules.py`), graph/haptic/mapping (`graph.py`, `haptic.py`, `mapping.py`), coordination geometry (`coord_geom.py`), rigid-body decomposition (`decompose.py`), ring-pucker states (`rings.py`) |
| `src/jacobik/ik/` | Forward/inverse kinematics (`fk.py`, `fk_compose.py`, `reachability.py`), solver & generated replay (`ik_solver.py`, `solver_replay.py`, `finish_phase.py`, `polish.py`), `beam.py`, `site_enumeration.py`, `stereo.py`, `angle_bends.py`, `scorer.py`, `kernels.py`, `geometry.py`, `kabsch.py`, `targets.py`, `dedup.py` |
| `src/jacobik/stats/` | Database loading and lookup (`stats.py`), keys/ladders (`keys.py`, `ladders.py`), mixtures (`mixture.py`), S_orient (`orientation.py`), physical Q6 steric repulsion (`steric.py`), hypothesis priors (`hypothesis_tables.py`), reference log-likelihood (`reference_ll.py`) |
| `src/fit/` | Fitting pipeline (`build_db.py`, `pipeline/{extract,fitting,confidence,compare}.py`, extractors/expanders/fitters `q1.py`–`q6.py`, `q6p.py`, `qo.py`, `bends.py`, `pooling.py`, `sanity.py`, `parser.py`, `data/benchmark_holdout_codes.json`) |
| `interface/` | CLI (`jacobik_cli.py`, `prepare_data.py`), ligand library & debug complexes (`ligand_library.py`), notebooks |
| `dev/` | Developer tree: unit/regression tests (`dev/tests/`), check levels (`dev/tools/check/`), parent runner (`dev/tools/run/`), dev/replay helpers (`dev/tools/jdev/`), SLURM scripts (`dev/tools/d2/`) |
| `research/` | Benchmarks (`bench/`), campaign contracts (`campaigns/`), paper drafts (`paper/`), Colab helper (`gpuwork/`), archive (`archive/{docs,code,tests,...}`) |

## 9. Limits and open items

- Rows are unrelaxed; relaxing never replaces fixing the generator.
- Very large parents can be slow; a few parents time out on the development panels.
- Remote stereo elements (`config_remote`) are not exercised by the development panels and stay off.
- The coordination-hypothesis layer (the `QG`, `QP` tables) and non-covalent terms (`NCI`) are parked.
- A learned correction of the environment (neural network) is planned after the generator is finished.

---

## Appendix: the statistics in brief

Notation: M the metal at the origin; D_s the donor point of site s; X substituents; pi the donor-to-site assignment.
- **Evidence weights.** The m rows of one complex in a key get weight 1/m; n_eff is the sum. Keys with fewer than 30 rows are
  not fitted alone.
- **Mixtures.** Weighted EM for K = 1..Kmax with K chosen by BIC (p = 3K - 1, plus 1 for the Student-t degrees of freedom);
  components with n_eff * weight below 10 are not selectable. Q1: Gaussians in A; Q2: Student-t (nu in {2, 4, 10}) in cosine
  space; Q5: von Mises with kappa from Best-Fisher; S_orient: full-covariance Gaussians (K up to 4, covariance floor 0.05^2).
- **Partial pooling.** alpha_k = n_k / (n_k + r) shrinks mean, variance and weights toward the parent; r and the Q1 ladder
  variant are chosen by held-out log-likelihood per complex.
- **Repulsive Q6.** Shifted-force Buckingham repulsion
  `U(r) = A [e^(-b r) - e^(-b r_m) + b e^(-b r_m)(r - r_m)] / RT` for r < r_m (zero beyond), with r_m the sum of Alvarez
  van-der-Waals radii and A, b per element combined as A_ij = sqrt(A_i A_j), b_ij = (b_i + b_j)/2, fitted to GFN2 dimer energies;
  linear continuation above 50 kT keeps a gradient in deep overlaps.
- **Thermal width.** sigma_T^2 = kT g^T H^+ g (g: gradient of the feature, H^+: pseudo-inverse of the GFN2 Hessian).
- **Solver constants.** Beam width 16, 8 coordination samples, 200 steps, learning rate 0.1; shape constraint with
  tau = 2, rho = 8, mu0 = 10; row culling at 0.3 of the steps keeping 0.35 of the rows (at least 16), niche weight 30;
  macrocycle closure with 4 final Gauss-Newton iterations.
