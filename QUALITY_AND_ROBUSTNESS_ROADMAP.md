# Quality and Robustness Roadmap

This is the working checklist for taking Jacobik from a capable prototype to a
chemically credible, reproducible research-quality tool. A change is accepted
only when it improves predefined chemical-realism metrics on a representative
benchmark suite, is reproducible, and has documented failure modes.

## Status conventions

- [ ] Not started
- [~] In progress
- [x] Complete
- [!] Blocked / needs a decision

For each completed item, add links to the relevant issue, commit, benchmark
data, and report directly beneath it.

---

## Phase 1 — Understand the current system and create a baseline

### 1.1 Architecture inventory

- [ ] Map the molecular data model: atoms, bonds, donor atoms, metals, charges,
  protonation, ligands, and complexes.
- [ ] Map coordinate construction, constraints, inverse kinematics (IK),
  optimization, scoring, geometry templates, fitting, and output validation.
- [ ] Document all user-facing parameters, defaults, units, random sources, and
  implicit chemistry assumptions.
- [ ] Trace the end-to-end execution path for at least one simple complex and
  one chelated complex.

**Done when:** `architecture.md` (or a linked companion document) describes the
data flow and identifies the modules/functions responsible for each stage.

### 1.2 Reproducible benchmark harness

- [ ] Define a machine-readable test-case format.
- [ ] Record exact input, package/version state, random seed, solver settings,
  intermediate solver trace, final coordinates, score components, convergence
  state, and runtime for every benchmark run.
- [ ] Make repeated runs with the same seed deterministic, or explicitly report
  unavoidable nondeterminism.
- [ ] Add a concise result summary that makes failures easy to compare.

**Done when:** a benchmark can be invoked in one command and produces an
inspectable, versioned result artifact.

---

## Phase 2 — Chemical-realism checks used by every benchmark

- [ ] Define metal–donor distance checks by metal, donor, oxidation state, and
  coordination context where data support them.
- [ ] Check coordination number, donor identity, formal charge, valence,
  protonation, and molecular topology consistency.
- [ ] Assess donor-angle distributions and polyhedral shape.
- [ ] Detect nonbonded clashes and excessive ligand strain.
- [ ] Add robustness checks: sensitivity to initial conformation, seed, and
  modest parameter perturbations.
- [ ] Define how reference structures and literature expectations are selected,
  normalized, and compared.

**Done when:** benchmark success is reported independently of the optimization
score, with clear pass/fail or warning criteria.

---

## Phase 3 — EDTA and chelation diagnosis

### 3.1 Controlled EDTA study

- [ ] Select chemically documented EDTA complexes spanning relevant metals,
  oxidation states, and plausible coordination modes.
- [ ] Define the input charge/protonation and intended donor set for each case.
- [ ] Sweep random seeds, initial conformations, target coordination numbers,
  target geometries, and material solver settings.
- [ ] Record convergence rate, final-score distribution, score-term breakdown,
  runtime, and chemical-realism checks.
- [ ] Inspect representative best, typical, and failed outputs geometrically.

### 3.2 Root-cause confirmation

- [ ] Test whether each observed failure is caused by representation, missing or
  contradictory constraints, score weighting, local minima, numerical
  conditioning, or an unsuitable geometry target.
- [ ] Build a minimal reproducible case for every confirmed failure mode.
- [ ] Compare successful and failed outcomes against known EDTA coordination
  chemistry, using primary literature/structure sources where possible.
- [ ] Convert every confirmed defect into a regression test.

**Done when:** there is an evidence-backed root-cause report, not merely a list
of suspicious symptoms, and regression cases reproduce the prior behavior.

---

## Phase 4 — Fitting pipeline audit and repair

- [ ] Trace fitting from data loading through feature construction,
  normalization, objective calculation, optimization, serialization, and use of
  fitted parameters.
- [ ] Verify units and scaling at each boundary.
- [ ] Test parameter bounds, initialization, optimizer convergence, numerical
  stability, and deterministic behavior.
- [ ] Test for train/test leakage and incorrect reuse of fitted state.
- [ ] Create synthetic datasets with known ground-truth parameters and verify
  recovery where the model is identifiable.
- [ ] Identify and reproduce each fitting bug before changing code.
- [ ] Implement fixes and add focused regression tests.

**Done when:** fitting results are validated on synthetic and held-out data, and
each fixed defect has a test that fails without its fix.

---

## Phase 5 — Shape control, Q2, and ideal-geometry guidance

### 5.1 Define target behavior

- [ ] Specify chemically meaningful shape metrics for each supported
  coordination family.
- [ ] Define when an ideal geometry should dominate and when chelation,
  sterics, or empirical/statistical terms should permit distortion.
- [ ] Select real complexes that include both near-ideal and strongly distorted
  coordination spheres.

### 5.2 Benchmark candidate objectives

- [ ] Establish baseline performance of the present Q2/shape guidance.
- [ ] Re-evaluate existing vector-decomposition and nullspace approaches on the
  same real-complex benchmark.
- [ ] Implement or prototype conventional continuous shape measures (CShM).
- [ ] Prototype a weighted CShM objective combined with current statistical and
  structural terms.
- [ ] Test fixed, staged, and adaptive weighting schedules.
- [ ] For each candidate, measure geometry accuracy, chemical validity,
  convergence rate, local-minimum susceptibility, compute cost, and stability
  across seeds and initial conformations.
- [ ] Check optimization properties required by the solver, including
  differentiability or a justified compatible alternative.

### 5.3 Decision gate

- [ ] Retain a method only if it improves held-out real complexes without
  materially degrading diverse baseline cases.
- [ ] Document known failure regions and tuning guidance.
- [ ] Add the selected objective and its benchmark cases to regression testing.

**Done when:** the recommended objective is selected by a published benchmark
table/analysis with explicit metrics—not by visual plausibility alone.

---

## Phase 6 — Coordination number and lability flexibility

- [ ] Inventory currently unsupported or unreliable coordination numbers,
  polyhedra, ligand denticities, bridging modes, and metal-specific cases.
- [ ] Separate facts supplied by the user from choices the solver may explore:
  donor set, permitted coordination numbers, geometry families, required bonds,
  and exchangeable/labile bonds.
- [ ] Design a multi-hypothesis search mode that can return and rank several
  chemically plausible coordination/lability models.
- [ ] Define ranking, deduplication, validity checks, and clear failure messages
  when constraints are incompatible.
- [ ] Add regression cases for high-coordinate and ambiguous/labile complexes.

**Done when:** users can deliberately constrain or explore coordination and
lability without hidden solver assumptions deciding the chemistry.

---

## Phase 7 — Solvation and extended structures (research/design track)

This phase is intentionally downstream of local-complex reliability work.

- [ ] Review low-cost solvation options appropriate to the intended fidelity:
  implicit dielectric/GB-like models, coordination-aware solvent placement, and
  limited explicit first-shell models.
- [ ] Establish test systems where solvent changes coordination behavior or
  conformation in a known way.
- [ ] Design a hierarchical MOF/macrostructure representation:
  1. high-resolution IK generates local coordination-sphere conformers;
  2. validated conformers are cached as rigid or semi-rigid fragments;
  3. coarse IK assembles the larger system using inter-fragment constraints;
  4. selective local refinement resolves introduced strain.
- [ ] Identify failure boundaries: cooperative flexibility, bridging ligands,
  long-range electrostatics, defects, solvent-mediated structure, and topology
  changes.
- [ ] Build a small proof of concept only after the approximation and validation
  metrics are defined.

**Done when:** a feasibility memo and small benchmark demonstrate where the
hierarchical approximation is reliable and where it is not.

---

## Acceptance standard for every solver/scoring change

- [ ] The hypothesis and expected mechanism are written down before the change.
- [ ] The claim is tested against a baseline and a representative benchmark set.
- [ ] Results include seed/init sensitivity and score-term diagnostics.
- [ ] Independent chemical-realism checks corroborate any score improvement.
- [ ] Regressions and trade-offs are reported, not omitted.
- [ ] Passing cases and known limitations are added to project documentation.
