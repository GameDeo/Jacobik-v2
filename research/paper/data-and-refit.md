# Training data and refitting

## Data roles

The model separates structural observations from electronic-structure calibration. Crystal-derived molecular geometries provide the feature distributions used for Q1–Q6 and QO. tmQM structures and the lanthanide extension are parsed into a common complex representation. Benchmark holdouts are excluded from fitting and histogram construction. Separate GFN2-xTB calculations provide dimer repulsion scans and Cartesian Hessians; MACE single points provide bend-distortion calibration and evaluation data. These calculations produce offline parameters or measurements and do not enter the solve-time coordinate loop.

## Structural cleaning and parsing

The refit pipeline audits charge labels and oxidation-state conventions, cleans contact observations, exports the lanthanide source data when needed, and shards structures for parallel extraction. Contact cleaning removes covalently constrained or otherwise unsuitable observations using graph and geometric criteria. The parser records atom identity, bonds, metal index, coordination sites, geometry, charge, spin, and source identity.

Training splits are assigned by stable structure identity. Lanthanide structures derived from the same ligand set are grouped as one independent source where applicable. Benchmark exclusions are applied before histogram accumulation because uncapped histogram artifacts cannot be filtered after aggregation.

## Feature extraction

For each complex, the extractor emits keyed observations for:

- Q1 metal–donor distances;
- Q2 donor–metal–donor angles;
- legacy Q3 approach angles and Q4 donor pyramidalization, where needed;
- Q5 torsions around eligible rotatable bonds;
- Q6 nonbonded distances;
- QO joint donor-orientation vectors;
- optional geometry and donor-candidate records used by research tools.

Features are assigned keys using the metal, spin, oxidation-state label, donor subclass, coordination geometry, ligand graph context, and other feature-specific context. Contact extraction also accumulates uncapped histograms for repulsive Q6 onset models, metal 1–3 contacts, and legacy NCI candidates. Histogram extraction shares the cleaned corpus pass to keep candidate selection and exclusions consistent.

## Evidence weighting and mixture fitting

Within a key, rows from one complex receive weights so the complex's total contribution is bounded independently of its number of observations. The sum of weights is the effective sample size used for model selection. Lanthanide siblings sharing one source ligand set are treated as one independent structure where the fitting level would otherwise overcount them.

Gaussian and Student-t mixtures are fit by weighted expectation-maximization. The fitter selects component count by BIC subject to minimum effective support for every component. Q5 is fit with von Mises mixtures. QO uses full-covariance Gaussian mixtures. A sparse key can inherit a parent model by MAP adaptation; its component means, widths, and weights move toward local observations in proportion to local evidence. Parent selection and partial-pooling strength are evaluated with held-out likelihood.

The fallback ladder is part of the model definition. A table is not looked up by unconstrained nearest-neighbor matching: the key builder emits a finite ordered list, and the runtime uses a supported entry according to the declared ladder and support rule.

## Q6 and NCI histogram artifacts

Q6 contact histograms preserve element-pair identity, topological class, and polar-H angular class. Repulsive contact fitting estimates an onset from the observed distance distribution and joins it to a short-range exponential core. The class-5 nonpolar physical repulsion is calibrated separately against GFN2-xTB dimers and tabulated by element. Metal 1–3 histograms are represented independently because those contacts have distinct topology.

The legacy NCI histogram pass records candidate noncovalent geometries and background placements for density-ratio fitting. It is distinct from NCI v2, which uses explicit opportunity enumeration, matched background sampling, smooth surfaces, and offline energy/force calibration. Experimental artifacts use separate versioned schemas and provenance records.

## Database artifacts

The fitted JSON database stores entries by typed prefixes. Q1–Q5 entries hold fitted distributions; QO holds joint orientation distributions; Q6 and Q6P hold legacy wall or repulsive contact parameters; QV and QVR hold bond-valence distributions and reference distances; NCI holds legacy NCI tables. The adjacent metadata file records model and fitting conventions required by runtime lookup, including key variants and symmetry-folding behavior.

QG/QP and donor-candidate records may be present in extracted or fitted data without being read by the production scorer. Artifact presence does not imply runtime use. A production database is versioned and kept immutable once adopted; new refits write a separate target directory before comparison or installation.

## Pipeline stages and provenance

`refit/run_refit.py` organizes the principal operations as audit, clean, extract, select, fit, check, and optional install. Thermal-width measurement and lanthanide export are separate utilities. The stages persist intermediate observations and reports so a fitted table can be traced to source structures, exclusions, fitting choices, and software schema.

Experimental artifacts should record input fingerprints, code or schema version, split membership, candidate-generation rules, fit settings, and any offline calibration source. The NCI v2 registry provides structured records for experiment, generation, ranking, evaluation, artifact, and metric provenance. Numerical outcomes belong in evidence reports; they are outside this architecture bank.
