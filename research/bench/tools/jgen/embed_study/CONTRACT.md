# F embed study: evaluation contract

**Written before inspecting F comparison outputs, 2026-10-28.**

## Cohort

- Every ligand instance in the `cal` and `tail` parents present in the current jgen `scaled_v1` manifest.
- A reproducible sample of 300 `surface_fit` ligand instances, selected by a fixed seed from the allowed split only.
- Named required cases: NEPNEF, QEZVEY, HEDJOP, JEGPUJ, WEYJIT. Exclude every `motion_validation` (`val`) and reserve/locked code from every cohort; in the tail list this excludes IFUROQ, SOBHAS, XEYTUT, YUCZUS, and ZELTAP, leaving the calibration parents only.
- Pair each extracted ligand with its matching bound-reference ligand using graph/atom mapping. A pair with unresolved mapping is retained in the failure ledger and excluded only from metrics requiring that mapping.

## Primary measurements

Measure the embed output from production's current settings (ETKDGv3 seed 42, production capping and chelate/EZ handling, MMFF and GFN2 refinement, no shell translation) against the same ligand's bound-reference coordinates. Report distributions per ligand class and per parent, with ligand instances as observations and parent-clustered summaries where comparing methods.

Frozen-coordinate classes are: bonded distances by element/bond order; angles not controlled by solver bend DOFs; hydrogen-containing angles separately; ring bond/angle/torsion internals and pucker class; out-of-plane impropers and planarity; assigned E/Z and tetrahedral stereo; donor–donor distances for chelates. Heavy-atom and all-atom results are separate. Do not count solver-controlled torsions/bends as frozen internals.

## Reference-free detectors

- Detector (b): for each bond and non-DOF angle, use empirical 0.5th–99.5th percentile ranges from bound `surface_fit` geometries, stratified by element/bond order or angle atom types and hydrogen participation. Report class coverage and false-positive rate against reference deviations.
- Detector (c): flag bonded distances above 1.25 times the covalent-radius sum; nonbonded intraligand distances below 0.8 times that sum; and missing/changed stereo or E/Z. Report each component separately and the union.
- Define reference-bad thresholds before scoring: bond error >0.15 Å; non-DOF angle error >15°; ring-pucker class mismatch; planarity/improper error >20°; stereo/EZ mismatch; or chelate donor-pair error >0.40 Å. Also report continuous errors so conclusions do not depend on these cutoffs.
- Against-reference performance (a) is the target label. Report sensitivity, specificity, precision, recall and AUROC for (b), (c), and their union; bootstrap CIs by parent.

## Strain reconstruction

For named cases plus the 20 worst cal parents by the existing A4 strain ranking, transplant only embed-frozen internal coordinates into each reference complex while holding metal placement, solver-controlled torsions/bends, and reference pose fixed. Record reconstruction/mapping failures individually. Evaluate valid reconstructed structures with MACE-POLAR-1-S in float64 CPU; report paired energy rises relative to the bound reference. If a profiled and parallelized estimate exceeds 12 hours, mark this item not done and report the estimate.

## Alternatives and controls

Compare current production with the specified cap alternatives, bound-state dummy-metal embed, donor-restrained GFN2, and GFN-FF if the installed xtb supports it. Use the same ligand instances, deterministic seeds, and matching controls. Report all frozen-coordinate metrics, time per ligand including setup, method failure rate, and reconstructed MACE energy rise where available. A method wins only if it lowers median non-DOF error on the failure set, does not worsen the good-embed control beyond its paired bootstrap noise, and costs no more than 2x current embed time. Otherwise conclude **none wins**.

## Reproducibility and exclusions

Record input manifest and exclusions hashes, sample seed, software versions, embed seeds, and complete failures. Do not load `val`, reserve, or locked cohorts. Do not read tmQM property columns. No production code or profile settings are changed in this study.
