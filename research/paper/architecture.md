# Solver architecture

## Purpose and boundary

Jacobik constructs candidate geometries for transition-metal and lanthanide complexes from a metal identity, total charge, spin multiplicity, coordination-geometry label, and ligand specifications. It does not optimize a full molecular Cartesian structure against an electronic-structure energy during a solve. Instead it embeds each ligand as a free molecule, represents the ligand as rigid fragments connected by selected internal coordinates, and moves those fragments to satisfy coordination geometry while maximizing a statistical score.

The scoring database contains distributions fitted to structural observations. Offline electronic-structure calculations are used to calibrate selected model terms and to evaluate generated structures. They do not provide coordinate forces to the runtime solver.

## Inputs and molecular state

A solve receives a metal, total charge, spin multiplicity, a supported coordination-geometry name, and one ligand specification per ligand instance. A ligand specification contains a SMILES string and donor atom indices; multiple atom indices can define one haptic donor group. The number of donor groups must match the number of occupied ideal sites.

The coordination geometry supplies a set of ideal site vectors. Oxidation state is inferred from total charge and ligand charge conventions because the solve input does not require a metal oxidation-state label. The inferred state selects state-conditioned statistical keys where supported; unknown state bypasses those key levels. Donor classification maps chemical environments to a hierarchical donor-subclass vocabulary.

## Pipeline

The solve has these stages:

1. Parse the complex description and infer charge-derived state labels.
2. Build one free-ligand conformer for each ligand instance.
3. Decompose ligand graphs into rigid bodies and kinematic joints.
4. Enumerate feasible donor-to-site assignments, quotienting by allowed symmetries.
5. Construct initial placements from sampled donor-distance targets and rigid alignment.
6. Optimize all candidate rows jointly with differentiable geometry scores and explicit coordination constraints.
7. Close macrocycle seams, re-score, remove symmetry-equivalent duplicates, and return ranked conformers.

The public entry point is `jacobik.engine.ComplexBuilder`; core modules are under `src/jacobik/`.

## Free-ligand embedding

RDKit ETKDG supplies an initial ligand geometry. MMFF refinement is used when parameters are available. The default profile can refine the free ligand with GFN2-xTB using a crude optimization while constraining donor-to-donor distances to the embedded values. Chelate-aware embedding applies flat-bottom donor-distance bounds and can retry with alternate ETKDG candidates when the first geometry violates those bounds.

Embedding adjustments include temporary caps for selected anionic donors, an optional bound-form surrogate for group-13 donors, ylide-like representations for selected carbene donors, and chelate-derived alkene stereochemistry. E/Z enumeration creates separate ligand starts for unresolved configurations when enabled. These are discrete starts; an alkene configuration does not change continuously during the IK solve. A failed semiempirical embedding may fall back to MMFF, so the actual embed method is execution provenance relevant to reproducibility.

## Assignment and coordination symmetry

The initial donor/site correspondence is estimated by matching donor direction vectors to ideal site vectors. A Hungarian assignment gives a low-cost bijection, followed by a chelate repair and a graph-reachability check. Reachability constrains donor pairs from one ligand according to their covalent path length and target site separation.

When the donor count equals the coordination number and geometry sites are fixed, the solver can enumerate reachable bijections. Assignment equivalence is determined by proper rotations of the coordination polyhedron and by exchanges of identical ligand instances. Reflection is excluded so mirror-related assignments remain distinct. The implementation is embed-aware: internal donor swaps are not automatically considered equivalent because a frozen ligand embed may not be able to realize the reflected structure through available joints.

## Kinematic representation

Rotatable bonds divide a ligand graph into rigid bodies. A root body receives translation and rotation; downstream bodies are composed by forward kinematics using absolute torsion variables. Additional coordinates may represent branch bends, small donor-ring openings, haptic-face bends, or macrocycle folds. Macrocycle opening is paired with a closure correction so the generated chain satisfies the ring seam constraint.

Most covalent bond lengths and bond angles remain those of the free-ligand embed. A bend coordinate changes a designated local angle while preserving adjacent bond lengths. This keeps the search space smaller than unconstrained Cartesian optimization and makes the degrees of freedom interpretable.

## Initialization and search

For each ligand, a torsion beam expands partial seeds using modes of fitted torsion distributions. The beam is local to the ligand and does not use the metal or other ligands when ranking partial seeds. Coordination initialization draws metal–donor distances from the relevant fitted distributions and maps donor points to assigned ideal sites using a rigid alignment. Randomized orientation and torsion choices provide multiple rows for each assignment.

Optimization uses batched Adam over rigid-body, torsion, and enabled bend variables. The objective combines the statistical score with augmented-Lagrangian penalties for coordination-shape deviation and symmetry ties. These constraints are separate from the statistical score. A culling stage may reduce large batches while maintaining representatives from distinct candidate and torsion-mode basins.

Each row retains its best scoring state during optimization. Macrocycle closure is corrected before final scoring. The candidate set is re-scored with the ranking function, then filtered by symmetry-aware heavy-atom RMSD and an output score window.

## Reproducibility and result state

`jacobik.profile` centralizes fields that can change generated results or score ordering. A named profile supplies defaults; JSON overrides support controlled experiments. The active module values can be captured as a profile snapshot. Device, threading, and executable paths are execution settings, but floating-point precision and nondeterministic embedding can affect repeatability and must be recorded for comparisons.

The resulting ensemble is an unweighted set of score-ranked conformers. It is not a Boltzmann ensemble. The solver does not calculate basin free energies or assign thermodynamic populations.
