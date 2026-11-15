# Comparison protocol, version 1

This protocol describes the original baseline. The primary specialized panel
uses [CHALLENGE_PROTOCOL.md](CHALLENGE_PROTOCOL.md), including haptic groups,
ensemble evidence, solvent choices and explicit adapter limitations.

The later [evaluation design](EVALUATION_DESIGN.md), based on reading
JacobIK's current API, supersedes this document's CN-only primary-comparison
proposal: JacobIK normally receives a geometry label, so the two methods
need matched geometry information or a separately budgeted open search.
The existing runner has not yet been adapted to that proposed policy.

## Scope and selection

Freeze `data/cases.json`, `data/ligands.json`, and the reference manifests before
running either method. The current version was selected before either builder
was run. It covers mononuclear complexes with ordinary atom-to-metal donor
bonds. Polynuclear complexes, bridging-metal networks, haptic/arene ligands,
actinyls, radical ligands, and spin-state prediction require separate panels.

The 40-case panel is stratified by construction difficulty. It is deliberately
rich in chelates and includes matched contrasts: Ni/Pd tetrachlorides, rigid
versus flexible Zn chelates, Fe/Cr tris(oxalate), Fe/Ru tris(bipyridine), and
macrocycles on different metal sizes. It does not support claims about overall
success rates across the CSD or all inorganic chemistry. Report cases and
strata individually as well as any aggregate.

The controls are implementation checks, not the principal performance score.
Designed cases are chemically motivated targets with explicit assumed spin
and protonation states; their existence, stability, or nominated geometry is
not claimed as an experimental fact. Several highly charged, undercoordinated,
or size-mismatched isolated ions are deliberate boundary cases. Keep them in
the failure analysis rather than assuming that a gas-phase minimum must retain
the nominated coordination sphere.

Inputs are independently authored from chemistry and the cited papers. No
tmQM inputs were used. Strict disjointness by molecular identity, CSD refcode,
or graph has not been audited. A later exclusion audit should record any
matches and publish the resulting panel version rather than silently deleting
cases after seeing method performance.

## Equal inputs and budgets

The primary task supplies the same metal, ligand graphs, exact donor atoms,
CN, oxidation state, charge, and unpaired-electron count to both methods.
Expected geometry and reference coordinates are hidden. A geometry-specified
task can be added later as a separate comparison; never pool its results with
the CN-only task. All ligands fully fill the requested CN, so additional
solvent ligands should not be introduced implicitly.

Default Architector settings are ten requested conformers per core geometry,
at least ten sampled symmetries, seeds 17/29/43, a 600-second process deadline,
GFN2-xTB assembly, GFN2-xTB final relaxation, no solvent, no CREST, and one
thread for the numerical backends. These are initial budgets to freeze after
Linux smoke verification, not tuned per case. Record requested and actual
structure counts because the conformer setting is per core geometry and does
not guarantee ten total outputs.

The seed controls NumPy-level sampling. Open Babel retains additional
randomness according to upstream documentation; identical integer seeds do
not guarantee bitwise-identical coordinates or synchronized sampling across
methods. Keep repeated runs, exact package locks, hardware, and input hashes.

Measure the entire builder call and the process wall time. The builder-call
time includes ligand preparation, geometry assignment, assembly, ranking, and
relaxation; it excludes Python import time, which is recorded separately when
the worker reaches that point. For a construction comparison, report the
`--no-final-relax` track separately and state that it still uses xTB during
assembly. A later common relaxation stage can compare end-to-end quality
under an identical optimizer, charge, spin, solvent, and convergence budget.

If JacobIK uses GPU acceleration, disclose GPU and CPU resources and report
the hardware configuration. Equal wall-time and equal energy-evaluation-count
comparisons answer different questions. Do not present a single CPU/GPU timing
ratio without those details.

## Measurements to implement before claiming a comparison

For all 40 cases, independently validate element counts, ligand connectivity,
donor-to-metal connectivity, CN, catastrophic clashes, and geometry retention.
Track success probability over seeds, number of distinct valid candidates,
time to first valid structure, and throughput. Separate API exceptions,
timeouts, empty outputs, chemical-identity changes, and relaxation-induced
dissociation. A nonempty Architector dictionary is an API outcome, not an
independent validity measurement. The current runner records outcomes and
structures; it does not yet implement this scoring layer.

For the eight experimental anchors, compare heavy-atom RMSD and metal/donor
geometry RMSD to the unrelaxed crystal molecule using proper rotations and
graph-constrained atom/identical-ligand permutations. Do not compare atom
arrays by index alone. Report lowest-ranked, best-of-k, and all-candidate
coverage separately. Do not count multiple solvates, crystallographic copies,
or the enantiomeric A05 crystals as independent cases.

Since input chirality is unconstrained, also report a clearly labeled
chirality-invariant RMSD obtained by considering the mirrored reference.
Keep the proper-rotation RMSD to the nominated experimental enantiomer as a
separate value. Spin states and expected geometry are benchmark assumptions
unless a source explicitly establishes them; hydrogen positions and
proton-transfer accuracy need a separate check, particularly for A06.

Crystal packing, omitted counterions, gas-phase relaxation, and alternative
low-energy isomers can shift the predicted geometry. Report geometry quality
and common-method energy as separate quantities. Energy rankings from
different scoring functions cannot be directly compared. The 32 cases without
experimental references must not receive invented RMSDs or enter a crystal
RMSD denominator.

Use paired per-case comparisons, failures included. Report each CN/denticity/
ligand-family stratum and uncertainty over repeats. A 40-case diagnostic
panel, with some shared ligand families, is too small for strong population
claims. Any later tuning set and final holdout should be separated by ligand
scaffold, not only by a metal substitution.

## Current completion boundary

The catalog, provenance archive, eight prepared references, offline chemistry
validation, isolated environment specifications, and Architector runner are
provided. Linux execution of Architector, the JacobIK adapter, independent
output scoring, and actual comparative results remain future work. No
JacobIK integration or edits are part of this setup.
