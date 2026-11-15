# Failure-mode challenge protocol, version 2

The 29-case challenge panel is the primary workload. The original 40 cases
remain a baseline/sanity panel. Selection preceded any builder runs; neither
builder's success was used to select or remove a case. JacobIK is not imported
or modified by this workspace.

The proposed future scoring layer is detailed in
[EVALUATION_DESIGN.md](EVALUATION_DESIGN.md). It adds matched geometry
information, native versus common-refinement tracks, independent energy
audits and explicit failure/coverage denominators. These are evaluation
requirements, not capabilities already implemented by the runner.

## Binding topology is part of the input

Each ligand has exact SMILES, formal charge, indexed coordinating atoms, and
an explicit partition into kappa contacts and eta groups. This distinguishes
an eta5 ring from five sigma donors and preserves multiple pi groups inside
one covalent ligand. For haptic cases, ordinary atomic CN is left null.
`metal_contact_count` counts the individual coordinating atoms;
`binding_site_count` counts one site per kappa contact or pi group. Neither
number is silently substituted for a geometry-template parameter.

The present Architector adapter uses its native `sandwich` path for a single
complete aromatic pi ring per ligand. Upstream reserves three facial template
slots for that group. Thus H01 has ten ring contacts but six Architector
template slots; H02 has nine contacts and six slots. These are adapter
conventions, not chemical coordination-number assignments.

H04's tethered pair of indenyl faces, H05's open-chain eta3 allyl, and H06's
two alkene pi groups require a richer validated adapter. Keep their complete
chemical targets and capability records. Splitting a tethered ligand or
pretending pi contacts are sigma donors would change the problem. A current
adapter limitation is not evidence that Architector itself cannot support
the chemistry.

S03 contains a hydride. The inspected input parser forces `relax=False` for
`[H-]`. Its full-relaxation export is therefore blocked explicitly; use
`--no-final-relax` and a common external optimization stage for this case.
All runtime support labels are source-inspected and offline-validated, not
Linux execution results.

## Reference types and ensemble evidence

D03 has a 55-heavy-atom experimental crystal reference extracted from the
publisher's Zr-HOPO CIF. All eight complete donor-atom assignments, the full
ligand graph, and the published Zr-O distances were checked. In this crowded
structure, four amide oxygens also lie inside a generous radius screen; the
published O1-O8 metal-bond labels identify the actual primary donors. They
must not be inferred from that broad distance screen alone.

B03 and B05 have twelve graph-mapped computed coordinate sets from the
published Xantphos/XL-Xantphos supplement. Each case has trans-Cl, cis-Cl e,
and cis-Cl a geometries at two optimization levels. These are three labeled
families per case, not six independent thermally populated states. The source
contains 26 frames in total, including other ligands; all are archived, with
an explicit frame index. Experimental and computed RMSD references must be
reported separately. Stationarity/frequency checks have not been repeated.

S03 supplies measured exchange among three sulfur diastereomers. The paper's
65-70 kJ/mol activation Gibbs energies are barriers, not free-energy gaps.
D01/D02 target the experimentally studied SAP/TSAP macrocycle problem. The
3.2:1 DTMA ratio is a background value cited by the source; experimental
conditions must be recovered before using it as a numerical target. No
population is assigned to the authored all-S sec-butyl input without matching
its exact experimental stereochemical definition.

For the custom targets, alternative conformers/isomers are hypotheses to
test. Do not call them stable or thermally accessible solely because a builder
returns multiple structures. Establish distinct optimized minima under a
common method; verify local stability and obtain consistent free energies
before assigning Boltzmann populations. Keep solvent, protonation, charge,
spin, temperature and standard state consistent. Several geometries from
different electronic-structure methods are not an ensemble population.

## What to score

| Challenge | Observable that diagnoses failure |
| --- | --- |
| Haptic rings / fused rings | Intended face and hapticity; metal-centroid distance, face normal, contact spread, ring slippage |
| Multiple pi groups / ansa tether | Both groups retained in the same covalent ligand; complete bridge/ring closure; correct capability accounting |
| N2S / N2S2 / SNS | Exact donor identities, bound sulfur pyramidal configuration, chelate folds, and missed sulfur diastereomers |
| Alternative binding modes | S04 retains one N2 and one N2S instance of the same ligand, rather than promoting/dropping both sulfurs |
| Small / wide / trans bite | Measured donor-metal-donor angles, ligand distortion and cis/trans family coverage; avoid forcing every chelate into ideal 90 degrees |
| Eight/nine donors / ten contacts | All specified arms close simultaneously; primary donors retained; no extra fill ligands or secondary-sphere substitutions |
| Secondary sphere / polar groups | Urea-hydroxo hydrogen-bond geometry, backbone folds, pendant-group collapse, unintended sulfonate/amide binding |
| Fixed stereocenters | Preserve all four S carbon centers in D02 while allowing metal, sulfur, arm and ring configurations to vary |
| End-on N2 | M-N-N orientation, N-N connectivity and distance, terminal-versus-distal binding, and cis/trans N2 arrangements |
| Ensemble search | Distinct topology-preserving conformers/isomer families at fixed budget; convergence across repeats and candidate-count curves |

The scoring implementation remains future work. The runner supplies exact
targets, input files, structures, timing, hashes and explicit capability
records. A returned structure alone is not independent geometry validation.

## Runs and denominators

Use identical chemical inputs for both builders, including binding groups and
specified ligand stereochemistry. Do not supply the reference geometry or
preselect its isomer. Use a common method for any final relaxation/energy
comparison. The initial screening settings remain 10 requested conformers
per core geometry, seeds 17/29/43, one numerical-backend thread and a
600-second deadline. For coverage curves, additionally freeze larger budgets,
for example 25 and 50 requested conformers, before comparing the methods.

`--no-final-relax` still permits xTB during assembly. Report that track
separately from full relaxation. Several polar targets specify `water` as
the xTB solvent; the original baseline remains gas phase. The nominal 298.15 K
metadata does not implement thermodynamic sampling or free-energy weighting.
Scalar unpaired-electron inputs are assumptions for the authored targets;
spin-state prediction and spin-orbit multiplets are outside this comparison.

The current full-relaxation adapter exports 25 of the 29 targets. The
assembly-only mode exports 26. Always publish the total target count alongside
the runnable subset, including the three adapter gaps and the hydride policy.
Do not count those records as geometric failures or quietly remove them.

For quality/coverage, separate experimental coordinates, published computed
coordinates, measured solution exchange, and unvalidated design hypotheses.
Use graph-constrained ligand/atom permutations; preserve specified ligand
chirality. Report proper-rotation and chirality-invariant scores separately
when metal chirality was not supplied. Do not let mirror invariance erase a
fixed carbon stereocenter or an intended sulfur-diastereomer distinction.

No tmQM files were used to author either panel. Molecular-identity overlap
with tmQM has not been excluded. Paper/motif citations are not evidence that
every designed derivative has been synthesized, has the assumed spin, or is
a stable isolated complex. Exact source relationships are recorded per case
and in `data/challenge/sources.json`.
