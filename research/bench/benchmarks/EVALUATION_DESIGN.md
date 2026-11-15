# Evaluation design for JacobIK versus Architector

Proposal, 2026-10-17. This specifies the future evaluator; it does not report
comparative results. JacobIK documentation and code were read, not modified.
The current Architector runner has not yet been executed on Linux.

The comparison should answer four separate questions: does the method build
the requested complex; does it recover relevant structures; does its own
ranking select good structures; and how much computation does that take?
Keep these outcomes separate rather than combining them into one score.

## What the existing implementations imply

[JacobIK architecture](../ARCHITECTURE.md), especially sections 1-3 and 8,
describes free-ligand embedding followed by statistical-potential IK, then
deduplication and ranking. It requires a coordination-geometry label and uses
spin multiplicity. Its statistical score is an ordinal surrogate, not a
free energy. Complex relaxation is outside its native solve.

The [Architector paper](https://doi.org/10.1038/s41467-023-38169-2) describes
search over core geometries, xTB-based construction and optional final
relaxation, followed by energy ranking. Our adapter currently supplies CN,
not a geometry label. `--no-final-relax` still uses xTB during assembly.

Those current inputs and outputs are not yet a matched comparison. The
geometry policy below supersedes the CN-only *primary comparison* proposed
in the older baseline protocol; it does not change the current runner.

## 1. Freeze equal chemical information

Use one builder-independent target record: complete ligand graphs, exact
donor groups, protonation/tautomer, ligand-instance identity, fixed ligand
stereochemistry, metal, oxidation state, total charge, scalar spin and
environment. Both adapters must preserve this same chemical problem.

- Convert `unpaired_electrons = u` to JacobIK `spin = u + 1`, and pass `u`
  to Architector's unpaired-electron fields. This is a scalar-spin task,
  not spin-state prediction or a treatment of spin-orbit multiplets.
- Supply the authored oxidation state to both, through their supported APIs.
  JacobIK's current constructor accepts `oxidation_state=`. Log effective
  state, charge and spin rather than relying on different inference rules.
- Preserve eta groups and covalent tethers. A pi face counts as one JacobIK
  donor group but can occupy multiple Architector template slots. Equal
  chemistry does not mean equal numerical CN parameters.
- Fixed stereochemistry is mandatory; unspecified coordination, sulfur or
  ring configurations remain search variables. No target coordinates,
  ligand conformers extracted from them, or reference site assignments enter
  generation. Each method may use its native free-ligand preparation, with
  its cost included.

Use two separately reported information settings:

| Setting | Information supplied | Interpretation |
| --- | --- | --- |
| Geometry supplied | Same nominated geometry family or frozen set of families, without donor-to-site assignment | Construction and isomer/conformer search given the geometry information JacobIK normally receives |
| Geometry open | Same chemical topology; explore the same predeclared geometry-family list under one total budget | End-to-end search, including choice of coordination geometry |

For the open setting, a benchmark wrapper must enumerate JacobIK geometry
labels and merge its outputs under a frozen ranking rule. All such solves
and embeddings count toward the case's total budget. This is a wrapped
pipeline, not a claim that native JacobIK infers geometry. Architector's
native geometry search must use the corresponding declared search scope.

The semantic mapping needs validation, especially for piano-stool/haptic
targets, CN10, and alternative high-CN arrangements. If the same search
scope cannot be expressed, record a capability gap and report that case
separately. Do not force an arbitrary template or choose the best geometry
after inspecting the reference or evaluation energies. Alternative families
such as SAP/TSAP must not be excluded by the supplied information.

## 2. Keep delivered structures and common refinement separate

| Output track | Measurement | Cost accounting |
| --- | --- | --- |
| Native output | Each method's delivered structures, native order, validity, reference similarity and common single-point energies | Entire native call, including its own embedding, assembly, selection and relaxation |
| Common refinement | Frozen number of submitted candidates from each method, relaxed by the same external procedure | Native generation plus the same permitted external refinement budget |

The native track answers what a user gets by running each pipeline. Different
internal algorithms are allowed, including Architector's xTB relaxation; their
cost is part of the result. Also report Architector assembly-only as an
explicit additional arm, including its existing hydride restriction.

Common refinement measures usefulness as a starting point for subsequent
chemistry. Use the same calculator, optimizer, charge, spin, solvent,
convergence criteria and maximum steps/evaluations for both, independently
of either native scorer. Label any externally refined JacobIK pipeline
explicitly; these structures are not native JacobIK outputs.

Freeze submitted K and the selection/deduplication rule. A proposed first
setting is the first 10 native-ranked candidates; invalid candidates occupy
their submitted slots and are not silently replaced. Fewer outputs remain
fewer outputs. Preserve native files and record each refinement's parent,
convergence, geometry displacement, energy change and topology changes.
An optimizer reaching its step limit is not a converged basin.

## 3. Apply independent validity checks before quality scores

Validate coordinates against the authored target, not against either
builder's bond declarations, score or success flag:

- Finite coordinates, exact atoms and hydrogen inventory, correct whole
  ligand graphs and tethers; no lost fragments or added filling ligands.
- Intended donor identities and binding modes, including specified primary
  versus secondary-sphere contacts. Do not count every nearby O as a donor.
- Fixed stereocenters and E/Z configurations retained; proton transfers or
  tautomer changes recorded as chemical-identity changes.
- Gross overlaps and broken ligand bonds. Use broad, frozen chemical checks,
  not JacobIK's fitted distributions or Architector's internal acceptance
  tests as the definition of validity.
- Haptic face, eta-contact pattern and slippage; all donor arms of a
  high-denticity ligand retained; hydride and end-on N2 topology retained.

Distance/contact checks need explicit metal/element and haptic conventions.
Calibrate broad thresholds on references and method-independent distorted
fixtures before scoring outputs; publish raw distances and threshold
sensitivity. Do not reject an unusual bite merely for departing from an
ideal 90-degree angle. Mark genuinely ambiguous coordination separately.

Store independent flags rather than one opaque Boolean. Distinguish adapter
gap, exception, timeout, empty output, invalid chemistry, ambiguous binding,
and refinement-induced dissociation. A physically dissociating designed ion
can indicate a problematic target assumption; investigate it under a shared
procedure and retain the original outcome in the record.

Native top-1 means the original first-ranked output. If it is invalid, top-1
fails; a valid lower-ranked structure may still contribute to best-of-K.

## 4. Measure geometry, generation and ranking independently

| Question | Metric |
| --- | --- |
| Can it construct the target? | Probability of any valid output, valid native top-1, valid fraction of returned candidates |
| Can it recover a known structure? | Native top-1 and best-of-K graph-matched heavy-atom RMSD; reference-hit curves at fixed budgets |
| Is the coordination sphere right? | Metal-plus-donor RMSD, graph-depth-two core RMSD, donor-distance and bite-angle errors |
| Does it explore alternatives? | Known isomer-family recovery, distinct valid optimized basins, coverage versus time and K |
| Does it rank well? | Native top-1 energy, pool floor, ranking regret, energy ordering within isomer classes and near the floor |
| Are outputs strained? | Same-method single-point minus converged local-minimum energy, refinement displacement and cost |
| Is it efficient? | Valid/reference-hit/basin coverage versus total generation time, plus total time including refinement |

Use graph-constrained atom mappings and permutations of truly interchangeable
whole ligands. Preserve donor roles and fixed stereochemistry; matching
elements with a free Hungarian permutation alone can conceal wrong
connectivity or isomerism. Do not align each ligand independently for the
whole-complex score. Define and freeze core atom selection, including eta
groups, before comparing core metrics.

Use proper rotations for the principal stereochemistry-aware comparison.
For unconstrained enantiomer pairs, also report mirror-equivalent recovery
without counting a permitted mirror as a construction failure. Reflection
must never erase a fixed carbon configuration or a sulfur diastereomer.

Keep crystal recovery separate from isolated/solvated-minimum recovery.
Packing, counterions and solvent can select a crystal conformer that is not
the isolated lowest-energy state. Compare original experimental coordinates
as archived, and prepare separately labeled common-method relaxed references.

For a common evaluator E, a valid candidate pool P and nominated reference r:

    signed reference gap = E(native top-1) - E(r)
    pool floor           = min(E(x) for valid x in P)
    ranking regret       = E(native top-1) - pool floor
    local strain         = E(x) - E(converged local relaxation of x)

Regret is defined only if native top-1 is valid and evaluable. Keep its failure
rate alongside conditional energy statistics. Do not make |reference gap|
the primary quality score: a valid alternative below the reference is not
automatically worse. Report signed gaps, identity and basin together.
Do not subtract energies produced by different electronic-structure models.

A per-case minimum over the union of both methods and independent reference
searches is a useful empirical energy baseline. It is a best-found value,
not a certified ground state; use the same frozen union for both comparisons.
Never pool absolute energies across different compositions.

Native ranking and common-energy reranking are different endpoints. Keep
both; common reranking must not overwrite the native top-1. JacobIK scores
and Architector energies have no common numeric scale. For correlations,
orient both rankings so that a better native rank should predict lower E;
report insufficient pool sizes and ties rather than manufacturing a value.

## 5. Use a shared energy yardstick, with an independent audit

MACE-POLAR-1 in float64 is a practical screening evaluator consistent with
JacobIK's current measurements. Pin the exact checkpoint, software version,
charge/spin convention and numerical settings. The
[MACE-POLAR-1 paper](https://arxiv.org/abs/2602.19411) describes training on
OMol25 and evaluation of transition-metal and lanthanide systems. That does
not establish accuracy for every target or for every model size/checkpoint.

Neither MACE nor GFN2 should be the sole final judge. JacobIK's development
has used MACE measurements and MACE-fitted bend parameters; Architector uses
GFN2 in generation and ranking. A shared calculator is useful, but that alone
does not make the evaluation independent of method development.

Before the comparison, nominate a manageable stratified electronic-structure
audit, for example 8-12 targets spanning haptics, sulfur, extreme bites,
macrocycles, polar interactions and N2. Evaluate equal, prespecified candidate
sets from both methods with a suitable independent DFT protocol. Specify
basis/ECP, relativity, dispersion, solvent, SCF/spin checks and convergence
before inspecting outcomes. Additional disagreement cases can be diagnostic,
but must not replace the prespecified audit. Report evaluator disagreement.

Keep environments explicit. Vacuum MACE energies are not aqueous free
energies; water used in one builder's optimization does not add implicit
solvation to a later vacuum evaluator. Report that vacuum diagnostic as such,
and use a consistent solvent treatment for solution-stability claims.

Existing crystal references are heavy-atom-only. Before reference-energy
comparisons, prepare explicit, atom-mapped hydrogens under one documented
protonation/tautomer policy, without altering the archived crystal heavy
atoms. Treat uncertain protons separately. No direct energy calculation on
a heavy-atom-only complex should enter the main comparison.

## 6. Make ensemble claims proportional to the evidence

Use two levels of classification: coordination/isomer family and optimized
conformational basin within that family. Deduplicate the union of both
methods using the same chemistry-aware rules and converged common-method
structures, retaining every parent's method, rank and seed. A shared relaxed
basin can have differently strained native representatives.

Report known-family recall only where an independently defined reference
family set exists. For the other cases, report distinct observed basins and
method overlap, not recall of all stable states. Do not reward broken-bond
structures, duplicates or high-energy distortions as ensemble diversity.

Relevant observables include rac/meso, fac/mer, cis/trans, SAP/TSAP,
metal/sulfur configurations, haptic face/slippage, and secondary-sphere
H-bond arrangements. Do not impose identical thresholds on every phenomenon.
Classifiers and basin tolerances need frozen definitions and sensitivity
checks; count results at the case level, not one replicate per source frame.

Multiple structures do not establish thermal accessibility. Population
comparisons require verified minima, consistent solution free energies,
temperature, standard state and degeneracies. Exchange barriers are not
isomer free-energy gaps. JacobIK's statistical score must not be converted
into Boltzmann weights.

At present the challenge panel has one experimental coordinate case (D03)
and two cases with published computed coordinates (B03/B05). Their twelve
frames comprise three labeled families per case at two computational levels,
not twelve independent states. The baseline has eight experimental cases.
Most challenge targets currently support validity and mechanism diagnostics,
not quantitative crystal-recovery or complete thermal-ensemble claims.

## 7. Match budgets and keep failures visible

Use the same Linux host for both methods. Record CPU, GPU, memory, numerical
threads and device path. A CPU-only comparison and each method's practical
accelerated configuration can both be informative; label them separately.
Do not quote a hardware-independent speedup from unlike resource allocations.

Start with three independent repeats (17/29/43) and a 600-second total
generation deadline per case/repeat, then add frozen 60/300-second budget
points for coverage curves. Equal integer seeds do not synchronize two
different algorithms. Include all geometry trials, failed attempts and
ligand preparation in the budget. Distinguish solver sampling repeats from
repeats that change the free-ligand embed; JacobIK's default embed seed is
fixed unless the adapter explicitly varies it.

Measure cold startup, warm throughput and per-case preparation separately.
Preload models/DBs equally for warm measurements, but do not give one method
free cached target-specific ligand preparation. Include compilation/warmup
in cold timing and document reuse in warm timing. Evaluation time is separate
from native generation and included for any generation-plus-refinement claim.

Architector's requested conformer count is per core geometry, not an equal
total output count or equal work budget. Report requested counts, actual
outputs and unique valid outputs; use matched total-time curves as the main
budget comparison and top-K curves as a complementary view. Evaluate all
outputs with cheap checks; expensive refinement has its own common K cap.

The present runner writes candidates only after the builder returns. It
cannot yet measure time to first valid output or recover unreturned partial
work on timeout. Use completed-call budget points unless a later adapter
adds reliable checkpoints for both; do not invent intermediate timestamps.

Publish an all-target capability/usable-output rate and a paired quality
comparison over the frozen common representable scope. An adapter gap is
not a demonstrated chemical failure of the underlying method. Runtime
failures inside that common scope remain in success-rate denominators.
Conditional RMSD/energy tables must show how many targets and outputs are
missing; do not hide failures by dropping rows or assign arbitrary energies.

Summarize seeds within each case, then compare methods with paired case-level
differences and cluster-bootstrap confidence intervals. Resample related
scaffold families together when estimating broader uncertainty. Source frames,
seeds and conformers are not independent complexes. Report each failure-mode
stratum, tails and individual cases; with this small, related panel, broad
population claims or precise subgroup rankings are not justified.

## 8. Freeze the evidence and build the evaluator outside both methods

Audit exact-complex and ligand-scaffold overlap with tmQM, LnQM, JacobIK's
training/refit records and Architector's published examples/benchmark data
where available. Track missing identifiers and unauditable exposure. The
current guarantee is independent sourcing, not verified training disjointness.
Freeze development versus final test scaffolds before inspecting results;
matched derivatives stay together. Cases repeatedly used to tune settings
become development cases, and a later final test needs additional untouched
scaffolds. Never delete a difficult case after seeing method performance.

Freeze both method versions, JacobIK profile and all overrides, constraints
DB plus sidecar, Architector environment lock, adapter settings, input/source
hashes, metrics, thresholds and evaluator checkpoints. JacobIK's current
default legacy profile and experimental u1 are different arms; select the
primary arm in advance. Record xTB availability and preparation failures so
silent MMFF fallback is not mistaken for the nominated pipeline.

Recommended implementation order, all under `benchmarks/`:

1. A common target/output schema and two adapters, with immutable native
   coordinates, native ranks, atom mappings, input provenance and run events.
2. Independent identity, stereochemistry, coordination and haptic checks;
   graph-aware geometry metrics and capability-aware failure accounting.
3. Offline reference preparation, common single-point evaluation and native
   ranking/generation metrics, with calculator failures kept explicit.
4. Common-refinement jobs, basin/isomer classification and the DFT audit.
5. Paired reports, per-case overlays and coverage/time curves with explicit
   denominators. Evaluators consume saved runs and never call back into a
   builder to repair, rerun or tune it.

Verify the evaluator with independent fixtures: rigid rotations/translations,
legal atom/ligand permutations, fixed-stereo mirrors, wrong donor identities,
dropped arms, altered hapticity, broken tethers, duplicate snapshots and
failed relaxations. These check the scientific definitions, not just whether
an implementation reproduces itself. No JacobIK core edits are required by
this design.
