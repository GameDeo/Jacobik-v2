# C-2 evaluation contract

## 0. Phase 0 measurements (registered 2026-11-02, before results)

Base: `b0134a2`, production fingerprint `5f74b14b57ed`. Seeds 11, 12, 13;
production solver and embedding defaults. No relaxation or energy optimisation
is performed by these measurements. Input/embed measurements may reuse A4b
unrelaxed pools (`273f2ccbf892`); solver measurements require current production.

Population: union of frozen cal and tail parents, subtract **all** lists of
codes in the main checkout's `locked_exclusions.json`. Assert exclusion before
loading geometry. Never replace missing/failed parents with successful ones.
FEKKAJ and NEPNEF remain in tables; exclude them only from primary statistics.
Failures are recorded per parent/seed and the loop continues. Five CPU processes
maximum, one local GPU job maximum; no Colab in Phase 0.

Placement: intercept each ligand's final theta initialisation before IK. Evaluate
FK and donor-group centroids (including haptic centroids). Store donor-to-target
Euclidean offsets, row maxima, M–D lengths and absolute M–D length errors. Repeat
after fitting the base pose to that row's torsion/bend-seeded conformer, retaining
orientation diversification with identical RNG state for the paired Kabsch draws.
Report quantiles 0, .25, .5, .75, .9, .95, 1 per parent, ligand class and overall.
Classes are macrocycle, haptic, mono-, bi-, polydentate in that priority order.
Build start_kabsch if >=10% of parents have median row maximum >0.3 A or the
paired fit lowers the overall median row maximum by >=30%.

Tiny pools: QAPFAS UYOVUA MIGTEE MIVXAR ZAZREZ SIVBUV HOXJUZ (eligible codes
only). Controls: first five eligible cal parents alphabetically outside tiny
and ring lists, fixed before measurement. Trace solved B, finite scores and
coordinates, embedding validity, every actual filter, 200-nat window, 0.5 A
symmetry dedup and output cap separately. Do not invent a validity filter if
the production path has none. Report score spread and pairwise symmetric RMSD
before dedup. Class (a): few DOFs and convergence to one structure; (b): loss of
distinct structures to a filter/window; (c): collapse of distinct starts by IK.
Only (b)/(c) may justify a general, default-off fix, defined before coding.

Ring census: BABNAV CIXYUD DOHTOM DUHHIX EFOXII FEKKAJ HOHZAG UYOVUA ZAFPOP.
Use unrelaxed A4b rows mapped to crystal atom order, seeds 11–13. Pick the row
with minimum shell_rmsd (ties: seed then row index). Compute the heavy-atom
RMSD-optimal automorphism and proper Kabsch alignment as in A4t ring_share.
Report each ring's sum of squared aligned atomic displacements divided by the
total heavy-atom sum. Overlapping rings share atoms: individual shares are
explicitly nonadditive; type totals use union-of-atoms. Classify rings as
ligand-internal nonaromatic saturated/partly saturated 5–8, macrocycle, or
metal-closed chelate; list aromatic/other rings separately to avoid silently
misclassifying aromatic rings. Record whether endocyclic bonds are FK joints.
Count eligible parents/ligands with nonaromatic 5–8 rings outside macrocycles.
Build ring_conf_pool only if ligand-internal rings carry deviation on >=4/9;
if chelate rings carry it on at least five of the nine parents, stop and
report per directive Amendment 1. Operationally a
type carries the deviation when its union accounts for >50% of total heavy-atom
squared displacement (the A4t R threshold). Mixed/overlapping types are reported.

Phase 0 report and c2/C2_PHASE0.json will record evidence and decisions. Commit
and continue unless the chelate-ring stop rule fires. No Phase 2 arms may run
until contract §1 is written and committed.

### Amendment 1 registration (2026-11-02, before added measurements)

The earlier any-parent stop interpretation is superseded. UYOVUA alone does
not stop the package. ring_conf_pool remains omitted (2/9); no topology fixes.
Count all nonaromatic >=9-member ligand rings outside the macrocycle path,
reporting ring size, all-single-bond test, donor count, and any further path
conditions. Retain UYOVUA tiny-pool class (a-prime), frozen by topology.

For BABNAV, CIXYUD, DUHHIX, EFOXII, ZAFPOP, inspect every aromatic ring with
individual displacement share >20%. Using the stored best-shell row and its
manifest mapping, identify FK torsion joints on attachment bonds. Report
wrapped row-minus-crystal dihedral and minimum wrapped distance from the
crystal dihedral to any initial beam seed of that joint, for the selected
seed. The 0.2-rad hit test uses the same FK quad convention. If no such joint
exists, report it; do not invent an attachment DOF. This measurement reuses
A4b unrelaxed rows and reconstructs production initialisation; differences
of production versions are explicit provenance, not a new A4b solve.

Placement and tiny-pool measurements use current production. Pair Kabsch
orientation random draws by saving/restoring Torch RNG state; measurements
must not alter solver inputs. Pairwise solved-row RMSD distributions use all
pairs up to 4096, otherwise a uniform deterministic sample without replacement
(seed 11); report pair count/total and symmetry caps. Numerical worker limits
are one thread per worker, at most four CPU census workers plus one GPU job.

## 1. Phase 2 registration (before any arm runs)

Arms: C production, U beam_union, S1 single_w48. Include K start_kabsch only
if Phase 0's placement condition passes, and T only if a measured general
class-(b)/(c) loss justifies a definition committed before coding. P omitted
under Amendment 1. Each tested arm and its C run in one bench invocation on
the same local RTX 2060 SUPER; no cross-invocation control substitution.
All eligible cal/tail parents and seeds 11–13, no relaxation. Failed parents
stay in the failure ledger and coverage-share denominator. Primary statistics
exclude FEKKAJ and NEPNEF, retaining their tables. No val/reserve/locked data.

Numerical analysis settings are registered in C2_SETTINGS.json; changing a
setting creates an explicitly reported analysis identity, never a silent
rerun. Bootstrap 10,000 parent-cluster resamples, seed 11; report percentile
95% CIs of mean paired parent differences. Seeds are averaged within parent
before resampling, and the same resampled parent indices pair arm/control.

Metrics: floor unrelaxed MACE energy minus crystal energy, minimum across
returned rows; top-1 is the highest returned score; minimum symmetric
heavy-atom RMSD against crystal; coverage indicators at <=1.0 and <=0.5 A;
minimum unrelaxed shell RMSD and shell-match indicators at <=0.3 and <=0.5 A;
distinct returned rows; Spearman correlation of returned score with unrelaxed
MACE energy (higher score means better, so ideal correlation is negative).
Undefined Spearman (one row or constant inputs) is null, with count reported.
Wall is prepare+seed generation+solve+dedup/assembly, excluding MACE, mapping,
export and analysis. The full shared prepare cost is charged equally to each
arm, while per-arm seed-generation costs are separate. Report wall ratio of
summed arm/control end-to-end walls, with parent-cluster CI.

S1 runs three independent width-48 seeds for noise. Each is compared with the
control union of seeds 11–13: union floor and minimum RMSD/shell values;
top-1 is the highest-scored row across that union; union rank correlation
uses all union rows. Control union distinct count uses 0.5-A symmetry-aware
best-first dedup of the union. Wall/compute charge is one S1 seed vs three C
seeds; total three-S1 wall is divided by three. Assert nominal IK row-steps
for each S1 seed vs C three-seed total within 10%; record actual solve-row
counts and steps as well. No pooling of S1's three seeds for its floor.

Regression guard: top-1 arm minus control >25 kJ/mol on >=2/3 seeds is a trip;
the mirror uses <-25 on >=2/3. Null is the larger trip count from rotating an
arm's seeds by one and two places against itself. Report C and arm nulls;
use the arm's null in its guard. Fail iff trips > null+2 AND trips >= mirror.
For S1, primary guard compares each wide seed with C union top-1; the arm's
self-rotation null still uses the three independent wide seeds.

Select an arm only if (floor paired CI upper <0 OR <=1.0-A coverage-share
paired CI lower >0), top-1 paired CI upper <+2 kJ/mol, guard passes, and
end-to-end wall <=1.3x. All four are required. S1 can only be a recommendation
to replace the three-seed budget. No qualifying arm => stop after Phase 2.
Otherwise use the directive's Phase 3 rules without changing these gates.

Before the panel, time a three-parent pilot (AJAQAD, BABNAV, JEGPUJ) for every
arm and log the extrapolated wall. Cold startup is reported separately when
available. Pilot data are instrumentation evidence, not a tuning round.

Phase 0 arm resolution: K included (101/120 parents exceed 0.3 A); T included
as `pool_window_fill`, the pre-coding definition in C2_PROGRESS.md. It retains
finite best-first distinct rows up to the existing output cap even outside
the global score window. It changes returned pool size, not IK rows or top-1.
No separate repair for SIVBUV class (c) is specified or claimed.
