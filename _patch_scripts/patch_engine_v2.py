#!/usr/bin/env python3
import sys

PATH = "src/jacobik/engine.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

steps_done = []


def do_replace(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"FATAL: pattern for step [{label}] not found", file=sys.stderr)
        sys.exit(1)
    if n != count:
        print(f"FATAL: pattern for step [{label}] found {n} times, expected {count}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, count)
    steps_done.append(label)


OLD_TAIL = '''         process or is preceded by hundreds of unrelated torch.randn/rand
         calls with a different manual_seed set beforehand.
"""'''
NEW_TAIL = '''         process or is preceded by hundreds of unrelated torch.randn/rand
         calls with a different manual_seed set beforehand.
2.6.0 : solve() gains use_shape_alm (default True), shape_rho, shape_tol,
         sym_weight, threaded through to JointIKSolver (ik_solver.py 2.3.0).
         Root cause addressed: the old anchor pulled every donor toward a
         FIXED absolute Cartesian target computed once, before the solve,
         from the ligand's *initial* pose -- as soon as the free rigid pose
         or the ligand's own torsions moved away from that pose (routine
         for flexible/chelating ligands such as EDTA), the anchor was
         scoring deviation from a stale frame, so a strong anchor fought
         the statistical score over an unreachable configuration while a
         weak one let the donor shell drift uncoordinated. No single fixed
         lambda(t) schedule could serve both regimes (e.g. rigid PtCl4 vs
         flexible EDTA) -- a documented failure mode of fixed-weight
         penalties in non-convex optimisation (see e.g. arXiv:2505.20628),
         not a tuning miss. use_shape_alm=True replaces it with a shape
         term computed via the optimal rigid rotation (Procrustes/Kabsch)
         between the ideal polyhedron and the CURRENT donor arrangement,
         RECOMPUTED EVERY STEP -- an approach related to continuous shape
         measures (CShM; Pinsky & Avnir; Alvarez group), the established
         way to score deviation from an ideal coordination polyhedron in
         the inorganic chemistry literature -- combined with an Augmented
         Lagrangian (dual-ascent) multiplier in place of the fixed
         schedule, so the enforcement strength self-adjusts per complex
         instead of one hand-tuned curve trying to serve every ligand.
         Also adds RDKit-canonical-rank-based detection of topologically
         equivalent donors (engine.py, new in this version) feeding a
         symmetry-tying restraint on their M-D distances (mirroring
         crystallography's SIMU/SAME restraints) -- addresses the observed
         spurious bond-length spread between chemically indistinguishable
         donors (e.g. EDTA's two amine N atoms differing by ~0.15 A with
         nothing previously tying them together). Backward compatible:
         use_shape_alm=False restores the pre-2.6.0 behaviour byte-
         identically. See ik/ik_solver.py JointIKSolver 2.3.0 changelog
         for the full technical rationale.
"""'''
do_replace("E5-version-history", OLD_TAIL, NEW_TAIL)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
