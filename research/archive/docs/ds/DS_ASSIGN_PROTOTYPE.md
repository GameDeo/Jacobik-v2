# Donor assignment prototype — 2026-11-11

`mapping._assign_donors` (enabled only by `JACOBIK_ASSIGN_PROTO=1`, under `donor_identity_reconcile`; no profile flag yet,
default path and fingerprint `5b56969afb9c` unchanged). One injective, evidence-ranked assignment replaces the rule chain
(sibling motifs, prior-contact preference, remote guard, restore, replacement).

Design: symmetry classes by colour refinement on the ligand heavy graph (element + H count, bond orders ignored so
resonance-equivalent atoms share a class). A declaration is a count of donors in its class. Per class, as many atoms as
declarations are chosen by evidence: supported contact (BO >= 0.15, ratio <= 1.21) first, then BO, then distance. Atoms beyond
ratio 1.6 are never chosen (declaration recorded as unrepresented). No prior rule is consulted.

## Results (local, no billed compute)
- Fixtures (JAYCEV, UKUWON, AGAPEG, ILEPOE): identical site sets to the rule chain. BADQIH: Rh {N2, N4, N6} + two eta2 faces,
  square-pyramidal, matching the CSD name (rule chain gave one N, t-shaped).
- 138-task census: 30/34 exact audited targets in both arms; 45/46 sources identical, only BADQIH differs.
- 2% quick extraction (about 2,500 parents, 36 s per arm): 38 parents (about 1.5%) differ in Q1 rows. Assignment has more rows in
  29, fewer in 1, equal in 8. Atoms gained: 90% supported, median BO 0.42, ratio 0.99. Atoms given up: 2% supported, median BO 0.00,
  ratio 1.73. The rule chain drops a declared donor on a collision (symmetric ligands); the assignment keeps distinct atoms.
  KIFXUS: rule chain kept a 3.68 A declared donor, assignment picks a 2.39 A one.
- Tests: all donor-set assertions pass; 5 fail only because they assert old event kinds (local_donor_identity,
  legacy_supported_identity_*), which the prototype does not emit.

## Open
- 16 of 242 gained atoms have BO < 0.05 and 4 have ratio > 1.3 (declared donors with no better class member).
- Colour refinement over-approximates automorphism orbits; no case found yet.
- Not yet run on the 1,285-parent review sample (Perlmutter) or the full corpus. Hapto face step unchanged.

## Update after the review pack (2026-11-11)
The first review exposed two defects, both fixed: (1) zero-BO atoms were chosen as donors (VEZJOD's two bogus arene faces; 16 gained
atoms at BO exactly 0, next lowest 0.106), so a chosen donor now needs BO >= 0.05; (2) QIXBOO's real N28 (2.22 A, BO 0.36), absent
from the source SMILES, was lost: a declaration with no bound class atom may now move to an unclaimed supported contact of its ligand
(`donor_assignment_transferred`, review event). Result on the 2% sample: 51 parents differ (30 assignment adds a donor, 8 swaps, 12
assignment drops only BO=0 declared donors); gained atoms 96% supported (min BO 0.106), given-up atoms 1% supported. Census unchanged
(30/34). Review pack: `assign_review_pack.py`; `_assign_donors` costs about 2 ms per parent (under 1% of assemble).

## Owner review (2026-11-11) and adoption
Two blind-ish packs of 22 disagreements checked against CSD by the owner: assignment correct or better in 17, tie in 1 (ZAFVOR),
indeterminate/neither in 4 (DOKCEL, WALMAB, NAPKAS not judged, YAJWEN adds a weak O that tmQM drift forces), rule chain never better.
The rule chain's errors were systematic: zero-BO declared donors kept (AFUFEM, ZIBMEG, BEZMIG, ENICAF: a carbon "close" only by
geometric necessity between two bound donors), the wrong symmetry twin chosen (NOGDAQ, XIWKAR, GUQVOG), donors dropped on collision.
The tmQM xTB refinement sometimes changes the binding mode away from CSD (DOKCEL, SOWPIE, KEXVUI, KIFXUS, YAJWEN): the tmQM
geometry, not the CSD structure, is the target, and the assignment follows the geometry.
Adopted as profile flag `donor_assignment` (default off; requires `donor_identity_reconcile`). New events: donor_assignment_changed,
donor_assignment_transferred, donor_assignment_unrepresented, supported_contact_not_declared (KIFXUS-type). Census arm via
DS_CENSUS_DR_EXTRA='{"donor_assignment": true}'. Rule chain (reconcile rules 5-10) stays until the 1,285-parent review passes.

## Experiment A: pi-capable seeds set the face distance scale (2026-11-11) — rejected
Change tried (under donor_assignment): `d0` in `_detect_haptic_groups_v2` from carbon/ring seeds only. NUHPEL's phantom eta3 face
disappeared on its own, but the census fell from 31/34 to 25/34 exact audited targets: DUCPAT, IBUTAD, UKUWON, WIJXIW, XAMGEY and
XIKWEU each lost real kappa donors (sites 6 -> 4/5, geometries changed). The seeds that define the distance scale also carry
sigma-donor sites downstream, so the scale cannot be narrowed without that coupling. Reverted; the face-evidence rule
(`hapto_face_unsupported`) stays. Not tuned further.

## Perlmutter review and final rule set (2026-11-11)
The 1,285-parent review (rounds 1-4, `DS_STEP3_REVIEW.md`) passed with these additions to the assignment: the **face-evidence rule** (an eta>=3 face is a substituent when no member
has BO >= 0.05 and none lies within ratio 1.21 of the metal; all 9 faces dropped in the sample were confirmed against CSD by the owner; NUHPEL is the motivating case) and the
whole-parent **`structure_sanity`** screen (a separate flag; 79 parents; all 10 sampled were broken). Two attempts to widen the assignment were measured and **not adopted**:
the pi-capable-seed distance scale (above) and a distance exception for non-carbon donors with a zero bond order (F6: 167 parents are touched, the owner preferring one evidence
rule to a special case; reverted in `8e6eeb0`). The reconcile rules 5-10 are superseded; their removal is prepared as a patch and not applied (`DS_RULE_INVENTORY.md`, `DS_HANDOFF.md`).
