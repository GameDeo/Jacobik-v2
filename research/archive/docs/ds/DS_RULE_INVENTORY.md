# Extraction rule inventory — donor identity, faces, sampling, filters (2026-11-11, updated after the Perlmutter review)

One row per rule that can change which atoms are donors or which observations enter a fit. Flags default off (fingerprint `5b56969afb9c`).
"Chain" = the `donor_identity_reconcile` rule chain; "Assign" = the `donor_assignment` replacement (requires reconcile).

## Donor identity (src/jacobik/stats/mapping.py, topology/donor_classifier.py)
| # | Rule | Flag | Trigger | Event kind | Motivating case | Test | Interacts with |
|---|---|---|---|---|---|---|---|
| 0 | Legacy: declared donor with BO < 0.15 moves to the best-BO atom of its heavy-graph component (any element) | none (default path) | unsupported declaration | none | D2 production | golden | all |
| 1 | H-aware graph alignment | reconcile | SMILES and BO graphs align differently once H counts are matched | hydrogen_aware_donor_mapping, hydrogen_alignment_ambiguous | symmetric ligands | test_donor_identity | 2-10 |
| 2 | Terminal O/S sibling motif | reconcile (Chain) | equal-element terminal atoms on one C/N/S centre | local_donor_identity | LUQVOK (22 Pd observations) | acetate unit tests | 3, 5 |
| 3 | Azine-pair sibling motif | reconcile (Chain) | two aromatic ring N sharing one aromatic C | local_donor_identity | JAYCEV | JAYCEV fixture | 2, 5 |
| 4 | Rejected-face kappa preservation | reconcile | C-X eta2 face rejected; X supported (BO >= 0.15, ratio <= 1.21) | rejected_face_kappa_preservation | DUCPAT | rejected-chelate tests | face stage |
| 5 | Supported-first priority | reconcile (Chain) | declaration already on a supported contact | none | — | unit | 2, 3 |
| 6 | Prior-contact preference (single-atom) | reconcile (Chain) | unsupported alignment image of a symmetric ligand | legacy_supported_identity_preserved / _collision | UKUWON, BADQIH | fixtures | 0 |
| 7 | Prior-contact substitution (multi-atom motif) | reconcile (Chain) | unsupported alignment atom in a motif | legacy_contact_identity_preserved | UGUGAF (ratio 1.24) | unit | 6 |
| 8 | Remote guard (ratio > 1.6) | reconcile (Chain) | declared donor beyond range | remote_declared_donor_unrepresented | ~12% of sampled parents | unit | 6, 9 |
| 9 | Restore | reconcile (Chain) | declaration left without donor; legacy had a supported contact | legacy_supported_donor_restored | QIXBOO-type | unit | 6, 8 |
| 10 | One-for-one replacement | reconcile (Chain) | count full, weak unsupported non-legacy donor | legacy_supported_donor_restored | — | unit | 9 |
| A1 | Symmetry classes (colour refinement) | donor_assignment | always | — | — | test_donor_assignment | A2 |
| A2 | Evidence-ranked injective assignment per class | donor_assignment | always | donor_assignment_changed, donor_assignment_unrepresented | BADQIH, NOGDAQ, XIWKAR, GUQVOG | 13 tests | replaces 2, 3, 5-10 |
| A3 | BO floor 0.05 and ratio <= 1.6 for a chosen donor | donor_assignment | declared atom with BO 0 | donor_assignment_unrepresented | AFUFEM, ZIBMEG, BEZMIG, ENICAF | unit | A2 |
| A4 | Transfer of an unbound declaration to an unclaimed supported contact of its ligand | donor_assignment | declaration with no bound class atom | donor_assignment_transferred | QIXBOO | unit | A2 |
| A5 | Undeclared supported contact: review only | donor_assignment | geometry binds an atom the source graph does not declare | supported_contact_not_declared | KIFXUS | census | none (never adds a donor) |
| A6 | Unbonded eta>=3 face is a substituent | donor_assignment | face with no member BO >= 0.05 and none within ratio 1.21 | hapto_face_unsupported | NUHPEL | NUHPEL fixture | face detector |

## Sampling and filters
| Rule | Flag | Effect | Test |
|---|---|---|---|
| Stable observation IDs, hash-priority ancestor sampling | stable_ancestor_sampling | zero collateral resampling in 250 watched buckets (73 under the legacy sampler) | test_ds_data |
| Charge/oxidation filter | os_sanity | whole-parent removal, about 1.6% per quantity, concentrated in sparse keys | existing |
| Locked exclusions | fit_exclude_locked | custody, not a model knob | existing |
| Broken-structure screen | structure_sanity | whole-parent exclusion when >= 25% of >= 6 carbons have exactly two neighbours at a bend below 150 degrees (79 parents in the full corpus; owner review: all 10 sampled are broken, e.g. JENDAH); a parent-level census rule, not a donor exception | test_structure_sanity |

## Known non-rules (decided, not changed)
tmQM xTB geometry drift away from CSD (DOKCEL, SOWPIE, KEXVUI, KIFXUS, YAJWEN): the tmQM geometry is the target. XIWKAR kappa2-vs-eta2 and similar face-stage questions: recorded,
unchanged. Rejected experiments: pi-capable seeds for the face distance scale (census 31/34 -> 25/34, `DS_ASSIGN_PROTOTYPE.md`); keeping non-carbon donors with a zero bond order on distance
evidence alone (F6: 167 parents are affected by the bond-order floor; the exception was reverted, the owner preferring one evidence rule to a special case).

## Removal plan
Prepared but not applied: `DS_reconcile_removal.patch` (see `DS_HANDOFF.md`). After the audit and a passing check level, rules 2, 3, 5-10 (`_legacy_donor_correspondence`,
`_reconcile_donor_identities`) are removed because `donor_assignment` supersedes them; rules 0, 1, 4 and the face stage remain. The legacy path stays for D2 reproduction.
