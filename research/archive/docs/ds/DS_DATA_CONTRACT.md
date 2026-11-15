# DS data package — Checkpoint A, 2026-11-10

Owner scope is donor data, stable sampling and confidence diagnostics.
Checkpoints stop for audit. Scoring and search changes follow separate review.

The branch carries the current stereo fold-in and D2 extraction pipeline.
Production defaults must retain fingerprint 5b56969afb9c and current golden
arrays. Experiments use donor_identity_reconcile and stable_ancestor_sampling.
Raw inputs and existing DBs remain read-only. Outputs use research/data/audit.

## Registered comparison

The four DBs are D0, D-data, D-sample and D-both, toggling the two fields
independently. All use the same source files, models, os_sanity and
fit_exclude_locked settings. D0 must reproduce the recorded D2 model surface
before interpreting comparisons. Stable sampling requires fresh IDs assigned
at extraction, before filtering. Legacy inputs keep their legacy path.

## Confidence

Report local independent support, resolved keys and pooling ancestry. Damage
uses independent parents, with LnQM siblings grouped by structure_id.
The doses are one parent and ceil(1%, 2%, 5% of support), with actual fractions
reported, 32 random replicates, seed 11. Also register 64 parent bootstraps.
Error families are observed source misinterpretations, parent removal and
uniform-ball source-coordinate displacement bounded by 0.02 A per atom.
Observed errors require source-supported replacement observations.

The metric is the first tested local evidence fraction with severe structural
failure, plus dose-response failure rates and Wilson 95% intervals. A severe
failure is >25 kJ/mol adverse top-1 change or newly broken coordination on
at least 2 of seeds 11–13 against the matched unperturbed case. Record single
seed events and execution failures. A missing trial never becomes a pass.
No observed failure is censored at the maximum tested dose. Label fixed-pool
and generation coverage explicitly. Sparse fits require their true ancestor
refits, never an independent substitute.

The development codes are LUQVOK, SESWIZ, LIPDEU, YABYEJ, LOPJEF, ZAQNUE,
HOXJUZ, USAPOU, repaired UMUBAF, MIGTEE and EFOXII. The neutral cases were
selected by smallest absolute historical D delta, with code tie-break.
Frozen protocol and exact hashes are in the main checkout at
research/data/audit/2026-11-10_DS/checkpoint_A/protocol.json.

## Source evidence and compute

Source review includes all 11 Pd training parents and audited preservation
fixtures. Corpus census uses retained observations first. A complete raw
re-extraction belongs to the approved CPU build job.

Measured D2 builds previously completed two 64-worker builds in 11:44 on one
CPU node. Two rounds for four DBs imply approximately 24 minutes before
new ID/sampling overhead. Budget estimate is 24–32 minutes, 0.4–0.54
CPU node-hours, with a 40-minute cap. The prepared regular-QOS script
requires the owner's per-job go. The full stress and gate jobs need their
own estimates after the first measured pilot.

Checkpoint A reports tests, source correction ledger, pilot extraction,
sampling invariance, confidence protocol and estimates. Checkpoint B covers
the four actual builds and structural stress results. Corrected gate
execution follows its own frozen contract and owner go.
