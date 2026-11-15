# D2 investigation checkpoint contract — 2026-11-10

Scope: the owner's approved causal investigation and separate gate-input audit.
Historical D3 verdicts are unchanged. No new generation or MACE runs, billed
jobs, promotion reserve access, tuning on stage 2, or production changes.

## Fixed-pool comparison

Use saved C, D and DQ6 rows at seeds 11–13 for LUQVOK, SESWIZ, LIPDEU, YABYEJ,
LOPJEF, ZAQNUE, HOXJUZ, USAPOU and UMUBAF. Keep arms, variant/assignment indices,
builder atom order and energy labels. Reconstruct native historical settings
from the recorded arm flags, not the possibly stale live fingerprint metadata.
Validate DB hashes, source coordinates, pool/record alignment and native
top/floor score including recorded bend prior (absolute tolerance 5e-4 on
exported coordinates). If exact scoring is not reconstructed, report the
comparison as inconclusive rather than adjusting it to match. Score on the
same runtime shell with only the DB changed; across-arm comparisons require
matching atom order, site graph, E/Z and assignment identities.

For every seed report top/floor and pool geometry, statistical score, saved
energy, term contributions, and rerank choices using native score plus the
fixed-coordinate statistical score delta. The latter preserves DB-independent
per-row priors; it is valid only after native score validation. Distinguish a
score preference flip on fixed rows from absence in the returned pool. Absence
does not establish where the search lost the basin.

Test LUQVOK with weight-only, mean/variance-only, Pd/O and Pd/C complete-table
substitutions. Keep all other fits, effective floor and geometry fixed, match
components by mean where counts agree, and recompute the Q1 wall where the
shape changes. Compare margins for C and D top rows; show physical-coordinate
rigid ligand translations as directional diagnostics, not relaxed minima.

## Fit provenance and audit

Trace important resolved keys to archived observations and available D2
observations, with ancestor sampling and pooling recorded. Compare direct
filtering to collateral sample changes; do not claim reconstructed D2 fits
unless model fields agree. Keep locked exclusions fixed in candidate fits.
Use the audit worker's independent input-only dispositions and correction
ledger to distinguish benchmark validity from statistical and search defects.
Report blocked provenance, failed reconstruction and ambiguous repairs.

After the source-only audit freezes, join its dispositions to existing stage-1
records. Report the original primary population and the disjoint audit
dispositions, with per-parent seed means and 10,000 parent bootstrap draws
(seed 11). These are retrospective descriptions, not a new gate. Do not
re-score stage 2 on a selected subset or reuse outputs as corrected targets.

All parent tasks use the CPU diagnostic runner. Freeze the inputs before
opening diagnostic outputs; store derived evidence under research/data/audit.
Commit only on investigation branches and stop after reporting the checkpoint.
