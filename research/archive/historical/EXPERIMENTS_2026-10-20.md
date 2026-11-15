# Research archive, 2026-10-20

This cleanup keeps the adopted solver state in the active tree and preserves the exploratory work
for reference.

## Archived code and document snapshot

`research_2026-10-20.zip` (local archive only since 2026-11-03: `research/archive/data/2026-11_repo_untracked/research/archive/historical/`) contains the expanded pre-cleanup `HANDOFF.md`,
`CHANGELOG.md`, and `ARCHITECTURE.md`, the one-off NCI/Q6 audit scripts and tests, and snapshots of
the MACE input-precision fix files. The adopted production implementation remains in `src/`.

## Full local experiment data

The raw report and refit folders are large and ignored by Git. They were moved intact to:

- `data/_archive/refit_reports_2026-10-20/` — 10/18–10/19 reports, plots, raw measurements, and
  source snapshots, plus older loose refit logs and the experimental BVS reports (about 707 MB before
  adding the loose outputs).
- `data/_archive/nci_experiments_2026-10-20/` — candidate NCI tables, histograms, and calibration
  artifacts (about 268 MB).

The compact records needed to resume work are in `data/refit/evidence/`: N1 baseline reconciliation,
dev80 NCI results, Q6 runtime/transfer checks, analytic-HB transfer results, and the MACE precision
audit. The deployed DB and established refit inputs remain under `data/refit/final/`, `clean/`,
`observations/`, `source/`, `thermal/`, and `q6rep/`.

`data/_archive/` is local-only and not tracked by Git. Keep it when moving this checkout if the full
raw results are needed; the ZIP preserves the uncommitted code and documentation snapshot in Git.
