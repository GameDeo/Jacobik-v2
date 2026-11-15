# D-A progress — 2026-11-07

- Read audit `audit_jgen_checkpoints_2026-11-06b.md` §1 in full.
- Corrected Q3–Q6 admission thresholds to legacy `N_MIN=30`; set Q6P to 3,000
  and NCI to 1,000 after checking each quantity's fit path.
- Q6 now uses both legacy metal-pooling stages. On the retained expanded Q6
  file it produces 535/535 keys with zero out-of-tolerance differences.
- Comparison now aligns mixture components by optimal parameter assignment.
  The regression test confirms reordering passes and a real weight change
  fails; it passed.
- Refit Q5/Q6 from retained expanded observations. Rebuilt only Q6P/NCI
  histograms from the retained raw shards (34 shards, two workers plus parent,
  one numerical thread each, zero parser/accumulator errors), then refit those
  quantities. No full build was run.
- Recompared the targeted refits plus persisted Q1–Q4/QO. Q2–Q4 now pass under
  order-independent comparison. Q5 means and Q6P/NCI exceptions, including
  runtime effects and the corrected Q6P cause, are detailed in
  [D_A_REPORT.md](D_A_REPORT.md).
- 2026-11-07 audit follow-up: merged master (`59b5298`, including D-B `13917d3`),
  corrected the post-alignment Q2–Q4 max deltas in [D_A_REPORT.md](D_A_REPORT.md),
  and restricted component permutation to the named mixture parameter fields.
- Wired Q7 sidecar generation into `src/fit/build_db.py` from an explicit
  `--fit-frame-path`: extract `surface_fit`, fail if parent extraction failures
  exceed 5%, write `constraints.bends.json`, and record its SHA256 plus the
  source frame hash in `.meta.json`. No refit or full build was run.
- Small checks passed: 20 focused tests, `py_compile`, and builder CLI help.
  D-A stops at this checkpoint for audit.
