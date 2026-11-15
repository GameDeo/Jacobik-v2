# T1 progress — 2026-11-02

Worktree `jgen/t-typing`, based on `347496f` (`273f2ccbf892`).

## Step 1: flagged typing

Added `hapto_bond_gate` and `hapto_label_canon`, both default off. The bond gate applies `orientation.ETA2_BOND_RATIO_MAX` with loader covalent radii; the label rule applies to Q1/Q2/S_orient subclasses and the coordination-site display label, without changing per-donor labels. Implementation commits: `0f06960` and `4e767bd`. `pytest -q tests/test_typing_t1.py`: 5 passed; default fingerprint and three flags-off classification hashes match.

## Steps 2–3: re-extraction and analysis

One-shard flags-off estimates: X1s000, 4,000 molecules, 186.5 s with three BLAS threads; X1s007, 4,000 molecules, 167.5 s with two. The first full flags-off attempt at one thread per worker was stopped after five additional shards when projected time exceeded the three-hour gate; completed files were reused. The revised estimate was 6,030 s for three settings on three workers, under the gate.

All 34 shards are now complete for off, gate, and both. Off processed 26 shards in 1,605 s and reused 8; gate processed 34 in 2,223 s; both processed 34 in 2,035 s. All settings had zero molecule/parser errors. The flags-off identity check matched every archive haptic row: X1 16,378/16,378, X2 16,657/16,657, X3 5,837/5,837, LNf 1,706/1,706. It found extractor-only rows 46/39/8/0 by split, reported as drift.

Analysis corrections: QO rows do not store a scalar `value`, so the source join uses raw-face identity and derives centroid distance. Q2 mixture windows are in cosine space; archived angles are converted to cosine before source-row matching. The final report and full CSV tables are in `research/bench/jgen_campaign/`.

## Step 4: default golden

Not run: T1 §2 forbids solves, but `golden.py check` performs CPU solves for seven stored cases. No solve was made. The focused default fingerprint and exact flags-off classification checks passed. This conflict and the no-solve decision are stated in `T1_REPORT.md`.

No refit, constraints DB write, solve, VM, or GPU run was performed. Stopping at the §7 checkpoint after the reporting commit.

## Amendment T1b: chelate exclusion and fitted-component diversity

Added `hapto_chelate_exclusion` (default off), applied before the bond gate and label canonicalization. It excludes a
candidate η² `{C,X}` face when the face carbon has at least two declared/SMILES κ donors in the same ligand, while
preserving those donors as κ sites. The new Zn acetate, dithiocarbamate, and η²-CO₂ integration tests pass (8 tests
total). A one-shard pilot (`X1s000`, 4,000 molecules) completed in 191.2 s with zero errors. The full 34-shard,
three-worker extraction completed in 2,238.9 s total with zero errors. No refit, DB write, or solve is part of this
amendment.

T1b extraction completed for 34/34 shards: the 33-shard pass took 2,047.7 s and the pilot took 191.2 s; all had zero errors. The analysis excluded 1,698 η² faces (C=N 1,143; C=O 516; C=S 35; C=P 4), found 297 fit-table key changes at |Δ|≥20, and no cal/tail parent changes. The Zn carboxylate Q1 key was 544 (gate-only 810; both 504), outside the requested tolerance; this is reported without tuning. The fitted-only census contains the expected 791 components (315 Q1 + 476 Q2), all with source rows. Median distinct source parents/component was 41; median largest-parent share was 3.48%. Tables and report are in the campaign directory.

## Amendment T1c: geometric bound test

Measured the remaining Zn C=O faces mapped to the missing Zn carboxylate Q1 rows. The exact gate-to-T1b join found
109 retained η² C=O faces corresponding to T1b `C_haptic`/`O_carboxylate` rows. Their face oxygen was undeclared in
all 109; its Zn-scaled ratio had p0/p5/p25/p50/p75/p95/p100 = 1.0565/1.0722/1.0853/1.0969/1.1032/1.1258/1.1394,
all ≤1.21. The first T1c extraction exposed a donor-retention issue: geometrically qualifying neighbors were used to
reject faces but were not restored as κ sites. That run was discarded. The classifier now retains all qualifying
bound neighbors as κ donors after rejection. The 2.2 Å and 3.2 Å acetate tests pass; focused typing tests: 10 passed.
The corrected pilot took 196.9 s; the remaining 33 shards took 2,084.1 s on three workers. All 34 shards completed with zero errors. T1c excluded 1,004 faces (C=O 959, C=N 30, C=S 9, C=P 4, B=C 2); the Zn carboxylate key is 1,044 (gate-only 810), overshooting the expected count by 234. There were 265 key changes at |Δ|≥20 and no cal/tail parent changes. The expectation miss and tables are recorded in T1_REPORT.md. No refit, DB write, or solve.

## Amendment T1d — 2026-11-04 final pass

Split chelate-face rejection from default-off geometric heteroatom promotion. Registered the promotion ratio (1.21) and scope (`rejected_chelate_face_neighbors`). Moved the extraction entry point to `src/fit/typing_reextract.py`; the research bench path is a thin wrapper. Made `fit` modules lazy-loaded so the runner's profile overrides apply before import.

Two T1d extraction passes were discarded before the final result: the first loaded flags before worker overrides due eager `fit` imports; the next exposed carbon neighbors in the promotion predicate. Fixed both, added a carbon-exclusion test, and reran the final settings. Final face-only pass: 34 shards, 1,974 s, zero errors. Final promotion pass: 34 shards, 2,107 s, zero errors. Both used three workers with `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.

Promotion adds 1,646 heteroatom donors and changes 1,223 of 125,810 source parents DB-wide; no eligible cal/tail parents change. It changes 324 keys by |Δ|≥20. The Zn carboxylate Q1 key moves 394→1,044. The added Zn-key rows have median/p95 M–O distances 2.062/2.149 Å versus 1.954/2.055 Å without promotion; 59.5% exceed the base p95 and 4.2% exceed the base maximum. This is a longer tail, so promotion stays off for D1. Full tables and provenance are in `T1_REPORT.md` and the `T1D_*.csv` files. No refit, DB write, solve, GPU, or VM.
