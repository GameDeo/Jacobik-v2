# Task 3 / Step 4 report

## Status

Step 4 completed through response labeling, descriptive Part C calibration, the Part 5a rank benchmark, and the five-parent Part 5b tail pilot. The placebo result was evaluated as a sensitivity finding and did not stop the authorized full-surface workflow. The final candidate uses zero NCI scales, so it is byte-identical to the base constraints database and produces no NCI-driven ranking changes in these diagnostics. It is a calibration-only descriptive result, not evidence of a useful NCI effect.

The current code branch contains the response-labeling and calibration performance fixes. The main checkout `data/` was read-only; no TMQM QM property fields or `motion_validation` data were accessed. The GPU request sentinel has been removed. No official registry entry was appended.

## Completed pipeline

- **A1 replay:** 360 calibration pools replayed with the amended parity tolerances; maximum raw-total residual was `3.973e-4` nats.
- **Step 3b all-root research fit:** completed 129 supported roots, with 85 new fits, reuse of the unchanged primary root, and 43 fold-support skips. The separate placebo stop diagnostic was sensitivity-only; the placebo gain (`8.24e-6` nats/row) was below its threshold (`1.23e-4`).
- **A2/A3 frozen full-surface selection and coverage:** 1,182 retained center-aware motion paths; 623 excluded for lacking original signal and none excluded solely by the coverage floor. Coverage retained 93 informative bond-atom, 73 bond-ring, and 42 ring-ring groups; 507 ref-only cells were disabled. Frozen surface SHA256: `3b427692e81a98772e5a09267028ea0a21c2f2e97b5f4b7eb15af0c416af97b6`.
- **A4 request budget:** 7,279 unique energy requests and 187 unique force centers. The preflight p95-based estimate remained below the 3-hour gate.
- **Part B labels:** 1,182 unique center-aware motions, 6,130 response rows, 3,503 unique energy evaluations, and 203 force evaluations. The final resumed labeling pass took 598.18 s for 614 newly generated motions; this is the resumed-pass time, not cumulative time across earlier runs and optimization restarts.
- **Part C:** 120 groups and 6,130 label rows; 4,000 bootstrap replicates. `gamma_shared=0`; all family scales are zero. Candidate DB SHA256 `48ac82b56d71fd6e43515ecf830d4712a71de3df533b4d6ff8762f7ebe7549bf` matches the base DB byte-for-byte. Sidecar SHA256: `4eca2a0a4f9ad835491a9a3a031f6b642`.
- **Part 5a:** 360 calibration keys and 15 calibration-tail keys. Since all scales are zero, the ranking benchmark safely skipped NCI runtime evaluations; paired NCI-on/off energy, force, regret, and top-1 changes were exactly zero.
- **Part 5b:** all five tail parents present in the calibration frame were re-solved (20 rows). The two batches took 36.37 s and 16.75 s; measured aggregate solver wall time was about 53 s, excluding model startup overhead.

## Performance work

- Vectorized NCI opportunity evaluation and cached tensorized root surfaces: about **10.6× faster** on the real-surface comparison, with value/gradient agreement to floating-point precision (`<=1.4e-16`).
- Reworked response-label scoring to reuse per-shard molecule/scorer setup and evaluate seed-specific center identities correctly. The old motion key merged distinct seed geometries; labels now preserve the center/seed identity.
- Reworked bootstrap weighting from repeated group-by-row scans to one row pass: about **7.2× faster** on 6,130 rows and 105 groups, with exactly identical weights.
- Skipped ranking-time NCI evaluation when every scale is exactly zero, avoiding work whose output is provably zero.

These measurements target the identified bottlenecks. They do not claim that the separate research CV/root-tuning stage was rewritten; that stage is materially more expensive than a production pass with frozen settings.

## Validation and data safety

Targeted response and surface tests passed: **34 passed, 2 xfailed**. The memory-guarded binning/columnar/control tests passed in a fresh process: **14 passed**. The full repository run completed with **547 passed, 9 failed, 4 skipped, 5 xfailed**. Eight failures were budget pauses because the sequential run had accumulated a 2,378 MiB process peak before tests enforcing a 1,280 MiB peak-RSS limit; those tests pass in the fresh-process run. The remaining failure is an unrelated existing Q6 test expecting the absent `jacobik.ik.scorer.Q6_AGG` symbol. The first full-suite attempt from the feature checkout also had five fixture-path errors; the reported corrected run used the main checkout as cwd and explicit fixture roots.

The experiment index, experiment README, and S2b manifest retain the recorded pre-run hashes/mtimes. The candidate registry record remains proposal-only. Step 4 spec SHA256: `21fdf96a79b36aa5713f32c65ce90217c580ab51709034e237ab4b745a41e334`.

## Key artifacts

- `task3_step4_report.json` — machine-readable completion report
- `step4_spec.json` — frozen Step 4 specification
- `a3_coverage_floored.json`, `surface_a3_frozen.json` — final full-surface coverage and frozen surface
- `gpu_labels_floored.json` and `.progress.json` — response labels and progress metadata
- `candidate/part_c_report.json`, `candidate/candidate_nci_v2.json`, `candidate/candidate_constraints.json` — descriptive calibration and candidate DB
- `part5a_rank_benchmark.json`, `part5b_tail_resolve_pilot.json` — ranking and tail-pilot diagnostics
- `proposed_registry_record.json` — proposal only; no official registry mutation

Code commits include `785a88b`, `62437b8`, `8c630ff`, `2644880`, `be3c53b`, `60c4c40`, `8314ebd`, and `0fbf119`.
