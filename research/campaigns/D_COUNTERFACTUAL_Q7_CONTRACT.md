# D-B Q7 fixed-row counterfactual contract — 2026-11-04

## Question and frozen comparison

On the same stored A4b pool rows, do `q7_bound_centre` and the combined D-B Q6+Q7 settings improve or worsen score ranking against stored MACE energies? This remains a fixed-row filter and does not estimate solver movement.

- Population: successful D0/F2 parents with complete A4b pool data, cal + tail after locked exclusions. Match by parent, seed, and row index. No val, reserve, or locked parents.
- Arms: existing production `control`; newly scored `q7_bound_centre=true`; and newly scored `q6_physical_all=true, q6_weight_14=0.0, q7_bound_centre=true`. The Q7 arms use the frozen provisional crystal-angle sidecar at `research/data/jgen/campaign/D/q7_bound_angles.bends.json`; the DB itself and its production `.meta.json` are copied unchanged into a temporary DB directory. `q7_shift_max=10.0` remains the registered default.
- The provisional Q7 table was built from the 120 cal/tail frames after locked exclusions. D-A typing tables/flags are not included in these arms because they are not present in this D-B worktree; this is explicitly a score-side-only filter.
- Pool Spearman: per-parent mean of seed-level Spearman(score, -MACE energy); report parent mean and parent-cluster bootstrap 95% CI.
- Top-1 regret: per-parent mean across seeds of MACE energy for the top-scoring row minus that seed's pool MACE floor; report kJ/mol and parent-cluster bootstrap 95% CI.
- Lenient rows: count where score is in the pool's top quartile and MACE energy exceeds the pool floor by >100 kJ/mol; report total rows and affected parents.
- Bootstrap: resample parent IDs with replacement, 2,000 draws, RNG seed 20261104. Report each arm and paired on-minus-control differences with percentile 95% CIs.
- Missing/failed parents and seeds are reported. No reruns or post-hoc exclusion except the pre-registered locked set and pools lacking at least two finite MACE-energy rows.

The filter is descriptive, not a gate. A flag is dropped only when both Spearman and regret worsen with CIs excluding zero, as registered in directive D. Results do not override the separate crystal-frame Q6 check.
