# D-B counterfactual rescoring contract — 2026-11-04

## Question and frozen comparison

On the stored A4b pool rows, does `q6_physical_all` improve or worsen score ranking against stored MACE energies? This is a fixed-row filter; it does not estimate solver movement.

- Population: every parent with a successful D0/F2 per-term record and complete A4b pool data, cal + tail after locked exclusions. Rows are matched by parent, seed, and row index. No val, reserve, or locked parents.
- Arms: current production defaults (control) and `q6_physical_all=true`, `q6_weight_14=0.0`. This intentionally omits 1–4 pairs, as amended by the owner on 2026-11-06. All other settings stay at defaults. Same code, CPU platform, DB, and rows.
- Pool Spearman: per-parent mean of seed-level Spearman(score, -MACE energy); report parent mean and parent-cluster bootstrap 95% CI.
- Top-1 regret: per-parent mean across seeds of MACE energy for the top-scoring row minus that seed's pool MACE floor; report kJ/mol and parent-cluster bootstrap 95% CI.
- Lenient rows: count where score is in the pool's top quartile and MACE energy exceeds the pool floor by >100 kJ/mol; report total rows and affected parents.
- Bootstrap: resample parent IDs with replacement, 2,000 draws, RNG seed 20261104. Report arm means and paired on-minus-control differences with percentile 95% CIs.
- Missing/failed parents and seeds are reported; no reruns or post-hoc exclusion except the pre-registered locked set and pools lacking finite MACE energy.

This analysis is descriptive and is not a gate. No flag is dropped from this result alone unless both Spearman and regret worsen with CIs excluding zero, as registered in directive D.
