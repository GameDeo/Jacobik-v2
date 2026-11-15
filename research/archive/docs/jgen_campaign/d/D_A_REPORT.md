# D-A database checkpoint — 2026-11-07

Branch: `jgen/t-typing`. Earlier D-A implementation and QO source fixes are in
`b1ac9bf` and `4b60865`. This is the audit §1b follow-up to `050c1e8`.

## Final comparison

The initial one-full-build run finished in 9,020.4 s, with 6,204.2 MB peak RSS,
two workers, and one numerical thread per process. No VM or GPU was used. This
follow-up did **not** run a full build. It refit only Q5, Q6, Q6P and NCI, then
compared those with the persisted Q1–Q4 and QO output against the protected
2026-11-01 target. The existing 1e-4 tolerance is unchanged. Mixture components
are aligned by optimal parameter assignment before comparison; a focused test
confirms a real weight change still fails.

| Quantity | Target / generated keys | Different | Missing | Extra | Exact | Within tolerance | Max absolute delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 118,673 / 118,673 | 0 | 0 | 0 | 61,123 | 57,550 | 2.67e-11 |
| Q2 | 298,737 / 298,737 | 0 | 0 | 0 | 96,777 | 201,960 | 2.89e-11 |
| Q3 | 10,267 / 10,267 | 0 | 0 | 0 | 4,932 | 5,335 | 3.81e-12 |
| Q4 | 1,306 / 1,306 | 0 | 0 | 0 | 933 | 373 | 6.16e-11 |
| Q5 | 4,155 / 4,155 | 218 | 0 | 0 | 1,827 | 2,110 | 154.6803 |
| Q6 | 535 / 535 | 0 | 0 | 0 | 508 | 27 | 8.88e-16 |
| Q6P | 336 / 318 | 182 | 18 | 0 | 136 | 0 | 33.7936 |
| QO | 10,220 / 10,220 | 0 | 0 | 0 | 8,390 | 1,830 | 4.89e-15 |
| NCI | 13 / 13 | 13 | 0 | 0 | 0 | 0 | 0.0252 |

The detailed quantity report is
`research/data/jgen/campaign/D/A1_full_reproduction/build_report_A2_comparison.json`;
the compared database with the targeted reruns merged is
`constraints_A2_quantities_rerun.json` in the same output directory.
The targeted artifact's sidecar is `constraints_A2_quantities_rerun.meta.json`
and records Q5's runtime model label plus hashes for its rerun outputs and all
12 reconstructed histogram inputs.
Q6P/NCI histogram reconstruction inputs and its shard census are under
`research/data/jgen/campaign/D/A2_hist_rebuild/` (`hist_rebuild_report.json`).

## Legacy admission thresholds checked

| Fit path | Admission setting | Value | Result |
|---|---|---:|---|
| Q1 | pooled fitter `N_INDEP` | 50 | Matches legacy pooling threshold |
| Q2 | pooled fitter `N_INDEP`; local `FT_N_MIN` | 50; 15 | Matches its separate legacy pooling/component rules |
| Q3, Q4, Q5, Q6 | fitter `N_MIN` | 30 | Corrected from 15 to 30 |
| QO | `orientation.N_MIN` | 50 | Already sourced from the production module |
| Q6P | PMF-table `N_MIN` | 3,000 | Corrected from 15 to 3,000 |
| NCI | surface-table `N_MIN` | 1,000 | Corrected from 15 to 1,000 |

Q6 now follows both legacy pooling stages: it pools period-group raw
observations before fitting, then calls
`fitter._add_pooled_q6_metal_keys` on fitted entries. The rerun on the retained
expanded Q6 file produced all 535 target keys, with zero out-of-tolerance
differences and no missing or extra keys.

## Remaining differences and runtime effect

- **Q2–Q4:** Their apparent mismatches were component-order swaps. Optimal
  assignment gives zero out-of-tolerance entries; the largest parameter deltas
  remain within the original tolerance. Permuting mixture components does not
  change the density or solver score.
- **Q5 means:** The 218 out-of-tolerance entries are all one-component fits
  with `kappa=0.01`; only the means differ. Their current weighted circular
  resultants are numerically zero (median `2.78e-17`, p95 `6.01e-17`, maximum
  `7.45e-17` across these keys), so the data contain no preferred phase. With
  the concentration at its floor, the fitted mean is unidentifiable and tiny
  fit/input perturbations select different phases. For any angle, moving this
  one-well mean changes its log score by at most `2*kappa = 0.02 kT` per
  torsion. The per-entry tag differs on all 4,155 rows (`wells` in the target,
  `wells_folded` in the refit); this is a label difference. Runtime fold
  behavior comes from the `.meta.json` `q5_model` field, not that per-entry
  label. The builder now writes `q5_model=wells_folded` explicitly.
- **Q6:** No remaining out-of-tolerance differences after restoring the
  30-row gate and both legacy pooling stages.
- **Q6P admission:** The real defect was `N_MIN=15` instead of 3,000. The
  earlier report's claim that the M13 glob caused the result is withdrawn:
  fitting the rebuilt histograms with and without the M13 filter gives
  identical 318-entry output. The filter is retained for clear source
  separation. Of 318 common entries, `r0`, `b`, `kappa` and `n` match; `w`
  differs on 182 entries, only in classes 4, 5 and m. The target `w` carries a
  PMF offset while the refit is a pure core; runtime subtracts `w(r0)`. The
  effective score difference is at most 0.18 kT below 50 kT and 3.8 kT deep in
  the core. The 18 absent entries are nine `bent` and nine `lin` angle keys.
  There is no runtime reader for these: `q6.build_runtime` requests only the
  empty-angle key and then the class-5 empty-angle fallback.
- **NCI:** Admission now uses the legacy threshold, producing the same 13 keys
  as the target. All 13 surfaces still differ. The one bounded inspection
  found the `NCI:('chp', '*')` entry differs only in its `f` grid (same 44×18
  shape, maximum cell delta 0.021); across all tables the maximum parameter
  delta is 0.0252. The remaining cause is a refitted surface-grid difference;
  its historical source is unresolved after the requested bounded check.
  Runtime effect is zero with production `nci=False`.

The owner-approved Q6P and NCI exceptions are therefore bounded and reported
with their causes and runtime effects. No tolerance was widened.

## QO and D-A flags

The prior isolated QO issue was filename-alphabetical loading (LNf before X1),
which changes seeded GMM initialization. QO now loads X1, X2, X3, LNf in the
archived order; all 10,220 entries match within tolerance. Its explicit source
directory and per-file hashes are in the run `.meta.json`.

The charge/oxidation census covered 125,810 parents across 44 metals: 1,626
implausible formal charges and 7,917 unknown oxidation states. Zn had 7,482
parents, 498 implausible charges and 457 unknown OS; NEPNEF is flagged at Zn
+14. The full census is in the run output. The full build used D flags off;
thermal pruning therefore changed zero entries/components. It recorded 360
locked parents and restored `ref_ll` for Q1 118,673; Q2 298,737; Q3 10,267;
Q4 1,306; Q5 4,155 entries.

T1d keeps `hapto_bond_gate`, `hapto_label_canon` and
`hapto_chelate_exclusion` on; `donor_promote_geometric` stays off. Promotion
added 1,646 Q1, 7,120 Q2 and 1,632 QO rows with a longer tail, changed 324 keys
by at least 20 rows, and changed no eligible cal/tail parent.

## 2026-11-07 audit follow-up checkpoint

Merged master in commit `59b5298` (including D-B `13917d3`). Corrected the
post-alignment maxima above and limited mixture component reordering to
`means`, `vars`, `covs`, `covars`, `kappas`, and `weights`; other list-valued
fields are left in place. The regression test checks that behavior and still
confirms a real weight change fails.

`src/fit/build_db.py` now requires an explicit `--fit-frame-path`, extracts the
Q7 table from the frame's `surface_fit` crystal geometries, writes the sibling
`constraints.bends.json`, fails if more than 5% of parent extractions fail, and
records the sidecar SHA256 and source-frame SHA256 in `constraints.meta.json`.
The extraction report is also included in `build_report.json`. The existing
Q6P/NCI differences remain owner-approved exceptions with causes and runtime
effects as detailed above.

Small checks only: 20 focused tests passed; `py_compile` and builder CLI help
passed. No full build, refit, VM, or GPU run was performed. D-A stops here for
audit.
