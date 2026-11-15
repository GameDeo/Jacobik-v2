# tools/ inventory (2026-10-17, P3 A4)

Status: IN USE (imported by a kept tool / tests / refit, named in CHANGELOG 2026-10-15 or later, or a current command in
tools/jdev/README.md), SUPERSEDED, ONE-OFF. SUPERSEDED / ONE-OFF files are in `archive/code_2026-10-17.zip`
(see its MANIFEST.txt) and removed from the tree. Kept: 13 jdev scripts + the MACE model, 25 budget scripts (incl. the new
common.py), the 3 dev_tmqm data files; archived: 46 files
(gpuwork/ 43, label_audit, q6_core_ablation, q6_onset_walls_dev.json).

## tools/jdev
| path | what it does | status | evidence |
|---|---|---|---|
| tools/jdev/__init__.py, __main__.py | `python -m tools.jdev` entry (fingerprint, guards, panel, case, diagnose, audit) | IN USE | README command table |
| tools/jdev/bench.py | benchmark matrices, paired statistics | IN USE | tests/test_devices, test_methodology_2026_10_15, budget/pools.py, tracks.py; CHANGELOG 10-15 |
| tools/jdev/build_replay_solver.py | generates src/jacobik/ik/solver_replay.py | IN USE | named in solver_replay.py header; CHANGELOG 10-03 |
| tools/jdev/cases.py | case loader (CSD fixtures, tmQM splits) | IN USE | tests/*, budget/* , refit/thermal_widths.py |
| tools/jdev/cuda_check.py | on-GPU consistency check | IN USE | tests/test_devices docstring ("the on-GPU test") |
| tools/jdev/diagnose.py | per-atom excess-energy diagnosis | IN USE | __main__ diagnose; CHANGELOG 10-16 d |
| tools/jdev/env.py | paths, fingerprint, loaders | IN USE | imported by __main__, golden, measure |
| tools/jdev/golden.py | golden check of stored-pool ranking scores | IN USE | required workflow; CHANGELOG 10-17 |
| tools/jdev/p4_compare.py | unified DB comparison, including archived add-on DBs | IN USE | HANDOFF P4; CHANGELOG 10-18 |
| tools/jdev/archive_db_leftovers.py | manifest + SHA256-verified DB archive writer (does not delete originals) | ONE-OFF | CHANGELOG 10-18; data/_archive/ |
| tools/jdev/measure.py | solve + MACE energy measurements | IN USE | __main__ case / audit / diagnose |
| tools/jdev/multiseed.py | all seeds of a complex in one solver pass | IN USE | tests/regression/run_panel.py, tests/test_gpu_fast_path.py |
| tools/jdev/start.py | oracle start geometries | IN USE | budget/pools.py, golden.py |
| tools/jdev/viz.py | benchmark figures | IN USE | README, notebooks/02_visualization.ipynb |
| tools/jdev/MACE-POLAR-1-S.model | MACE model of the budget tools | IN USE | budget/pools.py, forces.py |
| tools/jdev/label_audit.py | labeller before/after audit | ONE-OFF (archived) | no import, not in README, no CHANGELOG entry since 10-15 |

## tools/jdev/budget
| path | what it does | status | evidence |
|---|---|---|---|
| common.py (new) | shared loaders: OUT / DB, pool files, MACE dE, StatsLookup | IN USE | pools, tracks, gate, q6_diag, nci_census |
| pools.py | solve pools / MACE energies per arm | IN USE | required; golden.py |
| gate.py | offline re-rank gate for a changed score | IN USE | required; CHANGELOG 10-17 |
| oxidation_audit.py | exact dev80 and hash-sampled training label agreement, including mapping coverage | IN USE | CHANGELOG 10-18 oxidation experiment |
| co_edta_check.py | diagnose Co-EDTA donors, embeds, angle bends and seeds | IN USE | CHANGELOG 10-18 Co-EDTA investigation |
| tracks.py | paired comparison of pool arms | IN USE | required; CHANGELOG 10-16 b |
| forces.py | MACE curvature vs score terms | IN USE | required; CHANGELOG 10-16 b |
| q6_diag.py, q6_dists.py | Q6 wall / PMF diagnostics; pair distances | IN USE | required; CHANGELOG 10-16 f, 10-17 |
| bvs_check.py | bond-valence coupling check | IN USE | required; refit/bvs_fit.py imports it |
| gfn2_dimer_scan.py | GFN2 dimer scans for Q6 walls | IN USE | required; stats/q6pmf.py comment |
| nci_census.py | NCI type census | IN USE | required; stats/nci.py comment |
| nci_calibration.py, nci_calibration_analysis.py, nci_probe_mechanism.py | independent rigid physical probes, strict fragment energies, direction/counting diagnostics | IN USE | NCI calibration implementation 10-18 |
| nci_training_calibration.py, nci_training_analysis.py, nci_calibration_report.py | energy-blind scaffold-disjoint cohorts, full-complex response/curvature checks, conditional calibration and report | IN USE | NCI calibration implementation 10-18; versioned manifests |
| nci_thermal_audit.py, nci_density_balance_audit.py | original Q1 Hessian cohort reused for NCI compliance and parent/scaffold sampling sensitivity | IN USE | NCI calibration implementation 10-18; research only |
| nci_embedding_validation.py | separate common-start factorial, independent Cp/arene screen and mapped anionic-N-N census | IN USE | NCI calibration implementation 10-18; no embedding adoption |
| nci_targeted_rescore.py | exact Q6/NCI replacement on a fixed stored-pool cohort; no solves/new energies | IN USE | CHANGELOG 10-18 N2 targeted ranking |
| nci_contact_diagnosis.py | individual NCI contact geometry/scores and removed-Q6-well coverage | IN USE | CHANGELOG 10-18 N2 contact audit |
| energy_diagnosis.py | MACE64 forces, atomic model-local energy changes/global residual, exact-topology motion scans and score response | IN USE | CHANGELOG 10-18 force-localization suite; tests/test_energy_diagnosis.py |
| energy_viewer.py, energy_viewer.html | offline 3D force/energy/bond/motion inspector from cached measurements | IN USE | CHANGELOG 10-18 interactive viewer; tests/test_energy_viewer.py |
| histogram_parity.py | scratch-only unified extraction vs independent histogram passes, bounded by shard/limit | IN USE | CHANGELOG 10-18 N2 local fix |
| ../../refit/nci_refit.py | NCI-only regeneration, isolated table fit and cleaned-data/cached-pool census; real process workers | IN USE | CHANGELOG 10-18 parallel NCI regeneration |
| oracle_targets.py, isomer.py | crystal targets; reference isomer | IN USE | required; pools.py imports |
| internal_stereo.py, export_xyz.py | ligand stereo check; xyz export | IN USE | required; CHANGELOG 10-16 b |
| ensemble_stats.py, validity.py, start_error.py | ensemble rho / basins; validity filters; start error | IN USE | CHANGELOG 10-16 b / d |
| analyze.py, decomp.py, load.py | error budget; component decomposition; pool table | IN USE | CHANGELOG 10-15 (error-budget tools); ensemble_stats imports load |
| xtb_stage.py, xtb_dof.py, rmsd_stage.py | GFN2 stages; constrained relaxations; RMSD | IN USE | CHANGELOG 10-15; gfn2_dimer_scan imports xtb_stage |
| coord_audit.py, coord_report.py | per-donor coordination audit | IN USE | CHANGELOG 10-15 / 10-16; oracle_targets imports coord_audit |

## tools/q6_ablation
| path | what it does | status | evidence |
|---|---|---|---|
| dev_tmqm/ (xyz, bo, csv) | the 32-complex dev set `dev32` | IN USE | budget/pools.py DEV |
| q6_core_ablation.py | Q6 core ablation (2026-10-14) | ONE-OFF (archived) | finished study; results in its report; no importer (only a 10-16 c port note) |
| q6_onset_walls_dev.json | dev onset walls for that study | ONE-OFF (archived) | read only by q6_core_ablation.py |

## gpuwork/ (removed; all archived)
GPU-port sandbox of 2026-10-02/03 (00_env_check ... 15_dedup_check, fk_fast*, tuning, pipeline, _common, notebook,
backup/, backup2/, handback/, scratch/). `git grep gpuwork` in src/ and tests/ finds nothing; the work is integrated
(CHANGELOG 2026-10-03: src/jacobik/ik, tools/jdev/multiseed.py, build_replay_solver.py). SUPERSEDED / ONE-OFF.

The one-off 2026-10-19 NCI/HB/Q6 audit tools and their reports were archived during cleanup. See `archive/EXPERIMENTS_2026-10-20.md`.
