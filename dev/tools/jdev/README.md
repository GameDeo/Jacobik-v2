# jdev: Jacobik development tools

Run from the repo root, in the pixi env (`MKL_THREADING_LAYER=SEQUENTIAL`, `PYTHONPATH=.`). Inventory of every tool and
what was archived: `tools/INVENTORY.md`. Terms and evidence workflow: `ARCHITECTURE.md` sections 1?2.

## One case or the panel (`python -m tools.jdev ...`)
| command | what it does |
|---|---|
| `fingerprint` | hashes of `src`, the constraints DB (+ sidecar) and the MACE model |
| `guards` | `pytest tests -q` |
| `panel --jobs 2` | accuracy/speed panel vs baseline (`--update-baseline` to accept) |
| `cases` | list case names (also `tmqm:<dir>:<split>:<CODE>`) |
| `case NAME [--energy] [--start etkdg\|full_oracle\|junction_oracle] [--seed 11] [--solve JSON]` | solve one case: score, donors vs crystal, MACE `dE_sp` |
| `diagnose NAME` | where the excess energy sits (donors, per-atom MACE excess, worst angles, close contacts) |
| `audit NAME [--xyz f] [--relax 300]` | is the energy difference trustworthy (composition, charge, `dE_sp`, `dE_basin`, strain) |

Defaults: DB `data/db/constraints_2026-11-01.json` (+ `.meta.json`) (`--db` / `$JACOBIK_DB`), MACE `$JACOBIK_MACE_MODEL_PATH`.
Results go to `runs/<date>/` with the fingerprint.

## Other tools in `dev/tools/jdev/`
| file | usage |
|---|---|
| `golden.py` | `python -m tools.jdev.golden check` (from the repo root with `PYTHONPATH=src:dev:.:research/bench`; pins its CPU worker layout) (ranking-score golden values of stored pools; must print `golden: OK`) |
| `bench.py`, `multiseed.py`, `viz.py` | benchmark plan / statistics, multi-seed solver pass, figures (imported by `dev/tests/regression/run_panel.py`, `notebooks/`) |
| `build_replay_solver.py` | regenerates `src/jacobik/ik/solver_replay.py` from `ik_solver.py` (canonical generator at repo root `dev/tools/jdev/build_replay_solver.py`) |
| `cases.py`, `env.py`, `start.py`, `measure.py`, `diagnose.py` | case loading, paths / fingerprint, oracle start geometries, energy measurements (libraries) |
| `cuda_check.py` | `python -m tools.jdev.cuda_check --db DB --mace MODEL` (on-GPU check) |

## Budget tools (`dev/tools/jdev/budget/`; env: `JACOBIK_BUDGET_OUT` = pool folder, `JACOBIK_BUDGET_DB` = the pools' DB)
Pools are `<OUT>/pools/<arm>/<set>/<code>_s<seed>.npz` (+ `.mace64.json`). Shared loaders: `common.py`.

| file | usage |
|---|---|
| `pools.py` | `solve --sets dev32 n48 --seeds 11 12 13 --start etkdg [--isomer ref] [--targets crystal]`; `mace --dtype float64 [--dirs ARM ...]` |
| `gate.py` | `--sets dev32 --seeds 11 --start ARM --energy mace64 --override MOD:ATTR=VALUE` re-rank gate of a changed score |
| `tracks.py` | `--arms u1c u1c+rep --energy mace64` paired comparison of pool arms, problem-complex table |
| `forces.py` | `compute --sets dev32 n48 --top 4` / `report`: MACE curvature vs each score term along soft DOFs |
| `q6_diag.py` | `terms` / `pairs`: Q6 wall vs PMF per pair class, the pairs behind a PMF loss |
| `q6_dists.py` | Q6 pair-distance distributions of picks vs crystals |
| `bvs_check.py` | `[--min-bonds 30] [--out data/refit/bvs]` bond-valence coupling at the metal (imported by `refit/bvs_fit.py`) |
| `gfn2_dimer_scan.py` | `scan [--orient 4]` / `fit`: GFN2 dimer scans for the Q6 repulsive walls |
| `nci_census.py` | `[--arm ARM] [--train 3000]` NCI types in crystals vs picks vs training |
| `nci_targeted_rescore.py` | under u1: `--out REPORT.json`; fixed 12-case NCI/Q6 delta check on post-OS pools and existing MACE64, no solves |
| `nci_contact_diagnosis.py` | under u1: individual selected NCI contacts and exact removed-Q6-well partition; stored coordinates only |
| `energy_diagnosis.py` | under u1: `--cases FETSUU PEWBOL --seeds 11 12 --out DIR`; MACE64 forces, atomic model-local energy changes plus global residual, topology/contact motion slopes and curvatures, Q6/NCI response, CSV/NPZ/plots; no optimization |
| `energy_viewer.py` + `.html` | under u1: `--roots DIAG_DIR [DIAG_DIR ...] [--out PAGE.html]`; offline linked 3D reference/before/after force, atomic-energy, bond and mode inspector; cached measurements only, no MACE/solves. Current page `data/inspect/nci_force_viewer.html` |
| `histogram_parity.py` | `--split X2s009 --out REPORT.json [--data DIR] [--limit N] [--exercise-exclusion]`: compare unified/independent passes in temporary scratch; no production regeneration |
| `oracle_targets.py` | diagnostic crystal-positioned targets (used by `pools.py --targets`) |
| `internal_stereo.py` | `[--sets dev32 n48] [--arms ...] [--detail CODE ...]` cis/trans, ring stereo of the picks |
| `export_xyz.py` | `CODE ... [--seed 11] [--top 5]` structures to `data/inspect/<CODE>/` |
| `isomer.py` | reference isomer assignment (used by `pools.py --isomer ref`) |
| `ensemble_stats.py`, `validity.py`, `start_error.py` | rho_isomer / rho_within and basins; offline validity filters; start-geometry error |
| `analyze.py`, `decomp.py`, `load.py` | error budget / per-term re-ranking, component decomposition, pool table loader |
| `xtb_stage.py`, `xtb_dof.py`, `rmsd_stage.py`, `coord_audit.py`, `coord_report.py` | GFN2 stages, constrained relaxations, RMSD to the reference, per-donor coordination audit |

`tools/q6_ablation/dev_tmqm/` is the 32-complex dev set (`dev32`) read by `pools.py`.

The one-off 2026-10-19 NCI/HB/Q6 audit tools and their reports were archived during cleanup. See `archive/EXPERIMENTS_2026-10-20.md`.
