# Notebooks

Run from anywhere inside the repo (each notebook finds the folder containing `src/`).

| notebook | what it does | time |
|---|---|---|
| `03_full_refit.ipynb` | the 2026-10 refit, stage by stage (the same functions as `refit/run_refit.py`): OS audit, contact cleaning, sharded extraction (benchmark complexes removed), held-out selection of the Q1 ladder variant and the pooling relevance factor, full fit (Q1/Q2 pooled, Q3-Q6, S_orient incl. per-substituent haptic rows, QG/QP priors, sidecar), checks, optional DOE aqua validation, gated install | hours (12 workers: extraction 1-2 h, selection ~1 h, fit ~1 h) |
| `01_benchmark.ipynb` | guards with each DB, then configurations x DBs (old v6 vs new refit) x seeds on N48, RING10, held-out 28 and the 24-case panel; MACE single points of #1 (and of every returned conformer with `ENSEMBLE = True`); `bench.targets` table (#1 median, > 200, top-1, rho, solve time); paired differences against the seed noise; writes `data/benchmark_runs/<date>/all_results.csv` | ~5-10 min per configuration x DB x seed x case set (GPU) |
| `02_visualization.ipynb` | per-case |dE| with seed ranges, configuration scatter, solve time, 3-D overlays of the best conformer on the crystal | ~1 min |

Order: `03` (refit) -> `01` (benchmark) -> `02`. Older versions: `archive/`.

The mathematics: `../ARCHITECTURE.md`. What changed and why: `../CHANGELOG.md`.
