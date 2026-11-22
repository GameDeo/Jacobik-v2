# Jacobik v1.0b

Release notes for the two builds of Jacobik v1.0b. Both builds have the same API, statistics database and profile fingerprint (`1e4cde06304f`); install one of them.

| Build | Branch | Tag | Runs on |
|---|---|---|---|
| PyTorch | `master` | `v1.0b` | CPU, or any CUDA GPU through PyTorch (Linux, Windows) |
| GPU (CuPy + Triton) | `master-triton` | `v1.0b-gpu` | Linux with an NVIDIA GPU, CUDA 12; no PyTorch needed |

## What is new since v1.0a

- **MACE-POLAR ligand embed refinement.** Every free-ligand embed of a call is refined in one batched MACE-POLAR-1-S optimisation (donor-donor distances held); GFN2 refinement remains the fallback.
- **Settle stage.** After the solve, ten batched MACE-POLAR steps with the declared donors held; rows are returned at the settled geometry and ordered by settled energy within each configuration class (`settle_steps`, `settle_top`). Settled rows are not converged minima.
- **Optional relaxation** of the returned rows with MACE-POLAR (`relax=`, `relax_top`, `relax_binding`, `relax_etol`).
- **Search and ranking.** Macrocycle opening, root and pool ranking, Li-cap spreading, closed-form and capped-group deduplication, cage stereo candidates.
- **Faster start.** The constraints database is decoded lazily per entry: load time 5.5 s to 0.44 s, memory about 2 GB to 0.45 GB.

## Results against v1.0a

Paired benchmark on 120 held-out calibration and tail complexes, three seeds each, one A100 per job (details in `PERFORMANCE.md`):

| | v1.0a | v1.0b PyTorch | v1.0b GPU |
|---|---|---|---|
| Complexes with every seed solved | 111 / 120 | 120 / 120 | 120 / 120 |
| Top-1 within 5 kJ/mol of the relaxed reference (MACE energy) | 0.05 | 0.88 | 0.89 |
| Reference basin missed after GFN2 relaxation | 0.225 | 0.102 | 0.120 |
| Reference basin is the top-1 row | 0.414 | 0.505 | 0.495 |
| Median wall time per complex | 10.6 s | 24.3 s | 3.3 s |

The GPU build is 6.3 times faster per complex than the PyTorch build (median) and matches it within the confidence intervals of the benchmark. The PyTorch build is slower than v1.0a because v1.0b adds the MACE refinement and settle stages.

## Install

```bash
# PyTorch build
git clone <repository-url> && cd jacobik && pixi install -e linux
# GPU build
git clone --branch master-triton <repository-url> && cd jacobik && pixi install -e linux
python -c "import jacobik; print(jacobik.__version__, jacobik.BUILD)"
```

`xtb` (6.7.1 or newer, below 7) must be on `PATH`. The MACE-POLAR-1-S weights are not part of the repository (Academic Software License, non-commercial): they are downloaded from the official `ACEsuit/mace-foundations` release into `~/.cache/mace` on first use; set `JACOBIK_MACE_MODEL_PATH` to use a local copy.

## Notes

- The GPU build has no CPU path: `device="cpu"` raises.
- The two builds agree up to floating-point order; individual rows can differ at the 1e-3 A level.
- Statistics database: `src/jacobik/data/constraints_2026-11-12.npz`, packaged and loaded automatically.
