# Project Handoff (Post-Restructure, October 2026)

## 1. Current State & Deliverables

The repository consolidation and multi-phase restructure is complete on branch `restructure`:
- **Clean Production Core (`src/jacobik/` & `src/fit/`):**
  - All 37 runtime flags resolved and hardcoded into production code.
  - Production configuration: `nci = False`, `deterministic_embed = True`, `ring_fold_hinges = False`, `q6_model = "repulsive"`, `solve_floor = "thermal"`, `score_agg = rank_agg = "sum"`.
  - Empty experiment hook in `src/jacobik/profile.py` with reproducible production fingerprint **`6cc586e17ab1`** (Li-cap fold-in, superseding Package F `615ae97d2350`).
  - Production database: lossless float64 binary container `data/db/constraints.npz` (24.25 MB, 81.5% size reduction) and `data/db/constraints.meta.json`.
  - Canonical fitting pipeline in `src/fit/` reproduces all database entries bit-identically from `data/ref/`.
- **User Interface (`interface/`):**
  - `interface/jacobik_cli.py`: Batch complex solver CLI.
  - `interface/prepare_data.py`: Structure validation and tmQM formatter.
  - `interface/run_solver.ipynb` & `interface/prepare_data.ipynb`: Interactive workflows (both verified headlessly with `jupyter nbconvert --execute`).
  - `interface/examples/`: Shipped example crystal complexes (`OJIMIF`, `CBTYCO`).
- **Research & Archive Separation (`research/`):**
  - Research interaction codes archived to `research/interactions/` (`nci_v2/`, `nci.py`, `q6pmf.py`, `bvs.py`).
  - Flag decisions and evidence ledger in `research/archive/flags/FLAGS.md`.
  - Data move manifest in `research/data/MOVE_MANIFEST.json` (103,310 files verified, 100% PASS).
  - Benchmarks, golden checks, and diagnostic tools in `research/bench/`.

---

## 2. Verification Status & Test Health

- **Golden Output Invariance Gate:** **PASS**
  - `python research/bench/tools/jdev/golden.py check` produces 0 differences across 245 arrays for both production and u1.
- **Production Test Suite:** **PASS**
  - `pytest tests/`: **317 passed, 3 xfailed (memory peak guards in sequential suite), 0 failed, 0 errors in 277s**.
- **Fast Path & Solver Replay:** **PASS**
  - CUDA-graph fast path and eager solver loop agree bit-for-bit (`tests/test_gpu_fast_path.py` 18/18 pass).
  - Solver replay generator committed at `src/jacobik/ik/build_replay_solver.py` and wrapped at `tools/jdev/build_replay_solver.py`.
- **Colab Cloud Environment:** **ACTIVE**
  - Verified active with `cj.py credits` (527.52 compute units remaining, 3.27 spent of 180 cap).

---

## 3. Active Technical Plan & Next Steps

All ongoing technical roadmap objectives are tracked in **`roadmap_generation_v1.md`** at the repository root:
1. **Generation & Solver Capabilities:** Follow the phased goals outlined in `roadmap_generation_v1.md`.
2. **Experimentation Workflow:** Future feature investigations must use the experiment hook `profile.py` (which modifies the fingerprint from `6cc586e17ab1`), maintaining strict gate checks (`golden.py check` and `pytest tests/`).
3. **Data Integrity Binding Rule:** Always maintain the strict 3-column reference data rule (only read `CSD_code`, `SMILES`, and geometry `xyz`/`bo` from `data/ref/`; never read the 8 forbidden QM property columns).
