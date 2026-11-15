# Archived Tests Catalog

This directory contains research, benchmark, and exploration test suites archived during the October 2026 repository restructure. The production test suite in `tests/` contains strictly tests for active production code.

## Archived Test Suites

### 1. NCI & Surface Calibration Research (`test_nci_*.py`)
- `test_nci_v2_atom_map.py`, `test_nci_v2_frame.py`, `test_nci_v2_numerics.py`, `test_nci_v2_orchestrator.py`, `test_nci_v2_pilot_cli.py`, `test_nci_v2_registry.py`, `test_nci_v2_remediation.py`, `test_nci_v2_s2b_state.py`: Codex NCI v2 Steps 3–4 refit pipeline tests. Archived alongside `research/interactions/nci_v2/` after finding $\gamma_{\text{shared}} = 0.0$.
- `test_nci_bound_candidates.py`, `test_nci_c2_surface.py`, `test_nci_components.py`, `test_nci_internal_experiment.py`, `test_nci_refit.py`: Exploratory tests for initial NCI spline development and sensitivity scans.

### 2. Diagnostics and Benchmark Tools (`jgen/`, `test_energy_*.py`)
- `jgen/test_bench_metrics.py`, `jgen/test_bench_simple_control.py`: Benchmarking tool validation. Moved to archive as part of separating benchmarks (`research/bench/`) from unit tests.
- `jgen/test_ring_fold_hinges.py`: Verification of historical ring-fold hinge behavior (hardcoded off in production).
- `jgen/test_profile_p0.py`: Verification of Phase 0 profile switch matrix.
- `test_energy_diagnosis.py`, `test_energy_viewer.py`: Interactive and diagnostic visualization tests.

### 3. Historical Objective Explorations
- `test_unified_objective_2026_10_16.py`: Verification of legacy objective switches and trial formulations leading up to the resolved `u1` profile.
- `test_db_prune_study.py`: Archived DB pruning study tests (`research/archive/code/refit/db_prune_study.py`).
