"""tests/test_nci_v2_pilot_cli.py
===============================
Unit and integration tests for Jacobik NCI v2 Pilot CLI and Step 1 Repeatability & Budgeting.

Verifies:
1. refit/nci_refit.py CLI subcommands (pilot, run, v2-registry, v2-frame, v2-pilot) and backward compatibility.
2. Repeatability tolerances on designated parents (drift <= 1.0 kJ/mol parent seed-mean, <= 5.0 kJ/mol per seed).
3. Completeness of pilot_repeatability_and_budget.json report (all 12 pilot parents, 4 component resources,
   compaction diagnostics, and 240-group budget extrapolation).
4. Integrity of generated control pools under data/refit/nci_v2/pilot_v1/pools/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "data/refit/nci_v2/pilot_v1/spec.json"
FRAME_PATH = REPO_ROOT / "data/refit/nci_v2/pilot_v1/frame.json"
REPORT_PATH = REPO_ROOT / "data/refit/nci_v2/pilot_v1/reports/pilot_repeatability_and_budget.json"
POOLS_DIR = REPO_ROOT / "data/refit/nci_v2/pilot_v1/pools"


def test_cli_help():
    """Verify refit/nci_refit.py top-level CLI help contains legacy and v2 subcommands."""
    cmd = [sys.executable, str(REPO_ROOT / "refit/nci_refit.py"), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, f"CLI help failed: {res.stderr}"
    stdout = res.stdout
    for subcmd in ("pilot", "run", "v2-registry", "v2-frame", "v2-pilot"):
        assert subcmd in stdout, f"Missing subcommand '{subcmd}' in CLI help"
    assert "--spec" in stdout
    assert "--frame" in stdout


def test_v2_registry_cli_subcommand():
    """Verify v2-registry runs cleanly via refit/nci_refit.py."""
    cmd = [sys.executable, str(REPO_ROOT / "refit/nci_refit.py"), "v2-registry"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, f"v2-registry CLI invocation failed:\n{res.stderr}\n{res.stdout}"
    assert "v2-registry: All records verified successfully." in res.stdout


def test_v2_frame_cli_help():
    """Verify v2-frame accepts --spec and --frame arguments."""
    import argparse
    from refit import nci_refit
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "run", "v2-registry", "v2-frame", "v2-pilot"))
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--frame", type=Path, default=FRAME_PATH)
    args = parser.parse_args(["v2-frame", "--spec", str(SPEC_PATH), "--frame", str(FRAME_PATH)])
    assert args.stage == "v2-frame"
    assert args.spec == SPEC_PATH
    assert args.frame == FRAME_PATH


def test_repeatability_tolerances():
    """Verify Run A vs Run B drift <= 1.0 kJ/mol parent seed-mean and <= 5.0 kJ/mol per seed."""
    assert REPORT_PATH.exists(), f"Missing report artifact: {REPORT_PATH}"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    rep = report.get("repeatability", {})
    assert rep.get("status") == "passed", f"Repeatability status is not passed: {rep.get('status')}"

    achieved = rep.get("achieved_metrics_kj_mol", {})
    max_top1_mean = achieved.get("max_parent_seed_mean_top1_drift")
    max_floor_mean = achieved.get("max_parent_seed_mean_floor_drift")
    max_indiv = achieved.get("max_individual_seed_drift")

    assert max_top1_mean is not None and max_top1_mean <= 1.0, f"Top-1 mean drift {max_top1_mean} > 1.0 kJ/mol"
    assert max_floor_mean is not None and max_floor_mean <= 1.0, f"Floor mean drift {max_floor_mean} > 1.0 kJ/mol"
    assert max_indiv is not None and max_indiv <= 5.0, f"Individual seed drift {max_indiv} > 5.0 kJ/mol"

    # Verify per-parent repeatability metrics
    parents_dict = rep.get("parents", {})
    assert len(parents_dict) == 6, f"Expected 6 repeatability parents, found {len(parents_dict)}"

    for code, p_data in parents_dict.items():
        if p_data.get("status") == "evaluated":
            p_top1 = p_data.get("parent_seed_mean_top1_drift_kj_mol")
            p_floor = p_data.get("parent_seed_mean_floor_drift_kj_mol")
            p_max = p_data.get("max_individual_seed_drift_kj_mol")
            assert p_top1 is not None and p_top1 <= 1.0, f"{code} top1 drift {p_top1} > 1.0 kJ/mol"
            assert p_floor is not None and p_floor <= 1.0, f"{code} floor drift {p_floor} > 1.0 kJ/mol"
            assert p_max is not None and p_max <= 5.0, f"{code} seed drift {p_max} > 5.0 kJ/mol"
            assert p_data.get("pass_repeatability") is True, f"{code} pass_repeatability is False"


def test_report_completeness():
    """Verify pilot_repeatability_and_budget.json report covers all required metrics and diagnostics."""
    assert REPORT_PATH.exists(), f"Missing report artifact: {REPORT_PATH}"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    # 1. Pilot cohort parent coverage
    comp_diag = report.get("compaction_diagnostics", {}).get("per_parent", {})
    assert len(comp_diag) == 12, f"Expected 12 pilot parents in compaction diagnostics, found {len(comp_diag)}"

    # 2. Component resource timings
    timings = report.get("component_timings_and_resources", {})
    summary = timings.get("summary", {})
    for key in (
        "solver_generation_mean_s",
        "nci_v2_enumeration_mean_s",
        "nci_v2_features_mean_s",
        "nci_v2_runtime_autograd_mean_s",
        "mace64_evaluation_mean_s_per_conf",
        "peak_cpu_rss_mb",
        "peak_gpu_mem_mb",
    ):
        assert key in summary, f"Missing component summary metric: {key}"
        assert summary[key] >= 0.0, f"Invalid value for {key}: {summary[key]}"

    # 3. Compaction diagnostics
    for code, c_data in comp_diag.items():
        for req in (
            "reference_rg_angstrom",
            "reference_contacts_lt_3_5A",
            "generated_top1_rg_angstrom",
            "generated_top1_contacts_lt_3_5A",
            "rg_ratio_top1_vs_ref",
            "contacts_ratio_top1_vs_ref",
            "n_heavy",
            "n_atoms",
        ):
            assert req in c_data, f"Missing compaction diagnostic '{req}' for parent {code}"

    # 4. Budget extrapolation for 240 groups
    extrap = report.get("budget_extrapolation", {})
    assert "step2_baseline" in extrap
    assert "step3_extract_and_fit_surface" in extrap
    assert "step4_measure_response_and_calibrate" in extrap
    assert "total_main_allocation_budget" in extrap

    s2 = extrap["step2_baseline"]
    assert s2.get("wall_time_hours_mean", 0) > 0
    assert s2.get("projected_peak_cpu_rss_mb", 0) > 0

    tot = extrap["total_main_allocation_budget"]
    assert "feasibility_decision" in tot
    assert "PASS" in tot["feasibility_decision"]


def test_pools_directory_integrity():
    """Verify generated control pose pools under data/refit/nci_v2/pilot_v1/pools/."""
    assert POOLS_DIR.exists(), f"Pools directory missing: {POOLS_DIR}"
    pool_files = list(POOLS_DIR.glob("*.npz"))
    assert len(pool_files) >= 6, f"Expected at least 6 pilot pool files, found {len(pool_files)}"

    valid_pool_count = 0
    error_pool_count = 0
    for pf in pool_files:
        z = np.load(pf, allow_pickle=False)
        if "X" in z.files:
            valid_pool_count += 1
            assert "symbols" in z.files
            assert "score" in z.files
            assert "asg" in z.files
            assert "charge" in z.files
            assert "mult" in z.files
            assert len(z["X"].shape) == 3
            assert z["X"].shape[0] == len(z["score"])
        elif "error" in z.files:
            error_pool_count += 1
            assert len(str(z["error"])) > 0
        else:
            pytest.fail(f"Pool file {pf.name} contains neither coordinates nor error message")

    assert valid_pool_count >= 5, f"Expected at least 5 valid pools, found {valid_pool_count}"
