"""Unit tests for the Jacobik NCI v2 Experiment Registry and Provenance Auditor.

Verifies:
1. Schema validity for all records in data/refit/experiments/index.json
2. Deterministic README.md rendering
3. Exact checksum accuracy for all physical evidence files
4. Explicit separation between coordinates generation and ranking scorers
5. Accurate recording of historical failure metrics (n2_followup, nci_ctx1, etc.)
6. Dependency audit completeness for tools/jdev/budget/
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from refit.nci_v2_registry import (
    VALID_MODES,
    VALID_STATUSES,
    build_historical_records,
    build_registry,
    merge_registry_records,
    render_markdown,
    sha256_of_file,
    validate_record,
    validate_registry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "data" / "refit" / "experiments"
INDEX_PATH = EXP_DIR / "index.json"
README_PATH = EXP_DIR / "README.md"
DEP_AUDIT_PATH = EXP_DIR / "dependency_audit.json"


@pytest.fixture(scope="module")
def registry_records():
    assert INDEX_PATH.exists(), f"Registry file missing at {INDEX_PATH}"
    with open(INDEX_PATH, "r", encoding="utf-8") as fp:
        records = json.load(fp)
    return records


def test_schema_validity(registry_records):
    """Verify that all records in index.json pass strict schema validation."""
    assert len(registry_records) >= 16, f"Expected at least 16 records, found {len(registry_records)}"
    errors = validate_registry(registry_records, repo_root=REPO_ROOT)
    assert not errors, f"Registry validation failed with errors:\n" + "\n".join(errors)


def test_mandatory_historical_records_present(registry_records):
    """Verify all 16 required historical records are present with valid modes and statuses."""
    record_map = {r["id"]: r for r in registry_records}

    expected_ids = {
        # 5 dev80 scoreboard records
        "dev80__gen-u1-contact__rank-contact__mace64__s11-13": ("fresh", "superseded"),
        "dev80__gen-u1-rep__rank-rep__mace64__s11-13": ("fresh", "superseded"),
        "dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13": ("fresh", "superseded"),
        "dev80__gen-os-pool__rank-pure-rep__mace64__s11-13": ("rerank", "completed"),
        "dev80__gen-os-pool__rank-nci-hb__mace64__s11-13": ("rerank", "completed"),
        # Archived multi-type fresh follow-up
        "n2_followup_2026-10-18": ("fresh", "failed"),
        # Calibration v2
        "n2_calibration_implementation_2026-10-18": ("dimer", "completed"),
        # Contextual HB fresh failure
        "nci_ctx1": ("fresh", "failed"),
        # Analytic HB chemical and complex motion transfer
        "nci_hb_physical_chemical_validation_2026-10-19": ("dimer", "completed"),
        "nci_hb_physical_complex_motion_2026-10-19": ("motion_scan", "failed"),
        "nci_hb_physical_latest_diagnosis_2026-10-19": ("curvature", "completed"),
        # Q6 runtime form and MACE transfer
        "q6_runtime_form_audit_2026-10-19": ("audit", "completed"),
        "q6_mace_transfer_summary_2026-10-19": ("audit", "completed"),
        # MACE input precision audit
        "mace_input_precision_2026-10-19": ("audit", "completed"),
        # Embedding generality and Cp validation
        "n2_cap_generality_2026-10-18": ("embedding", "failed"),
        "n2_embedding_validation_2026-10-18": ("embedding", "completed"),
    }

    for exp_id, (expected_mode, expected_status) in expected_ids.items():
        assert exp_id in record_map, f"Missing required record: '{exp_id}'"
        rec = record_map[exp_id]
        assert rec["mode"] == expected_mode, f"Record '{exp_id}': expected mode '{expected_mode}', got '{rec['mode']}'"
        assert rec["status"] == expected_status, f"Record '{exp_id}': expected status '{expected_status}', got '{rec['status']}'"
        assert rec["mode"] in VALID_MODES
        assert rec["status"] in VALID_STATUSES


def test_pilot_v1_record_present(registry_records):
    """Verify pilot_v1 repeatability record is present and valid."""
    record_map = {r["id"]: r for r in registry_records}
    exp_id = "nci_v2_pilot_v1__rep-u1-cold-cache__mace64__s11-13"
    assert exp_id in record_map, f"Missing pilot_v1 record: {exp_id}"
    rec = record_map[exp_id]
    assert rec["mode"] == "fresh"
    assert rec["status"] == "completed"
    assert rec["split"] == "pilot_cohort"
    assert "data/refit/nci_v2/pilot_v1/reports/pilot_repeatability_and_budget.json" in [a["path"] for a in rec["artifacts"]]


def test_generation_vs_ranking_separation(registry_records):
    """Enforce explicit distinction between coordinates generation and ranking scorers.

    Specifically:
    - pure-rep and u1+nci_hb rerank on etkdg_u1_os pools must NOT claim fresh generation.
    - valid pool floor must be identically 72.46 kJ/mol across both rerank records and etkdg_u1_os.
    - paired difference against pure-rep must equal -0.69 kJ/mol, CI [-1.78, +0.01].
    """
    record_map = {r["id"]: r for r in registry_records}

    rec_os = record_map["dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13"]
    rec_pure_rep = record_map["dev80__gen-os-pool__rank-pure-rep__mace64__s11-13"]
    rec_hb_nci = record_map["dev80__gen-os-pool__rank-nci-hb__mace64__s11-13"]

    # Check modes
    assert rec_os["mode"] == "fresh"
    assert rec_pure_rep["mode"] == "rerank"
    assert rec_hb_nci["mode"] == "rerank"

    # Coordinates provenance
    assert rec_pure_rep["ranking"]["source_pool_ids"] == ["dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13"]
    assert rec_hb_nci["ranking"]["source_pool_ids"] == ["dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13"]

    # Scorer separation
    assert rec_pure_rep["ranking"]["full_profile"]["nci_types"] == []
    assert rec_hb_nci["ranking"]["full_profile"]["nci_types"] == ["hb"]

    # Pool floor must be identical because coordinates are identical
    floor_os = rec_os["metrics"]["values"]["floor_mean"]
    floor_pure_rep = rec_pure_rep["metrics"]["values"]["floor_mean"]
    floor_hb = rec_hb_nci["metrics"]["values"]["floor_mean"]

    assert pytest.approx(floor_pure_rep, rel=1e-4) == 72.4605
    assert pytest.approx(floor_hb, rel=1e-4) == 72.4605
    assert pytest.approx(floor_os, rel=1e-4) == 72.4605

    # Top-1 means and paired metrics
    assert pytest.approx(rec_pure_rep["metrics"]["values"]["pick_mean"], abs=0.01) == 87.03
    assert pytest.approx(rec_hb_nci["metrics"]["values"]["pick_mean"], abs=0.01) == 86.34

    unc = rec_hb_nci["metrics"]["uncertainty"]
    assert pytest.approx(unc["paired_mean_diff_vs_pure_rep"], abs=0.01) == -0.69
    assert unc["ci_95"] == [-1.78, 0.01]


def test_deterministic_readme_generation(registry_records):
    """Verify that README.md matches deterministic rendering from index.json."""
    assert README_PATH.exists(), f"README.md missing at {README_PATH}"
    with open(README_PATH, "r", encoding="utf-8") as fp:
        existing_md = fp.read()

    rendered_md = render_markdown(registry_records)
    assert existing_md == rendered_md, "README.md does not match deterministic render_markdown output"


def test_physical_evidence_checksum_accuracy(registry_records):
    """Verify checksum accuracy against existing physical evidence files on disk."""
    expected_evidence_hashes = {
        "data/refit/evidence/n1_n2_baseline_reconciliation_2026-10-18.json": "50eb2965de5e6797814d726aba2ddfd3f6f9cc6f9bd9d8f8c7cc90e12fdb9725",
        "data/refit/evidence/nci_dev80_benchmark_2026-10-18.json": "16ea492c3976af1a8bddecaf4e0234978f86e99f484224de32e2150bb0c28b7d",
        "data/refit/evidence/mace_input_precision_2026-10-19.json": "3907af7e90213ac5b125d72fabcd7c47a35cd3744748d4e10ac0692339167bb6",
        "data/refit/evidence/q6_runtime_form_audit_2026-10-19.json": "4c816b318bdcb4b5d70504987f343106facf4114540b0a4aa398da4f4bdad6b8",
        "data/refit/evidence/q6_mace_transfer_summary_2026-10-19.json": "a06d01b1bfef3a59e2fa4e8444ea3825baef9e18e6325f40e4c250da9d4edb01",
        "data/refit/evidence/nci_hb_physical_chemical_validation_2026-10-19.json": "5f74fd26ea8dd85fa4d30a61eab09ceb363e33800bbc5eec7d72dc6bfd6e6640",
        "data/refit/evidence/nci_hb_physical_complex_motion_2026-10-19.json": "6fcd8cff3663f666ecf4c893f4eca5d8cb880d85d529f7092ab69f09f9328aad",
        "data/refit/evidence/nci_hb_physical_latest_diagnosis_2026-10-19.md": "ecd43aaaa781e34dfd696053e854b6425ae1bf1d16bcac7c21f21177180e1f14",
        "data/refit/final/constraints_2026-10-18.json": "48ac82b56d71fd6e43515ecf830d4712a71de3df533b4d6ff8762f7ebe7549bf",
    }

    for rel_path, expected_sha in expected_evidence_hashes.items():
        full_p = REPO_ROOT / rel_path
        assert full_p.is_file(), f"Evidence file not found: {full_p}"
        actual_sha = sha256_of_file(full_p)
        assert actual_sha == expected_sha, f"Hash mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"


def test_historical_failure_and_audit_metrics(registry_records):
    """Verify exact metrics for multi-type fresh follow-up, contextual-HB, and DOVYES decomposition."""
    record_map = {r["id"]: r for r in registry_records}

    # 1. n2_followup_2026-10-18
    f_rec = record_map["n2_followup_2026-10-18"]
    f_vals = f_rec["metrics"]["values"]
    assert pytest.approx(f_vals["PEWBOL_mean_change"], abs=0.01) == 87.54
    assert pytest.approx(f_vals["FETSUU_mean_change"], abs=0.01) == 26.78
    assert pytest.approx(f_vals["pewbol_seed11_control_drift"], abs=0.01) == 89.13

    # 2. nci_ctx1 contextual-HB failure
    ctx_rec = record_map["nci_ctx1"]
    ctx_vals = ctx_rec["metrics"]["values"]
    assert pytest.approx(ctx_vals["hb_mean_change"], abs=0.01) == -0.16
    assert pytest.approx(ctx_vals["hb_mean_floor_change"], abs=0.01) == 9.26
    # /tmp output must be marked unavailable
    tmp_art = [a for a in ctx_rec["artifacts"] if a["path"].startswith("/tmp")]
    assert tmp_art, "Missing /tmp artifact in nci_ctx1"
    assert tmp_art[0]["availability"] == "unavailable"

    # 3. DOVYES derivative decomposition
    dov_rec = record_map["nci_hb_physical_latest_diagnosis_2026-10-19"]
    dov_vals = dov_rec["metrics"]["values"]
    assert pytest.approx(dov_vals["DOVYES_selected_pair"], abs=0.01) == -87.136
    assert pytest.approx(dov_vals["DOVYES_other_pairs"], abs=0.01) == -23.012
    assert pytest.approx(dov_vals["DOVYES_environment_residual"], abs=0.01) == 89.998
    assert pytest.approx(dov_vals["DOVYES_whole_mace"], abs=0.01) == -20.150
    assert pytest.approx(dov_vals["DOVYES_whole_analytic"], abs=0.01) == -239.507


def test_dependency_audit_completeness():
    """Verify that dependency_audit.json contains all 68 budget scripts and classifies them."""
    assert DEP_AUDIT_PATH.exists(), f"Dependency audit file missing at {DEP_AUDIT_PATH}"
    with open(DEP_AUDIT_PATH, "r", encoding="utf-8") as fp:
        audit = json.load(fp)

    metadata = audit["metadata"]
    assert metadata["total_budget_files"] == 68
    assert metadata["total_nci_files"] == 38

    scripts = audit["scripts"]
    assert len(scripts) == 68

    # Production dependencies must be preserved
    prod_scripts = audit["summary"]["production_dependent_scripts"]
    assert "common.py" in prod_scripts
    assert "pools.py" in prod_scripts
    assert "gate.py" in prod_scripts

    # Test dependencies must be preserved
    test_scripts = audit["summary"]["test_dependent_scripts"]
    assert "nci_c2_surface.py" in test_scripts
    assert "nci_surface_ownership_audit.py" in test_scripts


def test_build_keeps_extra_appended_record(tmp_path):
    """Verify that building over an existing index.json that holds an extra appended record keeps it."""
    tmp_index = tmp_path / "index.json"
    historical = build_historical_records()

    extra_record = {
        "id": "custom_appended_experiment_001",
        "readable_name": "extra_record_001",
        "mode": "fresh",
        "status": "completed",
        "question": "Does appended record survive build?",
        "parent_record_ids": [],
        "cohort_manifest": "data/refit/experiments/index.json",
        "split": "test_split",
        "selection_rule": "test rule",
        "exclusions": [],
        "seeds": [1, 2],
        "generation": {
            "code_hash": "dummy_gen_code",
            "db_hash": "dummy_gen_db",
            "full_profile": {},
            "settings": {},
            "environment": {},
        },
        "ranking": {
            "code_hash": "dummy_rank_code",
            "db_hash": "dummy_rank_db",
            "full_profile": {},
            "source_pool_ids": [],
        },
        "evaluation": {
            "model_hash": "dummy_model",
            "evaluator_hash": "dummy_eval",
            "dtype": "float64",
            "units": "kJ/mol",
            "state_reference_hashes": {},
        },
        "artifacts": [],
        "metrics": {
            "definitions": {},
            "aggregation_unit": "complex",
            "paired_comparator": None,
            "values": {"test_metric": 42.0},
            "uncertainty": None,
        },
        "conclusion": "Kept cleanly.",
        "limitations": "None.",
        "superseded_by": None,
    }

    initial_records = historical + [extra_record]
    with open(tmp_index, "w", encoding="utf-8") as fp:
        json.dump(initial_records, fp, indent=2)

    merged = build_registry(index_path=tmp_index, repo_root=REPO_ROOT, dry_run=False)

    assert len(merged) == len(historical) + 1
    assert merged[-1]["id"] == "custom_appended_experiment_001"
    assert merged[-1]["question"] == "Does appended record survive build?"
    assert merged[-1]["metrics"]["values"]["test_metric"] == 42.0

    # Ensure file on disk matches
    with open(tmp_index, "r", encoding="utf-8") as fp:
        on_disk = json.load(fp)
    assert on_disk == merged


def test_build_conflicting_hard_coded_record_raises(tmp_path):
    """Verify that a conflicting hard-coded record in the existing file raises ValueError and lists differing fields."""
    tmp_index = tmp_path / "index.json"
    historical = build_historical_records()

    # Tamper with the first historical record's question and status
    tampered = [dict(r) for r in historical]
    tampered[0] = dict(tampered[0])
    tampered[0]["question"] = "Conflicting altered question"
    tampered[0]["status"] = "active"

    with open(tmp_index, "w", encoding="utf-8") as fp:
        json.dump(tampered, fp, indent=2)

    with pytest.raises(ValueError) as excinfo:
        build_registry(index_path=tmp_index, repo_root=REPO_ROOT, dry_run=False)

    msg = str(excinfo.value)
    assert tampered[0]["id"] in msg
    assert "question" in msg
    assert "status" in msg

    # File on disk must remain unchanged (not overwritten)
    with open(tmp_index, "r", encoding="utf-8") as fp:
        disk_records = json.load(fp)
    assert disk_records == tampered


def test_build_amended_by_survives(tmp_path):
    """Verify that added fields such as 'amended_by' on existing records survive merging."""
    tmp_index = tmp_path / "index.json"
    historical = build_historical_records()

    # Add amended_by to record 19 (step2_control) and append repair record
    modified = [dict(r) for r in historical]
    target_id = "nci_v2_scaled_v1__step2_control__s11-13"
    amendment_id = "nci_v2_scaled_v1__s2_1_repair__amends_step2_control"

    found = False
    for i, r in enumerate(modified):
        if r["id"] == target_id:
            modified[i] = dict(r)
            modified[i]["amended_by"] = [amendment_id]
            found = True
            break
    assert found, f"Target record {target_id} not found in historical records"

    # Add dummy repair record so reference integrity is maintained
    repair_record = {
        "id": amendment_id,
        "readable_name": "s2_1_repair",
        "mode": "audit",
        "status": "completed",
        "question": "Repair audit",
        "parent_record_ids": [target_id],
        "cohort_manifest": "data/refit/experiments/index.json",
        "split": "test",
        "selection_rule": "test",
        "exclusions": [],
        "seeds": [],
        "generation": {
            "code_hash": "dummy_gen_code",
            "db_hash": "dummy_gen_db",
            "full_profile": {},
            "settings": {},
            "environment": {},
        },
        "ranking": {
            "code_hash": "dummy_rank_code",
            "db_hash": "dummy_rank_db",
            "full_profile": {},
            "source_pool_ids": [],
        },
        "evaluation": {
            "model_hash": "dummy_model",
            "evaluator_hash": "dummy_eval",
            "dtype": "float64",
            "units": "kJ/mol",
            "state_reference_hashes": {},
        },
        "artifacts": [],
        "metrics": {
            "definitions": {},
            "aggregation_unit": "complex",
            "paired_comparator": None,
            "values": {},
            "uncertainty": None,
        },
        "conclusion": "Repaired.",
        "limitations": "None.",
        "superseded_by": None,
    }
    modified.append(repair_record)

    with open(tmp_index, "w", encoding="utf-8") as fp:
        json.dump(modified, fp, indent=2)

    merged = build_registry(index_path=tmp_index, repo_root=REPO_ROOT, dry_run=False)

    merged_by_id = {r["id"]: r for r in merged}
    assert target_id in merged_by_id
    assert merged_by_id[target_id].get("amended_by") == [amendment_id]
    assert amendment_id in merged_by_id


def test_build_dry_run_writes_nothing(tmp_path, capsys):
    """Verify that --dry-run prints status and writes nothing to disk."""
    # Scenario A: index file does not exist yet
    nonexistent = tmp_path / "nonexistent_index.json"
    records = build_registry(index_path=nonexistent, repo_root=REPO_ROOT, dry_run=True)
    assert not nonexistent.exists(), "Dry-run should not create nonexistent file"
    assert len(records) > 0

    # Scenario B: existing file with custom content
    existing_file = tmp_path / "existing_index.json"
    initial_content = json.dumps([{"id": "dummy_test", "custom": 123}], indent=2)
    existing_file.write_text(initial_content, encoding="utf-8")

    mtime_before = existing_file.stat().st_mtime_ns

    build_registry(index_path=existing_file, repo_root=REPO_ROOT, dry_run=True)

    mtime_after = existing_file.stat().st_mtime_ns
    assert mtime_before == mtime_after, "File mtime changed during dry-run"
    assert existing_file.read_text(encoding="utf-8") == initial_content, "File content modified during dry-run"

    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "records kept and" in out
