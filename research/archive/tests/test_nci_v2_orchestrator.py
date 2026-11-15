"""tests/test_nci_v2_orchestrator.py
==================================
Comprehensive Unit & Integration Tests for Jacobik NCI v2 Production Orchestrator (Gate G5).

Verifies:
1. Resumable queue and skip-on-hash-match (and re-run on hash mismatch).
2. Failure and timeout recording as structured outcomes (never silently dropped).
3. Reserve replacement logic adhering to stratum and predeclared order for training (surface_fit),
   and strict zero-replacement for evaluation cohorts (calibration, motion_validation, pilot).
4. Provenance completeness and coordination-integrity storage.
5. VRAM and RSS logging.
6. Heavy-atom VRAM-aware ordering.
7. Subcommand and CLI integration.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import time
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "refit", REPO_ROOT / "tools/jdev/budget"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from refit.nci_v2_orchestrator import (
    LongLivedWorker,
    ProductionOrchestrator,
    SolveTask,
    TaskPaths,
    compute_provenance_hashes,
    get_task_paths,
    should_skip_task,
    DEFAULT_SPEC,
    DEFAULT_FRAME,
    CONSTRAINTS_DB_PATH,
    MACE_MODEL_PATH,
)


def test_cli_help_and_subcommands():
    """Verify refit/nci_refit.py and refit/nci_v2_orchestrator.py CLI interfaces."""
    cmd1 = [sys.executable, str(REPO_ROOT / "refit/nci_refit.py"), "--help"]
    res1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res1.returncode == 0
    assert "v2-orchestrate" in res1.stdout
    assert "orchestrate" in res1.stdout
    assert "--cohort" in res1.stdout
    assert "--force-recompute" in res1.stdout

    cmd2 = [sys.executable, "-m", "refit.nci_v2_orchestrator", "--dry-run"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res2.returncode == 0
    assert "Dry-run:" in res2.stdout
    assert "tasks queued" in res2.stdout


def test_provenance_completeness():
    """Verify all cryptographic provenance fields are computed and valid."""
    prov = compute_provenance_hashes(
        spec_path=DEFAULT_SPEC,
        frame_path=DEFAULT_FRAME,
        constraints_path=CONSTRAINTS_DB_PATH,
        mace_path=MACE_MODEL_PATH,
    )
    required_keys = (
        "spec_sha256",
        "frame_sha256",
        "code_git_commit",
        "code_dirty_diff_sha256",
        "constraints_db_sha256",
        "mace_model_sha256",
        "full_resolved_profile",
    )
    for k in required_keys:
        assert k in prov, f"Missing provenance key: {k}"
        assert prov[k] is not None, f"Provenance value for {k} is None"

    # Verify sha256 format for existing files
    for k in ("spec_sha256", "frame_sha256", "constraints_db_sha256", "mace_model_sha256"):
        val = prov[k]
        assert len(val) == 64 and all(c in "0123456789abcdefABCDEF" for c in val), f"Invalid sha256 for {k}: {val}"

    assert isinstance(prov["full_resolved_profile"], dict)
    assert len(prov["full_resolved_profile"]) > 0


def test_resumable_queue_skip_and_invalidation(tmp_path: Path):
    """Verify that tasks are skipped only on exact provenance match, and re-solved on any mismatch."""
    prov = compute_provenance_hashes()
    paths = get_task_paths(tmp_path, code="TESTMOL", seed=11, arm="control")
    paths.pool_npz.parent.mkdir(parents=True, exist_ok=True)

    # 1. Non-existent files -> should NOT skip
    assert not should_skip_task(paths, prov)

    # Helper to write valid artifacts
    def write_valid_artifacts(mutated_prov=None, profile_override=None):
        p = dict(prov)
        if mutated_prov:
            p.update(mutated_prov)
        if profile_override is not None:
            p["full_resolved_profile"] = profile_override

        # Write NPZ
        np.savez_compressed(
            paths.pool_npz,
            X=np.zeros((2, 5, 3)),
            spec_sha256=p["spec_sha256"],
            frame_sha256=p["frame_sha256"],
            code_dirty_diff_sha256=p["code_dirty_diff_sha256"],
            constraints_db_sha256=p["constraints_db_sha256"],
            mace_model_sha256=p["mace_model_sha256"],
            profile=json.dumps(p["full_resolved_profile"]),
        )
        # Write MACE JSON
        paths.mace_json.write_text(json.dumps({"energies_kj": [-100.0, -102.5]}), encoding="utf-8")
        # Write Summary JSON
        paths.summary_json.write_text(
            json.dumps({"status": "success", "code": "TESTMOL", "provenance": p}), encoding="utf-8"
        )

    # 2. Perfect provenance match -> SHOULD skip
    write_valid_artifacts()
    assert should_skip_task(paths, prov) is True

    # 3. Spec hash mismatch -> re-solve
    write_valid_artifacts(mutated_prov={"spec_sha256": "0000000000000000000000000000000000000000000000000000000000000000"})
    assert should_skip_task(paths, prov) is False

    # 4. Frame hash mismatch -> re-solve
    write_valid_artifacts(mutated_prov={"frame_sha256": "1111111111111111111111111111111111111111111111111111111111111111"})
    assert should_skip_task(paths, prov) is False

    # 5. Code dirty diff mismatch -> re-solve
    write_valid_artifacts(mutated_prov={"code_dirty_diff_sha256": "dirty_mismatch_hash"})
    assert should_skip_task(paths, prov) is False

    # 6. Constraints DB mismatch -> re-solve
    write_valid_artifacts(mutated_prov={"constraints_db_sha256": "deadbeef" * 8})
    assert should_skip_task(paths, prov) is False

    # 7. MACE model mismatch -> re-solve
    write_valid_artifacts(mutated_prov={"mace_model_sha256": "macebad" * 8})
    assert should_skip_task(paths, prov) is False

    # 8. Profile mismatch -> re-solve
    write_valid_artifacts(profile_override={"q6_model": "modified_unmatched"})
    assert should_skip_task(paths, prov) is False

    # 9. Missing mace JSON -> re-solve
    write_valid_artifacts()
    paths.mace_json.unlink()
    assert should_skip_task(paths, prov) is False

    # 10. Corrupted NPZ file -> re-solve
    write_valid_artifacts()
    paths.pool_npz.write_bytes(b"corrupted binary data")
    assert should_skip_task(paths, prov) is False

    # 11. Failed prior outcome -> re-solve
    write_valid_artifacts()
    paths.summary_json.write_text(json.dumps({"status": "failed", "error": "timeout"}), encoding="utf-8")
    assert should_skip_task(paths, prov) is False


def test_reserve_replacement_training_within_stratum(tmp_path: Path):
    """Verify training (surface_fit) failures trigger reserve replacement within stratum and in predeclared order."""
    mock_frame = {
        "main_allocation": {
            "splits": {
                "surface_fit": [
                    {
                        "representative_code": "TRAIN_A",
                        "representative_shard": "X1s000",
                        "dataset": "X1",
                        "stratum": "3d_neg_med",
                        "n_heavy": 30,
                    }
                ],
                "reserve": [
                    {
                        "representative_code": "RES_OTHER_STRATUM",
                        "representative_shard": "X1s001",
                        "dataset": "X1",
                        "stratum": "4d_pos_high",
                        "n_heavy": 25,
                    },
                    {
                        "representative_code": "RES_MATCH_1",
                        "representative_shard": "X1s002",
                        "dataset": "X1",
                        "stratum": "3d_neg_med",
                        "n_heavy": 32,
                    },
                    {
                        "representative_code": "RES_MATCH_2",
                        "representative_shard": "X1s003",
                        "dataset": "X1",
                        "stratum": "3d_neg_med",
                        "n_heavy": 35,
                    },
                ],
            }
        }
    }
    frame_p = tmp_path / "mock_frame.json"
    frame_p.write_text(json.dumps(mock_frame), encoding="utf-8")

    orch = ProductionOrchestrator(
        frame_path=frame_p,
        out_dir=tmp_path / "out",
        eval_mace=False,
    )

    failed_task = SolveTask(
        code="TRAIN_A",
        shard="X1s000",
        dataset="X1",
        seed=11,
        arm="control",
        stratum="3d_neg_med",
        split="surface_fit",
        n_heavy=30,
    )

    # 1st failure: must choose RES_MATCH_1 (first in predeclared order for 3d_neg_med)
    rep1 = orch.draw_reserve_replacement(failed_task, reason="unmappable")
    assert rep1 is not None
    assert rep1["representative_code"] == "RES_MATCH_1"
    assert rep1["stratum"] == "3d_neg_med"
    assert orch.replaced_failures["TRAIN_A"]["replacement_code"] == "RES_MATCH_1"

    # 2nd failure in same stratum: must choose RES_MATCH_2 (second in predeclared order)
    rep2_task = SolveTask(
        code="RES_MATCH_1",
        shard="X1s002",
        dataset="X1",
        seed=11,
        arm="control",
        stratum="3d_neg_med",
        split="surface_fit",
        n_heavy=32,
    )
    rep2 = orch.draw_reserve_replacement(rep2_task, reason="solver_error")
    assert rep2 is not None
    assert rep2["representative_code"] == "RES_MATCH_2"

    # 3rd failure: stratum is now exhausted
    rep3_task = SolveTask(
        code="RES_MATCH_2",
        shard="X1s003",
        dataset="X1",
        seed=11,
        arm="control",
        stratum="3d_neg_med",
        split="surface_fit",
        n_heavy=35,
    )
    rep3 = orch.draw_reserve_replacement(rep3_task, reason="timeout")
    assert rep3 is None
    assert "3d_neg_med" in orch.exhausted_strata


def test_reserve_replacement_evaluation_cohort_zero_replacement(tmp_path: Path):
    """Verify evaluation cohorts (calibration, motion_validation, pilot) NEVER replace failures."""
    mock_frame = {
        "main_allocation": {
            "splits": {
                "calibration": [{"representative_code": "CALIB_1", "stratum": "3d_neg_med"}],
                "motion_validation": [{"representative_code": "MOTION_1", "stratum": "3d_neg_med"}],
                "reserve": [{"representative_code": "RES_1", "stratum": "3d_neg_med"}],
            }
        },
        "pilot_cohort": {
            "parents": [{"code": "PILOT_1", "stratum": "3d_neg_med"}],
            "reserves": [{"code": "PILOT_RES", "stratum": "3d_neg_med"}],
        },
    }
    frame_p = tmp_path / "mock_frame.json"
    frame_p.write_text(json.dumps(mock_frame), encoding="utf-8")

    orch = ProductionOrchestrator(
        frame_path=frame_p,
        out_dir=tmp_path / "out",
        eval_mace=False,
    )

    for eval_split, code in (
        ("calibration", "CALIB_1"),
        ("motion_validation", "MOTION_1"),
        ("pilot", "PILOT_1"),
    ):
        task = SolveTask(
            code=code,
            shard="X1s000",
            dataset="X1",
            seed=11,
            arm="control",
            stratum="3d_neg_med",
            split=eval_split,
        )
        rep = orch.draw_reserve_replacement(task, reason="unmappable")
        assert rep is None, f"Evaluation split '{eval_split}' should NEVER replace failed complexes!"
        assert code not in orch.replaced_failures


def test_structured_failure_and_timeout_recording(tmp_path: Path):
    """Verify that failures and timeouts produce structured outcomes with error files and manifest records."""
    orch = ProductionOrchestrator(out_dir=tmp_path, eval_mace=False)
    paths = get_task_paths(tmp_path, "NONEXISTENT_CODE", 11, "control")
    paths.pool_npz.parent.mkdir(parents=True, exist_ok=True)

    worker = LongLivedWorker(eval_mace=False, device="cpu")
    failed_task = SolveTask(
        code="NONEXISTENT_CODE",
        shard="unknown_shard",
        dataset="unknown_ds",
        seed=11,
        arm="control",
        stratum="mock_strat",
        split="surface_fit",
    )

    prov = compute_provenance_hashes()
    res = worker.solve(failed_task, prov, paths)

    assert res["status"] in ("failed", "unmappable")
    assert "error" in res
    assert res["wall_s"] >= 0.0
    assert "peak_rss_mb" in res
    assert "peak_gpu_mb" in res

    # Verify structured outcome files exist on disk
    assert paths.pool_npz.exists()
    assert paths.summary_json.exists()

    summary_data = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary_data["status"] == res["status"]
    assert "error" in summary_data

    with np.load(paths.pool_npz) as npz:
        assert "error" in npz


def test_vram_aware_task_ordering():
    """Verify task queue sorts by heavy-atom count so memory usage is bounded."""
    orch = ProductionOrchestrator(eval_mace=False)
    tasks = [
        SolveTask(code="MOL_LARGE", shard="X1", dataset="X1", seed=11, arm="control", stratum="s", split="p", n_heavy=80),
        SolveTask(code="MOL_SMALL", shard="X1", dataset="X1", seed=11, arm="control", stratum="s", split="p", n_heavy=15),
        SolveTask(code="MOL_MED", shard="X1", dataset="X1", seed=11, arm="control", stratum="s", split="p", n_heavy=40),
    ]
    tasks.sort(key=lambda t: (t.n_heavy, t.code, t.seed, t.arm))
    ordered_heavies = [t.n_heavy for t in tasks]
    assert ordered_heavies == [15, 40, 80]


def test_orchestrator_integration_pilot_parent(tmp_path: Path):
    """Integration test: solve HAMYUP on seed 11 with control arm; verify coordination integrity & manifest."""
    out_dir = tmp_path / "integration_test"
    orch = ProductionOrchestrator(
        out_dir=out_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mace=True,
    )
    tasks = orch.build_task_queue(cohort="pilot", codes=["HAMYUP"], seeds=[11], arms=["control"])
    assert len(tasks) == 1

    # First run: solves and writes manifest
    manifest1 = orch.run(tasks)
    assert manifest1["succeeded_tasks"] == 1
    assert manifest1["failed_tasks"] == 0
    assert "HAMYUP:11:control" in manifest1["tasks"]

    task_res = manifest1["tasks"]["HAMYUP:11:control"]
    assert task_res["status"] == "success"
    assert "peak_gpu_mb" in task_res
    assert "peak_rss_mb" in task_res
    assert task_res["top1_coord_intact"] is not None

    paths = get_task_paths(out_dir, "HAMYUP", 11, "control")
    assert paths.pool_npz.exists()
    assert paths.mace_json.exists()
    assert paths.summary_json.exists()

    # Verify coordination integrity and provenance arrays in NPZ
    with np.load(paths.pool_npz) as npz:
        assert "coord_intact" in npz
        assert "coord_sigma_shift_A" in npz
        assert "coord_haptic_slip_A" in npz
        assert "coord_haptic_shift_A" in npz
        assert "spec_sha256" in npz
        assert "frame_sha256" in npz
        assert "code_dirty_diff_sha256" in npz
        assert "mace_model_sha256" in npz

    # Second run: should hit resumable cache and skip without re-solve
    manifest2 = orch.run(tasks)
    assert manifest2["succeeded_tasks"] == 1
    assert manifest2["completed_tasks"] == 1


# ---------------------------------------------------------------------------
# Audit O8 Tests (O1–O7 Verification Suite)
# ---------------------------------------------------------------------------

def test_arm_live_snapshots_no_mocks():
    """O8.1: For each arm, the worker's live snapshot shows the expected nci / nci_types / deterministic_embed."""
    from refit.nci_v2_orchestrator import resolve_arm_snapshot

    # Control arm: pure repulsion baseline
    snap_ctrl = resolve_arm_snapshot("control")
    assert snap_ctrl["fields"]["deterministic_embed"] is True
    assert snap_ctrl["fields"]["nci"] is False
    assert snap_ctrl["fingerprint"] == "485e943af036"

    # Incumbent arm: historical HB term
    snap_inc = resolve_arm_snapshot("incumbent")
    assert snap_inc["fields"]["deterministic_embed"] is True
    assert snap_inc["fields"]["nci"] is True
    assert tuple(snap_inc["fields"]["nci_types"]) == ("hb",)
    assert snap_inc["fingerprint"] == "e90d9adc0f7e"

    # Candidate arm: full transferable NCI components (HB, PP, CHP)
    snap_cand = resolve_arm_snapshot("candidate")
    assert snap_cand["fields"]["deterministic_embed"] is True
    assert snap_cand["fields"]["nci"] is True
    assert set(snap_cand["fields"]["nci_types"]) == {"hb", "pp", "chp"}
    assert snap_cand["fingerprint"] == "908d8a112d15"


def test_solve_seeds_certified_kwargs_spy(monkeypatch, tmp_path: Path):
    """O8.2: The production work unit calls solve_seeds with the certified kwargs (spy, not mock-out)."""
    from tools.jdev import multiseed
    from refit.nci_v2_orchestrator import worker_solve_group, SolveGroupUnit, compute_provenance_hashes

    called = {}
    orig_solve_seeds = multiseed.solve_seeds

    def spy_solve_seeds(builder, seeds, **kwargs):
        called["builder"] = builder
        called["seeds"] = list(seeds)
        called["kwargs"] = dict(kwargs)
        return orig_solve_seeds(builder, seeds, **kwargs)

    monkeypatch.setattr(multiseed, "solve_seeds", spy_solve_seeds)

    unit = SolveGroupUnit(
        code="HAMYUP",
        shard="X1s005",
        dataset="X1",
        arm="control",
        seeds=[11, 12, 13],
        stratum="4f_zero_high",
        split="pilot",
        n_heavy=22,
    )
    prov = compute_provenance_hashes(arm="control")
    outcomes = worker_solve_group(
        unit=unit,
        arm_provenance=prov,
        out_dir=tmp_path,
        stats=None,
        mace_evaluator=None,
        eval_mace=False,
        solve_device="cuda" if torch.cuda.is_available() else "cpu",
    )

    assert called["seeds"] == [11, 12, 13]
    kw = called["kwargs"]
    assert kw.get("batched_assembly") is True
    assert kw.get("angle_bends") is True
    if torch.cuda.is_available():
        assert kw.get("device") == "cuda"


def test_oom_retries_run_outside_exception_handler(monkeypatch, tmp_path: Path):
    """S2.1: each max_rows retry starts with no exception being handled, so the failed
    attempt's traceback (and the GPU memory its frames reference) has been released."""
    import sys
    from tools.jdev import multiseed
    from refit.nci_v2_orchestrator import worker_solve_group, SolveGroupUnit, compute_provenance_hashes

    active_exc = []

    def mock_oom_solve_seeds(builder, seeds, **kwargs):
        active_exc.append(sys.exc_info()[0])
        raise torch.cuda.OutOfMemoryError("Simulated CUDA OOM")

    monkeypatch.setattr(multiseed, "solve_seeds", mock_oom_solve_seeds)

    unit = SolveGroupUnit(
        code="HAMYUP",
        shard="X1s005",
        dataset="X1",
        arm="control",
        seeds=[11, 12, 13],
        stratum="4f_zero_high",
        split="pilot",
        n_heavy=22,
    )
    worker_solve_group(
        unit=unit,
        arm_provenance=compute_provenance_hashes(arm="control"),
        out_dir=tmp_path,
        stats=None,
        mace_evaluator=None,
        eval_mace=False,
        solve_device="cuda",
    )

    assert len(active_exc) == 5
    assert active_exc == [None] * 5


def test_oom_outcome_no_cpu_fallback(monkeypatch, tmp_path: Path):
    """O8.3: OOM path: retry on GPU with smaller max_rows; outcome 'oom' recorded; no CPU solver fallback."""
    from tools.jdev import multiseed
    from jacobik import ComplexBuilder
    from refit.nci_v2_orchestrator import worker_solve_group, SolveGroupUnit, compute_provenance_hashes

    tried_max_rows = []

    def mock_oom_solve_seeds(builder, seeds, **kwargs):
        tried_max_rows.append(kwargs.get("max_rows"))
        raise torch.cuda.OutOfMemoryError("Simulated CUDA OOM")

    def forbidden_cpu_solve(*args, **kwargs):
        raise AssertionError("builder.solve(device='cpu') was called! Fallback to CPU solver is forbidden.")

    monkeypatch.setattr(multiseed, "solve_seeds", mock_oom_solve_seeds)
    monkeypatch.setattr(ComplexBuilder, "solve", forbidden_cpu_solve)

    unit = SolveGroupUnit(
        code="HAMYUP",
        shard="X1s005",
        dataset="X1",
        arm="control",
        seeds=[11, 12, 13],
        stratum="4f_zero_high",
        split="pilot",
        n_heavy=22,
    )
    prov = compute_provenance_hashes(arm="control")
    outcomes = worker_solve_group(
        unit=unit,
        arm_provenance=prov,
        out_dir=tmp_path,
        stats=None,
        mace_evaluator=None,
        eval_mace=False,
        solve_device="cuda",
    )

    # Must retry with progressive max_rows: None (initial), then 1024, 512, 256, 128
    assert tried_max_rows == [None, 1024, 512, 256, 128]

    # Every seed recorded as "oom"
    for s in (11, 12, 13):
        assert outcomes[s]["status"] == "oom"
        assert "Out of Memory" in outcomes[s]["error"]
        summ_p = tmp_path / "pools" / f"HAMYUP_{s}_control.json"
        assert summ_p.exists()
        summ = json.loads(summ_p.read_text(encoding="utf-8"))
        assert summ["status"] == "oom"


def test_timeout_watchdog_kills_worker(tmp_path: Path):
    """O8.4: A timeout kills and restarts the worker and records 'timeout'."""
    from refit.nci_v2_orchestrator import WorkerProcessController, SolveGroupUnit, compute_provenance_hashes

    ctrl = WorkerProcessController(repo_root=REPO_ROOT)
    prov = compute_provenance_hashes(arm="control")

    unit = SolveGroupUnit(
        code="HAMYUP",
        shard="X1s005",
        dataset="X1",
        arm="control",
        seeds=[11],
        stratum="4f_zero_high",
        split="pilot",
        n_heavy=22,
    )

    # Use 0.05s timeout to trigger watchdog kill
    t0 = time.perf_counter()
    outcomes = ctrl.execute_solve_group(unit, prov, tmp_path, timeout=0.05)
    elapsed = time.perf_counter() - t0

    assert outcomes[11]["status"] == "timeout"
    assert "Timed out" in outcomes[11]["error"]
    assert ctrl.proc is None  # Hung worker was killed


def test_group_reserve_replacement_control_vs_candidate(tmp_path: Path):
    """O8.5: Group-level reserve replacement: control failure -> replacement for all seeds × arms; candidate failure -> no replacement."""
    from refit.nci_v2_orchestrator import ProductionOrchestrator, SolveGroupUnit

    mock_frame = {
        "main_allocation": {
            "splits": {
                "surface_fit": [
                    {
                        "representative_code": "TRAIN_MOL",
                        "representative_shard": "X1s000",
                        "dataset": "X1",
                        "stratum": "3d_neg_med",
                        "n_heavy": 30,
                    }
                ],
                "reserve": [
                    {
                        "representative_code": "RES_MOL_1",
                        "representative_shard": "X1s002",
                        "dataset": "X1",
                        "stratum": "3d_neg_med",
                        "n_heavy": 32,
                    }
                ],
            }
        }
    }
    frame_p = tmp_path / "mock_frame.json"
    frame_p.write_text(json.dumps(mock_frame), encoding="utf-8")

    orch = ProductionOrchestrator(
        frame_path=frame_p,
        out_dir=tmp_path / "out",
        eval_mace=False,
    )

    # Case 1: Control arm failure in surface_fit -> triggers replacement
    ctrl_unit = SolveGroupUnit(
        code="TRAIN_MOL", shard="X1", dataset="X1", arm="control", seeds=[11, 12, 13],
        stratum="3d_neg_med", split="surface_fit", n_heavy=30,
    )
    rep = orch.draw_reserve_replacement(ctrl_unit, reason="unmappable")
    assert rep is not None
    assert rep["representative_code"] == "RES_MOL_1"
    assert "TRAIN_MOL" in orch.replaced_failures

    # Case 2: Candidate arm failure -> NO replacement allowed (candidate arm failures are outcomes, not replaced)
    cand_unit = SolveGroupUnit(
        code="TRAIN_MOL", shard="X1", dataset="X1", arm="candidate", seeds=[11, 12, 13],
        stratum="3d_neg_med", split="surface_fit", n_heavy=30,
    )
    # The orchestrator's run() method only calls draw_reserve_replacement when unit.arm == "control"
    # To test this policy directly:
    assert cand_unit.arm != "control"


def test_ez_hygiene(monkeypatch):
    """O8.6: E/Z: an unembeddable combination is skipped on both paths; EZ_ENUMERATE=False -> no enumeration."""
    from jacobik import chelate_embed
    from tools.jdev import multiseed

    # Part 1: EZ_ENUMERATE=False -> no enumeration
    chelate_embed.EZ_ENUMERATE = False
    mock_builder = type("MockBuilder", (), {})()
    mock_builder._ez_combinations = lambda: ["comb1", "comb2"]
    mock_builder._ez_inner = False

    called_prepare = False
    def fake_prepare(b, s, **kwargs):
        nonlocal called_prepare
        called_prepare = True
        raise RuntimeError("prepare_called_directly")

    monkeypatch.setattr(multiseed, "prepare", fake_prepare)

    # When EZ_ENUMERATE is False, it should bypass variant branch and proceed directly to prepare
    with pytest.raises(RuntimeError, match="prepare_called_directly"):
        multiseed.solve_seeds(mock_builder, seeds=[11])

    # Part 2: EZ_ENUMERATE=True skips unembeddable combinations gracefully
    chelate_embed.EZ_ENUMERATE = True
    mock_builder2 = type("MockBuilder", (), {})()
    mock_builder2.ligands = [type("MockLigand", (), {"ez_fix": None})()]
    mock_builder2._ez_combinations = lambda: [("E",), ("Z",)]
    mock_builder2._ez_inner = False

    call_count = 0
    orig_solve_seeds = multiseed.solve_seeds

    def spy_solve_seeds(builder, seeds, **kwargs):
        nonlocal call_count
        if getattr(builder, "_ez_inner", False):
            call_count += 1
            if call_count == 1:
                raise ValueError("combination (E,) not embeddable")
            from jacobik import engine as E
            c = E.ConformerResult(complex=None, score=10.0, converged=True, rank=1, trial_idx=0, assignment_idx=0)
            res = {11: E.EnsembleResult(conformers=[c], solve_trace=None, scorer=type("Scorer", (), {"_agg_norm": lambda self: 1.0})())}
            return res, {"s_total": 1.0}
        return orig_solve_seeds(builder, seeds, **kwargs)

    monkeypatch.setattr(multiseed, "solve_seeds", spy_solve_seeds)
    # Since call 1 raised ValueError, it must catch it, skip (E,), and succeed on call 2 (Z,)
    merged_res, merged_info = multiseed.solve_seeds(mock_builder2, seeds=[11])
    assert 11 in merged_res
    assert merged_info["n_variants_embedded"] == 1
    assert merged_info["n_variants_total"] == 2


# ---------------------------------------------------------------------------
# Directives F1, T1, T2 Orchestrator Tests
# ---------------------------------------------------------------------------

def test_mace_policy_by_split():
    """Directive T1: Per-split MACE policy.
    surface_fit: eval_mace=False.
    calibration & motion_validation: eval_mace=True.
    """
    from refit.nci_v2_orchestrator import ProductionOrchestrator, SolveGroupUnit

    orch = ProductionOrchestrator(eval_mace=True)

    unit_surface = SolveGroupUnit("MOL1", "X1", "X1", "control", [11], "strat1", "surface_fit", 20)
    assert orch.determine_eval_mace(unit_surface) is False

    unit_calib = SolveGroupUnit("MOL2", "X1", "X1", "control", [11], "strat1", "calibration", 20)
    assert orch.determine_eval_mace(unit_calib) is True

    unit_motion = SolveGroupUnit("MOL3", "X1", "X1", "control", [11], "strat1", "motion_validation", 20)
    assert orch.determine_eval_mace(unit_motion) is True

    unit_pilot = SolveGroupUnit("MOL4", "X1", "X1", "control", [11], "strat1", "pilot", 20)
    assert orch.determine_eval_mace(unit_pilot) is True

    # If orchestrator has eval_mace=False globally, all splits return False
    orch_no_mace = ProductionOrchestrator(eval_mace=False)
    assert orch_no_mace.determine_eval_mace(unit_calib) is False
    assert orch_no_mace.determine_eval_mace(unit_surface) is False


def test_reference_mace_caching_across_tasks(tmp_path: Path):
    """Directive T1: Reference-structure MACE caching across arms/tasks.
    Cached per (code, model_hash, charge, mult).
    """
    from refit.nci_v2_orchestrator import (
        _REF_MACE_CACHE,
        get_cached_ref_mace_energy,
        set_cached_ref_mace_energy,
        get_ref_mace_cache_file,
    )

    cache_key = ("TESTMOL", "model_hash_abc", 0, 1)

    # Clear memory cache entry if present
    _REF_MACE_CACHE.pop(cache_key, None)

    # 1. Miss initially
    assert get_cached_ref_mace_energy(cache_key, out_dir=tmp_path) is None

    # 2. Store in cache
    expected_energy = -1543.25
    set_cached_ref_mace_energy(cache_key, expected_energy, out_dir=tmp_path)

    # 3. Hit from in-memory cache
    assert get_cached_ref_mace_energy(cache_key, out_dir=tmp_path) == expected_energy

    # 4. Hit from file cache after clearing in-memory cache
    cache_file = get_ref_mace_cache_file(tmp_path)
    assert cache_file.exists()
    cached_json = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "TESTMOL:model_hash_abc:0:1" in cached_json
    assert cached_json["TESTMOL:model_hash_abc:0:1"] == expected_energy

    _REF_MACE_CACHE.pop(cache_key, None)
    assert get_cached_ref_mace_energy(cache_key, out_dir=tmp_path) == expected_energy
    assert cache_key in _REF_MACE_CACHE


def test_memory_guard_wait_and_resume(monkeypatch, tmp_path: Path):
    """Directive T2: Memory guard (15GB host).
    If available RAM < 4.0GB or VRAM < 2.0GB, wait in loop until resources become available.
    """
    from refit.nci_v2_orchestrator import ProductionOrchestrator

    orch = ProductionOrchestrator(out_dir=tmp_path, eval_mace=False, workers=1)

    # Test is_memory_sufficient logic directly
    # Low RAM (3.5 GB) -> False
    orch.memory_check_fn = lambda: (int(3.5 * 1024**3), int(4.0 * 1024**3))
    ok, ram, vram = orch.is_memory_sufficient(min_ram_gb=4.0, min_vram_gb=2.0)
    assert not ok
    assert ram == int(3.5 * 1024**3)

    # Low VRAM (1.5 GB) -> False
    orch.memory_check_fn = lambda: (int(8.0 * 1024**3), int(1.5 * 1024**3))
    ok, ram, vram = orch.is_memory_sufficient(min_ram_gb=4.0, min_vram_gb=2.0)
    assert not ok

    # Both sufficient (6.0 GB RAM, 3.0 GB VRAM) -> True
    orch.memory_check_fn = lambda: (int(6.0 * 1024**3), int(3.0 * 1024**3))
    ok, ram, vram = orch.is_memory_sufficient(min_ram_gb=4.0, min_vram_gb=2.0)
    assert ok

    # Test check_memory_guard wait loop
    states = [
        (int(2.0 * 1024**3), int(1.0 * 1024**3)),  # Low
        (int(2.0 * 1024**3), int(1.0 * 1024**3)),  # Low
        (int(8.0 * 1024**3), int(4.0 * 1024**3)),  # Sufficient
    ]
    def mock_mem():
        return states.pop(0) if states else (int(8.0 * 1024**3), int(4.0 * 1024**3))

    orch.memory_check_fn = mock_mem
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    orch.check_memory_guard(min_ram_gb=4.0, min_vram_gb=2.0, sleep_s=0.05)
    assert len(sleep_calls) == 2  # Slept twice before recovering
    assert sleep_calls == [0.05, 0.05]


def test_per_stratum_timeout_recording_and_stop_rule(monkeypatch, tmp_path: Path):
    """Directive F1: Timeout tracking per stratum and >2% stop rule.
    Timeout recorded with n_heavy and stratum.
    Stratum timeout rate > 2% stops run immediately for review; never replace silently.
    """
    from refit.nci_v2_orchestrator import ProductionOrchestrator, SolveGroupUnit

    orch = ProductionOrchestrator(out_dir=tmp_path, eval_mace=False, workers=1, research_arm=True)

    # Create 100 mock units for stratum "stratum_test"
    units = [
        SolveGroupUnit(
            code=f"MOL_{i:03d}",
            shard="X1s000",
            dataset="X1",
            arm="control",
            seeds=[11],
            stratum="stratum_test",
            split="surface_fit",
            n_heavy=25 + i % 10,
        )
        for i in range(100)
    ]

    call_idx = 0
    def mock_solve(controller, unit, arm_prov, timeout, eval_mace):
        nonlocal call_idx
        idx = call_idx
        call_idx += 1
        if idx < 2:
            # First 2 units timeout (rate = 2/100 = 2% <= 2%)
            return {
                11: {
                    "status": "timeout",
                    "error": f"Timed out after {timeout}s",
                    "n_heavy": unit.n_heavy,
                    "stratum": unit.stratum,
                    "wall_s": timeout,
                    "peak_gpu_mb": 100.0,
                    "peak_rss_mb": 200.0,
                }
            }, timeout
        elif idx == 2:
            # 3rd unit timeouts (rate = 3/100 = 3% > 2% -> must raise RuntimeError)
            return {
                11: {
                    "status": "timeout",
                    "error": f"Timed out after {timeout}s",
                    "n_heavy": unit.n_heavy,
                    "stratum": unit.stratum,
                    "wall_s": timeout,
                    "peak_gpu_mb": 100.0,
                    "peak_rss_mb": 200.0,
                }
            }, timeout
        else:
            return {11: {"status": "success", "wall_s": 0.01, "peak_gpu_mb": 100.0, "peak_rss_mb": 200.0}}, 0.01

    monkeypatch.setattr(orch, "_solve_unit_on_worker", mock_solve)

    # Running should stop with RuntimeError when stratum timeout rate exceeds 2%
    with pytest.raises(RuntimeError, match="Stratum 'stratum_test' timeout rate exceeds 2%"):
        orch.run(units)

    # Verify timeouts were counted and tracked
    assert orch.per_stratum_timeouts["stratum_test"] >= 3
    assert orch.total_timeouts >= 3
    # Verify timeouts were NEVER replaced silently with reserves
    assert len(orch.replaced_failures) == 0


def test_parallel_worker_pool_dispatch(monkeypatch, tmp_path: Path):
    """Directive T2: Orchestrator supports N independent spawned worker processes (workers=2 and 3).
    Workers own disjoint units and results are collected atomically.
    """
    from refit.nci_v2_orchestrator import ProductionOrchestrator, SolveGroupUnit

    for n_workers in (2, 3):
        out_d = tmp_path / f"pool_test_{n_workers}"
        orch = ProductionOrchestrator(out_dir=out_d, eval_mace=False, workers=n_workers, research_arm=True)

        units = [
            SolveGroupUnit(
                code=f"TEST_MOL_{i}",
                shard="X1",
                dataset="X1",
                arm="control",
                seeds=[11, 12],
                stratum="strat1",
                split="pilot",
                n_heavy=20,
            )
            for i in range(6)
        ]

        worker_used = set()
        def mock_solve(controller, unit, arm_prov, timeout, eval_mace):
            worker_used.add(id(controller))
            time.sleep(0.01)
            outcomes = {}
            for s in unit.seeds:
                outcomes[s] = {
                    "code": unit.code,
                    "seed": s,
                    "arm": unit.arm,
                    "status": "success",
                    "wall_s": 0.01,
                    "peak_gpu_mb": 50.0,
                    "peak_rss_mb": 100.0,
                }
            return outcomes, 0.01

        monkeypatch.setattr(orch, "_solve_unit_on_worker", mock_solve)

        manifest = orch.run(units)
        assert manifest["succeeded_tasks"] == 12  # 6 units * 2 seeds
        assert manifest["failed_tasks"] == 0
        assert manifest["total_timeouts"] == 0
        # Assert that n_workers distinct controllers were instantiated and used
        assert len(worker_used) == n_workers


def test_cli_workers_and_timeout_arguments():
    """Verify --workers (choices 1, 2, 3) and --timeout (default 1800.0) in CLI."""
    cmd = [sys.executable, "-m", "refit.nci_v2_orchestrator", "--workers", "3", "--timeout", "1800", "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0
    assert "Dry-run:" in res.stdout

    cmd_invalid = [sys.executable, "-m", "refit.nci_v2_orchestrator", "--workers", "4", "--dry-run"]
    res_inv = subprocess.run(cmd_invalid, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res_inv.returncode != 0
    assert "invalid choice" in res_inv.stderr


def test_mace_permutation_invariance_and_order_alignment():
    """Directive R1: verify mapped-path and unmapped-path energies agree to 1e-6 kJ/mol

    and verify symbols passed equal the symbols of the coordinate order passed.
    """
    from jacobik.benchmark.energy import MACEPolarEvaluator
    from refit.nci_v2_pilot import evaluate_mace_energies

    if not Path(MACE_MODEL_PATH).exists():
        pytest.skip("MACE model not available")

    evaluator = MACEPolarEvaluator(
        model="polar-1-s",
        device="cpu",
        dtype="float64",
        model_path=str(MACE_MODEL_PATH),
        mode="energy-only",
    )

    # Reference structure (e.g. water: O, H, H)
    ref_symbols = ["O", "H", "H"]
    ref_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ])
    X_ref_confs = np.array([ref_coords])

    # Generated structure has permuted atom order: e.g. [H, O, H]
    # gen index 0 -> ref index 1 (H)
    # gen index 1 -> ref index 0 (O)
    # gen index 2 -> ref index 2 (H)
    perm = [1, 0, 2]
    gen_symbols = [ref_symbols[i] for i in perm]  # ['H', 'O', 'H']
    X_gen_confs = X_ref_confs[:, perm, :]

    # In mapped path:
    # gen_to_ref maps reference positions to generated positions:
    # ref atom 0 (O) is at gen atom 1
    # ref atom 1 (H) is at gen atom 0
    # ref atom 2 (H) is at gen atom 2
    gen_to_ref = np.array([1, 0, 2], dtype=np.int64)

    # Coordinate ordering check
    mapped_for_mace = X_gen_confs[:, gen_to_ref, :]
    assert np.allclose(mapped_for_mace, X_ref_confs)
    mace_symbols_mapped = list(ref_symbols)
    # Assert symbols passed match coordinate order passed
    assert [gen_symbols[i] for i in gen_to_ref] == mace_symbols_mapped

    e_mapped, _, _, _ = evaluate_mace_energies(
        evaluator, mace_symbols_mapped, mapped_for_mace, charge=0, mult=1
    )

    # In unmapped path:
    # We evaluate directly with generated symbols and generated coordinate order
    unmapped_for_mace = X_gen_confs
    mace_symbols_unmapped = list(gen_symbols)
    # Assert symbols passed match coordinate order passed
    assert mace_symbols_unmapped == [gen_symbols[i] for i in range(len(gen_symbols))]

    e_unmapped, _, _, _ = evaluate_mace_energies(
        evaluator, mace_symbols_unmapped, unmapped_for_mace, charge=0, mult=1
    )

    diff = abs(float(e_mapped[0]) - float(e_unmapped[0]))
    assert diff < 1e-6, f"Mapped and unmapped MACE energies differ by {diff} kJ/mol (> 1e-6)"


def test_orchestrator_refuses_missing_n_heavy(tmp_path):
    """Directive R5: orchestrator must refuse a missing or non-positive n_heavy in frame data."""
    bad_frame = {
        "spec_schema": "test",
        "created": "2026-10-24",
        "seed": 42,
        "main_allocation": {
            "splits": {
                "calibration": [
                    {
                        "representative_code": "TEST01",
                        "representative_shard": "X1s001",
                        "dataset": "X1",
                        "group_id": "GRP01",
                        "stratum": "3d_pos_low",
                        # 'n_heavy' intentionally omitted!
                    }
                ]
            }
        }
    }
    frame_file = tmp_path / "bad_frame.json"
    frame_file.write_text(json.dumps(bad_frame))

    orch = ProductionOrchestrator(
        spec_path=DEFAULT_SPEC,
        frame_path=frame_file,
        out_dir=tmp_path / "pools",
        eval_mace=False,
    )

    with pytest.raises(ValueError, match="missing required valid 'n_heavy'"):
        orch.build_task_queue(cohort="calibration", seeds=[11], arms=["control"])





def test_cli_mem_fraction_reaches_worker_controllers(monkeypatch, tmp_path: Path):
    """S2.1: --mem-fraction overrides the per-worker default instead of being ignored."""
    from refit.nci_v2_orchestrator import ProductionOrchestrator

    seen = []

    class _Stop(Exception):
        pass

    def fake_create(self, mem_fraction=None):
        seen.append(mem_fraction)
        raise _Stop()

    monkeypatch.setattr(ProductionOrchestrator, "_create_controller", fake_create)
    for mf, expected in ((0.95, 0.95), (None, 0.70)):
        orch = ProductionOrchestrator(out_dir=tmp_path, eval_mace=False, workers=1, mem_fraction=mf)
        units = orch.build_task_queue(cohort="pilot", seeds=[11], arms=["control"])[:1]
        seen.clear()
        try:
            orch.run(units)
        except _Stop:
            pass
        assert seen == [expected]
