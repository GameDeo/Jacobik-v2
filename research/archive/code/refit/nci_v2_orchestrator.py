"""research/archive/code/refit/nci_v2_orchestrator.py
==============================
Gate G5: Production Solve Orchestrator & CLI for Jacobik NCI v2.

Key features (Audited & Certified per O1–O7):
1. Spawned Long-Lived Workers with Strict Arm Profiles (O1):
   - Worker processes are spawned with arm environment variables (JACOBIK_PROFILE,
     JACOBIK_PROFILE_OVERRIDES) set BEFORE jacobik is imported.
   - Live module attributes also synchronized via apply_arm_profile_live.
   - Per-task assertion of live profile snapshot (nci, nci_types, deterministic_embed).
   - Dynamic live recording of deterministic_embed; per-arm profile fingerprint in provenance.
2. Certified GPU Solve Path (O2):
   - Restructured work unit to (code, arm) -> all seeds solved in one pass via
     multiseed.solve_seeds(builder, seeds=[11,12,13], device="cuda", batched_assembly=True, angle_bends=True).
   - One result file written per (code, seed, arm).
   - Refuses non-GPU configuration for Step 2–6 cohorts unless --research-arm is passed.
   - Provenance records solver class, device, batching, and max_rows.
3. No Silent Solver Switch (O3):
   - CUDA OOM triggers progressive max_rows retries on the GPU path (1024, 512, 256, 128).
   - If retries fail, status "oom" is recorded as an outcome. Never falls back to CPU solver.
4. Watchdog Timeout Enforcement (O4):
   - Per-group timeout enforced by parent watchdog.
   - On timeout, hung worker process is killed, status "timeout" is recorded with wall time,
     and a fresh worker is spawned.
5. Group-Level Reserve Replacement (O5):
   - Reserves replaced ONLY if control arm fails on predeclared criteria in surface_fit (training).
   - Replacement group is solved for ALL seeds × ALL arms.
   - Non-control arm failures are outcomes, never replaced. Strict zero-replacement for evaluation.
6. Cryptographic Provenance & Observability (G5/G6):
   - spec_sha256, frame_sha256, code_dirty_diff_sha256, constraints_db_sha256, mace_model_sha256,
     per-arm profile fingerprint.
   - First-shell coordination integrity evaluated on every pool.
   - ETA, RSS, peak VRAM logging, and final manifest.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
from dataclasses import asdict, dataclass, field
import datetime
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "research" / "refit", REPO_ROOT / "research/bench/tools/jdev/budget", REPO_ROOT / "research/bench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Ensure single-threaded sequential execution per guidelines
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.update(
    MKL_THREADING_LAYER="SEQUENTIAL",
    PYTHONIOENCODING="utf-8",
)

from nci_v2_provenance import code_state_sha256
from tools.jdev.budget.common import DB, stats_from_db
from tools.jdev.budget.pools import rank_terms
from jacobik.benchmark.coordination import coordination_integrity
from jacobik.inputs import load_mol, build_builder_and_inputs
from jacobik.paths import MACE_MODEL_PATH
from research.archive.code.refit.nci_v2_pilot import evaluate_mace_energies, get_gpu_mem_mb, get_rss_mb, reset_gpu_peak, save_pool_npz, sha256_of_file, write_json_atomic, DEFAULT_SPEC, DEFAULT_FRAME, CONSTRAINTS_DB_PATH


# Arm configuration specifications
ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "control": {
        "profile": "u1",
        "overrides": {"nci": False, "deterministic_embed": True},
        "expected_nci": False,
        "expected_nci_types": ("hb",),
        "expected_det_embed": True,
    },
    "incumbent": {
        "profile": "u1",
        "overrides": {"nci": True, "nci_types": ["hb"], "deterministic_embed": True},
        "expected_nci": True,
        "expected_nci_types": ("hb",),
        "expected_det_embed": True,
    },
    "candidate": {
        "profile": "u1",
        "overrides": {"nci": True, "nci_types": ["hb", "pp", "chp"], "deterministic_embed": True},
        "expected_nci": True,
        "expected_nci_types": ("hb", "pp", "chp"),
        "expected_det_embed": True,
    },
}


def apply_arm_profile_live(arm: str) -> None:
    """Set live module attributes for arm in case jacobik was already imported."""
    cfg = ARM_CONFIGS.get(arm)
    if not cfg:
        return
    try:
        from jacobik import profile as _P
        for k, v in cfg["overrides"].items():
            if k in _P.FIELDS:
                mod_name, attr_name, _, _ = _P.FIELDS[k]
                try:
                    mod = importlib.import_module(mod_name)
                    setattr(mod, attr_name, v)
                except Exception:
                    pass
    except Exception:
        pass


def assert_arm_profile(arm: str, live_snap: Dict[str, Any]) -> None:
    """Assert that a live profile snapshot matches the expected configuration for the arm."""
    cfg = ARM_CONFIGS.get(arm)
    if not cfg:
        return
    fields = live_snap.get("fields", {})
    if fields.get("deterministic_embed") is not True:
        raise AssertionError(
            f"Arm '{arm}': deterministic_embed must be True, got {fields.get('deterministic_embed')}"
        )
    if fields.get("nci") != cfg["expected_nci"]:
        raise AssertionError(
            f"Arm '{arm}': expected nci={cfg['expected_nci']}, got {fields.get('nci')}"
        )
    live_types = tuple(fields.get("nci_types", ()))
    exp_types = cfg["expected_nci_types"]
    if sorted(live_types) != sorted(exp_types):
        raise AssertionError(
            f"Arm '{arm}': expected nci_types={exp_types}, got {live_types}"
        )


def resolve_arm_snapshot(arm: str = "control", repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Resolve arm profile snapshot by executing a clean Python subprocess with arm env."""
    cfg = ARM_CONFIGS.get(arm, {"profile": "u1", "overrides": {}})
    env = os.environ.copy()
    env["JACOBIK_PROFILE"] = cfg["profile"]
    env["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(cfg["overrides"])
    pythonpath_parts = [
        str(repo_root / "src"),
        str(repo_root),
        str(repo_root / "research" / "refit"),
        str(repo_root / "research/bench/tools/jdev/budget"), str(repo_root / "research/bench"),
    ]
    if "PYTHONPATH" in os.environ:
        pythonpath_parts.append(os.environ["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    cmd = [
        sys.executable,
        "-c",
        "import json, jacobik.profile as P; print(json.dumps(P.snapshot()))",
    ]
    out = subprocess.check_output(cmd, env=env, cwd=repo_root, text=True)
    snap = json.loads(out.strip())
    assert_arm_profile(arm, snap)
    return snap


@dataclass
class SolveGroupUnit:
    """Specification of a group solve unit: one complex code, one arm, multiple seeds."""
    code: str
    shard: str
    dataset: str
    arm: str
    seeds: List[int]
    stratum: str
    split: str
    n_heavy: int = 0
    group_id: str = ""
    run_id: str = "orchestrator_run"

    @property
    def key(self) -> str:
        return f"{self.code}:{self.arm}"


@dataclass
class SolveTask:
    """Specification of an individual complex solve unit (retained for backward compatibility)."""
    code: str
    shard: str
    dataset: str
    seed: int
    arm: str
    stratum: str
    split: str
    n_heavy: int = 0
    group_id: str = ""
    run_id: str = "orchestrator_run"

    @property
    def key(self) -> str:
        return f"{self.code}:{self.seed}:{self.arm}"


@dataclass
class TaskPaths:
    """Paths to output artifacts for a single task."""
    pool_npz: Path
    mace_json: Path
    summary_json: Path


def get_task_paths(out_dir: Path, code: str, seed: int, arm: str) -> TaskPaths:
    pools_dir = Path(out_dir) / "pools"
    prefix = f"{code}_{seed}_{arm}"
    return TaskPaths(
        pool_npz=pools_dir / f"{prefix}.npz",
        mace_json=pools_dir / f"{prefix}.mace64.json",
        summary_json=pools_dir / f"{prefix}.json",
    )


def compute_provenance_hashes(
    spec_path: Path = DEFAULT_SPEC,
    frame_path: Path = DEFAULT_FRAME,
    constraints_path: Path = CONSTRAINTS_DB_PATH,
    mace_path: Path = MACE_MODEL_PATH,
    repo_root: Path = REPO_ROOT,
    arm: str = "control",
) -> Dict[str, Any]:
    """Compute complete cryptographic provenance hashes including resolved profile."""
    spec_path = Path(spec_path)
    frame_path = Path(frame_path)
    constraints_path = Path(constraints_path)
    mace_path = Path(mace_path)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    dirty_sha = code_state_sha256(repo_root)

    try:
        from jacobik import profile as _P
        apply_arm_profile_live(arm)
        profile_snap = _P.snapshot()
    except Exception:
        profile_snap = resolve_arm_snapshot(arm, repo_root)

    return {
        "spec_sha256": sha256_of_file(spec_path) if spec_path.exists() else "missing",
        "frame_sha256": sha256_of_file(frame_path) if frame_path.exists() else "missing",
        "code_git_commit": commit,
        "code_dirty_diff_sha256": dirty_sha,
        "constraints_db_sha256": sha256_of_file(constraints_path) if constraints_path.exists() else "missing",
        "mace_model_sha256": sha256_of_file(mace_path) if mace_path.exists() else "missing",
        "full_resolved_profile": profile_snap,
        "profile_fingerprint": profile_snap.get("fingerprint"),
    }


def _canonicalize_profile(prof: Any) -> Any:
    """Canonicalize profile dictionary so tuples and lists compare identically."""
    if isinstance(prof, dict):
        return {k: _canonicalize_profile(v) for k, v in prof.items()}
    if isinstance(prof, (list, tuple)):
        return [_canonicalize_profile(x) for x in prof]
    return prof


def should_skip_task(paths: TaskPaths, arm_provenance: Dict[str, Any]) -> bool:
    """Return True only if task artifacts exist, are valid, and provenance hashes match exactly."""
    if not (paths.pool_npz.exists() and paths.mace_json.exists() and paths.summary_json.exists()):
        return False

    try:
        summary_raw = paths.summary_json.read_text(encoding="utf-8")
        summary = json.loads(summary_raw)
        if summary.get("status") != "success":
            return False

        # Verify summary-level provenance
        prov = summary.get("provenance", {})
        hash_keys = (
            "spec_sha256",
            "frame_sha256",
            "code_dirty_diff_sha256",
            "constraints_db_sha256",
            "mace_model_sha256",
        )
        for hk in hash_keys:
            if prov.get(hk) != arm_provenance.get(hk):
                return False

        # Verify profile matches (canonicalized for JSON list vs tuple)
        saved_profile = prov.get("full_resolved_profile")
        current_profile = arm_provenance.get("full_resolved_profile")
        if saved_profile and current_profile:
            if _canonicalize_profile(saved_profile) != _canonicalize_profile(current_profile):
                return False

        # Verify NPZ archive readability and embedded hashes
        with np.load(paths.pool_npz, allow_pickle=True) as npz:
            if "error" in npz:
                return False
            for hk in hash_keys:
                if hk in npz:
                    val = str(npz[hk])
                    if val != str(arm_provenance.get(hk)):
                        return False
            if "profile" in npz and current_profile:
                try:
                    prof_val = json.loads(str(npz["profile"]))
                    if _canonicalize_profile(prof_val) != _canonicalize_profile(current_profile):
                        return False
                except Exception:
                    pass

        # Verify mace64 JSON is valid
        mace_data = json.loads(paths.mace_json.read_text(encoding="utf-8"))
        if "energies_kj" not in mace_data:
            return False
        if summary.get("top1_dE_kj") is not None and not mace_data["energies_kj"]:
            return False

        return True
    except Exception:
        return False


def _record_group_failure(
    unit: SolveGroupUnit,
    arm_provenance: Dict[str, Any],
    out_dir: Path,
    error_msg: str,
    status: str,
    wall_s: float,
    worker_pid: int = 0,
    live_profile_snap: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Record structured failure outcomes for all seeds in a group unit."""
    outcomes: Dict[int, Dict[str, Any]] = {}
    for seed in unit.seeds:
        paths = get_task_paths(out_dir, unit.code, seed, unit.arm)
        save_pool_npz(paths.pool_npz, (False, None, error_msg))
        summ: Dict[str, Any] = {
            "code": unit.code,
            "seed": seed,
            "arm": unit.arm,
            "group_id": unit.group_id,
            "stratum": unit.stratum,
            "split": unit.split,
            "n_heavy": unit.n_heavy,
            "run_id": unit.run_id,
            "worker_pid": worker_pid,
            "status": status,
            "error": error_msg,
            "wall_s": wall_s,
            "peak_rss_mb": get_rss_mb(),
            "peak_gpu_mb": get_gpu_mem_mb(),
            "provenance": arm_provenance,
        }
        if live_profile_snap:
            summ["profile_fingerprint"] = live_profile_snap.get("fingerprint")
        write_json_atomic(paths.summary_json, summ)
        outcomes[seed] = summ
    return outcomes


_REF_MACE_CACHE: Dict[Tuple[str, str, int, int], float] = {}


def get_ref_mace_cache_file(out_dir: Optional[Path]) -> Optional[Path]:
    if out_dir is None:
        return None
    return Path(out_dir) / "ref_mace_cache.json"


def get_cached_ref_mace_energy(
    cache_key: Tuple[str, str, int, int],
    out_dir: Optional[Path] = None,
) -> Optional[float]:
    """Retrieve reference MACE energy for (code, model_hash, charge, mult) from memory or file cache."""
    if cache_key in _REF_MACE_CACHE:
        return _REF_MACE_CACHE[cache_key]

    cache_file = get_ref_mace_cache_file(out_dir)
    if cache_file and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            str_key = f"{cache_key[0]}:{cache_key[1]}:{cache_key[2]}:{cache_key[3]}"
            if str_key in data:
                val = float(data[str_key])
                _REF_MACE_CACHE[cache_key] = val
                return val
        except Exception:
            pass
    return None


def set_cached_ref_mace_energy(
    cache_key: Tuple[str, str, int, int],
    energy_kj: float,
    out_dir: Optional[Path] = None,
) -> None:
    """Store reference MACE energy for (code, model_hash, charge, mult) in memory and atomic file cache."""
    _REF_MACE_CACHE[cache_key] = float(energy_kj)
    cache_file = get_ref_mace_cache_file(out_dir)
    if cache_file:
        try:
            current_data = {}
            if cache_file.exists():
                try:
                    current_data = json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    current_data = {}
            str_key = f"{cache_key[0]}:{cache_key[1]}:{cache_key[2]}:{cache_key[3]}"
            current_data[str_key] = float(energy_kj)
            write_json_atomic(cache_file, current_data)
        except Exception:
            pass


def build_s2b_state_arrays(confs, bend_meta) -> Dict[str, Any]:
    """S2b: per-pose solver state for the pool npz, aligned to the conformer order of X (P poses). Keys: schema tag,
    trial/assignment/embed indices (-1 where None), full final theta vector per pose (thetas_flat with per-ligand
    offsets), the bend-prior constants and the solver's own per-row bend prior. Fails closed when any state is absent."""
    if bend_meta is None:
        raise ValueError("S2b: solve_seeds returned no bend-prior constants")
    widths = [int(w) for w in bend_meta["theta_widths"]]
    offs = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    P, T = len(confs), int(offs[-1])
    thetas = np.zeros((P, T), dtype=np.float64)
    bend = np.zeros(P, dtype=np.float64)
    trial = np.full(P, -1, dtype=np.int64)
    asg = np.full(P, -1, dtype=np.int64)
    emb = np.full(P, -1, dtype=np.int64)
    for p, c in enumerate(confs):
        st = c.solver_state
        if st is None:
            raise ValueError(f"S2b: conformer {p} carries no solver state")
        if len(st["thetas"]) != len(widths):
            raise ValueError(f"S2b: conformer {p} has {len(st['thetas'])} ligand vectors, expected {len(widths)}")
        for li, th in enumerate(st["thetas"]):
            if th.shape[0] != widths[li]:
                raise ValueError(f"S2b: conformer {p} ligand {li} width {th.shape[0]} != {widths[li]}")
            thetas[p, offs[li]:offs[li + 1]] = th
        bend[p] = float(st["bend_prior"])
        trial[p] = -1 if c.trial_idx is None else int(c.trial_idx)
        asg[p] = -1 if c.assignment_idx is None else int(c.assignment_idx)
        emb[p] = -1 if c.embed_idx is None else int(c.embed_idx)
    return {
        "state_schema_version": np.array("s2b_state.1"),
        "trial_idx": trial,
        "assignment_idx": asg,
        "embed_idx": emb,
        "thetas_flat": thetas,
        "thetas_lig_offsets": offs,
        "bend_columns": np.asarray(bend_meta["bend_columns"], dtype=np.int64).reshape(-1, 2),
        "bend_k": np.asarray(bend_meta["bend_k"], dtype=np.float64),
        "bend_free_deg": np.asarray(bend_meta["bend_free_deg"], dtype=np.float64),
        "bend_scale": np.float64(bend_meta["bend_scale"]),
        "bend_per_z": np.bool_(bend_meta["bend_per_z"]),
        "rt_prior": np.float64(bend_meta["rt_prior"]),
        "bend_column_def_sha256": np.array(bend_meta["bend_column_def_sha256"]),
        "bend_prior_solver": bend,
    }


def worker_solve_group(
    unit: SolveGroupUnit,
    arm_provenance: Dict[str, Any],
    out_dir: Path,
    stats: Any,
    mace_evaluator: Optional[Any],
    eval_mace: bool,
    solve_device: str,
) -> Dict[int, Dict[str, Any]]:
    """Execute certified solve_seeds and evaluation for one (code, arm) across all seeds inside worker process."""
    from jacobik import profile as _P
    from jacobik.builder.atom_map import build_atom_mapping
    from tools.jdev import multiseed

    apply_arm_profile_live(unit.arm)
    live_snap = _P.snapshot()
    assert_arm_profile(unit.arm, live_snap)

    pid = os.getpid()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reset_gpu_peak()
    rss_start = get_rss_mb()
    t_start = time.perf_counter()

    mol = load_mol(unit.code, unit.shard, unit.dataset)
    if mol is None:
        err = f"Failed to load {unit.code} from shard {unit.shard} ({unit.dataset})"
        return _record_group_failure(
            unit, arm_provenance, out_dir, err, "failed", time.perf_counter() - t_start, pid, live_snap
        )

    builder, lig_in, err = build_builder_and_inputs(mol)
    if err or builder is None:
        err_msg = err or "build_builder_and_inputs_failed"
        return _record_group_failure(
            unit, arm_provenance, out_dir, err_msg, "unmappable", time.perf_counter() - t_start, pid, live_snap
        )

    # Execute multiseed.solve_seeds on certified GPU path with OOM max_rows retries (O2 & O3)
    t_solve_0 = time.perf_counter()
    used_max_rows = None
    results = None
    info = None

    try:
        # Retries run outside the except blocks: while an OutOfMemoryError is being handled,
        # its traceback keeps the failed attempt's frames (CUDA tensors, CUDA-graph pools) alive,
        # so empty_cache() cannot release them and every retry starts with that memory pinned.
        oom = False
        try:
            results, info = multiseed.solve_seeds(
                builder,
                seeds=unit.seeds,
                device=solve_device,
                batched_assembly=True,
                angle_bends=True,
                stats=stats,
                max_rows=None,
            )
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            oom_success = False
            # max_rows splits by seed only (multiseed.solve_seeds): once one seed exceeds the
            # limit, smaller values give the same one-seed-per-pass schedule.
            for r in (1024, 512, 256, 128):
                gc.collect()
                torch.cuda.empty_cache()
                retry_oom = False
                try:
                    results, info = multiseed.solve_seeds(
                        builder,
                        seeds=unit.seeds,
                        device=solve_device,
                        batched_assembly=True,
                        angle_bends=True,
                        stats=stats,
                        max_rows=r,
                    )
                except torch.cuda.OutOfMemoryError:
                    retry_oom = True
                if not retry_oom:
                    used_max_rows = r
                    oom_success = True
                    break
            if not oom_success:
                gc.collect()
                torch.cuda.empty_cache()
                err_msg = "CUDA Out of Memory on solve_seeds after retrying max_rows down to 128"
                return _record_group_failure(
                    unit, arm_provenance, out_dir, err_msg, "oom", time.perf_counter() - t_start, pid, live_snap
                )
        t_solve = time.perf_counter() - t_solve_0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        t_solve = time.perf_counter() - t_solve_0
        err_msg = f"Solver exception: {type(exc).__name__}: {exc}"
        return _record_group_failure(
            unit, arm_provenance, out_dir, err_msg, "failed", time.perf_counter() - t_start, pid, live_snap
        )

    # Process each seed outcome
    outcomes: Dict[int, Dict[str, Any]] = {}
    ref_X = np.array([a.position for a in mol.atoms])
    ref_symbols = np.array([a.symbol for a in mol.atoms])
    ref_energy_kj = 0.0

    if eval_mace and mace_evaluator is not None:
        model_hash = str(arm_provenance.get("mace_model_sha256", "unknown"))
        charge = int(getattr(mol, "charge", 0))
        mult = int(getattr(mol, "spin_state", 1))
        cache_key = (unit.code, model_hash, charge, mult)

        cached_ref = get_cached_ref_mace_energy(cache_key, out_dir)
        if cached_ref is not None:
            ref_energy_kj = cached_ref
        else:
            ref_kj_arr, _, _, _ = evaluate_mace_energies(
                mace_evaluator, list(ref_symbols), [ref_X], mol.charge, mol.spin_state
            )
            ref_energy_kj = float(ref_kj_arr[0])
            set_cached_ref_mace_energy(cache_key, ref_energy_kj, out_dir)

    for seed in unit.seeds:
        seed_paths = get_task_paths(out_dir, unit.code, seed, unit.arm)
        res = results.get(seed) if results else None
        if res is None or not res.conformers:
            save_pool_npz(seed_paths.pool_npz, (False, None, "empty_conformers"))
            summ = {
                "code": unit.code,
                "seed": seed,
                "arm": unit.arm,
                "group_id": unit.group_id,
                "stratum": unit.stratum,
                "split": unit.split,
                "run_id": unit.run_id,
                "worker_pid": pid,
                "status": "empty_conformers",
                "error": "empty_conformers",
                "t_solve": t_solve,
                "wall_s": time.perf_counter() - t_start,
                "peak_rss_mb": max(get_rss_mb(), rss_start),
                "peak_gpu_mb": get_gpu_mem_mb(),
                "solver_class": "multiseed.solve_seeds",
                "solve_device": solve_device,
                "batched_assembly": True,
                "angle_bends": True,
                "max_rows": used_max_rows,
                "provenance": arm_provenance,
            }
            write_json_atomic(seed_paths.summary_json, summ)
            outcomes[seed] = summ
            continue

        confs = res.conformers
        X = np.array([[at.position for at in c.complex.atoms] for c in confs], dtype=np.float64)
        sy = np.array([at.symbol for at in confs[0].complex.atoms])
        asg = np.array([c.assignment_idx or 0 for c in confs])
        scores = np.array([c.score for c in confs], dtype=np.float64)

        sc = res.scorer
        sc_dev = "cpu"
        if hasattr(sc, "_device") and sc._device:
            sc_dev = sc._device
        elif hasattr(sc, "_q1_means") and hasattr(sc._q1_means, "device"):
            sc_dev = sc._q1_means.device
        elif hasattr(sc, "_q1_cand") and sc._q1_cand and hasattr(sc._q1_cand[0], "device"):
            sc_dev = sc._q1_cand[0].device

        Xt = torch.tensor(X, dtype=torch.float32, device=sc_dev)
        rg = torch.as_tensor(asg, dtype=torch.long, device=sc_dev)
        tot, bd = rank_terms(sc, Xt, rg)
        Z = float(sc._agg_norm())

        save_dict: Dict[str, Any] = {
            "X": X,
            "symbols": sy,
            "score": scores,
            "asg": asg,
            "rank": tot.detach().cpu().numpy() if hasattr(tot, "detach") else tot,
            "Z": Z,
            "t_solve": t_solve,
            "ref_X": ref_X,
            "ref_symbols": ref_symbols,
            "charge": mol.charge,
            "mult": mol.spin_state,
            "profile": json.dumps(live_snap, default=str),
            "profile_fingerprint": live_snap["fingerprint"],
            "worker_pid": pid,
            "deterministic_embed": live_snap["fields"]["deterministic_embed"],
            "arm": unit.arm,
            "seed": seed,
            "code": unit.code,
            "solver_class": "multiseed.solve_seeds",
            "solve_device": solve_device,
            "batched_assembly": True,
            "angle_bends": True,
            "max_rows": used_max_rows,
        }
        for q, v in bd.items():
            arr = v.detach().cpu().numpy()
            save_dict[f"terms_{q}"] = arr * Z if q != "Q7" else arr
        # S2b: per-pose solver state (capture only; the pool's X, score, asg and ranking above are unchanged)
        save_dict.update(build_s2b_state_arrays(confs, info.get("s2b_bend") if info else None))

        # Coordination integrity & atom mapping
        coord_intact_arr = None
        sigma_shift_arr = None
        haptic_slip_arr = None
        haptic_shift_arr = None
        gen_to_ref = None

        try:
            am = build_atom_mapping(mol, gen_mol=confs[0].complex, X_ref=ref_X, X_gen=X[0])
            gen_to_ref = am.gen_to_ref_idx
            save_dict["gen_to_ref_idx"] = gen_to_ref
            save_dict["ref_to_gen_idx"] = am.ref_to_gen_idx
            save_dict["mapping_ambiguous"] = am.is_ambiguous
            save_dict["num_automorphisms"] = am.num_automorphisms

            mapped_confs = X[:, gen_to_ref, :] if gen_to_ref is not None else X
            ci = coordination_integrity(ref_symbols, ref_X, mapped_confs)

            coord_intact_arr = ci["intact"]
            sigma_shift_arr = ci["sigma_shift_A"]
            haptic_slip_arr = ci["haptic_slip_A"]
            haptic_shift_arr = ci["haptic_shift_A"]

            save_dict["coord_intact"] = coord_intact_arr
            save_dict["coord_sigma_shift_A"] = sigma_shift_arr
            save_dict["coord_haptic_slip_A"] = haptic_slip_arr
            save_dict["coord_haptic_shift_A"] = haptic_shift_arr
        except Exception as exc:
            save_dict["atom_mapping_error"] = str(exc)

        for pk in (
            "spec_sha256",
            "frame_sha256",
            "code_git_commit",
            "code_dirty_diff_sha256",
            "constraints_db_sha256",
            "mace_model_sha256",
        ):
            if pk in arm_provenance:
                save_dict[pk] = arm_provenance[pk]

        save_pool_npz(seed_paths.pool_npz, (True, save_dict, None))

        # Evaluate MACE energies
        mace_t = 0.0
        top1_idx = int(np.argmax(scores))
        floor_idx = 0
        top1_kj = 0.0
        floor_kj = 0.0
        energies_kj_list = []

        if eval_mace and mace_evaluator is not None:
            if gen_to_ref is not None:
                mapped_for_mace = X[:, gen_to_ref, :]
                mace_symbols = list(ref_symbols)
                assert list(sy[gen_to_ref]) == mace_symbols, (
                    f"Mapped symbols mismatch: {list(sy[gen_to_ref])} vs {mace_symbols}"
                )
            else:
                mapped_for_mace = X
                mace_symbols = list(sy)

            energies_kj_arr, mace_t, _, _ = evaluate_mace_energies(
                mace_evaluator, mace_symbols, mapped_for_mace, mol.charge, mol.spin_state
            )
            energies_kj_list = energies_kj_arr.tolist()
            floor_idx = int(np.argmin(energies_kj_arr))
            top1_kj = float(energies_kj_arr[top1_idx])
            floor_kj = float(energies_kj_arr[floor_idx])

            write_json_atomic(seed_paths.mace_json, {
                "code": unit.code,
                "seed": seed,
                "arm": unit.arm,
                "run_id": unit.run_id,
                "energies_kj": energies_kj_list,
                "ref_energy_kj": ref_energy_kj,
                "top1_idx": top1_idx,
                "floor_idx": floor_idx,
            })
        else:
            write_json_atomic(seed_paths.mace_json, {
                "code": unit.code,
                "seed": seed,
                "arm": unit.arm,
                "run_id": unit.run_id,
                "energies_kj": [],
                "ref_energy_kj": 0.0,
                "top1_idx": top1_idx,
                "floor_idx": 0,
            })

        peak_vram = get_gpu_mem_mb()
        peak_rss = max(get_rss_mb(), rss_start)
        wall_total = time.perf_counter() - t_start
        confs_hash = hashlib.sha256(X.tobytes()).hexdigest()

        summ = {
            "code": unit.code,
            "seed": seed,
            "arm": unit.arm,
            "group_id": unit.group_id,
            "stratum": unit.stratum,
            "split": unit.split,
            "run_id": unit.run_id,
            "worker_pid": pid,
            "status": "success",
            "t_solve": t_solve,
            "mace_t": mace_t,
            "stage_embed_s": float(info.get("s_embed_shell_enum", 0.0)) if info else 0.0,
            "stage_seed_rows_s": float(info.get("s_seed_rows", 0.0)) if info else 0.0,
            "stage_solver_s": float(info.get("s_solver", 0.0)) if info else 0.0,
            "stage_dedup_s": float(info.get("s_dedup_assembly", 0.0)) if info else 0.0,
            "stage_mace_s": float(mace_t),
            "stage_other_s": max(
                0.0,
                wall_total - (
                    (float(info.get("s_embed_shell_enum", 0.0)) if info else 0.0)
                    + (float(info.get("s_seed_rows", 0.0)) if info else 0.0)
                    + (float(info.get("s_solver", 0.0)) if info else 0.0)
                    + (float(info.get("s_dedup_assembly", 0.0)) if info else 0.0)
                    + float(mace_t)
                ),
            ),
            "wall_s": wall_total,
            "n_confs": len(confs),
            "top1_idx": top1_idx,
            "floor_idx": floor_idx,
            "top1_kj": top1_kj,
            "floor_kj": floor_kj,
            "ref_kj": ref_energy_kj,
            "top1_dE_kj": top1_kj - ref_energy_kj if eval_mace else None,
            "floor_dE_kj": floor_kj - ref_energy_kj if eval_mace else None,
            "confs_sha256": confs_hash,
            "top1_coord_intact": bool(coord_intact_arr[top1_idx]) if coord_intact_arr is not None else None,
            "floor_coord_intact": bool(coord_intact_arr[floor_idx]) if coord_intact_arr is not None else None,
            "pool_coord_intact_fraction": float(np.mean(coord_intact_arr)) if coord_intact_arr is not None else None,
            "peak_rss_mb": peak_rss,
            "peak_gpu_mb": peak_vram,
            "solver_class": "multiseed.solve_seeds",
            "solve_device": solve_device,
            "batched_assembly": True,
            "angle_bends": True,
            "max_rows": used_max_rows,
            "provenance": arm_provenance,
            "pool_path": str(seed_paths.pool_npz),
            "mace_path": str(seed_paths.mace_json),
            "summary_path": str(seed_paths.summary_json),
        }
        write_json_atomic(seed_paths.summary_json, summ)
        outcomes[seed] = summ

        del Xt, rg, tot, bd, X, confs, save_dict
        if coord_intact_arr is not None:
            del coord_intact_arr, sigma_shift_arr, haptic_slip_arr, haptic_shift_arr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    if results is not None:
        del results, info
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    return outcomes


def worker_server_main(
    arm: str,
    device: str = "cuda",
    solve_device: str = "cuda",
    eval_mace: bool = True,
    constraints_db_path: Path = CONSTRAINTS_DB_PATH,
    mace_model_path: Path = MACE_MODEL_PATH,
    mem_fraction: Optional[float] = None,
) -> None:
    """Entry point for spawned worker process running standard JSON protocol over stdin/stdout."""
    from jacobik import profile as _P
    from jacobik.benchmark.energy import MACEPolarEvaluator

    apply_arm_profile_live(arm)
    live_snap = _P.snapshot()
    assert_arm_profile(arm, live_snap)

    if mem_fraction is None:
        try:
            mem_fraction = float(os.environ.get("JACOBIK_MACE_MEM_FRACTION", "0.70"))
        except Exception:
            mem_fraction = 0.70

    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(mem_fraction)
        except Exception:
            pass

    mace_evaluator = None
    if eval_mace and Path(mace_model_path).exists():
        mace_evaluator = MACEPolarEvaluator(
            model="polar-1-s",
            device=device if (device == "cuda" and torch.cuda.is_available()) else "cpu",
            dtype="float64",
            model_path=str(mace_model_path),
            mode="energy-only",
            threads=1,
        ).load()

    stats = stats_from_db(constraints_db_path) if Path(constraints_db_path).exists() else stats_from_db(DB)

    # Handshake
    handshake = {
        "status": "ready",
        "pid": os.getpid(),
        "arm": arm,
        "profile_snapshot": live_snap,
    }
    sys.stdout.write(json.dumps(handshake) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        cmd = req.get("cmd")
        if cmd == "exit":
            break
        elif cmd == "solve_group":
            unit_data = req["unit"]
            unit = SolveGroupUnit(
                code=unit_data["code"],
                shard=unit_data["shard"],
                dataset=unit_data["dataset"],
                arm=unit_data["arm"],
                seeds=unit_data["seeds"],
                stratum=unit_data["stratum"],
                split=unit_data["split"],
                n_heavy=unit_data.get("n_heavy", 0),
                group_id=unit_data.get("group_id", ""),
                run_id=unit_data.get("run_id", "orchestrator_run"),
            )
            arm_provenance = req["arm_provenance"]
            out_dir = Path(req["out_dir"])
            eval_mace_unit = req.get("eval_mace", eval_mace)
            t0 = time.perf_counter()
            outcomes = worker_solve_group(
                unit=unit,
                arm_provenance=arm_provenance,
                out_dir=out_dir,
                stats=stats,
                mace_evaluator=mace_evaluator,
                eval_mace=eval_mace_unit,
                solve_device=solve_device,
            )
            resp = {
                "status": "ok",
                "wall_s": time.perf_counter() - t0,
                "outcomes": {str(s): summ for s, summ in outcomes.items()},
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


class WorkerProcessController:
    """Manages worker subprocess lifecycle, arm environment spawning, and parent watchdog."""

    def __init__(
        self,
        device: str = "cuda",
        solve_device: str = "cuda",
        eval_mace: bool = True,
        constraints_db_path: Path = CONSTRAINTS_DB_PATH,
        mace_model_path: Path = MACE_MODEL_PATH,
        repo_root: Path = REPO_ROOT,
        mem_fraction: Optional[float] = None,
    ) -> None:
        self.device = device
        self.solve_device = solve_device
        self.eval_mace = eval_mace
        self.constraints_db_path = Path(constraints_db_path)
        self.mace_model_path = Path(mace_model_path)
        self.repo_root = Path(repo_root)
        self.mem_fraction = mem_fraction
        self.proc: Optional[subprocess.Popen] = None
        self.active_arm: Optional[str] = None
        self.active_snapshot: Optional[Dict[str, Any]] = None

    def get_or_spawn_worker(self, arm: str) -> subprocess.Popen:
        """Return alive worker subprocess for arm, spawning a new one if necessary."""
        if self.proc is not None and self.active_arm == arm and self.proc.poll() is None:
            return self.proc

        self.shutdown_worker()
        cfg = ARM_CONFIGS.get(arm, {"profile": "u1", "overrides": {}})
        env = os.environ.copy()
        env["JACOBIK_PROFILE"] = cfg["profile"]
        env["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(cfg["overrides"])
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        if self.mem_fraction is not None:
            env["JACOBIK_MACE_MEM_FRACTION"] = str(self.mem_fraction)
        else:
            env["JACOBIK_MACE_MEM_FRACTION"] = "0.70"
        env["JACOBIK_MACE_BATCH_SIZE"] = "8"
        env["JACOBIK_MACE_BATCH_ATOMS"] = "200"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        pythonpath_parts = [
            str(self.repo_root / "src"),
            str(self.repo_root),
            str(self.repo_root / "research" / "refit"),
            str(self.repo_root / "research/bench/tools/jdev/budget", repo_root / "research/bench"),
        ]
        if "PYTHONPATH" in os.environ:
            pythonpath_parts.append(os.environ["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(pythonpath_parts)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "research.archive.code.refit.nci_v2_orchestrator",
            "--worker-server",
            "--worker-arm", arm,
            "--device", self.device,
            "--solve-device", self.solve_device,
            "--constraints-db", str(self.constraints_db_path),
            "--mace-model", str(self.mace_model_path),
        ]
        if self.mem_fraction is not None:
            cmd.extend(["--mem-fraction", str(self.mem_fraction)])
        if not self.eval_mace:
            cmd.append("--no-mace")

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            cwd=self.repo_root,
            env=env,
        )
        self.active_arm = arm

        # Read lines until handshake line arrives (skipping third-party library banner lines)
        ready_data = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                ret = self.proc.poll()
                raise RuntimeError(f"Worker for arm {arm} failed to start (exit code {ret})")
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("status") == "ready":
                    ready_data = data
                    break
            except Exception:
                continue

        self.active_snapshot = ready_data.get("profile_snapshot")
        assert_arm_profile(arm, self.active_snapshot)
        return self.proc

    def shutdown_worker(self) -> None:
        """Cleanly terminate worker process."""
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    try:
                        self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                        self.proc.stdin.flush()
                        self.proc.wait(timeout=2.0)
                    except Exception:
                        self.proc.terminate()
                        self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            finally:
                self.proc = None
                self.active_arm = None
                self.active_snapshot = None

    def execute_solve_group(
        self,
        unit: SolveGroupUnit,
        arm_provenance: Dict[str, Any],
        out_dir: Path,
        timeout: float = 1800.0,
        eval_mace: Optional[bool] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Send group solve request to worker and monitor execution with parent watchdog."""
        proc = self.get_or_spawn_worker(unit.arm)
        req = {
            "cmd": "solve_group",
            "unit": asdict(unit),
            "arm_provenance": arm_provenance,
            "out_dir": str(out_dir),
            "eval_mace": self.eval_mace if eval_mace is None else eval_mace,
        }
        t0 = time.perf_counter()
        try:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            self.shutdown_worker()
            return _record_group_failure(
                unit, arm_provenance, out_dir, f"Worker IPC error: {exc}", "failed", time.perf_counter() - t0
            )

        # Watchdog loop
        resp_data = None
        while True:
            elapsed = time.perf_counter() - t0
            time_left = max(0.01, timeout - elapsed)
            rlist, _, _ = select.select([proc.stdout], [], [], min(1.0, time_left))
            if rlist:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "status" in data and ("outcomes" in data or "error" in data):
                        resp_data = data
                        break
                except Exception:
                    continue

            if (time.perf_counter() - t0) >= timeout:
                break
            if proc.poll() is not None:
                break

        if resp_data is None and (time.perf_counter() - t0) >= timeout:
            wall_s = time.perf_counter() - t0
            err_msg = f"Timed out after {wall_s:.2f}s"
            self.shutdown_worker()
            return _record_group_failure(unit, arm_provenance, out_dir, err_msg, "timeout", wall_s)

        if resp_data is None and proc.poll() is not None:
            wall_s = time.perf_counter() - t0
            err_msg = f"Worker process crashed (exit code {proc.returncode})"
            self.shutdown_worker()
            return _record_group_failure(unit, arm_provenance, out_dir, err_msg, "crashed", wall_s)

        if resp_data and resp_data.get("status") == "ok":
            return {int(s): summ for s, summ in resp_data["outcomes"].items()}
        else:
            err = resp_data.get("error", "worker_error") if resp_data else "unknown_error"
            st = resp_data.get("status", "failed") if resp_data else "failed"
            return _record_group_failure(unit, arm_provenance, out_dir, err, st, time.perf_counter() - t0)


class LongLivedWorker:
    """Worker maintaining long-lived evaluator in memory (backward compatible in-process interface)."""

    def __init__(
        self,
        mace_model_path: Path = MACE_MODEL_PATH,
        constraints_db_path: Path = CONSTRAINTS_DB_PATH,
        device: str = "cuda",
        solve_device: str = "cuda",
        eval_mace: bool = True,
    ) -> None:
        from jacobik.benchmark.energy import MACEPolarEvaluator
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.solve_device = solve_device
        self.eval_mace = eval_mace
        self.mace_model_path = Path(mace_model_path)
        self.constraints_db_path = Path(constraints_db_path)
        self.stats = stats_from_db(self.constraints_db_path) if self.constraints_db_path.exists() else stats_from_db(DB)
        self.mace_evaluator = None

        if self.eval_mace and self.mace_model_path.exists():
            self.mace_evaluator = MACEPolarEvaluator(
                model="polar-1-s",
                device=self.device,
                dtype="float64",
                model_path=str(self.mace_model_path),
                mode="energy-only",
                threads=1,
            ).load()

    def solve(
        self,
        task: SolveTask,
        provenance: Dict[str, Any],
        paths: TaskPaths,
    ) -> Dict[str, Any]:
        """Solve a single task (compatibility helper for unit tests)."""
        apply_arm_profile_live(task.arm)
        unit = SolveGroupUnit(
            code=task.code,
            shard=task.shard,
            dataset=task.dataset,
            arm=task.arm,
            seeds=[task.seed],
            stratum=task.stratum,
            split=task.split,
            n_heavy=task.n_heavy,
            group_id=task.group_id,
            run_id=task.run_id,
        )
        outcomes = worker_solve_group(
            unit=unit,
            arm_provenance=provenance,
            out_dir=paths.pool_npz.parents[1],
            stats=self.stats,
            mace_evaluator=self.mace_evaluator,
            eval_mace=self.eval_mace,
            solve_device=self.solve_device,
        )
        return outcomes[task.seed]


class ProductionOrchestrator:
    """Production orchestrator managing task queue, resume, reserves, batching, and observability."""

    def __init__(
        self,
        spec_path: Path = DEFAULT_SPEC,
        frame_path: Path = DEFAULT_FRAME,
        constraints_db_path: Path = CONSTRAINTS_DB_PATH,
        mace_model_path: Path = MACE_MODEL_PATH,
        out_dir: Path = REPO_ROOT / "data/refit/nci_v2/orchestrator_runs",
        device: str = "cuda",
        solve_device: str = "cuda",
        research_arm: bool = False,
        eval_mace: bool = True,
        timeout: float = 1800.0,
        batch_size: int = 16,
        force_recompute: bool = False,
        workers: int = 2,
        mem_fraction: Optional[float] = None,
    ) -> None:
        self.spec_path = Path(spec_path)
        self.frame_path = Path(frame_path)
        self.constraints_db_path = Path(constraints_db_path)
        self.mace_model_path = Path(mace_model_path)
        self.out_dir = Path(out_dir)
        self.device = device
        self.solve_device = solve_device
        self.research_arm = research_arm
        self.eval_mace = eval_mace
        self.timeout = timeout
        self.batch_size = batch_size
        self.force_recompute = force_recompute
        self.workers = max(1, min(3, int(workers)))
        self.mem_fraction = mem_fraction

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "pools").mkdir(parents=True, exist_ok=True)

        self.frame_data = self._load_frame()
        self.base_provenance = compute_provenance_hashes(
            spec_path=self.spec_path,
            frame_path=self.frame_path,
            constraints_path=self.constraints_db_path,
            mace_path=self.mace_model_path,
            arm="control",
        )

        # Predeclared reserves partitioned by stratum
        self.reserves_by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.used_reserves: Set[str] = set()
        self._init_reserves()

        self.replaced_failures: Dict[str, Dict[str, Any]] = {}
        self.exhausted_strata: Set[str] = set()
        self.arm_snapshots: Dict[str, Dict[str, Any]] = {}
        self.arm_provenance: Dict[str, Dict[str, Any]] = {}

        # Stratum accounting and timeout tracking (Directive F1)
        self.per_stratum_timeouts: Dict[str, int] = defaultdict(int)
        self.total_timeouts: int = 0
        self.stratum_total_units: Dict[str, int] = defaultdict(int)
        self.stratum_processed_units: Dict[str, int] = defaultdict(int)
        self.memory_check_fn: Optional[Any] = None

    def _load_frame(self) -> Dict[str, Any]:
        if not self.frame_path.exists():
            return {}
        return json.loads(self.frame_path.read_text(encoding="utf-8"))

    def _init_reserves(self) -> None:
        """Initialize reserve list per stratum in predeclared order from frame.json."""
        res_list = self.frame_data.get("main_allocation", {}).get("splits", {}).get("reserve", [])
        for item in res_list:
            stratum = item.get("stratum", "unknown")
            self.reserves_by_stratum[stratum].append(item)

    def get_arm_provenance(self, arm: str) -> Dict[str, Any]:
        """Get or compute cryptographic provenance with live resolved profile snapshot for arm."""
        if arm not in self.arm_provenance:
            snap = resolve_arm_snapshot(arm)
            self.arm_snapshots[arm] = snap
            self.arm_provenance[arm] = {
                **self.base_provenance,
                "full_resolved_profile": snap,
                "profile_fingerprint": snap["fingerprint"],
            }
        return self.arm_provenance[arm]

    def draw_reserve_replacement(
        self,
        failed_task: Any,
        reason: str = "solve_failed",
        stratum: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Strict reserve replacement policy: training (surface_fit) only, matching stratum, predeclared order."""
        if isinstance(failed_task, (SolveTask, SolveGroupUnit)):
            failed_code = failed_task.code
            s_stratum = getattr(failed_task, "stratum", stratum)
            s_split = getattr(failed_task, "split", split)
        else:
            failed_code = str(failed_task)
            s_stratum = stratum or "unknown"
            s_split = split or "unknown"

        if s_split != "surface_fit":
            return None

        candidates = self.reserves_by_stratum.get(s_stratum, [])
        for cand in candidates:
            code = cand.get("representative_code")
            if code and code not in self.used_reserves and code != failed_code:
                self.used_reserves.add(code)
                self.replaced_failures[failed_code] = {
                    "stratum": s_stratum,
                    "failed_code": failed_code,
                    "replacement_code": code,
                    "reason": reason,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                return cand

        self.exhausted_strata.add(s_stratum)
        return None

    def build_task_queue(
        self,
        cohort: str = "pilot",
        codes: Optional[Sequence[str]] = None,
        seeds: Sequence[int] = (11, 12, 13),
        arms: Sequence[str] = ("control",),
    ) -> List[SolveGroupUnit]:
        """Build initial group task list from frame cohort specification."""
        units: List[SolveGroupUnit] = []
        code_filter = set(codes) if codes else None
        items_to_add: List[Tuple[Dict[str, Any], str]] = []

        if cohort in ("pilot", "all"):
            for p in self.frame_data.get("pilot_cohort", {}).get("parents", []):
                items_to_add.append((p, "pilot"))

        splits_in_frame = self.frame_data.get("main_allocation", {}).get("splits", {})
        if cohort in ("surface_fit", "training", "main", "all"):
            for p in splits_in_frame.get("surface_fit", []):
                items_to_add.append((p, "surface_fit"))
        if cohort in ("calibration", "main", "all"):
            for p in splits_in_frame.get("calibration", []):
                items_to_add.append((p, "calibration"))
        if cohort in ("motion_validation", "main", "all"):
            for p in splits_in_frame.get("motion_validation", []):
                items_to_add.append((p, "motion_validation"))

        seeds_list = [int(s) for s in seeds]
        for item, split_label in items_to_add:
            c = item.get("code") or item.get("representative_code")
            if not c:
                continue
            if code_filter and c not in code_filter:
                continue

            shard = item.get("shard") or item.get("representative_shard") or "unknown"
            dataset = item.get("dataset") or "unknown"
            stratum = item.get("stratum") or f"{item.get('metal_row')}_{item.get('charge_class')}_{item.get('cn_class')}"
            group_id = item.get("group_id", "")
            if "n_heavy" not in item or item["n_heavy"] is None or int(item["n_heavy"]) <= 0:
                raise ValueError(f"Task {c} in frame data is missing required valid 'n_heavy': {item.get('n_heavy')}")
            n_heavy = int(item["n_heavy"])

            for arm in arms:
                units.append(
                    SolveGroupUnit(
                        code=c,
                        shard=shard,
                        dataset=dataset,
                        arm=str(arm),
                        seeds=seeds_list,
                        stratum=stratum,
                        split=split_label,
                        n_heavy=int(n_heavy),
                        group_id=group_id,
                    )
                )

        # Order by arm, then heavy atoms (O5 arm grouping & VRAM-aware bounded allocation)
        units.sort(key=lambda u: (u.arm, u.n_heavy, u.code))
        return units

    def determine_eval_mace(self, unit: SolveGroupUnit) -> bool:
        """Directive T1: determine whether MACE should be evaluated based on split."""
        if not self.eval_mace:
            return False
        if unit.split == "surface_fit":
            return False
        elif unit.split in ("calibration", "motion_validation"):
            return True
        return True

    def is_memory_sufficient(self, min_ram_gb: float = 4.0, min_vram_gb: float = 2.0) -> Tuple[bool, int, Optional[int]]:
        min_ram_bytes = int(min_ram_gb * (1024**3))
        min_vram_bytes = int(min_vram_gb * (1024**3))
        if self.memory_check_fn is not None:
            ram_avail, vram_avail = self.memory_check_fn()
        else:
            ram_avail = psutil.virtual_memory().available
            vram_avail = None
            if torch.cuda.is_available():
                try:
                    free_b, _ = torch.cuda.mem_get_info()
                    vram_avail = free_b
                except Exception:
                    pass
        ram_ok = (ram_avail >= min_ram_bytes)
        vram_ok = (vram_avail is None) or (vram_avail >= min_vram_bytes)
        return (ram_ok and vram_ok), ram_avail, vram_avail

    def check_memory_guard(
        self,
        min_ram_gb: float = 4.0,
        min_vram_gb: float = 2.0,
        sleep_s: float = 1.0,
    ) -> None:
        """Directive T2: Memory guard (15 GB host).
        If available RAM < 4.0 GB or available VRAM < 2.0 GB, wait/sleep in a loop until available.
        """
        while True:
            ok, ram_avail, vram_avail = self.is_memory_sufficient(min_ram_gb, min_vram_gb)
            if ok:
                break
            ram_gb = ram_avail / (1024**3)
            vram_str = f"{vram_avail / (1024**3):.2f}GB" if vram_avail is not None else "N/A"
            print(
                f"[MemoryGuard] Available RAM={ram_gb:.2f}GB (min {min_ram_gb}GB), "
                f"VRAM={vram_str} (min {min_vram_gb}GB). Waiting {sleep_s}s for resources..."
            )
            time.sleep(sleep_s)

    def _create_controller(self, mem_fraction: Optional[float] = None) -> WorkerProcessController:
        return WorkerProcessController(
            device=self.device,
            solve_device=self.solve_device,
            eval_mace=self.eval_mace,
            constraints_db_path=self.constraints_db_path,
            mace_model_path=self.mace_model_path,
            repo_root=REPO_ROOT,
            mem_fraction=mem_fraction,
        )

    def _solve_unit_on_worker(
        self,
        controller: WorkerProcessController,
        unit: SolveGroupUnit,
        arm_prov: Dict[str, Any],
        timeout: float,
        eval_mace: bool,
    ) -> Tuple[Dict[int, Dict[str, Any]], float]:
        t0 = time.perf_counter()
        outcomes = controller.execute_solve_group(
            unit=unit,
            arm_provenance=arm_prov,
            out_dir=self.out_dir,
            timeout=timeout,
            eval_mace=eval_mace,
        )
        wall_s = time.perf_counter() - t0
        return outcomes, wall_s

    def run(
        self,
        units: Sequence[Any],
        controller: Optional[WorkerProcessController] = None,
    ) -> Dict[str, Any]:
        """Execute the group task queue with spawned worker controllers and group-level reserves."""
        # Convert any SolveTask to SolveGroupUnit for backward compatibility
        norm_units: List[SolveGroupUnit] = []
        for u in units:
            if isinstance(u, SolveGroupUnit):
                norm_units.append(u)
            elif isinstance(u, SolveTask):
                norm_units.append(
                    SolveGroupUnit(
                        code=u.code,
                        shard=u.shard,
                        dataset=u.dataset,
                        arm=u.arm,
                        seeds=[u.seed],
                        stratum=u.stratum,
                        split=u.split,
                        n_heavy=u.n_heavy,
                        group_id=u.group_id,
                        run_id=u.run_id,
                    )
                )
            else:
                norm_units.append(u)

        # Certification check for Step 2–6 cohorts (O2)
        production_cohorts = {"surface_fit", "calibration", "motion_validation", "main"}
        unit_splits = {u.split for u in norm_units}
        if (unit_splits & production_cohorts) and self.solve_device != "cuda" and not self.research_arm:
            raise ValueError(
                f"Certified solve configuration for production cohorts requires solve_device='cuda'. "
                f"Pass --research-arm to override for experimental runs."
            )

        if controller is not None:
            controllers = [controller]
        else:
            num_workers = self.workers
            if self.mem_fraction is not None:
                mem_frac = float(self.mem_fraction)
            else:
                mem_frac = min(0.70, max(0.28, 0.90 / max(1, num_workers)))
            controllers = [self._create_controller(mem_fraction=mem_frac) for _ in range(num_workers)]

        manifest_path = self.out_dir / "manifest.json"
        t_overall_start = time.perf_counter()
        task_outcomes: Dict[str, Dict[str, Any]] = {}
        peak_vram_run = 0.0
        peak_rss_run = 0.0

        # Stratum accounting initialization (Directive F1)
        self.per_stratum_timeouts.clear()
        self.total_timeouts = 0
        self.stratum_total_units.clear()
        self.stratum_processed_units.clear()
        for u in norm_units:
            self.stratum_total_units[u.stratum] += 1

        pending_units: List[SolveGroupUnit] = list(norm_units)
        next_unit_idx = 0
        completed_units_count = 0
        idle_workers = list(range(len(controllers)))
        active_futures: Dict[concurrent.futures.Future, Tuple[int, SolveGroupUnit]] = {}

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(controllers)) as executor:
                while next_unit_idx < len(pending_units) or active_futures:
                    while idle_workers and next_unit_idx < len(pending_units):
                        unit = pending_units[next_unit_idx]
                        arm_prov = self.get_arm_provenance(unit.arm)

                        # Resumable skip check: check all seeds
                        all_cached = True
                        cached_seed_summaries: Dict[int, Dict[str, Any]] = {}
                        if not self.force_recompute:
                            for s in unit.seeds:
                                paths = get_task_paths(self.out_dir, unit.code, s, unit.arm)
                                if should_skip_task(paths, arm_prov):
                                    cached_seed_summaries[s] = json.loads(paths.summary_json.read_text(encoding="utf-8"))
                                else:
                                    all_cached = False
                                    break
                        else:
                            all_cached = False

                        if all_cached:
                            next_unit_idx += 1
                            for s, summ in cached_seed_summaries.items():
                                task_key = f"{unit.code}:{s}:{unit.arm}"
                                task_outcomes[task_key] = summ
                            completed_units_count += 1
                            self.stratum_processed_units[unit.stratum] += 1
                            pct = (completed_units_count / len(pending_units)) * 100.0
                            print(
                                f"[{completed_units_count}/{len(pending_units)}] ({pct:5.1f}%) [CACHED] {unit.code} "
                                f"(arm={unit.arm}, seeds={unit.seeds}) - Provenance matched -> skipped"
                            )
                            continue

                        # Memory guard before assigning unit to worker (Directive T2)
                        ok, ram_avail, vram_avail = self.is_memory_sufficient()
                        if not ok:
                            if active_futures:
                                break
                            else:
                                self.check_memory_guard()

                        next_unit_idx += 1
                        # Pick idle worker (prefer one with active_arm matching unit.arm)
                        best_w = None
                        for w in idle_workers:
                            if controllers[w].active_arm == unit.arm:
                                best_w = w
                                break
                        if best_w is None:
                            best_w = idle_workers[0]
                        idle_workers.remove(best_w)

                        unit_eval_mace = self.determine_eval_mace(unit)
                        fut = executor.submit(
                            self._solve_unit_on_worker,
                            controllers[best_w],
                            unit,
                            arm_prov,
                            self.timeout,
                            unit_eval_mace,
                        )
                        active_futures[fut] = (best_w, unit)

                    if not active_futures:
                        break

                    done, _ = concurrent.futures.wait(
                        active_futures.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for fut in done:
                        w_idx, unit = active_futures.pop(fut)
                        idle_workers.append(w_idx)

                        outcomes, unit_wall = fut.result()
                        completed_units_count += 1
                        self.stratum_processed_units[unit.stratum] += 1

                        group_succeeded = True
                        failed_reason = ""
                        had_timeout = False

                        for s, summ in outcomes.items():
                            task_key = f"{unit.code}:{s}:{unit.arm}"
                            task_outcomes[task_key] = summ
                            peak_vram_run = max(peak_vram_run, summ.get("peak_gpu_mb", 0.0))
                            peak_rss_run = max(peak_rss_run, summ.get("peak_rss_mb", 0.0))
                            if summ.get("status") != "success":
                                group_succeeded = False
                                failed_reason = summ.get("error", "solve_failed")
                            if summ.get("status") == "timeout":
                                had_timeout = True

                        # Stratum accounting and timeout stop rule (Directive F1)
                        if had_timeout:
                            self.per_stratum_timeouts[unit.stratum] += 1
                            self.total_timeouts += 1
                            tot_units = max(self.stratum_total_units.get(unit.stratum, 0), self.stratum_processed_units[unit.stratum])
                            timeout_rate = self.per_stratum_timeouts[unit.stratum] / tot_units
                            if timeout_rate > 0.02:
                                for f in active_futures:
                                    f.cancel()
                                for c in controllers:
                                    c.shutdown_worker()
                                raise RuntimeError(
                                    f"Stratum '{unit.stratum}' timeout rate exceeds 2% "
                                    f"({self.per_stratum_timeouts[unit.stratum]}/{tot_units} = {timeout_rate:.2%}). "
                                    f"Run stopped for review. Never replace timeouts silently."
                                )

                        # Group-level reserve replacement (O5: control arm failures in surface_fit ONLY, never timeouts)
                        if not group_succeeded and unit.arm == "control" and unit.split == "surface_fit" and not had_timeout:
                            cand = self.draw_reserve_replacement(
                                failed_task=unit.code,
                                reason=failed_reason,
                                stratum=unit.stratum,
                                split=unit.split,
                            )
                            if cand:
                                rep_code = cand.get("representative_code")
                                rep_shard = cand.get("representative_shard") or "unknown"
                                rep_dataset = cand.get("dataset") or "unknown"
                                rep_stratum = cand.get("stratum") or unit.stratum
                                rep_heavy = cand.get("n_heavy", unit.n_heavy)

                                distinct_arms = sorted(list({u.arm for u in norm_units}))
                                for a in distinct_arms:
                                    rep_unit = SolveGroupUnit(
                                        code=rep_code,
                                        shard=rep_shard,
                                        dataset=rep_dataset,
                                        arm=a,
                                        seeds=unit.seeds,
                                        stratum=rep_stratum,
                                        split="surface_fit",
                                        n_heavy=rep_heavy,
                                    )
                                    pending_units.append(rep_unit)
                                    self.stratum_total_units[rep_stratum] += 1
                                print(
                                    f"  -> Group reserve replacement enqueued: {unit.code} (control failed) -> {rep_code} "
                                    f"(stratum={rep_stratum}, predeclared order, enqueued for all arms: {distinct_arms})"
                                )

                        # Progress & observability
                        elapsed = time.perf_counter() - t_overall_start
                        done_count = completed_units_count
                        rem_count = max(0, len(pending_units) - done_count)
                        avg_per_unit = elapsed / max(1, done_count)
                        eta_s = (rem_count * avg_per_unit) / max(1, len(controllers))
                        eta_str = str(datetime.timedelta(seconds=int(eta_s)))
                        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
                        pct = (done_count / len(pending_units)) * 100.0

                        first_summ = next(iter(outcomes.values())) if outcomes else {}
                        vram_mb = first_summ.get("peak_gpu_mb", 0.0)
                        rss_mb = first_summ.get("peak_rss_mb", 0.0)
                        status_tag = "SUCCESS" if group_succeeded else ("TIMEOUT" if had_timeout else "FAILED")

                        print(
                            f"[{done_count}/{len(pending_units)}] ({pct:5.1f}%) [{status_tag}] {unit.code} "
                            f"(arm={unit.arm}, seeds={unit.seeds}, n_heavy={unit.n_heavy}, worker={w_idx}) "
                            f"wall={unit_wall:.2f}s | VRAM: {vram_mb:.1f}MB | RSS: {rss_mb:.1f}MB "
                            f"| Elapsed: {elapsed_str} | ETA: {eta_str}"
                        )
        finally:
            for c in controllers:
                c.shutdown_worker()

        total_wall = time.perf_counter() - t_overall_start
        succeeded = sum(1 for o in task_outcomes.values() if o.get("status") == "success")
        failed = len(task_outcomes) - succeeded

        manifest: Dict[str, Any] = {
            "run_id": f"orchestrator_{int(time.time())}",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_tasks": len(task_outcomes),
            "completed_tasks": len(task_outcomes),
            "succeeded_tasks": succeeded,
            "failed_tasks": failed,
            "total_timeouts": self.total_timeouts,
            "per_stratum_timeouts": dict(self.per_stratum_timeouts),
            "total_wall_s": total_wall,
            "peak_vram_mb": peak_vram_run,
            "peak_rss_mb": peak_rss_run,
            "base_provenance": self.base_provenance,
            "arm_fingerprints": {a: p.get("profile_fingerprint") for a, p in self.arm_provenance.items()},
            "replaced_failures": self.replaced_failures,
            "exhausted_strata": sorted(list(self.exhausted_strata)),
            "tasks": task_outcomes,
        }
        write_json_atomic(manifest_path, manifest)
        return manifest


def run_orchestrator(
    spec_path: Path = DEFAULT_SPEC,
    frame_path: Path = DEFAULT_FRAME,
    constraints_db: Path = CONSTRAINTS_DB_PATH,
    mace_model: Path = MACE_MODEL_PATH,
    cohort: str = "pilot",
    codes: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (11, 12, 13),
    arms: Sequence[str] = ("control",),
    out_dir: Path = REPO_ROOT / "data/refit/nci_v2/orchestrator_runs",
    device: str = "cuda",
    solve_device: str = "cuda",
    research_arm: bool = False,
    eval_mace: bool = True,
    timeout: float = 1800.0,
    batch_size: int = 16,
    force_recompute: bool = False,
    workers: int = 2,
    mem_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """Top-level invocation entrypoint."""
    orch = ProductionOrchestrator(
        spec_path=spec_path,
        frame_path=frame_path,
        constraints_db_path=constraints_db,
        mace_model_path=mace_model,
        out_dir=out_dir,
        device=device,
        solve_device=solve_device,
        research_arm=research_arm,
        eval_mace=eval_mace,
        timeout=timeout,
        batch_size=batch_size,
        force_recompute=force_recompute,
        workers=workers,
        mem_fraction=mem_fraction,
    )
    units = orch.build_task_queue(cohort=cohort, codes=codes, seeds=seeds, arms=arms)
    print(f"Orchestrator: Queued {len(units)} group units (cohort={cohort}, seeds={seeds}, arms={arms}, workers={orch.workers}).")
    manifest = orch.run(units)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Jacobik NCI v2 Production Solve Orchestrator (Gate G5)")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Path to spec.json")
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME, help="Path to frame.json")
    parser.add_argument("--constraints-db", type=Path, default=CONSTRAINTS_DB_PATH, help="Path to constraints.json")
    parser.add_argument("--mace-model", type=Path, default=MACE_MODEL_PATH, help="Path to MACE model file")
    parser.add_argument(
        "--cohort",
        type=str,
        default="pilot",
        choices=("pilot", "surface_fit", "calibration", "motion_validation", "main", "all", "custom"),
    )
    parser.add_argument("--codes", type=str, default="", help="Comma-separated list of CSD codes")
    parser.add_argument("--seeds", type=str, default="11,12,13", help="Comma-separated integer seeds")
    parser.add_argument("--arms", type=str, default="control", help="Comma-separated arm names (control, incumbent, candidate)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/refit/nci_v2/orchestrator_runs")
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"), help="device for MACE model")
    parser.add_argument("--solve-device", type=str, default="cuda", choices=("cuda", "cpu"), help="device for complex builder solve (O2 certified)")
    parser.add_argument("--research-arm", action="store_true", help="Allow non-certified solve configuration for experimental arms")
    parser.add_argument("--no-mace", action="store_true", help="Disable MACE energy evaluation")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Per-solve group timeout in seconds (O4)")
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2, 3), help="Number of spawned worker processes (1-3, default: 2)")
    parser.add_argument("--mem-fraction", type=float, default=None, help="VRAM memory fraction for worker process")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for queue chunking")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore valid cached runs and force re-solve")
    parser.add_argument("--dry-run", action="store_true", help="Print task queue and exit")

    # Internal worker protocol arguments
    parser.add_argument("--worker-server", action="store_true", help="Internal flag: run as worker server")
    parser.add_argument("--worker-arm", type=str, default="control", help="Arm configuration for worker server")

    args = parser.parse_args()

    if args.worker_server:
        worker_server_main(
            arm=args.worker_arm,
            device=args.device,
            solve_device=args.solve_device,
            eval_mace=not args.no_mace,
            constraints_db_path=args.constraints_db,
            mace_model_path=args.mace_model,
            mem_fraction=args.mem_fraction,
        )
        return

    codes_list = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    arms_list = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.dry_run:
        orch = ProductionOrchestrator(
            spec_path=args.spec,
            frame_path=args.frame,
            constraints_db_path=args.constraints_db,
            mace_model_path=args.mace_model,
            out_dir=args.out_dir,
            device=args.device,
            solve_device=args.solve_device,
            research_arm=args.research_arm,
            eval_mace=not args.no_mace,
            timeout=args.timeout,
            batch_size=args.batch_size,
            force_recompute=args.force_recompute,
            workers=args.workers,
            mem_fraction=args.mem_fraction,
        )
        units = orch.build_task_queue(cohort=args.cohort, codes=codes_list, seeds=seeds_list, arms=arms_list)
        print(f"Dry-run: {len(units)} tasks queued:")
        for u in units[:15]:
            print(f"  {u.code} arm={u.arm} seeds={u.seeds} stratum={u.stratum} split={u.split} n_heavy={u.n_heavy}")
        if len(units) > 15:
            print(f"  ... and {len(units) - 15} more.")
        return

    manifest = run_orchestrator(
        spec_path=args.spec,
        frame_path=args.frame,
        constraints_db=args.constraints_db,
        mace_model=args.mace_model,
        cohort=args.cohort,
        codes=codes_list,
        seeds=seeds_list,
        arms=arms_list,
        out_dir=args.out_dir,
        device=args.device,
        solve_device=args.solve_device,
        research_arm=args.research_arm,
        eval_mace=not args.no_mace,
        timeout=args.timeout,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
        workers=args.workers,
        mem_fraction=args.mem_fraction,
    )
    print(f"\nExecution finished. Manifest written to {args.out_dir / 'manifest.json'}")
    print(f"Summary: Succeeded {manifest['succeeded_tasks']} / {manifest['total_tasks']} tasks.")


if __name__ == "__main__":
    main()
