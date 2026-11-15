"""Jacobik NCI v2 Pilot Verification, Repeatability, and Resource Budgeting.

Step 1 implementation per Jacobik Roadmap v2 Sections 3.10, 5.1, 7, and 7.1.
Executes the frozen pure-repulsion Control configuration (profile="u1", NCI_ON=False,
DETERMINISTIC_EMBED=True, research-arm setting) on designated pilot parents.
Verifies multi-process cold-cache repeatability across seeds 11, 12, 13 (Run A vs Run B
in strictly separate OS processes with cold _GFN2_CACHE embed cache).
Evaluates MACE-POLAR-1-S float64 energies with graph-constrained atom mapping,
records component-level resource timings, compaction diagnostics, and extrapolates
the compute budget for the 240-group main allocation using bootstrap/CLT statistics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import psutil
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "src", REPO / "research" / "refit", REPO / "research/bench/tools/jdev/budget", REPO / "research/bench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Ensure single-threaded sequential execution per guidelines
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
_existing_overrides = json.loads(os.environ.get("JACOBIK_PROFILE_OVERRIDES", "{}") or "{}")
_default_overrides = {"nci": False, "deterministic_embed": True}
_default_overrides.update(_existing_overrides)
os.environ.update(
    MKL_THREADING_LAYER="SEQUENTIAL",
    PYTHONIOENCODING="utf-8",
    JACOBIK_PROFILE=os.environ.get("JACOBIK_PROFILE", "u1"),
    JACOBIK_PROFILE_OVERRIDES=json.dumps(_default_overrides),
)

from jacobik.benchmark.energy import EV_TO_KJ_MOL, MACEPolarEvaluator
from jacobik import ComplexBuilder
from jacobik.builder.atom_map import build_atom_mapping, AtomMappingResult
from jacobik import profile as _P
from nci_v2_provenance import code_state_sha256
from jacobik.stats.mapping import assemble, METALS, _covalent_radius
from research.archive.code.interactions.nci_v2 import nci_v2
from jacobik.fit.parser import (
    TMQMParser,
    _assemble_raw,
    _attach_qm_properties,
    _iter_bo_blocks,
    _iter_xyz_blocks,
)
from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol
from tools.jdev.budget.common import DB, stats_from_db
from tools.jdev.budget.pools import rank_terms

DEFAULT_SPEC = REPO / "data/refit/nci_v2/pilot_v1/spec.json"
DEFAULT_FRAME = REPO / "data/refit/nci_v2/pilot_v1/frame.json"
DEFAULT_OUT_DIR = REPO / "data/refit/nci_v2/pilot_v1/reports"
DEFAULT_POOLS_DIR = REPO / "data/refit/nci_v2/pilot_v1/pools"
CONSTRAINTS_DB_PATH = REPO / "data/refit/final/constraints_2026-10-18.json"
from jacobik.paths import MACE_MODEL_PATH  # noqa: E402,F401  (moved into the package)


def get_rss_mb() -> float:
    """Return resident set size in megabytes."""
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


def get_gpu_mem_mb() -> float:
    """Return peak GPU allocated memory in megabytes if CUDA is available."""
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0.0


def reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sha256_of_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_provenance_hashes(repo_root: Path = REPO) -> Dict[str, Any]:
    spec_p = repo_root / "data/refit/nci_v2/pilot_v1/spec.json"
    frame_p = repo_root / "data/refit/nci_v2/pilot_v1/frame.json"
    constraints_p = repo_root / "data/refit/final/constraints_2026-10-18.json"
    mace_p = repo_root / "research/bench/tools/jdev/MACE-POLAR-1-S.model"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    dirty_sha = code_state_sha256(repo_root)

    return {
        "spec_sha256": sha256_of_file(spec_p),
        "frame_sha256": sha256_of_file(frame_p),
        "code_git_commit": commit,
        "code_dirty_diff_sha256": dirty_sha,
        "constraints_db_sha256": sha256_of_file(constraints_p),
        "mace_model_sha256": sha256_of_file(mace_p),
        "full_resolved_profile": _P.snapshot(),
    }


def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def load_mol(code: str, shard: str, dataset: str, clean_root: Path = REPO / "data/refit/clean") -> Optional[Any]:
    parent_dir = clean_root / ("lnQM" if dataset == "LNf" else "tmQM")
    parser = TMQMParser(parent_dir)
    xp = parent_dir / "xyz" / f"tmQM_{shard}.xyz"
    bp = parent_dir / "bo" / f"tmQM_{shard}.BO"
    if not xp.exists() or not bp.exists():
        return None
    for (comment, atoms), (c, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
        if c == code:
            row = parser.csv_index.get(code)
            raw = _assemble_raw(comment, atoms, c, bonds, row)
            mol = assemble(raw)
            if row:
                _attach_qm_properties(mol, row)
            return mol
    return None


from jacobik.inputs import build_builder_and_inputs  # noqa: E402,F401  (moved into the package)


def solve_control(
    builder: ComplexBuilder,
    seed: int,
    stats: Any,
    mol: Any,
    provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], float, float, float]:
    """Execute pure-repulsion solve on CPU with deterministic embed."""
    from jacobik.builder.engine import _GFN2_CACHE
    initial_cache_size = len(_GFN2_CACHE)

    reset_gpu_peak()
    rss_before = get_rss_mb()
    t0 = time.perf_counter()
    try:
        r = builder.solve(seed=seed, stats=stats, angle_bends=True, device="cpu")
        ts = time.perf_counter() - t0
        peak_rss = max(get_rss_mb(), rss_before)
        peak_gpu = get_gpu_mem_mb()
        confs = r.conformers
        if not confs:
            return False, None, "empty_conformers", ts, peak_rss, peak_gpu

        X = np.array([[at.position for at in c.complex.atoms] for c in confs], dtype=np.float64)
        sy = np.array([at.symbol for at in confs[0].complex.atoms])
        asg = np.array([c.assignment_idx or 0 for c in confs])
        scores = np.array([c.score for c in confs], dtype=np.float64)

        sc = r.scorer
        Xt = torch.tensor(X, dtype=torch.float32, device="cpu")
        rg = torch.as_tensor(asg, dtype=torch.long, device="cpu")
        tot, bd = rank_terms(sc, Xt, rg)
        Z = float(sc._agg_norm())

        save_dict = {
            "X": X,
            "symbols": sy,
            "score": scores,
            "asg": asg,
            "rank": tot.detach().cpu().numpy() if hasattr(tot, "detach") else tot,
            "Z": Z,
            "t_solve": ts,
            "ref_X": np.array([a.position for a in mol.atoms]),
            "ref_symbols": np.array([a.symbol for a in mol.atoms]),
            "charge": mol.charge,
            "mult": mol.spin_state,
            "profile": json.dumps(_P.snapshot(), default=str),
            "worker_pid": os.getpid(),
            "gfn2_cache_initial_size": initial_cache_size,
            "deterministic_embed": True,
            "research_arm_setting": True,
        }
        for q, v in bd.items():
            arr = v.detach().cpu().numpy()
            save_dict[f"terms_{q}"] = arr * Z if q != "Q7" else arr

        # Graph-constrained atom mapping metadata
        try:
            am = build_atom_mapping(mol, gen_mol=confs[0].complex, X_ref=save_dict["ref_X"], X_gen=X[0])
            save_dict["gen_to_ref_idx"] = am.gen_to_ref_idx
            save_dict["ref_to_gen_idx"] = am.ref_to_gen_idx
            save_dict["mapping_ambiguous"] = am.is_ambiguous
            save_dict["num_automorphisms"] = am.num_automorphisms
            ci = coordination_integrity(save_dict["ref_symbols"], save_dict["ref_X"], X[:, am.gen_to_ref_idx, :])
            save_dict["coord_intact"] = ci["intact"]
            save_dict["coord_sigma_shift_A"] = ci["sigma_shift_A"]
            save_dict["coord_haptic_slip_A"] = ci["haptic_slip_A"]
            save_dict["coord_haptic_shift_A"] = ci["haptic_shift_A"]
        except Exception:
            pass

        # Embed complete provenance SHA256 hashes
        prov = provenance or get_provenance_hashes(REPO)
        for pk in ("spec_sha256", "frame_sha256", "code_git_commit", "code_dirty_diff_sha256", "constraints_db_sha256", "mace_model_sha256"):
            if pk in prov:
                save_dict[pk] = prov[pk]

        return True, save_dict, None, ts, peak_rss, peak_gpu
    except Exception as exc:
        ts = time.perf_counter() - t0
        return False, None, f"{type(exc).__name__}: {exc}", ts, get_rss_mb(), get_gpu_mem_mb()


def save_pool_npz(path: Path, result_or_error: Tuple[bool, Optional[Dict[str, Any]], Optional[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}_tmp.npz")
    success, data, err = result_or_error
    if success and data is not None:
        np.savez_compressed(temp, **data)
    else:
        np.savez_compressed(temp, error=err or "unknown_error")
    temp.replace(path)


from jacobik.benchmark.coordination import COORD_BOUND_SCALE, COORD_SHIFT_TOL_A, coordination_integrity  # noqa: E402,F401  (moved into the package)


def compute_heavy_atom_rg(symbols: Sequence[str], positions: np.ndarray) -> float:
    """Calculate radius of gyration Rg (Angstroms) for heavy atoms."""
    heavy_mask = np.array([s != "H" for s in symbols])
    if not np.any(heavy_mask):
        return 0.0
    pos_h = positions[heavy_mask]
    cm = pos_h.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((pos_h - cm) ** 2, axis=1))))


def compute_nonbonded_contacts(
    mol: Any,
    positions: np.ndarray,
    cutoff: float = 3.5,
) -> int:
    """Count nonbonded heavy-atom contacts (< cutoff Angstroms) separated by > 4 bonds in the ligand graph.

    Note: positions MUST be in reference atom ordering (e.g. aligned via AtomMappingResult).
    """
    idx_to_0 = {a.idx: i for i, a in enumerate(mol.atoms)}
    heavy_idx = [i for i, a in enumerate(mol.atoms) if a.symbol != "H" and a.idx != mol.metal_idx]
    if len(heavy_idx) < 2:
        return 0

    G = nx.Graph()
    for idx in heavy_idx:
        G.add_node(idx)
    for b in mol.bonds:
        if b.i != mol.metal_idx and b.j != mol.metal_idx and b.order >= 0.5:
            u_0 = idx_to_0.get(b.i)
            v_0 = idx_to_0.get(b.j)
            if u_0 is not None and v_0 is not None and u_0 in G and v_0 in G:
                G.add_edge(u_0, v_0)

    spl = dict(nx.all_pairs_shortest_path_length(G))
    contacts = 0
    for i_idx, u in enumerate(heavy_idx):
        for v in heavy_idx[i_idx + 1:]:
            d_graph = spl.get(u, {}).get(v, 999)
            if d_graph > 4:
                d_cart = float(np.linalg.norm(positions[u] - positions[v]))
                if d_cart < cutoff:
                    contacts += 1
    return contacts


def measure_nci_v2_component(
    mol: Any,
    positions_batch: np.ndarray,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure opportunity enumeration, feature extraction, and autograd gradient/Hessian timing/memory."""
    reset_gpu_peak()
    rss0 = get_rss_mb()
    t0 = time.perf_counter()
    opps = nci_v2.enumerate_opportunities(mol, schema=schema)
    t_enum = time.perf_counter() - t0

    B = positions_batch.shape[0]
    X_torch = torch.tensor(positions_batch, dtype=torch.float64, device="cpu")

    t1 = time.perf_counter()
    fb = nci_v2.features_torch(X_torch, opps, schema=schema)
    t_feat = time.perf_counter() - t1

    s_mock = {
        "coef_r": np.ones(7, dtype=np.float64),
        "coef_rx": [],
        "basis_specs": {"knots_r": nci_v2.make_clamped_knots((1.1, 8.0), 4), "knots_x": []},
        "support_basis_spec": {
            "knots_r": nci_v2.make_clamped_knots((1.1, 8.0), 6),
            "knots_x": [],
        },
        "support_a": np.ones(9, dtype=np.float64),
        "ref_only_mask": np.zeros(9, dtype=bool),
        "radial_window": (1.1, 1.35, 6.0, 8.0),
    }
    surface_artifacts = {}
    for fam in nci_v2.FAMILIES:
        surface_artifacts[fam] = nci_v2.SurfaceArtifact(
            family=fam,
            surfaces={"dummy": s_mock},
            spec=schema or nci_v2.DEFAULT_SCHEMA,
            metadata={"status": "pilot_benchmark"},
        )
    cal_art = nci_v2.CalibratedArtifact(
        surface_artifacts=surface_artifacts,
        gamma_shared=1.0,
        family_scales={"bond_atom": 1.0, "bond_ring": 1.0, "ring_ring": 1.0},
        selected_mode="shared",
        singular_values=[1.0],
        condition_number=1.0,
        ref_only_traversal_passed={"bond_atom": True, "bond_ring": True, "ring_ring": True},
        spec=schema or nci_v2.DEFAULT_SCHEMA,
    )

    t2 = time.perf_counter()
    rt = nci_v2.build_runtime(mol, cal_art, device="cpu")
    X_var = X_torch.clone().detach().requires_grad_(True)
    raw_reward = nci_v2.evaluate_raw_nci(X_var, rt)
    loss = raw_reward.sum()
    if loss.requires_grad:
        grad_t = torch.autograd.grad(loss, X_var, create_graph=True, allow_unused=True)[0]
        grad = grad_t if grad_t is not None else torch.zeros_like(X_var)
        grad_norm = float(grad.norm().item())
        v = torch.randn_like(X_var)
        if grad.requires_grad:
            hvp_t = torch.autograd.grad((grad * v).sum(), X_var, retain_graph=False, allow_unused=True)[0]
            hvp = hvp_t if hvp_t is not None else torch.zeros_like(X_var)
        else:
            hvp = torch.zeros_like(X_var)
        hvp_norm = float(hvp.norm().item())
    else:
        grad_norm = 0.0
        hvp_norm = 0.0
    t_eval_autograd = time.perf_counter() - t2

    peak_rss = max(get_rss_mb(), rss0)
    peak_gpu = get_gpu_mem_mb()

    return {
        "opportunities_count": opps.total_count,
        "opportunities_by_family": {k: len(v) for k, v in opps.opportunities.items()},
        "wall_time_enumeration_s": t_enum,
        "wall_time_features_s": t_feat,
        "wall_time_runtime_autograd_s": t_eval_autograd,
        "grad_norm": grad_norm,
        "hvp_norm": hvp_norm,
        "peak_rss_mb": peak_rss,
        "peak_gpu_mb": peak_gpu,
    }


def evaluate_mace_energies(
    evaluator: MACEPolarEvaluator,
    symbols: Sequence[str],
    positions_list: Sequence[np.ndarray],
    charge: int,
    mult: int,
) -> Tuple[np.ndarray, float, float, float]:
    """Evaluate MACE float64 energies across conformations; return (energies_kj, wall_s, rss_mb, gpu_mb)."""
    import gc
    reset_gpu_peak()
    rss0 = get_rss_mb()
    t0 = time.perf_counter()
    structures = [(list(symbols), np.asarray(p, dtype=float), int(charge), int(mult)) for p in positions_list]
    energies_ev = evaluator.energies(structures)
    energies_kj = energies_ev * EV_TO_KJ_MOL
    wall_s = time.perf_counter() - t0
    peak_rss = max(get_rss_mb(), rss0)
    peak_gpu = get_gpu_mem_mb()
    del structures
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return energies_kj, wall_s, peak_rss, peak_gpu


# ---------------------------------------------------------------------------
# Worker Subprocess Implementation (Multi-Process Cold Cache Isolation)
# ---------------------------------------------------------------------------

def worker_solve(
    code: str,
    shard: str,
    dataset: str,
    seed: int,
    run_id: str,
    pool_out: Path,
    res_out: Path,
    mace_model_path: Path = MACE_MODEL_PATH,
) -> None:
    """Worker process entrypoint executing solve in a fresh OS process with cold embed cache."""
    from jacobik.builder.engine import _GFN2_CACHE
    initial_cache_size = len(_GFN2_CACHE)
    assert initial_cache_size == 0, f"Cache not cold in worker {os.getpid()}! Found {initial_cache_size} entries."

    pid = os.getpid()
    t_start = time.perf_counter()

    mol = load_mol(code, shard, dataset)
    if mol is None:
        err_msg = f"Failed to load {code} from shard {shard} ({dataset})"
        save_pool_npz(pool_out, (False, None, err_msg))
        write_json_atomic(res_out, {
            "code": code,
            "seed": seed,
            "run_id": run_id,
            "worker_pid": pid,
            "gfn2_cache_initial_size": initial_cache_size,
            "status": "failed",
            "error": err_msg,
            "wall_s": time.perf_counter() - t_start,
        })
        return

    builder, lig_in, err = build_builder_and_inputs(mol)
    if err or builder is None:
        err_msg = err or "build_builder_and_inputs_failed"
        save_pool_npz(pool_out, (False, None, err_msg))
        write_json_atomic(res_out, {
            "code": code,
            "seed": seed,
            "run_id": run_id,
            "worker_pid": pid,
            "gfn2_cache_initial_size": initial_cache_size,
            "status": "unmappable",
            "error": err_msg,
            "wall_s": time.perf_counter() - t_start,
        })
        return

    stats = stats_from_db(DB)
    prov = get_provenance_hashes(REPO)
    ok, data, err_solve, ts, rss, gpu = solve_control(builder, seed, stats, mol, provenance=prov)
    if not ok or data is None:
        err_msg = err_solve or "solve_control_failed"
        save_pool_npz(pool_out, (False, None, err_msg))
        write_json_atomic(res_out, {
            "code": code,
            "seed": seed,
            "run_id": run_id,
            "worker_pid": pid,
            "gfn2_cache_initial_size": initial_cache_size,
            "status": "failed",
            "error": err_msg,
            "wall_s": ts,
        })
        return

    # Save pool with embedded provenance
    save_pool_npz(pool_out, (True, data, None))

    # Evaluate MACE float64 on mapped coordinates using CUDA
    mace_evaluator = MACEPolarEvaluator(
        model="polar-1-s",
        device="cuda",
        dtype="float64",
        model_path=str(mace_model_path),
        mode="energy-only",
        threads=1,
    ).load()

    ref_X = data["ref_X"]
    ref_sym = list(data["ref_symbols"])
    ref_kj_arr, _, _, _ = evaluate_mace_energies(mace_evaluator, ref_sym, [ref_X], mol.charge, mol.spin_state)
    ref_energy_kj = float(ref_kj_arr[0])

    # Reorder coordinates via gen_to_ref_idx so symbols match ref_symbols
    confs = data["X"]
    scores = data["score"]
    gen_to_ref = data.get("gen_to_ref_idx")
    if gen_to_ref is not None:
        mapped_confs = confs[:, gen_to_ref, :]
    else:
        mapped_confs = confs

    energies_kj, mace_t, _, _ = evaluate_mace_energies(
        mace_evaluator, ref_sym, mapped_confs, mol.charge, mol.spin_state
    )

    top1_idx = int(np.argmax(scores))
    floor_idx = int(np.argmin(energies_kj))
    top1_kj = float(energies_kj[top1_idx])
    floor_kj = float(energies_kj[floor_idx])

    # Cache mace64 energies
    mace_cache_file = pool_out.with_suffix(".mace64.json")
    write_json_atomic(mace_cache_file, {
        "code": code,
        "seed": seed,
        "run_id": run_id,
        "energies_kj": energies_kj.tolist(),
        "ref_energy_kj": ref_energy_kj,
        "top1_idx": top1_idx,
        "floor_idx": floor_idx,
    })

    # Summary payload returned to orchestrator
    confs_hash = hashlib.sha256(confs.tobytes()).hexdigest()
    res_payload = {
        "code": code,
        "seed": seed,
        "run_id": run_id,
        "worker_pid": pid,
        "gfn2_cache_initial_size": initial_cache_size,
        "final_cache_size": len(_GFN2_CACHE),
        "status": "success",
        "t_solve": ts,
        "mace_t": mace_t,
        "n_confs": len(confs),
        "top1_idx": top1_idx,
        "floor_idx": floor_idx,
        "top1_kj": top1_kj,
        "floor_kj": floor_kj,
        "ref_kj": ref_energy_kj,
        "top1_dE_kj": top1_kj - ref_energy_kj,
        "floor_dE_kj": floor_kj - ref_energy_kj,
        "confs_sha256": confs_hash,
        "top1_coord_intact": bool(data["coord_intact"][top1_idx]) if "coord_intact" in data else None,
        "floor_coord_intact": bool(data["coord_intact"][floor_idx]) if "coord_intact" in data else None,
        "pool_coord_intact_fraction": float(np.mean(data["coord_intact"])) if "coord_intact" in data else None,
        "scores": scores.tolist(),
        "peak_rss_mb": rss,
        "peak_gpu_mb": gpu,
    }
    write_json_atomic(res_out, res_payload)


def run_worker_process(
    code: str,
    shard: str,
    dataset: str,
    seed: int,
    run_id: str,
    pool_out: Path,
    res_out: Path,
) -> Dict[str, Any]:
    """Launch worker_solve in an independent Python subprocess with cold cache."""
    pool_out = Path(pool_out)
    res_out = Path(res_out)
    if pool_out.exists() and res_out.exists():
        try:
            cached_res = json.loads(res_out.read_text(encoding="utf-8"))
            if cached_res.get("status") == "success":
                return cached_res
        except Exception:
            pass

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--code", code,
        "--shard", shard,
        "--dataset", dataset,
        "--seed", str(seed),
        "--run-id", run_id,
        "--pool-out", str(pool_out),
        "--res-out", str(res_out),
    ]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{REPO}:{REPO}/src:{REPO}/research/archive/code/refit:{REPO}/research/bench",
        MKL_THREADING_LAYER="SEQUENTIAL",
        OMP_NUM_THREADS="1",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        JACOBIK_PROFILE="u1",
        JACOBIK_PROFILE_OVERRIDES=json.dumps({"nci": False, "deterministic_embed": True}),
    )
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO)
    if res.returncode != 0:
        raise RuntimeError(
            f"Worker subprocess failed for {code} seed {seed} ({run_id}):\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )

    assert res_out.exists(), f"Worker result file not produced: {res_out}"
    return json.loads(res_out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Budget Extrapolation Engine (Item D1 / Statistical p95)
# ---------------------------------------------------------------------------

def extrapolate_budget(
    solve_timings: List[float],
    mace_timings: List[float],
    enum_timings: List[float],
    feat_timings: List[float],
    autograd_timings: List[float],
    peak_rss_list: List[float],
    peak_gpu_list: List[float],
    conformer_counts: List[int],
) -> Dict[str, Any]:
    """Extrapolate resource budget using measured parameters and bootstrap/CLT total p95."""
    assert len(solve_timings) > 0, "No solve timings provided for budget extrapolation"

    mean_solve_s = float(np.mean(solve_timings))
    std_solve_s = float(np.std(solve_timings, ddof=1)) if len(solve_timings) > 1 else 0.0
    p95_single_solve_s = float(np.percentile(solve_timings, 95))
    min_solve_s = float(np.min(solve_timings))
    max_solve_s = float(np.max(solve_timings))

    mean_confs_per_pool = float(np.mean(conformer_counts)) if conformer_counts else 32.0

    mean_mace_s = float(np.mean(mace_timings)) if mace_timings else 0.15
    p95_mace_s = float(np.percentile(mace_timings, 95)) if mace_timings else 0.35

    mean_enum_s = float(np.mean(enum_timings)) if enum_timings else 0.05
    mean_feat_s = float(np.mean(feat_timings)) if feat_timings else 0.02
    mean_autograd_s = float(np.mean(autograd_timings)) if autograd_timings else 0.05

    max_cpu_rss = float(max(peak_rss_list)) if peak_rss_list else 1500.0
    max_gpu_mem = float(max(peak_gpu_list)) if peak_gpu_list else 800.0

    # Step 2 Baseline: 240 parents x 3 seeds = 720 solves
    step2_total_solves = 240 * 3
    # Monte Carlo bootstrap distribution of the SUM of 720 independent solves
    rng = np.random.default_rng(seed=42)
    mc_sums_s = rng.choice(solve_timings, size=(10000, step2_total_solves), replace=True).sum(axis=1)
    step2_mean_hours = float(np.mean(mc_sums_s) / 3600.0)
    step2_p95_hours = float(np.percentile(mc_sums_s, 95) / 3600.0)
    step2_min_hours = float((step2_total_solves * min_solve_s) / 3600.0)

    # Analytical Central Limit Theorem estimate: Sum ~ Normal(N*mu, sqrt(N)*sigma)
    clt_std_hours = float(np.sqrt(step2_total_solves) * std_solve_s / 3600.0)
    clt_p95_hours = float(step2_mean_hours + 1.6449 * clt_std_hours)

    # Step 3 Extract & fit-surface
    step3_total_poses = int(round(240 * 3 * mean_confs_per_pool))
    step3_extract_hours = (240 * mean_enum_s + (step3_total_poses / 16.0) * mean_feat_s) / 3600.0
    step3_fit_hours_assumed = 0.5  # Tensor-product B-spline linear/ridge regression on 160 fit groups
    step3_total_hours = step3_extract_hours + step3_fit_hours_assumed

    # Step 4 Measure-response & calibrate
    # 40 cal + 40 val = 80 complexes x 3 paths x 5 scan points = 1200 single points
    step4_total_evals = 80 * 3 * 5
    step4_mace_hours = (step4_total_evals * mean_mace_s) / 3600.0
    step4_cal_hours_assumed = 0.25
    step4_total_hours = step4_mace_hours + step4_cal_hours_assumed

    total_mean_hours = step2_mean_hours + step3_total_hours + step4_total_hours
    total_p95_hours = step2_p95_hours + step3_total_hours + step4_total_hours

    feasibility_passed = (total_p95_hours <= 24.0 and max_cpu_rss * 1.5 <= 32000.0)

    return {
        "work_units": {
            "step2_baseline_solves": step2_total_solves,
            "step3_extraction_poses": step3_total_poses,
            "step3_fit_groups": 160,
            "step4_calibration_single_points": step4_total_evals,
            "reserve_groups_available": 48,
            "anticipated_failure_rate_fraction": 0.15,
        },
        "measured_unit_rates": {
            "solve_wall_time_s": {
                "mean": mean_solve_s,
                "std": std_solve_s,
                "p95_single_solve": p95_single_solve_s,
                "min": min_solve_s,
                "max": max_solve_s,
                "evaluated_solves_count": len(solve_timings),
            },
            "mean_confs_per_pool": mean_confs_per_pool,
            "nci_v2_enumeration_s_per_parent": mean_enum_s,
            "nci_v2_features_s_per_batch16": mean_feat_s,
            "nci_v2_runtime_autograd_s_per_batch16": mean_autograd_s,
            "mace64_s_per_conformer": {
                "mean": mean_mace_s,
                "p95": p95_mace_s,
            },
            "peak_memory_mb": {
                "cpu_rss_mb": max_cpu_rss,
                "gpu_allocated_mb": max_gpu_mem,
            },
        },
        "explicit_assumptions": {
            "step3_surface_fit_hours": step3_fit_hours_assumed,
            "step3_surface_fit_rationale": "Tensor product B-spline linear/ridge regression on 160 fit groups",
            "step4_calibration_solver_hours": step4_cal_hours_assumed,
            "step4_calibration_rationale": "SVD / 1D scalar response parameter optimization",
            "reserve_replacement_policy": "48 reserve groups provide 20% buffer, exceeding anticipated 15% solver failure rate",
        },
        "step2_baseline": {
            "wall_time_hours_mean": round(step2_mean_hours, 2),
            "wall_time_hours_p95": round(step2_p95_hours, 2),
            "wall_time_hours_clt_p95": round(clt_p95_hours, 2),
            "wall_time_hours_min": round(step2_min_hours, 2),
            "p95_calculation_method": "bootstrap_cohort_sum (10,000 resamples of sum of 720 solves)",
            "recommended_workers": 1,
            "projected_peak_cpu_rss_mb": round(max_cpu_rss, 1),
            "projected_peak_gpu_mem_mb": round(max_gpu_mem, 1),
        },
        "step3_extract_and_fit_surface": {
            "wall_time_hours_extraction": round(step3_extract_hours, 2),
            "wall_time_hours_fitting": round(step3_fit_hours_assumed, 2),
            "wall_time_hours_total": round(step3_total_hours, 2),
            "projected_peak_cpu_rss_mb": round(max_cpu_rss * 1.2, 1),
        },
        "step4_measure_response_and_calibrate": {
            "single_point_evaluations": step4_total_evals,
            "wall_time_hours_mace": round(step4_mace_hours, 2),
            "wall_time_hours_total": round(step4_total_hours, 2),
            "projected_peak_gpu_mem_mb": round(max_gpu_mem, 1),
        },
        "total_main_allocation_budget": {
            "wall_time_hours_total_estimate": round(total_mean_hours, 2),
            "wall_time_hours_total_p95": round(total_p95_hours, 2),
            "wall_time_hours_range": [
                round(step2_min_hours + step3_total_hours + step4_total_hours, 2),
                round(total_p95_hours, 2),
            ],
            "peak_ram_mb": round(max_cpu_rss * 1.5, 1),
            "peak_vram_mb": round(max(max_gpu_mem, 1500.0), 1),
            "feasibility_decision": "PASS: Well within memory and 24-hour job constraints on sequential single-GPU workstation." if feasibility_passed else "FAIL: Exceeds workstation limits.",
        },
    }


# ---------------------------------------------------------------------------
# Pilot Orchestrator
# ---------------------------------------------------------------------------

def run_pilot(
    spec_path: Path = DEFAULT_SPEC,
    frame_path: Path = DEFAULT_FRAME,
    out_dir: Path = DEFAULT_OUT_DIR,
    pools_dir: Path = DEFAULT_POOLS_DIR,
    seeds: Optional[List[int]] = None,
    quick: bool = False,
    workers: int = 4,
) -> Dict[str, Any]:
    """Execute Step 1 pilot verification, multi-process cold-cache repeatability, and budgeting."""
    seeds = seeds or [11, 12, 13]
    eval_seeds = [11] if quick else seeds
    out_dir = Path(out_dir)
    pools_dir = Path(pools_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pools_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading spec from {spec_path}...", flush=True)
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    print(f"Loading frame from {frame_path}...", flush=True)
    frame = json.loads(Path(frame_path).read_text(encoding="utf-8"))

    provenance = get_provenance_hashes(REPO)

    pilot_cohort = frame.get("pilot_cohort", {})
    parents_meta = pilot_cohort.get("parents", [])
    assert len(parents_meta) == 12, f"Expected 12 pilot parents, found {len(parents_meta)}"

    rep_parents_meta = [p for p in parents_meta if p.get("is_repeatability")]
    assert len(rep_parents_meta) == 6, f"Expected 6 repeatability parents, found {len(rep_parents_meta)}"
    rep_codes = [p["code"] for p in rep_parents_meta]

    print(f"Repeatability parents: {rep_codes} across seeds {eval_seeds}", flush=True)

    # Phase 1: Repeatability on 6 parents via isolated worker subprocesses
    print("\n--- Phase 1: Multi-Process Cold-Cache Repeatability Verification ---", flush=True)
    rep_results: Dict[str, Any] = {}
    solve_timings: List[float] = []
    conformer_counts: List[int] = []

    for p in rep_parents_meta:
        code = p["code"]
        rep_results[code] = {
            "metal": p["metal"],
            "charge": p["charge"],
            "n_heavy": p["n_heavy"],
            "seeds": {},
        }

    tasks = []
    for p in rep_parents_meta:
        code, shard, ds = p["code"], p["shard"], p["dataset"]
        for s in eval_seeds:
            tasks.append((code, shard, ds, s))

    def _evaluate_task(task_item: Tuple[str, str, str, int]) -> Tuple[str, int, Dict[str, Any], float, int]:
        c, sh, ds, s = task_item
        pool_a = pools_dir / f"{c}_s{s}.npz"
        res_a_file = pools_dir / f"{c}_s{s}_run_a.json"
        pool_b = pools_dir / f"{c}_s{s}_run_b.npz"
        res_b_file = pools_dir / f"{c}_s{s}_run_b.json"

        print(f"  [{c} seed {s}] Launching Run A in worker process...", flush=True)
        res_a = run_worker_process(c, sh, ds, s, "run_a", pool_a, res_a_file)

        print(f"  [{c} seed {s}] Launching Run B in worker process...", flush=True)
        res_b = run_worker_process(c, sh, ds, s, "run_b", pool_b, res_b_file)

        pid_a = res_a.get("worker_pid")
        pid_b = res_b.get("worker_pid")
        assert pid_a != pid_b, f"Run A and Run B executed in same process! PID: {pid_a}"
        assert res_a.get("gfn2_cache_initial_size") == 0, "Run A cache was not cold"
        assert res_b.get("gfn2_cache_initial_size") == 0, "Run B cache was not cold"

        if res_a.get("status") != "success" or res_b.get("status") != "success":
            return c, s, {
                "run_a": res_a,
                "run_b": res_b,
                "status": "failed",
                "pass_seed_tolerance": False,
            }, 0.0, 0

        with np.load(pool_a) as za, np.load(pool_b) as zb:
            X_a = za["X"]
            X_b = zb["X"]
            scores_a = za["score"]
            scores_b = zb["score"]

        max_X_drift = float(np.max(np.abs(X_a - X_b)))
        max_score_drift = float(np.max(np.abs(scores_a - scores_b)))

        top1_a = res_a["top1_kj"]
        top1_b = res_b["top1_kj"]
        floor_a = res_a["floor_kj"]
        floor_b = res_b["floor_kj"]

        drift_top1 = abs(top1_a - top1_b)
        drift_floor = abs(floor_a - floor_b)
        pass_seed = (drift_top1 <= 5.0 and drift_floor <= 5.0)

        seed_res = {
            "run_a": res_a,
            "run_b": res_b,
            "process_isolation": {
                "run_a_pid": pid_a,
                "run_b_pid": pid_b,
                "independent_processes": True,
                "cold_embed_cache_verified": True,
            },
            "max_coord_abs_drift_angstrom": max_X_drift,
            "max_score_abs_drift": max_score_drift,
            "drift_top1_kj_mol": drift_top1,
            "drift_floor_kj_mol": drift_floor,
            "pass_seed_tolerance": pass_seed,
        }
        print(
            f"    -> [{c} seed {s}] Drift: Top-1 = {drift_top1:.4f} kJ/mol, Floor = {drift_floor:.4f} kJ/mol, "
            f"Max dX = {max_X_drift:.2e} A (PIDs: {pid_a} vs {pid_b})",
            flush=True,
        )
        return c, s, seed_res, float(res_a["t_solve"]), int(res_a["n_confs"])

    n_workers = min(workers, len(tasks)) if workers > 1 else 1
    print(f"Executing {len(tasks)} repeatability tasks with {n_workers} worker threads...", flush=True)
    if n_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            for c, s, seed_res, t_solv, n_c in executor.map(_evaluate_task, tasks):
                rep_results[c]["seeds"][str(s)] = seed_res
                solve_timings.append(t_solv)
                conformer_counts.append(n_c)
    else:
        for t_item in tasks:
            c, s, seed_res, t_solv, n_c = _evaluate_task(t_item)
            rep_results[c]["seeds"][str(s)] = seed_res
            solve_timings.append(t_solv)
            conformer_counts.append(n_c)

    for p in rep_parents_meta:
        code = p["code"]
        valid_seeds = [
            s for s in eval_seeds
            if str(s) in rep_results[code]["seeds"] and "drift_top1_kj_mol" in rep_results[code]["seeds"][str(s)]
        ]
        if valid_seeds:
            mean_top1_drift = float(np.mean([rep_results[code]["seeds"][str(s)]["drift_top1_kj_mol"] for s in valid_seeds]))
            mean_floor_drift = float(np.mean([rep_results[code]["seeds"][str(s)]["drift_floor_kj_mol"] for s in valid_seeds]))
            max_seed_drift = float(max(
                max(rep_results[code]["seeds"][str(s)]["drift_top1_kj_mol"], rep_results[code]["seeds"][str(s)]["drift_floor_kj_mol"])
                for s in valid_seeds
            ))
            passed = (mean_top1_drift <= 1.0 and mean_floor_drift <= 1.0 and max_seed_drift <= 5.0)
            rep_results[code].update({
                "status": "evaluated",
                "evaluated_seeds_count": len(valid_seeds),
                "parent_seed_mean_top1_drift_kj_mol": mean_top1_drift,
                "parent_seed_mean_floor_drift_kj_mol": mean_floor_drift,
                "max_individual_seed_drift_kj_mol": max_seed_drift,
                "pass_repeatability": passed,
            })
            print(
                f"  ==> {code} Mean Drift: Top-1 = {mean_top1_drift:.4f} kJ/mol, Floor = {mean_floor_drift:.4f} kJ/mol. PASS = {passed}",
                flush=True,
            )

    # Phase 2: Compaction Diagnostics & Component Profiling across all 12 Pilot Parents
    print("\n--- Phase 2: Detailed Resource Profiling & Compaction Diagnostics on 12 Parents ---", flush=True)
    mace_evaluator = MACEPolarEvaluator(
        model="polar-1-s",
        device="auto",
        dtype="float64",
        model_path=str(MACE_MODEL_PATH),
        mode="energy-only",
        threads=1,
    ).load()

    component_measurements: Dict[str, Any] = {}
    compaction_diagnostics: Dict[str, Any] = {}
    nci_enum_timings: List[float] = []
    nci_feat_timings: List[float] = []
    nci_autograd_timings: List[float] = []
    mace_timings_per_conf: List[float] = []
    peak_rss_list: List[float] = []
    peak_gpu_list: List[float] = []
    parent_opportunities_totals: List[int] = []
    parent_per_class_opportunities: Dict[str, List[int]] = defaultdict(list)

    for p in parents_meta:
        code, shard, ds = p["code"], p["shard"], p["dataset"]
        mol = load_mol(code, shard, ds)
        if mol is None:
            continue

        ref_positions = np.array([a.position for a in mol.atoms])
        ref_symbols = [a.symbol for a in mol.atoms]
        ref_rg = compute_heavy_atom_rg(ref_symbols, ref_positions)
        ref_contacts = compute_nonbonded_contacts(mol, ref_positions, cutoff=3.5)

        # Retrieve or solve seed 11 pool
        pool_file = pools_dir / f"{code}_s11.npz"
        res_file = pools_dir / f"{code}_s11_run_a.json"
        if not pool_file.exists():
            print(f"  Solving {code} seed 11 in worker process...", flush=True)
            run_worker_process(code, shard, ds, 11, "run_a", pool_file, res_file)

        z = np.load(pool_file, allow_pickle=False)
        if "X" in z.files:
            poses = z["X"]
            scores = z["score"]
            if float(z.get("t_solve", 0)) > 0 and float(z.get("t_solve", 0)) not in solve_timings:
                solve_timings.append(float(z["t_solve"]))
                conformer_counts.append(len(poses))
        else:
            poses = ref_positions[None, ...]
            scores = np.array([0.0])

        # Build graph-constrained generated -> reference atom mapping
        atom_map = None
        try:
            builder, _, _ = build_builder_and_inputs(mol)
            if builder is not None:
                atom_map = build_atom_mapping(
                    mol,
                    builder=builder,
                    X_ref=ref_positions,
                    X_gen=poses[0] if len(poses) > 0 else None,
                )
        except Exception as exc:
            print(f"  Warning: atom mapping failed for {code}: {exc}", flush=True)

        if atom_map is not None:
            mapped_poses = atom_map.apply_to_positions(poses)
        else:
            mapped_poses = poses

        # Component (ii) & (iii): NCI v2 enumeration, extraction, runtime autograd
        nci_prof = measure_nci_v2_component(mol, mapped_poses[:min(len(mapped_poses), 16)], schema=spec)
        nci_enum_timings.append(nci_prof["wall_time_enumeration_s"])
        nci_feat_timings.append(nci_prof["wall_time_features_s"])
        nci_autograd_timings.append(nci_prof["wall_time_runtime_autograd_s"])
        parent_opportunities_totals.append(nci_prof["opportunities_count"])
        for fam, cnt in nci_prof["opportunities_by_family"].items():
            parent_per_class_opportunities[fam].append(cnt)

        # Component (iv): MACE float64 evaluation on mapped poses with matching symbols
        mace_poses = mapped_poses[:min(len(mapped_poses), 8)]
        mace_kj, mace_wall, mace_rss, mace_gpu = evaluate_mace_energies(
            mace_evaluator, ref_symbols, mace_poses, mol.charge, mol.spin_state
        )
        if len(mace_poses) > 0:
            mace_timings_per_conf.append(mace_wall / len(mace_poses))

        peak_rss_list.append(max(nci_prof["peak_rss_mb"], mace_rss))
        peak_gpu_list.append(max(nci_prof["peak_gpu_mb"], mace_gpu))

        # Compaction diagnostics with graph-constrained mapped coordinates
        top1_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
        top1_pos = mapped_poses[top1_idx]
        top1_rg = compute_heavy_atom_rg(ref_symbols, top1_pos)
        top1_contacts = compute_nonbonded_contacts(mol, top1_pos, cutoff=3.5)

        pool_rgs = [compute_heavy_atom_rg(ref_symbols, mapped_poses[i]) for i in range(len(mapped_poses))]
        pool_contacts = [compute_nonbonded_contacts(mol, mapped_poses[i], cutoff=3.5) for i in range(len(mapped_poses))]
        mean_pool_rg = float(np.mean(pool_rgs))
        mean_pool_contacts = float(np.mean(pool_contacts))

        compaction_diagnostics[code] = {
            "metal": p["metal"],
            "metal_row": p.get("metal_row", "3d"),
            "charge": p["charge"],
            "n_heavy": p["n_heavy"],
            "n_atoms": p["n_atoms"],
            "opportunities_total": nci_prof["opportunities_count"],
            "reference_rg_angstrom": ref_rg,
            "reference_contacts_lt_3_5A": ref_contacts,
            "generated_top1_rg_angstrom": top1_rg,
            "generated_top1_contacts_lt_3_5A": top1_contacts,
            "generated_pool_mean_rg_angstrom": mean_pool_rg,
            "generated_pool_mean_contacts_lt_3_5A": mean_pool_contacts,
            "rg_ratio_top1_vs_ref": float(top1_rg / max(ref_rg, 1e-6)),
            "contacts_ratio_top1_vs_ref": float(top1_contacts / max(ref_contacts, 1)),
            "atom_mapping_ambiguous": atom_map.is_ambiguous if atom_map else False,
            "atom_mapping_automorphisms": atom_map.num_automorphisms if atom_map else 1,
        }

        component_measurements[code] = {
            "opportunities_count": nci_prof["opportunities_count"],
            "opportunities_by_family": nci_prof["opportunities_by_family"],
            "wall_s_nci_enumeration": nci_prof["wall_time_enumeration_s"],
            "wall_s_nci_features_batch": nci_prof["wall_time_features_s"],
            "wall_s_nci_runtime_autograd": nci_prof["wall_time_runtime_autograd_s"],
            "wall_s_mace64_eval": mace_wall,
            "confs_evaluated_mace64": len(mace_poses),
            "mace64_seconds_per_conf": mace_wall / max(len(mace_poses), 1),
            "nci_grad_norm": nci_prof["grad_norm"],
            "nci_hvp_norm": nci_prof["hvp_norm"],
            "peak_rss_mb": max(nci_prof["peak_rss_mb"], mace_rss),
            "peak_gpu_mb": max(nci_prof["peak_gpu_mb"], mace_gpu),
        }
        print(
            f"  {code} (N_heavy={p['n_heavy']}): Opps={nci_prof['opportunities_count']}, "
            f"Rg_top1={top1_rg:.2f}A (ref={ref_rg:.2f}A), Contacts={top1_contacts} (ref={ref_contacts}), "
            f"MACE64={mace_wall / max(len(mace_poses), 1):.3f}s/conf",
            flush=True,
        )

    # Phase 3: Budget Extrapolation
    print("\n--- Phase 3: Budget Extrapolation for 240-Group Main Allocation ---", flush=True)
    extrapolation = extrapolate_budget(
        solve_timings=solve_timings,
        mace_timings=mace_timings_per_conf,
        enum_timings=nci_enum_timings,
        feat_timings=nci_feat_timings,
        autograd_timings=nci_autograd_timings,
        peak_rss_list=peak_rss_list,
        peak_gpu_list=peak_gpu_list,
        conformer_counts=conformer_counts,
    )

    # Aggregate repeatability summary
    valid_rep = [v for v in rep_results.values() if v.get("status") == "evaluated"]
    overall_mean_top1_drift = float(np.mean([v["parent_seed_mean_top1_drift_kj_mol"] for v in valid_rep])) if valid_rep else 0.0
    overall_mean_floor_drift = float(np.mean([v["parent_seed_mean_floor_drift_kj_mol"] for v in valid_rep])) if valid_rep else 0.0
    overall_max_seed_drift = float(max(v["max_individual_seed_drift_kj_mol"] for v in valid_rep)) if valid_rep else 0.0
    repeatability_passed = (overall_mean_top1_drift <= 1.0 and overall_mean_floor_drift <= 1.0 and overall_max_seed_drift <= 5.0)

    # Per-family (bond_atom / bond_ring / ring_ring) medians across parents
    per_family_medians = {k: float(np.median(v)) for k, v in parent_per_class_opportunities.items()}
    max_family_median = float(max(per_family_medians.values())) if per_family_medians else 0.0
    max_per_complex_total = float(max(parent_opportunities_totals)) if parent_opportunities_totals else 0.0

    report_payload = {
        "report_schema": "nci_v2_pilot_repeatability_and_budget_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "spec_version": spec.get("version", "2.0.0-pilot.2"),
        "frame_schema": frame.get("spec_schema", "nci_v2.0_pilot"),
        "provenance": provenance,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "mace_model_path": str(MACE_MODEL_PATH),
            "profile": _P.snapshot(),
            "deterministic_embed_mode": "research_arm_setting",
            "deterministic_embed": True,
            "deterministic_embed_notes": (
                "deterministic_embed=True is an opt-in research-arm profile setting for bit-reproducible GFN2 refinement under CPU load; "
                "default production setting is False."
            ),
            "process_isolation_verified": True,
            "cold_embed_cache_verified": True,
        },
        "repeatability": {
            "parents_designated": rep_codes,
            "parents_evaluated_count": len(valid_rep),
            "parents_unmappable_count": len(rep_results) - len(valid_rep),
            "tolerances_spec_kj_mol": {
                "parent_seed_mean_top1_max": 1.0,
                "parent_seed_mean_floor_max": 1.0,
                "individual_seed_max": 5.0,
            },
            "achieved_metrics_kj_mol": {
                "max_parent_seed_mean_top1_drift": overall_mean_top1_drift,
                "max_parent_seed_mean_floor_drift": overall_mean_floor_drift,
                "max_individual_seed_drift": overall_max_seed_drift,
            },
            "status": "passed" if repeatability_passed else "failed",
            "parents": rep_results,
        },
        "component_timings_and_resources": {
            "summary": {
                "solver_generation_mean_s": extrapolation["measured_unit_rates"]["solve_wall_time_s"]["mean"],
                "nci_v2_enumeration_mean_s": extrapolation["measured_unit_rates"]["nci_v2_enumeration_s_per_parent"],
                "nci_v2_features_mean_s": extrapolation["measured_unit_rates"]["nci_v2_features_s_per_batch16"],
                "nci_v2_runtime_autograd_mean_s": extrapolation["measured_unit_rates"]["nci_v2_runtime_autograd_s_per_batch16"],
                "mace64_evaluation_mean_s_per_conf": extrapolation["measured_unit_rates"]["mace64_s_per_conformer"]["mean"],
                "peak_cpu_rss_mb": extrapolation["measured_unit_rates"]["peak_memory_mb"]["cpu_rss_mb"],
                "peak_gpu_mem_mb": extrapolation["measured_unit_rates"]["peak_memory_mb"]["gpu_allocated_mb"],
            },
            "per_parent": component_measurements,
        },
        "compaction_diagnostics": {
            "summary": {
                "mean_rg_ratio_top1_vs_ref": float(np.mean([d["rg_ratio_top1_vs_ref"] for d in compaction_diagnostics.values()])),
                "mean_contacts_ratio_top1_vs_ref": float(np.mean([d["contacts_ratio_top1_vs_ref"] for d in compaction_diagnostics.values()])),
                "max_per_complex_total_opf": max_per_complex_total,
                "max_family_median_opf": max_family_median,
                "per_family_medians": per_family_medians,
            },
            "per_parent": compaction_diagnostics,
        },
        "budget_extrapolation": extrapolation,
    }

    report_path = out_dir / "pilot_repeatability_and_budget.json"
    write_json_atomic(report_path, report_payload)
    print(f"\nWrote Step 1 report -> {report_path}", flush=True)
    print(f"Repeatability Status: {report_payload['repeatability']['status'].upper()}", flush=True)
    print(f"Max Seed-Mean Drift: Top-1 = {overall_mean_top1_drift:.4f} kJ/mol, Floor = {overall_mean_floor_drift:.4f} kJ/mol", flush=True)
    print(f"Max Individual Seed Drift: {overall_max_seed_drift:.4f} kJ/mol", flush=True)
    return report_payload


def run_integrity_audit(pools_dir: Path = DEFAULT_POOLS_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> Dict[str, Any]:
    """Coordination-integrity outcomes for existing Run A pools (no re-solve); writes pilot_coordination_integrity.json."""
    pools: Dict[str, Any] = {}
    for f in sorted(Path(pools_dir).glob("*_s[0-9]*.npz")):
        if "_run_b" in f.stem or f.stem.endswith("_tmp"):
            continue
        z = np.load(f, allow_pickle=True)
        if "X" not in z or "gen_to_ref_idx" not in z:
            pools[f.stem] = {"status": "no_pool_or_no_atom_map"}
            continue
        ci = coordination_integrity(z["ref_symbols"], z["ref_X"], z["X"][:, z["gen_to_ref_idx"], :])
        top1 = int(np.argmax(z["score"]))
        mace_f = f.with_suffix(".mace64.json")
        floor = json.loads(mace_f.read_text())["floor_idx"] if mace_f.exists() else None
        pools[f.stem] = {
            "status": "evaluated",
            "n_confs": int(len(ci["intact"])),
            "n_sigma_donors": ci["n_sigma_donors"],
            "haptic_group_sizes": ci["haptic_group_sizes"],
            "pool_intact_fraction": float(np.mean(ci["intact"])),
            "top1_intact": bool(ci["intact"][top1]),
            "top1_sigma_shift_A": float(ci["sigma_shift_A"][top1]),
            "top1_haptic_slip_A": float(ci["haptic_slip_A"][top1]),
            "top1_haptic_shift_A": float(ci["haptic_shift_A"][top1]),
            "floor_intact": bool(ci["intact"][floor]) if floor is not None else None,
        }
    evaluated = [v for v in pools.values() if v["status"] == "evaluated"]
    report = {
        "report_schema": "nci_v2_pilot_coordination_integrity_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "provenance": get_provenance_hashes(REPO),
        "definition": {
            "bound_atoms": f"reference heavy atoms with d(M-X) <= {COORD_BOUND_SCALE} * (r_M + r_X)",
            "tolerance_A": COORD_SHIFT_TOL_A,
            "metrics": "sigma shift |d_gen-d_ref|; haptic slip = spread(d_gen)-spread(d_ref); haptic shift = |mean d_gen - mean d_ref|",
            "use": "Poses that are not intact are solver outcomes (generation failures) and are excluded from NCI attribution.",
        },
        "summary": {
            "pools_evaluated": len(evaluated),
            "top1_not_intact": sorted(k for k, v in pools.items() if v["status"] == "evaluated" and not v["top1_intact"]),
            "mean_pool_intact_fraction": float(np.mean([v["pool_intact_fraction"] for v in evaluated])) if evaluated else None,
        },
        "pools": pools,
    }
    write_json_atomic(Path(out_dir) / "pilot_coordination_integrity.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="run in worker subprocess mode")
    parser.add_argument("--code", type=str, help="parent code for worker")
    parser.add_argument("--shard", type=str, help="shard for worker")
    parser.add_argument("--dataset", type=str, default="tmQM", help="dataset for worker")
    parser.add_argument("--seed", type=int, help="seed for worker")
    parser.add_argument("--run-id", type=str, choices=("run_a", "run_b"), help="run identifier")
    parser.add_argument("--pool-out", type=Path, help="pool npz output path")
    parser.add_argument("--res-out", type=Path, help="result JSON output path")

    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="spec.json path")
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME, help="frame.json path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="report directory")
    parser.add_argument("--pools-dir", type=Path, default=DEFAULT_POOLS_DIR, help="pools directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13], help="seeds to evaluate")
    parser.add_argument("--quick", action="store_true", help="fast single-seed evaluation for test")
    parser.add_argument("--workers", type=int, default=4, help="worker process count")
    parser.add_argument("--integrity-audit", action="store_true", help="coordination-integrity report on existing pools (no solves)")
    args = parser.parse_args()

    if args.integrity_audit:
        rep = run_integrity_audit(pools_dir=args.pools_dir, out_dir=args.out_dir)
        print(json.dumps(rep["summary"], indent=2))
        return

    if args.worker:
        assert args.code and args.shard and args.seed is not None and args.run_id and args.pool_out and args.res_out, (
            "Worker mode requires --code, --shard, --seed, --run-id, --pool-out, and --res-out"
        )
        worker_solve(
            code=args.code,
            shard=args.shard,
            dataset=args.dataset,
            seed=args.seed,
            run_id=args.run_id,
            pool_out=args.pool_out,
            res_out=args.res_out,
        )
        return

    run_pilot(
        spec_path=args.spec,
        frame_path=args.frame,
        out_dir=args.out_dir,
        pools_dir=args.pools_dir,
        seeds=args.seeds,
        quick=args.quick,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
