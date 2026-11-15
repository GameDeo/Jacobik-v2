"""GPU arm runner for the simple bench (research/archive/docs/roadmap/roadmap_generation_v1.md 1.3, 8.3 B3/C).

Optimised for GPU saturation and multi-arm batching (Package C / session 2):
1. Multi-arm batch mode via --arms-spec: runs control and arms together, amortising load_mol,
   builder construction, GFN2 ligand embedding, and multiseed row generation across arms.
2. Background 1-thread prefetching: loads parent i+1 and warms _GFN2_CACHE on CPU while GPU solves parent i.
3. Batched MACE evaluation: evaluates reference geometry and all conformers across seeds and arms
   in a single batched mace.energies() call with deduplication of identical geometries.
4. Scale-invariance fast path: for arms without active scales on a parent (e.g. ring: 0.5 on non-ring parents),
   reuses control conformers with exact 0.000000e+00 delta and zero redundant GPU solve or MACE work.
5. Preserves 100% backward compatibility for single-arm CLI invocations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[4]
CONTROL_OVERRIDES = {"nci": False, "deterministic_embed": True}
EV_TO_KJ_MOL = 96.4853321233  # matches jacobik.core.energy.EV_TO_KJ_MOL


def _setup_env(flags: dict) -> None:
    """Profile override JSON must be in the environment before jacobik is imported. The interpreter's own env bin
    comes first on PATH (ptxas and nvcc for the GPU path)."""
    import shutil
    bindir = os.path.dirname(sys.executable)
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.path.join(os.path.dirname(bindir), "lib") + os.pathsep + os.environ.get(
        "LD_LIBRARY_PATH", "")
    xtb = shutil.which("xtb")
    if xtb is None or os.path.dirname(os.path.abspath(xtb)) != os.path.abspath(bindir):
        raise SystemExit(f"xtb must resolve to {bindir}/xtb (found {xtb})")
    os.environ["JACOBIK_XTB"] = os.path.abspath(xtb)
    over = dict(CONTROL_OVERRIDES)
    over.update(flags)
    os.environ["JACOBIK_PROFILE"] = "u1"
    os.environ["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(over)
    for p in (str(REPO / "src"), str(REPO / "dev"), str(REPO), str(REPO / "research" / "bench"),
              str(REPO / "dev" / "tools" / "jdev" / "budget")):
        if p not in sys.path:
            sys.path.insert(0, p)


class _InfeasibleCounter(logging.Handler):
    """Counts the engine's 'topologically infeasible' warnings (S3 census key) and CUDA replay fallbacks per unit."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.counts = {
            "infeasible_replaced": 0,
            "infeasible_kept": 0,
            "feasibility_check_failed": 0,
            "replay_fallback": 0,
            "polish_failure": 0,
            "pivot_correction_fallback": 0,
        }

    def emit(self, record):
        msg = record.getMessage()
        if "replaced by lowest-cost feasible" in msg:
            self.counts["infeasible_replaced"] += 1
        elif "topologically infeasible and no feasible" in msg:
            self.counts["infeasible_kept"] += 1
        elif "Baseline feasibility check failed" in msg:
            self.counts["feasibility_check_failed"] += 1
        elif "polish failed; keeping delivered baseline" in msg:
            self.counts["polish_failure"] += 1
        elif "C-2b pivot correction failed" in msg:
            self.counts["pivot_correction_fallback"] += 1
        elif "step recording failed at step" in msg:
            self.counts["replay_fallback"] += 1


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _assert_gfn2() -> dict:
    """The GFN2 refinement must really run (xtb moves the geometry, no fallback warning). Raises otherwise."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from jacobik.builder import engine
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(m)
    pos = m.GetConformer().GetPositions()
    out = engine._gfn2_refine(m, pos.copy(), donors=[[1, 3]])
    moved = float(np.abs(np.asarray(out) - pos).max())
    if not moved > 1e-3:
        raise SystemExit(f"GFN2 refinement did not move the geometry (max shift {moved})")
    return dict(xtb=os.environ["JACOBIK_XTB"], gfn2_selfcheck_max_shift_A=round(moved, 4))


def _has_active_scales(arm_flags: dict, fks: list) -> bool:
    """Check if this arm has any active scale or ramp that would change the solver trajectory from control."""
    from jacobik.core import profile as P
    if any(arm_flags.get(k, P.FIELDS[k][2]) is not None for k in ("finish_phase", "polish", "optimizer")):
        return True
    if "bend_release" in arm_flags and arm_flags["bend_release"] != [0.1, 100]:
        return True
    if "bend_free_deg" in arm_flags and arm_flags["bend_free_deg"] != 2.0:
        return True
    dof_scale = arm_flags.get("dof_step_scale")
    if not dof_scale:
        return False
    if float(dof_scale.get("rigid", 1.0)) != 1.0:
        return True
    s_torsion = float(dof_scale.get("torsion", 1.0))
    s_ring = float(dof_scale.get("ring", 1.0))
    s_bend = float(dof_scale.get("bend", 1.0))
    for fk in fks:
        n_joints = fk.n_joints
        n_macro = getattr(fk, "n_macro_joints", 0)
        n_tors = n_joints - n_macro
        n_bends = getattr(fk, "n_angle_bends", 0)
        if n_tors > 0 and s_torsion != 1.0:
            return True
        if n_macro > 0 and s_ring != 1.0:
            return True
        if n_bends > 0 and s_bend != 1.0:
            return True
    return False


#: fields _set_finish_flags switches per arm (module attributes; their value between arms is the previous arm's)
ARM_SWITCHED_BEAM = ("start_kabsch", "start_kabsch_pivot", "start_kabsch_diversity", "out_mirror",
                     "ring_states", "config_states", "config_near", "config_remote")
ARM_SWITCHED_SOLVE = ("finish_phase", "polish", "optimizer")


def _set_finish_flags(flags):
    from jacobik.core import profile as P
    from jacobik.ik import ik_solver as IKM, solver_replay as SR
    from jacobik.builder import engine as E
    for name in ARM_SWITCHED_BEAM:
        _, attr, default, _ = P.FIELDS[name]
        setattr(E, attr, flags.get(name, default))
    for name in ARM_SWITCHED_SOLVE:
        _, attr, default, _ = P.FIELDS[name]
        value = flags.get(name, default)
        setattr(IKM, attr, value)
        setattr(SR, attr, value)


def _prefetch_parent(code: str, shard: str, ds: str):
    """Prefetch (mol, builder) and warm _GFN2_CACHE for parent ligands on CPU."""
    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs
    from jacobik.builder import engine
    try:
        mol = inputs.load_mol(code, shard, ds)
        if mol is None:
            return None, None, f"no geometry for {code} in {shard}"
        builder, _, err = build_builder_and_inputs(mol)
        if builder is None:
            return mol, None, err
        # Embedding runs during actual preparation, under EmbeddingValidity.
        # The old optional warm-up called a nonexistent helper and did no work.
        return mol, builder, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {str(exc)[:200]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench run")
    ap.add_argument("--arm", default=None, help="Arm name (required if --arms-spec not set)")
    ap.add_argument("--panel", required=True, choices=["tail", "cal", "named", "val_census"],
                    help="named: the codes of --only (any frame parent); val_census: motion_validation")
    ap.add_argument("--out", default=None, help="Output directory (required if --arms-spec not set)")
    ap.add_argument("--flags", default="{}", help="Profile overrides JSON for single-arm run")
    ap.add_argument("--seeds", default="11,12,13")
    ap.add_argument("--seed-budget", choices=["three_w16", "single_w48"], default="three_w16",
                    help="single_w48: each independent seed uses width 48, compared with a three-seed width-16 union")
    ap.add_argument("--only", default="")
    ap.add_argument("--xyz-dir", default="", help="Full-pool export root (default: OUT/xyz)")
    ap.add_argument("--no-xyz", action="store_true", help="Disable pool and top1/floor XYZ export")
    ap.add_argument("--xyz-compression", choices=["none", "xz"], default="none",
                    help="xz: one multi-frame pool per parent/arm/seed")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--cpu-identity", action="store_true",
                    help="CPU identity comparison: one Torch/solver worker, no row culling")
    ap.add_argument("--arms-spec", default=None, help="JSON string or file path containing list of arm dicts")
    a = ap.parse_args(argv)
    if a.cpu_identity and a.device != "cpu":
        ap.error("--cpu-identity requires --device cpu")

    if a.arms_spec:
        if os.path.exists(a.arms_spec):
            arms_spec = json.loads(Path(a.arms_spec).read_text())
        else:
            arms_spec = json.loads(a.arms_spec)
        base_flags = arms_spec[0].get("flags", {})
    else:
        if not a.arm or not a.out:
            ap.error("--arm and --out are required when --arms-spec is not specified")
        base_flags = json.loads(a.flags)
        arms_spec = [{"arm": a.arm, "flags": base_flags, "out": a.out, "xyz_dir": a.xyz_dir}]

    _setup_env(base_flags)

    import importlib
    import numpy as np
    import torch
    from jacobik.core import profile as P
    from tools.jgen import inputs
    from tools.jgen.bench import panels
    from jacobik.core.env import build_builder_and_inputs
    from jacobik.core.env import MACE_MODEL_PATH
    from jacobik.builder.atom_map import build_atom_mapping
    from jacobik.core.energy import MACEPolarEvaluator
    from tools.jdev.budget.common import stats_from_db
    from tools.jdev import multiseed
    from jacobik.builder import engine as E
    from jacobik.ik import ik_solver as IKM
    import jacobik.ik.solver_replay as SR
    identity_kw = {}
    if a.cpu_identity:
        torch.set_num_threads(1)
        identity_kw["ik_config"] = IKM.IKSolverConfig(n_workers=1, cull_at=(), cull_keep=())

    seeds = [int(s) for s in a.seeds.split(",")]
    manifest = json.load(open(inputs.FRAME_V1 / "manifest.json"))
    frozen = manifest.get("base_provenance", {})
    db_sha = inputs.sha256_file(inputs.DB_PATH)
    shards = inputs.frame_shards()
    commit = _git_commit()

    only = {x for x in a.only.split(",") if x}
    if a.panel == "named":
        codes = sorted(only)
    else:
        pname = "val" if a.panel == "val_census" else a.panel
        sel = panels.select(manifest, pname)["units"]
        codes = []
        for t in sel:
            if t["code"] not in codes and (not only or t["code"] in only):
                codes.append(t["code"])

    # Optimised GPU memory fraction & MACE batch atom sizing
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.80)
    os.environ.setdefault("JACOBIK_MACE_BATCH_ATOMS", "900")
    mace = MACEPolarEvaluator(model="polar-1-s", device=a.device, dtype="float64", model_path=str(MACE_MODEL_PATH),
                              mode="energy-only", threads=1).load()
    mace._batch_atoms = 900
    stats = stats_from_db(inputs.DB_PATH)
    gfn2 = _assert_gfn2()
    counter = _InfeasibleCounter()
    logging.getLogger().addHandler(counter)
    from tools.jgen.embedding_validity import EmbeddingValidity
    embedding_validity = EmbeddingValidity(E).install()

    # Precompute snapshots and prepare record directories for each arm
    arm_snapshots = {}
    for arm in arms_spec:
        arm_name = arm["arm"]
        arm_flags = arm.get("flags", {})
        arm_out = Path(arm["out"])
        (arm_out / "records").mkdir(parents=True, exist_ok=True)
        live = dict(P.PRODUCTION_CONFIG)
        for k, (mod, attr, default_val, _) in P.FIELDS.items():
            try:
                live[k] = getattr(importlib.import_module(mod), attr)
            except Exception:
                live[k] = P._VALUES.get(k, default_val)
            if k in ARM_SWITCHED_BEAM or k in ARM_SWITCHED_SOLVE:   # the module holds another arm's value here
                live[k] = P._VALUES.get(k, default_val)
        for k, v in CONTROL_OVERRIDES.items():
            live[k] = v
        for k, v in arm_flags.items():
            live[k] = v
        differs = {k: live[k] for k in P.FIELDS if live.get(k) != P.FIELDS[k][2]}
        arm_snapshots[arm_name] = dict(
            fingerprint=P.fingerprint(live),
            differs_from_profile=differs,
            meta_common=dict(
                gfn2=gfn2, arm=arm_name, panel=a.panel, flags=arm_flags,
                seed_budget=arm.get("seed_budget", a.seed_budget),
                W=48 if arm.get("seed_budget", a.seed_budget)=="single_w48" else 16,
                fingerprint=P.fingerprint(live), fingerprint_fields_differing=differs,
                code_git_commit=commit, frame_sha256=frozen.get("frame_sha256"),
                constraints_db_sha256=db_sha, constraints_db_frozen=frozen.get("constraints_db_sha256"),
                mace_model="polar-1-s float64", mace_model_sha256=inputs.sha256_file(MACE_MODEL_PATH),
                production_fingerprint=P.PRODUCTION_FINGERPRINT,
                bench_runner_sha256=inputs.sha256_file(Path(__file__)),
                embedding_validity_guard="actual_gfn2_call_and_cached_failure_v1",
                cpu_identity_settings=({"torch_threads": 1, "n_workers": 1, "cull_at": [], "cull_keep": []}
                                       if a.cpu_identity else None),
                gpu_name=(torch.cuda.get_device_name(0) if a.device == "cuda" else None), device=a.device, machine=socket.gethostname(),
                bundle_commit=(json.loads((REPO/"COLAB_BUNDLE.json").read_text())["commit"] if (REPO/"COLAB_BUNDLE.json").exists() else commit)
            )
        )

    # Single-thread background prefetcher for CPU pipelining
    prefetcher = ThreadPoolExecutor(max_workers=1)
    next_future = None
    if len(codes) > 1:
        next_c = codes[1]
        next_sh, next_ds = shards[next_c]
        next_future = prefetcher.submit(_prefetch_parent, next_c, next_sh, next_ds)

    print(f"Starting run across {len(codes)} parents and {len(arms_spec)} arms (seeds={seeds})...", flush=True)

    for idx, code in enumerate(codes):
        _set_finish_flags(base_flags)
        if "_c2_arm_observer" in globals():
            _c2_arm_observer(None)
        embedding_validity.parent = code
        t0 = time.perf_counter()
        shard, ds = shards[code]

        # Get mol and builder for current code (from prefetcher if available)
        if idx == 0:
            mol, builder, err_msg = _prefetch_parent(code, shard, ds)
        else:
            mol, builder, err_msg = next_future.result()

        # Start prefetching parent i+1 while GPU solves parent i
        if idx + 1 < len(codes):
            next_c = codes[idx + 1]
            next_sh, next_ds = shards[next_c]
            next_future = prefetcher.submit(_prefetch_parent, next_c, next_sh, next_ds)
        else:
            next_future = None

        if builder is None:
            print(f"[{code}] FAILED: {err_msg}", flush=True)
            for arm in arms_spec:
                arm_out = Path(arm["out"])
                rec_dir = arm_out / "records"
                meta = arm_snapshots[arm["arm"]]["meta_common"]
                for s in seeds:
                    rec = dict(meta, code=code, seed=s, status="failed", reason=f"Unmappable: {err_msg}",
                               top1_dE=None, floor_dE=None, n_rows=0, n_finite=0)
                    (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))
            continue

        coverage_context = None
        ref_sym = [x.symbol for x in mol.atoms]
        ref_X = np.array([x.position for x in mol.atoms], float)

        # Check if parent requires E/Z enumeration (directive §2 Fix B)
        has_ez_multi = False
        try:
            from jacobik.builder import embed as _ce
            if getattr(_ce, "EZ_ENUMERATE", False) and hasattr(builder, "_ez_combinations"):
                ez_combs = builder._ez_combinations()
                if len(ez_combs) > 1:
                    has_ez_multi = True
        except Exception:
            has_ez_multi = False
        # Stereo-extra parents (stereo_candidates classes beyond production) solve through multiseed.solve_seeds, which
        # enumerates the classes. The direct prepare() path solved only the LAST class (engine._solve_stereo_variants
        # swallowed prepare's stop signal per class; fixed 2026-11-13, prepare now refuses a second capture).
        try:
            if multiseed._stereo_classes(builder) is not None:
                has_ez_multi = True
        except Exception as exc:
            print(f"[{code}] stereo class check failed ({exc}); production path", flush=True)
        # A ring_states arm solves through multiseed.solve_seeds (its own ligand states), but only where a ligand has a
        # ring it can change: on other parents the flag adds no rows, and the solve_seeds route (own prep per arm) made
        # the arm look up to x4 slower than its packed control (ZAZREZ, MIVXAR).
        if any(arm.get("flags", {}).get("ring_states") for arm in arms_spec) and multiseed._has_state_rings(
                builder, five=any(arm.get("flags", {}).get("ring_states_five") for arm in arms_spec)):
            has_ez_multi = True

        original_joint = builder._run_joint_ik
        current_replays = []
        def observed_joint(*args, **kwargs):
            value = original_joint(*args, **kwargs)
            if IKM.OPTIMIZER in ("irprop", "irprop_minus", "adam_score"):
                current_replays.append(dict(mode="native_score_only", replayed=0, T=arm_steps, steps_executed=arm_steps, error=None))
            else:
                current_replays.append({k: SR.LAST.get(k) for k in ("mode", "replayed", "T", "B", "steps_executed", "error")})
            return value
        builder._run_joint_ik = observed_joint
        try:
            arm_replays = {}
            arm_initial_digests = {}
            arm_results = {}
            arm_solve_walls = {}
            arm_deltas = {}
            base_control_arm = None

            with multiseed.safe_cuda_eigh_chunking():
                if not has_ez_multi:
                    # 1. Run multiseed.prepare and seed_rows ONCE per parent across all seeds and arms
                    try:
                        ctx, assignments, W, T = multiseed.prepare(
                            builder, seeds[0], device=a.device, angle_bends=True, stats=stats,
                            n_conformers=arms_spec[0].get("n_conformers"), **identity_kw
                        )
                        embedding_validity.check(code)
                        (fks, dpi, ge, scorer), rows = multiseed.seed_rows(builder, ctx, assignments, W, seeds)
                        ctx.score_norm = float(scorer._agg_norm()) if hasattr(scorer, "_agg_norm") else 0.0

                        W_total = W * len(assignments)
                        keep = None
                        ctx.early_cull = None
                        if E.CLASS_BUDGET == "split":
                            keep = builder._class_budget_rows(ctx, assignments, W)
                        elif E.CLASS_BUDGET == "early_cull":
                            n_cls = builder._n_topological_classes(ctx, assignments)
                            if n_cls is not None and n_cls < len(assignments):
                                ctx.early_cull = (E.EARLY_CULL_AT, n_cls / len(assignments))
                        if keep is not None:
                            kt = torch.as_tensor(keep, dtype=torch.long, device=ctx.device)
                            active_rows = [([x[kt] for x in th], X[kt] if X.dim() == 3 else X, ideal[kt], cand[keep]) for th, X, ideal, cand in rows]
                            W_total = len(keep) // max(ctx.n_coord_samples, 1)
                        else:
                            active_rows = rows
                        B1 = active_rows[0][0][0].shape[0]
                        control_initial = (fks, dpi, ge, scorer, active_rows, W_total, B1)
                        from tools.jgen.g3c0_distortion import start_digests
                        initial_theta_sha256 = start_digests(active_rows) if any(a.get("distortion") for a in arms_spec) else None
                        sym = ctx.symmetric_donor_groups
                        q3_sym = [g for g in sym if len({ctx.all_donor_groups[d][0] for d in g}) == 1]
                    except Exception as exc:
                        print(f"[{code}] SETUP/PREPARE FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                        for arm in arms_spec:
                            arm_out = Path(arm["out"])
                            rec_dir = arm_out / "records"
                            meta = arm_snapshots[arm["arm"]]["meta_common"]
                            for s in seeds:
                                rec = dict(meta, code=code, seed=s, status="failed", reason=f"{type(exc).__name__}: {str(exc)[:200]}",
                                           top1_dE=None, floor_dE=None, n_rows=0, n_finite=0)
                                (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))
                        continue

                # 2. Solve each arm (reusing control solve if arm has no active DOF scales on this complex)
                for arm in arms_spec:
                    arm_name = arm["arm"]
                    arm_flags = arm.get("flags", {})
                    current_replays = []
                    arm_replays[arm_name] = current_replays
                    _set_finish_flags(arm_flags)
                    if "_c2_arm_observer" in globals():
                        _c2_arm_observer(arm_name)
                    arm_steps = int(arm.get("n_ik_steps", T if not has_ez_multi else 200))
                    arm_budget = arm.get("seed_budget", a.seed_budget)
                    arm_width = 48 if arm_budget == "single_w48" else 16

                    if not has_ez_multi:
                        if arm_flags.get("start_kabsch", False) or arm_width != W:
                            (fks, dpi, ge, scorer), arm_rows = multiseed.seed_rows(builder, ctx, assignments, arm_width, seeds)
                            arm_keep = builder._class_budget_rows(ctx, assignments, arm_width) if E.CLASS_BUDGET == "split" else None
                            if arm_keep is not None:
                                kt = torch.as_tensor(arm_keep, dtype=torch.long, device=ctx.device)
                                active_rows = [([x[kt] for x in th], X[kt] if X.dim()==3 else X, ideal[kt], cand[arm_keep]) for th,X,ideal,cand in arm_rows]
                                W_total = len(arm_keep)//ctx.n_coord_samples
                            else:
                                active_rows = arm_rows
                                W_total = arm_width*len(assignments)
                            B1 = active_rows[0][0][0].shape[0]
                            expected = (len(arm_keep) if arm_keep is not None else len(assignments)*ctx.n_coord_samples*arm_width)
                            assert B1 == expected, "C-2 seed-row budget changed"
                        else:
                            fks, dpi, ge, scorer, active_rows, W_total, B1 = control_initial

                    if has_ez_multi:
                        arm_initial_digests[arm_name] = {s: {} for s in seeds}
                        def ez_diagnostics(confs, variant_ctx, variant_assignments, variant_ge, variant_scorer, variant_rows, seed_index):
                            from tools.jgen.g3c0_distortion import measure, start_digests
                            fixes = repr([lig.ez_fix for lig in builder.ligands])
                            arm_initial_digests[arm_name][seeds[seed_index]][fixes] = start_digests(variant_rows)[seed_index]
                            values = measure(confs, variant_ctx, variant_assignments, variant_ge, variant_scorer, builder.geometry)
                            for conf, value in zip(confs, values):
                                if conf.solver_state is None:
                                    conf.solver_state = {}
                                conf.solver_state["g3c0_distortion"] = value
                        counts_before_arm = dict(counter.counts)
                        t_solve0 = time.perf_counter()

                        IKM.BEND_RELEASE = arm_flags.get("bend_release", P.get("bend_release"))
                        IKM.DOF_STEP_SCALE = arm_flags.get("dof_step_scale", P.get("dof_step_scale"))
                        SR.BEND_RELEASE = arm_flags.get("bend_release", P.get("bend_release"))
                        SR.DOF_STEP_SCALE = arm_flags.get("dof_step_scale", P.get("dof_step_scale"))

                        os.environ["JACOBIK_LINEAGE_CODE"], os.environ["JACOBIK_LINEAGE_ARM"] = code, arm_name   # lineage tag (diagnostic)
                        arm_res_map = None
                        last_err = None
                        for mr in (None, 512, 256, 128, 64):
                            try:
                                res, ez_info = multiseed.solve_seeds(
                                    builder, seeds=seeds, device=a.device, batched_assembly=True,
                                    angle_bends=True, stats=stats, max_rows=mr, n_ik_steps=arm_steps,
                                    beam_width=arm_width,
                                    n_conformers=arm.get("n_conformers"),
                                    diagnostics=ez_diagnostics if arm.get("distortion") else None, **identity_kw
                                )
                                arm_res_map = res
                                break
                            except torch.cuda.OutOfMemoryError as exc:
                                last_err = exc
                                gc.collect()
                                torch.cuda.empty_cache()

                        if arm_res_map is None:
                            print(f"[{code}:{arm_name}] SOLVER OOM: {last_err}", flush=True)
                            arm_results[arm_name] = {}
                        else:
                            arm_results[arm_name] = arm_res_map

                        arm_solve_walls[arm_name] = (ez_info["s_solver"] + ez_info["s_dedup_assembly"]) if arm.get("distortion") and arm_res_map is not None else time.perf_counter() - t_solve0
                        arm_deltas[arm_name] = {k: counter.counts[k] - counts_before_arm[k] for k in counter.counts}
                        continue

                    has_scales = (_has_active_scales(arm_flags, fks) or arm_steps != T or arm_width != W
                                  or any(arm_flags.get(k, False) for k in ("start_kabsch",)))

                    if not has_scales and base_control_arm is not None:
                        # Reuse identical control solve: zero duplicate GPU work, exact 0.000 parity
                        arm_results[arm_name] = arm_results[base_control_arm]
                        arm_solve_walls[arm_name] = 0.0
                        arm_deltas[arm_name] = dict(arm_deltas[base_control_arm])
                        continue

                    counts_before_arm = dict(counter.counts)
                    t_solve0 = time.perf_counter()

                    IKM.BEND_RELEASE = arm_flags.get("bend_release", P.get("bend_release"))
                    IKM.DOF_STEP_SCALE = arm_flags.get("dof_step_scale", P.get("dof_step_scale"))
                    SR.BEND_RELEASE = arm_flags.get("bend_release", P.get("bend_release"))
                    SR.DOF_STEP_SCALE = arm_flags.get("dof_step_scale", P.get("dof_step_scale"))
                    import jacobik.ik.angle_bends as AB
                    arm_bend_free = float(arm_flags.get("bend_free_deg", P.get("bend_free_deg")))
                    AB.FREE_DEG = arm_bend_free
                    if ctx.bend_prior is not None:
                        ctx.bend_prior["free_deg"] = arm_bend_free
                    for fk in fks:
                        if getattr(fk, "_bend_plan", None) is not None and hasattr(fk, "_bend_specs"):
                            fk._bend_free = torch.tensor(
                                AB.free_zone(fk._bend_specs, free_deg=arm_bend_free),
                                dtype=torch.float32, device=ctx.device
                            )

                    # Solver pass with OOM chunk retry
                    tries = [None, 1024, 512, 256, 128, 64]
                    last_err = None
                    arm_res_map = None

                    for mr in tries:
                        try:
                            per = max(1, (mr // B1) if mr else len(seeds))
                            groups = [list(range(i, min(i + per, len(seeds)))) for i in range(0, len(seeds), per)]
                            res_map = {}
                            for g in groups:
                                th = [torch.cat([active_rows[i][0][li].clone() for i in g], dim=0) for li in range(len(fks))]
                                X_in = torch.cat([active_rows[i][1].clone() for i in g], dim=0) if active_rows[0][1].dim() == 3 else active_rows[0][1].clone()
                                ideal = torch.cat([active_rows[i][2].clone() for i in g], dim=0)
                                cand = np.concatenate([active_rows[i][3] for i in g])
                                from tools.jdev import lineage as _lin     # diagnostic (JACOBIK_LINEAGE_OUT), off by default
                                th_lin = [t_.detach().clone() for t_ in th] if _lin.enabled() else None
                                _t_lin = time.perf_counter()
                                thetas, conv, scores, trace = builder._run_joint_ik(
                                    ctx, fks, scorer, ge, X_in, th, dpi, ideal, sym, q3_sym, W_total * len(g), arm_steps, cand
                                )
                                bend_g = getattr(builder, "_last_bend_prior", None)
                                if ctx.device.type == "cuda":
                                    torch.cuda.synchronize()
                                lin = None if th_lin is None else dict(
                                    assignments=assignments, th_init=th_lin, solver_wall_s=time.perf_counter() - _t_lin,
                                    beam_width=arm_width, keep=locals().get("arm_keep", locals().get("keep")),
                                    meta=dict(seeds=[seeds[i] for i in g], code=code, arm=arm_name))
                                batched = multiseed.assemble_batched(
                                    builder, ctx, fks, X_in, thetas, scores, conv, W_total, cand, B1, len(g), {}, lineage=lin
                                )
                                for j, i_s in enumerate(g):
                                    confs = batched[j]
                                    sl = slice(j * B1, (j + 1) * B1)
                                    E.attach_solver_state(confs, [x[sl] for x in thetas], None if bend_g is None else bend_g[sl])
                                    res_map[seeds[i_s]] = E.EnsembleResult(conformers=confs, solve_trace=None, scorer=scorer)
                            arm_res_map = res_map
                            break
                        except torch.cuda.OutOfMemoryError as exc:
                            last_err = exc
                            gc.collect()
                            torch.cuda.empty_cache()

                    if arm_res_map is None:
                        print(f"[{code}:{arm_name}] SOLVER OOM: {last_err}", flush=True)
                        arm_results[arm_name] = {}
                    else:
                        arm_results[arm_name] = arm_res_map

                    arm_solve_walls[arm_name] = time.perf_counter() - t_solve0
                    arm_deltas[arm_name] = {k: counter.counts[k] - counts_before_arm[k] for k in counter.counts}
                    if not has_scales and base_control_arm is None and arm_res_map is not None:
                        base_control_arm = arm_name

            embedding_validity.check(code)
            if any(a.get("distortion") for a in arms_spec) and any(d.get("polish_failure", 0) for d in arm_deltas.values()):
                raise RuntimeError("Experimental score-only optimization failed; no baseline substituted for G-3c0")

            # 3. Build Atom Mapping ONCE for the complex
            am = None
            for arm_name in arm_results:
                for s in seeds:
                    res = arm_results[arm_name].get(s)
                    if res and res.conformers:
                        c0 = res.conformers[0]
                        X0 = np.array([at.position for at in c0.complex.atoms], float)
                        try:
                            am = build_atom_mapping(mol, gen_mol=c0.complex, X_ref=ref_X, X_gen=X0)
                            break
                        except Exception:
                            pass
                if am is not None:
                    break

            if am is None:
                print(f"[{code}] ATOM MAPPING FAILED", flush=True)
                for arm in arms_spec:
                    arm_out = Path(arm["out"])
                    rec_dir = arm_out / "records"
                    meta = arm_snapshots[arm["arm"]]["meta_common"]
                    for s in seeds:
                        rec = dict(meta, code=code, seed=s, status="failed", reason="atom_mapping_failed",
                                   top1_dE=None, floor_dE=None, n_rows=0, n_finite=0)
                        (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))
                continue

            g2r = np.asarray(am.gen_to_ref_idx, int)

            # 4. Batched MACE Evaluation across all conformers of all seeds and all arms
            unique_coords = [ref_X]  # index 0 is ref_X
            coord_map = {ref_X.tobytes(): 0}
            arm_seed_indices = {}

            for arm_name in arm_results:
                for s in seeds:
                    res = arm_results[arm_name].get(s)
                    if res and res.conformers:
                        idx_list = []
                        for c in res.conformers:
                            from tools.jgen.atom_export import symbols_for_conformer
                            symbols = symbols_for_conformer(c)
                            assert [symbols[i] for i in g2r] == list(ref_sym), "builder/reference atom mapping mismatch"
                            pos = np.array([at.position for at in c.complex.atoms], float)[g2r]
                            k = pos.tobytes()
                            # a structure and its exact reflection (out_mirror rows) share one MACE evaluation: the energy
                            # is reflection-invariant (checked: equal to 1e-6 kJ/mol). A pool without reflected rows never
                            # holds both, so its energies are unchanged.
                            k = min(k, (pos * np.array([-1.0, 1.0, 1.0])).tobytes())
                            idx = coord_map.get(k)
                            if idx is None:
                                idx = len(unique_coords)
                                coord_map[k] = idx
                                unique_coords.append(pos)
                            idx_list.append(idx)
                        arm_seed_indices[(arm_name, s)] = idx_list

            structures = [(ref_sym, pos, mol.charge, mol.spin_state) for pos in unique_coords]
            energies_ev = mace.energies(structures)
            energies_kj = energies_ev * EV_TO_KJ_MOL
            e_ref = float(energies_kj[0])

            # 5. Write per-unit JSON records and XYZ geometries for each arm
            for arm in arms_spec:
                arm_name = arm["arm"]
                arm_flags = arm.get("flags", {})
                arm_out = Path(arm["out"])
                rec_dir = arm_out / "records"
                xyz_dir = None if a.no_xyz else (arm.get("xyz_dir") or a.xyz_dir or arm_out / "xyz")
                meta = arm_snapshots[arm_name]["meta_common"]
                delta = arm_deltas.get(arm_name, {})

                for s in seeds:
                    res = arm_results.get(arm_name, {}).get(s)
                    if res is None or not res.conformers:
                        rec = dict(meta, code=code, seed=s, status="failed", reason="empty_conformers",
                                   top1_dE=None, floor_dE=None, n_rows=0, n_finite=0,
                                   infeasible_warnings=delta, replay_fallbacks=int(delta.get("replay_fallback", 0)))
                        (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))
                        continue

                    confs = res.conformers
                    indices = arm_seed_indices.get((arm_name, s), [])
                    E_arr = energies_kj[indices]
                    sc = np.array([c.score for c in confs], float)
                    dE = E_arr - e_ref
                    fin = np.isfinite(dE)
                    top = int(np.argmax(np.where(np.isfinite(sc), sc, -np.inf))) if np.isfinite(sc).any() else 0
                    floor_idx = int(np.argmin(np.where(fin, dE, np.inf))) if fin.any() else None
                    from tools.jgen.bench.pool_export import export_pool, coverage_export, coverage_rows, reference_context
                    if coverage_context is None:
                        coverage_context = reference_context(ref_sym, ref_X)
                    pool_manifest = None
                    if xyz_dir:
                        pool_manifest = export_pool(xyz_dir, code, arm_name, s, confs, E_arr,
                                                    ref_sym, ref_X, g2r, top, floor_idx, a.xyz_compression)
                        coverage = coverage_export(pool_manifest, coverage_context)
                    else:
                        coverage = coverage_rows([unique_coords[i] for i in indices], ref_sym, ref_X,
                                                 top, floor_idx, coverage_context)

                    if not fin.any():
                        rec = dict(meta, code=code, seed=s, status="failed", reason="no_finite_energy",
                                   top1_dE=None, floor_dE=None, n_rows=len(dE), n_finite=0, coverage=coverage,
                                   pool_manifest=str(pool_manifest) if pool_manifest else None,
                                   infeasible_warnings=delta, replay_fallbacks=int(delta.get("replay_fallback", 0)))
                        (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))
                        continue

                    floor = float(dE[fin].min())
                    top1 = float(dE[top]) if fin[top] else None

                    top1_bp = float(confs[top].solver_state["bend_prior"]) if getattr(confs[top], "solver_state", None) and "bend_prior" in confs[top].solver_state and confs[top].solver_state["bend_prior"] is not None else None
                    floor_bp = float(confs[floor_idx].solver_state["bend_prior"]) if floor_idx is not None and getattr(confs[floor_idx], "solver_state", None) and "bend_prior" in confs[floor_idx].solver_state and confs[floor_idx].solver_state["bend_prior"] is not None else None

                    status = "ok" if top1 is not None else "failed"
                    reason = "" if status == "ok" else "top1_not_finite"

                    diagnostics = None
                    if arm.get("distortion"):
                        from tools.jgen.g3c0_distortion import measure
                        diagnostics = ([c.solver_state["g3c0_distortion"] for c in confs] if has_ez_multi
                                       else measure(confs, ctx, assignments, ge, scorer, builder.geometry))
                    rec = dict(
                        meta,
                        code=code,
                        seed=s,
                        status=status,
                        reason=reason,
                        top1_dE=top1,
                        floor_dE=floor,
                        top1_bend_prior=top1_bp,
                        floor_bend_prior=floor_bp,
                        n_rows=int(len(dE)),
                        n_finite=int(fin.sum()),
                        ref_energy_kj=e_ref,
                        top1_score=float(sc[top]),
                        distortion=diagnostics,
                        initial_theta_sha256=((arm_initial_digests[arm_name][s] if has_ez_multi else initial_theta_sha256[seeds.index(s)]) if diagnostics is not None else None),
                        coverage=coverage,
                        pool_manifest=str(pool_manifest) if pool_manifest else None,
                        mapping_ambiguous=bool(am.is_ambiguous),
                        num_automorphisms=int(am.num_automorphisms),
                        energies_kj=[float(v) for v in E_arr],
                        scores=[float(v) for v in sc],
                        row_keys=[[c.embed_idx, int(c.assignment_idx), int(c.trial_idx)] for c in confs],
                        n_ik_steps=int(arm.get("n_ik_steps", 200)),
                        replay_chunks=arm_replays.get(arm_name, []),
                        early_stop_saved_steps=sum(max(0, int(r["T"])-int(r["steps_executed"])) for r in arm_replays.get(arm_name, []) if r.get("steps_executed") is not None),
                        wall_solve_s=round(arm_solve_walls.get(arm_name, 0.0), 2),
                        infeasible_warnings=delta,
                        replay_fallbacks=int(delta.get("replay_fallback", 0)),
                    )
                    (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))

            wall_parent = time.perf_counter() - t0
            print(f"[{code}] done in {wall_parent:.1f}s across {len(arms_spec)} arms ({len(unique_coords)} unique structures evaluated in MACE)", flush=True)
        except Exception as exc:
            import traceback
            reason = f"{type(exc).__name__}: {str(exc)[:500]}"
            print(f"[{code}] FAILED: {reason}\n{traceback.format_exc()}", flush=True)
            for arm in arms_spec:
                for seed in seeds:
                    dest = Path(arm["out"])/"records"/f"{code}_{seed}.json"
                    if not dest.exists():
                        rec = dict(arm_snapshots[arm["arm"]]["meta_common"], code=code, seed=seed,
                                   status="failed", reason=reason, top1_dE=None, floor_dE=None, n_rows=0, n_finite=0)
                        dest.write_text(json.dumps(rec))
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            builder._run_joint_ik = original_joint

    prefetcher.shutdown(wait=False)
    (Path(arms_spec[0]["out"]).parent/"embedding_validity.json").write_text(json.dumps(embedding_validity.invalid,indent=2))
    embedding_validity.close()
    print("ALL DONE across arms.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
