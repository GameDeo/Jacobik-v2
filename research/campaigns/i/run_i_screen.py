"""Paired GPU screen runner for Campaign I (Isomer Space: C, A, AS, ASP).

Executes all requested arms paired with Control C in a single invocation on the
120-complex cal+tail panel (C2_PANEL.json), sharing parent load, atom mapping,
and batched MACE evaluation across arms and seeds while preserving 100% bit-for-bit
identity with standalone ComplexBuilder.solve(seed=s) for every arm and seed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

REPO = Path(__file__).resolve().parents[4]
for _p in (
    str(REPO / "src"),
    str(REPO),
    str(REPO / "research" / "refit"),
    str(REPO / "research" / "bench"),
    str(REPO / "research" / "bench" / "tools" / "jdev" / "budget"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.jgen.bench import run as bench
bench._setup_env({})

import networkx as nx
import numpy as np
from rdkit import Chem
import torch

from jacobik.builder import embed as _ce
from jacobik.builder import engine as E
from jacobik.core import profile as P
from jacobik.core.energy import MACEPolarEvaluator
from jacobik.builder.atom_map import build_atom_mapping
from jacobik.ik import dedup as dd
from jacobik.ik import ik_solver as IKM
from jacobik.ik import solver_replay as SR
from jacobik.ik import stereo
from jacobik.core.env import MACE_MODEL_PATH
from jacobik.core.env import build_builder_and_inputs
from tools.jdev import multiseed
from tools.jdev.budget.common import stats_from_db
from tools.jgen import inputs
from tools.jgen.a4t.analyse import _load_parent
from tools.jgen.a4t.metrics import shell_rmsd
from tools.jgen.atom_export import symbols_for_conformer
from tools.jgen.bench.pool_export import coverage_export, coverage_rows, export_pool, reference_context
from tools.jgen.c2.rings import population
from tools.jgen.embedding_validity import EmbeddingValidity
from tools.runkit import geom
from tools.runkit.resources import gpu_lock

EV_TO_KJ_MOL = 96.4853321233
A4B_REF_PATH = Path(os.environ.get("JACOBIK_ISOMER_REFS", "/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/A4b/relax/references.json"))
M3_DBL = {frozenset(("C",)): 1.38, frozenset(("C", "N")): 1.31, frozenset(("N",)): 1.30}

ARM_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "C": {
        "dedup_factored": False,
        "dedup_rmsd": 0.5,
        "dedup_max_dev": 0.0,
        "stereo_candidates": False,
        "stereo_scope": "B",
        "stereo_cost_ceiling": 1.5,
        "out_per_class": False,
        "out_per_class_k": 8,
        "out_per_class_window": 200.0,
    },
    "A": {
        "dedup_factored": True,
        "dedup_rmsd": 0.5,
        "dedup_max_dev": 0.0,
        "stereo_candidates": False,
        "stereo_scope": "B",
        "stereo_cost_ceiling": 1.5,
        "out_per_class": False,
        "out_per_class_k": 8,
        "out_per_class_window": 200.0,
    },
    "AS": {
        "dedup_factored": True,
        "dedup_rmsd": 0.5,
        "dedup_max_dev": 0.0,
        "stereo_candidates": True,
        "stereo_scope": "B",
        "stereo_cost_ceiling": 1.5,
        "out_per_class": False,
        "out_per_class_k": 8,
        "out_per_class_window": 200.0,
    },
    "ASP": {
        "dedup_factored": True,
        "dedup_rmsd": 0.5,
        "dedup_max_dev": 0.0,
        "stereo_candidates": True,
        "stereo_scope": "B",
        "stereo_cost_ceiling": 1.5,
        "out_per_class": True,
        "out_per_class_k": 8,
        "out_per_class_window": 35.0,
    },
}


def set_arm_flags(flags: Dict[str, Any]) -> None:
    """Apply Campaign I flags to live modules."""
    unknown = set(flags) - set(P.FIELDS)
    if unknown:
        raise ValueError(f"Unregistered isomer arm flags: {sorted(unknown)}")
    for name, value in flags.items():
        module, attribute, _, _ = P.FIELDS[name]
        setattr(importlib.import_module(module), attribute, P._coerce(name, value))
    dd.DEDUP_FACTORED = bool(flags.get("dedup_factored", False))
    dd.DEDUP_RMSD = float(flags.get("dedup_rmsd", 0.5))
    dd.DEDUP_MAX_DEV = float(flags.get("dedup_max_dev", 0.0))

    E.STEREO_CANDIDATES = bool(flags.get("stereo_candidates", False))
    E.STEREO_SCOPE = str(flags.get("stereo_scope", "B"))
    E.STEREO_COST_CEILING = float(flags.get("stereo_cost_ceiling", 1.5))
    E.OUT_PER_CLASS = bool(flags.get("out_per_class", False))
    E.OUT_PER_CLASS_K = int(flags.get("out_per_class_k", 8))
    E.OUT_PER_CLASS_WINDOW = float(flags.get("out_per_class_window", 200.0))

    stereo.STEREO_CANDIDATES = E.STEREO_CANDIDATES
    stereo.STEREO_SCOPE = E.STEREO_SCOPE
    stereo.STEREO_COST_CEILING = E.STEREO_COST_CEILING
    stereo.OUT_PER_CLASS = E.OUT_PER_CLASS
    stereo.OUT_PER_CLASS_K = E.OUT_PER_CLASS_K
    stereo.OUT_PER_CLASS_WINDOW = E.OUT_PER_CLASS_WINDOW


def _m3_perceive(sym, X, nbr, rank, metal, ring_bonds):
    cen, dbl = [], []
    for a in range(len(sym)):
        if a == metal or len(nbr[a]) != 4:
            continue
        nb = sorted(nbr[a], key=lambda k: rank[k])
        if len({rank[k] for k in nb}) < 4:
            continue
        v = [X[k] - X[a] for k in nb[:3]]
        v = [u / np.linalg.norm(u) for u in v]
        vol = float(np.linalg.det(np.array(v)))
        if abs(vol) >= 0.2:
            cen.append((a, tuple(nb), metal in nbr[a]))
    for a in range(len(sym)):
        for b in nbr[a]:
            if b <= a or metal in (a, b) or metal in nbr[a] or metal in nbr[b]:
                continue
            key = frozenset((sym[a], sym[b]))
            if key not in M3_DBL or np.linalg.norm(X[a] - X[b]) > M3_DBL[key] or frozenset((a, b)) in ring_bonds:
                continue
            sa, sb = [k for k in nbr[a] if k != b], [k for k in nbr[b] if k != a]
            if not (1 <= len(sa) <= 2 and 1 <= len(sb) <= 2) or len(nbr[a]) > 3 or len(nbr[b]) > 3:
                continue
            if (len(sa) == 2 and rank[sa[0]] == rank[sa[1]]) or (len(sb) == 2 and rank[sb[0]] == rank[sb[1]]):
                continue
            ta, tb = max(sa, key=lambda k: rank[k]), max(sb, key=lambda k: rank[k])
            dbl.append((ta, a, b, tb))
    return cen, dbl


def _m3_labels(X, cen, dbl):
    c = [int(np.sign(np.linalg.det(np.array([X[k] - X[a] for k in nb[:3]])))) for a, nb, _ in cen]
    d = []
    for ta, a, b, tb in dbl:
        u, w = X[ta] - X[a], X[tb] - X[b]
        ax = X[b] - X[a]
        ax = ax / np.linalg.norm(ax)
        u, w = u - (u @ ax) * ax, w - (w @ ax) * ax
        d.append(int(np.sign(u @ w)))
    return np.array(c, dtype=int), np.array(d, dtype=int)


def build_m3_context(sym: Sequence[str], Xr: np.ndarray) -> Dict[str, Any]:
    """Build M3 reference stereo context (exact m3_stereo.py perception)."""
    n = len(sym)
    m = geom.find_metal(sym)
    rc = np.array([geom.rcov(s) for s in sym])
    D = np.linalg.norm(Xr[:, None] - Xr[None], axis=-1)
    B = (D < 1.2 * (rc[:, None] + rc[None])) & ~np.eye(n, dtype=bool)
    B[m, :] = B[:, m] = False
    for d in geom.donors_of(Xr, sym, m):
        B[m, int(d)] = B[int(d), m] = True
    nbr = [set(np.flatnonzero(B[i]).tolist()) for i in range(n)]
    rw = Chem.RWMol()
    for s in sym:
        at = Chem.Atom(s)
        at.SetNoImplicit(True)
        rw.AddAtom(at)
    for i in range(n):
        for j in nbr[i]:
            if j > i:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    G = nx.Graph([(i, j) for i in range(n) for j in nbr[i] if j > i and m not in (i, j)])
    ring_bonds = set()
    for cyc in nx.minimum_cycle_basis(G):
        if len(cyc) <= 7:
            cs = set(cyc)
            ring_bonds |= {frozenset((i, j)) for i in cs for j in nbr[i] if j in cs}
    cen, dbl = _m3_perceive(sym, Xr, nbr, rank, m, ring_bonds)
    c0, d0 = _m3_labels(Xr, cen, dbl)
    return dict(cen=cen, dbl=dbl, c0=c0, d0=d0)


def evaluate_m3_pool(m3_ctx: Dict[str, Any], pool_coords_ref_order: Sequence[np.ndarray], top_idx: int) -> Dict[str, Any]:
    """Evaluate M3 stereo match on a pool of conformers in reference atom order."""
    cen = m3_ctx["cen"]
    dbl = m3_ctx["dbl"]
    c0 = m3_ctx["c0"]
    d0 = m3_ctx["d0"]
    rows = []
    for i, Xg in enumerate(pool_coords_ref_order):
        c, d = _m3_labels(Xg, cen, dbl)
        dc = int((c != c0).sum()) if len(c0) else 0
        dd_ = int((d != d0).sum()) if len(d0) else 0
        match = (dd_ == 0) and (dc == 0 or dc == len(c0))
        dc_bind = int(sum(1 for k, x in enumerate(cen) if x[2] and c[k] != c0[k]))
        n_bind = int(sum(1 for x in cen if x[2]))
        bind_match = (dc_bind == 0 or dc_bind == n_bind)
        rows.append(dict(i=i, top1=(i == top_idx), match=bool(match), dc=dc, dd=dd_, dc_bind=dc_bind, bind_match=bool(bind_match)))
    top = rows[top_idx] if rows and 0 <= top_idx < len(rows) else (rows[0] if rows else None)
    best_match_rank = next((k for k, r in enumerate(rows) if r["match"]), None)
    return dict(
        n_cen=len(cen),
        n_cen_bind=int(sum(1 for x in cen if x[2])),
        n_dbl=len(dbl),
        n_rows=len(rows),
        top1_match=bool(top["match"]) if top else False,
        top1_dc=int(top["dc"]) if top else None,
        top1_dd=int(top["dd"]) if top else None,
        top1_dc_bind=int(top["dc_bind"]) if top else None,
        n_match=int(sum(1 for r in rows if r["match"])),
        present=bool(any(r["match"] for r in rows)),
        n_bind_match=int(sum(1 for r in rows if r["bind_match"])),
        bind_present=bool(any(r["bind_match"] for r in rows)),
        best_match_rank=best_match_rank,
    )


def evaluate_scope_b_pool(
    builder: Any,
    ref_X_gen_order: np.ndarray,
    ref_syms_gen_order: Sequence[str],
    confs: Sequence[Any],
    top_idx: int,
) -> Dict[str, Any]:
    """Evaluate Scope-B perceived stereo match against crystal reference."""
    elements = stereo.perceive_complex_stereo(builder, scope="B")
    if not elements:
        return dict(
            n_elements=0,
            crystal_stereo_label="none",
            achiral_env=True,
            top1_match=True,
            n_match=len(confs),
            present=bool(len(confs) > 0),
            best_match_rank=0 if confs else None,
        )
    cryst_map = stereo.compute_conformer_stereo_labels(elements, ref_X_gen_order, ref_syms_gen_order)
    cryst_str = ",".join(f"{k}:{v}" for k, v in sorted(cryst_map.items())) if cryst_map else "none"
    achiral_env = stereo.is_achiral_geometry(builder.geometry)
    # Build mirror-inverted crystal map (chiral signs flipped, E/Z unchanged)
    inv_map = {}
    for k, v in cryst_map.items():
        if v == "+":
            inv_map[k] = "-"
        elif v == "-":
            inv_map[k] = "+"
        else:
            inv_map[k] = v
    inv_str = ",".join(f"{k}:{v}" for k, v in sorted(inv_map.items())) if inv_map else "none"
    valid_targets = {cryst_str, inv_str} if achiral_env else {cryst_str}

    matches = [c.stereo_label in valid_targets for c in confs]
    top_match = bool(matches[top_idx]) if confs and 0 <= top_idx < len(confs) else False
    best_rank = next((k for k, m in enumerate(matches) if m), None)
    return dict(
        n_elements=len(elements),
        crystal_stereo_label=cryst_str,
        crystal_mirror_label=inv_str if achiral_env else cryst_str,
        achiral_env=bool(achiral_env),
        top1_match=top_match,
        n_match=int(sum(matches)),
        present=bool(any(matches)),
        best_match_rank=best_rank,
    )


def evaluate_m2_dupes(
    deduper: dd.StructureDeduper,
    confs: Sequence[Any],
    pair_cache: Dict[Tuple[bytes, bytes], Tuple[bool, float]],
    th: float = 0.5,
) -> Dict[str, Any]:
    """Compute M2 duplicate counts among returned conformers using exact automorphism + factored upper bound."""
    if len(confs) <= 1:
        return dict(
            is_capped=bool(deduper.is_capped),
            n_rows=len(confs),
            n_dup_pairs=0,
            n_rows_dup_of_better=0,
            dup_rows=[],
            top1_is_dup=False,
        )
    heavy = deduper.heavy.cpu().numpy()
    Xh = [np.asarray([a.position for a in c.complex.atoms], float)[heavy] for c in confs]
    Bt = torch.as_tensor(np.stack(Xh), dtype=deduper.dtype, device=deduper.device)
    Bt = Bt - Bt.mean(1, keepdim=True)
    Rr = deduper._radial(Bt)

    n_dup_pairs = 0
    dup_rows = set()
    for i in range(len(Xh)):
        ki = Xh[i].tobytes()
        for j in range(i):
            # Radial lower bound pruning
            if float(((Rr[j] - Rr[i]) ** 2).mean().sqrt()) > th:
                continue
            kj = Xh[j].tobytes()
            key = (kj, ki) if kj <= ki else (ki, kj)
            cached = pair_cache.get(key)
            if cached is None:
                is_dup = bool(deduper._full_any(Bt[i], Bt[j : j + 1], th))
                rmsd_val = float(th) if is_dup else 999.0
                if not is_dup and deduper.is_capped:
                    is_dup, rmsd_val, _, _ = deduper.factored_sym_upper(Xh[j], Xh[i], th=th)
                pair_cache[key] = (is_dup, float(rmsd_val))
            else:
                is_dup, rmsd_val = cached
            if is_dup:
                n_dup_pairs += 1
                dup_rows.add(i)
    return dict(
        is_capped=bool(deduper.is_capped),
        n_rows=len(confs),
        n_dup_pairs=int(n_dup_pairs),
        n_rows_dup_of_better=len(dup_rows),
        dup_rows=sorted(dup_rows),
        top1_is_dup=(0 in dup_rows),
    )


def audit_out_per_class(confs: Sequence[Any], z_norm: float, k_limit: int = 8, class_window: float = 35.0) -> Dict[str, Any]:
    """Audit per-class output retention on returned conformers."""
    if not confs:
        return dict(n_classes=0, max_rows_per_class=0, min_rows_per_class=0, all_within_k=True, all_within_window=True, class_counts={})
    zn = float(z_norm) if z_norm and z_norm > 0 else 1.0
    global_win = E.OUT_WINDOW / zn
    c_win = class_window / zn
    scores = [float(getattr(c.score, "total", c.score)) for c in confs]
    best_score = max(scores)
    by_class: Dict[str, List[float]] = defaultdict(list)
    for c, sc in zip(confs, scores):
        by_class[c.class_label].append(sc)
    counts = {cl: len(scs) for cl, scs in by_class.items()}
    all_within_k = all(cnt <= k_limit for cnt in counts.values())
    all_within_win = True
    for cl, scs in by_class.items():
        cl_best = max(scs)
        if cl_best < best_score - global_win - 1e-6:
            all_within_win = False
        if any(s < cl_best - c_win - 1e-6 for s in scs):
            all_within_win = False
    return dict(
        n_classes=len(counts),
        max_rows_per_class=max(counts.values()) if counts else 0,
        min_rows_per_class=min(counts.values()) if counts else 0,
        mean_rows_per_class=float(np.mean(list(counts.values()))) if counts else 0.0,
        all_within_k=bool(all_within_k),
        all_within_window=bool(all_within_win),
        class_counts=counts,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Output root directory")
    ap.add_argument("--arms", default="C,A,AS,ASP", help="Comma-separated arms")
    ap.add_argument("--seeds", default="11,12,13", help="Comma-separated seeds")
    ap.add_argument("--only", default="", help="Optional comma-separated subset of codes")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--resume", action="store_true", help="Skip parents whose records already exist for all arms/seeds")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    arm_names = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not arm_names:
        ap.error("At least one arm is required")
    for name in arm_names:
        if name not in ARM_DEFINITIONS:
            ap.error(f"Unknown arm {name}; allowed: {list(ARM_DEFINITIONS)}")
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    codes, population_meta = population()
    if a.only:
        req = [x.strip() for x in a.only.split(",") if x.strip()]
        assert set(req) <= set(codes), f"Unknown codes not in C2_PANEL: {set(req) - set(codes)}"
        codes = req

    spec = [
        dict(arm=name, flags=ARM_DEFINITIONS[name], out=str(a.out / name), seed_budget="three_w16")
        for name in arm_names
    ]
    (a.out / "arms.json").write_text(json.dumps(spec, indent=2) + "\n")

    if a.device == "cuda":
        smi = subprocess.check_output(["nvidia-smi"], text=True)
        (a.out / "nvidia-smi.txt").write_text(smi)
        torch.cuda.set_per_process_memory_fraction(0.80)

    os.environ.setdefault("JACOBIK_MACE_BATCH_ATOMS", "900")
    mace = MACEPolarEvaluator(
        model="polar-1-s",
        device=a.device,
        dtype="float64",
        model_path=str(MACE_MODEL_PATH),
        mode="energy-only",
        threads=1,
    ).load()
    mace._batch_atoms = 900
    stats = stats_from_db(inputs.DB_PATH)
    gfn2_info = bench._assert_gfn2()
    counter = bench._InfeasibleCounter()
    logging.getLogger().addHandler(counter)
    embedding_validity = EmbeddingValidity(E).install()

    a4b_refs = json.loads(A4B_REF_PATH.read_text()) if A4B_REF_PATH.exists() else {}
    manifest = json.load(open(inputs.FRAME_V1 / "manifest.json"))
    frozen = manifest.get("base_provenance", {})
    db_sha = inputs.sha256_file(inputs.DB_PATH)
    shards = inputs.frame_shards()
    commit = bench._git_commit()

    arm_snapshots = {}
    for arm in spec:
        arm_name = arm["arm"]
        arm_flags = arm["flags"]
        arm_out = Path(arm["out"])
        (arm_out / "records").mkdir(parents=True, exist_ok=True)
        live = dict(P.PRODUCTION_CONFIG)
        for k, (mod, attr, default_val, _) in P.FIELDS.items():
            try:
                live[k] = getattr(importlib.import_module(mod), attr)
            except Exception:
                live[k] = P._VALUES.get(k, default_val)
        for k, v in bench.CONTROL_OVERRIDES.items():
            live[k] = v
        for k, v in arm_flags.items():
            live[k] = v
        differs = {k: live[k] for k in P.FIELDS if live.get(k) != P.FIELDS[k][2]}
        arm_snapshots[arm_name] = dict(
            fingerprint=P.fingerprint(live),
            differs_from_profile=differs,
            meta_common=dict(
                gfn2=gfn2_info,
                arm=arm_name,
                panel="cal_tail_120",
                flags=arm_flags,
                seed_budget="three_w16",
                W=16,
                fingerprint=P.fingerprint(live),
                fingerprint_fields_differing=differs,
                code_git_commit=commit,
                frame_sha256=frozen.get("frame_sha256"),
                constraints_db_sha256=db_sha,
                constraints_db_frozen=frozen.get("constraints_db_sha256"),
                mace_model="polar-1-s float64",
                mace_model_sha256=inputs.sha256_file(MACE_MODEL_PATH),
                production_fingerprint=P.PRODUCTION_FINGERPRINT,
                bench_runner_sha256=inputs.sha256_file(Path(__file__)),
                embedding_validity_guard="actual_gfn2_call_and_cached_failure_v1",
                gpu_name=(torch.cuda.get_device_name(0) if a.device == "cuda" else None),
                device=a.device,
                machine=socket.gethostname(),
            ),
        )

    from tools.jgen.c2 import rings as _r  # noqa
    sys.path.insert(0, str(Path(__file__).parent))
    import fast_patches
    fast_patches.install()

    # Install within-parent pre-assembly IK cache hooks
    orig_build_shell = E.ComplexBuilder._build_shell
    orig_solve_for_assignments = E.ComplexBuilder._solve_for_assignments
    orig_check_feas = stereo.check_variant_feasibility

    shell_cache: Dict[Any, Any] = {}
    ik_cache: Dict[Any, Any] = {}
    feas_cache: Dict[Any, Any] = {}
    reuse_cache: Dict[Any, Any] = {}
    current_seed = [None]
    current_fixes = [None]
    saved_wall = [0.0]

    def get_fixes_key(b):
        return tuple(
            (
                repr(getattr(lig, "ez_fix", None)),
                repr(getattr(lig, "chiral_fix", None)),
                bool(getattr(lig, "_strict_stereo", False)),
            )
            for lig in b.ligands
        )

    def cached_check_feas(variant, b, elements=None, *args, **kwargs):
        k = variant.class_id
        if k in feas_cache:
            ok, dt = feas_cache[k]
            saved_wall[0] += dt
            return ok
        t0 = time.perf_counter()
        ok = orig_check_feas(variant, b, elements, *args, **kwargs)
        dt = time.perf_counter() - t0
        feas_cache[k] = (ok, dt)
        return ok

    def cached_build_shell(self, fast_embed, debug, embed_seed=None, chelate_embed=False, embed=None):
        fk = (get_fixes_key(self), fast_embed, embed_seed, chelate_embed, embed)
        current_fixes[0] = fk
        if fk in shell_cache:
            res, dt = shell_cache[fk]
            saved_wall[0] += dt
            return res
        t0 = time.perf_counter()
        res = orig_build_shell(self, fast_embed, debug, embed_seed=embed_seed, chelate_embed=chelate_embed, embed=embed)
        dt = time.perf_counter() - t0
        shell_cache[fk] = (res, dt)
        return res

    def cached_solve_for_assignments(self, ctx, assignments, beam_width, n_ik_steps):
        ik_key = (current_seed[0], current_fixes[0], beam_width, n_ik_steps)
        if ik_key in ik_cache:
            c_data, dt = ik_cache[ik_key]
            saved_wall[0] += dt
            (
                fks, X_full, thetas_final, scores, converged, beam_width_total,
                candidate_idx_per_b, solve_trace, scorer_joint, last_bp, score_norm,
            ) = c_data
            ctx.score_norm = score_norm
            self._last_bend_prior = last_bp
            conformers = self._assemble_conformers(
                ctx, fks, X_full, thetas_final, scores, converged, beam_width_total, candidate_idx_per_b
            )
            E.attach_solver_state(conformers, thetas_final, last_bp)
            result = E.EnsembleResult(conformers=conformers, solve_trace=solve_trace, scorer=scorer_joint)
            if len(assignments) > 1:
                result.assignment_search = dict(
                    n_candidates=len(assignments),
                    candidate_assignments=[arr.tolist() for arr in assignments],
                )
            return result

        t0 = time.perf_counter()
        do_out_per_class = getattr(self, "out_per_class", None)
        if do_out_per_class is None:
            do_out_per_class = E.OUT_PER_CLASS
        if do_out_per_class:
            k_out = E.OUT_PER_CLASS_K if E.OUT_PER_CLASS_K is not None else 8
            min_bw = (k_out + max(ctx.n_coord_samples, 1) - 1) // max(ctx.n_coord_samples, 1)
            if beam_width < min_bw:
                beam_width = min_bw
        symmetric_groups = ctx.symmetric_donor_groups
        q3_symmetric_groups = [
            g for g in symmetric_groups if len({ctx.all_donor_groups[d][0] for d in g}) == 1
        ]
        reuse_key = (current_fixes[0], beam_width, len(assignments))
        reuse_entry = reuse_cache.get(reuse_key)
        if reuse_entry is None:
            reuse_obj = multiseed._Reuse()
            t_b0 = time.perf_counter()
            with reuse_obj:
                (
                    fks, all_theta_inits, all_donor_pos_idxs, ge_joint, scorer_joint,
                    X_full, ideal_directions_batched, candidate_idx_per_b,
                ) = self._beam_search_all_candidates(ctx, assignments, beam_width)
            t_beam_first = time.perf_counter() - t_b0
            reuse_obj.next_pass()
            reuse_cache[reuse_key] = (reuse_obj, t_beam_first)
        else:
            reuse_obj, t_beam_first = reuse_entry
            reuse_obj.pos = {}
            t_b0 = time.perf_counter()
            with reuse_obj:
                (
                    fks, all_theta_inits, all_donor_pos_idxs, ge_joint, scorer_joint,
                    X_full, ideal_directions_batched, candidate_idx_per_b,
                ) = self._beam_search_all_candidates(ctx, assignments, beam_width)
            t_beam_reused = time.perf_counter() - t_b0
            saved_wall[0] += max(0.0, t_beam_first - t_beam_reused)

        beam_width_total = beam_width * len(assignments)
        keep = None
        ctx.early_cull = None
        if E.CLASS_BUDGET == "split":
            keep = self._class_budget_rows(ctx, assignments, beam_width)
        elif E.CLASS_BUDGET == "early_cull":
            n_cls = self._n_topological_classes(ctx, assignments)
            if n_cls is not None and n_cls < len(assignments):
                ctx.early_cull = (E.EARLY_CULL_AT, n_cls / len(assignments))
        if keep is not None:
            keep_t = torch.as_tensor(keep, dtype=torch.long, device=ctx.device)
            all_theta_inits = [th[keep_t] for th in all_theta_inits]
            if X_full.dim() == 3:
                X_full = X_full[keep_t]
            ideal_directions_batched = ideal_directions_batched[keep_t]
            candidate_idx_per_b = np.asarray(candidate_idx_per_b)[keep]
            beam_width_total = len(keep) // max(ctx.n_coord_samples, 1)
        thetas_final, converged, scores, solve_trace = self._run_joint_ik(
            ctx, fks, scorer_joint, ge_joint, X_full, all_theta_inits,
            all_donor_pos_idxs, ideal_directions_batched, symmetric_groups,
            q3_symmetric_groups, beam_width_total, n_ik_steps, candidate_idx_per_b,
        )
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        score_norm = float(scorer_joint._agg_norm()) if hasattr(scorer_joint, "_agg_norm") else 0.0
        ctx.score_norm = score_norm
        last_bp = getattr(self, "_last_bend_prior", None)
        dt = time.perf_counter() - t0
        if reuse_entry is not None:
            dt += max(0.0, t_beam_first - t_beam_reused)
        ik_cache[ik_key] = (
            (
                fks, X_full, thetas_final, scores, converged, beam_width_total,
                candidate_idx_per_b, solve_trace, scorer_joint, last_bp, score_norm,
            ),
            dt,
        )
        conformers = self._assemble_conformers(
            ctx, fks, X_full, thetas_final, scores, converged, beam_width_total, candidate_idx_per_b
        )
        E.attach_solver_state(conformers, thetas_final, last_bp)
        result = E.EnsembleResult(conformers=conformers, solve_trace=solve_trace, scorer=scorer_joint)
        if len(assignments) > 1:
            result.assignment_search = dict(
                n_candidates=len(assignments),
                candidate_assignments=[arr.tolist() for arr in assignments],
            )
        return result

    E.ComplexBuilder._build_shell = cached_build_shell
    E.ComplexBuilder._solve_for_assignments = cached_solve_for_assignments
    stereo.check_variant_feasibility = cached_check_feas

    prefetcher = ThreadPoolExecutor(max_workers=1)
    timing_ledger: Dict[str, Any] = {}
    t_screen_start = time.perf_counter()

    try:
        with gpu_lock("GeminiI") if a.device == "cuda" else nullcontext():
            torch.use_deterministic_algorithms(True)
            print(f"Starting Campaign I screen: {len(codes)} parents, arms={arm_names}, seeds={seeds}", flush=True)

            for idx, code in enumerate(codes):
                if a.resume and all(
                    (a.out / arm["arm"] / "records" / f"{code}_{s}.json").exists()
                    for arm in spec
                    for s in seeds
                ):
                    print(f"[{idx+1}/{len(codes)} {code}] already complete (--resume), skipping.", flush=True)
                    continue

                shell_cache.clear()
                ik_cache.clear()
                feas_cache.clear()
                reuse_cache.clear()

                set_arm_flags(ARM_DEFINITIONS[arm_names[0]])
                embedding_validity.parent = code
                t_parent0 = time.perf_counter()
                shard, ds = shards[code]

                t_pref0 = time.perf_counter()
                mol, builder, err_msg = bench._prefetch_parent(code, shard, ds)
                t_prefetch = time.perf_counter() - t_pref0

                if builder is None:
                    print(f"[{idx+1}/{len(codes)} {code}] FAILED: {err_msg}", flush=True)
                    for arm in spec:
                        rec_dir = Path(arm["out"]) / "records"
                        meta = arm_snapshots[arm["arm"]]["meta_common"]
                        for s in seeds:
                            rec = dict(
                                meta,
                                code=code,
                                seed=s,
                                status="failed",
                                reason=f"Unmappable: {err_msg}",
                                top1_dE=None,
                                floor_dE=None,
                                n_rows=0,
                                n_finite=0,
                            )
                            (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))
                    continue

                ref_sym = [x.symbol for x in mol.atoms]
                ref_X = np.array([x.position for x in mol.atoms], float)
                _, _, parent_donors = _load_parent(code)

                try:
                    arm_results: Dict[str, Dict[int, Any]] = {arm["arm"]: {} for arm in spec}
                    arm_standalone_walls: Dict[Tuple[str, int], float] = {}
                    arm_actual_walls: Dict[Tuple[str, int], float] = {}
                    arm_merges: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
                    arm_deltas: Dict[str, Dict[str, int]] = {}

                    with multiseed.safe_cuda_eigh_chunking():
                        for arm in spec:
                            arm_name = arm["arm"]
                            arm_flags = arm["flags"]
                            set_arm_flags(arm_flags)
                            counts_before = dict(counter.counts)

                            for s in seeds:
                                current_seed[0] = s
                                saved_wall[0] = 0.0
                                t_s0 = time.perf_counter()
                                try:
                                    res = builder.solve(device=a.device, seed=s, stats=stats, angle_bends=True)
                                except torch.cuda.OutOfMemoryError as exc:
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    print(f"[{code}:{arm_name}:s{s}] OOM: {exc}", flush=True)
                                    res = E.EnsembleResult(conformers=[], failure=f"OOM: {exc}")
                                if a.device == "cuda":
                                    torch.cuda.synchronize()
                                actual_dt = time.perf_counter() - t_s0
                                standalone_dt = actual_dt + saved_wall[0] + (t_prefetch / len(seeds))
                                arm_results[arm_name][s] = res
                                arm_actual_walls[(arm_name, s)] = actual_dt
                                arm_standalone_walls[(arm_name, s)] = standalone_dt
                                # Record deduper merges if available
                                merges_list = []
                                if res and res.conformers:
                                    try:
                                        ded = dd.deduper_for(res.conformers[0].complex)
                                        merges_list = list(getattr(ded, "merges", []))
                                    except Exception:
                                        pass
                                arm_merges[(arm_name, s)] = merges_list

                            arm_deltas[arm_name] = {k: counter.counts[k] - counts_before[k] for k in counter.counts}

                    embedding_validity.check(code)

                    # Build Atom Mapping once for the complex
                    am = None
                    c0_mol = None
                    for arm_name in arm_results:
                        for s in seeds:
                            res = arm_results[arm_name].get(s)
                            if res and res.conformers:
                                c0 = res.conformers[0]
                                c0_mol = c0.complex
                                X0 = np.array([at.position for at in c0.complex.atoms], float)
                                try:
                                    am = build_atom_mapping(mol, gen_mol=c0.complex, X_ref=ref_X, X_gen=X0)
                                    break
                                except Exception:
                                    pass
                        if am is not None:
                            break

                    if am is None or c0_mol is None:
                        print(f"[{code}] ATOM MAPPING FAILED", flush=True)
                        for arm in spec:
                            rec_dir = Path(arm["out"]) / "records"
                            meta = arm_snapshots[arm["arm"]]["meta_common"]
                            for s in seeds:
                                rec = dict(
                                    meta,
                                    code=code,
                                    seed=s,
                                    status="failed",
                                    reason="atom_mapping_failed",
                                    top1_dE=None,
                                    floor_dE=None,
                                    n_rows=0,
                                    n_finite=0,
                                )
                                (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))
                        continue

                    g2r = np.asarray(am.gen_to_ref_idx, int)
                    r2g = np.asarray(am.ref_to_gen_idx, int)
                    ref_X_gen_order = ref_X[r2g]
                    ref_sym_gen_order = [ref_sym[i] for i in r2g]

                    # Batched MACE Evaluation across all unique conformers of all seeds and arms
                    unique_coords = [ref_X]  # index 0 is ref_X in reference atom order
                    coord_map = {ref_X.tobytes(): 0}
                    arm_seed_indices: Dict[Tuple[str, int], List[int]] = {}

                    for arm_name in arm_results:
                        for s in seeds:
                            res = arm_results[arm_name].get(s)
                            if res and res.conformers:
                                idx_list = []
                                for c in res.conformers:
                                    symbols = symbols_for_conformer(c)
                                    assert [symbols[i] for i in g2r] == list(ref_sym), "builder/ref symbol mismatch"
                                    pos = np.array([at.position for at in c.complex.atoms], float)[g2r]
                                    k = pos.tobytes()
                                    u_idx = coord_map.get(k)
                                    if u_idx is None:
                                        u_idx = len(unique_coords)
                                        coord_map[k] = u_idx
                                        unique_coords.append(pos)
                                    idx_list.append(u_idx)
                                arm_seed_indices[(arm_name, s)] = idx_list

                    structures = [(ref_sym, pos, mol.charge, mol.spin_state) for pos in unique_coords]
                    energies_ev = mace.energies(structures)
                    energies_kj = energies_ev * EV_TO_KJ_MOL
                    e_ref = float(energies_kj[0])

                    # Compute shell_rmsd for all unique structures once
                    unique_shells = [0.0] + [
                        float(shell_rmsd(pos, ref_X, ref_sym, mol.metal_idx - 1, parent_donors))
                        for pos in unique_coords[1:]
                    ]

                    # Control union distinct count for C2 compatibility
                    c_union_X = []
                    c_union_scores = []
                    for s in seeds:
                        c_res = arm_results.get("A", arm_results.get("C", {})).get(s)
                        if c_res and c_res.conformers:
                            c_idxs = arm_seed_indices.get(("A" if "A" in arm_results else "C", s), [])
                            c_union_X.extend(unique_coords[i] for i in c_idxs)
                            c_union_scores.extend(float(c.score) for c in c_res.conformers)
                    control_union_distinct = None
                    union_capped = False
                    if c_union_X:
                        heavy_mask = np.asarray(ref_sym) != "H"
                        X_cu = np.asarray(c_union_X, float)
                        perms_cu, union_capped = geom.automorphisms(ref_X, ref_sym, mol.metal_idx - 1)
                        keep_cu = []
                        for i_cu in np.argsort(-np.asarray(c_union_scores), kind="stable"):
                            if not keep_cu or not np.any(
                                geom.sym_rmsd_to_ref(X_cu[keep_cu][:, heavy_mask], X_cu[i_cu, heavy_mask], perms_cu) <= 0.5
                            ):
                                keep_cu.append(int(i_cu))
                        control_union_distinct = len(keep_cu)

                    # Build M3 stereo context and deduplicator for M2 audit once per parent
                    Xr_m3 = np.asarray(a4b_refs[code]["X"], float) if code in a4b_refs else ref_X
                    m3_ctx = build_m3_context(ref_sym, Xr_m3)
                    m2_deduper = dd.StructureDeduper(c0_mol, device=a.device)
                    m2_pair_cache: Dict[Tuple[bytes, bytes], Tuple[bool, float]] = {}
                    coverage_context = reference_context(ref_sym, ref_X)

                    # Write records and XYZ pools per arm and seed
                    for arm in spec:
                        arm_name = arm["arm"]
                        arm_out = Path(arm["out"])
                        rec_dir = arm_out / "records"
                        xyz_dir = arm_out / "xyz"
                        meta = arm_snapshots[arm_name]["meta_common"]
                        delta = arm_deltas.get(arm_name, {})

                        for s in seeds:
                            res = arm_results.get(arm_name, {}).get(s)
                            if res is None or not res.conformers:
                                rec = dict(
                                    meta,
                                    code=code,
                                    seed=s,
                                    status="failed",
                                    reason=(res.failure if res and res.failure else "empty_conformers"),
                                    top1_dE=None,
                                    floor_dE=None,
                                    n_rows=0,
                                    n_finite=0,
                                    infeasible_warnings=delta,
                                    replay_fallbacks=int(delta.get("replay_fallback", 0)),
                                )
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

                            pool_manifest = export_pool(
                                xyz_dir, code, arm_name, s, confs, E_arr,
                                ref_sym, ref_X, g2r, top, floor_idx, "xz"
                            )
                            coverage = coverage_export(pool_manifest, coverage_context)

                            shells = [unique_shells[i] for i in indices]
                            c2_metrics = dict(
                                minimum_shell_rmsd_A=min(shells) if shells else None,
                                shell_rmsd_A=shells,
                                control_union_distinct=control_union_distinct,
                                union_symmetry_capped=bool(union_capped),
                            )
                            c2_timing = dict(
                                shared_prepare_s=t_prefetch,
                                end_to_end_s=float(arm_standalone_walls.get((arm_name, s), 0.0)),
                                actual_wall_s=float(arm_actual_walls.get((arm_name, s), 0.0)),
                            )

                            # Campaign I specific metrics
                            pool_coords_ref = [unique_coords[i] for i in indices]
                            m3_res = evaluate_m3_pool(m3_ctx, pool_coords_ref, top)
                            scope_b_res = evaluate_scope_b_pool(builder, ref_X_gen_order, ref_sym_gen_order, confs, top)
                            m2_res = evaluate_m2_dupes(m2_deduper, confs, m2_pair_cache, th=0.5)
                            z_norm = float(res.scorer._agg_norm()) if (res.scorer is not None and hasattr(res.scorer, "_agg_norm")) else 1.0
                            opc_audit = audit_out_per_class(confs, z_norm=z_norm, k_limit=8, class_window=35.0)

                            i_metrics = dict(
                                class_labels=[c.class_label for c in confs],
                                stereo_labels=[c.stereo_label for c in confs],
                                requested_stereo_labels=[getattr(c, "requested_stereo_label", "none") for c in confs],
                                stereo_variant_idxs=[getattr(c, "stereo_variant_idx", None) for c in confs],
                                n_distinct_classes=len({c.class_label for c in confs}),
                                n_distinct_stereo_labels=len({c.stereo_label for c in confs}),
                                n_stereo_variants_solved=(
                                    int(res.solve_trace.get("n_stereo_variants_solved", 1))
                                    if isinstance(res.solve_trace, dict)
                                    else 1
                                ),
                                merges_logged=len(arm_merges.get((arm_name, s), [])),
                                merges_sample=arm_merges.get((arm_name, s), [])[:50],
                                out_per_class_audit=opc_audit,
                                m2_dupes=m2_res,
                                m3_stereo=m3_res,
                                scope_b_stereo=scope_b_res,
                            )

                            if not fin.any():
                                rec = dict(
                                    meta,
                                    code=code,
                                    seed=s,
                                    status="failed",
                                    reason="no_finite_energy",
                                    top1_dE=None,
                                    floor_dE=None,
                                    n_rows=len(dE),
                                    n_finite=0,
                                    coverage=coverage,
                                    c2_metrics=c2_metrics,
                                    c2_timing=c2_timing,
                                    i_metrics=i_metrics,
                                    pool_manifest=str(pool_manifest),
                                    infeasible_warnings=delta,
                                    replay_fallbacks=int(delta.get("replay_fallback", 0)),
                                )
                                (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))
                                continue

                            floor_val = float(dE[fin].min())
                            top1_val = float(dE[top]) if fin[top] else None
                            status = "ok" if top1_val is not None else "failed"
                            reason = "" if status == "ok" else "top1_not_finite"

                            rec = dict(
                                meta,
                                code=code,
                                seed=s,
                                status=status,
                                reason=reason,
                                top1_dE=top1_val,
                                floor_dE=floor_val,
                                n_rows=int(len(dE)),
                                n_finite=int(fin.sum()),
                                ref_energy_kj=e_ref,
                                top1_score=float(sc[top]),
                                coverage=coverage,
                                c2_metrics=c2_metrics,
                                c2_timing=c2_timing,
                                i_metrics=i_metrics,
                                pool_manifest=str(pool_manifest),
                                mapping_ambiguous=bool(am.is_ambiguous),
                                num_automorphisms=int(am.num_automorphisms),
                                energies_kj=[float(v) for v in E_arr],
                                scores=[float(v) for v in sc],
                                row_keys=[[c.embed_idx, int(c.assignment_idx), int(c.trial_idx)] for c in confs],
                                n_ik_steps=200,
                                wall_solve_s=round(float(arm_standalone_walls.get((arm_name, s), 0.0)), 3),
                                actual_wall_s=round(float(arm_actual_walls.get((arm_name, s), 0.0)), 3),
                                infeasible_warnings=delta,
                                replay_fallbacks=int(delta.get("replay_fallback", 0)),
                            )
                            (rec_dir / f"{code}_{s}.json").write_text(json.dumps(rec))

                    wall_parent = time.perf_counter() - t_parent0
                    timing_ledger[code] = dict(
                        wall_parent_s=round(wall_parent, 3),
                        unique_mace_structures=len(unique_coords),
                        standalone_walls={f"{k[0]}_s{k[1]}": round(v, 3) for k, v in arm_standalone_walls.items()},
                        actual_walls={f"{k[0]}_s{k[1]}": round(v, 3) for k, v in arm_actual_walls.items()},
                    )
                    (a.out / "timing.json").write_text(
                        json.dumps(dict(wall_s=round(time.perf_counter() - t_screen_start, 2), parents=timing_ledger), indent=2)
                    )
                    print(
                        f"[{idx+1}/{len(codes)} {code}] done in {wall_parent:.1f}s "
                        f"({len(unique_coords)} unique MACE structs)",
                        flush=True,
                    )

                except Exception as exc:
                    import traceback
                    reason = f"{type(exc).__name__}: {str(exc)[:500]}"
                    print(f"[{code}] FAILED: {reason}\n{traceback.format_exc()}", flush=True)
                    for arm in spec:
                        for s in seeds:
                            dest = Path(arm["out"]) / "records" / f"{code}_{s}.json"
                            if not dest.exists():
                                rec = dict(
                                    arm_snapshots[arm["arm"]]["meta_common"],
                                    code=code,
                                    seed=s,
                                    status="failed",
                                    reason=reason,
                                    top1_dE=None,
                                    floor_dE=None,
                                    n_rows=0,
                                    n_finite=0,
                                )
                                dest.write_text(json.dumps(rec))
                    gc.collect()
                    if a.device == "cuda":
                        torch.cuda.empty_cache()

    finally:
        E.ComplexBuilder._build_shell = orig_build_shell
        E.ComplexBuilder._solve_for_assignments = orig_solve_for_assignments
        stereo.check_variant_feasibility = orig_check_feas
        prefetcher.shutdown(wait=False)
        (a.out / "embedding_validity.json").write_text(json.dumps(embedding_validity.invalid, indent=2))
        embedding_validity.close()
        (a.out / "timing.json").write_text(
            json.dumps(dict(wall_s=round(time.perf_counter() - t_screen_start, 2), parents=timing_ledger), indent=2)
        )

    print(f"ALL DONE in {time.perf_counter() - t_screen_start:.1f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
