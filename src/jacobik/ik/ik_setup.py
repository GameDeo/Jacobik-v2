"""Solve-state setup, finalization, and S2b/group-weight helpers for JointIKSolver."""

from __future__ import annotations

import atexit
import hashlib
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
import json
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

from jacobik.ik.kabsch import _ref_rows


_PROC_POOL: Optional[ProcessPoolExecutor] = None
_PROC_POOL_N: int = 0


def _worker_init() -> None:
    torch.set_num_threads(1)


def _shutdown_pool() -> None:
    global _PROC_POOL, _PROC_POOL_N
    if _PROC_POOL is not None:
        _PROC_POOL.shutdown(wait=False, cancel_futures=True)
    _PROC_POOL, _PROC_POOL_N = None, 0


atexit.register(_shutdown_pool)


def _get_pool(n: int) -> ProcessPoolExecutor:
    global _PROC_POOL, _PROC_POOL_N
    if _PROC_POOL is None or _PROC_POOL_N != n:
        _shutdown_pool()
        _PROC_POOL = ProcessPoolExecutor(
            max_workers=n,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
        )
        _PROC_POOL_N = n
    return _PROC_POOL


def _solve_chunk_worker(payload: bytes) -> bytes:
    # Plain pickle (tensors by value): the solver holds hundreds of small
    # tensors, which would each become a shared-memory handle otherwise.
    solver, args, kw = pickle.loads(payload)
    return pickle.dumps(solver._solve_impl(*args, **kw))



def merge_chunk_traces(traces: list) -> dict:
    """Concatenate per-chunk solve traces along the batch axis (axis 1 of each (T, B) array).

    Chunks can stop at different steps (plateau or finish early stop), so their (T, B_i) arrays may differ in T.
    The shorter ones are padded with NaN at the end, the convention already used for culled rows, so no step is
    invented. Where all lengths match this is the plain concatenation (bit-identical).
    """
    keys = list(traces[0])
    T = max(t[k].shape[0] for t in traces for k in keys)
    out = {}
    for k in keys:
        parts = []
        for t in traces:
            a = t[k]
            if a.shape[0] < T:
                dt = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64
                pad = np.full((T - a.shape[0],) + a.shape[1:], np.nan, dtype=dt)
                a = np.concatenate([a.astype(dt, copy=False), pad], axis=0)
            parts.append(a)
        out[k] = np.concatenate(parts, axis=1)
    return out


def _ik():
    from jacobik.ik import ik_solver

    return ik_solver


#: S2b solver-state schema tag written into the evaluation pools (capture only; the solver does not read it).
S2B_STATE_SCHEMA = "s2b_state.1"


def bend_state_constants(fks, bend_cfg, theta_widths, per_z=None) -> dict:
    """Constants of the angle-bend prior as JointIKSolver._bend_prior applies them, one entry per bend column of every
    ligand (ligand index li; column 6 + n_joints + j of that ligand's theta vector). Free zones are clamped to the
    prior's free_deg exactly as _bend_prior clamps them. With bend_cfg None the solver adds no prior and scale is 0.
    `theta_widths` are the per-ligand theta widths of the solve; `bend_column_def_sha256` hashes the definitions."""
    from jacobik.ik import angle_bends as AB

    cols, ks, frees = [], [], []
    for li, fk in enumerate(fks):
        nb = getattr(fk, "n_angle_bends", 0)
        if not nb:
            continue
        k = fk._bend_k.detach().double().cpu().numpy()
        if k.shape[0] != nb:
            raise ValueError(f"S2b: ligand {li} has {nb} bend columns but {k.shape[0]} stiffnesses")
        n0 = 6 + fk.n_joints
        if int(theta_widths[li]) < n0 + nb:
            raise ValueError(f"S2b: ligand {li} theta width {theta_widths[li]} < {n0 + nb}")
        fz = getattr(fk, "_bend_free", None)
        if fz is not None:
            fz_np = fz.detach().double().cpu().numpy()
            free = np.maximum(fz_np, float(bend_cfg["free_deg"])) if bend_cfg else fz_np
        else:
            free = np.full(nb, float(bend_cfg["free_deg"]) if bend_cfg else AB.FREE_DEG)
        cols += [(li, n0 + j) for j in range(nb)]
        ks += [float(x) for x in k]
        frees += [float(x) for x in free]
    scale = float(bend_cfg["scale"]) if bend_cfg else 0.0
    per_z_flag = bool(_ik()._BEND_PER_Z) if per_z is None else bool(per_z)
    rt = float(AB.RT_KJ)
    cols_arr = np.asarray(cols, dtype=np.int64).reshape(-1, 2)
    k_arr = np.asarray(ks, dtype=np.float64)
    free_arr = np.asarray(frees, dtype=np.float64)
    definition = {
        "schema": S2B_STATE_SCHEMA,
        "bend_columns": cols_arr.tolist(),
        "bend_k": k_arr.tolist(),
        "bend_free_deg": free_arr.tolist(),
        "bend_scale": scale,
        "bend_per_z": per_z_flag,
        "rt_prior": rt,
        "theta_widths": [int(w) for w in theta_widths],
    }
    sha = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return dict(
        bend_columns=cols_arr,
        bend_k=k_arr,
        bend_free_deg=free_arr,
        bend_scale=scale,
        bend_per_z=per_z_flag,
        rt_prior=rt,
        bend_column_def_sha256=sha,
        theta_widths=[int(w) for w in theta_widths],
    )


def _row_group_weights(groups, rows, device):
    """(B,) weight per group: 1/(number of the row's own groups) where the group applies to the row, else 0."""
    if not groups:
        return []
    on = torch.stack([rows[tuple(g)].to(device) for g in groups]).to(torch.float32)  # (G, B)
    w = on / on.sum(0).clamp_min(1.0)
    return list(w.unbind(0))


def _init_solve_state(
    solver,
    theta_inits: list,
    X_ref: torch.Tensor,
    donor_pos_idxs: list,
    ideal_directions: torch.Tensor,
    symmetric_groups: Optional[List[List[int]]],
    q3_symmetric_groups: Optional[List[List[int]]],
    cfg_mod,
) -> SimpleNamespace:
    """Allocate per-solve tensors, Q3 mode commitments, symmetry groups, step tables, and FinishPhase."""
    ik = _ik()
    finish_phase = getattr(cfg_mod, "FINISH_PHASE", ik.FINISH_PHASE)
    polish_flag = getattr(cfg_mod, "POLISH", ik.POLISH)
    irprop = ik._IRPROP
    eps = ik._EPS
    plateau_w = ik._PLATEAU_WINDOW
    plateau_atol = ik._PLATEAU_ATOL
    mu_shape_w = ik._MU_SHAPE_STALL_WINDOW
    sym_w = ik._SYM_STALL_WINDOW
    lambda_init = ik._LAMBDA_INIT
    sym_weight = ik._SYM_WEIGHT
    sym_angle_weight = ik._SYM_ANGLE_WEIGHT

    device = solver.device
    B = theta_inits[0].shape[0]
    B_full = B
    T = min(solver.n_steps, 250) if finish_phase is not None else solver.n_steps
    L = len(solver.fks)
    if solver.cull_at and solver._niche_plan is None:
        solver._niche_plan = solver._build_niche_plan(donor_pos_idxs)

    X_ref = X_ref.to(device)
    X_ref_full = X_ref  # (N, 3), or (B, N, 3) with per-row references
    # Per-ligand seam targets: each row's own starting seam distances.
    cl_d0 = {li: fk_li.seam_d0(X_ref) for li, fk_li in enumerate(solver.fks) if getattr(fk_li, "_cl", None) is not None}
    donor_pos_idxs = [idx.to(device) for idx in donor_pos_idxs]
    ideal_directions_dev = ideal_directions.to(device)

    thetas = [ti.detach().to(device).requires_grad_(True) for ti in theta_inits]

    _cl_a_list, _cl_b_list = [], []
    for _fk in solver.fks:
        for _a, _b in getattr(_fk, "closure_pairs", []):
            _cl_a_list.append(_a)
            _cl_b_list.append(_b)
    has_closure = len(_cl_a_list) > 0
    _cl_a: object = _cl_a_list
    _cl_b: object = _cl_b_list
    _cl_d0 = None
    if has_closure:
        _cl_a = torch.tensor(_cl_a_list, dtype=torch.long, device=device)
        _cl_b = torch.tensor(_cl_b_list, dtype=torch.long, device=device)
        # Targets: each row's own starting seam distances (closed by construction).
        _cl_d0 = (X_ref[..., _cl_a, :] - X_ref[..., _cl_b, :]).norm(dim=-1)  # (B, R) or (R,)

    handoff_delta = [torch.full_like(ti, 0.01) for ti in thetas] if polish_flag in irprop else None
    adam_m = [torch.zeros_like(ti) for ti in thetas]
    adam_v = [torch.zeros_like(ti) for ti in thetas]
    has_endpoint = ik.JointIKSolver.finish_endpoint_hook is not None
    endpoint_m = [torch.zeros_like(ti) for ti in thetas] if has_endpoint else None
    endpoint_v = [torch.zeros_like(ti) for ti in thetas] if has_endpoint else None
    endpoint_t = torch.zeros(B, dtype=torch.long, device=device) if has_endpoint else None

    best_score_ever = torch.full((B,), float("-inf"), device=device)
    best_score_thetas = [ti.detach().clone() for ti in thetas]

    # Stall detectors: ring buffers of recent values; convergence / freezing
    # is declared when the range over a full window is below its tolerance.
    W = plateau_w
    score_ring = torch.full((W, B), float("-inf"), device=device)
    ever_plateaued = torch.zeros(B, dtype=torch.bool, device=device)

    MW = mu_shape_w
    violation_ring = torch.full((MW, B), float("inf"), device=device)
    mu_shape_frozen = torch.zeros(B, dtype=torch.bool, device=device)

    SW = sym_w
    sym_violation_ring = torch.full((SW, B), float("inf"), device=device)
    sym_angle_violation_ring = torch.full((SW, B), float("inf"), device=device)
    mu_sym_frozen = torch.zeros(B, dtype=torch.bool, device=device)
    mu_sym_angle_frozen = torch.zeros(B, dtype=torch.bool, device=device)

    history_residuals = []
    history_scores = []
    history_grad_norms = []
    history_shape = []

    # Augmented-Lagrangian dual variables (initial values are the starting weights).
    mu_shape = torch.full((B,), lambda_init, device=device)
    mu_sym = torch.full((B,), sym_weight, device=device)
    mu_sym_angle = torch.full((B,), sym_angle_weight, device=device)

    # Site-index -> Q3-array-index map: ge.q3_approach_angles() returns only
    # Q3-eligible sites, in site order.
    _q3_groups = q3_symmetric_groups if q3_symmetric_groups is not None else symmetric_groups

    q3_site_to_idx: Dict[int, int] = {}
    if symmetric_groups:
        _q3_running = 0
        for _s_i, _elig in enumerate(solver.ge._q3_eligible):
            if _elig:
                q3_site_to_idx[_s_i] = _q3_running
                _q3_running += 1

    # Q3 mode commitment: (B, n_q3) int64 (-1 = no override), fixed from the
    # initial pose for the whole solve.
    q3_mode_override: Optional[torch.Tensor] = None
    if _q3_groups and q3_site_to_idx:
        q3_means_static = getattr(solver.scorer, "_q3_means", None)
        q3_eff_vars_static = getattr(solver.scorer, "_q3_eff_vars", None)
        q3_log_w_static = getattr(solver.scorer, "_q3_log_w", None)
        q3_weights_static = getattr(solver.scorer, "_q3_weights", None)
        if (
            q3_means_static is not None
            and q3_means_static.shape[0] > 0
            and q3_eff_vars_static is not None
            and q3_log_w_static is not None
        ):
            with torch.no_grad():
                X_init = _ref_rows(X_ref, B)
                for _fk_li, _theta_li in zip(solver.fks, thetas):
                    X_init = _fk_li.apply(_theta_li, X_init)
                q3_cos_init = torch.cos(solver.ge.q3_approach_angles(X_init))  # (B, n_q3)
                n_q3_total = q3_means_static.shape[0]
                override = torch.full((B, n_q3_total), -1, dtype=torch.long, device=device)
                any_group_qualifies = False
                for group in _q3_groups:
                    q3_idxs = [q3_site_to_idx[g] for g in group if g in q3_site_to_idx]
                    if len(q3_idxs) < 2:
                        continue
                    any_group_qualifies = True
                    idx_t = torch.as_tensor(q3_idxs, dtype=torch.long, device=device)
                    obs_g = q3_cos_init[:, idx_t]  # (B, G)
                    m_g = q3_means_static[idx_t]  # (G, K)
                    v_g = q3_eff_vars_static[idx_t]  # (G, K)
                    lw_g = q3_log_w_static[idx_t]  # (G, K)
                    # Un-normalised log-density: only the argmax over k matters.
                    diff_g = obs_g.unsqueeze(-1) - m_g.unsqueeze(0)  # (B, G, K)
                    log_comp_g = -0.5 * torch.log(v_g).unsqueeze(0) - 0.5 * diff_g**2 / v_g.unsqueeze(0)
                    log_p_g = log_comp_g + lw_g.unsqueeze(0)  # (B, G, K)
                    # Exclude zero-weight dummy slots across all group members before argmax.
                    if q3_weights_static is not None:
                        valid_g = q3_weights_static[idx_t] > eps  # (G, K)
                        log_p_g = log_p_g.masked_fill(~valid_g.unsqueeze(0), float("-inf"))
                    group_log_p_per_k = log_p_g.sum(dim=1)  # (B, K), all members forced to k
                    best_k = group_log_p_per_k.argmax(dim=-1)  # (B,)
                    if solver.sym_rows is None:
                        override[:, idx_t] = best_k.unsqueeze(-1).expand(-1, len(q3_idxs))
                    else:
                        _on = solver.sym_rows["q3"][tuple(group)].to(device)[:, None]
                        override[:, idx_t] = torch.where(
                            _on,
                            best_k.unsqueeze(-1).expand(-1, len(q3_idxs)),
                            override[:, idx_t],
                        )
                if any_group_qualifies:
                    q3_mode_override = override

    # Group index tensors are constant over the solve.
    _sym_group_idx_tensors = [
        torch.as_tensor(group, dtype=torch.long, device=device)
        for group in (symmetric_groups or [])
        if len(group) >= 2
    ]
    _n_sym_groups = len(_sym_group_idx_tensors)
    _sym_w = _q3_sym_w = None  # per-row group weights (sym_rows); None = plain mean over groups
    if solver.sym_rows is not None:
        _sym_w = _row_group_weights(
            [g for g in (symmetric_groups or []) if len(g) >= 2],
            solver.sym_rows["dist"],
            device,
        )

    _q3_sym_group_idx_tensors = []
    _q3_sym_haptic = False
    if q3_site_to_idx:
        for group in _q3_groups:
            q3_idxs = [q3_site_to_idx[g] for g in group if g in q3_site_to_idx]
            if len(q3_idxs) >= 2:
                _q3_sym_group_idx_tensors.append(torch.as_tensor(q3_idxs, dtype=torch.long, device=device))
                if any(solver.ge._donor_is_haptic[g] for g in group if g in q3_site_to_idx):
                    _q3_sym_haptic = True
                if solver.sym_rows is not None:
                    _q3_sym_w = (_q3_sym_w or []) + [tuple(group)]
    _n_q3_sym_groups = len(_q3_sym_group_idx_tensors)
    if _q3_sym_w is not None:
        _q3_sym_w = _row_group_weights(_q3_sym_w, solver.sym_rows["q3"], device)

    # Precompute per-ligand step-scale table (bend_release / dof_step_scale).
    _step_scale_tabs: Optional[List[Optional[torch.Tensor]]] = None
    _ik_ns = getattr(cfg_mod, "IKM", cfg_mod)
    _bend_rel = getattr(_ik_ns, "BEND_RELEASE", getattr(cfg_mod, "BEND_RELEASE", ik.BEND_RELEASE))
    _dof_scale = getattr(_ik_ns, "DOF_STEP_SCALE", getattr(cfg_mod, "DOF_STEP_SCALE", ik.DOF_STEP_SCALE))
    if isinstance(_dof_scale, str):
        _dof_scale = json.loads(_dof_scale)
    if _bend_rel is not None or _dof_scale is not None:
        _s0, _K = (float(_bend_rel[0]), int(_bend_rel[1])) if _bend_rel is not None else (1.0, 1)
        _s_rigid = float(_dof_scale.get("rigid", 1.0)) if _dof_scale else 1.0
        _s_torsion = float(_dof_scale.get("torsion", 1.0)) if _dof_scale else 1.0
        _s_ring = float(_dof_scale.get("ring", 1.0)) if _dof_scale else 1.0
        _s_bend = float(_dof_scale.get("bend", 1.0)) if _dof_scale else 1.0

        _step_scale_tabs = []
        for li in range(L):
            fk_li = solver.fks[li]
            D_li = thetas[li].shape[1]
            tab_li = torch.ones((T, D_li), dtype=thetas[li].dtype, device=device)
            if _s_rigid != 1.0:
                tab_li[:, 0:6] = _s_rigid
            n_joints = fk_li.n_joints
            n_macro = getattr(fk_li, "n_macro_joints", 0)
            n_tors = n_joints - n_macro
            if n_tors > 0 and _s_torsion != 1.0:
                tab_li[:, 6 : 6 + n_tors] = _s_torsion
            if n_macro > 0 and _s_ring != 1.0:
                tab_li[:, 6 + n_tors : 6 + n_joints] = _s_ring
            n_bends = fk_li.n_angle_bends
            if n_bends > 0:
                c_start = 6 + n_joints
                c_end = c_start + n_bends
                if _s_bend != 1.0:
                    tab_li[:, c_start:c_end] = _s_bend
                if _bend_rel is not None:
                    t_idx = torch.arange(T, dtype=thetas[li].dtype, device=device)
                    ramp = _s0 + (1.0 - _s0) * torch.clamp(t_idx / max(_K, 1), max=1.0)
                    tab_li[:, c_start:c_end] = tab_li[:, c_start:c_end] * ramp[:, None]
            _step_scale_tabs.append(tab_li)
        if not any(tab is not None for tab in _step_scale_tabs):
            _step_scale_tabs = None

    phase = None
    if finish_phase is not None:
        from jacobik.ik.finish_phase import FinishPhase

        phase = FinishPhase(
            thetas,
            solver.fks,
            T,
            int(_bend_rel[1]) if _bend_rel else 0,
            finish_phase,
            plateau_w,
            plateau_atol,
            search_lr=solver.lr,
        )

    return SimpleNamespace(
        device=device,
        B=B,
        B_full=B_full,
        T=T,
        L=L,
        X_ref=X_ref,
        X_ref_full=X_ref_full,
        cl_d0=cl_d0,
        donor_pos_idxs=donor_pos_idxs,
        ideal_directions_dev=ideal_directions_dev,
        thetas=thetas,
        has_closure=has_closure,
        _cl_a=_cl_a,
        _cl_b=_cl_b,
        _cl_d0=_cl_d0,
        handoff_delta=handoff_delta,
        adam_m=adam_m,
        adam_v=adam_v,
        endpoint_m=endpoint_m,
        endpoint_v=endpoint_v,
        endpoint_t=endpoint_t,
        best_score_ever=best_score_ever,
        best_score_thetas=best_score_thetas,
        W=W,
        score_ring=score_ring,
        ever_plateaued=ever_plateaued,
        MW=MW,
        violation_ring=violation_ring,
        mu_shape_frozen=mu_shape_frozen,
        SW=SW,
        sym_violation_ring=sym_violation_ring,
        sym_angle_violation_ring=sym_angle_violation_ring,
        mu_sym_frozen=mu_sym_frozen,
        mu_sym_angle_frozen=mu_sym_angle_frozen,
        history_residuals=history_residuals,
        history_scores=history_scores,
        history_grad_norms=history_grad_norms,
        history_shape=history_shape,
        mu_shape=mu_shape,
        mu_sym=mu_sym,
        mu_sym_angle=mu_sym_angle,
        _q3_groups=_q3_groups,
        q3_site_to_idx=q3_site_to_idx,
        q3_mode_override=q3_mode_override,
        _sym_group_idx_tensors=_sym_group_idx_tensors,
        _n_sym_groups=_n_sym_groups,
        _sym_w=_sym_w,
        _q3_sym_group_idx_tensors=_q3_sym_group_idx_tensors,
        _q3_sym_haptic=_q3_sym_haptic,
        _n_q3_sym_groups=_n_q3_sym_groups,
        _q3_sym_w=_q3_sym_w,
        _step_scale_tabs=_step_scale_tabs,
        phase=phase,
    )


def _finalize_solve_state(
    solver,
    best_thetas: list,
    X_ref_full: torch.Tensor,
    B: int,
    row_groups: Optional[torch.Tensor],
    q3_mode_override: Optional[torch.Tensor],
    endpoint_m,
    endpoint_v,
    endpoint_t,
    T: int,
    handoff_delta,
    ever_plateaued: torch.Tensor,
    cfg_mod,
):
    """Apply ring-closure projection, optional polish/IRPROP refinement, and final scoring."""
    ik = _ik()
    polish_flag = getattr(cfg_mod, "POLISH", ik.POLISH)
    optimizer_flag = getattr(cfg_mod, "OPTIMIZER", ik.OPTIMIZER)
    irprop = ik._IRPROP
    closure_w = ik._CLOSURE_W
    rank_floor = getattr(cfg_mod, "_RANK_FLOOR", ik._RANK_FLOOR)
    sc_mode = getattr(cfg_mod, "_SC_MODE", ik._SC_MODE)

    # Exact ring closure for macrocycles: fixed-iteration damped
    # Gauss-Newton on the ring dihedrals (no host sync, no autograd).
    seam_ss = None
    for li, fk_li in enumerate(solver.fks):
        if getattr(fk_li, "_cl", None) is not None:
            best_thetas[li], ss = fk_li.closure_project(best_thetas[li], X_ref_full)
            seam_ss = ss if seam_ss is None else seam_ss + ss

    if ik.JointIKSolver.finish_endpoint_hook is not None:
        ik.JointIKSolver.finish_endpoint_hook(best_thetas, endpoint_m, endpoint_v, endpoint_t)

    if polish_flag is not None or optimizer_flag in irprop:
        from jacobik.ik.polish import optimise

        def polish_xyz(th):
            X = _ref_rows(X_ref_full, B)
            for fk, theta in zip(solver.fks, th):
                X = fk.apply(theta, X)
            return X

        def polish_score(th):
            X = polish_xyz(th)
            rg = None if row_groups is None else row_groups.to(X.device)
            stat, _ = solver.scorer.score(X, q3_mode_override=q3_mode_override, row_groups=rg)
            seam = X.new_zeros(B)
            for fk in solver.fks:
                if getattr(fk, "_cl", None) is not None:
                    residual, _ = fk.closure_residual_jacobian(X, fk.seam_d0(X_ref_full))
                    seam = seam + residual.square().sum(-1)
            return stat + solver._bend_prior(th) - closure_w * seam

        try:
            best_thetas, _ = optimise(
                best_thetas,
                polish_score,
                method=optimizer_flag if optimizer_flag in irprop else polish_flag,
                steps=T if optimizer_flag in irprop else 50,
                delta0=handoff_delta,
            )
        except Exception:
            # The optimizer clones its inputs: the delivered baseline is
            # still valid if an experimental phase cannot finish.
            ik.log.warning("polish failed; keeping delivered baseline", exc_info=True)
        else:
            # The exact delivered objective includes the residual seam penalty;
            # projecting again here would change the selected best candidate.
            seam_ss = None
            X_polished = polish_xyz(best_thetas)
            for fk in solver.fks:
                if getattr(fk, "_cl", None) is not None:
                    residual, _ = fk.closure_residual_jacobian(X_polished, fk.seam_d0(X_ref_full))
                    ss = residual.square().sum(-1)
                    seam_ss = ss if seam_ss is None else seam_ss + ss

    with torch.no_grad():
        X_final = _ref_rows(X_ref_full, B)
        for fk_li, theta_li in zip(solver.fks, best_thetas):
            X_final = fk_li.apply(theta_li, X_final)

        _rg = None if row_groups is None else row_groups.to(X_final.device)
        final_score, _bd = solver.scorer.score(X_final, q3_mode_override=q3_mode_override, row_groups=_rg)  # (B,)
        _hook = ik.JointIKSolver.final_hook
        _rank = rank_floor and hasattr(solver.scorer, "set_gmm_floor")
        if _rank or _hook is not None:
            # the same rows scored with the fit-time floors (true fitted widths)
            solver.scorer.set_gmm_floor("fit")
            fs_fit, bd_fit = solver.scorer.score(X_final, q3_mode_override=q3_mode_override, row_groups=_rg)
            solver.scorer.set_gmm_floor(sc_mode)
            if _hook is not None:
                _hook(
                    dict(
                        score=final_score.detach().cpu().numpy(),
                        score_fit=fs_fit.detach().cpu().numpy(),
                        comps={k: v.detach().cpu().numpy() for k, v in _bd.items()} if isinstance(_bd, dict) else {},
                        comps_fit=(
                            {k: v.detach().cpu().numpy() for k, v in bd_fit.items()} if isinstance(bd_fit, dict) else {}
                        ),
                        bend=solver._bend_prior(best_thetas).detach().cpu().numpy(),
                    )
                )
            if _rank:
                final_score = fs_fit
        _bend_final = solver._bend_prior(best_thetas)
        final_score = final_score + _bend_final
        solver.last_bend_prior = _bend_final.detach().cpu().numpy()  # S2b capture: the value added above
        if seam_ss is not None:
            final_score = final_score - closure_w * seam_ss  # rank unclosed rings last
        # Converged = the score plateaued at some point during optimisation.
        converged = ever_plateaued

    return best_thetas, converged, final_score
