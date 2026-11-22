"""Batched gradient-descent inverse-kinematics solver for the conformer engine.

All ligands' degrees of freedom are optimised jointly against the Scorer's
statistical score plus two augmented-Lagrangian constraint families:

* shape: donor directions are pulled toward the best-fit rotation of the ideal
  coordination polyhedron (rotation recomputed every step, detached);
* symmetry ties: topologically equivalent donors are pulled toward a common
  M-D distance and Q3 approach angle.

Each constraint has its own dual variable that escalates on its violation and
freezes once the violation stalls (safeguarded dual ascent). Rows may be culled
mid-solve (successive halving with a basin-niche penalty), and on CPU the batch
is split into chunks solved concurrently by threads or a process pool.
"""

from __future__ import annotations

import copy as _copy
import logging
import math
import os

#: Rank the final rows with the fit-time GMM floors (true fitted widths) while optimising with the scorer's floors
#: (B2.7, profile field rank_floor_fit). Score-energy baseline
#: 2026-09-27 (12 cases x 3 seeds, MACE on the top of each pool): picked dE median/mean/worst 147/152/393 -> 138/147/360,
#: pool Spearman 0.1 -> 0.3. Optimising with true widths (C) or with tempered Q5 (D) made the pools worse.
# Production resolved configuration (hardcoded values)
_RANK_FLOOR: bool = False
_BEND_PER_Z: bool = True
_SC_MODE: str = "thermal"
BEND_RELEASE = [0.1, 100]

from jacobik.core import profile, timing
from jacobik.core.env import EPS
DOF_STEP_SCALE = profile.get("dof_step_scale")
FINISH_PHASE = profile.get("finish_phase")
POLISH = profile.get("polish")
OPTIMIZER = profile.get("optimizer")
for _name, _value in (("polish", POLISH), ("optimizer", OPTIMIZER)):
    if _value not in (None, "irprop", "irprop_minus"):
        raise ValueError(f"profile flag {_name!r} must be None, 'irprop' or 'irprop_minus', got {_value!r} "
                         "(the fire, adam_small and adam_score values were removed in the 2026-11 cleanup)")
_IRPROP = ("irprop", "irprop_minus")


import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from jacobik.ik.ik_setup import (  # noqa: F401
    S2B_STATE_SCHEMA,
    _get_pool,
    _row_group_weights,
    _shutdown_pool,
    _solve_chunk_worker,
    bend_state_constants,
    merge_chunk_traces,
)

if TYPE_CHECKING:
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer

log = logging.getLogger(__name__)

_EPS = EPS

# Adam
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8

# Convergence: score range over the last _PLATEAU_WINDOW steps below _PLATEAU_ATOL.
_PLATEAU_WINDOW = 8
_PLATEAU_ATOL = 0.02

# Shape constraint (dual variable mu_shape). The constrained statistic is the
_CLOSURE_W = 1000.0  # penalty weight (per Å²) on macrocycle ring-closure distances
# power-mean (temperature _LSE_TEMP) of per-donor squared unit-vector deviations.
_LAMBDA_INIT = 10.0
_LSE_TEMP = 2.0
_SHAPE_RHO = 8.0
_MU_SHAPE_STALL_WINDOW = 20
_MU_SHAPE_STALL_ATOL = 0.01

# Symmetry ties (dual variables mu_sym for M-D distance, mu_sym_angle for Q3 angle).
_SYM_WEIGHT = 2.0
_SYM_ANGLE_WEIGHT = 1.0
_SYM_RHO = 8.0
_SYM_TOL = 0.0
_SYM_STALL_WINDOW = 20
_SYM_STALL_ATOL = 0.01

# Chunked CPU execution: never split below this many rows per chunk.
_MIN_CHUNK = 128

# Row culling: only when the total batch has >= _CULL_MIN_BATCH rows, and never
# below _CULL_MIN_ROWS survivors. _CULL_NICHE scales the per-basin rank penalty
# (0 = plain top-k by score; large = round-robin over basins).
_CULL_MIN_ROWS = 16
_CULL_NICHE = 30.0
_CULL_MIN_BATCH = 256


from jacobik.ik.kernels import wrap_angle  # noqa: F401
from jacobik.ik.kabsch import (  # noqa: F401
    _matrix_to_axisangle_batched,
    _optimal_rotation_nocentroid,
    _ref_rows,
    _renorm_axisangle,
    kabsch_init,
)


# ---------------------------------------------------------------------------
# JointIKSolver


class _EagerSolveMixin:
    """Mixin implementing row-culling plan construction and the eager gradient-descent loop."""

    def _build_niche_plan(self, donor_pos_idxs: list) -> list:
        """
        Backbone torsion joints (bonds separating two donors of the same
        ligand, which set chelate-ring puckers / bridge conformations) and the
        Q5 mode rows they map to. Returns (ligand_index, theta_columns,
        q5_rows, hash_multipliers) tuples; built once per solver.
        """
        plan: list = []
        q5_means = getattr(self.scorer, "_q5_means", None)
        if q5_means is None or q5_means.shape[0] == 0:
            return plan
        ge = self.ge
        site_atoms: List[set] = []
        kp = hp = 0
        for s_i in range(ge.n_donors):
            if not ge._donor_is_haptic[s_i]:
                site_atoms.append({int(ge._kappa_pos_indices[kp])})
                kp += 1
            else:
                site_atoms.append({int(i) for i in ge._haptic_pos_lists[hp]})
                hp += 1
        gen = torch.Generator().manual_seed(12345)
        q5_off = 0
        for li, fk in enumerate(self.fks):
            Jb = getattr(fk, "n_joints_base", fk.n_joints)
            if Jb > 0:
                lig_atoms = set(fk._lig_pos_t.tolist())
                donor_atoms: set = set()
                for g in donor_pos_idxs[li].tolist():
                    donor_atoms |= site_atoms[g]
                cols, rows, offs = [], [], []
                al_rows = getattr(fk, "_q5_rows", None)
                al_offs = getattr(fk, "_q5_offsets", None)
                for k in range(Jb):
                    down = set(fk._downstream[k].tolist())
                    up = lig_atoms - down
                    if (donor_atoms & down) and (donor_atoms & up):
                        if al_rows is not None:  # the joint's own Q5 row, in the joint's dihedral convention
                            if int(al_rows[k]) < 0:
                                continue
                            rows.append(int(al_rows[k]))
                            offs.append(float(al_offs[k]))
                        else:
                            rows.append(q5_off + k)
                            offs.append(0.0)
                        cols.append(6 + k)
                if cols:
                    mult = torch.randint(1, 2**31 - 1, (len(cols),), generator=gen)
                    plan.append(
                        (
                            li,
                            torch.tensor(cols, dtype=torch.long, device=self.device),
                            torch.tensor(rows, dtype=torch.long, device=self.device),
                            mult.to(self.device),
                            torch.tensor(offs, dtype=torch.float32, device=self.device),
                        )
                    )
            q5_off += Jb
        return plan

    def _block_cull_keep(self, frac, score, thetas, groups, blocks, blocks_n0) -> torch.Tensor:
        """Rows kept by a cull at fraction `frac` applied inside each block on its own (config_states): a block of
        fewer than _CULL_MIN_BATCH original rows is never culled, as a solve of that size alone would not be."""
        keeps = []
        for g in torch.unique(blocks).tolist():
            idx = (blocks == g).nonzero().flatten()
            n = int(idx.numel())
            if blocks_n0[g] < _CULL_MIN_BATCH:
                keeps.append(idx)
                continue
            kg = max(int(math.ceil(frac * n)), min(_CULL_MIN_ROWS, n))
            if kg >= n:
                keeps.append(idx)
                continue
            key = self._cull_key(score[idx], [th[idx] for th in thetas], None if groups is None else groups[idx], kg)
            keeps.append(idx[torch.topk(key, kg).indices])
        return torch.cat(keeps).sort().values

    def _cull_key(
        self,
        score: torch.Tensor,  # (Bc,) best-seen score
        thetas: list,
        groups: Optional[torch.Tensor],  # (Bc,) or None
        k: int,
    ) -> torch.Tensor:
        """
        Selection key for the cull: score minus a niche penalty that grows
        with a row's rank inside its (group, backbone-mode-signature) basin.
        The penalty step is a fraction of the score gap between the best and
        k-th row; with a single basin the ordering is unchanged. Static shapes,
        no host synchronisation.
        """
        Bc = score.shape[0]
        dev = score.device
        score = torch.nan_to_num(score, nan=-1e30, posinf=-1e30, neginf=-1e30)
        sig = torch.zeros(Bc, dtype=torch.long, device=dev)
        has_sig = False
        if self._niche_plan:
            q5_m = self.scorer._q5_means
            q5_w = self.scorer._q5_weights
            for li, cols, rows, mult, offs in self._niche_plan:
                tors = thetas[li].detach()[:, cols]  # (Bc, n)
                d = wrap_angle(tors.unsqueeze(-1) - (q5_m[rows] + offs[:, None]).unsqueeze(0)).abs()
                d = d.masked_fill((q5_w[rows] <= _EPS).unsqueeze(0), float("inf"))
                sig = sig + (d.argmin(dim=-1) * mult.unsqueeze(0)).sum(dim=-1)
            has_sig = True
        if groups is not None:
            sig = sig + groups.long() * 1000003
            has_sig = True
        if not has_sig:
            return score
        o1 = torch.argsort(score, descending=True)
        o2 = torch.sort(sig[o1], stable=True).indices
        order = o1[o2]  # by signature, then score descending
        sig_s = sig[order]
        ar = torch.arange(Bc, device=dev)
        start = torch.ones(Bc, dtype=torch.bool, device=dev)
        start[1:] = sig_s[1:] != sig_s[:-1]
        first = torch.cummax(torch.where(start, ar, torch.zeros_like(ar)), dim=0).values
        rank = torch.empty_like(ar)
        rank[order] = ar - first
        top = torch.topk(score, k).values
        gap = (top[0] - top[-1]).clamp(min=1e-6, max=1e4)
        return score - (_CULL_NICHE * gap / k) * rank.to(score.dtype)

    def _run_eager_loop(
        self,
        st,
        row_groups: Optional[torch.Tensor],
        cull_enabled: bool,
        symmetric_groups: Optional[List[List[int]]],
        q3_symmetric_groups: Optional[List[List[int]]],
    ):
        """Run the eager augmented-Lagrangian IK step loop with optional mid-solve row culling."""
        device = st.device
        B, B_full, T, L = st.B, st.B_full, st.T, st.L
        X_ref = st.X_ref
        cl_d0 = st.cl_d0
        donor_pos_idxs = st.donor_pos_idxs
        ideal_directions_dev = st.ideal_directions_dev
        thetas = st.thetas
        has_closure, _cl_a, _cl_b, _cl_d0 = st.has_closure, st._cl_a, st._cl_b, st._cl_d0
        handoff_delta = st.handoff_delta
        adam_m, adam_v = st.adam_m, st.adam_v
        endpoint_m, endpoint_v, endpoint_t = st.endpoint_m, st.endpoint_v, st.endpoint_t
        best_score_ever, best_score_thetas = st.best_score_ever, st.best_score_thetas
        W, score_ring, ever_plateaued = st.W, st.score_ring, st.ever_plateaued
        MW, violation_ring, mu_shape_frozen = st.MW, st.violation_ring, st.mu_shape_frozen
        SW, sym_violation_ring, sym_angle_violation_ring = st.SW, st.sym_violation_ring, st.sym_angle_violation_ring
        mu_sym_frozen, mu_sym_angle_frozen = st.mu_sym_frozen, st.mu_sym_angle_frozen
        history_residuals, history_scores = st.history_residuals, st.history_scores
        history_grad_norms, history_shape = st.history_grad_norms, st.history_shape
        mu_shape, mu_sym, mu_sym_angle = st.mu_shape, st.mu_sym, st.mu_sym_angle
        q3_site_to_idx, q3_mode_override = st.q3_site_to_idx, st.q3_mode_override
        _sym_group_idx_tensors, _n_sym_groups, _sym_w = st._sym_group_idx_tensors, st._n_sym_groups, st._sym_w
        _q3_sym_group_idx_tensors, _q3_sym_haptic = st._q3_sym_group_idx_tensors, st._q3_sym_haptic
        _n_q3_sym_groups, _q3_sym_w = st._n_q3_sym_groups, st._q3_sym_w
        _step_scale_tabs, phase = st._step_scale_tabs, st.phase

        cull_plan: Dict[int, float] = {}
        for _f, _kp in zip(self.cull_at if cull_enabled and not (phase is not None and phase.whole) else (), self.cull_keep):
            _st = int(round(_f * T))
            if 0 < _st < T:
                cull_plan[_st] = float(_kp)
        alive: Optional[torch.Tensor] = None  # None = every original row still alive
        arch_thetas: Optional[list] = None
        arch_plateau: Optional[torch.Tensor] = None
        groups_c = row_groups.to(device) if row_groups is not None else None
        blocks_c = self.cull_blocks.to(device) if self.cull_blocks is not None else None
        blocks_n0 = torch.bincount(blocks_c).tolist() if blocks_c is not None else None
        q3_mode_override_full = q3_mode_override

        def _hist(x: torch.Tensor) -> torch.Tensor:
            """Per-step trace row at full original batch size (NaN for culled rows)."""
            if alive is None:
                return x
            out = x.new_full((B_full,), float("nan"))
            out[alive] = x
            return out

        for t in range(0 if OPTIMIZER in _IRPROP else T):
            X_current = _ref_rows(X_ref, B)  # (B, N, 3)
            for li, (fk_li, theta_li) in enumerate(zip(self.fks, thetas)):
                X_current = fk_li.apply(theta_li, X_current)

            # Computed once and shared with the scorer.
            all_donor_pos_grad = self.ge.donor_positions(X_current)  # (B, n_donors, 3)

            stat_score, _comps = self.scorer.score(
                X_current, q3_mode_override=q3_mode_override, donor_pos=all_donor_pos_grad, row_groups=groups_c,
            )  # (B,)
            _bp = self._bend_prior(thetas)
            stat_score = stat_score + _bp

            # Best-fit rotation detached via envelope theorem to avoid SVD/eigh backward instability.
            all_donor_pos_det = all_donor_pos_grad.detach()
            sym_penalty = X_ref.new_zeros(B)
            sym_angle_penalty = X_ref.new_zeros(B)

            donor_unit_vecs = F.normalize(all_donor_pos_grad, dim=-1)  # (B, n_donors, 3)

            with torch.no_grad():
                R_opt = _optimal_rotation_nocentroid(
                    ideal_directions_dev, donor_unit_vecs.detach()
                )  # (B, 3, 3): R_opt @ ideal_i ~= donor_unit_i

            aligned_ideal = torch.einsum("bij,bnj->bni", R_opt, ideal_directions_dev)  # (B, n_donors, 3)
            per_donor_sq = ((aligned_ideal - donor_unit_vecs) ** 2).sum(-1)  # (B, n_donors)

            n_sh = per_donor_sq.shape[-1]
            if n_sh > 1:
                tau = _LSE_TEMP
                shape_dev = (torch.logsumexp(tau * per_donor_sq, dim=-1) - math.log(n_sh)) / tau  # (B,)
            else:
                shape_dev = per_donor_sq.mean(dim=-1)

            violation = (shape_dev - self.shape_tol).clamp(min=0.0)  # (B,)
            constraint_loss = mu_shape.detach() * violation + 0.5 * _SHAPE_RHO * violation**2

            if symmetric_groups:
                all_donor_norms = all_donor_pos_grad.norm(dim=-1)  # (B, n_donors)
                for _gi, idx_t in enumerate(_sym_group_idx_tensors):
                    d_g = all_donor_norms[:, idx_t]  # (B, len(group))
                    dev_g = d_g - d_g.mean(dim=-1, keepdim=True)
                    if _sym_w is None:
                        sym_penalty = sym_penalty + (dev_g**2).mean(dim=-1)
                    else:
                        sym_penalty = sym_penalty + _sym_w[_gi] * (dev_g**2).mean(dim=-1)
                if _n_sym_groups > 0 and _sym_w is None:
                    sym_penalty = sym_penalty / _n_sym_groups

                # Q3 approach-angle tie, same form as the distance tie.
                if q3_site_to_idx and _q3_sym_group_idx_tensors:
                    q3_angles_now = self.ge.q3_approach_angles(X_current, haptic=_q3_sym_haptic)  # (B, n_q3) rad
                    for _gi, idx_t2 in enumerate(_q3_sym_group_idx_tensors):
                        a_g = q3_angles_now[:, idx_t2]  # (B, len(q3_idxs))
                        dev_a = a_g - a_g.mean(dim=-1, keepdim=True)
                        if _q3_sym_w is None:
                            sym_angle_penalty = sym_angle_penalty + (dev_a**2).mean(dim=-1)
                        else:
                            sym_angle_penalty = sym_angle_penalty + _q3_sym_w[_gi] * (dev_a**2).mean(dim=-1)
                    if _q3_sym_w is None:
                        sym_angle_penalty = sym_angle_penalty / _n_q3_sym_groups

            sym_violation = (sym_penalty - _SYM_TOL).clamp(min=0.0)
            sym_angle_violation = (sym_angle_penalty - _SYM_TOL).clamp(min=0.0)

            if phase is not None:
                phase.observe(
                    torch.tensor([t], device=device),
                    -stat_score.detach()
                    + constraint_loss.detach()
                    + mu_sym.detach() * sym_violation.detach()
                    + 0.5 * _SYM_RHO * sym_violation.detach() ** 2
                    + mu_sym_angle.detach() * sym_angle_violation.detach()
                    + 0.5 * _SYM_RHO * sym_angle_violation.detach() ** 2,
                )
                constraint_loss = phase.aid("shape", constraint_loss)
                sym_violation = phase.aid("symmetry", sym_violation)
                sym_angle_violation = phase.aid("symmetry", sym_angle_violation)

            constraint_loss = (
                constraint_loss
                + mu_sym.detach() * sym_violation
                + 0.5 * _SYM_RHO * sym_violation**2
                + mu_sym_angle.detach() * sym_angle_violation
                + 0.5 * _SYM_RHO * sym_angle_violation**2
            )

            snap_score = stat_score.detach()
            if has_closure:
                clos = (((X_current[:, _cl_a] - X_current[:, _cl_b]).norm(dim=-1) - _cl_d0) ** 2).sum(dim=-1)  # (B,)
                snap_score = snap_score - 1000.0 * clos.detach()

            # Plateau check, only every W steps to avoid (W, B) reductions each step.
            score_ring[t % W] = stat_score.detach()
            if t >= W - 1 and t % W == W - 1:
                score_range = score_ring.max(dim=0).values - score_ring.min(dim=0).values
                ever_plateaued = ever_plateaued | (score_range < _PLATEAU_ATOL)

            # Best-score snapshot.
            with torch.no_grad():
                improved = snap_score > best_score_ever
                best_score_ever = torch.where(improved, snap_score, best_score_ever)
                sel = improved.view(-1, *([1] * (thetas[0].dim() - 1)))
                for li, theta_li in enumerate(thetas):
                    best_score_thetas[li] = torch.where(
                        sel.expand_as(theta_li),
                        theta_li.detach(),
                        best_score_thetas[li],
                    )

            if JointIKSolver.diagnostic_hook is not None:
                _sv_pos = (sym_penalty - _SYM_TOL).clamp(min=0.0)
                _sav_pos = (sym_angle_penalty - _SYM_TOL).clamp(min=0.0)
                JointIKSolver.diagnostic_hook(
                    self,
                    t,
                    dict(
                        thetas=thetas,
                        X=X_current,
                        X_ref=X_ref,
                        alive=alive,
                        score_loss=-stat_score,
                        shape_loss=mu_shape.detach() * violation + 0.5 * _SHAPE_RHO * violation.square(),
                        symmetry_loss=(
                            mu_sym.detach() * _sv_pos
                            + 0.5 * _SYM_RHO * _sv_pos.square()
                            + mu_sym_angle.detach() * _sav_pos
                            + 0.5 * _SYM_RHO * _sav_pos.square()
                        ),
                        mu_shape=mu_shape,
                        mu_sym=mu_sym,
                        mu_sym_angle=mu_sym_angle,
                        ideal=ideal_directions_dev,
                        donor_pos_idxs=donor_pos_idxs,
                        symmetric_groups=symmetric_groups,
                        q3_symmetric_groups=q3_symmetric_groups,
                        q3_mode_override=q3_mode_override,
                        row_groups=groups_c,
                    ),
                )

            _hook = self.trace_hook
            if _hook is not None:  # diagnostics only (see JointIKSolver.trace_hook)
                _hook(
                    t,
                    dict(
                        alive=alive,
                        phase=phase,
                        comps={k: v.detach() for k, v in _comps.items()},
                        bend_prior=_bp.detach(),
                        shape_dev=shape_dev.detach(),
                        constraint=constraint_loss.detach(),
                        improved=improved,
                        mu_shape=mu_shape.detach(),
                        thetas=[th.detach() for th in thetas],
                        X=X_current.detach(),
                    ),
                )
            score_grads = None
            if phase is not None:
                score_grads = torch.autograd.grad(
                    -stat_score.sum(),
                    thetas,
                    retain_graph=True,
                    create_graph=phase.sigma_source == "curvature",
                    allow_unused=True,
                )
                score_grads = [g if g is not None else torch.zeros_like(th) for g, th in zip(score_grads, thetas)]
                if phase.sigma_source == "curvature":
                    phase.curvature(score_grads, thetas, float(self.scorer._agg_norm()))
            combined_loss = (-stat_score + constraint_loss).sum()  # scalar
            grads_all = torch.autograd.grad(combined_loss, thetas, allow_unused=True)

            total_grads: List[torch.Tensor] = []
            for li, (didx, theta_li) in enumerate(zip(donor_pos_idxs, thetas)):
                g = grads_all[li] if grads_all[li] is not None else torch.zeros_like(theta_li)

                n_d_i = didx.shape[0]
                if n_d_i == 2:
                    # Bidentate: project out rotation about donor-donor axis (null mode of donor positions).
                    with torch.no_grad():
                        d_pos = all_donor_pos_det[:, didx, :]  # (B, 2, 3)
                        null_vec = d_pos[:, 0, :] - d_pos[:, 1, :]  # (B, 3)
                        null_norm = null_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        n_hat = null_vec / null_norm
                        I3 = torch.eye(3, dtype=n_hat.dtype, device=n_hat.device).unsqueeze(0).expand(B, -1, -1)
                        P_keep = I3 - torch.bmm(n_hat.unsqueeze(-1), n_hat.unsqueeze(-2))
                    omega_proj = torch.bmm(P_keep, g[:, 3:6].unsqueeze(-1)).squeeze(-1)
                    g = g.clone()
                    g[:, 3:6] = (
                        torch.where(phase.active[:, None], g[:, 3:6], omega_proj)
                        if phase is not None and ("projection" in phase.release or phase.whole)
                        else omega_proj
                    )

                total_grads.append(g)

            r = -stat_score + constraint_loss  # (B,)

            with torch.no_grad():
                grad_norm_b = torch.cat([g.reshape(B, -1) for g in total_grads], dim=-1).norm(dim=-1)  # (B,)

                history_residuals.append(_hist(r.detach()))
                history_scores.append(_hist(stat_score.detach()))
                history_grad_norms.append(_hist(grad_norm_b))
                history_shape.append(_hist(shape_dev.detach()))

                # Dual ascent with stall safeguard: freeze multiplier once violation stalls over a window.
                violation_ring[t % MW] = violation.detach()
                if t >= MW - 1 and t % MW == MW - 1:
                    v_range = violation_ring.max(dim=0).values - violation_ring.min(dim=0).values
                    mu_shape_frozen = mu_shape_frozen | (v_range < _MU_SHAPE_STALL_ATOL)

                mu_increment = _SHAPE_RHO * violation.detach()
                mu_increment = torch.where(mu_shape_frozen, torch.zeros_like(mu_increment), mu_increment)
                mu_shape = (mu_shape + mu_increment).clamp(min=0.0)

                # Same mechanism, independently, for mu_sym and mu_sym_angle.
                sym_violation_ring[t % SW] = sym_violation.detach()
                sym_angle_violation_ring[t % SW] = sym_angle_violation.detach()
                if t >= SW - 1 and t % SW == SW - 1:
                    sym_v_range = sym_violation_ring.max(dim=0).values - sym_violation_ring.min(dim=0).values
                    sym_angle_v_range = (
                        sym_angle_violation_ring.max(dim=0).values - sym_angle_violation_ring.min(dim=0).values
                    )
                    mu_sym_frozen = mu_sym_frozen | (sym_v_range < _SYM_STALL_ATOL)
                    mu_sym_angle_frozen = mu_sym_angle_frozen | (sym_angle_v_range < _SYM_STALL_ATOL)

                mu_sym_increment = _SYM_RHO * sym_violation.detach()
                mu_sym_angle_increment = _SYM_RHO * sym_angle_violation.detach()
                mu_sym_increment = torch.where(mu_sym_frozen, torch.zeros_like(mu_sym_increment), mu_sym_increment)
                mu_sym_angle_increment = torch.where(
                    mu_sym_angle_frozen, torch.zeros_like(mu_sym_angle_increment), mu_sym_angle_increment
                )
                mu_sym = (mu_sym + mu_sym_increment).clamp(min=0.0)
                mu_sym_angle = (mu_sym_angle + mu_sym_angle_increment).clamp(min=0.0)

            adam_t = t + 1
            b1, b2, eps_a = _ADAM_BETA1, _ADAM_BETA2, _ADAM_EPS
            with torch.no_grad():
                if phase is not None:
                    scales = (
                        [tab.index_select(0, torch.tensor([t], device=device)) for tab in _step_scale_tabs]
                        if _step_scale_tabs is not None
                        else None
                    )
                    steps = phase.update(
                        torch.tensor([t], device=device),
                        total_grads,
                        [g * float(self.scorer._agg_norm()) for g in score_grads],
                        (-stat_score + constraint_loss).detach(),
                        scales,
                        self.lr,
                    )
                else:
                    steps = []
                    lr_t = self.lr
                    for li, g in enumerate(total_grads):
                        adam_m[li] = b1 * adam_m[li] + (1.0 - b1) * g
                        adam_v[li] = b2 * adam_v[li] + (1.0 - b2) * g**2
                        if endpoint_m is not None:
                            if alive is None:
                                endpoint_m[li].copy_(adam_m[li])
                                endpoint_v[li].copy_(adam_v[li])
                                endpoint_t.fill_(adam_t)
                            else:
                                endpoint_m[li].index_copy_(0, alive, adam_m[li])
                                endpoint_v[li].index_copy_(0, alive, adam_v[li])
                                endpoint_t.index_fill_(0, alive, adam_t)
                        m_hat = adam_m[li] / (1.0 - b1**adam_t)
                        v_hat = adam_v[li] / (1.0 - b2**adam_t)
                        step = lr_t * m_hat / (v_hat.sqrt() + eps_a)
                        # Inherit Adam's moments, before bend-release / DOF scales.
                        if handoff_delta is not None:
                            if alive is None:
                                handoff_delta[li].copy_(step.abs())
                            else:
                                handoff_delta[li].index_copy_(0, alive, step.abs())
                        if _step_scale_tabs is not None and _step_scale_tabs[li] is not None:
                            step = step * _step_scale_tabs[li][t]
                        steps.append(step)

                for li, (theta_li, step) in enumerate(zip(thetas, steps)):
                    if li in cl_d0:
                        # Ring closure folded into the update (no extra FK pass).
                        c0_ = self.fks[li]._macro_col0
                        c1_ = c0_ + self.fks[li].n_macro_joints
                        step = step.clone()
                        step[:, c0_:c1_] = self.fks[li].closure_fused_step(X_current, cl_d0[li], step[:, c0_:c1_])
                    if phase is not None:
                        step = torch.where(phase.done[:, None], 0.0, step)
                    theta_li.data -= step
                    if theta_li.shape[1] > 6:
                        theta_li.data[:, 6:] = wrap_angle(theta_li.data[:, 6:])
                    theta_li.data[:, 3:6] = _renorm_axisangle(theta_li.data[:, 3:6])

            if phase is not None and (t + 1) % 8 == 0 and bool(phase.stop.item()):
                break

            # Row cull (successive halving with a backbone-basin niche).
            if (t + 1) in cull_plan:
                keep_blocks = None
                if blocks_c is not None:
                    with torch.no_grad():
                        keep_blocks = self._block_cull_keep(
                            cull_plan[t + 1], best_score_ever, thetas, groups_c, blocks_c, blocks_n0
                        )
                    k_keep = int(keep_blocks.numel())
                else:
                    k_keep = max(int(math.ceil(cull_plan[t + 1] * B)), min(_CULL_MIN_ROWS, B))
                if k_keep < B:
                    with torch.no_grad():
                        if alive is None:
                            alive = torch.arange(B_full, device=device)
                            arch_thetas = [bt.detach().clone() for bt in best_score_thetas]
                            arch_plateau = ever_plateaued.clone()
                        for li in range(L):
                            arch_thetas[li].index_copy_(0, alive, best_score_thetas[li])
                        arch_plateau.index_copy_(0, alive, ever_plateaued)
                        if keep_blocks is None:
                            key = self._cull_key(best_score_ever, thetas, groups_c, k_keep)
                            keep = torch.topk(key, k_keep).indices.sort().values
                        else:
                            keep = keep_blocks
                            blocks_c = blocks_c[keep]
                        thetas = [th.detach()[keep].requires_grad_(True) for th in thetas]
                        adam_m = [m[keep] for m in adam_m]
                        adam_v = [v[keep] for v in adam_v]
                        if phase is not None:
                            phase.select(keep)
                        best_score_thetas = [bt[keep] for bt in best_score_thetas]
                        best_score_ever = best_score_ever[keep]
                        score_ring = score_ring[:, keep]
                        ever_plateaued = ever_plateaued[keep]
                        violation_ring = violation_ring[:, keep]
                        mu_shape_frozen = mu_shape_frozen[keep]
                        sym_violation_ring = sym_violation_ring[:, keep]
                        sym_angle_violation_ring = sym_angle_violation_ring[:, keep]
                        mu_sym_frozen = mu_sym_frozen[keep]
                        mu_sym_angle_frozen = mu_sym_angle_frozen[keep]
                        mu_shape = mu_shape[keep]
                        mu_sym = mu_sym[keep]
                        mu_sym_angle = mu_sym_angle[keep]
                        ideal_directions_dev = ideal_directions_dev[keep]
                        if X_ref.dim() == 3:
                            X_ref = X_ref[keep]
                        if has_closure and _cl_d0.dim() == 2:
                            _cl_d0 = _cl_d0[keep]
                        cl_d0 = {k_: (v_[keep] if v_.shape[0] > 1 else v_) for k_, v_ in cl_d0.items()}
                        if q3_mode_override is not None:
                            q3_mode_override = q3_mode_override[keep]
                        if groups_c is not None:
                            groups_c = groups_c[keep]
                        if _sym_w is not None:
                            _sym_w = [w[keep] for w in _sym_w]
                        if _q3_sym_w is not None:
                            _q3_sym_w = [w[keep] for w in _q3_sym_w]
                        alive = alive[keep]
                        B = k_keep

        # Assemble best-score positions for all original rows.
        if alive is not None:
            with torch.no_grad():
                for li in range(L):
                    arch_thetas[li].index_copy_(0, alive, best_score_thetas[li])
                arch_plateau.index_copy_(0, alive, ever_plateaued)
            best_score_thetas = arch_thetas
            ever_plateaued = arch_plateau
            q3_mode_override = q3_mode_override_full
            B = B_full

        return (
            best_score_thetas,
            ever_plateaued,
            q3_mode_override,
            B,
            history_residuals,
            history_scores,
            history_grad_norms,
            history_shape,
        )


@dataclass
class IKSolverConfig:
    """
    Public hyperparameters for JointIKSolver.

    n_steps, lr : Adam step count and learning rate for the joint DOF optimisation.
    shape_tol : violation tolerance of the shape constraint. None means the
        engine resolves a per-geometry tolerance; a standalone solver then
        uses ik/targets.SHAPE_TOL_POOLED.
    n_workers : number of batch chunks solved concurrently on CPU, each with
        single-threaded torch ops. None = ``JACOBIK_NUM_WORKERS`` if set, else
        min(os.cpu_count(), 8). 1 disables chunking; ignored on GPU. Rows are
        independent analytically, but batch shapes can change floating-point
        rounding and iterative trajectories; reproducibility checks pin this.
    worker_mode : "thread" or "process" (persistent spawn-based pool; no GIL,
        costs a one-off start-up; on Windows scripts need an
        ``if __name__ == "__main__":`` guard). Overridable with the
        ``JACOBIK_WORKER_MODE`` environment variable.
    cull_at, cull_keep : row culling by successive halving. At each step
        fraction in ``cull_at`` the batch shrinks to the ``cull_keep`` fraction
        of rows with the best best-seen score (with a basin-niche penalty).
        Empty tuple = off.
    """

    n_steps: int = 200
    lr: float = 0.01
    shape_tol: Optional[float] = None
    n_workers: Optional[int] = None
    worker_mode: str = "thread"
    cull_at: Tuple[float, ...] = (0.3,)
    cull_keep: Tuple[float, ...] = (0.35,)




class JointIKSolver(_EagerSolveMixin):
    """
    Joint gradient-descent IK solver optimising all ligands simultaneously.

    Objective: minimise -score(X) + shape/symmetry constraint penalties, where
    score() is the Scorer's unified statistical score. Both constraint families
    use safeguarded augmented-Lagrangian dual ascent (see module docstring).
    Quantity weights are set on the Scorer, not here.
    """

    #: Optional per-step diagnostics callback ``hook(t, info)`` (None in production). ``info`` holds detached tensors
    #: of the current chunk: score components, bend prior, shape deviation, constraint loss, snapshot-improved mask,
    #: mu_shape, thetas and alive (original row index of each row after culling, or None). Rows are chunk-local when
    #: the batch is split across workers; set JACOBIK_NUM_WORKERS=1 to trace whole batches.
    trace_hook = None
    # CPU diagnostic observer with live score/aid graphs, before backward.
    # None in production; observers must retain_graph and never mutate inputs.
    diagnostic_hook = None
    #: diagnostics: called once per solve batch with the final per-row scores under the scorer's floors and under the
    #: fit-time floors, plus the per-term breakdowns (score-energy baseline, 2026-09-27)
    final_hook = None
    #: S2b capture: the angle-bend prior added to each row's final score (numpy (B,)), set by a single-chunk solve;
    #: None when the solve was split across CPU workers (the copies' values are not returned) or was not run.
    last_bend_prior = None

    finish_endpoint_hook = None  # optional capture of final full-row Adam moments
    #: P2R config_states: (B,) int block id per row (one block per configuration class, solved in one batch). The cull
    #: then runs inside each block exactly as a separate solve of that block would (its own size, fraction and minimum).
    #: None (default) = one block, the production cull.
    cull_blocks = None
    #: P2R config_states: {"dist": {group tuple: (B,) bool}, "q3": {group tuple: (B,) bool}} - which rows each symmetry
    #: tie applies to, when the batch holds classes with different donor symmetry (symmetric_groups is then the union
    #: of the classes' groups). Each row's ties are averaged over its own groups, as its class's separate solve does.
    #: None (default) = every group on every row (production).
    sym_rows = None

    def __init__(
        self,
        fks: list,
        scorer: "Scorer",
        ge: "GeometryExtractor",
        config: Optional[IKSolverConfig] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.fks = fks
        self.scorer = scorer
        self.ge = ge
        self.device = torch.device(device)

        cfg = config or IKSolverConfig()
        self.n_steps = cfg.n_steps
        self.lr = cfg.lr
        if cfg.shape_tol is None:
            from jacobik.ik.targets import SHAPE_TOL_POOLED

            self.shape_tol = SHAPE_TOL_POOLED
        else:
            self.shape_tol = cfg.shape_tol
        self.n_workers = cfg.n_workers
        self.worker_mode = os.environ.get("JACOBIK_WORKER_MODE", cfg.worker_mode).lower()
        self.cull_at = tuple(cfg.cull_at)
        self.cull_keep = tuple(cfg.cull_keep)
        self._niche_plan = None

    def _resolve_workers(self, B: int, debug: bool) -> int:
        if debug or self.device.type != "cpu":
            return 1
        n = self.n_workers
        if n is None:
            env = os.environ.get("JACOBIK_NUM_WORKERS")
            n = int(env) if env else min(os.cpu_count() or 1, 8)
        return max(1, min(int(n), B // max(1, _MIN_CHUNK)))

    # ------------------------------------------------------------------
    # Row culling helpers
    # ------------------------------------------------------------------

    def _bend_prior(self, thetas) -> torch.Tensor:
        """Energy-calibrated angle-bend log-prior (ik/angle_bends.energy_prior), summed over ligands; 0 without bends."""
        total = None
        cfg = getattr(self, "bend_prior", None)
        if not cfg:
            return thetas[0].new_zeros(thetas[0].shape[0])
        from jacobik.ik import angle_bends as AB

        for fk, th in zip(self.fks, thetas):
            nb = getattr(fk, "n_angle_bends", 0)
            if not nb:
                continue
            n0 = 6 + fk.n_joints
            fz = getattr(fk, "_bend_free", None)
            fz = torch.clamp(fz, min=float(cfg["free_deg"])) if fz is not None else cfg["free_deg"]
            p = AB.energy_prior(th[:, n0:n0 + nb], fk._bend_k, fz, cfg["scale"])
            total = p if total is None else total + p
        if total is None:
            return thetas[0].new_zeros(thetas[0].shape[0])
        if _BEND_PER_Z and hasattr(self.scorer, "_agg_norm"):
            total = total / float(self.scorer._agg_norm())
        return total

    @timing.timed("ik_solve")
    def solve(
        self,
        theta_inits: list,  # [Tensor(B, 6+Ji), ...]
        X_ref: torch.Tensor,  # (N, 3)
        donor_pos_idxs: list,  # [LongTensor(n_d_i), ...]
        ideal_directions: torch.Tensor,  # (B, n_donors_total, 3)
        debug: bool = False,
        symmetric_groups: Optional[List[List[int]]] = None,
        q3_symmetric_groups: Optional[List[List[int]]] = None,
        row_groups: Optional[torch.Tensor] = None,  # (B,) int, e.g. candidate id
    ):
        """
        Joint IK solve for all ligands; arguments and return values as in
        ``_solve_impl``. On CPU the batch is split into ``n_workers``
        independent chunks solved concurrently. Pin the worker count for exact
        reproducibility: batch shapes can change floating-point trajectories.
        """
        B = theta_inits[0].shape[0]
        self.last_bend_prior = None
        if self.cull_at and self._niche_plan is None:
            self._niche_plan = self._build_niche_plan(donor_pos_idxs)
        n_w = self._resolve_workers(B, debug)
        if self.cull_blocks is not None or self.sym_rows is not None:
            n_w = 1   # blocks span the batch; the CPU chunking would split them
        kw = dict(
            cull_enabled=B >= _CULL_MIN_BATCH,
            debug=debug,
            symmetric_groups=symmetric_groups,
            q3_symmetric_groups=q3_symmetric_groups,
        )
        if n_w <= 1:
            return self._solve_impl(
                theta_inits, X_ref, donor_pos_idxs, ideal_directions, row_groups=row_groups, **kw
            )

        bounds = [round(i * B / n_w) for i in range(n_w + 1)]
        slices = [slice(bounds[i], bounds[i + 1]) for i in range(n_w)]

        def work(sl: slice):
            # Per-thread setting: keep each chunk's torch ops single-threaded.
            torch.set_num_threads(1)
            # Each chunk gets its own shallow solver + scorer copy. The final rescoring swaps the scorer's GMM floors
            # (set_gmm_floor "fit" -> back); on a shared scorer the chunk that finishes first switched the floors under
            # the other chunk's optimisation steps, so 2-worker solves were not reproducible (2026-09-30).
            # Tensors are shared read-only; set_gmm_floor rebinds attributes, it never writes into them.
            me = _copy.copy(self)
            me.scorer = _copy.copy(self.scorer)
            return me._solve_impl(
                [t[sl] for t in theta_inits],
                X_ref[sl] if X_ref.dim() == 3 else X_ref,
                donor_pos_idxs,
                ideal_directions[sl],
                row_groups=None if row_groups is None else row_groups[sl],
                **kw,
            )

        results = None
        if self.worker_mode == "process":
            try:
                pool = _get_pool(n_w)
                payloads = [
                    pickle.dumps(
                        (
                            self,
                            (
                                [t[sl].clone() for t in theta_inits],
                                X_ref[sl].clone() if X_ref.dim() == 3 else X_ref,
                                donor_pos_idxs,
                                ideal_directions[sl].clone(),
                            ),
                            dict(
                                kw,
                                row_groups=None if row_groups is None else row_groups[sl].clone(),
                            ),
                        )
                    )
                    for sl in slices
                ]
                results = [pickle.loads(b) for b in pool.map(_solve_chunk_worker, payloads)]
            except Exception as exc:  # broken pool, unpicklable state, ...
                log.warning("process-pool solve failed (%r); falling back to threads", exc)
                _shutdown_pool()
                results = None
        if results is None:
            prev_threads = torch.get_num_threads()
            try:
                with ThreadPoolExecutor(max_workers=n_w) as pool_t:
                    results = list(pool_t.map(work, slices))
            finally:
                torch.set_num_threads(prev_threads)

        n_lig = len(theta_inits)
        thetas_final = [torch.cat([r[0][li] for r in results], dim=0) for li in range(n_lig)]
        converged = torch.cat([r[1] for r in results], dim=0)
        final_score = torch.cat([r[2] for r in results], dim=0)
        solve_trace: dict = {}
        if all(r[3] for r in results):
            solve_trace = merge_chunk_traces([r[3] for r in results])
        return thetas_final, converged, final_score, solve_trace

    def _solve_impl(
        self,
        theta_inits: list,  # [Tensor(B, 6+Ji), ...]
        X_ref: torch.Tensor,  # (N, 3)
        donor_pos_idxs: list,  # [LongTensor(n_d_i), ...]
        ideal_directions: torch.Tensor,  # (B, n_donors_total, 3)
        debug: bool = False,
        symmetric_groups: Optional[List[List[int]]] = None,
        q3_symmetric_groups: Optional[List[List[int]]] = None,
        row_groups: Optional[torch.Tensor] = None,
        cull_enabled: bool = True,
    ):
        """
        Run the joint IK solve for all ligands simultaneously.

        Parameters
        ----------
        theta_inits    : list of (B, 6+Ji) initial DOF tensors, one per ligand.
        X_ref          : (N, 3) full-complex reference atom positions.
        donor_pos_idxs : list of (n_d_i,) LongTensors of pos-indices into X_ref.
        ideal_directions : (B, n_donors_total, 3) unit vectors, one per donor in
                         ge.donor_positions()'s global order, given per batch
                         item (each item may target a different polyhedron
                         assignment).
        row_groups     : optional (B,) int tensor; rows with different values
                         are never treated as one basin by the cull's niche
                         penalty. None = one group.
        debug          : forces a single chunk (no threading).
        symmetric_groups : donor-index groups (global donor order) of
                         topologically equivalent donors, tied to a common
                         M-D distance.
        q3_symmetric_groups : groups used instead of symmetric_groups for the
                         Q3 approach-angle tie and the Q3 mode commitment
                         (should only span donors of one ligand instance).
                         None falls back to symmetric_groups.

        A symmetric group whose Q3 target is multi-modal is committed, from the
        initial pose, to a single GMM component for the whole solve.

        Returns
        -------
        thetas_final   : list of (B, 6+Ji) DOFs at the best score seen
        converged      : (B,) bool, True iff the score plateaued
        final_score    : (B,) score at the best-seen positions
        solve_trace    : dict of per-step (T, B) numpy arrays: residual, score,
                         grad_norm (NaN for culled rows after the cull)
        """
        device = self.device
        import sys as _sys
        from jacobik.ik.ik_setup import _finalize_solve_state, _init_solve_state

        _cfg_mod = _sys.modules[__name__]
        _st = _init_solve_state(
            self, theta_inits, X_ref, donor_pos_idxs, ideal_directions, symmetric_groups, q3_symmetric_groups, _cfg_mod
        )
        B, B_full, T = _st.B, _st.B_full, _st.T
        X_ref, X_ref_full, cl_d0 = _st.X_ref, _st.X_ref_full, _st.cl_d0
        donor_pos_idxs, ideal_directions_dev, thetas = _st.donor_pos_idxs, _st.ideal_directions_dev, _st.thetas
        has_closure, _cl_a, _cl_b, _cl_d0 = _st.has_closure, _st._cl_a, _st._cl_b, _st._cl_d0
        handoff_delta, adam_m, adam_v = _st.handoff_delta, _st.adam_m, _st.adam_v
        endpoint_m, endpoint_v, endpoint_t = _st.endpoint_m, _st.endpoint_v, _st.endpoint_t
        best_score_ever, best_score_thetas = _st.best_score_ever, _st.best_score_thetas
        W, score_ring, ever_plateaued = _st.W, _st.score_ring, _st.ever_plateaued
        MW, violation_ring, mu_shape_frozen = _st.MW, _st.violation_ring, _st.mu_shape_frozen
        SW, sym_violation_ring, sym_angle_violation_ring = _st.SW, _st.sym_violation_ring, _st.sym_angle_violation_ring
        mu_sym_frozen, mu_sym_angle_frozen = _st.mu_sym_frozen, _st.mu_sym_angle_frozen
        mu_shape, mu_sym, mu_sym_angle = _st.mu_shape, _st.mu_sym, _st.mu_sym_angle
        q3_site_to_idx, q3_mode_override = _st.q3_site_to_idx, _st.q3_mode_override
        _sym_group_idx_tensors, _n_sym_groups, _sym_w = _st._sym_group_idx_tensors, _st._n_sym_groups, _st._sym_w
        _q3_sym_group_idx_tensors, _q3_sym_haptic = _st._q3_sym_group_idx_tensors, _st._q3_sym_haptic
        _n_q3_sym_groups, _q3_sym_w = _st._n_q3_sym_groups, _st._q3_sym_w
        _step_scale_tabs, phase = _st._step_scale_tabs, _st.phase

        # Row culling state: step -> fraction
        (
            best_score_thetas,
            ever_plateaued,
            q3_mode_override,
            B,
            history_residuals,
            history_scores,
            history_grad_norms,
            history_shape,
        ) = self._run_eager_loop(
            _st, row_groups, cull_enabled, symmetric_groups, q3_symmetric_groups
        )

        best_thetas = best_score_thetas  # list of (B, 6+Ji)
        best_thetas, converged, final_score = _finalize_solve_state(
            self,
            best_thetas,
            X_ref_full,
            B,
            row_groups,
            q3_mode_override,
            endpoint_m,
            endpoint_v,
            endpoint_t,
            T,
            handoff_delta,
            ever_plateaued,
            _cfg_mod,
        )

        solve_trace = {}
        if history_residuals:
            solve_trace = {
                "residual": torch.stack(history_residuals, dim=0).cpu().numpy(),  # (T, B)
                "score": torch.stack(history_scores, dim=0).cpu().numpy(),  # (T, B)
                "grad_norm": torch.stack(history_grad_norms, dim=0).cpu().numpy(),  # (T, B)
                "shape_dev": torch.stack(history_shape, dim=0).cpu().numpy(),  # (T, B)
            }

        thetas_final = [th.detach() for th in best_thetas]
        return thetas_final, converged, final_score, solve_trace
