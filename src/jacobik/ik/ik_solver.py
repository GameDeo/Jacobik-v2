"""GPU-only zero-autograd Triton CUDA-graph inverse-kinematics solver.

All ligands' degrees of freedom are optimised jointly against the Scorer's
statistical score plus two augmented-Lagrangian constraint families:

* shape: donor directions are pulled toward the best-fit rotation of the ideal
  coordination polyhedron (rotation recomputed every step, detached);
* symmetry ties: topologically equivalent donors are pulled toward a common
  M-D distance and Q3 approach angle.

Each constraint has its own dual variable that escalates on its violation and
freezes once the violation stalls (safeguarded dual ascent). Execution runs
exclusively on CUDA via fused Triton kernels, ComplexFK, and CuPy CUDA graphs.
"""

from __future__ import annotations

import logging
import math
import os
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import triton as _triton
import triton.language as _tl

from jacobik.core import gpu, profile, timing
from jacobik.core.env import EPS

cp = gpu.cp
_HAS_TRITON = True

# Production resolved configuration (hardcoded values)
_RANK_FLOOR: bool = False
_BEND_PER_Z: bool = True
_SC_MODE: str = "thermal"
BEND_RELEASE = [0.1, 100]

DOF_STEP_SCALE = profile.get("dof_step_scale")
FINISH_PHASE = profile.get("finish_phase")
POLISH = profile.get("polish")
OPTIMIZER = profile.get("optimizer")
for _name, _value in (("polish", POLISH), ("optimizer", OPTIMIZER)):
    if _value not in (None, "irprop", "irprop_minus"):
        raise ValueError(
            f"profile flag {_name!r} must be None, 'irprop' or 'irprop_minus', got {_value!r} "
            "(the fire, adam_small and adam_score values were removed in the 2026-11 cleanup)"
        )
_IRPROP = ("irprop", "irprop_minus")

from jacobik.ik.ik_setup import (  # noqa: F401
    S2B_STATE_SCHEMA,
    _row_group_weights,
    bend_state_constants,
    merge_chunk_traces,
)
from jacobik.ik.kabsch import (  # noqa: F401
    _matrix_to_axisangle_batched,
    _optimal_rotation_nocentroid,
    _ref_rows,
    _renorm_axisangle,
    kabsch_init,
)
from jacobik.ik.kernels import wrap_angle  # noqa: F401

if TYPE_CHECKING:
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer

log = logging.getLogger(__name__)
LAST: Dict[str, object] = {}

_EPS = EPS

# Adam
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8

# Convergence: score range over the last _PLATEAU_WINDOW steps below _PLATEAU_ATOL.
_PLATEAU_WINDOW = 8
_PLATEAU_ATOL = 0.02

# Shape constraint (dual variable mu_shape).
_CLOSURE_W = 1000.0  # penalty weight (per Å²) on macrocycle ring-closure distances
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

if _HAS_TRITON:

    @_triton.jit
    def _ring_dual_step(t, val, ring_ptr, frozen_ptr, mu_ptr, row, mask, B, W: _tl.constexpr, rho, stall_atol):
        slot = t % W
        _tl.store(ring_ptr + slot * B + row, val, mask=mask)
        v_min = val
        v_max = val
        for w in range(W):
            v = _tl.load(ring_ptr + w * B + row, mask=mask, other=0.0)
            v_min = _tl.minimum(v_min, v)
            v_max = _tl.maximum(v_max, v)
        frz = _tl.load(frozen_ptr + row, mask=mask, other=False)
        frz_new = frz | (((v_max - v_min) < stall_atol) & (slot == (W - 1)))
        _tl.store(frozen_ptr + row, frz_new, mask=mask)
        mu = _tl.load(mu_ptr + row, mask=mask, other=0.0)
        inc = _tl.where(frz_new, 0.0, rho * val)
        _tl.store(mu_ptr + row, _tl.maximum(mu + inc, 0.0), mask=mask)

    @_triton.jit
    def _al_ring_dual_update_kernel(
        tl_ptr, stat_score_ptr, snap_score_ptr, r_ptr, grad_norm_ptr, shape_dev_ptr,
        violation_ptr, sym_violation_ptr, sym_angle_violation_ptr,
        score_ring_ptr, ever_plateaued_ptr, best_score_ever_ptr, improved_ptr,
        hist_res_ptr, hist_score_ptr, hist_gn_ptr, hist_shape_ptr,
        violation_ring_ptr, mu_shape_frozen_ptr, mu_shape_ptr,
        sym_violation_ring_ptr, sym_angle_violation_ring_ptr,
        mu_sym_frozen_ptr, mu_sym_angle_frozen_ptr, mu_sym_ptr, mu_sym_angle_ptr,
        B, W: _tl.constexpr, MW: _tl.constexpr, SW: _tl.constexpr,
        plateau_atol, shape_rho, mu_shape_stall_atol, sym_rho, sym_stall_atol, BLOCK_SIZE: _tl.constexpr,
    ):
        pid = _tl.program_id(0)
        row = pid * BLOCK_SIZE + _tl.arange(0, BLOCK_SIZE)
        mask = row < B

        t = _tl.load(tl_ptr)
        stat_s = _tl.load(stat_score_ptr + row, mask=mask, other=0.0)
        snap_s = _tl.load(snap_score_ptr + row, mask=mask, other=0.0)
        r_val = _tl.load(r_ptr + row, mask=mask, other=0.0)
        gn_val = _tl.load(grad_norm_ptr + row, mask=mask, other=0.0)
        sh_val = _tl.load(shape_dev_ptr + row, mask=mask, other=0.0)
        viol = _tl.load(violation_ptr + row, mask=mask, other=0.0)
        sviol = _tl.load(sym_violation_ptr + row, mask=mask, other=0.0)
        saviol = _tl.load(sym_angle_violation_ptr + row, mask=mask, other=0.0)

        slot = t % W
        _tl.store(score_ring_ptr + slot * B + row, stat_s, mask=mask)
        s_min = stat_s
        s_max = stat_s
        for w in range(W):
            v = _tl.load(score_ring_ptr + w * B + row, mask=mask, other=0.0)
            s_min = _tl.minimum(s_min, v)
            s_max = _tl.maximum(s_max, v)
        ep = _tl.load(ever_plateaued_ptr + row, mask=mask, other=False)
        ep_new = ep | (((s_max - s_min) < plateau_atol) & (slot == (W - 1)))
        _tl.store(ever_plateaued_ptr + row, ep_new, mask=mask)

        bse = _tl.load(best_score_ever_ptr + row, mask=mask, other=float("-inf"))
        imp = snap_s > bse
        bse_new = _tl.where(imp, snap_s, bse)
        _tl.store(best_score_ever_ptr + row, bse_new, mask=mask)
        _tl.store(improved_ptr + row, imp, mask=mask)

        t_off = t * B + row
        _tl.store(hist_res_ptr + t_off, r_val, mask=mask)
        _tl.store(hist_score_ptr + t_off, stat_s, mask=mask)
        _tl.store(hist_gn_ptr + t_off, gn_val, mask=mask)
        _tl.store(hist_shape_ptr + t_off, sh_val, mask=mask)

        _ring_dual_step(t, viol, violation_ring_ptr, mu_shape_frozen_ptr, mu_shape_ptr, row, mask, B, MW, shape_rho, mu_shape_stall_atol)
        _ring_dual_step(t, sviol, sym_violation_ring_ptr, mu_sym_frozen_ptr, mu_sym_ptr, row, mask, B, SW, sym_rho, sym_stall_atol)
        _ring_dual_step(t, saviol, sym_angle_violation_ring_ptr, mu_sym_angle_frozen_ptr, mu_sym_angle_ptr, row, mask, B, SW, sym_rho, sym_stall_atol)

    @_triton.jit
    def _adam_lig_kernel(
        g_ptr,
        m_ptr,
        v_ptr,
        step_ptr,
        handoff_ptr,
        scale_tab_ptr,
        tl_ptr,
        bc1_ptr,
        bc2_ptr,
        B,
        D,
        lr_t,
        b1,
        b2,
        eps_a,
        HAS_HANDOFF: _tl.constexpr,
        HAS_SCALE: _tl.constexpr,
        BLOCK_SIZE: _tl.constexpr,
    ):
        pid = _tl.program_id(0)
        offs = pid * BLOCK_SIZE + _tl.arange(0, BLOCK_SIZE)
        total = B * D
        mask = offs < total
        col = offs % D

        t = _tl.load(tl_ptr)
        c1 = _tl.load(bc1_ptr + t)
        c2 = _tl.load(bc2_ptr + t)

        g = _tl.load(g_ptr + offs, mask=mask, other=0.0)
        m = _tl.load(m_ptr + offs, mask=mask, other=0.0)
        v = _tl.load(v_ptr + offs, mask=mask, other=0.0)

        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * (g * g)
        _tl.store(m_ptr + offs, m_new, mask=mask)
        _tl.store(v_ptr + offs, v_new, mask=mask)

        m_hat = m_new / c1
        v_hat = v_new / c2
        step = lr_t * m_hat / (_tl.sqrt(v_hat) + eps_a)
        if HAS_HANDOFF:
            _tl.store(handoff_ptr + offs, _tl.abs(step), mask=mask)
        if HAS_SCALE:
            sc = _tl.load(scale_tab_ptr + t * D + col, mask=mask, other=1.0)
            step = step * sc
        _tl.store(step_ptr + offs, step, mask=mask)

    @_triton.jit
    def _best_snapshot_copy_kernel(imp_ptr, theta_ptr, best_theta_ptr, B, D, BLOCK_SIZE: _tl.constexpr):
        pid = _tl.program_id(0)
        offs = pid * BLOCK_SIZE + _tl.arange(0, BLOCK_SIZE)
        total = B * D
        mask = offs < total
        row = offs // D
        imp = _tl.load(imp_ptr + row, mask=mask, other=False)
        m_imp = mask & imp
        th = _tl.load(theta_ptr + offs, mask=m_imp, other=0.0)
        _tl.store(best_theta_ptr + offs, th, mask=m_imp)

    @_triton.jit
    def _theta_post_update_kernel(
        theta_ptr,
        step_ptr,
        done_ptr,
        B,
        D,
        HAS_DONE: _tl.constexpr,
        BLOCK_B: _tl.constexpr,
    ):
        pid = _tl.program_id(0)
        b = pid * BLOCK_B + _tl.arange(0, BLOCK_B)
        mask_b = b < B
        done = _tl.load(done_ptr + b, mask=mask_b, other=False) if HAS_DONE else False

        base = b * D
        for c in range(3):
            th = _tl.load(theta_ptr + base + c, mask=mask_b, other=0.0)
            st = _tl.load(step_ptr + base + c, mask=mask_b, other=0.0)
            if HAS_DONE:
                st = _tl.where(done, 0.0, st)
            _tl.store(theta_ptr + base + c, th - st, mask=mask_b)

        s3 = _tl.load(step_ptr + base + 3, mask=mask_b, other=0.0)
        s4 = _tl.load(step_ptr + base + 4, mask=mask_b, other=0.0)
        s5 = _tl.load(step_ptr + base + 5, mask=mask_b, other=0.0)
        if HAS_DONE:
            s3 = _tl.where(done, 0.0, s3)
            s4 = _tl.where(done, 0.0, s4)
            s5 = _tl.where(done, 0.0, s5)
        w0 = _tl.load(theta_ptr + base + 3, mask=mask_b, other=0.0) - s3
        w1 = _tl.load(theta_ptr + base + 4, mask=mask_b, other=0.0) - s4
        w2 = _tl.load(theta_ptr + base + 5, mask=mask_b, other=0.0) - s5
        norm = _tl.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
        pi = 3.141592653589793
        eps = 1e-8
        scale = _tl.where(norm > pi, pi / _tl.maximum(norm, eps), 1.0)
        _tl.store(theta_ptr + base + 3, w0 * scale, mask=mask_b)
        _tl.store(theta_ptr + base + 4, w1 * scale, mask=mask_b)
        _tl.store(theta_ptr + base + 5, w2 * scale, mask=mask_b)

        two_pi = 6.283185307179586
        for c in range(6, D):
            th = _tl.load(theta_ptr + base + c, mask=mask_b, other=0.0)
            st = _tl.load(step_ptr + base + c, mask=mask_b, other=0.0)
            if HAS_DONE:
                st = _tl.where(done, 0.0, st)
            x = th - st + pi
            q = _tl.floor(x / two_pi)
            w = (x - q * two_pi) - pi
            _tl.store(theta_ptr + base + c, w, mask=mask_b)

    @_triton.jit
    def _bidentate_proj_kernel(
        d_pos0_ptr,
        d_pos1_ptr,
        g_ptr,
        active_ptr,
        B,
        D,
        RELEASE_PROJ: _tl.constexpr,
        BLOCK_B: _tl.constexpr,
    ):
        pid = _tl.program_id(0)
        b = pid * BLOCK_B + _tl.arange(0, BLOCK_B)
        mask = b < B
        # d_pos0, d_pos1 are (B, 3) contiguous
        dx = _tl.load(d_pos0_ptr + b * 3 + 0, mask=mask, other=0.0) - _tl.load(d_pos1_ptr + b * 3 + 0, mask=mask, other=0.0)
        dy = _tl.load(d_pos0_ptr + b * 3 + 1, mask=mask, other=0.0) - _tl.load(d_pos1_ptr + b * 3 + 1, mask=mask, other=0.0)
        dz = _tl.load(d_pos0_ptr + b * 3 + 2, mask=mask, other=0.0) - _tl.load(d_pos1_ptr + b * 3 + 2, mask=mask, other=0.0)
        nrm = _tl.maximum(_tl.sqrt(dx * dx + dy * dy + dz * dz), 1e-8)
        nx = dx / nrm
        ny = dy / nrm
        nz = dz / nrm

        base = b * D + 3
        gx = _tl.load(g_ptr + base + 0, mask=mask, other=0.0)
        gy = _tl.load(g_ptr + base + 1, mask=mask, other=0.0)
        gz = _tl.load(g_ptr + base + 2, mask=mask, other=0.0)
        dot = nx * gx + ny * gy + nz * gz
        px = gx - dot * nx
        py = gy - dot * ny
        pz = gz - dot * nz
        if RELEASE_PROJ:
            act = _tl.load(active_ptr + b, mask=mask, other=False)
            px = _tl.where(act, gx, px)
            py = _tl.where(act, gy, py)
            pz = _tl.where(act, gz, pz)
        _tl.store(g_ptr + base + 0, px, mask=mask)
        _tl.store(g_ptr + base + 1, py, mask=mask)
        _tl.store(g_ptr + base + 2, pz, mask=mask)

    @_triton.jit
    def _donor_err(ideal_ptr, dpos_ptr, off, mask, r00, r01, r02, r10, r11, r12, r20, r21, r22):
        ix = _tl.load(ideal_ptr + off + 0, mask=mask, other=0.0)
        iy = _tl.load(ideal_ptr + off + 1, mask=mask, other=0.0)
        iz = _tl.load(ideal_ptr + off + 2, mask=mask, other=0.0)
        dx = _tl.load(dpos_ptr + off + 0, mask=mask, other=0.0)
        dy = _tl.load(dpos_ptr + off + 1, mask=mask, other=0.0)
        dz = _tl.load(dpos_ptr + off + 2, mask=mask, other=0.0)
        dnorm = _tl.maximum(_tl.sqrt(dx * dx + dy * dy + dz * dz), 1e-12)
        ux, uy, uz = dx / dnorm, dy / dnorm, dz / dnorm
        ex = (r00 * ix + r01 * iy + r02 * iz) - ux
        ey = (r10 * ix + r11 * iy + r12 * iz) - uy
        ez = (r20 * ix + r21 * iy + r22 * iz) - uz
        return ux, uy, uz, dnorm, ex, ey, ez, ex * ex + ey * ey + ez * ez

    @_triton.jit
    def _shape_constraint_R_fwd_bwd_kernel(
        R_ptr,
        ideal_ptr,
        dpos_ptr,
        mu_ptr,
        active_ptr,
        shape_dev_ptr,
        viol_ptr,
        loss_ptr,
        grad_dpos_ptr,
        B,
        N_SH,
        shape_tol,
        tau,
        log_n_sh,
        rho,
        RELEASE_SHAPE: _tl.constexpr,
        BLOCK_B: _tl.constexpr,
    ):
        pid = _tl.program_id(0)
        b = pid * BLOCK_B + _tl.arange(0, BLOCK_B)
        mask = b < B

        rbase = b * 9
        r00 = _tl.load(R_ptr + rbase + 0, mask=mask, other=1.0)
        r01 = _tl.load(R_ptr + rbase + 1, mask=mask, other=0.0)
        r02 = _tl.load(R_ptr + rbase + 2, mask=mask, other=0.0)
        r10 = _tl.load(R_ptr + rbase + 3, mask=mask, other=0.0)
        r11 = _tl.load(R_ptr + rbase + 4, mask=mask, other=1.0)
        r12 = _tl.load(R_ptr + rbase + 5, mask=mask, other=0.0)
        r20 = _tl.load(R_ptr + rbase + 6, mask=mask, other=0.0)
        r21 = _tl.load(R_ptr + rbase + 7, mask=mask, other=0.0)
        r22 = _tl.load(R_ptr + rbase + 8, mask=mask, other=1.0)

        max_val = _tl.full([BLOCK_B], -1e30, dtype=_tl.float32)
        sum_sq = _tl.zeros([BLOCK_B], dtype=_tl.float32)
        for n in range(N_SH):
            _, _, _, _, _, _, _, sq = _donor_err(ideal_ptr, dpos_ptr, (b * N_SH + n) * 3, mask, r00, r01, r02, r10, r11, r12, r20, r21, r22)
            sum_sq += sq
            max_val = _tl.maximum(max_val, tau * sq)

        if N_SH > 1:
            sum_exp = _tl.zeros([BLOCK_B], dtype=_tl.float32)
            for n in range(N_SH):
                _, _, _, _, _, _, _, sq = _donor_err(ideal_ptr, dpos_ptr, (b * N_SH + n) * 3, mask, r00, r01, r02, r10, r11, r12, r20, r21, r22)
                sum_exp += _tl.exp(tau * sq - max_val)
            lse = max_val + _tl.log(sum_exp)
            shape_dev = (lse - log_n_sh) / tau
        else:
            lse = max_val
            shape_dev = sum_sq

        viol = _tl.maximum(shape_dev - shape_tol, 0.0)
        mu = _tl.load(mu_ptr + b, mask=mask, other=0.0)
        loss = mu * viol + 0.5 * rho * viol * viol
        if RELEASE_SHAPE:
            act = _tl.load(active_ptr + b, mask=mask, other=False)
            loss_out = _tl.where(act, 0.0, loss)
            dloss_dshape = _tl.where((shape_dev > shape_tol) & (~act), mu + rho * viol, 0.0)
        else:
            loss_out = loss
            dloss_dshape = _tl.where(shape_dev > shape_tol, mu + rho * viol, 0.0)

        _tl.store(shape_dev_ptr + b, shape_dev, mask=mask)
        _tl.store(viol_ptr + b, viol, mask=mask)
        _tl.store(loss_ptr + b, loss_out, mask=mask)

        for n in range(N_SH):
            off = (b * N_SH + n) * 3
            ux, uy, uz, dnorm, ex, ey, ez, sq = _donor_err(ideal_ptr, dpos_ptr, off, mask, r00, r01, r02, r10, r11, r12, r20, r21, r22)
            if N_SH > 1:
                wn = _tl.exp(tau * sq - lse)
            else:
                wn = _tl.full([BLOCK_B], 1.0, dtype=_tl.float32)
            g_scale = dloss_dshape * wn * 2.0
            gux, guy, guz = -g_scale * ex, -g_scale * ey, -g_scale * ez
            dot = gux * ux + guy * uy + guz * uz
            _tl.store(grad_dpos_ptr + off + 0, (gux - dot * ux) / dnorm, mask=mask)
            _tl.store(grad_dpos_ptr + off + 1, (guy - dot * uy) / dnorm, mask=mask)
            _tl.store(grad_dpos_ptr + off + 2, (guz - dot * uz) / dnorm, mask=mask)


@dataclass
class IKSolverConfig:
    """Public hyperparameters for JointIKSolver."""

    n_steps: int = 200
    lr: float = 0.01
    shape_tol: Optional[float] = None
    n_workers: Optional[int] = None
    worker_mode: str = "thread"
    cull_at: Tuple[float, ...] = (0.3,)
    cull_keep: Tuple[float, ...] = (0.35,)




class JointIKSolver:
    """
    Joint gradient-descent IK solver optimising all ligands simultaneously on CUDA
    using zero-autograd Triton kernels and CUDA-graph replay.
    """

    _fixed_shape_replay = True
    replay = True
    timing_sync = False
    trace_hook = None
    diagnostic_hook = None
    final_hook = None
    last_bend_prior = None
    finish_endpoint_hook = None
    cull_blocks = None
    sym_rows = None

    def __init__(
        self,
        fks: list,
        scorer: "Scorer",
        ge: "GeometryExtractor",
        config: Optional[IKSolverConfig] = None,
        device: str = "cuda",
    ) -> None:
        self.fks = fks
        self.scorer = scorer
        self.ge = ge
        dev_str = str(device).split(":")[0].lower()
        if dev_str != "cuda":
            raise RuntimeError("JointIKSolver requires a CUDA device (GPU-only Triton build).")
        self.device = "cuda"

        cfg = config or IKSolverConfig()
        self.n_steps = cfg.n_steps
        self.lr = cfg.lr
        if cfg.shape_tol is None:
            from jacobik.ik.targets import SHAPE_TOL_POOLED

            self.shape_tol = SHAPE_TOL_POOLED
        else:
            self.shape_tol = cfg.shape_tol
        self.n_workers = cfg.n_workers
        self.worker_mode = cfg.worker_mode
        self.cull_at = tuple(cfg.cull_at)
        self.cull_keep = tuple(cfg.cull_keep)

    def _bend_prior(self, thetas) -> cp.ndarray:
        """Energy-calibrated angle-bend log-prior (ik/angle_bends.energy_prior), summed over ligands; 0 without bends."""
        total = None
        cfg = getattr(self, "bend_prior", None)
        if not cfg:
            return cp.zeros((thetas[0].shape[0],), dtype=thetas[0].dtype)
        from jacobik.ik import angle_bends as AB

        for fk, th in zip(self.fks, thetas):
            nb = getattr(fk, "n_angle_bends", 0)
            if not nb:
                continue
            n0 = 6 + fk.n_joints
            fz = getattr(fk, "_bend_free", None)
            fz = cp.maximum(fz, float(cfg["free_deg"])) if fz is not None else cfg["free_deg"]
            p = AB.energy_prior(th[:, n0:n0 + nb], fk._bend_k, fz, cfg["scale"])
            total = p if total is None else total + p
        if total is None:
            return cp.zeros((thetas[0].shape[0],), dtype=thetas[0].dtype)
        if _BEND_PER_Z and hasattr(self.scorer, "_agg_norm"):
            total = total / float(self.scorer._agg_norm())
        return total

    @timing.timed("ik_solve")
    def solve(
        self,
        theta_inits: list,
        X_ref,
        donor_pos_idxs: list,
        ideal_directions,
        debug: bool = False,
        symmetric_groups: Optional[List[List[int]]] = None,
        q3_symmetric_groups: Optional[List[List[int]]] = None,
        row_groups=None,
    ):
        """Joint IK solve for all ligands on CUDA."""
        self.last_bend_prior = None
        return self._solve_impl(
            theta_inits,
            X_ref,
            donor_pos_idxs,
            ideal_directions,
            debug=debug,
            symmetric_groups=symmetric_groups,
            q3_symmetric_groups=q3_symmetric_groups,
            row_groups=row_groups,
            cull_enabled=False,
        )

    def _solve_impl(
        self,
        theta_inits: list,
        X_ref,
        donor_pos_idxs: list,
        ideal_directions,
        debug: bool = False,
        symmetric_groups: Optional[List[List[int]]] = None,
        q3_symmetric_groups: Optional[List[List[int]]] = None,
        row_groups=None,
        cull_enabled: bool = True,
    ):
        import time as _time0

        _t_enter = _time0.perf_counter()
        # ===== set-up: verbatim from JointIKSolver._solve_impl =====
        device = self.device
        import sys as _sys
        from jacobik.ik.ik_setup import _finalize_solve_state, _init_solve_state

        _cfg_mod = _sys.modules[__name__]
        _st = _init_solve_state(
            self, theta_inits, X_ref, donor_pos_idxs, ideal_directions, symmetric_groups, q3_symmetric_groups, _cfg_mod
        )
        B, T = _st.B, _st.T
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


        # ===== replay-friendly step loop =====
        groups_c = gpu.as_gpu(row_groups, dtype=cp.int64) if row_groups is not None else None
        dt = thetas[0].dtype
        tl = cp.zeros((1,), dtype=cp.int64)                      # step index, on the device
        bc1 = cp.asarray([1.0 - _ADAM_BETA1 ** (i + 1) for i in range(T)], dtype=dt)
        bc2 = cp.asarray([1.0 - _ADAM_BETA2 ** (i + 1) for i in range(T)], dtype=dt)
        nan = float("nan")
        hist_res = cp.full((T, B), nan, dtype=dt)
        hist_score = cp.full((T, B), nan, dtype=dt)
        hist_gn = cp.full((T, B), nan, dtype=dt)
        hist_shape = cp.full((T, B), nan, dtype=dt)
        if dt != cp.float32:
            raise RuntimeError(f"JointIKSolver requires float32 on CUDA, got {dt}.")
        if phase is not None and phase.sigma_source == "curvature":
            raise RuntimeError("JointIKSolver does not support curvature sigma_source on CUDA.")
        if JointIKSolver.diagnostic_hook is not None:
            raise RuntimeError("JointIKSolver does not support diagnostic_hook on CUDA.")

        improved_buf = cp.zeros((B,), dtype=cp.bool_)
        adam_step_bufs = [cp.empty_like(th) for th in thetas] if phase is None else None
        _agg_norm_val = float(self.scorer._agg_norm()) if phase is not None else 1.0
        _bidentate_didx = [(int(didx[0].item()), int(didx[1].item())) if didx.shape[0] == 2 else None for didx in donor_pos_idxs]
        _release_proj = phase is not None and ("projection" in phase.release or phase.whole)

        from jacobik.ik.fk_compose import ComplexFK
        cfk = ComplexFK(self.fks)        # unsupported layouts (a ligand without donor atoms) run the per-ligand path

        from jacobik.ik import angle_bends as _AB
        _bp_cfg = getattr(self, "bend_prior", None)
        _bp_agg = float(self.scorer._agg_norm()) if (_BEND_PER_Z and hasattr(self.scorer, "_agg_norm")) else 1.0
        _bp_ligs = []
        if _bp_cfg:
            for _li, _fk in enumerate(self.fks):
                _nb = getattr(_fk, "n_angle_bends", 0)
                if _nb:
                    _n0 = 6 + _fk.n_joints
                    _fz = getattr(_fk, "_bend_free", None)
                    _fz = cp.maximum(_fz, float(_bp_cfg["free_deg"])) if _fz is not None else float(_bp_cfg["free_deg"])
                    _bp_ligs.append((_li, _n0, _n0 + _nb, _fk._bend_k, _fz, float(_bp_cfg["scale"])))
        _release_shape = phase is not None and ("shape" in phase.release or phase.whole)
        _release_sym = phase is not None and ("symmetry" in phase.release or phase.whole)
        _shape_dev_buf = cp.empty((B,), dtype=dt)
        _viol_buf = cp.empty((B,), dtype=dt)
        _shape_loss_buf = cp.empty((B,), dtype=dt)
        _grad_dpos_buf = cp.empty((B, self.ge.n_donors, 3), dtype=dt)
        _has_q3_sym = bool(symmetric_groups and q3_site_to_idx and _q3_sym_group_idx_tensors)

        def step(need_score_grads=None):
            X_current = cfk.forward_no_grad(thetas, X_ref)
            all_donor_pos_det = self.ge.donor_positions(X_current)
            stat_score, grad_X_stat, _ = self.scorer.score_and_grad(
                X_current, q3_mode_override=q3_mode_override, donor_pos=all_donor_pos_det,
                row_groups=groups_c, return_comps=False,
            )
            bp_grads = {}
            if _bp_ligs:
                bp_tot = None
                for _li, _n0, _n1, _bk, _fz, _bsc in _bp_ligs:
                    _p, _bg = _AB.energy_prior_and_grad(thetas[_li][:, _n0:_n1], _bk, _fz, _bsc)
                    bp_tot = _p if bp_tot is None else (bp_tot + _p)
                    bp_grads[_li] = _bg / _bp_agg
                if bp_tot is not None:
                    stat_score = stat_score + bp_tot / _bp_agg

            _dn = cp.maximum(cp.sqrt(cp.sum(all_donor_pos_det * all_donor_pos_det, axis=-1, keepdims=True)), 1e-12)
            donor_unit_vecs = all_donor_pos_det / _dn
            R_opt = _optimal_rotation_nocentroid(ideal_directions_dev, donor_unit_vecs)
            N_sh = all_donor_pos_det.shape[1]
            _act_p = phase.active if _release_shape else _shape_dev_buf
            gpu.launch(
                _shape_constraint_R_fwd_bwd_kernel,
                (gpu.cdiv(B, 64),),
                cp.ascontiguousarray(R_opt), cp.ascontiguousarray(ideal_directions_dev),
                cp.ascontiguousarray(all_donor_pos_det),
                mu_shape, _act_p, _shape_dev_buf, _viol_buf, _shape_loss_buf, _grad_dpos_buf,
                B, N_sh, float(self.shape_tol), float(_LSE_TEMP),
                math.log(max(N_sh, 1)), float(_SHAPE_RHO),
                RELEASE_SHAPE=_release_shape, BLOCK_B=64,
            )
            shape_dev = _shape_dev_buf
            violation = _viol_buf
            constraint_loss = _shape_loss_buf

            sym_penalty = cp.zeros((B,), dtype=dt)
            sym_angle_penalty = cp.zeros((B,), dtype=dt)
            sym_d_grads = []
            if symmetric_groups:
                all_donor_norms = cp.sqrt(cp.sum(all_donor_pos_det * all_donor_pos_det, axis=-1))
                for _gi, idx_t in enumerate(_sym_group_idx_tensors):
                    d_g = all_donor_norms[:, idx_t]
                    dev_g = d_g - cp.mean(d_g, axis=-1, keepdims=True)
                    _wg = 1.0 / _n_sym_groups if _sym_w is None else _sym_w[_gi]
                    sym_penalty = sym_penalty + _wg * cp.mean(dev_g * dev_g, axis=-1)
                    sym_d_grads.append((idx_t, dev_g * (2.0 / d_g.shape[1]) * (_wg if _sym_w is None else _wg[:, None])))

            sym_violation = cp.maximum(sym_penalty - _SYM_TOL, 0.0)
            grad_X_q3_sym = None
            if _has_q3_sym:
                q3_angles_now = self.ge.q3_approach_angles(X_current, haptic=_q3_sym_haptic)
                sap = cp.zeros((B,), dtype=dt)
                q3_a_grads = []
                for _gi, idx_t2 in enumerate(_q3_sym_group_idx_tensors):
                    a_g = q3_angles_now[:, idx_t2]
                    dev_a = a_g - cp.mean(a_g, axis=-1, keepdims=True)
                    _wa = 1.0 / _n_q3_sym_groups if _q3_sym_w is None else _q3_sym_w[_gi]
                    sap = sap + _wa * cp.mean(dev_a * dev_a, axis=-1)
                    q3_a_grads.append((idx_t2, dev_a * (2.0 / a_g.shape[1]) * (_wa if _q3_sym_w is None else _wa[:, None])))
                sym_angle_penalty = sap
                sav = cp.maximum(sap - _SYM_TOL, 0.0)
                dloss_dsav = cp.where(sap > _SYM_TOL, mu_sym_angle + _SYM_RHO * sav, 0.0)
                if _release_sym:
                    dloss_dsav = cp.where(phase.active, 0.0, dloss_dsav)
                grad_q3_angles = cp.zeros_like(q3_angles_now)
                for idx_t2, g_da in q3_a_grads:
                    cp.add.at(grad_q3_angles, (slice(None), idx_t2), dloss_dsav[:, None] * g_da)
                grad_X_q3_sym = self.ge.q3_approach_angles_vjp(X_current, grad_q3_angles, haptic=_q3_sym_haptic)
            sym_angle_violation = cp.maximum(sym_angle_penalty - _SYM_TOL, 0.0)

            if phase is not None:
                raw_sym_loss = (
                    mu_sym * sym_violation + 0.5 * _SYM_RHO * sym_violation * sym_violation
                    + mu_sym_angle * sym_angle_violation + 0.5 * _SYM_RHO * sym_angle_violation * sym_angle_violation
                )
                phase.observe(tl, -stat_score + constraint_loss + raw_sym_loss)
                if _release_shape:
                    constraint_loss = cp.where(phase.active, 0.0, constraint_loss)
                if _release_sym:
                    sym_violation = cp.where(phase.active, 0.0, sym_violation)
                    sym_angle_violation = cp.where(phase.active, 0.0, sym_angle_violation)

            constraint_loss = (
                constraint_loss
                + mu_sym * sym_violation + 0.5 * _SYM_RHO * sym_violation * sym_violation
                + mu_sym_angle * sym_angle_violation + 0.5 * _SYM_RHO * sym_angle_violation * sym_angle_violation
            )

            if sym_d_grads:
                dloss_dsym = cp.where(sym_penalty > _SYM_TOL, mu_sym + _SYM_RHO * sym_violation, 0.0)
                if _release_sym:
                    dloss_dsym = cp.where(phase.active, 0.0, dloss_dsym)
                for idx_t, g_dg in sym_d_grads:
                    d_g = cp.maximum(all_donor_norms[:, idx_t], 1e-12)
                    g_vec = (dloss_dsym[:, None] * g_dg / d_g)[:, :, None] * all_donor_pos_det[:, idx_t, :]
                    cp.add.at(_grad_dpos_buf, (slice(None), idx_t, slice(None)), g_vec)

            grad_X_constr = cp.zeros_like(X_current)
            if self.ge.n_donors > 0:
                if self.ge._donor_all_kappa:
                    cp.add.at(grad_X_constr, (slice(None), self.ge._donor_flat_idx, slice(None)), _grad_dpos_buf)
                else:
                    _g_flat = (_grad_dpos_buf[:, :, None, :] * self.ge._donor_atom_w.astype(dt)).reshape(
                        B, self.ge.n_donors * self.ge._donor_max_f, 3
                    )
                    cp.add.at(grad_X_constr, (slice(None), self.ge._donor_flat_idx, slice(None)), _g_flat)
            if grad_X_q3_sym is not None:
                grad_X_constr = grad_X_constr + grad_X_q3_sym

            snap_score = stat_score
            if has_closure:
                _cl_diff = X_current[:, _cl_a] - X_current[:, _cl_b]
                _cl_dist = cp.sqrt(cp.sum(_cl_diff * _cl_diff, axis=-1))
                clos = cp.sum((_cl_dist - _cl_d0) ** 2, axis=-1)
                snap_score = snap_score - 1000.0 * clos

            _need_sg = (phase is not None) if need_score_grads is None else bool(need_score_grads)
            score_grads = None
            if _need_sg:
                sg_fk, cg_fk = cfk.vjp(cp.stack([-grad_X_stat, grad_X_constr], axis=0))
                for _li, _bg in bp_grads.items():
                    _, _n0, _n1, _, _, _ = _bp_ligs[[x[0] for x in _bp_ligs].index(_li)]
                    sg_fk[_li][:, _n0:_n1] -= _bg
                score_grads = sg_fk
                grads_all = [sg + cg for sg, cg in zip(sg_fk, cg_fk)]
            else:
                grads_all = cfk.vjp(grad_X_constr - grad_X_stat)
                for _li, _bg in bp_grads.items():
                    _, _n0, _n1, _, _, _ = _bp_ligs[[x[0] for x in _bp_ligs].index(_li)]
                    grads_all[_li][:, _n0:_n1] -= _bg

            total_grads: List[cp.ndarray] = []
            for li, (didx, theta_li) in enumerate(zip(donor_pos_idxs, thetas)):
                g = grads_all[li] if grads_all[li] is not None else cp.zeros_like(theta_li)
                bd = _bidentate_didx[li]
                if bd is not None:
                    g = cp.ascontiguousarray(g)
                    d0 = cp.ascontiguousarray(all_donor_pos_det[:, bd[0], :])
                    d1 = cp.ascontiguousarray(all_donor_pos_det[:, bd[1], :])
                    act_p = phase.active if _release_proj else d0
                    gpu.launch(
                        _bidentate_proj_kernel,
                        (gpu.cdiv(B, 64),),
                        d0, d1, g, act_p, B, g.shape[1],
                        RELEASE_PROJ=_release_proj, BLOCK_B=64,
                    )
                total_grads.append(g)

            r = -stat_score + constraint_loss
            grad_norm_b = cp.sqrt(sum(cp.sum(g * g, axis=-1) for g in total_grads))
            gpu.launch(
                _al_ring_dual_update_kernel,
                (gpu.cdiv(B, 128),),
                tl, stat_score, snap_score, r, grad_norm_b, shape_dev,
                violation, sym_violation, sym_angle_violation,
                score_ring, ever_plateaued, best_score_ever, improved_buf,
                hist_res, hist_score, hist_gn, hist_shape,
                violation_ring, mu_shape_frozen, mu_shape,
                sym_violation_ring, sym_angle_violation_ring,
                mu_sym_frozen, mu_sym_angle_frozen, mu_sym, mu_sym_angle,
                B, W=W, MW=MW, SW=SW,
                plateau_atol=float(_PLATEAU_ATOL), shape_rho=float(_SHAPE_RHO),
                mu_shape_stall_atol=float(_MU_SHAPE_STALL_ATOL),
                sym_rho=float(_SYM_RHO), sym_stall_atol=float(_SYM_STALL_ATOL),
                BLOCK_SIZE=128,
            )
            for li, theta_li in enumerate(thetas):
                D_li = theta_li.shape[1]
                gpu.launch(
                    _best_snapshot_copy_kernel,
                    (gpu.cdiv(B * D_li, 256),),
                    improved_buf, theta_li, best_score_thetas[li], B, D_li, BLOCK_SIZE=256,
                )

            b1, b2, eps_a = _ADAM_BETA1, _ADAM_BETA2, _ADAM_EPS
            lr_t = self.lr
            if phase is not None:
                scales = ([tab.take(tl, axis=0) for tab in _step_scale_tabs]
                          if _step_scale_tabs is not None else None)
                sg_scaled = [g * _agg_norm_val for g in score_grads] if score_grads is not None else None
                steps = phase.update(tl, total_grads, sg_scaled, r, scales, self.lr)
            elif endpoint_m is None:
                steps = []
                for li, (theta_li, g) in enumerate(zip(thetas, total_grads)):
                    D_li = theta_li.shape[1]
                    st_out = adam_step_bufs[li]
                    has_ho = handoff_delta is not None
                    ho_p = handoff_delta[li] if has_ho else st_out
                    has_sc = _step_scale_tabs is not None and _step_scale_tabs[li] is not None
                    sc_p = _step_scale_tabs[li] if has_sc else st_out
                    gpu.launch(
                        _adam_lig_kernel,
                        (gpu.cdiv(B * D_li, 256),),
                        cp.ascontiguousarray(g), adam_m[li], adam_v[li], st_out, ho_p, sc_p, tl, bc1, bc2,
                        B, D_li, float(lr_t), float(b1), float(b2), float(eps_a),
                        HAS_HANDOFF=has_ho, HAS_SCALE=has_sc, BLOCK_SIZE=256,
                    )
                    steps.append(st_out)
            else:
                c1 = bc1.take(tl, axis=0)
                c2 = bc2.take(tl, axis=0)
                steps = []
                for li, (theta_li, g) in enumerate(zip(thetas, total_grads)):
                    adam_m[li][...] = b1 * adam_m[li] + (1.0 - b1) * g
                    adam_v[li][...] = b2 * adam_v[li] + (1.0 - b2) * g * g
                    endpoint_m[li][...] = adam_m[li]
                    endpoint_v[li][...] = adam_v[li]
                    endpoint_t[...] = tl + 1
                    m_hat = adam_m[li] / c1
                    v_hat = adam_v[li] / c2
                    step_val = lr_t * m_hat / (cp.sqrt(v_hat) + eps_a)
                    if handoff_delta is not None:
                        handoff_delta[li][...] = cp.abs(step_val)
                    if _step_scale_tabs is not None and _step_scale_tabs[li] is not None:
                        step_val = step_val * _step_scale_tabs[li].take(tl, axis=0)
                    steps.append(step_val)
            for li, (theta_li, step_val) in enumerate(zip(thetas, steps)):
                if li in cl_d0:
                    c0_ = self.fks[li]._macro_col0
                    c1_ = c0_ + self.fks[li].n_macro_joints
                    step_val = step_val.copy()
                    step_val[:, c0_:c1_] = self.fks[li].closure_fused_step(X_current, cl_d0[li], step_val[:, c0_:c1_])
                done_p = phase.done if phase is not None else theta_li
                gpu.launch(
                    _theta_post_update_kernel,
                    (gpu.cdiv(B, 64),),
                    theta_li, cp.ascontiguousarray(step_val), done_p, B, theta_li.shape[1],
                    HAS_DONE=phase is not None, BLOCK_B=64,
                )
            tl[...] += 1


        import time as _time

        def _now():
            gpu.sync()
            return _time.perf_counter()

        LAST.clear()
        LAST.update(mode="eager", replayed=0, B=B, T=T, fast_fk=cfk is not None, error=None,
                    s_setup=_now() - _t_enter, s_step0=0.0, s_record=0.0, s_replay=0.0, s_eager=0.0, s_final=0.0)
        t = 0
        # Phase switch point for skipping score_grads during the pre-switch search steps
        _sg_start = None
        if phase is not None and phase.sigma_source != "curvature" and not phase.whole:
            _sg_start = int(phase.fixed_switch) if phase.fixed_switch is not None else int(phase.start)
        if self.replay and JointIKSolver.trace_hook is None and T > 2:
            stream = gpu.new_stream()
            _ts = self.timing_sync

            def _replay_until(limit: int, need_sg: bool, check_stop: bool) -> None:
                nonlocal t
                g = gpu.capture(stream, lambda: step(need_score_grads=need_sg))
                with stream:
                    while t < limit:
                        g.launch(stream)
                        if _ts:
                            gpu.sync(stream)
                        t += 1
                        LAST["replayed"] += 1
                        if check_stop and phase is not None and t % 8 == 0 and bool(phase.stop.item()):
                            break
                del g

            try:
                _t0 = _now()
                _split_phase = _sg_start is not None and 2 < _sg_start < T - 1
                _need_sg0 = False if (_sg_start is not None and _sg_start > 0) else (phase is not None)
                with stream:
                    step(need_score_grads=_need_sg0)              # step 0 eagerly: also the warm-up
                t = 1
                _t1 = _now()
                LAST["s_step0"] = _t1 - _t0
                LAST["mode"] = "replay"
                if _split_phase:
                    _t2 = _now()
                    _replay_until(_sg_start, False, False)
                    LAST["s_record"] = _t2 - _t1
                    with stream:
                        step(need_score_grads=True)
                    t += 1
                    if t < T:
                        _replay_until(T, True, True)
                    LAST["s_replay"] = _now() - _t2
                else:
                    _need_sg_all = False if (_sg_start is not None and _sg_start >= T) else (phase is not None)
                    _t2 = _now()
                    _replay_until(T, _need_sg_all, True)
                    LAST["s_record"] = _t2 - _t1
                    LAST["s_replay"] = _now() - _t2
                gpu.sync(stream)
            except Exception:
                LAST["error"] = traceback.format_exc()[-3000:]
                LAST["mode"] = "eager after failed recording at step %d" % t
                log.warning("step recording failed at step %d; continuing eagerly\n%s", t, LAST["error"])
                gpu.sync(stream)
                t = int(tl.item())                             # steps actually applied
        _t3 = _now()
        while t < T and not (phase is not None and t % 8 == 0 and bool(phase.stop.item())):
            _need_sg_t = (phase is not None) and (_sg_start is None or t >= _sg_start)
            step(need_score_grads=_need_sg_t)
            t += 1
            if phase is not None and t % 8 == 0 and bool(phase.stop.item()):
                break
        LAST["steps_executed"] = t
        _t4 = _now()
        LAST["s_eager"] = _t4 - _t3

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

        LAST["s_final"] = _now() - _t4
        solve_trace = {
            "residual": gpu.to_cpu(hist_res),
            "score": gpu.to_cpu(hist_score),
            "grad_norm": gpu.to_cpu(hist_gn),
            "shape_dev": gpu.to_cpu(hist_shape),
        }
        thetas_final = [th.copy() for th in best_thetas]
        return thetas_final, converged, final_score, solve_trace
