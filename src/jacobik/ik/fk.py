"""
fk.py
=====
Batched, differentiable forward kinematics for the IK engine.

Joint list: one canonical BFS over rigid bodies from the ligand's donor-free
root body (one torsion joint per rotatable bond).  Donors that are branch
points additionally get "bend" joints (see ForwardKinematics) after the
n_joints_base torsion joints.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import numpy as np

from jacobik.core.gpu import cp


def _wrap_angle(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _gather_atoms(X, idx):
    return X[:, idx]


def _skew(v):
    """Skew-symmetric matrix [v]_x for v of shape (..., 3) -> (..., 3, 3)."""
    S = cp.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    S[..., 0, 1] = -v[..., 2]; S[..., 0, 2] = v[..., 1]
    S[..., 1, 0] = v[..., 2];  S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]; S[..., 2, 1] = v[..., 0]
    return S


def _left_jacobian(omega):
    """SO(3) left Jacobian J_l(omega) for omega (B, 3) -> (B, 3, 3), capture-safe."""
    theta2 = (omega * omega).sum(axis=-1, keepdims=True)
    theta2_c = cp.maximum(theta2, 1e-12)
    angle = cp.sqrt(theta2_c)
    A_c = cp.where(theta2 < 1e-6, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - cp.cos(angle)) / theta2_c)
    Bc_c = cp.where(theta2 < 1e-6, 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0, (angle - cp.sin(angle)) / cp.maximum(angle * theta2, 1e-12))
    W = _skew(omega)
    WW = (W[:, :, :, None] * W[:, None, :, :]).sum(axis=2)
    Jl = A_c[..., None] * W + Bc_c[..., None] * WW
    Jl[:, 0, 0] += 1.0; Jl[:, 1, 1] += 1.0; Jl[:, 2, 2] += 1.0
    return Jl


def _axisangle_to_matrix(omega):
    theta2 = cp.sum(omega * omega, axis=-1, keepdims=True)
    theta = cp.sqrt(cp.maximum(theta2, 1e-12))
    K = _skew(omega / theta)
    s, c = cp.sin(theta)[..., None], cp.cos(theta)[..., None]
    KK = (K[:, :, :, None] * K[:, None, :, :]).sum(axis=2)
    R = s * K + (1.0 - c) * KK
    R[:, 0, 0] += 1.0; R[:, 1, 1] += 1.0; R[:, 2, 2] += 1.0
    W = _skew(omega)
    W[:, 0, 0] += 1.0; W[:, 1, 1] += 1.0; W[:, 2, 2] += 1.0
    return cp.where(theta2[..., None] < 1e-12, W, R)


def _rodrigues(pts, pivot, axis, delta, axis_normalized=True):
    if not axis_normalized:
        axis = axis / cp.maximum(cp.linalg.norm(axis, axis=-1, keepdims=True), 1e-8)
    v = pts - pivot[:, None, :]
    c = cp.cos(delta)[:, None, None]
    s = cp.sin(delta)[:, None, None]
    k = axis[:, None, :]
    v_rot = v * c + cp.cross(k, v, axis=-1) * s + k * (cp.sum(k * v, axis=-1, keepdims=True) * (1.0 - c))
    return pivot[:, None, :] + v_rot


def _dihedral(p0, p1, p2, p3):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    u = b1 / cp.maximum(cp.linalg.norm(b1, axis=-1, keepdims=True), 1e-8)
    n1 = cp.cross(b0, b1, axis=-1)
    n2 = cp.cross(b1, b2, axis=-1)
    return cp.arctan2(cp.sum(cp.cross(n1, n2, axis=-1) * u, axis=-1), cp.sum(n1 * n2, axis=-1))


def _dihedral_axis_from_quad(P):
    b0 = P[:, 1] - P[:, 0]
    b1 = P[:, 2] - P[:, 1]
    b2 = P[:, 3] - P[:, 2]
    u = b1 / cp.maximum(cp.linalg.norm(b1, axis=-1, keepdims=True), 1e-8)
    n1 = cp.cross(b0, b1, axis=-1)
    n2 = cp.cross(b1, b2, axis=-1)
    phi = cp.arctan2(cp.sum(cp.cross(n1, n2, axis=-1) * u, axis=-1), cp.sum(n1 * n2, axis=-1))
    return phi, u

if TYPE_CHECKING:
    from jacobik.core.types import Ligand, TMComplex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror geometry.py for consistency)
# ---------------------------------------------------------------------------

_BO_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Internal helpers (CPU, called only during __init__)
# ---------------------------------------------------------------------------


# Production resolved configuration (hardcoded values)
_SMALL_RING_OPEN_ADJ = True

from jacobik.core import profile as _profile  # noqa: E402

#: v1.0b: macrocycles get the opening-adjacent joints too (profile field macro_open_adj)
MACRO_OPEN_ADJ = _profile.get("macro_open_adj")


def _build_pos_map(mol: "TMComplex") -> Dict[int, int]:
    """atom.idx (1-based CSD) → position index (0-based in mol.atoms)."""
    return {a.idx: i for i, a in enumerate(mol.atoms)}


# ---------------------------------------------------------------------------
# ForwardKinematics
# ---------------------------------------------------------------------------



# Production resolved configuration (hardcoded values)
BRANCH_BENDS = True


class _FKEvalMixin:
    """FK evaluation, alignment, ring-closure projection, and Jacobian methods for ``ForwardKinematics``."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bend_seed_angles(self, X_ref):
        """
        Current dihedral angle of each bend joint in X_ref, used to seed the
        bend-joint columns of theta so they are a no-op at t=0.  Returns an
        empty (0,) array if no donor qualified for a bend joint.
        """
        if self.n_bend_joints == 0:
            return cp.zeros(0, dtype=X_ref.dtype)

        X = X_ref if X_ref.ndim == 2 else X_ref[0]
        k0, k1 = self.n_joints_base, self.n_joints_base + self.n_bend_joints
        qi = cp.asarray(self._flanking_i[k0:k1], dtype=cp.int64)
        qj = cp.asarray(self._pivot_j[k0:k1], dtype=cp.int64)
        qk = cp.asarray(self._pivot_k[k0:k1], dtype=cp.int64)
        ql = cp.asarray(self._flanking_l[k0:k1], dtype=cp.int64)
        return _dihedral(X[qi], X[qj], X[qk], X[ql])

    def q5_alignment(self, ge, X_ref):
        """
        Map this ligand's base torsion joints onto the scorer's Q5 rows.
        """
        J = self.n_joints_base
        if J == 0:
            return cp.zeros(0, dtype=cp.int64), cp.zeros(0, dtype=X_ref.dtype)
        X = X_ref if X_ref.ndim == 2 else X_ref[0]
        row_of = {}
        q5 = cp.stack([ge._q5_i, ge._q5_j, ge._q5_k, ge._q5_l], axis=1).tolist() if int(ge._q5_j.shape[0]) else []
        for r, (_, a, b, _) in enumerate(q5):
            row_of[(min(a, b), max(a, b))] = r

        qi = cp.asarray(self._flanking_i[:J], dtype=cp.int64)
        qj = cp.asarray(self._pivot_j[:J], dtype=cp.int64)
        qk = cp.asarray(self._pivot_k[:J], dtype=cp.int64)
        ql = cp.asarray(self._flanking_l[:J], dtype=cp.int64)
        phi_fk = _dihedral(X[qi], X[qj], X[qk], X[ql])

        rows_l = [-1] * J
        q5_match_k, q5_quads = [], []
        for k in range(J):
            pj, pk = self._pivot_j[k], self._pivot_k[k]
            r = row_of.get((min(int(pj), int(pk)), max(int(pj), int(pk))))
            if r is not None:
                rows_l[k] = r
                q5_match_k.append(k)
                q5_quads.append(q5[r])
        rows = cp.asarray(rows_l, dtype=cp.int64)
        offs = phi_fk.copy()
        if q5_match_k:
            qt = cp.asarray(q5_quads, dtype=cp.int64)
            phi_q5 = _dihedral(X[qt[:, 0]], X[qt[:, 1]], X[qt[:, 2]], X[qt[:, 3]])
            mk = cp.asarray(q5_match_k, dtype=cp.int64)
            offs[mk] = _wrap_angle(phi_fk[mk] - phi_q5)
        return rows, offs

    def macro_seed_angles(self, P):
        """
        Macrocycle-joint dihedrals for conformers, used to seed the trailing
        macro columns of theta.
        """
        if not self.macro_quads_atom:
            return cp.zeros((P.shape[0], 0), dtype=P.dtype)
        C = P.shape[0]
        M = len(self.macro_quads_atom)
        off = self._lig_first
        qt = cp.asarray([[fi - off, pj - off, pk - off, fl - off] for fi, pj, pk, fl in self.macro_quads_atom],
                        dtype=cp.int64)
        Pm = P[:, qt.reshape(-1)].reshape(C * M, 4, 3)
        return _dihedral(Pm[:, 0], Pm[:, 1], Pm[:, 2], Pm[:, 3]).reshape(C, M)

    def seam_d0(self, X_ref):
        """Seam distances of each row's own (closed) reference conformer: (B or 1, R)."""
        c = self._cl
        Xr = X_ref if X_ref.ndim == 3 else X_ref[None]
        xs = Xr[:, c["seam"]]
        return cp.linalg.norm(xs[:, c["a"]] - xs[:, c["b"]], axis=-1)

    def closure_residual_jacobian(self, X, d0):
        """
        Seam residuals r = |x_a - x_b| - d0 (B, R) and their exact Jacobian with
        respect to the macro-joint dihedrals (B, R, K).
        """
        c = self._cl
        xs = X[:, c["seam"]]
        pj = X[:, c["pj"]]
        u = X[:, c["pk"]] - pj
        u = u / cp.maximum(cp.linalg.norm(u, axis=-1, keepdims=True), 1e-8)
        rel = xs[:, None, :, :] - pj[:, :, None, :]
        dx = cp.cross(cp.broadcast_to(u[:, :, None, :], rel.shape), rel, axis=-1)
        dx = dx * c["mask"][None, :, :, None]
        diff = xs[:, c["a"]] - xs[:, c["b"]]
        dist = cp.maximum(cp.linalg.norm(diff, axis=-1), 1e-8)
        e = diff / dist[..., None]
        dpair = dx[:, :, c["a"]] - dx[:, :, c["b"]]
        J = (e[:, None, :, :] * dpair).sum(axis=-1).swapaxes(1, 2)
        return dist - d0, J

    @staticmethod
    def _solve_small_spd(A, b):
        """Capture-safe batched SPD linear solve A x = b via pure-CuPy Cholesky (no cuSOLVER/cuBLAS)."""
        R = A.shape[-1]
        if R == 1:
            return b / cp.where(cp.abs(A) > 1e-12, A, 1e-12)
        L = cp.zeros_like(A)
        for j in range(R):
            s = A[:, j, j] - (L[:, j, :j] * L[:, j, :j]).sum(axis=-1) if j > 0 else A[:, j, j]
            L[:, j, j] = cp.sqrt(cp.maximum(s, 1e-12))
            for i in range(j + 1, R):
                t = A[:, i, j] - (L[:, i, :j] * L[:, j, :j]).sum(axis=-1) if j > 0 else A[:, i, j]
                L[:, i, j] = t / L[:, j, j]
        y = cp.zeros_like(b[:, :, 0])
        for i in range(R):
            rhs_i = b[:, i, 0] - (L[:, i, :i] * y[:, :i]).sum(axis=-1) if i > 0 else b[:, i, 0]
            y[:, i] = rhs_i / L[:, i, i]
        x = cp.zeros_like(y)
        for i in range(R - 1, -1, -1):
            rhs_i = y[:, i] - (L[:, i + 1:, i] * x[:, i + 1:]).sum(axis=-1) if i + 1 < R else y[:, i]
            x[:, i] = rhs_i / L[:, i, i]
        return x[..., None]

    def closure_fused_step(self, X, d0, s_m):
        """
        Fold ring closure into an optimiser update without another FK pass.
        """
        r, J = self.closure_residual_jacobian(X, d0)
        rhs = r - (J * s_m[:, None, :]).sum(axis=-1)
        A = (J[:, :, None, :] * J[:, None, :, :]).sum(axis=-1)
        R = r.shape[-1]
        for i in range(R):
            A[:, i, i] += 1e-4
        sol = self._solve_small_spd(A, rhs[..., None])
        corr = -(J * sol).sum(axis=1)
        return s_m - cp.clip(corr, -0.5, 0.5)

    def closure_project(
        self,
        theta,
        X_ref,
        iters: int = 4,
        damping: float = 1e-4,
        max_step: float = 0.5,
    ):
        """
        Damped Gauss-Newton projection of the macro-joint dihedrals onto the
        ring-closure manifold.
        """
        if self._cl is None:
            return theta, cp.zeros(theta.shape[0], dtype=theta.dtype)
        c0 = self._macro_col0
        theta = theta.copy()
        eye = cp.eye(self._cl["a"].shape[0], dtype=theta.dtype)
        d0 = self.seam_d0(X_ref)
        for _ in range(iters):
            X = self.apply(theta, X_ref)
            r, J = self.closure_residual_jacobian(X, d0)
            A = J @ J.swapaxes(-1, -2) + damping * eye
            sol = cp.linalg.solve(A, r[..., None])
            step = -(J.swapaxes(-1, -2) @ sol)[..., 0]
            c1 = c0 + self.n_macro_joints
            theta[:, c0:c1] = _wrap_angle(theta[:, c0:c1] + cp.clip(step, -max_step, max_step))
        r, _ = self.closure_residual_jacobian(self.apply(theta, X_ref), d0)
        return theta, (r * r).sum(axis=-1)

    def apply(
        self,
        theta,
        X_ref,
    ):
        """Apply the full FK map: base transform then all J torsions."""
        n = 6 + self.n_joints
        X = self._apply_internal(theta[:, :n] if theta.shape[1] > n else theta, X_ref, self.n_joints)
        if self.n_angle_bends and theta.shape[1] >= n + self.n_angle_bends:
            X = self._bend_plan.apply(X, theta[:, n:n + self.n_angle_bends])
        return X

    def partial_apply(
        self,
        theta,
        X_ref,
        n_joints: int,
    ):
        """Apply the base transform and only the first n_joints torsions."""
        n_joints = max(0, min(n_joints, self.n_joints))
        return self._apply_internal(theta, X_ref, n_joints)

    def _apply_internal(
        self,
        theta,
        X_ref,
        n_joints: int,
    ):
        B = theta.shape[0]
        if X_ref.ndim == 2:
            X = cp.broadcast_to(X_ref[None, ...], (B, X_ref.shape[0], X_ref.shape[1])).copy()
        else:
            X = X_ref.copy()

        if self._lig_pos_t.size == 0:
            return X

        X = self._apply_base_transform(theta, X)
        for k in range(n_joints):
            X = self._apply_torsion(theta, X, k)
        return X

    def _apply_base_transform(
        self,
        theta,
        X,
    ):
        transl = theta[:, 0:3]
        omega = theta[:, 3:6]
        R = _axisangle_to_matrix(omega)

        if self._donor_pos_t.size > 0:
            donor_pos = _gather_atoms(X, self._donor_pos_t)
            pivot = donor_pos.mean(axis=1)
        else:
            pivot = X.mean(axis=1)

        lig = _gather_atoms(X, self._lig_pos_t)
        lig_centered = lig - pivot[:, None, :]
        lig_rotated = (lig_centered[:, :, None, :] * R[:, None, :, :]).sum(axis=-1)
        lig_new = lig_rotated + pivot[:, None, :] + transl[:, None, :]
        X[:, self._lig_pos_t, :] = lig_new
        return X

    def _apply_torsion(
        self,
        theta,
        X,
        k: int,
    ):
        target_phi = theta[:, 6 + k]
        P = X[:, self._quad_idx[k]]
        phi_cur, axis_unit = _dihedral_axis_from_quad(P)
        delta = target_phi - phi_cur

        down_idx = self._downstream[k]
        if down_idx.size == 0:
            return X

        pts = _gather_atoms(X, down_idx)
        pivot = P[:, 1]
        pts_rotated = _rodrigues(pts, pivot, axis_unit, delta, axis_normalized=True)
        X_out = X.copy()
        X_out[:, down_idx, :] = pts_rotated
        return X_out

    def forward_with_state(self, theta, X_ref):
        """Run FK and return (X_out, state) for analytical ``vjp()``."""
        B = theta.shape[0]
        if X_ref.ndim == 2:
            X = cp.broadcast_to(X_ref[None, ...], (B, X_ref.shape[0], X_ref.shape[1])).copy()
        else:
            X = X_ref.copy()
        if self._lig_pos_t.size == 0:
            return X, ("empty", theta.shape)

        # Base transform
        X_in_base = X.copy()
        transl = theta[:, 0:3]
        omega = theta[:, 3:6]
        R = _axisangle_to_matrix(omega)
        if self._donor_pos_t.size > 0:
            pivot_b = _gather_atoms(X, self._donor_pos_t).mean(axis=1)
        else:
            pivot_b = X.mean(axis=1)
        lig = _gather_atoms(X, self._lig_pos_t)
        lig_centered = lig - pivot_b[:, None, :]
        lig_new = (lig_centered[:, :, None, :] * R[:, None, :, :]).sum(axis=-1) + pivot_b[:, None, :] + transl[:, None, :]
        X[:, self._lig_pos_t, :] = lig_new
        base_state = (omega, R, lig_centered, X_in_base.shape[1])

        # Torsions
        n = 6 + self.n_joints
        th_t = theta[:, :n] if theta.shape[1] > n else theta
        torsion_states = []
        for k in range(self.n_joints):
            down_idx = self._downstream[k]
            if down_idx.size == 0:
                torsion_states.append(None)
                continue
            P = X[:, self._quad_idx[k]]
            phi_cur, axis_unit = _dihedral_axis_from_quad(P)
            delta = th_t[:, 6 + k] - phi_cur
            pts = _gather_atoms(X, down_idx)
            pivot = P[:, 1]
            pts_rot = _rodrigues(pts, pivot, axis_unit, delta, axis_normalized=True)
            torsion_states.append((k, P, axis_unit, delta, pts, pts_rot))
            X_next = X.copy()
            X_next[:, down_idx, :] = pts_rot
            X = X_next

        bend_state = None
        if self.n_angle_bends and theta.shape[1] >= n + self.n_angle_bends:
            X, bend_state = self._bend_plan.forward_with_state(X, theta[:, n:n + self.n_angle_bends])
        return X, ("ok", theta.shape, base_state, torsion_states, bend_state)

    def vjp(self, state, grad_X):
        """Analytical backward pass for ForwardKinematics: returns (grad_theta, grad_X_ref)."""
        if state[0] == "empty":
            return cp.zeros(state[1], dtype=grad_X.dtype), grad_X.copy()
        _, th_shape, base_state, torsion_states, bend_state = state
        gX = grad_X.copy()
        gtheta = cp.zeros(th_shape, dtype=grad_X.dtype)
        n = 6 + self.n_joints

        if bend_state is not None:
            gX, gbeta = self._bend_plan.vjp(bend_state, gX)
            gtheta[:, n:n + self.n_angle_bends] = gbeta

        for ts in reversed(torsion_states):
            if ts is None:
                continue
            k, P, u, delta, pts, pts_rot = ts
            down_idx = self._downstream[k]
            g_pts = gX[:, down_idx, :]
            pivot = P[:, 1]
            v = pts - pivot[:, None, :]
            v_rot = pts_rot - pivot[:, None, :]

            g_delta = cp.sum(g_pts * cp.cross(u[:, None, :], v_rot, axis=-1), axis=(1, 2))
            gtheta[:, 6 + k] += g_delta

            c = cp.cos(delta)[:, None, None]
            s = cp.sin(delta)[:, None, None]
            u3 = u[:, None, :]
            g_v = g_pts * c + cp.cross(g_pts, u3, axis=-1) * s + u3 * (cp.sum(g_pts * u3, axis=-1, keepdims=True) * (1.0 - c))
            g_pivot = g_pts.sum(axis=1) - g_v.sum(axis=1)

            uv = cp.sum(u3 * v, axis=-1, keepdims=True)
            guv = cp.sum(g_pts * u3, axis=-1, keepdims=True)
            g_u = cp.sum(
                (g_pts * uv + v * guv) * (1.0 - c) + cp.cross(v, g_pts, axis=-1) * s,
                axis=1,
            )

            gX[:, down_idx, :] = g_v
            cp.add.at(gX, (slice(None), self._quad_idx[k][:, 1] if self._quad_idx[k].ndim > 1 else self._quad_idx[k][1]), g_pivot)

            # Dihedral + axis VJP
            b0 = P[:, 1] - P[:, 0]
            b1 = P[:, 2] - P[:, 1]
            b2 = P[:, 3] - P[:, 2]
            b1_len = cp.maximum(cp.linalg.norm(b1, axis=-1, keepdims=True), 1e-8)
            n1 = cp.cross(b0, b1, axis=-1)
            n2 = cp.cross(b1, b2, axis=-1)
            n1_sq = cp.maximum(cp.sum(n1 * n1, axis=-1, keepdims=True), 1e-12)
            n2_sq = cp.maximum(cp.sum(n2 * n2, axis=-1, keepdims=True), 1e-12)
            b1_sq = b1_len * b1_len

            g_phi = -g_delta[:, None]
            dp0 = -g_phi * (b1_len / n1_sq) * n1
            dp3 = g_phi * (b1_len / n2_sq) * n2
            c01 = cp.sum(b0 * b1, axis=-1, keepdims=True) / b1_sq
            c21 = cp.sum(b2 * b1, axis=-1, keepdims=True) / b1_sq
            dp1 = -(1.0 + c01) * dp0 + c21 * dp3
            dp2 = -(dp0 + dp1 + dp3)

            g_b1 = (g_u - cp.sum(g_u * u, axis=-1, keepdims=True) * u) / b1_len
            dp1 = dp1 - g_b1
            dp2 = dp2 + g_b1

            q = self._quad_idx[k]
            cp.add.at(gX, (slice(None), q[0]), dp0)
            cp.add.at(gX, (slice(None), q[1]), dp1)
            cp.add.at(gX, (slice(None), q[2]), dp2)
            cp.add.at(gX, (slice(None), q[3]), dp3)

        # Base transform VJP
        omega, R, lig_centered, N_atoms = base_state
        g_lig = gX[:, self._lig_pos_t, :]
        gtheta[:, 0:3] += g_lig.sum(axis=1)
        g_R = (g_lig[:, :, :, None] * lig_centered[:, :, None, :]).sum(axis=1)
        A = (R[:, :, None, :] * g_R[:, None, :, :]).sum(axis=-1)
        g_vec = cp.stack([A[:, 1, 2] - A[:, 2, 1], A[:, 2, 0] - A[:, 0, 2], A[:, 0, 1] - A[:, 1, 0]], axis=-1)

        Jl = _left_jacobian(omega)
        gtheta[:, 3:6] += (Jl * g_vec[:, :, None]).sum(axis=1)

        g_lig_centered = (g_lig[:, :, None, :] * R[:, None, :, :]).sum(axis=2)
        g_pivot_b = g_lig.sum(axis=1) - g_lig_centered.sum(axis=1)
        gX[:, self._lig_pos_t, :] = g_lig_centered
        if self._donor_pos_t.size > 0:
            cp.add.at(gX, (slice(None), self._donor_pos_t), g_pivot_b[:, None, :] / float(self._donor_pos_t.size))
        else:
            gX = gX + g_pivot_b[:, None, :] / float(N_atoms)
        return gtheta, gX

    def jacobian_column_norm_sq(self, theta, X_ref):
        """Analytic diagonal of J^T J for this FK map, refreshed at the supplied pose."""
        B, width = theta.shape
        out = cp.zeros((B, width), dtype=theta.dtype)
        nlig = int(self._lig_pos_t.size)
        if nlig == 0:
            return out
        X_ref_b = cp.broadcast_to(X_ref[None, ...], (B, X_ref.shape[0], X_ref.shape[1])) if X_ref.ndim == 2 else X_ref

        out[:, :3] = float(nlig)
        Xbase = self._apply_base_transform(theta, X_ref_b.copy())
        X = Xbase
        joint_axes, joint_pivots, joint_down = [], [], []
        for k in range(self.n_joints):
            P = X[:, self._quad_idx[k]]
            _, axis = _dihedral_axis_from_quad(P)
            down = self._downstream[k]
            joint_axes.append(axis)
            joint_pivots.append(P[:, 1])
            joint_down.append(down)
            X = self._apply_torsion(theta, X, k)

        n = 6 + self.n_joints
        beta = theta[:, n:n + self.n_angle_bends] if self.n_angle_bends else None
        X_prebend = X
        Xfinal = self._bend_plan.apply(X, beta) if beta is not None and self._bend_plan is not None else X

        if self._donor_pos_t.size > 0:
            pivot = _gather_atoms(X_ref_b, self._donor_pos_t).mean(axis=1)
        else:
            pivot = X_ref_b.mean(axis=1)
        r = _gather_atoms(Xfinal, self._lig_pos_t) - pivot[:, None, :] - theta[:, None, :3]
        Jl = _left_jacobian(theta[:, 3:6])
        skew_r = _skew(r)
        jrot = -skew_r @ Jl[:, None, :, :]
        out[:, 3:6] = (jrot * jrot).sum(axis=(1, 2))

        for k, (axis, pv, down) in enumerate(zip(joint_axes, joint_pivots, joint_down)):
            if down.size == 0:
                continue
            lever = X_prebend[:, down] - pv[:, None, :]
            deriv = cp.cross(axis[:, None, :], lever, axis=-1)
            out[:, 6 + k] = (deriv * deriv).sum(axis=(1, 2))

        if beta is not None and self._bend_plan is not None:
            out[:, n:n + self.n_angle_bends] = self._bend_plan.jacobian_column_norm_sq(X_prebend, beta)
        return out


class ForwardKinematics(_FKEvalMixin):
    """
    Batched differentiable FK map for one ligand of a TMComplex.

    Joints are the ligand's rotatable bonds (n_joints_base torsion joints),
    followed by n_bend_joints "branch bend" joints: for a kappa donor with >= 2
    outgoing rotatable bonds, one joint per outgoing branch rotates only that
    branch's downstream subtree about the donor's incoming (root-ward) bond.
    The donor, its root-ward neighbour and sibling branches stay fixed.
    n_joints_base == n_joints when no donor qualifies.  Bend joints have no
    fitted Q5 statistics.

    Parameters
    ----------
    mol    : fully assembled TMComplex (quantized bonds, decomposed ligands)
    ligand : one Ligand from mol.ligands (must have pivot_chains + decomposition)
    device : torch.device
    """

    def __init__(
        self,
        mol: "TMComplex",
        ligand: "Ligand",
        device=None,
        angle_bends: bool = False,
    ) -> None:
        self.device = device

        atom_map = {a.idx: a for a in mol.atoms}
        idx_to_pos = _build_pos_map(mol)
        from jacobik.topology.graph import (
            get_heavy_ligand_neighbors,
            get_ligand_neighbors,
            heaviest_heavy_neighbor,
        )

        heavy_nb = get_heavy_ligand_neighbors(
            mol.atoms, mol.bonds, mol.metal_idx, bo_threshold=_BO_THRESHOLD
        )
        all_nb = get_ligand_neighbors(
            mol.atoms, mol.bonds, mol.metal_idx, bo_threshold=_BO_THRESHOLD
        )

        # ------------------------------------------------------------------
        # Ligand atom position indices
        # ------------------------------------------------------------------
        self._lig_pos_indices: List[int] = [
            idx_to_pos[idx] for idx in ligand.atom_indices if idx in idx_to_pos
        ]
        self._lig_first = min(ligand.atom_indices) if ligand.atom_indices else 0
        self._lig_pos_t = cp.asarray(self._lig_pos_indices, dtype=cp.int64)

        donor_atom_idxs: List[int] = []
        for chain in ligand.pivot_chains:
            donor_atom_idxs.extend(chain.donor_atoms)
        # Deduplicate while preserving order
        seen: Set[int] = set()
        donor_atom_idxs_dedup = []
        for idx in donor_atom_idxs:
            if idx not in seen and idx in idx_to_pos:
                seen.add(idx)
                donor_atom_idxs_dedup.append(idx)
        self._donor_pos_indices: List[int] = [idx_to_pos[idx] for idx in donor_atom_idxs_dedup]
        if self._donor_pos_indices:
            self._donor_pos_t = cp.asarray(self._donor_pos_indices, dtype=cp.int64)
        else:
            self._donor_pos_t = cp.zeros(0, dtype=cp.int64)

        decomp = ligand.decomposition  # _LigandGraph: body_of, pivot_bonds, rigid_bodies

        # body_of: atom.idx → body_index
        body_of: Dict[int, int] = decomp.body_of
        # rigid_bodies: list of sets of atom.idx per body
        body_sets: List[Set[int]] = decomp.rigid_bodies

        # body_adj_for_bfs[b] = list of (neighbour_body, atom_in_b, atom_in_neighbour)
        # for every rotatable bond crossing out of body b.
        body_adj_for_bfs: Dict[int, List[Tuple[int, int, int]]] = {}
        for rb in ligand.rotatable_bonds:
            u, v = rb.i, rb.j
            if u not in body_of or v not in body_of:
                continue
            bu, bv = body_of[u], body_of[v]
            if bu != bv:
                body_adj_for_bfs.setdefault(bu, []).append((bv, u, v))
                body_adj_for_bfs.setdefault(bv, []).append((bu, v, u))

        # Canonical BFS from the ligand's donor-free root body: an order-independent
        # joint list where every donor sits behind at least one real joint.
        root_body: int = getattr(decomp, "root_body", 0)
        ordered_joints: List[Tuple[int, int]] = []  # (j_atom, k_atom) atom.idx pairs
        visited_bodies_root: Set[int] = {root_body}
        queue_root: deque = deque([root_body])
        while queue_root:
            curr = queue_root.popleft()
            for nb_body, j_atom, k_atom in sorted(body_adj_for_bfs.get(curr, [])):
                if nb_body in visited_bodies_root:
                    continue
                visited_bodies_root.add(nb_body)
                queue_root.append(nb_body)
                ordered_joints.append((j_atom, k_atom))

        seen_bonds: Set[Tuple[int, int]] = set()
        from jacobik.stats.keys import _build_type_map, flank_side as _flank_side, linear_atoms

        _linear = linear_atoms(_build_type_map(mol))

        # Ordered lists parallel to the joint sequence
        flanking_i_list: List[int] = []
        pivot_j_list: List[int] = []  # pos indices
        pivot_k_list: List[int] = []
        flanking_l_list: List[int] = []
        downstream_lists: List[List[int]] = []  # pos indices per joint

        for j_atom, k_atom in ordered_joints:
            # Skip coordination bonds (metal-ligand, not a torsion DOF)
            if j_atom == mol.metal_idx or k_atom == mol.metal_idx:
                continue

            bond_key = (min(j_atom, k_atom), max(j_atom, k_atom))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)

            # Flanking atoms: heavy neighbours first, else H. At a linear (sp) atom the neighbours are collinear with
            # the bond and the measured dihedral is rounding noise (XECSON: 1e-6 A input shifts moved atoms 2.8 A), so
            # the flank comes from the first off-axis atom along the linear chain (key_builder.flank_side, shared with
            # the Q5 rows). A joint with no off-axis atom on one side (M-C#C-R, R-C#C-H) is not a torsion relative to
            # anything on that side: its rotation is the ligand's own base rotation about that axis, so it is left out.
            j_nbs = _flank_side(j_atom, k_atom, heavy_nb, all_nb, _linear)
            k_nbs = _flank_side(k_atom, j_atom, heavy_nb, all_nb, _linear)

            if not j_nbs or not k_nbs:
                continue

            fi_atom = heaviest_heavy_neighbor(j_nbs, atom_map)
            fl_atom = heaviest_heavy_neighbor(k_nbs, atom_map)

            for atom in (fi_atom, j_atom, k_atom, fl_atom):
                if atom not in idx_to_pos:
                    break
            else:
                # Downstream bodies: BFS from body_of[k_atom], not
                # allowed to cross back through body_of[j_atom].
                # This correctly handles branching chains.
                if k_atom not in body_of or j_atom not in body_of:
                    continue
                start_body = body_of[k_atom]
                forbid_body = body_of[j_atom]

                visited_bodies: Set[int] = {start_body}
                queue_b = [start_body]
                while queue_b:
                    curr_b = queue_b.pop()
                    for nb_b, _, _ in body_adj_for_bfs.get(curr_b, []):
                        if nb_b not in visited_bodies and nb_b != forbid_body:
                            visited_bodies.add(nb_b)
                            queue_b.append(nb_b)

                down_pos = sorted(
                    idx_to_pos[a_idx]
                    for b_idx in visited_bodies
                    for a_idx in body_sets[b_idx]
                    if a_idx in idx_to_pos
                )

                flanking_i_list.append(idx_to_pos[fi_atom])
                pivot_j_list.append(idx_to_pos[j_atom])
                pivot_k_list.append(idx_to_pos[k_atom])
                flanking_l_list.append(idx_to_pos[fl_atom])
                downstream_lists.append(down_pos)

        self.n_joints_base = len(pivot_j_list)

        # Bend joints reuse the parent joint's (flanking_i, pivot_j) and the child
        # joint's (pivot_k, downstream); _apply_torsion treats them like any joint.
        parent_joint_of: Dict[int, int] = {}
        child_joints_of: Dict[int, List[int]] = {}
        for jidx in range(len(pivot_j_list)):
            parent_joint_of.setdefault(pivot_k_list[jidx], jidx)
            child_joints_of.setdefault(pivot_j_list[jidx], []).append(jidx)

        kappa_donor_pos: List[int] = []
        for chain in ligand.pivot_chains:
            if chain.is_haptic:
                continue
            for d_idx in chain.donor_atoms:
                if d_idx in idx_to_pos:
                    kappa_donor_pos.append(idx_to_pos[d_idx])

        for d_pos in (kappa_donor_pos if BRANCH_BENDS else []):
            parent_idx = parent_joint_of.get(d_pos)
            children = child_joints_of.get(d_pos, [])
            if parent_idx is None or len(children) < 2:
                # d_pos is root itself (no parent joint), or not a real
                # branch point (<=1 outgoing branch -- the existing
                # incoming joint already gives it full solo control).
                continue
            for child_idx in children:
                flanking_i_list.append(flanking_i_list[parent_idx])
                pivot_j_list.append(pivot_j_list[parent_idx])
                pivot_k_list.append(d_pos)
                flanking_l_list.append(pivot_k_list[child_idx])
                downstream_lists.append(downstream_lists[child_idx])


        # ------------------------------------------------------------------
        # Macrocycle joints: after base and bend joints.  Each macrocycle is
        # opened at one bond; torsions on the remaining ring bonds (anchor
        # outward along two arms) move the atoms beyond them, and the
        # opening is closed by soft distance restraints (closure_pairs).
        # ------------------------------------------------------------------
        self.n_macro_joints = 0
        self._openings: List[Tuple[int, int]] = []   # closure bond of each jointed ring
        self.macro_quads_atom: List[Tuple[int, int, int, int]] = []
        closure_pairs: List[Tuple[int, int]] = []
        closure_d0: List[float] = []
        n_before_macro = len(pivot_j_list)
        macrocycles = getattr(ligand, "macrocycles", None) or []
        if macrocycles:
            import networkx as nx

            lig_set = set(ligand.atom_indices)
            Gl = nx.Graph()
            Gl.add_nodes_from(lig_set)
            for bnd in mol.bonds:
                if bnd.i in lig_set and bnd.j in lig_set:
                    Gl.add_edge(bnd.i, bnd.j)
            _bo = {}
            for bnd in mol.bonds:
                if bnd.i in lig_set and bnd.j in lig_set:
                    _bo[(min(bnd.i, bnd.j), max(bnd.i, bnd.j))] = float(bnd.order)

            def _single(u, v):
                return abs(_bo.get((min(u, v), max(u, v)), 1.0) - 1.0) < 0.25

            def _pi(x):
                return any(not _single(x, y) for y in Gl.neighbors(x))

            def _conj(u, v):
                lp = lambda x: atom_map[x].symbol in ("N", "O", "S")
                return (_pi(u) and (_pi(v) or lp(v))) or (_pi(v) and (_pi(u) or lp(u)))

            from jacobik.topology.decompose import _RING_JOINT_MIN_SIZE, MACRO_FUSED, RING_JOINTS
            min_n = _RING_JOINT_MIN_SIZE if RING_JOINTS else 8
            # bends this ligand gets (angle_bends): (fixed points, moved atoms) of each, to choose a ring opening no
            # bend disturbs
            _bend_moves = []
            if angle_bends:
                from jacobik.ik import angle_bends as _AB

                for sp in _AB.bend_specs(mol, ligand):
                    hinge = len(sp) > 4 and sp[4][0] == "hinge"
                    fixed = {sp[0], sp[1]} if hinge else {sp[0]}
                    moved = set(sp[3])
                    _bend_moves.append((fixed, moved))
            for ring in macrocycles:
                n = len(ring)
                if any(a not in idx_to_pos for a in ring) or n < min_n:
                    continue
                ring_set = set(ring)
                na = (n - 1) // 2
                nb_ = n - 1 - na
                # anchor: most heavy non-ring neighbours, among anchors whose opposite (closure) bond is single
                # (small donor rings may hold double bonds, which stay rigid and can never be the opening)
                cands = sorted(
                    range(n),
                    key=lambda q: -sum(
                        1
                        for nb in Gl.neighbors(ring[q])
                        if nb not in ring_set and atom_map[nb].symbol != "H"
                    ),
                )
                anchor = None
                if _bend_moves:
                    # Open the ring where no bend disturbs the seam: every bend either moves none of the seam atoms
                    # (the opening atoms and their neighbours) or moves all of them except points on its own axis, so
                    # swinging a fold or hinge never changes a seam distance and the closure never undoes it (COD: the
                    # folds the substituent-bend statistics guide). Any ring, any bridge length.
                    def _undisturbed(q):
                        u, v = ring[(q + na) % n], ring[(q + na + 1) % n]
                        if not _single(u, v):
                            return False
                        S = {u, v} | set(Gl.neighbors(u)) | set(Gl.neighbors(v))
                        return all(not (S & mv) or (S - fx) <= mv for fx, mv in _bend_moves)
                    anchor = next((q for q in cands if _undisturbed(q)), None)
                shared = set()
                if MACRO_FUSED:
                    # bonds this ring shares with another ring stay rigid and never open it
                    for cyc in nx.minimum_cycle_basis(Gl):
                        if set(cyc) != ring_set:
                            cs = set(cyc)
                            shared |= {(min(a, b), max(a, b)) for a, b in Gl.subgraph(cs).edges()}
                    def _opens(q):
                        u, v = ring[(q + na) % n], ring[(q + na + 1) % n]
                        return _single(u, v) and (min(u, v), max(u, v)) not in shared
                    # keep a bend-safe opening when there is one; else the first anchor whose opening is unshared
                    if anchor is None or not _opens(anchor):
                        anchor = next((q for q in cands if _opens(q) and (not _bend_moves or _undisturbed(q))),
                                      next((q for q in cands if _opens(q)), None))
                if anchor is None:
                    anchor = next((q for q in cands if _single(ring[(q + na) % n], ring[(q + na + 1) % n])), None)
                if anchor is None:
                    continue
                rr = ring[anchor:] + ring[:anchor]  # rr[0] is the anchor
                cl_u, cl_v = rr[na], rr[na + 1]  # closure bond
                self._openings.append((cl_u, cl_v))
                arm_a = [rr[m] for m in range(0, na + 1)]  # r0..r_na
                arm_b = [rr[0]] + [rr[-m] for m in range(1, nb_ + 1)]  # r0, r-1..
                arm_joints: List[List[Tuple[int, int, int, int]]] = []
                # Small donor rings (< 8 atoms or holding a non-single bond): the bond into each opening atom is a joint
                # too. It turns the opening atom's substituent frame, i.e. where the seam partner can sit; left out, that
                # torsion stays at its embed value and a 6-8 ring has no closure-preserving freedom left, so the ring
                # is locked in its seed fold (COD 2026-09-30: the bound tub unreachable from the free twist-boat).
                small = n < 8 or not all(_single(ring[q], ring[(q + 1) % n]) for q in range(n))
                last = 1 if ((small and _SMALL_RING_OPEN_ADJ) or MACRO_OPEN_ADJ) else 2
                for arm in (arm_a, arm_b):
                    js: List[Tuple[int, int, int, int]] = []
                    for m in range(len(arm) - last):
                        j_at, k_at = arm[m], arm[m + 1]
                        if not _single(j_at, k_at):
                            continue  # a double bond in the ring stays rigid (no torsion joint)
                        if MACRO_FUSED:
                            if (min(j_at, k_at), max(j_at, k_at)) in shared:
                                continue  # fused bond: rigid
                            if _conj(j_at, k_at):
                                continue  # conjugated single bond (amide, aryl-N, aryl-C=O): planar, rigid
                            H_ = Gl.copy()
                            H_.remove_edge(j_at, k_at)
                            H_.remove_edge(cl_u, cl_v)
                            if nx.has_path(H_, j_at, k_at):
                                continue  # not a torsion of the opened ring (another path joins its ends)
                        fi = ring[(ring.index(j_at) - 1) % n]
                        if fi == k_at:
                            fi = ring[(ring.index(j_at) + 1) % n]
                        fl = ring[(ring.index(k_at) - 1) % n]
                        if fl == j_at:
                            fl = ring[(ring.index(k_at) + 1) % n]
                        if MACRO_OPEN_ADJ and not small and {k_at, fl} == {cl_u, cl_v}:
                            # the joint into an opening atom: its ring neighbour across the opening sits on the other
                            # arm, so measuring the dihedral to it couples this joint to every joint of that arm (the
                            # analytic closure Jacobian was off by up to 0.8 A/rad, 2026-11-20); use a substituent
                            # of the opening atom instead, which moves rigidly with this joint
                            subs = sorted(x for x in Gl.neighbors(k_at) if x not in (j_at, cl_u, cl_v))
                            if not subs:
                                continue
                            fl = subs[0]
                        js.append((fi, j_at, k_at, fl))
                    arm_joints.append(js)
                # Interleave the arms so nested (outer) joints follow inner ones.
                ja, jb = arm_joints
                order_ab = []
                for m in range(max(len(ja), len(jb))):
                    if m < len(ja):
                        order_ab.append(ja[m])
                    if m < len(jb):
                        order_ab.append(jb[m])
                for fi, j_at, k_at, fl in order_ab:
                    H = Gl.copy()
                    H.remove_edge(j_at, k_at)
                    H.remove_edge(cl_u, cl_v)
                    comp = nx.node_connected_component(H, k_at)
                    down_pos = sorted(idx_to_pos[a] for a in comp if a in idx_to_pos)
                    flanking_i_list.append(idx_to_pos[fi])
                    pivot_j_list.append(idx_to_pos[j_at])
                    pivot_k_list.append(idx_to_pos[k_at])
                    flanking_l_list.append(idx_to_pos[fl])
                    downstream_lists.append(down_pos)
                    self.macro_quads_atom.append((fi, j_at, k_at, fl))
                # Closure: the opening bond length plus the distance from each opening atom to EVERY other neighbour
                # (H included) of the other one. That fixes the whole relative pose of the two ends except the
                # dihedral about the opening bond. With only the two ring 1-3 distances, each end could still spin
                # about its ring bond, which bends its H (and other substituents) into the opening
                # (TINBIE: C-C-H 17 deg, +1500 kJ/mol).
                seam_pairs = [(cl_u, cl_v)]
                seam_pairs += [(cl_u, x) for x in sorted(Gl.neighbors(cl_v)) if x != cl_u]
                seam_pairs += [(cl_v, y) for y in sorted(Gl.neighbors(cl_u)) if y != cl_v]
                for a_, b_ in seam_pairs:
                    closure_pairs.append((idx_to_pos[a_], idx_to_pos[b_]))
                    closure_d0.append(
                        float(np.linalg.norm(atom_map[a_].position - atom_map[b_].position))
                    )
        self.n_macro_joints = len(pivot_j_list) - n_before_macro
        self.closure_pairs = closure_pairs
        self.closure_d0 = closure_d0

        self.n_joints = len(pivot_j_list)
        self.n_bend_joints = self.n_joints - self.n_joints_base - self.n_macro_joints

        # Static tensors for the closure Jacobian / projection (built once).
        # Seam atoms S, residual pairs as indices into S, and for every macro
        # joint the axis atoms plus a (K, S) mask of seam atoms it moves.
        self._macro_col0 = 6 + self.n_joints_base + self.n_bend_joints
        self._cl = None
        if closure_pairs and self.n_macro_joints > 0:
            seam = sorted({a for pr in closure_pairs for a in pr})
            sidx = {a: i for i, a in enumerate(seam)}
            mk = slice(self.n_joints_base + self.n_bend_joints, self.n_joints)
            down_sets = [set(d) for d in downstream_lists[mk]]
            mask = [[a in ds for a in seam] for ds in down_sets]
            self._cl = dict(
                seam=cp.asarray(seam, dtype=cp.int64),
                a=cp.asarray([sidx[a] for a, _ in closure_pairs], dtype=cp.int64),
                b=cp.asarray([sidx[b] for _, b in closure_pairs], dtype=cp.int64),
                d0=cp.asarray(closure_d0, dtype=cp.float32),
                pj=cp.asarray(pivot_j_list[mk], dtype=cp.int64),
                pk=cp.asarray(pivot_k_list[mk], dtype=cp.int64),
                mask=cp.asarray(mask, dtype=cp.float32),
            )

        if self.n_joints > 0:
            self._flanking_i = flanking_i_list
            self._pivot_j = pivot_j_list
            self._pivot_k = pivot_k_list
            self._flanking_l = flanking_l_list
            self._downstream = [
                cp.asarray(d, dtype=cp.int64) for d in downstream_lists
            ]
            self._quad_idx = [
                cp.asarray([fi, pj, pk, fl], dtype=cp.int64)
                for fi, pj, pk, fl in zip(
                    flanking_i_list, pivot_j_list, pivot_k_list, flanking_l_list
                )
            ]
        else:
            self._flanking_i = []
            self._pivot_j = []
            self._pivot_k = []
            self._flanking_l = []
            self._downstream = []
            self._quad_idx = []

        # Optional bond-angle bends on chelate backbones (ik/angle_bends.py): extra theta columns after
        # every other joint, applied after the normal FK. Not counted in n_joints.
        self._bend_plan = None
        self.n_angle_bends = 0
        if angle_bends:
            from jacobik.ik import angle_bends as AB

            _specs = AB.bend_specs(mol, ligand, openings=self._openings or None)
            if _specs:
                self._bend_specs = _specs
                self._bend_plan = AB.BendPlan(_specs, idx_to_pos, device)
                self._bend_k = cp.asarray(AB.stiffness(mol, _specs), dtype=cp.float32)
                self._bend_free = cp.asarray(AB.free_zone(_specs), dtype=cp.float32)
                self.n_angle_bends = self._bend_plan.K

        log.debug(
            "ForwardKinematics: ligand %d, %d atoms, %d joints",
            ligand.ligand_id,
            len(self._lig_pos_indices),
            self.n_joints,
        )
