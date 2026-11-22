"""
kernels.py
==========
Pure differentiable math kernels for the IK engine.
"""

from __future__ import annotations

import math
from typing import Tuple

import cupy as cp
import cupyx.scipy.special as cspec

from jacobik.core import gpu  # noqa: F401  (sets CUDA_PATH and Triton driver)
from jacobik.core.env import EPS, LOG_2PI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
_LOG_2PI = LOG_2PI
_EPS = EPS  # safe divisor / small-angle threshold


# ---------------------------------------------------------------------------
# Atom gather (index_select based)
# ---------------------------------------------------------------------------


def gather_atoms(X: cp.ndarray, idx: cp.ndarray) -> cp.ndarray:
    """
    Equivalent to ``X[:, idx, :]`` for an integer array ``idx`` of any shape.

    X : (B, N, 3);  idx : (...)  ->  (B, *idx.shape, 3)
    """
    return cp.take(X, idx, axis=1)


# ---------------------------------------------------------------------------
# Safe normalisation (finite gradient at zero vector)
# ---------------------------------------------------------------------------


def _safe_normalize(v: cp.ndarray, dim: int = -1) -> cp.ndarray:
    """
    Normalize v along dim with a safe floor at zero vectors.
    """
    norm = cp.sqrt(cp.maximum((v * v).sum(axis=dim, keepdims=True), _EPS * _EPS))
    return v / norm


# ---------------------------------------------------------------------------
# Rodrigues rotation
# ---------------------------------------------------------------------------


def rodrigues(
    pts: cp.ndarray,  # (B, n, 3) — points to rotate
    pivot: cp.ndarray,  # (B, 3)    — centre of rotation
    axis: cp.ndarray,  # (B, 3)    — rotation axis (need not be unit)
    theta: cp.ndarray,  # (B,)      — rotation angle in radians
    axis_normalized: bool = False,
) -> cp.ndarray:
    """
    Rotate a batch of point sets around a pivot/axis by angle theta.
    """
    assert pts.shape[0] == pivot.shape[0] == axis.shape[0] == theta.shape[0], (
        f"Batch dimension mismatch: pts={pts.shape[0]}, pivot={pivot.shape[0]}, "
        f"axis={axis.shape[0]}, theta={theta.shape[0]}"
    )

    k = axis if axis_normalized else _safe_normalize(axis, dim=-1)  # (B, 3)
    cos_t = cp.cos(theta)[:, None, None]  # (B, 1, 1)
    sin_t = cp.sin(theta)[:, None, None]  # (B, 1, 1)
    eye = cp.eye(3, dtype=pts.dtype)
    k_exp = cp.broadcast_to(k[:, None, :], (k.shape[0], 3, 3))
    RT = (
        cos_t * eye
        + (1.0 - cos_t) * (k[:, :, None] * k[:, None, :])
        + sin_t * cp.cross(k_exp, cp.broadcast_to(eye, k_exp.shape), axis=-1)
    )
    return pivot[:, None, :] + (pts - pivot[:, None, :]) @ RT


def dihedral_axis_from_quad(P: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Signed dihedral of (P[:,0], P[:,1], P[:,2], P[:,3]) plus the unit
    P1 -> P2 axis.
    """
    b = P[:, 1:] - P[:, :-1]  # (B, 3, 3): b1, b2, b3
    n = cp.cross(b[:, :2], b[:, 1:], axis=-1)  # (B, 2, 3): n1, n2
    n1 = n[:, 0]
    n2 = n[:, 1]
    b2 = b[:, 1]
    b2_hat = b2 / cp.sqrt(cp.maximum((b2 * b2).sum(axis=-1, keepdims=True), _EPS * _EPS))
    m = cp.cross(n1, n2, axis=-1)
    sin_phi = (m * b2_hat).sum(axis=-1)
    cos_phi = (n1 * n2).sum(axis=-1)
    return cp.arctan2(sin_phi, cos_phi), b2_hat


# ---------------------------------------------------------------------------
# Axis-angle → rotation matrix
# ---------------------------------------------------------------------------


def axisangle_to_matrix(omega: cp.ndarray) -> cp.ndarray:
    """
    Convert a batch of axis-angle vectors to rotation matrices via Rodrigues.
    """
    angle_sq = cp.maximum((omega * omega).sum(axis=-1, keepdims=True), _EPS * _EPS)
    angle = cp.sqrt(angle_sq)  # (B, 1)
    safe = angle.squeeze(-1) >= _EPS  # (B,) bool

    k = cp.where(
        safe[:, None],
        omega / cp.maximum(angle, _EPS),
        omega,
    )  # (B, 3)

    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = cp.zeros_like(kx)
    K = cp.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero],
        axis=-1,
    ).reshape(-1, 3, 3)

    I = cp.eye(3, dtype=omega.dtype)[None, :, :]
    cos_a = cp.cos(angle)[..., None]  # (B, 1, 1)
    sin_a = cp.sin(angle)[..., None]  # (B, 1, 1)

    kk = k[..., None] * k[:, None, :]  # (B, 3, 3)
    R_full = cos_a * I + sin_a * K + (1.0 - cos_a) * kk

    K_omega = cp.stack(
        [zero, -omega[:, 2], omega[:, 1], omega[:, 2], zero, -omega[:, 0], -omega[:, 1], omega[:, 0], zero],
        axis=-1,
    ).reshape(-1, 3, 3)
    R_taylor = I + K_omega

    mask = safe[:, None, None]
    return cp.where(mask, R_full, R_taylor)


# ---------------------------------------------------------------------------
# Signed dihedral angle
# ---------------------------------------------------------------------------


def dihedral_with_axis(
    p_i: cp.ndarray,  # (B, 3)
    p_j: cp.ndarray,  # (B, 3)
    p_k: cp.ndarray,  # (B, 3)
    p_l: cp.ndarray,  # (B, 3)
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Signed dihedral of the i-j-k-l quadruplet plus the unit p_j -> p_k axis.
    """
    b1 = p_j - p_i
    b2 = p_k - p_j
    b3 = p_l - p_k

    n1 = cp.cross(b1, b2, axis=-1)
    n2 = cp.cross(b2, b3, axis=-1)

    b2_hat = _safe_normalize(b2, dim=-1)
    m = cp.cross(n1, n2, axis=-1)

    sin_phi = (m * b2_hat).sum(axis=-1)
    cos_phi = (n1 * n2).sum(axis=-1)

    return cp.arctan2(sin_phi, cos_phi), b2_hat


def dihedral(
    p_i: cp.ndarray,  # (B, 3)
    p_j: cp.ndarray,  # (B, 3)
    p_k: cp.ndarray,  # (B, 3)
    p_l: cp.ndarray,  # (B, 3)
) -> cp.ndarray:
    """
    Signed dihedral angle of the i-j-k-l quadruplet, IUPAC convention.
    """
    phi, _ = dihedral_with_axis(p_i, p_j, p_k, p_l)
    return phi


# ---------------------------------------------------------------------------
# Angle wrapping
# ---------------------------------------------------------------------------


def wrap_angle(theta: cp.ndarray) -> cp.ndarray:
    """
    Wrap angles to [−π, π].
    """
    return (theta + math.pi) % _TWO_PI - math.pi


# ---------------------------------------------------------------------------
# VMM log-probability
# ---------------------------------------------------------------------------


def _logsumexp_last(x: cp.ndarray) -> cp.ndarray:
    m = cp.max(x, axis=-1, keepdims=True)
    return m.squeeze(-1) + cp.log(cp.sum(cp.exp(x - m), axis=-1))


def vmm_log_prob(
    phi: cp.ndarray,  # (B,)  — observed angles in radians
    means: cp.ndarray,  # (K,)  — component means in radians
    kappas: cp.ndarray,  # (K,)  — concentration parameters (κ ≥ 0)
    weights: cp.ndarray,  # (K,)  — mixture weights, must sum to 1
) -> cp.ndarray:
    """
    Log-probability of phi under a Von Mises Mixture Model.
    """
    cos_diff = cp.cos(phi[:, None] - means[None, :])  # (B, K)
    log_norm = _LOG_2PI + cp.log(cp.maximum(cspec.i0e(kappas), _EPS)) + kappas
    log_comp = kappas[None, :] * cos_diff - log_norm[None, :]
    log_w = cp.log(cp.maximum(weights, _EPS))
    return _logsumexp_last(log_comp + log_w[None, :])


# ---------------------------------------------------------------------------
# Half-Gaussian soft-wall (Q6 repulsive penalty)
# ---------------------------------------------------------------------------


def halfgauss_log_prob(
    d: cp.ndarray,  # (B,) — observed distances
    mu: cp.ndarray,  # ()   — wall position (scalar or broadcastable)
    sigma: cp.ndarray,  # ()   — wall width    (scalar or broadcastable)
) -> cp.ndarray:
    """
    One-sided soft-wall penalty for Q6 steric contacts.
    """
    sig = cp.maximum(sigma, _EPS)
    penalty = -((d - mu) ** 2) / (2.0 * sig * sig)
    return cp.where(d <= mu, penalty, cp.zeros_like(d))


# ---------------------------------------------------------------------------
# Fused Q6 (non-bonded contact soft-min)
# ---------------------------------------------------------------------------


def q6_softmin(
    X: cp.ndarray,
    a_idx: cp.ndarray,
    b_idx: cp.ndarray,
    mu: cp.ndarray,
    sigma: cp.ndarray,
    tau: float,
) -> cp.ndarray:
    """
    Fused ``-(1/tau) * logsumexp(-tau * halfgauss(|X[:, a] - X[:, b]|; mu, sigma))``.

    X : (B, N, 3); a_idx, b_idx : (M,) int; mu, sigma : (M,) -> (B,)
    """
    Xc = cp.ascontiguousarray(X.transpose(2, 1, 0))  # (3, N, B)
    var2 = (2.0 * cp.maximum(sigma, _EPS) ** 2)[:, None]  # (M, 1)
    diff = cp.take(Xc, a_idx, axis=1) - cp.take(Xc, b_idx, axis=1)  # (3, M, B)
    d = cp.sqrt((diff * diff).sum(axis=0))  # (M, B)
    tneg = cp.minimum(d - mu[:, None], 0.0)
    x = (tneg * tneg) * (float(tau) / var2)
    m = cp.max(x, axis=0)  # (B,)
    e = cp.exp(cp.maximum(x - m[None, :], -80.0))
    return -(m + cp.log(cp.sum(e, axis=0))) / float(tau)


# ---------------------------------------------------------------------------
# Fused Triton Kernels for Statistical Potentials (Q1, Q2, Q5, S_orient)
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _load3(ptr, base, mask):
        return (
            tl.load(ptr + base + 0, mask=mask, other=0.0),
            tl.load(ptr + base + 1, mask=mask, other=0.0),
            tl.load(ptr + base + 2, mask=mask, other=0.0),
        )

    @triton.jit
    def _store3(ptr, base, x, y, z, mask):
        tl.store(ptr + base + 0, x, mask=mask)
        tl.store(ptr + base + 1, y, mask=mask)
        tl.store(ptr + base + 2, z, mask=mask)

    @triton.jit
    def _atomic_add3(ptr, base, x, y, z, mask):
        tl.atomic_add(ptr + base + 0, x, mask=mask)
        tl.atomic_add(ptr + base + 1, y, mask=mask)
        tl.atomic_add(ptr + base + 2, z, mask=mask)

    @triton.jit
    def _unit3(vx, vy, vz, eps):
        inv_r = 1.0 / tl.maximum(tl.sqrt(vx * vx + vy * vy + vz * vz), eps)
        return vx * inv_r, vy * inv_r, vz * inv_r, inv_r

    @triton.jit
    def _proj_unit3(gux, guy, guz, ux, uy, uz, inv_r):
        dot = gux * ux + guy * uy + guz * uz
        return (gux - dot * ux) * inv_r, (guy - dot * uy) * inv_r, (guz - dot * uz) * inv_r

    @triton.jit
    def _cross3(ax, ay, az, bx, by, bz):
        return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx

    @triton.jit
    def _gmm_1d(r, mu_ptr, ev_ptr, lw_ptr, row_base, K, mask, BLOCK_B: tl.constexpr):
        max_lp = tl.full((BLOCK_B,), -1e30, dtype=tl.float32)
        for k in range(K):
            mu = tl.load(mu_ptr + row_base + k, mask=mask, other=0.0)
            ev = tl.load(ev_ptr + row_base + k, mask=mask, other=1.0)
            lw = tl.load(lw_ptr + row_base + k, mask=mask, other=-1e4)
            diff = r - mu
            lp_k = -0.9189385332046727 - 0.5 * tl.log(ev) - 0.5 * diff * diff / ev + lw
            max_lp = tl.maximum(max_lp, lp_k)
        sum_exp = tl.zeros((BLOCK_B,), dtype=tl.float32)
        g_r = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for k in range(K):
            mu = tl.load(mu_ptr + row_base + k, mask=mask, other=0.0)
            ev = tl.load(ev_ptr + row_base + k, mask=mask, other=1.0)
            lw = tl.load(lw_ptr + row_base + k, mask=mask, other=-1e4)
            diff = r - mu
            lp_k = -0.9189385332046727 - 0.5 * tl.log(ev) - 0.5 * diff * diff / ev + lw
            e_k = tl.exp(tl.maximum(lp_k - max_lp, -80.0))
            sum_exp += e_k
            g_r += e_k * (-diff / ev)
        return max_lp + tl.log(sum_exp), g_r / sum_exp

    @triton.jit
    def _vmm_1d(cos_phi, sin_phi, mu_ptr, kap_ptr, const_ptr, row_base, K, BLOCK_B: tl.constexpr):
        max_lp = tl.full((BLOCK_B,), -1e30, dtype=tl.float32)
        for m in range(K):
            mu = tl.load(mu_ptr + row_base + m)
            kap = tl.load(kap_ptr + row_base + m)
            c_m = tl.load(const_ptr + row_base + m)
            lp_m = kap * (cos_phi * tl.cos(mu) + sin_phi * tl.sin(mu)) + c_m
            max_lp = tl.maximum(max_lp, lp_m)
        sum_exp = tl.zeros((BLOCK_B,), dtype=tl.float32)
        g_phi = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for m in range(K):
            mu = tl.load(mu_ptr + row_base + m)
            kap = tl.load(kap_ptr + row_base + m)
            c_m = tl.load(const_ptr + row_base + m)
            cos_m, sin_m = tl.cos(mu), tl.sin(mu)
            cos_d = cos_phi * cos_m + sin_phi * sin_m
            sin_d = sin_phi * cos_m - cos_phi * sin_m
            e_m = tl.exp(tl.maximum(kap * cos_d + c_m - max_lp, -80.0))
            sum_exp += e_m
            g_phi += e_m * (-kap * sin_d)
        return max_lp + tl.log(sum_exp), g_phi / sum_exp

    @triton.jit
    def _q1_q2_q5_fused_kernel(
        X_ptr,            # (B, N, 3) float32
        donor_pos_ptr,    # (B, n_d, 3) float32
        grad_X_ptr,       # (B, N, 3) float32
        grad_donor_ptr,   # (B, n_d, 3) float32 (used when SEPARATE_GRADS=False)
        gx_q1_m_ptr,      # (B, 3) float32 (used when SEPARATE_GRADS=True)
        gd_q1_ptr,        # (B, n_d, 3) float32 (used when SEPARATE_GRADS=True)
        gx_q2_m_ptr,      # (B, 3) float32 (used when SEPARATE_GRADS=True)
        gd_q2_ptr,        # (B, n_d, 3) float32 (used when SEPARATE_GRADS=True)
        row_groups_ptr,   # (B,) int32 or dummy
        has_row_groups: tl.constexpr,
        SEPARATE_GRADS: tl.constexpr,
        # Q1 tables: (C1, n_d, K1)
        q1_mu_ptr,
        q1_ev_ptr,
        q1_lw_ptr,
        q1_whi_ptr,       # (C1, n_d)
        q1_wsd_ptr,       # (C1, n_d)
        C1: int,
        K1: int,
        q1_wall_on: tl.constexpr,
        inv_div_q1: float,
        # Q2 tables: (C2, n_p, K2)
        q2_pair_a_ptr,    # (n_p,) int32
        q2_pair_b_ptr,    # (n_p,) int32
        q2_mu_ptr,
        q2_ev_ptr,
        q2_lw_ptr,
        q2_nu_ptr,        # (C2, n_p)
        q2_const_ptr,     # (C2, n_p, K2)
        C2: int,
        n_p: int,
        K2: int,
        inv_div_q2: float,
        # Q5 tables: (n_q5, K5)
        q5_i_ptr,         # (n_q5,) int32
        q5_j_ptr,         # (n_q5,) int32
        q5_k_ptr,         # (n_q5,) int32
        q5_l_ptr,         # (n_q5,) int32
        q5_mu_ptr,        # (n_q5, K5)
        q5_kap_ptr,       # (n_q5, K5)
        q5_const_ptr,     # (n_q5, K5)
        n_q5: int,
        K5: int,
        inv_div_q5: float,
        # Outputs
        out_q1_ptr,       # (B,)
        out_q2_ptr,       # (B,)
        out_q5_ptr,       # (B,)
        B: int,
        N: int,
        n_d: int,
        metal_idx: int,
        BLOCK_B: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_task = tl.program_id(1)
        b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b_offs < B

        if pid_task == 0:
            gi = tl.zeros((BLOCK_B,), dtype=tl.int32)
            if has_row_groups:
                gi = tl.load(row_groups_ptr + b_offs, mask=b_mask, other=0).to(tl.int32)
            g1 = gi if C1 > 1 else tl.zeros((BLOCK_B,), dtype=tl.int32)
            g2 = gi if C2 > 1 else tl.zeros((BLOCK_B,), dtype=tl.int32)

            base_m = b_offs * (N * 3) + metal_idx * 3
            mx, my, mz = _load3(X_ptr, base_m, b_mask)

            # ── Q1 ────────────────────────────────────────────────────────
            acc_q1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            acc_gMx1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            acc_gMy1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            acc_gMz1 = tl.zeros((BLOCK_B,), dtype=tl.float32)

            if n_d > 0 and K1 > 0 and inv_div_q1 > 0.0:
                for s in range(n_d):
                    base_d = b_offs * (n_d * 3) + s * 3
                    dpx, dpy, dpz = _load3(donor_pos_ptr, base_d, b_mask)
                    dx, dy, dz = dpx - mx, dpy - my, dpz - mz
                    r = tl.sqrt(dx * dx + dy * dy + dz * dz)
                    lp_s, g_r = _gmm_1d(r, q1_mu_ptr, q1_ev_ptr, q1_lw_ptr, (g1 * n_d + s) * K1, K1, b_mask, BLOCK_B)
                    if q1_wall_on:
                        whi = tl.load(q1_whi_ptr + g1 * n_d + s, mask=b_mask, other=10.0)
                        wsd = tl.load(q1_wsd_ptr + g1 * n_d + s, mask=b_mask, other=1.0)
                        ex = tl.maximum(r - whi, 0.0) / wsd
                        lp_s -= 0.5 * ex * ex
                        g_r -= ex / wsd

                    acc_q1 += lp_s * inv_div_q1
                    g_r = tl.where(r > 0.0, g_r * inv_div_q1 / tl.maximum(r, 1e-30), 0.0)
                    gdx, gdy, gdz = g_r * dx, g_r * dy, g_r * dz
                    if SEPARATE_GRADS:
                        _store3(gd_q1_ptr, base_d, gdx, gdy, gdz, b_mask)
                    else:
                        _atomic_add3(grad_donor_ptr, base_d, gdx, gdy, gdz, b_mask)
                    acc_gMx1 -= gdx; acc_gMy1 -= gdy; acc_gMz1 -= gdz

            tl.store(out_q1_ptr + b_offs, acc_q1, mask=b_mask)
            if SEPARATE_GRADS:
                _store3(gx_q1_m_ptr, b_offs * 3, acc_gMx1, acc_gMy1, acc_gMz1, b_mask)
            else:
                _atomic_add3(grad_X_ptr, base_m, acc_gMx1, acc_gMy1, acc_gMz1, b_mask)

            # ── Q2 ────────────────────────────────────────────────────────
            acc_q2 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            acc_gMx2 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            acc_gMy2 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            acc_gMz2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

            if n_p > 0 and K2 > 0 and inv_div_q2 > 0.0:
                for p in range(n_p):
                    ia = tl.load(q2_pair_a_ptr + p)
                    ib = tl.load(q2_pair_b_ptr + p)
                    base_da = b_offs * (n_d * 3) + ia * 3
                    base_db = b_offs * (n_d * 3) + ib * 3
                    ax, ay, az = _load3(donor_pos_ptr, base_da, b_mask)
                    bx, by, bz = _load3(donor_pos_ptr, base_db, b_mask)
                    uax, uay, uaz, inv_ra = _unit3(ax - mx, ay - my, az - mz, 1e-8)
                    ubx, uby, ubz, inv_rb = _unit3(bx - mx, by - my, bz - mz, 1e-8)
                    dot_ab = uax * ubx + uay * uby + uaz * ubz
                    in_bounds = (dot_ab > -0.9999999) & (dot_ab < 0.9999999)
                    c = tl.minimum(tl.maximum(dot_ab, -0.9999999), 0.9999999)

                    nu = tl.load(q2_nu_ptr + g2 * n_p + p, mask=b_mask, other=float("inf"))
                    is_t = nu < 1e10
                    nf = tl.where(is_t, nu, 1.0)
                    row_base = (g2 * n_p + p) * K2

                    max_lp = tl.full((BLOCK_B,), -1e30, dtype=tl.float32)
                    for k in range(K2):
                        mu = tl.load(q2_mu_ptr + row_base + k, mask=b_mask, other=0.0)
                        ev = tl.load(q2_ev_ptr + row_base + k, mask=b_mask, other=1.0)
                        c_k = tl.load(q2_const_ptr + row_base + k, mask=b_mask, other=-1e4)
                        z = (c - mu) * (c - mu) / ev
                        lp_k = tl.where(is_t, c_k - 0.5 * (nf + 1.0) * tl.log(1.0 + z / nf), c_k - 0.5 * z)
                        max_lp = tl.maximum(max_lp, lp_k)

                    sum_exp = tl.zeros((BLOCK_B,), dtype=tl.float32)
                    g_c = tl.zeros((BLOCK_B,), dtype=tl.float32)
                    for k in range(K2):
                        mu = tl.load(q2_mu_ptr + row_base + k, mask=b_mask, other=0.0)
                        ev = tl.load(q2_ev_ptr + row_base + k, mask=b_mask, other=1.0)
                        c_k = tl.load(q2_const_ptr + row_base + k, mask=b_mask, other=-1e4)
                        diff = c - mu
                        z = diff * diff / ev
                        lp_k = tl.where(is_t, c_k - 0.5 * (nf + 1.0) * tl.log(1.0 + z / nf), c_k - 0.5 * z)
                        e_k = tl.exp(tl.maximum(lp_k - max_lp, -80.0))
                        sum_exp += e_k
                        g_c += e_k * tl.where(is_t, -(nf + 1.0) * diff / ((nf + z) * ev), -diff / ev)

                    acc_q2 += (max_lp + tl.log(sum_exp)) * inv_div_q2
                    g_c = tl.where(in_bounds, g_c / sum_exp * inv_div_q2, 0.0)
                    gax, gay, gaz = g_c * (ubx - dot_ab * uax) * inv_ra, g_c * (uby - dot_ab * uay) * inv_ra, g_c * (ubz - dot_ab * uaz) * inv_ra
                    gbx, gby, gbz = g_c * (uax - dot_ab * ubx) * inv_rb, g_c * (uay - dot_ab * uby) * inv_rb, g_c * (uaz - dot_ab * ubz) * inv_rb
                    dst_ptr = gd_q2_ptr if SEPARATE_GRADS else grad_donor_ptr
                    _atomic_add3(dst_ptr, base_da, gax, gay, gaz, b_mask)
                    _atomic_add3(dst_ptr, base_db, gbx, gby, gbz, b_mask)
                    acc_gMx2 -= (gax + gbx); acc_gMy2 -= (gay + gby); acc_gMz2 -= (gaz + gbz)

            tl.store(out_q2_ptr + b_offs, acc_q2, mask=b_mask)
            if SEPARATE_GRADS:
                _store3(gx_q2_m_ptr, b_offs * 3, acc_gMx2, acc_gMy2, acc_gMz2, b_mask)
            else:
                _atomic_add3(grad_X_ptr, base_m, acc_gMx2, acc_gMy2, acc_gMz2, b_mask)

        else:
            # ── Q5 torsion q = pid_task - 1 ───────────────────────────────
            q = pid_task - 1
            base_i = b_offs * (N * 3) + tl.load(q5_i_ptr + q) * 3
            base_j = b_offs * (N * 3) + tl.load(q5_j_ptr + q) * 3
            base_k = b_offs * (N * 3) + tl.load(q5_k_ptr + q) * 3
            base_l = b_offs * (N * 3) + tl.load(q5_l_ptr + q) * 3

            pix, piy, piz = _load3(X_ptr, base_i, b_mask)
            pjx, pjy, pjz = _load3(X_ptr, base_j, b_mask)
            pkx, pky, pkz = _load3(X_ptr, base_k, b_mask)
            plx, ply, plz = _load3(X_ptr, base_l, b_mask)

            b1x, b1y, b1z = pjx - pix, pjy - piy, pjz - piz
            b2x, b2y, b2z = pkx - pjx, pky - pjy, pkz - pjz
            b3x, b3y, b3z = plx - pkx, ply - pky, plz - pkz

            n1x, n1y, n1z = _cross3(b1x, b1y, b1z, b2x, b2y, b2z)
            n2x, n2y, n2z = _cross3(b2x, b2y, b2z, b3x, b3y, b3z)
            r_b2 = tl.maximum(tl.sqrt(b2x * b2x + b2y * b2y + b2z * b2z), 1e-8)
            b2hx, b2hy, b2hz = b2x / r_b2, b2y / r_b2, b2z / r_b2
            mx_c, my_c, mz_c = _cross3(n1x, n1y, n1z, n2x, n2y, n2z)

            y_p = mx_c * b2hx + my_c * b2hy + mz_c * b2hz
            x_p = n1x * n2x + n1y * n2y + n1z * n2z
            r_p = tl.maximum(tl.sqrt(x_p * x_p + y_p * y_p), 1e-20)

            lp_q, g_phi = _vmm_1d(x_p / r_p, y_p / r_p, q5_mu_ptr, q5_kap_ptr, q5_const_ptr, q * K5, K5, BLOCK_B)
            tl.atomic_add(out_q5_ptr + b_offs, lp_q * inv_div_q5, mask=b_mask)
            g_phi = g_phi * inv_div_q5

            n1_sq = tl.maximum(n1x * n1x + n1y * n1y + n1z * n1z, 1e-20)
            n2_sq = tl.maximum(n2x * n2x + n2y * n2y + n2z * n2z, 1e-20)
            c_i = -g_phi * (r_b2 / n1_sq)
            c_l = g_phi * (r_b2 / n2_sq)
            gix, giy, giz = c_i * n1x, c_i * n1y, c_i * n1z
            glx, gly, glz = c_l * n2x, c_l * n2y, c_l * n2z

            b2_sq = r_b2 * r_b2
            dot12 = (b1x * b2x + b1y * b2y + b1z * b2z) / b2_sq
            dot23 = (b2x * b3x + b2y * b3y + b2z * b3z) / b2_sq
            gjx, gjy, gjz = (-1.0 - dot12) * gix + dot23 * glx, (-1.0 - dot12) * giy + dot23 * gly, (-1.0 - dot12) * giz + dot23 * glz
            gkx, gky, gkz = -(gix + gjx + glx), -(giy + gjy + gly), -(giz + gjz + glz)

            _atomic_add3(grad_X_ptr, base_i, gix, giy, giz, b_mask)
            _atomic_add3(grad_X_ptr, base_j, gjx, gjy, gjz, b_mask)
            _atomic_add3(grad_X_ptr, base_k, gkx, gky, gkz, b_mask)
            _atomic_add3(grad_X_ptr, base_l, glx, gly, glz, b_mask)

    @triton.jit
    def _sorient_kappa_kernel(
        X_ptr,           # (B, N, 3) float32
        grad_X_ptr,      # (B, N, 3) float32
        val_ptr,         # (B,) float32
        donor_ptr,       # (n,) int32
        subs_ptr,        # (n, k) int32
        mu_ptr,          # (n, K, dim) float32
        prec_ptr,        # (n, K, dim, dim) float32
        const_ptr,       # (n, K) float32
        B: int,
        N: int,
        n_sites: int,
        K_max: int,
        k_sub: tl.constexpr,
        metal_idx: int,
        inv_norm: float,
        BLOCK_B: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b_offs < B

        base_m = b_offs * (N * 3) + metal_idx * 3
        mx, my, mz = _load3(X_ptr, base_m, b_mask)
        base_d = b_offs * (N * 3) + tl.load(donor_ptr + pid_s) * 3
        dx, dy, dz = _load3(X_ptr, base_d, b_mask)
        uMx, uMy, uMz, inv_rM = _unit3(mx - dx, my - dy, mz - dz, 1e-12)

        if k_sub == 1:
            base_s0 = b_offs * (N * 3) + tl.load(subs_ptr + pid_s) * 3
            s0x, s0y, s0z = _load3(X_ptr, base_s0, b_mask)
            u0x, u0y, u0z, inv_r0 = _unit3(s0x - dx, s0y - dy, s0z - dz, 1e-12)
            f0 = uMx * u0x + uMy * u0y + uMz * u0z

            max_lp = tl.full((BLOCK_B,), -1e30, dtype=tl.float32)
            for m in range(K_max):
                df0 = f0 - tl.load(mu_ptr + pid_s * K_max + m)
                p00 = tl.load(prec_ptr + pid_s * K_max + m)
                max_lp = tl.maximum(max_lp, tl.load(const_ptr + pid_s * K_max + m) - 0.5 * df0 * p00 * df0)

            sum_exp = tl.zeros((BLOCK_B,), dtype=tl.float32)
            gf0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            for m in range(K_max):
                df0 = f0 - tl.load(mu_ptr + pid_s * K_max + m)
                p00 = tl.load(prec_ptr + pid_s * K_max + m)
                e_m = tl.exp(tl.maximum(tl.load(const_ptr + pid_s * K_max + m) - 0.5 * df0 * p00 * df0 - max_lp, -80.0))
                sum_exp += e_m
                gf0 += e_m * (-p00 * df0)

            tl.atomic_add(val_ptr + b_offs, (max_lp + tl.log(sum_exp)) * inv_norm, mask=b_mask)
            gf0 = gf0 / sum_exp * inv_norm
            gMx, gMy, gMz = _proj_unit3(gf0 * u0x, gf0 * u0y, gf0 * u0z, uMx, uMy, uMz, inv_rM)
            gs0x, gs0y, gs0z = _proj_unit3(gf0 * uMx, gf0 * uMy, gf0 * uMz, u0x, u0y, u0z, inv_r0)
            _atomic_add3(grad_X_ptr, base_m, gMx, gMy, gMz, b_mask)
            _atomic_add3(grad_X_ptr, base_s0, gs0x, gs0y, gs0z, b_mask)
            _atomic_add3(grad_X_ptr, base_d, -(gMx + gs0x), -(gMy + gs0y), -(gMz + gs0z), b_mask)

        elif k_sub == 2:
            base_s0 = b_offs * (N * 3) + tl.load(subs_ptr + pid_s * 2 + 0) * 3
            base_s1 = b_offs * (N * 3) + tl.load(subs_ptr + pid_s * 2 + 1) * 3
            s0x, s0y, s0z = _load3(X_ptr, base_s0, b_mask)
            s1x, s1y, s1z = _load3(X_ptr, base_s1, b_mask)
            u0x, u0y, u0z, inv_r0 = _unit3(s0x - dx, s0y - dy, s0z - dz, 1e-12)
            u1x, u1y, u1z, inv_r1 = _unit3(s1x - dx, s1y - dy, s1z - dz, 1e-12)

            c0, c1 = uMx * u0x + uMy * u0y + uMz * u0z, uMx * u1x + uMy * u1y + uMz * u1z
            sw01 = c0 > c1
            f0, f1 = tl.where(sw01, c1, c0), tl.where(sw01, c0, c1)
            w0x, w0y, w0z = tl.where(sw01, u1x, u0x), tl.where(sw01, u1y, u0y), tl.where(sw01, u1z, u0z)
            w1x, w1y, w1z = tl.where(sw01, u0x, u1x), tl.where(sw01, u0y, u1y), tl.where(sw01, u0z, u1z)
            f2 = w0x * w1x + w0y * w1y + w0z * w1z

            max_lp = tl.full((BLOCK_B,), -1e30, dtype=tl.float32)
            for m in range(K_max):
                mb, pb = (pid_s * K_max + m) * 3, (pid_s * K_max + m) * 9
                df0, df1, df2 = f0 - tl.load(mu_ptr + mb + 0), f1 - tl.load(mu_ptr + mb + 1), f2 - tl.load(mu_ptr + mb + 2)
                p00, p01, p02 = tl.load(prec_ptr + pb + 0), tl.load(prec_ptr + pb + 1), tl.load(prec_ptr + pb + 2)
                p11, p12, p22 = tl.load(prec_ptr + pb + 4), tl.load(prec_ptr + pb + 5), tl.load(prec_ptr + pb + 8)
                q0, q1, q2 = p00 * df0 + p01 * df1 + p02 * df2, p01 * df0 + p11 * df1 + p12 * df2, p02 * df0 + p12 * df1 + p22 * df2
                max_lp = tl.maximum(max_lp, tl.load(const_ptr + pid_s * K_max + m) - 0.5 * (df0 * q0 + df1 * q1 + df2 * q2))

            sum_exp = tl.zeros((BLOCK_B,), dtype=tl.float32)
            gf0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            gf1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            gf2 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            for m in range(K_max):
                mb, pb = (pid_s * K_max + m) * 3, (pid_s * K_max + m) * 9
                df0, df1, df2 = f0 - tl.load(mu_ptr + mb + 0), f1 - tl.load(mu_ptr + mb + 1), f2 - tl.load(mu_ptr + mb + 2)
                p00, p01, p02 = tl.load(prec_ptr + pb + 0), tl.load(prec_ptr + pb + 1), tl.load(prec_ptr + pb + 2)
                p11, p12, p22 = tl.load(prec_ptr + pb + 4), tl.load(prec_ptr + pb + 5), tl.load(prec_ptr + pb + 8)
                q0, q1, q2 = p00 * df0 + p01 * df1 + p02 * df2, p01 * df0 + p11 * df1 + p12 * df2, p02 * df0 + p12 * df1 + p22 * df2
                e_m = tl.exp(tl.maximum(tl.load(const_ptr + pid_s * K_max + m) - 0.5 * (df0 * q0 + df1 * q1 + df2 * q2) - max_lp, -80.0))
                sum_exp += e_m
                gf0 -= e_m * q0; gf1 -= e_m * q1; gf2 -= e_m * q2

            tl.atomic_add(val_ptr + b_offs, (max_lp + tl.log(sum_exp)) * inv_norm, mask=b_mask)
            scale = inv_norm / sum_exp
            gf0 *= scale; gf1 *= scale; gf2 *= scale

            gw0x, gw0y, gw0z = gf0 * uMx + gf2 * w1x, gf0 * uMy + gf2 * w1y, gf0 * uMz + gf2 * w1z
            gw1x, gw1y, gw1z = gf1 * uMx + gf2 * w0x, gf1 * uMy + gf2 * w0y, gf1 * uMz + gf2 * w0z
            gu0x, gu0y, gu0z = tl.where(sw01, gw1x, gw0x), tl.where(sw01, gw1y, gw0y), tl.where(sw01, gw1z, gw0z)
            gu1x, gu1y, gu1z = tl.where(sw01, gw0x, gw1x), tl.where(sw01, gw0y, gw1y), tl.where(sw01, gw0z, gw1z)

            gMx, gMy, gMz = _proj_unit3(gf0 * w0x + gf1 * w1x, gf0 * w0y + gf1 * w1y, gf0 * w0z + gf1 * w1z, uMx, uMy, uMz, inv_rM)
            gs0x, gs0y, gs0z = _proj_unit3(gu0x, gu0y, gu0z, u0x, u0y, u0z, inv_r0)
            gs1x, gs1y, gs1z = _proj_unit3(gu1x, gu1y, gu1z, u1x, u1y, u1z, inv_r1)
            _atomic_add3(grad_X_ptr, base_m, gMx, gMy, gMz, b_mask)
            _atomic_add3(grad_X_ptr, base_s0, gs0x, gs0y, gs0z, b_mask)
            _atomic_add3(grad_X_ptr, base_s1, gs1x, gs1y, gs1z, b_mask)
            _atomic_add3(grad_X_ptr, base_d, -(gMx + gs0x + gs1x), -(gMy + gs0y + gs1y), -(gMz + gs0z + gs1z), b_mask)

        elif k_sub == 3:
            base_s0 = b_offs * (N * 3) + tl.load(subs_ptr + pid_s * 3 + 0) * 3
            base_s1 = b_offs * (N * 3) + tl.load(subs_ptr + pid_s * 3 + 1) * 3
            base_s2 = b_offs * (N * 3) + tl.load(subs_ptr + pid_s * 3 + 2) * 3
            s0x, s0y, s0z = _load3(X_ptr, base_s0, b_mask)
            s1x, s1y, s1z = _load3(X_ptr, base_s1, b_mask)
            s2x, s2y, s2z = _load3(X_ptr, base_s2, b_mask)
            u0x, u0y, u0z, inv_r0 = _unit3(s0x - dx, s0y - dy, s0z - dz, 1e-12)
            u1x, u1y, u1z, inv_r1 = _unit3(s1x - dx, s1y - dy, s1z - dz, 1e-12)
            u2x, u2y, u2z, inv_r2 = _unit3(s2x - dx, s2y - dy, s2z - dz, 1e-12)

            c0, c1, c2 = uMx * u0x + uMy * u0y + uMz * u0z, uMx * u1x + uMy * u1y + uMz * u1z, uMx * u2x + uMy * u2y + uMz * u2z
            w0x, w0y, w0z, w1x, w1y, w1z, w2x, w2y, w2z = u0x, u0y, u0z, u1x, u1y, u1z, u2x, u2y, u2z

            sw1 = c0 > c1
            c0, c1 = tl.where(sw1, c1, c0), tl.where(sw1, c0, c1)
            w0x, w0y, w0z, w1x, w1y, w1z = (
                tl.where(sw1, w1x, w0x), tl.where(sw1, w1y, w0y), tl.where(sw1, w1z, w0z),
                tl.where(sw1, w0x, w1x), tl.where(sw1, w0y, w1y), tl.where(sw1, w0z, w1z),
            )
            sw2 = c1 > c2
            c1, c2 = tl.where(sw2, c2, c1), tl.where(sw2, c1, c2)
            w1x, w1y, w1z, w2x, w2y, w2z = (
                tl.where(sw2, w2x, w1x), tl.where(sw2, w2y, w1y), tl.where(sw2, w2z, w1z),
                tl.where(sw2, w1x, w2x), tl.where(sw2, w1y, w2y), tl.where(sw2, w1z, w2z),
            )
            sw3 = c0 > c1
            f0, f1, f2 = tl.where(sw3, c1, c0), tl.where(sw3, c0, c1), c2
            w0x, w0y, w0z, w1x, w1y, w1z = (
                tl.where(sw3, w1x, w0x), tl.where(sw3, w1y, w0y), tl.where(sw3, w1z, w0z),
                tl.where(sw3, w0x, w1x), tl.where(sw3, w0y, w1y), tl.where(sw3, w0z, w1z),
            )
            f3 = w0x * w1x + w0y * w1y + w0z * w1z
            f4 = w0x * w2x + w0y * w2y + w0z * w2z
            f5 = w1x * w2x + w1y * w2y + w1z * w2z

            max_lp = tl.full((BLOCK_B,), -1e30, dtype=tl.float32)
            for m in range(K_max):
                mb, pb = (pid_s * K_max + m) * 6, (pid_s * K_max + m) * 36
                df0, df1, df2 = f0 - tl.load(mu_ptr + mb + 0), f1 - tl.load(mu_ptr + mb + 1), f2 - tl.load(mu_ptr + mb + 2)
                df3, df4, df5 = f3 - tl.load(mu_ptr + mb + 3), f4 - tl.load(mu_ptr + mb + 4), f5 - tl.load(mu_ptr + mb + 5)
                q0 = tl.load(prec_ptr+pb+0)*df0 + tl.load(prec_ptr+pb+1)*df1 + tl.load(prec_ptr+pb+2)*df2 + tl.load(prec_ptr+pb+3)*df3 + tl.load(prec_ptr+pb+4)*df4 + tl.load(prec_ptr+pb+5)*df5
                q1 = tl.load(prec_ptr+pb+6)*df0 + tl.load(prec_ptr+pb+7)*df1 + tl.load(prec_ptr+pb+8)*df2 + tl.load(prec_ptr+pb+9)*df3 + tl.load(prec_ptr+pb+10)*df4 + tl.load(prec_ptr+pb+11)*df5
                q2 = tl.load(prec_ptr+pb+12)*df0 + tl.load(prec_ptr+pb+13)*df1 + tl.load(prec_ptr+pb+14)*df2 + tl.load(prec_ptr+pb+15)*df3 + tl.load(prec_ptr+pb+16)*df4 + tl.load(prec_ptr+pb+17)*df5
                q3 = tl.load(prec_ptr+pb+18)*df0 + tl.load(prec_ptr+pb+19)*df1 + tl.load(prec_ptr+pb+20)*df2 + tl.load(prec_ptr+pb+21)*df3 + tl.load(prec_ptr+pb+22)*df4 + tl.load(prec_ptr+pb+23)*df5
                q4 = tl.load(prec_ptr+pb+24)*df0 + tl.load(prec_ptr+pb+25)*df1 + tl.load(prec_ptr+pb+26)*df2 + tl.load(prec_ptr+pb+27)*df3 + tl.load(prec_ptr+pb+28)*df4 + tl.load(prec_ptr+pb+29)*df5
                q5 = tl.load(prec_ptr+pb+30)*df0 + tl.load(prec_ptr+pb+31)*df1 + tl.load(prec_ptr+pb+32)*df2 + tl.load(prec_ptr+pb+33)*df3 + tl.load(prec_ptr+pb+34)*df4 + tl.load(prec_ptr+pb+35)*df5
                max_lp = tl.maximum(max_lp, tl.load(const_ptr + pid_s * K_max + m) - 0.5 * (df0*q0 + df1*q1 + df2*q2 + df3*q3 + df4*q4 + df5*q5))

            sum_exp = tl.zeros((BLOCK_B,), dtype=tl.float32)
            gf0 = tl.zeros((BLOCK_B,), dtype=tl.float32); gf1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            gf2 = tl.zeros((BLOCK_B,), dtype=tl.float32); gf3 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            gf4 = tl.zeros((BLOCK_B,), dtype=tl.float32); gf5 = tl.zeros((BLOCK_B,), dtype=tl.float32)
            for m in range(K_max):
                mb, pb = (pid_s * K_max + m) * 6, (pid_s * K_max + m) * 36
                df0, df1, df2 = f0 - tl.load(mu_ptr + mb + 0), f1 - tl.load(mu_ptr + mb + 1), f2 - tl.load(mu_ptr + mb + 2)
                df3, df4, df5 = f3 - tl.load(mu_ptr + mb + 3), f4 - tl.load(mu_ptr + mb + 4), f5 - tl.load(mu_ptr + mb + 5)
                q0 = tl.load(prec_ptr+pb+0)*df0 + tl.load(prec_ptr+pb+1)*df1 + tl.load(prec_ptr+pb+2)*df2 + tl.load(prec_ptr+pb+3)*df3 + tl.load(prec_ptr+pb+4)*df4 + tl.load(prec_ptr+pb+5)*df5
                q1 = tl.load(prec_ptr+pb+6)*df0 + tl.load(prec_ptr+pb+7)*df1 + tl.load(prec_ptr+pb+8)*df2 + tl.load(prec_ptr+pb+9)*df3 + tl.load(prec_ptr+pb+10)*df4 + tl.load(prec_ptr+pb+11)*df5
                q2 = tl.load(prec_ptr+pb+12)*df0 + tl.load(prec_ptr+pb+13)*df1 + tl.load(prec_ptr+pb+14)*df2 + tl.load(prec_ptr+pb+15)*df3 + tl.load(prec_ptr+pb+16)*df4 + tl.load(prec_ptr+pb+17)*df5
                q3 = tl.load(prec_ptr+pb+18)*df0 + tl.load(prec_ptr+pb+19)*df1 + tl.load(prec_ptr+pb+20)*df2 + tl.load(prec_ptr+pb+21)*df3 + tl.load(prec_ptr+pb+22)*df4 + tl.load(prec_ptr+pb+23)*df5
                q4 = tl.load(prec_ptr+pb+24)*df0 + tl.load(prec_ptr+pb+25)*df1 + tl.load(prec_ptr+pb+26)*df2 + tl.load(prec_ptr+pb+27)*df3 + tl.load(prec_ptr+pb+28)*df4 + tl.load(prec_ptr+pb+29)*df5
                q5 = tl.load(prec_ptr+pb+30)*df0 + tl.load(prec_ptr+pb+31)*df1 + tl.load(prec_ptr+pb+32)*df2 + tl.load(prec_ptr+pb+33)*df3 + tl.load(prec_ptr+pb+34)*df4 + tl.load(prec_ptr+pb+35)*df5
                e_m = tl.exp(tl.maximum(tl.load(const_ptr + pid_s * K_max + m) - 0.5 * (df0*q0 + df1*q1 + df2*q2 + df3*q3 + df4*q4 + df5*q5) - max_lp, -80.0))
                sum_exp += e_m
                gf0 -= e_m * q0; gf1 -= e_m * q1; gf2 -= e_m * q2; gf3 -= e_m * q3; gf4 -= e_m * q4; gf5 -= e_m * q5

            tl.atomic_add(val_ptr + b_offs, (max_lp + tl.log(sum_exp)) * inv_norm, mask=b_mask)
            scale = inv_norm / sum_exp
            gf0 *= scale; gf1 *= scale; gf2 *= scale; gf3 *= scale; gf4 *= scale; gf5 *= scale

            guMx, guMy, guMz = gf0 * w0x + gf1 * w1x + gf2 * w2x, gf0 * w0y + gf1 * w1y + gf2 * w2y, gf0 * w0z + gf1 * w1z + gf2 * w2z
            gw0x, gw0y, gw0z = gf0 * uMx + gf3 * w1x + gf4 * w2x, gf0 * uMy + gf3 * w1y + gf4 * w2y, gf0 * uMz + gf3 * w1z + gf4 * w2z
            gw1x, gw1y, gw1z = gf1 * uMx + gf3 * w0x + gf5 * w2x, gf1 * uMy + gf3 * w0y + gf5 * w2y, gf1 * uMz + gf3 * w0z + gf5 * w2z
            gw2x, gw2y, gw2z = gf2 * uMx + gf4 * w0x + gf5 * w1x, gf2 * uMy + gf4 * w0y + gf5 * w1y, gf2 * uMz + gf4 * w0z + gf5 * w1z

            gw0x, gw0y, gw0z, gw1x, gw1y, gw1z = (
                tl.where(sw3, gw1x, gw0x), tl.where(sw3, gw1y, gw0y), tl.where(sw3, gw1z, gw0z),
                tl.where(sw3, gw0x, gw1x), tl.where(sw3, gw0y, gw1y), tl.where(sw3, gw0z, gw1z),
            )
            gw1x, gw1y, gw1z, gw2x, gw2y, gw2z = (
                tl.where(sw2, gw2x, gw1x), tl.where(sw2, gw2y, gw1y), tl.where(sw2, gw2z, gw1z),
                tl.where(sw2, gw1x, gw2x), tl.where(sw2, gw1y, gw2y), tl.where(sw2, gw1z, gw2z),
            )
            gu0x, gu0y, gu0z = tl.where(sw1, gw1x, gw0x), tl.where(sw1, gw1y, gw0y), tl.where(sw1, gw1z, gw0z)
            gu1x, gu1y, gu1z = tl.where(sw1, gw0x, gw1x), tl.where(sw1, gw0y, gw1y), tl.where(sw1, gw0z, gw1z)

            gMx, gMy, gMz = _proj_unit3(guMx, guMy, guMz, uMx, uMy, uMz, inv_rM)
            gs0x, gs0y, gs0z = _proj_unit3(gu0x, gu0y, gu0z, u0x, u0y, u0z, inv_r0)
            gs1x, gs1y, gs1z = _proj_unit3(gu1x, gu1y, gu1z, u1x, u1y, u1z, inv_r1)
            gs2x, gs2y, gs2z = _proj_unit3(gw2x, gw2y, gw2z, u2x, u2y, u2z, inv_r2)

            _atomic_add3(grad_X_ptr, base_m, gMx, gMy, gMz, b_mask)
            _atomic_add3(grad_X_ptr, base_s0, gs0x, gs0y, gs0z, b_mask)
            _atomic_add3(grad_X_ptr, base_s1, gs1x, gs1y, gs1z, b_mask)
            _atomic_add3(grad_X_ptr, base_s2, gs2x, gs2y, gs2z, b_mask)
            _atomic_add3(grad_X_ptr, base_d, -(gMx + gs0x + gs1x + gs2x), -(gMy + gs0y + gs1y + gs2y), -(gMz + gs0z + gs1z + gs2z), b_mask)

    @triton.jit
    def _sorient_haptic_kernel(
        X_ptr,           # (B, N, 3) float32
        grad_X_ptr,      # (B, N, 3) float32
        val_ptr,         # (B,) float32
        face_ptr,        # (n, F) int32
        fmask_ptr,       # (n, F) float32
        nxt_ptr,         # (n, F) int32
        emask_ptr,       # (n, F) float32
        sa_ptr,          # (n, S) int32
        sx_ptr,          # (n, S) int32
        smask_ptr,       # (n, S) float32
        eta2_ptr,        # (n,) float32
        pa_ptr,          # (n, Pm) int32
        px_ptr,          # (n, Pm) int32
        pm_ptr,          # (n, Pm) float32
        pon_ptr,         # (n,) float32
        mu_ptr,          # (n, K, dim) float32
        prec_ptr,        # (n, K, dim, dim) float32
        const_ptr,       # (n, K) float32
        w_ptr,           # (n,) float32
        B: int,
        N: int,
        n_rows: int,
        F_max: int,
        S_max: int,
        Pm_max: int,
        K_max: int,
        dim: tl.constexpr,
        sel0: tl.constexpr,
        sel1: tl.constexpr,
        sel2: tl.constexpr,
        has_plane: tl.constexpr,
        metal_idx: int,
        inv_norm: float,
        BLOCK_B: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_r = tl.program_id(1)
        b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b_offs < B

        base_m = b_offs * (N * 3) + metal_idx * 3
        mx, my, mz = _load3(X_ptr, base_m, b_mask)

        cx = tl.zeros((BLOCK_B,), dtype=tl.float32)
        cy = tl.zeros((BLOCK_B,), dtype=tl.float32)
        cz = tl.zeros((BLOCK_B,), dtype=tl.float32)
        d_sum = tl.zeros((BLOCK_B,), dtype=tl.float32)
        nf = 0.0
        for j in range(F_max):
            fm = tl.load(fmask_ptr + pid_r * F_max + j)
            px, py, pz = _load3(X_ptr, b_offs * (N * 3) + tl.load(face_ptr + pid_r * F_max + j) * 3, b_mask)
            cx += px * fm; cy += py * fm; cz += pz * fm
            dx, dy, dz = px - mx, py - my, pz - mz
            d_sum += tl.sqrt(dx * dx + dy * dy + dz * dz) * fm
            nf += fm
        inv_nf = 1.0 / tl.maximum(nf, 1.0)
        cx *= inv_nf; cy *= inv_nf; cz *= inv_nf
        dmean = d_sum * inv_nf

        nx_new = tl.zeros((BLOCK_B,), dtype=tl.float32)
        ny_new = tl.zeros((BLOCK_B,), dtype=tl.float32)
        nz_new = tl.zeros((BLOCK_B,), dtype=tl.float32)
        var_sum = tl.zeros((BLOCK_B,), dtype=tl.float32)
        p0x = tl.zeros((BLOCK_B,), dtype=tl.float32); p0y = tl.zeros((BLOCK_B,), dtype=tl.float32); p0z = tl.zeros((BLOCK_B,), dtype=tl.float32)
        p1x = tl.zeros((BLOCK_B,), dtype=tl.float32); p1y = tl.zeros((BLOCK_B,), dtype=tl.float32); p1z = tl.zeros((BLOCK_B,), dtype=tl.float32)

        for j in range(F_max):
            fm = tl.load(fmask_ptr + pid_r * F_max + j)
            em = tl.load(emask_ptr + pid_r * F_max + j)
            fj = tl.load(face_ptr + pid_r * F_max + j)
            fn = tl.load(face_ptr + pid_r * F_max + tl.load(nxt_ptr + pid_r * F_max + j))
            px, py, pz = _load3(X_ptr, b_offs * (N * 3) + fj * 3, b_mask)
            if j == 0:
                p0x, p0y, p0z = px, py, pz
            if j == 1:
                p1x, p1y, p1z = px, py, pz
            pnx, pny, pnz = _load3(X_ptr, b_offs * (N * 3) + fn * 3, b_mask)
            cxj, cyj, czj = _cross3(px - cx, py - cy, pz - cz, pnx - cx, pny - cy, pnz - cz)
            nx_new += cxj * em; ny_new += cyj * em; nz_new += czj * em
            dx, dy, dz = px - mx, py - my, pz - mz
            dd = tl.sqrt(dx * dx + dy * dy + dz * dz) - dmean
            var_sum += dd * dd * fm

        sd_d = tl.sqrt(var_sum * inv_nf + 1e-12)
        slip = sd_d / dmean
        uMx, uMy, uMz, inv_rM = _unit3(mx - cx, my - cy, mz - cz, 1e-12)
        bx, by, bz, inv_rb = _unit3(p1x - p0x, p1y - p0y, p1z - p0z, 1e-12)

        dot_uMb = uMx * bx + uMy * by + uMz * bz
        n2_mx, n2_my, n2_mz = uMx - dot_uMb * bx, uMy - dot_uMb * by, uMz - dot_uMb * bz

        e2 = tl.load(eta2_ptr + pid_r)
        pon = tl.load(pon_ptr + pid_r)

        use_ex = tl.abs(bx) < 0.9
        e1x, e1y, e1z, inv_rc1 = _unit3(tl.where(use_ex, 0.0, -bz), tl.where(use_ex, bz, 0.0), tl.where(use_ex, -by, bx), 1e-12)
        e2x, e2y, e2z = _cross3(bx, by, bz, e1x, e1y, e1z)

        sxy = tl.zeros((BLOCK_B,), dtype=tl.float32)
        sxx_yy = tl.zeros((BLOCK_B,), dtype=tl.float32)
        if has_plane:
            for j in range(Pm_max):
                pm_j = tl.load(pm_ptr + pid_r * Pm_max + j)
                pax, pay, paz = _load3(X_ptr, b_offs * (N * 3) + tl.load(pa_ptr + pid_r * Pm_max + j) * 3, b_mask)
                pxx, pxy, pxz = _load3(X_ptr, b_offs * (N * 3) + tl.load(px_ptr + pid_r * Pm_max + j) * 3, b_mask)
                ux, uy, uz, _ = _unit3(pxx - pax, pxy - pay, pxz - paz, 1e-12)
                xj, yj = (ux * e1x + uy * e1y + uz * e1z) * pm_j, (ux * e2x + uy * e2y + uz * e2z) * pm_j
                sxy += xj * yj
                sxx_yy += xj * xj - yj * yj

        y_at, x_at = 2.0 * sxy * pon, sxx_yy * pon + (1.0 - pon)
        th = 0.5 * tl.extra.cuda.libdevice.atan2(y_at, x_at)
        cos_th, sin_th = tl.cos(th), tl.sin(th)
        nsx, nsy, nsz = cos_th * e2x - sin_th * e1x, cos_th * e2y - sin_th * e1y, cos_th * e2z - sin_th * e1z
        n2x, n2y, n2z = pon * nsx + (1.0 - pon) * n2_mx, pon * nsy + (1.0 - pon) * n2_my, pon * nsz + (1.0 - pon) * n2_mz

        n0x, n0y, n0z, inv_rnr = _unit3(e2 * n2x + (1.0 - e2) * nx_new, e2 * n2y + (1.0 - e2) * ny_new, e2 * n2z + (1.0 - e2) * nz_new, 1e-12)
        tilt0 = n0x * uMx + n0y * uMy + n0z * uMz
        sgn = tl.where(tilt0 < 0.0, -1.0, 1.0)
        nrmx, nrmy, nrmz, tilt = n0x * sgn, n0y * sgn, n0z * sgn, tilt0 * sgn

        bend_sum = tl.zeros((BLOCK_B,), dtype=tl.float32)
        ns_cnt = 0.0
        for j in range(S_max):
            sm = tl.load(smask_ptr + pid_r * S_max + j)
            sax, say, saz = _load3(X_ptr, b_offs * (N * 3) + tl.load(sa_ptr + pid_r * S_max + j) * 3, b_mask)
            sxx, sxy_, sxz = _load3(X_ptr, b_offs * (N * 3) + tl.load(sx_ptr + pid_r * S_max + j) * 3, b_mask)
            ubx, uby, ubz, _ = _unit3(sxx - sax, sxy_ - say, sxz - saz, 1e-12)
            bend_sum += (ubx * nrmx + uby * nrmy + ubz * nrmz) * sm
            ns_cnt += sm
        inv_ns = 1.0 / tl.maximum(ns_cnt, 1.0)
        bend = bend_sum * inv_ns

        f0 = tilt if sel0 == 0 else (bend if sel0 == 1 else slip)
        f1 = tilt if sel1 == 0 else (bend if sel1 == 1 else slip)
        f2 = tilt if sel2 == 0 else (bend if sel2 == 1 else slip)

        w_row = tl.load(w_ptr + pid_r) * inv_norm

        max_lp = tl.full((BLOCK_B,), -1e30, dtype=tl.float32)
        for m in range(K_max):
            mb = (pid_r * K_max + m) * dim
            pb = (pid_r * K_max + m) * (dim * dim)
            if dim == 2:
                df0 = f0 - tl.load(mu_ptr + mb + 0)
                df1 = f1 - tl.load(mu_ptr + mb + 1)
                p00, p01, p11 = tl.load(prec_ptr + pb + 0), tl.load(prec_ptr + pb + 1), tl.load(prec_ptr + pb + 3)
                q0 = p00 * df0 + p01 * df1
                q1 = p01 * df0 + p11 * df1
                maha = df0 * q0 + df1 * q1
            else:
                df0 = f0 - tl.load(mu_ptr + mb + 0)
                df1 = f1 - tl.load(mu_ptr + mb + 1)
                df2 = f2 - tl.load(mu_ptr + mb + 2)
                p00, p01, p02 = tl.load(prec_ptr + pb + 0), tl.load(prec_ptr + pb + 1), tl.load(prec_ptr + pb + 2)
                p11, p12, p22 = tl.load(prec_ptr + pb + 4), tl.load(prec_ptr + pb + 5), tl.load(prec_ptr + pb + 8)
                q0 = p00 * df0 + p01 * df1 + p02 * df2
                q1 = p01 * df0 + p11 * df1 + p12 * df2
                q2 = p02 * df0 + p12 * df1 + p22 * df2
                maha = df0 * q0 + df1 * q1 + df2 * q2
            lp_m = tl.load(const_ptr + pid_r * K_max + m) - 0.5 * maha
            max_lp = tl.maximum(max_lp, lp_m)

        sum_exp = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gf0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gf1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gf2 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for m in range(K_max):
            mb = (pid_r * K_max + m) * dim
            pb = (pid_r * K_max + m) * (dim * dim)
            if dim == 2:
                df0 = f0 - tl.load(mu_ptr + mb + 0)
                df1 = f1 - tl.load(mu_ptr + mb + 1)
                p00, p01, p11 = tl.load(prec_ptr + pb + 0), tl.load(prec_ptr + pb + 1), tl.load(prec_ptr + pb + 3)
                q0 = p00 * df0 + p01 * df1
                q1 = p01 * df0 + p11 * df1
                q2 = tl.zeros((BLOCK_B,), dtype=tl.float32)
                maha = df0 * q0 + df1 * q1
            else:
                df0 = f0 - tl.load(mu_ptr + mb + 0)
                df1 = f1 - tl.load(mu_ptr + mb + 1)
                df2 = f2 - tl.load(mu_ptr + mb + 2)
                p00, p01, p02 = tl.load(prec_ptr + pb + 0), tl.load(prec_ptr + pb + 1), tl.load(prec_ptr + pb + 2)
                p11, p12, p22 = tl.load(prec_ptr + pb + 4), tl.load(prec_ptr + pb + 5), tl.load(prec_ptr + pb + 8)
                q0 = p00 * df0 + p01 * df1 + p02 * df2
                q1 = p01 * df0 + p11 * df1 + p12 * df2
                q2 = p02 * df0 + p12 * df1 + p22 * df2
                maha = df0 * q0 + df1 * q1 + df2 * q2
            lp_m = tl.load(const_ptr + pid_r * K_max + m) - 0.5 * maha
            e_m = tl.exp(tl.maximum(lp_m - max_lp, -80.0))
            sum_exp += e_m
            gf0 -= e_m * q0
            gf1 -= e_m * q1
            gf2 -= e_m * q2

        lp = (max_lp + tl.log(sum_exp)) * w_row
        tl.atomic_add(val_ptr + b_offs, lp, mask=b_mask)
        scale = w_row / sum_exp
        gf0 *= scale; gf1 *= scale; gf2 *= scale

        g_tilt = (gf0 if sel0 == 0 else 0.0) + (gf1 if sel1 == 0 else 0.0) + (gf2 if (dim == 3 and sel2 == 0) else 0.0)
        g_bend = (gf0 if sel0 == 1 else 0.0) + (gf1 if sel1 == 1 else 0.0) + (gf2 if (dim == 3 and sel2 == 1) else 0.0)
        g_slip = (gf0 if sel0 == 2 else 0.0) + (gf1 if sel1 == 2 else 0.0) + (gf2 if (dim == 3 and sel2 == 2) else 0.0)

        gnrmx = g_tilt * uMx
        gnrmy = g_tilt * uMy
        gnrmz = g_tilt * uMz
        guMx = g_tilt * nrmx
        guMy = g_tilt * nrmy
        guMz = g_tilt * nrmz

        for j in range(S_max):
            sm = tl.load(smask_ptr + pid_r * S_max + j)
            if sm > 0.0:
                base_sa = b_offs * (N * 3) + tl.load(sa_ptr + pid_r * S_max + j) * 3
                base_sx = b_offs * (N * 3) + tl.load(sx_ptr + pid_r * S_max + j) * 3
                sax, say, saz = _load3(X_ptr, base_sa, b_mask)
                sxx, sxy_, sxz = _load3(X_ptr, base_sx, b_mask)
                ubx, uby, ubz, inv_rub = _unit3(sxx - sax, sxy_ - say, sxz - saz, 1e-12)
                w_sub = g_bend * sm * inv_ns
                gnrmx += w_sub * ubx; gnrmy += w_sub * uby; gnrmz += w_sub * ubz
                gvx, gvy, gvz = _proj_unit3(w_sub * nrmx, w_sub * nrmy, w_sub * nrmz, ubx, uby, ubz, inv_rub)
                _atomic_add3(grad_X_ptr, base_sx, gvx, gvy, gvz, b_mask)
                _atomic_add3(grad_X_ptr, base_sa, -gvx, -gvy, -gvz, b_mask)

        gnrvx, gnrvy, gnrvz = _proj_unit3(gnrmx * sgn, gnrmy * sgn, gnrmz * sgn, n0x, n0y, n0z, inv_rnr)
        gn2x, gn2y, gn2z = e2 * gnrvx, e2 * gnrvy, e2 * gnrvz
        gnnewx, gnnewy, gnnewz = (1.0 - e2) * gnrvx, (1.0 - e2) * gnrvy, (1.0 - e2) * gnrvz
        gn2mx, gn2my, gn2mz = (1.0 - pon) * gn2x, (1.0 - pon) * gn2y, (1.0 - pon) * gn2z
        gnsx, gnsy, gnsz = pon * gn2x, pon * gn2y, pon * gn2z

        dot_gn2m_b = gn2mx * bx + gn2my * by + gn2mz * bz
        guMx += gn2mx - dot_gn2m_b * bx; guMy += gn2my - dot_gn2m_b * by; guMz += gn2mz - dot_gn2m_b * bz
        gbx, gby, gbz = -dot_uMb * gn2mx - dot_gn2m_b * uMx, -dot_uMb * gn2my - dot_gn2m_b * uMy, -dot_uMb * gn2mz - dot_gn2m_b * uMz

        if has_plane:
            g_th = gnsx * (-sin_th * e2x - cos_th * e1x) + gnsy * (-sin_th * e2y - cos_th * e1y) + gnsz * (-sin_th * e2z - cos_th * e1z)
            ge1x, ge1y, ge1z = -sin_th * gnsx, -sin_th * gnsy, -sin_th * gnsz
            ge2x, ge2y, ge2z = cos_th * gnsx, cos_th * gnsy, cos_th * gnsz
            denom_at = tl.maximum(y_at * y_at + x_at * x_at, 1e-30)
            g_sxy = 2.0 * (0.5 * g_th * x_at / denom_at) * pon
            g_sxx_yy = (-0.5 * g_th * y_at / denom_at) * pon

            for j in range(Pm_max):
                pm_j = tl.load(pm_ptr + pid_r * Pm_max + j)
                if pm_j > 0.0:
                    base_pa = b_offs * (N * 3) + tl.load(pa_ptr + pid_r * Pm_max + j) * 3
                    base_px = b_offs * (N * 3) + tl.load(px_ptr + pid_r * Pm_max + j) * 3
                    pax, pay, paz = _load3(X_ptr, base_pa, b_mask)
                    pxx, pxy, pxz = _load3(X_ptr, base_px, b_mask)
                    ux, uy, uz, inv_rv = _unit3(pxx - pax, pxy - pay, pxz - paz, 1e-12)
                    dot_ub = ux * bx + uy * by + uz * bz
                    upx, upy, upz = (ux - dot_ub * bx) * pm_j, (uy - dot_ub * by) * pm_j, (uz - dot_ub * bz) * pm_j
                    xj, yj = upx * e1x + upy * e1y + upz * e1z, upx * e2x + upy * e2y + upz * e2z
                    gxj, gyj = g_sxy * yj + 2.0 * g_sxx_yy * xj, g_sxy * xj - 2.0 * g_sxx_yy * yj

                    ge1x += gxj * upx; ge1y += gxj * upy; ge1z += gxj * upz
                    ge2x += gyj * upx; ge2y += gyj * upy; ge2z += gyj * upz
                    gupx, gupy, gupz = (gxj * e1x + gyj * e2x) * pm_j, (gxj * e1y + gyj * e2y) * pm_j, (gxj * e1z + gyj * e2z) * pm_j
                    dot_gup_b = gupx * bx + gupy * by + gupz * bz
                    gbx -= dot_ub * gupx + dot_gup_b * ux; gby -= dot_ub * gupy + dot_gup_b * uy; gbz -= dot_ub * gupz + dot_gup_b * uz
                    gvx, gvy, gvz = _proj_unit3(gupx - dot_gup_b * bx, gupy - dot_gup_b * by, gupz - dot_gup_b * bz, ux, uy, uz, inv_rv)
                    _atomic_add3(grad_X_ptr, base_px, gvx, gvy, gvz, b_mask)
                    _atomic_add3(grad_X_ptr, base_pa, -gvx, -gvy, -gvz, b_mask)

            cge2b_x, cge2b_y, cge2b_z = _cross3(ge2x, ge2y, ge2z, bx, by, bz)
            ce1ge2_x, ce1ge2_y, ce1ge2_z = _cross3(e1x, e1y, e1z, ge2x, ge2y, ge2z)
            ge1x += cge2b_x; ge1y += cge2b_y; ge1z += cge2b_z
            gbx += ce1ge2_x; gby += ce1ge2_y; gbz += ce1ge2_z
            gc1x, gc1y, gc1z = _proj_unit3(ge1x, ge1y, ge1z, e1x, e1y, e1z, inv_rc1)
            gbx += tl.where(use_ex, 0.0, gc1z); gby += tl.where(use_ex, -gc1z, 0.0); gbz += tl.where(use_ex, gc1y, -gc1x)

        gvbx, gvby, gvbz = _proj_unit3(gbx, gby, gbz, bx, by, bz, inv_rb)
        base_f0 = b_offs * (N * 3) + tl.load(face_ptr + pid_r * F_max + 0) * 3
        base_f1 = b_offs * (N * 3) + tl.load(face_ptr + pid_r * F_max + 1) * 3
        _atomic_add3(grad_X_ptr, base_f1, gvbx, gvby, gvbz, b_mask)
        _atomic_add3(grad_X_ptr, base_f0, -gvbx, -gvby, -gvbz, b_mask)

        gvMx, gvMy, gvMz = _proj_unit3(guMx, guMy, guMz, uMx, uMy, uMz, inv_rM)
        gMx_acc, gMy_acc, gMz_acc = gvMx, gvMy, gvMz
        gcx, gcy, gcz = -gvMx, -gvMy, -gvMz

        for j in range(F_max):
            fm = tl.load(fmask_ptr + pid_r * F_max + j)
            em = tl.load(emask_ptr + pid_r * F_max + j)
            fj = tl.load(face_ptr + pid_r * F_max + j)
            fn = tl.load(face_ptr + pid_r * F_max + tl.load(nxt_ptr + pid_r * F_max + j))
            base_fj, base_fn = b_offs * (N * 3) + fj * 3, b_offs * (N * 3) + fn * 3
            px, py, pz = _load3(X_ptr, base_fj, b_mask)
            pnx, pny, pnz = _load3(X_ptr, base_fn, b_mask)

            if em > 0.0:
                gqx, gqy, gqz = _cross3(pnx - cx, pny - cy, pnz - cz, gnnewx, gnnewy, gnnewz)
                gqnx, gqny, gqnz = _cross3(gnnewx, gnnewy, gnnewz, px - cx, py - cy, pz - cz)
                gqx *= em; gqy *= em; gqz *= em; gqnx *= em; gqny *= em; gqnz *= em
                _atomic_add3(grad_X_ptr, base_fj, gqx, gqy, gqz, b_mask)
                _atomic_add3(grad_X_ptr, base_fn, gqnx, gqny, gqnz, b_mask)
                gcx -= gqx + gqnx; gcy -= gqy + gqny; gcz -= gqz + gqnz

            if fm > 0.0:
                dx, dy, dz = px - mx, py - my, pz - mz
                dj = tl.maximum(tl.sqrt(dx * dx + dy * dy + dz * dz), 1e-12)
                gdj = g_slip * fm * inv_nf * ((dj - dmean) / (sd_d * dmean) - slip / dmean) / dj
                gdx, gdy, gdz = gdj * dx, gdj * dy, gdj * dz
                _atomic_add3(grad_X_ptr, base_fj, gdx, gdy, gdz, b_mask)
                gMx_acc -= gdx; gMy_acc -= gdy; gMz_acc -= gdz

        for j in range(F_max):
            fm = tl.load(fmask_ptr + pid_r * F_max + j)
            if fm > 0.0:
                base_fj = b_offs * (N * 3) + tl.load(face_ptr + pid_r * F_max + j) * 3
                w_c = fm * inv_nf
                _atomic_add3(grad_X_ptr, base_fj, gcx * w_c, gcy * w_c, gcz * w_c, b_mask)

        _atomic_add3(grad_X_ptr, base_m, gMx_acc, gMy_acc, gMz_acc, b_mask)

