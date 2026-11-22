"""Kabsch alignment and rigid-body rotation utilities for inverse kinematics (CuPy + Triton)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as _libdevice

from jacobik.core import gpu
from jacobik.core.env import EPS
from jacobik.core.gpu import cp

if TYPE_CHECKING:
    from jacobik.ik.geometry import GeometryExtractor

_EPS = EPS


def _axisangle_to_matrix_cp(omega: cp.ndarray) -> cp.ndarray:
    """Convert (B, 3) axis-angle vectors to (B, 3, 3) rotation matrices in CuPy."""
    angle_sq = cp.maximum((omega * omega).sum(axis=-1, keepdims=True), _EPS * _EPS)
    angle = cp.sqrt(angle_sq)
    safe = angle.squeeze(-1) >= _EPS
    k = cp.where(safe[:, None], omega / cp.maximum(angle, _EPS), omega)

    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    z = cp.zeros_like(kx)
    K = cp.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], axis=-1).reshape(-1, 3, 3)
    I = cp.eye(3, dtype=omega.dtype)[None]
    cos_a = cp.cos(angle)[..., None]
    sin_a = cp.sin(angle)[..., None]
    kk = k[:, :, None] * k[:, None, :]
    R_full = cos_a * I + sin_a * K + (1.0 - cos_a) * kk

    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    K_w = cp.stack([z, -wz, wy, wz, z, -wx, -wy, wx, z], axis=-1).reshape(-1, 3, 3)
    return cp.where(safe[:, None, None], R_full, I + K_w)


def _ref_rows(X_ref: cp.ndarray, B: int) -> cp.ndarray:
    """Fresh (B, N, 3) copy of a shared (N, 3) or per-row (B, N, 3) reference."""
    if X_ref.ndim == 3:
        return X_ref.copy()
    return cp.broadcast_to(X_ref[None], (B, X_ref.shape[0], 3)).copy()


def kabsch_init(
    ge: "GeometryExtractor",
    X_ref: cp.ndarray,
    donor_targets: cp.ndarray,
    donor_pos_idx: Optional[cp.ndarray] = None,
    diversify_orientation: bool = False,
    diversity_scale: float = 1.0,
) -> cp.ndarray:
    """Initialise base DOF (translation + axis-angle) per batch item by Kabsch alignment."""
    Q = gpu.as_gpu(donor_targets, cp.float32)
    B, n = Q.shape[0], Q.shape[1]
    Xr = gpu.as_gpu(X_ref, cp.float32)
    Xb = Xr if Xr.ndim == 3 else cp.broadcast_to(Xr[None], (B, Xr.shape[0], 3))
    metal_pos = Xb[:, ge.metal_pos_idx]
    dpos = ge.donor_positions(Xb)
    P = dpos[:, gpu.as_gpu(donor_pos_idx, cp.int64), :] if donor_pos_idx is not None else dpos
    eye3 = cp.eye(3, dtype=P.dtype)

    Pm, Qm = P.mean(axis=1), Q.mean(axis=1)
    Pc, Qc = P - Pm[:, None, :], Q - Qm[:, None, :]
    if n == 1:
        R = cp.broadcast_to(eye3[None], (B, 3, 3)).copy()
    else:
        H = cp.einsum("bni,bnj->bij", Pc, Qc)
        U, _, Vt = cp.linalg.svd(H)
        V = Vt.transpose(0, 2, 1)
        d = cp.linalg.det(V @ U.transpose(0, 2, 1))
        sign_fix = cp.ones((B, 3), dtype=P.dtype)
        sign_fix[:, 2] = cp.sign(d)
        R = (V * sign_fix[:, None, :]) @ U.transpose(0, 2, 1)

    rank = (cp.linalg.eigvalsh(Pc.transpose(0, 2, 1) @ Pc) > 1e-5).sum(axis=-1)
    _, Qe = cp.linalg.eigh(Qc.transpose(0, 2, 1) @ Qc)
    ax0 = Qm - metal_pos
    ax1 = Qe[:, :, -1]
    ax2 = cp.random.randn(B, 3, dtype=P.dtype)
    axis = cp.where((rank == 0)[:, None], ax0, cp.where((rank == 1)[:, None], ax1, ax2))
    axis = axis / cp.maximum(cp.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    u = cp.random.rand(B, dtype=P.dtype)
    angle = cp.where(rank <= 1, u * 2.0 * math.pi, u * diversity_scale * math.pi)
    if not diversify_orientation:
        angle = cp.where(rank <= 1, angle, cp.zeros_like(angle))
    degenerate = (rank == 0) & (cp.linalg.norm(ax0, axis=-1) <= 1e-5)
    angle = cp.where(degenerate, cp.zeros_like(angle), angle)
    R = _axisangle_to_matrix_cp(axis * angle[:, None]) @ R
    t = Qm - Pm

    base_theta = cp.zeros((B, 6), dtype=cp.float32)
    base_theta[:, 0:3] = t
    base_theta[:, 3:6] = _matrix_to_axisangle_batched(R)
    return base_theta


def compensate_fk_pivot(base: cp.ndarray, fk, original: cp.ndarray, seeded: cp.ndarray, ge, donor_indices) -> cp.ndarray:
    """Transport a seeded donor fit to FK's original donor-atom rotation pivot."""
    B = len(base)
    X = original if original.ndim == 3 else cp.broadcast_to(original[None], (B, original.shape[0], 3))
    c = X[:, fk._donor_pos_t].mean(axis=1)
    p = ge.donor_positions(seeded)[:, donor_indices].mean(axis=1)
    delta = p - c
    rotation = _axisangle_to_matrix_cp(base[:, 3:6])
    result = base.copy()
    result[:, :3] += delta - cp.einsum("bij,bj->bi", rotation, delta)
    return result


def _matrix_to_axisangle_batched(R: cp.ndarray) -> cp.ndarray:
    """Batched matrix-to-axisangle: (B, 3, 3) -> (B, 3)."""
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    theta = cp.arccos(cp.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    axis_g = cp.stack(
        [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=-1
    ) / (2.0 * cp.maximum(cp.sin(theta), _EPS))[:, None]
    axis_g = axis_g / cp.maximum(cp.linalg.norm(axis_g, axis=-1, keepdims=True), 1e-12)

    diag = cp.diagonal((R + cp.eye(3, dtype=R.dtype)) / 2.0, axis1=-2, axis2=-1)
    axis_pi = cp.sqrt(cp.maximum(diag, 0.0))
    sgn = cp.ones_like(axis_pi)
    sgn[:, 1] = cp.where(R[:, 0, 1] < 0, -1.0, 1.0)
    sgn[:, 2] = cp.where(R[:, 0, 2] < 0, -1.0, 1.0)
    axis_pi = axis_pi * sgn
    axis_pi = axis_pi / cp.maximum(cp.linalg.norm(axis_pi, axis=-1, keepdims=True), 1e-12)

    near_pi = (math.pi - theta) < 1e-4
    axis = cp.where(near_pi[:, None], axis_pi, axis_g)
    omega = axis * theta[:, None]
    return cp.where((cp.abs(theta) < _EPS)[:, None], cp.zeros_like(omega), omega)


@triton.jit
def _renorm_axisangle_k(w_ptr, out_ptr, B, PI: tl.constexpr, EPS: tl.constexpr, BLOCK_B: tl.constexpr):
    b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = b < B
    base = b * 3
    x = tl.load(w_ptr + base + 0, mask=mask, other=0.0)
    y = tl.load(w_ptr + base + 1, mask=mask, other=0.0)
    z = tl.load(w_ptr + base + 2, mask=mask, other=0.0)
    nrm = tl.sqrt(x * x + y * y + z * z)
    scale = tl.where(nrm > PI, PI / tl.maximum(nrm, EPS), 1.0)
    tl.store(out_ptr + base + 0, x * scale, mask=mask)
    tl.store(out_ptr + base + 1, y * scale, mask=mask)
    tl.store(out_ptr + base + 2, z * scale, mask=mask)


@triton.jit
def _jacobi_step4(
    app, aqq, apq, apr, aps, aqr, aqs,
    vp0, vp1, vp2, vp3, vq0, vq1, vq2, vq3,
):
    phi = 0.5 * _libdevice.atan2(2.0 * apq, aqq - app)
    c = tl.cos(phi)
    s = tl.sin(phi)
    cc = c * c
    ss = s * s
    cs2 = 2.0 * c * s
    app_n = cc * app - cs2 * apq + ss * aqq
    aqq_n = ss * app + cs2 * apq + cc * aqq
    apq_n = apq * 0.0
    apr_n = c * apr - s * aqr
    aqr_n = s * apr + c * aqr
    aps_n = c * aps - s * aqs
    aqs_n = s * aps + c * aqs

    vp0_n = c * vp0 - s * vq0
    vq0_n = s * vp0 + c * vq0
    vp1_n = c * vp1 - s * vq1
    vq1_n = s * vp1 + c * vq1
    vp2_n = c * vp2 - s * vq2
    vq2_n = s * vp2 + c * vq2
    vp3_n = c * vp3 - s * vq3
    vq3_n = s * vp3 + c * vq3
    return (
        app_n, aqq_n, apq_n, apr_n, aps_n, aqr_n, aqs_n,
        vp0_n, vp1_n, vp2_n, vp3_n, vq0_n, vq1_n, vq2_n, vq3_n,
    )


@triton.jit
def _optimal_rotation_nocentroid_k(
    A_ptr, B_ptr, R_ptr,
    Bsz, n_d, A_BATCH: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = b < Bsz
    a_b = tl.where(A_BATCH > 1, b, 0)

    s00 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    s01 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    s02 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    s10 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    s11 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    s12 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    s20 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    s21 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    s22 = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for i in range(n_d):
        oa = (a_b * n_d + i) * 3
        ob = (b * n_d + i) * 3
        ax = tl.load(A_ptr + oa + 0, mask=mask, other=0.0)
        ay = tl.load(A_ptr + oa + 1, mask=mask, other=0.0)
        az = tl.load(A_ptr + oa + 2, mask=mask, other=0.0)
        bx = tl.load(B_ptr + ob + 0, mask=mask, other=0.0)
        by = tl.load(B_ptr + ob + 1, mask=mask, other=0.0)
        bz = tl.load(B_ptr + ob + 2, mask=mask, other=0.0)
        s00 += ax * bx; s01 += ax * by; s02 += ax * bz
        s10 += ay * bx; s11 += ay * by; s12 += ay * bz
        s20 += az * bx; s21 += az * by; s22 += az * bz

    a00 = s00 + s11 + s22
    a01 = s12 - s21
    a02 = s20 - s02
    a03 = s01 - s10
    a11 = s00 - s11 - s22
    a12 = s01 + s10
    a13 = s20 + s02
    a22 = -s00 + s11 - s22
    a23 = s12 + s21
    a33 = -s00 - s11 + s22

    v00 = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
    v01 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v02 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v03 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v10 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v11 = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
    v12 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v13 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v20 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v21 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v22 = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
    v23 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v30 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v31 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v32 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    v33 = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)

    for _ in range(6):
        a00, a11, a01, a02, a03, a12, a13, v00, v10, v20, v30, v01, v11, v21, v31 = _jacobi_step4(
            a00, a11, a01, a02, a03, a12, a13, v00, v10, v20, v30, v01, v11, v21, v31
        )
        a00, a22, a02, a01, a03, a12, a23, v00, v10, v20, v30, v02, v12, v22, v32 = _jacobi_step4(
            a00, a22, a02, a01, a03, a12, a23, v00, v10, v20, v30, v02, v12, v22, v32
        )
        a00, a33, a03, a01, a02, a13, a23, v00, v10, v20, v30, v03, v13, v23, v33 = _jacobi_step4(
            a00, a33, a03, a01, a02, a13, a23, v00, v10, v20, v30, v03, v13, v23, v33
        )
        a11, a22, a12, a01, a13, a02, a23, v01, v11, v21, v31, v02, v12, v22, v32 = _jacobi_step4(
            a11, a22, a12, a01, a13, a02, a23, v01, v11, v21, v31, v02, v12, v22, v32
        )
        a11, a33, a13, a01, a12, a03, a23, v01, v11, v21, v31, v03, v13, v23, v33 = _jacobi_step4(
            a11, a33, a13, a01, a12, a03, a23, v01, v11, v21, v31, v03, v13, v23, v33
        )
        a22, a33, a23, a02, a12, a03, a13, v02, v12, v22, v32, v03, v13, v23, v33 = _jacobi_step4(
            a22, a33, a23, a02, a12, a03, a13, v02, v12, v22, v32, v03, v13, v23, v33
        )

    wmax = tl.maximum(tl.maximum(a00, a11), tl.maximum(a22, a33))
    m0 = a00 == wmax
    m1 = (~m0) & (a11 == wmax)
    m2 = (~m0) & (~m1) & (a22 == wmax)
    qw = tl.where(m0, v00, tl.where(m1, v01, tl.where(m2, v02, v03)))
    qx = tl.where(m0, v10, tl.where(m1, v11, tl.where(m2, v12, v13)))
    qy = tl.where(m0, v20, tl.where(m1, v21, tl.where(m2, v22, v23)))
    qz = tl.where(m0, v30, tl.where(m1, v31, tl.where(m2, v32, v33)))

    base_r = b * 9
    tl.store(R_ptr + base_r + 0, 1.0 - 2.0 * (qy * qy + qz * qz), mask=mask)
    tl.store(R_ptr + base_r + 1, 2.0 * (qx * qy - qz * qw), mask=mask)
    tl.store(R_ptr + base_r + 2, 2.0 * (qx * qz + qy * qw), mask=mask)
    tl.store(R_ptr + base_r + 3, 2.0 * (qx * qy + qz * qw), mask=mask)
    tl.store(R_ptr + base_r + 4, 1.0 - 2.0 * (qx * qx + qz * qz), mask=mask)
    tl.store(R_ptr + base_r + 5, 2.0 * (qy * qz - qx * qw), mask=mask)
    tl.store(R_ptr + base_r + 6, 2.0 * (qx * qz - qy * qw), mask=mask)
    tl.store(R_ptr + base_r + 7, 2.0 * (qy * qz + qx * qw), mask=mask)
    tl.store(R_ptr + base_r + 8, 1.0 - 2.0 * (qx * qx + qy * qy), mask=mask)


def _renorm_axisangle(omega: cp.ndarray) -> cp.ndarray:
    """Renormalise axis-angle vectors in-place-safe: if ||omega|| > pi, scale to pi."""
    B = omega.shape[0]
    if B == 0:
        return omega.copy()
    out = cp.empty_like(omega)
    gpu.launch(
        _renorm_axisangle_k, (gpu.cdiv(B, 128),),
        cp.ascontiguousarray(omega), out, B, PI=math.pi, EPS=_EPS, BLOCK_B=128, num_warps=4,
    )
    return out


def _optimal_rotation_nocentroid(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    """Batched Horn closed-form rotation alignment R[b] minimizing sum_i || R[b] @ A[b, i] - B[b, i] ||^2."""
    Bsz = B.shape[0]
    if Bsz == 0:
        return cp.zeros((0, 3, 3), dtype=B.dtype)
    n_d = B.shape[1]
    R = cp.empty((Bsz, 3, 3), dtype=B.dtype)
    gpu.launch(
        _optimal_rotation_nocentroid_k, (gpu.cdiv(Bsz, 64),),
        cp.ascontiguousarray(A), cp.ascontiguousarray(B), R, Bsz, n_d, A_BATCH=A.shape[0], BLOCK_B=64, num_warps=2,
    )
    return R
