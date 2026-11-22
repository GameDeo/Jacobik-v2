"""
angle_bends.py
==============
Optional bond-angle bend DOFs on chelate backbones (``solve(angle_bends=True)``; off by default).

DOFs: for every acyclic heavy-heavy bond (a bridge of the ligand graph) on a shortest path between two kappa donors
of the same ligand, and for each end atom of that bond that has another neighbour, two bend joints: rotations of the
bond's smaller side about axes through that end atom perpendicular to the bond (n1 = normal of the plane of the bond
and a reference neighbour -> in-plane bend; n2 = bond x n1 -> out-of-plane bend). A bend is a rigid rotation of one
side, like a torsion about a different axis, so bond lengths and ring internals never change. The bend columns come
after every other joint of the ligand (theta tail, start value 0 = the embed's angles) and are applied after the
normal FK. Ligands with macrocycle joints get none (their theta tail is the ring).

Penalty (score term "Q7"): every bond angle (H included) at every bend pivot, measured from X against its embed value:
    Q7 = -weight * mean_i 0.5 * (max(0, |dtheta_i| - free) / sigma)^2
a flat-bottom quadratic (a Gaussian prior on the angle change with a free zone). It is part of the score, so it also
counts when conformers are ranked.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import triton
import triton.language as tl

from jacobik.core import gpu, profile
from jacobik.core.env import RT_KJ
from jacobik.core.gpu import cp
FREE_DEG = float(profile.get("bend_free_deg"))

#: Bend stiffness k (kJ/mol/rad^2) by (axis, pivot hybridisation), from MACE-POLAR-1 single points on 110 bend DOFs of
#: 12 chelating ligands (free ligands after the chelate-aware embed; bends of -12..12 deg; E = k beta^2 / 2 fits well).
#: Measured once offline (2026-09-24); MACE is never called during a solve.
BEND_K = {("in", "sp2"): 1300.0, ("in", "sp3"): 1800.0, ("out", "sp2"): 310.0, ("out", "sp3"): 1100.0}
#: Lean wall for the free lean of small haptic faces (profile field haptic_lean_wall): the lean stays free (k = 0) inside
#: the range the data shows and gets the ordinary bend stiffness beyond it, so a substituent cannot flip through its
#: face atom (a 180-deg lean the per-substituent bend statistic cannot tell from a mildly bent-in H). Range = max
#: |asin(bend)| over the 2.5 / 97.5 % quantiles of the per-substituent H bend (tmQM + LnQM, 2026-09-27):
#: eta2 -0.84 / 0.83 -> 57 deg, eta3 -0.64 / 0.21 -> 40 deg.
# Production resolved configuration (hardcoded values)
HAPTIC_LEAN_WALL = True
LEAN_FREE_DEG = {2: 57.0, 3: 40.0}
HAPTIC_LEAN_ALL = True
LEAN_FREE_DEG_ALL = {2: 57.0, 3: 40.0, 4: 50.0, 5: 17.0, 6: 16.0, 7: 11.0, 8: 18.0}


#: Lean range of an eta2 substituent that leans out of the shared face plane (spec kind[3] set, eta2_face_plane): the
#: same rule on the faces that use the plane (tmQM + LnQM, per-substituent bend under eta2_plane_normal, 8100 faces,
#: 2026-11-01): H -0.412 / -0.010 -> 24.3 deg, heavy -0.388 / 0.101 -> 22.8 deg.
LEAN_FREE_DEG_PLANE = 24.0


def lean_free_deg() -> dict:
    """The lean ranges in force (HAPTIC_LEAN_ALL = True in production)."""
    return LEAN_FREE_DEG_ALL if HAPTIC_LEAN_ALL else LEAN_FREE_DEG


LOCAL_PLANE_DEG = 20.0
LOCAL_PLANE_SIN = math.sin(math.radians(LOCAL_PLANE_DEG))


def _cp_normalize(v: cp.ndarray, axis: int = -1, eps: float = 1e-12) -> cp.ndarray:
    n = cp.maximum(cp.linalg.norm(v, axis=axis, keepdims=True), eps)
    return v / n


def eta2_plane_normal_t(b: cp.ndarray, A: cp.ndarray, Xs: cp.ndarray, mask: cp.ndarray, on: cp.ndarray) -> cp.ndarray:
    """CuPy batched copy of orientation.eta2_plane_normal."""
    u = _cp_normalize(Xs - A, axis=-1)
    u = (u - (u * b[..., None, :]).sum(-1, keepdims=True) * b[..., None, :]) * mask[..., None]
    ex = cp.zeros_like(b)
    ey = cp.zeros_like(b)
    ex[..., 0] = 1.0
    ey[..., 1] = 1.0
    ax = cp.where(cp.abs(b[..., :1]) < 0.9, ex, ey)
    e1 = _cp_normalize(cp.cross(b, ax, axis=-1), axis=-1)
    e2 = cp.cross(b, e1, axis=-1)
    x = (u * e1[..., None, :]).sum(-1)
    y = (u * e2[..., None, :]).sum(-1)
    sxy = (x * y).sum(-1) * on
    dxy = ((x * x).sum(-1) - (y * y).sum(-1)) * on + (1.0 - on)
    th = 0.5 * cp.arctan2(2.0 * sxy, dxy)
    d = cp.cos(th)[..., None] * e1 + cp.sin(th)[..., None] * e2
    return cp.cross(b, d, axis=-1)


@triton.jit
def _mm3_tl(
    a00, a01, a02, a10, a11, a12, a20, a21, a22,
    b00, b01, b02, b10, b11, b12, b20, b21, b22,
):
    return (
        a00 * b00 + a01 * b10 + a02 * b20,
        a00 * b01 + a01 * b11 + a02 * b21,
        a00 * b02 + a01 * b12 + a02 * b22,
        a10 * b00 + a11 * b10 + a12 * b20,
        a10 * b01 + a11 * b11 + a12 * b21,
        a10 * b02 + a11 * b12 + a12 * b22,
        a20 * b00 + a21 * b10 + a22 * b20,
        a20 * b01 + a21 * b11 + a22 * b21,
        a20 * b02 + a21 * b12 + a22 * b22,
    )


@triton.jit
def _load12_tl(ptr, base, mask, diag):
    return (
        tl.load(ptr + base + 0, mask=mask, other=diag),
        tl.load(ptr + base + 1, mask=mask, other=0.0),
        tl.load(ptr + base + 2, mask=mask, other=0.0),
        tl.load(ptr + base + 3, mask=mask, other=0.0),
        tl.load(ptr + base + 4, mask=mask, other=diag),
        tl.load(ptr + base + 5, mask=mask, other=0.0),
        tl.load(ptr + base + 6, mask=mask, other=0.0),
        tl.load(ptr + base + 7, mask=mask, other=0.0),
        tl.load(ptr + base + 8, mask=mask, other=diag),
        tl.load(ptr + base + 9, mask=mask, other=0.0),
        tl.load(ptr + base + 10, mask=mask, other=0.0),
        tl.load(ptr + base + 11, mask=mask, other=0.0),
    )


@triton.jit
def _store12_tl(ptr, base, mask, m00, m01, m02, m10, m11, m12, m20, m21, m22, t0, t1, t2):
    tl.store(ptr + base + 0, m00, mask=mask)
    tl.store(ptr + base + 1, m01, mask=mask)
    tl.store(ptr + base + 2, m02, mask=mask)
    tl.store(ptr + base + 3, m10, mask=mask)
    tl.store(ptr + base + 4, m11, mask=mask)
    tl.store(ptr + base + 5, m12, mask=mask)
    tl.store(ptr + base + 6, m20, mask=mask)
    tl.store(ptr + base + 7, m21, mask=mask)
    tl.store(ptr + base + 8, m22, mask=mask)
    tl.store(ptr + base + 9, t0, mask=mask)
    tl.store(ptr + base + 10, t1, mask=mask)
    tl.store(ptr + base + 11, t2, mask=mask)


@triton.jit
def _rodrigues_mat_tl(ax, ay, az, sn, cs):
    omc = 1.0 - cs
    return (
        1.0 - omc * (ay * ay + az * az), -sn * az + omc * ax * ay, sn * ay + omc * ax * az,
        sn * az + omc * ax * ay, 1.0 - omc * (ax * ax + az * az), -sn * ax + omc * ay * az,
        -sn * ay + omc * ax * az, sn * ax + omc * ay * az, 1.0 - omc * (ax * ax + ay * ay),
    )


@triton.jit
def _rodrigues_bwd_tl(
    ax, ay, az, sn, cs,
    t00, t01, t02, t10, t11, t12, t20, t21, t22,
):
    omc = 1.0 - cs
    vx, vy, vz = t21 - t12, t02 - t20, t10 - t01
    dot_Kx = vx * ax + vy * ay + vz * az
    s01, s02, s12 = t01 + t10, t02 + t20, t12 + t21
    dot_Kx2 = (
        -t00 * (ay * ay + az * az) - t11 * (ax * ax + az * az) - t22 * (ax * ax + ay * ay)
        + s01 * ax * ay + s02 * ax * az + s12 * ay * az
    )
    g_ang = cs * dot_Kx + sn * dot_Kx2
    gax = sn * vx + omc * (-2.0 * (t11 + t22) * ax + s01 * ay + s02 * az)
    gay = sn * vy + omc * (s01 * ax - 2.0 * (t00 + t22) * ay + s12 * az)
    gaz = sn * vz + omc * (s02 * ax + s12 * ay - 2.0 * (t00 + t11) * az)
    return g_ang, gax, gay, gaz


@triton.jit
def _se3_chains_fwd_k(
    X_ptr, Y_ptr, T_ptr, G_hist_ptr, G_final_ptr,
    chain_steps_ptr, chain_atom_ptr, chain_atoms_ptr,
    B, R_ref, N, C, D, KT,
    ROW_VEC: tl.constexpr,
    SAVE_GFINAL: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    c = tl.program_id(1)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = b < B
    r_idx = tl.where(R_ref > 1, b, 0)

    g00 = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
    g01 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    g02 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    g10 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    g11 = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
    g12 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    g20 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    g21 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    g22 = tl.full((BLOCK_B,), 1.0, dtype=tl.float32)
    gt0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gt1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gt2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for d in range(D):
        _store12_tl(G_hist_ptr, ((b * C + c) * D + d) * 12, mask, g00, g01, g02, g10, g11, g12, g20, g21, g22, gt0, gt1, gt2)
        idx = tl.load(chain_steps_ptr + c * D + d)
        if idx < KT:
            a00, a01, a02, a10, a11, a12, a20, a21, a22, tx, ty, tz = _load12_tl(T_ptr, (b * KT + idx) * 12, mask, 1.0)
            if ROW_VEC:
                g00, g01, g02, g10, g11, g12, g20, g21, g22 = _mm3_tl(g00, g01, g02, g10, g11, g12, g20, g21, g22, a00, a01, a02, a10, a11, a12, a20, a21, a22)
                gt0, gt1, gt2 = gt0 * a00 + gt1 * a10 + gt2 * a20 + tx, gt0 * a01 + gt1 * a11 + gt2 * a21 + ty, gt0 * a02 + gt1 * a12 + gt2 * a22 + tz
            else:
                g00, g01, g02, g10, g11, g12, g20, g21, g22 = _mm3_tl(a00, a01, a02, a10, a11, a12, a20, a21, a22, g00, g01, g02, g10, g11, g12, g20, g21, g22)
                gt0, gt1, gt2 = a00 * gt0 + a01 * gt1 + a02 * gt2 + tx, a10 * gt0 + a11 * gt1 + a12 * gt2 + ty, a20 * gt0 + a21 * gt1 + a22 * gt2 + tz

    if SAVE_GFINAL:
        base_gf = (b * C + c) * 9
        tl.store(G_final_ptr + base_gf + 0, g00, mask=mask)
        tl.store(G_final_ptr + base_gf + 1, g01, mask=mask)
        tl.store(G_final_ptr + base_gf + 2, g02, mask=mask)
        tl.store(G_final_ptr + base_gf + 3, g10, mask=mask)
        tl.store(G_final_ptr + base_gf + 4, g11, mask=mask)
        tl.store(G_final_ptr + base_gf + 5, g12, mask=mask)
        tl.store(G_final_ptr + base_gf + 6, g20, mask=mask)
        tl.store(G_final_ptr + base_gf + 7, g21, mask=mask)
        tl.store(G_final_ptr + base_gf + 8, g22, mask=mask)

    a_start = tl.load(chain_atom_ptr + c)
    a_end = tl.load(chain_atom_ptr + c + 1)
    base_xr = r_idx * (N * 3)
    base_y = b * (N * 3)
    for m_idx in range(a_start, a_end):
        atom = tl.load(chain_atoms_ptr + m_idx)
        xx = tl.load(X_ptr + base_xr + atom * 3 + 0, mask=mask, other=0.0)
        xy = tl.load(X_ptr + base_xr + atom * 3 + 1, mask=mask, other=0.0)
        xz = tl.load(X_ptr + base_xr + atom * 3 + 2, mask=mask, other=0.0)
        if ROW_VEC:
            yx, yy, yz = xx * g00 + xy * g10 + xz * g20 + gt0, xx * g01 + xy * g11 + xz * g21 + gt1, xx * g02 + xy * g12 + xz * g22 + gt2
        else:
            yx, yy, yz = g00 * xx + g01 * xy + g02 * xz + gt0, g10 * xx + g11 * xy + g12 * xz + gt1, g20 * xx + g21 * xy + g22 * xz + gt2
        tl.store(Y_ptr + base_y + atom * 3 + 0, yx, mask=mask)
        tl.store(Y_ptr + base_y + atom * 3 + 1, yy, mask=mask)
        tl.store(Y_ptr + base_y + atom * 3 + 2, yz, mask=mask)


@triton.jit
def _se3_chains_bwd_k(
    X_ptr, gY_ptr, gX_ptr, T_ptr, G_hist_ptr, G_final_ptr, gT_chain_ptr,
    chain_steps_ptr, chain_atom_ptr, chain_atoms_ptr,
    B, R_ref, N, C, D, KT,
    ROW_VEC: tl.constexpr,
    USE_GFINAL: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    c = tl.program_id(1)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = b < B
    r_idx = tl.where(R_ref > 1, b, 0)

    if USE_GFINAL:
        base_gf = (b * C + c) * 9
        gf00 = tl.load(G_final_ptr + base_gf + 0, mask=mask, other=1.0)
        gf01 = tl.load(G_final_ptr + base_gf + 1, mask=mask, other=0.0)
        gf02 = tl.load(G_final_ptr + base_gf + 2, mask=mask, other=0.0)
        gf10 = tl.load(G_final_ptr + base_gf + 3, mask=mask, other=0.0)
        gf11 = tl.load(G_final_ptr + base_gf + 4, mask=mask, other=1.0)
        gf12 = tl.load(G_final_ptr + base_gf + 5, mask=mask, other=0.0)
        gf20 = tl.load(G_final_ptr + base_gf + 6, mask=mask, other=0.0)
        gf21 = tl.load(G_final_ptr + base_gf + 7, mask=mask, other=0.0)
        gf22 = tl.load(G_final_ptr + base_gf + 8, mask=mask, other=1.0)

    gg00 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gg01 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gg02 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gg10 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gg11 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gg12 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gg20 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gg21 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gg22 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ggt0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ggt1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ggt2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

    a_start = tl.load(chain_atom_ptr + c)
    a_end = tl.load(chain_atom_ptr + c + 1)
    base_xr = r_idx * (N * 3)
    base_y = b * (N * 3)
    for m_idx in range(a_start, a_end):
        atom = tl.load(chain_atoms_ptr + m_idx)
        xx = tl.load(X_ptr + base_xr + atom * 3 + 0, mask=mask, other=0.0)
        xy = tl.load(X_ptr + base_xr + atom * 3 + 1, mask=mask, other=0.0)
        xz = tl.load(X_ptr + base_xr + atom * 3 + 2, mask=mask, other=0.0)
        gyx = tl.load(gY_ptr + base_y + atom * 3 + 0, mask=mask, other=0.0)
        gyy = tl.load(gY_ptr + base_y + atom * 3 + 1, mask=mask, other=0.0)
        gyz = tl.load(gY_ptr + base_y + atom * 3 + 2, mask=mask, other=0.0)
        if USE_GFINAL:
            tl.store(gX_ptr + base_y + atom * 3 + 0, gf00 * gyx + gf10 * gyy + gf20 * gyz, mask=mask)
            tl.store(gX_ptr + base_y + atom * 3 + 1, gf01 * gyx + gf11 * gyy + gf21 * gyz, mask=mask)
            tl.store(gX_ptr + base_y + atom * 3 + 2, gf02 * gyx + gf12 * gyy + gf22 * gyz, mask=mask)
        ggt0 += gyx; ggt1 += gyy; ggt2 += gyz
        if ROW_VEC:
            gg00 += xx * gyx; gg01 += xx * gyy; gg02 += xx * gyz
            gg10 += xy * gyx; gg11 += xy * gyy; gg12 += xy * gyz
            gg20 += xz * gyx; gg21 += xz * gyy; gg22 += xz * gyz
        else:
            gg00 += gyx * xx; gg01 += gyx * xy; gg02 += gyx * xz
            gg10 += gyy * xx; gg11 += gyy * xy; gg12 += gyy * xz
            gg20 += gyz * xx; gg21 += gyz * xy; gg22 += gyz * xz

    for d_rev in range(D):
        d = D - 1 - d_rev
        idx = tl.load(chain_steps_ptr + c * D + d)
        if idx < KT:
            p00, p01, p02, p10, p11, p12, p20, p21, p22, pt0, pt1, pt2 = _load12_tl(
                G_hist_ptr, ((b * C + c) * D + d) * 12, mask, 0.0
            )
            if ROW_VEC:
                g00, g01, g02, g10, g11, g12, g20, g21, g22 = _mm3_tl(p00, p10, p20, p01, p11, p21, p02, p12, p22, gg00, gg01, gg02, gg10, gg11, gg12, gg20, gg21, gg22)
                g00 += pt0 * ggt0; g01 += pt0 * ggt1; g02 += pt0 * ggt2
                g10 += pt1 * ggt0; g11 += pt1 * ggt1; g12 += pt1 * ggt2
                g20 += pt2 * ggt0; g21 += pt2 * ggt1; g22 += pt2 * ggt2
            else:
                g00, g01, g02, g10, g11, g12, g20, g21, g22 = _mm3_tl(gg00, gg01, gg02, gg10, gg11, gg12, gg20, gg21, gg22, p00, p10, p20, p01, p11, p21, p02, p12, p22)
                g00 += ggt0 * pt0; g01 += ggt0 * pt1; g02 += ggt0 * pt2
                g10 += ggt1 * pt0; g11 += ggt1 * pt1; g12 += ggt1 * pt2
                g20 += ggt2 * pt0; g21 += ggt2 * pt1; g22 += ggt2 * pt2
            _store12_tl(gT_chain_ptr, ((b * C + c) * D + d) * 12, mask, g00, g01, g02, g10, g11, g12, g20, g21, g22, ggt0, ggt1, ggt2)

            a00, a01, a02, a10, a11, a12, a20, a21, a22, _, _, _ = _load12_tl(T_ptr, (b * KT + idx) * 12, mask, 1.0)
            if ROW_VEC:
                gg00, gg01, gg02, gg10, gg11, gg12, gg20, gg21, gg22 = _mm3_tl(gg00, gg01, gg02, gg10, gg11, gg12, gg20, gg21, gg22, a00, a10, a20, a01, a11, a21, a02, a12, a22)
                ggt0, ggt1, ggt2 = ggt0 * a00 + ggt1 * a01 + ggt2 * a02, ggt0 * a10 + ggt1 * a11 + ggt2 * a12, ggt0 * a20 + ggt1 * a21 + ggt2 * a22
            else:
                gg00, gg01, gg02, gg10, gg11, gg12, gg20, gg21, gg22 = _mm3_tl(a00, a10, a20, a01, a11, a21, a02, a12, a22, gg00, gg01, gg02, gg10, gg11, gg12, gg20, gg21, gg22)
                ggt0, ggt1, ggt2 = a00 * ggt0 + a10 * ggt1 + a20 * ggt2, a01 * ggt0 + a11 * ggt1 + a21 * ggt2, a02 * ggt0 + a12 * ggt1 + a22 * ggt2


def _build_csr_tables(atom_chain, moved, cs, C, D, K_total):
    """Build CSR chain->atoms and transform->(c*D+d) occurrence tables on GPU."""
    atom_chain_l, moved_l = atom_chain.tolist(), moved.tolist()
    chain_atoms, chain_atom_ptr = [], [0]
    for c in range(C):
        chain_atoms.extend(moved_l[i] for i, ac in enumerate(atom_chain_l) if ac == c)
        chain_atom_ptr.append(len(chain_atoms))
    cs_l = cs.tolist()
    occ_cd, occ_ptr = [], [0]
    for k in range(K_total):
        occ_cd.extend(c * D + d for c in range(C) for d in range(D) if cs_l[c][d] == k)
        occ_ptr.append(len(occ_cd))
    return (
        cp.asarray(chain_atoms, dtype=cp.int32),
        cp.asarray(chain_atom_ptr, dtype=cp.int32),
        cp.asarray(occ_cd, dtype=cp.int32),
        cp.asarray(occ_ptr, dtype=cp.int32),
    )


@triton.jit
def _bend_specs_rodrigues_fwd_k(
    X_ptr, beta_ptr, rv_ext_ptr,
    spec_piv_ptr, spec_oth_ptr, spec_ref_ptr,
    spec_dof0_ptr, spec_ndof_ptr,
    dof_mode_ptr, dof_col_ptr, dof_scale_ptr,
    Rt_ptr, spec_save_ptr,
    B, N, S, K_beta, Ki,
    HAS_RV_EXT: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    s = tl.program_id(1)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = b < B

    piv_i = tl.load(spec_piv_ptr + s)
    oth_i = tl.load(spec_oth_ptr + s)
    ref_i = tl.load(spec_ref_ptr + s)

    base_x = b * (N * 3)
    px = tl.load(X_ptr + base_x + piv_i * 3 + 0, mask=mask, other=0.0)
    py = tl.load(X_ptr + base_x + piv_i * 3 + 1, mask=mask, other=0.0)
    pz = tl.load(X_ptr + base_x + piv_i * 3 + 2, mask=mask, other=0.0)

    ox = tl.load(X_ptr + base_x + oth_i * 3 + 0, mask=mask, other=0.0)
    oy = tl.load(X_ptr + base_x + oth_i * 3 + 1, mask=mask, other=0.0)
    oz = tl.load(X_ptr + base_x + oth_i * 3 + 2, mask=mask, other=0.0)

    dux = ox - px
    duy = oy - py
    duz = oz - pz
    Lu = tl.sqrt(dux * dux + duy * duy + duz * duz)
    Lu_c = tl.maximum(Lu, 1e-12)
    ux = dux / Lu_c
    uy = duy / Lu_c
    uz = duz / Lu_c

    if HAS_RV_EXT:
        base_rv = (b * S + s) * 3
        rvx = tl.load(rv_ext_ptr + base_rv + 0, mask=mask, other=0.0)
        rvy = tl.load(rv_ext_ptr + base_rv + 1, mask=mask, other=0.0)
        rvz = tl.load(rv_ext_ptr + base_rv + 2, mask=mask, other=0.0)
    else:
        rx = tl.load(X_ptr + base_x + ref_i * 3 + 0, mask=mask, other=0.0)
        ry = tl.load(X_ptr + base_x + ref_i * 3 + 1, mask=mask, other=0.0)
        rz = tl.load(X_ptr + base_x + ref_i * 3 + 2, mask=mask, other=0.0)
        rvx = rx - px
        rvy = ry - py
        rvz = rz - pz

    c1x = uy * rvz - uz * rvy
    c1y = uz * rvx - ux * rvz
    c1z = ux * rvy - uy * rvx
    L1 = tl.sqrt(c1x * c1x + c1y * c1y + c1z * c1z)
    L1_c = tl.maximum(L1, 1e-12)
    n1x = c1x / L1_c
    n1y = c1y / L1_c
    n1z = c1z / L1_c

    c2x = uy * n1z - uz * n1y
    c2y = uz * n1x - ux * n1z
    c2z = ux * n1y - uy * n1x
    L2 = tl.sqrt(c2x * c2x + c2y * c2y + c2z * c2z)
    L2_c = tl.maximum(L2, 1e-12)
    n2x = c2x / L2_c
    n2y = c2y / L2_c
    n2z = c2z / L2_c

    base_ss = (b * S + s) * 18
    tl.store(spec_save_ptr + base_ss + 0, px, mask=mask)
    tl.store(spec_save_ptr + base_ss + 1, py, mask=mask)
    tl.store(spec_save_ptr + base_ss + 2, pz, mask=mask)
    tl.store(spec_save_ptr + base_ss + 3, rvx, mask=mask)
    tl.store(spec_save_ptr + base_ss + 4, rvy, mask=mask)
    tl.store(spec_save_ptr + base_ss + 5, rvz, mask=mask)
    tl.store(spec_save_ptr + base_ss + 6, ux, mask=mask)
    tl.store(spec_save_ptr + base_ss + 7, uy, mask=mask)
    tl.store(spec_save_ptr + base_ss + 8, uz, mask=mask)
    tl.store(spec_save_ptr + base_ss + 9, n1x, mask=mask)
    tl.store(spec_save_ptr + base_ss + 10, n1y, mask=mask)
    tl.store(spec_save_ptr + base_ss + 11, n1z, mask=mask)
    tl.store(spec_save_ptr + base_ss + 12, n2x, mask=mask)
    tl.store(spec_save_ptr + base_ss + 13, n2y, mask=mask)
    tl.store(spec_save_ptr + base_ss + 14, n2z, mask=mask)
    tl.store(spec_save_ptr + base_ss + 15, Lu_c, mask=mask)
    tl.store(spec_save_ptr + base_ss + 16, L1_c, mask=mask)
    tl.store(spec_save_ptr + base_ss + 17, L2_c, mask=mask)

    dof0 = tl.load(spec_dof0_ptr + s)
    ndof = tl.load(spec_ndof_ptr + s)
    for dk in range(ndof):
        k = dof0 + dk
        mode = tl.load(dof_mode_ptr + k)
        col = tl.load(dof_col_ptr + k)
        scale = tl.load(dof_scale_ptr + k)
        ang = tl.load(beta_ptr + b * K_beta + col, mask=mask, other=0.0) * scale
        sn = tl.sin(ang)
        cs = tl.cos(ang)
        omc = 1.0 - cs

        ax = tl.where(mode == 0, n1x, tl.where(mode == 1, n2x, ux))
        ay = tl.where(mode == 0, n1y, tl.where(mode == 1, n2y, uy))
        az = tl.where(mode == 0, n1z, tl.where(mode == 1, n2z, uz))

        r00, r01, r02, r10, r11, r12, r20, r21, r22 = _rodrigues_mat_tl(ax, ay, az, sn, cs)
        tx = px - (r00 * px + r01 * py + r02 * pz)
        ty = py - (r10 * px + r11 * py + r12 * pz)
        tz = pz - (r20 * px + r21 * py + r22 * pz)
        _store12_tl(Rt_ptr, (b * Ki + k) * 12, mask, r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz)


@triton.jit
def _bend_specs_rodrigues_bwd_k(
    beta_ptr, gbeta_ptr, gX_ptr, grv_ext_ptr,
    Rt_ptr, spec_save_ptr, gRt_chain_ptr,
    dof_occ_ptr_ptr, dof_occ_cd_ptr,
    spec_piv_ptr, spec_oth_ptr, spec_ref_ptr,
    spec_dof0_ptr, spec_ndof_ptr,
    dof_mode_ptr, dof_col_ptr, dof_scale_ptr,
    B, N, S, C, D, K_beta, Ki,
    HAS_RV_EXT: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    s = tl.program_id(1)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = b < B

    base_ss = (b * S + s) * 18
    px = tl.load(spec_save_ptr + base_ss + 0, mask=mask, other=0.0)
    py = tl.load(spec_save_ptr + base_ss + 1, mask=mask, other=0.0)
    pz = tl.load(spec_save_ptr + base_ss + 2, mask=mask, other=0.0)
    rvx = tl.load(spec_save_ptr + base_ss + 3, mask=mask, other=0.0)
    rvy = tl.load(spec_save_ptr + base_ss + 4, mask=mask, other=0.0)
    rvz = tl.load(spec_save_ptr + base_ss + 5, mask=mask, other=0.0)
    ux = tl.load(spec_save_ptr + base_ss + 6, mask=mask, other=0.0)
    uy = tl.load(spec_save_ptr + base_ss + 7, mask=mask, other=0.0)
    uz = tl.load(spec_save_ptr + base_ss + 8, mask=mask, other=0.0)
    n1x = tl.load(spec_save_ptr + base_ss + 9, mask=mask, other=0.0)
    n1y = tl.load(spec_save_ptr + base_ss + 10, mask=mask, other=0.0)
    n1z = tl.load(spec_save_ptr + base_ss + 11, mask=mask, other=0.0)
    n2x = tl.load(spec_save_ptr + base_ss + 12, mask=mask, other=0.0)
    n2y = tl.load(spec_save_ptr + base_ss + 13, mask=mask, other=0.0)
    n2z = tl.load(spec_save_ptr + base_ss + 14, mask=mask, other=0.0)
    Lu_c = tl.load(spec_save_ptr + base_ss + 15, mask=mask, other=1.0)
    L1_c = tl.load(spec_save_ptr + base_ss + 16, mask=mask, other=1.0)
    L2_c = tl.load(spec_save_ptr + base_ss + 17, mask=mask, other=1.0)

    gn1x = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gn1y = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gn1z = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gn2x = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gn2y = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gn2z = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gux = tl.zeros((BLOCK_B,), dtype=tl.float32)
    guy = tl.zeros((BLOCK_B,), dtype=tl.float32)
    guz = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gpx = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gpy = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gpz = tl.zeros((BLOCK_B,), dtype=tl.float32)

    dof0 = tl.load(spec_dof0_ptr + s)
    ndof = tl.load(spec_ndof_ptr + s)
    for dk in range(ndof):
        k = dof0 + dk
        occ_start = tl.load(dof_occ_ptr_ptr + k)
        occ_end = tl.load(dof_occ_ptr_ptr + k + 1)

        gr00 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gr01 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gr02 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gr10 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gr11 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gr12 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gr20 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gr21 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gr22 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gt0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gt1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        gt2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

        for o_idx in range(occ_start, occ_end):
            cd = tl.load(dof_occ_cd_ptr + o_idx)
            base_grt = (b * (C * D) + cd) * 12
            gr00 += tl.load(gRt_chain_ptr + base_grt + 0, mask=mask, other=0.0)
            gr01 += tl.load(gRt_chain_ptr + base_grt + 1, mask=mask, other=0.0)
            gr02 += tl.load(gRt_chain_ptr + base_grt + 2, mask=mask, other=0.0)
            gr10 += tl.load(gRt_chain_ptr + base_grt + 3, mask=mask, other=0.0)
            gr11 += tl.load(gRt_chain_ptr + base_grt + 4, mask=mask, other=0.0)
            gr12 += tl.load(gRt_chain_ptr + base_grt + 5, mask=mask, other=0.0)
            gr20 += tl.load(gRt_chain_ptr + base_grt + 6, mask=mask, other=0.0)
            gr21 += tl.load(gRt_chain_ptr + base_grt + 7, mask=mask, other=0.0)
            gr22 += tl.load(gRt_chain_ptr + base_grt + 8, mask=mask, other=0.0)
            gt0 += tl.load(gRt_chain_ptr + base_grt + 9, mask=mask, other=0.0)
            gt1 += tl.load(gRt_chain_ptr + base_grt + 10, mask=mask, other=0.0)
            gt2 += tl.load(gRt_chain_ptr + base_grt + 11, mask=mask, other=0.0)

        base_rt = (b * Ki + k) * 12
        r00 = tl.load(Rt_ptr + base_rt + 0, mask=mask, other=1.0)
        r01 = tl.load(Rt_ptr + base_rt + 1, mask=mask, other=0.0)
        r02 = tl.load(Rt_ptr + base_rt + 2, mask=mask, other=0.0)
        r10 = tl.load(Rt_ptr + base_rt + 3, mask=mask, other=0.0)
        r11 = tl.load(Rt_ptr + base_rt + 4, mask=mask, other=1.0)
        r12 = tl.load(Rt_ptr + base_rt + 5, mask=mask, other=0.0)
        r20 = tl.load(Rt_ptr + base_rt + 6, mask=mask, other=0.0)
        r21 = tl.load(Rt_ptr + base_rt + 7, mask=mask, other=0.0)
        r22 = tl.load(Rt_ptr + base_rt + 8, mask=mask, other=1.0)

        gpx += gt0 - (r00 * gt0 + r10 * gt1 + r20 * gt2)
        gpy += gt1 - (r01 * gt0 + r11 * gt1 + r21 * gt2)
        gpz += gt2 - (r02 * gt0 + r12 * gt1 + r22 * gt2)

        tgr00 = gr00 - gt0 * px
        tgr01 = gr01 - gt0 * py
        tgr02 = gr02 - gt0 * pz
        tgr10 = gr10 - gt1 * px
        tgr11 = gr11 - gt1 * py
        tgr12 = gr12 - gt1 * pz
        tgr20 = gr20 - gt2 * px
        tgr21 = gr21 - gt2 * py
        tgr22 = gr22 - gt2 * pz

        mode = tl.load(dof_mode_ptr + k)
        col = tl.load(dof_col_ptr + k)
        scale = tl.load(dof_scale_ptr + k)
        ang = tl.load(beta_ptr + b * K_beta + col, mask=mask, other=0.0) * scale
        sn = tl.sin(ang)
        cs = tl.cos(ang)
        omc = 1.0 - cs

        ax = tl.where(mode == 0, n1x, tl.where(mode == 1, n2x, ux))
        ay = tl.where(mode == 0, n1y, tl.where(mode == 1, n2y, uy))
        az = tl.where(mode == 0, n1z, tl.where(mode == 1, n2z, uz))

        gb_k, gax, gay, gaz = _rodrigues_bwd_tl(
            ax, ay, az, sn, cs,
            tgr00, tgr01, tgr02, tgr10, tgr11, tgr12, tgr20, tgr21, tgr22,
        )
        tl.store(gbeta_ptr + b * K_beta + col, gb_k * scale, mask=mask)

        gn1x += tl.where(mode == 0, gax, 0.0)
        gn1y += tl.where(mode == 0, gay, 0.0)
        gn1z += tl.where(mode == 0, gaz, 0.0)
        gn2x += tl.where(mode == 1, gax, 0.0)
        gn2y += tl.where(mode == 1, gay, 0.0)
        gn2z += tl.where(mode == 1, gaz, 0.0)
        gux += tl.where(mode == 2, gax, 0.0)
        guy += tl.where(mode == 2, gay, 0.0)
        guz += tl.where(mode == 2, gaz, 0.0)

    dot2 = gn2x * n2x + gn2y * n2y + gn2z * n2z
    gc2x = (gn2x - dot2 * n2x) / L2_c
    gc2y = (gn2y - dot2 * n2y) / L2_c
    gc2z = (gn2z - dot2 * n2z) / L2_c

    gux += n1y * gc2z - n1z * gc2y
    guy += n1z * gc2x - n1x * gc2z
    guz += n1x * gc2y - n1y * gc2x

    gn1x += gc2y * uz - gc2z * uy
    gn1y += gc2z * ux - gc2x * uz
    gn1z += gc2x * uy - gc2y * ux

    dot1 = gn1x * n1x + gn1y * n1y + gn1z * n1z
    gc1x = (gn1x - dot1 * n1x) / L1_c
    gc1y = (gn1y - dot1 * n1y) / L1_c
    gc1z = (gn1z - dot1 * n1z) / L1_c

    gux += rvy * gc1z - rvz * gc1y
    guy += rvz * gc1x - rvx * gc1z
    guz += rvx * gc1y - rvy * gc1x

    grvx = gc1y * uz - gc1z * uy
    grvy = gc1z * ux - gc1x * uz
    grvz = gc1x * uy - gc1y * ux

    dotu = gux * ux + guy * uy + guz * uz
    gdux = (gux - dotu * ux) / Lu_c
    gduy = (guy - dotu * uy) / Lu_c
    gduz = (guz - dotu * uz) / Lu_c

    gox = gdux
    goy = gduy
    goz = gduz
    gpx -= gdux
    gpy -= gduy
    gpz -= gduz

    piv_i = tl.load(spec_piv_ptr + s)
    oth_i = tl.load(spec_oth_ptr + s)
    ref_i = tl.load(spec_ref_ptr + s)
    base_x = b * (N * 3)

    if HAS_RV_EXT:
        base_rv = (b * S + s) * 3
        tl.store(grv_ext_ptr + base_rv + 0, grvx, mask=mask)
        tl.store(grv_ext_ptr + base_rv + 1, grvy, mask=mask)
        tl.store(grv_ext_ptr + base_rv + 2, grvz, mask=mask)
    else:
        gpx -= grvx
        gpy -= grvy
        gpz -= grvz
        tl.atomic_add(gX_ptr + base_x + ref_i * 3 + 0, grvx, mask=mask)
        tl.atomic_add(gX_ptr + base_x + ref_i * 3 + 1, grvy, mask=mask)
        tl.atomic_add(gX_ptr + base_x + ref_i * 3 + 2, grvz, mask=mask)

    tl.atomic_add(gX_ptr + base_x + piv_i * 3 + 0, gpx, mask=mask)
    tl.atomic_add(gX_ptr + base_x + piv_i * 3 + 1, gpy, mask=mask)
    tl.atomic_add(gX_ptr + base_x + piv_i * 3 + 2, gpz, mask=mask)
    tl.atomic_add(gX_ptr + base_x + oth_i * 3 + 0, gox, mask=mask)
    tl.atomic_add(gX_ptr + base_x + oth_i * 3 + 1, goy, mask=mask)
    tl.atomic_add(gX_ptr + base_x + oth_i * 3 + 2, goz, mask=mask)




class BendPlan:
    """Vectorised bend kinematics for one ligand (built once; no Python loop over bends at solve time).

    Every ordinary / haptic spec gives two bend DOFs (in-plane, out-of-plane) that rotate the moving set M about a line
    through the pivot p; a hinge spec gives one, rotating M about the line through its two face atoms (u). The axes are taken from the positions X0 that the bends act on (after the torsion FK):
        u  = unit(X0[o] - X0[p]),  n1 = unit(u x (X0[r] - X0[p])),  n2 = unit(u x n1).
    The bends are applied in an order where bend j comes before bend i whenever p_j lies in M_i, so when a bend is
    applied its pivot has not moved yet and its X0 axis line is exact. Each M is one side of a bridge, so the only bond
    crossing into M is the bridge bond and every bend preserves all bond lengths; later bends move earlier ones'
    sets rigidly. (This is a kinematic chain: each bend's hinge rides on the body of its pivot.)

    Evaluation: bend k is the rigid map x -> R_k x + t_k with t_k = p_k - R_k p_k. Atoms moved by the same ordered
    list of bends form a "chain"; the chain transforms are composed by a fixed number of batched matmuls (the maximum
    nesting depth D, a static loop), then gathered per atom and applied in one einsum + one index_copy.
    """

    def __init__(self, specs, pos, device):
        # DOFs: two per ordinary / haptic spec (axis modes 0 = n1, 1 = n2), one per hinge spec (mode 2 = u, the line
        # through the two face atoms). Column order = spec order, so stiffness() lines up.
        dof_spec, dof_mode, dof_col, dof_scale, dof_sets = [], [], [], [], []
        col = 0
        for si, sp in enumerate(specs):
            kind = sp[4] if len(sp) > 4 else None
            main = frozenset(pos[m] for m in sp[3])
            if kind is not None and kind[0] == "hinge":
                dof_spec.append(si); dof_mode.append(2); dof_col.append(col); dof_scale.append(1.0); dof_sets.append(main)
                col += 1
            else:
                for md in (0, 1):
                    dof_spec.append(si); dof_mode.append(md); dof_col.append(col); dof_scale.append(1.0); dof_sets.append(main)
                    col += 1
        self.K = col  # parameter columns (theta)
        K = len(dof_spec)  # internal DOFs
        piv = [pos[sp[0]] for sp in specs]
        oth = [pos[sp[1]] for sp in specs]
        fixed = [frozenset({piv[i], oth[i]}) if (len(sp) > 4 and sp[4][0] == "hinge") else frozenset({piv[i]})
                 for i, sp in enumerate(specs)]
        self.spec_piv = cp.asarray(piv, dtype=cp.int64)
        self.spec_oth = cp.asarray(oth, dtype=cp.int64)
        self.spec_ref = cp.asarray([pos[sp[2]] for sp in specs], dtype=cp.int64)
        self.dof_spec = cp.asarray(dof_spec, dtype=cp.int64)
        self.dof_mode = cp.asarray(dof_mode, dtype=cp.int64)
        self.dof_col = cp.asarray(dof_col, dtype=cp.int64)
        self.dof_scale = cp.asarray(dof_scale, dtype=cp.float32)
        self.dof_moved = [cp.asarray(sorted(s), dtype=cp.int64) for s in dof_sets]
        self.Ki = K
        dof_fix = [fixed[dof_spec[k]] for k in range(K)]
        dof_set = dof_sets
        # Order: j before i if a fixed point of j (pivot; for a hinge both axis atoms) lies in M_i, so every DOF's axis
        # line is still exact when it is applied (Kahn, ties by column index). Same spec: both rotations fix the axis.
        before = {i: {j for j in range(K) if j != i and dof_spec[j] != dof_spec[i] and dof_fix[j] & dof_set[i]}
                  for i in range(K)}
        order, done = [], set()
        while len(order) < K:
            ready = [i for i in range(K) if i not in done and before[i] <= done]
            if not ready:
                raise ValueError("angle-bend ordering has a cycle")
            order.append(ready[0])
            done.add(ready[0])
        rank = {k: n for n, k in enumerate(order)}
        # Chains: for every moved atom, the bends that move it, in application order.
        moved = sorted(set().union(*dof_set)) if K else []
        chain_of = {}
        for a in moved:
            chain_of[a] = tuple(sorted((k for k in range(K) if a in dof_set[k]), key=rank.get))
        chains = sorted(set(chain_of.values()))
        D = max((len(c) for c in chains), default=0)
        self.depth = D
        # (C, D) bend index per chain step, padded with K (= identity transform).
        self.chain_steps = cp.asarray([list(c) + [K] * (D - len(c)) for c in chains], dtype=cp.int64).reshape(len(chains), D)
        cid = {c: n for n, c in enumerate(chains)}
        # Haptic specs: axis reference = face normal (see bend_specs); face atoms padded with the pivot, mask 0.
        hap = [i for i, sp in enumerate(specs) if len(sp) > 4 and sp[4][0] == "haptic" and len(sp[4]) > 2]
        self.n_hapt = len(hap)
        if hap:
            Fm = max(len(specs[i][4][2]) for i in hap)
            Fm = max(Fm, 3)
            fidx = [[piv[i]] * Fm for i in range(len(specs))]
            fmask = [[0.0] * Fm for _ in range(len(specs))]
            part = list(piv)
            is_h = [0.0] * len(specs)
            is_2 = [0.0] * len(specs)
            for i in hap:
                fa = [pos[a] for a in specs[i][4][2]]
                fidx[i][:len(fa)] = fa
                fmask[i][:len(fa)] = [1.0] * len(fa)
                is_h[i] = 1.0
                if len(fa) == 2:
                    is_2[i] = 1.0
                    part[i] = fa[1] if fa[0] == piv[i] else fa[0]
            self.h_face = cp.asarray(fidx, dtype=cp.int64)
            self.h_fmask = cp.asarray(fmask, dtype=cp.float32)
            self.h_part = cp.asarray(part, dtype=cp.int64)
            self.h_is = cp.asarray(is_h, dtype=cp.float32)
            self.h_is2 = cp.asarray(is_2, dtype=cp.float32)
            # eta2 shared face plane (kind[3]: (face atom, substituent) pairs, empty = keep the local plane)
            Sm = max([len(specs[i][4][3]) for i in hap if len(specs[i][4]) > 3] + [1])
            pa = [[piv[i]] * Sm for i in range(len(specs))]
            px = [[oth[i]] * Sm for i in range(len(specs))]
            pm = [[0.0] * Sm for _ in range(len(specs))]
            pon = [0.0] * len(specs)
            for i in hap:
                pairs = specs[i][4][3] if len(specs[i][4]) > 3 else ()
                for j, (a_, x_) in enumerate(pairs):
                    pa[i][j], px[i][j], pm[i][j] = pos[a_], pos[x_], 1.0
                pon[i] = 1.0 if pairs else 0.0
            self.h_pa = cp.asarray(pa, dtype=cp.int64)
            self.h_px = cp.asarray(px, dtype=cp.int64)
            self.h_pm = cp.asarray(pm, dtype=cp.float32)
            self.h_pon = cp.asarray(pon, dtype=cp.float32)
            self.h_plane = any(pon)
        self.moved = cp.asarray(moved, dtype=cp.int64)
        self.atom_chain = cp.asarray([cid[chain_of[a]] for a in moved], dtype=cp.int64)
        if self.K > 0:
            self._ensure_triton_tables()

    def _haptic_ref_and_state(self, X, P, u, rv):
        """Axis reference of haptic specs (the other specs keep rv), plus saved intermediates for analytical VJP."""
        B, S = X.shape[0], self.spec_piv.size
        b_raw = X[:, self.h_part] - P
        b_len = cp.maximum(cp.linalg.norm(b_raw, axis=-1, keepdims=True), 1e-12)
        b = b_raw / b_len
        m2_raw = cp.cross(b, u, axis=-1)
        mid = 0.5 * (P + X[:, self.h_part])
        nf_raw = X[:, self.spec_ref] - mid
        nf_dot = cp.sum(nf_raw * b, axis=-1, keepdims=True)
        nf_perp = nf_raw - nf_dot * b
        nf_len = cp.maximum(cp.linalg.norm(nf_perp, axis=-1, keepdims=True), 1e-12)
        nf = nf_perp / nf_len
        sgn = cp.where(cp.sum(m2_raw * nf, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
        mn = cp.sqrt(cp.sum(m2_raw * m2_raw, axis=-1, keepdims=True) + 1e-12)
        wl = cp.clip(mn / LOCAL_PLANE_SIN, None, 1.0)
        m2_unit = m2_raw / mn
        m2 = wl * m2_unit * sgn + (1.0 - wl) * nf
        plane_state = None
        if getattr(self, "h_plane", False):
            on = cp.broadcast_to(self.h_pon[None, :], (B, S))
            Xa = X[:, self.h_pa]
            Xx = X[:, self.h_px]
            d = Xx - Xa
            d_len = cp.maximum(cp.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
            u_raw = d / d_len
            u_m = (u_raw - cp.sum(u_raw * b[:, :, None, :], axis=-1, keepdims=True) * b[:, :, None, :]) * self.h_pm[None, :, :, None]
            _c_ref = cp.abs(b[..., :1]) < 0.9
            ref = cp.concatenate(
                [_c_ref.astype(b.dtype), (~_c_ref).astype(b.dtype), cp.zeros_like(_c_ref, dtype=b.dtype)],
                axis=-1,
            )
            e1_raw = cp.cross(b, ref, axis=-1)
            e1_len = cp.maximum(cp.linalg.norm(e1_raw, axis=-1, keepdims=True), 1e-12)
            e1 = e1_raw / e1_len
            e2 = cp.cross(b, e1, axis=-1)
            x_c = cp.sum(u_m * e1[:, :, None, :], axis=-1)
            y_c = cp.sum(u_m * e2[:, :, None, :], axis=-1)
            sxy = cp.sum(x_c * y_c, axis=-1) * on
            dxy = cp.sum(x_c * x_c - y_c * y_c, axis=-1) * on + (1.0 - on)
            th = 0.5 * cp.arctan2(2.0 * sxy, dxy)
            d_vec = cp.cos(th)[..., None] * e1 + cp.sin(th)[..., None] * e2
            ns_raw = cp.cross(b, d_vec, axis=-1)
            sgn_ns = cp.where(cp.sum(ns_raw * nf, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
            ns = ns_raw * sgn_ns
            m2 = on[..., None] * ns + (1.0 - on[..., None]) * m2
            plane_state = (on, d_len, u_raw, u_m, ref, e1_len, e1, e2, x_c, y_c, sxy, dxy, th, d_vec, sgn_ns)
        f0, f1, f2 = (X[:, self.h_face[:, j]] for j in range(3))
        m3 = cp.cross(f1 - f0, f2 - f0, axis=-1)
        e2_m = self.h_is2[None, :, None]
        r = e2_m * m2 + (1.0 - e2_m) * m3
        hs = self.h_is[None, :, None]
        rv_out = hs * r + (1.0 - hs) * rv
        return rv_out, (b, b_len, u, m2_raw, nf_raw, nf_dot, nf_len, nf, sgn, mn, wl, m2_unit, f0, f1, f2, plane_state)

    def _haptic_ref_vjp(self, state, grv_ext, u_len, gX):
        (b, b_len, u, m2_raw, nf_raw, nf_dot, nf_len, nf, sgn, mn, wl, m2_unit, f0, f1, f2, plane_state) = state
        e2_m = self.h_is2[None, :, None]
        hs = self.h_is[None, :, None]
        g_r = hs * grv_ext
        g_rv_in = (1.0 - hs) * grv_ext
        g_m2 = e2_m * g_r
        g_m3 = (1.0 - e2_m) * g_r

        g_f10 = cp.cross(f2 - f0, g_m3, axis=-1)
        g_f20 = cp.cross(g_m3, f1 - f0, axis=-1)
        cp.add.at(gX, (slice(None), self.h_face[:, 0]), -(g_f10 + g_f20))
        cp.add.at(gX, (slice(None), self.h_face[:, 1]), g_f10)
        cp.add.at(gX, (slice(None), self.h_face[:, 2]), g_f20)

        g_b = cp.zeros_like(b)
        if plane_state is not None:
            (on, d_len, u_raw, u_m, ref, e1_len, e1, e2, x_c, y_c, sxy, dxy, th, d_vec, sgn_ns) = plane_state
            g_ns_raw = on[..., None] * g_m2 * sgn_ns
            g_m2 = (1.0 - on[..., None]) * g_m2
            g_d_vec = cp.cross(g_ns_raw, b, axis=-1)
            g_b = g_b + cp.cross(d_vec, g_ns_raw, axis=-1)
            g_th = cp.sum(g_d_vec * (-cp.sin(th)[..., None] * e1 + cp.cos(th)[..., None] * e2), axis=-1)
            r2_safe = cp.maximum(4.0 * sxy * sxy + dxy * dxy, 1e-12)
            g_sxy = (g_th * dxy / r2_safe) * on
            g_dxy = (-g_th * sxy / r2_safe) * on
            g_xc = g_sxy[..., None] * y_c + 2.0 * g_dxy[..., None] * x_c
            g_yc = g_sxy[..., None] * x_c - 2.0 * g_dxy[..., None] * y_c
            g_um = g_xc[..., None] * e1[:, :, None, :] + g_yc[..., None] * e2[:, :, None, :]
            g_e1 = g_d_vec * cp.cos(th)[..., None] + cp.sum(g_xc[..., None] * u_m, axis=2)
            g_e2 = g_d_vec * cp.sin(th)[..., None] + cp.sum(g_yc[..., None] * u_m, axis=2)
            g_b = g_b + cp.cross(e1, g_e2, axis=-1)
            g_e1 = g_e1 + cp.cross(g_e2, b, axis=-1)
            g_e1_raw = (g_e1 - cp.sum(g_e1 * e1, axis=-1, keepdims=True) * e1) / e1_len
            g_b = g_b + cp.cross(ref, g_e1_raw, axis=-1)
            g_um = g_um * self.h_pm[None, :, :, None]
            g_u_raw = g_um - cp.sum(g_um * b[:, :, None, :], axis=-1, keepdims=True) * b[:, :, None, :]
            g_b = g_b - cp.sum(
                cp.sum(g_um * b[:, :, None, :], axis=-1, keepdims=True) * u_raw
                + cp.sum(u_raw * b[:, :, None, :], axis=-1, keepdims=True) * g_um,
                axis=2,
            )
            g_d = (g_u_raw - cp.sum(g_u_raw * u_raw, axis=-1, keepdims=True) * u_raw) / d_len
            cp.add.at(gX, (slice(None), self.h_pa), -g_d)
            cp.add.at(gX, (slice(None), self.h_px), g_d)

        g_wl = cp.sum(g_m2 * (m2_unit * sgn - nf), axis=-1, keepdims=True)
        g_m2_unit = g_m2 * (wl * sgn)
        g_nf = g_m2 * (1.0 - wl)
        g_mn = cp.where(mn < LOCAL_PLANE_SIN, g_wl / LOCAL_PLANE_SIN, 0.0) - cp.sum(g_m2_unit * m2_raw, axis=-1, keepdims=True) / (mn * mn)
        g_m2_raw = g_m2_unit / mn + g_mn * (m2_raw / mn)

        g_nf_perp = (g_nf - cp.sum(g_nf * nf, axis=-1, keepdims=True) * nf) / nf_len
        g_nf_raw = g_nf_perp - cp.sum(g_nf_perp * b, axis=-1, keepdims=True) * b
        g_b = g_b - cp.sum(g_nf_perp * b, axis=-1, keepdims=True) * nf_raw - nf_dot * g_nf_perp

        g_b = g_b + cp.cross(u, g_m2_raw, axis=-1)
        g_u = cp.cross(g_m2_raw, b, axis=-1)

        g_b_raw = (g_b - cp.sum(g_b * b, axis=-1, keepdims=True) * b) / b_len
        g_u_raw = (g_u - cp.sum(g_u * u, axis=-1, keepdims=True) * u) / u_len

        g_P = -g_b_raw - 0.5 * g_nf_raw - g_rv_in - g_u_raw
        cp.add.at(gX, (slice(None), self.spec_piv), g_P)
        cp.add.at(gX, (slice(None), self.spec_oth), g_u_raw)
        cp.add.at(gX, (slice(None), self.spec_ref), g_nf_raw + g_rv_in)
        cp.add.at(gX, (slice(None), self.h_part), g_b_raw + 0.5 * g_nf_raw)

    def _haptic_ref(self, X, P, u, rv):
        rv_out, _ = self._haptic_ref_and_state(X, P, u, rv)
        return rv_out

    def axes_pivots(self, X):
        """Return the fixed axis and pivot used by each bend DOF for this pose."""
        P = X[:, self.spec_piv]
        u = _cp_normalize(X[:, self.spec_oth] - P)
        rv = X[:, self.spec_ref] - P
        if self.n_hapt:
            rv = self._haptic_ref(X, P, u, rv)
        n1 = _cp_normalize(cp.cross(u, rv, axis=-1))
        n2 = _cp_normalize(cp.cross(u, n1, axis=-1))
        axes = cp.stack([n1, n2, u], axis=2)
        return axes[:, self.dof_spec, self.dof_mode], P[:, self.dof_spec]

    def jacobian_column_norm_sq(self, X, beta):
        """Exact diagonal of the bend pullback metric for the rigid bend chain."""
        B = X.shape[0]
        K = self.K
        if K == 0:
            return cp.zeros((B, 0), dtype=X.dtype)
        beta_i = beta[:, self.dof_col] * self.dof_scale.astype(beta.dtype)
        axes, pivots = self.axes_pivots(X)
        ax, ay, az = axes[..., 0], axes[..., 1], axes[..., 2]
        z = cp.zeros_like(ax)
        Kx = cp.stack([z, -az, ay, az, z, -ax, -ay, ax, z], axis=-1).reshape(B, K, 3, 3)
        s, c = cp.sin(beta_i)[..., None, None], cp.cos(beta_i)[..., None, None]
        eye = cp.eye(3, dtype=X.dtype)
        R = eye + s * Kx + (1.0 - c) * (Kx @ Kx)
        t = pivots - (R @ pivots[..., None])[..., 0]
        R = cp.concatenate([R, cp.broadcast_to(eye[None, None], (B, 1, 3, 3))], axis=1)
        t = cp.concatenate([t, cp.zeros((B, 1, 3), dtype=t.dtype)], axis=1)

        atom_chain = self.atom_chain
        idx_by_atom = self.chain_steps[atom_chain]
        current = X[:, self.moved]
        n_atoms = current.shape[1]
        jac = cp.zeros((B, K, n_atoms, 3), dtype=X.dtype)
        for depth in range(self.depth):
            idx = idx_by_atom[:, depth]
            valid = idx < K
            Ri = R[:, idx]
            ti = t[:, idx]
            current = cp.einsum("baij,baj->bai", Ri, current) + ti
            jac = cp.einsum("baij,bkaj->bkai", Ri, jac)
            safe_idx = cp.minimum(idx, K - 1)
            derivative = cp.cross(axes[:, safe_idx], current - pivots[:, safe_idx], axis=-1)
            derivative = derivative * self.dof_scale[safe_idx][None, :, None].astype(derivative.dtype)
            one_hot = cp.eye(K, dtype=derivative.dtype)[safe_idx]
            add = (one_hot[None, :, :, None] * derivative[:, :, None, :]).transpose(0, 2, 1, 3)
            jac = jac + add * valid[None, None, :, None]
        return (jac * jac).sum(axis=(2, 3))

    def _ensure_triton_tables(self) -> None:
        if getattr(self, "_triton_tables_ready", False):
            return
        S = int(self.spec_piv.size)
        Ki = self.Ki
        C, D = self.chain_steps.shape
        dof_spec_l = self.dof_spec.tolist()
        spec_dof0, spec_ndof = [], []
        for s in range(S):
            ks = [k for k, ds in enumerate(dof_spec_l) if ds == s]
            spec_dof0.append(ks[0] if ks else 0)
            spec_ndof.append(len(ks))
        self._triton_spec_dof0 = cp.asarray(spec_dof0, dtype=cp.int32)
        self._triton_spec_ndof = cp.asarray(spec_ndof, dtype=cp.int32)

        self._triton_chain_atoms, self._triton_chain_atom_ptr, self._triton_dof_occ_cd, self._triton_dof_occ_ptr = (
            _build_csr_tables(self.atom_chain, self.moved, self.chain_steps, C, D, Ki)
        )
        self._triton_tables_ready = True

    def forward_with_state(self, X, beta):
        """Run bend FK and return (Y, state) for analytical vjp()."""
        if self.K == 0:
            return X, None
        self._ensure_triton_tables()
        Xc = cp.ascontiguousarray(X)
        betac = cp.ascontiguousarray(beta)
        B, N, _ = Xc.shape
        S = int(self.spec_piv.size)
        K_beta = betac.shape[1]
        Ki = self.Ki
        C, D = self.chain_steps.shape

        rv_ext = None
        h_state = None
        u_len = None
        if self.n_hapt:
            P = Xc[:, self.spec_piv]
            u_raw = Xc[:, self.spec_oth] - P
            u_len = cp.maximum(cp.linalg.norm(u_raw, axis=-1, keepdims=True), 1e-12)
            u = u_raw / u_len
            rv_ext, h_state = self._haptic_ref_and_state(Xc, P, u, Xc[:, self.spec_ref] - P)

        Y = Xc.copy()
        Rt = cp.empty((B, Ki, 12), dtype=Xc.dtype)
        spec_save = cp.empty((B, S, 18), dtype=Xc.dtype)
        G_hist = cp.empty((B, C, D, 12), dtype=Xc.dtype)
        G_final = cp.empty((B, C, 9), dtype=Xc.dtype)

        has_rv = rv_ext is not None
        rv_c = cp.ascontiguousarray(rv_ext) if has_rv else Xc

        BLOCK_B = 32
        gpu.launch(
            _bend_specs_rodrigues_fwd_k,
            (gpu.cdiv(B, BLOCK_B), S),
            Xc, betac, rv_c,
            self.spec_piv, self.spec_oth, self.spec_ref,
            self._triton_spec_dof0, self._triton_spec_ndof,
            self.dof_mode, self.dof_col, self.dof_scale,
            Rt, spec_save,
            B, N, S, K_beta, Ki,
            HAS_RV_EXT=has_rv,
            BLOCK_B=BLOCK_B,
            num_warps=1,
        )
        gpu.launch(
            _se3_chains_fwd_k,
            (gpu.cdiv(B, BLOCK_B), C),
            Xc, Y, Rt, G_hist, G_final,
            self.chain_steps, self._triton_chain_atom_ptr, self._triton_chain_atoms,
            B, B, N, C, D, Ki,
            ROW_VEC=False, SAVE_GFINAL=True, BLOCK_B=BLOCK_B, num_warps=1,
        )
        return Y, (Xc, betac, Rt, spec_save, G_hist, G_final, h_state, u_len)

    def vjp(self, state, gY):
        """Analytical backward pass for BendPlan: returns (gX, gbeta)."""
        if self.K == 0 or state is None:
            return gY, cp.zeros((gY.shape[0], 0), dtype=gY.dtype)
        Xc, betac, Rt, spec_save, G_hist, G_final, h_state, u_len = state
        gYc = cp.ascontiguousarray(gY)
        B, N, _ = Xc.shape
        S = int(self.spec_piv.size)
        K_beta = betac.shape[1]
        Ki = self.Ki
        C, D = self.chain_steps.shape
        has_rv = h_state is not None

        gX = gYc.copy()
        gbeta = cp.zeros_like(betac)
        gRt_chain = cp.empty((B, C, D, 12), dtype=Xc.dtype)
        grv_ext = cp.empty((B, S, 3), dtype=Xc.dtype) if has_rv else gX

        BLOCK_B = 32
        gpu.launch(
            _se3_chains_bwd_k,
            (gpu.cdiv(B, BLOCK_B), C),
            Xc, gYc, gX, Rt, G_hist, G_final, gRt_chain,
            self.chain_steps, self._triton_chain_atom_ptr, self._triton_chain_atoms,
            B, B, N, C, D, Ki,
            ROW_VEC=False, USE_GFINAL=True, BLOCK_B=BLOCK_B, num_warps=1,
        )
        gpu.launch(
            _bend_specs_rodrigues_bwd_k,
            (gpu.cdiv(B, BLOCK_B), S),
            betac, gbeta, gX, grv_ext,
            Rt, spec_save, gRt_chain,
            self._triton_dof_occ_ptr, self._triton_dof_occ_cd,
            self.spec_piv, self.spec_oth, self.spec_ref,
            self._triton_spec_dof0, self._triton_spec_ndof,
            self.dof_mode, self.dof_col, self.dof_scale,
            B, N, S, C, D, K_beta, Ki,
            HAS_RV_EXT=has_rv,
            BLOCK_B=BLOCK_B,
            num_warps=1,
        )
        if has_rv:
            self._haptic_ref_vjp(h_state, grv_ext, u_len, gX)
        return gX, gbeta

    def apply(self, X, beta):
        """X (B, N, 3) after the torsion FK, beta (B, K) -> X with every bend applied."""
        Y, _ = self.forward_with_state(X, beta)
        return Y




def pivot_hybridisation(mol, p) -> str:
    """'sp2' if the pivot has a ligand bond of order >= 1.4 (double / aromatic), else 'sp3'."""
    for b in mol.bonds:
        if p in (b.i, b.j) and mol.metal_idx not in (b.i, b.j) and b.order >= 1.4:
            return "sp2"
    return "sp3"


def bound_angle_fallbacks(key):
    """Specific-to-generic ladder for optional Q7 crystal pivot-angle entries."""
    element, hyb, neighbours, ring, row = key
    candidates = [key, (element, hyb, neighbours, None, row),
                  (element, hyb, neighbours, None, None), (element, hyb, None, None, None),
                  (element, None, None, None, None)]
    return list(dict.fromkeys(candidates))


def stiffness(mol, specs) -> List[float]:
    """k (kJ/mol/rad^2) for every bend DOF, in BendPlan column order: two per ordinary or haptic spec (column 0, column
    1), one per hinge.

    Haptic specs: the lean column of faces with <= HAPTIC_FREE_LEAN_MAX atoms (eta2, eta3) and every hinge get k = 0 --
    the prior measures strain against the free-ligand embed, but bend-back / fold is paid for by the metal bond, which
    it cannot see; the fitted S_orient bend of the face constrains them. The sideways column keeps the prior, and so do
    both columns on larger faces (Cp, arenes), where the face-mean bend alone would let single substituents wander."""
    out = []
    for sp in specs:
        p = sp[0]
        kind = sp[4] if len(sp) > 4 else None
        h = pivot_hybridisation(mol, p)
        if kind is not None and kind[0] == "hinge":
            out.append(0.0)
            continue
        k_in, k_out = BEND_K[("in", h)], BEND_K[("out", h)]
        if kind is not None and kind[0] == "haptic" and kind[1] <= HAPTIC_FREE_LEAN_MAX:
            out += [k_in if (HAPTIC_LEAN_WALL and kind[1] in lean_free_deg()) else 0.0, k_in]
            continue
        out += [k_in, k_out]
    return out


def free_zone(specs, free_deg: float = FREE_DEG) -> List[float]:
    """Free zone (deg) of every bend column, in BendPlan column order: free_deg, except the lean of small haptic faces
    under HAPTIC_LEAN_WALL (LEAN_FREE_DEG)."""
    out = []
    for sp in specs:
        kind = sp[4] if len(sp) > 4 else None
        if kind is not None and kind[0] == "hinge":
            out.append(free_deg)
            continue
        if (HAPTIC_LEAN_WALL and kind is not None and kind[0] == "haptic" and kind[1] <= HAPTIC_FREE_LEAN_MAX
                and kind[1] in lean_free_deg()):
            w = LEAN_FREE_DEG_PLANE if len(kind) > 3 and kind[3] else lean_free_deg()[kind[1]]
            out += [max(free_deg, w), free_deg]
            continue
        out += [free_deg, free_deg]
    return out


def energy_prior(beta, k, free_deg: float = FREE_DEG, scale: float = 1.0):
    """(B,) log-prior of bend amounts beta (B, n) in units of RT: -scale * sum_i k_i (max(0, |beta_i| - free))^2 / (2 RT).
    A Boltzmann factor of the harmonic bend energy beyond a free zone, summed over bends (not averaged)."""
    free = cp.deg2rad(free_deg)[None, :] if isinstance(free_deg, cp.ndarray) else math.radians(free_deg)
    ex = cp.maximum(cp.abs(beta) - free, 0.0)
    return -scale * (0.5 * k[None, :] * (ex * ex)).sum(axis=-1) / RT_KJ


def energy_prior_and_grad(
    beta, k, free_deg: float = FREE_DEG, scale: float = 1.0
):
    """Analytical value (B,) and gradient d(prior)/d(beta) (B, n) for energy_prior."""
    free = cp.deg2rad(free_deg)[None, :] if isinstance(free_deg, cp.ndarray) else math.radians(free_deg)
    ex = cp.maximum(cp.abs(beta) - free, 0.0)
    c = -scale / RT_KJ
    val = (0.5 * c) * (k[None, :] * (ex * ex)).sum(axis=-1)
    grad = c * k[None, :] * ex * cp.sign(beta)
    return val, grad


def _lig_graph(mol, ligand):
    import networkx as nx

    idxs = set(ligand.atom_indices)
    G = nx.Graph()
    G.add_nodes_from(idxs)
    for b in mol.bonds:
        if b.i in idxs and b.j in idxs and b.order >= 0.5:
            G.add_edge(b.i, b.j)
    return G


# Production resolved configuration (hardcoded values)
HAPTIC_BENDS = True
HAPTIC_FREE_LEAN_MAX = 6
#: eta2 group bend (production since the eta2 / haptic-cap fold-in, 2026-11-01): on an eta2 face that does not use the shared face plane, a face atom
#: with >= 2 substituents bends them as one rigid group (one lean, one swing) instead of one bend per substituent. Per-H
#: bends each get the 57 deg free lean, and nothing couples a geminal pair: two H can lean onto the same point (TEBWIH,
#: propargyl CH2 next to a C#C carbon: H-C-H 0.6 deg under the haptic cap, 2026-11-01). A group keeps H-C-H as embedded.
ETA2_GROUP_BEND = True


def bend_specs(mol, ligand, openings=None) -> List[Tuple[int, int, int, List[int]]]:
    """[(pivot, other end, reference neighbour, moving atoms)] in atom.idx space; [] if the ligand gets no bends.
    openings: closure bonds {(u, v), ...} of the ligand's joint rings (FK); unused since RING_LEAN was removed
    (2026-10-16), kept for the callers."""
    import networkx as nx

    joint_rings = [set(r) for r in (getattr(ligand, "macrocycles", None) or [])]
    if joint_rings and not _small_joint_rings(mol, joint_rings):
        return []   # classic macrocycle (>= 8, all single bonds): its ring joints replace every bend
    donors = list(ligand.donor_atoms)
    if len(donors) < 2:
        if not donors:
            return []
        return []
    sym = {a.idx: a.symbol for a in mol.atoms}
    G = _lig_graph(mol, ligand)
    bridges = {tuple(sorted(e)) for e in nx.bridges(G)}
    on_path = set()
    for i in range(len(donors)):
        for j in range(i + 1, len(donors)):
            try:
                p = nx.shortest_path(G, donors[i], donors[j])
            except nx.NetworkXNoPath:
                continue
            on_path |= {tuple(sorted(e)) for e in zip(p[:-1], p[1:])}
    out = []
    n = G.number_of_nodes()
    seen = set()
    for a, b in sorted(bridges & on_path):
        if sym[a] == "H" or sym[b] == "H":
            continue
        H = G.copy()
        H.remove_edge(a, b)
        sa = nx.node_connected_component(H, a)
        side = sa if len(sa) <= n - len(sa) else set(G.nodes) - sa
        for piv, oth in ((a, b), (b, a)):
            refs = [x for x in G.neighbors(piv) if x != oth]
            if not refs:
                continue
            refs.sort(key=lambda x: (sym[x] == "H", x))
            out.append((piv, oth, refs[0], sorted(side)))
            seen.add((piv, oth))
    # Haptic faces: turn the fitted S_orient "bend" (lean of the face substituents along the face normal) into motion
    # the solver can make. Without these the substituents stay where the free-ligand embed put them (an alkyne stays
    # linear, a partly bound ring stays flat).
    #  * free branch (bridge bond face atom -> substituent, H included): a bend about the face atom whose reference is
    #    the substituent's own local plane (eta2: bond + substituent; larger faces: the face plane), so column 0 is a
    #    pure lean out of that plane -- what the S_orient bend measures -- and column 1 the in-plane swing (a covalent
    #    angle: keeps its prior). A metal reference made the split depend on how the embed happened to orient the face
    #    (an edge-on alkene got an in-plane "lean" and lost its C=C-H angle). With orientation.ETA2_FACE_PLANE an eta2
    #    face whose atoms each carry >= 2 substituents leans out of the shared face plane instead (the plane S_orient
    #    scores; the kind carries its (face atom, substituent) pairs), so in-plane motion of one H is the stiff swing;
    #  * ring segment that leaves the face and returns to it (the CH2-CH end of an eta4-cyclohexadiene): one hinge,
    #    rotating the segment and everything hanging off it about the line through its two face atoms.
    if HAPTIC_BENDS:
        from jacobik.stats import orientation as O

        metal = mol.metal_idx
        mol_arom = {}
        for b in mol.bonds:
            if mol.metal_idx not in (b.i, b.j) and 1.4 <= b.order <= 1.6:
                mol_arom[b.i] = mol_arom[b.j] = True
        for h in getattr(ligand, "haptic_groups", None) or []:
            face = set(h.donor_indices)
            fsize = len(face)
            plane = ()
            if O.ETA2_FACE_PLANE and fsize == 2:
                pp = tuple((a, x) for a in sorted(face) for x in sorted(G.neighbors(a)) if x not in face)
                if all(sum(1 for a_, _ in pp if a_ == a) >= 2 for a in face):
                    plane = pp
            kind = ("haptic", fsize, tuple(sorted(face)), plane)
            for a in face:
                branches = []
                for x in G.neighbors(a):
                    if x in face or (a, x) in seen:
                        continue
                    e = tuple(sorted((a, x)))
                    if e not in bridges:
                        continue  # ring segment: handled by the hinge below
                    H = G.copy()
                    H.remove_edge(a, x)
                    sx = nx.node_connected_component(H, x)
                    if len(sx) >= n - len(sx):
                        continue
                    branches.append((x, sx))
                if ETA2_GROUP_BEND and fsize == 2 and not plane and len(branches) >= 2:
                    # one rigid group about the face atom; the axis reference is its first heavy (else first) substituent
                    x0 = min(branches, key=lambda b: (sym[b[0]] == "H", b[0]))[0]
                    out.append((a, x0, metal, sorted(set().union(*(sx for _, sx in branches))), kind))
                    seen.update((a, x) for x, _ in branches)
                    continue
                for x, sx in branches:
                    out.append((a, x, metal, sorted(sx), kind))
                    seen.add((a, x))
        # ring folds: a non-aromatic ring holding donors on both sides of a line through two non-adjacent ring atoms
        # (neither a face atom nor a donor) can fold along that line -- the half on one side rotates about it. This is
        # what closes COD from the staggered embed into the tub (both alkenes on one side). Only the angles at the two
        # axis atoms change; the donor geometry, the face statistics and Q6 decide the fold (no spring).
        donors_all = set(ligand.donor_atoms)
        fold_axis_atoms = set()
        try:
            rings_l = [list(c) for c in nx.cycle_basis(G.subgraph([v for v in G.nodes if sym[v] != "H"]))]
        except Exception:
            rings_l = []
        for ring in rings_l:
            rs = set(ring)
            if len(ring) < 6 or any(mol_arom.get(v, False) for v in ring):
                continue
            order_r = nx.cycle_basis(G.subgraph(ring))[0] if len(nx.cycle_basis(G.subgraph(ring))) == 1 else ring
            m = len(order_r)
            cand = [i for i in range(m) if order_r[i] not in donors_all]
            for ia in range(len(cand)):
                for ib in range(ia + 1, len(cand)):
                    i, j = cand[ia], cand[ib]
                    if (j - i) % m in (1, m - 1):
                        continue  # adjacent
                    a1, a2 = order_r[i], order_r[j]
                    H2 = G.copy()
                    H2.remove_nodes_from([a1, a2])
                    arc1 = {order_r[(i + k) % m] for k in range(1, (j - i) % m)}
                    comp1 = nx.node_connected_component(H2, next(iter(arc1)))
                    arc2 = rs - arc1 - {a1, a2}
                    if not arc2 or next(iter(arc2)) in comp1:
                        continue  # the two halves are joined elsewhere (fused ring): cannot fold
                    if not (donors_all & comp1) or not (donors_all & (set(G.nodes) - comp1 - {a1, a2})):
                        continue  # donors on only one side: nothing for a fold to change
                    fold_axis_atoms |= {a1, a2}
        # ring segments: components of the ligand without ANY of its face atoms that touch face atoms exactly twice
        # (one face, as in eta4-cyclohexadiene, or two, as the CH2-CH2 runs between the alkenes of COD) and carry no
        # donor: one hinge each, about the line through the two face atoms they hang from.
        faces = [set(h.donor_indices) for h in (getattr(ligand, "haptic_groups", None) or [])]
        all_face = set().union(*faces) if faces else set()
        size_of = {v: len(f) for f in faces for v in f}
        Gf = G.subgraph([v for v in G.nodes if v not in all_face])
        for comp in nx.connected_components(Gf):
            comp = set(comp)
            ends = [(f, y) for f in all_face for y in G.neighbors(f) if y in comp]
            if len(ends) != 2 or set(ligand.donor_atoms) & comp or comp & fold_axis_atoms:
                continue
            (a1, _), (a2, _) = ends
            if a1 == a2:
                continue  # (the segment moves even when it is the heavier side: the faces must stay put)
            out.append((a1, a2, a2, sorted(comp), ("hinge", min(size_of[a1], size_of[a2]))))
    if joint_rings:
        # small donor rings with torsion joints (topology.decompose.RING_JOINTS): no fold or hinge may move their atoms
        # (RING_KEEP_FOLDS, keeping the folds of rings holding a haptic face: tested 2026-09-30 on COD, worse; removed
        # 2026-10-16)
        drop = set().union(*joint_rings)
        out = [sp for sp in out if not (len(sp) > 4 and sp[4][0] == "hinge"
                                        and ({sp[0], sp[1]} | set(sp[3])) & drop)]
    return _compatible(out)


def _small_joint_rings(mol, rings) -> bool:
    """True if every joint ring is a small donor ring (RING_JOINTS): fewer than 8 atoms or holding a non-single bond."""
    bo = {}
    for b in mol.bonds:
        bo[(min(b.i, b.j), max(b.i, b.j))] = b.order
    for r in rings:
        if len(r) >= 8 and all(abs(bo.get((min(u, v), max(u, v)), 1.0) - 1.0) < 0.25
                               for u in r for v in r if u < v and (min(u, v), max(u, v)) in bo):
            return False
    return True


def _compatible(specs):
    """Keep specs in order, dropping any that would make the kinematic ordering cyclic with an earlier one: spec i
    conflicts with j when a fixed point of each (pivot; for a hinge both axis atoms) lies in the other's moving set
    (e.g. two ring folds whose axes cross). Deterministic."""
    def fixed(sp):
        return {sp[0], sp[1]} if (len(sp) > 4 and sp[4][0] == "hinge") else {sp[0]}
    kept = []
    for sp in specs:
        fi, si = fixed(sp), set(sp[3])
        if any(fi & set(kp[3]) and fixed(kp) & si for kp in kept):
            continue
        kept.append(sp)
    return kept


def angle_triplets(mol) -> List[Tuple[int, int, int]]:
    """(a, pivot, b) atom.idx triplets of every bond angle at every bend pivot of every ligand."""
    nbr = {}
    for b in mol.bonds:
        if mol.metal_idx in (b.i, b.j) or b.order < 0.5:
            continue
        nbr.setdefault(b.i, set()).add(b.j)
        nbr.setdefault(b.j, set()).add(b.i)
    piv = set()
    for lig in mol.ligands:
        for sp in bend_specs(mol, lig):
            piv.add(sp[0])
            if len(sp) > 4 and sp[4][0] == "hinge":
                piv.add(sp[1])
    out = []
    for p in sorted(piv):
        nb = sorted(nbr.get(p, ()))
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                out.append((nb[i], p, nb[j]))
    return out


def angles(X, t):
    """Bond angles (radians) for index triplets t (n, 3) in position space."""
    u = _cp_normalize(X[:, t[:, 0]] - X[:, t[:, 1]])
    v = _cp_normalize(X[:, t[:, 2]] - X[:, t[:, 1]])
    return cp.arccos(cp.clip(cp.sum(u * v, axis=-1), -1.0 + 1e-6, 1.0 - 1e-6))


def penalty(X, t, a0, free_deg: float, sigma_deg: float, weight: float):
    d = cp.abs(angles(X, t) - a0[None])
    if np.isscalar(free_deg):
        free_rad = math.radians(float(free_deg))
    else:
        free = cp.asarray(free_deg, dtype=d.dtype)
        free_rad = cp.deg2rad(free[None, :] if free.ndim == 1 else free)
    ex = cp.maximum(d - free_rad, 0.0) / math.radians(sigma_deg)
    return -weight * 0.5 * (ex * ex).mean(axis=-1)
