"""
Output deduplication: return distinct structures, not copies.

Two solver rows are the same structure when their heavy-atom (incl. metal) symmetry-aware RMSD is <= ``DEDUP_RMSD``
(0.5 A): minimum Kabsch RMSD over all automorphisms of the bonded heavy-atom graph (coordination bonds included), so
methyl/phenyl flips and swapped identical ligands count as the same structure. Mirror images are NOT merged
(no reflection in the superposition).

Greedy best-first: rows are visited in score order and a row is kept unless it duplicates an already-kept row. The
exact criterion is evaluated with two exact shortcuts so the full automorphism enumeration is rarely needed:
  * lower bound — distances to the centroid are invariant to rotation and to any class-preserving permutation;
    sorted within each symmetry class, their RMS difference is <= the symmetric RMSD. LB > threshold -> distinct.
  * upper bound — RMSD under one real automorphism (identity, then refined by alternating Kabsch / best automorphism
    for the fixed rotation). UB <= threshold -> duplicate.
Only pairs between the bounds get the full min over automorphisms. Validated against RDKit GetBestRMS
(identical distinct counts on 15 runs, |dRMSD| < 2e-5 A).
Cost (2026-10-03): a scan over the M automorphisms is one matrix product (the candidate under every inverse
automorphism against the kept rows) and the optimal-rotation overlap comes from Newton's method on the quartic of
Horn's 4x4 matrix (QCP) instead of an SVD per kept row and automorphism; agrees with the SVD form to 4e-13 A.
Cost (2026-10-16): the selection only needs "is some automorphism within th", not the minimum (_full_any): per
automorphism a radial lower bound (one matrix product) and a Frobenius upper bound on the overlap discard most
(row, automorphism) pairs before any eigenvalue fit, and Newton stops as soon as an iterate falls below the needed
overlap (the iterates decrease monotonically) or one converges above it. The upper-bound shortcut (_upper) is no
longer tried first: it rarely decided and cost more than the pruned scan. Same selections as before on 21
benchmark dedups (7 complexes x 3 seeds, up to 20000 automorphisms); YAVFEG (4608 automorphisms, 1536 rows,
192 kept) 20 s -> 7 s per seed on 4 CPU threads.
Cost (2026-11-21): fused Triton GPU kernels for pairwise radial + per-automorphism correlation + Davenport 4x4 QCP
Newton overlap (_pair_dup_vec_kernel), core-permutation RMSD scan (_qcp_core_rmsd_kernel), and closed-form 4x4
adjugate rotation extraction (_kabsch_qcp_kernel), so (N, M, S) intermediate matrices never hit global memory.
"""
from __future__ import annotations

import os
import logging
from collections import OrderedDict
import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import triton
import triton.language as tl
from triton.language.extra import libdevice

from jacobik.core import gpu
from jacobik.core import timing
from jacobik.core import profile
from jacobik.core.geometry import kabsch_fit_batch, kabsch_fit_np  # noqa: F401

cp = gpu.cp

log = logging.getLogger(__name__)
# Experiment / profile configuration
DEDUP_FACTORED = profile.get("dedup_factored")
DEDUP_RMSD = profile.get("dedup_rmsd")
DEDUP_MAX_DEV = profile.get("dedup_max_dev")
#: v1.0b speed: closed-form Kabsch (QCP) in the factored match instead of batched 3x3 SVDs
DEDUP_QCP = profile.get("dedup_qcp")
#: v1.0b speed (2026-11-20): on capped automorphism groups the factored match decides alone and its core-start scan
#: runs on the GPU when one is available. On 15 capped parents (QEZVEY, GOTWET, HEDJOP, XAQCAT, ...) the capped-map
#: scan (_full_any over the first MAX_AUTOMORPHISMS automorphisms) found no duplicate the factored match missed, yet
#: took half the dedup time. Fewer than 36 core starts is NOT safe: 4 starts missed 5 of 108 duplicates on XAQCAT
#: (symmetric cores: many donor permutations fit the core equally well).
DEDUP_CAPPED_FAST = profile.get("dedup_capped_fast")
#: fast path: after the first alternating round only starts with RMSD <= _ROUND0_CUT * th are refined further (every
#: duplicate on the 15 capped parents had a start within 1.0 * th after round 0)
_ROUND0_CUT = 1.5
#: fast path: starts whose nearest-same-class-atom RMSD at the start rotation exceeds _SCREEN_CUT * th are dropped
#: before the first round (duplicates on the 15 capped parents: at most 1.75 * th; about 99 % of distinct starts above)
_SCREEN_CUT = 2.5


@triton.jit
def _det3_triton(a00, a01, a02, a10, a11, a12, a20, a21, a22):
    return (a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20))


@triton.jit
def _qcp_horn_poly_triton(Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz):
    k00 = Sxx + Syy + Szz
    k01 = Syz - Szy
    k02 = Szx - Sxz
    k03 = Sxy - Syx
    k11 = Sxx - Syy - Szz
    k12 = Sxy + Syx
    k13 = Szx + Sxz
    k22 = -Sxx + Syy - Szz
    k23 = Syz + Szy
    k33 = -Sxx - Syy + Szz
    h_norm2 = (Sxx * Sxx + Sxy * Sxy + Sxz * Sxz
               + Syx * Syx + Syy * Syy + Syz * Syz
               + Szx * Szx + Szy * Szy + Szz * Szz)
    c2 = -2.0 * h_norm2
    c1 = -8.0 * _det3_triton(Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz)
    s0 = k00 * k11 - k01 * k01
    s1 = k00 * k12 - k01 * k02
    s2 = k00 * k13 - k01 * k03
    s3 = k01 * k12 - k11 * k02
    s4 = k01 * k13 - k11 * k03
    s5 = k02 * k13 - k12 * k03
    m5 = k22 * k33 - k23 * k23
    m4 = k12 * k33 - k13 * k23
    m3 = k12 * k23 - k13 * k22
    m2 = k02 * k33 - k03 * k23
    m1 = k02 * k23 - k03 * k22
    m0 = k02 * k13 - k03 * k12
    c0 = s0 * m5 - s1 * m4 + s2 * m3 + s3 * m2 - s4 * m1 + s5 * m0
    return k00, k01, k02, k03, k11, k12, k13, k22, k23, k33, h_norm2, c2, c1, c0


@triton.jit
def _qcp_max_eig_triton(
    c2, c1, c0, lam, active, need,
    MAX_ITER: tl.constexpr, TOL: tl.constexpr,
    CHECK_NEED: tl.constexpr, USE_INIT_SCALE: tl.constexpr,
):
    init_scale = tl.where(lam > 1e-300, lam, 1e-300)
    found = active & False
    conv_ok = active & False
    it = 0
    while it < MAX_ITER and tl.max(active.to(tl.int32), axis=0) != 0:
        it += 1
        l2 = lam * lam
        p = (l2 + c2) * l2 + c1 * lam + c0
        dp = (4.0 * l2 + 2.0 * c2) * lam + c1
        step = tl.where(tl.abs(dp) > 1e-300, p / tl.where(tl.abs(dp) > 1e-300, dp, 1.0), 0.0)
        lam = tl.where(active, lam - step, lam)
        if USE_INIT_SCALE:
            conv = tl.abs(step) <= TOL * init_scale
            conv_ok = tl.where(active, conv, conv_ok)
            active = active & (~conv)
        else:
            denom = tl.where(tl.abs(lam) > 1e-30, tl.abs(lam), 1e-30)
            conv = (tl.abs(step) / denom) < TOL
            if CHECK_NEED:
                above = lam >= need
                found = found | (active & conv & above)
                active = active & above & (~conv)
            else:
                active = active & (~conv)
    if CHECK_NEED:
        found = found | active
    return lam, conv_ok, found, init_scale


@triton.jit
def _kabsch_qcp_kernel(
    H_ptr, lam_ptr, R_ptr, ok_ptr, N,
    COMPUTE_ROT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    base9 = offs * 9
    Sxx = tl.load(H_ptr + base9 + 0, mask=mask, other=0.0)
    Sxy = tl.load(H_ptr + base9 + 1, mask=mask, other=0.0)
    Sxz = tl.load(H_ptr + base9 + 2, mask=mask, other=0.0)
    Syx = tl.load(H_ptr + base9 + 3, mask=mask, other=0.0)
    Syy = tl.load(H_ptr + base9 + 4, mask=mask, other=0.0)
    Syz = tl.load(H_ptr + base9 + 5, mask=mask, other=0.0)
    Szx = tl.load(H_ptr + base9 + 6, mask=mask, other=0.0)
    Szy = tl.load(H_ptr + base9 + 7, mask=mask, other=0.0)
    Szz = tl.load(H_ptr + base9 + 8, mask=mask, other=0.0)

    k00, k01, k02, k03, k11, k12, k13, k22, k23, k33, h_norm2, c2, c1, c0 = _qcp_horn_poly_triton(
        Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz
    )
    lam = libdevice.sqrt(tl.where(3.0 * h_norm2 > 0.0, 3.0 * h_norm2, 0.0))
    lam, ok, _, scale = _qcp_max_eig_triton(c2, c1, c0, lam, mask, 0.0, 50, 1e-12, False, True)

    if COMPUTE_ROT:
        m00 = k00 - lam
        m11 = k11 - lam
        m22 = k22 - lam
        m33 = k33 - lam
        c0_0 = _det3_triton(m11, k12, k13, k12, m22, k23, k13, k23, m33)
        c0_1 = -_det3_triton(k01, k12, k13, k02, m22, k23, k03, k23, m33)
        c0_2 = _det3_triton(k01, m11, k13, k02, k12, k23, k03, k13, m33)
        c0_3 = -_det3_triton(k01, m11, k12, k02, k12, m22, k03, k13, k23)
        bn = c0_0 * c0_0 + c0_1 * c0_1 + c0_2 * c0_2 + c0_3 * c0_3
        b0, b1, b2, b3 = c0_0, c0_1, c0_2, c0_3

        c1_0 = c0_1
        c1_1 = _det3_triton(m00, k02, k03, k02, m22, k23, k03, k23, m33)
        c1_2 = -_det3_triton(m00, k01, k03, k02, k12, k23, k03, k13, m33)
        c1_3 = _det3_triton(m00, k01, k02, k02, k12, m22, k03, k13, k23)
        n1 = c1_0 * c1_0 + c1_1 * c1_1 + c1_2 * c1_2 + c1_3 * c1_3
        sel = n1 > bn
        b0 = tl.where(sel, c1_0, b0); b1 = tl.where(sel, c1_1, b1); b2 = tl.where(sel, c1_2, b2); b3 = tl.where(sel, c1_3, b3)
        bn = tl.where(sel, n1, bn)

        c2_0 = c0_2
        c2_1 = c1_2
        c2_2 = _det3_triton(m00, k01, k03, k01, m11, k13, k03, k13, m33)
        c2_3 = -_det3_triton(m00, k01, k02, k01, m11, k12, k03, k13, k23)
        n2 = c2_0 * c2_0 + c2_1 * c2_1 + c2_2 * c2_2 + c2_3 * c2_3
        sel = n2 > bn
        b0 = tl.where(sel, c2_0, b0); b1 = tl.where(sel, c2_1, b1); b2 = tl.where(sel, c2_2, b2); b3 = tl.where(sel, c2_3, b3)
        bn = tl.where(sel, n2, bn)

        c3_0 = c0_3
        c3_1 = c1_3
        c3_2 = c2_3
        c3_3 = _det3_triton(m00, k01, k02, k01, m11, k12, k02, k12, m22)
        n3 = c3_0 * c3_0 + c3_1 * c3_1 + c3_2 * c3_2 + c3_3 * c3_3
        sel = n3 > bn
        b0 = tl.where(sel, c3_0, b0); b1 = tl.where(sel, c3_1, b1); b2 = tl.where(sel, c3_2, b2); b3 = tl.where(sel, c3_3, b3)
        bn = tl.where(sel, n3, bn)

        s2_scale = scale * scale
        s6_scale = s2_scale * s2_scale * s2_scale
        ok = ok & (bn > 1e-20 * s6_scale)
        inv_norm = 1.0 / libdevice.sqrt(tl.where(bn > 1e-300, bn, 1e-300))
        a = b0 * inv_norm
        b = b1 * inv_norm
        c = b2 * inv_norm
        d = b3 * inv_norm

        tl.store(R_ptr + base9 + 0, a * a + b * b - c * c - d * d, mask=mask)
        tl.store(R_ptr + base9 + 1, 2.0 * (b * c + a * d), mask=mask)
        tl.store(R_ptr + base9 + 2, 2.0 * (b * d - a * c), mask=mask)
        tl.store(R_ptr + base9 + 3, 2.0 * (b * c - a * d), mask=mask)
        tl.store(R_ptr + base9 + 4, a * a - b * b + c * c - d * d, mask=mask)
        tl.store(R_ptr + base9 + 5, 2.0 * (c * d + a * b), mask=mask)
        tl.store(R_ptr + base9 + 6, 2.0 * (b * d + a * c), mask=mask)
        tl.store(R_ptr + base9 + 7, 2.0 * (c * d - a * b), mask=mask)
        tl.store(R_ptr + base9 + 8, a * a - b * b - c * c + d * d, mask=mask)

    tl.store(lam_ptr + offs, lam, mask=mask)
    tl.store(ok_ptr + offs, ok, mask=mask)

@triton.jit
def _qcp_core_rmsd_kernel(
    Xg_ptr, Ac_ptr, gA_ptr, qi_ptr, kj_ptr, cp_ptr, out_ptr,
    Pn, Kp, nc, n_atoms,
    BLOCK_K: tl.constexpr,
):
    p = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = (p < Pn) & (k < Kp)
    if p >= Pn:
        return

    q_row = tl.load(qi_ptr + p)
    k_row = tl.load(kj_ptr + p)
    ga = tl.load(gA_ptr + k_row)

    q_base = q_row * n_atoms * 3
    ak_base = k_row * nc * 3
    cp_base = k * nc

    mx = tl.zeros([BLOCK_K], dtype=tl.float64)
    my = tl.zeros([BLOCK_K], dtype=tl.float64)
    mz = tl.zeros([BLOCK_K], dtype=tl.float64)
    for i in range(nc):
        atom_idx = tl.load(cp_ptr + cp_base + i, mask=mask_k, other=0)
        mx += tl.load(Xg_ptr + q_base + atom_idx * 3 + 0, mask=mask_k, other=0.0)
        my += tl.load(Xg_ptr + q_base + atom_idx * 3 + 1, mask=mask_k, other=0.0)
        mz += tl.load(Xg_ptr + q_base + atom_idx * 3 + 2, mask=mask_k, other=0.0)
    inv_nc = 1.0 / nc
    mx *= inv_nc
    my *= inv_nc
    mz *= inv_nc

    gB = tl.zeros([BLOCK_K], dtype=tl.float64)
    Sxx = tl.zeros([BLOCK_K], dtype=tl.float64)
    Sxy = tl.zeros([BLOCK_K], dtype=tl.float64)
    Sxz = tl.zeros([BLOCK_K], dtype=tl.float64)
    Syx = tl.zeros([BLOCK_K], dtype=tl.float64)
    Syy = tl.zeros([BLOCK_K], dtype=tl.float64)
    Syz = tl.zeros([BLOCK_K], dtype=tl.float64)
    Szx = tl.zeros([BLOCK_K], dtype=tl.float64)
    Szy = tl.zeros([BLOCK_K], dtype=tl.float64)
    Szz = tl.zeros([BLOCK_K], dtype=tl.float64)

    for i in range(nc):
        atom_idx = tl.load(cp_ptr + cp_base + i, mask=mask_k, other=0)
        bx = tl.load(Xg_ptr + q_base + atom_idx * 3 + 0, mask=mask_k, other=0.0) - mx
        by = tl.load(Xg_ptr + q_base + atom_idx * 3 + 1, mask=mask_k, other=0.0) - my
        bz = tl.load(Xg_ptr + q_base + atom_idx * 3 + 2, mask=mask_k, other=0.0) - mz
        gB += bx * bx + by * by + bz * bz
        ax = tl.load(Ac_ptr + ak_base + i * 3 + 0)
        ay = tl.load(Ac_ptr + ak_base + i * 3 + 1)
        az = tl.load(Ac_ptr + ak_base + i * 3 + 2)
        Sxx += bx * ax; Sxy += bx * ay; Sxz += bx * az
        Syx += by * ax; Syy += by * ay; Syz += by * az
        Szx += bz * ax; Szy += bz * ay; Szz += bz * az

    _, _, _, _, _, _, _, _, _, _, _, c2, c1, c0 = _qcp_horn_poly_triton(
        Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz
    )
    lam, _, _, _ = _qcp_max_eig_triton(c2, c1, c0, 0.5 * (ga + gB), mask_k, 0.0, 60, 1e-12, False, False)

    msd = (ga + gB - 2.0 * lam) * inv_nc
    rmsd = libdevice.sqrt(tl.where(msd > 0.0, msd, 0.0))
    tl.store(out_ptr + p * Kp + k, rmsd, mask=mask_k)

@triton.jit
def _pair_dup_vec_kernel(
    A_ptr, B_ptr, RA_ptr, RB_ptr,
    ga_ptr, gb_ptr, rA_ptr, rB_ptr, sA_ptr, sB_ptr,
    inv_maps_ptr, dup_ptr,
    b_len, R_len, M_len, n_atoms,
    th, th2, lower: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    ia = tl.program_id(0)
    ir = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = (ia < b_len) & (ir < R_len)
    if lower:
        mask_r = mask_r & (ir < ia)

    rad_diff2 = tl.zeros([BLOCK_R], dtype=tl.float64)
    for k in range(n_atoms):
        ra_k = tl.load(RA_ptr + ia * n_atoms + k)
        rb_k = tl.load(RB_ptr + ir * n_atoms + k, mask=mask_r, other=0.0)
        d = ra_k - rb_k
        rad_diff2 += d * d
    pair_ok = mask_r & (libdevice.sqrt(rad_diff2 / n_atoms) <= th)
    if tl.max(pair_ok.to(tl.int32), axis=0) == 0:
        return

    ga = tl.load(ga_ptr + ia)
    gb = tl.load(gb_ptr + ir, mask=mask_r, other=0.0)
    need = 0.5 * (ga + gb - n_atoms * th2)
    cs = 0.5 * (ga + gb)
    sa = tl.load(sA_ptr + ia)
    sb = tl.load(sB_ptr + ir, mask=mask_r, other=0.0)
    lb_thresh = (th2 * (1.0 + 1e-9) + 1e-12) * n_atoms

    b_base = ir * n_atoms * 3
    rb_base = ir * n_atoms
    a_base = ia * n_atoms * 3
    ra_base = ia * n_atoms

    found = tl.zeros([BLOCK_R], dtype=tl.int32) > 0
    m = 0
    while m < M_len and tl.max((pair_ok & (~found)).to(tl.int32), axis=0) != 0:
        inv_base = m * n_atoms
        m += 1
        live_r = pair_ok & (~found)
        dot_r = tl.zeros([BLOCK_R], dtype=tl.float64)
        for k in range(n_atoms):
            ik = tl.load(inv_maps_ptr + inv_base + k)
            ra_ik = tl.load(rA_ptr + ra_base + ik)
            rb_k = tl.load(rB_ptr + rb_base + k, mask=live_r, other=0.0)
            dot_r += ra_ik * rb_k
        surv = live_r & ((sa + sb - 2.0 * dot_r) <= lb_thresh)
        if tl.max(surv.to(tl.int32), axis=0) != 0:
            Sxx = tl.zeros([BLOCK_R], dtype=tl.float64)
            Sxy = tl.zeros([BLOCK_R], dtype=tl.float64)
            Sxz = tl.zeros([BLOCK_R], dtype=tl.float64)
            Syx = tl.zeros([BLOCK_R], dtype=tl.float64)
            Syy = tl.zeros([BLOCK_R], dtype=tl.float64)
            Syz = tl.zeros([BLOCK_R], dtype=tl.float64)
            Szx = tl.zeros([BLOCK_R], dtype=tl.float64)
            Szy = tl.zeros([BLOCK_R], dtype=tl.float64)
            Szz = tl.zeros([BLOCK_R], dtype=tl.float64)
            for k in range(n_atoms):
                ik = tl.load(inv_maps_ptr + inv_base + k)
                ax = tl.load(A_ptr + a_base + ik * 3 + 0)
                ay = tl.load(A_ptr + a_base + ik * 3 + 1)
                az = tl.load(A_ptr + a_base + ik * 3 + 2)
                bx = tl.load(B_ptr + b_base + k * 3 + 0, mask=surv, other=0.0)
                by = tl.load(B_ptr + b_base + k * 3 + 1, mask=surv, other=0.0)
                bz = tl.load(B_ptr + b_base + k * 3 + 2, mask=surv, other=0.0)
                Sxx += ax * bx; Sxy += ax * by; Sxz += ax * bz
                Syx += ay * bx; Syy += ay * by; Syz += ay * bz
                Szx += az * bx; Szy += az * by; Szz += az * bz

            trH = Sxx + Syy + Szz
            found = found | (surv & (trH >= need))
            rem = surv & (~found)
            if tl.max(rem.to(tl.int32), axis=0) != 0:
                _, _, _, _, _, _, _, _, _, _, h_norm2, c2, c1, c0 = _qcp_horn_poly_triton(
                    Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz
                )
                frob_ub = libdevice.sqrt(3.0 * h_norm2)
                lam = tl.where(cs < frob_ub, cs, frob_ub)
                active = rem & (lam >= need)
                if tl.max(active.to(tl.int32), axis=0) != 0:
                    _, _, found_qcp, _ = _qcp_max_eig_triton(
                        c2, c1, c0, lam, active, need, 60, 1e-14, True, False
                    )
                    found = found | found_qcp
    tl.store(dup_ptr + ia * R_len + ir, found, mask=mask_r)


def _det3(m):
    return (m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
            - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
            + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0]))


def _kabsch_qcp(H: np.ndarray, rot: bool = False):
    """Kabsch of correlation matrices H = sum_i q_i k_i^T (..., 3, 3) without SVD: returns (sv, R) with
    sv = s1 + s2 + sign(det H) s3 (so msd = (|q|^2 + |k|^2 - 2 sv) / n) and, if rot, the proper rotation R with
    q @ R ~ k (= U diag(1, 1, d) Vt). sv is the top eigenvalue of Horn's 4x4 key matrix found by Newton on its
    characteristic polynomial (Theobald 2005, QCP), R from its eigenvector (an adjugate column). Elementwise
    numpy: about 7x (sv) / 3x (sv + R) faster than numpy's batched SVD, agreeing to 1e-12. Entries where Newton
    has not converged (near-degenerate H: collinear points) are redone by SVD."""
    H = np.asarray(H, dtype=np.float64)
    Sxx, Sxy, Sxz = H[..., 0, 0], H[..., 0, 1], H[..., 0, 2]
    Syx, Syy, Syz = H[..., 1, 0], H[..., 1, 1], H[..., 1, 2]
    Szx, Szy, Szz = H[..., 2, 0], H[..., 2, 1], H[..., 2, 2]
    m00, m01, m02, m03 = Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx
    m11, m12, m13 = Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz
    m22, m23 = -Sxx + Syy - Szz, Syz + Szy
    m33 = -Sxx - Syy + Szz
    h_norm2 = (H * H).sum((-1, -2))
    c2 = -2.0 * h_norm2
    c1 = -8.0 * (Sxx * (Syy * Szz - Syz * Szy) - Sxy * (Syx * Szz - Syz * Szx) + Sxz * (Syx * Szy - Syy * Szx))
    s0, s1, s2 = m00 * m11 - m01 * m01, m00 * m12 - m01 * m02, m00 * m13 - m01 * m03
    s3, s4, s5 = m01 * m12 - m11 * m02, m01 * m13 - m11 * m03, m02 * m13 - m12 * m03
    k5, k4, k3 = m22 * m33 - m23 * m23, m12 * m33 - m13 * m23, m12 * m23 - m13 * m22
    k2, k1, k0 = m02 * m33 - m03 * m23, m02 * m23 - m03 * m22, m02 * m13 - m03 * m12
    c0 = s0 * k5 - s1 * k4 + s2 * k3 + s3 * k2 - s4 * k1 + s5 * k0
    lam = np.sqrt(np.maximum(3.0 * h_norm2, 0.0))      # >= s1 + s2 + s3: Newton descends to the top root
    scale = np.maximum(lam, 1e-300)
    ok = np.zeros(lam.shape, bool)
    for _ in range(50):
        l2 = lam * lam
        p = (l2 + c2) * l2 + c1 * lam + c0
        dp = (4.0 * l2 + 2.0 * c2) * lam + c1
        step = np.where(dp != 0, p / np.where(dp == 0, 1.0, dp), 0.0)
        lam = lam - step
        ok = np.abs(step) <= 1e-12 * scale
        if ok.all():
            break
    R = None
    if rot:
        K = np.stack([
            np.stack([m00, m01, m02, m03], -1),
            np.stack([m01, m11, m12, m13], -1),
            np.stack([m02, m12, m22, m23], -1),
            np.stack([m03, m13, m23, m33], -1)], -2)
        M = K - lam[..., None, None] * np.eye(4)
        rest = [[j for j in range(4) if j != i] for i in range(4)]
        best = bn = None
        for c in range(4):
            col = np.stack([((-1) ** (r + c)) * _det3(M[..., rest[c], :][..., :, rest[r]]) for r in range(4)], -1)
            nrm = (col * col).sum(-1)
            if best is None:
                best, bn = col, nrm
            else:
                sel = nrm > bn
                best = np.where(sel[..., None], col, best)
                bn = np.where(sel, nrm, bn)
        ok = ok & (bn > 1e-20 * scale ** 6)
        qv = best / np.sqrt(np.maximum(bn, 1e-300))[..., None]
        a, b, c, d = qv[..., 0], qv[..., 1], qv[..., 2], qv[..., 3]
        R = np.stack([
            np.stack([a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)], -1),
            np.stack([2 * (b * c - a * d), a * a - b * b + c * c - d * d, 2 * (c * d + a * b)], -1),
            np.stack([2 * (b * d + a * c), 2 * (c * d - a * b), a * a - b * b - c * c + d * d], -1)], -2)
    if not ok.all():
        bad = ~ok
        U, S, Vt = np.linalg.svd(H[bad])
        ds = np.sign(np.linalg.det(U @ Vt))
        lam[bad] = S[..., 0] + S[..., 1] + ds * S[..., 2]
        if rot:
            U[..., :, 2] *= ds[..., None]
            R[bad] = U @ Vt
    return lam, R


def _kabsch_qcp_gpu(H: Any):
    """Batched Kabsch QCP on CUDA (float32/float64): (sv, R) with q @ R ~ k; SVD where Newton has not converged
    or the adjugate column is degenerate. Dispatches unconditionally to the fused Triton QCP kernel."""
    if not isinstance(H, cp.ndarray):
        raise RuntimeError("GPU-only Triton build: _kabsch_qcp_gpu requires a CuPy GPU array")
    shape = H.shape[:-2]
    Hf = cp.ascontiguousarray(H.reshape(-1, 3, 3))
    N = Hf.shape[0]
    lam = cp.empty(N, dtype=Hf.dtype)
    R = cp.empty((N, 3, 3), dtype=Hf.dtype)
    ok = cp.empty(N, dtype=cp.bool_)
    if N > 0:
        BLOCK_N = 128
        gpu.launch(_kabsch_qcp_kernel, (gpu.cdiv(N, BLOCK_N),), Hf, lam, R, ok, N, COMPUTE_ROT=True, BLOCK_N=BLOCK_N)
        if not bool(ok.all()):
            bad = ~ok
            U, S, Vt = cp.linalg.svd(Hf[bad])
            ds = cp.sign(cp.linalg.det(U @ Vt))
            lam[bad] = S[..., 0] + S[..., 1] + ds * S[..., 2]
            U = U.copy(); U[..., :, 2] *= ds[..., None]
            R[bad] = U @ Vt
    return lam.reshape(shape), R.reshape(*shape, 3, 3)

OUT_DEDUP = True
MAX_AUTOMORPHISMS = 20000
#: select() takes rows in blocks: at most BLOCK_MAX rows, and about BLOCK_BUDGET (row, kept row, automorphism) triples
#: (2026-10-16, 46 complexes x 3 seeds: blocks win up to 48 automorphisms, row by row with early exit above 64).
BLOCK_MAX = 256
BLOCK_BUDGET = 1 << 21
BLOCK_MAX_MAPS = 48


class _FactoredDedupMixin:
    """Factored-symmetry, max-deviation verification, and mirror dedup methods for ``StructureDeduper``."""

    def _init_factored(self, mol, g, heavy, pos, adj):
        """Precompute factored symmetry structures for capped or factored dedup."""
        from rdkit import Chem

        n = len(heavy)
        self.adj = adj
        cls = np.array(list(Chem.CanonicalRankAtoms(g, breakTies=False)))
        self.cls = cls
        self.groups = [np.flatnonzero(cls == c) for c in np.unique(cls)]

        # Metal and donors
        _TM = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd La Hf Ta W Re Os Ir Pt Au Hg".split())
        metal_idx = getattr(mol, "metal_idx", None)
        metal_h = None
        if metal_idx is not None and (metal_idx - 1) in pos:
            metal_h = pos[metal_idx - 1]
        elif metal_idx is not None and metal_idx in pos:
            metal_h = pos[metal_idx]
        else:
            for k, h in enumerate(heavy):
                if mol.atoms[h].symbol in _TM:
                    metal_h = k
                    break
        if metal_h is None:
            metal_h = 0
        self.metal_h = metal_h
        donors = np.flatnonzero(adj[metal_h])
        self.donors = donors
        self.core = np.r_[metal_h, donors]

        # Donor within-element permutations (matching M2 method)
        syms_heavy = [mol.atoms[heavy[k]].symbol for k in range(n)]
        by_el: Dict[str, List[int]] = {}
        for p, d in enumerate(donors):
            by_el.setdefault(syms_heavy[d], []).append(p)
        els = list(by_el.values())
        donor_perms = []
        base = np.arange(len(donors))

        def _backtrack(idx, cur):
            if len(donor_perms) >= 5040:
                return
            if idx == len(els):
                donor_perms.append(cur.copy())
                return
            v = els[idx]
            for p in itertools.permutations(v):
                for src, dst in zip(v, p):
                    cur[src] = dst
                _backtrack(idx + 1, cur)
                if len(donor_perms) >= 5040:
                    break

        _backtrack(0, base)
        self.donor_perms = np.array(donor_perms)
        if len(donor_perms):
            self.core_perms_arr = np.concatenate([
                np.full((len(donor_perms), 1), metal_h, dtype=int),
                donors[self.donor_perms],
            ], axis=1)
        else:
            self.core_perms_arr = self.core[None, :]

        # Identify local terminal rotor groups (tBu, CF3, isopropyl, etc.)
        self.local_rotors = []
        for c in range(n):
            nbrs = np.flatnonzero(adj[c])
            term_by_el: Dict[str, List[int]] = {}
            for nb in nbrs:
                if len(np.flatnonzero(adj[nb])) == 1:
                    term_by_el.setdefault(mol.atoms[heavy[nb]].symbol, []).append(nb)
            for _el, leaves in term_by_el.items():
                if len(leaves) >= 2:
                    self.local_rotors.append(np.array(leaves, dtype=int))

    def factored_sym_upper(self, A: np.ndarray, B: np.ndarray, th: float = 0.5, max_dev_th: float = 0.0):
        """Compute an upper bound on symmetric RMSD between A and B over real graph automorphisms (M2 method).
        Returns (is_dup, min_rmsd, min_max_dev, best_permutation).
        """
        from scipy.optimize import linear_sum_assignment

        n = A.shape[0]
        n_core = len(self.core)

        # Batch Kabsch on core starts
        Ac = A[self.core] - A[self.core].mean(0)
        B_core = B[self.core_perms_arr]
        Bc = B_core - B_core.mean(1, keepdims=True)
        H = np.einsum("kni,nj->kij", Bc, Ac)
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.einsum("kij,kjl->kil", U, Vt)))
        s = S[..., 0] + S[..., 1] + d * S[..., 2]
        msd = ((Ac**2).sum() + (Bc**2).sum((-1, -2)) - 2 * s) / max(1, n_core)
        core_rmsd = np.sqrt(np.clip(msd, 0, None))

        max_core_th = th * np.sqrt(n / max(1, n_core)) + 0.25
        order = np.argsort(core_rmsd)
        good = [idx for idx in order if core_rmsd[idx] <= max_core_th][:36]
        if not good:
            good = order[:2].tolist()

        best = (np.inf, np.inf, False, None)
        for idx in good:
            perm_core = self.core_perms_arr[idx]
            _, _, R = kabsch_fit_np(A[self.core], B[perm_core])
            Bt = (B - B[perm_core].mean(0)) @ R + A[self.core].mean(0)
            mp = np.arange(n)
            for _ in range(12):
                new = np.empty(n, dtype=int)
                for gidx in self.groups:
                    if len(gidx) == 1:
                        new[gidx] = gidx
                        continue
                    C = ((A[gidx][:, None, :] - Bt[gidx][None, :, :]) ** 2).sum(-1)
                    r_, c_ = linear_sum_assignment(C)
                    new[gidx[r_]] = gidx[c_]
                r, md, R = kabsch_fit_np(A, B[new])
                Bt = (B - B[new].mean(0)) @ R + A.mean(0)
                if np.array_equal(new, mp):
                    break
                mp = new
            is_auto = bool((self.adj[np.ix_(mp, mp)] == self.adj).all())
            if is_auto and r < best[0]:
                best = (r, md, True, mp)
                if r <= th and (max_dev_th <= 0.0 or md <= max_dev_th):
                    return True, r, md, mp
            elif not best[2] and r < best[0]:
                best = (r, md, False, mp)

        is_dup = bool(best[2] and (best[0] <= th) and (max_dev_th <= 0.0 or best[1] <= max_dev_th))
        return is_dup, best[0], best[1], best[3]

    def _verify_max_dev(
        self,
        A_row: Any,
        B_cand: Any,
        th: float,
        max_dev_th: float,
        cand_indices: Any,
    ):
        """Find if any row in B_cand has Kabsch RMSD <= th and max per-atom deviation <= max_dev_th."""
        A_np = gpu.to_cpu(A_row)
        B_np = gpu.to_cpu(B_cand)
        cand_np = gpu.to_cpu(cand_indices)
        Ac = A_np - A_np.mean(0)
        best_match = (False, np.inf, np.inf, None)
        maps_np = gpu.to_cpu(self.maps)
        for c_idx in range(B_np.shape[0]):
            Bc = B_np[c_idx] - B_np[c_idx].mean(0)
            for m in maps_np:
                Bm = Bc[m]
                H = Bm.T @ Ac
                U, S, Vt = np.linalg.svd(H)
                d = np.sign(np.linalg.det(U @ Vt))
                R = U @ np.diag([1.0, 1.0, d]) @ Vt
                diff = Ac - (Bm @ R)
                sq = (diff ** 2).sum(-1)
                r = float(np.sqrt(sq.mean()))
                if r <= th:
                    md = float(np.sqrt(sq.max()))
                    if md <= max_dev_th:
                        return True, r, md, int(cand_np[c_idx])
                    if r < best_match[1]:
                        best_match = (False, r, md, int(cand_np[c_idx]))
        return best_match[0], best_match[1], best_match[2], best_match[3]

    def _factored_groups(self):
        """Cached partition of symmetry classes into small (<=4, permutation-enumerated) and large (>4, LSAP)."""
        cached = getattr(self, "_fgroups_cache", None)
        if cached is not None:
            return cached
        small, large = [], []
        for gidx in self.groups:
            g = len(gidx)
            if 2 <= g <= 4:
                pm = np.array(list(itertools.permutations(range(g))))
                small.append((gidx, np.arange(g)[None, :], pm))
            elif g > 4:
                large.append(gidx)
        cached = (small, large)
        self._fgroups_cache = cached
        return cached

    def _factored_match(
        self,
        K: np.ndarray,
        q: np.ndarray,
        th: float,
        max_dev_th: float = 0.0,
        stop_on_any: bool = False,
    ):
        """Batched ``factored_sym_upper`` over candidates ``K (P, n, 3)`` against ``q (n, 3)``.

        Returns ``(matched_mask, first_hit)``, where ``first_hit`` is ``(p_idx, rmsd, max_dev, perm)``
        for the lowest-index confirmed duplicate or ``None`` if no duplicate matched.
        """
        from scipy.optimize import linear_sum_assignment

        K = np.asarray(K, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        P, n = K.shape[0], K.shape[1]
        if P == 0:
            return np.zeros(0, bool), None
        Kc_off = K.mean(1, keepdims=True)
        if np.abs(Kc_off).max() > 1e-12:
            K = K - Kc_off
        qc_off = q.mean(0)
        if np.abs(qc_off).max() > 1e-12:
            q = q - qc_off

        core, cp = self.core, self.core_perms_arr
        nc = max(1, len(core))
        K_core = K[:, core]
        Ac = K_core - K_core.mean(1, keepdims=True)
        q_cp = q[cp]
        Bc = q_cp - q_cp.mean(1, keepdims=True)
        fast = DEDUP_CAPPED_FAST and self.is_capped
        core_rmsd = self._core_rmsd_gpu(Ac, Bc) if fast else None
        if core_rmsd is None:
            H0_core = np.einsum("kni,pnj->pkij", Bc, Ac)
            sv = self._kabsch_np(H0_core, rot=False)[0]
            msd = ((Ac ** 2).sum((-1, -2))[:, None] + (Bc ** 2).sum((-1, -2))[None] - 2.0 * sv) / nc
            core_rmsd = np.sqrt(np.maximum(msd, 0.0))
        max_core_th = th * np.sqrt(n / nc) + 0.25

        n_starts = 36
        order = np.argsort(core_rmsd, axis=1)
        sorted_vals = np.take_along_axis(core_rmsd, order, axis=1)
        order36 = order[:, :n_starts]
        cnt = (sorted_vals[:, :n_starts] <= max_core_th).sum(1)
        pidx_l, sidx_l = [], []
        k_fallback = min(2, order.shape[1])
        for p_ in range(P):
            c = int(cnt[p_])
            if c > 0:
                pidx_l.append(np.full(c, p_, dtype=int))
                sidx_l.append(order36[p_, :c])
            else:
                pidx_l.append(np.full(k_fallback, p_, dtype=int))
                sidx_l.append(order[p_, :k_fallback])
        pidx = np.concatenate(pidx_l)
        perm = cp[np.concatenate(sidx_l)]
        Ks = K[pidx]
        Sn = len(pidx)

        Kc_s = Ks[:, core]
        Qp_s = q[perm]
        Kc_m = Kc_s.mean(1, keepdims=True)
        Qp_m = Qp_s.mean(1, keepdims=True)
        H0 = (Qp_s - Qp_m).swapaxes(1, 2) @ (Kc_s - Kc_m)
        R0 = self._kabsch_np(H0, rot=True)[1]
        Bt = (q[None] - Qp_m) @ R0 + Kc_m
        if fast and len(pidx):
            # screen at the start rotation: every atom to its nearest same-class atom (no assignment, no refit);
            # starts above _SCREEN_CUT * th never reach a round (duplicates on 15 capped parents: <= 1.75 * th)
            sing = np.array([g[0] for g in self.groups if len(g) == 1], dtype=int)
            d2 = ((Ks[:, sing] - Bt[:, sing]) ** 2).sum((-1, -2)) if len(sing) else np.zeros(len(pidx))
            for gidx in self.groups:
                if len(gidx) > 1:
                    Ka, Bb = Ks[:, gidx], Bt[:, gidx]
                    C = (Ka * Ka).sum(-1)[:, :, None] + (Bb * Bb).sum(-1)[:, None, :] - 2.0 * (Ka @ Bb.swapaxes(1, 2))
                    d2 = d2 + np.maximum(C.min(2), 0.0).sum(1)
            ok = d2 <= (_SCREEN_CUT * th) ** 2 * n
            if not ok.all():
                pidx, Ks, Bt = pidx[ok], Ks[ok], Bt[ok]
                Sn = len(pidx)

        small_groups, large_groups = self._factored_groups()
        K_norm2 = (Ks * Ks).sum((-1, -2))
        q_norm2 = float((q * q).sum())
        Kg_small = [(gidx, ar_g, pm, Ks[:, gidx], (Ks[:, gidx] ** 2).sum(-1)[:, :, None]) for gidx, ar_g, pm in small_groups]
        Kg_large = [(gidx, Ks[:, gidx], (Ks[:, gidx] ** 2).sum(-1)[:, :, None]) for gidx in large_groups]
        Qs = np.broadcast_to(q[None], (Sn, n, 3))
        Q_norm2 = np.full(Sn, q_norm2, dtype=np.float64)
        matched = np.zeros(P, bool)
        first_hit = self._run_assignment_rounds(
            pidx, Ks, Qs, K_norm2, Q_norm2, Kg_small, Kg_large, Bt,
            th, max_dev_th, matched, stop_on_any=stop_on_any, fast_cut=fast, use_qcp=DEDUP_QCP,
        )
        return matched, first_hit

    @staticmethod
    def _kabsch_np(H: np.ndarray, rot: bool = False, use_qcp: bool = DEDUP_QCP):
        if use_qcp:
            return _kabsch_qcp(H, rot=rot)
        U, S_, Vt = np.linalg.svd(H)
        ds = np.sign(np.linalg.det(U @ Vt))
        sv = S_[..., 0] + S_[..., 1] + ds * S_[..., 2]
        if not rot:
            return sv, None
        U_r = U.copy()
        U_r[..., :, 2] *= ds[..., None]
        return sv, U_r @ Vt

    def _run_assignment_rounds(
        self, pidx, Ks, Qs, K_norm2, Q_norm2, Kg_small, Kg_large, Bt,
        th: float, max_dev_th: float, matched: np.ndarray, hits=None,
        stop_on_any: bool = False, fast_cut: bool = True, use_qcp: bool = True,
    ):
        from scipy.optimize import linear_sum_assignment

        n = Ks.shape[1]
        base_row = np.arange(n)
        mp = np.broadcast_to(base_row, (len(pidx), n)).copy()
        adj = self.adj
        first_hit = None

        for rnd in range(12):
            L = len(pidx)
            if not L:
                break
            ar = np.arange(L)[:, None]
            new = np.broadcast_to(base_row, (L, n)).copy()
            for gidx, ar_g, pm, Ka, Ka2 in Kg_small:
                Bb = Bt[:, gidx]
                C = Ka2 + (Bb * Bb).sum(-1)[:, None, :] - 2.0 * (Ka @ Bb.swapaxes(1, 2))
                new[:, gidx] = gidx[pm[C[:, ar_g, pm].sum(-1).argmin(1)]]
            for gidx, Ka, Ka2 in Kg_large:
                Bb = Bt[:, gidx]
                C = Ka2 + (Bb * Bb).sum(-1)[:, None, :] - 2.0 * (Ka @ Bb.swapaxes(1, 2))
                cols = C.argmin(2)
                srt = np.sort(cols, 1)
                for b in np.flatnonzero((srt != np.arange(len(gidx))[None]).any(1)):
                    cols[b] = linear_sum_assignment(C[b])[1]
                new[:, gidx] = gidx[cols]

            H = Qs[ar, new].swapaxes(1, 2) @ Ks
            sv, Rall = self._kabsch_np(H, rot=True, use_qcp=use_qcp)
            rr = np.sqrt(np.maximum((K_norm2 + Q_norm2 - 2.0 * sv) / n, 0.0))
            stable = (new == mp).all(1)
            cand_ok = (rr <= th) & (stable if rnd < 11 else True)
            if cand_ok.any():
                for s_idx in np.flatnonzero(cand_ok):
                    p_row = int(pidx[s_idx])
                    if matched[p_row]:
                        continue
                    m = new[s_idx]
                    if (adj[m][:, m] == adj).all():
                        md_h = float(np.sqrt(((Ks[s_idx] - Qs[s_idx][m] @ Rall[s_idx]) ** 2).sum(-1).max()))
                        if max_dev_th > 0.0 and md_h > max_dev_th:
                            continue
                        matched[p_row] = True
                        if hits is not None:
                            hits[p_row] = (float(rr[s_idx]), md_h, m.copy())
                        if first_hit is None or p_row < first_hit[0]:
                            first_hit = (p_row, float(rr[s_idx]), md_h, m.copy())
                        if stop_on_any:
                            return first_hit

            keep_s = (~stable) & (~matched[pidx])
            if fast_cut and rnd == 0:
                keep_s &= rr <= _ROUND0_CUT * th
            if not keep_s.any():
                break
            if int(keep_s.sum()) > 1:
                seen = set()
                for idx_k in np.flatnonzero(keep_s):
                    key = (int(pidx[idx_k]), new[idx_k].tobytes())
                    if key in seen:
                        keep_s[idx_k] = False
                    else:
                        seen.add(key)

            Rn = Rall[keep_s]
            Qs = Qs[keep_s]
            Bt = Qs @ Rn
            pidx = pidx[keep_s]
            Ks = Ks[keep_s]
            K_norm2 = K_norm2[keep_s]
            Q_norm2 = Q_norm2[keep_s]
            mp = new[keep_s]
            Kg_small = [(gidx, ar_g, pm, Ka[keep_s], Ka2[keep_s]) for gidx, ar_g, pm, Ka, Ka2 in Kg_small]
            Kg_large = [(gidx, Ka[keep_s], Ka2[keep_s]) for gidx, Ka, Ka2 in Kg_large]

        return first_hit

    def _core_rmsd_gpu(self, Ac: np.ndarray, Bc: np.ndarray, budget: int = 1 << 21):
        """Core RMSD of every candidate (Ac (P, nc, 3), centred) against every core permutation of the query
        (Bc (Kp, nc, 3), centred) -> (P, Kp), via the fused Triton QCP kernel on CUDA."""
        A = gpu.as_gpu(Ac, cp.float64)
        B = gpu.as_gpu(Bc, cp.float64)
        ga, gb = (A * A).sum(axis=(-1, -2)), (B * B).sum(axis=(-1, -2))
        nc = max(1, A.shape[1])
        out = []
        step = max(1, budget // max(1, B.shape[0]))
        BLOCK_N = 128
        for s in range(0, A.shape[0], step):
            H = cp.ascontiguousarray(cp.einsum("kni,pnj->pkij", B, A[s:s + step]))
            Hf = H.reshape(-1, 3, 3)
            N = Hf.shape[0]
            lam = cp.empty(N, dtype=cp.float64)
            ok = cp.empty(N, dtype=cp.bool_)
            if N > 0:
                gpu.launch(_kabsch_qcp_kernel, (gpu.cdiv(N, BLOCK_N),), Hf, lam, lam, ok, N, COMPUTE_ROT=False, BLOCK_N=BLOCK_N)
            sv = lam.reshape(H.shape[:-2])
            out.append(cp.sqrt(cp.maximum((ga[s:s + step, None] + gb[None] - 2.0 * sv) / nc, 0.0)))
        return gpu.to_cpu(cp.concatenate(out))

    def _factored_pairs(self, Xn: np.ndarray, Xg: cp.ndarray, qi: np.ndarray, kj: np.ndarray, th: float,
                        max_dev_th: float = 0.0, budget: int = 1 << 21):
        """The factored symmetric match (dedup_capped_fast form of _factored_match) for many (query, candidate) row
        pairs at once: is row qi[p] a duplicate of row kj[p]? Xn (B, n, 3) centred rows (numpy float64), Xg the same
        on the GPU."""
        Pn = len(qi)
        matched = np.zeros(Pn, bool)
        hits: Dict[int, Tuple[float, float, np.ndarray]] = {}
        if Pn == 0:
            return matched, hits
        n = Xn.shape[1]
        core = cp.asarray(np.asarray(self.core), dtype=cp.int64)
        cp_arr = cp.ascontiguousarray(cp.asarray(np.asarray(self.core_perms_arr), dtype=cp.int64))
        Kp, nc = cp_arr.shape[0], max(1, len(self.core))
        Ac_all = Xg[:, core]
        Ac_all = cp.ascontiguousarray(Ac_all - Ac_all.mean(axis=1, keepdims=True))
        gA = cp.ascontiguousarray((Ac_all * Ac_all).sum(axis=(-1, -2)))
        max_core_th = th * np.sqrt(n / nc) + 0.25
        n_starts = min(36, Kp)
        k_fallback = min(2, Kp)
        sing = cp.asarray([g[0] for g in self.groups if len(g) == 1], dtype=cp.int64)
        multi = [cp.asarray(g, dtype=cp.int64) for g in self.groups if len(g) > 1]
        scut2 = (_SCREEN_CUT * th) ** 2 * n
        qi_t = cp.ascontiguousarray(cp.asarray(qi, dtype=cp.int64))
        kj_t = cp.ascontiguousarray(cp.asarray(kj, dtype=cp.int64))
        surv_p, surv_Bt = [], []
        step = max(1, min(budget // max(1, Kp), (1 << 15) // n_starts))
        Xg_c = cp.ascontiguousarray(Xg)
        for s0 in range(0, Pn, step):
            qc, kc = qi_t[s0:s0 + step], kj_t[s0:s0 + step]
            c_len = qc.shape[0]
            core_rmsd = cp.empty((c_len, Kp), dtype=cp.float64)
            BLOCK_K = 16 if Kp <= 16 else 32
            gpu.launch(
                _qcp_core_rmsd_kernel,
                (c_len, gpu.cdiv(Kp, BLOCK_K)),
                Xg_c, Ac_all, gA, qc, kc, cp_arr, core_rmsd,
                c_len, Kp, nc, n, BLOCK_K=BLOCK_K,
            )
            order = cp.argsort(core_rmsd, axis=1)
            vals = cp.take_along_axis(core_rmsd, order, axis=1)
            cnt = (vals[:, :n_starts] <= max_core_th).sum(axis=1)
            cnt = cp.where(cnt > 0, cnt, k_fallback)
            sel = cp.arange(n_starts)[None] < cnt[:, None]
            ploc, srank = cp.nonzero(sel)
            if ploc.shape[0] == 0:
                continue
            perm = cp_arr[order[ploc, srank]]
            pg = ploc + s0
            Ks = Xg[kj_t[pg]]
            Qs = Xg[qi_t[pg]]
            Kc_s = Ks[:, core]
            Qp_s = cp.take_along_axis(Qs, perm[..., None], axis=1)
            Kc_m = Kc_s.mean(axis=1, keepdims=True)
            Qp_m = Qp_s.mean(axis=1, keepdims=True)
            H0 = cp.swapaxes(Qp_s - Qp_m, 1, 2) @ (Kc_s - Kc_m)
            R0 = _kabsch_qcp_gpu(H0)[1]
            Bt = (Qs - Qp_m) @ R0 + Kc_m
            d2 = ((Ks[:, sing] - Bt[:, sing]) ** 2).sum(axis=(-1, -2)) if len(sing) else cp.zeros(len(pg), dtype=Xg.dtype)
            for gidx in multi:
                Ka, Bb = Ks[:, gidx], Bt[:, gidx]
                C = (Ka * Ka).sum(axis=-1)[:, :, None] + (Bb * Bb).sum(axis=-1)[:, None, :] - 2.0 * (Ka @ cp.swapaxes(Bb, 1, 2))
                d2 = d2 + cp.maximum(C.min(axis=2), 0.0).sum(axis=1)
            ok = d2 <= scut2
            if bool(ok.any()):
                surv_p.append(gpu.to_cpu(pg[ok]))
                surv_Bt.append(gpu.to_cpu(Bt[ok]))
        if not surv_p:
            return matched, hits
        pidx = np.concatenate(surv_p)
        Bt = np.concatenate(surv_Bt)
        self._pair_rounds(Xn, qi, kj, pidx, Bt, th, max_dev_th, matched, hits)
        return matched, hits

    def _pair_rounds(self, Xn, qi, kj, pidx, Bt, th, max_dev_th, matched, hits):
        """The alternating assignment / rotation rounds of _factored_match (fast form) for starts of many pairs:
        pidx (S,) pair of each start, Bt (S, n, 3) the query rotated onto the candidate. Fills matched / hits."""
        Ks, Qs = Xn[kj[pidx]], Xn[qi[pidx]]
        small_groups, large_groups = self._factored_groups()
        X_norm2 = (Xn * Xn).sum((-1, -2))
        Xn_sq = (Xn * Xn).sum(-1)[:, :, None]
        k_rows = kj[pidx]
        Kg_small = [(gidx, ar_g, pm, Ks[:, gidx], Xn_sq[k_rows][:, gidx]) for gidx, ar_g, pm in small_groups]
        Kg_large = [(gidx, Ks[:, gidx], Xn_sq[k_rows][:, gidx]) for gidx in large_groups]
        self._run_assignment_rounds(
            pidx, Ks, Qs, X_norm2[kj[pidx]], X_norm2[qi[pidx]], Kg_small, Kg_large, Bt,
            th, max_dev_th, matched, hits=hits, stop_on_any=False, fast_cut=True, use_qcp=True,
        )

    def _select_capped_pairs(self, X, Rr, order: List[int], k: Optional[int], th: float,
                             max_dev_th: float):
        """select() on capped groups with dedup_capped_fast and a CUDA device: greedy best-first in blocks of rows,
        every (block row, kept row) pair and every earlier-row pair inside the block decided at once by
        _factored_pairs (the same pair decision as the row-by-row loop, so the same selection). None on any failure
        (the caller runs the row-by-row loop)."""
        try:
            Xn = gpu.to_cpu(X).astype(np.float64)
            Xg = gpu.as_gpu(Xn, cp.float64)
            Rg = gpu.as_gpu(Rr, cp.float64)
            keep: List[int] = []
            merges: List[Dict[str, Any]] = []
            pos = 0
            while pos < len(order) and (k is None or len(keep) < k):
                blk = order[pos:pos + BLOCK_MAX]
                pos += len(blk)
                if keep:
                    bt = cp.asarray(blk, dtype=cp.int64)
                    kt = cp.asarray(keep, dtype=cp.int64)
                    rad = cp.sqrt(cp.mean((Rg[bt][:, None] - Rg[kt][None]) ** 2, axis=-1)) <= th
                    bi, ki = [gpu.to_cpu(t) for t in cp.nonzero(rad)]
                    if len(bi):
                        blk_a, keep_a = np.asarray(blk), np.asarray(keep)
                        m, hits = self._factored_pairs(Xn, Xg, blk_a[bi], keep_a[ki], th, max_dev_th)
                        drop = {}
                        for p in np.flatnonzero(m):
                            b, kk = int(bi[p]), int(ki[p])
                            if b not in drop or kk < drop[b][0]:
                                drop[b] = (kk, p)
                        for b, (kk, p) in sorted(drop.items()):
                            r_val, md_val, perm = hits[p]
                            merges.append({"dropped": blk[b], "kept": keep[kk], "rmsd": r_val, "max_dev": md_val, "perm": perm.tolist()})
                        blk = [i for a, i in enumerate(blk) if a not in drop]
                W = None
                if len(blk) > 1:
                    bt = cp.asarray(blk, dtype=cp.int64)
                    Rb = Rg[bt]
                    rad = cp.sqrt(cp.mean((Rb[:, None] - Rb[None]) ** 2, axis=-1)) <= th
                    rad = cp.tril(rad, k=-1)
                    ai, bj = [gpu.to_cpu(t) for t in cp.nonzero(rad)]
                    W = {}
                    if len(ai):
                        blk_a = np.asarray(blk)
                        m, hits = self._factored_pairs(Xn, Xg, blk_a[ai], blk_a[bj], th, max_dev_th)
                        for p in np.flatnonzero(m):
                            W.setdefault(int(ai[p]), {})[int(bj[p])] = hits[p]
                new_local: List[int] = []
                for a, i in enumerate(blk):
                    if W is not None and a in W:
                        hit_b = [b for b in new_local if b in W[a]]
                        if hit_b:
                            r_val, md_val, perm = W[a][hit_b[0]]
                            merges.append({"dropped": i, "kept": blk[hit_b[0]], "rmsd": r_val, "max_dev": md_val, "perm": perm.tolist()})
                            continue
                    new_local.append(a)
                    keep.append(i)
                    if k is not None and len(keep) >= k:
                        break
            if keep and k is not None and len(keep) >= k:
                rank = {i: r for r, i in enumerate(order)}
                last = rank[keep[-1]]
                merges = [m for m in merges if rank.get(m["dropped"], 0) <= last]
            self.merges.extend(merges)
            return keep
        except Exception as exc:
            log.debug("batched capped dedup failed (%s); row-by-row", exc)
            return None

    def _factored_any(self, K: np.ndarray, q: np.ndarray, th: float, max_dev_th: float = 0.0) -> np.ndarray:
        """factored_sym_upper(K[p], q, th, max_dev_th)[0] for every row p of K (P, n, 3) at once (P2R out_mirror)."""
        return self._factored_match(K, q, th, max_dev_th=max_dev_th, stop_on_any=False)[0]

    def _mirror_pairs(self, A, B, RA, RB, th: float,
                      max_dev_th: float) -> Optional[List[bool]]:
        """mirror_dup on capped groups with dedup_capped_fast and a CUDA device: every radial candidate pair
        (reflection i, row j) with j >= i decided at once by _factored_pairs (the pair test of the row loop below;
        RMSD(refl i, j) = RMSD(refl j, i), so each unordered pair is tested once and a match marks both rows).
        QEZVEY (A100): 156 row calls, 15.9 s -> one batched call. None on any failure (the caller
        runs the row loop)."""
        try:
            K = A.shape[0]
            Xn = np.concatenate([gpu.to_cpu(B).astype(np.float64), gpu.to_cpu(A).astype(np.float64)])
            Xg = gpu.as_gpu(Xn, cp.float64)
            RAg, RBg = gpu.as_gpu(RA, cp.float64), gpu.as_gpu(RB, cp.float64)
            rad = cp.triu(cp.sqrt(cp.mean((RAg[:, None] - RBg[None]) ** 2, axis=-1)) <= th)
            ii, jj = [gpu.to_cpu(t) for t in cp.nonzero(rad)]
            dup = np.zeros(K, bool)
            if len(ii):
                m, _ = self._factored_pairs(Xn, Xg, ii + K, jj, th, max_dev_th)
                dup[ii[m]] = True
                dup[jj[m]] = True
            return dup.tolist()
        except Exception as exc:
            log.debug("batched mirror dedup failed (%s); row loop", exc)
            return None

    @timing.timed("dedup")
    def mirror_dup(self, Xk, th: Optional[float] = None) -> List[bool]:
        """P2R out_mirror: for each kept row i (K, N, 3), is its reflection a duplicate of any kept row j (itself
        included)? The decision of `any_dup(reflections, Xk)`, faster on capped automorphism groups, where the
        per-pair factored test (about 0.04 s per pair, exhaustive when the pair is distinct, which most reflections
        are) dominated (HEDJOP 32 rows: 809 pairs, 36 s per seed):
          * the batched capped-map test (_full_any) on every radial candidate, as in any_dup;
          * the factored test once per unordered pair (RMSD(refl i, j) = RMSD(refl j, i), so a pair found for i also
            marks j): all pairs in one batched GPU call (_mirror_pairs), else all candidates of a row per call
            (_factored_any).
        Uncapped groups: exactly any_dup"""
        if th is None:
            th = DEDUP_RMSD
        K = Xk.shape[0]
        if K == 0:
            return []
        Xk = gpu.as_gpu(Xk, self.dtype)
        refl = cp.asarray([-1.0, 1.0, 1.0], dtype=self.dtype)
        if not (DEDUP_FACTORED and (self.is_capped or DEDUP_MAX_DEV > 0.0)):
            return self.any_dup(Xk * refl, Xk, th)
        A = (Xk * refl)[:, self.heavy]; A = A - A.mean(axis=1, keepdims=True)
        B = Xk[:, self.heavy]; B = B - B.mean(axis=1, keepdims=True)
        RA, RB = self._radial(cp.nan_to_num(A)), self._radial(cp.nan_to_num(B))
        dup = [False] * K
        cands = []
        for i in range(K):
            cand = cp.nonzero(cp.sqrt(cp.mean((RB - RA[i]) ** 2, axis=1)) <= th)[0]
            cands.append(cand)
            if cand.size and not (DEDUP_CAPPED_FAST and self.is_capped) and self._scan_any(A[i], B[cand], RA[i], RB[cand], th):
                dup[i] = DEDUP_MAX_DEV <= 0.0 or self._verify_max_dev(A[i], B[cand], th, DEDUP_MAX_DEV, cand)[0]
        if not self.is_capped:
            return dup
        if DEDUP_CAPPED_FAST:
            batched = self._mirror_pairs(A, B, RA, RB, th, DEDUP_MAX_DEV)
            if batched is not None:
                return batched
        An, Bn = gpu.to_cpu(A), gpu.to_cpu(B)
        tested = set()
        for i in range(K):
            if dup[i]:
                continue
            cl = [int(c) for c in gpu.to_cpu(cands[i]).tolist() if (min(i, int(c)), max(i, int(c))) not in tested]
            tested.update((min(i, c), max(i, c)) for c in cl)
            if not cl:
                continue
            hit = self._factored_any(Bn[cl], An[i], th, DEDUP_MAX_DEV)
            if hit.any():
                dup[i] = True
                for c in np.asarray(cl)[hit]:
                    dup[int(c)] = True
        return dup


class StructureDeduper(_FactoredDedupMixin):
    def __init__(self, mol, device=None, dtype=cp.float64):
        from rdkit import Chem

        self.device = str(device) if device is not None else "cuda"
        self.dtype = dtype
        heavy = [k for k, a in enumerate(mol.atoms) if a.symbol != "H"]
        pos = {a: k for k, a in enumerate(heavy)}
        rw = Chem.RWMol()
        for k in heavy:
            at = Chem.Atom(mol.atoms[k].symbol)
            at.SetNoImplicit(True)
            rw.AddAtom(at)
        seen = set()
        adj = np.zeros((len(heavy), len(heavy)), dtype=bool)
        for b in mol.bonds:
            i, j = b.i - 1, b.j - 1
            if i in pos and j in pos and i != j and (min(i, j), max(i, j)) not in seen:
                seen.add((min(i, j), max(i, j)))
                rw.AddBond(pos[i], pos[j], Chem.BondType.SINGLE)
                adj[pos[i], pos[j]] = adj[pos[j], pos[i]] = True
        g = rw.GetMol()
        g.UpdatePropertyCache(strict=False)
        n = g.GetNumAtoms()
        m = np.array(g.GetSubstructMatches(g, uniquify=False, maxMatches=MAX_AUTOMORPHISMS, useChirality=False))
        self.is_capped = bool(m.shape[0] >= MAX_AUTOMORPHISMS)
        ident = np.arange(n)
        m = m[~(m == ident).all(1)] if len(m) else np.zeros((0, n), dtype=int)
        self.maps = cp.asarray(np.vstack([ident[None], m]), dtype=cp.int64)
        # inverse of every automorphism: sum_n f(A[n], B[map[n]]) == sum_k f(A[inv[k]], B[k]), which turns the scans
        # over automorphisms below into plain matrix products instead of (rows, maps, atoms, 3) gathers
        self.inv_maps = cp.argsort(self.maps, axis=1)
        self.heavy = cp.asarray(heavy, dtype=cp.int64)
        cls = np.array(list(Chem.CanonicalRankAtoms(g, breakTies=False)))
        order = np.argsort(cls, kind="stable")
        self._rad_order = cp.asarray(order, dtype=cp.int64)
        self._rad_groups = [cp.asarray(np.flatnonzero(cls[order] == c), dtype=cp.int64) for c in np.unique(cls)]
        self.merges: List[Dict[str, Any]] = []
        self._init_factored(mol, g, heavy, pos, adj)

    # --- geometry helpers -------------------------------------------------------------------------------------------
    def _radial(self, X):
        r = cp.linalg.norm(X - X.mean(axis=1, keepdims=True), axis=-1)[:, self._rad_order]
        out = r.copy()
        for g in self._rad_groups:
            out[:, g] = cp.sort(r[:, g], axis=1)
        return out

    def _ensure_gpu_cache(self, dev=None):
        if getattr(self, "_gpu_cache_dev", None) != dev:
            self._gpu_cache_dev = dev
            self._gpu_heavy = self.heavy
            self._gpu_inv_maps = cp.ascontiguousarray(self.inv_maps.astype(cp.int32))
            self._gpu_rad_order = self._rad_order
            self._gpu_rad_groups = self._rad_groups

    def _pair_dup_triton(self, A, B, RA, RB, th: float, lower: bool = False):
        """Fused Triton GPU kernel implementation of _pair_dup: evaluates class-sorted radial bound, per-automorphism
        radial bound, 3x3 correlation H, and 4x4 Horn QCP Newton decision in registers without materializing (b, m, R)
        intermediate tensors in HBM."""
        self._ensure_gpu_cache("cuda")
        Ag = cp.ascontiguousarray(gpu.as_gpu(A, cp.float64))
        Bg = cp.ascontiguousarray(gpu.as_gpu(B, cp.float64))
        RAg = cp.ascontiguousarray(gpu.as_gpu(RA, cp.float64))
        RBg = cp.ascontiguousarray(gpu.as_gpu(RB, cp.float64))
        b_len, n_atoms, _ = Ag.shape
        R_len = Bg.shape[0]
        M_len = self._gpu_inv_maps.shape[0]
        ga = cp.ascontiguousarray((Ag * Ag).sum(axis=(-1, -2)))
        gb = cp.ascontiguousarray((Bg * Bg).sum(axis=(-1, -2)))
        rA = cp.ascontiguousarray(cp.linalg.norm(Ag, axis=-1))
        rB = cp.ascontiguousarray(cp.linalg.norm(Bg, axis=-1))
        sA = cp.ascontiguousarray((rA * rA).sum(axis=1))
        sB = cp.ascontiguousarray((rB * rB).sum(axis=1))
        dup = cp.zeros((b_len, R_len), dtype=cp.bool_)
        if b_len > 0 and R_len > 0:
            BLOCK_R = 64
            grid = (b_len, gpu.cdiv(R_len, BLOCK_R))
            gpu.launch(
                _pair_dup_vec_kernel, grid,
                Ag, Bg, RAg, RBg, ga, gb, rA, rB, sA, sB, self._gpu_inv_maps, dup,
                b_len, R_len, M_len, n_atoms, float(th), float(th * th), lower=lower,
                BLOCK_R=BLOCK_R
            )
        return dup

    def _pair_dup(self, A, B, RA, RB, th, lower=False):
        """(a, r) duplicate matrix (b, R) bool via fused Triton kernel on CUDA."""
        return self._pair_dup_triton(A, B, RA, RB, th, lower=lower)

    def _scan_any(self, a, Bc, ra, RBc, th: float) -> bool:
        """Is any automorphism of the stored list (the capped list on capped groups) within th of a row of Bc? The
        scan over inv_maps that the factored match is OR-ed with when dedup_capped_fast is off, and the whole test on
        uncapped groups."""
        return bool(cp.any(self._pair_dup_triton(a[None], Bc, ra[None], RBc, float(th), lower=False)))

    # --- public -----------------------------------------------------------------------------------------------------
    @timing.timed("dedup")
    def any_dup(self, Xq, Xk, th: Optional[float] = None,
                factored_rows: Optional[List[List[int]]] = None) -> List[bool]:
        """For each query row (Q, N, 3): is it a duplicate of ANY reference row (K, N, 3)?"""
        if th is None:
            th = DEDUP_RMSD
        Q = Xq.shape[0]
        if Q == 0 or Xk.shape[0] == 0:
            return [False] * Q
        if not (DEDUP_FACTORED and (self.is_capped or DEDUP_MAX_DEV > 0.0)):
            self._ensure_gpu_cache("cuda")
            Ag = gpu.as_gpu(Xq, cp.float64)[:, self._gpu_heavy]
            Ag = cp.ascontiguousarray(Ag - Ag.mean(axis=1, keepdims=True))
            Bg = gpu.as_gpu(Xk, cp.float64)[:, self._gpu_heavy]
            Bg = cp.ascontiguousarray(Bg - Bg.mean(axis=1, keepdims=True))
            rA_ord = cp.linalg.norm(cp.nan_to_num(Ag), axis=-1)[:, self._gpu_rad_order]
            rB_ord = cp.linalg.norm(cp.nan_to_num(Bg), axis=-1)[:, self._gpu_rad_order]
            RAg, RBg = rA_ord.copy(), rB_ord.copy()
            for g in self._gpu_rad_groups:
                RAg[:, g] = cp.sort(rA_ord[:, g], axis=1)
                RBg[:, g] = cp.sort(rB_ord[:, g], axis=1)
            dup = self._pair_dup_triton(Ag, Bg, RAg, RBg, float(th), lower=False)
            return gpu.to_cpu(dup.any(axis=1)).tolist()
        A = gpu.as_gpu(Xq, self.dtype)[:, self.heavy]; A = A - A.mean(axis=1, keepdims=True)
        B = gpu.as_gpu(Xk, self.dtype)[:, self.heavy]; B = B - B.mean(axis=1, keepdims=True)
        RA, RB = self._radial(cp.nan_to_num(A)), self._radial(cp.nan_to_num(B))
        A_np = gpu.to_cpu(A) if self.is_capped else None
        B_np = gpu.to_cpu(B) if self.is_capped else None
        out: List[bool] = []
        for i in range(Q):
            cand = cp.nonzero(cp.sqrt(cp.mean((RB - RA[i]) ** 2, axis=1)) <= th)[0]
            dup = False
            if cand.size:
                if self.is_capped:
                    cl = gpu.to_cpu(cand).tolist()
                    if factored_rows is not None:
                        allowed = set(factored_rows[i]); cl = [c for c in cl if c in allowed]
                    if cl:
                        dup = bool(self._factored_match(B_np[cl], A_np[i], th=th, max_dev_th=DEDUP_MAX_DEV, stop_on_any=True)[0].any())
                    if not dup and not DEDUP_CAPPED_FAST and self._scan_any(A[i], B[cand], RA[i], RB[cand], th):
                        dup = DEDUP_MAX_DEV <= 0.0 or self._verify_max_dev(A[i], B[cand], th, DEDUP_MAX_DEV, cand)[0]
                elif self._scan_any(A[i], B[cand], RA[i], RB[cand], th):
                    dup = DEDUP_MAX_DEV <= 0.0 or self._verify_max_dev(A[i], B[cand], th, DEDUP_MAX_DEV, cand)[0]
            out.append(bool(dup))
        return out

    def _select_uncapped_triton(self, X_all, score, k: Optional[int], th: float) -> List[int]:
        """Full GPU Triton path for uncapped exact selection (max_dev_th <= 0): sorts and centers heavy-atom
        coordinates on GPU, computes class-sorted radial profiles, and runs _pair_dup_vec_kernel in blocks."""
        M_len = self.inv_maps.shape[0]
        self._ensure_gpu_cache("cuda")
        Xg = gpu.as_gpu(X_all, cp.float64)[:, self._gpu_heavy]
        Xg = cp.ascontiguousarray(Xg - Xg.mean(axis=1, keepdims=True))
        finite = cp.isfinite(Xg).all(axis=-1).all(axis=-1)
        sc = np.nan_to_num(gpu.to_cpu(score).astype(np.float64), nan=-1e30, posinf=-1e30, neginf=-1e30)
        finite_cpu = gpu.to_cpu(finite)
        sc = np.where(finite_cpu, sc, -1e31)
        order = [int(i) for i in np.argsort(-sc, kind="stable").tolist() if bool(finite_cpu[i])]
        if not order:
            return []
        order_t = cp.asarray(order, dtype=cp.int64)
        Xs = cp.ascontiguousarray(Xg[order_t])
        B, n_atoms, _ = Xs.shape
        rA = cp.ascontiguousarray(cp.linalg.norm(cp.nan_to_num(Xs), axis=-1))
        r_ord = rA[:, self._gpu_rad_order]
        Rs = r_ord.copy()
        for g in self._gpu_rad_groups:
            Rs[:, g] = cp.sort(r_ord[:, g], axis=1)
        Rs = cp.ascontiguousarray(Rs)
        ga = cp.ascontiguousarray((Xs * Xs).sum(axis=(-1, -2)))
        sA = cp.ascontiguousarray((rA * rA).sum(axis=1))

        keep: List[int] = []
        keep_pos: List[int] = []
        pos = 0
        BLOCK_R = 64
        while pos < B and (k is None or len(keep) < k):
            nb = int(min(BLOCK_MAX, max(1, BLOCK_BUDGET // (M_len * max(len(keep), 1)))))
            blk_end = min(B, pos + nb)
            blk_pos = list(range(pos, blk_end))
            pos = blk_end
            if keep_pos:
                bt = cp.asarray(blk_pos, dtype=cp.int64)
                kt = cp.asarray(keep_pos, dtype=cp.int64)
                b_len, R_len = len(blk_pos), len(keep_pos)
                dup_k = cp.zeros((b_len, R_len), dtype=cp.bool_)
                gpu.launch(
                    _pair_dup_vec_kernel, (b_len, gpu.cdiv(R_len, BLOCK_R)),
                    cp.ascontiguousarray(Xs[bt]),
                    cp.ascontiguousarray(Xs[kt]),
                    cp.ascontiguousarray(Rs[bt]),
                    cp.ascontiguousarray(Rs[kt]),
                    cp.ascontiguousarray(ga[bt]),
                    cp.ascontiguousarray(ga[kt]),
                    cp.ascontiguousarray(rA[bt]),
                    cp.ascontiguousarray(rA[kt]),
                    cp.ascontiguousarray(sA[bt]),
                    cp.ascontiguousarray(sA[kt]),
                    self._gpu_inv_maps, dup_k,
                    b_len, R_len, M_len, n_atoms, float(th), float(th * th), lower=False,
                    BLOCK_R=BLOCK_R
                )
                dk = gpu.to_cpu(dup_k.any(axis=1)).tolist()
                if DEDUP_FACTORED:
                    for bp, d in zip(blk_pos, dk):
                        if d:
                            self.merges.append({"dropped": order[bp], "kept": keep[0], "rmsd": float(th), "max_dev": 0.0})
                blk_pos = [bp for bp, d in zip(blk_pos, dk) if not d]
            W = None
            if len(blk_pos) > 1:
                bt = cp.asarray(blk_pos, dtype=cp.int64)
                b_len = len(blk_pos)
                dup_w = cp.zeros((b_len, b_len), dtype=cp.bool_)
                Xb = cp.ascontiguousarray(Xs[bt])
                Rb = cp.ascontiguousarray(Rs[bt])
                gab = cp.ascontiguousarray(ga[bt])
                rAb = cp.ascontiguousarray(rA[bt])
                sAb = cp.ascontiguousarray(sA[bt])
                gpu.launch(
                    _pair_dup_vec_kernel, (b_len, gpu.cdiv(b_len, BLOCK_R)),
                    Xb, Xb, Rb, Rb, gab, gab, rAb, rAb, sAb, sAb, self._gpu_inv_maps, dup_w,
                    b_len, b_len, M_len, n_atoms, float(th), float(th * th), lower=True,
                    BLOCK_R=BLOCK_R
                )
                W = gpu.to_cpu(dup_w).tolist()
            new: List[int] = []
            for a, bp in enumerate(blk_pos):
                if W is not None and any(W[a][r] for r in new):
                    if DEDUP_FACTORED:
                        self.merges.append({"dropped": order[bp], "kept": keep[-1] if keep else -1, "rmsd": float(th), "max_dev": 0.0})
                    continue
                new.append(a)
                keep_pos.append(bp)
                keep.append(order[bp])
                if k is not None and len(keep) >= k:
                    break
        return keep

    @timing.timed("dedup")
    def select(self, X_all, score, k: Optional[int], th: Optional[float] = None,
               exact: bool = True, max_dev_th: Optional[float] = None) -> List[int]:
        """X_all (B, N, 3) full coordinates, score (B,) higher = better. Returns up to k distinct row indices, best first."""
        if th is None:
            th = DEDUP_RMSD
        if max_dev_th is None:
            max_dev_th = DEDUP_MAX_DEV
        self.merges = []
        if (not self.is_capped or not DEDUP_FACTORED) and max_dev_th <= 0.0:
            return self._select_uncapped_triton(X_all, score, k, float(th))

        X = gpu.as_gpu(X_all, self.dtype)[:, self.heavy]
        X = X - X.mean(axis=1, keepdims=True)
        sc = np.nan_to_num(gpu.to_cpu(score).astype(np.float64), nan=-1e30, posinf=-1e30, neginf=-1e30)
        finite = gpu.to_cpu(cp.isfinite(X).all(axis=-1).all(axis=-1))
        sc = np.where(finite, sc, -1e31)
        Rr = self._radial(cp.nan_to_num(X))
        order = [int(i) for i in np.argsort(-sc, kind="stable").tolist() if bool(finite[i])]
        keep: List[int] = []

        if DEDUP_CAPPED_FAST and self.is_capped:
            kp = self._select_capped_pairs(X, Rr, order, k, th, max_dev_th)
            if kp is not None:
                return kp
            self.merges = []
        X_np = gpu.to_cpu(X) if self.is_capped else None
        for i in order:
            if keep:
                kt = cp.asarray(keep, dtype=cp.int64)
                cand = kt[cp.sqrt(cp.mean((Rr[kt] - Rr[i]) ** 2, axis=1)) <= th]
                if cand.size:
                    dup_found = False
                    if self.is_capped:
                        cl = gpu.to_cpu(cand).tolist()
                        _, hit = self._factored_match(X_np[cl], X_np[i], th=th, max_dev_th=max_dev_th, stop_on_any=True)
                        if hit is not None:
                            p_row, r_val, md_val, perm = hit
                            k_idx = cl[p_row]
                            dup_found = True
                            self.merges.append({"dropped": i, "kept": k_idx, "rmsd": r_val, "max_dev": md_val, "perm": perm.tolist()})
                            log.info("Factored dedup merge: row %d merged into row %d (RMSD=%.4f A, max_dev=%.4f A)", i, k_idx, r_val, md_val)
                    if not dup_found and (not self.is_capped or not DEDUP_CAPPED_FAST) and self._scan_any(X[i], X[cand], Rr[i], Rr[cand], th):
                        if max_dev_th > 0.0:
                            ok_dev, r_val, md_val, k_matched = self._verify_max_dev(X[i], X[cand], th, max_dev_th, cand)
                            if ok_dev:
                                dup_found = True
                                self.merges.append({"dropped": i, "kept": k_matched, "rmsd": r_val, "max_dev": md_val})
                        else:
                            dup_found = True
                            self.merges.append({"dropped": i, "kept": int(gpu.to_cpu(cand[0]).item()), "rmsd": float(th), "max_dev": 0.0})
                    if dup_found:
                        continue
            keep.append(i)
            if k is not None and len(keep) >= k:
                break
        return keep


_CACHE: "OrderedDict[tuple, StructureDeduper]" = OrderedDict()



def deduper_for(mol, device=None) -> StructureDeduper:
    """StructureDeduper(mol, device), reused while the heavy-atom graph is the same (the automorphism enumeration
    is the expensive part of building one: up to MAX_AUTOMORPHISMS substructure matches). Several seeds or embeds of
    one complex share it. Keyed by element symbols and bonds, so a different complex never gets a stale one."""
    dev = str(device) if device is not None else "cuda"
    key = (dev, tuple(a.symbol for a in mol.atoms),
           tuple(sorted((min(b.i, b.j), max(b.i, b.j)) for b in mol.bonds)))
    dd = _CACHE.get(key)
    if dd is None:
        dd = _CACHE[key] = StructureDeduper(mol, device=dev)
        while len(_CACHE) > 4:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    return dd


def select_distinct(mol, X_all, score, k: Optional[int], device=None,
                    th: Optional[float] = None, max_dev_th: Optional[float] = None) -> List[int]:
    """`StructureDeduper(mol).select(X_all, score, k, th=th, max_dev_th=max_dev_th)` on the GPU. Out of memory: the cache
    is released and the call repeated once. A failure that remains degrades (the generator never fails): the rows are
    returned best first without duplicate removal, with a warning."""
    dev = str(device) if device is not None else "cuda"
    try:
        return deduper_for(mol, dev).select(X_all, score, k, th=th, max_dev_th=max_dev_th)
    except gpu.OOMError as exc:
        log.warning("output dedup on %s out of memory (%s); clearing the cache and retrying", dev, str(exc)[:200])
        gpu.empty_cache()
        try:
            return deduper_for(mol, dev).select(X_all, score, k, th=th, max_dev_th=max_dev_th)
        except Exception as exc2:  # noqa: BLE001 - degrade below
            exc = exc2
    except Exception as exc_:  # noqa: BLE001 - degrade below
        exc = exc_
    log.warning("output dedup failed (%s: %s); rows kept best first WITHOUT duplicate removal",
                type(exc).__name__, str(exc).splitlines()[0][:200] if str(exc) else "")
    sc = np.nan_to_num(gpu.to_cpu(score).astype(np.float64), nan=-np.inf)
    finite = np.isfinite(gpu.to_cpu(gpu.as_gpu(X_all, cp.float64)).reshape(len(sc), -1)).all(axis=1)
    order = [int(i) for i in np.argsort(-sc, kind="stable") if finite[i] and np.isfinite(sc[i])]
    return order if k is None else order[:k]
