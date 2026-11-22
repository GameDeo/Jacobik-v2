"""Pure Triton + analytical-backward MACE-POLAR-1 inference engine (CUDA, float32 & float64).

Bypasses e3nn, TorchScript, torch_geometric, graph_longrange Python/dense-(4n)^2 overhead, and
PyTorch autograd for PolarMACE models with scalar hidden irreps (MACE-POLAR-1-S). Every algebraic
transformation is exact up to floating-point rounding (< 1e-10 eV / < 1e-12 eV/A in float64;
< 2e-5 eV / < 5e-5 eV/A in float32):

- Precomputes all constant species tables (node_embedding, skip_tp, linear_up, linear_res,
  source/target embeddings projected through layer-0 weights of conv_tp_weights, density_fn, and
  field_dependent_charges_maps), eliminating the (E, 1032) concatenation and reducing layer-0
  edge MLPs from 1032->64 to 8->64.
- Species-factored interaction fusion: merges conv_tp_weights.net[9], conv_tp, scatter_sum, and
  linear_1 into CSR-bucketed A[m, v, s, 65] (no atomics, fixed summation order, zero (E, 8192)
  mji or (E, 2048) tp_weights HBM allocation) followed by 4 contiguous cuBLAS GEMMs against
  lazily cached per-species M_fused[s, l].
- Fused Triton kernels (forward + analytical backward) for:
  * edge vectors, Bessel basis, polynomial cutoff, l=0..3 spherical harmonics, atomic-free force
    scatter using reverse-edge indices;
  * edge density MLP (8->64->1 + LayerNorm + SiLU + tanh(x^2)*cutoff);
  * fused bias + LayerNorm + SiLU (D=64, 128);
  * div_density + add_res + equivariant Gate;
  * register-only 77-triple SymmetricContraction (eliminating the dense (N, 512, 16, 16) einsum);
  * tiled-SRAM realspace monopole electrostatics (alpha+beta fused in 1 pass, N x N instead of
    (4N)^2 when dipoles are structurally zero, energy + forces + dE/dq fused in 1 pass, no
    atomics).
- Algebraically merges linear maps in field_dependent_charges_maps (W_dot @ W_nf/W_pot/W_chg)
  and local_electron_energy (W_dotq @ W_q, W_dotv @ W_v; eliminates both 512x512 GEMMs).
- Pure CuPy + Triton runtime (zero PyTorch); the forward and backward pass is CUDA-graph capturable (no host syncs).
  Run to run, float64 results agree to rounding; float32 to ~1e-5 eV (cuBLAS GEMM order).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

import numpy as np
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from jacobik.core import gpu, mace_weights

cp = gpu.cp

# Periodic table symbols (Z=1..118) for zero-ASE symbol->Z lookup
_ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
_SYM_TO_Z: Dict[str, int] = {sym: i + 1 for i, sym in enumerate(_ELEMENTS)}

# --------------------------------------------------------------------------- Constants

_S3 = tl.constexpr(math.sqrt(3.0))
_S5 = tl.constexpr(math.sqrt(5.0))
_S7 = tl.constexpr(math.sqrt(7.0))
_S15 = tl.constexpr(math.sqrt(15.0))
_S15_2 = tl.constexpr(0.5 * math.sqrt(15.0))
_C30 = tl.constexpr(math.sqrt(35.0 / 8.0))
_C31 = tl.constexpr(math.sqrt(105.0))
_C32 = tl.constexpr(math.sqrt(21.0 / 8.0))
_C33 = tl.constexpr(0.5 * math.sqrt(7.0))
_C35 = tl.constexpr(0.5 * math.sqrt(105.0))
_TWO_OVER_SQRT_PI = tl.constexpr(2.0 / math.sqrt(math.pi))
_FIELD_CONSTANT = 180.95128022746667

# Canonical 77 unordered triples (i <= j <= k) with nonzero U_matrix_3 for l_max=3 -> 0e
_CANONICAL_U3_TRIPLES: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5), (0, 6, 6), (0, 7, 7),
    (0, 8, 8), (0, 9, 9), (0, 10, 10), (0, 11, 11), (0, 12, 12), (0, 13, 13), (0, 14, 14), (0, 15, 15),
    (1, 1, 6), (1, 1, 8), (1, 2, 5), (1, 3, 4), (1, 4, 13), (1, 4, 15), (1, 5, 12), (1, 5, 14),
    (1, 6, 11), (1, 7, 10), (1, 8, 9), (1, 8, 11), (2, 2, 6), (2, 3, 7), (2, 4, 10), (2, 5, 11),
    (2, 6, 12), (2, 7, 13), (2, 8, 14), (3, 3, 6), (3, 3, 8), (3, 4, 9), (3, 4, 11), (3, 5, 10),
    (3, 6, 13), (3, 7, 12), (3, 7, 14), (3, 8, 13), (3, 8, 15), (4, 4, 6), (4, 5, 7), (4, 9, 13),
    (4, 10, 12), (4, 11, 13), (4, 11, 15), (5, 5, 6), (5, 5, 8), (5, 9, 14), (5, 10, 13), (5, 10, 15),
    (5, 11, 12), (5, 11, 14), (6, 6, 6), (6, 7, 7), (6, 8, 8), (6, 9, 9), (6, 11, 11), (6, 12, 12),
    (6, 13, 13), (6, 15, 15), (7, 7, 8), (7, 9, 10), (7, 10, 11), (7, 12, 13), (7, 13, 14), (7, 14, 15),
    (8, 9, 11), (8, 11, 11), (8, 12, 14), (8, 13, 13), (8, 13, 15),
)


def _tl_dtype(dt):
    return tl.float64 if dt == cp.float64 else tl.float32


def _mm(A: cp.ndarray, B: cp.ndarray, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    """CUDAGraph-safe matrix multiply ``A @ B`` via ``gpu.gemm``."""
    if not A.flags.c_contiguous:
        A = cp.ascontiguousarray(A)
    if not B.flags.c_contiguous:
        B = cp.ascontiguousarray(B)
    if out is None:
        out = cp.empty((A.shape[0], B.shape[1]), dtype=A.dtype)
    return gpu.gemm(A, B, out)


def _mm_tb(A: cp.ndarray, B: cp.ndarray, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    """CUDAGraph-safe matrix multiply ``A @ B.T`` via ``gpu.gemm(..., trans_b=True)``."""
    if not A.flags.c_contiguous:
        A = cp.ascontiguousarray(A)
    if not B.flags.c_contiguous:
        B = cp.ascontiguousarray(B)
    if out is None:
        out = cp.empty((A.shape[0], B.shape[0]), dtype=A.dtype)
    return gpu.gemm(A, B, out, trans_b=True)


def _mv(A: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
    """CUDAGraph-safe matrix-vector multiply ``A @ v`` via ``gpu.gemm``."""
    if not A.flags.c_contiguous:
        A = cp.ascontiguousarray(A)
    out = cp.empty((A.shape[0], 1), dtype=A.dtype)
    gpu.gemm(A, cp.ascontiguousarray(v[:, None]), out)
    return out.reshape(-1)


# --------------------------------------------------------------------------- Kernel 1: Edge Geom + Bessel + Cutoff + SH

@triton.jit
def _edge_geom_fwd_kernel(
    pos_ptr, src_ptr, dst_ptr, bw_ptr, sc_ptr,
    d_ptr, cut_ptr, bes_ptr, y_ptr,
    E,
    BLOCK_E: tl.constexpr, DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    m = offs < E

    r_max = tl.load(sc_ptr + 0)
    inv_rmax = tl.load(sc_ptr + 1)
    bessel_pref = tl.load(sc_ptr + 2)

    u_idx = tl.load(src_ptr + offs, mask=m, other=0)
    v_idx = tl.load(dst_ptr + offs, mask=m, other=0)

    vx = tl.load(pos_ptr + v_idx * 3 + 0, mask=m, other=0.0) - tl.load(pos_ptr + u_idx * 3 + 0, mask=m, other=0.0)
    vy = tl.load(pos_ptr + v_idx * 3 + 1, mask=m, other=0.0) - tl.load(pos_ptr + u_idx * 3 + 1, mask=m, other=0.0)
    vz = tl.load(pos_ptr + v_idx * 3 + 2, mask=m, other=0.0) - tl.load(pos_ptr + u_idx * 3 + 2, mask=m, other=0.0)

    d2 = vx * vx + vy * vy + vz * vz
    d = libdevice.sqrt(d2)
    zero = tl.zeros((BLOCK_E,), dtype=DTYPE)
    active = m & (d < r_max) & (d > 0.0)
    d_safe = tl.where(active, d, 1.0)
    inv_d = 1.0 / d_safe

    # PolynomialCutoff p=5: 1 - 21 u^5 + 35 u^6 - 15 u^7
    u = d * inv_rmax
    u2 = u * u
    u5 = u2 * u2 * u
    u6 = u5 * u
    u7 = u6 * u
    cut = tl.where(active, 1.0 - 21.0 * u5 + 35.0 * u6 - 15.0 * u7, zero)

    tl.store(d_ptr + offs, tl.where(active, d, zero), mask=m)
    tl.store(cut_ptr + offs, cut, mask=m)

    # BesselBasis (num_basis=8)
    pref_inv_d = bessel_pref * inv_d
    for k in tl.static_range(8):
        wk = tl.load(bw_ptr + k)
        bk = tl.where(active, pref_inv_d * libdevice.sin(wk * d), zero)
        tl.store(bes_ptr + offs * 8 + k, bk, mask=m)

    # Spherical harmonics l=0..3 on permuted unit vector (a, b, c) = (vy, vz, vx) / d
    a = tl.where(active, vy * inv_d, zero)
    b = tl.where(active, vz * inv_d, zero)
    c = tl.where(active, vx * inv_d, zero)
    a2 = a * a
    b2 = b * b
    c2 = c * c
    a2c2 = a2 + c2
    f4b2 = 4.0 * b2 - a2c2

    y0 = tl.where(active, zero + 1.0, zero)
    y1 = _S3 * a
    y2 = _S3 * b
    y3 = _S3 * c
    y4 = _S15 * a * c
    y5 = _S15 * a * b
    y6 = _S5 * (b2 - 0.5 * a2c2)
    y7 = _S15 * b * c
    y8 = _S15_2 * (c2 - a2)
    y9 = _C30 * a * (3.0 * c2 - a2)
    y10 = _C31 * a * b * c
    y11 = _C32 * a * f4b2
    y12 = _C33 * b * (2.0 * b2 - 3.0 * a2c2)
    y13 = _C32 * c * f4b2
    y14 = _C35 * b * (c2 - a2)
    y15 = _C30 * c * (c2 - 3.0 * a2)

    base_y = offs * 16
    tl.store(y_ptr + base_y + 0, y0, mask=m)
    tl.store(y_ptr + base_y + 1, y1, mask=m)
    tl.store(y_ptr + base_y + 2, y2, mask=m)
    tl.store(y_ptr + base_y + 3, y3, mask=m)
    tl.store(y_ptr + base_y + 4, y4, mask=m)
    tl.store(y_ptr + base_y + 5, y5, mask=m)
    tl.store(y_ptr + base_y + 6, y6, mask=m)
    tl.store(y_ptr + base_y + 7, y7, mask=m)
    tl.store(y_ptr + base_y + 8, y8, mask=m)
    tl.store(y_ptr + base_y + 9, y9, mask=m)
    tl.store(y_ptr + base_y + 10, y10, mask=m)
    tl.store(y_ptr + base_y + 11, y11, mask=m)
    tl.store(y_ptr + base_y + 12, y12, mask=m)
    tl.store(y_ptr + base_y + 13, y13, mask=m)
    tl.store(y_ptr + base_y + 14, y14, mask=m)
    tl.store(y_ptr + base_y + 15, y15, mask=m)


@triton.jit
def _edge_geom_bwd_kernel(
    pos_ptr, src_ptr, dst_ptr, bw_ptr, sc_ptr,
    d_ptr, g_cut_ptr, g_bes_ptr, g_y_ptr,
    g_vec_ptr,
    E,
    BLOCK_E: tl.constexpr, DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    m = offs < E

    inv_rmax = tl.load(sc_ptr + 1)
    bessel_pref = tl.load(sc_ptr + 2)

    d = tl.load(d_ptr + offs, mask=m, other=0.0)
    active = m & (d > 0.0)
    zero = tl.zeros((BLOCK_E,), dtype=DTYPE)
    d_safe = tl.where(active, d, 1.0)
    inv_d = 1.0 / d_safe
    inv_d2 = inv_d * inv_d

    # Radial derivative: cutoff + bessel
    u = d * inv_rmax
    u2 = u * u
    u4 = u2 * u2
    one_m_u = 1.0 - u
    dcut_dd = -105.0 * u4 * one_m_u * one_m_u * inv_rmax
    g_d = tl.load(g_cut_ptr + offs, mask=active, other=0.0) * dcut_dd

    for k in tl.static_range(8):
        wk = tl.load(bw_ptr + k)
        gbk = tl.load(g_bes_ptr + offs * 8 + k, mask=active, other=0.0)
        wd = wk * d
        db_dd = bessel_pref * (wk * d * libdevice.cos(wd) - libdevice.sin(wd)) * inv_d2
        g_d = g_d + gbk * db_dd

    # Edge unit vector (a, b, c) = (vy, vz, vx) / d
    u_idx = tl.load(src_ptr + offs, mask=active, other=0)
    v_idx = tl.load(dst_ptr + offs, mask=active, other=0)
    vx = tl.load(pos_ptr + v_idx * 3 + 0, mask=active, other=0.0) - tl.load(pos_ptr + u_idx * 3 + 0, mask=active, other=0.0)
    vy = tl.load(pos_ptr + v_idx * 3 + 1, mask=active, other=0.0) - tl.load(pos_ptr + u_idx * 3 + 1, mask=active, other=0.0)
    vz = tl.load(pos_ptr + v_idx * 3 + 2, mask=active, other=0.0) - tl.load(pos_ptr + u_idx * 3 + 2, mask=active, other=0.0)

    a = vy * inv_d
    b = vz * inv_d
    c = vx * inv_d
    a2 = a * a
    b2 = b * b
    c2 = c * c

    base_y = offs * 16
    g1 = tl.load(g_y_ptr + base_y + 1, mask=active, other=0.0)
    g2 = tl.load(g_y_ptr + base_y + 2, mask=active, other=0.0)
    g3 = tl.load(g_y_ptr + base_y + 3, mask=active, other=0.0)
    g4 = tl.load(g_y_ptr + base_y + 4, mask=active, other=0.0)
    g5 = tl.load(g_y_ptr + base_y + 5, mask=active, other=0.0)
    g6 = tl.load(g_y_ptr + base_y + 6, mask=active, other=0.0)
    g7 = tl.load(g_y_ptr + base_y + 7, mask=active, other=0.0)
    g8 = tl.load(g_y_ptr + base_y + 8, mask=active, other=0.0)
    g9 = tl.load(g_y_ptr + base_y + 9, mask=active, other=0.0)
    g10 = tl.load(g_y_ptr + base_y + 10, mask=active, other=0.0)
    g11 = tl.load(g_y_ptr + base_y + 11, mask=active, other=0.0)
    g12 = tl.load(g_y_ptr + base_y + 12, mask=active, other=0.0)
    g13 = tl.load(g_y_ptr + base_y + 13, mask=active, other=0.0)
    g14 = tl.load(g_y_ptr + base_y + 14, mask=active, other=0.0)
    g15 = tl.load(g_y_ptr + base_y + 15, mask=active, other=0.0)

    # dL/da, dL/db, dL/dc
    ga = (
        _S3 * g1
        + _S15 * (c * g4 + b * g5)
        - _S5 * a * g6
        - _S15 * a * g8
        + _C30 * (3.0 * (c2 - a2) * g9 - 6.0 * a * c * g15)
        + _C31 * b * c * g10
        + _C32 * ((4.0 * b2 - 3.0 * a2 - c2) * g11 - 2.0 * a * c * g13)
        - 6.0 * _C33 * a * b * g12
        - 2.0 * _C35 * a * b * g14
    )
    gb = (
        _S3 * g2
        + _S15 * (a * g5 + c * g7)
        + 2.0 * _S5 * b * g6
        + _C31 * a * c * g10
        + 8.0 * _C32 * b * (a * g11 + c * g13)
        + _C33 * (6.0 * b2 - 3.0 * (a2 + c2)) * g12
        + _C35 * (c2 - a2) * g14
    )
    gc = (
        _S3 * g3
        + _S15 * (a * g4 + b * g7)
        - _S5 * c * g6
        + _S15 * c * g8
        + _C30 * (6.0 * a * c * g9 + 3.0 * (c2 - a2) * g15)
        + _C31 * a * b * g10
        + _C32 * (-2.0 * a * c * g11 + (4.0 * b2 - a2 - 3.0 * c2) * g13)
        - 6.0 * _C33 * b * c * g12
        + 2.0 * _C35 * b * c * g14
    )

    dot = ga * a + gb * b + gc * c
    gvy = g_d * a + (ga - dot * a) * inv_d
    gvz = g_d * b + (gb - dot * b) * inv_d
    gvx = g_d * c + (gc - dot * c) * inv_d

    tl.store(g_vec_ptr + offs * 3 + 0, tl.where(active, gvx, zero), mask=m)
    tl.store(g_vec_ptr + offs * 3 + 1, tl.where(active, gvy, zero), mask=m)
    tl.store(g_vec_ptr + offs * 3 + 2, tl.where(active, gvz, zero), mask=m)


@triton.jit
def _scatter_edge_forces_kernel(
    g_vec_ptr, rev_ptr, node_ptr, g_pos_es_ptr, forces_ptr,
    N, DTYPE: tl.constexpr,
):
    v = tl.program_id(0)
    if v >= N:
        return
    start = tl.load(node_ptr + v)
    end = tl.load(node_ptr + v + 1)
    fx = tl.zeros((), dtype=DTYPE)
    fy = tl.zeros((), dtype=DTYPE)
    fz = tl.zeros((), dtype=DTYPE)
    for e in range(start, end):
        r = tl.load(rev_ptr + e)
        fx = fx + tl.load(g_vec_ptr + r * 3 + 0) - tl.load(g_vec_ptr + e * 3 + 0)
        fy = fy + tl.load(g_vec_ptr + r * 3 + 1) - tl.load(g_vec_ptr + e * 3 + 1)
        fz = fz + tl.load(g_vec_ptr + r * 3 + 2) - tl.load(g_vec_ptr + e * 3 + 2)
    # Subtract electrostatic dE/dpos (forces = -dE/dpos)
    fx = fx - tl.load(g_pos_es_ptr + v * 3 + 0)
    fy = fy - tl.load(g_pos_es_ptr + v * 3 + 1)
    fz = fz - tl.load(g_pos_es_ptr + v * 3 + 2)
    tl.store(forces_ptr + v * 3 + 0, fx)
    tl.store(forces_ptr + v * 3 + 1, fy)
    tl.store(forces_ptr + v * 3 + 2, fz)


# --------------------------------------------------------------------------- Kernel 2: Fused Bias + LayerNorm + SiLU & Edge Density

@triton.jit
def _bias_ln_silu_fwd_kernel(
    x_ptr, b_ptr, lnw_ptr, lnb_ptr,
    out_ptr, xn_ptr, rstd_ptr,
    M,
    ROW_BIAS: tl.constexpr,
    SAVE_BWD: tl.constexpr,
    D: tl.constexpr, BLOCK_M: tl.constexpr, DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    mask_m = m_offs < M
    idx = m_offs[:, None] * D + d_offs[None, :]
    mask = mask_m[:, None]

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    if ROW_BIAS:
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
    else:
        b = tl.load(b_ptr + d_offs)[None, :]
    x = x + b

    inv_D = 1.0 / D
    mean = tl.sum(x, axis=1) * inv_D
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * inv_D
    rstd = libdevice.rsqrt(var + 1e-5)
    xn = xc * rstd[:, None]

    lnw = tl.load(lnw_ptr + d_offs)[None, :]
    lnb = tl.load(lnb_ptr + d_offs)[None, :]
    z = xn * lnw + lnb
    sig = 1.0 / (1.0 + libdevice.exp(-z))
    out = z * sig

    tl.store(out_ptr + idx, out, mask=mask)
    if SAVE_BWD:
        tl.store(xn_ptr + idx, xn, mask=mask)
        tl.store(rstd_ptr + m_offs, rstd, mask=mask_m)


@triton.jit
def _bias_ln_silu_bwd_kernel(
    g_out_ptr, xn_ptr, rstd_ptr, lnw_ptr, lnb_ptr,
    g_x_ptr,
    M,
    D: tl.constexpr, BLOCK_M: tl.constexpr, DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    mask_m = m_offs < M
    idx = m_offs[:, None] * D + d_offs[None, :]
    mask = mask_m[:, None]

    go = tl.load(g_out_ptr + idx, mask=mask, other=0.0)
    xn = tl.load(xn_ptr + idx, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + m_offs, mask=mask_m, other=0.0)
    lnw = tl.load(lnw_ptr + d_offs)[None, :]
    lnb = tl.load(lnb_ptr + d_offs)[None, :]

    z = xn * lnw + lnb
    sig = 1.0 / (1.0 + libdevice.exp(-z))
    dz = go * sig * (1.0 + z * (1.0 - sig))
    dxn = dz * lnw

    inv_D = 1.0 / D
    c1 = tl.sum(dxn, axis=1) * inv_D
    c2 = tl.sum(dxn * xn, axis=1) * inv_D
    gx = rstd[:, None] * (dxn - c1[:, None] - xn * c2[:, None])
    tl.store(g_x_ptr + idx, gx, mask=mask)


@triton.jit
def _edge_density_fwd_kernel(
    h_ptr, bias_ptr, cut_ptr, lnw_ptr, lnb_ptr, w1_ptr, b1_ptr,
    ed_ptr, xn_ptr, rstd_ptr, d1_ptr,
    E,
    SAVE_BWD: tl.constexpr,
    BLOCK_E: tl.constexpr, DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    d_offs = tl.arange(0, 64)
    mask_e = e_offs < E
    cut = tl.load(cut_ptr + e_offs, mask=mask_e, other=0.0)
    idx = e_offs[:, None] * 64 + d_offs[None, :]
    mask = mask_e[:, None]

    x = tl.load(h_ptr + idx, mask=mask, other=0.0) + tl.load(bias_ptr + idx, mask=mask, other=0.0)
    inv_D = 1.0 / 64.0
    mean = tl.sum(x, axis=1) * inv_D
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * inv_D
    rstd = libdevice.rsqrt(var + 1e-5)
    xn = xc * rstd[:, None]

    lnw = tl.load(lnw_ptr + d_offs)[None, :]
    lnb = tl.load(lnb_ptr + d_offs)[None, :]
    z = xn * lnw + lnb
    sig = 1.0 / (1.0 + libdevice.exp(-z))
    act = z * sig

    w1 = tl.load(w1_ptr + d_offs)[None, :]
    b1 = tl.load(b1_ptr)
    d1 = tl.sum(act * w1, axis=1) + b1
    ed = libdevice.tanh(d1 * d1) * cut

    tl.store(ed_ptr + e_offs, ed, mask=mask_e)
    if SAVE_BWD:
        tl.store(xn_ptr + idx, xn, mask=mask)
        tl.store(rstd_ptr + e_offs, rstd, mask=mask_e)
        tl.store(d1_ptr + e_offs, d1, mask=mask_e)


@triton.jit
def _edge_density_bwd_kernel(
    g_den_ptr, dst_ptr, cut_ptr, xn_ptr, rstd_ptr, d1_ptr,
    lnw_ptr, lnb_ptr, w1_ptr,
    g_h_ptr, g_cut_ptr,
    E,
    BLOCK_E: tl.constexpr, DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    d_offs = tl.arange(0, 64)
    mask_e = e_offs < E
    v_idx = tl.load(dst_ptr + e_offs, mask=mask_e, other=0)
    g_ed = tl.load(g_den_ptr + v_idx, mask=mask_e, other=0.0)
    cut = tl.load(cut_ptr + e_offs, mask=mask_e, other=0.0)
    d1 = tl.load(d1_ptr + e_offs, mask=mask_e, other=0.0)

    t = libdevice.tanh(d1 * d1)
    # Accumulate into g_cut
    gc_old = tl.load(g_cut_ptr + e_offs, mask=mask_e, other=0.0)
    tl.store(g_cut_ptr + e_offs, gc_old + g_ed * t, mask=mask_e)

    g_d1 = g_ed * cut * (1.0 - t * t) * (2.0 * d1)
    idx = e_offs[:, None] * 64 + d_offs[None, :]
    mask = mask_e[:, None]
    xn = tl.load(xn_ptr + idx, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + e_offs, mask=mask_e, other=0.0)
    lnw = tl.load(lnw_ptr + d_offs)[None, :]
    lnb = tl.load(lnb_ptr + d_offs)[None, :]
    w1 = tl.load(w1_ptr + d_offs)[None, :]

    z = xn * lnw + lnb
    sig = 1.0 / (1.0 + libdevice.exp(-z))
    go = g_d1[:, None] * w1
    dz = go * sig * (1.0 + z * (1.0 - sig))
    dxn = dz * lnw
    inv_D = 1.0 / 64.0
    c1 = tl.sum(dxn, axis=1) * inv_D
    c2 = tl.sum(dxn * xn, axis=1) * inv_D
    gh = rstd[:, None] * (dxn - c1[:, None] - xn * c2[:, None])
    tl.store(g_h_ptr + idx, gh, mask=mask)


# --------------------------------------------------------------------------- Kernel 3: Species-Factored A Accumulation (CSR, atomic-free!)

@triton.jit
def _edge_to_A_fwd_kernel(
    seg_ptr, cut_ptr, y_ptr, act_ptr, ed_ptr,
    A_ptr, den_part_ptr,
    N, S,
    DTYPE: tl.constexpr,
):
    v = tl.program_id(0)
    s = tl.program_id(1)
    bkt = v * S + s
    start = tl.load(seg_ptr + bkt)
    end = tl.load(seg_ptr + bkt + 1)

    m_offs = tl.arange(0, 16)
    k_offs = tl.arange(0, 64)

    acc = tl.zeros((16, 64), dtype=DTYPE)
    acc_b = tl.zeros((16,), dtype=DTYPE)
    den_acc = tl.zeros((), dtype=DTYPE)

    for e in range(start, end):
        cut = tl.load(cut_ptr + e)
        if cut != 0.0:
            y_e = tl.load(y_ptr + e * 16 + m_offs) * cut
            act_e = tl.load(act_ptr + e * 64 + k_offs)
            acc = acc + y_e[:, None] * act_e[None, :]
            acc_b = acc_b + y_e
            den_acc = den_acc + tl.load(ed_ptr + e)

    # Store into A_all[m, v, s, k] with shape (16, N, S, 65)
    stride_m = N * S * 65
    base_vs = (v * S + s) * 65
    tl.store(A_ptr + m_offs[:, None] * stride_m + base_vs + k_offs[None, :], acc)
    tl.store(A_ptr + m_offs * stride_m + base_vs + 64, acc_b)
    tl.store(den_part_ptr + bkt, den_acc)


@triton.jit
def _edge_to_A_bwd_kernel(
    seg_ptr, cut_ptr, y_ptr, act_ptr, g_A_ptr,
    g_act_ptr, g_y_ptr, g_cut_ptr,
    N, S,
    DTYPE: tl.constexpr,
):
    v = tl.program_id(0)
    s = tl.program_id(1)
    bkt = v * S + s
    start = tl.load(seg_ptr + bkt)
    end = tl.load(seg_ptr + bkt + 1)

    m_offs = tl.arange(0, 16)
    k_offs = tl.arange(0, 64)
    stride_m = N * S * 65
    base_vs = (v * S + s) * 65

    gA = tl.load(g_A_ptr + m_offs[:, None] * stride_m + base_vs + k_offs[None, :])   # (16, 64)
    gAb = tl.load(g_A_ptr + m_offs * stride_m + base_vs + 64)                        # (16,)
    z64 = tl.zeros((64,), dtype=DTYPE)
    z16 = tl.zeros((16,), dtype=DTYPE)
    z0 = tl.zeros((), dtype=DTYPE)

    for e in range(start, end):
        cut = tl.load(cut_ptr + e)
        if cut != 0.0:
            y_e = tl.load(y_ptr + e * 16 + m_offs)
            act_e = tl.load(act_ptr + e * 64 + k_offs)
            # g_act[k] = cut * sum_m gA[m, k] * y_e[m]
            g_act_e = cut * tl.sum(gA * y_e[:, None], axis=0)
            # tmp[m] = gAb[m] + sum_k gA[m, k] * act_e[k]
            tmp_m = gAb + tl.sum(gA * act_e[None, :], axis=1)
            g_y_e = cut * tmp_m
            g_cut_e = tl.sum(tmp_m * y_e, axis=0)
            tl.store(g_act_ptr + e * 64 + k_offs, g_act_e)
            tl.store(g_y_ptr + e * 16 + m_offs, g_y_e)
            tl.store(g_cut_ptr + e, g_cut_e)
        else:
            tl.store(g_act_ptr + e * 64 + k_offs, z64)
            tl.store(g_y_ptr + e * 16 + m_offs, z16)
            tl.store(g_cut_ptr + e, z0)


# --------------------------------------------------------------------------- Kernel 4: DivDensity + AddRes + Gate

@triton.jit
def _div_res_gate_fwd_kernel(
    out0_ptr, outv_ptr, res_ptr, inv_den_ptr, sc_ptr,
    gated_ptr,
    N,
    BLOCK_C: tl.constexpr, DTYPE: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    inv_d = tl.load(inv_den_ptr + n)
    cst_silu = tl.load(sc_ptr + 3)
    cst_sig = tl.load(sc_ptr + 4)

    # l=0 scalar: silu(out0[:, :512] * inv_d + res[:, :512]) * cst_silu
    base0 = n * 2048
    s_in = tl.load(out0_ptr + base0 + c) * inv_d + tl.load(res_ptr + base0 + c)
    s_out = cst_silu * s_in / (1.0 + libdevice.exp(-s_in))
    tl.store(gated_ptr + (0 * N + n) * 512 + c, s_out)

    # l=1 (m=1..3), l=2 (m=4..8), l=3 (m=9..15)
    g1_in = tl.load(out0_ptr + base0 + 512 + c) * inv_d + tl.load(res_ptr + base0 + 512 + c)
    g1 = cst_sig / (1.0 + libdevice.exp(-g1_in)) * inv_d
    for m in tl.static_range(1, 4):
        v = tl.load(outv_ptr + ((m - 1) * N + n) * 512 + c)
        tl.store(gated_ptr + (m * N + n) * 512 + c, v * g1)

    g2_in = tl.load(out0_ptr + base0 + 1024 + c) * inv_d + tl.load(res_ptr + base0 + 1024 + c)
    g2 = cst_sig / (1.0 + libdevice.exp(-g2_in)) * inv_d
    for m in tl.static_range(4, 9):
        v = tl.load(outv_ptr + ((m - 1) * N + n) * 512 + c)
        tl.store(gated_ptr + (m * N + n) * 512 + c, v * g2)

    g3_in = tl.load(out0_ptr + base0 + 1536 + c) * inv_d + tl.load(res_ptr + base0 + 1536 + c)
    g3 = cst_sig / (1.0 + libdevice.exp(-g3_in)) * inv_d
    for m in tl.static_range(9, 16):
        v = tl.load(outv_ptr + ((m - 1) * N + n) * 512 + c)
        tl.store(gated_ptr + (m * N + n) * 512 + c, v * g3)


@triton.jit
def _div_res_gate_bwd_kernel(
    g_gated_ptr, out0_ptr, outv_ptr, res_ptr, inv_den_ptr, sc_ptr,
    g_out0_ptr, g_outv_ptr, g_inv_part_ptr,
    N,
    N_CB: tl.constexpr, BLOCK_C: tl.constexpr, DTYPE: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    inv_d = tl.load(inv_den_ptr + n)
    cst_silu = tl.load(sc_ptr + 3)
    cst_sig = tl.load(sc_ptr + 4)
    base0 = n * 2048

    # l=0 scalar
    o0 = tl.load(out0_ptr + base0 + c)
    s_in = o0 * inv_d + tl.load(res_ptr + base0 + c)
    sig0 = 1.0 / (1.0 + libdevice.exp(-s_in))
    gs0 = tl.load(g_gated_ptr + (0 * N + n) * 512 + c) * cst_silu * sig0 * (1.0 + s_in * (1.0 - sig0))
    tl.store(g_out0_ptr + base0 + c, gs0 * inv_d)
    g_inv_acc = gs0 * o0

    # l=1
    o_g1 = tl.load(out0_ptr + base0 + 512 + c)
    g1_in = o_g1 * inv_d + tl.load(res_ptr + base0 + 512 + c)
    sig1 = 1.0 / (1.0 + libdevice.exp(-g1_in))
    gate1 = cst_sig * sig1
    g_gate1 = tl.zeros((BLOCK_C,), dtype=DTYPE)
    for m in tl.static_range(1, 4):
        gg = tl.load(g_gated_ptr + (m * N + n) * 512 + c)
        ov = tl.load(outv_ptr + ((m - 1) * N + n) * 512 + c)
        tl.store(g_outv_ptr + ((m - 1) * N + n) * 512 + c, gg * gate1 * inv_d)
        g_gate1 = g_gate1 + gg * ov
    g_inv_acc = g_inv_acc + g_gate1 * gate1
    gl1 = (g_gate1 * inv_d) * gate1 * (1.0 - sig1)
    tl.store(g_out0_ptr + base0 + 512 + c, gl1 * inv_d)
    g_inv_acc = g_inv_acc + gl1 * o_g1

    # l=2
    o_g2 = tl.load(out0_ptr + base0 + 1024 + c)
    g2_in = o_g2 * inv_d + tl.load(res_ptr + base0 + 1024 + c)
    sig2 = 1.0 / (1.0 + libdevice.exp(-g2_in))
    gate2 = cst_sig * sig2
    g_gate2 = tl.zeros((BLOCK_C,), dtype=DTYPE)
    for m in tl.static_range(4, 9):
        gg = tl.load(g_gated_ptr + (m * N + n) * 512 + c)
        ov = tl.load(outv_ptr + ((m - 1) * N + n) * 512 + c)
        tl.store(g_outv_ptr + ((m - 1) * N + n) * 512 + c, gg * gate2 * inv_d)
        g_gate2 = g_gate2 + gg * ov
    g_inv_acc = g_inv_acc + g_gate2 * gate2
    gl2 = (g_gate2 * inv_d) * gate2 * (1.0 - sig2)
    tl.store(g_out0_ptr + base0 + 1024 + c, gl2 * inv_d)
    g_inv_acc = g_inv_acc + gl2 * o_g2

    # l=3
    o_g3 = tl.load(out0_ptr + base0 + 1536 + c)
    g3_in = o_g3 * inv_d + tl.load(res_ptr + base0 + 1536 + c)
    sig3 = 1.0 / (1.0 + libdevice.exp(-g3_in))
    gate3 = cst_sig * sig3
    g_gate3 = tl.zeros((BLOCK_C,), dtype=DTYPE)
    for m in tl.static_range(9, 16):
        gg = tl.load(g_gated_ptr + (m * N + n) * 512 + c)
        ov = tl.load(outv_ptr + ((m - 1) * N + n) * 512 + c)
        tl.store(g_outv_ptr + ((m - 1) * N + n) * 512 + c, gg * gate3 * inv_d)
        g_gate3 = g_gate3 + gg * ov
    g_inv_acc = g_inv_acc + g_gate3 * gate3
    gl3 = (g_gate3 * inv_d) * gate3 * (1.0 - sig3)
    tl.store(g_out0_ptr + base0 + 1536 + c, gl3 * inv_d)
    g_inv_acc = g_inv_acc + gl3 * o_g3

    tl.store(g_inv_part_ptr + n * N_CB + cb, tl.sum(g_inv_acc, axis=0))


# --------------------------------------------------------------------------- Kernel 5: Register-Only 77-Triple SymmetricContraction

@triton.jit
def _sym_contr_fwd_kernel(
    x_ptr, c1_ptr, c2_ptr, c3_ptr, out_ptr,
    N, C: tl.constexpr, BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    st = N * C
    base = n * C + c

    x0 = tl.load(x_ptr + 0 * st + base)
    x1 = tl.load(x_ptr + 1 * st + base)
    x2 = tl.load(x_ptr + 2 * st + base)
    x3 = tl.load(x_ptr + 3 * st + base)
    x4 = tl.load(x_ptr + 4 * st + base)
    x5 = tl.load(x_ptr + 5 * st + base)
    x6 = tl.load(x_ptr + 6 * st + base)
    x7 = tl.load(x_ptr + 7 * st + base)
    x8 = tl.load(x_ptr + 8 * st + base)
    x9 = tl.load(x_ptr + 9 * st + base)
    x10 = tl.load(x_ptr + 10 * st + base)
    x11 = tl.load(x_ptr + 11 * st + base)
    x12 = tl.load(x_ptr + 12 * st + base)
    x13 = tl.load(x_ptr + 13 * st + base)
    x14 = tl.load(x_ptr + 14 * st + base)
    x15 = tl.load(x_ptr + 15 * st + base)

    # Degree 1 & 2
    acc = tl.load(c1_ptr + c) * x0
    acc += tl.load(c2_ptr + 0 * C + c) * (x0 * x0)
    acc += tl.load(c2_ptr + 1 * C + c) * (x1 * x1 + x2 * x2 + x3 * x3)
    acc += tl.load(c2_ptr + 2 * C + c) * (x4 * x4 + x5 * x5 + x6 * x6 + x7 * x7 + x8 * x8)
    acc += tl.load(c2_ptr + 3 * C + c) * (
        x9 * x9 + x10 * x10 + x11 * x11 + x12 * x12 + x13 * x13 + x14 * x14 + x15 * x15
    )

    # Degree 3: all 77 canonical triples
    acc += tl.load(c3_ptr + 0 * C + c) * (x0 * x0 * x0)
    acc += tl.load(c3_ptr + 1 * C + c) * (x0 * x1 * x1)
    acc += tl.load(c3_ptr + 2 * C + c) * (x0 * x2 * x2)
    acc += tl.load(c3_ptr + 3 * C + c) * (x0 * x3 * x3)
    acc += tl.load(c3_ptr + 4 * C + c) * (x0 * x4 * x4)
    acc += tl.load(c3_ptr + 5 * C + c) * (x0 * x5 * x5)
    acc += tl.load(c3_ptr + 6 * C + c) * (x0 * x6 * x6)
    acc += tl.load(c3_ptr + 7 * C + c) * (x0 * x7 * x7)
    acc += tl.load(c3_ptr + 8 * C + c) * (x0 * x8 * x8)
    acc += tl.load(c3_ptr + 9 * C + c) * (x0 * x9 * x9)
    acc += tl.load(c3_ptr + 10 * C + c) * (x0 * x10 * x10)
    acc += tl.load(c3_ptr + 11 * C + c) * (x0 * x11 * x11)
    acc += tl.load(c3_ptr + 12 * C + c) * (x0 * x12 * x12)
    acc += tl.load(c3_ptr + 13 * C + c) * (x0 * x13 * x13)
    acc += tl.load(c3_ptr + 14 * C + c) * (x0 * x14 * x14)
    acc += tl.load(c3_ptr + 15 * C + c) * (x0 * x15 * x15)
    acc += tl.load(c3_ptr + 16 * C + c) * (x1 * x1 * x6)
    acc += tl.load(c3_ptr + 17 * C + c) * (x1 * x1 * x8)
    acc += tl.load(c3_ptr + 18 * C + c) * (x1 * x2 * x5)
    acc += tl.load(c3_ptr + 19 * C + c) * (x1 * x3 * x4)
    acc += tl.load(c3_ptr + 20 * C + c) * (x1 * x4 * x13)
    acc += tl.load(c3_ptr + 21 * C + c) * (x1 * x4 * x15)
    acc += tl.load(c3_ptr + 22 * C + c) * (x1 * x5 * x12)
    acc += tl.load(c3_ptr + 23 * C + c) * (x1 * x5 * x14)
    acc += tl.load(c3_ptr + 24 * C + c) * (x1 * x6 * x11)
    acc += tl.load(c3_ptr + 25 * C + c) * (x1 * x7 * x10)
    acc += tl.load(c3_ptr + 26 * C + c) * (x1 * x8 * x9)
    acc += tl.load(c3_ptr + 27 * C + c) * (x1 * x8 * x11)
    acc += tl.load(c3_ptr + 28 * C + c) * (x2 * x2 * x6)
    acc += tl.load(c3_ptr + 29 * C + c) * (x2 * x3 * x7)
    acc += tl.load(c3_ptr + 30 * C + c) * (x2 * x4 * x10)
    acc += tl.load(c3_ptr + 31 * C + c) * (x2 * x5 * x11)
    acc += tl.load(c3_ptr + 32 * C + c) * (x2 * x6 * x12)
    acc += tl.load(c3_ptr + 33 * C + c) * (x2 * x7 * x13)
    acc += tl.load(c3_ptr + 34 * C + c) * (x2 * x8 * x14)
    acc += tl.load(c3_ptr + 35 * C + c) * (x3 * x3 * x6)
    acc += tl.load(c3_ptr + 36 * C + c) * (x3 * x3 * x8)
    acc += tl.load(c3_ptr + 37 * C + c) * (x3 * x4 * x9)
    acc += tl.load(c3_ptr + 38 * C + c) * (x3 * x4 * x11)
    acc += tl.load(c3_ptr + 39 * C + c) * (x3 * x5 * x10)
    acc += tl.load(c3_ptr + 40 * C + c) * (x3 * x6 * x13)
    acc += tl.load(c3_ptr + 41 * C + c) * (x3 * x7 * x12)
    acc += tl.load(c3_ptr + 42 * C + c) * (x3 * x7 * x14)
    acc += tl.load(c3_ptr + 43 * C + c) * (x3 * x8 * x13)
    acc += tl.load(c3_ptr + 44 * C + c) * (x3 * x8 * x15)
    acc += tl.load(c3_ptr + 45 * C + c) * (x4 * x4 * x6)
    acc += tl.load(c3_ptr + 46 * C + c) * (x4 * x5 * x7)
    acc += tl.load(c3_ptr + 47 * C + c) * (x4 * x9 * x13)
    acc += tl.load(c3_ptr + 48 * C + c) * (x4 * x10 * x12)
    acc += tl.load(c3_ptr + 49 * C + c) * (x4 * x11 * x13)
    acc += tl.load(c3_ptr + 50 * C + c) * (x4 * x11 * x15)
    acc += tl.load(c3_ptr + 51 * C + c) * (x5 * x5 * x6)
    acc += tl.load(c3_ptr + 52 * C + c) * (x5 * x5 * x8)
    acc += tl.load(c3_ptr + 53 * C + c) * (x5 * x9 * x14)
    acc += tl.load(c3_ptr + 54 * C + c) * (x5 * x10 * x13)
    acc += tl.load(c3_ptr + 55 * C + c) * (x5 * x10 * x15)
    acc += tl.load(c3_ptr + 56 * C + c) * (x5 * x11 * x12)
    acc += tl.load(c3_ptr + 57 * C + c) * (x5 * x11 * x14)
    acc += tl.load(c3_ptr + 58 * C + c) * (x6 * x6 * x6)
    acc += tl.load(c3_ptr + 59 * C + c) * (x6 * x7 * x7)
    acc += tl.load(c3_ptr + 60 * C + c) * (x6 * x8 * x8)
    acc += tl.load(c3_ptr + 61 * C + c) * (x6 * x9 * x9)
    acc += tl.load(c3_ptr + 62 * C + c) * (x6 * x11 * x11)
    acc += tl.load(c3_ptr + 63 * C + c) * (x6 * x12 * x12)
    acc += tl.load(c3_ptr + 64 * C + c) * (x6 * x13 * x13)
    acc += tl.load(c3_ptr + 65 * C + c) * (x6 * x15 * x15)
    acc += tl.load(c3_ptr + 66 * C + c) * (x7 * x7 * x8)
    acc += tl.load(c3_ptr + 67 * C + c) * (x7 * x9 * x10)
    acc += tl.load(c3_ptr + 68 * C + c) * (x7 * x10 * x11)
    acc += tl.load(c3_ptr + 69 * C + c) * (x7 * x12 * x13)
    acc += tl.load(c3_ptr + 70 * C + c) * (x7 * x13 * x14)
    acc += tl.load(c3_ptr + 71 * C + c) * (x7 * x14 * x15)
    acc += tl.load(c3_ptr + 72 * C + c) * (x8 * x9 * x11)
    acc += tl.load(c3_ptr + 73 * C + c) * (x8 * x11 * x11)
    acc += tl.load(c3_ptr + 74 * C + c) * (x8 * x12 * x14)
    acc += tl.load(c3_ptr + 75 * C + c) * (x8 * x13 * x13)
    acc += tl.load(c3_ptr + 76 * C + c) * (x8 * x13 * x15)

    tl.store(out_ptr + base, acc)


@triton.jit
def _sym_contr_bwd_kernel(
    x_ptr, go_ptr, c1_ptr, c2_ptr, c3_ptr, gx_ptr,
    N, C: tl.constexpr, BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    st = N * C
    base = n * C + c

    go = tl.load(go_ptr + base)
    x0 = tl.load(x_ptr + 0 * st + base)
    x1 = tl.load(x_ptr + 1 * st + base)
    x2 = tl.load(x_ptr + 2 * st + base)
    x3 = tl.load(x_ptr + 3 * st + base)
    x4 = tl.load(x_ptr + 4 * st + base)
    x5 = tl.load(x_ptr + 5 * st + base)
    x6 = tl.load(x_ptr + 6 * st + base)
    x7 = tl.load(x_ptr + 7 * st + base)
    x8 = tl.load(x_ptr + 8 * st + base)
    x9 = tl.load(x_ptr + 9 * st + base)
    x10 = tl.load(x_ptr + 10 * st + base)
    x11 = tl.load(x_ptr + 11 * st + base)
    x12 = tl.load(x_ptr + 12 * st + base)
    x13 = tl.load(x_ptr + 13 * st + base)
    x14 = tl.load(x_ptr + 14 * st + base)
    x15 = tl.load(x_ptr + 15 * st + base)

    # Degree 1 & 2 grads
    w2_0 = 2.0 * go * tl.load(c2_ptr + 0 * C + c)
    w2_1 = 2.0 * go * tl.load(c2_ptr + 1 * C + c)
    w2_2 = 2.0 * go * tl.load(c2_ptr + 2 * C + c)
    w2_3 = 2.0 * go * tl.load(c2_ptr + 3 * C + c)

    gx0 = go * tl.load(c1_ptr + c) + w2_0 * x0
    gx1 = w2_1 * x1
    gx2 = w2_1 * x2
    gx3 = w2_1 * x3
    gx4 = w2_2 * x4
    gx5 = w2_2 * x5
    gx6 = w2_2 * x6
    gx7 = w2_2 * x7
    gx8 = w2_2 * x8
    gx9 = w2_3 * x9
    gx10 = w2_3 * x10
    gx11 = w2_3 * x11
    gx12 = w2_3 * x12
    gx13 = w2_3 * x13
    gx14 = w2_3 * x14
    gx15 = w2_3 * x15

    # Degree 3 grads (77 triples)
    w = go * tl.load(c3_ptr + 0 * C + c); gx0 += 3.0 * w * x0 * x0
    w = go * tl.load(c3_ptr + 1 * C + c); gx0 += w * x1 * x1; gx1 += 2.0 * w * x0 * x1
    w = go * tl.load(c3_ptr + 2 * C + c); gx0 += w * x2 * x2; gx2 += 2.0 * w * x0 * x2
    w = go * tl.load(c3_ptr + 3 * C + c); gx0 += w * x3 * x3; gx3 += 2.0 * w * x0 * x3
    w = go * tl.load(c3_ptr + 4 * C + c); gx0 += w * x4 * x4; gx4 += 2.0 * w * x0 * x4
    w = go * tl.load(c3_ptr + 5 * C + c); gx0 += w * x5 * x5; gx5 += 2.0 * w * x0 * x5
    w = go * tl.load(c3_ptr + 6 * C + c); gx0 += w * x6 * x6; gx6 += 2.0 * w * x0 * x6
    w = go * tl.load(c3_ptr + 7 * C + c); gx0 += w * x7 * x7; gx7 += 2.0 * w * x0 * x7
    w = go * tl.load(c3_ptr + 8 * C + c); gx0 += w * x8 * x8; gx8 += 2.0 * w * x0 * x8
    w = go * tl.load(c3_ptr + 9 * C + c); gx0 += w * x9 * x9; gx9 += 2.0 * w * x0 * x9
    w = go * tl.load(c3_ptr + 10 * C + c); gx0 += w * x10 * x10; gx10 += 2.0 * w * x0 * x10
    w = go * tl.load(c3_ptr + 11 * C + c); gx0 += w * x11 * x11; gx11 += 2.0 * w * x0 * x11
    w = go * tl.load(c3_ptr + 12 * C + c); gx0 += w * x12 * x12; gx12 += 2.0 * w * x0 * x12
    w = go * tl.load(c3_ptr + 13 * C + c); gx0 += w * x13 * x13; gx13 += 2.0 * w * x0 * x13
    w = go * tl.load(c3_ptr + 14 * C + c); gx0 += w * x14 * x14; gx14 += 2.0 * w * x0 * x14
    w = go * tl.load(c3_ptr + 15 * C + c); gx0 += w * x15 * x15; gx15 += 2.0 * w * x0 * x15
    w = go * tl.load(c3_ptr + 16 * C + c); gx1 += 2.0 * w * x1 * x6; gx6 += w * x1 * x1
    w = go * tl.load(c3_ptr + 17 * C + c); gx1 += 2.0 * w * x1 * x8; gx8 += w * x1 * x1
    w = go * tl.load(c3_ptr + 18 * C + c); gx1 += w * x2 * x5; gx2 += w * x1 * x5; gx5 += w * x1 * x2
    w = go * tl.load(c3_ptr + 19 * C + c); gx1 += w * x3 * x4; gx3 += w * x1 * x4; gx4 += w * x1 * x3
    w = go * tl.load(c3_ptr + 20 * C + c); gx1 += w * x4 * x13; gx4 += w * x1 * x13; gx13 += w * x1 * x4
    w = go * tl.load(c3_ptr + 21 * C + c); gx1 += w * x4 * x15; gx4 += w * x1 * x15; gx15 += w * x1 * x4
    w = go * tl.load(c3_ptr + 22 * C + c); gx1 += w * x5 * x12; gx5 += w * x1 * x12; gx12 += w * x1 * x5
    w = go * tl.load(c3_ptr + 23 * C + c); gx1 += w * x5 * x14; gx5 += w * x1 * x14; gx14 += w * x1 * x5
    w = go * tl.load(c3_ptr + 24 * C + c); gx1 += w * x6 * x11; gx6 += w * x1 * x11; gx11 += w * x1 * x6
    w = go * tl.load(c3_ptr + 25 * C + c); gx1 += w * x7 * x10; gx7 += w * x1 * x10; gx10 += w * x1 * x7
    w = go * tl.load(c3_ptr + 26 * C + c); gx1 += w * x8 * x9; gx8 += w * x1 * x9; gx9 += w * x1 * x8
    w = go * tl.load(c3_ptr + 27 * C + c); gx1 += w * x8 * x11; gx8 += w * x1 * x11; gx11 += w * x1 * x8
    w = go * tl.load(c3_ptr + 28 * C + c); gx2 += 2.0 * w * x2 * x6; gx6 += w * x2 * x2
    w = go * tl.load(c3_ptr + 29 * C + c); gx2 += w * x3 * x7; gx3 += w * x2 * x7; gx7 += w * x2 * x3
    w = go * tl.load(c3_ptr + 30 * C + c); gx2 += w * x4 * x10; gx4 += w * x2 * x10; gx10 += w * x2 * x4
    w = go * tl.load(c3_ptr + 31 * C + c); gx2 += w * x5 * x11; gx5 += w * x2 * x11; gx11 += w * x2 * x5
    w = go * tl.load(c3_ptr + 32 * C + c); gx2 += w * x6 * x12; gx6 += w * x2 * x12; gx12 += w * x2 * x6
    w = go * tl.load(c3_ptr + 33 * C + c); gx2 += w * x7 * x13; gx7 += w * x2 * x13; gx13 += w * x2 * x7
    w = go * tl.load(c3_ptr + 34 * C + c); gx2 += w * x8 * x14; gx8 += w * x2 * x14; gx14 += w * x2 * x8
    w = go * tl.load(c3_ptr + 35 * C + c); gx3 += 2.0 * w * x3 * x6; gx6 += w * x3 * x3
    w = go * tl.load(c3_ptr + 36 * C + c); gx3 += 2.0 * w * x3 * x8; gx8 += w * x3 * x3
    w = go * tl.load(c3_ptr + 37 * C + c); gx3 += w * x4 * x9; gx4 += w * x3 * x9; gx9 += w * x3 * x4
    w = go * tl.load(c3_ptr + 38 * C + c); gx3 += w * x4 * x11; gx4 += w * x3 * x11; gx11 += w * x3 * x4
    w = go * tl.load(c3_ptr + 39 * C + c); gx3 += w * x5 * x10; gx5 += w * x3 * x10; gx10 += w * x3 * x5
    w = go * tl.load(c3_ptr + 40 * C + c); gx3 += w * x6 * x13; gx6 += w * x3 * x13; gx13 += w * x3 * x6
    w = go * tl.load(c3_ptr + 41 * C + c); gx3 += w * x7 * x12; gx7 += w * x3 * x12; gx12 += w * x3 * x7
    w = go * tl.load(c3_ptr + 42 * C + c); gx3 += w * x7 * x14; gx7 += w * x3 * x14; gx14 += w * x3 * x7
    w = go * tl.load(c3_ptr + 43 * C + c); gx3 += w * x8 * x13; gx8 += w * x3 * x13; gx13 += w * x3 * x8
    w = go * tl.load(c3_ptr + 44 * C + c); gx3 += w * x8 * x15; gx8 += w * x3 * x15; gx15 += w * x3 * x8
    w = go * tl.load(c3_ptr + 45 * C + c); gx4 += 2.0 * w * x4 * x6; gx6 += w * x4 * x4
    w = go * tl.load(c3_ptr + 46 * C + c); gx4 += w * x5 * x7; gx5 += w * x4 * x7; gx7 += w * x4 * x5
    w = go * tl.load(c3_ptr + 47 * C + c); gx4 += w * x9 * x13; gx9 += w * x4 * x13; gx13 += w * x4 * x9
    w = go * tl.load(c3_ptr + 48 * C + c); gx4 += w * x10 * x12; gx10 += w * x4 * x12; gx12 += w * x4 * x10
    w = go * tl.load(c3_ptr + 49 * C + c); gx4 += w * x11 * x13; gx11 += w * x4 * x13; gx13 += w * x4 * x11
    w = go * tl.load(c3_ptr + 50 * C + c); gx4 += w * x11 * x15; gx11 += w * x4 * x15; gx15 += w * x4 * x11
    w = go * tl.load(c3_ptr + 51 * C + c); gx5 += 2.0 * w * x5 * x6; gx6 += w * x5 * x5
    w = go * tl.load(c3_ptr + 52 * C + c); gx5 += 2.0 * w * x5 * x8; gx8 += w * x5 * x5
    w = go * tl.load(c3_ptr + 53 * C + c); gx5 += w * x9 * x14; gx9 += w * x5 * x14; gx14 += w * x5 * x9
    w = go * tl.load(c3_ptr + 54 * C + c); gx5 += w * x10 * x13; gx10 += w * x5 * x13; gx13 += w * x5 * x10
    w = go * tl.load(c3_ptr + 55 * C + c); gx5 += w * x10 * x15; gx10 += w * x5 * x15; gx15 += w * x5 * x10
    w = go * tl.load(c3_ptr + 56 * C + c); gx5 += w * x11 * x12; gx11 += w * x5 * x12; gx12 += w * x5 * x11
    w = go * tl.load(c3_ptr + 57 * C + c); gx5 += w * x11 * x14; gx11 += w * x5 * x14; gx14 += w * x5 * x11
    w = go * tl.load(c3_ptr + 58 * C + c); gx6 += 3.0 * w * x6 * x6
    w = go * tl.load(c3_ptr + 59 * C + c); gx6 += w * x7 * x7; gx7 += 2.0 * w * x6 * x7
    w = go * tl.load(c3_ptr + 60 * C + c); gx6 += w * x8 * x8; gx8 += 2.0 * w * x6 * x8
    w = go * tl.load(c3_ptr + 61 * C + c); gx6 += w * x9 * x9; gx9 += 2.0 * w * x6 * x9
    w = go * tl.load(c3_ptr + 62 * C + c); gx6 += w * x11 * x11; gx11 += 2.0 * w * x6 * x11
    w = go * tl.load(c3_ptr + 63 * C + c); gx6 += w * x12 * x12; gx12 += 2.0 * w * x6 * x12
    w = go * tl.load(c3_ptr + 64 * C + c); gx6 += w * x13 * x13; gx13 += 2.0 * w * x6 * x13
    w = go * tl.load(c3_ptr + 65 * C + c); gx6 += w * x15 * x15; gx15 += 2.0 * w * x6 * x15
    w = go * tl.load(c3_ptr + 66 * C + c); gx7 += 2.0 * w * x7 * x8; gx8 += w * x7 * x7
    w = go * tl.load(c3_ptr + 67 * C + c); gx7 += w * x9 * x10; gx9 += w * x7 * x10; gx10 += w * x7 * x9
    w = go * tl.load(c3_ptr + 68 * C + c); gx7 += w * x10 * x11; gx10 += w * x7 * x11; gx11 += w * x7 * x10
    w = go * tl.load(c3_ptr + 69 * C + c); gx7 += w * x12 * x13; gx12 += w * x7 * x13; gx13 += w * x7 * x12
    w = go * tl.load(c3_ptr + 70 * C + c); gx7 += w * x13 * x14; gx13 += w * x7 * x14; gx14 += w * x7 * x13
    w = go * tl.load(c3_ptr + 71 * C + c); gx7 += w * x14 * x15; gx14 += w * x7 * x15; gx15 += w * x7 * x14
    w = go * tl.load(c3_ptr + 72 * C + c); gx8 += w * x9 * x11; gx9 += w * x8 * x11; gx11 += w * x8 * x9
    w = go * tl.load(c3_ptr + 73 * C + c); gx8 += w * x11 * x11; gx11 += 2.0 * w * x8 * x11
    w = go * tl.load(c3_ptr + 74 * C + c); gx8 += w * x12 * x14; gx12 += w * x8 * x14; gx14 += w * x8 * x12
    w = go * tl.load(c3_ptr + 75 * C + c); gx8 += w * x13 * x13; gx13 += 2.0 * w * x8 * x13
    w = go * tl.load(c3_ptr + 76 * C + c); gx8 += w * x13 * x15; gx13 += w * x8 * x15; gx15 += w * x8 * x13

    tl.store(gx_ptr + 0 * st + base, gx0)
    tl.store(gx_ptr + 1 * st + base, gx1)
    tl.store(gx_ptr + 2 * st + base, gx2)
    tl.store(gx_ptr + 3 * st + base, gx3)
    tl.store(gx_ptr + 4 * st + base, gx4)
    tl.store(gx_ptr + 5 * st + base, gx5)
    tl.store(gx_ptr + 6 * st + base, gx6)
    tl.store(gx_ptr + 7 * st + base, gx7)
    tl.store(gx_ptr + 8 * st + base, gx8)
    tl.store(gx_ptr + 9 * st + base, gx9)
    tl.store(gx_ptr + 10 * st + base, gx10)
    tl.store(gx_ptr + 11 * st + base, gx11)
    tl.store(gx_ptr + 12 * st + base, gx12)
    tl.store(gx_ptr + 13 * st + base, gx13)
    tl.store(gx_ptr + 14 * st + base, gx14)
    tl.store(gx_ptr + 15 * st + base, gx15)


# --------------------------------------------------------------------------- Kernel 6: Tiled-SRAM Monopole Electrostatics (Alpha+Beta Fused)

@triton.jit
def _es_features_fwd_kernel(
    pos_ptr, q_ab_ptr, ptr_ptr, sc_ptr,
    pot_ptr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, DTYPE: tl.constexpr,
):
    g = tl.program_id(0)
    bi = tl.program_id(1)
    g_start = tl.load(ptr_ptr + g)
    g_end = tl.load(ptr_ptr + g + 1)
    n_g = g_end - g_start

    inv_2w0 = tl.load(sc_ptr + 5)
    inv_2w1 = tl.load(sc_ptr + 6)
    sc0 = tl.load(sc_ptr + 7)
    sc1 = tl.load(sc_ptr + 8)

    i_loc = bi * BLOCK_I + tl.arange(0, BLOCK_I)
    mi = i_loc < n_g
    i_glob = g_start + i_loc

    xi = tl.load(pos_ptr + i_glob * 3 + 0, mask=mi, other=0.0)
    yi = tl.load(pos_ptr + i_glob * 3 + 1, mask=mi, other=0.0)
    zi = tl.load(pos_ptr + i_glob * 3 + 2, mask=mi, other=0.0)

    fa0 = tl.zeros((BLOCK_I,), dtype=DTYPE)
    fa1 = tl.zeros((BLOCK_I,), dtype=DTYPE)
    fb0 = tl.zeros((BLOCK_I,), dtype=DTYPE)
    fb1 = tl.zeros((BLOCK_I,), dtype=DTYPE)
    z_ij = tl.zeros((BLOCK_I, BLOCK_J), dtype=DTYPE)

    for j0 in range(0, n_g, BLOCK_J):
        j_loc = j0 + tl.arange(0, BLOCK_J)
        mj = j_loc < n_g
        j_glob = g_start + j_loc

        xj = tl.load(pos_ptr + j_glob * 3 + 0, mask=mj, other=0.0)
        yj = tl.load(pos_ptr + j_glob * 3 + 1, mask=mj, other=0.0)
        zj = tl.load(pos_ptr + j_glob * 3 + 2, mask=mj, other=0.0)
        qaj = tl.load(q_ab_ptr + j_glob * 2 + 0, mask=mj, other=0.0)
        qbj = tl.load(q_ab_ptr + j_glob * 2 + 1, mask=mj, other=0.0)

        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        valid = mi[:, None] & mj[None, :] & (i_loc[:, None] != j_loc[None, :])
        d2 = tl.where(valid, dx * dx + dy * dy + dz * dz, 1.0)
        d = libdevice.sqrt(d2)
        inv_dp = 1.0 / (d + 1e-6)
        k0 = tl.where(valid, libdevice.erf(d * inv_2w0) * inv_dp, z_ij)
        k1 = tl.where(valid, libdevice.erf(d * inv_2w1) * inv_dp, z_ij)

        fa0 = fa0 + tl.sum(k0 * qaj[None, :], axis=1)
        fa1 = fa1 + tl.sum(k1 * qaj[None, :], axis=1)
        fb0 = fb0 + tl.sum(k0 * qbj[None, :], axis=1)
        fb1 = fb1 + tl.sum(k1 * qbj[None, :], axis=1)

    tl.store(pot_ptr + i_glob * 4 + 0, fa0 * sc0, mask=mi)
    tl.store(pot_ptr + i_glob * 4 + 1, fa1 * sc1, mask=mi)
    tl.store(pot_ptr + i_glob * 4 + 2, fb0 * sc0, mask=mi)
    tl.store(pot_ptr + i_glob * 4 + 3, fb1 * sc1, mask=mi)


@triton.jit
def _es_features_bwd_kernel(
    pos_ptr, q_ab_ptr, g_pot_ptr, ptr_ptr, sc_ptr,
    g_q_ab_ptr, g_pos_ptr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, DTYPE: tl.constexpr,
):
    g = tl.program_id(0)
    bi = tl.program_id(1)
    g_start = tl.load(ptr_ptr + g)
    g_end = tl.load(ptr_ptr + g + 1)
    n_g = g_end - g_start

    inv_2w0 = tl.load(sc_ptr + 5)
    inv_2w1 = tl.load(sc_ptr + 6)
    sc0 = tl.load(sc_ptr + 7)
    sc1 = tl.load(sc_ptr + 8)

    i_loc = bi * BLOCK_I + tl.arange(0, BLOCK_I)
    mi = i_loc < n_g
    i_glob = g_start + i_loc

    xi = tl.load(pos_ptr + i_glob * 3 + 0, mask=mi, other=0.0)
    yi = tl.load(pos_ptr + i_glob * 3 + 1, mask=mi, other=0.0)
    zi = tl.load(pos_ptr + i_glob * 3 + 2, mask=mi, other=0.0)
    qai = tl.load(q_ab_ptr + i_glob * 2 + 0, mask=mi, other=0.0)
    qbi = tl.load(q_ab_ptr + i_glob * 2 + 1, mask=mi, other=0.0)
    gfa0i = tl.load(g_pot_ptr + i_glob * 4 + 0, mask=mi, other=0.0) * sc0
    gfa1i = tl.load(g_pot_ptr + i_glob * 4 + 1, mask=mi, other=0.0) * sc1
    gfb0i = tl.load(g_pot_ptr + i_glob * 4 + 2, mask=mi, other=0.0) * sc0
    gfb1i = tl.load(g_pot_ptr + i_glob * 4 + 3, mask=mi, other=0.0) * sc1

    gqa = tl.zeros((BLOCK_I,), dtype=DTYPE)
    gqb = tl.zeros((BLOCK_I,), dtype=DTYPE)
    gpx = tl.zeros((BLOCK_I,), dtype=DTYPE)
    gpy = tl.zeros((BLOCK_I,), dtype=DTYPE)
    gpz = tl.zeros((BLOCK_I,), dtype=DTYPE)
    z_ij = tl.zeros((BLOCK_I, BLOCK_J), dtype=DTYPE)
    c0 = _TWO_OVER_SQRT_PI * inv_2w0
    c1 = _TWO_OVER_SQRT_PI * inv_2w1

    for j0 in range(0, n_g, BLOCK_J):
        j_loc = j0 + tl.arange(0, BLOCK_J)
        mj = j_loc < n_g
        j_glob = g_start + j_loc

        xj = tl.load(pos_ptr + j_glob * 3 + 0, mask=mj, other=0.0)
        yj = tl.load(pos_ptr + j_glob * 3 + 1, mask=mj, other=0.0)
        zj = tl.load(pos_ptr + j_glob * 3 + 2, mask=mj, other=0.0)
        qaj = tl.load(q_ab_ptr + j_glob * 2 + 0, mask=mj, other=0.0)
        qbj = tl.load(q_ab_ptr + j_glob * 2 + 1, mask=mj, other=0.0)
        gfa0j = tl.load(g_pot_ptr + j_glob * 4 + 0, mask=mj, other=0.0) * sc0
        gfa1j = tl.load(g_pot_ptr + j_glob * 4 + 1, mask=mj, other=0.0) * sc1
        gfb0j = tl.load(g_pot_ptr + j_glob * 4 + 2, mask=mj, other=0.0) * sc0
        gfb1j = tl.load(g_pot_ptr + j_glob * 4 + 3, mask=mj, other=0.0) * sc1

        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        valid = mi[:, None] & mj[None, :] & (i_loc[:, None] != j_loc[None, :])
        d2 = tl.where(valid, dx * dx + dy * dy + dz * dz, 1.0)
        d = libdevice.sqrt(d2)
        dp = d + 1e-6
        inv_dp = 1.0 / dp
        inv_dp2 = inv_dp * inv_dp

        a0 = d * inv_2w0
        erf0 = libdevice.erf(a0)
        k0 = tl.where(valid, erf0 * inv_dp, z_ij)
        dk0_dd = (c0 * libdevice.exp(-a0 * a0) * dp - erf0) * inv_dp2

        a1 = d * inv_2w1
        erf1 = libdevice.erf(a1)
        k1 = tl.where(valid, erf1 * inv_dp, z_ij)
        dk1_dd = (c1 * libdevice.exp(-a1 * a1) * dp - erf1) * inv_dp2

        # dL/dq_i = sum_j (g_f[j] * k(i, j))
        gqa = gqa + tl.sum(k0 * gfa0j[None, :] + k1 * gfa1j[None, :], axis=1)
        gqb = gqb + tl.sum(k0 * gfb0j[None, :] + k1 * gfb1j[None, :], axis=1)

        # dL/dpos_i = sum_j [(g_f[i]*q[j] + g_f[j]*q[i]) * dk/dd / d] * (pos_i - pos_j)
        w0 = gfa0i[:, None] * qaj[None, :] + qai[:, None] * gfa0j[None, :] + gfb0i[:, None] * qbj[None, :] + qbi[:, None] * gfb0j[None, :]
        w1 = gfa1i[:, None] * qaj[None, :] + qai[:, None] * gfa1j[None, :] + gfb1i[:, None] * qbj[None, :] + qbi[:, None] * gfb1j[None, :]
        fac = tl.where(valid, (w0 * dk0_dd + w1 * dk1_dd) / d, z_ij)
        gpx = gpx + tl.sum(fac * dx, axis=1)
        gpy = gpy + tl.sum(fac * dy, axis=1)
        gpz = gpz + tl.sum(fac * dz, axis=1)

    # Accumulate into g_q_ab and g_pos
    tl.store(g_q_ab_ptr + i_glob * 2 + 0, tl.load(g_q_ab_ptr + i_glob * 2 + 0, mask=mi, other=0.0) + gqa, mask=mi)
    tl.store(g_q_ab_ptr + i_glob * 2 + 1, tl.load(g_q_ab_ptr + i_glob * 2 + 1, mask=mi, other=0.0) + gqb, mask=mi)
    tl.store(g_pos_ptr + i_glob * 3 + 0, tl.load(g_pos_ptr + i_glob * 3 + 0, mask=mi, other=0.0) + gpx, mask=mi)
    tl.store(g_pos_ptr + i_glob * 3 + 1, tl.load(g_pos_ptr + i_glob * 3 + 1, mask=mi, other=0.0) + gpy, mask=mi)
    tl.store(g_pos_ptr + i_glob * 3 + 2, tl.load(g_pos_ptr + i_glob * 3 + 2, mask=mi, other=0.0) + gpz, mask=mi)


@triton.jit
def _es_energy_fwd_bwd_kernel(
    pos_ptr, q_tot_ptr, ptr_ptr, sc_ptr,
    e_atom_ptr, g_q_ptr, g_pos_ptr,
    COMPUTE_GRAD: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, DTYPE: tl.constexpr,
):
    g = tl.program_id(0)
    bi = tl.program_id(1)
    g_start = tl.load(ptr_ptr + g)
    g_end = tl.load(ptr_ptr + g + 1)
    n_g = g_end - g_start

    inv_2w = tl.load(sc_ptr + 5)
    k4pi = tl.load(sc_ptr + 9)
    self_ov0 = tl.load(sc_ptr + 10)

    i_loc = bi * BLOCK_I + tl.arange(0, BLOCK_I)
    mi = i_loc < n_g
    i_glob = g_start + i_loc

    xi = tl.load(pos_ptr + i_glob * 3 + 0, mask=mi, other=0.0)
    yi = tl.load(pos_ptr + i_glob * 3 + 1, mask=mi, other=0.0)
    zi = tl.load(pos_ptr + i_glob * 3 + 2, mask=mi, other=0.0)
    qi = tl.load(q_tot_ptr + i_glob, mask=mi, other=0.0)

    phi = tl.zeros((BLOCK_I,), dtype=DTYPE)
    gpx = tl.zeros((BLOCK_I,), dtype=DTYPE)
    gpy = tl.zeros((BLOCK_I,), dtype=DTYPE)
    gpz = tl.zeros((BLOCK_I,), dtype=DTYPE)
    z_ij = tl.zeros((BLOCK_I, BLOCK_J), dtype=DTYPE)
    c0 = _TWO_OVER_SQRT_PI * inv_2w

    for j0 in range(0, n_g, BLOCK_J):
        j_loc = j0 + tl.arange(0, BLOCK_J)
        mj = j_loc < n_g
        j_glob = g_start + j_loc

        xj = tl.load(pos_ptr + j_glob * 3 + 0, mask=mj, other=0.0)
        yj = tl.load(pos_ptr + j_glob * 3 + 1, mask=mj, other=0.0)
        zj = tl.load(pos_ptr + j_glob * 3 + 2, mask=mj, other=0.0)
        qj = tl.load(q_tot_ptr + j_glob, mask=mj, other=0.0)

        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        valid = mi[:, None] & mj[None, :] & (i_loc[:, None] != j_loc[None, :])
        d2 = tl.where(valid, dx * dx + dy * dy + dz * dz, 1.0)
        d = libdevice.sqrt(d2)
        dp = d + 1e-6
        inv_dp = 1.0 / dp
        a = d * inv_2w
        erf_a = libdevice.erf(a)
        k = tl.where(valid, erf_a * inv_dp, z_ij)
        phi = phi + tl.sum(k * qj[None, :], axis=1)

        if COMPUTE_GRAD:
            dk_dd = (c0 * libdevice.exp(-a * a) * dp - erf_a) * (inv_dp * inv_dp)
            fac = tl.where(valid, (k4pi * qi[:, None] * qj[None, :]) * dk_dd / d, z_ij)
            gpx = gpx + tl.sum(fac * dx, axis=1)
            gpy = gpy + tl.sum(fac * dy, axis=1)
            gpz = gpz + tl.sum(fac * dz, axis=1)

    # Atomic contribution in float64
    e_i = (0.5 * k4pi * qi * phi + 0.5 * self_ov0 * qi * qi).to(tl.float64)
    tl.store(e_atom_ptr + i_glob, e_i, mask=mi)

    if COMPUTE_GRAD:
        gq_i = k4pi * phi + self_ov0 * qi
        tl.store(g_q_ptr + i_glob, gq_i, mask=mi)
        tl.store(g_pos_ptr + i_glob * 3 + 0, gpx, mask=mi)
        tl.store(g_pos_ptr + i_glob * 3 + 1, gpy, mask=mi)
        tl.store(g_pos_ptr + i_glob * 3 + 2, gpz, mask=mi)


# --------------------------------------------------------------------------- Kernel 7: Deterministic Float64 Per-Graph Reductions

@triton.jit
def _graph_sum_f64_1_kernel(src_ptr, ptr_ptr, out_ptr, G):
    g = tl.program_id(0)
    if g >= G:
        return
    start = tl.load(ptr_ptr + g)
    end = tl.load(ptr_ptr + g + 1)
    acc = tl.zeros((), dtype=tl.float64)
    for i in range(start, end):
        acc = acc + tl.load(src_ptr + i).to(tl.float64)
    tl.store(out_ptr + g, acc)


@triton.jit
def _graph_sum_f64_2_kernel(src_ptr, ptr_ptr, out_ptr, G):
    g = tl.program_id(0)
    if g >= G:
        return
    start = tl.load(ptr_ptr + g)
    end = tl.load(ptr_ptr + g + 1)
    a0 = tl.zeros((), dtype=tl.float64)
    a1 = tl.zeros((), dtype=tl.float64)
    for i in range(start, end):
        a0 = a0 + tl.load(src_ptr + i * 2 + 0).to(tl.float64)
        a1 = a1 + tl.load(src_ptr + i * 2 + 1).to(tl.float64)
    tl.store(out_ptr + g * 2 + 0, a0)
    tl.store(out_ptr + g * 2 + 1, a1)


@triton.jit
def _charge_cons_fwd_kernel(
    q_in_ptr, fk_raw_ptr, batch_ptr, ptr_ptr, q_tgt_ptr,
    q_out_ptr, fk_ptr, fk_sum_ptr, delta_ptr,
    G,
    DTYPE: tl.constexpr,
):
    """Per-graph deterministic charge conservation in float64 (zero atomics, 1 launch!)."""
    g = tl.program_id(0)
    if g >= G:
        return
    start = tl.load(ptr_ptr + g)
    end = tl.load(ptr_ptr + g + 1)
    fks0 = tl.zeros((), dtype=tl.float64)
    fks1 = tl.zeros((), dtype=tl.float64)
    qs0 = tl.zeros((), dtype=tl.float64)
    qs1 = tl.zeros((), dtype=tl.float64)
    for i in range(start, end):
        fks0 = fks0 + tl.load(fk_raw_ptr + i * 2 + 0).to(tl.float64)
        fks1 = fks1 + tl.load(fk_raw_ptr + i * 2 + 1).to(tl.float64)
        qs0 = qs0 + tl.load(q_in_ptr + i * 2 + 0).to(tl.float64)
        qs1 = qs1 + tl.load(q_in_ptr + i * 2 + 1).to(tl.float64)
    s0 = fks0.to(DTYPE)
    s1 = fks1.to(DTYPE)
    s0_safe = tl.where(s0 == 0.0, tl.zeros((), dtype=DTYPE) + 1.0, s0)
    s1_safe = tl.where(s1 == 0.0, tl.zeros((), dtype=DTYPE) + 1.0, s1)
    inv_s0 = 1.0 / s0_safe
    inv_s1 = 1.0 / s1_safe
    qt0 = tl.load(q_tgt_ptr + g * 2 + 0)
    qt1 = tl.load(q_tgt_ptr + g * 2 + 1)
    d0 = qt0 - qs0.to(DTYPE)
    d1 = qt1 - qs1.to(DTYPE)
    tl.store(fk_sum_ptr + g * 2 + 0, s0)
    tl.store(fk_sum_ptr + g * 2 + 1, s1)
    tl.store(delta_ptr + g * 2 + 0, d0)
    tl.store(delta_ptr + g * 2 + 1, d1)
    for i in range(start, end):
        f0 = tl.load(fk_raw_ptr + i * 2 + 0) * inv_s0
        f1 = tl.load(fk_raw_ptr + i * 2 + 1) * inv_s1
        tl.store(fk_ptr + i * 2 + 0, f0)
        tl.store(fk_ptr + i * 2 + 1, f1)
        q0 = tl.load(q_in_ptr + i * 2 + 0) + f0 * d0
        q1 = tl.load(q_in_ptr + i * 2 + 1) + f1 * d1
        tl.store(q_out_ptr + i * 2 + 0, q0)
        tl.store(q_out_ptr + i * 2 + 1, q1)


@triton.jit
def _charge_cons_bwd_kernel(
    g_qout_ptr, fk_ptr, fk_sum_ptr, delta_ptr, ptr_ptr,
    g_qin_ptr, g_fk_raw_ptr,
    G,
    DTYPE: tl.constexpr,
):
    g = tl.program_id(0)
    if g >= G:
        return
    start = tl.load(ptr_ptr + g)
    end = tl.load(ptr_ptr + g + 1)
    d0 = tl.load(delta_ptr + g * 2 + 0)
    d1 = tl.load(delta_ptr + g * 2 + 1)
    s0 = tl.load(fk_sum_ptr + g * 2 + 0)
    s1 = tl.load(fk_sum_ptr + g * 2 + 1)

    dot0 = tl.zeros((), dtype=tl.float64)
    dot1 = tl.zeros((), dtype=tl.float64)
    for i in range(start, end):
        gq0 = tl.load(g_qout_ptr + i * 2 + 0).to(tl.float64)
        gq1 = tl.load(g_qout_ptr + i * 2 + 1).to(tl.float64)
        f0 = tl.load(fk_ptr + i * 2 + 0).to(tl.float64)
        f1 = tl.load(fk_ptr + i * 2 + 1).to(tl.float64)
        dot0 = dot0 + gq0 * f0
        dot1 = dot1 + gq1 * f1
    dot0_t = dot0.to(DTYPE)
    dot1_t = dot1.to(DTYPE)

    nz0 = s0 != 0.0
    nz1 = s1 != 0.0
    inv_s0 = tl.where(nz0, 1.0 / tl.where(nz0, s0, tl.zeros((), dtype=DTYPE) + 1.0), tl.zeros((), dtype=DTYPE) + 1.0)
    inv_s1 = tl.where(nz1, 1.0 / tl.where(nz1, s1, tl.zeros((), dtype=DTYPE) + 1.0), tl.zeros((), dtype=DTYPE) + 1.0)
    # Note: sum_j (g_fk_j * fk_j) = delta * dot!
    sub0 = tl.where(nz0, d0 * dot0_t, tl.zeros((), dtype=DTYPE))
    sub1 = tl.where(nz1, d1 * dot1_t, tl.zeros((), dtype=DTYPE))

    for i in range(start, end):
        gq0 = tl.load(g_qout_ptr + i * 2 + 0)
        gq1 = tl.load(g_qout_ptr + i * 2 + 1)
        tl.store(g_qin_ptr + i * 2 + 0, gq0 - dot0_t)
        tl.store(g_qin_ptr + i * 2 + 1, gq1 - dot1_t)
        tl.store(g_fk_raw_ptr + i * 2 + 0, (gq0 * d0 - sub0) * inv_s0)
        tl.store(g_fk_raw_ptr + i * 2 + 1, (gq1 * d1 - sub1) * inv_s1)


# --------------------------------------------------------------------------- Model Inspection & Compilation

def _extract_u3_triples(U3_np: np.ndarray):
    triples: Dict[Tuple[int, int, int], np.ndarray] = {}
    K = U3_np.shape[3]
    for i in range(16):
        for j in range(16):
            for l in range(16):
                v = U3_np[i, j, l]
                if float(np.max(np.abs(v))) > 1e-12:
                    key = tuple(sorted((i, j, l)))
                    if key not in triples:
                        triples[key] = np.zeros(K, dtype=U3_np.dtype)
                    triples[key] = triples[key] + v
    keys = tuple(sorted(k for k, v in triples.items() if float(np.max(np.abs(v))) > 1e-12))
    mat = np.stack([triples[k] for k in keys], axis=0)
    return keys, mat


def _unpack_ln_mlp(w: Dict[str, np.ndarray], pfx: str, _g):
    """Unpack 3-layer LayerNorm-SiLU MLP weights ``(ln0_w, ln0_b, W1_T, b1, ln1_w, ln1_b, W2_T, b2, ln2_w, ln2_b)``."""
    return (
        _g(w[f"{pfx}1_w"]), _g(w[f"{pfx}1_b"]),
        _g(w[f"{pfx}3_w"].T), _g(w[f"{pfx}3_b"]), _g(w[f"{pfx}4_w"]), _g(w[f"{pfx}4_b"]),
        _g(w[f"{pfx}6_w"].T), _g(w[f"{pfx}6_b"]), _g(w[f"{pfx}7_w"]), _g(w[f"{pfx}7_b"]),
    )


def _unpack_c0_tables(w: Dict[str, np.ndarray], pfx: str, _g):
    """Pre-contract 0e SymmetricContraction U_matrix_{1,2,3} into ``(C1, C2, C3)``."""
    U1, U2, U3 = w[f"{pfx}U1"], w[f"{pfx}U2"], w[f"{pfx}U3"]
    C1 = _g(w[f"{pfx}weights_1"][0, 0] * U1[0, 0])
    u2_scales = np.array([U2[0, 0, 0], U2[1, 1, 1], U2[4, 4, 2], U2[9, 9, 3]])
    C2 = _g(w[f"{pfx}weights_0"][0] * u2_scales[:, None])
    keys, U3_u = _extract_u3_triples(U3)
    if keys != _CANONICAL_U3_TRIPLES:
        raise ValueError(f"Non-canonical {pfx}U_matrix_3 triples")
    return C1, C2, _g(U3_u @ w[f"{pfx}weights_max"][0])


class CompiledPolarMACE:
    """Extracted weights, precomputed species tables, and lazily cached M_fused tensors for PolarMACE."""

    def __init__(self, w: Dict[str, np.ndarray], dtype: str = "float32"):
        self.dtype = dtype
        self.dt = cp.float64 if dtype == "float64" else cp.float32
        self.tl_dt = _tl_dtype(self.dt)
        self.num_interactions = int(w["num_interactions"])
        if self.num_interactions != 1:
            # only the MACE-POLAR-1-S layout (one interaction, scalar hidden irreps) is implemented and checked against
            # mace-torch (float64 energies and forces equal to round-off)
            raise ValueError(f"the Triton MACE engine supports MACE-POLAR-1-S only (model has {self.num_interactions} "
                             "interactions)")

        def _g(arr: np.ndarray, dt=self.dt) -> cp.ndarray:
            return gpu.as_gpu(arr, dtype=dt)

        z_arr = [int(z) for z in w["atomic_numbers"]]
        self.num_species = len(z_arr)
        self.z_to_idx: Dict[int, int] = {z: i for i, z in enumerate(z_arr)}
        self.sym_to_idx: Dict[str, int] = {sym: self.z_to_idx[z] for sym, z in _SYM_TO_Z.items() if z in self.z_to_idx}

        self.r_max = float(w["r_max"])
        self.bessel_w = _g(w["bessel_weights"])
        self.bessel_pref = float(w["bessel_pref"])
        self.atomic_e = _g(w["atomic_energies"].reshape(-1), cp.float64)
        self.alpha = float(w["inter0_alpha"])
        self.beta = float(w["inter0_beta"])
        self.scale = float(w["energy_scale"])
        self.shift = float(w["energy_shift"])
        self.cst_silu = float(w["inter0_cst_silu"])
        self.cst_sigmoid = float(w["inter0_cst_sigmoid"])
        self.cst_fk_silu = float(w["fk_cst_silu"])

        # Precomputed per-species tables (NumPy float64 -> GPU in self.dt)
        E_node = w["node_emb_w"] * float(w["node_emb_pw"])         # (Z, 512)
        W_up = w["inter0_lin_up_0_0"]                              # (512, 512)
        W_skip = w["inter0_skip_0_0"]                              # (512, 512)
        W_res = w["inter0_lin_res_0_0"]                            # (512, 2048)
        E_node_up = E_node @ W_up                                  # (Z, 512)
        self.E_node_up = _g(E_node_up)
        self.E_sc = _g(E_node @ W_skip)                            # (Z, 512)
        self.E_res = _g(E_node_up @ W_res)                         # (Z, 2048)
        E_src = w["inter0_src_emb_0_0"]                            # (Z, 512)
        E_tgt = w["inter0_tgt_emb_0_0"]                            # (Z, 512)

        # Edge conv_tp_weights MLP
        W_tp0, b_tp0 = w["inter0_mlp_0_w"], w["inter0_mlp_0_b"]
        self.W_tp0_r_T = _g(W_tp0[:, :8].T)
        self.E_tp_src = _g(E_src @ W_tp0[:, 8:520].T + b_tp0)
        self.E_tp_tgt = _g(E_tgt @ W_tp0[:, 520:1032].T)
        (
            self.tp_ln0_w, self.tp_ln0_b, self.W_tp1_T, self.b_tp1,
            self.tp_ln1_w, self.tp_ln1_b, self.W_tp2_T, self.b_tp2,
            self.tp_ln2_w, self.tp_ln2_b,
        ) = _unpack_ln_mlp(w, "inter0_mlp_", _g)

        # Edge density_fn MLP
        W_den0 = w["inter0_den_0_w"]
        b_den0 = w["inter0_den_0_b"]
        self.W_den0_r_T = _g(W_den0[:, :8].T)
        self.E_den_src = _g(E_src @ W_den0[:, 8:520].T + b_den0)
        self.E_den_tgt = _g(E_tgt @ W_den0[:, 520:1032].T)
        self.den_ln0_w = _g(w["inter0_den_1_w"])
        self.den_ln0_b = _g(w["inter0_den_1_b"])
        self.W_den1 = _g(w["inter0_den_3_w"].reshape(64))
        self.b_den1 = _g(w["inter0_den_3_b"].reshape(1))

        # W9_aug & linear_1 -> lazily cached per-species M_fused
        W9 = w["inter0_mlp_9_w"].reshape(4, 512, 64)
        b9 = w["inter0_mlp_9_b"].reshape(4, 512, 1)
        tp_pw = w["inter0_tp_path_weights"]
        W9_aug = np.concatenate([W9, b9], axis=-1)
        for l in range(4):
            pw = float(tp_pw[l]) / math.sqrt(2 * l + 1)
            if abs(pw - 1.0) > 1e-12:
                W9_aug[l] = W9_aug[l] * pw
        self.W9_aug = _g(W9_aug)                                   # (4, 512, 65)
        self.W_L1_T = [_g(w[f"inter0_lin1_{l}_{l}"]) for l in range(4)]
        self._m_fused_cache: Dict[int, Tuple[cp.ndarray, ...]] = {}

        # linear_2
        self.W_L2_T = [_g(w[f"inter0_lin2_{l}_{l}"]) for l in range(4)]
        self.W_L2 = [_g(w[f"inter0_lin2_{l}_{l}"].T) for l in range(4)]

        # SymmetricContraction pre-contracted tables
        self.C1, self.C2, self.C3 = _unpack_c0_tables(w, "prod0_c0_", _g)

        # Post-product linear maps
        self.W_prod_T = _g(w["prod0_lin_0_0"])
        self.W_prod = _g(w["prod0_lin_0_0"].T)
        self.W_ro = _g(w["ro0_linear_0_0"].reshape(512))
        W_lr_T = np.concatenate([w["lr0_lin_0_0"], w["lr0_lin_0_2"]], axis=1)
        self.W_lr_T = _g(W_lr_T)                                       # (512, 2)
        self.W_lr = _g(W_lr_T.T)
        self.W_mix_T = _g(w["mix_0_0_0"])
        self.W_mix = _g(w["mix_0_0_0"].T)

        # Fukui MLP
        self.W_fk1_T = _g(w["fk_lin1_0_0"])
        self.W_fkm_T = _g(w["fk_linm_0_0"])
        self.b_fkm = _g(w["fk_linm_bias_0"])
        self.W_fk2_T = _g(w["fk_lin2_0_0"])
        self.b_fk2 = _g(w["fk_lin2_bias_0"])

        # Field-dependent charge maps (algebraically merged)
        self.fdcs = []
        r = 0
        while f"fdc{r}_dot_w" in w:
            p = f"fdc{r}_"
            W_pot = np.concatenate([w[f"{p}pe_pot_0_0"], w[f"{p}pe_pot_2_0"]], axis=0)   # (4, 512)
            W_nf = w[f"{p}pe_nf_0_0"]                                                   # (512, 512)
            W_chg = np.concatenate([w[f"{p}pe_chg_0_0"], w[f"{p}pe_chg_2_0"]], axis=0)   # (2, 512)
            W_dot = w[f"{p}dot_w"].reshape(512, 512) * float(w[f"{p}dot_pw"])
            W_dot_pot_T = _g(W_pot @ W_dot.T)                                           # (4, 512)
            W_dot_nf_T = _g(W_nf @ W_dot.T)                                             # (512, 512)
            W_dot_chg_T = _g(W_chg @ W_dot.T)                                           # (2, 512)

            W_fsrc = w[f"{p}src_emb_0_0"]                                               # (Z, 512)
            W_nl0 = w[f"{p}nl_0_w"]
            b_nl0 = w[f"{p}nl_0_b"]
            W_nl0_inv_T = _g(W_nl0[:, :512].T)                                          # (512, 64)
            E_fdc_src = _g(W_fsrc @ W_nl0[:, 512:].T + b_nl0)                           # (Z, 64)

            W_tpout_T = _g((w[f"{p}tp_out_w"].reshape(512, 512) * float(w[f"{p}tp_out_pw"])).T)
            W_r1_T = _g(w[f"{p}ro_lin1_0_0"][:, :32])
            W_rm_T = _g(w[f"{p}ro_linm_0_0"][:32, :32])
            b_rm = _g(w[f"{p}ro_linm_bias_0"][:32])
            W_r2_T = _g(np.concatenate([w[f"{p}ro_lin2_0_0"], w[f"{p}ro_lin2_0_2"], w[f"{p}ro_lin2_0_4"]], axis=1)[:32, :])
            b_r2 = _g(np.concatenate([w[f"{p}ro_lin2_bias_0"], w[f"{p}ro_lin2_bias_2"], w[f"{p}ro_lin2_bias_4"]], axis=0))
            cst_r = float(w[f"{p}ro_cst"])

            ln0_w, ln0_b, W_nl1_T, b_nl1, ln1_w, ln1_b, W_nl2_T, b_nl2, ln2_w, ln2_b = _unpack_ln_mlp(w, f"{p}nl_", _g)
            self.fdcs.append(dict(
                W_dot_pot_T=W_dot_pot_T, W_dot_nf_T=W_dot_nf_T, W_dot_chg_T=W_dot_chg_T,
                W_nl0_inv_T=W_nl0_inv_T, E_fdc_src=E_fdc_src,
                ln0_w=ln0_w, ln0_b=ln0_b, W_nl1_T=W_nl1_T, b_nl1=b_nl1,
                ln1_w=ln1_w, ln1_b=ln1_b, W_nl2_T=W_nl2_T, b_nl2=b_nl2,
                ln2_w=ln2_w, ln2_b=ln2_b,
                W_nl3_T=_g(w[f"{p}nl_9_w"].T), b_nl3=_g(w[f"{p}nl_9_b"]),
                W_tpout_T=W_tpout_T,
                W_r1_T=W_r1_T, W_rm_T=W_rm_T, b_rm=b_rm, W_r2_T=W_r2_T, b_r2=b_r2, cst_r=cst_r,
            ))
            r += 1

        # Local electron energy (ZERO 512x512 GEMMs!)
        W_lq = np.concatenate([w["lee_q_0_0"], w["lee_q_2_0"]], axis=0)                 # (2, 512)
        b_lq = w["lee_q_bias_0"]                                                        # (512,)
        W_lv = np.concatenate([w["lee_v_0_0"], w["lee_v_2_0"]], axis=0)                 # (4, 512)
        b_lv = w["lee_v_bias_0"]                                                        # (512,)
        W_dq = w["lee_dotq_w"].reshape(512, 512) * float(w["lee_dotq_pw"])
        W_dv = w["lee_dotv_w"].reshape(512, 512) * float(w["lee_dotv_pw"])
        self.W_le_q_T = _g(W_lq @ W_dq.T)                                               # (2, 512)
        self.b_le_q = _g(W_dq @ b_lq)                                                   # (512,)
        self.W_le_v_T = _g(W_lv @ W_dv.T)                                               # (4, 512)
        self.b_le_v = _g(W_dv @ b_lv)                                                   # (512,)
        self.W_le0_q_T = _g(w["lee_mlp_0_w"][:, :512].T)
        self.W_le0_v_T = _g(w["lee_mlp_0_w"][:, 512:].T)
        self.b_le0 = _g(w["lee_mlp_0_b"])
        (
            self.le_ln0_w, self.le_ln0_b, self.W_le1_T, self.b_le1,
            self.le_ln1_w, self.le_ln1_b, self.W_le2_T, self.b_le2,
            self.le_ln2_w, self.le_ln2_b,
        ) = _unpack_ln_mlp(w, "lee_mlp_", _g)
        self.W_le3 = _g(w["lee_mlp_9_w"].reshape(128))
        self.b_le3 = float(w["lee_mlp_9_b"].item())

        # Electrostatics constants
        self.k4pi = _FIELD_CONSTANT / (4.0 * math.pi)
        w0, w1 = float(w["es_total_width_factors"][0]), float(w["es_total_width_factors"][1])
        self.inv_2w0 = 0.5 / w0
        self.inv_2w1 = 0.5 / w1
        fn = w["es_field_norms"]
        l0f = w["es_l0_factors"]
        self.es_sc0 = self.k4pi * float(l0f[0] / fn[0])
        self.es_sc1 = self.k4pi * float(l0f[1] / fn[1])
        self.es_self0 = float(w["ce_self_overlap_constants"][0])

        # Pack all model scalar constants into a device array of dtype `self.dt`
        self.scalars = _g(np.array([
            self.r_max, 1.0 / self.r_max, self.bessel_pref,
            self.cst_silu, self.cst_sigmoid,
            self.inv_2w0, self.inv_2w1, self.es_sc0, self.es_sc1,
            self.k4pi, self.es_self0,
        ], dtype=np.float64))

    def get_m_fused(self, uniq_z_list: Sequence[int]) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """Lazily compute and cache per-species fused (W9 * h_up) @ W_L1^T matrices."""
        for z in uniq_z_list:
            if z not in self._m_fused_cache:
                h_up = self.E_node_up[z]                                   # (512,)
                mats = []
                for l in range(4):
                    hW_T = cp.ascontiguousarray((h_up[:, None] * self.W9_aug[l]).T)
                    mats.append(_mm(hW_T, self.W_L1_T[l]))
                self._m_fused_cache[z] = tuple(mats)
        M0 = cp.ascontiguousarray(cp.concatenate([self._m_fused_cache[z][0] for z in uniq_z_list], axis=0))
        M1 = cp.ascontiguousarray(cp.concatenate([self._m_fused_cache[z][1] for z in uniq_z_list], axis=0))
        M2 = cp.ascontiguousarray(cp.concatenate([self._m_fused_cache[z][2] for z in uniq_z_list], axis=0))
        M3 = cp.ascontiguousarray(cp.concatenate([self._m_fused_cache[z][3] for z in uniq_z_list], axis=0))
        return M0, M1, M2, M3


_COMPILED_CACHE: Dict[Tuple[str, str, str], CompiledPolarMACE] = {}


def load_compiled(
    model: str = "polar-1-s",
    model_path: Optional[str | Path] = None,
    dtype: str = "float32",
) -> CompiledPolarMACE:
    """Load and cache CompiledPolarMACE from .model -> ~/.cache/jacobik/mace_cache (zero torch)."""
    key = (str(model), "" if model_path is None else str(Path(model_path).expanduser()), str(dtype))
    cm = _COMPILED_CACHE.get(key)
    if cm is None:
        w = mace_weights.load_weights(model=model, model_path=model_path)
        cm = _COMPILED_CACHE[key] = CompiledPolarMACE(w, dtype=dtype)
    return cm


# --------------------------------------------------------------------------- Static Batch Plan

class TritonBatchPlan:
    """Pre-sorted CSR edge buckets, reverse-edge map, and precomputed per-node/per-edge constants."""

    def __init__(
        self,
        cm: CompiledPolarMACE,
        z_idx: cp.ndarray,           # (N,) int64
        batch: cp.ndarray,           # (N,) int64
        ptr: cp.ndarray,             # (G+1,) int64
        total_charge: cp.ndarray,    # (G,) float
        total_spin: cp.ndarray,      # (G,) float
        pi: cp.ndarray,              # (E,) int64 senders
        pj: cp.ndarray,              # (E,) int64 receivers
    ):
        self.cm = cm
        dt = cm.dt
        self.N = int(z_idx.size)
        self.G = int(ptr.size - 1)
        self.E = int(pi.size)
        self.z_idx = gpu.as_gpu(z_idx, dtype=cp.int64)
        self.batch = gpu.as_gpu(batch, dtype=cp.int64)
        self.ptr = gpu.as_gpu(ptr, dtype=cp.int64)
        self.n_max = int((self.ptr[1:] - self.ptr[:-1]).max()) if self.G > 0 else 0

        # Unique species in batch
        uniq_z, s_loc = cp.unique(self.z_idx, return_inverse=True)
        self.S = int(uniq_z.size)
        self.M0, self.M1, self.M2, self.M3 = cm.get_m_fused(uniq_z.tolist())

        # Sort edges by (dst, s_loc[src], src) -> contiguous (v, s) buckets and node slices
        pi = gpu.as_gpu(pi, dtype=cp.int64)
        pj = gpu.as_gpu(pj, dtype=cp.int64)
        if self.E > 0:
            sort_key = (pj * self.S + s_loc[pi]) * self.N + pi
            order = cp.argsort(sort_key)
            self.src = cp.ascontiguousarray(pi[order])
            self.dst = cp.ascontiguousarray(pj[order])
            bkt = self.dst * self.S + s_loc[self.src]
            counts = cp.bincount(bkt, minlength=self.N * self.S)
            self.seg_ptr = cp.concatenate([cp.zeros(1, dtype=cp.int32), cp.cumsum(counts, dtype=cp.int32)])
            self.node_ptr = cp.ascontiguousarray(self.seg_ptr[:: self.S])

            fwd_key = self.dst * self.N + self.src
            rev_key = self.src * self.N + self.dst
            o_fwd = cp.argsort(fwd_key)
            o_rev = cp.argsort(rev_key)
            self.rev_edge = cp.empty(self.E, dtype=cp.int32)
            self.rev_edge[o_fwd] = o_rev.astype(cp.int32)

            self.edge_tp_bias = cp.ascontiguousarray(cm.E_tp_src[self.src_z()] + cm.E_tp_tgt[self.dst_z()])
            self.edge_den_bias = cp.ascontiguousarray(cm.E_den_src[self.src_z()] + cm.E_den_tgt[self.dst_z()])
        else:
            self.src = cp.ascontiguousarray(pi)
            self.dst = cp.ascontiguousarray(pj)
            self.seg_ptr = cp.zeros(self.N * self.S + 1, dtype=cp.int32)
            self.node_ptr = cp.zeros(self.N + 1, dtype=cp.int32)
            self.rev_edge = cp.empty(0, dtype=cp.int32)
            self.edge_tp_bias = cp.empty((0, 64), dtype=dt)
            self.edge_den_bias = cp.empty((0, 64), dtype=dt)

        # Precompute constant per-atom lookups
        self.node_e0 = cp.ascontiguousarray(cm.atomic_e[self.z_idx])       # (N,) float64
        self.node_sc = cp.ascontiguousarray(cm.E_sc[self.z_idx])           # (N, 512)
        self.node_res = cp.ascontiguousarray(cm.E_res[self.z_idx])         # (N, 2048)
        self.node_fdc_src = [cp.ascontiguousarray(fdc["E_fdc_src"][self.z_idx]) for fdc in cm.fdcs]

        tc = gpu.as_gpu(total_charge, dtype=dt)
        ts = gpu.as_gpu(total_spin, dtype=dt)
        self.q_tgt_g = cp.ascontiguousarray(cp.stack([(tc + (ts - 1.0)) * 0.5, (tc - (ts - 1.0)) * 0.5], axis=-1))
        self.ptr_i32 = cp.ascontiguousarray(self.ptr.astype(cp.int32))
        self.batch_i32 = cp.ascontiguousarray(self.batch.astype(cp.int32))
        self.src_i32 = cp.ascontiguousarray(self.src.astype(cp.int32))
        self.dst_i32 = cp.ascontiguousarray(self.dst.astype(cp.int32))

    def src_z(self):
        return self.z_idx[self.src]

    def dst_z(self):
        return self.z_idx[self.dst]

    @classmethod
    def from_structures(
        cls,
        cm: CompiledPolarMACE,
        structs: Sequence[Tuple[Sequence[str], np.ndarray, int, int]],
        X: Optional[cp.ndarray] = None,
        r_list: Optional[float] = None,
    ) -> Tuple["TritonBatchPlan", Optional[Tuple[cp.ndarray, cp.ndarray]]]:
        """Build a TritonBatchPlan directly from raw structures (ZERO ASE / PyG DataLoader!)."""
        sym_to_idx = cm.sym_to_idx
        z_list, b_list, ptr_list, q_list, s_list = [], [], [0], [], []
        for g, (syms, _pos, q, spin) in enumerate(structs):
            na = len(syms)
            z_list.extend(sym_to_idx[s] for s in syms)
            b_list.extend([g] * na)
            ptr_list.append(ptr_list[-1] + na)
            q_list.append(float(q))
            s_list.append(float(spin))
        z_idx = cp.asarray(z_list, dtype=cp.int64)
        batch = cp.asarray(b_list, dtype=cp.int64)
        ptr = cp.asarray(ptr_list, dtype=cp.int64)
        tc = cp.asarray(q_list, dtype=cm.dt)
        ts = cp.asarray(s_list, dtype=cm.dt)

        ii, jj = [], []
        for g in range(len(structs)):
            a = cp.arange(ptr_list[g], ptr_list[g + 1], dtype=cp.int64)
            I, J = cp.meshgrid(a, a, indexing="ij")
            m = I != J
            ii.append(I[m])
            jj.append(J[m])
        pi = cp.concatenate(ii) if ii else cp.empty(0, dtype=cp.int64)
        pj = cp.concatenate(jj) if jj else cp.empty(0, dtype=cp.int64)

        out_pairs = None
        if X is not None and r_list is not None and pi.size > 0:
            keep = ((X[pi] - X[pj]) ** 2).sum(-1) < r_list * r_list
            out_pairs = (pi[~keep], pj[~keep])
            pi, pj = pi[keep], pj[keep]
        return cls(cm, z_idx, batch, ptr, tc, ts, pi, pj), out_pairs


# --------------------------------------------------------------------------- Forward + Analytical Backward Execution

def _scaled_silu_fwd(x: cp.ndarray, cst: float):
    sig = 1.0 / (1.0 + cp.exp(-x))
    return (cst * x * sig), sig


def _scaled_silu_bwd(g_out: cp.ndarray, x: cp.ndarray, sig: cp.ndarray, cst: float):
    return g_out * (cst * sig * (1.0 + x * (1.0 - sig)))


def _bias_ln_silu_fwd(
    x: cp.ndarray, bias: cp.ndarray, lnw: cp.ndarray, lnb: cp.ndarray,
    save_bwd: bool, row_bias: bool, tl_dt,
):
    M, D = x.shape
    out = cp.empty_like(x)
    if save_bwd:
        xn = cp.empty_like(x)
        rstd = cp.empty(M, dtype=x.dtype)
    else:
        xn = out
        rstd = out
    BLOCK_M = 16 if D == 64 else 8
    grid = (gpu.cdiv(M, BLOCK_M),)
    gpu.launch(
        _bias_ln_silu_fwd_kernel, grid,
        x, bias, lnw, lnb, out, xn, rstd, M,
        ROW_BIAS=row_bias, SAVE_BWD=save_bwd, D=D, BLOCK_M=BLOCK_M, DTYPE=tl_dt,
    )
    return out, xn, rstd


def _lin_l03(x: cp.ndarray, weights, out: Optional[cp.ndarray] = None, trans_b: bool = False) -> cp.ndarray:
    """Apply block-diagonal l=0..3 linear weights to ``(16, N, C)`` -> ``(16, N, D)``."""
    N, C = x.shape[1], x.shape[2]
    D = weights[0].shape[0] if trans_b else weights[0].shape[1]
    if out is None:
        out = cp.empty((16, N, D), dtype=x.dtype)
    fn = _mm_tb if trans_b else _mm
    fn(x[0], weights[0], out=out[0])
    for s, e, w in ((1, 4, weights[1]), (4, 9, weights[2]), (9, 16, weights[3])):
        fn(x[s:e].reshape((e - s) * N, C), w, out=out[s:e].reshape((e - s) * N, D))
    return out


def _mlp3_ln_silu_fwd(
    h0: cp.ndarray, b0: cp.ndarray, ln0_w: cp.ndarray, ln0_b: cp.ndarray,
    W1_T: cp.ndarray, b1: cp.ndarray, ln1_w: cp.ndarray, ln1_b: cp.ndarray,
    W2_T: cp.ndarray, b2: cp.ndarray, ln2_w: cp.ndarray, ln2_b: cp.ndarray,
    save_bwd: bool, row_bias0: bool, tl_dt,
):
    x0, xn0, rs0 = _bias_ln_silu_fwd(h0, b0, ln0_w, ln0_b, save_bwd, row_bias0, tl_dt)
    x1, xn1, rs1 = _bias_ln_silu_fwd(_mm(x0, W1_T), b1, ln1_w, ln1_b, save_bwd, False, tl_dt)
    x2, xn2, rs2 = _bias_ln_silu_fwd(_mm(x1, W2_T), b2, ln2_w, ln2_b, save_bwd, False, tl_dt)
    return x2, (xn0, rs0, xn1, rs1, xn2, rs2)


def _mlp3_ln_silu_bwd(
    g_x2: cp.ndarray, cache,
    W1_T: cp.ndarray, ln1_w: cp.ndarray, ln1_b: cp.ndarray,
    W2_T: cp.ndarray, ln2_w: cp.ndarray, ln2_b: cp.ndarray,
    ln0_w: cp.ndarray, ln0_b: cp.ndarray, tl_dt,
) -> cp.ndarray:
    xn0, rs0, xn1, rs1, xn2, rs2 = cache
    g_x1 = _mm_tb(_bias_ln_silu_bwd(g_x2, xn2, rs2, ln2_w, ln2_b, tl_dt), W2_T)
    g_x0 = _mm_tb(_bias_ln_silu_bwd(g_x1, xn1, rs1, ln1_w, ln1_b, tl_dt), W1_T)
    return _bias_ln_silu_bwd(g_x0, xn0, rs0, ln0_w, ln0_b, tl_dt)


def _ro_mlp_fwd(x: cp.ndarray, W1_T: cp.ndarray, Wm_T: cp.ndarray, bm: cp.ndarray, W2_T: cp.ndarray, b2: cp.ndarray, cst: float):
    z1 = _mm(x, W1_T)
    a1, s1 = _scaled_silu_fwd(z1, cst)
    z2 = _mm(a1, Wm_T) + bm
    a2, s2 = _scaled_silu_fwd(z2, cst)
    return _mm(a2, W2_T) + b2, (z1, s1, z2, s2)


def _ro_mlp_bwd(g_out: cp.ndarray, cache, W1_T: cp.ndarray, Wm_T: cp.ndarray, W2_T: cp.ndarray, cst: float) -> cp.ndarray:
    z1, s1, z2, s2 = cache
    g_z2 = _scaled_silu_bwd(_mm_tb(g_out, W2_T), z2, s2, cst)
    g_z1 = _scaled_silu_bwd(_mm_tb(g_z2, Wm_T), z1, s1, cst)
    return _mm_tb(g_z1, W1_T)


def _bias_ln_silu_bwd(
    g_out: cp.ndarray, xn: cp.ndarray, rstd: cp.ndarray,
    lnw: cp.ndarray, lnb: cp.ndarray, tl_dt,
):
    M, D = g_out.shape
    g_x = cp.empty_like(g_out)
    BLOCK_M = 16 if D == 64 else 8
    grid = (gpu.cdiv(M, BLOCK_M),)
    gpu.launch(
        _bias_ln_silu_bwd_kernel, grid,
        g_out, xn, rstd, lnw, lnb, g_x, M,
        D=D, BLOCK_M=BLOCK_M, DTYPE=tl_dt,
    )
    return g_x


def forward_and_forces(plan: TritonBatchPlan, pos: cp.ndarray, compute_force: bool = True):
    """Evaluate exact float64 energy (G,) and optional forces (N, 3) in model dtype via pure Triton + cuBLAS."""
    cm = plan.cm
    N, G, E, S = plan.N, plan.G, plan.E, plan.S
    dt, tl_dt = cm.dt, cm.tl_dt

    # ---- Stage 1: Edge geometry + Bessel + Cutoff + SH
    d_e = cp.empty(E, dtype=dt)
    cut_e = cp.empty(E, dtype=dt)
    bes_e = cp.empty((E, 8), dtype=dt)
    Y_e = cp.empty((E, 16), dtype=dt)
    if E > 0:
        gpu.launch(
            _edge_geom_fwd_kernel, (gpu.cdiv(E, 128),),
            pos, plan.src_i32, plan.dst_i32, cm.bessel_w, cm.scalars,
            d_e, cut_e, bes_e, Y_e,
            E,
            BLOCK_E=128, DTYPE=tl_dt,
        )

    # ---- Stage 2: Edge RadialMLP (8->64->64->64) + Density MLP (8->64->1)
    if E > 0:
        act_e, tp_cache = _mlp3_ln_silu_fwd(
            _mm(bes_e, cm.W_tp0_r_T), plan.edge_tp_bias, cm.tp_ln0_w, cm.tp_ln0_b,
            cm.W_tp1_T, cm.b_tp1, cm.tp_ln1_w, cm.tp_ln1_b,
            cm.W_tp2_T, cm.b_tp2, cm.tp_ln2_w, cm.tp_ln2_b, compute_force, True, tl_dt,
        )
        h_den0 = _mm(bes_e, cm.W_den0_r_T)
        ed_e = cp.empty(E, dtype=dt)
        den_xn = cp.empty((E, 64), dtype=dt) if compute_force else ed_e
        den_rs = cp.empty(E, dtype=dt) if compute_force else ed_e
        den_d1 = cp.empty(E, dtype=dt) if compute_force else ed_e
        gpu.launch(
            _edge_density_fwd_kernel, (gpu.cdiv(E, 16),),
            h_den0, plan.edge_den_bias, cut_e, cm.den_ln0_w, cm.den_ln0_b, cm.W_den1, cm.b_den1,
            ed_e, den_xn, den_rs, den_d1, E, SAVE_BWD=compute_force, BLOCK_E=16, DTYPE=tl_dt,
        )
    else:
        act_e = cp.empty((0, 64), dtype=dt)
        ed_e = cp.empty(0, dtype=dt)

    # ---- Stage 3: Species-factored CSR accumulation into A_all (16, N, S, 65) + M_fused GEMMs
    A_all = cp.empty((16, N, S, 65), dtype=dt)
    den_part = cp.empty((N, S), dtype=dt)
    gpu.launch(
        _edge_to_A_fwd_kernel, (N, S),
        plan.seg_ptr, cut_e, Y_e, act_e, ed_e,
        A_all, den_part, N, S, DTYPE=tl_dt,
    )
    density = den_part.sum(axis=1)                                          # (N,)
    inv_den = 1.0 / (density * cm.beta + cm.alpha)

    out_l1_0 = _mm(A_all[0].reshape(N, S * 65), plan.M0)                    # (N, 2048)
    out_l1_vec = cp.empty((15, N, 512), dtype=dt)
    _mm(A_all[1:4].reshape(3 * N, S * 65), plan.M1, out=out_l1_vec[0:3].reshape(3 * N, 512))
    _mm(A_all[4:9].reshape(5 * N, S * 65), plan.M2, out=out_l1_vec[3:8].reshape(5 * N, 512))
    _mm(A_all[9:16].reshape(7 * N, S * 65), plan.M3, out=out_l1_vec[8:15].reshape(7 * N, 512))

    # ---- Stage 4: DivDensity + AddRes + Gate
    gated = cp.empty((16, N, 512), dtype=dt)
    BLOCK_C = 64
    N_CB = 512 // BLOCK_C
    gpu.launch(
        _div_res_gate_fwd_kernel, (N, N_CB),
        out_l1_0, out_l1_vec, plan.node_res, inv_den, cm.scalars, gated,
        N, BLOCK_C=BLOCK_C, DTYPE=tl_dt,
    )

    # ---- Stage 5: linear_2 -> X16 (16, N, 512) -> Register-Only SymmetricContraction -> sc_out (N, 512)
    X16 = _lin_l03(gated, cm.W_L2_T)

    sc_out = cp.empty((N, 512), dtype=dt)
    gpu.launch(_sym_contr_fwd_kernel, (N, N_CB), X16, cm.C1, cm.C2, cm.C3, sc_out, N, C=512, BLOCK_C=BLOCK_C)

    # ---- Stage 6: Product linear + readout + lr_src + mixer + fukui
    node_feats = _mm(sc_out, cm.W_prod_T) + plan.node_sc                  # (N, 512)
    node_inter_es = _mv(node_feats, cm.W_ro) * cm.scale + cm.shift          # (N,)
    node_e_loc = plan.node_e0 + node_inter_es.astype(cp.float64)
    e_local = cp.empty(G, dtype=cp.float64)
    gpu.launch(_graph_sum_f64_1_kernel, (G,), node_e_loc, plan.ptr_i32, e_local, G)

    features_mixed = _mm(node_feats, cm.W_mix_T)                            # (N, 512)
    q_ab_raw = _mm(node_feats, cm.W_lr_T)                                   # (N, 2)
    fk_raw0, fk_cache = _ro_mlp_fwd(node_feats, cm.W_fk1_T, cm.W_fkm_T, cm.b_fkm, cm.W_fk2_T, cm.b_fk2, cm.cst_fk_silu)

    q_ab0 = cp.empty((N, 2), dtype=dt)
    fk0 = cp.empty((N, 2), dtype=dt)
    fk_sum0 = cp.empty((G, 2), dtype=dt)
    delta0 = cp.empty((G, 2), dtype=dt)
    gpu.launch(
        _charge_cons_fwd_kernel, (G,),
        q_ab_raw, fk_raw0, plan.batch_i32, plan.ptr_i32, plan.q_tgt_g,
        q_ab0, fk0, fk_sum0, delta0, G, DTYPE=tl_dt,
    )

    # ---- Stage 7: SCF recursion steps (tiled-SRAM monopole electrostatics + merged FDC)
    BLOCK_I, BLOCK_J = 32, 32
    es_grid = (G, gpu.cdiv(max(plan.n_max, 1), BLOCK_I))
    q_cur = q_ab0
    scf_cache = []

    for r, fdc in enumerate(cm.fdcs):
        pot_l0 = cp.empty((N, 4), dtype=dt)
        gpu.launch(
            _es_features_fwd_kernel, es_grid,
            pos, q_cur, plan.ptr_i32, cm.scalars, pot_l0,
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, DTYPE=tl_dt,
        )

        wx2 = (
            _mm(features_mixed, fdc["W_dot_nf_T"])
            + _mm(pot_l0, fdc["W_dot_pot_T"])
            + _mm(q_cur, fdc["W_dot_chg_T"])
        )
        inv_f = features_mixed * wx2

        nl_x2, nl_cache = _mlp3_ln_silu_fwd(
            _mm(inv_f, fdc["W_nl0_inv_T"]), plan.node_fdc_src[r], fdc["ln0_w"], fdc["ln0_b"],
            fdc["W_nl1_T"], fdc["b_nl1"], fdc["ln1_w"], fdc["ln1_b"],
            fdc["W_nl2_T"], fdc["b_nl2"], fdc["ln2_w"], fdc["ln2_b"], compute_force, True, tl_dt,
        )
        nonlin = _mm(nl_x2, fdc["W_nl3_T"]) + fdc["b_nl3"]
        tp_m = _mm(nonlin, fdc["W_tpout_T"])
        new_f = features_mixed * tp_m
        out_r, r_cache = _ro_mlp_fwd(new_f, fdc["W_r1_T"], fdc["W_rm_T"], fdc["b_rm"], fdc["W_r2_T"], fdc["b_r2"], fdc["cst_r"])

        q_pre = q_cur + out_r[:, :2]
        fk_raw_r = cp.ascontiguousarray(out_r[:, 2:])
        q_next = cp.empty((N, 2), dtype=dt)
        fk_r = cp.empty((N, 2), dtype=dt)
        fk_sum_r = cp.empty((G, 2), dtype=dt)
        delta_r = cp.empty((G, 2), dtype=dt)
        gpu.launch(
            _charge_cons_fwd_kernel, (G,),
            q_pre, fk_raw_r, plan.batch_i32, plan.ptr_i32, plan.q_tgt_g,
            q_next, fk_r, fk_sum_r, delta_r, G, DTYPE=tl_dt,
        )

        if compute_force:
            scf_cache.append((q_cur, wx2, nl_cache, tp_m, r_cache, fk_r, fk_sum_r, delta_r))
        q_cur = q_next

    pot_final = pot_l0

    # ---- Stage 8: Local Electron Energy (ZERO 512x512 GEMMs!)
    q_sum_in = q_cur + q_ab0
    uq = _mm(q_sum_in, cm.W_le_q_T) + cm.b_le_q
    uv = _mm(pot_final, cm.W_le_v_T) + cm.b_le_v
    dq_f = node_feats * uq
    dv_f = node_feats * uv
    le_x2, le_cache = _mlp3_ln_silu_fwd(
        _mm(dq_f, cm.W_le0_q_T) + _mm(dv_f, cm.W_le0_v_T), cm.b_le0, cm.le_ln0_w, cm.le_ln0_b,
        cm.W_le1_T, cm.b_le1, cm.le_ln1_w, cm.le_ln1_b,
        cm.W_le2_T, cm.b_le2, cm.le_ln2_w, cm.le_ln2_b, compute_force, False, tl_dt,
    )
    local_q_e = _mv(le_x2, cm.W_le3) + cm.b_le3
    e_le = cp.empty(G, dtype=cp.float64)
    gpu.launch(_graph_sum_f64_1_kernel, (G,), local_q_e, plan.ptr_i32, e_le, G)

    # ---- Stage 9: Coulomb Energy (+ dE/dq + dE/dpos fused in ONE tiled-SRAM pass!)
    q_tot = cp.ascontiguousarray(q_cur[:, 0] + q_cur[:, 1])
    e_es_atom = cp.empty(N, dtype=cp.float64)
    if compute_force:
        g_q_es = cp.empty(N, dtype=dt)
        g_pos_es = cp.empty((N, 3), dtype=dt)
    else:
        g_q_es = g_pos_es = q_tot
    gpu.launch(
        _es_energy_fwd_bwd_kernel, es_grid,
        pos, q_tot, plan.ptr_i32, cm.scalars,
        e_es_atom, g_q_es, g_pos_es,
        COMPUTE_GRAD=compute_force, BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, DTYPE=tl_dt,
    )
    e_es = cp.empty(G, dtype=cp.float64)
    gpu.launch(_graph_sum_f64_1_kernel, (G,), e_es_atom, plan.ptr_i32, e_es, G)

    total_energy = e_local + e_le + e_es
    if not compute_force:
        return total_energy, None

    # =========================================================================
    # ANALYTICAL BACKWARD PASS (ZERO PyTorch autograd, ZERO atomics!)
    # =========================================================================

    g_q_cur = cp.ascontiguousarray(cp.stack([g_q_es, g_q_es], axis=-1))    # (N, 2)

    # ---- Bwd Stage 8: local_electron_energy
    g_le_h0 = _mlp3_ln_silu_bwd(
        cp.broadcast_to(cm.W_le3[None, :], (N, 128)).copy(), le_cache,
        cm.W_le1_T, cm.le_ln1_w, cm.le_ln1_b,
        cm.W_le2_T, cm.le_ln2_w, cm.le_ln2_b, cm.le_ln0_w, cm.le_ln0_b, tl_dt,
    )

    g_dq_f = _mm_tb(g_le_h0, cm.W_le0_q_T)
    g_dv_f = _mm_tb(g_le_h0, cm.W_le0_v_T)

    g_node_feats = (cm.W_ro * cm.scale)[None, :] + g_dq_f * uq + g_dv_f * uv
    g_q_sum_in = _mm_tb(g_dq_f * node_feats, cm.W_le_q_T)                   # (N, 2)
    g_q_cur = g_q_cur + g_q_sum_in
    g_q_ab0_extra = g_q_sum_in
    g_pot_from_le = _mm_tb(g_dv_f * node_feats, cm.W_le_v_T)                # (N, 4)

    # ---- Bwd Stage 7: SCF recursion steps in reverse
    g_features_mixed = cp.zeros_like(features_mixed)
    for r in range(len(cm.fdcs) - 1, -1, -1):
        fdc = cm.fdcs[r]
        q_in_r, wx2, nl_cache, tp_m, r_cache, fk_r, fk_sum_r, delta_r = scf_cache[r]

        g_q_pre = cp.empty_like(g_q_cur)
        g_fk_raw_r = cp.empty_like(g_q_cur)
        gpu.launch(
            _charge_cons_bwd_kernel, (G,),
            g_q_cur, fk_r, fk_sum_r, delta_r, plan.ptr_i32,
            g_q_pre, g_fk_raw_r, G, DTYPE=tl_dt,
        )
        g_new_f = _ro_mlp_bwd(cp.concatenate([g_q_pre, g_fk_raw_r], axis=-1), r_cache, fdc["W_r1_T"], fdc["W_rm_T"], fdc["W_r2_T"], fdc["cst_r"])
        g_features_mixed = g_features_mixed + g_new_f * tp_m
        g_nonlin = _mm_tb(g_new_f * features_mixed, fdc["W_tpout_T"])
        g_nh0 = _mlp3_ln_silu_bwd(
            _mm_tb(g_nonlin, fdc["W_nl3_T"]), nl_cache,
            fdc["W_nl1_T"], fdc["ln1_w"], fdc["ln1_b"],
            fdc["W_nl2_T"], fdc["ln2_w"], fdc["ln2_b"], fdc["ln0_w"], fdc["ln0_b"], tl_dt,
        )

        g_inv_f = _mm_tb(g_nh0, fdc["W_nl0_inv_T"])
        g_features_mixed = g_features_mixed + g_inv_f * wx2
        g_wx2 = g_inv_f * features_mixed

        g_features_mixed = g_features_mixed + _mm_tb(g_wx2, fdc["W_dot_nf_T"])
        g_pot_r = _mm_tb(g_wx2, fdc["W_dot_pot_T"])
        if r == len(cm.fdcs) - 1:
            g_pot_r = g_pot_r + g_pot_from_le
        g_q_prev = cp.ascontiguousarray(g_q_pre + _mm_tb(g_wx2, fdc["W_dot_chg_T"]))

        gpu.launch(
            _es_features_bwd_kernel, es_grid,
            pos, q_in_r, cp.ascontiguousarray(g_pot_r), plan.ptr_i32, cm.scalars,
            g_q_prev, g_pos_es,
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, DTYPE=tl_dt,
        )
        g_q_cur = g_q_prev

    # ---- Bwd Stage 6: initial charge conservation + fukui + lr_src + mixer + prod.linear
    g_q_ab0 = g_q_cur + g_q_ab0_extra
    g_q_raw = cp.empty_like(g_q_ab0)
    g_fk_raw0 = cp.empty_like(g_q_ab0)
    gpu.launch(
        _charge_cons_bwd_kernel, (G,),
        g_q_ab0, fk0, fk_sum0, delta0, plan.ptr_i32,
        g_q_raw, g_fk_raw0, G, DTYPE=tl_dt,
    )
    g_node_feats = (
        g_node_feats
        + _mm(g_features_mixed, cm.W_mix)
        + _mm(g_q_raw, cm.W_lr)
        + _ro_mlp_bwd(g_fk_raw0, fk_cache, cm.W_fk1_T, cm.W_fkm_T, cm.W_fk2_T, cm.cst_fk_silu)
    )
    g_sc_out = _mm(g_node_feats, cm.W_prod)

    # ---- Bwd Stage 5: SymmetricContraction + linear_2
    g_X16 = cp.empty_like(X16)
    gpu.launch(_sym_contr_bwd_kernel, (N, N_CB), X16, g_sc_out, cm.C1, cm.C2, cm.C3, g_X16, N, C=512, BLOCK_C=BLOCK_C)
    g_gated = _lin_l03(g_X16, cm.W_L2)

    # ---- Bwd Stage 4: DivDensity + AddRes + Gate
    g_out_l1_0 = cp.empty_like(out_l1_0)
    g_out_l1_vec = cp.empty_like(out_l1_vec)
    g_inv_part = cp.empty((N, N_CB), dtype=dt)
    gpu.launch(
        _div_res_gate_bwd_kernel, (N, N_CB),
        g_gated, out_l1_0, out_l1_vec, plan.node_res, inv_den, cm.scalars,
        g_out_l1_0, g_out_l1_vec, g_inv_part,
        N, N_CB=N_CB, BLOCK_C=BLOCK_C, DTYPE=tl_dt,
    )
    g_density = g_inv_part.sum(axis=1) * (-cm.beta * inv_den * inv_den)

    # ---- Bwd Stage 3: M_fused GEMMs -> g_A_all -> _edge_to_A_bwd
    g_A_all = cp.empty_like(A_all)
    _mm_tb(g_out_l1_0, plan.M0, out=g_A_all[0].reshape(N, S * 65))
    _mm_tb(g_out_l1_vec[0:3].reshape(3 * N, 512), plan.M1, out=g_A_all[1:4].reshape(3 * N, S * 65))
    _mm_tb(g_out_l1_vec[3:8].reshape(5 * N, 512), plan.M2, out=g_A_all[4:9].reshape(5 * N, S * 65))
    _mm_tb(g_out_l1_vec[8:15].reshape(7 * N, 512), plan.M3, out=g_A_all[9:16].reshape(7 * N, S * 65))

    if E > 0:
        g_act_e = cp.empty_like(act_e)
        g_Y_e = cp.empty_like(Y_e)
        g_cut_e = cp.empty_like(cut_e)
        gpu.launch(
            _edge_to_A_bwd_kernel, (N, S),
            plan.seg_ptr, cut_e, Y_e, act_e, g_A_all,
            g_act_e, g_Y_e, g_cut_e, N, S, DTYPE=tl_dt,
        )

        # ---- Bwd Stage 2: Edge RadialMLP + Edge Density MLP
        g_bes_e = _mm_tb(
            _mlp3_ln_silu_bwd(g_act_e, tp_cache, cm.W_tp1_T, cm.tp_ln1_w, cm.tp_ln1_b, cm.W_tp2_T, cm.tp_ln2_w, cm.tp_ln2_b, cm.tp_ln0_w, cm.tp_ln0_b, tl_dt),
            cm.W_tp0_r_T,
        )

        g_h_den0 = cp.empty((E, 64), dtype=dt)
        gpu.launch(
            _edge_density_bwd_kernel, (gpu.cdiv(E, 16),),
            g_density, plan.dst_i32, cut_e, den_xn, den_rs, den_d1,
            cm.den_ln0_w, cm.den_ln0_b, cm.W_den1,
            g_h_den0, g_cut_e, E, BLOCK_E=16, DTYPE=tl_dt,
        )
        g_bes_e = g_bes_e + _mm_tb(g_h_den0, cm.W_den0_r_T)

        # ---- Bwd Stage 1: Edge Geom + Atomic-free force scatter
        g_vec_e = cp.empty((E, 3), dtype=dt)
        gpu.launch(
            _edge_geom_bwd_kernel, (gpu.cdiv(E, 128),),
            pos, plan.src_i32, plan.dst_i32, cm.bessel_w, cm.scalars,
            d_e, g_cut_e, g_bes_e, g_Y_e, g_vec_e,
            E, BLOCK_E=128, DTYPE=tl_dt,
        )
        forces = cp.empty((N, 3), dtype=dt)
        gpu.launch(
            _scatter_edge_forces_kernel, (N,),
            g_vec_e, plan.rev_edge, plan.node_ptr, g_pos_es, forces, N, DTYPE=tl_dt,
        )
    else:
        forces = -g_pos_es

    return total_energy, forces


# --------------------------------------------------------------------------- GraphEvaluator

class TritonGraphEvaluator:
    """CUDA-graph-capable MACE evaluator backed by pure Triton + CuPy (zero PyTorch)."""

    def __init__(
        self,
        cm: CompiledPolarMACE,
        plan: TritonBatchPlan,
        *,
        graph: bool = True,
        r_cut: Optional[float] = None,
        out_pairs: Optional[Tuple[cp.ndarray, cp.ndarray]] = None,
    ):
        self.cm = cm
        self.mdt = cm.dt
        self.plan = plan
        self.out = out_pairs
        self.r_cut = r_cut
        self.n_graphs = plan.G
        self.n_nodes = plan.N
        self.want_graph = bool(graph)
        self.g = None
        self._stream = None
        self.mode = "triton"
        self._pos_buf: Optional[cp.ndarray] = None
        self._e_buf: Optional[cp.ndarray] = None
        self._f_buf: Optional[cp.ndarray] = None

    def held_bytes(self) -> int:
        """GPU memory reserved by the captured graph's private pool (0 when eager)."""
        return int(self.g.pool.total_bytes()) if self.g is not None else 0

    def stale(self, X: cp.ndarray) -> bool:
        if self.out is None or self.out[0].size == 0:
            return False
        oi, oj = self.out
        return bool(((X[oi] - X[oj]) ** 2).sum(-1).min() < self.r_cut * self.r_cut)

    def _capture(self, X: cp.ndarray):
        s = gpu.new_stream()
        caller = cp.cuda.get_current_stream()
        ev = caller.record()
        s.wait_event(ev)
        try:
            with s:
                self._pos_buf = gpu.as_gpu(X, dtype=self.mdt).copy()
                gpu.set_blas_stream(s)
                forward_and_forces(self.plan, self._pos_buf, compute_force=True)
            def _run():
                self._e_buf, self._f_buf = forward_and_forces(self.plan, self._pos_buf, compute_force=True)
            self.g = gpu.capture(s, _run)
            self._stream = s
            ev_done = s.record()
            caller.wait_event(ev_done)
            self.mode = "graph"
        finally:
            gpu.set_blas_stream(caller)

    def __call__(self, X: cp.ndarray):
        if self.want_graph and self.g is None:
            try:
                self._capture(X)
            except Exception as exc:
                if isinstance(exc, gpu.OOMError) or "out of memory" in str(exc).lower():
                    raise
                self.want_graph, self.g, self.mode = False, None, f"triton-eager ({type(exc).__name__}: {exc})"[:200]
                _log.warning("MACE graph capture failed; this evaluator runs eagerly (%s)", self.mode)
                gpu.sync()
        if self.g is not None:
            caller = cp.cuda.get_current_stream()
            self._pos_buf[...] = X.astype(self.mdt, copy=False)
            ev_in = caller.record()
            self._stream.wait_event(ev_in)
            self.g.launch(self._stream)
            ev_out = self._stream.record()
            caller.wait_event(ev_out)
            e, f = self._e_buf, self._f_buf
        else:
            e, f = forward_and_forces(self.plan, gpu.as_gpu(X, dtype=self.mdt), compute_force=True)
        e64 = e.astype(cp.float64, copy=False)
        return e64, e64, f.astype(cp.float64, copy=False)


def eval_structures_ef(
    cm: CompiledPolarMACE,
    structs: Sequence[Tuple[Sequence[str], np.ndarray, int, int]],
    *,
    compute_force: bool = False,
) -> Tuple[cp.ndarray, Optional[cp.ndarray]]:
    """Evaluate energies (G,) and optional forces (N, 3) directly from raw structures (zero torch/ASE/PyG)."""
    plan, _ = TritonBatchPlan.from_structures(cm, structs)
    pos_list = [np.asarray(s[1], dtype=np.float64) for s in structs]
    pos_np = np.concatenate(pos_list, axis=0) if pos_list else np.empty((0, 3), dtype=np.float64)
    pos = gpu.as_gpu(pos_np, dtype=cm.dt)
    return forward_and_forces(plan, pos, compute_force=compute_force)
