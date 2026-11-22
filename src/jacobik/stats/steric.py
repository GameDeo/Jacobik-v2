"""
src/jacobik/stats/q6.py
=======================
Purely repulsive Q6 steric model runtime lookup and evaluation.

Every non-bonded atom pair (> 3 bonds apart) is scored purely repulsively:
- Class-5 non-polar pairs: physical exchange repulsion (shifted-force Buckingham exponential
  below Alvarez vdW sum r_m, zero at and beyond).
- Class-4, metal, and polar-H...acceptor pairs: analytical short-range onset core below
  empirical 0.1% onset r0.
"""
from __future__ import annotations

import functools
import math
from typing import Dict, List, Optional, Tuple

import cupy as cp
import numpy as np

from jacobik.core import gpu, profile as _profile
from jacobik.core.env import RT_KJ

R_LO, R_HI, DR_TAB = 0.6, 6.0, 0.02
W_LINEAR_ABOVE = 50.0
Q6_PHYSICAL_ALL = bool(_profile.get("q6_physical_all"))
Q6_WEIGHT_14 = float(_profile.get("q6_weight_14"))

REP_PARAMS: Dict[str, Tuple[float, float]] = {
    "H": (8304.2, 3.867), "B": (6217.9, 2.5), "C": (52908.4, 3.1943), "N": (2558.4, 2.5), "O": (1794.0, 2.5),
    "F": (4288.8, 2.5493), "Si": (107369.6, 2.5), "P": (30445.7, 2.5), "S": (30862.1, 2.5), "Cl": (69172.6, 2.8674),
    "Se": (102333.9, 2.5882), "Br": (451089.0, 3.0641), "I": (2844105.5, 3.2715),
}
ALVAREZ_RADII: Dict[str, float] = {
    "H": 1.20, "B": 1.91, "C": 1.77, "N": 1.66, "O": 1.50, "F": 1.46, "Si": 2.19,
    "P": 1.90, "S": 1.89, "Cl": 1.82, "Se": 1.82, "Br": 1.86, "I": 2.04,
}


@functools.lru_cache(maxsize=None)
def rep_pair_params(e1: str, e2: str, scale: float = 1.0) -> Optional[Tuple[float, float, float, float]]:
    """(A kJ/mol, b 1/A, r_m A, r_lin A) of the physical repulsion of an element pair, or None without parameters.
    ``scale`` moves the onset to scale * r_m (A and b unchanged); r_lin is recomputed for the moved onset.
    Memoized: a pure function of its arguments, called once per atom pair by build_runtime (230k calls, 16 s of
    FUTSOD's 55 s profiled run before 2026-11-18)."""
    e1, e2 = ("H" if e1 == "Hp" else e1), ("H" if e2 == "Hp" else e2)
    if e1 not in REP_PARAMS or e2 not in REP_PARAMS or e1 not in ALVAREZ_RADII or e2 not in ALVAREZ_RADII:
        return None
    A = math.sqrt(REP_PARAMS[e1][0] * REP_PARAMS[e2][0])
    b = 0.5 * (REP_PARAMS[e1][1] + REP_PARAMS[e2][1])
    rm = scale * (ALVAREZ_RADII[e1] + ALVAREZ_RADII[e2])
    u = lambda r: A * (math.exp(-b * r) - math.exp(-b * rm) + b * math.exp(-b * rm) * (r - rm)) / RT_KJ
    lo, hi = 0.0, rm
    if u(lo) <= W_LINEAR_ABOVE:
        return A, b, rm, 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if u(mid) > W_LINEAR_ABOVE else (lo, mid)
    return A, b, rm, hi


def rep_energy(d, A, b, rm, rl):
    """Physical repulsion (kT) at distances d."""
    d_arr = cp.asarray(d)
    A, b, rm, rl = (cp.asarray(v, dtype=d_arr.dtype) for v in (A, b, rm, rl))
    em = cp.exp(-b * rm)
    u = lambda r: A * (cp.exp(-b * r) - em + b * em * (r - rm)) / RT_KJ
    slope = A * b * (cp.exp(-b * rl) - em) / RT_KJ
    lin = u(rl) + slope * (rl - d_arr)
    out = cp.where(d_arr < rl, lin, u(cp.maximum(d_arr, rl)))
    return cp.where(d_arr < rm, out, cp.zeros_like(out))


def ligand_separation(mol, max_depth: int = 4) -> Dict[int, Dict[int, int]]:
    """{atom idx: {atom idx: bonds apart along LIGAND bonds (metal excluded)}} up to max_depth."""
    from jacobik.topology.graph import get_ligand_neighbors

    nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    out: Dict[int, Dict[int, int]] = {}
    for a in mol.atoms:
        if a.idx == mol.metal_idx:
            continue
        seen = {a.idx: 0}
        frontier = [a.idx]
        for d in range(1, max_depth + 1):
            nxt = []
            for u in frontier:
                for v in nb.get(u, ()):
                    if v not in seen:
                        seen[v] = d
                        nxt.append(v)
            frontier = nxt
        out[a.idx] = seen
    return out


def pair_class(i: int, j: int, metal_idx: int, sep: Dict[int, Dict[int, int]]) -> str:
    if i == metal_idx or j == metal_idx:
        return "m"
    return "4" if sep.get(i, {}).get(j) == 4 else "5"


def build_runtime(mol, ge, tables: Dict[tuple, dict], wall_mu, wall_sigma, device="cuda"):
    """Tensors for the purely repulsive Q6 scorer."""
    from jacobik.stats import keys as kb
    from jacobik.topology.mapping import raw_metal_bond_exclusion_pairs

    if not tables:
        return None
    atoms = list(mol.atoms)
    sep = ligand_separation(mol)
    rows: List[List[float]] = []
    row_r0: List[float] = []
    row_of: Dict[tuple, int] = {}

    def row(k):
        e = tables.get(k)
        if e is None:
            return -1
        if k not in row_of:
            row_of[k] = len(rows)
            rows.append(e["w"])
            row_r0.append(float(e.get("r0", R_LO)))
        return row_of[k]

    raw_m = raw_metal_bond_exclusion_pairs(mol.atoms, mol.bonds, mol.metal_idx, q6=True)
    a_idx = ge._q6_a.tolist()
    b_idx = ge._q6_b.tolist()
    t_tot = []
    rep = []
    enabled = []
    rep_weight = []
    donor_scale = getattr(ge, "q6_donor_scale", None)
    for p, (ia, ib) in enumerate(zip(a_idx, b_idx)):
        key = ge._q6_elem_keys[p]
        lo, hi = sorted((atoms[ia].idx, atoms[ib].idx))
        if (lo, hi) in raw_m:
            t_tot.append(-1)
            rep.append(None)
            # Preserve the pre-existing fallback-wall treatment for raw metal
            # exclusions; only the explicit 1-4 zero-weight case is omitted.
            enabled.append(True)
            rep_weight.append(1.0)
            continue
        ia_idx, ib_idx = atoms[ia].idx, atoms[ib].idx
        cls = pair_class(ia_idx, ib_idx, mol.metal_idx, sep)
        sep14 = ia_idx != mol.metal_idx and ib_idx != mol.metal_idx and sep.get(ia_idx, {}).get(ib_idx) == 3
        if Q6_PHYSICAL_ALL and sep14 and Q6_WEIGHT_14 == 0.0:
            # A zero energy multiplier is the documented way to omit a 1-4 pair;
            # do not route it through either the fitted wall or the fallback wall.
            t_tot.append(-1)
            rep.append(None)
            enabled.append(False)
            rep_weight.append(0.0)
            continue
        polar_acc = "Hp" in key and (key[0] in kb.Q6_ACCEPTORS or key[1] in kb.Q6_ACCEPTORS)
        if donor_scale is not None and donor_scale[p] is not None:
            # donor-pair extension (flag q6_donor_pairs): class-5 physical repulsion with the scaled onset
            rep.append(rep_pair_params(key[0], key[1], donor_scale[p]))
            rep_weight.append(1.0)
        else:
            phys = Q6_PHYSICAL_ALL or cls == "5"
            rep.append(rep_pair_params(key[0], key[1]) if phys and (Q6_PHYSICAL_ALL or not polar_acc) else None)
            rep_weight.append(Q6_WEIGHT_14 if Q6_PHYSICAL_ALL and sep14 else 1.0)
        enabled.append(True)
        tt = row((key[0], key[1], cls, ""))
        if tt < 0:
            tt = row((key[0], key[1], "5", ""))
        t_tot.append(tt)

    if not rows:
        return None
    T = cp.asarray(np.array(rows, dtype=np.float32), dtype=cp.float32)
    lt = lambda v: cp.asarray(v, dtype=cp.int64)
    has = cp.asarray([x >= 0 for x in t_tot], dtype=cp.bool_)
    rp = np.array([r if r is not None else (0.0, 1.0, 0.0, 0.0) for r in rep], dtype=np.float64)
    ft = lambda v: cp.asarray(v, dtype=cp.float32)
    return dict(
        T=T, tot=lt([max(x, 0) for x in t_tot]), has=has, r_lo=R_LO, dr=DR_TAB, G=T.shape[1],
        enabled=cp.asarray(enabled, dtype=cp.bool_), rep_weight=ft(rep_weight),
        mu=cp.asarray(wall_mu, dtype=cp.float32), sigma=cp.asarray(wall_sigma, dtype=cp.float32),
        n_tables=len(rows), n_pairs_with_table=int(has.sum()),
        r0=cp.asarray(row_r0, dtype=cp.float32),
        rep=cp.asarray([r is not None for r in rep], dtype=cp.bool_),
        rep_any=any(r is not None for r in rep), rep_A=ft(rp[:, 0]), rep_b=ft(rp[:, 1]), rep_rm=ft(rp[:, 2]),
        rep_rl=ft(rp[:, 3]),
    )


def _interp(T, rows, d, r_lo, dr, G):
    """Piecewise-linear w at distances d (B, M) for table rows (M,): linear extrapolation below r_lo, 0 above."""
    u = (d - r_lo) / dr
    i0 = cp.clip(cp.floor(u), 0, G - 2).astype(cp.int64)
    f = u - i0.astype(u.dtype)
    w0 = T[rows[None, :], i0]
    w1 = T[rows[None, :], i0 + 1]
    w = w0 + f * (w1 - w0)
    return cp.where(u >= G - 1, cp.zeros_like(w), w)


def _q6_triton_module():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None, None

    @triton.jit
    def _q6_pmf_2d_kernel(
        X_ptr,           # (B, N, 3) float32
        grad_X_ptr,      # (B, N, 3) float32
        val_ptr,         # (B,) float32
        a_idx_ptr,       # (M,) int32
        b_idx_ptr,       # (M,) int32
        kind_ptr,        # (M,) int8: 1=rep, 2=table, 3=wall
        p0_ptr,          # (M,) float32
        p1_ptr,          # (M,) float32
        p2_ptr,          # (M,) float32
        p3_ptr,          # (M,) float32
        p4_ptr,          # (M,) float32
        p5_ptr,          # (M,) float32
        p6_ptr,          # (M,) float32
        T_ptr,           # (n_tables, G) float32
        B: int,
        N: int,
        M: int,
        G: int,
        r_lo: float,
        inv_dr: float,
        scale: float,
        BLOCK_B: tl.constexpr,
        PAIRS_PER_PROG: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)
        b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b_offs < B

        acc_val = tl.zeros((BLOCK_B,), dtype=tl.float32)
        p_start = pid_p * PAIRS_PER_PROG

        for dp in range(PAIRS_PER_PROG):
            p = p_start + dp
            if p < M:
                ia = tl.load(a_idx_ptr + p)
                ib = tl.load(b_idx_ptr + p)
                kind = tl.load(kind_ptr + p)

                base_a = b_offs * (N * 3) + ia * 3
                base_b = b_offs * (N * 3) + ib * 3
                ax = tl.load(X_ptr + base_a + 0, mask=b_mask, other=0.0)
                ay = tl.load(X_ptr + base_a + 1, mask=b_mask, other=0.0)
                az = tl.load(X_ptr + base_a + 2, mask=b_mask, other=0.0)
                bx = tl.load(X_ptr + base_b + 0, mask=b_mask, other=0.0)
                by = tl.load(X_ptr + base_b + 1, mask=b_mask, other=0.0)
                bz = tl.load(X_ptr + base_b + 2, mask=b_mask, other=0.0)

                dx = ax - bx
                dy = ay - by
                dz = az - bz
                d2 = dx * dx + dy * dy + dz * dz
                d = tl.sqrt(d2)

                w_out = tl.zeros((BLOCK_B,), dtype=tl.float32)
                g_out = tl.zeros((BLOCK_B,), dtype=tl.float32)

                if kind == 1:
                    rm = tl.load(p2_ptr + p)
                    inside = b_mask & (d < rm)
                    if tl.sum(inside.to(tl.int32), axis=0) > 0:
                        A_rw = tl.load(p0_ptr + p)
                        b_par = tl.load(p1_ptr + p)
                        rl = tl.load(p3_ptr + p)
                        em = tl.load(p4_ptr + p)
                        slope = tl.load(p5_ptr + p)
                        u_rl = tl.load(p6_ptr + p)

                        dm = tl.maximum(d, rl)
                        ed = tl.exp(-b_par * dm)
                        lo = d < rl
                        u_rep = tl.where(lo, u_rl + slope * (rl - d), A_rw * (ed - em + b_par * em * (dm - rm)))
                        g_rep = tl.where(lo, -slope, A_rw * b_par * (em - ed))
                        w_out = tl.where(inside, u_rep, 0.0)
                        g_out = tl.where(inside, g_rep, 0.0)
                elif kind == 2:
                    r0 = tl.load(p0_ptr + p)
                    inside = b_mask & (d < r0)
                    if tl.sum(inside.to(tl.int32), axis=0) > 0:
                        w0 = tl.load(p1_ptr + p)
                        row_idx = tl.load(p2_ptr + p).to(tl.int32)
                        u = (d - r_lo) * inv_dr
                        out_grid = u >= (G - 1)
                        i0 = tl.minimum(tl.maximum(tl.floor(u).to(tl.int32), 0), G - 2)
                        wa = tl.load(T_ptr + row_idx * G + i0, mask=inside, other=0.0)
                        wb = tl.load(T_ptr + row_idx * G + i0 + 1, mask=inside, other=0.0)
                        s = wb - wa
                        w = wa + (u - i0.to(tl.float32)) * s
                        w = tl.where(out_grid, 0.0, w)
                        s_dr = tl.where(out_grid, 0.0, s * inv_dr)
                        wc = w - w0
                        core = inside & (wc >= 0.0)
                        w_out = tl.where(core, wc, 0.0)
                        g_out = tl.where(core, s_dr, 0.0)
                elif kind == 3:
                    mu = tl.load(p0_ptr + p)
                    inside = b_mask & (d < mu)
                    if tl.sum(inside.to(tl.int32), axis=0) > 0:
                        inv_sig2 = tl.load(p1_ptr + p)
                        t = mu - d
                        w_out = tl.where(inside, 0.5 * t * t * inv_sig2, 0.0)
                        g_out = tl.where(inside, -t * inv_sig2, 0.0)

                acc_val += w_out * scale
                has_force = b_mask & (g_out != 0.0) & (d > 0.0)
                if tl.sum(has_force.to(tl.int32), axis=0) > 0:
                    coeff = tl.where(has_force, (g_out * scale) / tl.maximum(d, 1e-30), 0.0)
                    fx = coeff * dx
                    fy = coeff * dy
                    fz = coeff * dz
                    tl.atomic_add(grad_X_ptr + base_a + 0, fx, mask=has_force)
                    tl.atomic_add(grad_X_ptr + base_a + 1, fy, mask=has_force)
                    tl.atomic_add(grad_X_ptr + base_a + 2, fz, mask=has_force)
                    tl.atomic_add(grad_X_ptr + base_b + 0, -fx, mask=has_force)
                    tl.atomic_add(grad_X_ptr + base_b + 1, -fy, mask=has_force)
                    tl.atomic_add(grad_X_ptr + base_b + 2, -fz, mask=has_force)

        tl.atomic_add(val_ptr + b_offs, acc_val, mask=b_mask)

    return triton, _q6_pmf_2d_kernel


_TRITON, _Q6_PMF_KERNEL = _q6_triton_module()


def _ensure_packed(rt, a_idx, b_idx, device="cuda"):
    """Pack active Q6 pair parameters once per (rt, a_idx, b_idx, enabled, rep_weight)."""
    en = rt.get("enabled")
    rw = rt.get("rep_weight", 1.0)
    key = (str(device), id(a_idx), id(b_idx), id(en), id(rw))
    cached = rt.get("_packed")
    if cached is not None and cached[0] == key:
        return cached[1]

    M_all = a_idx.shape[0]
    if en is None:
        en_t = cp.ones(M_all, dtype=cp.bool_)
    else:
        en_t = cp.asarray(en, dtype=cp.bool_)
    rep = cp.asarray(rt["rep"], dtype=cp.bool_)
    has = cp.asarray(rt["has"], dtype=cp.bool_)
    if isinstance(rw, cp.ndarray) or np.ndim(rw) > 0:
        rw_t = cp.asarray(rw, dtype=cp.float32)
    else:
        rw_t = cp.full((M_all,), float(rw), dtype=cp.float32)

    T = cp.ascontiguousarray(cp.asarray(rt["T"], dtype=cp.float32))
    rows = cp.asarray(rt["tot"], dtype=cp.int64)
    r_lo = float(rt["r_lo"])
    dr = float(rt["dr"])
    G = int(rt["G"])
    r0_all = cp.asarray(rt["r0"], dtype=cp.float32)
    w0_all = _interp(T, rows, r0_all[rows][None], r_lo, dr, G)[0]

    kind = cp.zeros(M_all, dtype=cp.int8)
    kind = cp.where(en_t & ~rep & ~has, cp.int8(3), kind)
    kind = cp.where(en_t & ~rep & has, cp.int8(2), kind)
    if rt.get("rep_any"):
        kind = cp.where(en_t & rep & (rw_t != 0.0), cp.int8(1), kind)

    active = kind > 0
    a_act = cp.ascontiguousarray(cp.asarray(a_idx, dtype=cp.int32)[active])
    b_act = cp.ascontiguousarray(cp.asarray(b_idx, dtype=cp.int32)[active])
    a_act_l = a_act.astype(cp.int64)
    b_act_l = b_act.astype(cp.int64)
    k_act = cp.ascontiguousarray(kind[active])
    M_act = int(a_act.shape[0])

    p0 = cp.zeros(M_act, dtype=cp.float32)
    p1 = cp.zeros(M_act, dtype=cp.float32)
    p2 = cp.zeros(M_act, dtype=cp.float32)
    p3 = cp.zeros(M_act, dtype=cp.float32)
    p4 = cp.zeros(M_act, dtype=cp.float32)
    p5 = cp.zeros(M_act, dtype=cp.float32)
    p6 = cp.zeros(M_act, dtype=cp.float32)

    if M_act > 0:
        m1 = k_act == 1
        if bool(m1.any()):
            A_1 = cp.asarray(rt["rep_A"], dtype=cp.float32)[active][m1] * rw_t[active][m1] / RT_KJ
            b_1 = cp.asarray(rt["rep_b"], dtype=cp.float32)[active][m1]
            rm_1 = cp.asarray(rt["rep_rm"], dtype=cp.float32)[active][m1]
            rl_1 = cp.asarray(rt["rep_rl"], dtype=cp.float32)[active][m1]
            em_1 = cp.exp(-b_1 * rm_1)
            erl_1 = cp.exp(-b_1 * rl_1)
            slope_1 = A_1 * b_1 * (erl_1 - em_1)
            u_rl_1 = A_1 * (erl_1 - em_1 + b_1 * em_1 * (rl_1 - rm_1))
            p0[m1], p1[m1], p2[m1], p3[m1], p4[m1], p5[m1], p6[m1] = A_1, b_1, rm_1, rl_1, em_1, slope_1, u_rl_1

        m2 = k_act == 2
        if bool(m2.any()):
            rows_2 = rows[active][m2]
            p0[m2] = r0_all[rows_2]
            p1[m2] = w0_all[active][m2]
            p2[m2] = rows_2.astype(cp.float32)

        m3 = k_act == 3
        if bool(m3.any()):
            p0[m3] = cp.asarray(rt["mu"], dtype=cp.float32)[active][m3]
            p1[m3] = 1.0 / (cp.maximum(cp.asarray(rt["sigma"], dtype=cp.float32)[active][m3], 1e-7) ** 2)

    r_cut = cp.where(k_act == 1, p2, p0)
    r_cut_sq = r_cut * r_cut
    packed = dict(
        M_act=M_act, a_act=a_act, b_act=b_act, a_act_l=a_act_l, b_act_l=b_act_l,
        k_act=k_act, p0=p0, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6,
        T=T, r_lo=r_lo, inv_dr=1.0 / dr, dr=dr, G=G, r_cut_sq=r_cut_sq,
    )
    rt["_packed"] = (key, packed)
    return packed


def evaluate_and_grad(rt, X, a_idx, b_idx, scale: float = 1.0, out_val=None, out_grad_X=None):
    """Evaluate Q6 sum (scaled by ``scale``) and analytical coordinate gradients in one pass."""
    if not isinstance(X, cp.ndarray):
        raise RuntimeError("Q6 steric evaluation requires a CuPy GPU array (GPU-only Triton build)")
    if _Q6_PMF_KERNEL is None:
        raise RuntimeError("Triton is required for CUDA Q6 evaluation")

    B, N, _ = X.shape
    if out_val is None:
        out_val = cp.zeros(B, dtype=X.dtype)
    if out_grad_X is None:
        out_grad_X = cp.zeros_like(X)

    pk = _ensure_packed(rt, a_idx, b_idx)
    M_act = pk["M_act"]
    if M_act == 0 or scale == 0.0:
        return out_val, out_grad_X

    Xc = cp.ascontiguousarray(X.astype(cp.float32, copy=False))
    gx_f32 = out_grad_X if (out_grad_X.dtype == cp.float32 and out_grad_X.flags.c_contiguous) else cp.zeros_like(Xc)
    val_f32 = out_val if (out_val.dtype == cp.float32 and out_val.flags.c_contiguous) else cp.zeros(B, dtype=cp.float32)
    BLOCK_B = 64
    PAIRS_PER_PROG = 64
    grid = (gpu.cdiv(B, BLOCK_B), gpu.cdiv(M_act, PAIRS_PER_PROG))
    gpu.launch(
        _Q6_PMF_KERNEL, grid,
        Xc, gx_f32, val_f32,
        pk["a_act"], pk["b_act"], pk["k_act"],
        pk["p0"], pk["p1"], pk["p2"], pk["p3"], pk["p4"], pk["p5"], pk["p6"],
        pk["T"], B, N, M_act, pk["G"], pk["r_lo"], pk["inv_dr"], float(scale),
        BLOCK_B=BLOCK_B, PAIRS_PER_PROG=PAIRS_PER_PROG,
    )
    if val_f32 is not out_val:
        out_val[...] = val_f32.astype(out_val.dtype)
    if gx_f32 is not out_grad_X:
        out_grad_X[...] = gx_f32.astype(out_grad_X.dtype)
    return out_val, out_grad_X


def evaluate(rt, X, a_idx, b_idx):
    """(B,) sum over Q6 pairs of purely repulsive penalty w >= 0 (kT)."""
    val, _ = evaluate_and_grad(rt, X, a_idx, b_idx)
    return val
