"""One forward-kinematics pass for all ligands of a complex (2026-10-03, GPU fast path).

    cfk = ComplexFK(fks)                  # fks: the solver's list of ForwardKinematics
    X = cfk.apply(thetas, X_ref)          # == for fk, th in zip(fks, thetas): X = fk.apply(th, X)

`ForwardKinematics` stays the reference implementation and the place where joints are defined; this class reads
its tables (`_quad_idx`, `_downstream`, `_bend_plan`) and never defines a joint itself.

What is composed (a fixed number of operations for the whole complex):
  * every ligand's base transform, as the outermost rigid map of its atoms' chains;
  * a leading block of each ligand's joints. Joint k is the rigid map x -> (x - p_k) RT_k + p_k about its
    reference bond axis with angle delta_k. Because the joints of the block only ever move a joint's dihedral
    atoms rigidly, the dihedral the sequential code would measure is known in advance:
        delta_k = theta_k - phi_ref_k - sum_m c_km delta_m
    where the sum runs over earlier joints m on the SAME bond axis that turn one side of k's dihedral
    (c_km = +-1; this is what makes branch-bend joints composable). phi_ref, axes and pivots come from X_ref and
    are cached while the same X_ref storage is passed, so a gradient step does not measure any dihedral.
  * all ligands' angle bends, by one merged `angle_bends.BendPlan` (evaluated by BendPlan.apply itself).

What stays on the reference path: joints after the first one of a ligand that fails the rule below (they run
through `fk._apply_torsion` in their original order, between the composed block and the bends). A joint type
this class does not know therefore stays correct. If a complex has a feature this class does not handle
(`supported` is False, or X_ref requires grad), `apply` runs the reference loop for the whole complex.

Build-time rule for joint k of a ligand, against every earlier joint m of that ligand:
  - k's four dihedral atoms are ligand atoms;
  - m has the same axis atoms as k                      -> coupling c_km, fine;
  - otherwise m moves none of k's dihedral atoms, or every one of them is downstream of m or on m's axis;
  - the downstream sets of m and k are disjoint, nested or equal (ring joints with partly overlapping sets fail).
Chain order (innermost = applied first to the reference coordinates): for m < k sharing an atom, k is inside m
when m moves k's axis; otherwise m is inside k.

Agreement with the reference loop (157 fixture complexes, solver start angles + random re-draws): 95 % of joints
composed (all base torsions, all branch-bend joints, 72 of 156 macrocycle joints); coordinates agree to 3e-4 A
(median 1e-5 A), gradients to 3e-4 relative. Exception: joints whose dihedral is undefined (three collinear atoms,
e.g. the phenyl torsions of diphenylacetylene) are numerically unstable in the reference path itself.
Test: tests/test_gpu_fast_path.py.
"""
from __future__ import annotations

from functools import cmp_to_key
from typing import List

import triton
import triton.language as tl

from jacobik.core import gpu
from jacobik.core.gpu import cp
from jacobik.ik.angle_bends import (
    _build_csr_tables,
    _se3_chains_bwd_k,
    _se3_chains_fwd_k,
    _store12_tl,
)

_EPS = 1e-8


def _plan_ligand(quad, down, lig_atoms):
    """-> (K0, couplings {k: [(m, c)]}) for one ligand."""
    J = len(quad)
    coup = {}
    for k in range(J):
        qk = quad[k]
        sk = set(qk)
        if not sk <= lig_atoms:
            return k, coup
        ck = []
        for m in range(k):
            qm = quad[m]
            axis = {qm[1], qm[2]}
            Dm = down[m] - axis
            if axis == {qk[1], qk[2]}:
                sgn = 1.0 if (qm[1], qm[2]) == (qk[1], qk[2]) else -1.0
                c = sgn * (float(qk[3] in Dm) - float(qk[0] in Dm))
                if c:
                    ck.append((m, c))
                continue
            if (Dm & sk) and not sk <= (down[m] | axis):
                return k, coup
        for m in range(k):                                   # moved sets that overlap must be nested (or equal)
            inter = down[m] & down[k]
            if inter and not (down[k] <= down[m] or down[m] <= down[k]):
                return k, coup
        if ck:
            coup[k] = ck
    return J, coup


def _inner_first(m, k, quad, down):
    """Order of two joints m < k of one ligand inside a chain: -1 if m is applied first (innermost)."""
    axis_m = {quad[m][1], quad[m][2]}
    piv_k = {quad[k][1], quad[k][2]}
    k_moved = piv_k <= (down[m] | axis_m) and bool((piv_k - axis_m) & down[m])
    return 1 if k_moved else -1


@triton.jit
def _cfk_transforms_fwd_k(
    th_all_ptr, delta_ptr,
    c_phi_ptr, c_ax_ptr, c_piv_ptr, c_pb_ptr,
    trans_col_ptr,
    At_ptr,
    B, R_ref, K, L, W_th,
    HAS_DELTA: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    s = tl.program_id(1)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = b < B
    r_idx = tl.where(R_ref > 1, b, 0)

    col = tl.load(trans_col_ptr + s)
    if s < K:
        if HAS_DELTA:
            d = tl.load(delta_ptr + b * K + s, mask=mask, other=0.0)
        else:
            th = tl.load(th_all_ptr + b * W_th + col, mask=mask, other=0.0)
            phi0 = tl.load(c_phi_ptr + r_idx * K + s, mask=mask, other=0.0)
            d = th - phi0

        cs = tl.cos(d)
        sn = tl.sin(d)
        omc = 1.0 - cs

        base_k = (r_idx * K + s) * 3
        ax = tl.load(c_ax_ptr + base_k + 0, mask=mask, other=0.0)
        ay = tl.load(c_ax_ptr + base_k + 1, mask=mask, other=0.0)
        az = tl.load(c_ax_ptr + base_k + 2, mask=mask, other=0.0)
        px = tl.load(c_piv_ptr + base_k + 0, mask=mask, other=0.0)
        py = tl.load(c_piv_ptr + base_k + 1, mask=mask, other=0.0)
        pz = tl.load(c_piv_ptr + base_k + 2, mask=mask, other=0.0)

        a00 = cs + omc * ax * ax
        a01 = omc * ax * ay + sn * az
        a02 = omc * ax * az - sn * ay
        a10 = omc * ay * ax - sn * az
        a11 = cs + omc * ay * ay
        a12 = omc * ay * az + sn * ax
        a20 = omc * az * ax + sn * ay
        a21 = omc * az * ay - sn * ax
        a22 = cs + omc * az * az

        tx = px - (px * a00 + py * a10 + pz * a20)
        ty = py - (px * a01 + py * a11 + pz * a21)
        tz = pz - (px * a02 + py * a12 + pz * a22)
    else:
        li = s - K
        base_th = b * W_th + col
        trx = tl.load(th_all_ptr + base_th + 0, mask=mask, other=0.0)
        try_ = tl.load(th_all_ptr + base_th + 1, mask=mask, other=0.0)
        trz = tl.load(th_all_ptr + base_th + 2, mask=mask, other=0.0)
        wx = tl.load(th_all_ptr + base_th + 3, mask=mask, other=0.0)
        wy = tl.load(th_all_ptr + base_th + 4, mask=mask, other=0.0)
        wz = tl.load(th_all_ptr + base_th + 5, mask=mask, other=0.0)

        ang_sq = tl.maximum(wx * wx + wy * wy + wz * wz, 1e-16)
        ang = tl.sqrt(ang_sq)
        safe = ang >= 1e-8
        ang_c = tl.maximum(ang, 1e-8)
        kx = tl.where(safe, wx / ang_c, wx)
        ky = tl.where(safe, wy / ang_c, wy)
        kz = tl.where(safe, wz / ang_c, wz)

        cs = tl.cos(ang)
        sn = tl.sin(ang)
        omc = 1.0 - cs

        a00 = tl.where(safe, cs + omc * kx * kx, 1.0)
        a01 = tl.where(safe, omc * kx * ky + sn * kz, wz)
        a02 = tl.where(safe, omc * kx * kz - sn * ky, -wy)
        a10 = tl.where(safe, omc * ky * kx - sn * kz, -wz)
        a11 = tl.where(safe, cs + omc * ky * ky, 1.0)
        a12 = tl.where(safe, omc * ky * kz + sn * kx, wx)
        a20 = tl.where(safe, omc * kz * kx + sn * ky, wy)
        a21 = tl.where(safe, omc * kz * ky - sn * kx, -wx)
        a22 = tl.where(safe, cs + omc * kz * kz, 1.0)

        base_pb = (r_idx * L + li) * 3
        px = tl.load(c_pb_ptr + base_pb + 0, mask=mask, other=0.0)
        py = tl.load(c_pb_ptr + base_pb + 1, mask=mask, other=0.0)
        pz = tl.load(c_pb_ptr + base_pb + 2, mask=mask, other=0.0)

        tx = px - (px * a00 + py * a10 + pz * a20) + trx
        ty = py - (px * a01 + py * a11 + pz * a21) + try_
        tz = pz - (px * a02 + py * a12 + pz * a22) + trz

    KL = K + L
    _store12_tl(At_ptr, (b * KL + s) * 12, mask, a00, a01, a02, a10, a11, a12, a20, a21, a22, tx, ty, tz)


@triton.jit
def _cfk_transforms_bwd_k(
    th_all_ptr, delta_ptr,
    g_th_all_ptr, g_delta_ptr,
    c_phi_ptr, c_ax_ptr, c_piv_ptr, c_pb_ptr,
    trans_col_ptr, trans_occ_ptr_ptr, trans_occ_cd_ptr,
    gAt_chain_ptr,
    B, R_ref, K, L, C, D, W_th,
    HAS_DELTA: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    s = tl.program_id(1)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = b < B
    r_idx = tl.where(R_ref > 1, b, 0)

    occ_start = tl.load(trans_occ_ptr_ptr + s)
    occ_end = tl.load(trans_occ_ptr_ptr + s + 1)

    ga00 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ga01 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ga02 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ga10 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ga11 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ga12 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ga20 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ga21 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    ga22 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gt0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gt1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    gt2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for o_idx in range(occ_start, occ_end):
        cd = tl.load(trans_occ_cd_ptr + o_idx)
        base_gat = (b * (C * D) + cd) * 12
        ga00 += tl.load(gAt_chain_ptr + base_gat + 0, mask=mask, other=0.0)
        ga01 += tl.load(gAt_chain_ptr + base_gat + 1, mask=mask, other=0.0)
        ga02 += tl.load(gAt_chain_ptr + base_gat + 2, mask=mask, other=0.0)
        ga10 += tl.load(gAt_chain_ptr + base_gat + 3, mask=mask, other=0.0)
        ga11 += tl.load(gAt_chain_ptr + base_gat + 4, mask=mask, other=0.0)
        ga12 += tl.load(gAt_chain_ptr + base_gat + 5, mask=mask, other=0.0)
        ga20 += tl.load(gAt_chain_ptr + base_gat + 6, mask=mask, other=0.0)
        ga21 += tl.load(gAt_chain_ptr + base_gat + 7, mask=mask, other=0.0)
        ga22 += tl.load(gAt_chain_ptr + base_gat + 8, mask=mask, other=0.0)
        gt0 += tl.load(gAt_chain_ptr + base_gat + 9, mask=mask, other=0.0)
        gt1 += tl.load(gAt_chain_ptr + base_gat + 10, mask=mask, other=0.0)
        gt2 += tl.load(gAt_chain_ptr + base_gat + 11, mask=mask, other=0.0)

    col = tl.load(trans_col_ptr + s)
    if s < K:
        base_k = (r_idx * K + s) * 3
        ax = tl.load(c_ax_ptr + base_k + 0, mask=mask, other=0.0)
        ay = tl.load(c_ax_ptr + base_k + 1, mask=mask, other=0.0)
        az = tl.load(c_ax_ptr + base_k + 2, mask=mask, other=0.0)
        px = tl.load(c_piv_ptr + base_k + 0, mask=mask, other=0.0)
        py = tl.load(c_piv_ptr + base_k + 1, mask=mask, other=0.0)
        pz = tl.load(c_piv_ptr + base_k + 2, mask=mask, other=0.0)

        tga00 = ga00 - px * gt0
        tga01 = ga01 - px * gt1
        tga02 = ga02 - px * gt2
        tga10 = ga10 - py * gt0
        tga11 = ga11 - py * gt1
        tga12 = ga12 - py * gt2
        tga20 = ga20 - pz * gt0
        tga21 = ga21 - pz * gt1
        tga22 = ga22 - pz * gt2

        if HAS_DELTA:
            d = tl.load(delta_ptr + b * K + s, mask=mask, other=0.0)
        else:
            th = tl.load(th_all_ptr + b * W_th + col, mask=mask, other=0.0)
            phi0 = tl.load(c_phi_ptr + r_idx * K + s, mask=mask, other=0.0)
            d = th - phi0

        cs = tl.cos(d)
        sn = tl.sin(d)

        tr = tga00 + tga11 + tga22
        aat = (tga00 * ax * ax + tga11 * ay * ay + tga22 * az * az
               + (tga01 + tga10) * ax * ay + (tga02 + tga20) * ax * az + (tga12 + tga21) * ay * az)
        kt = (tga01 - tga10) * az + (tga20 - tga02) * ay + (tga12 - tga21) * ax
        gd = -sn * (tr - aat) + cs * kt

        if HAS_DELTA:
            tl.store(g_delta_ptr + b * K + s, gd, mask=mask)
        else:
            tl.store(g_th_all_ptr + b * W_th + col, gd, mask=mask)
    else:
        li = s - K
        base_pb = (r_idx * L + li) * 3
        px = tl.load(c_pb_ptr + base_pb + 0, mask=mask, other=0.0)
        py = tl.load(c_pb_ptr + base_pb + 1, mask=mask, other=0.0)
        pz = tl.load(c_pb_ptr + base_pb + 2, mask=mask, other=0.0)

        gr00 = ga00 - px * gt0
        gr10 = ga01 - px * gt1
        gr20 = ga02 - px * gt2
        gr01 = ga10 - py * gt0
        gr11 = ga11 - py * gt1
        gr21 = ga12 - py * gt2
        gr02 = ga20 - pz * gt0
        gr12 = ga21 - pz * gt1
        gr22 = ga22 - pz * gt2

        base_th = b * W_th + col
        wx = tl.load(th_all_ptr + base_th + 3, mask=mask, other=0.0)
        wy = tl.load(th_all_ptr + base_th + 4, mask=mask, other=0.0)
        wz = tl.load(th_all_ptr + base_th + 5, mask=mask, other=0.0)

        ang_sq = tl.maximum(wx * wx + wy * wy + wz * wz, 1e-16)
        ang = tl.sqrt(ang_sq)
        safe = ang >= 1e-8
        ang_c = tl.maximum(ang, 1e-8)
        kx = tl.where(safe, wx / ang_c, wx)
        ky = tl.where(safe, wy / ang_c, wy)
        kz = tl.where(safe, wz / ang_c, wz)

        cs = tl.cos(ang)
        sn = tl.sin(ang)
        omc = 1.0 - cs

        vx = gr21 - gr12
        vy = gr02 - gr20
        vz = gr10 - gr01
        tr = gr00 + gr11 + gr22

        s00 = 2.0 * gr00
        s11 = 2.0 * gr11
        s22 = 2.0 * gr22
        s01 = gr01 + gr10
        s02 = gr02 + gr20
        s12 = gr12 + gr21

        wk_x = s00 * kx + s01 * ky + s02 * kz
        wk_y = s01 * kx + s11 * ky + s12 * kz
        wk_z = s02 * kx + s12 * ky + s22 * kz

        gk_x = sn * vx + omc * wk_x
        gk_y = sn * vy + omc * wk_y
        gk_z = sn * vz + omc * wk_z

        g_ang = -sn * tr + cs * (vx * kx + vy * ky + vz * kz) + 0.5 * sn * (kx * wk_x + ky * wk_y + kz * wk_z)
        dot_gk = gk_x * kx + gk_y * ky + gk_z * kz

        gw_x = tl.where(safe, (gk_x - dot_gk * kx) / ang_c + g_ang * kx, vx)
        gw_y = tl.where(safe, (gk_y - dot_gk * ky) / ang_c + g_ang * ky, vy)
        gw_z = tl.where(safe, (gk_z - dot_gk * kz) / ang_c + g_ang * kz, vz)

        tl.store(g_th_all_ptr + base_th + 0, gt0, mask=mask)
        tl.store(g_th_all_ptr + base_th + 1, gt1, mask=mask)
        tl.store(g_th_all_ptr + base_th + 2, gt2, mask=mask)
        tl.store(g_th_all_ptr + base_th + 3, gw_x, mask=mask)
        tl.store(g_th_all_ptr + base_th + 4, gw_y, mask=mask)
        tl.store(g_th_all_ptr + base_th + 5, gw_z, mask=mask)


class ComplexFK:
    def __init__(self, fks: List) -> None:
        from jacobik.ik import angle_bends as AB

        self.fks = fks
        self.dev = dev = fks[0].device
        self.supported = True
        self.ligs = [i for i, f in enumerate(fks) if f._lig_pos_t.size > 0]
        if any(fks[i]._donor_pos_t.size == 0 for i in self.ligs) or not self.ligs:
            self.supported = False           # pivot falls back to the centroid of ALL atoms: keep the reference path
            return
        L = len(self.ligs)
        quads, joint_lig, joint_col = [], [], []      # global joint tables
        coup_rows = []
        chains, atom_chain, moved = {}, [], []
        self.K_of = {}
        for li, i in enumerate(self.ligs):
            f = fks[i]
            J = f.n_joints
            quad = [f._quad_idx[k].tolist() for k in range(J)]
            down = [set(f._downstream[k].tolist()) for k in range(J)]
            lig_atoms = set(f._lig_pos_t.tolist())
            K0, coup = _plan_ligand(quad, down, lig_atoms)
            off = len(quads)
            # chains of this ligand's atoms; shrink the block if two joints cannot be ordered consistently
            while True:
                ok = True
                lig_chain = {}
                for a in sorted(lig_atoms):
                    js = [k for k in range(K0) if a in down[k] and a not in (quad[k][1], quad[k][2])]
                    js_sorted = sorted(js, key=cmp_to_key(lambda x, y: _inner_first(min(x, y), max(x, y), quad, down)
                                                          * (1 if x < y else -1)))
                    for p in range(len(js_sorted)):
                        for q in range(p + 1, len(js_sorted)):
                            x, y = js_sorted[p], js_sorted[q]
                            want = _inner_first(min(x, y), max(x, y), quad, down)      # -1: min first
                            if (want == -1) != (x < y):
                                ok = False
                                K0 = min(K0, max(x, y))
                    lig_chain[a] = js_sorted
                if ok:
                    break
            self.K_of[i] = K0
            for k in range(K0):
                quads.append(quad[k])
                joint_lig.append(li)
                joint_col.append(k)
                for (m, c) in coup.get(k, []):
                    coup_rows.append((off + k, off + m, c))
            for a in sorted(lig_atoms):
                key = tuple(off + k for k in lig_chain[a]) + (("base", li),)
                atom_chain.append(chains.setdefault(key, len(chains)))
                moved.append(a)
        K = len(quads)
        self.K, self.L = K, L
        ident = K + L
        steps = [[(K + j[1]) if isinstance(j, tuple) else j for j in c] for c in chains]
        self.depth = max(len(s) for s in steps)
        steps = [s + [ident] * (self.depth - len(s)) for s in steps]
        self.chain_steps = [cp.asarray([s[d] for s in steps], dtype=cp.int64) for d in range(self.depth)]
        self.atom_chain = cp.asarray(atom_chain, dtype=cp.int64)
        self.moved = cp.asarray(moved, dtype=cp.int64)
        self.quad_flat = cp.asarray([q for qd in quads for q in qd], dtype=cp.int64) if K else None
        self.joint_lig = joint_lig
        self.M_T = None
        if coup_rows:
            import numpy as np
            C = np.zeros((K, K), dtype=np.float64)
            for k, m, c in coup_rows:
                C[k, m] += c
            self.M_T = cp.asarray(np.linalg.inv(np.eye(K, dtype=np.float64) + C).T.astype(np.float32))
        N = int(max(int(f._lig_pos_t.max()) for f in (fks[i] for i in self.ligs))) + 1
        self._n_min = N
        self._donors = [fks[i]._donor_pos_t for i in self.ligs]
        # merged angle-bend plan (reuses BendPlan.apply)
        self.bend = None
        self.bend_ligs = [i for i in self.ligs if fks[i].n_angle_bends]
        if self.bend_ligs:
            self.bend = self._merge_bend_plans([fks[i]._bend_plan for i in self.bend_ligs], AB.BendPlan)
            self._bend_apply = AB.BendPlan.apply
        self._key = None
        if self.supported:
            self._ensure_triton_tables()

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_bend_plans(plans, cls):
        m = object.__new__(cls)
        cat = cp.concatenate
        s_off = k_off = ki_off = c_off = 0
        Ki_tot = sum(p.Ki for p in plans)
        depth = max(p.depth for p in plans)
        acc = {n: [] for n in ("spec_piv", "spec_oth", "spec_ref", "dof_spec", "dof_mode", "dof_col", "dof_scale",
                               "chain_steps", "moved", "atom_chain", "h_face", "h_part", "h_is", "h_is2",
                               "h_pa", "h_px", "h_pm", "h_pon")}
        any_h = any(p.n_hapt for p in plans)
        any_plane = any(getattr(p, "h_plane", False) for p in plans)
        Sm = max((p.h_pa.shape[1] for p in plans if getattr(p, "h_plane", False)), default=1)
        for p in plans:
            S = int(p.spec_piv.size)
            acc["spec_piv"].append(p.spec_piv); acc["spec_oth"].append(p.spec_oth); acc["spec_ref"].append(p.spec_ref)
            acc["dof_spec"].append(p.dof_spec + s_off); acc["dof_mode"].append(p.dof_mode)
            acc["dof_col"].append(p.dof_col + k_off); acc["dof_scale"].append(p.dof_scale)
            cs = p.chain_steps
            cs = cp.where(cs == p.Ki, cp.full_like(cs, Ki_tot), cs + ki_off)
            if cs.shape[1] < depth:
                cs = cat([cs, cp.full((cs.shape[0], depth - cs.shape[1]), Ki_tot, dtype=cs.dtype)], axis=1)
            acc["chain_steps"].append(cs)
            acc["moved"].append(p.moved); acc["atom_chain"].append(p.atom_chain + c_off)
            if any_h:
                if p.n_hapt:
                    acc["h_face"].append(p.h_face[:, :3]); acc["h_part"].append(p.h_part)
                    acc["h_is"].append(p.h_is); acc["h_is2"].append(p.h_is2)
                else:
                    acc["h_face"].append(cp.broadcast_to(p.spec_piv[:, None], (S, 3))); acc["h_part"].append(p.spec_piv)
                    z = cp.zeros(S, dtype=cp.float32)
                    acc["h_is"].append(z); acc["h_is2"].append(z)
            if any_plane:
                if getattr(p, "h_plane", False):
                    w = Sm - p.h_pa.shape[1]
                    acc["h_pa"].append(cat([p.h_pa, cp.broadcast_to(p.h_pa[:, :1], (S, w))], axis=1) if w else p.h_pa)
                    acc["h_px"].append(cat([p.h_px, cp.broadcast_to(p.h_px[:, :1], (S, w))], axis=1) if w else p.h_px)
                    acc["h_pm"].append(cp.pad(p.h_pm, ((0, 0), (0, w))) if w else p.h_pm)
                    acc["h_pon"].append(p.h_pon)
                else:
                    acc["h_pa"].append(cp.broadcast_to(p.spec_piv[:, None], (S, Sm)))
                    acc["h_px"].append(cp.broadcast_to(p.spec_oth[:, None], (S, Sm)))
                    acc["h_pm"].append(cp.zeros((S, Sm), dtype=cp.float32))
                    acc["h_pon"].append(cp.zeros(S, dtype=cp.float32))
            s_off += S; k_off += p.K; ki_off += p.Ki; c_off += cs.shape[0]
        for n, v in acc.items():
            if v:
                setattr(m, n, cat(v, axis=0))
        m.K, m.Ki, m.depth = k_off, Ki_tot, depth
        m.n_hapt = sum(p.n_hapt for p in plans)
        m.h_plane = any_plane
        if m.K > 0:
            m._ensure_triton_tables()
        return m

    # ------------------------------------------------------------------
    def reference(self, thetas, X_ref):
        X = X_ref
        for f, th in zip(self.fks, thetas):
            X = f.apply(th, X)
        return X

    def _ensure_triton_tables(self):
        if getattr(self, "_triton_tables_ready", False):
            return
        K, L = self.K, self.L
        KL = K + L
        base_cols, joint_cols, off = [], [], 0
        self._triton_lig_slices = []
        for i in self.ligs:
            Ki = self.K_of[i]
            base_cols.append(off)
            for k in range(Ki):
                joint_cols.append(off + 6 + k)
            self._triton_lig_slices.append((i, off, 6 + Ki))
            off += 6 + Ki
        self._triton_w_th = off
        self._triton_trans_col = cp.asarray(joint_cols + base_cols, dtype=cp.int32)

        cs = cp.ascontiguousarray(cp.stack(self.chain_steps, axis=1).astype(cp.int32))
        C, D = cs.shape
        self._triton_chain_steps = cs

        self._triton_chain_atoms, self._triton_chain_atom_ptr, self._triton_trans_occ_cd, self._triton_trans_occ_ptr = (
            _build_csr_tables(self.atom_chain, self.moved, cs, C, D, KL)
        )
        self._triton_tables_ready = True

    def _prepare(self, X_ref):
        key = (X_ref.data.ptr, tuple(X_ref.shape))
        if key == self._key:
            return
        Xr = X_ref if X_ref.ndim == 3 else X_ref[None, ...]
        R = Xr.shape[0]
        if self.K:
            P = Xr[:, self.quad_flat].reshape(R, self.K, 4, 3)
            b = P[:, :, 1:] - P[:, :, :-1]
            nrm = cp.cross(b[:, :, :2], b[:, :, 1:], axis=-1)
            n1, n2, b2 = nrm[:, :, 0], nrm[:, :, 1], b[:, :, 1]
            ax = b2 / cp.sqrt(cp.maximum((b2 * b2).sum(axis=-1, keepdims=True), _EPS * _EPS))
            m = cp.cross(n1, n2, axis=-1)
            self.c_phi = cp.arctan2((m * ax).sum(axis=-1), (n1 * n2).sum(axis=-1))
            self.c_ax = cp.ascontiguousarray(ax)
            self.c_piv_3 = cp.ascontiguousarray(P[:, :, 1])
        self.c_pb = cp.stack([Xr[:, d].mean(axis=1) for d in self._donors], axis=1)[:, :, None, :]
        self.c_pb_3 = cp.ascontiguousarray(self.c_pb[:, :, 0, :])
        self._key, self._ref = key, X_ref

    def apply(self, thetas, X_ref):
        fks = self.fks
        if not self.supported:
            X = X_ref
            for f, th in zip(fks, thetas):
                X = f.apply(th, X)
            return X
        n_of = [6 + f.n_joints for f in fks]
        bends = self.bend is not None and all(thetas[i].shape[1] >= n_of[i] + fks[i].n_angle_bends for i in self.bend_ligs)
        if self.bend is not None and not bends and any(thetas[i].shape[1] > n_of[i] for i in self.bend_ligs):
            X = X_ref
            for f, th in zip(fks, thetas):
                X = f.apply(th, X)
            return X
        X = self.forward_no_grad(thetas, X_ref)
        return X

    def forward_no_grad(self, thetas, X_ref):
        """Fast forward pass without autograd; caches state for ``vjp()``."""
        fks = self.fks
        n_of = [6 + f.n_joints for f in fks]
        bends = self.bend is not None and all(thetas[i].shape[1] >= n_of[i] + fks[i].n_angle_bends for i in self.bend_ligs)
        has_tail = any(self.K_of[i] < fks[i].n_joints for i in self.ligs)
        can_triton = (
            self.supported
            and not has_tail
            and not (self.bend is not None and not bends and any(thetas[i].shape[1] > n_of[i] for i in self.bend_ligs))
        )
        if not can_triton:
            X = X_ref
            fk_states = []
            for f, th in zip(fks, thetas):
                X, st = f.forward_with_state(th, X)
                fk_states.append(st)
            self._vjp_cache = ("fallback", fk_states)
            return X

        self._prepare(X_ref)
        self._ensure_triton_tables()
        th_all = cp.ascontiguousarray(cp.concatenate([thetas[i][:, :6 + self.K_of[i]] for i in self.ligs], axis=1))
        K, L = self.K, self.L
        delta = None
        if K and self.M_T is not None:
            th_k = cp.concatenate([thetas[i][:, 6:6 + self.K_of[i]] for i in self.ligs if self.K_of[i]], axis=1)
            diff_k = th_k - self.c_phi
            delta = (diff_k[:, :, None] * self.M_T[None, :, :]).sum(axis=1)
        Xr_c = cp.ascontiguousarray(X_ref if X_ref.ndim == 3 else X_ref[None, ...])
        B = th_all.shape[0]
        R_ref, N, _ = Xr_c.shape
        KL = K + L
        C, D = self._triton_chain_steps.shape
        W_th = self._triton_w_th
        Y = cp.broadcast_to(Xr_c, (B, N, 3)).copy() if R_ref == 1 else Xr_c.copy()
        At = cp.empty((B, KL, 12), dtype=th_all.dtype)
        G_hist = cp.empty((B, C, D, 12), dtype=th_all.dtype)
        has_delta = delta is not None
        delta_c = cp.ascontiguousarray(delta) if has_delta else th_all
        c_phi = self.c_phi if K else th_all
        c_ax = self.c_ax if K else th_all
        c_piv = self.c_piv_3 if K else th_all
        c_pb = self.c_pb_3
        BLOCK_B = 32
        gpu.launch(
            _cfk_transforms_fwd_k,
            (gpu.cdiv(B, BLOCK_B), KL),
            th_all, delta_c, c_phi, c_ax, c_piv, c_pb,
            self._triton_trans_col, At, B, R_ref, K, L, W_th,
            HAS_DELTA=has_delta, BLOCK_B=BLOCK_B, num_warps=1,
        )
        gpu.launch(
            _se3_chains_fwd_k,
            (gpu.cdiv(B, BLOCK_B), C),
            Xr_c, Y, At, G_hist, Y,
            self._triton_chain_steps, self._triton_chain_atom_ptr, self._triton_chain_atoms,
            B, R_ref, N, C, D, KL, ROW_VEC=True, SAVE_GFINAL=False, BLOCK_B=BLOCK_B, num_warps=1,
        )
        bend_cache = None
        X_out = Y
        if bends:
            beta = cp.ascontiguousarray(
                cp.concatenate([thetas[i][:, n_of[i]:n_of[i] + fks[i].n_angle_bends] for i in self.bend_ligs], axis=1)
            )
            X_out, bend_cache = self.bend.forward_with_state(Y, beta)
        self._vjp_cache = ("triton", [t.shape for t in thetas], n_of, th_all, delta_c, has_delta, Xr_c, At, G_hist, bend_cache)
        return X_out

    def vjp(self, grad_X):
        """Compute [dL/dtheta_0, ..., dL/dtheta_{L-1}] given grad_X = dL/dX (B, N, 3) or (M, B, N, 3)."""
        if grad_X.ndim == 4:
            return [self.vjp(grad_X[m]) for m in range(grad_X.shape[0])]
        cache = self._vjp_cache
        if cache[0] == "fallback":
            gX = cp.ascontiguousarray(grad_X)
            grads = [None] * len(self.fks)
            for i in range(len(self.fks) - 1, -1, -1):
                grads[i], gX = self.fks[i].vjp(cache[1][i], gX)
            return grads

        _, shapes, n_of, th_all, delta_c, has_delta, Xr_c, At, G_hist, bend_cache = cache
        gY = cp.ascontiguousarray(grad_X)
        B, W_th = th_all.shape
        R_ref, N, _ = Xr_c.shape
        BLOCK_B = 32
        gbeta = None
        if bend_cache is not None:
            gY, gbeta = self.bend.vjp(bend_cache, gY)
            gY = cp.ascontiguousarray(gY)

        K, L = self.K, self.L
        KL = K + L
        C, D = self._triton_chain_steps.shape
        g_th_all = cp.zeros_like(th_all)
        g_delta = cp.empty((B, K), dtype=th_all.dtype) if has_delta else g_th_all
        gAt_chain = cp.empty((B, C, D, 12), dtype=th_all.dtype)
        c_phi = self.c_phi if K else th_all
        c_ax = self.c_ax if K else th_all
        c_piv = self.c_piv_3 if K else th_all
        c_pb = self.c_pb_3
        gpu.launch(
            _se3_chains_bwd_k,
            (gpu.cdiv(B, BLOCK_B), C),
            Xr_c, gY, gY, At, G_hist, At, gAt_chain,
            self._triton_chain_steps, self._triton_chain_atom_ptr, self._triton_chain_atoms,
            B, R_ref, N, C, D, KL, ROW_VEC=True, USE_GFINAL=False, BLOCK_B=BLOCK_B, num_warps=1,
        )
        gpu.launch(
            _cfk_transforms_bwd_k,
            (gpu.cdiv(B, BLOCK_B), KL),
            th_all, delta_c, g_th_all, g_delta,
            c_phi, c_ax, c_piv, c_pb,
            self._triton_trans_col, self._triton_trans_occ_ptr, self._triton_trans_occ_cd,
            gAt_chain, B, R_ref, K, L, C, D, W_th,
            HAS_DELTA=has_delta, BLOCK_B=BLOCK_B, num_warps=1,
        )
        if has_delta:
            g_th_k = (g_delta[:, :, None] * self.M_T.T[None, :, :]).sum(axis=1)
        out = [cp.zeros(sh, dtype=th_all.dtype) for sh in shapes]
        k_off = 0
        for i, off, w in self._triton_lig_slices:
            out[i][:, :w] = g_th_all[:, off:off + w]
            if has_delta and self.K_of[i]:
                Ki = self.K_of[i]
                out[i][:, 6:6 + Ki] = g_th_k[:, k_off:k_off + Ki]
                k_off += Ki
        if gbeta is not None:
            b_off = 0
            for i in self.bend_ligs:
                nb = self.fks[i].n_angle_bends
                out[i][:, n_of[i]:n_of[i] + nb] = gbeta[:, b_off:b_off + nb]
                b_off += nb
        return out
