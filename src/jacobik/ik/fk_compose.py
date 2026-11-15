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

import torch

_EPS = 1e-8

#: How the 3x3 products of the composed maps are evaluated. "elementwise" (default): a broadcast multiply and a
#: 3-term sum (ordinary elementwise kernels, forward and backward). "bmm": torch matmul, i.e. cuBLAS batched gemm on
#: CUDA, which is slow for millions of 3x3 products: with it the solver took 1.1-1.7x as long at 3-8 seeds per pass
#: (RTX 2060 SUPER, 2026-10-03). Same mathematics; results agree to float rounding.
SMALL_MATMUL: str = "elementwise"


def _vecmat(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Row vectors times 3x3 matrices: (..., 3), (..., 3, 3) -> (..., 3); out_j = sum_i v_i M_ij."""
    if SMALL_MATMUL == "elementwise":
        return (v.unsqueeze(-1) * M).sum(dim=-2)
    return (v.unsqueeze(-2) @ M).squeeze(-2)


def _matmat(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """3x3 times 3x3: out_ij = sum_k A_ik B_kj."""
    if SMALL_MATMUL == "elementwise":
        return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(dim=-2)
    return A @ B


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


class ComplexFK:
    def __init__(self, fks: List) -> None:
        from jacobik.ik import angle_bends as AB

        self.fks = fks
        self.dev = dev = fks[0].device
        self.supported = True
        self.ligs = [i for i, f in enumerate(fks) if f._lig_pos_t.numel() > 0]
        if any(fks[i]._donor_pos_t.numel() == 0 for i in self.ligs) or not self.ligs:
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
        tl = lambda x: torch.tensor(x, dtype=torch.long, device=dev)  # noqa: E731
        self.chain_steps = [tl([s[d] for s in steps]) for d in range(self.depth)]
        self.atom_chain, self.moved = tl(atom_chain), tl(moved)
        self.quad_flat = tl([q for qd in quads for q in qd]) if K else None
        self.joint_lig = joint_lig
        self.M_T = None
        if coup_rows:
            C = torch.zeros(K, K, dtype=torch.float64)
            for k, m, c in coup_rows:
                C[k, m] += c
            self.M_T = torch.linalg.inv(torch.eye(K, dtype=torch.float64) + C).T.to(torch.float32).to(dev)
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

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_bend_plans(plans, cls):
        m = object.__new__(cls)
        cat = torch.cat
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
            S = p.spec_piv.numel()
            acc["spec_piv"].append(p.spec_piv); acc["spec_oth"].append(p.spec_oth); acc["spec_ref"].append(p.spec_ref)
            acc["dof_spec"].append(p.dof_spec + s_off); acc["dof_mode"].append(p.dof_mode)
            acc["dof_col"].append(p.dof_col + k_off); acc["dof_scale"].append(p.dof_scale)
            cs = p.chain_steps
            cs = torch.where(cs == p.Ki, torch.full_like(cs, Ki_tot), cs + ki_off)
            if cs.shape[1] < depth:
                cs = cat([cs, torch.full((cs.shape[0], depth - cs.shape[1]), Ki_tot, dtype=cs.dtype, device=cs.device)], 1)
            acc["chain_steps"].append(cs)
            acc["moved"].append(p.moved); acc["atom_chain"].append(p.atom_chain + c_off)
            if any_h:
                if p.n_hapt:
                    acc["h_face"].append(p.h_face[:, :3]); acc["h_part"].append(p.h_part)
                    acc["h_is"].append(p.h_is); acc["h_is2"].append(p.h_is2)
                else:
                    acc["h_face"].append(p.spec_piv[:, None].expand(S, 3)); acc["h_part"].append(p.spec_piv)
                    z = torch.zeros(S, dtype=torch.float32, device=p.spec_piv.device)
                    acc["h_is"].append(z); acc["h_is2"].append(z)
            if any_plane:
                if getattr(p, "h_plane", False):
                    w = Sm - p.h_pa.shape[1]
                    # padding repeats column 0 (a real bond, mask 0): no zero-length vector in the graph
                    acc["h_pa"].append(cat([p.h_pa, p.h_pa[:, :1].expand(S, w)], 1) if w else p.h_pa)
                    acc["h_px"].append(cat([p.h_px, p.h_px[:, :1].expand(S, w)], 1) if w else p.h_px)
                    acc["h_pm"].append(torch.nn.functional.pad(p.h_pm, (0, w)) if w else p.h_pm)
                    acc["h_pon"].append(p.h_pon)
                else:
                    acc["h_pa"].append(p.spec_piv[:, None].expand(S, Sm)); acc["h_px"].append(p.spec_oth[:, None].expand(S, Sm))
                    acc["h_pm"].append(torch.zeros(S, Sm, dtype=torch.float32, device=p.spec_piv.device))
                    acc["h_pon"].append(torch.zeros(S, dtype=torch.float32, device=p.spec_piv.device))
            s_off += S; k_off += p.K; ki_off += p.Ki; c_off += cs.shape[0]
        for n, v in acc.items():
            if v:
                setattr(m, n, cat(v, 0))
        m.K, m.Ki, m.depth = k_off, Ki_tot, depth
        m.n_hapt = sum(p.n_hapt for p in plans)
        m.h_plane = any_plane
        return m

    # ------------------------------------------------------------------
    def reference(self, thetas, X_ref):
        X = X_ref
        for f, th in zip(self.fks, thetas):
            X = f.apply(th, X)
        return X

    def _prepare(self, X_ref):
        key = (X_ref.data_ptr(), tuple(X_ref.shape), X_ref._version)
        if key == self._key:
            return
        with torch.no_grad():
            Xr = X_ref if X_ref.dim() == 3 else X_ref.unsqueeze(0)          # (R, N, 3), R = 1 or B
            R = Xr.shape[0]
            if self.K:
                P = Xr.index_select(1, self.quad_flat).reshape(R, self.K, 4, 3)
                b = P[:, :, 1:] - P[:, :, :-1]
                nrm = torch.linalg.cross(b[:, :, :2], b[:, :, 1:], dim=-1)
                n1, n2, b2 = nrm[:, :, 0], nrm[:, :, 1], b[:, :, 1]
                ax = b2 / (b2 * b2).sum(dim=-1, keepdim=True).clamp(min=_EPS * _EPS).sqrt()
                m = torch.linalg.cross(n1, n2, dim=-1)
                self.c_phi = torch.atan2((m * ax).sum(dim=-1), (n1 * n2).sum(dim=-1))        # (R, K)
                eye = torch.eye(3, dtype=Xr.dtype, device=Xr.device)
                self.c_aaT = ax[..., :, None] * ax[..., None, :]
                self.c_KT = torch.linalg.cross(ax[:, :, None, :].expand(-1, -1, 3, -1), eye.expand(R, self.K, 3, 3), dim=-1)
                self.c_piv = P[:, :, 1].contiguous()[:, :, None, :]                          # (R, K, 1, 3)
            self.c_pb = torch.stack([Xr.index_select(1, d).mean(dim=1) for d in self._donors], dim=1)[:, :, None, :]
            self.c_Xm = Xr.index_select(1, self.moved)[:, :, None, :]                        # (R, A, 1, 3)
        self._key, self._ref = key, X_ref                    # the reference keeps the storage (and its address) alive

    def apply(self, thetas, X_ref):
        from jacobik.ik.kernels import axisangle_to_matrix

        fks = self.fks
        if not self.supported or X_ref.requires_grad:
            return self.reference(thetas, X_ref)
        n_of = [6 + f.n_joints for f in fks]
        bends = self.bend is not None and all(thetas[i].shape[1] >= n_of[i] + fks[i].n_angle_bends for i in self.bend_ligs)
        if self.bend is not None and not bends and any(thetas[i].shape[1] > n_of[i] for i in self.bend_ligs):
            return self.reference(thetas, X_ref)
        self._prepare(X_ref)
        B = thetas[0].shape[0]
        K, L = self.K, self.L
        eye = torch.eye(3, dtype=X_ref.dtype, device=X_ref.device)
        maps_A, maps_b = [], []
        if K:
            th = torch.cat([thetas[i][:, 6:6 + self.K_of[i]] for i in self.ligs if self.K_of[i]], dim=1)   # (B, K)
            delta = th - self.c_phi
            if self.M_T is not None:
                delta = delta @ self.M_T
            c = torch.cos(delta)[..., None, None]
            s = torch.sin(delta)[..., None, None]
            RT = c * eye + (1.0 - c) * self.c_aaT + s * self.c_KT                             # (B, K, 3, 3)
            maps_A.append(RT)
            maps_b.append((self.c_piv - _vecmat(self.c_piv, RT.unsqueeze(-3)))[:, :, 0, :])
        base = torch.stack([thetas[i][:, :6] for i in self.ligs], dim=1)                      # (B, L, 6)
        Ab = axisangle_to_matrix(base[:, :, 3:6].reshape(B * L, 3)).reshape(B, L, 3, 3).transpose(-1, -2)
        maps_A.append(Ab)
        maps_b.append((self.c_pb - _vecmat(self.c_pb, Ab.unsqueeze(-3)))[:, :, 0, :] + base[:, :, 0:3])
        maps_A.append(eye.expand(B, 1, 3, 3))
        maps_b.append(base.new_zeros(B, 1, 3))
        A = torch.cat(maps_A, dim=1)
        t = torch.cat(maps_b, dim=1)
        GA = A.index_select(1, self.chain_steps[0])
        Gb = t.index_select(1, self.chain_steps[0])
        for d in range(1, self.depth):
            Ad = A.index_select(1, self.chain_steps[d])
            GA = _matmat(GA, Ad)
            Gb = _vecmat(Gb, Ad) + t.index_select(1, self.chain_steps[d])
        Xn = _vecmat(self.c_Xm, GA.index_select(1, self.atom_chain).unsqueeze(-3))[:, :, 0, :] + Gb.index_select(1, self.atom_chain)
        Xfull = X_ref if X_ref.dim() == 3 else X_ref.unsqueeze(0).expand(B, -1, -1)
        X = Xfull.index_copy(1, self.moved, Xn)
        for i in self.ligs:                                   # joints left on the reference path
            f = fks[i]
            if self.K_of[i] < f.n_joints:
                th_i = thetas[i][:, :n_of[i]] if thetas[i].shape[1] > n_of[i] else thetas[i]
                for k in range(self.K_of[i], f.n_joints):
                    X = f._apply_torsion(th_i, X, k)
        if bends:
            beta = torch.cat([thetas[i][:, n_of[i]:n_of[i] + fks[i].n_angle_bends] for i in self.bend_ligs], dim=1)
            X = self._bend_apply(self.bend, X, beta)
        return X
