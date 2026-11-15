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
import torch

from jacobik.core import profile
from jacobik.core.env import RT_KJ
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


def eta2_plane_normal_t(b: torch.Tensor, A: torch.Tensor, Xs: torch.Tensor, mask: torch.Tensor, on: torch.Tensor):
    """Torch copy of orientation.eta2_plane_normal, batched: b (..., 3) unit bond, A / Xs (..., S, 3) face atom /
    substituent positions, mask (..., S) 1 for a plane substituent, on (...) 1 where the face qualifies. Returns the
    unsigned unit normal (..., 3) of the plane through the bond that best fits unit(Xs - A) with the bond component
    removed (principal axis of their 2-D scatter about b, closed form). Rows with on = 0 return a finite placeholder
    (callers select them out); their scatter is set to the identity's first row so atan2 keeps a finite gradient."""
    F = torch.nn.functional
    u = F.normalize(Xs - A, dim=-1)
    u = (u - (u * b[..., None, :]).sum(-1, keepdim=True) * b[..., None, :]) * mask[..., None]
    # reference axes built on the device (a host constant here is a CPU->GPU copy, illegal under CUDA graph capture)
    ex, ey = torch.zeros_like(b), torch.zeros_like(b)
    ex[..., 0] = 1.0
    ey[..., 1] = 1.0
    ax = torch.where(b[..., :1].abs() < 0.9, ex, ey)
    e1 = F.normalize(torch.cross(b, ax, dim=-1), dim=-1)
    e2 = torch.cross(b, e1, dim=-1)
    x, y = (u * e1[..., None, :]).sum(-1), (u * e2[..., None, :]).sum(-1)
    sxy = (x * y).sum(-1) * on
    dxy = ((x * x).sum(-1) - (y * y).sum(-1)) * on + (1.0 - on)
    th = 0.5 * torch.atan2(2.0 * sxy, dxy)
    d = torch.cos(th)[..., None] * e1 + torch.sin(th)[..., None] * e2
    return torch.cross(b, d, dim=-1)


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
        self.spec_piv = torch.tensor(piv, dtype=torch.long, device=device)
        self.spec_oth = torch.tensor(oth, dtype=torch.long, device=device)
        self.spec_ref = torch.tensor([pos[sp[2]] for sp in specs], dtype=torch.long, device=device)
        self.dof_spec = torch.tensor(dof_spec, dtype=torch.long, device=device)
        self.dof_mode = torch.tensor(dof_mode, dtype=torch.long, device=device)
        self.dof_col = torch.tensor(dof_col, dtype=torch.long, device=device)
        self.dof_scale = torch.tensor(dof_scale, dtype=torch.float32, device=device)
        self.dof_moved = [torch.tensor(sorted(s), dtype=torch.long, device=device) for s in dof_sets]
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
        self.chain_steps = torch.tensor([list(c) + [K] * (D - len(c)) for c in chains], dtype=torch.long,
                                        device=device).reshape(len(chains), D)
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
            self.h_face = torch.tensor(fidx, dtype=torch.long, device=device)
            self.h_fmask = torch.tensor(fmask, dtype=torch.float32, device=device)
            self.h_part = torch.tensor(part, dtype=torch.long, device=device)
            self.h_is = torch.tensor(is_h, dtype=torch.float32, device=device)
            self.h_is2 = torch.tensor(is_2, dtype=torch.float32, device=device)
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
            self.h_pa = torch.tensor(pa, dtype=torch.long, device=device)
            self.h_px = torch.tensor(px, dtype=torch.long, device=device)
            self.h_pm = torch.tensor(pm, dtype=torch.float32, device=device)
            self.h_pon = torch.tensor(pon, dtype=torch.float32, device=device)
            self.h_plane = any(pon)
        self.moved = torch.tensor(moved, dtype=torch.long, device=device)
        self.atom_chain = torch.tensor([cid[chain_of[a]] for a in moved], dtype=torch.long, device=device)

    def _haptic_ref(self, X, P, u, rv):
        """Axis reference of haptic specs (the other specs keep rv).
        Intrinsic reference: the normal of the substituent's own local plane -- eta2: the plane of the bond and the
        substituent (b x u); larger faces: the face plane (first three face atoms). Independent of where the metal
        happens to sit in the embed, so every face gets the same lean (out of its plane) / sideways (in plane) split;
        the sign is irrelevant (symmetric prior, symmetric wall). With a shared eta2 face plane (h_pon) the reference is
        that plane's normal (eta2_plane_normal_t), the same for every substituent of the face."""
        F = torch.nn.functional
        b = F.normalize(X[:, self.h_part] - P, dim=-1)
        m2 = torch.cross(b, u, dim=-1)  # |m2| = sin(angle between bond and substituent)
        # Near-linear substituent (alkyne, nitrile, ... in the embed): its own plane is undefined, so the lean
        # direction falls back to the face normal (metal direction from the bond midpoint, bond component removed,
        # as in orientation.haptic_features); blended by how well defined the local plane is, sign-aligned.
        mid = 0.5 * (P + X[:, self.h_part])
        nf = X[:, self.spec_ref] - mid
        nf = F.normalize(nf - (nf * b).sum(-1, keepdim=True) * b, dim=-1)
        sgn = torch.where((m2 * nf).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        mn = torch.sqrt((m2 * m2).sum(-1, keepdim=True) + 1e-12)  # safe norm: finite gradient at exactly linear
        wl = torch.clamp(mn / LOCAL_PLANE_SIN, max=1.0)
        m2 = wl * (m2 / mn) * sgn + (1.0 - wl) * nf
        if getattr(self, "h_plane", False):  # Python flag set at build: no device sync inside a captured graph
            on = self.h_pon[None, :].expand(X.shape[0], -1)
            ns = eta2_plane_normal_t(b, X[:, self.h_pa], X[:, self.h_px], self.h_pm[None].expand(X.shape[0], -1, -1), on)
            ns = ns * torch.where((ns * nf).sum(-1, keepdim=True) < 0, -1.0, 1.0)  # metal side: stable column sign
            m2 = on[..., None] * ns + (1.0 - on[..., None]) * m2
        f0, f1, f2 = (X[:, self.h_face[:, j]] for j in range(3))
        m3 = torch.cross(f1 - f0, f2 - f0, dim=-1)
        e2 = self.h_is2[None, :, None]
        r = e2 * m2 + (1.0 - e2) * m3
        hs = self.h_is[None, :, None]
        return hs * r + (1.0 - hs) * rv

    def axes_pivots(self, X: torch.Tensor):
        """Return the fixed axis and pivot used by each bend DOF for this pose."""
        F = torch.nn.functional
        P = X[:, self.spec_piv]
        u = F.normalize(X[:, self.spec_oth] - P, dim=-1)
        rv = X[:, self.spec_ref] - P
        if self.n_hapt:
            rv = self._haptic_ref(X, P, u, rv)
        n1 = F.normalize(torch.cross(u, rv, dim=-1), dim=-1)
        n2 = F.normalize(torch.cross(u, n1, dim=-1), dim=-1)
        axes = torch.stack([n1, n2, u], dim=2)
        return axes[:, self.dof_spec, self.dof_mode], P[:, self.dof_spec]

    def jacobian_column_norm_sq(self, X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Exact diagonal of the bend pullback metric for the rigid bend chain.

        Propagates derivative vectors through the same fixed-shape transform
        chain as ``apply``. Cost is O(number of moved atoms × bend DOFs × chain
        depth); the depth is a small topology constant, not another FK/JVP.
        """
        B = X.shape[0]
        K = self.K
        if K == 0:
            return X.new_zeros((B, 0))
        beta_i = beta[:, self.dof_col] * self.dof_scale.to(beta.dtype)
        axes, pivots = self.axes_pivots(X)
        ax, ay, az = axes.unbind(-1)
        z = torch.zeros_like(ax)
        Kx = torch.stack([z, -az, ay, az, z, -ax, -ay, ax, z], dim=-1).reshape(B, K, 3, 3)
        s, c = torch.sin(beta_i)[..., None, None], torch.cos(beta_i)[..., None, None]
        eye = torch.eye(3, dtype=X.dtype, device=X.device)
        R = eye + s * Kx + (1.0 - c) * (Kx @ Kx)
        t = pivots - (R @ pivots[..., None])[..., 0]
        R = torch.cat([R, eye.expand(B, 1, 3, 3)], dim=1)
        t = torch.cat([t, t.new_zeros(B, 1, 3)], dim=1)

        atom_chain = self.atom_chain
        idx_by_atom = self.chain_steps[atom_chain]
        current = X[:, self.moved]
        n_atoms = current.shape[1]
        jac = X.new_zeros((B, K, n_atoms, 3))
        for depth in range(self.depth):
            idx = idx_by_atom[:, depth]
            valid = idx < K
            Ri = R[:, idx]
            ti = t[:, idx]
            current = torch.einsum("baij,baj->bai", Ri, current) + ti
            jac = torch.einsum("baij,bkaj->bkai", Ri, jac)
            safe_idx = idx.clamp(max=K - 1)
            derivative = torch.linalg.cross(axes[:, safe_idx], current - pivots[:, safe_idx], dim=-1)
            derivative = derivative * self.dof_scale[safe_idx][None, :, None].to(derivative.dtype)
            one_hot = torch.nn.functional.one_hot(safe_idx, num_classes=K).to(derivative.dtype)
            add = (one_hot[None, :, :, None] * derivative[:, :, None, :]).permute(0, 2, 1, 3)
            jac = jac + add * valid[None, None, :, None]
        return jac.square().sum(dim=(2, 3))

    def apply(self, X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """X (B, N, 3) after the torsion FK, beta (B, K) -> X with every bend applied."""
        if self.K == 0:
            return X
        beta = beta[:, self.dof_col] * self.dof_scale.to(beta.dtype)  # (B, Ki) internal DOF angles
        F = torch.nn.functional
        B = X.shape[0]
        P = X[:, self.spec_piv]  # (B, S, 3)
        u = F.normalize(X[:, self.spec_oth] - P, dim=-1)
        rv = X[:, self.spec_ref] - P
        if self.n_hapt:
            rv = self._haptic_ref(X, P, u, rv)
        n1 = F.normalize(torch.cross(u, rv, dim=-1), dim=-1)
        n2 = F.normalize(torch.cross(u, n1, dim=-1), dim=-1)
        axes = torch.stack([n1, n2, u], dim=2)  # (B, S, 3 modes, 3)
        axis = axes[:, self.dof_spec, self.dof_mode]  # (B, K, 3)
        Pk = P[:, self.dof_spec]  # (B, K, 3)
        # Rodrigues matrices R = I + sin(b) [a]x + (1 - cos(b)) [a]x^2
        ax, ay, az = axis.unbind(-1)
        z = torch.zeros_like(ax)
        Kx = torch.stack([z, -az, ay, az, z, -ax, -ay, ax, z], dim=-1).reshape(B, self.Ki, 3, 3)
        s, c = torch.sin(beta)[..., None, None], torch.cos(beta)[..., None, None]
        eye = torch.eye(3, dtype=X.dtype, device=X.device)
        R = eye + s * Kx + (1.0 - c) * (Kx @ Kx)  # (B, K, 3, 3)
        t = Pk - (R @ Pk[..., None])[..., 0]  # (B, K, 3)
        # identity appended at index K for padded chain steps
        R = torch.cat([R, eye.expand(B, 1, 3, 3)], dim=1)
        t = torch.cat([t, t.new_zeros(B, 1, 3)], dim=1)
        C = self.chain_steps.shape[0]
        GR = eye.expand(B, C, 3, 3)
        Gt = X.new_zeros(B, C, 3)
        for d in range(self.depth):  # static nesting depth, not data-dependent
            idx = self.chain_steps[:, d]
            Rd, td = R[:, idx], t[:, idx]
            GR = Rd @ GR
            Gt = (Rd @ Gt[..., None])[..., 0] + td
        g = self.atom_chain
        Xm = X[:, self.moved]  # (B, A, 3)
        Xn = torch.einsum("baij,baj->bai", GR[:, g], Xm) + Gt[:, g]
        Y = X.index_copy(1, self.moved, Xn)
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


def energy_prior(beta: torch.Tensor, k: torch.Tensor, free_deg: float = FREE_DEG, scale: float = 1.0) -> torch.Tensor:
    """(B,) log-prior of bend amounts beta (B, n) in units of RT: -scale * sum_i k_i (max(0, |beta_i| - free))^2 / (2 RT).
    A Boltzmann factor of the harmonic bend energy beyond a free zone, summed over bends (not averaged)."""
    free = torch.deg2rad(free_deg)[None, :] if torch.is_tensor(free_deg) else math.radians(free_deg)
    ex = torch.clamp(beta.abs() - free, min=0.0)
    return -scale * (0.5 * k[None, :] * ex**2).sum(-1) / RT_KJ


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


def angles(X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Bond angles (radians) for index triplets t (n, 3) in position space."""
    F = torch.nn.functional
    u = F.normalize(X[:, t[:, 0]] - X[:, t[:, 1]], dim=-1)
    v = F.normalize(X[:, t[:, 2]] - X[:, t[:, 1]], dim=-1)
    return torch.acos((u * v).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))


def penalty(X: torch.Tensor, t: torch.Tensor, a0: torch.Tensor, free_deg: float, sigma_deg: float,
            weight: float) -> torch.Tensor:
    d = (angles(X, t) - a0[None]).abs()
    if np.isscalar(free_deg):
        free_rad = math.radians(float(free_deg))
    else:
        free = torch.as_tensor(free_deg, dtype=d.dtype, device=d.device)
        free_rad = torch.deg2rad(free[None, :] if free.ndim == 1 else free)
    ex = torch.clamp(d - free_rad, min=0.0) / math.radians(sigma_deg)
    return -weight * 0.5 * (ex**2).mean(-1)
