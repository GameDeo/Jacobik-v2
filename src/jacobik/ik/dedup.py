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
"""
from __future__ import annotations

import os
import logging
from collections import OrderedDict
import itertools
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from jacobik.core import profile
from jacobik.core.env import torch_device
from jacobik.core.geometry import kabsch_fit_batch, kabsch_fit_np  # noqa: F401

log = logging.getLogger(__name__)
# Experiment / profile configuration
DEDUP_FACTORED = profile.get("dedup_factored")
DEDUP_RMSD = profile.get("dedup_rmsd")
DEDUP_MAX_DEV = profile.get("dedup_max_dev")
OUT_DEDUP = True
MAX_AUTOMORPHISMS = 20000
#: Where the RMSD work of the dedup runs for a CUDA solve: "cpu" (default) or "solver" (on the solver's GPU; same
#: algorithm, float64). The scans over automorphisms are matrix products and a closed-form rotation fit since
#: 2026-10-03 (JEFJUZ, 3200 automorphisms, 64 conformers: 39 s -> 2.4 s on the CPU, same selection), so the CPU is
#: fast enough; on the GPU the old per-matrix SVD scans took about 70 s per seed for that complex (RTX 2060 SUPER).
DEDUP_DEVICE = os.environ.get("JACOBIK_DEDUP_DEVICE", "cpu")
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
        A_row: torch.Tensor,
        B_cand: torch.Tensor,
        th: float,
        max_dev_th: float,
        cand_indices: torch.Tensor,
    ):
        """Find if any row in B_cand has Kabsch RMSD <= th and max per-atom deviation <= max_dev_th."""
        A_np = A_row.cpu().numpy()
        B_np = B_cand.cpu().numpy()
        Ac = A_np - A_np.mean(0)
        best_match = (False, np.inf, np.inf, None)
        maps_np = self.maps.cpu().numpy()
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
                        return True, r, md, cand_indices[c_idx].item()
                    if r < best_match[1]:
                        best_match = (False, r, md, cand_indices[c_idx].item())
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
        H0_core = np.einsum("kni,pnj->pkij", Bc, Ac)
        Uc, Sc, Vtc = np.linalg.svd(H0_core)
        dc = np.sign(np.linalg.det(np.einsum("pkij,pkjl->pkil", Uc, Vtc)))
        sv = Sc[..., 0] + Sc[..., 1] + dc * Sc[..., 2]
        msd = ((Ac ** 2).sum((-1, -2))[:, None] + (Bc ** 2).sum((-1, -2))[None] - 2.0 * sv) / nc
        core_rmsd = np.sqrt(np.maximum(msd, 0.0))
        max_core_th = th * np.sqrt(n / nc) + 0.25

        order = np.argsort(core_rmsd, axis=1)
        sorted_vals = np.take_along_axis(core_rmsd, order, axis=1)
        order36 = order[:, :36]
        cnt = (sorted_vals[:, :36] <= max_core_th).sum(1)
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
        U0, _, Vt0 = np.linalg.svd(H0)
        d0 = np.sign(np.linalg.det(U0 @ Vt0))
        U0[:, :, 2] *= d0[:, None]
        Bt = (q[None] - Qp_m) @ (U0 @ Vt0) + Kc_m

        small_groups, large_groups = self._factored_groups()
        K_norm2 = (Ks * Ks).sum((-1, -2))
        q_norm2 = float((q * q).sum())
        Kg_small = [(gidx, ar_g, pm, Ks[:, gidx], (Ks[:, gidx] ** 2).sum(-1)[:, :, None]) for gidx, ar_g, pm in small_groups]
        Kg_large = [(gidx, Ks[:, gidx], (Ks[:, gidx] ** 2).sum(-1)[:, :, None]) for gidx in large_groups]

        base_row = np.arange(n)
        mp = np.broadcast_to(base_row, (Sn, n)).copy()
        matched = np.zeros(P, bool)
        first_hit = None
        adj = self.adj

        for rnd in range(12):
            L = len(pidx)
            if not L:
                break
            new = np.broadcast_to(base_row, (L, n)).copy()
            for gidx, ar_g, pm, Ka, Ka2 in Kg_small:
                Bb = Bt[:, gidx]
                Bb2 = (Bb * Bb).sum(-1)[:, None, :]
                C = Ka2 + Bb2 - 2.0 * (Ka @ Bb.swapaxes(1, 2))
                cost = C[:, ar_g, pm].sum(-1)
                new[:, gidx] = gidx[pm[cost.argmin(1)]]
            for gidx, Ka, Ka2 in Kg_large:
                Bb = Bt[:, gidx]
                Bb2 = (Bb * Bb).sum(-1)[:, None, :]
                C = Ka2 + Bb2 - 2.0 * (Ka @ Bb.swapaxes(1, 2))
                cols = np.array([linear_sum_assignment(Ct)[1] for Ct in C])
                new[:, gidx] = gidx[cols]

            Qn = q[new]
            H = Qn.swapaxes(1, 2) @ Ks
            U, S_, Vt = np.linalg.svd(H)
            ds = np.sign(np.linalg.det(U @ Vt))
            sv = S_[:, 0] + S_[:, 1] + ds * S_[:, 2]
            rr = np.sqrt(np.maximum((K_norm2 + q_norm2 - 2.0 * sv) / n, 0.0))
            stable = (new == mp).all(1)

            cand_ok = (rr <= th) & (stable if rnd < 11 else True)
            if cand_ok.any():
                for s_idx in np.flatnonzero(cand_ok):
                    p_row = int(pidx[s_idx])
                    if matched[p_row]:
                        continue
                    m = new[s_idx]
                    if (adj[m][:, m] == adj).all():
                        U_h = U[s_idx].copy()
                        U_h[:, 2] *= ds[s_idx]
                        Rh = U_h @ Vt[s_idx]
                        md_h = float(np.sqrt(((Ks[s_idx] - q[m] @ Rh) ** 2).sum(-1).max()))
                        if max_dev_th > 0.0 and md_h > max_dev_th:
                            continue
                        matched[p_row] = True
                        if first_hit is None or p_row < first_hit[0]:
                            first_hit = (p_row, float(rr[s_idx]), md_h, m.copy())
                        if stop_on_any:
                            return matched, first_hit

            keep_s = (~stable) & (~matched[pidx])
            if not keep_s.any():
                break
            # Collapse starts on the same candidate that converged to identical permutations
            if int(keep_s.sum()) > 1:
                seen = set()
                for idx_k in np.flatnonzero(keep_s):
                    key = (int(pidx[idx_k]), new[idx_k].tobytes())
                    if key in seen:
                        keep_s[idx_k] = False
                    else:
                        seen.add(key)

            U_u = U[keep_s]
            U_u[:, :, 2] *= ds[keep_s, None]
            Rn = U_u @ Vt[keep_s]
            Bt = q[None] @ Rn
            pidx = pidx[keep_s]
            Ks = Ks[keep_s]
            K_norm2 = K_norm2[keep_s]
            mp = new[keep_s]
            Kg_small = [(gidx, ar_g, pm, Ka[keep_s], Ka2[keep_s]) for gidx, ar_g, pm, Ka, Ka2 in Kg_small]
            Kg_large = [(gidx, Ka[keep_s], Ka2[keep_s]) for gidx, Ka, Ka2 in Kg_large]

        return matched, first_hit

    def _factored_any(self, K: np.ndarray, q: np.ndarray, th: float, max_dev_th: float = 0.0) -> np.ndarray:
        """factored_sym_upper(K[p], q, th, max_dev_th)[0] for every row p of K (P, n, 3) at once (P2R out_mirror)."""
        return self._factored_match(K, q, th, max_dev_th=max_dev_th, stop_on_any=False)[0]

    def mirror_dup(self, Xk: torch.Tensor, th: Optional[float] = None) -> List[bool]:
        """P2R out_mirror: for each kept row i (K, N, 3), is its reflection a duplicate of any kept row j (itself
        included)? The decision of `any_dup(reflections, Xk)`, faster on capped automorphism groups, where the
        per-pair factored test (about 0.04 s per pair, exhaustive when the pair is distinct, which most reflections
        are) dominated (HEDJOP 32 rows: 809 pairs, 36 s per seed):
          * the batched capped-map test (_full_any) on every radial candidate, as in any_dup;
          * the factored test once per unordered pair (RMSD(refl i, j) = RMSD(refl j, i), so a pair found for i also
            marks j), all candidates of a row in one batched call (_factored_any).
        Uncapped groups: exactly any_dup."""
        if th is None:
            th = DEDUP_RMSD
        K = Xk.shape[0]
        if K == 0:
            return []
        refl = torch.tensor([-1.0, 1.0, 1.0], dtype=Xk.dtype, device=Xk.device)
        if not (DEDUP_FACTORED and (self.is_capped or DEDUP_MAX_DEV > 0.0)):
            return self.any_dup(Xk * refl, Xk, th)
        A = (Xk * refl)[:, self.heavy].to(self.device, self.dtype); A = A - A.mean(1, keepdim=True)
        B = Xk[:, self.heavy].to(self.device, self.dtype); B = B - B.mean(1, keepdim=True)
        RA, RB = self._radial(torch.nan_to_num(A)), self._radial(torch.nan_to_num(B))
        dup = [False] * K
        cands = []
        for i in range(K):
            cand = torch.nonzero(((RB - RA[i]) ** 2).mean(1).sqrt() <= th).flatten()
            cands.append(cand)
            if cand.numel() and self._full_any(A[i], B[cand], th):
                dup[i] = DEDUP_MAX_DEV <= 0.0 or self._verify_max_dev(A[i], B[cand], th, DEDUP_MAX_DEV, cand)[0]
        if not self.is_capped:
            return dup
        An, Bn = A.cpu().numpy(), B.cpu().numpy()
        tested = set()
        for i in range(K):
            if dup[i]:
                continue
            cl = [c for c in cands[i].tolist() if (min(i, c), max(i, c)) not in tested]
            tested.update((min(i, c), max(i, c)) for c in cl)
            if not cl:
                continue
            hit = self._factored_any(Bn[cl], An[i], th, DEDUP_MAX_DEV)
            if hit.any():
                dup[i] = True
                for c in np.asarray(cl)[hit]:
                    dup[int(c)] = True       # refl(c) ~ i as well
        return dup


class StructureDeduper(_FactoredDedupMixin):
    def __init__(self, mol, device=None, dtype=torch.float64):
        from rdkit import Chem

        self.device = device or torch.device("cpu")
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
        self.maps = torch.as_tensor(np.vstack([ident[None], m]), dtype=torch.long, device=self.device)
        # inverse of every automorphism: sum_n f(A[n], B[map[n]]) == sum_k f(A[inv[k]], B[k]), which turns the scans
        # over automorphisms below into plain matrix products instead of (rows, maps, atoms, 3) gathers
        self.inv_maps = torch.argsort(self.maps, dim=1)
        self.heavy = torch.as_tensor(heavy, dtype=torch.long, device=self.device)
        cls = np.array(list(Chem.CanonicalRankAtoms(g, breakTies=False)))
        order = np.argsort(cls, kind="stable")
        self._rad_order = torch.as_tensor(order, device=self.device)
        self._rad_groups = [torch.as_tensor(np.flatnonzero(cls[order] == c), device=self.device) for c in np.unique(cls)]
        self.merges: List[Dict[str, Any]] = []
        self._init_factored(mol, g, heavy, pos, adj)

    # --- geometry helpers -------------------------------------------------------------------------------------------
    def _radial(self, X):
        r = (X - X.mean(1, keepdim=True)).norm(dim=-1)[:, self._rad_order]
        out = r.clone()
        for g in self._rad_groups:
            out[:, g] = torch.sort(r[:, g], dim=1).values
        return out

    @staticmethod
    def _rms_fixed_map(A, B):
        """A (n,3), B (R,n,3), both centred -> (R,) Kabsch RMSD with the given atom order."""
        H = torch.einsum("ni,rnj->rij", A, B)
        U, S, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.det(U) * torch.det(Vt))
        s = S[:, 0] + S[:, 1] + d * S[:, 2]
        msd = ((A**2).sum() + (B**2).sum((-1, -2)) - 2 * s) / A.shape[0]
        return msd.clamp(min=0).sqrt()

    def _upper(self, A, B, iters=3):
        R = B.shape[0]
        cur = torch.zeros(R, dtype=torch.long, device=B.device)
        ar = torch.arange(R, device=B.device)
        for _ in range(iters):
            Bm = B[ar[:, None], self.maps[cur]]
            H = torch.einsum("rni,nj->rij", Bm, A)
            U, S, Vt = torch.linalg.svd(H)
            d = torch.sign(torch.det(U @ Vt))
            D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
            Rot = (U @ D @ Vt).transpose(-1, -2)
            BR = torch.einsum("rnj,rij->rni", B, Rot)
            # argmin_m sum_n |BR[map_m[n]] - A[n]|^2 = argmax_m sum_k BR[k] . A[inv_m[k]] (the norms do not depend on m)
            cur = (BR.reshape(R, -1) @ self._a_inv(A).reshape(self.maps.shape[0], -1).T).argmax(1)
        return self._rms_fixed_map(A, B[ar[:, None], self.maps[cur]])

    def _a_inv(self, A):
        """A (n, 3) under every inverse automorphism: (M, n, 3)."""
        return A[self.inv_maps]

    @staticmethod
    def _max_overlap(H, bound):
        """max over proper rotations R of sum_n (R a_n) . b_n for correlation matrices H = sum_n a_n b_n^T (..., 3, 3):
        the largest eigenvalue of Horn's symmetric 4x4 matrix, found by Newton's method on its characteristic
        polynomial l^4 + c2 l^2 + c1 l + c0 from the upper bound `bound` >= l_max (Theobald's QCP method). All roots
        are real, so the iteration decreases monotonically to the largest one. Equals s1 + s2 + sign(det H) s3 of the
        singular values, without a decomposition per matrix."""
        Sxx, Sxy, Sxz = H[..., 0, 0], H[..., 0, 1], H[..., 0, 2]
        Syx, Syy, Syz = H[..., 1, 0], H[..., 1, 1], H[..., 1, 2]
        Szx, Szy, Szz = H[..., 2, 0], H[..., 2, 1], H[..., 2, 2]
        m00, m01, m02, m03 = Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx
        m11, m12, m13 = Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz
        m22, m23 = -Sxx + Syy - Szz, Syz + Szy
        m33 = -Sxx - Syy + Szz
        c2 = -2.0 * (H * H).sum((-1, -2))
        c1 = -8.0 * (Sxx * (Syy * Szz - Syz * Szy) - Sxy * (Syx * Szz - Syz * Szx) + Sxz * (Syx * Szy - Syy * Szx))
        # determinant of the symmetric 4x4 matrix by 2x2 minors
        s0, s1, s2 = m00 * m11 - m01 * m01, m00 * m12 - m01 * m02, m00 * m13 - m01 * m03
        s3, s4, s5 = m01 * m12 - m11 * m02, m01 * m13 - m11 * m03, m02 * m13 - m12 * m03
        k5, k4, k3 = m22 * m33 - m23 * m23, m12 * m33 - m13 * m23, m12 * m23 - m13 * m22
        k2, k1, k0 = m02 * m33 - m03 * m23, m02 * m23 - m03 * m22, m02 * m13 - m03 * m12
        c0 = s0 * k5 - s1 * k4 + s2 * k3 + s3 * k2 - s4 * k1 + s5 * k0
        lam = bound.expand(c2.shape).clone()
        for _ in range(60):
            l2 = lam * lam
            p = (l2 + c2) * l2 + c1 * lam + c0
            dp = (4.0 * l2 + 2.0 * c2) * lam + c1
            step = torch.where(dp.abs() > 1e-300, p / dp, torch.zeros_like(p))
            lam = lam - step
            if float((step.abs() / lam.abs().clamp(min=1e-30)).max()) < 1e-14:
                break
        return lam

    def _full(self, A, B, chunk=2048):
        """Exact minimum Kabsch RMSD over all automorphisms: A (n, 3), B (R, n, 3), both centred -> (R,)."""
        best = torch.full((B.shape[0],), float("inf"), dtype=B.dtype, device=B.device)
        ga, gb = (A**2).sum(), (B**2).sum((-1, -2))  # (), (R,)
        bound = 0.5 * (ga + gb)[:, None]  # (R, 1) >= the largest eigenvalue (Cauchy-Schwarz)
        n = A.shape[0]
        Ainv = self._a_inv(A)  # (M, n, 3)
        for s in range(0, Ainv.shape[0], chunk):
            Ap = Ainv[s : s + chunk]
            M = Ap.shape[0]
            # H[r, m] = sum_k A[inv_m[k]] B[r, k]^T: one matrix product for all rows and automorphisms
            H = (Ap.transpose(1, 2).reshape(M * 3, n) @ B).reshape(B.shape[0], M, 3, 3)
            sv = self._max_overlap(H, bound)
            msd = (ga + gb[:, None] - 2 * sv) / n
            best = torch.minimum(best, msd.clamp(min=0).sqrt().min(1).values)
        return best

    @staticmethod
    def _quartic(H):
        """Coefficients (c2, c1, c0) of the characteristic polynomial l^4 + c2 l^2 + c1 l + c0 of Horn's 4x4 matrix
        for H (K, 3, 3) (see _max_overlap)."""
        Sxx, Sxy, Sxz = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
        Syx, Syy, Syz = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
        Szx, Szy, Szz = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
        m00, m01, m02, m03 = Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx
        m11, m12, m13 = Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz
        m22, m23 = -Sxx + Syy - Szz, Syz + Szy
        m33 = -Sxx - Syy + Szz
        c2 = -2.0 * (H * H).sum((-1, -2))
        c1 = -8.0 * (Sxx * (Syy * Szz - Syz * Szy) - Sxy * (Syx * Szz - Syz * Szx) + Sxz * (Syx * Szy - Syy * Szx))
        s0, s1, s2 = m00 * m11 - m01 * m01, m00 * m12 - m01 * m02, m00 * m13 - m01 * m03
        s3, s4, s5 = m01 * m12 - m11 * m02, m01 * m13 - m11 * m03, m02 * m13 - m12 * m03
        k5, k4, k3 = m22 * m33 - m23 * m23, m12 * m33 - m13 * m23, m12 * m23 - m13 * m22
        k2, k1, k0 = m02 * m33 - m03 * m23, m02 * m23 - m03 * m22, m02 * m13 - m03 * m12
        return c2, c1, s0 * k5 - s1 * k4 + s2 * k3 + s3 * k2 - s4 * k1 + s5 * k0

    @staticmethod
    def _newton_decide(H, lam, need, max_iter=60):
        """Per element of H (K, 3, 3): whether the largest Horn eigenvalue reaches `need` (K,); Newton from the upper
        bounds `lam` (K,). Element-wise the same rule as _newton_any: yes when the iterate converges at or above
        `need` (or is still above it after max_iter steps), no as soon as it falls below."""
        c2, c1, c0 = StructureDeduper._quartic(H)
        acc = torch.zeros_like(need, dtype=torch.bool)
        und = torch.ones_like(acc)
        for _ in range(max_iter):
            l2 = lam * lam
            p = (l2 + c2) * l2 + c1 * lam + c0
            dp = (4.0 * l2 + 2.0 * c2) * lam + c1
            step = torch.where(dp.abs() > 1e-300, p / dp, torch.zeros_like(p))
            lam = lam - step
            conv = (step.abs() / lam.abs().clamp(min=1e-30)) < 1e-14
            above = lam >= need
            acc |= und & conv & above
            und &= above & ~conv
            if not bool(und.any()):
                break
        return acc | und

    def _pair_dup(self, A, B, RA, RB, th, lower=False, budget=1 << 22):
        """(a, r) duplicate matrix (b, R) bool: min RMSD over automorphisms <= th, for A (b, n, 3) against B (R, n, 3),
        all centred; RA, RB their class-sorted radial profiles (_radial). Pair by pair the decision of _full_any
        (same bounds, same Newton rule), for a block of rows at once: the per-row loop of tensor operations was most
        of the dedup time for complexes with few automorphisms and many rows. `lower`: A is B, only pairs r < a.
        Work arrays stay under about `budget` doubles (automorphisms in chunks)."""
        b, n = A.shape[0], A.shape[1]
        R = B.shape[0]
        th2 = th * th
        dup = torch.zeros(b, R, dtype=torch.bool, device=A.device)
        pair = ((RA[:, None] - RB[None]) ** 2).mean(-1).sqrt() <= th  # class-sorted radial bound, as in select
        if lower:
            pair &= torch.ones(b, R, dtype=torch.bool, device=A.device).tril(-1)
        if not bool(pair.any()):
            return dup
        ga, gb = (A**2).sum((-1, -2)), (B**2).sum((-1, -2))
        need = 0.5 * (ga[:, None] + gb[None] - n * th2)  # (b, R) overlap needed for RMSD <= th
        cs = 0.5 * (ga[:, None] + gb[None])
        rA, rB = A.norm(dim=-1), B.norm(dim=-1)  # (b, n), (R, n)
        sA, sB = (rA * rA).sum(1), (rB * rB).sum(1)
        Bf = B.permute(1, 0, 2).reshape(n, R * 3)
        M = self.inv_maps.shape[0]
        mc = max(1, budget // max(b * R * 9, 1))
        for s in range(0, M, mc):
            live = pair & ~dup
            if not bool(live.any()):
                break
            inv = self.inv_maps[s : s + mc]  # (m, n)
            m = inv.shape[0]
            # per-automorphism radial bound for every (a, m, r): one matrix product
            lb2 = (sA[:, None, None] + sB[None, None, :] - 2.0 * (rA[:, inv].reshape(b * m, n) @ rB.T).view(b, m, R)) / n
            surv = (lb2 <= th2 * (1 + 1e-9) + 1e-12).logical_and_(live[:, None, :])  # (b, m, R)
            ai, mi = torch.nonzero(surv.any(2), as_tuple=True)
            if not ai.numel():
                continue
            Av = A[ai[:, None], inv[mi]]  # (V, n, 3): A[a][inv_m[k]] for the (a, m) with a surviving pair
            V = Av.shape[0]
            G = (Av.transpose(1, 2).reshape(V * 3, n) @ Bf).view(V, 3, R, 3)  # G[v, i, r, j] = H[(a, m), r][i, j]
            sv, nv = surv[ai, mi], need[ai]  # (V, R)
            acc = (G.diagonal(dim1=1, dim2=3).sum(-1) >= nv).logical_and_(sv)  # identity rotation: a lower bound
            ub = torch.minimum(cs[ai], (3.0 * (G * G).sum((1, 3))).sqrt())
            vi, ri = torch.nonzero((ub >= nv) & sv & ~acc, as_tuple=True)
            if vi.numel():
                res = self._newton_decide(G[vi, :, ri, :], ub[vi, ri], nv[vi, ri])
                acc[vi[res], ri[res]] = True
            av, ar = torch.nonzero(acc, as_tuple=True)
            dup[ai[av], ar] = True
        return dup

    @staticmethod
    def _newton_any(H, lam, need, max_iter=60):
        """Whether the largest Horn eigenvalue reaches `need` for any of the matrices H (K, 3, 3), Newton started from
        the upper bounds `lam` (K,) as in _max_overlap. The iterates fall monotonically to the eigenvalue, so an
        element whose iterate drops below its `need` can never reach it and is dropped; the answer is yes as soon as
        one element converges (same tolerance as _max_overlap) at or above its `need`."""
        c2, c1, c0 = StructureDeduper._quartic(H)
        for _ in range(max_iter):
            l2 = lam * lam
            p = (l2 + c2) * l2 + c1 * lam + c0
            dp = (4.0 * l2 + 2.0 * c2) * lam + c1
            step = torch.where(dp.abs() > 1e-300, p / dp, torch.zeros_like(p))
            lam = lam - step
            conv = (step.abs() / lam.abs().clamp(min=1e-30)) < 1e-14
            above = lam >= need
            if bool((conv & above).any()):
                return True
            live = above & ~conv
            if not bool(live.any()):
                return False
            if int(live.sum()) < live.numel():
                lam, need, c2, c1, c0 = lam[live], need[live], c2[live], c1[live], c0[live]
        return True  # unconverged elements still at or above `need`: what _max_overlap would report as a duplicate

    def _full_any(self, A, B, th, chunk=4096):
        """Whether the exact minimum Kabsch RMSD over all automorphisms (_full) is <= th for any row of B: A (n, 3),
        B (R, n, 3), both centred. Same decision as `_full(A, B).min() <= th`, with two exact prunings before the
        eigenvalue fit: (1) per automorphism m, RMSD_m >= the RMS difference of the distances to the centroid under
        m's atom correspondence (|a - Rb| >= ||a| - |b||), one matrix product for all rows and automorphisms;
        (2) the overlap is at most min(Cauchy-Schwarz, sqrt(3) |H|_F) (s1 + s2 + s3 <= sqrt(3) |s|), then
        _newton_any from that bound. Pairs that cannot reach the threshold never get an eigenvalue fit."""
        n = A.shape[0]
        th2 = th * th
        ga, gb = (A**2).sum(), (B**2).sum((-1, -2))  # (), (R,)
        need = 0.5 * (ga + gb - n * th2)  # overlap needed for RMSD <= th, per row
        cs = 0.5 * (ga + gb)
        rA, rB = A.norm(dim=-1), B.norm(dim=-1)  # (n,), (R, n)
        rAi = rA[self.inv_maps]  # (M, n): rA[inv_m[k]]
        lb2 = ((rA * rA).sum() + (rB * rB).sum(1)[:, None] - 2.0 * (rB @ rAi.T)) / n  # (R, M) lower bound of MSD
        surv = lb2 <= th2 * (1 + 1e-9) + 1e-12
        rows = surv.any(1)
        if not bool(rows.any()):
            return False
        B, need, cs, surv = B[rows], need[rows], cs[rows], surv[rows]
        R = B.shape[0]
        Bf = B.permute(1, 0, 2).reshape(n, R * 3)  # (n, R*3): every row's coordinates side by side
        survT = surv.T  # (M, R)
        Ainv = self._a_inv(A)  # (M, n, 3)
        for s in range(0, Ainv.shape[0], chunk):
            cols = survT[s : s + chunk]
            ms = cols.any(1)
            if not bool(ms.any()):
                continue
            Ap = Ainv[s : s + chunk][ms]
            Mc = Ap.shape[0]
            # G[m, i, r, j] = H[r, m][i, j] = sum_k A[inv_m[k], i] B[r, k, j]: ONE matrix product (Mc*3, n) @ (n, R*3)
            # for the automorphisms with a surviving pair
            G = (Ap.transpose(1, 2).reshape(Mc * 3, n) @ Bf).view(Mc, 3, R, 3)
            if bool((G.diagonal(dim1=1, dim2=3).sum(-1) >= need).logical_and_(cols[ms]).any()):
                return True  # identity rotation: a lower bound of the overlap
            ub = torch.minimum(cs, (3.0 * (G * G).sum((1, 3))).sqrt())  # (Mc, R)
            ok = (ub >= need).logical_and_(cols[ms])
            mi, ri = torch.nonzero(ok, as_tuple=True)
            if mi.numel() and self._newton_any(G[mi, :, ri, :], ub[mi, ri], need[ri]):
                return True
        return False

    # --- public -----------------------------------------------------------------------------------------------------
    def any_dup(self, Xq: torch.Tensor, Xk: torch.Tensor, th: Optional[float] = None,
                factored_rows: Optional[List[List[int]]] = None) -> List[bool]:
        """For each query row (Q, N, 3): is it a duplicate of ANY reference row (K, N, 3)? The same pair decision as
        `select` makes between a candidate and a kept row (same branch per automorphism regime), without testing the
        queries against each other or the references against each other (P2R mirror completion)."""
        if th is None:
            th = DEDUP_RMSD
        Q = Xq.shape[0]
        if Q == 0 or Xk.shape[0] == 0:
            return [False] * Q
        A = Xq[:, self.heavy].to(self.device, self.dtype); A = A - A.mean(1, keepdim=True)
        B = Xk[:, self.heavy].to(self.device, self.dtype); B = B - B.mean(1, keepdim=True)
        RA, RB = self._radial(torch.nan_to_num(A)), self._radial(torch.nan_to_num(B))
        M = self.inv_maps.shape[0]
        out: List[bool] = []
        if DEDUP_FACTORED and (self.is_capped or DEDUP_MAX_DEV > 0.0):
            A_np = A.cpu().numpy() if self.is_capped else None
            B_np = B.cpu().numpy() if self.is_capped else None
            for i in range(Q):
                cand = torch.nonzero(((RB - RA[i]) ** 2).mean(1).sqrt() <= th).flatten()
                dup = False
                if cand.numel():
                    # the batched capped-map test first (the decision is the OR of both tests); the batched factored
                    # test on all candidates, or on the caller's `factored_rows`
                    if self._full_any(A[i], B[cand], th):
                        dup = DEDUP_MAX_DEV <= 0.0 or self._verify_max_dev(A[i], B[cand], th, DEDUP_MAX_DEV, cand)[0]
                    if not dup and self.is_capped:
                        cl = cand.tolist()
                        if factored_rows is not None:      # caller's candidates for the factored test (e.g. the source)
                            allowed = set(factored_rows[i]); cl = [c for c in cl if c in allowed]
                        if cl:
                            dup = bool(self._factored_match(B_np[cl], A_np[i], th=th, max_dev_th=DEDUP_MAX_DEV, stop_on_any=True)[0].any())
                out.append(bool(dup))
            return out
        if M > BLOCK_MAX_MAPS:
            for i in range(Q):
                cand = torch.nonzero(((RB - RA[i]) ** 2).mean(1).sqrt() <= th).flatten()
                out.append(bool(cand.numel() and self._full_any(A[i], B[cand], th)))
            return out
        nb = int(min(BLOCK_MAX, max(1, BLOCK_BUDGET // (M * max(B.shape[0], 1)))))
        for s0 in range(0, Q, nb):
            out += [bool(d) for d in self._pair_dup(A[s0:s0 + nb], B, RA[s0:s0 + nb], RB, th).any(1).tolist()]
        return out

    def select(self, X_all: torch.Tensor, score: torch.Tensor, k: Optional[int], th: Optional[float] = None,
               exact: bool = True, max_dev_th: Optional[float] = None) -> List[int]:
        """X_all (B, N, 3) full coordinates, score (B,) higher = better. Returns up to k distinct row indices, best first."""
        if th is None:
            th = DEDUP_RMSD
        if max_dev_th is None:
            max_dev_th = DEDUP_MAX_DEV
        self.merges = []
        X = X_all[:, self.heavy].to(self.device, self.dtype)
        X = X - X.mean(1, keepdim=True)
        sc = torch.nan_to_num(score.detach().to("cpu", torch.float64), nan=-1e30, posinf=-1e30, neginf=-1e30)
        finite = torch.isfinite(X).all(-1).all(-1).cpu()
        sc = torch.where(finite, sc, torch.full_like(sc, -1e31))
        Rr = self._radial(torch.nan_to_num(X))
        order = [i for i in torch.argsort(sc, descending=True).tolist() if bool(finite[i])]
        keep: List[int] = []

        if DEDUP_FACTORED:
            if not self.is_capped and max_dev_th <= 0.0:
                # Production path on uncapped parents: bit-identical selection guaranteed
                M, pos = self.inv_maps.shape[0], 0
                if M > BLOCK_MAX_MAPS:
                    for i in order:
                        if keep:
                            kt = torch.as_tensor(keep, device=self.device)
                            cand = kt[((Rr[kt] - Rr[i]) ** 2).mean(1).sqrt() <= th]
                            if cand.numel() and self._full_any(X[i], X[cand], th):
                                self.merges.append({"dropped": i, "kept": cand[0].item(), "rmsd": float(th), "max_dev": 0.0})
                                continue
                        keep.append(i)
                        if k is not None and len(keep) >= k:
                            break
                    return keep
                while pos < len(order) and (k is None or len(keep) < k):
                    nb = int(min(BLOCK_MAX, max(1, BLOCK_BUDGET // (M * max(len(keep), 1)))))
                    blk = order[pos : pos + nb]
                    pos += len(blk)
                    if keep:
                        bt = torch.as_tensor(blk, device=self.device)
                        kt = torch.as_tensor(keep, device=self.device)
                        dk = self._pair_dup(X[bt], X[kt], Rr[bt], Rr[kt], th).any(1).tolist()
                        for bi, d in zip(blk, dk):
                            if d:
                                self.merges.append({"dropped": bi, "kept": keep[0], "rmsd": float(th), "max_dev": 0.0})
                        blk = [i for i, d in zip(blk, dk) if not d]
                    W = None
                    if len(blk) > 1:
                        bt = torch.as_tensor(blk, device=self.device)
                        W = self._pair_dup(X[bt], X[bt], Rr[bt], Rr[bt], th, lower=True).tolist()
                    new: List[int] = []
                    for a, i in enumerate(blk):
                        if W is not None and any(W[a][r] for r in new):
                            self.merges.append({"dropped": i, "kept": keep[-1] if keep else -1, "rmsd": float(th), "max_dev": 0.0})
                            continue
                        new.append(a)
                        keep.append(i)
                        if k is not None and len(keep) >= k:
                            break
                return keep
            else:
                # Capped parents (or when max_dev_th > 0): factored symmetry + production fallback
                X_np = X.cpu().numpy() if self.is_capped else None
                for i in order:
                    if keep:
                        kt = torch.as_tensor(keep, device=self.device)
                        cand = kt[((Rr[kt] - Rr[i]) ** 2).mean(1).sqrt() <= th]
                        if cand.numel():
                            dup_found = False
                            if self.is_capped:
                                cl = cand.tolist()
                                _, hit = self._factored_match(X_np[cl], X_np[i], th=th, max_dev_th=max_dev_th, stop_on_any=True)
                                if hit is not None:
                                    p_row, r_val, md_val, perm = hit
                                    k_idx = cl[p_row]
                                    dup_found = True
                                    self.merges.append({"dropped": i, "kept": k_idx, "rmsd": r_val, "max_dev": md_val, "perm": perm.tolist()})
                                    log.info("Factored dedup merge: row %d merged into row %d (RMSD=%.4f A, max_dev=%.4f A)", i, k_idx, r_val, md_val)
                            if not dup_found:
                                if self._full_any(X[i], X[cand], th):
                                    if max_dev_th > 0.0:
                                        ok_dev, r_val, md_val, k_matched = self._verify_max_dev(X[i], X[cand], th, max_dev_th, cand)
                                        if ok_dev:
                                            dup_found = True
                                            self.merges.append({"dropped": i, "kept": k_matched, "rmsd": r_val, "max_dev": md_val})
                                    else:
                                        dup_found = True
                                        self.merges.append({"dropped": i, "kept": cand[0].item(), "rmsd": float(th), "max_dev": 0.0})
                            if dup_found:
                                continue
                    keep.append(i)
                    if k is not None and len(keep) >= k:
                        break
                return keep

        if not exact:
            for i in order:
                if keep:
                    kt = torch.as_tensor(keep, device=self.device)
                    cand = kt[((Rr[kt] - Rr[i]) ** 2).mean(1).sqrt() <= th]
                    if cand.numel() and float(self._upper(X[i], X[cand]).min()) <= th:
                        continue
                keep.append(i)
                if k is not None and len(keep) >= k:
                    break
            return keep
        M, pos = self.inv_maps.shape[0], 0
        if M > BLOCK_MAX_MAPS:
            # many automorphisms: row by row, the scan stops at the first duplicate found (_full_any)
            for i in order:
                if keep:
                    kt = torch.as_tensor(keep, device=self.device)
                    cand = kt[((Rr[kt] - Rr[i]) ** 2).mean(1).sqrt() <= th]
                    if cand.numel() and self._full_any(X[i], X[cand], th):
                        continue
                keep.append(i)
                if k is not None and len(keep) >= k:
                    break
            return keep
        # Few automorphisms: greedy best-first in blocks of rows (score order), every (block row, kept row) decision at
        # once, then the block's survivors against each other in order. The same selection as row by row; it removes
        # the per-row loop of small tensor operations, which was most of the time for many rows and few automorphisms.
        while pos < len(order) and (k is None or len(keep) < k):
            nb = int(min(BLOCK_MAX, max(1, BLOCK_BUDGET // (M * max(len(keep), 1)))))
            blk = order[pos : pos + nb]
            pos += len(blk)
            if keep:
                bt = torch.as_tensor(blk, device=self.device)
                kt = torch.as_tensor(keep, device=self.device)
                dk = self._pair_dup(X[bt], X[kt], Rr[bt], Rr[kt], th).any(1).tolist()
                blk = [i for i, d in zip(blk, dk) if not d]
            W = None
            if len(blk) > 1:
                bt = torch.as_tensor(blk, device=self.device)
                W = self._pair_dup(X[bt], X[bt], Rr[bt], Rr[bt], th, lower=True).tolist()
            new: List[int] = []
            for a, i in enumerate(blk):
                if W is not None and any(W[a][r] for r in new):
                    continue
                new.append(a)
                keep.append(i)
                if k is not None and len(keep) >= k:
                    break
        return keep


_CACHE: "OrderedDict[tuple, StructureDeduper]" = OrderedDict()



def deduper_for(mol, device=None) -> StructureDeduper:
    """StructureDeduper(mol, device), reused while the heavy-atom graph is the same (the automorphism enumeration
    is the expensive part of building one: up to MAX_AUTOMORPHISMS substructure matches). Several seeds or embeds of
    one complex share it. Keyed by element symbols and bonds, so a different complex never gets a stale one."""
    dev = torch_device(device)
    key = (str(dev), tuple(a.symbol for a in mol.atoms),
           tuple(sorted((min(b.i, b.j), max(b.i, b.j)) for b in mol.bonds)))
    dd = _CACHE.get(key)
    if dd is None:
        dd = _CACHE[key] = StructureDeduper(mol, device=dev)
        while len(_CACHE) > 4:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    return dd


def select_distinct(mol, X_all: torch.Tensor, score: torch.Tensor, k: Optional[int], device=None,
                    th: Optional[float] = None, max_dev_th: Optional[float] = None) -> List[int]:
    """`StructureDeduper(mol).select(X_all, score, k, th=th, max_dev_th=max_dev_th)`; for a CUDA solve (`device`)
    the RMSD work runs on that GPU (see DEDUP_DEVICE). Any GPU failure (out of memory, a solver error) falls back to the CPU."""
    dev = torch_device(device)
    if dev.type == "cuda" and DEDUP_DEVICE != "cpu":
        try:
            return deduper_for(mol, dev).select(X_all, score, k, th=th, max_dev_th=max_dev_th)
        except RuntimeError as exc:
            log.warning("output dedup on %s failed (%s); repeating it on the CPU", dev, str(exc).splitlines()[0][:200])
            torch.cuda.empty_cache()
    return deduper_for(mol).select(X_all, score, k, th=th, max_dev_th=max_dev_th)
