"""Bench-only speedups for Campaign I screen (capped parents under factored dedup / out_per_class).
Active only when dedup_factored is on; arm C (non-factored path) is unchanged. Not production code."""
import numpy as np, torch
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from jacobik.builder import engine as E
from jacobik.ik import dedup as dd


def install():
    orig_assemble = E.ComplexBuilder._assemble_conformers
    # The screen uses engine assembly, rather than multiseed.assemble_batched.
    # Apply master's >=1000-map GPU policy here too, preserving CPU fallback.
    orig_select_distinct = E.select_distinct
    def screen_select(mol, X_all, score, k, device=None, th=None):
        cpu_dd = dd.deduper_for(mol)
        if X_all.is_cuda and cpu_dd.inv_maps.shape[0] >= 1000:
            try:
                return dd.deduper_for(mol, X_all.device).select(X_all, score, k, th=th)
            except RuntimeError:
                torch.cuda.empty_cache()
        return orig_select_distinct(mol, X_all, score, k, device, th)
    E.select_distinct = screen_select
    def fast_factored_sym_upper(self, A: np.ndarray, B: np.ndarray, th: float = 0.5, max_dev_th: float = 0.0):
        n = A.shape[0]
        n_core = len(self.core)

        Ac = A[self.core] - A[self.core].mean(0)
        B_core = B[self.core_perms_arr]
        Bc = B_core - B_core.mean(1, keepdims=True)
        H = np.einsum("kni,nj->kij", Bc, Ac)
        sv = np.linalg.svd(H, compute_uv=False)
        d = np.sign(np.linalg.det(H))
        s = sv[:, 0] + sv[:, 1] + d * sv[:, 2]
        msd = ((Ac ** 2).sum() + (Bc ** 2).sum((-1, -2)) - 2.0 * s) / max(1, n_core)
        core_rmsd = np.sqrt(np.clip(msd, 0.0, None))

        max_core_th = th * np.sqrt(n / max(1, n_core)) + 1e-6
        valid = np.flatnonzero(core_rmsd <= max_core_th)
        if not len(valid):
            return False, float(core_rmsd.min()), np.inf, None
        order = valid[np.argsort(core_rmsd[valid])[:6]]

        if not hasattr(self, "_g1"):
            g1 = [g[0] for g in self.groups if len(g) == 1]
            self._g1 = np.array(g1, dtype=int) if g1 else np.empty(0, dtype=int)
            g2 = [g for g in self.groups if len(g) == 2]
            self._g2 = np.array(g2, dtype=int) if g2 else np.empty((0, 2), dtype=int)
            self._g_large = [g for g in self.groups if len(g) > 2]

        best = (np.inf, np.inf, False, None)
        A_mean = A.mean(0)
        A_core_mean = A[self.core].mean(0)
        for idx in order:
            perm_core = self.core_perms_arr[idx]
            _, _, R = dd.kabsch_fit_np(A[self.core], B[perm_core])
            Bt = (B - B[perm_core].mean(0)) @ R + A_core_mean
            mp = np.arange(n)
            for _ in range(12):
                new = np.empty(n, dtype=int)
                if len(self._g1):
                    new[self._g1] = self._g1
                if len(self._g2):
                    g2 = self._g2
                    Ag2 = A[g2]
                    Btg2 = Bt[g2]
                    d_id = ((Ag2[:, 0] - Btg2[:, 0]) ** 2).sum(-1) + ((Ag2[:, 1] - Btg2[:, 1]) ** 2).sum(-1)
                    d_sw = ((Ag2[:, 0] - Btg2[:, 1]) ** 2).sum(-1) + ((Ag2[:, 1] - Btg2[:, 0]) ** 2).sum(-1)
                    swap = d_sw < d_id
                    new[g2[:, 0]] = np.where(swap, g2[:, 1], g2[:, 0])
                    new[g2[:, 1]] = np.where(swap, g2[:, 0], g2[:, 1])
                for gidx in self._g_large:
                    C = ((A[gidx][:, None, :] - Bt[gidx][None, :, :]) ** 2).sum(-1)
                    r_, c_ = linear_sum_assignment(C)
                    new[gidx[r_]] = gidx[c_]
                r, md, R = dd.kabsch_fit_np(A, B[new])
                Bt = (B - B[new].mean(0)) @ R + A_mean
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


    def fast_select(self, X_all: torch.Tensor, score: torch.Tensor, k=None, th=None, exact: bool = True,
                    class_ids=None, k_per_class=None):
        if th is None:
            th = dd.DEDUP_RMSD
        max_dev_th = dd.DEDUP_MAX_DEV
        self.merges = []
        X = X_all[:, self.heavy].to(self.device, self.dtype)
        X = X - X.mean(1, keepdim=True)
        sc = torch.nan_to_num(score.detach().to("cpu", torch.float64), nan=-1e30, posinf=-1e30, neginf=-1e30)
        finite = (torch.isfinite(X).all(-1).all(-1).cpu()) & torch.isfinite(score.detach().cpu())
        sc = torch.where(finite, sc, torch.full_like(sc, -1e31))
        Rr = self._radial(torch.nan_to_num(X))
        order = [i for i in torch.argsort(sc, descending=True).tolist() if bool(finite[i])]
        keep = []
        class_counts = defaultdict(int)

        if dd.DEDUP_FACTORED:
            if not self.is_capped and max_dev_th <= 0.0 and class_ids is None:
                M, pos = self.inv_maps.shape[0], 0
                if M > dd.BLOCK_MAX_MAPS:
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
                    nb = int(min(dd.BLOCK_MAX, max(1, dd.BLOCK_BUDGET // (M * max(len(keep), 1)))))
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
                    new = []
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
                X_np_all = X.cpu().numpy() if self.is_capped else None
                for i in order:
                    if class_ids is not None and k_per_class is not None:
                        if class_counts[class_ids[i]] >= k_per_class:
                            continue
                    if keep:
                        kt = torch.as_tensor(keep, device=self.device)
                        cand = kt[((Rr[kt] - Rr[i]) ** 2).mean(1).sqrt() <= th]
                        if cand.numel():
                            dup_found = False
                            # Check fast vectorized _full_any FIRST when max_dev_th <= 0
                            if max_dev_th <= 0.0 and self._full_any(X[i], X[cand], th):
                                dup_found = True
                                self.merges.append({"dropped": i, "kept": cand[0].item(), "rmsd": float(th), "max_dev": 0.0})
                            elif max_dev_th > 0.0 and self._full_any(X[i], X[cand], th):
                                ok_dev, r_val, md_val, k_matched = self._verify_max_dev(X[i], X[cand], th, max_dev_th, cand)
                                if ok_dev:
                                    dup_found = True
                                    self.merges.append({"dropped": i, "kept": k_matched, "rmsd": r_val, "max_dev": md_val})
                            if not dup_found and self.is_capped:
                                X_np_i = X_np_all[i]
                                for k_idx in cand.tolist():
                                    is_dup, r_val, md_val, _ = self.factored_sym_upper(X_np_all[k_idx], X_np_i, th=th, max_dev_th=max_dev_th)
                                    if is_dup:
                                        dup_found = True
                                        self.merges.append({"dropped": i, "kept": k_idx, "rmsd": r_val, "max_dev": md_val})
                                        break
                            if dup_found:
                                continue
                    keep.append(i)
                    if class_ids is not None:
                        class_counts[class_ids[i]] += 1
                    if k is not None and len(keep) >= k:
                        break
                return keep

        # Standard non-factored path unchanged
        M, pos = self.inv_maps.shape[0], 0
        if M > dd.BLOCK_MAX_MAPS:
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
        while pos < len(order) and (k is None or len(keep) < k):
            nb = int(min(dd.BLOCK_MAX, max(1, dd.BLOCK_BUDGET // (M * max(len(keep), 1)))))
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
            new = []
            for a, i in enumerate(blk):
                if W is not None and any(W[a][r] for r in new):
                    continue
                new.append(a)
                keep.append(i)
                if k is not None and len(keep) >= k:
                    break
        return keep


    # Also in select_distinct, pass class_ids / k_per_class when do_out_per_class is active
    def fast_assemble_conformers(self, ctx, fks, X_full, thetas_final, scores, converged, beam_width, candidate_idx_per_b):
        do_opc = getattr(self, "out_per_class", None)
        if do_opc is None:
            do_opc = E.OUT_PER_CLASS
        if not do_opc:
            return orig_assemble(self, ctx, fks, X_full, thetas_final, scores, converged, beam_width, candidate_idx_per_b)
        # Pre-mask scores outside global window or per-assignment class window so deduper.select never wastes time on ineligible rows
        score_np = scores.detach().cpu().numpy()
        zn = float(getattr(ctx, "score_norm", 1.0) or 1.0)
        if zn <= 0:
            zn = 1.0
        in_win = E.output_window(score_np, float(getattr(ctx, "score_norm", 0.0) or 0.0))
        c_win = (E.OUT_PER_CLASS_WINDOW or 35.0) / zn
        k_lim = E.OUT_PER_CLASS_K if E.OUT_PER_CLASS_K is not None else 8
        cands = np.asarray(candidate_idx_per_b, dtype=int)
        elig = in_win.copy()
        for cid in np.unique(cands):
            mask_c = (cands == cid) & in_win & np.isfinite(score_np)
            if mask_c.any():
                c_best = float(score_np[mask_c].max())
                elig[cands == cid] &= (score_np[cands == cid] >= c_best - c_win)
            else:
                elig[cands == cid] = False
        masked_scores = scores.clone()
        masked_scores[~torch.as_tensor(elig, dtype=torch.bool, device=scores.device)] = float("nan")
        # Also cap per-assignment class at k_lim if no stereo elements can split an assignment class within one embed
        from jacobik.ik import stereo as _st
        els = _st.perceive_complex_stereo(self, scope=E.STEREO_SCOPE)
        orig_select_distinct = E.select_distinct
        if not els:
            E.select_distinct = lambda mol, X_all, sc, nk, dev=None, th=None: dd.deduper_for(mol).select(
                X_all, sc, nk, th=th, class_ids=cands, k_per_class=k_lim
            )
        try:
            return orig_assemble(self, ctx, fks, X_full, thetas_final, masked_scores, converged, beam_width, candidate_idx_per_b)
        finally:
            E.select_distinct = orig_select_distinct



    dd.StructureDeduper.factored_sym_upper = fast_factored_sym_upper
    dd.StructureDeduper.select = fast_select
    E.ComplexBuilder._assemble_conformers = fast_assemble_conformers
