"""Crystal-positioned statistics: a DIAGNOSTIC oracle for the data-vs-solver split (plan Part B, 2026-10-16).

Question: how much of a complex's energy error comes from WHERE the statistics put their modes (data), and how much
remains when every mode sits exactly on the crystal value (solver: search, optimiser, frozen internals, Q6)?

Per term instance (Q1 per donor, Q2 per donor pair, Q5 per torsion, S_orient per kappa donor), the whole fitted curve
is TRANSLATED so that its component nearest the crystal value has its mean on the crystal value. Widths, weights,
multi-modality, folds and walls are unchanged (the Q1 wall moves with the curve). Not moved: Q6 (walls: no mode),
haptic S_orient (substituent H not mapped), and any instance involving a hydrogen that could not be mapped (H are mapped per
heavy parent by position after superposition). Both the solve scorer and the per-ligand beam scorer are patched, so beam seeds follow the moved
Q5 modes. Never part of the solver.

    with crystal_targets(*crystal_in_engine_order(case, mol, Xp)):   # crystal in the solver's atom order
        result = builder.solve(...)
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch


def crystal_in_engine_order(case, mol, Xp):
    """Crystal coordinates in the solver's atom order: heavy atoms through the SMILES->xyz map, relabelled by the
    graph automorphism that best matches the conformer Xp (coord_audit.symmetric_remap); H copied from Xp."""
    from coord_audit import symmetric_remap

    m = case.tmqm_mol
    rb = {a.idx: i for i, a in enumerate(m.atoms)}
    amap = {0: rb[m.metal_idx]}
    ys = [a.symbol for a in mol.atoms]
    for lig, lg in zip(sorted(mol.ligands, key=lambda l: l.ligand_id), case.ligand_groups):
        for rd, x1 in lg["rdkit_to_xyz"].items():
            if rd < len(lig.atom_indices) and ys[lig.atom_indices[rd] - 1] != "H":
                amap[lig.atom_indices[rd] - 1] = rb[x1]
    Xc = np.array(Xp, dtype=np.float64).copy()
    for k, v in amap.items():
        Xc[k] = case.ref_X[v]
    Xc = symmetric_remap(mol, Xp, Xc)
    return _map_hydrogens(case, ys, Xp, Xc)


def _map_hydrogens(case, ys, Xp, Xc):
    """Crystal H onto the solver's H: per heavy parent, Hungarian on positions after superimposing the crystal heavy
    atoms on the conformer. Returns (Xc with mapped H, boolean mask of H that could be mapped)."""
    from scipy.optimize import linear_sum_assignment

    ref, rsy = np.asarray(case.ref_X, float), list(case.ref_symbols)
    n = len(ys)
    heavy = [i for i in range(n) if ys[i] != "H"]
    eng_h = [i for i in range(n) if ys[i] == "H"]
    c_heavy = [j for j in range(len(rsy)) if rsy[j] != "H"]
    c_h = [j for j in range(len(rsy)) if rsy[j] == "H"]
    # crystal row of each solver heavy atom (Xc holds exact crystal coordinates)
    cidx = {k: c_heavy[int(np.argmin(((ref[c_heavy] - Xc[k]) ** 2).sum(-1)))] for k in heavy}
    A = Xc[heavy] - Xc[heavy].mean(0)
    B = Xp[heavy] - Xp[heavy].mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    refA = (ref - Xc[heavy].mean(0)) @ R + Xp[heavy].mean(0)  # crystal in the conformer's frame (H assignment only)
    par_e = {h: heavy[int(np.argmin(((Xp[heavy] - Xp[h]) ** 2).sum(-1)))] for h in eng_h}
    par_c = {h: c_heavy[int(np.argmin(((ref[c_heavy] - ref[h]) ** 2).sum(-1)))] for h in c_h}
    ok = np.zeros(n, bool)
    ok[heavy] = True
    for k in heavy:
        eh = [h for h in eng_h if par_e[h] == k]
        ch = [h for h in c_h if par_c[h] == cidx[k]]
        if not eh or len(eh) != len(ch):
            continue
        C = ((Xp[eh][:, None] - refA[ch][None]) ** 2).sum(-1)
        r, c = linear_sum_assignment(C)
        for a_, b_ in zip(r, c):
            Xc[eh[a_]] = ref[ch[b_]]
            ok[eh[a_]] = True
    return Xc, ok


def _nearest_shift(obs, mu, var, lw):
    """obs (n,), mu/var/lw (n, K) -> delta (n,) moving the most responsible component's mean onto obs."""
    k = torch.argmax(lw - 0.5 * (obs[:, None] - mu) ** 2 / var, dim=-1)
    return obs - mu.gather(1, k[:, None])[:, 0]


def _so_features(X, M, g):
    D = X[:, g["donor"], :]
    uM = torch.nn.functional.normalize(M[:, None, :] - D, dim=-1)
    us = torch.nn.functional.normalize(X[:, g["subs"], :] - D[:, :, None, :], dim=-1)
    cm = (uM[:, :, None, :] * us).sum(-1)
    cm_s, order = torch.sort(cm, dim=-1)
    us = torch.gather(us, 2, order[..., None].expand_as(us))
    feats = [cm_s]
    k = g["k"]
    if k > 1:
        feats.append(torch.stack([(us[:, :, i, :] * us[:, :, j, :]).sum(-1) for i in range(k) for j in range(i + 1, k)],
                                 dim=-1))
    return torch.cat(feats, dim=-1)[0]  # (n, dim)


def recenter(sc, Xc_np, is_h):
    """Translate the scorer's fitted curves onto the crystal values (in place). `is_h`: atoms whose crystal position is
    unknown (unmapped H); instances touching them keep their table. Returns {term: n instances moved}."""
    ge = sc._ge
    dev = sc._q6_mu.device if hasattr(sc, "_q6_mu") else torch.device("cpu")
    Xc = torch.tensor(Xc_np, dtype=torch.float32, device=dev)[None]
    H = torch.tensor(is_h, dtype=torch.bool, device=dev)
    moved = {}
    with torch.no_grad():
        # Q1 (donor groups: heavy atoms or haptic centroids)
        if getattr(sc, "_n_q1", 0) > 0:
            d = ge.q1_distances(Xc)[0]
            q1c = getattr(sc, "_q1_cand", None)
            if q1c is not None:
                mu, ev, lw = q1c
                var = sc._q1_cand_var.clamp(min=1e-6)
                new = mu.clone()
                hi, wsd = sc._q1_cand_wall
                hi = hi.clone()
                for c in range(mu.shape[0]):
                    delta = _nearest_shift(d, mu[c], var[c], lw[c])
                    new[c] = mu[c] + delta[:, None]
                    hi[c] = hi[c] + delta
                sc._q1_cand = (new, ev, lw)
                sc._q1_cand_wall = (hi, wsd)
            if getattr(sc, "_q1_means", None) is not None and sc._q1_means.shape[0] == d.shape[0]:
                delta = _nearest_shift(d, sc._q1_means, sc._q1_eff_vars, sc._q1_log_w)
                sc._q1_means = sc._q1_means + delta[:, None]
                if getattr(sc, "_q1_wall_hi", None) is not None:
                    sc._q1_wall_hi = sc._q1_wall_hi + delta
            moved["Q1"] = int(d.shape[0])
        # Q2 (cos of D-M-D)
        if getattr(sc, "_n_q2", 0) > 0:
            cz = torch.cos(ge.q2_dmd_angles(Xc)[0])
            q2c = getattr(sc, "_q2_cand", None)
            if q2c is not None:
                m2 = q2c[0].clone()
                var2 = sc._q2_cand_var.clamp(min=1e-8) if getattr(sc, "_q2_cand_var", None) is not None else q2c[1]
                for c in range(m2.shape[0]):
                    m2[c] = m2[c] + _nearest_shift(cz, q2c[0][c], var2[c], q2c[2][c])[:, None]
                sc._q2_cand = (m2,) + tuple(q2c[1:])
            if getattr(sc, "_q2_means", None) is not None and sc._q2_means.shape[0] == cz.shape[0]:
                sc._q2_means = sc._q2_means + _nearest_shift(cz, sc._q2_means, sc._q2_eff_vars, sc._q2_log_w)[:, None]
            moved["Q2"] = int(cz.shape[0])
        # Q5 (torsions; instances with an H flank keep their table)
        if ge.n_q5 > 0 and sc._q5_means.shape[0] > 0:
            t = ge.q5_torsions(Xc)[0]
            ok = ~(H[ge._q5_i] | H[ge._q5_j] | H[ge._q5_k] | H[ge._q5_l])
            mu, ka, w = sc._q5_means, sc._q5_kappas, sc._q5_weights
            score = torch.log(w.clamp(min=1e-30)) + ka * torch.cos(t[:, None] - mu)
            k = torch.argmax(score, dim=-1)
            mk = mu.gather(1, k[:, None])[:, 0]
            delta = torch.atan2(torch.sin(t - mk), torch.cos(t - mk))
            delta = torch.where(ok, delta, torch.zeros_like(delta))
            sc._q5_means = mu + delta[:, None]
            moved["Q5"] = int(ok.sum())
        # S_orient (kappa donors whose substituents are all heavy)
        n_so = 0
        M = Xc[:, ge.metal_pos_idx, :]
        for g in getattr(sc, "_so", []):
            f = _so_features(Xc, M, g)  # (n, dim)
            okg = ~(H[g["subs"]].any(-1))
            diff = f[:, None, :] - g["mu"]  # (n, K, dim)
            maha = torch.einsum("nki,nkij,nkj->nk", diff, g["prec"], diff)
            k = torch.argmax(g["logw"] - 0.5 * (maha + g["logdet"]), dim=-1)
            mk = g["mu"].gather(1, k[:, None, None].expand(-1, 1, g["mu"].shape[-1]))[:, 0]
            delta = torch.where(okg[:, None], f - mk, torch.zeros_like(f))
            g["mu"] = g["mu"] + delta[:, None, :]
            n_so += int(okg.sum())
        if getattr(sc, "_so", None):
            moved["SO"] = n_so
    return moved


@contextlib.contextmanager
def crystal_targets(Xc_np, mapped, log=None):
    """Patch jacobik.ik.scorer.Scorer so every scorer built inside the block is recentred on Xc_np (atoms with
    mapped=False are treated as unknown)."""
    import jacobik.ik.scorer as SC

    orig = SC.Scorer.__init__

    def init(self, mol, *a, **kw):
        orig(self, mol, *a, **kw)
        mv = recenter(self, Xc_np, ~np.asarray(mapped, bool))
        if log:
            log(f"    crystal targets: {mv}")

    SC.Scorer.__init__ = init
    try:
        yield
    finally:
        SC.Scorer.__init__ = orig


@contextlib.contextmanager
def reach_target(Xc_np, mapped, sigma=0.1, keep_q6=True):
    """DIAGNOSTIC (rung R4, "closest reachable pose"): every scorer built inside the block scores a structure by its
    closeness to the crystal, -0.5 * mean over mapped atoms of |X - R Xc|^2 / sigma^2, with R the best proper rotation
    about the metal for each row (frame-free), so the unmodified solver (starts, beam, optimiser, shape / symmetry
    constraints) drives every row as close to the crystal as its degrees of freedom allow. The bend prior and the
    ring-closure penalty stay as they are. pools.py records the production ranking terms of the resulting poses
    afterwards (outside the block), which gives the decoy test: production score of the reachable crystal pose vs the
    production #1."""
    import jacobik.ik.scorer as SC

    orig = SC.Scorer._score_components
    w = torch.tensor(np.asarray(mapped, float), dtype=torch.float32)

    def comps(self, X, q3_mode_override=None, donor_pos=None, row_groups=None):
        Y = torch.as_tensor(self._reach_Xc, dtype=X.dtype, device=X.device)
        ww = w.to(X.device)
        # superimpose the crystal on every row (rotation about the metal): the score must not depend on the frame
        M = X[:, self._ge.metal_pos_idx:self._ge.metal_pos_idx + 1, :]
        A = X - M
        Bc = (Y - Y[self._ge.metal_pos_idx])[None].expand_as(A)
        with torch.no_grad():  # optimal rotation held fixed in the gradient (envelope theorem: same gradient)
            H = torch.einsum("bn,bni,bnj->bij", ww[None].expand(A.shape[0], -1), Bc, A.detach())
            U, S, Vh = torch.linalg.svd(H)
            d = torch.sign(torch.det(U @ Vh))
            D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
            R = U @ D @ Vh
        Bt = Bc @ R
        sq = (((A - Bt) ** 2).sum(-1) * ww).sum(-1) / ww.sum()
        z = X.new_zeros(X.shape[0])
        q6 = orig(self, X, q3_mode_override=q3_mode_override, donor_pos=donor_pos, row_groups=row_groups)["Q6"]             if keep_q6 else z  # keep the production steric term: the closest pose WITHOUT clashes
        return {"Q1": -0.5 * sq / sigma ** 2, "Q2": z, "Q3": z, "Q4": z, "Q5": z, "Q6": q6}

    def init_hook(self, mol, *a, **kw):
        orig_init(self, mol, *a, **kw)
        self._reach_Xc = np.asarray(Xc_np, float) - np.asarray(Xc_np, float)[self._ge.metal_pos_idx]

    orig_init = SC.Scorer.__init__
    SC.Scorer.__init__ = init_hook
    SC.Scorer._score_components = comps
    try:
        yield
    finally:
        SC.Scorer._score_components = orig
        SC.Scorer.__init__ = orig_init
