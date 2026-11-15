"""
nci_runtime.py
==============
Runtime tensor construction and PyTorch evaluation for the NCI energy term.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_MOD = None


def _m():
    global _MOD
    if _MOD is None:
        from jacobik.fit import nci_model

        _MOD = nci_model
    return _MOD


# ----------------------------------------------------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------------------------------------------------
def build_runtime(mol, tables: Dict[tuple, dict], device, types=None):
    """Per type: C^2 radial-Chebyshev surface tensors (C, M) + legacy table tensor (T), candidate rows, and indices."""
    import torch
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors

    if not tables:
        return None
    atoms = list(mol.atoms)
    pos = {a.idx: i for i, a in enumerate(atoms)}
    m = _m(); cands = m.candidates(mol)
    rt = {}
    enabled = m.ACTIVE_TYPES if types is None else tuple(types)
    if any(t not in m.TYPES for t in enabled):
        raise ValueError(f"unknown NCI types: {enabled}")
    nb = None
    bound = None
    for t in m.TYPES:
        if t not in enabled:
            continue
        rows, f_rows, row_of, keep, cand_row = [], [], {}, [], []
        for k, c in enumerate(cands[t]):
            e = next(((key, tables[key]) for key in c[-1] if key in tables), None)
            if e is None:
                continue
            if e[0] not in row_of:
                row_of[e[0]] = len(rows)
                s_arr = np.asarray(e[1]["s"], dtype=np.float64)
                rows.append(s_arr)
                if "f" in e[1]:
                    f_rows.append(np.asarray(e[1]["f"], dtype=np.float64))
                else:
                    f_rows.append(m._untaper_empirical_table(t, s_arr))
            keep.append(k)
            cand_row.append(row_of[e[0]])
        if not keep:
            continue
        r_lo, dr, nr, a_lo, da, na = m._grid(t)
        c2_params = m.build_c2_surface_params(t, np.stack(f_rows, axis=0), s_clip=m.S_CLIP)
        idx_t = m.index_tensors(cands, pos, device, keep={**{x: [] for x in m.TYPES}, t: keep})[t]
        kept_cs = [cands[t][k] for k in keep]
        entry = dict(
            T=torch.tensor(np.array(rows, dtype=np.float32), device=device),
            C=torch.tensor(c2_params["C"], dtype=torch.float32, device=device),
            M=torch.tensor(c2_params["M"], dtype=torch.float32, device=device),
            row=torch.tensor(cand_row, dtype=torch.long, device=device),
            idx=idx_t,
            grid=(r_lo, dr, nr, a_lo, da, na),
            win=m.SPEC[t]["win"],
            s_clip=float(m.S_CLIP),
            scale=float(m.TYPE_SCALES.get(t, 1.0)),
            n=len(keep),
        )
        if t == "pp":
            if bound is None:
                bound = set(donor_indices_of(mol))
            entry["valid_mask"] = torch.tensor(
                [0.0 if (set(cand[0]) <= bound and set(cand[1]) <= bound) else 1.0 for cand in kept_cs],
                dtype=torch.float32,
                device=device,
            )[None, :]
        elif t == "chp":
            if nb is None:
                nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
            h_idx = idx_t["h"].tolist()
            entry["parent_idx"] = torch.tensor(
                [pos[nb[atoms[int(h)].idx][0]] for h in h_idx], dtype=torch.long, device=device
            )
            p_list = entry["parent_idx"].tolist()
            r_list = idx_t["ring"].tolist()
            groups = sorted(set(zip(p_list, r_list)))
            entry["groups"] = groups
            entry["group_indices"] = [
                torch.tensor(
                    [i for i, (p, r) in enumerate(zip(p_list, r_list)) if p == p_id and r == r_id],
                    dtype=torch.long,
                    device=device,
                )
                for p_id, r_id in groups
            ]
        rt[t] = entry
    return rt or None


def _bilinear(T, row, r, a, grid):
    """Legacy piecewise-bilinear evaluator retained for backward-compatible unit tests."""
    import torch

    r_lo, dr, nr, a_lo, da, na = grid
    u = ((r - r_lo) / dr - 0.5).clamp(0.0, nr - 1.0)
    v = ((a - a_lo) / da - 0.5).clamp(0.0, na - 1.0)
    i0 = torch.floor(u.detach()).long().clamp(0, nr - 2)
    j0 = torch.floor(v.detach()).long().clamp(0, na - 2)
    fu, fv = u - i0.to(u.dtype), v - j0.to(v.dtype)
    rw = row[None].expand_as(i0)
    g = lambda i, j: T[rw, i, j]
    s = (g(i0, j0) * (1 - fu) * (1 - fv) + g(i0 + 1, j0) * fu * (1 - fv) + g(i0, j0 + 1) * (1 - fu) * fv
         + g(i0 + 1, j0 + 1) * fu * fv)
    return torch.where(r >= r_lo + dr * nr, torch.zeros_like(s), s)


def evaluate_c2_from_rz_torch(C, M, row, r, z, grid, win, s_clip: Optional[float] = None):
    if s_clip is None:
        s_clip = _m().S_CLIP
    """Evaluate C^2 radial-Chebyshev surface s(r, z) at (r, z) of shape (B, M_cand)."""
    import torch

    r_lo, dr, nr, _, _, na = grid
    win_lo, win_hi = win
    C = C.to(dtype=r.dtype)
    M = M.to(dtype=r.dtype)

    T_list = [torch.ones_like(z), z]
    for _ in range(2, na):
        T_list.append(2.0 * z * T_list[-1] - T_list[-2])
    Cheb = torch.stack(T_list[:na], dim=-1)

    u = (r - r_lo) / dr - 0.5
    u_clamped = u.clamp(0.0, float(nr - 1))
    i0 = torch.floor(u_clamped.detach()).long().clamp(0, nr - 2)
    t = (u_clamped - i0.to(u.dtype)).clamp(0.0, 1.0)

    rw = row[None, :].expand_as(i0)
    y0 = C[rw, i0]
    y1 = C[rw, i0 + 1]
    m0 = M[rw, i0]
    m1 = M[rw, i0 + 1]

    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    f_cubic = h00[..., None] * y0 + h10[..., None] * (dr * m0) + h01[..., None] * y1 + h11[..., None] * (dr * m1)

    t4 = t2 * t2
    f_quartic = y0 + (y1 - y0) * (4.0 * t3 - 3.0 * t4)[..., None] - (dr * m1) * (t3 - t4)[..., None]

    f_modes = torch.where((i0 == 0)[..., None], f_quartic, f_cubic)
    f_modes = torch.where((u <= 0.0)[..., None], C[rw, 0], f_modes)

    f_val = (f_modes * Cheb).sum(dim=-1)
    f_sat = s_clip * torch.tanh(f_val / s_clip)
    taper = _m().quintic_taper_torch(r, win_lo, win_hi)
    return f_sat * taper


def _eval_pp_c5(e: dict, X):
    """Evaluate C5 stacking term with normal/slip decomposition and co-bound haptic face exclusion."""
    import torch

    m = _m(); c, n = m._ring_frame(X, e["idx"]["rings"])
    ci, cj = c[:, e["idx"]["ri"]], c[:, e["idx"]["rj"]]
    ni, nj = n[:, e["idx"]["ri"]], n[:, e["idx"]["rj"]]
    v = ci - cj
    r = v.norm(dim=-1)
    dot = (ni * nj).sum(-1)
    z = (2.0 * (dot * dot) - 1.0).clamp(-1.0, 1.0)
    s_base = evaluate_c2_from_rz_torch(e["C"], e["M"], e["row"], r, z, e["grid"], e["win"], e.get("s_clip", m.S_CLIP))

    proj_i = (v * ni).sum(-1)
    proj_j = (v * nj).sum(-1)
    d_perp2 = 0.5 * (proj_i * proj_i + proj_j * proj_j)
    d_par2 = torch.clamp(r * r - d_perp2, min=0.0)
    slip_window = torch.exp(-0.5 * ((d_par2 - 2.5) / 3.5) ** 2)
    valid_mask = e["valid_mask"].to(dtype=X.dtype) if "valid_mask" in e else 1.0
    eps = 1e-3
    s_pos = 0.5 * (s_base + torch.sqrt(s_base * s_base + eps * eps) - eps)
    s_neg = s_base - s_pos
    return ((s_neg + s_pos * slip_window) * valid_mask).sum(dim=-1)


def _eval_chp_c3_c4(e: dict, X, tau: float = 0.15):
    """Evaluate C3+C4 CH-pi term with C-H directionality gate and per-(carbon, ring) sibling aggregation."""
    import torch

    m = _m(); r, z = m.features_cos_torch("chp", X, e["idx"])
    s_raw = evaluate_c2_from_rz_torch(e["C"], e["M"], e["row"], r, z, e["grid"], e["win"], e.get("s_clip", m.S_CLIP))
    if "parent_idx" not in e:
        return s_raw.sum(dim=-1)

    h_idx = e["idx"]["h"]
    ring_idx = e["idx"]["ring"]
    parent_idx = e["parent_idx"]
    c_ring, _ = m._ring_frame(X, e["idx"]["rings"])
    centers = c_ring[:, ring_idx]
    H_pos = X[:, h_idx]
    C_pos = X[:, parent_idx]
    u = C_pos - H_pos
    v = centers - H_pos
    cos_dha = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-12)
    gate = torch.clamp(-cos_dha, min=0.0) ** 3
    eps = 1e-3
    s_pos = 0.5 * (s_raw + torch.sqrt(s_raw * s_raw + eps * eps) - eps)
    s_neg = s_raw - s_pos
    s_dir = s_neg + s_pos * gate

    s_pos2 = 0.5 * (s_dir + torch.sqrt(s_dir * s_dir + eps * eps) - eps)
    s_neg2 = s_dir - s_pos2
    total = X.new_zeros(X.shape[0])
    group_indices = e.get("group_indices")
    if group_indices is not None:
        for idx in group_indices:
            sp = s_pos2[:, idx]
            sn = s_neg2[:, idx]
            agg_pos = tau * torch.log1p(torch.expm1(sp / tau).sum(dim=-1))
            total = total + agg_pos + sn.sum(dim=-1)
    else:
        for p_id, r_id in e["groups"]:
            mask = (parent_idx == p_id) & (ring_idx == r_id)
            sp = s_pos2[:, mask]
            sn = s_neg2[:, mask]
            agg_pos = tau * torch.log1p(torch.expm1(sp / tau).sum(dim=-1))
            total = total + agg_pos + sn.sum(dim=-1)
    return total


def evaluate(rt, X):
    """(B,) sum over candidate interactions of calibrated C^2 NCI score (positive = favoured)."""
    tot = X.new_zeros(X.shape[0])
    for t, e in rt.items():
        scale = float(e.get("scale", 1.0))
        if scale == 0.0:
            continue
        if "C" not in e or "M" not in e:
            r, a = _m().features(t, X, e["idx"])
            tot = tot + scale * _bilinear(e["T"], e["row"], r, a, e["grid"]).sum(-1)
            continue
        if t == "hb":
            r, z = _m().features_cos_torch("hb", X, e["idx"])
            s = evaluate_c2_from_rz_torch(e["C"], e["M"], e["row"], r, z, e["grid"], e["win"], e.get("s_clip", _m().S_CLIP)).sum(-1)
        elif t == "pp":
            s = _eval_pp_c5(e, X)
        else:
            s = _eval_chp_c3_c4(e, X)
        tot = tot + scale * s
    return tot
