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

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from jacobik.core import profile as _profile
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


def rep_pair_params(e1: str, e2: str, scale: float = 1.0) -> Optional[Tuple[float, float, float, float]]:
    """(A kJ/mol, b 1/A, r_m A, r_lin A) of the physical repulsion of an element pair, or None without parameters.
    ``scale`` moves the onset to scale * r_m (A and b unchanged); r_lin is recomputed for the moved onset."""
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
    import torch

    A, b, rm, rl = (torch.as_tensor(v, dtype=d.dtype, device=d.device) for v in (A, b, rm, rl))
    em = torch.exp(-b * rm)
    u = lambda r: A * (torch.exp(-b * r) - em + b * em * (r - rm)) / RT_KJ
    slope = A * b * (torch.exp(-b * rl) - em) / RT_KJ
    lin = u(rl) + slope * (rl - d)
    out = torch.where(d < rl, lin, u(torch.maximum(d, rl)))
    return torch.where(d < rm, out, torch.zeros_like(out))


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


def build_runtime(mol, ge, tables: Dict[tuple, dict], wall_mu, wall_sigma, device):
    """Tensors for the purely repulsive Q6 scorer."""
    import torch

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
    T = torch.tensor(np.array(rows, dtype=np.float32), device=device)
    lt = lambda v: torch.tensor(v, dtype=torch.long, device=device)
    has = torch.tensor([x >= 0 for x in t_tot], dtype=torch.bool, device=device)
    rp = np.array([r if r is not None else (0.0, 1.0, 0.0, 0.0) for r in rep], dtype=np.float64)
    ft = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    return dict(
        T=T, tot=lt([max(x, 0) for x in t_tot]), has=has, r_lo=R_LO, dr=DR_TAB, G=T.shape[1],
        enabled=torch.tensor(enabled, dtype=torch.bool, device=device), rep_weight=ft(rep_weight),
        mu=wall_mu, sigma=wall_sigma, n_tables=len(rows), n_pairs_with_table=int(has.sum()),
        r0=torch.tensor(row_r0, dtype=torch.float32, device=device),
        rep=torch.tensor([r is not None for r in rep], dtype=torch.bool, device=device),
        rep_any=any(r is not None for r in rep), rep_A=ft(rp[:, 0]), rep_b=ft(rp[:, 1]), rep_rm=ft(rp[:, 2]),
        rep_rl=ft(rp[:, 3]),
    )


def _interp(T, rows, d, r_lo, dr, G):
    """Piecewise-linear w at distances d (B, M) for table rows (M,): linear extrapolation below r_lo, 0 above."""
    import torch

    u = (d - r_lo) / dr
    i0 = torch.clamp(torch.floor(u.detach()), 0, G - 2).long()
    f = u - i0.to(u.dtype)
    w0 = T[rows[None, :], i0]
    w1 = T[rows[None, :], i0 + 1]
    w = w0 + f * (w1 - w0)
    return torch.where(u >= G - 1, torch.zeros_like(w), w)


def evaluate(rt, X, a_idx, b_idx):
    """(B,) sum over Q6 pairs of purely repulsive penalty w >= 0 (kT)."""
    import torch

    d = (X[:, a_idx] - X[:, b_idx]).norm(dim=-1)
    r0 = rt["r0"][rt["tot"]][None].expand_as(d)
    w0 = _interp(rt["T"], rt["tot"], r0, rt["r_lo"], rt["dr"], rt["G"])
    w = _interp(rt["T"], rt["tot"], d, rt["r_lo"], rt["dr"], rt["G"])
    core = d < r0
    w_core = torch.where(core, (w - w0).clamp(min=0.0), torch.zeros_like(w))
    t = (rt["mu"] - d).clamp(min=0.0)
    wall = t ** 2 / (2.0 * rt["sigma"].clamp(min=1e-7) ** 2)
    w_out = torch.where(rt["has"][None], w_core, wall)
    if rt.get("rep_any"):
        u = rep_energy(d, rt["rep_A"][None], rt["rep_b"][None], rt["rep_rm"][None], rt["rep_rl"][None])
        w_out = torch.where(rt["rep"][None], u, w_out)
    w_out = torch.where(rt.get("enabled", torch.ones_like(rt["has"]))[None], w_out, torch.zeros_like(w_out))
    if rt.get("rep_any"):
        # Weight only the physical contribution. Pairs marked disabled above are
        # omitted from the whole Q6 term by the enabled mask.
        w_out = torch.where(rt["rep"][None], w_out * rt.get("rep_weight", 1.0)[None], w_out)
    return w_out.sum(-1)
