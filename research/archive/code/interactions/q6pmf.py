"""
q6pmf.py -- Purely repulsive Q6 steric model (plan 3 N1, finalized 2026-10-18).

Why. All data-region Q6PMF attractive modes ("all", "hbond", "contact") and polar-H attractive wells have been
deprecated and archived (archive/q6pmf_deprecated_2026-10-18.zip); directional noncovalent attractions belong
exclusively to jacobik.stats.nci.

What. Every Q6 atom pair (> 3 bonds apart) is scored purely repulsively:
  * Class-5 non-polar pairs: physical exchange repulsion (shifted-force Buckingham exponential below the Alvarez vdW
    contact sum r_m, zero value and force at and beyond r_m; A, b per element fitted jointly from GFN2-xTB dimers).
  * Class-4 (1-5 chain), metal ("m"), and polar-H...acceptor ("Hp") pairs: analytical short-range onset core
    w_core(r) = kappa * (exp(b * (r0 - r)) - 1) below the key's 0.1 % data onset r0, and identically 0 at and above r0
    (continued linearly above 50 kT so deep overlaps retain a finite gradient). Polar-H...acceptor pairs use their own
    short-range onset r0 (~1.66 A) rather than non-polar H vdW radii (r_m ~ 2.70 A) so H-bond approach is not blocked.
  * Rare pairs without a Q6P table fall back to their one-sided half-Gaussian wall below mu.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

R_MAX_HIST = 8.0
DR_HIST = 0.02
EDGES = np.round(np.arange(0.0, R_MAX_HIST + 1e-9, DR_HIST), 4)
NB = len(EDGES) - 1
ANGLE_SPLIT_COS = -0.5            # D-H...A angle 120 deg
ANGLE_PRIOR = {"": 1.0, "lin": 0.25, "bent": 0.75}
RT_KJ = 2.479                     # kJ/mol at 298.15 K
R_LO, R_HI, DR_TAB = 0.6, 6.0, 0.02
R_TAPER = 5.0
W_LINEAR_ABOVE = 50.0
N_MIN = 3000
N_MIN_BIN = 25
ONSET_Q = 1e-3
W_DATA_CLIP = 8.0

# GFN2-xTB head-on dimer scans (E = A exp(-b r) on the repulsive branch, A in kJ/mol, b in 1/A).
GFN2_CORES: Dict[str, Tuple[float, float]] = {
    "H_H": (8489.4, 4.191), "C_H": (31933.4, 3.603), "H_O": (88302.4, 5.078), "H_N": (192171.6, 5.585),
    "C_O": (136563.3, 3.528), "Hp_O": (824315.4, 7.699), "Hp_N": (3488633.6, 8.812), "C_C": (30022.0, 2.884),
}
GENERIC_B_H, GENERIC_B = 4.0, 3.2

#: Per-element exchange repulsion {element: (A kJ/mol, b 1/A)} from GFN2-xTB dimers and Alvarez 2013 vdW radii (A).
REP_PARAMS: Dict[str, Tuple[float, float]] = {
    "H": (8304.2, 3.867), "B": (6217.9, 2.5), "C": (52908.4, 3.1943), "N": (2558.4, 2.5), "O": (1794.0, 2.5),
    "F": (4288.8, 2.5493), "Si": (107369.6, 2.5), "P": (30445.7, 2.5), "S": (30862.1, 2.5), "Cl": (69172.6, 2.8674),
    "Se": (102333.9, 2.5882), "Br": (451089.0, 3.0641), "I": (2844105.5, 3.2715)}
ALVAREZ_RADII: Dict[str, float] = {"H": 1.20, "B": 1.91, "C": 1.77, "N": 1.66, "O": 1.50, "F": 1.46, "Si": 2.19,
                                   "P": 1.90, "S": 1.89, "Cl": 1.82, "Se": 1.82, "Br": 1.86, "I": 2.04}


def rep_pair_params(e1: str, e2: str) -> Optional[Tuple[float, float, float, float]]:
    """(A kJ/mol, b 1/A, r_m A, r_lin A) of the physical repulsion of an element pair, or None without parameters."""
    e1, e2 = ("H" if e1 == "Hp" else e1), ("H" if e2 == "Hp" else e2)
    if e1 not in REP_PARAMS or e2 not in REP_PARAMS or e1 not in ALVAREZ_RADII or e2 not in ALVAREZ_RADII:
        return None
    A = math.sqrt(REP_PARAMS[e1][0] * REP_PARAMS[e2][0])
    b = 0.5 * (REP_PARAMS[e1][1] + REP_PARAMS[e2][1])
    rm = ALVAREZ_RADII[e1] + ALVAREZ_RADII[e2]
    u = lambda r: A * (math.exp(-b * r) - math.exp(-b * rm) + b * math.exp(-b * rm) * (r - rm)) / RT_KJ
    lo, hi = 0.0, rm
    if u(lo) <= W_LINEAR_ABOVE:
        return A, b, rm, 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if u(mid) > W_LINEAR_ABOVE else (lo, mid)
    return A, b, rm, hi


def rep_energy(d, A, b, rm, rl):
    """Physical repulsion (kT) at distances d (torch, any shape; A, b, rm, rl broadcastable): shifted force below rm,
    0 at and beyond, linear below rl."""
    import torch

    A, b, rm, rl = (torch.as_tensor(v, dtype=d.dtype, device=d.device) for v in (A, b, rm, rl))
    em = torch.exp(-b * rm)
    u = lambda r: A * (torch.exp(-b * r) - em + b * em * (r - rm)) / RT_KJ
    slope = A * b * (torch.exp(-b * rl) - em) / RT_KJ
    lin = u(rl) + slope * (rl - d)
    out = torch.where(d < rl, lin, u(torch.maximum(d, rl)))
    return torch.where(d < rm, out, torch.zeros_like(out))


def key_name(key: Tuple[str, str]) -> str:
    return f"{key[0]}_{key[1]}"


# ----------------------------------------------------------------------------------------------------------------
# topology: bond-separation class of each Q6 pair (shared by extraction and runtime)
# ----------------------------------------------------------------------------------------------------------------
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


M13 = "m13"
M13_HAPTIC = "m13h"
M13_CLASSES = (M13, M13_HAPTIC)
M13_POOLED = "Mx"


def metal_13_pairs(mol) -> List[Tuple[int, int, str]]:
    """(metal idx, X idx, class), 1-based: X bonded through a ligand bond to a donor atom."""
    from jacobik.stats.mapping import raw_metal_bond_exclusion_pairs
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors

    m = mol.metal_idx
    don = set(donor_indices_of(mol))
    nb = get_ligand_neighbors(mol.atoms, mol.bonds, m)
    raw = raw_metal_bond_exclusion_pairs(mol.atoms, mol.bonds, m, q6=True)
    hapt = {d for si in getattr(mol, "coord_sites", None) or [] if si.kind == "haptic" for d in si.donor_indices}
    xs: Dict[int, str] = {}
    for d in don:
        for x in nb.get(d, ()):
            if x in don or x == m or tuple(sorted((m, x))) in raw:
                continue
            if d in hapt or x not in xs:
                xs[x] = M13_HAPTIC if d in hapt else M13
    return [(m, x, xs[x]) for x in sorted(xs)]


def m13_keys(sym_m: str, sym_x: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    own = tuple(sorted((sym_m, sym_x)))
    pooled = tuple(sorted((M13_POOLED, sym_x)))
    return own, pooled


def m13_records(mol):
    sym = {a.idx: a.symbol for a in mol.atoms}
    P = {a.idx: np.asarray(a.position, float) for a in mol.atoms}
    labels: List[str] = []
    dist: List[float] = []
    for m, x, cls in metal_13_pairs(mol):
        d = float(np.linalg.norm(P[m] - P[x]))
        for k in m13_keys(sym[m], sym[x]):
            labels.append(f"{key_name(k)}|{cls}|")
            dist.append(d)
    return labels, np.asarray(dist)


def fit_m13_walls(hist: Dict[str, np.ndarray], n_min: int = N_MIN) -> Dict[str, dict]:
    """Metal 1-3 half-Gaussian walls fitted from 0.02 A histograms."""
    rc = 0.5 * (EDGES[1:] + EDGES[:-1])
    out = {}
    for lab, h in hist.items():
        name, cls, ang = lab.split("|")
        if cls not in M13_CLASSES or ang != "" or h.sum() < n_min:
            continue
        cum = np.cumsum(h) / h.sum()
        mu = float(np.interp(0.05, cum, rc))
        left = rc <= mu
        sigma = float(np.sqrt((h[left] * (rc[left] - mu) ** 2).sum() / max(h[left].sum(), 1e-12)))
        sigma = max(sigma, 0.1)
        a_, b_ = name.split("_")
        out[f"Q6P:{(a_, b_, cls, 'wall')!r}"] = dict(type="halfgauss", n=int(h.sum()), k=1, mu=round(mu, 4),
                                                     sigma=round(sigma, 4))
    return out


def polar_donor(mol, polar: Iterable[int]) -> Dict[int, int]:
    """{polar H idx: its donor atom idx} (first ligand neighbour)."""
    from jacobik.topology.graph import get_ligand_neighbors

    nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    return {h: nb[h][0] for h in polar if nb.get(h)}


# ----------------------------------------------------------------------------------------------------------------
# extraction: uncapped histograms
# ----------------------------------------------------------------------------------------------------------------
def contact_records(mol, r_max: float = R_MAX_HIST):
    """Every Q6 pair of a crystal within r_max: (labels 'a_b|cls|ang', distances)."""
    from jacobik.stats import key_builder as kb
    from jacobik.stats.mapping import raw_metal_bond_exclusion_pairs
    from jacobik.topology.graph import donor_indices_of, q6_topological_exclusions

    atoms = list(mol.atoms)
    if len(atoms) < 2:
        return [], np.zeros(0)
    P = np.array([a.position for a in atoms], dtype=float)
    idx = [a.idx for a in atoms]
    pos = {a: i for i, a in enumerate(idx)}
    sym = [a.symbol for a in atoms]
    close = q6_topological_exclusions(mol.atoms, mol.bonds, mol.metal_idx, donor_indices_of(mol))
    close |= raw_metal_bond_exclusion_pairs(mol.atoms, mol.bonds, mol.metal_idx, q6=True)
    polar = kb.q6_polar_h(mol)
    donor = polar_donor(mol, polar)
    sep = ligand_separation(mol)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    ii, jj = np.nonzero(np.triu(D < r_max, k=1))
    labels: List[str] = []
    dist: List[float] = []
    for a, b in zip(ii.tolist(), jj.tolist()):
        lo, hi = (idx[a], idx[b]) if idx[a] < idx[b] else (idx[b], idx[a])
        if (lo, hi) in close:
            continue
        key = kb.q6_pair_key(sym[a], sym[b], idx[a] in polar, idx[b] in polar)
        cls = pair_class(idx[a], idx[b], mol.metal_idx, sep)
        base = key_name(key) + "|" + cls
        labels.append(base + "|")
        dist.append(D[a, b])
        if "Hp" in key:
            h, acc = (a, b) if (idx[a] in polar and sym[b] in kb.Q6_ACCEPTORS) else (b, a)
            dn = donor.get(idx[h])
            if dn is not None and dn in pos:
                u, v = P[pos[dn]] - P[h], P[acc] - P[h]
                c = float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
                labels.append(base + ("|lin" if c <= ANGLE_SPLIT_COS else "|bent"))
                dist.append(D[a, b])
    return labels, np.asarray(dist)


class HistAccumulator:
    """Per-label distance histograms (EDGES), summed over complexes."""

    def __init__(self, records=None):
        self.h: Dict[str, np.ndarray] = {}
        self.records = records or contact_records

    def add(self, mol, weight: float = 1.0) -> None:
        labels, d = self.records(mol)
        if not labels:
            return
        lab = np.asarray(labels)
        uniq, inv = np.unique(lab, return_inverse=True)
        b = np.clip((d / DR_HIST).astype(np.int64), 0, NB - 1)
        cnt = np.bincount(inv * NB + b, minlength=len(uniq) * NB).reshape(len(uniq), NB) * float(weight)
        for k, row in zip(uniq.tolist(), cnt):
            if k in self.h:
                self.h[k] += row
            else:
                self.h[k] = row.astype(np.float64)

    def save(self, path) -> None:
        np.savez_compressed(path, edges=EDGES, **{"h:" + k: v for k, v in self.h.items()})


def load_hist(paths: Iterable) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for p in paths:
        z = np.load(p)
        for k in z.files:
            if k.startswith("h:"):
                out[k[2:]] = out.get(k[2:], 0) + z[k]
    return out


# ----------------------------------------------------------------------------------------------------------------
# fit (onset r0 and analytical core below r0)
# ----------------------------------------------------------------------------------------------------------------
def core_params(name: str, r0: float) -> Tuple[float, float]:
    """(b, kappa) of the physical core for element-pair name 'A_B' at onset r0."""
    if name in GFN2_CORES:
        A, b = GFN2_CORES[name]
        return b, max(A * math.exp(-b * r0) / RT_KJ, 1.0)
    return (GENERIC_B_H if "H" in name.split("_") or "Hp" in name.split("_") else GENERIC_B), 1.0


def _smooth(h: np.ndarray) -> np.ndarray:
    k = np.ones(3)
    return np.convolve(h.astype(float), k, mode="same")


def _core(grid: np.ndarray, r0: float, w_r0: float, b: float, kappa: float) -> np.ndarray:
    """w(r0) + kappa (exp(b (r0 - r)) - 1) for r < r0, continued linearly above W_LINEAR_ABOVE."""
    wc = w_r0 + kappa * (np.exp(b * (r0 - grid)) - 1.0)
    lim = W_LINEAR_ABOVE
    if (wc > lim).any() and lim > w_r0:
        r_l = r0 - math.log((lim - w_r0) / kappa + 1.0) / b
        slope = kappa * b * math.exp(b * (r0 - r_l))
        over = wc > lim
        wc[over] = lim + slope * (r_l - grid[over])
    return wc


def fit_tables(hist: Dict[str, np.ndarray], n_min: int = N_MIN) -> Dict[str, dict]:
    """Fit purely repulsive Q6P core tables {"Q6P:('a', 'b', cls, '')": {...}} from merged histograms."""
    rc = 0.5 * (EDGES[1:] + EDGES[:-1])
    grid = np.round(np.arange(R_LO, R_HI + 1e-9, DR_TAB), 4)
    ref: Dict[str, np.ndarray] = {}
    for lab, h in hist.items():
        name, cls, ang = lab.split("|")
        if ang == "":
            ref[cls] = ref.get(cls, 0) + h
    out: Dict[str, dict] = {}
    for lab, h in hist.items():
        name, cls, ang = lab.split("|")
        if cls in M13_CLASSES and ang == "" and h.sum() >= n_min:
            i0 = int(np.searchsorted(np.cumsum(h) / h.sum(), ONSET_Q))
            r0 = float(rc[min(i0, NB - 1)])
            b, kappa = core_params(name, r0)
            tab = np.zeros(len(grid))
            core = grid < r0
            tab[core] = _core(grid[core], r0, 0.0, b, kappa)
            a_, b_ = name.split("_")
            out[f"Q6P:{(a_, b_, cls, '')!r}"] = dict(
                type="pmf", n=int(h.sum()), n_key=int(h.sum()), r0=round(r0, 3), core_b=round(b, 3),
                core_kappa=round(kappa, 3), r_lo=R_LO, dr=DR_TAB, w=[round(float(x), 4) for x in tab])
            continue
        if ang != "" or h.sum() < n_min or cls not in ref or cls in M13_CLASSES:
            continue
        hs, rs = _smooth(h), _smooth(ref[cls])
        ok = (hs >= N_MIN_BIN) & (rs > 0)
        cum = np.cumsum(h) / h.sum()
        i0 = int(np.searchsorted(cum, ONSET_Q))
        while i0 < NB and not ok[i0]:
            i0 += 1
        good = ok & (rc >= rc[min(i0, NB - 1)])
        if good.sum() < 2:
            continue
        r0 = float(rc[i0])
        b, kappa = core_params(name, r0)
        tab = np.zeros(len(grid))
        core = grid < r0
        tab[core] = _core(grid[core], r0, 0.0, b, kappa)
        a_, b_ = name.split("_")
        out[f"Q6P:{(a_, b_, cls, '')!r}"] = dict(
            type="pmf", n=int(h.sum()), n_key=int(h.sum()), r0=round(r0, 3), core_b=round(b, 3),
            core_kappa=round(kappa, 3), r_lo=R_LO, dr=DR_TAB, w=[round(float(x), 4) for x in tab])
    return out


# ----------------------------------------------------------------------------------------------------------------
# runtime (purely repulsive: physical exchange on class-5 non-polar pairs + onset core below r0 elsewhere)
# ----------------------------------------------------------------------------------------------------------------
def build_runtime(mol, ge, tables: Dict[tuple, dict], wall_mu, wall_sigma, device):
    """Tensors for the purely repulsive Q6 scorer."""
    import torch

    from jacobik.stats import key_builder as kb
    from jacobik.stats.mapping import raw_metal_bond_exclusion_pairs

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
    for p, (ia, ib) in enumerate(zip(a_idx, b_idx)):
        key = ge._q6_elem_keys[p]
        lo, hi = sorted((atoms[ia].idx, atoms[ib].idx))
        if (lo, hi) in raw_m:
            t_tot.append(-1)
            rep.append(None)
            continue
        cls = pair_class(atoms[ia].idx, atoms[ib].idx, mol.metal_idx, sep)
        polar_acc = "Hp" in key and (key[0] in kb.Q6_ACCEPTORS or key[1] in kb.Q6_ACCEPTORS)
        rep.append(rep_pair_params(key[0], key[1]) if cls == "5" and not polar_acc else None)
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

    d = (X[:, a_idx] - X[:, b_idx]).norm(dim=-1)  # (B, M)
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
    return w_out.sum(-1)
