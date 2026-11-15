"""
src/jacobik/fit/q6p.py
==============
Quantity Q6P: Purely repulsive Q6 PMF tables and metal 1-3 core tables & walls.

Key tables and models:
- PMF tables: Contact distance PMFs fit from histogram accumulators (q6hist_<split>.npz)
  across bond-separation classes ('1-4', '1-5', 'intra', 'inter') and H-bond angles.
- Metal 1-3 tables & walls: Metal 1-3 contacts (M...donor-substituent, M...C/N/O/P/S)
  fit from q6hist_m13_<split>.npz across M13_CLASSES ('m13', 'm13_cis', 'm13_trans').
- Fit model: Purely repulsive spline / potential of mean force with soft-core onset
  and analytical asymptotic exponential continuation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import math
from jacobik.core.env import RT_KJ
from jacobik.stats.steric import W_LINEAR_ABOVE, ligand_separation, pair_class

log = logging.getLogger(__name__)

QUANTITY: str = "Q6P"
N_MIN: int = 3000  # production PMF-table admission threshold
M13 = "m13"
M13_HAPTIC = "m13h"
M13_CLASSES = (M13, M13_HAPTIC)
M13_POOLED = "Mx"


# ---------------------------------------------------------------------------
# Extraction & Histograms
# ---------------------------------------------------------------------------


R_MAX_HIST = 8.0
DR_HIST = 0.02
EDGES = np.round(np.arange(0.0, R_MAX_HIST + 1e-9, DR_HIST), 4)
NB = len(EDGES) - 1
ANGLE_SPLIT_COS = -0.5            # D-H...A angle 120 deg
ANGLE_PRIOR = {"": 1.0, "lin": 0.25, "bent": 0.75}
R_LO, R_HI, DR_TAB = 0.6, 6.0, 0.02
R_TAPER = 5.0
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

def metal_13_pairs(mol) -> List[Tuple[int, int, str]]:
    """(metal idx, X idx, class), 1-based: X bonded through a ligand bond to a donor atom."""
    from jacobik.topology.mapping import raw_metal_bond_exclusion_pairs
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


def key_name(key: Tuple[str, str]) -> str:
    return f"{key[0]}_{key[1]}"


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
    from jacobik.stats import keys as kb
    from jacobik.topology.mapping import raw_metal_bond_exclusion_pairs
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


def create_contact_accumulator() -> HistAccumulator:
    """Create a fresh Q6PMF histogram accumulator for molecule contacts."""
    return HistAccumulator()


def create_m13_accumulator() -> HistAccumulator:
    """Create a fresh histogram accumulator for metal 1-3 contacts."""
    return HistAccumulator(records=m13_records)


def extract_contact_records(mol: Any, r_max: float = 6.0) -> List[Tuple]:
    """Extract contact histogram records (key, class, angle_class, d) from a complex."""
    return contact_records(mol, r_max=r_max)


def extract_m13_records(mol: Any) -> List[Tuple]:
    """Extract metal 1-3 contact records from a complex."""
    return m13_records(mol)


def fit_from_observations(
    contact_hist: Optional[Dict[str, np.ndarray]] = None,
    m13_hist: Optional[Dict[str, np.ndarray]] = None,
    n_min: int = N_MIN,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit full Q6P database entries from contact and metal 1-3 histograms.

    Combines:
    1. Purely repulsive PMF tables from contact_hist
    2. Metal 1-3 tables from m13_hist
    3. Metal 1-3 core walls from m13_hist
    """
    entries: Dict[str, Dict[str, Any]] = {}

    if contact_hist:
        pmf = fit_tables(contact_hist, n_min=n_min)
        entries.update(pmf)

    if m13_hist:
        m13_tabs = {
            k: v for k, v in fit_tables(m13_hist, n_min=n_min).items()
            if any(f"'{c}'" in k for c in M13_CLASSES)
        }
        m13_walls = fit_m13_walls(m13_hist, n_min=n_min)
        entries.update(m13_tabs)
        entries.update(m13_walls)

    return entries


def fit_all(
    hist_contact_paths: Sequence[Union[str, Path]],
    hist_m13_paths: Optional[Sequence[Union[str, Path]]] = None,
    out_path: Optional[Union[str, Path]] = None,
    n_min: int = N_MIN,
) -> Dict[str, Dict[str, Any]]:
    """Fit all Q6P entries from histogram npz files."""
    contact_hist = load_hist(hist_contact_paths) if hist_contact_paths else None
    m13_hist = load_hist(hist_m13_paths) if hist_m13_paths else None
    results = fit_from_observations(contact_hist, m13_hist, n_min=n_min)
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as fh:
            json.dump(results, fh, indent=1)
    return results
