"""
nci.py -- noncovalent interaction term "NCI" (plan 3 N2, 2026-10-18; profile field nci, default on in u1).

Why. Q6 is purely repulsive (q6pmf: physical exchange repulsion on class-5 non-polar pairs + analytical onset core
below r0 on class-4, metal, and polar-H pairs; all Q6PMF data-region attractive wells are deprecated and removed).
Context- and direction-dependent attractions are priced here by 2D C^2 radial-Chebyshev / cubic-quartic Hermite
surfaces regularized against full-complex thermal compliance SDs and calibrated against matched Pure-N1 MACE64
energy differences (~0-50 candidate interactions per complex, not ~1500 atom pairs).

Types (admitted by the census tools/jdev/budget/nci_census.py):
  "hb"   polar H (bonded to N, O, F) ... acceptor (N, O, F, S, Cl, Br, I): x = (H...A distance, D-H...A angle 0-180)
  "pp"   two sp2 5- / 6-rings: x = (centroid distance, angle between the ring planes 0-90) with C5 slip/co-bound filter
  "chp"  non-polar H ... sp2 ring: x = (H...centroid distance, angle of centroid->H to the ring normal 0-90) with
         C3 C-H directionality gate and C4 per-(carbon, ring) sibling log-sum-exp aggregation
Only pairs more than 4 ligand bonds apart or without a ligand path (other ligands, through the metal; for rings every
ring atom): intra-ligand geometry up to 1-5 belongs to the bonded terms and the Q5 torsions.
H-bonds whose donor D and acceptor A are both metal-bound are excluded: their coordination geometry belongs to Q2.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

NCI_ON = __import__("jacobik.profile", fromlist=["get"]).get("nci")  # profile field
TYPES = ("hb", "pp", "chp")
ACTIVE_TYPES = __import__("jacobik.profile", fromlist=["get"]).get("nci_types")  # profile field


def hbond_replacement_enabled():
    """Returns True when hb NCI is active (legacy helper; Q6 polar-H attractive wells are now permanently off)."""
    return NCI_ON and "hb" in ACTIVE_TYPES


#: per type: r grid (lo, hi, step A), angle grid (lo, hi, step deg), taper range (A), normalisation shell (A)
SPEC = {
    "hb": dict(r=(1.4, 3.6, 0.05), a=(0.0, 180.0, 5.0), win=(2.6, 3.0), shell=(3.0, 3.6)),
    "pp": dict(r=(3.0, 6.0, 0.1), a=(0.0, 90.0, 5.0), win=(4.5, 5.0), shell=(5.0, 6.0)),
    "chp": dict(r=(2.0, 4.2, 0.05), a=(0.0, 90.0, 5.0), win=(3.1, 3.5), shell=(3.5, 4.2)),
}
ACCEPTORS = frozenset({"N", "O", "F", "S", "Cl", "Br", "I"})
RING_ELEMENTS = frozenset({"C", "N", "O", "S", "B", "P"})
PSEUDO = 5.0
S_CLIP = 8.0
N_MIN = 1000

#: Median full-complex compliance SDs from n2_thermal_audit_2026-10-18 (one-sided sharpness regularizer ceiling)
THERMAL_COMPLIANCE_SD = {
    "hb": (0.1906741756954919, math.radians(14.566764014704948)),
    "pp": (0.26223202412665503, math.radians(10.205406667282261)),
    "chp": (0.35082490524557286, math.radians(8.94368205953425)),
}
GAMMA_COMPLIANCE = {"hb": 1.0, "pp": 0.5, "chp": 0.5}
#: Matched-baseline calibrated amplitudes (score units, fitted on leak-free 2026-10-18_training_calibration_v2)
TYPE_SCALES = {"hb": 1.135, "pp": 2.817, "chp": 0.448}


def _grid(t):
    s = SPEC[t]
    r_lo, r_hi, dr = s["r"]
    a_lo, a_hi, da = s["a"]
    return r_lo, dr, int(round((r_hi - r_lo) / dr)), a_lo, da, int(round((a_hi - a_lo) / da))


def _taper(r, lo, hi):
    return np.where(r <= lo, 1.0, np.where(r >= hi, 0.0, np.cos(0.5 * math.pi * (r - lo) / (hi - lo)) ** 2))


def quintic_taper_np(r: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """C2 quintic taper: 1 for r <= lo, 0 for r >= hi, zero 1st and 2nd derivatives at lo and hi."""
    t = np.clip((r - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - t * t * t * (10.0 - t * (15.0 - 6.0 * t))


def quintic_taper_torch(r, lo: float, hi: float):
    """Torch C2 quintic taper: 1 for r <= lo, 0 for r >= hi."""
    t = ((r - lo) / (hi - lo)).clamp(0.0, 1.0)
    return 1.0 - t * t * t * (10.0 - t * (15.0 - 6.0 * t))


# ----------------------------------------------------------------------------------------------------------------
# topology: candidates (shared by extraction and runtime)
# ----------------------------------------------------------------------------------------------------------------
def sp2_rings(mol, nb) -> List[List[int]]:
    """5- / 6-rings of the ligand graph (metal excluded) whose atoms are ring elements with <= 3 ligand neighbours
    (H counted): aromatic and other sp2 rings, topology only (no geometry, so extraction and runtime agree). Atom
    indices in ring order."""
    sym = {a.idx: a.symbol for a in mol.atoms}
    ok = {a.idx for a in mol.atoms if a.idx != mol.metal_idx and sym[a.idx] in RING_ELEMENTS
          and len(nb.get(a.idx, ())) <= 3}
    seen, out = set(), []

    def walk(start, cur, path):
        for v in nb.get(cur, ()):
            if v == start and len(path) in (5, 6):
                key = frozenset(path)
                if key not in seen:
                    seen.add(key)
                    out.append(list(path))
            elif v in ok and v > start and v not in path and len(path) < 6:
                walk(start, v, path + [v])

    for s in sorted(ok):
        walk(s, s, [s])
    return out


def hbond_candidates(mol, nb=None, sep=None) -> list:
    """Topology-only (D, H, A, ladder) candidates, shared with the geometric census.

    Exclude co-bound D/A, including haptic face atoms, but retain outer-sphere
    contacts with just one metal-bound partner. Use primary coordination sites,
    not incidental raw metal contacts. Optional graphs avoid rebuilding them.
    """
    from jacobik.stats import key_builder as kb
    from research.archive.code.interactions.q6pmf import ligand_separation
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors

    if nb is None:
        nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    if sep is None:
        sep = ligand_separation(mol, max_depth=4)
    bound = set(donor_indices_of(mol))
    sym = {a.idx: a.symbol for a in mol.atoms}
    acc = [a.idx for a in mol.atoms if a.idx != mol.metal_idx and a.symbol in ACCEPTORS]
    out = []
    for h in sorted(kb.q6_polar_h(mol, nb)):
        d = nb[h][0]
        for a in acc:
            if a != d and a not in sep.get(h, {}) and not (d in bound and a in bound):
                out.append((d, h, a, [("hb", sym[d], sym[a]), ("hb", "*", sym[a]), ("hb", "*", "*")]))
    return out


def candidates(mol) -> Dict[str, list]:
    """{"hb": [(D, H, A, keys)], "pp": [(ring_i, ring_j, keys)], "chp": [(H, ring, keys)]}: atom indices (mol.atoms
    .idx), keys = the ladder from most specific to pooled."""
    from jacobik.stats import key_builder as kb
    from research.archive.code.interactions.q6pmf import ligand_separation
    from jacobik.topology.graph import get_ligand_neighbors

    nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    sep = ligand_separation(mol, max_depth=4)
    m = mol.metal_idx
    far = lambda i, j: j not in sep.get(i, {})
    sym = {a.idx: a.symbol for a in mol.atoms}
    lig = [a.idx for a in mol.atoms if a.idx != m]
    polar = kb.q6_polar_h(mol, nb)
    out: Dict[str, list] = {t: [] for t in TYPES}
    out["hb"] = hbond_candidates(mol, nb, sep)
    rings = sp2_rings(mol, nb)
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            ri, rj = rings[i], rings[j]
            if not set(ri) & set(rj) and all(far(u, v) for u in ri for v in rj):
                out["pp"].append((ri, rj, [("pp", "*")]))
    for ring in rings:
        near = set(ring) | {x for u in ring for x in nb.get(u, ())}
        for h in lig:
            if sym[h] == "H" and h not in polar and h not in near and all(far(h, u) for u in ring):
                out["chp"].append((h, ring, [("chp", "*")]))
    return out


# ----------------------------------------------------------------------------------------------------------------
# features (torch; extraction uses them at B = 1)
# ----------------------------------------------------------------------------------------------------------------
def _ring_frame(X, rings):
    """Centroids (B, R, 3) and unit normals (B, R, 3) of rings given as an index tensor (R, 6) padded with -1 (5-rings);
    normal = normalised sum of consecutive cross products about the centroid (smooth, orientation-free)."""
    import torch

    valid = rings >= 0
    idx = rings.clamp(min=0)
    P = X[:, idx]                                                  # (B, R, 6, 3)
    w = valid.to(X.dtype)[None, :, :, None]
    c = (P * w).sum(2) / w.sum(2)
    Q = (P - c[:, :, None]) * w
    n_at = valid.sum(1)                                            # ring sizes
    nxt = torch.stack([torch.roll(torch.arange(6, device=X.device), -1)] * len(rings))
    nxt = torch.where(nxt >= n_at[:, None], torch.zeros_like(nxt), nxt)   # 5-ring: atom 4 -> 0
    Qn = torch.gather(Q, 2, nxt[None, :, :, None].expand_as(Q))
    n = torch.cross(Q, Qn, dim=-1).sum(2)
    return c, n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _deg_acos(c):
    import torch

    return torch.rad2deg(torch.acos(c.clamp(-1.0 + 1e-6, 1.0 - 1e-6)))


def features(rt_type: str, X, idx: dict):
    """(r, angle) (B, M) of one type's candidates; idx: index tensors (positions in X) built by index_tensors."""
    if rt_type == "hb":
        H, D, A = X[:, idx["h"]], X[:, idx["d"]], X[:, idx["a"]]
        u, v = D - H, A - H
        r = v.norm(dim=-1)
        return r, _deg_acos((u * v).sum(-1) / (u.norm(dim=-1) * r).clamp(min=1e-9))
    c, n = _ring_frame(X, idx["rings"])
    if rt_type == "pp":
        ci, cj, ni, nj = c[:, idx["ri"]], c[:, idx["rj"]], n[:, idx["ri"]], n[:, idx["rj"]]
        return (ci - cj).norm(dim=-1), _deg_acos((ni * nj).sum(-1).abs())
    Hh = X[:, idx["h"]]
    v = Hh - c[:, idx["ring"]]
    r = v.norm(dim=-1)
    return r, _deg_acos(((v * n[:, idx["ring"]]).sum(-1)).abs() / r.clamp(min=1e-9))


def features_cos_torch(rt_type: str, X, idx: dict):
    """Return (r, z) of shape (B, M) where z = cos(psi) in [-1, 1] is C^infty in Cartesian X:
      - For 'hb': z = cos(theta) = (u . v) / (||u|| ||v||)
      - For 'pp': z = 2 * (n_i . n_j)^2 - 1
      - For 'chp': z = 2 * ((v . n) / r)^2 - 1
    No acos() or abs() is used anywhere.
    """
    if rt_type == "hb":
        H, D, A = X[:, idx["h"]], X[:, idx["d"]], X[:, idx["a"]]
        u, v = D - H, A - H
        r = v.norm(dim=-1)
        un = u.norm(dim=-1)
        z = (u * v).sum(-1) / (un * r).clamp(min=1e-12)
        return r, z.clamp(-1.0, 1.0)
    c, n = _ring_frame(X, idx["rings"])
    if rt_type == "pp":
        ci, cj, ni, nj = c[:, idx["ri"]], c[:, idx["rj"]], n[:, idx["ri"]], n[:, idx["rj"]]
        r = (ci - cj).norm(dim=-1)
        dot = (ni * nj).sum(-1)
        z = 2.0 * (dot * dot) - 1.0
        return r, z.clamp(-1.0, 1.0)
    Hh = X[:, idx["h"]]
    v = Hh - c[:, idx["ring"]]
    r = v.norm(dim=-1)
    dot = (v * n[:, idx["ring"]]).sum(-1) / r.clamp(min=1e-12)
    z = 2.0 * (dot * dot) - 1.0
    return r, z.clamp(-1.0, 1.0)


def index_tensors(cands: Dict[str, list], pos: Dict[int, int], device, keep: Optional[Dict[str, list]] = None):
    """Per type: index tensors into X (positions pos[atom idx]); keep[t] = candidate indices to include (default all)."""
    import torch

    lt = lambda v: torch.tensor(v, dtype=torch.long, device=device)
    out = {}
    for t in TYPES:
        cs = cands[t] if keep is None else [cands[t][k] for k in keep[t]]
        if not cs:
            continue
        if t == "hb":
            out[t] = dict(d=lt([pos[c[0]] for c in cs]), h=lt([pos[c[1]] for c in cs]), a=lt([pos[c[2]] for c in cs]))
            continue
        rings, rid = [], {}

        def ring_id(r):
            k = tuple(r)
            if k not in rid:
                rid[k] = len(rings)
                rings.append([pos[u] for u in r] + [-1] * (6 - len(r)))
            return rid[k]

        if t == "pp":
            ri = [ring_id(c[0]) for c in cs]
            rj = [ring_id(c[1]) for c in cs]
            out[t] = dict(rings=lt(rings), ri=lt(ri), rj=lt(rj))
        else:
            rg = [ring_id(c[1]) for c in cs]
            out[t] = dict(rings=lt(rings), ring=lt(rg), h=lt([pos[c[0]] for c in cs]))
    return out


# ----------------------------------------------------------------------------------------------------------------
# extraction: 2D histograms
# ----------------------------------------------------------------------------------------------------------------
def records(mol):
    """{label "t|key": (r array, angle array)} of every candidate of a crystal within the type's r range; each
    candidate recorded under every ladder key."""
    import torch

    atoms = list(mol.atoms)
    pos = {a.idx: i for i, a in enumerate(atoms)}
    X = torch.tensor(np.array([a.position for a in atoms], dtype=np.float64))[None]
    cands = candidates(mol)
    idx = index_tensors(cands, pos, "cpu")
    out: Dict[str, Tuple[list, list]] = {}
    for t, ix in idx.items():
        r, a = features(t, X, ix)
        r, a = r[0].numpy(), a[0].numpy()
        lo, hi = SPEC[t]["r"][:2]
        for k in np.flatnonzero((r >= lo) & (r < hi)):
            for key in cands[t][k][-1]:
                lab = "|".join(str(x) for x in key)
                rr, aa = out.setdefault(lab, ([], []))
                rr.append(float(r[k]))
                aa.append(float(a[k]))
    return out


class HistAccumulator:
    """Per-label 2D histograms (SPEC grids), summed over complexes (weights: LnQM sibling shares)."""

    def __init__(self):
        self.h: Dict[str, np.ndarray] = {}

    def add(self, mol, weight: float = 1.0) -> None:
        for lab, (r, a) in records(mol).items():
            r_lo, dr, nr, a_lo, da, na = _grid(lab.split("|")[0])
            i = np.clip(((np.asarray(r) - r_lo) / dr).astype(int), 0, nr - 1)
            j = np.clip(((np.asarray(a) - a_lo) / da).astype(int), 0, na - 1)
            cnt = np.bincount(i * na + j, minlength=nr * na).reshape(nr, na) * float(weight)
            self.h[lab] = self.h.get(lab, 0) + cnt

    def save(self, path) -> None:
        np.savez_compressed(path, **{"h:" + k: v for k, v in self.h.items()})


def load_hist(paths) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for p in paths:
        z = np.load(p)
        for k in z.files:
            if k.startswith("h:"):
                out[k[2:]] = out.get(k[2:], 0) + z[k]
    return out


# ----------------------------------------------------------------------------------------------------------------
# fit & C^2 surface construction
# ----------------------------------------------------------------------------------------------------------------
def bin_volumes(t: str) -> np.ndarray:
    """Random-placement volume of each (r, angle) bin: (r_hi^3 - r_lo^3) / 3 x (cos a_lo - cos a_hi)."""
    r_lo, dr, nr, a_lo, da, na = _grid(t)
    re = r_lo + dr * np.arange(nr + 1)
    ae = np.radians(a_lo + da * np.arange(na + 1))
    return np.outer((re[1:] ** 3 - re[:-1] ** 3) / 3.0, np.cos(ae[:-1]) - np.cos(ae[1:]))


def _untaper_empirical_table(t: str, s_table: np.ndarray) -> np.ndarray:
    """Recover untapered, unclipped log-enrichment on bin centers inside the active window."""
    r_lo, dr, nr, _, _, _ = _grid(t)
    rc = r_lo + dr * (np.arange(nr) + 0.5)
    win_lo, win_hi = SPEC[t]["win"]
    old_taper = _taper(rc, win_lo, win_hi)[:, None]
    out = np.zeros_like(s_table, dtype=np.float64)
    mask = old_taper > 1e-3
    out[mask[:, 0]] = s_table[mask[:, 0]] / old_taper[mask[:, 0]]
    clipped = np.clip(out / S_CLIP, -0.995, 0.995)
    return S_CLIP * np.arctanh(clipped)


def fit_empirical_surface(
    t: str,
    h: Optional[np.ndarray] = None,
    s_existing: Optional[np.ndarray] = None,
    lam_rough: float = 0.0,
    lam_sharp: float = 0.0,
    gamma_compliance: float = 1.0,
    ess_ratio: float = 1.0,
) -> np.ndarray:
    """Fit pre-taper, pre-tanh log-enrichment grid F (nr, na) on bin centers with optional roughness
    and one-sided thermal-compliance sharpness regularization."""
    from scipy.optimize import minimize
    from scipy.signal import convolve2d

    r_lo, dr, nr, a_lo, da, na = _grid(t)
    rc = r_lo + dr * (np.arange(nr) + 0.5)
    if h is not None:
        vol = bin_volumes(t)
        k = np.ones((3, 3)) / 9.0
        sm = lambda x: convolve2d(x, k, mode="same", boundary="symm")
        hs, vs = sm(h), sm(vol)
        shell = (rc >= SPEC[t]["shell"][0]) & (rc < SPEC[t]["shell"][1])
        rho = hs[shell].sum() / max(vs[shell].sum(), 1e-12)
        E = rho * vs
        y = np.log((hs + PSEUDO) / (E + PSEUDO))
        w = float(ess_ratio) / (1.0 / (hs + PSEUDO) + 1.0 / (E + PSEUDO))
        w = w / np.mean(w)
    elif s_existing is not None:
        y = _untaper_empirical_table(t, np.asarray(s_existing, dtype=np.float64))
        w = np.ones_like(y)
    else:
        raise ValueError("Provide either raw histogram h or s_existing")

    if lam_rough <= 0.0 and lam_sharp <= 0.0:
        return y

    sd_r, sd_a_rad = THERMAL_COMPLIANCE_SD[t]
    da_rad = math.radians(da)
    ceil_r = gamma_compliance / (sd_r ** 2)
    ceil_a = gamma_compliance / (sd_a_rad ** 2)

    def objective_and_grad(flat: np.ndarray) -> Tuple[float, np.ndarray]:
        F = flat.reshape(nr, na)
        diff = F - y
        loss = float(np.sum(w * diff * diff))
        grad = 2.0 * w * diff

        d2r = F[2:, :] - 2.0 * F[1:-1, :] + F[:-2, :]
        F_pad = np.pad(F, ((0, 0), (1, 1)), mode="symmetric")
        d2a = F_pad[:, 2:] - 2.0 * F + F_pad[:, :-2]

        if lam_rough > 0.0:
            loss += lam_rough * float(np.sum(d2r * d2r) + np.sum(d2a * d2a))
            gr = 2.0 * lam_rough * d2r
            grad[2:, :] += gr
            grad[1:-1, :] -= 2.0 * gr
            grad[:-2, :] += gr

            ga = 2.0 * lam_rough * d2a
            grad -= 2.0 * ga
            grad[:, 1:] += ga[:, :-1]
            grad[:, 0] += ga[:, 0]
            grad[:, :-1] += ga[:, 1:]
            grad[:, -1] += ga[:, -1]

        if lam_sharp > 0.0:
            curv_r = -d2r / (dr * dr)
            excess_r = np.maximum(0.0, curv_r - ceil_r)
            if np.any(excess_r > 0.0):
                scale_r = lam_sharp / (ceil_r * ceil_r)
                loss += scale_r * float(np.sum(excess_r * excess_r))
                gr_s = -2.0 * scale_r * excess_r / (dr * dr)
                grad[2:, :] += gr_s
                grad[1:-1, :] -= 2.0 * gr_s
                grad[:-2, :] += gr_s

            curv_a = -d2a / (da_rad * da_rad)
            excess_a = np.maximum(0.0, curv_a - ceil_a)
            if np.any(excess_a > 0.0):
                scale_a = lam_sharp / (ceil_a * ceil_a)
                loss += scale_a * float(np.sum(excess_a * excess_a))
                ga_s = -2.0 * scale_a * excess_a / (da_rad * da_rad)
                grad -= 2.0 * ga_s
                grad[:, 1:] += ga_s[:, :-1]
                grad[:, 0] += ga_s[:, 0]
                grad[:, :-1] += ga_s[:, 1:]
                grad[:, -1] += ga_s[:, -1]

        return loss, grad.ravel()

    res = minimize(
        objective_and_grad,
        y.ravel().copy(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
    )
    return res.x.reshape(nr, na)


def _solve_radial_c2_slopes(Y: np.ndarray, dr: float) -> np.ndarray:
    """Solve for radial C^2 slopes M (..., nr, na) with quartic initial segment (f'(r0)=f''(r0)=0)."""
    *lead, nr, nm = Y.shape
    Y2 = Y.reshape(-1, nr, nm)
    B = Y2.shape[0]
    M = np.zeros_like(Y2, dtype=np.float64)

    N = nr - 1
    diag = np.full(N, 4.0, dtype=np.float64)
    diag[0] = 5.0
    diag[-1] = 2.0
    upper = np.ones(N - 1, dtype=np.float64)
    lower = np.ones(N - 1, dtype=np.float64)

    rhs = np.zeros((B, N, nm), dtype=np.float64)
    rhs[:, 0, :] = (6.0 * (Y2[:, 1, :] - Y2[:, 0, :]) + 3.0 * (Y2[:, 2, :] - Y2[:, 1, :])) / dr
    rhs[:, 1:-1, :] = 3.0 * (Y2[:, 3:, :] - Y2[:, 1:-2, :]) / dr
    rhs[:, -1, :] = 3.0 * (Y2[:, -1, :] - Y2[:, -2, :]) / dr

    c_star = np.zeros(N - 1, dtype=np.float64)
    d_star = np.zeros((B, N, nm), dtype=np.float64)
    c_star[0] = upper[0] / diag[0]
    d_star[:, 0, :] = rhs[:, 0, :] / diag[0]
    for i in range(1, N):
        denom = diag[i] - lower[i - 1] * c_star[i - 1]
        if i < N - 1:
            c_star[i] = upper[i] / denom
        d_star[:, i, :] = (rhs[:, i, :] - lower[i - 1] * d_star[:, i - 1, :]) / denom

    x = np.zeros((B, N, nm), dtype=np.float64)
    x[:, -1, :] = d_star[:, -1, :]
    for i in range(N - 2, -1, -1):
        x[:, i, :] = d_star[:, i, :] - c_star[i] * x[:, i + 1, :]

    M[:, 1:, :] = x
    return M.reshape(*lead, nr, nm)


def build_c2_surface_params(t: str, F_grid: np.ndarray, s_clip: float = S_CLIP) -> dict:
    """Convert pre-taper bin-center values F_grid (K, nr, na) or (nr, na) into C^2 radial-Chebyshev spline parameters."""
    from scipy.fft import dct

    single = F_grid.ndim == 2
    F = F_grid[None, :, :] if single else np.asarray(F_grid, dtype=np.float64)
    r_lo, dr, nr, a_lo, da, na = _grid(t)
    raw_dct = dct(F, type=2, axis=-1, norm=None)
    C = raw_dct / float(na)
    C[:, :, 0] *= 0.5
    M = _solve_radial_c2_slopes(C, dr)
    return dict(
        type=t,
        C=C[0] if single else C,
        M=M[0] if single else M,
        grid=(r_lo, dr, nr, a_lo, da, na),
        win=SPEC[t]["win"],
        s_clip=s_clip,
    )


def fit_table(t: str, h: np.ndarray, regularize: bool = True) -> dict:
    """s = ln g on bin centres (nr, na), tapered; g normalised to 1 over the shell; f = regularized pre-taper grid."""
    r_lo, dr, nr, a_lo, da, na = _grid(t)
    rc = r_lo + dr * (np.arange(nr) + 0.5)
    vol = bin_volumes(t)
    k = np.ones((3, 3)) / 9.0
    from scipy.signal import convolve2d

    sm = lambda x: convolve2d(x, k, mode="same", boundary="symm")
    hs, vs = sm(h), sm(vol)
    shell = (rc >= SPEC[t]["shell"][0]) & (rc < SPEC[t]["shell"][1])
    rho = hs[shell].sum() / max(vs[shell].sum(), 1e-12)
    E = rho * vs
    s = np.clip(np.log((hs + PSEUDO) / (E + PSEUDO)), -S_CLIP, S_CLIP)
    s = s * _taper(rc, *SPEC[t]["win"])[:, None]
    out = dict(type="nci", n=int(round(h.sum())), r_lo=r_lo, dr=dr, a_lo=a_lo, da=da,
               s=[[round(float(x), 4) for x in row] for row in s])
    if regularize:
        F = fit_empirical_surface(t, h=h, lam_rough=0.05, lam_sharp=2.0, gamma_compliance=GAMMA_COMPLIANCE[t])
        out["f"] = [[round(float(x), 4) for x in row] for row in F]
    return out


def fit_tables(hist: Dict[str, np.ndarray], n_min: int = N_MIN) -> Dict[str, dict]:
    out = {}
    for lab, h in hist.items():
        if h.sum() < n_min:
            continue
        key = tuple(lab.split("|"))
        out[f"NCI:{key!r}"] = fit_table(key[0], h)
    return out


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
    cands = candidates(mol)
    rt = {}
    enabled = ACTIVE_TYPES if types is None else tuple(types)
    if any(t not in TYPES for t in enabled):
        raise ValueError(f"unknown NCI types: {enabled}")
    nb = None
    bound = None
    for t in TYPES:
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
                    f_rows.append(_untaper_empirical_table(t, s_arr))
            keep.append(k)
            cand_row.append(row_of[e[0]])
        if not keep:
            continue
        r_lo, dr, nr, a_lo, da, na = _grid(t)
        c2_params = build_c2_surface_params(t, np.stack(f_rows, axis=0), s_clip=S_CLIP)
        idx_t = index_tensors(cands, pos, device, keep={**{x: [] for x in TYPES}, t: keep})[t]
        kept_cs = [cands[t][k] for k in keep]
        entry = dict(
            T=torch.tensor(np.array(rows, dtype=np.float32), device=device),
            C=torch.tensor(c2_params["C"], dtype=torch.float32, device=device),
            M=torch.tensor(c2_params["M"], dtype=torch.float32, device=device),
            row=torch.tensor(cand_row, dtype=torch.long, device=device),
            idx=idx_t,
            grid=(r_lo, dr, nr, a_lo, da, na),
            win=SPEC[t]["win"],
            s_clip=float(S_CLIP),
            scale=float(TYPE_SCALES.get(t, 1.0)),
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


def evaluate_c2_from_rz_torch(C, M, row, r, z, grid, win, s_clip: float = S_CLIP):
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
    taper = quintic_taper_torch(r, win_lo, win_hi)
    return f_sat * taper


def _eval_pp_c5(e: dict, X):
    """Evaluate C5 stacking term with normal/slip decomposition and co-bound haptic face exclusion."""
    import torch

    c, n = _ring_frame(X, e["idx"]["rings"])
    ci, cj = c[:, e["idx"]["ri"]], c[:, e["idx"]["rj"]]
    ni, nj = n[:, e["idx"]["ri"]], n[:, e["idx"]["rj"]]
    v = ci - cj
    r = v.norm(dim=-1)
    dot = (ni * nj).sum(-1)
    z = (2.0 * (dot * dot) - 1.0).clamp(-1.0, 1.0)
    s_base = evaluate_c2_from_rz_torch(e["C"], e["M"], e["row"], r, z, e["grid"], e["win"], e.get("s_clip", S_CLIP))

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

    r, z = features_cos_torch("chp", X, e["idx"])
    s_raw = evaluate_c2_from_rz_torch(e["C"], e["M"], e["row"], r, z, e["grid"], e["win"], e.get("s_clip", S_CLIP))
    if "parent_idx" not in e:
        return s_raw.sum(dim=-1)

    h_idx = e["idx"]["h"]
    ring_idx = e["idx"]["ring"]
    parent_idx = e["parent_idx"]
    c_ring, _ = _ring_frame(X, e["idx"]["rings"])
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
            r, a = features(t, X, e["idx"])
            tot = tot + scale * _bilinear(e["T"], e["row"], r, a, e["grid"]).sum(-1)
            continue
        if t == "hb":
            r, z = features_cos_torch("hb", X, e["idx"])
            s = evaluate_c2_from_rz_torch(e["C"], e["M"], e["row"], r, z, e["grid"], e["win"], e.get("s_clip", S_CLIP)).sum(-1)
        elif t == "pp":
            s = _eval_pp_c5(e, X)
        else:
            s = _eval_chp_c3_c4(e, X)
        tot = tot + scale * s
    return tot
