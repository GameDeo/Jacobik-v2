"""
orientation.py
==============
S_orient: the joint donor-orientation quantity ("QO"), which replaces Q3 (approach angle) and Q4
(pyramidalisation) when the constraints DB contains QO fits.

Quantity
--------
Haptic sites (2..8 heavy face atoms; k = 10 + face size) use g = a subset of [tilt, bend, slip] per face size
(HAPTIC_FEATS): the face normal against the centroid->metal direction, the mean lean of the face substituents along
the normal, and the relative spread of the metal-face-atom distances (haptic_features).

For every kappa donor D with k = 1..3 ligand substituents X_i (H included):
    u_M = unit(D -> M),  u_i = unit(D -> X_i)
    g   = [cos(M-D-X_i) for i sorted ascending] + [cos(X_i-D-X_j) for i < j in that sorted order]
dim(g) = 1 / 3 / 6 for k = 1 / 2 / 3. Sorting by cos(M-D-X) makes g invariant to relabelling equivalent
substituents. The score of one donor is log sum_m pi_m N(g; mu_m, Sigma_m) / dim(g), a full-covariance
Gaussian mixture, so every donor contributes on the scale of a one-dimensional term.

Keys and ladder
---------------
(k, ring, subclass, metal, geometry, anion bucket), where ring is the size (metal included) of the smallest
chelate ring through the donor: 0 = none, 3..6, 7 = 7 or more (shortest ligand-graph path to another donor + 2).
Lookup walks ``level_keys`` from most to least specific and takes the first fitted key.

Fitting
-------
``fit_orientation``: donors with an intramolecular H...acceptor contact shorter than HB_CUT are excluded (their
orientation is set by the H-bond, which Q6 models), every ladder key with >= N_MIN rows gets a full-covariance GMM
(K = 1..K_MAX by BIC, COV_FLOOR * I ADDED to every covariance as regularisation, components lighter than W_MIN
dropped).

The same ``donor_rows`` function is used by the extractor (training data) and, through ``site_specs``, by the
scorer, so training and scoring keys cannot diverge.
"""
from __future__ import annotations

import logging
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


log = logging.getLogger(__name__)

#: Fitting constants (recorded in the DB entries' "meta" for provenance).
N_MIN = 50            # rows needed to fit a key
HB_CUT = 2.3          # A: donors with an H...acceptor contact below this are excluded from fitting
W_MIN = 0.05          # mixture components lighter than this are dropped (weights renormalised)
K_MAX = 4             # max mixture components (chosen by BIC)
#: Covariance regularisation, passed to sklearn as reg_covar, which ADDS it to every component covariance (it is not
#: a floor: a fitted sd of 0.040 is stored as ~0.064). Kept by owner decision 2026-09-30 (refit spec §1); the name is
#: historical.
COV_FLOOR = 0.05**2
RING_CAP = 7          # chelate ring sizes >= 7 are pooled
DIM = {1: 1, 2: 3, 3: 6}
HAPTIC_K0 = 10        # haptic sites use k = HAPTIC_K0 + face size (heavy face atoms, HAPTIC_FACE_MIN..HAPTIC_FACE_MAX)
HAPTIC_FACE_MIN = 2   # eta2 (alkene, alkyne, C=O ...) included since v3
HAPTIC_FACE_MAX = 8
#: Haptic face features, always all computed and stored (g0..g2); each face size is fitted / scored on a subset.
HAPTIC_FEATURES = ("tilt", "bend", "slip")
#: Feature subset per face size (indices into HAPTIC_FEATURES), decided on held-out data: a feature the other two
#: predict with held-out R^2 >= 0.8 (boosted trees, 14,253 haptic rows from tmQM X1-X3 s000-s002 + LnQM, 2026-09-27)
#: is dropped. eta2: tilt and slip both measure the offset along the bond (R^2 0.96 / 0.99), keep [bend, slip];
#: eta3: tilt from bend + slip R^2 0.83, keep [bend, slip]; eta4-6: tilt 0.05-0.36 / slip 0.64-0.76, keep all three;
#: eta7-8 (33 rows) keep all three by default.
HAPTIC_FEATS: Dict[int, Tuple[int, ...]] = {2: (1, 2), 3: (1, 2), **{n: (0, 1, 2) for n in range(4, HAPTIC_FACE_MAX + 1)}}
HAPTIC_DIM = 2        # legacy DBs: haptic entries without "feats" are [tilt, bend]
#: Per-substituent haptic classes: the same face features with "bend" measured for ONE substituent instead of the
#: face mean (a face-mean bend lets one H swing while the mean stays typical). Substituents are split H vs heavy
#: (eta2, tmQM+LnQM: H bend sd 0.25, heavy sd 0.57, so a pooled fit would let H swing). Class k = face k + offset;
#: rows come from the stored raw face geometry (per_sub_rows), so no re-extraction.
PER_SUB_OFFSET = {"H": 100, "X": 200}
PER_SUB_H_MAX = 1.25  # A: a substituent closer than this to its face atom is an H
#: eta2 face plane (production since the eta2 fold-in, 2026-11-01): when each face atom carries >= 2 substituents (alkene, eta2 C-C of a
#: ring), the face normal is the normal of the plane through the bond that best fits the substituent bonds
#: (eta2_plane_normal), not the metal direction; the metal only sets its sign. A metal-derived normal makes tilt
#: identically 1 and moves any off-normal metal into each substituent's bend, so a pyramidal start can fold a geminal
#: pair while each H still scores as crystal-like (NUWFAM). Crystal survey (tmQM + LnQM eta2 faces, 2026-11-01): 8100
#: faces qualify; bonded faces (both M-C < 2.4 A) have tilt p1 0.91, and crystal H bent > 20 deg towards the metal
#: drop from 3.5 % to 0.08 %. Faces with an atom carrying < 2 substituents (alkyne, nitrile, C=O, O2 ...) put the metal
#: near the substituent plane (tilt median 0.05-0.15) and keep the metal normal.
ETA2_FACE_PLANE = True
#: The face plane changes the eta2 features: tilt becomes the metal's angle off the face normal (before it only
#: carried the offset along the bond, like slip). The R^2 rule on all eta2 rows said drop slip (0.89 from the other
#: two), but those rows include loader contact artefacts (below); without slip a metal bound to one end of the C=C,
#: edge-on in the alkene plane, scored as normal (NUWFAM dev, 2026-11-01). All three are kept (owner, 2026-11-01).
#: Bonded eta2 faces (survey and diagnostics only; not applied in the fit): every face atom i has
#: |M - F_i| / (r_cov(M) + r_cov(F_i)) <= ETA2_BOND_RATIO_MAX (loader radii, jacobik.topology.mapping; elements from the
#: extracted raw_face_elems). The cut is the eta5 p99 (1.21; eta5 faces are bonded). The eta2 class holds M...C contacts
#: the loader types as eta2 (C=C plane faces: 12 % of rows, 90 % of the edge-on ones) and kappa-X donors next to a
#: C=X carbon typed as eta2 C=X (C=S 93 %, C=N / C=O ~30 % unbonded). Cleaning them is the far-field / typing
#: package's job; a fit-time cut (DB v2-v4) was dropped at the fold-in (owner, 2026-11-01).
ETA2_BOND_RATIO_MAX = 1.21


def eta2_bonded(raw_face, metal: str, elems: Sequence[str]) -> bool:
    """Eta2 face (coordinates relative to the metal, face atom elements) passes the bonded-face fit rule above."""
    from jacobik.core.chemistry import covalent_radius as _covalent_radius

    F = np.asarray(raw_face, float).reshape(-1, 3)
    rs = np.array([_covalent_radius(metal) + _covalent_radius(e) for e in elems])
    return bool((np.linalg.norm(F, axis=1) / rs).max() <= ETA2_BOND_RATIO_MAX)
HAPTIC_FEATS = {**HAPTIC_FEATS, 2: (0, 1, 2)}  # eta2 under the face plane: tilt, bend and slip (above)


def face_size(k: int) -> int:
    """Face size of a haptic class k (face or per-substituent class)."""
    return (k - HAPTIC_K0) % 100


def per_sub_class(k: int) -> Optional[str]:
    """'H' / 'X' for a per-substituent haptic class, None otherwise."""
    if k < HAPTIC_K0:
        return None
    return {1: "H", 2: "X"}.get((k - HAPTIC_K0) // 100)


def haptic_feats(k: int) -> Tuple[int, ...]:
    """Feature indices fitted / scored for haptic class k = HAPTIC_K0 + face size."""
    return HAPTIC_FEATS.get(face_size(k), (0, 1, 2))


def feature_cols(k: int) -> List[str]:
    """Row columns holding the features of class k."""
    return [f"g{i}" for i in haptic_feats(k)] if k >= HAPTIC_K0 else [f"g{i}" for i in range(DIM[k])]
ACCEPTORS = frozenset({"N", "O", "F", "Cl", "Br", "I", "S"})
QUANTITY = "QO"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
def level_keys(sub: str, metal: str, geom: str, anion, k: int, ring: int) -> List[tuple]:
    """Ladder keys for one donor, most specific first."""
    from jacobik.stats import ladders as L

    bkt, fam = L.bucket_family(sub)
    pg = L.metal_period_group(metal)
    lv = [(sub, metal, geom, anion), (sub, metal, anion), (sub, anion), (sub, metal, geom), (sub, metal)]
    if pg != L.METAL_ANY:
        lv.append((sub, pg))
    lv += [(sub,), (bkt,), (fam,)]
    return [(k, ring) + key for key in lv] + [(k, ring), (k,)]


def key_str(key: tuple) -> str:
    """Stable text form of a key (used in the DB and in fit seeds)."""
    return "|".join(str(x) for x in key)


def chelate_ring_sizes(mol) -> Dict[int, int]:
    """donor atom idx -> size of the smallest chelate ring through it (0 = none, capped at RING_CAP)."""
    import networkx as nx

    G = nx.Graph()
    for b in mol.bonds:
        if mol.metal_idx not in (b.i, b.j):
            G.add_edge(b.i, b.j)
    donors = [s.donor_indices[0] for s in mol.coord_sites if not s.is_haptic]
    out: Dict[int, int] = {}
    for d in donors:
        best = 0
        if d in G:
            lengths = nx.single_source_shortest_path_length(G, d, cutoff=RING_CAP - 1)
            for d2 in donors:
                if d2 != d and d2 in lengths:
                    size = lengths[d2] + 2
                    best = size if best == 0 else min(best, size)
        out[d] = min(best, RING_CAP)
    return out


def site_specs(mol) -> List[dict]:
    """
    One dict per eligible kappa donor site: donor atom idx, substituent atom idxs (k = 1..3), and its key parts
    (sub, metal, geom, anion, k, ring). Topology only; no coordinates are read. Used by the scorer.
    """
    from jacobik.stats import keys as KB
    from jacobik.topology.graph import get_ligand_neighbors

    if not mol.coord_sites or mol.coord_geom is None:
        return []
    nbrs = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    rings = chelate_ring_sizes(mol)
    out = []
    for site in mol.coord_sites:
        if site.is_haptic:
            continue
        d = site.donor_indices[0]
        xs = list(nbrs.get(d, []))
        k = len(xs)
        if k < 1 or k > 3:
            continue
        try:
            out.append(dict(
                donor=d, subs=xs, k=k, ring=int(rings.get(d, 0)),
                sub=KB._site_subclass(site, mol), metal=mol.metal_symbol,
                geom=mol.coord_geom or "octahedral",
                anion=KB.anionic_co_donor_bucket(mol, exclude_rep_idx=KB._site_representative_idx(site)),
            ))
        except Exception as exc:  # pragma: no cover - defensive, mirrors key_builder
            log.debug("QO site skipped (%s): %s", getattr(mol, "csd_code", "?"), exc)
    return out


def face_order(face: Sequence[int], adj: Dict[int, set]) -> Tuple[List[int], bool]:
    """Face atoms in ring or chain order along ligand bonds, and whether they close a ring (topology only)."""
    face = list(face)
    fs = set(face)
    deg = {a: len(adj.get(a, set()) & fs) for a in face}
    start = min(face, key=lambda a: (deg[a], a))
    order, seen = [start], {start}
    while len(order) < len(face):
        nxt = sorted((adj.get(order[-1], set()) & fs) - seen)
        if not nxt:  # disconnected face: append the rest in index order
            order += sorted(fs - seen)
            break
        order.append(nxt[0])
        seen.add(nxt[0])
    closed = len(face) >= 3 and order[0] in adj.get(order[-1], set())
    return order, closed


def haptic_site_specs(mol) -> List[dict]:
    """
    One dict per haptic site with HAPTIC_FACE_MIN..HAPTIC_FACE_MAX heavy face atoms: face (ordered), closed (ring), subs as
    (face atom, substituent) pairs (ligand neighbours of face atoms outside the face, H included), and key parts with
    k = HAPTIC_K0 + face size, ring = 0. Topology only.
    """
    from jacobik.stats import keys as KB
    from jacobik.topology.graph import get_ligand_neighbors

    if not mol.coord_sites or mol.coord_geom is None:
        return []
    nbrs = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    adj = {a: set(v) for a, v in nbrs.items()}
    sym = {a.idx: a.symbol for a in mol.atoms}
    out = []
    for site in mol.coord_sites:
        if not site.is_haptic:
            continue
        face = [d for d in site.donor_indices if d != mol.metal_idx and sym.get(d) != "H"]
        if not HAPTIC_FACE_MIN <= len(face) <= HAPTIC_FACE_MAX:
            continue
        order, closed = face_order(face, adj)
        fs = set(face)
        subs = [(a, x) for a in order for x in sorted(adj.get(a, ())) if x not in fs and x != mol.metal_idx]
        try:
            out.append(dict(face=order, closed=closed, subs=subs, k=HAPTIC_K0 + len(face), ring=0,
                            sub=KB._site_subclass(site, mol), metal=mol.metal_symbol,
                            geom=mol.coord_geom or "octahedral",
                            anion=KB.anionic_co_donor_bucket(mol, exclude_rep_idx=KB._site_representative_idx(site))))
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("QO haptic site skipped (%s): %s", getattr(mol, "csd_code", "?"), exc)
    return out


def haptic_features(M: np.ndarray, F: np.ndarray, closed: bool, S: Sequence[Tuple[np.ndarray, np.ndarray]],
                    plane: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None):
    """
    g for one haptic face: [tilt, bend, slip] (HAPTIC_FEATURES).
      c = face centroid, uM = unit(M - c).
      n = face normal oriented towards M: Newell normal sum_i (F_i - c) x (F_{i+1} - c) (ring: cyclic) for >= 3
          atoms; for 2 atoms (eta2) the metal direction with its component along the bond removed, or with
          ETA2_FACE_PLANE the substituent plane through the bond (eta2_plane_normal) when both atoms carry >= 2;
          plane = the face's (face atom, substituent) pairs that define it (default S; a per-substituent bend passes
          one substituent as S and the whole face as plane).
      tilt = n . uM                                  (1 = metal on the normal; tilt or slip lowers it)
      bend = mean over substituents of unit(X - A) . n   (> 0 towards the metal, < 0 bent back / away)
      slip = std(|M - F_i|) / mean(|M - F_i|)            (0 = all face atoms equidistant from the metal)
    None if degenerate. The scorer computes the same quantities in torch.
    """
    c = F.mean(0)
    uM = _unit(M - c)
    if uM is None:
        return None
    if len(F) == 2:
        b = _unit(F[1] - F[0])
        if b is None:
            return None
        n = eta2_plane_normal(F, S if plane is None else plane) if ETA2_FACE_PLANE else None
        if n is None:
            n = _unit(uM - float(uM @ b) * b)
    else:
        P = F - c
        idx = list(range(len(P)))
        nxt = idx[1:] + ([0] if closed else [])
        n = _unit(sum(np.cross(P[i], P[j]) for i, j in zip(idx, nxt)))
    if n is None:
        return None
    if float(n @ uM) < 0:
        n = -n
    bends = [float(_unit(X - A) @ n) for A, X in S if _unit(X - A) is not None]
    d = np.linalg.norm(F - M, axis=1)
    return [float(n @ uM), float(np.mean(bends)) if bends else 0.0, float(d.std() / d.mean())]


def eta2_plane_normal(F: np.ndarray, S: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    """
    Unit normal (unsigned) of the plane that contains the eta2 bond F0-F1 and best fits the substituent bond unit vectors
    unit(X - A) with their bond component removed (uncentred least squares about the bond axis: the principal axis of
    their 2-D scatter, one angle in closed form). Geminal substituents bent the same way cancel, so the plane is the
    face's, not any one substituent's. None when a face atom carries < 2 substituents (the metal normal is kept) or
    the bond is degenerate. S: (face atom position, substituent position) pairs. The solver's torch copy is
    angle_bends.eta2_plane_normal_t.
    """
    b = _unit(F[1] - F[0])
    if b is None:
        return None
    side, us = [0, 0], []
    for A, X in S:
        u = _unit(np.asarray(X, float) - A)
        i = int(np.argmin(np.linalg.norm(F - A, axis=1)))
        if u is None:
            continue
        side[i] += 1
        us.append(u - float(u @ b) * b)
    if min(side) < 2:
        return None
    e1 = _unit(np.cross(b, (1.0, 0.0, 0.0) if abs(b[0]) < 0.9 else (0.0, 1.0, 0.0)))
    e2 = np.cross(b, e1)
    x, y = np.array([u @ e1 for u in us]), np.array([u @ e2 for u in us])
    th = 0.5 * np.arctan2(2.0 * float(x @ y), float(x @ x - y @ y))
    return np.cross(b, np.cos(th) * e1 + np.sin(th) * e2)


# ---------------------------------------------------------------------------
# Features (extraction)
# ---------------------------------------------------------------------------
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else None


def features(M: np.ndarray, D: np.ndarray, X: Sequence[np.ndarray]) -> Optional[List[float]]:
    """g for one donor from metal, donor and substituent positions (None if degenerate)."""
    uM = _unit(M - D)
    us = [_unit(np.asarray(x, float) - D) for x in X]
    if uM is None or any(u is None for u in us):
        return None
    cm = np.array([float(np.dot(uM, u)) for u in us])
    order = np.argsort(cm)
    us = [us[i] for i in order]
    k = len(us)
    return list(cm[order]) + [float(np.dot(us[i], us[j])) for i in range(k) for j in range(i + 1, k)]
