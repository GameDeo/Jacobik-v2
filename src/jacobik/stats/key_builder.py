"""
key_builder.py
==============
Produces raw (key_tuple, value) observations from a TMComplex for each
geometric quantity defined in DESIGN.md §4.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from jacobik.core.types import CoordinationSite, TMComplex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

KeyValue = Tuple[tuple, float]  # one raw (key, value) observation

# ---------------------------------------------------------------------------
# Metal period group lookup
# ---------------------------------------------------------------------------

_PERIOD_GROUP: Dict[str, str] = {
    # Period 4 d-block
    **{s: "period4_3d" for s in ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")},
    # Period 5 d-block
    **{s: "period5_3d" for s in ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd")},
    # Period 6 d-block
    **{s: "period6_3d" for s in ("Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg")},
    # Lanthanides
    **{
        s: "lanthanide"
        for s in (
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        )
    },
    # Actinides
    **{s: "actinide" for s in ("Ac", "Th", "Pa", "U", "Np", "Pu")},
}

METAL_ANY = "metal_any"


def metal_period_group(symbol: str) -> str:
    """
    Return the period-group label for a metal symbol.
    Returns METAL_ANY for unknowns (graceful fallback).
    """
    return _PERIOD_GROUP.get(symbol, METAL_ANY)


# ---------------------------------------------------------------------------
# Graph distance classification
# ---------------------------------------------------------------------------

_DIST_CHELATE_SHORT_MAX = 4
_DIST_CHELATE_LONG_MAX = 8

# Sentinel for 'no path found within the ligand graph'
_NO_PATH = 10_000


def _graph_dist_class(path_len: int, same_ligand: bool) -> str:
    """Map a shortest-path length to a DESIGN.md graph-distance class string."""
    if not same_ligand:
        return "different_ligand"
    if path_len <= _DIST_CHELATE_SHORT_MAX:
        return "chelate_short"
    if path_len <= _DIST_CHELATE_LONG_MAX:
        return "chelate_long"
    return "same_remote"


# ---------------------------------------------------------------------------
# Local bond atom type vocabulary (Q5)
# ---------------------------------------------------------------------------

# BO thresholds for hybridisation inference from quantized bond orders.
_BO_AROM = 1.5
_BO_DBL = 2.0
_BO_AROM_TOL = 0.15  # |BO - 1.5| < tol -> aromatic
_BO_DBL_TOL = 0.15  # |BO - 2.0| < tol -> double

# Elements for which we resolve hybridisation; others fall back to symbol.
_HYBRIDISE_ELEMS: FrozenSet[str] = frozenset({"C", "N", "O", "S", "P"})


# ---------------------------------------------------------------------------
# Q6 — parameters
# ---------------------------------------------------------------------------

# Hard spatial cutoff (Å).  Pairs further apart than this are skipped before
# any distance computation in the inner loop.  6 Å comfortably covers all
# sterically relevant contacts (sum of heaviest vdW radii ~4 Å) while
# eliminating the long-range O(n²) tail that dominates large complexes.
DIST_CUTOFF_Q6: float = 6.0

# See exp.md for detailed rationale
Q6_PAIR_CAP: Dict[Tuple[str, str], int] = {
    # Heavy–heavy (most abundant)
    ("C", "C"): 30,
    ("C", "N"): 25,
    ("C", "O"): 25,
    ("C", "P"): 20,
    ("C", "S"): 20,
    ("N", "N"): 20,
    # H-involving (very abundant due to ligand H count; capped tighter)
    ("C", "H"): 20,
    ("H", "H"): 10,
    ("H", "N"): 15,
    ("H", "O"): 15,
    ("H", "P"): 12,
    ("H", "S"): 12,
}


def _local_type(atom_idx: int, bo_neighbors: List[Tuple[int, float]]) -> str:
    """
    Assign a local bond atom type token from the heavy-atom neighbors and
    their quantized bond orders.

    Parameters
    ----------
    atom_idx    : 1-based xyz index (used only for debug)
    bo_neighbors: [(neighbor_idx, bond_order), ...] from the quantized bond
                  graph, metal excluded, H excluded.
    """
    # We need the element symbol — callers pass it separately; this function
    # receives it via the closure in _build_type_map.
    raise NotImplementedError("Use _build_type_map; not called directly.")


def _build_type_map(
    mol: TMComplex,
) -> Dict[int, str]:
    """
    Build a dict mapping atom_idx -> local bond atom type string for every
    non-metal, non-H atom in mol.
    """
    sym_of: Dict[int, str] = {a.idx: a.symbol for a in mol.atoms}
    metal_idx = mol.metal_idx

    # Build neighbor list: idx -> [(neighbor_idx, quantized_bo)]
    neighbors: Dict[int, List[Tuple[int, float]]] = {a.idx: [] for a in mol.atoms}
    for b in mol.bonds:
        if b.i == metal_idx or b.j == metal_idx:
            continue
        si = sym_of.get(b.i, "")
        sj = sym_of.get(b.j, "")
        if si == "H" or sj == "H":
            continue
        neighbors[b.i].append((b.j, b.order))
        neighbors[b.j].append((b.i, b.order))

    type_map: Dict[int, str] = {}
    for a in mol.atoms:
        if a.idx == metal_idx or a.symbol == "H":
            continue
        sym = a.symbol
        if sym not in _HYBRIDISE_ELEMS:
            type_map[a.idx] = sym
            continue

        bos = [bo for _, bo in neighbors[a.idx]]
        has_arom = any(abs(bo - _BO_AROM) < _BO_AROM_TOL for bo in bos)
        has_dbl = any(abs(bo - _BO_DBL) < _BO_DBL_TOL for bo in bos)

        if sym == "C":
            if has_arom:
                type_map[a.idx] = "C_ar"
            elif has_dbl:
                type_map[a.idx] = "C_sp2"
            else:
                type_map[a.idx] = "C_sp3"
        elif sym == "N":
            if has_arom:
                type_map[a.idx] = "N_ar"
            elif has_dbl:
                type_map[a.idx] = "N_sp2"
            else:
                type_map[a.idx] = "N_sp3"
        elif sym == "O":
            if has_dbl:
                type_map[a.idx] = "O_sp2"
            else:
                type_map[a.idx] = "O_sp3"
        elif sym == "S":
            if has_arom:
                type_map[a.idx] = "S_ar"
            elif has_dbl:
                type_map[a.idx] = "S_sp2"
            else:
                type_map[a.idx] = "S_sp3"
        elif sym == "P":
            if has_dbl:
                type_map[a.idx] = "P_sp2"
            else:
                type_map[a.idx] = "P_sp3"
        else:
            type_map[a.idx] = sym

    return type_map


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------


def _vec(mol: TMComplex, idx: int) -> np.ndarray:
    """Return the position of atom idx as a float64 array."""
    return np.asarray(mol.get_atom(idx).position, dtype=float)


def _angle_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Compute the angle a–vertex–b in degrees.
    Returns None if either vector is degenerate (near-zero length).
    """
    v1 = a - vertex
    v2 = b - vertex
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cos_t = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def _dihedral_deg(
    p_i: np.ndarray,
    p_j: np.ndarray,
    p_k: np.ndarray,
    p_l: np.ndarray,
) -> Optional[float]:
    """
    Signed dihedral angle i–j–k–l in degrees, range (–180, +180].
    Returns None if the geometry is degenerate.
    """
    b1 = p_j - p_i
    b2 = p_k - p_j
    b3 = p_l - p_k
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)
    if n1_len < 1e-8 or n2_len < 1e-8:
        return None
    n1 /= n1_len
    n2 /= n2_len
    cos_t = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    sin_t = float(np.dot(np.cross(n1, n2), b2 / (np.linalg.norm(b2) + 1e-12)))
    return float(np.degrees(np.arctan2(sin_t, cos_t)))


# ---------------------------------------------------------------------------
# Ligand graph utilities
# ---------------------------------------------------------------------------


def _build_ligand_graph(mol: TMComplex) -> nx.Graph:
    from jacobik.topology.graph import build_heavy_ligand_graph

    return build_heavy_ligand_graph(mol.atoms, mol.bonds, mol.metal_idx)


def _site_representative_idx(site: CoordinationSite) -> int:
    """Return a single representative atom index for a coordination site."""
    return min(site.donor_indices)


def _site_subclass(site: CoordinationSite, mol: TMComplex) -> str:
    """
    Return the donor subclass for a CoordinationSite.
    For kappa: use the single donor atom's DonorLabel.
    For haptic: use the first (min-index) donor atom's DonorLabel.
    Falls back to the site label string if no DonorLabel is available.
    """
    rep = _site_representative_idx(site)
    dl = mol.donor_labels.get(rep)
    if dl is not None:
        return dl.subclass
    return site.label


def _site_pair_key(
    site_a: CoordinationSite,
    site_b: CoordinationSite,
    mol: TMComplex,
) -> Tuple[str, str]:
    """
    Return canonically-ordered (subclass_A, subclass_B) for a donor pair.
    Ordering: alphabetical on subclass, then min-donor-index as tie-break.
    """
    sub_a = _site_subclass(site_a, mol)
    sub_b = _site_subclass(site_b, mol)
    idx_a = _site_representative_idx(site_a)
    idx_b = _site_representative_idx(site_b)
    if (sub_a, idx_a) <= (sub_b, idx_b):
        return sub_a, sub_b
    return sub_b, sub_a


def _shortest_path_between_sites(
    site_a: CoordinationSite,
    site_b: CoordinationSite,
    G: nx.Graph,
) -> int:
    """
    Shortest path length between two coordination sites through the ligand
    graph (metal excluded).
    """
    indices_a = [idx for idx in site_a.donor_indices if G.has_node(idx)]
    indices_b = [idx for idx in site_b.donor_indices if G.has_node(idx)]

    if not indices_a or not indices_b:
        return _NO_PATH

    best = _NO_PATH
    for a in indices_a:
        for b in indices_b:
            if a == b:
                return 0
            try:
                d = nx.shortest_path_length(G, a, b)
                if d < best:
                    best = d
            except nx.NetworkXNoPath:
                pass
            except nx.NodeNotFound:
                pass
    return best


def _same_ligand(
    site_a: CoordinationSite,
    site_b: CoordinationSite,
    mol: TMComplex,
) -> bool:
    """Return True if both sites have at least one donor atom in the same ligand."""
    lig_sets = []
    for site in (site_a, site_b):
        ligs: Set[int] = set()
        for donor_idx in site.donor_indices:
            for lig in mol.ligands:
                if donor_idx in lig.atom_indices:
                    ligs.add(lig.ligand_id)
        lig_sets.append(ligs)
    return bool(lig_sets[0] & lig_sets[1])


# ---------------------------------------------------------------------------
# Q1 — Bond lengths
# ---------------------------------------------------------------------------

#: Geometries for which a trans partner is not well-defined.
#: CN≤2 and "other" are excluded; "other" has no clear trans axis.
_NO_TRANS_GEOMS = frozenset({"linear", "other"})

#: Sentinel used when trans partner cannot be assigned.
TRANS_NONE = "none"


def _trans_partners(mol: TMComplex) -> Dict[int, str]:
    """
    Build a map from coordination-site index → trans-partner donor subclass.


    Parameters
    ----------
    mol : TMComplex
        Must have coord_sites and coord_geom populated.

    Returns
    -------
    dict mapping site list index (0-based) → trans-partner subclass string

    """
    sites = mol.coord_sites
    n = len(sites)

    if n <= 2 or mol.coord_geom in _NO_TRANS_GEOMS:
        return {i: TRANS_NONE for i in range(n)}

    metal_pos = _vec(mol, mol.metal_idx)

    # Precompute unit vectors from metal to each site
    unit: List[Optional[np.ndarray]] = []
    for site in sites:
        pos = np.asarray(site.position, dtype=float)
        v = pos - metal_pos
        n_ = float(np.linalg.norm(v))
        unit.append(v / n_ if n_ > 1e-8 else None)

    result: Dict[int, str] = {}
    for i in range(n):
        u_i = unit[i]
        if u_i is None:
            result[i] = TRANS_NONE
            continue

        best_cos = 2.0  # seek minimum cosine (= maximum angle)
        best_j = -1
        for j in range(n):
            if j == i:
                continue
            u_j = unit[j]
            if u_j is None:
                continue
            cos_t = float(np.clip(np.dot(u_i, u_j), -1.0, 1.0))
            if cos_t < best_cos:
                best_cos = cos_t
                best_j = j

        # Only assign a trans partner when the angle is genuinely trans-like
        # cos(150°) ≈ -0.866; we require cos < -0.766 (≈ angle > 140°)
        if best_j >= 0 and best_cos < -0.766:
            result[i] = _site_subclass(sites[best_j], mol)
        else:
            result[i] = TRANS_NONE

    return result


def q1_bond_lengths(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, distance_Å) pairs for Q1 (metal–donor bond length).
    """
    if not mol.coord_sites or mol.coord_geom is None:
        return []

    metal_pos = _vec(mol, mol.metal_idx)
    metal = mol.metal_symbol
    geom = mol.coord_geom
    spin = int(mol.spin_state)
    charge = int(mol.charge)
    trans_map = _trans_partners(mol)
    out: List[KeyValue] = []

    for i, site in enumerate(mol.coord_sites):
        try:
            subclass = _site_subclass(site, mol)
            pos = np.asarray(site.position, dtype=float)
            dist = float(np.linalg.norm(pos - metal_pos))
            trans = trans_map.get(i, TRANS_NONE)
            key = (metal, spin, charge, subclass, geom, trans)
            out.append((key, dist))
        except Exception as exc:
            log.debug("Q1 skip site %s: %s", site, exc)

    return out


# ---------------------------------------------------------------------------
# Q2 — D–M–D angles
# ---------------------------------------------------------------------------


def q2_dmd_angles(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, angle_deg) pairs for Q2 (donor–metal–donor angle).
    """
    sites = mol.coord_sites
    if len(sites) < 2 or mol.coord_geom is None:
        return []

    G = _build_ligand_graph(mol)
    metal_pos = _vec(mol, mol.metal_idx)
    metal = mol.metal_symbol
    geom = mol.coord_geom
    out: List[KeyValue] = []

    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            try:
                sa, sb = sites[i], sites[j]

                # Angle
                pa = np.asarray(sa.position, dtype=float)
                pb = np.asarray(sb.position, dtype=float)
                angle = _angle_deg(pa, metal_pos, pb)
                if angle is None:
                    continue

                # Graph distance class
                same = _same_ligand(sa, sb, mol)
                path_len = _shortest_path_between_sites(sa, sb, G)
                dist_class = _graph_dist_class(path_len, same)

                # Canonical donor pair
                sub_a, sub_b = _site_pair_key(sa, sb, mol)
                key = (geom, dist_class, metal, sub_a, sub_b)
                out.append((key, angle))
            except Exception as exc:
                log.debug("Q2 skip pair (%d,%d): %s", i, j, exc)

    return out


# ---------------------------------------------------------------------------
# Q3 — Donor approach angles
# ---------------------------------------------------------------------------


def q3_approach_angles(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, angle_deg) pairs for Q3 (donor approach angle).
    """
    if not mol.coord_sites or mol.coord_geom is None:
        return []

    atom_map = {a.idx: a for a in mol.atoms}
    metal_pos = _vec(mol, mol.metal_idx)
    metal_idx = mol.metal_idx
    metal = mol.metal_symbol
    geom = mol.coord_geom

    # Build full ligand-graph neighbor table (no metal, includes H) for kappa
    from jacobik.topology.graph import get_ligand_neighbors

    all_neighbors = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)

    out: List[KeyValue] = []

    for site in mol.coord_sites:
        try:
            subclass = _site_subclass(site, mol)

            if site.is_haptic:
                # ── haptic tilt angle ────────────────────────────────────────
                face_idxs = [
                    idx
                    for idx in site.donor_indices
                    if idx != metal_idx
                    and atom_map.get(idx) is not None
                    and atom_map[idx].symbol != "H"
                ]
                if len(face_idxs) < 3:
                    continue  # need at least 3 atoms to define a plane

                face_pts = np.array([atom_map[idx].position for idx in face_idxs], dtype=float)
                centroid = face_pts.mean(axis=0)

                # Best-fit ring-plane normal via SVD
                centered = face_pts - centroid
                _, _, Vt = np.linalg.svd(centered)
                normal = Vt[-1]  # smallest singular vector = plane normal

                # Metal–centroid unit vector
                mc = centroid - metal_pos
                mc_len = np.linalg.norm(mc)
                if mc_len < 1e-8:
                    continue
                u_mc = mc / mc_len

                # Orient normal toward metal
                if np.dot(normal, u_mc) < 0:
                    normal = -normal

                # Tilt = angle between normal and metal-centroid direction
                cos_t = float(np.clip(np.dot(normal, u_mc), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cos_t)))
                out.append(((subclass, metal, geom), angle))

            else:
                # ── kappa approach angle ─────────────────────────────────────
                donor_idx = site.donor_indices[0]
                nbs = all_neighbors.get(donor_idx, [])
                if not nbs:
                    continue  # k = 0: skip

                donor_pos = _vec(mol, donor_idx)

                sub_positions = np.array([atom_map[n].position for n in nbs], dtype=float)
                centroid = sub_positions.mean(axis=0)

                v_M = metal_pos - donor_pos
                v_C = centroid - donor_pos

                len_M = np.linalg.norm(v_M)
                len_C = np.linalg.norm(v_C)
                if len_M < 1e-8 or len_C < 1e-8:
                    continue

                cos_t = float(np.clip(np.dot(v_M / len_M, v_C / len_C), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cos_t)))
                out.append(((subclass, metal, geom), angle))

        except Exception as exc:
            log.debug("Q3 skip site %s: %s", site, exc)

    return out


# ---------------------------------------------------------------------------
# Q4 — Donor pyramidalization (M-inclusive)
# ---------------------------------------------------------------------------


def q4_pyramidalization(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, sigma_theta_deg) pairs for Q4 (donor pyramidalization).
    """
    if not mol.coord_sites or mol.coord_geom is None:
        return []

    atom_map = {a.idx: a for a in mol.atoms}
    metal_idx = mol.metal_idx
    metal_pos = _vec(mol, metal_idx)
    geom = mol.coord_geom

    # Build full ligand-graph neighbor table (no metal, includes H)
    from jacobik.topology.graph import get_ligand_neighbors

    all_neighbors = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)

    out: List[KeyValue] = []

    for site in mol.coord_sites:
        if site.is_haptic:
            continue
        try:
            donor_idx = site.donor_indices[0]
            subclass = _site_subclass(site, mol)
            nbs = all_neighbors.get(donor_idx, [])
            k = len(nbs)
            if k < 2:
                continue  # terminal donor — Q4 undefined (would just reproduce Q3)

            donor_pos = _vec(mol, donor_idx)

            # S = {M, X₁, ..., Xₖ}  —  positions of all n = k+1 substituents
            s_positions = [metal_pos] + [np.asarray(atom_map[n].position, dtype=float) for n in nbs]
            n = len(s_positions)  # = k + 1

            sigma_theta = 0.0
            valid = True
            for a_i in range(n):
                for b_i in range(a_i + 1, n):
                    ang = _angle_deg(s_positions[a_i], donor_pos, s_positions[b_i])
                    if ang is None:
                        valid = False
                        break
                    sigma_theta += ang
                if not valid:
                    break

            if not valid:
                continue

            n_pairs = n * (n - 1) // 2
            mean_angle = sigma_theta / n_pairs
            out.append(((subclass, geom), mean_angle))

        except Exception as exc:
            log.debug("Q4 skip site %s: %s", site, exc)

    return out


# ---------------------------------------------------------------------------
# Q5 helpers — canonical flanking-atom selection
# ---------------------------------------------------------------------------

# (Replaced _heaviest_neighbor and _ATOMIC_NUMBER with topology.graph imports where needed)


# ---------------------------------------------------------------------------
# Q5 helpers — chelate ring bond detection
# ---------------------------------------------------------------------------


def _chelate_ring_bonds(mol: TMComplex, G: nx.Graph) -> Set[Tuple[int, int]]:
    """
    Return the set of canonical bond keys (min_idx, max_idx) that lie on
    any shortest ligand-internal path between two donor atoms of the same
    chelate ligand (path length ≤ _DIST_CHELATE_LONG_MAX, i.e. the bond is
    within a chelate ring that closes through the metal).
    """
    ring_bonds: Set[Tuple[int, int]] = set()

    # Collect donor indices per ligand (from coord_sites so we only consider
    # atoms that are actually coordinating).
    lig_to_donor_sites: Dict[int, List[CoordinationSite]] = {}
    for lig in mol.ligands:
        lig_set = set(lig.atom_indices)
        sites_here = [s for s in mol.coord_sites if any(d in lig_set for d in s.donor_indices)]
        if len(sites_here) >= 2:
            lig_to_donor_sites[lig.ligand_id] = sites_here

    for sites in lig_to_donor_sites.values():
        for ii in range(len(sites)):
            for jj in range(ii + 1, len(sites)):
                sa, sb = sites[ii], sites[jj]
                path_len = _shortest_path_between_sites(sa, sb, G)
                if path_len > _DIST_CHELATE_LONG_MAX or path_len == _NO_PATH:
                    continue
                # Find one shortest path and collect its edges.
                # One path per site-pair is sufficient; break as soon as found.
                found = False
                for a_idx in sa.donor_indices:
                    if found:
                        break
                    for b_idx in sb.donor_indices:
                        if not G.has_node(a_idx) or not G.has_node(b_idx):
                            continue
                        try:
                            path = nx.shortest_path(G, a_idx, b_idx)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
                        for step in range(len(path) - 1):
                            u, v = path[step], path[step + 1]
                            ring_bonds.add((min(u, v), max(u, v)))
                        found = True
                        break

    return ring_bonds


# ---------------------------------------------------------------------------
# Q5 — Rotatable bond torsions
# ---------------------------------------------------------------------------


def q5_torsions(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, dihedral_deg) pairs for Q5 (rotatable bond torsion).
    """
    atom_map = {a.idx: a for a in mol.atoms}
    metal_idx = mol.metal_idx
    type_map = _build_type_map(mol)

    # Chelate ring bond set — built once for the whole complex.
    G_lig = _build_ligand_graph(mol)
    chelate_bonds = _chelate_ring_bonds(mol, G_lig)

    # Heavy-atom neighbors (no metal, no H)
    from jacobik.topology.graph import get_heavy_ligand_neighbors, heaviest_heavy_neighbor

    heavy_neighbors = get_heavy_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)

    seen_bonds: Set[Tuple[int, int]] = set()
    out: List[KeyValue] = []

    for lig in mol.ligands:
        for chain in lig.pivot_chains:
            for joint in chain.joints:
                j_idx = joint.atom_i  # donor-proximal end
                k_idx = joint.atom_j  # tail end
                bond_key = (min(j_idx, k_idx), max(j_idx, k_idx))
                if bond_key in seen_bonds:
                    continue
                seen_bonds.add(bond_key)

                try:
                    # Flanking atoms: heaviest heavy neighbor of j (not k) and
                    # heaviest heavy neighbor of k (not j)
                    j_nbs = [n for n in heavy_neighbors.get(j_idx, []) if n != k_idx]
                    k_nbs = [n for n in heavy_neighbors.get(k_idx, []) if n != j_idx]

                    if not j_nbs or not k_nbs:
                        continue  # terminal bond — no flanking atom on one side

                    i_idx = heaviest_heavy_neighbor(j_nbs, atom_map)
                    l_idx = heaviest_heavy_neighbor(k_nbs, atom_map)

                    # Dihedral i–j–k–l
                    p_i = _vec(mol, i_idx)
                    p_j = _vec(mol, j_idx)
                    p_k = _vec(mol, k_idx)
                    p_l = _vec(mol, l_idx)

                    dihedral = _dihedral_deg(p_i, p_j, p_k, p_l)
                    if dihedral is None:
                        continue

                    sym_i = atom_map[i_idx].symbol if i_idx in atom_map else "?"
                    sym_j = atom_map[j_idx].symbol if j_idx in atom_map else "?"
                    sym_k = atom_map[k_idx].symbol if k_idx in atom_map else "?"
                    sym_l = atom_map[l_idx].symbol if l_idx in atom_map else "?"
                    type_i = type_map.get(i_idx, sym_i)
                    type_j = type_map.get(j_idx, sym_j)
                    type_k = type_map.get(k_idx, sym_k)
                    type_l = type_map.get(l_idx, sym_l)

                    ring_ctx = "chelate" if bond_key in chelate_bonds else "free"
                    key = (type_j, type_k, type_i, type_l, ring_ctx)
                    out.append((key, dihedral))

                except Exception as exc:
                    log.debug("Q5 skip joint %s: %s", joint, exc)

    return out


# ---------------------------------------------------------------------------
# Q6 — Steric contacts
# ---------------------------------------------------------------------------


def q6_steric_contacts(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, distance_Å) pairs for Q6 (pairwise non-bonded contacts).
    """
    import random

    atom_map = {a.idx: a for a in mol.atoms}
    metal_idx = mol.metal_idx

    # All non-metal atoms (H included this time)
    all_atoms = [a for a in mol.atoms if a.idx != metal_idx]
    n = len(all_atoms)
    if n < 2:
        return []

    # Build position matrix for vectorised distance filtering
    positions = np.array([a.position for a in all_atoms], dtype=np.float64)  # (n, 3)

    # Build ligand bond graph — heavy atoms only (no H, no metal).
    from jacobik.topology.graph import build_heavy_ligand_graph, get_steric_exclusions

    G = build_heavy_ligand_graph(mol.atoms, mol.bonds, mol.metal_idx)
    close = get_steric_exclusions(mol.atoms, mol.bonds, mol.metal_idx, G)

    # Collect candidates: all pairs within DIST_CUTOFF_Q6 that are not 1,2/1,3
    # Accumulate per key into lists for reservoir sampling
    buckets: Dict[Tuple[str, str], List[float]] = {}

    cutoff_sq = DIST_CUTOFF_Q6 * DIST_CUTOFF_Q6
    for ii in range(n):
        ai = all_atoms[ii]
        pos_i = positions[ii]
        for jj in range(ii + 1, n):
            aj = all_atoms[jj]

            # Fast squared-distance check before anything else
            diff = pos_i - positions[jj]
            dsq = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            if dsq > cutoff_sq:
                continue

            pair = (ai.idx, aj.idx)
            if pair in close:
                continue

            dist = float(dsq**0.5)
            elem_a, elem_b = (
                (ai.symbol, aj.symbol) if ai.symbol <= aj.symbol else (aj.symbol, ai.symbol)
            )
            key = (elem_a, elem_b)

            if key not in buckets:
                buckets[key] = []
            buckets[key].append(dist)

    # Apply per-key cap via reservoir sampling, then flatten
    out: List[KeyValue] = []
    for key, distances in buckets.items():
        cap = Q6_PAIR_CAP.get(key)
        if cap is not None and len(distances) > cap:
            distances = random.sample(distances, cap)
        for d in distances:
            out.append((key, d))

    return out


# ---------------------------------------------------------------------------
# Convenience: produce all keys for a complex in one call
# ---------------------------------------------------------------------------


def all_keys(mol: TMComplex) -> Dict[str, List[KeyValue]]:
    """
    Return a dict mapping quantity name to its list of (key, value) pairs.
    """
    result: Dict[str, List[KeyValue]] = {}
    for name, fn in (
        ("Q1", q1_bond_lengths),
        ("Q2", q2_dmd_angles),
        ("Q3", q3_approach_angles),
        ("Q4", q4_pyramidalization),
        ("Q5", q5_torsions),
        ("Q6", q6_steric_contacts),
    ):
        try:
            result[name] = fn(mol)
        except Exception as exc:
            log.error("%s: %s failed: %s", mol.csd_code, name, exc)
            result[name] = []
    return result
