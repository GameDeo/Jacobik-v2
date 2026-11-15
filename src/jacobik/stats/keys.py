"""Statistical key extraction (Q1-Q6, Q5 context/symmetry/flanks) from assembled TMComplex objects."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from jacobik.core.chemistry import OS_UNKNOWN
from jacobik.core.types import CoordinationSite, TMComplex
from jacobik.stats.ladders import ANIONIC_BUCKET_MAX, PAIR_CLASS_NONE, is_anionic_oxygen_donor, pair_class
from jacobik.topology.donor_classifier import _canonical_eta2_subclass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Site & graph distance utilities
# ---------------------------------------------------------------------------
import networkx as nx
import numpy as np
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


KeyValue = Tuple[tuple, float]  # one raw (key, value) observation


_DIST_CHELATE_SHORT_MAX = 4


_DIST_CHELATE_LONG_MAX = 8


# Sentinel for 'no path found within the ligand graph'
_NO_PATH = 10_000


def _graph_dist_class(path_len: int, same_ligand: bool) -> str:
    """Map a shortest-path length to a graph-distance class string (ARCHITECTURE_2026-10-20.md section 4.1)."""
    if not same_ligand:
        return "different_ligand"
    if path_len <= _DIST_CHELATE_SHORT_MAX:
        return "chelate_short"
    if path_len <= _DIST_CHELATE_LONG_MAX:
        return "chelate_long"
    return "same_remote"


# BO thresholds for hybridisation inference from quantized bond orders.
_BO_AROM = 1.5


_BO_DBL = 2.0


_BO_AROM_TOL = 0.15  # |BO - 1.5| < tol -> aromatic


_BO_DBL_TOL = 0.15  # |BO - 2.0| < tol -> double


_BO_TRIPLE_MIN = 2.75  # BO >= this -> triple (ligand bond orders are SMILES-quantised: 1, 1.5, 2, 3)


# Elements for which we resolve hybridisation; others fall back to symbol.
_HYBRIDISE_ELEMS: FrozenSet[str] = frozenset({"C", "N", "O", "S", "P"})


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
        # sp (linear): a triple bond, or two double bonds (allene, CO2-like). Before 2026-10-13 these atoms fell
        # through to "_sp3" (a triple bond is neither aromatic nor double), so alkyne / nitrile / isocyanide rotors
        # were pooled with sp3 rotors in the Q5 keys.
        is_sp = any(bo >= _BO_TRIPLE_MIN for bo in bos) or sum(abs(bo - _BO_DBL) < _BO_DBL_TOL for bo in bos) >= 2

        if sym in ("C", "N") and is_sp:
            type_map[a.idx] = f"{sym}_sp"
            continue
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


def _build_ligand_graph(mol: TMComplex) -> nx.Graph:
    from jacobik.topology.graph import build_heavy_ligand_graph

    return build_heavy_ligand_graph(mol.atoms, mol.bonds, mol.metal_idx)


def _site_representative_idx(site: CoordinationSite) -> int:
    """Return a single representative atom index for a coordination site."""
    return min(site.donor_indices)


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


#: Geometries for which a trans partner is not well-defined.
#: CN≤2 and "other" are excluded; "other" has no clear trans axis.
_NO_TRANS_GEOMS = frozenset({"linear", "other"})


#: Sentinel used when trans partner cannot be assigned.
TRANS_NONE = "none"


def mol_oxidation_state(mol: TMComplex, override: Optional[int] = None) -> int:
    """Oxidation state used in Q1 keys: explicit override, else the complex's, else OS_UNKNOWN."""
    if override is not None:
        return int(override)
    os_ = getattr(mol, "oxidation_state", None)
    return OS_UNKNOWN if os_ is None else int(os_)


# ---------------------------------------------------------------------------
# Q5 torsion context, symmetry & flank helpers
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


#: Aryl-rotor ortho context (2026-10-01, refit spec §7 / B3): a rotor bond between an aromatic carbon and a non-ring
#: partner gets the number of heavy ortho substituents of its aryl side(s) in the key context ("free_o0", "free_o1",
#: "free_o2"; capped at 2). Measured on tmQM (claude/prerefit-checklist-2026-09-30.md B3): the median twist of
#: N_sp3/N_sp2/C_sp2/O_sp3-aryl rotors goes 43/47/40/1 deg (no ortho substituent) -> 80/77/75/63 deg (two).
Q5_ORTHO_MAX = 2


_ARYL_TYPE = "C_ar"


def _ring_path(start: int, target: int, banned: int, heavy_nb, max_len: int = 4) -> bool:
    """True if ``start`` reaches ``target`` within ``max_len`` heavy bonds without passing ``banned``."""
    frontier, seen = {start}, {start, banned}
    for _ in range(max_len):
        nxt = set()
        for a in frontier:
            for b in heavy_nb.get(a, ()):
                if b == target:
                    return True
                if b not in seen:
                    seen.add(b)
                    nxt.add(b)
        frontier = nxt
        if not frontier:
            break
    return False


def aryl_ortho_count(a: int, partner: int, heavy_nb, type_map) -> int:
    """Heavy substituents on the ring atoms ortho to aryl carbon ``a`` (its rotor partner excluded). A neighbour of
    an ortho atom is a ring member when it leads back to ``a`` within 4 bonds avoiding that ortho atom (5-/6-rings);
    anything else (incl. a fused ring's peri atom) counts as a substituent."""
    n = 0
    for o in heavy_nb.get(a, ()):
        if o == partner or type_map.get(o, "") not in ("C_ar", "N_ar"):
            continue
        for x in heavy_nb.get(o, ()):
            if x == a:
                continue
            if not _ring_path(x, a, o, heavy_nb):
                n += 1
    return n


def q5_context(ring_ctx: str, j: int, k: int, heavy_nb, type_map) -> str:
    """Q5 key context: the ring context, plus "_o<n>" (ortho substituents of the aryl side(s), capped at
    Q5_ORTHO_MAX) when a C_ar atom is bonded across the rotor to a non-aromatic atom outside any ring."""
    if ring_ctx != "free":
        return ring_ctx
    tj, tk = type_map.get(j, ""), type_map.get(k, "")
    sides = [(a, b) for a, b, ta, tb in ((j, k, tj, tk), (k, j, tk, tj)) if ta == _ARYL_TYPE]
    if not sides:
        return ring_ctx
    n = sum(aryl_ortho_count(a, b, heavy_nb, type_map) for a, b in sides)
    return f"{ring_ctx}_o{min(n, Q5_ORTHO_MAX)}"


def q5_symmetry_fn(mol: TMComplex, all_nb, type_map):
    """sym(j, k) -> (n_j, n_k): rotor symmetry numbers of the two ends of bond j-k (topology/graph.rotor_symmetry over
    Weisfeiler-Lehman classes of the full graph incl. metal-donor edges). The flank atom of an n-fold symmetric top is
    an arbitrary pick among equivalent atoms, so the physical torsion is defined modulo 360/n."""
    from jacobik.topology.graph import rotor_symmetry, wl_classes

    nb = {i: set(v) for i, v in all_nb.items()}
    for site in mol.coord_sites:
        for d in site.donor_indices:
            nb.setdefault(d, set()).add(mol.metal_idx)
            nb.setdefault(mol.metal_idx, set()).add(d)
    cls = wl_classes([a.idx for a in mol.atoms], nb, {a.idx: a.symbol for a in mol.atoms})

    def planar(a):
        return not str(type_map.get(a, "")).endswith("sp3")

    lin = linear_atoms(type_map)

    def end(a, b):
        # the rotor top of a linear (sp) end is the first non-linear atom along the chain (flank_side, same walk)
        prev, cur, seen = b, a, {b}
        while cur in lin:
            seen.add(cur)
            nxt = [n for n in all_nb.get(cur, []) if n != prev]
            if len(nxt) != 1 or nxt[0] in seen:
                return 1
            prev, cur = cur, nxt[0]
        return rotor_symmetry(cur, prev, nb, cls, planar(cur))

    def sym(j, k):
        return (end(j, k), end(k, j))

    return sym


def fold_shifts_deg(nj: int, nk: int) -> List[float]:
    """The nj * nk equivalent torsion shifts (degrees) of a rotor with end symmetries (nj, nk) (scorer Q5_FOLD)."""
    return [360.0 * (r / nk + t / nj) for r in range(nk) for t in range(nj)]


def linear_atoms(type_map) -> Set[int]:
    """Atoms typed sp ("C_sp", "N_sp"): every neighbour lies on one line, so a torsion cannot be measured from them."""
    return {a for a, t in type_map.items() if str(t).endswith("_sp")}


def flank_side(a: int, b: int, heavy_nb, all_nb, linear: Set[int]):
    """Candidates for the flank of torsion bond a-b on a's side: the neighbours of a other than b (heavy preferred,
    else H). When a is a linear (sp) atom its neighbours are collinear with a-b and the dihedral is undefined
    (rounding noise), so the walk continues along the linear chain to the first non-linear atom and takes ITS other
    neighbours (Ph-C#C-Ph: the ortho C of the far ring). Returns None when the chain ends without one (M-C#C-R seen
    from R, H-C#C-): the rotation about such a bond is not a torsion relative to anything on that side.
    2026-10-13: before this, the flank of an sp atom was its collinear partner (angle 178-180 deg)."""
    prev, cur, seen = b, a, {b}
    while cur in linear:
        seen.add(cur)
        nxt = [n for n in all_nb.get(cur, []) if n != prev]
        if len(nxt) != 1 or nxt[0] in seen:
            return None
        prev, cur = cur, nxt[0]
    cands = [n for n in heavy_nb.get(cur, []) if n != prev] or [n for n in all_nb.get(cur, []) if n != prev]
    return cands or None


def q5_flanks(j_idx, k_idx, heavy_nb, all_nb, atom_map, type_map):
    """
    Canonical flanking atoms (i on j's side, l on k's side).  Heavy neighbours
    are preferred; if a side has none, its H neighbours are used (so a
    hydroxyl / methyl rotor yields an X-C-O-H torsion).  Shared with the runtime
    (ik/geometry.py) so both build identical Q5 keys.  Returns (None, None) if a
    side has no neighbour at all, or (linear sp atoms, see flank_side) no off-axis atom.
    """
    from jacobik.topology.graph import canonical_flank

    heavy_deg = {i: len(v) for i, v in heavy_nb.items()}
    lin = linear_atoms(type_map)
    j_nbs = flank_side(j_idx, k_idx, heavy_nb, all_nb, lin)
    k_nbs = flank_side(k_idx, j_idx, heavy_nb, all_nb, lin)
    if not j_nbs or not k_nbs:
        return None, None
    return (canonical_flank(j_nbs, atom_map, type_map, heavy_deg),
            canonical_flank(k_nbs, atom_map, type_map, heavy_deg))


# ---------------------------------------------------------------------------
# Q1 / Q2 key builders
# ---------------------------------------------------------------------------
from jacobik.core import profile as _profile



HAPTO_LABEL_CANON = _profile.get("hapto_label_canon")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Graph distance classification
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Local bond atom type vocabulary (Q5)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Q6 — parameters
# ---------------------------------------------------------------------------

# Hard spatial cutoff (Å).  Pairs further apart than this are skipped before
# any distance computation in the inner loop.  6 Å comfortably covers all
# sterically relevant contacts (sum of heaviest vdW radii ~4 Å) while
# eliminating the long-range O(n²) tail that dominates large complexes.
DIST_CUTOFF_Q6: float = 6.0





# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Ligand graph utilities
# ---------------------------------------------------------------------------






def _site_subclass(site: CoordinationSite, mol: TMComplex) -> str:
    """Return the key subclass for a site, using the legacy representative when the T1 flag is off."""
    if HAPTO_LABEL_CANON:
        canonical = _canonical_eta2_subclass(site, mol)
        if canonical is not None:
            return canonical
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






# ---------------------------------------------------------------------------
# Q1 — Bond lengths
# ---------------------------------------------------------------------------




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




def q1_site_keys(mol: TMComplex, oxidation_state: Optional[int] = None) -> List[Optional[tuple]]:
    """
    Q1 L0 key for every coordination site (aligned with ``mol.coord_sites``;
    None for a site whose key cannot be built).  Used by the extractor AND the
    runtime scorer so they cannot disagree.
    """
    if not mol.coord_sites or mol.coord_geom is None:
        return [None] * len(mol.coord_sites)
    metal = mol.metal_symbol
    geom = mol.coord_geom
    spin = int(mol.spin_state)
    os_ = mol_oxidation_state(mol, oxidation_state)
    trans_map = _trans_partners(mol)
    keys: List[Optional[tuple]] = []
    for i, site in enumerate(mol.coord_sites):
        try:
            keys.append((metal, spin, os_, _site_subclass(site, mol), geom,
                         trans_map.get(i, TRANS_NONE)))
        except Exception as exc:
            log.debug("Q1 key skip site %s: %s", site, exc)
            keys.append(None)
    return keys


# Q2 — D–M–D angles
# ---------------------------------------------------------------------------


def q2_pair_classes(mol: TMComplex, site_vectors: Optional[np.ndarray] = None) -> Dict[Tuple[int, int], str]:
    """Site-pair class of every site pair (i < j): the IDEAL angle between the two sites' vertices ("c90", "c180",
    ...) in the best-fit assignment of the shell to the ideal polyhedron of ``mol.coord_geom``
    (coord_geom.best_fit_assignment). ``site_vectors`` (n, 3): use these site directions instead of the current
    shell (the solver passes each candidate's ideal site vectors, so its class is the candidate's own). "na" when
    the geometry has no ideal polyhedron of this size ("other", CN >= 10)."""
    from jacobik.topology.coord_geom import best_fit_assignment, ideal_pair_angles

    sites = mol.coord_sites
    n = len(sites)
    if site_vectors is None:
        metal_pos = _vec(mol, mol.metal_idx)
        site_vectors = np.array([np.asarray(s.position, dtype=float) - metal_pos for s in sites])
    ang = None
    try:
        if n >= 2 and not np.any(np.linalg.norm(site_vectors, axis=1) < 1e-8):
            ang = ideal_pair_angles(mol.coord_geom, best_fit_assignment(site_vectors, mol.coord_geom))
    except Exception as exc:  # never let a labelling failure drop the pair
        log.debug("Q2 pair classes failed (%s): %s", getattr(mol, "csd_code", "?"), exc)
    return {(i, j): pair_class(None if ang is None else ang[i, j]) for i in range(n) for j in range(i + 1, n)}


def q2_pair_keys(mol: TMComplex, classes: Optional[Dict[Tuple[int, int], str]] = None) -> List[Tuple[int, int, tuple]]:
    """
    Q2 L0 key for every site pair (i<j, in the order the runtime enumerates):
    (geom, pair_class, dist_class, metal, sub_a, sub_b).
    The graph-distance class is the shortest ligand-graph path over ALL donor
    atoms of both sites (a haptic face counts through every one of its atoms),
    and "same ligand" means any donor atom in a common ligand. ``classes``:
    pair classes to use (default: q2_pair_classes of the current shell).
    """
    sites = mol.coord_sites
    if len(sites) < 2 or mol.coord_geom is None:
        return []
    if classes is None:
        classes = q2_pair_classes(mol)
    G = _build_ligand_graph(mol)
    geom = mol.coord_geom
    metal = mol.metal_symbol
    out = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            sa, sb = sites[i], sites[j]
            same = _same_ligand(sa, sb, mol)
            path_len = _shortest_path_between_sites(sa, sb, G)
            dist_class = _graph_dist_class(path_len, same)
            sub_a, sub_b = _site_pair_key(sa, sb, mol)
            out.append((i, j, (geom, classes.get((i, j), PAIR_CLASS_NONE), dist_class, metal, sub_a, sub_b)))
    return out


# Q3, Q4, Q6 key builders (moved to key_q3_q6.py; re-exported for callers)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Q3 / Q4 / Q6 key helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Q3 — Donor approach angles
# ---------------------------------------------------------------------------


def anionic_co_donor_bucket(mol: TMComplex, exclude_rep_idx: Optional[int] = None) -> int:
    """
    Count coordination sites in ``mol``'s shell whose subclass is an anionic
    oxygen donor (O_hydroxide, O_oxo, O_carboxylate, O_sulfonate, O_nitro,
    O_phenoxide -- anything under donor_hierarchy's O_anionic_any bucket),
    clipped to ``ANIONIC_BUCKET_MAX``.

    This is a co-donor CONTEXT feature, not a per-donor label: it does not
    change what a site's own subclass is, only what else shares its shell.
    Data motivation (data/refit/reports/q3-anionic-context.json): for
    O_water specifically, raw Q3 approach angle drops monotonically with
    this count (mean ~127 deg at 0 anionic-O co-donors down to ~104 deg at
    3+), a much cleaner and more chemically legible split than metal
    identity or coordination geometry alone, and one that is NOT explained
    by data scarcity -- there are thousands of observations at every count.
    Shared by the extractor (here) and the live IK-solve scorer, exactly like
    every other key field, so training and query-time keys cannot diverge.
    """

    if not mol.coord_sites:
        return 0
    n = 0
    for site in mol.coord_sites:
        rep = _site_representative_idx(site)
        if exclude_rep_idx is not None and rep == exclude_rep_idx:
            continue
        sub = _site_subclass(site, mol)
        if is_anionic_oxygen_donor(sub):
            n += 1
    return min(n, ANIONIC_BUCKET_MAX)


# Q4 — Donor pyramidalization (M-inclusive)
# ---------------------------------------------------------------------------


# Q6 — Steric contacts
# ---------------------------------------------------------------------------


#: Q6 polar-H keys (2026-10-01): an H bonded to N / O / F is labelled "Hp" in its contacts with acceptor atoms, so
#: N-H...O / O-H...N hydrogen bonds get their own wall instead of the C-H...O one (they sit 0.5-1 A closer; the old
#: shared wall penalised every real H-bond, e.g. PEWBOL crystal -46 nats). The lookup falls back Hp -> H.
Q6_POLAR_H_PARENTS = frozenset({"N", "O", "F"})
Q6_ACCEPTORS = frozenset({"N", "O", "F", "S", "Cl", "Br", "I"})
Q6_POLAR_H = "Hp"


def q6_polar_h(mol: TMComplex, all_nb=None) -> Set[int]:
    """Atom indices of polar H (bonded to N, O or F through a ligand bond)."""
    if all_nb is None:
        from jacobik.topology.graph import get_ligand_neighbors

        all_nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    sym = {a.idx: a.symbol for a in mol.atoms}
    return {a.idx for a in mol.atoms
            if a.symbol == "H" and any(sym.get(n) in Q6_POLAR_H_PARENTS for n in all_nb.get(a.idx, ()))}


def q6_pair_key(sym_a: str, sym_b: str, polar_a: bool = False, polar_b: bool = False) -> Tuple[str, str]:
    """Sorted Q6 key of one contact; a polar H facing an acceptor is "Hp"."""
    la = Q6_POLAR_H if (polar_a and sym_b in Q6_ACCEPTORS) else sym_a
    lb = Q6_POLAR_H if (polar_b and sym_a in Q6_ACCEPTORS) else sym_b
    return (la, lb) if la <= lb else (lb, la)
