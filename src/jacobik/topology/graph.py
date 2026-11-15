"""
graph.py
========
Shared graph building and topological queries for ligand networks.
"""

from typing import Dict, List, Set, Tuple

import networkx as nx

from jacobik.core.types import Atom, Bond

# Atomic numbers for the heaviest_heavy_neighbor convention
_ATOMIC_NUMBER: Dict[str, int] = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Te": 52,
    "I": 53,
}


def heaviest_heavy_neighbor(neighbor_idxs: List[int], atom_map: Dict[int, Atom]) -> int:
    """
    Return the 'heaviest' heavy-atom neighbor index.
    Sorts by atomic number descending; lower atom index as tie-break.
    """

    def sort_key(idx: int) -> Tuple[int, int]:
        sym = atom_map[idx].symbol if idx in atom_map else ""
        return (_ATOMIC_NUMBER.get(sym, 0), -idx)

    return max(neighbor_idxs, key=sort_key)


def build_heavy_ligand_graph(
    atoms: List[Atom], bonds: List[Bond], metal_idx: int, bo_threshold: float = 0.5
) -> nx.Graph:
    """
    Build the heavy-atom ligand bond graph: all non-metal, non-H atoms;
    edges for bonds with quantized BO >= bo_threshold, metal bonds excluded.
    Node attribute: 'symbol'.
    """
    G = nx.Graph()
    atom_map = {a.idx: a for a in atoms}
    for a in atoms:
        if a.idx == metal_idx or a.symbol == "H":
            continue
        G.add_node(a.idx, symbol=a.symbol)

    for b in bonds:
        if b.i == metal_idx or b.j == metal_idx:
            continue
        if b.order < bo_threshold:
            continue
        ai = atom_map.get(b.i)
        aj = atom_map.get(b.j)
        if ai is None or aj is None or ai.symbol == "H" or aj.symbol == "H":
            continue
        G.add_edge(b.i, b.j)

    return G


def get_heavy_ligand_neighbors(
    atoms: List[Atom], bonds: List[Bond], metal_idx: int, bo_threshold: float = 0.5
) -> Dict[int, List[int]]:
    """
    Build atom_idx -> [neighbor_atom_idx, ...] for heavy-atom ligand bonds only.
    Excludes: metal bonds, H atoms, bonds with order < bo_threshold.
    """
    atom_map = {a.idx: a for a in atoms}
    neighbors: Dict[int, List[int]] = {a.idx: [] for a in atoms}
    for b in bonds:
        if b.i == metal_idx or b.j == metal_idx:
            continue
        if b.order < bo_threshold:
            continue
        ai = atom_map.get(b.i)
        aj = atom_map.get(b.j)
        if ai is None or aj is None or ai.symbol == "H" or aj.symbol == "H":
            continue
        neighbors[b.i].append(b.j)
        neighbors[b.j].append(b.i)
    return neighbors


def get_ligand_neighbors(
    atoms: List[Atom], bonds: List[Bond], metal_idx: int, bo_threshold: float = 0.5
) -> Dict[int, List[int]]:
    """
    Build atom_idx -> [neighbor_atom_idx, ...] for all ligand bonds.
    Excludes: metal bonds, bonds with order < bo_threshold.
    """
    neighbors: Dict[int, List[int]] = {a.idx: [] for a in atoms}
    for b in bonds:
        if b.i == metal_idx or b.j == metal_idx:
            continue
        if b.order < bo_threshold:
            continue
        neighbors[b.i].append(b.j)
        neighbors[b.j].append(b.i)
    return neighbors


# ---------------------------------------------------------------------------
# Q6 topological exclusion (shared by extraction and the runtime scorer)
# ---------------------------------------------------------------------------

#: Atom pairs separated by at most this many bonds in the FULL graph
#: (all atoms including H, ligand bonds plus metal-donor edges) are treated as
#: topologically constrained and are not part of the non-bonded steric (Q6)
#: distribution.  Extraction and runtime MUST use this same value.
Q6_EXCLUDE_MAX_BONDS = 3
#: Pairs involving the metal are excluded only up to this many bonds (B2.4): M...Y three bonds away (M-D-X-Y) depends
#: on the M-D-X-Y torsion like any 1-4 contact, so it is scored. The old rule (3) was retired 2026-10-16.
Q6_METAL_EXCLUDE_MAX_BONDS = 2
#: Ligand-ligand pairs whose only short path runs THROUGH the metal (X-D-M-D', 3 bonds: a donor's substituent and
#: another donor; a Cp H and a cis donor) are not held by any bond or angle: their distance depends on the D-M-D'
#: angle and on how the donors are oriented, like any contact. They are scored (2026-10-13; the old rule, retired
#: 2026-10-16, counted M-D edges as covalent and so never scored them). tmQM X1 (8000 complexes):
#: their lower tail matches ordinary contacts of the same element pair (0.1 % quantile C-H 2.53 vs 2.43 A, C-C 2.92 vs
#: 2.97 A), so they share the element-pair walls. Donor-donor pairs (D-M-D', Q2's quantity) stay excluded.
Q6_THROUGH_METAL: bool = True  # frozen 2026-10-16
from jacobik.core import profile as _profile
Q6_PHYSICAL_ALL: bool = bool(_profile.get("q6_physical_all"))


def donor_indices_of(mol) -> List[int]:
    """All atoms bonded to the metal (kappa donors and every haptic face atom)."""
    out: List[int] = []
    for site in getattr(mol, "coord_sites", None) or []:
        out.extend(int(d) for d in site.donor_indices)
    return sorted(set(out))


def q6_topological_exclusions(
    atoms: List[Atom],
    bonds: List[Bond],
    metal_idx: int,
    donor_idxs: List[int],
    max_bonds: int = Q6_EXCLUDE_MAX_BONDS,
    bo_threshold: float = 0.5,
) -> Set[Tuple[int, int]]:
    """
    (min_idx, max_idx) pairs within ``max_bonds`` bonds on the full molecular
    graph: every atom (H and the metal included), every ligand bond with
    order >= bo_threshold, and one edge per metal-donor bond.
    With Q6_THROUGH_METAL (default since 2026-10-13) ligand-ligand pairs count
    only paths along ligand bonds; the metal's pairs (within
    Q6_METAL_EXCLUDE_MAX_BONDS) and donor-donor pairs stay excluded, so a pair
    joined only through the metal (X-D-M-D') is scored.

    This also removes metal-X and H-X contacts whose distance is fixed by bonds and angles, so they do not bias the
    Q6 mean of their element pair.
    """
    adj: Dict[int, List[int]] = {a.idx: [] for a in atoms}
    for b in bonds:
        if b.order < bo_threshold or b.i not in adj or b.j not in adj:
            continue
        if b.i == metal_idx or b.j == metal_idx:
            continue  # metal edges come from the verified donor list only
        adj[b.i].append(b.j)
        adj[b.j].append(b.i)
    if metal_idx in adj:
        for d in donor_idxs:
            if d in adj and d != metal_idx:
                if d not in adj[metal_idx]:
                    adj[metal_idx].append(d)
                    adj[d].append(metal_idx)
    close: Set[Tuple[int, int]] = set()
    # through_metal: walk ligand bonds only for the ligand-ligand pairs (a path through the metal does not hold a pair);
    # the metal's own pairs and the donor-donor pairs are added below
    walk = adj
    if Q6_THROUGH_METAL and metal_idx in adj:
        walk = {u: [v for v in vs if v != metal_idx] for u, vs in adj.items() if u != metal_idx}
    for src in walk:
        frontier = [src]
        seen = {src}
        # The D-B physical-contact experiment includes 1-4 pairs; all closer
        # pairs remain topologically excluded. Keep production's depth unchanged.
        ligand_max_bonds = 2 if Q6_PHYSICAL_ALL else max_bonds
        for _ in range(ligand_max_bonds):
            nxt = []
            for u in frontier:
                for v in walk[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
                        close.add((src, v) if src < v else (v, src))
            frontier = nxt
            if not frontier:
                break
    if walk is not adj:
        dons = sorted(adj[metal_idx])
        for ii, a in enumerate(dons):
            close.add((a, metal_idx) if a < metal_idx else (metal_idx, a))
            for b in dons[ii + 1:]:
                close.add((a, b) if a < b else (b, a))
        # metal pairs up to max_bonds; the rule below re-opens those beyond Q6_METAL_EXCLUDE_MAX_BONDS
        dist = {metal_idx: 0}
        frontier = [metal_idx]
        for dcur in range(1, max_bonds + 1):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dcur
                        nxt.append(v)
            frontier = nxt
        for v, dv in dist.items():
            if v != metal_idx:
                close.add((metal_idx, v) if metal_idx < v else (v, metal_idx))
    if metal_idx in adj and Q6_METAL_EXCLUDE_MAX_BONDS < max_bonds:
        # metal pairs farther than Q6_METAL_EXCLUDE_MAX_BONDS bonds are scored again
        dist = {metal_idx: 0}
        frontier = [metal_idx]
        for dcur in range(1, max_bonds + 1):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dcur
                        nxt.append(v)
            frontier = nxt
        for v, dv in dist.items():
            if dv > Q6_METAL_EXCLUDE_MAX_BONDS:
                close.discard((metal_idx, v) if metal_idx < v else (v, metal_idx))
    return close


def canonical_flank(
    candidates: List[int],
    atom_map: Dict[int, Atom],
    type_map: Dict[int, str],
    heavy_deg: Dict[int, int],
) -> int:
    """
    Deterministic flanking-atom choice for a torsion: heaviest element, then
    local bond type string, then heavy-degree, and only then the atom index.
    The old rule ended in "lowest index", which depends on atom ordering and
    therefore differs between tmQM files and generated complexes.  Ties that
    remain are chemically equivalent atoms (e.g. the two ortho C of a phenyl).
    """

    def key(i: int):
        sym = atom_map[i].symbol if i in atom_map else ""
        return (_ATOMIC_NUMBER.get(sym, 0), type_map.get(i, sym), heavy_deg.get(i, 0), -i)

    return max(candidates, key=key)


# ---------------------------------------------------------------------------
# Rotor symmetry (Q5 folding, 2026-09-28 workspace)
# ---------------------------------------------------------------------------


def wl_classes(nodes, nb, label, iters: int = 6) -> Dict[int, int]:
    """Topological equivalence classes (Weisfeiler-Lehman refinement of element labels over the bond graph).
    Two atoms share a class when no bounded-depth walk distinguishes them (e.g. the three H of a methyl, the two
    ortho C of a phenyl). ``nb``: node -> iterable of bonded nodes (include metal-donor edges)."""
    c = {i: label[i] for i in nodes}
    for _ in range(iters):
        c = {i: hash((c[i], tuple(sorted(c[j] for j in nb.get(i, ()))))) for i in nodes}
    return c


def rotor_symmetry(a: int, other: int, nb, cls, planar: bool) -> int:
    """Symmetry number of the rotor top on atom ``a`` about the bond a-other: n when a's n >= 2 other neighbours
    are topologically equivalent AND their arrangement is n-fold symmetric (sp3 with 3 such neighbours: n = 3;
    PLANAR trigonal (sp2/aromatic, ``planar``) with 2: n = 2). A lone pair (pyramidal NH2, sulfonium) or any
    distinct neighbour breaks the symmetry (n = 1)."""
    N = [x for x in nb.get(a, ()) if x != other]
    if len(N) < 2 or len({cls[x] for x in N}) > 1:
        return 1
    deg = len(nb.get(a, ()))
    return len(N) if (deg == 4 and len(N) == 3) or (deg == 3 and len(N) == 2 and planar) else 1
