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


def get_steric_exclusions(
    atoms: List[Atom],
    bonds: List[Bond],
    metal_idx: int,
    heavy_graph: nx.Graph,
    bo_threshold: float = 0.5,
) -> Set[Tuple[int, int]]:
    """
    Returns the set of (min_idx, max_idx) atom pairs to exclude from steric contact lists:
    1. 1,2 and 1,3 heavy-heavy pairs via BFS cutoff=2 on the ligand graph.
    2. Direct bonds involving H (1,2 H-X pairs).
    """
    atom_map = {a.idx: a for a in atoms}
    close: Set[Tuple[int, int]] = set()

    for a in atoms:
        if a.symbol == "H" or a.idx == metal_idx:
            continue
        if not heavy_graph.has_node(a.idx):
            continue
        try:
            lengths = nx.single_source_shortest_path_length(heavy_graph, a.idx, cutoff=2)
        except Exception:
            continue
        for nb_idx, dist in lengths.items():
            if dist <= 2:
                lo, hi = (a.idx, nb_idx) if a.idx < nb_idx else (nb_idx, a.idx)
                close.add((lo, hi))

    for b in bonds:
        if b.order < bo_threshold:
            continue
        ai = atom_map.get(b.i)
        aj = atom_map.get(b.j)
        if ai is None or aj is None:
            continue
        if ai.symbol == "H" or aj.symbol == "H":
            lo, hi = (b.i, b.j) if b.i < b.j else (b.j, b.i)
            close.add((lo, hi))

    return close
