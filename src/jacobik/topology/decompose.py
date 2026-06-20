"""
decompose.py
============
Rigid-body decomposition of tmQM ligands for inverse kinematics.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx

from jacobik.core.types import (
    Ligand,
    PivotChain,
    PivotJoint,
    RigidBody,
    RotatableBond,
    TMComplex,
)

log = logging.getLogger(__name__)

# Bonds below this quantized BO are not real ligand-internal bonds
_BO_THRESHOLD: float = 0.5

# Exactly this quantized BO is treated as a single (rotatable candidate)
_BO_SINGLE: float = 1.0
_BO_SINGLE_TOL: float = 0.05  # allow BO in [0.95, 1.05] as single


# ---------------------------------------------------------------------------
# Internal container
# ---------------------------------------------------------------------------


@dataclass
class _LigandGraph:
    """
    Structured decomposition result stored in Ligand.decomposition.
    Accessed by visualizer.py to colour rigid bodies and highlight pivots.
    """

    body_of: Dict[int, int]  # xyz_idx -> body_index
    pivot_bonds: Set[Tuple[int, int]]  # set of (min_i, max_j)
    rigid_bodies: List[Set[int]]  # body_index -> set of atom indices


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_ligand_graph(
    mol: TMComplex,
    ligand: Ligand,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], float]]:
    """
    Build the undirected ligand bond graph and a bond-order lookup.

    Returns
    -------
    G       : nx.Graph, nodes are xyz atom indices, edge attr 'order'
    bo_map  : dict (min_i, max_j) -> quantized_bo
    """
    lig_set = set(ligand.atom_indices)
    G: nx.Graph = nx.Graph()
    for idx in ligand.atom_indices:
        G.add_node(idx)

    bo_map: Dict[Tuple[int, int], float] = {}
    for b in mol.bonds:
        if b.i not in lig_set or b.j not in lig_set:
            continue
        if b.order < _BO_THRESHOLD:
            continue
        G.add_edge(b.i, b.j, order=b.order)
        bo_map[(min(b.i, b.j), max(b.i, b.j))] = b.order

    return G, bo_map


# ---------------------------------------------------------------------------
# Ring edge detection
# ---------------------------------------------------------------------------


def _find_ring_edges(G: nx.Graph) -> Set[Tuple[int, int]]:
    """
    Return the set of (min_i, max_j) pairs that are ring edges.
    Uses minimum_cycle_basis so fused systems decompose into their
    individual small rings rather than a large envelope cycle.
    """
    ring_edges: Set[Tuple[int, int]] = set()
    try:
        for cycle in nx.minimum_cycle_basis(G):
            cycle_list = list(cycle)
            n = len(cycle_list)
            for k in range(n):
                u = cycle_list[k]
                v = cycle_list[(k + 1) % n]
                ring_edges.add((min(u, v), max(u, v)))
    except Exception as exc:
        log.debug("minimum_cycle_basis failed: %s", exc)
    return ring_edges


# ---------------------------------------------------------------------------
# Rotatable bond detection
# ---------------------------------------------------------------------------


def _find_rotatable_bonds(
    G: nx.Graph,
    bo_map: Dict[Tuple[int, int], float],
    ring_edges: Set[Tuple[int, int]],
    donor_set: Set[int],
) -> List[Tuple[int, int]]:
    """
    Return a list of (i, j) rotatable bond pairs.
    """
    rotatable: List[Tuple[int, int]] = []
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        bo = bo_map.get(key, 0.0)

        # Must be a single bond
        if abs(bo - _BO_SINGLE) > _BO_SINGLE_TOL:
            continue

        # Must not be a ring edge
        if key in ring_edges:
            continue

        # Terminal check (relaxed at donor atoms)
        u_ok = G.degree(u) > 1 or u in donor_set
        v_ok = G.degree(v) > 1 or v in donor_set
        if not (u_ok and v_ok):
            continue

        rotatable.append((u, v))

    return rotatable


# ---------------------------------------------------------------------------
# Rigid body construction
# ---------------------------------------------------------------------------


def _build_rigid_bodies(
    G: nx.Graph,
    rotatable: List[Tuple[int, int]],
    bo_map: Dict[Tuple[int, int], float],
    ring_edges: Set[Tuple[int, int]],
) -> Tuple[List[RigidBody], Dict[int, int], List[Set[int]]]:
    """
    Partition the ligand into rigid bodies by removing rotatable bond edges.

    Returns
    -------
    rigid_bodies : list[RigidBody] dataclass objects (for Ligand.rigid_bodies)
    body_of      : dict xyz_idx -> body_index
    body_sets    : list[set[int]] raw atom sets per body
    """
    G_rigid = G.copy()
    for u, v in rotatable:
        G_rigid.remove_edge(u, v)

    body_sets: List[Set[int]] = [set(comp) for comp in nx.connected_components(G_rigid)]
    body_of: Dict[int, int] = {atom: bi for bi, comp in enumerate(body_sets) for atom in comp}

    rigid_bodies: List[RigidBody] = []
    for bi, comp in enumerate(body_sets):
        # Determine if this body is a ring fragment
        subgraph = G_rigid.subgraph(comp)
        is_ring = subgraph.number_of_edges() >= len(comp)  # has a cycle

        # Aromatic: all internal bonds are aromatic (BO ≈ 1.5)
        internal_bos = [bo_map.get((min(u, v), max(u, v)), 0.0) for u, v in subgraph.edges()]
        is_aromatic = bool(internal_bos) and all(abs(bo - 1.5) < 0.15 for bo in internal_bos)

        rigid_bodies.append(
            RigidBody(
                atom_indices=sorted(comp),
                is_aromatic=is_aromatic,
                is_ring=is_ring,
                label=f"body_{bi}",
            )
        )

    return rigid_bodies, body_of, body_sets


# ---------------------------------------------------------------------------
# Pivot chain construction
# ---------------------------------------------------------------------------


def _build_body_adjacency(
    rotatable: List[Tuple[int, int]],
    body_of: Dict[int, int],
) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Build adjacency list at the body level.

    Returns
    -------
    body_adj : dict body_index -> list[(atom_i, atom_j, neighbor_body_index)]
               atom_i is in body_index, atom_j is in neighbor_body_index.
    """
    body_adj: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for u, v in rotatable:
        bu = body_of[u]
        bv = body_of[v]
        body_adj[bu].append((u, v, bv))
        body_adj[bv].append((v, u, bu))
    return body_adj


def _build_pivot_chain_for_donor(
    donor_idx: int,
    body_of: Dict[int, int],
    body_adj: Dict[int, List[Tuple[int, int, int]]],
    donor_atoms: List[int],
    is_haptic: bool,
) -> PivotChain:
    """
    BFS over bodies starting from the body containing donor_idx.
    Each traversed inter-body rotatable bond becomes a PivotJoint.
    """
    start_body = body_of.get(donor_idx)
    if start_body is None:
        return PivotChain(joints=[], donor_atoms=donor_atoms, is_haptic=is_haptic)

    joints: List[PivotJoint] = []
    visited: Set[int] = {start_body}
    queue: deque = deque([start_body])

    while queue:
        curr = queue.popleft()
        # Sort neighbours for deterministic output
        for atom_i, atom_j, next_body in sorted(body_adj[curr]):
            if next_body not in visited:
                visited.add(next_body)
                joints.append(PivotJoint(atom_i=atom_i, atom_j=atom_j))
                queue.append(next_body)

    return PivotChain(
        joints=joints,
        donor_atoms=donor_atoms,
        is_haptic=is_haptic,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decompose_ligand(mol: TMComplex, ligand: Ligand) -> None:
    """
    Perform rigid-body decomposition for one ligand and populate it in-place.


    Parameters
    ----------
    mol    : fully assembled TMComplex (quantized bonds, donor_atoms set)
    ligand : a Ligand from mol.ligands (modified in-place)

    """
    if not ligand.atom_indices:
        return

    # --- Build ligand graph ---
    G, bo_map = _build_ligand_graph(mol, ligand)

    # --- Ring edges ---
    ring_edges = _find_ring_edges(G)

    # --- Donor set (for relaxed terminal rule) ---
    donor_set: Set[int] = set(ligand.donor_atoms)
    for h in ligand.haptic_groups:
        donor_set.update(h.donor_indices)

    # --- Rotatable bonds ---
    rotatable_pairs = _find_rotatable_bonds(G, bo_map, ring_edges, donor_set)

    ligand.rotatable_bonds = [
        RotatableBond(i=u, j=v, order=bo_map.get((min(u, v), max(u, v)), 1.0))
        for u, v in rotatable_pairs
    ]

    # --- Rigid bodies ---
    rigid_bodies, body_of, body_sets = _build_rigid_bodies(G, rotatable_pairs, bo_map, ring_edges)
    ligand.rigid_bodies = rigid_bodies

    # --- Body adjacency for pivot chains ---
    body_adj = _build_body_adjacency(rotatable_pairs, body_of)

    # --- Pivot chains ---
    pivot_chains: List[PivotChain] = []

    # Collect all haptic donor indices across all groups on this ligand
    all_haptic_donors: Set[int] = set()
    for h in ligand.haptic_groups:
        all_haptic_donors.update(h.donor_indices)

    # One PivotChain per haptic group, rooted at the first donor in the group.
    # All donors in a haptic group belong to the same ring body, so any one
    # works as the BFS root.
    for h in ligand.haptic_groups:
        root = h.donor_indices[0]
        chain = _build_pivot_chain_for_donor(
            donor_idx=root,
            body_of=body_of,
            body_adj=body_adj,
            donor_atoms=h.donor_indices,
            is_haptic=True,
        )
        pivot_chains.append(chain)

    # Kappa donors: any donor_atom not claimed by any haptic group
    kappa_donors = [d for d in ligand.donor_atoms if d not in all_haptic_donors]

    for donor in kappa_donors:
        chain = _build_pivot_chain_for_donor(
            donor_idx=donor,
            body_of=body_of,
            body_adj=body_adj,
            donor_atoms=[donor],
            is_haptic=False,
        )
        pivot_chains.append(chain)

    ligand.pivot_chains = pivot_chains

    # --- Decomposition container ---
    pivot_bond_set: Set[Tuple[int, int]] = {(min(u, v), max(u, v)) for u, v in rotatable_pairs}
    ligand.decomposition = _LigandGraph(
        body_of=body_of,
        pivot_bonds=pivot_bond_set,
        rigid_bodies=body_sets,
    )

    log.debug(
        "%s ligand %d: %d atoms, %d rotatable, %d bodies, %d chains.",
        mol.csd_code,
        ligand.ligand_id,
        len(ligand.atom_indices),
        len(rotatable_pairs),
        len(rigid_bodies),
        len(pivot_chains),
    )
