"""
decompose.py
============
Rigid-body decomposition of tmQM ligands for inverse kinematics.

Version history
----------------
1.0.0 : initial release
1.1.0 : Fixed _find_rotatable_bonds's terminal-check relaxation
        (`or u in donor_set`), which let a bond count as rotatable purely
        because one endpoint is a donor atom, even when that donor is a
        true degree-1 leaf with nothing beyond it. Rotating such a bond is
        a mathematical no-op (a lone atom has no orientation to sweep) and
        additionally carves the donor into its own disconnected singleton
        rigid body via _build_rigid_bodies -- severing it from the one
        joint (its neighbour's other bond) that would have given it real
        torsional control. Confirmed on EDTA: one of four chemically
        equivalent carboxylate donors became permanently frozen (measured
        0.0000 A movement under any combination of the ligand's other
        torsions) while the other three, whose own O-C bond happened to be
        a formal double bond and were therefore already excluded even
        under the old code, moved 3.5-7.3 A under the same test. Fix:
        never relax the terminal check for a true degree-1 atom, donor or
        not -- it only ever costs a real joint and never buys a working
        one. Donors with degree > 1 (mid-chain, ring/haptic) are
        unaffected.
1.2.0 : Added _select_root_body() and _LigandGraph.root_body. 1.1.0 fixed
        the DOF asymmetry between chemically-equivalent donors but did not
        eliminate the underlying issue: exactly one rigid body in the tree
        is architecturally the "root" that only ever receives the shared
        base 6-DOF transform (fk.py) and is never swept by any joint --
        confirmed still true after 1.1.0 (the merged carboxylate group
        still measured 0.0000 A movement, just as a 3-atom unit instead of
        a lone atom). The root was previously an emergent side effect of
        rigid-body enumeration order, with no guarantee it would avoid a
        donor. _select_root_body explicitly picks a donor-free body when
        one exists (falling back to body 0 only for the degenerate case of
        zero donor-free atoms anywhere -- a single-atom donor or lone
        haptic ring, where n_joints == 0 and root choice is moot). Initial
        version selected the donor-free candidate minimising max distance
        to any donor body; found (empirically, on EDTA) that this can pick
        a candidate exactly 1 hop from a donor, which is just as broken as
        being the donor's own body -- see 1.2.1.
1.2.1 : Fixed the selection criterion: a torsion around bond (A, B) never
        moves A or B themselves, only atoms strictly beyond one side. A
        donor body exactly 1 hop from root IS an axis-endpoint of the
        joint connecting it, so 1.2.0's "minimise max distance" could (and
        on EDTA did) select the single most-central body while ignoring
        that it landed exactly 1 hop from N4, leaving N4 frozen exactly as
        before the fix -- confirmed via the same direct movement test.
        Root selection now prefers candidates >= 2 hops from EVERY donor
        body (only among those is eccentricity minimised); falls back to
        minimising the count of 1-hop donors, then eccentricity, if no
        candidate clears every donor by 2+ hops. See the KNOWN LIMIT note
        on _select_root_body: if two branch-point donors (single-atom body,
        degree >= 2) are close enough that no body clears both simultaneously
        (confirmed on EDTA: N4/N7, 2 hops apart via a 2-carbon bridge --
        every non-donor body is a direct neighbour of one or the other),
        at most one can ever be made correctable by root choice alone --
        this is a hard limit of rigid-body/torsion-only FK, not a bug.
        Net effect on EDTA: 5/6 donors torsion-correctable (up from
        1/6 before any of today's fixes), the 6th (whichever
        branch-point donor root ends up 1-hop from) bounded by an
        intrinsic topological constraint, not code.
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
    root_body: int = 0  # body_index of the FK tree root (see _select_root_body)


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
    donor_set: Set[int],  # noqa: ARG001 -- kept for call-site/API stability;
    # see BUGFIX note below for why it's no longer used in the body.
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

        # Terminal check.
        #
        # BUGFIX (topology 1.1.0): the old relaxation `or u in donor_set`
        # let a bond count as rotatable purely because ONE endpoint is a
        # donor atom, even when that donor has degree 1 (nothing beyond it
        # -- a true terminal leaf). Rotating a bond whose far side is a
        # single point with nothing attached is a mathematical no-op: a
        # lone atom has no orientation to sweep, so the "torsion" changes
        # nothing about its position. Worse, keeping the bond "rotatable"
        # in that case makes _build_rigid_bodies() carve the donor off into
        # its own disconnected singleton rigid body -- severing it from the
        # one joint (its neighbour's OTHER bond into the rest of the
        # ligand) that would have given it genuine torsional control.
        #
        # Confirmed on EDTA: with the relaxation, one of four chemically
        # equivalent carboxylate donors (whichever one's own O-C bond
        # happened to be a formal single bond) became a lone singleton body
        # and measurably moved 0.0000 A under ANY combination of the other
        # rotatable torsions -- frozen at whatever the one-shot rigid base
        # placement left it at, with zero recourse for the rest of the
        # solve. The other three donors (O-C formally double-bonded,
        # correctly excluded even under the old code) stayed merged with
        # their carbonyl carbon and moved 3.5-7.3 A under the same test,
        # because that merger gives them a real, position-changing joint.
        #
        # Fix: only relax the terminal check for a donor when it actually
        # has somewhere to go -- i.e. never for a true degree-1 leaf, since
        # in that case relaxing is pure downside (an inert DOF that costs a
        # real one). This makes ALL degree-1 single-atom donors behave like
        # the already-correct double-bond case: they stay part of their
        # neighbour's rigid body and get moved by that neighbour's real
        # joints, instead of being carved off into a dead end.
        # Fix: only relax the terminal check for a donor when it actually
        # has somewhere to go -- i.e. never for a true degree-1 leaf, since
        # in that case relaxing is pure downside (an inert DOF that costs a
        # real one). This makes ALL degree-1 single-atom donors behave like
        # the already-correct double-bond case: they stay part of their
        # neighbour's rigid body and get moved by that neighbour's real
        # joints, instead of being carved off into a dead end.
        #
        # Net effect: the `or ... in donor_set` relaxation is removed
        # entirely. It never helps (a degree-1 donor gains nothing from
        # being "rotatable" since there's nothing beyond it to move) and
        # is actively harmful in exactly the case that matters (severing a
        # donor from its one real, position-changing joint). Donor atoms
        # with degree > 1 (e.g. mid-chain or ring donors) are unaffected --
        # they already passed the plain degree check on their own merits.
        u_ok = G.degree(u) > 1
        v_ok = G.degree(v) > 1
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


def _select_root_body(
    body_of: Dict[int, int],
    body_sets: List[Set[int]],
    body_adj: Dict[int, List[Tuple[int, int, int]]],
    donor_set: Set[int],
) -> int:
    """
    Choose which rigid body anchors the FK tree's shared base transform.

    Every OTHER body reaches the root via a chain of real, position-changing
    joints and can be corrected by torsion optimisation. The root body
    itself cannot -- it only ever receives the one-shot base transform, with
    no joint to locally fix it afterwards.

    A body containing a donor atom must therefore never be chosen as root
    when any donor-free body exists (see engine.py 2.3.0/2.4.0 for the EDTA
    case that surfaced this).

    BUGFIX (1.3.0): donor-free alone is not sufficient. A torsion around
    bond (A, B) never moves A or B themselves -- only atoms strictly beyond
    B (or A). A donor body that is exactly 1 hop from root IS one of the two
    atoms defining that joint's axis, so it can never be moved by it, and
    since root has no further "parent" joint, there is no other joint that
    could move it either. At 2+ hops, some joint further out (root's side)
    sweeps the donor's body as part of a larger downstream group WITHOUT
    the donor being that joint's own axis atom -- so it IS correctable.
    The original "minimise max distance to any donor" criterion picked the
    single most-central body without checking this, and on EDTA it landed
    exactly 1 hop from a donor (N4, a genuine 3-way branch point -- all 3
    of its bonds are real, correctly-rotatable amine C-N single bonds, so
    unlike a merged terminal donor it can't be worked around by rigid-body
    grouping): confirmed frozen at 0.0000 A movement under the old
    criterion. Now prefers candidates with distance >= 2 from EVERY donor
    body; only among those does it minimise eccentricity (max distance) for
    gradient conditioning. Falls back to minimising the COUNT of donors at
    distance exactly 1 if no 2+-hop-clear candidate exists (only possible
    for a very small/tightly-clustered ligand).

    Falls back to body 0 if every body contains a donor (only possible for
    a ligand with no donor-free atom anywhere -- in practice this means a
    single-atom donor or a lone haptic ring with zero rotatable bonds, where
    there is exactly one body total and root choice doesn't matter, since
    n_joints == 0 either way).

    KNOWN LIMIT, not fixable by root choice alone: a single-atom donor body
    that is itself a real branch point (degree >= 2 in the body graph, e.g.
    a tertiary amine N with 3 genuinely-rotatable C-N bonds) has NO internal
    DOF of its own -- a point has no orientation to parameterise -- so it is
    always the axis-endpoint of whichever joint connects it toward root,
    and is only correctable if root is 2+ hops away (some joint further out
    sweeps it without being its own axis atom). If two such branch-point
    donors are close enough together that every donor-free body is 1 hop
    from at least one of them (confirmed on EDTA: N4 and N7, 2 hops apart
    via a 2-carbon bridge -- every non-donor body is a direct neighbour of
    one or the other), AT MOST ONE of the two can ever be made correctable,
    regardless of which candidate is chosen. This function picks which one
    (via the (n_one_hop, max_dist) fallback ordering) but cannot fix both;
    doing so would need bond-angle flexibility at the branch point itself,
    which this rigid-body/torsion-only FK parameterisation doesn't model.
    """
    n_bodies = len(body_sets)
    donor_bodies = {body_of[d] for d in donor_set if d in body_of}
    candidates = [b for b in range(n_bodies) if b not in donor_bodies]

    if not candidates:
        return 0

    if len(donor_bodies) == 0:
        # No donors at all (shouldn't normally happen) -- any body is fine.
        return candidates[0]

    # One BFS per donor body gives distances to every other body in a
    # single pass (cheaper than re-running BFS per candidate: bodies is
    # small, but there's no reason to redo the same traversal repeatedly).
    dist_to_donor: List[List[int]] = []  # per donor body: list of distances indexed by body
    for src in donor_bodies:
        dist: Dict[int, int] = {src: 0}
        queue: deque = deque([src])
        while queue:
            curr = queue.popleft()
            for _, _, nb in body_adj.get(curr, []):
                if nb not in dist:
                    dist[nb] = dist[curr] + 1
                    queue.append(nb)
        dist_to_donor.append([dist.get(b, n_bodies) for b in range(n_bodies)])

    def n_one_hop(b: int) -> int:
        return sum(1 for dd in dist_to_donor if dd[b] == 1)

    def max_dist(b: int) -> int:
        return max(dd[b] for dd in dist_to_donor)

    clear_candidates = [b for b in candidates if n_one_hop(b) == 0]
    if clear_candidates:
        return min(clear_candidates, key=max_dist)

    # No candidate is >= 2 hops from every donor -- minimise how many are
    # stuck at the bad 1-hop distance first, then fall back to eccentricity.
    return min(candidates, key=lambda b: (n_one_hop(b), max_dist(b)))


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

    # --- Root body: never a donor-containing body when a donor-free one
    # exists (see _select_root_body docstring) ---
    root_body = _select_root_body(body_of, body_sets, body_adj, donor_set)

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
        root_body=root_body,
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