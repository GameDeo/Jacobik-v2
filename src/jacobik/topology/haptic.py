"""Haptic face detection and geometry-gate helpers extracted from donor_classifier."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import networkx as nx
import numpy as np

from jacobik.core.types import Atom, Bond, HapticInteraction, Ligand, TMComplex
from jacobik.core.geometry import dihedral_deg

# Max angle between the face-centroid->metal vector and the face normal.
_FACE_MAX_OFFNORMAL_DEG: float = 50.0

# Pi-delocalization gate: minimum mean ring bond order / bond order counted as pi.
_FACE_MIN_MEAN_BO: float = 1.10
_FACE_PI_BO: float = 1.40

#: Bound-atom gap: an atom joins a face when its metal distance is <= this x the nearest face atom's distance.
#: KDE valley of d/dmin over 106,257 atoms of carbon clusters around metal-bound seeds (17,028 complexes, tmQM
#: X1-X3 s000-s002): bound mode 1.00-1.10, non-bonded beta-atom mode ~1.50, valley 1.244.
HAPTIC_BOUND_RATIO: float = 1.24
#: Fold rule: an open arc is split at a formally single bond between two pi bonds when the torsion across it exceeds
#: this. KDE valley of |torsion| at such junctions in the same data (261 junctions, 237 complexes): 36-37 deg for
#: bandwidths 0.08-0.12; two-component GMM beats one by 369 BIC (means 4 and 74 deg).
HAPTIC_FOLD_DEG: float = 36.0


def _recompute_haptic_interaction_geometry(
    mol: TMComplex,
    donor_indices: List[int],
    metal_bonds: List[Bond],
) -> Optional[HapticInteraction]:
    """
    Recompute centroid / avg_bond_order / donor_symbols for an ALREADY-KNOWN
    haptic face (donor_indices is authoritative, not discovered here).
    """
    atom_map = {a.idx: a for a in mol.atoms}
    metal_idx = mol.metal_idx
    heavy = [x for x in donor_indices if atom_map[x].symbol != "H"]
    if not heavy:
        return None
    positions = np.stack([atom_map[x].position for x in heavy])
    centroid = positions.mean(axis=0)
    metal_bo = {
        (min(b.i, b.j), max(b.i, b.j)): b.order for b in metal_bonds
    }
    raw_bos = [
        metal_bo.get((min(metal_idx, x), max(metal_idx, x)), 0.0)
        for x in heavy
    ]
    nz = [b for b in raw_bos if b > 0.0]
    avg_bo = float(np.mean(nz)) if nz else 0.0
    return HapticInteraction(
        metal_idx=metal_idx,
        donor_indices=sorted(heavy),
        haptic_order=len(heavy),
        centroid=centroid,
        donor_symbols=[atom_map[x].symbol for x in sorted(heavy)],
        avg_bond_order=avg_bo,
    )


def _detect_haptic_groups_v2(
    mol: TMComplex,
    ligand: Ligand,
    metal_bonds: List[Bond],
    smiles_path: bool = False,
) -> List[HapticInteraction]:
    """
    Haptic faces from the crystal geometry, graph first (v2).

    1. Bound atoms: from each seed (declared donor or raw metal contact), grow through bonded heavy atoms whose
       metal distance is <= HAPTIC_BOUND_RATIO x the cluster's nearest distance.
    2. A complete ring of bound atoms that passes the pi and off-normal gates is one closed face (Cp, arene,
       heteroarene).
    3. Every other contiguous piece of bound, unsaturated (<= 3 non-metal neighbours) atoms: two bonded atoms = one eta2 site (alkene, alkyne, C=O, O-O ...;
       a bonded kappa pair is an eta2 site by definition); three or more carbons along a chain = an open face, split at
       each formally single bond between two pi bonds whose torsion exceeds HAPTIC_FOLD_DEG (a folded, non-conjugated
       junction: DUFLAR = eta2 + eta4). Pieces of one atom stay kappa donors.
    """
    from jacobik.topology import donor_classifier as _dc

    lig_set = set(ligand.atom_indices)
    atom_map = {a.idx: a for a in mol.atoms}
    metal_idx = mol.metal_idx
    metal_pos = np.asarray(atom_map[metal_idx].position, float)
    pos = lambda i: np.asarray(atom_map[i].position, float)  # noqa: E731
    dist = lambda i: float(np.linalg.norm(pos(i) - metal_pos))  # noqa: E731

    seeds: Set[int] = set(ligand.donor_atoms)
    if not smiles_path:
        for b in metal_bonds:
            partner = b.j if b.i == metal_idx else b.i
            if partner in lig_set:
                seeds.add(partner)
    G = _dc._ligand_heavy_graph(mol, lig_set)
    seeds = {x for x in seeds if x in G}
    if not seeds:
        return []
    # Carbons as close to the metal as the ligand's nearest seed count as seeds too: the second alkene of a COD, or
    # the far end of an arc, need not be a declared donor or a raw metal contact.
    d0 = min(dist(x) for x in seeds)
    seeds |= {x for x in G if atom_map[x].symbol == "C" and dist(x) <= HAPTIC_BOUND_RATIO * d0}
    lig_bo = {frozenset((b.i, b.j)): b.order for b in mol.bonds if metal_idx not in (b.i, b.j)}
    n_nbr: Dict[int, int] = {}
    for b in mol.bonds:
        if metal_idx in (b.i, b.j) or b.order < 0.5:
            continue
        n_nbr[b.i] = n_nbr.get(b.i, 0) + 1
        n_nbr[b.j] = n_nbr.get(b.j, 0) + 1
    bo = lambda u, v: lig_bo.get(frozenset((u, v)), 1.0)  # noqa: E731

    try:
        rings = [list(c) for c in nx.minimum_cycle_basis(G) if len(c) >= 3]
    except Exception:
        rings = []
    ring_c = {x for r in rings for x in r}  # ring members (carbocycles, heteroarenes, inorganic P3/P5/B rings)

    # 1. bound clusters, started only from pi-capable atoms (C, or a ring member: pyrrolyl N, boratabenzene B,
    #    cyclo-P3): a kappa O/N/P/S/halide donor must not set the distance scale of a nearby pi face. Other
    #    heteroatoms can still join a cluster (the P of an eta2 P#C, the O of an eta2 acyl).
    clusters: List[Set[int]] = []
    seen: Set[int] = set()
    far_end: Set[int] = set()
    for s in sorted((x for x in seeds if atom_map[x].symbol == "C" or x in ring_c), key=dist):
        # nearest first; a seed next to a nearer cluster that did not reach it is the far end of that pi system
        if s in seen:
            continue
        if any(nb in seen for nb in G.neighbors(s)):
            seen.add(s)  # far end: also blocks clusters started further out along the same chain
            far_end.add(s)
            continue
        # grow, then re-anchor on the cluster's nearest atom until stable (order independent)
        core, cl = {s}, set()
        while True:
            dmin = min(dist(x) for x in core)
            cl, frontier = set(core), list(core)
            while frontier:
                cur = frontier.pop()
                for nb in G.neighbors(cur):
                    if nb not in cl and nb not in seen and dist(nb) <= HAPTIC_BOUND_RATIO * dmin:
                        cl.add(nb)
                        frontier.append(nb)
            near = min((x for x in cl if atom_map[x].symbol == "C" or x in ring_c), key=dist)  # pi atoms set the scale
            if dist(near) >= dmin - 1e-9:
                break
            core = {near}
        seen |= cl
        clusters.append(cl)
        far_end |= {nb for x in cl for nb in G.neighbors(x) if nb not in cl}

    def ring_ok(r: List[int]) -> bool:
        pts = np.stack([pos(x) for x in r])
        cen = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - cen)
        v = metal_pos - cen
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(vt[2] @ v)) / (np.linalg.norm(v) + 1e-12)))))
        bos = [bo(u, w) for i, u in enumerate(r) for w in r[i + 1:] if G.has_edge(u, w)]
        pi = bool(bos) and (np.mean(bos) >= _FACE_MIN_MEAN_BO or np.mean([x >= _FACE_PI_BO for x in bos]) >= 0.5)
        return ang <= _FACE_MAX_OFFNORMAL_DEG and pi

    faces: List[List[int]] = []
    for cl in clusters:
        used: Set[int] = set()
        # 2. closed rings, largest first
        for r in sorted((r for r in rings if set(r) <= cl), key=len, reverse=True):
            if set(r) & used or not ring_ok(r):
                continue
            faces.append(sorted(r))
            used |= set(r)
        # 3. open pieces, from unsaturated atoms only: an atom with four or more non-metal neighbours (H counted) has no
        #    pi electrons to share (e.g. the methyl of an inner C-methyl porphyrin carbon, SAYZAV); it stays kappa.
        #    A heteroatom outside any carbon ring joins a piece only through a pi bond (the P of P#C, the O of an eta2
        #    acyl); singly bonded ones (a B-O-M bridge, an alkoxide next to a face) stay kappa.
        def _pi_capable(x: int) -> bool:
            if atom_map[x].symbol == "C" or x in ring_c:
                return True
            return any(bo(x, y) >= _FACE_PI_BO for y in G.neighbors(x) if y in cl)

        rest = G.subgraph({x for x in cl - used if n_nbr.get(x, 0) <= 3 and _pi_capable(x)})
        for comp in nx.connected_components(rest):
            comp = set(comp)
            if len(comp) == 2:
                u, w = sorted(comp)
                # C-C pairs need no bond-order evidence (a coordinated alkene's Wiberg order can drop to ~1); a pair
                # with a heteroatom must show pi character (P#C, C=O, C=N) or lie in a carbon ring (B-C of a
                # boratabenzene), else its atoms stay kappa (B-O, O-O).
                if (atom_map[u].symbol == "C" and atom_map[w].symbol == "C") or bo(u, w) >= _FACE_PI_BO \
                        or any({u, w} <= set(r) for r in rings if set(r) <= ring_c):
                    faces.append([u, w])
                continue
            if len(comp) < 2:
                continue
            H = rest.subgraph(comp)
            carbons = all(atom_map[x].symbol == "C" for x in comp)
            ends = [x for x in comp if H.degree(x) == 1]
            if not carbons:
                # heteroatom-containing piece: one face when it lies within one ring (e.g. eta3 of a heteroarene),
                # otherwise its atoms stay kappa (open-chain haptics are carbon chains; this excludes O-S-O heads)
                if any(comp <= set(r) for r in rings):
                    faces.append(sorted(comp))
                continue
            if len(ends) != 2 or any(H.degree(x) > 2 for x in comp):
                faces.append(sorted(comp))  # branched (e.g. trimethylenemethane) or cyclic remainder: one face
                continue
            path = nx.shortest_path(H, ends[0], ends[1])
            pieces, cur = [], [path[0]]
            for k in range(1, len(path)):
                b_, c_ = path[k - 1], path[k]
                split = False
                if 1 <= k - 1 and k + 1 < len(path):
                    a_, d_ = path[k - 2], path[k + 1]
                    if bo(b_, c_) < 1.3 and bo(a_, b_) >= _FACE_PI_BO and bo(c_, d_) >= _FACE_PI_BO:
                        split = abs(dihedral_deg(pos(a_), pos(b_), pos(c_), pos(d_))) > HAPTIC_FOLD_DEG
                if split:
                    pieces.append(cur)
                    cur = [c_]
                else:
                    cur.append(c_)
            pieces.append(cur)
            faces.extend(sorted(p) for p in pieces if len(p) >= 2)

    # Declared donors on the far side of a bound cluster (bonded to it, or reached along the chain through such atoms)
    # that failed the bound-atom gap are the far end of that pi system, not separate coordination sites (the 2.87 A
    # ends of a bent allyl whose middle carbon is eta2-bound; the inner carbons of a sigma2 metallacyclopentene).
    bound = {x for cl in clusters for x in cl}
    ligand.dropped_donors = sorted(x for x in ligand.donor_atoms if x in G and x not in bound and x in far_end)

    out = []
    chelate_declared_donors: Set[int] = set()
    chelate_geometric_donors: Set[int] = set()
    for f in faces:
        heavy = [x for x in f if atom_map[x].symbol != "H"]
        if len(heavy) == 2 and _dc.HAPTO_CHELATE_EXCLUSION:
            is_chelate, declared_neighbors, _ = _eta2_chelate_donor_neighbors(
                mol, ligand, heavy, G, atom_map
            )
            if is_chelate:
                # Face exclusion preserves declared donor status.
                chelate_declared_donors.update(declared_neighbors)
                if _dc.DONOR_ASSIGNMENT:
                    chelate_geometric_donors.update(_supported_rejected_face_donors(
                        mol, ligand, heavy, G, atom_map, metal_bonds))
                continue
        if len(heavy) == 2 and _dc.HAPTO_BOND_GATE and not _eta2_face_bonded(mol, atom_map, heavy):
            if _dc.DONOR_ASSIGNMENT:
                chelate_geometric_donors.update(_supported_rejected_face_donors(
                    mol, ligand, heavy, G, atom_map, metal_bonds))
            # The original ligand.donor_atoms list is authoritative for kappa
            # sites. The default keeps declared donors only; reconciliation
            # may retain a separately supported local heteroatom above.
            continue
        mbo = {(min(b.i, b.j), max(b.i, b.j)): b.order for b in metal_bonds}
        ratios = _eta2_face_bond_ratios(mol, atom_map, heavy) if _dc.DONOR_ASSIGNMENT and len(heavy) >= 3 else None
        if (ratios is not None
                and max(mbo.get((min(metal_idx, x), max(metal_idx, x)), 0.0) for x in heavy) < _dc.HAPTO_FACE_BO_FLOOR
                and min(ratios) > _ETA2_RATIO_MAX()):
            # A face detected from geometry alone, with no metal bond order on any member and no member within bonding
            # distance (a phenyl of a bound phosphine/arsine lying near the metal), is a substituent, not a haptic
            # interaction. Distance counts as evidence too: ionic f-block/early-metal faces carry near-zero BO.
            events = getattr(mol, "donor_identity_events", None)
            if events is None:
                events = mol.donor_identity_events = []
            events.append({"kind": "hapto_face_unsupported", "face": sorted(heavy),
                           "contacts": [{"atom": x, "metal_bo": 0.0, "distance_ratio": r} for x, r in zip(heavy, ratios)],
                           "resolution": "no member carries a metal bond order and none is within bonding distance; not a haptic interaction"})
            continue
        cen = np.stack([pos(x) for x in heavy]).mean(0)
        nz = [v for v in (mbo.get((min(metal_idx, x), max(metal_idx, x)), 0.0) for x in heavy) if v > 0]
        out.append(HapticInteraction(metal_idx=metal_idx, donor_indices=sorted(heavy), haptic_order=len(heavy),
                                     centroid=cen, donor_symbols=[atom_map[x].symbol for x in sorted(heavy)],
                                     avg_bond_order=float(np.mean(nz)) if nz else 0.0))
    retained_chelate_donors = chelate_declared_donors | chelate_geometric_donors
    if retained_chelate_donors:
        ligand.dropped_donors = [d for d in ligand.dropped_donors if d not in retained_chelate_donors]
    if chelate_geometric_donors:
        ligand.donor_atoms = sorted(set(ligand.donor_atoms) | chelate_geometric_donors)
    return out


def _supported_rejected_face_donors(mol, ligand, face, graph, atom_map, metal_bonds):
    """Preserve independently bonded local heteroatoms when a C-X face is rejected.

    This never runs for authoritative builder groups or promotes a carbon. A
    distance cutoff alone is insufficient: a retained raw metal BO >= 0.15 is
    also required. Candidate atoms share the rejected face's carbon center;
    no other part of the ligand can acquire a new donor through this rule.
    """
    from jacobik.stats.orientation import ETA2_BOND_RATIO_MAX

    carbons = [i for i in face if atom_map[i].symbol == "C"]
    if len(face) != 2 or len(carbons) != 1 or ligand.donors_are_authoritative:
        return set()
    carbon = carbons[0]
    metal_bo = {b.j if b.i == mol.metal_idx else b.i: b.order for b in metal_bonds
                if mol.metal_idx in (b.i, b.j)}
    candidates = {i for i in graph.neighbors(carbon) if atom_map[i].symbol in {"N", "O", "S"}}
    supported = {i for i in candidates if metal_bo.get(i, 0.0) >= 0.15 and
                 _eta2_face_bond_ratios(mol, atom_map, [i])[0] <= ETA2_BOND_RATIO_MAX}
    added = supported - set(ligand.donor_atoms)
    if added:
        events = getattr(mol, "donor_identity_events", None)
        if events is None:
            events = mol.donor_identity_events = []
        events.append({"kind": "rejected_face_kappa_preservation", "face": sorted(face),
                       "added": sorted(added), "center": carbon,
                       "contacts": [{"atom": i, "metal_bo": metal_bo[i],
                                     "distance_ratio": _eta2_face_bond_ratios(mol, atom_map, [i])[0]}
                                    for i in sorted(added)]})
    return supported


def _ETA2_RATIO_MAX() -> float:
    from jacobik.stats.orientation import ETA2_BOND_RATIO_MAX
    return ETA2_BOND_RATIO_MAX


def _eta2_face_bond_ratios(mol: TMComplex, atom_map: Dict[int, Atom], face: Sequence[int]) -> List[float]:
    """Metal-scaled covalent-contact ratios for an eta2 face, in face atom order."""
    from jacobik.core.chemistry import covalent_radius as _covalent_radius

    metal_pos = np.asarray(atom_map[mol.metal_idx].position, float)
    metal_radius = _covalent_radius(mol.metal_symbol)
    return [float(np.linalg.norm(np.asarray(atom_map[i].position, float) - metal_pos)
                  / (metal_radius + _covalent_radius(atom_map[i].symbol))) for i in face]


def _eta2_face_bonded(mol: TMComplex, atom_map: Dict[int, Atom], face: Sequence[int]) -> bool:
    """Whether every atom in a two-atom face is within orientation's bond-ratio gate."""
    from jacobik.stats.orientation import ETA2_BOND_RATIO_MAX

    return max(_eta2_face_bond_ratios(mol, atom_map, face)) <= ETA2_BOND_RATIO_MAX


def _eta2_chelate_donor_neighbors(
    mol: TMComplex,
    ligand: Ligand,
    face: Sequence[int],
    ligand_graph: nx.Graph,
    atom_map: Dict[int, Atom],
) -> Tuple[bool, Set[int], Set[int]]:
    """Whether a C-containing eta2 pair closes a chelate and its bound κ neighbors.

    A candidate {C, X} is excluded when the face carbon has at least two
    neighbors that are either declared as donors of this metal or pass the
    eta2 covalent-radius bond-ratio gate. The donor list is populated from
    raw-coordinate donors on the fallback path and mapped SMILES donors on
    the SMILES path. Return declared neighbors and face-bound geometric neighbors separately.
    """
    if len(face) != 2:
        return False, set(), set()
    carbons = [i for i in face if atom_map[i].symbol == "C"]
    if len(carbons) != 1:
        return False, set(), set()
    carbon = carbons[0]
    declared_neighbors = set(ligand_graph.neighbors(carbon)) & set(ligand.donor_atoms)
    from jacobik.stats.orientation import ETA2_BOND_RATIO_MAX

    geometric_neighbors = set()
    for neighbor in ligand_graph.neighbors(carbon):
        ratio = _eta2_face_bond_ratios(mol, atom_map, [neighbor])[0]
        if ratio <= ETA2_BOND_RATIO_MAX:
            geometric_neighbors.add(neighbor)
    bound_neighbors = declared_neighbors | geometric_neighbors
    return len(bound_neighbors) >= 2, declared_neighbors, geometric_neighbors
