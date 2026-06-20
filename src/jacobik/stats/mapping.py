"""
mapping.py
==========
Synthesizes raw tmQM data into fully-assembled TMComplex objects.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from jacobik.core.types import (
    Atom,
    Bond,
    Ligand,
    TMComplex,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metal element set
# ---------------------------------------------------------------------------

METALS: FrozenSet[str] = frozenset(
    {
        # d-block (Groups 3-12)
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        # Lanthanides
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
        # Actinides
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
    }
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bonds above this raw BO threshold are treated as real ligand-internal bonds
# when building the connectivity graph for component decomposition.
_LIGAND_BO_THRESHOLD: float = 0.5

# Raw BO thresholds for the no-SMILES fallback quantizer.
# These are applied after metal and aromatic-ring detection.
_BO_SINGLE_LO: float = 0.75  # raw BO >= this -> single (below = coordination)
_BO_DOUBLE_LO: float = 1.30  # raw BO >= this -> tentatively double
_BO_AROM_LO: float = 1.05  # used within rings for aromatic detection
_BO_AROM_HI: float = 1.85
_BO_TRIPLE_LO: float = 2.30  # raw BO >= this -> triple

# Fallback-only distance sanity screening (covalent-radii based).
# Applied only when SMILES is absent to suppress long through-space contacts.
_DIST_MIN_SCALE: float = 0.55
_DIST_MIN_PAD: float = 0.20
_DIST_MAX_NONMETAL_SCALE: float = 1.25
_DIST_MAX_NONMETAL_PAD: float = 0.25
_DIST_MAX_METAL_SCALE: float = 1.55
_DIST_MAX_METAL_PAD: float = 0.45

_COVALENT_RADII: Dict[str, float] = {
    # Main-group / common ligand atoms
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Al": 1.21,
    "Ga": 1.22,
    "In": 1.42,
    "Sn": 1.39,
    "Li": 1.28,
    "Na": 1.66,
    "K": 2.03,
    "Rb": 2.20,
    "Cs": 2.44,
    "Be": 0.96,
    "Mg": 1.41,
    "Ca": 1.76,
    "Sr": 1.95,
    "Ba": 2.15,
    # d-block metals in tmQM
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    # Lanthanides + common actinides
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Ac": 2.15,
    "Th": 2.06,
    "Pa": 2.00,
    "U": 1.96,
    "Np": 1.90,
    "Pu": 1.87,
}


# ---------------------------------------------------------------------------
# Internal helpers: metal identification
# ---------------------------------------------------------------------------


def _find_metal(atoms: List[Atom]) -> Atom:
    """
    Return the unique metal atom in the complex.
    Raises ValueError if zero or more than one metal is present.
    """
    metals = [a for a in atoms if a.symbol in METALS]
    if len(metals) == 1:
        return metals[0]
    if len(metals) == 0:
        raise ValueError("No transition metal or lanthanide found in complex.")
    raise ValueError(
        f"Expected exactly one metal, found {len(metals)}: {[(m.idx, m.symbol) for m in metals]}"
    )


# ---------------------------------------------------------------------------
# Internal helpers: graph construction
# ---------------------------------------------------------------------------

# Removed _build_bo_heavy_graph, now imported from jacobik.topology.graph


def _build_bo_full_graph(
    atoms: List[Atom],
    bonds: List[Bond],
    metal_idx: int,
) -> nx.Graph:
    """
    Undirected graph of ALL non-metal atoms (including H).
    Edges for bonds with raw BO >= _LIGAND_BO_THRESHOLD (excluding metal bonds).
    Used for H-parent lookup after heavy-atom mapping.
    Node attribute: 'symbol'.
    """
    G: nx.Graph = nx.Graph()
    for a in atoms:
        if a.idx == metal_idx:
            continue
        G.add_node(a.idx, symbol=a.symbol)

    for b in bonds:
        if b.i == metal_idx or b.j == metal_idx:
            continue
        if b.order < _LIGAND_BO_THRESHOLD:
            continue
        G.add_edge(b.i, b.j)

    return G


def _covalent_radius(symbol: str) -> float:
    """Return a covalent radius in angstrom with robust fallback defaults."""
    r = _COVALENT_RADII.get(symbol)
    if r is not None:
        return r
    return 1.45 if symbol in METALS else 0.77


def _distance_sanity_pass(
    atom_i: Atom,
    atom_j: Atom,
    is_metal_bond: bool,
) -> bool:
    """
    Fallback-only geometric plausibility check for a candidate bond.
    """
    d = float(np.linalg.norm(atom_i.position - atom_j.position))
    r_sum = _covalent_radius(atom_i.symbol) + _covalent_radius(atom_j.symbol)

    low = max(0.4, _DIST_MIN_SCALE * r_sum - _DIST_MIN_PAD)
    if is_metal_bond:
        high = _DIST_MAX_METAL_SCALE * r_sum + _DIST_MAX_METAL_PAD
    else:
        high = _DIST_MAX_NONMETAL_SCALE * r_sum + _DIST_MAX_NONMETAL_PAD

    return low <= d <= high


# ---------------------------------------------------------------------------
# Internal helpers: SMILES path
# ---------------------------------------------------------------------------


def _smiles_to_rdkit(smiles: str):
    """
    Parse SMILES and add explicit hydrogens.
    Returns (mol_no_h, mol_with_h) or raises ValueError on parse failure.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for the SMILES alignment path.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol_h = Chem.AddHs(mol)
    return mol, mol_h


def _build_smiles_heavy_graph(mol) -> Tuple[nx.Graph, int, Dict[int, int]]:
    """
    Build a heavy-atom ligand graph from an RDKit mol (no explicit H needed).

    Returns
    -------
    G_smi   : nx.Graph, nodes are SMILES atom indices, attr 'symbol'
    metal_smi_idx : int, SMILES index of the metal atom
    donor_smi_map : dict smiles_donor_idx -> smiles_metal_idx
                    (donors identified from dative bonds)
    """
    from rdkit.Chem import rdchem

    metal_smi_idx: int = -1
    for a in mol.GetAtoms():
        if a.GetSymbol() in METALS:
            metal_smi_idx = a.GetIdx()
            break

    if metal_smi_idx == -1:
        raise ValueError("No metal atom found in SMILES mol.")

    G: nx.Graph = nx.Graph()
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym in METALS or sym == "H":
            continue
        G.add_node(a.GetIdx(), symbol=sym)

    donor_smi_map: Dict[int, int] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()

        if bt == rdchem.BondType.DATIVE:
            # Record donor; do not add edge to ligand graph
            donor_idx = i if j == metal_smi_idx else j
            donor_smi_map[donor_idx] = metal_smi_idx
            continue

        if i == metal_smi_idx or j == metal_smi_idx:
            continue

        # Skip H nodes
        a_i = mol.GetAtomWithIdx(i)
        a_j = mol.GetAtomWithIdx(j)
        if a_i.GetSymbol() == "H" or a_j.GetSymbol() == "H":
            continue

        G.add_edge(i, j)

    return G, metal_smi_idx, donor_smi_map


def _smiles_bond_order_table(mol) -> Dict[Tuple[int, int], float]:
    """
    Return a dict (min_i, max_j) -> discrete_bo for all non-dative,
    non-metal, non-H bonds in an RDKit mol (without explicit H).
    """
    from rdkit.Chem import rdchem

    metal_smi_idx: int = -1
    for a in mol.GetAtoms():
        if a.GetSymbol() in METALS:
            metal_smi_idx = a.GetIdx()
            break

    result: Dict[Tuple[int, int], float] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()

        if bt == rdchem.BondType.DATIVE:
            continue
        if i == metal_smi_idx or j == metal_smi_idx:
            continue
        if mol.GetAtomWithIdx(i).GetSymbol() == "H":
            continue
        if mol.GetAtomWithIdx(j).GetSymbol() == "H":
            continue

        if bt == rdchem.BondType.AROMATIC:
            bo = 1.5
        else:
            bo = bond.GetBondTypeAsDouble()

        result[(min(i, j), max(i, j))] = bo

    return result


# ---------------------------------------------------------------------------
# Internal helpers: VF2 graph isomorphism alignment
# ---------------------------------------------------------------------------


def _element_multiset(G: nx.Graph) -> Counter:
    return Counter(data["symbol"] for _, data in G.nodes(data=True))


def _align_components(
    G_smi: nx.Graph,
    G_bo: nx.Graph,
) -> Dict[int, int]:
    """
    Align heavy-atom connected components between the SMILES graph and the
    BO graph using VF2 graph isomorphism.

    Returns
    -------
    mapping : dict smiles_heavy_idx -> xyz_heavy_idx

    Raises
    ------
    ValueError if a SMILES component has no isomorphic match in the BO graph.
    """

    def _node_match(n1: dict, n2: dict) -> bool:
        return n1["symbol"] == n2["symbol"]

    smi_components = [G_smi.subgraph(c).copy() for c in nx.connected_components(G_smi)]
    bo_components = [G_bo.subgraph(c).copy() for c in nx.connected_components(G_bo)]

    # Index BO components by element multiset for O(1) candidate lookup
    bo_by_multiset: Dict[FrozenSet, List[nx.Graph]] = {}
    for comp in bo_components:
        key = frozenset(_element_multiset(comp).items())
        bo_by_multiset.setdefault(key, []).append(comp)

    full_mapping: Dict[int, int] = {}

    for smi_comp in smi_components:
        key = frozenset(_element_multiset(smi_comp).items())
        candidates = bo_by_multiset.get(key, [])

        if not candidates:
            raise ValueError(
                f"No BO-graph component matches SMILES component with element set {dict(key)}."
            )

        matched = False
        for bo_comp in candidates:
            GM = nx.algorithms.isomorphism.GraphMatcher(smi_comp, bo_comp, node_match=_node_match)
            mapping = next(GM.isomorphisms_iter(), None)
            if mapping is not None:
                full_mapping.update(mapping)
                # Mark this BO component as used (remove from candidates pool)
                candidates.remove(bo_comp)
                matched = True
                break

        if not matched:
            raise ValueError(
                f"VF2 isomorphism failed for SMILES component with element set {dict(key)}."
            )

    return full_mapping  # smiles_heavy_idx -> xyz_heavy_idx


def _assign_hydrogens(
    full_mapping: Dict[int, int],
    mol_h,  # RDKit mol with explicit H
    G_bo_full: nx.Graph,  # BO graph including H atoms
) -> Dict[int, int]:
    """
    Extend the heavy-atom mapping to include H atoms.


    Returns
    -------
    h_mapping : dict smiles_H_idx -> xyz_H_idx

    """
    h_mapping: Dict[int, int] = {}

    for smi_heavy, xyz_heavy in full_mapping.items():
        smi_atom = mol_h.GetAtomWithIdx(smi_heavy)
        smi_h_neighbors = sorted(
            nb.GetIdx() for nb in smi_atom.GetNeighbors() if nb.GetSymbol() == "H"
        )

        xyz_h_neighbors = sorted(
            nb for nb in G_bo_full.neighbors(xyz_heavy) if G_bo_full.nodes[nb]["symbol"] == "H"
        )

        if len(smi_h_neighbors) != len(xyz_h_neighbors):
            log.debug(
                "H count mismatch at heavy atom xyz=%d: SMILES has %d H, BO has %d H. Skipping.",
                xyz_heavy,
                len(smi_h_neighbors),
                len(xyz_h_neighbors),
            )
            continue

        for smi_h, xyz_h in zip(smi_h_neighbors, xyz_h_neighbors):
            h_mapping[smi_h] = xyz_h

    return h_mapping


# ---------------------------------------------------------------------------
# SMILES path: full alignment + bond order transfer
# ---------------------------------------------------------------------------


def _apply_smiles_path(
    atoms: List[Atom],
    bonds: List[Bond],
    metal_idx: int,
    smiles: str,
) -> Tuple[Dict[Tuple[int, int], float], Set[int]]:
    """
    Parse SMILES, align to the .BO atom set, and return:
      - bo_updates : dict (min_xyz_i, max_xyz_j) -> discrete_bo
                     for all ligand-internal bonds (NOT metal bonds)
      - xyz_donor_indices : set of xyz atom indices that are donors
    """
    mol, mol_h = _smiles_to_rdkit(smiles)

    G_smi, metal_smi_idx, donor_smi_map = _build_smiles_heavy_graph(mol)
    smiles_bo_table = _smiles_bond_order_table(mol)

    from jacobik.topology.graph import build_heavy_ligand_graph

    G_bo_heavy = build_heavy_ligand_graph(
        atoms, bonds, metal_idx, bo_threshold=_LIGAND_BO_THRESHOLD
    )
    G_bo_full = _build_bo_full_graph(atoms, bonds, metal_idx)

    # Heavy-atom isomorphism: smiles_idx -> xyz_idx
    heavy_mapping = _align_components(G_smi, G_bo_heavy)

    # Extend to H atoms
    h_mapping = _assign_hydrogens(heavy_mapping, mol_h, G_bo_full)

    # Combined: smiles_idx -> xyz_idx (heavy + H)
    full_map: Dict[int, int] = {**heavy_mapping, **h_mapping}

    # Build discrete BO updates keyed by xyz indices
    bo_updates: Dict[Tuple[int, int], float] = {}
    for (smi_i, smi_j), bo in smiles_bo_table.items():
        xyz_i = full_map.get(smi_i)
        xyz_j = full_map.get(smi_j)
        if xyz_i is None or xyz_j is None:
            continue
        bo_updates[(min(xyz_i, xyz_j), max(xyz_i, xyz_j))] = bo

    # See exp.md for detailed rationale

    # Map raw BO indexed by (min, max) pair for fast proximity remap lookup
    raw_bo_lookup: Dict[Tuple[int, int], float] = {
        (min(b.i, b.j), max(b.i, b.j)): b.order for b in bonds
    }

    atom_map_local: Dict[int, Atom] = {a.idx: a for a in atoms}

    # Build component membership in the heavy BO graph for proximity remap
    bo_components: Dict[int, Set[int]] = {}
    for comp in nx.connected_components(G_bo_heavy):
        comp_set = set(comp)
        for node in comp_set:
            bo_components[node] = comp_set

    _COORD_BO_SUPPLEMENT: float = 0.15

    xyz_donor_indices: Set[int] = set()

    for smi_donor_idx in donor_smi_map:
        xyz_donor = full_map.get(smi_donor_idx)
        if xyz_donor is None:
            continue

        # See exp.md for detailed rationale
        mapped_bo = raw_bo_lookup.get((min(metal_idx, xyz_donor), max(metal_idx, xyz_donor)), 0.0)
        if mapped_bo < _COORD_BO_SUPPLEMENT:
            comp = bo_components.get(xyz_donor, {xyz_donor})
            best_atom = xyz_donor
            best_bo = mapped_bo
            for candidate in comp:
                cand_bo = raw_bo_lookup.get(
                    (min(metal_idx, candidate), max(metal_idx, candidate)), 0.0
                )
                if cand_bo > best_bo:
                    best_bo = cand_bo
                    best_atom = candidate
            xyz_donor = best_atom

        # Only add non-H atoms — H is only a valid donor when the SMILES
        # explicitly encodes a dative bond from H (true hydrides).
        # This also guards against the proximity remap accidentally landing
        # on an H neighbor of the best ring atom.
        atom_d = atom_map_local.get(xyz_donor)
        if atom_d is not None and atom_d.symbol == "H":
            continue

        xyz_donor_indices.add(xyz_donor)

    return bo_updates, xyz_donor_indices


# ---------------------------------------------------------------------------
# Fallback path: heuristic BO quantization (no SMILES)
# ---------------------------------------------------------------------------


def _detect_aromatic_rings_fallback(
    atoms: List[Atom],
    bonds: List[Bond],
    metal_idx: int,
) -> Set[Tuple[int, int]]:
    """
    Return the set of (min_i, max_j) bond keys that belong to aromatic rings,
    detected purely from raw bond orders.
    """
    from jacobik.topology.graph import build_heavy_ligand_graph

    G = build_heavy_ligand_graph(atoms, bonds, metal_idx, bo_threshold=_LIGAND_BO_THRESHOLD)
    bo_map: Dict[Tuple[int, int], float] = {
        (min(b.i, b.j), max(b.i, b.j)): b.order
        for b in bonds
        if b.i != metal_idx and b.j != metal_idx
    }

    aromatic_bond_keys: Set[Tuple[int, int]] = set()

    for cycle in nx.minimum_cycle_basis(G):
        if len(cycle) not in (5, 6, 7):
            continue
        cycle_set = set(cycle)
        ring_bonds = [(u, v) for u, v in G.edges(cycle) if u in cycle_set and v in cycle_set]
        if not ring_bonds:
            continue
        bos = [bo_map.get((min(u, v), max(u, v)), 0.0) for u, v in ring_bonds]
        if all(_BO_AROM_LO <= bo <= _BO_AROM_HI for bo in bos):
            for u, v in ring_bonds:
                aromatic_bond_keys.add((min(u, v), max(u, v)))

    return aromatic_bond_keys


def _quantize_single_bo(raw_bo: float, is_aromatic: bool = False) -> float:
    """
    Map one raw bond order to a discrete value for a ligand-internal bond.
    Metal-touching bonds are never passed here (they keep raw BOs).
    """
    if is_aromatic:
        return 1.5
    if raw_bo < _BO_SINGLE_LO:
        return 1.0  # sub-single but above floor: treat as single (e.g. C-H 0.93)
    if raw_bo < _BO_DOUBLE_LO:
        return 1.0  # single
    if raw_bo < _BO_TRIPLE_LO:
        return 2.0  # double
    return 3.0  # triple


def _apply_fallback_path(
    atoms: List[Atom],
    bonds: List[Bond],
    metal_idx: int,
) -> Tuple[Dict[Tuple[int, int], float], Set[int], Set[Tuple[int, int]]]:
    """
    Quantize raw bond orders without SMILES guidance.

    Returns
    -------
    bo_updates : dict (min_xyz_i, max_xyz_j) -> discrete_bo
                 for ligand-internal bonds that pass distance sanity checks
    xyz_donor_indices : set of xyz atom indices inferred as donors
                        (metal neighbors that also pass distance sanity checks)
    allowed_coord_keys : set of (min_xyz_i, max_xyz_j) for metal-touching
                        bonds that pass distance sanity checks and should be kept
    """
    atom_map: Dict[int, Atom] = {a.idx: a for a in atoms}
    aromatic_keys = _detect_aromatic_rings_fallback(atoms, bonds, metal_idx)

    bo_updates: Dict[Tuple[int, int], float] = {}
    xyz_donor_indices: Set[int] = set()
    allowed_coord_keys: Set[Tuple[int, int]] = set()

    for b in bonds:
        key = (min(b.i, b.j), max(b.i, b.j))
        ai = atom_map.get(b.i)
        aj = atom_map.get(b.j)
        if ai is None or aj is None:
            continue

        # Metal-touching bonds: keep raw BO, record donors
        if b.i == metal_idx or b.j == metal_idx:
            if not _distance_sanity_pass(ai, aj, is_metal_bond=True):
                continue
            allowed_coord_keys.add(key)
            donor_idx = b.j if b.i == metal_idx else b.i
            atom_d = atom_map.get(donor_idx)
            # H atoms are only real donors (hydrides) when their raw BO
            # to the metal is firmly in single-bond territory (>= 0.75).
            # Weaker contacts are agostic or noise and should not produce
            # coord bond visualizations.
            if atom_d is not None and atom_d.symbol == "H" and b.order < 0.75:
                continue
            xyz_donor_indices.add(donor_idx)
            # Raw BO preserved on the Bond object itself; no update needed
            continue

        if not _distance_sanity_pass(ai, aj, is_metal_bond=False):
            continue

        is_aromatic = key in aromatic_keys
        bo_updates[key] = _quantize_single_bo(b.order, is_aromatic)

    return bo_updates, xyz_donor_indices, allowed_coord_keys


# ---------------------------------------------------------------------------
# Bond order application
# ---------------------------------------------------------------------------


def _apply_bo_updates(
    bonds: List[Bond],
    bo_updates: Dict[Tuple[int, int], float],
    metal_idx: int,
    smiles_path: bool,
    allowed_coord_keys: Optional[Set[Tuple[int, int]]] = None,
    atom_map: Optional[Dict[int, "Atom"]] = None,
) -> List[Bond]:
    """
    Return a new list of Bond objects with finalised bond orders.
    """
    updated: List[Bond] = []
    for b in bonds:
        key = (min(b.i, b.j), max(b.i, b.j))
        is_metal_bond = b.i == metal_idx or b.j == metal_idx

        if smiles_path:
            if is_metal_bond:
                updated.append(Bond(i=b.i, j=b.j, order=b.order))
            elif key in bo_updates:
                updated.append(Bond(i=b.i, j=b.j, order=bo_updates[key]))
            elif (
                atom_map is not None
                and b.order >= _LIGAND_BO_THRESHOLD
                and (atom_map.get(b.i, None) is not None and atom_map.get(b.j, None) is not None)
                and (atom_map[b.i].symbol == "H" or atom_map[b.j].symbol == "H")
            ):
                # H-heavy bonds never appear in smiles_bo_table; re-admit as
                # single bonds so H atoms are not left as isolated ligand nodes.
                updated.append(Bond(i=b.i, j=b.j, order=1.0))
            # else: heavy-heavy bond with no SMILES entry -> not a real bond -> drop
        else:
            if is_metal_bond:
                if allowed_coord_keys is not None and key not in allowed_coord_keys:
                    continue
                updated.append(Bond(i=b.i, j=b.j, order=b.order))
                continue

            if key not in bo_updates:
                continue
            new_order = bo_updates.get(key, b.order)
            updated.append(Bond(i=b.i, j=b.j, order=new_order))

    return updated


# ---------------------------------------------------------------------------
# Ligand extraction
# ---------------------------------------------------------------------------


def _extract_ligands(
    atoms: List[Atom],
    bonds: List[Bond],
    metal_idx: int,
    xyz_donor_indices: Set[int],
) -> List[Ligand]:
    """
    Identify connected components of the ligand graph (all non-metal atoms,
    edges for bonds with quantized BO >= 0.5) and build Ligand objects.
    """
    G = nx.Graph()
    for a in atoms:
        if a.idx != metal_idx:
            G.add_node(a.idx)

    for b in bonds:
        if b.i == metal_idx or b.j == metal_idx:
            continue
        if b.order >= _LIGAND_BO_THRESHOLD:
            G.add_edge(b.i, b.j)

    ligands: List[Ligand] = []
    for lig_id, component in enumerate(nx.connected_components(G), start=1):
        donor_atoms = sorted(xyz_donor_indices & component)
        ligands.append(
            Ligand(
                ligand_id=lig_id,
                atom_indices=sorted(component),
                donor_atoms=donor_atoms,
            )
        )

    return ligands


# ---------------------------------------------------------------------------
# Public assembly entry point
# ---------------------------------------------------------------------------


def assemble(raw) -> TMComplex:
    """
    Assemble a TMComplex from a _RawMolecule produced by parser.py.

    Parameters
    ----------
    raw : parser._RawMolecule
        Contains atoms, bonds (raw BOs), metadata, and optional SMILES.

    Returns
    -------
    TMComplex
        Fully assembled with quantized bond orders, coord_bonds, ligands.
        donor_labels and haptic_interactions are populated by
        donor_classifier.classify() called at the end.
        Ligand.decomposition is populated by decompose.decompose_ligand()
        called at the end.
        QM properties are all None; parser.py attaches them afterwards.
    """
    atoms: List[Atom] = raw.atoms
    bonds: List[Bond] = raw.bonds

    # --- 1. Identify metal ---
    metal_atom = _find_metal(atoms)
    metal_idx = metal_atom.idx

    # --- 2. Choose path: SMILES or fallback ---
    bo_updates: Dict[Tuple[int, int], float]
    xyz_donor_indices: Set[int]
    allowed_coord_keys: Optional[Set[Tuple[int, int]]] = None
    used_smiles = False

    if raw.smiles:
        try:
            bo_updates, xyz_donor_indices = _apply_smiles_path(atoms, bonds, metal_idx, raw.smiles)
            used_smiles = True
        except Exception as exc:
            log.warning(
                "%s: SMILES alignment failed (%s), falling back to heuristic.",
                raw.csd_code,
                exc,
            )
            bo_updates, xyz_donor_indices, allowed_coord_keys = _apply_fallback_path(
                atoms, bonds, metal_idx
            )
    else:
        bo_updates, xyz_donor_indices, allowed_coord_keys = _apply_fallback_path(
            atoms, bonds, metal_idx
        )

    log.debug(
        "%s: %s path, %d bond updates, %d donor atoms.",
        raw.csd_code,
        "SMILES" if used_smiles else "fallback",
        len(bo_updates),
        len(xyz_donor_indices),
    )

    # --- 3. Apply discrete bond orders ---
    _atom_map: Dict[int, Atom] = {a.idx: a for a in atoms}
    updated_bonds = _apply_bo_updates(
        bonds,
        bo_updates,
        metal_idx,
        used_smiles,
        allowed_coord_keys=allowed_coord_keys,
        atom_map=_atom_map,
    )

    # --- 4. Partition coord bonds ---
    coord_bonds = [b for b in updated_bonds if b.i == metal_idx or b.j == metal_idx]

    # --- 5. Extract ligands ---
    ligands = _extract_ligands(atoms, updated_bonds, metal_idx, xyz_donor_indices)

    # Raw metal-touching bonds: intermediate data used only to seed haptic
    # detection and compute mean metal-donor bond orders during
    # classification.  Deliberately NOT stored on the TMComplex — the verified
    # coord_sites produced by classify() are the authoritative record.
    metal_bonds = [b for b in updated_bonds if b.i == metal_idx or b.j == metal_idx]

    # --- 6. Assemble TMComplex shell ---
    mol = TMComplex(
        csd_code=raw.csd_code,
        charge=raw.charge,
        spin_state=raw.spin_state,
        stoichiometry=raw.stoichiometry,
        mnd=raw.mnd,
        csd_years=raw.csd_years,
        atoms=atoms,
        bonds=updated_bonds,
        metal_idx=metal_idx,
        metal_symbol=metal_atom.symbol,
        ligands=ligands,
    )

    # --- 7. Donor classification + haptic detection ---
    # Populates mol.donor_labels and mol.coord_sites in-place.  metal_bonds is
    # passed in (not stored) so classification can read raw metal BOs.
    try:
        from jacobik.topology.donor_classifier import classify

        classify(mol, metal_bonds=metal_bonds, smiles_path=used_smiles)
    except Exception as exc:
        log.warning("%s: donor classification failed: %s", raw.csd_code, exc)

    # --- 8. Ligand decomposition ---
    try:
        from jacobik.topology.decompose import decompose_ligand

        for lig in mol.ligands:
            try:
                decompose_ligand(mol, lig)
            except Exception as exc:
                log.warning(
                    "%s ligand %d: decomposition failed: %s",
                    raw.csd_code,
                    lig.ligand_id,
                    exc,
                )
    except ImportError:
        pass  # decompose not yet available

    # --- 9. Coordination geometry classification ---
    # Requires coord_sites to be populated (done in step 7).
    try:
        from jacobik.topology.coord_geom import classify as classify_geom

        mol.coord_geom = classify_geom(mol)
    except Exception as exc:
        log.warning("%s: geometry classification failed: %s", raw.csd_code, exc)

    return mol
