"""Donor classification taxonomy (hierarchy paths), atom classifier, and haptic face classification."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from jacobik.core.types import Atom, Bond, CoordinationSite, DonorLabel, HapticInteraction, Ligand, TMComplex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Donor taxonomy hierarchy
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

#: Global catch-all — topmost node in the donor dimension.
ELEMENT_ANY = "element_any"

# Child -> parent edges; every chain ends at ELEMENT_ANY.
_PARENT: Dict[str, str] = {
    # -----------------------------------------------------------------------
    # Carbon
    # -----------------------------------------------------------------------
    "C_donor": ELEMENT_ANY,
    # bucket: sigma donors
    "C_sigma": "C_donor",
    "C_carbonyl": "C_sigma",
    "C_isocyanide": "C_sigma",
    "C_cyanide": "C_sigma",
    "C_alkyl": "C_sigma",
    "C_acyl": "C_sigma",
    "C_iminoacyl": "C_sigma",
    "C_vinyl": "C_sigma",
    "C_acetylide": "C_sigma",
    "C_thiocarbonyl": "C_sigma",
    # bucket: metal-carbon multiple bonds (carbenes, alkylidenes, vinylidenes, alkylidynes)
    "C_mbond": "C_donor",
    "C_carbene": "C_mbond",
    "C_alkylidene": "C_mbond",
    "C_vinylidene": "C_mbond",
    "C_alkylidyne": "C_mbond",
    # bucket: kappa pi
    "C_pi_sigma": "C_donor",
    "C_pi": "C_pi_sigma",
    # bucket: haptic faces
    "C_haptic_any": "C_donor",
    "C_allyl": "C_haptic_any",  # η³
    "C_eta4": "C_haptic_any",  # η⁴
    "C_eta5": "C_haptic_any",  # η⁵ (Cp)
    "C_eta6": "C_haptic_any",  # η⁶ (arene)
    "C_eta7": "C_haptic_any",  # η⁷ (C7H7, tropylium)
    "C_eta8": "C_haptic_any",  # η⁸ (COT)
    # eta-2 and any unrecognised order
    "C_haptic": "C_haptic_any",
    # -----------------------------------------------------------------------
    # Nitrogen
    # -----------------------------------------------------------------------
    "N_donor": ELEMENT_ANY,
    # bucket: linear/terminal
    "N_linear": "N_donor",
    "N_dinitrogen": "N_linear",
    "N_cyanide": "N_linear",
    "N_nitrosyl": "N_linear",
    # bucket: aromatic
    "N_aromatic_any": "N_donor",
    "N_NHC": "N_aromatic_any",
    "N_pyridine": "N_aromatic_any",
    "N_azole": "N_aromatic_any",
    # fallback subclass string
    "N_aromatic": "N_aromatic_any",
    # bucket: sp3-like (flexible approach geometry)
    "N_sp3_any": "N_donor",
    "N_imine": "N_sp3_any",
    "N_amide": "N_sp3_any",
    "N_amine": "N_sp3_any",  # bucket / generic fallback
    "N_ammonia": "N_amine",
    "N_primary_amine": "N_amine",
    "N_secondary_amine": "N_amine",
    "N_tertiary_amine": "N_amine",
    "N_ammonium_like": "N_amine",
    "N_amido": "N_sp3_any",  # anionic, two substituents
    "N_imido": "N_sp3_any",  # terminal, <=1 substituent
    "N_amidinate": "N_sp3_any",     # resonance-shared N of N-C-N / N-N-N chelates
    "N_amidate": "N_sp3_any",
    "N_diketiminate": "N_sp3_any",  # nacnac
    "N_ketoiminate": "N_sp3_any",
    # -----------------------------------------------------------------------
    # Oxygen
    # -----------------------------------------------------------------------
    "O_donor": ELEMENT_ANY,
    # bucket: anionic / delocalized
    "O_anionic_any": "O_donor",
    "O_carboxylate": "O_anionic_any",
    "O_sulfonate": "O_anionic_any",
    "O_nitro": "O_anionic_any",
    "O_hydroxide": "O_anionic_any",  # M-OH- (no heavy neighbour, one H)
    "O_oxo": "O_anionic_any",        # M=O / M-O- (no heavy neighbour, no H)
    "O_phenoxide": "O_anionic_any",
    "O_diketonate": "O_anionic_any",   # beta-diketonate O (acac), resonance-shared
    "O_ketoiminate": "O_anionic_any",
    "O_amidate": "O_anionic_any",
    "O_dioxolene": "O_anionic_any",    # catecholate / semiquinone / quinone O, resonance-shared
    # bucket: neutral sp2 (lone pair on pi oxygen)
    "O_neutral_pi": "O_donor",
    "O_carbonyl": "O_neutral_pi",
    "O_heteroxo": "O_neutral_pi",  # P=O, S=O, N-oxide ... (terminal, non-C)
    # bucket: neutral sp3 (lone pair on sigma oxygen)
    "O_neutral_sigma": "O_donor",
    "O_water": "O_neutral_sigma",
    "O_alcohol": "O_neutral_sigma",
    "O_ether": "O_neutral_sigma",
    "O_ether_cyclic": "O_neutral_sigma",
    "O_alkoxide": "O_neutral_sigma",
    # -----------------------------------------------------------------------
    # Phosphorus  (no intermediate bucket; all three pool directly to P_donor)
    # -----------------------------------------------------------------------
    "P_donor": ELEMENT_ANY,
    "P_phosphine": "P_donor",
    "P_phosphite": "P_donor",
    "P_phosphinite": "P_donor",
    # -----------------------------------------------------------------------
    # Sulfur
    # -----------------------------------------------------------------------
    "S_donor": ELEMENT_ANY,
    "S_pi_any": "S_donor",
    "S_sulfoxide": "S_pi_any",
    "S_thiocarbonyl": "S_pi_any",
    "S_sigma_any": "S_donor",
    "S_thioether": "S_sigma_any",
    "S_thiolate": "S_sigma_any",
    # -----------------------------------------------------------------------
    # Halogens  (shallow: subclass -> family -> element_any)
    # -----------------------------------------------------------------------
    "Hal_donor": ELEMENT_ANY,
    "Hal_F": "Hal_donor",
    "Hal_Cl": "Hal_donor",
    "Hal_Br": "Hal_donor",
    "Hal_I": "Hal_donor",
    # -----------------------------------------------------------------------
    # Hydrogen  (shallow)
    # -----------------------------------------------------------------------
    "H_donor": ELEMENT_ANY,
    "H_hydride": "H_donor",
    # -----------------------------------------------------------------------
    # Boron  (shallow)
    # -----------------------------------------------------------------------
    "B_donor": ELEMENT_ANY,
    "B_boronate": "B_donor",
    # -----------------------------------------------------------------------
    # Chalcogenides / pnictogens  (no subclasses currently; leaf = family)
    # -----------------------------------------------------------------------
    "Se_donor": ELEMENT_ANY,
    "Te_donor": ELEMENT_ANY,
    "As_donor": ELEMENT_ANY,
}

# ---------------------------------------------------------------------------
# Element symbol -> family node
# ---------------------------------------------------------------------------

_ELEMENT_TO_FAMILY: Dict[str, str] = {
    "C": "C_donor",
    "N": "N_donor",
    "O": "O_donor",
    "P": "P_donor",
    "S": "S_donor",
    "F": "Hal_donor",
    "Cl": "Hal_donor",
    "Br": "Hal_donor",
    "I": "Hal_donor",
    "B": "B_donor",
    "Se": "Se_donor",
    "Te": "Te_donor",
    "As": "As_donor",
    "H": "H_donor",
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def ancestors(node: str) -> List[str]:
    """
    Return the ordered ancestry chain above *node*, from its immediate parent
    up to and including ELEMENT_ANY.

    Raises KeyError if *node* is not in the hierarchy.

    Examples
    --------
    >>> ancestors("N_pyridine")
    ['N_aromatic_any', 'N_donor', 'element_any']

    >>> ancestors("P_phosphine")
    ['P_donor', 'element_any']

    >>> ancestors("C_eta5")
    ['C_haptic_any', 'C_donor', 'element_any']

    >>> ancestors("Se_donor")
    ['element_any']
    """
    if node not in _PARENT and node != ELEMENT_ANY:
        raise KeyError(f"Unknown donor node: {node!r}")
    chain: List[str] = []
    current = node
    while current != ELEMENT_ANY:
        current = _PARENT[current]
        chain.append(current)
    return chain


def full_path(node: str) -> List[str]:
    """
    Return the full coarsening path from *node* to ELEMENT_ANY, inclusive.

    Examples
    --------
    >>> full_path("N_pyridine")
    ['N_pyridine', 'N_aromatic_any', 'N_donor', 'element_any']
    """
    return [node] + ancestors(node)


def is_known(node: str) -> bool:
    """Return True if *node* is a registered hierarchy node."""
    return node == ELEMENT_ANY or node in _PARENT


# ---------------------------------------------------------------------------
# Complex & atom donor classification
# ---------------------------------------------------------------------------
from jacobik.core import profile as _profile
from jacobik.core.geometry import dihedral_deg

from jacobik.topology.donor_rules import (  # noqa: F401  (moved; re-exported so existing imports keep working)
    _CARBENE_X_DEGREE,
    _COMPILED,
    _FILTER_TREE,
    _RESONANCE_LABEL,
    _build_ligand_rdkit_mol,
    _classify_carbon_structural,
    _classify_nitrogen_structural,
    _classify_oxygen_structural,
    _get_compiled,
    _resonance_donor_labels,
    bo_is_aromatic,
)

# Profile fields are read at import time, matching the other experiment flags.
HAPTO_BOND_GATE = _profile.get("hapto_bond_gate")
HAPTO_LABEL_CANON = _profile.get("hapto_label_canon")
HAPTO_CHELATE_EXCLUSION = _profile.get("hapto_chelate_exclusion")
DONOR_ASSIGNMENT = _profile.get("donor_assignment")
#: minimum metal BO any member of an eta>=3 face must carry (the quantum bond order is zero for a non-contact)
HAPTO_FACE_BO_FLOOR = 0.05

# ---------------------------------------------------------------------------
# Haptic detection
# ---------------------------------------------------------------------------


def _ligand_heavy_graph(mol: TMComplex, lig_set: Set[int]) -> nx.Graph:
    """Heavy-atom (no metal, no H) ligand bond graph, edges for BO >= 0.5."""
    atom_map = {a.idx: a for a in mol.atoms}
    G: nx.Graph = nx.Graph()
    for idx in lig_set:
        if atom_map[idx].symbol == "H":
            continue
        G.add_node(idx)
    for b in mol.bonds:
        if b.i == mol.metal_idx or b.j == mol.metal_idx or b.order < 0.5:
            continue
        if b.i not in lig_set or b.j not in lig_set:
            continue
        if atom_map[b.i].symbol == "H" or atom_map[b.j].symbol == "H":
            continue
        G.add_edge(b.i, b.j)
    return G


# ---------------------------------------------------------------------------
# Haptic detection
# ---------------------------------------------------------------------------

from jacobik.topology.haptic import (  # noqa: E402, F401  (moved; re-exported so existing imports keep working)
    HAPTIC_BOUND_RATIO,
    HAPTIC_FOLD_DEG,
    _ETA2_RATIO_MAX,
    _FACE_MAX_OFFNORMAL_DEG,
    _FACE_MIN_MEAN_BO,
    _FACE_PI_BO,
    _detect_haptic_groups_v2,
    _eta2_chelate_donor_neighbors,
    _eta2_face_bond_ratios,
    _eta2_face_bonded,
    _recompute_haptic_interaction_geometry,
    _supported_rejected_face_donors,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify(
    mol: TMComplex,
    metal_bonds: Optional[List[Bond]] = None,
    smiles_path: bool = False,
) -> None:
    """
    Classify all donor atoms in a TMComplex and build its coordination sites.

    Parameters
    ----------
    metal_bonds : raw metal-touching bonds (with raw Wiberg orders), supplied
        by mapping.assemble().  Used only to seed haptic detection and compute
        mean metal-donor bond orders; never stored on the complex.  If None,
        falls back to bonds in mol.bonds that touch the metal.
    smiles_path : whether mapping.assemble() succeeded on the SMILES path.
        When True, _detect_haptic_groups_v2 restricts haptic seeds to SMILES-
        flagged donors only, preventing raw BO contacts from introducing
        non-donor atoms (e.g. sulfonate S) into haptic arcs.

    Called by mapping.assemble() after bond quantization and ligand extraction.
    """
    try:
        from rdkit import Chem  # noqa: F401 – confirm available
    except ImportError:
        log.error("RDKit not available; donor classification skipped.")
        return

    from jacobik.core.types import CoordinationSite

    if metal_bonds is None:
        metal_bonds = [
            b for b in mol.bonds
            if b.i == mol.metal_idx or b.j == mol.metal_idx
        ]

    compiled = _get_compiled()
    donor_labels: Dict[int, DonorLabel] = {}
    haptic_interactions: List[HapticInteraction] = []

    atom_map = {a.idx: a for a in mol.atoms}

    # Build a global heavy-atom neighbor map from the bond graph.
    # Used by _classify_donor_atom to count non-H neighbors for the
    # O_water pre-check (RDKit GetDegree() includes H bonds).
    heavy_neighbor_map: Dict[int, Set[int]] = {a.idx: set() for a in mol.atoms}
    for b in mol.bonds:
        if b.order < 0.5:
            continue
        ai = atom_map.get(b.i)
        aj = atom_map.get(b.j)
        if ai is None or aj is None:
            continue
        if aj.symbol != "H":
            heavy_neighbor_map[b.i].add(b.j)
        if ai.symbol != "H":
            heavy_neighbor_map[b.j].add(b.i)

    # Quantized bond graph for structural O classification and ring/aromatic detection.
    sym_of: Dict[int, str] = {a.idx: a.symbol for a in mol.atoms}
    adj_map: Dict[int, List[Tuple[int, float]]] = {a.idx: [] for a in mol.atoms}
    for b in mol.bonds:
        if b.order < 0.5:
            continue
        if b.i == mol.metal_idx or b.j == mol.metal_idx:
            continue
        adj_map[b.i].append((b.j, b.order))
        adj_map[b.j].append((b.i, b.order))

    _Gheavy = nx.Graph()
    for a in mol.atoms:
        if a.idx == mol.metal_idx or a.symbol == "H":
            continue
        _Gheavy.add_node(a.idx)
    for b in mol.bonds:
        if b.order < 0.5 or b.i == mol.metal_idx or b.j == mol.metal_idx:
            continue
        if sym_of.get(b.i) == "H" or sym_of.get(b.j) == "H":
            continue
        _Gheavy.add_edge(b.i, b.j, order=b.order)

    ring_atoms: Set[int] = set()
    aromatic_atoms: Set[int] = set()
    try:
        for cycle in nx.minimum_cycle_basis(_Gheavy):
            ring_atoms.update(cycle)
            cyc = set(cycle)
            ring_bonds = [
                (u, v) for u, v in _Gheavy.edges(cycle)
                if u in cyc and v in cyc
            ]
            if ring_bonds and all(
                bo_is_aromatic(_Gheavy[u][v]["order"]) for u, v in ring_bonds
            ):
                aromatic_atoms.update(cycle)
    except Exception as exc:
        log.debug("ring/aromatic detection failed: %s", exc)

    for ligand in mol.ligands:
        if not ligand.donor_atoms and not ligand.haptic_groups:
            continue

        # Build RDKit mol for this ligand (for SMARTS matching)
        try:
            lig_mol, xyz_to_rdk, _ = _build_ligand_rdkit_mol(
                mol.atoms, mol.bonds, ligand.atom_indices
            )
        except Exception as exc:
            log.warning(
                "%s ligand %d: could not build RDKit mol: %s",
                mol.csd_code, ligand.ligand_id, exc,
            )
            lig_mol = None
            xyz_to_rdk = {}

        # Authoritative donors/faces are kept; only face geometry is refreshed.
        if ligand.donors_are_authoritative:
            groups = []
            for h in ligand.haptic_groups:
                refreshed = _recompute_haptic_interaction_geometry(
                    mol, h.donor_indices, metal_bonds
                )
                if refreshed is not None:
                    groups.append(refreshed)
        else:
            # --- Haptic detection: returns one entry per haptic face ---
            groups = _detect_haptic_groups_v2(mol, ligand, metal_bonds, smiles_path=smiles_path)
        ligand.haptic_groups = groups
        haptic_interactions.extend(groups)

        # Build the flat set of all haptic donor indices on this ligand,
        # and a map from donor_idx -> haptic_order for subclass assignment.
        haptic_donor_order: Dict[int, int] = {}
        for h in groups:
            for d in h.donor_indices:
                haptic_donor_order[d] = h.haptic_order

        dropped = set(getattr(ligand, "dropped_donors", ()) or ())
        if dropped:
            ligand.donor_atoms = [x for x in ligand.donor_atoms if x not in dropped]

        # Haptic donors are always part of donor_atoms.
        if haptic_donor_order:
            existing = set(ligand.donor_atoms)
            merged = sorted(existing | set(haptic_donor_order))
            ligand.donor_atoms = merged

        resonance = _resonance_donor_labels(list(ligand.donor_atoms), sym_of, adj_map, aromatic_atoms)

        # --- Classify each donor atom ---
        for donor_idx in ligand.donor_atoms:
            atom = atom_map.get(donor_idx)
            if atom is None:
                continue

            oxygen_subclass = None
            if atom.symbol == "O":
                oxygen_subclass = _classify_oxygen_structural(
                    donor_idx, sym_of, adj_map, aromatic_atoms, ring_atoms
                )

            carbon_subclass = None
            if atom.symbol == "C":
                carbon_subclass = _classify_carbon_structural(donor_idx, sym_of, adj_map, aromatic_atoms)

            nitrogen_subclass = None
            if atom.symbol == "N":
                nitrogen_subclass = _classify_nitrogen_structural(
                    donor_idx, sym_of, adj_map, aromatic_atoms
                )

            label = _classify_donor_atom(
                donor_idx,
                atom,
                lig_mol,
                xyz_to_rdk,
                compiled,
                heavy_neighbors=heavy_neighbor_map.get(donor_idx),
                oxygen_subclass=oxygen_subclass,
                nitrogen_subclass=nitrogen_subclass,
                carbon_subclass=carbon_subclass,
            )

            if donor_idx in resonance:
                label.subclass = resonance[donor_idx]

            # Mark haptic donors and assign specific C subclass by eta-order.
            # Subclass hierarchy: C_allyl (η3) / C_eta4 / C_eta5 / C_eta6 / C_eta7 / C_eta8,
            # with C_haptic for η2 and any other order. (η7 C7H7 / η8 COT used to fall into C_haptic and
            # polluted the η2 Q1 table with 1.5-1.8 A centroid distances; 2026-10-01.)
            if donor_idx in haptic_donor_order:
                label.is_haptic = True
                if atom.symbol == "C":
                    order = haptic_donor_order[donor_idx]
                    label.subclass = {
                        2: "C_haptic",   # η2 (rare, e.g. alkene σ-complex)
                        3: "C_allyl",
                        4: "C_eta4",
                        5: "C_eta5",
                        6: "C_eta6",
                        7: "C_eta7",
                        8: "C_eta8",
                    }.get(order, "C_haptic")

            donor_labels[donor_idx] = label

    mol.donor_labels = donor_labels

    # Coordination sites: one per haptic face, one per remaining donor.
    haptic_face_atoms: Set[int] = set()
    coord_sites: List[CoordinationSite] = []

    for h in haptic_interactions:
        haptic_face_atoms.update(h.donor_indices)
        # Label from the first face atom's donor label (e.g. "η⁵").
        first = h.donor_indices[0]
        lbl = donor_labels.get(first)
        label = lbl.short() if lbl is not None else f"η{h.haptic_order}"
        site = CoordinationSite(
            kind="haptic",
            donor_indices=list(h.donor_indices),
            position=np.asarray(h.centroid, dtype=float),
            label=label,
            avg_bond_order=h.avg_bond_order,
            haptic_order=h.haptic_order,
        )
        if HAPTO_LABEL_CANON and h.haptic_order == 2:

            canonical = _canonical_eta2_subclass(site, mol)
            if canonical is not None:
                site.label = canonical
        coord_sites.append(site)

    # Mean raw metal-donor BO for kappa sites.
    metal_idx = mol.metal_idx
    metal_bo_lookup = {
        (min(b.i, b.j), max(b.i, b.j)): b.order for b in metal_bonds
    }
    for donor_idx, lbl in donor_labels.items():
        if donor_idx in haptic_face_atoms:
            continue
        bo = metal_bo_lookup.get(
            (min(metal_idx, donor_idx), max(metal_idx, donor_idx)), 0.0
        )
        coord_sites.append(CoordinationSite(
            kind="kappa",
            donor_indices=[donor_idx],
            position=np.asarray(mol.get_atom(donor_idx).position, dtype=float),
            label=lbl.short(),
            avg_bond_order=bo,
            haptic_order=0,
        ))

    mol.coord_sites = coord_sites

    log.debug(
        "%s: classified %d donors, %d coord sites (%d haptic).",
        mol.csd_code, len(donor_labels), len(coord_sites),
        len(haptic_interactions),
    )


def _classify_donor_atom(
    donor_xyz_idx: int,
    atom: Atom,
    lig_mol,
    xyz_to_rdk: Dict[int, int],
    compiled: Dict[str, List[Tuple[str, object]]],
    heavy_neighbors: Optional[Set[int]] = None,
    oxygen_subclass: Optional[str] = None,
    nitrogen_subclass: Optional[str] = None,
    carbon_subclass: Optional[str] = None,
) -> DonorLabel:
    """
    Classify one donor atom using the filter tree.

    Parameters
    ----------
    heavy_neighbors : set of xyz indices of the donor's heavy-atom (non-H)
        neighbors within the ligand, derived from the xyz bond graph.
        Used as a final O_water safety net.
    oxygen_subclass : pre-computed structural O subclass from
        _classify_oxygen_structural (or None).  When provided and not the
        generic "O_donor", it is used directly and SMARTS is skipped for
        oxygen — the structural classifier is more reliable than degree
        SMARTS on explicit-H ligand mols.

    Hydride H and structurally classified O are resolved before SMARTS.
    """
    sym = atom.symbol

    # --- Hydride: always H_hydride regardless of environment ---
    if sym == "H":
        return DonorLabel(
            xyz_idx=donor_xyz_idx,
            family="H_donor",
            subclass="H_hydride",
        )

    family = _ELEMENT_TO_FAMILY.get(sym, "Unknown_donor")

    # --- Oxygen: prefer the structural classification ---
    if sym == "O":
        if oxygen_subclass is not None and oxygen_subclass != "O_donor":
            return DonorLabel(
                xyz_idx=donor_xyz_idx,
                family="O_donor",
                subclass=oxygen_subclass,
            )
        # Safety net: zero heavy neighbours is water even if the structural
        # pass was not supplied.
        n_heavy = len(heavy_neighbors) if heavy_neighbors is not None else None
        if n_heavy is not None and n_heavy == 0:
            return DonorLabel(
                xyz_idx=donor_xyz_idx,
                family="O_donor",
                subclass="O_water",
            )

    if sym == "C" and carbon_subclass is not None:
        return DonorLabel(xyz_idx=donor_xyz_idx, family="C_donor", subclass=carbon_subclass)

    if sym == "N" and nitrogen_subclass is not None:
        return DonorLabel(
            xyz_idx=donor_xyz_idx, family="N_donor", subclass=nitrogen_subclass,
        )

    patterns = compiled.get(family, [])

    rdk_idx = xyz_to_rdk.get(donor_xyz_idx)
    if rdk_idx is None or lig_mol is None:
        return DonorLabel(xyz_idx=donor_xyz_idx, family=family, subclass=family)

    for subclass, pat in patterns:
        try:
            matches = lig_mol.GetSubstructMatches(pat)
            if any(m[0] == rdk_idx for m in matches):
                return DonorLabel(
                    xyz_idx=donor_xyz_idx,
                    family=family,
                    subclass=subclass,
                )
        except Exception as exc:
            log.debug("SMARTS match error for %s: %s", subclass, exc)
            continue

    # No specific match: fall back to family label
    return DonorLabel(xyz_idx=donor_xyz_idx, family=family, subclass=family)


def _canonical_eta2_subclass(site: CoordinationSite, mol: TMComplex) -> Optional[str]:
    """Canonical subclass for an eta2 face, or None when the rule cannot label it."""
    if not (site.is_haptic and site.haptic_order == 2):
        return None
    symbols = {a.idx: a.symbol for a in mol.atoms}
    face = list(site.donor_indices)
    if any(symbols.get(i) == "C" for i in face):
        return "C_haptic"
    labels = {i: mol.donor_labels[i].subclass for i in face if i in mol.donor_labels}
    if labels:
        from rdkit import Chem

        periodic = Chem.GetPeriodicTable()
        highest_z = max(periodic.GetAtomicNumber(symbols[i]) for i in labels)
        candidates = [i for i in labels if periodic.GetAtomicNumber(symbols[i]) == highest_z]
        # Lowest subclass string resolves same-element ties independent of face order.
        return min(labels[i] for i in candidates)
    return None
