"""
donor_classifier.py
===================
Classifies donor atoms in a TMComplex and detects haptic interactions.

Version
-------
1.0.0 : initial release
1.1.0 : Fixed O_carboxylate misclassification for delocalized/Kekule-ambiguous
         carboxylate groups (EDTA, acetate, oxalate, etc.).
         Root cause: _classify_oxygen_structural required bo < 1.5 on the
         donor O–C bond to trigger the carboxylate branch.  CSD stores
         carboxylate C–O bonds as either delocalized (~1.5/~1.5) or in
         either Kekule form (1.0/2.0 or 2.0/1.0).  When the donor O carried
         the double-bond assignment (bo ≥ 1.5), the carboxylate check failed
         and the O fell through to O_carbonyl — an entirely different stats
         distribution (1.95 Å vs 2.35 Å peak).  For polydentate ligands like
         EDTA with 4 carboxylate groups, Kekule assignments vary per group,
         producing inconsistent subclasses (some O_carboxylate, some O_carbonyl)
         for chemically equivalent oxygens, causing the scorer to target
         wildly different bond lengths for the same donor type.
         Additional complication: the engine uses RDKit.Chem.AddHs() which
         adds an implicit H to the terminal O of a neutral carboxylic acid form
         (OC(=O) SMILES → the terminal O gets a phantom H).  The initial fix
         excluded O's with H (v3), which made the fix ineffective for the engine
         path.  The final rule (v5) instead requires: (a) C has at least one
         O bond with order >= 1.5 (double-bond character, excludes lactol where
         all C-O are single), AND (b) C has a terminal O with at most 1 heavy
         neighbour (terminal, excludes acetal where the other O is bridging).
         This correctly handles COO^- anion, COOH acid with explicit H, and
         all Kekule/delocalized forms without bond-order restrictions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from jacobik.core.types import (
    Atom, Bond, DonorLabel, HapticInteraction, Ligand, TMComplex,
)
from jacobik.topology.donor_hierarchy import _ELEMENT_TO_FAMILY

log = logging.getLogger(__name__)

# See exp.md for detailed rationale

_FILTER_TREE: Dict[str, List[Tuple[str, str]]] = {

    # See exp.md for detailed rationale
    "C_donor": [
        # CO ligand: terminal C=O (D1 or D2 for bridging CO)
        ("C_carbonyl",   "[C;D1,D2]=[O;D1]"),
        # Isocyanide-type: terminal C bonded to N (M-C≡N-R or M-C=N-R)
        ("C_isocyanide", "[C;D1]~[#7]"),
        # NHC carbene: C flanked by two N atoms (imidazolylidene C2 position)
        ("C_carbene",    "[C;D2](-[#7])-[#7]"),
        # Sigma-alkyl: sp3 or sp2 C, not carbonyl/isocyanide/aromatic
        ("C_alkyl",      "[C;D1,D2,D3,D4;!$([C]=[O,S,N]);!$([C]#[N,O]);!a]"),
        # Aromatic ring C (kappa-C_pi, e.g. terminal aryl sigma-donor)
        ("C_pi",         "[c]"),
    ],

    # -----------------------------------------------------------------------
    # Nitrogen donors
    # -----------------------------------------------------------------------
    "N_donor": [
        # Dinitrogen: N≡N (very specific, must come first)
        ("N_dinitrogen",  "[N;D1]#[N;D1]"),
        # Cyanide/nitrile coordinating via N: N≡C
        ("N_cyanide",     "[N;D1]#[C]"),
        # Nitrosyl: N=O  (linear or bent)
        ("N_nitrosyl",    "[N;D1,D2]=[O;D1]"),
        # NHC nitrogen: aromatic, in 5-membered ring, degree 3
        ("N_NHC",         "[n;D3;r5]"),
        # Pyridine-type aromatic N: 6-membered aromatic ring, degree 2.
        # Restricted to r6 so it no longer swallows 5-ring azole nitrogens.
        ("N_pyridine",    "[n;D2;r6]"),
        # Azole aromatic N: 5-membered aromatic ring, degree 2
        # (imidazole =N-, pyrazole, oxazole/thiazole N, triazole, etc.)
        ("N_azole",       "[n;D2;r5]"),
        # Pyridine-type non-aromatic 6-ring fallback — when sanitization
        # leaves a 6-ring N as non-aromatic (organometallic fragments).
        ("N_pyridine",    "[N;D2;r6](-[#6])=[#6]"),
        # General aromatic-N fallback ("Nar"): any other aromatic nitrogen
        # not caught by the specific ring-size patterns above.  This is the
        # catch-all for aromatic N so that unusual ring sizes / fusions get
        # a sensible aromatic label instead of falling through to N_amine.
        ("N_aromatic_any", "[n]"),
        # Imine: sp2, non-aromatic, not in ring
        ("N_imine",       "[N;D2;!a;!r]=[#6]"),
        # Amide: N bonded to a carbonyl C
        ("N_amide",       "[N;D1,D2,D3;!a]-[C]=[O]"),
        # Amine: sp3 N, not in ring, no C=O or O/S/P neighbours
        ("N_amine",       "[N;D1,D2,D3;!a;!r;!$(N-C=[O,S]);!$(N~[O,S,P])]"),
    ],

    # See exp.md for detailed rationale
    "O_donor": [
        # Sulfonate: coordinating O on -SO3R / SO3^-.
        # Use recursive SMARTS to match the sulfonyl fragment robustly.
        # S degree D3 or D4 (D3 when one O is the donor and not counted).
        ("O_sulfonate",   "[O;D1,D2;$([O]-[S;D3,D4](=[O])=[O])]"),
        # Nitro: O-N(=O)
        ("O_nitro",       "[O;D1,D2]-[N;D3](=[O])"),
        # Carboxylate: O bonded to a C that also has a bare terminal O neighbor
        # (D1, no H).  Covers delocalized COO~ and both Kekule forms.
        # Excludes: acetal (other O is D2, bridging), lactol/hemiacetal (other O
        # has H), ketone/amide (no other O on C at all).
        # Matches the logic of _classify_oxygen_structural (v1.1.0).
        # This SMARTS is a rarely-reached fallback; the structural classifier
        # handles >99% of O donors before SMARTS is reached.
        ("O_carboxylate", "[O;D1,D2;$([O]~[C;!$(C-[OH])]~[O;D1;!$([OH])])]"),
        # Phenoxide: O directly bonded to an aromatic C
        ("O_phenoxide",   "[O;D1,D2]-[c]"),
        # Cyclic ether (THF, furan-type): O in ring, not part of C=O
        ("O_ether_cyclic","[O;D2;r;!$(O~[C,S,P,N]=O)]"),
        # Carbonyl O (ketone, aldehyde, ester, urea): =O on C.
        # MUST come before O_hydroxo — RDKit may perceive a keto O as
        # carrying an implicit H under partial sanitization.
        ("O_carbonyl",    "[O;D1]=[C]"),
        # Hydroxo: exactly one heavy neighbor, NOT a C=O context.
        # The recursive exclusion guards against keto tautomers.
        ("O_hydroxo",     "[O;D1;!$([O]=[C]);!$([O]-[C](=[O]))]"),
        # Alkoxide: O-C, not phenoxide, not carboxylate/sulfonate-type
        ("O_alkoxide",    "[O;D1,D2;!$([O]-c);!$([O]~[C,S,P,N]=O)]"),
        # Ether: acyclic O(-C)(-C)
        ("O_ether",       "[O;D2;!r;!$([O]~[C,S,P,N]=O);!$([O]-c)]"),
    ],

    # See exp.md for detailed rationale
    "P_donor": [
        # See exp.md for detailed rationale
        ("P_phosphite",   "[P;D2,D3,D4](-[O])-[O]"),
        # Phosphinite: P with at least one O and at least one C neighbor
        ("P_phosphinite", "[P;D2,D3,D4](-[O])-[#6]"),
        # Phosphine: P with only C neighbors (no O)
        ("P_phosphine",   "[P;D2,D3,D4;!$([P]-[O])](-[#6])-[#6]"),
    ],

    # -----------------------------------------------------------------------
    # Sulfur donors
    # -----------------------------------------------------------------------
    "S_donor": [
        # Sulfoxide: S=O
        ("S_sulfoxide",    "[S;D3](=[O])"),
        # Thiocarbonyl: S=C
        ("S_thiocarbonyl", "[S;D1,D2]=[#6]"),
        # Thioether: S with 2 C neighbors
        ("S_thioether",    "[S;D2](-[#6])-[#6]"),
        # Thiolate: S with 1 C neighbor (or anionic)
        ("S_thiolate",     "[S;D1,D2;!$([S](-[#6])-[#6])]-[#6]"),
    ],

    # -----------------------------------------------------------------------
    # Halide donors (always terminal when donating to metal)
    # -----------------------------------------------------------------------
    "Hal_donor": [
        ("Hal_F",  "[F;D1,D0]"),
        ("Hal_Cl", "[Cl;D1,D0]"),
        ("Hal_Br", "[Br;D1,D0]"),
        ("Hal_I",  "[I;D1,D0]"),
    ],

    # -----------------------------------------------------------------------
    # Boron donors
    # -----------------------------------------------------------------------
    "B_donor": [
        ("B_boronate", "[B;D2,D3,D4]"),
    ],

    # -----------------------------------------------------------------------
    # Chalcogenide donors (Se, Te, As) – element only, no subclasses yet
    # -----------------------------------------------------------------------
    "Se_donor": [],
    "Te_donor": [],
    "As_donor": [],

    # -----------------------------------------------------------------------
    # Hydrogen donors (hydrides) – detected by element, not SMARTS
    # -----------------------------------------------------------------------
    "H_donor": [],
}

# Pre-compile SMARTS patterns once at import time
_COMPILED: Dict[str, List[Tuple[str, object]]] = {}

def _get_compiled() -> Dict[str, List[Tuple[str, object]]]:
    """Return the compiled SMARTS cache, building it on first call."""
    if _COMPILED:
        return _COMPILED
    try:
        from rdkit import Chem
    except ImportError:
        return _COMPILED  # empty; caller must handle

    for family, patterns in _FILTER_TREE.items():
        compiled_list = []
        for subclass, smarts in patterns:
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                log.warning("Invalid SMARTS for %s: %r", subclass, smarts)
                continue
            compiled_list.append((subclass, pat))
        _COMPILED[family] = compiled_list

    return _COMPILED


# ---------------------------------------------------------------------------
# Ligand RDKit mol builder (heavy atoms only, quantized bond types)
# ---------------------------------------------------------------------------

def _build_ligand_rdkit_mol(
    atoms: List[Atom],
    bonds: List[Bond],
    ligand_atom_indices: List[int],
):
    """
    Build an RDKit RWMol from the quantized bonds of one ligand.

    Returns
    -------
    (mol, xyz_to_rdk, rdk_to_xyz)
    mol        : sanitized RDKit mol (sanitization errors caught)
    xyz_to_rdk : dict xyz_idx -> rdkit_idx
    rdk_to_xyz : dict rdkit_idx -> xyz_idx
    """
    from rdkit import Chem
    from rdkit.Chem import RWMol, rdchem
    from rdkit.Chem import Atom as RDKAtom

    lig_set = set(ligand_atom_indices)
    atom_map = {a.idx: a for a in atoms}
    xyz_to_rdk: Dict[int, int] = {}

    rw = RWMol()
    for xyz_idx in sorted(lig_set):
        a = atom_map[xyz_idx]
        rdk_idx = rw.AddAtom(RDKAtom(a.symbol))
        xyz_to_rdk[xyz_idx] = rdk_idx

    rdk_to_xyz: Dict[int, int] = {v: k for k, v in xyz_to_rdk.items()}

    for b in bonds:
        if b.i not in lig_set or b.j not in lig_set:
            continue
        if b.order < 0.5:
            continue
        ri = xyz_to_rdk[b.i]
        rj = xyz_to_rdk[b.j]
        if b.order >= 2.5:
            bt = rdchem.BondType.TRIPLE
        elif bo_is_aromatic(b.order):
            bt = rdchem.BondType.AROMATIC
        elif b.order >= 1.5:
            bt = rdchem.BondType.DOUBLE
        else:
            bt = rdchem.BondType.SINGLE
        try:
            rw.AddBond(ri, rj, bt)
        except Exception:
            pass  # duplicate bond edge, skip

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            from rdkit.Chem import SanitizeFlags
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                catchErrors=True,
            )
        except Exception:
            pass  # best-effort; aromatic SMARTS may miss but degree-based will still work

    return mol, xyz_to_rdk, rdk_to_xyz


def bo_is_aromatic(order: float) -> bool:
    """Return True for bond orders that represent aromatic/delocalized bonds."""
    return 1.4 <= order <= 1.6


# ---------------------------------------------------------------------------
# Structural oxygen classification (bond-graph based, no SMARTS)
# ---------------------------------------------------------------------------

def _classify_oxygen_structural(
    o_idx: int,
    sym_of: Dict[int, str],
    adj: Dict[int, List[Tuple[int, float]]],
    aromatic_atoms: Set[int],
    ring_atoms: Set[int],
) -> str:
    """
    Classify a donor oxygen from the quantized xyz bond graph alone.
    """
    o_nbrs = adj.get(o_idx, [])
    heavy = [(n, bo) for n, bo in o_nbrs if sym_of.get(n) != "H"]
    has_H = any(sym_of.get(n) == "H" for n, _ in o_nbrs)

    if len(heavy) == 0:
        return "O_water"

    o_has_double = any(bo >= 1.5 for _, bo in heavy)

    def _has_double_to_O(a_idx: int, exclude: int) -> bool:
        return any(
            sym_of.get(n) == "O" and bo >= 1.5
            for n, bo in adj.get(a_idx, [])
            if n != exclude
        )

    def _o_count(a_idx: int) -> int:
        return sum(1 for n, _ in adj.get(a_idx, []) if sym_of.get(n) == "O")

    # Sulfonate / nitro by heteroatom neighbour
    for n, _ in heavy:
        s = sym_of.get(n)
        if s == "S" and _o_count(n) >= 2:
            return "O_sulfonate"
        if s == "N" and _has_double_to_O(n, o_idx):
            return "O_nitro"

    # Carboxylate: O bonded to a C that also has any other O neighbor.
    # Bond-order-agnostic: covers delocalized (~1.5/~1.5), Kekule form A
    # (donor O=C, terminal O–C), and Kekule form B (donor O–C, terminal O=C).
    # Carboxylate: O bonded to a C that satisfies BOTH conditions:
    #   (a) C has at least one O neighbour with bond order >= 1.5
    #       (double-bond character — rules out lactol where all C-O bonds
    #       are single, and acetal where C-O bonds are also all single)
    #   (b) C has at least one other O with at most 1 heavy neighbour (terminal)
    #       (rules out acetal where the "other O" is bridging, degree 2)
    #
    # Together these two conditions uniquely identify COO in all representations:
    #   delocalized ~1.5/~1.5 : condition (a) from the ~1.5 bond, condition (b)
    #                           from either terminal O
    #   Kekule A  O=C-O       : condition (a) from the donor O=2.0 bond,
    #                           condition (b) from the terminal O-1.0
    #   Kekule B  O-C=O       : condition (a) from the terminal O=2.0 bond,
    #                           condition (b) from that same terminal O
    #   COOH (RDKit AddHs)    : condition (a) from C=O(2.0), condition (b) from
    #                           the -OH oxygen which has 1 heavy nbr (C) even
    #                           though it has H — the "at most 1 heavy" threshold
    #                           correctly admits it.
    #
    # Correctly EXCLUDED:
    #   acetal  O-C(-O-C)  : other O is bridging (2 heavy nbrs)
    #   lactol  O-C(-OH)   : C has no O with bond order >= 1.5 (all single)
    #   ketone  O=C(C)(C)  : C has no other O at all
    #   amide   O=C(N)(C)  : C has no other O (only N)
    #
    # See v1.1.0 changelog for the full failure-mode analysis.
    for n, bo in heavy:
        if sym_of.get(n) != "C":
            continue
        # Condition (a): C has at least one O bond with double-bond character
        c_has_double_O = any(
            sym_of.get(nb) == "O" and nb_bo >= 1.5
            for nb, nb_bo in adj.get(n, [])
        )
        if not c_has_double_O:
            continue
        # Condition (b): C also has a terminal O (≤ 1 heavy neighbour)
        for nb, _ in adj.get(n, []):
            if nb == o_idx or sym_of.get(nb) != "O":
                continue
            nb_heavy = [x for x, _ in adj.get(nb, []) if sym_of.get(x) != "H"]
            if len(nb_heavy) == 1:
                return "O_carboxylate"

    # Carbonyl: this O double-bonds a C directly
    if o_has_double and any(sym_of.get(n) == "C" for n, _ in heavy):
        return "O_carbonyl"

    # Phenoxide: single bond to an aromatic C, no proton
    for n, bo in heavy:
        if sym_of.get(n) == "C" and n in aromatic_atoms and bo < 1.5 and not has_H:
            return "O_phenoxide"

    # Hydroxo: an O-H with one heavy C neighbour
    if has_H and len(heavy) == 1 and sym_of.get(heavy[0][0]) == "C":
        return "O_hydroxo"

    n_C = sum(1 for n, _ in heavy if sym_of.get(n) == "C")
    if o_idx in ring_atoms and n_C >= 1:
        return "O_ether_cyclic"
    if n_C >= 2:
        return "O_ether"
    if n_C == 1 and not has_H:
        return "O_alkoxide"

    return "O_donor"


# ---------------------------------------------------------------------------
# Single-atom classification
# ---------------------------------------------------------------------------

def _classify_donor_atom(
    donor_xyz_idx: int,
    atom: Atom,
    lig_mol,
    xyz_to_rdk: Dict[int, int],
    compiled: Dict[str, List[Tuple[str, object]]],
    heavy_neighbors: Optional[Set[int]] = None,
    oxygen_subclass: Optional[str] = None,
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

    Structural pre-checks (applied before SMARTS)
    ----------------------------------------------
    Oxygen: classified structurally from the bond graph (see
    _classify_oxygen_structural) rather than by SMARTS degree patterns,
    which mis-fire on explicit-H / partially-sanitized ligand mols.
    
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


# ---------------------------------------------------------------------------
# Haptic detection
# ---------------------------------------------------------------------------

# See exp.md for detailed rationale
_FACE_DIST_RATIO: float = 1.40

# Minimum number of near-equidistant ring members required to call a ring a
# haptic face.  Three is the smallest chemically meaningful eta arc (eta-3
# within a ring); open-chain allyl (eta-3) is handled separately below.
_FACE_MIN_MEMBERS: int = 3

# Open-chain (acyclic) haptic, e.g. allyl eta-3 / butadiene eta-4: a path of
# this many or more contiguous heavy atoms that are all near-equidistant to
# the metal and carry at least one seed donor.
_OPEN_FACE_MIN: int = 3

# See exp.md for detailed rationale
_FACE_MAX_OFFNORMAL_DEG: float = 50.0

# See exp.md for detailed rationale
_FACE_MIN_MEAN_BO: float = 1.10
_FACE_PI_BO: float = 1.40


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


def _detect_haptic_groups(
    mol: TMComplex,
    ligand: Ligand,
    metal_bonds: List[Bond],
    smiles_path: bool = False,
) -> List[HapticInteraction]:
    """
    Detect ALL haptic interactions for one ligand, geometry-first.
    """
    lig_set = set(ligand.atom_indices)
    donor_set = set(ligand.donor_atoms)
    atom_map = {a.idx: a for a in mol.atoms}
    metal_idx = mol.metal_idx
    metal_pos = atom_map[metal_idx].position

    # See exp.md for detailed rationale
    metal_bonded: Set[int] = set()
    if not smiles_path:
        for b in metal_bonds:
            partner = b.j if b.i == metal_idx else b.i
            if partner in lig_set:
                metal_bonded.add(partner)
    seeds = donor_set | metal_bonded
    if not seeds:
        return []

    G = _ligand_heavy_graph(mol, lig_set)

    metal_bo = {
        (min(b.i, b.j), max(b.i, b.j)): b.order
        for b in metal_bonds
    }

    # Ligand-internal bond orders (for the pi-character gate).
    lig_bo = {
        (min(b.i, b.j), max(b.i, b.j)): b.order
        for b in mol.bonds
        if b.i != metal_idx and b.j != metal_idx
    }

    def _ring_pi_ok(ring_members: List[int]) -> bool:
        """
        True when the ring's internal bonds show pi delocalization: mean ring
        bond order >= _FACE_MIN_MEAN_BO, or at least half are clearly multiple.
        Rejects saturated (all-single) macrocyclic chelate rings.
        """
        cyc = set(ring_members)
        bos: List[float] = []
        for u in ring_members:
            for v in cyc:
                if v <= u:
                    continue
                key = (u, v)
                if G.has_edge(u, v):
                    bos.append(lig_bo.get(key, 1.0))
        if not bos:
            return False
        mean_bo = float(np.mean(bos))
        frac_pi = sum(1 for b in bos if b >= _FACE_PI_BO) / len(bos)
        return mean_bo >= _FACE_MIN_MEAN_BO or frac_pi >= 0.5

    def _metal_dist(idx: int) -> float:
        return float(np.linalg.norm(atom_map[idx].position - metal_pos))

    def _offnormal_deg(face_atoms: List[int]) -> float:
        """
        Angle (deg) between the face-centroid->metal vector and the face
        best-fit-plane normal.  ~0 means the metal sits directly over the
        plane (eta-face); ~90 means the metal lies in the plane (sigma-aryl).
        Returns 0.0 for <3 points (no plane defined; caller gates on count).
        """
        pts = np.stack([atom_map[x].position for x in face_atoms])
        if len(pts) < 3:
            return 0.0
        cen = pts.mean(axis=0)
        # best-fit plane normal = smallest right-singular vector
        try:
            _, _, vt = np.linalg.svd(pts - cen)
        except np.linalg.LinAlgError:
            return 0.0
        normal = vt[2]
        v = metal_pos - cen
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return 0.0
        v = v / nv
        cos_a = abs(float(np.dot(normal, v)))
        cos_a = max(-1.0, min(1.0, cos_a))
        return float(np.degrees(np.arccos(cos_a)))

    def _make_interaction(face_atoms: List[int]) -> Optional[HapticInteraction]:
        heavy = [x for x in face_atoms if atom_map[x].symbol != "H"]
        if len(heavy) < 2:
            return None
        positions = np.stack([atom_map[x].position for x in heavy])
        centroid = positions.mean(axis=0)
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

    results: List[HapticInteraction] = []
    used: Set[int] = set()

    # --- Cyclic faces ---
    try:
        rings = [set(c) for c in nx.minimum_cycle_basis(G) if len(c) >= 3]
    except Exception as exc:
        log.debug("minimum_cycle_basis failed for ligand %d: %s",
                  ligand.ligand_id, exc)
        rings = []

    # See exp.md for detailed rationale
    candidates = []  # (completeness, -offnormal, n_face, ring, face)
    for ring in rings:
        members = sorted(ring)
        dists = {x: _metal_dist(x) for x in members}
        dmin = min(dists.values())
        face = [x for x in members if dists[x] <= _FACE_DIST_RATIO * dmin]
        if len(face) < _FACE_MIN_MEMBERS:
            continue
        if not any(x in seeds for x in face):
            continue
        ang = _offnormal_deg(face)
        if ang > _FACE_MAX_OFFNORMAL_DEG:
            continue
        # Require pi delocalization: reject saturated macrocyclic chelate rings
        # whose all-single backbone can otherwise pass the geometric tests.
        if not _ring_pi_ok(members):
            continue
        completeness = len(face) / len(ring)
        candidates.append((completeness, -ang, len(face), members, face))

    # Sort best-first; tuple ordering gives completeness desc, angle asc,
    # face-size desc.
    candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    for _comp, _negang, _nf, _members, face in candidates:
        if set(face) & used:
            continue
        inter = _make_interaction(face)
        if inter is not None:
            results.append(inter)
            used |= set(face)

    # --- Open-chain faces (allyl / open polyene) ---
    # Consider only seed atoms not already absorbed into a cyclic face.
    open_seeds = [s for s in seeds if s not in used and s in G]
    for s in open_seeds:
        if s in used:
            continue
        d_s = _metal_dist(s)
        # Grow a contiguous arc of near-equidistant heavy neighbours.
        arc = {s}
        frontier = [s]
        while frontier:
            cur = frontier.pop()
            for nb in G.neighbors(cur):
                if nb in arc or nb in used:
                    continue
                if _metal_dist(nb) <= _FACE_DIST_RATIO * d_s:
                    arc.add(nb)
                    frontier.append(nb)
        # Reject if this arc lives inside a ring (already handled) or too small.
        if len(arc) < _OPEN_FACE_MIN:
            continue
        if any(x in r for x in arc for r in rings):
            continue
        # Open-chain haptic (allyl η3, butadiene η4, etc.) is exclusively a
        # carbon-chain phenomenon.  Reject arcs containing any non-carbon heavy
        # atom — this cleanly excludes oxoanion heads (e.g. O-S-O of a
        # sulfonate) whose geometry and pi character otherwise pass all gates.
        if any(atom_map[x].symbol != "C" for x in arc):
            continue
        # Require pi delocalization across the arc (allyl/polyene), rejecting
        # saturated -CH2-CH2- chains that happen to lie near the metal.
        if not _ring_pi_ok(sorted(arc)):
            continue
        inter = _make_interaction(sorted(arc))
        if inter is not None:
            results.append(inter)
            used |= arc

    return results


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
        When True, _detect_haptic_groups restricts haptic seeds to SMILES-
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

    # See exp.md for detailed rationale
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

        # See exp.md for detailed rationale
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
            groups = _detect_haptic_groups(mol, ligand, metal_bonds, smiles_path=smiles_path)
        ligand.haptic_groups = groups
        haptic_interactions.extend(groups)

        # Build the flat set of all haptic donor indices on this ligand,
        # and a map from donor_idx -> haptic_order for subclass assignment.
        haptic_donor_order: Dict[int, int] = {}
        for h in groups:
            for d in h.donor_indices:
                haptic_donor_order[d] = h.haptic_order

        # See exp.md for detailed rationale
        if haptic_donor_order:
            existing = set(ligand.donor_atoms)
            merged = sorted(existing | set(haptic_donor_order))
            ligand.donor_atoms = merged

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

            label = _classify_donor_atom(
                donor_idx,
                atom,
                lig_mol,
                xyz_to_rdk,
                compiled,
                heavy_neighbors=heavy_neighbor_map.get(donor_idx),
                oxygen_subclass=oxygen_subclass,
            )

            # Mark haptic donors and assign specific C subclass by eta-order.
            # Subclass hierarchy: C_allyl (η3) / C_eta4 / C_eta5 / C_eta6,
            # with C_haptic as fallback for any other haptic order.
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
                    }.get(order, "C_haptic")

            donor_labels[donor_idx] = label

    mol.donor_labels = donor_labels

    # See exp.md for detailed rationale
    haptic_face_atoms: Set[int] = set()
    coord_sites: List[CoordinationSite] = []

    for h in haptic_interactions:
        haptic_face_atoms.update(h.donor_indices)
        # Label from the first face atom's donor label (e.g. "η⁵").
        first = h.donor_indices[0]
        lbl = donor_labels.get(first)
        label = lbl.short() if lbl is not None else f"η{h.haptic_order}"
        coord_sites.append(CoordinationSite(
            kind="haptic",
            donor_indices=list(h.donor_indices),
            position=np.asarray(h.centroid, dtype=float),
            label=label,
            avg_bond_order=h.avg_bond_order,
            haptic_order=h.haptic_order,
        ))

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