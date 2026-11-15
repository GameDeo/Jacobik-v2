"""Structural donor classification rules (SMARTS filter tree, O/N/C structural rules, and resonance labels)."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

from jacobik.core.types import Atom, Bond

log = logging.getLogger(__name__)


# SMARTS filter tree: family -> ordered (subclass, SMARTS); first match wins.
_FILTER_TREE: Dict[str, List[Tuple[str, str]]] = {
    # Carbon donors
    "C_donor": [
        # CO ligand: terminal C=O (D1 or D2 for bridging CO)
        ("C_carbonyl",   "[C;D1,D2]=[O;D1]"),
        # Isocyanide-type: terminal C bonded to N (M-C≡N-R or M-C=N-R)
        # Cyanide (M-C≡N: N has no other heavy neighbour) vs isocyanide
        # (M-C≡N-R: N carries a substituent) are different ligands with
        # different M-C distances and must not share statistics.
        ("C_cyanide",    "[C;D1]~[N;D1]"),
        ("C_isocyanide", "[C;D1]~[N;D2,D3]"),
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
        # Fallback only: sp3 N are normally resolved earlier by
        # _classify_nitrogen_structural (which sees the H atoms that the
        # heavy-atom ligand mol used for SMARTS does not).
        ("N_amine",       "[N;D0,D1,D2,D3;!a;!$(N-C=[O,S]);!$(N~[O,S,P])]"),
    ],

    # Oxygen donors
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
        # Mirrors the carboxylate rule of _classify_oxygen_structural.
        # This SMARTS is a rarely-reached fallback; the structural classifier
        # handles >99% of O donors before SMARTS is reached.
        ("O_carboxylate", "[O;D1;$([O]~[C;!$(C-[OH])]~[O;D1;!$([OH])])]"),
        # Phenoxide: O directly bonded to an aromatic C
        ("O_phenoxide",   "[O;D1,D2]-[c]"),
        # Cyclic ether (THF, furan-type): O in ring, not part of C=O
        ("O_ether_cyclic","[O;D2;r;!$(O~[C,S,P,N]=O)]"),
        # Carbonyl O (ketone, aldehyde, ester, urea): =O on C.
        # MUST come before O_heteroxo — RDKit may perceive a keto O as
        # carrying an implicit H under partial sanitization.
        ("O_carbonyl",    "[O;D1]=[C]"),
        # Hydroxo: exactly one heavy neighbor, NOT a C=O context.
        # The recursive exclusion guards against keto tautomers.
        ("O_heteroxo",    "[O;D1;!$([O]=[C]);!$([O]-[C](=[O]))]"),
        # Alkoxide: O-C, not phenoxide, not carboxylate/sulfonate-type
        ("O_alkoxide",    "[O;D1,D2;!$([O]-c);!$([O]~[C,S,P,N]=O)]"),
        # Ether: acyclic O(-C)(-C)
        ("O_ether",       "[O;D2;!r;!$([O]~[C,S,P,N]=O);!$([O]-c)]"),
    ],

    # Phosphorus donors
    "P_donor": [
        # Phosphite: P with at least two O neighbors
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
        # Split by the number of H on the donor: H2O (2), OH- (1), oxo/O2- (0).
        n_H = sum(1 for n, _ in o_nbrs if sym_of.get(n) == "H")
        if n_H >= 2:
            return "O_water"
        if n_H == 1:
            return "O_hydroxide"
        return "O_oxo"

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
        if s == "N" and (_has_double_to_O(n, o_idx) or _o_count(n) >= 2):
            return "O_nitro"  # nitro / nitrite / nitrate O, whichever O the input wrote as N=O

    # Carboxylate (bond-order-agnostic; covers delocalized and both Kekule forms):
    # the donor O must be terminal (one heavy neighbour) and bond to a C that
    # (a) has an O neighbour with bond order >= 1.5 (excludes lactol/acetal) and
    # (b) has another terminal O (<= 1 heavy neighbour; excludes acetal).
    # Ester alkoxy O (two heavy neighbours), ketones and amides do not match.
    if len(heavy) == 1:
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
        if len(heavy) == 1 and sym_of.get(n) == "C" and n in aromatic_atoms and bo < 1.5 and not has_H:
            return "O_phenoxide"  # terminal O only: an aryl ether (Ar-O-R) falls through to O_ether

    # Alcohol: an O-H with one heavy neighbour (C-OH; an sp3 sigma donor,
    # not the M-OH hydroxide, which has no heavy neighbour).
    if has_H and len(heavy) == 1:
        return "O_alcohol"
    # Terminal O on a non-carbon heavy atom without H (P=O, S=O, N-oxide, ...)
    if (not has_H and len(heavy) == 1 and sym_of.get(heavy[0][0]) != "C"):
        return "O_heteroxo"

    n_C = sum(1 for n, _ in heavy if sym_of.get(n) == "C")
    if o_idx in ring_atoms and n_C >= 1:
        return "O_ether_cyclic"
    if n_C >= 2:
        return "O_ether"
    if n_C == 1 and not has_H:
        return "O_alkoxide"

    return "O_donor"


# ---------------------------------------------------------------------------
# Nitrogen, carbon, and resonance structural rules
# ---------------------------------------------------------------------------
def _classify_nitrogen_structural(
    n_idx: int,
    sym_of: Dict[int, str],
    adj: Dict[int, List[Tuple[int, float]]],
    aromatic_atoms: Set[int],
) -> Optional[str]:
    """
    Classify a saturated (sp3-like) N donor from the bond graph *including H*.

    Returns None when the N is aromatic, multiply bonded, or attached to a
    carbonyl/heteroatom (those are resolved by the SMARTS tree), so the
    structural pass only takes over the family the SMARTS tree lumped into
    ``N_amine``.

    Counting the substituents (heavy + H) separates NH3, primary, secondary
    and tertiary amines, and the anionic amido (2 substituents) / imido
    (<=1 substituent) forms, which have very different M-N distances.
    """
    if n_idx in aromatic_atoms:
        return None
    nbrs = adj.get(n_idx, [])
    heavy = [(n, bo) for n, bo in nbrs if sym_of.get(n) != "H"]
    n_H = sum(1 for n, _ in nbrs if sym_of.get(n) == "H")
    if any(bo >= 1.4 for _, bo in heavy):
        return None
    for n, _ in heavy:
        s = sym_of.get(n)
        if s in ("N", "O", "S", "P"):
            return None
        if s == "C":
            # amide/urea/thioamide-type carbon: leave to the SMARTS tree
            for nb, bo in adj.get(n, []):
                if nb != n_idx and sym_of.get(nb) in ("O", "S") and bo >= 1.4:
                    return None
    subs = len(heavy) + n_H
    if subs >= 4:
        return "N_ammonium_like"
    if subs == 3:
        if n_H == 3:
            return "N_ammonia"
        if n_H == 2:
            return "N_primary_amine"
        if n_H == 1:
            return "N_secondary_amine"
        return "N_tertiary_amine"
    if subs == 2:
        return "N_amido"
    return "N_imido"


#: Heteroatom neighbours that make a two-coordinate donor carbon a heteroatom-stabilised carbene: a lone-pair
#: donor (N with three neighbours, O or S with two), whatever bond order the input wrote for the C-X bond.
_CARBENE_X_DEGREE = {"N": 3, "O": 2, "S": 2}


def _classify_carbon_structural(
    c_idx: int,
    sym_of: Dict[int, str],
    adj: Dict[int, List[Tuple[int, float]]],
    aromatic_atoms: Set[int],
) -> Optional[str]:
    """
    Classify a kappa C donor from its neighbours *including explicit H* (metal excluded).

    Topology first, bond orders last: the DB path gets bond orders from quantised crystal Wiberg orders and the solve
    path from the SMILES, and the two disagree for exactly the carbons that matter (C#O vs C=O, a carbene C-N
    written single or double). Neighbour elements, neighbour degrees and explicit H counts are the same on both paths.

      one neighbour  O -> C_carbonyl (terminal CO), S -> C_thiocarbonyl, N -> C_cyanide (N has no other neighbour) /
                     C_isocyanide, C -> by that carbon's degree: 2 C_acetylide, 3 C_vinylidene, else C_alkylidyne
      aromatic       two heteroatom neighbours (N/O/S) -> C_carbene (NHC, thiazolylidene); else C_pi (aryl)
      two neighbours terminal O -> C_acyl (incl. alkoxycarbonyl, carbamoyl); a lone-pair heteroatom neighbour
                     (N deg 3, O/S deg 2) and no C=C -> C_carbene (CAAC, saturated NHC, Fischer); C=N-R -> C_iminoacyl; a C=C double bond -> C_vinyl;
                     otherwise -> C_alkylidene (M=CR2 / M=CHR)
      three or more  C_alkyl
    Returns None only for a carbon with no neighbours (bare C: left to the SMARTS tree).
    """
    nbrs = adj.get(c_idx, [])
    n = len(nbrs)
    if n == 0:
        return None
    deg = lambda i: len(adj.get(i, []))  # noqa: E731
    heavy = [(j, bo) for j, bo in nbrs if sym_of.get(j) != "H"]
    if n == 1:
        j, _ = nbrs[0]
        s = sym_of.get(j)
        if s == "O":
            return "C_carbonyl"
        if s == "S":
            return "C_thiocarbonyl"
        if s == "N":
            return "C_cyanide" if deg(j) == 1 else "C_isocyanide"
        if s == "C":
            return {2: "C_acetylide", 3: "C_vinylidene"}.get(deg(j), "C_alkylidyne")
        return "C_alkylidyne"
    if c_idx in aromatic_atoms:
        het = [j for j, _ in heavy if sym_of.get(j) in ("N", "O", "S")]
        return "C_carbene" if n == 2 and len(het) == 2 else "C_pi"
    if n == 2:
        cc_double = any(sym_of.get(j) == "C" and bo >= 1.75 and j not in aromatic_atoms for j, bo in heavy)
        lone_pair_x = [j for j, _ in heavy if _CARBENE_X_DEGREE.get(sym_of.get(j)) == deg(j)]
        if any(sym_of.get(j) == "O" and deg(j) == 1 for j, _ in heavy):
            return "C_acyl"  # acyl, alkoxycarbonyl, carbamoyl: a terminal O rules out a carbene
        if lone_pair_x and not cc_double:
            return "C_carbene"
        if any(sym_of.get(j) == "N" and deg(j) == 2 for j, _ in heavy):
            return "C_iminoacyl"
        return "C_vinyl" if cc_double else "C_alkylidene"
    return "C_alkyl"


#: Shared labels for donor pairs that resonance makes equivalent: (pair kind, donor element, partner element).
_RESONANCE_LABEL = {
    ("13", "N", "N"): "N_amidinate", ("13", "N", "O"): "N_amidate", ("13", "O", "N"): "O_amidate",
    ("15", "O", "O"): "O_diketonate", ("15", "N", "N"): "N_diketiminate",
    ("15", "O", "N"): "O_ketoiminate", ("15", "N", "O"): "N_ketoiminate",
    ("14", "O", "O"): "O_dioxolene",
}


def _resonance_donor_labels(
    donors: Sequence[int],
    sym_of: Dict[int, str],
    adj: Dict[int, List[Tuple[int, float]]],
    aromatic_atoms: Set[int],
) -> Dict[int, str]:
    """
    {donor: shared subclass} for kappa N/O donor pairs of one ligand that resonance makes equivalent, found from the
    graph alone (no bond orders): the DB path localises the double bond by crystal bond orders, the solve path where
    the SMILES wrote it, so per-atom typing gives the two donors of one delocalised chelate swapped labels.

      1,3 pair (4-ring chelate)  X-Z-Y, Z a three-neighbour C or N:     amidinate / triazenide N,N; amidate N,O
      1,5 pair (6-ring chelate)  X-C-C-C-Y, all three C three-neighbour, the end C not aromatic:
                                 beta-diketonate O,O; beta-diketiminate (nacnac) N,N; ketoiminate O,N
      1,4 pair (5-ring chelate)  O-C-C-O, both C three-neighbour: dioxolene (catecholate / semiquinone / quinone)
    O,O on one C / N (carboxylate, nitrate) already has its own bond-order-agnostic label and is skipped.
    """
    deg = lambda i: len(adj.get(i, []))  # noqa: E731
    nbr = lambda i: [j for j, _ in adj.get(i, [])]  # noqa: E731
    D = [d for d in donors if sym_of.get(d) in ("N", "O")]
    out: Dict[int, str] = {}
    for ia in range(len(D)):
        for ib in range(ia + 1, len(D)):
            a, b = D[ia], D[ib]
            ea, eb = sym_of[a], sym_of[b]
            kind = None
            common = set(nbr(a)) & set(nbr(b))
            if any(sym_of.get(z) in ("C", "N") and deg(z) == 3 for z in common):
                if not (ea == "O" and eb == "O"):
                    kind = "13"
            elif ea == "O" and eb == "O" and any(
                    sym_of.get(c1) == "C" and deg(c1) == 3 and sym_of.get(c2) == "C" and deg(c2) == 3 and b in nbr(c2)
                    for c1 in nbr(a) for c2 in nbr(c1) if c2 != a):
                kind = "14"
            else:
                for c1 in nbr(a):
                    if sym_of.get(c1) != "C" or deg(c1) != 3 or c1 in aromatic_atoms:
                        continue
                    for c2 in nbr(c1):
                        if c2 in (a, b) or sym_of.get(c2) != "C" or deg(c2) != 3:
                            continue
                        for c3 in nbr(c2):
                            if c3 not in (c1, a) and sym_of.get(c3) == "C" and deg(c3) == 3 \
                                    and c3 not in aromatic_atoms and b in nbr(c3):
                                kind = "15"
                if kind is None:
                    continue
            if kind is None:
                continue
            la, lb = _RESONANCE_LABEL.get((kind, ea, eb)), _RESONANCE_LABEL.get((kind, eb, ea))
            if la and lb:
                out[a], out[b] = la, lb
    return out
