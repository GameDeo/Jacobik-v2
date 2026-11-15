"""
builder/ligands.py
================
Per-ligand descriptors for the tmQM reference complexes: split the complex SMILES on the metal, map each ligand
fragment onto the parsed xyz (RDKit heavy-atom graph -> xyz indices, with the ligand's graph automorphisms) and build
the engine's LigandInput list. Used by the benchmark tools, the regression panel and refit/thermal_widths.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RDKit warning suppression
# ---------------------------------------------------------------------------
# RDKit writes valence errors, sanitisation warnings, and kekulisation
# failures to stderr via its own logging system (rdBase), bypassing Python's
# logging module entirely.  Silencing these here — before any rdkit.Chem call
# — keeps benchmark output clean regardless of what SMILES are encountered.
try:
    from rdkit import RDLogger as _RDLogger
    _RDLogger.logger().setLevel(_RDLogger.CRITICAL)
    from rdkit.rdBase import DisableLog as _DisableLog
    for _ch in ("rdApp.error", "rdApp.warning", "rdApp.info", "rdApp.debug"):
        _DisableLog(_ch)
except ImportError:
    pass  # RDKit not installed; helpers fall back gracefully

# ---------------------------------------------------------------------------
# Per-ligand SMILES extraction from a full complex SMILES
# ---------------------------------------------------------------------------


def _ligand_smiles_from_complex(
    complex_smiles: str,
    metal_symbol: str,
) -> List[str]:
    """
    Split a complex SMILES into per-ligand SMILES by fragmenting on the metal.

    The metal is replaced by a dummy attachment point, and each connected
    fragment (ligand) is extracted as a canonical SMILES.

    Parameters
    ----------
    complex_smiles : full complex SMILES (from tmQM_y.csv)
    metal_symbol   : e.g. "Fe", "Ru"

    Returns
    -------
    list of canonical SMILES strings, one per ligand fragment.
    Empty list if RDKit cannot parse the SMILES.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(complex_smiles)
        if mol is None:
            return []

        # Find the metal atom index in RDKit mol
        metal_idx_rdkit: Optional[int] = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == metal_symbol:
                metal_idx_rdkit = atom.GetIdx()
                break
        if metal_idx_rdkit is None:
            return []

        # Remove the metal atom — this disconnects the ligands
        edit = Chem.RWMol(mol)
        edit.RemoveAtom(metal_idx_rdkit)
        frag_mol = edit.GetMol()

        # GetMolFrags(asMols=True) reliably returns one RDKit Mol per fragment,
        # avoiding the empty-string issue with MolFragmentToSmiles on charged atoms.
        frag_mols = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
        frag_smiles = [Chem.MolToSmiles(f, canonical=True) for f in frag_mols]

        return frag_smiles

    except Exception as exc:
        log.warning("SMILES fragmentation failed: %s", exc)
        return []


def canonical_smiles(smi: str) -> str:
    """Return RDKit canonical SMILES, or the input if RDKit fails."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smi


# ---------------------------------------------------------------------------
# Graph automorphisms of a molecule (intra-ligand symmetry)
# ---------------------------------------------------------------------------


def _mol_automorphisms(smi: str) -> List[Tuple[int, ...]]:
    """
    Return all graph automorphisms of the molecule given by *smi*.

    Each automorphism is a tuple of RDKit heavy-atom indices: position i
    holds the atom that maps TO position i under this symmetry operation.
    Concretely, automorphisms[k][i] is the index of the atom in the k-th
    symmetric image that corresponds to atom i in the canonical ordering.

    This is obtained via RDKit self-substructure matching:
        mol.GetSubstructMatches(mol, useChirality=False)
    which returns exactly the automorphism group of the molecular graph
    (ignoring stereo, which is appropriate for 3-D RMSD symmetry correction).

    The identity permutation is always included.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [tuple(range(mol.GetNumAtoms()))] if mol else [()]

        # Remove Hs for the automorphism search (we work in heavy-atom space)
        mol = Chem.RemoveHs(mol)
        n = mol.GetNumAtoms()

        # uniquify=False is required — default (True) returns only one match
        # per unique set of matched atoms, collapsing all automorphisms to one.
        matches = mol.GetSubstructMatches(mol, useChirality=False, uniquify=False)
        if not matches:
            return [tuple(range(n))]
        return list(matches)

    except Exception as exc:
        log.debug("Automorphism search failed for %r: %s", smi, exc)
        n_fallback = len(smi)  # crude upper bound; identity only
        return [tuple(range(n_fallback))]


# ---------------------------------------------------------------------------
# Build the xyz-index automorphism from RDKit automorphism + VF2 mapping
# ---------------------------------------------------------------------------


def _rdkit_to_xyz_automorphisms(
    rdkit_autos: List[Tuple[int, ...]],
    rdkit_to_xyz: Dict[int, int],  # heavy-atom rdkit_idx -> 1-based xyz idx
    all_xyz_indices: List[int],    # all xyz atom indices (heavy+H) for this ligand
    xyz_atoms,                      # list of Atom objects (for H neighbour lookup)
    xyz_bonds,                      # list of Bond objects
) -> List[List[int]]:
    """
    Lift RDKit heavy-atom automorphisms to full-atom (heavy+H) xyz permutations.

    For each RDKit automorphism (permutation of heavy atoms), we:
      1. Map heavy-atom rdkit indices → xyz indices via rdkit_to_xyz.
      2. For each heavy atom that moves, reassign its H neighbours to match
         (H atoms are locally symmetric around any sp3/sp2 centre, so any
         assignment is valid for RMSD purposes — we use sorted order).
      3. Return a list of permutations, each mapping old xyz index → new xyz index,
         covering ALL atoms in this ligand (heavy + H).

    Parameters
    ----------
    rdkit_autos    : output of _mol_automorphisms (RDKit heavy-atom indices)
    rdkit_to_xyz   : dict mapping RDKit heavy-atom idx → 1-based xyz idx
    all_xyz_indices: all 1-based xyz indices belonging to this ligand (heavy+H)
    xyz_atoms      : Atom list for the full complex
    xyz_bonds      : Bond list for the full complex

    Returns
    -------
    List of permutations. Each permutation is a list where
    permutation[i] = j means "old xyz index i maps to new xyz index j".
    We represent it as a dict: {old_xyz_idx: new_xyz_idx}.
    """
    # Build xyz→H-neighbours lookup
    xyz_atom_map = {a.idx: a for a in xyz_atoms}
    h_neighbours: Dict[int, List[int]] = {}
    for bond in xyz_bonds:
        for (src, dst) in [(bond.i, bond.j), (bond.j, bond.i)]:
            if src in rdkit_to_xyz.values() and dst not in rdkit_to_xyz.values():
                # dst is an H (or non-heavy) neighbour of a ligand heavy atom
                if xyz_atom_map.get(dst) and xyz_atom_map[dst].symbol == "H":
                    h_neighbours.setdefault(src, []).append(dst)
    for k in h_neighbours:
        h_neighbours[k] = sorted(h_neighbours[k])

    # Inverse mapping: xyz_idx -> rdkit_heavy_idx
    xyz_to_rdkit = {v: k for k, v in rdkit_to_xyz.items()}

    result_perms: List[Dict[int, int]] = []

    for auto in rdkit_autos:
        # auto[i] = j means rdkit atom i maps to rdkit atom j under this symmetry
        perm: Dict[int, int] = {}

        # Map heavy atoms
        for rdkit_src, rdkit_dst in enumerate(auto):
            xyz_src = rdkit_to_xyz.get(rdkit_src)
            xyz_dst = rdkit_to_xyz.get(rdkit_dst)
            if xyz_src is None or xyz_dst is None:
                continue
            perm[xyz_src] = xyz_dst

            # Map H neighbours of this heavy atom
            h_src = h_neighbours.get(xyz_src, [])
            h_dst = h_neighbours.get(xyz_dst, [])
            # Match H atoms in sorted order — valid since H neighbours are
            # locally equivalent under any graph automorphism of the heavy atom.
            for hs, hd in zip(sorted(h_src), sorted(h_dst)):
                perm[hs] = hd

        # Any unmapped atoms in this ligand map to themselves
        for idx in all_xyz_indices:
            if idx not in perm:
                perm[idx] = idx

        result_perms.append(perm)

    return result_perms


# ---------------------------------------------------------------------------
# Extract per-ligand information from a parsed TMComplex
# ---------------------------------------------------------------------------


def _build_ligand_groups(mol, complex_smiles: str) -> List[Dict]:
    """
    For each ligand in *mol*, build a descriptor dict containing:
      - canonical_smiles     : canonical SMILES for this ligand
      - xyz_indices_0based   : 0-based xyz indices of all atoms in this ligand
      - xyz_automorphisms    : list of {1based_xyz → 1based_xyz} dicts, one
                               per graph automorphism (intra-ligand symmetry)
      - donor_groups         : donor groups in RDKit heavy-atom index space
                               (for LigandInput construction)

    Index safety guarantee
    ----------------------
    Automorphisms are computed in RDKit heavy-atom index space, then lifted
    to 1-based xyz index space via the VF2 heavy-atom mapping (rdkit→xyz).
    The two index spaces are kept strictly separate and combined only here.
    """
    from rdkit import Chem
    import networkx as nx

    atom_map = {a.idx: a for a in mol.atoms}  # 1-based → Atom

    # Build the xyz→symbol lookup for H detection
    def is_hydrogen(xyz_1based: int) -> bool:
        a = atom_map.get(xyz_1based)
        return a is not None and a.symbol == "H"

    # Fragment the complex SMILES → per-ligand canonical SMILES
    frag_smiles_list = _ligand_smiles_from_complex(complex_smiles, mol.metal_symbol)

    # Sort ligands by ligand_id to have a stable order
    sorted_ligands = sorted(mol.ligands, key=lambda l: l.ligand_id)

    # We need to match each Ligand object to one of the SMILES fragments.
    # Strategy: match by element composition (multiset of symbols).
    # This works as long as ligands have distinct element multisets;
    # for identical ligands we assign in order.

    def element_multiset(xyz_indices: List[int]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for idx in xyz_indices:
            sym = atom_map.get(idx, None)
            if sym:
                counts[sym.symbol] = counts.get(sym.symbol, 0) + 1
        return counts

    def smi_element_multiset(smi: str) -> Dict[str, int]:
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return {}
            m = Chem.AddHs(m)
            counts: Dict[str, int] = {}
            for atom in m.GetAtoms():
                s = atom.GetSymbol()
                counts[s] = counts.get(s, 0) + 1
            return counts
        except Exception:
            return {}

    # Match ligands → SMILES fragments by element multiset
    unmatched_smiles = list(enumerate(frag_smiles_list))
    lig_to_smi: Dict[int, str] = {}

    for lig in sorted_ligands:
        lig_elements = element_multiset(lig.atom_indices)
        best_match_idx = None
        for j, (orig_j, smi) in enumerate(unmatched_smiles):
            smi_elements = smi_element_multiset(smi)
            if smi_elements == lig_elements:
                best_match_idx = j
                break
        if best_match_idx is not None:
            _, matched_smi = unmatched_smiles.pop(best_match_idx)
            lig_to_smi[lig.ligand_id] = matched_smi
        else:
            # Fallback: no element-matched SMILES found; use empty string
            lig_to_smi[lig.ligand_id] = ""
            log.warning(
                "Could not match ligand %d to a SMILES fragment for %s",
                lig.ligand_id, mol.csd_code,
            )

    groups: List[Dict] = []
    for lig in sorted_ligands:
        smi = lig_to_smi.get(lig.ligand_id, "")
        can_smi = canonical_smiles(smi) if smi else ""

        # 0-based xyz indices for all atoms in this ligand
        xyz_0based = [idx - 1 for idx in lig.atom_indices]  # atom_indices are 1-based

        # Compute intra-ligand graph automorphisms
        if smi:
            rdkit_autos = _mol_automorphisms(smi)
        else:
            # No SMILES: identity only
            rdkit_autos = [tuple(range(len(lig.atom_indices)))]

        # Build rdkit_heavy_idx → xyz_1based mapping via element-order matching.
        # We use the same VF2 logic as mapping.py: parse the SMILES, remove H,
        # and match against the heavy atoms of this ligand by symbol.
        rdkit_to_xyz: Dict[int, int] = {}
        try:
            rdmol = Chem.MolFromSmiles(smi)
            if rdmol is not None:
                rdmol_h = Chem.RemoveHs(rdmol)
                # List of (rdkit_idx, symbol) for SMILES heavy atoms
                smi_heavy = [(a.GetIdx(), a.GetSymbol()) for a in rdmol_h.GetAtoms()]
                smi_nh = {a.GetIdx(): a.GetTotalNumHs() for a in rdmol_h.GetAtoms()}
                # List of (xyz_1based, symbol) for ligand heavy atoms
                lig_heavy = [
                    (idx, atom_map[idx].symbol)
                    for idx in lig.atom_indices
                    if not is_hydrogen(idx)
                ]
                # VF2 isomorphism via networkx for correct mapping
                G_smi = nx.Graph()
                for rdkit_i, sym in smi_heavy:
                    G_smi.add_node(rdkit_i, symbol=sym)
                for bond in rdmol_h.GetBonds():
                    G_smi.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

                G_xyz = nx.Graph()
                for xyz_idx, sym in lig_heavy:
                    G_xyz.add_node(xyz_idx, symbol=sym)
                for bond in mol.bonds:
                    bi, bj = bond.i, bond.j
                    if bi in {x for x, _ in lig_heavy} and bj in {x for x, _ in lig_heavy}:
                        G_xyz.add_edge(bi, bj)

                # H count per heavy atom on both sides: without it, atoms that differ only in their H (a [C-3]
                # alkylidyne carbon vs the CH3 groups on the same Si, QOSGIP) are interchangeable and the donor
                # can be mapped onto a methyl. Fall back to element-only matching when the H counts disagree
                # somewhere (a SMILES protonated differently from the crystal).
                lig_heavy_set = {x for x, _ in lig_heavy}
                xyz_nh = {x: 0 for x in lig_heavy_set}
                for bond in mol.bonds:
                    for u, w in ((bond.i, bond.j), (bond.j, bond.i)):
                        if u in lig_heavy_set and is_hydrogen(w) and bond.order >= 0.5:
                            xyz_nh[u] += 1
                for n_ in G_smi.nodes:
                    G_smi.nodes[n_]["nh"] = smi_nh.get(n_, 0)
                for n_ in G_xyz.nodes:
                    G_xyz.nodes[n_]["nh"] = xyz_nh.get(n_, 0)

                def node_match(n1, n2):
                    return n1["symbol"] == n2["symbol"] and n1["nh"] == n2["nh"]

                GM = nx.algorithms.isomorphism.GraphMatcher(G_smi, G_xyz, node_match=node_match)
                if not GM.is_isomorphic():
                    GM = nx.algorithms.isomorphism.GraphMatcher(
                        G_smi, G_xyz, node_match=lambda a, b: a["symbol"] == b["symbol"])

                # Pick the geometrically consistent VF2 isomorphism.
                #
                # Problem: for symmetric ligands (e.g. two identical aziridine
                # arms), the first VF2 solution is arbitrary — it may map SMILES
                # atoms to the geometrically distant equivalent arm, making
                # rdkit_to_xyz a valid but spatially wrong correspondence.
                # For VUCVUN this caused 4.52 A self-RMSD and the long
                # cross-arm displacement vectors visible in the overlay.
                #
                # Solution: score each isomorphism by the sum of squared bond
                # lengths it implies in the reference geometry.  A correct
                # mapping preserves short SMILES bonds as short xyz distances;
                # a cross-arm swap inflates them.  We pick the minimum-cost
                # isomorphism.  Cap at 512 candidates for large automorphism
                # groups to avoid O(n!) enumeration on highly symmetric ligands.
                _MAX_ISO = 512
                best_iso = None
                best_iso_cost = float("inf")
                _n_iso_checked = 0
                for iso_candidate in GM.isomorphisms_iter():
                    if _n_iso_checked >= _MAX_ISO:
                        break
                    _n_iso_checked += 1
                    cost = 0.0
                    valid = True
                    for bond in rdmol_h.GetBonds():
                        a_rdkit = bond.GetBeginAtomIdx()
                        b_rdkit = bond.GetEndAtomIdx()
                        a_xyz = iso_candidate.get(a_rdkit)
                        b_xyz = iso_candidate.get(b_rdkit)
                        if a_xyz is None or b_xyz is None:
                            valid = False
                            break
                        pa = atom_map.get(a_xyz)
                        pb = atom_map.get(b_xyz)
                        if pa is None or pb is None:
                            valid = False
                            break
                        d = pa.position - pb.position
                        cost += float(d @ d)
                    if not valid:
                        continue
                    if cost < best_iso_cost:
                        best_iso_cost = cost
                        best_iso = iso_candidate

                if best_iso is not None:
                    rdkit_to_xyz = best_iso  # rdkit_idx -> xyz_1based
        except Exception as exc:
            log.debug("VF2 mapping failed for ligand %d: %s", lig.ligand_id, exc)

        # Lift automorphisms from RDKit idx space to xyz idx space
        if rdkit_to_xyz:
            xyz_autos = _rdkit_to_xyz_automorphisms(
                rdkit_autos,
                rdkit_to_xyz,
                lig.atom_indices,
                mol.atoms,
                mol.bonds,
            )
        else:
            # Identity only as fallback
            xyz_autos = [{idx: idx for idx in lig.atom_indices}]

        # Donor groups in RDKit heavy-atom index space (for LigandInput)
        # donor_atoms are 1-based xyz indices; map back to RDKit indices
        xyz_to_rdkit = {v: k for k, v in rdkit_to_xyz.items()}
        donor_groups_rdkit: List[List[int]] = []

        # Group donor atoms by haptic interactions first
        for site in mol.coord_sites:
            site_donors = set(site.donor_indices)
            lig_donor_set = set(lig.atom_indices)
            overlapping = sorted(site_donors & lig_donor_set)
            if not overlapping:
                continue
            rdkit_group = []
            for xyz_1based in overlapping:
                rdkit_i = xyz_to_rdkit.get(xyz_1based)
                if rdkit_i is not None:
                    rdkit_group.append(rdkit_i)
            if rdkit_group:
                donor_groups_rdkit.append(sorted(rdkit_group))

        # Deduplicate (same site might appear via multiple haptic atoms)
        seen = set()
        unique_donor_groups: List[List[int]] = []
        for grp in donor_groups_rdkit:
            key = tuple(sorted(grp))
            if key not in seen:
                seen.add(key)
                unique_donor_groups.append(grp)

        groups.append({
            "ligand_id": lig.ligand_id,
            "canonical_smiles": can_smi,
            "raw_smiles": smi,
            "xyz_indices_0based": xyz_0based,
            "xyz_automorphisms": xyz_autos,
            "donor_groups_rdkit": unique_donor_groups,
            "rdkit_to_xyz": rdkit_to_xyz,
        })

    return groups


# ---------------------------------------------------------------------------
# Build LigandInput list from a parsed TMComplex
# ---------------------------------------------------------------------------


def ligand_inputs_from_mol(mol, ligand_groups: List[Dict]):
    """
    Construct LigandInput objects for the engine from a parsed TMComplex
    and its pre-computed ligand_groups descriptor.

    Returns
    -------
    list of LigandInput objects (one per ligand, in sorted ligand_id order)
    """
    from jacobik.builder.result import LigandInput

    inputs = []
    for lg in ligand_groups:
        smi = lg["raw_smiles"]
        donor_groups = lg["donor_groups_rdkit"]
        if not smi or not donor_groups:
            log.warning(
                "Ligand %d: missing SMILES or donor groups — skipping complex",
                lg["ligand_id"],
            )
            return None
        inputs.append(LigandInput(smiles=smi, donors=donor_groups))
    return inputs
