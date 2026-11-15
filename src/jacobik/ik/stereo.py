"""
src/jacobik/ik/stereo.py
========================
Stereoisomer perception, enumeration, mirror merging, and labeling (Scope B).

Directive I (§1.2 & §4b):
- Unspecified tetrahedral centres on ligands.
- Binding-created donor centres (N, P, Si, C, S) that become chiral upon coordination.
- Acyclic E/Z double bonds outside rings <= 7, including M-C=X bonds (aminocarbenes, alkenyls).
- Specified stereo elements (@/@@, /\\) are preserved and never modified.
- Mirror pairs collapse unless the metal environment or assignment makes them diastereomeric.
- Feasibility prefilter checks donor-pair reachability before embed/IK.
- Uniform class sampling when total classes exceed the cost ceiling (1.5x cal wall).
- 3D chiral-volume sign and E/Z labeling for every returned conformer row.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem

from jacobik.core import profile

log = logging.getLogger(__name__)

# Registered profile settings
STEREO_CANDIDATES: bool = profile.get("stereo_candidates")
STEREO_SCOPE: str = profile.get("stereo_scope")
STEREO_COST_CEILING: float = profile.get("stereo_cost_ceiling")
OUT_PER_CLASS: bool = profile.get("out_per_class")
OUT_PER_CLASS_K: int = profile.get("out_per_class_k")
OUT_PER_CLASS_WINDOW: float = profile.get("out_per_class_window")


class StereoType(str, Enum):
    LIGAND_TETRAHEDRAL = "ligand_tetrahedral"
    BINDING_CREATED_DONOR = "binding_created_donor"
    DOUBLE_BOND_EZ = "double_bond_ez"
    MCX_ROTAMER = "mcx_rotamer"


@dataclass
class StereoElement:
    """A perceived stereochemical element in a complex."""
    element_type: StereoType
    ligand_idx: int
    atom_idx: int  # atom index in ligand (or bond begin for double bonds)
    atom_idx_end: Optional[int] = None  # bond end for double bonds
    atom_symbol: str = ""
    neighbors: Tuple[int, ...] = ()  # neighbor atom indices in rank order
    name: str = ""
    specified: bool = False
    specified_value: Any = None


@dataclass
class StereoVariant:
    """A specific assignment of stereochemical configurations to all perceived elements."""
    config: Dict[str, int]  # element_name -> +1 / -1 (or 1 / 0 for cis/trans)
    class_id: str
    feasible: bool = True
    sampled: bool = True
    #: superset construction: a production class (solved exactly as the E/Z path would) or an extra class on top of one
    production: bool = False
    #: per-ligand production E/Z fix lists (``Engine._ez_combinations`` row) this class is solved with, or None
    prod_ez: Optional[List[Optional[list]]] = None
    ligand_fixes: Dict[int, List[Tuple[Any, ...]]] = field(default_factory=dict)


def is_achiral_geometry(geometry_label: str) -> bool:
    """Whether the coordination geometry is intrinsically achiral (has improper rotation or mirror plane)."""
    lbl = geometry_label.lower()
    chiral_geometries = {"c1", "c2", "c3", "d3_chiral"}
    if "chiral" in lbl or lbl in chiral_geometries:
        return False
    return True


def perceive_ligand_centres(mol_rd: Chem.Mol, ligand_idx: int, donor_atoms: Set[int]) -> List[StereoElement]:
    """Find unspecified tetrahedral centres in a ligand (excluding binding-created donors and rigid cage bridgeheads)."""
    elements = []
    if mol_rd is None:
        return elements
    Chem.AssignStereochemistry(mol_rd, force=True, cleanIt=True)
    ranks = list(Chem.CanonicalRankAtoms(mol_rd, breakTies=False))
    ring_info = mol_rd.GetRingInfo()
    for a in mol_rd.GetAtoms():
        idx = a.GetIdx()
        if idx in donor_atoms:
            continue  # handled separately under binding-created donors
        if a.GetAtomicNum() <= 1:
            continue
        nbrs = [nb.GetIdx() for nb in a.GetNeighbors()]
        n_hs = a.GetTotalNumHs(includeNeighbors=False)
        total_degree = a.GetDegree() + n_hs
        if total_degree != 4:
            continue
        # If more than 1 H, cannot be chiral
        if n_hs > 1:
            continue
        # Polycyclic rigid cage check (e.g. DAMVAQ): bridgehead atoms shared by >= 2 small rings (<= 6) cannot invert independently
        small_rings = [r for r in ring_info.AtomRings() if idx in r and len(r) <= 6]
        if len(small_rings) >= 2:
            continue
        has_chiral_tag = a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        nbr_ranks = [ranks[nb] for nb in nbrs]
        if n_hs == 1:
            nbr_ranks.append(-1)  # hydrogen distinct rank
        if len(set(nbr_ranks)) == 4:
            is_specified = has_chiral_tag
            spec_val = None
            if is_specified:
                ct = a.GetChiralTag()
                if ct == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                    spec_val = 1
                elif ct == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    spec_val = -1
            name = f"L{ligand_idx}_C{idx}_{a.GetSymbol()}"
            elements.append(StereoElement(
                element_type=StereoType.LIGAND_TETRAHEDRAL,
                ligand_idx=ligand_idx,
                atom_idx=idx,
                atom_symbol=a.GetSymbol(),
                neighbors=tuple(sorted(nbrs, key=lambda k: ranks[k])),
                name=name,
                specified=is_specified,
                specified_value=spec_val,
            ))
    return elements


def perceive_binding_created_donors(mol_rd: Chem.Mol, ligand_idx: int, donor_groups: Sequence[Sequence[int]]) -> List[StereoElement]:
    """Find donor atoms (N, P, Si, C, S) that become chiral tetrahedral centres upon metal coordination."""
    elements = []
    if mol_rd is None:
        return elements
    Chem.AssignStereochemistry(mol_rd, force=True, cleanIt=True)
    ranks = list(Chem.CanonicalRankAtoms(mol_rd, breakTies=False))
    single_donors = [g[0] for g in donor_groups if len(g) == 1 and g[0] < mol_rd.GetNumAtoms()]
    for d_idx in single_donors:
        a = mol_rd.GetAtomWithIdx(d_idx)
        sym = a.GetSymbol()
        if sym not in ("N", "P", "Si", "C", "S", "As"):
            continue
        nbrs = [nb.GetIdx() for nb in a.GetNeighbors()]
        n_hs = a.GetTotalNumHs(includeNeighbors=False)
        total_degree = a.GetDegree() + n_hs
        # If more than 1 H (e.g. -NH2), two H substituents make it achiral
        if n_hs > 1:
            continue
        
        has_chiral_tag = a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        spec_val = None
        if has_chiral_tag:
            ct = a.GetChiralTag()
            if ct == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                spec_val = 1
            elif ct == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                spec_val = -1

        # Case 1: Trivalent donor in free ligand (e.g. secondary amine -NH-, trivalent P, sulfoxide S, [CH-] carbanion)
        # Coordination to M adds the 4th substituent. Chiral iff all 3 ligand substituents are distinct.
        if total_degree == 3:
            if a.GetIsAromatic() or a.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            nbr_ranks = [ranks[nb] for nb in nbrs]
            if n_hs == 1:
                nbr_ranks.append(-1)
            if len(set(nbr_ranks)) == 3:
                name = f"L{ligand_idx}_D{d_idx}_{sym}"
                elements.append(StereoElement(
                    element_type=StereoType.BINDING_CREATED_DONOR,
                    ligand_idx=ligand_idx,
                    atom_idx=d_idx,
                    atom_symbol=sym,
                    neighbors=tuple(sorted(nbrs, key=lambda k: ranks[k])),
                    name=name,
                    specified=has_chiral_tag,
                    specified_value=spec_val,
                ))
        # Case 2: Divalent donor with lone pair in free ligand (e.g. silylene [si-] in HOYBED)
        # Total degree is 2, 0 Hs. Coordination to M and lone pair gives 4 positions.
        # Planar sp2 divalent carbon ([c-] in aryl, NHC, alkenyl) and neutral thioethers (degree 2 uncapped S) are excluded.
        elif total_degree == 2 and n_hs == 0 and sym in ("Si", "P", "As") and a.GetFormalCharge() != 0:
            if len(nbrs) == 2 and ranks[nbrs[0]] != ranks[nbrs[1]]:
                name = f"L{ligand_idx}_D{d_idx}_{sym}"
                elements.append(StereoElement(
                    element_type=StereoType.BINDING_CREATED_DONOR,
                    ligand_idx=ligand_idx,
                    atom_idx=d_idx,
                    atom_symbol=sym,
                    neighbors=tuple(sorted(nbrs, key=lambda k: ranks[k])),
                    name=name,
                    specified=has_chiral_tag,
                    specified_value=spec_val,
                ))
    return elements


def perceive_double_bonds(
    mol_rd: Chem.Mol,
    ligand_idx: int,
    smiles: str,
    donor_atoms: Optional[Set[int]] = None,
    donor_groups: Optional[Sequence[Sequence[int]]] = None,
) -> List[StereoElement]:
    """Find acyclic E/Z double bonds (outside rings and chelate coordination paths)."""
    elements = []
    if mol_rd is None:
        return elements
    from jacobik.builder.embed import chelate_ring_stereo
    fixed_crs = {(min(b, c), max(b, c)) for b, c, *_ in chelate_ring_stereo(mol_rd, donor_groups or [])}
    Chem.AssignStereochemistry(mol_rd, force=True, cleanIt=True)
    ranks = list(Chem.CanonicalRankAtoms(mol_rd, breakTies=False))
    for bd in mol_rd.GetBonds():
        if bd.GetBondType() != Chem.BondType.DOUBLE or bd.GetIsAromatic():
            continue
        if bd.IsInRing():
            continue
        b = bd.GetBeginAtomIdx()
        c = bd.GetEndAtomIdx()
        if (min(b, c), max(b, c)) in fixed_crs:
            continue
        # Skip cumulene / allene / carbodiimide bonds where either end is SP hybridized
        if bd.GetBeginAtom().GetHybridization() == Chem.HybridizationType.SP or \
           bd.GetEndAtom().GetHybridization() == Chem.HybridizationType.SP:
            continue
        # Skip carbon donor bonds coordinated directly to metal; handled by perceive_mcx_rotamers
        if donor_atoms:
            if (b in donor_atoms and mol_rd.GetAtomWithIdx(b).GetAtomicNum() == 6) or \
               (c in donor_atoms and mol_rd.GetAtomWithIdx(c).GetAtomicNum() == 6):
                continue
        nb = [x.GetIdx() for x in bd.GetBeginAtom().GetNeighbors() if x.GetIdx() != c]
        nc = [x.GetIdx() for x in bd.GetEndAtom().GetNeighbors() if x.GetIdx() != b]
        if not nb or not nc:
            continue
        rb = [ranks[x] for x in nb]
        rc = [ranks[x] for x in nc]
        if len(set(rb)) != len(rb) or len(set(rc)) != len(rc):
            continue
        is_specified = bd.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
        spec_val = None
        if is_specified:
            st = bd.GetStereo()
            if st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
                spec_val = 1
            elif st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
                spec_val = 0
        name = f"L{ligand_idx}_B{b}_{c}"
        elements.append(StereoElement(
            element_type=StereoType.DOUBLE_BOND_EZ,
            ligand_idx=ligand_idx,
            atom_idx=b,
            atom_idx_end=c,
            atom_symbol=f"{bd.GetBeginAtom().GetSymbol()}={bd.GetEndAtom().GetSymbol()}",
            neighbors=(min(nb, key=lambda k: ranks[k]), min(nc, key=lambda k: ranks[k])),
            name=name,
            specified=is_specified,
            specified_value=spec_val,
        ))
    return elements


def perceive_mcx_rotamers(mol_rd: Chem.Mol, ligand_idx: int, donor_groups: Sequence[Sequence[int]]) -> List[StereoElement]:
    """Find M-C=X double bonds (aminocarbenes, alkenyls) where M-D=X-Y forms an E/Z rotamer."""
    elements = []
    if mol_rd is None:
        return elements
    from jacobik.builder.embed import chelate_ring_stereo
    fixed_crs = {(min(b, c), max(b, c)) for b, c, *_ in chelate_ring_stereo(mol_rd, donor_groups or [])}
    Chem.AssignStereochemistry(mol_rd, force=True, cleanIt=True)
    ranks = list(Chem.CanonicalRankAtoms(mol_rd, breakTies=False))
    single_donors = {g[0] for g in donor_groups if len(g) == 1 and g[0] < mol_rd.GetNumAtoms()}
    for d_idx in single_donors:
        d_atom = mol_rd.GetAtomWithIdx(d_idx)
        # M-C=X bonds have metal-bound donor atom C
        if d_atom.GetSymbol() != "C":
            continue
        heavy_nbrs = [nb for nb in d_atom.GetNeighbors() if nb.GetAtomicNum() > 1]
        for x in heavy_nbrs:
            x_idx = x.GetIdx()
            if x_idx in single_donors:
                continue
            bd = mol_rd.GetBondBetweenAtoms(d_idx, x_idx)
            if bd is None:
                continue
            bt = bd.GetBondType()
            # Must strictly be a DOUBLE bond (triple bonds are linear and never rotameric)
            if bt != Chem.BondType.DOUBLE:
                continue
            if bd.IsInRing():
                continue
            if (min(d_idx, x_idx), max(d_idx, x_idx)) in fixed_crs:
                continue

            x_nbrs = [nb.GetIdx() for nb in x.GetNeighbors() if nb.GetIdx() != d_idx and nb.GetAtomicNum() > 1]
            if not x_nbrs:
                continue
            if len(x_nbrs) == 2 and ranks[x_nbrs[0]] == ranks[x_nbrs[1]]:
                continue

            d_other = [n.GetIdx() for n in d_atom.GetNeighbors() if n.GetIdx() != x_idx and n.GetAtomicNum() > 1]
            r_d = min(d_other, key=lambda k: ranks[k]) if d_other else -1
            r_x = min(x_nbrs, key=lambda k: ranks[k]) if x_nbrs else -1
            st = bd.GetStereo()
            is_specified = st not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
            spec_val = None
            if is_specified:
                if st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
                    spec_val = 1
                elif st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
                    spec_val = 0
            name = f"L{ligand_idx}_MCX_{d_idx}_{x_idx}"
            elements.append(StereoElement(
                element_type=StereoType.MCX_ROTAMER,
                ligand_idx=ligand_idx,
                atom_idx=d_idx,
                atom_idx_end=x_idx,
                atom_symbol=f"M-C={x.GetSymbol()}",
                neighbors=(r_d, r_x),
                name=name,
                specified=is_specified,
                specified_value=spec_val,
            ))
    return elements


def perceive_complex_stereo(builder: Any, scope: str = "B") -> List[StereoElement]:
    """Perceive all stereochemical elements across all ligands of a complex builder."""
    all_elements: List[StereoElement] = []
    for li, lig in enumerate(builder.ligands):
        m = Chem.MolFromSmiles(lig.smiles)
        if m is None:
            continue
        donors_flat = {d for g in lig.donors for d in g}
        # 1. Unspecified tetrahedral centres
        centres = perceive_ligand_centres(m, li, donors_flat)
        all_elements.extend(centres)
        # 2. Binding-created donors (Scope B)
        if scope.upper() == "B":
            binding_donors = perceive_binding_created_donors(m, li, lig.donors)
            all_elements.extend(binding_donors)
        # 3. Double bonds
        bonds = perceive_double_bonds(m, li, lig.smiles, donor_atoms=donors_flat, donor_groups=lig.donors)
        all_elements.extend(bonds)
        # 4. M-C=X rotamers / alkenyls (Scope B)
        if scope.upper() == "B":
            mcx = perceive_mcx_rotamers(m, li, lig.donors)
            all_elements.extend(mcx)
    return all_elements


def build_variant_ligand_fixes(
    variant: StereoVariant,
    elements: Sequence[StereoElement],
    ligands: Sequence[Any],
) -> Dict[int, Dict[str, list]]:
    """Build per-ligand ez and chiral fix lists for a stereo variant."""
    from collections import defaultdict
    lig_fixes = defaultdict(lambda: {"ez": [], "chiral": []})
    for el in elements:
        if el.specified:
            continue
        if el.name in variant.config:
            val = variant.config[el.name]
            if el.element_type in (StereoType.DOUBLE_BOND_EZ, StereoType.MCX_ROTAMER):
                b = el.atom_idx
                c = el.atom_idx_end if el.atom_idx_end is not None else b
                a = el.neighbors[0] if len(el.neighbors) > 0 else -1
                e = el.neighbors[1] if len(el.neighbors) > 1 else -1
                cis = bool(val == 1)
                lig_fixes[el.ligand_idx]["ez"].append((b, c, a, e, cis))
            elif el.element_type in (StereoType.LIGAND_TETRAHEDRAL, StereoType.BINDING_CREATED_DONOR):
                lig = ligands[el.ligand_idx]
                lig_smiles = getattr(lig, "smiles", lig) if not isinstance(lig, str) else lig
                lig_rd = Chem.MolFromSmiles(lig_smiles)
                at = lig_rd.GetAtomWithIdx(el.atom_idx) if lig_rd else None
                rd_nbrs = [nb.GetIdx() for nb in at.GetNeighbors() if nb.GetAtomicNum() > 1] if at else []
                el_nbrs = list(el.neighbors[:len(rd_nbrs)])
                sgn = 1
                for i in range(len(el_nbrs)):
                    for j in range(i + 1, len(el_nbrs)):
                        if rd_nbrs.index(el_nbrs[i]) > rd_nbrs.index(el_nbrs[j]):
                            sgn = -sgn
                flip = -1 if (at and at.GetFormalCharge() < 0 and (at.GetTotalNumHs() > 0 or len(rd_nbrs) == 2)) else 1
                tag = Chem.ChiralType.CHI_TETRAHEDRAL_CCW if (val * sgn * flip > 0) else Chem.ChiralType.CHI_TETRAHEDRAL_CW
                lig_fixes[el.ligand_idx]["chiral"].append((el.atom_idx, tag))
    return lig_fixes


def check_variant_feasibility(
    variant: StereoVariant,
    builder: Any,
    elements: Optional[Sequence[StereoElement]] = None,
) -> bool:
    """Prefilter: verify that donor pairs can geometrically reach the metal coordination distance
    and that the requested stereo variant can actually be built by the embedder."""
    from jacobik.builder import embed as _ce
    if not variant.config:
        return True
    if elements is None:
        elements = getattr(builder, "_perceived_stereo_elements", None)
        if elements is None:
            elements = perceive_complex_stereo(builder, scope="B")

    # 1. Baseline donor pair bounds check
    for li, lig in enumerate(builder.ligands):
        if len(lig.donors) > 1:
            m = Chem.MolFromSmiles(lig.smiles)
            if m is not None:
                bounds = _ce.donor_pair_bounds(m, lig.donors)
                for i, j, lo, hi in bounds:
                    if hi <= lo or not np.isfinite(hi):
                        return False

    # 2. Per-variant ligand embedding check
    lig_fixes = build_variant_ligand_fixes(variant, elements, builder.ligands)
    for li, lig in enumerate(builder.ligands):
        fixes = lig_fixes[li]
        if not fixes["ez"] and not fixes["chiral"]:
            continue
        ez_extra = fixes["ez"] or None
        chiral_extra = fixes["chiral"] or None
        try:
            r = _ce.embed(lig.smiles, lig.donors, seed=42, mmff=True, ez_extra=ez_extra, chiral_extra=chiral_extra, strict_stereo=True)
        except Exception:
            return False
        if r is None:
            return False
        rdmol_h, pos, info = r
        if info.get("violation") is not None and info["violation"] > 0.5:
            return False

        # Check that double bond stereo requested is achieved in the embedded conformer
        for b, c, a, e, cis in (ez_extra or []):
            if a < 0 or e < 0 or a >= len(pos) or e >= len(pos):
                continue
            axis = pos[c] - pos[b]
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                return False
            axis = axis / norm
            u = pos[a] - pos[b]
            w = pos[e] - pos[c]
            u_perp = u - (u @ axis) * axis
            w_perp = w - (w @ axis) * axis
            dot = float(u_perp @ w_perp)
            achieved_cis = (dot >= 0)
            if achieved_cis != cis:
                return False

    return True


def enumerate_stereo_variants(
    elements: List[StereoElement],
    achiral_environment: bool = True,
    cost_ceiling_multiplier: float = 1.5,
    baseline_classes: int = 1,
    seed: Optional[int] = None,
    builder: Optional[Any] = None,
) -> Tuple[List[StereoVariant], bool]:
    """Enumerate stereoisomer combinations, collapse mirror pairs in achiral environments,
    filter infeasible variants, and uniformly sample if exceeding the cost ceiling. Returns (variants, sampled_flag)."""
    if not elements:
        return [StereoVariant(config={}, class_id="default")], False

    # Separate elements into chiral centres (tetrahedral / binding) vs double bonds (E/Z / MCX)
    chiral_centres = [e for e in elements if e.element_type in (StereoType.LIGAND_TETRAHEDRAL, StereoType.BINDING_CREATED_DONOR)]
    double_bonds = [e for e in elements if e.element_type in (StereoType.DOUBLE_BOND_EZ, StereoType.MCX_ROTAMER)]

    specified_centres = {e.name: (1 if e.specified_value == 1 else -1) for e in chiral_centres if e.specified}
    unspecified_centres = [e for e in chiral_centres if not e.specified]

    specified_bonds = {e.name: (1 if e.specified_value == 1 else 0) for e in double_bonds if e.specified}
    unspecified_bonds = [e for e in double_bonds if not e.specified]

    # Enumerate centres combinations (+1, -1)
    n_uc = len(unspecified_centres)
    raw_centre_combos = list(itertools.product((-1, 1), repeat=n_uc)) if n_uc > 0 else [()]

    # Mirror-pair collapsing for chiral centres in achiral environment:
    # Inverting all unspecified centres produces the enantiomer. We keep one canonical representative.
    merged_centre_combos = []
    seen_mirror = set()
    for combo in raw_centre_combos:
        if achiral_environment and len(combo) > 0:
            inv_tuple = tuple(-v for v in combo)
            canonical = min(combo, inv_tuple)
            if canonical in seen_mirror:
                continue
            seen_mirror.add(canonical)
            c_dict = dict(zip([e.name for e in unspecified_centres], canonical))
            merged_centre_combos.append(c_dict)
        else:
            c_dict = dict(zip([e.name for e in unspecified_centres], combo))
            merged_centre_combos.append(c_dict)

    # Enumerate double bonds combinations (0 = trans/E, 1 = cis/Z)
    n_ub = len(unspecified_bonds)
    raw_bond_combos = list(itertools.product((0, 1), repeat=n_ub)) if n_ub > 0 else [()]
    bond_combos = []
    for combo in raw_bond_combos:
        b_dict = dict(zip([e.name for e in unspecified_bonds], combo))
        bond_combos.append(b_dict)

    # Full product of centre configurations and double bond configurations
    raw_variants: List[StereoVariant] = []
    for cc in merged_centre_combos:
        for bc in bond_combos:
            cfg = {**cc, **bc}
            all_cfgs = {**cfg, **specified_centres, **specified_bonds}
            cid_parts = [
                f"{k}:{'Z' if v == 1 else 'E'}" if k in (specified_bonds or bc) else f"{k}:{'+' if v > 0 else '-'}"
                for k, v in sorted(all_cfgs.items())
            ]
            class_id = "_".join(cid_parts) if cid_parts else "achiral"
            raw_variants.append(StereoVariant(config=cfg, class_id=class_id))

    if builder is not None:
        feasible_variants = []
        for v in raw_variants:
            if check_variant_feasibility(v, builder, elements):
                v.feasible = True
                feasible_variants.append(v)
            else:
                v.feasible = False
        if feasible_variants:
            raw_variants = feasible_variants
        else:
            raw_variants = [StereoVariant(config={}, class_id="default")]

    n_total = len(raw_variants)
    max_allowed = max(1, int(math.ceil(baseline_classes * cost_ceiling_multiplier)))
    sampled = False
    if n_total > max_allowed:
        rng = np.random.default_rng(seed if seed is not None else 42)
        indices = sorted(rng.choice(n_total, size=max_allowed, replace=False))
        selected_variants = [raw_variants[i] for i in indices]
        sampled = True
    else:
        selected_variants = raw_variants

    return selected_variants, sampled


def compute_conformer_stereo_labels(*args, **kwargs) -> Dict[str, str]:
    """Compute chiral volume signs (+/-) and E/Z signs on 3D conformer coordinates."""
    elements = kwargs.get("elements", None)
    positions = kwargs.get("positions", None)
    symbols = kwargs.get("symbols", None)
    c = kwargs.get("c", None)

    if args:
        first = args[0]
        if isinstance(first, np.ndarray) and first.ndim == 2:
            positions = first
            symbols = args[1] if len(args) > 1 else None
            elements = args[2] if len(args) > 2 else None
            if len(args) > 3:
                c = args[4] if len(args) > 4 else args[3]
        else:
            elements = first
            positions = args[1] if len(args) > 1 else None
            symbols = args[2] if len(args) > 2 else None
            if len(args) > 3:
                c = args[3]

    if not elements or not isinstance(elements, (list, tuple)):
        return {}
    if not isinstance(elements[0], StereoElement):
        return {}

    pos = np.asarray(positions)
    labels = {}

    complex_mol = getattr(c, "complex", None) if c is not None else None
    ligands = getattr(complex_mol, "ligands", None) if complex_mol is not None else None
    metal_0 = (complex_mol.metal_idx - 1) if (complex_mol is not None and hasattr(complex_mol, "metal_idx")) else 0

    for el in elements:
        lig = ligands[el.ligand_idx] if (ligands is not None and el.ligand_idx < len(ligands)) else None
        if lig is not None and hasattr(lig, "atom_indices"):
            lig_map = [idx - 1 for idx in lig.atom_indices]
        else:
            lig_map = list(range(len(pos)))

        if el.element_type in (StereoType.LIGAND_TETRAHEDRAL, StereoType.BINDING_CREATED_DONOR):
            idx = el.atom_idx
            if idx >= len(lig_map):
                continue
            idx_g = lig_map[idx]
            p0 = pos[idx_g]
            n_g = [lig_map[n] for n in el.neighbors if 0 <= n < len(lig_map)]
            if len(n_g) == 2:
                h_nbrs_g = []
                if complex_mol is not None and hasattr(complex_mol, "bonds"):
                    target_1 = idx_g + 1
                    for bd in complex_mol.bonds:
                        other_1 = bd.j if bd.i == target_1 else (bd.i if bd.j == target_1 else None)
                        if other_1 is not None and 0 <= (other_1 - 1) < len(complex_mol.atoms):
                            if complex_mol.atoms[other_1 - 1].symbol == "H":
                                h_nbrs_g.append(other_1 - 1)
                if h_nbrs_g:
                    n_g.append(h_nbrs_g[0])
                else:
                    n_g.append(metal_0)

            if len(n_g) >= 3:
                try:
                    v = [pos[k] - p0 for k in n_g[:3]]
                    v = [u / (np.linalg.norm(u) + 1e-12) for u in v]
                    vol = float(np.linalg.det(np.array(v)))
                    labels[el.name] = "+" if vol >= 0 else "-"
                except Exception:
                    labels[el.name] = "?"
            else:
                labels[el.name] = "?"

        elif el.element_type in (StereoType.DOUBLE_BOND_EZ, StereoType.MCX_ROTAMER):
            if el.atom_idx_end is None:
                continue
            b, c_idx = el.atom_idx, el.atom_idx_end
            if b >= len(lig_map) or c_idx >= len(lig_map):
                continue
            b_g = lig_map[b]
            c_g = lig_map[c_idx]

            ta = el.neighbors[0] if len(el.neighbors) > 0 else -1
            tb = el.neighbors[1] if len(el.neighbors) > 1 else -1

            if 0 <= ta < len(lig_map) and ta != b:
                ta_g = lig_map[ta]
            else:
                ta_g = metal_0

            if 0 <= tb < len(lig_map):
                tb_g = lig_map[tb]
            else:
                continue

            try:
                axis = pos[c_g] - pos[b_g]
                axis = axis / (np.linalg.norm(axis) + 1e-12)
                u = pos[ta_g] - pos[b_g]
                w = pos[tb_g] - pos[c_g]
                u_perp = u - (u @ axis) * axis
                w_perp = w - (w @ axis) * axis
                dot = float(u_perp @ w_perp)
                labels[el.name] = "Z" if dot >= 0 else "E"
            except Exception:
                labels[el.name] = "?"

    return labels


# ---------------------------------------------------------------------------------------------------------------------
# Superset class construction (jgen/i-stereo, 2026-11-09)
# ---------------------------------------------------------------------------------------------------------------------
def complex_seed(builder: Any) -> int:
    """Deterministic seed of a complex: its ligand SMILES and metal, not the call order or the solve seed."""
    import zlib
    key = "|".join(sorted(str(getattr(l, "smiles", l)) for l in builder.ligands))
    key += "|" + str(getattr(builder, "metal", ""))
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def extra_stereo_elements(elements: Sequence[StereoElement], prod_combos: Sequence[Sequence[Optional[list]]]) -> List[StereoElement]:
    """Elements that can yield classes beyond the production E/Z enumeration.

    Dropped: single-bond M-C-X ``MCX_ROTAMER`` elements (torsions the solver already searches; not stereo classes) and
    every E/Z double bond that the production enumeration already fixes in ``prod_combos``."""
    covered = set()
    for combo in prod_combos:
        for li, fixes in enumerate(combo):
            for b, c, *_ in (fixes or []):
                covered.add((li, min(b, c), max(b, c)))
    out = []
    for e in elements:
        if e.element_type == StereoType.MCX_ROTAMER:
            continue
        if e.element_type == StereoType.DOUBLE_BOND_EZ and e.atom_idx_end is not None \
                and (e.ligand_idx, min(e.atom_idx, e.atom_idx_end), max(e.atom_idx, e.atom_idx_end)) in covered:
            continue
        out.append(e)
    return out


def near_elements(builder: Any, elements: Sequence[StereoElement], extra_el: Sequence[StereoElement],
                  achiral_environment: bool, mirror: bool,
                  remote_derived: bool = False) -> Tuple[List[StereoElement], bool, Dict[str, Any]]:
    """P2R config_near: the elements whose configurations become solved classes, whether mirror pairs collapse, info.

    Near = binding-created donor centres and open elements within 2 bonds of a donor atom (ligand graph). Remote open
    elements are kept in the product when they cannot be derived after the solve (a centre or bond in a ring: inverting
    it locally changes the ring); remote acyclic elements too unless `remote_derived` (config_remote: their classes
    are derived from the solved rows, engine/configs). Mirror pairs collapse only when the reflection supplies the other member: the
    environment is achiral, out_mirror is on and no tetrahedral centre is tagged (with a tag, inverting the open centres
    alone gives a diastereomer, not the enantiomer)."""
    near, remote_ring, remote_acyclic = [], [], []
    mols: Dict[int, Any] = {}
    for e in extra_el:
        if e.specified:
            near.append(e)                      # fixed in every class; kept so the class ids carry it
            continue
        if e.element_type == StereoType.BINDING_CREATED_DONOR:
            near.append(e)
            continue
        lig = builder.ligands[e.ligand_idx]
        if e.ligand_idx not in mols:
            mols[e.ligand_idx] = Chem.MolFromSmiles(lig.smiles)
        m = mols[e.ligand_idx]
        ends = [e.atom_idx] + ([e.atom_idx_end] if e.atom_idx_end is not None else [])
        donors = {d for g in lig.donors for d in g}
        if m is None or not donors:
            near.append(e)
            continue
        D = Chem.GetDistanceMatrix(m)
        if min(D[a][d] for a in ends for d in donors) <= 2:
            near.append(e)
        elif any(m.GetAtomWithIdx(a).IsInRing() for a in ends) if e.atom_idx_end is None else \
                m.GetBondBetweenAtoms(e.atom_idx, e.atom_idx_end) is not None and \
                m.GetBondBetweenAtoms(e.atom_idx, e.atom_idx_end).IsInRing():
            remote_ring.append(e)
        else:
            remote_acyclic.append(e)
    tagged = any(e.specified and e.element_type in (StereoType.LIGAND_TETRAHEDRAL, StereoType.BINDING_CREATED_DONOR)
                 for e in elements)
    collapse = bool(achiral_environment and mirror and not tagged)
    info = {"n_near": sum(not e.specified for e in near), "remote_ring": [e.name for e in remote_ring],
            "remote_acyclic": [e.name for e in remote_acyclic], "mirror_collapse": collapse}
    if remote_derived:
        info["remote_derived"] = info["remote_acyclic"]
    if remote_ring or remote_acyclic:
        log.info("config_near: %d near; remote in the product: ring %s%s", info["n_near"], info["remote_ring"],
                 f"; derived after the solve: {info['remote_acyclic']}" if remote_derived
                 else f"; acyclic {info['remote_acyclic']}")
    return near + remote_ring + ([] if remote_derived else remote_acyclic), collapse, info


def _raw_variants(elements: Sequence[StereoElement], achiral_environment: bool) -> List[StereoVariant]:
    """Every unspecified-element configuration (mirror pairs collapsed in an achiral environment), no feasibility test,
    no sampling. Order is deterministic (itertools.product over the element order)."""
    vs, _ = enumerate_stereo_variants(list(elements), achiral_environment=achiral_environment,
                                      cost_ceiling_multiplier=1e9, baseline_classes=1, builder=None)
    return [v for v in vs if v.config]


def superset_variants(
    builder: Any,
    elements: Sequence[StereoElement],
    prod_combos: Sequence[Sequence[Optional[list]]],
    achiral_environment: bool = True,
    cost_ceiling_multiplier: float = 1.5,
    near: bool = False,
    mirror: bool = False,
    remote: bool = False,
) -> Tuple[List[StereoVariant], Dict[str, Any]]:
    """Class list for ``stereo_candidates``: every production E/Z combination (same fixes, same order, solved exactly
    as the E/Z path does) followed by at most ``max(min(4, n_new_configs), ceil(n_prod * (multiplier - 1)))`` extra classes (at least 1 when extras
    exist) built from the new stereo elements. An extra is (new-element configuration) x (one production combination),
    so the production E/Z bonds stay fixed in it.

    Extras are chosen deterministically: candidate configurations are ordered by a pseudo-random permutation seeded
    from the complex (``complex_seed``; no cheap prior exists), the first feasible ones are taken (feasibility = the
    ligand embeds with the requested configuration), and the k-th extra is paired with production combination k mod n_prod.
    """
    combos = list(prod_combos) if prod_combos else [[None] * len(builder.ligands)]
    n_prod = len(combos)
    prod = [StereoVariant(config={}, class_id=f"prod{i}", production=True, prod_ez=[x or None for x in combo])
            for i, combo in enumerate(combos)]
    info: Dict[str, Any] = {"n_prod": n_prod, "n_extra": 0, "n_candidates": 0, "n_dropped_infeasible": 0, "cap": 0}
    extra_el = extra_stereo_elements(elements, combos)
    if not any(not e.specified for e in extra_el):
        return prod, info
    if near:
        extra_el, collapse, ninfo = near_elements(builder, elements, extra_el, achiral_environment, mirror,
                                                  remote_derived=remote)
        info.update(ninfo)
        cands = _raw_variants(extra_el, collapse) if any(not e.specified for e in extra_el) else []
    else:
        cands = _raw_variants(extra_el, achiral_environment)
    info["n_candidates"] = len(cands)
    if not cands:
        return prod, info
    cap = max(min(4, len(cands)), int(math.ceil(n_prod * (cost_ceiling_multiplier - 1.0))))
    if near:
        # one batch: a class costs its rows, not another sequential solve; the measured batch speed-up (x1.3-2.1 on
        # BOTYUH, FUTSOD, USAPOU, TEDPOI, XACMOE) pays for twice the sequential cap at about the same wall
        cap = 2 * cap
        if len(cands) > cap:
            info["cap_bites"] = True
            log.info("config_near: %d candidate classes, cap %d", len(cands), cap)
    info["cap"] = cap
    rng = np.random.default_rng(complex_seed(builder))
    order = rng.permutation(len(cands))
    extras: List[StereoVariant] = []
    for ci in order:
        if len(extras) >= cap:
            break
        v = cands[int(ci)]
        if not check_variant_feasibility(v, builder, extra_el):
            info["n_dropped_infeasible"] += 1
            continue
        k = len(extras)
        v.production = False
        v.prod_ez = [x or None for x in combos[k % n_prod]]
        v.class_id = f"x{k}_" + v.class_id
        extras.append(v)
    info["n_extra"] = len(extras)
    return prod + extras, info
