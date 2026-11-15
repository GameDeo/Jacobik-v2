"""Core dataclasses: atoms, bonds, coordination sites, ligands, TMComplex, donor labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Atomic / bond primitives
# ---------------------------------------------------------------------------


@dataclass
class Atom:
    """One atom in a tmQM complex."""

    idx: int  # 1-based index matching the .BO file
    symbol: str  # Element symbol
    position: np.ndarray  # (3,) Cartesian coords in Ångström

    # Filled in during bond-graph construction
    total_valence: float = 0.0  # Sum of all bond orders from .BO file

    def __repr__(self) -> str:
        return f"Atom({self.idx}, {self.symbol}, {self.position.round(3)})"


@dataclass
class Bond:
    """A bond between two atoms with a Wiberg/DDEC bond order."""

    i: int  # 1-based atom index (from)
    j: int  # 1-based atom index (to)
    order: float  # Bond order from the .BO file

    @property
    def is_coordination(self) -> bool:
        """
        Heuristic: a bond is treated as a metal coordination bond when the
        bond order is clearly sub-single (< 0.75) yet non-negligible (> 0.08).
        Real single bonds between main-group atoms have BO ≈ 0.9-1.1+.
        """
        return self.order < 0.75

    def __repr__(self) -> str:
        return f"Bond({self.i}-{self.j}, BO={self.order:.3f})"


# ---------------------------------------------------------------------------
# Haptic interaction
# ---------------------------------------------------------------------------


@dataclass
class HapticInteraction:
    """
    Describes an eta-n metal–ligand haptic interaction.
    """

    metal_idx: int  # 1-based index of the metal atom
    donor_indices: list[int]  # 1-based indices of the n donor atoms
    haptic_order: int  # n  (eta-n)
    centroid: np.ndarray  # (3,) mean position of donor atoms
    donor_symbols: list[str]  # element symbols of donor atoms
    avg_bond_order: float  # mean metal–donor BO (useful for scoring)

    def __repr__(self) -> str:
        return (
            f"HapticInteraction(eta-{self.haptic_order}, "
            f"metal={self.metal_idx}, donors={self.donor_indices}, "
            f"centroid={self.centroid.round(3)})"
        )


# ---------------------------------------------------------------------------
# Coordination site — the single, verified description of one metal contact
# ---------------------------------------------------------------------------


@dataclass
class CoordinationSite:
    """
    One verified coordination point on the metal, produced AFTER donor
    classification and haptic detection.  This is the authoritative
    coordination record (weak through-space contacts rejected by the
    classifier are excluded).
    """

    kind: str  # "kappa" or "haptic"
    donor_indices: list[int]  # 1-based atom indices (1 for kappa, n for haptic)
    position: np.ndarray  # (3,) atom position (kappa) or centroid (haptic)
    label: str  # short donor label, e.g. "py", "η⁵", "OAc"
    avg_bond_order: float = 0.0  # mean raw metal-donor Wiberg BO (for scoring)
    haptic_order: int = 0  # n for haptic; 0 for kappa
    anchor: Optional[np.ndarray] = None  # target position relative to metal center

    @property
    def is_haptic(self) -> bool:
        return self.kind == "haptic"

    def __repr__(self) -> str:
        tag = f"eta-{self.haptic_order}" if self.is_haptic else "kappa"
        return (
            f"CoordinationSite({tag}, donors={self.donor_indices}, "
            f"{self.label}, pos={self.position.round(3)})"
        )


# ---------------------------------------------------------------------------
# Ligand topology for inverse kinematics
# ---------------------------------------------------------------------------


@dataclass
class RotatableBond:
    """A single-bond that can rotate freely (not in a ring, not terminal)."""

    i: int  # atom index (closer to metal / donor end)
    j: int  # atom index (tail end)
    order: float


@dataclass
class RigidBody:
    """
    A set of atom indices that move as one rigid unit.
    Examples: aromatic ring, fused ring system, bridging fragment.
    """

    atom_indices: list[int]
    is_aromatic: bool = False
    is_ring: bool = False
    label: str = ""


@dataclass
class PivotJoint:
    """
    One rotatable bond along a kinematic chain.
    """

    atom_i: int  # 1-based xyz index, donor-proximal end
    atom_j: int  # 1-based xyz index, tail end


@dataclass
class PivotChain:
    """
    Ordered sequence of rotatable bonds forming the kinematic chain
    for one donor anchor point.
    """

    joints: list[PivotJoint]
    donor_atoms: list[int]
    is_haptic: bool = False


# ---------------------------------------------------------------------------
# Ligand
# ---------------------------------------------------------------------------


@dataclass
class Ligand:
    """
    One ligand extracted from a metal complex.
    """

    ligand_id: int
    atom_indices: list[int]  # 1-based indices within the complex
    donor_atoms: list[int]  # subset that touch the metal

    # Populated by donor_classifier.classify().
    haptic_groups: List[HapticInteraction] = field(default_factory=list)

    # True only when set by engine.py._build_complex_shell.
    donors_are_authoritative: bool = False
    # Declared donors removed by haptic detection (far ends of a pi system that failed the bound-atom gap).
    dropped_donors: list[int] = field(default_factory=list)

    # IK topology (populated by decompose.py)
    rotatable_bonds: list[RotatableBond] = field(default_factory=list)
    rigid_bodies: list[RigidBody] = field(default_factory=list)
    pivot_chains: list[PivotChain] = field(default_factory=list)
    decomposition: Any = field(default=None)  # _LigandGraph from decompose.py
    # Ordered atom-index cycles of flexible macrocycles (>= 8 all-single-bond
    # ring atoms containing >= 2 donors); populated by decompose.py.
    macrocycles: list[list[int]] = field(default_factory=list)

    @property
    def haptic(self) -> Optional[HapticInteraction]:
        return self.haptic_groups[0] if self.haptic_groups else None

    # Denticity: how many independent attachment points to the metal.
    # Each haptic group counts as one point; remaining donor_atoms that
    # are not part of any haptic group each count as one.
    @property
    def denticity(self) -> int:
        haptic_donor_set: set[int] = set()
        for h in self.haptic_groups:
            haptic_donor_set.update(h.donor_indices)
        kappa_count = sum(1 for d in self.donor_atoms if d not in haptic_donor_set)
        return len(self.haptic_groups) + kappa_count

    def __repr__(self) -> str:
        haptic_str = f", haptic_groups={len(self.haptic_groups)}" if self.haptic_groups else ""
        return (
            f"Ligand(id={self.ligand_id}, atoms={len(self.atom_indices)}, "
            f"donors={self.donor_atoms}{haptic_str})"
        )


# ---------------------------------------------------------------------------
# Top-level complex
# ---------------------------------------------------------------------------


@dataclass
class TMComplex:
    """
    A transition-metal complex from the tmQM dataset.
    """

    # --- identity ---
    csd_code: str
    charge: int  # formal charge q
    spin_state: int  # spin multiplicity (2S+1); see fit/parser.py for the tmQM S -> multiplicity conversion
    stoichiometry: str
    mnd: int  # max neighbour distance used in bond inference
    csd_years: str

    # --- atoms & bonds ---
    atoms: list[Atom]
    bonds: list[Bond]

    # --- metal center ---
    metal_idx: int  # 1-based index of the transition/lanthanide metal
    metal_symbol: str

    coord_sites: list[CoordinationSite] = field(default_factory=list)

    # Coordination geometry label from coord_geom.classify(); None if not yet run.
    coord_geom: Optional[str] = None
    # Runner-up ideal shape and its deviation margin over coord_geom (nearest-shape labelling); None when the CN has one.
    coord_geom_alt: Optional[str] = None
    coord_geom_gap: Optional[float] = None

    # --- ligands ---
    ligands: list[Ligand] = field(default_factory=list)

    # --- donor classification (populated by classify_donors after parsing) ---
    donor_labels: dict = field(default_factory=dict)  # dict[int, DonorLabel]

    # --- quantum-chemical properties (from CSV, optional) ---
    electronic_energy: Optional[float] = None
    dispersion_energy: Optional[float] = None
    dipole_moment: Optional[float] = None
    metal_charge: Optional[float] = None
    hl_gap: Optional[float] = None
    homo_energy: Optional[float] = None
    lumo_energy: Optional[float] = None
    polarizability: Optional[float] = None
    smiles: Optional[str] = None

    # --- metal oxidation state (ionic model) ---
    # None = unknown.  Training data: from SMILES/header (core/oxidation.py);
    # generated complexes: user-supplied or deduced from charge and ligands.
    oxidation_state: Optional[int] = None
    oxidation_source: str = ""

    # --- convenience accessors ---

    def get_atom(self, idx: int) -> Atom:
        """Return the Atom with the given 1-based index."""
        return self.atoms[idx - 1]

    def positions(self) -> np.ndarray:
        """Return all atom positions as (N, 3) array."""
        return np.stack([a.position for a in self.atoms])

    @property
    def haptic_sites(self) -> list[CoordinationSite]:
        """All haptic coordination sites."""
        return [s for s in self.coord_sites if s.is_haptic]

    def coordination_number(self) -> int:
        """
        Effective coordination number: one per coordination site (each haptic
        face counts once regardless of eta-order).
        """
        return len(self.coord_sites)

    def summary(self) -> str:
        lines = [
            f"TMComplex: {self.csd_code}",
            f"  Metal      : {self.metal_symbol} (atom {self.metal_idx})",
            f"  Stoich     : {self.stoichiometry}",
            f"  Charge / S : {self.charge} / {self.spin_state}",
            f"  Atoms      : {len(self.atoms)}",
            f"  Bonds      : {len(self.bonds)}",
            f"  Coord sites: {len(self.coord_sites)}",
            f"  Coord geom : {self.coord_geom or 'unknown'}",
            f"  Haptic     : {len(self.haptic_sites)}",
            f"  Ligands    : {len(self.ligands)}",
        ]
        for s in self.coord_sites:
            lines.append(f"    {s}")
        for lig in self.ligands:
            lines.append(f"    {lig}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TMComplex({self.csd_code}, {self.metal_symbol}, {len(self.atoms)} atoms)"


# ---------------------------------------------------------------------------
# Donor classification (populated by donor_classifier.py)
# ---------------------------------------------------------------------------


@dataclass
class DonorLabel:
    """
    Classification of a single donor atom in a TMComplex.
    """

    xyz_idx: int
    family: str
    subclass: str
    is_haptic: bool = False

    def short(self) -> str:
        """
        One-line label suitable for py3Dmol annotation.
        Returns the subclass; strips the family prefix for brevity when
        the subclass already encodes it.

        Examples
        --------
        "N_pyridine"  -> "py"
        "O_phenoxide" -> "OPh"
        "H_hydride"   -> "H⁻"
        "C_haptic"    -> "η"
        """
        return _DONOR_SHORT.get(self.subclass, self.subclass)

    def __repr__(self) -> str:
        haptic = " [haptic]" if self.is_haptic else ""
        return f"DonorLabel({self.xyz_idx}, {self.subclass}{haptic})"


# Short-label lookup for DonorLabel.short()
_DONOR_SHORT: dict = {
    # C donors
    "C_carbonyl": "CO",
    "C_isocyanide": "CNR",
    "C_cyanide": "CN⁻",
    "C_carbene": "C:",
    "C_alkylidene": "=CR2",
    "C_vinylidene": "=C=",
    "C_alkylidyne": "≡CR",
    "C_acyl": "C(O)R",
    "C_iminoacyl": "C(NR)R",
    "C_vinyl": "vinyl",
    "C_acetylide": "C≡CR",
    "C_thiocarbonyl": "CS",
    "C_alkyl": "Cσ",
    "C_pi": "Cπ",
    # Haptic C subclasses — assigned by eta-order in classify()
    "C_allyl": "η³",
    "C_eta4": "η⁴",
    "C_eta5": "η⁵",
    "C_eta6": "η⁶",
    "C_eta7": "η⁷",
    "C_eta8": "η⁸",
    "C_haptic_any": "η",  # canonical bucket-level node
    "C_haptic": "η",  # eta-2 and unrecognised orders
    "C_donor": "C",
    # N donors
    "N_cyanide": "CN",
    "N_dinitrogen": "N₂",
    "N_nitrosyl": "NO",
    "N_NHC": "NHC",
    "N_pyridine": "py",
    "N_azole": "az",
    "N_aromatic_any": "Nar",  # canonical bucket-level node
    "N_aromatic": "Nar",
    "N_imine": "im",
    "N_amide": "amid",
    "N_amine": "am",
    "N_ammonia": "NH₃",
    "N_primary_amine": "NH₂R",
    "N_secondary_amine": "NHR₂",
    "N_tertiary_amine": "NR₃",
    "N_ammonium_like": "NR₄",
    "N_amido": "NR₂⁻",
    "N_imido": "NR²⁻",
    "N_donor": "N",
    # O donors
    "O_water": "H₂O",
    "O_sulfonate": "OSO₃",
    "O_nitro": "ONO₂",
    "O_carboxylate": "OAc",
    "O_phenoxide": "OPh",
    "O_ether_cyclic": "Ocy",
    "O_carbonyl": "C=O",
    "O_hydroxide": "OH⁻",
    "O_oxo": "O²⁻",
    "O_alcohol": "ROH",
    "O_heteroxo": "O=X",
    "O_alkoxide": "OR",
    "O_ether": "OEt",
    "O_donor": "O",
    # P donors
    "P_phosphine": "PR₃",
    "P_phosphite": "P(OR)₃",
    "P_phosphinite": "P(OR)R₂",
    "P_donor": "P",
    # S donors
    "S_sulfoxide": "S=O",
    "S_thiocarbonyl": "S=C",
    "S_thioether": "SR₂",
    "S_thiolate": "SR",
    "S_donor": "S",
    # H donors
    "H_hydride": "H⁻",
    "H_donor": "H",
    # Halide donors
    "Hal_F": "F⁻",
    "Hal_Cl": "Cl⁻",
    "Hal_Br": "Br⁻",
    "Hal_I": "I⁻",
    "Hal_donor": "X⁻",
    # B donors
    "B_boronate": "B",
    # Chalcogenide donors (Se, Te, As)
    "Se_donor": "Se",
    "Te_donor": "Te",
    "As_donor": "As",
    # Catch-all
    "Unknown_donor": "?",
}
