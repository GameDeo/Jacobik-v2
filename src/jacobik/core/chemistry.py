"""Chemical tables, covalent radii, metal oxidation-state inference, and Shannon ionic radii."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Sentinel for unknown metal oxidation state and lanthanide element set.
OS_UNKNOWN = 99
LANTHANIDES = frozenset("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())


# ---------------------------------------------------------------------------
# Element tables & covalent / fallback vdW radii
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

#: d-block group numbers (oxidation-state upper bound; Shannon spin labels)
D_GROUP = {"Sc": 3, "Y": 3, "La": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5,
           "Cr": 6, "Mo": 6, "W": 6, "Mn": 7, "Tc": 7, "Re": 7, "Fe": 8, "Ru": 8, "Os": 8,
           "Co": 9, "Rh": 9, "Ir": 9, "Ni": 10, "Pd": 10, "Pt": 10, "Cu": 11, "Ag": 11,
           "Au": 11, "Zn": 12, "Cd": 12, "Hg": 12}

BOND_GATE_RADII: Dict[str, float] = {
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


def covalent_radius(symbol: str) -> float:
    """Return a covalent radius in angstrom with robust fallback defaults (bond-gate table)."""
    r = BOND_GATE_RADII.get(symbol)
    if r is not None:
        return r
    return 1.45 if symbol in METALS else 0.77


#: Covalent radii in Å for Q6 hard fallback (Alvarez 2008 values)
Q6_FALLBACK_RADII: Dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Te": 1.38,
    "I": 1.39,
    # Transition metals (typical values)
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.61,
    "Fe": 1.52,
    "Co": 1.50,
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
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
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
}
Q6_FALLBACK_DEFAULT_RADIUS = 1.50  # fallback for unknown elements

#: Symbols covered by the "Transition metals" section of Q6_FALLBACK_RADII
#: above. Used by _fallback_q6 to detect metal-involving pairs and apply the
# VDW-like scale/tighter sigma instead of raw covalent-radii-sum.
TRANSITION_METALS = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
        "Er", "Tm", "Yb", "Lu",
    }
)


# ---------------------------------------------------------------------------
# Oxidation state (ionic model)
# ---------------------------------------------------------------------------
# Production resolved configuration (hardcoded u1 value)
OXIDATION_MODEL = "training"

_LN_Z = {s: 57 + i for i, s in enumerate("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())}


def plausible(metal: str, os_: int) -> bool:
    top = 4 if metal in LANTHANIDES else D_GROUP.get(metal, 8)
    return -4 <= os_ <= top


def metal_formal_charge_from_smiles(smiles: Optional[str], metal: str) -> Optional[int]:
    """Return the formal charge written on the unique target metal in SMILES."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None
    atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == metal]
    return int(atoms[0].GetFormalCharge()) if len(atoms) == 1 else None


# ---------------------------------------------------------------------------
# Training labels
# ---------------------------------------------------------------------------

def _nitro_overcharge(mol, metal_bound: Iterable[int] = ()) -> int:
    """Neutral N([O-])[O-] drawings overstate the ligand's negative charge by one."""
    bound = set(metal_bound)
    return sum(
        1 for a in mol.GetAtoms()
        if a.GetSymbol() == "N" and a.GetFormalCharge() == 0
        and sum(x.GetSymbol() == "O" and x.GetFormalCharge() == -1 and x.GetDegree() == 1
                and x.GetIdx() not in bound
                for x in a.GetNeighbors()) >= 2
    )


def os_from_smiles(smiles: Optional[str], header_charge: int, metal: str) -> Tuple[Optional[int], str]:
    """
    Oxidation state from a tmQM-style SMILES (metal written with its formal
    charge, ligands as closed-shell ions, dative bonds).

    Rules:  OS = bracket charge of the metal + (# neutral H atoms bonded to the
    metal: hydrides are drawn neutral) - (# nitro-kN groups drawn N([O-])[O-],
    which over-charge the ligand by one and inflate the metal by one).
    Accepted only if the SMILES total charge equals the header charge, there is
    exactly one metal, and the result is plausible for the metal.
    Returns (os or None, reason).
    """
    if not smiles:
        return None, "no_smiles"
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception:
        return None, "rdkit_unavailable"
    if mol is None:
        return None, "unparsable"
    metals = [a for a in mol.GetAtoms() if a.GetSymbol() == metal]
    if len(metals) != 1:
        return None, "metal_count"
    m = metals[0]
    total = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if total != int(header_charge):
        return None, "charge_mismatch"
    os_ = m.GetFormalCharge()
    n_hydride = sum(1 for nb in m.GetNeighbors()
                    if nb.GetSymbol() == "H" and nb.GetFormalCharge() == 0)
    # N([O-])[O-] with neutral N (nitro drawn without the N+=O charge separation)
    # over-charges the ligand by one wherever it sits (donor OR aryl substituent);
    # the metal charge is inflated by the same amount to keep the total.
    n_nitro = _nitro_overcharge(mol)
    os_ = os_ + n_hydride - n_nitro
    if not plausible(metal, os_):
        return None, "implausible"
    return os_, "ok"


def _hund_unpaired(n_f: int) -> int:
    return n_f if n_f <= 7 else 14 - n_f


def os_from_header(metal: str, charge: int, spin_mult: int) -> Tuple[Optional[int], str]:
    """
    Datasets without SMILES (LnQM).  Ln(III) if the spin multiplicity equals the
    Hund-rule f^n count for the tripositive ion; otherwise unknown.
    """
    z = _LN_Z.get(metal)
    if z is None:
        return None, "not_ln"
    if spin_mult - 1 == _hund_unpaired(z - 57):
        return 3, "ln3_spin_consistent"
    return None, "ln_spin_inconsistent"


# ---------------------------------------------------------------------------
# Query-time deduction
# ---------------------------------------------------------------------------

def ligand_ionic_charge(smiles: str, donors: Sequence[Sequence[int]], *, model: Optional[str] = None) -> Tuple[Optional[int], str]:
    """
    Ligand charge under the selected query convention.

    Legacy infers ionic charges: formal charge minus radical electrons, with
    all-carbon haptic groups assigned by size (eta3/eta5 -1, eta2/eta4/eta6 0).
    The training model uses the formal charge of the WHOLE ligand as written,
    independent of hapticity. RDKit radicals produced by removing metal bonds
    are not ionic charges (a carbene's two radicals do not make it -2). Anionic
    donors must be written explicitly, e.g. [CH3-], [O-2], [cH-]. The dataset's
    neutral hydride and nitro conventions are shared with extraction.
    Returns (None, reason) if it cannot be decided.
    """
    model = OXIDATION_MODEL if model is None else model
    if model != "training":
        raise ValueError(f"unknown oxidation_model: {model!r} (production model is 'training')")
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None, "rdkit_unavailable"
    if mol is None:
        return None, "unparsable"
    bound = {i for grp in donors for i in grp}
    if any(i < 0 or i >= mol.GetNumAtoms() for i in bound):
        return None, "donor_index_out_of_range"
    formal = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    hydrides = sum(a.GetSymbol() == "H" and a.GetFormalCharge() == 0 and a.GetIdx() in bound
                   for a in mol.GetAtoms())
    return formal - hydrides + _nitro_overcharge(mol, bound), "ok"


def deduce_oxidation_state(
    metal: str,
    net_charge: int,
    ligands: Iterable,          # objects with .smiles and .donors
    *, model: Optional[str] = None,
) -> Tuple[int, str, List[str]]:
    """
    OS = net charge - sum(ligand ionic charges).
    Returns (os, confidence, notes) with confidence in {"high", "low"};
    a low-confidence result is returned as OS_UNKNOWN.
    """
    notes: List[str] = []
    total = 0
    for i, lig in enumerate(ligands):
        z, why = ligand_ionic_charge(lig.smiles, lig.donors, model=model)
        if z is None:
            notes.append(f"ligand {i}: cannot assign ionic charge ({why})")
            return OS_UNKNOWN, "low", notes
        notes.append(f"ligand {i}: {z:+d}")
        total += z
    os_ = int(net_charge) - total
    if not plausible(metal, os_):
        notes.append(f"deduced OS {os_:+d} is implausible for {metal}")
        return OS_UNKNOWN, "low", notes
    return os_, "high", notes


# ---------------------------------------------------------------------------
# Shannon (1976) ionic radii shifts
# ---------------------------------------------------------------------------
_ROMAN = {2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX",
          10: "X", 11: "XI", 12: "XII"}


@lru_cache(maxsize=1)
def _table() -> dict:
    p = Path(__file__).resolve().parent.parent / "data" / "shannon_radii.json"
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)["radii"]


def _spin_label(metal: str, os_: int, spin_mult: int) -> str:
    """'High Spin' / 'Low Spin' from d-count and multiplicity ('-' when not applicable)."""
    g = D_GROUP.get(metal)
    if g is None:
        return "-"
    n = g - os_
    if n < 4 or n > 7:
        return "-"
    unpaired = spin_mult - 1
    hs = n if n <= 5 else 10 - n
    return "High Spin" if unpaired >= hs else "Low Spin"


def ionic_radius(metal: str, os_: int, spin_mult: int, cn: Optional[str]) -> Optional[float]:
    by_os = _table().get(metal, {}).get(str(os_))
    if not by_os:
        return None
    target = int(cn[2:]) if cn and cn.startswith("CN") else 6
    def _n(r):  # roman -> int (SQ variants map to their CN)
        for v, name in _ROMAN.items():
            if r.replace("SQ", "") == name:
                return v
        return None
    cands = sorted(((abs((_n(r) or 99) - target), r) for r in by_os))
    label = _spin_label(metal, os_, spin_mult)
    for _, r in cands:
        spins = by_os[r]
        if label in spins:
            return spins[label]
        if "-" in spins:
            return spins["-"]
        if len(spins) == 1:
            return next(iter(spins.values()))
    return None


def radius_shift(metal: str, ref: Tuple[int, int], query: Tuple[int, int], cn: Optional[str]) -> Optional[float]:
    """r(query) - r(ref) in Å; states are (oxidation_state, spin_multiplicity)."""
    rq = ionic_radius(metal, query[0], query[1], cn)
    rr = ionic_radius(metal, ref[0], ref[1], cn)
    if rq is None or rr is None:
        return None
    return rq - rr
