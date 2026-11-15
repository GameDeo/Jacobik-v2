"""Fit-table input sanity rules shared by the reproducible DB builder."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional

import pandas as pd

from jacobik.core import profile as _profile

OS_SANITY = _profile.get("os_sanity")
FIT_EXCLUDE_LOCKED = _profile.get("fit_exclude_locked")
STRUCTURE_SANITY = _profile.get("structure_sanity")

#: A carbon with exactly two neighbours bent below this angle has lost a substituent (sp carbons are linear).
BENT_TWO_COORDINATE_DEG = 150.0
#: A parent whose carbon framework is this fraction under-bonded is a broken structure (e.g. a CSD entry without
#: hydrogens that tmQM optimised anyway: JENDAH); 0.25 leaves ~0.05% of the corpus, none of the clean parents.
UNDERBONDED_FRACTION_MAX = 0.25
UNDERBONDED_MIN_CARBONS = 6

# Formal metal-charge limits used by the input census.  This is deliberately
# stricter than the oxidation-state limits: tmQM SMILES store an atom formal
# charge, while OS is inferred from the complete molecular charge balance.
METAL_FORMAL_MAX = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    "Sc": 3, "Y": 3, "La": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5,
    "Cr": 6, "Mo": 6, "W": 6, "Mn": 7, "Tc": 7, "Re": 7, "Fe": 6, "Ru": 8, "Os": 8,
    "Co": 5, "Rh": 6, "Ir": 6, "Ni": 4, "Pd": 4, "Pt": 6, "Cu": 4, "Ag": 3, "Au": 5,
    "Zn": 2, "Cd": 2, "Hg": 2, "Al": 3, "Ga": 3, "In": 3, "Tl": 3, "Sn": 4, "Pb": 4,
    "Bi": 5, "Sb": 5,
}


def metal_formal_charge(smiles: str | None, metal: str) -> int | None:
    """Return the source SMILES formal charge on the unique target metal."""
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


def charge_is_plausible(metal: str, charge: int | None) -> bool:
    if charge is None:
        return True  # missing/unparseable source labels are reported separately
    return -2 <= charge <= METAL_FORMAL_MAX.get(metal, 4)


def underbonded_carbon_fraction(symbols, positions) -> float:
    """Fraction of carbons with exactly two neighbours (covalent-radius sum x 1.25, metals included) at an angle below 150 degrees."""
    import numpy as np
    from scipy.spatial.distance import cdist

    from jacobik.core.chemistry import covalent_radius as _covalent_radius

    sym = np.asarray(symbols)
    carbons = np.flatnonzero(sym == "C")
    if len(carbons) < UNDERBONDED_MIN_CARBONS:
        return 0.0
    pos = np.asarray(positions, float)
    rad = np.array([_covalent_radius(x) for x in sym])
    dist = cdist(pos[carbons], pos)
    near = (dist > 0.01) & (dist < 1.25 * (rad[carbons][:, None] + rad[None, :]))
    bent = 0
    for row, i in zip(near, carbons):
        idx = np.flatnonzero(row)
        if len(idx) == 2:
            a, b = pos[idx[0]] - pos[i], pos[idx[1]] - pos[i]
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            bent += int(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) < BENT_TWO_COORDINATE_DEG)
    return bent / len(carbons)


def census_rows(complexes: Iterable) -> list[dict]:
    """Summarize metal formal charge and the OS label used by the fitter."""
    rows = []
    for mol in complexes:
        charge = getattr(mol, "metal_charge", None)
        rows.append({
            "csd_code": mol.csd_code,
            "metal": mol.metal_symbol,
            "metal_formal_charge": charge,
            "formal_charge_range": [-2, METAL_FORMAL_MAX.get(mol.metal_symbol, 4)],
            "charge_plausible": charge_is_plausible(mol.metal_symbol, charge),
            "oxidation_state_used": getattr(mol, "oxidation_state", None),
            "oxidation_source": getattr(mol, "oxidation_source", None),
        })
    return rows


def _smiles_by_code(csv_path: Path) -> dict:
    """CSD_code -> SMILES (the only two columns read; per process cache); {} when the file is missing."""
    from jacobik.core.env import tmqm_smiles

    return tmqm_smiles(csv_path.parent) if csv_path.exists() else {}


def census_files(ref_dir: Path) -> list[tuple[str, Path]]:
    """[(dataset, xyz shard path)] in the census order: tmQM then lnQM, shard names sorted."""
    out = []
    for dataset in ("tmQM", "lnQM"):
        out.extend((dataset, p) for p in sorted((Path(ref_dir) / dataset / "xyz").glob("tmQM_*.xyz")))
    return out


def census_file(ref_dir: Path, dataset: str, xyz_path: Path) -> list[dict]:
    """Census rows of one xyz shard (no QM property column is read: the CSV reader asks for CSD_code and SMILES only,
    the XYZ parser for the block atom count, source charge/multiplicity and element symbols)."""
    from jacobik.core.chemistry import D_GROUP
    from jacobik.core.chemistry import _LN_Z, os_from_header, os_from_smiles

    metals = set(D_GROUP) | set(_LN_Z) | {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"}
    smiles_by_code = _smiles_by_code(Path(ref_dir) / dataset / "tmQM_y.csv")
    out = []
    with Path(xyz_path).open("r", encoding="utf-8") as fh:
        while True:
            first = fh.readline()
            if not first:
                break
            try:
                n_atoms = int(first.strip())
            except ValueError:
                continue
            header = fh.readline()
            atom_symbols, atom_xyz = [], []
            for _ in range(n_atoms):
                line = fh.readline()
                if not line:
                    break
                fields = line.split()
                if fields:
                    atom_symbols.append(fields[0])
                    atom_xyz.append([float(v) for v in fields[1:4]])
            m_code = re.search(r"CSD_code\s*=\s*([^|]+)", header)
            m_charge = re.search(r"\bq\s*=\s*(-?\d+)", header)
            m_spin = re.search(r"\bS\s*=\s*(-?\d+)", header)
            code = m_code.group(1).strip() if m_code else ""
            charge = int(m_charge.group(1)) if m_charge else 0
            spin = int(m_spin.group(1)) + 1 if m_spin else 1  # parser converts S to multiplicity
            found = [s for s in atom_symbols if s in metals]
            metal = found[0] if found else "unknown"
            smiles = smiles_by_code.get(code)
            metal_q = metal_formal_charge(smiles, metal)
            os_val, os_reason = os_from_smiles(smiles, charge, metal)
            if os_val is None and not smiles:
                os_val, os_reason = os_from_header(metal, charge, spin)
            underbonded = underbonded_carbon_fraction(atom_symbols, atom_xyz)
            out.append({
                "csd_code": code, "dataset": dataset, "metal": metal,
                "underbonded_carbon_fraction": underbonded, "structure_plausible": underbonded < UNDERBONDED_FRACTION_MAX,
                "metal_formal_charge": metal_q,
                "formal_charge_range": [-2, METAL_FORMAL_MAX.get(metal, 4)],
                "charge_plausible": charge_is_plausible(metal, metal_q),
                "oxidation_state_used": os_val, "oxidation_source": os_reason,
            })
    return out


def excluded_charge_codes(census: Iterable[dict]) -> set[str]:
    return {str(r["csd_code"]) for r in census if not r["charge_plausible"]}


def excluded_structure_codes(census: Iterable[dict]) -> set[str]:
    return {str(r["csd_code"]) for r in census if not r.get("structure_plausible", True)}


def locked_codes(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    return set(data.get("reserve_all_codes", ())) | set(data.get("motion_validation_codes", ()))


def filter_fit_frame(df: pd.DataFrame, *, charge_bad: set[str] | None = None,
                     locked: set[str] | None = None, os_sanity: bool | None = None,
                     exclude_locked: bool | None = None) -> pd.DataFrame:
    """Apply the registered whole-parent exclusions to an observation frame."""
    use_os_sanity = OS_SANITY if os_sanity is None else os_sanity
    use_locked = FIT_EXCLUDE_LOCKED if exclude_locked is None else exclude_locked
    excluded = set(charge_bad or ()) if use_os_sanity else set()
    if use_locked:
        excluded.update(locked or ())
    if not excluded or "csd_code" not in df.columns:
        return df
    return df.loc[~df["csd_code"].astype(str).isin(excluded)].copy()


# ---------------------------------------------------------------------------
# Observation identity & eligibility ledger
# ---------------------------------------------------------------------------


STABLE_ANCESTOR_SAMPLING = _profile.get("stable_ancestor_sampling")


def observation_id(dataset, code, quantity, original_key, occurrence):
    identity = ["observation.1", dataset, code, quantity, original_key, occurrence]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def priority(bucket, identity):
    return int.from_bytes(hashlib.sha256(
        json.dumps(["ancestor.1", *bucket, identity], separators=(",", ":")).encode()
    ).digest(), "big")


POLICY_VERSION = "eligibility.1"
LEDGER_VERSION = "extraction_ledger.1"
REVIEW_JSON = Path(__file__).with_name("data") / "source_review.json"

#: event kind -> (disposition, meaning). Dispositions: correction (source reading changed), preserved (evidence kept),
#: review (uncertain; retained, listed), info.
EVENT_KINDS: Dict[str, tuple] = {
    "hydrogen_aware_donor_mapping": ("correction", "heavy-atom alignment re-done with attached H counts"),
    "hydrogen_alignment_ambiguous": ("review", "H-aware alignment inconsistent; element-only alignment retained"),
    "local_donor_identity": ("correction", "donor moved to a supported sibling atom of the same terminal motif"),
    "legacy_supported_identity_preserved": ("preserved", "prior supported correspondence kept"),
    "legacy_supported_identity_collision": ("review", "prior supported correspondence collides with an occupied donor"),
    "legacy_contact_identity_preserved": ("preserved", "prior rule's best-contact atom kept over an unsupported alignment image"),
    "legacy_supported_donor_restored": ("preserved", "supported contact of the prior rule kept while a declaration had no donor"),
    "donor_assignment_changed": ("correction", "prototype: declaration re-assigned within its symmetry class by evidence"),
    "donor_assignment_transferred": ("review", "prototype: declaration without a bound class atom moved to an unclaimed supported contact of its ligand"),
    "supported_contact_not_declared": ("review", "prototype: supported metal contact that no source declaration accounts for"),
    "hapto_face_unsupported": ("review", "face of three or more atoms with no metal bond order on any member: not a haptic interaction"),
    "donor_assignment_unrepresented": ("review", "prototype: declaration with no class atom in bonding range"),
    "remote_declared_donor_unrepresented": ("review", "declared donor far beyond bonding range given no donor site"),
    "weak_declared_identity_retained": ("review", "declared donor with weak contact evidence retained"),
    "rejected_face_kappa_preservation": ("correction", "supported kappa neighbour kept after a false C-X face was rejected"),
}
#: dispositions that need a human look before the corpus is fitted
REVIEW_DISPOSITIONS = ("review",)

#: kinds (or other registered factors) whose parents a fit may exclude. Empty by policy; extending it requires a
#: separately attributed arm and a new POLICY_VERSION.
EXCLUSION_FACTORS: frozenset = frozenset()

LEDGER_COLUMNS = ["ledger_version", "csd_code", "dataset", "metal", "event_idx", "kind", "disposition",
                  "before", "after", "detail", "dependent"]


def disposition(kind: str) -> str:
    return EVENT_KINDS.get(kind, ("unclassified", ""))[0]


def load_review(path: Path = REVIEW_JSON) -> dict:
    return json.loads(Path(path).read_text())


def eligibility_codes(ledger, factors: Optional[Iterable[str]] = None) -> set:
    """Parent codes excluded by the registered ``factors`` (default: none). Unregistered factors raise."""
    factors = frozenset(factors or ())
    bad = factors - EXCLUSION_FACTORS
    if bad:
        raise ValueError(f"unregistered eligibility factors {sorted(bad)}; register in fit.sanity with attribution")
    return {str(r["csd_code"]) for r in _rows(ledger) if r["kind"] in factors}


def _rows(ledger) -> List[Mapping]:
    return ledger.to_dict("records") if hasattr(ledger, "to_dict") else list(ledger)


def reason_counts(ledger, review: Optional[dict] = None) -> dict:
    """Counts by event kind and disposition, affected parents, and dependent rows per quantity."""
    rows = _rows(ledger)
    kinds = Counter(r["kind"] for r in rows)
    disp = Counter(r["disposition"] for r in rows)
    parents = {str(r["csd_code"]) for r in rows}
    by_disp_parents: Dict[str, set] = {}
    dep: Counter = Counter()
    seen = set()
    for r in rows:
        by_disp_parents.setdefault(r["disposition"], set()).add(str(r["csd_code"]))
        if r["csd_code"] in seen:
            continue
        seen.add(r["csd_code"])
        d = r["dependent"]
        d = json.loads(d) if isinstance(d, str) else (d or {})
        for q, v in d.items():
            dep[q] += int(v.get("n", 0))
    unclassified = sorted(k for k in kinds if k not in EVENT_KINDS)
    out = {"policy_version": POLICY_VERSION, "ledger_version": LEDGER_VERSION, "events": len(rows),
           "events_by_kind": dict(sorted(kinds.items())), "events_by_disposition": dict(sorted(disp.items())),
           "parents_with_events": len(parents),
           "parents_by_disposition": {k: len(v) for k, v in sorted(by_disp_parents.items())},
           "dependent_rows_in_event_parents": dict(sorted(dep.items())),
           "unclassified_kinds": unclassified, "exclusion_factors": sorted(EXCLUSION_FACTORS),
           "excluded_parents": len(eligibility_codes(rows, EXCLUSION_FACTORS))}
    if review is not None:
        out["review_ledger"] = {"version": review.get("version"),
                                "unresolved": sorted(e["code"] for e in review["entries"] if e["status"] == "unresolved"),
                                "decided": sorted(e["code"] for e in review["entries"] if e["status"] == "decided"),
                                "unresolved_present_in_ledger": sorted(
                                    {e["code"] for e in review["entries"] if e["status"] == "unresolved"} & parents)}
    return out
