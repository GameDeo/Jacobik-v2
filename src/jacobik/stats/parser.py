"""
parser.py
=========
Streams and bulk-loads the raw tmQM dataset files into TMComplex objects.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np

from jacobik.core.types import Atom, Bond, TMComplex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bond orders below this floor are ignored entirely — they represent
# negligible through-space interactions, not real bonds.
# Value matches the floor already documented in Bond.is_coordination.
BO_FLOOR: float = 0.08

# Expected CSV column names (order-independent via DictReader)
_CSV_FLOAT_COLS = (
    "Electronic_E",
    "Dispersion_E",
    "Dipole_M",
    "Metal_q",
    "HL_Gap",
    "HOMO_Energy",
    "LUMO_Energy",
    "Polarizability",
)


# ---------------------------------------------------------------------------
# Intermediate raw structures (internal to this module)
# ---------------------------------------------------------------------------


@dataclass
class _RawMolecule:
    """
    Everything extracted from a single .xyz + .BO block pair, before
    mapping.py synthesises it into a TMComplex.
    """

    csd_code: str
    charge: int
    spin_state: int
    stoichiometry: str
    mnd: int
    csd_years: str
    atoms: List[Atom]  # positions + symbols; total_valence filled from .BO
    bonds: List[Bond]  # raw continuous BOs, all pairs above BO_FLOOR
    smiles: Optional[str]  # from CSV; None if absent


@dataclass
class _CSVRow:
    """One row from tmQM_y.csv."""

    csd_code: str
    electronic_energy: Optional[float]
    dispersion_energy: Optional[float]
    dipole_moment: Optional[float]
    metal_charge: Optional[float]
    hl_gap: Optional[float]
    homo_energy: Optional[float]
    lumo_energy: Optional[float]
    polarizability: Optional[float]
    csd_years: str
    smiles: Optional[str]


# ---------------------------------------------------------------------------
# .xyz parsing
# ---------------------------------------------------------------------------


def _parse_xyz_comment(comment: str) -> Tuple[str, int, int, str, int, str]:
    """
    Parse the second line of an .xyz block.


    Returns
    -------
    csd_code, charge, spin_state, stoichiometry, mnd, csd_years

    """
    fields: Dict[str, str] = {}
    csd_years = ""
    for part in comment.split("|"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            fields[k.strip()] = v.strip()
        elif "CSD" in part:
            # e.g. "1965-2020 CSD" or "2020-2024 CSD"
            csd_years = part.replace("CSD", "").strip()

    try:
        return (
            fields["CSD_code"],
            int(fields["q"]),
            int(fields["S"]),
            fields["Stoichiometry"],
            int(fields["MND"]),
            csd_years,
        )
    except KeyError as exc:
        raise ValueError(f"Malformed .xyz comment line: {comment!r}") from exc


def _iter_xyz_blocks(path: Path) -> Iterator[Tuple[str, List[str]]]:
    """
    Yield (comment_line, atom_lines) for each molecule block in an .xyz file.
    Reads the whole file lazily line by line.
    """
    with open(path, "r") as fh:
        while True:
            # --- atom count line ---
            count_line = fh.readline()
            if not count_line:
                break  # EOF
            count_line = count_line.strip()
            if not count_line:
                continue  # skip stray blank lines between blocks
            try:
                n_atoms = int(count_line)
            except ValueError:
                log.warning("Unexpected non-integer count line in %s: %r", path, count_line)
                continue

            # --- comment line ---
            comment = fh.readline()
            if not comment:
                raise EOFError(f"Truncated .xyz file at block starting with count {n_atoms}")
            comment = comment.strip()

            # --- atom lines ---
            atom_lines: List[str] = []
            for _ in range(n_atoms):
                line = fh.readline()
                if not line:
                    raise EOFError(f"Truncated .xyz file in block {comment!r}")
                atom_lines.append(line.strip())

            yield comment, atom_lines


def _build_atoms_from_xyz(atom_lines: List[str]) -> List[Atom]:
    """
    Build Atom objects from the atom lines of an .xyz block.
    Indices are 1-based (idx = line position + 1).
    total_valence is left at 0.0; it is filled by _apply_bo_block().
    """
    atoms: List[Atom] = []
    for i, line in enumerate(atom_lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line {i}: {line!r}")
        symbol = parts[0]
        try:
            position = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64
            )
        except ValueError as exc:
            raise ValueError(f"Non-numeric coordinate in atom line {i}: {line!r}") from exc
        atoms.append(Atom(idx=i, symbol=symbol, position=position))
    return atoms


# ---------------------------------------------------------------------------
# .BO parsing
# ---------------------------------------------------------------------------


def _parse_bo_header(line: str) -> str:
    """
    Extract CSD_code from a .BO header line.

    Format: "CSD_code = XXXX | YYYY-YYYY CSD"
    """
    for part in line.split("|"):
        part = part.strip()
        if part.startswith("CSD_code"):
            return part.split("=", 1)[1].strip()
    raise ValueError(f"Cannot extract CSD_code from .BO header: {line!r}")


def _iter_bo_blocks(path: Path) -> Iterator[Tuple[str, List[str]]]:
    """
    Yield (csd_code, atom_lines) for each molecule block in a .BO file.
    Blocks are separated by blank lines; each starts with a CSD_code header.
    """
    with open(path, "r") as fh:
        current_code: Optional[str] = None
        current_lines: List[str] = []

        for raw_line in fh:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                # Blank line = end of current block
                if current_code is not None and current_lines:
                    yield current_code, current_lines
                current_code = None
                current_lines = []
                continue

            if stripped.startswith("CSD_code"):
                # Flush any pending block (handles files without trailing blank line)
                if current_code is not None and current_lines:
                    yield current_code, current_lines
                current_code = _parse_bo_header(stripped)
                current_lines = []
            else:
                current_lines.append(stripped)

        # Flush the final block (file may not end with a blank line)
        if current_code is not None and current_lines:
            yield current_code, current_lines


def _apply_bo_block(
    atoms: List[Atom],
    bo_lines: List[str],
) -> List[Bond]:
    """
    Parse .BO atom lines and:
      1. Fill Atom.total_valence for each atom.
      2. Build the deduplicated Bond list (raw continuous BOs, above BO_FLOOR).


    Parameters
    ----------
    atoms    : List[Atom] already built from .xyz (indices 1-based)
    bo_lines : raw atom lines from the .BO block (no header, no blanks)

    Returns
    -------
    List[Bond]

    """
    atom_map: Dict[int, Atom] = {a.idx: a for a in atoms}
    seen: Dict[Tuple[int, int], float] = {}  # (min_i, max_j) -> bo

    for line in bo_lines:
        parts = line.split()
        if len(parts) < 3:
            continue

        try:
            atom_idx = int(parts[0])
            total_val = float(parts[2])
        except ValueError:
            log.debug("Skipping malformed .BO line: %r", line)
            continue

        if atom_idx in atom_map:
            atom_map[atom_idx].total_valence = total_val

        # Parse neighbour triplets: sym  idx  bo
        i = 3
        while i + 2 < len(parts):
            try:
                nb_idx = int(parts[i + 1])
                bo = float(parts[i + 2])
            except ValueError:
                i += 3
                continue

            if bo >= BO_FLOOR:
                key = (min(atom_idx, nb_idx), max(atom_idx, nb_idx))
                # Keep the first occurrence (neighbours are sorted descending,
                # so the first time we see a pair gives the correct BO)
                if key not in seen:
                    seen[key] = bo

            i += 3

    return [Bond(i=k[0], j=k[1], order=v) for k, v in seen.items()]


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> Dict[str, _CSVRow]:
    """
    Load tmQM_y.csv into a dict keyed by CSD_code.
    Float fields that are empty or non-numeric are stored as None.
    SMILES that are empty strings are stored as None.
    """
    index: Dict[str, _CSVRow] = {}

    def _opt_float(s: str) -> Optional[float]:
        s = s.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            code = row["CSD_code"].strip()
            smiles_raw = row.get("SMILES", "").strip()
            index[code] = _CSVRow(
                csd_code=code,
                electronic_energy=_opt_float(row.get("Electronic_E", "")),
                dispersion_energy=_opt_float(row.get("Dispersion_E", "")),
                dipole_moment=_opt_float(row.get("Dipole_M", "")),
                metal_charge=_opt_float(row.get("Metal_q", "")),
                hl_gap=_opt_float(row.get("HL_Gap", "")),
                homo_energy=_opt_float(row.get("HOMO_Energy", "")),
                lumo_energy=_opt_float(row.get("LUMO_Energy", "")),
                polarizability=_opt_float(row.get("Polarizability", "")),
                csd_years=row.get("CSD_years", "").strip(),
                smiles=smiles_raw if smiles_raw else None,
            )

    log.info("Loaded %d CSV rows from %s", len(index), path)
    return index


# ---------------------------------------------------------------------------
# Raw molecule assembly
# ---------------------------------------------------------------------------


def _assemble_raw(
    xyz_comment: str,
    xyz_atom_lines: List[str],
    bo_csd_code: str,
    bo_lines: List[str],
    csv_row: Optional[_CSVRow],
) -> _RawMolecule:
    """
    Combine one .xyz block and one .BO block into a _RawMolecule.
    Validates that both blocks refer to the same CSD_code.
    """
    csd_code, charge, spin_state, stoich, mnd, csd_years = _parse_xyz_comment(xyz_comment)

    if csd_code != bo_csd_code:
        raise ValueError(
            f"CSD_code mismatch: .xyz has {csd_code!r}, .BO has {bo_csd_code!r}. "
            f"Files may be out of sync."
        )

    atoms = _build_atoms_from_xyz(xyz_atom_lines)
    bonds = _apply_bo_block(atoms, bo_lines)

    smiles: Optional[str] = None
    if csv_row is not None:
        smiles = csv_row.smiles

    return _RawMolecule(
        csd_code=csd_code,
        charge=charge,
        spin_state=spin_state,
        stoichiometry=stoich,
        mnd=mnd,
        csd_years=csd_years,
        atoms=atoms,
        bonds=bonds,
        smiles=smiles,
    )


# ---------------------------------------------------------------------------
# QM property attachment
# ---------------------------------------------------------------------------


def _attach_qm_properties(mol: TMComplex, row: _CSVRow) -> None:
    """Write QM scalar fields from a CSV row onto an already-assembled TMComplex."""
    mol.electronic_energy = row.electronic_energy
    mol.dispersion_energy = row.dispersion_energy
    mol.dipole_moment = row.dipole_moment
    mol.metal_charge = row.metal_charge
    mol.hl_gap = row.hl_gap
    mol.homo_energy = row.homo_energy
    mol.lumo_energy = row.lumo_energy
    mol.polarizability = row.polarizability
    mol.smiles = row.smiles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TMQMParser:
    """
    Parses the tmQM dataset for a given split and assembles TMComplex objects.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing tmQM_X1.xyz, tmQM_X1.BO, ..., tmQM_y.csv
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._csv_index: Optional[Dict[str, _CSVRow]] = None

    # ------------------------------------------------------------------
    # CSV index (lazy, loaded once)
    # ------------------------------------------------------------------

    @property
    def csv_index(self) -> Dict[str, _CSVRow]:
        if self._csv_index is None:
            csv_path = self.data_dir / "tmQM_y.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            self._csv_index = _load_csv(csv_path)
        return self._csv_index

    # ------------------------------------------------------------------
    # Core streaming interface
    # ------------------------------------------------------------------

    def stream_split(
        self,
        split: str,
        limit: Optional[int] = None,
    ) -> Generator[TMComplex, None, None]:
        """
        Yield TMComplex objects for every molecule in a split, in file order.


        Parameters
        ----------
        splits : str
            Split name, e.g. "X1" or "X2".  Resolves to
            <data_dir>/xyz/tmQM_<split>.xyz and <data_dir>/bo/tmQM_<split>.BO
        limit : int, optional
            Stop after yielding this many molecules (useful for development).

        """
        # Import here to avoid circular imports at module load time
        from jacobik.stats.mapping import assemble

        xyz_path = self.data_dir / "xyz" / f"tmQM_{split}.xyz"
        bo_path = self.data_dir / "bo" / f"tmQM_{split}.BO"

        if not xyz_path.exists():
            raise FileNotFoundError(f".xyz file not found: {xyz_path}")
        if not bo_path.exists():
            raise FileNotFoundError(f".BO file not found: {bo_path}")

        xyz_iter = _iter_xyz_blocks(xyz_path)
        bo_iter = _iter_bo_blocks(bo_path)

        count = 0
        for (xyz_comment, xyz_atom_lines), (bo_code, bo_lines) in zip(xyz_iter, bo_iter):
            try:
                csv_row = self.csv_index.get(bo_code)
                raw = _assemble_raw(
                    xyz_comment,
                    xyz_atom_lines,
                    bo_code,
                    bo_lines,
                    csv_row,
                )
                mol = assemble(raw)
                if csv_row is not None:
                    _attach_qm_properties(mol, csv_row)
            except Exception as exc:
                log.error("Failed to parse molecule %r: %s", bo_code, exc, exc_info=True)
                continue

            yield mol
            count += 1
            if limit is not None and count >= limit:
                break

    # ------------------------------------------------------------------
    # Bulk loading interface (used by ComplexVisualizer)
    # ------------------------------------------------------------------

    def load_split(
        self,
        split: str,
        limit: Optional[int] = None,
    ) -> Dict[str, TMComplex]:
        """
        Load up to `limit` molecules from `split` into an ordered dict.

        Returns
        -------
        dict[csd_code -> TMComplex], in file order.
        """
        result: Dict[str, TMComplex] = {}
        for mol in self.stream_split(split, limit=limit):
            result[mol.csd_code] = mol
        log.info("Loaded %d complexes from split %s", len(result), split)
        return result

    # ------------------------------------------------------------------
    # Convenience: iterate a list of specific CSD codes
    # ------------------------------------------------------------------

    def load_codes(
        self,
        codes: List[str],
        split: str,
    ) -> Dict[str, TMComplex]:
        """
        Load only the specified CSD codes from a split.
        Stops streaming as soon as all requested codes have been found.
        Raises KeyError if a requested code is not found in the split.
        """
        wanted = set(codes)
        result: Dict[str, TMComplex] = {}
        for mol in self.stream_split(split):
            if mol.csd_code in wanted:
                result[mol.csd_code] = mol
                wanted.discard(mol.csd_code)
            if not wanted:
                break
        if wanted:
            raise KeyError(f"CSD codes not found in split {split!r}: {sorted(wanted)}")
        return result
