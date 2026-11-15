"""Data preparation utility for Jacobik.

Converts user structures (XYZ + SMILES, or XYZ + bonds) into the standard tmQM-format
directory structure expected by the tmQM parser and dataset readers:
  <out_dir>/xyz/tmQM_<split>.xyz
  <out_dir>/bo/tmQM_<split>.BO
  <out_dir>/tmQM_y.csv

Provides comprehensive geometric and chemical validation messages.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("prepare_data")

# Known transition metals, lanthanides, and actinides
METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}

# Covalent radii estimates (Pyykko & Atsumi) in Angstroms
COV_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
    "Pt": 1.36, "Co": 1.26, "Fe": 1.32, "Ni": 1.24, "Cu": 1.32,
    "Pd": 1.39, "Ru": 1.46, "Rh": 1.42, "Ir": 1.41, "Os": 1.44,
    "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
}


def parse_xyz(path: Union[str, Path]) -> Tuple[List[str], np.ndarray, str]:
    """Parse an XYZ file into atom symbols, (N, 3) positions, and header comment."""
    lines = Path(path).read_text().splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"XYZ file {path} has fewer than 3 lines")
    n_atoms = int(lines[0])
    comment = lines[1]
    symbols = []
    coords = []
    for i, line in enumerate(lines[2:2 + n_atoms], start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed coordinate line {i} in {path}: {line}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords, dtype=np.float64), comment


def validate_structure(
    symbols: List[str],
    coords: np.ndarray,
    csd_code: str,
    smiles: Optional[str] = None,
) -> Tuple[bool, List[str], Optional[int]]:
    """Validate chemical and geometric consistency of a structure."""
    msgs = []
    ok = True

    # Check 1: Non-empty atoms and coordinates
    if len(symbols) == 0 or len(coords) != len(symbols):
        return False, [f"[{csd_code}] Atom count mismatch or empty coordinates"], None

    # Check 2: Minimum interatomic distance (detect overlapping atoms)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_d = np.min(dist)
    if min_d < 0.65:
        ok = False
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        msgs.append(f"[{csd_code}] ERROR: Atoms {min_idx[0]} ({symbols[min_idx[0]]}) and {min_idx[1]} ({symbols[min_idx[1]]}) overlap (distance {min_d:.3f} A < 0.65 A)")
    else:
        msgs.append(f"[{csd_code}] OK: Minimum distance {min_d:.3f} A is physically valid")

    # Check 3: Identify metal center
    metal_indices = [i for i, s in enumerate(symbols) if s in METALS]
    if len(metal_indices) == 0:
        ok = False
        msgs.append(f"[{csd_code}] ERROR: No transition metal or lanthanide found in atom list: {sorted(set(symbols))}")
        metal_idx = None
    elif len(metal_indices) > 1:
        msgs.append(f"[{csd_code}] WARNING: Multiple metal atoms detected ({[symbols[i] for i in metal_indices]}); using primary metal at index {metal_indices[0]}")
        metal_idx = metal_indices[0]
    else:
        metal_idx = metal_indices[0]
        msgs.append(f"[{csd_code}] OK: Found primary metal center: {symbols[metal_idx]} (atom {metal_idx})")

    # Check 4: SMILES validation
    if smiles:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                ok = False
                msgs.append(f"[{csd_code}] ERROR: RDKit could not parse SMILES: {smiles}")
            else:
                msgs.append(f"[{csd_code}] OK: SMILES is valid ({mol.GetNumHeavyAtoms()} heavy atoms)")
        except ImportError:
            msgs.append(f"[{csd_code}] NOTE: RDKit not available for deep SMILES validation")

    return ok, msgs, metal_idx


def compute_stoichiometry(symbols: List[str]) -> str:
    """Compute standard Hill formula string (C first, then H, then alphabetical)."""
    counts = Counter(symbols)
    parts = []
    for el in ["C", "H"]:
        if el in counts:
            c = counts.pop(el)
            parts.append(f"{el}{c if c > 1 else ''}")
    for el in sorted(counts):
        c = counts[el]
        parts.append(f"{el}{c if c > 1 else ''}")
    return "".join(parts)


def estimate_bonds(
    symbols: List[str],
    coords: np.ndarray,
    metal_idx: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """Estimate chemical connectivity and bond orders using covalent radii."""
    n = len(symbols)
    bonds = []
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    for i in range(n):
        r_i = COV_RADII.get(symbols[i], 1.20)
        for j in range(i + 1, n):
            r_j = COV_RADII.get(symbols[j], 1.20)
            d = dist[i, j]
            r_sum = r_i + r_j
            # Coordination bond to metal
            if metal_idx in (i, j):
                if d <= r_sum * 1.35 and d >= 0.8:
                    bo = max(0.2, min(1.0, 1.0 - (d - r_sum * 0.95) / (r_sum * 0.4)))
                    bonds.append((i + 1, j + 1, round(bo, 4)))
            else:
                # Organic / ligand bond
                if d <= r_sum * 1.25 and d >= 0.8:
                    bo = 1.0
                    if d < r_sum * 0.95 and symbols[i] in ("C", "N", "O") and symbols[j] in ("C", "N", "O"):
                        bo = 2.0
                    bonds.append((i + 1, j + 1, round(bo, 4)))
    return bonds


def prepare_complex(
    csd_code: str,
    symbols: List[str],
    coords: np.ndarray,
    smiles: str,
    charge: int = 0,
    spin: int = 0,
    bonds: Optional[List[Tuple[int, int, float]]] = None,
) -> Tuple[str, str, str]:
    """Format single complex into XYZ block, BO block, and CSV row."""
    stoich = compute_stoichiometry(symbols)
    valid, msgs, metal_idx = validate_structure(symbols, coords, csd_code, smiles)
    for m in msgs:
        log.info(m)
    if not valid:
        raise ValueError(f"Structure validation failed for {csd_code}:\n" + "\n".join(msgs))

    if bonds is None:
        bonds = estimate_bonds(symbols, coords, metal_idx)

    # MND: coordination number around metal
    mnd = 0
    if metal_idx is not None:
        m1 = metal_idx + 1
        mnd = sum(1 for a, b, _ in bonds if a == m1 or b == m1)

    # XYZ block
    xyz_header = (
        f"CSD_code = {csd_code} | q = {charge} | S = {spin} | "
        f"Stoichiometry = {stoich} | MND = {mnd} | User-Prepared"
    )
    xyz_lines = [str(len(symbols)), xyz_header]
    for s, (x, y, z) in zip(symbols, coords):
        xyz_lines.append(f"{s:<3s} {x:12.6f} {y:12.6f} {z:12.6f}")
    xyz_block = "\n".join(xyz_lines)

    # BO block
    bo_header = f"CSD_code = {csd_code} | User-Prepared"
    bo_lines = [bo_header]
    for a, b, bo in bonds:
        bo_lines.append(f"{a:4d} {b:4d} {bo:8.4f}")
    bo_block = "\n".join(bo_lines)

    # CSV row
    csv_row = f"{csd_code};{smiles}"
    return xyz_block, bo_block, csv_row


def prepare_dataset(
    items: Sequence[dict],
    out_dir: Union[str, Path],
    split_name: str = "custom",
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Build a complete tmQM-style directory layout for a list of complex dictionaries."""
    out_dir = Path(out_dir)
    xyz_dir = out_dir / "xyz"
    bo_dir = out_dir / "bo"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    bo_dir.mkdir(parents=True, exist_ok=True)

    all_xyz, all_bo, all_csv = [], [], ["CSD_code;SMILES"]
    for it in items:
        code = it["csd_code"]
        smiles = it["smiles"]
        charge = it.get("charge", 0)
        spin = it.get("spin", 0)
        if "xyz_file" in it:
            xyz_p = Path(it["xyz_file"])
            if not xyz_p.is_absolute() and base_dir is not None:
                xyz_p = Path(base_dir) / xyz_p
            symbols, coords, _ = parse_xyz(xyz_p)
        else:
            symbols = it["symbols"]
            coords = np.asarray(it["coords"], dtype=np.float64)
        bonds = it.get("bonds", None)
        xyz_b, bo_b, csv_r = prepare_complex(code, symbols, coords, smiles, charge, spin, bonds)
        all_xyz.append(xyz_b)
        all_bo.append(bo_b)
        all_csv.append(csv_r)

    xyz_path = xyz_dir / f"tmQM_{split_name}.xyz"
    bo_path = bo_dir / f"tmQM_{split_name}.BO"
    csv_path = out_dir / "tmQM_y.csv"

    xyz_path.write_text("\n\n".join(all_xyz) + "\n")
    bo_path.write_text("\n\n".join(all_bo) + "\n")
    csv_path.write_text("\n".join(all_csv) + "\n")
    log.info("Prepared %d complexes into %s (split %s)", len(items), out_dir, split_name)
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Prepare user structures into tmQM format for Jacobik.")
    p.add_argument("--input", required=True, help="Input JSON manifest or XYZ file.")
    p.add_argument("--smiles", default=None, help="SMILES string (if single XYZ input).")
    p.add_argument("--code", default=None, help="CSD code / ID (if single XYZ input).")
    p.add_argument("--out-dir", required=True, help="Target directory for prepared dataset.")
    p.add_argument("--split", default="custom", help="Split label (default: custom).")
    args = p.parse_args()

    inp = Path(args.input)
    if inp.suffix.lower() == ".json":
        items = json.loads(inp.read_text())
        # resolve relative xyz paths relative to JSON file
        for it in items:
            if "xyz_file" in it:
                it_path = Path(it["xyz_file"])
                if not it_path.is_absolute():
                    it["xyz_file"] = str(inp.parent / it_path)
        prepare_dataset(items, args.out_dir, args.split)
    elif inp.suffix.lower() == ".xyz":
        if not args.smiles:
            raise ValueError("--smiles is required when providing a single XYZ file")
        code = args.code or inp.stem
        symbols, coords, _ = parse_xyz(inp)
        items = [{"csd_code": code, "smiles": args.smiles, "symbols": symbols, "coords": coords}]
        prepare_dataset(items, args.out_dir, args.split)
    else:
        raise ValueError(f"Unsupported input format: {inp}")


if __name__ == "__main__":
    main()
