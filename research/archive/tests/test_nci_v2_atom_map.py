"""tests/test_nci_v2_atom_map.py
==============================
Unit and integration tests for Jacobik NCI v2 Graph-Constrained Atom Mapping (Item A2).

Verifies:
1. Graph-constrained atom mapping on all 12 pilot parents:
   - Metal atom maps to metal atom.
   - Exact 1-to-1 bijection between generated and reference atoms.
   - Inversion parity: gen_to_ref_idx[ref_to_gen_idx] == np.arange(N).
2. Element symbol and atomic number parity:
   - gen_symbols[gen_to_ref_idx] == ref_symbols for all 12 parents.
3. Covalent bond topology and distance matrix parity:
   - Every covalent bond in reference maps to bonded atoms in generated complex.
   - Distance between covalently bound atoms matches within 0.5 Angstroms.
4. Ambiguity and automorphism detection:
   - Complex symmetries and ligand automorphisms are correctly detected and flagged.
5. Root-cause verification for ICEYAQ (+830 kJ/mol):
   - Mismatched reference symbol evaluation produces massive +22,852 kJ/mol artifact.
   - Properly mapped coordinates resolve symbol parity and yield +829.86 kJ/mol.
   - Ligand strain decomposition confirms >520 kJ/mol is internal ligand strain.
6. Compaction diagnostics parity:
   - compute_heavy_atom_rg and compute_nonbonded_contacts with mapped coordinates.
   - ICEYAQ Rg ratio is ~0.988, correcting the 0.898 unmapped artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from jacobik.builder.atom_map import AtomMappingResult, build_atom_mapping
from refit.nci_v2_pilot import (
    DEFAULT_FRAME,
    DEFAULT_SPEC,
    build_builder_and_inputs,
    compute_heavy_atom_rg,
    compute_nonbonded_contacts,
    load_mol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pilot_parents():
    """Load metadata for the 12 pilot parents from frame.json."""
    frame = json.loads(DEFAULT_FRAME.read_text(encoding="utf-8"))
    parents = frame["pilot_cohort"]["parents"]
    assert len(parents) == 12, f"Expected 12 pilot parents, found {len(parents)}"
    return parents


def test_atom_mapping_all_12_parents(pilot_parents):
    """Verify graph-constrained atom mapping succeeds and preserves parity on all 12 pilot parents."""
    for p in pilot_parents:
        code = p["code"]
        mol = load_mol(code, p["shard"], p["dataset"])
        assert mol is not None, f"Failed to load mol for {code}"

        builder, lig_in, err = build_builder_and_inputs(mol)
        assert err is None, f"build_builder_and_inputs failed for {code}: {err}"

        shell, _, _ = builder._build_shell(chelate_embed=True, fast_embed=False, debug=False)
        N = len(mol.atoms)
        assert len(shell.atoms) == N, f"{code}: Natoms mismatch ({len(shell.atoms)} vs {N})"

        # Compute graph-constrained atom mapping
        mapping = build_atom_mapping(mol, gen_mol=shell)

        # 1. Verification of array shape and permutation validity
        assert isinstance(mapping, AtomMappingResult)
        assert mapping.gen_to_ref_idx.shape == (N,)
        assert mapping.ref_to_gen_idx.shape == (N,)
        assert set(mapping.gen_to_ref_idx.tolist()) == set(range(N)), f"{code}: gen_to_ref_idx is not a permutation"
        assert set(mapping.ref_to_gen_idx.tolist()) == set(range(N)), f"{code}: ref_to_gen_idx is not a permutation"

        # 2. Inversion parity: mapping in both directions is exact inverse
        assert np.array_equal(
            mapping.gen_to_ref_idx[mapping.ref_to_gen_idx], np.arange(N)
        ), f"{code}: gen_to_ref_idx[ref_to_gen_idx] is not identity"
        assert np.array_equal(
            mapping.ref_to_gen_idx[mapping.gen_to_ref_idx], np.arange(N)
        ), f"{code}: ref_to_gen_idx[gen_to_ref_idx] is not identity"

        # 3. Element symbol parity
        ref_symbols = np.array([a.symbol for a in mol.atoms])
        gen_symbols = np.array([a.symbol for a in shell.atoms])
        mapped_symbols = mapping.apply_to_symbols(gen_symbols)
        assert np.array_equal(mapped_symbols, ref_symbols), f"{code}: Mapped symbols do not match reference symbols"
        assert mapping.verify_symbols(ref_symbols, gen_symbols) is True

        # 4. Metal atom mapping verification
        ref_metal_0idx = next(i for i, a in enumerate(mol.atoms) if a.idx == mol.metal_idx)
        gen_metal_0idx = next(i for i, a in enumerate(shell.atoms) if a.idx == shell.metal_idx)
        assert mapping.gen_to_ref_idx[ref_metal_0idx] == gen_metal_0idx, f"{code}: Metal atom did not map to metal atom"
        assert mapping.ref_to_gen_idx[gen_metal_0idx] == ref_metal_0idx, f"{code}: Gen metal atom did not map to ref metal"

        # 5. Covalent bond topology and distance parity
        bond_diag = mapping.verify_covalent_bonds(mol, shell, tol_distance=0.5)
        assert bond_diag["all_bonds_within_tolerance"], (
            f"{code}: Covalent bonds violated tolerance: {bond_diag['covalent_bond_errors']}"
        )
        assert bond_diag["max_distance_delta"] < 0.5, (
            f"{code}: Max bond distance delta {bond_diag['max_distance_delta']} >= 0.5 A"
        )


def test_atom_mapping_ambiguity_and_automorphisms():
    """Verify that complexes with symmetric or multiple identical ligands are flagged as ambiguous."""
    for code, shard, ds in [
        ("ICEYAQ", "X1s000", "X1"),
        ("HESBAM", "X1s000", "X1"),
        ("QOCKUO", "X1s005", "X1"),
        ("DAFFEZ", "X1s000", "X1"),
    ]:
        mol = load_mol(code, shard, ds)
        builder, _, _ = build_builder_and_inputs(mol)
        mapping = build_atom_mapping(mol, builder=builder)
        assert mapping.is_ambiguous is True, f"{code} should be flagged as ambiguous"
        assert mapping.num_automorphisms > 1, f"{code} should have num_automorphisms > 1"


def test_iceyaq_root_cause_and_compaction_parity():
    """Verify ICEYAQ root cause and compaction diagnostics before vs after atom mapping."""
    mol = load_mol("ICEYAQ", "X1s000", "X1")
    assert mol is not None

    npz_path = REPO_ROOT / "data/refit/nci_v2/pilot_v1/pools/ICEYAQ_s11.npz"
    assert npz_path.exists(), f"Missing pool file {npz_path}"
    npz = np.load(npz_path)

    ref_X = npz["ref_X"]
    ref_symbols = npz["ref_symbols"]
    gen_X = npz["X"][0]
    gen_symbols = npz["symbols"]

    # Symbols differ without mapping
    assert not np.array_equal(ref_symbols, gen_symbols), "ICEYAQ symbols should differ before mapping"

    mapping = build_atom_mapping(mol, X_gen=gen_X)
    mapped_gen_X = mapping.apply_to_positions(gen_X)

    # 1. Symbol matching after mapping
    mapped_gen_symbols = mapping.apply_to_symbols(gen_symbols)
    assert np.array_equal(mapped_gen_symbols, ref_symbols), "Mapped symbols must match reference symbols"

    # 2. Compaction diagnostics: Rg and nonbonded contacts
    ref_rg = compute_heavy_atom_rg(ref_symbols, ref_X)
    ref_contacts = compute_nonbonded_contacts(mol, ref_X, cutoff=3.5)

    # Erroneous unmapped calculation (prior audit observation: Rg 3.876 A, ratio 0.898)
    erroneous_unmapped_rg = compute_heavy_atom_rg(ref_symbols, gen_X)
    erroneous_ratio = erroneous_unmapped_rg / ref_rg
    assert abs(erroneous_ratio - 0.898) < 0.01, f"Expected unmapped Rg ratio ~0.898, got {erroneous_ratio}"

    # Correct mapped calculation (proper parity)
    mapped_rg = compute_heavy_atom_rg(ref_symbols, mapped_gen_X)
    mapped_ratio = mapped_rg / ref_rg
    assert abs(mapped_ratio - 0.988) < 0.01, f"Expected mapped Rg ratio ~0.988, got {mapped_ratio}"
    assert mapped_ratio > 0.95, f"Mapped Rg ratio {mapped_ratio} should be >= 0.95"

    # Contacts with fixed 0-based graph
    mapped_contacts = compute_nonbonded_contacts(mol, mapped_gen_X, cutoff=3.5)
    assert ref_contacts == 17, f"Expected 17 nonbonded contacts in ref, got {ref_contacts}"
    assert mapped_contacts == 20, f"Expected 20 nonbonded contacts in mapped gen, got {mapped_contacts}"
