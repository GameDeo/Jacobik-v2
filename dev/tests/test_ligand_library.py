"""Tests for interface/ligand_library.py, debug complexes, and library Python/CLI integration."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import jacobik
from jacobik.builder import cli as cli_mod
from interface.jacobik_cli import main as cli_main
from interface.ligand_library import (
    ALIASES,
    DEBUG_COMPLEXES,
    DEBUG_COMPLEX_CBTYCO,
    DEBUG_COMPLEX_OJIMIF,
    LIGANDS,
    debug_complex,
    ligand,
    ligand_modes,
    ligands,
    list_ligands,
)


def _dihedral_deg(coords: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    b0 = -(coords[b] - coords[a])
    b1 = coords[c] - coords[b]
    b2 = coords[d] - coords[c]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return abs(math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _lig_key(lig: jacobik.LigandInput):
    return (lig.smiles, lig.donors, lig.sites)


def test_ligand_library_chemistry_indices_aliases_and_modes():
    assert len(LIGANDS) >= 50
    for name, ldef in LIGANDS.items():
        mol = Chem.MolFromSmiles(ldef.smiles)
        assert mol is not None, f"{name}: invalid SMILES {ldef.smiles!r}"
        assert mol.GetNumHeavyAtoms() >= 2, f"{name}: single-atom donors should be excluded"
        assert Chem.GetFormalCharge(mol) == ldef.charge, f"{name}: charge mismatch"
        if name != "NO":
            assert all(a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms()), f"{name}: unexpected radical"

        mol_h = Chem.AddHs(mol)
        etkdg = AllChem.ETKDGv3()
        etkdg.randomSeed = 11
        assert AllChem.EmbedMolecule(mol_h, etkdg) == 0, f"{name}: ETKDGv3 failed"
        if name == "mnt":
            pos = mol_h.GetConformer().GetPositions()
            assert _dihedral_deg(pos, 3, 2, 4, 5) < 5.0, "mnt must be cis around C=C"

        n_atoms = mol.GetNumAtoms()
        assert ldef.default_mode in ldef.modes
        for mode_name, bm in ldef.modes.items():
            assert bm.denticity >= 1
            seen = set()
            for grp in bm.donors:
                assert len(grp) >= 1
                for idx in grp:
                    assert 0 <= idx < n_atoms and idx not in seen
                    seen.add(idx)
                    sym = mol.GetAtomWithIdx(idx).GetSymbol()
                    if mode_name.startswith("eta"):
                        assert sym == "C"
                    elif sym == "C":
                        assert ldef.category == "nhc" or name in ("CO", "CN") or mode_name == "kappa1_C"
            inp = ligand(f"{name}@{mode_name}")
            assert inp.smiles == ldef.smiles and [list(g) for g in bm.donors] == inp.donors

    # Aliases, binding modes, repeat syntax, and invalid-spec guards
    assert _lig_key(ligand("bipyridine")) == _lig_key(ligand("bpy"))
    assert _lig_key(ligand("isothiocyanate")) == _lig_key(ligand("SCN@kappa1_N"))
    assert _lig_key(ligand("thiocyanate")) == _lig_key(ligand("SCN@kappa1_S"))
    assert _lig_key(ligand("dmso", "kappa1_O")) == _lig_key(ligand("dmso@kappa1_O")) == _lig_key(ligand("dmso-o"))
    assert _lig_key(ligand("dmso@O")) != _lig_key(ligand("dmso@S"))
    assert _lig_key(ligand("OAc@kappa2")) == _lig_key(ligand("chelating-acetate"))

    bpy3 = ligands("bpy*3")
    assert len(bpy3) == 3 and all(_lig_key(x) == _lig_key(ligand("bpy")) for x in bpy3) and bpy3[0] is not bpy3[1]
    cpstar2 = ligands("Cp**2")
    assert len(cpstar2) == 2 and all(_lig_key(x) == _lig_key(ligand("Cp*")) for x in cpstar2)

    assert "kappa1_O" in ligand_modes("dmso")
    assert any(d.name == "Xantphos" for d in list_ligands(query="pincer"))
    assert len(ALIASES) >= 50

    with pytest.raises(KeyError, match="Did you mean"):
        ligand("bipyridinex")
    with pytest.raises(ValueError, match="no binding mode"):
        ligand("bpy@eta5")
    with pytest.raises(ValueError, match="repeat count"):
        cli_mod._parse_ligand_spec("N:[[0]]*0")
    assert jacobik.main(["solve", "--metal", "Pt", "--geometry", "square_plan", "--ligand", "N:[[0]]*0"]) == 1


def test_debug_complex_unpacking_and_string_ligands_solve(stats):
    assert set(DEBUG_COMPLEXES) == {"CBTYCO", "OJIMIF"}
    assert len(DEBUG_COMPLEX_OJIMIF) == 5 and len(DEBUG_COMPLEX_CBTYCO) == 5
    assert debug_complex("cbtyco") == DEBUG_COMPLEX_CBTYCO

    for name, dc in DEBUG_COMPLEXES.items():
        res = jacobik.generate(
            *dc, stats=stats, seed=11, n_conformers=2, beam_width=2,
            n_coord_samples=2, n_ik_steps=3, embed="etkdg", device="cpu",
        )
        assert res.failure is None and res.conformers, f"{name}: unexpected failure {res.failure!r}"
        assert all(np.isfinite([a.position for a in c.complex.atoms]).all() for c in res.conformers)

    assert _lig_key(jacobik.ligand("dmso@kappa1_O")) == _lig_key(ligand("dmso@kappa1_O"))
    assert [_lig_key(x) for x in jacobik.ligands("bpy*3")] == [_lig_key(x) for x in ligands("bpy*3")]
    assert jacobik.debug_complex("OJIMIF")[:4] == DEBUG_COMPLEX_OJIMIF[:4]
    assert [_lig_key(x) for x in jacobik.debug_complex("OJIMIF").ligands] == [
        _lig_key(x) for x in DEBUG_COMPLEX_OJIMIF.ligands
    ]

    res_str = jacobik.generate(
        metal="Fe", geometry="octahedral", charge=2, spin=1, ligands=["bpy*3"],
        stats=stats, seed=11, n_conformers=1, beam_width=2, n_coord_samples=2,
        n_ik_steps=3, embed="etkdg", device="cpu",
    )
    assert res_str.failure is None and res_str.conformers
    assert len(jacobik.ComplexBuilder("Pt", "square_plan", 0, 1, ["en", "dmso@kappa1_O", ("CC#N", [[2]])]).ligands) == 3

    with pytest.raises(TypeError, match="not str"):
        jacobik.ComplexBuilder("Fe", "octahedral", 2, 1, "bpy")


def test_cli_ligand_library_and_debug_complex(stats, capsys, tmp_path, monkeypatch):
    assert jacobik.main(["ligands"]) == 0
    listing = capsys.readouterr().out
    assert "CBTYCO" in listing and "bpy" in listing

    assert jacobik.main(["ligands", "dmso", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)["modes"]) == {"kappa1_S", "kappa1_O"}

    assert cli_main(["ligands", "--query", "thiocyanate"]) == 0
    assert "SCN" in capsys.readouterr().out

    # Unmocked CLI --debug-complex exercises end-to-end default packaged DB loading
    out_dbg = tmp_path / "ojimif.xyz"
    assert cli_main([
        "--debug-complex", "OJIMIF", "--embed", "etkdg", "--n-ik-steps", "2",
        "--beam-width", "2", "--n-coord-samples", "2", "--n-conformers", "1",
        "--device", "cpu", "--out", str(out_dbg),
    ]) == 0
    assert out_dbg.is_file() and out_dbg.stat().st_size > 0

    # Reuse session StatsLookup for the second CLI solve so we don't parse the 28MB DB twice
    orig_gen = cli_mod.generate
    monkeypatch.setattr(cli_mod, "generate", lambda *a, **kw: orig_gen(*a, stats=stats, **kw))
    out_lib = tmp_path / "pt_lib.xyz"
    assert jacobik.main([
        "solve", "--metal", "Pt", "--geometry", "square_plan", "--charge", "0", "--spin", "1",
        "--ligand", "en", "--ligand", "SCN@kappa1_N", "--ligand", "dmso@kappa1_O",
        "--embed", "etkdg", "--n-ik-steps", "2", "--beam-width", "2",
        "--n-coord-samples", "2", "--n-conformers", "1", "--device", "cpu", "--out", str(out_lib),
    ]) == 0
    assert out_lib.is_file() and out_lib.stat().st_size > 0
