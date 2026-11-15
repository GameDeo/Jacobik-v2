"""Donor identity reconciliation, assignment, and haptic typing guards."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from jacobik.core import profile
from jacobik.core.types import Atom, Bond, CoordinationSite, DonorLabel, Ligand, TMComplex
from jacobik.stats import keys as key_builder
from jacobik.topology import donor_classifier as dc
from jacobik.topology import mapping


def _assign_case(smiles, positions, metal_bo, declared):
    mol = Chem.MolFromSmiles(smiles)
    graph = nx.Graph()
    for a in mol.GetAtoms():
        graph.add_node(a.GetIdx(), symbol=a.GetSymbol(), hydrogens=a.GetTotalNumHs())
    graph.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    full_map = {i: i + 2 for i in graph}
    atoms = [Atom(1, "Pd", np.zeros(3))] + [
        Atom(i + 2, mol.GetAtomWithIdx(i).GetSymbol(), np.array([positions.get(i, 6.0), i * 0.1, 0]))
        for i in graph
    ]
    bonds = [Bond(1, i + 2, bo) for i, bo in metal_bo.items()]
    events = []
    got = mapping._assign_donors(mol, graph, full_map, atoms, bonds, 1, declared, events)
    return got, events


def _rejected_chelate(extra_bo=0.3, extra_x=1.776, authoritative=False):
    atoms = [
        Atom(i, s, np.array(p))
        for i, s, p in [
            (1, "Zn", [0.0, 0, 0]),
            (2, "O", [1.106, 1.156, 0]),
            (3, "C", [1.7, 0.0, 0]),
            (4, "O", [extra_x, 1.299, 0]),
            (5, "C", [3.0, -0.6, 0]),
        ]
    ]
    bonds = [Bond(2, 3, 1.5), Bond(3, 4, 1.5), Bond(3, 5, 1)]
    ligand = Ligand(0, [2, 3, 4, 5], [2], donors_are_authoritative=authoritative)
    mol = TMComplex("SYNTHETIC", 0, 1, "", 0, "", atoms, bonds, 1, "Zn", ligands=[ligand])
    return mol, ligand, [Bond(1, 2, 0.3), Bond(1, 4, extra_bo)]


def _complex(symbols, xyz, ligand_donors, bonds, metal_bonds):
    atoms = [Atom(1, symbols[0], np.asarray(xyz[0], float))]
    atoms.extend(Atom(i + 2, s, np.asarray(xyz[i + 1], float)) for i, s in enumerate(symbols[1:]))
    ligand = Ligand(0, list(range(2, len(symbols) + 1)), list(ligand_donors))
    mol = TMComplex(
        "TEST01", 0, 1, "test", 0, "", atoms, list(bonds), 1, symbols[0], coord_geom="other", ligands=[ligand]
    )
    return mol, ligand, list(metal_bonds)


def _enable_typing_flags(monkeypatch, *, assignment=True):
    for mod in (mapping, dc):
        monkeypatch.setattr(mod, "DONOR_ASSIGNMENT", assignment)
    monkeypatch.setattr(dc, "HAPTO_CHELATE_EXCLUSION", True)
    monkeypatch.setattr(dc, "HAPTO_BOND_GATE", True)
    monkeypatch.setattr(dc, "HAPTO_LABEL_CANON", True)


def test_donor_assignment_rules_alignment_and_defaults():
    for key in ("donor_assignment", "hapto_bond_gate",
                "hapto_label_canon", "hapto_chelate_exclusion"):
        assert profile.get(key) is True
    assert mapping.DONOR_ASSIGNMENT is True
    assert profile.snapshot()["fingerprint"] == "2c87e2940af2"

    a, b = nx.Graph(), nx.Graph()
    a.add_nodes_from([(0, dict(symbol="C", hydrogens=2)), (1, dict(symbol="C", hydrogens=3))])
    b.add_nodes_from([(10, dict(symbol="C", hydrogens=3)), (11, dict(symbol="C", hydrogens=2))])
    a.add_edge(0, 1)
    b.add_edge(10, 11)
    assert mapping._align_components(a, b, match_hydrogens=True) == {0: 11, 1: 10}
    b.nodes[11]["hydrogens"] = 1
    with pytest.raises(ValueError):
        mapping._align_components(a, b, match_hydrogens=True)

    got, ev = _assign_case("CC(=O)[O-]", {2: 2.65, 3: 2.05}, {2: 0.19, 3: 0.5}, {2})
    assert got == {5} and ev[0]["kind"] == "donor_assignment_changed"

    got, _ = _assign_case("CC(=O)[O-]", {2: 2.65, 3: 2.05}, {2: 0.19, 3: 0.5}, {2, 3})
    assert got == {4, 5}

    got, ev = _assign_case("CC(=O)[O-]", {2: 2.65, 3: 2.05}, {2: 0.0, 3: 0.5}, {2, 3})
    assert got == {5} and any(e["kind"] == "donor_assignment_unrepresented" for e in ev)

    got, ev = _assign_case("CC(=O)[O-]", {2: 2.4, 3: 2.4}, {2: 0.0, 3: 0.0}, {3})
    assert got == set() and any(e["kind"] == "donor_assignment_unrepresented" for e in ev)

    got, _ = _assign_case("Cc1ncccn1", {2: 2.1, 6: 4.0}, {2: 0.5}, {6})
    assert got == {4}

    got, ev = _assign_case("NCC(=O)[O-]", {0: 3.4, 4: 2.05}, {0: 0.0, 4: 0.5}, {0})
    assert got == {6} and any(e["kind"] == "donor_assignment_transferred" for e in ev)

    got, _ = _assign_case("CC(=O)[O-]", {2: 9.0, 3: 9.0}, {2: 0.3, 3: 0.3}, {3})
    assert got == set()


def test_source_fixtures_donor_identity_and_nuhpel(monkeypatch):
    from fit.parser import _assemble_raw

    data_dir = Path(__file__).parent / "data/donor_identity"
    cases = [
        ("JAYCEV", {5, 6, 7, 9}, {8, 10}),
        ("UKUWON", {2, 3, 4, 25, 26, 27}, {5, 28, 30}),
        ("BADQIH", {6, 8, 10, 12, 14}, {3, 7}),
        ("AGAPEG", {3}, {4}),
        ("ILEPOE", {6}, {7}),
    ]
    sources = {c: json.loads((data_dir / f"{c}.json").read_text()) for c, _, _ in cases}
    sources["NUHPEL"] = json.loads((data_dir / "NUHPEL.json").read_text())

    _enable_typing_flags(monkeypatch, assignment=True)
    for code, required, forbidden in cases:
        if code == "BADQIH":
            required = required | {2, 4}
        src = sources[code]
        raw = _assemble_raw(src["xyz_comment"], src["xyz_atoms"], code, src["bo_lines"], SimpleNamespace(smiles=src["smiles"]))
        mol = mapping.assemble(raw)
        atoms = {i for site in mol.coord_sites for i in site.donor_indices}
        assert required <= atoms and not (forbidden & atoms)
        if code == "UKUWON":
            assert atoms == required and len(mol.coord_sites) == 6

    src = sources["NUHPEL"]
    for assign in (True, False):
        _enable_typing_flags(monkeypatch, assignment=assign)
        raw = _assemble_raw(src["xyz_comment"], src["xyz_atoms"], "NUHPEL", src["bo_lines"], SimpleNamespace(smiles=src["smiles"]))
        mol = mapping.assemble(raw)
        if assign:
            assert sorted(tuple(s.donor_indices) for s in mol.coord_sites) == [(1,), (3,), (22,), (41,)]
            assert any(e["kind"] == "hapto_face_unsupported" for e in mol.donor_identity_events)
        else:
            assert any(len(s.donor_indices) == 3 for s in mol.coord_sites)


def test_rejected_face_kappa_preservation_and_local_recovery(monkeypatch):
    _enable_typing_flags(monkeypatch)
    mol, _, metal_bonds = _rejected_chelate()
    dc.classify(mol, metal_bonds=metal_bonds, smiles_path=True)
    assert sorted(s.donor_indices for s in mol.coord_sites) == [[2], [4]]
    assert any(e["kind"] == "rejected_face_kappa_preservation" for e in mol.donor_identity_events)

    for bo, x, authoritative in ((0.08, 1.776, False), (0.3, 3.2, False), (0.3, 1.776, True)):
        mol, lig, mb = _rejected_chelate(bo, x, authoritative)
        graph = dc._ligand_heavy_graph(mol, set(lig.atom_indices))
        assert dc._supported_rejected_face_donors(mol, lig, [3, 4], graph, {a.idx: a for a in mol.atoms}, mb) <= {2}


def test_hapto_bond_gate_chelate_exclusion_and_label_canon(monkeypatch):
    _enable_typing_flags(monkeypatch)

    mol, lig, mb = _complex(["Fe", "S", "C"], [[0, 0, 0], [2.0, 0, 0], [2.6, 0.9, 0]], [2], [Bond(2, 3, 1.8)], [Bond(1, 2, 0.3), Bond(1, 3, 0.2)])
    assert max(dc._eta2_face_bond_ratios(mol, {a.idx: a for a in mol.atoms}, [2, 3])) > 1.21
    dc.classify(mol, metal_bonds=mb)
    assert lig.haptic_groups == [] and lig.donor_atoms == [2]
    assert [(s.kind, s.donor_indices) for s in mol.coord_sites] == [("kappa", [2])]

    mol, lig, mb = _complex(["Fe", "C", "C"], [[0, 0, 0], [2.0, 0, 0], [2.0, 1.2, 0]], [2], [Bond(2, 3, 1.8)], [Bond(1, 2, 0.3), Bond(1, 3, 0.2)])
    dc.classify(mol, metal_bonds=mb)
    assert len(lig.haptic_groups) == 1 and lig.haptic_groups[0].haptic_order == 2 and set(lig.haptic_groups[0].donor_indices) == {2, 3}

    mol, lig, mb = _complex(["Fe", "C", "C"], [[0, 0, 0], [2.0, 0, 0], [2.8, 0.9, 0]], [2], [Bond(2, 3, 1.8)], [Bond(1, 2, 0.3), Bond(1, 3, 0.2)])
    dc.classify(mol, metal_bonds=mb)
    assert lig.haptic_groups == [] and lig.donor_atoms == [2]

    mol, lig, mb = _complex(["Zn", "O", "C", "O", "C"], [[0, 0, 0], [2.0, 0, 0], [2.3, 0.65, 0], [3.05, 0.8, 0], [3.7, 1.6, 0]], [2, 4], [Bond(2, 3, 1.5), Bond(3, 4, 1.5), Bond(3, 5, 1.0)], [Bond(1, 2, 0.3), Bond(1, 4, 0.2)])
    dc.classify(mol, metal_bonds=mb)
    assert lig.haptic_groups == [] and lig.donor_atoms == [2, 4]
    assert sorted((s.kind, tuple(s.donor_indices)) for s in mol.coord_sites) == [("kappa", (2,)), ("kappa", (4,))]

    mol, lig, mb = _complex(["Zn", "O", "C", "O", "C"], [[0, 0, 0], [1.106, 1.156, 0], [1.7, 0, 0], [1.776, 1.299, 0], [3.0, -0.6, 0]], [2], [Bond(2, 3, 1.5), Bond(3, 4, 1.5), Bond(3, 5, 1.0)], [Bond(1, 2, 0.3)])
    dc.classify(mol, metal_bonds=mb)
    assert lig.haptic_groups == [] and [s.donor_indices for s in mol.coord_sites] == [[2]]

    mol, lig, _ = _complex(["Zn", "O", "C", "O", "C"], [[0, 0, 0], [2.0, 0, 0], [2.3, 0.65, 0], [3.05, 0.8, 0], [2.15, -0.6, 0]], [2], [Bond(2, 3, 1.5), Bond(3, 4, 1.5), Bond(3, 5, 1.0)], [Bond(1, 2, 0.3)])
    graph = dc._ligand_heavy_graph(mol, set(lig.atom_indices))
    excluded, declared, geometric = dc._eta2_chelate_donor_neighbors(mol, lig, [3, 4], graph, {a.idx: a for a in mol.atoms})
    assert excluded and 2 in declared and 5 in geometric

    mol, lig, mb = _complex(["Zn", "O", "C", "O", "C"], [[0, 0, 0], [1.179, 1.082, 0], [1.9, 0, 0], [3.2, 0, 0], [2.8, -1.0, 0]], [2], [Bond(2, 3, 1.5), Bond(3, 4, 1.5), Bond(3, 5, 1.0)], [Bond(1, 2, 0.3)])
    assert dc._eta2_face_bond_ratios(mol, {a.idx: a for a in mol.atoms}, [4])[0] > 1.21
    dc.classify(mol, metal_bonds=mb)
    assert len(lig.haptic_groups) == 1 and set(lig.haptic_groups[0].donor_indices) == {2, 3}

    mol, lig, mb = _complex(["Zn", "S", "C", "S"], [[0, 0, 0], [2.0, 0, 0], [2.25, 0.65, 0], [3.05, 0.8, 0]], [2, 4], [Bond(2, 3, 1.5), Bond(3, 4, 1.5)], [Bond(1, 2, 0.3), Bond(1, 4, 0.2)])
    dc.classify(mol, metal_bonds=mb)
    assert lig.haptic_groups == [] and sorted(tuple(s.donor_indices) for s in mol.coord_sites) == [(2,), (4,)]

    mol, lig, mb = _complex(["Zn", "O", "C", "O"], [[0, 0, 0], [2.0, 0, 0], [2.0, 1.0, 0], [2.0, 2.2, 0]], [2], [Bond(2, 3, 1.5), Bond(3, 4, 1.5)], [Bond(1, 2, 0.3)])
    dc.classify(mol, metal_bonds=mb)
    assert len(lig.haptic_groups) == 1 and set(lig.haptic_groups[0].donor_indices) == {2, 3} and 4 not in lig.donor_atoms

    mol, _, _ = _complex(["Fe", "C", "N"], [[0, 0, 0], [2, 0, 0], [2, 1, 0]], [2, 3], [], [])
    mol.donor_labels = {2: DonorLabel(2, "C_donor", "C_haptic", True), 3: DonorLabel(3, "N_donor", "N_imine", True)}
    monkeypatch.setattr(key_builder, "HAPTO_LABEL_CANON", True)
    for face in ([2, 3], [3, 2]):
        assert key_builder._site_subclass(CoordinationSite("haptic", list(face), np.zeros(3), "η", haptic_order=2), mol) == "C_haptic"

    mol, _, mb = _complex(["Fe", "N", "C"], [[0, 0, 0], [2.0, 0, 0], [2.0, 1.2, 0]], [2, 3], [Bond(2, 3, 1.8)], [Bond(1, 2, 0.3), Bond(1, 3, 0.2)])
    dc.classify(mol, metal_bonds=mb)
    assert len(mol.coord_sites) == 1 and mol.coord_sites[0].label == "C_haptic"
    assert key_builder._site_subclass(mol.coord_sites[0], mol) == "C_haptic"


def test_legacy_flags_off_preserve_classify_outputs(legacy_flags):
    from tools.jgen import inputs

    expected = {
        "IYUKOC": "a9273f9dc3c7db2bdf3b96396ca188708d25d4345c1e81679c0118c098c8824f",
        "PENHEY": "c168dc4dd702b346a4d86cd7050755a1fe351aa4984868b4481c7939bcfebf81",
        "UMUBAF": "ffaa1ebe38a8ac7ce820b9c4f444ad24e8101b0cd9fcfcc79a8f6420ecad1e66",
    }
    frame = inputs.frame_shards()
    observed = {}
    for code in expected:
        shard, dataset = frame[code]
        mol = inputs.load_mol(code, shard, dataset)
        value = {
            "coord_geom": mol.coord_geom,
            "donor_labels": {str(k): [v.family, v.subclass, v.is_haptic] for k, v in sorted(mol.donor_labels.items())},
            "sites": [
                {
                    "kind": s.kind,
                    "donor_indices": s.donor_indices,
                    "position": s.position.tolist(),
                    "label": s.label,
                    "avg_bond_order": s.avg_bond_order,
                    "haptic_order": s.haptic_order,
                }
                for s in mol.coord_sites
            ],
            "haptic_groups": [
                [
                    {
                        "donor_indices": h.donor_indices,
                        "haptic_order": h.haptic_order,
                        "centroid": h.centroid.tolist(),
                        "donor_symbols": h.donor_symbols,
                        "avg_bond_order": h.avg_bond_order,
                    }
                    for h in lig.haptic_groups
                ]
                for lig in mol.ligands
            ],
        }
        observed[code] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert observed == expected
