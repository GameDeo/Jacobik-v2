"""Core jacobik domain unit tests: ladders, hierarchy, Q6 exclusions, oxidation states, Shannon shifts."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import torch

from jacobik.core import chemistry as O
from jacobik.core import chemistry as SH
from jacobik.core.types import Atom, Bond, CoordinationSite, DonorLabel, TMComplex
from jacobik.stats import ladders as L
from jacobik.topology import donor_classifier as DC
from jacobik.topology import donor_classifier as H


def _synthetic_eta5(tilt_deg=0.0, atom_order=None):
    """Planar eta5 face whose normal is tilted from the metal-centroid axis."""
    phi = np.deg2rad(np.arange(5) * 72.0)
    face = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(5)))
    a = np.deg2rad(tilt_deg)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    face = face @ rx.T
    order = list(range(5)) if atom_order is None else list(atom_order)
    atoms = [Atom(1, "Fe", np.array([0.0, 0.0, 2.0]))] + [Atom(i + 2, "C", face[i]) for i in order]
    donor_ids = [a.idx for a in atoms[1:]]
    site = CoordinationSite("haptic", donor_ids, face.mean(axis=0), "eta5", haptic_order=5)
    labels = {i: DonorLabel(i, "C_donor", "C_eta5", True) for i in donor_ids}
    bonds = [Bond(donor_ids[i], donor_ids[(i + 1) % 5], 1.5) for i in range(5)]
    return TMComplex("ETA5", 0, 1, "FeC5", 0, "", atoms, bonds, 1, "Fe",
                     coord_sites=[site], coord_geom="other", donor_labels=labels)


class _L:
    def __init__(self, smi, d):
        self.smiles, self.donors = smi, d


def test_ladders_expander_lookup_parity_and_cn():
    q1_keys = [
        ("Fe", 1, 2, "N_pyridine", "octahedral", "O_water"),
        ("La", 10, 3, "O_hydroxide", "square_anti", "none"),
        ("Fe", 5, L.OS_UNKNOWN, "N_ammonia", "other", "none"),
        ("Pt", 1, 2, "C_cyanide", "tricap_pris", "none"),
    ]
    for key in q1_keys:
        gen = {k for _, k in L.q1_ancestor_rows(key)} | {key}
        for var in L.Q1_VARIANTS:
            for c in L.q1_candidates(key, var):
                assert c in gen, (var, c)

    cases = {
        "Q2": ("octahedral", "c90", "chelate_short", "Fe", "N_pyridine", "O_water"),
        "Q3": ("N_pyridine", "Fe", "octahedral", 0),
        "Q4": ("N_ammonia", "octahedral", 3, 0),
        "Q5": ("C_sp3", "N_sp3", "C_ar", "H", "free"),
    }
    cand = {"Q2": L.q2_candidates, "Q3": L.q3_candidates, "Q4": L.q4_candidates, "Q5": L.q5_candidates}
    for q, key in cases.items():
        gen = {k for _, k in L.ANCESTOR_FN[q](key)} | {key}
        assert set(cand[q](key)) <= gen

    k_unk = ("Fe", 1, L.OS_UNKNOWN, "N_pyridine", "octahedral", "none")
    assert all(L.OS_UNKNOWN not in c for c in L.q1_candidates(k_unk))
    assert all(L.OS_UNKNOWN not in c for _, c in L.q1_ancestor_rows(k_unk))

    a = ("octahedral", "c90", "different_ligand", "Fe", "N_pyridine", "O_water")
    b = ("octahedral", "c90", "different_ligand", "Fe", "O_water", "N_pyridine")
    assert L.q2_candidates(a)[1:] == L.q2_candidates(b)[1:]
    assert L.geom_cn("tricap_pris") == "CN9"


def test_orientation_variants_and_anionic_co_donor_bucket():
    from jacobik.stats.keys import _site_representative_idx, anionic_co_donor_bucket
    from jacobik.stats.stats import _L0_ARITY

    assert L.DEFAULT_ORIENTATION_VARIANT == "anionic_context"
    assert _L0_ARITY["Q3"] == 4 and _L0_ARITY["Q4"] == 4

    q3 = L.q3_candidates(("O_water", "Fe", "octahedral", 0), "neutral_sigma_pool")
    q4 = L.q4_candidates(("O_water", "octahedral", 2, 0), "neutral_sigma_pool")
    assert q3[0] == ("O_neutral_sigma",) and q4[0] == ("O_neutral_sigma", 2)
    assert all("O_water" not in key for key in q3) and all("O_water" not in key for key in q4)
    assert L.q3_candidates(("N_pyridine", "Fe", "octahedral", 0), "neutral_sigma_pool")[0] == (
        "N_pyridine", "Fe", "octahedral", 0
    )
    assert L.q4_candidates(("N_pyridine", "octahedral", 2, 0), "neutral_sigma_pool")[0] == (
        "N_pyridine", "octahedral", 2, 0
    )

    key3 = ("O_water", "Fe", "octahedral", 1)
    key4 = ("O_water", "octahedral", 2, 1)
    q3 = L.q3_candidates(key3)
    q4 = L.q4_candidates(key4)
    assert q3[0] == key3 and q3[1] == ("O_water", "Fe", 1) and q3[2] == ("O_water", 1)
    assert ("O_water", "Fe", "octahedral") in q3 and q3[-1] == ("element_any",)
    assert q3.index(("O_neutral_sigma",)) > q3.index(("O_water", 1))
    assert q4[0] == key4 and q4[1] == ("O_water", 2, 1) and ("O_water", "octahedral", 2) in q4

    assert L.q3_candidates(("O_water", "Fe", "octahedral", 0), "fine_leaf")[0] == ("O_water", "Fe", "octahedral")
    assert L.q4_candidates(("O_water", "octahedral", 2, 0), "fine_leaf")[0] == ("O_water", "octahedral", 2)
    with pytest.raises(ValueError, match="unknown orientation variant"):
        L.q3_candidates(key3, "joint_gram_v1")

    def _mol(subclasses):
        atoms = [Atom(1, "Fe", np.zeros(3))] + [Atom(i + 2, "O", np.zeros(3)) for i in range(len(subclasses))]
        sites = [CoordinationSite("kappa", [i + 2], np.zeros(3), sub) for i, sub in enumerate(subclasses)]
        labels = {i + 2: DonorLabel(i + 2, "O_donor", sub, False) for i, sub in enumerate(subclasses)}
        return TMComplex("X", 0, 1, "FeOn", 0, "", atoms, [], 1, "Fe",
                         coord_sites=sites, coord_geom="octahedral", donor_labels=labels)

    assert anionic_co_donor_bucket(_mol(["O_water", "O_water"])) == 0
    m2 = _mol(["O_water", "O_hydroxide", "O_carboxylate"])
    assert anionic_co_donor_bucket(m2) == 2
    assert anionic_co_donor_bucket(_mol(["O_water", "O_oxo"])) == 1
    assert anionic_co_donor_bucket(m2, exclude_rep_idx=_site_representative_idx(m2.coord_sites[1])) == 1


def test_haptic_q3_extractor_runtime_and_permutation():
    from fit.q3 import q3_approach_angles
    from fit.q4 import q4_pyramidalization
    from jacobik.ik.geometry import GeometryExtractor

    for tilt in (0.0, 17.5, 41.0):
        mol = _synthetic_eta5(tilt)
        raw = q3_approach_angles(mol)
        assert len(raw) == 1 and raw[0][0] == ("C_eta5", "Fe", "other", 0)
        assert raw[0][1] == pytest.approx(tilt, abs=1e-5)
        ge = GeometryExtractor(mol, torch.device("cpu"))
        X = torch.tensor(mol.positions(), dtype=torch.float64).unsqueeze(0)
        assert np.rad2deg(ge.q3_approach_angles(X).detach().numpy()[0, 0]) == pytest.approx(raw[0][1], abs=1e-5)
        assert q4_pyramidalization(mol) == []

    a = q3_approach_angles(_synthetic_eta5(23.0))[0][1]
    b = q3_approach_angles(_synthetic_eta5(23.0, [3, 1, 4, 0, 2]))[0][1]
    assert a == pytest.approx(b, abs=1e-8)
    assert L.q3_candidates(("C_eta5", "Fe", "other", 0))[0] == ("C_eta5", "Fe", "other", 0)


def test_donor_hierarchy_q6_exclusion_and_flank(legacy_flags):
    from fit.q6 import q6_steric_contacts
    from jacobik.topology.graph import canonical_flank, q6_topological_exclusions
    from jacobik.topology.mapping import _apply_smiles_path, _raw_coordination_donor_indices, raw_metal_bond_exclusion_pairs

    for name in ("C_cyanide", "N_ammonia", "N_primary_amine", "N_secondary_amine", "N_tertiary_amine",
                 "N_amido", "N_imido", "O_water", "O_hydroxide", "O_oxo", "O_alcohol", "O_heteroxo"):
        assert H.is_known(name) and H.full_path(name)[-1] == H.ELEMENT_ANY

    def _oxy(nH, heavy=()):
        sym, adj, i = {0: "O"}, {0: []}, 1
        for s in ["H"] * nH + list(heavy):
            sym[i] = s; adj[0].append((i, 1.0)); adj[i] = [(0, 1.0)]; i += 1
        return DC._classify_oxygen_structural(0, sym, adj, set(), set())

    def _nit(nH, heavy):
        sym, adj, i = {0: "N"}, {0: []}, 1
        for s in ["H"] * nH + list(heavy):
            sym[i] = s; adj[0].append((i, 1.0)); adj[i] = [(0, 1.0)]; i += 1
        return DC._classify_nitrogen_structural(0, sym, adj, set())

    assert _oxy(2) == "O_water" and _oxy(1) == "O_hydroxide" and _oxy(0) == "O_oxo" and _oxy(1, ["C"]) == "O_alcohol"
    assert _nit(3, []) == "N_ammonia" and _nit(2, ["C"]) == "N_primary_amine"
    assert _nit(1, ["C", "C"]) == "N_secondary_amine" and _nit(0, ["C", "C", "C"]) == "N_tertiary_amine"
    assert _nit(0, ["C", "C"]) == "N_amido" and _nit(0, ["C"]) == "N_imido"

    atoms = [Atom(idx=i, symbol=s, position=np.zeros(3)) for i, s in zip(range(1, 7), ["Fe", "O", "C", "C", "C", "H"])]
    bonds = [Bond(2, 3, 1.0), Bond(3, 4, 1.0), Bond(4, 5, 1.0), Bond(2, 6, 1.0)]
    ex = q6_topological_exclusions(atoms, bonds, 1, [2], max_bonds=3)
    assert (1, 2) in ex and (1, 3) in ex and (1, 4) not in ex
    assert (1, 6) in ex and (2, 6) in ex and (3, 6) in ex and (4, 6) in ex
    assert (1, 5) not in ex and (5, 6) not in ex

    atoms_m = [Atom(idx=1, symbol="Fe", position=np.zeros(3)), Atom(idx=2, symbol="O", position=np.array([2.0, 0.0, 0.0]))]
    assert _raw_coordination_donor_indices(atoms_m, [Bond(1, 2, 0.60)], 1) == {2}
    assert raw_metal_bond_exclusion_pairs(atoms_m, [Bond(1, 2, 0.60)], 1) == {(1, 2)}
    assert _apply_smiles_path(atoms_m, [Bond(1, 2, 0.60)], 1, "[Fe]O")[1] == set()
    atoms_h = [atoms_m[0], Atom(idx=2, symbol="H", position=np.array([1.5, 0.0, 0.0]))]
    assert _raw_coordination_donor_indices(atoms_h, [Bond(1, 2, 0.40)], 1) == set()

    mol_raw = TMComplex("RAWQ6", 0, 1, "FeO", 0, "", atoms_m, [Bond(1, 2, 0.20)], 1, "Fe")
    assert mol_raw.coord_sites == [] and q6_steric_contacts(mol_raw) == []

    for perm in itertools.permutations([10, 11, 12]):
        amap = {i: Atom(idx=i, symbol="C", position=np.zeros(3)) for i in perm}
        tmap = {10: "C_sp3", 11: "C_ar", 12: "C_sp3"}
        assert tmap[canonical_flank(list(perm), amap, tmap, {10: 1, 11: 1, 12: 1})] == "C_sp3"


def test_oxidation_state_and_ligand_charge_conventions():
    for metal, q, ligs, exp in (
        ("Fe", 0, [_L("[cH-]1cccc1", [[0, 1, 2, 3, 4]])] * 2, 2),
        ("Fe", -3, [_L("[C-]#N", [[0]])] * 6, 3),
        ("Fe", -4, [_L("[C-]#N", [[0]])] * 6, 2),
        ("Mn", -1, [_L("[O-2]", [[0]])] * 4, 7),
        ("Pt", 0, [_L("N", [[0]])] * 2 + [_L("[Cl-]", [[0]])] * 2, 2),
        ("Fe", 3, [_L("O", [[0]])] * 6, 3),
        ("Cr", 0, [_L("c1ccccc1", [[0, 1, 2, 3, 4, 5]])] + [_L("[C-]#[O+]", [[0]])] * 3, 0),
    ):
        assert O.deduce_oxidation_state(metal, q, ligs)[0] == exp

    assert O.deduce_oxidation_state("Fe", 0, [_L("not a smiles", [[0]])])[0] == L.OS_UNKNOWN
    assert O.deduce_oxidation_state("Fe", 0, [_L("N", [[-1]])], model="training")[0] == L.OS_UNKNOWN

    for smiles, donors, expected in (
        ("[C]1NCCN1", [[0]], 0),
        ("[CH3-]", [[0]], -1),
        ("[O-2]", [[0]], -2),
        ("O=C([O-])C", [[0, 1], [2]], -1),
        ("[CH-]1C=C[CH-]C1", [[0, 1, 2, 3, 4]], -2),
        ("c1cc[cH-]c1", [[0, 1, 2]], -1),
        ("Cc1ccccc1", [[1, 2, 3]], 0),
        ("N([O-])[O-]", [[2]], -2),
        ("N([O-])[O-]", [[0]], -1),
        ("[H]", [[0]], -1),
    ):
        assert O.ligand_ionic_charge(smiles, donors, model="training") == (expected, "ok")

    label, why = O.os_from_smiles("[Fe+2](<-[Cl-])(<-[Cl-])<-[C]1NCCN1", 0, "Fe")
    ligands = [_L("[C]1NCCN1", [[0]]), _L("[Cl-]", [[0]]), _L("[Cl-]", [[0]])]
    assert label == 2 and why == "ok"
    assert O.deduce_oxidation_state("Fe", 0, ligands, model="training")[0] == label

    for full, charge, ligand, donors in (
        ("[Fe+2]<-N([O-])[O-]", 0, "N([O-])[O-]", [[0]]),
        ("[Fe+2]<-[O-]N[O-]", 0, "N([O-])[O-]", [[1]]),
        ("[H][Fe+2]", 2, "[H]", [[0]]),
    ):
        lbl, w = O.os_from_smiles(full, charge, "Fe")
        assert w == "ok" and O.deduce_oxidation_state("Fe", charge, [_L(ligand, donors)], model="training")[0] == lbl

    assert O.os_from_smiles("CC[N+](=O)[O-]", 0, "Fe")[0] is None
    assert O.os_from_smiles("[Fe+2]<-[Cl-]", 1, "Fe") == (2, "ok")
    assert O.os_from_smiles("[Fe+3]<-[Cl-]", 1, "Fe") == (None, "charge_mismatch")
    assert O.os_from_header("Gd", 0, 8)[0] == 3
    assert O.os_from_header("Gd", 0, 1)[0] is None


def test_shannon_fe_shifts():
    assert SH.radius_shift("Fe", (2, 5), (2, 1), "CN6") == pytest.approx(-0.17, abs=1e-6)
    assert SH.radius_shift("Fe", (2, 5), (3, 6), "CN6") == pytest.approx(-0.135, abs=1e-6)
