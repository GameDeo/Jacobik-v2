"""Focused N2 regression checks: coordination geometry is not an H-bond."""
from types import SimpleNamespace

import numpy as np
import pytest

from jacobik.core.types import Atom, Bond, TMComplex
from jacobik.stats import nci
from tools.jdev.budget.nci_census import census


def _pair(bound, donor="N", acceptor="Cl", haptic=False):
    atoms = [Atom(1, "Fe", np.array([0., 3., 0.])),
             Atom(2, donor, np.array([0., 0., 0.])),
             Atom(3, "H", np.array([1., 0., 0.])),
             Atom(4, acceptor, np.array([3., 0., 0.]))]
    # Raw metal edges are deliberately present even when not declared donors.
    mol = TMComplex("NCI_TEST", 0, 1, "", 0, "", atoms,
                    [Bond(2, 3, 1.), Bond(1, 2, .2), Bond(1, 4, .2)], 1, "Fe")
    groups = [list(bound)] if haptic else [[i] for i in bound]
    mol.coord_sites = [SimpleNamespace(donor_indices=g) for g in groups]
    return mol


@pytest.mark.parametrize("bound", [(), (2,), (4,), (2, 4)])
@pytest.mark.parametrize("donor,acceptor", [("N", "Cl"), ("O", "N")])
def test_only_cobound_hbond_is_removed_from_runtime_extraction_and_census(bound, donor, acceptor):
    mol = _pair(bound, donor, acceptor)
    expected = 0 if len(bound) == 2 else 1
    candidates = nci.candidates(mol)
    assert len(candidates["hb"]) == expected
    assert candidates["pp"] == candidates["chp"] == []
    assert census(mol)["hbond"] == expected
    records = nci.records(mol)
    assert len(records) == 3 * expected  # same interaction under its three ladder keys
    if expected:
        assert candidates["hb"][0][:3] == (2, 3, 4)
        assert all(len(r) == len(a) == 1 for r, a in records.values())


def test_all_haptic_face_atoms_count_as_bound():
    assert nci.candidates(_pair((2, 4), haptic=True))["hb"] == []


def test_census_ignores_nonpolar_h_without_a_ligand_neighbour():
    mol = _pair(())
    mol.atoms.append(Atom(5, "H", np.array([0., 2., 0.])))
    mol.bonds.append(Bond(1, 5, .2))
    counts = census(mol)
    assert counts["hbond"] == 1 and counts["agostic"] == 0


@pytest.mark.parametrize("carbons,expected", [(0, 0), (2, 0), (3, 1)])
def test_near_ligand_pairs_still_excluded_and_five_bonds_retained(carbons, expected):
    mol = _pair((), acceptor="N")
    acceptor = 4 + carbons
    mol.atoms = mol.atoms[:3] + [Atom(i, "C", np.array([float(i), 0., 0.]))
                               for i in range(4, acceptor)] + [Atom(acceptor, "N", np.array([3., 0., 0.]))]
    chain = [2, *range(4, acceptor + 1)]
    mol.bonds = [Bond(2, 3, 1.)] + [Bond(i, j, 1.) for i, j in zip(chain, chain[1:])]
    # H-D-(C...)A is 2, 4 or 5 ligand bonds apart.
    assert len(nci.candidates(mol)["hb"]) == expected
