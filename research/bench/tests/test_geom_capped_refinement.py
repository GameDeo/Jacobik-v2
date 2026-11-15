"""geom.sym_rmsd_to_ref on a capped automorphism group: the refinement permutes only within element + colour classes and
keeps a result only if the permutation is a graph automorphism (it used to be an element-blind Hungarian)."""
import numpy as np
import pytest

from tools.runkit import geom

SYM = ["N", "C", "C", "C", "O", "Pd"]
REF = np.array([[0, 0, 0], [1.5, 0, 0], [2.2, 1.3, 0], [2.2, -1.3, 0], [-1.3, 0.8, 0], [0, -4, 0]], float)


@pytest.fixture
def capped(monkeypatch):
    monkeypatch.setattr(geom, "CAP", 1)               # P = [identity] already counts as a truncated group
    P = np.arange(6)[None, :]
    return P, geom.heavy_graph(REF, SYM, 5)


def test_swapping_two_elements_must_not_lower_the_rmsd(capped):
    P, g = capped
    Y = REF.copy()
    Y[[0, 4]] = Y[[4, 0]]                              # N and O exchange positions: a different molecule
    plain = geom.sym_rmsd_to_ref(Y[None], REF, P)[0]
    refined = geom.sym_rmsd_to_ref(Y[None], REF, P, graph=g)[0]
    assert plain > 0.5
    assert refined == pytest.approx(plain)             # the old blind Hungarian returned ~0 here


def test_equivalent_atoms_are_still_refined(capped):
    P, g = capped
    Y = REF.copy()
    Y[[2, 3]] = Y[[3, 2]]                              # the two terminal methyls of C2 are symmetry equivalent
    assert geom.sym_rmsd_to_ref(Y[None], REF, P)[0] > 0.5
    assert geom.sym_rmsd_to_ref(Y[None], REF, P, graph=g)[0] < 1e-6


def test_non_automorphism_is_rejected(capped):
    P, g = capped
    sym = ["N", "C", "C", "C", "C", "Pd"]              # C3 and C4 same class but only if both hang on C2
    X = REF.copy()
    X[4] = [-1.3, 0.8, 0]                              # a fifth carbon on N: classes N-C vs C2-C differ
    g2 = geom.heavy_graph(X, sym, 5)
    Y = X.copy()
    Y[[2, 4]] = Y[[4, 2]]
    plain = geom.sym_rmsd_to_ref(Y[None], X, P)[0]
    assert geom.sym_rmsd_to_ref(Y[None], X, P, graph=g2)[0] == pytest.approx(plain)


def test_union_of_p_and_inverse_used_when_capped(monkeypatch):
    monkeypatch.setattr(geom, "CAP", 2)
    cyc = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 3, 4, 5]])     # a 3-cycle without its inverse
    Y = REF.copy()
    Y[[0, 1, 2]] = REF[[2, 0, 1]]
    r_union = geom.sym_rmsd_to_ref(Y[None], REF, cyc)[0]
    r_one = np.sqrt(geom._min_ss(*(lambda J, b: (J, (J ** 2).sum((1, 2)), b, float((b ** 2).sum()), geom._inv_perms(cyc), None))(
        (Y[None] - Y[None].mean(1, keepdims=True)), REF - REF.mean(0))) / 6)[0]
    assert r_union <= r_one + 1e-12
