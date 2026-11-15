"""A4t v2 analysis metrics (coverage package Part A): small synthetic cases for each new test."""
import json

import numpy as np
import pytest

from tools.jgen.a4t import metrics2 as m2
from tools.jgen.a4t.bundle import top1_ref_order
from tools.jgen.bench.pool_export import xyz_text
from tools.runkit import geom


def _chiral_cloud():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 3)) * 2.0
    return X


def test_hit_rmsd_reports_proper_and_global_inversion():
    X = _chiral_cloud()
    P = np.arange(9)[None, :]
    prop, mir = m2.rmsd_proper_mirror((X * np.array([1, 1, -1.0]))[None], X, P)
    assert prop[0] > 0.5 and mir[0] < 1e-6
    prop, mir = m2.rmsd_proper_mirror(X[None], X, P)
    assert prop[0] < 1e-6 and mir[0] > 0.5      # 'mirror' = RMSD of the reflected row; the hit rule uses the minimum


def test_best_alignment_flags_mirror_and_returns_row_in_reference_frame():
    X = _chiral_cloud()
    heavy = np.arange(9)
    P = np.arange(9)[None, :]
    Y = X * np.array([1, 1, -1.0]) + 5.0
    al = m2.best_alignment(Y, X, heavy, P)
    assert al["mirror"] and al["rmsd"] < 1e-6
    al = m2.best_alignment(X + 1.0, X, heavy, P)
    assert not al["mirror"] and al["rmsd"] < 1e-6


def test_verdict_disagreement_across_threshold_is_ambiguous():
    rows = [dict(hit=False, dE_mace=12.0, dE_gfn2=3.0)]
    assert m2.verdict(rows)["status"] == "A"
    assert m2.verdict([dict(hit=False, dE_mace=4.0, dE_gfn2=3.0)])["status"] == "F"
    assert m2.verdict([dict(hit=True, dE_mace=40.0, dE_gfn2=30.0)])["status"] == "F"
    assert m2.verdict([dict(hit=False, dE_mace=40.0, dE_gfn2=30.0)])["status"] == "miss"
    assert m2.verdict([dict(hit=False, dE_mace=None, dE_gfn2=None)])["status"] == "miss"
    cls, miss = m2.decision2("A", False, False, True, False, "R-rot")
    assert cls == "A" and miss == "R-rot"
    assert m2.decision2("miss", False, False, True, True, "R-rot")[0] == "Z"
    assert m2.decision2("miss", False, False, False, True, "R-rot")[0] == "S"


def test_shell_isomer_under_automorphisms_not_free_donor_permutation():
    # M plus four N donors in two chelates (1,2) and (3,4); automorphisms swap only within a chelate.
    ref = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, 0, 2]], float)
    sym = ["Pd", "N", "N", "N", "N"]
    P = np.array([[0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [0, 1, 2, 4, 3], [0, 2, 1, 4, 3]])
    heavy = np.arange(5)
    shell = [0, 1, 2, 3, 4]
    row = ref[[0, 1, 3, 2, 4]]   # donors of the two chelates exchanged: another positional isomer
    from tools.jgen.a4t.metrics import shell_rmsd
    assert shell_rmsd(row, ref, sym, 0, [1, 2, 3, 4]) < 1e-6          # v1 free permutation cannot see it
    assert min(m2.shell_rmsd_auto_any(row, ref, P, heavy, shell)) > 0.5
    assert m2.shell_rmsd_auto(ref[[0, 2, 1, 3, 4]], ref, P, heavy, shell) < 1e-6   # a real automorphism is fine
    assert m2.shell_rmsd_auto(ref * np.array([1, 1, -1.0]), ref, P, heavy, shell, mirror=True) < 1e-6


def _p_centre():
    # P bonded to the metal and to C, N, O (all inequivalent) + a pendant H-free environment.
    sym = ["Pd", "P", "C", "N", "O"]
    X = np.array([[0, 0, 1.9], [0, 0, 0], [1.7, 0, -0.6], [-0.85, 1.5, -0.6], [-0.85, -1.5, -0.6]], float)
    adj = np.zeros((5, 5), bool)
    for a, b in ((1, 2), (1, 3), (1, 4)):
        adj[a, b] = adj[b, a] = True
    return sym, X, adj


def test_signed_volume_flags_a_pyramidal_p_centre_but_not_the_enantiomer():
    sym, X, adj = _p_centre()
    cen = m2.stereocentres(X, sym, 0, [1], adj)
    assert list(cen) == [1]
    flipped_one = X.copy()
    flipped_one[[3, 4]] = flipped_one[[4, 3]]               # swap two substituents: inverted P
    f, n, nd = m2.chirality_flips(X, flipped_one, cen)
    assert f == [1] and n == 1 and nd == 0                 # single centre inverted = global inversion = enantiomer
    sym2 = sym + ["P", "C", "N", "O"]
    # two centres: one inverted only -> diastereomer
    X2 = np.vstack([X, X[1:5] + np.array([8.0, 0, 0])])
    adj2 = np.zeros((9, 9), bool)
    adj2[:5, :5] = adj
    for a, b in ((5, 6), (5, 7), (5, 8)):
        adj2[a, b] = adj2[b, a] = True
    cen2 = m2.stereocentres(X2, sym2, 0, [1, 5], adj2)
    assert sorted(cen2) == [1, 5]
    row = X2.copy()
    row[[3, 4]] = row[[4, 3]]
    f, n, nd = m2.chirality_flips(X2, row, cen2)
    assert n == 2 and nd == 1
    row_all = X2.copy()
    row_all[[3, 4, 7, 8]] = row_all[[4, 3, 8, 7]]
    assert m2.chirality_flips(X2, row_all, cen2)[2] == 0


def test_equivalent_neighbours_are_not_stereocentres():
    sym = ["Pd", "P", "C", "C", "O"]            # P(CH3)(CH3)O: two equivalent C neighbours
    _, X, adj = _p_centre()
    adj = adj.copy()
    assert m2.stereocentres(X, sym, 0, [1], adj) == {}


def test_planar_chelate_ring_that_folds_is_flagged():
    ring = [0, 1, 2, 3, 4]
    ang = np.arange(5) * 2 * np.pi / 5
    flat = np.stack([np.cos(ang), np.sin(ang), np.zeros(5)], 1) * 2.0
    folded = flat.copy()
    folded[[1, 3], 2] = 1.4
    folded[[2, 4], 2] = -1.4
    assert m2.chelate_planarity_flag(flat, flat, [ring]) == []
    out = m2.chelate_planarity_flag(flat, folded, [ring])
    assert len(out) == 1 and out[0][2] > m2.FOLD_ROW_DEG
    # a non-planar reference ring is not subject to the check
    assert m2.chelate_planarity_flag(folded, flat, [ring]) == []


def _cyclohexane(torsions=None):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdForceFieldHelpers as FF
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    if torsions is None:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    else:
        props = FF.MMFFGetMoleculeProperties(mol)
        ff = FF.MMFFGetMoleculeForceField(mol, props)
        order = [0, 1, 2, 3, 4, 5]
        for k, t in enumerate(torsions):
            a, b, c, d = [order[(k + j) % 6] for j in range(4)]
            ff.MMFFAddTorsionConstraint(a, b, c, d, False, t - 1, t + 1, 500.0)
        ff.Minimize(maxIts=5000)
    X = np.array(mol.GetConformer().GetPositions())
    return [a.GetSymbol() for a in mol.GetAtoms()], X


def test_saturated_ring_conformer_split_chair_vs_twist_boat():
    sym, chair = _cyclohexane()
    sym2, twist = _cyclohexane([30, 30, -60, 30, 30, -60])
    X = np.vstack([chair, [[20.0, 0, 0]]])
    sym = sym + ["Pd"]
    metal = len(sym) - 1
    rings = m2.ring_atoms(X, sym, metal, [])
    assert len(rings["sat"]) == 1 and rings["chel"] == [] and not rings["aryl"]
    t = np.vstack([twist, [[20.0, 0, 0]]])
    diff_same = m2.ring_conformer_diff(X, X, rings["sat"])
    diff_tb = m2.ring_conformer_diff(X, t, rings["sat"])
    assert max(diff_same) < 1e-6 and max(diff_tb) > m2.RING_TORS_DEG
    zero = {c: dict(n=0, share=0, frac=0, enrich=0, rms=0.0) for c in ("shell", "chelate", "sat_ring", "aryl", "pendant")}
    assert m2.split_residual(diff_tb, [], zero) == "R-sat"
    assert m2.split_residual(diff_same, [45.0], zero) == "R-chel"
    rot = dict(zero, aryl=dict(n=6, share=0.9, frac=0.3, enrich=3, rms=0.8))
    assert m2.split_residual(diff_same, [], rot) == "R-rot"
    assert m2.split_residual(diff_same, [], zero) == "C"


def test_aromatic_ring_is_aryl_not_saturated():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    X = np.vstack([mol.GetConformer().GetPositions(), [[20.0, 0, 0]]])
    sym = [a.GetSymbol() for a in mol.GetAtoms()] + ["Pd"]
    rings = m2.ring_atoms(X, sym, len(sym) - 1, [])
    assert rings["sat"] == [] and len(rings["aryl"]) == 6


def test_bundle_top1_uses_gen_to_ref_not_its_inverse(tmp_path):
    g2r = [2, 0, 1]                      # a 3-cycle: g2r != inverse permutation
    sym_b = ["C", "N", "O"]
    Xb = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], float)
    ref_sym = [sym_b[j] for j in g2r]
    f = tmp_path / "pool.xyz"
    f.write_text(xyz_text(sym_b, Xb, dict(row_index=0, is_top1=True, score=1.0)))
    man = dict(gen_to_ref_idx=g2r, ref_symbols=ref_sym, rows=[dict(file="pool.xyz")])
    (tmp_path / "m.json").write_text(json.dumps(man))
    sy, X, h = top1_ref_order(tmp_path / "m.json")
    assert np.allclose(X, Xb[g2r]) and not np.allclose(X, Xb[np.argsort(g2r)])
