"""S_orient (QO), chelate-aware embed, Q1 trans keys, angle bends, haptic planes, and donor typing."""

import copy
import json
import math
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem
from scipy.spatial.transform import Rotation

from fit import qo as QO
from jacobik.stats import orientation as O

FIX_40 = Path(__file__).parent / "regression" / "fixtures" / "tmqm_sample40"


def _cl2_en_builder():
    from jacobik import ComplexBuilder, LigandInput

    return ComplexBuilder(
        "Pt", "square_plan", charge=0, spin=1,
        ligands=[LigandInput("NCCN", donors=[[0], [3]]), LigandInput("[Cl-]", donors=[[0]]), LigandInput("[Cl-]", donors=[[0]])],
    )


def _bend_fk():
    from jacobik import ComplexBuilder, LigandInput
    from jacobik.ik.fk import ForwardKinematics
    from jacobik.topology.decompose import decompose_ligand

    b = ComplexBuilder(
        "Pt", "square_plan", charge=0, spin=1,
        ligands=[LigandInput("C[B-](C)(n1cccn1)n1cccn1", donors=[[7], [12]]),
                 LigandInput("[Cl-]", donors=[[0]]), LigandInput("[Cl-]", donors=[[0]])],
    )
    mol, _, _ = b._build_shell(False, False)
    for lig in mol.ligands:
        decompose_ligand(mol, lig)
    fk = ForwardKinematics(mol, mol.ligands[0], torch.device("cpu"), angle_bends=True)
    X0 = torch.tensor(np.array([a.position for a in mol.atoms]), dtype=torch.float32)
    return mol, fk, X0


def _graph(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    sym = {a.GetIdx(): a.GetSymbol() for a in m.GetAtoms()}
    adj = {a.GetIdx(): [] for a in m.GetAtoms()}
    for b in m.GetBonds():
        i, j, o = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()
        adj[i].append((j, o))
        adj[j].append((i, o))
    return sym, adj, {a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()}


def test_orientation_features_and_qo_db_roundtrip(tmp_path):
    from jacobik.stats.stats import StatsLookup, load_db

    rng = np.random.default_rng(0)
    M, D = np.zeros(3), np.array([0.0, 0.0, 2.1])
    rand_unit = lambda: (lambda v: v / np.linalg.norm(v))(rng.normal(size=3))
    X = [D + 1.4 * rand_unit() for _ in range(3)]
    g = O.features(M, D, X)
    assert len(g) == 6 and g[0] <= g[1] <= g[2]
    for perm in ([1, 0, 2], [2, 1, 0], [0, 2, 1]):
        assert np.allclose(O.features(M, D, [X[i] for i in perm]), g)
    for k in (1, 2, 3):
        Xk = [D + np.array([math.cos(t), math.sin(t), 0.5]) for t in np.linspace(0, 2, k)]
        assert len(O.features(M, D, Xk)) == O.DIM[k]

    keys = O.level_keys("O_water", "Fe", "octahedral", 0, 2, 0)
    assert keys[0] == (2, 0, "O_water", "Fe", "octahedral", 0) and keys[-2:] == [(2, 0), (2,)]
    assert all(k[0] == 2 for k in keys)

    rng1 = np.random.default_rng(1)
    df = pd.DataFrame([
        dict(csd_code=f"C{i:05d}", hb_min=np.nan if i % 10 else 2.0, ring=0, sub="O_water", metal="Fe",
             geom="octahedral", anion=0, k=2, sub_elems="HH", g0=-0.4 + 0.05 * rng1.normal(),
             g1=-0.3 + 0.05 * rng1.normal(), g2=-0.2 + 0.05 * rng1.normal(), g3=np.nan, g4=np.nan, g5=np.nan)
        for i in range(400)
    ])
    a, b = QO.fit_orientation(df, n_jobs=1), QO.fit_orientation(df, n_jobs=1)
    assert a.keys() == b.keys() and len(a) > 0
    for k in a:
        assert np.allclose(a[k]["means"], b[k]["means"])
    top = a["QO:(2, 0, 'O_water', 'Fe', 'octahedral', 0)"]
    assert top["n"] == 360 and top["type"] == "mvgmm" and np.isclose(sum(top["weights"]), 1.0)
    assert np.all(np.linalg.eigvalsh(np.asarray(top["covs"][0])) >= O.COV_FLOOR * 0.999)

    p = tmp_path / "db.json"
    p.write_text(json.dumps(a))
    st = StatsLookup(load_db(p))
    assert st.has_orientation
    assert st.orientation_dist("O_water", "Fe", "octahedral", 0, 2, 0)[0] == (2, 0, "O_water", "Fe", "octahedral", 0)
    key2, _ = st.orientation_dist("O_water", "Zn", "octahedral", 0, 2, 0)
    assert key2 is not None and "Zn" not in key2

    p_old = tmp_path / "old.json"
    p_old.write_text(json.dumps({"Q1:('Fe', 5, 99, 'O_water', 'octahedral', 'O_water')": {"type": "gmm", "n": 100, "means": [2.1], "vars": [0.01], "weights": [1.0]}}))
    assert not StatsLookup(load_db(p_old)).has_orientation


def test_chelate_embed_cis_order_stereo_and_caps():
    from jacobik.builder import embed as CE

    for smiles, donors in (("CCC([C-]=O)=C([C-]=O)CC", [[3], [6]]), ("COC([NH-])=C([NH-])C#N", [[3], [5]])):
        for seed in (0, 2, 5):
            mol, P, _ = CE.embed(smiles, donors, seed=seed)
            assert np.linalg.norm(P[donors[0][0]] - P[donors[1][0]]) < 3.2
            assert mol.GetNumAtoms() == P.shape[0]

    assert CE.embed("c1ccncc1", [[3]]) is None
    r = CE.embed("[O-]c1cccc2cccnc12", [[0], [9]])
    ref = Chem.AddHs(Chem.MolFromSmiles("[O-]c1cccc2cccnc12"))
    assert [a.GetSymbol() for a in r[0].GetAtoms()] == [a.GetSymbol() for a in ref.GetAtoms()]
    assert CE.cap_anions(Chem.MolFromSmiles("C[B-](C)(C)C"))[1] == []

    for smiles, donors, quad, cis in (
        ("CCCN=Cc1ccccn1", [[10], [3]], (2, 3, 4, 5), False),
        ("CC(C=C([O-])C(F)(F)F)=Nc1c(C)cccc1C", [[9], [4]], (0, 1, 9, 10), True),
    ):
        for seed in (1, 42):
            _, P, info = CE.embed(smiles, donors, seed=seed)
            assert info["ez"] >= 1
            a, b, c, d = (P[i] for i in quad)
            b1 = c - b
            v = (a - b) - ((a - b) @ b1) / (b1 @ b1) * b1
            w = (d - c) - ((d - c) @ b1) / (b1 @ b1) * b1
            assert (np.dot(v, w) > 0) == cis

    assert CE.chelate_ring_stereo(Chem.MolFromSmiles("CCC/N=C/c1ccccn1"), [[10], [3]]) == []
    assert CE.chelate_ring_stereo(Chem.MolFromSmiles("NCCCC=CCCCN"), [[0], [9]]) == []

    _, P_ac, _ = CE.embed("CCC([C-]=O)=C([C-]=O)CC", [[3], [6]], seed=1)
    u, v = P_ac[4] - P_ac[3], P_ac[2] - P_ac[3]
    assert math.degrees(math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))) < 150.0

    for smiles, donors, triple, lo, hi in (
        ("[CH]c1ccccc1OC(C)C", [[0], [7]], ("H", 0, 1), 105, 130),
        ("C(#Cc1ccccc1)c1ccccc1", [[1], [0]], (2, 1, 0), 135, 165),
        ("CC(C)c1cccc(C(C)C)c1N1[C]C2(C)CCC1(C)CC2C", [[13]], (12, 13, 14), 100, 125),
    ):
        mh, P, info = CE.embed(smiles, donors)
        assert info["caps"] >= 1 and len(P) == mh.GetNumAtoms()
        a, b, c = triple
        if a == "H":
            a = next(n.GetIdx() for n in mh.GetAtomWithIdx(b).GetNeighbors() if n.GetAtomicNum() == 1)
        uw, ww = P[a] - P[b], P[c] - P[b]
        assert lo < math.degrees(math.acos(float(uw @ ww / np.linalg.norm(uw) / np.linalg.norm(ww)))) < hi

    mh_me, P_me, _ = CE.embed("[CH3-]", [[0]])
    hs = [n.GetIdx() for n in mh_me.GetAtomWithIdx(0).GetNeighbors()]
    for i in range(3):
        for j in range(i + 1, 3):
            u_h, w_h = P_me[hs[i]] - P_me[0], P_me[hs[j]] - P_me[0]
            assert 100 < math.degrees(math.acos(float(u_h @ w_h / np.linalg.norm(u_h) / np.linalg.norm(w_h)))) < 118

    for smiles, c_idx, x_idx, lo, hi in (("CN1CC(C)(C)C[C]1", 7, 1, 1.24, 1.36), ("C[C]OC", 1, 2, 1.24, 1.36)):
        capped, caps = CE.cap_anions(Chem.MolFromSmiles(smiles), [[c_idx]])
        assert len(caps) == 1 and capped.GetAtomWithIdx(caps[0]).GetAtomicNum() == 1 and capped.GetAtomWithIdx(x_idx).GetFormalCharge() == 1
        _, P_y, _ = CE.embed(smiles, [[c_idx]])
        assert lo < float(np.linalg.norm(P_y[c_idx] - P_y[x_idx])) < hi

    for smiles, donors, capped_n in (
        ("c1cc[cH-]c1", [[0, 1, 2, 3, 4]], 0),
        ("Cc1c(C)c(C)[c-](C)c1C", [[6]], 0),
        ("CC(C)(C)P1CCCCCCCCCCCCCCP(C(C)(C)C)Oc2[c-]c(ccc2)O1", [[19], [4], [26]], 1),
        ("c1ccc2[n-]ccc2c1", [[4]], 1),
    ):
        assert len(CE.cap_anions(Chem.MolFromSmiles(smiles), donors)[1]) == capped_n


def test_solver_q1_trans_orientation_switch_and_angle_bends(stats, tmp_path):
    import jacobik.ik.scorer as S
    from jacobik import ComplexBuilder, LigandInput
    from jacobik.ik import angle_bends as AB
    from jacobik.ik.fk import ForwardKinematics
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.kernels import wrap_angle
    from jacobik.stats.stats import StatsLookup, load_db
    from jacobik.topology.decompose import decompose_ligand

    seen = {}
    orig = S.Scorer._build_q1_per_candidate

    def spy(self, mol, st, asg, dg, ideal, dev):
        r = orig(self, mol, st, asg, dg, ideal, dev)
        if r is not None:
            seen["asg"], seen["ideal"] = asg, ideal
        return r

    b = _cl2_en_builder()
    S.Scorer._build_q1_per_candidate = spy
    try:
        r1 = b.solve(stats=stats, seed=1, embed="etkdg", beam_width=4, n_coord_samples=2, n_ik_steps=10)
    finally:
        S.Scorer._build_q1_per_candidate = orig
    ideal = np.asarray(seen["ideal"])
    for a in seen["asg"]:
        assert np.dot(ideal[a[0]], ideal[a[1]]) > -0.5

    if stats.has_orientation:
        assert r1.scorer.orientation_model == "sorient"
        r2 = b.solve(stats=stats, seed=1, embed="etkdg", beam_width=4, n_coord_samples=2, n_ik_steps=10, orientation_model="q3q4")
        assert r2.scorer.orientation_model == "q3q4"

    p_empty = tmp_path / "empty_db.json"
    p_empty.write_text("{}")
    with pytest.raises(ValueError):
        b.solve(stats=StatsLookup(load_db(p_empty)), seed=1, embed="etkdg", beam_width=2, n_coord_samples=2, n_ik_steps=2, orientation_model="sorient")

    assert b.solve(stats=stats, seed=1, embed="etkdg", beam_width=4, n_coord_samples=2, n_ik_steps=10, angle_bends=False).scorer._q7 is None
    on = b.solve(stats=stats, seed=1, embed="etkdg", beam_width=4, n_coord_samples=2, n_ik_steps=10, angle_bends=True, bend_penalty="angle")
    assert on.scorer._q7 is not None
    X_on = torch.tensor(np.array([a.position for a in on.conformers[0].complex.atoms]), dtype=torch.float32)[None]
    _, br = on.scorer.score(X_on)
    assert "Q7" in br and float(br["Q7"]) <= 0.0

    k_t = torch.tensor([1300.0, 310.0])
    beta = torch.tensor([[math.radians(1.5), math.radians(-1.0)], [math.radians(10.0), 0.0]])
    pr = AB.energy_prior(beta, k_t)
    expected = -0.5 * 1300.0 * math.radians(8.0) ** 2 / AB.RT_KJ
    assert float(pr[0]) == 0.0 and abs(float(pr[1]) - expected) < 1e-4
    assert abs(float(AB.energy_prior(beta, k_t, scale=0.3)[1]) - 0.3 * expected) < 1e-4

    b_q5 = ComplexBuilder("Pt", "square_plan", charge=0, spin=1,
                          ligands=[LigandInput("CCOC(=O)CCN=Cc1ccccn1", donors=[[7], [14]]), LigandInput("[Cl-]", donors=[[0]]), LigandInput("[Cl-]", donors=[[0]])])
    mol_q5, _, _ = b_q5._build_shell(False, False)
    for lig in mol_q5.ligands:
        decompose_ligand(mol_q5, lig)
    ge = GeometryExtractor(mol_q5, torch.device("cpu"))
    X0_q5 = torch.tensor(np.array([a.position for a in mol_q5.atoms]), dtype=torch.float32)
    fk_q5 = ForwardKinematics(mol_q5, mol_q5.ligands[0], torch.device("cpu"))
    rows, offs = fk_q5.q5_alignment(ge, X0_q5)
    assert fk_q5.n_joints_base > 2 and bool((rows >= 0).all())
    target = torch.linspace(-2.5, 2.5, fk_q5.n_joints_base)
    q5 = ge.q5_torsions(fk_q5.apply(torch.cat([torch.zeros(6), wrap_angle(target + offs)])[None], X0_q5))[0]
    assert torch.allclose(wrap_angle(q5[rows] - target), torch.zeros_like(target), atol=1e-3)


def test_site_symmetry_and_embed_aware_dedup(stats):
    import itertools

    import jacobik.builder.engine as E
    from jacobik import ComplexBuilder, LigandInput
    from jacobik.ik.site_enumeration import (
        dedup_combined,
        dedup_exact,
        proper_site_symmetry_group,
        site_symmetry_group,
        topological_class_keys,
    )

    assert (len(site_symmetry_group("octahedral")), len(proper_site_symmetry_group("octahedral"))) == (48, 24)
    assert (len(site_symmetry_group("tetrahedral")), len(proper_site_symmetry_group("tetrahedral"))) == (24, 12)
    assert len(proper_site_symmetry_group("square_plan")) == len(site_symmetry_group("square_plan"))

    raw = [np.array(p) for p in itertools.permutations(range(6))]
    assert len(dedup_exact(raw, [], "octahedral").assignments) == 30
    assert len(dedup_combined(raw, [], "octahedral").assignments) == 15
    assert len(dedup_exact(raw, [[[d] for d in range(6)]], "octahedral").assignments) == 1
    keys = topological_class_keys(dedup_exact(raw, [], "octahedral").assignments, [], "octahedral")
    assert len(set(keys)) == 15 and all(keys.count(k) == 2 for k in set(keys))

    smi = "Cc1ccc([B-](c2cccc([B-](c3ccc(C)cc3)(n3cccn3)n3cccn3)n2)(n2cccn2)n2cccn2)cc1"
    b = ComplexBuilder("Sc", "octahedral", charge=0, spin=1,
                       ligands=[LigandInput(smi, donors=[[23], [28], [29], [39], [34]]), LigandInput("C1CCOC1", donors=[[3]])])
    seen = {}
    orig = E.ComplexBuilder._run_joint_ik

    def spy(self, ctx, fks, scorer, ge, X_full, inits, *a, **k):
        seen["rows"] = inits[0].shape[0]
        return orig(self, ctx, fks, scorer, ge, X_full, inits, *a, **k)

    E.ComplexBuilder._run_joint_ik = spy
    try:
        new = b.solve(stats=stats, seed=1, embed="etkdg", beam_width=4, n_coord_samples=2, n_ik_steps=2)
        rows_new = seen["rows"]
        old = b.solve(stats=stats, seed=1, embed="etkdg", beam_width=4, n_coord_samples=2, n_ik_steps=2, embed_aware_dedup=False)
        rows_old = seen["rows"]
    finally:
        E.ComplexBuilder._run_joint_ik = orig
    assert (old.assignment_search is None or old.assignment_search["n_candidates"] == 1) and new.assignment_search["n_candidates"] == 4
    assert rows_new == rows_old == 8

    b_ord = ComplexBuilder("Fe", "tetrahedral", charge=0, spin=1,
                           ligands=[LigandInput("CCP(CC)CCP(CC)CC", donors=[[2], [6]]), LigandInput("[Cl-]", donors=[[0]]),
                                    LigandInput("Cc1c(C)c(C)[c-](C)c1C", donors=[[1, 2, 4, 6, 8]])])
    mol, _, donor_idx_full = b_ord._build_shell(False, False)
    _, groups, _ = b_ord._collect_donor_groups(mol, donor_idx_full)
    sites = [frozenset(d for d in s.donor_indices if d != mol.metal_idx) for s in mol.coord_sites]
    assert sites == [frozenset(g) for _, _, g in groups] and not mol.coord_sites[0].is_haptic and mol.coord_sites[-1].is_haptic


def test_bend_plan_and_fk_pullback_metric():
    mol, fk, X0 = _bend_fk()
    plan = fk._bend_plan
    assert plan is not None and plan.K == fk.n_angle_bends >= 4
    Xb = X0[None].expand(8, -1, -1).clone()
    assert torch.equal(plan.apply(Xb, torch.zeros(8, plan.K)), Xb)
    pos = {a.idx: i for i, a in enumerate(mol.atoms)}
    bi = torch.tensor([[pos[b.i], pos[b.j]] for b in mol.bonds if mol.metal_idx not in (b.i, b.j)])
    bond_len = lambda X: (X[:, bi[:, 0]] - X[:, bi[:, 1]]).norm(dim=-1)
    torch.manual_seed(0)
    beta = (torch.rand(8, plan.K) - 0.5) * 1.2
    assert float((bond_len(plan.apply(Xb, beta)) - bond_len(Xb)).abs().max()) < 1e-4
    assert torch.autograd.gradcheck(lambda b: plan.apply(Xb[:2].double(), b), (beta[:2].double().requires_grad_(True),), eps=1e-6, atol=1e-4)

    beta_s = torch.zeros(1, plan.K)
    beta_s[0, 0] = 0.3
    X_s = plan.apply(X0[None], beta_s)[0]
    p = int(plan.spec_piv[0])
    moved = (X_s - X0).norm(dim=-1) > 1e-6
    assert torch.allclose((X0 - X0[p]).norm(dim=-1), (X_s - X_s[p]).norm(dim=-1), atol=1e-5) and int(moved.sum()) > 0 and not bool(moved[p])
    m = int(torch.nonzero(moved)[0])
    v0, v1 = X0[m] - X0[p], X_s[m] - X_s[p]
    assert abs(float(torch.arccos((v0 @ v1) / v0.norm() / v1.norm())) - 0.3) < 1e-4

    theta = torch.zeros(6 + fk.n_joints + fk.n_angle_bends, dtype=X0.dtype)
    jac = torch.autograd.functional.jacobian(lambda z: fk.apply(z[None], X0[None])[0, fk._lig_pos_t], theta)
    assert torch.allclose(fk.jacobian_column_norm_sq(theta[None], X0[None])[0], jac.square().sum(dim=(0, 1)), atol=2e-3, rtol=1e-4)


def test_haptic_sorient_and_eta2_face_plane_rules(stats, monkeypatch):
    from jacobik import ComplexBuilder, LigandInput
    from jacobik.core.chemistry import covalent_radius as r_cov
    from jacobik.ik import angle_bends as AB
    from jacobik.ik import scorer as SC

    base_mu = {0: (0.99, 0.90), 1: (-0.10, -0.02), 2: (0.02, 0.05)}
    orig = type(stats).orientation_dist
    b = ComplexBuilder("Fe", "trigonal", charge=0, spin=1,
                       ligands=[LigandInput("C[c-]1cccc1", donors=[[1, 2, 3, 4, 5]]),
                                LigandInput("[cH-]1cccc1", donors=[[0, 1, 2, 3, 4]]), LigandInput("C=C", donors=[[0, 1]])])
    for feats, per_sub, plane in ((None, False, False), ([0, 1, 2], False, False), ([0, 1, 2], True, True)):
        monkeypatch.setattr(SC, "HAPTIC_PER_SUB", per_sub)
        monkeypatch.setattr(O, "ETA2_FACE_PLANE", plane)
        sel = feats if feats is not None else [0, 1]
        d = len(sel)
        fd = types.SimpleNamespace(weights=[0.6, 0.4], means=[[base_mu[i][0] for i in sel], [base_mu[i][1] for i in sel]],
                                   covs=[(np.eye(d) * 0.01 + 0.002).tolist(), (np.eye(d) * 0.02).tolist()], feats=feats)
        type(stats).orientation_dist = lambda self, sub, metal, geom, anion, k, ring: ((k, ring, sub), fd) if k >= O.HAPTIC_K0 else (None, None)
        try:
            r = b.solve(stats=stats, seed=1, embed="etkdg", beam_width=2, n_coord_samples=2, n_ik_steps=4, orientation_model="sorient")
        finally:
            type(stats).orientation_dist = orig
        assert r.scorer._so_n == 3
        mol = r.conformers[0].complex
        got = float(r.scorer._sorient(torch.tensor(np.array([a.position for a in mol.atoms]), dtype=torch.float32)[None])[0])
        atoms = {a.idx: a for a in mol.atoms}
        M = np.asarray(atoms[mol.metal_idx].position, float)
        P = lambda i: np.asarray(atoms[i].position, float)
        dist = {"weights": fd.weights, "means": fd.means, "covs": fd.covs}
        ref = []
        for s in O.haptic_site_specs(mol):
            F = np.array([P(a) for a in s["face"]], float)
            if per_sub:
                face = [(P(a), P(x)) for a, x in s["subs"]]
                ref.append(float(np.mean([QO.mixture_logpdf(np.asarray(O.haptic_features(M, F, s["closed"], [(P(a), P(x))], face))[sel], dist) / d for a, x in s["subs"]])))
            else:
                ref.append(QO.mixture_logpdf(np.asarray(O.haptic_features(M, F, s["closed"], [(P(a), P(x)) for a, x in s["subs"]]))[sel], dist) / d)
        assert len(ref) == 3 and abs(got - float(np.mean(ref))) < 1e-3

    F = np.array([[0.67, 0.0, 0.0], [-0.67, 0.0, 0.0]])
    t = np.radians(20.0)
    sub = lambda i, phi: F[i] + 1.08 * np.array([(1.0 if i == 0 else -1.0) * np.cos(phi) * np.cos(t), np.sin(phi) * np.cos(t), -np.sin(t)])
    ph = np.radians(60.0)
    good = [(F[0], sub(0, ph)), (F[0], sub(0, -ph)), (F[1], sub(1, ph)), (F[1], sub(1, -ph))]
    fold = [good[0], (F[0], sub(0, np.radians(80.0))), good[2], good[3]]
    M_on, M_off = np.array([0.0, 0.0, 2.1]), np.array([0.0, 1.2, 1.7])

    def bends(M, S, on):
        monkeypatch.setattr(O, "ETA2_FACE_PLANE", on)
        return np.array([O.haptic_features(M, F, False, [s], S)[1] for s in S])

    assert np.allclose(bends(M_on, fold, False)[:2], bends(M_on, good, False)[:2], atol=0.03)
    g = bends(M_on, good, True)
    assert np.allclose(g, -np.sin(t), atol=1e-9) and np.allclose(bends(M_off, good, True), g, atol=1e-9)
    assert O.haptic_features(M_off, F, False, good)[0] < 0.9 and np.abs(bends(M_on, fold, True) - g).max() > 0.1
    assert O.eta2_plane_normal(F, good[:3]) is None and np.allclose(bends(M_off, good[:3], True), bends(M_off, good[:3], False), atol=1e-12)

    rng = np.random.default_rng(3)
    for _ in range(25):
        F3 = rng.normal(size=(2, 3))
        S3 = [(F3[i], F3[i] + rng.normal(size=3)) for i in (0, 0, 1, 1)]
        n3 = O.eta2_plane_normal(F3, S3)
        bt = torch.tensor((F3[1] - F3[0]) / np.linalg.norm(F3[1] - F3[0]))
        nt = AB.eta2_plane_normal_t(bt, torch.tensor(np.array([a for a, _ in S3])), torch.tensor(np.array([x for _, x in S3])), torch.ones(4, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)).numpy()
        assert abs(abs(float(n3 @ nt)) - 1.0) < 1e-9

    X_pl = np.array([[0.67, 0, 0], [-0.67, 0, 0], [1.2, 0.9, -0.3], [1.25, -0.9, -0.35], [-1.2, 0.9, -0.3], [-1.2, -0.9, -0.3], [0, 0.1, 2.1]])
    plane = ((0, 2), (0, 3), (1, 4), (1, 5))
    bp = AB.BendPlan([(0, 2, 6, [2], ("haptic", 2, (0, 1), plane)), (0, 3, 6, [3], ("haptic", 2, (0, 1), plane))], {i: i for i in range(7)}, "cpu")
    ns = O.eta2_plane_normal(X_pl[:2], [(X_pl[a], X_pl[x]) for a, x in plane])
    Xt_pl = torch.tensor(X_pl, dtype=torch.float32)[None]
    for col, h in ((0, 2), (2, 3)):
        for c, lean in ((col, True), (col + 1, False)):
            beta = torch.zeros(1, bp.K)
            beta[0, c] = 0.01
            u0 = X_pl[h] - X_pl[0]
            dz = float(((bp.apply(Xt_pl, beta)[0, h].double().numpy() - X_pl[0]) - u0) @ ns) / np.linalg.norm(u0)
            assert (abs(dz) > 0.009) if lean else (abs(dz) < 1e-4)

    assert AB.free_zone([(0, 2, 6, [2], ("haptic", 2, (0, 1), plane)), (0, 2, 6, [2], ("haptic", 2, (0, 1), ())),
                         (0, 2, 6, [2], ("haptic", 2, (0, 1)))], free_deg=2.0) == [AB.LEAN_FREE_DEG_PLANE, 2.0, AB.lean_free_deg()[2], 2.0, AB.lean_free_deg()[2], 2.0]

    rC, rS = r_cov("Fe") + r_cov("C"), r_cov("Fe") + r_cov("S")
    face_cs = [[0.8, 0.0, 1.0 * rC], [-0.8, 0.0, 1.15 * rS]]
    assert O.eta2_bonded(face_cs, "Fe", ["C", "S"]) and float(np.linalg.norm(face_cs[1])) / rC > O.ETA2_BOND_RATIO_MAX
    assert not O.eta2_bonded([[0.7, 0.0, 1.0 * rC], [-0.7, 0.0, 1.4 * rC]], "Fe", ["C", "C"])

    def eta2(d2):
        F2 = np.array([[0.7, 0.0, 2.0], [-0.7, 0.0, d2]])
        subs = [(0, F2[0] + [0.5, 0.9, 0.3]), (0, F2[0] + [0.5, -0.9, 0.3]), (1, F2[1] + [-0.5, 0.9, 0.3]), (1, F2[1] + [-0.5, -0.9, 0.3])]
        return dict(csd_code="X", hb_min=np.nan, ring=0, sub="C_haptic", metal="Fe", geom="octahedral", anion=0, k=O.HAPTIC_K0 + 2,
                    sub_elems="", g0=0.5, g1=0.5, g2=0.5, g3=np.nan, g4=np.nan, g5=np.nan, raw_face=F2.ravel().tolist(),
                    raw_face_elems=["C", "C"], raw_closed=False, raw_subs=[v for i, x in subs for v in [float(i)] + list(x)])

    df_fit = pd.DataFrame([eta2(2.0), eta2(1.5 * rC), dict(csd_code="Y", hb_min=np.nan, ring=0, sub="N_amine", metal="Fe", geom="octahedral", anion=0, k=1, sub_elems="C", g0=0.1, g1=0.2, g2=np.nan, g3=np.nan, g4=np.nan, g5=np.nan, raw_face=None, raw_face_elems=None, raw_closed=None, raw_subs=None)])
    monkeypatch.setattr(O, "ETA2_FACE_PLANE", False)
    pd.testing.assert_frame_equal(QO.fit_rows(df_fit), pd.concat([df_fit, QO.per_sub_rows(df_fit)], ignore_index=True))
    monkeypatch.setattr(O, "ETA2_FACE_PLANE", True)
    out = QO.fit_rows(df_fit)
    assert len(out[out.k == O.HAPTIC_K0 + 2]) == 2 and (out.k == 1).sum() == 1
    assert (out.k == O.HAPTIC_K0 + 2 + O.PER_SUB_OFFSET["H"]).sum() == 8

    # per_sub_rows & eta2_group_bend geminal angle
    X_g = np.array([[0.7, 0, 0], [-0.7, 0, 0], [1.25, 0.9, -0.3], [1.25, -0.9, -0.3], [-1.9, 0, 0], [0, 0.2, 2.2]])
    hch = lambda Y: float(np.degrees(np.arccos((Y[2] - Y[0]) @ (Y[3] - Y[0]) / np.linalg.norm(Y[2] - Y[0]) / np.linalg.norm(Y[3] - Y[0]))))
    a0 = hch(X_g)
    grp = AB.BendPlan([(0, 2, 5, [2, 3], ("haptic", 2, (0, 1), ()))], {i: i for i in range(6)}, "cpu")
    for beta in ([0.9, 0.0], [0.0, 0.5], [-1.0, 0.7]):
        assert abs(hch(grp.apply(torch.tensor(X_g, dtype=torch.float32)[None], torch.tensor([beta], dtype=torch.float32))[0].double().numpy()) - a0) < 1e-3
    assert AB.ETA2_GROUP_BEND is True


def test_donor_typing_carbon_resonance_ether_and_nitro():
    from jacobik.topology.donor_classifier import (
        _classify_carbon_structural,
        _classify_oxygen_structural,
        _resonance_donor_labels,
    )

    for smiles, idx, expected in (
        ("[C-]#[O+]", 0, "C_carbonyl"), ("[C]=O", 0, "C_carbonyl"),
        ("CN1CCN(C)[C]1", 6, "C_carbene"), ("CN1CC(C)(C)C[C]1", 7, "C_carbene"),
        ("CN1CC(C)(C)C=[C]1", 7, "C_vinyl"), ("C[C]OC", 1, "C_carbene"),
        ("C[C]=O", 1, "C_acyl"), ("CO[C]=O", 2, "C_acyl"),
        ("[CH]c1ccccc1", 0, "C_alkylidene"), ("[CH]=C", 0, "C_vinyl"),
        ("[C]#CC", 0, "C_acetylide"), ("[C]=C(C)C", 0, "C_vinylidene"), ("[C]C(C)(C)C", 0, "C_alkylidyne"),
        ("[CH3]", 0, "C_alkyl"), ("[C-]#N", 0, "C_cyanide"), ("[C-]#[N+]C", 0, "C_isocyanide"),
        ("[c]1ccccc1", 0, "C_pi"),
    ):
        sym, adj, arom = _graph(smiles)
        assert _classify_carbon_structural(idx, sym, adj, arom) == expected

    for kek_a, kek_b, donors, expected in (
        ("CC(=O)C=C(C)[O-]", "CC([O-])=CC(C)=O", (2, 6), {"O_diketonate"}),
        ("CC(=NC)C=C(C)[N-]C", "CC([N-]C)=CC(C)=NC", (2, 7), {"N_diketiminate"}),
        ("CN=C(C)[N-]C", "C[N-]C(C)=NC", (1, 4), {"N_amidinate"}),
        ("[O-]c1ccccc1[O-]", "O=C1C=CC=CC1=O", (0, 7), {"O_dioxolene"}),
    ):
        for smi in (kek_a, kek_b):
            sym, adj, arom = _graph(smi)
            lab = _resonance_donor_labels(list(donors), sym, adj, arom)
            assert set(lab) == set(donors) and set(lab.values()) == expected

    sym, adj, arom = _graph("COc1ccccc1")
    assert _classify_oxygen_structural(1, sym, adj, arom, set()) == "O_ether"
    for smi, o_idx in (("[O-][N+](=O)[O-]", 0), ("O=[N+]([O-])[O-]", 0)):
        sym, adj, arom = _graph(smi)
        assert _classify_oxygen_structural(o_idx, sym, adj, arom, set()) == "O_nitro"


def test_fixture_haptic_groups_crystal_donors_and_raw_faces():
    from fit.parser import TMQMParser
    from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol
    from tools.jdev.cases import donor_mismatches

    mols = list(TMQMParser(FIX_40).stream_split("S"))
    got_hap = {m.csd_code: sorted(s.haptic_order for s in m.coord_sites if s.kind == "haptic")
               for m in mols if m.csd_code in {"XECSON", "AHIGOQ", "MELLUN"}}
    assert got_hap == {"XECSON": [2], "AHIGOQ": [2, 2], "MELLUN": [2]}

    bad = {}
    rows = []
    for m in mols:
        groups = _build_ligand_groups(m, m.smiles)
        lig_in = ligand_inputs_from_mol(m, groups)
        if lig_in is not None:
            mm = donor_mismatches(SimpleNamespace(tmqm_mol=m, ligand_groups=groups, ligand_inputs=lig_in))
            if mm:
                bad[m.csd_code] = mm
        if m.csd_code in ("XECSON", "AHIGOQ"):
            rows.extend(r for r in QO.donor_rows(m) if r["k"] >= O.HAPTIC_K0)
    assert not bad
    assert len(rows) == 3
    for r in rows:
        F = np.asarray(r["raw_face"]).reshape(-1, 3)
        sv = np.asarray(r["raw_subs"]).reshape(-1, 4)
        assert np.allclose(O.haptic_features(np.zeros(3), F, r["raw_closed"], [(F[int(p)], x) for p, *x in sv.tolist()]),
                           [r["g0"], r["g1"], r["g2"]], atol=1e-9)


def test_stats_shape_hygiene_q1_wall_q6_3bond_arc_and_prior(legacy_flags, monkeypatch):
    from fit import fitter as FT
    from jacobik.core.types import Atom, Bond
    from jacobik.ik import scorer as SC
    from jacobik.ik.targets import IDEAL_GEOMETRIES
    from jacobik.stats import hypothesis_tables as HT
    from jacobik.stats import reference_ll as R
    from jacobik.topology import graph as G
    from jacobik.topology.coord_geom import SHAPES_BY_CN, shape_deviations

    rng = np.random.default_rng(0)
    for shapes in SHAPES_BY_CN.values():
        for g in shapes:
            v = IDEAL_GEOMETRIES[g] @ Rotation.random(random_state=3).as_matrix().T
            dev = shape_deviations(v[rng.permutation(len(v))] * 2.1)
            assert dev[g] < 1e-9 and min(dev, key=dev.get) == g

    x = np.concatenate([rng.normal(1.90, 0.03, 250), rng.normal(2.10, 0.10, 250), [3.14]])
    new = FT.fit_gmm(x, 4, var_floor=FT.VAR_FLOOR["Q1"], n_comp_min=FT.N_COMP_MIN)
    assert (np.array(new["weights"]) * new["n"]).min() >= FT.N_COMP_MIN - 1e-6
    assert np.array(new["vars"]).min() >= FT.VAR_FLOOR["Q1"] * 0.999 and max(new["means"]) < 2.6

    monkeypatch.setattr(SC, "GMM_FLOOR_MODE", "scorer")
    floor = math.sqrt(SC._GMM_VAR_MIN["Q1"])
    hi, sd = SC._q1_wall([1.90, 2.10], [0.0001, 0.0144], [0.5, 0.5])
    hi2, sd2 = SC._q1_wall([2.0], [1e-6], [1.0])
    assert abs(hi - (2.10 + SC.Q1_WALL_Z * 0.12)) < 1e-9 and abs(sd - max(0.12, floor)) < 1e-9
    assert abs(hi2 - (2.0 + SC.Q1_WALL_Z * SC.Q1_WALL_SD_MIN)) < 1e-9 and abs(sd2 - floor) < 1e-9

    atoms = [Atom(i, s, np.zeros(3)) for i, s in [(1, "Fe"), (2, "P"), (3, "C"), (4, "C"), (5, "C")]]
    ex = G.q6_topological_exclusions(atoms, [Bond(2, 3, 1.0), Bond(3, 4, 1.0), Bond(4, 5, 1.0), Bond(1, 2, 0.5)], 1, [2])
    assert (1, 2) in ex and (1, 3) in ex and (1, 4) not in ex and (1, 5) not in ex and (2, 5) in ex

    adj_o = {i: ({i + 1} if i < 6 else set()) | ({i - 1} if i > 1 else set()) for i in range(1, 7)}
    bo_o = {(i, i + 1): 1.0 + (i % 2) for i in range(1, 6)}
    assert HT.arc_pattern(range(1, 7), [[1, 2], [3, 4, 5, 6]], adj_o, bo_o)[3] == HT.arc_pattern(range(1, 7), [[1, 2, 3, 4], [5, 6]], adj_o, bo_o)[3] == "2+4"
    assert HT.arc_context([1, 2, 3, 4, 5, 6], [set(range(1, 7))]) == ("ring", 6) and HT.arc_context([1, 2], []) == ("chain", 0)

    rows = [dict(csd_code=f"A{i}", metal="Fe", os=2, cn=5, geom="square_pyr", geom_alt="trig_bipyr", geom_gap=0.5) for i in range(40)]
    rows += [dict(csd_code=f"B{i}", metal="Fe", os=2, cn=5, geom="square_pyr", geom_alt="trig_bipyr", geom_gap=0.0) for i in range(10)]
    G_prior = HT.fit_geometry_prior(pd.DataFrame(rows))["QG:('Fe', 2, 5)"]
    rows2 = rows + [dict(csd_code=f"C{i}", metal="Fe", os=2, cn=5, geom="square_pyr", geom_alt="trig_bipyr", geom_gap=0.2) for i in range(50)]
    rows2 = [r for r in rows2 if r["csd_code"][0] != "B"] + [dict(r, geom_gap=0.01) for r in rows2 if r["csd_code"][0] == "B"]
    G2_prior = HT.fit_geometry_prior(pd.DataFrame(rows2))["QG:('Fe', 2, 5)"]
    assert dict(zip(G_prior["labels"], G_prior["p"]))["square_pyr"] > 0.97 and 0.0 < dict(zip(G2_prior["labels"], G2_prior["p"]))["trig_bipyr"] < 0.1

    mu_g, s_g = R.ref_gmm({"means": [2.0], "vars": [0.09], "weights": [1.0]}, var_floor=0.04)
    assert abs(mu_g - (-0.5 * math.log(2 * math.pi * math.e * 0.09))) < 0.02 and abs(s_g - math.sqrt(0.5)) < 0.05
    assert R.ref_gmm({"means": [2.0], "vars": [0.0001], "weights": [1.0]}, var_floor=0.04)[0] < -0.5 * math.log(2 * math.pi * math.e * 0.0001)
    assert abs(R.ref_vmm({"means": [0.0], "kappas": [1e-6], "weights": [1.0]})[0] + math.log(2 * math.pi)) < 1e-3


def test_threaded_worker_chunks_isolated_scorer_state(stats, monkeypatch):
    import jacobik.ik.ik_solver as IK
    import jacobik.ik.scorer as S

    res = _cl2_en_builder().solve(stats=stats, seed=1, embed="etkdg", beam_width=2, n_coord_samples=2, n_ik_steps=4)
    sc = res.scorer
    before = {q: getattr(sc, f"_{q}_eff_vars", None) for q in ("q1", "q2", "q3", "q4")}
    c = copy.copy(sc)
    c.set_gmm_floor("fit")
    c.set_gmm_floor("scorer")
    assert all(getattr(sc, f"_{q}_eff_vars", None) is v for q, v in before.items())

    monkeypatch.setattr(IK, "_RANK_FLOOR", True)
    monkeypatch.setenv("JACOBIK_NUM_WORKERS", "2")
    monkeypatch.setenv("JACOBIK_WORKER_MODE", "thread")
    seen = []
    orig = S.Scorer.set_gmm_floor

    def spy(self, mode):
        if threading.current_thread() is not threading.main_thread():
            seen.append(id(self))
        return orig(self, mode)

    monkeypatch.setattr(S.Scorer, "set_gmm_floor", spy)
    r = _cl2_en_builder().solve(stats=stats, seed=3, embed="etkdg", beam_width=16, n_coord_samples=16, n_ik_steps=10)
    assert len(set(seen)) >= 2 and id(r.scorer) not in seen
