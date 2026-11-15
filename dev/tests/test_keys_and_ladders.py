"""Keys, ladders, refit mixture/pooling models, linear-atom FK stability, Q6 PMF, and fit pipeline guards."""
from __future__ import annotations

import ast
import inspect
import json
import math
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

FIX = Path(__file__).parent / "regression" / "fixtures" / "tmqm_sample40"
FIX_ROOT = Path(__file__).resolve().parent / "regression" / "fixtures"
SP_CASES = {"XECSON": ("tmqm_sample40", "S"), "TINBIE": ("tmqm_sample40", "S"), "CEXZIO": ("ring10", "N"), "AHIDAX": ("n48", "N")}


@pytest.fixture(scope="module")
def fixture_mols():
    from fit.parser import TMQMParser

    return {m.csd_code: m for m in TMQMParser(FIX).stream_split("S")}


def _mol(code):
    from fit.parser import TMQMParser

    folder, split = SP_CASES[code]
    return next(m for m in TMQMParser(FIX_ROOT / folder).stream_split(split) if m.csd_code == code)


@cache
def _fixture_pmf():
    from fit import q6p as q6pmf
    from fit.parser import TMQMParser

    acc = q6pmf.HistAccumulator()
    for m in TMQMParser(FIX).stream_split("S"):
        acc.add(m)
    ent = q6pmf.fit_tables(acc.h, n_min=500)
    return acc, ent, {ast.literal_eval(k.split(":", 1)[1]): v for k, v in ent.items()}


def test_mixture_evidence_weights_and_map_adaptation():
    from fit import fitter as FT
    from fit import pooling as PL
    from jacobik.stats import ladders as L
    from jacobik.stats import mixture as MX
    from jacobik.stats.stats import StatsLookup, load_db

    rng = np.random.default_rng(0)
    x = np.cos(np.deg2rad(np.r_[rng.normal(90, 4, 35), rng.normal(175, 3, 12)]))
    f = MX.fit_mixture(x, np.ones_like(x), "gauss", k_max=6, var_floor=0.002 ** 2, n_comp_min=10)
    assert f["K"] == 2
    ang = sorted(np.degrees(np.arccos(np.clip(f["mu"], -1, 1))))
    assert abs(ang[0] - 90) < 5 and abs(ang[1] - 175) < 8

    rng1 = np.random.default_rng(1)
    x1 = np.r_[rng1.normal(2.0, 0.03, 60), np.full(30, 2.6) + rng1.normal(0, 0.001, 30)]
    w1 = np.r_[np.ones(60), np.full(30, 1 / 30)]
    f1 = MX.fit_mixture(x1, w1, "gauss", k_max=6, var_floor=0.002 ** 2, n_comp_min=10)
    assert f1["K"] == 1 and abs(f1["mu"][0] - 2.0) < 0.02 and abs(f1["n_eff"] - 61) < 1e-9

    e = FT.fit_gmm(np.random.default_rng(2).normal(2.1, 0.05, 200), 4, var_floor=FT.VAR_FLOOR["Q1"], weights=np.full(200, 0.5))
    assert e["type"] == "gmm" and e["n"] == 200 and abs(e["n_eff"] - 100) < 1e-6

    parent = dict(mu=np.array([2.0, 2.4]), var=np.array([0.01, 0.01]), pi=np.array([0.5, 0.5]), nu=None)
    few = MX.adapt(parent, np.full(1, 2.1), np.ones(1), r=4.0, var_floor=1e-6)
    many = MX.adapt(parent, np.full(4000, 2.1), np.ones(4000), r=4.0, var_floor=1e-6)
    huge_r = MX.adapt(parent, np.full(5, 2.1), np.ones(5), r=1e9, var_floor=1e-6)
    assert np.allclose(huge_r["mu"], parent["mu"], atol=1e-6)
    assert abs(few["mu"][0] - 2.0) < abs(many["mu"][0] - 2.0) and abs(many["mu"][0] - 2.1) < 0.005 and many["pi"][0] > 0.95

    rng3 = np.random.default_rng(3)
    rows = []
    for os_, mean, ncx in ((2, 2.00, 120), (3, 1.90, 3)):
        for c in range(ncx):
            code = f"C{os_}{c:04d}"
            l0 = ("Fe", 5, os_, "N_pyridine", "octahedral", "N_pyridine")
            v = float(rng3.normal(mean, 0.02))
            rows.append((code, repr(l0), v, 0, True))
            for lvl, k in L.q1_ancestor_rows(l0):
                rows.append((code, repr(k), v, lvl, False))
    pf = PL.PooledFitter("Q1", pd.DataFrame(rows, columns=["csd_code", "key", "value", "level", "is_l0"]), n_indep=50)
    pf.fit_independent()
    ent = pf.entries(r=4.0)
    sparse, rich = repr(("Fe", 5, 3, "N_pyridine", "octahedral", "N_pyridine")), repr(("Fe", 5, 2, "N_pyridine", "octahedral", "N_pyridine"))
    assert ent[rich]["src"] == "fit" and ent[sparse]["src"] == "adapted"
    m_sparse = float(np.dot(ent[sparse]["weights"], ent[sparse]["means"]))
    assert 1.90 < m_sparse < 2.00
    fd = StatsLookup(load_db({f"Q1:{k}": e_ for k, e_ in ent.items()}, meta={})).dist("Q1", ("Fe", 5, 3, "N_pyridine", "octahedral", "N_pyridine"))
    assert fd.pooled and abs(np.dot(fd.weights, fd.means) - m_sparse) < 1e-9


def test_t_mixture_density_and_padding(stats):
    from scipy.stats import norm, t as student

    from jacobik.ik import scorer as SC
    from jacobik.ik.scorer import _cos_mix_logp
    from jacobik.stats import mixture as MX

    f = dict(mu=np.array([-0.95, 0.02]), var=np.array([0.01 ** 2, 0.05 ** 2]), pi=np.array([0.3, 0.7]), nu=4.0)
    x = np.linspace(-1, 1, 41)
    ref = np.log(sum(p * student.pdf(x, 4.0, m, math.sqrt(v)) for m, v, p in zip(f["mu"], f["var"], f["pi"])))
    assert np.allclose(MX.logp(f, x), ref, atol=1e-6)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    sc = _cos_mix_logp(t(x)[:, None], t(f["mu"])[None], t(f["var"])[None], t(np.log(f["pi"]))[None], t([4.0]))[:, 0].numpy()
    assert np.allclose(sc, ref, atol=1e-6)
    g = _cos_mix_logp(t(x)[:, None], t(f["mu"])[None], t(f["var"])[None], t(np.log(f["pi"]))[None], t([math.inf]))[:, 0].numpy()
    assert np.allclose(g, np.log(sum(p * norm.pdf(x, m, math.sqrt(v)) for m, v, p in zip(f["mu"], f["var"], f["pi"]))), atol=1e-6)

    assert torch.allclose(
        _cos_mix_logp(torch.tensor([[0.9]]), torch.tensor([[0.0, 0.0]]), torch.tensor([[0.01, 1.0]]), torch.tensor([[0.0, -1e4]]), torch.tensor([math.inf])),
        _cos_mix_logp(torch.tensor([[0.9]]), torch.tensor([[0.0]]), torch.tensor([[0.01]]), torch.tensor([[0.0]]), torch.tensor([math.inf])),
    )

    w = torch.tensor([[0.6, 0.4, 0.0]])
    lw = torch.log(w.clamp(min=SC._EPS))
    assert float(torch.where(w > 0, lw, torch.full_like(lw, -1e4))[0, 2]) <= -1e4
    assert 'if SCORE_AGG == "sum" or self._q5_fold:' not in inspect.getsource(SC.Scorer.__init__)


def test_q2_pair_classes_ladder_and_eta7_eta8(fixture_mols):
    from scipy.spatial.transform import Rotation

    from jacobik.ik.targets import IDEAL_GEOMETRIES
    from jacobik.stats import keys as KB
    from jacobik.stats import ladders as L
    from jacobik.stats.stats import StatsLookup, load_db
    from jacobik.topology.coord_geom import best_fit_assignment, ideal_pair_angles
    from jacobik.topology.donor_classifier import full_path

    rng = np.random.default_rng(4)
    for g in ("octahedral", "trig_bipyr", "square_plan", "pent_bipyr", "square_anti", "tricap_pris"):
        V = np.asarray(IDEAL_GEOMETRIES[g], float)
        p = rng.permutation(len(V))
        u = V[p] @ Rotation.random(random_state=5).as_matrix().T + rng.normal(0, 0.05, V.shape)
        ang = ideal_pair_angles(g, best_fit_assignment(u, g))
        Vn = V[p] / np.linalg.norm(V[p], axis=1, keepdims=True)
        assert np.allclose(ang, np.degrees(np.arccos(np.clip(Vn @ Vn.T, -1, 1))), atol=1e-6)

    n = 0
    for m in fixture_mols.values():
        if m.coord_geom == "octahedral":
            cls = [k[1] for _, _, k in KB.q2_pair_keys(m)]
            assert all(L.is_pair_class(c) for c in cls) and cls.count("c180") == 3 and cls.count("c90") == 12
            n += 1
    assert n > 0

    key = ("octahedral", "c90", "different_ligand", "Cr", "C_eta6", "C_carbonyl")
    assert all("c90" in k for k in L.q2_candidates(key))
    fam = [k for name, k in L.q2_named_levels(key) if name == "fam"][0]
    assert "C_haptic_any" in fam and "C_donor" in fam and L.q2_bucket_family("C_eta5") == ("C_haptic_any", "C_haptic_any")

    e = {"type": "gmm_cos", "n": 100, "means": [0.0], "vars": [0.01], "weights": [1.0]}
    st = StatsLookup(load_db({"Q2:('octahedral', 'different_ligand')": e}, meta={}))
    assert st.dist_trace("Q2", ("octahedral", "c90", "different_ligand", "Cr", "N_amine", "O_water"))[1] == ("octahedral", "different_ligand")
    assert full_path("C_eta7")[1] == "C_haptic_any" and full_path("C_eta8")[1] == "C_haptic_any"


def test_q5_wells_ortho_and_extraction_folding(fixture_mols):
    from fit import fitter as FT
    from fit import q5 as FQ5
    from jacobik.stats import ladders as L
    from jacobik.stats.keys import aryl_ortho_count, q5_context

    np.random.seed(0)
    rng = np.random.default_rng(6)
    x = np.r_[rng.normal(60, 8, 300), rng.normal(180, 8, 300), rng.normal(-60, 8, 300), rng.uniform(-180, 180, 60)]
    f = FT.fit_vmm_wells(x, np.ones(len(x)))
    assert f["k"] == 3 and all(k >= FT.KAPPA_WELL_MIN for k in f["kappas"]) and abs(sum(f["weights"]) - 1) < 1e-9

    nb = {1: [2, 6, 9], 2: [1, 3, 7], 3: [2, 4], 4: [3, 5], 5: [4, 6], 6: [5, 1, 8], 7: [2], 8: [6], 9: [1]}
    tm = {i: "C_ar" for i in range(1, 7)} | {7: "C_sp3", 8: "C_sp3", 9: "N_sp3"}
    assert aryl_ortho_count(1, 9, nb, tm) == 2 and q5_context("free", 9, 1, nb, tm) == "free_o2" and q5_context("chelate", 9, 1, nb, tm) == "chelate"
    assert dict(L.q5_named_levels(("N_sp3", "C_ar", "C_sp3", "C_ar", "free_o2")))["no_ortho"] == ("N_sp3", "C_ar", "C_sp3", "C_ar", "free")

    m = next(iter(fixture_mols.values()))
    rows = FQ5.q5_torsions(m)
    FQ5.Q5_FOLD_DATA = False
    try:
        raw = FQ5.q5_torsions(m)
    finally:
        FQ5.Q5_FOLD_DATA = True
    assert len(rows) >= len(raw) and {k for k, _ in rows} == {k for k, _ in raw}


def test_q6_wall_polar_h_through_metal_and_runtime_keys(fixture_mols, monkeypatch, legacy_flags):
    import jacobik.topology.graph as G
    from fit import fitter as Fz
    from fit.q6 import q6_steric_contacts
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.stats import keys as KB
    from jacobik.stats.stats import StatsLookup, load_db

    rng = np.random.default_rng(7)
    x = rng.normal(3.8, 0.3, 20000)
    ew = Fz.fit_contact_wall(x)
    assert abs(ew["mu"] - np.quantile(x, 0.001)) < 0.03 and 0.2 < ew["sigma"] < 0.4 and Fz.fit_contact_wall(x[:500]) is None

    assert KB.q6_pair_key("H", "O", True, False) == ("Hp", "O")
    assert KB.q6_pair_key("H", "C", True, False) == ("C", "H") and KB.q6_pair_key("H", "O", False, False) == ("H", "O")
    st = StatsLookup(load_db({"Q6:('H', 'O')": {"type": "halfgauss", "n": 500, "mu": 2.4, "sigma": 0.2}}, meta={}))
    fd, rk, _ = st.dist_trace("Q6", ("Hp", "O"))
    assert rk == ("H", "O") and fd.mu == 2.4

    for m in list(fixture_mols.values())[:6]:
        ext = {k for k, _ in q6_steric_contacts(m)}
        ge = GeometryExtractor(m, torch.device("cpu"))
        d = ge.q6_contacts(torch.tensor(np.array([a.position for a in m.atoms]), dtype=torch.float32)[None])[0]
        donor = getattr(ge, "q6_donor_scale", None) or [None] * len(ge._q6_elem_keys)
        run = {k for k, dd, s in zip(ge._q6_elem_keys, d.tolist(), donor) if dd <= KB.DIST_CUTOFF_Q6 and s is None}
        assert run == ext

    m_xec = fixture_mols["XECSON"]
    dn = G.donor_indices_of(m_xec)
    monkeypatch.setattr(G, "Q6_THROUGH_METAL", False)
    old = G.q6_topological_exclusions(m_xec.atoms, m_xec.bonds, m_xec.metal_idx, dn)
    monkeypatch.setattr(G, "Q6_THROUGH_METAL", True)
    new = G.q6_topological_exclusions(m_xec.atoms, m_xec.bonds, m_xec.metal_idx, dn)
    assert new < old
    cov = G.q6_topological_exclusions(m_xec.atoms, [b for b in m_xec.bonds if m_xec.metal_idx not in (b.i, b.j)], -1, [])
    for a, b in old - new:
        assert m_xec.metal_idx not in (a, b) and (a, b) not in cov and not (a in dn and b in dn)
    for i, a in enumerate(dn):
        for b in dn[i + 1:]:
            assert (min(a, b), max(a, b)) in new

    xw = 2.4 + np.abs(np.random.default_rng(0).normal(0.9, 0.5, 5000))
    on = Fz._fit_chunk([("Q6", "k", xw, np.ones_like(xw))], dict(Fz._DIST_TYPE, Q6="halfgauss"), Fz._K_MAX)[0][2]
    p5 = Fz._fit_chunk([("Q6", "k", xw, np.ones_like(xw))], dict(Fz._DIST_TYPE, Q6="halfgauss_p5"), Fz._K_MAX)[0][2]
    assert on["model"] == "onset" and p5["model"] == "p5" and p5["mu"] > on["mu"]
    with pytest.raises(ValueError):
        Fz.fit_all("/nonexistent", "/nonexistent.json", quantities=["Q6"], q6_model="bad")


def test_linear_atom_sp_flanks_and_fk_stability(stats, monkeypatch):
    from jacobik.ik import ik_solver as IKM
    from jacobik.stats import keys as kb
    from jacobik.topology.graph import get_heavy_ligand_neighbors, get_ligand_neighbors
    from tools.jdev.cases import get_case

    m_xec = _mol("XECSON")
    tm_xec = kb._build_type_map(m_xec)
    trip = [b for b in m_xec.bonds if b.order >= 2.75 and m_xec.metal_idx not in (b.i, b.j)]
    assert trip and all(tm_xec[b.i] == "C_sp" and tm_xec[b.j] == "C_sp" for b in trip)

    for code in SP_CASES:
        m = _mol(code)
        P = {a.idx: np.asarray(a.position, dtype=float) for a in m.atoms}
        hb = get_heavy_ligand_neighbors(m.atoms, m.bonds, m.metal_idx)
        al = get_ligand_neighbors(m.atoms, m.bonds, m.metal_idx)
        tm = kb._build_type_map(m)
        lin = kb.linear_atoms(tm)
        for lig in m.ligands:
            for ch in lig.pivot_chains:
                for jt in ch.joints:
                    j, k = jt.atom_i, jt.atom_j
                    i, l = kb.q5_flanks(j, k, hb, al, {a.idx: a for a in m.atoms}, tm)
                    if i is None:
                        assert j in lin or k in lin
                        continue
                    for f, (a, b) in ((i, (j, k)), (l, (k, j))):
                        ax = P[b] - P[a]
                        assert np.linalg.norm(np.cross(P[f] - P[a], ax / np.linalg.norm(ax))) > 0.3

    cap = {}
    orig = IKM.JointIKSolver.solve

    def grab(self, theta_inits, X_ref, *args, **kw):
        cap["x"] = (self.fks, [t.detach().clone() for t in theta_inits], X_ref.detach().clone())
        return orig(self, theta_inits, X_ref, *args, **kw)

    monkeypatch.setattr(IKM.JointIKSolver, "solve", grab)
    for code in ("XECSON", "TINBIE"):
        folder, split = SP_CASES[code]
        cap.clear()
        get_case(f"tmqm:{FIX_ROOT / folder}:{split}:{code}").make_builder().solve(
            stats=stats, seed=11, angle_bends=True, embed="etkdg", n_conformers=2, beam_width=2, n_coord_samples=2, n_ik_steps=2,
            stereo_enum=False,
        )
        fks, th0, X_ref = cap["x"]
        Xr = X_ref if X_ref.dim() == 2 else X_ref[0]
        P = Xr.double().numpy()
        for f in fks:
            for fi, pj, pk, fl in zip(f._flanking_i, f._pivot_j, f._pivot_k, f._flanking_l):
                u = P[pk] - P[pj]
                u /= np.linalg.norm(u)
                for f_at, base in ((fi, pj), (fl, pk)):
                    assert np.linalg.norm(np.cross(P[f_at] - P[base], u)) > 0.3, (code, fi, pj, pk, fl)

        def run(Xin):
            X = Xin
            for f, t in zip(fks, th0):
                X = f.apply(t, X)
            return X

        g = torch.Generator().manual_seed(0)
        X1 = run(X_ref)
        X2 = run(X_ref + 1e-6 * torch.randn(X_ref.shape, generator=g, dtype=X_ref.dtype))
        assert float((X1 - X2).norm(dim=-1).max()) < 1e-3


def test_q6pmf_tables_and_scorer_switch(stats, monkeypatch, legacy_flags):
    from fit import q6p as q6pmf
    from jacobik.ik import scorer as SC
    from tools.jdev.cases import get_case

    acc, _, tabs = _fixture_pmf()
    assert ("C", "H", "5", "") in tabs and ("H", "H", "5", "") in tabs
    grid = np.arange(q6pmf.R_LO, q6pmf.R_HI + 1e-9, q6pmf.DR_TAB)
    for k, e in tabs.items():
        w = np.asarray(e["w"])
        assert np.isfinite(w).all() and len(w) == len(grid)
        core = grid < e["r0"]
        assert (np.diff(w[core]) < 0).all(), k
        assert (w[0] > w[core][-1] + 5 if core.sum() > 20 else True) and abs(w[-1]) < 1e-6
    for lab in acc.h:
        n, c, a = lab.split("|")
        if a:
            assert sum(int(acc.h[f"{n}|{c}|{x}"].sum()) for x in ("lin", "bent") if f"{n}|{c}|{x}" in acc.h) <= acc.h[f"{n}|{c}|"].sum()

    monkeypatch.setattr(stats._db, "priors", dict(stats._db.priors, Q6P=tabs))
    folder, split = SP_CASES["XECSON"]
    case = get_case(f"tmqm:{FIX_ROOT / folder}:{split}:XECSON")
    monkeypatch.setattr(SC, "Q6_MODEL", "pmf")
    r = case.make_builder().solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", n_conformers=1, beam_width=2, n_coord_samples=2, n_ik_steps=2)
    sc = r.scorer
    assert sc._q6_pmf is not None and sc._q6_pmf["n_pairs_with_table"] > 0.9 * sc._ge.n_q6
    X = torch.tensor(np.array([a.position for a in r.conformers[0].complex.atoms])[None], dtype=torch.float32, requires_grad=True)
    _, comp = sc.score(X)
    comp["Q6"].sum().backward()
    assert torch.isfinite(comp["Q6"]).all() and torch.isfinite(X.grad).all()
    e = tabs[("C", "H", "5", "")]
    w16 = float(np.interp(1.6, q6pmf.R_LO + q6pmf.DR_TAB * np.arange(len(e["w"])), e["w"]))
    assert w16 > (stats.dist("Q6", ("C", "H")).mu - 1.6) ** 2 / (2 * stats.dist("Q6", ("C", "H")).sigma ** 2)

    monkeypatch.setattr(stats._db, "priors", {k: v for k, v in stats._db.priors.items() if k != "Q6P"})
    r2 = case.make_builder().solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", n_conformers=1, beam_width=2, n_coord_samples=2, n_ik_steps=2)
    assert r2.scorer._q6_pmf is None


def test_fit_pipeline_npz_interfaces_and_parser():
    from fit import bends, q1, q2, q3, q4, q5, q6, q6p, qo
    from fit.parser import TMQMParser
    from jacobik.core.container import load_lossless_npz, save_lossless_npz
    from jacobik.stats import stats as ST
    from tests._paths import DATA_ROOT, require

    sample_db = {
        "Q1:('Fe', 2, 2, 'N_pyrrole', 'octahedral', 'none')": {"type": "gmm", "n": 50, "means": [1.98], "vars": [0.005], "weights": [1.0], "k": 1, "pooled": True, "src": "fit", "n_eff": 50.0},
        "Q2:('square_plan', 'c90', 'different_ligand', 'Pd', 'Hal_Cl', 'N_imine')": {"type": "tmix_cos", "n": 100, "means": [0.0, 0.5], "vars": [0.01, 0.02], "weights": [0.6, 0.4], "k": 2, "pooled": True, "src": "adapted", "nu": 4.5, "n_eff": 85.0},
        "Q3:('octahedral', 'linear')": {"type": "gmm_cos", "n": 15, "means": [-1.0], "vars": [0.02], "weights": [1.0], "k": 1, "n_eff": 15.0},
        "Q4:('tetrahedral', 'Hal_Cl')": {"type": "gmm", "n": 25, "means": [1.90], "vars": [0.015], "weights": [1.0], "k": 1, "n_eff": 25.0},
        "Q5:('Fe', 'c_sp2')": {"type": "vmm", "n": 30, "means": [0.0, 3.14159], "kappas": [2.5, 3.0], "weights": [0.5, 0.5], "k": 2, "tail": 0.01, "model": "wells_folded", "n_eff": 28.0},
        "Q6:('Ni', 'square_plan')": {"type": "halfgauss", "n": 40, "k": 1, "mu": 0.05, "sigma": 0.02, "model": "halfgauss"},
        "QO:('Fe', 'octahedral', 6)": {"type": "mvgmm", "n": 60, "weights": [1.0], "means": [[1.9, 2.0]], "covs": [[[0.01, 0.0], [0.0, 0.02]]], "meta": {"dim": 2, "k": 1}, "feats": ["d1", "d2"], "per_sub": None},
        "Q6P:('C', 'C', 'inter', '')": {"type": "repulsive_table", "n": 500, "r0": 3.4, "core_b": 12.0, "core_kappa": 1.5, "w": [[0.1, 0.2], [0.05, 0.15]]},
    }
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        npz_path = Path(tf.name)
    try:
        save_lossless_npz(sample_db, npz_path)
        reloaded, _ = load_lossless_npz(npz_path)
        assert set(sample_db) == set(reloaded) and json.dumps(sample_db, sort_keys=True) == json.dumps(reloaded, sort_keys=True)
        cdb = ST.load_db(npz_path)
        assert isinstance(cdb, ST.ConstraintDB)
        fd1 = cdb.get("Q1", ("Fe", 2, 2, "N_pyrrole", "octahedral", "none"))
        assert fd1 is not None and fd1.means == [1.98] and fd1.weights == [1.0]
        assert ("C", "C", "inter", "") in cdb.priors.get("Q6P", {})
    finally:
        if npz_path.exists():
            npz_path.unlink()

    assert q1.QUANTITY == "Q1" and q1.DEFAULT_VARIANT == "m_sub_early"
    assert q2.QUANTITY == "Q2" and q3.QUANTITY == "Q3" and q4.QUANTITY == "Q4"
    assert q5.QUANTITY == "Q5" and q5.MODEL_NAME == "wells_folded"
    assert q6.QUANTITY == "Q6" and q6.MODEL_NAME == "p5" and qo.QUANTITY == "QO" and q6p.QUANTITY == "Q6P"
    assert bends.QUANTITY == "BENDS" and ("in", "sp2") in bends.BEND_K

    parser = TMQMParser(str(require(DATA_ROOT / "ref" / "tmQM", "tmQM reference data")))
    shards = [p.stem[len("tmQM_"):] for p in sorted((DATA_ROOT / "ref" / "tmQM" / "xyz").glob("tmQM_*.xyz"))]
    m = list(parser.stream_split(shards[0], limit=1))[0]
    assert m.csd_code and m.metal_symbol
    for col in ("electronic_energy", "dispersion_energy", "dipole_moment", "metal_charge", "hl_gap", "homo_energy", "lumo_energy", "polarizability"):
        assert getattr(m, col, None) is None
