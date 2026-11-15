"""test_stereo_configs: merged from test_isomer_space, test_stereo_superset, test_mirror_completion, test_config_states, test_ring_states, test_ring_chair, test_output_by_class, test_site_enumeration."""
from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import jacobik.core.profile as P
from jacobik.builder.result import ConformerResult, EnsembleResult
from jacobik.ik import dedup as dd
from jacobik.ik import stereo
from jacobik.ik.ik_solver import _row_group_weights
from jacobik.ik.reachability import (
    angular_cost_matrix,
    assignment_is_reachable,
    is_pair_reachable,
    repair_unreachable_assignment,
    site_angles_deg,
)
from jacobik.ik.site_enumeration import (
    DEFAULT_TOL_DEG,
    SiteEnumerationConfig,
    dedup_by_donor_equivalence,
    dedup_combined,
    enumerate_assignments,
    enumerate_unique_assignments,
    site_symmetry_group,
)
from jacobik.ik.targets import IDEAL_GEOMETRIES
from jacobik.topology import rings as rc
from tools.jgen.a4t import analyse as A
from tests.test_solver import _builder as _case_builder, one_thread__gpu_fast_path as one_thread  # noqa: F401


@pytest.fixture(autouse=True)
def restore_factored_default():
    original = dd.DEDUP_FACTORED
    yield
    dd.DEDUP_FACTORED = original


def test_profile_dedup_factored_and_max_dev():
    """Verify production fingerprint, Directive I flags, factored dedup parity/recovery, and max_dev fit."""
    snap = P.snapshot()
    assert snap["fingerprint"] == "2c87e2940af2" and snap["differs_from_profile"] == {}
    for flag in (
        "dedup_factored", "dedup_rmsd", "dedup_max_dev", "stereo_candidates",
        "stereo_scope", "stereo_cost_ceiling", "out_per_class", "out_per_class_k",
        "out_per_class_window", "out_mirror", "ring_states", "ring_states_five",
    ):
        assert flag in P.FIELDS
        _, _, default_val, _ = P.FIELDS[flag]
        if isinstance(default_val, bool):
            assert default_val is (flag in {"dedup_factored", "stereo_candidates", "out_mirror", "ring_states", "ring_states_five"})

    for code in ("DAMVAQ", "FETTAZ"):
        mol, _, _ = A._load_parent(code)
        deduper = dd.deduper_for(mol)
        assert not deduper.is_capped
        torch.manual_seed(42)
        coords = torch.randn(16, len(mol.atoms), 3, dtype=torch.float64)
        scores = torch.arange(16, dtype=torch.float64)
        try:
            dd.DEDUP_FACTORED = False
            sel_prod = deduper.select(coords, scores, k=None, th=0.5)
            dd.DEDUP_FACTORED = True
            sel_fact = deduper.select(coords, scores, k=None, th=0.5)
            assert sel_prod == sel_fact and len(deduper.merges) == 16 - len(sel_fact)
        finally:
            dd.DEDUP_FACTORED = False

    mol_x, _, _ = A._load_parent("XAQCAT")
    deduper_x = dd.deduper_for(mol_x)
    assert deduper_x.is_capped
    pool_path = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/A4b/xyz/XAQCAT/XAQCAT_control_s11_pool.json")
    if pool_path.exists():
        from tools.jgen.bench.pool_export import _read_frames
        man = json.loads(pool_path.read_text())
        frames = list(_read_frames(pool_path.parent / man["rows"][0]["file"]))
        g = np.asarray(man["gen_to_ref_idx"], int)
        scores_x = torch.tensor([r["score"] for r in man["rows"]])
        X_all = np.empty((len(man["rows"]), len(mol_x.atoms), 3))
        for r_idx, r in enumerate(man["rows"]):
            X_all[r_idx] = np.asarray(frames[r["row_index"]][2])[g]
        X_t = torch.tensor(X_all)
        try:
            dd.DEDUP_FACTORED = False
            sel_prod = deduper_x.select(X_t, scores_x, k=None, th=0.5)
            dd.DEDUP_FACTORED = True
            sel_fact = deduper_x.select(X_t, scores_x, k=None, th=0.5)
            assert len(sel_fact) <= len(sel_prod) and set(sel_fact) <= set(sel_prod) and deduper_x.merges
            for m in deduper_x.merges:
                perm = m.get("perm")
                if perm is not None:
                    perm_arr = np.array(perm, dtype=int)
                    assert (deduper_x.adj[np.ix_(perm_arr, perm_arr)] == deduper_x.adj).all()
        finally:
            dd.DEDUP_FACTORED = False

    P0 = np.zeros((20, 3))
    P0[:, 0] = np.arange(20)
    Q0 = P0.copy()
    Q0[10, 2] += 2.0
    rmsd, max_dev, _ = dd.kabsch_fit_np(P0, Q0)
    assert rmsd < 0.5 and max_dev > 0.7


def test_stereo_perception_mirror_and_ceilings():
    cases = {
        "DAMVAQ": (2, [stereo.StereoType.BINDING_CREATED_DONOR]),
        "FETTAZ": (2, [stereo.StereoType.BINDING_CREATED_DONOR]),
        "FUTSOD": (6, [stereo.StereoType.LIGAND_TETRAHEDRAL]),
        "XACMOE": (2, [stereo.StereoType.BINDING_CREATED_DONOR]),
        "ZAZREZ": (0, []),
        "LISVOC": (2, [stereo.StereoType.LIGAND_TETRAHEDRAL, stereo.StereoType.BINDING_CREATED_DONOR]),
        "BOCRAN": (0, []),
        "CIFZEZ": (0, []),
        "HOYBED": (1, [stereo.StereoType.BINDING_CREATED_DONOR]),
        "USAPOU": (2, [stereo.StereoType.LIGAND_TETRAHEDRAL]),
    }
    for code, (min_elem, types) in cases.items():
        _, builder, _ = A._load_parent(code)
        elements = stereo.perceive_complex_stereo(builder, scope="B")
        assert len(elements) >= min_elem, code
        elem_types = {e.element_type for e in elements}
        for exp_type in types:
            assert exp_type in elem_types, (code, exp_type)

    assert stereo.is_achiral_geometry("square_planar") is True
    assert stereo.is_achiral_geometry("octahedral") is True
    assert stereo.is_achiral_geometry("chiral_cluster") is False
    e1 = stereo.StereoElement(element_type=stereo.StereoType.LIGAND_TETRAHEDRAL, ligand_idx=0, atom_idx=1, name="E1")
    e2 = stereo.StereoElement(element_type=stereo.StereoType.LIGAND_TETRAHEDRAL, ligand_idx=0, atom_idx=2, name="E2")
    assert len(stereo.enumerate_stereo_variants([e1, e2], achiral_environment=True, baseline_classes=10)[0]) == 2
    assert len(stereo.enumerate_stereo_variants([e1, e2], achiral_environment=False, baseline_classes=10)[0]) == 4

    elems8 = [stereo.StereoElement(element_type=stereo.StereoType.LIGAND_TETRAHEDRAL, ligand_idx=0, atom_idx=i, name=f"E{i}") for i in range(8)]
    vars8, sampled = stereo.enumerate_stereo_variants(elems8, achiral_environment=False, cost_ceiling_multiplier=1.5, baseline_classes=4, seed=123)
    assert sampled is True and len(vars8) == 6

    # out_per_class keeps up to K distinct rows per class inside window
    rows = [{"score": float(i), "class_label": "Class_A"} for i in range(20)] + [{"score": float(50.0 + i * 5), "class_label": "Class_B"} for i in range(5)]
    rows.sort(key=lambda r: r["score"])
    kept_per_class, selected, best_score = {}, [], rows[0]["score"]
    for r in rows:
        if r["score"] - best_score <= 200.0 and kept_per_class.get(r["class_label"], 0) < 4:
            selected.append(r)
            kept_per_class[r["class_label"]] = kept_per_class.get(r["class_label"], 0) + 1
    assert kept_per_class == {"Class_A": 4, "Class_B": 4} and len(selected) == 8


def test_stereo_audit_items_1_to_6_and_chelate_embed(stats, caplog):
    from rdkit import Chem
    from jacobik import ComplexBuilder, LigandInput
    from jacobik.builder import embed as _ce

    # Item 1: specified tetrahedral and E/Z not overwritten
    b_ccw = ComplexBuilder("Pt", "square_plan", 0, 1, [LigandInput("N[C@H](C)C(=O)[O-]", donors=[[0], [4]]), LigandInput("CC(C)N", donors=[[3]]), LigandInput("[Cl-]", donors=[[0]])])
    b_cw = ComplexBuilder("Pt", "square_plan", 0, 1, [LigandInput("N[C@@H](C)C(=O)[O-]", donors=[[0], [4]]), LigandInput("CC(C)N", donors=[[3]]), LigandInput("[Cl-]", donors=[[0]])])
    elems_ccw = stereo.perceive_complex_stereo(b_ccw, scope="B")
    elems_cw = stereo.perceive_complex_stereo(b_cw, scope="B")
    el_ccw = next(e for e in elems_ccw if e.specified and e.atom_symbol == "C")
    el_cw = next(e for e in elems_cw if e.specified and e.atom_symbol == "C")
    assert el_ccw.specified_value == -1 and el_cw.specified_value == 1
    for v in stereo.enumerate_stereo_variants(elems_ccw, achiral_environment=True)[0]:
        assert el_ccw.name not in v.config

    b_ez_e = ComplexBuilder("Pt", "square_plan", 0, 1, [LigandInput("C/C=C/C(=O)[O-]", donors=[[4]])] + [LigandInput("[Cl-]", donors=[[0]])] * 3)
    b_ez_z = ComplexBuilder("Pt", "square_plan", 0, 1, [LigandInput("C/C=C\\C(=O)[O-]", donors=[[4]])] + [LigandInput("[Cl-]", donors=[[0]])] * 3)
    assert next(e for e in stereo.perceive_complex_stereo(b_ez_e, scope="B") if e.specified).specified_value == 0
    assert next(e for e in stereo.perceive_complex_stereo(b_ez_z, scope="B") if e.specified).specified_value == 1

    # Item 2 & default-path set_ez tolerance
    assert len(stereo.perceive_complex_stereo(A._load_parent("BOCRAN")[1], scope="B")) == 0
    assert len(stereo.perceive_complex_stereo(A._load_parent("CIFZEZ")[1], scope="B")) == 0
    for code in ("YABYEJ", "ZAFPOP"):
        elems = stereo.perceive_complex_stereo(A._load_parent(code)[1], scope="B")
        assert len(elems) == 1 and elems[0].element_type == stereo.StereoType.MCX_ROTAMER and elems[0].neighbors[0] != elems[0].atom_idx

    m_triple = Chem.AddHs(Chem.MolFromSmiles("CC#CC"))
    assert _ce.set_ez(m_triple, [(1, 2, 0, 3, True)]) == 0
    with pytest.raises(ValueError, match="DOUBLE"):
        _ce.set_ez(m_triple, [(1, 2, 0, 3, True)], strict=True)
    m_single = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    assert _ce.set_ez(m_single, [(1, 2, 0, 3, True)]) == 0 and _ce.set_ez(m_single, [(0, 3, 1, 2, True)]) == 0
    with pytest.raises(ValueError, match="DOUBLE"):
        _ce.set_ez(m_single, [(1, 2, 0, 3, True)], strict=True)
    with pytest.raises(ValueError, match="no bond"):
        _ce.set_ez(m_single, [(0, 3, 1, 2, True)], strict=True)
    with pytest.raises(ValueError, match="set_chiral"):
        _ce.set_chiral(m_single, [(999, Chem.ChiralType.CHI_TETRAHEDRAL_CW)])

    with caplog.at_level(logging.DEBUG, logger="jacobik.builder.embed"):
        res_ce = _ce.embed("CN(C)C=CN(C)C", [[1], [4]], seed=42, mmff=True, ez_extra=[(1, 3, 0, 4, True), (0, 5, 1, 4, False)])
    assert res_ce is not None and res_ce[2]["bounded"] is True and res_ce[2]["ez"] == 1
    assert "skipping non-double or missing bond" in caplog.text
    assert res_ce[0].GetBondBetweenAtoms(3, 4).GetStereo() == Chem.BondStereo.STEREONONE
    with pytest.raises(ValueError):
        _ce.embed("CN(C)C=CN(C)C", [[1], [4]], seed=42, mmff=True, ez_extra=[(1, 3, 0, 4, True)], strict_stereo=True)

    # Item 3: 3D stereo labels on FETTAZ
    _, b_fet, _ = A._load_parent("FETTAZ")
    b_fet.stereo_enum, b_fet.out_per_class = True, True
    res_fet = b_fet.solve(stats=stats, n_ik_steps=6, fast_embed=True, beam_width=2, n_coord_samples=2)
    assert res_fet.conformers
    for c in res_fet.conformers:
        assert c.requested_stereo_label is not None and c.stereo_label not in (None, "default")
        assert ("+" in c.stereo_label or "-" in c.stereo_label) and f"|S:{c.stereo_label}|" in c.class_label

    # Item 4 & Item 5: out_per_class without variants & merged pool symmetry dedup
    gly = lambda: LigandInput("NCC(=O)[O-]", donors=[[0], [4]])
    b_gly = ComplexBuilder("Pt", "square_plan", 0, 1, [gly(), gly()])
    b_gly.out_per_class, b_gly.stereo_enum = True, False
    res_gly = b_gly.solve(stats=stats, n_ik_steps=10, fast_embed=True, beam_width=4, n_conformers=16)
    by_cls = {}
    for c in res_gly.conformers:
        by_cls.setdefault(c.class_label, []).append(c)
    assert len(by_cls) >= 2 and all(len(v) > 1 and "|S:none|" in k for k, v in by_cls.items())

    b_gly.stereo_enum = True
    v1, v2 = stereo.StereoVariant(config={}, class_id="var1"), stereo.StereoVariant(config={}, class_id="var2")
    res_m = b_gly._solve_stereo_variants([v1, v2], [], dict(stats=stats, n_ik_steps=6, fast_embed=True, beam_width=2))
    coords_m = torch.tensor(np.array([[a.position for a in c.complex.atoms] for c in res_m.conformers]))
    scores_m = torch.tensor([c.score for c in res_m.conformers])
    assert len(dd.deduper_for(res_m.conformers[0].complex).select(coords_m, scores_m, k=None, th=0.5)) == len(res_m.conformers)

    # Item 6: chelate exclusions, feasibility filtering, LIPDEU phosphorus perception
    for code in ("AZUSIA", "HOXJUZ", "QEZVEY"):
        assert len(stereo.perceive_complex_stereo(A._load_parent(code)[1], scope="B")) == 0
    assert [e.name for e in stereo.perceive_complex_stereo(A._load_parent("LOPCAW")[1], scope="B")] == ["L0_B11_12"]
    _, b_ted, _ = A._load_parent("TEDPOI")
    vars_ted, _ = stereo.enumerate_stereo_variants(stereo.perceive_complex_stereo(b_ted, scope="B"), builder=b_ted)
    assert len(vars_ted) == 1 and vars_ted[0].class_id == "L0_B8_9:E_L1_B8_9:E"
    _, b_lip, _ = A._load_parent("LIPDEU")
    elems_lip = stereo.perceive_complex_stereo(b_lip, scope="B")
    assert [e.name for e in elems_lip] == ["L1_D4_P"]
    assert stereo.build_variant_ligand_fixes(stereo.StereoVariant(config={"L1_D4_P": 1}, class_id="+"), elems_lip, b_lip.ligands)[1]["chiral"] == [(4, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)]
    assert stereo.build_variant_ligand_fixes(stereo.StereoVariant(config={"L1_D4_P": -1}, class_id="-"), elems_lip, b_lip.ligands)[1]["chiral"] == [(4, Chem.ChiralType.CHI_TETRAHEDRAL_CW)]


def test_item7_stereo_inner_prevents_ez_variants_reentry(stats):
    from jacobik.builder.engine import ComplexBuilder
    _, b, _ = A._load_parent("YABYEJ")
    cb = ComplexBuilder(metal=b.metal, geometry=b.geometry, ligands=b.ligands, charge=b.charge, spin=b.spin, stereo_enum=True)
    res = cb.solve(stats=stats, seed=42, n_ik_steps=12, beam_width=4, n_coord_samples=2, embed="etkdg")
    assert res.conformers, res.failure
    labels = {c.stereo_label for c in res.conformers}
    assert "L1_MCX_2_3:E" in labels and "L1_MCX_2_3:Z" in labels


def test_stereo_superset_contract(monkeypatch):
    monkeypatch.setattr(stereo, "check_variant_feasibility", lambda v, b, e=None: True)
    mk = lambda sd: SimpleNamespace(ligands=[SimpleNamespace(smiles=s, donors=d) for s, d in sd], metal="Ru", geometry="octahedral")
    B = mk([("NC(C)C(F)(Cl)C(C)O", [[0]]), ("CC=CC(C)(F)C(Cl)N", [[8]])])
    PROD4 = [[None, None], [None, [(1, 2, 0, 3, False)]], [None, [(1, 2, 0, 3, True)]], [None, None]]
    els = stereo.perceive_complex_stereo(B)
    vs, _ = stereo.superset_variants(B, els, PROD4)
    prod = [v for v in vs if v.production]
    assert len(prod) == 4 and vs[:4] == prod
    assert [v.prod_ez for v in prod] == [[x or None for x in c] for c in PROD4]
    assert all(not v.production for v in vs[4:])

    for n_prod, mult, cap in ((1, 1.5, 4), (2, 1.5, 4), (4, 1.5, 4), (4, 3.0, 8), (1, 1.0, 4)):
        v_p, info = stereo.superset_variants(B, els, PROD4[:n_prod], cost_ceiling_multiplier=mult)
        assert sum(v.production for v in v_p) == n_prod and info["cap"] == cap
        assert 1 <= sum(not v.production for v in v_p) == min(cap, info["n_candidates"])

    stereo.superset_variants(mk([("NC(C)C(F)Cl", [[0]])]), [], [[None]])
    b_again, _ = stereo.superset_variants(B, stereo.perceive_complex_stereo(B), PROD4)
    assert [v.class_id for v in vs] == [v.class_id for v in b_again]

    b1 = mk([("CC=CC", [[0]])])
    vs1, info1 = stereo.superset_variants(b1, stereo.perceive_complex_stereo(b1), [[[(1, 2, 0, 3, False)]], [[(1, 2, 0, 3, True)]]])
    assert len(vs1) == 2 and all(v.production for v in vs1) and info1["n_extra"] == 0

    ex = stereo.extra_stereo_elements(els, PROD4)
    assert not any(e.element_type == stereo.StereoType.MCX_ROTAMER for e in ex)
    assert not any(e.element_type == stereo.StereoType.DOUBLE_BOND_EZ and e.ligand_idx == 1 for e in ex)
    assert any(e.element_type == stereo.StereoType.LIGAND_TETRAHEDRAL for e in ex)


def test_mirror_completion_and_dedup(stats, one_thread, monkeypatch):
    from jacobik.builder import engine as E
    from jacobik.ik.dedup import StructureDeduper

    assert P.FIELDS["out_mirror"][2] is True and P.snapshot()["fingerprint"] == P.PRODUCTION_FINGERPRINT
    r = _case_builder("AHIGOQ").solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", beam_width=4, n_coord_samples=2, n_conformers=None, n_ik_steps=12)
    mol = r.conformers[0].complex
    X = np.array([[a.position for a in c.complex.atoms] for c in r.conformers])[:8]
    sc = np.array([c.score for c in r.conformers])[:8]
    Xt = torch.tensor(X, dtype=torch.float32)

    dd_obj = StructureDeduper(mol)
    assert not dd_obj.is_capped
    noisy = Xt + 0.01 * torch.randn(Xt.shape, generator=torch.Generator().manual_seed(0))
    far = Xt.flip(0) * torch.tensor([-1.0, 1.0, 1.0])
    Q = torch.cat([noisy, far])
    got = dd_obj.any_dup(Q, Xt)
    for i in range(Q.shape[0]):
        keep = dd_obj.select(torch.cat([Xt, Q[i:i + 1]]), torch.cat([torch.ones(len(Xt)), torch.zeros(1)]), None)
        assert got[i] == (len(Xt) not in keep)
    assert all(got[:len(Xt)])
    assert dd_obj.mirror_dup(Xt) == dd_obj.any_dup(Xt * torch.tensor([-1.0, 1.0, 1.0]), Xt)

    b_ahi = _case_builder("AHIGOQ")
    monkeypatch.setattr(E, "OUT_MIRROR", False)
    assert E.mirror_completion(b_ahi, mol, X, sc, "cpu")[0] == []
    monkeypatch.setattr(E, "OUT_MIRROR", True)
    rows, rs = E.mirror_completion(b_ahi, mol, X, sc, "cpu")
    assert len(rs) == len(rows) and np.allclose(rs, sc[rows])
    allX = torch.tensor(np.concatenate([X, X[rows] * E.MIRROR]), dtype=torch.float32)
    assert len(dd_obj.select(allX, torch.linspace(1, 0, len(allX)), None)) == len(allX)

    rows_s, rs_s = E.mirror_completion(b_ahi, mol, X, sc, "cpu", lambda Y, cand: Y[:, 0, 0].astype(float))
    assert np.allclose(rs_s, sc[rows_s] - 2 * X[rows_s, 0, 0])

    monkeypatch.setattr(E, "_has_tetrahedral_tags", lambda b: True)
    assert E.mirror_completion(b_ahi, mol, X, sc, "cpu")[0] == []

    # Mirror rank tie-breaking
    cr = [ConformerResult(complex=None, score=1.0, converged=True, rank=0, mirror_of=3),
          ConformerResult(complex=None, score=1.0, converged=True, rank=0),
          ConformerResult(complex=None, score=2.0, converged=True, rank=0)]
    out = sorted(cr, key=lambda c: (getattr(c.score, "total", c.score), getattr(c, "mirror_of", None) is None), reverse=True)
    assert [c.score for c in out] == [2.0, 1.0, 1.0] and out[1].mirror_of is None and out[2].mirror_of == 3


def test_cpu_solve_out_mirror_matches_batched_and_appends_reflections(stats, one_thread):
    from jacobik.builder import engine as E
    assert E.OUT_MIRROR is True and E.BATCHED_ASSEMBLY is False
    r = _case_builder("MELLUN").solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", beam_width=8, n_coord_samples=4, n_conformers=8, n_ik_steps=16, device="cpu")
    mirrors = [c for c in r.conformers if c.mirror_of is not None]
    assert len(mirrors) > 0 and 8 <= len(r.conformers) <= 16
    by_trial = {c.trial_idx: c for c in r.conformers if c.mirror_of is None}
    for m in mirrors:
        src = by_trial[m.mirror_of]
        assert np.isclose(m.score, src.score)
        assert np.allclose([a.position for a in m.complex.atoms], np.array([a.position for a in src.complex.atoms]) * E.MIRROR, atol=1e-5)


def test_config_states_near_partition_and_remote_inversion(monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from jacobik.builder import configs as RC

    rows = {(0, 1): torch.tensor([True, True, False]), (2, 3): torch.tensor([True, False, False])}
    w = _row_group_weights([[0, 1], [2, 3]], rows, "cpu")
    assert torch.allclose(w[0], torch.tensor([0.5, 1.0, 0.0])) and torch.allclose(w[1], torch.tensor([0.5, 0.0, 0.0]))
    assert _row_group_weights([], rows, "cpu") == []

    el = lambda kind, li, idx, spec=False: stereo.StereoElement(
        element_type=kind, ligand_idx=li, atom_idx=idx, atom_idx_end=None,
        name=f"L{li}_{idx}", specified=spec, specified_value=1 if spec else None,
    )
    lig = SimpleNamespace(smiles="OC(C)CCC1CCC(C)CC1", donors=[[0]])
    b = SimpleNamespace(ligands=[lig], metal="Cu")
    near_c = el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 1)
    ring_c = el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 5)
    bcd = el(stereo.StereoType.BINDING_CREATED_DONOR, 0, 0)
    els, collapse, info = stereo.near_elements(b, [near_c, ring_c, bcd], [near_c, ring_c, bcd], True, True)
    assert collapse and info["n_near"] == 2 and info["remote_ring"] == ["L0_5"] and {e.name for e in els} == {"L0_1", "L0_5", "L0_0"}
    assert not stereo.near_elements(b, [near_c], [near_c], True, False)[1]
    assert not stereo.near_elements(b, [near_c, el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 9, True)], [near_c, el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 9, True)], True, True)[1]

    monkeypatch.setattr(stereo, "check_variant_feasibility", lambda *a, **k: True)
    b2 = SimpleNamespace(ligands=[SimpleNamespace(smiles="OC(C)CC", donors=[[0]])], metal="Cu")
    vs, info2 = stereo.superset_variants(b2, [el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 1)], [[None]], achiral_environment=True, near=True, mirror=True)
    vs_nm, _ = stereo.superset_variants(b2, [el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 1)], [[None]], achiral_environment=True, near=True, mirror=False)
    assert vs[0].production and info2["mirror_collapse"] and len(vs) == 2 and len(vs_nm) == 3

    def _stub(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m, randomSeed=7)
        P_ = m.GetConformer().GetPositions()
        atoms = [SimpleNamespace(symbol="Pt", position=P_[0] + np.array([2.0, 0, 0]))] + [SimpleNamespace(symbol=a.GetSymbol(), position=P_[a.GetIdx()]) for a in m.GetAtoms()]
        bonds = [SimpleNamespace(i=bn.GetBeginAtomIdx() + 2, j=bn.GetEndAtomIdx() + 2) for bn in m.GetBonds()] + [SimpleNamespace(i=1, j=2)]
        return SimpleNamespace(atoms=atoms, bonds=bonds, ligands=[SimpleNamespace(atom_indices=list(range(2, m.GetNumAtoms() + 2)), donor_atoms=[2])], metal_idx=1), np.array([a.position for a in atoms])

    mol, X = _stub("NC(C)CCC(C)CC")
    ops, bad = RC.plan(mol, [el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 1), el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 5)], ["L0_5"])
    assert bad == [] and len(ops) == 1 and 0 not in ops[0]["move"] and 1 not in ops[0]["move"]
    Xi = RC.apply(ops[0], X[None])[0]
    moved = list(ops[0]["move"]) + [ops[0]["c"]]
    assert np.allclose(np.linalg.norm(X[moved][:, None] - X[moved][None], axis=-1), np.linalg.norm(Xi[moved][:, None] - Xi[moved][None], axis=-1), atol=1e-8)
    c = ops[0]["c"]
    nb = [bn.j - 1 if bn.i - 1 == c else bn.i - 1 for bn in mol.bonds if c in (bn.i - 1, bn.j - 1)]
    vol = lambda Y: np.linalg.det(np.array([Y[n] - Y[c] for n in nb[:3]]))
    assert np.sign(vol(X)) == -np.sign(vol(Xi))
    mol2, _ = _stub("NCCCC1CCC(C)CC1")
    assert RC.plan(mol2, [el(stereo.StereoType.LIGAND_TETRAHEDRAL, 0, 4)], ["L0_4"]) == ([], ["L0_4"])


def test_ring_states_library_and_allocation(monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    assert P.FIELDS["ring_states"][2] is True and P.snapshot()["fingerprint"] == P.PRODUCTION_FINGERPRINT
    m = Chem.AddHs(Chem.MolFromSmiles("P(C1CCCCC1)(C1CCCCC1)C1CCCCC1"))
    p = AllChem.ETKDGv3(); p.randomSeed = 7
    AllChem.EmbedMolecule(m, p)
    X = m.GetConformer().GetPositions()
    rings = rc.near_rings(m, [0])
    L = rc.library(m, X, [0])
    assert len(rings) == 3 and L is not None and 2 <= len(L["dE_kj"]) <= rc.MAX_STATES
    C = L["coords"]
    assert np.allclose(C[0], X)
    moving = {a for r in rings for a in r} | {n.GetIdx() for r in rings for a in r for n in m.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() == 1}
    fixed = [i for i in range(len(X)) if i not in moving or i == 0]
    assert np.allclose(C[:, fixed], X[fixed], atol=1e-6)
    assert L["dE_kj"].min() == 0.0 and np.isclose(L["share"].sum(), 1.0) and L["share"].min() >= 0.5 / len(L["share"]) - 1e-12
    assert any(all(rc.is_chair(C[k], r) for r in rings) for k in range(len(C)))
    assert rc.library(m, np.full_like(X, np.nan), [0]) is None

    m_me = Chem.AddHs(Chem.MolFromSmiles("P(C)(C)C"))
    AllChem.EmbedMolecule(m_me, randomSeed=1)
    assert rc.library(m_me, m_me.GetConformer().GetPositions(), [0]) is None

    share = np.array([0.7, 0.2, 0.05, 0.05])
    for B in (1, 5, 64, 1000):
        idx = rc.allocate(share, B, np.random.default_rng(0))
        n = np.bincount(idx, minlength=4)
        assert len(idx) == B and (B < 4 or n.min() >= 1) and (B < 100 or np.all(np.abs(n - share * B) <= 1.0 + 1e-9))

    m_ca = Chem.AddHs(Chem.MolFromSmiles("[CH-]1CCCCC1"))
    AllChem.EmbedMolecule(m_ca, randomSeed=3)
    L_ca = rc.library(m_ca, m_ca.GetConformer().GetPositions(), [0])
    if L_ca is not None:
        assert np.allclose(L_ca["coords"][:, 0], m_ca.GetConformer().GetPositions()[0])
        ring_ca = rc.near_rings(m_ca, [0])[0]
        assert sum(rc.is_chair(c, ring_ca) for c in L_ca["coords"]) >= 2

    m_c5 = Chem.AddHs(Chem.MolFromSmiles("P(C1CCCC1)(C1CCCC1)C1CCCC1"))
    AllChem.EmbedMolecule(m_c5, randomSeed=5)
    monkeypatch.setattr(rc, "FIVE", False)
    assert rc.near_rings(m_c5, [0]) == []
    monkeypatch.setattr(rc, "FIVE", True)
    assert len(rc.near_rings(m_c5, [0])) == 3 and len(rc._patterns(5)) == 20
    rc._CACHE.clear()
    L5 = rc.library(m_c5, m_c5.GetConformer().GetPositions(), [0])
    rc._CACHE.clear()
    assert L5 is not None and 2 <= len(L5["dE_kj"]) <= rc.MAX_STATES
    assert all(set(lab.split("/")) <= {"env", "tw5"} for lab in L5["labels"][1:])


def test_ring_chair_all_cases_and_failure_fallback(monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def _embed(smiles, seed):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        p = AllChem.ETKDGv3(); p.randomSeed = seed
        AllChem.EmbedMolecule(m, p)
        AllChem.MMFFOptimizeMolecule(m)
        return m, m.GetConformer().GetPositions()

    def _cip(m, X):
        mm = Chem.Mol(m); mm.RemoveAllConformers()
        c = Chem.Conformer(m.GetNumAtoms())
        for i, pos in enumerate(X):
            c.SetAtomPosition(i, [float(v) for v in pos])
        mm.AddConformer(c, assignId=True)
        Chem.AssignStereochemistryFrom3D(mm, replaceExistingTags=True)
        return {a.GetIdx(): a.GetProp("_CIPCode") for a in mm.GetAtoms() if a.HasProp("_CIPCode")}

    cases = {
        "cyclohexane": "C1CCCCC1",
        "PCy3": "C1CCC(CC1)P(C1CCCCC1)C1CCCCC1",
        "trans-1,4-dimethyl": "C[C@H]1CC[C@@H](C)CC1",
        "cis-1,4-dimethyl": "C[C@H]1CC[C@H](C)CC1",
        "trans-1,2-dimethyl": "C[C@H]1CCCC[C@@H]1C",
        "cis-1,2-dimethyl": "C[C@H]1CCCC[C@H]1C",
        "trans-decalin": "C1CC[C@H]2CCCC[C@@H]2C1",
        "cis-decalin": "C1CC[C@H]2CCCC[C@H]2C1",
    }
    for name, smi in cases.items():
        for seed in (0, 1, 2):
            m, X = _embed(smi, seed)
            rings = rc.saturated_six_rings(m)
            Y, info = rc.chairify(m, X)
            assert rings and info["skipped"] is None
            assert all(rc.is_chair(Y, r) for r in rings) and _cip(m, Y) == _cip(m, X) and np.isfinite(Y).all()

    for smi in ("c1ccccc1", "C1=CCCCC1", "C1CCOCC1=O"):
        m, X = _embed(smi, 3)
        Y, info = rc.chairify(m, X)
        if info["n_rings"] == 0:
            assert np.array_equal(X, Y)

    m_pcy, X_pcy = _embed(cases["PCy3"], 0)
    monkeypatch.setattr(rc, "_minimise", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert np.array_equal(X_pcy, rc.chairify(m_pcy, X_pcy)[0])
    assert rc.chairify(m_pcy, X_pcy.copy()[:3])[1]["skipped"]


def test_output_by_class_and_xyz(tmp_path, capsys):
    _c = lambda rank, score, cls, prod=False: ConformerResult(
        complex=None, score=score, converged=True, rank=rank, assignment_idx=0,
        stereo_label=cls, class_label=f"oct|S:{cls}|A:0", stereo_production=prod,
    )
    e = EnsembleResult(conformers=[_c(1, -1.0, "a", True), _c(2, -1.5, "b"), _c(3, -2.0, "a", True)])
    assert list(e.by_class()) == ["oct|S:a|A:0", "oct|S:b|A:0"]
    assert e.classes()[0]["n_rows"] == 2 and e.conformers[1].descriptors()["mirror_of"] is None

    c_rep = _c(1, -1.0, "a", True)
    c_rep.mirror_of, c_rep.ring_state = 4, "L0:chair"
    r_rep = dataclasses.replace(c_rep, rank=7)
    assert r_rep.stereo_production and r_rep.mirror_of == 4 and r_rep.ring_state == "L0:chair"

    mol = lambda: SimpleNamespace(csd_code="X", atoms=[SimpleNamespace(symbol="Fe", position=(0.0, 0.0, 0.0)), SimpleNamespace(symbol="N", position=(2.0, 0.0, 0.0))])
    cs = [_c(1, -1.0, "a", True), _c(2, -1.5, "b"), _c(3, -2.0, "a", True)]
    for c in cs:
        c.complex = mol()
    cs[2].mirror_of = 0
    ens = EnsembleResult(conformers=cs, scorer=SimpleNamespace(_agg_norm=lambda: 20.0))
    assert ens.class_ids() == {"oct|S:a|A:0": "C1", "oct|S:b|A:0": "C2"}
    assert ens.classes()[0]["n_reflected"] == 1 and abs(ens.classes()[1]["delta_nats"] - 10.0) < 1e-9
    assert [r["rank"] for r in ens.to_records()] == [1, 2, 3]
    ens.summary()
    out = capsys.readouterr().out
    assert "C1" in out and "C2" in out and "reflected" in out
    assert ens.write_xyz(tmp_path / "all.xyz")[0].read_text().count("rank=") == 3
    per = ens.write_xyz(tmp_path / "cls", by_class=True)
    assert sorted(f.name for f in per) == ["C1.xyz", "C2.xyz"] and per[0].read_text().count("rank=") == 2

    plain = EnsembleResult(conformers=[ConformerResult(complex=mol(), score=-1.0, converged=True, rank=1)])
    assert plain.delta_nats(plain.conformers[0]) is None and plain.classes()[0]["geometry"] is None
    plain.summary()
    assert "1/1 converged" in capsys.readouterr().out


def test_site_symmetry_groups_all_geometries():
    for label, directions in sorted(IDEAL_GEOMETRIES.items()):
        n = directions.shape[0]
        angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1.0, 1.0)))
        group = site_symmetry_group(label)
        group_set = set(group)
        assert len(group) >= 1 and tuple(range(n)) in group_set
        if n <= 8:
            naive = {
                perm for perm in itertools.permutations(range(n))
                if np.max(np.abs(angles[np.ix_(perm, perm)] - angles)) <= DEFAULT_TOL_DEG
            }
            assert group_set == naive, label
        for perm in group:
            assert np.max(np.abs(angles[np.ix_(perm, perm)] - angles)) <= DEFAULT_TOL_DEG
            u, _, vt = np.linalg.svd(directions.T @ directions[list(perm)])
            q = u @ vt
            assert q @ q.T == pytest.approx(np.eye(3), abs=1e-8) and directions @ q == pytest.approx(directions[list(perm)], abs=1e-6)
            inv = [0] * n
            for i, v in enumerate(perm):
                inv[v] = i
            assert tuple(inv) in group_set
            for b in group:
                assert tuple(perm[b[i]] for i in range(n)) in group_set

    assert len(site_symmetry_group("trigonal")) == 6
    assert set(site_symmetry_group("linear")) == {(0, 1), (1, 0)}
    assert len(site_symmetry_group("octahedral")) == 48 and len(site_symmetry_group("tetrahedral")) == 24
    assert site_symmetry_group("octahedral") is site_symmetry_group("octahedral")
    with pytest.raises(KeyError):
        site_symmetry_group("not_a_real_geometry")


def test_site_enumeration_dedup_edta_and_reachability():
    assert len(enumerate_assignments(4, "tetrahedral", hop_matrix=None).assignments) == 24
    hop_cis = np.full((6, 6), -1, dtype=int); hop_cis[0, 1] = hop_cis[1, 0] = 3
    res_cis = enumerate_assignments(6, "octahedral", hop_matrix=hop_cis)
    assert len(res_cis.assignments) == 576 and not res_cis.truncated
    with pytest.raises(ValueError):
        enumerate_assignments(5, "octahedral")
    with pytest.raises(ValueError):
        enumerate_assignments(6, "octahedral", hop_matrix=np.zeros((5, 5), dtype=int))
    with pytest.raises(KeyError):
        enumerate_assignments(6, "not_a_real_geometry")
    assert len(enumerate_assignments(6, "octahedral", config=SiteEnumerationConfig(max_candidates=10, time_budget_s=None)).assignments) == 10
    assert enumerate_assignments(6, "octahedral", config=SiteEnumerationConfig(max_candidates=None, time_budget_s=0.0)).truncated

    a, b, c = np.array([0, 1, 2, 3, 4, 5]), np.array([1, 0, 2, 3, 4, 5]), np.array([0, 1, 2, 3, 5, 4])
    assert len(dedup_by_donor_equivalence([a, b, c], [[0, 1]])) == 2
    assert len(dedup_by_donor_equivalence([np.array([0, 1, 2]), np.array([1, 0, 2])], [[0]])) == 2
    pair = [np.array([0, 2, 3, 4, 1, 5]), np.array([0, 4, 2, 5, 1, 3])]
    assert len(dedup_by_donor_equivalence(pair, [[0, 1], [2, 3, 4, 5]])) == 2
    assert len(dedup_combined(pair, [[0, 1], [2, 3, 4, 5]], "octahedral").assignments) == 1
    assert dedup_combined([a, c], [], "octahedral", config=SiteEnumerationConfig(time_budget_s=-1.0)).truncated
    res_exp = enumerate_unique_assignments(6, "octahedral", donor_equivalence_groups=[list(range(6))], config=SiteEnumerationConfig(time_budget_s=-1.0))
    assert res_exp.truncated and res_exp.n_raw == 0 and not res_exp.dedup_truncated

    assert enumerate_unique_assignments(6, "octahedral", np.full((6, 6), -1, dtype=int), donor_equivalence_groups=[list(range(6))]).n_raw == 720
    assert len(enumerate_unique_assignments(6, "octahedral", np.full((6, 6), -1, dtype=int), donor_equivalence_groups=[list(range(6))]).assignments) == 1

    def _edta_hop(cross):
        h = np.full((6, 6), -1, dtype=int)
        for i, j, v in ((0, 1, 3), (0, 2, 3), (0, 3, 3), (1, 4, 3), (1, 5, 3), (2, 3, 6), (4, 5, 6),
                        (0, 4, cross), (0, 5, cross), (1, 2, cross), (1, 3, cross), (2, 4, 9), (2, 5, 9), (3, 4, 9), (3, 5, 9)):
            h[i, j] = h[j, i] = v
        return h

    r_real = enumerate_unique_assignments(6, "octahedral", _edta_hop(6), donor_equivalence_groups=[[0, 1], [2, 3, 4, 5]])
    r_unc = enumerate_unique_assignments(6, "octahedral", _edta_hop(11), donor_equivalence_groups=[[0, 1], [2, 3, 4, 5]])
    assert r_real.n_raw == 192 and len(r_real.assignments) == 1 and r_unc.n_raw == 192 and len(r_unc.assignments) == 1
    assert not is_pair_reachable(3, 180.0) and not is_pair_reachable(5, 180.0) and is_pair_reachable(5, 150.0) and is_pair_reachable(6, 180.0)

    ideal = IDEAL_GEOMETRIES["octahedral"]
    ang = site_angles_deg(ideal)
    bad = np.array([0, 2, 1, 3, 4, 5])
    new, repaired = repair_unreachable_assignment(bad, angular_cost_matrix(ideal[bad], ideal), ang, _edta_hop(6))
    assert repaired and assignment_is_reachable(new, ang, _edta_hop(6))
    assert not repair_unreachable_assignment(np.array(new), angular_cost_matrix(ideal[bad], ideal), ang, _edta_hop(6))[1]
