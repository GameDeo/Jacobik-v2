"""DB build determinism, containers, fit tasks, confidence stress driver, ledger, bends, and quantity modules."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jacobik.core import container as dbio
from jacobik.core import profile
from jacobik.stats.stats import StatsLookup, load_db

REPO = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get("JACOBIK_MAIN", str(REPO)))
BENDS_ENTRY = {
    "schema": "q7_bound_angles_v1",
    "provenance": {"min_n": 3},
    "entries": [
        {"key": ["C", "sp3", ["C", "N"], 5, 4], "n": 7, "mean_deg": 109.5, "sd_deg": 2.0, "n_parents": 4},
        {"key": ["N", "sp2", None, None, None], "n": 9, "mean_deg": 120.1, "sd_deg": 3.0, "n_parents": 5},
    ],
}
BENDS_TABLES = {"bends": BENDS_ENTRY}


def _entry(n=10, mean=1.5):
    return {"type": "gmm", "n": n, "k": 1, "n_eff": float(n), "weights": [1.0], "means": [mean], "vars": [0.01], "pooled": True, "src": "fit"}


def _db():
    return {"Q1:('Fe', 'N')": _entry(12, 2.0), "Q1:('Ni', 'N')": _entry(30, 1.9)}


def _write_embedded(path: Path, tables=None, meta_extra=None):
    meta = {"schema": "jacobik.constraints_container.1", "q1_variant": "m_sub_early", "q5_model": "wells_folded", **(meta_extra or {})}
    path.write_text(dbio.dumps_container(_db(), meta, BENDS_TABLES if tables is None else tables))
    return meta


def _sum_task(p):
    return sum(p)


def test_profiles_flags_env_and_sanity_defaults(tmp_path):
    from fit import sanity
    from fit.pipeline.extract import profile_env

    assert profile.snapshot()["fingerprint"] == "2c87e2940af2"
    d2 = profile.PROFILES["d2"]
    assert {k for k, v in d2.items() if v is True} == {"hapto_bond_gate", "hapto_label_canon", "hapto_chelate_exclusion", "q6_physical_all", "os_sanity", "fit_exclude_locked"}
    assert d2["q6_weight_14"] == 0.0 and "stereo_candidates" not in d2 and set(d2) <= set(profile.FIELDS)

    out_d2 = subprocess.run(
        [sys.executable, "-c", "from jacobik.core import profile; s=profile.snapshot(); print(s['profile'], s['fields']['hapto_bond_gate'], s['fields']['q6_physical_all'], s['fields']['config_remote'])"],
        env=dict(os.environ, JACOBIK_PROFILE="d2", PYTHONPATH=f"{REPO / 'src'}:{REPO}"), capture_output=True, text=True, check=True,
    ).stdout
    assert out_d2.split() == ["d2", "True", "True", "False"]

    flags = tmp_path / "flags.json"
    flags.write_text(json.dumps({"config_remote": True}))
    env = profile_env("production", flags)
    assert env == {"JACOBIK_PROFILE": "production", "JACOBIK_PROFILE_OVERRIDES": '{"config_remote":true}'}
    assert profile_env("d2")["JACOBIK_PROFILE_OVERRIDES"] == ""
    out_ov = subprocess.run(
        [sys.executable, "-c", "from jacobik.core import profile as p; s=p.snapshot(); print(s['profile'], s['fields']['config_remote'], s['differs_from_profile'])"],
        env=dict(os.environ, **env, PYTHONPATH=f"{REPO / 'src'}:{REPO}"), capture_output=True, text=True, check=True,
    ).stdout.split(None, 2)
    assert out_ov[0] == "production" and out_ov[1] == "True" and "config_remote" in out_ov[2]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"no_such_flag": 1}))
    r_bad = subprocess.run([sys.executable, "-c", "from jacobik.core import profile"], capture_output=True, text=True,
                           env=dict(os.environ, **profile_env("production", bad), PYTHONPATH=f"{REPO / 'src'}:{REPO}"))
    assert r_bad.returncode != 0 and "unknown fields" in r_bad.stderr

    assert [str(p) for p in (REPO / "src" / "fit").rglob("*.py") if "/mnt/e" in p.read_text()] == []
    assert profile.FIELDS["structure_sanity"][2] is True and sanity.STRUCTURE_SANITY is True

    def _ring(n=6, r=1.40, with_h=False):
        sym, pos = [], []
        for k in range(n):
            t = 2 * np.pi * k / n
            c = np.array([r * np.cos(t), r * np.sin(t), 0.0])
            sym.append("C"); pos.append(c)
            if with_h:
                sym.append("H"); pos.append(c * (1 + 1.09 / r))
        return sym, pos

    assert sanity.underbonded_carbon_fraction(*_ring(with_h=False)) == 1.0 and sanity.underbonded_carbon_fraction(*_ring(with_h=True)) == 0.0
    assert sanity.underbonded_carbon_fraction(["C"] * 6, [[1.2 * k, 0.0, 0.0] for k in range(6)]) == 0.0
    assert sanity.underbonded_carbon_fraction(*_ring(n=3)) == 0.0
    assert sanity.excluded_structure_codes([{"csd_code": "A", "structure_plausible": False}, {"csd_code": "B", "structure_plausible": True}, {"csd_code": "C"}]) == {"A"}


def test_container_formats_sidecars_and_table_checks(tmp_path, caplog, monkeypatch):
    from fit.pipeline.compare import compare_exact, load_container
    from jacobik.core.container import save_lossless_npz
    from tools.run.__main__ import _check_db_tables

    p = tmp_path / "constraints.json"
    meta = _write_embedded(p)
    assert dbio.read_container_head(p) == (meta, BENDS_TABLES)
    full = json.loads(p.read_text())
    assert list(full)[:2] == ["_meta", "_tables"] and len(full) == 2 + len(_db())

    db_ll = {k: dict(v) for k, v in _db().items()}
    db_ll["Q1:('Fe', 'N')"].update(ref_ll=-0.5, ref_sd=0.75)
    db_ll["Q5:('C', 'C')"] = {"type": "vmm", "n": 40, "means": [60.0], "kappas": [2.0], "weights": [1.0], "k": 1, "tail": 0.0, "model": "wells_folded", "n_eff": 40.0, "ref_ll": -1.25, "ref_sd": 0.5}
    npz = tmp_path / "c.npz"
    save_lossless_npz({"_meta": {"a": 1}, "_tables": BENDS_TABLES, **db_ll}, npz, header_meta={"a": 1})
    back, meta_b, tables_b = load_container(npz)
    assert compare_exact(back, db_ll, tables_b, BENDS_TABLES)["identical"] and meta_b == {"a": 1}
    assert back["Q1:('Ni', 'N')"].get("ref_ll") is None and back["Q5:('C', 'C')"]["ref_sd"] == 0.5

    c_empty = tmp_path / "empty.json"
    c_empty.write_text(dbio.dumps_container({}, {"a": 1}, {}))
    assert json.loads(c_empty.read_text()) == {"_meta": {"a": 1}, "_tables": {}} and dbio.read_container_head(c_empty) == ({"a": 1}, {})

    leg = tmp_path / "leg" / "constraints_2026-11-01.json"
    leg.parent.mkdir()
    leg.write_text(json.dumps(_db()))
    assert dbio.read_container_head(leg) == (None, None) and dbio.runtime_info(leg)["source"] == "none"
    db_notab = load_db(leg)
    assert "Q7" not in db_notab.priors and list(db_notab.tables) == [] and db_notab.meta == {}

    leg.with_name("constraints_2026-11-01.meta.json").write_text(json.dumps({"q1_variant": "m_sub_early", "q5_model": "wells_folded"}))
    leg.with_suffix(".bends.json").write_text(json.dumps(BENDS_ENTRY))
    a, b = load_db(leg), load_db(p)
    assert a.data == b.data and a.priors["Q7"] == b.priors["Q7"] and a.q1_variant == b.q1_variant == "m_sub_early" and list(a.tables) == ["bends"]

    real = tmp_path / "real" / "constraints_d2.json"
    real.parent.mkdir()
    _write_embedded(real)
    link = tmp_path / "other" / "constraints.json"
    link.parent.mkdir()
    link.symlink_to(real)
    assert "Q7" in load_db(link).priors and dbio.runtime_info(link)["tables"] == ["bends"]

    monkeypatch.setattr(profile, "FLAG_TABLES", {"table_flag": "bends"})
    monkeypatch.setitem(profile._VALUES, "table_flag", False)
    p_notab = tmp_path / "notab.json"
    _write_embedded(p_notab, tables={})
    assert dbio.missing_tables(dbio.runtime_info(p_notab)["tables"], flags={"table_flag": True}) == [("table_flag", "bends")]
    assert dbio.missing_tables(["bends"], flags={"table_flag": True}) == []
    monkeypatch.setitem(profile._VALUES, "table_flag", True)
    with caplog.at_level(logging.ERROR, logger="jacobik.stats.stats"):
        db_warn = load_db(p_notab)
    assert db_warn.data and "Q7" not in db_warn.priors and any("needs the 'bends' table" in r.getMessage() for r in caplog.records)

    plan = dict(env={"JACOBIK_DB": str(p_notab)}, repo=str(REPO), arms=[{"arm": "ok", "flags": {}}, {"arm": "q7", "flags": {"table_flag": True}}])
    with pytest.raises(SystemExit, match="lacks tables"):
        _check_db_tables(plan)
    plan["env"]["JACOBIK_DB"] = str(p)
    _check_db_tables(plan)


def test_pool_blocks_merge_reflik_and_tasks(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fit import pooling as PL
    from fit import q4
    from fit import qo as O
    from fit.extractor import QO_SCHEMA
    from fit.pipeline.compare import compare_exact
    from fit.pipeline.extract import merge_task, profile_env
    from fit.pipeline.fitting import TaskPool, fit_keys_task, make_blocks, pooled_fit_task, qo_fit_task, reflik_task
    from fit.quantity import add_reference_likelihoods

    env = profile_env("production")
    payloads = [list(range(i, i + 5)) for i in range(7)]
    with TaskPool(1, env) as p1:
        a = p1.map(_sum_task, payloads)
    with TaskPool(2, env) as p2:
        b = p2.map(_sum_task, payloads)
        assert p2.busy_s >= 0 and p2.max_worker_rss_mb > 0
    assert a == b == [sum(x) for x in payloads]

    blocks = make_blocks("Q2", [5000, 2000, 900, 300, 100, 50, 40, 30, 30, 30])
    assert [i for bl in blocks for i in bl] == list(range(10)) and blocks[0] == [0]

    def tbl(code, n, offset):
        return pa.table({"csd_code": [code] * n, "key": ["('C', 'C')"] * n, "value": [float(offset + i) for i in range(n)], "metal": ["Fe"] * n, "coord_geom": ["oct"] * n})

    f1, f2, f3 = tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "c.parquet"
    pq.write_table(tbl("AAAAAA", 4000, 0), f1)
    pq.write_table(tbl("BBBBBB", 4000, 5000), f2)
    pq.write_table(tbl("HOLDOUT", 10, 9000), f3)
    kw = dict(quantity="Q6", split="X1", exclude=["HOLDOUT"], files=[str(f1), str(f2), str(f3)])
    r1, r2 = merge_task(dict(kw, out=str(tmp_path / "o1.parquet"))), merge_task(dict(kw, out=str(tmp_path / "o2.parquet")))
    t1, t2 = pq.read_table(tmp_path / "o1.parquet"), pq.read_table(tmp_path / "o2.parquet")
    assert r1["rows"] == r2["rows"] == 5000 == t1.num_rows and t1.equals(t2) and "HOLDOUT" not in set(t1.column("csd_code").to_pylist())

    db = {f"Q1:('A', {i})": _entry(20, 1.0 + 0.1 * i) for i in range(5)}
    db["Q6P:('x',)"] = {"type": "pmf"}
    ref = {k: dict(v) for k, v in db.items()}
    assert add_reference_likelihoods(ref) == {"Q1": 5}
    got = reflik_task([(k, db[k]) for k in db])
    assert len(got) == 5 and all(ref[k]["ref_ll"] == ll and ref[k]["ref_sd"] == sd for k, ll, sd in got)

    rng = np.random.default_rng(3)
    items = [("('C', 2)", 100 + 30 * rng.standard_normal(80), np.ones(80)), ("('N', 1)", 120 + 5 * rng.standard_normal(10), np.ones(10))]
    fk = fit_keys_task(dict(quantity="Q4", items=items))
    assert fk["failures"] == [] and fk["entries"] == q4.fit_from_observations({k: (v, w) for k, v, w in items}, n_min=q4.N_MIN)
    pitem = ("('Fe', 'N')", rng.normal(2.0, 0.05, 120), np.ones(120), "gauss", 0.002 ** 2)
    assert pooled_fit_task(dict(items=[pitem]))["fits"] == [PL._fit_one(pitem)]
    G = rng.standard_normal((200, 3))
    assert qo_fit_task(dict(items=[(0, G, 7)], meta={"k_max": 3, "cov_floor": 0.0025, "w_min": 0.05}))["fits"][0][1] == O._fit_one(G, 7, 3, 0.0025, 0.05)

    ca, cb = _db(), _db()
    assert compare_exact(ca, cb, BENDS_TABLES, BENDS_TABLES)["identical"]
    cb["Q1:('Fe', 'N')"]["means"] = [2.0000001]
    assert compare_exact(ca, cb, BENDS_TABLES, BENDS_TABLES)["n_different"] == 1 and not compare_exact(ca, ca, BENDS_TABLES, {})["identical"]

    with_ids = "obs_id" in QO_SCHEMA.names
    assert with_ids
    cols = (["obs_id"] if with_ids else []) + ["csd_code", "hb_min", "ring", "sub", "metal", "geom", "anion", "k", "sub_elems"] + [f"g{i}" for i in range(6)]

    def _qof(n, haptic):
        rs = []
        for i in range(n):
            r = dict(obs_id=f"obs{i}", csd_code=f"C{i}", hb_min=3.0, ring=0, sub="N", metal="Fe", geom="oct", anion=0, k=1, sub_elems="C", g0=0.1, g1=0.2, g2=0.3, g3=None, g4=None, g5=None)
            r.update(raw_face=[0.0, 1.0, 2.0] if haptic else None, raw_face_elems=["C"] if haptic else None, raw_closed=True if haptic else None, raw_subs=[1.0] if haptic else None)
            rs.append(r)
        return pd.DataFrame(rs, columns=cols + ["raw_face", "raw_face_elems", "raw_closed", "raw_subs"])

    wf = _qof(3, True)
    wf["raw_closed"] = wf["raw_closed"].astype("boolean")
    pq.write_table(pa.Table.from_pandas(wf, schema=QO_SCHEMA, preserve_index=False), tmp_path / "qa.parquet")
    _qof(2, False).to_parquet(tmp_path / "qb.parquet")
    with pytest.raises(pa.ArrowInvalid):
        pa.concat_tables([pq.read_table(tmp_path / "qa.parquet"), pq.read_table(tmp_path / "qb.parquet")])
    mr = merge_task(dict(quantity="QO", split="X1", exclude=[], out=str(tmp_path / "qm.parquet"), files=[str(tmp_path / "qa.parquet"), str(tmp_path / "qb.parquet"), str(tmp_path / "qa.parquet")]))
    qm = pq.read_table(tmp_path / "qm.parquet")
    assert mr["rows"] == qm.num_rows == 8 and qm.schema.equals(QO_SCHEMA, check_metadata=False)


@pytest.mark.skipif(not (MAIN / "data/ref/tmQM/xyz/tmQM_X1s000.xyz").exists(), reason="needs data/ref")
def test_chunked_extraction_and_qo_schema(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fit import q6p as q6pmf
    from fit.extractor import QO_SCHEMA, extract_split, replay_q6
    from fit.pipeline.extract import chunk_tag

    data = MAIN / "data/ref/tmQM"
    kw = dict(progress=False, hypothesis=False, q6_hist_exclude=set(), q6_hist_weight={})
    single, chunked = tmp_path / "single", tmp_path / "chunked"
    extract_split(data, "X1s000", single, block_range=(0, 30), **kw)
    for i, rng in enumerate([(0, 12), (12, 30)]):
        extract_split(data, "X1s000", chunked, block_range=rng, q6_uncapped=True, out_tag=chunk_tag("X1s000", i), **kw)
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "QO"):
        whole = pq.read_table(single / f"observations_{q}_X1s000.parquet").to_pandas()
        parts = pa.concat_tables([pq.read_table(chunked / f"observations_{q}_{chunk_tag('X1s000', i)}.parquet") for i in range(2)]).to_pandas()
        assert whole.reset_index(drop=True).equals(parts.reset_index(drop=True)), q
    replay_q6([chunked / f"observations_Q6_{chunk_tag('X1s000', i)}.parquet" for i in range(2)], chunked / "q6.parquet")
    assert pq.read_table(single / "observations_Q6_X1s000.parquet").equals(pq.read_table(chunked / "q6.parquet"))
    for name in ("q6hist", "q6hist_m13"):
        whole, parts = q6pmf.load_hist([single / f"{name}_X1s000.npz"]), q6pmf.load_hist([chunked / f"{name}_{chunk_tag('X1s000', i)}.npz" for i in range(2)])
        assert set(whole) == set(parts) and all(np.allclose(whole[k], parts[k], rtol=0, atol=1e-9) for k in whole)
    assert pq.read_table(chunked / f"observations_QO_{chunk_tag('X1s000', 0)}.parquet").schema.equals(QO_SCHEMA, check_metadata=False)


def test_filter_task_charge_plausibility_compare_and_build_db(tmp_path, monkeypatch):
    import fit.build_db as build_db
    from fit.build_db import _align_mixture_components, compare_databases
    from fit.pipeline import extract as ex
    from fit.pipeline import fitting as fs
    from fit.quantity import add_reference_likelihoods
    from fit.sanity import charge_is_plausible, excluded_charge_codes, filter_fit_frame

    pd.DataFrame({"csd_code": ["A", "B", "C", "A", "D"], "key": ["k"] * 5, "value": [1.0] * 5}).to_parquet(tmp_path / "observations_Q4_X1.parquet")
    code = (
        "from fit.pipeline.fitting import filter_task; import json; "
        f"print(json.dumps(filter_task(dict(quantity='Q4', files=['{tmp_path}/observations_Q4_X1.parquet'], "
        f"out='{tmp_path}/o.parquet', charge_bad=['A', 'Z'], locked=['B', 'D']))))"
    )
    legacy = json.dumps({"os_sanity": False, "fit_exclude_locked": False, "structure_sanity": False})
    for prof, overrides, want in (("production", legacy, (0, 0, 5)), ("production", "", (1, 2, 1)), ("d2", "", (1, 2, 1))):
        r = json.loads(subprocess.run([sys.executable, "-c", code],
                                      env=dict(os.environ, JACOBIK_PROFILE=prof, JACOBIK_PROFILE_OVERRIDES=overrides, PYTHONPATH=f"{REPO / 'src'}:{REPO}"),
                                      capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        assert (r["parents_removed_charge_implausible"], r["parents_removed_locked"], r["rows"]) == want and r["rows_in"] == 5 and r["parents_in"] == 4

    assert not charge_is_plausible("Zn", 14) and charge_is_plausible("Zn", 2)
    assert excluded_charge_codes([{"csd_code": "NEPNEF", "charge_plausible": False}, {"csd_code": "OK", "charge_plausible": True}]) == {"NEPNEF"}
    fr = pd.DataFrame({"csd_code": ["locked", "ordinary"]})
    assert len(filter_fit_frame(fr, charge_bad={"ordinary"}, locked={"locked"})) == 0
    assert len(filter_fit_frame(fr, charge_bad={"ordinary"}, locked={"locked"}, os_sanity=False, exclude_locked=False)) == 2

    db_ll = {"Q1:('C_donor',)": {"means": [2.0], "vars": [0.01], "weights": [1.0]}, "QO:(1,)": {"means": [[0.0]], "covs": [[[0.01]]], "weights": [1.0]}}
    assert add_reference_likelihoods(db_ll) == {"Q1": 1} and np.isfinite(db_ll["Q1:('C_donor',)"]["ref_ll"]) and "ref_ll" not in db_ll["QO:(1,)"]

    tgt = {"Q3:('class',)": {"type": "gmm_cos", "k": 2, "means": [-0.8, 0.3], "vars": [0.02, 0.1], "weights": [0.4, 0.6], "diagnostic": ["keep-first", "keep-second"]}}
    reo = {"Q3:('class',)": {"type": "gmm_cos", "k": 2, "means": [0.3, -0.8], "vars": [0.1, 0.02], "weights": [0.6, 0.4], "diagnostic": ["keep-first", "keep-second"]}}
    assert _align_mixture_components(reo["Q3:('class',)"]["means"] and reo["Q3:('class',)"], tgt["Q3:('class',)"])[0]["diagnostic"] == ["keep-first", "keep-second"]
    assert compare_databases(reo, tgt, only_quantities=["Q3"])["Q3"]["different"] == 0
    assert compare_databases({"Q3:('class',)": {**reo["Q3:('class',)"], "weights": [0.61, 0.39]}}, tgt, only_quantities=["Q3"])["Q3"]["different"] == 1

    ref_dir, obs_dir, out_dir = tmp_path / "ref", tmp_path / "obs", tmp_path / "out"
    ref_dir.mkdir(); obs_dir.mkdir()
    (tmp_path / "locked.json").write_text("{}")
    (tmp_path / "frame.json").write_text("{}")
    monkeypatch.setenv("JACOBIK_PROFILE", "production")
    monkeypatch.delenv("JACOBIK_PROFILE_OVERRIDES", raising=False)
    bends_calls = []
    payload = {"schema": "q7_bound_angles_v1", "provenance": {"fit_frame_sha256": hashlib.sha256(b"{}").hexdigest(), "min_n": 3, "extraction_report": {"parents_requested": 1, "parents_failed": 0, "n_entries": 1}}, "entries": BENDS_ENTRY["entries"][:1]}
    monkeypatch.setattr(ex, "list_shards", lambda ref: [])
    monkeypatch.setattr(fs, "run_census", lambda pool, ref, shards: [])
    monkeypatch.setattr(fs, "run_filter", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(fs, "run_expand", lambda *a, **k: {})
    monkeypatch.setattr(fs, "fit_stage", lambda *a, **k: ({"Q1:test": {"type": "gmm", "n": 1, "means": [1.0], "vars": [1.0], "weights": [1.0], "k": 1, "n_eff": 1.0, "pooled": True, "src": "fit"}}, [], {"Q1": 1}))
    monkeypatch.setattr(fs, "run_reference_ll", lambda pool, db: {})
    monkeypatch.setattr(fs, "run_bends", lambda pool, frame, ref, **kw: bends_calls.append(kw) or payload)
    assert build_db.main(["--ref-dir", str(ref_dir), "--out-dir", str(out_dir), "--obs-dir", str(obs_dir), "--workers", "1", "--locked-path", str(tmp_path / "locked.json"), "--fit-frame-path", str(tmp_path / "frame.json"), "--quantities", "Q1", "Q7"]) == 0
    assert [c["min_n"] for c in bends_calls] == [3]
    meta_o, tab_o = dbio.read_container_head(out_dir / "constraints.json")
    assert tab_o == {"bends": payload} and meta_o["inputs"]["fit_frame_sha256"] == hashlib.sha256(b"{}").hexdigest() and not (out_dir / "constraints.bends.json").exists()
    for name in ("constraints.json", "constraints.npz"):
        db_o = load_db(out_dir / name)
        assert db_o.q1_variant == "m_sub_early" and len(db_o.priors["Q7"]) == 1
    assert json.loads((out_dir / "constraints.meta.json").read_text())["entries_sha256"] == meta_o["entries_sha256"]


def test_confidence_design_eval_summary_damage_and_registry():
    from fit.pipeline.confidence import design, evaluate, perturb_coordinates, perturb_observations, register, summarize, summarize_registered, wilson

    trial = design(["a" * 32 + "_La", "a" * 32 + "_Ce", "OTHER"], replicates=2)
    assert len(trial) == 2 and all(t["support_count"] == 2 and t["actual_fraction"] == 0.5 for t in trial)

    clean = dict(scores=[[1, 0]] * 3, energies_kj=[[0, 100]] * 3, broken_coordination=[[False, True]] * 3)
    tr = {**clean, "scores": [[0, 1], [1, 0], [1, 0]]}
    assert not evaluate(clean, tr)["severe"]
    tr["scores"][1] = [0, 1]
    assert evaluate(clean, tr)["severe"] and not evaluate(tr, tr)["severe"]
    mixed = dict(scores=[[0, 1], [0, 1], [1, 0]], energies_kj=[[0, 100], [0, 0], [0, 0]], broken_coordination=[[False, False], [False, True], [False, False]])
    assert evaluate(clean, mixed)["severe"]
    with pytest.raises(ValueError):
        evaluate(clean, {**tr, "scores": [[float("nan"), 0]] * 3})

    r0 = dict(status="ok", trial_id="a", scenario="parent_removal", actual_fraction=0.01, affected_count=1, support_count=100, outcome={"severe": False})
    out = summarize([r0], level="fixed_pool")
    assert out["first_tested_failing_fraction"] == {} and out["censored_at_maximum_tested"] == {"parent_removal": 0.01} and out["dose_response"][0]["interval95"][1] > 0.5
    assert summarize([{"status": "failed", "trial_id": "b"}], level="generation")["candidate_blocked"] and wilson(0, 32)[1] > 0
    for bad in ([r0, r0], []):
        with pytest.raises(ValueError):
            summarize(bad, level="generation")

    rows = pd.DataFrame({"csd_code": ["A", "A", "B"], "value": [1.0, 2.0, 3.0]})
    assert perturb_observations(rows, {"affected_parents": ["A"]}, "parent_removal").value.tolist() == [3.0]
    assert perturb_observations(rows, {"affected_parents": ["A"]}, "observed_input_error", rows.assign(value=[4.0, 5.0, 6.0])).value.tolist() == [3.0, 4.0, 5.0]
    with pytest.raises(ValueError):
        perturb_observations(rows, {"affected_parents": ["A"]}, "geometry_0.02A")
    xyz = np.zeros((100, 3))
    pert = perturb_coordinates(xyz, seed=11)
    assert np.max(np.linalg.norm(pert, axis=1)) <= 0.02 and np.array_equal(pert, perturb_coordinates(xyz, seed=11))

    rows_clean = pd.DataFrame({"csd_code": ["A", "B", "C"], "value": [1.0, 2.0, 3.0]})
    for sc in ("parent_removal", "observed_input_error"):
        pd.testing.assert_frame_equal(perturb_observations(rows_clean, {"affected_parents": []}, sc, replacements=rows_clean.iloc[:0]).reset_index(drop=True), rows_clean)
    xyz_n = np.random.default_rng(1).normal(size=(20, 3))
    assert np.array_equal(perturb_coordinates(xyz_n, seed=11, maximum_A=0.0), xyz_n)
    with pytest.raises(ValueError):
        perturb_observations(rows_clean, {"affected_parents": ["Z"]}, "parent_removal")

    codes = [f"P{i:03d}" for i in range(100)]
    scope = dict(quantity="Q1", resolved_key="k", parent_code="LUQVOK")
    reg_a, reg_b = register(scope, codes, {"db": "a" * 64}, replicates=4), register(scope, codes, {"db": "a" * 64}, replicates=4)
    assert reg_a["registry_id"] == reg_b["registry_id"] and len(reg_a["trials"]) == 3 * 4 * len({1, 1, 2, 5})
    with pytest.raises(ValueError):
        register(scope, codes, {"db": "short"})
    with pytest.raises(ValueError):
        register({"quantity": "Q1"}, codes, {"db": "a" * 64})

    reg2 = register(scope, codes, {"db": "a" * 64}, replicates=2)
    recs = lambda drop=0: [dict(status="ok", trial_id=t["trial_id"], scenario=t["scenario"], actual_fraction=t["actual_fraction"], affected_count=t["affected_count"], support_count=reg2["support_count"], broken_coordination_rule=reg2["broken_coordination_rule"], outcome={"severe": False}) for t in reg2["trials"]][drop:]
    assert not summarize_registered(reg2, recs(), level="fixed_pool")["candidate_blocked"]
    short = summarize_registered(reg2, recs(drop=3), level="fixed_pool")
    assert short["candidate_blocked"] and short["reconciliation"]["missing"] == 3 and len(short["execution_failures"]) == 3
    with pytest.raises(ValueError):
        summarize_registered(reg2, recs() + recs()[:1], level="fixed_pool")
    bad_r = recs(); bad_r[0]["support_count"] = 7
    assert summarize_registered(reg2, bad_r, level="fixed_pool")["candidate_blocked"]


def test_stable_sampling_q6_reservoir_and_closure_refit():
    from fit import pooling as PL
    from fit import q1
    from fit.pipeline import confidence as stress

    script1 = """
from fit.expander import _AncestorReservoir
from fit.sanity import observation_id
def rows(reverse=False,removed=None):
    seq=[(bucket,i) for i in range(100) for bucket in ('A','B')]
    if reverse: seq.reverse()
    r=_AncestorReservoir(cap=10)
    for bucket,i in seq:
        if (bucket,i)==removed: continue
        oid=observation_id('tmQM',str(i),'Q1',bucket,0)
        r.add_ancestor(bucket,'x',(str(i),bucket,float(i),'Pd','square_plan','X1',1,oid))
    return r.to_rows()
a=rows()
assert a==rows(True)
assert [r for r in a if r[1]=='A']==[r for r in rows(removed=('B',0)) if r[1]=='A']
assert len(set(a)-set(rows(removed=('A',int(next(r for r in a if r[1]=='A')[0])))))==1
from pathlib import Path
import tempfile, pyarrow.parquet as pq
from fit.extractor import _empty_buffers, _open_writers, _flush_q1_q5, observation_schema, QO_SCHEMA, _qo_table
from fit.expander import _expand_into_reservoir
with tempfile.TemporaryDirectory() as directory:
    buffers=_empty_buffers()
    for q, buffer in buffers.items():
        buffer.update(csd_code=['A','A'],key=["('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')"]*2,
                      value=[2.0,2.1],metal=['Pd']*2,coord_geom=['square_plan']*2,
                      obs_id=[observation_id('tmQM','A',q,'key',i) for i in range(2)])
    writers=_open_writers(Path(directory),'X1'); _flush_q1_q5(writers,buffers)
    for w in writers.values(): w.close()
    table=pq.read_table(Path(directory)/'observations_Q1_X1.parquet')
    assert table['obs_id'][0].as_py()!=table['obs_id'][1].as_py()
    reservoir=_AncestorReservoir(cap=1); _expand_into_reservoir('Q1',table.to_batches()[0],reservoir)
    assert reservoir.l0_count()==2 and 'obs_id' not in observation_schema('Q6').names
    try:
        _expand_into_reservoir('Q1',table.drop(['obs_id']).to_batches()[0],_AncestorReservoir())
    except ValueError: pass
    else: raise AssertionError('legacy rows silently accepted in stable mode')
    assert _qo_table([dict(csd_code='A',obs_id='qo-source-id')],QO_SCHEMA.names)['obs_id'][0].as_py()=='qo-source-id'
"""
    env_st = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO}", OMP_NUM_THREADS="1", JACOBIK_PROFILE_OVERRIDES=json.dumps({"stable_ancestor_sampling": True}))
    subprocess.run([sys.executable, "-c", script1], env=env_st, check=True)

    script2 = """
import json, random
from fit.extractor import _Q6Reservoir
r = _Q6Reservoir()
rng = random.Random(5)
for i in range(60000): r.add("('C', 'H')", rng.random(), f"P{i}", "Pd", "g")
for i in range(300): r.add("('Zz', 'Yy')", rng.random(), f"P{i}", "Pd", "g")
print(json.dumps([list(x) for x in r.to_rows()]))
"""
    outs = [subprocess.run([sys.executable, "-c", script2], env=dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO}", OMP_NUM_THREADS="1", JACOBIK_PROFILE_OVERRIDES=json.dumps({"stable_ancestor_sampling": fl})), check=True, capture_output=True, text=True).stdout for fl in (False, True)]
    assert outs[0] == outs[1] and len(json.loads(outs[0])) > 300

    rng = np.random.default_rng(3)
    rows = [(f"A{i:03d}", "('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')", 2.05 + 0.03 * rng.normal()) for i in range(80)]
    rows += [(f"B{i:03d}", "('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'N_pyridine')", 2.08 + 0.03 * rng.normal()) for i in range(4)]
    raw = pd.DataFrame(rows, columns=["csd_code", "key", "value"]).assign(metal="Pd", coord_geom="square_plan")
    raw["obs_id"] = [f"obs{i}" for i in range(len(raw))]
    expanded = stress.expand_rows(raw, "Q1")
    pf = PL.PooledFitter("Q1", expanded, n_indep=q1.N_INDEP)
    sparse = "('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'N_pyridine')"
    assert stress.closure_keys(pf, sparse, q1.DEFAULT_VARIANT)[0] == sparse and len(stress.closure_keys(pf, sparse, q1.DEFAULT_VARIANT)) >= 2
    entry = stress.refit_closure(pf, sparse)
    ref = PL.PooledFitter("Q1", expanded, n_indep=q1.N_INDEP); ref.fit_independent()
    full = ref.entries(r=q1.DEFAULT_POOLING_R, variant=q1.DEFAULT_VARIANT)[sparse]
    assert entry["src"] == full["src"] == "adapted" and all(np.allclose(entry[f], full[f], rtol=0, atol=1e-12) for f in ("means", "vars", "weights"))
    dmg = stress.expand_rows(raw[raw.csd_code.isin([f"A{i:03d}" for i in range(40)] + [f"B{i:03d}" for i in range(4)])], "Q1")
    changed = stress.refit_closure(PL.PooledFitter("Q1", dmg, n_indep=q1.N_INDEP), sparse)
    assert not np.allclose(changed["means"], entry["means"], atol=1e-9) or not np.allclose(changed["vars"], entry["vars"], atol=1e-9)


def test_ds_ledger_and_stress_driver(monkeypatch):
    from fit import sanity as eligibility
    from fit.extractor import LEDGER_SCHEMA, ledger_rows
    from fit.pipeline import confidence as stress_driver

    atoms = [SimpleNamespace(idx=i, symbol=s) for i, s in enumerate(["Pd", "N", "O", "O"])]
    mol = SimpleNamespace(csd_code="TEST01", metal_symbol="Pd", atoms=atoms, donor_identity_events=[
        {"kind": "local_donor_identity", "before": [3], "after": [2], "motif_atoms": [2, 3]},
        {"kind": "rejected_face_kappa_preservation", "face": [1, 2], "added": [2], "center": 1},
    ])
    kv = {"Q1": [(("Pd", "O"), 2.0), (("Pd", "O"), 2.1)], "Q2": [(("a",), 1.0)]}
    rows_l = ledger_rows(mol, "tmQM", kv, ["qo1"])
    assert [r["kind"] for r in rows_l] == ["local_donor_identity", "rejected_face_kappa_preservation"]
    assert json.loads(rows_l[0]["before"]) == [[3, "O"]] and json.loads(rows_l[0]["after"]) == [[2, "O"]] and json.loads(rows_l[1]["before"]) == [[1, "N"], [2, "O"]]
    assert json.loads(rows_l[0]["dependent"])["Q1"]["n"] == 2 and set(rows_l[0]) == set(LEDGER_SCHEMA.names)
    assert ledger_rows(SimpleNamespace(csd_code="T", metal_symbol="Pd", atoms=atoms, donor_identity_events=[]), "tmQM", kv, []) == []

    pol_rows = [{"csd_code": "A", "kind": "weak_declared_identity_retained", "disposition": "review", "dependent": "{}"},
                {"csd_code": "B", "kind": "local_donor_identity", "disposition": "correction", "dependent": json.dumps({"Q1": {"n": 4, "keys": []}})}]
    assert eligibility.eligibility_codes(pol_rows) == set()
    with pytest.raises(ValueError):
        eligibility.eligibility_codes(pol_rows, ["weak_declared_identity_retained"])
    rc = eligibility.reason_counts(pol_rows, eligibility.load_review())
    assert rc["excluded_parents"] == 0 and rc["review_ledger"]["unresolved"] == [] and rc["review_ledger"]["decided"] == ["AGAPEG", "BADQIH", "ILEPOE", "JAYCEV", "NUHPEL"]

    kinds = set()
    for f in (REPO / "src/jacobik/topology/mapping.py", REPO / "src/jacobik/topology/donor_classifier.py"):
        kinds |= set(re.findall(r'"kind":\s*"([a-z_]+)"', f.read_text())) | set(re.findall(r'if retained else\s*"([a-z_]+)"', f.read_text()))
    kinds.discard("")
    assert kinds and all(k in eligibility.EVENT_KINDS for k in kinds)
    rev = eligibility.load_review()
    assert all(e["fit_effect"] == "none" for e in rev["entries"])
    assert {e["code"]: e.get("target_overlay") for e in rev["entries"] if e["status"] == "decided"} == {"AGAPEG": "kappa1", "ILEPOE": "kappa1", "JAYCEV": None, "BADQIH": None, "NUHPEL": None}

    assert set(stress_driver.fitdist_kwargs({"type": "gmm", "n": 5, "level": 1, "means": [2.0], "vars": [0.01], "weights": [1.0], "src": "fit", "k": 1})) == {"type", "n", "level", "means", "vars", "weights"}
    codes = [f"P{i:03d}" for i in range(40)]
    raw_df = pd.DataFrame({"csd_code": codes, "key": ["('K',)"] * 40, "value": np.linspace(2.0, 2.2, 40)})
    reg = stress_driver.register({"quantity": "Q1", "resolved_key": "('K',)", "parent_code": "T"}, codes, {"db": "a" * 64}, scenarios=("parent_removal", "geometry_0.02A"), replicates=2, bootstrap_replicates=2)
    scores = lambda sh=0.0: dict(scores=np.tile([[1.0, 0.5, 0.2]], (3, 1)) + sh * np.array([[0.0, 1.0, 0.0]]), energies_kj=np.tile([[10.0, 60.0, 90.0]], (3, 1)), broken_coordination=np.zeros((3, 3), bool))
    monkeypatch.setattr(stress_driver, "refit_resolved", lambda *a, **k: {"type": "gmm", "n": 1, "level": 1})
    out_b = stress_driver.run_registered(reg, raw_df, lambda fit: scores(), stable=True)
    assert out_b["summary"]["candidate_blocked"] and not out_b["reconciliation"]["complete"]

    calls = {"n": 0}
    def _flaky(fit):
        calls["n"] += 1
        if fit is not None and calls["n"] == 3:
            raise RuntimeError("pool unreadable")
        return scores()
    out_err = stress_driver.run_registered(reg, raw_df, _flaky, only=[reg["trials"][0]["trial_id"], reg["trials"][1]["trial_id"]])
    assert any(r["status"] == "failed" and "pool unreadable" in r["reason"] for r in out_err["records"])
    pr_ids = [t["trial_id"] for t in reg["trials"] if t["scenario"] == "parent_removal"]
    out_sev = stress_driver.run_registered(reg, raw_df, lambda fit: scores() if fit is None else scores(2.0), only=pr_ids)
    assert len(out_sev["records"]) == len(pr_ids) and all(r["outcome"]["severe"] for r in out_sev["records"])

    seen_n = {}
    geo_ids = [t["trial_id"] for t in reg["trials"] if t["scenario"].startswith("geometry")]
    monkeypatch.setattr(stress_driver, "refit_resolved", lambda damaged, *a, **k: seen_n.setdefault("n", len(damaged)) and {"type": "gmm", "n": 1, "level": 1})
    out_geo = stress_driver.run_registered(reg, raw_df, lambda fit: scores(), stable=True, only=geo_ids, replacements=raw_df.assign(value=raw_df["value"] + 0.01))
    geo = [r for r in out_geo["records"] if r["scenario"].startswith("geometry")]
    assert geo and all(r["status"] == "ok" for r in geo) and seen_n["n"] == len(raw_df)


def test_bound_bends_and_stress_geometry(tmp_path, monkeypatch):
    import fit.bends as bends
    from fit.bends import bound_angle_fallbacks, bound_angle_key, build_bound_angle_table, build_bound_angle_table_from_frame, write_bends_sidecar, write_bound_angle_sidecar
    from fit.parser import _iter_bo_blocks, _iter_xyz_blocks
    from tools.jdev.budget.common import stats_from_db

    sys.path.insert(0, str(REPO / "research/campaigns/ds"))
    import stress_geometry as SG

    key = bound_angle_key("C", "sp2", ["N", "C"], 5, 4)
    obs = [{"key": key, "angle_deg": 118.0, "code": "AAA"}, {"key": key, "angle_deg": 122.0, "code": "BBB"}]
    table = build_bound_angle_table(obs)
    assert table[key]["n"] == 2 and table[key]["mean_deg"] == 120.0 and ("C", "sp2", ("C", "N"), None, None) in table
    assert bound_angle_fallbacks(key)[0] == key and bound_angle_fallbacks(key)[-1] == ("C", None, None, None, None)
    assert not build_bound_angle_table([{"key": bound_angle_key("N", "sp3", ["C", "C"], 6, 4), "angle_deg": 109.0}], min_n=2)
    payload = write_bound_angle_sidecar(tmp_path / "angles.json", obs, {"source": "unit-test"})
    assert json.loads((tmp_path / "angles.json").read_text()) == json.loads(json.dumps(payload))

    db_p = tmp_path / "constraints.json"
    db_p.write_text("{}")
    write_bound_angle_sidecar(db_p.with_suffix(".bends.json"), [{"key": key, "angle_deg": 121.0, "code": "AAA"}])
    assert ("C", "sp2", ("C", "N"), None, None) in StatsLookup(load_db(db_p)).q7_bound_angle_table()
    assert ("C", "sp2", ("C", "N"), None, None) in stats_from_db(db_p).q7_bound_angle_table()

    monkeypatch.setattr(bends, "extract_bound_angle_observations", lambda mol, code: [{"key": key, "angle_deg": 119.0, "code": code}])
    frame = {"main_allocation": {"splits": {"surface_fit": [{"code": "AAA", "shard": "s1", "dataset": "tmQM"}, {"representative_code": "BBB", "representative_shard": "s2", "dataset": "LNf"}]}}}
    calls = []
    tab_f, rep_f = build_bound_angle_table_from_frame(frame, lambda c, s, d: calls.append((c, s, d)) or c)
    assert calls == [("AAA", "s1", "tmQM"), ("BBB", "s2", "LNf")] and tab_f[key]["n_parents"] == 2 and rep_f["parents_failed"] == 0
    fp = tmp_path / "frame.json"
    fp.write_text(json.dumps(frame))
    calls.clear()
    res = write_bends_sidecar(fp, tmp_path / "raw", tmp_path / "out", load_complex=lambda c, s, d: calls.append((c, s, d)) or c, min_n=2)
    assert len(json.loads((tmp_path / "out/constraints.bends.json").read_text())["entries"]) == 5 and res["provenance"]["min_n"] == 2

    blk = dict(dataset="tmQM", comment="CSD_code = AAAAAA | q = 0", atoms=[("Pd", 0.0, 0.0, 0.0), ("N", 2.0, 0.0, 0.0), ("C", 0.0, 2.0, 0.5)], bo=["1 Pd 3.0 N 2 0.5", "2 N 3.0 Pd 1 0.5", "3 C 4.0"], smiles="C")
    pa, pb, pc = SG.perturbed_block(blk, "t1", "AAAAAA"), SG.perturbed_block(blk, "t1", "AAAAAA"), SG.perturbed_block(blk, "t2", "AAAAAA")
    dists = np.linalg.norm(np.array([x[1:] for x in pa]) - np.array([x[1:] for x in blk["atoms"]]), axis=1)
    assert pa == pb and pa != pc and dists.max() <= SG.MAX_A + 1e-12 and (dists > 0).all()
    assert SG.perturbed_block(blk, "t1", "AAAAAA", maximum_A=0.0) == blk["atoms"]
    ds_mini = SG.write_mini_dataset(tmp_path / "mini", "tmQM", [("AAAAAA", blk)], "t1", 0.0)
    (cm, xlines), = list(_iter_xyz_blocks(ds_mini / "xyz" / "tmQM_T.xyz"))
    (cd, blines), = list(_iter_bo_blocks(ds_mini / "bo" / "tmQM_T.BO"))
    assert cm == blk["comment"] and cd == "AAAAAA" and blines == blk["bo"] and [l.split()[0] for l in xlines] == ["Pd", "N", "C"]


def test_fit_quantity_modules_ladders_fittings_and_audit():
    from fit import bends, q1, q2, q3, q4, q5, q6, q6p, qo

    rng = np.random.default_rng(42)

    k1 = ("Pt", "Pt", 2, "P_tert", "square_plan", "none")
    assert ("Pt", "P_tert") in q1.ladder_candidates(k1, variant="m_sub_early") and q1.ancestor_rows(k1)
    v1 = rng.normal(2.32, 0.03, 40)
    assert q1.fit_entry(v1, var_floor=0.002 ** 2)["type"] == "gmm"
    assert any(k.startswith("Q1:") for k in q1.fit_from_observations(pd.DataFrame({"key": [repr(k1)] * 40, "value": v1, "csd_code": [f"C{i}" for i in range(40)], "metal": ["Pt"] * 40, "coord_geom": ["square_plan"] * 40}), variant="m_sub_early", r=4.0, n_indep=10))

    k2 = ("square_plan", "c90", "same", "Pt", "P_tert", "P_tert")
    v2 = rng.normal(90.0, 3.0, 40)
    assert k2 in q2.ladder_candidates(k2) and q2.ancestor_rows(k2) and q2.fit_entry(v2)["type"] == "tmix_cos"
    assert any(k.startswith("Q2:") for k in q2.fit_from_observations(pd.DataFrame({"key": [repr(k2)] * 40, "value": v2, "csd_code": [f"C{i}" for i in range(40)], "metal": ["Pt"] * 40, "coord_geom": ["square_plan"] * 40}), r=4.0, n_indep=10))

    k3 = ("O_neutral_sigma", "Ru", "octahedral", 0)
    v3 = rng.normal(172.0, 4.0, 30)
    assert q3.ladder_candidates(k3, variant="anionic_context") and q3.ancestor_rows(k3) and q3.fit_entry(v3)["type"] == "gmm_cos"
    assert f"Q3:{k3!r}" in q3.fit_from_observations(pd.DataFrame({"key": [repr(k3)] * 30, "value": v3}), n_min=15)

    k4 = ("N_tert", "octahedral", 3, 0)
    v4 = rng.normal(109.5, 3.0, 30)
    assert q4.ladder_candidates(k4, variant="anionic_context") and q4.ancestor_rows(k4) and q4.fit_entry(v4)["type"] == "gmm"
    assert f"Q4:{k4!r}" in q4.fit_from_observations(pd.DataFrame({"key": [repr(k4)] * 30, "value": v4}), n_min=15)

    k5 = ("C_sp3", "C_sp3", "C_sp3", "C_sp3", "free")
    v5 = rng.normal(180.0, 8.0, 35) % 360.0
    assert q5.ladder_candidates(k5) and q5.ancestor_rows(k5) and q5.fit_entry(v5)["model"] == "wells_folded"
    assert f"Q5:{k5!r}" in q5.fit_from_observations(pd.DataFrame({"key": [repr(k5)] * 35, "value": v5}), n_min=15)

    d6 = rng.normal(3.2, 0.4, 40)
    assert q6.fit_entry(d6, model="p5")["model"] == "p5"
    obs6 = {repr(("Pt", "C")): (d6, np.ones(len(d6)))}
    assert len(q6.add_pooled_metal_keys(obs6, n_min=15)) >= 2 and any("Q6:" in k for k in q6.fit_from_observations(obs6, n_min=15, pool_metals=True))

    assert len(qo.ladder_level_keys("P_tert", "Pt", "square_plan", 0, 3, 0)) > 1
    assert qo.fit_entry(rng.normal(0.0, 0.2, (30, 3)), seed=123, k_max=2)["type"] == "mvgmm"
    assert q6p.create_contact_accumulator() is not None and q6p.create_m13_accumulator() is not None
    assert isinstance(q6p.fit_from_observations(contact_hist={"('C', 'C')|1-4|all": np.ones(50) * 2.0}, n_min=15), dict)

    assert bends.BEND_K[("in", "sp2")] == 1300.0 and bends.BEND_K[("out", "sp2")] == 310.0 and bends.LEAN_FREE_DEG[2] == 57.0
    beta = np.radians([-12, -8, -4, 4, 8, 12])
    assert abs(bends.fit_bend_stiffness(0.5 * 1200.0 * (beta ** 2), beta) - 1200.0) < 1e-6
    assert bends.fit_from_observations([{"pivot": "C", "hyb": "sp2", "axis": "in", "k": 1310.0}, {"pivot": "C", "hyb": "sp2", "axis": "in", "k": 1290.0}])["C|sp2|in"]["median"] == 1300.0

    audit = (REPO / "src" / "fit" / "TABLES_AUDIT.md").read_text()
    assert all(tok in audit for tok in ("NCI", "QG", "QP", "QB", "QV", "QVR", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO", "Q6P"))
