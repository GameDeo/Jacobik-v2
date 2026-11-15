import numpy as np
import pytest

from tools.jdev.rescore import _pool_metrics


def test_two_pool_metrics():
    first = _pool_metrics("ABC", 11, np.array([1., 2., 3.]), np.array([2., 3., 4.]))
    second = _pool_metrics("XYZ", 12, np.array([1., 2., 3.]), np.array([4., 2., 1.]))
    rows = sorted([first, second], key=lambda row: row["spearman_rho"])
    assert [r["code"] for r in rows] == ["XYZ", "ABC"]
    assert rows[0]["spearman_rho"] == pytest.approx(-1)
    assert not rows[0]["top1_unchanged"]
    assert rows[0]["top1_delta_score"] == 1
    assert rows[0]["max_abs_delta_score"] == 3
    assert rows[1]["top1_unchanged"]
    assert rows[1]["n_rows"] == 3


def test_fallback_is_rejected():
    import logging
    from tools.jdev.rescore import _RejectEmbedFallback
    handler = _RejectEmbedFallback()
    record = logging.LogRecord("jacobik.builder.engine", logging.WARNING, "", 1,
                               "GFN2 embed refinement failed for %s", ("X",), None)
    with pytest.raises(RuntimeError, match="GFN2 embed refinement failed"):
        handler.emit(record)


def test_rescore_returns_sorted_two_pool_table(tmp_path, monkeypatch):
    import json
    import torch
    from tools.jdev import rescore
    from jacobik.stats import stats
    from jacobik.core import env as jinputs
    from tools.jgen import inputs
    from tools.jdev import multiseed
    from tools.jdev.budget import pools

    (tmp_path / "manifest.json").write_text(json.dumps({"tasks": [
        {"status": "success", "split": "surface_fit", "code": "ABC", "seed": s}
        for s in (11, 12, 13)]}))
    (tmp_path / "frame.json").write_text(json.dumps({"main_allocation": {"splits": {
        "surface_fit": [{"code": "ABC"}]}}}))
    for seed, x in [(11, [1., 2., 3.]), (12, [3., 2., 1.])]:
        np.savez(tmp_path / f"ABC_{seed}_control.npz", X=np.array(x), asg=np.zeros(3))
    monkeypatch.setattr(stats, "load_db", lambda db: db)
    monkeypatch.setattr(stats, "StatsLookup", lambda db: db)
    monkeypatch.setattr(inputs, "load_mol", lambda *a, **kw: object())
    monkeypatch.setattr(jinputs, "build_builder_and_inputs", lambda mol: (object(), None, None))
    monkeypatch.setattr(multiseed, "prepare", lambda *a, stats, **kw: (stats, None, None, None))
    monkeypatch.setattr(multiseed, "seed_rows", lambda builder, ctx, *a: ((None, None, None, ctx), None))
    monkeypatch.setattr(pools, "rank_terms", lambda sc, x, rg: (x if sc == "target" else torch.tensor([1., 2., 3.]), {}))
    result = rescore.rescore_60_pools("test", "target", pools_dir=tmp_path,
        frame_path=tmp_path / "frame.json", manifest_path=tmp_path / "manifest.json", ref_dir=tmp_path)
    assert result["total_pools"] == 2
    assert [row["seed"] for row in result["per_pool"]] == [12, 11]
    assert result["fraction_top1_unchanged"] == .5
    assert result["spearman_rho_min"] == pytest.approx(-1.)
