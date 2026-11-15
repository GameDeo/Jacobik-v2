import json

import numpy as np
import pytest

from tools.runkit import cache, policy, stats


def test_policy_refuses_val_and_reserve(disk_tmp):
    p = policy.DataPolicy()
    assert p.val and p.reserve
    v = sorted(p.val)[0]
    r = sorted(p.reserve)[0]
    with pytest.raises(policy.PolicyError):
        p.assert_allowed([v])
    with pytest.raises(policy.PolicyError):
        p.assert_allowed(["QEZVEY", r])
    rec = p.assert_allowed(["QEZVEY"])
    assert rec["sha256"] == p.sha256 and rec["parents"] == ["QEZVEY"]


def test_policy_panels():
    p = policy.DataPolicy()
    p.assert_panel("tail", ["SOBHAS", "HOXJUZ"])               # tail is the documented exception
    with pytest.raises(policy.PolicyError):
        p.assert_panel("tail", ["QEZVEY"])                       # not in tail
    with pytest.raises(policy.PolicyError):
        p.assert_panel("cal", ["SOBHAS"])                        # val on cal
    with pytest.raises(policy.PolicyError):
        p.assert_panel("val", ["SOBHAS"])
    with pytest.raises(policy.PolicyError):
        p.assert_panel("cal", [sorted(p.reserve)[0]])
    p.assert_panel("cal", ["QEZVEY"])


def test_policy_refuses_empty_partition(disk_tmp):
    f = disk_tmp / "ex.json"
    f.write_text(json.dumps(dict(reserve_all_codes=["AAAAAA"], motion_validation_codes=[])))
    with pytest.raises(policy.PolicyError):
        policy.DataPolicy(f)
    with pytest.raises(policy.PolicyError):
        policy.DataPolicy(disk_tmp / "missing.json")


def test_tail_constant_matches_panels():
    from tools.jgen.bench import panels
    assert tuple(panels.TAIL) == policy.TAIL_PANEL


def test_cache_roundtrip_and_corruption(disk_tmp):
    s = cache.Store(disk_tmp / "c")
    X = np.random.default_rng(0).normal(size=(3, 3))
    k = cache.key("e", "1", ["H", "H", "O"], X, 0, 1, {"a": 1})
    assert s.get(k) is None
    s.put(k, dict(E=-1.5, X=X.tolist()))
    assert np.array_equal(np.array(s.get(k)["X"]), X)       # exact float round trip
    assert cache.key("e", "1", ["H", "H", "O"], X + 1e-12, 0, 1, {"a": 1}) != k
    p = s._p(k)
    p.write_text(p.read_text().replace("-1.5", "-1.6"))
    assert s.get(k) is None and s.verify() == [k]


def test_bootstrap_matches_bench_metrics():
    from tools.jgen.bench import metrics
    v = np.random.default_rng(1).normal(size=17)
    assert stats.bootstrap_mean(v, n_boot=500, seed=7) == metrics.bootstrap_mean(v, n_boot=500, seed=7)
    r = stats.parent_bootstrap({"A": [1, 3], "B": 4.0, "C": [2]}, n_boot=200)
    assert r["n_parents"] == 3 and abs(r["mean"] - (2 + 4 + 2) / 3) < 1e-12
