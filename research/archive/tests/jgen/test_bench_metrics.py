"""Bench metrics and panels (roadmap_generation_v1.md 1.4): synthetic units, no data needed except where marked."""
import json

import numpy as np
import pytest

from tests.jgen._data import need_data
from tools.jgen.bench import metrics, panels


def _rec(code, seed, top1, floor, status="ok", reason=""):
    return dict(code=code, seed=seed, status=status, reason=reason, top1_dE=top1, floor_dE=floor)


def test_identical_arms_give_zero_change():
    base = {("A", 11): _rec("A", 11, 10.0, 5.0), ("A", 12): _rec("A", 12, 20.0, 7.0)}
    s = metrics.summarize(dict(base), base, list(base))
    assert s["e"]["mean"] == 0.0 and s["f"]["mean"] == 0.0 and s["r"]["mean"] == 0.0
    assert s["parent_band_exceed_count"] == 0 and s["units_gt10"] == []


def test_parent_change_is_mean_over_seeds_and_regret_is_e_minus_f():
    base = {("A", 11): _rec("A", 11, 100.0, 50.0), ("A", 12): _rec("A", 12, 80.0, 40.0)}
    cand = {("A", 11): _rec("A", 11, 90.0, 60.0), ("A", 12): _rec("A", 12, 80.0, 40.0)}
    pc = metrics.parent_changes(cand, base, [("A", 11), ("A", 12)])
    assert pc["A"]["e"] == pytest.approx(-5.0)
    assert pc["A"]["f"] == pytest.approx(5.0)
    # regret change: (90-60)-(100-50) = -20 ; (80-40)-(80-40) = 0 ; mean -10
    assert pc["A"]["r"] == pytest.approx(-10.0)


def test_missing_and_invalid_units_stay_in_the_ledger_and_denominator():
    base = {("A", 11): _rec("A", 11, 1.0, 0.0), ("A", 12): _rec("A", 12, 1.0, 0.0), ("B", 11): _rec("B", 11, 2.0, 1.0)}
    cand = {("A", 11): _rec("A", 11, None, None, status="failed", reason="empty"),
            ("B", 11): _rec("B", 11, 2.0, 1.0)}
    keys = sorted(base)
    s = metrics.summarize(cand, base, keys)
    assert s["units_panel"] == 3 and s["units_paired"] == 1
    reasons = {(e["code"], e["seed"]): e["reason"] for e in s["ledger"]}
    assert reasons[("A", 11)].startswith("candidate_invalid")
    assert reasons[("A", 12)] == "missing_candidate"
    assert s["valid_top1_share"] == pytest.approx(1 / 3)


def test_band_and_unit_threshold():
    base = {("A", 11): _rec("A", 11, 0.0, 0.0), ("B", 11): _rec("B", 11, 0.0, 0.0)}
    cand = {("A", 11): _rec("A", 11, 1.0, 0.0), ("B", 11): _rec("B", 11, 12.0, 0.0)}
    s = metrics.summarize(cand, base, sorted(base))
    assert s["parent_band_exceed"] == ["A", "B"]
    assert [u[0] for u in s["units_gt10"]] == [["B", 11]]


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    v = np.array([1.0, -2.0, 3.0, 0.5, -0.5])
    a = metrics.bootstrap_mean(v)
    b = metrics.bootstrap_mean(v)
    assert a == b
    m, lo, hi = a
    assert lo <= m <= hi


@need_data
def test_tail_panel_has_ten_parents_and_all_panel_is_cal_plus_val():
    from tools.jgen.bench.panels import select
    from tools.jgen.inputs import SCALED_V1
    man = json.load(open(SCALED_V1 / "manifest.json"))
    tail = select(man, "tail")
    assert len({t["code"] for t in tail["units"]}) + len(tail["not_in_cohort"]) == 10
    allp = select(man, "all")
    cal = select(man, "cal")
    val = select(man, "val")
    assert len(allp["units"]) == len(cal["units"]) + len(val["units"])
    assert "not_in_cohort" in tail and "not_in_cohort" in select(man, "failures")


def test_unknown_panel_raises():
    with pytest.raises(ValueError):
        panels.select({"tasks": {}}, "reserve")
