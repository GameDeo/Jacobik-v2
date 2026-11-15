import json
import time

from tools.runkit import runprofile as rp


def test_projection_arithmetic():
    p = rp.project(n_items=90, per_item_s=20, workers=3, speed=1.0, overlap=0.0, units_per_h=1.5)
    assert p.wall_s == 600 and abs(p.units - 0.25) < 1e-12
    q = rp.project(100, 10, workers=4, speed=0.5, overlap=0.25)
    assert abs(q.wall_s - 100 * 10 / 2 * 0.75) < 1e-9


def test_require_budget_refusal_record(disk_tmp):
    ok = rp.require_budget(rp.project(10, 6), max_min=15, record=disk_tmp / "r.jsonl")
    assert ok["ok"] and not (disk_tmp / "r.jsonl").exists()
    bad = rp.require_budget(rp.project(100, 60), max_min=15, record=disk_tmp / "r.jsonl")
    assert not bad["ok"] and "defer" in bad["refusal"]
    assert json.loads((disk_tmp / "r.jsonl").read_text())["projected_min"] == 100.0


def test_trace_and_profile_json(disk_tmp):
    tr = rp.Trace()
    with tr("a"):
        time.sleep(0.05)
    with tr("a"):
        pass
    tr.add("xtb", 2.0, 8.0)
    assert tr.stages["a"]["n"] == 2 and tr.stages["a"]["wall_s"] >= 0.05 and tr.stages["xtb"]["cpu_s"] == 8.0
    d = json.loads(tr.write(disk_tmp).read_text())
    assert "stages" in d and d["total_wall_s"] > 2.0
    assert rp.measure_items(lambda x: time.sleep(0.01), range(5), n=2) >= 0.01
