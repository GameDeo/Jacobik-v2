"""bench simple with a control-run reference (--control-dir): paired baseline from records, not stored pools."""
import pytest

from tests.jgen._data import need_data


@need_data
def test_control_dir_pairs_by_records():
    from tools.jgen import inputs
    from tools.jgen.bench import panels, simple

    manifest_path = str(inputs.SCALED_V1 / "manifest.json")
    import json
    manifest = json.load(open(manifest_path))
    keys = simple.unit_keys(panels.select(manifest, "tail")["units"])
    control = {k: dict(code=k[0], seed=k[1], status="ok", reason="", top1_dE=10.0, floor_dE=5.0) for k in keys}
    cand = {k: dict(v) for k, v in control.items()}
    bumped = keys[0][0]
    for k in keys:
        if k[0] == bumped:
            cand[k]["top1_dE"] = 13.0  # +3 kJ/mol on every seed of one parent
    s = simple.run("tail", cand, None, manifest_path, "", control=control)
    assert s["parent_changes"][bumped]["e"] == pytest.approx(3.0)
    assert bumped in s["parent_band_exceed"]
    assert s["parent_band_exceed_count"] == 1
    assert s["status"] == "descriptive"
