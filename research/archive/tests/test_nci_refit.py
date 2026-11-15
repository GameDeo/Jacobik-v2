"""Scoped NCI table replacement must preserve every other entry and the input DB."""
import json

import numpy as np

from jacobik.stats import nci
from refit import nci_refit


def test_candidate_fit_preserves_input_and_non_nci_entries(tmp_path, monkeypatch):
    base = tmp_path / "base.json"
    original = {"Q1:example": {"mu": 2.123456789, "weights": [.25, .75]},
                "Q6P:example": {"core_b": 3.2}, "NCI:old": {"obsolete": True}}
    base.write_text(json.dumps(original))
    meta = {"q1_variant": "m_sub_early", "benchmark_holdout": ["TEST"]}
    base.with_suffix(".meta.json").write_text(json.dumps(meta))
    before = base.read_bytes()
    out = tmp_path / "candidate"
    out.mkdir()
    h = nci.bin_volumes("hb") * 1e5
    np.savez_compressed(out / "ncihist_TEST.npz", **{"h:hb|*|*": h})
    monkeypatch.setattr(nci_refit, "BASE", base)
    nci_refit.fit(out, [{"shard": "TEST"}])
    got = json.loads((out / "constraints_nci_2026-10-18.json").read_text())
    assert base.read_bytes() == before
    assert {k: v for k, v in got.items() if not k.startswith("NCI:")} == {
        k: v for k, v in original.items() if not k.startswith("NCI:")}
    assert "NCI:old" not in got and "NCI:('hb', '*', '*')" in got
    new_meta = json.loads((out / "constraints_nci_2026-10-18.meta.json").read_text())
    assert all(new_meta[k] == v for k, v in meta.items())
    assert new_meta["nci_regeneration"]["enabled_by_default"] is False
