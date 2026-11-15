"""H-R: coverage uses every serialized row and preserves builder atom order."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tools.jgen import inputs
from tools.jgen.bench.pool_export import export_pool, coverage_export, _read_frames


@pytest.mark.parametrize("code", ["DAMVAQ", "JEGPUJ"])
@pytest.mark.parametrize("compression", ["none", "xz"])
def test_full_pool_coverage_contains_top_and_floor(tmp_path, code, compression):
    source = inputs.FRAME_V1 / "pools" / f"{code}_11_control.npz"
    if not source.exists():
        pytest.skip("read-only calibration fixture unavailable")
    with np.load(source, allow_pickle=True) as pool:
        ref_X = pool["ref_X"]
        ref_symbols = pool["ref_symbols"].tolist()
    # Reverse builder order deliberately; translation must not count as a miss.
    mapping = np.arange(len(ref_X))[::-1]
    geometries = [ref_X * 4, ref_X + 3, ref_X * 5]
    scores = [-0.7, -1.2, -2.1]
    confs = [NS(complex=NS(atoms=[NS(symbol=ref_symbols[j], position=X[j]) for j in mapping]),
                score=score, embed_idx=None, assignment_idx=2, trial_idx=i)
             for i, (X, score) in enumerate(zip(geometries, scores))]
    manifest_path = export_pool(tmp_path, code, "test", 11, confs,
                                [-100., float("nan"), -120.], ref_symbols, ref_X,
                                mapping, top=0, floor=2, compression=compression)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_rows"] == 3
    result = coverage_export(manifest_path)
    assert result["top1_or_floor_within_1A"] <= result["full_pool_within_1A"]
    assert not result["top1_within_1A"] and not result["floor_within_1A"]
    assert result["full_pool_within_1A"]  # only the middle row hits
    frames = []
    for filename in dict.fromkeys(row["file"] for row in manifest["rows"]):
        frames.extend(_read_frames(manifest_path.parent / filename))
    assert len(frames) == 3
    for i, (header, symbols, X) in enumerate(frames):
        assert header["score"] == confs[i].score
        assert header["row_key"] == [None, 2, i]
        assert header["is_top1"] == (i == 0)
        assert header["is_floor"] == (i == 2)
        assert symbols == [ref_symbols[j] for j in mapping]
        np.testing.assert_allclose(X, geometries[i][mapping], atol=5.1e-7, rtol=0)
    assert frames[1][0]["mace_energy_kj"] is None
    # plain: 3 rows + top/floor + manifest. xz: 1 bundle + top/floor + manifest.
    assert len(list(manifest_path.parent.iterdir())) == (6 if compression == "none" else 4)
