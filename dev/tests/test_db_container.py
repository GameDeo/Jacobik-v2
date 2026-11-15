"""Unit tests for the lossless binary NPZ constraint database container and ladder pooling."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from fit.pooling import pool_ladder
from jacobik.core.container import (
    decode_lossless,
    encode_lossless,
    load_lossless_npz,
    save_lossless_npz,
)
from jacobik.core.env import default_db_path
from jacobik.stats.stats import load_db


def test_lossless_container_production_db_roundtrip_and_legacy_nci(tmp_path):
    """Verify bit-identical NPZ roundtrip across all quantity families in production DB, plus legacy NCI ignore."""
    prod_db = default_db_path()
    assert prod_db is not None and prod_db.is_file()
    npz_size = prod_db.stat().st_size
    assert 15 * 1024 * 1024 <= npz_size <= 40 * 1024 * 1024, f"Unexpected NPZ size: {npz_size}"

    full_db, full_meta = load_lossless_npz(prod_db)
    assert len(full_db) > 100_000
    assert full_meta.get("q1_variant") and full_meta.get("q5_model")

    # Deterministically sample up to 300 keys per quantity prefix for fast exact roundtrip checks
    by_prefix = defaultdict(list)
    for k in sorted(full_db):
        prefix = k.split(":", 1)[0]
        if len(by_prefix[prefix]) < 300:
            by_prefix[prefix].append(k)
    sample_db = {k: full_db[k] for ks in by_prefix.values() for k in ks}
    assert {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO"}.issubset(by_prefix)

    # Also test encode_lossless / decode_lossless directly and legacy NCI entry handling
    sample_plus_nci = dict(sample_db)
    sample_plus_nci["NCI:('hb', 'O', 'O')"] = {"type": "nci", "n": 120, "r_lo": 1.5, "dr": 0.1, "s": [[0.1]]}

    encoded = encode_lossless(sample_plus_nci, header_meta=full_meta)
    dec_db = decode_lossless(encoded)
    assert set(dec_db) == set(sample_plus_nci)

    tmp_npz = tmp_path / "sample_roundtrip.npz"
    save_lossless_npz(sample_plus_nci, tmp_npz, header_meta=full_meta)
    reloaded_db, reloaded_meta = load_lossless_npz(tmp_npz)
    assert set(reloaded_db) == set(sample_plus_nci)
    assert reloaded_meta.get("q1_variant") == full_meta.get("q1_variant")

    for k, orig_entry in sample_plus_nci.items():
        rel_entry = reloaded_db[k]
        assert orig_entry.keys() == rel_entry.keys(), f"Field mismatch at {k}"
        for field, val in orig_entry.items():
            rel_val = rel_entry[field]
            if isinstance(val, float):
                assert abs(val - rel_val) < 1e-15, f"Float diff at {k}.{field}"
            elif isinstance(val, list):
                try:
                    val_arr = np.array(val, dtype=np.float64)
                    rel_arr = np.array(rel_val, dtype=np.float64)
                    diff = float(np.max(np.abs(val_arr - rel_arr))) if val_arr.size > 0 else 0.0
                    assert diff < 1e-15, f"Array diff at {k}.{field}: {diff}"
                except (ValueError, TypeError):
                    assert val == rel_val, f"Mismatch at {k}.{field}"
            else:
                assert val == rel_val, f"Mismatch at {k}.{field}"

    cdb = load_db(tmp_npz)
    assert len(cdb.data["Q1"]) > 0
    assert len(cdb.data["Q2"]) > 0
    assert "NCI" not in cdb.priors


def test_pool_ladder_propagation():
    """Verify pool_ladder empirical Bayes propagation along a ladder."""
    root_key, mid_key, leaf_key = "root", "mid", "leaf"
    estimates = {
        root_key: {
            "fit": {"mu": [2.0], "var": [0.01], "pi": [1.0]},
            "x": np.array([2.0, 2.01, 1.99]),
            "w": np.array([1.0, 1.0, 1.0]),
            "src": "fit",
        },
        mid_key: {"x": np.array([2.1]), "w": np.array([1.0])},
        leaf_key: {"x": np.array([2.2, 2.22]), "w": np.array([1.0, 1.0])},
    }
    pooled = pool_ladder(estimates, [root_key, mid_key, leaf_key], r=4.0)
    assert pooled[root_key]["src"] == "fit"
    assert pooled[mid_key]["src"] == "adapted"
    assert pooled[leaf_key]["src"] == "adapted"
    assert 2.0 < pooled[mid_key]["mu"][0] < 2.1
    assert pooled[mid_key]["mu"][0] < pooled[leaf_key]["mu"][0]
