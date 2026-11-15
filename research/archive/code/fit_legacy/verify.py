"""Extraction check: observations extracted from data/ref against an archived observation set, on the parents both hold.

``python -m jacobik.fit.pipeline.verify NEW_OBS_DIR ARCHIVE_OBS_DIR`` compares Q1-Q5 (row multisets of csd_code, key, value) and
QO (row multisets of the identifying and feature columns). Q6 is a per-split reservoir sample and is not row-comparable.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SPLITS = ("X1", "X2", "X3", "LNf")


def _read(d: Path, q: str) -> pd.DataFrame:
    parts = [pd.read_parquet(d / f"observations_{q}_{s}.parquet") for s in SPLITS if (d / f"observations_{q}_{s}.parquet").exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def compare_observations(new_dir: Path, ref_dir: Path, quantities=("Q1", "Q2", "Q3", "Q4", "Q5", "QO")) -> dict:
    out = {}
    for q in quantities:
        new, ref = _read(Path(new_dir), q), _read(Path(ref_dir), q)
        if new.empty or ref.empty:
            out[q] = {"skipped": "missing table"}
            continue
        codes = set(new["csd_code"].astype(str))
        ref = ref[ref["csd_code"].astype(str).isin(codes)]
        cols = ["csd_code", "key", "value"] if q != "QO" else ["csd_code", "k", "ring", "sub", "metal", "geom", "anion",
                                                                "hb_min", "g0", "g1", "g2", "g3", "g4", "g5"]
        cols = [c for c in cols if c in new.columns and c in ref.columns]
        a = new[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)
        b = ref[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)
        ca = Counter("|".join(map(repr, r)) for r in a.astype(object).where(a.notna(), None).itertuples(index=False))
        cb = Counter("|".join(map(repr, r)) for r in b.astype(object).where(b.notna(), None).itertuples(index=False))
        only_new = sum((ca - cb).values())
        only_ref = sum((cb - ca).values())
        # the same multisets with floats rounded to 1e-9 (extraction on another CPU/BLAS differs in the last bits)
        fl = [c for c in cols if a[c].dtype.kind == "f"]
        ra, rb = a.copy(), b.copy()
        ra[fl], rb[fl] = ra[fl].round(9), rb[fl].round(9)
        ra_c = Counter("|".join(map(repr, r)) for r in ra.astype(object).where(ra.notna(), None).itertuples(index=False))
        rb_c = Counter("|".join(map(repr, r)) for r in rb.astype(object).where(rb.notna(), None).itertuples(index=False))
        out[q] = {"parents": len(codes), "rows_new": len(a), "rows_ref": len(b), "rows_only_in_new": only_new,
                  "rows_only_in_ref": only_ref, "identical": only_new == 0 and only_ref == 0,
                  "rows_differing_beyond_1e-9": sum((ra_c - rb_c).values()),
                  "identical_to_1e-9": ra_c == rb_c}
    return out


if __name__ == "__main__":
    print(json.dumps(compare_observations(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
