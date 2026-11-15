"""Frozen selection of at most five diagnostic relaxations per unit."""
from __future__ import annotations
import json
from pathlib import Path

def select(manifest, record):
    rows = manifest["rows"]
    if int(manifest["n_rows"]) != int(record["n_rows"]):
        raise ValueError("pool/record row count mismatch")
    cov = record.get("coverage", {}).get("rmsd_A")
    if cov is None or len(cov) != len(rows):
        raise ValueError("record lacks per-row coverage RMSDs")
    top = next((i for i, r in enumerate(rows) if r.get("is_top1")), None)
    floor = next((i for i, r in enumerate(rows) if r.get("is_floor")), None)
    if top is None or floor is None: raise ValueError("pool lacks top1/floor row")
    near = sorted((float(d), i) for i, d in enumerate(cov) if d is not None and float(d) <= 1.5)[:3]
    return sorted({top, floor, *(i for _, i in near)})

def load(path): return json.loads(Path(path).read_text())
