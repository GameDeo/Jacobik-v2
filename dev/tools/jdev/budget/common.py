"""Shared loaders of the budget tools: where the pools live, which files a pool arm holds, its MACE energies, the stats.

Used by pools.py, tracks.py, gate.py, q6_diag.py, nci_census.py (import with `from common import ...` after putting this
folder on sys.path, as the tools do). Pools are <OUT>/pools/<arm>/<set>/<code>_s<seed>.npz (+ .mace.json / .mace64.json
/ .xtb.json next to them); OUT is JACOBIK_BUDGET_OUT.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

_this_repo = Path(__file__).resolve().parents[4]
from tools.repo import main_checkout

REPO = main_checkout(_this_repo)


def budget_out(default: str = "energy_scale_refit") -> Path:
    """The pools folder: $JACOBIK_BUDGET_OUT, else research/archive/data/<default> or data/<default>."""
    arch_path = REPO / "research/archive/data" / default
    default_path = arch_path if arch_path.exists() else REPO / "data" / default
    return Path(os.environ.get("JACOBIK_BUDGET_OUT", default_path))


OUT = budget_out()


def _resolve_db(path_str: Optional[str]) -> Path:
    if path_str:
        p = Path(path_str)
        if p.exists():
            return p.resolve()
        for cand in [
            REPO / "research/archive/data/refit/final" / p.name,
            REPO / "data/db" / p.name,
            REPO / "data/refit/final" / p.name,
        ]:
            if cand.exists():
                return cand.resolve()
        return p.resolve()
    for cand in [
        REPO / "data/db/constraints_2026-10-18.json",
        REPO / "data/refit/final/constraints_2026-10-18.json",
    ]:
        if cand.exists():
            return cand.resolve()
    return (REPO / "data/db/constraints_2026-10-18.json").resolve()


#: Current deployed DB. Historical pools must name their original DB explicitly.
DB = _resolve_db(os.environ.get("JACOBIK_BUDGET_DB"))


def pool_path(arm: str, set_name: str, code: str, seed: int, out: Path = OUT) -> Path:
    return out / "pools" / arm / set_name / f"{code}_s{seed}.npz"


def pool_files(arm: str, sets, out: Path = OUT, seeds=None) -> Iterator[tuple]:
    """(set, code, seed, path) of every pool of `arm` in the sets (sorted by file name), optionally only these seeds."""
    for s in sets:
        for f in sorted((out / "pools" / arm / s).glob("*.npz")):
            code, seed = f.stem.rsplit("_s", 1)
            if seeds and int(seed) not in seeds:
                continue
            yield s, code, int(seed), f


def read_pool(f: Path):
    """The npz of a pool, or None when the solve failed (the file then holds only `error`)."""
    z = np.load(f, allow_pickle=False)
    return None if "error" in z.files else z


def pool_energy(f: Path, energy: str = "mace") -> Optional[np.ndarray]:
    """MACE dE (kJ/mol, per conformer, against the reference) of a pool: energy "mace" (float32) or "mace64" (float64,
    pools.py mace --dtype float64); None when not computed."""
    mf = f.with_suffix(f".{energy}.json")
    return np.array(json.loads(mf.read_text())["dE"]) if mf.exists() else None


def stats_from_db(db: Path):
    """StatsLookup of a constraints DB with its .meta.json sidecar (jacobik must be importable: call setup() first)."""
    from jacobik.stats.stats import StatsLookup, load_db
    db = Path(db)
    meta = db.with_name(db.stem + ".meta.json")
    # Keep the path intact so load_db can discover optional sibling sidecars such as
    # ``<db stem>.bends.json``. Passing a parsed dict silently disabled those readers.
    return StatsLookup(load_db(db, meta=json.loads(meta.read_text()) if meta.exists() else None))
