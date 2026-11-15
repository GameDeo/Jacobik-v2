"""Read-only input loaders for jgen (research/archive/docs/roadmap/roadmap_generation_v1.md 1.1).

tmQM rule (binding): only the CSD_code, SMILES and geometry columns are read. The loaders here never parse the
electronic, dispersion, dipole, metal-charge, gap, orbital or polarizability columns, and they never call
``fit.parser.TMQMParser.csv_index`` or ``_attach_qm_properties`` (both keep those columns).

Data root: ``JACOBIK_DATA_ROOT`` (default ``<repo>/data``). A git worktree has no ``data/`` of its own, so the bench and
the tests point this at the main checkout's ``data/`` and only read from it.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from tools.repo import main_checkout

REPO = Path(__file__).resolve().parents[3]


def _default_data_root() -> Path:
    """<repo>/data if present, else the data/ of the main checkout (a git worktree has no data/ of its own). Read only."""
    local = REPO / "data"
    return local if local.exists() else main_checkout(REPO) / "data"


DATA = Path(os.environ.get("JACOBIK_DATA_ROOT", _default_data_root()))
CLEAN = DATA / "ref" if (DATA / "ref").exists() else DATA / "refit" / "clean"
#: the production constraints DB (DS data fold-in, 2026-11-12: constraints_2026-11-12.npz, sha256 62c58d53...; built by fit.build_db at the default profile.
#: Previous production: constraints_2026-11-01.json (eta2 fold-in = 2026-10-18 with the eta2 S_orient keys refit under the face plane; the frame_v1
#: control pools were generated with 2026-10-18, constraints_db_sha256 48ac82b5...))
DB_PATH = DATA / "db" / "constraints_2026-11-12.npz"
#: JACOBIK_DB: an experiment DB (e.g. a refit that goes with a flag); the record's constraints_db_sha256 shows which ran
if os.environ.get("JACOBIK_DB"):
    DB_PATH = Path(os.environ["JACOBIK_DB"]).resolve()
    # stats.load_db reads the runtime model choices (q5_model, q1_variant, ...) from the container's embedded _meta, or for
    # a legacy DB from <stem>.meta.json; a DB with neither silently runs other models than production on every parent
    # (2026-11-01: every eta2 experiment DB). A flag that needs a table (profile.FLAG_TABLES) must find it in the DB.
    from jacobik.core import container as dbio
    _info = dbio.runtime_info(DB_PATH)
    if _info["source"] == "none":
        raise SystemExit(f"JACOBIK_DB {DB_PATH} has no embedded _meta and no {DB_PATH.stem}.meta.json sidecar")
    _missing = dbio.missing_tables(_info["tables"])
    if _missing:
        raise SystemExit(f"JACOBIK_DB {DB_PATH} lacks tables needed by active flags: {_missing}")
FRAME_V1 = DATA / "refit" / "frame_v1"

from jacobik.core.env import _Row, load_mol, tmqm_smiles  # noqa: E402,F401  (moved into the package; kept importable here)


def frame_shards(root: Path = FRAME_V1) -> Dict[str, Tuple[str, str]]:
    """representative_code -> (shard, dataset) from the frozen frame."""
    f = json.load(open(root / "frame.json"))
    out = {}
    for gs in f["main_allocation"]["splits"].values():
        for g in gs:
            out[g["representative_code"]] = (g["representative_shard"], g["dataset"])
    return out


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
