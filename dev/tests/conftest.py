"""Shared setup for tests: always test the src tree (never the installed `src` package).

DB used by the tests: $JACOBIK_DB, else data/db/constraints_2026-11-12.npz (the deployed DB since the DS data fold-in),
resolved in this tree, else under the data root of tests/_paths.py ($JACOBIK_DATA_ROOT or the main checkout's data/;
a git worktree has no data/ of its own).
"""
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def restore_jacobik_log_level():
    """Benchmark case loaders/workers must not silence later validity tests."""
    import logging
    logger = logging.getLogger("jacobik")
    previous = logger.level
    yield
    logger.setLevel(previous)

REPO = Path(__file__).resolve().parents[2]
SRC_NEXT = REPO / "src"
BENCH_ROOT = REPO / "research" / "bench"
for p in reversed((SRC_NEXT, REPO / "dev", REPO, BENCH_ROOT)):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# subprocesses started by the tests resolve the same trees (absolute paths; the ini ``pythonpath`` is for this process only)
_env_paths = [str(p) for p in (SRC_NEXT, REPO / "dev", REPO, BENCH_ROOT)]
os.environ["PYTHONPATH"] = os.pathsep.join(_env_paths + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x])

# Guard against silently testing the wrong tree (this is how the 2026-09-23 dedup regression slipped
# through: tests/ik imported the pip-installed `src` package, never src).
import jacobik  # noqa: E402

assert Path(jacobik.__file__).resolve().is_relative_to(SRC_NEXT), (
    f"tests imported jacobik from {jacobik.__file__}, not {SRC_NEXT}"
)

_DB_NAME = "constraints_2026-11-12.npz"
_default_db = REPO / "data" / "db" / _DB_NAME
if not _default_db.exists():
    from tests._paths import DATA_ROOT  # noqa: E402

    _default_db = DATA_ROOT / "db" / _DB_NAME
DB_PATH = Path(os.environ.get("JACOBIK_DB", _default_db))


@pytest.fixture(scope="session")
def db_path():
    if not DB_PATH.exists():
        pytest.skip(f"constraints DB not found: {DB_PATH}")
    return DB_PATH


@pytest.fixture(scope="session")
def stats(db_path):
    from jacobik.stats.stats import StatsLookup, load_db

    return StatsLookup(load_db(db_path))


#: The flags that became production defaults in the DS data fold-in (2026-11-12), with the module attributes that hold them.
_FOLDED_IN = {
    "jacobik.topology.donor_classifier": ("HAPTO_BOND_GATE", "HAPTO_LABEL_CANON", "HAPTO_CHELATE_EXCLUSION", "DONOR_ASSIGNMENT"),
    "jacobik.topology.mapping": ("DONOR_ASSIGNMENT",),
    "fit.extractor": ("DONOR_ASSIGNMENT",),
    "jacobik.stats.keys": ("HAPTO_LABEL_CANON",),
    "fit.sanity": ("OS_SANITY", "FIT_EXCLUDE_LOCKED", "STRUCTURE_SANITY", "STABLE_ANCESTOR_SAMPLING"),
    "jacobik.topology.graph": ("Q6_PHYSICAL_ALL",),
    "jacobik.stats.steric": ("Q6_PHYSICAL_ALL",),
    "jacobik.ik.geometry": ("Q6_PHYSICAL_ALL",),
}


@pytest.fixture
def legacy_flags(monkeypatch):
    """Run a test with the flags folded in on 2026-11-12 switched off in the already-imported modules (the previous production
    behaviour). For tests of a legacy path; the module-level flags are the only state, so monkeypatch restores them."""
    import importlib

    for module, attrs in _FOLDED_IN.items():
        mod = importlib.import_module(module)
        for attr in attrs:
            monkeypatch.setattr(mod, attr, False)
