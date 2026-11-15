"""Location of repo files and of the data that lives only in the main checkout (2026-11-08).

Tests must not depend on the working directory or on data inside a git worktree (a worktree has no data/ or
research/archive/). Data resolves like dev/tools/jdev/golden.py: $JACOBIK_DATA_ROOT, else <main checkout>/data.
Absent data is a skip with a reason, never a failure.
"""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


from tools.repo import main_checkout

MAIN = main_checkout(REPO)
DATA_ROOT = Path(os.environ["JACOBIK_DATA_ROOT"]) if os.environ.get("JACOBIK_DATA_ROOT") else MAIN / "data"
ARCHIVE = MAIN / "research" / "archive" / "data"


def require(path: Path, what: str) -> Path:
    """Return `path` or skip the calling test with a clear reason."""
    if not Path(path).exists():
        pytest.skip(f"{what} not present: {path}")
    return Path(path)
