"""Fixtures. pytest's tmp_path is under /tmp (RAM-backed here), so tests that need disk use RUN_DIR/_tests."""
import shutil
import uuid
from pathlib import Path

import pytest

from tools.runkit import resources


@pytest.fixture
def disk_tmp():
    d = resources.RUN_DIR / "_tests" / uuid.uuid4().hex[:8]
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
