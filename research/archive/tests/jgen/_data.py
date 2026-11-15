"""Skip markers for tests that need the frozen data root (JACOBIK_DATA_ROOT, default <repo>/data)."""
import pytest

from tools.jgen.inputs import CLEAN, DB_PATH, SCALED_V1

_have = DB_PATH.exists() and (SCALED_V1 / "frame.json").exists() and (CLEAN / "tmQM").exists()

need_data = pytest.mark.skipif(not _have, reason="frozen data root not available (set JACOBIK_DATA_ROOT)")
