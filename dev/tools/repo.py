"""The main checkout of this repository (stdlib only, so the runners and the check levels do not import torch).

A git worktree has no ``data/`` or ``research/data/`` of its own; they live in the main checkout, the parent of
``git rev-parse --git-common-dir``. ``jacobik.core.env.main_checkout`` is the same function for the library (which cannot
import ``tools``)."""
from __future__ import annotations

import subprocess
from pathlib import Path


#: the pools' constraints DB (2026-10-17 json + its .meta.json sidecar) that the golden references were scored with:
#: a support file of the golden directory, named for that file and NOT for the production npz (the golden check
#: rescoring stored pools needs this exact DB; the production DB is data/db/constraints_2026-11-12.npz).
BUDGET_DB_SUPPORT = "research/data/golden/support/constraints_2026-10-17.json"


def main_checkout(repo: Path) -> Path:
    """The main checkout of the git repository ``repo`` belongs to; ``repo`` itself outside git."""
    repo = Path(repo)
    try:
        common = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                                check=True).stdout.strip()
        return (Path(common) if Path(common).is_absolute() else (repo / common)).resolve().parent
    except Exception:
        return repo
