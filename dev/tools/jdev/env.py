"""Paths, source-tree fingerprint, shared loaders. Every jdev result embeds fingerprint()."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

_this_repo = Path(__file__).resolve().parents[3]
from tools.repo import main_checkout

REPO = main_checkout(_this_repo)

#: the checkout this code runs from (REPO is the main checkout: data, runs)
CODE_REPO = _this_repo
SRC = _this_repo / "src"
FIX = _this_repo / "dev" / "tests" / "regression" / "fixtures"
RUNS = REPO / "runs"
DEFAULT_DB = REPO / "data" / "db" / "constraints_2026-11-12.npz"  # production DB since the DS data fold-in (2026-11-12)
DEFAULT_MACE = REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model" if (REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model").exists() else REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model"


def use_src() -> None:
    """Put src first on sys.path and refuse to run against any other jacobik tree."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import jacobik

    if not Path(jacobik.__file__).resolve().is_relative_to(SRC):
        raise RuntimeError(f"jacobik imported from {jacobik.__file__}, expected {SRC}")
    import logging

    logging.getLogger("jacobik").setLevel(logging.ERROR)


def _md5(path: Path, chunk: int = 1 << 20) -> str | None:
    if not path or not Path(path).exists():
        return None
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def tree_hash(root: Path = SRC) -> str:
    """Hash of every .py file under root (path + content, line endings normalised so a Windows and a
    Linux checkout of the same code hash identically)."""
    h = hashlib.md5()
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def db_path(arg: str | None = None) -> Path:
    return Path(arg or os.environ.get("JACOBIK_DB") or DEFAULT_DB)


def mace_path(arg: str | None = None) -> Path | None:
    p = arg or os.environ.get("JACOBIK_MACE_MODEL_PATH") or (DEFAULT_MACE if DEFAULT_MACE.exists() else None)
    return Path(p) if p else None


def fingerprint(db: Path | None = None, mace: Path | None = None) -> dict:
    db = db or db_path()
    meta = Path(str(db).replace(".json", ".meta.json"))
    return {
        "src": tree_hash(),
        "db": db.name, "db_md5": _md5(db),
        "db_meta": json.loads(meta.read_text()) if meta.exists() else None,
        "mace_md5": _md5(mace) if mace else None,
        "machine": platform.node(), "python": sys.version.split()[0],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_stats(db: Path, q1_variant: str | None = None, orientation_variant: str | None = None):
    """StatsLookup for db; variants default to the DB sidecar (then the code default)."""
    use_src()
    from jacobik.stats.stats import StatsLookup, load_db

    return StatsLookup(load_db(db), q1_variant=q1_variant, orientation_variant=orientation_variant)


def mace_evaluator(mace: Path | None):
    if mace is None:
        return None
    use_src()
    from jacobik.core.energy import MACEPolarEvaluator

    return MACEPolarEvaluator(model_path=str(mace))


def save_run(kind: str, name: str, payload: dict, xyz: tuple | None = None) -> Path:
    """runs/<date>/<kind>_<name>_<time>_<srchash8>.json (+ .xyz of the structure if given)."""
    day = time.strftime("%Y-%m-%d")
    stem = f"{kind}_{name}_{time.strftime('%H%M%S')}_{payload['fingerprint']['src'][:8]}"
    out = RUNS / day
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=1, default=float))
    if xyz is not None:
        sy, X, comment = xyz
        with open(out / f"{stem}.xyz", "w") as fh:
            fh.write(f"{len(sy)}\n{comment}\n")
            for s, p in zip(sy, X):
                fh.write(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return out / f"{stem}.json"
