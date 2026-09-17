"""Verified acquisition and layout validation for the public tmQM release."""

from __future__ import annotations

import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/uiocompcat/tmqm/master/tmQM"


def required_files(splits: tuple[str, ...] = ("X1", "X2", "X3")) -> list[str]:
    files = ["tmQM_y.csv"]
    for split in splits:
        files.extend((f"xyz/tmQM_{split}.xyz", f"bo/tmQM_{split}.BO"))
    return files


def validate_layout(root: str | Path, splits: tuple[str, ...] = ("X1",)) -> None:
    root = Path(root)
    missing = [name for name in required_files(splits) if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("incomplete tmQM dataset; missing: " + ", ".join(missing))
    if (root / "tmQM_y.csv").stat().st_size == 0:
        raise ValueError("tmQM_y.csv is empty")


def _download(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "jacobik-benchmark/1"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        while block := response.read(1024 * 1024):
            digest.update(block)
            out.write(block)
    tmp.replace(destination)
    return digest.hexdigest()


def fetch_tmqm(root: str | Path, splits: tuple[str, ...] = ("X1",), *, force: bool = False) -> dict[str, str]:
    """Download selected tmQM splits and atomically decompress XYZ files.

    Existing files are preserved unless ``force=True``. SHA-256 values of the
    downloaded wire files are returned for provenance logging.
    """
    root = Path(root)
    (root / "xyz").mkdir(parents=True, exist_ok=True)
    (root / "bo").mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    csv_path = root / "tmQM_y.csv"
    if force or not csv_path.exists():
        hashes["tmQM_y.csv"] = _download(f"{BASE_URL}/tmQM_y.csv", csv_path)
    for split in splits:
        if split not in {"X1", "X2", "X3"}:
            raise ValueError(f"unknown tmQM split: {split}")
        for kind, suffix in (("xyz", "xyz"), ("bo", "BO")):
            dest = root / kind / f"tmQM_{split}.{suffix}"
            if dest.exists() and not force:
                continue
            # GitHub stores XYZ compressed. BO files may be served either way,
            # so try the documented gzip artifact before the plain file.
            gz = dest.with_suffix(dest.suffix + ".gz")
            try:
                hashes[gz.name] = _download(f"{BASE_URL}/tmQM_{split}.{suffix}.gz", gz)
                with gzip.open(gz, "rb") as source, dest.with_suffix(dest.suffix + ".part").open("wb") as out:
                    shutil.copyfileobj(source, out)
                dest.with_suffix(dest.suffix + ".part").replace(dest)
                gz.unlink()
            except Exception:
                if gz.exists():
                    gz.unlink()
                hashes[dest.name] = _download(f"{BASE_URL}/tmQM_{split}.{suffix}", dest)
    validate_layout(root, splits)
    return hashes
