"""writer.py
=========
Deterministic constraint database writer.

Writes both lossless binary (.npz) and JSON formats alongside sidecar metadata (.meta.json).
Records comprehensive provenance:
- SHA-256 hashes of all reference input files in data/ref/
- Git commit hash (git rev-parse HEAD)
- Random seed
- ISO 8601 UTC timestamp
- Fit parameters and ladder specifications
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Union

from jacobik.db.container import save_lossless_npz

log = logging.getLogger(__name__)


def compute_input_hashes(ref_dir: Union[str, Path]) -> Dict[str, str]:
    """Compute SHA-256 hashes of all regular files in reference directory."""
    ref_path = Path(ref_dir)
    if not ref_path.exists():
        log.warning("Reference directory does not exist for hashing: %s", ref_path)
        return {}

    hashes: Dict[str, str] = {}
    for p in sorted(ref_path.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                while chunk := fh.read(65536):
                    h.update(chunk)
            rel_path = str(p.relative_to(ref_path)).replace("\\", "/")
            hashes[rel_path] = h.hexdigest()
    return hashes


def get_git_commit(cwd: Optional[Union[str, Path]] = None) -> str:
    """Retrieve current git commit hash (HEAD)."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception as exc:
        log.warning("Unable to determine git commit: %s", exc)
        return "unknown"


def build_provenance_meta(
    ref_dir: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build provenance metadata dictionary."""
    meta: Dict[str, Any] = {}
    if extra_meta:
        meta.update(extra_meta)

    meta["timestamp"] = datetime.now(timezone.utc).isoformat()
    meta["git_commit"] = get_git_commit()
    if seed is not None:
        meta["seed"] = seed
    if params is not None:
        meta["parameters"] = dict(params)

    if ref_dir is not None:
        meta["input_hashes"] = compute_input_hashes(ref_dir)

    return meta


class DBWriter:
    """Deterministic writer for constraint databases with provenance tracking."""

    def __init__(
        self,
        out_dir: Union[str, Path],
        ref_dir: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.ref_dir = Path(ref_dir) if ref_dir is not None else None
        self.seed = seed
        self.params = params or {}

    def write(
        self,
        db_dict: Dict[str, Any],
        stem: str = "constraints",
        meta: Optional[Dict[str, Any]] = None,
        write_json: bool = True,
        write_npz: bool = True,
    ) -> Dict[str, Path]:
        """Write constraints database and metadata sidecar."""
        return write_db(
            db_dict=db_dict,
            out_dir=self.out_dir,
            stem=stem,
            meta=meta,
            seed=self.seed,
            params=self.params,
            ref_dir=self.ref_dir,
            write_json=write_json,
            write_npz=write_npz,
        )


def write_db(
    db_dict: Dict[str, Any],
    out_dir: Union[str, Path],
    stem: str = "constraints",
    meta: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    ref_dir: Optional[Union[str, Path]] = None,
    write_json: bool = True,
    write_npz: bool = True,
) -> Dict[str, Path]:
    """Write constraint database and metadata sidecar deterministically.

    Parameters
    ----------
    db_dict : Dict[str, Any]
        Constraint database dictionary mapping quantity keys to distributions.
    out_dir : Union[str, Path]
        Target directory to write files.
    stem : str
        Base filename (default "constraints").
    meta : Optional[Dict[str, Any]]
        Fit metadata (variants, models, etc.).
    seed : Optional[int]
        Random seed used in fitting.
    params : Optional[Dict[str, Any]]
        Fitting parameters.
    ref_dir : Optional[Union[str, Path]]
        Reference data directory to compute provenance file hashes from.
    write_json : bool
        Whether to write <stem>.json.
    write_npz : bool
        Whether to write <stem>.npz.

    Returns
    -------
    Dict[str, Path]
        Mapping of output format ('npz', 'json', 'meta') to written file Paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    full_meta = build_provenance_meta(
        ref_dir=ref_dir,
        seed=seed,
        params=params,
        extra_meta=meta,
    )

    written: Dict[str, Path] = {}

    # Write metadata sidecar (.meta.json)
    meta_path = out_path / f"{stem}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(full_meta, fh, indent=2, sort_keys=True)
    written["meta"] = meta_path

    # Write lossless compressed binary (.npz)
    if write_npz:
        npz_path = out_path / f"{stem}.npz"
        save_lossless_npz(db_dict, npz_path, header_meta=full_meta)
        written["npz"] = npz_path

    # Write deterministic JSON (.json)
    if write_json:
        json_path = out_path / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(db_dict, fh, sort_keys=True, separators=(",", ":"))
        written["json"] = json_path

    log.info(
        "Successfully wrote constraint DB to %s (formats: %s)",
        out_path,
        ", ".join(written.keys()),
    )
    return written
