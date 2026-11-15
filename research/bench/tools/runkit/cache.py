"""Content-addressed result store: key = sha256(engine id | version | options | symbols | coordinate bytes | charge | spin).

Writes are atomic (.tmp then os.replace); reads verify a stored checksum and treat a damaged entry as a miss.
`ledger` appends JSONL records (append is line-atomic for small lines)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def key(engine: str, version: str, symbols, X, charge: int, spin: int, options: dict | None = None) -> str:
    h = hashlib.sha256()
    h.update(f"{engine}|{version}|{json.dumps(options or {}, sort_keys=True)}|{','.join(map(str, symbols))}|{int(charge)}|{int(spin)}|".encode())
    h.update(np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes())
    return h.hexdigest()


def atomic_write(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Store:
    """Directory store of JSON results (arrays are stored as nested lists of float64 repr, which round-trips exactly)."""

    def __init__(self, root):
        self.root = Path(root)
        self.hits = self.misses = 0

    def _p(self, k: str) -> Path:
        return self.root / k[:2] / (k + ".json")

    def get(self, k: str):
        p = self._p(k)
        try:
            rec = json.loads(p.read_text())
            body = json.dumps(rec["value"], sort_keys=True)
            if hashlib.sha256(body.encode()).hexdigest() != rec["sha256"]:
                raise ValueError("checksum")
        except (OSError, ValueError, KeyError):
            self.misses += 1
            return None
        self.hits += 1
        return rec["value"]

    def put(self, k: str, value: dict):
        body = json.dumps(value, sort_keys=True)
        atomic_write(self._p(k), json.dumps(dict(sha256=hashlib.sha256(body.encode()).hexdigest(), value=value)).encode())

    def verify(self) -> list:
        """Keys of damaged entries."""
        bad = []
        for p in self.root.glob("*/*.json"):
            try:
                rec = json.loads(p.read_text())
                if hashlib.sha256(json.dumps(rec["value"], sort_keys=True).encode()).hexdigest() != rec["sha256"]:
                    bad.append(p.stem)
            except (OSError, ValueError, KeyError):
                bad.append(p.stem)
        return bad


def ledger(path, record: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
