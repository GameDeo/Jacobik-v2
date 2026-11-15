"""Isolated CPU tasks over explicit parents; no solver or GPU setup.

A task is a module:function receiving one manifest row and returning a JSON
object. A fresh process for each row keeps import-time profiles isolated.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def add_arguments(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True, help="importable module:function")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--seconds-per-parent", type=float, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--estimate", action="store_true")


def _write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _worker(task, result_path):
    module, name = task.split(":")
    parent = json.load(sys.stdin)
    result = getattr(importlib.import_module(module), name)(parent)
    if not isinstance(result, dict):
        raise TypeError("diagnostic task must return a JSON object")
    _write(Path(result_path), result)


def run_diagnostic(args):
    rows = json.loads(args.manifest.read_text())
    if isinstance(rows, dict):
        rows = rows["parents"]
    if not isinstance(rows, list) or not rows:
        raise SystemExit("manifest must contain a nonempty parent list")
    if not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", args.task):
        raise SystemExit("task must be module:function")
    if args.timeout_s <= 0 or args.seconds_per_parent <= 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("time limits, estimates and row limit must be positive")
    ids = [row.get("id", row.get("code")) for row in rows]
    if any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in ids):
        raise SystemExit("each row needs a safe id or code")
    if len(set(ids)) != len(ids):
        raise SystemExit("diagnostic IDs must be unique")
    # Resume only within the same manifest and task. For a revised diagnostic
    # use a fresh output directory, preserving the previous evidence.
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    identity = {"manifest_sha256": digest, "task": args.task}
    out = args.out.resolve()
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise SystemExit("diagnostic output belongs to a different manifest or task")
    work = list(zip(ids, rows))[:args.limit]
    records = out / "records"
    pending = [(key, row) for key, row in work if not (records / f"{key}.json").exists()]
    estimate = (len(pending) + args.jobs - 1) // args.jobs * args.seconds_per_parent
    print(json.dumps({"parents": len(work), "pending": len(pending), "jobs": args.jobs,
                      "estimate_s": estimate, "seconds_per_parent": args.seconds_per_parent}), flush=True)
    if args.estimate:
        return 0
    if estimate > 1200:
        raise SystemExit("diagnostic estimate exceeds 20 minutes; restructure before running")
    records.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    _write(identity_path, identity)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

    def one(item):
        key, row = item
        result_path = records / f"{key}.json"
        staging = records / f"{key}.result"
        started = time.monotonic()
        try:
            with (logs / f"{key}.log").open("wb") as log:
                proc = subprocess.run([sys.executable, "-m", "tools.run.diagnostic",
                                       args.task, str(staging)], input=json.dumps(row).encode(),
                                      stdout=log, stderr=subprocess.STDOUT, env=env,
                                      timeout=args.timeout_s, check=False)
            if proc.returncode != 0:
                result = {"status": "failed", "reason": f"worker exit {proc.returncode}"}
            else:
                result = json.loads(staging.read_text())
                result.setdefault("status", "ok")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            staging.unlink(missing_ok=True)
        _write(result_path, {**result, "id": key, "wall_s": time.monotonic() - started})

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for _ in executor.map(one, pending):
            pass
    failures = [key for key, _ in work
                if json.loads((records / f"{key}.json").read_text())["status"] != "ok"]
    summary = {"parents": len(work), "failures": failures}
    _write(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    _worker(*sys.argv[1:])
