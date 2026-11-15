"""pytest in N shards (one process per shard), outcomes read from JUnit XML.

Files are balanced by their duration in the previous run (pytest_durations.json beside the shard outputs, refreshed
after every run), else by file size. pytest-xdist is not in the lock; sharding by file uses only the existing
environment.
"""
from __future__ import annotations

import json

import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import REPO, base_env


def test_files() -> list[Path]:
    files = sorted((REPO / "dev" / "tests").glob("test_*.py"))
    return [f for f in files if f.is_file()]


def shard(files: list[Path], n: int, durations: dict | None = None) -> list[list[Path]]:
    """Longest first to the least loaded shard; cost = previous duration (s) or, without one, size/2000 (rough s)."""
    durations = durations or {}
    cost = lambda f: float(durations.get(f.name, f.stat().st_size / 2000.0))
    bins: list[list[Path]] = [[] for _ in range(max(1, n))]
    sizes = [0.0] * len(bins)
    for f in sorted(files, key=lambda p: -cost(p)):
        i = sizes.index(min(sizes))
        bins[i].append(f)
        sizes[i] += cost(f)
    return [b for b in bins if b]


def _file_durations(xml_path: Path) -> dict:
    out: dict = {}
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        mod = (tc.get("classname") or "").split(".")
        name = next((m for m in reversed(mod) if m.startswith("test_")), None)
        if name:
            out[name + ".py"] = out.get(name + ".py", 0.0) + float(tc.get("time") or 0.0)
    return out


def _outcomes(xml_path: Path) -> dict:
    out = {}
    root = ET.parse(xml_path).getroot()
    for tc in root.iter("testcase"):
        name = f"{tc.get('classname')}::{tc.get('name')}"
        status = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                status = child.tag
                break
            if child.tag == "skipped":
                status = "skipped"
        out[name] = dict(status=status, message=(next((c.get("message", "") for c in tc if c.tag in ("failure", "error", "skipped")), "") or "")[:300])
    return out


def run(out_dir: Path, shards: int = 4, only: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = test_files()
    if only:
        files = [f for f in files if any(o in f.name for o in only)]
    dur_path = out_dir / "pytest_durations.json"
    try:
        durations = json.loads(dur_path.read_text())
    except Exception:
        durations = {}
    groups = shard(files, shards, durations)
    env = base_env()

    def one(i_files):
        i, fs = i_files
        xml = out_dir / f"junit_{i}.xml"
        log = out_dir / f"pytest_{i}.log"
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}",
               *[str(f) for f in fs]]
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        return i, rc, xml, log, [f.name for f in fs]

    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
        results = list(pool.map(one, enumerate(groups)))
    outcomes: dict = {}
    shard_info = []
    for i, rc, xml, log, names in results:
        shard_info.append(dict(shard=i, rc=rc, files=names, log=str(log)))
        if xml.exists():
            outcomes.update(_outcomes(xml))
        else:
            outcomes[f"shard{i}::no-junit"] = dict(status="error", message=f"no junit xml, rc={rc}, log {log}")
    new_dur = dict(durations)
    for i, rc, xml, log, names in results:
        if xml.exists():
            try:
                new_dur.update(_file_durations(xml))
            except Exception:
                pass
    try:
        dur_path.write_text(json.dumps(new_dur, indent=1, sort_keys=True))
    except Exception:
        pass
    counts = {}
    for v in outcomes.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    failing = sorted(k for k, v in outcomes.items() if v["status"] in ("failure", "error"))
    return dict(counts=counts, failing=failing, outcomes=outcomes, shards=shard_info)
