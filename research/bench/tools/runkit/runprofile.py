"""Stage traces, cost projection and the "estimate before a panel run" rule. (Named runprofile: jacobik.core.profile is the solver profile.)"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

from . import cache as cache_mod

L4_UNITS_PER_H = 1.5


class Trace:
    """Named stages with wall and process-CPU seconds: `with tr("xtb"): ...`; also tr.add("xtb_cpu_s", 12.3) for imported timings."""

    def __init__(self):
        self.stages: dict = {}

    @contextmanager
    def __call__(self, name: str):
        w, c = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            s = self.stages.setdefault(name, dict(wall_s=0.0, cpu_s=0.0, n=0))
            s["wall_s"] += time.perf_counter() - w
            s["cpu_s"] += time.process_time() - c
            s["n"] += 1

    def add(self, name: str, wall_s: float, cpu_s: float | None = None):
        s = self.stages.setdefault(name, dict(wall_s=0.0, cpu_s=0.0, n=0))
        s["wall_s"] += wall_s
        s["cpu_s"] += wall_s if cpu_s is None else cpu_s
        s["n"] += 1

    def total(self) -> float:
        return sum(s["wall_s"] for s in self.stages.values())

    def write(self, out, extra: dict | None = None):
        p = Path(out) / "profile.json"
        cache_mod.atomic_write(p, json.dumps(dict(stages=self.stages, total_wall_s=self.total(), **(extra or {})), indent=1).encode())
        return p


def measure_items(fn, items, n: int = 2) -> float:
    """Mean seconds per item over the first n items (profile 1 to 2 cases before a panel-scale run)."""
    items = list(items)[:max(1, n)]
    t0 = time.perf_counter()
    for it in items:
        fn(it)
    return (time.perf_counter() - t0) / len(items)


@dataclass
class Projection:
    n_items: int
    per_item_s: float
    workers: int
    speed: float
    overlap: float
    wall_s: float
    units: float

    def minutes(self) -> float:
        return self.wall_s / 60


def project(n_items: int, per_item_s: float, workers: int = 1, speed: float = 1.0, overlap: float = 0.0,
            units_per_h: float = 0.0) -> Projection:
    """wall = n * per_item / (workers * speed) * (1 - overlap) ; units = wall_h * units_per_h (Colab L4: 1.5)."""
    wall = n_items * per_item_s / (max(1, workers) * speed) * (1.0 - overlap)
    return Projection(n_items, per_item_s, workers, speed, overlap, wall, wall / 3600 * units_per_h)


def require_budget(proj: Projection, max_min: float = 15, record=None) -> dict:
    """The refusal rule: returns {'ok': True} when the projection fits, else a refusal record (also appended to `record`,
    a jsonl path) and ok False. The caller must not start the run when ok is False."""
    ok = proj.minutes() <= max_min
    rec = dict(ok=ok, projected_min=round(proj.minutes(), 2), max_min=max_min, projected_units=round(proj.units, 3),
               projection=asdict(proj))
    if not ok:
        rec["refusal"] = f"estimated {proj.minutes():.1f} min exceeds {max_min} min: defer or get approval"
        if record:
            cache_mod.ledger(record, rec)
    return rec


def gpu_step_time(fn, repeats: int = 5, warmup: int = 2) -> dict:
    """Median seconds per call of a CUDA callable with proper synchronisation (eager time; graph capture stays in nnc)."""
    import statistics
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return dict(median_s=statistics.median(ts), min_s=min(ts), repeats=repeats)
