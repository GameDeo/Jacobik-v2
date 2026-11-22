"""Wall time per pipeline stage (embed, MACE embed, assignment search, IK solve, dedup, settle, relax).

Stages are inclusive and may nest (the MACE embed runs inside the embed; relax runs inside settle); a stage that re-enters
itself is counted once, at its outermost call. GPU work is asynchronous, so a stage synchronises CUDA when it ends: the
time is charged to the stage that launched the work. The cost is one synchronisation per stage call (a few per parent).

    from jacobik.core import timing
    timing.reset()
    ...                         # generate(), or anything that calls the instrumented functions
    timing.snapshot()           # {"embed": 1.2, "ik_solve": 3.4, ...} seconds, and {"<stage>_n": calls}

`generate()` resets the counters and attaches the snapshot to its result (EnsembleResult.timings).
"""
from __future__ import annotations

import functools
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict

_lock = threading.Lock()
_seconds: Dict[str, float] = defaultdict(float)
_calls: Dict[str, int] = defaultdict(int)
_depth = threading.local()


def _sync() -> None:
    try:
        import sys

        gpu = sys.modules.get("jacobik.core.gpu")
        if gpu is not None:
            gpu.sync()
    except Exception:  # timing never fails the caller
        pass


@contextmanager
def stage(name: str):
    """Charge the wall time of the block to `name` (outermost entry only)."""
    depth = getattr(_depth, "d", None)
    if depth is None:
        depth = _depth.d = defaultdict(int)
    depth[name] += 1
    outer = depth[name] == 1
    t0 = time.perf_counter()
    try:
        yield
    finally:
        depth[name] -= 1
        if outer:
            _sync()
            dt = time.perf_counter() - t0
            with _lock:
                _seconds[name] += dt
                _calls[name] += 1


def timed(name: str) -> Callable:
    """Decorator form of stage()."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            with stage(name):
                return fn(*a, **k)
        return inner
    return wrap


def reset() -> None:
    with _lock:
        _seconds.clear()
        _calls.clear()


def snapshot() -> Dict[str, float]:
    with _lock:
        out = {k: round(v, 3) for k, v in _seconds.items()}
        out.update({f"{k}_n": n for k, n in _calls.items()})
    return out
