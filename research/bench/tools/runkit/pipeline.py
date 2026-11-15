"""Pipeline pieces: Watcher (producer/consumer by completeness + settle), Prefetcher, split_by_cost, RemoteFeed, run_groups.

Generalised from A4 run.py (_ready_parents, --shard/--skip-file/--done-file), A4 local_feed.py and Gemini's _prefetch_parent.
"""
from __future__ import annotations

import glob
import os
import queue
import shutil
import threading
import time
import traceback
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import cache as cache_mod


def crc_shard(key: str, shard: str = "0/1") -> bool:
    """True if `key` belongs to shard "K/N" (crc32(key) % N == K), as in A4 run.py."""
    k, n = (int(x) for x in shard.split("/"))
    return zlib.crc32(str(key).encode()) % n == k


def code_of(path: str, sep: str = "_s") -> str:
    return os.path.basename(path).split(sep)[0]


def zip_ok(path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


class Watcher:
    """Iterates over groups of files that are ready, optionally waiting for more.

    pattern: glob inside `directory`; group_key: filename -> group key; complete(files, finished) -> bool (default: 3 files,
    or any once the done file exists); settle: seconds the newest file must have been quiet (a writer is finished);
    shard "K/N"; skip_file lists keys to skip (re-read every poll); done_file: its existence means the producer is finished;
    state_file: keys already consumed (call mark_done(key) when a group is processed; a restart skips them).
    watch=False makes one pass. Yields (key, sorted files)."""

    def __init__(self, directory, pattern="*_s*.npz", group_key=code_of, complete=None, settle=8.0, shard="0/1",
                 skip_file=None, done_file=None, state_file=None, poll=20.0, watch=True, only=None, sleep=time.sleep):
        self.dir, self.pattern, self.group_key = str(directory), pattern, group_key
        self.complete = complete or (lambda files, finished: len(files) >= 3 or finished)
        self.settle, self.shard, self.skip_file, self.done_file = settle, shard, skip_file, done_file
        self.state_file, self.poll, self.watch, self.only, self._sleep = state_file, poll, watch, set(only or []), sleep
        self.done = set(Path(state_file).read_text().split()) if state_file and os.path.exists(state_file) else set()
        self.seen = set()

    def mark_done(self, key):
        self.done.add(key)
        if self.state_file:
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "a") as fh:
                fh.write(key + "\n")

    def ready(self, finished: bool) -> dict:
        by = {}
        for f in sorted(glob.glob(os.path.join(self.dir, self.pattern))):
            by.setdefault(self.group_key(f), []).append(f)
        now = time.time()
        return {k: fs for k, fs in by.items()
                if self.complete(fs, finished) and now - max(os.path.getmtime(f) for f in fs) > self.settle}

    def _skip(self) -> set:
        return set(Path(self.skip_file).read_text().split()) if self.skip_file and os.path.exists(self.skip_file) else set()

    def __iter__(self):
        while True:
            finished = bool(self.done_file and os.path.exists(self.done_file))
            ready, skip = self.ready(finished), self._skip()
            todo = [k for k in sorted(ready) if k not in self.seen and k not in self.done and k not in skip
                    and (not self.only or k in self.only) and crc_shard(k, self.shard)]
            for k in todo:
                self.seen.add(k)
                yield k, ready[k]
            if not self.watch or (finished and not todo):
                return
            if not todo:
                self._sleep(self.poll)


class Failed:
    """Returned in place of a value when the prefetch function raised (the loop decides what to do)."""

    def __init__(self, exc, tb=""):
        self.exc, self.tb = exc, tb

    def __repr__(self):
        return f"Failed({self.exc!r})"


class Prefetcher:
    """Runs fn(item) in a background thread up to `depth` items ahead; iterates (item, value-or-Failed) in order."""

    def __init__(self, fn, items, depth: int = 1):
        self.fn, self.items, self.q = fn, list(items), queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def _work(self):
        for it in self.items:
            if self._stop.is_set():
                break
            try:
                v = self.fn(it)
            except BaseException as exc:
                v = Failed(exc, traceback.format_exc()[-1500:])
            while not self._stop.is_set():
                try:
                    self.q.put((it, v), timeout=0.2)
                    break
                except queue.Full:
                    pass
        self.q.put(None) if not self._stop.is_set() else None

    def close(self):
        self._stop.set()

    def __iter__(self):
        try:
            while True:
                r = self.q.get()
                if r is None:
                    return
                yield r
        finally:
            self.close()


@dataclass
class Machine:
    name: str
    cores: int
    speed: float = 1.0   # per-core speed relative to local (Colab L4 vCPUs: 0.5 until measured)

    @property
    def capacity(self) -> float:
        return self.cores * self.speed


def split_by_cost(items, cost, machines) -> dict:
    """Greedy longest-first split of items over machines minimising the finish time (load / capacity).
    cost: callable item -> float. Returns {machine name: [items]} (each list keeps the input order)."""
    load = {m.name: 0.0 for m in machines}
    out = {m.name: [] for m in machines}
    pos = {id(it): i for i, it in enumerate(items)}
    for it in sorted(items, key=lambda x: -cost(x)):
        m = min(machines, key=lambda m: (load[m.name] + cost(it)) / m.capacity)
        load[m.name] += cost(it)
        out[m.name].append(it)
    for k in out:
        out[k].sort(key=lambda it: pos[id(it)])
    return out


def split_makespan(split, cost, machines) -> dict:
    return {m.name: sum(cost(i) for i in split[m.name]) / m.capacity for m in machines}


def write_skip_file(split: dict, keep: str, path):
    """Skip file for machine `keep`: every item assigned to some other machine."""
    cache_mod.atomic_write(path, ("\n".join(str(i) for k, v in split.items() if k != keep for i in v) + "\n").encode())


class RemoteFeed:
    """Copies finished, valid files from a VM directory to a local one (A4 local_feed, no hard-coded paths).

    `session` needs .get(remote_dir, local_stage_dir) (colab.Session has it). Only groups in `codes` are kept (the local share).
    Files are validated (validate(path) -> bool, default zip_ok for .npz) and copied atomically."""

    def __init__(self, session, remote, local, codes, stage, every=150.0, pattern="*.npz", validate=zip_ok,
                 group_key=code_of, sleep=time.sleep):
        self.session, self.remote, self.local, self.stage = session, remote, Path(local), Path(stage)
        self.codes, self.every, self.pattern, self.validate = set(codes) if codes is not None else None, every, pattern, validate
        self.group_key, self._sleep = group_key, sleep
        self._stop = threading.Event()
        self.thread = None
        self.copied = self.rejected = 0

    def sync_once(self) -> int:
        self.local.mkdir(parents=True, exist_ok=True)
        base = os.path.basename(self.remote.rstrip("/"))
        shutil.rmtree(self.stage / base, ignore_errors=True)
        try:
            self.session.get(self.remote, str(self.stage))
        except Exception:
            return 0
        n = 0
        for f in sorted((self.stage / base).glob(self.pattern)):
            dst = self.local / f.name
            if dst.exists() or (self.codes is not None and self.group_key(f.name) not in self.codes):
                continue
            if not self.validate(f):
                self.rejected += 1
                continue
            tmp = str(dst) + ".tmp"
            shutil.copy(f, tmp)
            os.replace(tmp, dst)
            n += 1
        self.copied += n
        return n

    def _loop(self):
        while not self._stop.is_set():
            self.sync_once()
            for _ in range(int(self.every)):
                if self._stop.is_set():
                    break
                self._sleep(1)
        self.sync_once()   # one last fetch after stop

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self, timeout: float = 900):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout)


def run_groups(fn, groups, workers: int = 1, ledger=None) -> dict:
    """Apply fn(key, payload) to every group; a failure is recorded (and in the ledger) and the loop continues.
    groups: iterable of (key, payload). Returns {key: result or {'failed': msg}}."""
    results = {}

    def one(kp):
        k, p = kp
        try:
            return k, fn(k, p)
        except BaseException as exc:
            rec = dict(key=k, failed=f"{type(exc).__name__}: {str(exc)[:200]}", tb=traceback.format_exc()[-1500:])
            if ledger:
                cache_mod.ledger(ledger, rec)
            return k, dict(failed=rec["failed"])

    if workers <= 1:
        for kp in groups:
            k, r = one(kp)
            results[k] = r
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, r in ex.map(one, list(groups)):
                results[k] = r
    return results
