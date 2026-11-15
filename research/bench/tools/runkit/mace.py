"""MACE-POLAR-1-S energies: one evaluator per process, exact-duplicate structures evaluated once, OOM-safe.

Wraps jacobik.core.energy.MACEPolarEvaluator exactly as A4's relax.mace_kj builds it (float64, energy-only, same model
file), so energies are bit-identical to A4's for the same structure list. Dedup keeps first occurrences in order and
passes only those to the model in ONE call (the evaluator batches internally by atoms), then scatters the values back.
"""
from __future__ import annotations

import hashlib
import os
import time
import warnings
from pathlib import Path

import numpy as np

from . import cache as cache_mod
from . import resources

EV_TO_KJ = 96.4853321233
MODEL_REL = "research/bench/tools/jdev/MACE-POLAR-1-S.model"


def model_path() -> Path:
    for base in (Path(__file__).resolve().parents[4], resources.MAIN):
        if (base / MODEL_REL).exists():
            return base / MODEL_REL
    return Path(__file__).resolve().parents[4] / MODEL_REL


def structure_key(symbols, X, charge, spin) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update((",".join(map(str, symbols)) + f"|{int(charge)}|{int(spin)}|").encode())
    h.update(np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes())
    return h.digest()


def _is_oom(exc) -> bool:
    return "out of memory" in str(exc).lower()


class Evaluator:
    """energies_kj(structures) with structures = [(symbols, X, charge, spin), ...] -> ndarray (kJ/mol).

    device: "auto" (cuda when visible), "cuda" or "cpu". On CPU a one-time warning gives the measured s/structure.
    `backend=` lets tests inject a mock with `.energies(structs) -> eV array`."""

    def __init__(self, device: str = "auto", dtype: str = "float64", batch_atoms: int = 900, threads: int = 1,
                 mem_fraction: float = 0.80, backend=None, cache=None):
        self.dtype, self.batch_atoms, self.threads, self.cache = dtype, int(batch_atoms), threads, cache
        self._backend, self._mem = backend, mem_fraction
        self.device = device if device != "auto" else self._auto()
        self.stats = dict(calls=0, requested=0, evaluated=0, dedup_saved=0, oom_retries=0, cache_hits=0, wall_s=0.0)
        self._warned = False

    @staticmethod
    def _auto() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self):
        if self._backend is None:
            if self.device == "cuda":
                import torch
                torch.cuda.set_per_process_memory_fraction(self._mem)
                os.environ.setdefault("JACOBIK_MACE_BATCH_ATOMS", str(self.batch_atoms))
            from jacobik.core.energy import MACEPolarEvaluator
            ev = MACEPolarEvaluator(model="polar-1-s", device=self.device, dtype=self.dtype, model_path=str(model_path()),
                                    mode="energy-only", threads=self.threads).load()
            if self.device == "cuda":
                ev._batch_atoms = self.batch_atoms
            self._backend = ev
        return self._backend

    @property
    def model_sha256(self) -> str:
        return cache_mod.sha256_file(model_path())

    def _call(self, structs):
        ev = self._load()
        try:
            return np.asarray(ev.energies(structs), float)
        except RuntimeError as exc:
            if not _is_oom(exc) or len(structs) == 1:
                raise
            self.stats["oom_retries"] += 1
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            h = len(structs) // 2
            return np.concatenate([self._call(structs[:h]), self._call(structs[h:])])

    def energies_ev(self, structures) -> np.ndarray:
        structures = [(list(s), np.asarray(x, float), int(q), int(m)) for (s, x, q, m) in structures]
        keys = [structure_key(*st) for st in structures]
        first, order = {}, []
        for i, k in enumerate(keys):
            if k not in first:
                first[k] = len(order)
                order.append(i)
        t0 = time.perf_counter()
        vals = np.empty(len(order))
        todo = list(range(len(order)))
        ckeys = [None] * len(order)
        if self.cache is not None:
            mid = f"mace:{self.dtype}:{self.device}"
            for j, i in enumerate(order):
                ckeys[j] = cache_mod.key("mace", mid, structures[i][0], structures[i][1], structures[i][2], structures[i][3])
            todo = []
            for j in range(len(order)):
                hit = self.cache.get(ckeys[j])
                if hit is None:
                    todo.append(j)
                else:
                    vals[j] = hit["eV"]
                    self.stats["cache_hits"] += 1
        if todo:
            out = self._call([structures[order[j]] for j in todo])
            for j, v in zip(todo, out):
                vals[j] = v
                if self.cache is not None:
                    self.cache.put(ckeys[j], dict(eV=float(v)))
        dt = time.perf_counter() - t0
        s = self.stats
        s["calls"] += 1
        s["requested"] += len(structures)
        s["evaluated"] += len(todo)
        s["dedup_saved"] += len(structures) - len(order)
        s["wall_s"] += dt
        if self.device == "cpu" and todo and not self._warned and self._backend is not None and not hasattr(self._backend, "_mock"):
            self._warned = True
            warnings.warn(f"runkit.mace on CPU: {dt / len(todo):.2f} s/structure (a GPU is ~100x faster)")
        return np.array([vals[first[k]] for k in keys], float)

    def energies_kj(self, structures) -> np.ndarray:
        return self.energies_ev(structures) * EV_TO_KJ

    def single_kj(self, symbols, Xs, charge: int, spin: int) -> np.ndarray:
        """Same-molecule convenience, equal to A4's relax.mace_kj(symbols, Xs, charge, spin)."""
        return self.energies_kj([(symbols, x, charge, spin) for x in Xs])


_SINGLETON: dict = {}


def evaluator(device: str = "cpu", threads: int = 1, **kw) -> Evaluator:
    key = (device, threads)
    if key not in _SINGLETON:
        _SINGLETON[key] = Evaluator(device=device, threads=threads, **kw)
    return _SINGLETON[key]


# ---------------------------------------------------------------------------------------------------------------------
# CPU-node MACE: one single-threaded Evaluator per process. Per-core throughput is far better than intra-op threads
# (Perlmutter EPYC, MACE-POLAR-1-S float64: 1.28 s/structure on 1 thread vs 0.41 s on 16 threads), so a 128-core node
# evaluates about 100 structures/s, close to one A100.
_POOL_EV = None


def _pool_init(dtype: str):
    import os
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    global _POOL_EV
    _POOL_EV = Evaluator(device="cpu", dtype=dtype, threads=1)


def _pool_call(chunk):
    return _POOL_EV.energies_ev(chunk)


class PoolEvaluator:
    """energies_kj(structures) on CPU across `workers` processes (spawned once per call, largest structures first).

    Same model, dtype and per-structure arithmetic as Evaluator(device="cpu"); only the scheduling differs."""

    def __init__(self, workers: int, dtype: str = "float64", chunk: int = 4):
        self.workers, self.dtype, self.chunk = max(1, int(workers)), dtype, max(1, int(chunk))
        self.stats = dict(calls=0, requested=0, evaluated=0, wall_s=0.0, workers=self.workers, device="cpu-pool")

    def energies_ev(self, structures) -> np.ndarray:
        import multiprocessing as mp
        structs = [(list(s), np.asarray(x, float), int(q), int(m)) for (s, x, q, m) in structures]
        if not structs:
            return np.zeros(0)
        t0 = time.perf_counter()
        order = sorted(range(len(structs)), key=lambda i: -len(structs[i][0]))
        chunks = [order[i:i + self.chunk] for i in range(0, len(order), self.chunk)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(self.workers, len(chunks)), initializer=_pool_init, initargs=(self.dtype,)) as pool:
            res = pool.map(_pool_call, [[structs[i] for i in c] for c in chunks], chunksize=1)
        out = np.empty(len(structs))
        for c, vals in zip(chunks, res):
            out[c] = vals
        s = self.stats
        s["calls"] += 1
        s["requested"] += len(structs)
        s["evaluated"] += len(structs)
        s["wall_s"] += time.perf_counter() - t0
        return out

    def energies_kj(self, structures) -> np.ndarray:
        return self.energies_ev(structures) * EV_TO_KJ
