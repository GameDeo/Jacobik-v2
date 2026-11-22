"""Pure CuPy + Triton GPU runtime for jacobik (zero PyTorch dependency).

Provides:
- CuPy NVRTC ``CUDA_PATH`` configuration and Triton 3.7+ driver binding (``CuPyTritonDriver``).
- ``PtrWrap`` / ``wrap`` / ``launch`` for passing ``cupy.ndarray`` directly to ``@triton.jit`` kernels.
- Minimal array, stream, RNG, memory, and CUDAGraph-safe ``libcublas`` GEMM helpers.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

_TARGET_INC = Path(sys.prefix) / "targets" / "x86_64-linux"
if (_TARGET_INC / "include" / "cuda_runtime.h").is_file():
    os.environ.setdefault("CUDA_PATH", str(_TARGET_INC))

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402
import triton  # noqa: E402
from triton.backends.nvidia.driver import CudaDriver, CudaLauncher, CudaUtils  # noqa: E402

OOMError = cp.cuda.memory.OutOfMemoryError


class CuPyTritonDriver(CudaDriver):
    """Bind Triton's active CUDA driver to CuPy's device and stream without importing torch."""

    def __init__(self) -> None:
        self.utils = CudaUtils()
        self.launcher_cls = CudaLauncher
        self.get_device_capability = lambda dev=None: tuple(
            map(int, cp.cuda.Device(cp.cuda.get_device_id() if dev is None else dev).compute_capability)
        )
        self.get_current_stream = lambda dev=0: cp.cuda.get_current_stream().ptr
        self.get_current_device = cp.cuda.get_device_id
        self.set_current_device = lambda dev: cp.cuda.Device(dev).use()

    @staticmethod
    def is_active() -> bool:
        return True


def _init_driver() -> CuPyTritonDriver:
    drv = CuPyTritonDriver()
    triton.runtime.driver.set_active(drv)
    return drv


_DRIVER = _init_driver()


class PtrWrap:
    """Lightweight Triton pointer adapter for ``cupy.ndarray`` (exposes ``.data_ptr()`` and ``.dtype``)."""

    __slots__ = ("_arr", "dtype")

    def __init__(self, arr: cp.ndarray) -> None:
        self._arr = arr
        self.dtype = str(arr.dtype)

    def data_ptr(self) -> int:
        return int(self._arr.data.ptr)


def wrap(x: Any) -> Any:
    """Wrap ``cupy.ndarray`` for Triton kernel dispatch; pass scalars and other args through unchanged."""
    return PtrWrap(x) if isinstance(x, cp.ndarray) else x


def launch(kernel: Any, grid: Tuple[int, ...], *args: Any, **kwargs: Any) -> Any:
    """Launch a Triton ``@jit`` kernel with automatic ``cupy.ndarray`` pointer wrapping."""
    return kernel[grid](*(wrap(a) for a in args), **kwargs)


def cdiv(x: int, y: int) -> int:
    """Ceiling integer division ``(x + y - 1) // y``."""
    return (x + y - 1) // y


def as_gpu(x: Any, dtype: Any = cp.float32) -> cp.ndarray:
    """Ensure ``x`` is a contiguous ``cupy.ndarray`` on the active device."""
    if isinstance(x, cp.ndarray):
        return cp.ascontiguousarray(x.astype(dtype, copy=False))
    return cp.ascontiguousarray(cp.asarray(np.asarray(x), dtype=dtype))


def to_cpu(x: Any) -> np.ndarray:
    """Copy a GPU array to a NumPy array (or return ``np.asarray(x)`` if already on host)."""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def empty(shape: Sequence[int] | int, dtype: Any = cp.float32) -> cp.ndarray:
    return cp.empty(shape, dtype=dtype)


def zeros(shape: Sequence[int] | int, dtype: Any = cp.float32) -> cp.ndarray:
    return cp.zeros(shape, dtype=dtype)


def ones(shape: Sequence[int] | int, dtype: Any = cp.float32) -> cp.ndarray:
    return cp.ones(shape, dtype=dtype)


def full(shape: Sequence[int] | int, fill_value: float | int, dtype: Any = cp.float32) -> cp.ndarray:
    return cp.full(shape, fill_value, dtype=dtype)


def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: Any = cp.int64) -> cp.ndarray:
    return cp.arange(start, stop, step, dtype=dtype)


def new_stream(non_blocking: bool = True) -> cp.cuda.Stream:
    """Allocate a new CUDA stream."""
    return cp.cuda.Stream(non_blocking=non_blocking)


def sync(stream: Optional[cp.cuda.Stream] = None) -> None:
    """Synchronize ``stream`` (or the active device if ``stream is None``)."""
    if stream is not None:
        stream.synchronize()
    else:
        cp.cuda.runtime.deviceSynchronize()


class CapturedGraph:
    """A captured CUDA graph and the private memory pool its kernels read and write.

    Every array allocated while ``fn`` was captured comes from ``pool``, which only this graph uses: blocks its
    temporaries return to the pool stay reserved for its replays, so neither another allocation nor a
    ``free_all_blocks()`` of the default pool (out-of-memory retries, ``empty_cache``) can hand them to other work
    while the graph is alive (torch's private graph pools do the same). Arrays the caller keeps from the capture
    (output buffers) keep their own blocks valid after it goes. When the graph is released its pool's free blocks (its
    temporaries) go back to the device: a buffer allocated during capture and kept by its owner (a cached scorer
    workspace, say) would otherwise keep the whole pool and its cached blocks alive, about 0.7 GB per solve on FMNBCU."""

    __slots__ = ("graph", "pool")

    def __init__(self, graph: Any, pool: cp.cuda.MemoryPool) -> None:
        self.graph, self.pool = graph, pool

    def launch(self, stream: cp.cuda.Stream) -> None:
        self.graph.launch(stream)

    def __del__(self) -> None:
        # no CUDA call here: garbage collection can run while another stream is being captured, where a synchronize,
        # a graph destroy or a cudaFree is illegal (the capture would fail). The graph and its pool are retired and
        # released by _release_retired outside any capture (the next capture, empty_cache).
        if self.pool is not None:
            _RETIRED.append((self.graph, self.pool))


#: (graph, pool) of released CapturedGraphs, freed by _release_retired
_RETIRED: list = []


def release_retired() -> None:
    """Public form of _release_retired, for callers between captures (core.relax after a pass)."""
    _release_retired()


def _release_retired() -> None:
    """Destroy retired graphs and return their pools' free blocks to the device (call outside stream capture)."""
    if not _RETIRED:
        return
    cp.cuda.Device().synchronize()                      # no replay of a retired graph still in flight
    while _RETIRED:
        graph, pool = _RETIRED.pop()
        del graph
        pool.free_all_blocks()


def capture(stream: cp.cuda.Stream, fn: Callable[[], Any]) -> CapturedGraph:
    """Capture ``fn()`` on ``stream`` into a graph with its own memory pool (call ``.launch(stream)``)."""
    _release_retired()
    # the default pool's cached blocks (e.g. the eager warm-up of this very computation) are not available to the
    # private pool below; no graph uses default-pool blocks, so releasing them here is safe and leaves the capture
    # the memory the pass sizing (core.relax) counted on
    cp.get_default_memory_pool().free_all_blocks()
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc), stream:
        stream.begin_capture()
        try:
            fn()
        except BaseException:
            try:
                stream.end_capture()
            except Exception:  # noqa: BLE001 - the capture already failed; its own error is the one raised
                pass
            raise
        return CapturedGraph(stream.end_capture(), pool)


class SegmentSum:
    """Deterministic scatter-add for a fixed layout: ``out[dest[t]] += values[t]`` without atomics.

    The destinations are sorted once (stable) at construction; each call gathers the values in that order, takes one
    float64 prefix sum and differences it at the segment ends, so the same values give the same sums on every run
    (``cp.add.at`` adds in whatever order the atomics land). ``values`` may carry trailing dimensions."""

    def __init__(self, dest: cp.ndarray, n_out: int) -> None:
        d = cp.asarray(dest, dtype=cp.int64).ravel()
        self.n_in, self.n_out = int(d.size), int(n_out)
        self.order = cp.argsort(d, kind="stable") if self.n_in else cp.zeros(0, dtype=cp.int64)
        ds = d[self.order]
        first = cp.ones(self.n_in, dtype=cp.bool_)
        if self.n_in > 1:
            first[1:] = ds[1:] != ds[:-1]
        start = cp.flatnonzero(first)
        self.uniq = ds[start]
        self.end = cp.concatenate([start[1:], cp.asarray([self.n_in], dtype=start.dtype)]) - 1
        self.before = start - 1                              # prefix index just before each segment (-1: none)

    def __call__(self, values: cp.ndarray) -> cp.ndarray:
        tail = values.shape[1:]
        out = cp.zeros((self.n_out,) + tail, dtype=values.dtype)
        if self.n_in == 0:
            return out
        c = cp.cumsum(values.reshape(self.n_in, -1)[self.order].astype(cp.float64), axis=0)
        c0 = cp.concatenate([cp.zeros((1, c.shape[1]), dtype=c.dtype), c])
        seg = c0[self.end + 1] - c0[self.before + 1]
        out.reshape(self.n_out, -1)[self.uniq] = seg.astype(values.dtype)
        return out


def empty_cache() -> None:
    """Release retired graphs and the unused blocks of CuPy's default and pinned memory pools."""
    _release_retired()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def mem_info() -> Tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for the active CUDA device."""
    return cp.cuda.runtime.memGetInfo()


def seed(s: int) -> None:
    """Seed CuPy's default GPU RNG."""
    cp.random.seed(int(s))


def rng_state() -> Any:
    """Set the active CuPy generator aside: the caller's generator is returned and a fresh one installed, so draws
    made until `rng_restore` do not advance the caller's stream (CuPy has no state export; swapping the generator
    object is the equivalent of saving and restoring its state)."""
    st = cp.random.get_random_state()
    cp.random.set_random_state(cp.random.RandomState(0))
    return st


def rng_restore(state: Any) -> None:
    """Put back the generator `rng_state` set aside."""
    cp.random.set_random_state(state)


# ---------------------------------------------------------------------------
# CUDAGraph-safe cuBLAS GEMM (avoids CuPy's per-call cublasSetStream check)
# ---------------------------------------------------------------------------

def _load_cublas():
    p = Path(sys.prefix) / "lib" / "libcublas.so.12"
    lib = ctypes.CDLL(str(p) if p.is_file() else "libcublas.so.12")
    lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cublasSgemm_v2.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_int,
    ]
    lib.cublasDgemm_v2.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p, ctypes.c_int,
    ]
    handle = ctypes.c_void_p()
    err = lib.cublasCreate_v2(ctypes.byref(handle))
    if err != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with error {err}")
    return lib, handle


_CUBLAS_LIB = None
_CUBLAS_HANDLE = None


def _get_cublas():
    global _CUBLAS_LIB, _CUBLAS_HANDLE
    if _CUBLAS_LIB is None:
        _CUBLAS_LIB, _CUBLAS_HANDLE = _load_cublas()
    return _CUBLAS_LIB, _CUBLAS_HANDLE


def set_blas_stream(stream: Optional[cp.cuda.Stream] = None) -> None:
    """Bind the internal cuBLAS handle to ``stream`` (call once before CUDA graph capture)."""
    lib, handle = _get_cublas()
    ptr = cp.cuda.get_current_stream().ptr if stream is None else stream.ptr
    err = lib.cublasSetStream_v2(handle, ctypes.c_void_p(ptr))
    if err != 0:
        raise RuntimeError(f"cublasSetStream_v2 failed with error {err}")


def gemm(
    A: cp.ndarray,
    B: cp.ndarray,
    out: cp.ndarray,
    trans_a: bool = False,
    trans_b: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> cp.ndarray:
    """Compute ``out = alpha * op(A) @ op(B) + beta * out`` in-place via ``libcublas`` (CUDAGraph-safe)."""
    lib, handle = _get_cublas()
    if not (A.dtype == B.dtype == out.dtype) or A.ndim != 2 or B.ndim != 2 or out.ndim != 2:
        raise TypeError(f"gemm needs 2-D operands of one dtype, got {A.dtype}{A.shape}, {B.dtype}{B.shape}, {out.dtype}{out.shape}")
    if not (A.flags.c_contiguous and B.flags.c_contiguous and out.flags.c_contiguous):
        raise ValueError("gemm needs C-contiguous operands (leading dimensions are taken from the shapes)")
    m, k = (A.shape[1], A.shape[0]) if trans_a else A.shape
    k2, n = (B.shape[1], B.shape[0]) if trans_b else B.shape
    if k != k2 or out.shape != (m, n):
        raise ValueError(f"Incompatible GEMM shapes: op(A)=({m},{k}), op(B)=({k2},{n}), out={out.shape}")
    lda = A.shape[1]
    ldb = B.shape[1]
    ldc = out.shape[1]
    op_a = 1 if trans_a else 0
    op_b = 1 if trans_b else 0
    if out.dtype == cp.float32:
        a_c = ctypes.c_float(alpha)
        b_c = ctypes.c_float(beta)
        err = lib.cublasSgemm_v2(
            handle, op_b, op_a, n, m, k,
            ctypes.byref(a_c), B.data.ptr, ldb, A.data.ptr, lda,
            ctypes.byref(b_c), out.data.ptr, ldc,
        )
    elif out.dtype == cp.float64:
        a_c = ctypes.c_double(alpha)
        b_c = ctypes.c_double(beta)
        err = lib.cublasDgemm_v2(
            handle, op_b, op_a, n, m, k,
            ctypes.byref(a_c), B.data.ptr, ldb, A.data.ptr, lda,
            ctypes.byref(b_c), out.data.ptr, ldc,
        )
    else:
        raise TypeError(f"Unsupported GEMM dtype: {out.dtype}")
    if err != 0:
        raise RuntimeError(f"cublasGemm failed with error {err}")
    return out
