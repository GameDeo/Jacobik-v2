"""xtb adapter: provenance guard, parallel crude relaxation (never raises), gradient/energy.

`opt` runs the same command as A4's relax.xtb_opt (`xtb in.xyz --gfn 2 --opt crude --cycles N --chrg Q --uhf M-1 -P 1`,
OMP_NUM_THREADS=1), so energies and geometries are bit-identical to it. Additions: timeout with process-group kill, optional
content-addressed cache, binary hash and version in every result, failures.jsonl.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import cache as cache_mod

HARTREE_KJ = 2625.4996394799
BOHR_A = 0.529177210903
_E_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_CONV_RE = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS")
FALLBACK_MSG = "GFN2 embed refinement failed"
_INFO: dict = {}


def resolve(binary: str | None = None) -> str:
    p = binary or os.environ.get("JACOBIK_XTB") or shutil.which("xtb")
    if not p or not os.path.exists(p):
        raise SystemExit("xtb does not resolve (put the pixi env bin on PATH or set JACOBIK_XTB); without it the builder "
                         "silently falls back to MMFF and every structure is invalid")
    return os.path.abspath(p)


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info(binary: str | None = None) -> dict:
    """path, sha256 and version of the binary (cached per path)."""
    p = resolve(binary)
    if p not in _INFO:
        try:
            out = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=30).stdout
            m = re.search(r"version\s+(\d+\.\d+\.\d+)", out)
        except Exception:
            m = None
        _INFO[p] = dict(path=p, sha256=_sha(p), version=m.group(1) if m else "unknown")
    return dict(_INFO[p])


_WATER = (["O", "H", "H"], np.array([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0], [-0.30, 0.95, 0.20]]))


def require(binary: str | None = None, selfcheck: bool = True, version_prefix: str | None = None) -> dict:
    """Resolve xtb and prove it really works: a distorted water must relax to about -5.07 Eh and the geometry must move.
    Raises SystemExit with the explanation (so a script stops before it builds invalid structures)."""
    inf = info(binary)
    if version_prefix and not inf["version"].startswith(version_prefix):
        raise SystemExit(f"xtb {inf['version']} found, {version_prefix}* required")
    if selfcheck and "selfcheck" not in _INFO.get(inf["path"], {}):
        sym, X = _WATER
        with tempfile.TemporaryDirectory(dir=os.environ.get("RUNKIT_SELFCHECK_DIR") or None) as td:
            r = opt(sym, X, 0, 1, td, cycles=50, binary=inf["path"])
        moved = float(np.abs(r["X"] - X).max()) if r["X"] is not None else 0.0
        if r["E_h"] is None or not moved > 1e-3 or not (-5.2 < r["E_h"] < -4.9):
            raise SystemExit(f"xtb self-check failed ({r['status']}, E={r['E_h']}, shift={moved:.4f} A): "
                             f"{inf['path']} does not run GFN2 optimisations")
        _INFO[inf["path"]]["selfcheck"] = dict(E_h=r["E_h"], max_shift_A=round(moved, 4))
    return dict(_INFO[inf["path"]])


class FallbackWatch(logging.Handler):
    """Context manager / handler that records the builder's "GFN2 embed refinement failed" warning (silent MMFF fallback).
    `with FallbackWatch() as fw: ...; fw.check()` raises RuntimeError if it fired; `raise_on_exit=True` does it on exit."""

    def __init__(self, raise_on_exit: bool = False):
        super().__init__(level=logging.DEBUG)
        self.hit = 0
        self.messages: list = []
        self.raise_on_exit = raise_on_exit

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if FALLBACK_MSG in msg:
            self.hit += 1
            if len(self.messages) < 20:
                self.messages.append(msg[:300])

    def check(self):
        if self.hit:
            raise RuntimeError(f"{self.hit} embeds fell back to MMFF ('{FALLBACK_MSG}'): every structure built that way is invalid")

    def __enter__(self):
        logging.getLogger().addHandler(self)
        return self

    def __exit__(self, et, ev, tb):
        logging.getLogger().removeHandler(self)
        if self.raise_on_exit and et is None:
            self.check()
        return False


def _xyz(symbols, X) -> str:
    return f"{len(symbols)}\n\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(symbols, X))


def _run(cmd, cwd, env, timeout):
    """subprocess.run with a process-group kill on timeout; returns (rc, stdout, stderr) or raises TimeoutExpired."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        p.communicate()
        raise
    return p.returncode, out, err


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="500M")


def opt(symbols, X, charge: int, spin: int, workroot: str, cycles: int = 200, level: str = "crude", timeout: float = 3600,
        cache=None, binary: str | None = None) -> dict:
    """One relaxation in its own temporary directory under workroot. Never raises: failure goes in the result.
    Result: status ('ok'|'not_converged'|failure text), E_h, cycles, converged, X, wall_s, stderr_tail, cached, xtb_sha256, xtb_version."""
    t0 = time.perf_counter()
    res = dict(status="failed", E_h=None, cycles=None, converged=False, X=None, wall_s=None, stderr_tail="", cached=False,
               xtb_sha256=None, xtb_version=None)
    try:
        inf = info(binary)
        res["xtb_sha256"], res["xtb_version"] = inf["sha256"], inf["version"]
        k = None
        if cache is not None:
            k = cache_mod.key("xtb-opt", inf["sha256"] + ":" + inf["version"], symbols, X, charge, spin,
                              dict(level=level, cycles=int(cycles), gfn=2))
            hit = cache.get(k)
            if hit is not None:
                hit["X"] = np.array(hit["X"]) if hit.get("X") is not None else None
                hit.update(cached=True, wall_s=time.perf_counter() - t0)
                return hit
        Path(workroot).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workroot) as td:
            with open(f"{td}/in.xyz", "w") as f:
                f.write(_xyz(symbols, X))
            cmd = [inf["path"], "in.xyz", "--gfn", "2", "--opt", level, "--cycles", str(cycles), "--chrg", str(int(charge)),
                   "--uhf", str(int(spin) - 1), "-P", "1"]
            rc, out, err = _run(cmd, td, _env(), timeout)
            res["stderr_tail"] = err[-300:]
            m = _E_RE.findall(out)
            c = _CONV_RE.search(out)
            if c:
                res["cycles"], res["converged"] = int(c.group(1)), True
            elif "FAILED TO CONVERGE" in out:
                res["cycles"] = cycles
            else:
                n = len(re.findall(r"^\s*\d+\s+-?\d+\.\d+\s+-?\d+\.\d+", out, re.M))
                res["cycles"] = n or None
            if m and os.path.exists(f"{td}/xtbopt.xyz"):
                L = open(f"{td}/xtbopt.xyz").read().split("\n")[2:2 + len(symbols)]
                res["X"] = np.array([[float(v) for v in ln.split()[1:4]] for ln in L])
                res["E_h"] = float(m[-1])
                res["status"] = "ok" if res["converged"] else "not_converged"
            else:
                res["status"] = f"no_output(rc={rc})"
        if k is not None and res["status"] in ("ok", "not_converged"):
            cache.put(k, {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in res.items() if kk != "wall_s"})
    except subprocess.TimeoutExpired:
        res["status"] = f"timeout({timeout:.0f}s)"
    except BaseException as exc:  # recorded, never raised (SystemExit from a missing binary included)
        res["status"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    res["wall_s"] = time.perf_counter() - t0
    return res


def opt_many(jobs, workers: int, workroot: str, cycles: int = 200, cache=None, timeout: float = 3600,
             failures=None, codes=None, binary: str | None = None, level: str = "crude") -> list:
    """jobs: list of (symbols, X, charge, spin). Results in job order. Threads only wait on subprocesses.
    Failed jobs are appended to `failures` (a jsonl path) with the job index and, if given, codes[i]."""
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(opt, s, X, q, m, workroot, cycles, level, timeout, cache, binary) for (s, X, q, m) in jobs]
        out = []
        for i, f in enumerate(futs):
            try:
                r = f.result()
            except BaseException as exc:  # opt never raises; this is a belt for executor errors
                r = dict(status=f"{type(exc).__name__}: {exc}"[:150], E_h=None, X=None, converged=False, cycles=None,
                         wall_s=None, stderr_tail="", cached=False)
            out.append(r)
            if failures and r["status"] not in ("ok", "not_converged"):
                cache_mod.ledger(failures, dict(job=i, code=codes[i] if codes else None, status=r["status"],
                                                stderr_tail=r.get("stderr_tail", "")))
    return out


def timing(results) -> dict:
    """Aggregate wall/CPU-seconds of a batch of opt results (feeds runprofile)."""
    w = [r["wall_s"] for r in results if r.get("wall_s") is not None]
    comp = [r["wall_s"] for r in results if r.get("wall_s") is not None and not r.get("cached")]
    return dict(n=len(results), n_failed=sum(r["status"] not in ("ok", "not_converged") for r in results),
                n_cached=sum(bool(r.get("cached")) for r in results), cpu_s=float(sum(comp)), wall_sum_s=float(sum(w)))


def parse_gradient(text: str, n: int):
    """Parse an xtb `gradient` file (Turbomole format): returns (E_h, grad (n,3) in Eh/bohr)."""
    lines = text.splitlines()
    head = next((ln for ln in lines if "SCF energy" in ln), None)
    if head is None:
        raise ValueError("no SCF energy line in the gradient file")
    e = float(re.search(r"SCF energy\s*=\s*([-+\d.EeDd]+)", head).group(1).replace("D", "E"))
    start = lines.index(head) + 1 + n
    g = np.array([[float(v.replace("D", "E")) for v in ln.split()] for ln in lines[start:start + n]])
    if g.shape != (n, 3) or not np.isfinite(g).all():
        raise ValueError("invalid xtb gradient block")
    return e, g


def grad(symbols, X, charge: int, spin: int, workroot: str, timeout: float = 180, binary: str | None = None,
         solvent: str | None = None, acc: float = 1.0) -> dict:
    """GFN2 energy and gradient. Never raises. E_h (Eh), grad_h_bohr (Eh/bohr), E_kj, grad_kj_A (kJ/mol/A), status, wall_s."""
    t0 = time.perf_counter()
    res = dict(status="failed", E_h=None, grad_h_bohr=None, E_kj=None, grad_kj_A=None, wall_s=None, stderr_tail="")
    try:
        inf = info(binary)
        res.update(xtb_sha256=inf["sha256"], xtb_version=inf["version"])
        Path(workroot).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workroot) as td:
            Path(td, "in.xyz").write_text(_xyz(symbols, X))
            cmd = [inf["path"], "in.xyz", "--gfn", "2", "--grad", "--chrg", str(int(charge)), "--uhf", str(int(spin) - 1),
                   "--acc", str(acc), "--parallel", "1"] + (["--alpb", solvent] if solvent else [])
            rc, out, err = _run(cmd, td, _env(), timeout)
            res["stderr_tail"] = err[-300:]
            if rc or "normal termination of xtb" not in (out + err).lower():
                res["status"] = f"no_output(rc={rc})"
            else:
                e, g = parse_gradient(Path(td, "gradient").read_text(), len(symbols))
                res.update(status="ok", E_h=e, grad_h_bohr=g, E_kj=e * HARTREE_KJ, grad_kj_A=g * HARTREE_KJ / BOHR_A)
    except subprocess.TimeoutExpired:
        res["status"] = f"timeout({timeout:.0f}s)"
    except BaseException as exc:
        res["status"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    res["wall_s"] = time.perf_counter() - t0
    return res


def energy(symbols, X, charge: int, spin: int, workroot: str, **kw) -> dict:
    """GFN2 single point (via the gradient run): same dict as grad()."""
    return grad(symbols, X, charge, spin, workroot, **kw)
