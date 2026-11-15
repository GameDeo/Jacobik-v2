"""CUDA MPS for the lanes of one run (runner option --mps, off by default).

Without MPS, lanes are separate CUDA contexts that time-slice the GPU: kernels of different lanes never run at the
same time. Under MPS they share one server context and their kernels can overlap on the SMs.

The daemon is private to the run: its pipe and log directories are fresh (short, under /tmp: a unix socket path is
limited to ~100 characters; only a socket and a few small log files go there), and the lanes find it through
CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY, which they inherit from this process. `stop()` sends `quit` to that
daemon and waits until its control and server processes (found by PID in /proc, never by name matching) are gone.
On Perlmutter compute nodes the same applies per node (the job script calls the runner with --mps).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

NAMES = ("nvidia-cuda-mps", "nvidia-cuda-mps-control", "nvidia-cuda-mps-server")


def _own_mps_pids(pipe_dir: str | None = None) -> list[int]:
    """PIDs of this user's MPS control/server processes; with pipe_dir, only those started for that directory."""
    out, uid = [], os.getuid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != uid:
                continue
            comm = (p / "comm").read_text().strip()
            if not comm.startswith("nvidia-cuda-mps"):
                continue
            if pipe_dir is not None:
                env = (p / "environ").read_bytes().split(b"\0")
                if f"CUDA_MPS_PIPE_DIRECTORY={pipe_dir}".encode() not in env:
                    continue
            out.append(int(p.name))
        except (OSError, ValueError):
            continue
    return out


class MPS:
    def __init__(self):
        self.pipe = self.log = None
        self.root = None
        self.started = False

    def start(self) -> "MPS":
        exe = shutil.which("nvidia-cuda-mps-control")
        if exe is None:
            raise RuntimeError("--mps: nvidia-cuda-mps-control not found")
        self.root = tempfile.mkdtemp(prefix="jkmps", dir="/tmp")
        self.pipe, self.log = os.path.join(self.root, "p"), os.path.join(self.root, "l")
        os.makedirs(self.pipe)
        os.makedirs(self.log)
        env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=self.pipe, CUDA_MPS_LOG_DIRECTORY=self.log)
        r = subprocess.run([exe, "-d"], env=env, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"--mps: daemon did not start: {r.stderr.strip()[:300]}")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = self.log
        os.environ["TOOLS_RUN_MPS_CLIENT"] = "1"
        self.started = True
        return self

    def stop(self, timeout: float = 30.0) -> dict:
        """Quit the daemon; returns {"stopped": bool, "left": [pids]}. Kills leftovers of this daemon by PID."""
        if not self.started:
            return dict(stopped=True, left=[])
        env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=self.pipe, CUDA_MPS_LOG_DIRECTORY=self.log)
        try:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env=env, capture_output=True, text=True,
                           timeout=timeout)
        except Exception:
            pass
        t0 = time.time()
        left = _own_mps_pids(self.pipe)
        while left and time.time() - t0 < timeout:
            time.sleep(0.5)
            left = _own_mps_pids(self.pipe)
        for pid in left:  # ours (same pipe directory), by PID
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        for k in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "TOOLS_RUN_MPS_CLIENT"):
            os.environ.pop(k, None)
        self.started = False
        server_log = ""
        try:
            server_log = (Path(self.log) / "server.log").read_text()[-2000:]
        except OSError:
            pass
        shutil.rmtree(self.root, ignore_errors=True)
        return dict(stopped=not _own_mps_pids(self.pipe), left=left, server_log_tail=server_log[-400:])
