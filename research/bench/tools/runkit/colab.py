"""Colab sessions over cj.py (the substrate, loaded from the same checkout): up without blocking, one-exec status,
fetch -> count -> down, idle watchdog, cost projection and stop rule.

Hard rules (owner 2026-10-28): the VM is always taken down on exit (also after exceptions and on SIGTERM/atexit), and a
count mismatch never keeps a VM beyond the grace period. Only VMs this process started are ever stopped.
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

from . import cache as cache_mod
from . import resources
from .resources import Refused

L4_UNITS_PER_H = 1.5
CPU_UNITS_PER_H = 0.1   # to be replaced by the measured value in machines.json after the probe
LOG_DIR = resources.RUN_DIR / "colab"


def default_cj_path() -> Path:
    for base in (Path(__file__).resolve().parents[4], resources.MAIN):
        p = base / "research/gpuwork/colab/cj.py"
        if p.exists():
            return p
    return Path(__file__).resolve().parents[4] / "research/gpuwork/colab/cj.py"


def load_cj(path=None):
    p = Path(path) if path else default_cj_path()
    spec = importlib.util.spec_from_file_location("cj_" + p.stem + str(abs(hash(str(p))) % 10**6), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def projected_units(hours: float, kind: str = "gpu") -> float:
    return hours * (L4_UNITS_PER_H if kind == "gpu" else CPU_UNITS_PER_H)


def stop_rule(projected: float, cap_units: float | None = None, cj=None) -> float:
    """Refuse (Refused) before launching when the projected spend exceeds `cap_units` or the remaining shared cap.
    Returns the remaining shared units (cap - spent) if cj is given."""
    if cap_units is not None and projected > cap_units:
        raise Refused(f"projected {projected:.2f} units exceeds the run cap {cap_units:.2f}")
    if cj is not None:
        spent = cj.START_BALANCE - cj.balance()[0]
        remaining = cj.CAP_UNITS - spent
        if projected > remaining:
            raise Refused(f"projected {projected:.2f} units exceeds the remaining shared cap {remaining:.2f}")
        return remaining
    return float("nan")


class Session:
    """One Colab VM. `with Session("rk1", repo=wt) as vm:` brings it up (blocking only inside __enter__, with a log file and
    READY marker) and always brings it down on exit."""

    def __init__(self, name: str, gpu: str = "L4", repo=None, ref: str = "HEAD", kind: str = "gpu", cj_path=None,
                 python=None, log_dir=None, up_timeout_min: float = 110, keep_on_mismatch_min: float = 0.0, poll_s: float = 5.0):
        self.name, self.kind, self.ref = name, kind, ref
        self.repo = repo if repo is not None else Path(__file__).resolve().parents[4]
        self.gpu = gpu if kind == "gpu" else "none"
        self.cj_path = Path(cj_path) if cj_path else default_cj_path()
        self.cj = load_cj(self.cj_path)
        self.cli = [python or sys.executable, str(self.cj_path)]
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.up_timeout_min, self.grace_min, self.poll_s = up_timeout_min, keep_on_mismatch_min, poll_s
        self.started = False
        self.downed = False
        self._lock = threading.RLock()   # a second caller of down() waits until the first has finished
        self.t_up = None
        self.units_before = self.units_after = None
        self._proc = None
        self.alerts: list = []
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", PATH=str(resources.PIXI_BIN) + ":" + os.environ.get("PATH", ""))

    # -------- helpers
    def _cli(self, *args, timeout=1800, check=False):
        r = subprocess.run(self.cli + [str(a) for a in args], capture_output=True, text=True, timeout=timeout, env=self.env)
        if check and r.returncode:
            raise RuntimeError(f"cj {args[0]} failed rc={r.returncode}: {(r.stdout + r.stderr)[-800:]}")
        return r

    def credits(self):
        try:
            return float(self.cj.balance()[0])
        except BaseException:
            return None

    def _log(self, event, **kw):
        cache_mod.ledger(self.log_dir / "events.jsonl", dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), vm=self.name, event=event, **kw))

    # -------- lifecycle
    def start(self):
        """Start `cj.py up` in a subprocess with its own log; returns immediately."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.units_before = self.credits()
        self.up_log = self.log_dir / f"{self.name}_up.log"
        self.up_log.write_text("")
        cmd = self.cli + ["up", "--name", self.name, "--gpu", self.gpu, "--ref", self.ref]
        if self.repo is not None:
            cmd += ["--repo", str(self.repo)]
        self._proc = subprocess.Popen(cmd, stdout=open(self.up_log, "ab"), stderr=subprocess.STDOUT, env=self.env,
                                      start_new_session=True)
        self.started = True
        self.t_up = time.time()
        atexit.register(self.down)
        self._log("up_started", gpu=self.gpu, credits=self.units_before, pid=self._proc.pid)
        return self

    def wait_ready(self):
        t0 = time.time()
        try:
            while True:
                txt = self.up_log.read_text(errors="replace")
                if "READY:" in txt:
                    self._log("ready")
                    return txt[txt.rindex("READY:"):].strip()
                if self._proc.poll() is not None:
                    raise RuntimeError("cj up ended without READY (fingerprint check or setup failed): " + txt[-800:])
                if time.time() - t0 > self.up_timeout_min * 60:
                    raise TimeoutError(f"no READY after {self.up_timeout_min} min")
                time.sleep(self.poll_s)
        except BaseException:
            self.down()
            raise

    def up(self):
        self.start()
        self.ready_line = self.wait_ready()
        return self

    def __enter__(self):
        return self.up()

    def __exit__(self, et, ev, tb):
        self.down()
        return False

    def down(self):
        """Idempotent and thread-safe. Stops the up subprocess if still running, then the VM (3 tries), logs the spend.
        `downed` becomes True only after the VM has really been stopped; concurrent callers wait for the first one."""
        with self._lock:
            if self.downed or not self.started:
                return
            try:
                if self._proc is not None and self._proc.poll() is None:
                    try:
                        os.killpg(self._proc.pid, 15)   # PID of our own child only
                    except OSError:
                        pass
                err = ""
                for _ in range(3):
                    r = self._cli("down", "--name", self.name, timeout=300)
                    if r.returncode == 0:
                        err = ""
                        break
                    err = (r.stdout + r.stderr)[-300:]
                    time.sleep(5)
                self.units_after = self.credits()
                spent = None if None in (self.units_before, self.units_after) else round(self.units_before - self.units_after, 3)
                self._log("down", ok=not err, error=err, credits_after=self.units_after, spent_units=spent,
                          hours=round((time.time() - (self.t_up or time.time())) / 3600, 3))
                if err:
                    warnings.warn(f"runkit: `cj down {self.name}` FAILED: {err}  -> stop it by hand!")
                else:
                    self.downed = True
            finally:
                self._down_attempted = True

    # -------- work on the VM
    def run(self, tag: str, cmd, env: dict | None = None):
        a = ["run", "--name", self.name, "--tag", tag]
        for k, v in (env or {}).items():
            a += ["--env", f"{k}={v}"]
        return self._cli(*a, "--", *([cmd] if isinstance(cmd, str) else cmd), check=True)

    def wait(self, tag: str, max_min: float = 60) -> int:
        return self._cli("wait", "--name", self.name, "--tag", tag, "--max-min", max_min, timeout=max_min * 60 + 300).returncode

    def get(self, remote: str, local_dir: str, timeout=1800):
        self._cli("get", "--name", self.name, remote, local_dir, timeout=timeout, check=True)

    def status(self, dirs=(), jobs_dir="/content/jobs", timeout: float = 90) -> dict:
        """One exec round trip (never `tail`): uptime, GPU use, per-job state, file counts of `dirs`."""
        code = (
            "import subprocess, json, os, glob\n"
            "def sh(c):\n"
            "    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30); return (r.stdout + r.stderr).strip()\n"
            "o = {'uptime': sh('uptime'), 'gpu': sh('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader')}\n"
            f"o['jobs'] = {{os.path.basename(f)[:-4]: ('CJ_EXIT' in open(f).read()[-400:]) for f in glob.glob({jobs_dir!r} + '/*.log')}}\n"
            f"o['counts'] = {{d: len(glob.glob(d + '/**/*', recursive=True)) for d in {list(dirs)!r}}}\n"
            "print('RK_STATUS ' + json.dumps(o))\n")
        out = self.cj.remote_py(self.name, code, timeout=timeout) or ""
        m = re.search(r"RK_STATUS (\{.*\})", out)
        if not m:
            return dict(error="no status", raw=out[-300:])
        s = json.loads(m.group(1))
        g = re.match(r"\s*(\d+)\s*%", s.get("gpu", ""))
        s["gpu_util"] = int(g.group(1)) if g else None
        s["running_jobs"] = [j for j, done in s["jobs"].items() if not done]
        return s

    def fetch_verified(self, remote: str, local_dir: str, expected: int | None = None, pattern: str = "*", tries: int = 2) -> dict:
        """cj get, then count local files matching `pattern` under local_dir/basename(remote). Retries if short."""
        base = Path(local_dir) / os.path.basename(remote.rstrip("/"))
        n = 0
        for i in range(max(1, tries)):
            try:
                self.get(remote, local_dir)
            except BaseException as exc:
                self._log("fetch_failed", error=str(exc)[:200])
            n = len([p for p in base.rglob(pattern) if p.is_file()]) if base.exists() else 0
            if expected is None or n >= expected:
                break
        ok = expected is None or n >= expected
        self._log("fetch", remote=remote, count=n, expected=expected, ok=ok)
        return dict(ok=ok, count=n, expected=expected, dir=str(base))

    def finish(self, remote=None, local_dir=None, expected_files: int | None = None, pattern: str = "*") -> dict:
        """Fetch, count, then down. A short count raises an alert (file + warning) and, if keep_on_mismatch_min > 0, keeps
        re-fetching for that many minutes; after that the VM is taken down anyway."""
        res = dict(ok=True, count=None, expected=expected_files)
        try:
            if remote:
                res = self.fetch_verified(remote, local_dir, expected_files, pattern)
                t0 = time.time()
                while not res["ok"] and time.time() - t0 < self.grace_min * 60:
                    time.sleep(60)
                    res = self.fetch_verified(remote, local_dir, expected_files, pattern, tries=1)
                if not res["ok"]:
                    self.alert(f"{self.name}: fetched {res['count']} of {res['expected']} expected files from {remote}; taking the VM down anyway")
        finally:
            self.down()
        return res

    def alert(self, msg: str):
        self.alerts.append(msg)
        self._log("ALERT", message=msg)
        (self.log_dir / "ALERT.txt").parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "ALERT.txt", "a") as fh:
            fh.write(time.strftime("%H:%M ") + msg + "\n")
        warnings.warn("runkit ALERT: " + msg)

    def cost(self) -> dict:
        h = (time.time() - self.t_up) / 3600 if self.t_up else 0.0
        return dict(hours=round(h, 3), units=round(h * (L4_UNITS_PER_H if self.kind == "gpu" else CPU_UNITS_PER_H), 3),
                    credits_before=self.units_before, credits_now=self.credits())


class Watchdog:
    """Idle-VM watchdog: samples status() every `interval_s`; when no job runs and the GPU is idle for `idle_min` minutes it
    fetches (verified) and takes the VM down. `clock`/`sleep` are injectable for tests."""

    def __init__(self, session: Session, idle_min: float = 20, interval_s: float = 120, remote=None, local_dir=None,
                 expected: int | None = None, pattern: str = "*", gpu_busy_pct: int = 5, clock=time.time, sleep=time.sleep):
        self.s, self.idle_min, self.interval = session, idle_min, interval_s
        self.remote, self.local_dir, self.expected, self.pattern = remote, local_dir, expected, pattern
        self.gpu_busy, self.clock, self.sleep = gpu_busy_pct, clock, sleep
        self.idle_since = None
        self.fired = False
        self._stop = threading.Event()
        self.thread = None

    def idle(self, st: dict) -> bool:
        if "error" in st:
            return False          # unknown is not idle
        busy = bool(st.get("running_jobs")) or (st.get("gpu_util") or 0) >= self.gpu_busy
        return not busy

    def tick(self) -> bool:
        """One sample; returns True if it fired (VM taken down)."""
        if self.s.downed:
            return True
        st = self.s.status(dirs=[self.remote] if self.remote else ())
        now = self.clock()
        if not self.idle(st):
            self.idle_since = None
            return False
        self.idle_since = self.idle_since or now
        if now - self.idle_since >= self.idle_min * 60:
            self.s.alert(f"{self.s.name} idle for {self.idle_min} min: fetching and taking it down")
            self.fired = True
            self.s.finish(self.remote, self.local_dir, self.expected, self.pattern)
            return True
        return False

    def _loop(self):
        while not self._stop.is_set():
            try:
                if self.tick():
                    return
            except BaseException as exc:
                self.s._log("watchdog_error", error=str(exc)[:200])
            self.sleep(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
