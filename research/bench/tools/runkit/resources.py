"""Local resources: one environment block, CPU quota, process registry, kill-by-PID, GPU lock, work directories, manifest.

Enforcement (owner decision 2026-10-28): the GPU lock, kill-by-PID and the Colab down-on-exit are hard (they raise
`Refused`); the CPU quota is warn-only until `budget.json` sets `enforce.cpu` to true.
"""
from __future__ import annotations

import fcntl
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

try:  # psutil is in the pixi env; the fallbacks keep import working without it
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class Refused(RuntimeError):
    """A rule that is enforced hard was about to be broken (GPU lock, kill by PID, work in RAM-backed /tmp)."""


def _main() -> Path:
    e = os.environ.get("RUNKIT_MAIN")
    if e:
        return Path(e)
    here = Path(__file__).resolve()
    for p in here.parents:                       # <main>/.claude/worktrees/<wt>/research/bench/tools/runkit/x.py
        if p.name == ".claude":
            return p.parent
    return here.parents[4]                       # plain checkout, or /content/jk on a VM


MAIN = _main()
RUN_DIR = Path(os.environ.get("RUNKIT_DIR", MAIN / "research/data/runkit"))
BUDGET_FILE = RUN_DIR / "budget.json"
PIXI_BIN = MAIN / ".pixi/envs/linux/bin"
RAM_PREFIXES = ("/tmp", "/dev/shm", "/run/shm", "/var/tmp")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

DEFAULT_BUDGET = {
    "schema": 1,
    "cpu": {"Gemini": 5, "A4": 4, "Codex": 3, "runkit": 4},
    "gpu_owner": "Gemini",
    "colab": {"cap_units": 180.0, "max_vms": 4},
    "enforce": {"cpu": False, "gpu": True, "kill": True, "vm_down": True},
}


def is_remote() -> bool:
    """True on a Colab VM (the quota file and the local GPU rule do not apply there)."""
    return bool(os.environ.get("RUNKIT_REMOTE")) or str(MAIN).startswith("/content")


def load_budget(path: Path | None = None) -> dict:
    p = Path(path) if path else BUDGET_FILE
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        warnings.warn(f"runkit: {p} unreadable, using the built-in split {DEFAULT_BUDGET['cpu']}")
        return json.loads(json.dumps(DEFAULT_BUDGET))


def env(threads: int = 1, cuda: bool | None = None, xtb: str | None = "auto", extra: dict | None = None,
        base: dict | None = None) -> dict:
    """The single environment block for child processes.

    threads: OMP/MKL/OPENBLAS thread count (1 keeps the embed and the reductions deterministic).
    cuda: False sets CUDA_VISIBLE_DEVICES="" ; True leaves it alone; None leaves it alone.
    xtb: "auto" resolves JACOBIK_XTB (existing value, else the pixi env, else PATH); a path sets it; None skips.
    PATH gets the pixi env bin first (xtb must resolve, see AGENTS.md); PYTHONPATH defaults to src:.:research/bench."""
    e = dict(os.environ if base is None else base)
    pb = str(PIXI_BIN)
    if PIXI_BIN.exists() and pb not in e.get("PATH", "").split(":")[:1]:
        e["PATH"] = pb + ":" + e.get("PATH", "")
    for v in THREAD_VARS:
        e[v] = str(int(threads))
    e.setdefault("OMP_STACKSIZE", "500M")
    e.setdefault("PYTHONPATH", "src:.:research/bench")
    if cuda is False:
        e["CUDA_VISIBLE_DEVICES"] = ""
    if xtb == "auto":
        cand = e.get("JACOBIK_XTB") or shutil.which("xtb", path=e.get("PATH"))
        if cand:
            e["JACOBIK_XTB"] = os.path.abspath(cand)
    elif xtb:
        e["JACOBIK_XTB"] = str(xtb)
    e.update({k: str(v) for k, v in (extra or {}).items()})
    return e


# ---------------------------------------------------------------- registry and quota

def _locked_update(path: Path, fn):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            cur = json.loads(path.read_text()) if path.exists() else []
        except ValueError:
            cur = []
        new = fn(cur)
        tmp = str(path) + ".tmp"
        Path(tmp).write_text(json.dumps(new, indent=1))
        os.replace(tmp, path)
        return new


def _ctime(pid: int):
    if psutil is None:
        return None
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _alive(pid: int, ctime=None) -> bool:
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                return False
            return ctime is None or abs(p.create_time() - ctime) < 1.0
        except psutil.Error:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Budget:
    """An agent's CPU share and process registry. `Budget("A4")` takes its share from budget.json; `cpus=` overrides it."""

    def __init__(self, agent: str, cpus: int | None = None, budget_file: Path | None = None, run_dir: Path | None = None):
        self.agent = agent
        self.cfg = load_budget(budget_file)
        self.run_dir = Path(run_dir) if run_dir else RUN_DIR
        share = self.cfg.get("cpu", {}).get(agent)
        if share is None and cpus is None:
            warnings.warn(f"runkit: agent {agent!r} has no CPU share in budget.json; assuming 1")
        self.quota = int(cpus if cpus is not None else (share if share is not None else 1))
        self.registry_path = self.run_dir / "registry" / f"{agent}.json"

    @property
    def enforce(self) -> dict:
        return self.cfg.get("enforce", DEFAULT_BUDGET["enforce"])

    @property
    def gpu_owner(self) -> str | None:
        return self.cfg.get("gpu_owner")

    def entries(self) -> list:
        try:
            return json.loads(self.registry_path.read_text())
        except (OSError, ValueError):
            return []

    def running(self) -> list:
        return [r for r in self.entries() if not r.get("ended") and _alive(r["pid"], r.get("ctime"))]

    def in_use(self) -> int:
        return sum(int(r.get("cpus", 1)) for r in self.running())

    def lookup(self, pid: int):
        for r in self.entries():
            if r["pid"] == pid:
                return r
        return None

    def register(self, rec: dict):
        _locked_update(self.registry_path, lambda cur: [r for r in cur if r["pid"] != rec["pid"]] + [rec])

    def mark_ended(self, pid: int):
        def f(cur):
            for r in cur:
                if r["pid"] == pid:
                    r["ended"] = time.time()
            return cur
        _locked_update(self.registry_path, f)


class Proc:
    """A spawned, registered child. Wraps Popen; `kill()` goes through kill_tree (registered PIDs only)."""

    def __init__(self, popen, budget: Budget, tag: str):
        self.popen, self.budget, self.tag, self.pid = popen, budget, tag, popen.pid

    def poll(self):
        rc = self.popen.poll()
        if rc is not None:
            self.budget.mark_ended(self.pid)
        return rc

    def wait(self, timeout=None):
        rc = self.popen.wait(timeout=timeout)
        self.budget.mark_ended(self.pid)
        return rc

    def kill(self, grace: float = 10.0):
        return kill_tree(self.pid, self.budget, grace=grace)


def spawn(cmd, budget: Budget, tag: str, log=None, env: dict | None = None, cpus: int = 1, gpu: bool = False,
          cwd=None) -> Proc:
    """Start a child in its own session, count it against the agent's quota and record it in the registry.

    Quota: warn-only unless budget.json enforce.cpu is true (then Refused). GPU: a child only sees the GPU if gpu=True,
    and gpu=True is refused for any agent but budget.json gpu_owner (hard), and unless the caller holds `gpu_lock`."""
    cmd = [str(c) for c in cmd]
    e = dict(env) if env is not None else globals()["env"](threads=1, cuda=not gpu)
    if gpu:
        if not is_remote() and budget.agent != budget.gpu_owner and budget.enforce.get("gpu", True):
            raise Refused(f"{budget.agent} may not start GPU jobs here: the local GPU belongs to {budget.gpu_owner}")
        if not _holds_gpu_lock():
            raise Refused("gpu=True needs `with gpu_lock(agent):` around the spawn")
    else:
        e["CUDA_VISIBLE_DEVICES"] = ""
    want = budget.in_use() + int(cpus)
    if want > budget.quota:
        msg = f"{budget.agent} would use {want} CPU processes, quota {budget.quota} (budget.json)"
        if budget.enforce.get("cpu", False):
            raise Refused(msg)
        warnings.warn("runkit (warn-only): " + msg)
    out = None
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        out = open(log, "ab")
    p = subprocess.Popen(cmd, stdout=out or subprocess.DEVNULL, stderr=subprocess.STDOUT, env=e, cwd=cwd,
                         start_new_session=True)
    if out:
        out.close()
    budget.register(dict(pid=p.pid, ctime=_ctime(p.pid), tag=tag, cmd=cmd, cpus=int(cpus), gpu=bool(gpu),
                         log=str(log) if log else None, started=time.time(), host=socket.gethostname()))
    return Proc(p, budget, tag)


def kill_tree(pid: int, budget: Budget, grace: float = 10.0) -> list:
    """Terminate a registered process and its children by PID (children first), kill after `grace` seconds.

    Refuses (Refused) any PID that is not in this agent's registry or that has been reused by another process;
    there is deliberately no way to kill by name or pattern."""
    rec = budget.lookup(int(pid))
    if rec is None:
        raise Refused(f"pid {pid} is not in the {budget.agent} registry; runkit kills only processes it spawned")
    if not _alive(pid, rec.get("ctime")):
        if psutil is not None and _alive(pid):
            raise Refused(f"pid {pid} was reused by another process; refusing")
        budget.mark_ended(pid)
        return []
    victims = []
    if psutil is not None:
        root = psutil.Process(pid)
        victims = root.children(recursive=True)[::-1] + [root]
        for v in victims:
            try:
                v.terminate()
            except psutil.Error:
                pass
        gone, alive = psutil.wait_procs(victims, timeout=grace)
        for v in alive:
            try:
                v.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=5)
    else:  # pragma: no cover
        os.killpg(pid, signal.SIGTERM)
        time.sleep(min(grace, 1.0))
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
    budget.mark_ended(pid)
    return [v.pid for v in victims]


# ---------------------------------------------------------------- GPU lock

_HELD: set = set()


def _holds_gpu_lock() -> bool:
    return bool(_HELD)


def _smi_compute_pids() -> list:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()] if r.returncode == 0 else []
    except Exception:
        return []


class gpu_lock:
    """Exclusive local GPU lock (flock on RUN_DIR/gpu.lock) for the agent that owns the GPU.

    Hard refusals: another holder (after `timeout` seconds), an agent that is not budget.json gpu_owner, or a foreign compute
    process shown by nvidia-smi. On a Colab VM the owner rule is skipped (the VM is yours)."""

    def __init__(self, agent: str, timeout: float = 0.0, check_smi: bool = True, run_dir: Path | None = None,
                 budget_file: Path | None = None):
        self.agent, self.timeout, self.check_smi = agent, timeout, check_smi
        self.path = (Path(run_dir) if run_dir else RUN_DIR) / "gpu.lock"
        self.cfg = load_budget(budget_file)
        self.fh = None

    def __enter__(self):
        hard = self.cfg.get("enforce", {}).get("gpu", True)
        if hard and not is_remote() and self.agent != self.cfg.get("gpu_owner"):
            raise Refused(f"{self.agent} may not take the local GPU; it belongs to {self.cfg.get('gpu_owner')}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        t0 = time.time()
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() - t0 >= self.timeout:
                    holder = ""
                    try:
                        self.fh.seek(0)
                        holder = self.fh.read()[-300:]
                    except OSError:
                        pass
                    self.fh.close()
                    self.fh = None
                    raise Refused("the GPU lock is held by another job: " + holder)
                time.sleep(0.5)
        if self.check_smi and hard:
            foreign = [p for p in _smi_compute_pids() if p != os.getpid()]
            if foreign:
                self.__exit__()
                raise Refused(f"nvidia-smi shows other compute processes {foreign}; one GPU job at a time")
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(json.dumps(dict(agent=self.agent, pid=os.getpid(), since=time.time())))
        self.fh.flush()
        _HELD.add(id(self))
        return self

    def __exit__(self, *exc):
        _HELD.discard(id(self))
        if self.fh:
            try:
                self.fh.seek(0)
                self.fh.truncate()
                fcntl.flock(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()
                self.fh = None
        return False


# ---------------------------------------------------------------- work directory and manifest

def workroot(out, allow_ram: bool = False) -> str:
    """Scratch directory for xtb (<out>/_xtbtmp). /tmp and /dev/shm are RAM-backed here: refused unless allow_ram."""
    p = (Path(out) / "_xtbtmp")
    rp = str(p.resolve())
    if not allow_ram and any(rp == q or rp.startswith(q + "/") for q in RAM_PREFIXES):
        raise Refused(f"{rp} is RAM-backed; give an output directory on disk (or allow_ram=True for tiny jobs)")
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def manifest(extra: dict | None = None, torch_info: bool = True) -> dict:
    """Provenance record for a run: host, threads, affinity, platform (GPU), versions, xtb, git commit, profile fingerprint."""
    m = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname(), python=sys.version.split()[0],
             system=platform.platform(), cores=os.cpu_count(),
             affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
             env={k: os.environ.get(k) for k in (*THREAD_VARS, "JACOBIK_NUM_WORKERS", "CUDA_VISIBLE_DEVICES",
                                                  "JACOBIK_XTB")},
             git_commit=_git("rev-parse", "HEAD"), git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"))
    xb = os.environ.get("JACOBIK_XTB") or shutil.which("xtb")
    m["xtb_path"] = os.path.abspath(xb) if xb else None
    if "torch" in sys.modules or torch_info:
        try:
            import torch
            m["torch"] = torch.__version__
            m["torch_threads"] = torch.get_num_threads()
            m["platform"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() and
                             os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
        except Exception:
            m["platform"] = "cpu"
    try:
        from jacobik.core import profile as P
        m["fingerprint"] = P.snapshot()["fingerprint"]
    except Exception:
        m["fingerprint"] = None
    m.update(extra or {})
    return m
