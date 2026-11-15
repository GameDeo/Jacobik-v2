"""What this machine can give us right now: cores, memory and GPU memory, net of other users.

Works the same on the workstation, a Colab VM, a Perlmutter login node (user cgroup: 32 CPUs, 30 GB) and a compute
node. Reads cgroup v2 limits where they exist, /proc/meminfo, the load average and nvidia-smi. No third-party imports.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

GB = 1 << 30


def _cgroup_dirs() -> list[Path]:
    """This process's cgroup v2 directory and its ancestors (innermost first)."""
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().splitlines()[0].split(":", 2)[2]
    except Exception:
        return []
    out, p = [], Path("/sys/fs/cgroup") / rel.lstrip("/")
    while True:
        if p.exists():
            out.append(p)
        if p == Path("/sys/fs/cgroup") or p.parent == p:
            break
        p = p.parent
    return out


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except Exception:
        return None


def cgroup_cpu_quota() -> float | None:
    """Smallest cpu.max quota (in CPUs) along the cgroup chain, or None if unlimited."""
    best = None
    for d in _cgroup_dirs():
        v = _read(d / "cpu.max")
        if not v:
            continue
        q, _, per = v.partition(" ")
        if q != "max" and per:
            cpus = int(q) / int(per)
            best = cpus if best is None else min(best, cpus)
    return best


def cgroup_mem_free() -> int | None:
    """Smallest (memory.max - memory.current) along the cgroup chain, in bytes, or None if unlimited."""
    best = None
    for d in _cgroup_dirs():
        mx, cur = _read(d / "memory.max"), _read(d / "memory.current")
        if mx and mx != "max" and cur:
            free = int(mx) - int(cur)
            best = free if best is None else min(best, free)
    return best


def affinity_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # not nproc: GNU nproc honours OMP_NUM_THREADS
    except Exception:
        return os.cpu_count() or 1


def cpu_budget(cap: int | None = None) -> float:
    """CPUs this job may use: affinity, cgroup quota and the caller's cap, whichever is smallest."""
    b = float(affinity_cpus())
    q = cgroup_cpu_quota()
    if q is not None:
        b = min(b, q)
    if cap:
        b = min(b, float(cap))
    return b


def meminfo_available() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def mem_available() -> int:
    """Bytes we can still allocate: MemAvailable, capped by the cgroup's remaining limit."""
    a = meminfo_available()
    c = cgroup_mem_free()
    return min(a, c) if c is not None else a


def load1() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return 0.0


def foreign_cpu_load(own_busy: float) -> float:
    """Runnable work that is not ours, in CPUs. Only meaningful without a per-user cgroup quota (the workstation):
    on a login node the load average counts every user of a 256-thread machine, and our share is the quota."""
    if cgroup_cpu_quota() is not None:
        return 0.0
    return max(0.0, load1() - own_busy)


def proc_rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def children(pid: int) -> list[int]:
    try:
        out = []
        for t in Path(f"/proc/{pid}/task").iterdir():
            txt = _read(t / "children") or ""
            out += [int(x) for x in txt.split()]
        return out
    except Exception:
        return []


def tree_pids(pid: int) -> list[int]:
    seen, stack = [], [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.append(p)
        stack += children(p)
    return seen


def tree_rss(pid: int) -> int:
    return sum(proc_rss(p) for p in tree_pids(pid))


def _float_or_nan(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def gpu_index() -> int:
    """The physical GPU this run uses, for nvidia-smi (which ignores CUDA_VISIBLE_DEVICES): TOOLS_RUN_GPU_INDEX, else
    the first entry of CUDA_VISIBLE_DEVICES when it is a number (one runner per GPU on a 4-GPU node), else 0."""
    v = os.environ.get("TOOLS_RUN_GPU_INDEX") or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(v) if v.isdigit() else 0


def gpu_query(index: int | None = None) -> dict | None:
    """Utilisation and memory of one GPU, plus every compute process on it (pid -> MiB). None without a GPU."""
    if index is None:
        index = gpu_index()
    try:
        r = subprocess.run(["nvidia-smi", f"--id={index}",
                            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        parts = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        name, util, used, total = parts[:4]
        power = _float_or_nan(parts[4]) if len(parts) > 4 else float("nan")
        sm_clock = _float_or_nan(parts[5]) if len(parts) > 5 else float("nan")
        apps = {}
        r2 = subprocess.run(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,used_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        for line in r2.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                try:
                    apps[int(parts[0])] = float(parts[1])
                except ValueError:
                    apps[int(parts[0])] = float("nan")
        return dict(name=name, util=float(util), used_mib=float(used), total_mib=float(total), power_w=power,
                    sm_clock_mhz=sm_clock, apps=apps, t=time.time())
    except Exception:
        return None


def snapshot(own_root: int | None = None) -> dict:
    """One line of machine state, for the plan printout and the telemetry trace."""
    g = gpu_query()
    own = tree_pids(own_root) if own_root else []
    return dict(t=time.time(), load1=load1(), cpu_budget=cpu_budget(), mem_available_gb=mem_available() / GB,
                own_rss_gb=(sum(proc_rss(p) for p in own) / GB) if own else None,
                gpu=None if g is None else dict(name=g["name"], util=g["util"], used_mib=g["used_mib"],
                                                total_mib=g["total_mib"],
                                                own_mib=sum(v for p, v in g["apps"].items() if p in own),
                                                foreign_pids=[p for p in g["apps"] if p not in own]))
