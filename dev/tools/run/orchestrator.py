"""Resource-aware parent scheduler: queues of whole parents (one queue per job), longest first, fed to persistent lanes.

One scheduler owns the GPU. It serves one job (`python -m tools.run panel`) or, as the broker
(`python -m tools.run broker`), every job that agents drop into its spool directory; jobs interleave at parent
granularity. A lane is a persistent process bound to one job (one repo checkout, one import-time profile, one DB), so
profile flags read at import are isolated per process, and the job's parents run in it exactly as the job's own
sequential runner runs them (same calls, same order inside a parent).

Why process lanes, not threads in one process: the runners set module globals per arm (finish flags, replay state,
warning counters, the validity ledger's current parent), so two parents in one interpreter would race on them; and the
launch overhead that bounds a solve is Python work under the GIL. Each lane costs about 2.5 GB RSS (torch + MACE), so
the lane count is bounded by RAM before it is bounded by the GPU.

Admission (rechecked every poll): extra lanes start only after a first parent has finished (giants go first, so the
measured peak RSS and GPU memory are those of the largest parent), one at a time as each new lane finishes loading, free CPUs cover it (cgroup quota or affinity, minus other users' load where there
is no quota), MemAvailable (capped by the cgroup limit) minus a headroom covers the measured peak RSS and the optional
--max-rss-gb cap holds, free GPU memory covers the measured GPU peak, and the GPU is not already >= 90 % busy. Under
memory pressure the newest lane is retired after its parent; under critical pressure it is killed and its parent
requeued once.

No parent stalls a run: each has a wall budget (3 x its estimate, clamped to [--min-budget, --parent-cap], default cap
5 min); an overrun kills that lane, records status "timeout" for the parent and the run moves on. Lanes send a
heartbeat every 60 s (parent, stage, elapsed), logged in run_events.jsonl. Giants go first.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import resources as R
from ..repo import main_checkout as _main_checkout
from .telemetry import Sampler, one_line

REPO = Path(__file__).resolve().parents[3]
LANE_RSS_DEFAULT = 2.8 * R.GB      # torch + MACE + a mid-size parent, measured 2.5-2.9 GB; replaced by measurements
LANE_GPU_DEFAULT_MIB = 2048.0
INIT_FAIL_LIMIT = 6                # consecutive lane-init failures (with no live lane) before a job's queue is given up


def main_checkout(repo: Path = REPO) -> Path:
    return _main_checkout(repo)


def lane_env(plan: dict) -> dict:
    """The job's checkout first (its jacobik, its tools.jgen), this runner's tools.run last; one numerical thread."""
    repo = Path(plan.get("repo") or REPO)
    env = {k: v for k, v in os.environ.items() if k != "JACOBIK_PROFILE_OVERRIDES"}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(repo / "src"), str(repo / "dev"), str(repo), str(repo / "research" / "bench"), str(REPO / "dev"), str(REPO)]),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if not (repo / "data").exists() and "JACOBIK_DATA_ROOT" not in env:
        env["JACOBIK_DATA_ROOT"] = str(main_checkout(repo) / "data")
    tmp = Path(plan["out"]) / "_lanes" / "tmp"  # xtb scratch (tempfile): never the RAM-backed /tmp
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp)
    if plan.get("device") == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif env.get("TOOLS_RUN_MPS_CLIENT") == "1":
        # The private daemon already selected the physical GPU and remaps it to device 0.
        # Reapplying its physical mask to the client hides that remapped device on g1-g3.
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update({k: str(v) for k, v in (plan.get("env") or {}).items()})
    return env


class Lane:
    def __init__(self, idx: int, job: "Job", events: queue.Queue):
        self.idx, self.job, self.events = idx, job, events
        self.log_path = job.out / "_lanes" / f"l{idx}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a")
        repo = Path(job.plan.get("repo") or REPO)
        self.p = subprocess.Popen([sys.executable, "-u", "-m", "tools.run.lane", "--plan", str(job.plan_path),
                                   "--lane", str(idx)], cwd=str(repo), env=lane_env(job.plan), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                  start_new_session=True)
        self.pid = self.p.pid
        self.state = "starting"   # starting -> idle <-> busy -> stopped | dead
        self.code = None
        self.t_task = None
        self.budget = None
        self.ready: dict = {}
        self.done = 0
        self.peak_rss = 0
        self.gpu_peak_reserved = 0
        self.gpu_peak_allocated = 0
        self.retiring = False
        self.timed_out = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.p.stdout:
            try:
                self.events.put((self.idx, json.loads(line)))
            except ValueError:
                pass
        self.events.put((self.idx, {"eof": True, "rc": self.p.wait()}))

    def send(self, msg: dict) -> bool:
        try:
            self.p.stdin.write(json.dumps(msg) + "\n")
            self.p.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def kill(self):
        """Our own child and its process group (its xtb children): by PID, never by name."""
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except OSError:
            try:
                self.p.kill()
            except OSError:
                pass

    @property
    def live(self):
        return self.state not in ("stopped", "dead")


class Job:
    def __init__(self, plan: dict, resume: bool = True):
        self.out = Path(plan["out"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.arm_names = [a["arm"] for a in plan["arms"]]
        self.arm_dir = {a["arm"]: Path(a.get("out") or self.out / a["arm"]) for a in plan["arms"]}
        self.seeds = [int(s) for s in plan["seeds"]]
        # work items: a whole parent ("CODE"), or one seed of it ("CODE@11") when plan["split_seeds"] is set (only
        # for runners whose seeds are independent solves, e.g. the isomer screen; the bench shares one wide solve)
        self.split = bool(plan.get("split_seeds")) and len(self.seeds) > 1
        pc = {p["code"]: p.get("cost") for p in plan["parents"]}
        items = []
        self.cost = {}
        for p in plan["parents"]:
            for it in ([f"{p['code']}@{s}" for s in self.seeds] if self.split else [p["code"]]):
                items.append(it)
                self.cost[it] = (pc[p["code"]] / len(self.seeds) if self.split else pc[p["code"]]) if pc[p["code"]] else None
        self.packed = set()
        self.packed_status = {}
        self.packed_walls = {}
        self.pack_root = Path(plan.get("pack_root") or self.out)
        def pack_number(path):
            suffix = path.stem.removeprefix("run_pack")
            return int(suffix) if suffix.isdigit() else -1
        for tf in sorted(self.pack_root.glob("run_pack*.tar"), key=pack_number):  # resume across packed runs
            import tarfile
            try:
                with tarfile.open(tf) as t:
                    self.packed |= set(t.getnames())
                    for member in t.getmembers():
                        if "/records/" in member.name and member.name.endswith(".json") and member.isfile():
                            try:
                                record = json.load(t.extractfile(member))
                                self.packed_status[member.name] = record.get("status")
                                self.packed_walls[member.name] = (record.get("run_timing") or {}).get("parent_wall_s") or record.get("elapsed_s")
                            except (ValueError, OSError):
                                pass
            except Exception:
                pass
        for item in items:
            code, _, seed = item.partition("@")
            wall_dirs = [Path(a.get("out") or self.out / a["arm"]) for a in plan.get("wall_arms", plan["arms"])]
            for arm_dir in wall_dirs:
                for s in ([int(seed)] if seed else self.seeds):
                    path = arm_dir / "records" / f"{code}_{s}.json"
                    try:
                        if path.exists():
                            record = json.loads(path.read_text())
                            wall = (record.get("run_timing") or {}).get("parent_wall_s") or record.get("elapsed_s")
                        else:
                            wall = self.packed_walls.get(str(path.relative_to(self.pack_root)))
                        if isinstance(wall, (int, float)) and wall > 0:
                            self.cost[item] = max(self.cost[item] or 0, wall)
                    except (OSError, ValueError):
                        pass
        known = sorted(v for v in self.cost.values() if v)
        self.median = __import__("statistics").median(known) if known else None
        self.retry_timeouts = bool(plan.get("retry_timeouts"))
        self.prior_timeouts = {c for c in items if self.has_timeout(c)}
        self.todo = [c for c in items if not (resume and self.complete(c))]
        self.skipped_timeouts = [c for c in items if resume and c not in self.todo and c in self.prior_timeouts]
        self.skipped = len(items) - len(self.todo)
        self.todo.sort(key=lambda c: (-(self.cost.get(c) or self.median or 0.0), c))  # giants first
        self.plan = dict(plan, run_started=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.plan_path = self.out / "run_plan.json"
        self.plan_path.write_text(json.dumps(self.plan, indent=1))
        self.name = plan.get("name") or self.out.name
        self.results: list[dict] = []
        self.requeued: set = set()
        self.not_run: list = []
        self.init_failures = 0        # consecutive lane-init failures (reset when a lane becomes ready)
        self.init_block_until = 0.0   # back-off: no new lane for this job before this time
        # Shared queue across runners (one per GPU): a work item is claimed by an atomic file create, so every runner
        # gets the full list and takes what it can; a fast GPU simply takes more. Claims live in claim_dir.
        self.claim_dir = Path(plan["claim_dir"]) if plan.get("claim_dir") else None
        self.mine: set = set()
        if self.claim_dir:
            self.claim_dir.mkdir(parents=True, exist_ok=True)

    def claim(self, it) -> bool:
        if self.claim_dir is None or it in self.mine:
            return True
        try:
            os.close(os.open(self.claim_dir / it.replace("@", "_at_"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            return False
        self.mine.add(it)
        return True

    def prune(self):
        """Drop queued items another runner has already claimed (so the queue length is the work still ours to take)."""
        if self.claim_dir is not None:
            self.todo[:] = [c for c in self.todo if c in self.mine or not (self.claim_dir / c.replace("@", "_at_")).exists()]

    def unclaim(self, it):
        if self.claim_dir is not None and it in self.mine:
            self.mine.discard(it)
            try:
                (self.claim_dir / it.replace("@", "_at_")).unlink()
            except OSError:
                pass

    def item(self, it):
        code, _, seed = it.partition("@")
        return code, ([int(seed)] if seed else self.seeds)

    def complete(self, it):
        code, seeds = self.item(it)
        def have(a, s):
            f = self.arm_dir[a] / "records" / f"{code}_{s}.json"
            if f.exists():
                if self.retry_timeouts:
                    try:
                        if json.loads(f.read_text()).get("status") == "timeout":
                            return False
                    except (OSError, ValueError):
                        pass
                return True
            try:
                name = str(f.relative_to(self.pack_root))
                return name in self.packed and not (self.retry_timeouts and self.packed_status.get(name) == "timeout")
            except ValueError:
                return False
        return all([have(a, s) for a in self.arm_names for s in seeds])

    def remaining_cost(self):
        return sum(self.cost.get(c) or self.median or 60.0 for c in self.todo)

    def has_timeout(self, it):
        code, seeds = self.item(it)
        for arm in self.arm_names:
            for seed in seeds:
                path = self.arm_dir[arm] / "records" / f"{code}_{seed}.json"
                try:
                    status = json.loads(path.read_text()).get("status") if path.exists() else self.packed_status.get(str(path.relative_to(self.pack_root)))
                except (OSError, ValueError):
                    status = None
                if status == "timeout":
                    return True
        return False

    def budget_for(self, it, lo, cap):
        est = self.cost.get(it)
        return cap if not est else min(cap, max(lo, 3.0 * est))

    def write_status(self, it, status, reason, elapsed_s=None):
        code, seeds = self.item(it)
        for a in self.arm_names:
            d = self.arm_dir[a] / "records"
            d.mkdir(parents=True, exist_ok=True)
            for s in seeds:
                f = d / f"{code}_{s}.json"
                replace_timeout = False
                if f.exists() and self.retry_timeouts:
                    try:
                        replace_timeout = json.loads(f.read_text()).get("status") == "timeout"
                    except (OSError, ValueError):
                        pass
                if not f.exists() or replace_timeout:
                    f.write_text(json.dumps(dict(code=code, seed=s, arm=a, status=status, reason=reason, elapsed_s=elapsed_s, n_rows=0,
                                                 n_finite=0, top1_dE=None, floor_dE=None)))


def estimate(jobs: list[Job], lanes: int) -> dict:
    """Wall estimate from recorded per-parent seconds: max(longest parent, total / lanes). Recorded walls come from the
    hardware that recorded them (C-2b and isomer records: the 2060), so this is a planning number, not a promise."""
    costs = [j.cost.get(c) or j.median or 60.0 for j in jobs for c in j.todo]
    unknown = sum(1 for j in jobs for c in j.todo if not j.cost.get(c))
    tot = sum(costs)
    wall = max(max(costs, default=0.0), tot / max(1, lanes))
    return dict(parents=len(costs), unknown_cost=unknown, serial_s=round(tot), lanes=lanes, wall_s=round(wall),
                longest_s=round(max(costs, default=0.0)))


def predicted_lanes(device: str, cpu_cap=None, cpu_per_lane=1.5, mem_headroom_gb=2.0, max_rss_gb=None,
                    max_lanes=None) -> int:
    n_cpu = int((R.cpu_budget(cpu_cap) - R.foreign_cpu_load(0.0) - 0.5) // cpu_per_lane)
    mem = R.mem_available() - mem_headroom_gb * R.GB
    if max_rss_gb is not None:
        mem = min(mem, max_rss_gb * R.GB)
    n_mem = int(mem // LANE_RSS_DEFAULT)
    n = min(n_cpu, n_mem)
    if device == "cuda":
        g = R.gpu_query()
        if g is not None:
            n = min(n, int((g["total_mib"] - g["used_mib"] - max(1024, 0.1 * g["total_mib"])) // LANE_GPU_DEFAULT_MIB))
    if max_lanes:
        n = min(n, max_lanes)
    return max(1, n)


class Scheduler:
    def __init__(self, out: Path, device: str = "cuda", *, max_lanes=None, lanes_fixed=None, cpu_cap=None,
                 cpu_per_lane=1.5, mem_headroom_gb=2.0, max_rss_gb=None, gpu_headroom_mib=None, max_wall_s=None,
                 kill_grace_s=60.0, parent_cap_s=900.0, min_budget_s=180.0, poll_s=2.0, verbose=True):
        if lanes_fixed is not None and lanes_fixed < 1:
            raise ValueError("--lanes must be >= 1")
        if max_lanes is not None and max_lanes < 1:
            raise ValueError("--max-lanes must be >= 1")
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_lanes, self.lanes_fixed, self.cpu_cap, self.cpu_per_lane = max_lanes, lanes_fixed, cpu_cap, cpu_per_lane
        self.mem_headroom_gb, self.max_rss_gb, self.max_wall_s, self.kill_grace_s = mem_headroom_gb, max_rss_gb, max_wall_s, kill_grace_s
        self.parent_cap_s, self.min_budget_s, self.poll_s, self.verbose = parent_cap_s, min_budget_s, poll_s, verbose
        self.events: queue.Queue = queue.Queue()
        self.lanes: dict[int, Lane] = {}
        self.jobs: list[Job] = []
        self.lane_gpu_seen: dict[int, float] = {}
        self.gutil: list = []
        self.t0 = time.time()
        g = R.gpu_query() if device == "cuda" else None
        self.gpu_headroom_mib = gpu_headroom_mib if gpu_headroom_mib is not None else (
            max(1024.0, 0.10 * g["total_mib"]) if g else 0.0)
        self.last_retire = 0.0
        self.last_spawn_t = 0.0
        self.last_reason = None
        self.stop_dispatch = False

    # ------------------------------------------------------------------ bookkeeping
    def event(self, kind, **kw):
        row = dict(t=round(time.time() - self.t0, 2), kind=kind, **kw)
        with (self.out / "run_events.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        if self.verbose:
            print(json.dumps(row, default=str), flush=True)

    def live(self):
        return [l for l in self.lanes.values() if l.live]

    def busy(self):
        return [l for l in self.lanes.values() if l.state in ("busy", "starting")]

    def todo_total(self):
        return sum(len(j.todo) for j in self.jobs)

    def add_job(self, job: Job):
        self.jobs.append(job)
        self.event("job_added", job=job.name, out=str(job.out), parents=len(job.todo), skipped_resume=job.skipped,
                   arms=job.arm_names, seeds=job.seeds, backend=job.plan.get("backend", "bench"))

    def rss_est(self):
        vals = [max(l.peak_rss, l.ready.get("rss", 0)) for l in self.lanes.values() if l.done]
        return 1.15 * max(vals) if vals else None

    def gpu_est(self):
        """Per-lane GPU need: peak allocated tensors + CUDA context and MACE model (lane's baseline at ready), x 1.2.
        Not nvidia-smi's per-process figure: that includes the caching allocator's slack, which on a 40 GB card reaches
        the runners' 80 % cap (32 GB) for one lane and would forbid a second lane that measurably helps."""
        vals = [l.gpu_peak_allocated / 2**20 + 800.0 for l in self.lanes.values() if l.done]
        return 1.2 * max(vals) if vals else None

    # ------------------------------------------------------------------ admission
    def can_add(self):
        n = len(self.live())
        if self.lanes_fixed is not None:
            return n < self.lanes_fixed, f"fixed lane count {self.lanes_fixed}"
        if self.max_lanes is not None and n >= self.max_lanes:
            return False, f"max lanes {self.max_lanes}"
        if n == 0:
            return True, "first lane"
        if not any(l.done for l in self.lanes.values()):
            # A known outlier should not prevent the rest of the queue from starting.
            # The usual conservative default resource estimates still apply below.
            long_running = any(l.state == "busy" and (l.code in l.job.prior_timeouts or
                               (l.job.median and (l.job.cost.get(l.code) or 0) > 2 * l.job.median))
                               for l in self.live())
            if not long_running:
                return False, "waiting for the first parent (per-lane peak memory not measured yet)"
        if any(l.state == "starting" for l in self.live()):
            return False, "waiting for the newest lane to load"
        if time.time() - self.last_spawn_t < 10:
            return False, "ramp: one lane per 10 s"
        if self.todo_total() <= sum(1 for l in self.live() if l.state == "idle"):
            return False, "enough lanes for the queue"
        budget = R.cpu_budget(self.cpu_cap)
        own = self.cpu_per_lane * n
        free_cpu = budget - R.foreign_cpu_load(own) - own - 0.5
        if free_cpu < self.cpu_per_lane:
            return False, f"CPU: {free_cpu:.1f} free of {budget:.0f}"
        est = self.rss_est() or LANE_RSS_DEFAULT
        avail = R.mem_available()
        if avail - self.mem_headroom_gb * R.GB < est:
            return False, f"RAM: {avail / R.GB:.1f} GB available, lane peak {est / R.GB:.1f} GB + headroom {self.mem_headroom_gb} GB"
        if self.max_rss_gb is not None and R.tree_rss(os.getpid()) + est > self.max_rss_gb * R.GB:
            return False, f"own RSS cap {self.max_rss_gb} GB"
        if self.device == "cuda":
            g = R.gpu_query()
            if g is None:
                return False, "nvidia-smi failed"
            gest = self.gpu_est() or LANE_GPU_DEFAULT_MIB
            own = {l.pid for l in self.live()}
            foreign = g["used_mib"] - sum(v for p, v in g["apps"].items() if p in own)
            free = g["total_mib"] - foreign - n * gest  # our lanes counted at their measured need, not their cache
            if free - self.gpu_headroom_mib < gest:
                return False, f"GPU memory: {free:.0f} MiB free after {n} lanes x {gest:.0f} MiB and others' {foreign:.0f} MiB"
            recent = [u for t, u in self.gutil if t > time.time() - 30]
            if len(recent) >= 5 and sum(recent) / len(recent) >= 90:
                return False, "GPU already >= 90 % busy"
        return True, "resources available"

    def job_for_new_lane(self):
        """The job with the most queued work per live lane (fair interleave across agents' jobs)."""
        cand = [j for j in self.jobs if j.todo and time.time() >= j.init_block_until]
        if not cand:
            return None
        return max(cand, key=lambda j: j.remaining_cost() / (1 + sum(1 for l in self.live() if l.job is j)))

    def spawn(self, job: Job):
        idx = max(self.lanes) + 1 if self.lanes else 0
        self.lanes[idx] = Lane(idx, job, self.events)
        self.last_spawn_t = time.time()
        self.event("lane_spawn", lane=idx, pid=self.lanes[idx].pid, job=job.name,
                   rss_est_gb=round((self.rss_est() or LANE_RSS_DEFAULT) / R.GB, 2), gpu_est_mib=self.gpu_est())

    def dispatch(self, l: Lane):
        j = l.job
        if self.stop_dispatch or not j.todo or l.retiring:
            return
        code = None
        while j.todo:
            cand = j.todo.pop(0)
            if j.claim(cand):
                code = cand
                break
        if code is None:
            return
        c, seeds = j.item(code)
        if l.send({"code": c, "seeds": seeds, "item": code}):
            l.state, l.code, l.t_task = "busy", code, time.time()
            l.budget = j.budget_for(code, self.min_budget_s, self.parent_cap_s)
            l.timed_out = False
        else:
            j.unclaim(code)
            j.todo.insert(0, code)

    # ------------------------------------------------------------------ messages
    def handle(self, idx, msg):
        l = self.lanes[idx]
        j = l.job
        if msg.get("eof"):
            was = l.code if l.state == "busy" else None
            l.state = "stopped" if l.state == "stopped" else "dead"
            self.event("lane_exit", lane=idx, rc=msg.get("rc"), in_flight=was, timed_out=l.timed_out)
            if was and not l.timed_out:
                if was in j.requeued or self.stop_dispatch:
                    j.write_status(was, "failed", f"lane exited (rc {msg.get('rc')}); see {l.log_path}")
                    j.results.append(dict(code=was, ok=False, status="failed", error="lane exited", lane=idx))
                else:
                    j.requeued.add(was)
                    j.todo.insert(0, was)
            l.code = None
            return
        if msg.get("hb"):
            self.event("heartbeat", lane=idx, job=j.name, code=msg.get("code"), stage=msg.get("stage"),
                       elapsed_s=msg.get("elapsed_s"), rss_gb=msg.get("rss_gb"))
            return
        if "ready" in msg:
            if msg["ready"]:
                l.state, l.ready = "idle", msg
                j.init_failures = 0
                self.event("lane_ready", lane=idx, job=j.name, startup_s=round(msg["startup_s"], 1),
                           rss_gb=round(msg["rss"] / R.GB, 2))
            else:
                # A lane that cannot start (typically CUDA OOM while other lanes hold the GPU) says nothing about the
                # parents: back off and try again. The queue is given up only when no lane of this job is alive and
                # starting keeps failing; those parents are NOT recorded as failed (not_run, so a resume picks them up).
                j.init_failures += 1
                j.init_block_until = time.time() + min(120.0, 15.0 * j.init_failures)
                self.event("lane_init_failed", lane=idx, job=j.name, error=msg.get("error"), consecutive=j.init_failures)
                l.send({"stop": True})
                l.state = "dead"
                if j.init_failures >= INIT_FAIL_LIMIT and not any(x.job is j for x in self.live()):
                    self.event("job_given_up", job=j.name, not_run=len(j.todo), error=msg.get("error"))
                    j.not_run.extend(j.todo)
                    j.todo.clear()
            return
        if "code" in msg:
            l.done += 1
            l.peak_rss = max(l.peak_rss, msg.get("peak_rss") or 0)
            l.gpu_peak_reserved = max(l.gpu_peak_reserved, msg.get("gpu_peak_reserved") or 0)
            l.gpu_peak_allocated = max(l.gpu_peak_allocated, msg.get("gpu_peak_allocated") or 0)
            msg.update(lane=idx, t_end=round(time.time() - self.t0, 2),
                       t_start=round(l.t_task - self.t0, 2) if l.t_task else None, budget_s=l.budget,
                       status="ok" if msg.get("ok") else "failed")
            j.results.append(msg)
            self.event("parent_done", lane=idx, job=j.name, code=msg["code"], ok=msg["ok"],
                       wall_s=round(msg.get("wall_s") or 0, 1), peak_rss_gb=round((msg.get("peak_rss") or 0) / R.GB, 2),
                       gpu_peak_mib=round((msg.get("gpu_peak_reserved") or 0) / 2**20),
                       gpu_alloc_mib=round((msg.get("gpu_peak_allocated") or 0) / 2**20),
                       oom_retries=msg.get("oom_retries"), error=msg.get("error"))
            l.state, l.code = "idle", None

    # ------------------------------------------------------------------ periodic checks
    def periodic(self):
        now = time.time()
        for j in self.jobs:
            j.prune()
        if self.device == "cuda":
            g = R.gpu_query()
            if g is not None:
                self.gutil.append((now, g["util"]))
                self.gutil[:] = self.gutil[-120:]
                for l in self.lanes.values():
                    if l.pid in g["apps"]:
                        self.lane_gpu_seen[l.pid] = max(self.lane_gpu_seen.get(l.pid, 0.0), g["apps"][l.pid])
        # per-parent wall budget
        for l in self.live():
            if l.state == "busy" and l.budget and now - l.t_task > l.budget:
                self.event("parent_timeout", lane=l.idx, job=l.job.name, code=l.code, budget_s=round(l.budget),
                           elapsed_s=round(now - l.t_task))
                l.job.write_status(l.code, "timeout", f"parent wall budget {l.budget:.0f} s exceeded", now - l.t_task)
                l.job.results.append(dict(code=l.code, ok=False, status="timeout", wall_s=now - l.t_task,
                                          budget_s=l.budget, lane=l.idx))
                l.timed_out = True
                l.kill()
        # memory pressure
        avail = R.mem_available()
        lv = self.live()
        over_cap = self.max_rss_gb is not None and R.tree_rss(os.getpid()) > self.max_rss_gb * R.GB
        if len(lv) > 1 and (avail < self.mem_headroom_gb * R.GB or over_cap) and now - self.last_retire > 30:
            newest = max((l for l in lv if not l.retiring), key=lambda l: l.idx, default=None)
            if newest is not None:
                newest.retiring = True
                self.last_retire = now
                self.event("lane_retire", lane=newest.idx, mem_available_gb=round(avail / R.GB, 2), over_cap=over_cap)
            if avail < self.mem_headroom_gb * R.GB / 3:
                victim = max(lv, key=lambda l: l.idx)
                self.event("lane_kill_pressure", lane=victim.idx, code=victim.code, mem_available_gb=round(avail / R.GB, 2))
                if victim.code:
                    victim.job.requeued.discard(victim.code)
                victim.kill()
        # run deadline
        if self.max_wall_s is not None and now - self.t0 > self.max_wall_s and not self.stop_dispatch:
            self.stop_dispatch = True
            for j in self.jobs:
                j.not_run.extend(j.todo)
                j.todo.clear()
            self.event("deadline", not_dispatched=sum(len(j.not_run) for j in self.jobs))
        if self.max_wall_s is not None and now - self.t0 > self.max_wall_s + self.kill_grace_s:
            for l in self.live():
                if l.state == "busy":
                    self.event("deadline_kill", lane=l.idx, code=l.code)
                    elapsed = now - l.t_task
                    if elapsed >= self.max_wall_s:
                        l.job.write_status(l.code, "timeout", "run wall budget exceeded", elapsed)
                        l.job.results.append(dict(code=l.code, ok=False, status="timeout", wall_s=elapsed, lane=l.idx))
                    else:
                        l.job.not_run.append(l.code)
                    l.timed_out = True
                l.kill()
                l.state = "stopped"
        # no lane and none admissible for 10 minutes: stop instead of waiting forever
        if self.todo_total() and not self.live():
            self.stalled_since = getattr(self, "stalled_since", None) or now
            if now - self.stalled_since > 600:
                self.stop_dispatch = True
                for j in self.jobs:
                    j.not_run.extend(j.todo)
                    j.todo.clear()
                self.event("stalled", reason=self.last_reason)
        else:
            self.stalled_since = None
        # admission
        if self.todo_total() and not self.stop_dispatch:
            ok, reason = self.can_add()
            job = self.job_for_new_lane() if ok else None
            if ok and job is not None:
                self.spawn(job)
            elif reason.split(":")[0] != (self.last_reason or "").split(":")[0]:  # log changes of kind, not of numbers
                self.event("admission_hold", reason=reason, lanes=len(self.live()))
            self.last_reason = reason

    def dispatch_all(self):
        for l in list(self.lanes.values()):
            if l.state != "idle":
                continue
            if l.retiring or not l.job.todo or self.stop_dispatch:
                l.send({"stop": True})
                l.state = "stopped"
            else:
                self.dispatch(l)

    # ------------------------------------------------------------------ main loop
    def run(self, spool: Path | None = None, idle_exit_s: float = 0.0) -> dict:
        def extra():
            return dict(lanes=len(self.live()), busy=len(self.busy()), queued=self.todo_total(),
                        done=sum(len(j.results) for j in self.jobs))
        sampler = Sampler(self.out / "telemetry.jsonl", root_pid=os.getpid(), period=self.poll_s, extra=extra).start()
        self.event("start", machine=R.snapshot(), jobs=[j.name for j in self.jobs])
        last_check, last_spool, idle_since = 0.0, 0.0, None
        try:
            while True:
                try:
                    while True:
                        idx, msg = self.events.get(timeout=0.2)
                        self.handle(idx, msg)
                except queue.Empty:
                    pass
                now = time.time()
                if spool is not None and now - last_spool > 10:
                    last_spool = now
                    for f in sorted((spool / "queue").glob("*.json")):
                        try:
                            plan = json.loads(f.read_text())
                            os.replace(f, spool / "active" / f.name)
                            self.add_job(Job(plan, resume=True))
                        except Exception as exc:
                            os.replace(f, spool / "rejected" / f.name)
                            self.event("job_rejected", file=f.name, error=repr(exc))
                if now - last_check >= self.poll_s:
                    last_check = now
                    self.periodic()
                self.dispatch_all()
                active = self.todo_total() or self.busy() or any(l.state == "idle" for l in self.lanes.values())
                if active:
                    idle_since = None
                    continue
                if spool is None:
                    break
                for j in [j for j in self.jobs if not j.todo and not any(l.job is j and l.live for l in self.lanes.values())]:
                    self.finish_job(j)
                    self.jobs.remove(j)
                    src = spool / "active" / f"{j.name}.json"
                    if src.exists():
                        os.replace(src, spool / "done" / src.name)
                idle_since = idle_since or now
                if idle_exit_s and now - idle_since > idle_exit_s:
                    break
        finally:
            for l in self.lanes.values():
                if l.p.poll() is None:
                    l.send({"stop": True})
            deadline = time.time() + 30
            for l in self.lanes.values():
                try:
                    l.p.wait(timeout=max(1.0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    l.kill()
            sampler.stop()
        wall = time.time() - self.t0
        summaries = [self.finish_job(j) for j in self.jobs]
        tel = sampler.summary()
        n = sum(s["parents_done"] for s in summaries)
        n_par = sum(s["distinct_parents"] for s in summaries)
        summ = dict(out=str(self.out), device=self.device, wall_s=round(wall, 1), items_done=n, parents_done=n_par,
                    parents_per_min=round(n_par / (wall / 60), 3) if wall > 0 else None, lanes_started=len(self.lanes),
                    max_concurrent_lanes=max((r.get("lanes", 0) for r in sampler.rows), default=0),
                    lane_peak_rss_gb=round(max((l.peak_rss for l in self.lanes.values()), default=0) / R.GB, 2),
                    lane_gpu_peak_mib=round(max(self.lane_gpu_seen.values(), default=0.0)), telemetry=tel, jobs=summaries)
        summ["line"] = one_line(dict(tel, wall_s=wall), n_par) + f", lanes max {summ['max_concurrent_lanes']}"
        (self.out / "run_summary.json").write_text(json.dumps(summ, indent=1, default=str))
        print(summ["line"], flush=True)
        return summ

    def finish_job(self, j: Job) -> dict:
        validity = {}
        for f in sorted((j.out / "_lanes").glob("l*/embedding_validity.json")):
            try:
                validity.update(json.loads(f.read_text()))
            except ValueError:
                pass
        (j.out / "embedding_validity.json").write_text(json.dumps(validity, indent=2))
        stage_tot: dict = {}
        for r in j.results:
            for k, v in (r.get("stages") or {}).items():
                stage_tot[k] = stage_tot.get(k, 0.0) + (v or 0.0)
        s = dict(job=j.name, out=str(j.out), backend=j.plan.get("backend", "bench"), arms=j.arm_names, seeds=j.seeds,
                 parents_done=len(j.results), distinct_parents=len({r["code"] for r in j.results}), parents_ok=sum(1 for r in j.results if r.get("ok")),
                 parents_failed=[r["code"] for r in j.results if r.get("status") == "failed"],
                 records_failed=sum(len(r.get("failed_records") or []) for r in j.results),
                 parents_timeout=j.skipped_timeouts + [r["code"] for r in j.results if r.get("status") == "timeout"],
                 skipped_resume=j.skipped, not_run=j.not_run,
                 sum_parent_wall_s=round(sum(r.get("wall_s") or 0 for r in j.results), 1),
                 stage_totals_s={k: round(v, 1) for k, v in stage_tot.items()},
                 oom_retries=sum(r.get("oom_retries") or 0 for r in j.results))
        (j.out / "run_results.json").write_text(json.dumps(j.results, indent=1, default=str))
        if j.plan.get("pack"):
            s["packed"] = pack_outputs(j.out)
        (j.out / "job_summary.json").write_text(json.dumps(s, indent=1, default=str))
        return s


KEEP_LOOSE = {"run_plan.json", "run_results.json", "job_summary.json", "run_summary.json", "run_events.jsonl",
              "telemetry.jsonl", "embedding_validity.json"}


def pack_outputs(out: Path) -> dict:
    """Inode budget (Perlmutter): every loose output under OUT (records, pools, lane logs) goes into one new
    OUT/run_pack<N>.tar and the loose copies are removed; summaries stay loose. Records in the pack count for resume.
    Read a record without unpacking: tarfile.open(...).extractfile('C/records/X_21.json')."""
    import tarfile
    files = [f for f in sorted(out.rglob("*")) if f.is_file() and f.name not in KEEP_LOOSE
             and not f.name.startswith("run_pack") and "/_lanes/tmp/" not in str(f)]
    if not files:
        return dict(files=0)
    n = len(list(out.glob("run_pack*.tar")))
    tf = out / f"run_pack{n}.tar"
    with tarfile.open(tf, "w") as t:
        for f in files:
            t.add(f, arcname=str(f.relative_to(out)))
    for f in files:
        f.unlink()
    for d in sorted((d for d in out.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
        try:
            d.rmdir()
        except OSError:
            pass
    return dict(files=len(files), tar=str(tf), bytes=tf.stat().st_size)


def run_panel(plan: dict, **kw) -> dict:
    resume = kw.pop("resume", True)
    s = Scheduler(plan["out"], plan.get("device", "cuda"), **kw)
    s.add_job(Job(plan, resume=resume))
    return s.run()
