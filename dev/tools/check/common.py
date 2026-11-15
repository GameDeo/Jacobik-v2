"""Shared helpers for the check levels: repo paths, env, stage timing, reports."""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXPECTED_FINGERPRINT = "2c87e2940af2"  # donor_assignment consolidation (2026-11-14), superseding a47283387172 (P2R)
GOLDEN_ARRAYS_EXPECTED = 245


from ..repo import BUDGET_DB_SUPPORT, main_checkout as _main_checkout


def main_checkout() -> Path:
    """The main checkout of this repository (worktrees share its data/ and research/data)."""
    return _main_checkout(REPO)


#: where local outputs go: the main checkout's research/data (gitignored; shared by all worktrees; never /tmp, which is
#: RAM-backed). On Perlmutter the levels write under $SCRATCH/jacobik/runs/ instead.
FASTCHECK_OUT = main_checkout() / "research" / "data" / "infra" / "fastcheck"


def base_env(extra: dict | None = None) -> dict:
    """Environment for every check child: one numerical thread, the repo source first, no override leakage."""
    env = {k: v for k, v in os.environ.items() if k != "JACOBIK_PROFILE_OVERRIDES"}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "dev"), str(REPO), str(REPO / "research" / "bench")]),
               PYTHONIOENCODING="utf-8")
    env.setdefault("JACOBIK_DATA_ROOT", str(main_checkout() / "data"))
    env.setdefault("JACOBIK_BUDGET_OUT", str(main_checkout() / "research/archive/data/energy_scale_refit"))
    env.setdefault("JACOBIK_BUDGET_DB", str(main_checkout() / BUDGET_DB_SUPPORT))
    if extra:
        env.update(extra)
    return env


def load1() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return float("nan")


def stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class Stage:
    name: str
    wall_s: float = 0.0
    load1_before: float = float("nan")
    load1_after: float = float("nan")
    ok: bool = False
    summary: str = ""
    detail: dict = field(default_factory=dict)


class Report:
    """Collects stages, prints the timing table and the pass/fail summary, writes JSON."""

    def __init__(self, level: str, out_dir: Path):
        self.level = level
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stages: list[Stage] = []
        self.started = time.time()
        self.notes: list[str] = []

    def run(self, name: str, fn):
        """Time `fn()`; it returns (ok, summary, detail). Exceptions are a failure of that stage, not a crash."""
        st = Stage(name=name, load1_before=load1())
        t0 = time.time()
        try:
            st.ok, st.summary, st.detail = fn()
        except Exception as exc:  # the stage fails, the other stages still run
            st.ok, st.summary, st.detail = False, f"exception: {exc!r}", {}
        st.wall_s = time.time() - t0
        st.load1_after = load1()
        self.stages.append(st)
        self._print_stage(st)
        return st

    def run_parallel(self, named_fns: list):
        """Run several stages at once (threads; each stage starts its own processes). Each keeps its own wall."""
        from concurrent.futures import ThreadPoolExecutor

        def one(item):
            name, fn = item
            st = Stage(name=name, load1_before=load1())
            t0 = time.time()
            try:
                st.ok, st.summary, st.detail = fn()
            except Exception as exc:
                st.ok, st.summary, st.detail = False, f"exception: {exc!r}", {}
            st.wall_s = time.time() - t0
            st.load1_after = load1()
            self._print_stage(st)
            return st

        with ThreadPoolExecutor(max_workers=len(named_fns)) as pool:
            stages = list(pool.map(one, named_fns))
        self.stages.extend(stages)
        return stages

    def _print_stage(self, st: Stage):
        print(f"[{self.level}] {st.name:<22} {'PASS' if st.ok else 'FAIL'}  {st.wall_s:8.1f} s  "
              f"load1 {st.load1_before:.2f}->{st.load1_after:.2f}  {st.summary}", flush=True)

    def table(self) -> str:
        total = time.time() - self.started
        lines = [f"{'stage':<24}{'wall s':>10}{'load1 before':>14}{'after':>8}  result"]
        for s in self.stages:
            lines.append(f"{s.name:<24}{s.wall_s:>10.1f}{s.load1_before:>14.2f}{s.load1_after:>8.2f}  "
                         f"{'PASS' if s.ok else 'FAIL'} {s.summary}")
        lines.append(f"{'TOTAL wall':<24}{total:>10.1f}")
        return "\n".join(lines)

    @property
    def ok(self) -> bool:
        return bool(self.stages) and all(s.ok for s in self.stages)

    def finish(self) -> int:
        print("\n" + self.table())
        failed = [s.name for s in self.stages if not s.ok]
        verdict = "PASS" if self.ok else f"FAIL ({', '.join(failed) or 'no stages ran'})"
        print(f"\n[{self.level}] {verdict}")
        path = self.out_dir / f"{self.level}_{stamp()}.json"
        payload = dict(level=self.level, ok=self.ok, total_wall_s=time.time() - self.started,
                       started=_dt.datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
                       stages=[asdict(s) for s in self.stages], notes=self.notes, repo=str(REPO),
                       commit=_commit())
        path.write_text(json.dumps(payload, indent=1, default=str))
        print(f"[{self.level}] report: {path}")
        return 0 if self.ok else 1


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


#: measured peak RSS per child (workstation, 2026-11-08): golden child about 1.5 GB, pytest shard about 1.2 GB
GOLDEN_CHILD_GB = 1.5
PYTEST_SHARD_GB = 1.2


def resource_plan(golden_jobs: int, shards: int, headroom_gb: float = 2.0, max_rss_gb: float | None = None) -> dict:
    """Shrink golden jobs and pytest shards to what MemAvailable (capped by the cgroup) and the CPU budget allow, net
    of other users' load. Refuses (ok False) when not even one golden child and one shard fit."""
    from tools.run import resources as R
    avail = R.mem_available() / R.GB - headroom_gb
    if max_rss_gb is not None:
        avail = min(avail, max_rss_gb)
    cpus = R.cpu_budget() - R.foreign_cpu_load(0.0)
    # golden is the long pole (one child per profile x group): it gets memory first, pytest one shard at least
    jobs_n = max(0, min(golden_jobs, int((avail - PYTEST_SHARD_GB) // GOLDEN_CHILD_GB), max(1, int(cpus - 1))))
    left = avail - jobs_n * GOLDEN_CHILD_GB
    shards_n = max(0, min(shards, int(left // PYTEST_SHARD_GB), max(1, int(cpus - jobs_n))))
    ok = jobs_n >= 1 and shards_n >= 1
    plan = dict(ok=ok, golden_jobs=jobs_n, pytest_shards=shards_n, mem_available_gb=round(avail + headroom_gb, 1),
                cpu_free=round(cpus, 1), expected_peak_gb=round(jobs_n * GOLDEN_CHILD_GB + shards_n * PYTEST_SHARD_GB, 1),
                requested=dict(golden_jobs=golden_jobs, pytest_shards=shards))
    return plan
