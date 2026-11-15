"""Whole-run trace: GPU utilisation and memory, our RSS, MemAvailable, load, lanes. One sampler thread per run.

`python -m tools.run sample --out DIR -- <command ...>` wraps any existing runner (the sequential bench, the isomer
screen, the D3 harness) with the same trace, so "before" and "after" are measured the same way.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from . import resources as R


class Sampler:
    def __init__(self, path: Path, root_pid: int | None = None, period: float = 2.0, extra=None):
        self.path, self.root, self.period, self.extra = Path(path), root_pid or os.getpid(), period, extra
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True, name="telemetry")
        self.t0 = time.time()

    def _loop(self):
        with self.path.open("a") as fh:
            while not self._stop.is_set():
                try:
                    own = R.tree_pids(self.root)
                    g = R.gpu_query()
                    row = dict(t=round(time.time() - self.t0, 2), load1=R.load1(),
                               mem_available_gb=round(R.mem_available() / R.GB, 3),
                               own_rss_gb=round(sum(R.proc_rss(p) for p in own) / R.GB, 3), n_procs=len(own))
                    if g is not None:
                        row.update(gpu_util=g["util"], gpu_power_w=g.get("power_w"), gpu_sm_mhz=g.get("sm_clock_mhz"),
                                   gpu_used_mib=g["used_mib"], gpu_total_mib=g["total_mib"],
                                   gpu_own_mib=sum(v for p, v in g["apps"].items() if p in own),
                                   gpu_foreign=len([p for p in g["apps"] if p not in own]))
                    if self.extra:
                        row.update(self.extra())
                    self.rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                except Exception as exc:  # telemetry never stops a run
                    fh.write(json.dumps(dict(t=round(time.time() - self.t0, 2), error=repr(exc))) + "\n")
                self._stop.wait(self.period)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=30)

    def summary(self, t_from: float | None = None, t_to: float | None = None) -> dict:
        rows = [r for r in self.rows if "error" not in r and (t_from is None or r["t"] >= t_from)
                and (t_to is None or r["t"] <= t_to)]
        return summarise(rows)


def summarise(rows: list[dict]) -> dict:
    out = dict(n_samples=len(rows))
    if not rows:
        return out
    g = [r["gpu_util"] for r in rows if "gpu_util" in r]
    if g:
        # nvidia-smi utilization.gpu = share of the sample period in which a kernel was running
        pw = [r["gpu_power_w"] for r in rows if r.get("gpu_power_w") is not None and r["gpu_power_w"] == r["gpu_power_w"]]
        if pw:
            out.update(gpu_power_mean_w=round(sum(pw) / len(pw), 1), gpu_power_max_w=round(max(pw), 1))
        out.update(gpu_busy_pct=round(sum(g) / len(g), 1), gpu_samples_idle_pct=round(100 * sum(x < 5 for x in g) / len(g), 1),
                   gpu_peak_used_mib=max(r.get("gpu_used_mib", 0) for r in rows),
                   gpu_peak_own_mib=max(r.get("gpu_own_mib", 0) for r in rows))
    out.update(peak_own_rss_gb=max(r["own_rss_gb"] for r in rows), min_mem_available_gb=min(r["mem_available_gb"] for r in rows),
               mean_load1=round(sum(r["load1"] for r in rows) / len(rows), 2))
    return out


def load_trace(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
        except ValueError:
            pass
    return rows


def sample_command(cmd: list[str], out: Path, period: float = 2.0, records_glob: str | None = None) -> int:
    """Run `cmd` (an existing runner) under the sampler; write out/telemetry.jsonl and out/sample_summary.json."""
    import subprocess
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (out / "sampled_command.log").open("w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        s = Sampler(out / "telemetry.jsonl", root_pid=p.pid, period=period).start()
        try:
            rc = p.wait()
        finally:
            s.stop()
    wall = time.time() - t0
    n = None
    if records_glob:
        import glob
        n = len(glob.glob(records_glob))
    summ = dict(cmd=cmd, rc=rc, wall_s=round(wall, 1), records=n, **s.summary())
    (out / "sample_summary.json").write_text(json.dumps(summ, indent=1))
    print(one_line(summ, n_parents=None), flush=True)
    return rc


def one_line(s: dict, n_parents: int | None) -> str:
    busy = s.get("gpu_busy_pct")
    rate = f", {n_parents / (s['wall_s'] / 60):.2f} parents/min" if n_parents and s.get("wall_s") else ""
    pw = s.get("gpu_power_mean_w")
    power = f", GPU power {pw} W mean / {s.get('gpu_power_max_w')} W max" if pw is not None else ""
    return (f"GPU busy {busy if busy is not None else 'n/a'} %{power}, wall {s.get('wall_s', 0):.0f} s{rate}, "
            f"peak RSS {s.get('peak_own_rss_gb', 0):.1f} GB, min MemAvailable {s.get('min_mem_available_gb', 0):.1f} GB")
