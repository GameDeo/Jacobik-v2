"""Measure where one parent's time goes (Phase 2 §1): stage split, step-loop kernels, host syncs, power.

  python -m tools.run.profile_parent --code AWONOR --out DIR [--device cuda] [--seeds 21,22,23] [--arm C]
                                     [--torch-profile] [--sync-debug] [--repeat 2]

Runs the parent exactly as a tools.run bench lane does (BenchBackend: the bench's own code path, deterministic
algorithms on, C-2b arm table), in this process, `--repeat` times (the first run pays one-off costs: GFN2 cache,
CUDA graph pools, cuBLAS/cuSOLVER handles; the last run is the one reported). Around every joint IK solve it records:
  * wall of the solve (synchronised), solver_replay.LAST (replay mode, steps replayed / executed, eigh / half split);
  * with --torch-profile: CUDA kernels launched and their summed and merged (union of intervals) device time, so
    kernel-busy share = merged kernel time / solve wall; kernels per step; the top kernels;
  * with --sync-debug: torch.cuda.set_sync_debug_mode("warn") counts every synchronising call (.item(), .cpu(),
    nonzero, eigh's info check, ...) grouped by Python call site;
  * nvidia-smi power.draw, utilization.gpu and clocks.sm sampled every 0.25 s while the solve runs.
Nothing here changes what is computed. Writes DIR/profile_<code>.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path


class PowerSampler:
    """nvidia-smi in loop mode (one process, -lms) for power / util / SM clock while a window is open."""

    def __init__(self, period_ms: int = 250):
        self.period_ms = period_ms
        self.rows: list[tuple] = []
        self.proc = None
        self._t = None

    def start(self):
        try:
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,clocks.sm",
                                          "--format=csv,noheader,nounits", f"-lms={self.period_ms}"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            self.proc = None
            return self

        def read():
            for line in self.proc.stdout:
                try:
                    p, u, c = [float(x) for x in line.strip().split(",")]
                    self.rows.append((time.monotonic(), p, u, c))
                except ValueError:
                    pass
        self._t = threading.Thread(target=read, daemon=True)
        self._t.start()
        return self

    def window(self, t0: float, t1: float) -> dict:
        w = [r for r in self.rows if t0 <= r[0] <= t1]
        if not w:
            return {}
        n = len(w)
        return dict(n=n, power_mean_w=round(sum(r[1] for r in w) / n, 1), power_max_w=max(r[1] for r in w),
                    util_mean_pct=round(sum(r[2] for r in w) / n, 1), sm_mhz_mean=round(sum(r[3] for r in w) / n))

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()  # our own child, by handle (PID)
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


def merged_busy_us(intervals: list[tuple[float, float]]) -> float:
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def kernel_stats(prof) -> dict:
    """CUDA kernel count and summed device time from the profiler's C++ aggregate (key_averages; walking every event
    in Python takes minutes for an eager solve). One stream, so summed ~ merged busy time."""
    n, tot, rows = 0, 0.0, []
    for e in prof.key_averages():
        dt = getattr(e, "self_device_time_total", None)
        if dt is None:
            dt = getattr(e, "self_cuda_time_total", 0.0)
        if dt and e.device_type is not None and str(e.device_type).endswith("CUDA"):
            n += e.count
            tot += dt
            rows.append((e.key, e.count, round(dt / 1e3, 2)))
    if not rows:  # older torch: kernels appear as CPU-side keys with device time
        for e in prof.key_averages():
            dt = getattr(e, "self_device_time_total", 0.0) or 0.0
            if dt > 0:
                n += e.count
                tot += dt
                rows.append((e.key, e.count, round(dt / 1e3, 2)))
    return dict(n_kernels=n, kernel_sum_ms=round(tot / 1e3, 2), kernel_merged_ms=round(tot / 1e3, 2),
                top_by_time=sorted(rows, key=lambda r: -r[2])[:15], top_by_count=sorted(rows, key=lambda r: -r[1])[:15])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.run.profile_parent")
    ap.add_argument("--code", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seeds", default="21,22,23")
    ap.add_argument("--arm", default="C")
    ap.add_argument("--repeat", type=int, default=3,
                    help="runs of the parent; the last is instrumented (profiler / sync debug), the one before is the plain timing")
    ap.add_argument("--torch-profile", action="store_true")
    ap.add_argument("--sync-debug", action="store_true")
    ap.add_argument("--no-deterministic", action="store_true")
    ap.add_argument("--timing-sync", action="store_true",
                    help="ReplaySolver.timing_sync: synchronise after each half so LAST splits half A / eigh / half B "
                         "(adds 3 syncs per step; for the split only)")
    ap.add_argument("--graph-nodes", action="store_true",
                    help="count the nodes (kernels, memcpy/memset) of every captured CUDA graph (driver cuGraphGetNodes; "
                         "graphs are kept after capture for the count)")
    ap.add_argument("--dump-dedup", action="store_true",
                    help="save every StructureDeduper.select call of the last run (deduper, coordinates, scores, k, "
                         "kept rows, wall) to OUT/dedup_<code>_<i>.pt, for offline timing and exact-identity tests")
    ap.add_argument("--eager-profile", action="store_true",
                    help="instrumented run with ReplaySolver.replay = False (the same step functions, eagerly): the "
                         "torch profiler crashes on CUDA-graph replay (CUPTI), so kernels per step are counted here")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from tools.run.__main__ import C2_ARMS
    from tools.run.lane import BenchBackend
    plan = dict(out=str(out / "run"), device=a.device, seeds=[int(s) for s in a.seeds.split(",")],
                arms=[dict(arm=a.arm, flags=C2_ARMS[a.arm], seed_budget="single_w48" if a.arm == "S1" else "three_w16")],
                deterministic=not a.no_deterministic, backend="bench")
    t_imp = time.monotonic()
    be = BenchBackend(plan, 0)
    startup = time.monotonic() - t_imp
    import torch
    import jacobik.ik.solver_replay as SR
    from jacobik.ik import ik_solver as IKM

    SR.ReplaySolver.timing_sync = bool(a.timing_sync)
    graph_nodes: list[int] = []
    if a.graph_nodes and a.device == "cuda":
        import ctypes
        _cu = ctypes.CDLL("libcuda.so.1")
        _Base = torch.cuda.CUDAGraph

        class CountingGraph(_Base):
            def __new__(cls, keep_graph=False):
                return super().__new__(cls, keep_graph=True)

            def __init__(self, keep_graph=False):
                super().__init__(keep_graph=True)

            def capture_end(self):
                super().capture_end()
                n = ctypes.c_size_t(0)
                if _cu.cuGraphGetNodes(ctypes.c_void_p(self.raw_cuda_graph()), None, ctypes.byref(n)) == 0:
                    graph_nodes.append(int(n.value))
        torch.cuda.CUDAGraph = CountingGraph
    solves: list[dict] = []
    opts = dict(profile=False)
    power = PowerSampler().start() if a.device == "cuda" else None
    sync = (lambda: torch.cuda.synchronize()) if a.device == "cuda" else (lambda: None)

    def wrap(cls):
        orig = cls._solve_impl

        def _solve_impl(self, *args, **kw):
            sync()
            graph_nodes.clear()
            rec = dict(cls=cls.__name__, B=int(args[0][0].shape[0]), n_lig=len(args[0]),
                       dof=[int(t.shape[1]) for t in args[0]], n_atoms=int(args[1].shape[-2]))
            caught = []
            prof = None
            ctx_w = warnings.catch_warnings(record=True)
            wlist = ctx_w.__enter__()
            warnings.simplefilter("always")
            if opts["profile"] and a.sync_debug and a.device == "cuda":
                torch.cuda.set_sync_debug_mode("warn")
            if opts["profile"] and a.torch_profile:
                acts = [torch.profiler.ProfilerActivity.CPU]
                if a.device == "cuda":
                    acts.append(torch.profiler.ProfilerActivity.CUDA)
                prof = torch.profiler.profile(activities=acts, record_shapes=False)
                prof.__enter__()
            t0 = time.monotonic()
            try:
                res = orig(self, *args, **kw)
                sync()
            finally:
                t1 = time.monotonic()
                if prof is not None:
                    prof.__exit__(None, None, None)
                if a.device == "cuda":
                    torch.cuda.set_sync_debug_mode(0)
                ctx_w.__exit__(None, None, None)
                caught = [w for w in wlist if "synchroniz" in str(w.message).lower()]
            rec.update(wall_s=round(t1 - t0, 4), last={k: (round(v, 4) if isinstance(v, float) else v)
                                                       for k, v in SR.LAST.items() if k != "error"},
                       replay_error=(SR.LAST.get("error") or "")[-400:] or None)
            if opts["profile"] and a.sync_debug:
                sites = Counter(f"{Path(w.filename).name}:{w.lineno}" for w in caught)
                rec.update(n_syncs=len(caught), sync_sites=sites.most_common(20))
            if prof is not None:
                rec.update(kernel_stats(prof))
            if power is not None:
                rec["power"] = power.window(t0, t1)
            if graph_nodes:
                rec["graph_nodes"] = list(graph_nodes)  # [half A, half B]: kernels per replayed step = sum
            steps = rec["last"].get("steps_executed") or 0
            if steps:
                if "n_kernels" in rec:
                    rec["kernels_per_step"] = round(rec["n_kernels"] / steps, 1)
                    rec["kernel_busy_pct"] = round(100 * rec["kernel_merged_ms"] / 1e3 / rec["wall_s"], 1)
                if "n_syncs" in rec:
                    rec["syncs_per_step"] = round(rec["n_syncs"] / steps, 2)
            solves.append(rec)
            return res
        cls._solve_impl = _solve_impl
    wrap(SR.ReplaySolver)
    wrap(IKM.JointIKSolver)

    dumps = {"on": False, "n": 0}
    if a.dump_dedup:
        from jacobik.ik import dedup as DD
        _orig_select = DD.StructureDeduper.select

        def select(self, X_all, score, k, *args, **kw):
            t = time.monotonic()
            keep = _orig_select(self, X_all, score, k, *args, **kw)
            dt = time.monotonic() - t
            if dumps["on"]:
                torch.save(dict(deduper=self, X_all=X_all.detach().cpu(), score=score.detach().cpu(), k=k, args=args,
                                kw=kw, keep=list(keep), wall_s=dt, n_maps=int(self.inv_maps.shape[0])),
                           out / f"dedup_{a.code}_{dumps['n']}.pt")
                dumps["n"] += 1
            return keep
        DD.StructureDeduper.select = select

    if a.dump_dedup:  # MACE inputs too (same switch): the structures of every energies() call of the last run
        _mace = be.mace
        _orig_en = type(_mace).energies

        def energies(this, structures, *args, **kw):
            t = time.monotonic()
            e = _orig_en(this, structures, *args, **kw)
            if dumps["on"]:
                torch.save(dict(structures=list(structures), energies=e, wall_s=time.monotonic() - t,
                                batch_atoms=this._batch_atoms), out / f"mace_{a.code}_{dumps['n']}.pt")
                dumps["n"] += 1
            return e
        type(_mace).energies = energies

    runs = []
    for i in range(a.repeat):
        last = i == a.repeat - 1
        opts["profile"] = last
        SR.ReplaySolver.replay = not (last and a.eager_profile)
        dumps["on"] = last
        solves.clear()
        be.stage.clear()
        t0 = time.monotonic()
        try:
            res = be.run_parent(a.code)
        except Exception:
            res = dict(error=traceback.format_exc()[-800:])
        wall = time.monotonic() - t0
        recs = sorted((out / "run" / a.arm / "records").glob(f"{a.code}_*.json"))
        rr = [json.loads(p.read_text()) for p in recs]
        phases = (rr[0].get("run_timing") or {}).get("phases", {}) if rr else {}
        runs.append(dict(i=i, wall_s=round(wall, 3), error=res.get("error"), stages=dict(be.stage), phases=phases,
                         wall_solve_s=[r.get("wall_solve_s") for r in rr], n_rows=[r.get("n_rows") for r in rr],
                         solves=list(solves)))
        for p in recs:
            p.unlink()
    if power is not None:
        power.stop()
    doc = dict(code=a.code, device=a.device, gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else None,
               host=os.uname().nodename, torch=torch.__version__, seeds=plan["seeds"], arm=a.arm,
               deterministic=plan["deterministic"], startup_s=round(startup, 2), runs=runs)
    (out / f"profile_{a.code}.json").write_text(json.dumps(doc, indent=1, default=str))
    r = runs[-1]
    print(f"{a.code} on {doc['gpu'] or 'cpu'}: parent wall {r['wall_s']:.1f} s; phases {r['phases']}; "
          f"stages {{{', '.join(f'{k}: {v:.2f}' for k, v in r['stages'].items())}}}")
    for s in r["solves"]:
        L = s["last"]
        print(f"  solve B={s['B']} lig={s['n_lig']} dof={s['dof']} atoms={s['n_atoms']}: wall {s['wall_s']:.2f} s, "
              f"mode {L.get('mode')}, steps {L.get('steps_executed')} (replayed {L.get('replayed')}), "
              f"setup {L.get('s_setup')} step0 {L.get('s_step0')} record {L.get('s_record')} replay {L.get('s_replay')} "
              f"final {L.get('s_final')}; kernels/step {s.get('kernels_per_step')}, kernel busy {s.get('kernel_busy_pct')} %, "
              f"syncs/step {s.get('syncs_per_step')}, power {s.get('power')}")
        if s.get("sync_sites"):
            print("    sync sites:", s["sync_sites"][:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
