"""One lane: a persistent process that owns a CUDA context and runs whole parents, one at a time, on request.

Started by the orchestrator (dev/tools/run/orchestrator.py): `python -u -m tools.run.lane --plan PLAN --lane I`.
Protocol: one JSON object per line. stdout carries only protocol lines; everything else goes to stderr (the lane log).
  lane -> {"ready": true, "pid", "startup_s", "rss", "gpu_mib"}      after imports, MACE, DB, GFN2 self-check
  orch -> {"code": "ABCDEF"}                                          run one parent (all arms, all seeds)
  lane -> {"code", "ok", "error", "wall_s", "stages", "peak_rss", "gpu_peak_reserved", "oom_retries", "records"}
  orch -> {"stop": true}

Backends (plan["backend"]):
  bench   tools.jgen.bench.run.main(--arms-spec ..., --only CODE), the path of the C-2b driver and the D3 worker:
          every arm of a parent in one call (shared prepare), deterministic algorithms on, records with the bench's
          fields plus c2_timing / c2_identity computed exactly as dev/tools/jgen/c2/driver.py does, plus run_timing.
  isomer  research/campaigns/i/run_i_screen.py main(--only CODE) from the path in plan["isomer_script"]
          (the isomer agent's script, imported, not modified). Its GPU lock is held once by the orchestrator.

Heavy objects load once per lane (MACE model, constraints DB, GFN2 self-check), as the D3 worker does. A parent's
failure is a record; an exception inside a backend is caught and reported, never fatal to the lane.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def _vm_hwm() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _rss() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _reset_hwm():
    try:
        Path("/proc/self/clear_refs").write_text("5")  # resets VmHWM (peak RSS) to the current RSS
    except Exception:
        pass


class _Common:
    """Shared once-per-lane state and the hooks that time stages. Nothing here changes what is computed."""

    def __init__(self, plan: dict, lane: int):
        self.plan, self.lane = plan, lane
        self.out = Path(plan["out"])
        self.lane_dir = self.out / "_lanes" / f"l{lane}"
        self.lane_dir.mkdir(parents=True, exist_ok=True)
        self.device = plan.get("device", "cuda")
        self.seeds = [int(s) for s in plan["seeds"]]
        self.arm_dir = {a["arm"]: Path(a.get("out") or self.out / a["arm"]) for a in plan["arms"]}
        self.stage: dict = {}
        self.cur = {"code": None, "stage": None, "t0": None}

    def add(self, key, dt):
        self.stage[key] = self.stage.get(key, 0.0) + dt

    def mark(self, name):
        self.cur["stage"] = name

    def install_common(self, energy_mod, bench, common_mod, ev_mod, engine):
        import torch
        from jacobik.core.env import MACE_MODEL_PATH
        self.torch = torch
        mace = energy_mod.MACEPolarEvaluator(model="polar-1-s", device=self.device, dtype="float64",
                                            model_path=str(MACE_MODEL_PATH), mode="energy-only", threads=1).load()
        orig_energies = type(mace).energies

        def energies(this, *a, **k):
            self.mark("mace")
            t = time.monotonic()
            try:
                return orig_energies(this, *a, **k)
            finally:
                self.add("mace_s", time.monotonic() - t)
        type(mace).energies = energies
        self.mace = mace
        # the runners construct the evaluator by name in their own namespace: hand back the loaded one
        factory = lambda *a, **k: mace  # noqa: E731
        energy_mod.MACEPolarEvaluator = factory
        self.mace_factory = factory
        db_seen = {}

        orig_stats = common_mod.stats_from_db

        def cached_stats(db):
            key = str(Path(db).resolve())
            if key not in db_seen:
                db_seen[key] = orig_stats(db)
            return db_seen[key]
        common_mod.stats_from_db = cached_stats
        self.cached_stats = cached_stats
        gfn2 = bench._assert_gfn2()
        bench._assert_gfn2 = lambda: gfn2
        # one validity ledger per lane, as in the D3 worker: a ligand whose GFN2 refinement failed for an earlier
        # parent of this lane (and now comes from _GFN2_CACHE) is still flagged
        self.validity = ev_mod.EmbeddingValidity(engine)
        ev_mod.EmbeddingValidity = lambda e: self.validity
        self.validity_factory = ev_mod.EmbeddingValidity
        # OOM retries inside the runners call torch.cuda.empty_cache(); count them per parent
        self.n_empty_cache = 0
        orig_empty = torch.cuda.empty_cache

        def empty_cache():
            self.n_empty_cache += 1
            return orig_empty()
        torch.cuda.empty_cache = empty_cache
        self._orig_empty = orig_empty

    def gpu_begin(self):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def gpu_end(self) -> tuple:
        """(peak reserved, peak allocated) bytes of the parent. Reserved includes the caching allocator's slack
        (the runners allow 80 % of the card per process); allocated is what the tensors needed."""
        if self.device != "cuda":
            return None, None
        peak = (int(self.torch.cuda.max_memory_reserved()), int(self.torch.cuda.max_memory_allocated()))
        gc.collect()
        self._orig_empty()  # give cached blocks back so other lanes can use them
        return peak


class BenchBackend(_Common):
    def __init__(self, plan: dict, lane: int):
        super().__init__(plan, lane)
        from tools.jgen.bench import run as bench
        # import-time profile = the first arm's flags, as bench.main sets it for a fresh process (and as the D3
        # worker does); later arms switch their flags inside bench.main exactly as in a sequential run
        bench._setup_env(plan["arms"][0].get("flags", {}))
        import torch
        from jacobik.builder import engine as E
        from jacobik.core import energy
        from tools.jdev.budget import common
        from tools.jdev import multiseed
        from tools.jgen import embedding_validity
        from tools.jgen.bench import pool_export
        from tools.jgen.g3c0_distortion import start_digests
        self.bench, self.E, self.multiseed = bench, E, multiseed
        self.install_common(energy, bench, common, embedding_validity, E)
        if plan.get("deterministic", True):  # as dev/tools/jgen/c2/driver.py and the D3 worker; off = plain `bench run`
            torch.use_deterministic_algorithms(True)
        self.arms = plan["arms"]
        self.names = [a["arm"] for a in self.arms]
        spec = [dict(arm=a["arm"], flags=a.get("flags", {}), seed_budget=a.get("seed_budget", "three_w16"),
                     out=str(self.lane_dir / a["arm"]), xyz_dir=str(a.get("xyz_dir") or self.arm_dir[a["arm"]] / "xyz"),
                     **{k: v for k, v in a.items() if k not in ("arm", "flags", "seed_budget", "out", "xyz_dir")})
                for a in self.arms]
        self.spec_path = self.lane_dir / "arms.json"
        self.spec_path.write_text(json.dumps(spec, indent=1))
        # --- the C-2b driver's observation hooks (dev/tools/jgen/c2/driver.py), per parent instead of per run
        self.times, self.row_steps, self.digests, self.beam_times = {}, {}, {}, {}
        self.state = {"arm": None}
        builders = {}
        sync = (lambda: torch.cuda.synchronize()) if self.device == "cuda" else (lambda: None)

        def add_phase(code, key, value):
            self.times.setdefault(code, {}).setdefault(key, []).append(value)
        orig_prefetch = bench._prefetch_parent
        self.orig_prefetch = orig_prefetch
        self.orig_ms = {n: getattr(multiseed, n) for n in ("prepare", "seed_rows")}

        def prefetch(code, *a, **k):
            self.mark("prefetch")
            t = time.monotonic()
            res = orig_prefetch(code, *a, **k)
            add_phase(code, "prefetch", time.monotonic() - t)
            if res[1] is not None:
                builders[id(res[1])] = code
            return res
        bench._prefetch_parent = prefetch
        bench._c2_arm_observer = lambda name: self.state.update(arm=name)
        for name in ("prepare", "seed_rows"):
            original = getattr(multiseed, name)

            def wrapped(builder, *a, _name=name, _original=original, **k):
                code = builders.get(id(builder))
                self.mark(_name)
                sync()
                t = time.monotonic()
                try:
                    res = _original(builder, *a, **k)
                    if _name == "seed_rows":
                        self.digests.setdefault(code, {})[self.state["arm"] or "shared"] = start_digests(res[1])
                    return res
                finally:
                    sync()
                    add_phase(code, _name if self.state["arm"] is None else self.state["arm"] + "_" + _name,
                              time.monotonic() - t)
            setattr(multiseed, name, wrapped)
        orig_beam = E.ComplexBuilder._beam_search_all_candidates

        def beam(builder, *a, **k):
            res = orig_beam(builder, *a, **k)
            code = builders.get(id(builder))
            self.beam_times.setdefault(code, {}).setdefault(self.state["arm"] or "shared", []).append(
                getattr(builder, "_c2_beam_wall_s", 0.0))
            return res
        E.ComplexBuilder._beam_search_all_candidates = beam
        orig_joint = E.ComplexBuilder._run_joint_ik

        self.packer = None
        if plan.get("pack_arms"):
            from .armpack import ArmPacker
            self.packer = ArmPacker(self, orig_joint)

        def joint(builder, *a, **k):
            code = builders.get(id(builder))
            B, T = int(a[5][0].shape[0]), int(a[11])
            self.mark(f"solve {self.state['arm']} B={B}")
            res = self.packer.lookup(builder, a) if self.packer is not None else None
            if res is None:
                res = orig_joint(builder, *a, **k)
            self.row_steps.setdefault(code, {}).setdefault(self.state["arm"], []).append(dict(B=B, T=T, row_steps=B * T))
            return res
        E.ComplexBuilder._run_joint_ik = joint
        orig_export = pool_export.export_pool

        def export_pool(*a, **k):
            self.mark("export")
            t = time.monotonic()
            try:
                return orig_export(*a, **k)
            finally:
                self.add("export_s", time.monotonic() - t)
        pool_export.export_pool = export_pool

    def run_parent(self, code: str) -> dict:
        bench, E = self.bench, self.E
        self.stage.clear()
        self.state["arm"] = None
        for d in (self.times, self.row_steps, self.digests, self.beam_times):
            d.pop(code, None)
        root = logging.getLogger()
        handlers = list(root.handlers)
        orig_refine = E._gfn2_refine
        argv = ["--arms-spec", str(self.spec_path), "--panel", "named", "--only", code, "--device", self.device,
                "--seeds", ",".join(map(str, self.seeds)), "--xyz-compression", "xz"]
        if self.plan.get("cpu_identity"):
            argv.append("--cpu-identity")
        error = None
        t0 = time.monotonic()
        self.pack_info = None
        if self.packer is not None:
            self.mark("pack presolve")
            self.pack_info = self.packer.presolve(code)
        try:
            bench.main(argv)
        except BaseException as exc:  # noqa: BLE001 - a parent never takes the lane down
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
        finally:
            E._gfn2_refine = orig_refine
            for h in list(root.handlers):
                if h not in handlers:
                    root.removeHandler(h)
        wall = time.monotonic() - t0
        self.failed_records = []
        recs = self._finish_records(code, wall, error)
        return dict(error=error, wall_s=wall, records=recs, failed_records=self.failed_records)

    def _finish_records(self, code, wall, error) -> int:
        """c2_timing / c2_identity exactly as dev/tools/jgen/c2/driver.py computes them, plus run_timing; then move the
        records from the lane directory to OUT/<arm>/records (one rename per file)."""
        phases = self.times.get(code, {})
        prep = sum(phases.get("prefetch", [])) + sum(phases.get("prepare", []))
        nseeds = len(self.seeds)
        n = 0
        for arm in self.arms:
            name = arm["arm"]
            seedtime = sum(phases.get(name + "_seed_rows", []))
            if name in ("C", "T"):
                seedtime += sum(phases.get("seed_rows", []))
            src_dir = self.lane_dir / name / "records"
            dst_dir = self.arm_dir[name] / "records"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for s in self.seeds:
                p = src_dir / f"{code}_{s}.json"
                if not p.exists():
                    r = dict(code=code, seed=s, arm=name, status="failed",
                             reason=error or "lane: bench wrote no record", n_rows=0, n_finite=0)
                else:
                    r = json.loads(p.read_text())
                ownprep = sum(phases.get(name + "_prepare", []))
                total = seedtime + (r.get("wall_solve_s") or 0.0)
                is_wide = arm.get("seed_budget", "three_w16") == "single_w48"
                e2e = prep + ownprep + total / nseeds if is_wide else (prep + ownprep + total) / nseeds
                r["c2_timing"] = dict(shared_prepare_s=prep, seed_rows_s=seedtime, end_to_end_s=e2e,
                                      beam_wall_s=sum(self.beam_times.get(code, {}).get(
                                          name, self.beam_times.get(code, {}).get("shared", []))),
                                      allocation="S1 full prepare per independent wide seed; C/ordinary prepare once "
                                                 "per three-seed experiment")
                r["c2_identity"] = dict(seed_budget=arm.get("seed_budget", "three_w16"), W=48 if is_wide else 16,
                                        initial_digest=self.digests.get(code, {}).get(
                                            name, self.digests.get(code, {}).get("shared")),
                                        row_steps=self.row_steps.get(code, {}).get(name, []))
                r["run_timing"] = dict(parent_wall_s=round(wall, 3), lane=self.lane, pid=os.getpid(),
                                       pack_arms=self.pack_info,
                                       phases={k: round(sum(v), 4) for k, v in phases.items()},
                                       mace_s=round(self.stage.get("mace_s", 0.0), 4),
                                       export_s=round(self.stage.get("export_s", 0.0), 4),
                                       note="parent-level stages, shared by every arm and seed of the parent")
                if r.get("status") != "ok":
                    self.failed_records.append(dict(arm=name, seed=s, status=r.get("status"),
                                                    reason=str(r.get("reason") or "")[:200]))
                tmp = dst_dir / f".{code}_{s}.json.tmp"
                tmp.write_text(json.dumps(r))
                os.replace(tmp, dst_dir / f"{code}_{s}.json")
                if p.exists():
                    p.unlink()
                n += 1
        return n


def _apply_isomer_overrides(definitions, arms, default_for):
    """Apply runner arm flags to the screen's definitions; do not silently discard them."""
    names = {key for arm in arms for key in arm.get("flags", {})}
    for arm in arms:
        name = arm["arm"]
        flags = {**definitions[name], **arm.get("flags", {})}
        for key in names - flags.keys():
            flags[key] = default_for(key)
        definitions[name] = flags


class IsomerBackend(_Common):
    """The isomer screen's own main(), one parent per call, in a persistent process (its script is not modified)."""

    def __init__(self, plan: dict, lane: int):
        super().__init__(plan, lane)
        import importlib.util
        script = Path(plan["isomer_script"]).resolve()
        sys.path.insert(0, str(script.parent))
        import fast_patches  # noqa: F401  - installs once below, not once per parent
        once = {"done": False}
        orig_install = fast_patches.install

        def install_once():
            if not once["done"]:
                once["done"] = True
                return orig_install()
        fast_patches.install = install_once
        spec = importlib.util.spec_from_file_location("run_i_screen_lane", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _apply_isomer_overrides(mod.ARM_DEFINITIONS, plan["arms"], mod.P.get)
        self.mod = mod
        from jacobik.core import energy
        from jacobik.builder import engine as E
        from tools.jgen import embedding_validity
        from tools.jdev.budget import common
        self.E = E
        self.install_common(energy, mod.bench, common, embedding_validity, E)
        mod.MACEPolarEvaluator = self.mace_factory
        mod.stats_from_db = self.cached_stats
        mod.EmbeddingValidity = self.validity_factory
        mod.gpu_lock = lambda *a, **k: nullcontext()  # the orchestrator holds the lock once for all lanes
        self.arms = plan["arms"]

    def run_parent(self, code: str) -> dict:
        self.stage.clear()
        E = self.E
        root = logging.getLogger()
        handlers = list(root.handlers)
        orig_refine = E._gfn2_refine
        argv = ["--out", str(self.lane_dir), "--arms", ",".join(a["arm"] for a in self.arms),
                "--seeds", ",".join(map(str, self.seeds)), "--only", code, "--device", self.device]
        error = None
        t0 = time.monotonic()
        try:
            self.mod.main(argv)
        except BaseException as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
        finally:
            E._gfn2_refine = orig_refine
            for h in list(root.handlers):
                if h not in handlers:
                    root.removeHandler(h)
        wall = time.monotonic() - t0
        n = 0
        failed: list = []
        tim = self.lane_dir / "timing.json"
        parent_timing = None
        if tim.exists():
            try:
                parent_timing = json.loads(tim.read_text()).get("parents", {}).get(code)
            except ValueError:
                pass
            (self.out / "_timing").mkdir(parents=True, exist_ok=True)
            (self.out / "_timing" / f"{code}_s{'-'.join(map(str, self.seeds))}.json").write_text(json.dumps(parent_timing))
        lane_prefix, out_prefix = str(self.lane_dir), str(self.out)
        for arm in self.arms:
            name = arm["arm"]
            xsrc = self.lane_dir / name / "xyz" / code
            xdst = self.out / name / "xyz" / code
            if xsrc.exists():  # file by file: other seeds of this parent may come from other lanes
                xdst.mkdir(parents=True, exist_ok=True)
                for f in list(xsrc.iterdir()):
                    if f.suffix == ".json":
                        txt = f.read_text()
                        if lane_prefix in txt:
                            f.write_text(txt.replace(lane_prefix, out_prefix))
                    os.replace(f, xdst / f.name)
            dst_dir = self.arm_dir[name] / "records"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for s in self.seeds:
                p = self.lane_dir / name / "records" / f"{code}_{s}.json"
                if p.exists():
                    txt = p.read_text().replace(lane_prefix, out_prefix)
                    r = json.loads(txt)
                    p.unlink()
                else:
                    r = dict(code=code, seed=s, arm=name, status="failed",
                             reason=error or "lane: screen wrote no record", n_rows=0, n_finite=0)
                r["run_timing"] = dict(parent_wall_s=round(wall, 3), lane=self.lane, pid=os.getpid(),
                                       mace_s=round(self.stage.get("mace_s", 0.0), 4), screen_timing=parent_timing)
                if r.get("status") != "ok":
                    failed.append(dict(arm=name, seed=s, status=r.get("status"), reason=str(r.get("reason") or "")[:200]))
                tmp = dst_dir / f".{code}_{s}.json.tmp"
                tmp.write_text(json.dumps(r))
                os.replace(tmp, dst_dir / f"{code}_{s}.json")
                n += 1
        return dict(error=error, wall_s=wall, records=n, failed_records=failed)


def _mark_failed_records(res: dict) -> dict:
    """A parent is ok only if no exception escaped AND every record it wrote has status "ok". A bench record can be
    "failed" (e.g. GFN2 embed fallback) without any exception reaching the lane (audit_wrapup_2026-11-10, R)."""
    bad = res.get("failed_records") or []
    if bad and res.get("error") is None:
        res["error"] = (f"{len(bad)} record(s) not ok: " + "; ".join(
            f"{b['arm']}/s{b['seed']} {b['status']}: {b['reason']}" for b in bad[:3]))[:400]
    return res


BACKENDS = {"bench": BenchBackend, "isomer": IsomerBackend}


def serve(plan: dict, lane: int):
    proto = sys.stdout
    sys.stdout = sys.stderr
    t0 = time.monotonic()
    try:
        backend = BACKENDS[plan.get("backend", "bench")](plan, lane)
        gpu_mib = None
        if backend.device == "cuda":
            free, total = backend.torch.cuda.mem_get_info()
            gpu_mib = (total - free) / 2**20  # whole device; the orchestrator reads per-process use from nvidia-smi
        proto.write(json.dumps(dict(ready=True, pid=os.getpid(), startup_s=time.monotonic() - t0,
                                    rss=_vm_hwm(), gpu_device_used_mib=gpu_mib)) + "\n")
        proto.flush()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        proto.write(json.dumps(dict(ready=False, error=f"{type(exc).__name__}: {str(exc)[:400]}")) + "\n")
        proto.flush()
        return 2
    import threading
    lock = threading.Lock()

    def emit(obj):
        with lock:
            proto.write(json.dumps(obj) + "\n")
            proto.flush()

    def heartbeat():
        while True:
            time.sleep(60)
            c = backend.cur
            if c["code"] is not None:
                emit(dict(hb=True, code=c["code"], stage=c["stage"], elapsed_s=round(time.monotonic() - c["t0"]),
                          rss_gb=round(_rss() / 2**30, 2)))
    threading.Thread(target=heartbeat, daemon=True).start()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("stop"):
            break
        code = req["code"]
        backend.seeds = [int(x) for x in (req.get("seeds") or plan["seeds"])]
        _reset_hwm()
        backend.n_empty_cache = 0
        backend.gpu_begin()
        backend.cur.update(code=code, stage="start", t0=time.monotonic())
        try:
            res = backend.run_parent(code)
        except BaseException as exc:  # noqa: BLE001
            res = dict(error=f"lane: {type(exc).__name__}: {str(exc)[:400]}", wall_s=None, records=0)
        g_res, g_alloc = backend.gpu_end()
        _mark_failed_records(res)
        res.update(code=code, item=req.get("item", code), seeds=backend.seeds, ok=res.get("error") is None, peak_rss=_vm_hwm(),
                   gpu_peak_reserved=g_res, gpu_peak_allocated=g_alloc,
                   oom_retries=backend.n_empty_cache, stages=dict(backend.stage))
        backend.cur.update(code=None, stage=None)
        emit(res)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.run.lane")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--lane", type=int, required=True)
    a = ap.parse_args(argv)
    return serve(json.loads(Path(a.plan).read_text()), a.lane)


if __name__ == "__main__":
    raise SystemExit(main())
