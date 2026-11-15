"""Run Architector jobs from prepared inputs, one subprocess per job, with per-job accounting and resume (bench env).

No jacobik imports. Inputs are the root-env exports (research/bench/tools/architector/export_inputs.py) and, for the
challenge suite, catalog.make_input (native mode only; matched needs a crystal geometry label the case file lacks).

Per job (run_dir/structures/<panel>/<case>/<mode>/seed_<n>/):
  input.json, result.json (status, structures, per-structure scalars, Architector optimiser step counts)
  arch_init_<k>.xyz (unrelaxed assembly), arch_final_<k>.xyz (Architector's xTB-relaxed output)
Accounting per job (job.json): wall, CPU (user+sys of the child), peak RSS (wait4 rusage), exit status, timeout flag.
Timeouts kill the job's process group by its PID (the child leads its own session). Finished jobs are skipped on rerun.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[0]  # research/bench/benchmarks
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS")


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- worker (runs inside the Architector env)
def _mol2_atoms(text):
    """(symbols, positions) from the ATOM block of an Architector mol2 string (its unrelaxed assembly)."""
    from ase.data import chemical_symbols
    syms, pos, on = [], [], False
    for line in text.splitlines():
        if line.startswith("@<TRIPOS>"):
            on = line.strip() == "@<TRIPOS>ATOM"
            continue
        if on and line.strip():
            f = line.split()
            name = re.sub(r"[^A-Za-z]", "", f[1])
            el = None
            for cand in (name[:2].capitalize(), name[:1].upper()):
                if cand in chemical_symbols[1:]:
                    el = cand
                    break
            if el is None:
                raise ValueError(f"unknown element in mol2 atom name {f[1]!r}")
            syms.append(el)
            pos.append([float(f[2]), float(f[3]), float(f[4])])
    return syms, pos


def _install_step_counter():
    """Record ASE optimiser steps per relaxation without changing the algorithm (subclass of the class Architector uses)."""
    import architector.io_calc as ioc
    from ase.optimize import LBFGSLineSearch
    records = []
    if os.environ.get("ARCH_STOCK") == "1":  # check mode: stock Architector, no instrumentation
        return records

    class CountingLBFGS(LBFGSLineSearch):
        def run(self, fmax=0.05, steps=100000000):
            conv = super().run(fmax=fmax, steps=steps)
            records.append({"nsteps": int(getattr(self, "nsteps", -1)), "converged": bool(conv)})
            return conv

    ioc.LBFGSLineSearch = CountingLBFGS
    return records


def worker(input_path: Path) -> int:
    import numpy as np
    from ase import Atoms
    from ase.io import write as ase_write
    jobdir = input_path.parent
    t0 = time.perf_counter()
    result = {"status": "exception", "structures": [], "error": None}
    try:
        from architector import build_complex
        import importlib.metadata as md
        steps = _install_step_counter()
        spec = json.loads(input_path.read_text())
        t_build = time.perf_counter()
        out = build_complex(spec)
        result["build_seconds"] = time.perf_counter() - t_build
        if not isinstance(out, dict):
            raise TypeError(f"Expected dict, got {type(out).__name__}")
        for k, (key, row) in enumerate(out.items()):
            entry = {"rank": k, "upstream_key": str(key), "files": {}, "scalars": {}}
            scal = {}
            for name, val in row.items():
                if name in ("ase_atoms", "inputDict", "mol2string", "init_mol2string", "full_complex_class"):
                    continue
                if hasattr(val, "item"):
                    try:
                        val = val.item()
                    except (ValueError, TypeError):
                        continue
                if isinstance(val, bool) or isinstance(val, str) or val is None:
                    scal[name] = val
                elif isinstance(val, int):
                    scal[name] = val
                elif isinstance(val, float) and np.isfinite(val):
                    scal[name] = val
            entry["scalars"] = scal
            atoms = row.get("ase_atoms")
            if atoms is not None:
                fn = f"arch_final_{k:03d}.xyz"
                ase_write(str(jobdir / fn), atoms, format="xyz")
                entry["files"]["arch_final"] = fn
                entry["final_natoms"] = len(atoms)
            init = row.get("init_mol2string")
            if init:
                s, p = _mol2_atoms(init)
                fn = f"arch_init_{k:03d}.xyz"
                ase_write(str(jobdir / fn), Atoms(s, positions=np.asarray(p, float)), format="xyz")
                entry["files"]["arch_init"] = fn
                entry["init_natoms"] = len(s)
            result["structures"].append(entry)
        result["opt_steps"] = list(steps)
        # Architector writes 10000 (eV) when its own xTB call or relaxation raises; count them, never hide them.
        result["sentinel_energy_count"] = sum(1 for e in result["structures"]
                                              if e["scalars"].get("energy") == 10000)
        result["architector_version"] = md.version("architector")
        result["status"] = "returned_structures" if out else "empty_output"
    except Exception as exc:  # the worker records the failure; it never hides it
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    result["worker_seconds"] = time.perf_counter() - t0
    _write_json(jobdir / "result.json", result)
    return 0 if result["status"] == "returned_structures" else 1


# ---------------------------------------------------------------- driver
def _spawn_and_wait(cmd, cwd, env, timeout_s, logpath):
    """Run one job. Returns dict(exit_code, timed_out, wall_s, cpu_s, peak_rss_kb). Kills the job's group on timeout."""
    t0 = time.perf_counter()
    with open(logpath, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        pid = proc.pid
        timed_out = False
        while True:
            wpid, status, ru = os.wait4(pid, os.WNOHANG)
            if wpid == pid:
                break
            if time.perf_counter() - t0 > timeout_s:
                timed_out = True
                try:
                    os.killpg(pid, signal.SIGKILL)  # group led by this job's PID
                except ProcessLookupError:
                    pass
                _, status, ru = os.wait4(pid, 0)
                break
            time.sleep(0.5)
    code = os.waitstatus_to_exitcode(status)
    return dict(exit_code=code, timed_out=timed_out, wall_s=time.perf_counter() - t0,
                cpu_s=ru.ru_utime + ru.ru_stime, peak_rss_kb=ru.ru_maxrss)


def _env(scratch):
    env = os.environ.copy()
    env.update({v: "1" for v in THREAD_VARS})
    env.update({v: str(scratch) for v in ("TMPDIR", "TMP", "TEMP", "RUNKIT_SELFCHECK_DIR")})
    return env


def run_job(job, workers_timeout):
    """job: dict(panel, case, mode, seed, spec, run_dir). Skips a job whose job.json is complete (resume)."""
    rd = Path(job["run_dir"])
    jdir = rd / "structures" / job["panel"] / job["case"] / job["mode"] / f"seed_{job['seed']}"
    logp = rd / "logs" / job["panel"] / job["case"] / job["mode"] / f"seed_{job['seed']}.log"
    jobfile = jdir / "job.json"
    if jobfile.exists():
        return json.loads(jobfile.read_text())
    jdir.mkdir(parents=True, exist_ok=True)
    logp.parent.mkdir(parents=True, exist_ok=True)
    scratch = rd / "scratch" / job["panel"] / job["case"] / job["mode"] / f"seed_{job['seed']}"
    scratch.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(job["spec"]))
    spec.setdefault("parameters", {})["temp_prefix"] = str(scratch) + os.sep
    spec["parameters"]["seed"] = int(job["seed"])
    spec["parameters"]["n_conformers"] = int(job["n_conformers"])
    spec["parameters"]["n_symmetries"] = max(10, int(job["n_conformers"]))
    inp = jdir / "input.json"
    _write_json(inp, spec)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(inp)]
    acc = _spawn_and_wait(cmd, str(jdir), _env(scratch), workers_timeout, logp)
    res = jdir / "result.json"
    status = "timeout" if acc["timed_out"] else "worker_crash"
    n = 0
    err = None
    if res.exists():
        r = json.loads(res.read_text())
        status = r["status"]
        n = len(r.get("structures", []))
        err = r.get("error")
        if status == "returned_structures" and n == 0:
            status = "empty_output"
    sentinel = 0
    if res.exists():
        sentinel = int(json.loads(res.read_text()).get("sentinel_energy_count", 0))
    rec = dict(panel=job["panel"], case=job["case"], mode=job["mode"], seed=int(job["seed"]),
               n_conformers=int(job["n_conformers"]), status=status, n_structures=n, error=err,
               sentinel_energy_count=sentinel,
               exit_code=acc["exit_code"], timed_out=acc["timed_out"], wall_s=round(acc["wall_s"], 3),
               cpu_s=round(acc["cpu_s"], 3), peak_rss_mb=round(acc["peak_rss_kb"] / 1024.0, 1),
               directory=str(jdir.relative_to(rd)), log=str(logp.relative_to(rd)),
               timeout_s=workers_timeout, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    _write_json(jobfile, rec)
    print(f"{job['panel']}/{job['case']}/{job['mode']}/seed_{job['seed']}: {status} wall={rec['wall_s']}s "
          f"rss={rec['peak_rss_mb']}MB n={n}", flush=True)
    return rec


def build_jobs(args):
    rd = Path(args.run_dir)
    jobs = []
    modes = args.mode or ["native"]
    seeds = args.seed or [11, 12, 13]
    for path in sorted((rd / "inputs").glob("*/*.json")):
        panel = path.parent.name
        if panel != "doe_aqua":
            continue  # tmQM panels are out of scope (Amendment 1)
        rec = json.loads(path.read_text())
        if args.case and rec["case"] not in args.case:
            continue
        if args.panel and panel not in args.panel:
            continue
        for mode in modes:
            body = rec["modes"].get(mode)
            if body is None or body["status"] != "direct":
                continue
            for seed in seeds:
                jobs.append(dict(panel=panel, case=rec["case"], mode=mode, seed=seed, spec=body["spec"],
                                 run_dir=str(rd), n_conformers=args.n_conformers))
    if args.catalog_suite:
        cat = importlib_catalog()
        cases, ligands = cat.load_catalog(args.catalog_suite)
        for case in cases:
            if args.case and case["id"] not in args.case:
                continue
            try:
                spec_seed = {seed: cat.make_input(case, ligands, seed, args.n_conformers, True, None) for seed in seeds}
            except cat.UnsupportedInput as exc:
                cap = rd / "capability_catalog.jsonl"
                with cap.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"panel": args.catalog_suite, "case": case["id"], "mode": "all",
                                         "status": exc.status, "reason": str(exc)}) + "\n")
                continue
            for seed in seeds:
                # Crystal-geometry matched mode needs a geometry label; the challenge/baseline case files do not carry one.
                if "native" in modes:
                    jobs.append(dict(panel=args.catalog_suite, case=case["id"], mode="native", seed=seed,
                                     spec=spec_seed[seed], run_dir=str(rd), n_conformers=args.n_conformers))
    return jobs


def importlib_catalog():
    import catalog as cat  # scripts/catalog.py (offline, no jacobik)
    return cat


class _Gate:
    """Admits at most `workers` concurrent jobs, or `low_mem_workers` while MemAvailable is below `low_mem_mb`."""

    def __init__(self, workers, low_mem_workers=2, low_mem_mb=1500):
        self.workers, self.low_w, self.low_mb = max(1, workers), max(1, low_mem_workers), low_mem_mb
        self.active, self.lock = 0, threading.Lock()

    @staticmethod
    def mem_available_mb():
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
        return float("inf")

    def run(self, fn, job, timeout):
        while True:
            with self.lock:
                low = self.mem_available_mb() < self.low_mb
                cap = min(self.workers, self.low_w) if low else self.workers
                if self.active < cap:
                    self.active += 1
                    break
            time.sleep(10)
        try:
            return fn(job, timeout)
        finally:
            with self.lock:
                self.active -= 1


def main_driver(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--panel", action="append")
    ap.add_argument("--case", action="append")
    ap.add_argument("--mode", action="append", choices=["native", "matched"], help="Default native (Amendment 1)")
    ap.add_argument("--seed", action="append", type=int, help="Default 11, 12, 13")
    ap.add_argument("--n-conformers", type=int, default=1, help="Architector default (parameters n_conformers=1)")
    ap.add_argument("--catalog-suite", choices=["baseline", "challenge"], default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    jobs = build_jobs(args)
    print(f"{len(jobs)} jobs", flush=True)
    if args.dry_run:
        for j in jobs:
            print(j["panel"], j["case"], j["mode"], j["seed"])
        return 0
    recs = []
    gate = _Gate(args.workers, low_mem_workers=2, low_mem_mb=1500)
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(gate.run, run_job, j, args.timeout) for j in jobs]
        for f in cf.as_completed(futs):
            try:
                recs.append(f.result())
            except Exception as exc:  # a failing job never fails the batch
                print("driver error:", type(exc).__name__, exc, flush=True)
    counts = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("outcomes", counts, flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        raise SystemExit(worker(Path(sys.argv[2]).resolve()))
    raise SystemExit(main_driver())
