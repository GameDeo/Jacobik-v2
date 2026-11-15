"""Unified parent runner: one entry point for every GPU (or CPU) run of parents.

  python -m tools.run panel  --out DIR (--codes A,B | --manifest M.json | --c2-panel) [--arms C,P | --arms-spec JSON]
                             --seeds 21,22,23 [--backend bench|isomer] [--repo CHECKOUT] [--env K=V ...]
                             [--device cuda|cpu] [--lanes N | --max-lanes K] [--max-rss-gb G] [--parent-cap-min M]
                             [--max-wall-min M] [--no-resume] [--yes]
  python -m tools.run submit --spool DIR  <panel options>     queue a job for a running broker
  python -m tools.run broker --spool DIR [--max-lanes K ...]  the long-lived GPU owner: serves every submitted job
  python -m tools.run status --spool DIR
  python -m tools.run estimate <panel options>                wall estimate and lane plan; runs nothing
  python -m tools.run sample --out DIR [--records GLOB] -- <any existing runner command>   telemetry for "before"
  python -m tools.run compare --new DIR/C --ref DIR/C --seeds 21,22,23 [--codes A,B]

Before every run the wall is estimated from recorded per-parent seconds; above 20 minutes `panel` needs --yes.
Records: <arm out>/records/<code>_<seed>.json and <arm out>/xyz/<code>/ exactly as the job's own runner writes them
(bench: c2_timing, c2_identity as the C-2b driver), plus run_timing; OUT/run_summary.json, job_summary.json,
run_results.json, run_events.jsonl (heartbeats, timeouts, admission decisions), telemetry.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from .orchestrator import REPO, Job, Scheduler, estimate, main_checkout, predicted_lanes
from . import resources as R

#: the C-2 arm table (dev/tools/jgen/c2/driver.py); any other arm set comes as --arms-spec
C2_ARMS = dict(C={}, K={"start_kabsch": True}, S1={}, P={"start_kabsch": True, "start_kabsch_pivot": True})
C2_ARMS["F"] = dict(C2_ARMS["P"], start_kabsch_diversity=0.0)
CONFIRM_LIMIT_S = 20 * 60


def locked_codes() -> set:
    p = main_checkout() / "research/bench/benchmarks/locked_exclusions.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text())
    return set(d.get("reserve_all_codes", [])) | set(d.get("motion_validation_codes", []))


COSTS = Path(__file__).with_name("costs.json")


def regenerate_costs() -> dict:
    """Rebuild dev/tools/run/costs.json from the local campaign records (C-2b confirmation, isomer dev_11_13 timings)."""
    import glob
    main = main_checkout()
    bench = {}
    for f in glob.glob(str(main / "research/data/jgen/campaign/C2b/confirmation/C/records/*.json")):
        r = json.loads(Path(f).read_text())
        bench.setdefault(r["code"], []).append(float((r.get("c2_timing") or {}).get("end_to_end_s") or r.get("wall_solve_s") or 0))
    bench = {c: round(sum(v) / len(v), 2) for c, v in bench.items()}
    iso = {}
    for t in glob.glob(str(main / "research/data/jgen/campaign/I/dev_11_13*/**/timing.json"), recursive=True):
        for c, v in json.loads(Path(t).read_text()).get("parents", {}).items():
            iso[c] = max(iso.get(c, 0.0), round(float(v.get("wall_parent_s") or 0), 2))
    doc = json.loads(COSTS.read_text())["_doc"] if COSTS.exists() else ""
    out = dict(_doc=doc, bench_c_per_seed_s=dict(sorted(bench.items())), isomer_parent_s=dict(sorted(iso.items())))
    COSTS.write_text(json.dumps(out, indent=0))
    return dict(bench=len(bench), isomer=len(iso))


def recorded_costs(codes: list[str], backend: str, n_arms: int, n_seeds: int) -> dict:
    """Recorded seconds per whole parent (every arm and seed), from dev/tools/run/costs.json (shipped with the code, so
    Colab and Perlmutter order queues the same way). Bench: C-arm seconds per seed x seeds x arms. Isomer: the
    screen's own wall_parent_s where recorded, else the bench figure x 4 (standalone solves per arm and seed).
    Only orders the queue, sets wall budgets and the estimate; never changes what runs."""
    t = json.loads(COSTS.read_text()) if COSTS.exists() else {}
    per_seed, iso = t.get("bench_c_per_seed_s", {}), t.get("isomer_parent_s", {})
    out = {}
    for c in codes:
        if backend == "isomer" and iso.get(c):
            out[c] = iso[c] * n_seeds * max(1, n_arms) / 12.0
        elif per_seed.get(c):
            out[c] = round(per_seed[c] * n_seeds * max(1, n_arms) * (4.0 if backend == "isomer" else 1.0), 1)
    return out


def _check_db_tables(plan: dict) -> None:
    """Refuse to start when an arm's flags need a DB table (profile.FLAG_TABLES) that the DB does not carry."""
    from jacobik.core import container as dbio
    from .orchestrator import main_checkout

    db = plan["env"].get("JACOBIK_DB") or os.environ.get("JACOBIK_DB")
    path = Path(db) if db else main_checkout(Path(plan["repo"])) / "data/db/constraints_2026-11-12.npz"
    if not path.is_file():
        return  # a missing DB is reported by the lanes / control-cache key with its own message
    info = dbio.runtime_info(path)
    for arm in plan["arms"]:
        missing = dbio.missing_tables(info["tables"], flags=arm.get("flags") or {})
        if missing:
            raise SystemExit(f"arm {arm.get('arm')}: DB {path} lacks tables needed by its flags: {missing}")


def build_plan(a) -> dict:
    if a.manifest:
        spec = json.loads(Path(a.manifest).read_text())
        parents = [dict(code=p["code"]) for p in spec["parents"]]
        seeds = a.seeds or ",".join(map(str, spec.get("seeds", [11])))
    elif a.c2_panel:
        panel = json.loads((REPO / "research/campaigns/c2/C2_PANEL.json").read_text())
        parents = [dict(code=c) for c in panel["codes"]]
        seeds = a.seeds or "11,12,13"
    else:
        parents = [dict(code=c) for c in a.codes.split(",") if c]
        seeds = a.seeds or "11,12,13"
    bad = sorted({p["code"] for p in parents} & locked_codes())
    if bad:
        raise SystemExit(f"refusing locked codes (reserve / motion_validation): {bad}")
    if a.arms_spec:
        txt = Path(a.arms_spec).read_text() if Path(a.arms_spec).exists() else a.arms_spec
        arms = json.loads(txt)
    elif a.backend == "isomer":
        arms = [dict(arm=n) for n in a.arms.split(",")]
    else:
        arms = [dict(arm=n, flags=C2_ARMS[n], seed_budget="single_w48" if n == "S1" else "three_w16")
                for n in a.arms.split(",")]
    seeds_l = [int(s) for s in seeds.split(",")]
    costs = recorded_costs([p["code"] for p in parents], a.backend, len(arms), len(seeds_l))
    for p in parents:
        p["cost"] = costs.get(p["code"])
    env = dict(kv.split("=", 1) for kv in (a.env or []))
    plan = dict(name=a.name or Path(a.out).name, backend=a.backend, device=a.device, out=str(Path(a.out).resolve()),
                seeds=seeds_l, arms=arms, parents=parents, cpu_identity=bool(a.cpu_identity), pack_arms=bool(a.pack_arms), deterministic=not a.no_deterministic,
                pack=bool(a.pack or (a.pack is None and os.environ.get("NERSC_HOST"))),
                repo=str(Path(a.repo).resolve()) if a.repo else str(REPO), env=env, submitted=time.strftime("%Y-%m-%dT%H:%M:%S"))
    plan["retry_timeouts"] = bool(getattr(a, "retry_timeouts", False))
    if getattr(a, "target_overlay", None):
        plan["target_overlay"] = str(Path(a.target_overlay).resolve())
        if getattr(a, "gate_eligibility", None):
            plan["gate_eligibility_sha256"] = hashlib.sha256(Path(a.gate_eligibility).read_bytes()).hexdigest()
    _check_db_tables(plan)
    if getattr(a, "claim_dir", None):
        plan["claim_dir"] = str(Path(a.claim_dir).resolve())
    if a.backend == "isomer":
        # the screen's seeds are independent standalone solves (its contract: bit-identical to
        # ComplexBuilder.solve(seed=s) per arm and seed), so one seed of a parent is a work item of its own
        plan["split_seeds"] = not a.no_split_seeds
        if not a.isomer_script:
            raise SystemExit("--backend isomer needs --isomer-script PATH/run_i_screen.py")
        plan["isomer_script"] = str(Path(a.isomer_script).resolve())
    return plan


def _sel(p, out_required=True):
    p.add_argument("--out", required=out_required)
    p.add_argument("--name", default="")
    p.add_argument("--codes", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--c2-panel", action="store_true", help="the 120-parent C-2 / isomer cal+tail panel")
    p.add_argument("--arms", default="C")
    p.add_argument("--arms-spec", default="", help="JSON (file or string): [{arm, flags, seed_budget, out?}, ...]")
    p.add_argument("--seeds", default="")
    p.add_argument("--backend", choices=["bench", "isomer"], default="bench")
    p.add_argument("--isomer-script", default="")
    p.add_argument("--repo", default="", help="checkout whose code runs (default: this one), e.g. an agent's worktree")
    p.add_argument("--env", action="append", help="K=V for the lanes, e.g. JACOBIK_DB=/path/db.json")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-identity", action="store_true")
    p.add_argument("--pack", action="store_true", default=None,
                   help="pack outputs into OUT/run_pack<N>.tar at the end (default on Perlmutter: inode quota)")
    p.add_argument("--no-pack", dest="pack", action="store_false", help="keep outputs loose (e.g. for a compare)")
    p.add_argument("--no-deterministic", action="store_true",
                   help="bench: leave torch deterministic algorithms off, as plain `tools.jgen bench run` (default on, "
                        "as the C-2b driver and the D3 worker)")
    p.add_argument("--no-split-seeds", action="store_true", help="isomer: keep all seeds of a parent in one work item")
    p.add_argument("--retry-timeouts", action="store_true", help="retry recorded timeout work items on resume")
    p.add_argument("--target-overlay", default=None,
                   help="corrected-target overlay JSON; IDENTITY ONLY: its hash (with the source-review policy) enters the "
                        "control identity, so a corrected-target run never reuses an old control. It is NOT applied to the "
                        "builder inputs: targets come from the production typing (since d5e2f4113497 it reproduces 31/34 "
                        "audited repairs; see research/campaigns/ds/GATE_TARGETS_d5e2f4113497.md)")
    p.add_argument("--gate-eligibility", default=None, help="frozen gate-eligibility JSON; hashed into the control identity")
    p.add_argument("--claim-dir", default=None,
                   help="shared queue: runners pointed at the same DIR (one per GPU, each with the full --codes list) claim "
                        "work items by atomic file create, so a fast GPU takes more and none idles while another is behind. "
                        "Use a fresh DIR per phase")
    p.add_argument("--pack-arms", action="store_true",
                   help="bench: solve every packable arm of a parent (same solve-time settings) in one joint IK batch "
                        "(dev/tools/run/armpack.py); off by default")


def _sched(p):
    p.add_argument("--lanes", type=int, default=None, help="fixed lane count (default: adaptive)")
    p.add_argument("--max-lanes", type=int, default=None)
    p.add_argument("--cpu-cap", type=int, default=None)
    p.add_argument("--cpu-per-lane", type=float, default=1.5)
    p.add_argument("--mem-headroom-gb", type=float, default=2.0)
    p.add_argument("--max-rss-gb", type=float, default=None, help="cap on this run's total RSS")
    p.add_argument("--parent-cap-min", type=float, default=5.0,
                   help="largest per-parent wall budget; an overrun kills the lane and records a timeout (default 5)")
    p.add_argument("--min-budget-min", type=float, default=3.0,
                   help="smallest per-parent wall budget (raise it for --device cpu: recorded costs are GPU seconds)")
    p.add_argument("--max-wall-min", type=float, default=None, help="stop dispatching after M minutes; kill 1 min later")
    p.add_argument("--mps", action="store_true",
                   help="run the lanes under a private CUDA MPS daemon (stopped at the end); off by default")


def _sched_kw(a):
    return dict(max_lanes=a.max_lanes, lanes_fixed=a.lanes, cpu_cap=a.cpu_cap, cpu_per_lane=a.cpu_per_lane,
                mem_headroom_gb=a.mem_headroom_gb, max_rss_gb=a.max_rss_gb, parent_cap_s=a.parent_cap_min * 60, min_budget_s=a.min_budget_min * 60,
                max_wall_s=a.max_wall_min * 60 if a.max_wall_min else None)


class _mps:
    """--mps: a private MPS daemon around the run; always stopped (and checked) at the end, also after errors."""

    def __init__(self, a, out: Path):
        self.on, self.out, self.m = bool(getattr(a, "mps", False)) and a.device == "cuda", Path(out), None

    def __enter__(self):
        if self.on:
            from .mps import MPS
            self.m = MPS().start()
            print(f"MPS daemon started (pipe {self.m.pipe})", flush=True)
        return self

    def __exit__(self, *exc):
        if self.m is not None:
            res = self.m.stop()
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "mps.json").write_text(json.dumps(res, indent=1))
            print(f"MPS daemon stopped: {res['stopped']} (leftover pids killed: {res['left']})", flush=True)
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    from .diagnostic import add_arguments, run_diagnostic
    add_arguments(sub.add_parser("diagnostic", help="run CPU diagnostics over a frozen parent manifest"))
    p = sub.add_parser("panel")
    _sel(p)
    _sched(p)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--yes", action="store_true", help="run even if the estimate exceeds 20 minutes")
    p.add_argument("--control-cache", default="", metavar="ARM",
                   help="reuse ARM's recorded records when its key (code, DB, device, flags, seeds, settings) matches; "
                        "a canary of a few cached parents is rerun and must match (dev/tools/run/control_cache.py)")
    p.add_argument("--control-cache-refresh", action="store_true", help="ignore cached records for ARM and re-record them")
    p.add_argument("--canary", type=int, default=5, help="parents whose control arm is rerun from --repo as the canary")
    p.add_argument("--control-repo", default="",
                   help="clean checkout the control arm runs from and is keyed on (e.g. the master worktree); "
                        "default: --repo")
    e = sub.add_parser("estimate")
    _sel(e)
    _sched(e)
    q = sub.add_parser("submit")
    _sel(q)
    q.add_argument("--spool", required=True)
    b = sub.add_parser("broker")
    b.add_argument("--spool", required=True)
    b.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    b.add_argument("--idle-exit-min", type=float, default=0.0, help="exit after M idle minutes (0: run until stopped)")
    _sched(b)
    sub.add_parser("costs", help="regenerate dev/tools/run/costs.json from local campaign records")
    st = sub.add_parser("status")
    st.add_argument("--spool", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--records", default=None)
    s.add_argument("--period", type=float, default=2.0)
    s.add_argument("command", nargs=argparse.REMAINDER)
    c = sub.add_parser("compare")
    c.add_argument("--new", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--codes", default="")
    c.add_argument("--seeds", required=True)
    c.add_argument("--json", default="")
    a = ap.parse_args(argv)

    if a.cmd == "diagnostic":
        return run_diagnostic(a)
    if a.cmd == "costs":
        print(regenerate_costs())
        return 0
    if a.cmd == "sample":
        cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
        from .telemetry import sample_command
        return sample_command(cmd, Path(a.out), a.period, a.records)
    if a.cmd == "compare":
        from .compare import compare, print_compare
        res = compare(Path(a.new), Path(a.ref), [x for x in a.codes.split(",") if x] or None,
                      [int(x) for x in a.seeds.split(",")], "")
        print_compare(res)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1))
        return 0 if res["identical"] == res["n"] else 1
    if a.cmd in ("broker", "status"):
        spool = Path(a.spool).resolve()
        for d in ("queue", "active", "done", "rejected"):
            (spool / d).mkdir(parents=True, exist_ok=True)
        if a.cmd == "status":
            for d in ("queue", "active", "done", "rejected"):
                print(d, sorted(f.name for f in (spool / d).glob("*.json")))
            ev = spool / "broker" / "run_events.jsonl"
            if ev.exists():
                print("".join(ev.read_text().splitlines(True)[-10:]))
            return 0
        sched = Scheduler(spool / "broker", a.device, **_sched_kw(a))
        with _mps(a, spool / "broker"):
            sched.run(spool=spool, idle_exit_s=a.idle_exit_min * 60)
        return 0
    plan = build_plan(a)
    if a.cmd == "submit":
        spool = Path(a.spool).resolve()
        (spool / "queue").mkdir(parents=True, exist_ok=True)
        f = spool / "queue" / f"{plan['name']}.json"
        if f.exists() or (spool / "active" / f.name).exists():
            raise SystemExit(f"a job named {plan['name']} is already queued or running")
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(plan, indent=1))
        tmp.replace(f)
        print(f"queued {f}")
        return 0
    cc = cc_info = None
    if a.cmd == "panel" and getattr(a, "control_cache", ""):
        cc, cc_info, jobs = _control_cache_jobs(a, plan)
        job = None
    else:
        job = Job(plan, resume=not getattr(a, "no_resume", False)) if a.cmd == "panel" else None
        jobs = [job] if job else [Job(dict(plan, out=str(Path(plan["out"]) / "_estimate")), resume=True)]
    n = predicted_lanes(a.device, a.cpu_cap, a.cpu_per_lane, a.mem_headroom_gb, a.max_rss_gb, a.lanes or a.max_lanes)
    est = estimate(jobs, n)
    print(json.dumps(dict(estimate=est, machine=R.snapshot(),
                          basis="recorded per-parent seconds (C-2b / isomer records, local 2060); lanes predicted from "
                                "free CPU, MemAvailable / 2.8 GB and free GPU memory / 2 GB"), indent=1), flush=True)
    if a.cmd == "estimate":
        return 0
    if est["wall_s"] > CONFIRM_LIMIT_S and not a.yes:
        print(f"estimated wall {est['wall_s'] / 60:.0f} min > 20 min: rerun with --yes to start", flush=True)
        return 3
    sched = Scheduler(plan["out"], plan["device"], **_sched_kw(a))
    for j in jobs:
        sched.add_job(j)
    with _mps(a, Path(plan["out"])):
        summ = sched.run()
    bad = any(js["parents_failed"] or js["not_run"] or js["parents_timeout"] for js in summ["jobs"])
    if cc is not None:
        return _control_cache_finish(cc, cc_info, plan) or (1 if bad else 0)
    return 1 if bad else 0


def _control_cache_jobs(a, plan):
    """Split a panel into jobs around a control cache (see dev/tools/run/control_cache.py). Every arm writes to OUT/<arm>
    as usual; each job keeps its plan and logs in OUT/_jobs/<name>.

    - the control arm runs from --control-repo (default: --repo), and its cache key hashes that checkout;
    - the other arms run from --repo (e.g. an agent's worktree) on every parent;
    - the canary reruns the control arm from --repo on a few parents and must match the control records, so a branch
      whose flags-off behaviour drifted from the control checkout is caught."""
    from .control_cache import ControlCache
    out = Path(plan["out"])
    arms = [dict(x, out=str(Path(x.get("out") or out / x["arm"]))) for x in plan["arms"]]
    plan = dict(plan, arms=arms)
    crepo = str(Path(a.control_repo).resolve()) if getattr(a, "control_repo", "") else plan["repo"]
    cplan = dict(plan, repo=crepo)
    cc = ControlCache(cplan, a.control_cache)
    codes = [p["code"] for p in plan["parents"]]
    cached = [] if a.control_cache_refresh else cc.cached_codes(codes)
    uncached = [c for c in codes if c not in set(cached)]
    canary = cc.canary(codes, a.canary)
    info = dict(arm=cc.name, key=cc.key, usable=cc.usable, reason=cc.inputs.get("reason"), cache_dir=str(cc.dir),
                control_repo=crepo, arm_repo=plan["repo"], n_parents=len(codes), cached=len(cached),
                uncached=uncached, canary=canary)
    print("control cache: " + json.dumps({k: v for k, v in info.items() if k != "uncached"}), flush=True)
    if not cc.usable and crepo != plan["repo"]:
        raise SystemExit(f"control cache: the control checkout {crepo} has no key ({info['reason']}); "
                         "a control from a different checkout is only allowed through the cache")
    others = [x for x in arms if x["arm"] != cc.name]
    ctrl = next(x for x in arms if x["arm"] == cc.name)
    resume = not getattr(a, "no_resume", False)
    jobs = []

    def sub(name, base, parent_codes, sub_arms):
        if not parent_codes or not sub_arms:
            return
        keep = set(parent_codes)
        # sub-jobs never pack (their records live outside their job dirs); the whole OUT is packed at the end
        jobs.append(Job(dict(base, pack=False, pack_root=str(out), wall_arms=arms, name=f"{plan['name']}:{name}", out=str(out / "_jobs" / name),
                             arms=sub_arms, parents=[p for p in plan["parents"] if p["code"] in keep]),
                        resume=resume and name != "canary"))

    if cached:
        n = cc.restore(cached, Path(ctrl["out"]))
        print(f"control cache: restored {n} {cc.name} records for {len(cached)} parents", flush=True)
    if crepo == plan["repo"]:
        sub("uncached", plan, uncached, arms)               # same checkout: control and arms share each parent's lane call
        sub("cached", plan, cached, others)
    else:
        sub("control", cplan, uncached, [ctrl])
        sub("arms", plan, codes, others)
    import uuid
    info["canary_dir"] = str(out / "_canary" / uuid.uuid4().hex / cc.name)
    sub("canary", plan, canary, [dict(ctrl, out=info["canary_dir"])])
    return cc, info, jobs


def _control_cache_finish(cc, info, plan) -> int:
    """Canary check, then add the newly run control parents to the cache. Non-zero if the canary failed."""
    out = Path(plan["out"])
    ctrl_dir = Path(next(x.get("out") or out / x["arm"] for x in plan["arms"] if x["arm"] == cc.name))
    rc = 0
    if info["canary"]:
        from .control_cache import compare_canary_packed
        res = compare_canary_packed(Path(info["canary_dir"]), ctrl_dir, info["canary"], cc.seeds, out)
        info["canary_result"] = dict(identical=res["identical"], n=res["n"],
                                     differ=[r for r in res["rows"] if not r.get("identical")][:20])
        ok = res["identical"] == res["n"]
        print(f"control cache canary ({info['arm_repo']} vs control records): {res['identical']}/{res['n']} top-1 "
              "identical" + ("" if ok else " -> REJECTED: the cache is stale or this checkout's flags-off behaviour "
                             "differs from the control checkout; nothing was stored"), flush=True)
        rc = 0 if ok else 2
    if rc == 0 and cc.usable:
        from .control_cache import store_packed
        info["stored"] = store_packed(cc, info.get("uncached", []), ctrl_dir, out)
        print(f"control cache: stored {info['stored']} new {cc.name} parents under {cc.dir}", flush=True)
    (out / "control_cache.json").write_text(json.dumps(info, indent=1, default=str))
    if plan.get("pack"):
        from .orchestrator import pack_outputs
        info["packed"] = pack_outputs(out)  # after the canary compare and the cache store, which read loose files
    return rc


if __name__ == "__main__":
    sys.exit(main())
