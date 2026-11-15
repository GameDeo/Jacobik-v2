"""Quality levels: python -m tools.check {ws,pm-quick,pm-full} [options].

Options only pick paths, worker counts, and --only (a quick subset while developing). They never change what a level
checks. Exit status is non-zero on any failing stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .common import EXPECTED_FINGERPRINT, FASTCHECK_OUT, REPO, Report, base_env, main_checkout


def fingerprint_stage():
    import subprocess
    code = "from jacobik.core import profile as P; print(P.snapshot()['fingerprint'])"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=base_env(), capture_output=True, text=True)
    fp = (r.stdout.strip().splitlines() or [""])[-1]
    return fp == EXPECTED_FINGERPRINT, f"fingerprint {fp or 'n/a'} (expected {EXPECTED_FINGERPRINT})", {"fingerprint": fp}


def golden_stage(golden_dir: Path, jobs: int, groups: list[str], out: Path):
    from .golden_par import run_parallel
    res = run_parallel(golden_dir, jobs, groups, out)
    ok = not res["unexpected"] and not res["failed"] and res["arrays_ref"] > 0
    summary = f"{len(res['unexpected'])}/{res['arrays_ref']} arrays differ unexpectedly"
    if res["known_pending"]:
        summary += (f"; {len(res['known_pending'])} KNOWN PENDING (owner re-record): "
                    + "; ".join(res["known_pending"]))
    if res["failed"]:
        summary += f"; failed: {res['failed']}"
    return ok, summary, dict(groups=res["groups"], unexpected=res["unexpected"][:50],
                             known_pending=res["known_pending"], children=res["children"])


def pytest_stage(out: Path, shards: int, only: list[str] | None):
    from .pytest_shard import run
    res = run(out, shards, only)
    ok = not res["failing"] and res["counts"].get("passed", 0) > 0
    c = res["counts"]
    summary = f"{c.get('passed', 0)} passed, {c.get('failure', 0) + c.get('error', 0)} failed, {c.get('skipped', 0)} skipped"
    return ok, summary, dict(counts=c, failing=res["failing"], shards=res["shards"])


def ws(args) -> int:
    out = Path(args.out_dir) if args.out_dir else FASTCHECK_OUT / "ws"
    rep = Report("ws", out)
    only = set(args.only.split(",")) if args.only else {"fingerprint", "golden", "pytest"}
    golden_dir = Path(os.environ.get("JACOBIK_GOLDEN_DIR") or (REPO / "research/data/golden" if (REPO / "research/data/golden").exists() else main_checkout() / "research/data/golden"))
    from .golden_par import GROUPS
    from .common import resource_plan
    groups = list(GROUPS)
    plan = resource_plan(args.jobs, args.shards, max_rss_gb=args.max_rss_gb)
    print("ws plan: " + json.dumps(plan), flush=True)
    if not plan["ok"]:
        print("ws refused: not enough free memory/CPU for one golden child and one pytest shard "
              "(another heavy job is running?)", flush=True)
        return 2
    args.jobs, args.shards = plan["golden_jobs"], plan["pytest_shards"]
    rep.notes.append("resource plan " + json.dumps(plan))
    if "fingerprint" in only:
        rep.run("fingerprint", fingerprint_stage)
    # golden and pytest at once: golden's critical path is one solve (aqua_La3_8H2O), so pytest fills the other cores
    par = []
    if "golden" in only:
        par.append(("golden (parallel)", lambda: golden_stage(golden_dir, args.jobs, groups, out / "golden")))
    if "pytest" in only:
        par.append(("pytest (sharded)", lambda: pytest_stage(out / "pytest", args.shards, None)))
    if par:
        rep.run_parallel(par)
    rep.notes.append(f"golden dir {golden_dir}; jobs {args.jobs}; shards {args.shards}")
    return rep.finish()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="level", required=True)
    for name in ("ws", "pm-quick", "pm-full"):
        q = sub.add_parser(name)
        q.add_argument("--out-dir", default=None, help="report and output directory (default research/data/infra/fastcheck/<level>)")
        q.add_argument("--jobs", type=int, default=7, help="concurrent golden processes (results do not depend on it)")
        q.add_argument("--shards", type=int, default=4, help="pytest shards")
        q.add_argument("--only", default="", help="comma list of stages to run (quick subset while developing)")
        q.add_argument("--workers", type=int, default=3, help="pm-quick dev suite: max lanes")
        q.add_argument("--max-rss-gb", type=float, default=None, help="cap on the level's total RSS")
        if name == "pm-full":
            q.add_argument("--shape", choices=["c2b", "isomer"], default="c2b", help="gate shape (120 x 5 x 3 / 120 x 4 x 3)")
            q.add_argument("--qos", default=None, help="the QOS the owner approved for this job")
            q.add_argument("--approved", action="store_true", help="the owner's go for this billed job")
            q.add_argument("--test-only", action="store_true", help="sbatch --test-only on the generated script")
            q.add_argument("--no-mps", action="store_true", help="runners without MPS (default: a private MPS daemon per GPU)")
            q.add_argument("--no-pack-arms", action="store_true", help="c2b: solve arms separately")
            q.add_argument("--remote-out", default="/pscratch/sd/a/achen310/jacobik/runs",
                           help="absolute output root on Perlmutter (SBATCH -o does not expand variables)")
    args = p.parse_args(argv)
    if args.level == "ws":
        return ws(args)
    if args.level == "pm-quick":
        from .pm_quick import run as run_pm_quick
        return run_pm_quick(args)
    from .pm_full import run as run_pm_full
    return run_pm_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
