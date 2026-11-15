"""pm-quick on the Perlmutter LOGIN node: ws stages against Perlmutter references, plus the dev suite on the login A100.

Login-node etiquette (owner): nvidia-smi first, nice, <= 32 threads, <= 30 GB, GPU phase <= 10 minutes. No batch
submission of any kind. Outputs go under $SCRATCH/jacobik/runs/. Run from ~/repos/jacobik after jk_update_src.sh.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .common import EXPECTED_FINGERPRINT, REPO, Report, stamp

GPU_PHASE_CAP_S = 600
MAX_THREADS = 32
DEV_MANIFEST = Path("dev/tools/check/DEV_MANIFEST.json")


def _gpu_gate(max_wait_s: int = 300) -> tuple[bool, str]:
    """Refuse (or wait, bounded) when the login GPU is busy with someone else."""
    waited = 0
    while True:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            name, util, used, total = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        except Exception as exc:
            return False, f"nvidia-smi failed: {exc!r}"
        if float(util) < 50 and float(used) < 0.5 * float(total):
            return True, f"{name} util {util}% mem {used}/{total} MiB (waited {waited} s)"
        if waited >= max_wait_s:
            return False, f"GPU busy: {name} util {util}% mem {used}/{total} MiB after {waited} s; not used"
        time.sleep(15)
        waited += 15


def run(args) -> int:
    os.nice(10)
    scr = os.environ.get("SCRATCH")
    if not scr:
        raise SystemExit("SCRATCH not set (run from the Perlmutter login node after source jk_env.sh)")
    jk = Path(scr) / "jacobik"
    env_defaults = {
        "JACOBIK_DATA_ROOT": str(jk / "data"),
        "JACOBIK_GOLDEN_DIR": str(jk / "data" / "golden_pm" / EXPECTED_FINGERPRINT),
        "JACOBIK_BUDGET_OUT": str(jk / "data" / "energy_scale_refit"),
        "JACOBIK_BUDGET_DB": str(jk / "data" / "golden_pm" / "support" / "constraints_2026-10-17.json"),
        "JACOBIK_DB": str(jk / "data" / "db" / "constraints_2026-11-12.npz"),
    }
    for k, v in env_defaults.items():
        os.environ.setdefault(k, v)
    out = Path(args.out_dir) if args.out_dir else jk / "runs" / f"pm-quick-{stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    rep = Report("pm-quick", out)
    rep.notes.append(f"node {os.uname().nodename}; SCRATCH {scr}; threads cap {MAX_THREADS}; nice +10")
    only = set(args.only.split(",")) if args.only else {"fingerprint", "golden", "pytest", "dev"}
    from .__main__ import fingerprint_stage, golden_stage, pytest_stage
    from .golden_par import GROUPS
    if "fingerprint" in only:
        rep.run("fingerprint", fingerprint_stage)
    jobs = min(args.jobs, MAX_THREADS // 8)
    if "golden" in only:
        ref = Path(os.environ["JACOBIK_GOLDEN_DIR"])
        rep.run("golden (login, PM refs)", lambda: golden_stage(ref, jobs, list(GROUPS), out / "golden"))
    if "pytest" in only:
        rep.run("pytest (sharded)", lambda: pytest_stage(out / "pytest", min(args.shards, 4), None))
    if "dev" in only:
        rep.run("dev suite (login A100)", lambda: dev_stage(out / "dev", args))
    return rep.finish()


def dev_stage(out: Path, args):
    ok_gpu, gpu_msg = _gpu_gate()
    if not ok_gpu:
        return False, gpu_msg, {"gpu": gpu_msg}
    man = REPO / DEV_MANIFEST
    if not man.exists():
        return False, f"dev manifest missing: {man}", {}
    spec = json.loads(man.read_text())
    import hashlib
    sha = hashlib.sha256(man.read_bytes()).hexdigest()
    expected = (REPO / "dev/tools/check/DEV_MANIFEST.sha256").read_text().split()[0]
    if sha != expected:
        return False, f"dev manifest sha256 {sha[:12]} != expected {expected[:12]}", {}
    # control and one arm in the same bench call per parent (shared prepare), same node and GPU, through the runner
    from tools.run.orchestrator import Job, Scheduler
    plan = dict(name="dev", backend="bench", device="cuda", out=str(out), seeds=spec.get("seeds", [11]),
                arms=[dict(arm="C", flags={}, seed_budget="three_w16"),
                      dict(arm="K", flags={"start_kabsch": True}, seed_budget="three_w16")],
                parents=[dict(code=p["code"], cost=p.get("cost_s")) for p in spec["parents"]], repo=str(REPO), pack=False)
    t0 = time.time()
    sched = Scheduler(out, "cuda", max_lanes=args.workers, cpu_cap=24, max_rss_gb=26,
                      max_wall_s=GPU_PHASE_CAP_S - 60, kill_grace_s=50)
    sched.add_job(Job(plan, resume=False))
    res = sched.run()
    js = res["jobs"][0]
    summary = dict(wall_s=res["wall_s"], line=res["line"], failed=js["parents_failed"], timeout=js["parents_timeout"],
                   not_run=js["not_run"], stage_totals_s=js["stage_totals_s"], telemetry=res["telemetry"])
    (out / "dev_summary.json").write_text(json.dumps(dict(gpu_gate=gpu_msg, manifest_sha256=sha, summary=summary),
                                                     indent=1, default=str))
    # the stage checks the machinery (every parent ran and wrote records); generator outcomes (status failed) are
    # results, not a failure of the check
    ok = not js["not_run"]
    return ok, (f"{len(spec['parents'])} parents x 2 arms; failed {len(js['parents_failed'])}; timeout "
                f"{len(js['parents_timeout'])}; not run {len(js['not_run'])}; wall {time.time() - t0:.0f} s; {res['line']}"), summary
