"""Live Colab check (NOT a pytest test; costs credits). One L4 VM named rk1 for about 30 min.

    python -m tools.runkit.tests.live_colab_check [--name rk1] [--kind gpu|cpu]

Exercises Session.up (READY detection), run/wait, status (one exec), fetch_verified (count), Watchdog (idle -> fetch -> count -> down)
and the down-on-exit guarantee. Logs credits before and after. Always ends with the VM down."""
import argparse
import json
import sys
import time
from pathlib import Path

from tools.runkit import colab, resources

JOB = ("python -c 'import json,os; from tools.runkit import xtb; i=xtb.require(); os.makedirs(\"/content/out/pool\",exist_ok=True); "
       "[open(\"/content/out/pool/f%d.json\"%k,\"w\").write(json.dumps(i)) for k in range(5)]; print(i)'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rk1")
    ap.add_argument("--kind", default="gpu")
    ap.add_argument("--out", default=str(resources.RUN_DIR / "live"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(name=a.name, kind=a.kind, t0=time.strftime("%H:%M:%S"))
    vm = colab.Session(a.name, kind=a.kind)
    try:
        rep["credits_before"] = vm.credits()
        t = time.time()
        vm.up()
        rep["up_min"] = round((time.time() - t) / 60, 1)
        rep["ready"] = vm.ready_line
        rep["status_idle"] = vm.status(dirs=["/content/out/pool"])
        vm.run("rkcheck", JOB)
        rep["wait_rc"] = vm.wait("rkcheck", max_min=10)
        rep["status_after_job"] = vm.status(dirs=["/content/out/pool"])
        rep["fetch"] = vm.fetch_verified("/content/out/pool", str(out / "got"), expected=5, pattern="*.json")
        wd = colab.Watchdog(vm, idle_min=1.0, interval_s=20, remote="/content/out/pool", local_dir=str(out / "got_wd"),
                            expected=5, pattern="*.json")
        t = time.time()
        wd.start()
        while not vm.downed and time.time() - t < 600:
            time.sleep(10)
        rep["watchdog_fired"] = wd.fired
        rep["watchdog_min"] = round((time.time() - t) / 60, 1)
        rep["wd_files"] = len(list((out / "got_wd").rglob("*.json")))
    except BaseException as exc:
        rep["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        vm.down()
        rep["downed"] = vm.downed
        rep["credits_after"] = vm.credits()
        rep["cost"] = vm.cost()
        (out / f"live_{a.kind}.json").write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
