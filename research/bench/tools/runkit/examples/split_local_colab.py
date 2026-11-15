"""Example 3: local + Colab split. Skeleton: a Colab VM runs the solve, the local CPU analyses its share as pools arrive.

    PYTHONPATH=src:.:research/bench python research/bench/tools/runkit/examples/split_local_colab.py   (costs credits)
"""
from pathlib import Path

from tools.runkit import colab, pipeline, resources, runprofile as rp

WT = Path(__file__).resolve().parents[5]
CODES = ["AJAQAD", "AWONOR", "AYOGUR", "QEZVEY"]               # cal parents; policy.assert_allowed(CODES) first in real use
NATOMS = {"AJAQAD": 60, "AWONOR": 70, "AYOGUR": 55, "QEZVEY": 99}
OUT = resources.RUN_DIR / "examples" / "split"


def main():
    machines = [pipeline.Machine("local", cores=4, speed=1.0), pipeline.Machine("vm", cores=12, speed=0.5)]
    split = pipeline.split_by_cost(CODES, cost=lambda c: NATOMS[c] ** 2, machines=machines)
    proj = rp.project(n_items=len(CODES) * 3, per_item_s=60.0, workers=1, units_per_h=colab.L4_UNITS_PER_H)
    colab.stop_rule(proj.units, cap_units=8)                    # refuse before spending
    budget = resources.Budget("A4")
    with colab.Session("rk1", gpu="L4", repo=WT, ref="HEAD") as vm:       # __exit__ ALWAYS downs the VM
        vm.run("solve", "python -m tools.jgen bench run --arm control --panel cal --only " + ",".join(split["vm"]) +
               " --pool-dir /content/out/pool --out /content/out/ctl")
        feed = pipeline.RemoteFeed(vm, "/content/out/pool", OUT / "pool", codes=split["local"], stage=OUT / "stage", every=150).start()
        local = resources.spawn(["python", "-m", "tools.jgen.a4.run", "--pool-dir", str(OUT / "pool"), "--out", str(OUT / "a4"),
                                 "--watch", "--done-file", str(OUT / "DONE")], budget, tag="a4-local", log=OUT / "local.log", cpus=4)
        wd = colab.Watchdog(vm, idle_min=20, remote="/content/out/pool", local_dir=str(OUT / "vm_pool")).start()
        rc = vm.wait("solve", max_min=120)
        feed.stop()
        (OUT / "DONE").write_text("")
        local.wait()
        vm.finish("/content/out/pool", str(OUT / "vm_pool"), expected_files=len(split["vm"]) * 3, pattern="*.npz")


if __name__ == "__main__":
    main()
