"""Machine probe: cores and single-core xtb speed, local or on a Colab CPU runtime. Results go to RUN_DIR/machines.json.

    python -m tools.runkit.probe --local                    # measure this machine
    python -m tools.runkit.probe --colab rk-cpu             # one CPU-only runtime (costs about 0.1 unit/h), once
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from . import resources, xtb


def fixed_molecule():
    """A tripeptide (about 60 atoms), ETKDG seed 7, MMFF: the same coordinates on every machine (same RDKit)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles("CC(C)CC(NC(C)=O)C(=O)NC(Cc1ccccc1)C(=O)NCC(=O)O"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    return [a.GetSymbol() for a in m.GetAtoms()], m.GetConformer().GetPositions()


def bench_xtb(workdir, n: int = 4) -> dict:
    xtb.require()
    sym, X = fixed_molecule()
    ts, es = [], []
    for _ in range(n):
        r = xtb.opt(sym, X, 0, 1, workdir)
        if r["status"] != "ok":
            return dict(error=r["status"])
        ts.append(r["wall_s"])
        es.append(r["E_h"])
    return dict(median_s=statistics.median(ts), min_s=min(ts), n=n, E_h=es[0], same_energy=len(set(es)) == 1,
                nproc=os.cpu_count(), affinity=len(os.sched_getaffinity(0)), xtb=xtb.info()["version"])


def update_machines(name: str, rec: dict):
    p = resources.RUN_DIR / "machines.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[name] = rec
    resources.RUN_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--remote-out", help="(on a VM) write the probe json here")
    ap.add_argument("--colab", help="session name of a CPU-only runtime to create, probe and stop")
    a = ap.parse_args(argv)
    wd = resources.workroot(resources.RUN_DIR / "probe")
    if a.remote_out:
        Path(a.remote_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.remote_out).write_text(json.dumps(bench_xtb(wd)))
        return
    loc = bench_xtb(wd)
    update_machines("local", dict(loc, measured=time.strftime("%Y-%m-%d")))
    print("local", loc)
    if a.colab:
        from . import colab
        out = resources.RUN_DIR / "probe_colab"
        with colab.Session(a.colab, kind="cpu") as vm:
            vm.run("probe", "python -m tools.runkit.probe --remote-out /content/out/probe/probe.json")
            rc = vm.wait("probe", max_min=10)
            r = vm.finish("/content/out/probe", str(out), expected_files=1, pattern="*.json")
        if r["ok"]:
            rec = json.loads((out / "probe" / "probe.json").read_text())
            rec["speed_vs_local"] = round(loc["median_s"] / rec["median_s"], 3) if "median_s" in rec else None
            rec["measured"] = time.strftime("%Y-%m-%d")
            update_machines("colab_cpu", rec)
            print("colab cpu", rec)


if __name__ == "__main__":
    main()
