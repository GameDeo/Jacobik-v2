"""CPU smoke run and parity check: runkit.xtb / runkit.mace against A4's relax.py on the same structures.

    python -m tools.runkit.smoke_parity --a4-relax PATH/TO/a4/relax.py --pool POOL.npz [--rows 3] [--cycles 30] [--workers 2]

Runs xtb_opt_many (A4) and xtb.opt_many (runkit) on `rows` rows of the pool, then MACE (CPU) on the reference and the relaxed
rows with both code paths, and requires exact equality (E_h, X, cycles, status; MACE energies). Writes profile.json and
parity.json under --out. The pool must belong to cal or tail (policy.assert_panel); never val or reserve."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import mace, policy, resources, runprofile, xtb


def load_a4(path):
    spec = importlib.util.spec_from_file_location("a4_relax_ref", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4-relax", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--panel", default="cal")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(resources.RUN_DIR / "smoke"))
    a = ap.parse_args(argv)
    code = Path(a.pool).name.split("_s")[0]
    prov = policy.assert_panel(a.panel, [code])
    xtb_info = xtb.require()
    A4 = load_a4(a.a4_relax)
    d = np.load(a.pool)
    sym = [str(s) for s in d["symbols"]]
    q, m = int(d["charge"]), int(d["spin"])
    rows = d["rows_X"][:a.rows].astype(float)
    jobs = [(sym, x, q, m) for x in rows]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wr = resources.workroot(out)
    tr = runprofile.Trace()
    with tr("a4_xtb"):
        ra = A4.xtb_opt_many(jobs, a.workers, wr + "/a4", a.cycles)
    with tr("runkit_xtb"):
        rb = xtb.opt_many(jobs, a.workers, wr + "/rk", a.cycles, failures=out / "failures.jsonl")
    res = dict(code=code, n_atoms=len(sym), rows=len(rows), cycles=a.cycles, xtb=xtb_info, policy=prov, xtb_rows=[])
    ok = True
    for i, (x, y) in enumerate(zip(ra, rb)):
        same = (x["status"] == y["status"] and x["E_h"] == y["E_h"] and x["cycles"] == y["cycles"] and x["converged"] == y["converged"]
                and ((x["X"] is None and y["X"] is None) or np.array_equal(x["X"], y["X"])))
        ok &= same
        res["xtb_rows"].append(dict(row=i, status=y["status"], E_h=y["E_h"], cycles=y["cycles"], identical=bool(same),
                                    a4_wall_s=round(x["wall_s"], 2), rk_wall_s=round(y["wall_s"], 2)))
    structs = [(sym, np.asarray(d["ref"], float), q, m)] + [(sym, r["X"], q, m) for r in rb if r["X"] is not None]
    with tr("a4_mace"):
        ea = np.concatenate([A4.mace_kj(sym, [np.asarray(d["ref"], float)], q, m, "cpu", 1),
                             A4.mace_kj(sym, [r["X"] for r in rb if r["X"] is not None], q, m, "cpu", 1)])
    ev = mace.Evaluator("cpu")
    with tr("runkit_mace"):
        eb = ev.energies_kj(structs)
    mace_same = bool(np.array_equal(ea, eb))
    # note: A4 evaluates ref and rows in two calls, runkit in one; equality below also shows batch composition did not matter
    res.update(mace_identical=mace_same, mace_kj=[float(v) for v in eb], mace_a4_kj=[float(v) for v in ea],
               mace_max_abs_diff=float(np.abs(ea - eb).max()), xtb_identical=bool(ok), mace_model_sha256=ev.model_sha256,
               mace_stats=ev.stats)
    (out / "parity.json").write_text(json.dumps(res, indent=1, default=str))
    tr.write(out, extra=dict(manifest=resources.manifest(torch_info=False)))
    print(json.dumps({k: res[k] for k in ("code", "n_atoms", "xtb_identical", "mace_identical", "mace_max_abs_diff")}))
    return 0 if (ok and mace_same) else 1


if __name__ == "__main__":
    sys.exit(main())
