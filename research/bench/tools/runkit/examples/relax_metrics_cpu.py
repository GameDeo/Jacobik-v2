"""Example 2: relax + metrics, CPU heavy with a MACE energy pass (A4-style).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:.:research/bench python research/bench/tools/runkit/examples/relax_metrics_cpu.py POOL_DIR OUT
"""
import sys

import numpy as np

from tools.runkit import cache, geom, mace, pipeline, policy, resources, runprofile as rp, xtb


def main(pool_dir, out):
    xtb.require()
    budget = resources.Budget("A4")                              # 4 processes from budget.json
    store = cache.Store(f"{out}/cache")                          # a rerun of the same structures is a cache hit
    watcher = pipeline.Watcher(pool_dir, "*_s*.npz", shard="0/1", skip_file=f"{out}/skip.txt",
                               done_file=f"{out}/DONE", state_file=f"{out}/state.txt", poll=20)
    ev = mace.Evaluator("auto")
    tr = rp.Trace()

    def one_parent(code, files):
        policy.assert_allowed([code])                            # strict: val and reserve refused
        d = np.load(files[0])
        sym, q, m = [str(s) for s in d["symbols"]], int(d["charge"]), int(d["spin"])
        P, _ = geom.automorphisms(d["ref"].astype(float), sym, geom.find_metal(sym))
        jobs = [(sym, x, q, m) for x in d["rows_X"][:3].astype(float)]
        with tr("xtb"):
            res = xtb.opt_many(jobs, workers=budget.quota, workroot=resources.workroot(out), cycles=200, cache=store,
                               failures=f"{out}/failures.jsonl", codes=[code] * len(jobs))
        with tr("mace"):
            E = ev.energies_kj([(sym, r["X"], q, m) for r in res if r["X"] is not None])
        return dict(n_ok=sum(r["status"] == "ok" for r in res), E=E.tolist())

    for code, files in watcher:                                  # yields each parent once its pool is complete
        r = pipeline.run_groups(lambda k, f: one_parent(k, f), [(code, files)], ledger=f"{out}/failed.jsonl")
        watcher.mark_done(code)
    tr.write(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
