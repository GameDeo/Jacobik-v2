"""P2R Step 3, item 1: where does the reference class go missing? (CPU; reads a lineage run with JACOBIK_LINEAGE_XYZ=1)

    python -m tools.jgen.a4c.lineage_class --run RUN --lineage LIN --out OUT [--workers 24]

RUN = tools.run output (control/xyz/<code>/*_pool.json gives the reference and gen_to_ref_idx); LIN = the
JACOBIK_LINEAGE_OUT directory (lineage_*.jsonl + xyz_*.npz: every solved start of every pass, with its outcome:
kept / merged / outside_window / beyond_cap / nonfinite). Each start is compared with the UNRELAXED reference
(a4c.classes.Reference): same class, core RMSD. Per unit (code, seed):
  * n_starts, n_same_class by outcome, best core RMSD of same-class starts by outcome, assignment indices (cand) that
    produced a same-class start;
  * verdict: never_solved_into_class (no start of any pass is the reference class: not enumerated, or not reachable
    from the enumerated starts) / lost_in_cull (same-class starts exist but none kept) / kept (a kept row is the class:
    should be offered in the pool; a mismatch with the pool is reported).
Loops never fail the batch: a unit error is recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _units(lin):
    """{(code, seed): [xyz files]} from the lineage jsonl records."""
    by = defaultdict(list)
    for f in Path(lin).glob("lineage_*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("xyz_file"):
                by[(r.get("code"), int(r["seed"]))].append(dict(file=r["xyz_file"], ez_variant=r.get("ez_variant"),
                                                                 rows=r.get("rows")))
    return by


def unit(args):
    run, lin, code, seed, files = args
    from tools.jgen.a4c.classes import Reference
    out = dict(code=code, seed=seed, error=None)
    try:
        m = json.loads((Path(run) / "control" / "xyz" / code / f"{code}_control_s{seed}_pool.json").read_text())
        mp = np.asarray(m["gen_to_ref_idx"], int)
        ref = Reference(m["ref_symbols"], np.asarray(m["ref_X"], float))
        stats = defaultdict(lambda: dict(n=0, same=0, ena=0, best_core=None))
        cands = defaultdict(int)
        passes = []
        for f in files:
            z = np.load(Path(lin) / f["file"], allow_pickle=False)
            X, oc, cd = z["X"].astype(float), z["outcome"], z["cand"]
            same_p = 0
            for i in range(len(X)):
                o = str(oc[i]); s = stats[o]; s["n"] += 1
                if not np.isfinite(X[i]).all():
                    continue
                v = ref.compare(X[i][mp])
                if v["same_class"]:
                    s["same"] += 1; same_p += 1; cands[int(cd[i])] += 1
                    c = v["core_rmsd_A"]
                    s["best_core"] = c if s["best_core"] is None else min(s["best_core"], c)
                elif v["enantiomer"]:
                    s["ena"] += 1
            passes.append(dict(file=f["file"], ez_variant=f["ez_variant"], rows=int(len(X)), same=same_p))
        tot = sum(s["same"] for s in stats.values())
        kept = stats["kept"]["same"] if "kept" in stats else 0
        out.update(stats=dict(stats), cands_with_class=dict(cands), passes=passes, n_starts=sum(s["n"] for s in stats.values()),
                   n_same=tot, n_enantiomer=sum(s["ena"] for s in stats.values()),
                   verdict="kept" if kept else ("lost_in_cull" if tot else "never_solved_into_class"))
    except Exception as exc:  # never fail the batch
        out["error"] = f"{type(exc).__name__}: {exc}"; out["traceback"] = traceback.format_exc()[-1500:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--lineage", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--codes", nargs="*")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    by = _units(a.lineage)
    jobs = [(a.run, a.lineage, c, s, f) for (c, s), f in sorted(by.items()) if not a.codes or c in a.codes]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(unit, jobs, chunksize=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for r in res:
        print(r["code"], r["seed"], r.get("error") or r["verdict"], "starts", r.get("n_starts"), "same", r.get("n_same"),
              "ena", r.get("n_enantiomer"), {k: (v["same"], v["best_core"]) for k, v in (r.get("stats") or {}).items()})


if __name__ == "__main__":
    main()
