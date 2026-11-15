"""Unrelaxed quality per unit (the product: unrelaxed rows): core RMSD to the reference of the top-1 row and of the
nearest same-class row, before any relaxation. python -m tools.jgen.a4c.unrelaxed --run RUN --out OUT.json [--workers N]"""
from __future__ import annotations
import argparse, json, os, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def unit(args):
    run, code = args
    from tools.jgen.a4c.retro import _pool
    from tools.jgen.a4c.classes import Reference
    out = []
    for s in (11, 12, 13):
        u = dict(code=code, seed=s)
        try:
            m, frames, top = _pool(Path(run) / "control", code, s)
            ref = Reference(m["ref_symbols"], np.asarray(m["ref_X"], float))
            v = [ref.compare(X) for X in frames]
            same = [(x["core_rmsd_A"], i) for i, x in enumerate(v) if x["same_class"]]
            u.update(n_rows=len(frames), top1_same_class=v[top]["same_class"], top1_core=v[top]["core_rmsd_A"],
                     nearest_same_core=min(same)[0] if same else None, n_same=len(same))
        except Exception as exc:
            u["error"] = f"{type(exc).__name__}: {exc}"
        out.append(u)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    codes = sorted(p.name for p in (Path(a.run) / "control" / "xyz").iterdir() if p.is_dir())
    with ProcessPoolExecutor(a.workers) as ex:
        res = [u for r in ex.map(unit, [(a.run, c) for c in codes], chunksize=1) for u in r]
    Path(a.out).write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
