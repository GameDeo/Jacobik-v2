"""Unrelaxed product quality per relaxed row (owner 2026-11-12): how far a row moves to ITS OWN basin on GFN2 relaxation
(symmetry-aware heavy and core RMSD unrelaxed -> relaxed) and the MACE energy drop. Diagnostic, not a coverage gate.
python -m tools.jgen.a4c.own_basin --post POST --out OUT.json [--workers N]   (POST = rerun post OUT: a4b/xyz, a4b/relax)"""
from __future__ import annotations
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def parent(args):
    root, code = args
    from tools.jgen.a4c.retro import _pool
    from tools.jgen.a4c.classes import Reference, core_rmsd
    from tools.runkit import geom
    out = []
    for s in (11, 12, 13):
        try:
            m, frames, top = _pool(Path(root), code, s)
            rel = json.loads((Path(root) / "relax" / f"{code}_{s}.json").read_text())["rows"]
        except Exception:
            continue
        ref = Reference(m["ref_symbols"], np.asarray(m["ref_X"], float))
        for k, r in rel.items():
            i = int(k)
            if r.get("X") is None or r.get("status") not in ("ok", "capped"):
                continue
            U, R = frames[i], np.asarray(r["X"], float)
            own = Reference(m["ref_symbols"], R, graph_X=ref.graph_X, perms=ref.perms, capped=ref.capped)
            v = own.compare(U)
            full = float(geom.sym_rmsd_to_ref(U[ref.heavy][None], R[ref.heavy], ref.perms)[0])
            e_u = m["rows"][i].get("mace_energy_kj")
            out.append(dict(code=code, seed=s, row=i, top1=i == top, full=full, core=v["core_rmsd_A"],
                            same_class_as_own_basin=v["same_class"], reason=v["reason"],
                            drop_kj=None if e_u is None or r.get("E_mace_kj") is None else float(e_u - r["E_mace_kj"]),
                            cycles=r.get("cycles")))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--post", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    root = Path(a.post) / "a4b"
    codes = sorted(p.name for p in (root / "xyz").iterdir() if p.is_dir())
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for rs in ex.map(parent, [(str(root), c) for c in codes], chunksize=1) for r in rs]
    Path(a.out).write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
