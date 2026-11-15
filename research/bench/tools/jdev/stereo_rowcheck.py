"""Row-preservation check for stereo_candidates (jgen/i-stereo): production vs flag-on solves of the golden parents.

    python research/bench/tools/jdev/stereo_rowcheck.py CODE [CODE ...] [--out JSON]

Per parent (CPU, seed 11, the golden call): solve with stereo_enum=False (production) and stereo_enum=True in one
process and report production rows found in the flag-on pool (coordinates equal to --tol), rows excluded only by the
output window, top-1 scores, class counts and wall time. Exit 1 if a production row is lost or top-1 drops."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parents[4]
for p in (_repo / "research/bench/tools/jdev/budget", _repo / "research/bench", _repo, _repo / "src"):
    sys.path.insert(0, str(p))
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                  JACOBIK_NUM_WORKERS=os.environ.get("JACOBIK_NUM_WORKERS", "8"), JACOBIK_WORKER_MODE="thread")


def rows(r):
    import numpy as np
    return (np.array([[a.position for a in c.complex.atoms] for c in r.conformers]),
            np.array([float(getattr(c.score, "total", c.score)) for c in r.conformers]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from pools import load_stats, set_cases, setup
    setup()
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    from jacobik.builder import engine
    st = load_stats()
    names = {}
    for s in ("dev32", "n48"):
        for c, n in set_cases(s):
            names[c] = n
    res, bad = {}, False
    for code in a.codes:
        rec = {}
        try:
            case = get_case(names.get(code, code))
            out = {}
            for tag, flag in (("prod", False), ("flag", True)):
                t0 = time.time()
                with starting_geometry(case, "etkdg"):
                    b = case.make_builder()
                    r = b.solve(seed=11, stats=st, angle_bends=True, device=a.device, stereo_enum=flag)
                wall = time.time() - t0
                X, s = rows(r) if r.conformers else (np.zeros((0, 1, 3)), np.zeros(0))
                z = float(r.scorer._agg_norm()) if getattr(r, "scorer", None) is not None and hasattr(r.scorer, "_agg_norm") else 0.0
                out[tag] = (X, s, z, wall, getattr(b, "_stereo_class_info", None), r)
            Xp, sp, zp, wp, _, _ = out["prod"]
            Xf, sf, zf, wf, info, rf = out["flag"]
            found, windowed, lost = 0, 0, 0
            for i in range(len(sp)):
                hit = any(Xf[j].shape == Xp[i].shape and np.abs(Xf[j] - Xp[i]).max() <= a.tol for j in range(len(sf)))
                if hit:
                    found += 1
                elif zf > 0 and sf.size and sp[i] < sf.max() - engine.OUT_WINDOW / zf:
                    windowed += 1
                else:
                    lost += 1
            n_cls = len({getattr(c, "stereo_variant_idx", None) for c in rf.conformers})
            rec = dict(prod_top1=float(sp.max()) if sp.size else None, flag_top1=float(sf.max()) if sf.size else None,
                       n_prod_rows=int(len(sp)), n_flag_rows=int(len(sf)), found=found, windowed_out=windowed, lost=lost,
                       classes_in_flag_pool=n_cls, class_info=info, wall_prod_s=round(wp, 1), wall_flag_s=round(wf, 1))
            ok = lost == 0 and (sf.size and sp.size and sf.max() >= sp.max() - 1e-6)
            rec["ok"] = bool(ok)
            bad |= not ok
        except Exception as exc:
            rec = dict(error=f"{type(exc).__name__}: {exc}", ok=False)
            bad = True
        res[code] = rec
        print(code, json.dumps(rec), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
