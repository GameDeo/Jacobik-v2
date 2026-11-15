"""A4c retrospective on stored A4b data (old production 273f2ccbf892): validates the metric, not a production number.

python -m tools.jgen.a4c.retro --a4b research/data/jgen/campaign/A4b --out research/data/p2r/metric/retro [--jobs 3]

Per parent: one Reference from the relaxed reference (graph + automorphisms from the frozen reference geometry); every
relaxed A4b row is compared (class, core RMSD, A4b offset). Old buckets from A4b `units.csv`. The unrelaxed screen walks
the FULL stored pools: rows outside the A4b relax set with full RMSD > 1.5 A that are the reference class with core RMSD
<= 0.5 A (unrelaxed, vs the unrelaxed reference) size the selection bias; they are not relaxed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

#: unit seeds; the P2R fold-in gate runs fresh seeds 31-33 (JACOBIK_A4_SEEDS="31,32,33")
SEEDS = tuple(int(x) for x in os.environ.get("JACOBIK_A4_SEEDS", "11,12,13").split(","))
SCREEN_CORE_A = 0.5
SCREEN_FULL_A = 1.5


def _pool(root, code, seed):
    from tools.jgen.bench.pool_export import _read_frames
    found = sorted((root / "xyz" / code).glob(f"{code}_*_s{seed}_pool.json"))   # any arm name (one arm per data dir)
    if len(found) != 1:
        raise FileNotFoundError(f"{code} s{seed}: {len(found)} pool manifests")
    m = json.loads(found[0].read_text())
    mp = np.asarray(m["gen_to_ref_idx"], int)
    frames = []
    for name in dict.fromkeys(r["file"] for r in m["rows"]):
        frames += [X[mp] for _, _, X in _read_frames(root / "xyz" / code / name)]
    if len(frames) != m["n_rows"]:
        raise ValueError("pool row count mismatch")
    top = next(i for i, r in enumerate(m["rows"]) if r["is_top1"])
    return m, np.array(frames), top


def parent(args):
    root, code, depth = args
    root = Path(root)
    from tools.jgen.a4c.classes import Reference
    from tools.runkit import geom
    t0 = time.perf_counter()
    out = dict(code=code, units=[], rows=[], error=None)
    try:
        refs = json.loads((root / "relax" / "references.json").read_text())
        ref_rel = refs.get(code)
        pools = {}
        for s in SEEDS:
            try:
                pools[s] = _pool(root, code, s)
            except (OSError, ValueError, StopIteration) as exc:
                pools[s] = exc
        m0 = next(v[0] for v in pools.values() if not isinstance(v, Exception))
        syms, ref_X = m0["ref_symbols"], np.asarray(m0["ref_X"], float)
        unrel = Reference(syms, ref_X, core_depth=depth)
        rel = None
        if ref_rel and ref_rel.get("X") is not None:
            rel = Reference(syms, np.asarray(ref_rel["X"], float), graph_X=ref_X, perms=unrel.perms, capped=unrel.capped,
                            core_depth=depth)
        out.update(capped=unrel.capped, n_perms=int(len(unrel.perms)), n_desc=len(rel.desc) if rel else None,
                   n_sites=len(rel.sites) if rel else None, ref_relaxed=rel is not None,
                   ref_desc_kinds={k: sum(d[0] == k for d in (rel or unrel).desc) for k in ("pair", "triple", "chi", "ez")},
                   ref_relax_same_class=(unrel.compare(np.asarray(ref_rel["X"], float))["same_class"] if rel else None))
        for s in SEEDS:
            u = dict(code=code, seed=s, status="ok")
            relax_p = root / "relax" / f"{code}_{s}.json"
            if isinstance(pools[s], Exception) or not relax_p.exists() or rel is None:
                u["status"] = "missing"; out["units"].append(u); continue
            m, frames, top = pools[s]
            rrows = json.loads(relax_p.read_text())["rows"]
            hits = {}
            for k, r in rrows.items():
                i = int(k)
                row = dict(code=code, seed=s, row=i, is_top1=i == top, relax_status=r.get("status"),
                           old_hit=r.get("hit"), old_equivalent=r.get("equivalent"), offset=r.get("offset"),
                           old_rmsd=r.get("rmsd_refrel_A"))
                if r.get("X") is not None and r.get("E_mace_kj") is not None:
                    X = np.asarray(r["X"], float)
                    v = rel.compare(X, energy_offset=r.get("offset"))
                    full = float(geom.sym_rmsd_to_ref(X[rel.heavy][None], rel.Xh, rel.perms)[0])
                    row.update(same_class=v["same_class"], mirror=v["mirror"], enantiomer=v["enantiomer"], core_rmsd=v["core_rmsd_A"],
                               donors_same=v["donors_same"], topology_same=v["topology_same"], guided=v["guided"],
                               hit=v["hit"], hit_by=v["hit_by"], full_rmsd=full, reason=v["reason"])
                    hits[i] = v["hit"]
                out["rows"].append(row)
            u["bucket"] = "C-TOP1" if hits.get(top) else ("C-RANK" if any(hits.values()) else "C-MISS")
            u["n_relaxed_rows"] = len(hits)
            # unrelaxed screen over the full pool
            rmsd_full = geom.sym_rmsd_to_ref(frames[:, unrel.heavy], unrel.Xh, unrel.perms)
            cand = [i for i in range(len(frames)) if str(i) not in rrows]          # every row A4b never relaxed
            found = []
            for i in cand:
                v = unrel.compare(frames[i])
                if v["same_class"] and v["core_rmsd_A"] <= SCREEN_CORE_A:
                    found.append(i)
            u.update(n_pool=len(frames), n_screened=len(cand), n_screen_class_core=len(found),
                     n_screen_class_core_beyond_1p5=sum(rmsd_full[i] > SCREEN_FULL_A for i in found),
                     screen_rows=found[:20])
            out["units"].append(u)
    except Exception as exc:  # loops over complexes never fail the batch
        out["error"] = f"{type(exc).__name__}: {exc}"; out["traceback"] = traceback.format_exc()[-2000:]
    out["wall_s"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--codes", nargs="*")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    excl = json.loads(Path("research/bench/benchmarks/locked_exclusions.json").read_text())["reserve_all_codes"]
    units = list(csv.DictReader((a.a4b / "units.csv").open()))
    codes = sorted({u["code"] for u in units})
    locked = sorted(set(codes) & set(excl))
    codes = [c for c in codes if c not in locked and (not a.codes or c in a.codes)]
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    with ProcessPoolExecutor(max_workers=min(3, a.jobs)) as ex:
        for r in ex.map(parent, [(str(a.a4b), c, a.depth) for c in codes]):
            res.append(r)
            print(r["code"], r.get("error") or "ok", f"{r['wall_s']:.1f}s", flush=True)
    (a.out / "parents.json").write_text(json.dumps(dict(locked_dropped=locked, depth=a.depth, parents=res), default=float))


if __name__ == "__main__":
    main()
