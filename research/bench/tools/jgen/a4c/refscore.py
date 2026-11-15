"""P2R Step 3 reachability probe, part 1: does the production score prefer the reference basin? (CPU, no solve)

    python -m tools.jgen.a4c.refscore --run RUN --out OUT.json --codes HOHZAG ZAQNUE ... [--jobs 3] [--arm control]

RUN = tools.run output (ARM/{records,xyz}); the ranking score is the production one whatever the arm's search flags. Per parent, with the production profile and DB:
  * every pool row is re-scored with the production ranking score (tools.jdev.budget.pools.rank_terms, the path of
    tools.jgen.ds.cross_score); check: top-1 row score - rescore - top1_bend_prior ~ 0 (reported, never assumed);
  * the tmQM reference geometry (builder atom order) is scored under every offered donor-to-site assignment of every E/Z
    variant; the best is kept (no bend prior: compared with the rows' rescores, which exclude it too);
  * each pool row is classed against the unrelaxed reference (a4c.classes).
Reported per unit: reference best score, pool top-1 and best same-class rescores, how many pool rows outscore the
reference, the top-1 check residual. Reading: reference above the pool -> the score supports the reference basin and the
search did not reach it; reference below -> the score prefers other basins (score surface).
Loops never fail the batch.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np


def run(args):
    run_dir, code, arm = (*args, "control")[:3]
    out = dict(code=code, error=None, units=[])
    try:
        from tools.jgen.bench.run import _setup_env
        _setup_env({})                                   # production profile; must precede the jacobik import
        import torch
        torch.set_num_threads(1)
        from jacobik.stats.stats import StatsLookup, load_db
        from tools.jgen.inputs import frame_shards, load_mol
        from tools.jgen.bench.pool_export import _read_frames
        from tools.jgen.ds.cross_score import _source_scorers, _build_scorers, _signature, _pose_rows, _terms
        from tools.jgen.a4c.classes import Reference
        db = os.environ.get("JACOBIK_DB") or str(Path(os.environ.get("JACOBIK_DATA_ROOT", "data")) / "db/constraints_2026-11-12.npz")
        lookups = {"D": StatsLookup(load_db(db))}
        shard, dataset = frame_shards()[code]
        mol = load_mol(code, shard, dataset)
        if mol is None:
            raise ValueError("source geometry missing")
        pools = []
        for s in (11, 12, 13):
            mp = Path(run_dir) / arm / "xyz" / code / f"{code}_{arm}_s{s}_pool.json"
            rp = Path(run_dir) / arm / "records" / f"{code}_{s}.json"
            if not mp.exists() or not rp.exists():
                continue
            rec = json.loads(rp.read_text())
            if not mp.exists() or rec.get("status") != "ok":
                continue
            m = json.loads(mp.read_text())
            frames = [fr for f in dict.fromkeys(r["file"] for r in m["rows"]) for fr in _read_frames(mp.parent / f)]
            pools.append((s, rec, m, frames))
        if not pools:
            raise ValueError("no ok pool")
        m0 = pools[0][2]
        g2r = np.asarray(m0["gen_to_ref_idx"], int)
        ref_X = np.asarray(m0["ref_X"], float)
        Xb = np.zeros_like(ref_X); Xb[g2r] = ref_X        # reference in builder order: Xb[g2r] == ref_X
        cls = Reference(m0["ref_symbols"], ref_X)
        ref_best = dict(score=-np.inf, variant=None, group=None)
        per = {s: dict(rows={}) for s, *_ in pools}
        with torch.no_grad():
            for variant, ctx, assignments, fixes in _source_scorers(dict(runtime="D"), mol, lookups):
                sig = _signature(ctx, assignments)
                _, scorers = _build_scorers(ctx, assignments, lookups)
                sc = scorers["D"]
                G = len(assignments)
                if [ctx.mol.atoms[i].symbol for i in g2r] != list(m0["ref_symbols"]):
                    raise ValueError("builder atoms do not map onto the reference symbols")
                Xr = torch.as_tensor(np.repeat(Xb[None], G, 0), dtype=torch.float32)
                val, _ = _terms(sc, Xr, torch.arange(G))
                v = val.numpy()
                j = int(np.nanargmax(v))
                if np.isfinite(v[j]) and v[j] > ref_best["score"]:
                    ref_best = dict(score=float(v[j]), variant=variant, group=j, ez_fixes=str(fixes))
                for s, rec, m, frames in pools:
                    ii, X, groups = _pose_rows(frames, variant, sig)
                    if not ii:
                        continue
                    val, _ = _terms(sc, X, groups)
                    for k, i in enumerate(ii):
                        per[s]["rows"][i] = float(val[k])
        for s, rec, m, frames in pools:
            rows = per[s]["rows"]
            top = next(i for i, r in enumerate(m["rows"]) if r["is_top1"])
            u = dict(seed=s, n_rows=len(m["rows"]), n_rescored=len(rows))
            if top in rows and rec.get("top1_bend_prior") is not None:
                u["top1_check"] = float(m["rows"][top]["score"] - rows[top] - rec["top1_bend_prior"])
            same = []
            for i, (_, _, X) in enumerate(frames):
                v = cls.compare(np.asarray(X)[g2r])
                if v["same_class"]:
                    same.append((i, v["core_rmsd_A"]))
            u["top1_rescore"] = rows.get(top)
            sc_same = [(rows[i], i, c) for i, c in same if i in rows]
            u["best_same_class"] = None if not sc_same else dict(zip(("score", "row", "core_rmsd"), max(sc_same)))
            u["nearest_same_class"] = None if not sc_same else dict(zip(("score", "row", "core_rmsd"),
                                                                         min(sc_same, key=lambda t: t[2])))
            u["n_same_class"] = len(same)
            u["n_rows_above_ref"] = int(sum(v > ref_best["score"] for v in rows.values()))
            out["units"].append(u)
        out["reference"] = ref_best
    except Exception as exc:  # never fail the batch
        out["error"] = f"{type(exc).__name__}: {exc}"; out["traceback"] = traceback.format_exc()[-2000:]
    return json.loads(json.dumps(out, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--codes", nargs="+", required=True); ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--arm", default="control")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(run, [(a.run, c, a.arm) for c in a.codes], chunksize=1))
    Path(a.out).write_text(json.dumps(res, indent=1))
    for r in res:
        if r["error"]:
            print(r["code"], "ERROR", r["error"]); continue
        ref = r["reference"]["score"]
        for u in r["units"]:
            b = u["best_same_class"]
            print(r["code"], u["seed"], f"ref {ref:.3f}", f"top1 {u['top1_rescore']:.3f}" if u["top1_rescore"] is not None else "top1 -",
                  f"check {u.get('top1_check', float('nan')):+.2e}", "best same", None if not b else (round(b["score"], 3), round(b["core_rmsd"], 2)),
                  f"rows above ref {u['n_rows_above_ref']}/{u['n_rescored']}")


if __name__ == "__main__":
    main()
