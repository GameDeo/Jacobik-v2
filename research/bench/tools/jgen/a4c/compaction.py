"""Is the relaxation of a pool row a compaction? Pairwise-dispersion and size change from the row to its minimum.

    python -m tools.jgen.a4c.compaction --run SOLVE_RUN --arm cfg --relax RELAXLEVEL/a4b --out OUT.json [--jobs 8]

For every relaxed row of every unit (X0 from the pool, Xn from `a4c.relax_level`; reflected rows skipped), no solver:
  * radius of gyration of the heavy atoms at X0 and Xn;
  * a Grimme-D2-type pairwise dispersion energy (kJ/mol; s6 = 1, damping d = 20) over non-metal pairs four or more
    bonds apart or on different ligands, at X0 and Xn. It is a ruler for "how much attractive contact", not a fitted
    term: the production score has a repulsive Q6 and no attraction;
  * the same split into the crude step (X0 -> crude) is not available here (crude geometries are not kept).
Read with the MACE energy released: if Xn is systematically more dispersion-bound than X0 and the difference tracks the
released energy, the solver's rows are held too open and the missing term is attraction. Never fails the batch.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

C6 = {"H": 0.14, "B": 3.13, "C": 1.75, "N": 1.23, "O": 0.70, "F": 0.75, "Si": 9.23, "P": 7.84, "S": 5.57, "Cl": 5.07,
      "Se": 12.64, "Br": 12.47, "I": 31.50}                      # J nm^6 / mol (Grimme 2006)
R0 = {"H": 1.001, "B": 1.485, "C": 1.452, "N": 1.397, "O": 1.342, "F": 1.287, "Si": 1.716, "P": 1.705, "S": 1.683,
      "Cl": 1.639, "Se": 1.771, "Br": 1.749, "I": 1.892}


def _sep(adj, limit=4):
    n = len(adj)
    out = np.full((n, n), limit + 1, int)
    nb = [np.flatnonzero(adj[i]) for i in range(n)]
    for s in range(n):
        d = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            if d[u] >= limit:
                continue
            for v in nb[u]:
                if v not in d:
                    d[int(v)] = d[u] + 1
                    q.append(int(v))
        for v, k in d.items():
            out[s, v] = k
    return out


def parent(args):
    run, arm, relax, code = args
    out = dict(code=code, rows=[], error=None)
    try:
        from tools.jgen.bench.pool_export import _read_frames
        from tools.runkit import geom
        pair = None
        for seed in (11, 12, 13):
            mp = Path(run) / arm / "xyz" / code / f"{code}_{arm}_s{seed}_pool.json"
            rp = Path(relax) / "relax" / f"{code}_{seed}.json"
            if not mp.exists() or not rp.exists():
                continue
            m = json.loads(mp.read_text())
            g2r = np.asarray(m["gen_to_ref_idx"], int)
            syms = list(m["ref_symbols"])
            frames = [np.asarray(fr[2], float)[g2r] for f in dict.fromkeys(r["file"] for r in m["rows"])
                      for fr in _read_frames(mp.parent / f)]                       # reference order
            if pair is None:
                metal = geom.find_metal(syms)
                adj = np.asarray(geom.ligand_adjacency(np.asarray(m["ref_X"], float), syms, metal), bool)
                sep = _sep(adj)
                ok = np.array([s in C6 for s in syms])
                iu = np.triu_indices(len(syms), 1)
                keep = (sep[iu] >= 4) & ok[iu[0]] & ok[iu[1]]
                ia, ib = iu[0][keep], iu[1][keep]
                c6 = np.array([np.sqrt(C6[syms[a]] * C6[syms[b]]) * 1e3 for a, b in zip(ia, ib)])   # kJ/mol A^6
                rr = np.array([R0[syms[a]] + R0[syms[b]] for a, b in zip(ia, ib)])
                heavy = np.array([s != "H" for s in syms])
                pair = True

            def disp(X):
                d = np.linalg.norm(X[ia] - X[ib], axis=1)
                return float(-(c6 / d ** 6 / (1.0 + np.exp(-20.0 * (d / rr - 1.0)))).sum())

            def rg(X):
                h = X[heavy]
                return float(np.sqrt(((h - h.mean(0)) ** 2).sum(1).mean()))
            top = next(i for i, r in enumerate(m["rows"]) if r["is_top1"])
            for k, r in json.loads(rp.read_text())["rows"].items():
                i = int(k)
                if r.get("X") is None or not r.get("level") or m["rows"][i].get("mirror_of") is not None:
                    continue
                X0, Xn = frames[i], np.asarray(r["X"], float)
                e0 = m["rows"][i].get("mace_energy_kj")
                out["rows"].append(dict(seed=seed, row=i, is_top1=i == top, rg0=rg(X0), rgn=rg(Xn), disp0=disp(X0), dispn=disp(Xn),
                                        E_released=None if e0 is None or r.get("E_mace_kj") is None else float(e0) - float(r["E_mace_kj"]),
                                        E_released_crude=None if e0 is None or (r.get("crude") or {}).get("E_mace_kj") is None
                                        else float(e0) - float(r["crude"]["E_mace_kj"]), n_atoms=len(syms)))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--arm", default="control")
    ap.add_argument("--relax", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    codes = sorted({p.name.rsplit("_", 1)[0] for p in (Path(a.relax) / "relax").glob("*_1[123].json")})
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(parent, [(a.run, a.arm, a.relax, c) for c in codes], chunksize=2))
    Path(a.out).write_text(json.dumps(res))
    rows = [r for p in res for r in p["rows"]]
    err = [p["code"] for p in res if p["error"]]
    dd = np.array([r["dispn"] - r["disp0"] for r in rows]); dr = np.array([r["rgn"] - r["rg0"] for r in rows])
    E = np.array([r["E_released"] if r["E_released"] is not None else np.nan for r in rows])
    Ec = np.array([r["E_released_crude"] if r["E_released_crude"] is not None else np.nan for r in rows])
    ok = np.isfinite(E) & np.isfinite(Ec)
    from scipy.stats import spearmanr
    pct = lambda v: " ".join(f"{np.percentile(v, p):8.2f}" for p in (10, 25, 50, 75, 90))
    print(f"rows {len(rows)} parents {len(res)} errors {err}")
    print("dispersion Xn - X0 (kJ/mol; negative = more bound at the minimum): ", pct(dd), " share < 0:", round(float((dd < 0).mean()), 3))
    print("Rg Xn - X0 (A):                                                  ", pct(dr), " share < 0:", round(float((dr < 0).mean()), 3))
    print("spearman(E_released, -d_disp) =", round(float(spearmanr(E[ok], -dd[ok]).correlation), 3),
          "| spearman(E_released - E_released_crude, -d_disp) =", round(float(spearmanr((E - Ec)[ok], -dd[ok]).correlation), 3))


if __name__ == "__main__":
    main()
