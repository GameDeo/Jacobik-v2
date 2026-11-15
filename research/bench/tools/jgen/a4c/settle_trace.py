"""Per-cycle relaxation trace of the best rows per class: how fast does a row settle, and where do the cycles go?

    python -m tools.jgen.a4c.settle_trace --run SOLVE_RUN --arm cfg --post POST/a4c/parents.json --inputs A4B_INPUTS
        --out OUT [--seeds 11] [--jobs 16] [--engines ancopt lbfgs] [--codes ...] [--summary-only]

Per unit the rows traced are the best-ranked row of the reference class among the post's relaxed rows (`same_class`)
and the top-1 row. Each is relaxed from its unrelaxed geometry X0 with GFN2 (`--opt normal`, cycle cap 400) and every
cycle's energy is read from xtbopt.log; the tmQM reference is relaxed the same way (it is a GFN2 minimum already, so it
gives E_ref and a control on the cycle count of a structure that sits in its basin). Reported per trace, GFN2 kJ/mol:
E(cycle k) - E(final) and E(cycle k) - E_ref, the cycles to come within 1 / 2 / 5 kJ/mol of the final energy, the
heavy-atom RMSD of each cycle to the final geometry, and the cycles to reach E_ref + 5. With `--engines lbfgs` the same
start is also relaxed with xtb's L-BFGS engine (a control on the optimiser, not on the geometry). Each unit is saved as
it finishes (OUT/rows/) and a rerun resumes. Never fails the batch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

H2KJ = 2625.4996
_LOG_E = re.compile(r"energy:\s+(-?\d+\.\d+)\s+gnorm:\s+(\d+\.\d+)")


def _kabsch(A, B):
    a, b = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    S[-1] *= np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.sqrt(max((a ** 2).sum() + (b ** 2).sum() - 2 * S.sum(), 0.0) / len(a)))


def _trace(binary, syms, X, q, mult, td, engine="ancopt", cycles=400, timeout=900):
    with open(f"{td}/in.xyz", "w") as f:
        f.write(f"{len(syms)}\n\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(syms, X)))
    for n in ("xtbopt.xyz", "xtbopt.log", "xtbrestart", "o.inp", ".xtboptok"):
        Path(td, n).unlink(missing_ok=True)
    cmd = [binary, "in.xyz", "--gfn", "2", "--chrg", str(q), "--uhf", str(mult - 1), "-P", "1",
           "--opt", "normal", "--cycles", str(cycles)]
    if engine != "ancopt":
        Path(td, "o.inp").write_text(f"$opt\n engine={engine}\n$end\n")
        cmd += ["--input", "o.inp"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="500M")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout, env=env)
    wall = time.perf_counter() - t0
    log = Path(td, "xtbopt.log")
    if not log.exists():
        raise ValueError(f"no xtbopt.log (rc {p.returncode})")
    L = log.read_text().split("\n")
    n = len(syms)
    E, G, F = [], [], []
    i = 0
    while i + n + 1 < len(L) + 1 and L[i].strip():
        m = _LOG_E.search(L[i + 1])
        if not m:
            break
        E.append(float(m.group(1))); G.append(float(m.group(2)))
        F.append(np.array([[float(v) for v in ln.split()[1:4]] for ln in L[i + 2:i + 2 + n]]))
        i += n + 2
    conv = "GEOMETRY OPTIMIZATION CONVERGED" in p.stdout
    return np.array(E), np.array(G), F, conv, wall


def unit(args):
    run, arm, code, seed, rows, q, mult, outdir, engines = args
    done = Path(outdir) / "rows" / f"{code}_{seed}.json"
    if done.exists():
        return json.loads(done.read_text())
    out = dict(code=code, seed=seed, error=None, traces=[])
    try:
        from tools.jgen.bench.pool_export import _read_frames
        from tools.runkit import xtb
        binary = xtb.info(None)["path"]
        mp = Path(run) / arm / "xyz" / code / f"{code}_{arm}_s{seed}_pool.json"
        m = json.loads(mp.read_text())
        g2r = np.asarray(m["gen_to_ref_idx"], int)
        syms = list(m["ref_symbols"])
        heavy = np.array([s != "H" for s in syms])
        frames = [np.asarray(fr[2], float)[g2r] for f in dict.fromkeys(r["file"] for r in m["rows"])
                  for fr in _read_frames(mp.parent / f)]
        (Path(outdir) / "tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path(outdir) / "tmp") as td:
            Er, Gr, Fr, cr, wr = _trace(binary, syms, np.asarray(m["ref_X"], float), q, mult, td)
            e_ref = float(Er[-1])
            out["ref"] = dict(cycles=len(Er), converged=cr, released=(Er[0] - Er[-1]) * H2KJ,
                              rmsd_heavy=_kabsch(np.asarray(m["ref_X"], float)[heavy], Fr[-1][heavy]))
            for role, i, is_top in rows:
                for eng in engines:
                    rec = dict(role=role, row=int(i), is_top1=is_top, engine=eng, error=None)
                    try:
                        E, G, F, conv, wall = _trace(binary, syms, frames[i], q, mult, td, engine=eng)
                        rec.update(cycles=len(E), converged=conv, wall_s=wall,
                                   e_minus_final=((E - E[-1]) * H2KJ).tolist(), e_minus_ref=((E - e_ref) * H2KJ).tolist(),
                                   gnorm=G.tolist(), rmsd_to_final=[_kabsch(f[heavy], F[-1][heavy]) for f in F],
                                   rmsd_x0_final=_kabsch(frames[i][heavy], F[-1][heavy]))
                    except Exception as exc:
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                    out["traces"].append(rec)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    try:
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(out))
    except OSError:
        pass
    return out


def _first(v, thr):
    v = np.asarray(v)
    k = np.flatnonzero(v <= thr)
    return int(k[0]) + 1 if len(k) else None                         # cycles counted from 1 (first frame = cycle 1)


def summary(res):
    def q(v):
        v = np.array([x for x in v if x is not None], float)
        return f"n {len(v):3d}  p25 {np.percentile(v, 25):6.1f}  med {np.median(v):6.1f}  p75 {np.percentile(v, 75):6.1f}  p90 {np.percentile(v, 90):6.1f}" if len(v) else "n 0"
    ok = [r for r in res if not r["error"]]
    print(f"units {len(res)} ok {len(ok)}; failed {[(r['code'], r['seed'], r['error']) for r in res if r['error']][:10]}")
    print("reference (already a GFN2 minimum): cycles", q([r["ref"]["cycles"] for r in ok]))
    for role in ("class_best", "top1"):
        for eng in sorted({t["engine"] for r in ok for t in r["traces"]}):
            T = [t for r in ok for t in r["traces"] if (t["role"] == role or (role == "top1" and t.get("is_top1")))
                 and t["engine"] == eng and not t["error"]]
            if not T:
                continue
            print(f"\n== {role} / {eng}: {len(T)} traces")
            print("  cycles to converge          ", q([t["cycles"] for t in T]))
            for thr in (5, 2, 1):
                print(f"  cycles to within {thr} of final ", q([_first(t["e_minus_final"], thr) for t in T]))
            print("  cycles to <= E_ref + 5      ", q([_first(t["e_minus_ref"], 5) for t in T]),
                  f" never {sum(_first(t['e_minus_ref'], 5) is None for t in T)}")
            print("  cycles to <= E_ref          ", q([_first(t["e_minus_ref"], 0) for t in T]),
                  f" never {sum(_first(t['e_minus_ref'], 0) is None for t in T)}")
            for k in (1, 5, 10, 20):
                print(f"  after cycle {k:2d}: E - final ", q([t["e_minus_final"][min(k, t["cycles"]) - 1] for t in T]),
                      "| heavy RMSD to final", q([t["rmsd_to_final"][min(k, t["cycles"]) - 1] for t in T]))
            print("  final E - E_ref             ", q([t["e_minus_ref"][-1] for t in T]))
            print("  heavy RMSD X0 -> final      ", q([t["rmsd_x0_final"] for t in T]))
            print("  energy released, cycles 1-10 vs after 10 (sum over traces): "
                  f"{sum(t['e_minus_final'][0] - t['e_minus_final'][min(10, t['cycles']) - 1] for t in T):.0f} vs "
                  f"{sum(t['e_minus_final'][min(10, t['cycles']) - 1] for t in T):.0f} kJ/mol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+"); ap.add_argument("--arm", default="cfg")
    ap.add_argument("--post", required=True, nargs="+"); ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[11])
    ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--engines", nargs="+", default=["ancopt"])
    ap.add_argument("--codes", nargs="*"); ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summary_only:
        res = [json.loads(p.read_text()) for p in sorted((out / "rows").glob("*.json"))]
        summary(res); return
    cm = json.loads((Path(a.inputs) / "charge_mult.json").read_text())
    work = []
    for run, post in zip(a.run, a.post):
        for p in json.load(open(post))["parents"]:
            c = p["code"]
            if c not in cm or (a.codes and c not in a.codes):
                continue
            for u in p["units"]:
                if u["seed"] not in a.seeds or u.get("status") != "ok":
                    continue
                rs = sorted((r for r in p["rows"] if r["seed"] == u["seed"]), key=lambda r: r["row"])
                sc = [r for r in rs if r.get("same_class") and r.get("relax_status") == "ok"]
                top = [r for r in rs if r.get("is_top1")]
                rows = []                                            # (role, row, is_top1); a top-1 row of the class is traced once
                if sc:
                    rows.append(("class_best", sc[0]["row"], bool(sc[0].get("is_top1"))))
                if top and (not sc or top[0]["row"] != sc[0]["row"]):
                    rows.append(("top1", top[0]["row"], True))
                if rows:
                    work.append((run, a.arm, c, u["seed"], rows, int(cm[c]["charge"]), int(cm[c]["mult"]), str(out), a.engines))
    print(f"units {len(work)}", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(unit, work, chunksize=1))
    (out / "settle_trace.json").write_text(json.dumps(res))
    summary(res)


if __name__ == "__main__":
    main()
