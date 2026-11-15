"""How close are the embedded ligands' internal bonds and angles to the bound (reference) structure? No solve.

    python -m tools.jgen.a4c.embed_internals --xyz POST/a4b/xyz --out OUT.json [--jobs 3] [--fit DISPLACEMENT.json]
        [--codes ...]

The solver moves ligand poses, torsions and a few bends; every other bond length and angle of a ligand is the embed's.
Per parent the complex shell is built (production path, no solve) and each ligand-internal bond and angle is compared
with the tmQM reference (a GFN2 minimum of the bound complex), by the classes of `a4c.displacement`: B_donor, B_other,
A_donor, A_adj, A_other. Reported: RMS deviation per class and, with --fit (the force constants fitted by
`a4c.displacement`; an attribution, not a force field), the energy those deviations stand for.
Tried with this ruler on cal + tail (117 parents, 2026-11-13) and dropped, neither changed the internals: a converged
(--opt normal) embed refinement (median -0.8 of 48.6 kJ/mol) and Li caps on neutral sigma donors merged into one site
cap per chelate (donor bonds 0.0187 -> 0.0154 A, everything else unchanged, median -1.0 kJ/mol).
Pool manifests under --xyz give the builder -> reference atom map and the reference. Never fails the batch.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

INTERNAL = ("B_donor", "B_other", "A_donor", "A_adj", "A_other")
VARIANTS = {"prod": None}


def parent(args):
    xyz, code, variants = args
    out = dict(code=code, error=None, variants={})
    try:
        from tools.jgen.bench.run import _setup_env, _prefetch_parent
        _setup_env({})
        from tools.jgen import inputs
        from tools.jgen.a4c.displacement import coordinates, values
        mp = sorted((Path(xyz) / code).glob(f"{code}_*_pool.json"))[0]
        m = json.loads(mp.read_text())
        g2r = np.asarray(m["gen_to_ref_idx"], int)
        syms, ref_X = list(m["ref_symbols"]), np.asarray(m["ref_X"], float)
        coords = coordinates(syms, ref_X)
        cls = np.array([c[0] for c in coords])
        v_ref = values(ref_X, coords)
        shard, ds = inputs.frame_shards()[code]
        mol0, builder, err = _prefetch_parent(code, shard, ds)
        if builder is None:
            raise ValueError(f"unmappable: {err}")
        out["n"] = {c: int((cls == c).sum()) for c in INTERNAL}
        for name in variants:
            t0 = time.perf_counter()
            try:
                mol, rdmols, donor_idx = builder._build_shell(False, False, embed_seed=None, chelate_embed=True, embed=None)
                Xb = np.array([a.position for a in mol.atoms], float)
                X = Xb[g2r]                                           # reference order
                if [mol.atoms[int(i)].symbol for i in g2r] != syms:
                    raise ValueError("builder atoms do not map onto the reference symbols")
                dv = values(X, coords) - v_ref
                out["variants"][name] = dict(ss={c: float((dv[cls == c] ** 2).sum()) for c in INTERNAL},
                                             rms={c: (float(np.sqrt((dv[cls == c] ** 2).mean())) if (cls == c).any() else None)
                                                  for c in INTERNAL},
                                             wall_s=time.perf_counter() - t0)
            except Exception as exc:
                out["variants"][name] = dict(error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        import traceback
        out["error"] = f"{type(exc).__name__}: {exc}"; out["traceback"] = traceback.format_exc()[-1500:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--fit")
    ap.add_argument("--codes", nargs="*")
    a = ap.parse_args()
    a.variants = list(VARIANTS)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    work = []
    for x in a.xyz:
        for p in sorted(Path(x).iterdir()):
            if p.is_dir() and (not a.codes or p.name in a.codes):
                work.append((x, p.name, a.variants))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(parent, work, chunksize=1))
    Path(a.out).write_text(json.dumps(res))
    kfit = None
    if a.fit:
        f = json.loads(Path(a.fit).read_text())["fit"]
        kfit = dict(zip(f["classes"], f["k"]))
    ok = [r for r in res if not r["error"] and all("ss" in r["variants"].get(v, {}) for v in a.variants)]
    print(f"parents {len(res)} complete {len(ok)}; failed: "
          f"{[(r['code'], r['error'] or {v: d.get('error') for v, d in r['variants'].items() if d.get('error')}) for r in res if r not in ok][:12]}")
    deg = np.degrees(1.0)
    print(f"{'variant':12s}" + "".join(f"{c + (' A' if c[0] == 'B' else ' deg'):>14s}" for c in INTERNAL) +
          ("   internal strain kJ/mol: median  mean" if kfit else "") + "   wall s (median)")
    base = None
    for v in a.variants:
        rms = [np.median([r["variants"][v]["rms"][c] * (1 if c[0] == "B" else deg) for r in ok if r["variants"][v]["rms"][c] is not None])
               for c in INTERNAL]
        line = f"{v:12s}" + "".join(f"{x:14.4f}" for x in rms)
        if kfit:
            e = np.array([sum(kfit[c] * r["variants"][v]["ss"][c] for c in INTERNAL) for r in ok])
            base = e if base is None else base
            line += f"   {np.median(e):30.1f} {e.mean():6.1f}   paired vs first: median {np.median(e - base):+.1f} mean {(e - base).mean():+.1f} better {int((e < base - 1).sum())} worse {int((e > base + 1).sum())}"
        line += f"   {np.median([r['variants'][v]['wall_s'] for r in ok]):.1f}"
        print(line)


if __name__ == "__main__":
    main()
