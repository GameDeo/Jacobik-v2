"""Which coordinates move when a pool row relaxes, in the fast step and in the slow one?

    python -m tools.jgen.a4c.displacement --run SOLVE_RUN --arm control --crude POST/a4b --normal RELAXLEVEL/a4b \
        --out OUT.json [--jobs 8]

Three geometries per relaxed row (reflected rows skipped): X0 the pool row, Xc after the crude relaxation (the gate's
"few steps"), Xn the converged minimum (`a4c.relax_level`). For the fast step X0 -> Xc and the slow step Xc -> Xn the
change of every internal coordinate is collected by class:

  bonds    B_MD metal-donor | B_donor donor-ligand atom | B_other
  angles   A_DMD_cis, A_DMD_trans at the metal | A_MDX metal-donor-neighbour (donor tilt) | A_donor centred on a donor
           (ligand atoms only) | A_adj centred on an atom bonded to a donor | A_other
  torsions T_chel bond on a metallacycle but in no ligand ring (chelate backbone) | T_ring bond in a ligand ring |
           T_donor acyclic bond with a donor end | T_big / T_small acyclic rotor with >= 4 / 1-3 heavy atoms on the
           smaller side | T_H rotor that moves hydrogens only
Per row and step: n and the sum of squared changes per class (A^2, rad^2), the MACE energy released in the step.
The summary gives the median absolute change per class and a non-negative least-squares fit
E_released(step) ~ sum_c k_c * SS_c over all rows and both steps: effective force constants per class and, from them,
the share of the released energy each class carries in each step. The fit is an attribution, not a force field
(classes are correlated); read the ordering, not the decimals. Never fails the batch.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BONDS = ("B_MD", "B_donor", "B_other")
ANGLES = ("A_DMD_cis", "A_DMD_trans", "A_MDX", "A_donor", "A_adj", "A_other")
TORS = ("T_chel", "T_ring", "T_donor", "T_big", "T_small", "T_H")
CLASSES = BONDS + ANGLES + TORS


def _connected(nb, a, b, skip):
    seen, q = {a}, deque([a])
    while q:
        u = q.popleft()
        for v in nb[u]:
            if (u, v) == skip or (v, u) == skip or v in seen:
                continue
            if v == b:
                return True
            seen.add(v); q.append(v)
    return False


def _side(nb, start, block):
    seen, q = {start}, deque([start])
    while q:
        u = q.popleft()
        for v in nb[u]:
            if v != block and v not in seen:
                seen.add(v); q.append(v)
    return seen


def coordinates(syms, Xg):
    """Internal coordinates of one complex from its bond graph (geometry Xg): [(class, kind, atom tuple)]."""
    from tools.runkit import geom
    n = len(syms)
    metal = geom.find_metal(syms)
    adj = np.asarray(geom.ligand_adjacency(np.asarray(Xg, float), syms, metal), bool)
    rc = np.array([geom.rcov(s) for s in syms])
    d = np.linalg.norm(Xg - Xg[metal], axis=1)
    donors = [int(i) for i in range(n) if i != metal and syms[i] != "H" and d[i] < rc[metal] + rc[i] + 0.4]
    dset = set(donors)
    nbl = [[int(v) for v in np.flatnonzero(adj[i])] for i in range(n)]                 # ligand bonds only
    nbm = [list(x) for x in nbl]
    for x in donors:
        nbm[x].append(metal); nbm[metal].append(x)
    heavy = [s != "H" for s in syms]
    out = [("B_MD", "b", (metal, x)) for x in donors]
    for i in range(n):
        for j in nbl[i]:
            if i < j:
                out.append(("B_donor" if (i in dset or j in dset) else "B_other", "b", (i, j)))
    for a, b in itertools.combinations(donors, 2):
        va, vb = Xg[a] - Xg[metal], Xg[b] - Xg[metal]
        ang = np.degrees(np.arccos(np.clip(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb), -1, 1)))
        out.append(("A_DMD_trans" if ang > 150 else "A_DMD_cis", "a", (a, metal, b)))
    for x in donors:
        for y in nbl[x]:
            out.append(("A_MDX", "a", (metal, x, y)))
    adjd = {y for x in donors for y in nbl[x]} - dset
    for c in range(n):
        if c == metal:
            continue
        cls = "A_donor" if c in dset else ("A_adj" if c in adjd else "A_other")
        for a, b in itertools.combinations(nbl[c], 2):
            out.append((cls, "a", (a, c, b)))
    for j in range(n):
        for k in nbl[j]:
            if j >= k:
                continue
            nj = [i for i in nbl[j] if i != k]
            nk = [i for i in nbl[k] if i != j]
            if not nj or not nk:
                continue
            in_lig_ring = _connected(nbl, j, k, (j, k))
            if in_lig_ring:
                cls = "T_ring"
            elif _connected(nbm, j, k, (j, k)):
                cls = "T_chel"
            else:
                sj, sk = _side(nbl, j, k), _side(nbl, k, j)
                small = min(sum(heavy[a] for a in sj) - 1, sum(heavy[a] for a in sk) - 1)
                cls = ("T_donor" if (j in dset or k in dset) else "T_H" if small == 0 else "T_small" if small <= 3 else "T_big")
            hj = [i for i in nj if heavy[i]] or nj
            hk = [i for i in nk if heavy[i]] or nk
            out.append((cls, "t", (hj[0], j, k, hk[0])))
    return out


def values(X, coords):
    v = np.empty(len(coords))
    for n, (_, kind, at) in enumerate(coords):
        if kind == "b":
            v[n] = np.linalg.norm(X[at[0]] - X[at[1]])
        elif kind == "a":
            a, b = X[at[0]] - X[at[1]], X[at[2]] - X[at[1]]
            v[n] = np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))
        else:
            i, j, k, l = at
            b0, b1, b2 = X[i] - X[j], X[k] - X[j], X[l] - X[k]
            b1 = b1 / np.linalg.norm(b1)
            p, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
            v[n] = np.arctan2(np.cross(b1, p) @ w, p @ w)
    return v


def parent(args):
    run, arm, crude, normal, code = args
    out = dict(code=code, rows=[], error=None)
    try:
        from tools.jgen.bench.pool_export import _read_frames
        coords = None
        for seed in (11, 12, 13):
            mp = Path(run) / arm / "xyz" / code / f"{code}_{arm}_s{seed}_pool.json"
            cp, np_ = Path(crude) / "relax" / f"{code}_{seed}.json", Path(normal) / "relax" / f"{code}_{seed}.json"
            if not (mp.exists() and cp.exists() and np_.exists()):
                continue
            m = json.loads(mp.read_text())
            g2r = np.asarray(m["gen_to_ref_idx"], int)
            syms = list(m["ref_symbols"])
            frames = [np.asarray(fr[2], float)[g2r] for f in dict.fromkeys(r["file"] for r in m["rows"])
                      for fr in _read_frames(mp.parent / f)]
            if coords is None:
                coords = coordinates(syms, np.asarray(m["ref_X"], float))
                cls = np.array([c[0] for c in coords]); tors = np.array([c[1] == "t" for c in coords])
                masks = {c: cls == c for c in CLASSES}
            rc_, rn_ = json.loads(cp.read_text())["rows"], json.loads(np_.read_text())["rows"]
            top = next(i for i, r in enumerate(m["rows"]) if r["is_top1"])
            for k, r in rn_.items():
                i = int(k)
                c = rc_.get(k)
                if (r.get("X") is None or not r.get("level") or c is None or c.get("X") is None
                        or m["rows"][i].get("mirror_of") is not None):
                    continue
                V = [values(np.asarray(X, float), coords) for X in (frames[i], c["X"], r["X"])]
                e0, ec, en = m["rows"][i].get("mace_energy_kj"), c.get("E_mace_kj"), r.get("E_mace_kj")
                rec = dict(seed=seed, row=i, is_top1=i == top, n_atoms=len(syms), cycles_crude=c.get("cycles"),
                           cycles_more=r.get("cycles"), E_fast=None if None in (e0, ec) else float(e0) - float(ec),
                           E_slow=None if None in (ec, en) else float(ec) - float(en), n={c_: int(mk.sum()) for c_, mk in masks.items()})
                for name, (a, b) in (("fast", (V[0], V[1])), ("slow", (V[1], V[2]))):
                    dv = b - a
                    dv[tors] = (dv[tors] + np.pi) % (2 * np.pi) - np.pi
                    rec[f"ss_{name}"] = {c_: float((dv[mk] ** 2).sum()) for c_, mk in masks.items()}
                    rec[f"mad_{name}"] = {c_: (float(np.median(np.abs(dv[mk]))) if mk.any() else None) for c_, mk in masks.items()}
                    if name == "fast":
                        rec["bias_fast"] = {c_: (float(np.mean(-dv[mk])) if mk.any() else None) for c_, mk in masks.items()}   # X0 - Xc
                out["rows"].append(rec)
    except Exception as exc:
        import traceback
        out["error"] = f"{type(exc).__name__}: {exc}"; out["traceback"] = traceback.format_exc()[-1500:]
    return out


def summarise(res):
    from scipy.optimize import nnls
    rows = [r for p in res for r in p["rows"]]
    unit = {c: (1.0 if c in BONDS else np.degrees(1.0)) for c in CLASSES}
    lines = [f"rows {len(rows)} parents {len(res)} errors {[p['code'] for p in res if p['error']]}"]
    A, y, tag = [], [], []
    for r in rows:
        for step in ("fast", "slow"):
            e = r[f"E_{step}"]
            if e is None:
                continue
            A.append([r[f"ss_{step}"][c] for c in CLASSES]); y.append(e); tag.append(step)
    A, y, tag = np.asarray(A), np.asarray(y), np.asarray(tag)
    ok = np.abs(y) < 400                                              # a broken row would own the fit
    k, _ = nnls(A[ok], np.clip(y[ok], 0, None))
    pred = A @ k
    r2 = 1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    lines.append(f"NNLS fit of the released energy on the squared changes: R2 {r2:.2f} over {int(ok.sum())} row-steps")
    for step in ("fast", "slow"):
        s = tag == step
        tot = float(np.median(y[s]))
        lines.append(f"== {step} step: MACE energy released median {tot:.1f} kJ/mol (mean {y[s].mean():.1f}); "
                     f"cycles median {np.median([r['cycles_crude' if step == 'fast' else 'cycles_more'] or 0 for r in rows]):.0f}")
        lines.append(f"   {'class':12s} {'n/row':>6s} {'median |change|':>16s} {'attributed kJ/mol (mean)':>26s} {'share':>6s}")
        attr = (A[s] * k[None]).mean(0)
        for j, c in enumerate(CLASSES):
            mads = [r[f"mad_{step}"][c] for r in rows if r[f"mad_{step}"][c] is not None]
            nn = np.mean([r["n"][c] for r in rows])
            u = "A" if c in BONDS else "deg"
            lines.append(f"   {c:12s} {nn:6.1f} {np.median(mads) * unit[c] if mads else float('nan'):13.3f} {u:3s} {attr[j]:20.1f} {attr[j] / max(attr.sum(), 1e-9):11.2f}")
    lines.append("== systematic offsets of the solver's row against its crude-relaxed geometry (X0 - Xc, mean over rows of the per-row mean)")
    for c in CLASSES:
        b = [r["bias_fast"][c] for r in rows if r["bias_fast"][c] is not None]
        if c in TORS:
            continue
        lines.append(f"   {c:12s} {np.mean(b) * unit[c]:+8.3f} {'A' if c in BONDS else 'deg'}   (sd between rows {np.std(b) * unit[c]:.3f})")
    return "\n".join(lines), dict(classes=CLASSES, k=k.tolist(), r2=float(r2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--arm", default="control")
    ap.add_argument("--crude", required=True); ap.add_argument("--normal", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    for kk in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[kk] = "1"
    codes = sorted({p.name.rsplit("_", 1)[0] for p in (Path(a.normal) / "relax").glob("*_1[123].json")})
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(parent, [(a.run, a.arm, a.crude, a.normal, c) for c in codes], chunksize=2))
    text, fit = summarise(res)
    Path(a.out).write_text(json.dumps(dict(fit=fit, parents=res)))
    print(text)


if __name__ == "__main__":
    main()
