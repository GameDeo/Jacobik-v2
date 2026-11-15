"""A4t v2 analysis (coverage package Part A): re-analyses stored A4b/A4t data, no solves.

    python -m tools.jgen.a4t.analyse2 --a4b <A4b root: xyz/, records/> --a4t <A4t root: relax/> --out <dir> \
        [--parents A,B,..] [--old <v1 parents.csv>]

Differences to `analyse.py` (frozen v1), RC_DIAGNOSIS section 5.2:
 (a) hit = heavy-atom RMSD <= 0.3 A with global inversion allowed (both values reported);
 (b) equivalence (dE <= 5 kJ/mol) on MACE and on the stored GFN2 energies; verdicts that disagree -> class A (ambiguous);
 (c) R split into R-sat (saturated-ring conformer), R-chel (chelate / macrocycle pucker), R-rot (aryl / pendant rotor);
     C is the residual;
 (d) shell test under graph automorphisms (proper or mirror), no free donor permutation;
 (e) Z adds signed-volume checks at pyramidal centres and a planar-chelate-fold check to the RDKit stereo test.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
import json
from collections import Counter, defaultdict
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

from tools.jgen.a4t import metrics2 as m2
from tools.jgen.a4t.analyse import PARENTS, SEEDS, _load_parent, _write, sha256
from tools.jgen.a4t.metrics import (KNOWN_TYPING, csm_by_geometry, stereo_diff, stereo_graph)
from tools.runkit import geom


def _row_table(root, code, refout, symbols, heavy, P, shell, refX, n_rows_by_seed, graph=None):
    """One record per pool row (all seeds) with the v2 per-row quantities, plus the relaxed coordinates."""
    recs, coords = [], {}
    Rh = refX[heavy]
    for seed in SEEDS:
        p = Path(root) / "relax" / f"{code}_{seed}.json"
        rows = json.loads(p.read_text()).get("rows", {}) if p.exists() else {}
        ok = [(int(i), r) for i, r in rows.items() if r.get("status") == "ok" and r.get("X") is not None]
        X = np.asarray([r["X"] for _, r in ok], float) if ok else np.zeros((0, len(symbols), 3))
        if len(ok):
            rp, rm = m2.rmsd_proper_mirror(X[:, heavy, :], Rh, P, graph)
        okmap = {i: k for k, (i, _) in enumerate(ok)}
        for i in range(int(n_rows_by_seed[seed])):
            r = rows.get(str(i), {})
            rec = dict(code=code, seed=seed, row_index=i, pool_rank=i + 1, relax_status=r.get("status", "missing"),
                       E_mace_kj=r.get("E_mace_kj"), E_h=r.get("E_h"),
                       dE_mace=None, dE_gfn2=None, rmsd_proper=None, rmsd_mirror=None, rmsd_any=None, hit_proper=False,
                       hit=False, shell_proper=None, shell_mirror=None, shell_any=None, shell_match=False)
            if r.get("E_mace_kj") is not None:
                rec["dE_mace"] = float(r["E_mace_kj"]) - float(refout["E_refrel_mace_kj"])
            if r.get("E_h") is not None:
                rec["dE_gfn2"] = (float(r["E_h"]) - float(refout["E_h"])) * m2.H2KJ
            if i in okmap:
                k = okmap[i]
                sp, sm = m2.shell_rmsd_auto_any(X[k], refX, P, heavy, shell)
                rec.update(rmsd_proper=float(rp[k]), rmsd_mirror=float(rm[k]), rmsd_any=float(min(rp[k], rm[k])),
                           hit_proper=bool(rp[k] <= m2.HIT_RADIUS), hit=bool(min(rp[k], rm[k]) <= m2.HIT_RADIUS),
                           shell_proper=sp, shell_mirror=sm, shell_any=min(sp, sm), shell_match=bool(min(sp, sm) <= 0.3))
                coords[(seed, i)] = X[k]
            recs.append(rec)
    return recs, coords


def analyse2(a4b_root, a4t_root, out, parents=None, old_csv=None, workers=None):
    parents = list(parents if parents is not None else PARENTS)
    base, root, out = Path(a4b_root), Path(a4t_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    refs = json.loads((root / "relax" / "references.json").read_text())
    old = {}
    if old_csv and Path(old_csv).exists():
        old = {r["code"]: r["classification"] for r in csv.DictReader(open(old_csv))}
    prow, pcols, all_rows = [], [], []
    for c in [c for c in parents if c not in refs]:
        print(f"A4t2 {c}: no relaxed reference (no ok unit in this arm), skipped", flush=True)
    parents = [c for c in parents if c in refs]
    tasks = [(code, base, root, refs, old) for code in parents]
    workers = workers if workers is not None else len(os.sched_getaffinity(0))
    if workers < 1: raise ValueError("workers must be positive")
    with ProcessPoolExecutor(max_workers=workers, initializer=_pin_worker) as pool:
        for pr, rows in pool.map(_analyse_parent, tasks):
            prow.extend(pr); all_rows.extend(rows)
    _write(out / "rows2.csv", all_rows)
    _write(out / "parents2.csv", prow)
    meta = dict(n_parents=len(parents), n_rows=len(all_rows), class_counts=dict(Counter(p["classification"] for p in prow)),
                class_counts_mace=dict(Counter(p.get("class_mace") for p in prow)),
                class_counts_gfn2=dict(Counter(p.get("class_gfn2") for p in prow)),
                old_to_new=dict(Counter(f"{p.get('classification_old')}->{p['classification']}" for p in prow)),
                units_sha256=sha256(base / "units.csv") if (base / "units.csv").exists() else None)
    (out / "summary2.json").write_text(json.dumps(meta, indent=2))
    return prow, all_rows, meta


def _pin_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _analyse_parent(args):
    code, base, root, refs, old = args
    prow, all_rows = [], []
    try:
        mol, builder, builder_donors = _load_parent(code)
    except Exception as exc:
        print(f"A4t2 {code}: load failed {exc}", flush=True)
        prow.append(dict(code=code, classification="load_failed", classification_old=old.get(code), error=str(exc)))
        return prow, all_rows
    symbols = [a.symbol for a in mol.atoms]
    metal = int(mol.metal_idx) - 1
    refout = refs[code]
    refX = np.asarray(refout["X"], float)
    heavy = np.asarray([i for i, s in enumerate(symbols) if s != "H"], int)
    ref_donors = geom.donors_of(refX, symbols, metal, pad=.6).tolist()
    donor_mismatch = set(builder_donors) != set(map(int, ref_donors))
    ideal = {len(s) for s in __import__("jacobik.ik.targets", fromlist=["IDEAL_GEOMETRIES"]).IDEAL_GEOMETRIES.values()}
    geometry_evaluable = len(ref_donors) in ideal
    csm = csm_by_geometry(refX, metal, ref_donors) if geometry_evaluable else {}
    ref_geom = min(csm, key=csm.get) if csm else None
    builder_geom = str(builder.geometry)
    bm = csm.get(builder_geom)
    geom_mismatch = (builder_geom != ref_geom and bm is not None and bm - csm[ref_geom] > 1.0) if geometry_evaluable else None
    P, capped = geom.automorphisms(refX, symbols, metal)
    shell = [metal, *map(int, ref_donors)]
    n_rows = {}
    for s in SEEDS:
        man = json.loads((base / "xyz" / code / f"{code}_control_s{s}_pool.json").read_text())
        n_rows[s] = int(man["n_rows"])
    recs, coords = _row_table(root, code, refout, symbols, heavy, P, shell, refX, n_rows, geom.heavy_graph(refX, symbols, metal))
    all_rows.extend(recs)
    # ---- best shell-matched row (global-inversion rule) and its structural diagnosis
    cand = [r for r in recs if r["shell_match"]]
    any_shell = bool(cand)
    diag = dict(residual=None, n_centres=None, n_flipped=None, n_diastereo=None, cip_mismatch=False, chelate_fold=None,
                sat_diff=None, chel_diff=None, shares=None, best_seed=None, best_row=None,
                best_mirror=None, n_sat_rings=None, n_chel_rings=None)
    z_flag = False
    residual = "C"
    if any_shell:
        best = min(cand, key=lambda r: r["rmsd_any"])
        Xb = coords[(best["seed"], best["row_index"])]
        try:
            al = m2.best_alignment(Xb, refX, heavy, P)
            rings = m2.ring_atoms(refX, symbols, metal, ref_donors)
            cat = m2.atom_categories(symbols, metal, ref_donors, rings)
            shares = m2.deviation_shares(al["delta"], heavy, cat)
            cen = m2.stereocentres(refX, symbols, metal, ref_donors, rings["adj"])
            flipped, nc, nd = m2.chirality_flips(refX, al["Xr_ref"], cen)
            fold = m2.chelate_planarity_flag(refX, al["Xr_ref"], rings["chel"])
            ref_stereo = stereo_graph(mol, symbols, refX)
            row_stereo = stereo_graph(mol, symbols, Xb)
            cip = bool(stereo_diff(ref_stereo, row_stereo, al["atom_map"], metal=metal))
            sat_d = m2.ring_conformer_diff(refX, al["Xr_ref"], rings["sat"])
            chel_d = m2.ring_conformer_diff(refX, al["Xr_ref"], rings["chel"])
            z_flag = bool(cip or nd > 0 or fold)
            residual = m2.split_residual(sat_d, chel_d, shares); diag["residual"] = residual
            diag.update(n_centres=nc, n_flipped=len(flipped), n_diastereo=nd, cip_mismatch=cip,
                        chelate_fold=json.dumps([[symbols[a] + str(a) for a in o] + [round(t0), round(t1)] for o, t0, t1 in fold]),
                        sat_diff=json.dumps([round(x) for x in sat_d]), chel_diff=json.dumps([round(x) for x in chel_d]),
                        shares=json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in shares.items()}),
                        best_seed=best["seed"], best_row=best["row_index"], best_mirror=al["mirror"],
                        n_sat_rings=len(rings["sat"]), n_chel_rings=len(rings["chel"]))
        except Exception as exc:
            diag["error"] = f"{type(exc).__name__}: {exc}"
            print(f"A4t2 {code}: diagnosis failed {exc}", flush=True)
    ver = m2.verdict([r for r in recs if r["relax_status"] == "ok"])
    cls, cls_miss = m2.decision2(ver["status"], donor_mismatch, geom_mismatch, any_shell, z_flag, residual, code=code,
                                 known_typing=KNOWN_TYPING, geometry_evaluable=geometry_evaluable)
    seed_cls = {}
    for s in SEEDS:
        vs = m2.verdict([r for r in recs if r["seed"] == s and r["relax_status"] == "ok"])
        seed_cls[s] = m2.decision2(vs["status"], donor_mismatch, geom_mismatch, any(r["shell_match"] for r in recs if r["seed"] == s),
                                   z_flag, residual, code=code, known_typing=KNOWN_TYPING, geometry_evaluable=geometry_evaluable)[0]
    okr = [r for r in recs if r["relax_status"] == "ok"]
    lm = min((r["dE_mace"] for r in okr if r["dE_mace"] is not None), default=None)
    lg = min((r["dE_gfn2"] for r in okr if r["dE_gfn2"] is not None), default=None)
    prow.append(dict(code=code, classification=cls, class_if_miss=cls_miss, classification_old=old.get(code),
                     F_mace=ver["F_mace"], F_gfn2=ver["F_gfn2"],
                     class_mace=("F" if ver["F_mace"] else cls_miss), class_gfn2=("F" if ver["F_gfn2"] else cls_miss),
                     per_seed_classes=json.dumps({str(k): v for k, v in seed_cls.items()}),
                     n_rows=len(recs), n_ok=len(okr), n_hit=sum(r["hit"] for r in okr), n_hit_proper=sum(r["hit_proper"] for r in okr),
                     n_shell_match=len(cand), n_eq_mace=sum(r["dE_mace"] is not None and r["dE_mace"] <= m2.EQ_KJ for r in okr),
                     n_eq_gfn2=sum(r["dE_gfn2"] is not None and r["dE_gfn2"] <= m2.EQ_KJ for r in okr),
                     lowest_dE_mace=lm, lowest_dE_gfn2=lg,
                     best_rmsd_proper=min((r["rmsd_proper"] for r in okr), default=None),
                     best_rmsd_any=min((r["rmsd_any"] for r in okr), default=None),
                     donor_set_mismatch=donor_mismatch, geometry_mismatch=geom_mismatch, geometry_evaluable=geometry_evaluable,
                     z_flag=z_flag, automorph_capped=bool(capped), n_automorph=len(P), **diag))
    print(f"A4t2 {code}: v2={cls} (mace {prow[-1]['class_mace']}, gfn2 {prow[-1]['class_gfn2']}, miss-class {cls_miss}) "
          f"old={old.get(code)} rows={len(recs)}", flush=True)

    return prow, all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4b", required=True)
    ap.add_argument("--a4t", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parents", default=None, help="comma separated codes (default: the 36 of a4t_parents.json via --parents-json)")
    ap.add_argument("--parents-json", default=None)
    ap.add_argument("--old", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    codes = a.parents.split(",") if a.parents else (json.loads(Path(a.parents_json).read_text()) if a.parents_json else None)
    _, _, meta = analyse2(a.a4b, a.a4t, a.out, codes, a.old, a.workers)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
