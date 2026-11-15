"""Does the A4c verdict depend on the relaxation level? Re-relax a post's rows at another xtb level and re-evaluate.

    python -m tools.jgen.a4c.relax_level --src POST/a4b --a4c POST/a4c --out NEW --level normal --cycles 1500 \
        --buckets C-MISS --workers 28 --mace-device cuda

The A4b/A4c post relaxes pool rows with `xtb --opt crude` (energy 5e-4 Eh, gradient 1e-2 Eh/a0). The tmQM references are
GFN2 minima already (1-2 cycles), so a crude-converged row can sit kJ/mol above its own minimum and a few tenths of an
Angstrom from it while the reference sits exactly on its minimum: a one-sided bias against the row in both legs of the
hit (core RMSD <= 0.3 A, MACE offset <= +5 kJ/mol). This tool measures it.

NEW/a4b is an A4b-layout directory: everything except relax/ is linked from --src; relax/ holds
  * every reference re-relaxed at --level from its stored relaxed geometry;
  * the rows of the selected units (--buckets, from --a4c parents.json; default every unit) re-relaxed at --level from
    their stored crude-relaxed geometry, MACE energy and offset recomputed against the re-relaxed reference. The crude
    values are kept under "crude". A row whose re-relaxation fails keeps its crude record (flag level_failed);
  * the other units copied unchanged (their stored offsets stay crude row vs crude reference).
NEW/a4c is `a4c.baseline.evaluate` on it: the same code path as every post. Never fails the batch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

KEEP_CRUDE = ("E_h", "cycles", "converged", "E_mace_kj", "offset", "wall_s", "status")


def _selected(a4c: Path | None, buckets):
    if a4c is None or not buckets:
        return None
    d = json.loads((a4c / "parents.json").read_text())
    return {(u["code"], int(u["seed"])) for p in d["parents"] for u in p.get("units", []) if u.get("bucket") in buckets}


def run(src: Path, out: Path, level: str, cycles: int, workers: int, mace_device: str, a4c: Path | None, buckets,
        timeout: float, codes=None) -> dict:
    from tools.runkit import xtb, mace
    t0 = time.perf_counter()
    data = out / "a4b"
    (data / "relax").mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name in ("relax", "tmp"):
            continue
        d = data / p.name
        if not (d.exists() or d.is_symlink()):
            d.symlink_to(p.resolve(), target_is_directory=p.is_dir())
    for name in ("selection.json",):
        if (src / "relax" / name).exists():
            shutil.copy2(src / "relax" / name, data / "relax" / name)
    cm = json.loads((src / "charge_mult.json").read_text())
    refs = json.loads((src / "relax" / "references.json").read_text())
    sel = _selected(a4c, buckets)
    units = {}
    for p in sorted((src / "relax").glob("*_*.json")):
        code, _, seed = p.stem.rpartition("_")
        if not seed.isdigit() or (codes and code not in codes):
            continue
        units[(code, int(seed))] = json.loads(p.read_text())
    syms = {}
    for code in {c for c, _ in units}:
        mp = next(iter(sorted((src / "xyz" / code).glob(f"{code}_*_pool.json"))), None)
        if mp is not None:
            syms[code] = json.loads(mp.read_text())["ref_symbols"]
    jobs, slots = [], []
    for code, r in refs.items():
        if code in syms and r.get("X") is not None:
            jobs.append((syms[code], np.asarray(r["X"], float), int(cm[code]["charge"]), int(cm[code]["mult"])))
            slots.append(("ref", code))
    for (code, seed), u in units.items():
        if (sel is not None and (code, seed) not in sel) or code not in syms:
            continue
        for k, r in u["rows"].items():
            if r.get("X") is not None and r.get("status") in ("ok", "capped"):
                jobs.append((syms[code], np.asarray(r["X"], float), int(cm[code]["charge"]), int(cm[code]["mult"])))
                slots.append(("row", code, seed, k))
    order = sorted(range(len(jobs)), key=lambda i: -len(jobs[i][0]))
    xtb.require()
    rr = xtb.opt_many([jobs[i] for i in order], workers=workers, workroot=str(out / "tmp"), cycles=cycles, level=level,
                      timeout=timeout) if jobs else []
    res = [None] * len(jobs)
    for i, r in zip(order, rr):
        res[i] = r
    t_x = time.perf_counter() - t0
    okk = [i for i, r in enumerate(res) if r.get("X") is not None and r.get("status") in ("ok", "not_converged")]
    ev = mace.Evaluator(device="cuda") if mace_device == "cuda" else mace.PoolEvaluator(workers)
    E = ev.energies_kj([(jobs[i][0], res[i]["X"], jobs[i][2], jobs[i][3]) for i in okk]) if okk else []
    emap = {i: float(e) for i, e in zip(okk, E)}
    new_refs = {}
    n_ref_fail = 0
    for i, s in enumerate(slots):
        if s[0] != "ref":
            continue
        old = refs[s[1]]
        if i in emap and res[i]["status"] == "ok":
            r = res[i]
            new_refs[s[1]] = dict(old, X=r["X"].tolist(), E_h=r["E_h"], cycles=r["cycles"], converged=r["converged"],
                                  E_mace_kj=emap[i], level=level, crude={k: old.get(k) for k in KEEP_CRUDE})
        else:
            new_refs[s[1]] = dict(old, level_failed=res[i].get("status")); n_ref_fail += 1
    for code, r in refs.items():
        new_refs.setdefault(code, r)
    (data / "relax" / "references.json").write_text(json.dumps(new_refs, allow_nan=False))
    touched = {}
    for i, s in enumerate(slots):
        if s[0] != "row":
            continue
        _, code, seed, k = s
        old = units[(code, seed)]["rows"][k]
        ref = new_refs[code]
        if i in emap and ref.get("level") == level and res[i]["status"] in ("ok", "not_converged"):
            r = res[i]
            new = dict(old, X=r["X"].tolist(), E_h=r["E_h"], cycles=r["cycles"], converged=r["converged"],
                       E_mace_kj=emap[i], offset=emap[i] - ref["E_mace_kj"], level=level,
                       level_status=r["status"], level_wall_s=r["wall_s"],
                       level_shift_A=float(np.sqrt(((r["X"] - np.asarray(old["X"])) ** 2).sum(1).mean())),
                       crude={kk: old.get(kk) for kk in KEEP_CRUDE})
        else:
            new = dict(old, level_failed=res[i].get("status"))
        touched.setdefault((code, seed), {})[k] = new
    for (code, seed), u in units.items():
        rows = dict(u["rows"])
        rows.update(touched.get((code, seed), {}))
        (data / "relax" / f"{code}_{seed}.json").write_text(json.dumps(dict(u, rows=rows), allow_nan=False))
    n_row = sum(s[0] == "row" for s in slots)
    meta = dict(src=str(src), level=level, cycles=cycles, n_refs=len(slots) - n_row, n_rows=n_row, n_ref_failed=n_ref_fail,
                n_row_failed=sum(1 for v in touched.values() for r in v.values() if "level_failed" in r),
                units_touched=len(touched), xtb_wall_s=t_x, wall_s=time.perf_counter() - t0,
                buckets=sorted(buckets) if buckets else None)
    (out / "relax_level.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="POST/a4b")
    ap.add_argument("--a4c", type=Path, help="POST/a4c (for --buckets)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", default="normal")
    ap.add_argument("--cycles", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--mace-device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--buckets", nargs="*", help="re-relax only units in these A4c buckets (default: every unit)")
    ap.add_argument("--codes", nargs="*")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    meta = run(a.src, a.out, a.level, a.cycles, a.workers, a.mace_device, a.a4c, set(a.buckets or ()), a.timeout,
               set(a.codes) if a.codes else None)
    print(json.dumps(meta, indent=1))
    if not a.skip_eval:
        from tools.jgen.a4c.baseline import evaluate
        s = evaluate(a.out / "a4b", a.out / "a4c", workers=min(a.workers, 24))
        print(json.dumps({p: {k: s[p].get(k) for k in ("n_units", "miss", "top1_energy_le0", "top1_energy_le5")}
                          for p in ("cal", "tail") if p in s}, indent=1, default=str))


if __name__ == "__main__":
    main()
