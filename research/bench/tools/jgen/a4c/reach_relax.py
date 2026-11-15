"""Relax the reach probe's solved rows and judge them with the panel's hit rule.

    python -m tools.jgen.a4c.reach_relax --reach REACH_DIR --inputs A4B_INPUTS --out OUT [--level normal] [--workers 28]

`a4c.reach` starts the production solver at the reference and reports how far the solved rows drift. That drift is an
unrelaxed distance, and the panel's hit is decided after relaxation (rows 0.3-0.6 A away before relaxation hit), so the
drift alone cannot say "the score does not hold the basin". Here every solved row is GFN2-relaxed (same level for rows
and reference), given its MACE offset to the relaxed reference, and compared with `a4c.classes.Reference.compare`.
Per parent:
  hold        a solved row hits (the solver's surface has a maximum that relaxes into the reference basin or below it)
  would_keep  ... and its score is at or above the pool's lowest kept score (it would have been output): search miss
  low_score   ... but it scores below every kept pool row: the basin exists and the ranking/selection drops it
  no_hold     no solved row hits: the solver's coordinates or score cannot represent the basin near the reference
Per-term score differences (solved-at-reference row minus the pool top-1, nats) name the term that pays.
`--print-only` re-prints a stored OUT/reach_relax.json.
Never fails the batch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", type=Path, required=True, help="tools.run diagnostic output of a4c.reach:probe")
    ap.add_argument("--inputs", type=Path, required=True, help="A4b inputs (charge_mult.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", default="normal")
    ap.add_argument("--cycles", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--mace-device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--rows", type=int, default=6, help="solved rows relaxed per parent: the best-scoring same-class rows first")
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    if a.print_only:
        return report(json.loads((a.out / "reach_relax.json").read_text()))
    from tools.runkit import xtb, mace
    from tools.jgen.a4c.classes import Reference
    a.out.mkdir(parents=True, exist_ok=True)
    cm = json.loads((a.inputs / "charge_mult.json").read_text())
    probes = []
    for p in sorted(a.reach.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for r in (d if isinstance(d, list) else [d]):
            r = r.get("result", r) if isinstance(r, dict) else None
            if isinstance(r, dict) and r.get("rows") and r.get("code") and "X_solved" in r["rows"][0]:
                probes.append(r)
    probes = list({(r["code"], r["seed"]): r for r in probes}.values())
    jobs, slots, refs = [], [], {}
    for r in probes:
        code = r["code"]
        run = None
        q, mult = int(cm[code]["charge"]), int(cm[code]["mult"])
        syms = r["ref_symbols"]
        refs[(code, r["seed"])] = (syms, np.asarray(r["ref_X"], float), q, mult)
        jobs.append((syms, np.asarray(r["ref_X"], float), q, mult)); slots.append((code, r["seed"], "ref"))
        order = sorted(range(len(r["rows"])), key=lambda j: (not r["rows"][j]["solved_same"], -r["rows"][j]["score"]))
        seen = []
        for j in order:
            X = np.asarray(r["rows"][j]["X_solved"], float)
            if any(np.sqrt(((X - Y) ** 2).sum(1).mean()) < 0.05 for Y in seen):     # identical solves of one basin
                continue
            seen.append(X)
            jobs.append((syms, X, q, mult)); slots.append((code, r["seed"], j))
            if len(seen) >= a.rows:
                break
    xtb.require()
    order = sorted(range(len(jobs)), key=lambda i: -len(jobs[i][0]))
    rr = xtb.opt_many([jobs[i] for i in order], workers=a.workers, workroot=str(a.out / "tmp"), cycles=a.cycles,
                      level=a.level, timeout=a.timeout)
    res = [None] * len(jobs)
    for i, x in zip(order, rr):
        res[i] = x
    ok = [i for i, x in enumerate(res) if x.get("X") is not None and x.get("status") in ("ok", "not_converged")]
    ev = mace.Evaluator(device="cuda") if a.mace_device == "cuda" else mace.PoolEvaluator(a.workers)
    E = dict(zip(ok, map(float, ev.energies_kj([(jobs[i][0], res[i]["X"], jobs[i][2], jobs[i][3]) for i in ok]))))
    by = {}
    for i, s in enumerate(slots):
        by.setdefault((s[0], s[1]), {})[s[2]] = i
    out = []
    for r in probes:
        key = (r["code"], r["seed"])
        rec = dict(code=r["code"], seed=r["seed"], pool_top1_score=r.get("pool_top1_score"), pool_min_score=r.get("pool_min_score"),
                   pool_n=r.get("pool_n"), pool_best_same_class=r.get("pool_best_same_class"), score_norm=r.get("score_norm"),
                   fit_rmsd_min=r.get("fit_rmsd_min"), rows=[], error=None)
        try:
            ir = by[key]["ref"]
            if ir not in E:
                raise ValueError(f"reference relaxation failed: {res[ir].get('status')}")
            ref = Reference(r["ref_symbols"], res[ir]["X"], graph_X=np.asarray(r["ref_X"], float))
            for j, i in by[key].items():
                if j == "ref":
                    continue
                row = r["rows"][j]
                d = dict(j=j, score=row["score"], solved_core=row["solved_core"], solved_same=row["solved_same"],
                         fit_rmsd=row["fit_rmsd"], terms=row.get("terms"), status=res[i].get("status"), cycles=res[i].get("cycles"))
                if i in E:
                    v = ref.compare(res[i]["X"], energy_offset=E[i] - E[ir])
                    d.update(offset=E[i] - E[ir], same_class=v["same_class"], core_rmsd=v["core_rmsd_A"], hit=v["hit"],
                             hit_by=v["hit_by"], reason=v["reason"], enantiomer=v["enantiomer"])
                rec["rows"].append(d)
            hits = [d for d in rec["rows"] if d.get("hit")]
            best = max(hits, key=lambda d: d["score"]) if hits else None
            rec["verdict"] = ("no_hold" if best is None else
                              "would_keep" if (r.get("pool_min_score") is None or best["score"] >= r["pool_min_score"]) else "low_score")
            rec["best_hit"] = None if best is None else {k: best[k] for k in ("j", "score", "offset", "core_rmsd", "hit_by", "solved_core")}
            rec["min_offset_same_class"] = min((d["offset"] for d in rec["rows"] if d.get("same_class")), default=None)
            ref_row = best or (max(rec["rows"], key=lambda d: d["score"]) if rec["rows"] else None)
            if ref_row and ref_row.get("terms") and r.get("pool_top1_terms"):
                rec["terms_minus_top1"] = {k: ref_row["terms"][k] - r["pool_top1_terms"].get(k, 0.0) for k in ref_row["terms"]}
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out.append(rec)
    (a.out / "reach_relax.json").write_text(json.dumps(out, indent=1, default=float))
    report(out)


def report(out):
    def num(v, f):
        return "-" if v is None else format(v, f)
    for rec in sorted(out, key=lambda x: (str(x.get("verdict")), x["code"])):
        b = rec.get("best_hit") or {}
        z = rec.get("score_norm") or 1.0
        t = rec.get("terms_minus_top1") or {}
        if b:
            what = (f"hit: score {num(b.get('score'), '.3f')} offset {num(b.get('offset'), '+.1f')} core {num(b.get('core_rmsd'), '.2f')} "
                    f"({b.get('hit_by')}; unrelaxed core {num(b.get('solved_core'), '.2f')})")
        else:
            what = f"min same-class offset {num(rec.get('min_offset_same_class'), '+.1f')}"
        print(rec["code"], rec["seed"], rec.get("verdict") or rec["error"], what,
              f"| pool top {num(rec.get('pool_top1_score'), '.3f')} min {num(rec.get('pool_min_score'), '.3f')} n {rec.get('pool_n')}",
              "| terms vs pool top-1 (nats):", " ".join(f"{k}:{v * z:+.1f}" for k, v in t.items()))


if __name__ == "__main__":
    main()
