"""Summary tables for the A4c retrospective (parent-clustered bootstrap, 10k resamples, seed 20261112).

python -m tools.jgen.a4c.summary --a4b <A4b dir> --retro <retro dir> [--retro-alt <depth-1 retro dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

B, SEED = 10_000, 20261112
OLD = ("R-TOP1", "R-RANK", "R-MISS")
NEW = ("C-TOP1", "C-RANK", "C-MISS")


def boot(per_parent):
    v = np.asarray(per_parent, float)
    if not len(v):
        return dict(n=0, mean=None, ci=None)
    rng = np.random.default_rng(SEED)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return dict(n=len(v), mean=float(v.mean()), ci=[float(np.quantile(m, .025)), float(np.quantile(m, .975))])


def shares(units, key, labels):
    by = defaultdict(list)
    for u in units:
        by[u["code"]].append(u[key])
    return {b: boot([np.mean([x == b for x in xs]) for xs in by.values()]) for b in labels}


def load(retro):
    return json.loads((Path(retro) / "parents.json").read_text())


def joined(a4b, retro):
    old = {(u["code"], int(u["seed"])): u for u in csv.DictReader((Path(a4b) / "units.csv").open())}
    d = load(retro)
    units, rows, parents = [], [], d["parents"]
    for p in parents:
        for u in p.get("units", []):
            o = old.get((u["code"], u["seed"]), {})
            if u.get("status") != "ok" or not o.get("r_bucket"):
                continue
            units.append(dict(u, panel=o["panel"], r_bucket=o["r_bucket"], capped=p.get("capped")))
        rows += p.get("rows", [])
    return d, units, rows, parents


def build(a4b, retro, retro_alt=None):
    d, units, rows, parents = joined(a4b, retro)
    out = dict(locked_dropped=d["locked_dropped"], depth=d["depth"],
               parent_errors={p["code"]: p["error"] for p in parents if p.get("error")})
    out["capped_parents"] = sorted(p["code"] for p in parents if p.get("capped"))
    # counterfactual: units that would hit if enantiomers counted as the same class (owner 2026-11-12: they do not)
    ena = set()
    for p in parents:
        for r in p.get("rows", []):
            if r.get("enantiomer") and ((r.get("core_rmsd") or 9) <= 0.3 or (r.get("offset") is not None and r["offset"] <= 5)):
                ena.add((p["code"], r["seed"]))
    out["enantiomer_only_units"] = sorted(u for u in ena if not any(x["code"] == u[0] and x["seed"] == u[1]
                                                                    and x["bucket"] != "C-MISS" for x in units))
    # owner's top-1 criterion: top-1 relaxes to a basin at or below the reference energy (any class)
    tops = {}
    for p in parents:
        for r in p.get("rows", []):
            if r.get("is_top1"):
                tops[(p["code"], r["seed"])] = r.get("offset") if r.get("same_class") is not None else None
    for u in units:
        o = tops.get((u["code"], u["seed"]))
        u["top1_le5"] = o is not None and o <= 5.0
        u["top1_le0"] = o is not None and o <= 0.0
    for panel in ("cal", "tail"):
        us = [u for u in units if u["panel"] == panel]
        t = Counter((u["r_bucket"], u["bucket"]) for u in us)
        out[panel] = dict(n_units=len(us), n_parents=len({u["code"] for u in us}),
                          old=shares(us, "r_bucket", OLD), new=shares(us, "bucket", NEW),
                          transitions={f"{a}->{b}": t[(a, b)] for a in OLD for b in NEW if t[(a, b)]},
                          miss_to_hit=sorted({u["code"] for u in us if u["r_bucket"] == "R-MISS" and u["bucket"] != "C-MISS"}),
                          hit_to_miss=sorted({u["code"] for u in us if u["r_bucket"] != "R-MISS" and u["bucket"] == "C-MISS"}),
                          top1_energy_le5=boot([np.mean([u["top1_le5"] for u in us if u["code"] == c])
                                                for c in sorted({u["code"] for u in us})]),
                          top1_energy_le0=boot([np.mean([u["top1_le0"] for u in us if u["code"] == c])
                                                for c in sorted({u["code"] for u in us})]),
                          strict_any_new=boot([any(u["bucket"] != "C-MISS" for u in us if u["code"] == c)
                                               for c in sorted({u["code"] for u in us})]),
                          strict_all_new=boot([all(u["bucket"] != "C-MISS" for u in us if u["code"] == c)
                                               for c in sorted({u["code"] for u in us})]))
    ev = [r for r in rows if "same_class" in r]
    out["rows"] = dict(
        n_relaxed_rows=len(ev),
        hit_by=Counter(str(r["hit_by"]) for r in ev),
        old_equivalent_other_class=sorted({(r["code"], r["seed"], r["row"]) for r in ev if r["old_equivalent"] and not r["same_class"]}),
        old_rmsd_hit_other_class=sorted({(r["code"], r["seed"], r["row"]) for r in ev if r["old_hit"] and not r["same_class"]}),
        old_rmsd_hit_new_nonhit=sorted({(r["code"], r["seed"], r["row"]) for r in ev if r["old_hit"] and not r["hit"]}),
        enantiomer_rows=sorted({(r["code"], r["seed"], r["row"]) for r in ev if r.get("enantiomer")}),
        not_same_class=Counter(str(r.get("reason")).split("[")[0] for r in ev if not r["same_class"]),
        old_equivalent_other_class_reason=Counter(str(r.get("reason")).split("[")[0] for r in ev
                                                  if r["old_equivalent"] and not r["same_class"]),
        core_gt_full=sum(1 for r in ev if r["core_rmsd"] is not None and r["core_rmsd"] > r["full_rmsd"] + 0.05),
        guided_rows=sum(1 for r in ev if r.get("guided")))
    capped = set(out["capped_parents"])
    diffs = [abs(r["full_rmsd"] - r["old_rmsd"]) for r in ev if r["old_rmsd"] is not None and r["code"] not in capped]
    out["order_check"] = dict(n_rows=len(diffs), max_abs_full_rmsd_diff_uncapped=float(max(diffs)) if diffs else None)
    bias = [u for u in units if u.get("n_screen_class_core")]
    out["selection_bias"] = dict(units_with_unrelaxed_class_core_row_never_relaxed=len(bias),
                                 units_with_such_row_beyond_1p5A=sum(bool(u.get("n_screen_class_core_beyond_1p5")) for u in bias),
                                 cal_new_miss_units=sum(u["bucket"] == "C-MISS" and u["panel"] == "cal" for u in bias),
                                 of_which_new_miss=sum(u["bucket"] == "C-MISS" for u in bias),
                                 parents=sorted({u["code"] for u in bias}), rows=sum(u["n_screen_class_core"] for u in bias),
                                 screened_rows=sum(u.get("n_screened", 0) for u in units))
    if retro_alt:
        _, ua, _, _ = joined(a4b, retro_alt)
        alt = {(u["code"], u["seed"]): u["bucket"] for u in ua}
        out["depth_sensitivity"] = dict(depth_alt=load(retro_alt)["depth"],
            cal_new_alt=shares([u for u in ua if u["panel"] == "cal"], "bucket", NEW),
            changed_units=sorted((u["code"], u["seed"], u["bucket"], alt.get((u["code"], u["seed"])))
                                 for u in units if alt.get((u["code"], u["seed"])) not in (None, u["bucket"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4b", required=True); ap.add_argument("--retro", required=True); ap.add_argument("--retro-alt")
    a = ap.parse_args()
    s = build(a.a4b, a.retro, a.retro_alt)
    (Path(a.retro) / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
