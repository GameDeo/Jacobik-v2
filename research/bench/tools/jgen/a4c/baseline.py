"""A4c evaluation of a fresh A4b-layout run (P2R Step 2 baseline; called by `tools.jgen.a4b.rerun post --select a4c`).

    python -m tools.jgen.a4c.baseline --data OUT/a4b --out OUT/a4c --workers 128

data/ is the post stage's A4b directory: xyz/ pools, relax/ (rows chosen by a4c.relax_select, GFN2 crude + MACE),
relax/selection.json, relax/references.json, units.csv (A4b tables). Per parent `retro.parent` (same code as the
validated retrospective) gives the A4c rows and buckets; `summary.build` the parent-bootstrap tables; this module adds
the Step-2 diagnostics:
  * mirror-completed miss: a unit also counts as covered when a relaxed row is the reference's enantiomer at hit level
    (what Step-4 mirror completion would add; enantiomers stay distinct classes, owner 2026-11-12);
  * offered vs not offered: C-MISS units split by whether ANY unrelaxed pool row is the reference class
    (selection.json n_same_class_any): not offered -> the class is absent from the pool before relaxation;
  * class completeness: classes per unit in the unrelaxed pool (n_classes), rows the cap dropped;
  * garbage: relaxed rows with a covalent/contact topology change or > +50 kJ/mol over the relaxed reference.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GARBAGE_KJ = 50.0
#: frozen source-based eligibility (stage 1 = this 120-parent cal+tail panel): primary root = valid + repaired
ELIGIBILITY = Path(__file__).resolve().parents[4] / "campaigns" / "ds" / "gate_eligibility_v1.json"


def evaluate(data, out, workers=3, depth=2, codes=None):
    from tools.jgen.a4c.retro import parent
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((data / "vm_split.json").read_text())
    have = {p.name for p in (data / "xyz").iterdir() if p.is_dir()}
    allc = sorted((set(split["cal_parents"]) | set(split["tail_parents"])) & have)
    if codes:
        allc = [c for c in allc if c in set(codes)]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(parent, [(str(data), c, depth) for c in allc], chunksize=1))
    (out / "parents.json").write_text(json.dumps(dict(locked_dropped=split.get("locked_excluded", []), depth=depth,
                                                      parents=res), default=float))
    s = summarise(data, out)
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    return s


def _boot_parent(units, fn):
    from tools.jgen.a4c.summary import boot
    by = defaultdict(list)
    for u in units:
        by[u["code"]].append(fn(u))
    return boot([np.mean(v) for _, v in sorted(by.items())])


def summarise(data, out):
    from tools.jgen.a4c.summary import build, joined
    data, out = Path(data), Path(out)
    s = build(str(data), str(out))
    _, units, rows, parents = joined(str(data), str(out))
    sel = {}
    sp = data / "relax" / "selection.json"
    if sp.exists():
        for v in json.loads(sp.read_text()).values():
            if "code" in v:
                sel[(v["code"], int(v["seed"]))] = v
    ena = {tuple(u) for u in s["enantiomer_only_units"]}
    garbage = defaultdict(int); n_rel = defaultdict(int)
    for r in rows:
        if "same_class" not in r:
            continue
        k = (r["code"], r["seed"]); n_rel[k] += 1
        garbage[k] += bool((r.get("topology_same") is False) or (r.get("offset") is not None and r["offset"] > GARBAGE_KJ))
    for u in units:
        k = (u["code"], u["seed"]); g = sel.get(k, {})
        u["miss"] = u["bucket"] == "C-MISS"
        u["miss_mirror_completed"] = u["miss"] and k not in ena
        u["offered"] = bool(g.get("n_same_class_any", 0))
        u["n_classes"] = g.get("n_classes")
        u["n_dropped"] = len(g.get("dropped", []))
        u["garbage_share"] = garbage[k] / n_rel[k] if n_rel[k] else None
    elig = {}
    if ELIGIBILITY.exists():
        elig = {p["code"]: p for p in json.loads(ELIGIBILITY.read_text())["parents"] if p.get("stage") == 1}
    for u in units:
        u["primary"] = bool(elig.get(u["code"], {}).get("in_corrected_primary"))
        u["stratum"] = elig.get(u["code"], {}).get("stratum")
    for panel in ("cal", "tail"):
        us = [u for u in units if u["panel"] == panel]
        if not us:
            continue
        pr = [u for u in us if u["primary"]]
        s[panel]["primary_root"] = dict(n_parents=len({u["code"] for u in pr}),
                                        miss=_boot_parent(pr, lambda u: u["miss"]) if pr else None,
                                        miss_mirror_completed=_boot_parent(pr, lambda u: u["miss_mirror_completed"]) if pr else None)
        s[panel]["miss_by_stratum"] = {st: _boot_parent([u for u in us if u["stratum"] == st], lambda u: u["miss"])
                                       for st in sorted({str(u["stratum"]) for u in us})
                                       if any(str(u["stratum"]) == st for u in us)}
        miss = [u for u in us if u["miss"]]
        nc = [u["n_classes"] for u in us if u["n_classes"] is not None]
        gs = [u["garbage_share"] for u in us if u["garbage_share"] is not None]
        s[panel].update(
            miss=_boot_parent(us, lambda u: u["miss"]),
            miss_mirror_completed=_boot_parent(us, lambda u: u["miss_mirror_completed"]),
            miss_not_offered=_boot_parent(us, lambda u: u["miss"] and not u["offered"]),
            miss_offered=_boot_parent(us, lambda u: u["miss"] and u["offered"]),
            miss_units_not_offered=sorted((u["code"], u["seed"]) for u in miss if not u["offered"]),
            miss_units_offered=sorted((u["code"], u["seed"]) for u in miss if u["offered"]),
            classes_per_unit=dict(median=float(np.median(nc)) if nc else None, max=max(nc) if nc else None),
            units_cap_dropped=sum(u["n_dropped"] > 0 for u in us),
            garbage_row_share=dict(mean=float(np.mean(gs)) if gs else None, units_any=sum(g > 0 for g in gs)))
    s["n_units_selection_failed"] = sum(1 for v in (json.loads(sp.read_text()).values() if sp.exists() else [])
                                        if "error" in v)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--codes", nargs="*")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    s = evaluate(a.data, a.out, a.workers, a.depth, a.codes)
    print(json.dumps({p: {k: s[p].get(k) for k in ("n_units", "miss", "miss_mirror_completed", "miss_not_offered",
                                                     "top1_energy_le0", "strict_any_new", "strict_all_new")}
                      for p in ("cal", "tail") if p in s}, indent=1, default=str))


if __name__ == "__main__":
    main()
