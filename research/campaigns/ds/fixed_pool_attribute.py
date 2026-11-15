"""Step 5a attribution: which looked-up distributions explain a fixed-pool difference between the control DB and an alternative DB.

    python fixed_pool_attribute.py OUT.json CASE ALT_DB.json

Every (quantity, requested key) the scorer looks up is traced under both DBs. Where the two answers differ (different resolved key or mixture), the alternative
entry is swapped alone into the control lookup (marginal effect on the selected-row energy, 3 seeds); the interaction remainder is reported as the gap between
the sum of marginals and the full-DB difference.
"""
import json, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np

MAIN = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full")
from fit.pipeline import confidence
from tools.jgen.ds import cross_score as CS
from tools.jgen.ds import stress_trial as ST


def main(out, code, alt_path):
    parent = ST.manifest_row(MAIN, code, "D")
    ctx = ST._load(parent)
    state, ref_d, lookup, Traced, load_db, FD = ctx["state"], ctx["ref_d"], ctx["lookup"], ctx["Traced"], ctx["load_db"], ctx["FittedDist"]
    clean = ST._score(state, ref_d, lookup); trace0 = dict(lookup.trace)
    alt = Traced(load_db(alt_path)); full = ST._score(state, ref_d, alt); trace1 = dict(alt.trace)
    ev_full = confidence.evaluate(clean, full)
    print("full-DB difference:", np.round(ev_full["energy_loss_kj"], 1), flush=True)

    def sig(entry):
        res, d = entry
        d = d if isinstance(d, dict) else (asdict(d) if d is not None else None)
        return json.dumps([list(res) if isinstance(res, (list, tuple)) else res, None if d is None else {k: d.get(k) for k in ("means", "weights", "vars")}], default=str, sort_keys=True)

    differing = [k for k in trace0 if k in trace1 and sig(trace0[k]) != sig(trace1[k])]
    print(len(trace0), "looked-up keys;", len(differing), "differ between the DBs", flush=True)
    rows = []
    for k in differing:
        q, req = k
        res1, d1 = trace1[k]
        fd = d1 if not isinstance(d1, dict) else FD(**{x: d1[x] for x in d1 if x in FD.__dataclass_fields__})
        lookup.override.clear(); lookup.override[k] = (res1, fd) if q == "QO" else fd
        try:
            ev = confidence.evaluate(clean, ST._score(state, ref_d, lookup))
        finally:
            lookup.override.clear()
        loss = ev["energy_loss_kj"]
        if max(abs(x) for x in loss) > 0:
            rows.append(dict(quantity=q, requested=str(req), control_resolved=str(trace0[k][0]), alt_resolved=str(res1), loss_kj=loss))
    rows.sort(key=lambda r: -max(r["loss_kj"]))
    for r in rows[:12]:
        print(r["quantity"], r["requested"][:70], "->", r["alt_resolved"][:60], np.round(r["loss_kj"], 1), flush=True)
    Path(out).write_text(json.dumps(dict(code=code, alt=alt_path, full=ev_full, n_lookups=len(trace0), n_differ=len(differing), changing=rows), indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
