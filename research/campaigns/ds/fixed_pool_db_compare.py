"""Step 5a: re-score the frozen D3 D-arm pools (seeds 11-13) under candidate constraints DBs; DB is the only thing that varies.

    python research/campaigns/ds/fixed_pool_db_compare.py OUT.json NAME=path/to/constraints.json [...] [--cases A B ...]

The control is the pools' generating DB (recorded D2). Each arm is scored on the same rows; selection is argmax per seed. Per case and arm:
the confidence.evaluate outcome (selected-energy change, new broken coordination, severe rule), whether the selected row changed,
and the selected row's score margin over the runner-up. A case whose pool signature changed (repaired target) has no valid pool and is reported as such.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

MAIN = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full")
from fit.pipeline import confidence
from tools.jgen.ds import cross_score as CS
from tools.jgen.ds import stress_trial as ST


def _selected(res):
    sel = np.argmax(res["scores"], axis=1)
    srt = np.sort(res["scores"], axis=1)
    return sel.tolist(), (srt[:, -1] - srt[:, -2]).tolist()


def main(argv):
    out, rest = Path(argv[0]), argv[1:]
    cases = CS.CASES
    if "--cases" in rest:
        i = rest.index("--cases"); cases = rest[i + 1:]; rest = rest[:i]
    arms = dict(a.split("=", 1) for a in rest)
    rows = []
    for code in cases:
        t0 = time.time()
        try:
            parent = ST.manifest_row(MAIN, code, "D")
            score_db, info = ST.prepare_db_scorer(parent)
            clean = score_db(parent["dbs"]["D"]["path"])
        except Exception as exc:                                # a case without a valid pool is data, not a crash
            rows.append(dict(code=code, status="no_valid_pool", reason=f"{type(exc).__name__}: {exc}"))
            print(code, "no valid pool:", str(exc)[:100], flush=True)
            continue
        sel0, margin0 = _selected(clean)
        for name, path in arms.items():
            try:
                res = score_db(path)
                ev = confidence.evaluate(clean, res)
                sel, margin = _selected(res)
                rows.append(dict(code=code, status="ok", arm=name, pool_rows=info["pool_rows"], selected_control=sel0, selected=sel,
                                 changed_seeds=int(sum(a != b for a, b in zip(sel0, sel))), margin_control=margin0, margin=margin,
                                 energy_control_kj=[float(clean["energies_kj"][s, sel0[s]]) for s in range(3)],
                                 energy_kj=[float(clean["energies_kj"][s, sel[s]]) for s in range(3)], **ev))
                print(code, name, "changed", rows[-1]["changed_seeds"], "loss", np.round(ev["energy_loss_kj"], 1), "severe", ev["severe"], flush=True)
            except Exception as exc:
                rows.append(dict(code=code, status="failed", arm=name, reason=f"{type(exc).__name__}: {exc}"))
                print(code, name, "FAILED", exc, flush=True)
        print(code, "done %.0fs" % (time.time() - t0), flush=True)
    out.write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
