"""Seed dependence of the fixed-pool outcome: per case, seed and DB, the selected row's gap to the pool's energy floor.

    python fixed_pool_seed_spread.py OUT.json NAME=path.json [...] [--cases ...]

The first NAME is the reference (D2). Per seed: pool floor (kJ/mol), selected energy, gap = selected - floor, energy rank of the selected row (0 = floor),
rows within 25 kJ/mol of the floor, whether the selected row is broken against the 1.25 x source-distance rule. Spread = max - min of the gap over seeds.
"""
import json, sys
from pathlib import Path
import numpy as np

MAIN = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full")
from tools.jgen.ds import cross_score as CS
from tools.jgen.ds import stress_trial as ST


def main(argv):
    out, rest = Path(argv[0]), argv[1:]
    cases = CS.CASES
    if "--cases" in rest:
        i = rest.index("--cases"); cases = rest[i + 1:]; rest = rest[:i]
    arms = dict(a.split("=", 1) for a in rest)
    rows = []
    for code in cases:
        try:
            parent = ST.manifest_row(MAIN, code, "D")
            ctx = ST._load(parent)
        except Exception as exc:
            print(code, "no valid pool", str(exc)[:80], flush=True); continue
        st, ref_d = ctx["state"], ctx["ref_d"]
        for name, path in arms.items():
            res = ST._score(st, ref_d, ctx["Traced"](ctx["load_db"](path)))
            seeds = []
            for s in range(3):
                n = len(st["per_seed"][s][3])
                en = res["energies_kj"][s, :n]; sc = res["scores"][s, :n]
                sel = int(np.argmax(sc)); floor = float(en.min())
                ok = ~res["broken_coordination"][s, :n]
                vfloor = float(en[ok].min()) if ok.any() else None      # lowest energy among rows that keep every donor within 1.25 x source
                seeds.append(dict(seed=st["per_seed"][s][0], rows=n, floor=floor, floor_row_broken=bool(res["broken_coordination"][s, int(np.argmin(en))]),
                                  valid_floor=vfloor, gap_to_valid_floor=None if vfloor is None else float(en[sel] - vfloor), selected_energy=float(en[sel]), gap=float(en[sel] - floor),
                                  energy_rank=int((en < en[sel]).sum()), within25=int((en <= floor + 25).sum()),
                                  broken=bool(res["broken_coordination"][s, sel]), n_broken_in_pool=int(res["broken_coordination"][s, :n].sum())))
            gaps = [x["gap"] for x in seeds]
            rows.append(dict(code=code, arm=name, seeds=seeds, gap_spread=max(gaps) - min(gaps)))
            print(code, name, "gaps", np.round(gaps, 1).tolist(), "spread %.1f" % rows[-1]["gap_spread"], "ranks", [x["energy_rank"] for x in seeds],
                  "broken sel", [int(x["broken"]) for x in seeds],
                  "floor broken", [int(x["floor_row_broken"]) for x in seeds], "gap to valid floor", [None if x["gap_to_valid_floor"] is None else round(x["gap_to_valid_floor"], 1) for x in seeds], flush=True)
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
