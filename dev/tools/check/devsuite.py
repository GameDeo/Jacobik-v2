"""Dev suite manifest: about 30 parents stratified by pool size from cal + tail (never val, reserve or locked), plus the
eligible D3 named cases, fixed by sha256 (dev/tools/check/DEV_MANIFEST.json). Runs go through the unified runner:

    python -m tools.check.devsuite manifest       (rebuild the manifest and its sha256; changes what pm-quick runs)
    python -m tools.run panel --manifest dev/tools/check/DEV_MANIFEST.json --arms C,K --out DIR

(Replaces the earlier dev/tools/check/fastrun.py, whose static per-worker bins and one bench process per worker are
superseded by dev/tools/run.)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .common import REPO

TAIL_SPLIT = "tail"
#: giants: parents above either cap are set aside and listed in the manifest (never silently dropped). Caps from the
#: C-2b confirmation C arm (local 2060): pool rows (max over seeds 21-23) and end-to-end wall summed over the 3 seeds.
ROW_CAP = 256
COST_CAP_S = 60.0


def _panel_of(code: str, tail: set[str]) -> str:
    return TAIL_SPLIT if code in tail else "cal"


def build_dev_manifest(out_path: Path, n_strat: int = 30, seed: int = 20261108) -> dict:
    """About 30 parents stratified by pool size from cal + tail (never val, reserve or locked), plus the named
    failure cases that are eligible, with a fixed SHA256 so runs are comparable."""
    import random
    import sys
    sys.path.insert(0, str(REPO / "research/bench"))
    from tools.jgen.c2.rings import population
    from tools.jgen.bench.panels import TAIL
    from tools.jgen.d3.analyse import NAMED  # the D3 named cases

    codes, meta = population()
    codes = sorted(codes)
    sizes = {}
    from .common import main_checkout
    c2 = main_checkout() / "research/data/jgen/campaign/C2b/confirmation/C/records"
    for c in codes:
        # pool size = the largest C-arm pool over the three confirmation seeds
        for s in (21, 22, 23):
            f = c2 / f"{c}_{s}.json"
            if f.exists():
                try:
                    sizes[c] = max(sizes.get(c, 0), int(json.loads(f.read_text()).get("n_rows") or 0))
                except Exception:
                    pass
    costs = recorded_costs(c2.parent.parent, codes)
    big = lambda c: sizes.get(c, 0) > ROW_CAP or (costs.get(c) or 0.0) > COST_CAP_S
    known = sorted([c for c in codes if c in sizes], key=lambda c: (sizes[c], c))
    giants = [c for c in known if big(c)]
    pool = [c for c in known if not big(c)]
    rng = random.Random(seed)
    picked = []
    if pool:
        # size-stratified: equal-count strata by pool size, one draw each
        k = min(n_strat, len(pool))
        for i in range(k):
            lo, hi = i * len(pool) // k, (i + 1) * len(pool) // k
            picked.append(pool[rng.randrange(lo, max(hi, lo + 1))])
    named = [c for c in NAMED if c in codes and c not in picked and not big(c)]
    named_out = [c for c in NAMED if c not in codes]
    tail = set(TAIL)
    parents = [dict(code=c, panel=_panel_of(c, tail), rows_c2b=sizes.get(c), cost_s=costs.get(c), group="stratified")
               for c in picked]
    parents += [dict(code=c, panel=_panel_of(c, tail), rows_c2b=sizes.get(c), cost_s=costs.get(c), group="named")
                for c in named]
    body = dict(source="research/bench/tools/jgen/c2/rings.py population() minus locked_exclusions.json",
                row_cap=ROW_CAP, cost_cap_s=COST_CAP_S, seeds=[11], n_strat=n_strat, rng_seed=seed,
                parents=sorted(parents, key=lambda d: d["code"]),
                giants_set_aside=giants, named_not_eligible=named_out,
                unsized=[c for c in codes if c not in sizes],
                population_meta=meta)
    text = json.dumps(body, indent=1, sort_keys=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    out_path.with_suffix(".sha256").write_text(f"{digest}  {out_path.name}\n")
    return dict(path=str(out_path), sha256=digest, n_parents=len(parents), giants=giants)


def recorded_costs(ref_root: Path, codes: list[str]) -> dict:
    """Per-parent recorded wall from C-2b records (c2_timing.end_to_end_s x seeds, else wall_solve_s)."""
    out = {}
    rdir = ref_root / "C" / "records"
    for c in codes:
        tot = 0.0
        for f in rdir.glob(f"{c}_*.json"):
            try:
                r = json.loads(f.read_text())
                tot += float((r.get("c2_timing") or {}).get("end_to_end_s") or r.get("wall_solve_s") or 0.0)
            except Exception:
                pass
        if tot:
            out[c] = tot
    return out




def _cli(argv=None) -> int:
    import sys
    if (argv or sys.argv[1:])[:1] != ["manifest"]:
        print(__doc__)
        return 2
    print(json.dumps(build_dev_manifest(REPO / "dev/tools/check/DEV_MANIFEST.json"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
