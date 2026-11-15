"""Gate follow-up: is a D-minus-C regression a search failure (the better basin is not in D's pool) or a scoring failure (D's scorer ranks it below)?

    python gate_cross_rescore.py NATIVE_ARM OUT.json CODE [CODE ...]      # NATIVE_ARM = C or D; run once per arm (the flags of the pool's own arm are set at import)

The saved J3 pools of NATIVE_ARM (stage 1 `all` root) are re-scored with the fixed-pool scorer (``tools.jgen.ds.stress_trial``) under the pool's own DB (validation: the selected
row must equal the saved one) and under the other arm's DB (flags stay those of the pool's arm, as in the earlier fixed-pool comparisons). Selection is argmax score per seed.
Reported per seed: selected-row dE (energy minus the record's reference energy) under both DBs, the pool floor, and the rank of the pool's lowest-energy row.
"""
import json, sys
from pathlib import Path
import numpy as np

MAIN = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full")
GATE = Path(__file__).resolve().parents[4] / "research/data/ds_proto/gate"
ROOT = GATE / "stage1/all"
DBS = {"C": MAIN / "data/db/constraints_2026-11-01.json", "D": GATE / "DF/constraints.json"}


def parent_row(code, native):
    from tools.jgen.ds import cross_score as CS
    from tools.jgen.inputs import frame_shards

    frame = frame_shards(MAIN / "data/refit/frame_v1")
    dbs = {k: {"path": str(v), "sha256": CS._sha(v), "meta_sha256": CS._sha(v.with_suffix(".meta.json"))} for k, v in DBS.items()}
    paths = {native: [{"record": str(ROOT / native / "records" / f"{code}_{s}.json"),
                       "pool": json.loads((ROOT / native / "records" / f"{code}_{s}.json").read_text())["pool_manifest"]} for s in (11, 12, 13)]}
    return dict(id=f"{code}_{native}", code=code, runtime=native, paths=paths, dbs=dbs, shard=frame[code][0], dataset=frame[code][1], ref_dir=str(MAIN / "data/ref"))


def main():
    native, out, codes = sys.argv[1], sys.argv[2], sys.argv[3:]
    from tools.jgen.ds import stress_trial as ST

    other = "D" if native == "C" else "C"
    result = {}
    for code in codes:
        parent = parent_row(code, native)
        recs = {s: json.loads(Path(p["record"]).read_text()) for s, p in zip((11, 12, 13), parent["paths"][native])}
        ctx = ST._load(parent)                                   # scorer under the pool's own DB
        own = ST._score(ctx["state"], ctx["ref_d"], ctx["lookup"])
        oth_lookup = ctx["Traced"](ctx["load_db"](str(DBS[other])))
        alt = ST._score(ctx["state"], ctx["ref_d"], oth_lookup)
        entry = {}
        for i, (seed, _X, _g, energy) in enumerate(ctx["state"]["per_seed"]):
            rec = recs[seed]; ref = rec["ref_energy_kj"]
            saved_top = int(np.argmax(rec["scores"]))
            n = len(energy)
            sc_own, sc_alt = own["scores"][i][:n], alt["scores"][i][:n]
            lo = int(np.argmin(energy))
            rank = lambda sc, j: int((sc > sc[j]).sum()) + 1
            entry[seed] = dict(
                saved_top_row=saved_top, own_top_row=int(np.argmax(sc_own)), validation_same_row=bool(saved_top == int(np.argmax(sc_own))),
                dE_floor=float(energy.min() - ref), dE_selected_saved=float(energy[saved_top] - ref),
                dE_selected_own_db=float(energy[int(np.argmax(sc_own))] - ref), dE_selected_other_db=float(energy[int(np.argmax(sc_alt))] - ref),
                lowest_energy_row_rank_own_db=rank(sc_own, lo), lowest_energy_row_rank_other_db=rank(sc_alt, lo),
                score_gap_top_to_lowest_own=float(sc_own.max() - sc_own[lo]), score_gap_top_to_lowest_other=float(sc_alt.max() - sc_alt[lo]),
                top_score_own_db=float(sc_own.max()), top_score_other_db=float(sc_alt.max()), bend_prior_top1=float(rec.get('top1_bend_prior') or 0.0),
                n_rows=n)
        result[code] = dict(native=native, scored_under_other=other, seeds=entry)
        print(code, native, {s: (round(e["dE_selected_saved"], 1), round(e["dE_selected_other_db"], 1), round(e["dE_floor"], 1), e["validation_same_row"]) for s, e in entry.items()}, flush=True)
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
