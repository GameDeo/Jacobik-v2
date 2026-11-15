"""Assemble validated fixed-pool comparisons, without loading or scoring parents."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text())


def _bundle(result, arm, seed):
    key = f"{arm}_{seed}"
    rows, residual = {}, {}
    for variant, data in result.get("variants", {}).items():
        if key not in data["pools"]:
            continue
        pool = data["pools"][key]
        residual.update(pool["native_score_residual"])
        for i, row in enumerate(pool["rows"]):
            index = row["row_index"]
            if index in rows:
                raise ValueError(f"duplicate row {key}/{index}")
            rows[index] = dict(row=row, variant=variant,
                               stat={db: v[i] for db, v in pool["stat"].items()},
                               terms={db: {q: v[i] for q, v in terms.items()}
                                      for db, terms in pool["terms"].items()},
                               q1_dist_A=pool["q1_dist_A"][i])
    return rows, residual


def _top(rows, tag):
    [row] = [r for r in rows.values() if r["row"][f"is_{tag}"]]
    return row


def assemble(manifest_path, records):
    parents = defaultdict(dict)
    for row in _read(manifest_path):
        parents[row["code"]][row["runtime"]] = row
    summary = {}
    for code, tasks in parents.items():
        loaded = {arm: _read(records / f"{row['id']}.json") for arm, row in tasks.items()}
        native = {}
        issues = {}
        validations = {}
        for arm, task in tasks.items():
            native[arm] = {}
            for pair in task["paths"][arm]:
                rec = _read(pair["record"])
                seed = rec["seed"]
                rows, residual = _bundle(loaded[arm], arm, seed)
                if len(rows) != rec["n_rows"] or set(residual) != {"top1", "floor"} or max(map(abs, residual.values())) > 5e-4:
                    issues[f"{arm}_{seed}"] = dict(reason=loaded[arm].get("reason", "native replay validation failed"),
                                                  rows=len(rows), expected=rec["n_rows"], residual=residual)
                    continue
                native[arm][seed] = rows
                validations[f"{arm}_{seed}"] = max(map(abs, residual.values()))

        comparisons = {}
        for arm in ("D", "DQ6"):
            out = {}
            for seed in (11, 12, 13):
                if seed not in native["C"] or seed not in native[arm]:
                    continue
                c, d = native["C"][seed], native[arm][seed]
                ct, dt = _top(c, "top1"), _top(d, "top1")
                cf, df = _top(c, "floor"), _top(d, "floor")
                energy = lambda row: row["row"]["mace_energy_kj"]
                out[seed] = entry = dict(top1_delta_kj=energy(dt) - energy(ct),
                                        floor_delta_kj=energy(df) - energy(cf),
                                        top1_Q1_dist_A={"C": ct["q1_dist_A"], arm: dt["q1_dist_A"]},
                                        within_pool_reranks={})
                # Native-score + fixed-coordinate statistical delta preserves
                # that row's saved DB-independent prior and closure penalty.
                for label, rows, db in (("C", c, "C"), (arm, d, "D")):
                    swap = "D" if db == "C" else "C"
                    best = max(rows.values(), key=lambda r: r["row"]["score"] + r["stat"][swap] - r["stat"][db])
                    entry["within_pool_reranks"][label] = dict(
                        row_index=best["row"]["row_index"],
                        energy_change_kj=energy(best) - energy(_top(rows, "top1")))

                # For a shared C runtime, require the native contexts to have
                # identical atom/site/assignment identities for every variant.
                variants = {r["variant"] for r in [*c.values(), *d.values()]}
                compatible = all(
                    v in loaded["C"]["variants"] and v in loaded[arm]["variants"]
                    and all(loaded["C"]["variants"][v][k] == loaded[arm]["variants"][v][k]
                            for k in ("signature", "ez_fixes"))
                    for v in variants
                )
                entry["common_runtime_compatible"] = compatible
                if not compatible:
                    continue
                d_on_c, _ = _bundle(loaded["C"], arm, seed)
                if d_on_c.keys() != d.keys():
                    entry["common_runtime_compatible"] = False
                    continue
                # Two independent contrasts: statistical margin (mechanism)
                # and delivered-objective margin (retaining saved FK priors).
                entry["fixed_top_stat_margin_D_minus_C"] = {
                    db: _top(d_on_c, "top1")["stat"][db] - ct["stat"][db] for db in ("C", "D")
                }
                entry["term_margin_shift_Ddb_minus_Cdb"] = {
                    q: (_top(d_on_c, "top1")["terms"]["D"][q] - ct["terms"]["D"][q])
                    - (_top(d_on_c, "top1")["terms"]["C"][q] - ct["terms"]["C"][q])
                    for q in ct["terms"]["C"]
                }
                candidates = []
                for label, by_idx, origin in (("C", c, c), (arm, d_on_c, d)):
                    db = "C" if label == "C" else "D"
                    for i, r in by_idx.items():
                        prior = origin[i]["row"]["score"] - origin[i]["stat"][db]
                        candidates.append((label, i, r, prior))
                for db in ("C", "D"):
                    ranked = max(candidates, key=lambda x: x[2]["stat"][db] + x[3])
                    label, i, winner, _ = ranked
                    entry[f"common_runtime_{db}_best"] = dict(pool=label, row=i,
                                                               energy_vs_C_top_kj=energy(winner) - energy(ct))
                    a, b = _top(d_on_c, "top1"), ct
                    entry[f"common_runtime_{db}_top_margin"] = (
                        a["stat"][db] + dt["row"]["score"] - dt["stat"]["D"]
                        - b["stat"][db] - ct["row"]["score"] + ct["stat"]["C"]
                    )
            comparisons[arm] = out
        summary[code] = dict(validation_max_residual=validations, issues=issues, comparisons=comparisons)
    return summary
