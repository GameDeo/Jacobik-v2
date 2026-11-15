"""Summarize the registered unrelaxed O2 trope gate against its paired control."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def median(xs):
    return statistics.median(xs) if xs else None


def parent_metrics(rec):
    ps = [v for v in rec.get("per_seed", {}).values() if v.get("status") == "ok"]
    if not ps:
        return None
    top = [float(p.get("dE_unrel", next(b["dE_unrel"] for b in p["buckets"] if b["is_top1"]))) for p in ps]
    floors = [min(map(float, p["all_dE_unrel"])) for p in ps if p.get("all_dE_unrel")]
    all_e = [float(x) for p in ps for x in p.get("all_dE_unrel", [])]
    scores = [float(p["top1_score"]) for p in ps]
    return {"top1_score": statistics.mean(scores), "top1_dE_unrel": statistics.mean(top),
            "floor_dE_unrel": min(floors) if floors else None,
            "worst_top1_dE_unrel": max(top), "n_rows": len(all_e),
            "obscene_count": sum(x > 200.0 for x in all_e), "all_dE_unrel": all_e}


def arm_metrics(root: Path, arm: str):
    path = root / f"{arm}_unrel"
    records = {p.stem: json.loads(p.read_text()) for p in path.glob("*.json")}
    parents = {c: parent_metrics(r) for c, r in records.items()}
    valid = {c: m for c, m in parents.items() if m is not None and records[c].get("status") == "ok"}
    all_e = [x for m in valid.values() for x in m["all_dE_unrel"]]
    return {"records": records, "parents": valid,
            "n_ok": len(valid), "n_failed": len(records) - len(valid),
            "median_top1_score": median([m["top1_score"] for m in valid.values()]),
            "median_top1_dE_unrel": median([m["top1_dE_unrel"] for m in valid.values()]),
            "obscene_rate": sum(x > 200.0 for x in all_e) / len(all_e) if all_e else None,
            "n_rows": len(all_e), "floor_mean": statistics.mean([m["floor_dE_unrel"] for m in valid.values() if m["floor_dE_unrel"] is not None]) if valid else None,
            "worst": max(((m["worst_top1_dE_unrel"], c) for c, m in valid.items()), default=(None, None)),
            "mean_wall_s": statistics.mean([r.get("wall_s", math.nan) for c, r in records.items() if c in valid]) if valid else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-parents", type=int, default=9)
    args = ap.parse_args()
    root = Path(args.root)
    c = arm_metrics(root, "control")
    result = {"control": {k: v for k, v in c.items() if k != "records"}, "arms": {}}
    for arm in ("adam_metric", "fire_metric"):
        a = arm_metrics(root, arm)
        codes = sorted(set(c["parents"]) & set(a["parents"]))
        ds = [a["parents"][x]["top1_score"] - c["parents"][x]["top1_score"] for x in codes]
        de = [a["parents"][x]["top1_dE_unrel"] - c["parents"][x]["top1_dE_unrel"] for x in codes]
        floor_d = [a["parents"][x]["floor_dE_unrel"] - c["parents"][x]["floor_dE_unrel"]
                   for x in codes if None not in (a["parents"][x]["floor_dE_unrel"], c["parents"][x]["floor_dE_unrel"])]
        worst_d = max(((a["parents"][x]["top1_dE_unrel"] - c["parents"][x]["top1_dE_unrel"], x) for x in codes), default=(None, None))
        wall_ratio = a["mean_wall_s"] / c["mean_wall_s"] if c["mean_wall_s"] and a["mean_wall_s"] else None
        guards = {"all_expected_parents_ok": len(codes) == args.expected_parents,
                  "no_parent_top1_worsening_over_25": worst_d[0] is not None and worst_d[0] <= 25.0,
                  "mean_floor_not_worse": bool(floor_d) and statistics.mean(floor_d) <= 0.0,
                  "obscene_rate_not_worse": a["obscene_rate"] is not None and c["obscene_rate"] is not None and a["obscene_rate"] <= c["obscene_rate"]}
        promising = (guards["all_expected_parents_ok"] and guards["no_parent_top1_worsening_over_25"]
                     and guards["mean_floor_not_worse"] and guards["obscene_rate_not_worse"]
                     and median(ds) is not None and median(ds) > 0.0)
        result["arms"][arm] = {k: v for k, v in a.items() if k != "records"}
        result["arms"][arm].update({"paired_parents": codes, "median_paired_score_change": median(ds),
                                    "median_paired_top1_dE_change": median(de),
                                    "worst_paired_top1_dE_change": worst_d,
                                    "mean_paired_floor_change": statistics.mean(floor_d) if floor_d else None,
                                    "wall_ratio_vs_control": wall_ratio, "guards": guards,
                                    "promising_screen": promising})
    candidates = [(v["median_paired_score_change"], k) for k, v in result["arms"].items() if v["promising_screen"]]
    result["selected_arm_for_nine_relaxations"] = max(candidates)[1] if candidates else None
    result["note"] = "Unrelaxed gate only. No tail/cal authorization is inferred by this summary."
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("arm            d_score_med   dE_top1_med   obscene   worst(top1 dE, parent)   wall×   gate")
    print(f"control        {'—':>11s} {'—':>13s} {c['obscene_rate']!s:>9s} {str(c['worst']):>25s} {'—':>7s} baseline")
    for arm, m in result["arms"].items():
        print(f"{arm:14s} {m['median_paired_score_change']!s:>11s} {m['median_paired_top1_dE_change']!s:>13s} "
              f"{m['obscene_rate']!s:>9s} {str(m['worst']):>25s} {m['wall_ratio_vs_control']!s:>7s} {m['promising_screen']}")
    print("selected:", result["selected_arm_for_nine_relaxations"])


if __name__ == "__main__":
    main()
