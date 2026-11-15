"""Join the four F-4 screen arms into full-census transition tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMED = {"QEZVEY", "OHECUZ", "FIMCAH", "BOCRAN", "LOPCAW", "EBAYIQ"}
ARMS = ("production", "guard", "li_cap", "combined")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    data = {arm: json.loads((a.directory / f"screen_f4_{arm}.json").read_text()) for arm in ARMS}
    records = {arm: {(r["code"], int(r["ligand_idx"])): r for r in data[arm]["records"]} for arm in ARMS}
    keys = sorted(set().union(*(set(r) for r in records.values())))
    baseline = records["production"]
    labels = {key: baseline[key].get("bad") if baseline[key].get("status") == "ok" else None for key in keys}
    out = {"n_source_attempts": len(keys), "arms": {}, "transitions": [], "named": {},
           "trigger_parents": {}, "failed_refinements": {}}
    for arm in ARMS:
        rr = records[arm]
        if arm == "production":
            repairs = regressions = 0
            denominator = sum(v is False for v in labels.values())
        else:
            repairs = sum(labels[k] is True and rr.get(k, {}).get("status") == "ok" and rr[k].get("bad") is False
                          for k in labels)
            regressions = sum(labels[k] is False and (k not in rr or rr[k].get("status") != "ok" or rr[k].get("bad") is True)
                              for k in labels)
            denominator = sum(v is False for v in labels.values())
        secs = [float(r.get("seconds", 0.0)) for r in rr.values()]
        fails = [r for r in rr.values() if r.get("status") != "ok"]
        gfn_failures = [(r["code"], r["ligand_idx"], msg) for r in rr.values() for msg in r.get("gfn2_failures", [])]
        out["failed_refinements"][arm] = gfn_failures
        tr_parents = sorted({r["code"] for r in rr.values() for e in r.get("guard_events", []) if e.get("triggered")})
        out["trigger_parents"][arm] = tr_parents
        out["arms"][arm] = {"scoreable": sum(r.get("status") == "ok" for r in rr.values()),
                            "failed_labels_or_embeds": len(fails), "repairs": int(repairs),
                            "regressions": int(regressions), "baseline_good_denominator": int(denominator),
                            "regression_rate": float(regressions / denominator) if denominator else None,
                            "mean_seconds_per_ligand": float(np.mean(secs)) if secs else None,
                            "median_seconds_per_ligand": float(np.median(secs)) if secs else None,
                            "max_seconds_per_ligand": float(np.max(secs)) if secs else None,
                            "guard_trigger_events": sum(bool(e.get("triggered")) for r in rr.values()
                                                         for e in r.get("guard_events", [])),
                            "guard_fallback_counts": data[arm].get("fallbacks", {})}
    for key in keys:
        base_rec = baseline.get(key, {})
        row = {"code": key[0], "ligand_idx": key[1], "production": labels[key]}
        for arm in ARMS[1:]:
            r = records[arm].get(key)
            row[arm] = (r.get("bad") if r and r.get("status") == "ok" else
                        "failed" if r else "missing")
        out["transitions"].append(row)
        if key[0] in NAMED:
            out["named"].setdefault(key[0], []).append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"n_source_attempts": out["n_source_attempts"], "arms": out["arms"],
                      "named": out["named"], "trigger_parents": out["trigger_parents"],
                      "failed_refinement_counts": {k: len(v) for k, v in out["failed_refinements"].items()},
                      "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
