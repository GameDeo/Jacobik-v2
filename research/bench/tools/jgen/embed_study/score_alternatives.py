"""Apply the frozen F-2 paired winner rule to run_alternatives output."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

FAILURE_PENALTY = 10.0


def _parent_bootstrap_deltas(pairs, n=2000, seed=20261030):
    by_parent = defaultdict(list)
    for row in pairs:
        by_parent[row["code"]].append(row)
    parents = sorted(by_parent)
    rng = random.Random(seed)
    diffs=[]
    for _ in range(n):
        sample=[rng.choice(parents) for _ in parents]
        rows=[r for p in sample for r in by_parent[p]]
        diffs.append(float(np.median([r["arm_error"]-r["production_error"] for r in rows])))
    return [float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))]


def compare(arm, records, labels, bootstraps):
    indexed={(r["code"],r["ligand_idx"],r["arm"]):r for r in records}
    keys=sorted({(r["code"],r["ligand_idx"]) for r in records
                 if r["arm"]=="production" and r["label"] in labels})
    pairs=[]
    for code,ix in keys:
        p=indexed.get((code,ix,"production")); a=indexed.get((code,ix,arm))
        if not p or not a: continue
        pe=FAILURE_PENALTY if p["status"]=="failed" or p["normalized_error"] is None else float(p["normalized_error"])
        ae=FAILURE_PENALTY if a["status"]=="failed" or a["normalized_error"] is None else float(a["normalized_error"])
        pairs.append({"code":code,"ligand_idx":ix,"production_error":pe,"arm_error":ae,
                      "production_seconds":float(p["seconds"]),"arm_seconds":float(a["seconds"]),
                      "production_failed":p["status"]=="failed","arm_failed":a["status"]=="failed"})
    delta=[r["arm_error"]-r["production_error"] for r in pairs]
    return {"n_pairs":len(pairs),"production_median_error":float(np.median([r["production_error"] for r in pairs])) if pairs else None,
            "arm_median_error":float(np.median([r["arm_error"] for r in pairs])) if pairs else None,
            "median_paired_delta":float(np.median(delta)) if delta else None,
            "paired_parent_bootstrap_95_ci":_parent_bootstrap_deltas(pairs,bootstraps) if pairs else None,
            "production_failure_rate":sum(r["production_failed"] for r in pairs)/len(pairs) if pairs else None,
            "arm_failure_rate":sum(r["arm_failed"] for r in pairs)/len(pairs) if pairs else None,
            "production_median_seconds":float(np.median([r["production_seconds"] for r in pairs])) if pairs else None,
            "arm_median_seconds":float(np.median([r["arm_seconds"] for r in pairs])) if pairs else None,
            "pairs":pairs}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--bootstraps",type=int,default=2000)
    a=ap.parse_args(); raw=json.load(open(a.input)); records=raw["records"]
    methods=sorted({r["arm"] for r in records}-{"production"})
    results={}
    for arm in methods:
        fail=compare(arm,records,{"named_failure","uncorrected_nitro","corrected_label_worst20"},a.bootstraps)
        good=compare(arm,records,{"good_control"},a.bootstraps)
        all_eval=compare(arm,records,{"named_failure","uncorrected_nitro","corrected_label_worst20","good_control"},a.bootstraps)
        t_ratio=(all_eval["arm_median_seconds"]/all_eval["production_median_seconds"]
                 if all_eval["arm_median_seconds"] is not None and all_eval["production_median_seconds"] else None)
        ci_f=fail["paired_parent_bootstrap_95_ci"]
        ci_g=good["paired_parent_bootstrap_95_ci"]
        wins=bool(fail["n_pairs"] and good["n_pairs"] and fail["median_paired_delta"]<0 and ci_f[1]<0
                  and ci_g[0]<=0<=ci_g[1] and t_ratio is not None and t_ratio<=2)
        results[arm]={"failure_set":{k:v for k,v in fail.items() if k!="pairs"},
                      "good_control":{k:v for k,v in good.items() if k!="pairs"},
                      "combined_timing":{"production_median_seconds":all_eval["production_median_seconds"],
                          "arm_median_seconds":all_eval["arm_median_seconds"]},
                      "combined_median_time_ratio":t_ratio,"wins":wins}
    out={"source":str(a.input),"failure_penalty_normalized_units":FAILURE_PENALTY,
         "winner_rule":"lower failure-set median with paired parent-bootstrap 95% CI < 0; good-control paired 95% CI includes 0; combined failure-set and good-control median time <=2x production",
         "arms":results}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:{"wins":v["wins"],"failure_delta":v["failure_set"]["median_paired_delta"],
        "failure_ci":v["failure_set"]["paired_parent_bootstrap_95_ci"],"control_ci":v["good_control"]["paired_parent_bootstrap_95_ci"],
        "time_ratio":v["combined_median_time_ratio"]} for k,v in results.items()}))


if __name__=="__main__": main()
