"""Headline scoreboard and tail-parent diagnostics for the Step 2 control pools.

CPU only. Reads stored pool summaries (JSON) and repaired overlays; never touches
the GPU or QM property fields. Run from any directory:

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d01_pool_scoreboard.py \
        --data /mnt/e/Dev/Projects/jacobik_2026-10-02_full/data/refit/frame_v1 \
        --out-json <path>

Definitions (roadmapv2 section 5.2): top1_dE = native top-1 energy minus reference,
floor_dE = best valid pool energy minus reference, regret = top1_dE - floor_dE.
Repaired values from repair_s2_1 overlays take precedence over the original JSON.
"""
import argparse
import collections
import json
import os

import numpy as np


def load_rows(data):
    m = json.load(open(os.path.join(data, "manifest.json")))
    rm = json.load(open(os.path.join(data, "repair_s2_1", "repair_manifest.json")))
    rows = []
    for key, rec in m["tasks"].items():
        if rec.get("status") != "success":
            continue
        if rec.get("split") not in ("calibration", "motion_validation"):
            continue
        ov = rm["pools"].get(key)
        f = t = None
        if ov:
            z = np.load(os.path.join(data, ov["overlay_path"]))
            if "floor_dE_kj" in z.files:
                f = float(z["floor_dE_kj"])
                t = float(z["top1_dE_kj"])
        if f is None:
            f = rec.get("floor_dE_kj")
            t = rec.get("top1_dE_kj")
        if f is None or t is None:
            continue
        rows.append(dict(key=key, code=rec["code"], seed=rec["seed"], top1=t, floor=f,
                         stratum=rec.get("stratum"), split=rec.get("split")))
    return m, rows


def summ(x):
    x = np.asarray(x, float)
    return dict(mean=round(float(x.mean()), 1), median=round(float(np.median(x)), 1),
                p90=round(float(np.percentile(x, 90)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    m, rows = load_rows(a.data)
    T = np.array([r["top1"] for r in rows])
    F = np.array([r["floor"] for r in rows])
    G = T - F
    out = dict(pools=len(rows), top1=summ(T), floor=summ(F), regret=summ(G),
               top1_within_4p184_of_floor=round(float(np.mean(G <= 4.184)), 3))
    for th in (100, 200, 500, 1000):
        out["pools_top1_gt_%d" % th] = int((T > th).sum())
    out["mean_top1_capped_500"] = round(float(np.minimum(T, 500).mean()), 1)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["code"]].append(r)
    full = {c: v for c, v in by.items() if len(v) == 3}
    rng = {c: max(x["top1"] for x in v) - min(x["top1"] for x in v) for c, v in full.items()}
    out["parents_3_seeds"] = len(full)
    out["seed_range_gt_50"] = sum(1 for v in rng.values() if v > 50)
    out["seed_range_gt_20"] = sum(1 for v in rng.values() if v > 20)
    out["seed_range_median"] = round(float(np.median(list(rng.values()))), 1)
    out["stored_conformer_rows_total"] = int(sum(
        int(m["tasks"][r["key"]].get("n_confs") or 0) for r in rows))

    # Tail parents: coordination integrity of the native top-1 and of the floor pose.
    tails = sorted({r["code"] for r in rows if r["top1"] > 500})
    tail_rows = []
    for c in tails:
        for r in sorted(by[c], key=lambda x: x["seed"]):
            rec = m["tasks"][r["key"]]
            tail_rows.append(dict(code=c, seed=r["seed"], top1=round(r["top1"], 1),
                                  floor=round(r["floor"], 1), stratum=r["stratum"],
                                  n_confs=rec.get("n_confs"),
                                  top1_coord_intact=rec.get("top1_coord_intact"),
                                  floor_coord_intact=rec.get("floor_coord_intact"),
                                  pool_intact_frac=round(rec.get("pool_coord_intact_fraction") or 0, 3)))
    out["tail_parents"] = tails
    out["tail_rows"] = tail_rows

    # Does a non-intact native top-1 explain the tail across ALL pools?
    ti = np.array([bool(rec.get("top1_coord_intact")) for rec in
                   (m["tasks"][r["key"]] for r in rows)])
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poolio import Repairs, load_effective
    _rep = Repairs(a.data)
    _eff = {}
    for r in rows:
        p_ = load_effective(os.path.join(a.data, "pools"), _rep, r["code"], r["seed"])
        _eff[r["key"]] = p_
    def _ci(key):
        e = _eff.get(key)
        if e is None or e["coord_intact"] is None:
            return None
        return e["coord_intact"]
    # effective top-1 intact (argmax of score on effective pools) and floor intact
    ti_eff = []
    for r in rows:
        e = _eff[r["key"]]
        if e is None or e["coord_intact"] is None:
            continue
        ti_eff.append((r["top1"] > 500, bool(e["coord_intact"][int(np.argmax(e["score"]))]), r["top1"] <= 100))
    out["effective_top1_intact_share_all"] = round(float(np.mean([x[1] for x in ti_eff])), 3)
    out["effective_top1_intact_share_gt500"] = round(float(np.mean([x[1] for x in ti_eff if x[0]])), 3)
    out["effective_top1_intact_share_le100"] = round(float(np.mean([x[1] for x in ti_eff if x[2]])), 3)
    out["top1_coord_intact_share_all"] = round(float(ti.mean()), 3)
    out["top1_coord_intact_share_tail_gt500"] = round(
        float(np.mean([bool(m["tasks"][r["key"]].get("top1_coord_intact"))
                       for r in rows if r["top1"] > 500])), 3)
    out["top1_coord_intact_share_le100"] = round(
        float(np.mean([bool(m["tasks"][r["key"]].get("top1_coord_intact"))
                       for r in rows if r["top1"] <= 100])), 3)
    out["floor_coord_intact_share_all"] = round(float(np.mean(
        [bool(m["tasks"][r["key"]].get("floor_coord_intact")) for r in rows])), 3)

    # Stage timing means (eval split vs fit split).
    for split in ("calibration", "motion_validation", "surface_fit"):
        recs = [m["tasks"][r["key"]] for r in rows if r["split"] == split]
        if not recs:
            continue
        for k in ("wall_s", "stage_solver_s", "stage_dedup_s", "stage_embed_s",
                  "stage_seed_rows_s", "stage_mace_s", "stage_other_s"):
            vals = [x.get(k) for x in recs if x.get(k) is not None]
            if vals:
                out.setdefault("timing_" + split, {})[k] = round(float(np.mean(vals)), 2)

    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(txt)


if __name__ == "__main__":
    main()
