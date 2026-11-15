"""Run-to-run repeatability of the control arm: Step 2 (effective) versus S2b regenerated units (CPU only).

S2b (registry record 22, code commit f71f196) regenerated the 240 calibration and motion_validation units under
the same configuration, with per-pose state captured. Both sets are compared per (parent, seed) unit:
  top1_dE    = dE at argmax(score)                (the native top-1 of each run)
  floor_dE   = min dE over the rows of the run
  d_top1     = top1_dE(S2b) - top1_dE(Step 2 effective)     (Step 2 effective = raw + S2.1 overlay, poolio.py)
  d_floor    = floor_dE(S2b) - floor_dE(Step 2 effective)
Per parent (seeds 11-13 kept together): d_p^(e) = mean over the seeds present of d_top1 and
d_p^(f) = mean over the seeds present of d_floor (roadmapv2 section 5.2 sign convention: S2b minus Step 2).
Also: units present in one set and not the other (the success sets differ), and the MACE reference energy
of each unit (must match between runs for the comparison to be meaningful; reported as max abs difference).

Outputs: the distribution of |d_top1| and |d_floor| over units, the per-parent table, the worst units.
This is the noise band for the simple bench mode (research/archive/docs/roadmap/roadmap_generation_v1.md section 1.4.3).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d12_s2b_repeatability.py \
        --root <.../frame_v1> --out-json <path>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poolio import Repairs, load_effective  # noqa: E402


def dE_summary(npz_path, mace_path):
    z = np.load(npz_path, allow_pickle=True)
    if "error" in z.files:
        return None
    mj = json.load(open(mace_path))
    de = np.asarray(mj["energies_kj"], float) - float(mj["ref_energy_kj"])
    s = np.asarray(z["score"], float)
    return dict(top1=float(de[int(np.argmax(s))]), floor=float(de.min()), ref=float(mj["ref_energy_kj"]), n=len(de))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    step2 = os.path.join(a.root, "pools")
    s2b = os.path.join(a.root, "s2b_eval_state_v1", "pools")
    rep = Repairs(a.root)
    m = json.load(open(os.path.join(a.root, "manifest.json")))
    units = sorted({(r["code"], int(r["seed"])) for k, r in m["tasks"].items()
                    if r.get("status") == "success" and r.get("split") in ("calibration", "motion_validation")})
    rows, only_step2, only_s2b, ref_diff = [], [], [], []
    for code, seed in units:
        p2 = os.path.join(s2b, "%s_%d_control.npz" % (code, seed))
        e = load_effective(step2, rep, code, seed)
        if e is None:
            continue
        dd = dict(code=code, seed=seed, n_step2=len(e["dE"]))
        dd["top1_step2"] = float(e["dE"][int(np.argmax(e["score"]))])
        dd["floor_step2"] = float(e["dE"].min())
        if not os.path.exists(p2):
            only_step2.append((code, seed))
            continue
        r2 = dE_summary(p2, os.path.join(s2b, "%s_%d_control.mace64.json" % (code, seed)))
        if r2 is None:
            only_step2.append((code, seed))
            continue
        dd.update(top1_s2b=r2["top1"], floor_s2b=r2["floor"], n_s2b=r2["n"])
        dd["d_top1"] = r2["top1"] - dd["top1_step2"]
        dd["d_floor"] = r2["floor"] - dd["floor_step2"]
        dd["ref_step2"] = float(json.load(open(os.path.join(step2, "%s_%d_control.mace64.json" % (code, seed))))["ref_energy_kj"])
        ref_diff.append(abs(dd["ref_step2"] - r2["ref"]))
        rows.append(dd)
    import glob
    unit_set = set(units)
    for fp in sorted(glob.glob(os.path.join(s2b, "*_control.npz"))):
        code, seed = os.path.basename(fp).split("_")[0], int(os.path.basename(fp).split("_")[1])
        if (code, seed) in unit_set:
            continue
        if "error" in np.load(fp, allow_pickle=True).files:
            continue
        only_s2b.append((code, seed))  # regenerated S2b pool with no Step 2 success unit
    dt = np.array([abs(r["d_top1"]) for r in rows])
    df = np.array([abs(r["d_floor"]) for r in rows])
    out = dict(units_compared=len(rows), only_step2=len(only_step2), only_s2b_pools=len(only_s2b),
               ref_energy_max_abs_diff=round(float(max(ref_diff)), 6) if ref_diff else None,
               abs_d_top1_median=round(float(np.median(dt)), 4), abs_d_top1_p95=round(float(np.percentile(dt, 95)), 3),
               abs_d_top1_gt_1=int((dt > 1).sum()), abs_d_top1_max=round(float(dt.max()), 2),
               abs_d_floor_median=round(float(np.median(df)), 4), abs_d_floor_p95=round(float(np.percentile(df, 95)), 3),
               abs_d_floor_gt_1=int((df > 1).sum()), abs_d_floor_max=round(float(df.max()), 2),
               abs_d_top1_gt_5=int((dt > 5).sum()), abs_d_top1_gt_10=int((dt > 10).sum()),
               abs_d_floor_gt_5=int((df > 5).sum()), abs_d_floor_gt_10=int((df > 10).sum()))
    by = {}
    for r in rows:
        by.setdefault(r["code"], []).append(r)
    parents = []
    for c, v in by.items():
        parents.append(dict(code=c, seeds=len(v), d_e=round(float(np.mean([x["d_top1"] for x in v])), 3),
                            d_f=round(float(np.mean([x["d_floor"] for x in v])), 3)))
    pe = np.array([abs(x["d_e"]) for x in parents])
    pf = np.array([abs(x["d_f"]) for x in parents])
    out["parents"] = len(parents)
    out["parent_abs_d_e_median"] = round(float(np.median(pe)), 4)
    out["parent_abs_d_e_p95"] = round(float(np.percentile(pe, 95)), 3)
    out["parent_abs_d_f_median"] = round(float(np.median(pf)), 4)
    out["parent_abs_d_f_p95"] = round(float(np.percentile(pf, 95)), 3)
    out["worst_units"] = sorted(
        [(r["code"], r["seed"], round(r["d_top1"], 2), round(r["d_floor"], 2)) for r in rows],
        key=lambda t: -abs(t[2]))[:6]
    out["only_step2_units"] = only_step2[:20]
    out["only_s2b_units"] = only_s2b[:20]
    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if a.out_json:
        open(a.out_json, "w").write(txt)


if __name__ == "__main__":
    main()
