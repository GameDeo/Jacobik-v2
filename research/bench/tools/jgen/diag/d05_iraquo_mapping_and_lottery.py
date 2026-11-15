"""Two CPU diagnostics on stored pools.

(a) IRAQUO stray-H test. The S2.1 report found one hydrogen per complex 28-30 A from its carbon in
    reference-mapped coordinates. Decide between
      MAPPING   : the generated structure still has a hydrogen bonded to that carbon (1.0-1.2 A) at a
                  different index, so the reference-to-generated H assignment is wrong, or
      PLACEMENT : no generated hydrogen sits within 1.25 A of that carbon, so the H itself is misplaced.
    Uses the raw generated X (not reference-mapped) and gen_to_ref_idx from the pool npz.
(b) Seed lottery. For the 3-seed parents: floor of each seed, 3-seed union floor (min over seeds),
    and parents whose floor gains more than 10 kJ/mol from the union. Matches scratchpad lottery.py.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/jgen/diag/d05_iraquo_mapping_and_lottery.py \
        --pools <.../pools> --manifest <.../manifest.json> --repair <.../repair_s2_1/repair_manifest.json>
"""
import argparse
import collections
import json
import os

import numpy as np

COV = {"H": .31, "C": .76, "N": .71, "O": .66, "S": 1.05, "P": 1.07, "F": .57, "Cl": 1.02, "Br": 1.20, "I": 1.39}


def iraquo(pools):
    out = {}
    for seed in (11, 12, 13):
        base = os.path.join(pools, "IRAQUO_%d_control" % seed)
        z = np.load(base + ".npz", allow_pickle=True)
        if "error" in z.files:
            out[seed] = "error pool"
            continue
        sym = [str(s) for s in z["ref_symbols"]]
        X = np.asarray(z["X"], float)
        g2r = np.asarray(z["gen_to_ref_idx"], int)
        Xr = np.asarray(z["ref_X"], float)
        # row-level: find H atoms whose mapped position is far from their carbon, check generated neighbourhood
        rows = []
        for r in range(min(X.shape[0], 8)):
            Y = X[r][g2r]  # reference order
            for h in range(len(sym)):
                if sym[h] != "H":
                    continue
                # nearest heavy atom in reference geometry
                dref = np.linalg.norm(Xr - Xr[h], axis=1)
                dref[h] = 1e9
                c = int(np.argmin(dref))
                if sym[c] == "H" or dref[c] > 1.3:
                    continue
                d_map = np.linalg.norm(Y[h] - Y[c])
                if d_map < 2.0:
                    continue
                # raw generated frame: reference atom j is generated index g2r[j] (Y = X[r][g2r] is reference order)
                Xg = X[r]
                gen_H = [int(g2r[j]) for j in range(len(sym)) if sym[j] == "H"]
                gc = int(g2r[c])
                dmin = min(np.linalg.norm(Xg[k] - Xg[gc]) for k in gen_H) if gen_H else 9e9
                # mapped frame (reference order): does the carbon have a hydrogen at bonding distance?
                hs = [j for j in range(len(sym)) if sym[j] == "H"]
                dmin_map = min(np.linalg.norm(Y[j] - Y[c]) for j in hs) if hs else 9e9
                verdict = "MAPPING" if dmin < 1.25 else "PLACEMENT"
                rows.append((r, h, c, round(float(d_map), 2), round(float(dmin), 2), round(float(dmin_map), 2), verdict))
        out[seed] = rows[:6] if rows else "no stray-H rows in first 8 conformers"
    return out


def lottery(pools, manifest, repair):
    m = json.load(open(manifest))
    rm = json.load(open(repair))
    # Same rule as the auditor's lottery.py: parents with any R1-repaired overlay are excluded (not repaired),
    # and floors are the stored (unrepaired) MACE floors of the remaining parents.
    by = collections.defaultdict(dict)
    r1 = set()
    for key, rec in m["tasks"].items():
        if rec.get("status") != "success" or rec.get("split") not in ("calibration", "motion_validation"):
            continue
        ov = rm["pools"].get(key)
        if ov and "floor_dE_kj" in np.load(os.path.join(os.path.dirname(manifest), ov["overlay_path"])).files:
            r1.add(rec["code"])
            continue
        f = rec.get("floor_dE_kj")
        if f is None:
            continue
        by[rec["code"]][rec["seed"]] = f
    for c in r1:
        by.pop(c, None)
    full = {c: v for c, v in by.items() if len(v) == 3}
    single = np.array([np.mean(list(v.values())) for v in full.values()])
    union = np.array([min(v.values()) for v in full.values()])
    gain = single - union
    return dict(parents=len(full), mean_single_seed_floor=round(float(single.mean()), 1),
                mean_union3_floor=round(float(union.mean()), 1),
                median_single=round(float(np.median(single)), 1), median_union=round(float(np.median(union)), 1),
                parents_gain_gt_10=int((gain > 10).sum()), parents_gain_gt_50=int((gain > 50).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repair", required=True)
    a = ap.parse_args()
    # IRAQUO mapping is reported by d11_iraquo_table.py; this script reports the seed lottery only.
    print(json.dumps({"lottery": lottery(a.pools, a.manifest, a.repair)}, indent=1))


if __name__ == "__main__":
    main()
