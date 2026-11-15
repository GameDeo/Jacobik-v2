"""Scoreboards split by input label, failures on consistent inputs, and the score's separation (CPU, read-only).

Inputs: the label JSON written by d14_input_consistency.py (--split pools) and the effective Step 2 pools
(poolio.py, S2.1 overlays applied). Nothing is written under data/.

Outputs
  1. Per label (consistent / ambiguous / inconsistent): parents, pools, top-1 and floor means and medians,
     regret, the share of pools with top-1 above 100 / 200 / 500 kJ/mol, and the median seed range.
  2. Failures on consistent inputs: pools with top-1 above 200 kJ/mol, parents with seed range above 50 kJ/mol,
     and the tail parents (top-1 above 500 in some seed), each with its label.
  3. Score separation (D): AUC of the native top-1 score and of the pool's maximum score for separating
     inconsistent from consistent parents, and the same AUC for the energy failure (top-1 above 200 kJ/mol),
     so that the score's relation to the energy can be read next to its relation to the label. The AUC is the
     probability that a randomly chosen consistent pool scores higher than a randomly chosen inconsistent pool
     (ties count one half). Pool-level, seeds pooled.

    python tools/jgen/diag/d17_label_scoreboard.py --labels <d14 json> --out-json <path>
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402
from poolio import Repairs, control_tasks, load_effective  # noqa: E402


def auc(pos, neg):
    """P(score(pos) > score(neg)) with ties counted one half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return round(float((gt + 0.5 * eq) / (len(pos) * len(neg))), 3)


def summ(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    return dict(n=int(x.size), mean=round(float(x.mean()), 1), median=round(float(np.median(x)), 1),
                p90=round(float(np.percentile(x, 90)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", default=str(ctxio.SCALED))
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    lab = {p["code"]: p["label"] for p in json.load(open(a.labels))["parents"]}
    rep = Repairs(a.root)
    pools_dir = os.path.join(a.root, "pools")
    rows = []
    for code, seed, rec in control_tasks(os.path.join(a.root, "manifest.json")):
        if code not in lab:
            continue
        p = load_effective(pools_dir, rep, code, seed)
        if p is None:
            continue
        de = p["dE"]
        score = p["score"]
        top = int(np.argmax(score))
        ci = p["coord_intact"]
        rows.append(dict(code=code, seed=int(seed), label=lab[code], top1=float(de[top]), floor=float(de.min()),
                         top1_score=float(score[top]), max_score=float(score.max()), n_rows=int(len(de)),
                         n_classes=int(len(set(np.asarray(p["asg"]).tolist()))),
                         top1_intact=None if ci is None else bool(ci[top]),
                         pool_intact_share=None if ci is None else round(float(np.mean(ci)), 3),
                         repaired=bool(p["repairs"])))
    out = dict(pools=len(rows), parents=len({r["code"] for r in rows}))
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    out["per_label"] = {}
    for L in ("consistent", "ambiguous", "inconsistent"):
        rs = by_label.get(L, [])
        T = np.array([r["top1"] for r in rs])
        F = np.array([r["floor"] for r in rs])
        codes = sorted({r["code"] for r in rs})
        rng = []
        for c in codes:
            v = [r["top1"] for r in rs if r["code"] == c]
            if len(v) == 3:
                rng.append(max(v) - min(v))
        if len(T) == 0:
            out["per_label"][L] = dict(parents=0, pools=0)
            continue
        out["per_label"][L] = dict(
            parents=len(codes), pools=int(len(T)), top1=summ(T), floor=summ(F), regret=summ(T - F),
            share_top1_gt_100=round(float(np.mean(T > 100)), 3), share_top1_gt_200=round(float(np.mean(T > 200)), 3),
            share_top1_gt_500=round(float(np.mean(T > 500)), 3), parents_3_seeds=len(rng),
            seed_range_median=round(float(np.median(rng)), 1) if rng else None,
            seed_range_gt_50=int(sum(1 for x in rng if x > 50)))

    # 2. failures on consistent inputs (and where the other labels fall).
    cons = [r for r in rows if r["label"] == "consistent"]
    out["consistent_pools_top1_gt_200"] = [dict(code=r["code"], seed=r["seed"], top1=round(r["top1"], 1),
                                                floor=round(r["floor"], 1), top1_intact=r["top1_intact"],
                                                pool_intact_share=r["pool_intact_share"], n_rows=r["n_rows"])
                                           for r in cons if r["top1"] > 200]
    cats = collections.Counter()
    for d in out["consistent_pools_top1_gt_200"]:
        if d["top1_intact"]:
            c = "top1_intact"
        elif (d["pool_intact_share"] or 0) > 0:
            c = "top1_nonintact_intact_row_exists"
        else:
            c = "top1_nonintact_no_intact_row"
        cats[c + ("_few_rows" if d["n_rows"] <= 4 else "")] += 1
    out["consistent_top1_gt_200_categories"] = dict(cats)
    by_code = collections.defaultdict(list)
    for r in rows:
        by_code[r["code"]].append(r)
    rng_all = {c: max(x["top1"] for x in v) - min(x["top1"] for x in v) for c, v in by_code.items() if len(v) == 3}
    out["seed_range_gt_50"] = sorted([dict(code=c, label=lab[c], range=round(v, 1)) for c, v in rng_all.items() if v > 50],
                                     key=lambda d: -d["range"])
    tails = sorted({r["code"] for r in rows if r["top1"] > 500})
    out["tail_parents"] = [dict(code=c, label=lab[c]) for c in tails]
    out["seed_range_gt_50_by_label"] = dict(collections.Counter(d["label"] for d in out["seed_range_gt_50"]))

    # 3. score separation.
    bad = [r for r in rows if r["label"] == "inconsistent"]
    good = [r for r in rows if r["label"] == "consistent"]
    amb = [r for r in rows if r["label"] == "ambiguous"]
    fail = lambda r: r["top1"] > 200
    out["auc"] = dict(
        top1_score_inconsistent_vs_consistent=auc([r["top1_score"] for r in good], [r["top1_score"] for r in bad]),
        max_score_inconsistent_vs_consistent=auc([r["max_score"] for r in good], [r["max_score"] for r in bad]),
        top1_score_ambiguous_vs_consistent=auc([r["top1_score"] for r in good], [r["top1_score"] for r in amb]),
        energy_fail_auc_by_top1_score=auc([-r["top1_score"] for r in rows if fail(r)],
                                          [-r["top1_score"] for r in rows if not fail(r)]),
        energy_fail_auc_by_max_score=auc([-r["max_score"] for r in rows if fail(r)],
                                         [-r["max_score"] for r in rows if not fail(r)]),
        energy_fail_auc_by_label_inconsistent=auc([1 if r["label"] == "inconsistent" else 0 for r in rows if fail(r)],
                                                  [1 if r["label"] == "inconsistent" else 0 for r in rows if not fail(r)]),
        n_bad=len(bad), n_good=len(good), n_amb=len(amb))
    print(ctxio.to_json(out))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(out))


if __name__ == "__main__":
    main()
