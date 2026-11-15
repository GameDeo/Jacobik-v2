"""Paired A4b comparison of two runs (control vs flag arm): per-parent differences of unit values, parent-clustered
bootstrap CIs, bucket transitions.  Pure post-processing of two `units.csv` files.

    python -m tools.jgen.a4b.pair --control A4R/a4b/units.csv --arm CHAIR/a4b/units.csv [--out pair.json]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

COLS = ("top1_dE", "offset_top1", "strain_top1", "ranking_loss")


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(path):
    return {(r["code"], int(r["seed"])): r for r in csv.DictReader(open(path)) if r.get("status") == "ok"}


def boot_ci(per_parent, n=5000, seed=0):
    v = np.asarray(per_parent, float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pair(control, arm, panels=("cal", "tail")):
    out = {}
    for panel in (*panels, "all"):
        keys = [k for k in control if k in arm and (panel == "all" or control[k]["panel"] == panel)]
        res = dict(n_units=len(keys), n_parents=len({k[0] for k in keys}))
        for col in COLS:
            by = defaultdict(list)
            for k in keys:
                a, b = _f(control[k].get(col)), _f(arm[k].get(col))
                if a is not None and b is not None:
                    by[k[0]].append(b - a)
            res[col] = boot_ci([np.mean(v) for v in by.values()]) if by else None
        for col in ("u_bucket", "r_bucket"):
            res[col + "_transitions"] = dict(Counter(f"{control[k][col]}->{arm[k][col]}" for k in keys))
        out[panel] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = pair(load(a.control), load(a.arm))
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
