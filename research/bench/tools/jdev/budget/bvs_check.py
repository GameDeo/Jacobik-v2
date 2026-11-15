"""Plan 3 S2, offline check 1-2 (2026-10-17): is there bond-length coupling at the metal that independent Q1 misses?

Brown's bond-valence sum: s = exp((R0 - R) / B) per metal-donor bond, V = sum of s over the metal's bonds ~ the metal's
oxidation state. R0 per (metal, donor element) is fitted on half of the training complexes (sigma-bonded only: complexes
with a haptic donor are left out of this check, their handling is check 3) by least squares on V / OS - 1; the spread of
V / OS is then measured on the other half, against the same spread when every bond length is replaced by the length of a
random bond of the same key (metal, oxidation state, donor subclass) from another complex: the distribution independent
per-bond statistics imply. If real complexes are much narrower than the shuffled ones, the bonds are coupled and a
reference-ratio term has something to add; if not, S2 stops here.

    python tools/jdev/budget/bvs_check.py [--min-bonds 30] [--out data/refit/bvs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
OBS = REPO / "data/refit/observations"
import sys
sys.path.insert(0, str(REPO / "src"))
from research.archive.code.interactions.bvs import B, OS_UNKNOWN, fit_r0, load as load_observations, robust_sd, shuffled_lengths


def load():
    return load_observations(OBS, ("X1", "X2", "X3", "LNf"))


def vsum(d, r0, R=None):
    R = d.value.to_numpy() if R is None else R
    s = np.exp((d.pair.map(r0).to_numpy() - R) / B)
    return pd.Series(s).groupby(d.csd_code.to_numpy()).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-bonds", type=int, default=30)
    p.add_argument("--out", default=str(REPO / "data/refit/bvs"))
    a = p.parse_args()
    rng = np.random.default_rng(0)
    d, hapt = load()
    n_all = d.csd_code.nunique()
    d = d[~hapt & (d.os != OS_UNKNOWN) & (d.os > 0)].copy()
    cnt = d.pair.value_counts()
    rare = set(cnt[cnt < a.min_bonds].index)
    bad = set(d.loc[d.pair.isin(rare), "csd_code"])
    d = d[~d.csd_code.isin(bad)].copy()
    cx = d.csd_code.unique()
    rng.shuffle(cx)
    fit_set, test_set = set(cx[: len(cx) // 2]), set(cx[len(cx) // 2:])
    pairs = sorted(d.pair.unique())
    print(f"complexes: {n_all} in the observations; {d.csd_code.nunique()} sigma-only with a known OS > 0 and only "
          f"pairs with >= {a.min_bonds} bonds ({len(pairs)} pairs); fit {len(fit_set)} / test {len(test_set)}")
    r0 = fit_r0(d[d.csd_code.isin(fit_set)], pairs)
    t = d[d.csd_code.isin(test_set)].copy()
    v = vsum(t, r0)
    os_c = t.groupby("csd_code").os.first()
    metal_c = t.groupby("csd_code").metal.first()
    real = (v / os_c).reindex(os_c.index)
    shuf = (vsum(t, r0, shuffled_lengths(t, rng)) / os_c).reindex(os_c.index)
    print(f"\nV / OS on the test half (1 = Brown's rule): real median {real.median():.3f}, robust sd {robust_sd(real):.3f}; "
          f"independent bonds (shuffled within key) robust sd {robust_sd(shuf):.3f}  -> ratio {robust_sd(shuf) / robust_sd(real):.2f}")
    print("per metal (>= 300 test complexes): robust sd of V / OS, real vs independent")
    rows = []
    for m, g in real.groupby(metal_c):
        if len(g) >= 300:
            rows.append((m, len(g), robust_sd(g), robust_sd(shuf[g.index])))
    for m, n, s1, s2 in sorted(rows, key=lambda r: -r[1]):
        print(f"  {m:3s} n {n:5d}  real {s1:.3f}  independent {s2:.3f}  ratio {s2 / s1:4.2f}")
    # metal only (no oxidation state): V per metal, real vs independent
    vm = v.reindex(os_c.index)
    vs = vsum(t, r0, shuffled_lengths(t, rng)).reindex(os_c.index)
    rel = lambda x, g: robust_sd(x[g.index]) / np.median(x[g.index])
    r_real = np.median([rel(vm, g) for _, g in vm.groupby(metal_c) if len(g) >= 300])
    r_ind = np.median([rel(vs, g) for _, g in vs.groupby(metal_c) if len(g) >= 300])
    print(f"metal-only level (OS unknown): median relative robust sd of V real {r_real:.3f}, independent {r_ind:.3f}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "r0.json").write_text(json.dumps({k: round(float(x), 4) for k, x in r0.items()}, indent=1))
    print(f"\nR0 written to {out / 'r0.json'}; examples: " +
          ", ".join(f"{k} {r0[k]:.3f}" for k in ("Fe_N", "Cu_N", "Cu_O", "Pd_P", "Ta_N", "Zn_O", "Co_O", "Ni_N") if k in r0))


if __name__ == "__main__":
    main()
