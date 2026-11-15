"""
bvs.py -- tables of the bond-valence coupling term QV (plan 3 S2, 2026-10-17; profile field bvs, default off).

Brown's bond-valence sum V = sum over a complex's sigma metal-donor bonds of exp((R0 - R) / B), B = 0.37 A, is ~ the
metal's oxidation state. Q1 scores every bond independently, so the spread of V it implies is wider than real complexes
show where bonds are coupled. The term adds only that coupling, as a reference ratio (Hamelryck 2010):

  log p_obs(x) - log p_indep(x), both Gaussian in x  ->  -c / 2 (x - mu_obs)^2 + const,  c = max(0, 1/s_obs^2 - 1/s_indep^2)

x = V / OS on the oxidation-state levels, V / median V on the metal-only level (input without an OS). One term per
complex; it constrains only the sum. Scope: complexes whose sites are all sigma (haptic: open). The runtime is
scorer._bvs_runtime.

Ladders (most specific with enough data):
  R0  (metal, donor element) -> (period-group, donor element) -> (donor element); least squares on V / OS - 1;
  V   (metal, OS) -> (period-group, OS) -> (metal).
Spreads are robust (16-84 % half-width); the independent spread is the mean over N_SHUFFLE shuffles of every bond's
length within its Q1 key (metal, OS, donor subclass).

Status (CHANGELOG 2026-10-17): gate FAIL on the u1 pools (+0.49 [+0.02, +1.18]); kept off by owner decision. Future
test: R0 and levels keyed by coordination number and spin, pi-acceptor C donors excluded.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

B = 0.37  # A, Brown's universal softness
HAPTIC = {"C_eta2", "C_eta3", "C_eta4", "C_eta5", "C_eta6", "C_eta7", "C_eta8", "C_haptic", "C_pi"}
OS_UNKNOWN = 99
N_MIN_BONDS = 30     # R0 entry
N_MIN = 200          # V level entry (complexes)
N_SHUFFLE = 3


def element(sub: str) -> str:
    head, _, tail = sub.partition("_")
    return tail if head == "Hal" else head


def pg_of(metal: str) -> str:
    from jacobik.stats.ladders import metal_period_group

    return str(metal_period_group(metal))


def load(obs_dir: Path, datasets: Iterable[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Q1 observations of the datasets with (spin, os, sub, el, pair) columns, and a per-row flag: complex has a haptic
    site."""
    d = pd.concat([pd.read_parquet(Path(obs_dir) / f"observations_Q1_{s}.parquet") for s in datasets],
                  ignore_index=True)
    k = d.key.map(ast.literal_eval)
    d["spin"], d["os"], d["sub"] = k.str[1], k.str[2], k.str[3]
    d["el"] = d["sub"].map(element)
    d["pair"] = d.metal + "_" + d.el
    hapt = d.groupby("csd_code")["sub"].transform(lambda s: s.isin(HAPTIC).any() or s.str.startswith("C_eta").any())
    return d, hapt


def fit_r0(d: pd.DataFrame, pairs) -> Dict[str, float]:
    """R0 per pair minimising sum over complexes of (V / OS - 1)^2 (scipy least squares, sparse Jacobian)."""
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix

    pi = {p: i for i, p in enumerate(pairs)}
    cx, inv = np.unique(d.csd_code.to_numpy(), return_inverse=True)
    os_c = d.groupby("csd_code").os.first().reindex(cx).to_numpy(float)
    col = d.pair.map(pi).to_numpy()
    R = d.value.to_numpy()
    # start: R0 such that a typical bond carries OS / CN of valence
    cn = np.bincount(inv)
    s0 = os_c[inv] / cn[inv]
    r0_init = pd.Series(R + B * np.log(s0)).groupby(col).median().reindex(range(len(pairs))).to_numpy()

    def resid(r0):
        s = np.exp((r0[col] - R) / B)
        return np.bincount(inv, weights=s, minlength=len(cx)) / os_c - 1.0

    def jac(r0):
        s = np.exp((r0[col] - R) / B) / B
        return csr_matrix((s / os_c[inv], (inv, col)), shape=(len(cx), len(pairs)))

    sol = least_squares(resid, r0_init, jac=jac, method="trf", loss="soft_l1", f_scale=0.2)
    return dict(zip(pairs, sol.x))


def shuffled_lengths(d: pd.DataFrame, rng) -> np.ndarray:
    """Each bond's length replaced by a random length of the same (metal, OS, subclass) key."""
    key = d.metal + "|" + d.os.astype(str) + "|" + d["sub"]
    out = d.value.to_numpy().copy()
    for _, idx in d.groupby(key).indices.items():
        out[idx] = d.value.to_numpy()[rng.choice(idx, len(idx))]
    return out


def robust_sd(x) -> float:
    q = np.percentile(x, [16, 84])
    return 0.5 * (q[1] - q[0])


def fit_r0_ladder(d: pd.DataFrame) -> Dict[tuple, float]:
    """{('el', metal, el) | ('pg', pg, el) | ('*', el): R0}: each level fitted on the complexes whose bonds all have an
    entry at that level or above (pairs with fewer than N_MIN_BONDS bonds pooled one level down)."""
    out = {}
    for lvl, col in (("el", "pair"), ("pg", "pgpair"), ("*", "el")):
        cnt = d[col].value_counts()
        ok = set(cnt[cnt >= N_MIN_BONDS].index)
        bad = set(d.loc[~d[col].isin(ok), "csd_code"])
        sub = d[~d.csd_code.isin(bad)].copy()
        sub["pair"] = sub[col]
        r0 = fit_r0(sub, sorted(sub[col].unique()))
        for k, v in r0.items():
            parts = k.split("|") if lvl != "*" else [k]
            out[(lvl, *parts)] = float(v)
    return out


def r0_of(r0: Dict[tuple, float], metal: str, pg: str, el: str):
    for k in (("el", metal, el), ("pg", pg, el), ("*", el)):
        if k in r0:
            return r0[k]
    return None


def fit_entries(obs_dir: Path, datasets: Iterable[str], seed: int = 0) -> Dict[str, dict]:
    """DB entries {"QV:(level, ...)": level entry, "QVR:(level, ...)": {R0}} from the Q1 observations."""
    rng = np.random.default_rng(seed)
    d, hapt = load(obs_dir, datasets)
    d = d[~hapt & (d.os != OS_UNKNOWN) & (d.os > 0)].copy()
    d["pg"] = d.metal.map(pg_of)
    d["pair"] = d.metal + "|" + d.el
    d["pgpair"] = d.pg + "|" + d.el
    r0 = fit_r0_ladder(d)
    R0 = np.array([r0_of(r0, m, p, e) for m, p, e in zip(d.metal, d.pg, d.el)], dtype=float)
    keep = ~d.csd_code.isin(set(d.loc[np.isnan(R0), "csd_code"]))
    d, R0 = d[keep].copy(), R0[keep.to_numpy()]
    d["s"] = np.exp((R0 - d.value.to_numpy()) / B)
    sh = [np.exp((R0 - shuffled_lengths(d, rng)) / B) for _ in range(N_SHUFFLE)]
    g = d.groupby("csd_code", sort=False)
    cx = pd.DataFrame(dict(metal=g.metal.first(), os=g.os.first(), pg=g.pg.first(), V=g.s.sum()))
    cx_sh = [pd.Series(s, index=d.index).groupby(d.csd_code.to_numpy(), sort=False).sum().reindex(cx.index) for s in sh]
    entries: Dict[str, dict] = {}
    for lvl, cols in (("mo", ["metal", "os"]), ("po", ["pg", "os"]), ("m", ["metal"])):
        for key, grp in cx.groupby(cols):
            key = key if isinstance(key, tuple) else (key,)
            if len(grp) < N_MIN:
                continue
            ref = grp.os.astype(float) if lvl in ("mo", "po") else pd.Series(np.median(grp.V), index=grp.index)
            x = grp.V / ref
            s_o = robust_sd(x.to_numpy())
            s_i = float(np.mean([robust_sd((c[grp.index] / ref).to_numpy()) for c in cx_sh]))
            c = max(0.0, 1.0 / s_o ** 2 - 1.0 / s_i ** 2)
            entries[f"QV:{(lvl, *[k if isinstance(k, str) else int(k) for k in key])!r}"] = dict(
                n=int(len(grp)), mu=round(float(np.median(x)), 4), s_obs=round(s_o, 4), s_indep=round(s_i, 4),
                c=round(c, 3), v_ref=None if lvl != "m" else round(float(ref.iloc[0]), 4))
    for k, v in r0.items():
        entries[f"QVR:{k!r}"] = dict(n=1, R0=round(v, 4))
    return entries
