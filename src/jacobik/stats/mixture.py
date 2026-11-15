"""
mixture.py
==========
Evidence-weighted 1-D mixtures for the refit (2026-10-01; refit spec §2-§4, claude/refit-spec-additions-2026-09-30.md).

* ``fit_mixture``   Gaussian or Student-t mixture by weighted EM. Each row carries a weight; the fitter gives every
                    complex total weight 1 per key (1/m for its m rows), so ``n_eff`` = the number of independent
                    complexes. K is chosen by BIC with n_eff as the sample size, and a K whose fit has a component
                    resting on fewer than ``n_comp_min`` effective complexes is not selectable (this replaces the old
                    ``n // 30`` component cap, which forced every key with < 60 rows to K = 1).
* ``adapt``         MAP adaptation of a parent mixture to a key's own data (Reynolds, Quatieri & Dunn 2000, GMM-UBM):
                    each component moves toward the data it explains in proportion n_k / (n_k + r), r = relevance
                    factor in independent complexes. This is the partial pooling down a fallback ladder: a key with
                    little data stays close to its parent, a key with much data follows its own data.
* Student-t components (Peel & McLachlan 2000): the same EM with the latent precision weights u = (nu + 1) /
  (nu + z); nu is shared by the components and chosen by likelihood from ``NU_GRID``.

Pure numpy; deterministic for a given seed.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.special import gammaln

#: LnQM entries are ONE ligand set re-optimised with each lanthanide: uid "<32 hex>_<Ln>" (lnqm.h5: 17 269 entries,
#: 1 205 ligand sets, up to 15 Ln each). They are one independent structure for the evidence weights, the held-out
#: split and the metal-free S_orient levels (2026-10-15); tmQM CSD codes are their own structure.
_LNQM_UID = re.compile(r"^([0-9a-f]{32})_[A-Z][a-z]?$")


def structure_id(code: str) -> str:
    """The independent-structure id of a training complex (LnQM uid -> its ligand set; anything else unchanged)."""
    m = _LNQM_UID.match(code)
    return m.group(1) if m else code


def structure_ids(codes) -> np.ndarray:
    """structure_id of every element of an array-like of codes (vectorised over the unique codes)."""
    import pandas as pd

    c = pd.Categorical(np.asarray(codes, dtype=object))
    cats = np.array([structure_id(x) for x in c.categories], dtype=object)
    return cats[c.codes]


def structure_representative(codes) -> np.ndarray:
    """True for the rows of ONE complex per independent structure (the smallest code; every tmQM row is True). For
    unweighted fits at metal-free levels, where the LnQM siblings of a ligand set would otherwise count up to 15x."""
    import pandas as pd

    c = np.asarray(codes, dtype=object)
    first = pd.Series(c).groupby(structure_ids(c)).transform("min").to_numpy()
    return c == first


#: degrees-of-freedom grid for Student-t mixtures (prototype diag/q2_fits.py; held-out tested)
NU_GRID: Sequence[float] = (2.0, 4.0, 10.0)
EM_ITERS: int = 300
EM_TOL: float = 1e-7
N_SEEDS: int = 2


def _logpdf(x: np.ndarray, mu: np.ndarray, var: np.ndarray, nu: Optional[float]) -> np.ndarray:
    """(n, K) component log densities."""
    x = x[:, None]
    if nu is None:
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu) ** 2 / var
    z = (x - mu) ** 2 / var
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * var)
            - (nu + 1) / 2 * np.log1p(z / nu))


def _lse(a: np.ndarray) -> np.ndarray:
    m = a.max(1, keepdims=True)
    return m[:, 0] + np.log(np.exp(a - m).sum(1))


def em(x: np.ndarray, w: np.ndarray, K: int, nu: Optional[float], var_floor: float, seed: int = 0,
       iters: int = EM_ITERS) -> dict:
    """Weighted EM for a K-component Gaussian (nu None) or Student-t (shared nu) mixture."""
    rng = np.random.default_rng(seed)
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    mu = x[o][np.searchsorted(cw, (np.arange(K) + 0.5) / K).clip(0, len(x) - 1)]   # weighted quantiles
    mu = mu + rng.normal(0.0, 1e-3 * (np.std(x) + 1e-12), K)
    var = np.full(K, max(float(np.average((x - np.average(x, weights=w)) ** 2, weights=w)), var_floor))
    pi = np.full(K, 1.0 / K)
    W = float(w.sum())
    ll_old = -np.inf
    ll = -np.inf
    nk = np.full(K, W / K)
    for _ in range(iters):
        lp = np.log(pi + 1e-300)[None] + _logpdf(x, mu, var, nu)
        lse = _lse(lp)
        ll = float((w * lse).sum())
        r = np.exp(lp - lse[:, None]) * w[:, None]
        if nu is None:
            ru = r
        else:
            ru = r * (nu + 1) / (nu + (x[:, None] - mu) ** 2 / var)
        nk = r.sum(0) + 1e-12
        pi = nk / W
        mu = (ru * x[:, None]).sum(0) / ru.sum(0).clip(1e-12)
        var = np.maximum((ru * (x[:, None] - mu) ** 2).sum(0) / nk, var_floor)
        if abs(ll - ll_old) < EM_TOL * max(abs(ll), 1.0):
            break
        ll_old = ll
    return dict(mu=mu, var=var, pi=pi, nu=nu, ll=ll, nk=nk)


def fit_mixture(x: np.ndarray, w: np.ndarray, family: str = "gauss", k_max: int = 6, var_floor: float = 1e-6,
                n_comp_min: float = 10.0, nu_grid: Sequence[float] = NU_GRID, n_seeds: int = N_SEEDS) -> Optional[dict]:
    """Best mixture by BIC (sample size = n_eff = sum of weights). None if n_eff < n_comp_min.
    family: "gauss" or "t". Returns dict(mu, var, pi, nu, ll, bic, K, n_eff)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    n_eff = float(w.sum())
    if len(x) == 0 or n_eff < n_comp_min:
        return None
    best = None
    for nu in ((None,) if family == "gauss" else tuple(nu_grid)):
        best_nu_k = None
        for K in range(1, k_max + 1):
            if n_eff < K * n_comp_min:
                break
            f = max((em(x, w, K, nu, var_floor, seed=s) for s in range(n_seeds)), key=lambda f: f["ll"])
            if K > 1 and float(f["nk"].min()) < n_comp_min:
                continue
            npar = 3 * K - 1 + (0 if nu is None else 1)
            f["bic"] = -2.0 * f["ll"] + npar * math.log(max(n_eff, 2.0))
            f["K"] = K
            if best_nu_k is None or f["bic"] < best_nu_k["bic"]:
                best_nu_k = f
            elif K >= best_nu_k["K"] + 2:
                break                         # two K past the best without improvement
        if best_nu_k is not None and (best is None or best_nu_k["bic"] < best["bic"]):
            best = best_nu_k
    if best is not None:
        best["n_eff"] = n_eff
    return best


def adapt(f: dict, x: np.ndarray, w: np.ndarray, r: float, var_floor: float) -> dict:
    """MAP adaptation of mixture ``f`` to data (x, w) with relevance factor r (Reynolds et al. 2000). Means,
    variances and weights all adapt; component k moves by alpha_k = n_k / (n_k + r) toward its own sufficient
    statistics. Student-t components use the precision-weighted statistics, nu unchanged."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    mu0, var0, pi0, nu = np.asarray(f["mu"], float), np.asarray(f["var"], float), np.asarray(f["pi"], float), f.get("nu")
    if len(x) == 0:
        return dict(mu=mu0, var=var0, pi=pi0, nu=nu)
    lp = np.log(pi0 + 1e-300)[None] + _logpdf(x, mu0, var0, nu)
    g = np.exp(lp - _lse(lp)[:, None]) * w[:, None]
    gu = g if nu is None else g * (nu + 1) / (nu + (x[:, None] - mu0) ** 2 / var0)
    n = g.sum(0) + 1e-12
    su = gu.sum(0) + 1e-12
    ex = (gu * x[:, None]).sum(0) / su
    sc = (gu * (x[:, None] - ex) ** 2).sum(0) / n                 # scale (variance) about the component's own mean
    a = n / (n + r)
    pi = a * n / n.sum() + (1 - a) * pi0
    pi = pi / pi.sum()
    mu = a * ex + (1 - a) * mu0
    var = np.maximum(a * (sc + ex ** 2) + (1 - a) * (var0 + mu0 ** 2) - mu ** 2, var_floor)
    return dict(mu=mu, var=var, pi=pi, nu=nu)


def logp(f: dict, x) -> np.ndarray:
    """log density of mixture f at x (any shape)."""
    x = np.asarray(x, float)
    sh = x.shape
    lp = np.log(np.asarray(f["pi"], float) + 1e-300)[None] + _logpdf(x.ravel(), np.asarray(f["mu"], float),
                                                                      np.asarray(f["var"], float), f.get("nu"))
    return _lse(lp).reshape(sh)


def to_entry(f: dict, typ: str, n: int, n_eff: float, pooled: bool = False, **extra) -> Dict:
    """DB entry (components sorted by mean). typ: "gmm" / "gmm_cos" / "tmix_cos"."""
    o = np.argsort(f["mu"])
    e = {"type": typ, "n": int(n), "n_eff": round(float(n_eff), 3), "k": int(len(o)),
         "means": np.asarray(f["mu"], float)[o].tolist(), "vars": np.asarray(f["var"], float)[o].tolist(),
         "weights": np.asarray(f["pi"], float)[o].tolist()}
    if f.get("nu") is not None:
        e["nu"] = float(f["nu"])
    if pooled:
        e["pooled"] = True
    e.update(extra)
    return e


def from_entry(e: Dict) -> dict:
    return dict(mu=np.asarray(e["means"], float), var=np.asarray(e["vars"], float),
                pi=np.asarray(e["weights"], float), nu=e.get("nu"))
