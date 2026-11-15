"""
reference_ll.py
===============
B2.6: per-term reference log-likelihood. For every fitted distribution, the mean and sd of log p(x) when x is a
typical real observation (drawn from the fitted distribution itself) and p is evaluated exactly as the scorer does
(GMM variances floored at scorer._GMM_VAR_MIN; S_orient per-dim average). A hypothesis-level score can then be
normalised per term, (log p - ref_ll) / ref_sd, so hypotheses with different site counts or term sets compare fairly
(plan 2026-09-26, B4.5). The DB loader reads "ref_ll" / "ref_sd" from the entries; nothing writes them any more (the
helper that added them was removed 2026-10-17), only the 1-D references (Q1-Q4 mixtures, Q5 vMF) remain.

Deterministic: 1-D mixtures use stratified quantiles per component (no RNG).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

N_REF = 512


def _strata(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def _mixture_samples_1d(means, sds, weights, n: int = N_REF) -> np.ndarray:
    from scipy.stats import norm

    out = []
    w = np.asarray(weights, float)
    w = w / w.sum()
    counts = np.maximum(1, np.round(w * n).astype(int))
    for m, s, c in zip(means, sds, counts):
        out.append(m + s * norm.ppf(_strata(int(c))))
    return np.concatenate(out), np.repeat(w / counts, counts)  # samples, weights (sum 1)


def _gmm_logp(x, means, vars_, weights) -> np.ndarray:
    m, v, w = (np.asarray(a, float) for a in (means, vars_, weights))
    z = -0.5 * np.log(2 * np.pi * v) - 0.5 * (x[:, None] - m) ** 2 / v + np.log(np.maximum(w, 1e-300))
    return np.logaddexp.reduce(z, axis=1)


def ref_gmm(entry: dict, var_floor: float, cos_space: bool = False) -> Optional[tuple]:
    if not entry.get("means") or entry.get("vars") is None:
        return None
    v = np.maximum(np.asarray(entry["vars"], float), 1e-12)
    x, wx = _mixture_samples_1d(entry["means"], np.sqrt(v), entry["weights"])
    if cos_space:
        x = np.clip(x, -1.0, 1.0)
    lp = _gmm_logp(x, entry["means"], np.maximum(v, var_floor), entry["weights"])
    mu = float((wx * lp).sum())
    return mu, float(math.sqrt(max((wx * (lp - mu) ** 2).sum(), 0.0)))


def ref_vmm(entry: dict, n_grid: int = 720) -> Optional[tuple]:
    """Circular mixtures: E[log p] and sd by quadrature on a fixed angle grid (the fitted density is also the
    scored one; no floor)."""
    from scipy.special import i0e

    if not entry.get("means") or entry.get("kappas") is None:
        return None
    m, k, w = (np.asarray(entry[a], float) for a in ("means", "kappas", "weights"))
    w = w / w.sum()
    x = -np.pi + (np.arange(n_grid) + 0.5) * (2 * np.pi / n_grid)
    ln = np.log(2 * np.pi) + np.log(np.maximum(i0e(k), 1e-300)) + k
    z = k * np.cos(x[:, None] - m) - ln + np.log(np.maximum(w, 1e-300))
    lp = np.logaddexp.reduce(z, axis=1)
    p = np.exp(lp)
    p = p / p.sum()
    mu = float((p * lp).sum())
    return mu, float(math.sqrt(max((p * (lp - mu) ** 2).sum(), 0.0)))
