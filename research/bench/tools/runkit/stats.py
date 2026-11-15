"""One parent-clustered bootstrap (replaces four copies; see README "Migration").

`bootstrap_mean` draws exactly like tools.jgen.bench.metrics.bootstrap_mean (same generator, same index array), so a
caller that switches gets identical numbers. `parent_bootstrap` takes per-parent value lists, averages inside each parent
first, then resamples whole parents."""
from __future__ import annotations

import numpy as np

N_BOOT = 10000
SEED = 20261027


def bootstrap_mean(values, n_boot: int = N_BOOT, seed: int = SEED):
    """(mean, lo95, hi95) of the mean over units, resampling units with replacement (percentile interval)."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def parent_bootstrap(by_parent: dict, n_boot: int = N_BOOT, seed: int = SEED, stat=np.mean):
    """by_parent: {code: value or list of values (seeds)}. Seeds are averaged within a parent, parents are resampled.
    Returns dict(n_parents, mean, lo, hi). One-sided bounds: use the 'draws' key (n_boot bootstrap means)."""
    codes = sorted(by_parent)
    v = np.array([np.mean(np.atleast_1d(by_parent[c]).astype(float)) for c in codes])
    if len(v) == 0:
        return dict(n_parents=0, mean=float("nan"), lo=float("nan"), hi=float("nan"), draws=np.zeros(0))
    rng = np.random.default_rng(seed)
    draws = stat(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return dict(n_parents=len(v), mean=float(stat(v)), lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)), draws=draws)
