"""
pooling.py
==========
Partial pooling of 1-D mixture fits down a fallback ladder (refit spec §3, 2026-10-01; prototype diag/q1_pool.py).

Hard back-off (the lookup walks to the first ladder level with >= n_min observations) throws away a key's own data
until it reaches n_min, and then trusts it completely. Partial pooling instead gives every key a fit that is its
ladder parent's fit MAP-adapted to the key's own data (mixture.adapt, Reynolds et al. 2000):

    key with >= N_INDEP independent complexes  -> fitted on its own data (evidence-weighted EM, K by BIC);
    any other key                               -> adapt(parent's fit, own data, r)

walking each chain from its coarsest level to L0, so a key with 2 complexes sits close to its parent and a key with
40 follows its data. Every entry is stored with ``pooled: True``; the lookup (StatsLookup._resolve) takes the most
specific pooled entry it finds. Held-out evidence on the 3-shard sample (claude/independent-audit-2026-09-29.md):
mean / 5th-percentile log p per held-out complex at keys with 0 training complexes 1.17 / -1.34 (hard back-off) ->
1.36 / -0.31 (pooled, r = 4); 1-9 complexes 1.38 / -0.35 -> 1.73 / +0.52; equal at rich keys.

The parent of a key is the next key PRESENT in the data along the runtime candidate order of an L0 key that reaches it
(ladders.q1_candidates for the chosen Q1 variant, ladders.q2_candidates). Independent fits do not depend on r or on
the ladder variant, so ``PooledFitter.entries(r, variant)`` is cheap after ``fit_independent``.
"""
from __future__ import annotations

import ast
import logging
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from jacobik.stats import ladders as L
from jacobik.stats import mixture as MX

log = logging.getLogger(__name__)

#: a key with at least this many independent complexes is fitted on its own data
N_INDEP: int = 50
#: relevance factor (independent complexes); chosen on held-out complexes in the refit notebook (tested 4 / 10 / 25)
R_DEFAULT: float = 4.0
K_MAX_POOL: int = 6
N_COMP_MIN: float = 10.0

#: per quantity: (mixture family, entry type, value transform, variance floor)
SPEC: Dict[str, tuple] = {
    "Q1": ("gauss", "gmm", None, 0.002 ** 2),
    "Q2": ("t", "tmix_cos", "cos_deg", 0.002 ** 2),
}


def _transform(values: np.ndarray, how: Optional[str]) -> np.ndarray:
    return np.cos(np.deg2rad(values)) if how == "cos_deg" else values


def fit_one_checked(args):
    """(key, fit or None, error text or None): the independent fit of one key; a raised exception is returned, not lost."""
    key_str, x, w, family, var_floor = args
    seed = zlib.crc32(key_str.encode()) & 0x7FFFFFFF
    np.random.seed(seed)
    try:
        return key_str, MX.fit_mixture(x, w, family, k_max=K_MAX_POOL, var_floor=var_floor, n_comp_min=N_COMP_MIN), None
    except Exception as exc:  # pragma: no cover
        log.warning("independent fit failed for %s: %s", key_str, exc)
        return key_str, None, f"{type(exc).__name__}: {exc}"[:300]


def _fit_one(args):
    key_str, fit, _ = fit_one_checked(args)
    return key_str, fit


def _fit_chunk(chunk):
    return [_fit_one(a) for a in chunk]


class PooledFitter:
    """Partial pooling for one quantity (Q1 or Q2) over an expanded observation table."""

    def __init__(self, quantity: str, df: pd.DataFrame, n_indep: int = N_INDEP):
        """df: expanded rows with columns key, value, csd_code, level, is_l0 (expander output)."""
        if quantity not in SPEC:
            raise ValueError(f"no pooling spec for {quantity}")
        self.q = quantity
        self.family, self.typ, how, self.var_floor = SPEC[quantity]
        self.n_indep = n_indep
        t0 = time.perf_counter()
        df = df[["key", "value", "csd_code", "level", "is_l0"]].copy()
        df["x"] = _transform(df["value"].to_numpy(np.float64), how)
        # independent structures (LnQM siblings of one ligand set are one; mixture.structure_id)
        df["sid"] = MX.structure_ids(df["csd_code"].to_numpy())
        df["w"] = 1.0 / df.groupby(["key", "sid"], sort=False)["value"].transform("size").to_numpy()
        g = df.groupby("key", sort=False)
        x_all, w_all = df["x"].to_numpy(), df["w"].to_numpy()
        self.data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {k: (x_all[i], w_all[i]) for k, i in g.indices.items()}
        self.n_rows = {k: len(v[0]) for k, v in self.data.items()}
        self.n_cx = g["sid"].nunique().to_dict()
        self.tuples = {k: ast.literal_eval(k) for k in self.data}
        self.present = {t: k for k, t in self.tuples.items()}
        # L0 query keys whose candidate chains define the parent links
        l0 = set(df.loc[df["is_l0"], "key"].unique())
        self.l0_keys = [self.tuples[k] for k in l0]
        if quantity == "Q1":
            # rows without a trans partner are not L0 rows: their first level is "geom" (5-tuple); query = + "none"
            geom_lvl = L.Q1_LEVEL_INT["geom"]
            gk = set(df.loc[(~df["is_l0"]) & (df["level"] == geom_lvl), "key"].unique())
            self.l0_keys += [self.tuples[k] + (L.TRANS_NONE,) for k in gk]
        self.indep: Dict[str, Optional[dict]] = {}
        log.info("%s pooling: %d keys, %d L0 query keys (%.1fs)", quantity, len(self.data), len(self.l0_keys),
                 time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def candidates(self, key: tuple, variant: Optional[str]) -> List[tuple]:
        if self.q == "Q1":
            return L.q1_candidates(key, variant or L.DEFAULT_Q1_VARIANT)
        return L.q2_candidates(key)

    def parents(self, variant: Optional[str] = None) -> Dict[str, Optional[str]]:
        """key -> parent key (next present key along a runtime candidate chain); None for chain roots. The first
        chain that reaches a key decides (chains are visited in a fixed order: most data first)."""
        par: Dict[str, Optional[str]] = {}
        order = sorted(self.l0_keys, key=lambda t: (-self.n_cx.get(repr(t), 0), repr(t)))
        for l0 in order:
            chain = [self.present[c] for c in self.candidates(l0, variant) if c in self.present]
            for a, b in zip(chain, chain[1:] + [None]):
                if a not in par:
                    par[a] = b
        return par

    def independent_keys(self) -> List[str]:
        """Keys fitted on their own data: >= n_indep independent complexes, or a chain root of the default variant."""
        roots = {k for k, p in self.parents().items() if p is None}
        return sorted(k for k in self.data if self.n_cx.get(k, 0) >= self.n_indep or k in roots)

    def independent_items(self, keys: Optional[Sequence[str]] = None) -> List[tuple]:
        """Work items ``(key, x, w, family, var_floor)`` for ``_fit_one``, largest first (the pool stays busy)."""
        if keys is None:
            keys = self.independent_keys()
        todo = [k for k in keys if k not in self.indep]
        items = [(k, self.data[k][0], self.data[k][1], self.family, self.var_floor) for k in todo]
        items.sort(key=lambda a: (-len(a[1]), a[0]))
        return items

    def fit_independent(self, keys: Optional[Sequence[str]] = None, n_workers: int = 1, chunk: int = 32,
                        progress: Optional[Callable[[str], None]] = None) -> int:
        """Fit every key with >= n_indep complexes (and every chain root of the default variant) on its own data."""
        items = self.independent_items(keys)
        # big keys first so the pool stays busy
        items.sort(key=lambda a: -len(a[1]))
        t0 = time.perf_counter()
        if n_workers > 1 and len(items) > chunk:
            from joblib import Parallel, delayed, parallel_config

            chunks = [items[i::max(1, len(items) // chunk)] for i in range(max(1, len(items) // chunk))]
            with parallel_config(backend="loky", inner_max_num_threads=1, max_nbytes="1M", mmap_mode="r"):
                done = 0
                for res in Parallel(n_jobs=n_workers, return_as="generator_unordered")(
                        delayed(_fit_chunk)(c) for c in chunks):
                    for k, f in res:
                        self.indep[k] = f
                    done += 1
                    if progress and done % max(1, len(chunks) // 10) == 0:
                        progress(f"{self.q}: independent fits {done}/{len(chunks)} chunks "
                                 f"({time.perf_counter() - t0:.0f}s)")
        else:
            for a in items:
                k, f = _fit_one(a)
                self.indep[k] = f
        return len(items)

    def entries(self, r: float = R_DEFAULT, variant: Optional[str] = None) -> Dict[str, dict]:
        """{key_str: DB entry} for every key that gets a fit (independent or adapted)."""
        par = self.parents(variant)
        memo: Dict[str, Optional[dict]] = {}

        def get(k: str) -> Optional[dict]:
            if k in memo:
                return memo[k]
            f = self.indep.get(k)
            src = "fit"
            if f is None:
                p = par.get(k)
                pf = get(p) if p is not None else None
                if pf is None:
                    memo[k] = None
                    return None
                x, w = self.data[k]
                f = MX.adapt(pf, x, w, r, self.var_floor)
                src = "adapted"
            memo[k] = dict(f, src=src)
            return memo[k]

        # iterative deepening guard: resolve coarse keys first to keep recursion shallow
        out: Dict[str, dict] = {}
        for k in sorted(self.data, key=lambda k: len(self.tuples[k])):
            f = get(k)
            if f is None:
                continue
            out[k] = MX.to_entry(f, self.typ, self.n_rows[k], self.n_cx.get(k, 0), pooled=True, src=f["src"])
        return out


def load_expanded(expanded_dir: Path, quantity: str) -> pd.DataFrame:
    path = Path(expanded_dir) / f"expanded_{quantity}.parquet"
    return pd.read_parquet(str(path), columns=["key", "value", "csd_code", "level", "is_l0"])


# ---------------------------------------------------------------------------
# Ladder propagation of arbitrary estimates (dict child -> parent, candidate generator or chain)
# ---------------------------------------------------------------------------

VAR_FLOOR_DEFAULT: float = 0.002 ** 2


def pool_ladder(
    estimates: Dict[Any, Any],
    ladder: Union[Sequence[Any], Dict[Any, Any], Callable[[Any], List[Any]]],
    r: float = R_DEFAULT,
    var_floor: float = VAR_FLOOR_DEFAULT,
) -> Dict[Any, Any]:
    """Propagate empirical Bayes partial pooling down a fallback ladder.

    Parameters
    ----------
    estimates : Dict[Any, Any]
        Dictionary of estimates for each key along the ladder.
        Each entry can be:
        - A dict containing:
            - "fit": Optional prior or independent fit dict (with 'mu', 'var', 'pi', optional 'nu')
            - "x": 1D array of observations
            - "w": 1D array of weights (optional, defaults to 1.0)
            - "src": "fit" or "adapted"
        - Or directly a fit dict (keys 'mu', 'var', 'pi').
        - Or a tuple (x, w) with observations and weights.
    ladder : Union[Sequence[Any], Dict[Any, Any], Callable]
        Specification of ladder relationships:
        - Sequence of keys: [k0, k1, ...] either coarse-to-fine or fine-to-coarse.
        - Mapping: {child_key: parent_key}
        - Candidate generator: callable(key) -> list of candidate keys.
    r : float
        Relevance factor for MAP adaptation (default 4.0).
    var_floor : float
        Variance floor for Gaussian / Student-t components.

    Returns
    -------
    Dict[Any, Any]
        Dictionary mapping keys to their pooled/adapted fit dictionaries.
    """
    pooled_results: Dict[Any, Any] = {}

    def extract_entry(val: Any) -> Tuple[Optional[dict], np.ndarray, np.ndarray, str]:
        if isinstance(val, dict):
            if "mu" in val and "var" in val and "pi" in val:
                # Direct fit dict
                x = np.empty(0, dtype=np.float64)
                w = np.empty(0, dtype=np.float64)
                return val, x, w, val.get("src", "fit")
            fit = val.get("fit")
            x = np.asarray(val.get("x", []), dtype=np.float64)
            w = np.asarray(val.get("w", np.ones_like(x)), dtype=np.float64)
            src = val.get("src", "fit" if fit is not None else "adapted")
            return fit, x, w, src
        elif isinstance(val, tuple) and len(val) == 2:
            x = np.asarray(val[0], dtype=np.float64)
            w = np.asarray(val[1], dtype=np.float64)
            return None, x, w, "adapted"
        else:
            return None, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), "adapted"

    # Case 1: ladder is a dictionary of child -> parent
    if isinstance(ladder, dict):
        par_map: Dict[Any, Any] = dict(ladder)
        memo: Dict[Any, Optional[dict]] = {}

        def get_fit(k: Any) -> Optional[dict]:
            if k in memo:
                return memo[k]
            if k not in estimates:
                memo[k] = None
                return None
            val = estimates[k]
            fit, x, w, src = extract_entry(val)
            if fit is not None and src == "fit":
                res = dict(fit, src="fit", pooled=True)
                memo[k] = res
                return res

            p = par_map.get(k)
            pf = get_fit(p) if p is not None else None
            if pf is None:
                # Root or parent missing: use own fit if available
                if fit is not None:
                    res = dict(fit, src=src, pooled=True)
                    memo[k] = res
                    return res
                memo[k] = None
                return None

            # MAP adapt from parent
            adapted = MX.adapt(pf, x, w, r=r, var_floor=var_floor)
            res = dict(adapted, src="adapted", pooled=True)
            memo[k] = res
            return res

        for k in estimates:
            f = get_fit(k)
            if f is not None:
                pooled_results[k] = f
        return pooled_results

    # Case 2: ladder is a candidate generator function
    if callable(ladder):
        # Build child -> parent mapping from candidate chains
        par_map = {}
        for k in estimates:
            cands = ladder(k)
            for c in cands:
                if c != k and c in estimates:
                    par_map[k] = c
                    break
        return pool_ladder(estimates, par_map, r=r, var_floor=var_floor)

    # Case 3: ladder is a sequence of keys along a single chain
    chain = list(ladder)
    if not chain:
        return {}

    # Check whether chain is fine-to-coarse (e.g. leaf to root) or coarse-to-fine
    # If the last item has an independent fit and the first does not, reverse
    first_fit, _, _, first_src = extract_entry(estimates.get(chain[0]))
    last_fit, _, _, last_src = extract_entry(estimates.get(chain[-1]))
    if (last_fit is not None and first_fit is None) or (first_fit is not None and first_src == "adapted" and last_src == "fit"):
        chain = list(reversed(chain))

    # Walk from root to leaf
    curr_fit: Optional[dict] = None
    for k in chain:
        if k not in estimates:
            continue
        val = estimates[k]
        fit, x, w, src = extract_entry(val)
        if fit is not None and src == "fit":
            curr_fit = dict(fit, src="fit", pooled=True)
        elif curr_fit is not None:
            curr_fit = dict(
                MX.adapt(curr_fit, x, w, r=r, var_floor=var_floor),
                src="adapted",
                pooled=True,
            )
        elif fit is not None:
            curr_fit = dict(fit, src=src, pooled=True)

        if curr_fit is not None:
            pooled_results[k] = dict(curr_fit)

    return pooled_results
