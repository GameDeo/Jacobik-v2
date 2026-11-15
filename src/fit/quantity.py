"""Shared steps of the per-quantity fit modules (``fit.q1`` ... ``q6``).

Each quantity module holds what is specific to it (key extractor, ladder, model, constants); the observation table,
the ladder expansion, the grouped independent fit and the JSON writer are the same for all of them and live here."""
from __future__ import annotations

import ast
import json
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def observation_rows(mol: Any, key_values: Iterable[Tuple[tuple, float]]) -> List[Dict[str, Any]]:
    """Tabular observation records of one TMComplex from its (key tuple, value) pairs."""
    code = getattr(mol, "csd_code", "")
    metal = getattr(mol, "metal_symbol", "")
    geom = getattr(mol, "coord_geom", "unknown")
    return [
        {"csd_code": code, "key": repr(key_tuple), "key_tuple": key_tuple, "value": float(val),
         "metal": metal, "coord_geom": geom}
        for key_tuple, val in key_values
    ]


def expand_to_ancestors(df: pd.DataFrame, ancestor_rows: Callable[[tuple], Sequence[Tuple[int, tuple]]]) -> pd.DataFrame:
    """Raw L0 observations -> the expanded table (one row per ladder level the observation feeds)."""
    if "level" in df.columns and "is_l0" in df.columns:
        return df
    records = []
    for row in df.itertuples(index=False):
        raw_key = getattr(row, "key")
        key_tup = ast.literal_eval(raw_key) if isinstance(raw_key, str) else raw_key
        val = float(getattr(row, "value"))
        code = getattr(row, "csd_code")
        records.append({"key": repr(key_tup), "value": val, "csd_code": code, "level": 0, "is_l0": True})
        for lvl_int, anc_key in ancestor_rows(key_tup):
            records.append({"key": repr(anc_key), "value": val, "csd_code": code, "level": lvl_int, "is_l0": False})
    return pd.DataFrame(records)


def fit_pooled(quantity: str, df: pd.DataFrame, ancestor_rows, *, r: float, n_indep: int, n_workers: int,
               **entries_kw) -> Dict[str, Dict[str, Any]]:
    """Empirical Bayes pooled entries of one quantity from an observation table (expanded here when raw)."""
    from fit import pooling as PL

    pf = PL.PooledFitter(quantity, expand_to_ancestors(df, ancestor_rows), n_indep=n_indep)
    pf.fit_independent(n_workers=n_workers)
    return {f"{quantity}:{k}": entry for k, entry in pf.entries(r=r, **entries_kw).items()}


def group_observations(obs: Union[Dict[str, Tuple[np.ndarray, np.ndarray]], pd.DataFrame]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{key: (values, weights)} from that dict or from a DataFrame with columns key, value (and optional weight)."""
    if not isinstance(obs, pd.DataFrame):
        return obs
    grouped: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key_str, sub in obs.groupby("key", sort=False):
        vals = sub["value"].to_numpy(dtype=float)
        wts = sub["weight"].to_numpy(dtype=float) if "weight" in sub.columns else np.ones(len(vals), dtype=float)
        grouped[str(key_str)] = (vals, wts)
    return grouped


def fit_independent_groups(quantity: str, obs, fit_entry: Callable[..., Optional[dict]], n_min: int,
                           **fit_kw) -> Dict[str, Dict[str, Any]]:
    """One independent entry per key with at least ``n_min`` observations; the random state is seeded by the key."""
    out: Dict[str, Dict[str, Any]] = {}
    for key_str, (vals, wts) in group_observations(obs).items():
        if len(vals) < n_min:
            continue
        np.random.seed(zlib.crc32(f"{quantity}:{key_str}".encode()) & 0x7FFFFFFF)
        entry = fit_entry(vals, weights=wts, **fit_kw)
        if entry is not None:
            out[f"{quantity}:{key_str}"] = entry
    return out


def write_results(results: Dict[str, Any], out_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Write ``results`` as JSON when a path is given; always return them."""
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as fh:
            json.dump(results, fh, indent=1)
    return results


# ---------------------------------------------------------------------------
# Per-entry reference log-likelihood metadata
# ---------------------------------------------------------------------------

from jacobik.stats.reference_ll import ref_gmm, ref_vmm


_VAR_FLOOR = {"Q1": 0.002 ** 2, "Q2": 0.002 ** 2, "Q3": 0.002 ** 2, "Q4": 0.2 ** 2}


def entry_reference_ll(key: str, entry: dict):
    """(quantity, ref_ll, ref_sd) of one entry, or None when the entry has no 1-D reference density.

    Q1--Q4 use the same small numerical floor as the fitted scorer tables;
    Q2/Q3 are represented in cosine space. Q5 uses circular quadrature.
    Multivariate S_orient and spline tables do not have a 1-D reference density.
    """
    if not isinstance(key, str) or ":" not in key or not isinstance(entry, dict):
        return None
    quantity = key.split(":", 1)[0]
    if quantity in _VAR_FLOOR:
        result = ref_gmm(entry, _VAR_FLOOR[quantity], cos_space=quantity in ("Q2", "Q3"))
    elif quantity == "Q5":
        result = ref_vmm(entry)
    else:
        return None
    if result is None:
        return None
    return quantity, float(result[0]), float(result[1])


def add_reference_likelihoods(db: dict) -> dict[str, int]:
    """Write ref_ll/ref_sd on supported 1-D fitted entries, in place (serial reference path)."""
    counts: dict[str, int] = {}
    for key, entry in db.items():
        got = entry_reference_ll(key, entry)
        if got is None:
            continue
        quantity, ll, sd = got
        entry["ref_ll"], entry["ref_sd"] = ll, sd
        counts[quantity] = counts.get(quantity, 0) + 1
    return counts
