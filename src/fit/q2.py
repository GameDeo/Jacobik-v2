"""
src/fit/q2.py
=============
Quantity 2: D-M-D cis / bite angle distributions.

Key hierarchy and empirical Bayes pooling:
- Key shape (L0): (coord_geom, pair_class, dist_class, metal, subA, subB)
- Fallback ladder: jacobik.stats.ladders.q2_candidates
- Fit model: Cosine-space Student-t / Normal mixture (tmix_cos) with empirical Bayes
  pooling (r=4.0) down the pair-class-preserving ladder.
- Variance floor: 0.002^2 (in cosine space).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from fit.quantity import fit_pooled, observation_rows, write_results
from jacobik.stats import ladders as L
from jacobik.stats import mixture as MX
from fit import pooling as PL
from jacobik.core.types import TMComplex
from jacobik.stats.keys import KeyValue, _angle_deg, _vec, q2_pair_keys

log = logging.getLogger(__name__)

QUANTITY: str = "Q2"
DEFAULT_POOLING_R: float = 4.0
VAR_FLOOR: float = 0.002 ** 2
K_MAX: int = 6
N_COMP_MIN: float = 10.0
N_INDEP: int = 50


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def q2_dmd_angles(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, angle_deg) pairs for Q2 (donor–metal–donor angle).
    """
    sites = mol.coord_sites
    if len(sites) < 2 or mol.coord_geom is None:
        return []
    metal_pos = _vec(mol, mol.metal_idx)
    out: List[KeyValue] = []
    for i, j, key in q2_pair_keys(mol):
        pa = np.asarray(sites[i].position, dtype=float)
        pb = np.asarray(sites[j].position, dtype=float)
        angle = _angle_deg(pa, metal_pos, pb)
        if angle is not None:
            out.append((key, angle))
    return out


def extract_key_values(mol: Any) -> List[Tuple[tuple, float]]:
    """Extract (key_tuple, angle_deg) pairs from a TMComplex."""
    return q2_dmd_angles(mol)


def extract_rows(mol: Any) -> List[Dict[str, Any]]:
    """Extract tabular observation records from a TMComplex."""
    return observation_rows(mol, extract_key_values(mol))


# ---------------------------------------------------------------------------
# Ladder & Ancestors
# ---------------------------------------------------------------------------

def ladder_candidates(key: tuple) -> List[tuple]:
    """Ordered candidate keys for a Q2 query."""
    return L.q2_candidates(key)


def ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    """(level_int, ancestor_key) rows an L0 Q2 observation feeds."""
    return L.q2_ancestor_rows(key)


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

def fit_entry(
    values_deg: np.ndarray | Sequence[float],
    weights: Optional[np.ndarray | Sequence[float]] = None,
    k_max: int = K_MAX,
    var_floor: float = VAR_FLOOR,
    n_comp_min: float = N_COMP_MIN,
) -> Optional[Dict[str, Any]]:
    """
    Fit a Student-t / Normal mixture on cos(angle) for D-M-D angles.
    """
    arr = np.asarray(values_deg, dtype=float)
    n = len(arr)
    if n < FT_N_MIN(n_comp_min):
        return None
    x = np.cos(np.deg2rad(arr))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    fit = MX.fit_mixture(x, w, family="t", k_max=k_max, var_floor=var_floor, n_comp_min=n_comp_min)
    if fit is None:
        return None
    return MX.to_entry(fit, "tmix_cos", n, fit["n_eff"])


def FT_N_MIN(n_comp_min: float) -> int:
    return max(15, int(n_comp_min))


def fit_from_observations(
    df: pd.DataFrame,
    r: float = DEFAULT_POOLING_R,
    n_indep: int = N_INDEP,
    n_workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit empirical Bayes pooled tmix_cos distributions for Q2 from an observation DataFrame.

    If df is raw observations (without 'level' and 'is_l0' columns), expands L0
    observations to their ladder ancestors before running PooledFitter.
    """
    return fit_pooled(QUANTITY, df, ancestor_rows, r=r, n_indep=n_indep, n_workers=n_workers)


def fit_all(
    expanded_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    r: float = DEFAULT_POOLING_R,
    n_workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Fit all Q2 entries from expanded observations parquet in expanded_dir."""
    expanded_dir = Path(expanded_dir)
    df = PL.load_expanded(expanded_dir, QUANTITY)
    results = fit_from_observations(df, r=r, n_workers=n_workers)
    return write_results(results, out_path)
