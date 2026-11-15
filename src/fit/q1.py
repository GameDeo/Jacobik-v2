"""
src/fit/q1.py
=============
Quantity 1: M-D bond distance distributions.

Key hierarchy and empirical Bayes pooling:
- Key shape (L0): (metal, metal_symbol, oxidation_state, subclass, coord_geom, trans_partner)
- Active ladder variant: "m_sub_early" (matching constraints_2026-10-18.meta.json)
- Fit model: Gaussian Mixture Model (fit_gmm) with empirical Bayes pooling (r=4.0)
  across ladder candidate ancestors.
- Variance floor: 0.002^2 (numerical floor for xTB/GFN2 precision).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from fit.quantity import fit_pooled, observation_rows, write_results
from fit import fitter as FT
from jacobik.stats import ladders as L
from fit import pooling as PL
from jacobik.core.types import TMComplex
from jacobik.stats.keys import KeyValue, _vec, q1_site_keys

log = logging.getLogger(__name__)

QUANTITY: str = "Q1"
DEFAULT_VARIANT: str = "m_sub_early"
DEFAULT_POOLING_R: float = 4.0
VAR_FLOOR: float = 0.002 ** 2
K_MAX: int = 6
N_COMP_MIN: int = 10
N_INDEP: int = 50


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def q1_bond_lengths(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, distance_Å) pairs for Q1 (metal–donor bond length).
    For haptic sites the distance is metal -> face centroid (same as the runtime).
    """
    if not mol.coord_sites or mol.coord_geom is None:
        return []
    metal_pos = _vec(mol, mol.metal_idx)
    out: List[KeyValue] = []
    for site, key in zip(mol.coord_sites, q1_site_keys(mol)):
        if key is None:
            continue
        pos = np.asarray(site.position, dtype=float)
        out.append((key, float(np.linalg.norm(pos - metal_pos))))
    return out


def extract_key_values(mol: Any) -> List[Tuple[tuple, float]]:
    """Extract (key_tuple, bond_length_A) pairs from a TMComplex."""
    return q1_bond_lengths(mol)


def extract_rows(mol: Any) -> List[Dict[str, Any]]:
    """Extract tabular observation records from a TMComplex."""
    return observation_rows(mol, extract_key_values(mol))


# ---------------------------------------------------------------------------
# Ladder & Ancestors
# ---------------------------------------------------------------------------

def ladder_candidates(key: tuple, variant: str = DEFAULT_VARIANT) -> List[tuple]:
    """Ordered candidate keys for a Q1 query under the selected ladder variant."""
    return L.q1_candidates(key, variant=variant)


def ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    """(level_int, ancestor_key) rows an L0 Q1 observation feeds."""
    return L.q1_ancestor_rows(key)


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

def fit_entry(
    values: np.ndarray | Sequence[float],
    weights: Optional[np.ndarray | Sequence[float]] = None,
    k_max: int = K_MAX,
    var_floor: float = VAR_FLOOR,
    n_comp_min: int = N_COMP_MIN,
) -> Optional[Dict[str, Any]]:
    """Fit a single GMM distribution on M-D bond distance values (Angstrom)."""
    val_arr = np.asarray(values, dtype=float)
    wt_arr = np.asarray(weights, dtype=float) if weights is not None else None
    return FT.fit_gmm(
        val_arr,
        k_max=k_max,
        var_floor=var_floor,
        n_comp_min=n_comp_min,
        weights=wt_arr,
    )


def fit_from_observations(
    df: pd.DataFrame,
    variant: str = DEFAULT_VARIANT,
    r: float = DEFAULT_POOLING_R,
    n_indep: int = N_INDEP,
    n_workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit empirical Bayes pooled GMM distributions for Q1 from an observation DataFrame.

    If df is raw observations (without 'level' and 'is_l0' columns), this function
    expands L0 observations to their ladder ancestors before running PooledFitter.
    """
    return fit_pooled(QUANTITY, df, ancestor_rows, r=r, n_indep=n_indep, n_workers=n_workers, variant=variant)


def fit_all(
    expanded_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    variant: str = DEFAULT_VARIANT,
    r: float = DEFAULT_POOLING_R,
    n_workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Fit all Q1 entries from expanded observations parquet in expanded_dir."""
    expanded_dir = Path(expanded_dir)
    df = PL.load_expanded(expanded_dir, QUANTITY)
    results = fit_from_observations(df, variant=variant, r=r, n_workers=n_workers)
    return write_results(results, out_path)
