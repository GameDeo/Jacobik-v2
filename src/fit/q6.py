"""
src/fit/q6.py
=============
Quantity 6: Steric non-bonded exchange repulsive wall.

Key shape and model:
- Key shape: ('elemA', 'elemB') where polar hydrogens are distinguished as 'Hp'
- Fit model: 5th-percentile half-Gaussian (fit_halfgauss, model="p5")
  mu = 5th percentile of non-bonded contact distances
  sigma = standard deviation from mirrored left flank (floor 0.1 A)
- Metal pooling: Per-metal (M, X) keys pool into period-group keys (period_group, X).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from fit.quantity import group_observations, observation_rows, write_results
from fit import fitter as FT
from jacobik.core.types import TMComplex
from jacobik.stats.keys import DIST_CUTOFF_Q6, KeyValue, q6_pair_key, q6_polar_h
log = logging.getLogger(__name__)

log = logging.getLogger(__name__)

QUANTITY: str = "Q6"
MODEL_NAME: str = "p5"
N_MIN: int = 30  # legacy fitter N_MIN
SIGMA_FLOOR: float = 0.1


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

Q6_PAIR_CAP: Dict[Tuple[str, str], int] = {
    # Heavy–heavy (most abundant)
    ("C", "C"): 30,
    ("C", "N"): 25,
    ("C", "O"): 25,
    ("C", "P"): 20,
    ("C", "S"): 20,
    ("N", "N"): 20,
    # H-involving (very abundant due to ligand H count; capped tighter)
    ("C", "H"): 20,
    ("H", "H"): 10,
    ("H", "N"): 15,
    ("H", "O"): 15,
    ("H", "P"): 12,
    ("H", "S"): 12,
}


def q6_steric_contacts(mol: TMComplex, seed: Optional[int] = None) -> List[KeyValue]:
    """
    Produce (key, distance_Å) pairs for Q6 (pairwise non-bonded contacts).

    * The metal is INCLUDED (key = sorted element pair with the metal symbol),
      matching what the runtime scorer evaluates.
    * Pairs within ``Q6_EXCLUDE_MAX_BONDS`` bonds on the full graph (H and
      metal-donor edges included) are excluded; the same function is used by
      ``ik.geometry`` so extraction and scoring agree.
    """
    import random

    from jacobik.topology.mapping import raw_metal_bond_exclusion_pairs
    from jacobik.topology.graph import donor_indices_of, q6_topological_exclusions

    all_atoms = list(mol.atoms)
    n = len(all_atoms)
    if n < 2:
        return []

    positions = np.array([a.position for a in all_atoms], dtype=np.float64)
    close = q6_topological_exclusions(
        mol.atoms, mol.bonds, mol.metal_idx, donor_indices_of(mol)
    )
    # Raw M--X BO evidence only suppresses the direct pair from Q6.  It does
    # not create donor graph edges or change Q1--Q5 coordination assignment.
    close.update(raw_metal_bond_exclusion_pairs(mol.atoms, mol.bonds, mol.metal_idx, q6=True))

    polar = q6_polar_h(mol)
    d2 = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1)
    ii, jj = np.nonzero(np.triu(d2 <= DIST_CUTOFF_Q6 * DIST_CUTOFF_Q6, k=1))

    buckets: Dict[Tuple[str, str], List[float]] = {}
    for a_i, a_j in zip(ii.tolist(), jj.tolist()):
        ai = all_atoms[a_i]
        aj = all_atoms[a_j]
        lo, hi = (ai.idx, aj.idx) if ai.idx < aj.idx else (aj.idx, ai.idx)
        if (lo, hi) in close:
            continue
        key = q6_pair_key(ai.symbol, aj.symbol, ai.idx in polar, aj.idx in polar)
        buckets.setdefault(key, []).append(float(d2[a_i, a_j] ** 0.5))

    rng = random.Random(seed if seed is not None else 0)
    out: List[KeyValue] = []
    for key, distances in buckets.items():
        cap = Q6_PAIR_CAP.get(key)
        if cap is not None and len(distances) > cap:
            distances = rng.sample(distances, cap)
        for d in distances:
            out.append((key, d))
    return out


def extract_key_values(mol: Any, seed: Optional[int] = None) -> List[Tuple[tuple, float]]:
    """Extract non-bonded steric contact distance pairs from a TMComplex."""
    return q6_steric_contacts(mol, seed=seed)


def extract_rows(mol: Any, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Extract tabular observation records from a TMComplex."""
    return observation_rows(mol, extract_key_values(mol, seed=seed))


# ---------------------------------------------------------------------------
# Metal Pooling
# ---------------------------------------------------------------------------

def add_pooled_metal_keys(
    obs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_min: int = N_MIN,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Add (period_group, X) pooled keys built from (M, X) rows.
    Lookup order: exact (M, X) -> (period_group, X) -> covalent-radius fallback.
    """
    return FT._add_q6_pooled_metal_keys(obs, n_min=n_min)


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

def fit_entry(
    values: np.ndarray | Sequence[float],
    model: str = MODEL_NAME,
    n_min: int = N_MIN,
) -> Optional[Dict[str, Any]]:
    """
    Fit half-Gaussian wall at 5th percentile with mirrored left flank spread.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < n_min:
        return None
    fit = FT.fit_halfgauss(arr)
    if fit is not None:
        fit["model"] = model
    return fit


def fit_from_observations(
    obs: Union[Dict[str, Tuple[np.ndarray, np.ndarray]], pd.DataFrame],
    n_min: int = N_MIN,
    pool_metals: bool = True,
    model: str = MODEL_NAME,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit half-Gaussian repulsive walls for all Q6 element-pair keys.

    obs can be:
    - dict: {key_str: (values_array, weights_array)}
    - DataFrame: columns ['key', 'value']
    """
    grouped = dict(group_observations(obs))

    if pool_metals:
        # Legacy fitter adds period-group observations before fitting so
        # pooled fallback walls use the full raw distributions.
        grouped = add_pooled_metal_keys(grouped, n_min=n_min)

    out: Dict[str, Dict[str, Any]] = {}
    for key_str, (vals, _) in grouped.items():
        entry = fit_entry(vals, model=model, n_min=n_min)
        if entry is not None:
            out[f"{QUANTITY}:{key_str}"] = entry
    if pool_metals:
        # Match the legacy fitter: derive period-group fallbacks from fitted
        # metal walls after fitting, rather than pooling raw observations.
        FT._add_pooled_q6_metal_keys(out)
    return out


def fit_all(
    expanded_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    n_min: int = N_MIN,
    model: str = MODEL_NAME,
) -> Dict[str, Dict[str, Any]]:
    """Fit all Q6 entries from expanded observations parquet in expanded_dir."""
    expanded_dir = Path(expanded_dir)
    obs = FT._load_observations(expanded_dir, QUANTITY, n_min=n_min)
    results = fit_from_observations(obs, n_min=n_min, model=model)
    return write_results(results, out_path)
