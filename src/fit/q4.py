"""
src/fit/q4.py
=============
Quantity 4: L-D-M bend angle distributions (donor pyramidalization).

Key hierarchy and model:
- Key shape (L0): (subclass, coord_geom, k, anion_bucket)
  where k is the number of ligand substituents on the donor.
- Fallback ladder: jacobik.stats.ladders.q4_candidates (default variant: "anionic_context")
- Fit model: Gaussian Mixture Model in degree space (fit_gmm)
- Variance floor: 0.2^2 (in degree space).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from fit.quantity import fit_independent_groups, observation_rows, write_results
from fit import fitter as FT
from jacobik.stats import ladders as L
from jacobik.core.types import TMComplex
from jacobik.stats.keys import KeyValue, _angle_deg, _site_representative_idx, _vec
log = logging.getLogger(__name__)

log = logging.getLogger(__name__)

QUANTITY: str = "Q4"
DEFAULT_VARIANT: str = "anionic_context"
VAR_FLOOR: float = 0.2 ** 2
K_MAX: int = 4
N_MIN: int = 30  # legacy fitter N_MIN; keep the extraction gate aligned with fit_gmm
N_COMP_MIN: int = 10


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def q4_pyramidalization(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, mean_angle_deg) pairs for Q4 (donor pyramidalization).
    Key = (subclass, geom, k, anion_bucket) with k = number of ligand
    substituents on the donor (H included): the mean of the k+1 substituent
    angles depends strongly on k. anion_bucket is the same co-donor-context
    feature as Q3 (see anionic_co_donor_bucket); Q3 and Q4 are ~0.99
    correlated for O_water, so whatever context explains one should be
    expected to explain the other.
    """
    from jacobik.stats.keys import _site_subclass, anionic_co_donor_bucket
    from jacobik.topology.graph import get_ligand_neighbors

    if not mol.coord_sites or mol.coord_geom is None:
        return []

    atom_map = {a.idx: a for a in mol.atoms}
    metal_idx = mol.metal_idx
    metal_pos = _vec(mol, metal_idx)
    geom = mol.coord_geom

    # Build full ligand-graph neighbor table (no metal, includes H)
    all_neighbors = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)

    out: List[KeyValue] = []

    for site in mol.coord_sites:
        if site.is_haptic:
            continue
        try:
            donor_idx = site.donor_indices[0]
            subclass = _site_subclass(site, mol)
            anion_bucket = anionic_co_donor_bucket(mol, exclude_rep_idx=_site_representative_idx(site))
            nbs = all_neighbors.get(donor_idx, [])
            k = len(nbs)
            if k < 2:
                continue  # terminal donor — Q4 undefined (would just reproduce Q3)

            donor_pos = _vec(mol, donor_idx)

            # S = {M, X₁, ..., Xₖ}  —  positions of all n = k+1 substituents
            s_positions = [metal_pos] + [np.asarray(atom_map[n].position, dtype=float) for n in nbs]
            n = len(s_positions)  # = k + 1

            sigma_theta = 0.0
            valid = True
            for a_i in range(n):
                for b_i in range(a_i + 1, n):
                    ang = _angle_deg(s_positions[a_i], donor_pos, s_positions[b_i])
                    if ang is None:
                        valid = False
                        break
                    sigma_theta += ang
                if not valid:
                    break

            if not valid:
                continue

            n_pairs = n * (n - 1) // 2
            mean_angle = sigma_theta / n_pairs
            out.append(((subclass, geom, k, anion_bucket), mean_angle))

        except Exception as exc:
            log.debug("Q4 skip site %s: %s", site, exc)

    return out


def extract_key_values(mol: Any) -> List[Tuple[tuple, float]]:
    """Extract (key_tuple, angle_deg) pairs for Q4 pyramidalization from a TMComplex."""
    return q4_pyramidalization(mol)


def extract_rows(mol: Any) -> List[Dict[str, Any]]:
    """Extract tabular observation records from a TMComplex."""
    return observation_rows(mol, extract_key_values(mol))


# ---------------------------------------------------------------------------
# Ladder & Ancestors
# ---------------------------------------------------------------------------

def ladder_candidates(key: tuple, variant: str = DEFAULT_VARIANT) -> List[tuple]:
    """Ordered candidate keys for a Q4 query."""
    return L.q4_candidates(key, variant=variant)


def ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    """(level_int, ancestor_key) rows an L0 Q4 observation feeds."""
    return L.q4_ancestor_rows(key)


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

def fit_entry(
    values_deg: np.ndarray | Sequence[float],
    weights: Optional[np.ndarray | Sequence[float]] = None,
    k_max: int = K_MAX,
    var_floor: float = VAR_FLOOR,
    n_comp_min: int = N_COMP_MIN,
) -> Optional[Dict[str, Any]]:
    """Fit a GMM in degree space for L-D-M pyramidalization bend angles."""
    val_arr = np.asarray(values_deg, dtype=float)
    wt_arr = np.asarray(weights, dtype=float) if weights is not None else None
    return FT.fit_gmm(
        val_arr,
        k_max=k_max,
        var_floor=var_floor,
        n_comp_min=n_comp_min,
        weights=wt_arr,
    )


def fit_from_observations(
    obs: Union[Dict[str, Tuple[np.ndarray, np.ndarray]], pd.DataFrame],
    n_min: int = N_MIN,
    k_max: int = K_MAX,
    var_floor: float = VAR_FLOOR,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit independent GMM entries across grouped observations.

    obs can be:
    - dict: {key_str: (values_array, weights_array)}
    - DataFrame: columns ['key', 'value'] (and optional 'weight')
    """
    return fit_independent_groups(QUANTITY, obs, fit_entry, n_min, k_max=k_max, var_floor=var_floor)


def fit_all(
    expanded_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    n_min: int = N_MIN,
) -> Dict[str, Dict[str, Any]]:
    """Fit all Q4 entries from expanded observations parquet in expanded_dir."""
    expanded_dir = Path(expanded_dir)
    obs = FT._load_observations(expanded_dir, QUANTITY, n_min=n_min)
    results = fit_from_observations(obs, n_min=n_min)
    return write_results(results, out_path)
