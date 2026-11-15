"""
src/fit/q3.py
=============
Quantity 3: D-M-D trans angle / donor approach angle distributions.

Key hierarchy and model:
- Key shape (L0): (subclass, metal, coord_geom, anion_bucket)
- Fallback ladder: jacobik.stats.ladders.q3_candidates (default variant: "anionic_context")
- Fit model: Gaussian Mixture Model on cos(angle) (fit_gmm_cosangle / gmm_cos)
- Variance floor: 0.002^2 (in cosine space).
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
from jacobik.stats.keys import KeyValue, _site_representative_idx, _vec
log = logging.getLogger(__name__)

log = logging.getLogger(__name__)

QUANTITY: str = "Q3"
DEFAULT_VARIANT: str = "anionic_context"
VAR_FLOOR: float = 0.002 ** 2
K_MAX: int = 6
N_MIN: int = 30  # legacy fitter N_MIN; keep the extraction gate aligned with fit_gmm
N_COMP_MIN: int = 10


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def q3_approach_angles(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, angle_deg) pairs for Q3 (donor approach angle).
    """
    from jacobik.stats.keys import _site_subclass, anionic_co_donor_bucket
    from jacobik.topology.graph import get_ligand_neighbors

    if not mol.coord_sites or mol.coord_geom is None:
        return []

    atom_map = {a.idx: a for a in mol.atoms}
    metal_pos = _vec(mol, mol.metal_idx)
    metal_idx = mol.metal_idx
    metal = mol.metal_symbol
    geom = mol.coord_geom

    # Build full ligand-graph neighbor table (no metal, includes H) for kappa
    all_neighbors = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)

    out: List[KeyValue] = []

    for site in mol.coord_sites:
        try:
            subclass = _site_subclass(site, mol)
            anion_bucket = anionic_co_donor_bucket(mol, exclude_rep_idx=_site_representative_idx(site))

            if site.is_haptic:
                # ── haptic tilt angle ────────────────────────────────────────
                face_idxs = [
                    idx
                    for idx in site.donor_indices
                    if idx != metal_idx
                    and atom_map.get(idx) is not None
                    and atom_map[idx].symbol != "H"
                ]
                if len(face_idxs) < 3:
                    continue  # need at least 3 atoms to define a plane

                face_pts = np.array([atom_map[idx].position for idx in face_idxs], dtype=float)
                centroid = face_pts.mean(axis=0)

                # Best-fit ring-plane normal via SVD
                centered = face_pts - centroid
                _, _, Vt = np.linalg.svd(centered)
                normal = Vt[-1]  # smallest singular vector = plane normal

                # Metal–centroid unit vector
                mc = centroid - metal_pos
                mc_len = np.linalg.norm(mc)
                if mc_len < 1e-8:
                    continue
                u_mc = mc / mc_len

                # Orient normal toward metal
                if np.dot(normal, u_mc) < 0:
                    normal = -normal

                # Tilt = angle between normal and metal-centroid direction
                cos_t = float(np.clip(np.dot(normal, u_mc), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cos_t)))
                out.append(((subclass, metal, geom, anion_bucket), angle))

            else:
                # ── kappa approach angle ─────────────────────────────────────
                donor_idx = site.donor_indices[0]
                nbs = all_neighbors.get(donor_idx, [])
                if not nbs:
                    continue  # k = 0: skip

                donor_pos = _vec(mol, donor_idx)

                sub_positions = np.array([atom_map[n].position for n in nbs], dtype=float)
                centroid = sub_positions.mean(axis=0)

                v_M = metal_pos - donor_pos
                v_C = centroid - donor_pos

                len_M = np.linalg.norm(v_M)
                len_C = np.linalg.norm(v_C)
                if len_M < 1e-8 or len_C < 1e-8:
                    continue

                cos_t = float(np.clip(np.dot(v_M / len_M, v_C / len_C), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cos_t)))
                out.append(((subclass, metal, geom, anion_bucket), angle))

        except Exception as exc:
            log.debug("Q3 skip site %s: %s", site, exc)

    return out


def extract_key_values(mol: Any) -> List[Tuple[tuple, float]]:
    """Extract (key_tuple, angle_deg) pairs for Q3 approach angles from a TMComplex."""
    return q3_approach_angles(mol)


def extract_rows(mol: Any) -> List[Dict[str, Any]]:
    """Extract tabular observation records from a TMComplex."""
    return observation_rows(mol, extract_key_values(mol))


# ---------------------------------------------------------------------------
# Ladder & Ancestors
# ---------------------------------------------------------------------------

def ladder_candidates(key: tuple, variant: str = DEFAULT_VARIANT) -> List[tuple]:
    """Ordered candidate keys for a Q3 query."""
    return L.q3_candidates(key, variant=variant)


def ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    """(level_int, ancestor_key) rows an L0 Q3 observation feeds."""
    return L.q3_ancestor_rows(key)


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
    """Fit a GMM on cos(angle) for bounded [0, 180] deg approach angles."""
    val_arr = np.asarray(values_deg, dtype=float)
    wt_arr = np.asarray(weights, dtype=float) if weights is not None else None
    return FT.fit_gmm_cosangle(
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
    Fit independent GMM-cos entries across grouped observations.

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
    """Fit all Q3 entries from expanded observations parquet in expanded_dir."""
    expanded_dir = Path(expanded_dir)
    obs = FT._load_observations(expanded_dir, QUANTITY, n_min=n_min)
    results = fit_from_observations(obs, n_min=n_min)
    return write_results(results, out_path)
