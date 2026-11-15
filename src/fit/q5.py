"""
src/fit/q5.py
=============
Quantity 5: Ligand torsion distributions.

Key hierarchy and model:
- Key shape (L0): (type_j, type_k, type_i, type_l, ring_ctx)
- Fallback ladder: jacobik.stats.ladders.q5_candidates
- Fit model: von Mises Mixture merged to folded wells (wells_folded)
- Extraction folds symmetric rotors over equivalent flank choices.
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
from jacobik.stats.keys import (
    KeyValue,
    _build_ligand_graph,
    _build_type_map,
    _chelate_ring_bonds,
    _dihedral_deg,
    _vec,
    fold_shifts_deg,
    q5_context,
    q5_flanks,
    q5_symmetry_fn,
)
log = logging.getLogger(__name__)

log = logging.getLogger(__name__)

QUANTITY: str = "Q5"
MODEL_NAME: str = "wells_folded"
K_MAX: int = 6
N_MIN: int = 30  # legacy fitter N_MIN


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

#: Fold Q5 observations by rotor symmetry at extraction (2026-10-01, A6; claude/q5-investigation-2026-09-28.md §6).
Q5_FOLD_DATA: bool = True


def q5_torsions(mol: TMComplex) -> List[KeyValue]:
    """
    Produce (key, dihedral_deg) pairs for Q5 (rotatable bond torsion).
    """
    atom_map = {a.idx: a for a in mol.atoms}
    type_map = _build_type_map(mol)

    # Chelate ring bond set — built once for the whole complex.
    G_lig = _build_ligand_graph(mol)
    chelate_bonds = _chelate_ring_bonds(mol, G_lig)

    # Heavy-atom neighbors (no metal, no H)
    from jacobik.topology.graph import get_heavy_ligand_neighbors

    heavy_neighbors = get_heavy_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    from jacobik.topology.graph import get_ligand_neighbors

    all_nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)

    sym_of = q5_symmetry_fn(mol, all_nb, type_map) if Q5_FOLD_DATA else None

    seen_bonds: Set[Tuple[int, int]] = set()
    out: List[KeyValue] = []

    for lig in mol.ligands:
        for chain in lig.pivot_chains:
            for joint in chain.joints:
                j_idx = joint.atom_i  # donor-proximal end
                k_idx = joint.atom_j  # tail end
                bond_key = (min(j_idx, k_idx), max(j_idx, k_idx))
                if bond_key in seen_bonds:
                    continue
                seen_bonds.add(bond_key)

                try:
                    # Flanking atoms: heaviest heavy neighbor of j (not k) and
                    # heaviest heavy neighbor of k (not j)
                    i_idx, l_idx = q5_flanks(
                        j_idx, k_idx, heavy_neighbors, all_nb, atom_map, type_map
                    )
                    if i_idx is None:
                        continue  # terminal bond — no flanking atom on one side

                    # Dihedral i–j–k–l
                    p_i = _vec(mol, i_idx)
                    p_j = _vec(mol, j_idx)
                    p_k = _vec(mol, k_idx)
                    p_l = _vec(mol, l_idx)

                    dihedral = _dihedral_deg(p_i, p_j, p_k, p_l)
                    if dihedral is None:
                        continue

                    sym_i = atom_map[i_idx].symbol if i_idx in atom_map else "?"
                    sym_j = atom_map[j_idx].symbol if j_idx in atom_map else "?"
                    sym_k = atom_map[k_idx].symbol if k_idx in atom_map else "?"
                    sym_l = atom_map[l_idx].symbol if l_idx in atom_map else "?"
                    type_i = type_map.get(i_idx, sym_i)
                    type_j = type_map.get(j_idx, sym_j)
                    type_k = type_map.get(k_idx, sym_k)
                    type_l = type_map.get(l_idx, sym_l)

                    ring_ctx = "chelate" if bond_key in chelate_bonds else "free"
                    ctx = q5_context(ring_ctx, j_idx, k_idx, heavy_neighbors, type_map)
                    key = (type_j, type_k, type_i, type_l, ctx)
                    if sym_of is None:
                        out.append((key, dihedral))
                    else:
                        # symmetry folding of the DATA: every equivalent flank choice of a symmetric top is one copy
                        # (the fitter weights a complex's rows 1/m, so the copies share one complex's evidence)
                        for sh in fold_shifts_deg(*sym_of(j_idx, k_idx)):
                            out.append((key, (dihedral + sh + 180.0) % 360.0 - 180.0))

                except Exception as exc:
                    log.debug("Q5 skip joint %s: %s", joint, exc)

    return out


def extract_key_values(mol: Any) -> List[Tuple[tuple, float]]:
    """Extract (key_tuple, dihedral_deg) pairs for Q5 torsions from a TMComplex."""
    return q5_torsions(mol)


def extract_rows(mol: Any) -> List[Dict[str, Any]]:
    """Extract tabular observation records from a TMComplex."""
    return observation_rows(mol, extract_key_values(mol))


# ---------------------------------------------------------------------------
# Ladder & Ancestors
# ---------------------------------------------------------------------------

def ladder_candidates(key: tuple) -> List[tuple]:
    """Ordered candidate keys for a Q5 query."""
    return L.q5_candidates(key)


def ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    """(level_int, ancestor_key) rows an L0 Q5 observation feeds."""
    return L.q5_ancestor_rows(key)


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

def fit_entry(
    values_deg: np.ndarray | Sequence[float],
    weights: Optional[np.ndarray | Sequence[float]] = None,
    k_max: int = K_MAX,
    model: str = MODEL_NAME,
) -> Optional[Dict[str, Any]]:
    """
    Fit von Mises Mixture merged to folded wells for dihedral torsions [0, 360) deg.
    """
    val_arr = np.asarray(values_deg, dtype=float)
    wt_arr = np.asarray(weights, dtype=float) if weights is not None else None
    fit = FT.fit_vmm_wells(val_arr, weights=wt_arr, k_max=k_max)
    if fit is not None:
        fit["model"] = model
    return fit


def fit_from_observations(
    obs: Union[Dict[str, Tuple[np.ndarray, np.ndarray]], pd.DataFrame],
    n_min: int = N_MIN,
    k_max: int = K_MAX,
    model: str = MODEL_NAME,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit independent VMM folded-well entries across grouped observations.

    obs can be:
    - dict: {key_str: (values_array, weights_array)}
    - DataFrame: columns ['key', 'value'] (and optional 'weight')
    """
    return fit_independent_groups(QUANTITY, obs, fit_entry, n_min, k_max=k_max, model=model)


def fit_all(
    expanded_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    n_min: int = N_MIN,
    model: str = MODEL_NAME,
) -> Dict[str, Dict[str, Any]]:
    """Fit all Q5 entries from expanded observations parquet in expanded_dir."""
    expanded_dir = Path(expanded_dir)
    obs = FT._load_observations(expanded_dir, QUANTITY, n_min=n_min)
    results = fit_from_observations(obs, n_min=n_min, model=model)
    return write_results(results, out_path)
