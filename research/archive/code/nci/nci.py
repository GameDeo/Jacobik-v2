"""
src/jacobik/fit/nci.py
==============
Quantity NCI: Calibrated C2 Noncovalent Interaction spline tables (hb, pp, chp).

Key interaction types and models:
- 'hb': Hydrogen bonding (donor-acceptor distance r, D-H...A angle alpha)
- 'pp': Pi-pi aromatic stacking (centroid-centroid distance r, normal-normal angle alpha)
- 'chp': CH-pi interactions (H-centroid distance r, C-H...centroid angle alpha)

Surface fitting:
- Empirical log-enrichment surface on (r, alpha) grid with taper envelope.
- C2 continuous bicubic Hermite spline parameterization.
- Calibrated one-sided thermal-compliance sharpness regularizer ensuring physical curvatures.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from jacobik.fit import nci_model as nci

log = logging.getLogger(__name__)

QUANTITY: str = "NCI"
N_MIN: int = 1000  # production NCI-table admission threshold
S_CLIP: float = nci.S_CLIP
TYPES: Tuple[str, ...] = ("hb", "pp", "chp")
SPEC: Dict[str, dict] = nci.SPEC
THERMAL_COMPLIANCE_SD: Dict[str, Tuple[float, float]] = nci.THERMAL_COMPLIANCE_SD
GAMMA_COMPLIANCE: Dict[str, float] = nci.GAMMA_COMPLIANCE


def grid(t: str) -> Tuple[float, float, int, float, float, int]:
    """Return (r_lo, dr, nr, a_lo, da, na) for interaction type t ('hb', 'pp', 'chp')."""
    return nci._grid(t)


# ---------------------------------------------------------------------------
# Extraction & Histograms
# ---------------------------------------------------------------------------

HistAccumulator = nci.HistAccumulator


def create_nci_accumulator() -> nci.HistAccumulator:
    """Create a histogram accumulator for noncovalent interactions."""
    return nci.HistAccumulator()


def extract_records(mol: Any) -> List[Tuple]:
    """Extract candidate NCI interaction geometry records from a TMComplex."""
    return nci.records(mol)


def load_hist(paths: Iterable[Union[str, Path]]) -> Dict[str, np.ndarray]:
    """Load and sum NCI histogram tables from one or more .npz files."""
    return nci.load_hist(paths)


# ---------------------------------------------------------------------------
# Model Fitting & Surface Calibration
# ---------------------------------------------------------------------------

def fit_empirical_surface(
    t: str,
    h: Optional[np.ndarray] = None,
    s_existing: Optional[np.ndarray] = None,
    lam_rough: float = 0.05,
    lam_sharp: float = 2.0,
    gamma_compliance: Optional[float] = None,
    ess_ratio: float = 1.0,
) -> np.ndarray:
    """
    Fit regularized empirical log-enrichment surface F (nr, na) on bin centers.
    """
    gamma = gamma_compliance if gamma_compliance is not None else GAMMA_COMPLIANCE.get(t, 1.0)
    return nci.fit_empirical_surface(
        t,
        h=h,
        s_existing=s_existing,
        lam_rough=lam_rough,
        lam_sharp=lam_sharp,
        gamma_compliance=gamma,
        ess_ratio=ess_ratio,
    )


def build_c2_surface_params(
    t: str,
    F_grid: np.ndarray,
    s_clip: float = S_CLIP,
) -> Dict[str, Any]:
    """Construct bicubic Hermite C2 spline parameters from regularized F grid."""
    return nci.build_c2_surface_params(t, F_grid, s_clip=s_clip)


def fit_table(
    t: str,
    h: np.ndarray,
    regularize: bool = True,
) -> Dict[str, Any]:
    """Fit a single NCI surface table (empirical s-grid and regularized f-grid)."""
    return nci.fit_table(t, h, regularize=regularize)


def fit_tables(
    hist: Dict[str, np.ndarray],
    n_min: int = N_MIN,
    regularize: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit calibrated NCI surface tables for all labels in hist with sum >= n_min.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for lab, h in hist.items():
        if h.sum() < n_min:
            continue
        key = tuple(lab.split("|"))
        out[f"{QUANTITY}:{key!r}"] = fit_table(key[0], h, regularize=regularize)
    return out


def fit_from_observations(
    hist: Dict[str, np.ndarray],
    n_min: int = N_MIN,
    regularize: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Fit all NCI database entries from loaded histogram arrays."""
    return fit_tables(hist, n_min=n_min, regularize=regularize)


def fit_all(
    hist_paths: Sequence[Union[str, Path]],
    out_path: Optional[Union[str, Path]] = None,
    n_min: int = N_MIN,
    regularize: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Fit all NCI entries from histogram npz files."""
    hist = load_hist(hist_paths)
    results = fit_from_observations(hist, n_min=n_min, regularize=regularize)
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as fh:
            json.dump(results, fh, indent=1)
    return results
