"""Focused unit tests for the end-to-end C2 NCI surface fitter and runtime evaluator."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools/jdev/budget"))

import nci_c2_surface as c2
import nci_surface_ownership_audit as audit
from jacobik.stats import nci


def _sample_raw_hist():
    raw_hist = {}
    for typ, key_label, r_peak, r_sd, a_peak, a_sd in [
        ("hb", "hb|O|O", 1.95, 0.25, 165.0, 22.0),
        ("pp", "pp|*", 3.65, 0.35, 15.0, 18.0),
        ("chp", "chp|*", 2.85, 0.35, 12.0, 18.0),
    ]:
        r0, dr, nr, a0, da, na = nci._grid(typ)
        r = r0 + (np.arange(nr) + 0.5) * dr
        a = a0 + (np.arange(na) + 0.5) * da
        s = 1.2 * np.exp(-0.5 * ((r[:, None] - r_peak) / r_sd) ** 2) * np.exp(-0.5 * ((a[None, :] - a_peak) / a_sd) ** 2)
        vol = nci.bin_volumes(typ)
        raw_hist[key_label] = 50.0 * vol * np.exp(s)
    return raw_hist


def test_c2_invariants_all_types():
    raw_hist = _sample_raw_hist()
    res = audit.verify_c2_invariants(raw_hist)
    for typ in ("hb", "pp", "chp"):
        info = res[typ]
        assert info["untapered_bin_center_max_error"] < 1e-10
        assert info["radial_knot_d1_jump_c2"] < 1e-5
        assert info["radial_knot_d2_jump_c2"] < 1e-3
        assert info["radial_knot_d1_jump_bilinear"] > 0.01
        assert info["radial_knot_d1_jump_c2"] < 1e-3 * info["radial_knot_d1_jump_bilinear"]
        assert info["angular_endpoint_slope_abs_max"] < 1e-12
        assert info["cartesian_pullback_hessian_max_error"] < 1e-8
        assert info["ring_normal_sign_flip_hessian_max_error"] < 1e-11


def test_empirical_fitting_objective_preserves_broad_and_shrinks_sharp():
    raw_hist = _sample_raw_hist()
    h = raw_hist["hb|O|O"].copy()
    _, dr, _, _, _, _ = nci._grid("hb")
    # Add an artificially narrow peak around r_idx=10, a_idx=30
    h[9:12, 29:32] *= 15.0
    h[10, 30] *= 5.0
    F_unreg = c2.fit_empirical_surface("hb", h=h, lam_rough=0.0, lam_sharp=0.0)
    F_sharp_reg = c2.fit_empirical_surface("hb", h=h, lam_rough=0.05, lam_sharp=2.0, gamma_compliance=1.0, ess_ratio=0.5)
    # Maximum local attractive radial curvature is reduced by the one-sided sharpness regularizer
    max_curv_unreg = float(np.max(-(F_unreg[2:, :] - 2.0 * F_unreg[1:-1, :] + F_unreg[:-2, :]) / (dr * dr)))
    max_curv_reg = float(np.max(-(F_sharp_reg[2:, :] - 2.0 * F_sharp_reg[1:-1, :] + F_sharp_reg[:-2, :]) / (dr * dr)))
    assert max_curv_reg < max_curv_unreg
    # Broad well region far from the spike remains well-preserved
    assert abs(float(np.mean(F_sharp_reg[15:25, 28:35]) - np.mean(F_unreg[15:25, 28:35]))) < 0.25
