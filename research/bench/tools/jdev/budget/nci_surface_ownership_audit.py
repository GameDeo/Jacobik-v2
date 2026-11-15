"""Phase 1 (Steps 1 & 4): Audit current NCI table shapes, bilinear derivative jumps,
Q6 polar-H well ownership, and evaluate single-factor ablations A0 (baseline),
A1 (end-to-end C2 pipeline only), A2 (qualified sharpness regularizer only),
and C1 (Q6 covered-pair ownership only) on physical probes and calibration data.

No source/DB/default changes. Writes versioned report under
data/refit/reports/n2_surface_ownership_audit_2026-10-18/.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, stats_from_db
from pools import setup
from nci_targeted_rescore import CANDIDATE
from nci_c2_surface import (
    THERMAL_COMPLIANCE_SD,
    angle_deg_to_z_torch,
    build_c2_surface_params,
    evaluate_c2_from_rz_torch,
    evaluate_c2_runtime,
    features_cos_torch,
    fit_empirical_surface,
)

OUT = REPO / "data/refit/reports/n2_surface_ownership_audit_2026-10-18"
HIST_DIR = REPO / "data/refit/nci/2026-10-18_bound_fix"
PROBE_DIR = REPO / "data/refit/nci/2026-10-18_energy_calibration_v2"
TRAIN_V2_DIR = REPO / "data/refit/nci/2026-10-18_training_calibration_v2"
KT = 2.479


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_c2_invariants(raw_hist: Dict[str, np.ndarray]) -> dict:
    """Verify mathematical C2 continuity across radial knots, taper cutoffs, angular endpoints
    (0, 90, 180 deg), ring-normal sign flips, and nonlinear Cartesian pullback Hessians."""
    from research.archive.code.interactions import nci

    results = {}
    for t, key_label in [("hb", "hb|O|O"), ("pp", "pp|*"), ("chp", "chp|*")]:
        h = raw_hist[key_label]
        F = fit_empirical_surface(t, h=h, lam_rough=0.0, lam_sharp=0.0)
        params = build_c2_surface_params(t, F)
        C = torch.tensor(params["C"][None], dtype=torch.float64)
        M = torch.tensor(params["M"][None], dtype=torch.float64)
        row = torch.zeros(1, dtype=torch.long)
        r_lo, dr, nr, a_lo, da, na = params["grid"]
        win_lo, win_hi = params["win"]

        # 1. Check exact interpolation at bin centers inside un-tapered window (r <= win_lo)
        rc = torch.tensor(r_lo + dr * (np.arange(nr) + 0.5), dtype=torch.float64)
        ac = torch.tensor(a_lo + da * (np.arange(na) + 0.5), dtype=torch.float64)
        rr, aa = torch.meshgrid(rc, ac, indexing="ij")
        zz = angle_deg_to_z_torch(t, aa)
        s_c2 = evaluate_c2_from_rz_torch(
            C, M, row.expand(nr * na), rr.reshape(1, -1), zz.reshape(1, -1), params["grid"], params["win"]
        ).reshape(nr, na).numpy()
        untapered_mask = rc.numpy() <= win_lo
        target_untapered = nci.S_CLIP * np.tanh(F[untapered_mask] / nci.S_CLIP)
        interp_max_err = float(np.max(np.abs(s_c2[untapered_mask] - target_untapered)))
        assert interp_max_err < 1e-10, (t, interp_max_err)

        # 2. Check radial 1st and 2nd derivative continuity across ALL radial knots + win_lo + win_hi
        test_angles = torch.tensor([15.0, 45.0, 85.0] if t != "hb" else [15.0, 90.0, 175.0], dtype=torch.float64)
        test_z = angle_deg_to_z_torch(t, test_angles)
        knots = np.unique(np.concatenate([rc.numpy(), [r_lo, win_lo, win_hi, r_lo + nr * dr]]))
        eps = 1e-8
        max_d1_jump_c2 = 0.0
        max_d2_jump_c2 = 0.0
        max_d1_jump_bilin = 0.0

        tab_bilin = torch.tensor(np.array(nci.fit_table(t, h)["s"])[None], dtype=torch.float64)
        for rk in knots:
            for sign in (-1.0, 1.0):
                r_pt = torch.full((1, len(test_z)), rk + sign * eps, dtype=torch.float64, requires_grad=True)
                val = evaluate_c2_from_rz_torch(C, M, torch.zeros(len(test_z), dtype=torch.long), r_pt, test_z[None], params["grid"], params["win"])
                g1 = torch.autograd.grad(val.sum(), r_pt, create_graph=True)[0]
                g2 = torch.autograd.grad(g1.sum(), r_pt)[0]
                if sign < 0:
                    g1_minus, g2_minus = g1.detach().numpy(), g2.detach().numpy()
                else:
                    g1_plus, g2_plus = g1.detach().numpy(), g2.detach().numpy()
            max_d1_jump_c2 = max(max_d1_jump_c2, float(np.max(np.abs(g1_plus - g1_minus))))
            max_d2_jump_c2 = max(max_d2_jump_c2, float(np.max(np.abs(g2_plus - g2_minus))))

            # Compare with _bilinear 1st derivative jump
            r_m = torch.full((1, len(test_angles)), rk - 1e-6, dtype=torch.float64, requires_grad=True)
            r_p = torch.full((1, len(test_angles)), rk + 1e-6, dtype=torch.float64, requires_grad=True)
            bm = nci._bilinear(tab_bilin, torch.zeros(len(test_angles), dtype=torch.long), r_m, test_angles[None], params["grid"])
            bp = nci._bilinear(tab_bilin, torch.zeros(len(test_angles), dtype=torch.long), r_p, test_angles[None], params["grid"])
            gm = torch.autograd.grad(bm.sum(), r_m)[0].detach().numpy()
            gp = torch.autograd.grad(bp.sum(), r_p)[0].detach().numpy()
            max_d1_jump_bilin = max(max_d1_jump_bilin, float(np.max(np.abs(gp - gm))))

        assert max_d1_jump_c2 < 1e-5, (t, max_d1_jump_c2)
        assert max_d2_jump_c2 < 1e-3, (t, max_d2_jump_c2)

        # 3. Check angular boundary ds/dtheta == 0 at 0 and 180 (hb) or 0 and 90 (pp/chp)
        end_angles = torch.tensor([0.0, 180.0] if t == "hb" else [0.0, 90.0], dtype=torch.float64, requires_grad=True)
        r_mid = torch.full((1, 2), float(0.5 * (r_lo + win_lo)), dtype=torch.float64)
        z_end = angle_deg_to_z_torch(t, end_angles)
        val_end = evaluate_c2_from_rz_torch(C, M, torch.zeros(2, dtype=torch.long), r_mid, z_end[None], params["grid"], params["win"])
        d_theta_end = torch.autograd.grad(val_end.sum(), end_angles)[0].detach().numpy()
        max_end_slope = float(np.max(np.abs(d_theta_end)))
        assert max_end_slope < 1e-12, (t, max_end_slope)

        # 4. Check Cartesian differentiability across folded 90 deg / linear 180 deg & pullback Hessian identity
        if t == "hb":
            # Linear D-H...A (180 deg): D=(0,0,0), H=(1,0,0), A=(3,0,0)
            X0 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
            idx = {"d": torch.tensor([0]), "h": torch.tensor([1]), "a": torch.tensor([2])}
            rt_c2 = {t: dict(C=C, M=M, row=row, idx=idx, grid=params["grid"], win=params["win"], s_clip=params["s_clip"])}
            # Pullback Hessian check at off-axis geometry
            X_gen = torch.tensor([[[0.0, 0.1, 0.0], [1.0, 0.0, 0.0], [2.9, 0.3, -0.1]]], dtype=torch.float64)
            H_auto = torch.autograd.functional.hessian(lambda x: evaluate_c2_runtime(rt_c2, x).sum(), X_gen).reshape(9, 9).numpy()

            def feat_fn(x):
                r_, z_ = features_cos_torch(t, x, idx)
                return torch.stack([r_[0, 0], z_[0, 0]])

            J_xi = torch.autograd.functional.jacobian(feat_fn, X_gen).reshape(2, 9).numpy()
            H_xi_0 = torch.autograd.functional.hessian(lambda x: feat_fn(x)[0], X_gen).reshape(9, 9).numpy()
            H_xi_1 = torch.autograd.functional.hessian(lambda x: feat_fn(x)[1], X_gen).reshape(9, 9).numpy()
            xi_val = feat_fn(X_gen).detach().requires_grad_(True)
            s_xi = lambda xi: evaluate_c2_from_rz_torch(C, M, row, xi[0:1][None], xi[1:2][None], params["grid"], params["win"]).sum()
            grad_s = torch.autograd.grad(s_xi(xi_val), xi_val)[0].detach().numpy()
            hess_s = torch.autograd.functional.hessian(s_xi, xi_val).reshape(2, 2).numpy()
            H_pullback = J_xi.T @ hess_s @ J_xi + grad_s[0] * H_xi_0 + grad_s[1] * H_xi_1
            pullback_err = float(np.max(np.abs(H_auto - H_pullback)))
            assert pullback_err < 1e-8, pullback_err

            # Check finite Hessian at exact 180 deg linear geometry
            H_lin = torch.autograd.functional.hessian(lambda x: evaluate_c2_runtime(rt_c2, x).sum(), X0).reshape(9, 9).numpy()
            assert np.isfinite(H_lin).all()
            ring_sign_err = 0.0
        else:
            # Build 6-membered ring + H or second ring at exact 90 deg approach (where .abs() in nci.features kinks)
            angles_ring = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            ring1 = np.column_stack([1.4 * np.cos(angles_ring), 1.4 * np.sin(angles_ring), np.zeros(6)])
            if t == "chp":
                # H in the ring plane at (3.0, 0, 0) -> theta = 90 deg (v . n = 0)
                coords = np.vstack([ring1, [[3.0, 0.0, 0.0]]])
                X90 = torch.tensor(coords[None], dtype=torch.float64)
                idx_fwd = {"rings": torch.tensor([[0, 1, 2, 3, 4, 5]]), "ring": torch.tensor([0]), "h": torch.tensor([6])}
                idx_rev = {"rings": torch.tensor([[0, 5, 4, 3, 2, 1]]), "ring": torch.tensor([0]), "h": torch.tensor([6])}
            else:
                # Second ring perpendicular (normal along x-axis), displaced along z by 3.8 A
                ring2 = np.column_stack([np.zeros(6), 1.4 * np.cos(angles_ring), 3.8 + 1.4 * np.sin(angles_ring)])
                coords = np.vstack([ring1, ring2])
                X90 = torch.tensor(coords[None], dtype=torch.float64)
                idx_fwd = {"rings": torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]), "ri": torch.tensor([0]), "rj": torch.tensor([1])}
                idx_rev = {"rings": torch.tensor([[0, 5, 4, 3, 2, 1], [6, 7, 8, 9, 10, 11]]), "ri": torch.tensor([0]), "rj": torch.tensor([1])}
            rt_fwd = {t: dict(C=C, M=M, row=row, idx=idx_fwd, grid=params["grid"], win=params["win"], s_clip=params["s_clip"])}
            rt_rev = {t: dict(C=C, M=M, row=row, idx=idx_rev, grid=params["grid"], win=params["win"], s_clip=params["s_clip"])}
            H_90 = torch.autograd.functional.hessian(lambda x: evaluate_c2_runtime(rt_fwd, x).sum(), X90).numpy()
            assert np.isfinite(H_90).all()
            # Ring-normal sign reversal invariance check on value, gradient, and Hessian
            X_test = X90 + 0.15
            H_fwd = torch.autograd.functional.hessian(lambda x: evaluate_c2_runtime(rt_fwd, x).sum(), X_test).numpy()
            H_rev = torch.autograd.functional.hessian(lambda x: evaluate_c2_runtime(rt_rev, x).sum(), X_test).numpy()
            ring_sign_err = float(np.max(np.abs(H_fwd - H_rev)))
            assert ring_sign_err < 1e-11, (t, ring_sign_err)
            pullback_err = 0.0

        results[t] = dict(
            table_tested=key_label,
            untapered_bin_center_max_error=interp_max_err,
            radial_knot_d1_jump_c2=max_d1_jump_c2,
            radial_knot_d2_jump_c2=max_d2_jump_c2,
            radial_knot_d1_jump_bilinear=max_d1_jump_bilin,
            angular_endpoint_slope_abs_max=max_end_slope,
            cartesian_pullback_hessian_max_error=pullback_err,
            ring_normal_sign_flip_hessian_max_error=ring_sign_err,
        )
    return results


def audit_tables_and_q6_wells(raw_hist: Dict[str, np.ndarray], stats) -> dict:
    """Audit all 13 candidate NCI tables alongside the replaced Q6 polar-H ('Hp') wells."""
    from research.archive.code.interactions import nci, q6pmf

    nci_tables = stats.nci_tables()
    q6_tables = stats.q6_pmf_tables()
    table_audit = {}

    for key, entry in sorted(nci_tables.items(), key=lambda kv: str(kv[0])):
        t = key[0]
        r_lo, dr, nr, a_lo, da, na = nci._grid(t)
        rc = r_lo + dr * (np.arange(nr) + 0.5)
        ac = a_lo + da * (np.arange(na) + 0.5)
        s = np.asarray(entry["s"], dtype=np.float64)
        peak_idx = np.unravel_index(int(np.argmax(s)), s.shape)
        min_idx = np.unravel_index(int(np.argmin(s)), s.shape)

        # Bilinear derivative jumps across interior radial and angular bin edges
        ds_dr = np.diff(s, axis=0) / dr
        radial_d1_jumps = np.abs(np.diff(ds_dr, axis=0))
        ds_da_rad = np.diff(s, axis=1) / math.radians(da)
        angular_d1_jumps = np.abs(np.diff(ds_da_rad, axis=1))

        # Finite-difference attractive curvatures (-d2s/dr^2 and -d2s/dtheta^2) on bin grid
        curv_r = -np.diff(s, n=2, axis=0) / (dr * dr)
        curv_a = -np.diff(s, n=2, axis=1) / (math.radians(da) ** 2)
        sd_r, sd_a_rad = THERMAL_COMPLIANCE_SD[t]
        comp_ceil_r = 1.0 / (sd_r ** 2)
        comp_ceil_a = 1.0 / (sd_a_rad ** 2)

        table_audit[str(key)] = dict(
            n_raw=int(entry["n"]),
            peak_r_A=float(rc[peak_idx[0]]),
            peak_angle_deg=float(ac[peak_idx[1]]),
            peak_score_nats=float(s[peak_idx]),
            peak_surrogate_energy_kj=float(-KT * s[peak_idx]),
            min_r_A=float(rc[min_idx[0]]),
            min_angle_deg=float(ac[min_idx[1]]),
            min_score_nats=float(s[min_idx]),
            clip_hit_bins=int(np.sum(np.abs(s) >= nci.S_CLIP - 1e-3)),
            max_bilinear_radial_d1_jump_nat_per_A=float(np.max(radial_d1_jumps)),
            max_bilinear_angular_d1_jump_nat_per_rad=float(np.max(angular_d1_jumps)),
            max_attractive_curv_r_nat_per_A2=float(np.max(curv_r)),
            compliance_ref_curv_r_nat_per_A2=float(comp_ceil_r),
            bins_exceeding_compliance_ref_r=int(np.sum(curv_r > comp_ceil_r)),
            max_attractive_curv_a_nat_per_rad2=float(np.max(curv_a)),
            compliance_ref_curv_a_nat_per_rad2=float(comp_ceil_a),
            bins_exceeding_compliance_ref_a=int(np.sum(curv_a > comp_ceil_a)),
        )

    # Compare Q6 polar-H ('Hp', acc, '5', 'lin'/'bent') attractive wells vs NCI hb tables
    q6_grid = np.round(np.arange(q6pmf.R_LO, q6pmf.R_HI + 1e-9, q6pmf.DR_TAB), 4)
    taper_q6 = q6pmf._taper(q6_grid, *q6pmf.HBOND_TAPER)
    q6_vs_nci_hb = {}
    for acc in sorted(nci.ACCEPTORS):
        k_pair = tuple(sorted(("Hp", acc)))
        tot_entry = q6_tables.get((k_pair[0], k_pair[1], "5", ""))
        lin_entry = q6_tables.get((k_pair[0], k_pair[1], "5", "lin"))
        bent_entry = q6_tables.get((k_pair[0], k_pair[1], "5", "bent"))
        if tot_entry is None or lin_entry is None or bent_entry is None:
            q6_vs_nci_hb[acc] = dict(has_q6_well=False)
            continue
        r0 = float(tot_entry["r0"])
        w_lin = np.asarray(lin_entry["w"], dtype=np.float64)
        w_bent = np.asarray(bent_entry["w"], dtype=np.float64)
        w0_lin = float(np.interp(r0, q6_grid, w_lin))
        w0_bent = float(np.interp(r0, q6_grid, w_bent))
        well_lin_kt = np.minimum(0.0, np.where(q6_grid < r0, w0_lin, w_lin)) * taper_q6
        well_bent_kt = np.minimum(0.0, np.where(q6_grid < r0, w0_bent, w_bent)) * taper_q6
        idx_min_lin = int(np.argmin(well_lin_kt))
        idx_min_bent = int(np.argmin(well_bent_kt))

        # Check matching NCI hb keys
        nci_matches = {}
        for d_el in ("O", "N", "*"):
            k_nci = ("hb", d_el, acc) if ("hb", d_el, acc) in nci_tables else (("hb", "*", acc) if ("hb", "*", acc) in nci_tables else ("hb", "*", "*"))
            s_tab = np.asarray(nci_tables[k_nci]["s"], dtype=np.float64)
            r_lo, dr, nr, a_lo, da, na = nci._grid("hb")
            rc = r_lo + dr * (np.arange(nr) + 0.5)
            ac = a_lo + da * (np.arange(na) + 0.5)
            # Linear region (angle >= 150 deg, indices where ac >= 150) and bent region (ac < 120)
            lin_cols = ac >= 150.0
            bent_cols = (ac >= 90.0) & (ac < 120.0)
            peak_lin_nats = float(np.max(s_tab[:, lin_cols]))
            peak_lin_r = float(rc[int(np.argmax(np.max(s_tab[:, lin_cols], axis=1)))])
            peak_bent_nats = float(np.max(s_tab[:, bent_cols]))
            nci_matches[f"D={d_el}"] = dict(
                resolved_key=str(k_nci),
                is_pooled_fallback=(k_nci == ("hb", "*", "*")),
                peak_lin_r_A=peak_lin_r,
                peak_lin_energy_kj=float(-KT * peak_lin_nats),
                peak_bent_energy_kj=float(-KT * peak_bent_nats),
            )

        q6_vs_nci_hb[acc] = dict(
            has_q6_well=True,
            q6_r0_A=r0,
            q6_lin_min_r_A=float(q6_grid[idx_min_lin]),
            q6_lin_min_energy_kj=float(KT * well_lin_kt[idx_min_lin]),
            q6_bent_min_r_A=float(q6_grid[idx_min_bent]),
            q6_bent_min_energy_kj=float(KT * well_bent_kt[idx_min_bent]),
            q6_plateau_below_r0_lin_kj=float(KT * min(0.0, w0_lin)),
            nci_replacements=nci_matches,
        )

    return dict(nci_tables=table_audit, q6_polar_h_vs_nci_hb=q6_vs_nci_hb)


def build_variant_c2_tables(
    raw_hist: Dict[str, np.ndarray],
    nci_tables: Dict[tuple, dict],
    lam_rough: float = 0.0,
    lam_sharp: float = 0.0,
    gamma_compliance: float = 1.0,
) -> Dict[tuple, dict]:
    """Build C2 surface parameter dictionaries for all 13 eligible NCI tables."""
    out = {}
    for key in nci_tables:
        lab = "|".join(key)
        t = key[0]
        F = fit_empirical_surface(
            t,
            h=raw_hist[lab],
            lam_rough=lam_rough,
            lam_sharp=lam_sharp,
            gamma_compliance=gamma_compliance,
        )
        out[key] = build_c2_surface_params(t, F)
    return out


def build_runtime_c2(mol, c2_tables: Dict[tuple, dict], device, types=None) -> dict:
    """Build runtime tensor dictionary for evaluate_c2_runtime."""
    from research.archive.code.interactions import nci

    atoms = list(mol.atoms)
    pos = {a.idx: i for i, a in enumerate(atoms)}
    cands = nci.candidates(mol)
    enabled = nci.TYPES if types is None else tuple(types)
    rt = {}
    for t in enabled:
        rows_C, rows_M, row_of, keep, cand_row = [], [], {}, [], []
        for k, c in enumerate(cands[t]):
            e = next(((key, c2_tables[key]) for key in c[-1] if key in c2_tables), None)
            if e is None:
                continue
            if e[0] not in row_of:
                row_of[e[0]] = len(rows_C)
                rows_C.append(e[1]["C"])
                rows_M.append(e[1]["M"])
            keep.append(k)
            cand_row.append(row_of[e[0]])
        if not keep:
            continue
        r_lo, dr, nr, a_lo, da, na = nci._grid(t)
        rt[t] = dict(
            C=torch.tensor(np.array(rows_C, dtype=np.float64), device=device),
            M=torch.tensor(np.array(rows_M, dtype=np.float64), device=device),
            row=torch.tensor(cand_row, dtype=torch.long, device=device),
            idx=nci.index_tensors(cands, pos, device, keep={**{x: [] for x in nci.TYPES}, t: keep})[t],
            grid=(r_lo, dr, nr, a_lo, da, na),
            win=nci.SPEC[t]["win"],
            s_clip=nci.S_CLIP,
            n=len(keep),
        )
    return rt


def evaluate_probes_ablations(raw_hist: Dict[str, np.ndarray], stats) -> dict:
    """Evaluate single-factor ablations A0, A1, and A2 (gamma=0.5, 1.0, 2.0) on the
    1,008 rigid physical probe geometries in 2026-10-18_energy_calibration_v2."""
    from rdkit import Chem
    from research.archive.code.interactions import nci
    from nci_calibration import MONOMERS, topology

    nci_tables = stats.nci_tables()
    variants = {
        "A1_c2_pipeline_only": build_variant_c2_tables(raw_hist, nci_tables, lam_rough=0.0, lam_sharp=0.0),
        "A2_sharp_reg_gamma0.5": build_variant_c2_tables(raw_hist, nci_tables, lam_rough=0.05, lam_sharp=1.0, gamma_compliance=0.5),
        "A2_sharp_reg_gamma1.0": build_variant_c2_tables(raw_hist, nci_tables, lam_rough=0.05, lam_sharp=1.0, gamma_compliance=1.0),
        "A2_sharp_reg_gamma2.0": build_variant_c2_tables(raw_hist, nci_tables, lam_rough=0.05, lam_sharp=1.0, gamma_compliance=2.0),
    }

    probe_files = sorted(PROBE_DIR.glob("hb_*.npz")) + sorted(PROBE_DIR.glob("pp_*.npz")) + sorted(PROBE_DIR.glob("chp_*.npz"))
    by_type = {t: {m: {"MACE64": [], "GFN2": []} for m in ["off", "A0_bilinear"] + list(variants.keys())} for t in nci.TYPES}
    per_probe = {}

    for f in probe_files:
        meta = json.loads(f.with_suffix(".json").read_text())
        kind = meta["type"]
        labels = meta["labels"]
        with np.load(f) as z:
            x = z["X"]
            old = z["q6_old"].astype(float)
            core = z["q6_core"].astype(float)
            s_a0 = z["score_" + kind].astype(float)
        rdA = Chem.AddHs(Chem.MolFromSmiles(MONOMERS[meta["a"]]))
        rdB = Chem.AddHs(Chem.MolFromSmiles(MONOMERS[meta["b"]]))
        mol, _ = topology(rdA, rdB)
        xt = torch.tensor(x, dtype=torch.float64)

        preds = {
            "off": old,
            "A0_bilinear": (core if kind == "hb" else old) - KT * s_a0,
        }
        for vname, c2_tabs in variants.items():
            rt_c2 = build_runtime_c2(mol, c2_tabs, "cpu", types=(kind,))
            s_c2 = evaluate_c2_runtime(rt_c2, xt).numpy() if rt_c2 else np.zeros(len(x))
            preds[vname] = (core if kind == "hb" else old) - KT * s_c2

        scans = sorted(set(l["scan"] for l in labels))
        per_probe[f.stem] = {}
        for evaluator, ext in [("MACE64", "mace64"), ("GFN2", "gfn2")]:
            energy = np.array(json.loads(f.with_suffix("." + ext + ".json").read_text())["interaction"], dtype=float)
            shifted_e = np.zeros_like(energy)
            for scan in scans:
                ids = [i for i, l in enumerate(labels) if l["scan"] == scan]
                far = next(i for i in ids if labels[i]["far"])
                shifted_e[ids] = energy[ids] - energy[far]
            good = (shifted_e < 50.0) & np.array([not l["far"] for l in labels])

            per_probe[f.stem][evaluator] = {}
            for mname, pred in preds.items():
                shifted_p = np.zeros_like(pred)
                radial_regret = []
                wrong_curv = []
                for scan in scans:
                    all_ids = [i for i, l in enumerate(labels) if l["scan"] == scan]
                    far = next(i for i in all_ids if labels[i]["far"])
                    shifted_p[all_ids] = pred[all_ids] - pred[far]
                    ids = sorted([i for i in all_ids if not labels[i]["far"]], key=lambda i: labels[i]["r"])
                    radial_regret.append(float(energy[ids[int(np.argmin(pred[ids]))]] - energy[ids].min()))
                    if len(ids) > 2:
                        ek = np.diff(energy[ids], n=2)
                        sk = np.diff(pred[ids], n=2)
                        valid = (ek > 1e-4) & (energy[ids][1:-1] < 50.0)
                        wrong_curv.extend((sk[valid] <= 0.0).tolist())
                rmse = float(np.sqrt(np.mean((shifted_p[good] - shifted_e[good]) ** 2)))
                reg = float(np.mean(radial_regret))
                wc = float(np.mean(wrong_curv)) if wrong_curv else 0.0
                rec = dict(rmse=rmse, radial_regret=reg, wrong_curvature_share=wc)
                per_probe[f.stem][evaluator][mname] = rec
                by_type[kind][mname][evaluator].append(rec)

    summary = {}
    for t in nci.TYPES:
        summary[t] = {}
        for mname, ev_dict in by_type[t].items():
            summary[t][mname] = {}
            for evaluator, recs in ev_dict.items():
                summary[t][mname][evaluator] = dict(
                    mean_rmse=float(np.mean([r["rmse"] for r in recs])),
                    mean_radial_regret=float(np.mean([r["radial_regret"] for r in recs])),
                    mean_wrong_curvature_share=float(np.mean([r["wrong_curvature_share"] for r in recs])),
                )
    return dict(summary_by_type=summary, probes=per_probe)


def evaluate_v2_calibration_ablations(raw_hist: Dict[str, np.ndarray], stats) -> dict:
    """Evaluate single-factor ablations A0, A1, A2, and C1 (ownership-only) strictly on the
    16-complex-per-type CALIBRATION split of 2026-10-18_training_calibration_v2, including
    two-step (0.02 vs 0.04 A) surrogate curvature stability."""
    from research.archive.code.interactions import nci, q6pmf
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors
    from nci_training_calibration import stream
    from nci_training_analysis import metrics as base_metrics, paired_ci

    selection = json.loads((TRAIN_V2_DIR / "selection.json").read_text())
    cal_items = [r for r in selection["selected"] if r["split"] == "calibration"]
    assert len(cal_items) == 48

    nci_tables = stats.nci_tables()
    variants = {
        "A1_c2_pipeline_only": build_variant_c2_tables(raw_hist, nci_tables, lam_rough=0.0, lam_sharp=0.0),
        "A2_sharp_reg_gamma0.5": build_variant_c2_tables(raw_hist, nci_tables, lam_rough=0.05, lam_sharp=1.0, gamma_compliance=0.5),
        "A2_sharp_reg_gamma1.0": build_variant_c2_tables(raw_hist, nci_tables, lam_rough=0.05, lam_sharp=1.0, gamma_compliance=1.0),
        "A2_sharp_reg_gamma2.0": build_variant_c2_tables(raw_hist, nci_tables, lam_rough=0.05, lam_sharp=1.0, gamma_compliance=2.0),
    }

    # Stream the calibration molecules to build C2 runtimes and audit C1 Q6 ownership
    chosen = {(r["shard"], r["code"]): dict(code=r["code"], shard=r["shard"], dir=r["directory"]) for r in cal_items}
    mols = {}
    for row, mol, err in stream(chosen.values()):
        assert err is None and mol is not None, (row, err)
        for lig in mol.ligands:
            decompose_ligand(mol, lig)
        mols[row["code"]] = mol

    by_type_records = {t: [] for t in nci.TYPES}
    hb_ownership_audit = []

    for r in cal_items:
        t = r["type"]
        code = r["code"]
        mol = mols[code]
        npz_path = TRAIN_V2_DIR / f"{t}_{code}.npz"
        meta = json.loads(npz_path.with_suffix(".json").read_text())
        energy = np.array(json.loads(npz_path.with_suffix(".mace64.json").read_text())["relative"], dtype=float)
        with np.load(npz_path) as z:
            X = z["X"]
            old = -KT * z["off_total"].astype(float)
            base = old - KT * (z["on_Q6"].astype(float) - z["off_Q6"].astype(float)) if t == "hb" else old.copy()
            nci_a0 = -KT * z["NCI_" + t].astype(float)

        shift = lambda arr: arr - arr[0]
        xt = torch.tensor(X, dtype=torch.float64)
        case_entry = dict(
            request=r,
            meta=meta,
            energy=energy,
            old=shift(old),
            base=shift(base),
            A0_bilinear=shift(nci_a0),
        )

        for vname, c2_tabs in variants.items():
            rt_c2 = build_runtime_c2(mol, c2_tabs, "cpu", types=(t,))
            s_c2 = evaluate_c2_runtime(rt_c2, xt).numpy() if rt_c2 else np.zeros(len(X))
            case_entry[vname] = shift(-KT * s_c2)

        # For 'hb', also compute C1 (covered-only Q6 polar-H well removal)
        if t == "hb":
            ge = GeometryExtractor(mol, torch.device("cpu"))
            sc = Scorer(mol, ge, stats, torch.device("cpu"))
            qr = sc._q6_pmf
            ai, bi = ge._q6_a, ge._q6_b
            allc = nci.candidates(mol)
            # Pairs that actually resolve to an eligible NCI hb table
            covered_pairs = {
                frozenset((c[1], c[2]))
                for c in allc["hb"]
                if any(k in nci_tables for k in c[-1])
            }
            nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
            bound = set(donor_indices_of(mol))
            sym = {a.idx: a.symbol for a in mol.atoms}

            xf = torch.tensor(X, dtype=torch.float32)
            d = (xf[:, ai] - xf[:, bi]).norm(dim=-1)
            r0 = qr["r0"][qr["tot"]][None].expand_as(d)
            w = q6pmf._interp(qr["T"], qr["tot"], d, qr["r_lo"], qr["dr"], qr["G"])
            w0 = q6pmf._interp(qr["T"], qr["tot"], r0, qr["r_lo"], qr["dr"], qr["G"])
            if qr["hb_any"]:
                u, v = xf[:, qr["d"]] - xf[:, qr["h"]], xf[:, qr["acc"]] - xf[:, qr["h"]]
                cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-8)
                mix = torch.sigmoid((q6pmf.ANGLE_SPLIT_COS - cos) / 0.08)
                for dst, dist in [(w, d), (w0, r0)]:
                    wl = q6pmf._interp(qr["T"], qr["lin"], dist, qr["r_lo"], qr["dr"], qr["G"])
                    wb = q6pmf._interp(qr["T"], qr["bent"], dist, qr["r_lo"], qr["dr"], qr["G"])
                    dst.copy_(torch.where(qr["hb"][None], mix * wl + (1.0 - mix) * wb, dst))
            wells = torch.where(d < r0, w0, w).clamp(max=0.0) * q6pmf._taper_t(d, *q6pmf.HBOND_TAPER)
            wells *= (qr["hp"] & qr["has"] & ~qr["rep"])[None]

            cov_mask = np.zeros(len(ai), dtype=bool)
            cobound_count = 0
            other_uncov_count = 0
            for j, (ia, ib) in enumerate(zip(ai.tolist(), bi.tolist())):
                aa, bb = mol.atoms[ia].idx, mol.atoms[ib].idx
                if frozenset((aa, bb)) in covered_pairs:
                    cov_mask[j] = True
                elif wells[:, j].abs().max() > 1e-3:
                    h_at, acc_at = (aa, bb) if sym[aa] == "H" else (bb, aa)
                    parent = nb.get(h_at, [])
                    if parent and parent[0] in bound and acc_at in bound:
                        cobound_count += 1
                    else:
                        other_uncov_count += 1

            wells_np = wells.numpy() * KT  # negative kJ/mol
            removed_all_kj = -wells_np.sum(axis=-1)
            removed_cov_kj = -wells_np[:, cov_mask].sum(axis=-1)
            # Verify partition against (base - old) = -KT * (z['on_Q6'] - z['off_Q6'])
            part_err = float(np.max(np.abs(removed_all_kj - (base - old))))
            assert part_err < 1e-2, (code, part_err)

            base_c1 = old + removed_cov_kj
            case_entry["base_C1"] = shift(base_c1)
            hb_ownership_audit.append(
                dict(
                    code=code,
                    active_cobound_hp_pairs=cobound_count,
                    active_other_uncovered_hp_pairs=other_uncov_count,
                    max_uncovered_well_variation_kj=float(np.ptp(removed_all_kj - removed_cov_kj)),
                )
            )

        by_type_records[t].append(case_entry)

    def sur_step_stability(d, pred: np.ndarray) -> dict:
        """Measure two-step (0.02 vs 0.04 A) curvature stability of the surrogate on all modes
        and on physically stable modes."""
        e = d["energy"]
        steps = np.array(d["meta"]["steps"])
        n = len(steps)
        all_agree = []
        phys_stable_agree = []
        max_step_diff = 0.0
        for j in range(len(d["meta"]["modes"])):
            def deriv(y, h):
                a = 1 + j * n + int(np.flatnonzero(np.isclose(steps, -h))[0])
                b = 1 + j * n + int(np.flatnonzero(np.isclose(steps, h))[0])
                return (y[b] + y[a] - 2.0 * y[0]) / (h * h)

            ek04 = deriv(e, 0.04)
            ek08 = deriv(e, 0.08)
            pk02 = deriv(pred, 0.02)
            pk04 = deriv(pred, 0.04)
            diff = abs(pk04 - pk02)
            max_step_diff = max(max_step_diff, float(diff))
            agree = bool(diff < 0.25 * max(abs(pk04), 1.0))
            all_agree.append(agree)
            if ek04 > 0.1 and abs(ek04 - ek08) < 0.25 * max(abs(ek04), 1.0):
                phys_stable_agree.append(agree)
        return dict(
            surrogate_2step_stable_all=int(sum(all_agree)),
            surrogate_2step_total_all=len(all_agree),
            surrogate_2step_stable_on_phys=int(sum(phys_stable_agree)),
            surrogate_2step_total_on_phys=len(phys_stable_agree),
            max_curv_02_04_diff=max_step_diff,
        )

    summary = {}
    model_names = ["A0_bilinear", "A1_c2_pipeline_only", "A2_sharp_reg_gamma0.5", "A2_sharp_reg_gamma1.0", "A2_sharp_reg_gamma2.0"]
    for t in nci.TYPES:
        ds = by_type_records[t]
        t_models = list(model_names) + (["C1_ownership_covered_only"] if t == "hb" else [])
        summary[t] = {}

        # Baseline 'off' metrics
        off_mr = [base_metrics(d, d["old"]) for d in ds]
        off_st = [sur_step_stability(d, d["old"]) for d in ds]
        off_ratios = [v for m in off_mr for v in m["curvature_energy_over_surrogate"]]
        summary[t]["off"] = dict(
            mean_case_rmse=float(np.mean([m["rmse"] for m in off_mr])),
            mean_sampled_minimum_regret=float(np.mean([m["sampled_minimum_regret"] for m in off_mr])),
            wrong_sign_curvatures=int(sum(m["wrong_sign_curvatures"] for m in off_mr)),
            median_curvature_ratio=float(np.median(off_ratios)) if off_ratios else None,
            surrogate_2step_stable_all=int(sum(s["surrogate_2step_stable_all"] for s in off_st)),
            surrogate_2step_total_all=int(sum(s["surrogate_2step_total_all"] for s in off_st)),
            max_curv_02_04_diff=float(max(s["max_curv_02_04_diff"] for s in off_st)),
        )

        for mname in t_models:
            if mname == "C1_ownership_covered_only":
                base_key = "base_C1"
                nci_key = "A0_bilinear"
            else:
                base_key = "base"
                nci_key = mname

            num = sum(float(d[nci_key][1:] @ (d["energy"][1:] - d[base_key][1:])) / (len(d["energy"]) - 1) for d in ds)
            den = sum(float(d[nci_key][1:] @ d[nci_key][1:]) / (len(d["energy"]) - 1) for d in ds)
            coeff = max(0.0, num / den) if den > 1e-12 else 0.0

            summary[t][mname] = dict(fitted_coefficient=coeff)
            for scale_mode, w_scale in [("unit", 1.0), ("calibrated", coeff)]:
                mr = []
                st = []
                nci_only_st = []
                for d in ds:
                    pred = d[base_key] + w_scale * d[nci_key]
                    mr.append(base_metrics(d, pred))
                    st.append(sur_step_stability(d, pred))
                    nci_only_st.append(sur_step_stability(d, d[nci_key]))
                ratios = [v for m in mr for v in m["curvature_energy_over_surrogate"]]
                delta = [off_mr[i]["rmse"] - mr[i]["rmse"] for i in range(len(ds))]
                summary[t][mname][scale_mode] = dict(
                    mean_case_rmse=float(np.mean([m["rmse"] for m in mr])),
                    rmse_improvement_ci95=paired_ci(delta),
                    mean_sampled_minimum_regret=float(np.mean([m["sampled_minimum_regret"] for m in mr])),
                     gradient_correct=int(sum(m["gradient_correct"] for m in mr)),
                    gradient_tested=int(sum(m["gradient_tested"] for m in mr)),
                    wrong_sign_curvatures=int(sum(m["wrong_sign_curvatures"] for m in mr)),
                    median_curvature_ratio=float(np.median(ratios)) if ratios else None,
                    surrogate_2step_stable_all=int(sum(s["surrogate_2step_stable_all"] for s in st)),
                    surrogate_2step_total_all=int(sum(s["surrogate_2step_total_all"] for s in st)),
                    nci_term_2step_stable_all=int(sum(s["surrogate_2step_stable_all"] for s in nci_only_st)),
                    nci_term_max_curv_02_04_diff=float(max(s["max_curv_02_04_diff"] for s in nci_only_st)),
                )

    return dict(
        scope="Calibration split only (16 complexes/type) of 2026-10-18_training_calibration_v2. Validation split untouched.",
        calibration_summary=summary,
        hb_ownership_audit=hb_ownership_audit,
    )


def run():
    from research.archive.code.interactions import nci
    from tools.jdev.env import tree_hash

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    raw_hist = nci.load_hist(sorted(HIST_DIR.glob("ncihist_*.npz")))
    stats = stats_from_db(CANDIDATE)

    print("1/4 Verifying C2 mathematical invariants...", flush=True)
    invariants = verify_c2_invariants(raw_hist)
    print(json.dumps(invariants, indent=2), flush=True)

    print("2/4 Auditing 13 NCI tables and Q6 polar-H well replacement...", flush=True)
    table_audit = audit_tables_and_q6_wells(raw_hist, stats)

    print("3/4 Running single-factor ablations A0, A1, A2 on 1,008 physical probe geometries...", flush=True)
    probe_results = evaluate_probes_ablations(raw_hist, stats)
    print(json.dumps(probe_results["summary_by_type"], indent=2), flush=True)

    print("4/4 Running single-factor ablations A0, A1, A2, C1 on v2 calibration split...", flush=True)
    cal_results = evaluate_v2_calibration_ablations(raw_hist, stats)
    print(json.dumps(cal_results["calibration_summary"], indent=2), flush=True)

    report = dict(
        source_hash=tree_hash(),
        candidate_db_sha256=digest(CANDIDATE),
        c2_invariants=invariants,
        table_and_q6_audit=table_audit,
        probe_ablations=probe_results,
        v2_calibration_ablations=cal_results,
    )
    write_json(OUT / "audit_and_ablations.json", report)
    print("Wrote", OUT / "audit_and_ablations.json", flush=True)


if __name__ == "__main__":
    setup()
    run()
