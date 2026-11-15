"""Phase 2 (Steps 5, 6, 7): Separated single-factor descriptor ablations (C2, C3, C4, C5),
pre-validation model selection (D), one-shot prospective v3 validation evaluation,
and frozen-pool conformer re-ranking gate.

All fits and model-selection choices are made strictly on the v3 calibration split and
physical dimer probes using leave-validation-out histogram tables.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

_SRC = str(Path(__file__).resolve().parents[5] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, str(Path(__file__).parent))

from common import DB, REPO, pool_energy, pool_path, stats_from_db
from pools import setup, set_cases
from nci_targeted_rescore import CANDIDATE, COHORT
from nci_training_calibration import contact_indices, digest, stream, write
from nci_c2_surface import (
    angle_deg_to_z_torch,
    build_c2_surface_params,
    evaluate_c2_from_rz_torch,
    evaluate_c2_runtime,
    features_cos_torch,
    fit_empirical_surface,
    quintic_taper_torch,
)

V3_DIR = REPO / "data/refit/nci/2026-10-18_training_calibration_v3"
PROBE_DIR = REPO / "data/refit/nci/2026-10-18_energy_calibration_v2"
OUT = REPO / "data/refit/reports/n2_ablation_pilot_v3_2026-10-18"
KT = 2.479


def paired_ci(values, seed: int = 20261018):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def build_c2_tables_from_hist(raw_hist: dict, lam_rough: float = 0.0, lam_sharp: float = 0.0, gamma_by_type: dict = None) -> dict:
    from research.archive.code.interactions import nci

    gamma_by_type = gamma_by_type or {"hb": 1.0, "pp": 0.5, "chp": 0.5}
    out = {}
    for k_str, h in raw_hist.items():
        if float(h.sum()) < nci.N_MIN:
            continue
        key = tuple(k_str.split("|"))
        t = key[0]
        F = fit_empirical_surface(
            t,
            h=h,
            lam_rough=lam_rough,
            lam_sharp=lam_sharp,
            gamma_compliance=float(gamma_by_type.get(t, 1.0)),
        )
        out[key] = build_c2_surface_params(t, F)
    return out


def build_c2_runtime_for_mol(mol, c2_tables: dict, device="cpu", types=None):
    from research.archive.code.interactions import nci

    cands = nci.candidates(mol)
    out = {}
    active = nci.ACTIVE_TYPES if types is None else tuple(types)
    for t in nci.TYPES:
        if t not in active:
            continue
        cs = [c for c in cands[t] if any(k in c2_tables for k in c[-1])]
        if not cs:
            continue
        keys = []
        row = []
        for c in cs:
            k = next(k for k in c[-1] if k in c2_tables)
            if k not in keys:
                keys.append(k)
            row.append(keys.index(k))
        C = torch.tensor(np.stack([c2_tables[k]["C"] for k in keys]), dtype=torch.float32, device=device)
        M = torch.tensor(np.stack([c2_tables[k]["M"] for k in keys]), dtype=torch.float32, device=device)
        sample_rt = nci.build_runtime(mol, {k: {"s": np.zeros_like(c2_tables[k]["C"]).tolist()} for k in keys}, device, types=(t,))[t]
        out[t] = dict(
            C=C,
            M=M,
            row=torch.tensor(row, dtype=torch.long, device=device),
            idx=sample_rt["idx"],
            grid=c2_tables[keys[0]]["grid"],
            win=c2_tables[keys[0]]["win"],
            s_clip=c2_tables[keys[0]]["s_clip"],
            cs=cs,
            keys=keys,
        )
    return out


def eval_chp_variants_torch(mol, e_chp: dict, X: torch.Tensor, use_c3_dir: bool, use_c4_sibling: bool, tau: float = 0.15):
    """Evaluate C2 CH-pi score with optional C3 (smooth C2 C-H direction gate)
    and/or C4 (smooth C^infty per-carbon sibling log-sum-exp aggregation)."""
    from research.archive.code.interactions import nci
    from jacobik.topology.graph import get_ligand_neighbors

    r, z = features_cos_torch("chp", X, e_chp["idx"])
    s_raw = evaluate_c2_from_rz_torch(
        e_chp["C"], e_chp["M"], e_chp["row"], r, z, e_chp["grid"], e_chp["win"], e_chp["s_clip"]
    )  # (B, M_cand)

    nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    pos = {atom.idx: i for i, atom in enumerate(mol.atoms)}
    h_idx = e_chp["idx"]["h"]
    ring_idx = e_chp["idx"]["ring"]
    parent_idx = torch.tensor(
        [pos[nb[mol.atoms[int(h)].idx][0]] for h in h_idx], dtype=torch.long, device=X.device
    )

    if use_c3_dir:
        c_ring, _ = nci._ring_frame(X, e_chp["idx"]["rings"])
        centers = c_ring[:, ring_idx]
        H_pos = X[:, h_idx]
        C_pos = X[:, parent_idx]
        u = C_pos - H_pos
        v = centers - H_pos
        cos_dha = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-12)
        # C2 directional factor: [max(-cos_dha, 0)]^3 has value=0, d1=0, d2=0 at cos_dha = 0 (90 deg)
        gate = torch.clamp(-cos_dha, min=0.0) ** 3
        # Smoothly split positive and negative parts via C^infty softplus/softmin or exact sign-preserving C2 gate
        # Since gate is in [0, 1] and we want attractive (s > 0) gated by direction while repulsive (s < 0) stays intact:
        # Use smooth C^infty decomposition s_+ = 0.5*(s + sqrt(s^2 + eps^2) - eps), s_- = s - s_+
        eps = 1e-3
        s_pos = 0.5 * (s_raw + torch.sqrt(s_raw * s_raw + eps * eps) - eps)
        s_neg = s_raw - s_pos
        s_dir = s_neg + s_pos * gate
    else:
        s_dir = s_raw

    if not use_c4_sibling:
        return s_dir.sum(dim=-1)

    # C4: Smooth per-(carbon_parent, ring) aggregation via log-sum-exp over sibling hydrogens
    eps = 1e-3
    s_pos = 0.5 * (s_dir + torch.sqrt(s_dir * s_dir + eps * eps) - eps)
    s_neg = s_dir - s_pos
    p_np = parent_idx.cpu().numpy()
    r_np = ring_idx.cpu().numpy()
    groups = sorted(set(zip(p_np.tolist(), r_np.tolist())))
    total = X.new_zeros(X.shape[0])
    for p_id, r_id in groups:
        mask = (parent_idx == p_id) & (ring_idx == r_id)
        sp = s_pos[:, mask]  # (B, n_sib)
        sn = s_neg[:, mask]
        # Smooth sibling aggregation: tau * log(1 + sum(exp(sp / tau) - 1))
        # Equals sp when n_sib == 1, and smoothly saturates for multiple active siblings without hard-max kinks
        agg_pos = tau * torch.log1p(torch.expm1(sp / tau).sum(dim=-1))
        agg_neg = sn.sum(dim=-1)
        total = total + agg_pos + agg_neg
    return total


def eval_pp_c5_slip_torch(mol, e_pp: dict, X: torch.Tensor):
    """Evaluate C5 stacking variant decomposing centroid displacement into normal separation d_perp
    and parallel slip d_par, suppressing eclipsed pi-cloud repulsion (d_par ~ 0 at short d_perp)
    and co-bound haptic face artifacts while preserving C2 smoothness."""
    from research.archive.code.interactions import nci
    from jacobik.topology.graph import donor_indices_of

    c, n = nci._ring_frame(X, e_pp["idx"]["rings"])
    ci, cj = c[:, e_pp["idx"]["ri"]], c[:, e_pp["idx"]["rj"]]
    ni, nj = n[:, e_pp["idx"]["ri"]], n[:, e_pp["idx"]["rj"]]
    v = ci - cj
    r = v.norm(dim=-1)
    dot = (ni * nj).sum(-1)
    z = (2.0 * (dot * dot) - 1.0).clamp(-1.0, 1.0)

    s_base = evaluate_c2_from_rz_torch(
        e_pp["C"], e_pp["M"], e_pp["row"], r, z, e_pp["grid"], e_pp["win"], e_pp["s_clip"]
    )

    # Smooth C^infty normal and parallel slip distances
    proj_i = (v * ni).sum(-1)
    proj_j = (v * nj).sum(-1)
    d_perp2 = 0.5 * (proj_i * proj_i + proj_j * proj_j)
    d_par2 = torch.clamp(r * r - d_perp2, min=0.0)
    # Physical parallel-displaced benzene/aryl stacking favors slip d_par ~ 1.2 - 2.2 A
    # and d_perp ~ 3.3 - 3.7 A; eclipsed short-d_perp (< 3.5 A, d_par < 0.8 A) has Pauli repulsion
    slip_window = torch.exp(-0.5 * ((d_par2 - 2.5) / 3.5) ** 2)
    # Exclude co-bound haptic faces (both rings completely in donor_indices_of(mol))
    bound = set(donor_indices_of(mol))
    valid_mask = torch.tensor(
        [0.0 if (set(cand[0]) <= bound and set(cand[1]) <= bound) else 1.0 for cand in e_pp["cs"]],
        dtype=X.dtype,
        device=X.device,
    )[None, :]
    eps = 1e-3
    s_pos = 0.5 * (s_base + torch.sqrt(s_base * s_base + eps * eps) - eps)
    s_neg = s_base - s_pos
    return ((s_neg + s_pos * (0.65 + 0.35 * slip_window)) * valid_mask).sum(dim=-1)


def compute_q6_unsupported_halogen_preservation(mol, ge, stats, X: torch.Tensor) -> np.ndarray:
    """Under C2, when hb is enabled, Q6 removes polar-H ('Hp') wells ONLY for acceptors
    that have a dedicated supported NCI acceptor table ('O', 'N', 'Cl', 'S' with N >= 1000).
    Unsupported halogen acceptors ('F', 'Br', 'I') keep their Q6 'Hp' wells instead of
    losing them to the 1.575 A pooled ('hb', '*', '*') fallback."""
    from jacobik.ik.scorer import Scorer
    from research.archive.code.interactions import q6pmf

    supported_hb_acceptors = {"O", "N", "Cl", "S"}
    sc = Scorer(mol, ge, stats, torch.device("cpu"))
    qr = sc._q6_pmf
    if qr is None:
        return np.zeros(len(X), dtype=np.float64)
    ai, bi = ge._q6_a, ge._q6_b
    sym = [a.symbol for a in mol.atoms]
    keep_mask = np.zeros(len(ai), dtype=bool)
    for j, (ia, ib) in enumerate(zip(ai.tolist(), bi.tolist())):
        sa, sb = sym[ia], sym[ib]
        if (sa == "H" and sb not in supported_hb_acceptors) or (sb == "H" and sa not in supported_hb_acceptors):
            keep_mask[j] = True
    if not np.any(keep_mask):
        return np.zeros(len(X), dtype=np.float64)

    with torch.no_grad():
        xf = X.to(dtype=torch.float32)
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
        # -wells is the positive raw score contribution (in nats) of attractive polar-H wells
        kept_nats = (-wells[:, keep_mask]).sum(dim=-1).numpy().astype(np.float64)
    return kept_nats


def evaluate_probes_all_ablations(c2_tables_A1: dict, c2_tables_A2: dict):
    """Evaluate A0, A1, A2, C2, C3, C4, C3+C4, C5 on the 1,008 rigid dimer probe geometries."""
    from rdkit import Chem
    from research.archive.code.interactions import nci
    from nci_calibration import MONOMERS, topology

    stats = stats_from_db(CANDIDATE)
    tables_bilin = stats.nci_tables()

    probe_results = {}
    by_type = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    probe_files = sorted(PROBE_DIR.glob("hb_*.npz")) + sorted(PROBE_DIR.glob("pp_*.npz")) + sorted(PROBE_DIR.glob("chp_*.npz"))
    for path in probe_files:
        name = path.stem
        meta = json.loads(path.with_suffix(".json").read_text())
        typ = meta["type"]
        labels = meta["labels"]
        scans = sorted(set(l["scan"] for l in labels))
        rdA = Chem.AddHs(Chem.MolFromSmiles(MONOMERS[meta["a"]]))
        rdB = Chem.AddHs(Chem.MolFromSmiles(MONOMERS[meta["b"]]))
        mol, _ = topology(rdA, rdB)

        with np.load(path) as z:
            coords = z["X"]
            q6_old = z["q6_old"].astype(np.float64)
            q6_core = z["q6_core"].astype(np.float64)
            score_bilin = z[f"score_{typ}"].astype(np.float64)

        xt = torch.tensor(coords, dtype=torch.float32)
        rt_A1 = build_c2_runtime_for_mol(mol, c2_tables_A1, "cpu", types=(typ,))
        rt_A2 = build_c2_runtime_for_mol(mol, c2_tables_A2, "cpu", types=(typ,))

        with torch.no_grad():
            s_A1 = evaluate_c2_runtime(rt_A1, xt).numpy().astype(np.float64) if rt_A1 else np.zeros(len(coords))
            s_A2 = evaluate_c2_runtime(rt_A2, xt).numpy().astype(np.float64) if rt_A2 else np.zeros(len(coords))

        base_e = q6_core if typ == "hb" else q6_old
        preds = {
            "off": q6_old,
            "A0_bilinear": base_e - KT * score_bilin,
            "A1_c2_only": base_e - KT * s_A1,
            "A2_sharp_reg": base_e - KT * s_A2,
        }

        if typ == "hb":
            # C2: chemical/charge hierarchy + depth alignment to Q6 polar-H replacement baseline
            is_anionic_probe = any(k in name for k in ("sulfonate", "acetate", "chloride"))
            # Neutral vs anionic stratum scale fitted on calibration
            scale_c2 = 2.45 if not is_anionic_probe else 3.20
            preds["C2_hb_hierarchy"] = base_e - KT * (scale_c2 * s_A2)
        elif typ == "chp":
            with torch.no_grad():
                s_c3 = eval_chp_variants_torch(mol, rt_A2["chp"], xt, use_c3_dir=True, use_c4_sibling=False).numpy().astype(np.float64)
                s_c4 = eval_chp_variants_torch(mol, rt_A2["chp"], xt, use_c3_dir=False, use_c4_sibling=True).numpy().astype(np.float64)
                s_c34 = eval_chp_variants_torch(mol, rt_A2["chp"], xt, use_c3_dir=True, use_c4_sibling=True).numpy().astype(np.float64)
            preds["C3_chp_direction_only"] = base_e - KT * s_c3
            preds["C4_chp_sibling_only"] = base_e - KT * s_c4
            preds["C3_plus_C4_chp"] = base_e - KT * s_c34
        elif typ == "pp":
            with torch.no_grad():
                s_c5 = eval_pp_c5_slip_torch(mol, rt_A2["pp"], xt).numpy().astype(np.float64)
            preds["C5_pp_slip_context"] = base_e - KT * s_c5

        probe_results[name] = {}
        for ext, label_ext in [("mace64", "MACE64"), ("gfn2", "GFN2")]:
            e = np.asarray(json.loads(path.with_suffix(f".{ext}.json").read_text())["interaction"], dtype=np.float64)
            probe_results[name][label_ext] = {}
            for arm, pred in preds.items():
                de = e.copy()
                dp = pred.copy()
                regrets = []
                wrong_curv = 0
                total_curv = 0
                corrs = []
                for scan in scans:
                    ids = np.array([i for i, l in enumerate(labels) if l["scan"] == scan])
                    far = next(i for i in ids if labels[i]["far"])
                    de[ids] -= e[far]
                    dp[ids] -= pred[far]
                    near = ids[[not labels[i]["far"] for i in ids]]
                    regrets.append(float(e[near[np.argmin(pred[near])]] - e[near].min()))
                    if len(near) >= 4 and np.ptp(de[near]) > 1e-8 and np.ptp(dp[near]) > 1e-8:
                        corrs.append(float(spearmanr(de[near], dp[near]).statistic))
                    r_vals = np.array([labels[i]["r"] for i in near])
                    order = np.argsort(r_vals)
                    de_ord = de[near[order]]
                    dp_ord = dp[near[order]]
                    if len(de_ord) >= 3:
                        d2e = de_ord[2:] - 2.0 * de_ord[1:-1] + de_ord[:-2]
                        d2p = dp_ord[2:] - 2.0 * dp_ord[1:-1] + dp_ord[:-2]
                        mask = d2e > 0.05
                        total_curv += int(mask.sum())
                        wrong_curv += int(((d2p <= 0.0) & mask).sum())

                good = (de < 50.0) & np.array([not l["far"] for l in labels])
                rmse = float(np.sqrt(np.mean((de[good] - dp[good]) ** 2)))
                reg = float(np.mean(regrets))
                wc = float(wrong_curv / max(total_curv, 1))
                sp = float(np.median(corrs)) if corrs else 0.0
                probe_results[name][label_ext][arm] = dict(
                    rmse=rmse,
                    radial_regret=reg,
                    wrong_curvature_share=wc,
                    median_scan_spearman=sp,
                )
                by_type[typ][arm][label_ext].append((rmse, reg, wc, sp))

    summary = {}
    for typ, arms in by_type.items():
        summary[typ] = {}
        for arm, exts in arms.items():
            summary[typ][arm] = {}
            for ext_name, vals in exts.items():
                arr = np.array(vals, dtype=np.float64)
                summary[typ][arm][ext_name] = dict(
                    mean_rmse=float(arr[:, 0].mean()),
                    mean_radial_regret=float(arr[:, 1].mean()),
                    mean_wrong_curvature_share=float(arr[:, 2].mean()),
                    median_scan_spearman=float(np.median(arr[:, 3])),
                )
    return dict(summary_by_type=summary, probes=probe_results)


def case_metrics(energy: np.ndarray, pred: np.ndarray, nci_term: np.ndarray, meta: dict) -> dict:
    """Compute complete energy-to-score correspondence metrics for a 49-pose complex motion case."""
    steps = np.asarray(meta["steps"], dtype=np.float64)
    n = len(steps)
    regrets = []
    grads = []
    ks = []
    stable_phys = []
    surr_2step_stable = []
    nci_2step_stable = []

    for j in range(len(meta["modes"])):
        ids = np.r_[0, 1 + j * n + np.arange(n)]
        regrets.append(float(energy[ids[np.argmin(pred[ids])]] - energy[ids].min()))

        def deriv(y, h):
            a = 1 + j * n + int(np.flatnonzero(np.isclose(steps, -h))[0])
            b = 1 + j * n + int(np.flatnonzero(np.isclose(steps, h))[0])
            return (y[b] - y[a]) / (2.0 * h), (y[b] + y[a] - 2.0 * y[0]) / (h * h)

        eg, ek = deriv(energy, 0.04)
        pg, pk04 = deriv(pred, 0.04)
        _, pk02 = deriv(pred, 0.02)
        _, nk04 = deriv(nci_term, 0.04)
        _, nk02 = deriv(nci_term, 0.02)
        _, ek08 = deriv(energy, 0.08)

        if abs(eg) > 0.25:
            grads.append(bool(eg * pg > 0.0))
        ks.append((ek, pk04))
        is_phys_stable = bool(ek > 0.1 and abs(ek - ek08) < 0.25 * max(abs(ek), 1.0))
        stable_phys.append(is_phys_stable)
        surr_2step_stable.append(bool(abs(pk04 - pk02) <= 0.25 * max(abs(pk04), 1.0)))
        nci_2step_stable.append(bool(abs(nk04 - nk02) <= 0.25 * max(abs(nk04), 1.0)))

    k = np.asarray(ks, dtype=np.float64)
    mask = np.asarray(stable_phys, dtype=bool)
    pos = (k[:, 0] > 0.1) & (k[:, 1] > 0.1) & mask

    sp = float(spearmanr(energy[1:], pred[1:]).statistic) if np.ptp(pred[1:]) > 1e-10 and np.ptp(energy[1:]) > 1e-10 else 0.0
    pr = float(pearsonr(energy[1:], pred[1:]).statistic) if np.ptp(pred[1:]) > 1e-10 and np.ptp(energy[1:]) > 1e-10 else 0.0
    den = float(pred[1:] @ pred[1:])
    slope_ratio = float(pred[1:] @ energy[1:]) / den if den > 1e-12 else None

    return dict(
        rmse=float(np.sqrt(np.mean((energy[1:] - pred[1:]) ** 2))),
        spearman_r=sp,
        pearson_r=pr,
        slope_energy_over_surrogate=slope_ratio,
        sampled_minimum_regret=float(np.mean(regrets)),
        gradient_correct=int(sum(grads)),
        gradient_tested=len(grads),
        stable_positive_energy_curvatures=int(mask.sum()),
        wrong_sign_curvatures=int(((k[:, 1] <= 0.0) & mask).sum()),
        curvature_ratios=[float(v) for v in (k[pos, 0] / k[pos, 1])],
        surrogate_2step_stable=int(sum(surr_2step_stable)),
        nci_2step_stable=int(sum(nci_2step_stable)),
        modes_total=len(meta["modes"]),
    )


def summarize_case_metrics(mr_list: list, off_mr_list: list = None) -> dict:
    ratios = [v for m in mr_list for v in m["curvature_ratios"]]
    sps = [m["spearman_r"] for m in mr_list]
    prs = [m["pearson_r"] for m in mr_list]
    slopes = [m["slope_energy_over_surrogate"] for m in mr_list if m["slope_energy_over_surrogate"] is not None]
    gc = int(sum(m["gradient_correct"] for m in mr_list))
    gt = int(sum(m["gradient_tested"] for m in mr_list))
    n2_st = int(sum(m["nci_2step_stable"] for m in mr_list))
    s2_st = int(sum(m["surrogate_2step_stable"] for m in mr_list))
    m_tot = int(sum(m["modes_total"] for m in mr_list))

    out = dict(
        mean_case_rmse=float(np.mean([m["rmse"] for m in mr_list])),
        median_spearman_r=float(np.median(sps)),
        mean_spearman_r=float(np.mean(sps)),
        median_pearson_r=float(np.median(prs)),
        mean_pearson_r=float(np.mean(prs)),
        median_slope_ratio=float(np.median(slopes)) if slopes else None,
        mean_sampled_minimum_regret=float(np.mean([m["sampled_minimum_regret"] for m in mr_list])),
        gradient_correct=gc,
        gradient_tested=gt,
        gradient_agreement_fraction=float(gc / max(gt, 1)),
        stable_positive_energy_curvatures=int(sum(m["stable_positive_energy_curvatures"] for m in mr_list)),
        wrong_sign_curvatures=int(sum(m["wrong_sign_curvatures"] for m in mr_list)),
        median_curvature_ratio=float(np.median(ratios)) if ratios else None,
        surrogate_2step_stable_fraction=float(s2_st / max(m_tot, 1)),
        nci_2step_stable_fraction=float(n2_st / max(m_tot, 1)),
    )
    if off_mr_list is not None:
        delta = [m_off["rmse"] - m["rmse"] for m, m_off in zip(mr_list, off_mr_list)]
        out["rmse_improvement_mean"] = float(np.mean(delta))
        out["rmse_improvement_ci95"] = paired_ci(delta)
        out["cases_better_worse_tied"] = [
            int(sum(v > 1e-6 for v in delta)),
            int(sum(v < -1e-6 for v in delta)),
            int(sum(abs(v) <= 1e-6 for v in delta)),
        ]
    return out


def run_v3_ablations_and_validation(c2_tables_A1: dict, c2_tables_A2: dict):
    """Evaluate all single-factor ablations on v3 calibration, lock candidate D per type,
    and perform the single prospective evaluation on v3 validation."""
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.topology.decompose import decompose_ligand

    selection = json.loads((V3_DIR / "selection.json").read_text())
    complete = json.loads((V3_DIR / "measurement_complete.json").read_text())
    manifest = json.loads((V3_DIR / "measurement_manifest.json").read_text())
    assert selection["coverage_pass"]
    assert complete["manifest_sha256"] == digest(V3_DIR / "measurement_manifest.json")

    stats = stats_from_db(CANDIDATE)
    chosen = {
        (r["shard"], r["code"]): dict(code=r["code"], shard=r["shard"], dir=r["directory"], weight=r.get("weight", 1.0))
        for r in selection["selected"]
    }
    mols_by_code = {}
    for row, mol, error in stream(chosen.values()):
        if not error and mol is not None:
            for lig in mol.ligands:
                decompose_ligand(mol, lig)
            mols_by_code[row["code"]] = mol

    shift = lambda x: x - x[0]
    cases_data = []
    for r in selection["selected"]:
        t = r["type"]
        code = r["code"]
        mol = mols_by_code[code]
        p = V3_DIR / f"{t}_{code}.npz"
        meta = json.loads(p.with_suffix(".json").read_text())
        assert manifest["coordinates_and_scores"][p.name] == digest(p)
        e_mace = np.asarray(json.loads(p.with_suffix(".mace64.json").read_text())["relative"], dtype=np.float64)

        with np.load(p) as z:
            coords = z["X"]
            old = -KT * z["off_total"].astype(np.float64)
            base = old - KT * (z["on_Q6"].astype(np.float64) - z["off_Q6"].astype(np.float64)) if t == "hb" else old.copy()
            nci_bilin = -KT * z["NCI_" + t].astype(np.float64)

        xt = torch.tensor(coords, dtype=torch.float32)
        rt_A1 = build_c2_runtime_for_mol(mol, c2_tables_A1, "cpu", types=(t,))
        rt_A2 = build_c2_runtime_for_mol(mol, c2_tables_A2, "cpu", types=(t,))
        with torch.no_grad():
            nci_A1 = -KT * (evaluate_c2_runtime(rt_A1, xt).numpy().astype(np.float64) if rt_A1 else np.zeros(len(coords)))
            nci_A2 = -KT * (evaluate_c2_runtime(rt_A2, xt).numpy().astype(np.float64) if rt_A2 else np.zeros(len(coords)))

        extra_terms = {}
        if t == "hb":
            ge = GeometryExtractor(mol, torch.device("cpu"))
            q6_unsup_keep = compute_q6_unsupported_halogen_preservation(mol, ge, stats, xt)
            # In energy units (-KT * raw_nats): keeping attractive Q6 wells adds -KT * q6_unsup_keep
            extra_terms["base_c2"] = shift(base - KT * q6_unsup_keep)
            extra_terms["stratum"] = r.get("stratum", "neutral_partner")
        elif t == "chp":
            with torch.no_grad():
                s_c3_A1 = eval_chp_variants_torch(mol, rt_A1["chp"], xt, use_c3_dir=True, use_c4_sibling=False).numpy().astype(np.float64)
                s_c4_A1 = eval_chp_variants_torch(mol, rt_A1["chp"], xt, use_c3_dir=False, use_c4_sibling=True).numpy().astype(np.float64)
                s_c34_A2 = eval_chp_variants_torch(mol, rt_A2["chp"], xt, use_c3_dir=True, use_c4_sibling=True).numpy().astype(np.float64)
            extra_terms["nci_C3_only"] = shift(-KT * s_c3_A1)
            extra_terms["nci_C4_only"] = shift(-KT * s_c4_A1)
            extra_terms["nci_C34_A2"] = shift(-KT * s_c34_A2)
        elif t == "pp":
            with torch.no_grad():
                s_c5_A1 = eval_pp_c5_slip_torch(mol, rt_A1["pp"], xt).numpy().astype(np.float64)
                s_c5_A2 = eval_pp_c5_slip_torch(mol, rt_A2["pp"], xt).numpy().astype(np.float64)
            extra_terms["nci_C5_only"] = shift(-KT * s_c5_A1)
            extra_terms["nci_C5_A2"] = shift(-KT * s_c5_A2)

        cases_data.append(
            dict(
                request=r,
                meta=meta,
                energy=e_mace,
                old=shift(old),
                base=shift(base),
                nci_A0=shift(nci_bilin),
                nci_A1=shift(nci_A1),
                nci_A2=shift(nci_A2),
                **extra_terms,
            )
        )

    def fit_scalar(train_cases, nci_key: str, base_key: str = "base") -> float:
        num = sum(float(d[nci_key][1:] @ (d["energy"][1:] - d[base_key][1:])) / (len(d["energy"]) - 1) for d in train_cases)
        den = sum(float(d[nci_key][1:] @ d[nci_key][1:]) / (len(d["energy"]) - 1) for d in train_cases)
        return max(0.0, num / den) if den > 1e-12 else 0.0

    calibration_ablations = {}
    locked_candidates_D = {}
    prospective_validation = {}

    for typ in ("hb", "pp", "chp"):
        cal = [d for d in cases_data if d["request"]["type"] == typ and d["request"]["split"] == "calibration"]
        val = [d for d in cases_data if d["request"]["type"] == typ and d["request"]["split"] == "validation"]
        assert len(cal) == 16 and len(val) == 16

        cal_off_mr = [case_metrics(d["energy"], d["old"], np.zeros_like(d["old"]), d["meta"]) for d in cal]
        val_off_mr = [case_metrics(d["energy"], d["old"], np.zeros_like(d["old"]), d["meta"]) for d in val]

        arms_spec = [
            ("A0_bilinear", "nci_A0", "base", "single"),
            ("A1_c2_only", "nci_A1", "base", "single"),
            ("A2_sharp_reg", "nci_A2", "base", "single"),
        ]
        if typ == "hb":
            arms_spec.extend(
                [
                    ("C2a_unsupported_q6_preserved", "nci_A1", "base_c2", "single"),
                    ("C2b_charge_hierarchy_only", "nci_A1", "base_c2", "stratum"),
                    ("D_combined_A2_plus_C2", "nci_A2", "base_c2", "stratum"),
                ]
            )
        elif typ == "chp":
            arms_spec.extend(
                [
                    ("C3_direction_only", "nci_C3_only", "base", "single"),
                    ("C4_carbon_sibling_only", "nci_C4_only", "base", "single"),
                    ("D_combined_A2_C3_C4", "nci_C34_A2", "base", "single"),
                ]
            )
        elif typ == "pp":
            arms_spec.extend(
                [
                    ("C5_slip_context_only", "nci_C5_only", "base", "single"),
                    ("D_combined_A2_plus_C5", "nci_C5_A2", "base", "single"),
                ]
            )

        calibration_ablations[typ] = {"off": summarize_case_metrics(cal_off_mr)}
        fitted_params_by_arm = {}

        for arm_name, nci_key, base_key, mode in arms_spec:
            if mode == "single":
                c_fit = fit_scalar(cal, nci_key, base_key)
                fitted_params_by_arm[arm_name] = {"mode": "single", "c": c_fit, "nci_key": nci_key, "base_key": base_key}
                mr_unit = [case_metrics(d["energy"], d[base_key] + d[nci_key], d[nci_key], d["meta"]) for d in cal]
                mr_cal = [case_metrics(d["energy"], d[base_key] + c_fit * d[nci_key], c_fit * d[nci_key], d["meta"]) for d in cal]
            else:
                cal_neut = [d for d in cal if d["stratum"] == "neutral_partner"]
                cal_anio = [d for d in cal if d["stratum"] == "anionic_partner"]
                c_neut = fit_scalar(cal_neut, nci_key, base_key)
                c_anio = fit_scalar(cal_anio, nci_key, base_key)
                fitted_params_by_arm[arm_name] = {
                    "mode": "stratum",
                    "c_neutral": c_neut,
                    "c_anionic": c_anio,
                    "nci_key": nci_key,
                    "base_key": base_key,
                }
                mr_unit = [case_metrics(d["energy"], d[base_key] + d[nci_key], d[nci_key], d["meta"]) for d in cal]
                mr_cal = [
                    case_metrics(
                        d["energy"],
                        d[base_key] + (c_anio if d["stratum"] == "anionic_partner" else c_neut) * d[nci_key],
                        (c_anio if d["stratum"] == "anionic_partner" else c_neut) * d[nci_key],
                        d["meta"],
                    )
                    for d in cal
                ]

            calibration_ablations[typ][arm_name] = dict(
                fitted_params=fitted_params_by_arm[arm_name],
                unit=summarize_case_metrics(mr_unit, cal_off_mr),
                calibrated=summarize_case_metrics(mr_cal, cal_off_mr),
            )

        # Lock candidate D strictly from calibration results before touching validation
        d_arm_name = [a[0] for a in arms_spec if a[0].startswith("D_")][0]
        locked_candidates_D[typ] = dict(
            locked_arm=d_arm_name,
            fitted_params=fitted_params_by_arm[d_arm_name],
            calibration_metrics=calibration_ablations[typ][d_arm_name]["calibrated"],
        )

        # Now evaluate ONLY 'off', 'A0_bilinear', and locked 'D' on the untouched v3 validation split
        prospective_validation[typ] = {"off": summarize_case_metrics(val_off_mr)}
        for eval_arm in ("A0_bilinear", d_arm_name):
            fp = fitted_params_by_arm[eval_arm]
            nk, bk = fp["nci_key"], fp["base_key"]
            mr_val_unit = [case_metrics(d["energy"], d[bk] + d[nk], d[nk], d["meta"]) for d in val]
            if fp["mode"] == "single":
                c_val = fp["c"]
                mr_val_cal = [case_metrics(d["energy"], d[bk] + c_val * d[nk], c_val * d[nk], d["meta"]) for d in val]
                active_coeff = c_val > 0.0
            else:
                cn, ca = fp["c_neutral"], fp["c_anionic"]
                mr_val_cal = [
                    case_metrics(
                        d["energy"],
                        d[bk] + (ca if d["stratum"] == "anionic_partner" else cn) * d[nk],
                        (ca if d["stratum"] == "anionic_partner" else cn) * d[nk],
                        d["meta"],
                    )
                    for d in val
                ]
                active_coeff = (cn > 0.0) or (ca > 0.0)

            s_unit = summarize_case_metrics(mr_val_unit, val_off_mr)
            s_cal = summarize_case_metrics(mr_val_cal, val_off_mr)
            ratio = s_cal["median_curvature_ratio"]
            gate_pass = bool(
                active_coeff
                and s_cal["rmse_improvement_ci95"][0] > 0.0
                and s_cal["mean_sampled_minimum_regret"] <= prospective_validation[typ]["off"]["mean_sampled_minimum_regret"]
                and s_cal["wrong_sign_curvatures"] <= prospective_validation[typ]["off"]["wrong_sign_curvatures"]
                and ratio is not None
                and 0.7 <= ratio <= 1.5
                and s_cal["nci_2step_stable_fraction"] >= 0.90
            )
            prospective_validation[typ][eval_arm] = dict(
                fitted_params=fp,
                unit=s_unit,
                calibrated=s_cal,
                prospective_gate_pass=gate_pass,
            )

    return dict(
        calibration_ablations=calibration_ablations,
        locked_candidates_D=locked_candidates_D,
        prospective_validation=prospective_validation,
    )


def run_frozen_pool_gate(c2_tables_A2: dict, locked_D: dict):
    """Evaluate frozen-pool conformer re-ranking across the 36 saved pools (12 cases x 3 seeds)
    for baseline A0_bilinear, hb-only D, and all-types D, preserving stored assignment terms
    and bend prior."""
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from research.archive.code.interactions import nci
    from jacobik.topology.decompose import decompose_ligand
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry

    stats = stats_from_db(CANDIDATE)
    names = {c: (n, s) for s in ("dev32", "n48") for c, n in set_cases(s)}

    fp_hb = locked_D["hb"]["fitted_params"]
    c_hb_mean = 0.5 * (fp_hb.get("c_neutral", 1.0) + fp_hb.get("c_anionic", 1.0)) if fp_hb["mode"] == "stratum" else fp_hb["c"]
    c_pp = locked_D["pp"]["fitted_params"]["c"]
    c_chp = locked_D["chp"]["fitted_params"]["c"]

    arms = ["A0_bilinear_all", "D_hb_only_calibrated", "D_hb_only_unit", "D_all_calibrated"]
    pool_rows_by_arm = {a: [] for a in arms}
    case_means_by_arm = {a: {} for a in arms}

    for group, codes in COHORT.items():
        for code in codes:
            name, set_name = names[code]
            case = get_case(name)
            with starting_geometry(case, "full_oracle"):
                mol, _, _ = case.make_builder()._build_shell(False, False)
            for lig in mol.ligands:
                decompose_ligand(mol, lig)
            ge = GeometryExtractor(mol, torch.device("cpu"))
            scorer = Scorer(mol, ge, stats, torch.device("cpu"))
            rt_bilin = nci.build_runtime(mol, stats.nci_tables(), "cpu")
            rt_c2 = build_c2_runtime_for_mol(mol, c2_tables_A2, "cpu", types=("hb", "pp", "chp"))

            per_arm_case = {a: [] for a in arms}
            for seed in (11, 12, 13):
                f = pool_path("etkdg_u1_os", set_name, code, seed)
                energy = pool_energy(f, "mace64")
                with np.load(f) as z:
                    Z = float(z["Z"])
                    X = torch.tensor(z["X"], dtype=torch.float32)
                    scores_base = z["score"].copy()

                    nci.NCI_ON = False
                    scorer._nci = None
                    with torch.no_grad():
                        _, base_bd = scorer.score(X)
                        nci.NCI_ON = True
                        nci.ACTIVE_TYPES = ("hb", "pp", "chp")
                        scorer._nci = rt_bilin
                        _, cand_bd = scorer.score(X)
                    nci.NCI_ON = False
                    scorer._nci = None

                    q6_diff_all = (cand_bd["Q6"] - base_bd["Q6"]).numpy()  # already divided by Z
                    q6_unsup_keep_raw = compute_q6_unsupported_halogen_preservation(mol, ge, stats, X)
                    q6_diff_c2 = q6_diff_all + (q6_unsup_keep_raw / Z)

                    nci_bilin_delta = cand_bd.get("NCI", torch.zeros(len(X))).numpy() + q6_diff_all

                    with torch.no_grad():
                        s_hb_c2 = evaluate_c2_runtime({"hb": rt_c2["hb"]}, X).numpy() / Z if "hb" in rt_c2 else np.zeros(len(X))
                        s_pp_c2 = eval_pp_c5_slip_torch(mol, rt_c2["pp"], X).numpy() / Z if "pp" in rt_c2 else np.zeros(len(X))
                        s_chp_c2 = (
                            eval_chp_variants_torch(mol, rt_c2["chp"], X, use_c3_dir=True, use_c4_sibling=True).numpy() / Z
                            if "chp" in rt_c2
                            else np.zeros(len(X))
                        )

                    deltas = {
                        "A0_bilinear_all": nci_bilin_delta,
                        "D_hb_only_calibrated": q6_diff_c2 + c_hb_mean * s_hb_c2,
                        "D_hb_only_unit": q6_diff_c2 + s_hb_c2,
                        "D_all_calibrated": q6_diff_c2 + c_hb_mean * s_hb_c2 + c_pp * s_pp_c2 + c_chp * s_chp_c2,
                    }

                    floor_e = float(energy.min())
                    low_mask = energy <= (floor_e + 50.0)
                    ib = int(scores_base.argmax())
                    base_e = float(energy[ib])
                    base_rho = (
                        float(spearmanr(-scores_base[low_mask], energy[low_mask]).statistic)
                        if low_mask.sum() >= 5 and np.ptp(scores_base[low_mask]) > 1e-10
                        else 0.0
                    )

                    for arm, d_score in deltas.items():
                        sc_arm = scores_base + d_score
                        ic = int(sc_arm.argmax())
                        cand_e = float(energy[ic])
                        cand_rho = (
                            float(spearmanr(-sc_arm[low_mask], energy[low_mask]).statistic)
                            if low_mask.sum() >= 5 and np.ptp(sc_arm[low_mask]) > 1e-10
                            else 0.0
                        )
                        row = dict(
                            group=group,
                            case=code,
                            seed=seed,
                            base_energy=base_e,
                            cand_energy=cand_e,
                            energy_change=cand_e - base_e,
                            base_regret=base_e - floor_e,
                            cand_regret=cand_e - floor_e,
                            base_top1_within_1kcal=bool(base_e - floor_e <= 4.184),
                            cand_top1_within_1kcal=bool(cand_e - floor_e <= 4.184),
                            base_spearman_floor50=base_rho,
                            cand_spearman_floor50=cand_rho,
                            changed_pick=bool(ic != ib),
                        )
                        pool_rows_by_arm[arm].append(row)
                        per_arm_case[arm].append(row)

            for arm in arms:
                rlist = per_arm_case[arm]
                case_means_by_arm[arm][code] = dict(
                    group=group,
                    base_mean=float(np.mean([r["base_energy"] for r in rlist])),
                    cand_mean=float(np.mean([r["cand_energy"] for r in rlist])),
                    mean_change=float(np.mean([r["energy_change"] for r in rlist])),
                    changed_picks=int(sum(r["changed_pick"] for r in rlist)),
                )

    summary_by_arm = {}
    for arm in arms:
        rlist = pool_rows_by_arm[arm]
        cdict = case_means_by_arm[arm]
        summary_by_arm[arm] = dict(
            pools=len(rlist),
            changed_picks=int(sum(r["changed_pick"] for r in rlist)),
            mean_energy_change_kj=float(np.mean([r["energy_change"] for r in rlist])),
            median_case_energy_change_kj=float(np.median([v["mean_change"] for v in cdict.values()])),
            mean_base_regret_kj=float(np.mean([r["base_regret"] for r in rlist])),
            mean_cand_regret_kj=float(np.mean([r["cand_regret"] for r in rlist])),
            base_top1_within_1kcal_count=int(sum(r["base_top1_within_1kcal"] for r in rlist)),
            cand_top1_within_1kcal_count=int(sum(r["cand_top1_within_1kcal"] for r in rlist)),
            median_base_spearman_floor50=float(np.median([r["base_spearman_floor50"] for r in rlist])),
            median_cand_spearman_floor50=float(np.median([r["cand_spearman_floor50"] for r in rlist])),
            better_cases_over_10kj=int(sum(v["mean_change"] < -10.0 for v in cdict.values())),
            worse_cases_over_10kj=int(sum(v["mean_change"] > 10.0 for v in cdict.values())),
            per_case_mean_change_kj={k: v["mean_change"] for k, v in cdict.items()},
        )
    return summary_by_arm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    # Load leave-validation-out histograms prepared by nci_training_calibration_v3.py
    with np.load(V3_DIR / "leave_val_out_hist.npz") as z:
        leave_val_out_hist = {k: z[k].astype(np.float64) for k in z.files}

    c2_tables_A1 = build_c2_tables_from_hist(leave_val_out_hist, lam_rough=0.0, lam_sharp=0.0)
    c2_tables_A2 = build_c2_tables_from_hist(
        leave_val_out_hist, lam_rough=0.05, lam_sharp=2.0, gamma_by_type={"hb": 1.0, "pp": 0.5, "chp": 0.5}
    )

    probes = evaluate_probes_all_ablations(c2_tables_A1, c2_tables_A2)
    v3_eval = run_v3_ablations_and_validation(c2_tables_A1, c2_tables_A2)
    pool_gate = run_frozen_pool_gate(c2_tables_A2, v3_eval["locked_candidates_D"])

    report = dict(
        source_hash=__import__("tools.jdev.env", fromlist=["tree_hash"]).tree_hash(),
        base_db_sha256=digest(DB),
        candidate_db_sha256=digest(CANDIDATE),
        v3_selection_sha256=digest(V3_DIR / "selection.json"),
        v3_measurement_manifest_sha256=digest(V3_DIR / "measurement_manifest.json"),
        probe_ablations=probes["summary_by_type"],
        v3_calibration_ablations=v3_eval["calibration_ablations"],
        locked_candidates_D_before_validation=v3_eval["locked_candidates_D"],
        prospective_v3_validation=v3_eval["prospective_validation"],
        frozen_pool_gate_36_pools=pool_gate,
    )
    write(OUT / "ablation_pilot_v3_report.json", report)
    print(json.dumps(v3_eval["prospective_validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
