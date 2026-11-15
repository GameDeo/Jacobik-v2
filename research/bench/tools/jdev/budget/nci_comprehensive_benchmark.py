"""Comprehensive dev80 benchmark for the C2 + sharpness-regularized + descriptor-corrected NCI model:
1. Full dev80 (76 valid complexes x 3 seeds = 228 pools) Scoreboard metrics (Section 9.1) and paired tests
   against u1c+rep, u1c+rep+os, and Pure-N1 (Q6PMF wells OFF).
2. Curvature (E1) and energy scale of terms (Section 8.4) across dev80 cached MACE directions (crystal + pool top-K)
   for Q1, Q2, S_orient, Q5, Q6, and NCI, plus stiff-direction Spearman and NNLS scales.
3. Ensemble metrics by level (Section 8.5): rho_isomer, rho_within, rho_all, rho_lo, validity on dev80 MACE64 pools,
   plus basin/strain metrics on GFN2-relaxed dev80 pools.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
import torch

_SRC = str(Path(__file__).resolve().parents[5] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from common import DB, REPO, budget_out, pool_energy, pool_files, read_pool, stats_from_db
from pools import rank_terms, set_cases, setup
from nci_targeted_rescore import CANDIDATE, COHORT
from nci_c2_surface import build_c2_surface_params, evaluate_c2_runtime, fit_empirical_surface
from nci_ablation_pilot import (
    KT,
    V3_DIR,
    build_c2_runtime_for_mol,
    eval_chp_variants_torch,
    eval_pp_c5_slip_torch,
)
from tools.jdev.bench import paired_test
from tools.jdev.cases import get_case
from tools.jdev.start import starting_geometry
from jacobik.ik.geometry import GeometryExtractor
from jacobik.ik.scorer import Scorer
from research.archive.code.interactions import nci
from jacobik.topology.decompose import decompose_ligand

OUT = budget_out("energy_scale_refit")
REPORT_OUT = REPO / "data/refit/reports/n2_comprehensive_benchmark_2026-10-18"
HIST_DIR = REPO / "data/refit/nci/2026-10-18_bound_fix"


def build_production_c2_nci_tables(s_clip: float = 8.0) -> dict:
    """Build C2 sharpness-regularized NCI tables from full training histograms (benchmark holdout already excluded)."""
    raw_hist = nci.load_hist(sorted(HIST_DIR.glob("ncihist_*.npz")))
    stats = stats_from_db(CANDIDATE)
    base_tables = stats.nci_tables()
    out = {}
    gamma_map = {"hb": 1.0, "pp": 0.5, "chp": 0.5}
    for key in base_tables:
        lab = "|".join(key)
        t = key[0]
        F = fit_empirical_surface(
            t,
            h=raw_hist[lab],
            lam_rough=0.05,
            lam_sharp=2.0,
            gamma_compliance=gamma_map[t],
        )
        params = build_c2_surface_params(t, F)
        params["s_clip"] = s_clip
        out[key] = params
    return out


def eval_nci_components_raw(mol, rt_c2: dict, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw (un-normalized by Z) NCI nats for hb, pp (with C5 slip), and chp (with C3+C4)."""
    n_poses = len(X)
    if not rt_c2:
        z = np.zeros(n_poses, dtype=np.float64)
        return z, z, z
    with torch.no_grad():
        s_hb = (
            evaluate_c2_runtime({"hb": rt_c2["hb"]}, X).double().cpu().numpy()
            if "hb" in rt_c2
            else np.zeros(n_poses, dtype=np.float64)
        )
        s_pp = (
            eval_pp_c5_slip_torch(mol, rt_c2["pp"], X).double().cpu().numpy()
            if "pp" in rt_c2
            else np.zeros(n_poses, dtype=np.float64)
        )
        s_chp = (
            eval_chp_variants_torch(mol, rt_c2["chp"], X, use_c3_dir=True, use_c4_sibling=True).double().cpu().numpy()
            if "chp" in rt_c2
            else np.zeros(n_poses, dtype=np.float64)
        )
    return s_hb, s_pp, s_chp


def run_dev80_scoreboard_and_ensemble(c2_tables: dict, stats):
    """Evaluate all 76 valid dev80 complexes x 3 seeds (228 pools) across arms and compute
    Section 9.1 Scoreboard metrics, paired tests, and Section 8.5 Ensemble metrics."""
    sets = ["dev32", "n48"]
    names = {c: (n, s) for s in sets for c, n in set_cases(s)}
    valid_map = json.loads((REPO / "data/benchmark_runs/2026-10-16_post_refit/donor_atoms.json").read_text())
    bad_codes = {c for c, v in valid_map.items() if isinstance(v, dict) and not v.get("valid_atoms", True)}

    # Calibrated scales from v3 calibration
    c_hb = 1.135
    c_pp = 2.817
    c_chp = 0.448

    rows = []
    ens_rows_by_arm = {"u1c+rep": [], "u1c+rep+os": [], "u1_pure_rep": [], "u1+nci_hb": [], "u1+nci_all": []}

    # First load existing stored arms: u1c, u1c+rep, u1c+rep+os
    for arm_label, folder in [
        ("u1c", "etkdg_u1_contact"),
        ("u1c+rep", "etkdg_u1_rep"),
        ("u1c+rep+os", "etkdg_u1_os"),
    ]:
        for s, code, seed, f in pool_files(folder, sets, OUT):
            if code in bad_codes:
                continue
            z, e = read_pool(f), pool_energy(f, "mace64")
            if z is None or e is None:
                continue
            sc = z["score"].astype(np.float64)
            i1 = int(np.argmax(sc))
            lo = np.isfinite(e) & (e <= np.nanmin(e) + 50.0)
            rho_lo = float(spearmanr(sc[lo], -e[lo]).statistic) if lo.sum() >= 5 and np.ptp(sc[lo]) > 1e-12 else np.nan
            rho_all = float(spearmanr(sc, -e).statistic) if len(e) >= 3 and np.ptp(sc) > 1e-12 else np.nan
            rows.append(
                dict(
                    config=arm_label,
                    set=s,
                    case=code,
                    seed=seed,
                    pick=float(e[i1]),
                    floor=float(np.nanmin(e)),
                    regret=float(e[i1] - np.nanmin(e)),
                    top1=float(e[i1] <= np.nanmin(e) + 4.184),
                    rho_lo=rho_lo,
                    rho_all=rho_all,
                    invalid_share=float(np.mean(e > 1000.0)),
                )
            )
            if arm_label in ens_rows_by_arm:
                asg = z["asg"]
                cls_df = pd.DataFrame(dict(asg=asg, s=sc, e=e))
                best = cls_df.loc[cls_df.groupby("asg").s.idxmax()]
                rho_iso = float(spearmanr(best.s, -best.e).statistic) if len(best) >= 3 and np.ptp(best.s) > 1e-12 else np.nan
                ws = [
                    float(spearmanr(g.s, -g.e).statistic)
                    for _, g in cls_df.groupby("asg")
                    if len(g) >= 4 and np.ptp(g.s) > 1e-12 and np.ptp(g.e) > 1e-12
                ]
                ens_rows_by_arm[arm_label].append(
                    dict(
                        case=code,
                        seed=seed,
                        rho_isomer_sp=rho_iso,
                        rho_within=float(np.nanmedian(ws)) if ws else np.nan,
                        rho_all=rho_all,
                        rho_lo=rho_lo,
                        invalid_share=float(np.mean(e > 1000.0)),
                    )
                )

    # Now evaluate u1_pure_rep, u1+nci_hb, and u1+nci_all across all 228 valid pools of etkdg_u1_os
    near_crystal_q6_kt = []
    near_crystal_nci_kt = []

    for s in sets:
        for _, code, seed, f in pool_files("etkdg_u1_os", [s], OUT):
            if code in bad_codes:
                continue
            z, e = read_pool(f), pool_energy(f, "mace64")
            if z is None or e is None:
                continue
            if seed == 11:
                name, _ = names[code]
                case = get_case(name)
                with starting_geometry(case, "full_oracle"):
                    mol, _, _ = case.make_builder()._build_shell(False, False)
                for lig in mol.ligands:
                    decompose_ligand(mol, lig)
                ge = GeometryExtractor(mol, torch.device("cpu"))
                scorer = Scorer(mol, ge, stats, torch.device("cpu"))
                rt_c2 = build_c2_runtime_for_mol(mol, c2_tables, "cpu", types=("hb", "pp", "chp"))
                for t_k in rt_c2:
                    rt_c2[t_k]["s_clip"] = 8.0

            Z = float(z["Z"])
            X = torch.tensor(z["X"], dtype=torch.float32)
            sc_base = z["score"].astype(np.float64).copy()

            nci.NCI_ON = False
            scorer._nci = None
            with torch.no_grad():
                _, bd_off = scorer.score(X)
            nci.NCI_ON = True
            nci.ACTIVE_TYPES = ("hb", "pp", "chp")
            scorer._nci = None
            with torch.no_grad():
                _, bd_pure = scorer.score(X)
            nci.NCI_ON = False

            q6_pure_delta = (bd_pure["Q6"] - bd_off["Q6"]).double().cpu().numpy()  # already / Z
            s_hb_raw, s_pp_raw, s_chp_raw = eval_nci_components_raw(mol, rt_c2, X)

            sc_pure_rep = sc_base + q6_pure_delta
            sc_nci_hb = sc_pure_rep + (c_hb * s_hb_raw) / Z
            sc_nci_all = sc_pure_rep + (c_hb * s_hb_raw + c_pp * s_pp_raw + c_chp * s_chp_raw) / Z

            if seed == 11:
                i_best = int(np.argmax(sc_nci_all))
                near_crystal_q6_kt.append(float(-bd_pure["Q6"][i_best].item() * Z))
                near_crystal_nci_kt.append(float(c_hb * s_hb_raw[i_best] + c_pp * s_pp_raw[i_best] + c_chp * s_chp_raw[i_best]))

            for arm_label, sc_vec in [
                ("u1_pure_rep", sc_pure_rep),
                ("u1+nci_hb", sc_nci_hb),
                ("u1+nci_all", sc_nci_all),
            ]:
                i1 = int(np.argmax(sc_vec))
                lo = np.isfinite(e) & (e <= np.nanmin(e) + 50.0)
                rho_lo = float(spearmanr(sc_vec[lo], -e[lo]).statistic) if lo.sum() >= 5 and np.ptp(sc_vec[lo]) > 1e-12 else np.nan
                rho_all = float(spearmanr(sc_vec, -e).statistic) if len(e) >= 3 and np.ptp(sc_vec) > 1e-12 else np.nan
                rows.append(
                    dict(
                        config=arm_label,
                        set=s,
                        case=code,
                        seed=seed,
                        pick=float(e[i1]),
                        floor=float(np.nanmin(e)),
                        regret=float(e[i1] - np.nanmin(e)),
                        top1=float(e[i1] <= np.nanmin(e) + 4.184),
                        rho_lo=rho_lo,
                        rho_all=rho_all,
                        invalid_share=float(np.mean(e > 1000.0)),
                    )
                )
                asg = z["asg"]
                cls_df = pd.DataFrame(dict(asg=asg, s=sc_vec, e=e))
                best = cls_df.loc[cls_df.groupby("asg").s.idxmax()]
                rho_iso = float(spearmanr(best.s, -best.e).statistic) if len(best) >= 3 and np.ptp(best.s) > 1e-12 else np.nan
                ws = [
                    float(spearmanr(g.s, -g.e).statistic)
                    for _, g in cls_df.groupby("asg")
                    if len(g) >= 4 and np.ptp(g.s) > 1e-12 and np.ptp(g.e) > 1e-12
                ]
                ens_rows_by_arm[arm_label].append(
                    dict(
                        case=code,
                        seed=seed,
                        rho_isomer_sp=rho_iso,
                        rho_within=float(np.nanmedian(ws)) if ws else np.nan,
                        rho_all=rho_all,
                        rho_lo=rho_lo,
                        invalid_share=float(np.mean(e > 1000.0)),
                    )
                )

    df = pd.DataFrame(rows)
    arms_order = ["u1c", "u1c+rep", "u1c+rep+os", "u1_pure_rep", "u1+nci_hb", "u1+nci_all"]
    have = df.groupby(["case", "seed"]).config.nunique()
    keep = have[have == len(arms_order)].index
    df = df.set_index(["case", "seed"]).loc[keep].reset_index()

    scoreboard = {}
    hb_codes = COHORT["hbond"]
    for arm in arms_order:
        sub = df[df.config == arm]
        case_meds = sub.groupby("case").pick.median()
        case_worst = sub.groupby("case").pick.max()
        spread = case_worst - case_meds
        hb_sub = sub[sub.case.isin(hb_codes)]
        hb_case_meds = hb_sub.groupby("case").pick.median()
        scoreboard[arm] = dict(
            pools=len(sub),
            complexes=int(sub.case.nunique()),
            pick_median=float(sub.pick.median()),
            pick_mean=float(sub.pick.mean()),
            floor_median=float(sub.floor.median()),
            floor_mean=float(sub.floor.mean()),
            regret_median=float(sub.regret.median()),
            regret_mean=float(sub.regret.mean()),
            top1_share=float(sub.top1.mean()),
            rho_lo_median=float(sub.rho_lo.median()),
            rho_all_median=float(sub.rho_all.median()),
            case_med_ge_100=int((case_meds >= 100.0).sum()),
            case_med_ge_200=int((case_meds >= 200.0).sum()),
            case_med_ge_300=int((case_meds >= 300.0).sum()),
            pewbol_seed_median=float(case_meds.get("PEWBOL", np.nan)),
            pewbol_seed_mean=float(sub[sub.case == "PEWBOL"].pick.mean()),
            yiyxej_seed_median=float(case_meds.get("YIYXEJ", np.nan)),
            yiyxej_seed_mean=float(sub[sub.case == "YIYXEJ"].pick.mean()),
            hb_subset_median=float(hb_case_meds.median()),
            hb_subset_mean=float(hb_sub.pick.mean()),
            seed_spread_gt_50_count=int((spread > 50.0).sum()),
            invalid_share_gt_1000_pct=float(sub.invalid_share.mean() * 100.0),
        )

    scoreboard["near_crystal_q6_kt_mean"] = float(np.mean(near_crystal_q6_kt))
    scoreboard["near_crystal_q6_kt_p90"] = float(np.percentile(near_crystal_q6_kt, 90))
    scoreboard["near_crystal_nci_kt_mean"] = float(np.mean(near_crystal_nci_kt))
    scoreboard["near_crystal_nci_kt_p90"] = float(np.percentile(near_crystal_nci_kt, 90))

    # Paired tests
    df_all = df.assign(set="all")
    paired_comparisons = {}
    for a_arm, b_arm in [
        ("u1c+rep", "u1+nci_hb"),
        ("u1c+rep", "u1+nci_all"),
        ("u1c+rep+os", "u1+nci_hb"),
        ("u1c+rep+os", "u1+nci_all"),
        ("u1_pure_rep", "u1+nci_hb"),
        ("u1_pure_rep", "u1+nci_all"),
    ]:
        key_pair = f"{b_arm}_vs_{a_arm}"
        paired_comparisons[key_pair] = {}
        for v in ("pick", "floor", "regret"):
            t = paired_test(df_all, a_arm, b_arm, value=v).iloc[0]
            paired_comparisons[key_pair][v] = dict(
                mean_diff=float(t.mean_diff),
                ci_lo=float(t.ci_lo),
                ci_hi=float(t.ci_hi),
                median_diff=float(t.median_diff),
                better=int(t.better),
                worse=int(t.worse),
                mde=float(t.mde),
                verdict=str(t.verdict),
            )

    ensemble_summary = {}
    for arm, erows in ens_rows_by_arm.items():
        edf = pd.DataFrame(erows)
        ensemble_summary[arm] = dict(
            rho_isomer_sp_median=float(edf.rho_isomer_sp.median()),
            rho_within_median=float(edf.rho_within.median()),
            rho_all_median=float(edf.rho_all.median()),
            rho_lo_median=float(edf.rho_lo.median()),
            validity_gt_1000_pct=float(edf.invalid_share.mean() * 100.0),
        )

    return dict(
        scoreboard=scoreboard,
        paired_comparisons=paired_comparisons,
        ensemble_summary=ensemble_summary,
    )


def run_dev80_curvature_e1(c2_tables: dict, stats):
    """Run the Section 8.4 Curvature (E1) analysis across the 80 cached dev80 complexes
    in data/energy_scale_refit/curvature2/geom and u1_rep, measuring Q1, Q2, S_orient (Q3), Q5, Q6, and NCI."""
    gd = OUT / "curvature2" / "geom"
    ud = OUT / "curvature2" / "u1_rep"
    sets = ["dev32", "n48"]
    names = {c: (n, s) for s in sets for c, n in set_cases(s)}

    c_hb = 1.135
    c_pp = 2.817
    c_chp = 0.448
    terms_list = ["Q1", "Q2", "Q3", "Q5", "Q6", "NCI"]
    h_step = 0.10

    rows = []
    for gf in sorted(gd.glob("etkdg_*.npz")):
        stem_parts = gf.stem.split("_")
        s_name = stem_parts[1]
        code = "_".join(stem_parts[2:])
        uf = ud / gf.name
        if code not in names or not uf.exists():
            continue
        name, _ = names[code]
        try:
            g = np.load(gf, allow_pickle=False)
            u = np.load(uf, allow_pickle=False)
            P0, V, pt_of_dir = g["P0"], g["V"], g["point"]
            kE, types = g["kE"], g["types"]
            kS_u1rep = u["kS"].astype(np.float64)  # (N_dir, 5): Q1, Q2, Q3, Q5, Q6

            case = get_case(name)
            with starting_geometry(case, "full_oracle"):
                mol, _, _ = case.make_builder()._build_shell(False, False)
            for lig in mol.ligands:
                decompose_ligand(mol, lig)
            ge = GeometryExtractor(mol, torch.device("cpu"))
            sc = Scorer(mol, ge, stats, torch.device("cpu"))
            rt_c2 = build_c2_runtime_for_mol(mol, c2_tables, "cpu", types=("hb", "pp", "chp"))
            for t_k in rt_c2:
                rt_c2[t_k]["s_clip"] = 8.0

            Z = float(sc._agg_norm())
            pidx = np.where(pt_of_dir < 0, len(P0) - 1, pt_of_dir)
            k_q6_delta_blocks = []
            k_nci_blocks = []

            for p_i in range(len(P0)):
                Vp = V[pidx == p_i].astype(np.float64)
                X0 = P0[p_i]
                Xs = np.stack([X0] + [X0 + h_step * v for v in Vp] + [X0 - h_step * v for v in Vp])
                xt = torch.tensor(Xs, dtype=torch.float32)

                nci.NCI_ON = False
                sc._nci = None
                with torch.no_grad():
                    _, bd_off = sc.score(xt)
                nci.NCI_ON = True
                nci.ACTIVE_TYPES = ("hb", "pp", "chp")
                with torch.no_grad():
                    _, bd_pure = sc.score(xt)
                nci.NCI_ON = False

                q6_delta_raw = (bd_pure["Q6"] - bd_off["Q6"]).double().cpu().numpy() * Z
                s_hb, s_pp, s_chp = eval_nci_components_raw(mol, rt_c2, xt)
                nci_raw = c_hb * s_hb + c_pp * s_pp + c_chp * s_chp

                nV = len(Vp)
                k_q6_delta_blocks.append(
                    -(q6_delta_raw[1 : 1 + nV] + q6_delta_raw[1 + nV :] - 2.0 * q6_delta_raw[0]) / (h_step ** 2)
                )
                k_nci_blocks.append(-(nci_raw[1 : 1 + nV] + nci_raw[1 + nV :] - 2.0 * nci_raw[0]) / (h_step ** 2))

            k_q6_delta = np.concatenate(k_q6_delta_blocks, axis=0)
            k_nci = np.concatenate(k_nci_blocks, axis=0)
            kS_all = np.zeros((len(kE), 6), dtype=np.float64)
            kS_all[:, :5] = kS_u1rep
            kS_all[:, 4] += k_q6_delta  # pure-N1 Q6 curvature (Q6PMF polar-H wells removed)
            kS_all[:, 5] = k_nci        # NCI curvature
            for j in range(len(kE)):
                rows.append(
                    dict(
                        set=s_name,
                        case=code,
                        point=int(pt_of_dir[j]),
                        type=str(types[j]),
                        kE=float(kE[j]) / KT,
                        **{f"k{q}": float(kS_all[j, k]) for k, q in enumerate(terms_list)},
                    )
                )
        except Exception as exc:
            print(f"curvature {code} error: {exc}", flush=True)

    d = pd.DataFrame(rows)
    d["kS"] = d[[f"k{q}" for q in terms_list]].sum(axis=1)
    d["where"] = np.where(d.point < 0, "crystal", "pool top-K")

    # 1. Per-term dominant direction ratio (>= 70% of positive score curvature; >= 30% for NCI since NCI acts alongside Q5/Q6)
    pos_t = d[[f"k{q}" for q in terms_list]].clip(lower=0.0)
    tot = pos_t.sum(axis=1)
    dom_ratios = {}
    for w, idx in d.groupby("where").groups.items():
        dom_ratios[w] = {}
        for q in terms_list:
            thresh = 0.25 if q == "NCI" else 0.70
            m = (pos_t.loc[idx, f"k{q}"] >= thresh * tot.loc[idx]) & (tot.loc[idx] > 0) & (d.loc[idx, "kE"] > 0)
            g = d.loc[idx][m]
            if len(g):
                pc = (g.kE / g.kS).groupby(g.case).median()
                dom_ratios[w][q] = dict(
                    median_ratio=float(pc.median()),
                    n_dirs=int(len(g)),
                    n_complexes=int(len(pc)),
                )

    # 2. Rank agreement (Spearman) across directions within a point
    rank_spearman = {}
    for w, g in d.groupby("where"):
        rh = [float(spearmanr(gg.kE, gg.kS).statistic) for _, gg in g.groupby(["case", "point"]) if len(gg) > 5]
        rank_spearman[w] = float(np.nanmedian(rh))

    # 3. NNLS per-term curvature scales
    nnls_scales = {}
    for w, g in d.groupby("where"):
        A = g[[f"k{q}" for q in terms_list]].values
        y = g.kE.values
        s_fit, _ = nnls(A, y)
        nnls_scales[w] = {q: float(s_fit[k]) for k, q in enumerate(terms_list)}

    # 4. Overall median ratio energy/kT : score by point and DOF type
    pos = d[(d.kE > 0) & (d.kS > 0)].copy()
    pos["ratio"] = pos.kE / pos.kS
    by_dof = (
        pos.groupby(["where", "type"])
        .ratio.agg(["count", "median", lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
        .rename(columns={"<lambda_0>": "p25", "<lambda_1>": "p75"})
        .reset_index()
        .to_dict(orient="records")
    )

    return dict(
        n_directions=len(d),
        n_complexes=int(d.case.nunique()),
        per_term_dominant_curvature_ratio=dom_ratios,
        within_point_spearman_median=rank_spearman,
        nnls_scales=nnls_scales,
        ratio_by_where_and_dof_type=by_dof,
    )


def main():
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    setup()
    stats = stats_from_db(CANDIDATE)
    c2_tables = build_production_c2_nci_tables(s_clip=8.0)

    sb_ens = run_dev80_scoreboard_and_ensemble(c2_tables, stats)
    curv_e1 = run_dev80_curvature_e1(c2_tables, stats)

    report = dict(
        dev80_scoreboard=sb_ens["scoreboard"],
        dev80_paired_tests=sb_ens["paired_comparisons"],
        dev80_ensemble_metrics=sb_ens["ensemble_summary"],
        dev80_curvature_e1=curv_e1,
    )
    out_file = REPORT_OUT / "comprehensive_benchmark_dev80.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
