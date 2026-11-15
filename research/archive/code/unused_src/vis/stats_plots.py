"""
stats_plots.py
==============
GMM/VMM likelihood density helpers and ensemble term-breakdown widget.
"""

from __future__ import annotations

import math
from typing import Any, List

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display

from jacobik.ik.scorer import _gmm_fields, _vmm_fields

# GMM/VMM evaluation functions for individual term analysis
# ---------------------------------------------------------------------------

#: Q2/Q3 (2.7.0) cos-space GMM variance floor -- mirrors
#: ik/scorer.py's _GMM_VAR_MIN["Q2"]/["Q3"]. 0.04 (this module's old
#: hardcoded default, still used for Q1/Q4's Å²/deg² floors) is the wrong
#: scale for a dimensionless cos-space quantity in [-1,1] -- see
#: fit/fitter.py's fit_gmm_cosangle() docstring for why Q2/Q3 are fit in
#: cos-space at all.
_Q2Q3_GMM_COS_VAR_FLOOR = 0.0025


def eval_gmm_log_prob(
    x: float,
    means: List[float],
    vars_: List[float],
    weights: List[float],
    var_floor: float = 0.04,
) -> float:
    _LOG_2PI = math.log(2 * math.pi)
    log_probs = []
    for m, v, w in zip(means, vars_, weights):
        v = max(v, var_floor)  # variance floor
        lp = -0.5 * _LOG_2PI - 0.5 * math.log(v) - 0.5 * (x - m) ** 2 / v + math.log(max(w, 1e-12))
        log_probs.append(lp)
    from scipy.special import logsumexp

    return float(logsumexp(log_probs))


def eval_vmm_log_prob(
    x_rad: float, means_rad: List[float], kappas: List[float], weights: List[float]
) -> float:
    _LOG_2PI = math.log(2 * math.pi)
    log_probs = []
    import scipy.special as sp

    for m, k, w in zip(means_rad, kappas, weights):
        i0e_val = sp.i0e(k)
        log_norm = _LOG_2PI + math.log(max(i0e_val, 1e-12)) + k
        lp = k * math.cos(x_rad - m) - log_norm + math.log(max(w, 1e-12))
        log_probs.append(lp)
    from scipy.special import logsumexp

    return float(logsumexp(log_probs))


def get_gmm_max_log_prob(
    means: List[float], vars_: List[float], weights: List[float], var_floor: float = 0.04
) -> float:
    # Evaluate GMM over a dense 1D grid to find the peak value
    stds = [math.sqrt(max(v, var_floor)) for v in vars_]
    grid_min = min(means) - 3.0 * max(stds)
    grid_max = max(means) + 3.0 * max(stds)
    xs = np.linspace(grid_min, grid_max, 500)
    lps = [eval_gmm_log_prob(x, means, vars_, weights, var_floor=var_floor) for x in xs]
    return float(max(lps))


def get_vmm_max_log_prob(
    means_rad: List[float], kappas: List[float], weights: List[float]
) -> float:
    # Evaluate VMM over full circle to find the peak value
    xs_rad = np.linspace(0.0, 2.0 * math.pi, 500)
    lps = [eval_vmm_log_prob(x, means_rad, kappas, weights) for x in xs_rad]
    return float(max(lps))



def show_ensemble_stats(res: Any, scr: Any) -> None:
    all_confs = res.conformers
    if not all_confs:
        print("EnsembleResult is empty.")
        return

    state = {"idx": 0}

    # Widgets
    btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
    btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
    hdr_html = widgets.HTML(value="")
    plot_out = widgets.Output()

    def _render():
        idx = state["idx"]
        conf = all_confs[idx]
        mol = conf.complex
        X = torch.tensor(
            np.array([a.position for a in mol.atoms]), dtype=torch.float32, device=scr.device
        ).unsqueeze(0)

        hdr_html.value = (
            f"<b style='color:#AAD4FF'>Conformer Rank {conf.rank} / {len(all_confs)}</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;<span style='color:#00E5FF'>score {conf.score:.4f}</span>"
        )
        btn_prev.disabled = idx == 0
        btn_next.disabled = idx == len(all_confs) - 1

        # Compute stats
        terms = []  # dict of term details
        metal_sym = mol.metal_symbol
        coord_geom = mol.coord_geom or "octahedral"
        spin = mol.spin_state
        charge = mol.charge
        from jacobik.ik.scorer import (
            _ligand_graph_distances,
            _trans_partners,
        )

        trans_map = _trans_partners(mol)
        graph_dists = _ligand_graph_distances(mol)

        site_ligand = {}
        for s_i, site in enumerate(mol.coord_sites):
            d_min = min(site.donor_indices)
            for lig in mol.ligands:
                if d_min in lig.atom_indices:
                    site_ligand[s_i] = lig.ligand_id
                    break
            else:
                site_ligand[s_i] = -1

        # --- Q1 ---
        q1_d = scr._ge.q1_distances(X)[0].cpu().numpy()
        for s_i, site in enumerate(mol.coord_sites):
            d_idx = site.donor_indices[0]
            label = mol.donor_labels.get(d_idx)
            subclass = label.subclass if label is not None else "Unknown"
            trans_p = trans_map.get(s_i, "none")
            key = (metal_sym, spin, charge, subclass, coord_geom, trans_p)
            fd, resolved_key, _ = scr.stats.dist_trace("Q1", key)
            val = float(q1_d[s_i])

            fields = _gmm_fields(fd, key, "Q1")
            if fields is not None:
                means, vars_, weights = fields
                lp = eval_gmm_log_prob(val, means, vars_, weights)
                max_lp = get_gmm_max_log_prob(means, vars_, weights)
                residual = max(0.0, max_lp - lp)
            else:
                residual = 0.0  # fallback
                means, vars_, weights = None, None, None

            terms.append(
                {
                    "label": f"Q1: D{s_i} ({subclass})",
                    "value": f"{val:.3f} Å",
                    "val_num": val,
                    "resolved_key": str(resolved_key or "Fallback"),
                    "residual": residual,
                    "unit": "Å",
                    "dist": {"type": "GMM", "means": means, "vars": vars_, "weights": weights}
                    if means is not None
                    else None,
                }
            )

        # --- Q2 ---
        if scr._q2_means.shape[0] > 0:
            q2_angles = scr._ge.q2_dmd_angles(X)[0].cpu().numpy()  # radians
            for p, (s_i, s_j, dist_class) in enumerate(scr._q2_pair_meta):
                val_deg = float(np.degrees(q2_angles[p]))
                site_i = mol.coord_sites[s_i]
                site_j = mol.coord_sites[s_j]
                di_idx = site_i.donor_indices[0]
                dj_idx = site_j.donor_indices[0]
                li = mol.donor_labels.get(di_idx)
                lj = mol.donor_labels.get(dj_idx)
                sub_i = li.subclass if li is not None else "Unknown"
                sub_j = lj.subclass if lj is not None else "Unknown"

                key_i = min(site_i.donor_indices)
                key_j = min(site_j.donor_indices)
                if (sub_i, key_i) <= (sub_j, key_j):
                    sub_A, sub_B = sub_i, sub_j
                else:
                    sub_A, sub_B = sub_j, sub_i

                q2_key = (coord_geom, dist_class, metal_sym, sub_A, sub_B)
                fd, resolved_key, _ = scr.stats.dist_trace("Q2", q2_key)

                # 2.7.0: Q2 moved from VMM to GMM-on-cos(angle) -- see
                # fit/fitter.py's fit_gmm_cosangle() docstring. means/
                # vars are already dimensionless cos-space values, no
                # degree->radian conversion needed.
                fields = _gmm_fields(fd, q2_key, "Q2")
                if fields is not None:
                    means, vars_, weights = fields
                    val_cos = math.cos(q2_angles[p])
                    lp = eval_gmm_log_prob(
                        val_cos, means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                    )
                    max_lp = get_gmm_max_log_prob(
                        means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                    )
                    residual = max(0.0, max_lp - lp)
                else:
                    residual = 0.0
                    means, vars_, weights = None, None, None

                terms.append(
                    {
                        "label": f"Q2: D{s_i}–M–D{s_j}",
                        "value": f"{val_deg:.1f}°",
                        "val_num": q2_angles[p],
                        "val_deg": val_deg,
                        "resolved_key": str(resolved_key or "Fallback"),
                        "residual": residual,
                        "unit": "deg",
                        "dist": {
                            "type": "GMM_cos",
                            "means": means,
                            "vars": vars_,
                            "weights": weights,
                        }
                        if means is not None
                        else None,
                    }
                )

        # --- Q3 ---
        if scr._n_q3 > 0 and scr._q3_means.shape[0] > 0:
            q3_a = scr._ge.q3_approach_angles(X)[0].cpu().numpy()  # radians
            q3_idx = 0
            for s_i, site in enumerate(mol.coord_sites):
                if not scr._ge._q3_eligible[s_i]:
                    continue
                val_deg = float(np.degrees(q3_a[q3_idx]))
                d_idx = site.donor_indices[0]
                label = mol.donor_labels.get(d_idx)
                subclass = label.subclass if label is not None else "Unknown"
                key = (subclass, metal_sym, coord_geom)
                fd, resolved_key, _ = scr.stats.dist_trace("Q3", key)

                # 2.7.0: Q3 moved from VMM to GMM-on-cos(angle) -- see
                # fit/fitter.py's fit_gmm_cosangle() docstring.
                fields = _gmm_fields(fd, key, "Q3")
                if fields is not None:
                    means, vars_, weights = fields
                    val_cos = math.cos(q3_a[q3_idx])
                    lp = eval_gmm_log_prob(
                        val_cos, means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                    )
                    max_lp = get_gmm_max_log_prob(
                        means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                    )
                    residual = max(0.0, max_lp - lp)
                else:
                    residual = 0.0
                    means, vars_, weights = None, None, None

                terms.append(
                    {
                        "label": f"Q3: D{s_i} approach",
                        "value": f"{val_deg:.1f}°",
                        "val_num": q3_a[q3_idx],
                        "val_deg": val_deg,
                        "resolved_key": str(resolved_key or "Fallback"),
                        "residual": residual,
                        "unit": "deg",
                        "dist": {
                            "type": "GMM_cos",
                            "means": means,
                            "vars": vars_,
                            "weights": weights,
                        }
                        if means is not None
                        else None,
                    }
                )
                q3_idx += 1

        # --- Q4 ---
        if scr._ge.n_q4 > 0 and scr._q4_means.shape[0] > 0:
            q4_v = scr._ge.q4_pyramidalization(X)[0].cpu().numpy()  # degrees
            pos_to_site = {}
            idx_to_pos = {a.idx: i for i, a in enumerate(mol.atoms)}
            for s_i, site in enumerate(mol.coord_sites):
                if not site.is_haptic:
                    d_pos = idx_to_pos.get(site.donor_indices[0], -1)
                    if d_pos >= 0:
                        pos_to_site[d_pos] = s_i

            for i, d_pos in enumerate(scr._ge._q4_donor_pos.tolist()):
                val_deg = float(q4_v[i])
                s_i = pos_to_site.get(d_pos, -1)
                site = mol.coord_sites[s_i] if s_i >= 0 else None
                subclass = "Unknown"
                if site:
                    d_idx = site.donor_indices[0]
                    label = mol.donor_labels.get(d_idx)
                    subclass = label.subclass if label is not None else "Unknown"

                key = (subclass, coord_geom)
                fd, resolved_key, _ = scr.stats.dist_trace("Q4", key)

                fields = _gmm_fields(fd, key, "Q4")
                if fields is not None:
                    means, vars_, weights = fields
                    lp = eval_gmm_log_prob(val_deg, means, vars_, weights)
                    max_lp = get_gmm_max_log_prob(means, vars_, weights)
                    residual = max(0.0, max_lp - lp)
                else:
                    residual = 0.0
                    means, vars_, weights = None, None, None

                terms.append(
                    {
                        "label": f"Q4: D{s_i} pyr",
                        "value": f"{val_deg:.1f}°",
                        "val_num": val_deg,
                        "resolved_key": str(resolved_key or "Fallback"),
                        "residual": residual,
                        "unit": "deg",
                        "dist": {
                            "type": "GMM",
                            "means": means,
                            "vars": vars_,
                            "weights": weights,
                        }
                        if means is not None
                        else None,
                    }
                )

        # --- Q5 ---
        if scr._ge.n_q5 > 0 and scr._q5_means.shape[0] > 0:
            q5_t = scr._ge.q5_torsions(X)[0].cpu().numpy()  # radians
            for i, q5_key in enumerate(scr._ge._q5_keys):
                val_deg = float(np.degrees(q5_t[i]))
                fd, resolved_key, _ = scr.stats.dist_trace("Q5", q5_key)

                fields = _vmm_fields(fd, q5_key, "Q5")
                if fields is not None:
                    means, kappas, weights = fields
                    means_rad = [math.radians(m) for m in means]
                    lp = eval_vmm_log_prob(q5_t[i], means_rad, kappas, weights)
                    max_lp = get_vmm_max_log_prob(means_rad, kappas, weights)
                    residual = max(0.0, max_lp - lp)
                else:
                    residual = 0.0
                    means, kappas, weights = None, None, None

                terms.append(
                    {
                        "label": f"Q5: Torsion {i}",
                        "value": f"{val_deg:.1f}°",
                        "val_num": q5_t[i],
                        "val_deg": val_deg,
                        "resolved_key": str(resolved_key or "Fallback"),
                        "residual": residual,
                        "unit": "deg",
                        "dist": {
                            "type": "VMM",
                            "means": means_rad,
                            "kappas": kappas,
                            "weights": weights,
                        }
                        if means is not None
                        else None,
                    }
                )

        # --- Q6 Sterics (active only) ---
        if scr._ge.n_q6 > 0 and scr._q6_mu.shape[0] > 0:
            q6_d = scr._ge.q6_contacts(X)[0].cpu().numpy()  # Å
            for p, elem_key in enumerate(scr._ge._q6_elem_keys):
                val = float(q6_d[p])
                mu = float(scr._q6_mu[p].cpu().numpy())
                if val <= mu:
                    sigma = float(scr._q6_sigma[p].cpu().numpy())
                    clash_penalty = ((val - mu) ** 2) / (2.0 * max(sigma, 1e-4) ** 2)
                    fd, resolved_key, _ = scr.stats.dist_trace("Q6", elem_key)
                    terms.append(
                        {
                            "label": f"Q6: clash {elem_key[0]}–{elem_key[1]}",
                            "value": f"{val:.2f} Å (vdW {mu:.2f})",
                            "val_num": val,
                            "mu": mu,
                            "sigma": sigma,
                            "resolved_key": str(
                                resolved_key or f"covalent fallback mu={mu:.2f}"
                            ),
                            "residual": clash_penalty,
                            "unit": "clash",
                        }
                    )

                    # Render Matplotlib Subplots
        with plot_out:
            plot_out.clear_output(wait=True)
            n_terms = len(terms)
            cols = 3
            rows = math.ceil(n_terms / cols)
            if rows == 0:
                plt.show()
                return
            fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2.5), facecolor="#0F0F1A")
            if n_terms > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            for i, t in enumerate(terms):
                ax = axes[i]
                ax.set_facecolor("#0F0F1A")
                ax.tick_params(colors="#C8D8E8", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333355")

                dist = t.get("dist")
                if t["unit"] == "clash":
                    mu = t.get("mu", 0.0)
                    sigma = t.get("sigma", 0.1)
                    val = t["val_num"]
                    xs = np.linspace(max(0, mu - 4 * sigma), mu + 2 * sigma, 200)
                    ys = np.exp(-np.where(xs <= mu, ((xs - mu) ** 2) / (2 * sigma**2), 0.0))
                    ax.plot(xs, ys, color="#FF6B6B")
                    ax.axvline(val, color="#FFFFFF", linestyle="--")
                    ax.set_title(
                        f"{t['label']} Res: {t['residual']:.2f} | Key: {t['resolved_key']}",
                        color="#AAD4FF",
                        fontsize=9,
                    )
                elif dist is not None:
                    if dist["type"] == "GMM":
                        val = t["val_num"]
                        means = dist["means"]
                        vars_ = dist["vars"]
                        weights = dist["weights"]
                        stds = [math.sqrt(max(v, 0.04)) for v in vars_]
                        grid_min = min(means) - 3.0 * max(stds)
                        grid_max = max(means) + 3.0 * max(stds)
                        xs = np.linspace(grid_min, grid_max, 200)
                        ys = np.array(
                            [math.exp(eval_gmm_log_prob(x, means, vars_, weights)) for x in xs]
                        )
                        ax.plot(xs, ys, color="#00E5FF")
                        ax.axvline(val, color="#FFFFFF", linestyle="--")
                        ax.set_title(
                            f"{t['label']} Res: {t['residual']:.2f} | Key: {t['resolved_key']}",
                            color="#AAD4FF",
                            fontsize=9,
                        )
                    elif dist["type"] == "VMM":
                        val_rad = t["val_num"]
                        val_deg = t["val_deg"]
                        means_rad = dist["means"]
                        kappas = dist["kappas"]
                        weights = dist["weights"]
                        xs_rad = np.linspace(-math.pi, math.pi, 200)
                        ys = np.array(
                            [
                                math.exp(eval_vmm_log_prob(x, means_rad, kappas, weights))
                                for x in xs_rad
                            ]
                        )
                        xs_deg = np.degrees(xs_rad)

                        v_plot = (val_rad + math.pi) % (2 * math.pi) - math.pi

                        ax.plot(xs_deg, ys, color="#00E5FF")
                        ax.axvline(np.degrees(v_plot), color="#FFFFFF", linestyle="--")
                        ax.set_title(
                            f"{t['label']} Res: {t['residual']:.2f} | Key: {t['resolved_key']}",
                            color="#AAD4FF",
                            fontsize=9,
                        )
                    elif dist["type"] == "GMM_cos":
                        # 2.7.0: Q2/Q3, fit in cos-space (bounded
                        # [0,180] deg via arccos, not circular -- see
                        # fitter.py's fit_gmm_cosangle() docstring).
                        # Plotted here in degrees for readability, with
                        # the Jacobian |d(cos theta)/d theta| = sin(theta)
                        # applied so the curve is a proper density over
                        # degrees rather than a raw cos-space density
                        # stretched onto a degree axis.
                        val_deg = t["val_deg"]
                        means = dist["means"]
                        vars_ = dist["vars"]
                        weights = dist["weights"]
                        xs_deg = np.linspace(0.5, 179.5, 200)
                        xs_rad = np.radians(xs_deg)
                        xs_cos = np.cos(xs_rad)
                        log_p_cos = np.array(
                            [
                                eval_gmm_log_prob(
                                    x, means, vars_, weights, var_floor=_Q2Q3_GMM_COS_VAR_FLOOR
                                )
                                for x in xs_cos
                            ]
                        )
                        ys = np.exp(log_p_cos) * np.sin(xs_rad)
                        ax.plot(xs_deg, ys, color="#00E5FF")
                        ax.axvline(val_deg, color="#FFFFFF", linestyle="--")
                        ax.set_title(
                            f"{t['label']} Res: {t['residual']:.2f} | Key: {t['resolved_key']}",
                            color="#AAD4FF",
                            fontsize=9,
                        )
                else:
                    ax.text(
                        0.5, 0.5, "No Distribution", color="#FF6B6B", ha="center", va="center"
                    )
                    ax.set_title(
                        f"{t['label']} Res: {t['residual']:.2f} | Key: {t['resolved_key']}",
                        color="#AAD4FF",
                        fontsize=9,
                    )

            for j in range(n_terms, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()

    def on_prev(_):
        state["idx"] = max(0, state["idx"] - 1)
        _render()

    def on_next(_):
        state["idx"] = min(len(all_confs) - 1, state["idx"] + 1)
        _render()

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)

    nav_bar = widgets.HBox(
        [btn_prev, btn_next, hdr_html], layout=widgets.Layout(align_items="center", gap="8px")
    )
    ui = widgets.VBox(
        [nav_bar, plot_out],
        layout=widgets.Layout(
            border="1px solid #333", padding="10px", background_color="#0F0F1A"
        ),
    )
    display(ui)
    _render()

