"""
visualizer.py
=============
Unified interactive visualizer for the Jacobik conformer ensemble engine.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display

from jacobik.core.types import TMComplex
from jacobik.ik.scorer import _gmm_fields, _vmm_fields
from jacobik.ik.targets import IDEAL_GEOMETRIES
from jacobik.vis.stats_view import (
    _PANEL_STYLE,
    _finalise,
    _make_view,
    _render_complex,
    _row,
)

# ---------------------------------------------------------------------------
# CPK and Shape helpers for custom overlay / mapping rendering
# ---------------------------------------------------------------------------
CPK = {
    "H": "#FFFFFF",
    "C": "#404040",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
    "B": "#FFB5B5",
    "Se": "#FFA100",
}


def _cpk(symbol: str) -> str:
    return CPK.get(symbol, "#FF69B4")


def _xyz(v: np.ndarray) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


# ---------------------------------------------------------------------------
# Custom complex renderer supporting opacity (for overlay)
# ---------------------------------------------------------------------------
def _render_complex_with_opacity(
    view,
    mol: TMComplex,
    opacity: float = 1.0,
    show_hydrogens: bool = False,
    draw_annotations: bool = True,
):
    """
    Renders the complex into a py3Dmol view with a custom opacity.
    If draw_annotations=False, suppresses labels and dashed lines.
    """
    atom_map = {a.idx: a for a in mol.atoms}
    metal_pos = atom_map[mol.metal_idx].position

    donor_labels = getattr(mol, "donor_labels", {})
    h_donor_set: set[int] = {
        idx for idx, lbl in donor_labels.items() if lbl.subclass == "H_hydride"
    }

    # Draw atoms
    for a in mol.atoms:
        if not show_hydrogens and a.symbol == "H" and a.idx not in h_donor_set:
            continue
        view.addSphere(
            {
                "center": _xyz(a.position),
                "radius": 0.22 if a.idx != mol.metal_idx else 0.28,
                "color": _cpk(a.symbol) if a.idx != mol.metal_idx else "#D0D0E0",
                "opacity": opacity,
            }
        )

    # Draw bonds
    metal_bond_pairs = {
        (min(b.i, b.j), max(b.i, b.j))
        for b in mol.bonds
        if b.i == mol.metal_idx or b.j == mol.metal_idx
    }
    for b in mol.bonds:
        key = (min(b.i, b.j), max(b.i, b.j))
        if key in metal_bond_pairs:
            continue
        if b.order < 0.5:
            continue
        ai, aj = atom_map[b.i], atom_map[b.j]
        if not show_hydrogens and (
            (ai.symbol == "H" and b.i not in h_donor_set)
            or (aj.symbol == "H" and b.j not in h_donor_set)
        ):
            continue

        mid = _lerp(ai.position, aj.position, 0.5)
        view.addCylinder(
            {
                "start": _xyz(ai.position),
                "end": _xyz(mid),
                "radius": 0.06,
                "color": _cpk(ai.symbol),
                "opacity": opacity,
            }
        )
        view.addCylinder(
            {
                "start": _xyz(mid),
                "end": _xyz(aj.position),
                "radius": 0.06,
                "color": _cpk(aj.symbol),
                "opacity": opacity,
            }
        )

    # Draw metal-donor dashes, donor circles, and labels
    if draw_annotations:
        for site in mol.coord_sites:
            pos = np.asarray(site.position, dtype=float)
            # Dashed line
            n_beads = 9
            for k in range(n_beads):
                t = (k + 0.5) / n_beads
                view.addSphere(
                    {
                        "center": _xyz(_lerp(pos, metal_pos, t)),
                        "radius": 0.05,
                        "color": "#CCCCCC",
                        "opacity": 0.7 * opacity,
                    }
                )
            # Ghost sphere
            view.addSphere(
                {
                    "center": _xyz(pos),
                    "radius": 0.55 * (1.2 if site.is_haptic else 1.0),
                    "color": "#88DDFF" if site.is_haptic else "#FFFFFF",
                    "opacity": 0.18 * opacity,
                }
            )
            if site.label:
                label_pos = pos + np.array([0.0, 0.7, 0.0])
                view.addLabel(
                    site.label,
                    {
                        "position": _xyz(label_pos),
                        "fontSize": 10,
                        "fontColor": "white",
                        "backgroundColor": "black",
                        "backgroundOpacity": 0.45 * opacity,
                        "borderThickness": 0.0,
                        "inFront": True,
                        "showBackground": True,
                    },
                )


# ---------------------------------------------------------------------------
# GMM/VMM evaluation functions for individual term analysis
# ---------------------------------------------------------------------------
def eval_gmm_log_prob(
    x: float, means: List[float], vars_: List[float], weights: List[float]
) -> float:
    _LOG_2PI = math.log(2 * math.pi)
    log_probs = []
    for m, v, w in zip(means, vars_, weights):
        v = max(v, 0.04)  # variance floor
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


def get_gmm_max_log_prob(means: List[float], vars_: List[float], weights: List[float]) -> float:
    # Evaluate GMM over a dense 1D grid to find the peak value
    stds = [math.sqrt(max(v, 0.04)) for v in vars_]
    grid_min = min(means) - 3.0 * max(stds)
    grid_max = max(means) + 3.0 * max(stds)
    xs = np.linspace(grid_min, grid_max, 500)
    lps = [eval_gmm_log_prob(x, means, vars_, weights) for x in xs]
    return float(max(lps))


def get_vmm_max_log_prob(
    means_rad: List[float], kappas: List[float], weights: List[float]
) -> float:
    # Evaluate VMM over full circle to find the peak value
    xs_rad = np.linspace(0.0, 2.0 * math.pi, 500)
    lps = [eval_vmm_log_prob(x, means_rad, kappas, weights) for x in xs_rad]
    return float(max(lps))


# ---------------------------------------------------------------------------
# Visualizer class definition
# ---------------------------------------------------------------------------
class Visualizer:
    """
    Unified public-facing visualizer for Jacobik conformer ensembles.
    """

    def __init__(
        self,
        result: Optional[Any] = None,
        scorer: Optional[Any] = None,
        show_hydrogen: bool = False,
    ) -> None:
        self.result = result
        self.scorer = scorer
        self.show_hydrogen = show_hydrogen

        # Auto-extract scorer and solve_trace from EnsembleResult if possible
        if result is not None:
            if scorer is None and hasattr(result, "scorer"):
                self.scorer = result.scorer

    def _get_result(self, result: Optional[Any]) -> Any:
        r = result if result is not None else self.result
        if r is None:
            raise ValueError("No EnsembleResult provided. Pass it to the method or constructor.")
        return r

    def _get_scorer(self, scorer: Optional[Any]) -> Any:
        s = scorer if scorer is not None else self.scorer
        if s is None:
            raise ValueError("No Scorer provided. Pass it to the method or constructor.")
        return s

    # -----------------------------------------------------------------------
    # 1. view_ensemble_structs()
    # -----------------------------------------------------------------------
    def view_ensemble_structs(
        self, result: Optional[Any] = None, width: int = 620, height: int = 460
    ) -> None:
        """
        Interactive 3D structure viewer to step through conformers.
        """
        res = self._get_result(result)
        all_confs = res.conformers
        if not all_confs:
            print("EnsembleResult is empty — nothing to display.")
            return

        state = {"idx": 0, "conv_only": False}

        def _visible():
            if state["conv_only"]:
                v = [c for c in all_confs if c.converged]
                return v if v else all_confs
            return all_confs

        # Widgets
        btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
        btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
        hdr_html = widgets.HTML(value="")
        info_html = widgets.HTML(value="")
        cb_conv = widgets.Checkbox(
            value=False,
            description="Converged only",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="160px"),
        )
        viewer_out = widgets.Output()

        def _render():
            vis = _visible()
            idx = max(0, min(state["idx"], len(vis) - 1))
            state["idx"] = idx
            conf = vis[idx]
            mol = conf.complex

            # Custom header HTML
            score_str = f"{conf.score:.4f}" if not math.isnan(conf.score) else "nan"
            conv_lbl = "converged" if conf.converged else "not converged"
            conv_col = "#00FF88" if conf.converged else "#FF6B6B"
            hdr_html.value = (
                f"<b style='color:#AAD4FF'>Conformer Rank {conf.rank} / {len(vis)}</b>"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"<span style='color:#00E5FF'>score {score_str}</span>"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"<span style='color:{conv_col}'>{conv_lbl}</span>"
                f"&nbsp;&nbsp;<span style='color:#888'>({idx + 1} of {len(vis)} shown)</span>"
            )

            # Info panel HTML
            cn = len(mol.coord_sites)
            conv_flag = (
                "<span style='color:#00FF88'>yes</span>"
                if conf.converged
                else "<span style='color:#FF6B6B'>no</span>"
            )
            site_rows = ""
            atom_map = {a.idx: a for a in mol.atoms}
            metal_pos = atom_map[mol.metal_idx].position
            for s_idx, site in enumerate(mol.coord_sites):
                d = float(np.linalg.norm(np.asarray(site.position) - np.asarray(metal_pos)))
                lbl = site.label or f"Site {s_idx}"
                site_rows += _row(f"  {lbl}", f"{d:.3f} Å", "#AAD4FF")

            info_html.value = (
                f"<div style='{_PANEL_STYLE}'>"
                + _row("Rank", str(conf.rank))
                + _row("Score", score_str, "#00E5FF")
                + _row("Converged", conv_flag)
                + _row("Metal", f"{mol.metal_symbol}  ({mol.coord_geom or '?'}  CN={cn})")
                + _row("M–D bonds", "")
                + site_rows
                + "</div>"
            )

            btn_prev.disabled = idx == 0
            btn_next.disabled = idx == len(vis) - 1

            with viewer_out:
                viewer_out.clear_output(wait=True)
                v = _make_view(width, height)
                _render_complex(v, mol, show_hydrogens=self.show_hydrogen)
                _finalise(v)
                display(v)

        def on_prev(_):
            state["idx"] = max(0, state["idx"] - 1)
            _render()

        def on_next(_):
            state["idx"] = min(len(_visible()) - 1, state["idx"] + 1)
            _render()

        def on_conv(change):
            state["conv_only"] = change["new"]
            state["idx"] = 0
            _render()

        btn_prev.on_click(on_prev)
        btn_next.on_click(on_next)
        cb_conv.observe(on_conv, names="value")

        nav_bar = widgets.HBox(
            [btn_prev, btn_next, hdr_html], layout=widgets.Layout(align_items="center", gap="8px")
        )
        controls = widgets.HBox([cb_conv], layout=widgets.Layout(margin="4px 0px"))
        body = widgets.HBox(
            [viewer_out, info_html], layout=widgets.Layout(align_items="flex-start", gap="12px")
        )

        ui = widgets.VBox(
            [nav_bar, controls, body],
            layout=widgets.Layout(
                border="1px solid #333", padding="10px", background_color="#0F0F1A"
            ),
        )
        display(ui)
        _render()

    # -----------------------------------------------------------------------
    # 2. view_ensemble_stats()
    # -----------------------------------------------------------------------
    def view_ensemble_stats(
        self, result: Optional[Any] = None, scorer: Optional[Any] = None
    ) -> None:
        """
        Interactive tool to view FittedDist satisfaction details.
        """
        res = self._get_result(result)
        scr = self._get_scorer(scorer)

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

                    fields = _vmm_fields(fd, q2_key, "Q2")
                    if fields is not None:
                        means, kappas, weights = fields
                        means_rad = [math.radians(m) for m in means]
                        lp = eval_vmm_log_prob(q2_angles[p], means_rad, kappas, weights)
                        max_lp = get_vmm_max_log_prob(means_rad, kappas, weights)
                        residual = max(0.0, max_lp - lp)
                    else:
                        residual = 0.0
                        means, kappas, weights = None, None, None

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
                                "type": "VMM",
                                "means": means_rad,
                                "kappas": kappas,
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

                    fields = _vmm_fields(fd, key, "Q3")
                    if fields is not None:
                        means, kappas, weights = fields
                        means_rad = [math.radians(m) for m in means]
                        lp = eval_vmm_log_prob(q3_a[q3_idx], means_rad, kappas, weights)
                        max_lp = get_vmm_max_log_prob(means_rad, kappas, weights)
                        residual = max(0.0, max_lp - lp)
                    else:
                        residual = 0.0
                        means, kappas, weights = None, None, None

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
                                "type": "VMM",
                                "means": means_rad,
                                "kappas": kappas,
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

    # -----------------------------------------------------------------------
    # 3. view_ranking()
    # -----------------------------------------------------------------------
    def view_ranking(self, result: Optional[Any] = None) -> None:
        """
        Plots a bar chart of the ensemble scores, ranked highest to lowest.
        """
        res = self._get_result(result)
        all_confs = res.conformers
        if not all_confs:
            print("EnsembleResult is empty.")
            return

        ranks = [c.rank for c in all_confs]
        scores = [c.score for c in all_confs]
        conv_list = [c.converged for c in all_confs]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0F0F1A")
        ax.set_facecolor("#0F0F1A")

        # Color code bars by convergence status
        colors = ["#00FF88" if conv else "#FF6B6B" for conv in conv_list]

        bars = ax.bar(ranks, scores, color=colors, edgecolor="#333355", width=0.6)
        ax.set_xlabel("Conformer Rank", color="#888888", fontsize=9)
        ax.set_ylabel("Score (Log Probability)", color="#888888", fontsize=9)
        ax.set_title(
            "Ensemble Conformer Rankings (Green = Converged, Red = Failed)",
            color="#AAD4FF",
            fontsize=11,
            pad=12,
        )
        ax.set_xticks(ranks)
        ax.tick_params(colors="#666666", labelsize=8)
        ax.grid(axis="y", color="#22223B", linestyle="--")

        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        # Text labels on top of bars
        for bar, score in zip(bars, scores):
            yval = bar.get_height()
            offset = 0.5 if yval >= 0 else -1.5
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + offset,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                color="#C8D8E8",
                fontsize=7,
            )

        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # 4. view_overlay()
    # -----------------------------------------------------------------------
    def view_overlay(
        self, result: Optional[Any] = None, width: int = 620, height: int = 460
    ) -> None:
        """
        Interactive viewer showing all conformers overlayed ghostly transparent,
        with the currently selected conformer drawn in full color.
        """
        res = self._get_result(result)
        all_confs = res.conformers
        if not all_confs:
            print("EnsembleResult is empty.")
            return

        state = {"idx": 0}

        # Widgets
        btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
        btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
        hdr_html = widgets.HTML(value="")
        viewer_out = widgets.Output()

        def _render():
            idx = state["idx"]
            active_conf = all_confs[idx]

            hdr_html.value = (
                f"<b style='color:#AAD4FF'>Active Conformer Rank {active_conf.rank} / {len(all_confs)}</b>"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;<span style='color:#00E5FF'>score {active_conf.score:.4f}</span>"
                f"&nbsp;&nbsp;<span style='color:#888'>(showing other conformers as ghosts)</span>"
            )
            btn_prev.disabled = idx == 0
            btn_next.disabled = idx == len(all_confs) - 1

            with viewer_out:
                viewer_out.clear_output(wait=True)
                v = _make_view(width, height)

                # Render ghosts first (low opacity, no labels/annotations to prevent clutter)
                for ci, conf in enumerate(all_confs):
                    if ci == idx:
                        continue
                    _render_complex_with_opacity(
                        v,
                        conf.complex,
                        opacity=0.12,
                        show_hydrogens=self.show_hydrogen,
                        draw_annotations=False,
                    )

                # Render active in full color on top (with metal-donor lines and labels)
                _render_complex_with_opacity(
                    v,
                    active_conf.complex,
                    opacity=1.0,
                    show_hydrogens=self.show_hydrogen,
                    draw_annotations=True,
                )

                _finalise(v)
                display(v)

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
            [nav_bar, viewer_out],
            layout=widgets.Layout(
                border="1px solid #333", padding="10px", background_color="#0F0F1A"
            ),
        )
        display(ui)
        _render()

    # -----------------------------------------------------------------------
    # 5. view_solve()
    # -----------------------------------------------------------------------
    def view_solve(self, result: Optional[Any] = None) -> None:
        """
        Interactive solver trajectory dashboard showing residuals, scores,
        donor errors, and gradient norms as a function of steps.
        """
        res = self._get_result(result)
        trace = getattr(res, "solve_trace", None)
        if trace is None or not trace:
            print("Solve trace is not available on this result.")
            return

        all_confs = res.conformers
        if not all_confs:
            print("EnsembleResult is empty.")
            return

        # trace contains arrays of shape (T, B) where B is the number of trials
        T, B = trace["residual"].shape
        state = {"idx": 0}

        # Widgets
        btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
        btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
        hdr_html = widgets.HTML(value="")
        plot_out = widgets.Output()

        def _render():
            idx = state["idx"]
            active_conf = all_confs[idx]
            trial_idx = active_conf.trial_idx

            hdr_html.value = (
                f"<b style='color:#AAD4FF'>Showing Solve Trajectory for Rank {active_conf.rank}</b>"
                f"&nbsp;&nbsp;<span style='color:#888'>(original trial element index = {trial_idx})</span>"
            )
            btn_prev.disabled = idx == 0
            btn_next.disabled = idx == len(all_confs) - 1

            with plot_out:
                plot_out.clear_output(wait=True)
                fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="#0F0F1A")
                steps = np.arange(T)

                # Helper to plot background trials and highlight the active one
                def plot_metric(ax, data_key, label_y, title):
                    ax.set_facecolor("#0F0F1A")
                    data = trace[data_key]  # (T, B)
                    # Plot all other trials in background
                    for b in range(B):
                        if b != trial_idx:
                            ax.plot(steps, data[:, b], color="#22223B", alpha=0.4, linewidth=0.8)
                    # Highlight active trial in cyan
                    ax.plot(
                        steps,
                        data[:, trial_idx],
                        color="#00E5FF",
                        linewidth=1.8,
                        label=f"Rank {active_conf.rank}",
                    )
                    ax.set_title(title, color="#AAD4FF", fontsize=9, pad=5)
                    ax.set_xlabel("Iteration step", color="#666666", fontsize=7)
                    ax.set_ylabel(label_y, color="#666666", fontsize=7)
                    ax.tick_params(colors="#666666", labelsize=7)
                    ax.grid(color="#1A1A2E", linestyle=":")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#333355")
                    ax.legend(
                        fontsize=7, facecolor="#1A1A2E", labelcolor="#C8D8E8", edgecolor="#333355"
                    )

                plot_metric(
                    axes[0, 0], "residual", "Residual Loss", "Joint Residual (-score + λ*anchor)"
                )
                plot_metric(
                    axes[0, 1],
                    "score",
                    "Log Prob (Score)",
                    "Chemical Statistics Score (higher = better)",
                )
                plot_metric(
                    axes[1, 0],
                    "worst_error",
                    "Max Donor Error (Å)",
                    "Worst-Ligand Target Position Placement Error",
                )
                plot_metric(
                    axes[1, 1], "grad_norm", "Gradient Norm", "Parameter Gradient Magnitude"
                )

                fig.suptitle(
                    f"Optimization Trajectory Dashboard (CSD: {active_conf.complex.csd_code})",
                    color="#AAD4FF",
                    fontsize=11,
                    y=0.98,
                )
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

    # -----------------------------------------------------------------------
    # 6. geometry_mappings()
    # -----------------------------------------------------------------------
    def geometry_mappings(self, width: int = 600, height: int = 460) -> None:
        """
        Interactive tool to step through and view index order mappings of
        ideal coordination geometries.
        """
        geom_names = sorted(IDEAL_GEOMETRIES.keys())

        # Widgets
        drop_geom = widgets.Dropdown(
            options=geom_names, value="octahedral", description="Geometry:"
        )
        info_html = widgets.HTML(value="")
        viewer_out = widgets.Output()

        def _render(*args):
            geom_name = drop_geom.value
            coords = IDEAL_GEOMETRIES[geom_name]  # (CN, 3)
            cn = coords.shape[0]

            # Custom text description
            desc = f"<div style='{_PANEL_STYLE}'>"
            desc += f"<b style='color:#00E5FF'>{geom_name.upper()}</b><br>"
            desc += f"Coordination Number: {cn}<br><br>"
            desc += "<b>Ideal Site Mappings:</b><br>"
            for idx, vec in enumerate(coords):
                desc += _row(
                    f"Site {idx}", f"({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})", "#AAD4FF"
                )
            desc += "</div>"
            info_html.value = desc

            with viewer_out:
                viewer_out.clear_output(wait=True)
                v = _make_view(width, height)

                # Draw fake metal at origin
                v.addSphere(
                    {
                        "center": {"x": 0, "y": 0, "z": 0},
                        "radius": 0.28,
                        "color": "#E6A23C",  # Gold metal atom representation
                        "opacity": 1.0,
                    }
                )
                v.addLabel(
                    "M",
                    {
                        "position": {"x": 0, "y": 0.4, "z": 0},
                        "fontSize": 10,
                        "fontColor": "white",
                        "backgroundColor": "black",
                        "backgroundOpacity": 0.5,
                        "borderThickness": 0.0,
                        "inFront": True,
                        "showBackground": True,
                    },
                )

                # Draw sites at direction * 2.0 Å
                for idx, dir_vec in enumerate(coords):
                    pos = dir_vec * 2.0
                    # Draw site sphere
                    v.addSphere(
                        {
                            "center": _xyz(pos),
                            "radius": 0.20,
                            "color": "#00E5FF",
                            "opacity": 0.9,
                        }
                    )
                    # Draw label showing site index
                    v.addLabel(
                        f"Site {idx}",
                        {
                            "position": _xyz(pos + np.array([0.0, 0.4, 0.0])),
                            "fontSize": 10,
                            "fontColor": "white",
                            "backgroundColor": "black",
                            "backgroundOpacity": 0.5,
                            "borderThickness": 0.0,
                            "inFront": True,
                            "showBackground": True,
                        },
                    )
                    # Draw connecting line
                    n_beads = 9
                    for k in range(n_beads):
                        t = (k + 0.5) / n_beads
                        v.addSphere(
                            {
                                "center": _xyz(_lerp(pos, np.zeros(3), t)),
                                "radius": 0.04,
                                "color": "#AAAAAA",
                                "opacity": 0.6,
                            }
                        )

                _finalise(v)
                display(v)

        drop_geom.observe(_render, names="value")

        controls = widgets.HBox([drop_geom], layout=widgets.Layout(margin="4px 0px"))
        body = widgets.HBox(
            [viewer_out, info_html], layout=widgets.Layout(align_items="flex-start", gap="12px")
        )

        ui = widgets.VBox(
            [controls, body],
            layout=widgets.Layout(
                border="1px solid #333", padding="10px", background_color="#0F0F1A"
            ),
        )
        display(ui)
        _render()

    def view_ensemble_heatmap(self, result: Optional[Any] = None) -> None:
        """
        Draws a heatmap of the pairwise RMSD between conformers in the ensemble.
        """
        res = self._get_result(result)
        all_confs = res.conformers
        if not all_confs:
            print("EnsembleResult is empty — nothing to display.")
            return

        n = len(all_confs)
        rmsd_matrix = np.zeros((n, n))

        coords = np.array(
            [[a.position for a in conf.complex.atoms] for conf in all_confs]
        )  # (n, N_atoms, 3)

        for i in range(n):
            for j in range(i + 1, n):
                diff = coords[i] - coords[j]
                rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=-1)))
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0F0F1A")
        ax.set_facecolor("#0F0F1A")

        im = ax.imshow(rmsd_matrix, cmap="magma", origin="upper")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="#C8D8E8", labelcolor="#C8D8E8")
        cbar.set_label("RMSD (Å)", color="#C8D8E8")

        ax.set_title("Pairwise Conformer RMSD", color="#AAD4FF", fontsize=12)
        ax.set_xlabel("Conformer Index", color="#C8D8E8")
        ax.set_ylabel("Conformer Index", color="#C8D8E8")
        ax.tick_params(colors="#C8D8E8")

        if n <= 15:
            for i in range(n):
                for j in range(n):
                    color = "white" if rmsd_matrix[i, j] < (rmsd_matrix.max() / 2) else "black"
                    if rmsd_matrix[i, j] > 0:
                        ax.text(
                            j,
                            i,
                            f"{rmsd_matrix[i, j]:.1f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=8,
                        )

        plt.tight_layout()
        plt.show()
