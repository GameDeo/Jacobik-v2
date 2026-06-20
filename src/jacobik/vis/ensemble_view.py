"""
ensemble_view.py
================
Version: 1.3.0
"""

from __future__ import annotations

import math

import ipywidgets as widgets
import numpy as np
from IPython.display import display

from jacobik.vis.stats_view import (
    _PANEL_STYLE,
    _Q_COLOR,
    _add_qty_dot,
    _add_qty_label,
    _finalise,
    _make_view,
    _render_complex,
    _row,
)

# ---------------------------------------------------------------------------
# Info panel
# ---------------------------------------------------------------------------


def _ensemble_info_html(conf) -> str:
    mol = conf.complex
    cn = len(mol.coord_sites)
    conv = (
        "<span style='color:#00FF88'>yes</span>"
        if conf.converged
        else "<span style='color:#FF6B6B'>no</span>"
    )
    score_str = f"{conf.score:.4f}" if not math.isnan(conf.score) else "nan"

    site_rows = ""
    atom_map = {a.idx: a for a in mol.atoms}
    metal_pos = atom_map[mol.metal_idx].position
    for site in mol.coord_sites:
        d = float(np.linalg.norm(np.asarray(site.position) - np.asarray(metal_pos)))
        lbl = site.label or "?"
        site_rows += _row(f"  {lbl}", f"{d:.3f} Å", "#AAD4FF")

    return (
        f"<div style='{_PANEL_STYLE}'>"
        + _row("Rank", str(conf.rank))
        + _row("Score", score_str, "#00E5FF")
        + _row("Converged", conv)
        + _row("Metal", f"{mol.metal_symbol}  ({mol.coord_geom or '?'}  CN={cn})")
        + _row("Ligands", str(len(mol.ligands)))
        + _row("M–D bonds", "")
        + site_rows
        + "</div>"
    )


def _ensemble_header_html(conf, idx, total):
    mol = conf.complex
    score = f"{conf.score:.4f}" if not math.isnan(conf.score) else "nan"
    conv = "converged" if conf.converged else "not converged"
    color = "#00FF88" if conf.converged else "#FF6B6B"
    return (
        f"<b style='color:#AAD4FF'>Rank {conf.rank} / {total}</b>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<span style='color:#00E5FF'>score {score}</span>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<span style='color:{color}'>{conv}</span>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<span style='color:#C8D8E8'>{mol.metal_symbol} {mol.coord_geom or ''}</span>"
        f"&nbsp;&nbsp;<span style='color:#888'>({idx + 1} of {total} shown)</span>"
    )


# ---------------------------------------------------------------------------
# Q1 3-D overlay
# ---------------------------------------------------------------------------


def _render_q1(view, mol) -> None:
    color = _Q_COLOR["Q1"]
    atom_map = {a.idx: a for a in mol.atoms}
    metal_pos = np.asarray(atom_map[mol.metal_idx].position, dtype=float)
    for site in mol.coord_sites:
        pos = np.asarray(site.position, dtype=float)
        dist = float(np.linalg.norm(pos - metal_pos))
        lbl = site.label or "?"
        _add_qty_dot(view, pos, color)
        _add_qty_label(view, pos, f"{lbl}\n{dist:.3f} Å", color)


# ---------------------------------------------------------------------------
# Distribution panel  (matplotlib, shown below the 3-D viewer)
# ---------------------------------------------------------------------------


def _draw_distributions(conf, scorer, out_widget):
    """Render Q1 and Q2 fitted distributions with current values marked."""
    try:
        import matplotlib

        matplotlib.use("module://ipympl.backend_nbagg")  # interactive if available
    except Exception:
        pass
    import matplotlib.pyplot as plt

    mol = conf.complex
    atom_map = {a.idx: a for a in mol.atoms}
    metal_pos = np.asarray(atom_map[mol.metal_idx].position, dtype=float)

    # ── Collect Q1 data ──────────────────────────────────────────────────
    q1_entries = []  # (label, actual_dist, means, vars_, weights)
    n_q1 = scorer._q1_means.shape[0]
    for d in range(n_q1):
        site = mol.coord_sites[d] if d < len(mol.coord_sites) else None
        lbl = (site.label or f"donor {d}") if site else f"donor {d}"
        pos = np.asarray(site.position, dtype=float) if site else np.zeros(3)
        dist = float(np.linalg.norm(pos - metal_pos))
        means = scorer._q1_means[d].cpu().numpy()
        vars_ = scorer._q1_vars[d].cpu().numpy()
        weights = scorer._q1_weights[d].cpu().numpy()
        q1_entries.append((lbl, dist, means, vars_, weights))

    # ── Collect Q2 data ──────────────────────────────────────────────────
    q2_entries = []  # (label, actual_angle_deg, means_deg, kappas, weights)
    n_pairs = scorer._q2_means.shape[0]
    sites = mol.coord_sites
    for p, (si, sj, dist_cls) in enumerate(scorer._q2_pair_meta[:n_pairs]):
        pa = np.asarray(sites[si].position, dtype=float)
        pb = np.asarray(sites[sj].position, dtype=float)
        va, vb = pa - metal_pos, pb - metal_pos
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-8 or nb < 1e-8:
            continue
        cos_t = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_t)))
        la = sites[si].label or f"D{si}"
        lb = sites[sj].label or f"D{sj}"
        means_deg = np.degrees(scorer._q2_means[p].cpu().numpy())
        kappas = scorer._q2_kappas[p].cpu().numpy()
        weights_arr = scorer._q2_weights[p].cpu().numpy()
        q2_entries.append((f"{la}–M–{lb}", angle_deg, means_deg, kappas, weights_arr))

    total_plots = len(q1_entries) + len(q2_entries)
    if total_plots == 0:
        return

    ncols = min(4, total_plots)
    nrows = math.ceil(total_plots / ncols)

    with out_widget:
        out_widget.clear_output(wait=True)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols, 2.8 * nrows),
            facecolor="#0F0F1A",
        )
        axes_flat = np.array(axes).flatten() if total_plots > 1 else [axes]

        # ── Q1 subplots ──────────────────────────────────────────────────
        _LOG_2PI = math.log(2 * math.pi)
        for i, (lbl, actual_d, means, vars_, weights) in enumerate(q1_entries):
            ax = axes_flat[i]
            ax.set_facecolor("#0F0F1A")

            # x range: cover distribution support plus actual value
            x_lo = max(0.5, min(means) - 4 * math.sqrt(max(vars_) + 1e-8))
            x_hi = max(means) + 4 * math.sqrt(max(vars_) + 1e-8)
            x_hi = max(x_hi, actual_d + 0.3)
            x_lo = min(x_lo, actual_d - 0.3)
            xs = np.linspace(x_lo, x_hi, 300)

            # GMM density
            density = np.zeros_like(xs)
            active = weights > 1e-6
            for k in range(len(means)):
                if not active[k]:
                    continue
                var_k = max(vars_[k], 0.04)  # same floor as solver
                log_p = -0.5 * _LOG_2PI - 0.5 * math.log(var_k) - 0.5 * (xs - means[k]) ** 2 / var_k
                density += weights[k] * np.exp(log_p)

            ax.plot(xs, density, color=_Q_COLOR["Q1"], linewidth=1.5)
            ax.axvline(
                actual_d,
                color="#FFFFFF",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label=f"{actual_d:.3f} Å",
            )
            ax.set_title(f"Q1  {lbl}", color="#C8D8E8", fontsize=9, pad=3)
            ax.set_xlabel("M–D distance (Å)", color="#888888", fontsize=7)
            ax.set_ylabel("density", color="#888888", fontsize=7)
            ax.tick_params(colors="#666666", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            ax.legend(fontsize=7, facecolor="#1A1A2E", labelcolor="#C8D8E8", edgecolor="#333355")

        # ── Q2 subplots ──────────────────────────────────────────────────
        offset = len(q1_entries)
        for j, (lbl, actual_deg, means_deg, kappas, weights_arr) in enumerate(q2_entries):
            ax = axes_flat[offset + j]
            ax.set_facecolor("#0F0F1A")

            xs_deg = np.linspace(0, 180, 360)
            xs_rad = np.radians(xs_deg)
            density = np.zeros_like(xs_rad)
            active = weights_arr > 1e-6
            for k in range(len(means_deg)):
                if not active[k]:
                    continue
                kappa = float(kappas[k])
                mu_rad = math.radians(means_deg[k])
                # See exp.md for detailed rationale
                lp = kappa * np.cos(xs_rad - mu_rad)
                density += weights_arr[k] * np.exp(lp - lp.max())  # relative
            if density.max() > 0:
                density /= density.max()  # normalise to 1 for display

            ax.plot(xs_deg, density, color=_Q_COLOR["Q2"], linewidth=1.5)
            ax.axvline(
                actual_deg,
                color="#FFFFFF",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label=f"{actual_deg:.1f}°",
            )
            ax.set_title(f"Q2  {lbl}", color="#C8D8E8", fontsize=9, pad=3)
            ax.set_xlabel("D–M–D angle (°)", color="#888888", fontsize=7)
            ax.set_ylabel("rel. density", color="#888888", fontsize=7)
            ax.set_xlim(0, 180)
            ax.tick_params(colors="#666666", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            ax.legend(fontsize=7, facecolor="#1A1A2E", labelcolor="#C8D8E8", edgecolor="#333355")

        # Hide unused axes
        for ax in axes_flat[total_plots:]:
            ax.set_visible(False)

        fig.suptitle(
            "Fitted distributions  (dashed = current conformer)",
            color="#AAD4FF",
            fontsize=10,
            y=1.01,
        )
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def ensemble_view(
    result,
    scorer=None,
    width: int = 620,
    height: int = 460,
    show_q1: bool = False,
) -> None:
    """
    Interactive viewer for an EnsembleResult from ComplexBuilder.solve().

    Parameters
    ----------
    result   : EnsembleResult returned by builder.solve()
    scorer   : Scorer instance from the solve (scorer_joint).
               When provided, a distribution panel is shown below the 3-D
               viewer with Q1 GMM curves and Q2 VMM curves, with the current
               conformer's actual values marked.
    width    : viewer width in pixels
    height   : viewer height in pixels
    show_q1  : start with Q1 bond-length overlay enabled
    """
    all_confs = result.conformers
    if not all_confs:
        print("EnsembleResult is empty — nothing to display.")
        return

    state = {"idx": 0, "conv_only": False, "show_q1": show_q1}

    def _visible():
        if state["conv_only"]:
            v = [c for c in all_confs if c.converged]
            return v if v else all_confs
        return all_confs

    # ── widgets ──────────────────────────────────────────────────────────────
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
    cb_q1 = widgets.Checkbox(
        value=show_q1,
        description="Q1 bond lengths",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="160px"),
    )
    viewer_out = widgets.Output()
    dist_out = widgets.Output()  # distribution panel lives here

    # ── render ───────────────────────────────────────────────────────────────
    def _render():
        vis = _visible()
        idx = max(0, min(state["idx"], len(vis) - 1))
        state["idx"] = idx
        conf = vis[idx]
        mol = conf.complex

        hdr_html.value = _ensemble_header_html(conf, idx, len(vis))
        info_html.value = _ensemble_info_html(conf)
        btn_prev.disabled = idx == 0
        btn_next.disabled = idx == len(vis) - 1

        with viewer_out:
            viewer_out.clear_output(wait=True)
            v = _make_view(width, height)
            _render_complex(v, mol)
            if state["show_q1"]:
                _render_q1(v, mol)
            _finalise(v)
            display(v)

        if scorer is not None:
            _draw_distributions(conf, scorer, dist_out)

    # ── callbacks ────────────────────────────────────────────────────────────
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

    def on_q1(change):
        state["show_q1"] = change["new"]
        _render()

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    cb_conv.observe(on_conv, names="value")
    cb_q1.observe(on_q1, names="value")

    # ── layout ───────────────────────────────────────────────────────────────
    nav_bar = widgets.HBox(
        [btn_prev, btn_next, hdr_html], layout=widgets.Layout(align_items="center", gap="8px")
    )
    controls = widgets.HBox([cb_conv, cb_q1], layout=widgets.Layout(gap="16px", margin="4px 0px"))
    body = widgets.HBox(
        [viewer_out, info_html], layout=widgets.Layout(align_items="flex-start", gap="12px")
    )

    children = [nav_bar, controls, body]
    if scorer is not None:
        children.append(dist_out)

    ui = widgets.VBox(
        children,
        layout=widgets.Layout(border="1px solid #333", padding="10px", background_color="#0F0F1A"),
    )
    display(ui)
    _render()


# ---------------------------------------------------------------------------
# Standalone TMComplex viewer
# ---------------------------------------------------------------------------


def show_complex(
    m, width: int = 600, height: int = 450, title: str | None = None, show_hydrogens: bool = True
) -> None:
    """
    Render a single TMComplex without the ensemble UI.
    """
    if title:
        print(title)
    v = _make_view(width, height)
    _render_complex(v, m, show_hydrogens=show_hydrogens)
    _finalise(v)
    display(v)
