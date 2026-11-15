"""
render_3d.py
============
3D structure, overlay, and geometry-mapping viewers for :class:`jacobik.vis.visualizer.Visualizer`.
"""

from __future__ import annotations

import math
from typing import Any

import ipywidgets as widgets
import numpy as np
from IPython.display import display

from jacobik.core.types import TMComplex
from jacobik.ik.targets import IDEAL_GEOMETRIES
from jacobik.vis.stats_view import (
    _PANEL_STYLE,
    _finalise,
    _lerp,
    _make_view,
    _render_complex,
    _row,
    _xyz,
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

            # Draw Hungarian matching ghostly anchor and vector
            if getattr(site, "anchor", None) is not None:
                anchor_pos = metal_pos + np.asarray(site.anchor, dtype=float)
                # Ghost sphere for anchor
                view.addSphere(
                    {
                        "center": _xyz(anchor_pos),
                        "radius": 0.45 * (1.2 if site.is_haptic else 1.0),
                        "color": "#FFA500",  # warm orange/gold color for anchor
                        "opacity": 0.25 * opacity,
                    }
                )
                # Vector arrow from donor position to anchor
                view.addArrow(
                    {
                        "start": _xyz(pos),
                        "end": _xyz(anchor_pos),
                        "radius": 0.05,
                        "color": "#FFA500",
                        "opacity": 0.6 * opacity,
                    }
                )



def show_ensemble_structs(
    res: Any, show_hydrogen: bool = False, width: int = 620, height: int = 460
) -> None:
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
            _render_complex(v, mol, show_hydrogens=show_hydrogen)
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



def show_overlay(
    res: Any, show_hydrogen: bool = False, width: int = 620, height: int = 460
) -> None:
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
                    show_hydrogens=show_hydrogen,
                    draw_annotations=False,
                )

            # Render active in full color on top (with metal-donor lines and labels)
            _render_complex_with_opacity(
                v,
                active_conf.complex,
                opacity=1.0,
                show_hydrogens=show_hydrogen,
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



def show_geometry_mappings(width: int = 600, height: int = 460) -> None:
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
