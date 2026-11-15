"""Figures for the benchmark notebooks (matplotlib, static) and a 3-D overlay page (3Dmol.js, vendored).

Colour follows the configuration, in a fixed order (validated categorical palette, first three slots; with more than
three configurations the figure is split rather than adding hues). Text stays in neutral ink.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # categorical slots 1-3 (light surface)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _plain_log(axis):
    """log axis with plain-number tick labels (10, 30, 100, 300 ...), no minor labels."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    lo, hi = axis.get_view_interval()
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if lo * 0.999 <= t <= hi * 1.001]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(lambda v, _: f"{v:g}")
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def case_dots(df: pd.DataFrame, configs: Sequence[str], group: str = "tmQM", value: str = "absdE",
              title: Optional[str] = None, log: bool = True, refs=(100, 300), ax=None,
              xlabel: str = "|ΔE| vs crystal, kJ/mol (dot: mean over seeds; line: min–max)"):
    """Per case: seed mean (dot) and seed min-max (line) of |dE| for up to three configurations."""
    import matplotlib.pyplot as plt

    if len(configs) > 3:
        raise ValueError("at most three configurations per figure (split into several figures)")
    d = df[(df.group == group) & df.config.isin(configs)]
    stats = d.groupby(["case", "config"])[value].agg(["mean", "min", "max"]).reset_index()
    order = stats[stats.config == configs[0]].sort_values("mean")["case"].tolist()
    order += [c for c in stats.case.unique() if c not in order]
    y = {c: i for i, c in enumerate(order)}
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(order) + 1.4), facecolor=SURFACE)
    _style(ax)
    off = np.linspace(-0.22, 0.22, len(configs)) if len(configs) > 1 else [0.0]
    for k, (cfg, o) in enumerate(zip(configs, off)):
        s = stats[stats.config == cfg]
        yy = np.array([y[c] for c in s.case]) + o
        ax.hlines(yy, s["min"], s["max"], color=SERIES[k], linewidth=2, alpha=0.55)
        ax.plot(s["mean"], yy, "o", ms=6, color=SERIES[k], markeredgecolor=SURFACE, markeredgewidth=1.2, label=cfg)
    for r in refs or ():
        ax.axvline(r, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(r, -0.9, f"{r}", color=INK2, fontsize=8, ha="center", va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color=INK)
    ax.set_ylim(len(order) - 0.4, -1.2)
    if log:
        ax.set_xscale("log")
        _plain_log(ax.xaxis)
    ax.set_xlabel(xlabel)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(configs), labelcolor=INK)
    ax.set_title(title or f"{group}: per-case |ΔE|", color=INK, fontsize=11, loc="left", pad=26)
    return ax


def paired_scatter(df: pd.DataFrame, a: str, b: str, group: str = "tmQM", value: str = "absdE", ax=None):
    """Seed-mean |dE| per case, configuration a (x) vs b (y); below the diagonal = b is better."""
    import matplotlib.pyplot as plt

    m = df[df.group == group].groupby(["case", "config"])[value].mean().unstack("config")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    lo, hi = max(1.0, np.nanmin(m[[a, b]].values) * 0.7), np.nanmax(m[[a, b]].values) * 1.4
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.8)
    ax.scatter(m[a], m[b], s=36, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for c, r in m.iterrows():
        if abs(np.log(max(r[b], 1) / max(r[a], 1))) > np.log(1.5):
            ax.annotate(c, (r[a], r[b]), fontsize=7.5, color=INK2, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    _plain_log(ax.xaxis); _plain_log(ax.yaxis)
    ax.set_xlabel(f"{a}: |ΔE| kJ/mol"); ax.set_ylabel(f"{b}: |ΔE| kJ/mol")
    ax.set_title(f"{group}: {b} vs {a} (below the line = {b} better)", color=INK, fontsize=11, loc="left")
    return ax


def time_dots(df: pd.DataFrame, configs: Sequence[str], ax=None):
    """Solve time per case (seed mean) for up to three configurations."""
    return case_dots(df, configs, group="tmQM", value="solve_s", log=False, refs=(),
                     title="tmQM: solve time per case", ax=ax, xlabel="solve time, s (dot: mean over seeds; line: min–max)")


# ----------------------------------------------------------------------------------------------------------------
# 3-D overlay
# ----------------------------------------------------------------------------------------------------------------
def _read_xyz(p: Path):
    L = [x.split() for x in Path(p).read_text().splitlines()[2:] if x.strip()]
    return [x[0] for x in L], np.array([[float(v) for v in x[1:4]] for x in L])


def _kabsch(P, Q):
    """R, t minimising |R P + t - Q| (rows are points)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def overlay_html(xyz_dir: Path, case: str, out: Path) -> dict:
    """Crystal (blue) and solved best conformer (orange), Kabsch-aligned on the mapped heavy atoms; returns the
    heavy-atom RMSD and writes a self-contained HTML page (3Dmol.js inlined)."""
    xyz_dir = Path(xyz_dir)
    sy, X = _read_xyz(xyz_dir / f"{case}_solved.xyz")
    sr, Xr = _read_xyz(xyz_dir / f"{case}_crystal.xyz")
    amap = {int(k): v for k, v in json.loads((xyz_dir / f"{case}_map.json").read_text()).items()}
    idx = sorted(amap)
    R, t = _kabsch(X[idx], Xr[[amap[i] for i in idx]])
    Xa = X @ R.T + t
    rmsd = float(np.sqrt(((Xa[idx] - Xr[[amap[i] for i in idx]]) ** 2).sum(1).mean()))
    js = (Path(__file__).resolve().parents[3] / "dev" / "tools" / "jdev" / "vendor" / "3Dmol-min.js")
    lib = js.read_text(encoding="utf-8").replace("</script", "<\\/script")
    xyz = lambda s_, P, c: f"{len(s_)}\\n{c}\\n" + "".join(f"{a} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\\n" for a, p in zip(s_, P))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{case} overlay</title><script>{lib}</script>
<style>body{{font-family:system-ui;margin:12px;color:{INK};background:{SURFACE}}}#v{{width:100%;height:560px;position:relative}}
.k{{display:inline-block;width:10px;height:10px;border-radius:5px;margin:0 4px 0 12px}}</style></head><body>
<div><b>{case}</b> heavy-atom RMSD after alignment {rmsd:.3f} Å
<span class="k" style="background:{SERIES[0]}"></span>crystal<span class="k" style="background:{SERIES[1]}"></span>solved (best score)
<label style="margin-left:16px"><input type="checkbox" id="h" checked> hydrogens</label></div><div id="v"></div>
<script>const v=$3Dmol.createViewer('v',{{backgroundColor:'{SURFACE}'}});
function draw(){{v.clear();const showH=document.getElementById('h').checked;
v.addModel("{xyz(sr, Xr, 'crystal')}",'xyz');v.setStyle({{model:0}},{{stick:{{radius:.09,color:'{SERIES[0]}',opacity:.55}}}});
v.addModel("{xyz(sy, Xa, 'solved')}",'xyz');v.setStyle({{model:1}},{{stick:{{radius:.11,color:'{SERIES[1]}'}}}});
if(!showH){{v.setStyle({{model:0,elem:'H'}},{{}});v.setStyle({{model:1,elem:'H'}},{{}});}}
v.zoomTo();v.render();}}document.getElementById('h').onchange=draw;draw();</script></body></html>"""
    Path(out).write_text(html, encoding="utf-8")
    return {"case": case, "heavy_rmsd": rmsd, "html": str(out)}
