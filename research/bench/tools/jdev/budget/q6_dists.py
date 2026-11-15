"""What the Q6 PMF tables are made of (2026-10-16 f): per key, the crystal distance distribution p_key(r), the pooled
reference p_ref(r) of its class (every key of the class summed, the PMF's denominator), w = -ln(p_key / p_ref) and the
table as the scorer uses it (max(w, 0) for non-polar keys), against the 5th-percentile wall.

Also, per key: the mean table value a CRYSTAL pair of that key pays (sum over the key's own distribution of
max(w, 0)), split below / above 3.5 A. For a pair potential that only describes contacts this would be ~0 beyond
contact.

    python tools/jdev/budget/q6_dists.py [--png data/inspect/q6pmf_tables.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
from research.archive.code.interactions import q6pmf as Q  # noqa: E402

OBS = REPO / "data/refit/observations"
DB = REPO / "data/refit/final/constraints_2026-10-17.json"
PANELS = [("C_H", "5"), ("H_H", "5"), ("C_C", "5"), ("H_H", "4"), ("C_N", "4"), ("Hp_O", "5")]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dc"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--png", default=str(REPO / "data/inspect/q6pmf_tables.png"))
    a = p.parse_args()
    hist = Q.load_hist(sorted(OBS.glob("q6hist_*.npz")))
    db = json.loads(DB.read_text())
    rc = 0.5 * (Q.EDGES[1:] + Q.EDGES[:-1])
    ref = {}
    for lab, h in hist.items():
        name, cls, ang = lab.split("|")
        if ang == "" and cls not in Q.M13_CLASSES:
            ref[cls] = ref.get(cls, 0) + h
    tab = lambda name, cls: db.get(f"Q6P:{tuple(name.split('_')) + (cls, '')!r}")
    wall = lambda name: db.get(f"Q6:{tuple(name.split('_'))!r}")

    # what composes the pooled reference
    for cls in ("4", "5"):
        tot = ref[cls].sum()
        parts = sorted(((h.sum() / tot, lab.split("|")[0]) for lab, h in hist.items()
                        if lab.endswith(f"|{cls}|")), reverse=True)[:5]
        print(f"class {cls}: reference = {tot:.3g} pairs; largest shares " +
              ", ".join(f"{k} {s:.2f}" for s, k in parts))

    print("\nmean table value per crystal pair of the key (its own distribution), kT; share of the key's pairs > 3.5 A")
    print(f"  {'key':10s} {'n pairs':>10s} {'r0':>5s}  {'< 3.5 A':>8s} {'3.5-6 A':>8s}  {'pairs > 3.5':>11s}   as used (clamped) | unclamped")
    rows = []
    for lab, h in hist.items():
        name, cls, ang = lab.split("|")
        e = tab(name, cls)
        if ang or e is None or cls not in ("4", "5", "m") or h.sum() < 1e6:
            continue
        g = e["r_lo"] + e["dr"] * np.arange(len(e["w"]))
        w = np.interp(rc, g, e["w"], right=0.0)
        wc = w if "Hp" in name else np.maximum(w, 0.0)
        pk = h / h.sum()
        near, far = rc < 3.5, (rc >= 3.5) & (rc < 6.0)
        rows.append((name, cls, h.sum(), e["r0"], (pk * wc)[near].sum(), (pk * wc)[far].sum(), pk[rc >= 3.5].sum(),
                     (pk * w)[rc < 6].sum()))
    for r in sorted(rows, key=lambda r: -r[2])[:16]:
        print(f"  {r[0] + '|' + r[1]:10s} {r[2]:10.3g} {r[3]:5.2f}  {r[4]:+8.3f} {r[5]:+8.3f}  {r[6]:11.2f}   "
              f"{r[4] + r[5]:+8.3f}          | {r[7]:+8.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, len(PANELS), figsize=(3.0 * len(PANELS), 6.2), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1]})
    for j, (name, cls) in enumerate(PANELS):
        h = hist[f"{name}|{cls}|"]
        e, wl = tab(name, cls), wall(name)
        dens = lambda x: Q._smooth(x) / x.sum() / Q.DR_HIST
        top, bot = ax[0, j], ax[1, j]
        top.plot(rc, dens(h), color=BLUE, lw=2, label=f"{name} pairs")
        top.plot(rc, dens(ref[cls]), color=ORANGE, lw=2, label=f"all class-{cls} pairs (reference)")
        top.set_title(f"{name}, class {cls}  (n = {h.sum():.2g})", color=INK, fontsize=9)
        top.set_xlim(1.5, 6.5)
        top.grid(color=GRID, lw=0.6)
        g = e["r_lo"] + e["dr"] * np.arange(len(e["w"]))
        w = np.array(e["w"])
        bot.axhline(0, color=MUTED, lw=0.8)
        bot.plot(g, w, color=BLUE, lw=1.2, ls=":", label="w = -ln(p_key / p_ref)")
        bot.plot(g, w if "Hp" in name else np.maximum(w, 0), color=BLUE, lw=2, label="table as used")
        if wl:
            bot.plot(g, np.clip(wl["mu"] - g, 0, None) ** 2 / (2 * wl["sigma"] ** 2), color=ORANGE, lw=2,
                     label=f"wall (mu {wl['mu']:.2f})")
        bot.axvline(e["r0"], color=MUTED, lw=0.8, ls="--")
        bot.set_ylim(-3, 6)
        bot.grid(color=GRID, lw=0.6)
        bot.set_xlabel("distance (A)")
        if j == 0:
            top.set_ylabel("density (1/A)")
            bot.set_ylabel("kT per pair")
            top.legend(frameon=False, fontsize=8, loc="upper left")
            bot.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Q6 PMF: crystal distance distributions (top) and the resulting tables (bottom); dashed line = onset r0",
                 color=INK, fontsize=10)
    fig.tight_layout()
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.png, dpi=130)
    print(f"\nfigure: {a.png}")


if __name__ == "__main__":
    main()
