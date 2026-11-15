"""Report of coord_audit.py (etkdg and full_oracle starts): where the coordination-sphere error of the #1 picks comes
from. A donor is "off" when |solved - crystal| > 0.05 A (Q1); a donor pair when > 5 deg (Q2). For an off item:
    statistic prefers crystal  (log p(crystal) > log p(solved)): the table points the right way, the solver did not get
                               there (competing terms, frozen ligand internals such as bite angles, optimisation);
    statistic prefers solved   : the table itself points away from the crystal (key too coarse / pooled modes)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load import OUT  # noqa: E402

TOL = {"Q1": 0.05, "Q2": 5.0}


def summarize(df: pd.DataFrame, label: str):
    print(f"\n=== {label}: {df.case.nunique()} complexes")
    for q, g in df.groupby("q"):
        err = (g.solved - g.crystal).abs()
        off = g[err > TOL[q]]
        pref_c = (off.lp_crystal > off.lp_solved).mean() if len(off) else np.nan
        unit = "A" if q == "Q1" else "deg"
        print(f"  {q}: items {len(g)}, |solved-crystal| median {err.median():.3f} {unit}, p90 {err.quantile(.9):.3f}; "
              f"off (> {TOL[q]} {unit}): {len(off) / len(g):.0%}; of those, statistic prefers CRYSTAL {pref_c:.0%}, "
              f"prefers solved {1 - pref_c:.0%}")
        if q == "Q1" and len(off):
            far = (off.mode_near_crystal - off.crystal).abs()
            print(f"      off donors: a table mode within 0.05 A of the crystal value for {(far <= 0.05).mean():.0%}; "
                  f"multi-modal tables {(off.n_modes > 1).mean():.0%}; median mode sd {off.sd_mode.median():.3f} A")


if __name__ == "__main__":
    for start in ("etkdg", "full_oracle"):
        f = OUT / (f"coord_audit_{start}.csv" if (OUT / f"coord_audit_{start}.csv").exists() or start != "etkdg"
                   else "coord_audit.csv")
        if f.exists():
            summarize(pd.read_csv(f), start)
