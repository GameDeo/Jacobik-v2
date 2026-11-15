"""Compare pool arms (2026-10-16): open vs isomer-matched track, crystal internals, crystal-positioned statistics.

Arms are pool directories under <OUT>/pools/:
    etkdg                          O     production (open isomer choice)
    etkdg_isoref                   I     reference isomer only
    full_oracle                    O+int crystal ligand internals
    full_oracle_isoref             I+int
    etkdg_isoref_crystgt           I+tgt statistics translated onto the crystal values
    full_oracle_isoref_crystgt     I+int+tgt
    etkdg_u1                       u1    unified objective (profile u1)
Per pool: #1 MACE dE (by stored score), floor (lowest dE), regret, and for the problem-complex table the per-case
seed medians. Paired tests: bench.paired_test (cases resampled, seeds kept together).

    python tools/jdev/budget/tracks.py [--sets dev32 n48] [--problem 100] [--energy mace64]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, budget_out, pool_energy, pool_files, read_pool  # noqa: E402

sys.path.insert(0, str(REPO))
OUT = budget_out("energy_scale_refit")
ARMS = {"O": "etkdg", "I": "etkdg_isoref", "O+int": "full_oracle", "I+int": "full_oracle_isoref",
        "I+tgt": "etkdg_isoref_crystgt", "I+int+tgt": "full_oracle_isoref_crystgt",
        # abandoned u1 arms (u1, +mass, +lean, +m13, +ez, +sur, +pmf, +contact): pools in data/_archive/pools_abandoned_arms.zip
        "u1c": "etkdg_u1_contact",  # u1 as of 2026-10-17 (contact Q6, E/Z, surrogate, lean): the new baseline
        "u1c+rep": "etkdg_u1_rep",  # repulsive Q6, adopted 10-17
        "u1c+rep+os": "etkdg_u1_os"}  # training-compatible deduction experiment, 10-18


def load(sets, arms=None, energy="mace"):
    from scipy.stats import spearmanr
    rows = []
    for arm, d in ARMS.items():
        if arms and arm not in arms:
            continue
        for s, code, seed, f in pool_files(d, sets, OUT):
            z, e = read_pool(f), pool_energy(f, energy)
            if z is None or e is None:
                continue
            i1 = int(np.argmax(z["score"]))
            lo = np.isfinite(e) & (e <= np.nanmin(e) + 50.0)
            rho_lo = float(spearmanr(z["score"][lo], -e[lo]).statistic) if lo.sum() >= 5 else np.nan
            rows.append(dict(config=arm, set=s, case=code, seed=seed, pick=e[i1], floor=e.min(),
                             regret=e[i1] - e.min(), n=len(e), n_asg=len(np.unique(z["asg"])),
                             score1=float(z["score"][i1]), rank1=float(z["rank"][i1]), Z=float(z["Z"]),
                             top1=float(e[i1] <= e.min() + 4.184), rho_lo=rho_lo,
                             invalid_share=float(np.mean(e > 1000.0))))
    return pd.DataFrame(rows)


def main():
    from tools.jdev.bench import paired_test
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["dev32", "n48"])
    p.add_argument("--problem", type=float, default=100.0)
    p.add_argument("--arms", nargs="*", default=None, help="subset of ARMS (default: every arm that ran)")
    p.add_argument("--energy", choices=["mace", "mace64"], default="mace",
                   help="mace64: float64 MACE energies (pools.py mace --dtype float64)")
    a = p.parse_args()
    df = load(a.sets, a.arms, a.energy)
    valid = json.loads((REPO / "data/benchmark_runs/2026-10-16_post_refit/donor_atoms.json").read_text())
    bad = {c for c, v in valid.items() if isinstance(v, dict) and not v.get("valid_atoms", True)}
    df = df[~df.case.isin(bad)]
    # only complexes present in every arm that ran
    arms = [x for x in ARMS if (df.config == x).any()]
    have = df.groupby(["case", "seed"]).config.nunique()
    keep = have[have == len(arms)].index
    df = df.set_index(["case", "seed"]).loc[keep].reset_index()
    out = []
    P = out.append
    P(f"arms {arms}; complexes {df.case.nunique()} x seeds (pools per arm {len(keep)}); invalid donor sets excluded: "
      f"{sorted(bad)}")
    P("\n#### medians / means over pools (kJ/mol)")
    P(df.groupby("config")[["pick", "floor", "regret", "n", "n_asg"]].agg(["median", "mean"]).round(1)
      .reindex(arms).to_string())
    P("\n#### ranking and validity (basin share requires separate GFN2 basin measurements)")
    P(df.groupby("config")[["top1", "rho_lo", "invalid_share"]].agg(["median", "mean"]).round(4)
      .reindex(arms).to_string())
    P("\n#### paired tests on #1 (pick), floor and regret, all sets pooled (b - a; negative = b lower)")
    df["set_all"] = "all"
    for a_, b_ in [("O", "I"), ("O", "O+int"), ("I", "I+int"), ("I", "I+tgt"), ("I+int", "I+int+tgt"),
                   ("O", "I+int+tgt"), ("O", "u1c"), ("u1c", "u1c+rep"), ("O", "u1c+rep"),
                   ("u1c+rep", "u1c+rep+os")]:
        if a_ not in arms or b_ not in arms:
            continue
        for v in ("pick", "floor", "regret"):
            t = paired_test(df.assign(set=df.set_all), a_, b_, value=v)
            r = t.iloc[0]
            P(f"{b_:>10s} vs {a_:<6s} {v:6s} mean {r.mean_diff:+7.1f} [{r.ci_lo:+7.1f}, {r.ci_hi:+7.1f}] median "
              f"{r.median_diff:+6.1f}  better/worse {int(r.better)}/{int(r.worse)}  MDE {r.mde:.1f}  {r.verdict}")
            for set_name, rset in paired_test(df, a_, b_, value=v).iterrows():
                P(f"  {set_name:6s} {v:6s} mean {rset.mean_diff:+7.1f} "
                  f"[{rset.ci_lo:+7.1f}, {rset.ci_hi:+7.1f}] better/worse "
                  f"{int(rset.better)}/{int(rset.worse)} MDE {rset.mde:.1f} {rset.verdict}")
    P(f"\n#### problem complexes (open-track #1 seed median >= {a.problem:.0f} kJ/mol): seed medians per arm")
    med = df.pivot_table(index="case", columns="config", values="pick", aggfunc="median")[arms]
    fl = df.pivot_table(index="case", columns="config", values="floor", aggfunc="median")[arms]
    base_arm = "O" if "O" in arms else arms[0]
    prob = med[med[base_arm] >= a.problem].sort_values(base_arm, ascending=False)
    t = prob.round(0).astype(int).astype(str) + " / " + fl.loc[prob.index].round(0).astype(int).astype(str)
    P("(cells: #1 / floor)")
    P(t.to_string())
    if "I+int+tgt" in arms:
        tot = prob["O"]
        P("\nshares of the problem-set open-track #1 (median over complexes of each step's drop / the open #1):")
        steps = [("isomer (O -> I)", "O", "I"), ("internals (I -> I+int)", "I", "I+int"),
                 ("data positioning (I+int -> I+int+tgt)", "I+int", "I+int+tgt")]
        for lab, x, y in steps:
            P(f"  {lab:40s} median drop {np.median(prob[x] - prob[y]):7.1f} kJ/mol  "
              f"share {np.median((prob[x] - prob[y]) / tot):.2f}")
        P(f"  {'residual (I+int+tgt #1: solver / Q6 / other)':40s} median {np.median(prob['I+int+tgt']):7.1f} kJ/mol  "
          f"share {np.median(prob['I+int+tgt'] / tot):.2f}")
    txt = "\n".join(out)
    (OUT / "tracks_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
