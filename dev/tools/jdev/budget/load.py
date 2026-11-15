"""Load the energy-scale study pools into one tidy conformer table."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
OUT = Path(os.environ.get("JACOBIK_BUDGET_OUT", REPO / "data" / "energy_scale"))
HARTREE = 2625.4996  # kJ/mol
KT = 2.479  # kJ/mol at 298 K
TERMS = ["Q1", "Q2", "Q3", "Q5", "Q6", "extra"]  # Q3 = S_orient when present; extra = bend prior + closure (nats-eq.)


def load(starts=("etkdg", "full_oracle")) -> pd.DataFrame:
    rows = []
    for start in starts:
        for f in sorted((OUT / "pools" / start).rglob("*.npz")):
            z = np.load(f, allow_pickle=False)
            if "error" in z.files:
                continue
            code, seed = f.stem.rsplit("_s", 1)
            n = len(z["score"])
            Z = float(z["Z"])
            d = dict(start=[start] * n, set=[f.parent.name] * n, case=[code] * n, seed=[int(seed)] * n,
                     idx=np.arange(n), score=z["score"], rank=z["rank"], asg=z["asg"], Z=[Z] * n,
                     n_atoms=[z["X"].shape[1]] * n)
            for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
                d[q] = z[f"terms_{q}"] if f"terms_{q}" in z.files else np.zeros(n)
            d["Q3"] = d["Q3"] + d["Q4"]
            d["extra"] = (z["score"] - z["rank"]) * Z
            d["n_Q6"] = [int(z["n_Q6"])] * n
            mf, xf = f.with_suffix(".mace.json"), f.with_suffix(".xtb.json")
            d["mace"] = json.loads(mf.read_text())["dE"] if mf.exists() else [np.nan] * n
            gsp, gb, mv = [np.nan] * n, [np.nan] * n, [np.nan] * n
            if xf.exists():
                x = json.loads(xf.read_text())
                if x.get("ref_sp") is not None and "sp" in x:
                    gsp = [(e - x["ref_sp"]) * HARTREE if e is not None else np.nan for e in x["sp"]]
                ro = x.get("ref_opt")
                if ro and "opt" in x:
                    for i, v in x["opt"].items():
                        if v is not None:
                            gb[int(i)] = (v[0] - ro[0]) * HARTREE
                            mv[int(i)] = v[1]
            d["gsp"], d["gbasin"], d["moved"] = gsp, gb, mv
            rows.append(pd.DataFrame({k: v for k, v in d.items() if k != "Q4"}))
    df = pd.concat(rows, ignore_index=True)
    df["gstrain"] = df.gsp - df.gbasin
    df["pool"] = df.start + "|" + df.set + "|" + df.case + "|" + df.seed.astype(str)
    return df


if __name__ == "__main__":
    df = load()
    print(df.groupby(["start", "set"]).agg(pools=("pool", "nunique"), confs=("idx", "size"),
                                          mace=("mace", lambda s: s.notna().mean()),
                                          gsp=("gsp", lambda s: s.notna().mean()),
                                          gbasin=("gbasin", lambda s: s.notna().mean())))
