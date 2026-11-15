"""Energy-scale study: decomposition of each seed-11 #1 pick's GFN2 single-point dE into
   torsion/peripheral (SP -> coordination-frozen DOF relax), coordination sphere (-> DOF relax),
   frozen internals (-> full relax), basin (full relax - relaxed reference);
plus the crystal-internals oracle's MACE pool-best / pick as an independent estimate of the internals share."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load import HARTREE, OUT  # noqa: E402


def rows():
    out = []
    for f in sorted((OUT / "pools" / "etkdg").rglob("*_s11.npz")):
        z = np.load(f, allow_pickle=False)
        if "error" in z.files:
            continue
        jf, mf, rf = f.with_suffix(".xtb.json"), f.with_suffix(".mace.json"), f.with_suffix(".rmsd.json")
        if not jf.exists():
            continue
        d = json.loads(jf.read_text())
        if d.get("ref_sp") is None or not d.get("ref_opt"):
            continue
        r, ro = d["ref_sp"], d["ref_opt"][0]
        i = str(int(np.argmax(z["score"])))
        g = lambda k: (d[k][i][0] - r) * HARTREE if d.get(k, {}).get(i) else np.nan  # noqa: E731
        row = dict(set=f.parent.name, case=f.stem[:-4], n_atoms=z["X"].shape[1],
                   sp=(d["sp"][int(i)] - r) * HARTREE if d["sp"][int(i)] is not None else np.nan,
                   dofc=g("dofc"), dof=g("dof"), full=g("opt"), ref_relax=(ro - r) * HARTREE)
        if mf.exists():
            row["mace"] = json.loads(mf.read_text())["dE"][int(i)]
        if rf.exists():
            R = json.loads(rf.read_text())
            if "rmsd" in R:
                row["rmsd"] = R["rmsd"][int(i)]
                row["rmsd_mirror_ok"] = min(R["rmsd"][int(i)], R["rmsd_mirror"][int(i)])
        out.append(row)
    df = pd.DataFrame(out)
    df["basin"] = df.full - df.ref_relax
    df["internals"] = df.dof - df.full
    df["coord"] = df.dofc - df.dof
    df["torsion_periph"] = df.sp - df.dofc
    df["solver_dof"] = df.sp - df.dof
    return df


def oracle_compare():
    """Per complex: MACE pool-best and #1 with ETKDG internals vs crystal internals (seed 11)."""
    out = []
    for f in sorted((OUT / "pools" / "full_oracle").rglob("*_s11.mace.json")):
        g = OUT / "pools" / "etkdg" / f.parent.name / f.name
        fo, fe = f.with_name(f.name.replace(".mace.json", ".npz")), g.with_name(g.name.replace(".mace.json", ".npz"))
        if not g.exists():
            continue
        eo, ee = json.loads(f.read_text())["dE"], json.loads(g.read_text())["dE"]
        so, se = np.load(fo)["score"], np.load(fe)["score"]
        out.append(dict(set=f.parent.name, case=f.name.split("_s")[0], best_etkdg=min(ee), best_oracle=min(eo),
                        pick_etkdg=ee[int(np.argmax(se))], pick_oracle=eo[int(np.argmax(so))]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = rows()
    have = df.dropna(subset=["dof", "full"])
    print(f"picks with DOF + full relax: {len(have)}  (with coordination-frozen too: {df.dofc.notna().sum()})")
    cols = ["sp", "solver_dof", "internals", "basin", "full", "mace"]
    print("\nmedian (kJ/mol, GFN2 unless 'mace'):")
    print(have.groupby("set")[cols].median().round(1).to_string())
    print("\nshare of the summed SP dE (over picks):")
    tot = have.sp.sum()
    for k in ("solver_dof", "internals", "basin"):
        print(f"  {k:12s} {have[k].sum() / tot:6.1%}")
    hc = df.dropna(subset=["dofc", "dof", "full"])
    if len(hc):
        print(f"\nfour-way split on {len(hc)} picks (median / share of sum):")
        for k in ("torsion_periph", "coord", "internals", "basin"):
            print(f"  {k:15s} {hc[k].median():7.1f}  {hc[k].sum() / hc.sp.sum():6.1%}")
    print("\nright structure? pick basin <= 10 kJ/mol:", f"{(have.basin <= 10).mean():.0%}",
          "| heavy-atom RMSD (mirror allowed) median:", round(df.rmsd_mirror_ok.median(), 2),
          "| RMSD <= 1.0 A:", f"{(df.rmsd_mirror_ok <= 1.0).mean():.0%}")
    o = oracle_compare()
    print("\ncrystal-internals oracle (MACE, seed 11), medians:")
    print(o.groupby("set")[["best_etkdg", "best_oracle", "pick_etkdg", "pick_oracle"]].median().round(1).to_string())
    o["internals_share_pick"] = (o.pick_etkdg - o.pick_oracle) / o.pick_etkdg
    print("internals share of the pick (oracle estimate), median:", round(o.internals_share_pick.median(), 2))
    df.to_csv(OUT / "pick_decomposition.csv", index=False)
    o.to_csv(OUT / "oracle_compare.csv", index=False)
