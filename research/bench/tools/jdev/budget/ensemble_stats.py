"""Ensemble-level statistics of the budget pools (ARCHITECTURE.md, 2026-10-16).

Measurement only (GFN2 relaxations from xtb_stage.py opt). For every pool with relaxations:
  * bond-graph check: did the loose GFN2 relaxation keep the covalent graph of its start (conformer and reference)?
    A relaxation that breaks / forms a bond is not a basin of the same molecule and is excluded from basin numbers.
  * basin labels: relaxed structures of the same isomer class (assignment) whose relaxed energies agree within 1 kJ/mol
    AND heavy-atom RMSD (Kabsch, no permutations) < 0.25 A are one basin.
  * two rho (owner, 10-16): rho_isomer = across isomer classes, class-best score vs class energy (SP and relaxed);
    rho_within = inside each class, score vs SP energy (median over classes with >= 4 members);
    plus rho_basin (basin best score vs basin energy) and rho_inbasin (score vs SP inside one basin).

    python tools/jdev/budget/ensemble_stats.py [--seed 11]       (JACOBIK_BUDGET_OUT selects the pools)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from tools.jdev.budget.load import HARTREE, KT, OUT  # noqa: E402

# covalent radii (A), Cordero 2008, enough for tmQM / LnQM elements; others fall back to 1.5
_RCOV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, Se=1.20, Br=1.20,
             Te=1.38, I=1.39, As=1.19, Ge=1.20, Sn=1.39, Al=1.21, Ga=1.22, Sb=1.39, Bi=1.48, Hg=1.32)
METALS = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg La Ce Pr Nd Pm Sm "
             "Eu Gd Tb Dy Ho Er Tm Yb Lu".split())


def bonds(sy, X, scale=1.2):
    """Ligand covalent graph (metal excluded: M-D bonds legitimately stretch on relaxation), as a frozenset of pairs."""
    keep = [i for i, s in enumerate(sy) if s not in METALS]
    r = np.array([_RCOV.get(sy[i], 1.5) for i in keep])
    P = X[keep]
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    ok = (D < scale * (r[:, None] + r[None])) & (D > 0.1)
    ii, jj = np.nonzero(np.triu(ok, 1))
    return frozenset((keep[a], keep[b]) for a, b in zip(ii, jj))


def metal_contacts(sy, X, cut=3.0):
    m = [i for i, s in enumerate(sy) if s in METALS]
    out = set()
    for i in m:
        d = np.sqrt(((X - X[i]) ** 2).sum(-1))
        out |= {(i, j) for j in np.nonzero((d < cut) & (d > 0.1))[0] if sy[j] != "H"}
    return out


def kabsch_rmsd(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    S[-1] *= d
    return float(np.sqrt(max(((A ** 2).sum() + (B ** 2).sum() - 2 * S.sum()) / len(A), 0.0)))


def srho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(spearmanr(a[ok], b[ok])[0])


def pool_records(seed=None, start="etkdg"):
    for f in sorted((OUT / "pools" / start).rglob("*.npz")):
        if f.parent.name == "relaxed" or "relaxed" in f.parts:
            continue
        code, s = f.stem.rsplit("_s", 1)
        if seed and int(s) != seed:
            continue
        if SCORES is not None and f"{f.parent.name}|{code}|{int(s)}" not in SCORES:
            continue
        z = np.load(f, allow_pickle=False)
        if "error" in z.files:
            continue
        mf, xf = f.with_suffix(".mace.json"), f.with_suffix(".xtb.json")
        mace = np.array(json.loads(mf.read_text())["dE"]) if mf.exists() else np.full(len(z["score"]), np.nan)
        x = json.loads(xf.read_text()) if xf.exists() else {}
        yield f, code, int(s), z, mace, x


SCORES = None  # {pool: {key: scores}} from gate.py --dump (main --scores)
SCORE_KEY = "cand"


def analyse_pool(f, code, seed, z, mace, x):
    sy = [str(s) for s in z["symbols"]]
    rsy = [str(s) for s in z["ref_symbols"]]
    n = len(z["score"])
    rank = z["rank"]
    if SCORES is not None:  # --scores: another configuration's scores of the same conformers (gate.py --dump)
        rank = np.asarray(SCORES[f"{f.parent.name}|{code}|{seed}"][SCORE_KEY], float)
    asg = z["asg"]
    heavy = np.array([s != "H" for s in sy])
    gsp = np.full(n, np.nan)
    if x.get("ref_sp") is not None and "sp" in x:
        gsp = np.array([(e - x["ref_sp"]) * HARTREE if e is not None else np.nan for e in x["sp"]])
    rec = dict(set=f.parent.name, case=code, seed=seed, n=n, n_atoms=len(sy), n_isomers=int(len(np.unique(asg))))
    thermal = (3 * len(sy) - 6) * KT / 2
    rec["thermal"] = thermal
    # two rho on MACE SP (all pools) -------------------------------------------------------------------------------
    E = mace
    rec["rho_all"] = srho(-rank, E)
    lo = np.isfinite(E) & (E <= np.nanmin(E) + 50) if np.isfinite(E).any() else np.zeros(n, bool)
    rec["rho_lo"] = srho(-rank[lo], E[lo]) if lo.sum() >= 4 else np.nan
    cls = pd.DataFrame(dict(asg=asg, s=rank, e=E))
    best = cls.loc[cls.groupby("asg").s.idxmax()]
    rec["rho_isomer_sp"] = srho(-best.s, best.e) if len(best) >= 3 else np.nan
    ws = [srho(-g.s, g.e) for _, g in cls.groupby("asg") if len(g) >= 4]
    ws = [w for w in ws if np.isfinite(w)]
    rec["rho_within"] = float(np.median(ws)) if ws else np.nan
    # relaxations ----------------------------------------------------------------------------------------------------
    opt = x.get("opt") or {}
    ro = x.get("ref_opt")
    rd = f.parent / "relaxed" / f.stem
    if not opt or not ro or not rd.exists():
        return rec, None
    ref_ok = None
    if (rd / "-1.npy").exists():
        Yr = np.load(rd / "-1.npy")
        ref_ok = bonds(rsy, z["ref_X"]) == bonds(rsy, Yr)
    rec["ref_graph_kept"] = ref_ok
    rows = []
    for i, v in opt.items():
        i = int(i)
        if v is None or not (rd / f"{i}.npy").exists():
            continue
        Y = np.load(rd / f"{i}.npy")
        X0 = z["X"][i]
        kept = bonds(sy, X0) == bonds(sy, Y)
        mc0, mc1 = metal_contacts(sy, X0), metal_contacts(sy, Y)
        rows.append(dict(i=i, ebasin=(v[0] - ro[0]) * HARTREE, moved=v[1], graph_kept=kept,
                         mcontact_change=len(mc0 ^ mc1), asg=int(asg[i]), s=float(rank[i]), sp=float(gsp[i]),
                         mace=float(mace[i]), Y=Y[heavy]))
    if not rows:
        return rec, None
    R = pd.DataFrame(rows)
    rec["n_relaxed"] = len(R)
    rec["frac_graph_broken"] = float((~R.graph_kept).mean())
    good = R[R.graph_kept].reset_index(drop=True)
    rec["best_basin_all"] = float(R.ebasin.min())
    rec["best_basin_kept"] = float(good.ebasin.min()) if len(good) else np.nan
    if len(good) == 0:
        return rec, R.drop(columns="Y")
    # basin labels: same isomer, |dE| < 1 kJ/mol and RMSD < 0.25 A (greedy, energy-sorted) -------------------------
    lab = -np.ones(len(good), int)
    reps = []
    for k in np.argsort(good.ebasin.values):
        for b, rk in enumerate(reps):
            if good.asg[k] == good.asg[rk] and abs(good.ebasin[k] - good.ebasin[rk]) < 1.0 and \
                    kabsch_rmsd(good.Y[k], good.Y[rk]) < 0.25:
                lab[k] = b
                break
        else:
            lab[k] = len(reps)
            reps.append(k)
    good["basin"] = lab
    rec["n_basins"] = len(reps)
    rec["conf_per_basin"] = len(good) / len(reps)
    top = good.loc[good.s.idxmax()]
    rec["pick_basin"] = float(top.ebasin)
    rec["pick_in_lowest"] = bool(top.ebasin - good.ebasin.min() < 4.184)
    rec["ref_basin_present"] = bool((good.ebasin < 10.0).any() and (good.ebasin > -10.0).any())
    rec["lower_than_ref"] = bool(good.ebasin.min() < -10.0)
    bb = good.groupby("basin").agg(s=("s", "max"), e=("ebasin", "min"), n=("s", "size"))
    rec["rho_basin"] = srho(-bb.s, bb.e) if len(bb) >= 3 else np.nan
    ib = [srho(-g.s, g.sp) for _, g in good.groupby("basin") if len(g) >= 4]
    ib = [v for v in ib if np.isfinite(v)]
    rec["rho_inbasin"] = float(np.median(ib)) if ib else np.nan
    cb = good.groupby("asg").agg(s=("s", "max"), e=("ebasin", "min"))
    rec["rho_isomer_relaxed"] = srho(-cb.s, cb.e) if len(cb) >= 3 else np.nan
    rec["strain_pick"] = float(top.sp - top.ebasin) if np.isfinite(top.sp) else np.nan
    return rec, R.drop(columns="Y")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--scores", default="", help="gate.py --dump JSON: rank the pools with these scores instead")
    p.add_argument("--key", default="cand", help="which arm of --scores (base / cand)")
    a = p.parse_args()
    global SCORES, SCORE_KEY
    if a.scores:
        SCORES, SCORE_KEY = json.loads(Path(a.scores).read_text()), a.key
    recs, details = [], []
    for args in pool_records(seed=a.seed or None):
        rec, R = analyse_pool(*args)
        recs.append(rec)
        if R is not None:
            R.insert(0, "case", rec["case"])
            R.insert(1, "seed", rec["seed"])
            details.append(R)
    df = pd.DataFrame(recs)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    out = Path(a.out) if a.out else OUT / "ensemble_stats"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pools.csv", index=False)
    if details:
        pd.concat(details).to_csv(out / "relaxed.csv", index=False)
    cols = ["rho_all", "rho_lo", "rho_isomer_sp", "rho_within", "n_isomers"]
    print("#### two rho (MACE SP), median over pools [n pools with a value]")
    for s, g in df.groupby("set"):
        print(s, "  ".join(f"{c} {g[c].median():.2f} [{g[c].notna().sum()}]" for c in cols))
    print("all", "  ".join(f"{c} {df[c].median():.2f} [{df[c].notna().sum()}]" for c in cols))
    if "n_relaxed" in df:
        r = df[df.n_relaxed.notna()]
        print(f"\n#### relaxed pools: {len(r)}")
        print(f"reference relaxation kept its graph: {r.ref_graph_kept.mean():.2f}; "
              f"conformer relaxations that broke/formed a bond: {r.frac_graph_broken.mean():.3f} (mean per pool)")
        print(f"pools with a conformer relaxing > 10 kJ/mol below the reference: all relaxations "
              f"{(r.best_basin_all < -10).sum()}, graph-kept only {(r.best_basin_kept < -10).sum()}")
        for c in ["n_basins", "conf_per_basin", "pick_basin", "strain_pick", "rho_basin", "rho_inbasin",
                  "rho_isomer_relaxed"]:
            print(f"{c:20s} median {r[c].median():8.2f}  mean {r[c].mean():8.2f}  [n {r[c].notna().sum()}]")
        for c in ["pick_in_lowest", "ref_basin_present", "lower_than_ref"]:
            print(f"{c:20s} {r[c].astype(float).mean():.2f}")
        print("\nlowest relaxed basins (graph kept):")
        print(r.sort_values("best_basin_kept")[["set", "case", "best_basin_all", "best_basin_kept", "pick_basin",
                                                "ref_graph_kept", "frac_graph_broken"]].head(15).round(1).to_string())


if __name__ == "__main__":
    main()
