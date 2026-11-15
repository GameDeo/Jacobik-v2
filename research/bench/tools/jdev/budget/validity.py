"""Offline study of obscene ensemble members (plan Part C, 2026-10-16). Measurement only, no solver change.

stage `features`: per pool conformer, the solver-side quantities a validity filter could use (no reference, no energy):
    q1_z      max over donors of the distance to the NEAREST Q1 component, in that component's fit sd
    q1_dA     the same in Angstrom
    q1_wall   max over donors of the Q1 wall excess (d - d_hi) / sd (0 inside the table's support)
    q6_min    min over Q6 pairs of d / mu (5th-percentile wall onset)
    q6_n07    number of Q6 pairs below 0.7 mu
    nb_min    min over heavy-atom pairs >= 3 bonds apart (ligand graph + M-D bonds) of d / (rcov_i + rcov_j); this
              also sees the pairs Q6 leaves out (through-metal 1-3, ligand 1-4)
    gap       production ranking score below the pool best, in summed nats ((rank_best - rank) * Z)
stage `report`: precision / recall of candidate filters against MACE dE, losses of good conformers, isomer classes and
    #1 changes; writes <OUT>/validity/report.txt.

    python tools/jdev/budget/validity.py features --sets dev32 n48     (JACOBIK_BUDGET_OUT / _DB select the pools)
    python tools/jdev/budget/validity.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pools import DB, OUT, check_db, set_cases, setup  # noqa: E402

VD = OUT / "validity"
_RCOV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, Se=1.20, Br=1.20,
             Te=1.38, I=1.39, As=1.19, Ge=1.20, Sn=1.39, Al=1.21, Ga=1.22, Sb=1.39, Bi=1.48)


def _stats(db):
    from jacobik.stats.stats import StatsLookup, load_db
    meta = db.with_name(db.stem + ".meta.json")
    return StatsLookup(load_db(json.loads(db.read_text()), meta=json.loads(meta.read_text()) if meta.exists() else None))


def _far_pairs(sy, X0, metal, donors):
    """Heavy-atom pairs >= 3 bonds apart in the covalent graph of the pool's best conformer plus M-D bonds."""
    import networkx as nx
    n = len(sy)
    r = np.array([_RCOV.get(s, 1.5) for s in sy])
    D = np.sqrt(((X0[:, None] - X0[None]) ** 2).sum(-1))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    lig = [i for i in range(n) if i != metal]
    for a_ in range(len(lig)):
        for b_ in range(a_ + 1, len(lig)):
            i, j = lig[a_], lig[b_]
            if D[i, j] < 1.2 * (r[i] + r[j]):
                G.add_edge(i, j)
    for d in donors:
        G.add_edge(metal, d)
    sp = dict(nx.all_pairs_shortest_path_length(G, cutoff=2))
    heavy = [i for i in range(n) if sy[i] != "H"]
    a, b = [], []
    for x in range(len(heavy)):
        for y in range(x + 1, len(heavy)):
            i, j = heavy[x], heavy[y]
            if j not in sp.get(i, {}):
                a.append(i)
                b.append(j)
    return np.array(a, int), np.array(b, int), r


def cmd_features(a):
    setup()
    import torch
    from tools.jdev.cases import get_case
    check_db()
    st = _stats(DB)
    VD.mkdir(parents=True, exist_ok=True)
    names = {}
    for s in a.sets:
        names.update({c: n for c, n in set_cases(s)})
    for s in a.sets:
        of = VD / f"features_{s}.csv"
        done = set(pd.read_csv(of)["pool"]) if of.exists() else set()
        for f in sorted((OUT / "pools" / a.start / s).glob("*.npz")):
            code, seed = f.stem.rsplit("_s", 1)
            pool = f"{s}|{code}|{seed}"
            mf = f.with_suffix(".mace.json")
            if pool in done or not mf.exists():
                continue
            z = np.load(f, allow_pickle=False)
            if "error" in z.files:
                continue
            try:
                case = get_case(names[code])
                r = case.make_builder().solve(seed=int(seed), stats=st, angle_bends=True, n_conformers=1, n_ik_steps=2,
                                              device="cpu")
                sc = r.scorer
                ge = sc._ge
                X = torch.tensor(z["X"], dtype=torch.float32)
                asg = torch.as_tensor(z["asg"], dtype=torch.long)
                with torch.no_grad():
                    d = ge.q1_distances(X)  # (B, n)
                    q1c = getattr(sc, "_q1_cand", None)
                    if q1c is not None:
                        mu, lw = q1c[0][asg], q1c[2][asg]
                        var = sc._q1_cand_var[asg]
                        hi, wsd = sc._q1_cand_wall[0][asg], sc._q1_cand_wall[1][asg]
                    else:
                        B = X.shape[0]
                        mu = sc._q1_means[None].expand(B, -1, -1)
                        lw = sc._q1_log_w[None].expand(B, -1, -1)
                        var = sc._q1_eff_vars[None].expand(B, -1, -1)
                        hi, wsd = sc._q1_wall_hi[None], sc._q1_wall_sd[None]
                    real = lw > -1e3
                    sd = var.clamp(min=1e-6).sqrt()
                    dist = (d.unsqueeze(-1) - mu).abs()
                    z_ = torch.where(real, dist / sd, torch.full_like(dist, 1e9)).min(-1).values
                    dA = torch.where(real, dist, torch.full_like(dist, 1e9)).min(-1).values
                    wall = torch.clamp(d - hi, min=0) / wsd
                    q1_z = z_.max(-1).values.numpy() if d.shape[1] else np.zeros(len(X))
                    q1_dA = dA.max(-1).values.numpy() if d.shape[1] else np.zeros(len(X))
                    q1_wall = wall.max(-1).values.numpy() if d.shape[1] else np.zeros(len(X))
                    if ge.n_q6 > 0 and sc._q6_mu.shape[0] > 0:
                        d6 = (X[:, ge._q6_a] - X[:, ge._q6_b]).norm(dim=-1) / sc._q6_mu
                        q6_min = d6.min(-1).values.numpy()
                        q6_n07 = (d6 < 0.7).sum(-1).numpy()
                    else:
                        q6_min, q6_n07 = np.ones(len(X)), np.zeros(len(X), int)
                sy = [str(t) for t in z["symbols"]]
                metal = int(ge.metal_pos_idx)
                donors = sorted({int(i) for lst in ge._haptic_pos_lists for i in lst} | set(int(i) for i in ge._kappa_pos_indices))
                best = int(np.argmax(z["score"]))
                pa, pb, rc = _far_pairs(sy, z["X"][best], metal, donors)
                if len(pa):
                    Dn = np.sqrt(((z["X"][:, pa] - z["X"][:, pb]) ** 2).sum(-1)) / (rc[pa] + rc[pb])
                    nb_min = Dn.min(-1)
                else:
                    nb_min = np.full(len(X), np.nan)
            except Exception as exc:
                print(f"  {pool}: skipped ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
                continue
            mace = np.array(json.loads(mf.read_text())["dE"])
            Z = float(z["Z"])
            n = len(mace)
            df = pd.DataFrame(dict(pool=[pool] * n, set=s, case=code, seed=int(seed), idx=np.arange(n),
                                   n_atoms=len(sy), asg=z["asg"], score=z["score"], rank=z["rank"],
                                   gap=(z["rank"].max() - z["rank"]) * Z, q1_z=q1_z, q1_dA=q1_dA, q1_wall=q1_wall,
                                   q6_min=q6_min, q6_n07=q6_n07, nb_min=nb_min, mace=mace))
            for q in ("Q1", "Q2", "Q3", "Q5", "Q6"):
                k = f"terms_{q}"
                df[q] = z[k] if k in z.files else 0.0
            df.to_csv(of, mode="a", header=not of.exists(), index=False)
            print(f"  {pool}: {n} conformers", flush=True)


def _filters():
    """name -> function(df) -> boolean mask of conformers the filter DROPS."""
    F = {}
    for t in (4, 6, 8, 10, 15):
        F[f"F1 q1_z>{t}"] = (lambda t: lambda d: d.q1_z > t)(t)
    for t in (0.3, 0.5, 0.8):
        F[f"F1b q1_dA>{t}"] = (lambda t: lambda d: d.q1_dA > t)(t)
    for t in (1, 3):
        F[f"F1c q1_wall>{t}"] = (lambda t: lambda d: d.q1_wall > t)(t)
    for t in (0.6, 0.7, 0.75):
        F[f"F2 q6_min<{t}"] = (lambda t: lambda d: d.q6_min < t)(t)
    for t in (0.6, 0.7, 0.8):
        F[f"F2b nb_min<{t}"] = (lambda t: lambda d: d.nb_min < t)(t)
    for t in (50, 100, 200, 500):
        F[f"F3 gap>{t}"] = (lambda t: lambda d: d.gap > t)(t)
    F["F1 q1_z>8 | F2b nb_min<0.7"] = lambda d: (d.q1_z > 8) | (d.nb_min < 0.7)
    F["F1 q1_z>10 | F2b nb_min<0.6"] = lambda d: (d.q1_z > 10) | (d.nb_min < 0.6)
    F["F1b q1_dA>0.5 | F2b nb_min<0.7"] = lambda d: (d.q1_dA > 0.5) | (d.nb_min < 0.7)
    return F


def cmd_report(a):
    fs = sorted(VD.glob("features_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    lines = []
    P = lines.append
    thermal = (3 * d.n_atoms - 6) * 2.479 / 2
    d["floor"] = d.groupby("pool").mace.transform("min")
    d["bad500"] = d.mace > 500
    d["bad1000"] = d.mace > 1000
    d["obscene"] = d.mace > 3 * thermal
    d["good"] = d.mace <= d.floor + 50
    P(f"pools {d.pool.nunique()}, conformers {len(d)}; > 500: {d.bad500.mean():.3f}, > 1000: {d.bad1000.mean():.3f}, "
      f"> 3x thermal: {d.obscene.mean():.3f}")
    P("\n#### what the > 1000 kJ/mol conformers look like (medians; 'good' = within floor + 50)")
    cols = ["q1_z", "q1_dA", "q1_wall", "q6_min", "q6_n07", "nb_min", "gap"]
    P(d.groupby(np.where(d.bad1000, "bad1000", np.where(d.bad500, "500-1000", np.where(d.good, "good", "mid"))))[cols]
      .median().round(2).to_string())
    P("\n#### candidate filters (drop = filter removes the conformer)")
    rows = []
    for name, fn in _filters().items():
        drop = fn(d).fillna(False).values
        g = d.assign(drop=drop)
        # isomer classes that lose every member, among classes that had a good member
        cls = g.groupby(["pool", "asg"]).agg(all_drop=("drop", "all"), any_good=("good", "any"))
        lost_cls = int((cls.all_drop & cls.any_good).sum())
        # pool #1 change and whether the new #1 is better / worse in MACE (within 4.184 kJ/mol = same)
        def pick(gg, mask):
            gg = gg[mask]
            return gg.mace.values[np.argmax(gg["score"].values)] if len(gg) else np.nan
        dif = []
        for p, gg in g.groupby("pool"):
            keep = ~gg["drop"].values
            if keep.sum() == 0:
                dif.append(np.nan)
                continue
            dif.append(pick(gg, keep) - pick(gg, np.ones(len(gg), bool)))
        dif = np.array(dif)
        rows.append(dict(filter=name, drop_frac=drop.mean(), recall1000=drop[d.bad1000].mean(),
                         recall500=drop[d.bad500].mean(), recall_obscene=drop[d.obscene].mean(),
                         good_dropped=drop[d.good].mean(), floor_lost=int(g[g.mace == g.floor]["drop"].sum()),
                         classes_lost=lost_cls, pools_emptied=int(np.isnan(dif).sum()),
                         pick_better=int((dif < -4.184).sum()), pick_worse=int((dif > 4.184).sum()),
                         pick_mean_diff=np.nanmean(dif)))
    P(pd.DataFrame(rows).round(3).to_string(index=False))
    P("\n#### #1 picks that are themselves > 500 kJ/mol")
    tops = d.loc[d.groupby("pool")["score"].idxmax()]
    P(tops[tops.mace > 500][["pool", "mace", "floor"] + cols].round(2).to_string(index=False))
    txt = "\n".join(lines)
    (VD / "report.txt").write_text(txt)
    print(txt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["features", "report"])
    p.add_argument("--sets", nargs="+", default=["dev32", "n48"])
    p.add_argument("--start", default="etkdg")
    a = p.parse_args()
    {"features": cmd_features, "report": cmd_report}[a.stage](a)


if __name__ == "__main__":
    main()
