"""Curvature matching measurement (plan E1, 2026-10-16). Measurement only, no scorer change.

If the score S were -E/kT, its curvature along any direction v would be the energy's over kT:
    -d2S/dh2 = (d2E/dh2) / kT.
That is the property the planned curvature-biased (2-shot) relaxation needs. It is LOCAL, so it can calibrate the
core of each term without touching the tails that veto broken structures (the failure mode of the 10-15 energy
regressions and of the 10-16 SHAPE tail caps). A first version compared gradients with MACE forces: meaningless at the
pool conformers, which are stationary points of the solver's score (gradient ~0 by construction).

Directions: the solver's soft degrees of freedom at X (per ligand: 3 translations, 3 rotations about its centroid;
one rotation per rotatable bond, moving its smaller side), each scaled to 1 A of total displacement, plus N_RAND random
unit combinations. Central differences with step H (A): curvature of MACE-POLAR-1-S energy (kJ/mol/A^2) and of each
production ranking-score term (summed nats/A^2, sign flipped so that a well is positive).
Points: the top-K conformers by ranking score per pool, and the CRYSTAL (a GFN2 minimum: there the energy curvature is
the physical stiffness the score should reproduce).

    python tools/jdev/budget/forces.py compute --sets dev32 n48 --top 4          # active profile (JACOBIK_PROFILE)
    JACOBIK_PROFILE=u1 python tools/jdev/budget/forces.py compute               # reuses the cached MACE curvatures
    python tools/jdev/budget/forces.py report        (per profile: JACOBIK_PROFILE selects the directory)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pools import OUT, REPO, check_db, load_stats, set_cases, setup  # noqa: E402

# Output: OUT/curvature2/geom (points, directions, MACE energy curvatures; made once) and OUT/curvature2/<profile>
# (score curvatures under JACOBIK_PROFILE). The 2026-10-16 b results (one rng across complexes) stay in curvature/ and
# curvature_solve/.
KT = 2.479  # kJ/mol
EV = 96.485  # kJ/mol per eV
TERMS = ["Q1", "Q2", "Q3", "Q5", "Q6"]
H_STEP = 0.10  # A of total displacement
N_RAND = 12
_RCOV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, Se=1.20, Br=1.20,
             Te=1.38, I=1.39, As=1.19, Ge=1.20, Sn=1.39, Al=1.21, Ga=1.22, Sb=1.39, Bi=1.48)


def dof_basis(sy, X, metal):
    """(D, N, 3) displacement fields of the soft DOFs at X, each scaled to unit norm, and their types."""
    import networkx as nx
    n = len(sy)
    r = np.array([_RCOV.get(s, 1.5) for s in sy])
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n) if i != metal])
    for i in range(n):
        for j in range(i + 1, n):
            if i != metal and j != metal and D[i, j] < 1.2 * (r[i] + r[j]):
                G.add_edge(i, j)
    cols, types = [], []
    ring_edges = set()
    for cyc in nx.cycle_basis(G):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            ring_edges.add(frozenset((a, b)))
    for comp in nx.connected_components(G):
        comp = sorted(comp)
        c = X[comp].mean(0)
        for ax in np.eye(3):
            v = np.zeros((n, 3))
            v[comp] = ax
            cols.append(v)
            types.append("trans")
        for ax in np.eye(3):
            v = np.zeros((n, 3))
            v[comp] = np.cross(ax, X[comp] - c)
            cols.append(v)
            types.append("rot")
        sub = G.subgraph(comp)
        for a, b in sub.edges():
            if frozenset((a, b)) in ring_edges or sub.degree(a) < 2 or sub.degree(b) < 2:
                continue
            Hg = sub.copy()
            Hg.remove_edge(a, b)
            side_b = nx.node_connected_component(Hg, b)
            side_a = set(comp) - side_b
            piv, side = (b, side_b) if len(side_b) <= len(side_a) else (a, side_a)
            other = a if piv == b else b
            u = X[piv] - X[other]
            u = u / np.linalg.norm(u)
            v = np.zeros((n, 3))
            idx = sorted(side)
            v[idx] = np.cross(u, X[idx] - X[piv])
            cols.append(v)
            types.append("tors")
    B = np.stack(cols)
    nrm = np.sqrt((B ** 2).sum((1, 2)))
    keep = nrm > 1e-6
    return B[keep] / nrm[keep][:, None, None], [t for t, k in zip(types, keep) if k]


MODE = "rank"  # "rank": ranking score in summed nats (fit floors); "solve": the solver's objective (scorer floors,
               # per-quantity means, Q6 soft-min) x Z, i.e. on the same per-complex scale


def term_values(sc, X, rg):
    """(B, T) score terms at X: ranking settings (as pools.rank_terms, summed nats) or the solve objective (x Z)."""
    import torch
    from pools import rank_terms
    Z = float(sc._agg_norm())
    if MODE == "solve":
        with torch.no_grad():
            _, bd = sc.score(torch.tensor(X, dtype=torch.float32), row_groups=rg)
    else:
        _, bd = rank_terms(sc, torch.tensor(X, dtype=torch.float32), rg)
    cols = []
    for q in TERMS:
        keys = [q, "Q4"] if q == "Q3" else [q]
        v = sum(bd[k].double().cpu().numpy() for k in keys if k in bd)
        cols.append(np.asarray(v, float) * Z if np.ndim(v) else np.zeros(len(X)))
    return np.stack(cols, 1)


def _out_dir(a):
    """Per-profile score curvatures: OUT/curvature2/<profile>[_solve]/ (energy curvatures shared: curvature2/geom)."""
    from jacobik.core import profile as P
    return OUT / "curvature2" / (P.ACTIVE_NAME + ("_solve" if MODE == "solve" else "")
                                 + (f"_{a.tag}" if getattr(a, "tag", "") else ""))


def cmd_compute(a):
    """Score curvatures under the ACTIVE profile (JACOBIK_PROFILE) along cached directions. The points, directions and
    MACE energy curvatures are made once per complex (curvature2/geom, MACE on `--device`) and reused by every profile,
    so profiles are compared on identical displacements. Directions are seeded per complex (crc32 of the code)."""
    import zlib
    setup()
    import torch
    from oracle_targets import crystal_in_engine_order
    from tools.jdev.cases import get_case
    check_db()
    st = load_stats()
    ev = None
    gd = OUT / "curvature2" / "geom"
    od = _out_dir(a)
    gd.mkdir(parents=True, exist_ok=True)
    od.mkdir(parents=True, exist_ok=True)
    for s in a.sets:
        names = dict(set_cases(s))
        for f in sorted((OUT / "pools" / a.start / s).glob(f"*_s{a.seed}.npz")):
            code = f.stem.rsplit("_s", 1)[0]
            of, gf = od / f"{a.start}_{s}_{code}.npz", gd / f"{a.start}_{s}_{code}.npz"
            if of.exists():
                continue
            z = np.load(f, allow_pickle=False)
            mf = f.with_suffix(".mace.json")
            if "error" in z.files or not mf.exists():
                continue
            try:
                case = get_case(names[code])
                r = case.make_builder().solve(seed=a.seed, stats=st, angle_bends=True, n_conformers=1, n_ik_steps=2,
                                              device="cpu")
                sc, mol = r.scorer, r.conformers[0].complex
                sy = [str(t) for t in z["symbols"]]
                sel = list(np.argsort(-z["rank"])[: a.top])
                if gf.exists():
                    g = np.load(gf, allow_pickle=False)
                    P0, V, pt_of_dir, asg = g["P0"], g["V"], g["point"], [int(x) for x in g["asg"]]
                    kE, gE, types = g["kE"], g["gE"], g["types"]
                else:
                    if ev is None:
                        from jacobik.core.energy import MACEPolarEvaluator
                        ev = MACEPolarEvaluator(model="polar-1-s", device=a.device, mode="energy-only",
                                                dtype="float64", model_path=str(REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model"))
                    rng = np.random.default_rng(zlib.crc32(code.encode()))
                    metal = int(sc._ge.metal_pos_idx)
                    q, mult = int(z["charge"]), int(z["mult"])
                    pts = [z["X"][i] for i in sel]
                    asg = [int(z["asg"][i]) for i in sel]
                    Xc, ok = crystal_in_engine_order(case, mol, z["X"][sel[0]])
                    if ok.all():
                        pts.append(Xc)
                        asg.append(asg[0])
                    P0, V, pt_of_dir, kE, gE, types = [], [], [], [], [], []
                    for p_i, X0 in enumerate(pts):
                        Bs, ty = dof_basis(sy, X0, metal)
                        R = rng.normal(size=(N_RAND, len(Bs)))
                        Vr = np.einsum("rd,dnk->rnk", R, Bs)
                        Vr /= np.sqrt((Vr ** 2).sum((1, 2)))[:, None, None]
                        Vp = np.concatenate([Bs, Vr])
                        Xs = [X0] + [X0 + H_STEP * v for v in Vp] + [X0 - H_STEP * v for v in Vp]
                        E = np.array(ev.energies([(sy, x, q, mult) for x in Xs])) * EV  # kJ/mol
                        nV = len(Vp)
                        kE.append((E[1:1 + nV] + E[1 + nV:] - 2 * E[0]) / H_STEP ** 2)
                        gE.append((E[1:1 + nV] - E[1 + nV:]) / (2 * H_STEP))
                        P0.append(X0)
                        V.append(Vp)
                        types += list(ty) + ["rand"] * N_RAND
                        pt_of_dir += [p_i if p_i < len(sel) else -1] * nV
                    P0, V = np.stack(P0), np.concatenate(V)
                    kE, gE, types, pt_of_dir = np.concatenate(kE), np.concatenate(gE), np.array(types), np.array(pt_of_dir)
                    np.savez_compressed(gf, P0=P0, V=V.astype(np.float32), point=pt_of_dir, asg=np.array(asg), kE=kE,
                                        gE=gE, types=types, n_atoms=len(sy),
                                        mace=np.array(json.loads(mf.read_text())["dE"])[sel])
                # score curvatures under the active profile
                kS = []
                pidx = np.where(pt_of_dir < 0, len(P0) - 1, pt_of_dir)
                for p_i in range(len(P0)):
                    Vp = V[pidx == p_i].astype(np.float64)
                    X0 = P0[p_i]
                    Xs = [X0] + [X0 + H_STEP * v for v in Vp] + [X0 - H_STEP * v for v in Vp]
                    rg = torch.full((len(Xs),), asg[p_i], dtype=torch.long)
                    S = term_values(sc, np.stack(Xs), rg)  # (1 + 2n, T)
                    nV = len(Vp)
                    kS.append(-(S[1:1 + nV] + S[1 + nV:] - 2 * S[0]) / H_STEP ** 2)
                np.savez_compressed(of, kE=kE, kS=np.concatenate(kS), gE=gE, types=types, point=pt_of_dir,
                                    n_atoms=len(sy))
                print(f"{s}/{code}: {len(P0)} points, crystal {'yes' if (pt_of_dir < 0).any() else 'no'}", flush=True)
            except Exception as exc:
                print(f"{s}/{code}: ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)


def cmd_report(a):
    from scipy.optimize import nnls
    from scipy.stats import spearmanr
    rows = []
    setup()  # src on sys.path (_out_dir imports jacobik.core.profile)
    fd = _out_dir(a)
    for f in sorted(fd.glob(f"{a.start}_*.npz")):
        z = np.load(f, allow_pickle=False)
        s, code = f.stem[len(a.start) + 1:].split("_", 1)
        for j in range(len(z["kE"])):
            rows.append(dict(set=s, case=code, point=int(z["point"][j]), type=str(z["types"][j]),
                             kE=float(z["kE"][j]) / KT, **{f"k{q}": float(z["kS"][j, k]) for k, q in enumerate(TERMS)}))
    if not rows:
        print("no data")
        return
    d = pd.DataFrame(rows)
    d["kS"] = d[[f"k{q}" for q in TERMS]].sum(1)
    d["where"] = np.where(d.point < 0, "crystal", "pool top-K")
    out = []
    P = out.append
    P(f"directions {len(d)}, complexes {d.case.nunique()}; curvatures in nats / A^2 (energy divided by kT); "
      f"step {H_STEP} A of total displacement")
    P("\n#### median curvature, energy/kT vs score (all terms) and per term, by point and DOF type")
    cols = ["kE", "kS"] + [f"k{q}" for q in TERMS]
    P(d.groupby(["where", "type"])[cols].median().round(2).to_string())
    P("\n#### ratio energy/kT : score per direction (1 = calibrated; > 1 score too soft), directions with kE > 0")
    pos = d[(d.kE > 0) & (d.kS > 0)]
    P(pos.assign(ratio=pos.kE / pos.kS).groupby(["where", "type"]).ratio.describe(percentiles=[.25, .5, .75])
      [["count", "25%", "50%", "75%"]].round(2).to_string())
    P(f"fraction of directions where the energy curves up but the score is flat or curves down (kS <= 0): "
      + "  ".join(f"{w} {((g.kE > 0) & (g.kS <= 0)).mean():.2f}" for w, g in d.groupby("where")))
    P("\n#### rank agreement of curvatures across directions within a point (spearman, median over points)")
    for w, g in d.groupby("where"):
        rh = [spearmanr(gg.kE, gg.kS)[0] for _, gg in g.groupby(["case", "point"]) if len(gg) > 5]
        P(f"{w}: {np.nanmedian(rh):.2f}  (n points {len(rh)})")
    P("\n#### per-term ratio energy/kT : score on directions where that term carries >= 70 % of the positive score "
      "curvature (median over complexes of the per-complex median; the robust per-term read, NNLS below is confounded "
      "by correlated terms)")
    pos_t = d[[f"k{q}" for q in TERMS]].clip(lower=0)
    tot = pos_t.sum(axis=1)
    for w, idx in d.groupby("where").groups.items():
        parts = []
        for q in TERMS:
            m = (pos_t.loc[idx, f"k{q}"] >= 0.7 * tot.loc[idx]) & (tot.loc[idx] > 0) & (d.loc[idx, "kE"] > 0)
            g = d.loc[idx][m]
            if len(g):
                pc = (g.kE / g.kS).groupby(g.case).median()
                parts.append(f"{q} {pc.median():.2f} (n {len(g)}, {len(pc)} cx)")
        P(f"{w}: " + "  ".join(parts))
    P("\n#### per-term curvature scales s_k >= 0 (NNLS: kE ~ sum_k s_k kS_k), complex-bootstrap 95% CI")
    for w, g in d.groupby("where"):
        A = g[[f"k{q}" for q in TERMS]].values
        y = g.kE.values
        s, _ = nnls(A, y)
        cases = g.case.unique()
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(200):
            pick = rng.choice(cases, len(cases))
            gg = pd.concat([g[g.case == c] for c in pick])
            bs.append(nnls(gg[[f"k{q}" for q in TERMS]].values, gg.kE.values)[0])
        bs = np.array(bs)
        P(f"{w}: " + "  ".join(f"{q} {s[k]:.2f} [{np.percentile(bs[:, k], 2.5):.2f}, {np.percentile(bs[:, k], 97.5):.2f}]"
                               for k, q in enumerate(TERMS)))
        pred = A @ s
        P(f"   R^2 of the fit {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():.2f}; "
          f"with unit scales (production) {1 - ((y - A.sum(1)) ** 2).sum() / ((y - y.mean()) ** 2).sum():.2f}")
    txt = "\n".join(out)
    (fd / f"report_{a.start}.txt").write_text(txt)
    print(txt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["compute", "report"])
    p.add_argument("--sets", nargs="+", default=["dev32", "n48"])
    p.add_argument("--start", default="etkdg")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["rank", "solve"], default="rank")
    p.add_argument("--tag", default="", help="suffix of the output directory (curvature2/<profile>[_solve][_tag])")
    a = p.parse_args()
    global MODE
    MODE = a.mode
    {"compute": cmd_compute, "report": cmd_report}[a.stage](a)


if __name__ == "__main__":
    main()
