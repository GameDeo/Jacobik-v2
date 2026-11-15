"""Energy-scale study, stage 3: error budget, implied energy scale per term, cross-validated re-ranking."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from load import KT, TERMS, load  # noqa: E402

RNG = np.random.default_rng(0)
KCAL = 4.184


def pool_metrics(g: pd.DataFrame, s: np.ndarray, E: str):
    e = g[E].to_numpy()
    ok = np.isfinite(e)
    if ok.sum() < 2:
        return None
    s, e = s[ok], e[ok]
    i = int(np.argmax(s))
    lo = e <= e.min() + 50
    return dict(pick=e[i], best=e.min(), regret=e[i] - e.min(), top1=float(e[i] - e.min() <= KCAL),
                rho=spearmanr(s, -e).statistic if len(e) >= 4 else np.nan,
                rho_lo=spearmanr(s[lo], -e[lo]).statistic if lo.sum() >= 5 else np.nan, n=len(e))


def evaluate(df, scorefn, E):
    out = []
    for pool, g in df.groupby("pool", sort=False):
        m = pool_metrics(g, scorefn(g), E)
        if m:
            m.update(pool=pool, set=g.set.iat[0], case=g.case.iat[0], seed=g.seed.iat[0], start=g.start.iat[0])
            out.append(m)
    return pd.DataFrame(out)


def summarize(r, by="set"):
    return r.groupby(by).agg(pools=("pick", "size"), pick_med=("pick", "median"), pick_mean=("pick", "mean"),
                             best_med=("best", "median"), regret_med=("regret", "median"),
                             regret_mean=("regret", "mean"), top1=("top1", "mean"), rho=("rho", "median"),
                             rho_lo=("rho_lo", "median")).round(2)


def paired(ra, rb, col="pick", n_boot=4000):
    """Per-complex mean over seeds of (b - a); cluster bootstrap CI over complexes."""
    m = ra.merge(rb, on=["pool"], suffixes=("_a", "_b"))
    m["d"] = m[f"{col}_b"] - m[f"{col}_a"]
    pc = m.groupby("case_a").d.mean().to_numpy()
    bs = [RNG.choice(pc, len(pc)).mean() for _ in range(n_boot)]
    return dict(n=len(pc), mean=pc.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                median=np.median(pc), better=int((pc < -KCAL).sum()), worse=int((pc > KCAL).sum()))


def demean(df, cols):
    return df[cols] - df.groupby("pool")[cols].transform("mean")


CLIP = 150.0  # kJ/mol above the pool's best: a catastrophe counts as "very bad", not as 3000 kJ/mol of leverage
TOPK = 16  # fit on each pool's top-k by production score (selection on the score, not on the outcome)


def prep(df, E, topk=None, clip=None):
    topk = TOPK if topk is None else topk
    clip = CLIP if clip is None else clip
    d = df[np.isfinite(df[E])].copy()
    if topk:
        d = d[d.groupby("pool").score.rank(ascending=False, method="first") <= topk]
    if clip:
        d[E] = np.minimum(d[E], d.groupby("pool")[E].transform("min") + clip)
    return d[d.groupby("pool")[E].transform("size") >= 3]


def fit_beta(df, E, terms=TERMS, huber=True, raw=False):
    from sklearn.linear_model import HuberRegressor, LinearRegression
    d = df if raw else prep(df, E)
    X = demean(d, terms).to_numpy()
    y = (d[E] - d.groupby("pool")[E].transform("mean")).to_numpy()
    sd = X.std(0)
    sd[sd == 0] = 1
    m = HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=2000, fit_intercept=False) if huber else LinearRegression(fit_intercept=False)
    m.fit(X / sd, y)
    return m.coef_ / sd


def beta_table(df, E, terms=TERMS, n_boot=200):
    df = prep(df, E)
    b = fit_beta(df, E, terms, raw=True)
    cases = df.case.unique()
    bs = []
    for _ in range(n_boot):
        pick = RNG.choice(cases, len(cases))
        parts = [df[df.case == c].assign(pool=lambda x, k=k: x.pool + f"#{k}") for k, c in enumerate(pick)]
        bs.append(fit_beta(pd.concat(parts), E, terms, raw=True))
    bs = np.array(bs)
    t = pd.DataFrame({"beta_kJ_per_nat": b, "lo": np.percentile(bs, 2.5, 0), "hi": np.percentile(bs, 97.5, 0)},
                     index=terms)
    t["in_kT"] = t.beta_kJ_per_nat / KT
    # within-pool spread of each term (nats, median over pools of the sd) and its energy-equivalent
    sd = df.groupby("pool")[terms].std().median()
    t["pool_sd_nats"] = sd
    t["pool_sd_kJ"] = (sd * t.beta_kJ_per_nat).abs()
    # partial Spearman-ish: within-pool rank correlation of each term alone with -E (median over pools)
    t["rho_alone"] = [df.groupby("pool").apply(lambda g, q=q: spearmanr(g[q], -g[E]).statistic
                                                if g[E].notna().sum() >= 4 and g[q].std() > 0 else np.nan).median()
                      for q in terms]
    return t.round(3)


def loco_rank(df, E_train, terms=TERMS, group="case"):
    """Leave-one-complex-out learned linear ranking: returns dict pool -> scores."""
    out = {}
    fin = df[np.isfinite(df[E_train])]
    for c in df[group].unique():
        tr, te = fin[fin[group] != c], df[df[group] == c]
        b = fit_beta(tr, E_train, terms)
        pred = -(te[terms].to_numpy() @ b)  # higher = predicted lower energy
        for pool, idx in te.groupby("pool").indices.items():
            full = np.full(int(te.idx.to_numpy()[idx].max()) + 1, np.nan)
            full[te.idx.to_numpy()[idx]] = pred[idx]
            out[pool] = full
    return out


def main():
    df = load()
    df = df[np.isfinite(df.mace)]
    et = df[df.start == "etkdg"].copy()
    prod = lambda g: g.score.to_numpy()  # noqa: E731

    print("\n#### 1. Error budget (production ranking), MACE SP dE (kJ/mol)")
    print(summarize(evaluate(df, prod, "mace"), ["start", "set"]).to_string())
    for E in ("gsp", "gbasin"):
        print(f"\n#### 1b. Same, {E}")
        print(summarize(evaluate(df[np.isfinite(df[E])], prod, E), ["start", "set"]).to_string())
    # strain share of the pick (GFN2, seed 11)
    r = evaluate(et[np.isfinite(et.gbasin)], prod, "gsp")
    picks = []
    for pool, g in et[np.isfinite(et.gbasin)].groupby("pool"):
        i = int(np.argmax(g.score.to_numpy()))
        j = int(np.argmin(g.gsp.to_numpy()))
        k = int(np.argmin(g.gbasin.to_numpy()))
        picks.append(dict(set=g.set.iat[0], case=g.case.iat[0], sp=g.gsp.iat[i], basin=g.gbasin.iat[i],
                          strain=g.gstrain.iat[i], moved=g.moved.iat[i],
                          sp_of_spbest=g.gsp.iat[j], basin_of_spbest=g.gbasin.iat[j],
                          best_basin=g.gbasin.iat[k], n_atoms=g.n_atoms.iat[0],
                          pick_right_basin=float(g.gbasin.iat[i] <= 10), any_right_basin=float(g.gbasin.min() <= 10)))
    P = pd.DataFrame(picks)
    print("\n#### 1c. #1 pick decomposed (GFN2, seed 11): SP = strain + basin")
    print(P.groupby("set")[["sp", "strain", "basin", "best_basin", "sp_of_spbest", "basin_of_spbest",
                            "pick_right_basin", "any_right_basin"]].median().round(1).to_string())
    print(P.groupby("set")[["pick_right_basin", "any_right_basin"]].mean().round(2).to_string())
    P["thermal"] = (3 * P.n_atoms - 6) * KT / 2
    print("strain / thermal scale (3N-6)kT/2, median:", round((P.strain / P.thermal).median(), 3),
          "  strain per vibrational mode in kT:", round((P.strain / (3 * P.n_atoms - 6) / KT).median(), 3))
    P.to_csv(OUT_DIR / "pick_decomposition.csv", index=False)

    print("\n#### 2. Implied energy scale of each term (within-pool, Huber; cluster-bootstrap 95% CI)")
    print("    If the score were -E/kT, every beta would be -1.0 kT per nat.")
    for E in ("mace", "gsp", "gbasin"):
        d = et[np.isfinite(et[E])]
        print(f"\n--- target {E}: {d.pool.nunique()} pools, {len(d)} conformers")
        print(beta_table(d, E).to_string())

    print("\n#### 3. Re-ranking (deterministic, pools fixed): production vs single-term drops vs learned LOCO")
    res = {"prod": prod}
    for q in TERMS:
        res[f"drop_{q}"] = (lambda q: lambda g: (g.score * g.Z - g[q]).to_numpy())(q)
    for E_tr in ("mace", "gsp", "gbasin"):
        sc = loco_rank(et, E_tr)
        res[f"learn_{E_tr}"] = (lambda sc: lambda g: sc[g.pool.iat[0]][g.idx.to_numpy()])(sc)
    for E in ("mace", "gsp", "gbasin"):
        d = et[np.isfinite(et[E])]
        base = evaluate(d, prod, E)
        print(f"\n--- evaluated on {E} ({d.pool.nunique()} pools): paired vs prod, per-complex mean pick diff [95% CI]")
        rows = []
        for name, fn in res.items():
            r = evaluate(d, fn, E)
            p = paired(base, r, "pick")
            rows.append(dict(ranker=name, pick_med=r.pick.median(), pick_mean=r.pick.mean(), regret_med=r.regret.median(),
                             top1=r.top1.mean(), rho=r.rho.median(), rho_lo=r.rho_lo.median(),
                             d_mean=p["mean"], lo=p["lo"], hi=p["hi"], better=p["better"], worse=p["worse"]))
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    df.to_parquet(OUT_DIR / "conformers.parquet")


def yardstick(df):
    """How well do the two energy references agree within a pool? A ceiling for any score-vs-energy rho."""
    out = []
    for pool, g in df[df.start == "etkdg"].groupby("pool"):
        ok = g.mace.notna() & g.gsp.notna()
        if ok.sum() < 5:
            continue
        a, b = g.mace[ok].to_numpy(), g.gsp[ok].to_numpy()
        lo = a <= a.min() + 50
        out.append(dict(set=g.set.iat[0], rho=spearmanr(a, b).statistic,
                        rho_lo=spearmanr(a[lo], b[lo]).statistic if lo.sum() >= 5 else np.nan,
                        same_best=float(int(np.argmin(a)) == int(np.argmin(b)) or b[int(np.argmin(a))] <= b.min() + KCAL)))
    r = pd.DataFrame(out)
    print("\n#### 0. Yardstick: MACE vs GFN2 single points within a pool (ceiling for score rho)")
    print(r.groupby("set")[["rho", "rho_lo", "same_best"]].median().round(3).to_string())
    print("all pools: rho median", round(r.rho.median(), 3), "| rho_lo median", round(r.rho_lo.median(), 3),
          "| MACE's lowest is within 1 kcal/mol of GFN2's lowest:", f"{r.same_best.mean():.0%}")


# ---------------------------------------------------------------------------------------------------------------------
# Pairwise ranking objective (RankingSVM / logistic on within-pool pairs), non-negative weights, LOCO
# ---------------------------------------------------------------------------------------------------------------------
def pair_data(df, E, terms=TERMS, min_gap=4.184, topk=16, clip=150.0):
    d = prep(df, E, topk=topk, clip=clip)
    Xs, ys, ws = [], [], []
    for _, g in d.groupby("pool"):
        x = g[terms].to_numpy()
        e = g[E].to_numpy()
        i, j = np.triu_indices(len(e), 1)
        gap = e[i] - e[j]
        keep = np.abs(gap) > min_gap
        if not keep.any():
            continue
        i, j, gap = i[keep], j[keep], gap[keep]
        Xs.append(x[i] - x[j]); ys.append((gap < 0).astype(float))  # 1 = i lower in energy
        ws.append(np.full(len(i), 1.0 / len(i)))  # each pool counts once
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def fit_pairwise(df, E, terms=TERMS, l2=1e-3, nonneg=True):
    from scipy.optimize import minimize
    X, y, w = pair_data(df, E, terms)
    sd = X.std(0); sd[sd == 0] = 1
    Xn = X / sd
    s = 2 * y - 1

    def f(b):
        z = s * (Xn @ b)
        loss = (w * np.logaddexp(0, -z)).sum() + l2 * (b @ b)
        g = -(Xn * (w * s / (1 + np.exp(z)))[:, None]).sum(0) + 2 * l2 * b
        return loss, g
    b0 = np.ones(len(terms)) * 0.1
    r = minimize(f, b0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(terms) if nonneg else None)
    return r.x / sd  # weights on the raw nats (higher score = predicted lower energy)


def loco_pairwise(df, E_train, terms=TERMS, group="case"):
    out, W = {}, []
    fin = df[np.isfinite(df[E_train])]
    for c in df[group].unique():
        b = fit_pairwise(fin[fin[group] != c], E_train, terms)
        W.append(b / max(b.sum(), 1e-12))
        te = df[df[group] == c]
        pred = te[terms].to_numpy() @ b
        for pool, idx in te.groupby("pool").indices.items():
            full = np.full(int(te.idx.to_numpy()[idx].max()) + 1, np.nan)
            full[te.idx.to_numpy()[idx]] = pred[idx]
            out[pool] = full
    return out, np.array(W)


def main_pairwise():
    df = load(("etkdg",))
    df = df[np.isfinite(df.mace)]
    prod = lambda g: g.score.to_numpy()  # noqa: E731
    b_all = fit_pairwise(df, "mace")
    print("pairwise weights on all data (relative to Q1..extra, normalised to sum 1):",
          dict(zip(TERMS, np.round(b_all / b_all.sum(), 3))))
    res = {"prod": prod, "equal_sum": lambda g: g[TERMS].sum(1).to_numpy()}
    for E_tr in ("mace", "gsp"):
        sc, W = loco_pairwise(df, E_tr)
        print(f"LOCO weights trained on {E_tr}: mean {dict(zip(TERMS, np.round(W.mean(0), 3)))}  "
              f"sd {dict(zip(TERMS, np.round(W.std(0), 3)))}")
        res[f"pair_{E_tr}"] = (lambda sc: lambda g: sc[g.pool.iat[0]][g.idx.to_numpy()])(sc)
    for E in ("mace", "gsp", "gbasin"):
        d = df[np.isfinite(df[E])]
        base = evaluate(d, prod, E)
        rows = []
        for name, fn in res.items():
            r = evaluate(d, fn, E)
            p = paired(base, r, "pick")
            rows.append(dict(ranker=name, pick_med=r.pick.median(), pick_mean=r.pick.mean(),
                             regret_med=r.regret.median(), top1=r.top1.mean(), rho=r.rho.median(),
                             rho_lo=r.rho_lo.median(), d_mean=p["mean"], lo=p["lo"], hi=p["hi"],
                             better=p["better"], worse=p["worse"]))
        print(f"\n--- evaluated on {E} ({d.pool.nunique()} pools)")
        print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    from load import OUT as OUT_DIR  # noqa: E402
    import warnings
    warnings.filterwarnings("ignore")
    yardstick(load(("etkdg",)))
    main()
    print("\n#### 4. Pairwise ranking objective (non-negative weights, leave-one-complex-out)")
    main_pairwise()
