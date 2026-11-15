import json, glob, numpy as np, collections, sys
rng = np.random.default_rng(0)
recs = {}
for f in sorted(glob.glob(sys.argv[1] + "/lineage_*.jsonl")):
    for l in open(f):
        r = json.loads(l); recs[(r["code"], r["seed"])] = r
units = []
for k, r in recs.items():
    t = r["rows_table"]; oc = r["outcome_counts"]
    if not r["full_layout"]: continue
    if oc.get("merged", 0) != r["n_merged_with_target"]: continue          # all merge targets resolved
    units.append(r)
print("units with resolved lineage:", len(units), "of", len(recs))
def fam(r):
    t = r["rows_table"]; out = collections.defaultdict(list)
    for b, o in enumerate(t["outcome"]):
        if o == "kept": out[b].append(b)
        elif o == "merged" and t["merged_into"][b] is not None: out[t["merged_into"][b]].append(b)
    return out
def sim(select, trials=1):
    kept_ret = []; top_ret = []; saved = []; n_pool_ret = []
    for r in units:
        t = r["rows_table"]; F = fam(r); B = r["rows"]
        kept = [b for b in F]
        if not kept: continue
        top = [b for b in kept if t["rank"][b] == 1]
        for _ in range(trials):
            S = select(r)                         # boolean mask of retained starts
            ret = [any(S[x] for x in F[b]) for b in kept]
            kept_ret.append(np.mean(ret)); saved.append(1 - S.mean())
            if top: top_ret.append(float(any(S[x] for x in F[top[0]])))
    return np.mean(kept_ret), np.mean(top_ret), np.mean(saved)
def rnd(p): return lambda r: rng.random(r["rows"]) < p
def mcut(frac):
    def f(r):
        t = r["rows_table"]; M = r["M"]; return np.array([m < max(1, int(round(M * frac))) for m in t["m"]])
    return f
def wcut(frac):
    def f(r):
        t = r["rows_table"]; W = r["W"]; return np.array([w < max(1, int(round(W * frac))) for w in t["w"]])
    return f
print("strategy: kept-row retention, top-1 family retention, GPU saved")
for name, s, tr in (("random 75%", rnd(.75), 5), ("random 50%", rnd(.5), 5), ("random 25%", rnd(.25), 5), ("m < M/2", mcut(.5), 1), ("m < M/4", mcut(.25), 1), ("w < W/2", wcut(.5), 1), ("w < W/4", wcut(.25), 1)):
    a, b, c = sim(s, tr); print(f"{name:12s} kept {a:.3f}  top1 {b:.3f}  saved {c:.3f}")
# per-unit distribution of kept rows and family sizes
fs = [len(v) for r in units for v in fam(r).values()]
print("family size mean", np.mean(fs), "median", np.median(fs), "share singleton", np.mean(np.array(fs) == 1))
# units with big waste: kept rows
print("kept rows per unit: median", np.median([r["outcome_counts"].get("kept", 0) for r in units]))
