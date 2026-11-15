"""Aggregate lineage JSONL (+ pool manifests of the same run) -> summary JSON. Runs anywhere (stdlib + numpy)."""
import json, sys, glob, collections, tarfile, io
from pathlib import Path
import numpy as np
lin_dir, run_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
recs = []
for f in sorted(glob.glob(str(lin_dir / "*.jsonl"))):
    for l in open(f):
        try: recs.append(json.loads(l))
        except Exception: pass
# one record per (code, arm, seed): keep the last (resume safe)
by = {}
for r in recs:
    by[(r.get("code"), r.get("arm"), r.get("seed"))] = r
recs = list(by.values())
OUTC = ["kept", "merged", "outside_window", "nonfinite", "beyond_cap"]
_TAR = {}
def _index_tars():
    for t in sorted(glob.glob(str(run_dir / "run_pack*.tar"))):
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if m.name.endswith("_pool.json"):
                    _TAR[m.name.split("/")[-1]] = (t, m.name)
_index_tars()
def mani(code, seed):
    p = run_dir / "control" / "xyz" / code / f"{code}_control_s{seed}_pool.json"
    if p.exists(): return json.loads(p.read_text())
    ent = _TAR.get(f"{code}_control_s{seed}_pool.json")
    if ent:
        with tarfile.open(ent[0]) as tf:
            return json.loads(tf.extractfile(ent[1]).read())
    return None
tot = collections.Counter(); gpu = collections.Counter(); per = []
w_top = collections.Counter(); w_floor = collections.Counter(); w_kept = collections.Counter(); w_all = collections.Counter()
m_top = collections.Counter(); c_top = collections.Counter(); cand_all = collections.Counter()
rel = collections.Counter(); red = collections.defaultdict(list); gpu_per_kept = []
top1_from = []
n_with_w = 0
for r in recs:
    t = r["rows_table"]; oc = r["outcome_counts"]; rows = r["rows"]; s_row = r["solver_s_per_row"]
    for k, v in oc.items():
        tot[k] += v; gpu[k] += v * s_row
    nk = oc.get("kept", 0)
    gpu_per_kept.append(r["solver_wall_s"] / max(r["n_seeds_in_pass"], 1) / max(nk, 1))
    waste = (oc.get("merged", 0) + oc.get("outside_window", 0) + oc.get("nonfinite", 0)) / rows
    per.append(dict(code=r["code"], seed=r["seed"], rows=rows, waste_share=waste, trimmed_share=oc.get("beyond_cap", 0) / rows,
                    kept=nk, gpu_s_per_kept=gpu_per_kept[-1], full_layout=r["full_layout"]))
    for tol, d in r["redundancy"].items(): red[tol].append((d["share_with_earlier_twin"], d["share_of_twins_ending_same"], rows))
    if r["full_layout"]:
        n_with_w += 1
        for b in range(rows):
            w_all[t["w"][b]] += 1; cand_all[t["cand"][b]] += 1
            if t["outcome"][b] == "kept": w_kept[t["w"][b]] += 1
            tg = t["merged_into"][b]
            if tg is not None:
                rel[("same_cand" if t["cand"][b] == t["cand"][tg] else "diff_cand", "same_m" if t["m"][b] == t["m"][tg] else "diff_m", "same_w" if t["w"][b] == t["w"][tg] else "diff_w")] += 1
    m = mani(r["code"], r["seed"])
    if m is not None and r["full_layout"]:
        for row in m["rows"]:
            b = row["row_key"][2]
            if row.get("is_top1"):
                w_top[t["w"][b]] += 1; m_top[t["m"][b]] += 1; c_top[t["cand"][b]] += 1
                top1_from.append((r["code"], r["seed"], t["cand"][b], t["m"][b], t["w"][b], t["rank"][b]))
            if row.get("is_floor"): w_floor[t["w"][b]] += 1
N = sum(tot.values()); G = sum(gpu.values())
summary = dict(n_records=len(recs), n_parents=len({r["code"] for r in recs}), starts=N,
    outcome_share_starts={k: tot[k] / N for k in OUTC if k in tot},
    outcome_share_gpu_time={k: gpu[k] / G for k in OUTC if k in gpu},
    waste_merged_outside_nonfinite_gpu_share=(gpu["merged"] + gpu["outside_window"] + gpu["nonfinite"]) / G,
    trimmed_gpu_share=gpu["beyond_cap"] / G,
    per_unit_waste_share_quantiles={q: float(np.quantile([p["waste_share"] for p in per], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    per_unit_trimmed_share_quantiles={q: float(np.quantile([p["trimmed_share"] for p in per], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    frac_units_waste_ge_30=float(np.mean([p["waste_share"] >= 0.3 for p in per])),
    gpu_s_per_kept_row=dict(median=float(np.median(gpu_per_kept)), mean=float(np.mean(gpu_per_kept)), total=float(sum(r["solver_wall_s"] / max(r["n_seeds_in_pass"], 1) for r in recs) / max(sum(r["outcome_counts"].get("kept", 0) for r in recs), 1))),
    solver_wall_total_s=float(sum(r["solver_wall_s"] / max(r["n_seeds_in_pass"], 1) for r in recs)),
    analysis_s_total=float(sum(r["analysis_s"] for r in recs)),
    redundancy={tol: dict(share_starts_with_earlier_twin=float(sum(a * n for a, _, n in v) / sum(n for _, _, n in v)),
                          share_of_twins_ending_same_row=float(np.mean([b for _, b, _ in v]))) for tol, v in red.items()},
    n_units_full_layout=n_with_w,
    w_distribution=dict(all=dict(sorted(w_all.items())), kept=dict(sorted(w_kept.items())), top1=dict(sorted(w_top.items())), floor=dict(sorted(w_floor.items()))),
    top1_cand=dict(c_top), top1_m=dict(m_top),
    merged_relation=dict((" ".join(k), v) for k, v in rel.most_common()),
    top1_rank_check=collections.Counter(x[5] for x in top1_from).most_common(3),
    worst_units=sorted(per, key=lambda p: -p["waste_share"])[:8])
out.write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: summary[k] for k in ("n_records", "n_parents", "starts", "outcome_share_starts", "outcome_share_gpu_time", "waste_merged_outside_nonfinite_gpu_share", "trimmed_gpu_share", "frac_units_waste_ge_30", "gpu_s_per_kept_row", "redundancy")}, indent=1))
