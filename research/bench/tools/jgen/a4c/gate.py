"""P2R fold-in gate: the six registered gates of research/campaigns/p2r/FOLDIN_GATE_CONTRACT.md.

    python -m tools.jgen.a4c.gate --crude C=POST_C P=POST_P --normal C=NORM_C P=NORM_P \
        --records C=RUN_C/C/records P=RUN_P/P/records --eligibility ELIG.json --out gate.json

POST_* = `a4b.rerun post --select a4c` outputs (crude relaxation, the registered metric); NORM_* = `a4c.relax_level
--level normal` outputs of the same posts. Units = parent x seed; a unit that did not solve counts as a miss in both arms.
Parent-clustered bootstrap (10,000 resamples, 95 %) of per-parent means of the paired unit differences P - C. The
primary root (eligibility `in_corrected_primary`) decides; the all-root is reported beside it. Written before any
fresh-seed result exists (2026-11-14). A failed gate is reported as failed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MISS, TOP1 = "C-MISS", "C-TOP1"
GARBAGE_KJ = 50.0


def _kv(items):
    return dict(s.split("=", 1) for s in items)


def _buckets(post: Path) -> dict:
    d = json.loads((Path(post) / "a4c" / "parents.json").read_text())
    out, rows = {}, {}
    for p in d["parents"]:
        for u in p.get("units", []):
            out[(p["code"], int(u["seed"]))] = (u.get("bucket") if u.get("status") == "ok" else None) or MISS
        for r in p.get("rows", []):
            rows.setdefault((p["code"], int(r["seed"])), []).append(r)
    return out, rows


def _records(d: Path) -> dict:
    out = {}
    for p in Path(d).glob("*.json"):
        try:
            r = json.loads(p.read_text())
            out[(r["code"], int(r["seed"]))] = r
        except Exception:
            pass
    return out


def _boot(per_parent: dict, stat=np.mean, n=10000, seed=0):
    v = np.array([x for x in per_parent.values() if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0, est=None, lo=None, hi=None)
    rng = np.random.default_rng(seed)
    bs = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)])
    return dict(n=int(len(v)), est=float(stat(v)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def _per_parent(units: dict) -> dict:
    acc = {}
    for (code, _), x in units.items():
        if x is not None:
            acc.setdefault(code, []).append(x)
    return {c: float(np.mean(v)) for c, v in acc.items()}


def evaluate(crude, normal, records, codes, seeds, n_boot=10000) -> dict:
    bc = {a: _buckets(p) for a, p in crude.items()}
    bn = {a: _buckets(p) for a, p in normal.items()}
    rec = {a: _records(p) for a, p in records.items()}
    U = [(c, s) for c in codes for s in seeds]
    g = {}

    def bucket(src, arm, u):
        return src[arm][0].get(u, MISS)

    def rate_diff(src, name):
        d = {u: float(bucket(src, "P", u) == name) - float(bucket(src, "C", u) == name) for u in U}
        rates = {a: float(np.mean([bucket(src, a, u) == name for u in U])) for a in ("C", "P")}
        return rates, _boot(_per_parent(d), n=n_boot)

    r, b = rate_diff(bc, MISS)
    g["1_cmiss_crude"] = dict(rates=r, diff=b, rule="hi < 0", passed=b["hi"] is not None and b["hi"] < 0)
    r, b = rate_diff(bn, MISS)
    g["2_cmiss_normal"] = dict(rates=r, diff=b, rule="hi <= +0.02", passed=b["hi"] is not None and b["hi"] <= 0.02)
    r, b = rate_diff(bc, TOP1)
    de = {}
    for u in U:
        rc, rp = rec["C"].get(u), rec["P"].get(u)
        if rc and rp and rc.get("status") == "ok" and rp.get("status") == "ok" and \
                rc.get("top1_dE") is not None and rp.get("top1_dE") is not None:
            de[u] = float(rp["top1_dE"]) - float(rc["top1_dE"])
    be = _boot(_per_parent(de), stat=np.median, n=n_boot)
    g["3_top1"] = dict(rates=r, ctop1_diff=b, top1_dE_median_diff_kj=be,
                       rule="ctop1 lo >= -0.03 and median dE hi <= +2 kJ/mol",
                       passed=b["lo"] is not None and b["lo"] >= -0.03 and be["hi"] is not None and be["hi"] <= 2.0)
    ok = {a: {u for u in U if rec[a].get(u, {}).get("status") == "ok"} for a in ("C", "P")}
    p_only_fail = sorted(f"{c}_{s}" for c, s in ok["C"] - ok["P"])
    g["4_failures"] = dict(failed_P_solved_C=p_only_fail, failed_C_solved_P=sorted(f"{c}_{s}" for c, s in ok["P"] - ok["C"]),
                           failed_both=sorted(f"{c}_{s}" for c, s in set(U) - ok["C"] - ok["P"]),
                           rule="no P-only failure", passed=not p_only_fail)

    def walls(arm):                                # parent wall: one per parent (shared by its seeds), max over seeds
        w = {}
        for (c, s), r in rec[arm].items():
            t = (r.get("run_timing") or {}).get("parent_wall_s")
            if t is not None and c in codes:
                w[c] = max(w.get(c, 0.0), float(t))
        return w
    wc, wp = walls("C"), walls("P")
    both = sorted(c for c in wc if c in wp and all(((c, s) in ok["C"] and (c, s) in ok["P"]) for s in seeds))
    ratio = sum(wp[c] for c in both) / max(1e-9, sum(wc[c] for c in both)) if both else None
    g["5_wall"] = dict(ratio=ratio, n_parents=len(both), total_C_s=sum(wc[c] for c in both), total_P_s=sum(wp[c] for c in both),
                       top_ratio=sorted(((wp[c] / wc[c], c) for c in both if wc[c] > 0), reverse=True)[:8],
                       rule="ratio <= 1.3", passed=ratio is not None and ratio <= 1.3)

    def garbage(arm, u):
        rows = [r for r in bc[arm][1].get(u, []) if r.get("relax_status") == "ok"]
        if not rows:
            return None
        return float(np.mean([(r.get("topology_same") is False) or (r.get("offset") is not None and r["offset"] > GARBAGE_KJ)
                              for r in rows]))
    gd = {}
    for u in U:
        a, b_ = garbage("C", u), garbage("P", u)
        if a is not None and b_ is not None:
            gd[u] = b_ - a
    bg = _boot(_per_parent(gd), n=n_boot)
    g["6_garbage"] = dict(diff=bg, rates={a: float(np.nanmean([x for x in (garbage(a, u) for u in U) if x is not None]))
                                          for a in ("C", "P")},
                          rule="hi <= +0.02", passed=bg["hi"] is not None and bg["hi"] <= 0.02)
    # transitions, reported
    trans = {}
    for name, src in (("crude", bc), ("normal", bn)):
        trans[name] = dict(
            gained=sorted(f"{c}_{s}" for c, s in U if bucket(src, "C", (c, s)) == MISS and bucket(src, "P", (c, s)) != MISS),
            lost=sorted(f"{c}_{s}" for c, s in U if bucket(src, "C", (c, s)) != MISS and bucket(src, "P", (c, s)) == MISS))
    return dict(n_parents=len(codes), n_units=len(U), gates=g, all_passed=all(v["passed"] for v in g.values()),
                transitions=trans)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crude", nargs=2, required=True); ap.add_argument("--normal", nargs=2, required=True)
    ap.add_argument("--records", nargs=2, required=True); ap.add_argument("--eligibility", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33]); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(argv)
    crude, normal, records = _kv(a.crude), _kv(a.normal), _kv(a.records)
    el = json.loads(Path(a.eligibility).read_text())
    c_all = set()
    for post in crude.values():
        d = json.loads((Path(post) / "a4c" / "parents.json").read_text())
        c_all |= {p["code"] for p in d["parents"]}
    prim = {p["code"] for p in el["parents"] if p.get("in_corrected_primary") and p.get("stage") == 1}
    out = dict(contract="research/campaigns/p2r/FOLDIN_GATE_CONTRACT.md", seeds=a.seeds,
               primary=evaluate(crude, normal, records, sorted(c_all & prim), a.seeds, a.boot),
               all=evaluate(crude, normal, records, sorted(c_all), a.seeds, a.boot))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for root in ("primary", "all"):
        r = out[root]
        print(f"== {root}: {r['n_parents']} parents, {r['n_units']} units, all gates passed: {r['all_passed']}")
        for k, v in r["gates"].items():
            print(f"  {k:16s} {'PASS' if v['passed'] else 'FAIL'}  {v['rule']}  "
                  + json.dumps({kk: vv for kk, vv in v.items() if kk not in ('passed', 'rule', 'top_ratio')}, default=str)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
