"""Paired A4c comparison of arms of one tools.run solve (P2R Step 4 dev loop).

    python -m tools.jgen.a4c.paired --control POST_C --arm NAME=POST_A [--arm ...] --solve RUN [--out OUT.json]

POST_* = `tools.jgen.a4b.rerun post --select a4c` outputs (OUT/a4c/parents.json). Per unit (parent x seed) the A4c
bucket; a unit that did not solve (timeout, failure) counts as a miss in both arms (same rule as the baseline report).
Reported per arm, on the parents present in both arms:
  * C-MISS rate, C-TOP1 rate, and the paired difference arm - control with a parent-clustered bootstrap 95 % CI;
  * unit transitions (miss -> hit, hit -> miss) with the parents; each gain tagged by the mechanism of its hitting row
    (reflected / chair embed / production row, from the pool row_keys);
  * untouched units (no reflected and no chair-embed row in the arm pool) and their flips: a run-to-run noise check;
  * unrelaxed quality of the relaxed rows: median MACE energy released on relaxation (manifest energy - relaxed), share
    of rows whose class changes on relaxation (from the post's relaxed rows: relax/<code>_<seed>.json);
  * solve wall per parent (records' run_timing.parent_wall_s, packed arms share it: reported for information only).
Loops never fail the batch.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MISS = "C-MISS"


def _units(post) -> dict:
    if isinstance(post, (list, tuple)):          # an arm solved in parts (one post per part)
        out = {}
        for q in post:
            out.update(_units(q))
        return out
    d = json.loads((post / "a4c" / "parents.json").read_text())
    out = {}
    for p in d["parents"]:
        for u in p.get("units", []):
            b = u.get("bucket") if u.get("status") == "ok" else MISS
            out[(p["code"], int(u["seed"]))] = b or MISS
    return out


def _quality(post) -> dict:
    """{(code, seed): [(energy released kJ/mol, class kept)]} from the relaxed rows and their pool manifests."""
    if isinstance(post, (list, tuple)):
        res = {}
        for q in post:
            res.update(_quality(q))
        return res
    res = defaultdict(list)
    rel = post / "a4b" / "relax"
    for f in rel.glob("*_*.json"):
        if f.name in ("references.json", "failures.json", "run_metrics.json", "selection.json"):
            continue
        try:
            d = json.loads(f.read_text())
            mp = next((post / "a4b" / "xyz" / d["code"]).glob(f"{d['code']}_*_s{d['seed']}_pool.json"))
            mrows = json.loads(mp.read_text())["rows"]
            for k, r in d.get("rows", {}).items():
                e0, e1 = mrows[int(k)].get("mace_energy_kj"), r.get("E_mace_kj")
                res[(d["code"], int(d["seed"]))].append((None if e0 is None or e1 is None else e0 - e1, r.get("hit")))
        except Exception:
            pass
    return res


def _attribution(post) -> dict:
    """{(code, seed): set of mechanisms of the unit's hit rows}: 'reflected' (out_mirror: the row's `mirror_of`
    descriptor, or a second row with its source's row_key in pools without descriptors), 'derived' (config_remote:
    `config_derived`), 'ring_state' (row_key embed index >= 1000: a ring_states
    non-embed state), 'stereo_extra' (a row of a non-production stereo class: `stereo_production` false with a
    requested class label), 'other' (a production-path row)."""
    if isinstance(post, (list, tuple)):
        res = {}
        for q in post:
            res.update(_attribution(q))
        return res
    d = json.loads((post / "a4c" / "parents.json").read_text())
    res = {}
    for p in d["parents"]:
        code = p["code"]
        for seed in {int(u["seed"]) for u in p.get("units", [])}:
            hits = [r["row"] for r in p.get("rows", []) if int(r["seed"]) == seed and r.get("hit")]
            if not hits:
                continue
            try:
                mp = next((post / "a4b" / "xyz" / code).glob(f"{code}_*_s{seed}_pool.json"))
                rows = json.loads(mp.read_text())["rows"]
            except (StopIteration, OSError):
                continue
            seen, tag = set(), {}
            for i, r in enumerate(rows):
                key = json.dumps(r.get("row_key"))
                emb = (r.get("row_key") or [None])[0]
                refl = r.get("mirror_of") is not None if "mirror_of" in r else key in seen
                tag[i] = ("reflected" if refl else "derived" if r.get("config_derived")
                          else "ring_state" if (emb or 0) >= 1000
                          else "stereo_extra" if (r.get("requested_stereo_label") and r.get("stereo_production") is False)
                          else "other")
                seen.add(key)
            res[(code, seed)] = {tag.get(int(h), "other") for h in hits}
    return res


def _touched(post) -> dict:
    """{(code, seed): (n reflected rows, n ring-state rows)} over the arm's pool manifests (units the flags changed)."""
    if isinstance(post, (list, tuple)):
        res = {}
        for q in post:
            res.update(_touched(q))
        return res
    res = {}
    for mp in (post / "a4b" / "xyz").glob("*/*_pool.json"):
        try:
            rows = json.loads(mp.read_text())["rows"]
            seed = int(mp.name.rsplit("_s", 1)[1].split("_")[0])
            keys = [json.dumps(r.get("row_key")) for r in rows]
            res[(mp.parent.name, seed)] = (len(keys) - len(set(keys)),
                                           sum(((r.get("row_key") or [0])[0] or 0) >= 1000 for r in rows))
        except Exception:
            pass
    return res


def _boot(codes, per_code_diff, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    codes = list(codes)
    tot = np.array([per_code_diff[c][0] for c in codes], float)
    cnt = np.array([per_code_diff[c][1] for c in codes], float)
    if not len(codes):
        return (float("nan"),) * 3
    est = tot.sum() / cnt.sum()
    idx = rng.integers(0, len(codes), size=(n, len(codes)))
    bs = tot[idx].sum(1) / cnt[idx].sum(1)
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def compare(control: Path, arms: dict, solve: Path | None = None) -> dict:
    uc = _units(control)
    out = dict(control=str(control), arms={})
    for name, post in arms.items():
        try:
            ua = _units(post)
            keys = sorted(set(uc) & set(ua))
            codes = sorted({c for c, _ in keys})
            diff = {c: [0.0, 0] for c in codes}
            mc = ma = tc = ta = 0
            gain, loss = [], []
            for c, s in keys:
                a_miss, c_miss = ua[(c, s)] == MISS, uc[(c, s)] == MISS
                mc += c_miss; ma += a_miss; tc += uc[(c, s)] == "C-TOP1"; ta += ua[(c, s)] == "C-TOP1"
                diff[c][0] += int(a_miss) - int(c_miss); diff[c][1] += 1
                if c_miss and not a_miss:
                    gain.append(f"{c}:{s}")
                if a_miss and not c_miss:
                    loss.append(f"{c}:{s}")
            dtop = {c: [0.0, 0] for c in codes}
            for c, s in keys:
                dtop[c][0] += int(ua[(c, s)] == "C-TOP1") - int(uc[(c, s)] == "C-TOP1"); dtop[c][1] += 1
            n = len(keys)
            rec = dict(n_units=n, n_parents=len(codes), miss_control=mc / n, miss_arm=ma / n,
                       top1_control=tc / n, top1_arm=ta / n,
                       d_miss=dict(zip(("est", "lo", "hi"), _boot(codes, diff))),
                       d_top1=dict(zip(("est", "lo", "hi"), _boot(codes, dtop))),
                       miss_to_hit=gain, hit_to_miss=loss)
            for tag, post_dir in (("control", control), ("arm", post)):
                q = _quality(post_dir)
                drops = [e for k in keys for e, _ in q.get(k, []) if e is not None]
                rec[f"energy_released_median_kj_{tag}"] = float(np.median(drops)) if drops else None
                rec[f"n_relaxed_rows_{tag}"] = len(drops)
            att = _attribution(post)
            rec["miss_to_hit_by"] = {u: sorted(att.get((u.split(":")[0], int(u.split(":")[1])), {"?"})) for u in gain}
            only = lambda u, m: att.get((u.split(":")[0], int(u.split(":")[1]))) == {m}
            rec["miss_to_hit_counts"] = dict(reflected_only=sum(only(u, "reflected") for u in gain),
                                             ring_state_only=sum(only(u, "ring_state") for u in gain),
                                             stereo_extra_only=sum(only(u, "stereo_extra") for u in gain),
                                             derived_only=sum(only(u, "derived") for u in gain),
                                             production_row=sum("other" in att.get((u.split(":")[0], int(u.split(":")[1])), set()) for u in gain))
            tch = _touched(post)
            un = [k for k in keys if tch.get(k) == (0, 0)]
            rec["untouched"] = dict(n_units=len(un), n_parents=len({c for c, _ in un}),
                                    miss_flips=[f"{c}:{s_}" for c, s_ in un if (ua[(c, s_)] == MISS) != (uc[(c, s_)] == MISS)],
                                    top1_flips=[f"{c}:{s_}" for c, s_ in un if (ua[(c, s_)] == "C-TOP1") != (uc[(c, s_)] == "C-TOP1")])
            out["arms"][name] = rec
        except Exception as exc:  # never fail the batch
            out["arms"][name] = dict(error=f"{type(exc).__name__}: {exc}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=POST_DIR (or NAME=DIR1+DIR2 for an arm posted in parts)")
    ap.add_argument("--solve", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arm)
    split = lambda v: [Path(x) for x in v.split("+")] if "+" in v else Path(v)
    res = compare(split(a.control), {k: split(v) for k, v in arms.items()}, Path(a.solve) if a.solve else None)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    for name, r in res["arms"].items():
        if "error" in r:
            print(name, "ERROR", r["error"]); continue
        d = r["d_miss"]; t = r["d_top1"]
        print(f"{name}: units {r['n_units']} parents {r['n_parents']} | miss {r['miss_control']:.3f} -> {r['miss_arm']:.3f} "
              f"(d {d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]) | top1 {r['top1_control']:.3f} -> {r['top1_arm']:.3f} "
              f"(d {t['est']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]) | released kJ/mol median "
              f"{r['energy_released_median_kj_control']} -> {r['energy_released_median_kj_arm']}")
        print("  miss->hit", " ".join(f"{u}[{'/'.join(r['miss_to_hit_by'].get(u, []))}]" for u in r["miss_to_hit"]) or "-")
        print("  by mechanism", r["miss_to_hit_counts"])
        print("  hit->miss", " ".join(r["hit_to_miss"]) or "-")
        u = r["untouched"]
        print(f"  untouched units {u['n_units']} ({u['n_parents']} parents): miss flips {u['miss_flips'] or '-'}, top1 flips {u['top1_flips'] or '-'}")


if __name__ == "__main__":
    main()
