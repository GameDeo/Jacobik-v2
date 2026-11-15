"""Within-class conformations and relaxed basins (P2R fold-in gate, reported, not gated).

    python -m tools.jgen.a4c.class_basins select --solve RUN --arms C P --seed 31 --inputs A4B_INPUTS --out OUT
    python -m tools.jgen.a4c.class_basins relax  --solve RUN --out OUT --workers 128
    python -m tools.jgen.a4c.class_basins report --out OUT

Questions (owner 2026-11-14), per class (isomer class of the unrelaxed pool rows), on a representative parent subset:
  1. how many distinct conformations a class holds (pool rows are already RMSD-deduplicated, so class size);
  2. how often the conformations of a class collapse on relaxation into the class's lowest basin;
  3. how many distinct relaxed basins a class ends with;
  4. how well the unrelaxed ranking (score) agrees with the relaxed ranking (GFN2 energy), within a class and per pool.
Every row of the chosen units is relaxed ONCE with GFN2 `--opt normal`. The crude level is read from the same
trajectory: xtb's crude run is the exact prefix of the normal run (checked 2026-11-14 on 6 rows, HOHZAG/SILROY/DOHTOM:
identical energies and geometries), ending at the first cycle with |dE| < 5e-4 Eh and gnorm < 1e-2 Eh/a0. So both
levels are reported at the cost of one relaxation.

Definitions (fixed before any gate result):
  * classes: greedy over rows in score order; a row joins the first class whose representative (`classes.Reference` of
    its unrelaxed geometry, the reference's graph and automorphisms) calls it `same_class`; the reference's own class is
    seeded by the unrelaxed reference. Enantiomers are distinct classes.
  * a relaxed row LEAVES its class when the class representative no longer calls the relaxed geometry `same_class`
    (or topology changes); it is counted as left and excluded from that class's basins.
  * same basin: |dE| <= 1.0 kJ/mol and heavy-atom RMSD <= 0.25 A, minimised over the reference's automorphisms
    (proper rotation). Greedy in energy order: a row joins the lowest basin it matches.
  * collapse share of a class: rows in the class's lowest basin / rows kept in the class (classes with >= 2 kept rows).
  * ranking: Spearman of score vs relaxed energy within classes with >= 3 kept rows, and over all relaxed rows of the
    pool; plus whether the best-ranked row of a class lies in its lowest basin.
Subset: one seed, same parents in every arm, stratified by panel (cal/tail) and pool size tertile, filled in a fixed
hash order until the estimated relaxation cost reaches the budget (default 1.0 x the gate's own relaxation cost
estimate: <= 5 rows per unit, every seed, every arm; cost = rows x atoms^2). Never fails the batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

H2KJ = 2625.4996
ECONV_CRUDE, GCONV_CRUDE = 5e-4, 1e-2
SAME_E_KJ, SAME_RMSD_A = 1.0, 0.25
_LOG_E = re.compile(r"energy:\s+(-?\d+\.\d+)\s+gnorm:\s+(\d+\.\d+)")


def _manifest(run: Path, arm: str, code: str, seed: int) -> Path | None:
    d = run / arm / "xyz" / code
    p = d / f"{code}_{arm}_s{seed}_pool.json"
    return p if p.exists() else next(iter(sorted(d.glob(f"{code}_*_s{seed}_pool.json"))), None)


def _frames(mp: Path):
    from tools.jgen.bench.pool_export import _read_frames
    m = json.loads(mp.read_text())
    g2r = np.asarray(m["gen_to_ref_idx"], int)
    X = [np.asarray(fr[2], float)[g2r] for f in dict.fromkeys(r["file"] for r in m["rows"])
         for fr in _read_frames(mp.parent / f)]
    if len(X) != int(m["n_rows"]):
        raise ValueError("XYZ/manifest row count mismatch")
    return m, X


# -- select ---------------------------------------------------------------------------------------------------------
def cmd_select(a) -> int:
    run, out = Path(a.solve), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((Path(a.inputs) / "vm_split.json").read_text())
    panel = {c: "cal" for c in split["cal_parents"]} | {c: "tail" for c in split["tail_parents"]}
    info, gate_cost = {}, 0.0
    for code in sorted(panel):
        rows = {}
        for arm in a.arms:
            for seed in a.gate_seeds:
                mp = _manifest(run, arm, code, seed)
                if mp is None:
                    continue
                m = json.loads(mp.read_text())
                n, at = int(m["n_rows"]), len(m["ref_symbols"])
                gate_cost += min(5, n) * at ** 2
                if seed == a.seed:
                    rows[arm] = (n, at)
        if len(rows) == len(a.arms):                       # paired: the unit exists in every arm
            info[code] = dict(panel=panel[code], cost=sum(n * at ** 2 for n, at in rows.values()),
                              n_pool=max(n for n, _ in rows.values()))
    budget = a.budget * gate_cost
    sizes = np.array([v["n_pool"] for v in info.values()])
    t1, t2 = np.percentile(sizes, [100 / 3, 200 / 3]) if len(sizes) else (0, 0)
    for v in info.values():
        v["stratum"] = f"{v['panel']}-{'S' if v['n_pool'] <= t1 else 'M' if v['n_pool'] <= t2 else 'L'}"
    strata = {}
    for code in sorted(info, key=lambda c: hashlib.sha256(f"class_basins:{c}".encode()).hexdigest()):
        strata.setdefault(info[code]["stratum"], []).append(code)
    # the same fraction of every stratum (first parents in hash order, so no bias towards cheap parents), the largest
    # fraction whose estimated cost fits the budget
    chosen, spent = [], 0.0
    for f in np.arange(1.0, 0.0, -0.01):
        pick = [c for s in sorted(strata) for c in strata[s][:int(np.ceil(f * len(strata[s])))]]
        cost = sum(info[c]["cost"] for c in pick)
        if cost <= budget:
            chosen, spent = pick, cost
            break
    sel = dict(arms=a.arms, seed=a.seed, codes=sorted(chosen), budget_factor=a.budget, gate_cost=gate_cost,
               class_cost=spent, cost_ratio=spent / gate_cost if gate_cost else None, tertiles=[float(t1), float(t2)],
               strata={s: sorted(c for c in chosen if info[c]["stratum"] == s) for s in sorted(strata)},
               n_eligible=len(info))
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps({k: v for k, v in sel.items() if k != "strata"} | {"per_stratum": {s: len(v) for s, v in sel["strata"].items()}}))
    return 0


# -- relax ----------------------------------------------------------------------------------------------------------
def _xtb_trace(binary, syms, X, q, mult, td, cycles=1500, timeout=1800):
    """One GFN2 --opt normal; returns crude checkpoint and final (energy kJ/mol, geometry, cycle count)."""
    with open(f"{td}/in.xyz", "w") as f:
        f.write(f"{len(syms)}\n\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(syms, X)))
    for n in ("xtbopt.xyz", "xtbopt.log", "xtbrestart", ".xtboptok", "charges", "wbo", "xtbtopo.mol"):
        Path(td, n).unlink(missing_ok=True)
    cmd = [binary, "in.xyz", "--gfn", "2", "--chrg", str(q), "--uhf", str(mult - 1), "-P", "1",
           "--opt", "normal", "--cycles", str(cycles)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="500M")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout, env=env)
    L = Path(td, "xtbopt.log").read_text().split("\n")
    n, E, G, F, i = len(syms), [], [], [], 0
    while i + 1 < len(L) and L[i].strip():
        m = _LOG_E.search(L[i + 1])
        if not m:
            break
        E.append(float(m.group(1))); G.append(float(m.group(2)))
        F.append(np.array([[float(v) for v in ln.split()[1:4]] for ln in L[i + 2:i + 2 + n]]))
        i += n + 2
    if not E:
        raise ValueError(f"empty xtbopt.log (rc {p.returncode})")
    E = np.array(E)
    jc = next((j for j in range(1, len(E)) if abs(E[j] - E[j - 1]) < ECONV_CRUDE and G[j] < GCONV_CRUDE), None)
    conv = "GEOMETRY OPTIMIZATION CONVERGED" in p.stdout
    jc_ = jc if jc is not None else len(E) - 1
    return dict(converged=conv, cycles=len(E), wall_s=time.perf_counter() - t0, E0_kj=E[0] * H2KJ,
                crude=dict(E_kj=E[jc_] * H2KJ, cycles=jc_ + 1, reached=jc is not None, X=F[jc_].tolist()),
                normal=dict(E_kj=E[-1] * H2KJ, cycles=len(E), X=F[-1].tolist()))


def _relax_one(args):
    binary, syms, X, q, mult, workroot, key = args
    try:
        with tempfile.TemporaryDirectory(dir=workroot) as td:
            return key, _xtb_trace(binary, syms, X, q, mult, td)
    except Exception as exc:
        return key, dict(error=f"{type(exc).__name__}: {exc}")


def cmd_relax(a) -> int:
    from tools.runkit import xtb
    run, out = Path(a.solve), Path(a.out)
    sel = json.loads((out / "selection.json").read_text())
    cm = json.loads((Path(a.inputs) / "charge_mult.json").read_text())
    binary = xtb.require()["path"]
    (out / "units").mkdir(exist_ok=True); work = out / "tmp"; work.mkdir(exist_ok=True)
    jobs, units = [], {}
    for arm in sel["arms"]:
        for code in sel["codes"]:
            dst = out / "units" / f"{arm}_{code}_{sel['seed']}.json"
            if dst.exists():
                continue
            try:
                mp = _manifest(run, arm, code, sel["seed"])
                m, X = _frames(mp)
            except Exception as exc:
                dst.write_text(json.dumps(dict(arm=arm, code=code, error=f"{type(exc).__name__}: {exc}"))); continue
            q, mu = int(cm[code]["charge"]), int(cm[code]["mult"])
            units[(arm, code)] = dict(arm=arm, code=code, seed=sel["seed"], manifest=str(mp), syms=m["ref_symbols"],
                                      ref_X=m["ref_X"], scores=[float(r["score"]) for r in m["rows"]],
                                      top=next((i for i, r in enumerate(m["rows"]) if r.get("is_top1")), 0),
                                      X0=[x.tolist() for x in X], res={}, n=len(X))
            jobs.append((binary, m["ref_symbols"], np.asarray(m["ref_X"], float), q, mu, str(work), (arm, code, "ref")))
            jobs += [(binary, m["ref_symbols"], x, q, mu, str(work), (arm, code, i)) for i, x in enumerate(X)]
    jobs.sort(key=lambda j: -len(j[1]))                                           # big molecules first
    t0, done = time.perf_counter(), 0
    print(f"[class_basins] {len(units)} units, {len(jobs)} relaxations, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(_relax_one, j) for j in jobs]):
            (arm, code, k), r = fut.result()
            u = units[(arm, code)]
            u["res"][str(k)] = r
            done += 1
            if len(u["res"]) == u["n"] + 1:
                (out / "units" / f"{arm}_{code}_{u['seed']}.json").write_text(json.dumps(u))
            if done % 500 == 0:
                print(f"[class_basins {time.perf_counter() - t0:7.1f} s] {done}/{len(jobs)}", flush=True)
    print(f"[class_basins {time.perf_counter() - t0:7.1f} s] relax done", flush=True)
    return 0


# -- analyse --------------------------------------------------------------------------------------------------------
def _rmsd_perm(A, B, P):
    """Heavy-atom RMSD of A onto B minimised over automorphisms P (rows of A permuted), proper rotation."""
    from .classes import core_rmsd
    return core_rmsd(A, B, P, np.arange(A.shape[0]))


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse_unit(u: dict) -> dict:
    from .classes import Reference
    syms, Xr = u["syms"], np.asarray(u["ref_X"], float)
    X0 = np.asarray(u["X0"], float)
    s = np.asarray(u["scores"], float)
    sgn = 1.0 if s[u["top"]] <= s.min() else -1.0                  # rank key: lower = better
    key = sgn * s
    ref = Reference(syms, Xr)
    heavy = ref.heavy
    order = list(np.argsort(key, kind="stable"))
    reps = [("ref", ref)]
    cls = {}
    for i in order:
        for name, R in reps:
            if R.compare(X0[i])["same_class"]:
                cls[int(i)] = name; break
        else:
            name = f"c{int(i)}"
            reps.append((name, Reference(syms, X0[i], graph_X=ref.graph_X, perms=ref.perms, capped=ref.capped)))
            cls[int(i)] = name
    repd = dict(reps)
    out = dict(arm=u["arm"], code=u["code"], seed=u["seed"], n_rows=u["n"], n_classes=len(set(cls.values())),
               ref_class_rows=sum(v == "ref" for v in cls.values()), levels={})
    rr = u["res"].get("ref", {})
    for lvl in ("crude", "normal"):
        e_ref = rr.get(lvl, {}).get("E_kj")
        rows = {i: u["res"].get(str(i), {}) for i in range(u["n"])}
        ok = {i: r[lvl] for i, r in rows.items() if lvl in r}
        left, kept = {}, {}
        for i, r in ok.items():
            Y = np.asarray(r["X"], float)
            if repd[cls[i]].compare(Y)["same_class"]:
                kept.setdefault(cls[i], []).append(i)
            else:
                left[i] = cls[i]
        per = []
        for c, members in kept.items():
            ms = sorted(members, key=lambda i: ok[i]["E_kj"])
            basins = []                                             # [(lead row, [rows])]
            for i in ms:
                Yi = np.asarray(ok[i]["X"], float)[heavy]
                for b in basins:
                    j = b[0]
                    if ok[i]["E_kj"] - ok[j]["E_kj"] <= SAME_E_KJ and \
                            _rmsd_perm(Yi, np.asarray(ok[j]["X"], float)[heavy], ref.perms) <= SAME_RMSD_A:
                        b[1].append(i); break
                else:
                    basins.append((i, [i]))
            best_ranked = min(members, key=lambda i: key[i])
            e_min = ok[ms[0]]["E_kj"]
            d = dict(cls=c, n_members=sum(v == c for v in cls.values()), n_kept=len(members), n_basins=len(basins),
                     collapse=len(basins[0][1]) / len(members), within5=sum(ok[i]["E_kj"] - e_min <= 5 for i in members) / len(members),
                     best_in_lowest=best_ranked in basins[0][1], best_dE=ok[best_ranked]["E_kj"] - e_min,
                     rho=_spearman(key[members], [ok[i]["E_kj"] for i in members]) if len(members) >= 3 else None,
                     cycles_med=float(np.median([ok[i]["cycles"] for i in members])))
            if c == "ref" and e_ref is not None:
                d["min_minus_ref"] = e_min - e_ref
            per.append(d)
        allr = sorted(ok)
        out["levels"][lvl] = dict(classes=per, n_relaxed=len(ok), n_left_class=len(left),
                                  rho_pool=_spearman(key[allr], [ok[i]["E_kj"] for i in allr]),
                                  failed=sum("error" in r for r in rows.values()))
    return out


def cmd_report(a) -> int:
    out = Path(a.out)
    units = sorted((out / "units").glob("*.json"))
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {}
        for p in units:
            u = json.loads(p.read_text())
            if "error" in u:
                res.append(dict(arm=u["arm"], code=u["code"], error=u["error"])); continue
            futs[ex.submit(analyse_unit, u)] = p
        for f in as_completed(futs):
            try:
                res.append(f.result())
            except Exception as exc:
                res.append(dict(path=str(futs[f]), error=f"{type(exc).__name__}: {exc}"))
    (out / "class_basins.json").write_text(json.dumps(res, indent=0, default=float))
    summary = summarise(res, a.boot)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def _boot_mean(per_parent: dict, n=10000, seed=0):
    v = np.array([x for x in per_parent.values() if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=len(v))


def summarise(res, n_boot=10000) -> dict:
    """Per arm and level: parent means of per-class quantities (parent-clustered bootstrap), and paired arm differences."""
    ok = [r for r in res if "error" not in r]
    arms = sorted({r["arm"] for r in ok})
    metrics = {
        "rows_per_class": lambda L, r: [c["n_members"] for c in L["classes"]],
        "classes_per_unit": lambda L, r: [r["n_classes"]],
        "basins_per_class": lambda L, r: [c["n_basins"] for c in L["classes"]],
        "basins_per_class_ge2": lambda L, r: [c["n_basins"] for c in L["classes"] if c["n_kept"] >= 2],
        "collapse_to_lowest": lambda L, r: [c["collapse"] for c in L["classes"] if c["n_kept"] >= 2],
        "within5_of_class_min": lambda L, r: [c["within5"] for c in L["classes"] if c["n_kept"] >= 2],
        "best_ranked_in_lowest": lambda L, r: [float(c["best_in_lowest"]) for c in L["classes"] if c["n_kept"] >= 2],
        "best_ranked_dE_kj": lambda L, r: [c["best_dE"] for c in L["classes"] if c["n_kept"] >= 2],
        "rho_within_class": lambda L, r: [c["rho"] for c in L["classes"] if c["rho"] is not None],
        "rho_pool": lambda L, r: [L["rho_pool"]] if L["rho_pool"] is not None else [],
        "left_class_share": lambda L, r: [L["n_left_class"] / max(1, L["n_relaxed"])],
        "ref_class_min_minus_ref_kj": lambda L, r: [c["min_minus_ref"] for c in L["classes"] if "min_minus_ref" in c],
        "cycles_median": lambda L, r: [c["cycles_med"] for c in L["classes"]],
    }
    out = dict(n_units={a: sum(r["arm"] == a for r in ok) for a in arms}, failed=[r for r in res if "error" in r][:20],
               definitions=dict(same_basin_kj=SAME_E_KJ, same_basin_rmsd_A=SAME_RMSD_A, crude="first cycle |dE|<5e-4 Eh and gnorm<1e-2"),
               levels={})
    for lvl in ("crude", "normal"):
        L = {}
        per = {a: {} for a in arms}
        for r in ok:
            for name, f in metrics.items():
                v = [x for x in f(r["levels"][lvl], r) if x is not None]
                per[r["arm"]].setdefault(name, {})[r["code"]] = float(np.mean(v)) if v else None
        for name in metrics:
            L[name] = {a: _boot_mean(per[a].get(name, {}), n_boot) for a in arms}
            if len(arms) == 2:
                a0, a1 = arms
                common = {c: per[a1][name][c] - per[a0][name][c] for c in per[a0].get(name, {})
                          if per[a1].get(name, {}).get(c) is not None and per[a0][name][c] is not None}
                L[name][f"{a1}-{a0}"] = _boot_mean(common, n_boot)
        out["levels"][lvl] = L
    return out


def main(argv=None) -> int:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--solve", required=True); s.add_argument("--inputs", required=True); s.add_argument("--out", required=True)
    s.add_argument("--arms", nargs="+", required=True); s.add_argument("--seed", type=int, default=31)
    s.add_argument("--gate-seeds", type=int, nargs="+", default=[31, 32, 33])
    s.add_argument("--budget", type=float, default=1.0, help="class-analysis cost / gate relaxation cost estimate")
    r = sub.add_parser("relax")
    r.add_argument("--solve", required=True); r.add_argument("--inputs", required=True); r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=3)
    p.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(argv)
    return dict(select=cmd_select, relax=cmd_relax, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
