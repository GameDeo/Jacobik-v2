"""Re-rank gate (2026-10-15): the cheap, noise-free first test of any scorer change.

Re-scores the conformers of existing pools (written by pools.py, energies by `pools.py mace` / xtb_stage.py) with a
candidate scorer configuration and compares the candidate's #1 against the production #1 on the SAME conformers.
There is no solver noise (the pools are fixed), so every difference is the change's own ranking effect, and the
comparison is paired per complex with a cluster-bootstrap CI.

What it does not measure: the change's effect on GENERATION (which conformers the solver produces). Run the solver
benchmark for that, but only for changes that pass this gate: a change that ranks the same conformers worse has no
mechanism by which steering the solver with it would come out better (2026-10-14 Q6 ablation: the ranking-only order
of the wall / WCA variants matched the full solve order, at a fraction of the cost).

    python tools/jdev/budget/gate.py --sets dev32 n48 --override Q6_MODEL=pmf
    python tools/jdev/budget/gate.py --sets n48 --db data/refit/final/constraints_2026-10-17.json
    python tools/jdev/budget/gate.py --sets dev32 n48 --profile u1      (base = the active profile, normally legacy)
Overrides are attributes of jacobik.ik.scorer (e.g. Q6_MODEL, Q6_AGG, RANK_AGG, Q1_TAIL_SIGMA, SCORE_AGG; the module
globals jacobik/profile.py fills from the active profile), or module:ATTR for another module
(--override jacobik.stats.q6pmf:PMF_WELLS=repulsive);
values are parsed as float when possible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, pool_energy, pool_files, read_pool, stats_from_db  # noqa: E402
from pools import DB, OUT, check_db, set_cases, setup  # noqa: E402

KCAL = 4.184


def _parse(kv):
    """KEY=VALUE -> {KEY: value} (KEY an attribute of jacobik.ik.scorer, or module:ATTR, e.g.
    jacobik.stats.q6pmf:PMF_WELLS=repulsive); values: true / false, float, else string."""
    out = {}
    for s in kv or []:
        k, v = s.split("=", 1)
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = float(v)
            except ValueError:
                pass
        out[tuple(k.split(":", 1)) if ":" in k else k] = v
    return out


def _log(*a):
    print(*a, flush=True)


def profile_overrides(name):
    """{(module, attribute): value} of every field of profile `name` (jacobik/profile.py), to apply in-process."""
    from jacobik.core import profile as P
    vals = {k: v[2] for k, v in P.FIELDS.items()}
    vals.update(P.PROFILES[name])
    return {(P.FIELDS[k][0], P.FIELDS[k][1]): v for k, v in vals.items()}


def rescore(sets, overrides, db=None, start="etkdg", seeds=None, log=_log):
    """{pool: dict(base=..., cand=..., extra=..., mace=..., gsp=..., gbasin=...)} for every readable pool.
    `overrides`: {attribute of jacobik.ik.scorer or (module, attribute): value} for the candidate."""
    setup()
    import importlib
    import torch
    import jacobik.ik.scorer as SC
    from tools.jdev.cases import get_case

    overrides = {(k if isinstance(k, tuple) else ("jacobik.ik.scorer", k)): v for k, v in overrides.items()}
    oxidation_only = set(overrides) == {("jacobik.core.chemistry", "OXIDATION_MODEL")} and db is None
    mods = {m: importlib.import_module(m) for m, _ in overrides}

    check_db()
    st_base = stats_from_db(DB)
    st_cand = stats_from_db(Path(db)) if db else st_base
    names = {}
    for s in sets:
        names.update({c: n for c, n in set_cases(s)})
    saved = {(m, k): getattr(mods[m], k) for m, k in overrides}
    out = {}
    # one scorer per (complex, arm), reused across the complex's seeds: the scorer depends on the complex and the
    # settings, not on the seed (2026-10-17: it was rebuilt, embed included, for every pool and arm). Checked per pool
    # below: the base re-score must equal the pool's stored ranking score.
    cache: dict = {}
    for s in sets:
        for _, code, seed, f in pool_files(start, [s], OUT, seeds):
            z, e32, e64 = read_pool(f), pool_energy(f, "mace"), pool_energy(f, "mace64")
            if z is None or (e32 is None and e64 is None):  # pools since 10-17 have float64 MACE only
                continue
            X = torch.tensor(z["X"], dtype=torch.float32)
            rg = torch.as_tensor(z["asg"], dtype=torch.long)
            res = {}
            if not any(k[0] == (s, code) for k in cache):
                cache.clear()  # pools come sorted by complex: keep only the current one
            try:
                case = get_case(names[code])
                for tag, st, ov in (("base", st_base, {}), ("cand", st_cand, overrides)):
                    if tag == "cand" and not overrides and db is None:
                        res["cand"] = res["base"].copy()
                        res["zbend_cand"] = res["zbend_base"]
                        continue
                    for (m, k), v in saved.items():
                        setattr(mods[m], k, v)
                    for (m, k), v in ov.items():
                        setattr(mods[m], k, v)
                    builder = None
                    if tag == "cand" and oxidation_only:
                        builder = case.make_builder()
                        if builder.oxidation_state == cache[((s, code), "base_os")]:
                            # This flag is consumed only by query deduction. If the
                            # resulting OS is identical, every scorer input and key
                            # is identical; avoid rebuilding the same beam/IK setup.
                            res["cand"] = res["base"].copy()
                            res["zbend_cand"] = res["zbend_base"]
                            res["reused_unchanged_oxidation"] = True
                            continue
                    if ((s, code), tag) not in cache:
                        builder = builder or case.make_builder()
                        if tag == "base":
                            cache[((s, code), "base_os")] = builder.oxidation_state
                        r = builder.solve(seed=int(seed), stats=st, angle_bends=True, n_conformers=1,
                                          n_ik_steps=2, device="cpu")
                        cache[((s, code), tag)] = r.scorer
                    scorer = cache[((s, code), tag)]
                    res[tag] = scorer.ranking_score(X, row_groups=rg).numpy()
                    # bend prior in score units (profile field bend_per_z): the stored extra is in raw nats
                    import jacobik.ik.ik_solver as IK
                    res[f"zbend_{tag}"] = float(scorer._agg_norm()) if IK._BEND_PER_Z else 1.0
                dev = float(np.abs(res["base"] - z["rank"]).max()) if "rank" in z.files else 0.0
                res["base_max_diff"] = dev
                if dev > 1e-4:
                    log(f"  {code} s{seed}: base re-score differs from the stored ranking score by {dev:.2e}")
            except Exception as exc:
                log(f"  {code} s{seed}: skipped ({type(exc).__name__}: {str(exc)[:100]})")
                continue
            finally:
                for (m, k), v in saved.items():
                    setattr(mods[m], k, v)
            # the bend-prior / ring-closure part of the pool's score (not part of ranking_score), kept for both arms in
            # RAW nats: pools solved with bend_per_z (profile u1) stored it divided by Z
            res["extra"] = z["score"] - z["rank"]
            if "profile" in z.files and json.loads(str(z["profile"]))["fields"].get("bend_per_z"):
                res["extra"] = res["extra"] * float(z["Z"])
            if e32 is not None:
                res["mace"] = e32
            if e64 is not None:  # float64 MACE (pools.py mace --dtype float64), --energy mace64
                res["mace64"] = e64
            xf = f.with_suffix(".xtb.json")
            if xf.exists():
                x = json.loads(xf.read_text())
                if x.get("ref_sp") is not None and "sp" in x:
                    res["gsp"] = np.array([(e - x["ref_sp"]) * 2625.5 if e is not None else np.nan for e in x["sp"]])
            out[f"{s}|{code}|{seed}"] = res
            log(f"  {s}/{code} s{seed}: {len(X)} conformers")
    return out


def compare(scores, energy="mace", n_boot=4000, seed=0):
    """Paired comparison (per complex, mean over seeds) of the candidate's #1 vs the production #1."""
    rng = np.random.default_rng(seed)
    rows = []
    for pool, r in scores.items():
        if energy not in r:
            continue
        e = r[energy]
        ok = np.isfinite(e)
        if ok.sum() < 2:
            continue
        b = (r["base"] + r["extra"] / r.get("zbend_base", 1.0))[ok]
        c = (r["cand"] + r["extra"] / r.get("zbend_cand", 1.0))[ok]
        e = e[ok]
        ib, ic = int(np.argmax(b)), int(np.argmax(c))
        s, case, sd = pool.split("|")
        rows.append(dict(set=s, case=case, seed=int(sd), base=e[ib], cand=e[ic], best=e.min(),
                         top1_base=e[ib] <= e.min() + KCAL, top1_cand=e[ic] <= e.min() + KCAL))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    pc = df.assign(d=df.cand - df.base).groupby(["set", "case"]).d.mean().to_numpy()
    bs = np.array([rng.choice(pc, len(pc)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    summary = dict(energy=energy, pools=len(df), complexes=len(pc), pick_base_median=df.base.median(),
                   pick_cand_median=df.cand.median(), mean_diff=pc.mean(), ci_lo=lo, ci_hi=hi,
                   better=int((pc < -KCAL).sum()), worse=int((pc > KCAL).sum()),
                   top1_base=df.top1_base.mean(), top1_cand=df.top1_cand.mean(),
                   verdict="PASS (ranks better)" if hi < 0 else "FAIL (ranks worse)" if lo > 0 else "no detectable change")
    return df, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=["dev32"])
    p.add_argument("--override", nargs="*", default=[], help="scorer module attributes, KEY=VALUE")
    p.add_argument("--profile", default=None, help="candidate = this profile (jacobik/profile.py), then --override")
    p.add_argument("--dump", default=None, help="write every pool's full scores {pool: {base: [...], cand: [...]}} "
                                                "(ranking score + bend prior in the arm's units) to this JSON")
    p.add_argument("--db", default=None, help="candidate constraints DB (default: the pools' DB)")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--start", default="etkdg", help="pool directory under <OUT>/pools (e.g. etkdg_u1)")
    p.add_argument("--energy", nargs="+", default=["mace", "gsp"])
    p.add_argument("--csv", default=None, help="write the per-pool picks (base / candidate / best) to <csv>_<energy>.csv")
    p.add_argument("--report", default=None, help="JSON report, including base re-score mismatch counts")
    a = p.parse_args()
    setup()  # src on sys.path before profile_overrides imports jacobik
    ov = profile_overrides(a.profile) if a.profile else {}
    ov.update(_parse(a.override))
    sc = rescore(a.sets, ov, a.db, start=a.start, seeds=a.seeds)
    checks = dict(rescored_pools=len(sc), base_mismatches=sum(r['base_max_diff'] > 1e-4 for r in sc.values()),
                  base_max_diff=max((r['base_max_diff'] for r in sc.values()), default=0.0),
                  reused_unchanged_oxidation=sum(r.get('reused_unchanged_oxidation', False) for r in sc.values()))
    print("base check:", json.dumps(checks), flush=True)
    summaries = {}
    validity_path = REPO / "data/benchmark_runs/2026-10-16_post_refit/donor_atoms.json"
    validity = json.loads(validity_path.read_text()) if validity_path.exists() else {}
    bad_inputs = {code for code, v in validity.items() if isinstance(v, dict) and not v.get("valid_atoms", True)}
    valid_sc = {k: r for k, r in sc.items() if k.split('|')[1] not in bad_inputs}
    if a.dump:
        Path(a.dump).write_text(json.dumps({k: {t: (r[t] + r["extra"] / r.get(f"zbend_{t}", 1.0)).tolist()
                                                for t in ("base", "cand")} for k, r in sc.items()}))
    for e in a.energy:
        df, s = compare(sc, e)
        if a.csv and not df.empty:
            df.to_csv(f"{a.csv}_{e}.csv", index=False)
        if s:
            summaries[e] = s
            print(f"\n[{e}] " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
        if bad_inputs:
            valid_df, valid_summary = compare(valid_sc, e)
            if valid_summary:
                summaries[f"{e}_valid_inputs"] = valid_summary
                print(f"\n[{e}, valid donor inputs] " + json.dumps(valid_summary), flush=True)
                if a.csv:
                    valid_df.to_csv(f"{a.csv}_{e}_valid_inputs.csv", index=False)
    if a.report:
        Path(a.report).write_text(json.dumps(dict(checks=checks, summaries=summaries), indent=2)+'\n')
    if not sc or checks['base_mismatches']:
        raise SystemExit("Gate invalid: empty re-score or base scorer drift")


if __name__ == "__main__":
    main()
