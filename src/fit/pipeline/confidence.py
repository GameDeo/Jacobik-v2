"""Frozen contamination designs and structural failure summaries.

All fractions refer to independent training parents of the resolved fit.
The scorer supplies trial scores on a fixed pool, or scores for generated
rows. This module never changes generator settings.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import tempfile
import traceback
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence
import numpy as np
import pandas as pd
from fit import expander as EX
from fit import pooling as PL
from jacobik.stats import mixture as MX

SCENARIOS = ("observed_input_error", "parent_removal", "geometry_0.02A")
DOSES = (0.01, 0.02, 0.05)


def perturb_coordinates(positions, *, seed, maximum_A=0.02):
    """Independent uniform-ball displacements bounded by the frozen amplitude."""
    xyz = np.asarray(positions, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or maximum_A < 0:
        raise ValueError("invalid geometry or perturbation amplitude")
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=xyz.shape)
    norm = np.linalg.norm(direction,axis=1,keepdims=True)
    direction /= np.maximum(norm,np.finfo(float).tiny)
    displacement = maximum_A * rng.random((len(xyz),1)) ** (1/3) * direction
    return xyz + displacement


def perturb_observations(rows, trial, scenario, replacements=None):
    """Whole-parent damage applied to a retained observation table.

    Geometry scenarios must be re-extracted from perturbed source coordinates
    by the parent runner. Replacements carry those resulting observations.
    """
    from jacobik.stats.mixture import structure_id

    if scenario not in SCENARIOS:
        raise ValueError("unregistered corruption mechanism")
    affected = set(trial["affected_parents"])
    codes = rows["csd_code"].astype(str).map(structure_id)
    present = set(codes)
    if not affected <= present:
        raise ValueError("damage set contains evidence outside this fit")
    keep = rows.loc[~codes.isin(affected)].copy()
    if scenario == "parent_removal":
        return keep
    if replacements is None:
        raise ValueError("source-supported replacement observations required")
    rcodes = replacements["csd_code"].astype(str).map(structure_id)
    if not affected <= set(rcodes):
        raise ValueError("replacement evidence missing an affected parent")
    import pandas as pd
    return pd.concat([keep,replacements.loc[rcodes.isin(affected)]],ignore_index=True)


def refit_independent_q1(rows, key):
    """Exact production independent fit for supported Q1 keys.

    Sparse keys require their ancestor dependency closure and are explicitly
    rejected here, so an independent fit cannot masquerade as partial pooling.
    """
    from jacobik.stats import mixture
    from fit import pooling
    from fit import q1
    if "key" in rows and not (rows["key"] == repr(tuple(key))).all():
        raise ValueError("independent refit contains observations from other keys")

    ids = mixture.structure_ids(rows["csd_code"].astype(str).to_numpy())
    count = len(set(ids))
    if count < q1.N_INDEP:
        raise ValueError("sparse Q1 fit requires ancestor refitting")
    import pandas as pd
    multiplicity = pd.Series(ids).groupby(ids).transform("size").to_numpy()
    _, fit, error = pooling.fit_one_checked(
        (repr(tuple(key)), rows["value"].to_numpy(float),1/multiplicity,"gauss",q1.VAR_FLOOR))
    if error or fit is None:
        raise ValueError(error or "Q1 fit failed")
    return mixture.to_entry(fit,"gmm",len(rows),count,pooled=True,src="fit")


def design(codes, *, seed=11, replicates=32):
    """Nested random damage sets, including the minimum one-parent dose."""
    from jacobik.stats.mixture import structure_id

    parents = sorted({structure_id(code) for code in codes})
    if not parents or replicates < 1:
        raise ValueError("confidence design requires parents and positive replicates")
    counts = sorted({1, *(max(1, math.ceil(f * len(parents))) for f in DOSES)})
    rng = np.random.default_rng(seed)
    trials = []
    for replicate in range(replicates):
        ordered = rng.permutation(parents).tolist()
        for count in counts:
            trials.append(dict(replicate=replicate, affected_parents=ordered[:count],
                               affected_count=count, support_count=len(parents),
                               actual_fraction=count / len(parents)))
    return trials


def bootstrap_design(codes, *, seed=11, replicates=64):
    from jacobik.stats.mixture import structure_id
    parents = sorted({structure_id(code) for code in codes})
    if not parents or replicates < 1:
        raise ValueError("bootstrap requires independent source parents")
    rng = np.random.default_rng(seed)
    return [rng.choice(parents,len(parents),replace=True).tolist() for _ in range(replicates)]


def wilson(failures, total):
    if not 0 <= failures <= total or total < 1:
        raise ValueError("invalid Bernoulli counts")
    z = 1.959963984540054
    p = failures / total
    denominator = 1 + z*z / total
    centre = (p + z*z / (2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, centre-half), min(1.0, centre+half)]


def evaluate(clean, trial):
    """Three seed-paired selected structures, using the recorded row energies."""
    energies, broken, floors = [], [], []
    for name, data in (("clean", clean), ("trial", trial)):
        scores = np.asarray(data["scores"], dtype=float)
        energy = np.asarray(data["energies_kj"], dtype=float)
        invalid = np.asarray(data["broken_coordination"])
        if invalid.dtype.kind != "b":
            raise ValueError(f"{name}: coordination indicators must be explicit booleans")
        if scores.ndim != 2 or scores.shape[0] != 3 or scores.shape != energy.shape or scores.shape != invalid.shape:
            raise ValueError(f"{name}: expected three seed x row arrays")
        if not np.isfinite(scores).all() or not np.isfinite(energy).all() or scores.shape[1] == 0:
            raise ValueError(f"{name}: nonfinite or empty result is an execution failure")
        selected = np.argmax(scores, axis=1)
        energies.append(energy[np.arange(3), selected])
        broken.append(invalid[np.arange(3), selected])
        floors.append(energy.min(axis=1))
    loss = energies[1] - energies[0]
    new_break = broken[1] & ~broken[0]
    severe_seed = (loss > 25) | new_break
    return dict(energy_loss_kj=loss.tolist(), floor_change_kj=(floors[1]-floors[0]).tolist(),
                new_broken_coordination=new_break.tolist(),
                severe=bool(severe_seed.sum() >= 2),
                single_seed_severe=bool(severe_seed.any()))


def summarize(records, *, level):
    """One failure draw is one corruption trial, with all three seeds paired."""
    if level not in ("fixed_pool", "generation"):
        raise ValueError("confidence level must identify fixed_pool or generation")
    if not records:
        raise ValueError("confidence requires completed or failed trial records")
    groups = defaultdict(list)
    failures = []
    for row in records:
        if row["status"] != "ok":
            failures.append(row["trial_id"])
            continue
        dose = row["actual_fraction"]
        if not 0 < dose <= 1 or row["affected_count"] / row["support_count"] != dose:
            raise ValueError("invalid local contamination fraction")
        groups[row["scenario"], dose].append(row)
    result = []
    for (scenario, dose), rows in sorted(groups.items()):
        ids = [r["trial_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate trial IDs")
        severe = sum(r["outcome"]["severe"] for r in rows)
        result.append(dict(scenario=scenario, actual_fraction=dose, completed_trials=len(rows),
                           severe_trials=severe, failure_rate=severe/len(rows),
                           interval95=wilson(severe,len(rows))))
    first = {}
    for row in result:
        if row["severe_trials"]:
            first.setdefault(row["scenario"], row["actual_fraction"])
    maxima = {scenario:max(row["actual_fraction"] for row in result if row["scenario"]==scenario)
              for scenario in {row["scenario"] for row in result}}
    return dict(level=level, dose_response=result, first_tested_failing_fraction=first,
                maximum_tested_fraction=maxima,
                censored_at_maximum_tested={scenario: maximum for scenario, maximum in maxima.items()
                                           if scenario not in first and not failures},
                execution_failures=failures, candidate_blocked=bool(first or failures),
                interpretation="Tolerance describes only the tested mechanisms, doses and coverage.")


def run_task(parent):
    """tools.run task consuming immutable score/energy trial evidence."""
    import hashlib
    import json
    from pathlib import Path

    path = Path(parent["evidence"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != parent["sha256"]:
        raise ValueError("confidence evidence changed after registration")
    evidence = json.loads(raw)
    scope = evidence["scope"]
    if not all(k in scope for k in ("quantity","resolved_key","parent_code","support_count")):
        raise ValueError("confidence evidence requires the resolved fit and target scope")
    if any(r.get("status")=="ok" and r["support_count"]!=scope["support_count"] for r in evidence["trials"]):
        raise ValueError("mixed independent-support denominators")
    return dict(summary=summarize(evidence["trials"], level=evidence["level"]),
                scope=scope, evidence_sha256=parent["sha256"])


# ---------------------------------------------------------------------------
# Trial registry & completeness accounting
# ---------------------------------------------------------------------------


REGISTRY_VERSION = "trial_registry.1"
#: the coordination indicator entering ``confidence.evaluate``: explicit booleans per seed x row, true when the
#: row's metal coordination is broken (a registered donor beyond contact range or a lost site); the same function
#: computes it for clean and trial arms.
BROKEN_COORDINATION_RULE = "explicit_boolean_per_seed_row.v1"


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def trial_id(scope: Mapping, scenario: str, replicate: int, count: int) -> str:
    return _h(["trial.1", scope["quantity"], scope["resolved_key"], scope["parent_code"], scenario, replicate, count])[:24]


def register(scope: Mapping, codes: Iterable[str], inputs: Mapping[str, str], *,
             scenarios: Iterable[str] = SCENARIOS, seed: int = 11, replicates: int = 32,
             bootstrap_replicates: int = 64, seeds=(11, 12, 13)) -> dict:
    """Freeze the expected trials of one resolved fit and target. ``inputs`` maps names to immutable sha256 values."""
    for k in ("quantity", "resolved_key", "parent_code"):
        if k not in scope:
            raise ValueError(f"registry scope requires {k}")
    if not inputs or any(not isinstance(v, str) or len(v) != 64 for v in inputs.values()):
        raise ValueError("registry requires sha256 hashes for every immutable input")
    plan = design(codes, seed=seed, replicates=replicates)
    support = plan[0]["support_count"]
    trials = []
    for scenario in scenarios:
        for t in plan:
            trials.append(dict(trial_id=trial_id(scope, scenario, t["replicate"], t["affected_count"]), scenario=scenario,
                               replicate=t["replicate"], affected_count=t["affected_count"],
                               actual_fraction=t["actual_fraction"], affected_parents=t["affected_parents"]))
    boots = bootstrap_design(codes, seed=seed, replicates=bootstrap_replicates)
    reg = dict(version=REGISTRY_VERSION, scope=dict(scope), support_count=support, seeds=list(seeds),
               seed=seed, replicates=replicates, broken_coordination_rule=BROKEN_COORDINATION_RULE,
               inputs=dict(inputs), trials=trials, bootstrap_sha256=_h(boots), bootstrap_replicates=len(boots))
    reg["registry_id"] = _h({k: v for k, v in reg.items() if k != "registry_id"})
    ids = [t["trial_id"] for t in trials]
    if len(set(ids)) != len(ids):
        raise ValueError("registered trial IDs are not unique")
    return reg


def reconcile(registry: Mapping, records: Iterable[Mapping]) -> dict:
    """Expected vs returned trials. ``complete`` is true only when every registered trial returned ok, exactly once."""
    records = list(records)
    expected = {t["trial_id"] for t in registry["trials"]}
    counts = Counter(r.get("trial_id") for r in records)
    ok = {r["trial_id"] for r in records if r.get("status") == "ok"}
    failed = sorted({r["trial_id"] for r in records if r.get("status") != "ok" and r.get("trial_id") in expected})
    missing = sorted(expected - set(counts))
    duplicates = sorted(i for i, c in counts.items() if c > 1)
    unexpected = sorted(i for i in counts if i not in expected)
    wrong_support = sorted(r["trial_id"] for r in records if r.get("status") == "ok"
                           and r.get("support_count") != registry["support_count"])
    wrong_rule = sorted(r["trial_id"] for r in records if r.get("status") == "ok"
                        and r.get("broken_coordination_rule") != registry["broken_coordination_rule"])
    return dict(expected=len(expected), completed=len(ok & expected), failed=failed, missing=missing,
                duplicates=duplicates, unexpected=unexpected, wrong_support=wrong_support, wrong_rule=wrong_rule,
                complete=not (failed or missing or duplicates or unexpected or wrong_support or wrong_rule))


def summarize_registered(registry: Mapping, records: Iterable[Mapping], *, level: str) -> dict:
    """``confidence.summarize`` over registered trials; any gap is an execution failure that blocks acceptance."""
    records = list(records)
    rec = reconcile(registry, records)
    if rec["duplicates"] or rec["unexpected"]:
        raise ValueError(f"records do not match the registry: duplicates={rec['duplicates'][:3]} "
                         f"unexpected={rec['unexpected'][:3]}")
    expected = {t["trial_id"]: t for t in registry["trials"]}
    padded = [r for r in records if r.get("status") != "ok" or
              (r.get("support_count") == registry["support_count"]
               and r.get("broken_coordination_rule") == registry["broken_coordination_rule"])]
    padded += [dict(status="failed", trial_id=i, reason="wrong support or coordination rule")
               for i in rec["wrong_support"] + rec["wrong_rule"]]
    padded = [r for r in padded if r["trial_id"] not in set(rec["wrong_support"] + rec["wrong_rule"])
              or r.get("status") != "ok"]
    padded += [dict(status="failed", trial_id=i, reason="missing") for i in rec["missing"]]
    out = summarize(padded, level=level)
    out["registry_id"] = registry["registry_id"]
    out["reconciliation"] = {k: (v if not isinstance(v, list) else len(v)) for k, v in rec.items()}
    out["candidate_blocked"] = bool(out["candidate_blocked"] or not rec["complete"])
    return out


# ---------------------------------------------------------------------------
# Quantity-specific stress refits
# ---------------------------------------------------------------------------


def _spec(quantity: str):
    from fit import q1, q2
    if quantity == "Q1":
        return q1.N_INDEP, q1.DEFAULT_POOLING_R, q1.DEFAULT_VARIANT
    if quantity == "Q2":
        return q2.N_INDEP, q2.DEFAULT_POOLING_R, None
    raise ValueError(f"no pooled refit for {quantity}")


def expand_rows(raw: pd.DataFrame, quantity: str, *, workdir: Optional[Path] = None) -> pd.DataFrame:
    """Ancestor-expanded table of a raw observation frame (columns csd_code, key, value, metal, coord_geom[, obs_id])."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)
        (tmp / "in").mkdir()
        raw.to_parquet(tmp / "in" / f"observations_{quantity}.parquet", index=False)
        EX.expand_quantity(tmp / "in", tmp / "out", quantities=[quantity])
        return pd.read_parquet(tmp / "out" / f"expanded_{quantity}.parquet",
                               columns=["key", "value", "csd_code", "level", "is_l0"])


def closure_keys(pf: PL.PooledFitter, key: str, variant: Optional[str]) -> List[str]:
    """The resolved key followed by the chain of parents it is adapted from, down to a chain root."""
    par = pf.parents(variant)
    if key not in pf.data:
        raise KeyError(f"{key} has no data in this expansion")
    chain, seen = [key], {key}
    while par.get(chain[-1]) is not None:
        nxt = par[chain[-1]]
        if nxt in seen:
            raise ValueError("cyclic parent chain")
        chain.append(nxt)
        seen.add(nxt)
    return chain


def refit_closure(pf: PL.PooledFitter, key: str, *, variant: Optional[str] = None, r: Optional[float] = None) -> dict:
    """The production entry of ``key``: independent fits where the chain needs them, adaptation otherwise.

    Mirrors ``PooledFitter.entries`` for one key without fitting the rest of the table."""
    _, r_default, variant_default = _spec(pf.q)
    variant = variant if variant is not None else variant_default
    r = r_default if r is None else r
    chain = closure_keys(pf, key, variant)
    independent = set(pf.independent_keys())
    pf.fit_independent([k for k in chain if k in independent])
    par = pf.parents(variant)
    memo: Dict[str, Optional[dict]] = {}

    def get(k: str):
        if k in memo:
            return memo[k]
        f = pf.indep.get(k)
        src = "fit"
        if f is None:
            p = par.get(k)
            pfit = get(p) if p is not None else None
            if pfit is None:
                memo[k] = None
                return None
            x, w = pf.data[k]
            f = MX.adapt(pfit, x, w, r, pf.var_floor)
            src = "adapted"
        memo[k] = dict(f, src=src)
        return memo[k]

    fit = get(key)
    if fit is None:
        raise ValueError(f"{key}: no fit along its chain {chain}")
    return MX.to_entry(fit, pf.typ, pf.n_rows[key], pf.n_cx.get(key, 0), pooled=True, src=fit["src"])


def feeds_closure(raw: pd.DataFrame, quantity: str, chain: Sequence[str]) -> pd.Series:
    """Rows of ``raw`` whose own key or ancestor keys include a chain key (exact input of a stable-sampling refit)."""
    wanted = set(chain)
    fn = EX._ANCESTOR_FN[quantity]
    cache: Dict[str, bool] = {}

    def hit(k: str) -> bool:
        if k not in cache:
            tup = ast.literal_eval(k)
            keys = {k}
            try:
                keys |= {repr(anc) for _, anc in fn(tup)}
            except Exception:
                pass
            cache[k] = bool(keys & wanted)
        return cache[k]

    return raw["key"].map(hit)


def refit_resolved(raw: pd.DataFrame, quantity: str, key: str, *, workdir: Optional[Path] = None,
                   stable: bool = False, chain: Optional[Sequence[str]] = None) -> dict:
    """Entry of ``key`` refitted from raw observations through the whole dependency closure.

    ``stable`` must equal the sampler the observations were extracted for; a restricted expansion (rows feeding the
    chain only) is used only then, because the legacy sampler's shared random stream makes it inexact."""
    if stable and chain is not None:
        raw = raw.loc[feeds_closure(raw, quantity, chain)]
    expanded = expand_rows(raw, quantity, workdir=workdir)
    n_indep, _, _ = _spec(quantity)
    pf = PL.PooledFitter(quantity, expanded, n_indep=n_indep)
    return refit_closure(pf, key)


def refit_qo(raw_qo: pd.DataFrame, group_key: tuple) -> dict:
    """The production QO fit of one ladder group (e.g. the all-donor fallback ``(2, 4)``) from raw QO rows.

    QO groups are independent fits: the same function ``fit_stage`` runs per group, on the same group assembly
    (haptic duplication, H-bond cut, one LnQM sibling per ligand set above the metal level)."""
    from jacobik.stats import orientation as O
    from fit import qo as QO

    groups = QO.training_groups(QO.fit_rows(raw_qo), n_min=O.N_MIN, hb_cut=O.HB_CUT)
    if tuple(group_key) not in groups:
        raise KeyError(f"QO group {tuple(group_key)} has fewer than {O.N_MIN} training rows after damage")
    G = np.ascontiguousarray(groups[tuple(group_key)])
    return QO._fit_one(G, QO.group_seed(tuple(group_key)), O.K_MAX, O.COV_FLOOR, O.W_MIN)


# ---------------------------------------------------------------------------
# Registered confidence driver
# ---------------------------------------------------------------------------


def fitdist_kwargs(entry: Mapping) -> dict:
    """DB-entry dict -> ``FittedDist`` keyword arguments (extra bookkeeping fields such as ``src``/``k`` are dropped)."""
    from jacobik.stats.stats import FittedDist

    names = {f.name for f in dataclasses.fields(FittedDist)}
    return {k: v for k, v in entry.items() if k in names}


def refit_trial(raw: pd.DataFrame, quantity: str, key: str, trial: Mapping, scenario: str, *,
                stable: bool, chain: Optional[Sequence[str]] = None, replacements: Optional[pd.DataFrame] = None,
                workdir=None) -> dict:
    """Entry of ``key`` refitted from ``raw`` after one registered damage draw.

    ``geometry_*`` scenarios need the affected parents re-extracted from perturbed source coordinates: the caller passes those
    rows as ``replacements`` (see ``stress_geometry.geometry_replacements``). Without them the trial fails (never passes)."""
    damaged = perturb_observations(raw, trial, scenario, replacements)
    if quantity == "QO":
        return refit_qo(damaged, tuple(__import__("ast").literal_eval(key)) if isinstance(key, str) else tuple(key))
    return refit_resolved(damaged, quantity, key, workdir=workdir, stable=stable, chain=chain)


def run_registered(reg: Mapping, raw: pd.DataFrame, score_fn: Callable[[Optional[dict]], dict], *,
                   level: str = "fixed_pool", stable: bool = True, chain: Optional[Sequence[str]] = None,
                   replacements: Optional[pd.DataFrame] = None, only: Optional[Iterable[str]] = None,
                   workdir=None, log: Callable[[str], None] = lambda s: None) -> dict:
    """Execute every registered trial (or the ``only`` subset) and return records plus the registry summary."""
    scope = reg["scope"]
    quantity, key = scope["quantity"], scope["resolved_key"]
    clean = score_fn(None)
    wanted = set(only) if only is not None else None
    records: List[Dict] = []
    for t in reg["trials"]:
        if wanted is not None and t["trial_id"] not in wanted:
            continue
        base = dict(trial_id=t["trial_id"], scenario=t["scenario"], replicate=t["replicate"],
                    affected_count=t["affected_count"], actual_fraction=t["actual_fraction"],
                    support_count=reg["support_count"], broken_coordination_rule=reg["broken_coordination_rule"])
        try:
            entry = refit_trial(raw, quantity, key, t, t["scenario"], stable=stable, chain=chain,
                                replacements=replacements, workdir=workdir)
            outcome = evaluate(clean, score_fn(fitdist_kwargs(entry)))
            records.append(dict(base, status="ok", outcome=outcome))
        except Exception as exc:  # a trial failure is data, not a crash
            records.append(dict(base, status="failed", reason=f"{type(exc).__name__}: {exc}",
                                trace=traceback.format_exc(limit=3)))
        log(f"{t['trial_id']} {t['scenario']} n={t['affected_count']} -> {records[-1]['status']}")
    summary = summarize_registered(reg, records, level=level) if (wanted is None) else None
    return dict(records=records, summary=summary, reconciliation=reconcile(reg, records))
