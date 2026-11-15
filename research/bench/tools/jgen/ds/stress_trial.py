"""Fixed-pool scorer for the stress driver: re-score the saved D3 pools of one case under a refitted Q1/QO entry.

``prepare(parent)`` loads the frozen DB, the source geometry and the three seed pools of the native arm once and returns a
``score_fn(fit_kwargs | None)`` for ``fit.pipeline.confidence.run_registered``. The clean call (None) scores with the
frozen DB; a trial call overrides every requested lookup key that resolves to the refitted entry's key
(``Traced.override`` from ``cross_score``), so a sparse key's descendants pick it up exactly as in a rebuilt DB.

Scores are the statistical scorer terms without the bend prior, identical for the clean and the trial arm; selection is
argmax per seed. The broken-coordination boolean (rule ``explicit_boolean_per_seed_row.v1``) is true for a row in which
any metal-donor Q1 distance exceeds ``BROKEN_RATIO`` times that donor's source distance.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.jgen.ds import cross_score as CS

BROKEN_RATIO = 1.25   # provisional; the Step 5 contract fixes the final value before any result is opened
SEEDS = (11, 12, 13)


def manifest_row(root, code, runtime="D"):
    rows = [r for r in CS.manifest(Path(root)) if r["code"] == code and r["runtime"] == runtime]
    if len(rows) != 1:
        raise KeyError(f"no unique manifest row for {code}/{runtime}")
    return rows[0]


def _load(parent, score_db=None):
    """Heavy objects shared by the stress scorer and the DB comparison: source molecule, the three native seed pools and the
    builder of scorers; the lookup of the frozen (generating) DB."""
    import numpy as np
    import torch
    from tools.jgen.bench.run import _setup_env

    native = parent["runtime"]
    flags = json.loads(Path(parent["paths"][native][0]["record"]).read_text())["flags"]
    _setup_env(flags)  # before the first jacobik import
    from jacobik.stats.stats import FittedDist, load_db
    from tools.jgen.inputs import load_mol

    torch.set_num_threads(1)
    dbs = parent["dbs"]
    for db in dbs.values():
        path = Path(db["path"])
        if CS._sha(path) != db["sha256"] or CS._sha(path.with_suffix(".meta.json")) != db["meta_sha256"]:
            raise ValueError("database or sidecar changed after manifest freeze")
    Traced = CS._lookup_type()
    native_db = "C" if native == "C" else "D"
    # pools are validated against the generating DB above; the scoring DB (the stress base) may differ
    lookup = Traced(load_db(score_db or dbs[native_db]["path"]))
    mol = load_mol(parent["code"], parent["shard"], parent["dataset"], clean_root=Path(parent["ref_dir"]))
    if mol is None:
        raise ValueError("source geometry missing")
    pools = [CS._read_pool(pair, dbs) for pair in parent["paths"][native]]
    state = {}
    with torch.no_grad():
        for variant, ctx, assignments, _ in CS._source_scorers(parent, mol, {native_db: lookup}):
            signature = CS._signature(ctx, assignments)
            build, scorers = CS._build_scorers(ctx, assignments, {native_db: lookup})
            per_seed = []
            for rec, m, frames in pools:
                ii, X, groups = CS._pose_rows(frames, variant, signature)
                if not ii:
                    continue
                energy = np.array([m["rows"][i]["mace_energy_kj"] for i in ii], float)
                per_seed.append((rec["seed"], X, groups, energy))
            if len(per_seed) == len(SEEDS):
                state = dict(variant=variant, build=build, per_seed=sorted(per_seed, key=lambda t: t[0]), ge=ctx and scorers[native_db]._ge)
                break
    if not state:
        raise ValueError("no EZ variant has all three seed pools")
    n_rows = {len(t[3]) for t in state["per_seed"]}   # seed pools may differ in size; _score pads the shorter ones
    # donor distances in the source geometry define "broken"
    ref_X = torch.as_tensor(np.asarray([a.position for a in mol.atoms]), dtype=torch.float32)[None]
    first_map = np.argsort(pools[0][1]["gen_to_ref_idx"])   # builder atom order differs from the source order (as in cross_score)
    ref_d = state["ge"].q1_distances(ref_X[:, first_map])[0]
    return dict(state=state, ref_d=ref_d, lookup=lookup, Traced=Traced, load_db=load_db, FittedDist=FittedDist,
                n_rows=int(max(n_rows)), native=native)


def _score(state, ref_d, lookup):
    import torch

    with torch.no_grad():
        sc = state["build"](lookup)
        scores, energies, broken = [], [], []
        for _, X, groups, energy in state["per_seed"]:
            s, _parts = CS._terms(sc, X, groups)
            d = sc._ge.q1_distances(X)
            broken.append((d > BROKEN_RATIO * ref_d).any(dim=1).numpy())
            scores.append(s.numpy())
            energies.append(energy)
    import numpy as np

    # Pools of different sizes are padded with rows that can never be selected (score -1e9, energy 1e9, not broken):
    # selection, floors and the severity rule are unaffected, and the padding is a fixed part of the scorer.
    width = max(len(x) for x in scores)
    pad = lambda a, v: np.concatenate([a, np.full(width - len(a), v, dtype=a.dtype)])
    return dict(scores=np.stack([pad(x, -1e9) for x in scores]), energies_kj=np.stack([pad(x, 1e9) for x in energies]),
                broken_coordination=np.stack([pad(x, False) for x in broken]))


def prepare_db_scorer(parent):
    """Return ``score_db(path) -> dict`` scoring the frozen seed pools under any constraints DB (selection per seed is argmax)."""
    ctx = _load(parent)

    def score_db(path):
        lookup = ctx["Traced"](ctx["load_db"](path))
        return _score(ctx["state"], ctx["ref_d"], lookup)

    return score_db, dict(code=parent["code"], runtime=ctx["native"], variant=ctx["state"]["variant"], pool_rows=ctx["n_rows"])


def prepare(parent, *, resolved_key, quantity="Q1", score_db=None):
    """Return (score_fn, clean, info). Heavy objects are created once. ``score_db``: stress base DB (default: the generating DB)."""
    ctx = _load(parent, score_db)
    state, ref_d, lookup, FittedDist = ctx["state"], ctx["ref_d"], ctx["lookup"], ctx["FittedDist"]
    target = tuple(resolved_key)

    def score_fn(fit):
        override = {}
        if fit is not None:
            for (q, req), (res, orig) in lookup.trace.items():
                if q == quantity and tuple(res) == target:
                    # a refit carries the new mixture; the ladder level stays that of the original entry
                    fd = FittedDist(**{"level": orig["level"], **fit})
                    override[(q, req)] = (res, fd) if quantity == "QO" else fd   # orientation lookups return (resolved key, dist)
        lookup.override.clear()
        lookup.override.update(override)
        try:
            return _score(state, ref_d, lookup)
        finally:
            lookup.override.clear()

    # one clean pass populates lookup.trace so trial calls know which requested keys resolve to the target
    clean = score_fn(None)
    hits = [req for (q, req), (res, _) in lookup.trace.items() if q == quantity and tuple(res) == target]
    info = dict(code=parent["code"], runtime=ctx["native"], variant=state["variant"], pool_rows=ctx["n_rows"],
                requested_keys_resolving_to_target=len(hits))
    if not hits:
        raise ValueError(f"no lookup in this case resolves to {target}: the key does not affect the pool")
    return score_fn, clean, info
