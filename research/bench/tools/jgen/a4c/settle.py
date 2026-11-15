"""Own-basin probe: how far is a pool row from the minimum it relaxes into, and which score terms disagree with it?

    python -m tools.run diagnostic --manifest M.json --task tools.jgen.a4c.settle:probe --out OUT --timeout-s 600
    M.json rows: {"code": ..., "seeds": [11, 12, 13], "run": tools.run output dir (ARM/{records,xyz}),
                  "arm": "control", "relax": A4b-layout dir whose relax/CODE_SEED.json holds the rows' relaxed minima
                  (a4c.relax_level, level normal)}
One parent per call: the scorer is built once (the score does not depend on the seed) and serves every seed's pool.

The product is the unrelaxed row, and it should relax into its basin within a few steps. For every relaxed row of the
unit (unrelaxed geometry X0 from the pool, own minimum Xn from the normal-level relaxation; a reflected row repeats its
source and is skipped) this reports, with no fit and no extra energy call:

  * settle cost: relaxation cycles (crude + continuation), MACE energy released E(X0) - E(Xn), heavy-atom RMSD X0 -> Xn;
  * the production ranking score per term (nats = score x Z) at X0 and at Xn, both evaluated on the Cartesian geometry
    with the row's own site assignment. A term that is much LOWER at the minimum than at the solver's row is a term whose
    optimum is displaced from the energy minimum for this complex (the solver sits on the term's optimum instead).
    The bend prior and the ring seam are functions of the solver's angles, not of the geometry, and are not included:
    `score_x0_residual` = pool score - sum of terms at X0 is their value at the solver's row;
  * internal coordinates at X0 and Xn: metal-donor distances and donor-metal-donor angles (signed X0 - Xn), the
    coordinates the stiff score terms Q1 and Q2 set.

An ascent started from a geometric fit of the solver's coordinates to Xn was tried first and dropped: a 0.04 A all-atom
fit error already costs tens of kJ/mol and nats, so fit points say nothing about either surface.
Stereo-extra parents (several classes) are not probed: `multiseed.prepare` refuses class loops. Never raises.
"""
from __future__ import annotations

import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np


def _kabsch_rmsd(A, B):
    a, b = A - A.mean(0), B - B.mean(0)
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    S[-1] *= np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.sqrt(max((a ** 2).sum() + (b ** 2).sum() - 2 * S.sum(), 0.0) / len(a)))


def probe(row: dict) -> dict:
    code = row["code"]
    seeds = [int(x) for x in row.get("seeds", [row.get("seed", 11)])]
    run, arm = Path(row["run"]), row.get("arm", "control")
    out = dict(code=code, seeds=seeds, error=None, rows=[], unit_errors={})
    t0 = time.perf_counter()
    try:
        from tools.jgen.bench.run import _setup_env, _prefetch_parent
        _setup_env({})
        import torch
        torch.set_num_threads(1)
        from jacobik.ik import ik_solver as IKM
        from tools.jgen import inputs
        from tools.jdev import multiseed
        from tools.jdev.budget.common import stats_from_db
        from tools.jdev.budget.pools import rank_terms
        from tools.jgen.bench.pool_export import _read_frames
        from tools.runkit import geom
        shard, ds = inputs.frame_shards()[code]
        mol, builder, err = _prefetch_parent(code, shard, ds)
        if builder is None:
            raise ValueError(f"unmappable: {err}")
        stats = stats_from_db(inputs.DB_PATH)
        cfg = IKM.IKSolverConfig(cull_at=(), cull_keep=())
        ctx, assignments, W, T = multiseed.prepare(builder, seeds[0], device="cpu", angle_bends=True, stats=stats, ik_config=cfg)
        (fks, dpi, ge, scorer), _ = multiseed.seed_rows(builder, ctx, assignments, W, [seeds[0]])
        z = float(scorer._agg_norm()) if hasattr(scorer, "_agg_norm") else 1.0
        metal = 0                                                     # the builder puts the metal first

        def terms(X, c):
            with torch.no_grad():
                _, bd = rank_terms(scorer, torch.as_tensor(X[None], dtype=torch.float32),
                                   torch.as_tensor([int(c)], dtype=torch.long))
            return {k: float(v[0]) * z for k, v in bd.items()}

        def ang(X, i, j):
            a, b = X[i] - X[metal], X[j] - X[metal]
            return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

        for seed in seeds:
            try:
                mp = run / arm / "xyz" / code / f"{code}_{arm}_s{seed}_pool.json"
                m = json.loads(mp.read_text())
                g2r = np.asarray(m["gen_to_ref_idx"], int)
                syms_ref = list(m["ref_symbols"])
                frames = [np.asarray(fr[2], float) for f in dict.fromkeys(r["file"] for r in m["rows"])
                          for fr in _read_frames(mp.parent / f)]
                rel = json.loads((Path(row["relax"]) / "relax" / f"{code}_{seed}.json").read_text())["rows"]
                top = next(i for i, r in enumerate(m["rows"]) if r["is_top1"])
                todo = [int(k) for k, r in rel.items() if r.get("X") is not None and r.get("level")
                        and r.get("E_mace_kj") is not None and m["rows"][int(k)].get("mirror_of") is None]
                heavy_b = np.zeros(len(g2r), bool)
                heavy_b[g2r] = np.array([s != "H" for s in syms_ref])
                syms_b = [None] * len(g2r)
                for i_ref, i_b in enumerate(g2r):
                    syms_b[i_b] = syms_ref[i_ref]
                rc = np.array([geom.rcov(s) for s in syms_b])

                def donors(X):
                    d = np.linalg.norm(X - X[metal], axis=1)
                    return {int(i) for i in np.flatnonzero(heavy_b & (d < rc[metal] + rc + 0.4)) if i != metal}
            except Exception as exc:
                out["unit_errors"][str(seed)] = f"{type(exc).__name__}: {exc}"
                continue
            for i in sorted(todo, key=lambda i: (i != top, i)):
                rec = dict(seed=seed, row=i, is_top1=i == top, error=None)
                try:
                    r = rel[str(i)]
                    c_row = int(m["rows"][i]["row_key"][1])
                    X0 = frames[i]
                    Xn = np.zeros_like(X0)
                    Xn[g2r] = np.asarray(r["X"], float)
                    t0_, tn_ = terms(X0, c_row), terms(Xn, c_row)
                    D = sorted(donors(X0) & donors(Xn))
                    md = [(syms_b[d], float(np.linalg.norm(X0[d] - X0[metal])), float(np.linalg.norm(Xn[d] - Xn[metal]))) for d in D]
                    dmd = [(ang(X0, a, b), ang(Xn, a, b)) for a, b in itertools.combinations(D, 2)]
                    crude = r.get("crude") or {}
                    e0 = m["rows"][i].get("mace_energy_kj")
                    rec.update(
                        pool_score_nats=float(m["rows"][i]["score"]) * z, terms_x0=t0_, terms_xn=tn_,
                        terms_xn_minus_x0={k: tn_[k] - t0_[k] for k in t0_},
                        score_x0_residual=float(m["rows"][i]["score"]) * z - sum(t0_.values()),
                        E_released=None if e0 is None else float(e0) - float(r["E_mace_kj"]),
                        E_released_crude=None if (e0 is None or crude.get("E_mace_kj") is None) else float(e0) - float(crude["E_mace_kj"]),
                        cycles_crude=crude.get("cycles"), cycles_more=r.get("cycles"),
                        rmsd_heavy=_kabsch_rmsd(X0[heavy_b], Xn[heavy_b]), rmsd_all=_kabsch_rmsd(X0, Xn),
                        donor_set_same=donors(X0) == donors(Xn), md=md,
                        md_delta=[a - b for _, a, b in md], dmd_delta=[a - b for a, b in dmd],
                        dmd_trans_delta=[a - b for a, b in dmd if b > 150], dmd_cis_delta=[a - b for a, b in dmd if b <= 150],
                        ring_state=m["rows"][i].get("ring_state"))
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"; rec["traceback"] = traceback.format_exc()[-1500:]
                out["rows"].append(rec)
        out.update(score_norm=z, n_atoms=len(ctx.mol.atoms), wall_s=time.perf_counter() - t0)
    except Exception as exc:  # never fail the batch
        out["error"] = f"{type(exc).__name__}: {exc}"; out["traceback"] = traceback.format_exc()[-3000:]
    return json.loads(json.dumps(out, default=float))
