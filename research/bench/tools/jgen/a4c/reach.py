"""P2R reachability probe (fit, then ascend): does the production score hold the reference basin? One parent per call.

    python -m tools.run diagnostic --manifest M.json --task tools.jgen.a4c.reach:probe --out OUT --timeout-s 900
    M.json rows: {"code": ..., "run": tools.run output dir with control/{records,xyz}, "seed": 11, "k": 16}

1. Production prepare and starting rows for the seed (multiseed.prepare / seed_rows, production profile and DB).
2. The tmQM reference (builder atom order) is placed in each start's frame (Kabsch on metal + donors) and every start's
   solver DOFs are fitted to it (Adam on the heavy-atom squared distance through the kinematics; bonds, angles and rings
   are the embed's, so the fit has a floor). The k best-fitting starts are kept.
3. Those starts are solved with the production solver, culling off.
Reported: fit heavy RMSD, core RMSD to the reference of the fitted start and of the solved row (a4c.classes), class kept,
solved scores against the pool's top-1 score and best same-class score of the same seed (record / manifest), the
per-term ranking score of every solved row and of the pool's top-1 row, and the solved coordinates (reference order).
The unrelaxed core RMSD of a solved row does NOT say whether it relaxes into the reference basin: panel hits come from
rows 0.3-0.6 A away before relaxation. `a4c.reach_relax` relaxes the solved rows and applies the panel's hit rule; read
the probe from that, not from the drift.
Never raises: errors are returned in the result.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np


def _kabsch(P, Q):
    """R, t with R @ P_i + t ~ Q_i (no reflection)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def probe(row: dict) -> dict:
    code, run, seed, k = row["code"], Path(row["run"]), int(row.get("seed", 11)), int(row.get("k", 16))
    out = dict(code=code, seed=seed, error=None)
    t0 = time.perf_counter()
    try:
        from tools.jgen.bench.run import _setup_env, _prefetch_parent
        _setup_env({})
        import torch
        torch.set_num_threads(1)
        from jacobik.builder import engine as E
        from jacobik.ik import ik_solver as IKM
        from tools.jgen import inputs
        from tools.jdev import multiseed
        from tools.jdev.budget.common import stats_from_db
        from tools.jgen.a4c.classes import Reference
        device = row.get("device", "cpu")
        rec = json.loads((run / "control" / "records" / f"{code}_{seed}.json").read_text())
        mp = run / "control" / "xyz" / code / f"{code}_control_s{seed}_pool.json"
        m = json.loads(mp.read_text())
        g2r = np.asarray(m["gen_to_ref_idx"], int)
        ref_X = np.asarray(m["ref_X"], float)
        ref = Reference(m["ref_symbols"], ref_X)
        Xb = np.zeros_like(ref_X); Xb[g2r] = ref_X
        heavy = np.array([s != "H" for s in m["ref_symbols"]])
        hb = np.zeros(len(g2r), bool); hb[g2r] = heavy                      # heavy mask in builder order
        shard, ds = inputs.frame_shards()[code]
        mol, builder, err = _prefetch_parent(code, shard, ds)
        if builder is None:
            raise ValueError(f"unmappable: {err}")
        stats = stats_from_db(inputs.DB_PATH)
        cfg = IKM.IKSolverConfig(cull_at=(), cull_keep=())
        ctx, assignments, W, T = multiseed.prepare(builder, seed, device=device, angle_bends=True, stats=stats, ik_config=cfg)
        (fks, dpi, ge, scorer), rows = multiseed.seed_rows(builder, ctx, assignments, W, [seed])
        ctx.score_norm = float(scorer._agg_norm()) if hasattr(scorer, "_agg_norm") else 0.0
        th, X, ideal, cand = rows[0]
        B = th[0].shape[0]
        metal = 0                                   # the builder puts the metal first, at the origin

        def fk_all(thl, Xr):
            Xa = Xr if Xr.dim() == 3 else Xr.unsqueeze(0).expand(thl[0].shape[0], -1, -1)
            for li, f in enumerate(fks):
                Xa = f.apply(thl[li], Xa)
            return Xa

        with torch.no_grad():
            X0 = fk_all(th, X).cpu().numpy()
        donors = sorted({int(p) for f in fks for p in getattr(f, "_donor_pos_indices", [])})
        anc = sorted({metal} | set(donors))          # FK positions index the full complex array (fit RMSD checks it)
        targets = np.empty_like(X0)
        for b in range(B):
            R, t = _kabsch(Xb[anc], X0[b][anc])
            targets[b] = Xb @ R.T + t
        tgt = torch.as_tensor(targets, dtype=torch.float32, device=th[0].device)
        hmask = torch.as_tensor(hb, device=th[0].device)
        params = [t_.detach().clone().requires_grad_(True) for t_ in th]
        opt = torch.optim.Adam(params, lr=0.05)
        Xr = X.detach()
        for it in range(int(row.get("fit_steps", 400))):
            opt.zero_grad()
            Xa = fk_all(params, Xr)
            loss_b = ((Xa - tgt)[:, hmask] ** 2).sum(-1).mean(-1)
            loss_b.sum().backward()
            opt.step()
        with torch.no_grad():
            Xa = fk_all(params, Xr)
            fit = ((Xa - tgt)[:, hmask] ** 2).sum(-1).mean(-1).sqrt().cpu().numpy()
        ncs = max(int(ctx.n_coord_samples), 1)
        kk = max(ncs, (k // ncs) * ncs)
        sel = np.argsort(fit)[:kk]
        st = torch.as_tensor(sel, dtype=torch.long, device=th[0].device)
        th_sel = [p.detach()[st].clone() for p in params]
        X_sel = Xr[st] if Xr.dim() == 3 else Xr
        ideal_sel = ideal[st]
        cand_sel = np.asarray(cand)[sel]
        with torch.no_grad():
            Xfit = fk_all(th_sel, X_sel).cpu().numpy()
        sym = ctx.symmetric_donor_groups
        q3_sym = [g for g in sym if len({ctx.all_donor_groups[d][0] for d in g}) == 1]
        thetas, conv, scores, _ = builder._run_joint_ik(ctx, fks, scorer, ge, X_sel, th_sel, dpi, ideal_sel, sym, q3_sym,
                                                        kk // ncs, T, cand_sel)
        with torch.no_grad():
            Xs = fk_all(thetas, X_sel).cpu().numpy()
        sc = scores.detach().cpu().numpy()
        # per-term ranking score (summed nats) of the solved rows: which term pays for staying at the reference
        terms = {}
        try:
            from tools.jdev.budget.pools import rank_terms
            rg = torch.as_tensor(np.asarray(cand_sel), dtype=torch.long, device=th[0].device)
            _, bd = rank_terms(scorer, torch.as_tensor(Xs, dtype=torch.float32, device=th[0].device), rg)
            terms = {k_: v.detach().cpu().numpy() for k_, v in bd.items()}
        except Exception as exc:  # diagnostic only
            out["terms_error"] = f"{type(exc).__name__}: {exc}"
        res = []
        for j in range(kk):
            a = ref.compare(Xfit[j][g2r]); b = ref.compare(Xs[j][g2r])
            res.append(dict(row=int(sel[j]), cand=int(cand_sel[j]), fit_rmsd=float(fit[sel[j]]),
                            fit_core=a.get("core_rmsd_A"), fit_same=bool(a.get("same_class")),
                            solved_core=b.get("core_rmsd_A"), solved_same=bool(b.get("same_class")),
                            score=float(sc[j]), terms={k_: float(v[j]) for k_, v in terms.items()},
                            X_solved=np.round(Xs[j][g2r], 5).tolist()))          # reference atom order (a4c.reach_relax)
        pool_scores = [float(r["score"]) for r in m["rows"]]
        out.update(score_norm=ctx.score_norm, pool_min_score=min(pool_scores), pool_n=len(pool_scores),
                   ref_symbols=list(m["ref_symbols"]), ref_X=ref_X.tolist())
        out.update(n_starts=B, k=kk, pool_top1_score=float(rec.get("top1_score")), pool_max_score=max(pool_scores),
                   fit_rmsd_min=float(fit.min()), rows=res,
                   best_solved_same=min(([r["solved_core"], r["score"]] for r in res if r["solved_same"]), default=None),
                   wall_s=time.perf_counter() - t0)
        # pool rows of the reference class, best score (from the unrelaxed pool)
        from tools.jgen.bench.pool_export import _read_frames
        frames = [fr for f in dict.fromkeys(r["file"] for r in m["rows"]) for fr in _read_frames(mp.parent / f)]
        same = [(pool_scores[i], ref.compare(np.asarray(Xp)[g2r]).get("core_rmsd_A")) for i, (_, _, Xp) in enumerate(frames)
                if ref.compare(np.asarray(Xp)[g2r]).get("same_class")]
        out["pool_best_same_class"] = max(same) if same else None
        try:    # the pool top-1 row's terms, for the same decomposition
            top = next(i for i, r in enumerate(m["rows"]) if r["is_top1"])
            ct = int(m["rows"][top]["row_key"][1])
            _, bd = rank_terms(scorer, torch.as_tensor(np.asarray(frames[top][2])[None], dtype=torch.float32, device=th[0].device),
                               torch.as_tensor([ct], dtype=torch.long, device=th[0].device))
            out["pool_top1_terms"] = {k_: float(v[0]) for k_, v in bd.items()}
        except Exception as exc:
            out["pool_terms_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # never fail the batch
        out["error"] = f"{type(exc).__name__}: {exc}"; out["traceback"] = traceback.format_exc()[-3000:]
    return json.loads(json.dumps(out, default=float))
