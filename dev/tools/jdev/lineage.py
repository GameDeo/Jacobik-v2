"""Beam-start lineage telemetry (coverage package Part C; diagnostic, off by default, no effect on results).

Enabled by the environment variable JACOBIK_LINEAGE_OUT=<dir>: `multiseed.assemble_batched` then calls `record()` once
per solver pass; nothing is written and nothing is computed when it is unset.  The record traces every start row
b of a seed (layout of `ComplexBuilder._beam_search_all_candidates`: b = (cand * M + m) * W + w, cand = donor-to-site
assignment, m = coordinate sample, w = beam seed; with a class budget `keep` the layout is subset and only `cand` is
known) to its outcome:

  nonfinite        score or coordinates not finite (cannot be kept)
  outside_window   solved, more than OUT_WINDOW nats below the best row of its seed (garbage / "obscene" proxy)
  merged           solved, inside the window, examined and dropped by the output dedup: a better-scored kept row has the
                   same structure (symmetry-aware heavy-atom RMSD <= DEDUP_RMSD); `merged_into` = that row (searched within
                   JACOBIK_LINEAGE_BUDGET_S seconds per pass, else null)
  kept             distinct kept row (`rank` by score)
  beyond_cap       never examined by the dedup: the output cap n_keep was reached with better-scored rows first (distinct or
                   duplicate; a time-boxed probe estimates the duplicate share)

The CUDA replay solver does not cull rows (every start runs all steps), so the GPU time per start is the solver wall
divided by the rows of the pass; `solver_wall_s` and `rows` are recorded for that.  Torsion-space redundancy before the
solve: for several wrapped-angle tolerances, the share of starts that have an earlier start (same seed) within the
tolerance in every torsion, and how often that pair ends in the same kept row.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

TOLS = (0.05, 0.1, 0.2, 0.4)      # rad, max-norm over the torsion columns


def enabled() -> bool:
    return bool(os.environ.get("JACOBIK_LINEAGE_OUT"))


def _wrapped_maxdist(T: torch.Tensor) -> torch.Tensor:
    """(B, J) torsions -> (B, B) max over columns of the wrapped absolute difference (rad)."""
    d = (T[:, None, :] - T[None, :, :] + np.pi) % (2 * np.pi) - np.pi
    return d.abs().amax(-1)


def _prep(deduper, X):
    """Centred heavy-atom coordinates and radial profiles of all rows of one seed, computed once per pass."""
    Xh = X[:, deduper.heavy].to(deduper.device, deduper.dtype)
    Xh = Xh - Xh.mean(1, keepdim=True)
    return Xh, deduper._radial(torch.nan_to_num(Xh))


def _find_merge_target(deduper, prep, i, kept, th, deadline=None):
    """Best-scored kept row that i duplicates (same test as the output dedup), else None."""
    Xh, Rr = prep
    kt = torch.as_tensor(kept, device=Xh.device)
    cand = kt[((Rr[kt] - Rr[i]) ** 2).mean(1).sqrt() <= th]
    for k in cand.tolist():
        if deadline is not None and time.perf_counter() > deadline:
            return None
        try:
            if getattr(deduper, "is_capped", False):
                ok, _, _, _ = deduper.factored_sym_upper(Xh[k].cpu().numpy(), Xh[i].cpu().numpy(), th=th, max_dev_th=0.0)
                if ok:
                    return int(k)
            if deduper._full_any(Xh[i], Xh[k:k + 1], th):
                return int(k)
        except Exception:
            continue
    return None


def record(ctx, assignments, th_init, thetas, scores, conv, X_all, cand, B1, n_seeds, selected, deduper, n_keep,
           solver_wall_s, beam_width, keep=None, meta=None):
    """Write one JSON line per seed of the pass.  th_init: list (per ligand) of initial theta (n_seeds * B1, 6 + J...).
    selected: list per seed of global row indices kept by the output dedup (as returned by assemble_batched)."""
    out_dir = Path(os.environ["JACOBIK_LINEAGE_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    from jacobik.builder import engine as E
    t0 = time.perf_counter()
    _dbg = os.environ.get("JACOBIK_LINEAGE_DEBUG")
    if _dbg:
        print(f"[lineage] start rows={B1} seeds={n_seeds}", flush=True)
    M = int(max(ctx.n_coord_samples, 1))
    n_assign = len(assignments)
    W = int(beam_width)
    from jacobik.ik import dedup as D
    th = float(D.DEDUP_RMSD)
    score_np = scores.detach().cpu().numpy().astype(float)
    cand = np.asarray(cand)
    # torsion columns of every ligand, concatenated (the first 6 columns are the rigid pose)
    cols = [t[:, 6:] for t in th_init if t.shape[1] > 6]
    T = torch.cat(cols, dim=1).to(X_all.device).float() if cols else None
    lines = []
    for j in range(n_seeds):
        sl = slice(j * B1, (j + 1) * B1)
        sc = score_np[sl]
        finite = np.isfinite(sc) & torch.isfinite(X_all[sl]).all(-1).all(-1).cpu().numpy()
        ok = E.output_window(sc, float(getattr(ctx, "score_norm", 0.0) or 0.0))
        sel = [int(b - j * B1) for b in np.asarray(selected[j]).tolist()]
        sel_sorted = sorted(sel, key=lambda b: -sc[b])
        kept_rank = {b: r + 1 for r, b in enumerate(sel_sorted)}
        outcome, merged_into = [], {}
        # many automorphisms (QEZVEY: 20000) make one duplicate test take seconds: outcomes stay (they are positional),
        # merge targets and the beyond-cap probe are skipped
        cheap = deduper is not None and not getattr(deduper, 'is_capped', False) and int(deduper.inv_maps.shape[0]) <= 512 if hasattr(deduper, 'inv_maps') else deduper is not None
        prep = _prep(deduper, X_all[sl]) if cheap else None
        if _dbg:
            print(f"[lineage] seed {j}: prep done at {time.perf_counter() - t0:.1f}s cheap={cheap}", flush=True)
        fin_sc = np.where(finite, sc, -np.inf)
        order_all = [int(x) for x in np.argsort(-fin_sc, kind="stable")]       # the dedup's best-first order
        pos = {b: i for i, b in enumerate(order_all)}
        capped = n_keep is not None and len(sel) >= n_keep
        last_pos = max((pos[b] for b in sel), default=-1)
        budget_s = float(os.environ.get("JACOBIK_LINEAGE_BUDGET_S", "10"))
        t_search = time.perf_counter()
        deadline = t_search + budget_s
        n_dup_probe = n_dup_hit = 0
        for b in range(B1):
            if b in kept_rank:
                outcome.append("kept")
            elif not finite[b]:
                outcome.append("nonfinite")
            elif not ok[b]:
                outcome.append("outside_window")
            elif capped and pos[b] > last_pos:
                # never examined by the dedup: the output cap (n_keep) was reached first.  A time-boxed probe tells how many of
                # these would have been duplicates of a kept row anyway.
                outcome.append("beyond_cap")
                if cheap and sel_sorted and n_dup_probe < 48 and time.perf_counter() - t_search < budget_s:
                    n_dup_probe += 1
                    n_dup_hit += _find_merge_target(deduper, prep, b, sel_sorted, th, deadline) is not None
            else:
                outcome.append("merged")                  # examined by the dedup and dropped: a duplicate of a better kept row
                if cheap and time.perf_counter() - t_search < budget_s:
                    tgt = _find_merge_target(deduper, prep, b, sel_sorted, th, deadline)
                    if tgt is not None:
                        merged_into[b] = tgt
        # start metadata
        full_idx = np.asarray(keep, int) if keep is not None else np.arange(B1)
        full = len(full_idx) == B1 and (full_idx.max(initial=-1) < n_assign * M * W)
        starts = []
        for b in range(B1):
            if full:
                f = int(full_idx[b])
                c, m, w = f // (M * W), (f // W) % M, f % W
            else:
                c, m, w = int(cand[j * B1 + b]), None, None
            starts.append((int(c), None if m is None else int(m), None if w is None else int(w)))
        # pre-solve torsion-space redundancy: share of starts with an earlier start within tol, ending in the same row
        redundancy = {}
        if T is not None and T.shape[1] > 0:
            Tj = T[sl]
            final = np.array([merged_into.get(b, b) if outcome[b] in ("kept", "merged") else -1 - b for b in range(B1)])   # unresolved = own row
            fin_t = torch.as_tensor(final, device=Tj.device)
            has = {tol: torch.zeros(B1, dtype=torch.bool, device=Tj.device) for tol in TOLS}
            same_ = {tol: torch.zeros(B1, dtype=torch.bool, device=Tj.device) for tol in TOLS}
            idx = torch.arange(B1, device=Tj.device)
            for i0 in range(0, B1, 128):                              # row blocks: never the full (B, B, J) tensor
                blk = slice(i0, min(i0 + 128, B1))
                d = (Tj[blk, None, :] - Tj[None, :, :] + np.pi) % (2 * np.pi) - np.pi
                dm = d.abs().amax(-1)                                   # (nb, B)
                earlier = idx[None, :] < idx[blk, None]
                same = (fin_t[blk, None] == fin_t[None, :]) & (fin_t[blk, None] >= 0)
                for tol in TOLS:
                    near = (dm <= tol) & earlier
                    has[tol][blk] = near.any(1)
                    same_[tol][blk] = (near & same).any(1)
            for tol in TOLS:
                redundancy[str(tol)] = dict(share_with_earlier_twin=float(has[tol].float().mean()),
                                            share_twin_same_final_row=float(same_[tol].float().mean()),
                                            share_of_twins_ending_same=float(same_[tol].sum() / max(int(has[tol].sum()), 1)))
        if _dbg:
            print(f"[lineage] seed {j}: outcomes+redundancy done at {time.perf_counter() - t0:.1f}s (search {time.perf_counter() - t_search:.1f}s)", flush=True)
        counts = {k: outcome.count(k) for k in set(outcome)}
        rec = dict(**(meta or {}), seed=(meta or {}).get('seeds', [None] * n_seeds)[j] if meta else None, pass_seed_index=j, rows=int(B1), n_assign=n_assign, M=M, W=W, full_layout=bool(full),
                   solver_wall_s=float(solver_wall_s), solver_s_per_row=float(solver_wall_s) / max(n_seeds * B1, 1),
                   n_seeds_in_pass=int(n_seeds), outcome_counts=counts, redundancy=redundancy, n_merged_with_target=len(merged_into), target_search_skipped=not cheap,
                   beyond_cap_dup_probe=dict(probed=n_dup_probe, duplicates=int(n_dup_hit)), search_s=time.perf_counter() - t_search,
                   score_window_nats=float(getattr(E, "OUT_WINDOW", 0.0)),
                   rows_table=dict(cand=[m_[0] for m_ in starts], m=[m_[1] for m_ in starts], w=[m_[2] for m_ in starts],
                                   outcome=outcome, score=[float(x) for x in sc],
                                   rank=[kept_rank.get(b) for b in range(B1)],
                                   merged_into=[merged_into.get(b) for b in range(B1)],
                                   converged=[bool(x) for x in conv[sl].detach().cpu().numpy()]),
                   analysis_s=time.perf_counter() - t0)
        if os.environ.get("JACOBIK_LINEAGE_XYZ"):   # every start's solved coordinates (builder atom order), for class checks
            try:
                tag = f"{(meta or {}).get('code')}_{rec['seed']}_{time.time_ns()}"
                np.savez_compressed(out_dir / f"xyz_{tag}.npz", X=X_all[sl].detach().cpu().numpy().astype(np.float32),
                                    score=np.asarray(sc, np.float32), outcome=np.asarray(outcome),
                                    cand=np.asarray([m_[0] for m_ in starts]))
                rec["xyz_file"] = f"xyz_{tag}.npz"
            except Exception as exc:
                rec["xyz_error"] = f"{type(exc).__name__}: {exc}"
        lines.append(rec)
    path = out_dir / f"lineage_{os.getpid()}.jsonl"
    with path.open("a") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    return lines
