"""Joint IK orchestration, deduplication, mirror completion, and conformer export."""
from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from jacobik.builder.result import ConformerResult
from jacobik.core import gpu

cp = gpu.cp


def _e():
    from jacobik.builder import engine

    return engine


MIRROR = np.array([-1.0, 1.0, 1.0])


def _mirror_score_fn(ctx):
    """The function the solve ranked its rows with, for rescoring reflected rows: X (n, N, 3) numpy, cand (n,) ->
    (n,) numpy; None when the scorer is not known. Final row score = solver score + (ranking - fit) (_run_joint_ik),
    and the solver's bend-prior and ring-closure terms are internal (reflection-invariant), so a reflected row's score
    is its source's plus f(reflection) - f(source) of this function."""
    sc = getattr(ctx, "_mirror_scorer", None)
    if sc is None:
        return None
    from jacobik.ik import scorer as _scm
    from jacobik.ik.ik_solver import _RANK_FLOOR as _rf

    def f(X, cand):
        Xt = gpu.as_gpu(X, cp.float32)
        rg = gpu.as_gpu(cand, cp.int64)
        if ((_scm.RANK_AGG == "sum" and _scm.SCORE_AGG != "sum") or _rf or _scm.Q1_TAIL_SIGMA > 0) and hasattr(sc, "ranking_score"):
            return gpu.to_cpu(sc.ranking_score(Xt, row_groups=rg)).astype(np.float64)
        return gpu.to_cpu(sc.score(Xt, row_groups=rg)[0]).astype(np.float64)
    return f


def mirror_completion(builder, mol, X_sel, score_sel, device, score_fn=None, cand=None):
    """P2R flag out_mirror: (rows, scores): the kept rows whose reflection is added to the output, and each added
    reflection's score (empty when off).

    The reflection of a row is its enantiomer at exactly the same energy, so it keeps its source's score and ranks
    right after it. The score is not mirror-invariant (|f(reflection) - f(source)| up to 0.055 measured, CIFZEZ):
    rescoring by `score_fn` (_mirror_score_fn) let that error decide which enantiomer is top-1 (P2R full panel: all 7
    top-1 losses were a reflection 0.0006-0.024 above its equal-energy source), so the callers do not pass it; it
    stays for diagnostics (score track: a mirror-invariant score). Only
    without tetrahedral tags (a tagged centre must keep its configuration; E/Z survives a reflection). A reflection
    that superposes (proper rotations, symmetry-aware) onto a kept row (an achiral row, or that enantiomer already
    present) is dropped (StructureDeduper.mirror_dup). Never raises: on any failure no rows are added."""
    if not _e().OUT_MIRROR or X_sel is None or len(X_sel) == 0:
        return [], np.zeros(0)
    try:
        if _e()._has_tetrahedral_tags(builder):
            return [], np.zeros(0)
        X_sel = np.asarray(X_sel)
        score_sel = np.asarray(score_sel, float)
        n = len(X_sel)
        s_m = score_sel.copy()
        if score_fn is not None:
            try:
                cc = np.zeros(n, int) if cand is None else np.asarray(cand)
                s_m = score_sel + (score_fn(X_sel * _e().MIRROR, cc) - score_fn(X_sel, cc))
                s_m = np.where(np.isfinite(s_m), s_m, score_sel)
            except Exception as exc:
                _e().log.warning("output mirror rescoring failed (%s); reflections keep their source scores",
                            str(exc).splitlines()[0][:200])
                s_m = score_sel.copy()
        dev = _e().resolve_device(device)
        Xo = gpu.as_gpu(X_sel, cp.float32)
        dd = _e().deduper_for(mol, dev)
        dup = dd.mirror_dup(Xo)
        rows = [i for i in range(n) if not dup[i]]
        _e().log.info("Output mirror: %d reflected rows added to %d", len(rows), n)
        return rows, s_m[rows]
    except Exception as exc:  # the generator never fails: skip the completion
        _e().log.warning("output mirror completion failed (%s); skipped", str(exc).splitlines()[0][:200])
        return [], np.zeros(0)


def output_window(score_np: np.ndarray, z_norm: float) -> np.ndarray:
    """Rows eligible for output: finite, and within OUT_WINDOW summed nats (score x z_norm) of the best row."""
    ok = np.isfinite(score_np)
    if _e().OUT_WINDOW > 0 and z_norm > 0 and ok.any():
        ok &= score_np >= score_np[ok].max() - _e().OUT_WINDOW / z_norm
    return ok



class _AssembleMixin:
    """Joint IK execution and deduplicated ConformerResult assembly."""

    def _run_joint_ik(
        self,
        ctx,
        fks,
        scorer_joint,
        ge_joint,
        X_full,
        all_theta_inits,
        all_donor_pos_idxs,
        ideal_directions_full_t,
        symmetric_groups,
        q3_symmetric_groups,
        beam_width,
        n_ik_steps,
        candidate_idx_per_b,
    ):
        """
        Joint IK solve of all ligands and candidates at once. ``beam_width`` is
        the total beam width across candidates, so the batch size is
        ``beam_width * n_coord_samples``.
        """
        from jacobik.ik.ik_solver import IKSolverConfig, JointIKSolver

        if _e().OUT_MIRROR:
            ctx._mirror_scorer = scorer_joint     # P2R out_mirror: rescoring of reflected rows (read only)

        debug = ctx.debug
        B = beam_width * ctx.n_coord_samples
        _e().log.info("Running joint IK: %d ligands × B=%d × T=%d steps", len(fks), B, n_ik_steps)

        cfg = replace(ctx.ik_config or IKSolverConfig(), n_steps=n_ik_steps, lr=ctx.lr)
        ec = getattr(ctx, "early_cull", None)
        if ec is not None and cfg.cull_at:
            # Embed-aware dedup: extra early cull down to one candidate's rows per graph-level class (see solve()).
            cfg = replace(cfg, cull_at=(ec[0],) + tuple(cfg.cull_at), cull_keep=(ec[1],) + tuple(cfg.cull_keep))
        if cfg.shape_tol is None:
            from jacobik.ik.targets import shape_tolerance

            cfg = replace(cfg, shape_tol=shape_tolerance(self.geometry))
        joint_solver = JointIKSolver(
            fks=fks,
            scorer=scorer_joint,
            ge=ge_joint,
            config=cfg,
            device=ctx.device,
        )
        joint_solver.bend_prior = ctx.bend_prior  # energy-calibrated angle-bend prior (None = off)
        thetas_final, converged, scores, solve_trace = joint_solver.solve(
            theta_inits=all_theta_inits,
            X_ref=X_full,
            donor_pos_idxs=all_donor_pos_idxs,
            ideal_directions=ideal_directions_full_t,
            debug=debug,
            symmetric_groups=symmetric_groups,
            q3_symmetric_groups=q3_symmetric_groups,
            row_groups=cp.asarray(np.asarray(candidate_idx_per_b), dtype=cp.int64),
        )
        self._last_bend_prior = joint_solver.last_bend_prior  # S2b capture (None unless single-chunk)

        from jacobik.ik import scorer as _scm
        from jacobik.ik.ik_solver import _RANK_FLOOR as _rf

        if ((_scm.RANK_AGG == "sum" and _scm.SCORE_AGG != "sum") or _rf or _scm.Q1_TAIL_SIGMA > 0) and hasattr(scorer_joint, "ranking_score"):
            # Re-rank every row (single thread, after the solve): replace the fit-floor mean score inside the final
            # score by the ranking score; the bend prior and ring-closure penalty terms are kept as they are.
            Bn = scores.shape[0]
            X_all = X_full if X_full.ndim == 3 else cp.broadcast_to(X_full[None], (Bn, X_full.shape[0], 3))
            for li, fk_li in enumerate(fks):
                X_all = fk_li.apply(thetas_final[li], X_all)
            rg = cp.asarray(np.asarray(candidate_idx_per_b), dtype=cp.int64)
            scorer_joint.set_gmm_floor("fit" if _rf else _scm.GMM_FLOOR_MODE)  # the floor the solver ranked with
            fit, _ = scorer_joint.score(X_all, row_groups=rg)
            scorer_joint.set_gmm_floor(_scm.GMM_FLOOR_MODE)
            rank = scorer_joint.ranking_score(X_all, row_groups=rg)
            scores = scores + (rank - fit).astype(scores.dtype)

        off = getattr(ctx, "_row_state_E", None) if _e().RING_STATES else None
        if off is not None and len(off) == scores.shape[0]:
            # P2R ring_states: the row's ring-state energy as a population term, -dE / (RT Z) (score = summed
            # log-likelihood / Z), so the dedup and the ranking see one score
            from jacobik.core.env import RT_KJ
            z = float(scorer_joint._agg_norm()) if hasattr(scorer_joint, "_agg_norm") else 1.0
            scores = scores - cp.asarray(np.asarray(off) / (RT_KJ * z), dtype=scores.dtype)

        if debug:
            print(
                f"\n[DEBUG] Joint IK done: {int(converged.sum())}/{B} converged"
                f"  score min={float(scores.min()):.4f}  max={float(scores.max()):.4f}"
            )

        return thetas_final, converged, scores, solve_trace

    def _assemble_conformers(
        self,
        ctx,
        fks,
        X_full,
        thetas_final,
        scores,
        converged,
        beam_width,
        candidate_idx_per_b,
    ):
        """
        Assemble a full complex for each of the best ``n_conformers`` IK trials
        (default ``max(beam_width, n_coord_samples)``), ranked by score.
        """
        mol = ctx.mol
        from jacobik.ik import stereo
        scope = _e().STEREO_SCOPE
        st_key = (scope, tuple((l.smiles, tuple(map(tuple, l.donors))) for l in self.ligands))
        cached_st = getattr(self, "_cached_stereo_elements", None)
        if cached_st is not None and cached_st[0] == st_key:
            elements = cached_st[1]
        else:
            elements = stereo.perceive_complex_stereo(self, scope=scope)
            self._cached_stereo_elements = (st_key, elements)
        do_out_per_class = getattr(self, "out_per_class", None)
        if do_out_per_class is None:
            do_out_per_class = _e().OUT_PER_CLASS

        n_keep = (
            None if do_out_per_class else (ctx.n_conformers if ctx.n_conformers is not None else max(beam_width, ctx.n_coord_samples))
        )
        conformers: List[ConformerResult] = []

        score_np = gpu.to_cpu(scores)
        converged_np = gpu.to_cpu(converged)
        sorted_b = np.argsort(-score_np)
        in_window = _e().output_window(score_np, float(getattr(ctx, "score_norm", 0.0) or 0.0))
        n_out = int((~in_window).sum())
        if n_out:
            _e().log.info("Output window: %d of %d rows more than %.0f nats below the best", n_out, len(score_np), _e().OUT_WINDOW)
        selected = [b for b in sorted_b if in_window[b]]
        if n_keep is not None:
            selected = selected[:n_keep]
        X_all = None
        if _e().OUT_DEDUP:
            # distinct structures (symmetry-aware heavy-atom RMSD), best-scored representative of each
            B_all = scores.shape[0]
            X_all = cp.broadcast_to(X_full, (B_all, X_full.shape[1], 3)) if X_full.ndim == 3 and X_full.shape[0] == 1 else (
                X_full if X_full.ndim == 3 else cp.broadcast_to(X_full[None], (B_all, X_full.shape[0], 3)))
            for li, fk_li in enumerate(fks):
                X_all = fk_li.apply(thetas_final[li], X_all)
            selected = np.asarray(_e().select_distinct(mol, X_all, scores, n_keep, ctx.device, th=ctx.dedup_rmsd, max_dev_th=ctx.dedup_max_dev), dtype=int)
            # rows outside the window are worse than every row inside it, so the best-first dedup only reaches them
            # once the rows inside are exhausted: dropping them afterwards is the same as never offering them
            selected = selected[in_window[selected]]
            _e().log.info("Output dedup: %d distinct of %d rows kept", len(selected), B_all)
        X_sel = None
        if len(selected):
            sel_t = cp.asarray(np.asarray(selected), dtype=cp.int64)
            if X_all is None:
                Xs = X_full[sel_t] if X_full.ndim == 3 and X_full.shape[0] > 1 else (
                    cp.broadcast_to(X_full, (len(selected), X_full.shape[1], 3)) if X_full.ndim == 3
                    else cp.broadcast_to(X_full[None], (len(selected), X_full.shape[0], 3)))
                for li, fk_li in enumerate(fks):
                    Xs = fk_li.apply(thetas_final[li][sel_t], Xs)
            else:
                Xs = X_all[sel_t]
            X_sel = gpu.to_cpu(Xs)  # one device-to-host copy

        # P2R mirror completion (flag out_mirror): the reflection of a row is the enantiomer at the same energy. Only
        # without tetrahedral tags (a tagged centre must keep its configuration; E/Z survives a reflection). Reflected
        # rows that superpose onto a kept row (achiral, or the enantiomer already present) are dropped by the same dedup.
        mirror_rows, mirror_sc = _e().mirror_completion(
            self, mol, X_sel, score_np[selected], ctx.device) if X_sel is not None and len(selected) else ([], np.zeros(0))
        jobs = [(ci, b, False, float(score_np[b])) for ci, b in enumerate(selected)] + \
            [(ci, selected[ci], True, float(s)) for ci, s in zip(mirror_rows, mirror_sc)]

        for ci, b, is_mirror, row_score in jobs:
            mol_copy = _e()._shallow_copy_mol_for_conformer(mol)

            if X_sel is not None:
                X_np = X_sel[ci] * _e().MIRROR if is_mirror else X_sel[ci]
            else:
                X_assembled = X_full[b : b + 1] if X_full.ndim == 3 else X_full[None]
                for li, fk_li in enumerate(fks):
                    theta_b = thetas_final[li][b : b + 1]  # (1, 6+Ji)
                    X_assembled = fk_li.apply(theta_b, X_assembled)  # (1, N, 3)

                X_np = gpu.to_cpu(X_assembled[0])
            X_row = np.array(X_np, copy=True)
            for atom, pos_row in zip(mol_copy.atoms, X_row):
                atom.position = pos_row

            _e()._refresh_coord_geometry(mol_copy)

            geom_label = getattr(mol_copy, "coordination_geometry", getattr(self, "geometry", "ideal"))
            assign_idx = int(candidate_idx_per_b[b])
            c_temp = ConformerResult(
                complex=mol_copy,
                score=row_score,
                converged=bool(converged_np[b]),
                rank=ci + 1,
                trial_idx=int(b),
                assignment_idx=assign_idx,
            )
            labels = stereo.compute_conformer_stereo_labels(elements, X_np, c=c_temp) if elements else {}
            st_label = ",".join(f"{k}:{v}" for k, v in sorted(labels.items())) if labels else "none"
            c_temp.stereo_label = st_label
            c_temp.requested_stereo_label = "none"
            c_temp.stereo_variant_idx = None
            c_temp.class_label = f"{geom_label}|S:{st_label}|A:{assign_idx}"
            c_temp.mirror_of = int(b) if is_mirror else None
            _lab = getattr(ctx, "_row_state_lab", None)
            c_temp.ring_state = str(_lab[b]).rstrip(";") if _lab is not None else None
            conformers.append(c_temp)

        conformers.sort(key=lambda c: (getattr(c.score, 'total', c.score), getattr(c, 'mirror_of', None) is None), reverse=True)
        if do_out_per_class:
            conformers = self._apply_out_per_class(conformers, z_norm=float(getattr(ctx, "score_norm", 1.0) or 1.0))
        else:
            for i, c in enumerate(conformers):
                c.rank = i + 1

        _e().log.info(
            "Engine done: %d conformers, %d converged, best score=%.4f",
            len(conformers),
            sum(1 for c in conformers if c.converged),
            conformers[0].score if conformers else float("nan"),
        )
        return conformers
