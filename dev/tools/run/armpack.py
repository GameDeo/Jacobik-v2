"""Arm packing (runner option --pack-arms, off by default): every arm of a parent in ONE joint IK solve.

Why only arms (and seeds) of the same parent, not different parents
--------------------------------------------------------------------
Rows of one batch are independent along the batch axis: on the CPU a row's result does not depend on which other
rows share the batch (the bench already stacks the seeds of a parent into one solve on that basis; the seeds test
`test_seeds_in_one_pass_match_separate_solves` covers it). Rows of DIFFERENT parents have different atom counts,
ligand kinematic trees, DOF widths, donor counts and score tables; one tensor for them needs padding along those
axes, and every reduction over a padded axis (score term sums, logsumexp over mixture components or donors, the
shape-fit power mean, bend sums) changes its floating-point summation tree: zero (or -inf) padding changed the CPU
result of `sum` / `logsumexp` in 25-50 % of rows in a direct test (dev/tools/run/README "cross-row operations"). Exact
per-parent identity is therefore reachable only without padding, i.e. for rows that share every non-batch shape:
the arms and seeds of one parent whose flags change start poses only (C-2b: C, K, P; same width W).

How
---
Before the bench runs a parent, `presolve` prepares the parent once (its own builder, the bench's own code: prepare,
seed_rows per arm with that arm's flags, class-budget rows), stacks the rows of every packable arm (all seeds),
runs ONE `_run_joint_ik`, and caches each arm's slice under a digest of its exact solver inputs (start angles,
reference coordinates, ideal directions, candidate ids, step count). The bench then runs unchanged; its
`_run_joint_ik` call for an arm returns the cached slice when the digest matches (and solves normally when it does not:
a miss can never change a result, it only loses the gain). Packable = same solve-time settings as the first arm
(finish flags, bend release, DOF scales, bend free zone, step count) and same reference-coordinate layout.

Cross-row operations of the solve (segmenting): culling (CPU only, B >= 256: top-k over the whole batch) is NOT
segmented, so packing is exact only where culling is off: CUDA replay (never culls) and --cpu-identity (cull_at=());
on CPU without --cpu-identity the option refuses. Early stop (finish phase: stop when every row is done) is batch-wide:
a packed solve may run more steps than a solo one, but rows that are done take zero steps and keep their best
snapshot, so their outputs are unchanged (checked by the identity test, not assumed).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np


def _digest(th, X, ideal, cand, steps) -> str:
    h = hashlib.sha256()
    for t in list(th) + [X, ideal]:
        a = t.detach().contiguous().cpu().numpy()
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    h.update(np.ascontiguousarray(np.asarray(cand)).tobytes())
    h.update(str(int(steps)).encode())
    return h.hexdigest()


#: solve-time settings an arm must share with the first packed arm (bench: set per arm before its solve)
SOLVE_KEYS = ("bend_release", "dof_step_scale", "bend_free_deg", "finish_phase", "polish", "optimizer", "n_ik_steps")


class ArmPacker:
    def __init__(self, backend, orig_joint):
        self.be = backend
        self.orig_joint = orig_joint      # the engine's _run_joint_ik (unwrapped)
        self.cache: dict[str, tuple] = {}
        self.info: dict = {}

    # -- the bench-side hook ------------------------------------------------------------------------------------
    def lookup(self, builder, args):
        th, X, ideal, cand, steps = args[5], args[4], args[7], args[12], args[11]
        hit = self.cache.pop(_digest(th, X, ideal, cand, steps), None) if self.cache else None
        if hit is None:
            if self.info.get("packed_arms"):
                self.info["misses"] = self.info.get("misses", 0) + 1
            return None
        thetas, conv, scores, bend, last = hit
        builder._last_bend_prior = bend
        import jacobik.ik.solver_replay as SR
        SR.LAST.clear()
        SR.LAST.update(last)
        self.info["hits"] = self.info.get("hits", 0) + 1
        return [t.clone() for t in thetas], conv.clone(), scores.clone(), {}

    # -- before the bench runs the parent -----------------------------------------------------------------------
    def presolve(self, code: str) -> dict:
        self.cache.clear()
        self.info = dict(code=code, packed_arms=[], skipped={})
        t0 = time.monotonic()
        try:
            self._presolve(code)
        except Exception as exc:  # noqa: BLE001 - packing never fails a parent: the bench solves normally
            self.cache.clear()
            self.info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        self.info["presolve_wall_s"] = round(time.monotonic() - t0, 3)
        return self.info

    def _presolve(self, code: str):
        import torch
        from jacobik.builder import engine as E
        from jacobik.core import profile as P
        from jacobik.ik import ik_solver as IKM
        import jacobik.ik.solver_replay as SR
        from tools.jgen import inputs
        from tools.jdev.budget.common import stats_from_db
        be = self.be
        bench = be.bench
        prepare_fn, seed_rows_fn = be.orig_ms["prepare"], be.orig_ms["seed_rows"]
        multiseed = be.multiseed
        plan, seeds, device = be.plan, be.seeds, be.device
        if device == "cpu" and not plan.get("cpu_identity"):
            self.info["skipped"]["all"] = "CPU without --cpu-identity culls across the whole batch: not packable"
            return
        identity_kw = {}
        if plan.get("cpu_identity"):
            identity_kw["ik_config"] = IKM.IKSolverConfig(n_workers=1, cull_at=(), cull_keep=())
        spec = [dict(arm=a["arm"], flags=a.get("flags", {}), seed_budget=a.get("seed_budget", "three_w16"),
                     **{k: v for k, v in a.items() if k not in ("arm", "flags", "seed_budget", "out", "xyz_dir")})
                for a in plan["arms"]]
        shard, ds = inputs.frame_shards()[code]
        mol, builder, err = be.orig_prefetch(code, shard, ds)
        if builder is None:
            self.info["skipped"]["all"] = f"unmappable: {err}"
            return
        from jacobik.builder import embed as _ce
        if getattr(_ce, "EZ_ENUMERATE", False) and hasattr(builder, "_ez_combinations") and len(builder._ez_combinations()) > 1:
            self.info["skipped"]["all"] = "E/Z enumeration (per-variant solves)"
            return
        stats = stats_from_db(inputs.DB_PATH)
        bench._set_finish_flags(spec[0]["flags"])
        try:
            with multiseed.safe_cuda_eigh_chunking():
                ctx, assignments, W, T = prepare_fn(builder, seeds[0], device=device, angle_bends=True, stats=stats,
                                                           n_conformers=spec[0].get("n_conformers"), **identity_kw)
                (fks, dpi, ge, scorer), rows = seed_rows_fn(builder, ctx, assignments, W, seeds)
                ctx.score_norm = float(scorer._agg_norm()) if hasattr(scorer, "_agg_norm") else 0.0
                W_total = W * len(assignments)
                keep = None
                ctx.early_cull = None
                if E.CLASS_BUDGET == "split":
                    keep = builder._class_budget_rows(ctx, assignments, W)
                elif E.CLASS_BUDGET == "early_cull":
                    n_cls = builder._n_topological_classes(ctx, assignments)
                    if n_cls is not None and n_cls < len(assignments):
                        ctx.early_cull = (E.EARLY_CULL_AT, n_cls / len(assignments))
                if keep is not None:
                    kt = torch.as_tensor(keep, dtype=torch.long, device=ctx.device)
                    active = [([x[kt] for x in th], X[kt] if X.dim() == 3 else X, ideal[kt], cand[keep])
                              for th, X, ideal, cand in rows]
                    W_total = len(keep) // max(ctx.n_coord_samples, 1)
                else:
                    active = rows
                sym = ctx.symmetric_donor_groups
                q3_sym = [g for g in sym if len({ctx.all_donor_groups[d][0] for d in g}) == 1]
                ref = None
                parts = []  # (arm, th list, X, ideal, cand, W_total_arm)
                for arm in spec:
                    flags = arm["flags"]
                    name = arm["arm"]
                    if arm.get("seed_budget", "three_w16") == "single_w48":
                        self.info["skipped"][name] = "single_w48 (own prepare per seed, other width)"
                        continue
                    if flags.get("ring_states"):
                        self.info["skipped"][name] = "own ligand embeds (ring pucker)"
                        continue
                    steps = int(arm.get("n_ik_steps", T))
                    settings = {k: flags.get(k, P.get(k) if k in P.FIELDS else None) for k in SOLVE_KEYS if k != "n_ik_steps"}
                    settings["n_ik_steps"] = steps
                    if ref is None:
                        ref = settings
                    elif settings != ref:
                        self.info["skipped"][name] = "solve-time settings differ from the first packed arm"
                        continue
                    bench._set_finish_flags(flags)
                    if flags.get("start_kabsch", False):  # own beam seeds
                        (_f, _d, _g, _s), arm_rows = seed_rows_fn(builder, ctx, assignments, W, seeds)
                        arm_keep = builder._class_budget_rows(ctx, assignments, W) if E.CLASS_BUDGET == "split" else None
                        if arm_keep is not None:
                            kt = torch.as_tensor(arm_keep, dtype=torch.long, device=ctx.device)
                            arm_active = [([x[kt] for x in th], X[kt] if X.dim() == 3 else X, ideal[kt], cand[arm_keep])
                                          for th, X, ideal, cand in arm_rows]
                            wt = len(arm_keep) // ctx.n_coord_samples
                        else:
                            arm_active, wt = arm_rows, W * len(assignments)
                    else:
                        arm_active, wt = active, W_total
                    # exactly the bench's per-arm stacking of all seeds (one group: no OOM retry split)
                    th = [torch.cat([arm_active[i][0][li].clone() for i in range(len(seeds))], dim=0) for li in range(len(fks))]
                    X_in = (torch.cat([arm_active[i][1].clone() for i in range(len(seeds))], dim=0)
                            if arm_active[0][1].dim() == 3 else arm_active[0][1].clone())
                    ideal = torch.cat([arm_active[i][2].clone() for i in range(len(seeds))], dim=0)
                    cand = np.concatenate([arm_active[i][3] for i in range(len(seeds))])
                    parts.append((name, th, X_in, ideal, cand, wt * len(seeds), steps))
                bench._set_finish_flags(spec[0]["flags"])
                dims = {p[2].dim() for p in parts}
                if len(dims) > 1 or (dims == {2} and not all(torch.equal(p[2], parts[0][2]) for p in parts)):
                    for p in parts[1:]:
                        self.info["skipped"][p[0]] = "reference-coordinate layout differs"
                    parts = parts[:1]
                if len(parts) < 2:
                    self.info["skipped"].setdefault("pack", "fewer than two packable arms")
                    return
                # the first packed arm's solve-time module settings, as the bench sets them before its solve
                f0 = spec[[a["arm"] for a in spec].index(parts[0][0])]["flags"]
                import jacobik.ik.angle_bends as AB
                for mod in (IKM, SR):
                    mod.BEND_RELEASE = f0.get("bend_release", P.get("bend_release"))
                    mod.DOF_STEP_SCALE = f0.get("dof_step_scale", P.get("dof_step_scale"))
                bf = float(f0.get("bend_free_deg", P.get("bend_free_deg")))
                AB.FREE_DEG = bf
                if ctx.bend_prior is not None:
                    ctx.bend_prior["free_deg"] = bf
                for fk in fks:
                    if getattr(fk, "_bend_plan", None) is not None and hasattr(fk, "_bend_specs"):
                        fk._bend_free = torch.tensor(AB.free_zone(fk._bend_specs, free_deg=bf), dtype=torch.float32,
                                                     device=ctx.device)
                th_all = [torch.cat([p[1][li] for p in parts], dim=0) for li in range(len(fks))]
                X_all = torch.cat([p[2] for p in parts], dim=0) if parts[0][2].dim() == 3 else parts[0][2]
                ideal_all = torch.cat([p[3] for p in parts], dim=0)
                cand_all = np.concatenate([p[4] for p in parts])
                digests = [_digest(p[1], p[2], p[3], p[4], p[6]) for p in parts]
                t0 = time.monotonic()
                thetas, conv, scores, _trace = self.orig_joint(builder, ctx, fks, scorer, ge, X_all, th_all, dpi, ideal_all,
                                                               sym, q3_sym, sum(p[5] for p in parts), parts[0][6], cand_all)
                if ctx.device.type == "cuda":
                    torch.cuda.synchronize()
                solve_s = time.monotonic() - t0
                bend = getattr(builder, "_last_bend_prior", None)
                last = {k: SR.LAST.get(k) for k in ("mode", "replayed", "T", "B", "steps_executed", "error")}
                last["packed_arms"] = [p[0] for p in parts]
                o = 0
                for p, dg in zip(parts, digests):
                    n = p[1][0].shape[0]
                    sl = slice(o, o + n)
                    self.cache[dg] = ([t[sl] for t in thetas], conv[sl], scores[sl],
                                      None if bend is None else bend[sl], dict(last))
                    o += n
                self.info.update(packed_arms=[p[0] for p in parts], packed_rows=o, packed_solve_s=round(solve_s, 3),
                                 replay=last)
        finally:
            bench._set_finish_flags(spec[0]["flags"])
