"""Several seeds of ONE complex in one solver pass (2026-10-03). Development tooling for the benchmark runner
(dev/tests/regression/run_panel.py, `--seeds-in-one-pass`); not part of the jacobik API.

    results, info = solve_seeds(builder, seeds=[11, 12, 13], stats=st, device="cuda", angle_bends=True, n_conformers=8)
    results[12].conformers        # what builder.solve(seed=12, ...) returns

What a seed changes in a solve: the M-D target lengths drawn for each donor, the random start of the bend joints, the
macrocycle fold given to each row and the random part of the base-pose fit. All of it happens in
`ComplexBuilder._beam_search_all_candidates`. Everything else (embed, shell, assignment enumeration, kinematics,
scorer, beam search) is the same for every seed. So:

  1. solve() runs once up to the per-assignment work (embed, shell, enumeration);
  2. `_beam_search_all_candidates` runs once per seed with that seed's generators; the seed-independent objects it
     builds (kinematics, geometry extractor, scorer, beam result, ligand decomposition) are built on the first pass
     and handed back on the later ones, so a later pass costs only the draws;
  3. the rows of all seeds are stacked and solved in one `_run_joint_ik` call (groups of seeds if `max_rows` is set);
  4. output dedup and assembly run per seed on that seed's rows (`batched_assembly=True`: coordinates of all rows
     from one kinematics pass, see `assemble_batched`).

Each seed's starting rows are the ones its own solve() would use. Rows do not interact in the no-culling GPU solver,
so each seed's result is its own solve's result up to float noise. With row culling (the CPU solver) the cull would
rank rows of all seeds together, so this is for the replay path only; `solve_seeds` refuses otherwise.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

#: threads for the per-seed output dedups in assemble_batched (independent seeds; identical selections at any value)
DEDUP_SEED_THREADS = 3
#: where assemble_batched dedups the rows of a CUDA solve: "solver" (that GPU, CPU fallback) or "cpu"
BENCH_DEDUP_DEVICE = "solver"
#: ... from this many automorphisms on (below it the many small GPU operations lose to the CPU: 128 maps 1.2 s CPU vs
#: 3.9 s GPU, 192 maps 1.4 vs 2.7 s; 1536 maps 8.1 vs 2.0 s; 20000 maps 36 vs 3.6 s per seed, RTX 2060 SUPER)
BENCH_DEDUP_GPU_MIN_MAPS = 1000

log = logging.getLogger(__name__)


@contextmanager
def safe_cuda_eigh_chunking(chunk_size: int = 4096):
    """Scope-limited monkeypatch of torch.linalg.eigh and eigvalsh to chunk large CUDA batches."""
    orig_eigvalsh = torch.linalg.eigvalsh
    orig_eigh = torch.linalg.eigh

    def _safe_eigvalsh(A, *args, **kwargs):
        if A.is_cuda and A.dim() >= 3 and A.shape[0] > chunk_size:
            return torch.cat([orig_eigvalsh(c, *args, **kwargs) for c in A.split(chunk_size)], dim=0)
        return orig_eigvalsh(A, *args, **kwargs)

    def _safe_eigh(A, *args, **kwargs):
        if A.is_cuda and A.dim() >= 3 and A.shape[0] > chunk_size:
            parts = [orig_eigh(c, *args, **kwargs) for c in A.split(chunk_size)]
            return torch.cat([p[0] for p in parts], dim=0), torch.cat([p[1] for p in parts], dim=0)
        return orig_eigh(A, *args, **kwargs)

    torch.linalg.eigvalsh = _safe_eigvalsh
    torch.linalg.eigh = _safe_eigh
    try:
        yield
    finally:
        torch.linalg.eigvalsh = orig_eigvalsh
        torch.linalg.eigh = orig_eigh


class _Prepared(Exception):
    pass


def prepare(builder, seed, _solve=None, **solve_kw):
    """solve() up to `_solve_for_assignments`: returns (ctx, assignments, beam_width, n_ik_steps).

    Exactly one solve may reach `_solve_for_assignments`. A solve() that loops over classes and catches exceptions per
    class (engine._solve_stereo_variants) swallowed the stop signal and every class overwrote the capture: the bench
    solved only the LAST stereo class of every stereo-extra parent (2026-11-13). Callers handle classes themselves
    (solve_seeds: E/Z combinations, _solve_stereo_classes); a second capture raises."""
    got = {}

    def capture(ctx, assignments, beam_width, n_ik_steps):
        if "v" in got:
            got["n"] = got.get("n", 1) + 1
            raise _Prepared
        got["v"] = (ctx, assignments, beam_width, n_ik_steps)
        raise _Prepared

    builder._solve_for_assignments = capture          # instance attribute shadows the method for this call
    try:
        (_solve or type(builder).solve)(builder, seed=seed, **solve_kw)
    except _Prepared:
        pass
    finally:
        del builder._solve_for_assignments
    if got.get("n", 1) > 1:
        raise RuntimeError(f"multiseed.prepare: solve() reached _solve_for_assignments {got['n']} times (a class loop "
                           "inside solve); the caller must enumerate the classes")
    return got["v"]


class _Reuse:
    """Seed-independent objects of `_beam_search_all_candidates`: built on the first pass, handed back afterwards.

    The function looks the classes up in their modules on every call, so replacing the module attributes is enough.
    Objects are returned in construction order, which is fixed (same complex, same assignments)."""

    def __init__(self):
        from jacobik.ik import beam, fk, geometry, scorer
        from jacobik.topology import decompose

        self.targets = [(beam, "BeamSearch"), (fk, "ForwardKinematics"), (geometry, "GeometryExtractor"),
                        (scorer, "Scorer"), (decompose, "decompose_ligand")]
        self.orig = [getattr(m, n) for m, n in self.targets]
        self.store: Dict[str, list] = {n: [] for _, n in self.targets}
        self.pos: Dict[str, int] = {}
        self.first = True

    def _factory(self, name, orig):
        reuse = self

        def make(*a, **k):
            if reuse.first:
                obj = orig(*a, **k)
                if name == "BeamSearch":
                    real_search = obj.search

                    def search(*sa, **sk):
                        out = real_search(*sa, **sk)
                        obj._kept = out
                        return out

                    obj.search = search
                reuse.store[name].append(obj)
                return obj
            i = reuse.pos.get(name, 0)
            reuse.pos[name] = i + 1
            if name == "decompose_ligand":
                return None
            obj = reuse.store[name][i]
            if name == "BeamSearch":
                kept = obj._kept
                obj.search = lambda *sa, **sk: tuple(t.clone() if torch.is_tensor(t) else t for t in kept)
            return obj

        return make

    def __enter__(self):
        for (m, n), o in zip(self.targets, self.orig):
            setattr(m, n, self._factory(n, o))
        return self

    def next_pass(self):
        self.first = False
        self.pos = {}

    def __exit__(self, *exc):
        for (m, n), o in zip(self.targets, self.orig):
            setattr(m, n, o)


def seed_rows(builder, ctx, assignments, beam_width, seeds: Sequence[int]):
    """Starting rows of every seed. Returns (shared, per_seed): shared = (fks, donor_pos_idxs, ge, scorer),
    per_seed[i] = (theta_inits, X_ref, ideal_directions, candidate_idx_per_b)."""
    shared, per_seed = None, []
    with _Reuse() as reuse:
        for i, s in enumerate(seeds):
            # exactly what solve() does with `seed` before any draw
            ctx.np_rng = np.random.default_rng(s)
            torch.manual_seed(s)
            np.random.seed(s)
            fks, th, dpi, ge, sc, X, ideal, cand = builder._beam_search_all_candidates(ctx, assignments, beam_width)
            if shared is None:
                shared = (fks, dpi, ge, sc)
            per_seed.append((th, X, ideal, np.asarray(cand)))
            # P2R ring_states: this seed's per-row state energies and codes (None when off or no ring states)
            ctx._seed_row_state = getattr(ctx, "_seed_row_state", []) if i else []
            ctx._seed_row_state.append((getattr(ctx, "_row_state_E", None), getattr(ctx, "_row_state_tag", None),
                                        getattr(ctx, "_row_state_lab", None)))
            reuse.next_pass()
    return shared, per_seed


def assemble_batched(builder, ctx, fks, X, thetas, scores, conv, W_total, cand, B1, n_seeds, timing, lineage=None):
    """Dedup and conformer assembly for `n_seeds` stacked seeds from ONE kinematics pass over all rows.

    `ComplexBuilder._assemble_conformers` applies the kinematics to a seed's rows (for the dedup) and then again,
    one row and one ligand at a time, for each returned conformer; on the GPU each of those is some hundred eagerly
    launched operations. Here the coordinates of every row come from one `ComplexFK.apply` (the kinematics the
    solver itself used), the dedup runs per seed on its slice, and the selected rows are copied to the host once."""
    from jacobik.builder import engine as E
    from jacobik.ik.fk_compose import ComplexFK

    mol = ctx.mol
    n_keep = ctx.n_conformers if ctx.n_conformers is not None else max(W_total, ctx.n_coord_samples)
    t0 = time.perf_counter()
    with torch.no_grad():
        X_all = ComplexFK(fks).apply(thetas, X)                       # (B, N, 3)
    score_np = scores.cpu().numpy()
    conv_np = conv.cpu().numpy()
    t1 = time.perf_counter()
    deduper = E.deduper_for(mol) if E.OUT_DEDUP else None             # graph automorphisms: once, not per seed
    # CUDA solves of complexes with many automorphisms (>= BENCH_DEDUP_GPU_MIN_MAPS): the dedup's RMSD work runs on the
    # solver's GPU (same algorithm, float64; as dedup.select_distinct with DEDUP_DEVICE="solver"). QEZVEY (20000
    # automorphisms, 1536 rows per seed): 36-45 s -> 3.6 s per seed on an RTX 2060 SUPER; identical kept lists on all
    # 26 recorded dedups tested (6 parents, dev/tools/run/dedup_bench.py). A GPU failure repeats that seed on the CPU.
    # JACOBIK_BENCH_DEDUP_DEVICE=cpu keeps the CPU.
    gpu_dd = None
    if (deduper is not None and X_all.is_cuda and deduper.inv_maps.shape[0] >= BENCH_DEDUP_GPU_MIN_MAPS
            and os.environ.get("JACOBIK_BENCH_DEDUP_DEVICE", BENCH_DEDUP_DEVICE) == "solver"):
        try:
            gpu_dd = E.deduper_for(mol, X_all.device)
        except RuntimeError:
            gpu_dd = None

    def dedup_select(Xs, ss):
        if gpu_dd is not None:
            try:
                return gpu_dd.select(Xs, ss, n_keep)
            except RuntimeError as exc:
                log.warning("bench output dedup on %s failed (%s); repeating it on the CPU", Xs.device,
                            str(exc).splitlines()[0][:200])
                torch.cuda.empty_cache()
        return deduper.select(Xs, ss, n_keep)

    def one_seed(j):
        sl = slice(j * B1, (j + 1) * B1)
        ok = E.output_window(score_np[sl], float(getattr(ctx, "score_norm", 0.0) or 0.0))
        if deduper is not None:
            sel = np.asarray(dedup_select(X_all[sl], scores[sl]), dtype=int)
            sel = sel[ok[sel]]  # as engine._assemble_conformers: outside rows only fill slots left over
        else:
            sel = np.array([b for b in np.argsort(-score_np[sl]) if ok[b]][:n_keep], dtype=int)
        return sel + j * B1

    # The seeds' dedups are independent (each its own slice, its own greedy order; the deduper is read-only), so they
    # run in threads (the torch matrix products release the GIL). Each thread computes exactly what the sequential
    # loop computed (same operations, same single intra-op thread): identical selections. JACOBIK_DEDUP_SEED_THREADS=1
    # restores the sequential loop.
    n_thr = min(n_seeds, max(1, int(os.environ.get("JACOBIK_DEDUP_SEED_THREADS", DEDUP_SEED_THREADS))))
    if deduper is not None and n_thr > 1 and gpu_dd is None:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_thr) as ex:
            selected = list(ex.map(one_seed, range(n_seeds)))
    else:
        selected = [one_seed(j) for j in range(n_seeds)]
    t2 = time.perf_counter()
    if lineage is not None:       # diagnostic (JACOBIK_LINEAGE_OUT): trace every start to its outcome; no effect on results
        try:
            from tools.jdev import lineage as _lin
            _lin.record(ctx, lineage["assignments"], lineage["th_init"], thetas, scores, conv, X_all, cand, B1, n_seeds,
                        selected, deduper, n_keep, lineage["solver_wall_s"], lineage["beam_width"], lineage.get("keep"),
                        lineage.get("meta"))
        except Exception as exc:
            log.warning("lineage telemetry failed (%s: %s)", type(exc).__name__, exc)
    flat = np.concatenate(selected)
    X_sel = X_all.index_select(0, torch.as_tensor(flat, dtype=torch.long, device=X_all.device)).cpu().numpy()
    out, k = [], 0
    for j, sel in enumerate(selected):
        confs = []
        Xj = X_sel[k:k + len(sel)]
        k += len(sel)
        mirror, mirror_sc = E.mirror_completion(builder, mol, Xj, score_np[sel], X_all.device) if len(sel) else ([], np.zeros(0))
        msc = dict(zip(mirror, mirror_sc))
        for ci, b, refl in [(ci, b, False) for ci, b in enumerate(sel)] + [(ci, sel[ci], True) for ci in mirror]:
            mol_copy = E._shallow_copy_mol_for_conformer(mol)
            X_np = Xj[ci] * E.MIRROR if refl else Xj[ci]
            for i_at, atom in enumerate(mol_copy.atoms):
                atom.position = X_np[i_at].copy()
            E._refresh_coord_geometry(mol_copy)
            c = E.ConformerResult(complex=mol_copy, score=float(msc[ci]) if refl else float(score_np[b]), converged=bool(conv_np[b]),
                                  rank=ci + 1, trial_idx=int(b - j * B1), assignment_idx=int(cand[b]))
            c.mirror_of = int(b - j * B1) if refl else None
            tag = getattr(ctx, "_row_state_tag", None)
            if tag is not None and int(tag[b]):                      # P2R ring_states: row started in a non-embed state
                c.embed_idx = 1000 * int(tag[b])
            lab = getattr(ctx, "_row_state_lab", None)
            c.ring_state = str(lab[b]).rstrip(";") if lab is not None else None
            confs.append(c)
        confs.sort(key=lambda c: (c.score, c.mirror_of is None), reverse=True)     # a reflection ranks after its source
        for i, c in enumerate(confs):
            c.rank = i + 1
        out.append(confs)
    t3 = time.perf_counter()
    timing["kinematics"] = timing.get("kinematics", 0.0) + t1 - t0
    timing["dedup"] = timing.get("dedup", 0.0) + t2 - t1
    timing["build"] = timing.get("build", 0.0) + t3 - t2
    return out


def _has_state_rings(builder, five: bool = False) -> bool:
    """Whether any ligand has a saturated ring ring_states can offer states for (six-rings; five-rings with
    ring_states_five). Parents without one gain nothing from the ring_states route."""
    try:
        from rdkit import Chem
        from jacobik.topology import rings as ring_chair, rings as ring_states
        for lig in builder.ligands:
            m = Chem.MolFromSmiles(lig.smiles)
            if m is None:
                continue
            m = Chem.AddHs(m)
            if ring_chair.saturated_six_rings(m) or ((five or ring_states.FIVE) and ring_chair.saturated_rings(m, 5)):
                return True
    except Exception as exc:
        log.warning("ring_states ring check failed (%s); ring route kept", exc)
        return True
    return False


def _multi_class(builder) -> bool:
    """More than one configuration class (stereo extras or E/Z combinations): the engine's config_states path."""
    from jacobik.builder import embed as _ce
    if getattr(builder, "_ez_inner", False) or getattr(builder, "_stereo_inner", False):
        return False
    if _stereo_classes(builder) is not None:
        return True
    return bool(_ce.EZ_ENUMERATE and hasattr(builder, "_ez_combinations") and len(builder._ez_combinations()) > 1)


def _solve_config_states(builder, seeds, _solve, solve_kw, max_rows=None):
    """P2R config_states: every class of every seed in ONE joint IK. Each seed's classes are prepared as the engine's
    solve(seed=s) prepares them (engine._config_prepare); the (class, seed) blocks then run as one batch
    (engine._config_run) and each seed is merged on its own. `max_rows` (the bench's OOM retry) caps the rows per
    batch: classes stay whole and are split over several batches."""
    import time as _time
    solve = _solve or type(builder).solve
    kw = {k: v for k, v in solve_kw.items() if k not in ("seed",)}
    results, t_solver, t0 = {}, 0.0, _time.perf_counter()
    builder._config_max_rows = max_rows
    try:
        builder._config_defer = True
        try:
            out = {s: solve(builder, seed=s, **kw) for s in seeds}
        finally:
            builder._config_defer = False
        jobs = [s for s in seeds if isinstance(out[s], dict) and out[s].get("config_job")]
        for s in seeds:
            if s not in jobs:
                results[s] = out[s]
        if jobs:
            for s, res in zip(jobs, builder._config_run([out[s] for s in jobs])):
                results[s] = res
    finally:
        builder._config_max_rows = None
    results = {s: results[s] for s in seeds}
    for s, res in results.items():
        cs = (getattr(res, "solve_trace", None) or {}).get("config_states") or {}
        t_solver += float(cs.get("t_solve_s", 0.0))
    total = _time.perf_counter() - t0
    return results, {"s_solver": t_solver, "s_dedup_assembly": total - t_solver, "s_total": total,
                     "assembly_parts": {}, "config_states": True}


def _stereo_classes(builder):
    """The engine's stereo class list for this parent (engine.solve, stereo_candidates branch): (variants, elements,
    info) when there are extra classes, else None (E/Z combinations or a single class: solve_seeds as before)."""
    from jacobik.builder import embed as _ce
    from jacobik.builder import engine as E
    if not getattr(E, "STEREO_CANDIDATES", False) or getattr(builder, "_stereo_inner", False) \
            or getattr(builder, "_ez_inner", False) or not hasattr(builder, "_ez_combinations"):
        return None
    try:
        from jacobik.ik import stereo
        prod = builder._ez_combinations() if _ce.EZ_ENUMERATE else [[None] * len(builder.ligands)]
        elements = stereo.perceive_complex_stereo(builder, scope=E.STEREO_SCOPE)
        variants, info = stereo.superset_variants(builder, elements, prod,
                                                  achiral_environment=stereo.is_achiral_geometry(builder.geometry),
                                                  cost_ceiling_multiplier=E.STEREO_COST_CEILING,
                                                  near=bool(getattr(E, "CONFIG_NEAR", False) and getattr(E, "CONFIG_STATES", False)),
                                                  mirror=getattr(E, "OUT_MIRROR", False),
                                                  remote=bool(getattr(E, "CONFIG_REMOTE", False) and getattr(E, "CONFIG_NEAR", False)
                                                              and getattr(E, "CONFIG_STATES", False)))
    except Exception as exc:  # as the engine: production classes only
        log.warning("stereo candidates: class construction failed (%s); production classes only", exc)
        return None
    builder._stereo_class_info = info
    derived = bool(getattr(E, "CONFIG_STATES", False) and info.get("remote_derived"))
    return (variants, elements, info) if info.get("n_extra", 0) > 0 or derived else None


def _set_class_fixes(builder, v, elements):
    """Ligand fixes of one stereo class, exactly as engine._solve_stereo_variants sets them."""
    from jacobik.ik import stereo
    if getattr(v, "production", False):
        for li, lig in enumerate(builder.ligands):
            lig.ez_fix = v.prod_ez[li] if v.prod_ez else None
            lig.chiral_fix = None
            lig._strict_stereo = False
    else:
        fx = stereo.build_variant_ligand_fixes(v, elements, builder.ligands)
        for li, lig in enumerate(builder.ligands):
            pz = list(v.prod_ez[li] or []) if getattr(v, "prod_ez", None) else []
            lig.ez_fix = (pz + list(fx[li]["ez"])) or None
            lig.chiral_fix = fx[li]["chiral"] or None
            lig._strict_stereo = True


def _solve_stereo_classes(builder, seeds, classes, max_rows, batched_assembly, _solve, diagnostics, solve_kw):
    """Every stereo class of the parent (engine._solve_stereo_variants semantics), each solved with all seeds in one
    pass; merged per seed as the engine merges: cross-class dedup with production classes first (superset rule), then
    the output window (or out_per_class) and the n_conformers cap. Conformers carry stereo_variant_idx,
    stereo_production, requested_stereo_label, stereo_label and class_label as in the engine."""
    from dataclasses import replace
    from jacobik.builder import engine as E
    from jacobik.ik import dedup as dd
    from jacobik.ik import stereo
    variants, elements, _info = classes
    per_class, infos, failures = [], [], []
    try:
        builder._stereo_inner = True
        for vi, v in enumerate(variants):
            if not getattr(v, "feasible", True):
                continue
            _set_class_fixes(builder, v, elements)
            try:
                res, info = solve_seeds(builder, seeds=seeds, max_rows=max_rows, batched_assembly=batched_assembly,
                                        _solve=_solve, diagnostics=diagnostics, **solve_kw)
            except Exception as exc:  # a class that cannot be embedded or solved is not a candidate (as the engine)
                failures.append(f"variant {v.class_id}: {type(exc).__name__}: {exc}")
                log.info("stereo class %s failed (%s), skipped", v.class_id, exc)
                continue
            per_class.append((vi, v, res))
            infos.append(info)
    finally:
        builder._stereo_inner = False
        for lig in builder.ligands:
            lig.ez_fix = None
            lig.chiral_fix = None
            lig._strict_stereo = False
    if not per_class:
        raise ValueError("stereo classes: no variant succeeded" + (f" ({'; '.join(failures[:3])})" if failures else ""))
    out_pc = builder.out_per_class if getattr(builder, "out_per_class", None) is not None else E.OUT_PER_CLASS
    merged = {}
    for s in seeds:
        allc = []
        for vi, v, res in per_class:
            for c in res[s].conformers:
                c.stereo_variant_idx = vi
                c.stereo_production = bool(getattr(v, "production", False))
                c.requested_stereo_label = v.class_id
                pos = np.array([a.position for a in c.complex.atoms])
                syms = [a.symbol for a in c.complex.atoms]
                labels = stereo.compute_conformer_stereo_labels(elements, pos, syms, c=c)
                c.stereo_label = ",".join(f"{k}:{x}" for k, x in sorted(labels.items())) if labels else "none"
                geom = getattr(c.complex, "coordination_geometry", getattr(builder, "geometry", "ideal"))
                c.class_label = f"{geom}|S:{c.stereo_label}|A:{c.assignment_idx if c.assignment_idx is not None else 0}"
                allc.append((c, res[s]))
        if not allc:
            merged[s] = E.EnsembleResult(conformers=[], failure="stereo classes returned no conformers")
            continue
        allc.sort(key=lambda t: (t[0].score, getattr(t[0], 'mirror_of', None) is None), reverse=True)  # reflection after source
        deduper = dd.deduper_for(allc[0][0].complex)
        coords = torch.tensor(np.array([[a.position for a in c.complex.atoms] for c, _ in allc]))
        sc_t = torch.tensor([c.score for c, _ in allc], dtype=torch.float64)
        prod = torch.tensor([1.0 if c.stereo_production else 0.0 for c, _ in allc], dtype=torch.float64)
        refl = torch.tensor([1.0e-9 if c.mirror_of is not None else 0.0 for c, _ in allc], dtype=torch.float64)
        keep = deduper.select(coords, sc_t + 1.0e6 * prod - refl, k=None, th=dd.DEDUP_RMSD)
        keep = sorted(keep, key=lambda i: (-float(sc_t[i]), allc[i][0].mirror_of is not None))
        allc = [allc[i] for i in keep]
        best = allc[0][1]
        z = float(best.scorer._agg_norm()) if hasattr(best.scorer, "_agg_norm") else 0.0
        confs = [c for c, _ in allc]
        if out_pc:
            confs = builder._apply_out_per_class(confs, z_norm=z if z else 1.0)
        else:
            ok = E.output_window(np.array([c.score for c in confs]), z)
            confs = [c for c, k in zip(confs, ok) if k]
            n = solve_kw.get("n_conformers")
            confs = confs[:n] if n is not None else confs
        for i, c in enumerate(confs, start=1):           # in place, as the engine (replace() drops non-field attributes)
            c.rank = i
        merged[s] = E.EnsembleResult(conformers=confs, solve_trace=best.solve_trace, scorer=best.scorer)
        merged[s].solve_trace = dict(merged[s].solve_trace or {}, n_stereo_variants_solved=len(per_class))
    tot = lambda k: sum(i.get(k, 0.0) for i in infos)
    info = dict(s2b_bend=next((i.get("s2b_bend") for i in infos if i.get("s2b_bend") is not None), None),
                s_embed_shell_enum=tot("s_embed_shell_enum"), s_seed_rows=tot("s_seed_rows"), s_solver=tot("s_solver"),
                s_dedup_assembly=tot("s_dedup_assembly"), s_diagnostics=tot("s_diagnostics"), s_total=tot("s_total"),
                assembly_parts={}, stereo_classes=dict(n_solved=len(per_class), n_total=len(variants),
                                                       failures=failures[:5]))
    return merged, info


def solve_seeds(builder, seeds: Sequence[int], max_rows: Optional[int] = None, batched_assembly: bool = False,
                _solve=None, diagnostics=None, **solve_kw):
    """One embed and set-up, all `seeds` in one solver pass. Returns ({seed: EnsembleResult}, info).
    `_solve`: the unbound ComplexBuilder.solve to use (for callers that have replaced the class attribute).

    max_rows: upper limit of rows per solver pass (GPU memory); seeds are split into consecutive groups to stay
    under it (a single seed always runs, whatever its size)."""
    from jacobik.builder import embed as _ce
    from jacobik.builder import engine as E
    from jacobik.ik.solver_replay import ReplaySolver, solver_class

    if getattr(E, "CONFIG_STATES", False) and _multi_class(builder):
        return _solve_config_states(builder, seeds, _solve, solve_kw, max_rows)
    classes = _stereo_classes(builder)
    if classes is not None:
        return _solve_stereo_classes(builder, seeds, classes, max_rows, batched_assembly, _solve, diagnostics, solve_kw)
    with safe_cuda_eigh_chunking():
        if _ce.EZ_ENUMERATE and not getattr(builder, "_ez_inner", False) and not getattr(builder, "_stereo_inner", False) \
                and hasattr(builder, "_ez_combinations"):
            variants = builder._ez_combinations()
            if len(variants) > 1:
                try:
                    builder._ez_inner = True
                    var_results = []
                    variant_infos = []
                    for vi, fixes in enumerate(variants):
                        for lig, f in zip(builder.ligands, fixes):
                            lig.ez_fix = f
                        try:
                            res, info = solve_seeds(builder, seeds=seeds, max_rows=max_rows, batched_assembly=batched_assembly,
                                                    _solve=_solve, diagnostics=diagnostics, **solve_kw)
                            var_results.append((res, vi))
                            variant_infos.append(info)
                        except ValueError as exc:
                            log.info("E/Z enumeration (solve_seeds): combination %s not embeddable (%s), skipped", fixes, exc)
                finally:
                    builder._ez_inner = False
                    for lig in builder.ligands:
                        lig.ez_fix = None
                if not var_results:
                    raise ValueError("E/Z enumeration: no combination could be embedded")
                from dataclasses import replace
                merged_results = {}
                var_res_map = {vi: res for res, vi in var_results}
                for s in seeds:
                    allc = [(replace(c, embed_idx=vi), vi) for res, vi in var_results for c in res[s].conformers]
                    allc.sort(key=lambda t: (t[0].score, getattr(t[0], 'mirror_of', None) is None), reverse=True)  # reflection after source
                    best_v = allc[0][1] if allc else var_results[0][1]
                    best_res = var_res_map[best_v]
                    sc = best_res[s].scorer
                    z = float(sc._agg_norm()) if hasattr(sc, "_agg_norm") else 0.0
                    keep = E.output_window(np.array([c.score for c, _ in allc]), z)
                    allc = [t for t, k in zip(allc, keep) if k]
                    n = solve_kw.get("n_conformers")
                    allc = allc[:n] if n is not None else allc
                    kept = [replace(c, rank=rank) for rank, (c, _) in enumerate(allc, start=1)]
                    merged_results[s] = E.EnsembleResult(conformers=kept, solve_trace=best_res[s].solve_trace, scorer=sc)
                merged_parts = {}
                for v in variant_infos:
                    for k, val in v.get("assembly_parts", {}).items():
                        merged_parts[k] = merged_parts.get(k, 0.0) + val
                # S2b: the saved bend constants must describe one structure for every variant (stop, do not guess)
                metas = [v.get("s2b_bend") for v in variant_infos]
                keys = {(m["bend_column_def_sha256"], tuple(m["theta_widths"])) for m in metas if m is not None}
                if len(keys) > 1:
                    raise RuntimeError("S2b: E/Z variants differ in angle-bend or theta structure "
                                       f"({len(keys)} distinct structures); refusing to save one state for the pool")
                merged_info = {
                    "s2b_bend": next((m for m in metas if m is not None), None),
                    "s_embed_shell_enum": sum(v.get("s_embed_shell_enum", 0.0) for v in variant_infos),
                    "s_seed_rows": sum(v.get("s_seed_rows", 0.0) for v in variant_infos),
                    "s_solver": sum(v.get("s_solver", 0.0) for v in variant_infos),
                    "s_dedup_assembly": sum(v.get("s_dedup_assembly", 0.0) for v in variant_infos),
                    "s_diagnostics": sum(v.get("s_diagnostics", 0.0) for v in variant_infos),
                    "s_total": sum(v.get("s_total", 0.0) for v in variant_infos),
                    "assembly_parts": merged_parts,
                    "variant_infos": variant_infos,
                    "n_variants_embedded": len(var_results),
                    "n_variants_total": len(variants),
                }
                return merged_results, merged_info

        seeds = [int(s) for s in seeds]
        t = [time.perf_counter()]
        ctx, assignments, W, T = prepare(builder, seeds[0], _solve=_solve, **solve_kw)
        from jacobik.ik import ik_solver as IKM
        native_score_only = ctx.device.type == 'cuda' and IKM.OPTIMIZER in ('irprop', 'irprop_minus', 'adam_score')
        if solver_class(ctx.device) is not ReplaySolver and not native_score_only:
            raise RuntimeError("solve_seeds needs the no-culling replay solver (CUDA, JACOBIK_GPU_REPLAY=1); with row "
                               "culling the seeds would be culled against each other")
        t.append(time.perf_counter())
        (fks, dpi, ge, scorer), rows = seed_rows(builder, ctx, assignments, W, seeds)
        ctx.score_norm = float(scorer._agg_norm()) if hasattr(scorer, "_agg_norm") else 0.0  # output window (engine)
        t.append(time.perf_counter())

        # row budget per class / early cull: as _solve_for_assignments
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
            rows = [([x[kt] for x in th], X[kt] if X.dim() == 3 else X, ideal[kt], cand[keep]) for th, X, ideal, cand in rows]
            ctx._seed_row_state = [tuple(None if a is None else a[keep] for a in t) for t in getattr(ctx, "_seed_row_state", [])]
            W_total = len(keep) // max(ctx.n_coord_samples, 1)
        B1 = rows[0][0][0].shape[0]                                   # rows of one seed
        # S2b capture: bend-prior constants and theta layout of this solve (read only; nothing here changes a solve)
        from jacobik.ik.ik_solver import bend_state_constants
        bend_meta = bend_state_constants(fks, ctx.bend_prior, [int(t.shape[1]) for t in rows[0][0]])
        sym = ctx.symmetric_donor_groups
        q3_sym = [g for g in sym if len({ctx.all_donor_groups[d][0] for d in g}) == 1]

        per = max(1, (max_rows // B1) if max_rows else len(seeds))
        groups = [list(range(i, min(i + per, len(seeds)))) for i in range(0, len(seeds), per)]
        results, t_solve, t_out, t_parts = {}, 0.0, 0.0, {}
        diagnostic_total = 0.0
        for g in groups:
            t0 = time.perf_counter()
            th = [torch.cat([rows[i][0][li] for i in g], dim=0) for li in range(len(fks))]
            X = torch.cat([rows[i][1] for i in g], dim=0) if rows[g[0]][1].dim() == 3 else rows[g[0]][1]
            ideal = torch.cat([rows[i][2] for i in g], dim=0)
            cand = np.concatenate([rows[i][3] for i in g])
            rs = getattr(ctx, "_seed_row_state", None)
            if rs and all(rs[i][0] is not None for i in g):
                ctx._row_state_E = np.concatenate([rs[i][0] for i in g])
                ctx._row_state_tag = np.concatenate([rs[i][1] for i in g])
                ctx._row_state_lab = np.concatenate([rs[i][2] for i in g])
            else:
                ctx._row_state_E = ctx._row_state_tag = ctx._row_state_lab = None
            from tools.jdev import lineage as _lin
            th_lin = [t_.detach().clone() for t_ in th] if (_lin.enabled() and batched_assembly) else None
            thetas, conv, scores, trace = builder._run_joint_ik(
                ctx, fks, scorer, ge, X, th, dpi, ideal, sym, q3_sym, W_total * len(g), T, cand)
            bend_g = getattr(builder, "_last_bend_prior", None)  # S2b capture: bend prior per row of this pass
            if ctx.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            lin = None if th_lin is None else dict(
                assignments=assignments, th_init=th_lin, solver_wall_s=t1 - t0, beam_width=W, keep=keep,
                meta=dict(seeds=[seeds[i] for i in g], ez_variant=getattr(builder, "_lineage_variant", None),
                          code=os.environ.get("JACOBIK_LINEAGE_CODE"), arm=os.environ.get("JACOBIK_LINEAGE_ARM")))
            batched = assemble_batched(builder, ctx, fks, X, thetas, scores, conv, W_total, cand, B1, len(g), t_parts, lineage=lin) \
                if batched_assembly else None
            diagnostic_group = 0.0
            for j, i in enumerate(g):
                sl = slice(j * B1, (j + 1) * B1)
                confs = batched[j] if batched is not None else builder._assemble_conformers(
                    ctx, fks, X[sl] if X.dim() == 3 else X, [x[sl] for x in thetas], scores[sl], conv[sl], W_total, cand[sl])
                E.attach_solver_state(confs, [x[sl] for x in thetas], None if bend_g is None else bend_g[sl])  # S2b
                if diagnostics is not None:
                    diagnostic_start = time.perf_counter()
                    diagnostics(confs, ctx, assignments, ge, scorer, rows, i)
                    diagnostic_group += time.perf_counter() - diagnostic_start
                tr = {k: v[:, sl] for k, v in trace.items()} if trace else trace
                res = E.EnsembleResult(conformers=confs, solve_trace=tr, scorer=scorer)
                if len(assignments) > 1:
                    res.assignment_search = dict(n_candidates=len(assignments),
                                                 candidate_assignments=[a.tolist() for a in assignments])
                results[seeds[i]] = res
            t_solve += t1 - t0
            t_out += time.perf_counter() - t1 - diagnostic_group
            diagnostic_total += diagnostic_group
        info = dict(s2b_bend=bend_meta, rows_per_seed=int(B1), seeds_per_pass=[len(g) for g in groups],
                    s_embed_shell_enum=t[1] - t[0], s_seed_rows=t[2] - t[1], s_solver=t_solve, s_dedup_assembly=t_out,
                    s_total=time.perf_counter() - t[0], assembly_parts=t_parts, s_diagnostics=diagnostic_total)
        return results, info
