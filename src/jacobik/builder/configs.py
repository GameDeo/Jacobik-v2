"""P2R config_remote: derive the classes of REMOTE stereo elements after the solve (plan Step 4, stereo scaling rule).

A remote element (an open tetrahedral centre or E/Z bond more than two bonds from every donor, not in a ring) does not
change which basins the metal-bound part of the complex can reach, so its classes are not solved: each solved class's
best rows are inverted at that element locally and repaired by a short joint IK.

The local inversion is an exact isometry of the moved atoms:
  * centre c with neighbours a, b, n3, n4: the subtrees behind a and b are reflected through the plane (c, n3, n4).
    That plane swaps the two subtrees and inverts c; every bond length and every angle at c is kept;
  * E/Z bond b=c: the subtrees behind c's two other neighbours are turned 180 degrees about the bond axis.
Moved subtrees must not lead back to the element (ring), and must contain no metal, donor or atom of another stereo
element (that element would be inverted too). An element without two such subtrees is not derivable; it is logged
and belongs in the solved product ("widen near")."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from jacobik.builder.result import ConformerResult, EnsembleResult, attach_solver_state
from jacobik.core import gpu

cp = gpu.cp

log = logging.getLogger(__name__)


def _adjacency(mol) -> Dict[int, List[int]]:
    """0-based adjacency of the complex graph (mol.bonds are 1-based)."""
    adj: Dict[int, List[int]] = {i: [] for i in range(len(mol.atoms))}
    for bd in mol.bonds:
        i, j = bd.i - 1, bd.j - 1
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _branch(adj, start: int, block: int) -> Optional[Set[int]]:
    """Atoms reachable from `start` without passing `block`; None if the walk reaches `block` again (ring)."""
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w == block:
                if u != start:
                    return None
                continue
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def _global(mol, li: int, idx: int) -> int:
    """0-based complex index of ligand `li`'s SMILES atom `idx`."""
    return int(mol.ligands[li].atom_indices[idx]) - 1


def plan(mol, elements: Sequence, names: Sequence[str]) -> Tuple[List[dict], List[str]]:
    """Inversion operations for the remote elements `names`: ([op], [names not derivable]).
    op = {"name", "kind": "centre"|"bond", "move": int array, "c", "n3", "n4"} (centre) or {"b", "c"} (bond)."""
    from jacobik.ik.stereo import StereoType
    adj = _adjacency(mol)
    metal = int(getattr(mol, "metal_idx", 1)) - 1
    donors: Set[int] = set()
    for li, lig in enumerate(mol.ligands):
        for g in getattr(lig, "donor_atoms", None) or []:
            donors.add(int(g) - 1)
    for a in range(len(mol.atoms)):
        if metal in adj[a]:
            donors.add(a)
    stereo_atoms: Dict[str, Set[int]] = {}
    for e in elements:
        try:
            s = {_global(mol, e.ligand_idx, e.atom_idx)}
            if e.atom_idx_end is not None:
                s.add(_global(mol, e.ligand_idx, e.atom_idx_end))
            stereo_atoms[e.name] = s
        except Exception:
            continue
    ops, bad = [], []
    by_name = {e.name: e for e in elements}
    for name in names:
        e = by_name.get(name)
        if e is None or name not in stereo_atoms:
            bad.append(name)
            continue
        others = set().union(*[s for k, s in stereo_atoms.items() if k != name]) if len(stereo_atoms) > 1 else set()
        forbidden = donors | others | {metal}
        if e.element_type == StereoType.LIGAND_TETRAHEDRAL:
            c = _global(mol, e.ligand_idx, e.atom_idx)
            nb = adj[c]
            if len(nb) != 4:
                bad.append(name)
                continue
            br = {n: _branch(adj, n, c) for n in nb}
            ok = [n for n in nb if br[n] is not None and not (br[n] & forbidden)]
            ok.sort(key=lambda n: len(br[n]))
            if len(ok) < 2:
                bad.append(name)
                continue
            a, b = ok[0], ok[1]
            n3, n4 = [n for n in nb if n not in (a, b)]
            ops.append(dict(name=name, kind="centre", move=np.array(sorted(br[a] | br[b])), c=c, n3=n3, n4=n4))
        elif e.element_type == StereoType.DOUBLE_BOND_EZ and e.atom_idx_end is not None:
            u, v = _global(mol, e.ligand_idx, e.atom_idx), _global(mol, e.ligand_idx, e.atom_idx_end)
            done = False
            for b, c in ((u, v), (v, u)):            # turn the substituents of c (the end away from the metal)
                subs = [n for n in adj[c] if n != b]
                if len(subs) != 2:
                    continue
                brs = [_branch(adj, n, c) for n in subs]
                if any(x is None or (x & forbidden) for x in brs):
                    continue
                ops.append(dict(name=name, kind="bond", move=np.array(sorted(brs[0] | brs[1])), b=b, c=c))
                done = True
                break
            if not done:
                bad.append(name)
        else:
            bad.append(name)
    return ops, bad


def apply(op: dict, X: np.ndarray) -> np.ndarray:
    """The inverted coordinates of rows X (B, N, 3) under one operation (an isometry of the moved atoms)."""
    X = np.array(X, dtype=float, copy=True)
    m = op["move"]
    if op["kind"] == "centre":
        p = X[:, op["c"]]
        n = np.cross(X[:, op["n3"]] - p, X[:, op["n4"]] - p)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        d = ((X[:, m] - p[:, None]) * n[:, None]).sum(-1, keepdims=True)
        X[:, m] = X[:, m] - 2.0 * d * n[:, None]
    else:
        p = X[:, op["c"]]
        u = X[:, op["c"]] - X[:, op["b"]]
        u /= np.linalg.norm(u, axis=-1, keepdims=True) + 1e-12
        r = X[:, m] - p[:, None]
        X[:, m] = p[:, None] + 2.0 * (r * u[:, None]).sum(-1, keepdims=True) * u[:, None] - r
    return X

def _e():
    from jacobik.builder import engine

    return engine


class _ConfigsPrepMixin:
    """Variant enumeration, per-class capping, and result merging helpers."""

    def _ez_combinations(self) -> List[List[Optional[list]]]:
        """Per combination, one E/Z fix list per ligand (chelate_embed.ez_variants); one all-None combination if there
        is nothing to enumerate. H-explicit SMILES are left alone (their donor indices are full-graph ones)."""
        import itertools

        from jacobik.builder import embed as _ce

        per = []
        for lig in self.ligands:
            if "[H" in lig.smiles:
                per.append([None])
                continue
            per.append([x or None for x in _ce.ez_variants(lig.smiles, lig.donors)])
        combos = [list(c) for c in itertools.product(*per)]
        return combos if len(combos) <= 2 ** _ce.EZ_MAX_BONDS else [[None] * len(self.ligands)]

    def _solve_ez_variants(self, variants, kw) -> EnsembleResult:
        """Solve once per E/Z combination and merge by score (the output window across all of them); each conformer's
        ``embed_idx`` is its combination. The scorer / trace returned are those of the #1's combination."""
        results, kept_variants, failures = [], [], []
        try:
            self._ez_inner = True
            for fixes in variants:
                for lig, f in zip(self.ligands, fixes):
                    lig.ez_fix = f
                try:
                    result = self.solve(**kw)
                    if not result.conformers:
                        failures.append(result.failure or "variant returned no conformers")
                        _e().log.info("E/Z enumeration: combination %s returned no conformers, skipped", fixes)
                        continue
                    results.append(result)
                    kept_variants.append(fixes)
                except _e()._LigandEmbedFailure as exc:  # a configuration ETKDG cannot embed at all is not a candidate
                    failures.append(f"{type(exc).__name__}: {exc}")
                    _e().log.info("E/Z enumeration: combination %s not embeddable (%s), skipped", fixes, exc)
                except Exception as exc:  # an internal error is not an embed failure: record it, keep the other combinations
                    failures.append(f"internal error {type(exc).__name__}: {exc}")
                    _e().log.warning("E/Z enumeration: combination %s failed internally, skipped", fixes, exc_info=True)
        finally:
            self._ez_inner = False
            for lig in self.ligands:
                lig.ez_fix = None
        return self._merge_ez_results(results, kept_variants, failures, kw)

    def _merge_ez_results(self, results, kept_variants, failures, kw) -> EnsembleResult:
        """Merge per-combination E/Z results by score (embed_idx = combination), output window, ranks."""
        if not results:
            return EnsembleResult(conformers=[], failure="E/Z enumeration: no combination could be embedded" +
                                  (f" ({'; '.join(failures)}" if failures else "") +
                                  (")" if failures else ""))
        variants = kept_variants
        allc = [(replace(c, embed_idx=vi), vi) for vi, r in enumerate(results) for c in r.conformers]
        if not allc:
            reasons = sorted({r.failure for r in results if r.failure})
            return EnsembleResult(conformers=[], failure="; ".join(reasons) or "E/Z enumeration returned no conformers")
        allc.sort(key=lambda t: (t[0].score, getattr(t[0], 'mirror_of', None) is None), reverse=True)
        best_v = allc[0][1] if allc else 0
        sc = results[best_v].scorer
        z = float(sc._agg_norm()) if hasattr(sc, "_agg_norm") else 0.0
        keep = _e().output_window(np.array([c.score for c, _ in allc]), z) if allc else []
        allc = [t for t, k in zip(allc, keep) if k]
        n = kw.get("n_conformers")
        allc = allc[:n] if n is not None else allc
        kept = [replace(c, rank=rank) for rank, (c, _) in enumerate(allc, start=1)]
        _e().log.info("E/Z enumeration: %d combinations, #1 from combination %d", len(variants), best_v)
        return EnsembleResult(conformers=kept, solve_trace=results[best_v].solve_trace, scorer=sc,
                              assignment_search=results[best_v].assignment_search,
                              failure=results[best_v].failure)

    def _apply_out_per_class(self, conformers: List[ConformerResult], z_norm: float = 1.0) -> List[ConformerResult]:
        """Keep up to OUT_PER_CLASS_K distinct conformers per class_label within OUT_PER_CLASS_WINDOW."""
        from collections import defaultdict
        if not conformers:
            return conformers
        conformers.sort(key=lambda c: (getattr(c.score, 'total', c.score), getattr(c, 'mirror_of', None) is None), reverse=True)  # a reflection after its source
        best_score = getattr(conformers[0].score, 'total', conformers[0].score)
        zn = float(z_norm) if z_norm and z_norm > 0 else 1.0
        global_win = _e().OUT_WINDOW / zn
        class_win = (_e().OUT_PER_CLASS_WINDOW or 35.0) / zn

        by_class = defaultdict(list)
        for c in conformers:
            by_class[c.class_label].append(c)

        kept = []
        for _cl, cl_confs in by_class.items():
            cl_best = getattr(cl_confs[0].score, 'total', cl_confs[0].score)
            if cl_best >= best_score - global_win:
                eligible = [c for c in cl_confs if getattr(c.score, 'total', c.score) >= cl_best - class_win]
                k_lim = _e().OUT_PER_CLASS_K if _e().OUT_PER_CLASS_K is not None else 8
                kept.extend(eligible[:k_lim])

        kept.sort(key=lambda c: (getattr(c.score, 'total', c.score), getattr(c, 'mirror_of', None) is None), reverse=True)
        for rank, c in enumerate(kept, start=1):
            c.rank = rank
        return kept

    def _solve_stereo_variants(self, variants, elements, kw, out_per_class: bool = False) -> EnsembleResult:
        """Solve once per stereoisomer variant, compute 3D conformer labels, and merge conformers."""
        from collections import defaultdict
        from dataclasses import replace
        from jacobik.ik import stereo
        from rdkit import Chem

        results = []
        kept_variants = []
        failures = []

        try:
            self._stereo_inner = True
            for vi, v in enumerate(variants):
                if not getattr(v, "feasible", True):
                    continue
                self._set_variant_fixes(v, elements)

                try:
                    res = self.solve(**kw)
                    if res.conformers:
                        self._label_class_rows(res.conformers, vi, v, elements)
                        results.append((res, vi))
                        kept_variants.append(v)
                    else:
                        failures.append(res.failure or f"variant {v.class_id} produced 0 conformers")
                except Exception as exc:
                    failures.append(f"variant {v.class_id}: {exc}")
        finally:
            self._stereo_inner = False
            self._clear_variant_fixes()
        return self._merge_stereo_results(results, failures, kw, out_per_class)

    def _set_variant_fixes(self, v, elements) -> None:
        """Ligand fixes of one stereo class (production class: the E/Z path's inputs, ez fixes only, non-strict)."""
        from jacobik.ik import stereo
        if getattr(v, "production", False):
            for li, lig in enumerate(self.ligands):
                lig.ez_fix = v.prod_ez[li] if v.prod_ez else None
                lig.chiral_fix = None
                lig._strict_stereo = False
        else:
            lig_fixes = stereo.build_variant_ligand_fixes(v, elements, self.ligands)
            for li, lig in enumerate(self.ligands):
                pz = list(v.prod_ez[li] or []) if getattr(v, "prod_ez", None) else []
                lig.ez_fix = (pz + list(lig_fixes[li]["ez"])) or None
                lig.chiral_fix = lig_fixes[li]["chiral"] or None
                lig._strict_stereo = True

    def _prefetch_embeds(self, variants, elements, kw, mode) -> None:
        """embed_refine: one collect pass over every class's shell build, so the ligand embeds of all classes are refined
        in ONE batched MACE optimisation (builder.mace_embed); the class preparations then read them from its cache."""
        from jacobik.builder import mace_embed

        try:
            with mace_embed.collecting():
                for v in variants:
                    if mode != "ez" and not getattr(v, "feasible", True):
                        continue
                    if mode == "ez":
                        for lig, f in zip(self.ligands, v):
                            lig.ez_fix = f
                    else:
                        self._set_variant_fixes(v, elements)
                    try:
                        self._build_shell(kw.get("fast_embed", False), False, embed_seed=kw.get("embed_seed"),
                                          chelate_embed=kw.get("chelate_embed", True), embed=kw.get("embed"))
                    except Exception as exc:  # the class's own preparation reports it
                        _e().log.debug("embed prefetch: class shell failed (%s)", exc)
        finally:
            self._clear_variant_fixes()

    def _clear_variant_fixes(self) -> None:
        for lig in self.ligands:
            lig.ez_fix = None
            lig.chiral_fix = None
            lig._strict_stereo = False

    def _label_class_rows(self, conformers, vi, v, elements) -> None:
        """Class descriptors of one class's rows: variant, production flag, requested and perceived stereo labels."""
        from jacobik.ik import stereo
        if not conformers:
            return
        cached_st = getattr(self, "_cached_stereo_elements", None)
        already_labeled = cached_st is not None and cached_st[1] == elements
        syms = None if already_labeled else [a.symbol for a in conformers[0].complex.atoms]
        geom_default = getattr(self, "geometry", "ideal")
        for c in conformers:
            c.stereo_variant_idx = vi
            c.stereo_production = bool(getattr(v, "production", False))
            c.requested_stereo_label = v.class_id
            if not already_labeled or c.stereo_label is None:
                if syms is None:
                    syms = [a.symbol for a in c.complex.atoms]
                pos = np.array([a.position for a in c.complex.atoms])
                labels = stereo.compute_conformer_stereo_labels(elements, pos, syms, c=c)
                c.stereo_label = ",".join(f"{k}:{v}" for k, v in sorted(labels.items())) if labels else "none"
            geom_label = getattr(c.complex, "coordination_geometry", geom_default)
            a_idx = c.assignment_idx if c.assignment_idx is not None else 0
            c.class_label = f"{geom_label}|S:{c.stereo_label}|A:{a_idx}"

    def _merge_stereo_results(self, results, failures, kw, out_per_class) -> EnsembleResult:
        """Merge per-class results: cross-class dedup (production rows first), output window, ranks."""
        if not results:
            return EnsembleResult(conformers=[], failure="Stereo candidates: no variant succeeded" +
                                  (f" ({'; '.join(failures[:3])})" if failures else ""))

        all_confs = [c for res, _vi in results for c in res.conformers]
        if not all_confs:
            return EnsembleResult(conformers=[], failure="Stereo candidates returned no conformers")

        all_confs.sort(key=lambda c: (getattr(c.score, 'total', c.score), getattr(c, 'mirror_of', None) is None), reverse=True)  # a reflection after its source

        # Deduplicate merged conformers across variants
        from jacobik.ik import dedup as dd
        deduper = dd.deduper_for(all_confs[0].complex)
        coords = gpu.as_gpu(np.array([[a.position for a in c.complex.atoms] for c in all_confs]), cp.float64)
        scores_t = np.array([getattr(c.score, 'total', c.score) for c in all_confs], dtype=np.float64)
        # superset rule: rows of production classes are visited first (never dropped for an extra row); only extra
        # rows can be removed as duplicates of kept ones.
        prod_flag = np.array([1.0 if getattr(c, "stereo_production", False) else 0.0 for c in all_confs], dtype=np.float64)
        # out_mirror: at equal score the source is visited (and ranked) before its reflection (0 in production)
        refl = np.array([1.0e-9 if getattr(c, "mirror_of", None) is not None else 0.0 for c in all_confs], dtype=np.float64)
        keep_idxs = deduper.select(coords, cp.asarray(scores_t + 1.0e6 * prod_flag - refl), k=None, th=dd.DEDUP_RMSD)
        keep_idxs = sorted(keep_idxs, key=lambda i: (-float(scores_t[i]), getattr(all_confs[i], "mirror_of", None) is not None))
        all_confs = [all_confs[i] for i in keep_idxs]

        best_c = all_confs[0]
        best_res = next(r for r, vi in results if any(c is best_c for c in r.conformers))

        if out_per_class:
            sc = getattr(best_res, "scorer", None)
            z = float(sc._agg_norm()) if (sc is not None and hasattr(sc, "_agg_norm")) else 1.0
            all_confs = self._apply_out_per_class(all_confs, z_norm=z)
        else:
            sc = best_res.scorer
            z = float(sc._agg_norm()) if hasattr(sc, "_agg_norm") else 0.0
            keep = _e().output_window(np.array([c.score for c in all_confs]), z)
            all_confs = [c for c, k in zip(all_confs, keep) if k]
            n = kw.get("n_conformers")
            all_confs = all_confs[:n] if n is not None else all_confs
            for rank, c in enumerate(all_confs, start=1):
                c.rank = rank

        merged_trace = {}
        all_traces = [res.solve_trace for res, vi in results if res.solve_trace is not None]
        if all_traces:
            for k in all_traces[0].keys():
                vals = [tr[k] for tr in all_traces if k in tr]
                if not vals:
                    continue
                v0 = vals[0]
                if isinstance(v0, (int, float, np.integer, np.floating)):
                    merged_trace[k] = sum(vals)
                elif isinstance(v0, np.ndarray):
                    try:
                        merged_trace[k] = np.concatenate(vals, axis=-1 if v0.ndim == 2 else 0)
                    except Exception:
                        merged_trace[k] = v0
                else:
                    merged_trace[k] = v0
        merged_trace["n_stereo_variants_solved"] = len(results)
        merged_trace["variant_traces"] = all_traces

        sc = getattr(best_res, "scorer", None)
        assign_search = getattr(best_res, "assignment_search", None)
        return EnsembleResult(
            conformers=all_confs,
            solve_trace=merged_trace or best_res.solve_trace,
            scorer=sc,
            assignment_search=assign_search,
            failure=best_res.failure,
        )


class _ConfigsBatchedMixin:
    """Batched multi-class joint solve and remote stereo inversion helpers."""

    #: config_remote: best rows per class inverted at the remote elements, and the repair budget for each subset.
    REMOTE_ROWS = 4
    REMOTE_STEPS = 60
    REMOTE_MAX_SUBSETS = 7

    def _solve_classes_batched(self, variants, elements, kw, mode: str, out_per_class: bool = False) -> EnsembleResult:
        """P2R config_states: every configuration class in ONE joint IK (plan Step 4.2).

        Each class is prepared exactly as its own solve would be (its fixes, its embed, assignments, beam seeds, Q1
        draws, class budget), then all classes' rows are stacked as blocks of one batch: per-row reference coordinates,
        per-row assignment index into the union of the classes' assignments, a per-block cull (the cull of each block's
        own solve). Each block is assembled as its own solve (per-class dedup, n_keep, window, mirror completion) and
        the classes are merged as before (`mode` "ez": `_merge_ez_results`; "stereo": `_merge_stereo_results`). A class
        whose solve layout differs from the first (theta widths, donors, atoms) is solved on its own, and that is logged.
        `variants`: StereoVariants ("stereo") or per-ligand E/Z fix lists ("ez").
        With `self._config_defer` set (bench seed packing) the prepared job is returned instead; `_config_run` then
        solves several seeds' jobs in one batch."""
        job = self._config_prepare(variants, elements, kw, mode, out_per_class)
        if getattr(self, "_config_defer", False):
            return job
        return self._config_run([job])[0]

    def _config_prepare(self, variants, elements, kw, mode, out_per_class) -> dict:
        """config_states: every class of one solve call prepared (rows, fixes, ring-state arrays) and the remote
        inversion plan (config_remote). A plain dict, solved by `_config_run`."""
        preps, failures = [], []
        inner_attr = "_ez_inner" if mode == "ez" else "_stereo_inner"
        t0 = __import__("time").perf_counter()
        if _e().EMBED_REFINE:
            self._prefetch_embeds(variants, elements, kw, mode)
        try:
            setattr(self, inner_attr, True)
            for vi, v in enumerate(variants):
                if mode != "ez" and not getattr(v, "feasible", True):
                    continue
                if mode == "ez":
                    for lig, f in zip(self.ligands, v):
                        lig.ez_fix = f
                else:
                    self._set_variant_fixes(v, elements)
                try:
                    self._capture_prep = True
                    try:
                        res = self.solve(**kw)
                    except _e()._PrepCapture as cap:
                        ctx, assignments, beam_width, n_ik_steps = cap.args[0]
                        res = None
                    finally:
                        self._capture_prep = False
                    if res is not None:                       # the class failed before its solve (embed etc.)
                        failures.append(res.failure or f"class {vi} returned no conformers")
                        continue
                    rows = self._prepare_rows(ctx, assignments, beam_width)
                    rows.update(ctx=ctx, assignments=assignments, n_ik_steps=n_ik_steps, vi=vi, v=v,
                                fixes=[(getattr(l, "ez_fix", None), getattr(l, "chiral_fix", None),
                                         bool(getattr(l, "_strict_stereo", False))) for l in self.ligands],
                                state=(getattr(ctx, "_row_state_E", None), getattr(ctx, "_row_state_tag", None),
                                       getattr(ctx, "_row_state_lab", None)))
                    preps.append(rows)
                except _e()._LigandEmbedFailure as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")
                    _e().log.info("config_states: class %d not embeddable (%s), skipped", vi, exc)
                except Exception as exc:
                    failures.append(f"internal error {type(exc).__name__}: {exc}")
                    _e().log.warning("config_states: class %d failed in preparation, skipped", vi, exc_info=True)
        finally:
            setattr(self, inner_attr, False)
            self._clear_variant_fixes()
        t_prep = __import__("time").perf_counter() - t0
        # An extra class whose prepared configuration repeats a kept one is NOT dropped: it is a second embed and a
        # second search of that configuration, and it found basins the first did not (LIPDEU, 3/3 seeds).
        job = dict(preps=preps, failures=failures, kw=kw, mode=mode, elements=elements, out_per_class=out_per_class,
                   inner_attr=inner_attr, t_prep=t_prep, remote_ops=[], config_job=True)
        remote_ops = []
        if mode == "stereo" and preps:
            base = preps[0]
            if _e().CONFIG_REMOTE and (getattr(self, "_stereo_class_info", None) or {}).get("remote_derived"):
                names = self._stereo_class_info["remote_derived"]
                try:
                    remote_ops, bad = plan(base["ctx"].mol, elements, names)
                except Exception as exc:
                    remote_ops, bad = [], list(names)
                    _e().log.warning("config_remote: inversion plan failed (%s)", exc)
                self._stereo_class_info["remote_not_derivable"] = bad
                if bad:
                    _e().log.warning("config_remote: %s not derivable locally (widen near); their classes are not offered", bad)
            job["remote_ops"] = remote_ops
        return job

    def _config_run(self, jobs) -> List[EnsembleResult]:
        """config_states: the rows of every class of every job (one job per seed in the bench, else one) as blocks of
        one joint IK (row cap `_config_max_rows`: whole classes per batch), each block assembled as its own solve,
        then each job merged on its own. Returns one EnsembleResult per job."""
        for j, job in enumerate(jobs):
            for r in job["preps"]:
                r["job"] = j
        allp = [r for job in jobs for r in job["preps"]]
        out_res = [None] * len(jobs)
        for j, job in enumerate(jobs):
            if not job["preps"]:
                out_res[j] = (self._merge_ez_results([], [], job["failures"], job["kw"]) if job["mode"] == "ez"
                              else self._merge_stereo_results([], job["failures"], job["kw"], job["out_per_class"]))
        if not allp:
            return out_res

        def layout(r):
            return ([tuple(t.shape[1:]) for t in r["thetas"]], [tuple(gpu.to_cpu(d).tolist()) for d in r["dpi"]],
                    len(r["ctx"].mol.atoms), int(r["ctx"].n_coord_samples), r["n_ik_steps"])
        base = allp[0]
        blocks = [r for r in allp if layout(r) == layout(base)]
        singles = [r for r in allp if layout(r) != layout(base)]
        if singles:
            _e().log.warning("config_states: %d of %d classes have another solve layout; solved on their own",
                        len(singles), len(allp))
        def run_group(blocks):
            """One joint IK over `blocks` (whole classes): ({job: [(res, vi)]}, {job: [variant]}, solve wall, rows)."""
            union, keys = [], {}
            for r in blocks:
                m = []
                for a in r["assignments"]:
                    k = tuple(np.asarray(a).tolist())
                    if k not in keys:
                        keys[k] = len(union)
                        union.append(a)
                    m.append(keys[k])
                r["cand_union"] = np.asarray(m, dtype=np.int64)[np.asarray(r["cand"])]
            ctx0 = base["ctx"]
            scorer_u = self._joint_scorer(ctx0, base["ge"], union)
            N = len(ctx0.mol.atoms)
            sizes = [int(r["thetas"][0].shape[0]) for r in blocks]
            thetas = [cp.concatenate([r["thetas"][li] for r in blocks], axis=0) for li in range(len(base["thetas"]))]
            X = cp.concatenate([r["X"] if r["X"].ndim == 3 else cp.broadcast_to(r["X"][None], (n, N, 3)) for r, n in zip(blocks, sizes)], axis=0)
            ideal = cp.concatenate([r["ideal"] for r in blocks], axis=0)
            cand = np.concatenate([r["cand_union"] for r in blocks])
            has_state = any(r["state"][0] is not None for r in blocks)
            ctx0._row_state_E = np.concatenate([r["state"][0] if r["state"][0] is not None else np.zeros(n) for r, n in zip(blocks, sizes)]) if has_state else None
            blk = cp.concatenate([cp.full((n,), i, dtype=cp.int64) for i, n in enumerate(sizes)])
            bw_total = int(sum(sizes)) // max(ctx0.n_coord_samples, 1)
            sym_u, q3_u, sym_rows = list(base["sym"] or []), list(base["q3sym"] or []), None
            if any([list(map(tuple, r["sym"] or [])), list(map(tuple, r["q3sym"] or []))]
                   != [list(map(tuple, base["sym"] or [])), list(map(tuple, base["q3sym"] or []))] for r in blocks):
                sym_rows = {"dist": {}, "q3": {}}
                for key, lst in (("dist", "sym"), ("q3", "q3sym")):
                    order = list(dict.fromkeys(tuple(g) for r in blocks for g in (r[lst] or [])))
                    for g in order:
                        sym_rows[key][g] = cp.concatenate([cp.full((n,), g in {tuple(x) for x in (r[lst] or [])}, dtype=cp.bool_)
                                                           for r, n in zip(blocks, sizes)])
                    if key == "dist":
                        sym_u = [list(g) for g in order]
                    else:
                        q3_u = [list(g) for g in order]
            from jacobik.ik.ik_solver import JointIKSolver
            JointIKSolver.cull_blocks, JointIKSolver.sym_rows = blk, sym_rows
            t1 = __import__("time").perf_counter()
            try:
                thetas_final, converged, scores, _ = self._run_joint_ik(
                    ctx0, base["fks"], scorer_u, base["ge"], X, thetas, base["dpi"], ideal, sym_u, q3_u, bw_total, base["n_ik_steps"], cand)
            finally:
                JointIKSolver.cull_blocks = JointIKSolver.sym_rows = None
            t_solve = __import__("time").perf_counter() - t1
            z = float(scorer_u._agg_norm()) if hasattr(scorer_u, "_agg_norm") else 0.0
            results, kept, off = {}, {}, 0
            for r, n in zip(blocks, sizes):
                sl = slice(off, off + n); off += n
                ctx, job = r["ctx"], jobs[r["job"]]
                failures, mode, elements, remote_ops = job["failures"], job["mode"], job["elements"], job["remote_ops"]
                for lig, (ez, ch, st) in zip(self.ligands, r["fixes"]):
                    lig.ez_fix, lig.chiral_fix, lig._strict_stereo = ez, ch, st
                try:
                    ctx._row_state_E, ctx._row_state_tag, ctx._row_state_lab = r["state"]
                    ctx.score_norm = z
                    th_k = [t[sl] for t in thetas_final]
                    confs = self._assemble_conformers(ctx, base["fks"], X[sl], th_k, scores[sl], converged[sl], r["bw_total"], r["cand"])
                    attach_solver_state(confs, th_k, None)
                except Exception as exc:
                    failures.append(f"class {r['vi']} assembly: {type(exc).__name__}: {exc}")
                    _e().log.warning("config_states: class %d assembly failed, skipped", r["vi"], exc_info=True)
                    continue
                finally:
                    self._clear_variant_fixes()
                res = EnsembleResult(conformers=confs, solve_trace={}, scorer=scorer_u)
                if len(r["assignments"]) > 1:
                    res.assignment_search = dict(n_candidates=len(r["assignments"]), candidate_assignments=[a.tolist() for a in r["assignments"]])
                if not confs:
                    failures.append(f"class {r['vi']} produced 0 conformers")
                    continue
                if mode != "ez":
                    self._label_class_rows(confs, r["vi"], r["v"], elements)
                results.setdefault(r["job"], []).append((res, r["vi"]))
                kept.setdefault(r["job"], []).append(r["v"])
                if remote_ops:
                    sub = None if sym_rows is None else {k: {g: m[sl] for g, m in d.items()} for k, d in sym_rows.items()}
                    der = self._derive_remote(ctx0, base, r, X[sl], th_k, scores[sl], ideal[sl], sym_u, q3_u, sub, remote_ops, scorer_u, z, elements)
                    if der is not None:
                        results[r["job"]].append(der[:2]); kept[r["job"]].append(der[2])
            return results, kept, t_solve, int(sum(sizes))

        cap = getattr(self, "_config_max_rows", None)
        groups, cur, cur_n = [], [], 0
        for r in blocks:                                  # row cap (GPU memory): whole classes per group
            n = int(r["thetas"][0].shape[0])
            if cur and cap is not None and cur_n + n > cap:
                groups.append(cur); cur, cur_n = [], 0
            cur.append(r); cur_n += n
        if cur:
            groups.append(cur)
        results = {j: [] for j in range(len(jobs))}
        kept = {j: [] for j in range(len(jobs))}
        t_solve, n_rows = 0.0, 0

        def add(rs, ks):
            for j in rs:
                results[j] += rs[j]
                kept[j] += ks[j]
        for g in groups:
            try:
                rs, ks, ts, nr = run_group(g)
                add(rs, ks)
            except gpu.OOMError:
                if len(g) == 1:   # one class too large for a batch: its own solve, which degrades (engine._solve_degrading)
                    _e().log.warning("config_states: class %d alone is out of GPU memory; solved on its own", g[0]["vi"])
                    gpu.empty_cache()
                    singles.append(g[0])
                    continue
                _e().log.warning("config_states: %d-class batch out of GPU memory; classes solved one batch each", len(g))
                gpu.empty_cache()
                ts, nr = 0.0, 0
                for r in g:
                    a_, b_, c_, d_ = run_group([r])
                    add(a_, b_)
                    ts += c_; nr += d_
            t_solve += ts; n_rows += nr
        # classes with another layout: their own solve (logged above)
        for r in singles:
            job = jobs[r["job"]]
            v, mode, elements = r["v"], job["mode"], job["elements"]
            try:
                setattr(self, job["inner_attr"], True)
                if mode == "ez":
                    for lig, f in zip(self.ligands, v):
                        lig.ez_fix = f
                else:
                    self._set_variant_fixes(v, elements)
                res = self.solve(**job["kw"])
                if res.conformers:
                    if mode != "ez":
                        self._label_class_rows(res.conformers, r["vi"], v, elements)
                    results[r["job"]].append((res, r["vi"]))
                    kept[r["job"]].append(v)
                else:
                    job["failures"].append(res.failure or f"class {r['vi']} produced 0 conformers")
            except Exception as exc:
                job["failures"].append(f"class {r['vi']}: {type(exc).__name__}: {exc}")
            finally:
                setattr(self, job["inner_attr"], False)
                self._clear_variant_fixes()
        _e().log.info("config_states: %d classes of %d job(s) in %d batch(es) (%d rows), %d alone; solve %.1f s",
                 len(blocks), len(jobs), len(groups), n_rows, len(singles), t_solve)
        for j, job in enumerate(jobs):
            if out_res[j] is not None:
                continue
            info = dict(n_classes=len(job["preps"]), n_batched=sum(r["job"] == j for r in blocks),
                        n_single=sum(r["job"] == j for r in singles),
                        n_batches=len(groups), n_jobs_packed=len(jobs), rows=n_rows, t_prep_s=job["t_prep"],
                        t_solve_s=t_solve / len(jobs))
            self._config_states_info = info
            order = sorted(range(len(results[j])), key=lambda i: results[j][i][1])   # class order, as the sequential loop
            res_j = [results[j][i] for i in order]
            kept_j = [kept[j][i] for i in order]
            if job["mode"] == "ez":
                out = self._merge_ez_results([r for r, _ in res_j], kept_j, job["failures"], job["kw"])
            else:
                out = self._merge_stereo_results(res_j, job["failures"], job["kw"], job["out_per_class"])
            out.solve_trace = dict(out.solve_trace or {}, config_states=info, n_stereo_variants_solved=len(res_j))
            out_res[j] = out
        return out_res

    def _derive_remote(self, ctx0, base, r, X_b, th_b, sc_b, ideal_b, sym_u, q3_u, sym_rows_b, ops, scorer_u, z,
                       elements):
        """config_remote: the classes of the remote elements derived from one solved class (one block): its best
        REMOTE_ROWS rows, inverted at every non-empty subset of the remote elements (at most REMOTE_MAX_SUBSETS, logged),
        repaired by a REMOTE_STEPS joint IK that starts from the inverted coordinates (per-row reference, zero angles),
        rescored, assembled and labelled from geometry. Returns (EnsembleResult, vi, pseudo variant) or None; never
        raises."""
        import itertools
        from types import SimpleNamespace
        from jacobik.ik.ik_solver import JointIKSolver
        try:
            subsets = [s for k in range(1, len(ops) + 1) for s in itertools.combinations(range(len(ops)), k)]
            if len(subsets) > self.REMOTE_MAX_SUBSETS:
                _e().log.info("config_remote: %d inversion subsets, first %d used", len(subsets), self.REMOTE_MAX_SUBSETS)
                subsets = subsets[:self.REMOTE_MAX_SUBSETS]
            top = cp.argsort(-sc_b)[:self.REMOTE_ROWS]
            Xs = X_b[top]
            for fk, th in zip(base["fks"], th_b):
                Xs = fk.apply(th[top], Xs)
            Xn = gpu.to_cpu(Xs)
            new = []
            for s in subsets:
                Xi = Xn
                for oi in s:
                    Xi = apply(ops[oi], Xi)
                new.append(Xi)
            src = np.tile(gpu.to_cpu(top), len(subsets))
            src_t = cp.asarray(src, dtype=cp.int64)
            Xd = gpu.as_gpu(np.concatenate(new), X_b.dtype)
            # the repair starts AT the inverted rows: zero rigid-body columns, every joint at its measured dihedral
            # (joint columns are absolute dihedrals), angle-bend columns zero; checked: FK(theta0, X_inv) = X_inv
            from jacobik.ik.fk import dihedral_axis_from_quad
            th0 = []
            for fk, t in zip(base["fks"], th_b):
                t0 = cp.zeros_like(t[src_t])
                for k in range(fk.n_joints):
                    t0[:, 6 + k] = dihedral_axis_from_quad(Xd[:, fk._quad_idx[k]])[0]
                th0.append(t0)
            Xc = Xd
            for fk, t0 in zip(base["fks"], th0):
                Xc = fk.apply(t0, Xc)
            err = float(cp.abs(Xc - Xd).max())
            if err > 1e-3:
                _e().log.warning("config_remote: FK(theta0, X_inv) != X_inv (%.2e); no derived rows", err)
                return None
            st = [None if a is None else np.asarray(a)[src] for a in r["state"]]
            cand_u = np.asarray(r["cand_union"])[src]
            ctx0._row_state_E = st[0]
            JointIKSolver.sym_rows = None if sym_rows_b is None else {
                k: {g: m[src_t] for g, m in d.items()} for k, d in sym_rows_b.items()}
            try:
                th_f, conv, sc, _ = self._run_joint_ik(
                    ctx0, base["fks"], scorer_u, base["ge"], Xd, th0, base["dpi"], ideal_b[src_t],
                    sym_u, q3_u, max(len(src) // max(ctx0.n_coord_samples, 1), 1), self.REMOTE_STEPS, cand_u)
            finally:
                JointIKSolver.sym_rows = None
            ctx = r["ctx"]
            for lig, (ez, ch, stf) in zip(self.ligands, r["fixes"]):
                lig.ez_fix, lig.chiral_fix, lig._strict_stereo = ez, ch, stf
            try:
                ctx._row_state_E, ctx._row_state_tag, ctx._row_state_lab = st
                ctx.score_norm = z
                confs = self._assemble_conformers(ctx, base["fks"], Xd, th_f, sc, conv, r["bw_total"],
                                                  np.asarray(r["cand"])[src])
                attach_solver_state(confs, th_f, None)
            finally:
                self._clear_variant_fixes()
            if not confs:
                return None
            names = "+".join(sorted({ops[i]["name"] for s in subsets for i in s}))
            v = SimpleNamespace(production=False, class_id=f"{r['v'].class_id}|derived:{names}")
            self._label_class_rows(confs, r["vi"], v, elements)
            for c in confs:
                c.config_derived = True
            _e().log.info("config_remote: class %d: %d derived rows kept of %d", r["vi"], len(confs), len(src))
            return EnsembleResult(conformers=confs, solve_trace={}, scorer=scorer_u), r["vi"] + 0.5, v
        except Exception as exc:
            _e().log.warning("config_remote: derivation failed for class %d (%s); no derived rows", r["vi"], exc,
                        exc_info=True)
            return None
