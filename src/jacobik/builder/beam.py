"""Initial row seeding and torsion beam search mixins for ComplexBuilder."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from jacobik.builder.result import EnsembleResult, attach_solver_state
from jacobik.core import gpu
from jacobik.stats.ladders import OS_UNKNOWN

cp = gpu.cp


def _e():
    from jacobik.builder import engine

    return engine


def _wrap(a: np.ndarray) -> np.ndarray:
    """wrap radians to [-pi, pi)"""
    return np.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def _wrap_gpu(a: cp.ndarray) -> cp.ndarray:
    """wrap radians to [-pi, pi) on GPU"""
    return cp.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def _diverse_ranked(ang, score, k: int, good_max: Optional[float]):
    """
    Greedy best-first selection of up to ``k`` conformers whose ring dihedrals
    differ by >= 0.6 rad RMS from every already-kept one.  With ``good_max``
    only conformers scoring below it are considered.  Returns
    (kept indices, kept indices that satisfy good_max / all kept if None).
    """
    ang_np = np.asarray(gpu.to_cpu(ang), dtype=np.float64)
    score_np = np.asarray(gpu.to_cpu(score), dtype=np.float64)
    order = np.argsort(score_np, kind="stable").tolist()
    keep: List[int] = []
    for c in order:
        if len(keep) >= k:
            break
        if good_max is not None and float(score_np[c]) >= good_max:
            break
        if keep:
            dd = np.remainder(ang_np[keep] - ang_np[c] + math.pi, 2.0 * math.pi) - math.pi
            if float(np.sqrt((dd * dd).mean(axis=-1)).min()) < 0.6:
                continue
        keep.append(c)
    return keep, keep


def _lone_pair_convergence(P, donors: List[int], nbrs: List[List[int]]) -> np.ndarray:
    """
    (C,) convergence score of sp3-donor lone pairs for conformers P (C, n, 3):
    RMS distance of the rays (donor -> away from its neighbours' centroid) from
    their least-squares common point, plus the total backward-pointing length.
    Low = all lone pairs aim at one point (a metal-binding conformation).
    """
    P_np = np.asarray(gpu.to_cpu(P), dtype=np.float64)
    N = P_np[:, donors]
    m = np.stack([P_np[:, n].mean(axis=1) for n in nbrs], axis=1)
    diff_nm = N - m
    nrm = np.linalg.norm(diff_nm, axis=-1, keepdims=True)
    l = diff_nm / np.maximum(nrm, 1e-12)
    I3 = np.eye(3, dtype=np.float64)
    Pm = I3 - l[..., :, None] * l[..., None, :]  # (C, D, 3, 3)
    A = Pm.sum(axis=1) + 1e-4 * I3
    b = np.einsum("cdij,cdj->ci", Pm, N)
    pt = np.linalg.solve(A, b[..., None])[..., 0]
    diff = pt[:, None] - N
    res = np.linalg.norm(np.einsum("cdij,cdj->cdi", Pm, diff), axis=-1)
    back = np.maximum(-(l * diff).sum(axis=-1), 0.0).sum(axis=1)
    return np.sqrt((res * res).mean(axis=1)) + back


def _donor_face_convergence(P, kappa: List[int], kappa_nbrs: List[List[int]],
                            faces: List[List[int]], face_nbrs: List[List[int]]) -> np.ndarray:
    """
    (C,) metal-binding score of ring conformers P (C, n, 3) of any macrocycle (profile field
    macro_pool_rank): every kappa donor gives a ray (donor -> away from its neighbours' centroid, backward length
    penalised as in _lone_pair_convergence), every haptic face a line through its centroid along the face normal.
    """
    P_np = np.asarray(gpu.to_cpu(P), dtype=np.float64)
    origins, dirs, directed = [], [], []
    for d, nb in zip(kappa, kappa_nbrs):
        o = P_np[:, d]
        origins.append(o)
        vec = o - P_np[:, nb].mean(axis=1)
        dirs.append(vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-12))
        directed.append(True)
    for g, nb in zip(faces, face_nbrs):
        Q = P_np[:, list(g) + list(nb)]
        Qc = Q - Q.mean(axis=1, keepdims=True)
        _, V = np.linalg.eigh(np.swapaxes(Qc, 1, 2) @ Qc)
        origins.append(P_np[:, list(g)].mean(axis=1))
        dirs.append(V[..., 0])
        directed.append(False)
    N = np.stack(origins, axis=1)
    l = np.stack(dirs, axis=1)
    I3 = np.eye(3, dtype=np.float64)
    Pm = I3 - l[..., :, None] * l[..., None, :]
    A = Pm.sum(axis=1) + 1e-4 * I3
    b = np.einsum("cdij,cdj->ci", Pm, N)
    pt = np.linalg.solve(A, b[..., None])[..., 0]
    diff = pt[:, None] - N
    res = np.linalg.norm(np.einsum("cdij,cdj->cdi", Pm, diff), axis=-1)
    dmask = np.asarray(directed, dtype=np.float64)
    back = (np.maximum(-(l * diff).sum(axis=-1), 0.0) * dmask).sum(axis=1)
    return np.sqrt((res * res).mean(axis=1)) + back



class _RowsMixin:
    """Seed row preparation, macrocycle pool construction, and joint Scorer creation."""

    def _prepare_rows(self, ctx, assignments, beam_width) -> dict:
        """Everything of `_solve_for_assignments` before the joint IK: beam search, start rows, class budget."""
        do_out_per_class = getattr(self, "out_per_class", None)
        if do_out_per_class is None:
            do_out_per_class = _e().OUT_PER_CLASS
        if do_out_per_class:
            k_out = _e().OUT_PER_CLASS_K if _e().OUT_PER_CLASS_K is not None else 8
            min_bw = (k_out + max(ctx.n_coord_samples, 1) - 1) // max(ctx.n_coord_samples, 1)
            if beam_width < min_bw:
                beam_width = min_bw

        symmetric_groups = ctx.symmetric_donor_groups
        # Q3 approach-angle ties only apply within one ligand instance; the
        # M-D bond-length tie uses the unrestricted groups.
        q3_symmetric_groups = [
            g for g in symmetric_groups if len({ctx.all_donor_groups[d][0] for d in g}) == 1
        ]
        if ctx.debug:
            print(f"\n[DEBUG] Symmetric donor groups: {symmetric_groups}")
            print(f"[DEBUG] Q3 (within-ligand) groups: {q3_symmetric_groups}")

        (
            fks,
            all_theta_inits,
            all_donor_pos_idxs,
            ge_joint,
            scorer_joint,
            X_full,
            ideal_directions_batched,
            candidate_idx_per_b,
        ) = self._beam_search_all_candidates(ctx, assignments, beam_width)
        beam_width_total = beam_width * len(assignments)
        keep = None
        ctx.early_cull = None
        if _e().CLASS_BUDGET == "split":
            keep = self._class_budget_rows(ctx, assignments, beam_width)
        elif _e().CLASS_BUDGET == "early_cull":
            n_cls = self._n_topological_classes(ctx, assignments)
            if n_cls is not None and n_cls < len(assignments):
                ctx.early_cull = (_e().EARLY_CULL_AT, n_cls / len(assignments))
        if keep is not None:
            # One row budget per topological class (see _class_budget_rows): subset every per-row array.
            keep_g = cp.asarray(keep, dtype=cp.int64)
            all_theta_inits = [th[keep_g] for th in all_theta_inits]
            if X_full.ndim == 3:
                X_full = X_full[keep_g]
            ideal_directions_batched = ideal_directions_batched[keep_g]
            candidate_idx_per_b = np.asarray(candidate_idx_per_b)[keep]
            for _a in ("_row_state_E", "_row_state_tag", "_row_state_lab"):
                if getattr(ctx, _a, None) is not None:
                    setattr(ctx, _a, getattr(ctx, _a)[keep])
            beam_width_total = len(keep) // max(ctx.n_coord_samples, 1)
        return dict(fks=fks, thetas=all_theta_inits, dpi=all_donor_pos_idxs, ge=ge_joint, scorer=scorer_joint,
                    X=X_full, ideal=ideal_directions_batched, cand=candidate_idx_per_b, bw_total=beam_width_total,
                    sym=symmetric_groups, q3sym=q3_symmetric_groups)

    def _finish_rows(self, ctx, assignments, rows, thetas_final, converged, scores, solve_trace) -> EnsembleResult:
        """Everything of `_solve_for_assignments` after the joint IK: conformer assembly and the result."""
        scorer_joint = rows["scorer"]
        ctx.score_norm = float(scorer_joint._agg_norm()) if hasattr(scorer_joint, "_agg_norm") else 0.0
        conformers = self._assemble_conformers(
            ctx,
            rows["fks"],
            rows["X"],
            thetas_final,
            scores,
            converged,
            rows["bw_total"],
            rows["cand"],
        )
        attach_solver_state(conformers, thetas_final, getattr(self, "_last_bend_prior", None))  # S2b capture
        result = EnsembleResult(conformers=conformers, solve_trace=solve_trace, scorer=scorer_joint)
        if len(assignments) > 1:
            result.assignment_search = dict(
                n_candidates=len(assignments),
                candidate_assignments=[a.tolist() for a in assignments],
            )
        return result

    def _n_topological_classes(self, ctx, assignments):
        """Number of graph-level classes among the candidates (None if not applicable)."""
        if len(assignments) < 2:
            return None
        from jacobik.ik.site_enumeration import topological_class_keys

        try:
            return len(set(topological_class_keys(assignments, ctx.symmetric_donor_groups, self.geometry)))
        except Exception as exc:
            _e().log.warning("class count skipped (%s)", exc)
            return None

    def _class_budget_rows(self, ctx, assignments, beam_width):
        """Rows to keep so that every topological class gets the row budget of ONE candidate.

        dedup_exact keeps assignments apart that differ only by swapping topologically equivalent donors within a
        ligand (or by a reflection), because a fixed embed can reach only some of them. They are one class at the
        graph level, so they share one candidate's budget of n_coord_samples x beam_width rows: the class's members
        take the beam seeds in turn (member j of s gets seeds j, j+s, ...; beam seeds are shared by all candidates),
        per coordinate sample. None = keep all."""
        if len(assignments) < 2 or not ctx.symmetric_donor_groups:
            return None
        from jacobik.ik.site_enumeration import topological_class_keys

        try:
            keys = topological_class_keys(assignments, ctx.symmetric_donor_groups, self.geometry)
        except Exception as exc:  # unknown geometry label etc.
            _e().log.warning("class budget skipped (%s)", exc)
            return None
        size = {k: keys.count(k) for k in set(keys)}
        if max(size.values()) == 1:
            return None
        M, W = ctx.n_coord_samples, beam_width
        member = {}
        keep = []
        do_opc = getattr(self, "out_per_class", None)
        if do_opc is None:
            do_opc = _e().OUT_PER_CLASS
        k_min = _e().OUT_PER_CLASS_K if _e().OUT_PER_CLASS_K is not None else 8
        min_seeds_per_cand = max(1, (k_min + max(M, 1) - 1) // max(M, 1)) if do_opc else 1
        for c, k in enumerate(keys):
            j = member[k] = member.get(k, -1) + 1  # this candidate's index within its class
            s_k = size[k]
            # Beam seeds are shared by all candidates: interleave them (member j gets seeds j, j+s, j+2s, ...), so the
            # class as a whole still covers all W seeds instead of every member repeating the top W/s.
            ws = list(range(j % W, W, s_k)) if s_k <= W else [j % W]
            if do_opc and len(ws) < min_seeds_per_cand:
                extra = [w for w in range(W) if w not in ws]
                ws.extend(extra[: min_seeds_per_cand - len(ws)])
                ws.sort()
            for m in range(M):
                base = (c * M + m) * W
                keep.extend(base + w for w in ws)
        return np.asarray(keep, dtype=np.int64)

    def _ideal_directions(self, ctx, assignment):
        """(n_donors, 3) float32 GPU array of the ideal site direction of each donor."""
        ideal_directions_full = np.stack(
            [ctx.ideal[assignment[di]] for di in range(ctx.n_donors)], axis=0
        )
        return gpu.as_gpu(ideal_directions_full, cp.float32)

    def _macro_pool(self, ctx, li, fk, B, template=None):
        """
        Per-row ring conformers for a macrocyclic ligand.

        Returns ``(angles, coords)``: (B, n_macro) ring-joint dihedrals and
        (B, n_lig_atoms, 3) ligand coordinates (ligand-local atom order) of
        each row's conformer, so rows start in different ring folds with
        exactly closed rings.

        Conformers come from ETKDG embedded in batches and ranked by how well
        the ring sp3-nitrogen donor lone pairs converge on one point (a cheap
        CPU proxy that rejects inverted ring nitrogens, which torsions cannot
        repair).  Embedding stops as soon as ``_POOL_TARGET`` mutually distinct
        good folds are found; ``ctx.macro_pool`` caps the number embedded.
        Only the kept conformers are MMFF-relaxed, and the kept set is cached
        per ligand.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        rd0 = ctx.rdmols[li]
        lig = ctx.mol.ligands[li]
        first = fk._lig_first
        ref = np.asarray(rd0.GetConformer().GetPositions())
        if not _e().MACRO_TEMPLATE:
            template = None
        tkey = None if template is None else (tuple(map(tuple, template[0])), tuple(np.round(template[1], 2)))
        key = (Chem.MolToSmiles(rd0), tuple(sorted(lig.donor_atoms)), int(ctx.macro_pool), ref.round(3).tobytes(), tkey)
        hit = _e()._MACRO_POOL_CACHE.get(key)
        if hit is None:
            hit = self._build_macro_pool(ctx, rd0, lig, fk, first, ref, template)
            _e()._MACRO_POOL_CACHE[key] = hit
        P = gpu.as_gpu(hit, cp.float32)
        ang = fk.macro_seed_angles(P)  # (K, n_macro)
        K = P.shape[0]
        rows = cp.asarray(
            ctx.np_rng.permutation(np.arange(B) % K), dtype=cp.int64
        )
        return ang[rows], P[rows]

    @staticmethod
    def _haptic_site_spread(P, faces, rd0) -> float:
        """Smallest (over face sides) largest distance between the metal sites of the haptic faces of one fold: a site
        is _HAPTIC_SITE_R off the face centroid along the face normal (plane of the face atoms and their heavy
        neighbours)."""
        import itertools

        sites = []
        for g in faces:
            gs = set(g)
            nb = {n.GetIdx() for a in g for n in rd0.GetAtomWithIdx(a).GetNeighbors()
                  if n.GetIdx() not in gs and n.GetAtomicNum() > 1}
            Q = P[list(g) + sorted(nb)]
            c = P[list(g)].mean(0)
            nrm = np.linalg.svd(Q - Q.mean(0))[2][2]
            sites.append((c + _e()._HAPTIC_SITE_R * nrm, c - _e()._HAPTIC_SITE_R * nrm))
        best = np.inf
        for pick in itertools.product((0, 1), repeat=len(sites)):
            S = np.array([s[k] for s, k in zip(sites, pick)])
            best = min(best, float(np.max(np.linalg.norm(S[:, None] - S[None], axis=-1))))
        return best

    @staticmethod
    def _build_macro_pool(ctx, rd0, lig, fk, first, ref, template=None):
        """Embed, rank, diversify and MMFF-relax ring conformers -> (K, n, 3) CPU tensor.

        With ``template`` = (local donor groups, M-D radii) (profile field macro_template) the conformers are embedded
        around a virtual metal (embed.templated_conformers) and kept in template order, diversified by ring
        dihedrals; the free-ligand ETKDG pool is the fallback when that embed yields nothing."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        if template is not None:
            from jacobik.builder.embed import templated_conformers

            confs = templated_conformers(rd0, template[0], template[1], max(int(ctx.macro_pool), 1), seed=7)
            if confs:
                P = np.stack([ref] + confs).astype(np.float32)
                order = np.concatenate([np.arange(1, P.shape[0]), np.zeros(1, dtype=np.int64)])
                rank = np.empty(P.shape[0], dtype=np.float32)
                rank[order] = np.arange(P.shape[0], dtype=np.float32)  # template order first, the embed last
                keep = _e()._diverse_ranked(fk.macro_seed_angles(gpu.as_gpu(P, cp.float32)), rank, _e()._POOL_TARGET, None)[0]
                return P[keep]

        ring_atoms = {a for ring in lig.macrocycles for a in ring}
        sp3 = [
            d - first
            for d in lig.donor_atoms
            if d in ring_atoms
            and rd0.GetAtomWithIdx(d - first).GetAtomicNum() == 7
            and rd0.GetAtomWithIdx(d - first).GetDegree() == 3
        ]
        nbrs = [[n.GetIdx() for n in rd0.GetAtomWithIdx(d).GetNeighbors()] for d in sp3]
        cap = max(int(ctx.macro_pool), 1)
        use_sel = len(sp3) >= 3
        score_fn = lambda P_: _e()._lone_pair_convergence(P_, sp3, nbrs)  # noqa: E731
        if _e().MACRO_POOL_RANK:
            # v1.0b macro_pool_rank: only rings with >= 3 sp3 ring nitrogens were ranked; crown ethers, thiacrowns and
            # mixed N/O/S rings took the first _POOL_TARGET free-ligand folds unranked. Rank every ring with >= 3 kappa
            # donors, any element, by its lone-pair convergence. Rings with a haptic segment (COD, CDT, N/olefin
            # rings) keep the production pool: ranking them by haptic face normals made COD worse (AWONOR, USAPOU,
            # XOJHIN, 3 seeds, 2026-11-18: AWONOR worse on every seed, XOJHIN on 2 of 3).
            if not any(set(h.donor_indices) & ring_atoms for h in lig.haptic_groups):
                kap = [d - first for d in lig.donor_atoms if d in ring_atoms]
                kap_n = [[n.GetIdx() for n in rd0.GetAtomWithIdx(d).GetNeighbors()] for d in kap]
                use_sel = len(kap) >= 3
                score_fn = lambda P_: _donor_face_convergence(P_, kap, kap_n, [], [])  # noqa: E731
        target = _e()._POOL_TARGET
        pool = [ref]
        score = None
        n_emb = 0
        batch_i = 0
        faces = [[a - first for a in h.donor_indices] for h in lig.haptic_groups
                 if len(h.donor_indices) >= 2 and set(h.donor_indices) & ring_atoms]
        if _e().MACRO_HAPTIC_FILTER and len(faces) >= 2 and not use_sel:
            # macro_haptic_filter: a ring with >= 2 haptic faces keeps only folds whose faces can all reach one metal
            # (free COD: the chair, 33 of 64 ETKDG folds, has its olefins facing opposite ways; the solve, moving
            # torsions with the ring closed, rarely inverts it: AWONOR rows 17-52). The input embed stays first.
            n_drop = 0
            try:
                while n_emb < cap and len(pool) - 1 < target:
                    n_b = min(_e()._POOL_BATCH, cap - n_emb)
                    ps = AllChem.ETKDGv3()
                    ps.randomSeed = 7 + batch_i
                    ps.numThreads = 0
                    ps.useRandomCoords = _e()._POOL_RANDOM_COORDS
                    rd = Chem.Mol(rd0)
                    ids = list(AllChem.EmbedMultipleConfs(rd, n_b, ps))
                    n_emb += n_b
                    batch_i += 1
                    if ids:  # the folds separate after MMFF (free COD 1.1 / 1.6 / 3.5 A), not on raw ETKDG geometry
                        AllChem.MMFFOptimizeMoleculeConfs(rd, numThreads=0, maxIters=200)
                    for i in ids:
                        Pc = np.asarray(rd.GetConformer(i).GetPositions())
                        if _RowsMixin._haptic_site_spread(Pc, faces, rd0) <= _e()._HAPTIC_SITE_TOL:
                            if len(pool) - 1 < target:
                                pool.append(Pc)
                        else:
                            n_drop += 1
                    if not ids:
                        break
            except Exception as exc:  # never fail the caller: the input embed alone is a valid pool
                _e().log.warning("macro_haptic_filter: %s; pool so far kept", exc)
            _e().log.info("macro_haptic_filter: %d folds kept, %d dropped (%d embedded)", len(pool) - 1, n_drop, n_emb)
            n_emb = cap                      # skip the unfiltered loop below
        while n_emb < cap:
            n_b = min(_e()._POOL_BATCH, cap - n_emb, target if not use_sel else _e()._POOL_BATCH)
            ps = AllChem.ETKDGv3()
            ps.randomSeed = 7 + batch_i
            ps.numThreads = 0
            ps.useRandomCoords = _e()._POOL_RANDOM_COORDS
            rd = Chem.Mol(rd0)
            ids = list(AllChem.EmbedMultipleConfs(rd, n_b, ps))
            pool += [np.asarray(rd.GetConformer(i).GetPositions()) for i in ids]
            n_emb += n_b
            batch_i += 1
            if not use_sel:
                break
            P = np.stack(pool).astype(np.float32)
            score = score_fn(P)
            keep = _e()._diverse_ranked(fk.macro_seed_angles(gpu.as_gpu(P, cp.float32)), score, target, _e()._POOL_GOOD)
            if len(keep[1]) >= target:  # enough distinct good folds
                break
        P = np.stack(pool).astype(np.float32)
        if use_sel:
            keep = _e()._diverse_ranked(fk.macro_seed_angles(gpu.as_gpu(P, cp.float32)), score, target, None)[0]
        else:
            keep = list(range(min(len(pool), target)))
        rk = Chem.Mol(rd0)
        rk.RemoveAllConformers()
        for c in keep:
            conf = Chem.Conformer(rd0.GetNumAtoms())
            for ai in range(rd0.GetNumAtoms()):
                conf.SetAtomPosition(ai, tuple(float(x) for x in pool[c][ai]))
            rk.AddConformer(conf, assignId=True)
        AllChem.MMFFOptimizeMoleculeConfs(rk, numThreads=0, maxIters=200)
        confs = [np.asarray(rk.GetConformer(c.GetId()).GetPositions(), float) for c in rk.GetConformers()]
        if _e().MACRO_POOL_REFINE == "mace" and confs:
            from jacobik.builder import mace_embed

            li_ = next(i for i, l in enumerate(ctx.mol.ligands) if l is lig)
            groups = [[int(a) - first for a in grp] for (l_, _, grp) in ctx.all_donor_groups if l_ == li_]
            confs = mace_embed.refine_many(rd0, confs, groups)
        if _e().MACRO_POOL_REFINE == "gfn2" and confs:
            # The rows' bond lengths and angles are those of their pool conformer. A free-ligand MMFF minimum of a
            # charged or crowded macrocycle opens its ring angles (DOTA4-: N-C-C 118 deg, GFN2 embed 108), so refine
            # every conformer as the embed itself is refined (GFN2 crude, donor-donor distances held), in parallel
            # single-threaded xtb runs. Any failure keeps that conformer's MMFF geometry (_gfn2_refine never raises).
            import os
            from concurrent.futures import ThreadPoolExecutor

            li_ = next(i for i, l in enumerate(ctx.mol.ligands) if l is lig)
            groups = [[int(a) - first for a in grp] for (l_, _, grp) in ctx.all_donor_groups if l_ == li_]
            with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 1, len(confs)))) as ex:
                confs = list(ex.map(lambda P: _e()._gfn2_refine(rd0, P, groups), confs))
        return np.stack(confs).astype(np.float32)

    def _joint_scorer(self, ctx, ge_joint, assignments):
        """The joint scorer of a solve over `assignments` (row_groups index this list)."""
        from jacobik.ik.scorer import Scorer
        return Scorer(
            ctx.mol,
            ge_joint,
            ctx.stats,
            device=ctx.device,
            quantity_weights=ctx.quantity_weights,
            assignments=assignments,
            donor_groups=[grp for (_, _, grp) in ctx.all_donor_groups],
            ideal=ctx.ideal,
            orientation_model=ctx.orientation_model,
            angle_bend_penalty=ctx.angle_bend_penalty,
        )


class _BeamMixin:
    """Per-ligand torsion beam search over candidate donor-to-site assignments."""

    def _beam_search_all_candidates(self, ctx, assignments, beam_width):
        """Build coordination-sphere targets, per-ligand FK, beam-search torsion seeds, and Kabsch base DOFs
        batched over all candidate assignments (candidates -> coord samples -> beam width)."""
        import time
        from jacobik.ik.beam import BeamSearch
        from jacobik.ik.fk import ForwardKinematics
        from jacobik.ik.geometry import GeometryExtractor
        from jacobik.ik.ik_solver import kabsch_init
        from jacobik.topology.decompose import decompose_ligand

        _c2_beam_wall, mol, debug, n_candidates = 0.0, ctx.mol, ctx.debug, len(assignments)
        targets_global_per_cand: List[np.ndarray] = []
        ideal_directions_per_cand: List[cp.ndarray] = []

        for assignment in assignments:
            targets_global = np.zeros((ctx.n_donors, ctx.n_coord_samples, 3), dtype=float)
            for di, (li, gi, grp) in enumerate(ctx.all_donor_groups):
                lbl = mol.donor_labels.get(grp[0])
                subclass, family = (lbl.subclass, lbl.family) if lbl else ("element_any", "element_any")
                site_vec = ctx.ideal[assignment[di]]
                trans_p = "none"
                if len(ctx.all_donor_groups) > 2 and self.geometry not in ("linear", "other"):
                    for dj, (_, _, grpj) in enumerate(ctx.all_donor_groups):
                        if dj != di and float(np.dot(site_vec, ctx.ideal[assignment[dj]])) < -0.766:
                            lblj = mol.donor_labels.get(grpj[0])
                            trans_p = lblj.subclass if lblj else "element_any"
                            break
                q1_key = (self.metal, self.spin, OS_UNKNOWN if self.oxidation_state is None else self.oxidation_state,
                          subclass, self.geometry, trans_p)
                sampled_dists = _e()._q1_sample_distances(ctx.stats, q1_key, family, ctx.n_coord_samples, ctx.np_rng)
                targets_global[di] = site_vec[None, :] * sampled_dists[:, None]
                if debug and n_candidates == 1:
                    print(f"[DEBUG] donor {di} (lig={li} grp={gi}) subclass={subclass!r} trans={trans_p!r} "
                          f"M-D mean={float(np.mean(sampled_dists)):.3f} [{float(np.min(sampled_dists)):.3f}, "
                          f"{float(np.max(sampled_dists)):.3f}] Å")
            targets_global_per_cand.append(targets_global)
            ideal_directions_per_cand.append(self._ideal_directions(ctx, assignment))

        for lig_obj in mol.ligands:
            decompose_ligand(mol, lig_obj)
            if ctx.macro_pool <= 0:
                lig_obj.macrocycles = []

        ge_joint = GeometryExtractor(mol, ctx.device)
        scorer_joint = self._joint_scorer(ctx, ge_joint, assignments)
        X_full = gpu.as_gpu(np.stack([a.position for a in mol.atoms]), cp.float32)

        W, M = beam_width, ctx.n_coord_samples
        B_per_cand = M * W
        B = B_per_cand * n_candidates
        candidate_idx_per_b = np.repeat(np.arange(n_candidates), B_per_cand)
        ideal_directions_batched = cp.concatenate(
            [cp.broadcast_to(d[None], (B_per_cand, d.shape[0], 3)) for d in ideal_directions_per_cand], axis=0
        )

        fks, X_rows = [], None
        row_state_E = np.zeros(B)
        row_state_tag = np.zeros(B, dtype=np.int64)
        _tag_mult = 1
        row_state_lab = np.full(B, "", dtype=object)
        all_theta_inits: List[cp.ndarray] = []
        all_donor_pos_idxs: List[cp.ndarray] = []

        for li, lig_obj in enumerate(mol.ligands):
            _e().log.info("Ligand %d/%d: FK + beam search (%d candidate assignment(s))...", li + 1, len(self.ligands), n_candidates)
            fk = ForwardKinematics(mol, lig_obj, ctx.device, angle_bends=ctx.angle_bends)
            fks.append(fk)
            lig_di = [di for di, (lig, _, _) in enumerate(ctx.all_donor_groups) if lig == li]
            n_d_lig = len(lig_di)
            donor_targets_np = np.stack(
                [np.stack([tg[di] for di in lig_di], axis=1) for tg in targets_global_per_cand], axis=0
            )

            J, J_full = fk.n_joints_base, fk.n_joints
            _e().log.info("  Ligand %d: J=%d joints (%d bend), beam_width=%d", li, J_full, J_full - J, beam_width)
            rows, offs = fk.q5_alignment(ge_joint, X_full)
            fk._q5_rows, fk._q5_offsets = rows, offs
            if J > 0 and scorer_joint._q5_means.shape[0] > 0:
                has = rows >= 0
                r0 = cp.maximum(rows, 0)
                q5_m = _wrap_gpu(scorer_joint._q5_means[r0] + offs[:, None])
                q5_k, q5_w = scorer_joint._q5_kappas[r0].copy(), scorer_joint._q5_weights[r0].copy()
                if not bool(has.all()):
                    miss = ~has
                    q5_m[miss], q5_m[miss, 0], q5_k[miss], q5_w[miss], q5_w[miss, 0] = 0.0, offs[miss], 1.0, 0.0, 1.0
            else:
                q5_m, q5_k, q5_w = scorer_joint._q5_means[:0], scorer_joint._q5_kappas[:0], scorer_joint._q5_weights[:0]

            beam = BeamSearch(fk, beam_width=beam_width, device=ctx.device)
            _c2_beam_start = time.perf_counter()
            _q6_widx = getattr(ge_joint, "_q6_wall_idx", None)
            q6_pairs = None
            if getattr(ge_joint, "n_q6", 0) > 0:
                q6_pairs = (ge_joint._q6_a, ge_joint._q6_b) if _q6_widx is None else (ge_joint._q6_a[_q6_widx], ge_joint._q6_b[_q6_widx])
            beam_torsions, _ = beam.search(
                X_ref=X_full, base_theta=cp.zeros((1, 6), dtype=cp.float32),
                q5_means=q5_m, q5_kappas=q5_k, q5_weights=q5_w,
                q6_mu=scorer_joint._q6_mu if _q6_widx is None else scorer_joint._q6_mu[_q6_widx],
                q6_sigma=scorer_joint._q6_sigma if _q6_widx is None else scorer_joint._q6_sigma[_q6_widx],
                q6_pairs=q6_pairs,
            )
            donor_targets_gpu = cp.ascontiguousarray(
                cp.broadcast_to(gpu.as_gpu(donor_targets_np, cp.float32)[:, :, None, :, :], (n_candidates, M, W, n_d_lig, 3)).reshape(B, n_d_lig, 3)
            )
            beam_torsions_B = cp.ascontiguousarray(cp.broadcast_to(beam_torsions[None, None, :, :], (n_candidates, M, W, J)).reshape(B, J))
            assert beam_torsions_B.shape == (B, J), "C-2 beam changed the seed-row budget"
            gpu.sync()
            _c2_beam_wall += time.perf_counter() - _c2_beam_start

            n_bend = fk.n_bend_joints
            if n_bend > 0:
                bend_seed = fk.bend_seed_angles(X_full)
                noise = cp.asarray((ctx.np_rng.random((B, n_bend)) - 0.5) * 2.0 * math.pi, dtype=cp.float32)
                beam_torsions_B = cp.concatenate([beam_torsions_B, cp.broadcast_to(bend_seed[None, :], (B, n_bend)) + noise], axis=-1)

            n_macro = fk.n_macro_joints
            if n_macro > 0:
                tmpl = None
                try:
                    grp_local = [[int(a) - fk._lig_first for a in ctx.all_donor_groups[di][2]] for di in lig_di]
                    tmpl = (grp_local, np.linalg.norm(donor_targets_np, axis=-1).mean(axis=(0, 1)))
                except Exception as exc:
                    _e().log.warning("macrocycle template radii unavailable (%s); free-ligand pool", exc)
                macro_seed, macro_coords = self._macro_pool(ctx, li, fk, B, tmpl)
                beam_torsions_B = cp.concatenate([beam_torsions_B, macro_seed], axis=-1)
                if X_rows is None:
                    X_rows = cp.broadcast_to(X_full[None, :, :], (B, X_full.shape[0], 3)).copy()
                lig_pos = fk._lig_pos_t
                d_ref = X_full[fk._donor_pos_t].mean(axis=0)
                d_row = macro_coords[:, fk._donor_pos_t - lig_pos[0], :].mean(axis=1)
                X_rows[:, lig_pos, :] = macro_coords + (d_ref - d_row)[:, None, :]

            ring_rows = False
            if _e().RING_STATES and n_macro == 0:
                try:
                    from jacobik.topology import rings as _rs
                    lig_pos = fk._lig_pos_t
                    lib = _rs.library(ctx.rdmols[li], gpu.to_cpu(X_full[lig_pos]).astype(float), [int(d) - fk._lig_first for d in lig_obj.donor_atoms])
                    if lib is not None:
                        st = _rs.allocate(lib["share"], B, ctx.np_rng)
                        if X_rows is None:
                            X_rows = cp.broadcast_to(X_full[None, :, :], (B, X_full.shape[0], 3)).copy()
                        X_rows[:, lig_pos, :] = gpu.as_gpu(lib["coords"], X_rows.dtype)[cp.asarray(st, dtype=cp.int64)]
                        row_state_E += lib["dE_kj"][st]
                        row_state_tag += st.astype(np.int64) * _tag_mult
                        row_state_lab = row_state_lab + np.asarray([f"L{li}:{lib['labels'][k]};" for k in st], dtype=object)
                        _tag_mult *= len(lib["dE_kj"])
                        ring_rows = True
                except Exception as exc:
                    _e().log.warning("ring_states rows failed for ligand %d (%s: %s); embed rings kept", li, type(exc).__name__, exc)

            donor_pos_idx_t = cp.asarray(lig_di, dtype=cp.int64)
            X_kabsch = X_full
            if n_macro > 0 or _e().START_KABSCH or ring_rows:
                X_kabsch = fk.apply(cp.concatenate([cp.zeros((B, 6), dtype=cp.float32), beam_torsions_B], axis=-1), X_rows if X_rows is not None else X_full)
            base_theta = kabsch_init(
                ge=ge_joint, X_ref=X_kabsch, donor_targets=donor_targets_gpu, donor_pos_idx=donor_pos_idx_t,
                diversify_orientation=True, diversity_scale=_e().START_KABSCH_DIVERSITY if _e().START_KABSCH else 1.0,
            )
            if _e().START_KABSCH and _e().START_KABSCH_PIVOT:
                try:
                    from jacobik.ik.kabsch import compensate_fk_pivot
                    base_theta = compensate_fk_pivot(base_theta, fk, X_rows if X_rows is not None else X_full, X_kabsch, ge_joint, donor_pos_idx_t)
                except Exception as exc:
                    _e().log.warning("C-2b pivot correction failed for ligand %d; retaining fitted pose: %s", li, exc)
            theta_init = cp.concatenate([base_theta, beam_torsions_B], axis=-1)
            if fk.n_angle_bends:
                theta_init = cp.concatenate([theta_init, cp.zeros((B, fk.n_angle_bends), dtype=theta_init.dtype)], axis=-1)
            if debug:
                print(f"[DEBUG] ligand {li} ({self.ligands[li].smiles}): J={J_full} ({n_bend} bend), donors={lig_di}, theta_init={tuple(theta_init.shape)}")
            all_theta_inits.append(theta_init)
            all_donor_pos_idxs.append(donor_pos_idx_t)

        self._c2_beam_wall_s = _c2_beam_wall
        has_rings = _e().RING_STATES and _tag_mult > 1
        ctx._row_state_E = row_state_E if has_rings else None
        ctx._row_state_tag = row_state_tag if has_rings else None
        ctx._row_state_lab = row_state_lab if has_rings else None
        return fks, all_theta_inits, all_donor_pos_idxs, ge_joint, scorer_joint, (X_full if X_rows is None else X_rows), ideal_directions_batched, candidate_idx_per_b
