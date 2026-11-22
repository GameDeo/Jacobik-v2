"""Initial row seeding and torsion beam search mixins for ComplexBuilder."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from jacobik.builder.result import EnsembleResult, attach_solver_state
from jacobik.stats.ladders import OS_UNKNOWN


def _e():
    from jacobik.builder import engine

    return engine


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """wrap radians to [-pi, pi)"""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _diverse_ranked(ang: torch.Tensor, score: torch.Tensor, k: int, good_max: Optional[float]):
    """
    Greedy best-first selection of up to ``k`` conformers whose ring dihedrals
    differ by >= 0.6 rad RMS from every already-kept one.  With ``good_max``
    only conformers scoring below it are considered.  Returns
    (kept indices, kept indices that satisfy good_max / all kept if None).
    """
    order = torch.argsort(score).tolist()
    keep: List[int] = []
    for c in order:
        if len(keep) >= k:
            break
        if good_max is not None and float(score[c]) >= good_max:
            break
        if keep:
            dd = torch.remainder(ang[keep] - ang[c] + math.pi, 2 * math.pi) - math.pi
            if float(dd.pow(2).mean(-1).sqrt().min()) < 0.6:
                continue
        keep.append(c)
    return keep, keep


def _lone_pair_convergence(P: torch.Tensor, donors: List[int], nbrs: List[List[int]]) -> torch.Tensor:
    """
    (C,) convergence score of sp3-donor lone pairs for conformers P (C, n, 3):
    RMS distance of the rays (donor -> away from its neighbours' centroid) from
    their least-squares common point, plus the total backward-pointing length.
    Low = all lone pairs aim at one point (a metal-binding conformation).
    """
    N = P[:, donors]
    m = torch.stack([P[:, n].mean(1) for n in nbrs], dim=1)
    l = F.normalize(N - m, dim=-1)
    I3 = torch.eye(3, dtype=P.dtype, device=P.device)
    Pm = I3 - l[..., :, None] * l[..., None, :]  # (C, D, 3, 3)
    A = Pm.sum(1) + 1e-4 * I3
    b = torch.einsum("cdij,cdj->ci", Pm, N)
    pt = torch.linalg.solve(A, b[..., None])[..., 0]
    diff = pt[:, None] - N
    res = torch.einsum("cdij,cdj->cdi", Pm, diff).norm(dim=-1)
    back = torch.relu(-(l * diff).sum(-1)).sum(1)
    return res.pow(2).mean(1).sqrt() + back


def _donor_face_convergence(P: torch.Tensor, kappa: List[int], kappa_nbrs: List[List[int]],
                            faces: List[List[int]], face_nbrs: List[List[int]]) -> torch.Tensor:
    """
    (C,) metal-binding score of ring conformers P (C, n, 3) of any macrocycle (profile field
    macro_pool_rank): every kappa donor gives a ray (donor -> away from its neighbours' centroid, backward length
    penalised as in _lone_pair_convergence), every haptic face a line through its centroid along the face normal (the
    pi lobes point both ways: a line, no backward term). The normal is that of the least-squares plane through the
    face atoms and their substituents (an eta2 C=C alone defines no plane). Score = RMS distance of the rays/lines from
    their least-squares common point + backward length: exactly _lone_pair_convergence when every donor is kappa
    (a ring with sp3 nitrogens only is ranked as before), extended to any donor element and to haptic faces. Low = a
    binding fold.
    """
    origins, dirs, directed = [], [], []
    for d, nb in zip(kappa, kappa_nbrs):
        o = P[:, d]
        origins.append(o)
        dirs.append(F.normalize(o - P[:, nb].mean(1), dim=-1))
        directed.append(True)
    for g, nb in zip(faces, face_nbrs):
        Q = P[:, list(g) + list(nb)]
        Qc = Q - Q.mean(1, keepdim=True)
        _, V = torch.linalg.eigh(Qc.transpose(1, 2) @ Qc)
        origins.append(P[:, list(g)].mean(1))
        dirs.append(V[..., 0])
        directed.append(False)
    N = torch.stack(origins, 1)
    l = torch.stack(dirs, 1)
    I3 = torch.eye(3, dtype=P.dtype, device=P.device)
    Pm = I3 - l[..., :, None] * l[..., None, :]
    A = Pm.sum(1) + 1e-4 * I3
    b = torch.einsum("cdij,cdj->ci", Pm, N)
    pt = torch.linalg.solve(A, b[..., None])[..., 0]
    diff = pt[:, None] - N
    res = torch.einsum("cdij,cdj->cdi", Pm, diff).norm(dim=-1)
    dmask = torch.tensor(directed, dtype=P.dtype, device=P.device)
    back = (torch.relu(-(l * diff).sum(-1)) * dmask).sum(1)
    return res.pow(2).mean(1).sqrt() + back



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
            # One row budget per topological class (see _class_budget_rows): subset every per-row tensor.
            keep_t = torch.as_tensor(keep, dtype=torch.long, device=ctx.device)
            all_theta_inits = [th[keep_t] for th in all_theta_inits]
            if X_full.dim() == 3:
                X_full = X_full[keep_t]
            ideal_directions_batched = ideal_directions_batched[keep_t]
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
        """(n_donors, 3) float32 tensor of the ideal site direction of each donor."""
        ideal_directions_full = np.stack(
            [ctx.ideal[assignment[di]] for di in range(ctx.n_donors)], axis=0
        )
        return torch.tensor(ideal_directions_full, dtype=torch.float32, device=ctx.device)

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
        P = hit.to(ctx.device)
        ang = fk.macro_seed_angles(P)  # (K, n_macro)
        K = P.shape[0]
        rows = torch.as_tensor(
            ctx.np_rng.permutation(np.arange(B) % K), dtype=torch.long, device=ctx.device
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
                P = torch.tensor(np.stack([ref] + confs), dtype=torch.float32)
                order = torch.cat([torch.arange(1, P.shape[0]), torch.zeros(1, dtype=torch.long)])
                rank = torch.empty(P.shape[0])
                rank[order] = torch.arange(P.shape[0], dtype=torch.float32)  # template order first, the embed last
                keep = _e()._diverse_ranked(fk.macro_seed_angles(P), rank, _e()._POOL_TARGET, None)[0]
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
            P = torch.tensor(np.stack(pool), dtype=torch.float32)
            score = score_fn(P)
            keep = _e()._diverse_ranked(fk.macro_seed_angles(P), score, target, _e()._POOL_GOOD)
            if len(keep[1]) >= target:  # enough distinct good folds
                break
        P = torch.tensor(np.stack(pool), dtype=torch.float32)
        if use_sel:
            keep = _e()._diverse_ranked(fk.macro_seed_angles(P), score, target, None)[0]
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
        return torch.tensor(np.stack(confs), dtype=torch.float32)

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
        """
        Build coordination-sphere targets, per-ligand FK, beam-search torsion
        seeds and Kabsch-initialised base DOFs, batched over every candidate
        assignment (candidates outermost, then coord samples, then beam width).

        Returns ``(fks, all_theta_inits, all_donor_pos_idxs, ge_joint,
        scorer_joint, X_full, ideal_directions_batched, candidate_idx_per_b)``
        where ``candidate_idx_per_b`` is a (B,) int array mapping each batch row
        to its index in ``assignments``.
        """
        from jacobik.ik.beam import BeamSearch
        from jacobik.ik.fk import ForwardKinematics
        from jacobik.ik.geometry import GeometryExtractor
        from jacobik.ik.ik_solver import kabsch_init
        from jacobik.ik.scorer import Scorer
        from jacobik.topology.decompose import decompose_ligand
        import time
        _c2_beam_wall = 0.0

        mol = ctx.mol
        debug = ctx.debug
        n_candidates = len(assignments)

        # Per-candidate coordination-sphere targets (Q1 GMM samples along each site).
        targets_global_per_cand: List[np.ndarray] = []  # each: (n_donors, M, 3)
        ideal_directions_per_cand: List[torch.Tensor] = []  # each: (n_donors, 3)

        for assignment in assignments:
            targets_global = np.zeros((ctx.n_donors, ctx.n_coord_samples, 3), dtype=float)

            for di, (li, gi, grp) in enumerate(ctx.all_donor_groups):
                lbl = mol.donor_labels.get(grp[0])
                subclass = lbl.subclass if lbl else "element_any"
                family = lbl.family if lbl else "element_any"
                site_vec = ctx.ideal[assignment[di]]  # (3,) unit vector

                # Trans partner subclass (first donor whose site is opposite).
                # Same gating and threshold as the extractor (key_builder._trans_partners):
                # no trans partner for CN<=2 or geometries without a trans axis.
                trans_p = "none"
                for dj, (_, _, grpj) in enumerate(ctx.all_donor_groups):
                    if dj == di or len(ctx.all_donor_groups) <= 2 or self.geometry in ("linear", "other"):
                        continue
                    dot = float(np.dot(site_vec, ctx.ideal[assignment[dj]]))
                    if dot < -0.766:
                        lblj = mol.donor_labels.get(grpj[0])
                        trans_p = lblj.subclass if lblj else "element_any"
                        break

                q1_key = (
                    self.metal,
                    self.spin,
                    OS_UNKNOWN if self.oxidation_state is None else self.oxidation_state,
                    subclass,
                    self.geometry,
                    trans_p,
                )
                sampled_dists = _e()._q1_sample_distances(
                    ctx.stats,
                    q1_key,
                    family,
                    ctx.n_coord_samples,
                    ctx.np_rng,
                )  # (M,)
                targets_global[di] = site_vec[None, :] * sampled_dists[:, None]

                if debug and n_candidates == 1:
                    print(
                        f"[DEBUG] donor {di} (lig={li} grp={gi}) subclass={subclass!r} "
                        f"trans={trans_p!r} M-D mean={float(np.mean(sampled_dists)):.3f} "
                        f"[{float(np.min(sampled_dists)):.3f}, "
                        f"{float(np.max(sampled_dists)):.3f}] Å"
                    )

            targets_global_per_cand.append(targets_global)
            ideal_directions_per_cand.append(self._ideal_directions(ctx, assignment))

        for lig_obj in mol.ligands:
            decompose_ligand(mol, lig_obj)
            if ctx.macro_pool <= 0:
                lig_obj.macrocycles = []

        # Shared (assignment-independent) geometry extractor and scorer.
        ge_joint = GeometryExtractor(mol, ctx.device)
        scorer_joint = self._joint_scorer(ctx, ge_joint, assignments)

        X_full = torch.tensor(
            np.stack([a.position for a in mol.atoms]),
            dtype=torch.float32,
            device=ctx.device,
        )  # (N_total, 3)

        W = beam_width
        M = ctx.n_coord_samples
        B_per_cand = M * W
        B = B_per_cand * n_candidates

        candidate_idx_per_b = np.repeat(np.arange(n_candidates), B_per_cand)

        ideal_directions_batched = torch.cat(
            [d.unsqueeze(0).expand(B_per_cand, -1, -1) for d in ideal_directions_per_cand],
            dim=0,
        )  # (B, n_donors, 3)

        fks: List = []
        X_rows = None  # (B, N, 3) per-row reference when a macrocycle is sampled
        row_state_E = np.zeros(B)  # P2R ring_states: per-row state energy (kJ/mol), summed over ligands
        row_state_tag = np.zeros(B, dtype=np.int64)  # ... and a per-row state code (0 = every ligand in its embed state)
        _tag_mult = 1
        row_state_lab = np.full(B, "", dtype=object)  # ... and its label per ligand ("L0:chair/twist;"), for the output
        all_theta_inits: List[torch.Tensor] = []
        all_donor_pos_idxs: List[torch.Tensor] = []
        fk_n_joints: List[int] = []  # base (torsion) joint count per ligand

        for li, lig_obj in enumerate(mol.ligands):
            _e().log.info(
                "Ligand %d/%d: FK + beam search (%d candidate assignment(s))...",
                li + 1,
                len(self.ligands),
                n_candidates,
            )

            fk = ForwardKinematics(mol, lig_obj, ctx.device, angle_bends=ctx.angle_bends)
            fks.append(fk)
            fk_n_joints.append(fk.n_joints_base)

            lig_di = [di for di, (lig, _, _) in enumerate(ctx.all_donor_groups) if lig == li]
            n_d_lig = len(lig_di)

            # (n_candidates, M, n_d_lig, 3)
            donor_targets_np = np.stack(
                [
                    np.stack([targets_global[di] for di in lig_di], axis=1)
                    for targets_global in targets_global_per_cand
                ],
                axis=0,
            )

            # Q5 statistics cover torsion joints only; bend joints (J_full - J)
            # are appended after beam search.
            J = fk.n_joints_base
            J_full = fk.n_joints
            _e().log.info(
                "  Ligand %d: J=%d joints (%d bend), beam_width=%d",
                li,
                J_full,
                J_full - J,
                beam_width,
            )
            ge_li = GeometryExtractor(mol, ctx.device)
            scorer_li = Scorer(
                mol,
                ge_li,
                ctx.stats,
                device=ctx.device,
                quantity_weights=ctx.quantity_weights,
                orientation_model="q3q4",  # the beam only reads Q5/Q6 tables; skip building S_orient here
            )
            # The scorer's Q5 tables span all ligands; slice this ligand's rows.
            # Q5 statistics of each FK joint's own bond, expressed in the FK joint's dihedral convention
            # (fk.q5_alignment); joints without a Q5 row get one mode at their current dihedral.
            rows, offs = fk.q5_alignment(ge_li, X_full)
            fk._q5_rows, fk._q5_offsets = rows, offs
            if J > 0 and scorer_li._q5_means.shape[0] > 0:
                has = rows >= 0
                r0 = rows.clamp(min=0)
                q5_m = _e()._wrap(scorer_li._q5_means[r0] + offs[:, None])
                q5_k = scorer_li._q5_kappas[r0].clone()
                q5_w = scorer_li._q5_weights[r0].clone()
                if not bool(has.all()):
                    miss = ~has
                    q5_m[miss] = 0.0
                    q5_m[miss, 0] = offs[miss]
                    q5_k[miss] = 1.0
                    q5_w[miss] = 0.0
                    q5_w[miss, 0] = 1.0
            else:
                q5_m = scorer_li._q5_means[:0]
                q5_k = scorer_li._q5_kappas[:0]
                q5_w = scorer_li._q5_weights[:0]
            beam = BeamSearch(fk, beam_width=beam_width, device=ctx.device)
            _c2_beam_start = time.perf_counter()
            # Torsion seeds are shared by all candidates.
            beam_torsions, _ = beam.search(
                X_ref=X_full,
                base_theta=torch.zeros(1, 6, device=ctx.device),
                q5_means=q5_m,
                q5_kappas=q5_k,
                q5_weights=q5_w,
                # donor-pair extension pairs (flag q6_donor_pairs) carry the repulsion, not the wall: the beam keeps the wall pairs only
                q6_mu=scorer_li._q6_mu if getattr(ge_li, "_q6_wall_idx", None) is None else scorer_li._q6_mu[ge_li._q6_wall_idx],
                q6_sigma=scorer_li._q6_sigma if getattr(ge_li, "_q6_wall_idx", None) is None else scorer_li._q6_sigma[ge_li._q6_wall_idx],
                q6_pairs=((ge_li._q6_a, ge_li._q6_b) if getattr(ge_li, "_q6_wall_idx", None) is None
                          else (ge_li._q6_a[ge_li._q6_wall_idx], ge_li._q6_b[ge_li._q6_wall_idx]))
                if getattr(ge_li, "n_q6", 0) > 0 else None,  # ARCHITECTURE.md section 3 (CHANGELOG 2026-09-28) (fixed)
            )  # (W, J)

            # Tile n_candidates x M x W -> B parallel seeds.
            donor_targets_torch = (
                torch.tensor(
                    donor_targets_np,
                    dtype=torch.float32,
                    device=ctx.device,
                )
                .unsqueeze(2)  # (n_candidates, M, 1, n_d_lig, 3)
                .expand(n_candidates, M, W, n_d_lig, 3)
                .reshape(B, n_d_lig, 3)
            )

            beam_torsions_B = (
                beam_torsions.unsqueeze(0)
                .unsqueeze(0)  # (1, 1, W, J)
                .expand(n_candidates, M, W, J)
                .reshape(B, J)
            )
            assert beam_torsions_B.shape == (B, J), "C-2 beam changed the seed-row budget"
            if ctx.device.type == "cuda":
                torch.cuda.synchronize(ctx.device)
            _c2_beam_wall += time.perf_counter()-_c2_beam_start

            # Bend joints: seeded at their current dihedral plus per-trial
            # random noise; refined by gradient descent only.
            n_bend = fk.n_bend_joints
            if n_bend > 0:
                bend_seed = fk.bend_seed_angles(X_full)  # (n_bend,)
                bend_seed_B = bend_seed.unsqueeze(0).expand(B, n_bend).clone()
                # drawn on the CPU generator and moved: the same seed gives the same starts on every device
                noise = (torch.rand(B, n_bend).to(ctx.device) - 0.5) * 2.0 * math.pi
                bend_seed_B = bend_seed_B + noise
                beam_torsions_B = torch.cat([beam_torsions_B, bend_seed_B], dim=-1)  # (B, J_full)

            # Macrocycle joints: seed from a pool of RDKit ring conformers
            # (the reference embedding first), so rows start in different folds.
            n_macro = fk.n_macro_joints
            if n_macro > 0:
                tmpl = None
                try:
                    grp_local = [[int(a) - fk._lig_first for a in ctx.all_donor_groups[di][2]] for di in lig_di]
                    radii = np.linalg.norm(donor_targets_np, axis=-1).mean(axis=(0, 1))
                    tmpl = (grp_local, radii)
                except Exception as exc:
                    _e().log.warning("macrocycle template radii unavailable (%s); free-ligand pool", exc)
                macro_seed, macro_coords = self._macro_pool(ctx, li, fk, B, tmpl)
                beam_torsions_B = torch.cat([beam_torsions_B, macro_seed], dim=-1)
                if X_rows is None:
                    X_rows = X_full.unsqueeze(0).expand(B, -1, -1).clone()
                # Keep each row's ligand at the reference donor centroid.
                lig_pos = fk._lig_pos_t
                d_ref = X_full[fk._donor_pos_t].mean(0)
                d_row = macro_coords[:, fk._donor_pos_t - lig_pos[0], :].mean(1)
                X_rows[:, lig_pos, :] = macro_coords + (d_ref - d_row)[:, None, :]

            # P2R ring_states: rows start in the ligand's ring-pucker states (state library, one per ligand), split
            # by state energy; the state energy enters the row score (_run_joint_ik). Nothing is forced: state 0 is
            # the embed. Any failure keeps the embed for every row.
            ring_rows = False
            if _e().RING_STATES and n_macro == 0:
                try:
                    from jacobik.topology import rings as _rs
                    lig_pos = fk._lig_pos_t
                    X_lig = X_full[lig_pos].detach().cpu().numpy().astype(float)
                    lib = _rs.library(ctx.rdmols[li], X_lig, [int(d) - fk._lig_first for d in lig_obj.donor_atoms])
                    if lib is not None:
                        st = _rs.allocate(lib["share"], B, ctx.np_rng)
                        if X_rows is None:
                            X_rows = X_full.unsqueeze(0).expand(B, -1, -1).clone()
                        coords = torch.as_tensor(lib["coords"], dtype=X_rows.dtype, device=ctx.device)
                        X_rows[:, lig_pos, :] = coords[torch.as_tensor(st, dtype=torch.long, device=ctx.device)]
                        row_state_E += lib["dE_kj"][st]
                        row_state_tag += st.astype(np.int64) * _tag_mult
                        row_state_lab = row_state_lab + np.asarray([f"L{li}:{lib['labels'][k]};" for k in st], dtype=object)
                        _tag_mult *= len(lib["dE_kj"])
                        ring_rows = True
                except Exception as exc:
                    _e().log.warning("ring_states rows failed for ligand %d (%s: %s); embed rings kept", li, type(exc).__name__, exc)

            # Group indices (not atom indices), so haptic centroids resolve.
            donor_pos_idx_t = torch.tensor(lig_di, dtype=torch.long, device=ctx.device)

            X_kabsch = X_full
            if n_macro > 0 or _e().START_KABSCH or ring_rows:
                # Fit the base pose to each row's own conformer with its torsion seeds.
                zero6 = torch.zeros(B, 6, device=ctx.device)
                X_kabsch = fk.apply(torch.cat([zero6, beam_torsions_B], dim=-1), X_rows if X_rows is not None else X_full)
            base_theta = kabsch_init(
                ge=ge_joint,
                X_ref=X_kabsch,
                donor_targets=donor_targets_torch,
                donor_pos_idx=donor_pos_idx_t,
                diversify_orientation=True,
                diversity_scale=_e().START_KABSCH_DIVERSITY if _e().START_KABSCH else 1.0,
            )  # (B, 6)
            if _e().START_KABSCH and _e().START_KABSCH_PIVOT:
                try:
                    from jacobik.ik.kabsch import compensate_fk_pivot
                    base_theta = compensate_fk_pivot(
                        base_theta, fk, X_rows if X_rows is not None else X_full,
                        X_kabsch, ge_joint, donor_pos_idx_t,
                    )
                except Exception as exc:
                    _e().log.warning("C-2b pivot correction failed for ligand %d; retaining fitted pose: %s", li, exc)
            theta_init = torch.cat([base_theta, beam_torsions_B], dim=-1)  # (B, 6+J)
            if fk.n_angle_bends:
                # Angle-bend columns start at 0 (the embed's angles); see ik/angle_bends.py.
                theta_init = torch.cat([theta_init, theta_init.new_zeros(B, fk.n_angle_bends)], dim=-1)

            if debug:
                print(
                    f"[DEBUG] ligand {li} ({self.ligands[li].smiles}): J={J_full} "
                    f"({n_bend} bend), donors={lig_di}, theta_init={tuple(theta_init.shape)}"
                )

            all_theta_inits.append(theta_init)
            all_donor_pos_idxs.append(donor_pos_idx_t)

        self._c2_beam_wall_s = _c2_beam_wall
        ctx._row_state_E = row_state_E if _e().RING_STATES and _tag_mult > 1 else None
        ctx._row_state_tag = row_state_tag if _e().RING_STATES and _tag_mult > 1 else None
        ctx._row_state_lab = row_state_lab if _e().RING_STATES and _tag_mult > 1 else None
        return (
            fks,
            all_theta_inits,
            all_donor_pos_idxs,
            ge_joint,
            scorer_joint,
            X_full if X_rows is None else X_rows,
            ideal_directions_batched,
            candidate_idx_per_b,
        )
