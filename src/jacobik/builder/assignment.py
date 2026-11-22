"""Donor-to-site graph assignment, symmetry enumeration, and multi-embed dispatch."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from jacobik.builder.result import ConformerResult, EnsembleResult
from jacobik.core import timing


def _e():
    from jacobik.builder import engine

    return engine


class _AssignmentMixin:
    """Shell construction, Hungarian baseline, feasibility repair, and symmetry enumeration."""

    def solve_multi_embed(
        self,
        embed_seeds: Tuple[int, ...] = tuple(range(8)),
        n_conformers: Optional[int] = None,
        **solve_kwargs,
    ) -> EnsembleResult:
        """
        Solve once per seed in ``embed_seeds`` -- each an independent RDKit
        ETKDG embedding of every ligand -- and merge the resulting
        conformers, re-ranked by score, keeping the best ``n_conformers``
        overall (default: everything).

        Why this exists: ``solve()``'s single ligand embedding (fixed seed
        42 unless ``embed_seed`` is passed) is fine whenever torsion
        optimisation can reach any relative donor geometry the target
        needs -- true for almost every ligand. It stops being true for a
        rigid multidentate ligand whose donor atoms sit on two or more
        rigid sub-structures joined by few or no rotatable bonds (a
        bis-chelate cage, say): there, the ONE sampled embedding
        permanently fixes the relative orientation of those pieces, and a
        bad draw makes good coordination geometrically impossible for
        EVERY donor-to-site assignment and however much IK optimisation is
        thrown at it -- typically visible as a collapsed metal-donor bond.
        Different ``embed_seeds`` sample different relative orientations;
        solving several and keeping the best conformers found is the fix,
        at the cost of ``len(embed_seeds)`` times the work. Each returned
        conformer's ``embed_idx`` records which seed (by position in
        ``embed_seeds``) produced it.

        ``solve_kwargs`` are passed through to every ``solve()`` call
        (``stats=`` in particular should be built once by the caller and
        passed here, not reloaded per seed).
        """
        all_conformers: List[ConformerResult] = []
        last_trace = None
        last_scorer = None
        last_assignment_search = None
        for ei, es in enumerate(embed_seeds):
            result = self.solve(embed_seed=es, n_conformers=n_conformers, **solve_kwargs)
            all_conformers.extend(replace(c, embed_idx=ei) for c in result.conformers)
            last_trace = result.solve_trace
            last_scorer = result.scorer
            last_assignment_search = result.assignment_search

        all_conformers.sort(key=lambda c: c.score, reverse=True)
        keep = len(all_conformers) if n_conformers is None else n_conformers
        kept = [replace(c, rank=rank) for rank, c in enumerate(all_conformers[:keep], start=1)]
        return EnsembleResult(
            conformers=kept,
            solve_trace=last_trace,
            scorer=last_scorer,
            assignment_search=last_assignment_search,
        )

    def _load_stats(self, constraints_db, stats, shannon=False):
        """Load the constraints DB unless a StatsLookup was supplied."""
        from jacobik.stats.stats import StatsLookup, load_db

        if stats is None:
            if constraints_db is None:
                from jacobik.core.env import default_db_path

                constraints_db = default_db_path()
                if constraints_db is None:
                    raise FileNotFoundError("no constraints DB found: pass constraints_db=..., set JACOBIK_DB, or place the "
                                            "production DB in data/db (see jacobik.core.env.default_db_path)")
            _e().log.info("Loading constraints DB: %s", constraints_db)
            db = load_db(constraints_db)
            stats = StatsLookup(db, shannon=shannon)
        else:
            _e().log.info("Using pre-built StatsLookup (skipping load_db)")
        return stats

    @timing.timed("embed")
    def _build_shell(self, fast_embed, debug, embed_seed=None, chelate_embed=False, embed=None):
        """Build the initial TMComplex shell from the ligand SMILES and classify donors."""
        from jacobik.topology.donor_classifier import classify

        _e().log.info("Building complex shell (%s %s)...", self.metal, self.geometry)
        mol, rdmols, donor_idx_full = _e()._build_complex_shell(
            metal_symbol=self.metal,
            charge=self.charge,
            spin=self.spin,
            geometry=self.geometry,
            ligand_inputs=self.ligands,
            fast_embed=fast_embed,
            embed_seed=embed_seed,
            chelate_embed=chelate_embed,
            embed=embed,
        )

        mol.oxidation_state = self.oxidation_state
        mol.oxidation_source = self.oxidation_source
        classify(mol, smiles_path=True)
        _e().log.info("Classification done: %d coord sites", len(mol.coord_sites))
        _e()._order_sites_like_donor_groups(mol, donor_idx_full)

        if debug:
            print("\n[DEBUG] Complex shell built")
            print(
                f"  metal={self.metal}  geometry={self.geometry}  "
                f"charge={self.charge}  spin={self.spin}"
            )
            print(f"  Total atoms : {len(mol.atoms)}")
            print(f"  Coord sites : {len(mol.coord_sites)}")
            for si, site in enumerate(mol.coord_sites):
                hap = "haptic" if site.is_haptic else "kappa"
                print(f"    site[{si}] {hap}  donor_indices={site.donor_indices}")

        if not mol.coord_sites:
            raise RuntimeError(
                "Donor classification produced no coordination sites. "
                "Check that donor indices point to actual donor atoms."
            )
        return mol, rdmols, donor_idx_full

    def _collect_donor_groups(self, mol, donor_idx_full):
        """Ideal site directions and all donor groups (ligand, group, atom
        indices) in order across ligands."""
        from jacobik.ik.targets import IDEAL_GEOMETRIES

        ideal = IDEAL_GEOMETRIES.get(self.geometry)
        if ideal is None:
            # "other" geometry — use identity directions
            cn = len(mol.coord_sites)
            ideal = np.eye(cn, 3, dtype=float)

        all_donor_groups: List[Tuple[int, int, List[int]]] = []
        for li in range(len(self.ligands)):
            for gi, grp in enumerate(donor_idx_full[li]):
                all_donor_groups.append((li, gi, grp))

        n_donors = len(all_donor_groups)
        return ideal, all_donor_groups, n_donors

    def _detect_chelate_pairs(self, mol, all_donor_groups, n_donors):
        """
        Same-ligand donor pairs within a few ligand-graph hops (chelate pairs),
        used by hungarian_assign to avoid placing a short chelate on distant
        sites. Also returns the heavy-atom ligand graph (None if it could not
        be built).
        """
        import networkx as nx

        from jacobik.topology.graph import build_heavy_ligand_graph

        _CHELATE_SHORT_MAX = 4  # ligand-graph bond hops
        chelate_pairs_global: List[Tuple[int, int]] = []
        heavy_graph = None
        try:
            heavy_graph = build_heavy_ligand_graph(mol.atoms, mol.bonds, mol.metal_idx)
            for _di in range(n_donors):
                _li, _gi, _grp = all_donor_groups[_di]
                _a0 = _grp[0]
                if not heavy_graph.has_node(_a0):
                    continue
                for _dj in range(_di + 1, n_donors):
                    _lj, _gj, _grpj = all_donor_groups[_dj]
                    if _lj != _li:
                        continue  # chelate pairs only within the same ligand
                    _b0 = _grpj[0]
                    if not heavy_graph.has_node(_b0):
                        continue
                    try:
                        _hops = nx.shortest_path_length(heavy_graph, _a0, _b0)
                    except nx.NetworkXNoPath:
                        continue
                    if _hops <= _CHELATE_SHORT_MAX:
                        chelate_pairs_global.append((_di, _dj))
        except Exception as exc:
            _e().log.warning("Chelate-pair detection failed (%s); skipping repair pass.", exc)
            chelate_pairs_global = []
        return chelate_pairs_global, heavy_graph

    def _build_hop_matrix(self, n_donors, all_donor_groups, heavy_graph):
        """Ligand-graph hop distances between donor atoms (-1 where no path),
        or None if the heavy-atom graph is unavailable (no reachability
        constraint then)."""
        import networkx as nx

        if heavy_graph is None:
            _e().log.warning("Hop matrix skipped: no heavy-atom ligand graph.")
            return None

        hop_matrix_full: Optional[np.ndarray] = np.full((n_donors, n_donors), -1, dtype=int)
        try:
            for _hi in range(n_donors):
                _ai = all_donor_groups[_hi][2][0]
                if not heavy_graph.has_node(_ai):
                    continue
                for _hj in range(_hi + 1, n_donors):
                    _aj = all_donor_groups[_hj][2][0]
                    if not heavy_graph.has_node(_aj):
                        continue
                    try:
                        _hops = nx.shortest_path_length(heavy_graph, _ai, _aj)
                    except nx.NetworkXNoPath:
                        continue
                    hop_matrix_full[_hi, _hj] = _hops
                    hop_matrix_full[_hj, _hi] = _hops
        except Exception as exc:
            _e().log.warning("Hop-matrix build failed (%s); no reachability constraint.", exc)
            hop_matrix_full = None
        return hop_matrix_full

    def _compute_baseline_assignment(
        self, mol, ideal, all_donor_groups, n_donors, chelate_pairs_global, heavy_graph=None
    ):
        """Baseline donor-to-site assignment: fully explicit, partially explicit
        (Hungarian on the free donors) or fully Hungarian; non-explicit results
        are then repaired if ligand-graph reachability rules them out."""
        from jacobik.ik.targets import hungarian_assign

        atom_map = {a.idx: a for a in mol.atoms}
        metal_pos = mol.atoms[0].position

        donor_dirs = np.zeros((n_donors, 3), dtype=float)
        for di, (_li, _gi, grp) in enumerate(all_donor_groups):
            centroid = np.mean([atom_map[i].position for i in grp], axis=0)
            vec = centroid - metal_pos
            norm = np.linalg.norm(vec)
            donor_dirs[di] = vec / norm if norm > 1e-8 else ideal[di % len(ideal)]

        if all(lig_inp.sites is not None for lig_inp in self.ligands):
            # Fully explicit
            assignment = np.empty(n_donors, dtype=int)
            idx = 0
            for li, lig_inp in enumerate(self.ligands):
                for gi, site in enumerate(lig_inp.sites):
                    assignment[idx] = site
                    idx += 1
        else:
            # Hungarian (or partial explicit — treat unassigned as Hungarian)
            explicit_map: Dict[int, int] = {}
            idx = 0
            for li, lig_inp in enumerate(self.ligands):
                for gi in range(len(lig_inp.donors)):
                    if lig_inp.sites is not None:
                        explicit_map[idx] = lig_inp.sites[gi]
                    idx += 1

            if explicit_map:
                # Partial explicit: fix explicit ones, Hungarian on rest
                free_donors = [i for i in range(n_donors) if i not in explicit_map]
                used_sites = set(explicit_map.values())
                free_sites = [s for s in range(len(ideal)) if s not in used_sites]

                assignment = np.empty(n_donors, dtype=int)
                for di, si in explicit_map.items():
                    assignment[di] = si

                if free_donors and free_sites:
                    free_dirs = donor_dirs[free_donors]
                    free_ideal = ideal[free_sites]
                    _free_local = {g: k for k, g in enumerate(free_donors)}
                    _free_chelate_pairs = [
                        (_free_local[a], _free_local[b])
                        for a, b in chelate_pairs_global
                        if a in _free_local and b in _free_local
                    ]
                    free_assign = hungarian_assign(
                        free_dirs,
                        free_ideal,
                        chelate_pairs=_free_chelate_pairs,
                    )
                    for di, local_si in zip(free_donors, free_assign):
                        assignment[di] = free_sites[local_si]
            else:
                assignment = hungarian_assign(
                    donor_dirs,
                    ideal,
                    chelate_pairs=chelate_pairs_global,
                )

            assignment = self._repair_infeasible_baseline(
                assignment,
                ideal,
                donor_dirs,
                explicit_map,
                all_donor_groups,
                n_donors,
                heavy_graph,
            )

        return assignment

    def _repair_infeasible_baseline(
        self, assignment, ideal, donor_dirs, explicit_map, all_donor_groups, n_donors, heavy_graph
    ):
        """Replace a topologically unreachable non-explicit baseline by the
        lowest-cost reachable assignment (caller-pinned sites are never moved)."""
        from jacobik.ik.reachability import (
            angular_cost_matrix,
            assignment_is_reachable,
            repair_unreachable_assignment,
            site_angles_deg,
        )

        if n_donors != len(ideal):
            return assignment
        try:
            hop = self._build_hop_matrix(n_donors, all_donor_groups, heavy_graph)
            site_ang = site_angles_deg(ideal)
            if hop is None or assignment_is_reachable(assignment, site_ang, hop):
                return assignment
            C = angular_cost_matrix(donor_dirs, ideal)
            for di, si in explicit_map.items():  # never move caller-pinned donors
                C[di, :] = 1e6
                C[di, si] = 0.0
            new, repaired = repair_unreachable_assignment(assignment, C, site_ang, hop)
            if repaired and all(int(new[di]) == si for di, si in explicit_map.items()):
                _e().log.warning(
                    "Baseline (Hungarian) assignment %s was topologically infeasible "
                    "(a donor pair is placed at an angle its ligand-graph distance "
                    "cannot reach); replaced by lowest-cost feasible assignment %s.",
                    assignment.tolist(),
                    new.tolist(),
                )
                return new
            if not repaired:
                _e().log.warning(
                    "Baseline assignment %s is topologically infeasible and no feasible "
                    "alternative was found; keeping it.",
                    assignment.tolist(),
                )
        except Exception as exc:  # never let a diagnostic repair break solve()
            _e().log.warning("Baseline feasibility check failed (%s); keeping baseline.", exc)
        return assignment

    def _symmetric_donor_groups(self, mol, rdmols, all_donor_groups):
        """
        Topologically equivalent single-atom donor groups (size >= 2), as lists
        of donor indices. Two donors are equivalent iff they share a SMILES and
        the same RDKit canonical atom rank (ties not broken), both within one
        ligand and across repeated ligand instances. Haptic groups are skipped.
        """
        try:
            from rdkit import Chem

            key_to_donors: Dict[Tuple[str, int], List[int]] = {}
            rank_cache: Dict[int, list] = {}
            for di, (li, _gi, grp) in enumerate(all_donor_groups):
                if len(grp) != 1:
                    continue
                if li not in rank_cache:
                    rank_cache[li] = list(Chem.CanonicalRankAtoms(rdmols[li], breakTies=False))
                offset = mol.ligands[li].atom_indices[0]
                local_idx = grp[0] - offset
                if local_idx < 0 or local_idx >= len(rank_cache[li]):
                    continue
                rank = rank_cache[li][local_idx]
                smiles = self.ligands[li].smiles
                key_to_donors.setdefault((smiles, rank), []).append(di)
            return [v for v in key_to_donors.values() if len(v) >= 2]
        except Exception as exc:
            _e().log.warning("Symmetric-donor-group detection failed (%s); using none.", exc)
            return []

    def _identical_ligand_instances(self, ctx):
        """Groups of identical ligand instances (same SMILES and donor count, >= 2 instances), each instance as its
        donor indices in order: the only donor relabelings that are exact symmetries of the embedded ligands
        (swapping two equivalent donors inside ONE ligand also swaps their 3D arrangement in the embed, which
        torsions cannot always undo; see site_enumeration.dedup_exact)."""
        by_lig: Dict[int, List[int]] = {}
        for di, (li, _gi, _grp) in enumerate(ctx.all_donor_groups):
            by_lig.setdefault(li, []).append(di)
        by_key: Dict[Tuple[str, int], List[List[int]]] = {}
        for li, ds in sorted(by_lig.items()):
            by_key.setdefault((self.ligands[li].smiles, len(ds)), []).append(ds)
        return [g for g in by_key.values() if len(g) >= 2]

    @timing.timed("assign")
    def _run_symmetry_enumeration(self, ctx, assignment, debug, embed_aware=True):
        """
        Enumerate every symmetry-distinct, reachability-consistent donor-to-site
        bijection (ik/site_enumeration.py). Returns the candidate list, or
        ``[assignment]`` (baseline) when the geometry has no fixed site set,
        the complex is not fully saturated, or enumeration fails or finds
        nothing.
        """
        from jacobik.ik.targets import IDEAL_GEOMETRIES

        if self.geometry not in IDEAL_GEOMETRIES:
            _e().log.info(
                "Symmetry enumeration skipped: geometry %r has no fixed site set.",
                self.geometry,
            )
            return [assignment]
        if ctx.n_donors != len(ctx.ideal):
            _e().log.info(
                "Symmetry enumeration skipped: n_donors (%d) != n_sites (%d) of %r.",
                ctx.n_donors,
                len(ctx.ideal),
                self.geometry,
            )
            return [assignment]

        from jacobik.ik.site_enumeration import enumerate_unique_assignments

        hop_matrix_full = self._build_hop_matrix(
            ctx.n_donors, ctx.all_donor_groups, ctx.heavy_graph
        )

        try:
            se_result = enumerate_unique_assignments(
                ctx.n_donors,
                self.geometry,
                hop_matrix_full,
                donor_equivalence_groups=ctx.symmetric_donor_groups,
                # embed_aware: only exact symmetries of the embedded ligands (proper rotations, whole-ligand swaps)
                instance_groups=self._identical_ligand_instances(ctx) if embed_aware else None,
            )
        except Exception as exc:
            _e().log.warning("Symmetry enumeration failed (%s); using the baseline assignment.", exc)
            return [assignment]

        if not se_result.assignments:
            _e().log.warning(
                "Symmetry enumeration found no reachable assignment; using the baseline assignment."
            )
            return [assignment]

        candidates = list(se_result.assignments)
        if se_result.truncated or se_result.dedup_truncated:
            _e().log.warning(
                "Symmetry enumeration hit its safety budget (raw truncated=%s, "
                "dedup truncated=%s); the %d candidate(s) found are valid but the "
                "symmetry-unique set may be incomplete.",
                se_result.truncated,
                se_result.dedup_truncated,
                len(candidates),
            )
        _e().log.info(
            "Symmetry enumeration: %d candidate assignment(s) (n_raw=%d before dedup)",
            len(candidates),
            se_result.n_raw,
        )
        if debug:
            print(f"\n[DEBUG] Symmetry enumeration: {len(candidates)} candidate(s)")
            for i, cand in enumerate(candidates):
                print(f"  {'(baseline)' if i == 0 else f'(alt {i})'} {cand.tolist()}")

        return candidates

    def _solve_for_assignments(self, ctx, assignments, beam_width, n_ik_steps):
        """
        Beam search, joint IK and conformer assembly for one or more candidate
        donor-to-site ``assignments``, batched into a single JointIKSolver call
        of ``len(assignments) * beam_width * n_coord_samples`` trials. Conformers
        from all candidates are ranked together by score
        (``ConformerResult.assignment_idx`` says which candidate produced each).
        """
        if getattr(self, "_capture_prep", False):      # config_states: one class's prepared solve, not solved here
            raise _e()._PrepCapture((ctx, assignments, beam_width, n_ik_steps))
        rows = self._prepare_rows(ctx, assignments, beam_width)
        thetas_final, converged, scores, solve_trace = self._run_joint_ik(
            ctx,
            rows["fks"],
            rows["scorer"],
            rows["ge"],
            rows["X"],
            rows["thetas"],
            rows["dpi"],
            rows["ideal"],
            rows["sym"],
            rows["q3sym"],
            rows["bw_total"],
            n_ik_steps,
            rows["cand"],
        )
        return self._finish_rows(ctx, assignments, rows, thetas_final, converged, scores, solve_trace)
