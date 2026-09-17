#!/usr/bin/env python3
import sys

PATH = "src/jacobik/engine.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

steps_done = []


def do_replace(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"FATAL: pattern for step [{label}] not found", file=sys.stderr)
        sys.exit(1)
    if n != count:
        print(f"FATAL: pattern for step [{label}] found {n} times, expected {count}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, count)
    steps_done.append(label)


# ---------------------------------------------------------------------------
# Step E1: solve() signature -- add use_shape_alm/shape_rho/shape_tol/sym_weight
# ---------------------------------------------------------------------------
OLD_SIG = '''        lambda_init: float = 10.0,
        anchor_floor_frac: float = 0.0,
        diversify_orientation: bool = True,'''
NEW_SIG = '''        lambda_init: float = 10.0,
        anchor_floor_frac: float = 0.0,
        use_shape_alm: bool = True,
        shape_rho: float = 8.0,
        shape_tol: float = 0.02,
        sym_weight: float = 2.0,
        diversify_orientation: bool = True,'''
do_replace("E1-solve-signature", OLD_SIG, NEW_SIG)

# ---------------------------------------------------------------------------
# Step E2: solve() docstring -- document the new params
# ---------------------------------------------------------------------------
OLD_DOC = '''        lambda_init      : anchor weight at t=0; decays toward 0 as score
                           improves (init-only anchor semantics).
        diversify_orientation : bool, default True (BUGFIX 2.5.1 -- was False).'''
NEW_DOC = '''        lambda_init      : anchor weight at t=0; decays toward 0 as score
                           improves (init-only anchor semantics). Under
                           use_shape_alm=True this instead seeds mu_shape's
                           initial value for the Augmented-Lagrangian dual
                           ascent (ik_solver.py 2.3.0) -- see that module's
                           changelog.
        use_shape_alm    : bool, default True (2.3.0). Replaces the old
                           fixed-schedule anchor (a pull toward FIXED
                           absolute Cartesian targets computed once, before
                           the solve, from the ligand's *initial* pose) with
                           an Augmented-Lagrangian shape constraint computed
                           via the optimal rotation between the ideal
                           polyhedron and the CURRENT donor arrangement,
                           RECOMPUTED EVERY STEP -- plus a symmetry-tying
                           restraint on topologically-equivalent donors
                           (e.g. EDTA's two amine N, or all 4 Cl- in PtCl4).
                           See ik/ik_solver.py's JointIKSolver 2.3.0
                           changelog for the full rationale and the
                           literature it follows (constrained optimisation
                           over fixed penalties; continuous shape measures).
                           Set False to restore the pre-2.3.0 behaviour
                           byte-identically, e.g. for A/B comparison.
        shape_rho        : Augmented-Lagrangian dual-ascent step size for the
                           shape constraint (only used when use_shape_alm).
                           Default 8.0 -- starting value, not empirically
                           fit; tune against the test-suite notebook.
        shape_tol        : shape-deviation tolerance below which the
                           constraint is considered satisfied (only used
                           when use_shape_alm). Default 0.02 (roughly an
                           8-degree worst-donor deviation from the ideal
                           polyhedron -- see JointIKSolver docstring).
        sym_weight       : fixed weight on the symmetry-tying restraint
                           (only used when use_shape_alm). Default 2.0.
        diversify_orientation : bool, default True (BUGFIX 2.5.1 -- was False).'''
do_replace("E2-solve-docstring", OLD_DOC, NEW_DOC)

# ---------------------------------------------------------------------------
# Step E3: build ideal_directions_full + symmetric_groups right after the
# site assignment is finalised.
# ---------------------------------------------------------------------------
OLD_ASSIGN_LOG = '''        log.info("Site assignment: %s", assignment.tolist())

        # ------------------------------------------------------------------
        # 4. Build coord-sphere targets (M samples per donor)
        # ------------------------------------------------------------------'''
NEW_ASSIGN_LOG = '''        log.info("Site assignment: %s", assignment.tolist())

        # ------------------------------------------------------------------
        # 3b. Ideal polyhedron directions in global donor order (2.3.0) --
        #     feeds JointIKSolver's per-step Augmented-Lagrangian shape
        #     constraint (ik_solver.py 2.3.0). Replaces the old fixed
        #     absolute-Cartesian-target anchor, whose target orientation
        #     went stale as soon as the ligand's free rigid pose or
        #     torsions moved away from the initial Kabsch fit -- see that
        #     module's changelog for the full rationale.
        # ------------------------------------------------------------------
        ideal_directions_full = np.stack(
            [ideal[assignment[di]] for di in range(n_donors)], axis=0
        )  # (n_donors, 3) unit vectors, fixed for the whole solve
        ideal_directions_full_t = torch.tensor(
            ideal_directions_full, dtype=torch.float32, device=_device
        )

        # ------------------------------------------------------------------
        # 3c. Topologically-equivalent donor groups (2.3.0) -- RDKit
        #     canonical ranking (breakTies=False) detects donor atoms
        #     related by a graph automorphism of their own ligand's
        #     molecular graph. Two donors are equivalent iff they share a
        #     SMILES string AND have the same canonical rank within that
        #     SMILES's graph -- true both for donors within one symmetric
        #     polydentate ligand (e.g. EDTA's two amine N) and for donors
        #     across separate instances of a repeated ligand (e.g. all 4
        #     Cl- in PtCl4, or corresponding N atoms across Co(en)3's three
        #     en ligands), since CanonicalRankAtoms is a pure function of
        #     molecular-graph topology -- identical SMILES always gives
        #     identical ranks. Feeds a soft M-D-distance similarity
        #     restraint (ik_solver.py 2.3.0, mirroring crystallography's
        #     SIMU/SAME restraints) addressing the observed spurious
        #     bond-length spread between donors that are chemically
        #     indistinguishable (e.g. EDTA's ~0.15 A spread between its two
        #     amine N atoms, or Co(en)3's inconsistent M-N lengths).
        #     Multi-atom (haptic) donor groups are skipped -- rare and
        #     untested for this restraint.
        # ------------------------------------------------------------------
        symmetric_groups: List[List[int]] = []
        try:
            from rdkit import Chem as _Chem

            _sym_key_to_donors: Dict[Tuple[str, int], List[int]] = {}
            _rank_cache: Dict[int, list] = {}
            for _di, (_li, _gi, _grp) in enumerate(all_donor_groups):
                if len(_grp) != 1:
                    continue
                if _li not in _rank_cache:
                    _rank_cache[_li] = list(
                        _Chem.CanonicalRankAtoms(rdmols[_li], breakTies=False)
                    )
                _offset = mol.ligands[_li].atom_indices[0]
                _local_idx = _grp[0] - _offset
                if _local_idx < 0 or _local_idx >= len(_rank_cache[_li]):
                    continue
                _rank = _rank_cache[_li][_local_idx]
                _smiles = self.ligands[_li].smiles
                _sym_key_to_donors.setdefault((_smiles, _rank), []).append(_di)
            symmetric_groups = [v for v in _sym_key_to_donors.values() if len(v) >= 2]
            if debug and symmetric_groups:
                print(f"\\n[DEBUG]   Symmetric donor groups (2.3.0): {symmetric_groups}")
        except Exception as exc:
            log.warning(
                "Symmetric-donor detection failed (%s); skipping similarity restraint.", exc
            )
            symmetric_groups = []

        # ------------------------------------------------------------------
        # 4. Build coord-sphere targets (M samples per donor)
        # ------------------------------------------------------------------'''
do_replace("E3-ideal-directions-and-symmetry", OLD_ASSIGN_LOG, NEW_ASSIGN_LOG)

# ---------------------------------------------------------------------------
# Step E4: JointIKSolver(...) construction -- pass shape_rho/shape_tol/sym_weight
# ---------------------------------------------------------------------------
OLD_CTOR = '''        joint_solver = JointIKSolver(
            fks=fks,
            scorer=scorer_joint,
            ge=ge_joint,
            n_steps=n_ik_steps,
            lr=lr,
            lambda_init=lambda_init,
            lse_temp=lse_temp,
            adam_betas=adam_betas,
            anchor_floor_frac=anchor_floor_frac,
            device=_device,
        )
        thetas_final, converged, scores, solve_trace = joint_solver.solve(
            theta_inits=all_theta_inits,
            X_ref=X_full,
            donor_targets=all_donor_targets,
            donor_pos_idxs=all_donor_pos_idxs,
            debug=debug,
        )'''
NEW_CTOR = '''        joint_solver = JointIKSolver(
            fks=fks,
            scorer=scorer_joint,
            ge=ge_joint,
            n_steps=n_ik_steps,
            lr=lr,
            lambda_init=lambda_init,
            lse_temp=lse_temp,
            adam_betas=adam_betas,
            anchor_floor_frac=anchor_floor_frac,
            shape_rho=shape_rho,
            shape_tol=shape_tol,
            sym_weight=sym_weight,
            device=_device,
        )
        thetas_final, converged, scores, solve_trace = joint_solver.solve(
            theta_inits=all_theta_inits,
            X_ref=X_full,
            donor_targets=all_donor_targets,
            donor_pos_idxs=all_donor_pos_idxs,
            debug=debug,
            ideal_directions=ideal_directions_full_t if use_shape_alm else None,
            symmetric_groups=symmetric_groups if use_shape_alm else None,
        )'''
do_replace("E4-jointiksolver-call", OLD_CTOR, NEW_CTOR)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
