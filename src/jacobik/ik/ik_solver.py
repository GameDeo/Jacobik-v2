"""
ik_solver.py
============
Gradient-descent IK solver for the conformer ensemble engine.

Version
-------
2.0.0 : Unified objective (init-only anchor, score-plateau convergence,
        single-tier best-score selection). See REWRITE_PLAN findings F1–F3.
2.1.0 : Speed improvements.
        - Single backward pass: anchor and chemistry losses combined into one
          expression before backward().  Removes the retain_graph=True call
          and the second graph traversal (~1.5× speedup on backward cost).
        - Plateau check cadence: ring buffer max/min evaluated every
          plateau_window steps rather than every step.  Same semantics, 5–7×
          less reduction overhead (12 ms → 2.5 ms per 200-step solve).
        - q2_weight removed from score() call site — now lives as a
          Scorer constructor argument (quantity_weights={"Q2": ...}).
          score() takes no per-call float args; all weights are baked into
          precomputed tensors at Scorer.__init__ time.
2.2.0 : kabsch_init() gains diversify_orientation/diversity_scale — extends
        the existing rank<2 (degenerate Kabsch fit) random-orientation
        injection to the general rank>=2 case (3+ non-collinear donors
        fit jointly), which previously had NO diversity injection at all
        because a well-posed least-squares fit doesn't need one to be
        mathematically correct -- it just isn't the only torsion-reachable
        wrap-around topology for a flexible ligand. Confirmed on EDTA (6
        donors, rank=3): every batch trial got an identical base
        orientation, collapsing 128 raw IK trials to 3 distinct final
        geometries. See engine.py 2.3.0 for the measured effect.
        JointIKSolver gains anchor_floor_frac: lambda_eff no longer decays
        all the way to 0, retaining `anchor_floor_frac * lambda_init`
        permanently. Tested in isolation on EDTA with negligible effect;
        kept as an available lever (default 0.0 = old behaviour) since it
        may help in combination with diversify_orientation on other cases.
2.3.0 : Shape enforcement reframed as a constrained-optimisation problem
        instead of a fixed-weight penalty, following the literature on why
        fixed penalty coefficients cannot reliably reach a constrained
        optimum in non-convex settings (e.g. arXiv:2505.20628) and on
        continuous shape measures (CShM; Pinsky & Avnir; Alvarez group) as
        the established, dilution-resistant way to score deviation from an
        ideal coordination polyhedron.

        Root cause this addresses: the old anchor pulled every donor toward
        a FIXED absolute Cartesian point (direction * sampled Q1 radius)
        computed once, before the solve, from the ligand's *initial* pose.
        As soon as the free 6-DOF rigid pose or the ligand's own torsions
        moved away from that initial pose during optimisation (routine for
        flexible/chelating ligands such as EDTA or en), the anchor target
        stopped representing the best-fit polyhedron for the *current*
        geometry -- it was scoring deviation from a stale frame. A strong
        anchor then fights the statistical score over an unreachable
        configuration ("destroys convergence"); a weak one lets each donor
        drift independently once nothing recomputes a coordinated target
        ("shells drift/dangle"). No single fixed lambda(t) schedule can
        serve both regimes at once -- this is not a tuning miss, it is the
        documented failure mode of fixed-weight penalties in non-convex
        optimisation.

        Fix: when `ideal_directions` is supplied to solve(), the anchor is
        replaced by a shape term computed via the OPTIMAL rigid rotation
        (Procrustes/Kabsch, rotation-only -- both point sets already share
        the metal as a common physical origin, so no translation/centroid
        step is used) between the fixed ideal polyhedron directions and the
        CURRENT donor unit vectors, RECOMPUTED EVERY STEP. This can never
        fight an unreachable frozen target: it always measures deviation
        from the best achievable instance of the target shape. The
        rotation is computed under torch.no_grad() and only the residual
        against it is backpropped -- by the envelope theorem this is exact
        to first order at the rotation's own optimum (d(loss)/dR = 0
        there), and it sidesteps the numerical instability of backpropping
        through SVD on near-degenerate inputs. Because R is shared across
        all donors, a single badly-placed donor also perturbs the fit the
        others are judged against -- unlike Q2's plain per-pair mean, this
        measure cannot be diluted away by averaging over many terms.

        The fixed lambda(t) decay schedule is replaced by an Augmented
        Lagrangian (dual-ascent) multiplier mu, updated every step as
        mu <- clamp(mu + shape_rho * violation, min=0) where violation is
        the (soft-max-aggregated) shape deviation above a small tolerance
        shape_tol. mu self-adjusts to whatever strength the CURRENT ligand
        needs -- no more hand-tuning one schedule to serve both a rigid
        PtCl4-like case and a flexible EDTA-like case simultaneously.

        A separate, small-fixed-weight symmetry-tying restraint
        (sym_weight, `symmetric_groups`) is also added: topologically
        equivalent donor atoms (detected via RDKit canonical ranking,
        engine.py) get an explicit soft restraint pulling their M-D
        distances toward each other, mirroring crystallographic SIMU/SAME
        restraints. Without this, two chemically-identical donors (e.g.
        EDTA's two amine N atoms) are independent draws from the same Q1
        distribution with nothing tying them together, so nothing prevents
        the spurious spread (~0.1-0.2 A observed) between them.

        Backward compatible: when `ideal_directions` is None (not passed
        by the caller), solve() falls back byte-identically to the old
        fixed-schedule anchor. anchor_floor_frac is accepted but ignored
        under the new path (the dual multiplier already re-inflates
        whenever the constraint is violated again -- no floor needed).
2.4.0 : Symmetry-tying restraint (2.3.0) extended to also tie Q3 (donor
        approach angle) between topologically-equivalent donors, not just
        M-D bond length.

        Root cause this addresses: EVERY Q3 entry in the live
        constraints.json (all 2956 of them) turned out to be type="gmm"
        misfiled into what ik/scorer.py's old _vmm_fields() treated as a
        VMM slot -- Q3 was never actually a clean native VMM fit in
        production, it was always going through a "synthesise kappa from
        vars, then clamp to _KAPPA_SYNTH_MAX=10" fallback shim. For both
        of EDTA's donor types the raw fitted variances were tight enough
        that every component saturated that clamp, so Q3 was in practice
        an almost-toothless, uniformly ~18 deg-sigma constraint for its
        entire history, not the differentiated preference it was designed
        to be. The Q2/Q3 GMM-cos refit (stats/fitter.py, same session)
        replaced those stale entries with a real native fit for the first
        time -- which is a big part of why the CO/TBP orientation fix
        worked, but it also means two chemically-identical donors (e.g.
        EDTA's two amine N, or its four carboxylate O) are now each
        independently pulled by a real, sometimes sharp Q3 preference for
        the first time, with nothing tying their outcomes together -- the
        symmetric_groups restraint only ever tied M-D distance. Plausible,
        evidence-consistent mechanism for the "uneven N anchoring /
        carboxylate bond-length spread" reported after that refit: one
        donor settles near a good Q3 mode, its symmetric partner settles
        near a different locally-reachable one, and the Q1/shape terms
        compromise differently for each to reconcile the overall CShM
        target.

        Fix: a second, independently-weighted soft restraint
        (sym_angle_weight, default 1.0 -- starting value, not empirically
        fit, same caveat as shape_rho/shape_tol/sym_weight) pulls each
        symmetric group's Q3-eligible members' approach angles toward
        their group mean, mirroring the existing bond-length restraint's
        pattern exactly (mean-squared deviation from the group mean,
        averaged over active groups). Deliberately a SEPARATE weight
        rather than reusing sym_weight: bond-length deviations (Å²) and
        approach-angle deviations (rad²) have different natural scales,
        and conflating them risks exactly the kind of cross-quantity
        imbalance this whole change is meant to fix, not another instance
        of it. Opt-in the same way sym_weight already is (only active when
        symmetric_groups is supplied and a group has >=2 Q3-eligible
        members) -- backward compatible, zero effect otherwise.
2.5.0 : Mode-committed Q3 initialization -- solve() gains q3_mode_commit
        (default True).

        Root cause this addresses: with Q3 genuinely live (2.7.0 refit +
        the stats.py _is_empty_dist bugfix, same session), some donor
        subclasses have a REAL multi-modal fitted GMM -- confirmed on
        Co/octahedral O_carboxylate: 5 well-separated cos-space modes
        (means -0.638/-0.486/-0.439/-0.028/+0.045, i.e. roughly 87-130
        deg), reflecting genuinely different carboxylate binding modes
        pooled across the CSD (monodentate/bridging/chelating etc.), not
        a fitting artifact. EDTA's four carboxylate O are one RDKit
        canonical-rank symmetric group (all four -CH2COO- arms are
        graph-equivalent), so each independently settles near whichever
        mode it's nearest to, and the 2.4.0 sym_angle_weight restraint
        then pulls them toward their MEAN angle -- for two well-separated
        modes, the mean lands in the low-density valley BETWEEN them,
        which is worse than either mode alone and fights each donor's own
        strong pull toward its nearest mode. Observed effect: EDTA's
        n_converged collapsed (29 -> 1-2/32) and was essentially
        insensitive to sym_angle_weight (0.0/1.0/3.0 all similarly bad),
        consistent with the trajectory oscillating between competing
        modes rather than settling into a plateau ("converged" here means
        score-plateaued, not merely score-above-threshold).

        Fix: rather than letting each symmetric donor drift to its own
        mode and then averaging, COMMIT the whole group to one mode up
        front. Once per solve() call (not once per IK step -- the choice
        is made from the initial pose and held fixed for the whole
        trajectory, mirroring how theta_inits itself is fixed per trial),
        for each symmetric group with >=2 Q3-eligible members: evaluate
        the group's INITIAL Q3 angle (from theta_inits, before any
        gradient steps) in cos-space against every component of the
        group's own fitted Q3 GMM(s), sum log-density across the group's
        members for each candidate component k, and pick
        argmax_k(sum). This is exactly "which single mode best explains
        where this group already started", using the DB's own fitted
        weights and variances -- no new statistical assumption, just a
        hard per-group, per-trial commitment instead of independent
        per-donor soft draws from the same mixture. The resulting
        (B, n_q3) component-index tensor (-1 for non-grouped/ungrouped
        terms) is passed as q3_mode_override to every scorer.score() call
        for the rest of that solve() (scorer.py 2.8.0), which collapses
        the mixture to that single Gaussian for those terms -- turning a
        multi-modal tug-of-war into a well-behaved, shared unimodal
        target by construction. Because the choice is made per TRIAL
        (batch index), different trials in the same beam can still commit
        to different modes -- diversity across the beam is preserved,
        only the within-trial inconsistency is removed.

        Opt-in via q3_mode_commit (default True): only computed/applied
        when symmetric_groups is supplied and has a qualifying group (same
        gating as sym_angle_weight); q3_mode_commit=False restores
        pre-2.5.0 behaviour byte-identically (q3_mode_override stays None,
        scorer.py's new code path is a no-op). Not yet empirically
        validated against a live solve at the time of writing (implemented
        from static constraints.json inspection + code-path tracing only,
        since Claude cannot run torch in the environment this was
        developed in) -- see the test-suite notebook's shape-ALM A/B
        section for the intended comparison against q3_mode_commit=False
        and the sym_angle_weight variants (2.4.0).

2.6.0 : mu_shape stall safeguard + sym_weight_tracks_mu -- addresses a
        distinct failure mode found while diagnosing EDTA's convergence
        collapse empirically, downstream of the 2.3.0-2.5.0 fixes above:
        even where those fixes are irrelevant (EDTA's problem turned out
        not to be Q3 multi-modality), the shape-ALM dual variable
        mu_shape itself grows WITHOUT BOUND whenever the shape constraint
        is effectively infeasible for a ligand's real connectivity
        (confirmed via solve_trace inspection: EDTA's shape_dev plateaus
        at ~0.89, ~45x shape_tol, within 50-75 steps and never moves
        again, while mu_shape keeps climbing linearly for the full T --
        classical dual ascent assumes escalating the multiplier will
        eventually drive the violation to zero, which cannot happen for
        a genuinely-unreachable target). By late steps mu_shape reaches
        1500-2000+ (5000+ on the worst, actively-diverging trials), which
        doesn't just fail on its own terms -- it comes to dominate the
        ENTIRE combined loss, reducing the FIXED-weight
        sym_weight/sym_angle_weight symmetry-tying restraints
        (2.3.0/2.4.0) to a rounding error (measured ratio of
        shape-penalty : sym-distance-tie-penalty magnitude: 17-84x at
        step 0, 3000-3600x by step 199). This is the direct, confirmed
        mechanism behind EDTA's two topologically-equivalent amine N
        ending up at very different M-D distances despite an explicit
        restraint meant to prevent exactly that -- and it retroactively
        explains why the 2.4.0 changelog's sym_angle_weight A/B test
        (0.0/1.0/3.0) found "statistically indistinguishable" results:
        against a competing term already in the thousands, that weight
        range is noise.

        Two independent, complementary fixes, both opt-out via a bool
        flag for byte-identical pre-2.6.0 A/B comparison:

        mu_shape_freeze_on_stall (default True): a ring-buffer stall
        detector on `violation` (same non-overlapping-window pattern as
        the existing score-plateau ring buffer above) -- once violation
        stops improving by more than mu_shape_stall_atol over
        mu_shape_stall_window steps, mu_shape is FROZEN (dual ascent
        stops for that trial) instead of continuing to escalate against
        a violation it demonstrably isn't reducing. This is a standard
        safeguarded-ALM technique (cf. Birgin & Martinez, "Practical
        Augmented Lagrangian Methods for Constrained Optimization" --
        classical safeguards only grow the penalty when infeasibility
        hasn't decreased sufficiently since the last check; see also
        gradient-based analogues for Adam-style training such as
        conditionally-adaptive penalty updates for constrained neural
        nets, e.g. arXiv:2508.15695). Freezing is permanent once
        triggered (mirrors ever_plateaued's own monotonic-OR semantics
        above) -- deliberately conservative rather than attempting to
        detect and react to a later re-violation. mu_shape_max (default
        None) is an orthogonal hard ceiling, active regardless of the
        stall detector, as a defense-in-depth safety net. Well-behaved
        cases (PtCl4 and other geometries where shape_dev actually
        reaches shape_tol) are unaffected: violation vanishes quickly, so
        the dual-ascent increment is already ~0 long before any stall
        window could close on a nonzero plateau.

        sym_weight_tracks_mu (default True): scales sym_weight and
        sym_angle_weight by (mu_shape / lambda_init) each step, so the
        RATIO between the shape penalty and the symmetry-tying penalties
        stays pinned near its initial (presumably reasonable) value for
        the whole solve, instead of decaying by 1000x+ as mu_shape grows.
        This is what actually keeps topologically-equivalent donors tied
        together in cases where mu_shape legitimately needs to grow large
        (freezing alone bounds the damage but a frozen-at-1500 mu_shape
        can still dwarf a fixed sym_weight=2.0).

        Neither fix touches shape_tol, the Hungarian donor-vertex
        assignment, or Q3 handling -- deliberately orthogonal to the
        2.3.0-2.5.0 mechanisms and to the (separately logged, not yet
        fixed) site-orbit-blindness bug in engine.py's symmetric_groups.
        Not yet empirically validated against a live solve at the time of
        writing (implemented from solve_trace data + code-path tracing
        only, since Claude cannot run torch in the environment this was
        developed in) -- see the test-suite notebook for the intended
        EDTA / PtCl4 / TBP A/B comparison against
        mu_shape_freeze_on_stall=False, sym_weight_tracks_mu=False.

2.7.0 : sym_use_own_alm -- sym_weight/sym_angle_weight promoted to their
        own independent safeguarded-ALM dual variables (mu_sym,
        mu_sym_angle), replacing 2.6.0's sym_weight_tracks_mu ratio as the
        default mechanism. Empirical validation of 2.6.0 (test-suite
        notebook section 14, run against live PtCl4/Fe(CO)5/EDTA solves)
        found sym_weight_tracks_mu: (a) measurably regresses PtCl4's
        bond_spread_A (0.028 -> 0.103 A) even though PtCl4 has no
        symmetric-tie donors at all, because pinning the RATIO to
        mu_shape's escalation still couples an unrelated case's sym_weight
        to shape's own dynamics; (b) only reduced EDTA's
        symmetric_pair_spread_A by ~12% (1.073 -> 0.946 A), far short of
        resolving the amine-asymmetry symptom it was meant to fix. Root
        cause: tying sym_weight to mu_shape via a fixed ratio only
        prevents that ratio from decaying FURTHER as mu_shape grows -- it
        does nothing about the ratio already being wrong at t=0 (measured
        17-84x in shape's favor, see 2.6.0 changelog), and it makes
        sym_weight's behaviour depend on a different constraint's
        multiplier rather than on whether the symmetry tie is itself
        satisfied. Neither problem can be fixed by picking a better fixed
        ratio or a better fixed sym_weight constant: any single constant
        tuned against a handful of test cases is a fit to those cases'
        symptoms, not a mechanism, and won't generalize to a different
        ligand or seed distribution (this was flagged during development
        before being empirically confirmed, and the ratio-preservation
        idea above already had the same defect in a subtler form).

        The generalizable fix (standard multi-constraint safeguarded-ALM,
        same Birgin & Martinez tradition mu_shape already uses): give
        EACH constraint its own multiplier that dual-ascends off its OWN
        violation, independent of every other constraint. sym_penalty and
        sym_angle_penalty each get their own dual variable (mu_sym,
        mu_sym_angle) that escalates only while that specific restraint
        is unsatisfied, with its own stall-freeze safeguard (identical
        ring-buffer mechanism to mu_shape_freeze_on_stall, governed by
        sym_stall_window/sym_stall_atol/sym_freeze_on_stall). mu_sym and
        mu_sym_angle are initialised from sym_weight and sym_angle_weight
        respectively for continuity with their pre-2.6.0 meaning as
        starting weights. This self-calibrates per case with no
        hand-tuned constant: PtCl4 has no symmetric donors, so
        sym_penalty is always ~0 and mu_sym never has anything to
        escalate against (stays inert, explaining why it should never
        have regressed PtCl4 in the first place); Fe(CO)5's small
        axial/equatorial coupling escalates a little and plateaus;
        EDTA's genuinely-conflicting amine tie escalates until it is
        satisfied or demonstrably stalled, exactly like mu_shape does for
        the shape constraint itself.

        sym_use_own_alm (default True) selects this mechanism.
        sym_weight_tracks_mu (2.6.0) is retained for backward
        compatibility and is only consulted when sym_use_own_alm=False;
        with both False, behaviour is byte-identical to pre-2.6.0 fixed
        weights. Not yet empirically validated against a live solve at
        the time of writing (implemented from the section-14 trace
        evidence above, since Claude cannot run torch in the environment
        this was developed in) -- see the test-suite notebook for the
        intended EDTA / PtCl4 / Fe(CO)5 A/B/C comparison against
        sym_use_own_alm=False (both the tracks_mu=True and tracks_mu=False
        sub-variants).
2.8.0 : solve() gains q3_symmetric_groups, an optional, narrower stand-in
        for symmetric_groups used ONLY by the Q3 approach-angle tie
        (sym_angle_penalty) and q3_mode_commit; symmetric_groups itself is
        untouched and still used as-is for the M-D bond-length tie
        (sym_penalty). None (default) falls back to symmetric_groups,
        preserving pre-2.8.0 behaviour byte-for-byte for any caller that
        doesn't pass it.

        Root cause addressed (found empirically this session, live solve --
        first live validation q3_mode_commit has had since its 2.5.0
        introduction): q3_mode_commit's "commit the whole group to whichever
        GMM component best explains its initial pose" logic is sound for a
        single ligand's own internally-symmetric donor arms (EDTA's 4
        carboxylate O, mechanically coupled through one backbone -- a
        genuine joint decision), which is the only configuration it was
        designed and reasoned through for. Called on a symmetric_groups
        entry spanning MULTIPLE separate ligand instances of a repeated
        monodentate ligand instead (e.g. all 6 waters of [Fe(H2O)6]3+,
        which land in one group purely because they share a SMILES and
        canonical rank), the same mechanism hard-locks chemically
        independent donors together with no chemical justification -- nothing
        requires 6 separate, freely-rotating water molecules to share one
        Q3 approach-angle mode, and the real fitted Q3 distribution for
        O_water is itself genuinely multi-modal (6 components, pooled from
        many different real crystal environments). Live-tested on
        [Fe(H2O)6]3+ (octahedral): the initial-pose argmax reliably selected
        a ~4%-weight minority mode identically for all 6 waters, every
        trial, regardless of seed -- producing an M-O-H geometry of ~68 deg
        / ~172 deg (one H nearly eclipsing the metal, the other pointing
        almost straight out) instead of the chemically expected symmetric
        ~126/126 deg, and doing so at a HIGHER solve score than the correct
        geometry (q3_mode_commit=False on the same case: clean ~126/126 deg,
        lower score) -- i.e. this is not an optimizer-quality problem, the
        committed objective itself was rewarding the wrong answer.

        This is a strong candidate for the "CO/TBP donor-orientation
        straightness fix, then a later change regressed it again" pattern
        reported this session: q3_mode_commit was layered on top of the
        very Q2/Q3 GMM-cos refit that made the original CO/TBP fix work
        (2.4.0 changelog), specifically to patch EDTA's within-ligand
        carboxylate multi-modality, and was never subsequently validated
        against a live solve of any simple repeated-monodentate case (its
        own 2.5.0 changelog says as much). See engine.py's 2.12.0 changelog
        for the q3_symmetric_groups construction (within-single-ligand-
        instance filter of symmetric_groups) that fixes this while leaving
        EDTA -- already single-instance -- byte-identical.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from jacobik.ik.kernels import axisangle_to_matrix, wrap_angle

if TYPE_CHECKING:
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer

log = logging.getLogger(__name__)

_EPS = 1e-7


def kabsch_init(
    ge: "GeometryExtractor",
    X_ref: torch.Tensor,  # (N, 3) — full complex reference positions
    donor_targets: torch.Tensor,  # (B, n_d_lig, 3) targets for THIS ligand
    donor_pos_idx: Optional[torch.Tensor] = None,  # (n_d_lig,) position indices into X_ref
    diversify_orientation: bool = False,
    diversity_scale: float = 1.0,
) -> torch.Tensor:
    """
    Initialise base DOF (translation + axis-angle) for each batch item
    via Kabsch alignment of current donor positions to targets.

    Parameters
    ----------
    diversify_orientation : bool
        BUGFIX (2.2.0): the original orientational-diversity injection below
        only fires for rank 0/1 fits (1 donor, or collinear donors) -- the
        cases where the least-squares rigid alignment is genuinely
        underdetermined. For 3+ non-collinear donors (rank 2-3, the common
        case for tridentate+ ligands), _kabsch(P, Q) has a UNIQUE optimal
        solution: every trial in the batch gets the identical base
        orientation regardless of beam_width or n_coord_samples, because
        that's the correct answer to "best rigid-body fit of this point
        cloud" -- it just isn't the only *reachable* wrap-around topology
        for a flexible, torsion-jointed ligand. Measured on EDTA (6 donors,
        rank=3): 128 raw IK trials collapsed to 3 distinct final geometries
        (pairwise RMSD ~0 within each cluster) because every trial started
        from the same rigid-body placement and could only locally wiggle
        from there. Setting diversify_orientation=True perturbs R with an
        independent random rotation per trial (same mechanism already used
        for the rank<2 case) even when the Kabsch fit is well-posed, so
        beam_width/n_coord_samples actually explore distinct basins instead
        of W x M near-copies of one.
    diversity_scale : float
        Max perturbation angle in units of pi (1.0 = full random SO(3)-ish
        perturbation about a random axis, matching the existing rank<2
        code's `torch.rand(1) * 2*pi`). Lower values keep trials closer to
        the Kabsch optimum while still breaking exact ties.
    """
    device = donor_targets.device
    B = donor_targets.shape[0]

    with torch.no_grad():
        X_ref_dev = X_ref.to(device)  # (N, 3)

        if donor_pos_idx is not None:
            # Slice out this ligand's donor positions directly
            donor_pos_idx = donor_pos_idx.to(device)
            X_ref_b = X_ref_dev.unsqueeze(0).expand(B, -1, -1)
            all_dpos = ge.donor_positions(X_ref_b)
            donor_pos = all_dpos[:, donor_pos_idx, :]
        else:
            # Fallback: use ge (returns all coord sites)
            X_ref_b = X_ref_dev.unsqueeze(0).expand(B, -1, -1)
            donor_pos = ge.donor_positions(X_ref_b)  # (B, n_d, 3)

    base_theta = torch.zeros(B, 6, device=device)

    for b in range(B):
        P = donor_pos[b]  # (n_d_lig, 3) current donor positions
        Q = donor_targets[b]  # (n_d_lig, 3) target positions

        R, t = _kabsch(P, Q)

        # ---- INJECT ORIENTATIONAL DIVERSITY FOR UNDERDETERMINED FITS ----
        # If P is a single point or collinear, the Kabsch fit has unconstrained
        # rotational DOFs. We inject random rotations in those DOFs to ensure
        # the batch explores diverse starting orientations.
        Pc = P - P.mean(dim=0)
        cov = Pc.T @ Pc
        eigenvalues = torch.linalg.eigvalsh(cov)
        rank = (eigenvalues > 1e-5).sum().item()

        if rank == 0:
            # 1 donor. The Kabsch fit aligns the centroid, but orientation is free.
            # The most geometrically relevant unconstrained DOF is rotation about
            # the Metal-Donor axis (the null axis for a single donor constraint).
            metal_pos = X_ref_dev[ge.metal_pos_idx]
            axis = Q.mean(dim=0) - metal_pos
            if axis.norm() > 1e-5:
                axis = F.normalize(axis, dim=0)
                angle = torch.rand(1, device=device).item() * 2.0 * math.pi
                R_rand = axisangle_to_matrix((axis * angle).unsqueeze(0)).squeeze(0)
                R = R_rand @ R
                t = Q.mean(dim=0) - R @ P.mean(dim=0)

        elif rank == 1:
            # Collinear donors. 1 unconstrained DOF (rotation around the axis).
            # The axis in Q-space is the principal eigenvector of Q's covariance.
            Qc = Q - Q.mean(dim=0)
            Q_cov = Qc.T @ Qc
            _, Q_eigvecs = torch.linalg.eigh(Q_cov)
            axis = Q_eigvecs[:, -1]
            if axis.norm() > 1e-5:
                axis = F.normalize(axis, dim=0)
                angle = torch.rand(1, device=device).item() * 2.0 * math.pi
                R_rand = axisangle_to_matrix((axis * angle).unsqueeze(0)).squeeze(0)
                R = R_rand @ R
                t = Q.mean(dim=0) - R @ P.mean(dim=0)

        elif diversify_orientation:
            # rank >= 2: the Kabsch fit is well-posed (unique optimum) but
            # that uniqueness is exactly the bug for flexible multi-donor
            # ligands -- see diversify_orientation docstring above. Perturb
            # with an independent random rotation about a random axis, same
            # form as the rank<2 branches, scaled by diversity_scale.
            rand_axis = torch.randn(3, device=device)
            axis_norm = rand_axis.norm()
            if axis_norm > 1e-8:
                axis = rand_axis / axis_norm
                angle = torch.rand(1, device=device).item() * diversity_scale * math.pi
                R_rand = axisangle_to_matrix((axis * angle).unsqueeze(0)).squeeze(0)
                R = R_rand @ R
                t = Q.mean(dim=0) - R @ P.mean(dim=0)
        # -----------------------------------------------------------------

        omega = _matrix_to_axisangle(R)

        base_theta[b, 0:3] = t
        base_theta[b, 3:6] = omega

    return base_theta  # (B, 6)


# ---------------------------------------------------------------------------
# Kabsch alignment (CPU, no autograd needed)
# ---------------------------------------------------------------------------


def _kabsch(
    P: torch.Tensor,  # (n, 3) — source points (current donor positions)
    Q: torch.Tensor,  # (n, 3) — target points (desired donor positions)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kabsch algorithm: find (R, t) minimising ‖R·P + t − Q‖²_F.

    Returns
    -------
    R : (3, 3) rotation matrix with det(R) = +1
    t : (3,)   translation vector
    """
    n = P.shape[0]

    if n == 1:
        # Single point: no rotation can be determined — use identity
        R = torch.eye(3, dtype=P.dtype, device=P.device)
        t = (Q - P).squeeze(0)
        return R, t

    # Centre both point sets
    P_mean = P.mean(dim=0)  # (3,)
    Q_mean = Q.mean(dim=0)
    Pc = P - P_mean  # (n, 3)
    Qc = Q - Q_mean

    # Cross-covariance matrix H = Pc^T · Qc
    H = Pc.T @ Qc  # (3, 3)

    # SVD
    try:
        U, S, Vt = torch.linalg.svd(H)
    except Exception:
        # Degenerate case (all donors colinear) — fall back to identity
        R = torch.eye(3, dtype=P.dtype, device=P.device)
        t = Q_mean - P_mean
        return R, t

    V = Vt.T  # (3, 3)

    # Correct for reflection: ensure det(V·U^T) = +1
    d = torch.linalg.det(V @ U.T)
    sign_fix = torch.ones(3, dtype=P.dtype, device=P.device)
    sign_fix[-1] = d.sign()
    V = V * sign_fix[None, :]  # flip last column if det < 0

    R = V @ U.T  # (3, 3)
    t = Q_mean - R @ P_mean  # (3,)

    return R, t


def _matrix_to_axisangle(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to an axis-angle vector ω.


    Parameters
    ----------
    R : (3, 3) rotation matrix

    Returns
    -------
    omega : (3,) axis-angle vector

    """
    # Angle from trace: cos θ = (tr(R) − 1) / 2
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # scalar in [0, π]

    if theta.abs() < _EPS:
        # Near-identity rotation
        return torch.zeros(3, dtype=R.dtype, device=R.device)

    if math.pi - theta.item() < 1e-4:
        # θ ≈ π: axis from symmetric part (R + I)/2
        # Each diagonal element = (1 + cos θ)/2 + axis_i² · (1 − cos θ)/2
        # At θ=π: diag element ≈ axis_i²
        sym = (R + torch.eye(3, dtype=R.dtype, device=R.device)) / 2.0
        axis = sym.diagonal().clamp(min=0.0).sqrt()
        # Resolve sign ambiguity from off-diagonal elements
        if R[0, 1].item() < 0:
            axis[1] = -axis[1]
        if R[0, 2].item() < 0:
            axis[2] = -axis[2]
        axis = F.normalize(axis.unsqueeze(0), dim=-1).squeeze(0)
        return axis * theta

    # General case: axis from skew-symmetric part of R
    # [R - R^T] = 2 sin(θ) · [axis]×
    axis = torch.stack(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    ) / (2.0 * torch.sin(theta).clamp(min=_EPS))

    axis = F.normalize(axis.unsqueeze(0), dim=-1).squeeze(0)
    return axis * theta


# ---------------------------------------------------------------------------
# Axis-angle renormalisation
# ---------------------------------------------------------------------------


def _renorm_axisangle(omega: torch.Tensor) -> torch.Tensor:
    """
    Renormalise axis-angle vectors: if ‖ω‖ > π, replace ω → ω · π/‖ω‖.


    Parameters
    ----------
    omega : (B, 3) axis-angle vectors

    Returns
    -------
    omega_normed : (B, 3) with ‖ω‖ ≤ π for all rows

    """
    norms = omega.norm(dim=-1, keepdim=True)  # (B, 1)
    scale = torch.where(
        norms > math.pi,
        math.pi / norms.clamp(min=_EPS),
        torch.ones_like(norms),
    )
    return omega * scale


# ---------------------------------------------------------------------------
# Batched rotation-only Procrustes alignment (no translation/centroid step)
# ---------------------------------------------------------------------------


def _optimal_rotation_nocentroid(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched Procrustes/Kabsch rotation-only alignment, with NO translation
    and NO centroid subtraction: A and B are already metal-centred unit
    vectors (both point sets share a common *physical* origin -- the metal
    atom -- so translating either away from it would break that physical
    meaning, unlike a generic point-cloud Procrustes problem).

    For each batch element b, finds the rotation R[b] minimising
    sum_i || R[b] @ A[i] - B[b, i] ||^2.

    Same SVD convention as _kabsch() above (H = P^T Q, R = V U^T with a
    reflection fix so det(R) = +1), just batched over B and without the
    centroid/translation step, since here both point sets already share
    the same physical origin by construction.

    Parameters
    ----------
    A : (n, 3) fixed reference unit vectors (e.g. ideal polyhedron
        directions), static across the whole solve.
    B : (Batch, n, 3) current unit vectors (recomputed every step).

    Returns
    -------
    R : (Batch, 3, 3) rotation matrices, det(R) = +1.

    Note: callers in this file always call this under torch.no_grad() and
    then backprop only through the *residual* against the detached R (see
    the 2.3.0 changelog above for why that is exact to first order at R's
    own optimum) -- so this function itself does not need to be
    autograd-stable through near-degenerate singular values.
    """
    Bsz = B.shape[0]
    H = torch.einsum("ni,bnj->bij", A, B)  # (Batch, 3, 3) = A^T @ B[b]
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)
    Ut = U.transpose(-2, -1)
    det_sign = torch.det(torch.matmul(V, Ut)).sign()  # (Batch,) in {-1, 0, +1}
    det_sign = torch.where(det_sign == 0, torch.ones_like(det_sign), det_sign)
    D = torch.eye(3, dtype=A.dtype, device=A.device).unsqueeze(0).repeat(Bsz, 1, 1)
    D[:, 2, 2] = det_sign
    R = torch.matmul(torch.matmul(V, D), Ut)  # (Batch, 3, 3): R @ A[i] ~= B[b, i]
    return R


# ---------------------------------------------------------------------------
# JointIKSolver
# ---------------------------------------------------------------------------


class JointIKSolver:
    """
    Joint gradient-descent IK solver that optimises all ligands simultaneously.

    Objective
    ---------
    Minimise −score(X) + λ(t)·anchor(X, targets), where score() is the
    Scorer's unified statistical score (Scorer.score(), scorer.py 2.0.0).
    The anchor weight λ(t) starts at lambda_init and decays toward zero as
    the score improves, implementing init-only anchor semantics: the anchor
    seeds the correct donor→site basin, then yields to the score.

    Quantity weights (including Q2 enable/disable) are set on the Scorer
    at construction time via its quantity_weights argument — not here.

    Parameters
    ----------
    fks          : list of ForwardKinematics, one per ligand, in order
    scorer       : Scorer built on the full complex (one shared instance)
    ge           : GeometryExtractor for the full complex
    n_steps      : gradient-descent iterations T  (default 200)
    lr           : learning rate for Adam on the combined gradient (default 0.01)
    lambda_init  : anchor weight at t=0; decays toward 0 as score improves
                   (default 10.0)
    lse_temp     : temperature for the log-sum-exp soft-max over per-donor
                   squared errors in the anchor loss.  Reused from scorer.py
                   Q6 LSE (temperature 2.0 = one device, one constant).
                   Higher → mean; lower → hard-max on worst donor.  (default 2.0)
    adam_betas   : (β₁, β₂) momentum coefficients for Adam (default (0.9, 0.999))
    plateau_window : steps over which score must plateau to declare convergence
                   (default 8)
    plateau_atol : absolute score change below which a step is "not improving"
                   for plateau detection (default 0.02 nats)
    anchor_floor_frac : fraction of lambda_init retained permanently instead
                   of decaying fully to zero (default 0.0 = old behaviour).
                   EDTA-class crowded/flexible polydentate ligands were
                   observed to drift to a WORSE score and geometry under a
                   longer solve once the anchor fully vanished (Q1-Q6 alone
                   underconstrains the donor shell once nothing is pinning
                   it to the target polyhedron) -- a small nonzero floor
                   (~0.02-0.05) keeps a gentle permanent pull toward the
                   assigned geometry without preventing the statistical
                   score from doing the local refinement.
                   DEPRECATED under the 2.3.0 shape-ALM path (solve() called
                   with ideal_directions given): the dual multiplier
                   mu_shape already re-inflates whenever the constraint is
                   violated again, so a manual floor is unnecessary there.
                   Still honoured on the old fixed-schedule fallback path.
    shape_rho    : Augmented-Lagrangian penalty/step-size rho for the shape
                   constraint dual ascent (2.3.0+; only used when solve() is
                   called with ideal_directions). mu_shape <- clamp(mu_shape
                   + shape_rho * violation, min=0) every step. (default 8.0)
    shape_tol    : shape-deviation tolerance below which the constraint is
                   considered satisfied and contributes no violation
                   (2.3.0+; units: mean/LSE per-donor squared unit-vector
                   residual after optimal rotation -- a value of ~0.02
                   corresponds to roughly an 8-degree worst-donor deviation
                   from the ideal polyhedron. Chosen to match how CShM is
                   used in the inorganic chemistry literature: near-zero
                   deviation isn't required, only "close enough to be the
                   same shape family". Starting default, not empirically
                   fit -- tune against the test-suite notebook.)
                   (default 0.02)
    sym_weight   : fixed weight on the symmetry-tying restraint that pulls
                   topologically-equivalent donors' M-D distances toward
                   their group mean (2.3.0+; see engine.py's RDKit
                   canonical-rank symmetric-donor detection). This is a soft
                   chemical prior (analogous to crystallography's SIMU/SAME
                   restraints), not a hard constraint, so it uses a fixed
                   weight rather than the dual-ascent machinery above.
                   (default 2.0)
    sym_angle_weight : fixed weight on the Q3 approach-angle symmetry-tying
                   restraint (2.4.0+; same symmetric_groups as sym_weight
                   above, but pulls each group's Q3-eligible members'
                   approach angles toward their group mean instead of their
                   M-D distances). Deliberately separate from sym_weight --
                   see the 2.4.0 changelog entry for why the two shouldn't
                   share one weight. Starting default, not empirically fit.
                   (default 1.0)
    mu_shape_stall_window : steps over which the shape violation must show
                   no improvement before mu_shape is frozen (2.6.0; only
                   used when mu_shape_freeze_on_stall). (default 20)
    mu_shape_stall_atol : violation range below which, over
                   mu_shape_stall_window steps, the violation counts as
                   stalled (2.6.0). (default 0.01)
    mu_shape_freeze_on_stall : bool, default True (2.6.0). Freezes
                   mu_shape's dual-ascent update once shape violation has
                   stalled (see 2.6.0 changelog) instead of letting it
                   grow without bound against a target that dual ascent
                   isn't actually closing. False restores pre-2.6.0
                   behaviour byte-identically.
    mu_shape_max : Optional[float], default None (2.6.0). Hard ceiling on
                   mu_shape, applied regardless of the stall detector, as
                   a defense-in-depth safety net against runaway growth.
    sym_weight_tracks_mu : bool, default True (2.6.0). SUPERSEDED by
                   sym_use_own_alm as the default mechanism (2.7.0); only
                   consulted when sym_use_own_alm=False. Scales sym_weight
                   and sym_angle_weight by (mu_shape / lambda_init) each
                   step so their influence relative to the shape penalty
                   doesn't decay as mu_shape grows (see 2.6.0 changelog).
                   False (with sym_use_own_alm also False) restores
                   pre-2.6.0 fixed-weight behaviour byte-identically.
    sym_use_own_alm : bool, default True (2.7.0). Promotes sym_penalty and
                   sym_angle_penalty to their own independent
                   safeguarded-ALM constraints (dual variables mu_sym,
                   mu_sym_angle, each escalating off its own violation --
                   see 2.7.0 changelog) instead of the 2.6.0
                   sym_weight_tracks_mu ratio. This is the recommended,
                   generalizable mechanism: it self-calibrates per case
                   with no hand-tuned constant, unlike a fixed weight or
                   fixed ratio fit against a handful of test cases. False
                   falls back to sym_weight_tracks_mu's behaviour.
    sym_rho      : Augmented-Lagrangian penalty/step-size for the
                   sym_use_own_alm dual ascent (2.7.0; shared by mu_sym
                   and mu_sym_angle, mirroring shape_rho's role for
                   mu_shape). (default 8.0)
    sym_tol      : violation tolerance below which sym_penalty /
                   sym_angle_penalty are considered satisfied (2.7.0;
                   mirrors shape_tol's role). Unlike shape_tol, there's no
                   literature-grounded nonzero target here -- a symmetry
                   tie is either satisfied or it isn't -- so the default
                   is 0.0 (drive fully to equality, subject to the stall
                   safeguard below once that's genuinely infeasible).
                   (default 0.0)
    sym_stall_window : steps over which sym_penalty's (and
                   sym_angle_penalty's) violation must show no
                   improvement before its dual variable is frozen (2.7.0;
                   only used when sym_freeze_on_stall). Same mechanism as
                   mu_shape_stall_window, kept as a separate parameter
                   since the two constraints' violations are on different
                   scales and may need different windows. (default 20)
    sym_stall_atol : violation range below which, over sym_stall_window
                   steps, sym_penalty's (or sym_angle_penalty's)
                   violation counts as stalled (2.7.0). Starting default,
                   not empirically fit -- tune against the test-suite
                   notebook if mu_sym/mu_sym_angle freeze too eagerly or
                   too late. (default 0.01)
    sym_freeze_on_stall : bool, default True (2.7.0). Freezes mu_sym's
                   (and independently, mu_sym_angle's) dual ascent once
                   that constraint's own violation has stalled, exactly
                   as mu_shape_freeze_on_stall does for mu_shape -- so an
                   infeasible symmetry tie can no longer grow its
                   multiplier without bound either.
    mu_sym_max   : Optional[float], default None (2.7.0). Hard ceiling on
                   both mu_sym and mu_sym_angle, applied regardless of the
                   stall detector, mirroring mu_shape_max.
    device       : torch device
    """

    def __init__(
        self,
        fks: list,
        scorer: "Scorer",
        ge: "GeometryExtractor",
        n_steps: int = 200,
        lr: float = 0.01,
        lambda_init: float = 10.0,
        lse_temp: float = 2.0,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        plateau_window: int = 8,
        plateau_atol: float = 0.02,
        anchor_floor_frac: float = 0.0,
        shape_rho: float = 8.0,
        shape_tol: float = 0.02,
        sym_weight: float = 2.0,
        sym_angle_weight: float = 1.0,
        mu_shape_stall_window: int = 20,
        mu_shape_stall_atol: float = 0.01,
        mu_shape_freeze_on_stall: bool = True,
        mu_shape_max: Optional[float] = None,
        sym_weight_tracks_mu: bool = True,
        sym_use_own_alm: bool = True,
        sym_rho: float = 8.0,
        sym_tol: float = 0.0,
        sym_stall_window: int = 20,
        sym_stall_atol: float = 0.01,
        sym_freeze_on_stall: bool = True,
        mu_sym_max: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.fks = fks
        self.scorer = scorer
        self.ge = ge
        self.n_steps = n_steps
        self.lr = lr
        self.lambda_init = lambda_init
        self.lse_temp = lse_temp
        self.adam_beta1, self.adam_beta2 = adam_betas
        self.adam_eps = 1e-8
        self.plateau_window = plateau_window
        self.plateau_atol = plateau_atol
        self.anchor_floor_frac = anchor_floor_frac
        self.shape_rho = shape_rho
        self.shape_tol = shape_tol
        self.sym_weight = sym_weight
        self.sym_angle_weight = sym_angle_weight
        self.mu_shape_stall_window = mu_shape_stall_window
        self.mu_shape_stall_atol = mu_shape_stall_atol
        self.mu_shape_freeze_on_stall = mu_shape_freeze_on_stall
        self.mu_shape_max = mu_shape_max
        self.sym_weight_tracks_mu = sym_weight_tracks_mu
        self.sym_use_own_alm = sym_use_own_alm
        self.sym_rho = sym_rho
        self.sym_tol = sym_tol
        self.sym_stall_window = sym_stall_window
        self.sym_stall_atol = sym_stall_atol
        self.sym_freeze_on_stall = sym_freeze_on_stall
        self.mu_sym_max = mu_sym_max
        self.device = device

    def solve(
        self,
        theta_inits: list,  # [Tensor(B, 6+Ji), ...]
        X_ref: torch.Tensor,  # (N, 3)
        donor_targets: list,  # [Tensor(B, n_d_i, 3), ...]
        donor_pos_idxs: list,  # [LongTensor(n_d_i), ...]
        debug: bool = False,
        ideal_directions: Optional[torch.Tensor] = None,  # (n_donors_total, 3)
        symmetric_groups: Optional[List[List[int]]] = None,
        q3_symmetric_groups: Optional[List[List[int]]] = None,
        q3_mode_commit: bool = True,
    ):
        """
        Run the joint IK solve for all ligands simultaneously.

        Anchor schedule (init-only)
        ---------------------------
        λ(t) = lambda_init · max(0, 1 − score_gain(t) / lambda_init)²
        where score_gain(t) = max(0, best_score_so_far − score_at_t0).
        The anchor is full strength at the start; as the score rises above
        its initial value, the anchor weight falls quadratically toward zero.
        The donor anchor targets are ideal-octahedron vertices — correct for
        seeding the basin, not for the final geometry.

        Parameters
        ----------
        theta_inits    : list of (B, 6+Ji) initial DOF tensors, one per ligand.
                         All must have the same batch size B.
        X_ref          : (N, 3) full-complex reference atom positions.
        donor_targets  : list of (B, n_d_i, 3) target tensors, one per ligand.
                         Still used to seed kabsch_init() before solve() is
                         called; inside solve() itself it now only feeds the
                         OLD anchor fallback path (ideal_directions=None).
        donor_pos_idxs : list of (n_d_i,) LongTensors of pos-indices into X_ref.
        debug          : print per-step diagnostics.
        ideal_directions : (n_donors_total, 3) fixed unit vectors, one per
                         donor in ge.donor_positions()'s global donor order
                         (= ideal[assignment[di]] from engine.py's Hungarian
                         assignment). When given, activates the 2.3.0
                         Augmented-Lagrangian shape constraint in place of
                         the old fixed-target anchor (see class docstring).
        symmetric_groups : list of donor-index groups (each a list of
                         indices into the same global donor order as
                         ideal_directions) that are topologically equivalent
                         (RDKit canonical-rank match, engine.py) and should
                         be pulled toward a common M-D distance. Optional;
                         only meaningful alongside ideal_directions, though
                         harmless without it.
        q3_symmetric_groups : list of donor-index groups, same index
                         convention as symmetric_groups. Optional (2.8.0);
                         when given, used IN PLACE OF symmetric_groups for
                         the Q3 approach-angle tie and q3_mode_commit only
                         -- the M-D bond-length tie (sym_weight) always uses
                         the unrestricted symmetric_groups. When None
                         (default), falls back to symmetric_groups, i.e.
                         pre-2.8.0 behaviour. Callers should pass a
                         within-single-ligand-instance-only subset here;
                         see engine.py 2.12.0 changelog for why the Q3 tie
                         needs a narrower grouping than the bond-length tie.
        q3_mode_commit  : bool, default True (2.5.0). When True and a
                         q3_symmetric_groups (or, if not given,
                         symmetric_groups) group has >=2 Q3-eligible
                         members, commit that group to a single GMM
                         component (chosen from the initial pose) for the
                         whole solve, instead of letting each member
                         independently draw from the full (possibly
                         multi-modal) Q3 mixture. See class 2.5.0 and 2.8.0
                         changelogs. False restores pre-2.5.0 behaviour
                         byte-identically -- useful for A/B comparison.

        Returns
        -------
        thetas_final   : list of (B, 6+Ji) — DOFs giving best honest score seen
        converged      : (B,) bool — True iff score plateaued
        final_score    : (B,) — honest score at the best-seen positions
        solve_trace    : dict of per-step numpy arrays (for diagnostics)
        """
        device = self.device
        B = theta_inits[0].shape[0]
        T = self.n_steps
        L = len(self.fks)

        X_ref = X_ref.to(device)
        donor_targets = [dt.to(device) for dt in donor_targets]
        donor_pos_idxs = [idx.to(device) for idx in donor_pos_idxs]

        # One leaf theta per ligand — all optimised jointly
        thetas = [ti.detach().to(device).requires_grad_(True) for ti in theta_inits]

        # Adam state — single optimiser for the combined gradient.
        # Per-DOF lr adaptation compensates for gradient dilution in the joint
        # score: a rotation DOF receiving small c_grad gets proportionally more
        # effective lr than one with large c_grad, making convergence rate
        # independent of complex size (no separate chem_lr needed).
        adam_m = [torch.zeros_like(ti) for ti in thetas]
        adam_v = [torch.zeros_like(ti) for ti in thetas]

        _debug_steps = {0, T // 4, T // 2, 3 * T // 4, T - 1} if debug else set()

        # Single-tier selection: best honest score seen, no gate.
        # The old two-tier (best_score + best_at_plateau) is removed because
        # the plateau snapshot is only useful if it's different from best_score,
        # which requires a non-trivial gap — and we can't know that in advance.
        # Best-score-ever is the simplest defensible choice.
        best_score_ever = torch.full((B,), float("-inf"), device=device)
        best_score_thetas = [ti.detach().clone() for ti in thetas]

        # Score plateau: ring buffer of recent scores (B,).
        # Convergence = range over last plateau_window steps < plateau_atol.
        W = self.plateau_window
        score_ring = torch.full((W, B), float("-inf"), device=device)
        ever_plateaued = torch.zeros(B, dtype=torch.bool, device=device)

        # Shape-violation stall detector (2.6.0) -- same ring-buffer
        # pattern as score_ring above, but on `violation` instead of
        # score, used to freeze mu_shape's dual ascent once it's
        # demonstrably not reducing the constraint violation any further
        # (see class 2.6.0 changelog). Only meaningful on the shape-ALM
        # path (ideal_directions given); harmless no-op otherwise.
        MW = self.mu_shape_stall_window
        violation_ring = torch.full((MW, B), float("inf"), device=device)
        mu_shape_frozen = torch.zeros(B, dtype=torch.bool, device=device)

        # Same stall-ring pattern, independently, for the symmetry-tying
        # constraints' own dual variables (2.7.0 sym_use_own_alm -- see
        # class 2.7.0 changelog). Harmless no-op when sym_use_own_alm is
        # False or a case has no symmetric_groups (sym_penalty stays 0,
        # so violation stays 0 and these never trigger).
        SW = self.sym_stall_window
        sym_violation_ring = torch.full((SW, B), float("inf"), device=device)
        sym_angle_violation_ring = torch.full((SW, B), float("inf"), device=device)
        mu_sym_frozen = torch.zeros(B, dtype=torch.bool, device=device)
        mu_sym_angle_frozen = torch.zeros(B, dtype=torch.bool, device=device)

        # Score at step 0 — used to compute score_gain for anchor decay.
        # Initialised to -inf; actual value set on first forward pass.
        score_t0 = torch.full((B,), float("-inf"), device=device)

        history_residuals = []
        history_scores = []
        history_anchors = []
        history_grad_norms = []
        history_mu_shape = []
        history_sym_penalty = []
        history_sym_angle_penalty = []
        history_mu_sym = []
        history_mu_sym_angle = []

        # Augmented-Lagrangian dual variable for the shape constraint (only
        # meaningful when ideal_directions is not None -- see 2.3.0
        # changelog above). Initialised at lambda_init for continuity with
        # the old anchor's initial strength; unlike the old fixed lambda(t)
        # schedule it then self-adjusts via dual ascent every step instead
        # of decaying on a fixed curve.
        mu_shape = torch.full((B,), self.lambda_init, device=device)

        # Augmented-Lagrangian dual variables for the symmetry-tying
        # constraints (2.7.0 sym_use_own_alm), initialised from
        # sym_weight/sym_angle_weight for continuity with their pre-2.6.0
        # meaning as starting weights -- see 2.7.0 changelog.
        mu_sym = torch.full((B,), self.sym_weight, device=device)
        mu_sym_angle = torch.full((B,), self.sym_angle_weight, device=device)
        ideal_directions_dev = (
            ideal_directions.to(device) if ideal_directions is not None else None
        )

        # Site-index -> Q3-array-index map (2.4.0): ge.q3_approach_angles()
        # only returns Q3-eligible sites' angles, in site-index order (same
        # convention ik/scorer.py's Q3 construction loop relies on), so this
        # recovers which q3-array slot (if any) each symmetric_groups donor
        # index corresponds to. Static (depends only on ge._q3_eligible), so
        # built once here rather than every step of the T-loop below.
        # 2.8.0: use the caller's narrower q3_symmetric_groups for the Q3
        # tie/mode-commit when given, else fall back to symmetric_groups
        # (pre-2.8.0 behaviour) -- see engine.py 2.12.0 changelog.
        _q3_groups = (
            q3_symmetric_groups if q3_symmetric_groups is not None else symmetric_groups
        )

        q3_site_to_idx: Dict[int, int] = {}
        if symmetric_groups:
            _q3_running = 0
            for _s_i, _elig in enumerate(self.ge._q3_eligible):
                if _elig:
                    q3_site_to_idx[_s_i] = _q3_running
                    _q3_running += 1

        # Mode-committed Q3 initialization (2.5.0) -- see class changelog.
        # q3_mode_override: (B, n_q3) int64, -1 = no override (default,
        # scorer.py 2.8.0 no-ops on this). Computed ONCE here from the
        # initial pose (theta_inits, before any gradient steps) and held
        # fixed for the whole trajectory -- deliberately not recomputed
        # every step, since the point is a single up-front commitment per
        # trial, not a moving target.
        q3_mode_override: Optional[torch.Tensor] = None
        if q3_mode_commit and _q3_groups and q3_site_to_idx:
            q3_means_static = getattr(self.scorer, "_q3_means", None)
            q3_eff_vars_static = getattr(self.scorer, "_q3_eff_vars", None)
            q3_log_w_static = getattr(self.scorer, "_q3_log_w", None)
            if (
                q3_means_static is not None
                and q3_means_static.shape[0] > 0
                and q3_eff_vars_static is not None
                and q3_log_w_static is not None
            ):
                with torch.no_grad():
                    X_init = X_ref.unsqueeze(0).expand(B, -1, -1).clone()
                    for _fk_li, _theta_li in zip(self.fks, thetas):
                        X_init = _fk_li.apply(_theta_li, X_init)
                    q3_cos_init = torch.cos(self.ge.q3_approach_angles(X_init))  # (B, n_q3)
                    n_q3_total = q3_means_static.shape[0]
                    override = torch.full(
                        (B, n_q3_total), -1, dtype=torch.long, device=device
                    )
                    any_group_qualifies = False
                    for group in _q3_groups:
                        q3_idxs = [
                            q3_site_to_idx[g] for g in group if g in q3_site_to_idx
                        ]
                        if len(q3_idxs) < 2:
                            continue
                        any_group_qualifies = True
                        idx_t = torch.as_tensor(q3_idxs, dtype=torch.long, device=device)
                        obs_g = q3_cos_init[:, idx_t]  # (B, G)
                        m_g = q3_means_static[idx_t]  # (G, K)
                        v_g = q3_eff_vars_static[idx_t]  # (G, K)
                        lw_g = q3_log_w_static[idx_t]  # (G, K)
                        # Un-normalised log-density is enough here (only the
                        # argmax over k matters, and the -0.5*log(2*pi) term
                        # is a k-independent constant that drops out).
                        diff_g = obs_g.unsqueeze(-1) - m_g.unsqueeze(0)  # (B, G, K)
                        log_comp_g = (
                            -0.5 * torch.log(v_g).unsqueeze(0)
                            - 0.5 * diff_g**2 / v_g.unsqueeze(0)
                        )
                        log_p_g = log_comp_g + lw_g.unsqueeze(0)  # (B, G, K)
                        group_log_p_per_k = log_p_g.sum(dim=1)  # (B, K) -- total if ALL members forced to k
                        best_k = group_log_p_per_k.argmax(dim=-1)  # (B,)
                        override[:, idx_t] = best_k.unsqueeze(-1).expand(-1, len(q3_idxs))
                    if any_group_qualifies:
                        q3_mode_override = override
                    if debug and any_group_qualifies:
                        print(
                            f"\n[DEBUG]   Q3 mode-commit (2.5.0): committed "
                            f"{int((override[0] >= 0).sum().item())} of {n_q3_total} "
                            f"Q3 terms for trial 0 -- modes {override[0][override[0] >= 0].tolist()}"
                        )

        if debug:
            print(
                f"\n[DEBUG]   JointIKSolver  L={L}  B={B}  T={T}  "
                f"lr={self.lr}  λ_init={self.lambda_init}  "
                f"lse_temp={self.lse_temp}  "
                f"plateau_window={W}  plateau_atol={self.plateau_atol}"
            )

        for t in range(T):
            # --- Score-driven anchor decay (init-only anchor) ---
            # λ(t) = lambda_init · clamp(1 − gain/lambda_init, 0, 1)²
            # where gain = max(0, best_score_so_far − score_at_step_0).
            # At t=0, gain=0 → λ = lambda_init (full anchor, seeds the basin).
            # As score rises, gain grows, λ falls toward 0 (score owns geometry).
            # score_t0 is initialised to -inf; the first forward pass sets it.
            # nan_to_num guards the -inf - (-inf) = nan edge case on step 0.
            with torch.no_grad():
                gain = torch.nan_to_num(best_score_ever - score_t0, nan=0.0).clamp(min=0.0)
                frac = (1.0 - gain / max(self.lambda_init, _EPS)).clamp(min=0.0)
                floor = self.anchor_floor_frac
                lambda_eff = self.lambda_init * (frac**2 * (1.0 - floor) + floor)  # (B,)

            # --- Forward: apply all FKs sequentially to build X_current ---
            X_current = X_ref.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N, 3)
            for li, (fk_li, theta_li) in enumerate(zip(self.fks, thetas)):
                X_current = fk_li.apply(theta_li, X_current)  # (B, N, 3)

            # --- Unified score: one call, one number, honest objective ---
            stat_score, _ = self.scorer.score(X_current, q3_mode_override=q3_mode_override)
            # stat_score : (B,)  — the scalar optimised and reported

            # Seed score_t0 on first iteration
            if t == 0:
                score_t0 = stat_score.detach().clone()

            # --- Shape / anchor term (2.3.0) ---
            # Two mechanisms, selected once per call by whether the caller
            # supplied ideal_directions (see class docstring / changelog):
            #  * NEW (ideal_directions given): Augmented-Lagrangian shape
            #    constraint via a per-step-recomputed optimal rotation
            #    (Procrustes/Kabsch, rotation-only) between the fixed ideal
            #    polyhedron directions and the current donor unit vectors,
            #    plus a small symmetry-tying restraint on topologically
            #    equivalent donors (symmetric_groups).
            #  * OLD (ideal_directions is None, backward-compat fallback):
            #    fixed absolute-Cartesian-target anchor with the original
            #    lambda(t) decay schedule, unchanged from pre-2.3.0.
            all_donor_pos_grad = self.ge.donor_positions(X_current)  # (B, n_donors, 3)
            all_donor_pos_det = all_donor_pos_grad.detach()
            sym_penalty = X_ref.new_zeros(B)
            sym_angle_penalty = X_ref.new_zeros(B)

            if ideal_directions_dev is not None:
                donor_unit_vecs = F.normalize(all_donor_pos_grad, dim=-1)  # (B, n_donors, 3)

                # Optimal rotation recomputed every step from the CURRENT
                # donor arrangement, under no_grad and detached: by the
                # envelope theorem d(loss)/dR = 0 at R's own optimum, so
                # detaching it costs nothing to first order while avoiding
                # SVD-backward instability on near-degenerate inputs.
                with torch.no_grad():
                    R_opt = _optimal_rotation_nocentroid(
                        ideal_directions_dev, donor_unit_vecs.detach()
                    )  # (B, 3, 3): R_opt @ ideal_i ~= donor_unit_i

                aligned_ideal = torch.einsum(
                    "bij,nj->bni", R_opt, ideal_directions_dev
                )  # (B, n_donors, 3)
                per_donor_sq = ((aligned_ideal - donor_unit_vecs) ** 2).sum(-1)  # (B, n_donors)

                n_sh = per_donor_sq.shape[-1]
                if self.lse_temp > 0.0 and n_sh > 1:
                    tau = self.lse_temp
                    shape_dev = torch.logsumexp(tau * per_donor_sq, dim=-1) / tau  # (B,)
                else:
                    shape_dev = per_donor_sq.mean(dim=-1)

                violation = (shape_dev - self.shape_tol).clamp(min=0.0)  # (B,)
                constraint_loss = (
                    mu_shape.detach() * violation + 0.5 * self.shape_rho * violation**2
                )
                anchor_loss = shape_dev  # trace/back-compat alias (see solve_trace)

                if symmetric_groups:
                    all_donor_norms = all_donor_pos_grad.norm(dim=-1)  # (B, n_donors)
                    n_groups = 0
                    for group in symmetric_groups:
                        if len(group) < 2:
                            continue
                        idx_t = torch.as_tensor(group, dtype=torch.long, device=device)
                        d_g = all_donor_norms[:, idx_t]  # (B, len(group))
                        dev_g = d_g - d_g.mean(dim=-1, keepdim=True)
                        sym_penalty = sym_penalty + (dev_g**2).mean(dim=-1)
                        n_groups += 1
                    if n_groups > 0:
                        sym_penalty = sym_penalty / n_groups

                    # --- Q3 approach-angle symmetry tie (2.4.0) ---------------
                    # See 2.4.0 changelog: the distance tie above leaves each
                    # donor's own Q3 approach-angle outcome as an independent
                    # draw, with nothing preventing two topologically-
                    # equivalent donors (e.g. EDTA's two amine N) from landing
                    # near different modes of a (now genuinely differentiated,
                    # post Q2/Q3-refit) Q3 distribution. Mirrors the distance
                    # tie's pattern exactly, just on a different observable.
                    if q3_site_to_idx:
                        q3_angles_now = None  # lazy: skip the extra fwd pass if no group qualifies
                        n_angle_groups = 0
                        for group in _q3_groups:
                            q3_idxs = [q3_site_to_idx[g] for g in group if g in q3_site_to_idx]
                            if len(q3_idxs) < 2:
                                continue
                            if q3_angles_now is None:
                                q3_angles_now = self.ge.q3_approach_angles(X_current)  # (B, n_q3) rad
                            idx_t2 = torch.as_tensor(q3_idxs, dtype=torch.long, device=device)
                            a_g = q3_angles_now[:, idx_t2]  # (B, len(q3_idxs))
                            dev_a = a_g - a_g.mean(dim=-1, keepdim=True)
                            sym_angle_penalty = sym_angle_penalty + (dev_a**2).mean(dim=-1)
                            n_angle_groups += 1
                        if n_angle_groups > 0:
                            sym_angle_penalty = sym_angle_penalty / n_angle_groups

                # --- Symmetry-tying constraints (2.7.0) ---------------------
                # sym_use_own_alm (default, see 2.7.0 changelog): sym_penalty
                # and sym_angle_penalty each get their OWN safeguarded-ALM
                # dual variable (mu_sym, mu_sym_angle) that escalates off its
                # OWN violation -- self-calibrating per case, no hand-tuned
                # constant. sym_violation/sym_angle_violation are computed
                # unconditionally (cheap) so the dual-ascent block below can
                # use them regardless of which branch was active this step.
                sym_violation = (sym_penalty - self.sym_tol).clamp(min=0.0)
                sym_angle_violation = (sym_angle_penalty - self.sym_tol).clamp(min=0.0)

                if self.sym_use_own_alm:
                    constraint_loss = (
                        constraint_loss
                        + mu_sym.detach() * sym_violation
                        + 0.5 * self.sym_rho * sym_violation**2
                        + mu_sym_angle.detach() * sym_angle_violation
                        + 0.5 * self.sym_rho * sym_angle_violation**2
                    )
                elif self.sym_weight_tracks_mu:
                    # 2.6.0 fixed-ratio heuristic, retained for exact
                    # backward compat when sym_use_own_alm=False -- see
                    # 2.6.0 changelog (superseded by sym_use_own_alm above).
                    mu_scale = (mu_shape.detach() / self.lambda_init).clamp(min=1.0)
                    eff_sym_weight = self.sym_weight * mu_scale
                    eff_sym_angle_weight = self.sym_angle_weight * mu_scale
                    constraint_loss = (
                        constraint_loss
                        + eff_sym_weight * sym_penalty
                        + eff_sym_angle_weight * sym_angle_penalty
                    )
                else:
                    # pre-2.6.0 fixed-weight behaviour, byte-identical.
                    constraint_loss = (
                        constraint_loss
                        + self.sym_weight * sym_penalty
                        + self.sym_angle_weight * sym_angle_penalty
                    )
            else:
                # --- OLD anchor: fixed absolute-Cartesian target, LSE soft-max ---
                anchor_loss = X_ref.new_zeros(B)
                for li, (didx, dt) in enumerate(zip(donor_pos_idxs, donor_targets)):
                    donor_pos_li = all_donor_pos_grad[:, didx, :]  # (B, n_d_i, 3)
                    diff_li = donor_pos_li - dt  # (B, n_d_i, 3)
                    sq_li = (diff_li**2).sum(-1)  # (B, n_d_i)

                    if self.lse_temp > 0.0 and sq_li.shape[1] > 1:
                        tau = self.lse_temp
                        lse_li = torch.logsumexp(tau * sq_li, dim=-1) / tau  # (B,)
                        anchor_loss = anchor_loss + lse_li / sq_li.shape[1]
                    else:
                        anchor_loss = anchor_loss + sq_li.mean(-1)

                anchor_loss = anchor_loss / L
                # lambda_eff (score-driven decay schedule) is computed above,
                # before the forward pass -- used unchanged on this path only.
                constraint_loss = lambda_eff * anchor_loss

            # --- Plateau detection (convergence = score stopped improving) ---
            # Check only every plateau_window steps to avoid expensive
            # (W, B) max/min reductions on every iteration.
            score_ring[t % W] = stat_score.detach()
            if t >= W - 1 and t % W == W - 1:
                score_range = score_ring.max(dim=0).values - score_ring.min(dim=0).values
                ever_plateaued = ever_plateaued | (score_range < self.plateau_atol)

            # --- Best-score snapshot (single tier, no gate) ---
            with torch.no_grad():
                improved = stat_score > best_score_ever
                best_score_ever = torch.where(improved, stat_score, best_score_ever)
                sel = improved.view(-1, *([1] * (thetas[0].dim() - 1)))
                for li, theta_li in enumerate(thetas):
                    best_score_thetas[li] = torch.where(
                        sel.expand_as(theta_li),
                        theta_li.detach(),
                        best_score_thetas[li],
                    )

            # --- Single combined backward pass ---
            # Combined loss: -score + λ·anchor, computed in one expression
            # so a single backward() traversal covers both gradients.
            # This replaces the previous retain_graph=True + second backward,
            # saving one full graph traversal per step (~1.5× speedup).
            combined_loss = (-stat_score + constraint_loss).sum()  # scalar
            grads_all = torch.autograd.grad(combined_loss, thetas, allow_unused=True)

            total_grads: List[torch.Tensor] = []
            for li, (didx, theta_li) in enumerate(zip(donor_pos_idxs, thetas)):
                g = grads_all[li] if grads_all[li] is not None else torch.zeros_like(theta_li)

                n_d_i = didx.shape[0]
                if n_d_i == 2:
                    # Bidentate: null-space projector on the anchor rotation
                    # gradient component.  The donor-donor axis is the unique
                    # rotation null direction; P_keep = I − n̂n̂ᵀ zeroes only
                    # that component.  Closed-form (no autograd): ~2000× faster.
                    #
                    # With combined loss the gradient mixes anchor and score
                    # contributions.  We project only the base rotation DOFs
                    # (indices 3:6) — the torsion DOFs (6:) are unaffected by
                    # the projector, and the score gradient's rotation component
                    # along the null axis is ~0 by construction (any rotation
                    # about the donor–donor axis leaves donor positions and hence
                    # the score unchanged).
                    with torch.no_grad():
                        d_pos = all_donor_pos_det[:, didx, :]  # (B, 2, 3)
                        null_vec = d_pos[:, 0, :] - d_pos[:, 1, :]  # (B, 3)
                        null_norm = null_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        n_hat = null_vec / null_norm
                        I3 = (
                            torch.eye(3, dtype=n_hat.dtype, device=n_hat.device)
                            .unsqueeze(0)
                            .expand(B, -1, -1)
                        )
                        P_keep = I3 - torch.bmm(n_hat.unsqueeze(-1), n_hat.unsqueeze(-2))
                    omega_proj = torch.bmm(P_keep, g[:, 3:6].unsqueeze(-1)).squeeze(-1)
                    g = g.clone()
                    g[:, 3:6] = omega_proj
                # n_d_i >= 3: P_keep = I (no-op).
                # n_d_i in {0,1}: anchor rotational gradient ~0 by construction.

                total_grads.append(g)

            r = -stat_score + constraint_loss  # (B,) for logging

            # --- Record trace (on-device, detached) ---
            with torch.no_grad():
                grad_norm_b = torch.cat([g.reshape(B, -1) for g in total_grads], dim=-1).norm(
                    dim=-1
                )  # (B,)

                history_residuals.append(r.detach())
                history_scores.append(stat_score.detach())
                history_anchors.append(anchor_loss.detach())
                history_grad_norms.append(grad_norm_b)
                history_mu_shape.append(mu_shape.detach().clone())
                history_sym_penalty.append(sym_penalty.detach())
                history_sym_angle_penalty.append(sym_angle_penalty.detach())
                history_mu_sym.append(mu_sym.detach().clone())
                history_mu_sym_angle.append(mu_sym_angle.detach().clone())

                # --- mu_shape stall safeguard (2.6.0) -----------------------
                # Classical dual ascent (mu += rho*violation every step,
                # unconditionally) assumes escalating mu will eventually
                # drive violation to zero. When the constraint is
                # effectively infeasible, violation instead plateaus at
                # some floor > 0 and mu grows WITHOUT BOUND -- see class
                # 2.6.0 changelog for the EDTA evidence. Standard
                # safeguarded-ALM fix: once violation stops improving over
                # a window, stop escalating the penalty for it.
                if ideal_directions_dev is not None and self.mu_shape_freeze_on_stall:
                    violation_ring[t % MW] = violation.detach()
                    if t >= MW - 1 and t % MW == MW - 1:
                        v_range = (
                            violation_ring.max(dim=0).values - violation_ring.min(dim=0).values
                        )
                        mu_shape_frozen = mu_shape_frozen | (v_range < self.mu_shape_stall_atol)

                # Dual ascent: mu_shape self-adjusts to the violation just
                # observed this step, UNLESS frozen by the stall safeguard
                # above. No-op on the old-anchor fallback path (violation
                # is only defined when ideal_directions is given).
                if ideal_directions_dev is not None:
                    mu_increment = self.shape_rho * violation.detach()
                    mu_increment = torch.where(
                        mu_shape_frozen, torch.zeros_like(mu_increment), mu_increment
                    )
                    mu_shape = (mu_shape + mu_increment).clamp(min=0.0)
                    if self.mu_shape_max is not None:
                        mu_shape = mu_shape.clamp(max=self.mu_shape_max)

                # --- sym ALM stall safeguard + dual ascent (2.7.0) ----------
                # Identical mechanism to mu_shape's above, applied
                # independently to mu_sym and mu_sym_angle -- see 2.7.0
                # changelog. No-op (rings stay at their init value, dual
                # ascent increments stay ~0) whenever sym_use_own_alm=False
                # or a case has no symmetric_groups.
                if ideal_directions_dev is not None and self.sym_use_own_alm and self.sym_freeze_on_stall:
                    sym_violation_ring[t % SW] = sym_violation.detach()
                    sym_angle_violation_ring[t % SW] = sym_angle_violation.detach()
                    if t >= SW - 1 and t % SW == SW - 1:
                        sym_v_range = (
                            sym_violation_ring.max(dim=0).values
                            - sym_violation_ring.min(dim=0).values
                        )
                        sym_angle_v_range = (
                            sym_angle_violation_ring.max(dim=0).values
                            - sym_angle_violation_ring.min(dim=0).values
                        )
                        mu_sym_frozen = mu_sym_frozen | (sym_v_range < self.sym_stall_atol)
                        mu_sym_angle_frozen = mu_sym_angle_frozen | (
                            sym_angle_v_range < self.sym_stall_atol
                        )

                if ideal_directions_dev is not None and self.sym_use_own_alm:
                    mu_sym_increment = self.sym_rho * sym_violation.detach()
                    mu_sym_angle_increment = self.sym_rho * sym_angle_violation.detach()
                    mu_sym_increment = torch.where(
                        mu_sym_frozen, torch.zeros_like(mu_sym_increment), mu_sym_increment
                    )
                    mu_sym_angle_increment = torch.where(
                        mu_sym_angle_frozen,
                        torch.zeros_like(mu_sym_angle_increment),
                        mu_sym_angle_increment,
                    )
                    mu_sym = (mu_sym + mu_sym_increment).clamp(min=0.0)
                    mu_sym_angle = (mu_sym_angle + mu_sym_angle_increment).clamp(min=0.0)
                    if self.mu_sym_max is not None:
                        mu_sym = mu_sym.clamp(max=self.mu_sym_max)
                        mu_sym_angle = mu_sym_angle.clamp(max=self.mu_sym_max)

            if t in _debug_steps and debug:
                with torch.no_grad():
                    total_gn = sum(g.norm().item() for g in total_grads)
                print(
                    f"  [t={t:>4}]  λ_eff min={lambda_eff.min().item():.4f}"
                    f"  max={lambda_eff.max().item():.4f}"
                    f"  score min={stat_score.min().item():.4f}"
                    f"  max={stat_score.max().item():.4f}"
                    f"  nan={torch.isnan(stat_score).any().item()}"
                    f"  |  plateaued={int(ever_plateaued.sum().item())}/{B}"
                    f"  |  total_grad_norm={total_gn:.4f}"
                )

            # --- Adam update on combined gradient ---
            # Single optimiser for the combined (anchor + chemistry) gradient.
            # Adam's per-DOF lr adaptation compensates for gradient dilution:
            # a DOF with small total gradient gets a proportionally larger
            # effective lr, making convergence rate complex-size independent.
            adam_t = t + 1
            b1, b2, eps_a = self.adam_beta1, self.adam_beta2, self.adam_eps
            with torch.no_grad():
                for li, (theta_li, g) in enumerate(zip(thetas, total_grads)):
                    adam_m[li] = b1 * adam_m[li] + (1.0 - b1) * g
                    adam_v[li] = b2 * adam_v[li] + (1.0 - b2) * g**2
                    m_hat = adam_m[li] / (1.0 - b1**adam_t)
                    v_hat = adam_v[li] / (1.0 - b2**adam_t)
                    step = self.lr * m_hat / (v_hat.sqrt() + eps_a)
                    theta_li.data -= step
                    if theta_li.shape[1] > 6:
                        theta_li.data[:, 6:] = wrap_angle(theta_li.data[:, 6:])
                    theta_li.data[:, 3:6] = _renorm_axisangle(theta_li.data[:, 3:6])

        # --- Post-loop: assemble best-score positions and report ---
        # Single tier: best honest score seen, no gate, no template distance.
        # Convergence = score plateaued at any point during optimisation.
        best_thetas = best_score_thetas  # list of (B, 6+Ji)

        with torch.no_grad():
            X_final = X_ref.unsqueeze(0).expand(B, -1, -1).clone()
            for fk_li, theta_li in zip(self.fks, best_thetas):
                X_final = fk_li.apply(theta_li, X_final)

            final_score, _ = self.scorer.score(X_final, q3_mode_override=q3_mode_override)  # (B,)
            # Convergence = score plateaued (honest criterion).
            # A trajectory that plateaued at a good score is converged;
            # one that never plateaued (still improving at T) is not.
            converged = ever_plateaued

        if debug:
            n_conv = int(converged.sum().item())
            n_plat = int(ever_plateaued.sum().item())
            print(
                f"\n[DEBUG]   JointIKSolver done: {n_conv}/{B} converged "
                f"(plateau criterion)  "
                f"best_score min={best_score_ever.min().item():.4f}  "
                f"final_score min={final_score.min().item():.4f}  "
                f"max={final_score.max().item():.4f}  "
                f"nan={torch.isnan(final_score).any().item()}"
            )

        solve_trace = {}
        if history_residuals:

            solve_trace = {
                "residual": torch.stack(history_residuals, dim=0).cpu().numpy(),  # (T, B)
                "score": torch.stack(history_scores, dim=0).cpu().numpy(),  # (T, B)
                # anchor_loss: OLD path -> fixed-target squared error (as before);
                # NEW path (ideal_directions given) -> shape_dev, the raw
                # (pre-multiplier) shape-deviation diagnostic. See 2.3.0 changelog.
                "anchor_loss": torch.stack(history_anchors, dim=0).cpu().numpy(),  # (T, B)
                "grad_norm": torch.stack(history_grad_norms, dim=0).cpu().numpy(),  # (T, B)
                "mu_shape": torch.stack(history_mu_shape, dim=0).cpu().numpy(),  # (T, B)
                "sym_penalty": torch.stack(history_sym_penalty, dim=0).cpu().numpy(),  # (T, B)
                "sym_angle_penalty": torch.stack(history_sym_angle_penalty, dim=0).cpu().numpy(),  # (T, B)
                "mu_sym": torch.stack(history_mu_sym, dim=0).cpu().numpy(),  # (T, B)
                "mu_sym_angle": torch.stack(history_mu_sym_angle, dim=0).cpu().numpy(),  # (T, B)
            }

        thetas_final = [th.detach() for th in best_thetas]
        return thetas_final, converged, final_score, solve_trace