"""
ik_solver.py
============
Gradient-descent IK solver for the conformer ensemble engine.

Jointly optimises every ligand's degrees of freedom against a combined
objective: the Scorer's statistical score, an Augmented-Lagrangian shape
constraint that pulls donors toward the best-fit rotation of the ideal
coordination polyhedron (recomputed every step via Procrustes/Kabsch), and
symmetry-tying restraints that keep topologically-equivalent donors (e.g.
two chemically identical amine nitrogens) consistent with each other on
M-D distance and Q3 approach angle. All three constraint families use
safeguarded dual ascent (Birgin & Martinez-style Augmented Lagrangian):
each dual variable escalates its own penalty until its constraint is
satisfied or demonstrably stalled, then freezes rather than growing
without bound. See `IKSolverConfig` for the tunable hyperparameters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
    Initialise base DOF (translation + axis-angle) for each batch item via
    Kabsch alignment of current donor positions to targets.

    For a single donor or collinear donors, the rigid-body fit leaves one or
    more rotational DOFs unconstrained; those are filled with a random
    rotation so different batch trials explore different orientations. For
    3+ non-collinear donors the fit is uniquely determined and every trial
    would otherwise get an identical starting orientation regardless of
    beam width — `diversify_orientation=True` perturbs that well-posed fit
    with an independent random rotation per trial as well, so the batch
    actually explores distinct basins for flexible, torsion-jointed
    ligands. `diversity_scale` caps the perturbation angle in units of pi.
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
            # rank >= 2: the fit is well-posed (unique optimum) but that
            # uniqueness collapses batch diversity for flexible multi-donor
            # ligands -- see docstring above. Perturb with an independent
            # random rotation about a random axis, scaled by diversity_scale.
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
    vectors (both point sets share the metal atom as a common physical
    origin, so subtracting a centroid would break that physical meaning).

    For each batch element b, finds the rotation R[b] minimising
    sum_i || R[b] @ A[i] - B[b, i] ||^2.

    Parameters
    ----------
    A : (n, 3) fixed reference unit vectors (ideal polyhedron directions),
        static across the whole solve.
    B : (Batch, n, 3) current unit vectors, recomputed every step.

    Returns
    -------
    R : (Batch, 3, 3) rotation matrices, det(R) = +1.

    Note: callers always invoke this under torch.no_grad() and backprop
    only through the residual against the detached R — by the envelope
    theorem this is exact to first order at R's own optimum, and it
    sidesteps the numerical instability of backpropping through SVD on
    near-degenerate inputs.
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


@dataclass
class IKSolverConfig:
    """
    Tunable hyperparameters for JointIKSolver.

    Core optimisation
    -----------------
    n_steps, lr, adam_betas   : gradient-descent step count and Adam
        learning rate / momentum for the joint per-ligand DOF optimisation.
    plateau_window, plateau_atol : convergence is declared once the score
        stops improving by more than plateau_atol over plateau_window steps.
    lambda_init  : initial value of the shape constraint's dual variable
        mu_shape, and (via mu_sym/mu_sym_angle init below) the starting
        point for the symmetry-tie multipliers too.
    lse_temp     : temperature for the log-sum-exp soft-max used to
        aggregate per-donor shape-deviation residuals into one scalar.
        Higher -> closer to a plain mean; lower -> closer to a hard max
        on the single worst-fit donor.

    Shape constraint (donors pulled toward the best-fit rotation of the
    ideal coordination polyhedron, recomputed every step)
    ----------------------------------------------------------------------
    shape_rho, shape_tol      : Augmented-Lagrangian step size and
        satisfied-constraint tolerance.
    mu_shape_stall_window, mu_shape_stall_atol, mu_shape_freeze_on_stall :
        freeze mu_shape's escalation once the violation stops improving,
        so an unreachable target can't drive it up without bound.
    mu_shape_max : optional hard ceiling on mu_shape.

    Symmetry ties (topologically-equivalent donors pulled toward a common
    M-D distance and Q3 approach angle)
    ----------------------------------------------------------------------
    sym_weight, sym_angle_weight : initial value of mu_sym / mu_sym_angle.
    sym_rho, sym_tol             : Augmented-Lagrangian step size and
        satisfied-constraint tolerance, shared by both dual variables.
    sym_stall_window, sym_stall_atol, sym_freeze_on_stall : same
        stall-freeze safeguard as the shape constraint, applied
        independently to mu_sym and mu_sym_angle.
    mu_sym_max : optional hard ceiling on both mu_sym and mu_sym_angle.
    """

    n_steps: int = 200
    lr: float = 0.01
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    plateau_window: int = 8
    plateau_atol: float = 0.02
    lambda_init: float = 10.0
    lse_temp: float = 2.0

    shape_rho: float = 8.0
    shape_tol: float = 0.02
    mu_shape_stall_window: int = 20
    mu_shape_stall_atol: float = 0.01
    mu_shape_freeze_on_stall: bool = True
    mu_shape_max: Optional[float] = None

    sym_weight: float = 2.0
    sym_angle_weight: float = 1.0
    sym_rho: float = 8.0
    sym_tol: float = 0.0
    sym_stall_window: int = 20
    sym_stall_atol: float = 0.01
    sym_freeze_on_stall: bool = True
    mu_sym_max: Optional[float] = None


class JointIKSolver:
    """
    Joint gradient-descent IK solver that optimises all ligands simultaneously.

    Objective: minimise -score(X) + shape/symmetry constraint penalties,
    where score() is the Scorer's unified statistical score
    (Scorer.score(), scorer.py). The shape and symmetry-tie constraints
    are each enforced via safeguarded Augmented-Lagrangian dual ascent —
    see IKSolverConfig for their hyperparameters and module docstring for
    the overall mechanism.

    Quantity weights (including Q2 enable/disable) are set on the Scorer
    at construction time via its quantity_weights argument — not here.
    """

    def __init__(
        self,
        fks: list,
        scorer: "Scorer",
        ge: "GeometryExtractor",
        config: Optional[IKSolverConfig] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.fks = fks
        self.scorer = scorer
        self.ge = ge
        self.device = device

        cfg = config or IKSolverConfig()
        self.n_steps = cfg.n_steps
        self.lr = cfg.lr
        self.lambda_init = cfg.lambda_init
        self.lse_temp = cfg.lse_temp
        self.adam_beta1, self.adam_beta2 = cfg.adam_betas
        self.adam_eps = 1e-8
        self.plateau_window = cfg.plateau_window
        self.plateau_atol = cfg.plateau_atol
        self.shape_rho = cfg.shape_rho
        self.shape_tol = cfg.shape_tol
        self.sym_weight = cfg.sym_weight
        self.sym_angle_weight = cfg.sym_angle_weight
        self.mu_shape_stall_window = cfg.mu_shape_stall_window
        self.mu_shape_stall_atol = cfg.mu_shape_stall_atol
        self.mu_shape_freeze_on_stall = cfg.mu_shape_freeze_on_stall
        self.mu_shape_max = cfg.mu_shape_max
        self.sym_rho = cfg.sym_rho
        self.sym_tol = cfg.sym_tol
        self.sym_stall_window = cfg.sym_stall_window
        self.sym_stall_atol = cfg.sym_stall_atol
        self.sym_freeze_on_stall = cfg.sym_freeze_on_stall
        self.mu_sym_max = cfg.mu_sym_max

    def solve(
        self,
        theta_inits: list,  # [Tensor(B, 6+Ji), ...]
        X_ref: torch.Tensor,  # (N, 3)
        donor_pos_idxs: list,  # [LongTensor(n_d_i), ...]
        ideal_directions: torch.Tensor,  # (n_donors_total, 3)
        debug: bool = False,
        symmetric_groups: Optional[List[List[int]]] = None,
        q3_symmetric_groups: Optional[List[List[int]]] = None,
        q3_mode_commit: bool = True,
    ):
        """
        Run the joint IK solve for all ligands simultaneously.

        Parameters
        ----------
        theta_inits    : list of (B, 6+Ji) initial DOF tensors, one per ligand.
                         All must have the same batch size B.
        X_ref          : (N, 3) full-complex reference atom positions.
        donor_pos_idxs : list of (n_d_i,) LongTensors of pos-indices into X_ref.
        ideal_directions : (n_donors_total, 3) fixed unit vectors, one per
                         donor in ge.donor_positions()'s global donor order
                         (= ideal[assignment[di]] from engine.py's Hungarian
                         assignment). Donors are pulled toward the best-fit
                         rotation of this ideal polyhedron, recomputed every
                         step (see module docstring).
        debug          : print per-step diagnostics.
        symmetric_groups : list of donor-index groups (each a list of
                         indices into the same global donor order as
                         ideal_directions) that are topologically equivalent
                         (RDKit canonical-rank match, engine.py) and should
                         be pulled toward a common M-D distance.
        q3_symmetric_groups : list of donor-index groups, same index
                         convention as symmetric_groups. Used IN PLACE OF
                         symmetric_groups for the Q3 approach-angle tie and
                         q3_mode_commit only — the M-D bond-length tie
                         (sym_weight) always uses the unrestricted
                         symmetric_groups. None (default) falls back to
                         symmetric_groups. Callers should pass a
                         within-single-ligand-instance-only subset here
                         (see engine.py's symmetric_groups construction) —
                         committing multiple separate ligand instances (e.g.
                         all 6 waters of a hexaaqua complex) to one Q3 mode
                         is not chemically justified.
        q3_mode_commit  : bool, default True. When a symmetric donor
                         group's Q3 target is genuinely multi-modal, commit
                         the whole group to a single GMM component (chosen
                         from the initial pose) for the whole solve, instead
                         of letting the Q3 angle-tie pull members toward a
                         mean that can fall in the low-density valley
                         between two real modes.

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
        donor_pos_idxs = [idx.to(device) for idx in donor_pos_idxs]
        ideal_directions_dev = ideal_directions.to(device)

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
        best_score_ever = torch.full((B,), float("-inf"), device=device)
        best_score_thetas = [ti.detach().clone() for ti in thetas]

        # Score plateau: ring buffer of recent scores (B,).
        # Convergence = range over last plateau_window steps < plateau_atol.
        W = self.plateau_window
        score_ring = torch.full((W, B), float("-inf"), device=device)
        ever_plateaued = torch.zeros(B, dtype=torch.bool, device=device)

        # Shape-violation stall detector: freezes mu_shape's dual ascent once
        # the violation demonstrably stops shrinking (ring-buffer pattern,
        # same as score_ring above).
        MW = self.mu_shape_stall_window
        violation_ring = torch.full((MW, B), float("inf"), device=device)
        mu_shape_frozen = torch.zeros(B, dtype=torch.bool, device=device)

        # Same stall-ring pattern, independently, for the symmetry-tying
        # constraints' own dual variables. No-op when a case has no
        # symmetric_groups (sym_penalty stays 0, so these never trigger).
        SW = self.sym_stall_window
        sym_violation_ring = torch.full((SW, B), float("inf"), device=device)
        sym_angle_violation_ring = torch.full((SW, B), float("inf"), device=device)
        mu_sym_frozen = torch.zeros(B, dtype=torch.bool, device=device)
        mu_sym_angle_frozen = torch.zeros(B, dtype=torch.bool, device=device)

        history_residuals = []
        history_scores = []
        history_anchors = []
        history_grad_norms = []
        history_mu_shape = []
        history_sym_penalty = []
        history_sym_angle_penalty = []
        history_mu_sym = []
        history_mu_sym_angle = []

        # Augmented-Lagrangian dual variables, initialised from lambda_init /
        # sym_weight / sym_angle_weight for continuity with their meaning as
        # starting weights; each then self-adjusts via dual ascent every step.
        mu_shape = torch.full((B,), self.lambda_init, device=device)
        mu_sym = torch.full((B,), self.sym_weight, device=device)
        mu_sym_angle = torch.full((B,), self.sym_angle_weight, device=device)

        # Site-index -> Q3-array-index map: ge.q3_approach_angles() only
        # returns Q3-eligible sites' angles, in site-index order (same
        # convention ik/scorer.py's Q3 construction loop relies on), so this
        # recovers which q3-array slot (if any) each symmetric_groups donor
        # index corresponds to. Static, so built once here rather than every
        # step of the T-loop below. Uses the caller's narrower
        # q3_symmetric_groups for the Q3 tie/mode-commit when given, else
        # falls back to symmetric_groups.
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

        # Mode-committed Q3 initialization: q3_mode_override is (B, n_q3)
        # int64, -1 = no override (scorer.py no-ops on this). Computed ONCE
        # here from the initial pose (before any gradient steps) and held
        # fixed for the whole trajectory — a single up-front commitment per
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
                            f"\n[DEBUG]   Q3 mode-commit: committed "
                            f"{int((override[0] >= 0).sum().item())} of {n_q3_total} "
                            f"Q3 terms for trial 0 -- modes {override[0][override[0] >= 0].tolist()}"
                        )

        if debug:
            print(
                f"\n[DEBUG]   JointIKSolver  L={L}  B={B}  T={T}  "
                f"lr={self.lr}  lambda_init={self.lambda_init}  "
                f"lse_temp={self.lse_temp}  "
                f"plateau_window={W}  plateau_atol={self.plateau_atol}"
            )

        for t in range(T):
            # --- Forward: apply all FKs sequentially to build X_current ---
            X_current = X_ref.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N, 3)
            for li, (fk_li, theta_li) in enumerate(zip(self.fks, thetas)):
                X_current = fk_li.apply(theta_li, X_current)  # (B, N, 3)

            # --- Unified score: one call, one number, honest objective ---
            stat_score, _ = self.scorer.score(X_current, q3_mode_override=q3_mode_override)
            # stat_score : (B,)  — the scalar optimised and reported

            # --- Shape constraint + symmetry ties ---
            # Optimal rotation recomputed every step from the CURRENT donor
            # arrangement, under no_grad and detached: by the envelope
            # theorem d(loss)/dR = 0 at R's own optimum, so detaching it
            # costs nothing to first order while avoiding SVD-backward
            # instability on near-degenerate inputs.
            all_donor_pos_grad = self.ge.donor_positions(X_current)  # (B, n_donors, 3)
            all_donor_pos_det = all_donor_pos_grad.detach()
            sym_penalty = X_ref.new_zeros(B)
            sym_angle_penalty = X_ref.new_zeros(B)

            donor_unit_vecs = F.normalize(all_donor_pos_grad, dim=-1)  # (B, n_donors, 3)

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
            anchor_loss = shape_dev  # kept as solve_trace's diagnostic name

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

                # --- Q3 approach-angle symmetry tie ---
                # Mirrors the distance tie's pattern exactly, on Q3 approach
                # angle instead of M-D distance.
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

            # sym_penalty / sym_angle_penalty each get their own
            # safeguarded-ALM dual variable (mu_sym, mu_sym_angle) that
            # escalates off its own violation, self-calibrating per case
            # with no hand-tuned constant.
            sym_violation = (sym_penalty - self.sym_tol).clamp(min=0.0)
            sym_angle_violation = (sym_angle_penalty - self.sym_tol).clamp(min=0.0)

            constraint_loss = (
                constraint_loss
                + mu_sym.detach() * sym_violation
                + 0.5 * self.sym_rho * sym_violation**2
                + mu_sym_angle.detach() * sym_angle_violation
                + 0.5 * self.sym_rho * sym_angle_violation**2
            )

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
            # Combined loss: -score + constraints, computed in one expression
            # so a single backward() traversal covers both gradients.
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

                # --- mu_shape stall safeguard --------------------------------
                # Classical dual ascent (mu += rho*violation every step,
                # unconditionally) assumes escalating mu will eventually
                # drive violation to zero. When the constraint is
                # effectively infeasible, violation instead plateaus at some
                # floor > 0 and mu would grow without bound -- once
                # violation stops improving over a window, stop escalating.
                if self.mu_shape_freeze_on_stall:
                    violation_ring[t % MW] = violation.detach()
                    if t >= MW - 1 and t % MW == MW - 1:
                        v_range = (
                            violation_ring.max(dim=0).values - violation_ring.min(dim=0).values
                        )
                        mu_shape_frozen = mu_shape_frozen | (v_range < self.mu_shape_stall_atol)

                # Dual ascent: mu_shape self-adjusts to the violation just
                # observed this step, unless frozen by the stall safeguard.
                mu_increment = self.shape_rho * violation.detach()
                mu_increment = torch.where(
                    mu_shape_frozen, torch.zeros_like(mu_increment), mu_increment
                )
                mu_shape = (mu_shape + mu_increment).clamp(min=0.0)
                if self.mu_shape_max is not None:
                    mu_shape = mu_shape.clamp(max=self.mu_shape_max)

                # --- sym ALM stall safeguard + dual ascent -------------------
                # Identical mechanism to mu_shape's above, applied
                # independently to mu_sym and mu_sym_angle.
                if self.sym_freeze_on_stall:
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
                    f"  [t={t:>4}]  score min={stat_score.min().item():.4f}"
                    f"  max={stat_score.max().item():.4f}"
                    f"  nan={torch.isnan(stat_score).any().item()}"
                    f"  |  plateaued={int(ever_plateaued.sum().item())}/{B}"
                    f"  |  total_grad_norm={total_gn:.4f}"
                )

            # --- Adam update on combined gradient ---
            # Single optimiser for the combined (constraint + chemistry)
            # gradient. Adam's per-DOF lr adaptation compensates for
            # gradient dilution: a DOF with small total gradient gets a
            # proportionally larger effective lr, making convergence rate
            # complex-size independent.
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
        best_thetas = best_score_thetas  # list of (B, 6+Ji)

        with torch.no_grad():
            X_final = X_ref.unsqueeze(0).expand(B, -1, -1).clone()
            for fk_li, theta_li in zip(self.fks, best_thetas):
                X_final = fk_li.apply(theta_li, X_final)

            final_score, _ = self.scorer.score(X_final, q3_mode_override=q3_mode_override)  # (B,)
            # Convergence = score plateaued at any point during optimisation.
            # A trajectory that plateaued at a good score is converged;
            # one that never plateaued (still improving at T) is not.
            converged = ever_plateaued

        if debug:
            n_conv = int(converged.sum().item())
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
                "anchor_loss": torch.stack(history_anchors, dim=0).cpu().numpy(),  # (T, B) -- shape_dev
                "grad_norm": torch.stack(history_grad_norms, dim=0).cpu().numpy(),  # (T, B)
                "mu_shape": torch.stack(history_mu_shape, dim=0).cpu().numpy(),  # (T, B)
                "sym_penalty": torch.stack(history_sym_penalty, dim=0).cpu().numpy(),  # (T, B)
                "sym_angle_penalty": torch.stack(history_sym_angle_penalty, dim=0).cpu().numpy(),  # (T, B)
                "mu_sym": torch.stack(history_mu_sym, dim=0).cpu().numpy(),  # (T, B)
                "mu_sym_angle": torch.stack(history_mu_sym_angle, dim=0).cpu().numpy(),  # (T, B)
            }

        thetas_final = [th.detach() for th in best_thetas]
        return thetas_final, converged, final_score, solve_trace
