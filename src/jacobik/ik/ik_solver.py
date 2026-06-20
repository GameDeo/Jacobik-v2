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
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

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
) -> torch.Tensor:
    """
    Initialise base DOF (translation + axis-angle) for each batch item
    via Kabsch alignment of current donor positions to targets.
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
        self.device = device

    def solve(
        self,
        theta_inits: list,  # [Tensor(B, 6+Ji), ...]
        X_ref: torch.Tensor,  # (N, 3)
        donor_targets: list,  # [Tensor(B, n_d_i, 3), ...]
        donor_pos_idxs: list,  # [LongTensor(n_d_i), ...]
        debug: bool = False,
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
        donor_pos_idxs : list of (n_d_i,) LongTensors of pos-indices into X_ref.
        debug          : print per-step diagnostics.

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

        # Score at step 0 — used to compute score_gain for anchor decay.
        # Initialised to -inf; actual value set on first forward pass.
        score_t0 = torch.full((B,), float("-inf"), device=device)

        history_residuals = []
        history_scores = []
        history_anchors = []
        history_grad_norms = []

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
                lambda_eff = self.lambda_init * frac**2  # (B,)

            # --- Forward: apply all FKs sequentially to build X_current ---
            X_current = X_ref.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N, 3)
            for li, (fk_li, theta_li) in enumerate(zip(self.fks, thetas)):
                X_current = fk_li.apply(theta_li, X_current)  # (B, N, 3)

            # --- Unified score: one call, one number, honest objective ---
            stat_score, _ = self.scorer.score(X_current)
            # stat_score : (B,)  — the scalar optimised and reported

            # Seed score_t0 on first iteration
            if t == 0:
                score_t0 = stat_score.detach().clone()

            # --- Anchor loss with LSE soft-max over per-donor errors ---
            # Focuses gradient on the worst-placed donor (lse_temp=2.0,
            # same temperature as Q6 LSE in scorer.py — one device, one constant).
            anchor_loss = X_ref.new_zeros(B)
            all_donor_pos_grad = self.ge.donor_positions(X_current)  # (B, n_donors, 3)
            all_donor_pos_det = all_donor_pos_grad.detach()

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
            combined_loss = (-stat_score + lambda_eff * anchor_loss).sum()  # scalar
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

            r = -stat_score + lambda_eff * anchor_loss  # (B,) for logging

            # --- Record trace (on-device, detached) ---
            with torch.no_grad():
                grad_norm_b = torch.cat([g.reshape(B, -1) for g in total_grads], dim=-1).norm(
                    dim=-1
                )  # (B,)

                history_residuals.append(r.detach())
                history_scores.append(stat_score.detach())
                history_anchors.append(anchor_loss.detach())
                history_grad_norms.append(grad_norm_b)

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

            final_score, _ = self.scorer.score(X_final)  # (B,)
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
                "anchor_loss": torch.stack(history_anchors, dim=0).cpu().numpy(),  # (T, B)
                "grad_norm": torch.stack(history_grad_norms, dim=0).cpu().numpy(),  # (T, B)
            }

        thetas_final = [th.detach() for th in best_thetas]
        return thetas_final, converged, final_score, solve_trace
