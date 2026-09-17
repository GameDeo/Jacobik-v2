"""
kernels.py
==========
Pure differentiable math kernels for the IK engine.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
_LOG_2PI = math.log(_TWO_PI)
_EPS = 1e-7  # safe divisor / small-angle threshold


# ---------------------------------------------------------------------------
# Safe normalisation (finite gradient at zero vector)
# ---------------------------------------------------------------------------


def _safe_normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Normalize v along dim with a finite gradient at zero vectors.
    """
    norm = (v * v).sum(dim=dim, keepdim=True).clamp(min=_EPS * _EPS).sqrt()
    return v / norm


# ---------------------------------------------------------------------------
# Rodrigues rotation
# ---------------------------------------------------------------------------


def rodrigues(
    pts: torch.Tensor,  # (B, n, 3) — points to rotate
    pivot: torch.Tensor,  # (B, 3)    — centre of rotation
    axis: torch.Tensor,  # (B, 3)    — rotation axis (need not be unit)
    theta: torch.Tensor,  # (B,)      — rotation angle in radians
) -> torch.Tensor:
    """
    Rotate a batch of point sets around a pivot/axis by angle theta.


    Parameters
    ----------
    pts   : (B, n, 3) — points to rotate
    pivot : (B, 3)    — pivot point (subtracted before, added after)
    axis  : (B, 3)    — rotation axis; normalised internally
    theta : (B,)      — rotation angle in radians

    Returns
    -------
    (B, n, 3) rotated points

    """
    assert pts.shape[0] == pivot.shape[0] == axis.shape[0] == theta.shape[0], (
        f"Batch dimension mismatch: pts={pts.shape[0]}, pivot={pivot.shape[0]}, "
        f"axis={axis.shape[0]}, theta={theta.shape[0]}"
    )

    # Normalise axis — safe for zero vector (will produce a zero rotation)
    k = _safe_normalize(axis, dim=-1)  # (B, 3)

    # Translate to pivot frame
    v = pts - pivot[:, None, :]  # (B, n, 3)

    cos_t = torch.cos(theta)[:, None, None]  # (B, 1, 1)
    sin_t = torch.sin(theta)[:, None, None]  # (B, 1, 1)
    k_b = k[:, None, :]  # (B, 1, 3) for broadcast

    # k × v :  cross product over last dim, broadcast over n
    # torch.linalg.cross requires matching shapes; expand k to (B, n, 3)
    k_exp = k_b.expand_as(v)  # (B, n, 3)
    k_cross = torch.linalg.cross(k_exp, v, dim=-1)  # (B, n, 3)

    # k · v : dot product, keep dim for broadcast
    k_dot_v = (k_b * v).sum(dim=-1, keepdim=True)  # (B, n, 1)

    rotated = v * cos_t + k_cross * sin_t + k_b * k_dot_v * (1.0 - cos_t)

    return rotated + pivot[:, None, :]


# ---------------------------------------------------------------------------
# Axis-angle → rotation matrix
# ---------------------------------------------------------------------------


def axisangle_to_matrix(omega: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of axis-angle vectors to rotation matrices via Rodrigues.


    Parameters
    ----------
    omega : (B, 3) — axis-angle vector; direction encodes axis, magnitude
                     encodes angle in radians

    Returns
    -------
    R : (B, 3, 3) rotation matrices

    """
    # Safe norm: clamp the squared magnitude before sqrt so grad is finite at 0
    angle_sq = (omega * omega).sum(dim=-1, keepdim=True).clamp(min=_EPS * _EPS)
    angle = angle_sq.sqrt()  # (B, 1)
    safe = angle.squeeze(-1) >= _EPS  # (B,) bool

    # Unit axis — use omega directly for small angles (angle ≈ 0)
    k = torch.where(
        safe[:, None],
        omega / angle.clamp(min=_EPS),
        omega,
    )  # (B, 3)

    # Skew-symmetric [k]×
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [
            zero,
            -kz,
            ky,
            kz,
            zero,
            -kx,
            -ky,
            kx,
            zero,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)  # (B, 3, 3)

    I = torch.eye(3, dtype=omega.dtype, device=omega.device).unsqueeze(0)

    cos_a = torch.cos(angle).unsqueeze(-1)  # (B, 1, 1)
    sin_a = torch.sin(angle).unsqueeze(-1)  # (B, 1, 1)

    # Full Rodrigues: R = I·cos + [k]×·sin + k⊗k·(1−cos)
    kk = k.unsqueeze(-1) * k.unsqueeze(-2)  # (B, 3, 3)  outer product
    R_full = cos_a * I + sin_a * K + (1.0 - cos_a) * kk

    # Taylor fallback: R ≈ I + [ω]×  (first order, ω already small)
    K_omega = torch.stack(
        [
            zero,
            -omega[:, 2],
            omega[:, 1],
            omega[:, 2],
            zero,
            -omega[:, 0],
            -omega[:, 1],
            omega[:, 0],
            zero,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    R_taylor = I + K_omega

    mask = safe[:, None, None]  # (B, 1, 1) bool
    return torch.where(mask, R_full, R_taylor)


# ---------------------------------------------------------------------------
# Skew-symmetric (cross-product) matrix and null-space projector
#
# Added to support per-ligand null-space-aware anchor relaxation in
# ik_solver.py (see null_space_anchor_relaxation_plan.md):
# for a multi-donor ligand, the rotation about the axis joining its donor
# anchors moves none of them, so neither the anchor term nor Q1/Q2 (for
# this ligand's own donors) have any gradient signal along that axis.
# null_space_keep_projector() identifies that axis per ligand per step so
# the anchor's rotational gradient can be projected onto the complementary
# (anchor-relevant) subspace only, leaving Q3/Q4/Q6 free to act on it.
# ---------------------------------------------------------------------------


def skew(v: torch.Tensor) -> torch.Tensor:
    """
    Batched skew-symmetric (cross-product) matrix.

    Parameters
    ----------
    v : (..., 3)

    Returns
    -------
    (..., 3, 3) such that skew(v) @ omega == cross(v, omega) for any omega.
    Sign convention is internal-use only: callers that need only the null
    space of skew(v) (e.g. null_space_keep_projector below) are invariant
    to the sign, since negating a matrix does not change its null space.
    """
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(vx)
    row0 = torch.stack([zero, -vz, vy], dim=-1)
    row1 = torch.stack([vz, zero, -vx], dim=-1)
    row2 = torch.stack([-vy, vx, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def null_space_keep_projector(
    donor_pos: torch.Tensor,
    rel_tol: float = 1e-4,
    abs_floor: float = 1e-8,
) -> torch.Tensor:
    """
    Projector onto rotations that move at least one of donor_pos's points.

    For a single donor this is exactly the zero matrix (any rotation about
    the lone donor leaves it fixed — the whole rotational gradient from an
    anchor-style loss is redundant there). For two donors it is the
    projector that kills exactly the axis joining them. For three or more
    non-collinear donors it is (numerically) the identity — no redundant
    axis exists. Degrades gracefully for collinear configurations via the
    same SVD-threshold logic, with no special-casing by donor count.

    Parameters
    ----------
    donor_pos : (B, n_d, 3) — this ligand's donor anchor positions (the
                same site-level positions used for the anchor loss).
    rel_tol   : singular values below rel_tol * largest singular value are
                treated as redundant (null) directions.
    abs_floor : absolute floor on the threshold, for the n_d == 1 case
                where all singular values are exactly 0.

    Returns
    -------
    (B, 3, 3) projector P_keep. Apply to a raw rotation gradient via
    torch.bmm(P_keep, grad.unsqueeze(-1)).squeeze(-1) to zero its component
    along directions that move none of this ligand's donors.
    """
    B, n_d, _ = donor_pos.shape
    pivot = donor_pos.mean(dim=1, keepdim=True)
    v = donor_pos - pivot
    J = skew(v).reshape(B, n_d * 3, 3)
    _, S, Vh = torch.linalg.svd(J, full_matrices=False)
    V = Vh.transpose(-1, -2)
    s_max = S.max(dim=-1, keepdim=True).values
    tol = torch.clamp(s_max * rel_tol, min=abs_floor)
    mask = (S >= tol).float()
    return torch.einsum("bik,bk,bjk->bij", V, mask, V)


# ---------------------------------------------------------------------------
# Signed dihedral angle
# ---------------------------------------------------------------------------


def dihedral(
    p_i: torch.Tensor,  # (B, 3)
    p_j: torch.Tensor,  # (B, 3)
    p_k: torch.Tensor,  # (B, 3)
    p_l: torch.Tensor,  # (B, 3)
) -> torch.Tensor:
    """
    Signed dihedral angle of the i–j–k–l quadruplet, IUPAC convention.


    Parameters
    ----------
    p_i, p_j, p_k, p_l : (B, 3) atom positions

    Returns
    -------
    phi : (B,) signed dihedral in radians ∈ (−π, π]

    """
    b1 = p_j - p_i  # (B, 3)
    b2 = p_k - p_j  # (B, 3)
    b3 = p_l - p_k  # (B, 3)

    n1 = torch.linalg.cross(b1, b2, dim=-1)  # (B, 3)
    n2 = torch.linalg.cross(b2, b3, dim=-1)  # (B, 3)

    b2_hat = _safe_normalize(b2, dim=-1)  # (B, 3)

    # m = n1 × n2, projected onto b2_hat gives the sin component
    m = torch.linalg.cross(n1, n2, dim=-1)  # (B, 3)

    sin_phi = (m * b2_hat).sum(dim=-1)  # (B,)
    cos_phi = (n1 * n2).sum(dim=-1)  # (B,)

    return torch.atan2(sin_phi, cos_phi)


# ---------------------------------------------------------------------------
# Angle wrapping
# ---------------------------------------------------------------------------


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles to [−π, π] in-place-safe (returns new tensor).

    Parameters
    ----------
    theta : any shape — angles in radians

    Returns
    -------
    Same shape as theta, values in (−π, π].
    """
    return (theta + math.pi) % _TWO_PI - math.pi


# ---------------------------------------------------------------------------
# GMM log-probability
# ---------------------------------------------------------------------------


def gmm_log_prob(
    x: torch.Tensor,  # (B,)     — observed values
    means: torch.Tensor,  # (K,)     — component means
    vars_: torch.Tensor,  # (K,)     — component variances (σ²)
    weights: torch.Tensor,  # (K,)     — mixture weights, must sum to 1
    var_min: float = 0.0,  # minimum variance floor (set >0 during IK to bound grads)
) -> torch.Tensor:
    """
    Log-probability of x under a Gaussian Mixture Model.

    Uses log-sum-exp for numerical stability.

    Parameters
    ----------
    x       : (B,) observed values
    means   : (K,) component means
    vars_   : (K,) component variances (not std deviations)
    weights : (K,) mixture weights
    var_min : float — minimum variance floor applied before computing log_prob.
              Defaults to 0 (disabled).  During IK gradient descent, set this
              to a positive value (e.g. 0.04 Å² ↔ σ≥0.2 Å) to prevent the
              gradient -(x−μ)/σ² from exceeding ~|x−μ|/var_min.

              Without this floor, very tight fitted distributions (σ≈0.02 Å,
              σ²≈4e-4) produce gradients of ~1000 per Å when the atom is
              even 0.4 Å from the mean.  With lr=0.01 that is a 10 Å step
              per IK iteration, causing float32 overflow within ~20 steps:
              diff²/σ² overflows → log_p = −∞ for all components →
              logsumexp(−∞, …) → gradient = NaN → permanently broken solve.

    Returns
    -------
    log_p : (B,) log-probabilities
    """
    # Apply variance floor.  Max safe gradient ≈ |x−μ| / var_min.
    # Separate clamp from the _EPS numerical-stability clamp below.
    effective_vars = vars_.clamp(min=max(var_min, _EPS))

    # Expand for broadcasting: (B, K)
    x_exp = x[:, None]  # (B, 1)
    diff = x_exp - means[None, :]  # (B, K)

    # log N(x | mu, sigma^2) = -0.5*log(2pi*var) - 0.5*(x-mu)^2/var
    log_comp = (
        -0.5 * _LOG_2PI - 0.5 * torch.log(effective_vars) - 0.5 * diff**2 / effective_vars
    )  # (B, K)

    log_w = torch.log(weights.clamp(min=_EPS))  # (K,)
    log_p = log_comp + log_w[None, :]  # (B, K)

    return torch.logsumexp(log_p, dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# VMM log-probability
# ---------------------------------------------------------------------------


def vmm_log_prob(
    phi: torch.Tensor,  # (B,)  — observed angles in radians
    means: torch.Tensor,  # (K,)  — component means in radians
    kappas: torch.Tensor,  # (K,)  — concentration parameters (κ ≥ 0)
    weights: torch.Tensor,  # (K,)  — mixture weights, must sum to 1
) -> torch.Tensor:
    """
    Log-probability of phi under a Von Mises Mixture Model.


    Parameters
    ----------
    phi     : (B,) observed angles in radians
    means   : (K,) component means in radians
    kappas  : (K,) concentration parameters
    weights : (K,) mixture weights

    Returns
    -------
    log_p : (B,) log-probabilities

    """
    # (B, K) broadcasting
    cos_diff = torch.cos(phi[:, None] - means[None, :])  # (B, K)

    # Stable log-normaliser: log(2π·I₀(κ)) = log(2π) + log(i0e(κ)) + κ
    log_norm = _LOG_2PI + torch.log(torch.special.i0e(kappas).clamp(min=_EPS)) + kappas
    # (K,)

    log_comp = kappas[None, :] * cos_diff - log_norm[None, :]  # (B, K)
    log_w = torch.log(weights.clamp(min=_EPS))  # (K,)
    log_p = log_comp + log_w[None, :]  # (B, K)

    return torch.logsumexp(log_p, dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# Half-Gaussian soft-wall (Q6 repulsive penalty)
# ---------------------------------------------------------------------------


def halfgauss_log_prob(
    d: torch.Tensor,  # (B,) — observed distances
    mu: torch.Tensor,  # ()   — wall position (scalar or broadcastable)
    sigma: torch.Tensor,  # ()   — wall width    (scalar or broadcastable)
) -> torch.Tensor:
    """
    One-sided soft-wall penalty for Q6 steric contacts.


    Parameters
    ----------
    d     : (B,) pairwise distances in Å
    mu    : wall position in Å (van-der-Waals sum for this atom pair)
    sigma : wall width in Å

    Returns
    -------
    penalty : (B,) values ≤ 0; zero when d > mu

    """
    penalty = -((d - mu) ** 2) / (2.0 * sigma.clamp(min=_EPS) ** 2)
    return torch.where(d <= mu, penalty, torch.zeros_like(d))


# ---------------------------------------------------------------------------
# VMM sampler
# ---------------------------------------------------------------------------


def vmm_sample(
    means: torch.Tensor,  # (K,) component means in radians
    kappas: torch.Tensor,  # (K,) concentration parameters
    weights: torch.Tensor,  # (K,) mixture weights
    n: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample n angles from a Von Mises Mixture Model.


    Parameters
    ----------
    means     : (K,) component means in radians
    kappas    : (K,) concentration parameters (κ > 0)
    weights   : (K,) mixture weights (must sum to 1)
    n         : number of samples
    generator : optional torch.Generator for reproducibility

    Returns
    -------
    samples : (n,) CPU tensor in radians ∈ (−π, π]

    """
    # Sample component assignments
    cat = torch.distributions.Categorical(probs=weights.cpu())
    idx = cat.sample(
        (n,),
    )  # (n,)
    # Note: Categorical.sample does not accept a generator directly in all
    # torch versions; use manual_seed pattern via the generator's seed.

    # Draw from VonMises for each assigned component
    sel_means = means.cpu()[idx]  # (n,)
    sel_kappas = kappas.cpu()[idx]  # (n,)

    vm = torch.distributions.VonMises(sel_means, sel_kappas)
    samples = vm.sample()  # (n,)

    return wrap_angle(samples)


# ---------------------------------------------------------------------------
# GMM sampler
# ---------------------------------------------------------------------------


def gmm_sample(
    means: torch.Tensor,  # (K,) component means
    variances: torch.Tensor,  # (K,) component variances
    weights: torch.Tensor,  # (K,) mixture weights
    n: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample n values from a Gaussian Mixture Model.


    Parameters
    ----------
    means     : (K,) component means
    variances : (K,) component variances (σ², not σ)
    weights   : (K,) mixture weights (must sum to 1)
    n         : number of samples
    generator : optional torch.Generator for reproducibility

    Returns
    -------
    samples : (n,) CPU tensor

    """
    cat = torch.distributions.Categorical(probs=weights.cpu())
    idx = cat.sample((n,))  # (n,)

    sel_means = means.cpu()[idx]  # (n,)
    sel_stds = variances.cpu()[idx].clamp(min=_EPS).sqrt()  # (n,)

    normal = torch.distributions.Normal(sel_means, sel_stds)
    samples = normal.sample()  # (n,)

    return samples
