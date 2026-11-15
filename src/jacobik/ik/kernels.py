"""
kernels.py
==========
Pure differentiable math kernels for the IK engine.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from jacobik.core.env import EPS, LOG_2PI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
_LOG_2PI = LOG_2PI
_EPS = EPS  # safe divisor / small-angle threshold


# ---------------------------------------------------------------------------
# Atom gather (index_select based)
# ---------------------------------------------------------------------------


def gather_atoms(X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to ``X[:, idx, :]`` for a LongTensor ``idx`` of any shape, but
    built on ``index_select``.  Advanced indexing's backward is an
    ``index_put_(accumulate=True)`` which is several times slower on CPU than
    ``index_select``'s ``index_add_`` backward; values and gradients are identical.

    X : (B, N, 3);  idx : (...)  ->  (B, *idx.shape, 3)
    """
    if idx.dim() == 1:
        return X.index_select(1, idx)
    return X.index_select(1, idx.reshape(-1)).reshape(X.shape[0], *idx.shape, X.shape[-1])


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
    axis_normalized: bool = False,
) -> torch.Tensor:
    """
    Rotate a batch of point sets around a pivot/axis by angle theta.

    Parameters
    ----------
    pts   : (B, n, 3) — points to rotate
    pivot : (B, 3)    — pivot point (subtracted before, added after)
    axis  : (B, 3)    — rotation axis; normalised internally unless
                        axis_normalized=True
    theta : (B,)      — rotation angle in radians
    axis_normalized : if True, `axis` is already unit-length and the internal
                      normalisation is skipped

    Returns
    -------
    (B, n, 3) rotated points
    """
    assert pts.shape[0] == pivot.shape[0] == axis.shape[0] == theta.shape[0], (
        f"Batch dimension mismatch: pts={pts.shape[0]}, pivot={pivot.shape[0]}, "
        f"axis={axis.shape[0]}, theta={theta.shape[0]}"
    )

    # Normalise axis — safe for zero vector (will produce a zero rotation)
    k = axis if axis_normalized else _safe_normalize(axis, dim=-1)  # (B, 3)

    # Build the (B, 3, 3) rotation matrix once and apply it with one batched matmul.
    # RT is R^T so that rows-of-points @ RT = (R v)^T.
    cos_t = torch.cos(theta)[:, None, None]  # (B, 1, 1)
    sin_t = torch.sin(theta)[:, None, None]  # (B, 1, 1)
    eye = torch.eye(3, dtype=pts.dtype, device=pts.device)
    k_exp = k[:, None, :].expand(-1, 3, -1)  # (B, 3, 3)
    # cross(k, e_i) is row i of K^T (K = [k]x); R^T = c I + (1-c) k k^T + s K^T
    RT = (
        cos_t * eye
        + (1.0 - cos_t) * (k[:, :, None] * k[:, None, :])
        + sin_t * torch.linalg.cross(k_exp, eye.expand_as(k_exp), dim=-1)
    )
    return torch.baddbmm(pivot[:, None, :], pts - pivot[:, None, :], RT)


def dihedral_axis_from_quad(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Signed dihedral of (P[:,0], P[:,1], P[:,2], P[:,3]) plus the unit
    P1 -> P2 axis.  Same math as ``dihedral_with_axis`` but operates on one
    stacked (B, 4, 3) tensor so the whole thing is ~12 ops instead of ~35.
    """
    b = P[:, 1:] - P[:, :-1]  # (B, 3, 3): b1, b2, b3
    n = torch.linalg.cross(b[:, :2], b[:, 1:], dim=-1)  # (B, 2, 3): n1, n2
    n1 = n[:, 0]
    n2 = n[:, 1]
    b2 = b[:, 1]
    b2_hat = b2 / (b2 * b2).sum(dim=-1, keepdim=True).clamp(min=_EPS * _EPS).sqrt()
    m = torch.linalg.cross(n1, n2, dim=-1)
    sin_phi = (m * b2_hat).sum(dim=-1)
    cos_phi = (n1 * n2).sum(dim=-1)
    return torch.atan2(sin_phi, cos_phi), b2_hat


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
# Signed dihedral angle
# ---------------------------------------------------------------------------


def dihedral_with_axis(
    p_i: torch.Tensor,  # (B, 3)
    p_j: torch.Tensor,  # (B, 3)
    p_k: torch.Tensor,  # (B, 3)
    p_l: torch.Tensor,  # (B, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Signed dihedral of the i-j-k-l quadruplet plus the unit p_j -> p_k axis.

    Parameters
    ----------
    p_i, p_j, p_k, p_l : (B, 3) atom positions

    Returns
    -------
    (phi, b2_hat) : ((B,) signed dihedral in radians ∈ (−π, π], (B, 3) unit vector)
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

    return torch.atan2(sin_phi, cos_phi), b2_hat


def dihedral(
    p_i: torch.Tensor,  # (B, 3)
    p_j: torch.Tensor,  # (B, 3)
    p_k: torch.Tensor,  # (B, 3)
    p_l: torch.Tensor,  # (B, 3)
) -> torch.Tensor:
    """
    Signed dihedral angle of the i-j-k-l quadruplet, IUPAC convention.

    Parameters
    ----------
    p_i, p_j, p_k, p_l : (B, 3) atom positions

    Returns
    -------
    phi : (B,) signed dihedral in radians ∈ (−π, π]
    """
    phi, _ = dihedral_with_axis(p_i, p_j, p_k, p_l)
    return phi


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
# Fused Q6 (non-bonded contact soft-min) with a hand-written backward
# ---------------------------------------------------------------------------


class _Q6SoftMinFn(torch.autograd.Function):
    """
    value_b = -(1/tau) * logsumexp_k( -tau * halfgauss(d_bk) )      (B,)

    where d_bk = |X[b, a_k] - X[b, b_k]| and
    halfgauss(d) = -(d-mu)^2 / (2 sigma^2) for d <= mu else 0.

    Equivalent to the unfused halfgauss_log_prob -> logsumexp chain, but
    (i) runs on a component-major (3, N, B) layout so every gather / scatter
    moves contiguous B-length slabs instead of 12-byte rows, and (ii) uses a
    closed-form backward (softmax weights * lever arms) instead of ~15
    autograd nodes over (B, n_pairs) tensors.  The additive constant
    log(n)/tau is left to the caller.
    """

    @staticmethod
    def forward(ctx, Xc, a_idx, b_idx, mu, var2, tau):
        # Xc: (3, N, B); a_idx/b_idx: (M,) long; mu/var2: (M, 1)
        diff = Xc.index_select(1, a_idx)
        diff.sub_(Xc.index_select(1, b_idx))  # (3, M, B)
        d = (diff * diff).sum(0).sqrt_()  # (M, B)
        # t = min(d - mu, 0): nonzero only inside the wall (d <= mu).
        tneg = (d - mu).clamp_(max=0.0)
        # x = -tau * halfgauss = tau * t^2 / var2 (var2 = 2 sigma^2), >= 0
        x = tneg.square().mul_(tau / var2)
        m = x.max(dim=0).values  # (B,)
        # x - m <= 0; floor at -80 so exp() never produces denormals (which are
        # ~10x slower on CPU); e^-80 ~ 1.8e-35 is far below float32 resolution
        # relative to the >= 1 term in the sum, so the value is unchanged.
        e = (x - m).clamp_(min=-80.0).exp_()
        s = e.sum(dim=0)  # (B,)
        val = -(m + s.log()) / tau
        ctx.save_for_backward(diff, d, tneg, e, s, a_idx, b_idx, var2)
        ctx.n_atoms = Xc.shape[1]
        return val

    @staticmethod
    def backward(ctx, g):
        diff, d, tneg, e, s, a_idx, b_idx, var2 = ctx.saved_tensors
        # dval/dd_k = -(e_k/s) * dx_k/dd_k / tau, dx_k/dd_k = 2 tau t / var2 (inside wall)
        gc = e * (g / s).unsqueeze(0)  # softmax weights * upstream grad
        gc.mul_(tneg).mul_(-2.0 / var2).div_(d.clamp(min=1e-12))  # (M, B)
        ga = diff * gc.unsqueeze(0)  # (3, M, B)
        gX = torch.zeros(3, ctx.n_atoms, ga.shape[-1], dtype=ga.dtype, device=ga.device)
        gX.index_add_(1, a_idx, ga)
        gX.index_add_(1, b_idx, ga.neg_())
        return gX, None, None, None, None, None


def q6_softmin(
    X: torch.Tensor,
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    Fused ``-(1/tau) * logsumexp(-tau * halfgauss(|X[:, a] - X[:, b]|; mu, sigma))``.

    X : (B, N, 3); a_idx, b_idx : (M,) long; mu, sigma : (M,) -> (B,)
    (Add ``log(M)/tau`` for the soft-min form used by Scorer.)
    """
    Xc = X.permute(2, 1, 0).contiguous()  # (3, N, B)
    var2 = (2.0 * sigma.clamp(min=_EPS) ** 2).unsqueeze(-1)  # (M, 1)
    return _Q6SoftMinFn.apply(Xc, a_idx, b_idx, mu.unsqueeze(-1), var2, float(tau))
