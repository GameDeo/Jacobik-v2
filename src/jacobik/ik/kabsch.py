"""Kabsch alignment and rigid-body rotation utilities for inverse kinematics."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from jacobik.core.env import EPS
from jacobik.ik.fk import axisangle_to_matrix

if TYPE_CHECKING:
    from jacobik.core.geometry import GeometryExtractor

_EPS = EPS


def _ref_rows(X_ref: torch.Tensor, B: int) -> torch.Tensor:
    """Fresh (B, N, 3) copy of a shared (N, 3) or per-row (B, N, 3) reference."""
    if X_ref.dim() == 3:
        return X_ref.clone()
    return X_ref.unsqueeze(0).expand(B, -1, -1).clone()


def kabsch_init(
    ge: "GeometryExtractor",
    X_ref: torch.Tensor,  # (N, 3) shared, or (B, N, 3) per-row reference positions
    donor_targets: torch.Tensor,  # (B, n_d_lig, 3) targets for THIS ligand
    donor_pos_idx: Optional[torch.Tensor] = None,  # (n_d_lig,) group indices into ge donors
    diversify_orientation: bool = False,
    diversity_scale: float = 1.0,
) -> torch.Tensor:
    """
    Initialise base DOF (translation + axis-angle) per batch item by Kabsch
    alignment of the current donor positions to the targets.

    Fully batched (one SVD over all rows, all random draws in one call, no
    per-row Python or host synchronisation).  For one or collinear donors the
    fit leaves rotational DOFs free; those are filled with a random rotation
    so trials explore different orientations.  With ``diversify_orientation``
    the well-posed (3+ donor) fit is also perturbed by a random rotation of at
    most ``diversity_scale`` * pi.  Returns (B, 6): translation, axis-angle.
    """
    device = donor_targets.device
    Q = donor_targets
    B, n = Q.shape[0], Q.shape[1]
    with torch.no_grad():
        Xr = X_ref.to(device)
        Xb = Xr if Xr.dim() == 3 else Xr.unsqueeze(0).expand(B, -1, -1)
        metal_pos = Xb[:, ge.metal_pos_idx]  # (B, 3)
        dpos = ge.donor_positions(Xb)
        P = dpos[:, donor_pos_idx.to(device), :] if donor_pos_idx is not None else dpos
        eye3 = torch.eye(3, dtype=P.dtype, device=device)

        Pm, Qm = P.mean(dim=1), Q.mean(dim=1)  # (B, 3)
        Pc, Qc = P - Pm[:, None, :], Q - Qm[:, None, :]
        if n == 1:
            R = eye3.expand(B, 3, 3).clone()
        else:
            H = torch.einsum("bni,bnj->bij", Pc, Qc)  # (B, 3, 3)
            U, _, Vt = torch.linalg.svd(H)
            V = Vt.transpose(-1, -2)
            d = torch.linalg.det(V @ U.transpose(-1, -2))
            sign_fix = torch.ones(B, 3, dtype=P.dtype, device=device)
            sign_fix[:, 2] = d.sign()
            R = (V * sign_fix[:, None, :]) @ U.transpose(-1, -2)

        # Rank of each row's donor cloud: 0 = one donor, 1 = collinear.
        rank = (torch.linalg.eigvalsh(Pc.transpose(-1, -2) @ Pc) > 1e-5).sum(dim=-1)  # (B,)

        # Extra random rotation axis/angle, chosen per row by rank (all masks, no branches).
        _, Qe = torch.linalg.eigh(Qc.transpose(-1, -2) @ Qc)
        ax0 = Qm - metal_pos  # rank 0: about the metal-donor axis
        ax1 = Qe[:, :, -1]  # rank 1: principal axis of the targets
        # random draws are made on the CPU generator and moved, so a seed gives the same starts on every device
        ax2 = torch.randn(B, 3, dtype=P.dtype).to(device)  # well-posed: random axis
        axis = torch.where((rank == 0)[:, None], ax0, torch.where((rank == 1)[:, None], ax1, ax2))
        axis = F.normalize(axis, dim=-1)
        u = torch.rand(B, dtype=P.dtype).to(device)
        angle = torch.where(rank <= 1, u * 2.0 * math.pi, u * diversity_scale * math.pi)
        if not diversify_orientation:
            angle = torch.where(rank <= 1, angle, torch.zeros_like(angle))
        degenerate = (rank == 0) & (ax0.norm(dim=-1) <= 1e-5)  # no usable metal-donor axis
        angle = torch.where(degenerate, torch.zeros_like(angle), angle)
        R = axisangle_to_matrix(axis * angle[:, None]) @ R
        # FK rotates about the donor centroid (pivot = Pm), so the centroid lands at Pm + t.
        t = Qm - Pm

        base_theta = torch.zeros(B, 6, dtype=torch.float32, device=device)
        base_theta[:, 0:3] = t
        base_theta[:, 3:6] = _matrix_to_axisangle_batched(R)
    return base_theta  # (B, 6)


def compensate_fk_pivot(base, fk, original, seeded, ge, donor_indices):
    """Transport a seeded donor fit to FK's original donor-atom rotation pivot.

    Internal torsions commute with a rigid transform, but their donor centroid
    can move. Kabsch uses seeded group centroid p; FK uses original atom
    centroid c. Its final points are R(x-c)+c+t, so add (I-R)(p-c) to t.
    Haptic group and donor-atom centroids are intentionally distinguished.
    """
    with torch.no_grad():
        B = len(base)
        X = original if original.dim() == 3 else original.unsqueeze(0).expand(B, -1, -1)
        c = X[:, fk._donor_pos_t].mean(1)
        p = ge.donor_positions(seeded)[:, donor_indices].mean(1)
        delta = p - c
        rotation = axisangle_to_matrix(base[:, 3:6])
        result = base.clone()
        result[:, :3] += delta - torch.bmm(rotation, delta.unsqueeze(-1)).squeeze(-1)
    return result


def _matrix_to_axisangle_batched(R: torch.Tensor) -> torch.Tensor:
    """Batched ``_matrix_to_axisangle``: (B, 3, 3) -> (B, 3), same branch logic."""
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    theta = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))  # (B,)
    # General case: axis from the skew-symmetric part
    axis_g = torch.stack(
        [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], dim=-1
    ) / (2.0 * torch.sin(theta).clamp(min=_EPS)).unsqueeze(-1)
    axis_g = F.normalize(axis_g, dim=-1)
    # theta ~ pi: axis from the symmetric part
    diag = ((R + torch.eye(3, dtype=R.dtype, device=R.device)) / 2.0).diagonal(dim1=-2, dim2=-1)
    axis_pi = diag.clamp(min=0.0).sqrt()
    sgn = torch.ones_like(axis_pi)
    sgn[:, 1] = torch.where(R[:, 0, 1] < 0, -1.0, 1.0)
    sgn[:, 2] = torch.where(R[:, 0, 2] < 0, -1.0, 1.0)
    axis_pi = F.normalize(axis_pi * sgn, dim=-1)
    near_pi = (math.pi - theta) < 1e-4
    axis = torch.where(near_pi.unsqueeze(-1), axis_pi, axis_g)
    omega = axis * theta.unsqueeze(-1)
    return torch.where((theta.abs() < _EPS).unsqueeze(-1), torch.zeros_like(omega), omega)


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


def _optimal_rotation_nocentroid(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched rotation-only Procrustes alignment: for each b, the rotation R[b]
    minimising sum_i || R[b] @ A[b, i] - B[b, i] ||^2, with no translation or
    centroid subtraction (both sets are metal-centred unit vectors).

    Uses Horn's closed-form quaternion method (Horn 1987, J. Opt. Soc. Am. A
    4:629): the optimal quaternion is the top eigenvector of a 4x4 symmetric
    matrix built from H = A^T B. Intended for use under no_grad.

    A : (Batch, n, 3) reference unit vectors (ideal polyhedron directions).
    B : (Batch, n, 3) current unit vectors.
    Returns R : (Batch, 3, 3), det(R) = +1.
    """
    Bsz = B.shape[0]
    H = torch.einsum("bni,bnj->bij", A, B)  # (Batch, 3, 3) = A[b]^T @ B[b]
    Sxx, Sxy, Sxz = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    Syx, Syy, Syz = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    Szx, Szy, Szz = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
    N = H.new_zeros(Bsz, 4, 4)
    N[:, 0, 0] = Sxx + Syy + Szz
    N[:, 0, 1] = Syz - Szy
    N[:, 0, 2] = Szx - Sxz
    N[:, 0, 3] = Sxy - Syx
    N[:, 1, 0] = N[:, 0, 1]
    N[:, 1, 1] = Sxx - Syy - Szz
    N[:, 1, 2] = Sxy + Syx
    N[:, 1, 3] = Szx + Sxz
    N[:, 2, 0] = N[:, 0, 2]
    N[:, 2, 1] = N[:, 1, 2]
    N[:, 2, 2] = -Sxx + Syy - Szz
    N[:, 2, 3] = Syz + Szy
    N[:, 3, 0] = N[:, 0, 3]
    N[:, 3, 1] = N[:, 1, 3]
    N[:, 3, 2] = N[:, 2, 3]
    N[:, 3, 3] = -Sxx - Syy + Szz
    # eigh eigenvalues are ascending: the optimal quaternion is the last eigenvector.
    _, evecs = torch.linalg.eigh(N)
    q = evecs[:, :, -1]  # (Batch, 4): (w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack(
        [
            torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1
            ),
            torch.stack(
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1
            ),
            torch.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1
            ),
        ],
        dim=-2,
    )  # (Batch, 3, 3): R @ A[i] ~= B[b, i]
    return R
