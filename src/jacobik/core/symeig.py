"""Batched eigen-decomposition of small symmetric matrices (n <= 4) with plain tensor arithmetic.

torch.linalg.eigh on CUDA dispatches batched small matrices to cuSOLVER with a workspace of about 275 kB per
4x4 matrix: 20k rows need 5.5 GB, so a macrocycle batch runs out of memory on an 8 GB card, and the call checks its
status on the host, which splits every replayed solver step into two CUDA graphs. A cyclic Jacobi sweep is a few
dozen element-wise kernels over the batch: no workspace, no host synchronisation (capturable in a CUDA graph), the
same result on every device. Cyclic Jacobi converges quadratically; six sweeps reach float32 round-off for n <= 4.
"""
from __future__ import annotations

from typing import Tuple

import torch

#: cyclic sweeps; 4x4 symmetric matrices converge to float32 round-off in 4-5 (quadratic convergence)
SWEEPS = 6


def sym_eigh(A: torch.Tensor, sweeps: int = SWEEPS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues (ascending) and eigenvectors (columns) of symmetric (..., n, n) matrices, n <= 4.

    Same contract as torch.linalg.eigh: A = V diag(w) V^T, w sorted ascending, V[..., :, k] the k-th eigenvector.
    Degenerate eigenvalues return some orthonormal basis of their space (as eigh does). Not differentiable through
    the sweeps' angle choice; use under no_grad (every caller fits a rotation or a rank).
    """
    n = A.shape[-1]
    shape = A.shape[:-2]
    a = A.reshape(-1, n, n).clone()
    a = 0.5 * (a + a.transpose(-1, -2))
    v = torch.eye(n, dtype=a.dtype, device=a.device).expand(a.shape[0], n, n).clone()
    pairs = [(p, q) for p in range(n - 1) for q in range(p + 1, n)]
    # [p, q] index tensors made on the device by a factory (arange with step q - p): no host copy, so the sweep can
    # be recorded in a CUDA graph
    idx = {pq: torch.arange(pq[0], pq[1] + 1, pq[1] - pq[0], device=a.device) for pq in pairs}
    for _ in range(sweeps):
        for p, q in pairs:
            ix = idx[(p, q)]
            app, aqq, apq = a[:, p, p], a[:, q, q], a[:, p, q]
            # J^T A J zeroes A[p, q] for tan(2 phi) = 2 a_pq / (a_qq - a_pp); atan2 is defined at a_pq = 0
            phi = 0.5 * torch.atan2(2.0 * apq, aqq - app)
            c, s = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
            # rows p, q of A (J^T A), then columns p, q (.. J), then columns p, q of V (V J)
            ap, aq = a[:, p, :], a[:, q, :]
            a = a.index_copy(1, ix, torch.stack([c * ap - s * aq, s * ap + c * aq], 1))
            ap, aq = a[:, :, p], a[:, :, q]
            a = a.index_copy(2, ix, torch.stack([c * ap - s * aq, s * ap + c * aq], 2))
            vp, vq = v[:, :, p], v[:, :, q]
            v = v.index_copy(2, ix, torch.stack([c * vp - s * vq, s * vp + c * vq], 2))
    w = torch.diagonal(a, dim1=-2, dim2=-1)
    w, order = torch.sort(w, dim=-1)
    v = torch.gather(v, -1, order[:, None, :].expand(-1, n, -1))
    return w.reshape(*shape, n), v.reshape(*shape, n, n)


def sym_eigvalsh(A: torch.Tensor, sweeps: int = SWEEPS) -> torch.Tensor:
    """Eigenvalues (ascending) of symmetric (..., n, n) matrices, n <= 4."""
    return sym_eigh(A, sweeps)[0]


def eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh on the CPU (LAPACK: exact, no workspace issue, production CPU results unchanged), the
    capturable Jacobi sweep on CUDA (cuSOLVER's batched syevj workspace ran an 8 GB card out of memory on
    macrocycle batches, and it cannot sit inside one recorded CUDA graph)."""
    if A.device.type == "cuda":
        return sym_eigh(A)
    return torch.linalg.eigh(A)


def eigvalsh(A: torch.Tensor) -> torch.Tensor:
    """As eigh: torch.linalg.eigvalsh on the CPU, the Jacobi sweep on CUDA."""
    if A.device.type == "cuda":
        return sym_eigh(A)[0]
    return torch.linalg.eigvalsh(A)
