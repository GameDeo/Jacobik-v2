"""Numpy internal coordinates and Kabsch superposition utilities (torch IK kernels live in ``jacobik.ik``)."""
from __future__ import annotations

import numpy as np


def dihedral_deg(p0, p1, p2, p3) -> float:
    """Signed dihedral p0-p1-p2-p3 in degrees, range (-180, 180] (the IUPAC sign)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def kabsch_fit_batch(P: np.ndarray, Q: np.ndarray):
    """Rotations aligning each Q[s] onto P[s] (both (S, n, 3)): returns rmsd (S,), max_dev (S,), R (S, 3, 3).

    Both sets are centred on their own centroid; apply as ``(Q - Q.mean) @ R + P.mean``."""
    Pc, Qc = P - P.mean(1, keepdims=True), Q - Q.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.einsum("sni,snj->sij", Qc, Pc))
    dd = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros((len(dd), 3, 3))
    D[:, 0, 0] = 1
    D[:, 1, 1] = 1
    D[:, 2, 2] = dd
    R = U @ D @ Vt
    sq = ((Pc - Qc @ R) ** 2).sum(-1)
    return np.sqrt(sq.mean(1)), np.sqrt(sq.max(1)), R


def kabsch_fit_np(P: np.ndarray, Q: np.ndarray):
    """Kabsch rotation aligning Q onto P. Returns rmsd, max_dev, R."""
    rmsd, max_dev, R = kabsch_fit_batch(P[None], Q[None])
    return float(rmsd[0]), float(max_dev[0]), R[0]
