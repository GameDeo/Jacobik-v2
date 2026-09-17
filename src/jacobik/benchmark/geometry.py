"""Numerically stable, symmetry-aware structural feasibility metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray]:
    """Align *mobile* onto *reference* using a proper (non-reflecting) rotation."""
    p = np.asarray(mobile, dtype=np.float64)
    q = np.asarray(reference, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError(f"coordinates must have the same non-empty (N, 3) shape: {p.shape}, {q.shape}")
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ValueError("coordinates contain NaN or infinity")
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    rotation = vt.T @ correction @ u.T
    aligned = pc @ rotation.T + q.mean(0)
    return float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=1)))), aligned


def symmetry_corrected_kabsch(
    mobile: np.ndarray,
    reference: np.ndarray,
    permutations: Iterable[Sequence[int]] | None = None,
    *,
    max_permutations: int = 100_000,
) -> tuple[float, np.ndarray, tuple[int, ...]]:
    """Minimum proper-Kabsch RMSD over explicit graph-automorphism permutations.

    Permutations map reference rows to symmetry-equivalent mobile rows.  They
    should be graph automorphisms, not element-only assignments: the latter can
    silently equate different constitutional isomers.
    """
    p = np.asarray(mobile, dtype=np.float64)
    identity = tuple(range(len(p)))
    candidates = permutations if permutations is not None else (identity,)
    best = (float("inf"), p.copy(), identity)
    seen: set[tuple[int, ...]] = set()
    for count, candidate in enumerate(candidates, 1):
        if count > max_permutations:
            raise RuntimeError(
                f"symmetry group exceeds max_permutations={max_permutations}; "
                "increase the explicit safety limit"
            )
        perm = tuple(int(i) for i in candidate)
        if perm in seen:
            continue
        seen.add(perm)
        if sorted(perm) != list(range(len(p))):
            raise ValueError("each symmetry operation must be a permutation of all coordinate rows")
        rmsd, aligned = kabsch(p[list(perm)], reference)
        if rmsd < best[0]:
            best = rmsd, aligned, perm
    return best


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))


@dataclass(frozen=True)
class GeometryMetrics:
    symmetry_rmsd_angstrom: float
    coordination_sphere_rmsd_angstrom: float
    metal_donor_mae_angstrom: float
    metal_donor_max_error_angstrom: float
    donor_metal_donor_angle_mae_deg: float
    donor_metal_donor_angle_max_error_deg: float
    ligand_internal_rmsd_angstrom: float
    min_nonbonded_distance_angstrom: float
    severe_clash_count: int
    coordination_retained_fraction: float
    symmetry_permutation: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_geometry(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    donor_indices: Sequence[int],
    metal_index: int,
    bonds: Iterable[tuple[int, int]],
    atomic_numbers: Sequence[int],
    permutations: Iterable[Sequence[int]] | None = None,
    sphere_indices: Sequence[int] | None = None,
    ligand_groups: Sequence[Sequence[int]] | None = None,
    coordination_cutoff_scale: float = 1.35,
    clash_scale: float = 0.65,
) -> GeometryMetrics:
    """Evaluate global, first-sphere, internal-ligand, and clash geometry.

    Indices are zero based. Bond/reference-distance ratios make retention and
    clash tests chemistry-adaptive without imposing one cutoff on all metals.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    rmsd, aligned, perm = symmetry_corrected_kabsch(pred, ref, permutations)
    donors = np.asarray(donor_indices, dtype=int)
    md_ref = np.linalg.norm(ref[donors] - ref[metal_index], axis=1)
    md_pred = np.linalg.norm(aligned[donors] - aligned[metal_index], axis=1)
    md_err = np.abs(md_pred - md_ref)

    angle_err: list[float] = []
    for i, di in enumerate(donors):
        for dj in donors[i + 1 :]:
            ar = _angle(ref[di] - ref[metal_index], ref[dj] - ref[metal_index])
            ap = _angle(aligned[di] - aligned[metal_index], aligned[dj] - aligned[metal_index])
            if np.isfinite(ar) and np.isfinite(ap):
                angle_err.append(abs(ap - ar))

    sphere = np.asarray(sphere_indices if sphere_indices is not None else [metal_index, *donors], dtype=int)
    sphere_rmsd, _ = kabsch(pred[list(perm)][sphere], ref[sphere])
    lig_rmsds = []
    for group in ligand_groups or ():
        idx = np.asarray(group, dtype=int)
        if len(idx):
            lig_rmsds.append(kabsch(pred[list(perm)][idx], ref[idx])[0])

    bonded = {tuple(sorted((int(i), int(j)))) for i, j in bonds}
    # ASE's covalent radii are optional; a compact fallback covers tmQM Z<=83.
    try:
        from ase.data import covalent_radii
        radii = np.asarray([covalent_radii[z] for z in atomic_numbers])
    except ImportError:
        radii = np.asarray([0.31 if z == 1 else 0.76 for z in atomic_numbers])
    min_nb, clashes = float("inf"), 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if (i, j) in bonded:
                continue
            distance = float(np.linalg.norm(pred[i] - pred[j]))
            min_nb = min(min_nb, distance)
            if distance < clash_scale * (radii[i] + radii[j]):
                clashes += 1

    retained = float(np.mean(md_pred <= coordination_cutoff_scale * md_ref)) if len(donors) else float("nan")
    return GeometryMetrics(
        symmetry_rmsd_angstrom=rmsd,
        coordination_sphere_rmsd_angstrom=sphere_rmsd,
        metal_donor_mae_angstrom=float(np.mean(md_err)) if len(md_err) else float("nan"),
        metal_donor_max_error_angstrom=float(np.max(md_err)) if len(md_err) else float("nan"),
        donor_metal_donor_angle_mae_deg=float(np.mean(angle_err)) if angle_err else float("nan"),
        donor_metal_donor_angle_max_error_deg=float(np.max(angle_err)) if angle_err else float("nan"),
        ligand_internal_rmsd_angstrom=float(np.mean(lig_rmsds)) if lig_rmsds else float("nan"),
        min_nonbonded_distance_angstrom=min_nb,
        severe_clash_count=clashes,
        coordination_retained_fraction=retained,
        symmetry_permutation=perm,
    )
