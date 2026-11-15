"""Strict, chemistry-typed HB labels and 3D opportunity features for NCI-CTX1.

The selected native contacts are deliberately conservative training labels, not
a universal definition of hydrogen bonding. Their score surface is fit later
against matched pure-repulsive solver poses. The third coordinate records the
approach direction relative to the acceptor's local outward axis.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from nci_hb_chemistry import candidates as typed_candidates


POSITIVE_MAX_HA = 2.50
POSITIVE_MIN_DHA_DEG = 150.0
POSITIVE_MIN_ACCEPTOR_COS = 0.0
SUPPORT_MAX_HA = 3.50


@dataclass(frozen=True)
class HBFeature:
    donor: int
    hydrogen: int
    acceptor: int
    keys: tuple[tuple[str, ...], ...]
    r_ha: float
    cos_dha: float
    cos_acceptor: float
    acceptor_axis_defined: bool

    @property
    def dha_deg(self) -> float:
        return math.degrees(math.acos(float(np.clip(self.cos_dha, -1.0, 1.0))))

    @property
    def is_high_confidence(self) -> bool:
        return (
            self.r_ha <= POSITIVE_MAX_HA
            and self.dha_deg >= POSITIVE_MIN_DHA_DEG
            and (
                not self.acceptor_axis_defined
                or self.cos_acceptor >= POSITIVE_MIN_ACCEPTOR_COS
            )
        )


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def acceptor_axis(mol, acceptor: int, X: np.ndarray, pos: dict[int, int], neighbors=None,
                  bound: set[int] | None = None) -> np.ndarray | None:
    """Return an outward, graph-defined acceptor axis; ``None`` means isotropic/undefined.

    The axis points away from the normalized sum of covalent-neighbor directions.
    Explicit attached hydrogens supply the local frame for water/ammonia. A
    primary metal-bound acceptor includes its M--A vector in the local context.
    No coordinates are used to assign chemical identity.
    """
    from jacobik.topology.graph import get_ligand_neighbors

    nb = neighbors if neighbors is not None else get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    ids = list(nb.get(acceptor, ()))
    if bound and acceptor in bound:
        ids.append(mol.metal_idx)
    if not ids:
        return None
    a = X[pos[acceptor]]
    vectors = [_unit(X[pos[j]] - a) for j in ids if j in pos]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return None
    return _unit(-np.sum(vectors, axis=0))


def features(mol, X: np.ndarray | None = None, chemistry: dict | None = None) -> list[HBFeature]:
    """Build graph-authoritative typed candidates and their continuous geometry features."""
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors

    coords = np.asarray([a.position for a in mol.atoms], dtype=np.float64) if X is None else np.asarray(X, dtype=np.float64)
    if coords.shape != (len(mol.atoms), 3):
        raise ValueError(f"expected coordinates {(len(mol.atoms), 3)}, got {coords.shape}")
    pos = {a.idx: i for i, a in enumerate(mol.atoms)}
    nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    bound = set(donor_indices_of(mol))
    out: list[HBFeature] = []
    axes: dict[int, np.ndarray | None] = {}
    typed = chemistry if chemistry is not None else __import__("nci_hb_chemistry").from_training(mol)
    for d, h, a, ladder in typed_candidates(mol, typed):
        pd, ph, pa = coords[pos[d]], coords[pos[h]], coords[pos[a]]
        v_dh = pd - ph
        v_ha = pa - ph
        r = float(np.linalg.norm(v_ha))
        if r < 1e-9:
            continue
        c_dha = float(np.dot(v_dh, v_ha) / max(float(np.linalg.norm(v_dh)) * r, 1e-12))
        if a not in axes:
            axes[a] = acceptor_axis(mol, a, coords, pos, nb, bound)
        axis = axes[a]
        c_acceptor = 0.0 if axis is None else float(np.dot((ph - pa) / r, axis))
        out.append(HBFeature(d, h, a, tuple(tuple(k) for k in ladder), r,
                             float(np.clip(c_dha, -1.0, 1.0)),
                             float(np.clip(c_acceptor, -1.0, 1.0)), axis is not None))
    return out


def select_high_confidence(items: list[HBFeature]) -> list[HBFeature]:
    """Select one strongest high-confidence acceptor per donor H.

    Selection is fixed before any energy/model comparison. If equivalent
    acceptors pass the strict gate for the same donor hydrogen, keep the most
    linear and then shortest contact, matching the later acceptor-choice
    aggregation and avoiding contact multiplicity in the positive examples.
    """
    by_h: dict[int, list[HBFeature]] = defaultdict(list)
    for item in items:
        if item.is_high_confidence:
            by_h[item.hydrogen].append(item)
    out = []
    for h in sorted(by_h):
        out.append(sorted(by_h[h], key=lambda x: (x.dha_deg, -x.r_ha, x.cos_acceptor), reverse=True)[0])
    return out


def high_confidence_contacts(mol, X: np.ndarray | None = None, chemistry: dict | None = None) -> list[HBFeature]:
    """Build strict labels and keep at most one acceptor per donor hydrogen."""
    return select_high_confidence(features(mol, X, chemistry))


def key_for(item: HBFeature, level: int = 0) -> tuple[str, ...]:
    """Select one charge-preserving fallback key from the chemistry ladder."""
    if not 0 <= level < len(item.keys):
        raise IndexError(f"key level {level} outside ladder of length {len(item.keys)}")
    return item.keys[level]

