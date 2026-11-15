"""Coordination-integrity check of generated structures against a reference (first shell of the metal).

Moved from ``research/archive/code/refit/nci_v2_pilot.py`` (which re-exports it); the diagnostics of the jgen bench use it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from jacobik.chem.elements import METALS, covalent_radius as _covalent_radius

COORD_BOUND_SCALE = 1.15      # reference metal-bound atom: d <= 1.15 * (r_M + r_X); 1.30 admitted M...C secondary contacts
COORD_SHIFT_TOL_A = 0.30      # sigma donor distance shift / haptic slip tolerance


def coordination_integrity(ref_symbols: Sequence[str], ref_X: np.ndarray, X_mapped: np.ndarray) -> Dict[str, Any]:
    """Compare first-shell coordination of reference-ordered generated conformers with the reference.

    Metal-bound heavy atoms are taken from the reference and grouped into donor sets (connected via
    ligand bonds): single atoms are sigma donors, larger sets are haptic. Per conformer:
      sigma shift  = |d_gen - d_ref|
      haptic slip  = (max - min of M-X distances, gen) - (same, ref)   [eta-n -> eta-k slip]
      haptic shift = |mean d_gen - mean d_ref|
    A conformer is intact when all three stay <= COORD_SHIFT_TOL_A.
    """
    sym = list(ref_symbols)
    X_mapped = np.asarray(X_mapped, dtype=np.float64).reshape(-1, len(sym), 3)
    m = next(i for i, s in enumerate(sym) if s in METALS)
    d_ref = np.linalg.norm(ref_X - ref_X[m], axis=1)
    bound = [i for i, s in enumerate(sym)
             if i != m and s != "H" and d_ref[i] <= COORD_BOUND_SCALE * (_covalent_radius(sym[m]) + _covalent_radius(s))]
    bset = set(bound)
    adj: Dict[int, List[int]] = {i: [] for i in bound}
    for a in bound:
        for b in bound:
            if a < b and np.linalg.norm(ref_X[a] - ref_X[b]) <= 1.25 * (_covalent_radius(sym[a]) + _covalent_radius(sym[b])) + 0.25:
                adj[a].append(b)
                adj[b].append(a)
    groups, seen = [], set()
    for a in bound:
        if a in seen:
            continue
        comp, stack = [], [a]
        seen.add(a)
        while stack:
            c = stack.pop()
            comp.append(c)
            for x in adj[c]:
                if x in bset and x not in seen:
                    seen.add(x)
                    stack.append(x)
        groups.append(sorted(comp))

    d_gen = np.linalg.norm(X_mapped - X_mapped[:, m:m + 1, :], axis=2)
    n = X_mapped.shape[0]
    sigma_shift = np.zeros(n)
    haptic_slip = np.zeros(n)
    haptic_shift = np.zeros(n)
    for g in groups:
        dr, dg = d_ref[g], d_gen[:, g]
        if len(g) == 1:
            sigma_shift = np.maximum(sigma_shift, np.abs(dg[:, 0] - dr[0]))
        else:
            haptic_slip = np.maximum(haptic_slip, (dg.max(axis=1) - dg.min(axis=1)) - (dr.max() - dr.min()))
            haptic_shift = np.maximum(haptic_shift, np.abs(dg.mean(axis=1) - dr.mean()))
    intact = (sigma_shift <= COORD_SHIFT_TOL_A) & (haptic_slip <= COORD_SHIFT_TOL_A) & (haptic_shift <= COORD_SHIFT_TOL_A)
    return {
        "n_sigma_donors": sum(len(g) == 1 for g in groups),
        "haptic_group_sizes": [len(g) for g in groups if len(g) > 1],
        "sigma_shift_A": sigma_shift,
        "haptic_slip_A": haptic_slip,
        "haptic_shift_A": haptic_shift,
        "intact": intact,
    }
