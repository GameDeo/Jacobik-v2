"""Binding policy for the final relaxation (EnsembleResult.relax(binding=...)).

"keep" (default): relax freely inside the requested binding mode, never out of it. Each declared donor (coordination
sites, every haptic atom) is held inside its bonding range (covalent sum + BOUND_PAD, the a4c contact rule); every other
heavy atom that starts outside the bonding range may not move into it (by more than SHIFT), and one already inside may
not come closer than the covalent sum (a bond), except atoms bonded to a declared donor (they follow it) and atoms conjugated with a declared haptic group (a hapticity slip on the same pi system, e.g. Cp* eta3 -> eta5, is
allowed and reported). Restraints are flat-bottom: inside the allowed region the relaxation is unconstrained, so all
strain within the mode is removed. A row whose restraint is still active at the end is flagged "held": in the gas phase
the requested mode is not a minimum there (e.g. methanol trans to an oxo, an axial aqua ligand).
"free": unconstrained; rows whose metal contacts change are reported (binding_changed) and keep their class label.

Owner review 2026-11-19: 11.9 % of settled rows changed binding on unconstrained relaxation (cal/tail, 115 parents).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

BOUND_PAD = 0.6           # A over the covalent sum
HOLD_K = 10.0             # eV/A^2 beyond the flat bottom
HELD_TOL = 0.02           # A beyond a bound at the end (0.2 eV/A at HOLD_K): the row is "held". Calibrated 2026-11-19 on
                          # 1743 settled rows of 17 cal/tail parents: 37 % of rows that leave their mode when free, 10 %
                          # of rows that keep it (weak residual pushes on long dative bonds)
SHIFT = 0.3               # A: a contact change also needs this much movement (a4c CONTACT_SHIFT_A)
EDGE_TOL = 0.1            # A past the cut before a declared donor counts as unbound (flat-bottom overshoot)
MARGIN = 0.05             # A: holds sit this far on the safe side of the cut (overshoot at 0.5 eV/A and HOLD_K)


def _rcov(symbols):
    from jacobik.core.relax import MACERelaxer

    return np.asarray(MACERelaxer._rcov(list(symbols)), float)


def _declared(mol):
    m = int(mol.metal_idx) - 1
    return m, sorted({int(d) - 1 for s in (mol.coord_sites or []) for d in s.donor_indices})


def _pi_extension(mol, m, declared) -> set:
    """0-based atoms conjugated (bond order >= 1.5, not through the metal) with a declared haptic atom."""
    hap = {int(d) - 1 for s in (mol.coord_sites or []) if getattr(s, "kind", "") == "haptic" for d in s.donor_indices}
    if not hap:
        return set()
    adj = {}
    for b in getattr(mol, "bonds", None) or []:
        i, j = int(b.i) - 1, int(b.j) - 1
        if m in (i, j) or float(b.order) < 1.5:
            continue
        adj.setdefault(i, []).append(j); adj.setdefault(j, []).append(i)
    seen, st = set(hap), list(hap)
    while st:
        a = st.pop()
        for b in adj.get(a, ()):
            if b not in seen:
                seen.add(b); st.append(b)
    return seen - set(declared)


def _donor_neighbours(mol, m, declared) -> set:
    """0-based atoms bonded (not through the metal) to a declared donor: they come in with the donor."""
    dec = set(declared)
    out = set()
    for b in getattr(mol, "bonds", None) or []:
        i, j = int(b.i) - 1, int(b.j) - 1
        if m in (i, j) or float(b.order) < 0.75:
            continue
        if i in dec:
            out.add(j)
        if j in dec:
            out.add(i)
    return out - dec


def keep_restraints(mol, X) -> Optional[np.ndarray]:
    """(n, 5) flat-bottom restraints (i, j, lo, hi, k) of the "keep" policy for complex mol at positions X; None when the
    complex declares no donors."""
    try:
        m, dec = _declared(mol)
    except (AttributeError, TypeError, ValueError):
        return None
    if not dec:
        return None
    sym = [a.symbol for a in mol.atoms]
    rc = _rcov(sym)
    X = np.asarray(X, float)
    d = np.linalg.norm(X - X[m], axis=1)
    out = [(m, j, 0.0, rc[m] + rc[j] + BOUND_PAD - MARGIN, HOLD_K) for j in dec]
    skip = set(dec) | _pi_extension(mol, m, dec) | {m}
    nbr = _donor_neighbours(mol, m, dec)
    for j, s in enumerate(sym):
        if j in skip or s == "H":
            continue
        cut = rc[m] + rc[j] + BOUND_PAD
        # inside already, or bonded to a donor (it follows the donor in: JAYCEV pyrimidine C17 / C24): no closer than a
        # bond; otherwise it may not move into the bonding range
        lo = rc[m] + rc[j] if (d[j] <= cut or j in nbr) else min(cut + MARGIN, d[j] - SHIFT + EDGE_TOL)
        out.append((m, j, lo, 1e6, HOLD_K))
    return np.asarray(out, float)


def report(mol, X0, X1, restraints=None) -> dict:
    """Binding record of one relaxed row: held (restraints still carrying force: 1-based atom, distance, bound),
    lost / gained metal contacts (1-based) between X0 and X1 by the bonding-range rule, hapticity changes included."""
    m, dec = _declared(mol)
    sym = [a.symbol for a in mol.atoms]
    rc = _rcov(sym)
    X0, X1 = np.asarray(X0, float), np.asarray(X1, float)
    d0, d1 = np.linalg.norm(X0 - X0[m], axis=1), np.linalg.norm(X1 - X1[m], axis=1)
    cut = rc[m] + rc + BOUND_PAD
    heavy = [j for j, s in enumerate(sym) if s != "H" and j != m]
    moved = np.abs(d1 - d0) > SHIFT
    lost = [j + 1 for j in heavy if d0[j] <= cut[j] < d1[j] and moved[j]]
    gained = [j + 1 for j in heavy if d1[j] <= cut[j] < d0[j] and moved[j]]
    held: List[tuple] = []
    if restraints is not None:
        for i, j, lo, hi, _k in np.asarray(restraints, float):
            j = int(j)
            if d1[j] > hi + HELD_TOL:
                held.append((j + 1, round(float(d1[j]), 3), round(float(hi), 3)))
            elif d1[j] < lo - HELD_TOL:
                held.append((j + 1, round(float(d1[j]), 3), round(float(lo), 3)))
    unbound = [j + 1 for j in dec if d1[j] > cut[j] + EDGE_TOL]
    pen = 0.0
    if restraints is not None:
        for i, j, lo, hi, _k in np.asarray(restraints, float):
            pen = max(pen, d1[int(j)] - hi, lo - d1[int(j)])
    skip = set(dec) | _pi_extension(mol, m, dec) | _donor_neighbours(mol, m, dec)
    new_contacts = [j for j in gained if (j - 1) not in skip]
    return dict(held=held, lost=lost, gained=gained, unbound_declared=unbound, new_contacts=new_contacts,
                hold_force_ev_A=round(float(HOLD_K * pen), 4), binding_changed=bool(lost or gained),
                mode_intact=not unbound and not new_contacts)
