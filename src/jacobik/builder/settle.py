"""Settle stage (profile flag settle_steps, off): a few batched MACE-POLAR steps on an enlarged candidate pool, then
output by settled energy.

Measured 2026-11-19 on the 117 cal/tail control pools (7248 rows): rows leave the solver ~100 kJ/mol above the relaxed
minimum of their class, and that strain varies more between rows (IQR 24 kJ/mol) than the relaxed energies of the
distinct basins of a class (IQR 10). Within a class neither the score (Spearman 0.15 against the relaxed energy) nor the
exact MACE energy at the solver geometry (0.31) can then rank basins. After 5 / 10 MACE steps: 0.44 / 0.55, and the
lowest-energy row of a class sits 17 / 7 kJ/mol above the class relaxed minimum (score-best raw row: ~100).

ComplexBuilder.solve with settle_steps = k > 0:
1. solves with a candidate pool settle_pool times the requested size (n_conformers, or out_per_class_k);
1b. settle_top = k (default None: every row): in every class with more than k rows only the k rows of lowest MACE
   energy at the solver geometry are settled and returned (one batched evaluation per row, ~1/10 of settling it).
   That energy is a better pick than the score: on the 2026-11-19 pools (117 cal/tail parents, 628 classes) the class
   leader relaxed within 1 kcal/mol of its class minimum in 0.814 of classes settling every row, 0.812 / 0.804 with the
   12 / 8 lowest-energy rows per class (47 / 36 % of the rows), 0.791 / 0.787 with the 12 / 8 best-scored rows; the
   ensemble's first row 0.803 all, 0.803 / 0.786 by energy, 0.718 / 0.718 by score. The best rows across the
   ensemble instead of per class did worse at equal cost (32 rows, 42 %: leader 0.780);
2. settles every (remaining) row with k steps of core.relax.MACERelaxer (fast evaluation on CUDA), each declared metal-donor
   distance (coordination sites) held
   by a flat-bottom restraint of +/- settle_hold A so a row keeps its binding mode, never reaching past the bonding
   range (covalent sum + 0.6 A), so a donor the solver left unbound is pulled in;
2b. drops rows whose declared donor is still beyond the bonding range (not the requested binding mode), unless no
   bound row remains;
3. keeps every row (settle_dedup, off: deduplicating the settled rows cost 12 % of the low relaxed basins on cal/tail,
   since rows that settle close together still relax to different basins);
4. keeps every class's (class_label) lowest settled row, then fills by settled energy up to the requested cap (no
   cap, the default: every distinct row; the output window is then the only limit, so settle_pool has no effect);
   classes ranked by their lowest settled energy, rows within a class by energy.
Rows carry the settled coordinates and a relaxation record (level "settle<k>", kJ/mol, dE_class_kj).
Any failure (no MACE, out of memory, a model error) returns the solver ensemble unchanged, with a warning.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import replace as _replace
from typing import Optional

import numpy as np

from jacobik.core import profile, timing

log = logging.getLogger(__name__)

SETTLE_STEPS = int(profile.get("settle_steps") or 0)
SETTLE_POOL = float(profile.get("settle_pool"))
SETTLE_HOLD = float(profile.get("settle_hold"))
SETTLE_DEDUP = bool(profile.get("settle_dedup"))
SETTLE_CARRY = bool(profile.get("relax_carry"))
_HOLD_K = 10.0            # eV/A^2 beyond the flat bottom
_BOUND_PAD = 0.6          # A over the covalent sum: a declared donor beyond it is not bound (the a4c contact rule)
_FMAX = 0.05              # eV/A: rows already converged stop early
MIRROR = np.array([-1.0, 1.0, 1.0])     # builder.assemble.MIRROR

def _relaxer(device: str):
    """the process's float32 relaxer on this device (core.relax.shared; the final relaxation uses the same one)"""
    from jacobik.core.relax import shared

    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    return shared(dev, "float32", fast=(dev == "cuda"))


def _screen(rows, top: int, rx):
    """settle_top: (rows kept, rows screened). Classes with more than `top` rows keep their `top` rows of lowest MACE
    energy at the solver geometry (one evaluation each: a relax call stopped after its first evaluation); smaller
    classes are kept whole without an evaluation. Ties and non-finite energies fall back to the solver rank. A failed
    evaluation keeps the `top` best-ranked rows of the class (warning), never fails the stage."""
    by = {}
    for i, c in enumerate(rows):
        by.setdefault(c.class_label, []).append(i)
    big = [i for idx in by.values() if len(idx) > top for i in idx]
    if not big:
        return rows, 0
    e0 = {}
    try:
        structs = [([a.symbol for a in rows[i].complex.atoms], np.array([a.position for a in rows[i].complex.atoms], float),
                    int(rows[i].complex.charge), int(rows[i].complex.spin_state)) for i in big]
        for i, r in zip(big, rx.relax(structs, fmax=_FMAX, max_steps=1)):
            e0[i] = float(r.energy0_ev) if np.isfinite(r.energy0_ev) else float("inf")
    except Exception as exc:                                  # noqa: BLE001
        log.warning("settle_top screen failed (%s); the best-ranked rows of each class are settled", exc)
        e0 = {}
    keep = []
    for idx in by.values():
        keep += idx if len(idx) <= top else sorted(idx, key=lambda i: (e0.get(i, float("inf")), i))[:top]
    return [rows[i] for i in sorted(keep)], len(big)


def enlarged(n: Optional[int], factor: float) -> Optional[int]:
    """The solver request for a cap of n rows. No cap (None): the output window already returns every distinct row in
    it, and rows outside it are mostly > 1000 kJ/mol off, so there is nothing to enlarge."""
    return None if n is None else max(int(n), int(math.ceil(n * factor)))


def _declared_pairs(mol):
    """0-based (metal, donor) pairs from the complex's coordination sites (kappa donors and every haptic atom), or None
    when the complex carries none."""
    try:
        m = int(mol.metal_idx) - 1                                # TMComplex indices are 1-based
        pairs = [(m, int(d) - 1) for s in (mol.coord_sites or []) for d in s.donor_indices]
    except (AttributeError, TypeError, ValueError):
        return None
    return pairs or None


def _holds(symbols, X, half_width: float, mol=None):
    """(m, 5) flat-bottom restraints (i, j, lo, hi, k) on every metal-donor distance: the complex's declared donors
    (coordination sites), else heavy atoms whose start distance is below 1.3 x the covalent sum (the preconditioner's
    metal-donor rule). The geometric rule also held non-donor atoms that the solver left close to the metal (HOXJUZ:
    four thione C at 2.1-2.4 A, 3.0-3.1 A in the crystal and after free relaxation, ~180 kJ/mol), 2026-11-19."""
    from jacobik.core.relax import _NONMETALS, MACERelaxer

    if half_width <= 0:
        return None
    pairs = _declared_pairs(mol) if mol is not None else None
    if pairs:
        # the window never reaches past the bonding range (covalent sum + _BOUND_PAD): a donor the solver left
        # unbound is pulled back in instead of held out (2026-11-19: JAYCEV Re-N 2.95-3.2 A, CEXPOM Zn-OH2 3.7-3.9 A,
        # AWONOR one olefin 2.7-3.0 A from Fe, all held there by a window centred on the start distance)
        rc = np.asarray(MACERelaxer._rcov(list(symbols)), float)
        out = []
        for i, j in pairs:
            d = float(np.linalg.norm(X[i] - X[j]))
            hi = min(d + half_width, rc[i] + rc[j] + _BOUND_PAD)
            out.append((i, j, min(d - half_width, hi - 2 * half_width), hi, _HOLD_K))
        return np.asarray(out, float)
    metal = np.array([s not in _NONMETALS for s in symbols])
    if not metal.any():
        return None
    rc = np.asarray(MACERelaxer._rcov(list(symbols)), float)               # Angstrom
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    out = []
    for m in np.flatnonzero(metal):
        for j in range(len(symbols)):
            if j == m or metal[j] or symbols[j] == "H":
                continue
            if D[m, j] < 1.3 * (rc[m] + rc[j]):
                out.append((m, j, D[m, j] - half_width, D[m, j] + half_width, _HOLD_K))
    return np.asarray(out, float) if out else None


def unbound_donors(mol, X) -> list:
    """1-based indices of the complex's declared donors farther from the metal than covalent sum + _BOUND_PAD."""
    from jacobik.core.relax import MACERelaxer

    pairs = _declared_pairs(mol)
    if not pairs:
        return []
    sym = [a.symbol for a in mol.atoms]
    rc = np.asarray(MACERelaxer._rcov(sym), float)
    return [j + 1 for i, j in pairs if np.linalg.norm(X[i] - X[j]) > rc[i] + rc[j] + _BOUND_PAD]


@timing.timed("settle")
def settle(res, steps: int, device: str = "cpu", pool: float = 1.0, hold: Optional[float] = None,
           dedup_rmsd: Optional[float] = None, n_request: Optional[int] = None, dedup_on: Optional[bool] = None,
           per_class: Optional[int] = None, top: Optional[int] = None):
    """The settle stage on a solved EnsembleResult (see the module docstring); never raises. top: settle_top (step
    1b; None: every row)."""
    if steps <= 0 or not getattr(res, "conformers", None):
        return res
    rows = sorted(res.conformers, key=lambda c: c.rank)
    hold = SETTLE_HOLD if hold is None else hold
    dedup_on = SETTLE_DEDUP if dedup_on is None else dedup_on
    try:
        import torch

        from jacobik.core.relax import EV_TO_KJ_MOL
        from jacobik.ik import dedup

        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()                     # the solver's cached blocks
        rx = _relaxer(device)
        n_screened = 0
        if top is not None:
            rows, n_screened = _screen(rows, int(top), rx)
        # a reflected row (out_mirror completion: exactly X * MIRROR of its source) is settled through its source and
        # reflected back: the MACE energy is reflection-invariant, so the settled pair stays an exact mirror pair
        Xs = [np.array([a.position for a in c.complex.atoms], float) for c in rows]
        src = {X.tobytes(): i for i, (c, X) in enumerate(zip(rows, Xs)) if c.mirror_of is None}
        via = {}
        for i, (c, X) in enumerate(zip(rows, Xs)):
            j = src.get((X * MIRROR).tobytes()) if c.mirror_of is not None else None
            if j is not None:
                via[i] = j
        todo = [i for i in range(len(rows)) if i not in via]
        structs, rs = [], []
        for i in todo:
            c = rows[i]
            sym = [a.symbol for a in c.complex.atoms]
            structs.append((sym, Xs[i], int(c.complex.charge), int(c.complex.spin_state)))
            rs.append(_holds(sym, Xs[i], hold, c.complex))
        got = rx.relax(structs, fmax=_FMAX, max_steps=int(steps),
                       restraints=rs if any(r is not None for r in rs) else None, keep_state=SETTLE_CARRY)
        rec_of = dict(zip(todo, got))
        out = []
        for i, c in enumerate(rows):
            r = rec_of[via.get(i, i)]
            pos = r.positions * MIRROR if i in via else r.positions
            mol = copy.deepcopy(c.complex)
            ok = bool(np.isfinite(r.energy_ev))
            for a, x in zip(mol.atoms, pos):
                a.position = np.asarray(x, dtype=float)
            carry = None if r.state is None else (r.state.reflected(MIRROR) if i in via else r.state)
            out.append(_replace(c, complex=mol, opt_state=carry, relaxation=dict(
                level=f"settle{int(steps)}", energy_kj=r.energy_ev * EV_TO_KJ_MOL if ok else float("inf"),
                energy0_kj=r.energy0_ev * EV_TO_KJ_MOL, released_kj=r.released_kj, fmax_ev_A=r.fmax,
                n_eval=r.n_eval, converged=r.converged, reason=r.reason)))
        # a row whose declared donor is still unbound after settling is not the requested binding mode: dropped while
        # any bound row remains (never all rows: the generator degrades, it does not return nothing)
        unb = [unbound_donors(c.complex, np.array([a.position for a in c.complex.atoms], float)) for c in out]
        for c, u in zip(out, unb):
            c.relaxation["unbound_donors"] = u
        n_unbound = sum(1 for u in unb if u)
        bound = [i for i, u in enumerate(unb) if not u]
        if n_unbound and bound:
            out = [out[i] for i in bound]
        elif n_unbound:
            log.warning("settle: every row has an unbound declared donor; rows kept")
        E = np.array([c.relaxation["energy_kj"] for c in out], float)
        X_all = torch.as_tensor(np.stack([[a.position for a in c.complex.atoms] for c in out]), dtype=torch.float64)
        if dedup_on:   # off (default): no dedup call at all (it was computed and discarded: QEZVEY 213 s, 2026-11-20)
            keep = dedup.deduper_for(out[0].complex).select(
                X_all, torch.as_tensor(-np.nan_to_num(E, nan=1e30, posinf=1e30)), None, th=dedup_rmsd)
        kept = sorted({int(i) for i in keep} if dedup_on else range(len(out)), key=lambda i: E[i])
        # every class keeps its lowest settled row; the rest fill by settled energy up to the requested cap (no cap:
        # all distinct rows)
        firsts, rest_ = [], []
        seen = {}
        for i in kept:
            lab = out[i].class_label
            m = seen.get(lab, 0)
            if per_class is not None and m >= int(per_class):           # out_per_class: at most k per class
                continue
            (rest_ if m else firsts).append(i)
            seen[lab] = m + 1
        n_out = len(kept) if n_request is None else max(len(firsts), int(n_request))
        take = firsts + rest_[:max(0, n_out - len(firsts))]
        sel = [out[i] for i in take]
        best = {}
        for c in sel:
            best[c.class_label] = min(best.get(c.class_label, float("inf")), c.relaxation["energy_kj"])
        for c in sel:
            c.relaxation["dE_class_kj"] = c.relaxation["energy_kj"] - best[c.class_label]
        sel.sort(key=lambda c: (best[c.class_label], c.class_label or "", c.relaxation["energy_kj"]))
        sel = [_replace(c, rank=i) for i, c in enumerate(sel, start=1)]
        new = _replace(res, conformers=sel)
        new.relax_stats = dict(rx.stats, settle_steps=int(steps), settle_pool=pool, settle_top=top,
                               n_screened=n_screened, n_candidates=len(out),
                               n_settled=len(todo), n_distinct=len(kept),
                               n_unbound=n_unbound, n_unbound_dropped=n_unbound if bound else 0)
        log.info("Settle: %d candidates, %d distinct, %d kept (%d MACE steps)", len(out), len(kept), len(sel), steps)
        return new
    except Exception as exc:                                   # the generator never fails: degrade
        log.warning("settle stage failed (%s: %s); returning the solver ensemble", type(exc).__name__, exc)
        return res
