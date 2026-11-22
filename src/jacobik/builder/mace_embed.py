"""MACE refinement of the ligand embeds (profile field embed_refine), batched: the replacement of the GFN2 embed refinement.

Every ligand embed of a call is refined together in one batched MACE-POLAR optimisation (core.relax.MACERelaxer):
the engine first collects the refinement requests of every configuration class (`collecting`), refines them in one
pass (`flush`), and the actual shell builds then read the results from the cache. A request outside a collect pass
is refined on its own (a batch of one).

Mode "mace" (the only one): the GFN2 refinement on the MACE surface: the free ligand (with the temporary Li caps of
the GFN2 embed on anionic donors) relaxed with every donor-donor distance held (stiff restraints), as xtb's $constrain
does. (Refining the ligand together with its metal was tried, 2026-11-18, and dropped: on 60 cal/tail parents top-1 dE
went +52 kJ/mol median, CI [23, 121], with broken structures; earlier attempts failed the same way.)
Any failure (MACE not installed, an evaluation error, a non-finite result) falls back to the GFN2 refinement for that
ligand: the generator never fails here.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import numpy as np

from jacobik.core import timing

log = logging.getLogger(__name__)

MODES = ("mace",)
#: optimiser budget of the embed refinement (xtb --opt crude converges the gradient to 1e-2 Eh/bohr = 0.51 eV/A); a
#: batch runs as long as its slowest member, so the step cap bounds the pass (FMAX 0.10 / 150 steps took 129 s on
#: FASPOH's 4 ligands, local 2060, with the batch at the cap); model-Hessian preconditioner (core.relax "ff")
FMAX, MAX_STEPS, PRECON = 0.25, 60, "ff"
#: donor-donor hold (mode "mace"): 1 Eh/bohr^2 as xtb's $constrain force constant=1.0 (eV/A^2)
K_HOLD = 97.17
#: The embed refinement runs in float64 and its coordinates are rounded to 1e-6 A: float32 MACE differs run to run by
#: 1e-5 to 1e-4 A (CPU and CUDA, even with deterministic algorithms), and the joint IK search amplifies that into
#: different output rows, so the same seed did not reproduce (Co(en)3, 2026-11-20). float64 agrees to 1e-12 A; cost
#: about 1.9x on an RTX 2060 for this pass (0.4-2.5 s per parent on an A100 in float32).
DTYPE, ROUND_DECIMALS = "float64", 6

_CACHE: Dict[tuple, np.ndarray] = {}
_PENDING: Optional[List[dict]] = None
_CTX: Dict[str, object] = {}
STATS = {"requests": 0, "refined": 0, "batches": 0, "fallback": 0, "wall_s": 0.0}


def set_context(device: Optional[str] = None) -> None:
    """The generator's device for the refinements that follow (ComplexBuilder.solve sets it)."""
    if device is not None:
        _CTX["device"] = str(device)


def relaxer(device: Optional[str] = None):
    """The MACERelaxer of the refinement: JACOBIK_MACE_DEVICE, else the given or the generator's device (set_context),
    else "auto"; one per device for the process (core.relax.shared). The validated embed optimizer (fast=False); on CUDA
    evaluated through the sync-free pair path (same energies). Until 2026-11-21 the device was never set, so a CPU
    solve refined on the GPU whenever one was present (and fell back to GFN2 when that GPU was full)."""
    from jacobik.core.relax import shared

    dev = os.environ.get("JACOBIK_MACE_DEVICE") or device or _CTX.get("device") or "auto"
    return shared(dev, DTYPE, fast=False)


def _key(mode, rdmol_h, positions, donors, experiment_flags):
    from rdkit import Chem

    dn = tuple(tuple(int(a) for a in g) for g in (donors or []))
    return (mode, Chem.MolToSmiles(rdmol_h), positions.shape[0], dn, bool(experiment_flags),
            hash(np.round(np.asarray(positions, dtype=np.float64), 4).tobytes()))


def _request(mode, rdmol_h, positions, donors, experiment_flags) -> dict:
    """symbols, start coordinates, charge, multiplicity and restraints of one refinement."""
    from rdkit import Chem

    from jacobik.builder import embed as E

    P = np.asarray(positions, dtype=np.float64)
    sy = [a.GetSymbol() for a in rdmol_h.GetAtoms()]
    n = len(sy)
    q_lig = Chem.GetFormalCharge(rdmol_h)
    groups = [sorted({int(a) for a in g}) for g in (donors or [])]
    dn = sorted({a for g in groups for a in g})
    eng = E._e()
    use_li = bool(experiment_flags and eng.EMBED_LI_CAP)
    use_hli = bool(experiment_flags and eng.EMBED_HAPTIC_LI_CAP)
    li = E._li_cap_positions(rdmol_h, P, donors, use_li, use_hli, experiment_flags)
    X = np.vstack([P, li]) if li else P
    sy = sy + ["Li"] * len(li)
    q = q_lig + len(li)
    uhf = (sum(a.GetAtomicNum() for a in rdmol_h.GetAtoms()) + 3 * len(li) - q) % 2
    R = [(a, b, float(np.linalg.norm(P[a] - P[b])), float(np.linalg.norm(P[a] - P[b])), K_HOLD)
         for ii, a in enumerate(dn) for b in dn[ii + 1:]]
    return dict(symbols=sy, X=X, q=int(q), mult=int(uhf) + 1, R=np.asarray(R, float).reshape(-1, 5), n=n)


def _aligned(P, ligand):
    """ligand coordinates fitted onto the input frame P (Kabsch; as the GFN2 refinement returns them)."""
    ca, cb = P.mean(0), ligand.mean(0)
    U, _, Vt = np.linalg.svd((ligand - cb).T @ (P - ca))
    Rm = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return (ligand - cb) @ Rm + ca


@contextmanager
def collecting():
    """Within the block refine() records its requests and returns the inputs unrefined; on exit every recorded request
    is refined in one batch into the cache (the builds after it read the cache)."""
    global _PENDING
    outer = _PENDING is not None
    if not outer:
        _PENDING = []
    try:
        yield
    finally:
        if not outer:
            pend, _PENDING = _PENDING, None
            flush(pend)


@timing.timed("mace_embed")
def flush(requests: Sequence[dict]) -> None:
    import time

    todo, seen = [], set()
    for r in requests:
        if r["key"] not in _CACHE and r["key"] not in seen:
            seen.add(r["key"])
            todo.append(r)
    if not todo:
        return
    t0 = time.perf_counter()
    rng = _rng_state()   # loading the model (parameter init) draws random numbers: the seeded solve after it must not see that
    try:
        rx = relaxer()
        recs = rx.relax([(r["req"]["symbols"], r["req"]["X"], r["req"]["q"], r["req"]["mult"]) for r in todo],
                        fmax=FMAX, max_steps=MAX_STEPS, level="normal", restraints=[r["req"]["R"] for r in todo],
                        precon=PRECON)
    except Exception as exc:  # MACE missing or failed as a whole: GFN2 for everything
        log.warning("MACE embed refinement unavailable (%s); GFN2 refinement instead", exc)
        recs = [None] * len(todo)
    STATS["batches"] += 1
    for r, rec in zip(todo, recs):
        out = None
        if rec is not None and not rec.reason.startswith("error") and np.isfinite(rec.positions).all():
            out = np.round(_aligned(r["P"], np.asarray(rec.positions[: r["req"]["n"]], float)), ROUND_DECIMALS)
            STATS["refined"] += 1
            STATS["evals"] = STATS.get("evals", 0) + int(rec.n_eval)
            STATS["unconverged"] = STATS.get("unconverged", 0) + int(not rec.converged)
        else:
            from jacobik.builder import embed as E

            STATS["fallback"] += 1
            out = E._gfn2_refine(r["rdmol_h"], r["P"], r["donors"], experiment_flags=r["flags"])
        _CACHE[r["key"]] = out
    _rng_restore(rng)
    STATS["wall_s"] += time.perf_counter() - t0


def _rng_state():
    from jacobik.core import gpu

    return (np.random.get_state(), gpu.rng_state())


def _rng_restore(st) -> None:
    from jacobik.core import gpu

    try:
        np.random.set_state(st[0])
        gpu.rng_restore(st[1])
    except Exception as exc:  # noqa: BLE001 - never fail the embed on the RNG bookkeeping
        log.debug("RNG state not restored (%s)", exc)


@timing.timed("mace_embed")
def refine(rdmol_h, positions: np.ndarray, donors=None, *, mode: str = "mace", experiment_flags=True) -> np.ndarray:
    """MACE embed refinement of one ligand (cached; collected when inside collecting())."""
    if mode not in MODES:
        raise ValueError(f"embed_refine must be one of {MODES}")
    P = np.asarray(positions, dtype=np.float64)
    key = _key(mode, rdmol_h, P, donors, experiment_flags)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit.copy()
    STATS["requests"] += 1
    try:
        req = _request(mode, rdmol_h, P, donors, experiment_flags)
    except Exception as exc:
        log.warning("MACE embed request failed (%s); GFN2 refinement", exc)
        from jacobik.builder import embed as E

        return E._gfn2_refine(rdmol_h, P, donors, experiment_flags=experiment_flags)
    from rdkit import Chem

    item = dict(key=key, req=req, P=P, rdmol_h=Chem.Mol(rdmol_h), donors=donors, flags=experiment_flags)
    if _PENDING is not None:
        _PENDING.append(item)
        return P.copy()
    flush([item])
    return _CACHE[key].copy()


def refine_many(rdmol_h, confs: Sequence[np.ndarray], donors, *, mode: str = "mace") -> List[np.ndarray]:
    """Refine several conformers of one ligand in one batch (macrocycle pool, macro_pool_refine)."""
    with collecting():
        for P in confs:
            refine(rdmol_h, P, donors, mode=mode)
    return [refine(rdmol_h, P, donors, mode=mode) for P in confs]
