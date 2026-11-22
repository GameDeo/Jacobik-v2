"""Batched MACE-POLAR relaxation of generated conformers (optional final stage; needs the `benchmark` extra).

Every structure of a call is relaxed together: one batched MACE energy + force evaluation per optimiser step for all
structures not yet converged, packed into GPU passes of at most ``max_batch_atoms`` atoms. The model's input batch is
collated once per pass membership; per step only positions and the neighbour list change, and the neighbour list is
built on the device (all intra-structure pairs within the model cutoff), so a step is one model forward/backward.

Optimiser: L-BFGS (memory 10) preconditioned with the exponential connectivity preconditioner of Packwood et al.
(J. Chem. Phys. 144, 164109, 2016): P = mu (L_exp + c I), L_exp the graph Laplacian with weights
exp(-A (r_ij / r_nn - 1)) for r_ij < 2 r_nn. It separates stiff bonded motion from soft collective motion (ligand
rotation, ring folding), the slow modes of a Cartesian optimiser on a complex. Every structure carries its own
trust radius: a step that raises the energy is rejected, the memory is cleared and the radius halved; accepted
steps grow it again. A structure stops when max_i |F_i| <= fmax on an evaluated geometry (the record holds that
geometry with its energy and forces), when its step budget is spent, or when its evaluation fails (the last good
state is kept). Nothing is ever raised to the caller for one bad structure.

Fast path (default on CUDA since v1.0b; ``JACOBIK_RELAX_FAST=1``/``0`` forces it on/off, ``MACERelaxer(fast=...)`` per instance):
evaluations through core.mace_triton (full-pair edges, sync-free long-range block, CUDA-graph replay of one captured
batch, recaptured when the active rows hold half its atoms), the energy summed in float64 from the model's per-atom
terms (the float32 total of ~6e4 eV is quantised to 3.9 meV and jitters by several ulps between neighbouring
geometries and batch memberships, more than the step-rejection tolerance: rows stalled at 200 evaluations with 70 %
of their steps rejected), the accepted energies re-evaluated whenever the batch membership changes, a trust floor of
1e-3 A, and the "ff" preconditioner with metal-donor pairs in its topology (rho 0.3). Each setting can be overridden
by its own variable (JACOBIK_RELAX_PRECON, JACOBIK_RELAX_FF_MD_RHO, JACOBIK_RELAX_TRUST_MIN).

Levels (``level=``): "loose" fmax 0.15 eV/A, 40 evaluations (clears contacts and strain, keeps the basin); "normal"
fmax 0.05 eV/A, 200; "tight" fmax 0.01 eV/A, 600 (run with dtype="float64" for converged energies).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from jacobik.core import timing

log = logging.getLogger(__name__)

EV_TO_KJ_MOL = 96.4853321233

LEVELS = {
    "loose": dict(fmax=0.15, max_steps=40),
    "normal": dict(fmax=0.05, max_steps=200),
    "tight": dict(fmax=0.01, max_steps=600),
}

#: Exp preconditioner (Packwood 2016): decay A, cutoff in units of the nearest-neighbour distance, stabiliser c
_PRECON_A, _PRECON_RCUT, _PRECON_C = 3.0, 2.0, 1.0
#: initial stiffness scale mu (eV/A^2) before the L-BFGS memory supplies its own, and the trust radius (A per atom)
_TRUST_MIN_ENV = os.environ.get("JACOBIK_RELAX_TRUST_MIN")
_MU0, _TRUST0, _TRUST_MIN, _TRUST_MAX = 20.0, 0.2, float(_TRUST_MIN_ENV or "0.01"), 0.3
_MEMORY = 10
#: rebuild the preconditioner when an atom moved this far (A) since the last build
_PRECON_REBUILD_A = 0.3
#: energy rise above which a step is rejected: absolute (eV) plus a share of |E| (float32 round-off of a total energy
#: of -6e4 eV is ~4e-3 eV; an absolute test at 1e-4 eV rejected 70 % of the steps near convergence on noise)
_REJECT_TOL_EV = 1e-4
_REJECT_TOL_REL = {"float32": 2e-7, "float64": 1e-12}
#: research switch: False runs plain L-BFGS (identity preconditioner)
_USE_PRECON = True
#: preconditioner: "exp" (Packwood 2016, isotropic per atom) or "ff" (anisotropic 3n x 3n model Hessian, Lindh et al.,
#: Chem. Phys. Lett. 241, 423 (1995): stretch / bend / torsion terms of the bonded topology with force constants
#: 0.45 rho_ij, 0.15 rho_ij rho_jk, 0.005 rho_ij rho_jk rho_kl Eh (bohr^-2, rad^-2), rho_ij = exp(alpha (r_ref^2 - r^2)))
PRECON = os.environ.get("JACOBIK_RELAX_PRECON", "exp")
_LINDH_ALPHA = np.array([[1.0, 0.3949, 0.3949], [0.3949, 0.28, 0.28], [0.3949, 0.28, 0.28]])
_LINDH_RREF = np.array([[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]])   # bohr
_BOHR = 0.529177210903
_EH = 27.211386245988
#: a pair is bonded for the model Hessian when rho_ij exceeds this. With Lindh's table M-D bonds stay out (rho ~0.02 at
#: 2.1 A); measured (127 cal/tail rows, normal level): exp median 89 evaluations / 65 % converged, ff 35 / 87 %, final
#: energies median 2.6 kJ/mol lower
_LINDH_BOND_RHO = 0.10
#: "ff" reference distances: R_cov(i) + R_cov(j) (any element, metals included) instead of Lindh's per-row table, which
#: leaves M-D bonds (rho ~0.02 at 2.1 A) out of the topology
_FF_RREF_COV = os.environ.get("JACOBIK_RELAX_FF_RREF", "table") == "cov"   # cov tested worse (stiff M-D terms)
#: stabiliser added to the model Hessian (eV/A^2): rigid motions and soft modes
_FF_C = 0.5
#: "ff": metal-donor pairs (start distance < 1.3 x covalent sum) join the topology with this fixed rho (stretch
#: 0.45 rho Eh/bohr^2, bends and torsions through the M-D bond scaled by rho like any other bond); 0 = Lindh only
_FF_MD_RHO_ENV = os.environ.get("JACOBIK_RELAX_FF_MD_RHO")
_FF_MD_RHO = float(_FF_MD_RHO_ENV or "0")
_NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te",
              "I", "Xe", "At", "Rn"}
#: fast evaluation (core.mace_triton): full-pair edges, sync-free long-range block, CUDA-graph replay; on by default.
#: The captured batch is rebuilt when the active rows hold at most this share of its atoms.
#: JACOBIK_RELAX_FAST: "1" fast path everywhere, "0" never; unset (v1.0b default): fast on CUDA, default path on CPU
FAST = {"1": True, "0": False}.get(os.environ.get("JACOBIK_RELAX_FAST", ""), None)


def _fast_default(device) -> bool:
    if FAST is not None:
        return FAST
    d = str(device)
    if d == "auto":
        return True
    return d.startswith("cuda")
#: what the fast path sets unless the environment says otherwise: the ff preconditioner with metal-donor terms
#: (rho 0.3) and a trust floor of 1e-3 A (resolvable once energies are summed in float64; at 1e-2 A rows stalled with
#: every step 0.4 meV uphill). Measured on 8 cal/tail whole pools, A100: see the module docstring.
_FAST_PRECON = os.environ.get("JACOBIK_RELAX_PRECON") or "ff"
_FAST_MD_RHO = float(_FF_MD_RHO_ENV or "0.3")
_FAST_TRUST_MIN = float(_TRUST_MIN_ENV or "1e-3")
_FAST_SHRINK = 0.5
#: a captured batch at or below this many atoms is kept to the end (replay is launch-bound there: ~9 ms for 34 atoms
#: on an A100, while a recapture costs about three evaluations)
_FAST_KEEP_ATOMS = int(os.environ.get("JACOBIK_RELAX_FAST_KEEP", "300"))
#: neighbour list with a skin: pairs within r_max + _FAST_SKIN at capture, rebuilt when a pair left out comes inside r_max.
#: Used when it holds under 3/4 of all pairs (large molecules: 118 atoms, 8 rows: 116 -> 85 ms per replay, 110k ->
#: 65k edges; 40 atoms: no gain)
_FAST_SKIN = 1.5


#: energy stop (relax(etol=...)): a structure stops when its energy fell by less than etol over this many evaluations
ETOL_WINDOW = 5


@dataclass
class OptState:
    """The optimizer state a relaxation ended with, for one structure: the geometry it belongs to, the L-BFGS
    curvature pairs (oldest first) formed on the MACE surface alone (a pair with a restraint active at either end is
    left out: the next relaxation may hold other distances, or none) and the trust radius. relax(warm=...) starts
    from it when the structure's coordinates are exactly `X`; otherwise that structure starts cold."""

    X: np.ndarray                    # (N, 3) A
    s: np.ndarray                    # (m, N, 3) steps
    y: np.ndarray                    # (m, N, 3) force decrements (gradient differences)
    trust: float

    def reflected(self, mirror: np.ndarray) -> "OptState":
        """The state of the mirror image (coordinates times `mirror`, e.g. [-1, 1, 1]): steps and gradients reflect
        the same way, so the pairs stay exact."""
        return OptState(self.X * mirror, self.s * mirror, self.y * mirror, self.trust)

    def fits(self, X: np.ndarray) -> bool:
        return self.X.shape == X.shape and np.array_equal(self.X, X)


@dataclass
class RelaxRecord:
    """Outcome of one structure."""

    positions: np.ndarray            # (N, 3) A: the last evaluated geometry that was accepted
    energy_ev: float                 # MACE energy at ``positions``
    energy0_ev: float                # MACE energy of the input geometry
    fmax: float                      # max |F| (eV/A) at ``positions``
    n_eval: int                      # energy + force evaluations spent on this structure (incl. rejected steps)
    converged: bool                  # fmax criterion met at ``positions``
    reason: str                      # "fmax", "energy" (etol), "max_steps" or "error: ..."
    forces: Optional[np.ndarray] = field(default=None, repr=False)
    state: Optional[OptState] = field(default=None, repr=False)     # relax(keep_state=True)

    @property
    def released_kj(self) -> float:
        """Energy released by the relaxation (kJ/mol, >= 0 for a downhill run)."""
        return (self.energy0_ev - self.energy_ev) * EV_TO_KJ_MOL


def _default_batch_atoms(device, dtype: str) -> int:
    """Atoms per MACE pass based on free GPU memory."""
    free = _allocatable(device)
    per_atom = 4e6 * (2.5 if dtype == "float64" else 1.0)
    return int(max(200, min(20000, 0.6 * free / per_atom)))


#: measured GPU memory of relax passes per (GPU, dtype, graph): sorted (units, peak bytes) points of this
#: process; batches are sized from them instead of a constant (MACERelaxer._cap_units)
_UNIT_BYTES: Dict[tuple, list] = {}


def _dev_index(dev) -> int:
    s = str(dev)
    return int(s.split(":")[1]) if ":" in s else 0


def _allocatable(dev) -> int:
    """Bytes this process can still allocate on `dev`."""
    from jacobik.core import gpu

    gpu.cp.cuda.Device(_dev_index(dev)).use()
    free, _total = gpu.mem_info()
    pool = gpu.cp.get_default_memory_pool()
    avail = free + pool.free_bytes()
    limit = pool.get_limit()                     # JACOBIK_MACE_MEM_FRACTION (energy.MACEPolarEvaluator) or the bench's
    if limit:
        avail = min(avail, limit - pool.used_bytes())
    return int(max(0, avail))


_SHARED: Dict[tuple, "MACERelaxer"] = {}


def shared(device="auto", dtype: str = "float32", fast: Optional[bool] = None,
           max_batch_atoms: Optional[int] = None) -> "MACERelaxer":
    """The process's relaxer for one setting. The MACE embed, the settle stage and the final relaxation take theirs from
    here, so a model is loaded and warmed up once per (device, dtype, optimizer, pass cap), and a run on the CPU never
    loads a model onto the GPU (each caller passes the generator's device)."""
    from jacobik.core.env import resolve_device

    dev = resolve_device(str(device))
    fast_ = _fast_default(dev) if fast is None else bool(fast)
    key = (dev, str(dtype), fast_, max_batch_atoms)
    rx = _SHARED.get(key)
    if rx is None:
        rx = _SHARED[key] = MACERelaxer(device=dev, dtype=dtype, fast=fast_, max_batch_atoms=max_batch_atoms)
    return rx


class MACERelaxer:
    """Loads MACE-POLAR once; ``relax(structures, level=...)`` relaxes any number of structures in batched passes.

    structures: sequence of (symbols, positions (N, 3) A, total charge, spin multiplicity).
    """

    #: instance values come from the constructor; these defaults describe an unconfigured relaxer
    fast = False
    graph = False

    def __init__(self, model: str = "polar-1-s", device: str = "auto", dtype: str = "float32",
                 model_path: Optional[str] = None, max_batch_atoms: Optional[int] = None,
                 fast: Optional[bool] = None, graph: Optional[bool] = None):
        from jacobik.core.energy import MACEPolarEvaluator

        if model_path is None:
            from jacobik.core.env import MACE_MODEL_PATH

            model_path = os.environ.get("JACOBIK_MACE_MODEL_PATH") or (str(MACE_MODEL_PATH) if MACE_MODEL_PATH else None)
        self._ev = MACEPolarEvaluator(model=model, device=device, dtype=dtype, model_path=model_path)
        self.dtype = dtype
        #: fast: optimizer preset (ff preconditioner with metal-donor terms, 1e-3 A trust floor); default on CUDA
        self.fast = _fast_default(device) if fast is None else bool(fast)
        #: graph: capture each Triton MACE pass as a CUDA graph and replay it (CUDA only; default on)
        self.graph = True if graph is None else bool(graph)
        self.max_batch_atoms = max_batch_atoms
        self.stats = {}

    def _mem_key(self):
        from jacobik.core import gpu

        props = gpu.cp.cuda.runtime.getDeviceProperties(_dev_index(self._device))
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        return (name, self.dtype, self.graph)

    def _record_pass(self, units: int, peak_bytes: int) -> None:
        """Keep the passes' (units, peak bytes); the largest two of clearly different size give fixed + per-unit."""
        if units > 0 and peak_bytes > 0:
            pts = _UNIT_BYTES.setdefault(self._mem_key(), [])
            pts.append((int(units), int(peak_bytes)))
            pts.sort()
            del pts[:-8]

    def _record_oom(self, units: int) -> None:
        """A pass of `units` did not fit: recorded as needing all the memory free now (an upper point of the fit)."""
        from jacobik.core import gpu

        used = gpu.cp.get_default_memory_pool().used_bytes()
        self._record_pass(units, int(_allocatable(self._device) + used))

    def _cap_units(self) -> int:
        """Cost units per pass from the memory measured on this GPU / model / dtype / backend."""
        const = _default_batch_atoms(self._device, self.dtype)
        pts = _UNIT_BYTES.get(self._mem_key()) or []
        # the points are measured high-water marks (pool growth + graph pools), so a 10 % margin suffices; a pass that
        # still does not fit is halved by the out-of-memory path
        budget = 0.9 * _allocatable(self._device)
        if len(pts) >= 2 and pts[-1][0] >= 1.5 * pts[0][0]:
            (u1, p1), (u2, p2) = pts[0], pts[-1]
            b = max((p2 - p1) / (u2 - u1), 1.0)
            a = max(p1 - b * u1, 0.0)
            return int(max(1, (budget - a) / b))
        if pts and pts[-1][0] >= 0.5 * const:
            return int(max(1, budget / (pts[-1][1] / pts[-1][0])))
        return const

    # ------------------------------------------------------------------ model plumbing
    def _warm_up(self):
        stats = self.stats
        try:
            self.relax([(["O", "H", "H"], np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]), 0, 1)],
                       level="loose", max_steps=1)
        except Exception as exc:                          # noqa: BLE001
            log.debug("MACE warm-up failed (%s)", exc)
        self.stats = stats

    def _setup(self):
        cm = self._ev._get_compiled()
        self._cm = cm
        self._mdt = cm.dt
        self._device = str(self._ev.device)
        self._rmax = cm.r_max
        logging.getLogger(__name__).info("MACE relaxer: model %s on %s, triton True, fast optimizer %s, graph %s",
                                         self._mdt, self._device, self.fast, self.graph)

    # ------------------------------------------------------------------ preconditioner
    @staticmethod
    def _precon_inv(Xp, mask):
        """Inverse of the Exp preconditioner (without mu) for padded (S, n, 3) positions; (S, n, n)."""
        from jacobik.core import gpu

        cp = gpu.cp
        S, n, _ = Xp.shape
        if not _USE_PRECON:
            return cp.broadcast_to(cp.eye(n, dtype=Xp.dtype), (S, n, n)).copy()
        D = cp.linalg.norm(Xp[:, :, None, :] - Xp[:, None, :, :], axis=-1)
        big = float(cp.finfo(D.dtype).max)
        pm = mask[:, :, None] & mask[:, None, :]
        eye_b = cp.eye(n, dtype=cp.bool_)
        Dn = cp.where(pm & ~eye_b, D, big)
        rnn = cp.clip(Dn.min(axis=-1), a_min=None, a_max=3.0)
        rnn = cp.where(mask, rnn, 0.0)
        r_nn = (rnn.sum(axis=-1) / cp.maximum(mask.sum(axis=-1), 1))[:, None, None]
        w = cp.exp(-_PRECON_A * (D / r_nn - 1.0)) * ((D < _PRECON_RCUT * r_nn) & pm)
        eye = cp.eye(n, dtype=D.dtype)
        w = w * (1.0 - eye)
        deg = w.sum(axis=-1)
        L = cp.zeros((S, n, n), dtype=D.dtype)
        idx = cp.arange(n)
        L[:, idx, idx] = deg + _PRECON_C
        L = L - w
        # padded atoms: identity rows (decoupled)
        diag_pad = cp.zeros((S, n, n), dtype=D.dtype)
        diag_pad[:, idx, idx] = (~mask).astype(D.dtype)
        L = cp.where(pm, L, diag_pad)
        return cp.linalg.inv(L)

    # ------------------------------------------------------------------ model-Hessian preconditioner ("ff")
    @staticmethod
    def _row(sym):
        z = {"H": 1, "He": 1}.get(sym)
        if z is not None:
            return 0
        return 1 if sym in ("Li", "Be", "B", "C", "N", "O", "F", "Ne") else 2

    _RCOV_CACHE: Dict[str, float] = {}

    @classmethod
    def _rcov(cls, symbols):
        cache = cls._RCOV_CACHE
        missing = [s for s in symbols if s not in cache]
        if missing:
            from rdkit.Chem import GetPeriodicTable

            pt = GetPeriodicTable()
            for s in missing:
                cache[s] = pt.GetRcovalent(pt.GetAtomicNumber(s))
        return np.array([cache[s] for s in symbols])

    @classmethod
    def _ff_terms(cls, symbols, X, md_rho=None):
        """Bonded topology of one structure from its start geometry (rho > _LINDH_BOND_RHO): arrays of pairs (i, j, row
        pair), angles (i, j, k) and torsions (i, j, k, l)."""
        rows = np.array([cls._row(s) for s in symbols])
        Xb = np.asarray(X, float) / _BOHR
        D2 = ((Xb[:, None] - Xb[None]) ** 2).sum(-1)
        A = _LINDH_ALPHA[rows[:, None], rows[None]]
        R2 = _LINDH_RREF[rows[:, None], rows[None]] ** 2
        rc = None
        if _FF_RREF_COV:
            rc = cls._rcov(symbols) / _BOHR
            R2 = (rc[:, None] + rc[None]) ** 2
        rho = np.exp(A * (R2 - D2))
        metal = np.array([s not in _NONMETALS for s in symbols])
        md_rho = _FF_MD_RHO if md_rho is None else md_rho
        if md_rho > 0 and metal.any():
            rcv = cls._rcov(symbols)
            md = (metal[:, None] | metal[None]) & (np.sqrt(D2) * _BOHR < 1.3 * (rcv[:, None] + rcv[None]))
            rho = np.where(md, md_rho, rho)
        np.fill_diagonal(rho, 0.0)
        bonded = rho > _LINDH_BOND_RHO
        i, j = np.nonzero(np.triu(bonded, 1))
        nb = [np.nonzero(bonded[a])[0] for a in range(len(symbols))]
        ang_parts = []
        for b, nb_b in enumerate(nb):
            k = len(nb_b)
            if k >= 2:
                ix, iy = np.triu_indices(k, 1)
                ang_parts.append(np.column_stack([nb_b[ix], np.full(len(ix), b, dtype=np.int64), nb_b[iy]]))
        ang_arr = np.concatenate(ang_parts) if ang_parts else np.zeros((0, 3), dtype=np.int64)
        tor_parts = []
        for b, c in zip(i, j):
            nb_b = nb[b]
            nb_c = nb[c]
            aa = nb_b[nb_b != c]
            dd = nb_c[nb_c != b]
            if len(aa) and len(dd):
                A_ = np.repeat(aa, len(dd))
                D_ = np.tile(dd, len(aa))
                keep = A_ != D_
                if keep.any():
                    A_, D_ = A_[keep], D_[keep]
                    tor_parts.append(np.column_stack([A_, np.full(len(A_), b, dtype=np.int64),
                                                      np.full(len(A_), c, dtype=np.int64), D_]))
        tor_arr = np.concatenate(tor_parts) if tor_parts else np.zeros((0, 4), dtype=np.int64)
        return (np.stack([i, j], 1), ang_arr, tor_arr, rows, rc, (metal, md_rho) if md_rho > 0 else None)

    @staticmethod
    def _ff_pack(terms, dev):
        """The per-structure terms of a chunk as flat device arrays (structure index first in every term)."""
        from jacobik.core import gpu

        cp = gpu.cp
        P, A, T, rows, rcs, mts, md_rho = [], [], [], [], [], [], 0.0
        nmax = max((len(t[3]) for t in terms if t[0] is not None), default=1)
        for s, (pairs, ang, tor, rw, rc, metal) in enumerate(terms):
            r = np.zeros(nmax, np.int64)
            c = np.zeros(nmax)
            m = np.zeros(nmax, bool)
            if pairs is not None:
                r[:len(rw)] = rw
                if rc is not None:
                    c[:len(rc)] = rc
                if metal is not None:
                    m[:len(metal[0])] = metal[0]
                    md_rho = metal[1]
                P.append(np.column_stack([np.full(len(pairs), s), pairs]))
                A.append(np.column_stack([np.full(len(ang), s), ang]))
                T.append(np.column_stack([np.full(len(tor), s), tor]))
            rows.append(r); rcs.append(c); mts.append(m)
        cat = lambda L, w: gpu.as_gpu(np.concatenate(L) if L else np.zeros((0, w), np.int64), dtype=cp.int64)
        use_rc = any(t[4] is not None for t in terms if t[0] is not None)
        use_mt = any(t[5] is not None for t in terms if t[0] is not None)
        return dict(pairs=cat(P, 3), ang=cat(A, 4), tor=cat(T, 5),
                    rows=gpu.as_gpu(np.stack(rows), dtype=cp.int64),
                    rc=gpu.as_gpu(np.stack(rcs), dtype=cp.float64) if use_rc else None,
                    metal=gpu.as_gpu(np.stack(mts), dtype=cp.bool_) if use_mt else None, md_rho=md_rho)

    @staticmethod
    def _ff_hessian(Xp, mask, pack, sel=None):
        """(R, 3n, 3n) model Hessian (eV/A^2) of the structures sel (all when None) at padded positions Xp."""
        from jacobik.core import gpu

        cp = gpu.cp
        S, n, _ = Xp.shape
        dt = Xp.dtype
        sel = cp.arange(S, dtype=cp.int64) if sel is None else sel
        alpha = gpu.as_gpu(_LINDH_ALPHA, dtype=dt)
        rref2 = gpu.as_gpu(_LINDH_RREF ** 2, dtype=dt)
        rw, rc, mt = pack["rows"], pack["rc"], pack["metal"]

        def rho(s, a, b):
            d2 = (((Xp[s, a] - Xp[s, b]) / _BOHR) ** 2).sum(axis=-1)
            r2 = rref2[rw[s, a], rw[s, b]] if rc is None else (rc[s, a].astype(dt) + rc[s, b].astype(dt)) ** 2
            r = cp.exp(alpha[rw[s, a], rw[s, b]] * (r2 - d2))
            if mt is not None:
                r = cp.where(mt[s, a] | mt[s, b], pack["md_rho"], r)
            return r

        # every term of every structure contributes k B B^T to its 3m x 3m block; the contributions are summed by one
        # gpu.SegmentSum whose layout (fixed per pack and shape) is built on first use, so H is reproducible
        D3 = 3 * n
        ar3 = cp.arange(3, dtype=cp.int64)
        seg = pack.get("_hseg")
        if seg is not None and seg[0] != (S, n):
            seg = None
        vals, dests = [], []

        def add(s, atoms, Bv, k):
            T_, m = atoms.shape
            if T_ == 0:
                return
            idx = (3 * atoms[..., None] + ar3).reshape(T_, 3 * m)
            Bf = Bv.reshape(T_, 3 * m)
            vals.append((k[:, None, None] * Bf[:, :, None] * Bf[:, None, :]).ravel())
            if seg is None:
                dests.append(((s[:, None, None] * D3 + idx[:, :, None]) * D3 + idx[:, None, :]).ravel())

        p = pack["pairs"]
        if len(p):
            s, a, b = p[:, 0], p[:, 1], p[:, 2]
            v = Xp[s, a] - Xp[s, b]
            u = v / cp.maximum(cp.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
            add(s, cp.stack([a, b], 1), cp.stack([u, -u], 1), 0.45 * rho(s, a, b) * _EH / _BOHR ** 2)
        q = pack["ang"]
        if len(q):
            s, a, b, c = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            u, w = Xp[s, a] - Xp[s, b], Xp[s, c] - Xp[s, b]
            lu = cp.maximum(cp.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
            lw = cp.maximum(cp.linalg.norm(w, axis=-1, keepdims=True), 1e-9)
            eu, ew = u / lu, w / lw
            cos = cp.clip((eu * ew).sum(axis=-1, keepdims=True), -0.999999, 0.999999)
            sin = cp.sqrt(1.0 - cos ** 2)
            ba = (cos * eu - ew) / (lu * sin)
            bc = (cos * ew - eu) / (lw * sin)
            ok = (sin[:, 0] > 0.02).astype(dt)
            add(s, cp.stack([a, b, c], 1), cp.stack([ba, -ba - bc, bc], 1),
                0.15 * rho(s, a, b) * rho(s, b, c) * _EH * ok)
        t = pack["tor"]
        if len(t):
            s, a, b, c, d = t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            f, g, h = Xp[s, a] - Xp[s, b], Xp[s, b] - Xp[s, c], Xp[s, d] - Xp[s, c]
            A_ = cp.cross(f, g, axis=-1)
            B_ = cp.cross(h, g, axis=-1)
            gl = cp.maximum(cp.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
            A2 = cp.maximum((A_ ** 2).sum(axis=-1, keepdims=True), 1e-9)
            B2 = cp.maximum((B_ ** 2).sum(axis=-1, keepdims=True), 1e-9)
            fg = (f * g).sum(axis=-1, keepdims=True)
            hg = (h * g).sum(axis=-1, keepdims=True)
            ta = -gl / A2 * A_
            td = gl / B2 * B_
            tb = gl / A2 * A_ + fg / (A2 * gl) * A_ - hg / (B2 * gl) * B_
            tc = -gl / B2 * B_ - fg / (A2 * gl) * A_ + hg / (B2 * gl) * B_
            ok = ((A2[:, 0] > 1e-3) & (B2[:, 0] > 1e-3)).astype(dt)
            add(s, cp.stack([a, b, c, d], 1), cp.stack([ta, tb, tc, td], 1),
                0.005 * rho(s, a, b) * rho(s, b, c) * rho(s, c, d) * _EH * ok)
        if seg is None:
            seg = pack["_hseg"] = ((S, n), gpu.SegmentSum(cp.concatenate(dests) if dests else cp.zeros(0, cp.int64), S * D3 * D3))
        H = seg[1](cp.concatenate(vals) if vals else cp.zeros(0, dt)).reshape(S, D3, D3)[sel]
        m3 = cp.repeat(mask[sel], 3, axis=1)
        eye = cp.eye(3 * n, dtype=dt)
        H = H + _FF_C * eye
        H = cp.where(m3[:, :, None] & m3[:, None, :], H, eye)
        return H

    def _ff_precon_inv(self, Xp, mask, pack, sel=None):
        from jacobik.core import gpu

        cp = gpu.cp
        H = self._ff_hessian(Xp, mask, pack, sel)
        try:
            L = cp.linalg.cholesky(H)
            if not bool(cp.isfinite(L).all()):
                raise cp.linalg.LinAlgError("non-finite Cholesky")
            Linv = cp.linalg.inv(L)
            return Linv.swapaxes(1, 2) @ Linv
        except cp.linalg.LinAlgError:
            ev, V = cp.linalg.eigh(H)
            ev = cp.maximum(ev, _FF_C)
            return (V * (1.0 / ev)[:, None, :]) @ V.swapaxes(1, 2)

    # ------------------------------------------------------------------ driver
    @timing.timed("relax")
    def relax(self, structures: Sequence[Tuple[Sequence[str], np.ndarray, int, int]], level: str = "normal",
              fmax: Optional[float] = None, max_steps: Optional[int] = None,
              restraints: Optional[Sequence[Optional[np.ndarray]]] = None,
              precon: Optional[str] = None, etol: Optional[float] = None,
              warm: Optional[Sequence[Optional[OptState]]] = None, keep_state: bool = False) -> List[RelaxRecord]:
        """restraints: per structure None or an (m, 5) array of flat-bottom distance restraints (i, j, lo, hi, k):
        energy k/2 (d - hi)^2 above hi, k/2 (lo - d)^2 below lo (eV, A), added to the MACE energy and forces (the
        convergence test and the step acceptance use the total; RelaxRecord.energy_ev stays the MACE energy).
        precon: "exp" or "ff" (default: PRECON, env JACOBIK_RELAX_PRECON).
        etol: energy stop (kJ/mol; None = off): a structure also stops, converged with reason "energy", once its energy
        (with restraints) fell by less than etol over the last ETOL_WINDOW evaluations.
        warm: per structure None or the OptState of an earlier relaxation (RelaxRecord.state) to continue from; used
        only where the coordinates are exactly the state's. keep_state: return each structure's OptState.
        Every sum is order-fixed (graph-local MACE sums, gpu.SegmentSum for the restraints and the ff Hessian), so the
        same input gives the same rows."""
        if structures and not (hasattr(self, "_cm") or hasattr(self, "_model")):
            self._setup()
            self._warm_up()
        return self._relax(structures, level, fmax, max_steps, restraints, precon, etol, warm, keep_state)

    def _relax(self, structures, level, fmax, max_steps, restraints, precon, etol=None, warm=None,
               keep_state=False) -> List[RelaxRecord]:
        from jacobik.core import gpu

        if level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}")
        fmax = float(LEVELS[level]["fmax"] if fmax is None else fmax)
        max_steps = int(LEVELS[level]["max_steps"] if max_steps is None else max_steps)
        if not structures:
            return []
        t0 = time.perf_counter()
        fixed_cap = self.max_batch_atoms
        cap = fixed_cap or self._cap_units()
        ceiling = None                                     # cost units of the smallest pass that ran out of memory
        recs: List[Optional[RelaxRecord]] = [None] * len(structures)
        order = sorted(range(len(structures)), key=lambda k: -len(structures[k][0]))
        cost = {}
        for k in order:
            na = len(structures[k][0])
            Xs = np.asarray(structures[k][1], float)
            n_pairs = int((((Xs[:, None] - Xs[None]) ** 2).sum(-1) < (6.0 + _FAST_SKIN) ** 2).sum()) - na
            if n_pairs >= 0.75 * na * (na - 1):
                n_pairs = na * (na - 1)
            na = max(na, n_pairs // 20)
            cost[k] = na

        def pack(ks, cap):
            out, cur, n = [], [], 0
            for k in ks:
                if cur and n + cost[k] > cap:
                    out.append(cur); cur, n = [], 0
                cur.append(k); n += cost[k]
            return out + ([cur] if cur else [])

        chunks = pack(order, cap)
        n_pass = 0
        pool = gpu.cp.get_default_memory_pool()
        while chunks:
            if n_pass and fixed_cap is None:
                now = self._cap_units()
                if ceiling is not None:
                    now = min(now, max(1, ceiling // 2))
                if sum(cost[k] for k in chunks[0]) > now or (now > 1.25 * cap and len(chunks) > 1):
                    cap = now
                    chunks = pack([k for c in chunks for k in c], cap)
            ch = chunks.pop(0)
            oom = False
            try:
                rs = None if restraints is None else [restraints[k] for k in ch]
                opts = dict(etol=etol, keep_state=keep_state, warm=None if warm is None else [warm[k] for k in ch])
                opts = {k: v for k, v in opts.items() if v}
                # peak of the pass: how far the default pool grew (cached blocks released first, so the growth of its
                # total is the pass's high-water mark; free space left inside split chunks is in both totals and
                # cancels) plus the private pools of its captured MACE graphs
                gpu.release_retired()                    # the last pass's graphs (never inside a capture)
                pool.free_all_blocks()
                base = pool.total_bytes()
                self._graph_bytes = 0
                for k, r in zip(ch, self._relax_chunk([structures[k] for k in ch], fmax, max_steps, rs, precon,
                                                                  **opts)):
                    recs[k] = r
                n_pass += 1
                self._record_pass(sum(cost[k] for k in ch), max(0, pool.total_bytes() - base) + self._graph_bytes)
            except (gpu.OOMError, RuntimeError) as exc:
                if not isinstance(exc, gpu.OOMError) and "out of memory" not in str(exc).lower():
                    raise
                oom = True
            if oom:
                import gc

                gc.collect()
                gpu.empty_cache()
                if len(ch) == 1:
                    s = structures[ch[0]]
                    recs[ch[0]] = RelaxRecord(np.asarray(s[1], float), float("nan"), float("nan"), float("nan"), 0,
                                              False, "error: out of memory")
                    continue
                u_fail = sum(cost[k] for k in ch)
                ceiling = u_fail if ceiling is None else min(ceiling, u_fail)
                if fixed_cap is None:
                    self._record_oom(u_fail)
                cap = max(1, min(cap, u_fail) // 2)
                chunks = pack(ch + [k for c in chunks for k in c], cap)
                log.info("MACE relax: out of memory, batch cap now %d atoms", cap)
        gpu.release_retired()
        self.max_batch_atoms = self.max_batch_atoms or None
        self.stats = dict(wall_s=time.perf_counter() - t0, n_structures=len(structures), n_passes=n_pass,
                          batch_atoms=cap, level=level, fmax=fmax, max_steps=max_steps)
        return recs

    @staticmethod
    def _restraint_ef(Xp, rest):
        """Flat-bottom restraint energy (S,) and forces (S, nmax, 3) for padded positions Xp;
        rest = (s, i, j, lo, hi, k, seg_e, seg_f) with the segment sums of energies per structure and forces per atom."""
        from jacobik.core import gpu

        cp = gpu.cp
        s, i, j, lo, hi, k, seg_e, seg_f = rest
        v = Xp[s, i] - Xp[s, j]
        d = cp.maximum(cp.linalg.norm(v, axis=-1), 1e-9)
        over = cp.maximum(d - hi, 0.0) - cp.maximum(lo - d, 0.0)
        g = (k * over / d)[:, None] * v
        E = seg_e(0.5 * k * over * over)
        F = seg_f(cp.concatenate([-g, g])).reshape(Xp.shape)
        return E, F

    @staticmethod
    def _warm_start(warm, structs, sizes, nmax, s_h, y_h, rho, valid, clean, trust):
        """L-BFGS memory and trust radius from earlier OptStates (newest pairs last), one host-to-device copy."""
        from jacobik.core import gpu

        cp = gpu.cp
        S, M, D = s_h.shape
        s0, y0 = np.zeros((S, M, D)), np.zeros((S, M, D))
        v0, t0 = np.zeros((S, M), bool), gpu.to_cpu(trust).copy()
        for k, w in enumerate(warm):
            if w is None or not w.fits(np.asarray(structs[k][1], float)) or len(w.s) == 0:
                continue
            m, na = min(len(w.s), M), sizes[k]
            s0[k, M - m:, :na * 3] = w.s[-m:].reshape(m, -1)
            y0[k, M - m:, :na * 3] = w.y[-m:].reshape(m, -1)
            v0[k, M - m:] = True
            t0[k] = w.trust
        if not v0.any():
            return s_h, y_h, rho, valid, clean, trust
        s_h, y_h = cp.asarray(s0), cp.asarray(y0)
        sy = (s_h * y_h).sum(axis=-1)
        valid = cp.asarray(v0) & (sy > 1e-12)
        rho = cp.where(valid, 1.0 / cp.maximum(sy, 1e-30), 0.0)
        return s_h, y_h, rho, valid, valid.copy(), cp.asarray(t0)

    def _relax_chunk(self, structs, fmax, max_steps, restraints=None, precon=None, etol=None, warm=None,
                     keep_state=False) -> List[RelaxRecord]:
        from jacobik.core import gpu, mace_triton

        cp = gpu.cp
        f64 = cp.float64
        S = len(structs)
        sizes = [len(s[0]) for s in structs]
        nmax = max(sizes)
        triton_cm = self._cm
        mask = cp.zeros((S, nmax), dtype=cp.bool_)
        for k, na in enumerate(sizes):
            mask[k, :na] = True
        Xp = cp.zeros((S, nmax, 3), dtype=f64)
        for k, s in enumerate(structs):
            Xp[k, :sizes[k]] = gpu.as_gpu(np.asarray(s[1], float), dtype=f64)
        rest = None
        if restraints is not None and any(r is not None and len(r) for r in restraints):
            rows = [np.column_stack([np.full(len(r), k), np.asarray(r, float)]) for k, r in enumerate(restraints)
                    if r is not None and len(r)]
            Rr = gpu.as_gpu(np.concatenate(rows), dtype=f64)
            rs_, ri_, rj_ = Rr[:, 0].astype(cp.int64), Rr[:, 1].astype(cp.int64), Rr[:, 2].astype(cp.int64)
            rest = (rs_, ri_, rj_, Rr[:, 3], Rr[:, 4], Rr[:, 5], gpu.SegmentSum(rs_, S),
                    gpu.SegmentSum(cp.concatenate([rs_ * nmax + ri_, rs_ * nmax + rj_]), S * nmax))

        def evaluate(sel):
            """E (S,), F (S, nmax, 3) for the structures in `sel` (bool S); others NaN."""
            idx = cp.nonzero(sel)[0].tolist()
            return evaluate_triton(idx)

        def evaluate_triton(idx):
            cap = cache.get("cap")
            n_idx = sum(sizes[k] for k in idx)
            if (cap is None or not set(idx) <= cap["set"]
                    or (n_idx <= _FAST_SHRINK * cap["atoms"] and cap["atoms"] > _FAST_KEEP_ATOMS)):
                cache.pop("cap", None)
                cap = cache["cap"] = dict(idx=list(idx), set=set(idx), atoms=n_idx,
                                          m=mask[idx], sel_cache={})
                cap["ge"] = new_ge(cap, Xp)
                self._n_capture = getattr(self, "_n_capture", 0) + 1
                if "Xbest" in state:
                    cache["rebase"] = (list(idx), evaluate_cap(cap, idx, state["Xbest"]))
            return evaluate_cap(cap, idx, Xp)

        def new_ge(cap, Xsrc):
            from jacobik.core.mace_triton import TritonBatchPlan, TritonGraphEvaluator

            ci = cap["idx"]
            Xf = Xsrc[ci][cap["m"]]
            r_list = self._rmax + _FAST_SKIN
            sub_structs = [structs[k] for k in ci]
            pi_count = sum(sizes[k] * (sizes[k] - 1) for k in ci)
            plan, out_pairs = TritonBatchPlan.from_structures(triton_cm, sub_structs, X=Xf, r_list=r_list)
            if pi_count > 0 and plan.E >= 0.75 * pi_count and out_pairs is not None and out_pairs[0].size > 0:
                plan, out_pairs = TritonBatchPlan.from_structures(triton_cm, sub_structs)
            return TritonGraphEvaluator(
                triton_cm, plan, graph=self.graph, r_cut=self._rmax, out_pairs=out_pairs,
            )

        def evaluate_cap(cap, idx, Xsrc):
            E = cp.full((S,), float("nan"), dtype=f64)
            F = cp.zeros((S, nmax, 3), dtype=f64)
            ci = cap["idx"]
            m = cap["m"]
            Xf = Xsrc[ci][m]
            if cap["ge"].stale(Xf):
                cap.pop("ge")
                cap["ge"] = new_ge(cap, Xsrc)
                self._n_capture = getattr(self, "_n_capture", 0) + 1
            Ec, _, f = cap["ge"](Xf)
            self._fast_mode = cap["ge"].mode
            Fc = cp.zeros((len(ci), nmax, 3), dtype=f64)
            Fc[m] = f
            idx_key = tuple(idx)
            sc = cap["sel_cache"].get(idx_key)
            if sc is None:
                sidx = set(idx)
                sel_t = cp.asarray([k in sidx for k in ci], dtype=cp.bool_)
                cit = cp.asarray(ci, dtype=cp.int64)
                sc = cap["sel_cache"][idx_key] = (sel_t, cit[sel_t])
            sel_t, cit_sel = sc
            E[cit_sel] = Ec[sel_t]
            F[cit_sel] = Fc[sel_t]
            Ei = E.copy()
            if rest is not None:
                Er, Fr = self._restraint_ef(Xsrc, rest)
                Ei[idx] = Ei[idx] + Er[idx]
                F[idx] = F[idx] + Fr[idx]
            return E, Ei, F

        cache, state = {}, {}
        D = nmax * 3
        s_h = cp.zeros((S, _MEMORY, D), dtype=f64)
        y_h = cp.zeros_like(s_h)
        rho = cp.zeros((S, _MEMORY), dtype=f64)
        valid = cp.zeros((S, _MEMORY), dtype=cp.bool_)
        clean = cp.zeros((S, _MEMORY), dtype=cp.bool_)       # pair formed with no restraint active (OptState)
        trust = cp.full((S,), _TRUST0, dtype=f64)
        if warm:
            s_h, y_h, rho, valid, clean, trust = self._warm_start(warm, structs, sizes, nmax, s_h, y_h, rho, valid, clean,
                                                                  trust)
        etol_ev = None if etol is None else float(etol) / EV_TO_KJ_MOL
        e_ring = cp.full((S, ETOL_WINDOW), float("inf"), dtype=f64) if etol_ev is not None else None
        active = cp.ones(S, dtype=cp.bool_)
        n_eval = cp.zeros(S, dtype=cp.int64)
        reason_code = cp.zeros(S, dtype=cp.int64)
        reason = [""] * S
        ff = (precon or (_FAST_PRECON if self.fast else PRECON)) == "ff" and _USE_PRECON
        trust_min = _FAST_TRUST_MIN if self.fast else _TRUST_MIN
        if ff:
            terms = []
            for s_ in structs:
                try:
                    terms.append(self._ff_terms(list(s_[0]), np.asarray(s_[1], float),
                                                _FAST_MD_RHO if self.fast else None))
                except Exception:
                    terms.append((None, None, None, None, None, None))
            pack = self._ff_pack(terms, self._device)
            Pinv = self._ff_precon_inv(Xp, mask, pack)
            mu0 = 1.0
        else:
            Pinv = self._precon_inv(Xp, mask)
            mu0 = _MU0

        def papply(P_, v):
            if ff:
                return cp.matmul(P_, v[:, :, None])[:, :, 0]
            return cp.einsum("bij,bjc->bic", P_, v.reshape(S, nmax, 3)).reshape(S, D)

        X_built = Xp.copy()

        E, Ei, F = evaluate(active)
        n_eval += active.astype(cp.int64)
        E0 = E.copy()
        Xbest, Ebest, Eibest, Fbest = Xp.copy(), E.copy(), Ei.copy(), F.copy()
        state["Xbest"] = Xbest
        if getattr(self, "_snap", None):
            snap_E = {kk: cp.zeros_like(Ebest) for kk in self._snap}
            snap_X = {kk: cp.zeros_like(Xbest) for kk in self._snap}
            snap_done = {kk: cp.zeros(S, dtype=cp.bool_) for kk in self._snap}
        rel_tol = 0.0

        def fnorm(F_):
            return cp.linalg.norm(F_, axis=-1).max(axis=-1)

        n_reject_t = cp.zeros((), dtype=cp.int64)
        n_steps_t = cp.zeros((), dtype=cp.int64)
        while True:
            fm = fnorm(Fbest)
            done = active & (fm <= fmax)
            if e_ring is not None:          # energy stop: fell by less than etol over the last ETOL_WINDOW evaluations
                edone = active & ~done & ((e_ring[:, 0] - Eibest) < etol_ev)
                reason_code = cp.where(edone, 3, reason_code)
                done = done | edone
            budget = active & ~done & (n_eval >= max_steps)
            reason_code = cp.where(done & (reason_code != 3), 1, cp.where(budget, 2, reason_code))
            active = active & ~done & ~budget
            if not bool(active.any()):
                break
            g = -Fbest.reshape(S, D)
            q = g.copy()
            alpha = cp.zeros((S, _MEMORY), dtype=f64)
            for i in range(_MEMORY - 1, -1, -1):
                a_i = rho[:, i] * (s_h[:, i] * q).sum(axis=-1) * valid[:, i]
                alpha[:, i] = a_i
                q = q - a_i[:, None] * y_h[:, i]
            r = papply(Pinv, q)
            last = valid[:, -1]
            yPy = (y_h[:, -1] * papply(Pinv, y_h[:, -1])).sum(axis=-1)
            sy = (s_h[:, -1] * y_h[:, -1]).sum(axis=-1)
            gamma = cp.where(last & (yPy > 0), sy / cp.maximum(yPy, 1e-30), 1.0 / mu0)
            r = gamma[:, None] * r
            for i in range(_MEMORY):
                b_i = rho[:, i] * (y_h[:, i] * r).sum(axis=-1) * valid[:, i]
                r = r + s_h[:, i] * ((alpha[:, i] - b_i) * valid[:, i])[:, None]
            step = -r.reshape(S, nmax, 3) * mask[..., None]
            bad = (step.reshape(S, D) * g).sum(axis=-1) >= 0
            sd = papply(Pinv, Fbest.reshape(S, D)).reshape(S, nmax, 3) / mu0
            step = cp.where(bad[:, None, None], sd, step)
            valid = valid & ~bad[:, None]
            longest = cp.linalg.norm(step, axis=-1).max(axis=-1)
            step = step * cp.minimum(trust / cp.maximum(longest, 1e-12), 1.0)[:, None, None]
            step = step * active[:, None, None]
            Xp = Xbest + step
            try:
                E, Ei, F = evaluate(active)
            except gpu.OOMError:
                raise
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    raise
                for k in cp.nonzero(active)[0].tolist():
                    reason[k] = f"error: {type(exc).__name__}: {exc}"[:200]
                break
            n_eval += active.astype(cp.int64)
            rbase = cache.pop("rebase", None)
            if rbase is not None:
                ridx = cp.asarray(rbase[0], dtype=cp.int64)
                Eb, Eib, Fb = rbase[1]
                Ebest[ridx], Eibest[ridx], Fbest[ridx] = Eb[ridx], Eib[ridx], Fb[ridx]
                n_eval[ridx] += 1
            finite = cp.isfinite(E) & cp.isfinite(F.reshape(S, -1)).all(axis=-1)
            accept = active & finite & (Ei <= Eibest + _REJECT_TOL_EV + rel_tol * cp.abs(Eibest))
            reject = active & ~accept
            s_new = (Xp - Xbest).reshape(S, D)
            y_new = (Fbest - F).reshape(S, D)
            sy_new = (s_new * y_new).sum(axis=-1)
            push = accept & (sy_new > 1e-12)
            p_m = push[:, None, None]
            s_h_next = cp.roll(s_h, -1, axis=1)
            y_h_next = cp.roll(y_h, -1, axis=1)
            rho_next = cp.roll(rho, -1, axis=1)
            valid_next = cp.roll(valid, -1, axis=1)
            s_h_next[:, -1] = s_new
            y_h_next[:, -1] = y_new
            rho_next[:, -1] = cp.where(push, 1.0 / cp.maximum(sy_new, 1e-30), 0.0)
            valid_next[:, -1] = True
            clean_next = cp.roll(clean, -1, axis=1)
            clean_next[:, -1] = ~((Ei - E) > 0) & ~((Eibest - Ebest) > 0)    # restraint energy 0 at both ends
            clean = cp.where(push[:, None], clean_next, clean)
            s_h = cp.where(p_m, s_h_next, s_h)
            y_h = cp.where(p_m, y_h_next, y_h)
            rho = cp.where(push[:, None], rho_next, rho)
            valid = cp.where(push[:, None], valid_next, valid)
            dE_tr = Ei - Eibest
            Xbest = cp.where(accept[:, None, None], Xp, Xbest)
            state["Xbest"] = Xbest
            Ebest = cp.where(accept, E, Ebest)
            Eibest = cp.where(accept, Ei, Eibest)
            Fbest = cp.where(accept[:, None, None], F, Fbest)
            if e_ring is not None:
                e_ring = cp.concatenate([e_ring[:, 1:], cp.where(active, Eibest, e_ring[:, -1])[:, None]], axis=1)
            trust = cp.where(accept, cp.minimum(trust * 1.25, _TRUST_MAX), cp.maximum(trust * 0.5, trust_min))
            if getattr(self, "_trace", None) is not None:
                self._trace.append(dict(fm=gpu.to_cpu(fnorm(F)), accept=gpu.to_cpu(accept),
                                        active=gpu.to_cpu(active), trust=gpu.to_cpu(trust),
                                        dE=gpu.to_cpu(dE_tr)))
            if getattr(self, "_snap", None):
                for kk in self._snap:
                    hit = (n_eval >= kk) & ~snap_done[kk]
                    if bool(hit.any()):
                        hi = cp.nonzero(hit)[0]
                        snap_E[kk][hi], snap_X[kk][hi] = Ebest[hi], Xbest[hi]
                        snap_done[kk][hi] = True
            valid = valid & ~reject[:, None]
            n_reject_t = n_reject_t + reject.sum()
            n_steps_t = n_steps_t + active.sum()
            Xp = Xbest
            moved = cp.linalg.norm(Xbest - X_built, axis=-1).max(axis=-1)
            if ff:
                rb = (moved > _PRECON_REBUILD_A) & active
                if bool(rb.any()):
                    ri = cp.nonzero(rb)[0]
                    Pinv[ri] = self._ff_precon_inv(Xbest, mask, pack, ri)
                    X_built[ri] = Xbest[ri]
            elif bool((moved > _PRECON_REBUILD_A).any()):
                Pinv = self._precon_inv(Xbest, mask)
                X_built = Xbest.copy()
        self._n_reject = getattr(self, "_n_reject", 0) + int(n_reject_t.item())
        self._n_steps = getattr(self, "_n_steps", 0) + int(n_steps_t.item())
        ge = (cache.get("cap") or {}).get("ge")
        self._graph_bytes = int(getattr(ge, "held_bytes", lambda: 0)()) if ge is not None else 0      # for the pass sizing in _relax
        fm = fnorm(Fbest)
        out = []
        Xc, Ec, Fc, E0c, fmc, nec, rcc = (gpu.to_cpu(Xbest), gpu.to_cpu(Ebest), gpu.to_cpu(Fbest), gpu.to_cpu(E0),
                                          gpu.to_cpu(fm), gpu.to_cpu(n_eval), gpu.to_cpu(reason_code))
        if keep_state:
            keep_c, sh_c, yh_c, tr_c = (gpu.to_cpu(valid & clean), gpu.to_cpu(s_h), gpu.to_cpu(y_h), gpu.to_cpu(trust))
        for k in range(S):
            na = sizes[k]
            rs = reason[k] or {1: "fmax", 3: "energy"}.get(int(rcc[k]), "max_steps")
            out.append(RelaxRecord(positions=Xc[k, :na].copy(), energy_ev=float(Ec[k]), energy0_ev=float(E0c[k]),
                                   fmax=float(fmc[k]), n_eval=int(nec[k]),
                                   converged=bool(fmc[k] <= fmax) or (rs == "energy"),
                                   reason=rs, forces=Fc[k, :na].copy()))
            if keep_state and not reason[k]:
                sel = np.flatnonzero(keep_c[k])
                out[-1].state = OptState(Xc[k, :na].copy(), sh_c[k, sel, :na * 3].reshape(-1, na, 3).copy(),
                                         yh_c[k, sel, :na * 3].reshape(-1, na, 3).copy(), float(tr_c[k]))
            if getattr(self, "_snap", None):
                out[-1].snap = {kk: (float(snap_E[kk][k]) if bool(snap_done[kk][k]) else float(Ec[k]),
                                     gpu.to_cpu(snap_X[kk][k, :na]).copy() if bool(snap_done[kk][k]) else Xc[k, :na].copy())
                                for kk in self._snap}
        return out
