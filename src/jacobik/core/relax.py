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
evaluations through core.mace_fast (full-pair edges, sync-free long-range block, CUDA-graph replay of one captured
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

import contextlib
import logging
import math
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
#: fast evaluation (core.mace_fast): full-pair edges, sync-free long-range block, CUDA-graph replay; off by default.
#: The captured batch is rebuilt when the active rows hold at most this share of its atoms.
#: JACOBIK_RELAX_FAST: "1" fast path everywhere, "0" never; unset (v1.0b default): fast on CUDA, default path on CPU
FAST = {"1": True, "0": False}.get(os.environ.get("JACOBIK_RELAX_FAST", ""), None)


def _fast_default(device) -> bool:
    if FAST is not None:
        return FAST
    d = str(device)
    if d == "auto":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:                                 # noqa: BLE001
            return False
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
    """Atoms per MACE pass: ~4 MB per atom for energy + forces in float32 (MACE-POLAR-1-S, e3nn: 16.6 GB peak at 4113
    atoms on an A100, 2026-11-18), x2.5 in float64; 60 % of the memory free to this process, counting blocks PyTorch
    has cached but not allocated (mem_get_info alone omits them: a second relax() call fell to 1722 atoms). CPU: 256 (one
    pass per complex is the fastest on CPU and ~3.8 MB per atom: 2000 atoms held ~7.6 GB; QEZVEY 99 atoms, 1 thread: 0.98
    s per row-evaluation and 1.4 GB alone, 1.25 s and 3.3 GB six per pass, 2026-11-20)."""
    import torch

    if device.type != "cuda":
        return 256
    free = _allocatable(device)
    per_atom = 4e6 * (2.5 if dtype == "float64" else 1.0)
    return int(max(200, min(20000, 0.6 * free / per_atom)))


#: deterministic CUDA kernels inside relax() (JACOBIK_RELAX_DETERMINISTIC=0 turns it off). The MACE scatters otherwise
#: accumulate with atomics: float32 settle rows moved ~1e-3 A from run to run and reordered the ensemble (same seed,
#: two processes, 2026-11-21); the cost is ~8 % of a settle pass.
DETERMINISTIC = os.environ.get("JACOBIK_RELAX_DETERMINISTIC", "1") != "0"


@contextlib.contextmanager
def _deterministic(on: bool):
    """torch deterministic algorithms for the block, the caller's setting restored afterwards."""
    if not on:
        yield
        return
    import torch

    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


#: measured GPU memory of relax passes per (GPU, dtype, cueq, graph): sorted (units, peak bytes) points of this
#: process; batches are sized from them instead of a constant (MACERelaxer._cap_units)
_UNIT_BYTES: Dict[tuple, list] = {}


def _allocatable(dev) -> int:
    """Bytes this process can still allocate on `dev`: the device's free memory plus this process's cached blocks,
    bounded by the allocator's per-process cap (set_per_process_memory_fraction; MACEPolarEvaluator sets 0.85, the bench
    0.80). Sizing from free device memory alone overshot that cap: the allocator then freed its cache and retried
    (QEZVEY settle on the 2060: 23 retries with no other process on the GPU)."""
    import torch

    dev = torch.device(dev)
    if dev.index is None:                  # get_per_process_memory_fraction needs an index ("cuda" -> current device)
        dev = torch.device("cuda", torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(dev)
    alloc, cached = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
    frac = torch.cuda.get_per_process_memory_fraction(dev) if hasattr(torch.cuda, "get_per_process_memory_fraction") else 1.0
    return int(max(0, min(free + cached - alloc, frac * total - alloc)))


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

    #: instance values come from the constructor; these defaults describe an unconfigured relaxer (library evaluation)
    fast = False
    pairs = False
    graph = False
    cueq = False

    def __init__(self, model: str = "polar-1-s", device: str = "auto", dtype: str = "float32",
                 model_path: Optional[str] = None, max_batch_atoms: Optional[int] = None,
                 cueq: Optional[bool] = None, fast: Optional[bool] = None, pairs: Optional[bool] = None,
                 graph: Optional[bool] = None):
        """cueq: cuEquivariance kernels (fused convolution; CUDA only, needs cuequivariance-torch). Default (None): on
        when installed, JACOBIK_CUEQ=0/1 forces it. Same energies to float32 round-off; QEZVEY settle rows (99 atoms,
        2026-11-20): RTX 2060 12 rows 5.55 -> 2.78 s, 1.9 -> 0.71 GB per pass; A100 192 rows 13.7 -> 5.1 GB per pass.
        (The 2026-11-18 "slower" measurement ran the fast path's one-capture-per-small-pass orchestration.)"""
        from jacobik.core.energy import MACEPolarEvaluator

        env = {"1": True, "0": False}.get(os.environ.get("JACOBIK_CUEQ", ""), None)
        if cueq is None:
            import importlib.util

            cueq = env if env is not None else importlib.util.find_spec("cuequivariance_torch") is not None

        if model_path is None:
            from jacobik.core.env import MACE_MODEL_PATH

            model_path = os.environ.get("JACOBIK_MACE_MODEL_PATH") or (str(MACE_MODEL_PATH) if MACE_MODEL_PATH else None)
        self._ev = MACEPolarEvaluator(model=model, device=device, dtype=dtype, model_path=model_path,
                                      enable_cueq=bool(cueq))
        self.dtype = dtype
        self.cueq = bool(self._ev.enable_cueq)
        #: fast: the fast path's optimizer (ff preconditioner with metal-donor terms, 1e-3 A trust floor); default on CUDA
        self.fast = _fast_default(device) if fast is None else bool(fast)
        #: pairs: evaluate through core.mace_fast (pair-list edges, sync-free long-range block with dense charge sums,
        #: float64 energy sums), independent of the optimizer; default on CUDA. Off: the library's own evaluation
        #: (the CPU path; per-molecule host syncs in the long-range block make it latency-bound on a GPU).
        on_cuda = str(self._ev.device).startswith("cuda")
        self.pairs = on_cuda if pairs is None else (bool(pairs) and on_cuda)
        #: graph: capture each pass as a CUDA graph and replay it (pairs only; default on). Measured on 80 settle rows
        #: (SEJKOK, 10 steps, RTX 2060): library 14.0 s, pairs 9.4 s, pairs + graph 8.5 s.
        self.graph = self.pairs if graph is None else (bool(graph) and self.pairs)
        self.max_batch_atoms = max_batch_atoms
        self.stats = {}

    def _mem_key(self):
        import torch

        return (torch.cuda.get_device_name(self._device), self.dtype, self.cueq, self.pairs, self.graph)

    def _record_pass(self, units: int, peak_bytes: int) -> None:
        """Keep the passes' (units, peak bytes); the largest two of clearly different size give fixed + per-unit."""
        if units > 0 and peak_bytes > 0:
            pts = _UNIT_BYTES.setdefault(self._mem_key(), [])
            pts.append((int(units), int(peak_bytes)))
            pts.sort()
            del pts[:-8]

    def _record_oom(self, units: int) -> None:
        """A pass of `units` did not fit: recorded as needing all the memory free now (an upper point of the fit)."""
        import torch

        dev = torch.device(self._device)
        self._record_pass(units, int(_allocatable(dev) + torch.cuda.memory_allocated(dev)))

    def _cap_units(self) -> int:
        """Cost units per pass from the memory measured on this GPU / model / dtype / backend: peak = a + b * units
        fitted on two passes of clearly different size (a single pass counts only if it is at least half the constant
        cap: a 1-row warm-up is mostly fixed cost and would freeze the passes small); 70 % of the memory free to this
        process (at 85 % the allocator retried 18 times on 96 QEZVEY rows). The conservative constant until then."""
        import torch

        dev = torch.device(self._device)
        const = _default_batch_atoms(dev, self.dtype)
        if dev.type != "cuda":
            return const
        pts = _UNIT_BYTES.get(self._mem_key()) or []
        budget = 0.7 * _allocatable(dev)
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
        """One step on a water molecule after loading: the first evaluation of a process takes another numerical path
        (TorchScript profiling run), so without it the first relax() of a process differed from every later one by
        ~1e-5 A and the same seed gave different rows in one process (2026-11-20). Never raises."""
        stats = self.stats
        try:
            self.relax([(["O", "H", "H"], np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]), 0, 1)],
                       level="loose", max_steps=1)
        except Exception as exc:                          # noqa: BLE001
            log.debug("MACE warm-up failed (%s)", exc)
        self.stats = stats

    def _setup(self):
        import torch
        from mace import data as mace_data

        calc = self._ev._get_calculator()
        calc.arrays_keys.update({calc.charges_key: "charges"})
        self._calc = calc
        self._keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
        self._model = calc.models[0]
        self._mdt = next(self._model.parameters()).dtype
        self._device = torch.device(calc.device) if not isinstance(calc.device, torch.device) else calc.device
        self._rmax = float(calc.r_max)
        logging.getLogger(__name__).info("MACE relaxer: model %s on %s, cueq %s, fast optimizer %s, pairs %s, graph %s",
                                         self._mdt, self._device, self.cueq, self.fast, self.pairs, self.graph)

    def _template(self, symbols, X, q, mult):
        from ase import Atoms
        from mace import data as mace_data
        from mace.tools import torch_tools

        at = Atoms(symbols=list(symbols), positions=np.asarray(X, float), pbc=False)
        at.info.update(charge=int(q), spin=int(mult), external_field=[0.0, 0.0, 0.0])
        with torch_tools.default_dtype(self._calc.default_dtype):
            cfg = mace_data.config_from_atoms(at, key_specification=self._keyspec, head_name=self._calc.head)
            return mace_data.AtomicData.from_config(cfg, z_table=self._calc.z_table, cutoff=self._rmax,
                                                    heads=self._calc.available_heads)

    def _collate(self, templates):
        import torch
        from mace.tools import torch_geometric

        b = next(iter(torch_geometric.dataloader.DataLoader(dataset=templates, batch_size=len(templates), shuffle=False)))
        b = b.to(self._device)
        d = b.to_dict()
        for k, v in list(d.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                d[k] = v.to(self._mdt)
        # all intra-structure atom pairs (i != j), filtered by distance every step
        ptr = d["ptr"].tolist()
        ii, jj = [], []
        for g in range(len(ptr) - 1):
            a = torch.arange(ptr[g], ptr[g + 1], device=self._device)
            I, J = torch.meshgrid(a, a, indexing="ij")
            m = I != J
            ii.append(I[m]); jj.append(J[m])
        d["_pairs"] = (torch.cat(ii), torch.cat(jj))
        return d

    @staticmethod
    def _subset(d, idx):
        """The collated batch ``d`` restricted to structures ``idx`` (sorted list), by indexing its tensors on the
        device: per-atom tensors by atom, per-structure ones by structure, ``batch``/``ptr``/``_pairs`` renumbered.
        Re-collating through the MACE DataLoader cost about as much as an energy evaluation each time the active set
        shrank (189 ms for 16 structures on the 2060)."""
        import torch

        ptr, batch = d["ptr"], d["batch"]
        G, N = ptr.numel() - 1, batch.numel()
        dev = batch.device
        gi = torch.as_tensor(idx, dtype=torch.long, device=dev)
        gnew = torch.full((G,), -1, dtype=torch.long, device=dev)
        gnew[gi] = torch.arange(len(idx), device=dev)
        nmask = gnew[batch] >= 0
        nidx = nmask.nonzero()[:, 0]
        nnew = torch.full((N,), -1, dtype=torch.long, device=dev)
        nnew[nidx] = torch.arange(nidx.numel(), device=dev)
        out = {}
        for k, v in d.items():
            if k == "ptr":
                counts = ptr[1:] - ptr[:-1]
                out[k] = torch.cat([ptr.new_zeros(1), counts[gi].cumsum(0)])
            elif k == "batch":
                out[k] = gnew[batch[nidx]]
            elif k == "_pairs":
                pi, pj = v
                keep = nmask[pi]
                out[k] = (nnew[pi[keep]], nnew[pj[keep]])
            elif torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N and k not in ("edge_index",):
                out[k] = v[nidx]
            elif torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == G:
                out[k] = v[gi]
            elif k in ("cell", "rcell") and torch.is_tensor(v) and v.shape[0] == 3 * G:
                out[k] = v.reshape(G, 3, *v.shape[1:])[gi].reshape(-1, *v.shape[1:])
            else:
                out[k] = v
        return out

    def _ef(self, d, X):
        """Energies (G,) eV and forces (N, 3) eV/A of the collated batch ``d`` at flat positions X (N, 3)."""
        import torch

        pi, pj = d["_pairs"]
        Xm = X.to(self._mdt)
        with torch.no_grad():
            keep = ((Xm[pi] - Xm[pj]) ** 2).sum(-1) < self._rmax ** 2
            ei = torch.stack([pi[keep], pj[keep]])
        inp = {k: v for k, v in d.items() if not k.startswith("_")}
        inp["positions"] = Xm.detach().clone()
        inp["edge_index"] = ei
        z = torch.zeros(ei.shape[1], 3, dtype=self._mdt, device=self._device)
        inp["shifts"], inp["unit_shifts"] = z, z
        out = self._model(inp, compute_force=True, compute_virials=False, compute_stress=False, training=False)
        u = self._calc.energy_units_to_eV
        e = out["energy"].detach().double() * u
        # (interaction_energy is not the energy the forces differentiate: it leaves out the long-range terms)
        return e, e, out["forces"].detach().double() * u

    # ------------------------------------------------------------------ preconditioner
    @staticmethod
    def _precon_inv(Xp, mask):
        """Inverse of the Exp preconditioner (without mu) for padded (S, n, 3) positions; (S, n, n)."""
        import torch

        S, n, _ = Xp.shape
        if not _USE_PRECON:
            return torch.eye(n, dtype=Xp.dtype, device=Xp.device).expand(S, n, n).clone()
        D = torch.cdist(Xp, Xp)
        big = torch.finfo(D.dtype).max
        pm = mask[:, :, None] & mask[:, None, :]
        Dn = torch.where(pm & ~torch.eye(n, dtype=torch.bool, device=Xp.device), D, torch.full_like(D, big))
        rnn = Dn.min(-1).values.clamp(max=3.0)
        rnn = torch.where(mask, rnn, torch.zeros_like(rnn))
        r_nn = (rnn.sum(-1) / mask.sum(-1).clamp(min=1))[:, None, None]          # per structure mean NN distance
        w = torch.exp(-_PRECON_A * (D / r_nn - 1.0)) * ((D < _PRECON_RCUT * r_nn) & pm)
        w = w * (1.0 - torch.eye(n, dtype=D.dtype, device=D.device))
        L = torch.diag_embed(w.sum(-1)) - w + _PRECON_C * torch.eye(n, dtype=D.dtype, device=D.device)
        # padded atoms: identity rows (decoupled)
        L = torch.where(pm, L, torch.diag_embed((~mask).to(D.dtype)))
        return torch.linalg.inv(L)

    # ------------------------------------------------------------------ model-Hessian preconditioner ("ff")
    @staticmethod
    def _row(sym):
        z = {"H": 1, "He": 1}.get(sym)
        if z is not None:
            return 0
        return 1 if sym in ("Li", "Be", "B", "C", "N", "O", "F", "Ne") else 2

    @staticmethod
    def _rcov(symbols):
        from rdkit.Chem import GetPeriodicTable

        pt = GetPeriodicTable()
        return np.array([pt.GetRcovalent(pt.GetAtomicNumber(s)) for s in symbols])

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
        ang = [(a, b, c) for b in range(len(symbols)) for x, a in enumerate(nb[b]) for c in nb[b][x + 1:]]
        tor = []
        for b, c in zip(i, j):
            for a in nb[b]:
                if a == c:
                    continue
                for d in nb[c]:
                    if d == b or d == a:
                        continue
                    tor.append((a, b, c, d))
        return (np.stack([i, j], 1), np.asarray(ang, dtype=np.int64).reshape(-1, 3),
                np.asarray(tor, dtype=np.int64).reshape(-1, 4), rows, rc, (metal, md_rho) if md_rho > 0 else None)

    @staticmethod
    def _ff_pack(terms, dev):
        """The per-structure terms of a chunk as flat device tensors (structure index first in every term)."""
        import torch

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
        cat = lambda L, w: torch.as_tensor(np.concatenate(L) if L else np.zeros((0, w), np.int64), dtype=torch.long,
                                           device=dev)
        use_rc = any(t[4] is not None for t in terms if t[0] is not None)
        use_mt = any(t[5] is not None for t in terms if t[0] is not None)
        return dict(pairs=cat(P, 3), ang=cat(A, 4), tor=cat(T, 5),
                    rows=torch.as_tensor(np.stack(rows), device=dev),
                    rc=torch.as_tensor(np.stack(rcs), device=dev) if use_rc else None,
                    metal=torch.as_tensor(np.stack(mts), device=dev) if use_mt else None, md_rho=md_rho)

    @staticmethod
    def _ff_hessian(Xp, mask, pack, sel=None):
        """(R, 3n, 3n) model Hessian (eV/A^2) of the structures sel (all when None) at padded positions Xp, all terms
        of all structures in one pass (a Python loop over structures cost 0.1-0.2 s per rebuild of 184 rows)."""
        import torch

        S, n, _ = Xp.shape
        dev, dt = Xp.device, Xp.dtype
        sel = torch.arange(S, device=dev) if sel is None else sel
        R = sel.numel()
        loc = torch.full((S,), -1, dtype=torch.long, device=dev)
        loc[sel] = torch.arange(R, device=dev)
        alpha = torch.as_tensor(_LINDH_ALPHA, dtype=dt, device=dev)
        rref2 = torch.as_tensor(_LINDH_RREF ** 2, dtype=dt, device=dev)
        rw, rc, mt = pack["rows"], pack["rc"], pack["metal"]

        def rho(s, a, b):
            d2 = ((Xp[s, a] - Xp[s, b]) / _BOHR).pow(2).sum(-1)
            r2 = rref2[rw[s, a], rw[s, b]] if rc is None else (rc[s, a].to(dt) + rc[s, b].to(dt)) ** 2
            r = torch.exp(alpha[rw[s, a], rw[s, b]] * (r2 - d2))
            if mt is not None:
                r = torch.where(mt[s, a] | mt[s, b], torch.full_like(r, pack["md_rho"]), r)
            return r

        H = torch.zeros(R, 3 * n, 3 * n, dtype=dt, device=dev)

        def add(s, atoms, Bv, k):
            T_, m = atoms.shape
            if T_ == 0:
                return
            idx = (3 * atoms[..., None] + torch.arange(3, device=dev)).reshape(T_, 3 * m)
            Bf = Bv.reshape(T_, 3 * m)
            outer = k[:, None, None] * Bf[:, :, None] * Bf[:, None, :]
            sl = loc[s][:, None, None].expand(T_, 3 * m, 3 * m)
            H.index_put_((sl, idx[:, :, None].expand(T_, 3 * m, 3 * m), idx[:, None, :].expand(T_, 3 * m, 3 * m)),
                         outer, accumulate=True)

        def keep(t):
            return t[loc[t[:, 0]] >= 0]

        p = keep(pack["pairs"])
        if len(p):
            s, a, b = p[:, 0], p[:, 1], p[:, 2]
            v = Xp[s, a] - Xp[s, b]
            u = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            add(s, torch.stack([a, b], 1), torch.stack([u, -u], 1), 0.45 * rho(s, a, b) * _EH / _BOHR ** 2)
        q = keep(pack["ang"])
        if len(q):
            s, a, b, c = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            u, w = Xp[s, a] - Xp[s, b], Xp[s, c] - Xp[s, b]
            lu, lw = u.norm(dim=-1, keepdim=True).clamp(min=1e-9), w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            eu, ew = u / lu, w / lw
            cos = (eu * ew).sum(-1, keepdim=True).clamp(-0.999999, 0.999999)
            sin = (1 - cos ** 2).sqrt()
            ba = (cos * eu - ew) / (lu * sin)
            bc = (cos * ew - eu) / (lw * sin)
            ok = (sin[:, 0] > 0.02).to(dt)                                   # near-linear: no bend term
            add(s, torch.stack([a, b, c], 1), torch.stack([ba, -ba - bc, bc], 1),
                0.15 * rho(s, a, b) * rho(s, b, c) * _EH * ok)
        t = keep(pack["tor"])
        if len(t):
            s, a, b, c, d = t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            f, g, h = Xp[s, a] - Xp[s, b], Xp[s, b] - Xp[s, c], Xp[s, d] - Xp[s, c]
            A_ = torch.cross(f, g, dim=-1)
            B_ = torch.cross(h, g, dim=-1)
            gl = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            A2 = A_.pow(2).sum(-1, keepdim=True).clamp(min=1e-9)
            B2 = B_.pow(2).sum(-1, keepdim=True).clamp(min=1e-9)
            fg = (f * g).sum(-1, keepdim=True)
            hg = (h * g).sum(-1, keepdim=True)
            ta = -gl / A2 * A_
            td = gl / B2 * B_
            tb = gl / A2 * A_ + fg / (A2 * gl) * A_ - hg / (B2 * gl) * B_
            tc = -gl / B2 * B_ - fg / (A2 * gl) * A_ + hg / (B2 * gl) * B_
            ok = ((A2[:, 0] > 1e-3) & (B2[:, 0] > 1e-3)).to(dt)
            add(s, torch.stack([a, b, c, d], 1), torch.stack([ta, tb, tc, td], 1),
                0.005 * rho(s, a, b) * rho(s, b, c) * rho(s, c, d) * _EH * ok)
        m3 = mask[sel].repeat_interleave(3, dim=1)
        eye = torch.eye(3 * n, dtype=dt, device=dev)
        H = H + _FF_C * eye
        H = torch.where(m3[:, :, None] & m3[:, None, :], H, eye.expand_as(H))      # padded atoms decoupled
        return H

    def _ff_precon_inv(self, Xp, mask, pack, sel=None):
        import torch

        H = self._ff_hessian(Xp, mask, pack, sel)
        L, info = torch.linalg.cholesky_ex(H)
        bad = info != 0
        if bool(bad.any()):
            ev, V = torch.linalg.eigh(H[bad])
            Hb = V @ torch.diag_embed(ev.clamp(min=_FF_C)) @ V.transpose(1, 2)
            L[bad] = torch.linalg.cholesky(Hb)
        return torch.cholesky_inverse(L)

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
        On CUDA the call runs with deterministic kernels (DETERMINISTIC): the same input gives the same rows."""
        if structures and not hasattr(self, "_model"):
            self._setup()
            self._warm_up()
        with _deterministic(DETERMINISTIC and str(getattr(self, "_device", "cpu")).startswith("cuda")):
            return self._relax(structures, level, fmax, max_steps, restraints, precon, etol, warm, keep_state)

    def _relax(self, structures, level, fmax, max_steps, restraints, precon, etol=None, warm=None,
               keep_state=False) -> List[RelaxRecord]:
        import torch

        if level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}")
        fmax = float(LEVELS[level]["fmax"] if fmax is None else fmax)
        max_steps = int(LEVELS[level]["max_steps"] if max_steps is None else max_steps)
        if not structures:
            return []
        t0 = time.perf_counter()
        fixed_cap = self.max_batch_atoms
        on_cuda = torch.device(self._device).type == "cuda"
        cap = fixed_cap or self._cap_units()
        ceiling = None                                     # cost units of the smallest pass that ran out of memory
        recs: List[Optional[RelaxRecord]] = [None] * len(structures)
        order = sorted(range(len(structures)), key=lambda k: -len(structures[k][0]))
        cost = {}
        for k in order:
            na = len(structures[k][0])
            if self.pairs:                      # edges: 0.10 MB each (A100 capture, 1000-6000 atoms), against 4 MB
                Xs = np.asarray(structures[k][1], float)          # per atom in the cutoff-edge estimate
                n_pairs = int((((Xs[:, None] - Xs[None]) ** 2).sum(-1) < (6.0 + _FAST_SKIN) ** 2).sum()) - na
                if n_pairs >= 0.75 * na * (na - 1):              # no neighbour list (new_ge): every pair is an edge
                    n_pairs = na * (na - 1)
                # 0.10 MB per edge measured (A100 and RTX 2060, 61-atom rows), doubled: the warm-up's blocks and the
                # captured graph's private pool are both held at capture time (2060: four OOM halvings at x1)
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
        while chunks:
            if n_pass and fixed_cap is None and on_cuda:
                # capacity re-read per chunk from free memory and the measured bytes per cost unit: other processes on
                # the GPU (the bench's parallel lanes) take and release memory between chunks, and the first pass is
                # packed from a conservative constant, so the pending chunks are repacked when the cap moves
                now = self._cap_units()
                if ceiling is not None:                    # never again at or above a size that failed in this call
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
                if on_cuda:
                    torch.cuda.reset_peak_memory_stats(self._device)
                    base = torch.cuda.memory_allocated(self._device)
                for k, r in zip(ch, self._relax_chunk([structures[k] for k in ch], fmax, max_steps, rs, precon,
                                                                  **opts)):
                    recs[k] = r
                n_pass += 1
                if on_cuda:
                    self._record_pass(sum(cost[k] for k in ch), torch.cuda.max_memory_allocated(self._device) - base)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc):
                    raise
                oom = True
            if oom:
                # outside the except clause: its traceback held the failed chunk's tensors and CUDA graphs, so
                # empty_cache() inside it released nothing and every split failed again (2026-11-19, 2060: 58 retries)
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                if len(ch) == 1:
                    s = structures[ch[0]]
                    recs[ch[0]] = RelaxRecord(np.asarray(s[1], float), float("nan"), float("nan"), float("nan"), 0,
                                              False, "error: out of memory")
                    continue
                # half the failed chunk, and every pending chunk repacked to it: chunks packed to the old cap would
                # each fail once more (2026-11-19, A100 with 3 lanes: 50 retries on one parent)
                u_fail = sum(cost[k] for k in ch)
                ceiling = u_fail if ceiling is None else min(ceiling, u_fail)
                if fixed_cap is None and on_cuda:
                    self._record_oom(u_fail)
                cap = max(1, min(cap, u_fail) // 2)
                chunks = pack(ch + [k for c in chunks for k in c], cap)
                log.info("MACE relax: out of memory, batch cap now %d atoms", cap)
        self.max_batch_atoms = self.max_batch_atoms or None
        self.stats = dict(wall_s=time.perf_counter() - t0, n_structures=len(structures), n_passes=n_pass,
                          batch_atoms=cap, level=level, fmax=fmax, max_steps=max_steps)
        return recs

    @staticmethod
    def _restraint_ef(Xp, rest):
        """Flat-bottom restraint energy (S,) and forces (S, nmax, 3) for padded positions Xp; rest = (s, i, j, lo, hi, k)
        tensors over all restraints of the chunk."""
        import torch

        s, i, j, lo, hi, k = rest
        v = Xp[s, i] - Xp[s, j]
        d = v.norm(dim=-1).clamp(min=1e-9)
        over = (d - hi).clamp(min=0.0) - (lo - d).clamp(min=0.0)            # signed violation
        e = 0.5 * k * over * over
        E = torch.zeros(Xp.shape[0], dtype=Xp.dtype, device=Xp.device).index_add_(0, s, e)
        g = (k * over / d)[:, None] * v                                       # dE/dx_i
        F = torch.zeros_like(Xp)
        F.index_put_((s, i), -g, accumulate=True)
        F.index_put_((s, j), g, accumulate=True)
        return E, F

    @staticmethod
    def _warm_start(warm, structs, sizes, s_h, y_h, rho, valid, clean, trust):
        """L-BFGS memory and trust radius from earlier OptStates (newest pairs last), one host-to-device copy."""
        import torch

        S, M, D = s_h.shape
        s0, y0 = np.zeros((S, M, D)), np.zeros((S, M, D))
        v0, t0 = np.zeros((S, M), bool), trust.cpu().numpy().copy()
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
        dev, f64 = s_h.device, s_h.dtype
        s_h, y_h = torch.as_tensor(s0, dtype=f64, device=dev), torch.as_tensor(y0, dtype=f64, device=dev)
        sy = (s_h * y_h).sum(-1)
        valid = torch.as_tensor(v0, device=dev) & (sy > 1e-12)
        rho = torch.where(valid, 1.0 / sy.clamp(min=1e-30), torch.zeros_like(sy))
        return s_h, y_h, rho, valid, valid.clone(), torch.as_tensor(t0, dtype=f64, device=dev)

    def _relax_chunk(self, structs, fmax, max_steps, restraints=None, precon=None, etol=None, warm=None,
                     keep_state=False) -> List[RelaxRecord]:
        import torch

        dev, f64 = self._device, torch.float64
        S = len(structs)
        sizes = [len(s[0]) for s in structs]
        nmax = max(sizes)
        templates = [self._template(*s) for s in structs]
        mask = torch.zeros(S, nmax, dtype=torch.bool, device=dev)
        for k, na in enumerate(sizes):
            mask[k, :na] = True
        Xp = torch.zeros(S, nmax, 3, dtype=f64, device=dev)
        for k, s in enumerate(structs):
            Xp[k, :sizes[k]] = torch.as_tensor(np.asarray(s[1], float), dtype=f64, device=dev)
        flat_idx = mask.reshape(-1).nonzero()[:, 0]                      # padded -> flat atom positions
        rest = None
        if restraints is not None and any(r is not None and len(r) for r in restraints):
            rows = [np.column_stack([np.full(len(r), k), np.asarray(r, float)]) for k, r in enumerate(restraints)
                    if r is not None and len(r)]
            Rr = torch.as_tensor(np.concatenate(rows), dtype=f64, device=dev)
            rest = (Rr[:, 0].long(), Rr[:, 1].long(), Rr[:, 2].long(), Rr[:, 3], Rr[:, 4], Rr[:, 5])

        def evaluate(sel):
            """E (S,), F (S, nmax, 3) for the structures in `sel` (bool S); others NaN."""
            idx = sel.nonzero()[:, 0].tolist()
            E = torch.full((S,), float("nan"), dtype=f64, device=dev)
            Ei = torch.full((S,), float("nan"), dtype=f64, device=dev)
            F = torch.zeros(S, nmax, 3, dtype=f64, device=dev)
            if self.pairs:
                return evaluate_fast(idx)
            key = tuple(idx)
            if key != cache.get("key"):
                if "full" not in cache:
                    cache["full"] = self._collate(templates)
                cache["key"] = key
                cache["d"] = cache["full"] if len(idx) == S else self._subset(cache["full"], idx)
            d = cache["d"]
            m = mask[idx]
            Xf = Xp[idx][m]
            e, ei, f = self._ef(d, Xf)
            E[idx] = e
            Ei[idx] = ei
            Fi = torch.zeros(len(idx), nmax, 3, dtype=f64, device=dev)
            Fi[m] = f
            F[idx] = Fi
            if rest is not None:
                Er, Fr = self._restraint_ef(Xp, rest)
                Ei[idx] = Ei[idx] + Er[idx]                               # E stays the MACE energy
                F[idx] = F[idx] + Fr[idx]
            return E, Ei, F

        def evaluate_fast(idx):
            """as evaluate, on a captured batch that holds every active row (core.mace_fast.GraphEvaluator)"""
            from jacobik.core.mace_fast import GraphEvaluator

            cap = cache.get("cap")
            n_idx = sum(sizes[k] for k in idx)
            if (cap is None or not set(idx) <= cap["set"]
                    or (n_idx <= _FAST_SHRINK * cap["atoms"] and cap["atoms"] > _FAST_KEEP_ATOMS)):
                if "full" not in cache:
                    cache["full"] = self._collate(templates)
                cache.pop("cap", None)
                d_ = cache["full"] if len(idx) == S else self._subset(cache["full"], idx)
                cap = cache["cap"] = dict(idx=list(idx), set=set(idx), atoms=n_idx, d=d_)
                cap["ge"] = new_ge(cap, Xp)
                self._n_capture = getattr(self, "_n_capture", 0) + 1
                if "Xbest" in state:
                    # A float32 total energy (~6e4 eV) shifts by a few ulps (4 meV each) with the batch membership,
                    # more than the step-rejection tolerance: re-evaluate the accepted geometries in the new batch.
                    cache["rebase"] = (list(idx), evaluate_cap(cap, idx, state["Xbest"]))
            return evaluate_cap(cap, idx, Xp)

        def new_ge(cap, Xsrc):
            from jacobik.core.mace_fast import GraphEvaluator

            ci = cap["idx"]
            Xf = Xsrc[ci][mask[ci]]
            pi, pj = cap["d"]["_pairs"]
            r_list = self._rmax + _FAST_SKIN
            n_list = int((((Xf[pi] - Xf[pj]) ** 2).sum(-1) < r_list ** 2).sum())
            if n_list < 0.75 * pi.numel():
                return GraphEvaluator(self._model, cap["d"], self._mdt, self._calc.energy_units_to_eV, graph=self.graph,
                                      X=Xf, r_list=r_list, r_cut=self._rmax)
            return GraphEvaluator(self._model, cap["d"], self._mdt, self._calc.energy_units_to_eV, graph=self.graph)

        def evaluate_cap(cap, idx, Xsrc):
            E = torch.full((S,), float("nan"), dtype=f64, device=dev)
            F = torch.zeros(S, nmax, 3, dtype=f64, device=dev)
            ci = cap["idx"]
            m = mask[ci]
            Xf = Xsrc[ci][m]
            if cap["ge"].stale(Xf):                                       # neighbour list expired: same batch
                cap.pop("ge")                                             # release the old graph before capturing
                cap["ge"] = new_ge(cap, Xsrc)
                self._n_capture = getattr(self, "_n_capture", 0) + 1
            Ec, _, f = cap["ge"](Xf)
            self._fast_mode = cap["ge"].mode
            Fc = torch.zeros(len(ci), nmax, 3, dtype=f64, device=dev)
            Fc[m] = f
            sidx = set(idx)
            sel_t = torch.as_tensor([k in sidx for k in ci], device=dev)
            cit = torch.as_tensor(ci, device=dev)
            E[cit[sel_t]] = Ec[sel_t]
            F[cit[sel_t]] = Fc[sel_t]
            Ei = E.clone()
            if rest is not None:
                Er, Fr = self._restraint_ef(Xsrc, rest)
                Ei[idx] = Ei[idx] + Er[idx]
                F[idx] = F[idx] + Fr[idx]
            return E, Ei, F

        cache, state = {}, {}
        D = nmax * 3
        s_h = torch.zeros(S, _MEMORY, D, dtype=f64, device=dev)
        y_h = torch.zeros_like(s_h)
        rho = torch.zeros(S, _MEMORY, dtype=f64, device=dev)
        valid = torch.zeros(S, _MEMORY, dtype=torch.bool, device=dev)
        clean = torch.zeros(S, _MEMORY, dtype=torch.bool, device=dev)    # pair formed with no restraint active (OptState)
        trust = torch.full((S,), _TRUST0, dtype=f64, device=dev)
        if warm:
            s_h, y_h, rho, valid, clean, trust = self._warm_start(warm, structs, sizes, s_h, y_h, rho, valid, clean, trust)
        etol_ev = None if etol is None else float(etol) / EV_TO_KJ_MOL
        e_ring = torch.full((S, ETOL_WINDOW), float("inf"), dtype=f64, device=dev) if etol_ev is not None else None
        active = torch.ones(S, dtype=torch.bool, device=dev)
        n_eval = torch.zeros(S, dtype=torch.long, device=dev)
        reason = [""] * S
        ff = (precon or (_FAST_PRECON if self.fast else PRECON)) == "ff" and _USE_PRECON
        trust_min = _FAST_TRUST_MIN if self.fast else _TRUST_MIN
        if ff:
            terms = []
            for s_ in structs:
                try:
                    terms.append(self._ff_terms(list(s_[0]), np.asarray(s_[1], float),
                                                _FAST_MD_RHO if self.fast else None))
                except Exception:                                         # no model Hessian: identity block
                    terms.append((None, None, None, None, None, None))
            pack = self._ff_pack(terms, dev)
            Pinv = self._ff_precon_inv(Xp, mask, pack)
            mu0 = 1.0
        else:
            Pinv = self._precon_inv(Xp, mask)
            mu0 = _MU0

        def papply(P_, v):
            """preconditioner inverse applied to flat (S, D) vectors"""
            if ff:
                return torch.bmm(P_, v[:, :, None])[:, :, 0]
            return torch.einsum("bij,bjc->bic", P_, v.reshape(S, nmax, 3)).reshape(S, D)

        X_built = Xp.clone()

        E, Ei, F = evaluate(active)
        n_eval += active.long()
        E0 = E.clone()
        Xbest, Ebest, Eibest, Fbest = Xp.clone(), E.clone(), Ei.clone(), F.clone()
        state["Xbest"] = Xbest
        if getattr(self, "_snap", None):
            snap_E = {kk: torch.zeros_like(Ebest) for kk in self._snap}
            snap_X = {kk: torch.zeros_like(Xbest) for kk in self._snap}
            snap_done = {kk: torch.zeros(S, dtype=torch.bool, device=dev) for kk in self._snap}
        rel_tol = 0.0 if self.pairs else _REJECT_TOL_REL.get(self.dtype, 3e-6)   # pairs: float64-summed energies

        def fnorm(F_):
            return F_.norm(dim=-1).amax(-1)

        g_prev = None
        while True:
            fm = fnorm(Fbest)
            done = active & (fm <= fmax)
            for k in done.nonzero()[:, 0].tolist():
                reason[k] = "fmax"
            if e_ring is not None:          # energy stop: fell by less than etol over the last ETOL_WINDOW evaluations
                edone = active & ~done & ((e_ring[:, 0] - Eibest) < etol_ev)
                for k in edone.nonzero()[:, 0].tolist():
                    reason[k] = "energy"
                done = done | edone
            budget = active & ~done & (n_eval >= max_steps)
            for k in budget.nonzero()[:, 0].tolist():
                reason[k] = "max_steps"
            active = active & ~done & ~budget
            if not bool(active.any()):
                break
            # L-BFGS direction from the accepted state (preconditioned two-loop recursion)
            g = -Fbest.reshape(S, D)
            q = g.clone()
            alpha = torch.zeros(S, _MEMORY, dtype=f64, device=dev)
            for i in range(_MEMORY - 1, -1, -1):
                a_i = rho[:, i] * (s_h[:, i] * q).sum(-1) * valid[:, i]
                alpha[:, i] = a_i
                q = q - a_i[:, None] * y_h[:, i]
            r = papply(Pinv, q)
            last = valid[:, -1]
            yPy = (y_h[:, -1] * papply(Pinv, y_h[:, -1])).sum(-1)
            sy = (s_h[:, -1] * y_h[:, -1]).sum(-1)
            gamma = torch.where(last & (yPy > 0), sy / yPy.clamp(min=1e-30), torch.full_like(sy, 1.0 / mu0))
            r = gamma[:, None] * r
            for i in range(_MEMORY):
                b_i = rho[:, i] * (y_h[:, i] * r).sum(-1) * valid[:, i]
                r = r + s_h[:, i] * ((alpha[:, i] - b_i) * valid[:, i])[:, None]
            step = -r.reshape(S, nmax, 3) * mask[..., None]
            # not a descent direction (memory went bad): steepest preconditioned descent
            bad = (step.reshape(S, D) * g).sum(-1) >= 0
            if bool(bad.any()):
                sd = papply(Pinv, Fbest.reshape(S, D)).reshape(S, nmax, 3) / mu0
                step = torch.where(bad[:, None, None], sd, step)
                valid[bad] = False
            longest = step.norm(dim=-1).amax(-1)
            step = step * (trust / longest.clamp(min=1e-12)).clamp(max=1.0)[:, None, None]
            step = step * active[:, None, None]
            Xp = Xbest + step
            try:
                E, Ei, F = evaluate(active)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:                                      # never fail the batch for one call
                if "out of memory" in str(exc):                           # TorchScript wraps CUDA OOM: split the chunk
                    raise
                for k in active.nonzero()[:, 0].tolist():
                    reason[k] = f"error: {type(exc).__name__}: {exc}"[:200]
                break
            n_eval += active.long()
            rbase = cache.pop("rebase", None)
            if rbase is not None:
                ridx = torch.as_tensor(rbase[0], device=dev)
                Eb, Eib, Fb = rbase[1]
                Ebest[ridx], Eibest[ridx], Fbest[ridx] = Eb[ridx], Eib[ridx], Fb[ridx]
                n_eval[ridx] += 1
            finite = torch.isfinite(E) & torch.isfinite(F.reshape(S, -1)).all(-1)
            accept = active & finite & (Ei <= Eibest + _REJECT_TOL_EV + rel_tol * Eibest.abs())
            reject = active & ~accept
            # memory update for accepted steps (curvature condition s.y > 0)
            s_new = (Xp - Xbest).reshape(S, D)
            y_new = (Fbest - F).reshape(S, D)                             # gradient difference g_new - g_old
            sy_new = (s_new * y_new).sum(-1)
            push = accept & (sy_new > 1e-12)
            if bool(push.any()):
                pidx = push.nonzero()[:, 0]
                s_h[pidx] = torch.roll(s_h[pidx], -1, dims=1)
                y_h[pidx] = torch.roll(y_h[pidx], -1, dims=1)
                rho[pidx] = torch.roll(rho[pidx], -1, dims=1)
                valid[pidx] = torch.roll(valid[pidx], -1, dims=1)
                clean[pidx] = torch.roll(clean[pidx], -1, dims=1)
                clean[pidx, -1] = ~((Ei - E)[pidx] > 0) & ~((Eibest - Ebest)[pidx] > 0)   # restraint 0 at both ends
                s_h[pidx, -1] = s_new[pidx]
                y_h[pidx, -1] = y_new[pidx]
                rho[pidx, -1] = 1.0 / sy_new[pidx]
                valid[pidx, -1] = True
            dE_tr = Ei - Eibest
            Xbest = torch.where(accept[:, None, None], Xp, Xbest)
            state["Xbest"] = Xbest
            Ebest = torch.where(accept, E, Ebest)
            Eibest = torch.where(accept, Ei, Eibest)
            Fbest = torch.where(accept[:, None, None], F, Fbest)
            if e_ring is not None:
                e_ring = torch.cat([e_ring[:, 1:], torch.where(active, Eibest, e_ring[:, -1])[:, None]], dim=1)
            trust = torch.where(accept, (trust * 1.25).clamp(max=_TRUST_MAX), (trust * 0.5).clamp(min=trust_min))
            if getattr(self, "_trace", None) is not None:                 # research: per-step state of every row
                self._trace.append(dict(fm=fnorm(F).cpu().numpy(), accept=accept.cpu().numpy(),
                                        active=active.cpu().numpy(), trust=trust.cpu().numpy(),
                                        dE=dE_tr.cpu().numpy()))
            if getattr(self, "_snap", None):                              # research: best state at k evaluations
                for kk in self._snap:
                    hit = (n_eval >= kk) & ~snap_done[kk]
                    if bool(hit.any()):
                        hi = hit.nonzero()[:, 0]
                        snap_E[kk][hi], snap_X[kk][hi] = Ebest[hi], Xbest[hi]
                        snap_done[kk][hi] = True
            valid[reject] = False
            self._n_reject = getattr(self, "_n_reject", 0) + int(reject.sum())
            self._n_steps = getattr(self, "_n_steps", 0) + int(active.sum())
            Xp = Xbest
            moved = (Xbest - X_built).norm(dim=-1).amax(-1)
            if ff:                                                        # rebuild only the rows that moved
                rb = (moved > _PRECON_REBUILD_A) & active
                if bool(rb.any()):
                    ri = rb.nonzero()[:, 0]
                    Pinv[ri] = self._ff_precon_inv(Xbest, mask, pack, ri)
                    X_built[ri] = Xbest[ri]
            elif bool((moved > _PRECON_REBUILD_A).any()):
                Pinv = self._precon_inv(Xbest, mask)
                X_built = Xbest.clone()
        fm = fnorm(Fbest)
        out = []
        Xc, Ec, Fc, E0c, fmc, nec = (Xbest.cpu().numpy(), Ebest.cpu().numpy(), Fbest.cpu().numpy(), E0.cpu().numpy(),
                                     fm.cpu().numpy(), n_eval.cpu().numpy())
        if keep_state:
            keep_c, sh_c, yh_c, tr_c = ((valid & clean).cpu().numpy(), s_h.cpu().numpy(), y_h.cpu().numpy(),
                                        trust.cpu().numpy())
        for k in range(S):
            na = sizes[k]
            rs = reason[k] or "max_steps"
            out.append(RelaxRecord(positions=Xc[k, :na].copy(), energy_ev=float(Ec[k]), energy0_ev=float(E0c[k]),
                                   fmax=float(fmc[k]), n_eval=int(nec[k]),
                                   converged=bool(fmc[k] <= fmax) or (rs == "energy"),
                                   reason=rs, forces=Fc[k, :na].copy()))
            if keep_state and rs in ("fmax", "energy", "max_steps"):
                sel = np.flatnonzero(keep_c[k])
                out[-1].state = OptState(Xc[k, :na].copy(), sh_c[k, sel, :na * 3].reshape(-1, na, 3).copy(),
                                         yh_c[k, sel, :na * 3].reshape(-1, na, 3).copy(), float(tr_c[k]))
            if getattr(self, "_snap", None):                              # rows finished earlier: their final state
                out[-1].snap = {kk: (float(snap_E[kk][k]) if bool(snap_done[kk][k]) else float(Ec[k]),
                                     snap_X[kk][k, :na].cpu().numpy() if bool(snap_done[kk][k]) else Xc[k, :na].copy())
                                for kk in self._snap}
        return out
