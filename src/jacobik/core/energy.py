"""MACE-POLAR-1 single-point comparisons on matched molecular geometries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from jacobik.core.env import resolve_device

EV_TO_KJ_MOL = 96.4853321233

#: Conformers per MACE forward pass when JACOBIK_MACE_BATCH_SIZE is not set.
DEFAULT_BATCH_SIZE = {"cpu": 4, "cuda": 32}
#: Atoms per MACE forward pass on CUDA (JACOBIK_MACE_BATCH_ATOMS). MACE-POLAR-1-S peaks at ~2-3.4 MB per atom (23 -> 113
#: atom complexes, float32, measured 2026-10-16), so 32 conformers of a 113-atom complex need 12 GB; 600 atoms is
#: ~2 GB. float64 takes ~2.5x the memory.
DEFAULT_BATCH_ATOMS = {"float32": 600, "float64": 300}
#: Share of the card's memory the caching allocator may take (JACOBIK_MACE_MEM_FRACTION). On Windows the driver's
#: system-memory fallback lets an allocation past the card's memory succeed in shared system RAM (slow, and it
#: starves the host) instead of raising, so the out-of-memory retry below never ran. The cap makes it raise.
DEFAULT_MEM_FRACTION = 0.85


@dataclass(frozen=True)
class EnergyComparison:
    reference_ev: float
    predicted_ev: float
    delta_ev: float
    delta_kj_mol: float
    absolute_delta_kj_mol: float
    absolute_delta_kj_mol_per_atom: float
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


class MACEPolarEvaluator:
    """Lazy MACE-POLAR-1 ASE backend.

    Absolute energies from tmQM (TPSSh-D3BJ) and MACE-POLAR-1
    (omegaB97M-V training level) must not be subtracted. This class evaluates
    both geometries with the *same* model and reports the meaningful matched-
    composition strain/basin energy ``E(predicted) - E(reference)``.
    """

    def __init__(
        self, model: str = "polar-1-m", device: str = "auto", dtype: str = "float32",
        model_path: str | Path | None = None, mode: str = "energy-only", threads: int | None = None,
        enable_cueq: bool = False,
    ):
        """Create an evaluator, optionally with an already-downloaded checkpoint.

        ``mace_polar`` interprets symbolic model names as a request to download
        a checkpoint.  ``model_path`` (or ``JACOBIK_MACE_MODEL_PATH``) makes
        benchmarking reproducible and usable offline.  The path is deliberately
        not hard-coded: it may live in a user cache, shared drive, or job store.

        ``device``: "auto" (CUDA when available, else CPU), "cpu", "cuda" or "cuda:N"; also
        ``JACOBIK_MACE_DEVICE``.  The calculator is built once on that device on first use and
        stays there for the life of this object: keep one evaluator per process and reuse it.
        """
        if device in (None, "auto"):
            device = os.environ.get("JACOBIK_MACE_DEVICE") or "auto"
        device = resolve_device(device)
        if model not in {"polar-1-s", "polar-1-m", "polar-1-l"}:
            raise ValueError("model must be polar-1-s, polar-1-m, or polar-1-l")
        if mode not in {"energy-only", "full"}:
            raise ValueError("mode must be 'energy-only' or 'full'")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if threads is not None and int(threads) < 1:
            raise ValueError("threads must be at least 1")
        configured_path = model_path or os.environ.get("JACOBIK_MACE_MODEL_PATH")
        self.model, self.device, self.dtype, self.mode = model, device, dtype, mode
        self.threads = None if threads is None else int(threads)
        #: cuEquivariance kernels for the tensor products (CUDA only; needs cuequivariance-torch)
        self.enable_cueq = bool(enable_cueq) and str(device).startswith("cuda")
        self.model_path = None if configured_path is None else Path(configured_path).expanduser()
        if self.model_path is not None and not self.model_path.is_file():
            raise FileNotFoundError(
                f"MACE checkpoint does not exist: {self.model_path}. "
                "Set JACOBIK_MACE_MODEL_PATH to a downloaded .model file or omit it to download."
            )
        self._calculator = None
        #: conformers per forward pass actually in use (lowered automatically after a CUDA out-of-memory)
        self._batch_size: int | None = None
        #: atoms per forward pass actually in use on CUDA (lowered with the batch size after an out-of-memory)
        self._batch_atoms: int | None = None
        self.load_s: float | None = None

    @property
    def on_cuda(self) -> bool:
        return str(self.device).startswith("cuda")

    @contextmanager
    def _thread_budget(self):
        """Temporarily give CPU inference its configured intra-op thread budget (no-op on CUDA)."""
        import torch

        if self.on_cuda:
            yield
            return
        configured = self.threads
        if configured is None and os.environ.get("JACOBIK_MACE_THREADS"):
            configured = max(1, int(os.environ["JACOBIK_MACE_THREADS"]))
        previous = torch.get_num_threads()
        try:
            if configured is not None and configured != previous:
                torch.set_num_threads(configured)
            yield
        finally:
            if torch.get_num_threads() != previous:
                torch.set_num_threads(previous)

    def _get_calculator(self):
        if self._calculator is None:
            try:
                from mace.calculators import mace_polar
            except ImportError as exc:
                raise RuntimeError(
                    "MACE-POLAR-1 is optional; install with `pip install -e .[benchmark]`"
                ) from exc
            # mace_polar currently changes torch's process-global default
            # dtype while constructing a calculator. Preserve it because the
            # Jacobik solver may run again after an energy evaluation.
            import time
            import torch
            previous_dtype = torch.get_default_dtype()
            t0 = time.perf_counter()
            try:
                if self.on_cuda:
                    index = torch.device(self.device).index or 0
                    torch.cuda.set_per_process_memory_fraction(
                        float(os.environ.get("JACOBIK_MACE_MEM_FRACTION", DEFAULT_MEM_FRACTION)), index)
                extra = {"enable_cueq": True} if self.enable_cueq else {}
                self._calculator = mace_polar(
                    model=self.model_path or self.model, device=self.device, default_dtype=self.dtype, **extra
                )
                for model in self._calculator.models:
                    model.eval()
            finally:
                torch.set_default_dtype(previous_dtype)
            self.load_s = time.perf_counter() - t0
        return self._calculator

    def load(self) -> "MACEPolarEvaluator":
        """Build the calculator now (otherwise it is built on the first energy call)."""
        self._get_calculator()
        return self

    def describe(self) -> dict:
        """Where the model lives, for run records."""
        info = {"device": str(self.device), "dtype": self.dtype, "mode": self.mode, "batch_size": self._batch_size,
                "batch_atoms": self._batch_atoms}
        if self.on_cuda:
            import torch

            index = torch.device(self.device).index or 0
            info["gpu"] = torch.cuda.get_device_name(index)
            info["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated(index) / 2**20, 1)
        return info

    def energy(self, symbols: Sequence[str], positions: np.ndarray, charge: int, multiplicity: int) -> float:
        """Energy in eV using the configured ``energy-only`` or ``full`` path."""
        return float(self.energies([(symbols, positions, charge, multiplicity)])[0])

    def properties(
        self, symbols: Sequence[str], positions: np.ndarray, charge: int, multiplicity: int,
    ) -> dict:
        """Run MACE's full ASE calculation and return a detached copy of every property.

        This deliberately preserves the slower historical path that computes forces, stress and
        the other calculator outputs alongside energy.  It is useful for force diagnostics and
        finite-difference Hessian work; routine geometry benchmarks should use ``energy-only``.
        """
        try:
            from ase import Atoms
        except ImportError as exc:
            raise RuntimeError("ASE is required for energy benchmarking") from exc

        atoms = Atoms(symbols=list(symbols), positions=np.asarray(positions, dtype=float), pbc=False)
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(multiplicity)
        atoms.info["external_field"] = [0.0, 0.0, 0.0]
        import torch
        previous_dtype = torch.get_default_dtype()
        try:
            with self._thread_budget():
                atoms.calc = self._get_calculator()
                atoms.get_potential_energy()
                return {
                    key: value.copy() if hasattr(value, "copy") else value
                    for key, value in atoms.calc.results.items()
                }
        finally:
            torch.set_default_dtype(previous_dtype)

    def energies(
        self,
        structures: Sequence[tuple[Sequence[str], np.ndarray, int, int]],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate several independent geometries in energy-only mini-batches.

        The stock ASE calculator requests forces and stress along with energy and accepts one
        ``Atoms`` object at a time.  Benchmarking needs only total energies, so calling the same
        loaded PolarMACE model with ``compute_force=False`` and ``compute_stress=False`` avoids
        derivative graphs and amortises model overhead across conformers.  No solver state is used.
        """
        if not structures:
            return np.empty(0, dtype=np.float64)
        if self.mode == "full":
            return np.asarray(
                [self.properties(*structure)["energy"] for structure in structures], dtype=np.float64
            )
        try:
            from ase import Atoms
        except ImportError as exc:
            raise RuntimeError("ASE is required for energy benchmarking") from exc

        import torch
        from mace import data as mace_data
        # MACE ships a compatible, vendored PyG subset even when the standalone
        # torch_geometric distribution is absent (as in the benchmark environment).
        from mace.tools import torch_geometric, torch_tools

        previous_dtype = torch.get_default_dtype()
        try:
            with self._thread_budget():
                calc = self._get_calculator()
                data = []
                calc.arrays_keys.update({calc.charges_key: "charges"})
                keyspec = mace_data.KeySpecification(
                    info_keys=calc.info_keys, arrays_keys=calc.arrays_keys,
                )
                # Match MACECalculator._atoms_to_batch: constructing at the caller's
                # float32 default and upcasting later irreversibly rounds positions,
                # even when the model itself runs in float64.
                with torch_tools.default_dtype(calc.default_dtype):
                    for symbols, positions, charge, multiplicity in structures:
                        atoms = Atoms(symbols=list(symbols), positions=np.asarray(positions, dtype=float), pbc=False)
                        atoms.info["charge"] = int(charge)
                        atoms.info["spin"] = int(multiplicity)
                        atoms.info["external_field"] = [0.0, 0.0, 0.0]
                        config = mace_data.config_from_atoms(
                            atoms, key_specification=keyspec, head_name=calc.head,
                        )
                        data.append(mace_data.AtomicData.from_config(
                            config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads,
                        ))

                if batch_size is None:
                    batch_size = self._batch_size or int(os.environ.get(
                        "JACOBIK_MACE_BATCH_SIZE", DEFAULT_BATCH_SIZE["cuda" if self.on_cuda else "cpu"]))
                batch_size = max(1, int(batch_size))
                max_atoms = self._batch_atoms or (
                    int(os.environ.get("JACOBIK_MACE_BATCH_ATOMS", DEFAULT_BATCH_ATOMS[self.dtype])) if self.on_cuda else None)
                while True:
                    try:
                        values = self._forward_batches(calc, data, batch_size, max_atoms, torch, torch_geometric)
                        break
                    except RuntimeError as exc:
                        # CUDA out of memory: free the failed batch, halve and retry, and keep the smaller
                        # size for the rest of the run (the model itself stays loaded on the GPU).
                        oom = self.on_cuda and ("out of memory" in str(exc).lower())
                        if not oom or (batch_size == 1 and (max_atoms or 0) <= 1):
                            raise
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        max_atoms = max(1, max_atoms // 2) if max_atoms else None
                self._batch_size, self._batch_atoms = batch_size, max_atoms
                return np.asarray(values, dtype=np.float64)
        finally:
            torch.set_default_dtype(previous_dtype)

    @staticmethod
    def _forward_batches(calc, data, batch_size, max_atoms, torch, torch_geometric) -> list:
        """Energies in input order; a batch holds at most batch_size graphs and (when max_atoms is set) at most
        max_atoms atoms in all, so a large complex runs fewer conformers per pass (always at least one)."""
        chunks, cur, n_at = [], [], 0
        for d in data:
            n = int(d.num_nodes)
            if cur and (len(cur) >= batch_size or (max_atoms and n_at + n > max_atoms)):
                chunks.append(cur)
                cur, n_at = [], 0
            cur.append(d)
            n_at += n
        if cur:
            chunks.append(cur)
        values = []
        for chunk in chunks:
            batch_base = next(iter(torch_geometric.dataloader.DataLoader(
                dataset=chunk, batch_size=len(chunk), shuffle=False, drop_last=False)))
            batch_base = batch_base.to(calc.device)
            per_model = []
            for model in calc.models:
                batch = calc._clone_batch(batch_base)
                model_dtype = next(model.parameters()).dtype
                for key in batch.keys:
                    value = batch[key]
                    if torch.is_tensor(value) and torch.is_floating_point(value):
                        batch[key] = value.to(dtype=model_dtype)
                with torch.no_grad():
                    out = model(
                        batch.to_dict(),
                        compute_force=False,
                        compute_virials=False,
                        compute_stress=False,
                        training=False,
                        compute_edge_forces=False,
                        compute_atomic_stresses=False,
                    )
                per_model.append(out["energy"].detach())
            # one device->host transfer per batch
            energy = torch.stack(per_model).mean(0).to("cpu", torch.float64).numpy()
            values.extend(energy.reshape(-1) * calc.energy_units_to_eV)
        return values

    def forces(self, symbols: Sequence[str], positions: np.ndarray, charge: int, multiplicity: int) -> np.ndarray:
        """Return MLIP forces in eV/Å (the local downhill direction)."""
        try:
            from ase import Atoms
        except ImportError as exc:
            raise RuntimeError("ASE is required for force benchmarking") from exc
        atoms = Atoms(symbols=list(symbols), positions=np.asarray(positions, dtype=float), pbc=False)
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(multiplicity)
        atoms.info["external_field"] = [0.0, 0.0, 0.0]
        import torch
        previous_dtype = torch.get_default_dtype()
        try:
            with self._thread_budget():
                atoms.calc = self._get_calculator()
                return np.asarray(atoms.get_forces(), dtype=np.float64)
        finally:
            torch.set_default_dtype(previous_dtype)

    def compare(
        self,
        symbols: Sequence[str],
        reference: np.ndarray,
        predicted: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        predicted_symbols: Sequence[str] | None = None,
    ) -> EnergyComparison:
        if np.asarray(reference).shape != np.asarray(predicted).shape:
            raise ValueError("energy comparison requires identical composition and atom count")
        pred_symbols = list(predicted_symbols) if predicted_symbols is not None else list(symbols)
        if sorted(pred_symbols) != sorted(symbols):
            raise ValueError("energy comparison requires identical elemental composition")
        e_ref = self.energy(symbols, reference, charge, multiplicity)
        e_pred = self.energy(pred_symbols, predicted, charge, multiplicity)
        delta = e_pred - e_ref
        delta_kj = delta * EV_TO_KJ_MOL
        return EnergyComparison(
            reference_ev=e_ref,
            predicted_ev=e_pred,
            delta_ev=delta,
            delta_kj_mol=delta_kj,
            absolute_delta_kj_mol=abs(delta_kj),
            absolute_delta_kj_mol_per_atom=abs(delta_kj) / max(len(symbols), 1),
            model=self.model,
        )
