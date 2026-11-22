"""MACE-POLAR-1 single-point comparisons on matched molecular geometries (pure CuPy + Triton)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from jacobik.core.env import resolve_device

EV_TO_KJ_MOL = 96.4853321233

#: Conformers per MACE forward pass when JACOBIK_MACE_BATCH_SIZE is not set.
DEFAULT_BATCH_SIZE = 32
#: Atoms per MACE forward pass on CUDA (JACOBIK_MACE_BATCH_ATOMS).
DEFAULT_BATCH_ATOMS = {"float32": 600, "float64": 300}
#: Share of the card's memory the caching allocator may take (JACOBIK_MACE_MEM_FRACTION).
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
    """Lazy MACE-POLAR-1 Triton/CuPy evaluator (zero PyTorch)."""

    def __init__(
        self, model: str = "polar-1-s", device: str = "auto", dtype: str = "float32",
        model_path: str | Path | None = None, mode: str = "energy-only", threads: int | None = None,
    ):
        if device in (None, "auto"):
            device = os.environ.get("JACOBIK_MACE_DEVICE") or "auto"
        device = resolve_device(device)
        if not str(device).startswith("cuda"):
            raise RuntimeError("MACEPolarEvaluator requires CUDA (GPU-only Triton build)")
        if model != "polar-1-s":
            raise ValueError("the GPU build runs MACE-POLAR-1-S only (model='polar-1-s')")
        if mode not in {"energy-only", "full"}:
            raise ValueError("mode must be 'energy-only' or 'full'")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if threads is not None and int(threads) < 1:
            raise ValueError("threads must be at least 1")
        configured_path = model_path or os.environ.get("JACOBIK_MACE_MODEL_PATH")
        self.model, self.device, self.dtype, self.mode = model, device, dtype, mode
        self.threads = None if threads is None else int(threads)
        self.model_path = None if configured_path is None else Path(configured_path).expanduser()
        if self.model_path is not None and not self.model_path.is_file():
            raise FileNotFoundError(
                f"MACE checkpoint does not exist: {self.model_path}. "
                "Set JACOBIK_MACE_MODEL_PATH to a downloaded .model file or omit it to download."
            )
        self._compiled = None
        self._batch_size: int | None = None
        self._batch_atoms: int | None = None
        self.load_s: float | None = None

    @property
    def on_cuda(self) -> bool:
        return True

    def _get_compiled(self):
        if self._compiled is None:
            import time
            from jacobik.core import gpu, mace_triton

            t0 = time.perf_counter()
            dev_str = str(self.device)
            idx = int(dev_str.split(":")[1]) if ":" in dev_str else 0
            gpu.cp.cuda.Device(idx).use()
            pool = gpu.cp.get_default_memory_pool()
            if not pool.get_limit():          # the caller's limit (e.g. the bench's) wins
                pool.set_limit(fraction=float(os.environ.get("JACOBIK_MACE_MEM_FRACTION", DEFAULT_MEM_FRACTION)))
            self._compiled = mace_triton.load_compiled(
                model=self.model, model_path=self.model_path, dtype=self.dtype,
            )
            self.load_s = time.perf_counter() - t0
        return self._compiled

    def load(self) -> "MACEPolarEvaluator":
        """Load and compile the Triton model now."""
        self._get_compiled()
        return self

    def describe(self) -> dict:
        """Where the model lives, for run records."""
        from jacobik.core import gpu

        dev_str = str(self.device)
        idx = int(dev_str.split(":")[1]) if ":" in dev_str else 0
        props = gpu.cp.cuda.runtime.getDeviceProperties(idx)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        used = gpu.cp.get_default_memory_pool().used_bytes()
        return {
            "device": dev_str,
            "dtype": self.dtype,
            "mode": self.mode,
            "batch_size": self._batch_size,
            "batch_atoms": self._batch_atoms,
            "gpu": name,
            "gpu_mem_peak_mb": round(used / 2**20, 1),
        }

    def energy(self, symbols: Sequence[str], positions: np.ndarray, charge: int, multiplicity: int) -> float:
        """Energy in eV using the configured ``energy-only`` or ``full`` path."""
        return float(self.energies([(symbols, positions, charge, multiplicity)])[0])

    def properties(
        self, symbols: Sequence[str], positions: np.ndarray, charge: int, multiplicity: int,
    ) -> dict:
        """Evaluate energy and forces in eV and eV/Å via Triton."""
        from jacobik.core import gpu, mace_triton

        cm = self._get_compiled()
        e_t, f_t = mace_triton.eval_structures_ef(
            cm, [(symbols, positions, charge, multiplicity)], compute_force=True,
        )
        return {
            "energy": float(gpu.to_cpu(e_t)[0]),
            "forces": np.asarray(gpu.to_cpu(f_t), dtype=np.float64),
        }

    def energies(
        self,
        structures: Sequence[tuple[Sequence[str], np.ndarray, int, int]],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate several independent geometries in energy-only mini-batches via Triton."""
        if not structures:
            return np.empty(0, dtype=np.float64)
        if self.mode == "full":
            return np.asarray(
                [self.properties(*structure)["energy"] for structure in structures], dtype=np.float64
            )
        from jacobik.core import gpu

        cm = self._get_compiled()
        if batch_size is None:
            batch_size = self._batch_size or int(os.environ.get(
                "JACOBIK_MACE_BATCH_SIZE", DEFAULT_BATCH_SIZE))
        batch_size = max(1, int(batch_size))
        max_atoms = self._batch_atoms or int(os.environ.get(
            "JACOBIK_MACE_BATCH_ATOMS", DEFAULT_BATCH_ATOMS[self.dtype]))
        while True:
            try:
                values = self._forward_batches(cm, structures, batch_size, max_atoms)
                break
            except (gpu.OOMError, RuntimeError) as exc:
                oom = isinstance(exc, gpu.OOMError) or ("out of memory" in str(exc).lower())
                if not oom or (batch_size == 1 and (max_atoms or 0) <= 1):
                    raise
                gpu.empty_cache()
                batch_size = max(1, batch_size // 2)
                max_atoms = max(1, max_atoms // 2) if max_atoms else None
        self._batch_size, self._batch_atoms = batch_size, max_atoms
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _forward_batches(cm, structures, batch_size, max_atoms) -> list:
        """Energies in input order; a batch holds at most batch_size graphs and max_atoms atoms."""
        from jacobik.core import gpu, mace_triton

        chunks, cur, n_at = [], [], 0
        for s in structures:
            n = len(s[0])
            if cur and (len(cur) >= batch_size or (max_atoms and n_at + n > max_atoms)):
                chunks.append(cur)
                cur, n_at = [], 0
            cur.append(s)
            n_at += n
        if cur:
            chunks.append(cur)
        values = []
        for chunk in chunks:
            e_triton, _ = mace_triton.eval_structures_ef(cm, chunk, compute_force=False)
            values.extend(np.asarray(gpu.to_cpu(e_triton), dtype=np.float64).reshape(-1))
        return values

    def forces(self, symbols: Sequence[str], positions: np.ndarray, charge: int, multiplicity: int) -> np.ndarray:
        """Return MLIP forces in eV/Å (the local downhill direction)."""
        return self.properties(symbols, positions, charge, multiplicity)["forces"]

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
