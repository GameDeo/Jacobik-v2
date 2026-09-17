"""MACE-POLAR-1 single-point comparisons on matched molecular geometries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

EV_TO_KJ_MOL = 96.4853321233


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

    def __init__(self, model: str = "polar-1-m", device: str = "cpu", dtype: str = "float64"):
        if model not in {"polar-1-s", "polar-1-m", "polar-1-l"}:
            raise ValueError("model must be polar-1-s, polar-1-m, or polar-1-l")
        self.model, self.device, self.dtype = model, device, dtype
        self._calculator = None

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
            import torch
            previous_dtype = torch.get_default_dtype()
            try:
                self._calculator = mace_polar(
                    model=self.model, device=self.device, default_dtype=self.dtype
                )
            finally:
                torch.set_default_dtype(previous_dtype)
        return self._calculator

    def energy(self, symbols: Sequence[str], positions: np.ndarray, charge: int, multiplicity: int) -> float:
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
            atoms.calc = self._get_calculator()
            return float(atoms.get_potential_energy())
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
