"""test_benchmark_energy: MACE input precision and failure restoration."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

pytest.importorskip("mace")
from mace import data as mace_data
from mace.tools import AtomicNumberTable
from jacobik.core.energy import MACEPolarEvaluator


def _evaluator(monkeypatch, dtype):
    ev = MACEPolarEvaluator(device="cpu", dtype=dtype)
    calc = SimpleNamespace(
        arrays_keys={},
        charges_key="charges",
        info_keys={"total_charge": "charge", "total_spin": "spin", "external_field": "external_field"},
        head="Default",
        available_heads=["Default"],
        z_table=AtomicNumberTable([1]),
        r_max=5.0,
        default_dtype=dtype,
    )
    monkeypatch.setattr(ev, "_get_calculator", lambda: calc)
    return ev


def test_mace_converter_precision_charge_and_failure_cleanup(monkeypatch):
    """Converter keeps requested precision/charge across ambient dtypes and restores caller dtype on failure."""
    X = np.array([[0.0, 0.0, 0.0], [1.0 + 2.0 ** -40, 0.0, 0.0]])
    for ambient in (torch.float32, torch.float64):
        for requested in ("float32", "float64"):
            ev = _evaluator(monkeypatch, requested)
            seen = []

            def forward(calc, data, *args):
                seen.extend(data)
                return [float(d.positions[1, 0]) for d in data]

            monkeypatch.setattr(ev, "_forward_batches", forward)
            old = torch.get_default_dtype()
            try:
                torch.set_default_dtype(ambient)
                result = ev.energies([(["H", "H"], X, -1, 2)])
                assert torch.get_default_dtype() == ambient
            finally:
                torch.set_default_dtype(old)
            expected = torch.float64 if requested == "float64" else torch.float32
            assert seen[0].positions.dtype == expected
            assert result[0] == float(torch.tensor(X[1, 0], dtype=expected))
            if requested == "float64":
                assert result[0] != float(np.float32(X[1, 0]))
            assert seen[0].total_charge.item() == -1 and seen[0].total_spin.item() == 2

    ev_fail = _evaluator(monkeypatch, "float64")

    def fail(*args, **kwargs):
        assert torch.get_default_dtype() == torch.float64
        raise RuntimeError("conversion failure")

    monkeypatch.setattr(mace_data.AtomicData, "from_config", fail)
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with pytest.raises(RuntimeError, match="conversion failure"):
            ev_fail.energies([(["H", "H"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 0, 1)])
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(old)
