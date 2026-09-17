import numpy as np
import pytest

from jacobik.benchmark.energy import EV_TO_KJ_MOL, MACEPolarEvaluator


class FakeEvaluator(MACEPolarEvaluator):
    def energy(self, symbols, positions, charge, multiplicity):
        return float(np.asarray(positions).sum())


def test_energy_delta_uses_same_model_for_both_geometries():
    evaluator = FakeEvaluator()
    ref = np.zeros((2, 3))
    pred = np.ones((2, 3))
    result = evaluator.compare(["H", "H"], ref, pred, charge=0, multiplicity=1)
    assert result.delta_ev == 6.0
    assert result.delta_kj_mol == pytest.approx(6 * EV_TO_KJ_MOL)
    assert result.absolute_delta_kj_mol_per_atom == pytest.approx(3 * EV_TO_KJ_MOL)


def test_composition_mismatch_rejected():
    with pytest.raises(ValueError):
        FakeEvaluator().compare(
            ["C"], np.zeros((1, 3)), np.zeros((1, 3)), charge=0,
            multiplicity=1, predicted_symbols=["N"],
        )
