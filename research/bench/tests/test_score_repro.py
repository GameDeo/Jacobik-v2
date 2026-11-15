from types import SimpleNamespace

import pytest

import numpy as np
import torch

from tools.jgen.score_repro import SolveCapture, conformer_key


def test_conformer_key_includes_variant_assignment_and_trial():
    conf = SimpleNamespace(embed_idx=2, assignment_idx=4, trial_idx=17)
    assert conformer_key(conf) == (2, 4, 17)


@pytest.mark.parametrize("missing", ["trial_idx", "assignment_idx"])
def test_conformer_key_rejects_missing_row_identity(missing):
    values = {"embed_idx": 0, "assignment_idx": 1, "trial_idx": 2}
    values[missing] = None
    with pytest.raises(ValueError):
        conformer_key(SimpleNamespace(**values))


def test_score_capture_evaluates_row_and_keeps_float64():
    class FK:
        _cl = None

        def apply(self, theta, X):
            return X + theta[:, 0, None, None]

    class Scorer:
        def score(self, X, row_groups=None):
            return X[:, 0, 0], {}

    cap = SolveCapture(
        fks=[FK()], scorer=Scorer(), ge=None,
        X_full=torch.zeros((2, 1, 3), dtype=torch.float32),
        theta_inits=[torch.zeros((2, 1))], row_groups=torch.tensor([0, 1]),
        bend_prior=None, score_kwargs={}, scores=torch.tensor([0.25, -0.5]),
        thetas_final=[torch.zeros((2, 1))], candidate_idx_per_b=np.array([0, 1]), variant_index=0,
    )
    score = cap.score([np.array([0.125], dtype=np.float64)], 0)
    assert score.dtype == torch.float64
    assert score.item() == pytest.approx(0.125)
