"""Beam-start lineage telemetry: outcome classes on a tiny synthetic pass (CPU, fake deduper)."""
import json
import types

import numpy as np
import pytest
import torch

from tools.jdev import lineage


class FakeDeduper:
    device = torch.device("cpu")
    dtype = torch.float64
    is_capped = False

    def __init__(self, n):
        self.heavy = list(range(n))

    def _radial(self, X):
        return X.norm(dim=-1)

    def _full_any(self, Xi, Xk, th):
        a = Xi - Xi.mean(0)
        return bool(((a - (Xk[0] - Xk[0].mean(0))) ** 2).sum(-1).mean().sqrt() <= th)


def test_outcomes_and_no_output_without_env(tmp_path, monkeypatch):
    assert not lineage.enabled()
    monkeypatch.setenv("JACOBIK_LINEAGE_OUT", str(tmp_path))
    assert lineage.enabled()
    base = torch.tensor([[0., 0, 0], [1.5, 0, 0], [0, 2.0, 0], [0, 0, 2.5]], dtype=torch.float64)
    rows = [base, base + 0.01, base * torch.tensor([1., 1.5, 1., 1.]).view(4, 1), base, base, base * torch.tensor([2., 2., 2., 2.]).view(4, 1)]
    X_all = torch.stack(rows)
    X_all[3] = float("nan")                         # non-finite coordinates
    scores = torch.tensor([5.0, 4.0, 3.0, 2.0, -1e9, 1.0])
    ctx = types.SimpleNamespace(n_coord_samples=1, score_norm=1.0)
    th_init = [torch.zeros(6, 6 + 2)]
    th_init[0][:, 6] = torch.tensor([0.0, 0.02, 1.0, 2.0, 3.0, 0.0])    # rows 0,1,5 are torsion-space twins
    conv = torch.ones(6)
    selected = [np.array([0, 2])]
    lines = lineage.record(ctx, [np.array([0, 1])], th_init, None, scores, conv, X_all, np.zeros(6, int), 6, 1, selected,
                           FakeDeduper(4), 2, 1.5, 6, None, dict(seeds=[11], code="XXXXXX", arm="control"))
    rec = lines[0]
    out = rec["rows_table"]["outcome"]
    assert out[0] == "kept" and out[2] == "kept"
    assert out[1] == "merged" and rec["rows_table"]["merged_into"][1] == 0
    assert out[3] == "nonfinite"
    assert out[4] == "outside_window"
    assert out[5] == "beyond_cap"
    assert rec["code"] == "XXXXXX" and rec["seed"] == 11 and rec["rows"] == 6
    assert rec["solver_s_per_row"] == pytest.approx(0.25)
    assert rec["redundancy"]["0.05"]["share_with_earlier_twin"] > 0
    assert (tmp_path / f"lineage_{__import__('os').getpid()}.jsonl").exists()
