"""S2b: per-pose solver state saved in the NCI v2 evaluation pools (capture only).

Synthetic inputs only (no GPU, no solver run). The GPU round-trip checks are in refit/nci_v2_s2b_verify.py.
"""
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from jacobik import engine as E  # noqa: E402
from jacobik.ik import angle_bends as AB  # noqa: E402
from jacobik.ik.ik_solver import S2B_STATE_SCHEMA, bend_state_constants  # noqa: E402
from refit import nci_v2_orchestrator as orch  # noqa: E402


def _conf(trial, asg, emb, thetas, bend):
    c = E.ConformerResult(complex=None, score=0.0, converged=True, rank=1, trial_idx=trial,
                          assignment_idx=asg, embed_idx=emb)
    c.solver_state = {"thetas": [np.asarray(t, dtype=np.float64) for t in thetas], "bend_prior": float(bend)}
    return c


def _meta(widths, cols=((0, 8), (1, 7)), k=(1800.0, 310.0), free=(2.0, 2.0)):
    cols_arr = np.asarray(cols, dtype=np.int64).reshape(-1, 2)
    return dict(bend_columns=cols_arr, bend_k=np.asarray(k, dtype=np.float64),
                bend_free_deg=np.asarray(free, dtype=np.float64), bend_scale=1.0, bend_per_z=True,
                rt_prior=AB.RT_KJ, bend_column_def_sha256="ab" * 32, theta_widths=list(widths))


def test_state_keys_shapes_dtypes_and_alignment():
    widths = [9, 8]  # two ligands: 6 + n_joints + n_bends each
    confs = [
        _conf(trial=3, asg=0, emb=None, thetas=[np.arange(9.0), np.arange(8.0) + 100], bend=-0.5),
        _conf(trial=0, asg=2, emb=1, thetas=[np.arange(9.0) + 10, np.arange(8.0) + 200], bend=-1.25),
    ]
    out = orch.build_s2b_state_arrays(confs, _meta(widths))
    P, T = 2, sum(widths)
    assert str(out["state_schema_version"]) == S2B_STATE_SCHEMA == "s2b_state.1"
    for key in ("trial_idx", "assignment_idx", "embed_idx"):
        assert out[key].shape == (P,) and out[key].dtype == np.int64
    assert out["thetas_flat"].shape == (P, T) and out["thetas_flat"].dtype == np.float64
    assert out["thetas_lig_offsets"].dtype == np.int64
    assert out["thetas_lig_offsets"].tolist() == [0, 9, 17]
    assert out["bend_columns"].shape == (2, 2) and out["bend_columns"].dtype == np.int64
    assert out["bend_k"].shape == (2,) and out["bend_k"].dtype == np.float64
    assert out["bend_free_deg"].shape == (2,) and out["bend_free_deg"].dtype == np.float64
    assert out["bend_scale"].dtype == np.float64 and out["rt_prior"].dtype == np.float64
    assert out["bend_per_z"].dtype == np.bool_
    assert out["bend_prior_solver"].shape == (P,) and out["bend_prior_solver"].dtype == np.float64


def test_pose_order_follows_conformer_order_and_none_is_minus_one():
    widths = [9, 8]
    confs = [
        _conf(trial=3, asg=0, emb=None, thetas=[np.arange(9.0), np.arange(8.0) + 100], bend=-0.5),
        _conf(trial=0, asg=2, emb=1, thetas=[np.arange(9.0) + 10, np.arange(8.0) + 200], bend=-1.25),
    ]
    out = orch.build_s2b_state_arrays(confs, _meta(widths))
    assert out["trial_idx"].tolist() == [3, 0]
    assert out["assignment_idx"].tolist() == [0, 2]
    assert out["embed_idx"].tolist() == [-1, 1]  # None is -1, never 0
    assert np.array_equal(out["thetas_flat"][0, :9], np.arange(9.0))
    assert np.array_equal(out["thetas_flat"][0, 9:], np.arange(8.0) + 100)
    assert np.array_equal(out["thetas_flat"][1, :9], np.arange(9.0) + 10)
    assert np.array_equal(out["thetas_flat"][1, 9:], np.arange(8.0) + 200)
    assert out["bend_prior_solver"].tolist() == [-0.5, -1.25]


def test_no_object_arrays_and_npz_roundtrip_without_pickle():
    confs = [_conf(trial=0, asg=0, emb=None, thetas=[np.zeros(6), np.ones(5)], bend=0.0)]
    out = orch.build_s2b_state_arrays(confs, _meta([6, 5]))
    for key, val in out.items():
        assert np.asarray(val).dtype != object, key
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as z:
        assert str(z["state_schema_version"]) == "s2b_state.1"
        assert z["thetas_flat"].shape == (1, 11)


def test_missing_state_fails_closed():
    c = E.ConformerResult(complex=None, score=0.0, converged=True, rank=1, trial_idx=0, assignment_idx=0)
    with pytest.raises(ValueError, match="no solver state"):
        orch.build_s2b_state_arrays([c], _meta([6, 5]))
    with pytest.raises(ValueError, match="no bend-prior constants"):
        orch.build_s2b_state_arrays([_conf(0, 0, None, [np.zeros(6), np.zeros(5)], 0.0)], None)


def test_width_mismatch_fails_closed():
    with pytest.raises(ValueError, match="width"):
        orch.build_s2b_state_arrays([_conf(0, 0, None, [np.zeros(7), np.zeros(5)], 0.0)], _meta([6, 5]))


def _fake_fk(n_joints, nb, k, free):
    return SimpleNamespace(n_joints=n_joints, n_angle_bends=nb, _bend_k=torch.tensor(k, dtype=torch.float32),
                           _bend_free=torch.tensor(free, dtype=torch.float32))


def test_bend_constants_columns_clamp_and_hash():
    fks = [_fake_fk(3, 2, [1800.0, 310.0], [1.0, 5.0]), _fake_fk(0, 0, [], [])]
    cfg = {"free_deg": 2.0, "scale": 1.0}
    meta = bend_state_constants(fks, cfg, theta_widths=[6 + 3 + 2, 6])
    # ligand 0: columns 6+3+j = 9, 10; ligand 1 has no bends
    assert meta["bend_columns"].tolist() == [[0, 9], [0, 10]]
    assert meta["bend_k"].tolist() == pytest.approx([1800.0, 310.0])
    # free zone is max(_bend_free, free_deg) per column
    assert meta["bend_free_deg"].tolist() == pytest.approx([2.0, 5.0])
    assert meta["bend_scale"] == 1.0 and meta["rt_prior"] == AB.RT_KJ
    again = bend_state_constants(fks, cfg, theta_widths=[11, 6])
    assert again["bend_column_def_sha256"] == meta["bend_column_def_sha256"]
    other = bend_state_constants([_fake_fk(3, 2, [1801.0, 310.0], [1.0, 5.0]), _fake_fk(0, 0, [], [])], cfg,
                                 theta_widths=[11, 6])
    assert other["bend_column_def_sha256"] != meta["bend_column_def_sha256"]


def test_bend_constants_without_prior_have_zero_scale():
    fks = [_fake_fk(3, 2, [1800.0, 310.0], [2.0, 2.0])]
    meta = bend_state_constants(fks, None, theta_widths=[11])
    assert meta["bend_scale"] == 0.0


def test_recomputed_energy_matches_solver_prior_formula():
    """E_bend = scale * sum(k * max(0,|beta|-free)^2 / 2) [kJ/mol]; solver adds -E/RT_KJ (per Z when bend_per_z)."""
    torch.manual_seed(0)
    k = torch.tensor([1800.0, 310.0, 1100.0])
    free = torch.tensor([2.0, 2.0, 17.0])
    beta = torch.deg2rad(torch.tensor([[12.0, -9.0, 30.0], [0.5, 1.0, -2.5]]))
    scale, Z = 1.0, 37.0
    solver_val = AB.energy_prior(beta, k, free, scale)  # what _bend_prior adds (before the per-Z division)
    e_kj = scale * (0.5 * k[None, :] * torch.clamp(beta.abs() - torch.deg2rad(free)[None, :], min=0.0) ** 2).sum(-1)
    recon = -e_kj / AB.RT_KJ
    assert torch.allclose(solver_val.double(), recon.double(), atol=1e-6)
    per_z = solver_val / Z
    assert torch.allclose((-per_z * AB.RT_KJ * Z).double(), e_kj.double(), atol=1e-5)


def test_attach_solver_state_indexes_rows_by_trial_idx():
    th = [torch.arange(12.0).reshape(3, 4), torch.arange(6.0).reshape(3, 2) + 50]
    bend = np.array([-0.1, -0.2, -0.3])
    c1 = E.ConformerResult(complex=None, score=1.0, converged=True, rank=1, trial_idx=2)
    c2 = E.ConformerResult(complex=None, score=0.5, converged=True, rank=2, trial_idx=0)
    E.attach_solver_state([c1, c2], th, bend)
    assert c1.solver_state["thetas"][0].tolist() == [8.0, 9.0, 10.0, 11.0]
    assert c1.solver_state["thetas"][1].tolist() == [54.0, 55.0]
    assert c1.solver_state["bend_prior"] == pytest.approx(-0.3)
    assert c2.solver_state["thetas"][0].dtype == np.float64
    assert c2.solver_state["bend_prior"] == pytest.approx(-0.1)


def test_attach_is_noop_without_bend():
    c = E.ConformerResult(complex=None, score=1.0, converged=True, rank=1, trial_idx=0)
    E.attach_solver_state([c], [torch.zeros(1, 4)], None)
    assert c.solver_state is None
