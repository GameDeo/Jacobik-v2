"""test_solver: merged from test_finish_phase, test_finish_phase_switches, test_polish, test_fastcheck_solver_fixes, test_gpu_fast_path, test_kabsch_init, test_macrocycle, c2_src."""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jacobik import ComplexBuilder, LigandInput
from jacobik.builder.engine import ComplexBuilder as _CB, _LigandEmbedFailure
from jacobik.ik import ik_solver as IKM
from jacobik.ik.finish_phase import FinishPhase
from jacobik.ik.fk import ForwardKinematics, axisangle_to_matrix
from jacobik.ik.ik_solver import IKSolverConfig, merge_chunk_traces
from jacobik.ik.kernels import dihedral
from jacobik.ik.polish import IRpropState, optimise
from jacobik.topology.decompose import decompose_ligand

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "dev" / "tests" / "regression" / "fixtures"
CASES = {
    "MELLUN": ("tmqm_sample40", "S"),
    "AHIGOQ": ("tmqm_sample40", "S"),
    "PIDXOT": ("tmqm_sample40", "S"),
    "HAFZOG": ("tmqm_sample40", "S"),
    "CUWMEP": ("n48", "N"),
    "FMNBCU": ("n48", "N"),
}


def _builder(code):
    from tools.jdev.cases import get_case
    folder, split = CASES[code]
    return get_case(f"tmqm:{FIX / folder}:{split}:{code}").make_builder()


@pytest.fixture()
def one_thread__gpu_fast_path(monkeypatch):
    monkeypatch.setenv("JACOBIK_NUM_WORKERS", "1")
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture
def legacy_polish_context(monkeypatch):
    from jacobik.ik import solver_replay as SR
    for module in (IKM, SR):
        monkeypatch.setattr(module, "FINISH_PHASE", None)


def _phase(**kw):
    config = dict(
        release=["shape", "symmetry", "projection"],
        cap=8,
        sigma=dict(translation=0.0585, rotation=0.08609, torsion=0.2402, ring=0.2402, bend=0.08609),
    )
    config.update(kw)
    return FinishPhase([torch.zeros(2, 8)], [SimpleNamespace(n_joints=2, n_macro_joints=0)], 24, 8, config, 4, 1e-5)


def test_finish_phase_state_and_adaptive_schedule():
    from jacobik.core import profile

    # Row switch + aid + stopped-row zeroing + select + short-budget search
    p = _phase()
    p.m[0].fill_(2)
    for t in range(12):
        p.observe(torch.tensor([t]), torch.tensor([1.0, float(t)]))
    assert p.active.tolist() == [True, False] and p.m[0][0].eq(0).all() and p.m[0][1].eq(2).all()
    assert p.aid("shape", torch.ones(2)).tolist() == [0.0, 1.0]
    p.observe(torch.tensor([16]), torch.tensor([1.0, 16.0]))
    assert p.active.all()

    ps = _phase(score_only=True, optimizer="irprop-")
    ps.observe(torch.tensor([0]), torch.ones(2))
    g0 = torch.full((2, 8), 10.0); g0[0] = 0
    steps = ps.update(torch.tensor([0]), [g0], [g0], torch.ones(2), None, 0.1)
    assert ps.done.tolist() == [True, False] and steps[0][0].eq(0).all()
    assert torch.allclose(steps[0][1], 0.14 * ps.sigma[0][1])
    ps.select(torch.tensor([1]))
    assert ps.active.shape == (1,) and ps.sigma[0].shape == (1, 8)

    p2 = FinishPhase([torch.zeros(2, 8)], [SimpleNamespace(n_joints=2, n_macro_joints=0)], 2, 100, profile.get("finish_phase"), 4, 1e-5)
    for step in range(2):
        p2.observe(torch.tensor([step]), torch.ones(2))
        assert not p2.active.any()
        assert torch.isfinite(p2.update(torch.tensor([step]), [torch.ones(2, 8)], [torch.ones(2, 8)], torch.ones(2), None, 0.1)[0]).all()

    # Curvature scaling affects only switching rows
    pc = _phase(sigma_source="curvature")
    th = torch.ones(2, 8, requires_grad=True)
    grad = torch.autograd.grad((th.square() * 2).sum(), th, create_graph=True)[0]
    before = pc.sigma[0].clone()
    pc.switched[0] = True
    pc.curvature([grad], [th], 2.0)
    assert torch.allclose(pc.sigma[0][0], torch.full((8,), 1 / 8 ** 0.5)) and torch.equal(pc.sigma[0][1], before[1])

    # Fixed schedule without cap == legacy explicit cap=144 for R2s and IA
    for mode in ("R2s", "IA"):
        cfg = dict(
            release=["shape", "symmetry", "projection"],
            sigma={k: 0.2 for k in ["translation", "rotation", "torsion", "ring", "bend"]},
            optimizer="irprop-" if mode == "R2s" else "irprop+",
            search="irprop-" if mode == "R2s" else "irprop+",
        )
        cfg.update(score_only=True) if mode == "R2s" else cfg.update(switch=100)
        no_cap = FinishPhase([torch.zeros(2, 8)], [SimpleNamespace(n_joints=2, n_macro_joints=0)], 250, 100, cfg, 4, 1e-5)
        legacy = FinishPhase([torch.zeros(2, 8)], [SimpleNamespace(n_joints=2, n_macro_joints=0)], 250, 100, dict(cfg, cap=144), 4, 1e-5)
        for t in range(250):
            tick = torch.tensor([t]); loss = torch.tensor([float(t), float(-t)])
            g = [torch.full((2, 8), 10.0 if (t // 7) % 2 else -10.0)]
            no_cap.observe(tick, loss); legacy.observe(tick, loss)
            assert all(torch.equal(x, y) for x, y in zip(no_cap.update(tick, g, g, loss, None, 0.1), legacy.update(tick, g, g, loss, None, 0.1)))
            for f in ("active", "done", "count", "switched"):
                assert torch.equal(getattr(no_cap, f), getattr(legacy, f))

    with pytest.raises(ValueError, match="requires explicit cap"):
        FinishPhase([torch.zeros(2, 8)], [SimpleNamespace(n_joints=2, n_macro_joints=0)], 250, 100, dict(release=[], sigma={}), 4, 1e-5)


def test_phase_reference_replay_parity(stats, one_thread__gpu_fast_path, monkeypatch):
    from jacobik.ik import solver_replay as SR

    for code, optimizer, sigma_source in (
        ("MELLUN", "adam", "thermal"),
        ("AHIGOQ", "irprop+", "thermal"),
        ("PIDXOT", "irprop-", "thermal"),
        ("MELLUN", "adam", "curvature"),
    ):
        config = dict(
            release=["shape", "symmetry", "projection"],
            cap=8,
            sigma=dict(translation=0.0585, rotation=0.08609, torsion=0.2402, ring=0.2402, bend=0.08609),
            optimizer=optimizer,
            sigma_source=sigma_source,
        )
        monkeypatch.setattr(IKM, "FINISH_PHASE", config)
        monkeypatch.setattr(SR, "FINISH_PHASE", config)
        monkeypatch.setattr(IKM, "BEND_RELEASE", [0.1, 8])
        monkeypatch.setattr(SR, "BEND_RELEASE", [0.1, 8])
        monkeypatch.setattr(SR, "FORCE_ON_CPU", False)
        run = lambda: _builder(code).solve(
            stats=stats, seed=11, angle_bends=True, embed="etkdg", beam_width=4, n_coord_samples=2,
            n_conformers=4, n_ik_steps=16, ik_config=IKSolverConfig(cull_at=(), cull_keep=()),
        )
        a = run()
        monkeypatch.setattr(SR, "FORCE_ON_CPU", True)
        monkeypatch.setattr(SR, "ENABLED", True)
        monkeypatch.setattr(SR, "FAST_FK", False)
        b = run()
        assert SR.LAST["mode"] == "eager"
        assert np.array_equal([c.score for c in a.conformers], [c.score for c in b.conformers])
        assert np.array_equal([[x.position for x in c.complex.atoms] for c in a.conformers],
                              [[x.position for x in c.complex.atoms] for c in b.conformers])


@pytest.mark.xfail(strict=False, reason="known, owner 2026-11-11")
def test_phase_cuda_capture(stats, one_thread__gpu_fast_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from jacobik.ik import solver_replay as SR
    from tools.jdev import multiseed

    for mode in ("adam", "R2s"):
        config = dict(
            release=["shape", "symmetry", "projection"], cap=16,
            sigma=dict(translation=0.0585, rotation=0.08609, torsion=0.2402, ring=0.2402, bend=0.08609),
            optimizer="irprop-" if mode == "R2s" else "adam",
        )
        if mode == "R2s":
            config.update(search="irprop-", score_only=True)
        monkeypatch.setattr(IKM, "FINISH_PHASE", config)
        monkeypatch.setattr(SR, "FINISH_PHASE", config)
        monkeypatch.setattr(SR, "ENABLED", True)
        monkeypatch.setattr(SR, "FAST_FK", True)

        def run(replay):
            monkeypatch.setattr(SR.ReplaySolver, "replay", replay)
            r = _builder("MELLUN").solve(
                stats=stats, seed=11, device="cuda", angle_bends=True, embed="etkdg",
                beam_width=4, n_coord_samples=2, n_conformers=8, n_ik_steps=40,
                ik_config=IKSolverConfig(cull_at=(), cull_keep=()),
            )
            return np.array([c.score for c in r.conformers]), np.array([[a.position for a in c.complex.atoms] for c in r.conformers])

        with multiseed.safe_cuda_eigh_chunking():
            a, b = run(False), run(True)
        assert SR.LAST["mode"] == "replay" and SR.LAST.get("error") is None, SR.LAST
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_multiarm_runner_polish_removed_values_and_cli_reject(monkeypatch, tmp_path, capsys):
    from jacobik.core import profile as P
    from jacobik.ik import solver_replay as SR
    from tools.jgen.analyze_g3a2_cal import clustered
    from tools.jgen.bench.run import _has_active_scales, _set_finish_flags, main as bench_main

    assert _has_active_scales({"polish": "irprop"}, []) and _has_active_scales({"optimizer": "irprop_minus"}, [])
    assert _has_active_scales({"finish_phase": {"score_only": True}}, []) and _has_active_scales({}, [])
    assert not _has_active_scales({"finish_phase": None}, [])
    for module in (IKM, SR):
        for attr in ("FINISH_PHASE", "POLISH", "OPTIMIZER"):
            monkeypatch.setattr(module, attr, getattr(module, attr), raising=False)
    _set_finish_flags({"polish": "irprop_minus"})
    assert IKM.POLISH == SR.POLISH == "irprop_minus"
    _set_finish_flags({})
    assert IKM.POLISH is SR.POLISH is None
    cfg = dict(score_only=True, search="irprop-", optimizer="irprop-")
    _set_finish_flags({"finish_phase": cfg})
    assert IKM.FINISH_PHASE == SR.FINISH_PHASE == cfg
    _set_finish_flags({})
    assert IKM.FINISH_PHASE == SR.FINISH_PHASE == P.get("finish_phase")

    # Removed polish/optimizer values fail loudly at import
    for flag, value in (("polish", "fire"), ("polish", "adam_small"), ("optimizer", "adam_score")):
        env = dict(os.environ, JACOBIK_PROFILE_OVERRIDES=json.dumps({flag: value}))
        p = subprocess.run([sys.executable, "-c", "import jacobik.ik.ik_solver"], capture_output=True, text=True, env=env)
        assert p.returncode != 0 and "removed" in p.stderr, (flag, value)

    rows = [{"code": "a", "stratum": "N", "d": 0.0} for _ in range(3)] + [{"code": "b", "stratum": "O", "d": 10.0}]
    stat = clustered(rows, "d")
    assert stat["parents"] == 2 and stat["units"] == 4 and stat["mean"] == 5.0 and stat["ci95"] == [0.0, 10.0]
    assert clustered(rows, "d", "N")["ci95"] == [0.0, 0.0]

    with pytest.raises(SystemExit) as exc:
        bench_main(["--arm", "production", "--out", str(tmp_path), "--panel", "named", "--cpu-identity", "--device", "cuda"])
    assert exc.value.code == 2 and "--cpu-identity requires --device cpu" in capsys.readouterr().err


def test_legacy_polish_states_and_errors(stats, legacy_polish_context, monkeypatch, caplog):
    from jacobik.ik import polish, solver_replay as SR

    for method in ("irprop", "irprop_minus"):
        start = [torch.tensor([[0.0], [1.0]], dtype=torch.float64)]
        score = lambda th: -(th[0][:, 0] - 1).square()
        best, val = optimise(start, score, method=method)
        assert val[0] > score(start)[0] and val[1] == 0 and best[0][1, 0] == 1 and torch.equal(val, score(best))

    st_p = IRpropState([torch.zeros(2, 2, dtype=torch.float64)], torch.tensor([0.0, 0.0]))
    g = [torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)]
    assert torch.equal(st_p.update(g, torch.tensor([0.0, 0.0]), 0)[0], torch.full((2, 2), 0.01, dtype=torch.float64))
    rev = [torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)]
    assert torch.allclose(st_p.update(rev, torch.tensor([-1.0, 1.0]), 1)[0], torch.tensor([[-0.01, 0.012], [0.0, 0.012]], dtype=torch.float64))
    assert torch.equal(st_p.prev_grad[0][:, 0], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(st_p.update(rev, torch.tensor([-1.0, 1.0]), 2)[0][:, 0], torch.full((2,), -0.005, dtype=torch.float64))

    st_m = IRpropState([torch.zeros(2, 2, dtype=torch.float64)], torch.tensor([0.0, 0.0]), backtracking=False)
    st_m.update([torch.ones(2, 2, dtype=torch.float64)], torch.tensor([0.0, 0.0]), 0)
    assert torch.equal(st_m.update(rev, torch.tensor([-1.0, 1.0]), 1)[0][:, 0], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(st_m.update(rev, torch.tensor([-1.0, 1.0]), 2)[0][:, 0], torch.full((2,), -0.005, dtype=torch.float64))

    with pytest.raises(ValueError):
        optimise([torch.zeros(1, 1)], lambda th: -th[0].sum(-1), method="fire")

    monkeypatch.setenv("JACOBIK_NUM_WORKERS", "1")
    old_t = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        # Whole irprop_minus skips Adam loop
        visited, phases = [], []
        monkeypatch.setattr(IKM, "OPTIMIZER", "irprop_minus")
        monkeypatch.setattr(IKM, "POLISH", None)
        monkeypatch.setattr(IKM.JointIKSolver, "trace_hook", lambda *a: visited.append(True))
        orig_opt = polish.optimise
        monkeypatch.setattr(polish, "optimise", lambda *a, **kw: (phases.append((kw["method"], kw["steps"], kw["delta0"])), orig_opt(*a, **kw))[1])
        r_m = _builder("MELLUN").solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", beam_width=2, n_coord_samples=2, n_conformers=4, n_ik_steps=2, device="cpu")
        assert r_m.conformers and not visited and phases and all(p == ("irprop_minus", 2, None) for p in phases)

        # Polish failure keeps delivered baseline + default layout CPU replay parity
        monkeypatch.setattr(IKM, "OPTIMIZER", "adam")
        caplog.set_level(logging.WARNING, logger="jacobik.ik.ik_solver")
        base_fn = lambda: _builder("MELLUN").solve(
            stats=stats, seed=11, angle_bends=True, embed="etkdg", beam_width=4, n_coord_samples=2, n_conformers=4, n_ik_steps=12, device="cpu",
            ik_config=IKSolverConfig(cull_at=(), cull_keep=()),
        )
        monkeypatch.setattr(IKM, "POLISH", None)
        baseline = base_fn()
        monkeypatch.setattr(polish, "optimise", orig_opt)

        # Parity with method in [None, 'irprop'] and capped worker layout
        class _LimExec(ThreadPoolExecutor):
            def __init__(self, max_workers=None, **kw):
                super().__init__(max_workers=min(2, max_workers or 2), **kw)
        monkeypatch.setattr(IKM, "ThreadPoolExecutor", _LimExec)
        for method in (None, "irprop"):
            monkeypatch.setattr(IKM, "POLISH", method)
            monkeypatch.setattr(SR, "POLISH", method)
            monkeypatch.setattr(SR, "FORCE_ON_CPU", False)
            b0 = base_fn()
            monkeypatch.setattr(SR, "FORCE_ON_CPU", True)
            monkeypatch.setattr(SR, "ENABLED", True)
            monkeypatch.setattr(SR, "FAST_FK", False)
            r0 = base_fn()
            assert [c.score for c in b0.conformers] == [c.score for c in r0.conformers]

        monkeypatch.setattr(SR, "FORCE_ON_CPU", False)
        monkeypatch.setattr(polish, "optimise", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("injected optimizer failure")))
        monkeypatch.setattr(IKM, "POLISH", "irprop")
        fb = base_fn()
        assert [c.score for c in fb.conformers] == [c.score for c in baseline.conformers]
        assert "keeping delivered baseline" in caplog.text
    finally:
        torch.set_num_threads(old_t)


def test_chunk_traces_and_ez_error_handling():
    rng = np.random.default_rng(0)
    a = {"residual": rng.standard_normal((4, 3)).astype(np.float32), "score": rng.standard_normal((4, 3))}
    b = {"residual": rng.standard_normal((4, 2)).astype(np.float32), "score": rng.standard_normal((4, 2))}
    out = merge_chunk_traces([a, b])
    assert all(np.array_equal(out[k], np.concatenate([a[k], b[k]], axis=1)) for k in a)

    a2 = {"residual": np.arange(6, dtype=np.float32).reshape(3, 2), "score": np.ones((3, 2))}
    b2 = {"residual": np.zeros((5, 3), dtype=np.float32), "score": np.zeros((5, 3))}
    out2 = merge_chunk_traces([a2, b2])
    assert out2["residual"].shape == (5, 5) and np.isnan(out2["residual"][3:, :2]).all() and np.array_equal(out2["residual"][:, 2:], b2["residual"])

    seen = []
    cb = _CB.__new__(_CB)
    cb.ligands = [SimpleNamespace(ez_fix=None)]
    def _solve(**kw):
        seen.append(cb.ligands[0].ez_fix)
        if len(seen) == 1:
            raise ValueError("internal trace shape mismatch")
        raise _LigandEmbedFailure("ETKDG failed for seed 0")
    cb.solve = _solve
    res = cb._solve_ez_variants([["a"], ["b"]], {})
    assert seen == ["a", "b"] and res.conformers == [] and "internal trace shape mismatch" in res.failure and "ETKDG failed" in res.failure


def test_complex_fk_matches_reference(stats, one_thread__gpu_fast_path, monkeypatch):
    from jacobik.ik.fk_compose import ComplexFK

    cap, orig = {}, IKM.JointIKSolver.solve
    def grab(self, theta_inits, X_ref, *args, **kw):
        cap["x"] = (self.fks, [t.detach().clone() for t in theta_inits], X_ref.detach().clone())
        return orig(self, theta_inits, X_ref, *args, **kw)
    monkeypatch.setattr(IKM.JointIKSolver, "solve", grab)

    for code in ("MELLUN", "AHIGOQ", "PIDXOT", "HAFZOG", "CUWMEP", "FMNBCU"):
        cap.clear()
        _builder(code).solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", stereo_enum=False, beam_width=2, n_coord_samples=2, n_conformers=2, n_ik_steps=1)
        fks, th0, X_ref = cap["x"]
        cfk = ComplexFK(fks)
        assert cfk.supported and sum(cfk.K_of.values()) > 0
        g = torch.Generator().manual_seed(0)
        for trial in range(3):
            ths = [t.clone() for t in th0]
            if trial:
                for f, t in zip(fks, ths):
                    J = f.n_joints
                    t[:, 6:6 + J] = (torch.rand(t.shape[0], J, generator=g) - 0.5) * 2 * math.pi
                    t[:, :6] += torch.randn(t.shape[0], 6, generator=g) * 0.3
                    if t.shape[1] > 6 + J:
                        t[:, 6 + J:] = (torch.rand(t.shape[0], t.shape[1] - 6 - J, generator=g) - 0.5) * 0.4
            ths = [t.requires_grad_(True) for t in ths]
            w = torch.randn(3, generator=g)
            Xa, Xb = cfk.reference(ths, X_ref), cfk.apply(ths, X_ref)
            ga, gb = torch.autograd.grad((Xa * w).sum(), ths, allow_unused=True), torch.autograd.grad((Xb * w).sum(), ths, allow_unused=True)
            assert float((Xa - Xb).detach().abs().max()) < 1e-3, code
            for x, y in zip(ga, gb):
                if x is not None:
                    assert float((x - y).abs().max() / x.abs().max().clamp(min=1e-6)) < 1e-2, code


def test_replay_loop_matches_package_loop(stats, one_thread__gpu_fast_path, monkeypatch):
    """Referenced by build_replay_solver.py: rewritten step loop matches JointIKSolver bit-for-bit."""
    from jacobik.ik import solver_replay as SR

    def compare_trace(left, right):
        if isinstance(left, dict):
            assert isinstance(right, dict) and set(left) == set(right)
            for k in left:
                if k not in {"t_prep_s", "t_solve_s"}:
                    compare_trace(left[k], right[k])
        elif isinstance(left, (list, tuple)):
            assert type(left) is type(right) and len(left) == len(right)
            for a, b in zip(left, right):
                compare_trace(a, b)
        else:
            np.testing.assert_array_equal(left, right)

    for code in ("MELLUN", "AHIGOQ"):
        monkeypatch.setattr(SR, "FORCE_ON_CPU", False)
        run = lambda: _builder(code).solve(
            stats=stats, seed=11, angle_bends=True, embed="etkdg", beam_width=4, n_coord_samples=2, n_conformers=4, n_ik_steps=20,
            ik_config=IKSolverConfig(cull_at=(), cull_keep=()),
        )
        r0 = run()
        monkeypatch.setattr(SR, "FORCE_ON_CPU", True)
        monkeypatch.setattr(SR, "ENABLED", True)
        monkeypatch.setattr(SR, "FAST_FK", False)
        r1 = run()
        assert SR.LAST.get("mode") == "eager" and SR.LAST.get("fast_fk") is False
        assert np.array_equal([c.score for c in r0.conformers], [c.score for c in r1.conformers])
        assert np.array_equal([[a.position for a in c.complex.atoms] for c in r0.conformers],
                              [[a.position for a in c.complex.atoms] for c in r1.conformers])
        compare_trace(r0.solve_trace, r1.solve_trace)


def test_fast_path_and_batched_assembly_cpu(stats, one_thread__gpu_fast_path, monkeypatch):
    from jacobik.builder import engine as E
    from jacobik.ik import solver_replay as SR

    monkeypatch.setattr(SR, "ENABLED", True)
    monkeypatch.setattr(SR, "FORCE_ON_CPU", False)
    assert SR.solver_class("cpu") is IKM.JointIKSolver and SR.solver_class(torch.device("cuda")) is SR.ReplaySolver
    monkeypatch.setattr(SR, "ENABLED", False)
    assert SR.solver_class("cuda") is IKM.JointIKSolver
    monkeypatch.setattr(SR, "ENABLED", True)

    # One-pass FK on CPU
    monkeypatch.setattr(SR, "FORCE_ON_CPU", True)
    monkeypatch.setattr(SR, "FAST_FK", True)
    res = _builder("MELLUN").solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", beam_width=8, n_coord_samples=4, n_conformers=8, n_ik_steps=16)
    assert SR.LAST.get("fast_fk") is True and sum(c.mirror_of is None for c in res.conformers) == 8
    X = np.array([[a.position for a in c.complex.atoms] for c in res.conformers])
    assert np.isfinite(X).all() and all(np.isfinite(c.score) for c in res.conformers)

    # Batched assembly vs per-conformer assembly
    monkeypatch.setattr(SR, "FORCE_ON_CPU", False)
    monkeypatch.setattr(E, "OUT_MIRROR", False)
    for code, n_conf in (("MELLUN", 4), ("AHIGOQ", None)):
        def run(flag):
            monkeypatch.setattr(E, "BATCHED_ASSEMBLY", flag)
            r = _builder(code).solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", beam_width=4, n_coord_samples=2, n_conformers=n_conf, n_ik_steps=12)
            return ([c.trial_idx for c in r.conformers], [c.score for c in r.conformers],
                    np.array([[a.position for a in c.complex.atoms] for c in r.conformers]))
        rows0, sc0, X0 = run(False)
        rows1, sc1, X1 = run(True)
        assert rows0 == rows1 and sc0 == sc1 and np.abs(X0 - X1).max() < 1e-4


def test_dedup_kernels_and_select_distinct(stats, one_thread__gpu_fast_path):
    from jacobik.ik.dedup import StructureDeduper as D, select_distinct

    r = _builder("MELLUN").solve(stats=stats, seed=11, angle_bends=True, embed="etkdg", n_conformers=8, n_ik_steps=10)
    mol = r.conformers[0].complex
    Xm = torch.tensor(np.array([[a.position for a in c.complex.atoms] for c in r.conformers]), dtype=torch.float32)
    sc = torch.tensor([c.score for c in r.conformers])
    assert select_distinct(mol, Xm, sc, 4, torch.device("cpu")) == D(mol).select(Xm, sc, 4)

    g = torch.Generator().manual_seed(0)
    for n, near in ((5, False), (12, True), (31, False), (4, True)):
        A = torch.randn(n, 3, dtype=torch.float64, generator=g); A = A - A.mean(0)
        B = (A[None] + 0.05 * torch.randn(100, n, 3, dtype=torch.float64, generator=g)) if near else 2 * torch.randn(100, n, 3, dtype=torch.float64, generator=g)
        B = B - B.mean(1, keepdim=True)
        H = torch.einsum("ni,rnj->rij", A, B)
        U, S, Vt = torch.linalg.svd(H)
        ref = S[:, 0] + S[:, 1] + torch.sign(torch.det(U) * torch.det(Vt)) * S[:, 2]
        assert float((ref - D._max_overlap(H, 0.5 * ((A ** 2).sum() + (B ** 2).sum((-1, -2))))).abs().max()) < 1e-9
        d = object.__new__(D)
        maps = torch.stack([torch.cat([torch.roll(torch.arange(min(4, n)), k), torch.arange(min(4, n), n)]) for k in range(min(4, n))])
        d.maps, d.inv_maps = maps, torch.argsort(maps, dim=1)
        brute = torch.stack([D._rms_fixed_map(A, B[:, m]) for m in maps]).min(0).values
        assert float((d._full(A, B) - brute).abs().max()) < 1e-7

    g1 = torch.Generator().manual_seed(1)
    for n, scale in ((6, 0.1), (14, 0.3), (25, 0.6), (40, 1.0)):
        A = torch.randn(n, 3, dtype=torch.float64, generator=g1); A = A - A.mean(0)
        k = min(6, n)
        blk = [torch.roll(torch.arange(k), s) for s in range(k)]
        blk += [b.flip(0) for b in blk]
        maps = torch.stack([torch.cat([b, torch.arange(k, n)]) for b in blk])
        d = object.__new__(D); d.maps, d.inv_maps = maps, torch.argsort(maps, dim=1)
        B = A[maps[torch.randint(len(maps), (60,), generator=g1)]] + scale * torch.randn(60, n, 3, dtype=torch.float64, generator=g1)
        Q, _ = torch.linalg.qr(torch.randn(60, 3, 3, dtype=torch.float64, generator=g1))
        Q = Q * torch.sign(torch.det(Q))[:, None, None]
        B = torch.einsum("rij,rnj->rni", Q, B); B = B - B.mean(1, keepdim=True)
        full = d._full(A, B)
        ths = torch.cat([full.quantile(torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)), full[:2] * (1 + 1e-9), full[:2] * (1 - 1e-9)])
        for th in ths.tolist():
            assert [d._full_any(A, B[r:r + 1], th) for r in range(len(B))] == (full <= th).tolist()
            assert d._full_any(A, B[:30], th) == bool((full[:30] <= th).any())


def test_seeds_in_one_pass_match_separate_solves(one_thread__gpu_fast_path, monkeypatch, tmp_path):
    import importlib
    from jacobik.core.env import PRODUCTION_DB_NAME
    from jacobik.ik import solver_replay as SR
    from tests._paths import DATA_ROOT, require
    from tools.jdev import bench

    rp = importlib.import_module("tests.regression.run_panel")
    db = require(Path(os.environ.get("JACOBIK_DB") or DATA_ROOT / "db" / PRODUCTION_DB_NAME), "constraints DB")
    specs = bench.matrix_specs({"t": {"angle_bends": True, "embed": "etkdg", "beam_width": 2, "n_coord_samples": 2, "n_ik_steps": 10}}, [11, 12], db, tmp_path, "n48", cases=["KUYNEX"], ensemble=True)
    ctxs = {int(run["ctx"]["solve_kw"]["seed"]): run["ctx"] for run, _ in specs}
    assert len({rp._seed_group_key(run) for run, _ in specs}) == 1
    monkeypatch.setattr(SR, "FORCE_ON_CPU", True)
    single = {s: rp._worker("KUYNEX", str(db), c, None, "cpu", True) for s, c in ctxs.items()}
    multi = rp._worker_seeds("KUYNEX", str(db), ctxs, "cpu", True)
    for s in ctxs:
        assert multi[s]["status"] == "ok" and multi[s]["seeds_in_pass"] == 2
        assert multi[s]["score"] == single[s]["score"] and multi[s]["scores_all"] == single[s]["scores_all"]
        assert np.array_equal(multi[s]["_structures"][1][1], single[s]["_structures"][1][1])
    monkeypatch.setattr(SR, "FORCE_ON_CPU", False)
    fallback = rp._worker_seeds("KUYNEX", str(db), ctxs, "cpu", True)
    assert all(r["status"] == "ok" and "seeds_in_pass" not in r and r.get("seeds_in_pass_skipped") for r in fallback.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_dedup_and_cuda_replay_e2e(stats):
    from jacobik.ik import solver_replay as SR
    from jacobik.ik.dedup import StructureDeduper

    r = _builder("AHIGOQ").solve(stats=stats, seed=11, angle_bends=True, n_conformers=None, n_ik_steps=20)
    mol = r.conformers[0].complex
    X = torch.tensor(np.array([[a.position for a in c.complex.atoms] for c in r.conformers]), dtype=torch.float32)
    X = torch.cat([X, X + 0.01 * torch.randn(X.shape, generator=torch.Generator().manual_seed(0))])
    sc = torch.linspace(1.0, 0.0, X.shape[0])
    assert StructureDeduper(mol).select(X, sc, None) == StructureDeduper(mol, device=torch.device("cuda")).select(X.cuda(), sc.cuda(), None)

    res = _builder("MELLUN").solve(stats=stats, seed=11, angle_bends=True, n_conformers=8, device="cuda")
    assert SR.LAST.get("mode") == "replay", SR.LAST.get("error")
    assert SR.LAST.get("replayed", 0) >= 190 and 8 <= len(res.conformers) <= 16


def test_replay_generator_and_kabsch_init(stats, tmp_path, monkeypatch):
    import inspect
    import jacobik.builder.engine as E
    from jacobik.ik import dedup as dd
    from jacobik.ik.kabsch import compensate_fk_pivot, kabsch_init

    # Replay generator reproduces solver_replay.py byte-for-byte
    out_file = tmp_path / "solver_replay.py"
    subprocess.run([sys.executable, str(REPO / "dev" / "tools" / "jdev" / "build_replay_solver.py"), "--out", str(out_file)], check=True, capture_output=True)
    assert out_file.read_bytes() == (REPO / "src" / "jacobik" / "ik" / "solver_replay.py").read_bytes()

    # Initial donors start on M-D targets (no origin/pivot offset regression)
    seen, orig = {}, IKM.JointIKSolver.solve
    def spy(self, theta_inits, X_ref, donor_pos_idxs, *a, **k):
        X = IKM._ref_rows(X_ref, theta_inits[0].shape[0])
        for fk, th in zip(self.fks, theta_inits):
            X = fk.apply(th, X)
        seen["d"] = (self.ge.donor_positions(X) - X[:, self.ge.metal_pos_idx][:, None, :]).norm(dim=-1)
        return orig(self, theta_inits, X_ref, donor_pos_idxs, *a, **k)
    monkeypatch.setattr(IKM.JointIKSolver, "solve", spy)
    ComplexBuilder("Fe", "octahedral", charge=3, spin=6, ligands=[LigandInput("O", donors=[[0]]) for _ in range(6)]).solve(stats=stats, seed=11, n_ik_steps=1)
    d = seen["d"].numpy()
    assert d.max() < 3.3 and 1.8 < d.mean() < 2.6

    # output_window, START_KABSCH flags at import, pivot compensation, _output_options, solve(dedup_rmsd=...)
    assert E.output_window(np.array([1.0, 0.0, -500.0, np.nan, -np.inf]), 1).tolist() == [True, True, False, False, False]
    env = dict(os.environ, JACOBIK_PROFILE_OVERRIDES=json.dumps({"start_kabsch": True, "start_kabsch_pivot": True, "start_kabsch_diversity": 0.0}))
    fp = subprocess.check_output([sys.executable, "-c", 'from jacobik.builder import engine; from jacobik.core import profile; assert engine.START_KABSCH and engine.START_KABSCH_PIVOT; assert engine.START_KABSCH_DIVERSITY == 0.0; print(profile.snapshot()["fingerprint"])'], env=env, text=True).strip()
    assert fp != "5f74b14b57ed"

    original = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    seeded = original[None] + torch.tensor([[[2.0, 1.0, 0.0]]])
    target = torch.bmm(seeded, axisangle_to_matrix(torch.tensor([[0.4, -0.7, 0.9]])).transpose(1, 2)) + 5.0
    ge, fk = SimpleNamespace(metal_pos_idx=0, donor_positions=lambda x: x), SimpleNamespace(_donor_pos_t=torch.arange(3))
    base = kabsch_init(ge, seeded, target, torch.arange(3))
    apply = lambda th: torch.bmm(seeded - original.mean(0), axisangle_to_matrix(th[:, 3:]).transpose(1, 2)) + original.mean(0) + th[:, None, :3]
    assert (apply(kabsch_init(ge, original, target, torch.arange(3))) - target).norm(dim=-1).max() > 0.5
    assert (apply(base) - target).norm(dim=-1).max() > 0.5
    torch.testing.assert_close(apply(compensate_fk_pivot(base, fk, original, seeded, ge, torch.arange(3))), target, atol=2e-6, rtol=0)

    sig = inspect.signature(E.ComplexBuilder.solve).parameters
    for name in ("out_per_class", "out_per_class_k", "out_per_class_window", "out_window", "dedup_rmsd", "n_conformers"):
        assert sig[name].default is None
    with E._output_options(out_window=50.0, dedup_rmsd=1.5, out_per_class_k=3, out_per_class_window=10.0):
        assert (E.OUT_WINDOW, dd.DEDUP_RMSD, E.OUT_PER_CLASS_K, E.OUT_PER_CLASS_WINDOW) == (50.0, 1.5, 3, 10.0)
    for bad in ({"out_window": -1.0}, {"dedup_rmsd": 0.0}, {"out_per_class_k": 0}, {"out_per_class_window": 0.0}):
        with pytest.raises(ValueError):
            with E._output_options(**bad):
                pass
    gly = lambda: LigandInput("NCC(=O)[O-]", donors=[[0], [4]])
    kw = dict(stats=stats, n_ik_steps=8, fast_embed=True, beam_width=2, seed=3, n_conformers=4)
    assert len(ComplexBuilder("Pt", "square_plan", 0, 1, [gly(), gly()]).solve(dedup_rmsd=50.0, **kw).conformers) == 1
    assert dd.DEDUP_RMSD == 0.5 and E.OUT_WINDOW == 200.0


def test_macrocycle_and_small_ring_joints(stats, monkeypatch):
    import jacobik.topology.decompose as D

    cyclen = "Cc1cc(C)c(NC(=O)CN2CCN(C)CCN(C)CCN(C)CC2)c(C)c1"
    builder_mc = ComplexBuilder("Ni", "octahedral", 1, 1, [LigandInput(cyclen, donors=[[8], [10], [21], [17], [13]]), LigandInput("[Cl-]", donors=[[0]])])
    mol, _, _ = builder_mc._build_shell(True, False)
    for lig in mol.ligands:
        decompose_ligand(mol, lig)
    fk = ForwardKinematics(mol, next(l for l in mol.ligands if l.macrocycles), torch.device("cpu"))
    X = torch.tensor(np.stack([a.position for a in mol.atoms]), dtype=torch.float32)
    th = torch.zeros(1, 6 + fk.n_joints)
    for k in range(fk.n_joints):
        th[0, 6 + k] = dihedral(*[X[i][None] for i in fk._quad_idx[k]])[0]

    assert fk.n_macro_joints == 9 and len(fk.closure_pairs) >= 3
    torch.manual_seed(0)
    th2 = th.repeat(8, 1); c0 = fk._macro_col0
    th2[:, c0:] += (torch.rand(8, th2.shape[1] - c0) - 0.5) * 0.6
    thp, ss = fk.closure_project(th2, X, iters=12)
    Xc = fk.apply(thp, X)
    u, v = fk.closure_pairs[0]
    nbr_v, nbr_u = [b for a, b in fk.closure_pairs[1:] if a == u], [b for a, b in fk.closure_pairs[1:] if a == v]
    ang = lambda P, a, b, c: torch.rad2deg(torch.arccos(((P[..., a, :] - P[..., b, :]) * (P[..., c, :] - P[..., b, :])).sum(-1) / (P[..., a, :] - P[..., b, :]).norm(dim=-1) / (P[..., c, :] - P[..., b, :]).norm(dim=-1)))
    ok = ss < 1e-4
    assert nbr_u and nbr_v and int(ok.sum()) >= 6
    assert all(float((ang(Xc, x, u, v) - ang(X, x, u, v)).abs()[ok].max()) < 1.0 for x in nbr_u)
    assert all(float((ang(Xc, u, v, y) - ang(X, u, v, y)).abs()[ok].max()) < 1.0 for y in nbr_v)

    d0 = fk.seam_d0(X); r0, J = fk.closure_residual_jacobian(fk.apply(th, X), d0)
    assert float(r0.abs().max()) < 1e-4
    for k in range(fk.n_macro_joints):
        t2 = th.clone(); t2[0, c0 + k] += 1e-3
        assert torch.allclose(J[0, :, k], (fk.closure_residual_jacobian(fk.apply(t2, X), d0)[0] - r0)[0] / 1e-3, atol=2e-2)

    torch.manual_seed(0)
    t3 = th.clone(); t3[0, c0:] += 0.4 * torch.randn(fk.n_macro_joints)
    assert float(fk.closure_project(t3, X, iters=6)[1]) < 1e-6

    torch.manual_seed(1)
    t_f, plain, K = th.clone(), th.clone(), fk.n_macro_joints
    for _ in range(20):
        s_m = 0.1 * torch.randn(1, K)
        t_f[:, c0:] = t_f[:, c0:] - fk.closure_fused_step(fk.apply(t_f, X), d0, s_m)
        plain[:, c0:] = plain[:, c0:] - 0.1 * torch.randn(1, K)
    assert float(fk.closure_residual_jacobian(fk.apply(t_f, X), d0)[0].norm()) < 0.1 and float(fk.closure_residual_jacobian(fk.apply(plain, X), d0)[0].norm()) > 1.0

    assert builder_mc.solve(stats=stats, beam_width=2, n_coord_samples=2, n_ik_steps=10, seed=1, macrocycle_pool=0, embed="etkdg").conformers

    # Small donor rings (RING_JOINTS) & rigid aromatic ring
    _decomposed = lambda m, g, ls: ([decompose_ligand(mm, l) for mm in [ComplexBuilder(m, g, 0, 1, ls)._build_shell(True, False)[0]] for l in mm.ligands], mm)[1]
    def _dec(m, g, ls):
        mm, _, _ = ComplexBuilder(m, g, 0, 1, ls)._build_shell(True, False)
        for l in mm.ligands:
            decompose_ligand(mm, l)
        return mm

    tri = ("Ti", "octahedral", [LigandInput("CN1CN(C)CN(C)C1", donors=[[1], [3], [6]])] + [LigandInput("[Cl-]", donors=[[0]])] * 3)
    cod = ("Rh", "square_plan", [LigandInput("C1=CCCC=CCC1", donors=[[4, 5], [0, 1]]), LigandInput("[Cl-]", donors=[[0]]), LigandInput("[Cl-]", donors=[[0]])])
    ben = ("Cr", "tetrahedral", [LigandInput("c1ccccc1", donors=[[0, 1, 2, 3, 4, 5]])] + [LigandInput("[C-]#[O+]", donors=[[0]])] * 3)

    monkeypatch.setattr(D, "RING_JOINTS", True)
    assert not any(l.macrocycles for l in _dec(*ben).ligands)
    for case, size in ((tri, 6), (cod, 8)):
        monkeypatch.setattr(D, "RING_JOINTS", False)
        assert not any(l.macrocycles for l in _dec(*case).ligands)
        monkeypatch.setattr(D, "RING_JOINTS", True)
        mc_mol = _dec(*case)
        lig_r = next(l for l in mc_mol.ligands if l.macrocycles)
        assert len(lig_r.macrocycles[0]) == size
        fk_r = ForwardKinematics(mc_mol, lig_r, torch.device("cpu"))
        order = {(min(b.i, b.j), max(b.i, b.j)): float(b.order) for b in mc_mol.bonds}
        assert fk_r.n_macro_joints >= 2 and all(abs(order[(min(j, k), max(j, k))] - 1.0) < 0.25 for _, j, k, _ in fk_r.macro_quads_atom)
        X_r = torch.tensor(np.stack([a.position for a in mc_mol.atoms]), dtype=torch.float32)
        th_r = torch.zeros(1, 6 + fk_r.n_joints)
        for k in range(fk_r.n_joints):
            th_r[0, 6 + k] = dihedral(*[X_r[i][None] for i in fk_r._quad_idx[k]])[0]
        torch.manual_seed(0)
        th_r[0, fk_r._macro_col0:fk_r._macro_col0 + fk_r.n_macro_joints] += 0.3 * torch.randn(fk_r.n_macro_joints)
        assert float(fk_r.closure_project(th_r, X_r, iters=8)[1]) < 1e-5
