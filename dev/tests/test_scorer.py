"""test_scorer: physical Q6 donor pairs, Q6 physical_all topology, torsion tail, and H4 rescore skip."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from jacobik.core import profile
from jacobik.core.types import Atom, Bond
from jacobik.ik.scorer import THERMAL_SD, _torsion_tail_logp
from jacobik.stats import steric as Q
from jacobik.topology import graph

REPO = Path(__file__).resolve().parents[2]


def _has_data():
    try:
        from tools.jgen import inputs
        from fit.parser import TMQMParser  # noqa: F401
        return inputs.load_mol("FEKKAJ", "X2s003", "X2") is not None
    except Exception:
        return False


def _fekkaj_subproc(overrides, check=True):
    env = {k: v for k, v in os.environ.items() if k not in ("JACOBIK_PROFILE", "JACOBIK_PROFILE_OVERRIDES")}
    env["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(overrides)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO / "dev"), str(REPO), str(REPO / "research" / "bench")])
    code = r"""
import json, torch
from tools.jgen import inputs
from jacobik.ik.geometry import GeometryExtractor
from jacobik.stats import steric as Q
class AnyTable(dict):
    def __bool__(self): return True
    def get(self, k, default=None): return {"w": [0.0] * 40, "r0": 1.0}
mol = inputs.load_mol("FEKKAJ", "X2s003", "X2")
ge = GeometryExtractor(mol, torch.device("cpu"))
sc = getattr(ge, "q6_donor_scale", None) or [None] * ge.n_q6
mu = torch.full((ge.n_q6,), 3.4); sg = torch.full((ge.n_q6,), 0.3)
rt = Q.build_runtime(mol, ge, AnyTable(), mu, sg, torch.device("cpu"))
print(json.dumps({"n_q6": ge.n_q6, "n_donor": ge.n_q6_donor, "n_rep": sum(rt["rep"].tolist()),
                  "scales": sorted({s for s in sc if s is not None}),
                  "lig_ok": all(ge._q6_a[p] != ge._q6_b[p] for p in range(ge.n_q6))}))
"""
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO), capture_output=True, text=True, check=check)


def test_q6_repulsion_params_energy_and_flags(monkeypatch):
    """Shifted onset, default scale=1, energy weight/disable, flags, and 1-4 topological reopening."""
    A, b, rm, rl = Q.rep_pair_params("Cl", "Cl", 0.7)
    base = Q.ALVAREZ_RADII["Cl"] * 2
    assert rm == pytest.approx(0.7 * base)
    P = [torch.tensor(v, dtype=torch.float64) for v in (A, b, rm, rl)]
    e = Q.rep_energy(torch.tensor([0.7 * base, 0.7 * base + 0.3], dtype=torch.float64), *P)
    assert float(e[0]) == pytest.approx(0.0, abs=1e-9) and float(e[1]) == 0.0
    assert float(Q.rep_energy(torch.tensor([0.7 * base - 0.2], dtype=torch.float64), *P)[0]) > 0.0

    assert Q.rep_pair_params("C", "N") == Q.rep_pair_params("C", "N", 1.0)
    d = torch.tensor([2.2], dtype=torch.float64)
    e_full = float(Q.rep_energy(d, *(torch.tensor(v) for v in Q.rep_pair_params("Cl", "Cl")))[0])
    e_half = float(Q.rep_energy(d, *(torch.tensor(v) for v in Q.rep_pair_params("Cl", "Cl", 0.7)))[0])
    assert e_full > e_half >= 0.0

    params = Q.rep_pair_params("Cl", "Cl")
    X = torch.tensor([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]], dtype=torch.float32)
    rt = {
        "T": torch.zeros((1, 2)), "tot": torch.tensor([0]), "has": torch.tensor([False]),
        "r0": torch.tensor([2.0]), "r_lo": 0.6, "dr": 0.02, "G": 2,
        "mu": torch.tensor([3.4]), "sigma": torch.tensor([0.3]),
        "rep_any": True, "rep": torch.tensor([True]), "rep_A": torch.tensor([params[0]]),
        "rep_b": torch.tensor([params[1]]), "rep_rm": torch.tensor([params[2]]),
        "rep_rl": torch.tensor([params[3]]), "enabled": torch.tensor([True]),
        "rep_weight": torch.tensor([0.25]),
    }
    idx = torch.tensor([0])
    full = Q.rep_energy(torch.tensor([2.5]), *(torch.tensor(v) for v in params))[0]
    assert float(Q.evaluate(rt, X, idx, idx + 1)[0]) == pytest.approx(float(full) * 0.25, rel=1e-6)
    rt["enabled"] = torch.tensor([False])
    rt["rep_weight"] = torch.tensor([0.0])
    assert float(Q.evaluate(rt, X, idx, idx + 1)[0]) == 0.0

    assert profile.get("q6_physical_all") is True and profile.get("q6_weight_14") == 0.0
    assert Q.Q6_PHYSICAL_ALL is True and Q.Q6_WEIGHT_14 == 0.0

    atoms = [Atom(i, sym, [float(i), 0.0, 0.0]) for i, sym in enumerate(("Fe", "C", "C", "C", "C"), 1)]
    bonds = [Bond(2, 3, 1.0), Bond(3, 4, 1.0), Bond(4, 5, 1.0), Bond(1, 2, 0.5)]
    monkeypatch.setattr(graph, "Q6_PHYSICAL_ALL", False)
    ex_off = graph.q6_topological_exclusions(atoms, bonds, 1, [2])
    monkeypatch.setattr(graph, "Q6_PHYSICAL_ALL", True)
    ex_on = graph.q6_topological_exclusions(atoms, bonds, 1, [2])
    assert (2, 3) in ex_off and (2, 4) in ex_off and (2, 5) in ex_off
    assert (2, 3) in ex_on and (2, 4) in ex_on and (2, 5) not in ex_on


@pytest.mark.skipif(not _has_data(), reason="tmQM clean data not available")
def test_fekkaj_donor_pairs_physical_all_and_immutable_switch():
    """FEKKAJ cross-ligand donor scales, physical_all 1-4/polar counts, and strict refusal of q6_donor_pairs=None."""
    from tools.jgen import inputs
    from jacobik.ik.geometry import GeometryExtractor

    class AnyTable(dict):
        def __bool__(self): return True
        def get(self, k, default=None): return {"w": [0.0] * 40, "r0": 1.0}

    mol = inputs.load_mol("FEKKAJ", "X2s003", "X2")
    ge = GeometryExtractor(mol, torch.device("cpu"))
    sc = getattr(ge, "q6_donor_scale", None) or [None] * ge.n_q6
    scales = sorted({s for s in sc if s is not None})
    assert ge.n_q6_donor > 0 and set(scales) <= {0.95, 0.90, 0.80} and 0.95 in scales
    assert all(ge._q6_a[p] != ge._q6_b[p] for p in range(ge.n_q6))
    mu, sg = torch.full((ge.n_q6,), 3.4), torch.full((ge.n_q6,), 0.3)
    on_rep = sum(Q.build_runtime(mol, ge, AnyTable(), mu, sg, torch.device("cpu"))["rep"].tolist())

    off = json.loads(_fekkaj_subproc({"q6_physical_all": False}).stdout.strip().splitlines()[-1])
    assert ge.n_q6 > off["n_q6"] and on_rep > off["n_rep"]

    bad = _fekkaj_subproc({"q6_donor_pairs": None}, check=False)
    assert bad.returncode != 0


def test_torsion_continuation_curvature_and_periodicity():
    mu = torch.tensor([[0.0]], dtype=torch.float64)
    k = torch.tensor([[8.0]], dtype=torch.float64)
    ln = torch.zeros_like(k)
    x = torch.tensor([[0.02]], dtype=torch.float64, requires_grad=True)
    actual = _torsion_tail_logp(x, mu, k, ln).sum()
    expected = (k * torch.cos(x)).sum()
    ga = torch.autograd.grad(actual, x, create_graph=True)[0]
    ge = torch.autograd.grad(expected, x, create_graph=True)[0]
    assert torch.equal(actual, expected) and torch.equal(ga, ge)
    assert torch.equal(torch.autograd.grad(ga.sum(), x)[0], torch.autograd.grad(ge.sum(), x)[0])

    far = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(_torsion_tail_logp(far, mu, k, ln).sum(), far, create_graph=True)[0]
    h = torch.autograd.grad(g.sum(), far)[0]
    assert torch.allclose(h, torch.full_like(h, -1 / THERMAL_SD["Q5"] ** 2))
    assert torch.allclose(_torsion_tail_logp(far, mu, k, ln), _torsion_tail_logp(far + 2 * math.pi, mu, k, ln))


def test_rescore_equals_solver_score(stats, monkeypatch):
    """H4: At production defaults, ranking_score equals solver score and the duplicate rescore pass is skipped."""
    from jacobik import ComplexBuilder, LigandInput
    from jacobik.ik import ik_solver as IK
    from jacobik.ik import scorer as SC

    assert SC.RANK_AGG == "sum" and SC.SCORE_AGG == "sum" and not IK._RANK_FLOOR and SC.Q1_TAIL_SIGMA == 0.0
    monkeypatch.setenv("JACOBIK_NUM_WORKERS", "1")
    calls, captured = [], {}
    orig_rank, orig_score = SC.Scorer.ranking_score, SC.Scorer.score

    def spy_score(self, X, q3_mode_override=None, donor_pos=None, row_groups=None):
        captured["scorer"], captured["X"], captured["rg"] = self, X.detach(), row_groups
        return orig_score(self, X, q3_mode_override=q3_mode_override, donor_pos=donor_pos, row_groups=row_groups)

    def spy_rank(self, X, row_groups=None):
        calls.append(X.shape[0])
        return orig_rank(self, X, row_groups=row_groups)

    monkeypatch.setattr(SC.Scorer, "score", spy_score)
    monkeypatch.setattr(SC.Scorer, "ranking_score", spy_rank)
    builder = ComplexBuilder("Fe", "octahedral", charge=2, spin=1,
                             ligands=[LigandInput(smiles="NCCN", donors=[[0], [3]]) for _ in range(3)])
    res = builder.solve(stats=stats, seed=11, beam_width=4, n_ik_steps=15, n_coord_samples=2, embed="etkdg")
    assert res.conformers and calls == []
    sc, X, rg = captured["scorer"], captured["X"], captured["rg"]
    with torch.no_grad():
        s_fit, breakdown = orig_score(sc, X, row_groups=rg)
        s_rank = orig_rank(sc, X, row_groups=rg)
    assert torch.equal(s_fit, s_rank)
    assert torch.allclose(sum(breakdown.values()), s_rank, atol=1e-5, rtol=1e-5)

    monkeypatch.setattr(IK, "_RANK_FLOOR", True)
    res_rf = builder.solve(stats=stats, seed=11, beam_width=4, n_ik_steps=15, n_coord_samples=2, embed="etkdg")
    assert res_rf.conformers and len(calls) >= 1
