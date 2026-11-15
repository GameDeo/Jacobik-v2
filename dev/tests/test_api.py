"""Public API, environment check, CLI, dedup options, and packaging tests."""
from __future__ import annotations

import ast
import json
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

import jacobik
from jacobik.builder.cli import build_parser
from jacobik.builder.result import _output_options
from jacobik.core import env as _env
from jacobik.ik import dedup as _dd

_REPO = Path(__file__).resolve().parents[2]


def test_public_exports_packaging_and_check_environment(monkeypatch, tmp_path):
    expected = {
        "ComplexBuilder",
        "ConformerResult",
        "EnsembleResult",
        "JacobikEnvironmentError",
        "LigandInput",
        "check_environment",
        "default_db_path",
        "generate",
        "geometry_sites",
        "main",
    }
    assert expected.issubset(set(jacobik.__all__))
    for name in expected:
        assert hasattr(jacobik, name), f"Missing public export jacobik.{name}"
    assert sorted(p.name for p in (_REPO / "src" / "jacobik").glob("*.py")) == ["__init__.py"]

    # pyproject declares every external import in src/jacobik and src/fit
    data = tomllib.loads((_REPO / "pyproject.toml").read_text())
    assert data["project"]["scripts"]["jacobik"] == "jacobik:main"
    import_to_pkg = {"sklearn": "scikit-learn", "mace": "mace-torch"}
    declared = {s.split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip().lower() for s in data["project"]["dependencies"]}
    optional = {
        s.split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip().lower()
        for grp in data["project"].get("optional-dependencies", {}).values()
        for s in grp
    }
    stdlib = sys.stdlib_module_names
    src_files = list((_REPO / "src" / "jacobik").rglob("*.py")) + list((_REPO / "src" / "fit").rglob("*.py"))
    missing = []
    for path in src_files:
        rel = path.relative_to(_REPO).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                tops = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                tops = [node.module.split(".")[0]]
            else:
                tops = []
            for top in tops:
                if top in stdlib or top in ("jacobik", "fit"):
                    continue
                if rel == "src/fit/build_db.py" and top == "tools":
                    continue
                dist = import_to_pkg.get(top, top).lower()
                if dist not in declared and not (rel == "src/jacobik/core/energy.py" and dist in optional):
                    missing.append((rel, top, dist))
    assert not missing, f"Undeclared imports in src/: {missing}"

    # Environment checks
    info = jacobik.check_environment()
    assert info["db"] and Path(info["db"]).is_file() and info["xtb"]

    monkeypatch.setenv("JACOBIK_XTB", str(tmp_path / "missing_xtb_binary"))
    with pytest.raises(jacobik.JacobikEnvironmentError, match="xtb"):
        jacobik.check_environment()
    with pytest.raises(jacobik.JacobikEnvironmentError, match="xtb"):
        jacobik.generate("Pt", "square_plan", 2, 1, [("N", [[0]])] * 4)

    ok = jacobik.check_environment(require_xtb=False)
    assert ok["xtb"] is None and ok["db"]

    monkeypatch.setenv("JACOBIK_DB", str(tmp_path / "missing_db.npz"))
    with pytest.raises(jacobik.JacobikEnvironmentError, match="JACOBIK_DB"):
        jacobik.check_environment(require_xtb=False)

    monkeypatch.delenv("JACOBIK_DB", raising=False)
    monkeypatch.setattr(_env, "default_db_path", lambda: None)
    with pytest.raises(jacobik.JacobikEnvironmentError, match="Constraints DB"):
        jacobik.check_environment(require_xtb=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(jacobik.JacobikEnvironmentError, match="CUDA"):
        jacobik.check_environment(require_xtb=False, require_db=False, device="cuda")


def test_dedup_options_max_dev_and_generate_smoke(stats, tmp_path):
    assert _dd.DEDUP_RMSD == pytest.approx(0.5)
    assert _dd.DEDUP_MAX_DEV == pytest.approx(0.0)

    with pytest.raises(ValueError, match="dedup_rmsd"):
        with _output_options(dedup_rmsd=0.0):
            pass
    with pytest.raises(ValueError, match="dedup_max_dev"):
        with _output_options(dedup_max_dev=-0.1):
            pass
    try:
        with _output_options(dedup_rmsd=0.25, dedup_max_dev=0.4):
            assert _dd.DEDUP_RMSD == pytest.approx(0.25) and _dd.DEDUP_MAX_DEV == pytest.approx(0.4)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _dd.DEDUP_RMSD == pytest.approx(0.5) and _dd.DEDUP_MAX_DEV == pytest.approx(0.0)

    # Functional dedup max_dev guard
    class _Atom:
        def __init__(self, sym):
            self.symbol = sym

    class _Bond:
        def __init__(self, i, j):
            self.i, self.j = i, j

    class _FakeMol:
        atoms = [_Atom("Fe")] + [_Atom("C")] * 19
        bonds = [_Bond(i, i + 1) for i in range(19)]
        metal_idx = 0

    X = torch.zeros((2, 20, 3), dtype=torch.float64)
    X[0, :, 0] = torch.arange(20, dtype=torch.float64)
    X[1] = X[0].clone()
    X[1, -1, 1] = 0.9
    scores = torch.tensor([1.0, 0.5], dtype=torch.float64)
    dd = _dd.StructureDeduper(_FakeMol())
    assert dd.select(X, scores, k=None, th=0.5, max_dev_th=0.0) == [0]
    assert dd.select(X, scores, k=None, th=0.5, max_dev_th=0.3) == [0, 1]

    # Smoke generate() + ComplexBuilder.solve() + write_xyz() + gfn2 check
    res = jacobik.generate(
        metal="Pt", geometry="square_plan", charge=2, spin=1,
        ligands=[("N", [[0]])] * 4, stats=stats, seed=11,
        n_ik_steps=10, beam_width=2, n_coord_samples=2, n_conformers=2,
        device="cpu", embed="mmff", dedup_rmsd=0.35, dedup_max_dev=0.8,
    )
    assert res.conformers and all(np.isfinite([a.position for a in c.complex.atoms]).all() for c in res.conformers)
    assert _dd.DEDUP_RMSD == pytest.approx(0.5) and _dd.DEDUP_MAX_DEV == pytest.approx(0.0)

    out_xyz = tmp_path / "pt_nh3_4.xyz"
    assert out_xyz in res.write_xyz(out_xyz) and out_xyz.is_file()

    ligs_obj = [jacobik.LigandInput("N", donors=[[0]]) for _ in range(4)]
    res2 = jacobik.ComplexBuilder("Pt", "square_plan", 2, 1, ligs_obj).solve(
        stats=stats, seed=11, n_ik_steps=10, beam_width=2, n_coord_samples=2,
        n_conformers=2, device="cpu", embed="mmff", dedup_rmsd=0.35, dedup_max_dev=0.8,
    )
    assert res2.conformers

    if shutil.which("xtb"):
        res_gfn2 = jacobik.generate(
            metal="Pt", geometry="square_plan", charge=2, spin=1,
            ligands=ligs_obj, stats=stats, seed=11,
            n_ik_steps=6, beam_width=2, n_coord_samples=2, n_conformers=1, device="cpu",
        )
        assert res_gfn2.conformers


def test_cli_entry_points(capsys):
    assert jacobik.main(["fingerprint"]) == 0
    assert capsys.readouterr().out.strip() == jacobik.profile.snapshot()["fingerprint"]

    assert jacobik.main(["check"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert Path(info["db"]).is_file()

    parser = build_parser()
    solve_help = parser._subparsers._group_actions[0].choices["solve"].format_help()
    collapsed = "".join(solve_help.split())
    assert _env.PRODUCTION_DB_NAME in collapsed and "data/db/constraints.npz" not in collapsed

    with pytest.raises(SystemExit) as exc:
        jacobik.main(["--help"])
    assert exc.value.code == 0
