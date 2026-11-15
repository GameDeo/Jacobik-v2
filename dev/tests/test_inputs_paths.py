"""jacobik.core.env: data-root resolution, tmQM loaders/custody, import hygiene, and packaged default DB."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

import jacobik.core.env as jp
from jacobik.core.env import load_mol, tmqm_smiles


def test_data_root_and_tmqm_loaders(monkeypatch, tmp_path):
    """Data root resolution, worktree fallback, cal parent load, missing CSV errors, and no-QM-column rule."""
    # Worktree fallback & cal parent
    monkeypatch.delenv("JACOBIK_DATA_ROOT", raising=False)
    root = jp.data_dir()
    if root.exists():
        assert (root / "ref").exists() and (root / "db").exists()
        assert jp.DATA_REF_DIR == root / "ref"
        assert jp.frame_v1_dir() == root / "refit" / "frame_v1"
    if (jp.data_ref_dir() / "tmQM").exists():
        mol = load_mol("IYUKOC", "X3s000", "tmQM")
        assert mol is not None and mol.metal_symbol == "Ni" and mol.charge == -1 and mol.smiles

    # JACOBIK_DATA_ROOT override & JACOBIK_DB override/fallback
    custom = tmp_path / "custom_data_dir"
    monkeypatch.setenv("JACOBIK_DATA_ROOT", str(custom))
    assert jp.data_dir() == custom and jp.DATA_DIR == custom and jp.DATA_REF_DIR == custom / "ref"

    monkeypatch.delenv("JACOBIK_DB", raising=False)
    pkg_db = (Path(jp.__file__).resolve().parent.parent / "data" / jp.PRODUCTION_DB_NAME).resolve()
    assert pkg_db.is_file() and jp.default_db_path() == pkg_db
    db = tmp_path / "x.npz"
    db.write_bytes(b"")
    monkeypatch.setenv("JACOBIK_DB", str(db))
    assert jp.default_db_path() == db
    monkeypatch.setenv("JACOBIK_DB", str(tmp_path / "absent.npz"))
    assert jp.default_db_path() == pkg_db
    monkeypatch.setattr(jp, "_PACKAGE_DIR", tmp_path / "nopkg")
    repo_db = (jp.REPO_ROOT / "data" / "db" / jp.PRODUCTION_DB_NAME) if jp.REPO_ROOT else None
    if repo_db is not None and repo_db.is_file():
        assert jp.default_db_path() == repo_db.resolve()
    monkeypatch.setattr(jp, "REPO_ROOT", None)
    monkeypatch.chdir(tmp_path)
    assert jp.default_db_path() is None

    # Missing tmQM_y.csv raises cleanly and QM property columns are never parsed (poisoned cells ignored)
    with pytest.raises(FileNotFoundError, match="tmQM_y.csv"):
        tmqm_smiles(tmp_path / "empty")
    dataset_dir = tmp_path / "tmQM"
    (dataset_dir / "xyz").mkdir(parents=True)
    (dataset_dir / "bo").mkdir(parents=True)
    (dataset_dir / "xyz" / "tmQM_00.xyz").write_text("1\nCSD_code = DUMMY01\nC 0.0 0.0 0.0\n")
    (dataset_dir / "bo" / "tmQM_00.BO").write_text("DUMMY01\n0\n")
    with pytest.raises(FileNotFoundError, match="tmQM_y.csv"):
        load_mol("DUMMY01", "00", "tmQM", clean_root=tmp_path)

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    with open(csv_dir / "tmQM_y.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["CSD_code", "Electronic_E", "SMILES", "Dispersion_E"])
        w.writerow(["AAAAAA", "POISON-not-a-float", "CC", "POISON"])
        w.writerow(["BBBBBB", "x", "", "y"])
    assert tmqm_smiles(csv_dir) == {"AAAAAA": "CC", "BBBBBB": None}


def test_package_imports_without_research_or_tools():
    """``import jacobik`` and ``import fit`` need neither ``research`` nor ``tools`` on the path (no outside-src imports)."""
    repo = Path(jp.REPO_ROOT or ".")
    code = (
        "import sys\n"
        f"sys.path[:] = [{str(repo / 'src')!r}] + [p for p in sys.path if 'research' not in p and p not in ('', '.') "
        f"and {str(repo)!r} != p]\n"
        "import jacobik, jacobik.core.env, jacobik.core.container, jacobik.core.profile, jacobik.stats.stats\n"
        "import fit, fit.q6p, fit.build_db\n"
        "assert not any(m == 'research' or m.startswith('research.') or m == 'tools' or m.startswith('tools.') "
        "for m in sys.modules), [m for m in sys.modules if m.split('.')[0] in ('research', 'tools')]\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd="/")
    assert p.returncode == 0, p.stderr[-2000:]


def test_packaged_db_shipped_and_default_in_isolated_cwd(monkeypatch, tmp_path):
    """The production .npz ships under jacobik/data alongside shannon_radii.json and is the default execution DB."""
    import tomllib

    repo = Path(jp.REPO_ROOT or ".")
    pkg_data = repo / "src" / "jacobik" / "data"
    assert (pkg_data / "shannon_radii.json").is_file()
    pkg_db = (pkg_data / jp.PRODUCTION_DB_NAME).resolve()
    assert pkg_db.is_file()

    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    pkg_globs = pyproject["tool"]["setuptools"]["package-data"]["jacobik"]
    assert "data/*.json" in pkg_globs and "data/*.npz" in pkg_globs

    monkeypatch.delenv("JACOBIK_DB", raising=False)
    monkeypatch.setenv("JACOBIK_DATA_ROOT", str(tmp_path / "nodata"))
    monkeypatch.chdir(tmp_path)

    # Even if ./data/db exists locally in cwd, packaged DB takes precedence unless overridden via JACOBIK_DB or arg
    local_db = tmp_path / "data" / "db" / jp.PRODUCTION_DB_NAME
    local_db.parent.mkdir(parents=True, exist_ok=True)
    local_db.write_bytes(b"shadow")
    assert jp.default_db_path() == pkg_db

    env_info = jp.check_environment(require_xtb=False, require_db=True)
    assert Path(env_info["db"]).resolve() == pkg_db
    from jacobik.builder.cli import main as cli_main
    assert cli_main(["check", "--no-xtb"]) == 0

    # Spy stats.load_db so we verify BOTH DB contents and generate() using the packaged DB in a single load
    import jacobik
    from jacobik.stats import stats as ST
    orig_load = ST.load_db
    loaded = []

    def spy_load(path=None, *a, **kw):
        cdb = orig_load(path, *a, **kw)
        loaded.append((path, sum(cdb.sizes().values())))
        return cdb

    monkeypatch.setattr(ST, "load_db", spy_load)
    res = jacobik.generate(
        metal="Pt", geometry="square_plan", charge=2, spin=1,
        ligands=[jacobik.LigandInput("N", donors=[[0]])] * 4,
        embed="etkdg", n_ik_steps=2, beam_width=2, n_coord_samples=2, n_conformers=1,
        device="cpu", seed=11, check_env=False,
    )
    assert len(res.conformers) >= 1
    assert loaded and Path(loaded[0][0]).resolve() == pkg_db and loaded[0][1] > 400000

    alt_db = tmp_path / "explicit.npz"
    alt_db.write_bytes(b"explicit")
    explicit_info = jp.check_environment(require_xtb=False, require_db=True, constraints_db=alt_db)
    assert Path(explicit_info["db"]).resolve() == alt_db.resolve()
