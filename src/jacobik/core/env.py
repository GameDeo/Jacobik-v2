"""Core runtime environment: constants, device selection, checkout/DB path resolution, and tmQM/lnQM input loaders."""
from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: gas constant times 298.15 K in kJ/mol (the energy unit of the bend and Q6 tables)
RT_KJ = 2.479
#: safe divisor / small-angle threshold of the torch kernels
EPS = 1e-7
LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def resolve_device(device: str | None = "auto") -> str:
    """Turn ``"auto"`` / ``"cpu"`` / ``"cuda"`` / ``"cuda:N"`` into a device string torch accepts.

    ``auto`` picks CUDA when this torch build can use it. An explicit CUDA request on a machine
    where torch cannot see a GPU raises here, with the reason, instead of failing later inside MACE
    (the usual cause is a CPU-only torch wheel: ``torch.version.cuda`` is then None).
    """
    import torch

    name = (device or "auto").strip().lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {device!r} requested but torch cannot use CUDA: torch {torch.__version__}, "
                f"built for CUDA {torch.version.cuda!r}, {torch.cuda.device_count()} device(s) visible. "
                "Install a CUDA build of torch in this environment (a '+cpu' wheel has no GPU support) "
                "or run with device='cpu'."
            )
        index = torch.device(name).index
        if index is not None and index >= torch.cuda.device_count():
            raise RuntimeError(f"device {device!r} requested but only {torch.cuda.device_count()} CUDA device(s) exist")
        return name
    if name != "cpu":
        raise ValueError(f"unknown device {device!r}; use 'auto', 'cpu', 'cuda' or 'cuda:N'")
    return "cpu"


def torch_device(device: Any = None):
    """``torch.device`` for a spec (a string, a ``torch.device`` or None); None means the CPU."""
    import torch

    return torch.device(device) if device is not None else torch.device("cpu")


# ---------------------------------------------------------------------------
# Paths & DB location
# ---------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Optional

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _find_repo_root() -> Optional[Path]:
    cand = _PACKAGE_DIR.parents[1]
    if (cand / "pyproject.toml").exists() and (cand / "src" / "jacobik").is_dir():
        return cand
    return None


#: the source checkout this file sits in; ``None`` for an installed package
REPO_ROOT: Optional[Path] = _find_repo_root()


def main_checkout(repo: Optional[Path] = None) -> Path:
    """The main checkout of the git repository ``repo`` belongs to (a worktree's ``--git-common-dir`` parent); ``repo``
    itself outside git. ``repo`` defaults to the source checkout this file sits in (the working directory for an installed
    package)."""
    import subprocess
    repo = Path(repo) if repo is not None else (REPO_ROOT if REPO_ROOT is not None else Path.cwd())
    try:
        common = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                                check=True).stdout.strip()
        return (Path(common) if Path(common).is_absolute() else (repo / common)).resolve().parent
    except Exception:
        return repo


def _default_data_root() -> Path:
    """<repo>/data if present, else the data/ of the main checkout (a git worktree has no data/ of its own). Read only."""
    base = REPO_ROOT if REPO_ROOT is not None else Path.cwd()
    local = base / "data"
    if local.exists() or REPO_ROOT is None:
        return local
    return main_checkout(REPO_ROOT) / "data"


def data_dir() -> Path:
    """The data root (``JACOBIK_DATA_ROOT`` overrides)."""
    return Path(os.environ.get("JACOBIK_DATA_ROOT", _default_data_root()))


def data_ref_dir() -> Path:
    """Raw inputs (tmQM, lnQM) of ``fit.build_db``."""
    return data_dir() / "ref"


def frame_v1_dir() -> Path:
    """The frozen fit frame, panel manifest and stored Step-2 pools (``data/refit/frame_v1``)."""
    return data_dir() / "refit" / "frame_v1"


#: file name of the production constraints DB (DS data fold-in, 2026-11-12; sha256 62c58d53...; embedded meta, no sidecar)
PRODUCTION_DB_NAME = "constraints_2026-11-12.npz"


def default_db_path() -> Optional[Path]:
    """The constraints DB to use when none is given: ``$JACOBIK_DB`` (if the file exists), else the production DB
    packaged with the library (``jacobik/data``), in ``./data/db``, or in the data root's ``db/``; ``None`` when none exists."""
    env = os.environ.get("JACOBIK_DB")
    if env and Path(env).exists():
        return Path(env)
    candidates = [_PACKAGE_DIR / "data" / PRODUCTION_DB_NAME, Path("data") / "db" / PRODUCTION_DB_NAME,
                  data_dir() / "db" / PRODUCTION_DB_NAME]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def _default_mace_model() -> Optional[Path]:
    env = os.environ.get("JACOBIK_MACE_MODEL")
    if env:
        return Path(env)
    base = REPO_ROOT if REPO_ROOT is not None else Path.cwd()
    local = base / "research" / "bench" / "tools" / "jdev" / "MACE-POLAR-1-S.model"
    return local if local.is_file() else None   # None: the official release file is downloaded to ~/.cache/mace on first use


#: a local MACE-POLAR-1-S model file (``JACOBIK_MACE_MODEL`` overrides); None when there is none (the weights are not part of the repository)
MACE_MODEL_PATH = _default_mace_model()


# ---------------------------------------------------------------------------
# Reference input loaders (tmQM / lnQM)
# ---------------------------------------------------------------------------
_SMILES_CACHE: Dict[str, Dict[str, Optional[str]]] = {}


class _Row:
    """The one field of a CSV row that ``_assemble_raw`` reads (SMILES)."""

    def __init__(self, smiles: Optional[str]):
        self.smiles = smiles


def tmqm_smiles(parent_dir: Path) -> Dict[str, Optional[str]]:
    """CSD_code -> SMILES. Reads two named columns of tmQM_y.csv; every other cell is discarded unparsed."""
    parent_dir = Path(parent_dir)
    key = str(parent_dir.resolve())
    if key not in _SMILES_CACHE:
        csv_path = parent_dir / "tmQM_y.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing SMILES file: {csv_path}")
        out: Dict[str, Optional[str]] = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            rdr = csv.reader(fh, delimiter=";")
            header = next(rdr)
            i_code, i_smi = header.index("CSD_code"), header.index("SMILES")
            for row in rdr:
                if row:
                    out[row[i_code].strip()] = (row[i_smi].strip() or None) if len(row) > i_smi else None
        _SMILES_CACHE[key] = out
    return _SMILES_CACHE[key]


def load_mol(code: str, shard: str, dataset: str = "tmQM", clean_root: Optional[Path] = None) -> Optional[Any]:
    """The TMComplex of one parent, built from its .xyz and .BO blocks and its SMILES (no QM properties attached).

    ``clean_root`` is the directory holding ``tmQM/`` and ``lnQM/`` (default: ``paths.data_ref_dir()``).
    """
    from jacobik.topology.mapping import assemble
    from fit.parser import _assemble_raw, _iter_bo_blocks, _iter_xyz_blocks

    clean_root = data_ref_dir() if clean_root is None else Path(clean_root)
    parent_dir = clean_root / ("lnQM" if dataset == "LNf" else "tmQM")
    xp = parent_dir / "xyz" / f"tmQM_{shard}.xyz"
    bp = parent_dir / "bo" / f"tmQM_{shard}.BO"
    if not xp.exists() or not bp.exists():
        return None
    smiles = tmqm_smiles(parent_dir)
    for (comment, atoms), (c, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
        if c == code:
            raw = _assemble_raw(comment, atoms, c, bonds, _Row(smiles.get(code)))
            mol = assemble(raw)
            # _attach_qm_properties is the only other place that sets smiles; it is not called (no QM columns)
            mol.smiles = smiles.get(code)
            return mol
    return None


def build_builder_and_inputs(mol: Any) -> Tuple[Optional[Any], Optional[List[Any]], Optional[str]]:
    """``(ComplexBuilder, ligand inputs, None)`` from a parsed parent, or ``(None, None, error)`` when it is unmappable."""
    try:
        from jacobik.builder.engine import ComplexBuilder
        from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol

        groups = _build_ligand_groups(mol, mol.smiles)
        lig_in = ligand_inputs_from_mol(mol, groups)
        if lig_in is None:
            return None, None, "unmappable_ligand_inputs (ligand_inputs_from_mol returned None)"
        builder = ComplexBuilder(
            metal=mol.metal_symbol,
            geometry=mol.coord_geom,
            charge=mol.charge,
            spin=mol.spin_state,
            ligands=lig_in,
        )
        return builder, lig_in, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


class JacobikEnvironmentError(RuntimeError):
    """Raised when required external binaries (`xtb`) or the constraints DB are unavailable."""


def check_environment(
    *,
    require_xtb: bool = True,
    require_db: bool = True,
    constraints_db: str | Path | None = None,
    device: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Validate runtime dependencies (`xtb`, constraints DB, optional compute device) and return resolved paths.

    Raises ``JacobikEnvironmentError`` when a required component is missing or invalid instead of letting
    downstream GFN2 embedding silently degrade to MMFF.
    """
    xtb_spec = os.environ.get("JACOBIK_XTB", "xtb").strip()
    xtb_resolved: Optional[str] = None
    if xtb_spec:
        xtb_resolved = shutil.which(xtb_spec)
        if xtb_resolved is None:
            cand = Path(xtb_spec)
            if cand.is_file() and os.access(cand, os.X_OK):
                xtb_resolved = str(cand.resolve())
    if require_xtb and not xtb_resolved:
        raise JacobikEnvironmentError(
            f"xtb executable {xtb_spec or 'xtb'!r} not found (install xtb 6.7.x on PATH or set JACOBIK_XTB)"
        )

    db_path: Optional[Path] = None
    if constraints_db is not None:
        db_path = Path(constraints_db)
        if require_db and not db_path.is_file():
            raise JacobikEnvironmentError(f"Constraints DB not found: {db_path}")
    elif os.environ.get("JACOBIK_DB", "").strip():
        db_path = Path(os.environ["JACOBIK_DB"].strip())
        if require_db and not db_path.is_file():
            raise JacobikEnvironmentError(f"JACOBIK_DB points to a missing file: {db_path}")
    else:
        db_path = default_db_path()
        if require_db and (db_path is None or not Path(db_path).is_file()):
            raise JacobikEnvironmentError(
                f"Constraints DB not found (pass constraints_db=..., set JACOBIK_DB, or ensure {PRODUCTION_DB_NAME} is in jacobik/data or data/db)"
            )

    dev_str: Optional[str] = None
    if device is not None:
        try:
            dev_str = resolve_device(device)
        except (RuntimeError, ValueError) as exc:
            raise JacobikEnvironmentError(str(exc)) from exc
    else:
        import torch

        dev_str = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "xtb": xtb_resolved,
        "db": str(Path(db_path).resolve()) if db_path is not None and Path(db_path).is_file() else None,
        "device": dev_str,
    }


def __getattr__(name: str) -> Any:
    if name == "DATA_DIR":
        return data_dir()
    if name == "DATA_REF_DIR":
        return data_ref_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
