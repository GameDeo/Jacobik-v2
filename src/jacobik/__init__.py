import os as _os

# cuBLAS needs a fixed workspace for deterministic results (the relaxer runs deterministic kernels on CUDA); it must be
# set before the first CUDA call, so it is set on import unless the caller chose one
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from jacobik.builder.cli import main  # noqa: E402
from jacobik.builder.engine import ComplexBuilder, LigandInput, debug_complex, generate, ligand, ligands  # noqa: E402
from jacobik.builder.result import ConformerResult, EnsembleResult, geometry_sites  # noqa: E402
from jacobik.core.env import JacobikEnvironmentError, check_environment, default_db_path  # noqa: E402

#: release version and build ("torch": portable PyTorch build; "gpu": CuPy + Triton build, Linux + NVIDIA)
__version__ = "1.0b0+gpu"
BUILD = "gpu"

__all__ = [
    "ComplexBuilder",
    "ConformerResult",
    "EnsembleResult",
    "JacobikEnvironmentError",
    "LigandInput",
    "check_environment",
    "debug_complex",
    "default_db_path",
    "generate",
    "geometry_sites",
    "ligand",
    "ligands",
    "main",
]


def __getattr__(name: str):
    if name == "profile":
        from jacobik.core import profile

        return profile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
