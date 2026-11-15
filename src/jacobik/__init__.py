from jacobik.builder.cli import main
from jacobik.builder.engine import ComplexBuilder, LigandInput, debug_complex, generate, ligand, ligands
from jacobik.builder.result import ConformerResult, EnsembleResult, geometry_sites
from jacobik.core.env import JacobikEnvironmentError, check_environment, default_db_path

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
