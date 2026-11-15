"""
fit
===========
Modular per-quantity fitting framework for Jacobik constraint databases.

Quantities:
- q1: M-D bond distance distributions (GMM pooled, m_sub_early)
- q2: D-M-D cis / bite angle distributions (tmix_cos pooled)
- q3: D-M-D trans angle / donor approach angle distributions (gmm_cos)
- q4: L-D-M bend angle distributions (gmm)
- q5: Ligand torsion distributions (wells_folded)
- q6: Steric non-bonded exchange repulsive wall (halfgauss p5)
- qo: S_orient joint donor orientation distributions (mvgmm)
- q6p: Purely repulsive Q6 PMF tables and metal 1-3 core tables & walls
- bends: Ligand backbone angle bend stiffness constants and haptic lean walls
"""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "qo",
    "q6p",
    "bends",
]


def __getattr__(name):
    """Load quantity modules on demand so per-worker profile overrides apply."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
