"""runkit: shared hardware use and orchestration for jacobik runs (xtb, MACE, local CPU, Colab).

Importing this package needs no GPU, no xtb and no network. torch, rdkit, mace and the jacobik package are
imported lazily inside the functions that need them. Nothing here is imported by src/jacobik, so the production
fingerprint cannot change. See README.md.
"""
__version__ = "0.2.0"
