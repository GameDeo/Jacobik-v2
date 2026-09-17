"""
reproduce_edta_results.py
==========================
STANDALONE — not part of the jacobik package/build.

Exact reproduction of the EDTA-Co results discussed this session, using the
patched files (decompose.py, fk.py, scorer.py, ik_solver.py, engine.py,
stats.py) delivered alongside this script.

Why you were seeing different numbers than my terminal runs
-------------------------------------------------------------
Two independent causes, both fixed in the delivered files:

1. solve(seed=...) never actually seeded PyTorch's global RNG (engine.py
   2.5.0) -- it seeded an orphaned, never-used local torch.Generator()
   instead. Every unqualified torch.randn/torch.rand call in the pipeline
   (beam search, kabsch_init's diversify_orientation) was reading from
   whatever global RNG state happened to exist from code that ran BEFORE
   solve() -- different across notebooks/scripts/call order even for the
   same seed. Now fixed: torch.manual_seed(seed) is called directly.

2. Environment differences (torch/numpy/rdkit versions) can shift results
   at the margins even with identical seeds, since floating-point ops and
   default algorithm selection aren't always bit-identical across library
   versions. The versions used to produce every number quoted this session
   are printed by check_environment() below -- if yours differ, that's
   worth checking first if numbers still don't match exactly.

Version history
----------------
1.0.0 : initial release
"""

from __future__ import annotations

import sys


def check_environment() -> None:
    """Print the exact environment versions results in this session were produced with."""
    import numpy
    import rdkit
    import torch

    print("Expected (this session):")
    print("  python : 3.12.3")
    print("  torch  : 2.12.1+cu130")
    print("  numpy  : 2.4.4")
    print("  rdkit  : 2026.03.3")
    print()
    print("Yours:")
    print(f"  python : {sys.version.split()[0]}")
    print(f"  torch  : {torch.__version__}")
    print(f"  numpy  : {numpy.__version__}")
    print(f"  rdkit  : {rdkit.__version__}")


def reproduce(
    constraints_db_path: str,
    seed: int = 42,
    beam_width: int = 32,
    n_ik_steps: int = 200,
    lr: float = 0.1,
):
    """
    Run the exact EDTA-Co solve discussed this session and return the result.

    Parameters
    ----------
    constraints_db_path : path to a constraints.json / subset.json.
    seed                : same seed => bit-identical result now (engine.py
                           2.5.0). Every number quoted this session used
                           seed=0 or seed=42; both are reproducible now.
    beam_width           : W. Session numbers used 32 and 64 at different
                           points -- larger W keeps improving score
                           monotonically post-2.3.0 (was flat/worse before).
    n_ik_steps           : T.
    lr                   : optimizer learning rate.
    """
    import logging

    logging.basicConfig(level=logging.ERROR)

    from jacobik.engine import ComplexBuilder, LigandInput

    # The corrected anionic EDTA SMILES with donor atoms chosen consistently
    # (all four carboxylate donors are the true single-bonded anionic
    # oxygen, not a mix of anionic/carbonyl oxygens -- see decompose.py
    # 1.1.0/1.2.1 for why that mattered).
    smiles = "[O-]C(=O)CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]"
    donors = [[0], [4], [7], [11], [15], [19]]

    ligand = LigandInput(smiles, donors=donors)
    builder = ComplexBuilder(
        metal="Co", geometry="octahedral", charge=-1, spin=1, ligands=[ligand]
    )
    result = builder.solve(
        constraints_db=constraints_db_path,
        beam_width=beam_width,
        n_ik_steps=n_ik_steps,
        lr=lr,
        seed=seed,
        diversify_orientation=True,  # engine.py 2.3.0 -- off by default, on for every session result
    )
    return result


def summarize(result, constraints_db_path: str) -> None:
    """
    Print the same score / geometry / sterics breakdown used throughout
    this session.

    constraints_db_path : must be the SAME path passed to reproduce() --
                           needed again here to rebuild a Scorer for the
                           breakdown (the result object doesn't carry it).
    """
    import math

    import numpy as np
    import torch

    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from jacobik.stats.stats import StatsLookup, load_db

    best = result.best
    mol = best.complex
    print(f"best score: {best.score:.6f}   converged: {best.converged}")

    positions = {i: np.array(a.position) for i, a in enumerate(mol.atoms, start=1)}
    metal_pos = positions[mol.metal_idx]
    print("\ndonor distances to metal:")
    for d in sorted(mol.donor_labels.keys()):
        dist = np.linalg.norm(positions[d] - metal_pos)
        print(f"  atom#{d:3d} ({mol.atoms[d - 1].symbol})  {dist:.3f} A  {mol.donor_labels[d]}")

    donor_atoms = set(mol.donor_labels.keys())
    closest = min(
        np.linalg.norm(positions[a.idx] - metal_pos)
        for a in mol.atoms
        if a.idx != mol.metal_idx and a.idx not in donor_atoms and a.symbol != "H"
    )
    print(f"\nclosest non-donor heavy atom to metal: {closest:.3f} A")

    stats = StatsLookup(load_db(constraints_db_path))
    ge = GeometryExtractor(mol, torch.device("cpu"))
    scorer = Scorer(mol, ge, stats, device=torch.device("cpu"))
    X = torch.tensor(np.stack([a.position for a in mol.atoms]), dtype=torch.float32).unsqueeze(0)
    _, breakdown = scorer.score(X)
    print("\nscore breakdown:")
    for k, v in breakdown.items():
        print(f"  {k}: {v.item():.4f}")

    q2_angles = ge.q2_dmd_angles(X)[0]
    angs = [math.degrees(a.item()) for a in q2_angles]
    near90 = sum(1 for a in angs if 80 < a < 100)
    near180 = sum(1 for a in angs if a > 160)
    print(f"\nD-M-D angles: near-90 = {near90}/15, near-180 = {near180}/15 (ideal octahedron: 12/15, 3/15)")


def visualize(result) -> None:
    """
    Run the interactive visualizer for the given result (same as test_suite.ipynb).
    Intended for use in a Jupyter notebook environment.
    """
    from IPython.display import display
    from jacobik import Visualizer

    vis = Visualizer(result, show_hydrogen=True)
    
    display(vis.view_ensemble_structs())
    #display(vis.view_ensemble_stats())
    display(vis.view_ranking())
    #display(vis.view_overlay())
    display(vis.view_ensemble_heatmap())
    #display(vis.view_solve())
    #display(vis.geometry_mappings())