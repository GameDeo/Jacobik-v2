# Jacobik

Jacobik is a Python package providing a conformer ensemble engine for transition-metal complexes.

## Installation

You can install `jacobik` locally using `pip`:

```bash
git clone <repository_url>
cd jacobik
pip install -e .
```

To install development dependencies (for code formatting and linting):
```bash
pip install -e .[dev]
```

## Basic Usage

Below is a simple demonstration of how to setup, run, and visualize a complex using the `jacobik` module.

```python
from jacobik import ComplexBuilder, LigandInput, Visualizer

metal = "Pt"
geometry = "square_plan"
charge = 2
spin = 1

ligands = [
    LigandInput("c1ccc(cc1)P(c2ccccc2)Nc3ccccn3", donors=[[6], [19]]),
    LigandInput("c1ccc(cc1)P(c2ccccc2)Nc3ccccn3", donors=[[6], [19]])
]

# 1. Initialize the complex builder
builder = ComplexBuilder(
    metal=metal,
    geometry=geometry,
    charge=charge,
    spin=spin,
    ligands=ligands
)

# 2. Run the solver to generate valid conformers
result = builder.solve(
    constraints_db="src/jacobik/constraints.json",
    beam_width=8,
    n_ik_steps=30,
    n_coord_samples=4,
    convergence_tol=0.3,
    debug=False
)

# Check how many conformers successfully converged
n_conv = sum(1 for c in result.conformers if c.converged)
print(f"Solver completed: {n_conv} out of {len(result.conformers)} trajectories converged.")

# 3. Visualize the ensemble
vis = Visualizer(result, show_hydrogen=True)

# Interactive 3D view
vis.view_ensemble_structs()
```

## tmQM benchmark suite

The benchmark is deliberately outside the solver and does not alter its
optimisation loop. It reconstructs each tmQM entry from metal, ligand SMILES,
donor groups, charge, spin multiplicity, and classified coordination geometry;
runs the public `ComplexBuilder` API; and compares the resulting ensemble to the
reference structure.

Install the optional benchmark stack and fetch one split:

```bash
pip install -e ".[benchmark,dev]"
jacobik-benchmark fetch data/tmqm --splits X1
```

Run a quick, deterministic smoke benchmark:

```bash
jacobik-benchmark run data/tmqm --split X1 --limit 25 \
  --beam-width 16 --n-ik-steps 200 --seed 42 \
  --mace --mace-model polar-1-m --output benchmark.json
```

Run all three splits by invoking `run` once per split. MACE-POLAR-1 model
weights are downloaded and cached by `mace-torch` on first use.

The JSON output contains rank-1 and best-of-ensemble symmetry-corrected Kabsch
RMSD, coordination-sphere RMSD, ligand-local RMSD, atom-map coverage, metal–donor
distance error, donor–metal–donor angle error, retained-coordination fraction,
steric-clash diagnostics, convergence and timing data, and matched-level
MACE-POLAR-1 energies. Symmetry correction uses molecular graph automorphisms
and exchange of identical ligands, so equivalent phenyl flips and rotations do
not inflate RMSD. Reflection is never allowed.

Energy values are reported as `E(reconstructed) - E(tmQM geometry)` after both
single points are evaluated by the same MACE-POLAR-1 checkpoint. The tabulated
tmQM DFT absolute energy is not subtracted from MACE because the methods and
energy zeros differ. Use `polar-1-m` for routine screening and `polar-1-l` for a
more expensive confirmation run.
