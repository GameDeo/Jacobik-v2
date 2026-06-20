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
