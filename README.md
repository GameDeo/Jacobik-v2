<p align="center">
  <img src="docs/banner.png" alt="Jacobik — Statistically informed, configuration-complete metal conformer generation" width="100%">
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-F59E0B?style=flat-square" alt="MIT License"></a>
  <a href="PERFORMANCE.md"><img src="https://img.shields.io/badge/version-1.0a-0EA5E9?style=flat-square" alt="Version 1.0a"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#output--options">Output &amp; Options</a> ·
  <a href="#repository">Repository</a> ·
  <a href="PERFORMANCE.md">Benchmarks</a> ·
  <a href="ARCHITECTURE.md">Architecture</a> ·
  <a href="#license">License</a>
</p>

**Jacobik** generates configuration-complete 3D conformer ensembles for transition-metal and lanthanide complexes from a metal, formal charge, spin, coordination geometry, and ligand SMILES + donor atoms:

1. **Enumerate configurations** — donor-to-site assignments, stereocentres (including metal-binding-induced chirality), and `E`/`Z` double bonds.
2. **Joint batched inverse kinematics** — optimises ligand torsions, ring puckers, and backbone bond-angle bends together (CPU or CUDA).
3. **Statistical crystal ranking** — scores structures with log-likelihoods fitted to `tmQM` and `lnQM`, with no force field in the solve loop.

> [!NOTE]
> Output structures are unrelaxed. Relax or re-rank them with an energy model (e.g. GFN2-xTB or MACE) before quantitative energetic comparison across classes.

---

## Install

**pixi (recommended)** — pins Python 3.11, PyTorch with CUDA, RDKit, and `xtb`:

```bash
pixi install -e linux
export PATH="$PWD/.pixi/envs/linux/bin:$PATH"
```

**pip + conda-forge** — `xtb` (`6.7.1 <= version < 7`) is required on `PATH` (or `JACOBIK_XTB`) for GFN2 ligand refinement:

```bash
conda install -c conda-forge "xtb>=6.7.1,<7"
pip install -r requirements.txt && pip install -e .
```

| Requirement | Role & Default |
|---|---|
| `xtb` on `PATH` | Mandatory for GFN2 ligand embed refinement (missing `xtb` falls back to invalid MMFF embeds). Verify with `jacobik check`. |
| Statistics DB | `src/jacobik/data/constraints_2026-11-12.npz` (~28 MB, self-describing) ships packaged and loads automatically. Override via `constraints_db=`, `--db`, or `JACOBIK_DB`, or rebuild with `python -m fit.build_db`. |
| Hardware / Optional | Runs on CPU or CUDA (`device="cuda"` / `--device auto`). Energy re-ranking uses optional `.[mace]` extras (`requirements.txt`). |

---

## Quickstart

### Python API

Use library aliases/binding modes (`bpy*3`, `dmso@kappa1_O`), unpackable debug complexes (`DEBUG_COMPLEX_CBTYCO`, `DEBUG_COMPLEX_OJIMIF`), or explicit `LigandInput(smiles, donors)` definitions:

```python
import jacobik
from jacobik import ComplexBuilder, LigandInput
from interface.ligand_library import DEBUG_COMPLEX_CBTYCO, ligand

# Library aliases, binding modes, or predefined debug complexes
res = jacobik.generate("Fe", "octahedral", 2, 1, ligands=["bpy*3"], seed=11)
res = jacobik.generate(*DEBUG_COMPLEX_CBTYCO)
res = jacobik.generate("Pt", "square_plan", 0, 1, ligands=["en", "dmso@kappa1_O", ligand("isothiocyanate")])

# Arbitrary SMILES: inner lists define coordination points (e.g. hexadentate EDTA, haptic Cp [[0,1,2,3,4]])
edta = LigandInput("[O-]C(=O)CN(CC(=O)[O-])CCN(CC(=O)[O-])CC(=O)[O-]", donors=[[0], [4], [8], [11], [15], [19]])
res = ComplexBuilder("Co", "octahedral", charge=-1, spin=1, ligands=[edta]).solve(seed=11, device="cpu")

res.summary()
res.best.write_xyz("co_edta_top.xyz")
```

Supported geometries: `linear`, `trigonal`, `t_shaped`, `square_plan`, `tetrahedral`, `see_saw`, `square_pyr`, `trig_bipyr`, `octahedral`, `trig_pris`, `pent_bipyr`, `square_anti`, `tricap_pris`, and `other`. Pass `LigandInput(..., sites=[...])` to pin donors to explicit site indices (`jacobik.geometry_sites(geom)`).

### Command Line

Installed as `jacobik` (also runnable directly via `python interface/jacobik_cli.py`):

```bash
jacobik check                                     # verify xtb binary, packaged DB, and compute device
jacobik fingerprint                               # print active production profile hash
jacobik ligands                                   # list curated ligands, binding modes, and debug complexes
jacobik ligands dmso                              # inspect modes and aliases for a ligand

jacobik solve --metal Fe --geometry octahedral --charge 2 --spin 1 --ligand "bpy*3" --out fe_bpy3.xyz --by-class
jacobik solve --debug-complex CBTYCO --out-dir ./output --n-conformers 3
jacobik solve --dataset ./data_custom --split user --out-dir ./output --n-conformers 3
```

Interactive notebooks: [`interface/run_solver.ipynb`](interface/run_solver.ipynb) and [`interface/prepare_data.ipynb`](interface/prepare_data.ipynb).

---

## Output & Options

A **class** groups structures sharing one configuration (donor-site assignment, stereocentres, `E`/`Z` geometry); conformers within a class are distinct 3D conformations.

```python
for cls in result.classes():                     # C1, C2, ... ranked by best score
    print(cls["class_id"], cls["geometry"], cls["stereo"], cls["n_rows"], cls["delta_nats"])

top_rows = result.by_class()[result.classes()[0]["class_label"]]
result.write_xyz("out/", by_class=True)          # out/C1.xyz, out/C2.xyz, ... (rank/score in comment line)
records = result.to_records()                    # plain dicts: rank, score, delta_nats, class_id, descriptors
```

- `delta_nats` measures log-likelihood gap below the top-ranked row (it orders rows as solved, **not** relative energy).
- Enantiomers of achiral-ligand configurations are completed by reflection (`n_reflected` in `classes()`).

| `ComplexBuilder.solve` Option | Default | Effect |
|---|---|---|
| `n_conformers` | `max(beam_width, n_coord_samples)` (16) | Maximum conformers returned (`--n-conformers`). |
| `beam_width` | `16` | Torsion seeds kept per ligand in beam search. |
| `n_ik_steps` | `200` | Joint inverse-kinematics refinement steps. |
| `n_coord_samples` | `8` | Coordination-sphere distance samples. |
| `embed` | `"gfn2"` | Free-ligand embed method: `"gfn2"`, `"mmff"`, or `"etkdg"` (`--fast-embed`). |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` (CLI `--device auto`). |
| `seed`, `embed_seed` | `None`, internal | Solver seed and ligand-embed seed (`solve_multi_embed` varies embeddings). |
| `angle_bends` | `True` | Backbone bond-angle bend DOFs on chelate rings. |
| `stereo_enum`, `out_per_class` | profile | Enumerate stereo classes; keep up to `out_per_class_k` (`8`) rows per class. |
| `use_symmetry_enumeration`, `embed_aware_dedup` | `True` | Symmetry-distinct site enumeration merged only by true embedded ligand symmetry. |
| `constraints_db`, `stats` | packaged DB | Custom `.npz` path or pre-loaded `StatsLookup`. |

Output deduplication keeps structures with heavy-atom symmetric RMSD above **0.5 A** (`dedup_rmsd`) within a **200 nat** window (`out_window`). Override profile settings before import via `JACOBIK_PROFILE_OVERRIDES='{"dedup_rmsd": 0.3}'` and audit provenance with `jacobik.profile.snapshot()`.

---

## Repository

| Path | Contents |
|---|---|
| `src/jacobik/` | Generator package (`builder/`, `core/`, `topology/`, `ik/`, `stats/`, packaged `data/`) |
| `src/fit/` | Statistics database fitting pipeline (`python -m fit.build_db`) |
| `interface/` | CLI wrapper (`jacobik_cli.py`), `ligand_library.py`, dataset preparation & notebooks |
| `dev/` | Unit/parity tests (`dev/tests/`), verification checks (`tools.check`), panel runner (`tools.run`) |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for pipeline details, [`PERFORMANCE.md`](PERFORMANCE.md) for `v1.0a` benchmarks, and [`CHANGELOG.md`](CHANGELOG.md) for release history.

---

## License

Released under the [MIT License](LICENSE).

`tmQM` and `lnQM` datasets are used under their respective terms; `xtb` and `MACE` are external third-party tools. Citation: to be added (`CITATION.cff`).
