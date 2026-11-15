# Independent coordination-complex benchmark

This directory prepares an Architector comparison without importing or changing
JacobIK. Everything added for this work is contained here, including a separate
Pixi manifest, lockfile, and local environment.

The proposed independent scoring layer is described in
[EVALUATION_DESIGN.md](EVALUATION_DESIGN.md). It accounts for JacobIK's
geometry input, native versus commonly refined outputs, validity, isomer
coverage, ranking and computational budgets. Scoring is not implemented yet;
the current Architector runner still uses its original CN-only inputs.

The primary workload is now the **29-case failure-mode challenge panel** in
[CHALLENGE.md](CHALLENGE.md), with its own
[comparison protocol](CHALLENGE_PROTOCOL.md). It includes 14 bespoke designed
targets, explicit haptic groups, sulfur stereochemistry, narrow/wide/trans
bites, eight/nine-donor ligands, secondary-sphere interactions, and ensemble
targets. Its sources include an experimental Zr-HOPO CIF and twelve mapped
published computed Xantphos/XL-Xantphos geometries.

The original baseline panel contains **40 complexes, 27 ligand definitions, 16 metals, and
coordination numbers 2 through 9**. It is a diagnostic panel chosen to exercise
specific construction difficulties, rather than a statistical sample of CSD.
The full selection and rationale are in [CATALOG.md](CATALOG.md).

- **8 literature anchors:** original crystallographic CIFs, citations, and
  extracted, atom-mapped experimental heavy-atom references.
- **6 controls:** small complexes that isolate coordination number, geometry,
  charge, and spin handling.
- **26 designed cases:** specified ligand combinations covering rigid and
  flexible chelates, N/O/P/S donors, macrocycles, denticity up to six, bulky
  phosphines, and larger coordination spheres.

No tmQM data, existing JacobIK datasets, or Architector benchmark structures
were used to assemble the panel. Independently sourcing a complex does not
establish that its chemical identity is absent from tmQM. An identity-overlap
audit has not been performed; the current claim is independent provenance.

## Installation and platform status

**Use Linux for the packaged Architector runtime.** A native Windows install
was attempted: `architector==0.0.10` could not resolve because conda-forge has no
`win-64` candidate for its required `pynauty` dependency. `xtb-python` itself
does have Windows packages. This is a limitation of the available packaged
environment, not proof that a custom Windows source build is impossible.

The `audit` environment works on Windows and Linux. It prepares and validates
the data, extracts references, and exports Architector inputs. The `default`
environment adds Architector on Linux. CREST is optional and is disabled in the
benchmark protocol. No WSL installation or operating-system changes were made.

The runtime is pinned to Architector 0.0.10 and Python 3.11, with the complete
Linux solve in `pixi.lock`. Conservative NumPy, ASE, pandas, and mendeleev
constraints reduce compatibility drift for this older comparator. The lockfile
pins their actual resolved versions.

From this directory, on Linux with Pixi installed:

```sh
pixi install --environment default --locked
pixi run --environment default doctor
pixi run --environment default validate
pixi run --environment default smoke
```

The Linux environment has now been installed and checked on Xubuntu. The
Architector import check and real baseline/challenge S01 runs pass in both
relaxed and unrelaxed tracks. [SETUP_STATUS.md](SETUP_STATUS.md) records the
current checks and the separate challenge D03 timeout. Installation initially
requires internet access.

Windows preparation commands, also run from this directory:

```powershell
pixi install --environment audit --locked
pixi run --environment audit validate
pixi run --environment audit references
pixi run --environment audit python scripts/run_architector.py --all --dry-run
```

Use `pixi run` so that Open Babel's plugin paths are activated. Invoking an
environment's `python.exe` directly can leave its SMILES plugin unavailable on
Windows.

## Running the comparator

Start with one case on Linux:

```sh
pixi run python scripts/run_architector.py --suite challenge --case D03 --seed 17 --n-conformers 3 --timeout 600
```

Export or run the challenge panel with three seeds and up to ten requested conformers
per core geometry:

```sh
pixi run python scripts/run_architector.py --all --dry-run
pixi run python scripts/run_architector.py --all --timeout 600
```

The runner defaults to `--suite challenge`. Use `--suite baseline` to select
the original 40 cases (including A01-A08). IDs are local to a suite: challenge
S01 and baseline S01 are different targets. Every run records its suite.

H04-H06 currently export full target specifications and explicit adapter-gap
records. S03 requires `--no-final-relax` because of upstream hydride handling.
The full-relaxation dry run therefore exports 75 inputs and 12 capability
records across 87 jobs; assembly-only exports 78 inputs and 9 capability records.
These limitations remain in the run manifest rather than being silently skipped.

The default seeds are 17, 29, and 43. Every job gets its own process, 600-second
wall-time limit, log, temporary directory, and explicit charge/spin inputs.
Numerical-library thread counts are set to one. Results go to a new directory
under `results/<suite>/<relaxed|unrelaxed>/<run-id>/` with exact input JSON, environment metadata, input/script
fingerprints, output MOL2/XYZ files, initial structures when supplied by
Architector, and a checkpointed run manifest. Empty outputs, exceptions,
crashes, and timeouts remain visible. `returned_structures` records an API
outcome; independent geometry-quality scoring is still to be implemented.

Baseline inputs disclose the metal, coordination number, ligand SMILES, coordinating
atom indices, formal charge, oxidation state, and assumed unpaired-electron
count. Expected geometry and crystal coordinates are withheld. Conformers use
GFN2-xTB assembly and final relaxation by default. `--no-final-relax` provides a
separate track; it still uses xTB during assembly and is not a force-field-free
or energy-free timing measurement.

## Offline data and references

All chemical definitions and reference coordinates are local. The scripts do
not download structures or query a database at run time. This is an offline
dataset and runner, not a bundled offline installer for Linux dependencies.

| File | Purpose |
| --- | --- |
| `data/cases.json` | Selected cases, composition, charge, spin, CN, and rationale |
| `data/ligands.json` | Exact SMILES and zero-based donor indices, checked with RDKit and Open Babel |
| `data/sources.json` | Paper citations, DOI/CCDC IDs, download URLs, nominated CIF blocks, preparation notes |
| `sources/A01.cif` through `A08.cif` | Unmodified downloaded crystallographic data |
| `sources/manifest.json` | Checksums of the archived sources |
| `data/references/*.xyz` | Extracted experimental heavy-atom complexes, in angstrom |
| `data/references/*.json` | Mapping from ligand atom order to CIF labels and periodic images |
| `data/references.json` | Extraction method, file checksums, and preparation status |

Reference extraction expands the supplied CIF symmetry operations, unwraps
periodic molecules, removes spectator ions/solvents, and matches every complete
ligand graph and its metal-bound donor set to the catalog. All selected ligand
sites have occupancy 1, and all metal-donor distances agree with the CIF's
published geometry table within 0.005 angstrom. Two expanded monoclinic space-group names are
normalized in memory for ASE; the supplied symmetry operations and original
CIF files are preserved. Original crystal distortions are retained.

The references contain heavy atoms only. They do not independently establish
hydrogen positions, proton tautomers, spin states, or stereochemical labels.
The other 32 cases have no experimental coordinate reference and belong in
construction/validity analyses, not crystal-RMSD averages. See
[PROTOCOL.md](PROTOCOL.md) for the comparison design and remaining work.
This paragraph and the table above describe the baseline references; challenge
data, reference types and checks are documented in [CHALLENGE.md](CHALLENGE.md).

## Research basis

Architector: Taylor et al., *Nature Communications* **14**, 2786 (2023),
[DOI: 10.1038/s41467-023-38169-2](https://doi.org/10.1038/s41467-023-38169-2).
Installation and API were checked against the
[official LANL repository](https://github.com/lanl/Architector) and
[conda-forge distribution](https://anaconda.org/conda-forge/architector).

| Anchor | Selected molecular unit | Paper |
| --- | --- | --- |
| A01 | [Co(en)3]3+ | Conrad-Marut et al. (2021), [10.1107/S2056989021009336](https://doi.org/10.1107/S2056989021009336) |
| A02 | [Zn(terpy)Cl2] | Kong et al. (2014), [10.1107/S1600536814023605](https://doi.org/10.1107/S1600536814023605) |
| A03 | [Co(NCS-N)4]2- | Oliver et al. (2023), [10.1107/S2056989023000981](https://doi.org/10.1107/S2056989023000981) |
| A04 | [Cu(acac)2(ambim)] | Siddikova et al. (2025), [10.1107/S2056989024011538](https://doi.org/10.1107/S2056989024011538) |
| A05 | [Fe(oxalate)3]3- | Piro et al. (2018), [10.1107/S2056989018008022](https://doi.org/10.1107/S2056989018008022) |
| A06 | [Ni(Hdmg)2] | Li et al. (2003), [10.1107/S1600536803024656](https://doi.org/10.1107/S1600536803024656); [COD 2202758](https://www.crystallography.net/cod/2202758.html) |
| A07 | [Ni(NCS-N)2(3-methylpyridine)4] | Krebs et al. (2022), [10.1107/S2056989022008891](https://doi.org/10.1107/S2056989022008891) |
| A08 | [Zn(4'-Cl-terpy)2]2+ | Adrian and Arman (2022), [10.1107/S2414314622010963](https://doi.org/10.1107/S2414314622010963) |
