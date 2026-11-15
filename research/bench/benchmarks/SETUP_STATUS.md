# Setup verification

Prepared on 2026-10-17 from a Windows host with Pixi 0.81.0.

The subsequent challenge-panel revision adds 29 targets and 39 ligand
definitions. Its offline chemistry checks pass, including fixed-stereocenter
round trips and pi-group/site accounting. The adapter supports 25 input
exports in the full-relaxation track, plus one hydride input in the
assembly-only track; three richer haptic targets have explicit adapter gaps.
An experimental 55-heavy-atom Zr-HOPO reference and twelve graph-mapped
published computed Pd geometries are prepared locally. See
[CHALLENGE.md](CHALLENGE.md). None of these support labels constitutes a Linux
runtime test.

Challenge dry runs passed with 75 inputs plus 12 explicit capability records
in the full-relaxation track, and 78 inputs plus 9 capability records in the
assembly-only track (29 cases x 3 seeds each). A subsequent baseline dry run
exported all 40 cases with one seed. Full targets embed their exact ligand
definitions even where a capability record replaces an Architector input.

| Check | Observed result |
| --- | --- |
| Native Windows standard Architector solve | Failed: required `pynauty` has no conda-forge `win-64` candidate |
| Linux Architector 0.0.10 environment | Installed on Xubuntu 2026-10-20; doctor and real S01 runs pass |
| Separate Windows `audit` environment | Installed under `benchmarks/.pixi/envs/audit` |
| Baseline catalog chemistry validation | Passed for all 40 cases and 27 ligands |
| RDKit/Open Babel agreement | Heavy-atom order, bond connectivity, ligand charges, donor indices passed |
| Stoichiometry and electronic bookkeeping | Ligand counts, CN, total charge, and electron/spin parity passed |
| Experimental source archive | Eight complete CIFs parse locally; expected paper DOIs verified |
| Reference preparation | All eight complexes extracted with complete ligand-graph and donor-set matches |
| Reference integrity | Source hashes, extracted-file hashes, heavy-atom formulas, CN, and donor distances checked |
| Published bond geometry | All 42 selected metal-donor distances agree with their CIF bond tables within 0.005 angstrom |
| Full offline runner dry run | All 120 case/seed combinations exported successfully |

The Linux `doctor` and real `smoke` task must pass before calling the comparator
runtime verified. The current Windows `doctor` correctly reports absent
Architector, xTB bindings, and pynauty in the preparation-only environment.

No WSL, Docker, or system-wide package changes were made. The installed
Architector environment remains separate from the root project environment.

## Xubuntu runtime follow-up (2026-10-20)

Pixi 0.82.0 and the locked Linux Architector 0.0.10 environment are installed
on the shared NTFS project drive. The Pixi package cache is also on that drive;
the Xubuntu NVMe is not holding the project environment or package cache.

The doctor passed for Architector 0.0.10, xTB Python 22.1, Open Babel, pynauty,
ASE, mendeleev, and RDKit. One real case from each suite (baseline S01 and
challenge S01, seed 17, one requested conformer) returned structures in both
relaxed and unrelaxed tracks. Challenge D03 timed out at 180 seconds and is
recorded as a timeout. These pilot runs are in `results/<suite>/<track>/<run-id>/`.

The full chemistry validation currently reports mismatches between local files
and existing source/reference hashes. Those data files were left untouched.
The `doctor` and the real S01 builds passed independently of that validation.
