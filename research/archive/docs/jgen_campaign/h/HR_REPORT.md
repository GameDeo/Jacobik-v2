# Checkpoint H-R — full returned-pool export, 2026-10-30

Branch: `jgen/b-solver`, based on the accepted G-3c step 1 + H-1 commit
`40a0cd4`. Scope: benchmark runner, export/coverage helper, and tests only.
Production fingerprint remains `6cc586e17ab1`. No solver changes, golden run,
H-3 evaluation, or bulk export. Colab spend: zero; no VM or GPU job started.

## Size estimate (announced before exporting)

H-3 consists of production and combined patch arms, seeds 11–13, cal + tail.
Count cal and tail separately, conservatively including their overlap and NEPNEF:
120 + 10 = 130 parent-panel entries, 780 parent/seed/arm units.
Read only the shapes of existing seed-11 control pools on those panels.

XYZ size estimate: 42 bytes per atom plus 512 bytes per row for atom count and
JSON header. Sampled 117 cal and all 10 tail parents; FEKKAJ, MIVXAR and QEZVEY
have no usable coordinate sample. Across the 127 samples: mean 64.06 atoms,
maximum 145; mean row size 3,202 bytes, maximum 6,602 bytes.

| Panel | Sampled parents | Mean rows/parent | Max rows | Estimated bytes/parent |
|---|---:|---:|---:|---:|
| cal | 117 / 120 | 31.97 | 144 | 104,381 |
| tail | 10 / 10 | 54.50 | 224 | 181,179 |

Using panel means for the three missing samples:
`(104381.33 × 120 + 181178.8 × 10) × 3 seeds × 2 arms = 86,025,288 bytes`
(about 87 MB rounded). This sums per-parent row/atom products; it does not
multiply independent means when they are correlated.

Conservative observed-max scenario, explicitly in the requested factor order:
`6,602 bytes/row × 224 rows/parent × 130 parents × 3 seeds × 2 arms`
`= 1,153,501,440 bytes` (1.15 GB), plus small manifests and top-1/floor copies.
Even allowing about 20 MB for those copies/manifests remains below 2 GB.
These are projections from historical pool sizes, not an upper bound on future
solver outputs. Re-estimate if row budgets or arm/seed counts change.

Default is therefore uncompressed XYZ. `--xyz-compression xz` is available:
one multi-frame `.xyz.xz` per parent/arm/seed, rather than per-row compression.
It preserves every row and also saves plain top-1/floor copies. Use xz if a
revised run estimate exceeds 2 GB. No panel-scale export was performed here.

## Runner behavior and file contract

`python -m tools.jgen bench run` now defaults to full returned-pool export under
`OUT/xyz/CODE/`; `--xyz-dir DIR` overrides the root (also respected per arm in
`--arms-spec`). `--no-xyz` disables all XYZ/manifest writes, retaining in-memory
full-pool coverage. `--device cpu` selects CPU for both solving and MACE;
CUDA remains the default.

For each parent/arm/seed:

- `CODE_ARM_sSEED_row00000.xyz`, etc.: one file per returned conformer.
- `CODE_ARM_sSEED_top1.xyz` and `_floor.xyz`: compatibility filenames, exact
  copies of the corresponding row contents (builder order, JSON comments).
- `CODE_ARM_sSEED_pool.json`: complete row manifest, reference geometry and
  symbols, explicit builder-to-reference index permutation, and format version.
- With xz: row files become one `CODE_ARM_sSEED_pool.xyz.xz` multi-frame file;
  the manifest and top-1/floor copies remain.

Labels come from `conf.complex.atoms`, and coordinates come from those same
atoms in builder order. Every comment is JSON containing row index, row key
`[embed_idx, assignment_idx, trial_idx]`, the **returned `conf.score`** at full
precision, `score_scale=returned_conformer_score`, absolute `mace_energy_kj`,
and independent `is_top1` / `is_floor` booleans. A row can carry both roles.
Nonfinite energies/scores serialize as JSON null, and do not remove the row.
The floor uses the minimum finite energy, preventing NaNs from selecting it.
Exports also run when the top-1 energy is nonfinite; failures remain failures.
The pool means all returned conformers after the generator's existing output
window/dedup, not discarded optimization trajectories. No row-selection policy
was changed.

The old diagnostic pool's `final_score` is never used. The saved score is the
same `conf.score` used for `scores` and `top1_score` in runner records.

## Full-pool reference coverage

Every unit record now contains `coverage`: per-row RMSDs, minimum RMSD,
top-1 hit, floor hit, their union, and `full_pool_within_1A`. A hit is <= 1 Å
symmetry-aware heavy-atom Kabsch RMSD, including the metal, using the established
`tools.runkit.geom` interface. Atom order is explicitly mapped to reference
order; automorphism count/capping is recorded. Nonfinite geometry cannot hit.
Reference automorphisms are cached once per parent and rows are processed in
chunks of 64. The reference and permutation are in the manifest so recomputation
needs neither raw tmQM tables nor MACE.

When export is enabled, the runner reads **all serialized rows back** to compute
coverage and checks metadata and counts against the manifest. Thus coverage is
computed at saved XYZ precision. With export disabled it uses all in-memory rows.
Unit hit booleans support parent coverage as the union over the declared seeds;
failed/missing units must remain in the H-3 contract's denominator.

Standalone recomputation (accepts one or more explicit manifests; failures are
recorded per input and do not abort the batch):

```bash
python -m tools.jgen.bench.pool_export \
  research/data/jgen/campaign/B/hr/cpu/xyz/DAMVAQ/DAMVAQ_HR_cpu_s11_pool.json \
  --out research/data/jgen/campaign/B/hr/recomputed.json
```

## Gates and artifacts

- `pytest -q tests/test_jgen_pool_export.py`: **4 passed** (0.35 s).
  Two permitted tail parents, DAMVAQ and JEGPUJ, each tested in plain and xz
  modes. Real reference geometries supply the fixtures; synthetic top/floor
  rows miss while a middle pool row hits, proving top-1/floor coverage <=
  full-pool coverage, strictly in these fixtures. Reversed builder atom order,
  role markers, exact returned scores, row keys, null energy, coordinates,
  compressed readback, and file counts are checked.
- One parent end to end on CPU, actual production solve/GFN2/MACE, seed 11:

```bash
export PATH=/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.pixi/envs/linux/bin:$PATH
export PYTHONPATH=src:.:research/bench
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tools.jgen bench run \
  --arm HR_cpu --panel tail --only DAMVAQ --seeds 11 --device cpu \
  --out research/data/jgen/campaign/B/hr/cpu
```

  **PASS**, status ok, 27.7 s parent wall time, 16 rows, 17 unique MACE
  structures including reference. File count in the parent export directory:
  **16 row XYZ + 2 role XYZ + 1 manifest = 19 files**, 80,044 bytes.
  Plus one record JSON and the run's embedding-validity ledger (empty).
  Every exported score/energy/key equals the corresponding record entry;
  top-1 comment score equals reported top-1 score `-2.6486687660217285`.
  Serialized-file recomputation exactly matches record coverage. All 16
  geometries finite; best RMSD 1.240308 Å, hence no 1 Å hit in this smoke unit.
  This is an I/O correctness gate, not an arm evaluation result.
- Default and CPU record fingerprint both **6cc586e17ab1**. No production
  source files touched; golden deliberately omitted per directive.
- Compilation and `git diff --check` pass.

Artifacts are gitignored under this worktree's
`research/data/jgen/campaign/B/hr/`: `cpu.log`, `cpu/records/`, `cpu/xyz/`,
`embedding_validity.json`, and `recomputed.json`. Only cal/tail fixture pools
and permitted geometry/SMILES input were used. No QM property columns read.
No failed gate. Stop at Checkpoint H-R for the auditor/owner merge; H-3 and
G-3d remain outside this checkpoint.
