# D3 harness commands (DRAFT, not frozen)

Run from `.claude/worktrees/jgen-d3`, branch `jgen/d3-harness`:

```bash
export PATH=/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.pixi/envs/linux/bin:$PATH
export PYTHONPATH=src:.:research/bench
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Planning (no VM, no generation):

```bash
python -m tools.jgen.d3.run stage1 --plan-only
python -m tools.jgen.d3.run stage2 --plan-only
python -m tools.jgen.d3.run estimate
python -m pytest -q tests/test_d3_harness.py
```

D2 handoff: fill `D3_ARMS.json`'s D/DS DB paths, exact flags and fingerprints, and all four ablation recipes.
Each recipe contains `db`, complete `flags`, `fingerprint`, and `ds_fingerprint`; typing/OS/fit changes require
reproducibly built ablated DBs, not a runtime toggle over the full DB. Reproducibility meta schema is
`build: {code_commit: <40 hex>, flags: {...}, input_hashes: {<input>: <64 hex>}}`; the same keys at top level are accepted.
The legacy C DB's meta is checked and hashed, but exempt from the D2 build-schema requirement.

The auditor must freeze and commit both the contract and config before real launches. Real launches refuse the
current draft/unresolved settings. Stage 1 automatically computes C/D/DS Q7-inclusive curvature on CPU first; existing summaries
can be supplied with `--curvature-dir` (subdirectories `C/summary.json`, `D/summary.json`, `DS/summary.json`, exact identities checked).
All arms for each parent run consecutively in one VM worker invocation, with persistent arm subprocesses. No local GPU.

After D2 and audit freeze only:

```bash
python -m tools.jgen.d3.run stage1
python -m tools.jgen.d3.analyse stage1 --root /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/D3/stage1/d3_output
python -m tools.jgen.d3.run stage2
python -m tools.jgen.d3.analyse stage2 --root /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/D3/stage2/d3_output
```

Stage 3 accepts only an auditor-supplied run-local code file and SHA256. Do not open `.claude/locked/`.
The auditor creates the run directory and its `OWNER_GO` file; stage 1 must qualify, code/config identities must
match, and `foldin_check` must pass. The run goes in a fresh `execution` subdirectory, preventing reopening it.
Replace the example date/path/hash with the auditor's supplied values:

```bash
python -m tools.jgen.d3.run stage3 --codes-file /auditor/supplied/codes.txt --codes-sha256 <supplied-sha256> --stage1-analysis /path/to/stage1/d3_output/analysis.json --out /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/locked_runs/jgen_promotion_reserve_<date>
python -m tools.jgen.d3.analyse stage3 --root /mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/locked_runs/jgen_promotion_reserve_<date>/execution/d3_output
```

The analysis is once-only (`analysis.json` is never overwritten). Qualification is stage-specific: stage 1's candidate
is provisional; fold-in additionally requires the stage 2 holdout and stage 3 gates. DRYRUN/reproduction never qualifies. Curvature summaries require affirmative Scorer Q7 coverage and matching FK prior identities across arms.
The FK prior is out of curvature scope (audit 2026-11-06 §4); cached summaries without these fields must be regenerated.
When q7_bound_centre is enabled, the DB must have its hashed .bends.json sidecar; it is shipped and verified on the VM.
An invocation failure is not authorization to rerun a stage. Inspect its lifecycle and report every failure.

Only following a failed real stage 1, with registered D2 ablation recipes (no setting retuning):

```bash
python -m tools.jgen.d3.run ablate --stage1-analysis /path/to/stage1/d3_output/analysis.json --out /path/to/D3/ablations
```

Order is far-field form, physical Q6, Q7 bound centre, typing/OS. Only failed candidate arms rerun; original C/S
records are linked for analysis. Analyse each ablated output with `analyse stage1`; report all groups.

The amendment-1 owner-approved harness smoke is `run stage1 --smoke`: exactly three parents, seed 11, C/DS with the production DB
substitute, every record labelled DRYRUN, VM `jgD3s`. The one D3-H audit-authorized attempt has PASSED (6/6); no further retry or smoke is authorized. Data and lifecycle logs are under the shared `D3/dryrun/smoke_amendment1/` directory. The launch plan is embedded in the bundle.
