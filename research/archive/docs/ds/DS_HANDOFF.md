# DS investigation: handoff and integration package (2026-11-11)

Everything that can be done without a billed job is done. What is left is three billed jobs (each needs the owner's go with the QOS named), one local CPU step, and the owner's decision on fold-in.
Nothing here merges, pushes or overwrites a production DB or golden.

## 1. State
- **Candidate database `DF`** = the Step 4 `D-final`: `donor_assignment` + face-evidence rule + stable ancestor sampling + `structure_sanity`, on the D2 package (hapto typing, physical Q6, `os_sanity`, `fit_exclude_locked`).
  Cleaner than D2: the erroneous 2.66 A Pd-O component in LUQVOK is gone, BADQIH/NOGDAQ/XIWKAR-type donor errors fixed, 79 broken parents screened, sampling made stable (zero collateral resampling).
  Reproducible: D0 (rules off) reproduces the recorded D2 key by key (425,635 entries); D-final has 419,644 entries.
- **Evidence it is not worse where it matters (fixed pools, 9 cases, `DS_STEP5_RESULTS.md`):** mean selected-row gap to the pool floor 41.3 -> 29.5 kJ/mol, median 14.2 -> 4.7; one regression (HOXJUZ, one sparse Ni-S key, a coordination trade the owner accepted).
- **Sensitivity (conditional on those pools, not a guarantee):** 445 of 448 registered trials scored, none severe, none changed the selected energy; the three failed trials are a bookkeeping case restated by contract v2 A1 (post-hoc). The fits themselves flip component count under 1-5 reverted parents (F8); not changed (Step 7 record).
- **Generated structures (J2, `DS_STEP6_RESULTS.md`):** corrected DB against production, top-1 -13.9 kJ/mol [-41.4, +10.7] over 11 development parents (not different from zero); USAPOU and UMUBAF much better, YABYEJ, LOPJEF, ZAQNUE worse (the same three as in the D2 package; a pool/search lead, not a DS rule). The existing P/S start **fails** its candidate rule (upper bound 5.26 vs 5.0 kJ/mol), so the gate has arms C and D only. Damaged-DB trials: a weight shift moves LIPDEU's generated winner by 50-90 kJ/mol, toward lower energy and with intact coordination; LUQVOK by at most 17.
- **Gate (J3, `DS_GATE_RESULTS.md`):** stage 1 passes on both roots (primary top-1 -9.27 kJ/mol [-18.31, -1.58]; all seven gates true). Stage 2 fails the registered holdout and failures gates because five parents did not complete (ADINOR and LADJAB `KeyError: 'v'` in both arms, UZOREI timeout in both, EPUPEL control timeout, IWOLEN mapping failure in `all`), although its top-1 is -24.19 [-43.96, -7.14]. Under the registered rule the DB is **not qualified**; options (a)-(c) in `DS_GATE_RESULTS.md` are your call.
- **Production is untouched:** every new flag is default off; the merged gate branch reproduces fingerprint `5b56969afb9c`; golden 0/490 differ; pytest 706 passed, 14 failed (the known CUDA-capture failures of the 2060), 5 skipped (`DS_PROGRESS.md`).

## 2. Branches (tips at the time of writing: `data/ds-cleanup` `e076386`, `data/ds-gate` `d84ec6f`; the Perlmutter snapshot `ds-570adf7` predates the last docs-only merges, so rebuild it from the gate tip before J1)
| Branch | Worktree | What |
|---|---|---|
| `data/ds-cleanup` | `.claude/worktrees/ds-data` | all DS work (code, tests, docs, job scripts); based on an older master |
| `data/ds-gate` | `.claude/worktrees/ds-gate` | master `54c19f4` (stereo fold-in) merged with `data/ds-cleanup`; the code every job runs and the branch to merge. One conflict (`STEREO_PROGRESS.md`, ours was a strict prefix; master's version taken) |

`data/ds-gate` must be re-merged with `data/ds-cleanup` after any new commit on the latter (`git merge data/ds-cleanup` in the ds-gate worktree) and the snapshot rebuilt from its tip.

## 3. The jobs (none submitted)
Snapshot (workstation, from the ds-gate worktree, after the final merge):
```bash
FULL=$(git rev-parse HEAD); SHA=${FULL:0:7}
git archive --format=tar --prefix=ds-$SHA/ HEAD | ssh perlmutter "cd \$SCRATCH/jacobik/ext && tar x && echo $FULL > ds-$SHA/COMMIT"
# scp does not expand $SCRATCH: use the resolved path /pscratch/sd/a/achen310
ssh perlmutter 'mkdir -p $SCRATCH/jacobik/data/ds_gate_inputs'
scp research/data/audit/2026-11-10_DS/gate_inputs/target_overlays_v1.json perlmutter:/pscratch/sd/a/achen310/jacobik/data/ds_gate_inputs/   # sha 9a9c820b... (already there)
```
Common env: `DS_CODE=$SCRATCH/jacobik/ext/ds-$SHA`, `PIPE=$SCRATCH/jacobik/runs/ds-pipeline-20261111-1734`, `OVERLAY=$SCRATCH/jacobik/data/ds_gate_inputs/target_overlays_v1.json`.
All three scripts post Discord start / progress / end notices and refuse to start on a failed preflight (fingerprint, DB hashes, MACE model, xtb, GPU count, candidate-DB provenance).

| # | Job | Script | Node, QOS | Cap | Est. | Stop condition |
|---|---|---|---|---|---|---|
| J1 | Rebuild DF at the gate tip with a reproduction check against D-final, then the geometry stress trials (LUQVOK Q1 96, LIPDEU QO 128) | `tools/d2/ds_final.sbatch` | CPU, `regular`, 1 node | 45 min | ~14 min (rebuild and geometry trials run together), ~0.25 node-h | `DF rebuild exit` is nonzero (a mismatch with D-final is reported, not tuned) |
| J2 (done: 61160368, 3 min 37 s, all phases rc 0) | Step 6: current start vs P/S on production and the candidate DB, 13 parents, plus four damaged-trial DBs | `ds_step6.sbatch` (generated) | GPU, `regular`, 1 node (4 A100) | 30 min | ~12 min est., 3.6 min actual, ~0.06 node-h | any phase `rc != 0` |
| J3 (done: 61160804, 19 min 34 s) | Step 8 gate: stage 1 (120 parents) and stage 2 (110 holdout parents), seeds 11-13; with `--with-p` only if J2's P/S rule passes | `ds_gate.sbatch` (generated) | GPU, `regular`, 1 node (4 A100) | 60 min | ~16 min (24 with P), ~0.3-0.5 node-h | any phase `rc != 0`; per-parent cap 5 min, a timeout is recorded and never retried |

Total billed if all run: roughly 0.8 node-hours (budget about 10).

**J1**
```bash
sbatch --export=ALL,DS_CODE=$DS_CODE,PIPE=$PIPE,DS_OUT=$SCRATCH/jacobik/runs/ds-final-$(date +%Y%m%d-%H%M) $DS_CODE/tools/d2/ds_final.sbatch
```
After it: the new DB is `$DS_OUT/DF/constraints.json` (+ `constraints.meta.json`, `DF.sha256`). Its meta must show a 40-hex `code_commit`. Copy it to `$SCRATCH/jacobik/data/db_ds/DF/` and record its sha256.
Then, locally: fetch `luqvok_q1.geometry.entries.jsonl` and `lipdeu_qo.geometry.entries.jsonl` and score them on the saved pools with `stress_run.py score ... --scenarios geometry_0.02A` (as the pilot), then `stress_rank.py`.

**Between J1 and J2** (login node, minutes):
```bash
python research/bench/jgen_campaign/ds/make_damaged_dbs.py $PIPE <stress dir with the *.rank.json copied from the workstation> $SCRATCH/jacobik/data/db_ds/damaged   # uses the pipeline D-final
```
(The four damaged DBs are also built and tested locally in `research/data/ds_proto/step6/damaged/`; the ones from the pipeline D-final are the ones the contract names.)

**J2 / J3** generate the scripts on the workstation and copy them:
```bash
python -m tools.jgen.ds.gate_pm build step6 --out DIR --minutes 30
python -m tools.jgen.ds.gate_pm build gate  --out DIR --minutes 60 [--with-p]
# copy DIR/ds_*.sbatch to Perlmutter first (scp ... perlmutter:/pscratch/sd/a/achen310/jacobik/runs/), then:
sbatch --export=ALL,DS_CODE=$DS_CODE,DF_DB=$SCRATCH/jacobik/data/db_ds/DF/constraints.json,OVERLAY=$OVERLAY,DAMAGED_DIR=$SCRATCH/jacobik/data/db_ds/damaged ds_step6.sbatch
sbatch --export=ALL,DS_CODE=$DS_CODE,DF_DB=$SCRATCH/jacobik/data/db_ds/DF/constraints.json,OVERLAY=$OVERLAY ds_gate.sbatch
```

## 4. After the jobs (local; commands are written and tested on stand-in records)
- **Step 6:** `python research/bench/jgen_campaign/ds/step6_analyse.py <fetched J2 dir> <out>`; reads D vs C (corrected DB, current start), DS vs D (P/S), the P/S candidate rule, the JEGPUJ/SIVBUV supplement, and the damaged-trial cost. Update `DS_STEP7_DECISION.md` from it (a Step 6 failure with one attributable mechanism is the only trigger for a Step 7 change).
- **Gate:** per stage, `python -m tools.jgen.ds.gate_pm assemble stage1 --runs <J3 dir> --out <stage root> --pm-prefix <J3 $OUT on Perlmutter> --df-sha <sha>`, then `manifest stage1 ...`, then the unchanged `python -m tools.jgen.d3.analyse stage1 --root <stage root>` (once only; `analysis.json` is never overwritten). Same for `stage2`.
  **Two roots per stage:** `all` (every parent; ambiguous and unrepaired targets stay visible) and `primary` (`manifest --strata primary --link-from <all root>`: the frozen source-based eligibility, 109 parents in stage 1 and 95 in stage 2). The qualification decision reads `primary`; `all` is reported beside it.
- **Curvature gate (stage 1, workstation CPU):** `python -m tools.jgen.d3.curvature --db <DF constraints.json> --flags '<DF_FLAGS json>' --out <dir>/D` and the same for the control on production (`--db data/db/constraints_2026-11-01.json --flags '{}' --out <dir>/C`); pass `--curvature <dir>` to `manifest`. The gate is false without both summaries.
- **Verdict:** the gates are those of `contract_D3_2026-11-09.md` (energy, guard without the artefact list, catastrophic count, curvature, eta2 holdout, wall <= 1.3x) plus no severe registered stress trial. A failed gate stays failed; any ablation needs its own go.

## 5. If the gate passes: fold-in (owner merges; the auditor re-records golden)
1. Merge `data/ds-gate` into master.
2. Move `DF` (`constraints.json` + `.meta.json`) into `data/db/` with its sha256 in `MANIFEST.sha256`; point production `JACOBIK_DB` at it.
3. In `src/jacobik/profile.py` make the package the production default: `hapto_bond_gate`, `hapto_label_canon`, `hapto_chelate_exclusion`, `q6_physical_all`, `os_sanity`, `fit_exclude_locked`, `donor_identity_reconcile`, `donor_assignment`, `stable_ancestor_sampling`, `structure_sanity` (and `S` flags only if a P/S arm qualified). New production fingerprint; golden regenerated with the auditor (expect changes where the new typing and donor rules change targets); update `AGENTS.md`'s fingerprint line.
4. Apply `DS_reconcile_removal.patch` (removes the superseded reconcile rules 2, 3, 5-10; the affected tests pass on a scratch copy); rerun `tools.check` after it.
5. Tidy: the `jgen/d3-pm` branch carries the D3 launcher this package borrows from; it is not required for the DS scripts, which are self-contained.

## 6. Evidence index (all under the ds-data worktree)
`DS_FINDINGS.md` (F1-F6), `DS_STEP3_REVIEW.md` (closing report), `DS_STEP4_RESULTS.md`, `DS_STEP5_RESULTS.md` (fixed-pool, seed spread, pilot, ranking read-out, F8), `DS_STEP5_6_8_CONTRACT.md` and `_v2.md`,
`DS_STEP7_DECISION.md`, `DS_RULE_INVENTORY.md`, `DS_ASSIGN_PROTOTYPE.md`, `DS_PROGRESS.md`. Data (gitignored): `research/data/ds_proto/pipeline_20261111/`, `research/data/ds_proto/step6/`.

## 7. Known limits, stated plainly
- The sensitivity statements are conditional on 32-row saved pools, three seeds and two keys; they do not show that a weight shift between modes cannot move the generated winner. The damaged-trial phases of J2 are the first test of that.
- The pipeline DBs have `code_commit: unknown`; only the J1 rebuild is usable by the D3 harness.
- Unresolved leads (Step 7 record): F7 Q5 mode flips, F8 component-count instability, the broad LIPDEU QO fallback, SESWIZ ranking seed spread, tmQM-versus-CSD drift parents, 26 reference-incompatible gate targets.
- The workstation 2060 cannot run the CUDA-graph parity tests (14 known failures; not a gate, per the owner's 2026-11-11 note).

## 8. Fold-in (2026-11-12, branch `foldin/ds`)
Owner accepted the gate evidence and asked for the fold-in. Production is now fingerprint `d5e2f4113497` with `data/db/constraints_2026-11-12.npz`
(sha256 `be4e0571...e4db`, 419,644 entries; meta records code commit, flags, input hashes). The ten DS flags are defaults and part of
`PRODUCTION_CONFIG`; the reconcile rule chain is removed (`donor_assignment` is the one rule set); start flags P/S stay off (failed rule).
See `RESTRUCTURE_LOG.md` (2026-11-12) for the full list. Same-machine check: folded-in defaults give the same scores and energies as the gated tree with
explicit DF flags (EFOXII, MIGTEE, HOXJUZ, seed 11). New golden: `research/data/golden_d5e2f4113497/`; the shared golden is swapped by the auditor after merge.
`gate_pm.py` stays pinned to the pre-fold-in production on purpose. Open leads carried forward: `KeyError 'v'` in ADINOR/LADJAB (both arms, a pre-existing
generator defect), search-side regressions YABYEJ/ZAQNUE/HOHZAG, score-surface limit LOPJEF.
