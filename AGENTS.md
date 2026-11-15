# AGENTS.md: onboarding for coding agents (Claude, Gemini, Codex)

Read this first in every session. It holds the rules and traps that are not obvious from the code. Your task comes from your
directive in `.claude/` of the main checkout (`/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/`, gitignored, so worktrees do not
contain it: use absolute paths). `.claude/README.md` indexes the directives, plans, handoffs and the current worktree list. If a
directive and this file disagree, the directive wins.

## Roles
- **Owner:** directs the work and merges branches. **Claude (Opus):** auditor and planner; writes directives, audits checkpoints.
- **Implementers:** Gemini, Codex, Claude subagents; each works in its own worktree and branch.

## Sources of truth
| Topic | File |
|---|---|
| Architecture, pipeline, profile/fingerprint | `ARCHITECTURE.md`; flag history `src/jacobik/core/profile.py` header, `research/archive/flags/FLAGS.md`, `CHANGELOG.md` |
| jgen (structure generator) goals, metrics, process rules, queue | `.claude/PLAN_jgen.md`, `.claude/PLAN_p2_restart_2026-11-12.md` |
| NN correction (parked) | `.claude/plan_nn_correction_2026-10-25.md` |
| NERSC Perlmutter: access, budget, job template | `.claude/REF_perlmutter.md`, `.claude/perlmutter_job_template.sbatch` |
| Locked sets | `research/bench/benchmarks/LOCKED_SETS.md` |
| Old notes (D2, DS, stereo, roadmap, restructure log) | `research/archive/docs/` (index in its README) |

## Environment (every shell)
```bash
export PATH=/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.pixi/envs/linux/bin:$PATH
export PYTHONPATH=src:dev:.:research/bench        # from the worktree root
```
- **Production fingerprint `a47283387172`** (P2R fold-in on `master`). Default DB `src/jacobik/data/constraints_2026-11-12.npz` (shipped packaged alongside `shannon_radii.json` and resolved by default unless `constraints_db=` / `--db` / `JACOBIK_DB` is passed; identical protected copy at `data/db/constraints_2026-11-12.npz`, sha256 `62c58d53...`; lossless container with embedded meta, no sidecar
  needed; never point a default at a json). Golden references: `research/data/golden/` (`a47283387172`); older ones (`d5e2f4113497`, `5b56969afb9c`, `6b47c3fb45a9`, ...) in
  `research/archive/data/2026-11_golden_history/golden_<fp>/` (set `JACOBIK_GOLDEN_DIR`). **Never overwrite a golden directory without
  the auditor.** Any experiment flag at its default must leave the fingerprint unchanged; it changes only at an owner-approved fold-in.
- **A legacy json DB needs its `<stem>.meta.json` sidecar** (q5_model, q1_variant). Without it every parent silently runs another Q5
  model; the jgen bench refuses a `JACOBIK_DB` without one.
- **`xtb` must resolve** (`which xtb`). Otherwise the builder silently falls back to the MMFF embed and logs "GFN2 embed refinement
  failed": every structure built that way is invalid. Scripts that build shells must check for that message and fail loudly. On Colab,
  `cj.py` sets `JACOBIK_XTB`.
- Hardware: 12 CPU cores, 15 GB RAM, one RTX 2060 SUPER (8 GB). Perlmutter A100s; Colab L4s through `cj.py`.

## Layout
| Path | What |
|---|---|
| `src/jacobik/` | Production generator (`builder/`, `core/`, `topology/`, `ik/`, `stats/`; `ik/solver_replay.py` is generated; `core/profile.py`) |
| `src/fit/` | Fitting pipeline for the statistics database (`python -m fit.build_db`) |
| `dev/` | Developer tree: tests (`dev/tests/`); `tools.check` (`dev/tools/check/`, levels `ws`, `pm-quick`, `pm-full`; CPU reference is `research/bench/tools/jdev/golden.py`), `tools.run` (`dev/tools/run/`), `dev/tools/jdev/build_replay_solver.py` (canonical generator of `solver_replay.py`) |
| `interface/` | CLI, notebooks, shipped examples |
| `research/` | `bench/` (`benchmarks/`, `tools/jgen/` benchmark runner), `campaigns/` (contracts and inputs), `paper/`, `gpuwork/colab/cj.py` (use your worktree's copy), `archive/` |
| `data/`, `research/data/`, `research/archive/data/` | Gitignored data. Outputs go under `research/data/<track>/`; archive is read-only |

## Protected: never write
`data/db/` (all DBs and sidecars, incl. `constraints_2026-10-18.json`), `data/ref/`, `data/refit/frame_v1/`,
`data/refit/experiments/index.json`, every `research/data/golden*`, locked sets, `research/data/p2r/final/`, `research/data/architector/`.

## Binding rules
- **Git:** commit only on your own branch, in your own worktree (use absolute paths for others; do not `cd` into them). Never merge,
  push, rebase shared branches or delete branches; the owner merges. Checkpoints mean stop: report, commit, wait for the audit.
- **Reproducible DB:** any rule that changes the constraints DB (typing, exclusions, donor sets, fit forms, priors) goes into the
  production pipeline (`src/fit/`, run by `build_db.py`), never only a bench or diag script. A DB's meta records code commit,
  flags and input hashes, so it can be rebuilt.
- **Data custody:** tmQM: read only `CSD_code`, `SMILES` and geometry. Never read QM property columns, not even via `hasattr`.
- **Panels:** develop on `tail` and `cal` only. Never use `val` (`motion_validation`), the reserve panel or any locked set for tuning
  or evaluation. The auditor is custodian of the locked sets.
- **Solver changes:** every experiment flag is registered in `profile.FIELDS` and read by its module at import; overriding a hardcoded
  production setting raises on purpose; production behaviour at the default is bit-identical; keep CUDA-graph replay working
  (regenerate `solver_replay.py` with the canonical generator, never hand-edit); the generator never fails (degrade, never raise to
  the caller); no embed spot-fixes for individual complexes.
- **Comparisons:** pair every arm with a control on the same machine and GPU type (GPU results are platform-dependent; non-top-1
  pool rows vary even on one L4). Do not rerun an unchanged control: `tools.run panel ... --control-cache C` reuses the recorded
  control when its key matches (committed code under `src/` and the bench, DB and sidecars, GPU type, flags, seeds, settings),
  reruns a 5-parent canary on the same machine, and rejects the cache if any top-1 differs; a dirty checkout gets no cache. Write the
  evaluation contract before opening results; report parent-clustered bootstrap CIs.
- **Processes:** every run of parents goes through `python -m tools.run` (no hand-rolled loops, no per-agent GPU scripts): your code
  with `--repo <worktree>`, flags with `--arms-spec`, your DB with `--env JACOBIK_DB=...`. Run `tools.run estimate` first; over 20
  minutes needs restructuring or the owner's go. Local GPU: submit to the broker (`tools.run submit --spool research/data/runs/spool`)
  when one runs, otherwise one `tools.run panel` at a time; `nvidia-smi` first. Kill by PID; never `pkill -f` or `pgrep -f`. `/tmp`
  is RAM-backed: nothing large there.
- **Loops over complexes never fail the batch:** record the failure and continue.
- **Monitors and pollers:** one watcher per running job, at most; reuse it, never start a second for the same job or log; stop it when
  the job ends; poll no faster than the state changes (>= 60 s for jobs and VMs, >= 5 min for `watch` on a Perlmutter login node);
  no watchers "just in case".
- **Failures are reported, not tuned away.** A failed gate is stated as failed.

## Compute rules (owner)
- **Perlmutter is the default for anything beyond a quick check.** Login node (free): `pm-quick`, dev suites, <= 10-minute A100 bursts.
  Compute nodes (billed): gates and full panels via `pm-full`, one shared A100 (or a 4-GPU node when speed matters) per comparison,
  all arms in the same job. **Every billed job needs the owner's go, per job, with the QOS named.** Never `sbatch`/`salloc`/`srun`
  without it.
- **Workstation:** unit tests, fingerprint, 1-3 parent checks, `tools.check ws`. The 2060 is for dev runs, not gates.
- **Local limits:** <= 3 processes and <= 4 GB RSS per agent unless the directive says otherwise; check `free -g` first; nothing heavy
  below 4 GB available; every process pins numerical libraries to one thread (`OMP/MKL/OPENBLAS_NUM_THREADS=1`).
- **Perlmutter login node** (all agents share one allowance): <= 32 threads (`taskset`), <= 30 GB, `nice`, `nvidia-smi` before GPU
  use; pack many-file outputs (inode quota; `tools.run` packs by default); record transfers in `$SCRATCH/jacobik/data/MANIFEST.sha256`.
- **Judge GPU use by power draw and wall time, not `nvidia-smi` util %** (a starved A100 shows high util at 60 W).
- **Saturate before scaling:** profile on 1-2 cases, vectorise and batch, overlap CPU and GPU, estimate wall time before any
  panel-scale run; fill the hardware you have before asking for more.
- **Validation tiers:** every change: targeted tests + fingerprint + 1-3 parents (seconds to 2 min). Every checkpoint:
  `python -m tools.check ws` (about 5 min) or `pm-quick`. Once per package, before fold-in: panels and gates through `tools.run` /
  `pm-full` with the control cache.
- **Colab (`cj.py`, fallback only):** run it from your own worktree (the main checkout's copy is stale); VM prefixes `jgA*`, `jgB*`,
  `nn0*`, L4 unless told; shared cap 180 units (`cj.py credits`). `cj.py down` as soon as the work ends, including after errors;
  fetch and count files locally before `down`; `cj.py up` can block (run it in the background); poll with `cj.py get`, not `tail`;
  never stop a VM you did not start.

## Commits
End commit messages with the attribution line your directive gives. Keep `PROGRESS`/`REPORT` files updated with dated entries,
cloud spend and plainly stated failures.
