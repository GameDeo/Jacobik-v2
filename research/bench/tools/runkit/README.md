# runkit: shared hardware use and orchestration

Package `tools.runkit` (`research/bench/tools/runkit/`, outside `src/`: it never changes the production fingerprint).
Design and rationale: `research/archive/docs/runkit/DESIGN.md`. Phase 2 report: `research/archive/docs/runkit/PHASE2_REPORT.md`.

```bash
export PATH=/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.pixi/envs/linux/bin:$PATH
export PYTHONPATH=src:.:research/bench        # from the worktree root
python -m pytest research/bench/tools/runkit/tests -q          # 48 tests, about 10 s, no GPU
```
Importing `tools.runkit` needs no GPU, xtb or network; torch, rdkit, mace and jacobik load lazily.
Runtime files live in `research/data/runkit/` (gitignored): `budget.json`, `registry/<agent>.json`, `gpu.lock`, `colab/`.

## Modules
| Module | Use |
|---|---|
| `resources` | `env()` one environment block (PATH with xtb, threads=1, CUDA off); `Budget(agent)`; `spawn()` (registered, own session, quota); `kill_tree(pid, budget)` (registered PIDs only); `gpu_lock(agent)`; `workroot(out)` (refuses /tmp); `manifest()` |
| `policy` | `assert_allowed(codes)` strict (val and reserve refused); `assert_panel("tail"/"cal", codes)` (tail's val parents are the documented exception; reserve never) |
| `cache` | content-addressed `Store`, `atomic_write`, `ledger` |
| `xtb` | `require()` (resolve, version, behavioural self-check), `FallbackWatch`, `opt()`/`opt_many()` (never raise, timeout, cache, failures.jsonl), `grad()`/`energy()` |
| `mace` | `Evaluator` (dedup, OOM halving, cache, model sha256, CPU warning), `energies_kj()` |
| `pipeline` | `Watcher` (completeness + settle, shard, skip/done/state files), `Prefetcher`, `split_by_cost`, `RemoteFeed`, `run_groups` |
| `colab` | `Session` (up with READY marker, run/wait/get, `status()` one exec, `fetch_verified`, `finish` = fetch, count, down), `Watchdog`, `stop_rule` |
| `runprofile` | `Trace`, `measure_items`, `project`, `require_budget` (the "estimate before a panel run" refusal), `gpu_step_time` |
| `geom` | symmetry facade (A4 code today; the factored pass lands underneath) |
| `stats` | the one `parent_bootstrap` / `bootstrap_mean` |

## Enforcement (budget.json, owner 2026-10-28)
Hard (raise `resources.Refused`): GPU lock and GPU owner, kill-by-PID (unregistered or reused PIDs refused, no kill by name),
Colab down on exit (also after exceptions, failed `up`, and atexit), RAM-backed work directories.
Warn-only: the CPU quota (`enforce.cpu` false until agents have migrated). `budget.json` is written by the auditor:

```json
{"schema": 1, "cpu": {"Gemini": 5, "A4": 4, "Codex": 3, "runkit": 4}, "gpu_owner": "Gemini",
 "colab": {"cap_units": 180.0, "max_vms": 4}, "enforce": {"cpu": false, "gpu": true, "kill": true, "vm_down": true}}
```
`cpu`: concurrent local CPU processes per agent. `gpu_owner`: the only agent for `gpu_lock` / `spawn(gpu=True)` on this machine.
`colab`: mirrors cj.py's CAP_UNITS / MAX_VMS (cj.py enforces them). `enforce`: true = refuse, false = warn.

## Equivalence with the old code
`xtb.opt` issues A4 relax.py's exact command; `mace.Evaluator` builds the same MACEPolarEvaluator. `python -m tools.runkit.smoke_parity
--a4-relax <a4/relax.py> --pool <pool.npz>` compares both on the same rows and requires exact equality (E_h, geometry, cycles; MACE energies).

## Examples (`examples/`)
`panel_solve_local.py` (GPU panel, Gemini-style), `relax_metrics_cpu.py` (relax + metrics, A4-style), `split_local_colab.py` (local + VM).

## Migration (mandatory for new runs after the merge; in-flight runs exempt)
- A4: `relax.py` -> `xtb`+`mace`; `geom.py` -> `runkit.geom`; `local_feed.py` -> `pipeline.RemoteFeed` (no hard-coded worktree path); watch/shard flags -> `Watcher`.
- Gemini: MACE block of `bench/run.py` -> `mace.Evaluator`; `_prefetch_parent` -> `Prefetcher`; `_assert_gfn2`/`_InfeasibleCounter` -> `xtb.require`/`FallbackWatch`.
- Codex: `panel_local.py` -> `resources.spawn`; `colab_gate.py` -> `colab.Session`; `nnc/policy.py` -> `runkit.policy` (note: that one refuses val on the tail panel too).
- Bootstrap copies to replace with `runkit.stats` (not done, other agents' files): `jgen/bench/metrics.py::bootstrap_mean` (identical draws, tested),
  `jgen/a4/aggregate2.py::boot`, `jgen/diag/analyze_dof_screening.py::bootstrap_parent_ci`, `jgen/diag/analyze_pkgc_confirmation.py::parent_bootstrap_ci`
  (and the inline bootstrap in `nnc/monitors.py`).

## cj.py
The master copy `research/gpuwork/colab/cj.py` is canonical; the copy on the main checkout (`n3b`) is stale (single VM, old fingerprint).
Phase 2 changed it only to allow `--gpu none` (CPU runtime) and a CPU-safe setup check. `colab.Session` drives it through its CLI.
