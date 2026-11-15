# runkit: one shared module for hardware use and orchestration (Phase 1: inventory and design)

Branch `infra/runkit` (base `8ae029b`), 2026-10-27. Phase 1 only: nothing is implemented. All sources below were read
read-only from the other worktrees; the A4 files marked "uncommitted" were read from the working tree of `jgen-a`.

## 1. Inventory

Sources read: Gemini `jgen-b` (`tools/jgen/bench/run.py`, `jdev/multiseed.py`, `bench/replay_parity.py`, `diag/run_pkgc_relaxations.py`,
`B_REPORT.md` Package C section C.1), A4 `jgen-a` (`tools/jgen/a4/{run,relax,geom,metrics,local_feed,strain_run,aggregate2,test_a4}.py`,
`A4_REPORT.md`, `A_PROGRESS.md`), Codex `nn-phase0` (`src/jacobik/nnc/{cost,chunked,relaxation,engines,policy,dataset,monitors,panel_local,gpu_gate,colab_gate}.py`,
`PHASE0_REPORT.md`, `PROGRESS.md`), shared (`research/gpuwork/colab/cj.py`, `research/bench/tools/jdev/*`, `research/refit/nci_v2_orchestrator.py`,
`research/bench/tools/jdev/bench_workers.py`), the auditor scratchpad (`vmstat.sh`, `symrmsd*.py`, `restart_fit.py`, `remap_*`), and the
archived branches (`research/archive/branches/*`, which hold only `tools/jgen/bench` precursors, `nonnci/run_arms.sh`, and the NCI v2 orchestrator history).

### 1.1 Capability table

"Coupled" means how tied the code is to one track's data model. Keep = base of the runkit module as is; Merge = take the idea or
code and combine it with others; Drop = leave in place, not carried over.

| # | Capability | Implementation(s) and location | Quality notes | Decision |
|---|---|---|---|---|
| 1 | Parallel xtb relaxation (crude opt, 1 thread per job, never raises) | A4 `a4/relax.py::xtb_opt`, `xtb_opt_many` (ThreadPool waiting on subprocesses, cycle cap, charge/uhf from parent, per-job tempdir, result dict with status/cycles/converged/wall_s); Gemini `diag/run_pkgc_relaxations.py::run_xtb_opt` (same call, max 3 workers, no cap); original `relax_2_2b` | A4 version is the best: failure recorded not raised, cycle cap, stderr tail, per-job wall. Tested only through `test_a4.py` stored-pool test. General (symbols, X, charge, spin). Uses `/tmp`-free workroot arg. | **Keep A4**; add timeout, optional cache, process-group kill |
| 2 | xtb gradient / single point adapter | Codex `nnc/engines.py::XtbGradient` (version check 6.7.x, binary sha256, ALPB option, parse of `gradient` file, finite checks, 180 s timeout, `--parallel 1`) | Clean and strict, raises on failure (right for NN, wrong for batch loops, so wrap). Coupled to NN metadata only through `.metadata`. | **Merge** into `xtb` module as `grad()`; metadata (binary hash, version) goes to every xtb result |
| 3 | xtb provenance / silent MMFF fallback guard | A4 `run.py` (unit invalid if `embed_fallback != 0` or `xtb_path` not ending `/xtb`); Gemini `run.py::_assert_gfn2` (embeds acetate, asserts geometry moved); `relax.xtb_path()` (SystemExit if not resolvable); Codex golden notes (needs pixi PATH) | Three partial guards. The behavioural self-check (`_assert_gfn2`) plus a log handler for "GFN2 embed refinement failed" is the robust pair. Gemini's `_InfeasibleCounter` shows how to install a logging handler. | **Merge** into one `xtb.require()` (resolve, version, self-check) and a `FallbackWatch` log handler that fails loudly |
| 4 | Energy cache for relax/MACE results | A4 `run.py` `--reuse-relax` (stored `X_relaxed`, `E_rel`, `E_chk` in unit npz, asserts reps equal); Gemini `ref_mace_cache.json` use; Codex `dataset.py` content-addressed store with sha256 check; `nci_v2_orchestrator` `get_cached_ref_mace_energy` and `should_skip_task` (provenance-hash skip) | Four ad hoc forms. Codex's content addressing (key = hash of symbols, coordinates bytes, charge, spin, engine id and version) is the general one; orchestrator's provenance-hash skip is the resume form. | **Merge**: `cache.py` keyed by content hash, atomic write; A4 `--reuse-relax` becomes a cache hit |
| 5 | MACE batched evaluation with dedup, float64, GPU/CPU | Gemini `run.py` (collect all conformers of all arms/seeds per parent, dedup by coordinate bytes, one `mace.energies()` call, `JACOBIK_MACE_BATCH_ATOMS=900`, `_batch_atoms`, memory fraction 0.80, no per-arm `empty_cache`); A4 `relax.mace_evaluator/mace_kj` (device/thread keyed singleton, CPU ~0.65 s/structure, `--mace-device`); Codex `engines.MaceGas` (CPU, forces, model sha256) | Gemini's is the fast path (part of 4.1x). A4 is the general API. Codex adds forces and hash. All use `MACEPolarEvaluator`; none unit-tested in isolation. | **Merge**: A4 API shape, Gemini dedup/batch, Codex hash and optional forces |
| 6 | Multi-arm batching (shared load_mol, builder, free-ligand GFN2 cache, seed-row prep) | Gemini `run.py --arms-spec` | Excellent but fused into the runner `main()` (609 lines); depends on `inputs.load_mol`, `build_builder_and_inputs`, arm profile snapshots. Coupled to bench. | **Keep in bench**; runkit supplies the generic parts (prefetch, dedup, scale-invariance hook), not the solver glue |
| 7 | CPU background prefetch (parent i+1 while GPU solves i) | Gemini `_prefetch_parent` + single-thread `ThreadPoolExecutor` | Small, works (warms `_GFN2_CACHE`). The pattern is generic: `Prefetcher(fn, items, depth)`. Warm-up body is bench-specific. | **Merge** into `pipeline.Prefetcher` |
| 8 | Scale-invariance fast path (reuse control rows if arm flags are inactive for the parent) | Gemini `_has_active_scales` | Specific to dof_step_scale / bend_release flags. | **Drop** from runkit (stays in bench); expose a generic "equivalent-arm" hook only |
| 9 | cuSOLVER eigh chunking | Gemini `multiseed.safe_cuda_eigh_chunking` (scope-limited monkeypatch, chunk 4096) | Already in shared `jdev/multiseed.py`; correct and general for any big CUDA batch. | **Keep in place**; runkit re-exports for non-solver users |
| 10 | Replay-parity harness | Gemini `bench/replay_parity.py` (326 lines) | Part of gating, not orchestration. | **Drop** (stays a bench tool) |
| 11 | Producer/consumer watch with shard, skip file, done file | A4 `run.py` `--watch --shard K/N --skip-file --done-file --poll`, `_ready_parents` (3 seeds complete and newest file older than `settle` s), `crc32(code) % N` sharding | Works in production (A4 run). Generic idea (ready-detection by completeness plus mtime settle) is the key piece. Tied to `*_s{seed}.npz` names. | **Keep** as `pipeline.Watcher`, with the naming pattern and completeness rule as parameters |
| 12 | Cost-balanced split local vs VM | A4 plan: "4 local workers take a fixed 25 % share of cal parents by xtb cost (atoms^2)", `skip_local.txt`, scratch `local_codes.csv`, `natoms.json`; atoms^2 cost model from A4_REPORT (3.2 s at 68 atoms, 7.0 s at 99) | Manual and done once; cost model is fitted but not stored. | **Merge**: `pipeline.split_by_cost(items, cost_fn, capacities)` with cost from measured per-core speed |
| 13 | Colab VM lifecycle | `cj.py`: `colab()` retry/timeout wrapper, `remote_py/sh`, `balance`, `guard_credits` (cap 180 since 530.79), `build_bundle` (git archive + sanitised CSVs, cached per commit), `push` (chunked, sha256), `up` (blocks ~10 min, fingerprint gate), `run` (detached, `CJ_EXIT`), `wait`, `tail` (hangs), `get`, `down`; worktree copies: `MAX_VMS=4`, fingerprint `711411157fb4`, scaled_v1 files shipped | Solid and shared. **The main checkout copy is stale** (one VM, fingerprint `485e943af036`, no scaled_v1 files); all worktrees carry the newer copy. No tests. `up` and `tail` are the known pain points. | **Keep cj.py as the substrate**; runkit wraps it (see 14 to 17) |
| 14 | Async up with READY detection, poll without blocking | Codex `colab_gate.py` (drives `cj.colab/push/remote_py` via importlib, polls `NNC_GATE_ENV_READY`, `try/finally cmd_down`); A4 session ran `up` in background; auditor `vmstat.sh` | Codex's try/finally down is the right shape. Nobody has a clean async `up`. | **Merge**: `colab.Session` context manager (up in a thread/subprocess, READY marker, `finally down`) |
| 15 | Fetch, count, then down guard | `cj.get` + manual counting (A4 `local_feed.py`: repeated get every 150 s, `testzip` integrity per npz, atomic copy `.tmp`+`os.replace`, `--stop-file`, last fetch after stop); Codex `colab_gate` (fetch on exit) | `local_feed` covers incremental sync with integrity; the "expected count" check exists only by habit. | **Merge**: `colab.fetch_verified(expected)` then `down`; `local_feed` becomes `pipeline.RemoteFeed` |
| 16 | Status monitor on a VM | Auditor `vmstat.sh` (one `colab exec` that prints pool count, parents done/failed, log tails, `uptime`) | Hard-coded paths of one session, but the one-exec status call is the right design (one round trip). | **Merge** as `colab.status(session, probes=[...])` |
| 17 | Idle watchdog, cost projection, stop rule | Nowhere. Credits only checked at `up`/`run`/`wait`; "idle VM is the most common failure" | Gap. | **New** |
| 18 | Local process control (CPU split, thread env, kill by PID) | Codex `panel_local.py` (workers 1..3, env `CUDA_VISIBLE_DEVICES=""`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `JACOBIK_NUM_WORKERS=1`, per-parent timeout, `stop_owned` kills the process tree via psutil, resumable via completeness check, ledger); A4 and Gemini use ad hoc nohup and `feed.pid`/`localrun.pid`; `AGENTS.md` quotas | Codex's supervisor is the best; its tree kill is by PID. Hard-coded 1..3 worker limit and NN records. | **Merge**: `resources.Budget` + `resources.spawn()` + `resources.kill_tree(pid)` |
| 19 | GPU single-job lock | Rule only (`nvidia-smi` first); `torch.cuda.set_per_process_memory_fraction(0.80)` in Gemini runner; Codex `gpu_gate.isolated()` (spawn a process per measurement with timeout so OOM cannot contaminate later work) | No lock exists. | **New** (`flock` on a file in the main checkout's run dir) plus **Merge** Codex `isolated()` |
| 20 | Registry of own processes | PID files `feed.pid`, `pilot.pid`, `localrun.pid` in scratchpad | Informal. | **New**: JSON registry per agent; `kill` refuses PIDs not in it (no `pkill -f/-x`) |
| 21 | Symmetry-aware RMSD, automorphisms, clustering | A4 `a4/geom.py` (VF2 automorphisms, identity first, cap 4000; exact all-pairs below threshold with radial and Frobenius lower bounds, one GEMM for covariances; 335 s to 16 s per pool; tested by `test_a4.py`); auditor `symrmsd.py`/`symrmsd2.py` (20 s over 81 parents, the origin); `src/jacobik/ik/dedup.py` (production, RDKit matches cap 20000); `nnc/relaxation.aligned_rmsd` (explicit loop); `builder/atom_map.py` (cap 512) | Four RMSD implementations. A4 is vectorised and tested. Production `dedup.py` must not change. Item §7.5/§7a-H (factored automorphisms) will replace enumeration. | **Keep A4 `geom`** as `runkit.geom` facade; the planned symmetry pass lands underneath (see 3.7) |
| 22 | Timing traces, stage profile | A4 `prof` dict with `tick()` (autos_ctx, rmsd_ref, dedup, labels, xtb_wall, xtb_cpu_s, mace); `jdev/bench_workers.py` (groups/h at 1,2,3 workers, per-stage embed/seed_rows/solver/dedup/MACE, projection from measured rates); `diag/d06_stage_timing.py`; `JACOBIK_TIMING_TRACE=1` (roadmap 5.2); Codex `cost.measure` (eager vs CUDA graph step time, peak bytes, capture time, replay parity assert, repeats); gate "estimate > 15 min then defer" (`gpu_gate.main`) | Pieces are good, none shared. `gpu_gate`'s defer-if-over-budget rule is exactly the owner's "estimate before a panel run". | **Merge** into `profile.py` (named `runprofile` to avoid clashing with `jacobik.profile`) |
| 23 | Wall and cost projection | A4_REPORT table (1.2 h one VM; 0.9 h with 4 local), `bench_workers` projection, Gemini 1080 units in 40.6 min; manual | Ad hoc arithmetic each time. | **New**: `project()` from measured per-item cost, cores, GPU rate, VM price (1.5 units/h L4) |
| 24 | CUDA graph capture harness, stream ownership | Codex `nnc/cost.py::measure` (non-default stream for warm-up and capture, `assert_close` replay vs eager) | NN-specific callbacks; the capture pattern is generic and carefully done. | **Merge** the capture helper into `profile.gpu_step_time()`; the rest stays in nnc |
| 25 | Static pair chunks with recomputation (`torch.utils.checkpoint`) | Codex `nnc/chunked.py::ChunkedResidual` | Model-specific (ContactFeatures). | **Drop** (stays in nnc) |
| 26 | Evaluation-counted engine calls with caps | Codex `nnc/relaxation.py` (`evaluations` counter incl. failed calls and line-search probes, geometry-bytes cache, `EvaluationCap`) | Good pattern; coupled to tangent-space relax. | **Merge** the counter+cache wrapper as `engine.Counted` |
| 27 | Fail-closed locked-set exclusion | Codex `nnc/policy.py::DataPolicy` (loads `locked_exclusions.json`, requires non-empty mandatory partitions, `assert_allowed(codes)` returns sha256 provenance, `smoke_limit` 1..5 parents); hard-coded absolute path | Exactly the guard every runner needs (`LOCKED_SETS.md`). Path should be resolved relative to the main checkout. | **Keep** as `runkit.policy`, called by every runner entry point |
| 28 | Atomic, content-addressed result storage | Codex `nnc/dataset.py::SmokeStore` (sha256 per file, checksum verify on load); A4 `.tmp`+`os.replace`; orchestrator `TaskPaths` | Mixed. | **Merge** into `cache.atomic_write()` and `cache.verify()` |
| 29 | Bootstrap CI and paired statistics | Gemini `bench/metrics.py`, Codex `monitors.OnsetMonitor.summary` (parent-clustered, one-sided), A4 `aggregate*.py` | Required by AGENTS.md; belongs to analysis not orchestration. | **Out of scope** for runkit v1 (offer a small `stats.parent_bootstrap`) |
| 30 | Long-lived solve workers with strict profile (orchestrator) | `research/refit/nci_v2_orchestrator.py` (`WorkerProcessController`, `LongLivedWorker`, profile asserted per task, provenance-hash skip, failure ledger) and `solver` `_get_pool` (spawn pool, `JACOBIK_NUM_WORKERS`) | Largest piece (1800 lines); NCI v2 specific, but its patterns (spawn with env set before import, assert snapshot) are what `resources.spawn` must honour. | **Drop** (reference only); copy the "env before import" rule |
| 31 | Bundling and read-only data shipping with sanitised CSV | `cj.build_bundle`, Codex `colab_gate` (explicit file list + manifest sha256 per file) | cj's is the maintained one; Codex's manifest idea (sha per file) is a good addition. | **Merge** manifest into `colab.bundle` |
| 32 | Env setup (`PATH`, `PYTHONPATH`, `JACOBIK_XTB`, `LD_LIBRARY_PATH`) | Gemini `run.py::_setup_env`, `cj.REMOTE_ENV`, AGENTS.md block | Three copies. | **Merge** into `resources.env()` |

### 1.2 Lessons the module must encode (each maps to code)

| Lesson | Where it is enforced |
|---|---|
| xtb PATH trap (silent MMFF fallback) | `xtb.require()` self-check and `FallbackWatch` log handler; every pool or relax result carries `xtb_path` and `embed_fallback`; consumers mark the unit invalid |
| `/tmp` is RAM-backed | `resources.workroot()` returns a disk path under the output dir (`<out>/_xtbtmp`) and refuses `/tmp`/`/dev/shm` unless `allow_ram=True` and size is small |
| one local GPU job | `resources.gpu_lock()` (flock, refuses if `nvidia-smi` shows another compute process) |
| kill by PID only | `resources.kill_tree(pid)` only accepts PIDs from the caller's registry, never matches by name |
| fetch, count, then down | `colab.Session.finish(expected_files=...)`; down is in `finally`, and a failed count keeps the VM only if `keep_on_mismatch=True` (default: still down after a second fetch attempt, because an idle VM costs more) |
| `up` blocks, `tail` hangs | `up` runs in a subprocess with a log file and READY marker; status uses one-shot `exec` with a hard timeout, never `tail` |
| L4 vCPUs about 2x slower per core | `pipeline.split_by_cost` takes measured per-core speed per machine type; defaults to 0.5 for Colab until measured; CPU-heavy default placement is local |
| MACE on CPU is the bottleneck | `mace.energies()` defaults to CUDA when available, batches by atoms, dedups by coordinate bytes |
| deterministic embed needs OMP=1 | `resources.env(threads=1)` sets `OMP/MKL/OPENBLAS=1`; xtb calls inherit `OMP_STACKSIZE` |
| thread count and affinity change reduction order | `resources.env()` records `torch.get_num_threads()`, affinity, `JACOBIK_NUM_WORKERS` in the run manifest; `Budget` never sets affinity unless asked; golden check runs in a stock environment (Codex note) |
| GPU results platform-dependent; pair controls | `manifest.platform` (GPU name, driver, torch); `pipeline` refuses to compare manifests from different platforms unless `force` |
| CPU split Gemini 5, A-track 4, Codex 3 | `resources.Budget.from_agentsmd(agent)` reads a small `budget.json` kept next to the lock dir; `spawn()` refuses beyond quota |

### 1.3 Findings worth the owner's attention

1. The main checkout (`n3b`) `cj.py` is stale: one VM, fingerprint `485e943af036`, no `scaled_v1`/`pilot_v1` files. AGENTS.md says to run `cj.py` from the main checkout. Every worktree has the newer file (`MAX_VMS=4`, `711411157fb4`). One canonical copy is needed.
2. A4 shares `cj.py` through `local_feed.py` with hard-coded paths into the `jgen-a` worktree.
3. Nobody enforces the GPU lock, the CPU quota or the idle-VM stop; all three are conventions only.
4. There are four RMSD implementations and four "skip if done" mechanisms.

## 2. Package location

`research/bench/tools/runkit/`, imported as `tools.runkit` (the same root as `tools.jgen` and `tools.jdev`, already on
`PYTHONPATH=src:.:research/bench`, and already shipped to Colab inside `research/`). Reasons: it is development tooling, it
stays out of `src/jacobik/`, so the production solver never imports it and the fingerprint `711411157fb4` cannot change (the
fingerprint covers `profile.FIELDS` and source of the solver, not `research/`). A top-level `src/jacobik_runkit/` would be
picked up by the `src` path used by the solver and risks the "not imported by production" rule being broken by accident.

Rules for the package: pure stdlib + numpy (+ psutil if present); torch, rdkit, mace imported lazily inside the functions that need
them; `import tools.runkit` must work with no GPU, no xtb and no network; no module imports `jacobik` at import time except
`tools.runkit.xtb.require()` and the Colab fingerprint gate, which do it lazily.

Tests live in `research/bench/tools/runkit/tests/` (pytest, same style as `a4/test_a4.py`).

## 3. Modules

### 3.1 `resources`
- Starts from: Codex `panel_local.py` (supervisor, `stop_owned`, env block), Gemini `_setup_env`, `cj.REMOTE_ENV`.
- Merges: A4/Gemini PID-file habit into a registry; Codex `gpu_gate.isolated` for per-measurement spawn with timeout.
- API: `env(threads=1, cuda=None, xtb=None) -> dict` (single place for PATH, LD_LIBRARY_PATH, `JACOBIK_XTB`, PYTHONPATH, thread vars);
  `Budget(agent, cpus)`; `spawn(cmd, budget, tag, log, env=None) -> Proc` (counts against quota, writes registry);
  `kill_tree(pid)` (psutil children first, terminate, wait 10 s, kill; refuses unregistered PIDs); `gpu_lock(timeout)` context manager;
  `workroot(out)`; `manifest()` (host, cores, affinity, torch threads, GPU name, driver, torch/mace versions, xtb path and version, git commit, profile fingerprint).
- Quota source: a plain JSON file (`budget.json`) with the three agent shares; the owner can change it without a code change.

### 3.2 `xtb`
- Starts from: A4 `relax.xtb_opt/xtb_opt_many`.
- Merges: Codex `XtbGradient` parsing and metadata (binary sha256, version), Gemini `_assert_gfn2` and `_InfeasibleCounter`, A4 provenance check.
- API: `require() -> info` (resolve binary, version, behavioural self-check once per process, raises SystemExit with the explanation);
  `FallbackWatch` (logging handler that records "GFN2 embed refinement failed" and exposes `.hit`); `opt(...)` and `opt_many(jobs, workers, workroot, cycles, cache=None, timeout=...)` returning result dicts
  (`status, E_h, cycles, converged, X, wall_s, stderr_tail, cached`) and never raising; `grad(...)` and `energy(...)`.
- Failure records go to `failures.jsonl` in the workroot parent (code, job id, status, stderr tail).
- Timing: per-job wall and the CPU-seconds total, returned in a `Timing` object consumed by `runprofile`.

### 3.3 `mace`
- Starts from: A4 `mace_evaluator/mace_kj` (singleton per device and threads).
- Merges: Gemini batch/dedup/memory fraction (`JACOBIK_MACE_BATCH_ATOMS`), Codex model hash and optional forces.
- API: `Evaluator(device="auto", dtype="float64", batch_atoms=900)`; `.energies_kj(structures) -> ndarray` with coordinate-bytes dedup
  (hash of symbols, rounded-free exact bytes, charge, spin) and cache hook; `.properties(...)` for forces; `.model_sha256`.
  On CPU it warns once with the measured s/structure so callers can see the cost; on GPU it calls `empty_cache` only on OOM retry (halve the batch and retry, record it).
- Parity: a test that dedup does not change values (bit-identical to the undeduplicated call on a toy set).

### 3.4 `pipeline`
- Starts from: A4 `run.py` watch loop (`_ready_parents`, shard, skip file, done file, poll) and `local_feed.py`.
- Merges: Gemini `_prefetch_parent` (generic `Prefetcher`), Codex resumable completeness check and ledger, A4 atomic copy and zip integrity check.
- API:
  - `Watcher(dir, pattern, group_key, complete=lambda files: len(files)>=3, settle=8, shard="K/N", skip_file=None, done_file=None, poll=20)` as an iterator of ready groups, with a `done` set persisted to disk.
  - `Prefetcher(fn, depth=1)` background thread for load-ahead.
  - `split_by_cost(items, cost, machines) -> {machine: [items]}` using `machines = [Machine("local", cores=4, speed=1.0), Machine("vm", cores=12, speed=0.5)]`; writes the skip file.
  - `RemoteFeed(session, remote, local, codes, every=150, validate=zip_ok)` (A4 `local_feed` generalised; no hard-coded worktree paths).
  - `run_groups(fn, groups, workers, ledger)` that records failures and continues (AGENTS.md "loops never fail the batch").

### 3.5 `colab`
- Starts from: `cj.py` (kept unchanged as the substrate and as the CLI), reached by `importlib` from `research/gpuwork/colab/cj.py` of the same checkout (Codex's method), so the shipped file is the one under test.
- Merges: Codex `colab_gate` (`try/finally` down, READY marker poll, per-file manifest hashes), A4 session patterns, auditor `vmstat.sh` (one-exec status).
- API: `Session(name, gpu="L4", repo, ref, kind="gpu"|"cpu")` context manager:
  `up()` starts `cj.py up` as a subprocess with its own log and returns at the `READY:` line (or raises on the fingerprint failure and downs the VM);
  `run(tag, cmd)`, `status(probes)` (single exec, hard timeout), `wait(tag, max_min)`, `fetch_verified(remote, local, expected_count)`,
  `finish()` = fetch, count, then `down`; `__exit__` always downs, even after exceptions, and writes a spend line (credits before/after).
  `Watchdog(session, idle_min=20)` samples `uptime`/GPU use through `status()`; if the VM is idle (no running job, GPU 0 %) for the interval it fetches, verifies and downs, and writes a notice.
  `cost(session)` reports hours and units; `stop_rule(projected_units, cap)` raises before launching if the projection exceeds the owner's cap.
- CPU runtimes: `Session(kind="cpu")` calls `colab new` without `--gpu`. Cores, per-core speed (a 10 s xtb benchmark through `xtb.opt` on a fixed small molecule) and price are measured at first use and stored in `machines.json`. This is a Phase 2 probe (needs a VM), flagged in section 8.

### 3.6 `runprofile`
- Starts from: A4 `tick()` stage profile and `bench_workers.py` stage/throughput table.
- Merges: Codex `cost.measure` capture harness (stream ownership, replay-vs-eager `assert_close`), `gpu_gate` "defer if estimate over budget".
- API: `Trace` (named stages, wall and CPU seconds, `with trace("xtb"):`), `measure_items(fn, items[:2])` (profile 1 to 2 cases), `project(n_items, per_item_s, workers, speed, overlap=...) -> Projection(wall_s, units)`, `require_budget(projection, max_min=15)` (the "estimate before panel run" rule; returns a refusal record rather than running), `gpu_step_time(fn, var)` (eager vs graph).
- Writes `profile.json` beside each run output.

### 3.7 `geom`
- Starts from: A4 `a4/geom.py` unchanged in behaviour (the vectorised, bounded RMSD; 335 s to 16 s per pool). Facade exports `automorphisms`, `sym_rmsd_to_ref`, `sym_rmsd_matrix`, `cluster_single_linkage`, `ligand_adjacency`, `donors_of`.
- Merges: auditor `symrmsd2.py` as a regression reference (same numbers on the 81 census parents), Codex `aligned_rmsd` as the slow oracle in tests.
- Coordination with the planned symmetry pass (`PLAN_jgen.md` §7 item 5 and §7a-H, factored automorphisms with orbits/generators and a Hungarian assignment over identical ligands, Burnside count, exactness check): `runkit.geom` is **the consumer-facing interface only**. The factored implementation is written by the symmetry subagent against the same function signatures, with the A4 enumeration as the exactness oracle ("identical results where no cap triggers", item 5 step 2c). Until then `geom` re-exports A4 code; when the pass lands, `automorphisms()` returns a group object that still exposes `.P` (the explicit array) for small groups, so callers do not change. Phase 2 does not touch `src/jacobik/ik/dedup.py` or `builder/atom_map.py`.

### 3.8 `policy` and `cache` (small)
- `policy`: Codex `DataPolicy`, path resolved from the main checkout, with `assert_panel(panel in {"tail","cal"})`; refuses `val`, reserve, locked codes; provenance dict into every manifest.
- `cache`: content-addressed store (`key = sha256(engine id|version|symbols|X bytes|charge|spin|options)`), atomic write (`.tmp` then `os.replace`), checksum verify on read, JSONL ledger.

## 4. API sketches

### 4.1 Solve panel (GPU, local, Gemini-style)
```python
from tools.runkit import resources, xtb, mace, pipeline, runprofile as rp, policy

env = resources.env(threads=1, xtb="auto")              # PATH, LD_LIBRARY_PATH, JACOBIK_XTB
xtb.require()                                            # raises if MMFF fallback would happen
policy.assert_panel("tail", codes)                       # tail/cal only, locked sets refused
proj = rp.project(n_items=len(codes)*3*len(arms), per_item_s=rp.measure_items(solve_one, codes[:2]))
rp.require_budget(proj, max_min=60)                      # refuses and records if over budget
with resources.gpu_lock(), xtb.FallbackWatch() as fw:
    ev = mace.Evaluator(device="cuda", batch_atoms=900)
    for mol, builder in pipeline.Prefetcher(prefetch_parent, codes):    # CPU ahead of GPU
        rows = solve_all_arms(builder)                                   # bench-specific
        E = ev.energies_kj(all_structures(rows))                         # deduplicated single pass
        write(rows, E, manifest=resources.manifest(extra={"fallback": fw.hit}))
```

### 4.2 Relax and metrics (CPU heavy, GPU energy pass)
```python
budget = resources.Budget("A4", cpus=4)
w = pipeline.Watcher(pool_dir, "*_s*.npz", group_key=code_of, shard="0/2", skip_file=skip, done_file=done)
cache = cache_mod.Store(out/"cache")
for code, files in w:                                          # yields parents as pools complete
    jobs = make_jobs(files)                                    # (symbols, X, charge, spin)
    res = xtb.opt_many(jobs, workers=budget.cpus, workroot=resources.workroot(out), cycles=200, cache=cache)
    E = mace.Evaluator("auto").energies_kj(relaxed_structures(res))   # GPU when present, else CPU with a warning
    metrics_for(code, files, res, E)                           # unchanged A4 code, now using runkit.geom
```

### 4.3 Split local + Colab
```python
machines = [pipeline.Machine("local", cores=4, speed=1.0),
            pipeline.Machine("vm", cores=12, speed=0.5)]       # speed measured, not assumed, after first parents
split = pipeline.split_by_cost(codes, cost=lambda c: natoms[c]**2, machines=machines)
with colab.Session("jgA1", gpu="L4", repo=wt, ref="HEAD") as vm:     # up() returns at READY; __exit__ always downs
    colab.stop_rule(rp.project(...).units, cap_units=8)
    vm.run("solve", "python -m tools.jgen bench run ... --pool-dir /content/out/pool")
    feed = pipeline.RemoteFeed(vm, "/content/out/pool", local_pool, codes=split["local"], every=150)
    local = resources.spawn([...a4.run --watch ...], budget, tag="a4-local")
    vm.wait("solve", max_min=120); feed.stop(); local.wait()
    vm.finish(expected_files=n_expected)                   # fetch, verify count, down; watchdog also armed
```

## 5. Migration

- Phase 2 adds files only (`research/bench/tools/runkit/`, tests, one README). No existing script is edited, no other branch is touched, `src/jacobik/` is unchanged, `cj.py` is unchanged (a single follow-up: the owner decides which `cj.py` copy is canonical; see open question 1).
- Adoption is opt-in per owner, at their own checkpoint:
  1. **New work first.** Anything written after merge imports `tools.runkit`.
  2. **A4** (lowest-risk, highest gain): swap `a4/relax.py` for `runkit.xtb` + `runkit.mace` (identical call line, so results are bit-identical; verified by a test that runs both on the same 3 structures), `a4/geom.py` for the facade, `local_feed.py` for `pipeline.RemoteFeed`. Their `run.py` flags (`--watch --shard --skip-file --done-file`) stay; they map to `Watcher`.
  3. **Gemini:** swap the MACE block for `mace.Evaluator` (same dedup) and `_prefetch_parent` for `Prefetcher`; `--arms-spec`, scale-invariance and replay parity stay in the bench. Output must match the Package C control (zero difference on the 10-parent tail panel; checked once at the end of the package, per the light-gates rule).
  4. **Codex:** `panel_local.py` becomes `resources.spawn` with `Budget("nn0", 3)`; `colab_gate.py` becomes `colab.Session`; `engines.XtbGradient` is wrapped; `policy` moves over unchanged.
  5. **Paper/auditor:** use `vmstat`-style `colab.status` and `resources.manifest()` for run provenance.
- While running jobs exist (today), nothing here touches their processes, VMs or files. The registry is per agent and starts empty; nothing is retro-registered.
- Compatibility rule: each runkit function that replaces an existing one has a test that calls both and compares outputs, so owners can switch with evidence. The old copy then stays until its owner deletes it.

## 6. Tests

Unit (no GPU, no xtb unless marked; all fast):
- `resources`: env contents; quota refusal; `kill_tree` refuses an unregistered PID, kills a spawned `sleep` tree by PID; `workroot` refuses `/tmp`; gpu lock is exclusive (two processes).
- `xtb`: `require()` raises with a fake binary that does not move the geometry; `FallbackWatch` catches the log line; `opt` never raises (bad binary, timeout, non-converged mock); cycle cap; cache hit skips the call; `grad` parsing from a canned `gradient` file.
- `mace`: dedup does not change values on a mock evaluator; OOM retry halves the batch; CPU warning.
- `pipeline`: Watcher readiness (3 seeds and settle), shard disjointness over `crc32`, skip file re-read, done file termination, resume after a crash; `split_by_cost` balance property; `RemoteFeed` rejects a truncated zip; failure in one group does not stop the loop.
- `colab` (with a fake `cj` module): READY detection, `__exit__` downs on exception, count mismatch path, watchdog fires, `stop_rule` refuses.
- `runprofile`: projection arithmetic and `require_budget` refusal record.
- `geom`: A4's three RMSD tests re-run through the facade; agreement with `aligned_rmsd` oracle on a small complex.
- `policy`: refuses a `val` code and a reserve code; refuses an empty partition file.

Smoke (CPU, 1 to 2 parents from `tail`/`cal`, ours: one cal and one tail parent chosen from the manifest, never `val`/reserve):
build or load one pool already on disk (read-only), run `xtb.opt_many` on 3 rows with 1 to 2 workers (cycle cap 30, runs seconds), `mace.Evaluator("cpu")` on the reference and 3 rows, compare against the same calls in A4's `relax.py` (bit-identical required), print a `profile.json`. Local processes: at most 2.

## 7. Phase 2 effort estimate

| Part | Work |
|---|---|
| `resources`, `policy`, `cache` | 0.5 day |
| `xtb` (+ FallbackWatch, grad) | 0.5 day |
| `mace` (+ OOM retry, dedup tests) | 0.5 day |
| `pipeline` (Watcher, Prefetcher, split, RemoteFeed, run_groups) | 1 day |
| `colab` (Session, Watchdog, fetch_verified, status, fake-cj tests) | 1 day, plus 0.5 day of live checks on one VM (CPU runtime measurement included) |
| `runprofile`, `geom` facade | 0.5 day |
| README, usage notes, three example scripts | 0.5 day |
| Equivalence tests versus the old implementations and the CPU smoke run | 0.5 day |
| **Total** | **about 5 to 5.5 working days** of one Sonnet-class agent; 3.5 days without the live Colab checks |

Colab spend in Phase 2: one L4 VM for about 30 minutes (about 1 unit) to verify `Session.up`, `status`, `fetch_verified`, watchdog; one CPU runtime for 10 minutes to measure cores, speed and price. Both need owner approval.

## 8. Open questions for the owner

1. **Which `cj.py` is canonical?** The main checkout copy (the one AGENTS.md says to run) is stale (single VM, old fingerprint). Should the `8ae029b` version replace it on `n3b`, and should `cj.py` become a thin CLI over `runkit.colab` after Phase 2?
2. **Package location:** `research/bench/tools/runkit/` (proposed) or `src/jacobik_runkit/`? The first is safer for the production-fingerprint rule.
3. **Enforcement strength:** should `spawn`, `gpu_lock` and `Session` be hard (refuse) or soft (warn and log)? Proposal: hard for the GPU lock, kill-by-PID and the VM down-on-exit; soft (warn) for the CPU quota until agents have migrated.
4. **Where does the quota file live?** A single `budget.json` under the main checkout's `.claude/` (read-only for agents) or under `research/data/runkit/`? Who updates it when the split changes?
5. **CPU-only Colab runtimes:** may Phase 2 start one for a 10-minute measurement, and is there a credit price cap per hour for them?
6. **Idle-VM watchdog:** is automatic `down` after the idle interval (fetch first, then down) acceptable, or should it only alert?
7. **Scope:** keep bootstrap/CI statistics out of runkit (proposed), or share one `parent_bootstrap`?
8. **Symmetry pass:** confirm that `runkit.geom` is the interface and the factored implementation lands underneath, and which track owns that pass.
9. **Adoption deadline:** opt-in at each owner's next checkpoint (proposed), or mandatory for all new runs after merge?
