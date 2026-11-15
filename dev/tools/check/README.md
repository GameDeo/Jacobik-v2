# Checks: which entry point to use (infra/fastcheck, 2026-11-08)

Agents use three entry points. Run them from the worktree root with the environment from AGENTS.md
(`PATH` = the pixi env, `PYTHONPATH=src:dev:.:research/bench`, one thread).

| Level | Command | Where | Target wall | What it checks |
|---|---|---|---|---|
| `ws` | `python -m tools.check ws` | workstation, 12 threads | <= 4 min | fingerprint at default; golden (2 profiles x 4 complex groups, each with the 8-part layout, run in parallel); pytest in 4 shards |
| `pm-quick` | `python -m tools.check pm-quick` (on the Perlmutter login node, from `~/repos/jacobik`) | login node, <= 32 threads, nice | 5-10 min | the `ws` stages against the Perlmutter references, plus the dev suite (control vs arm on the login A100, GPU phase <= 10 min) |
| `pm-full` | `python -m tools.check pm-full --shape c2b\|isomer [--qos NAME --approved]` | Perlmutter GPU node (BILLED) | C-2b ~29 min, isomer ~23 min (estimates) | one node, 4 runners (one per GPU, parents sharded by cost, every arm of a parent on one GPU), preflight, packed outputs per GPU (extract the four `g*/run_pack*.tar` into one directory for `d3/analyse.py` / the C-2b analyses). Without both `--qos` and `--approved` it writes the script and prints the node-hour estimate; `--test-only` runs `sbatch --test-only`. Not submitted by this package. |

Every level exits non-zero on any failing stage, prints a timing table with the load before and after each stage, and
writes a JSON report (`research/data/infra/fastcheck/<level>/<level>_<stamp>.json`, on Perlmutter under `$SCRATCH/jacobik/runs/`).
Options only choose paths, the worker counts (`--jobs`, `--shards`, `--workers`) and `--only` (a stage subset while
developing). They never change what a level checks.

Which one when:
- every change: targeted tests (`pytest dev/tests/<file>`), the fingerprint, and 1-3 parents (seconds to 2 min);
- every checkpoint: `ws` (or `pm-quick` for Perlmutter-side changes);
- once per package, before fold-in: panels and gates (not this directive).

Golden references: `research/data/golden/` (workstation, authoritative). Perlmutter references live in
`$SCRATCH/jacobik/data/golden_pm/5f74b14b57ed/`, recorded only from a commit that already passes the workstation golden,
with `python -m tools.check.golden_par record <empty dir>` (refuses to overwrite).

## Running parents: `python -m tools.run` (the one runner)
Every GPU (or CPU) run of parents goes through one scheduler: the bench, gates (parents x arms x seeds, C-2b / D3
shape), isomer-style screens, Colab and Perlmutter jobs.

| Command | What |
|---|---|
| `python -m tools.run estimate <selection>` | wall estimate (recorded per-parent seconds) and the predicted lane count; runs nothing |
| `python -m tools.run panel --out DIR <selection>` | one job; asks for `--yes` when the estimate exceeds 20 min |
| `python -m tools.run broker --spool DIR` | the long-lived GPU owner: serves every job submitted to the spool, interleaved at parent granularity |
| `python -m tools.run submit --spool DIR --out DIR <selection>` / `status --spool DIR` | queue a job for the broker / show the queue |
| `python -m tools.run sample --out DIR -- <command>` | the same telemetry around any existing runner (for before/after) |
| `python -m tools.run compare --new A/C --ref B/C --seeds ...` | top-1 identity between two record directories |

Selection: `--codes` / `--manifest` / `--c2-panel`; `--arms C,P` (C-2 table) or `--arms-spec JSON` (any flags, per-arm
`out`); `--seeds`; `--backend bench|isomer`; `--repo CHECKOUT` (run an agent's worktree code); `--env K=V` (e.g.
`JACOBIK_DB`); `--device cuda|cpu`; `--cpu-identity`. Scheduling: `--lanes N` (fixed) or adaptive with `--max-lanes`,
`--max-rss-gb`, `--cpu-cap`; `--parent-cap-min` (per-parent wall budget cap, default 15); `--max-wall-min`.

How it works (dev/tools/run/orchestrator.py docstring has the detail):
- **Lanes**: persistent processes, each bound to one job (one checkout, one import-time profile, one DB), each loading
  torch, MACE, the DB and the GFN2 self-check once, then running whole parents on request exactly as the job's own
  sequential runner does. Several lanes interleave their kernels on the GPU and fill each other's CPU phases.
- **Resource-aware admission**: a lane is added only after the previous one has finished a parent (peak RSS and GPU
  memory measured), and only while free CPUs (cgroup quota / affinity minus other users' load), MemAvailable (cgroup
  capped) minus headroom, the optional own-RSS cap, and free GPU memory cover it, and the GPU is below 90 % busy.
  Under memory pressure the newest lane is retired; under critical pressure killed and its parent requeued.
- **Giants first**, per-parent wall budget (3 x estimate, clamped to [3, 15] min): an overrun is recorded as status
  `timeout` and the run moves on. Heartbeat every 60 s per lane (parent, stage, elapsed) in `run_events.jsonl`.
- **Records**: exactly the job's own runner's records (bench: plus `c2_timing` / `c2_identity` computed as the C-2b
  driver does, so `d3/analyse.py` and the C-2b analyses read them), plus `run_timing`. Resume skips finished parents
  (also inside packed runs). `--pack` (default on Perlmutter) packs all loose outputs into one `run_pack<N>.tar`.
- **Telemetry**: `telemetry.jsonl` (GPU util and memory, own RSS, MemAvailable, load, lanes) and a one-line summary.

### How existing runners map onto it
| Existing runner | Equivalent | Notes |
|---|---|---|
| `python -m tools.jgen bench run --panel named --only X --arms-spec S ...` (incl. D-B's `--device cpu --cpu-identity`) | `python -m tools.run panel --codes X --arms-spec S --repo <worktree> --env JACOBIK_DB=<db> [--device cpu --cpu-identity] --no-deterministic --out DIR` | bench backend; same records. D-B on the GPU: drop `--device cpu --cpu-identity` (GPU numbers differ from CPU identity runs: pair arms on the same device) |
| `tools.jgen.c2.driver` (C-2b gates) | `python -m tools.run panel --c2-panel --arms C,K,P,KU,S1 --seeds 21,22,23` | deterministic algorithms on (default), c2_timing/c2_identity as the driver; the driver's frozen-contract checks are not repeated (run the driver's checks or keep the driver for frozen confirmations) |
| `research/campaigns/i/run_i_screen.py` (isomer) | `python -m tools.run panel --backend isomer --isomer-script <path> --repo <jgen-i worktree> --c2-panel --arms C,A,AS,ASP --seeds 11,12,13` | the script is imported, not modified; one seed of a parent is one work item (its seeds are independent standalone solves); its GPU lock is replaced by the scheduler |
| `tools.jgen.d3.run` / `d3.worker` (D3 on Colab L4) | bench backend with the D3 arm identities as `--arms-spec` and `--env JACOBIK_DB=...`, one job per arm DB | not switched: D3's identity checks and enrichment (`enrich_record`) stay in the harness; proposal: its `drive_lockstep` dispatches to tools.run lanes |
| `research/gpuwork/colab/cj.py` jobs | run `python -m tools.run panel` (or `broker`) on the VM | cj.py keeps VM up/down/fetch |
| `dev/tools/check` pm-quick dev suite | uses `tools.run` (Scheduler) directly | |

## Modules
| File | Role |
|---|---|
| `__main__.py` | CLI and the `ws` level; the `fingerprint`, `golden` and `pytest` stages; `ws` sizes itself from MemAvailable and CPU (prints the plan, refuses if nothing fits) |
| `golden_par.py` | golden in parallel by complex group; same arrays, tolerances and reference files as `research/bench/tools/jdev/golden.py` (not modified) |
| `pytest_shard.py` | pytest in shards (balanced by file size) read from JUnit XML; no pytest-xdist (not in the lock) |
| `devsuite.py` | dev suite manifest builder (size-stratified, cal + tail, locked codes excluded) |
| `pm_quick.py` | Perlmutter login-node level; GPU gate (refuses if busy, bounded wait); nice; caps; dev suite through tools.run |
| `pm_full.py` | billed-job script builder (dry run only here) |
| `common.py` | repo paths, environment, stage timing, report, resource plan |
| `DEV_MANIFEST.json`, `DEV_MANIFEST.sha256` | the dev suite manifest (39 parents, giants XUKKIA and QEZVEY set aside), fixed by sha256 |
| `../run/` | the unified runner (`resources.py`, `telemetry.py`, `lane.py`, `orchestrator.py`, `compare.py`) |

## Phase 2 (2026-11-08): where the GPU time goes, and the levers

Measured with `python -m tools.run.profile_parent` (the lane's own bench path, deterministic on, C arm, seeds 21-23,
warm run; outputs under `research/data/infra/fastcheck/phase2/measure/`). Local RTX 2060 SUPER.

| Parent | Solver B | Parent wall | Step loop | Output dedup + assembly | MACE | Prepare | GPU during step loop |
|---|---|---|---|---|---|---|---|
| AWONOR (median) | 768 (3 x 256) | 13.1 s | 7.2 s (55 %) | 0.9 s | 4.0 s (31 %) | 0.4 s | 106 W, 86 % util, 1935 MHz |
| QEZVEY (giant) | 4608 | 201 s | 45 s (22 %) | 116 s (58 %, one CPU thread) | 27 s (13 %) | 0.9 s | 135 W, 98 % util |

Inside AWONOR's step loop: every step replays (2 CUDA graphs, 198 of 200 steps); **7,718 kernels per step** (graph
A 1,145 + graph B 6,573), about 4.5 us per kernel; **1.4 host syncs per step** (the eigh status check between the two
graphs, plus the early-stop check every 8 steps). Split: graph A 6.6 ms, eigh 0.3 ms, graph B 27 ms per step.
Step time vs rows (same parent, more seeds): 21 / 35 / 75 ms at B = 256 / 768 / 2304, i.e. a fixed floor of about
14 ms per step plus 0.026 ms per row.

**Bound.** Median parents: kernel-latency bound in the GPU step loop (thousands of small kernels; the fixed per-step
floor is about 40 % of a step at B = 768), not host-sync bound and not CPU-bound. Giants: CPU-bound in the
single-threaded symmetry-aware output dedup (`StructureDeduper._full_any` via `multiseed.assemble_batched`), with the
GPU at 18-40 W. The 50-60 W seen in nvidia-smi is the average over dedup, MACE and CPU phases; during solves the
2060 draws 90-135 W. Solver batches are 384-6912 rows (C-2b records, median 768), not ~32 (32 is the pool size).
The login A100 runs parents at about the 2060's speed (244 s vs 268 s recorded on the same 12 parents): the step floor
is kernel latency, and the CPU stages do not get faster.

### Levers measured
| Lever | Result | Status |
|---|---|---|
| CUDA MPS (`--mps`) | 2060, 10 parents, C x 3 seeds: 1 lane 184 s off / 186 s on; 2 lanes 138 / 126 s; 3 lanes 138 / 117 s (-15 %). Login A100, 12 parents: 4 lanes 118 / 96 s (-19 %), 8 lanes 113 / 98 s. Top-1 identical 29-30/30 (2060), 35-36/36 (A100) vs 1 lane | runner option, off by default locally; pm-full uses it; private daemon, always stopped (no leftover processes after any run) |
| Arm packing (`--pack-arms`) | CPU identity: XOJHIN, ZOJDEH, DOHTOM (768-2304 packed rows), arms C/K/P x 2 seeds, 18/18 records bit-identical (E/Z parent TEDPOI not packed; it fails under --cpu-identity in both modes, as before). 2060, 8 parents x C/K/P x 3 seeds: top-1 identical 72/72; solve phase -20 to -35 % (AWONOR 28.1 -> 20.9 s), parent wall only 377 -> 361 s (-4 %): MACE, a second prepare and CPU stages dominate | runner option, off by default; worth it only together with the other stages fixed |
| Host syncs / replay coverage | 1.4 syncs/step, every step already replayed: little to gain (eigh check 0.3 ms/step) | candidates only (below), not changed |
| Batched MACE across parents | not built; MACE already runs in 900-atom batches per parent; estimate below | estimate only |

### Cross-row operations (batch axis) in the solver and scorer, and how each is segmented
| Operation | Where | Segmented? |
|---|---|---|
| Row cull: `_cull_key` (argsort, stable sort by niche signature, cummax ranks, top-k score gap) + `topk(key, ceil(keep x B))`; enabled only when B >= 256 | `ik_solver._solve_impl` (CPU; CUDA replay never culls) | **not segmented**: packing is allowed only where culling is off (CUDA replay, `--cpu-identity`) |
| CPU chunking: `n_workers = min(n, B // 128)` chunks; each chunk culls on its own | `JointIKSolver.solve` | not segmented: packing refused on CPU without `--cpu-identity` |
| Early stop: `FinishPhase.stop = done.all()` (every 8 steps, a host sync); `FinishSchedule.stop` likewise (default off) | `finish_phase.update`, both loops | batch-wide; exact anyway: done rows take zero steps and keep their best snapshot (a packed solve may run more steps) |
| Trace / `SR.LAST` (`steps_executed`, `early_stop_saved_steps`) | solver diagnostics, record fields | batch-wide values; differ under packing (diagnostics only) |
| Gradient of `sum over rows` | `autograd.grad` | exact per row (backward of a sum) |
| Q3 mode commitment, plateau / dual stall rings, Adam / iRprop state, best snapshot | solver | per row |
| `_agg_norm()`, per-candidate Q1/Q2 tables indexed by `row_groups`, Q6 pair lists | scorer | per scorer (= per parent) / per row |
| Output window (score >= best - window), dedup, assembly | `engine.output_window`, `multiseed.assemble_batched` | per seed slice after the solve (per parent) |
| OOM retry grouping of seeds (`max_rows`) | bench | packing falls back to the unpacked solve on any exception |

Different parents cannot share one tensor without padding along atoms, ligands / kinematic trees, DOF widths, donors
and mixture components; every reduction over a padded axis changes its floating-point summation tree. Direct test
(torch CPU, one thread): padding with zeros changed `sum` in 25-50 % of rows and `-inf` padding changed `logsumexp`
in some rows, for term counts 5-300. Exact per-parent identity is therefore not reachable for cross-parent packing;
a ragged multi-parent solver (each parent's own kernels, one graph) keeps identity but not the kernel count, so it
only helps through concurrency (MPS / streams). Estimated effort for a padded multi-parent solver with per-parent
segments: several days (FK, geometry, scorer Q1-Q7, shape fit, symmetry groups), and it could only promise top-1
agreement, not bit identity.

`--pack-arms` packs what is exact: the arms (and seeds) of one parent that share every non-batch shape and the
solve-time settings (C-2b: C, K, P, KU; not S1, not E/Z parents). See `dev/tools/run/armpack.py`.

### Candidates not changed (production code; owner: report only)
- Output dedup for giants (`StructureDeduper._full_any`, CPU, one thread): 58 % of QEZVEY's wall. A GPU path exists
  for the engine's own assembly (`select_distinct`); the bench's `assemble_batched` uses the CPU deduper.
- eigh status sync between the two graphs: 1 sync per step (0.3 ms of a 35 ms step on AWONOR); removing it would
  allow one graph per step, worth little at B >= 768.
- Kernel count: 7.7k kernels per step; fusing (e.g. `torch.compile` of the score) is the lever for the 14 ms floor,
  but it changes the arithmetic and needs its own identity contract.

### Output dedup (bench `multiseed.assemble_batched`), 2026-11-08
The bench dedups each seed's rows on the CPU, one thread, one seed after the other, whatever the solve device.
Changed (bench code only, `research/bench/tools/jdev/multiseed.py`; production `jacobik.ik.dedup` untouched):
- CUDA solves of complexes with >= 1000 automorphisms dedup on the solver's GPU (same algorithm, float64; CPU fallback
  on any GPU error; `JACOBIK_BENCH_DEDUP_DEVICE=cpu` keeps the CPU). Below 1000 the GPU loses (128 maps: 1.2 s CPU vs
  3.9 s GPU).
- Otherwise the seeds' dedups run in 3 threads (`JACOBIK_DEDUP_SEED_THREADS`); each computes exactly what the
  sequential loop computed.
Identity: `python -m tools.run.dedup_bench` on 26 recorded dedup calls of 6 parents (AWONOR 32 maps, XUKKIA 128,
BUXJIO 192, FEKKAJ 1536, GOTWET and QEZVEY 20000): identical kept lists (same rows, same order) for the CPU
reference, 3 seed threads, 4 and 8 intra-op threads, and the GPU. Per-seed wall (2060 / Ryzen 3600): QEZVEY 36-45 s ->
3.6 s, GOTWET 4.0 -> 0.7 s, FEKKAJ 2.7 -> 0.7 s. End to end (2060, C x 3 seeds, 1 lane): 15/15 records
bit-identical (all pool rows); QEZVEY 206.6 -> 109.2 s, GOTWET 54.5 -> 44.6 s, FEKKAJ 84.2 -> 79.6 s, AWONOR 14.5 -> 14.2 s;
the 5 parents 446 -> 335 s.
Production candidate (not changed): `dedup.DEDUP_DEVICE` defaults to "cpu" for CUDA solves; the same threshold rule
would help the engine's own `select_distinct`.

### MACE as its own batched phase: measured, not built
MACE-POLAR-1-S float64 on recorded pool structures (2118 structures of 6 parents, `python -m tools.run.mace_bench`):
login A100 116 / 133 / 106 / 92 structures/s at 900 / 2700 / 8100 / 24300-atom batches; parents mixed in one call:
114 / 110 structures/s at 900 / 8100. Larger or mixed batches gain at most 15 % on the MACE part, and MACE is 7.6 % of
the A100 parent wall (12 parents, 1 lane), so a separate cross-parent MACE phase would save about 1 % of a gate on the
gate platform, while it needs the bench's record writing (energies, floors, pool exports, coverage) re-implemented
outside the bench. Energies agree to 1.5e-11 eV between batch sizes and between the 2060 and the A100 (float64).
On the 2060, MACE is 30 % of a median parent (float64 at 1/32 rate); there lanes overlap it with other lanes' solves.

### Login A100: lanes x MPS (same 12 parents, C x 3 seeds, deterministic off)
| Lanes | MPS off | MPS on |
|---|---|---|
| 1 | 252 s, 75 W mean, 28 % busy | 247 s, 77 W |
| 4 | 118 s, 109 W, 66 % | 96 s, 121 W (265 W peak) |
| 8 | 113 s, 118 W, 71 %, 21 GB RSS | 98 s, 121 W, 20 GB RSS |
At 4 and 8 lanes the wall is set by the longest parents (XUKKIA 62-85 s under contention), so 12 parents cannot show
more; on the login node (30 GB) lanes are RAM-bound at about 10 (2.7 GB each). Top-1 identical 35-36/36 per config
(one GPU run-to-run difference, as on L4). The A100 draws 75-120 W mean of 400 W: the kernel-latency floor, as on
the 2060.

## Not done (stated plainly)
- Batched MACE across parents: MACE still runs once per parent inside each lane; lanes overlap it with other lanes'
  solves instead. A dedicated MACE stage needs the record writing split out of the bench loop.
- Several parents in flight inside one process (threads / CUDA streams): not done; the runners set module globals per
  arm and per parent, so in-process concurrency races. Lanes are processes (about 2.5 GB RSS each).
- pytest-xdist: not added to the lock; sharding uses the existing environment.
- `pm-full` submits nothing in this package.

## Proposed AGENTS.md changes (not applied)
"Validation tiers": every change = targeted tests + fingerprint + 1-3 parents (seconds to 2 min); every checkpoint =
`python -m tools.check ws` (or `pm-quick` on Perlmutter); once per package = panels and gates through `tools.run`.

"GPU": replace "one GPU job at a time; the local GPU belongs to <agent>" with: the local GPU is owned by one
`python -m tools.run broker --spool research/data/runs/spool` process; agents `submit` jobs (panel, arms, seeds, flags,
DB, their worktree as `--repo`) and never start their own GPU processes; the broker sizes lanes from free RAM, CPU and
GPU memory. On Perlmutter: one broker per GPU. Pairing rule unchanged: arms of a comparison run in the same job.

"Gates" (Phase 2): a gate (C-2b / D3 / isomer-screen shape) runs on ONE Perlmutter GPU node (4 x A100), all arms of
the comparison on that node (`python -m tools.check pm-full --shape c2b|isomer`, which co-locates every arm of a
parent on one GPU), submitted only with the owner's go and the QOS named (`--qos NAME --approved`). The local 2060
is for dev runs and the dev suite only. Judge GPU use by power draw, not utilisation %.

## Inventory and archive proposals
See the section below (written 2026-11-08). Nothing listed there was moved or deleted. Items marked "in use" are
used by the D or isomer agents' branches; check those before any move.

### Inventory (as found on 2026-11-08, master bb5f084 + infra/fastcheck)

| Location | What it holds | Status | Proposal |
|---|---|---|---|
| `research/bench/tools/jdev/golden.py` | golden check (2 profiles, sequential) | authoritative check | keep; `tools.check ws` runs the same arrays in parallel |
| `research/bench/tools/jdev/` (other 20 modules: `bench.py`, `bench_workers.py`, `multiseed.py`, `cases.py`, `start.py`, `env.py`, `measure.py`, `diagnose.py`, `viz.py`, `cuda_check.py`, `foldin_check.py`, `archive_db_leftovers.py`, `p4_compare.py`, `p5_archive.py`) | jdev runners and helpers used by golden, tests and the replay generator | in use (golden, tests, budget) | keep; `archive_db_leftovers.py`, `p4_compare.py`, `p5_archive.py` are one-off migrations: archive candidates after a check that no open branch imports them |
| `dev/tools/jdev/build_replay_solver.py` | canonical generator for `solver_replay.py` | keep (regenerate, never hand-edit) | keep |
| `research/bench/tools/jgen/` top level (84 .py) | jgen analysis, runners and one-off `analyze_*`, `d_*`, `g3*`, `h3*`, `e2_*`, `run_*` scripts | mixed | 41 of 84 are referenced by nothing else on master: archive candidates (list below) |
| `research/bench/tools/jgen/{bench,c2,d3,diag,a4b,a4t,embed_study}/` | the bench runner (`bench/run.py`), C-2b (`c2/`), D3 harness (`d3/`), diagnostics | in use (bench, C-2b, D3) | keep; `diag/d*.py` are case diagnostics, archive after the D track closes |
| `research/bench/tools/jgen/d3/` | D3 gate harness | in use (D track) | keep |
| `research/bench/tools/runkit/` | run toolkit (`cache`, `colab`, `mace`, `pipeline`, `policy`, `probe`, `resources`, `runprofile`, `smoke_parity`, `stats`, `xtb`, `trope`, `tests`, `examples`) | in use by the infra agent | keep; `trope/`, `examples/` look like scratch: check before any move |
| `research/gpuwork/colab/cj.py` | Colab VM helper | in use (jgA/jgB/jgD VMs) | keep; its timing output could feed the fast runner's stage timing (not done) |
| `research/campaigns/` (`*.py` and `*_PROGRESS.md` / `*_REPORT.md`) | campaign scripts and reports | records, mostly archive material | archive the `archive/` subfolder; keep reports |
| `~/scripts/jacobik/` (Perlmutter) | login helpers (`jk_*`) | in use | `jk_check.sh` now calls `pm-quick` for the quick check; `jk_smoke.sbatch` pytest working directory fixed (see below) |

Archive candidates (no references on master; not moved; the D and isomer branches may still import them, check first):
`research/bench/tools/jgen/`: `analyze_g1_g3`, `analyze_h3b`, `d_counterfactual_analyse`, `d_counterfactual_q6_q7_run`,
`d_counterfactual_q7_analyse`, `d_counterfactual_q7_run`, `d_counterfactual_run`, `d_f1_order_check`, `d_f1_scan`,
`d_mace_smoke`, `d_q6_crystal`, `d_q7_a4b_check`, `d_q7_crystal_table`, `evaluate_g3a2_filter`, `evaluate_r2_energy`,
`finalize_g3a2_diagnostics`, `g3d_convergence144`, `g3d_resume_step0`, `g3d_screen_report`, `g3d_select_rows`,
`g3d_xojhin_trace`, `g3e_cpu_batch`, `g3e_twist_components`, `g3e_twist_diagnostic`, `g3e_twist_report`, `g3f_colab`,
`g3f_identity`, `g3f_noise`, `g3f_noise_report`, `g3f_report`, `g3g_golden_summary`, `invalidate_g3a2_fallbacks`,
`relax_o2_trope_top1`, `render_g3c0_report`, `repair_g3_reference_energies`, `run_bend_distribution`, `run_g3_a1_sanity`,
`run_g3_golden`, `run_g3_trope`, `run_g3a2_continuous_diagnostics`, `run_g3a2_diagnostics`, `run_g3a2_filter`,
`run_g3c0`, `run_g3prime_r2`, `run_g3prime_sanity`, `run_o1_g2_selected_relax`, `run_o2_trope_screen`,
`summarize_g3_sanity`, `summarize_g3_trope`, `summarize_g3a2_filter`, `summarize_g3prime_sanity`, `summarize_o2_trope_screen`.
Also `research/campaigns/archive/` (already an archive folder) and `research/bench/tools/jgen/__pycache__` (generated).
Not proposed: `research/bench/tools/jgen/c2/`, `d3/`, `bench/` (in use), `diag/` (see above).
