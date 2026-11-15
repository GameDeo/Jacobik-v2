# runkit Phase 2 (core): report for Checkpoint R1, 2026-10-28

Branch `infra/runkit`, rebased onto master `e96ed2b` (one commit on top: DESIGN.md), then Phase 2 commits. Package:
`research/bench/tools/runkit/` (usage: its `README.md`). Nothing under `src/` changed; `jacobik.profile.snapshot()["fingerprint"] == "615ae97d2350"` (checked in this worktree).

## What was built
`resources`, `policy`, `cache`, `xtb` (+ `FallbackWatch`, `grad`), `mace`, `pipeline` (`Watcher`, `Prefetcher`, `split_by_cost`, `RemoteFeed`, `run_groups`),
`colab` (`Session`, `Watchdog`, `fetch_verified`, `status`, `stop_rule`), `runprofile`, `geom` (A4 code behind the facade), `stats` (one `parent_bootstrap`),
`probe` (machine benchmark), `smoke_parity.py`, 3 examples, README. `research/data/runkit/budget.json` created with a documented schema
(Gemini 5, A4 4, Codex 3, runkit 4, GPU owner Gemini, Colab cap 180 / 4 VMs; hard: GPU, kill, vm_down; warn-only: CPU).

Policy detail the auditor should check: `assert_allowed` refuses every val and reserve code (strict, for new suites). `assert_panel("tail", ...)` allows the
five val parents inside the tail panel (IFUROQ, SOBHAS, XEYTUT, YUCZUS, ZELTAP), because AGENTS.md makes `tail` a development panel; reserve is never allowed, and
`val` or any other panel name is refused. The locked file is read from the main checkout (it is untracked, so absent in worktrees); `.claude/locked/` is never read.

## Tests (49 pass, about 10 s, CPU only: `python -m pytest research/bench/tools/runkit/tests -q`)
resources 7 (env, quota warn then refuse, kill_tree refuses unregistered PID and kills a real process tree by PID, GPU lock exclusive across two processes, owner rule, workroot refuses /tmp),
policy/cache/stats 6 (val and reserve refused, empty partition refused, tail constant equals `panels.TAIL`, cache corruption is a miss, `bootstrap_mean` draws identical to `bench/metrics.py`),
xtb 9 (fake binaries: `require` rejects one that does not move the geometry, `FallbackWatch`, never-raises on bad binary/timeout/no output/non-converged, cycle cap, charge and uhf in the command, cache hit skips the call, failures.jsonl, canned `gradient` parse; plus one real-xtb water test),
mace 4 (dedup bit-identical to the undeduplicated call and one model call, OOM halving, cache hook, CPU warning), pipeline 8, colab 8 (fake cj: READY detection, down after exception, down after failed up, never downs a VM it did not start, count ok/short, status, watchdog fires and resets, stop_rule),
runprofile 3, geom 3 (A4's RMSD tests through the facade plus a slow explicit-permutation oracle, agreement 1e-9).

## Smoke parity with A4 relax.py: bit-identical
`python -m tools.runkit.smoke_parity` on QEZVEY_s11 (cal, 99 atoms), 3 rows, cycle cap 30, 2 workers, MACE CPU. Policy check passed (cal).
xtb: status, E_h, cycles, converged and the relaxed geometry are exactly equal for all 3 rows (E_h -124.447887603894 / -124.446260202298 / -124.455552275091, 21/22/22 cycles).
MACE (ref + 3 relaxed rows): runkit (one call) vs A4 (two calls) max abs difference 0.0 kJ/mol. Model sha256 e4495612037b identical in both worktrees. Output: `research/data/runkit/smoke/{parity.json,profile.json}`.
(Not tested: MACE on CUDA and OOM against the real model; the OOM path is tested with a mock.)

## Colab
- `cj.py credits` before: 514.93 (0 active); after everything: 514.68 (0 active, verified with `colab usage`). Total spend about 0.25 unit.
- rk1 (L4, 0.245 h, 0.25 unit): `Session.up` returned at READY after 12.8 min (fingerprint 615ae97d2350 on the VM); `status()` (one exec) showed idle then job done and 5 files;
  a remote job ran `xtb.require()` from the shipped package; `fetch_verified` counted 5 of 5; `Watchdog(idle_min=1)` fired after 1.7 min: fetch, count 5 of 5, down. VM confirmed gone.
  Record: `research/data/runkit/live/live_gpu.json`, `colab/events.jsonl`.
- Bug found by the live run and fixed: `Session.down` set `downed=True` before the VM was stopped, so a caller could exit mid-stop. The VM did get stopped, but the "down" log line was lost. `downed` now becomes True only after the stop, with a lock; regression test added. Also the `up` log is now unbuffered.
- CPU-runtime probe (the one allowed): FAILED, nothing measured. The pixi env cannot be installed on a CPU-only runtime (`--frozen` lock needs a CUDA virtual package; "target this machine cannot run"). The failure path worked: `Session` downed the VM (0.1 h, about 0.01 unit). I added to the `cj.py` setup script `command -v nvidia-smi || export CONDA_OVERRIDE_CUDA=12.4` before `pixi install` (GPU path unchanged). **That fix is untested**; a second CPU probe needs the auditor's/owner's OK (I did not retry: "once"). Until then `machines.json` has only the local measurement (xtb 16-atom-class tripeptide, 0.46 s per opt, 12 cores); the Colab per-core speed stays the assumed 0.5.

## cj.py
Only three small edits, all in the master copy: `--gpu none` (CPU runtime), CPU-safe `get_device_name` in the setup check, the CUDA override above. It is **not** yet a thin CLI over `runkit.colab`; `Session` drives `cj.py` through its CLI (the other way round). Left for after the audit, to avoid refactoring the shared substrate while others use it.
The copy on `n3b` (main checkout) is stale (one VM, fingerprint 485e943af036); I did not touch it. Mark it stale in its README (not done by me; not my worktree).

## Not migrated / open
- No other agent's files were edited. `parent_bootstrap` copies to replace later: `jgen/bench/metrics.py::bootstrap_mean` (draws identical, tested), `jgen/a4/aggregate2.py::boot`, `jgen/diag/analyze_dof_screening.py::bootstrap_parent_ci`, `jgen/diag/analyze_pkgc_confirmation.py::parent_bootstrap_ci`; also an inline one in `nnc/monitors.py`. I did not import runkit in `bench/metrics.py` (Gemini's file, merge conflicts, no gain).
- A4 `run.py` (watch/shard) and `strain_run.py`, Gemini's runner glue, Codex's `colab_gate` and `panel_local` are not rewritten; runkit offers the pieces (README, Migration).
- `geom` is a verbatim copy of A4's file (jgen/a-census 4ad04de, sha a0880a68...) because A4's branch is not merged; after merge there will be two copies until A4 switches.
- `geom` imports rdkit/networkx at import time (DESIGN said lazy); the other modules import none of torch, rdkit, mace, jacobik at import.
- The CPU quota is warn-only by design; `registry/` is per agent and empty until agents use `spawn`.
- Stop and wait for the auditor: Phase 2b (trope suite) not started.

---

# Checkpoint R2: trope suite (Phase 2b), 2026-10-28

Code: `research/bench/tools/runkit/trope/` (`features.py`, `tags.py`, `cover.py`, `build_suite.py`, `suite.py`), tests `tests/test_trope.py`
(55 tests in all, pass). The suite itself: `research/runkit/trope_suite/suite.json`. Demo outputs: `research/runkit/trope_suite/demo/`.
Local processes: at most 2 throughout (auditor's limit; `budget.json` shows runkit 2 and A4 2, left as is for the auditor to restore). No GPU, no Colab for 2b, nothing under `src/`; fingerprint `615ae97d2350`.

## Parents and the cover
- Candidates: the 1,200 `surface_fit` + 120 `calibration` groups of the frame; groups with any member in `locked_exclusions.json` are dropped (0 groups needed dropping: val and reserve are separate splits). The chosen codes pass `policy.assert_allowed`; no val code is in the suite (SOBHAS and YUCZUS are not). All 1,320 were featurised (geometry, bond orders and SMILES only; no QM fields read), 0 errors.
- Tag rules (`tags.py`, version 1; full text in its docstring): chelate ring size of 4, 5, 6, 7, 8+ (metallacycle atoms = shortest ligand path between two donors + 2); haptic eta 2 to 6 (data-model haptic with a carbon donor; N,N "eta-2" amidinate-like pairs are excluded); macrocycle; bulky phosphine (P donor, three P-C bonds, each carbon with >= 2 further heavy neighbours); anionic donor by element class (halide, O, N, C, S subclass sets); carbene (NHC: donor C with >= 2 N and no H; other: C_carbene/C_alkylidene); CN 4/5/6 with the data model's geometry label (7 labels); lanthanide (La..Lu).
- Cover: greedy, rarest tags weigh most, small parents preferred, a fixed grid of (max atoms, atom penalty) searched for the fewest uncovered tags, then fewest parents, then fewest atoms. Result: **9 parents, 357 atoms, all 27 tags covered**:

| Parent | Atoms | Metal, CN, geometry | Tags it adds |
|---|---|---|---|
| SUQWEG | 62 | La, 4 tetrahedral | lanthanide, haptic eta3, eta5, cn4_tetrahedral |
| CAFWIQ | 42 | Pt, 6 trig. prism | macrocycle, haptic eta2, ring 5, anionic N, anionic halide, cn6_trig_pris |
| MUNSUJ | 25 | Fe, 4 tetrahedral | haptic eta4 |
| SIKNAE | 95 | Ni, 5 trig. bipyr. | bulky phosphine, anionic C, ring 4, 7, 8+, cn5_trig_bipyr |
| RIGWAJ | 28 | W, 6 octahedral | carbene (other), cn6_octahedral |
| YUNPIG | 23 | Ni, 4 square planar | anionic S, ring 6, cn4_square_plan |
| SUXROT | 33 | Hg, 4 see-saw | carbene (NHC), cn4_see_saw |
| CANXAQ | 26 | Cr, 4 tetrahedral | haptic eta6 |
| SISKAJ | 23 | Mo, 5 square pyr. | anionic O, cn5_square_pyr |

Rarest tags have few candidates (lanthanide 4, cn6_trig_pris 10, eta4 15, eta3 22): their coverage rests on one real parent each. All suite parents are `surface_fit`, which the DB has seen (not held out; this is a development filter).

## Footprint and per-trope deltas
`suite.py run` runs one parent per subprocess on CPU (`builder.solve(device="cpu")`, the golden check's call; 1 thread, seeds fixed, fingerprint recorded, MMFF-fallback watch), no relaxation. Per (parent, seed): top-1 score, dE_unrel (MACE CPU, top-1 minus reference, kJ/mol), a strain proxy (harmonic over the reference's bonds with generic force constants; a proxy, not A4's relaxation strain) and RMSD to the reference. SHA-256 digests of all scores and all geometries give the bit-identity test.
A change declares `{"tags": [...], "parents": [...]}` (and `fingerprint_change_ok` for a flag that changes the fingerprint). Seeds 11,12,13 run only for parents inside the footprint; the others run seed 11. `suite.py compare` requires bit-identical digests outside the footprint (else `VIOLATION`, exit 2), refuses to pair different hosts or fingerprints, and reports per-parent and per-trope deltas (arm minus control, seed means). With 9 parents no CI is claimed; the panels decide.

## What was run (all results below are checks of the mechanism, not evaluations of a flag)
- Cost: one suite pass is about 4.4 min of total process wall (261 s control), about 2.5 min wall with 2 processes.
- Determinism: control vs an identical repeat (`control_b`), same host: **all 9 parents bit-identical** (scores and geometry digests; deltas exactly 0.0).
- Footprint check, demo change `dof_step_scale {'ring': 0.5}` with footprint tags macrocycle, ring 7, ring 8+: outside the footprint **7 of 7 parents bit-identical**, no violation. Inside: CAFWIQ (macrocycle): d_score -0.008, d_dE_unrel -41.7 kJ/mol, d_strain +0.07; SIKNAE (ring 7, 8+): identical. The flag is a demonstration only; no conclusion about it is drawn. The cmp JSONs are in `trope_suite/demo/`.
- Not tested: a change that should affect a parent outside its declared footprint (the violation path is covered by a unit test with synthetic records); GPU-vs-CPU agreement; the suite against the golden file.

## Colab CPU-runtime probe (approved by the owner, one try)
Credits 514.68 before, 514.67 after (0.17 h, 0.01 unit); `Active assignments: 0` afterwards. It worked with the `CONDA_OVERRIDE_CUDA` setup change.
Result in `research/data/runkit/machines.json`: Colab CPU runtime has **2 cores**; single-core xtb speed **0.57x** of this machine (0.84 s vs 0.48 s per optimisation of a 60-atom peptide, same energy to 1e-10 Eh). The assumed 0.5 in `split_by_cost` is therefore about right; per-core is 0.57, and a CPU runtime has only 2 cores, so it is not useful for CPU-heavy work compared with the local 12 cores.

## Other items from the auditor's rulings
- `geom` now imports rdkit and networkx lazily (verified: neither is in `sys.modules` after `import tools.runkit.geom`).
- Bootstrap copies: runkit-owned code has none besides `runkit.stats`. Still to replace by their owners (not touched): `jgen/bench/metrics.py::bootstrap_mean`, `jgen/a4/aggregate2.py::boot`, `jgen/diag/analyze_dof_screening.py::bootstrap_parent_ci`, `jgen/diag/analyze_pkgc_confirmation.py::parent_bootstrap_ci`, and the inline one in `nnc/monitors.py`.
- Open: `cj.py` thin-CLI refactor stays deferred (auditor). Budget file is at runkit 2 / A4 2 until the auditor restores it.

---

# R2b: real strain per isomer bucket (auditor's required fix), 2026-10-28

Changes in `trope/suite.py` (tests updated, 55 pass; fingerprint 615ae97d2350; nothing under `src/`; 2 local processes; no GPU, no Colab).
- The bond-length proxy is renamed `md_bond_dev` (kept as a record column only; it is blind to bends and torsions and is not strain).
- Buckets are the solver's own donor-to-site assignment: `ConformerResult.assignment_idx`, with `embed_idx` for E/Z variants (key `(embed_idx, assignment_idx)`). Both are already exposed on every conformer, so **no solver capture was needed and nothing in `src/` changed**. The A4 v2 geometric label is not used. Per (parent, seed): the best-scoring row of each bucket, best 8 buckets (the top-1 row is always the first), `n_buckets` (existing) and `n_buckets_kept` recorded.
- Each selected row is relaxed with `runkit.xtb.opt` (GFN2 `--opt crude`, cycle cap 200). strain = E_MACE(row) - E_MACE(relaxed), offset = E_MACE(relaxed) - E_MACE(reference), CPU MACE float64, kJ/mol. Measurement only: the result rows and the digests stay unrelaxed (checked: CAFWIQ digests identical to the pre-change run). A relaxation failure is recorded (`relax_status`, `relax_stderr`) and the bucket gets strain None.
- `compare` inside the footprint reports per-parent top-1 deltas and per-bucket deltas (buckets matched by seed and key; unmatched counted) of score, dE_unrel, strain and offset, and per-trope means of both. Outside the footprint the bit-identity check is unchanged.

## Runs (2 processes, same host)
- Wall time per suite pass: control 199 s, control repeat 221 s, ring05 arm 241 s (was about 130 s for the unrelaxed suite on 2 processes; the relaxations add roughly 1.5 to 2x).
- Determinism: control vs control repeat, all 9 parents bit-identical (digests), and every derived delta (dE_unrel, strain, offset, per bucket) exactly 0.0, 0 unmatched buckets. The relaxation outputs are reproducible too.
- Footprint demo (`dof_step_scale {'ring': 0.5}`, footprint macrocycle, ring 7, ring 8+; a mechanism check, no verdict on the flag): outside the footprint 7 of 7 parents bit-identical. Inside: SIKNAE identical; CAFWIQ (macrocycle, 12 buckets, 8 kept, 3 seeds) top-1 d_score -0.0076, d_dE_unrel -41.7, **d_strain -61.8**, d_offset +20.1 kJ/mol; bucket means d_dE -23.6, d_strain -25.9, d_offset +2.3 (22 matched buckets, 2 unmatched). Per bucket the changes vary in sign and size (seed 11: top-1 bucket strain +174.0, bucket [0,10] +746.9, [0,11] -703.9), so a top-1-only reading hides most of the movement. Rows of the top-1 bucket of CAFWIQ have strain of order 760 kJ/mol (control), i.e. real strain is large for this macrocycle.
- Bucket counts (control, per seed): SUQWEG 1, CAFWIQ 12 (8 kept), MUNSUJ 1, SIKNAE 2, RIGWAJ 1, YUNPIG 2, SUXROT 4, CANXAQ 1, SISKAJ 12 (8 kept).

## Failure, stated plainly
SUQWEG (La, 62 atoms): xtb crashed (rc 1, Fortran backtrace, stderr in the record) on its top-1 row, so strain and offset are None for that parent in every arm; its dE_unrel and digests are fine. The row is badly distorted (md_bond_dev 11,811). I did not tune around it; the lanthanide tag therefore has no strain data. Options for the auditor: a second lanthanide parent, or relax from a steepest-descent pre-step (not done).
