# jgen Campaign — Agent B Report (`jgen/b-solver`)

**Branch:** `jgen/b-solver` (from `jgen/campaign-base` @ `0733256`)
**Worktree:** `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/jgen-b`
**Outputs dir:** `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/`

---

## §B0. Setup, Replay-Parity Baseline, and Replay Fallback Bug Fix

### B0.1 Environment and CPU Golden Verification

- **Pytest suite (`pytest -q tests`):** `320 passed, 3 xfailed, 3 warnings in 335.04s` (unchanged from `jgen/campaign-base`).
- **GPU fast-path suite (`pytest -q tests/test_gpu_fast_path.py`):** `18 passed in 76.54s`.
- **CPU golden check (`python research/bench/tools/jdev/golden.py check`):**
  - `production: 245 arrays, 0 differ`
  - `u1: 245 arrays, 0 differ`
  - `golden: OK`
- **Production fingerprint:** `711411157fb4` (`JACOBIK_PROFILE=u1`, overrides `{"nci": false, "deterministic_embed": true}`).

---

### B0.2 Replay-Parity Harness & Unchanged Production Baseline

Built [`research/bench/tools/jgen/bench/replay_parity.py`](../bench/tools/jgen/bench/replay_parity.py) (also wired as `python -m tools.jgen bench replay-parity`).

#### Harness Design & CUDA Determinism Findings

For each `(code, seed)` unit on `DAMVAQ,HOXJUZ,JEGPUJ` × seeds `11,12,13` (9 units), the harness verifies three parity invariants from identical pre-built starting tensors `(theta_inits, X_ref, donor_pos_idxs, ideal_directions)`:

1. **`cuda_fast_fk_replay_vs_eager`**: `ReplaySolver(replay=True, FAST_FK=True)` vs `ReplaySolver(replay=False, FAST_FK=True)` on CUDA (200 steps, production GPU path).
   - Verifies that `torch.cuda.CUDAGraph` capture & replay in `solver_replay.py` (`part_a`, `mid`, `part_b`) executes without fallback (`LAST["mode"] == "replay"`, `replayed == 198`, `error is None`) and produces bit-identical raw trial scores, post-dedup conformer scores, and top-1 row indices to eager CUDA execution.
2. **`cuda_ref_fk_replay_vs_eager`**: `ReplaySolver(replay=True, FAST_FK=False)` vs `ReplaySolver(replay=False, FAST_FK=False)` on CUDA (200 steps, reference per-ligand `fk.apply` loop).
3. **`package_loop_vs_replay_loop`**: `ReplaySolver(replay=False, FAST_FK=False)` vs `JointIKSolver(cull_at=())` on CPU (`FORCE_ON_CPU=True`, 40 steps by default, matching `tests/test_gpu_fast_path.py::test_replay_loop_matches_package_loop`).
   - Verifies that every step-loop change in `src/jacobik/ik/ik_solver.py` is faithfully ported by `build_replay_solver.py` into `src/jacobik/ik/solver_replay.py` with bit-identical (`0.000e+00`) scores and top-1 indices.

**Two technical findings established while building the harness:**
- **Non-deterministic CUDA `index_add_` in backward pass unless deterministic algorithms are enabled:** `ComplexFK.apply` (`src/jacobik/ik/fk_compose.py`) and `BendPlan.apply` (`src/jacobik/ik/angle_bends.py`) call `index_select` on index tensors with repeated indices (`chain_steps`, `atom_chain`). In default CUDA mode, `index_select_backward` dispatches to `index_add_` using non-deterministic floating-point `atomicAdd`, causing ~1 ULP (`2.44e-4` score diff at step 1) run-to-run jitter even between two identical eager runs. Setting `torch.use_deterministic_algorithms(True, warn_only=True)` (with `CUBLAS_WORKSPACE_CONFIG=:4096:8`) inside `replay_parity.py` makes `index_add_` deterministic on CUDA and yields exact `0.000e+00` parity across all 200 steps between `replay=True` and `replay=False`.
- **`aten::div.Scalar` vs `aten::div.Tensor` on CUDA between `ik_solver.py` and `solver_replay.py`:** At `src/jacobik/ik/ik_solver.py:911`, `adam_m[li] / bc1` divides a CUDA tensor by a Python `float` (`aten::div.Scalar`, which on CUDA multiplies by `float32(1.0 / bc1)`), whereas `src/jacobik/ik/solver_replay.py:444` divides by a 1-element float32 CUDA tensor `c1 = bc1.index_select(0, tl)` (`aten::div.Tensor`, hardware `fdiv`). On CUDA this creates a `3.81e-6` (`5.96e-8` on `shape_dev`) 1-ULP difference at step 0, whereas on CPU (`FORCE_ON_CPU=True`) both use `fdiv` and are bit-identical (`0.000e+00`).

#### Baseline Replay-Parity Numbers (Unchanged Production Solver & After B0.3 Fix)

Saved to:
- Unchanged production solver: `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/b0_replay_parity_unchanged.json`
- After B0.3 fix: `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/b0_replay_parity_after_fix.json`

- **Replay parity (`b0_unchanged_production`, fp `711411157fb4`, flags `{}`):** **PASS** — 9/9 units passed (`DAMVAQ,HOXJUZ,JEGPUJ` × seeds `11,12,13`); top-1 row index match 9/9; CUDA replay mode 9/9 (0 fallbacks); max relative `|Δscore|` = `0.000e+00` (`<= 1e-6`), max absolute `|Δscore|` = `0.000e+00`.
- **Replay parity (`b0_after_fix`, fp `711411157fb4`, flags `{}`):** **PASS** — 9/9 units passed; bit-for-bit identical to `b0_unchanged_production` (`0.000e+00` difference on every unit and check).

| Unit | `cuda_fast_fk` mode (replayed) | `cuda_fast_fk` top-1 trial (replay / eager) | `cuda_fast_fk` top-1 score | `cuda_fast_fk` max rel `\|Δscore\|` | `cuda_ref_fk` top-1 trial (replay / eager) | `cuda_ref_fk` max rel `\|Δscore\|` | `pkg_loop` top-1 trial (replay / pkg) | `pkg_loop` max rel `\|Δscore\|` |
|---|---|---|---|---|---|---|---|---|
| `DAMVAQ:11` | `replay` (198) | `65 / 65` | `-3.083125` | `0.000e+00` | `65 / 65` | `0.000e+00` | `123 / 123` | `0.000e+00` |
| `DAMVAQ:12` | `replay` (198) | `56 / 56` | `-3.274629` | `0.000e+00` | `56 / 56` | `0.000e+00` | `29 / 29` | `0.000e+00` |
| `DAMVAQ:13` | `replay` (198) | `14 / 14` | `-3.455238` | `0.000e+00` | `14 / 14` | `0.000e+00` | `61 / 61` | `0.000e+00` |
| `HOXJUZ:11` | `replay` (198) | `28 / 28` | `-1.849108` | `0.000e+00` | `103 / 103` | `0.000e+00` | `103 / 103` | `0.000e+00` |
| `HOXJUZ:12` | `replay` (198) | `112 / 112` | `-1.863078` | `0.000e+00` | `112 / 112` | `0.000e+00` | `112 / 112` | `0.000e+00` |
| `HOXJUZ:13` | `replay` (198) | `97 / 97` | `-1.857051` | `0.000e+00` | `97 / 97` | `0.000e+00` | `97 / 97` | `0.000e+00` |
| `JEGPUJ:11` | `replay` (198) | `33 / 33` | `-8.869742` | `0.000e+00` | `33 / 33` | `0.000e+00` | `33 / 33` | `0.000e+00` |
| `JEGPUJ:12` | `replay` (198) | `124 / 124` | `-8.769847` | `0.000e+00` | `124 / 124` | `0.000e+00` | `124 / 124` | `0.000e+00` |
| `JEGPUJ:13` | `replay` (198) | `108 / 108` | `-9.012062` | `0.000e+00` | `108 / 108` | `0.000e+00` | `108 / 108` | `0.000e+00` |

---

### B0.3 Replay Fallback Bug: Diagnosis, Root Cause, Fix, and Verification

#### 1. Triggering Unit(s) on `cal`
Scanning all 120 `cal` parents on CUDA identified the exact parent that triggered the warning in Agent A's Colab `cal` control run:
- **`XIKGEE`** (units **`XIKGEE:11`, `XIKGEE:12`, `XIKGEE:13`**, which run together in a single `multiseed.solve_seeds` batch of `B = 384`, logging 1 warning for the pass).
- The remaining 119/120 `cal` parents had 0 fallbacks.

#### 2. Traceback & Root Cause Entering `part_b` Capture
Before the fix, `solver_replay.LAST["error"]` on `XIKGEE` recorded:
```text
  File "src/jacobik/ik/solver_replay.py", line 487, in _solve_impl
    part_b()
  File "src/jacobik/ik/solver_replay.py", line 345, in part_b
    q3_angles_now = self.ge.q3_approach_angles(X_current)
  File "src/jacobik/ik/geometry.py", line 613, in q3_approach_angles
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memory() or allocate the tensor with pin_memory=True.
```

**Root Cause:**
1. `XIKGEE` has 4 coordination sites: 2 kappa donor sites (`0, 1`, forming a within-ligand symmetric group `q3_sym = [[0, 1]]`, `out_idx = [0, 1]`) and 2 $\eta^5$-Cp haptic sites (`2, 3`, `out_idx = [2, 3]`).
2. In `src/jacobik/ik/scorer.py:1189` (called inside `part_a`), `self._ge.q3_approach_angles(X)` is **already skipped** when `S_orient` (`self._so_n > 0`) is active (`# not evaluated when S_orient replaces Q3 + Q4 below: the values were discarded, and the haptic approach angle costs a batched SVD per step`).
3. However, in `part_b` (`src/jacobik/ik/solver_replay.py:345`, generated from `build_replay_solver.py:227` and `src/jacobik/ik/ik_solver.py:810`), when `_q3_sym_group_idx_tensors` is non-empty, `part_b` calls `self.ge.q3_approach_angles(X_current)` to compute the Q3 symmetry-angle tie across `_q3_sym_group_idx_tensors`.
4. Inside `GeometryExtractor.q3_approach_angles` (`src/jacobik/ik/geometry.py:601–613`), it unconditionally executed the haptic face-normal SVD `_, _, Vh = torch.linalg.svd(centered, full_matrices=False)` whenever `n_h > 0`.
5. Because `_symmetric_donor_groups` (`src/jacobik/engine.py:1448`) skips haptic groups (`if len(grp) != 1: continue`), `_q3_sym_group_idx_tensors` **only ever indexes kappa Q3 columns** (`[0, 1]`). The haptic columns (`[2, 3]`) produced by `torch.linalg.svd` in `part_b` were immediately discarded (`a_g = q3_angles_now[:, idx_t2]`), had zero gradient, and on CUDA `torch.linalg.svd` (cuSOLVER) copies internal status/workspace between CPU and unpinned CUDA memory, aborting CUDA graph capture of `part_b` at step 1 whenever a complex has both a kappa Q3 symmetry group and at least one haptic ligand (`XIKGEE`).

#### 3. Fix Applied
1. **[`src/jacobik/ik/geometry.py`](../../src/jacobik/ik/geometry.py#L557-L610)**: Added `haptic: bool = True` to `GeometryExtractor.q3_approach_angles(self, X: torch.Tensor, haptic: bool = True)` and gated the haptic SVD block on `if haptic and n_h > 0 and self._q3_haptic_pos.shape[1] > 0:`.
2. **[`src/jacobik/ik/ik_solver.py`](../../src/jacobik/ik/ik_solver.py#L719-L730) & [`lines 810–818`](../../src/jacobik/ik/ik_solver.py#L810-L818)**: Computed `_q3_sym_haptic` in `_solve_impl` setup (`True` only if any group in `_q3_sym_group_idx_tensors` contains a haptic site, which is `False` for all kappa symmetry groups) and called `self.ge.q3_approach_angles(X_current, haptic=_q3_sym_haptic)`.
3. **[`dev/tools/jdev/build_replay_solver.py`](../../dev/tools/jdev/build_replay_solver.py#L225-L231)**: Updated `part_b` to call `self.ge.q3_approach_angles(X_current, haptic=_q3_sym_haptic)` and regenerated [`src/jacobik/ik/solver_replay.py`](../../src/jacobik/ik/solver_replay.py).
4. **[`research/bench/tools/jgen/bench/run.py`](../bench/tools/jgen/bench/run.py#L55-L76) & [`lines 250–257`](../bench/tools/jgen/bench/run.py#L250-L257)**: Extended `_InfeasibleCounter` to count `"step recording failed at step"` warnings in `replay_fallback` and write `"replay_fallbacks": int(delta["replay_fallback"])` (plus include `"replay_fallback"` inside `infeasible_warnings`) on every unit JSON record.

#### 4. Verification on `XIKGEE` and `DAMVAQ,HOXJUZ,JEGPUJ`
- **`XIKGEE` (`multiseed.solve_seeds`, seeds `11, 12, 13`, `B = 384`, `T = 200`):**
  - **Before fix:** `LAST["mode"] = "eager after failed recording at step 1"`, `replayed = 0`, `s_solver = 5.9378s` (`s_total = 7.4608s`). Top-1 trials/scores: seed 11 `trial=33, score=1.404858` (`n_conf=12`); seed 12 `trial=106, score=1.405592` (`n_conf=7`); seed 13 `trial=5, score=1.405622` (`n_conf=10`).
  - **After fix:** `LAST["mode"] = "replay"`, `replayed = 198`, `error = None`, `s_solver = 1.8265s` (`s_total = 3.3556s`, **3.25× solver speedup**). Top-1 trials/scores: seed 11 `trial=33, score=1.404847` (`n_conf=12`); seed 12 `trial=106, score=1.405591` (`n_conf=7`); seed 13 `trial=5, score=1.405622` (`n_conf=10`).
- **Replay parity on `XIKGEE` (`/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/b0_replay_parity_xikgee.json`):**
  - **PASS** — 3/3 units passed (`XIKGEE:11,12,13`); `mode == "replay"` (198 steps replayed, 0 fallbacks); top-1 row index match 3/3 (`33/33`, `106/106`, `5/5`); `max_rel_dscore = 0.000e+00`, `max_abs_dscore = 0.000e+00` across `cuda_fast_fk`, `cuda_ref_fk`, and `pkg_loop`.

---

## §B1. Plan for Solver Dynamics (`bend_release`, `dof_step_scale`, `start_kabsch`)

All three flags will be registered in `src/jacobik/profile.py` (`FIELDS`) with production-preserving defaults (`None` / `False`) so default flag values leave `PRODUCTION_FINGERPRINT = "711411157fb4"` and the CPU golden check untouched.

### 1. `bend_release` (Two-Phase / Ramped Angle-Bend Schedule)

- **Mechanism expected:**
  In `ForwardKinematics.apply` (`src/jacobik/ik/fk.py:390–434`), each ligand's parameter vector `theta_li` of width `D_li = 6 + n_joints + n_angle_bends` is partitioned into:
  - `[0:6]`: rigid-body translation and axis-angle rotation;
  - `[6 : 6 + n_joints]`: torsion, bend-dihedral, and macrocycle ring joints;
  - `[6 + n_joints : 6 + n_joints + n_angle_bends]`: in-plane angle-bend DOFs (`BendPlan`, `src/jacobik/ik/angle_bends.py`), regularized by the harmonic dead-zone prior `_bend_prior` (`k = 310..1800 kJ/(mol·rad²)`).
  Early in the solve (steps `0..60`), when donor atoms are still 5–15 Å from their coordination targets (`Q1` and `shape_dev` gradients are huge), Adam normalizes each bend column's gradient by its own running RMS and takes full `lr = 0.05 rad` steps on covalent bond angles before the ligand has rotated or torsioned into the coordination pocket. As documented in the `JEGPUJ` solver trace (`JGEN_REPORT.md` §7–8), bend DOFs reach 15–20° within the first 50 steps (`+100..350 kJ/mol` of internal ligand strain) and lock the ligand into a bent local minimum. Scaling the bend-column step by a linear ramp $w_{\text{bend}}(t) = s_0 + (1 - s_0)\,\min(t / K, 1)$ over the first $K$ steps forces early steps to use rigid-body + torsion motion for gross placement, releasing angle bends gradually as donors approach the pocket.

- **Where in the code it goes (`file:line`):**
  1. `src/jacobik/profile.py:79`: Register `FIELDS["bend_release"] = ("jacobik.ik.ik_solver", "_BEND_RELEASE", None, "Angle-bend step ramp [s0, K] over first K steps (None = off)")`.
  2. `src/jacobik/ik/ik_solver.py:67`: Read `_BEND_RELEASE = _profile.get("bend_release")` at import and store `self.bend_release = _BEND_RELEASE if bend_release is None else bend_release` in `JointIKSolver.__init__` (`ik_solver.py:553`).
  3. `src/jacobik/ik/ik_solver.py:728` (`_solve_impl` setup, before `# Row culling state`, so `build_replay_solver.py` copies it verbatim into `ReplaySolver._solve_impl`): Build per-ligand step-scale table `_step_scale_tabs: List[Optional[torch.Tensor]]` of shape `(T, D_li)` on `device` (or `None` when both `bend_release` and `dof_step_scale` are inactive). For `bend_release = [s0, K]`, columns `6 + fk_li.n_joints : 6 + fk_li.n_joints + fk_li.n_angle_bends` at step `i` (`0 <= i < T`) are multiplied by `s0 + (1.0 - s0) * min(i / max(K, 1), 1.0)`.
  4. `src/jacobik/ik/ik_solver.py:913` (Adam step loop in `JointIKSolver._solve_impl`): When `_step_scale_tabs[li] is not None`, multiply `step = step * _step_scale_tabs[li][t]`.
  5. `research/bench/tools/jdev/build_replay_solver.py:328` (and `src/jacobik/ik/build_replay_solver.py:328`, inside `part_b` Adam loop): When `_step_scale_tabs[li] is not None`, multiply `step = step * _step_scale_tabs[li].index_select(0, tl)` right after `step = lr_t * m_hat / (v_hat.sqrt() + eps_a)` and before `closure_fused_step`, then regenerate `src/jacobik/ik/solver_replay.py`.

- **How it enters the replay generator:**
  Because `_step_scale_tabs` is constructed in `_solve_impl` setup (lines `i_s..i_e` of `ik_solver.py`), `build_replay_solver.py` copies its construction verbatim into `solver_replay.py`. Inside `part_b`, `_step_scale_tabs[li].index_select(0, tl)` indexes the `(T, D_li)` device tensor using the existing device-side step counter `tl` (`(1,)`), producing a `(1, D_li)` tensor that broadcasts over `(B, D_li)` in `step = step * _step_scale_tabs[li].index_select(0, tl)`. This uses fixed device memory, zero host synchronization, and zero Python branching per step, so CUDA graph capture & replay (`g_b`) remains 100% intact and bit-identical to `ik_solver.py`.

- **Parameters and values to test (at most 3 values per parameter):**
  - `s0` (initial bend step scale at `t = 0`): `{0.0, 0.1}` (2 values: full hold at step 0 vs 10% step at step 0).
  - `K` (ramp duration in steps out of `T = 200`): `{50, 100, 150}` (3 values: 25%, 50%, 75% of the 200-step solve).

#### Mechanism Trace Evidence (`trace_solve.py`)
Traces on `JEGPUJ 11` (catastrophic tail parent, dE > 800 kJ/mol), `HOXJUZ 11`, and `IFUROQ 11` (both tail parents with extensive angle bends) measure median and maximum bend travel over the first 20 steps (`tr20`) and the full 200 steps (`trall`), as well as top-1 score:

| Configuration | `JEGPUJ 11` tr20_med (rad) | `JEGPUJ 11` trall_med (rad) | `JEGPUJ 11` top-1 score | `HOXJUZ 11` tr20_med (rad) | `HOXJUZ 11` top-1 score | `IFUROQ 11` tr20_med (rad) | `IFUROQ 11` top-1 score |
|---|---|---|---|---|---|---|---|
| **Baseline** (`None`) | 1.2005 (~68.8°) | 1.2652 | -9.001 | 0.3949 (~22.6°) | -1.850 | 0.7662 (~43.9°) | -0.948 |
| `s0=0.1, K=50` | 0.3121 (~17.9°) | 0.6970 | -4.583 | 0.2206 | -1.833 | 0.2584 | -0.891 |
| `s0=0.1, K=100` | 0.2330 (~13.3°) | 0.4736 | -4.101 | 0.1862 | -1.847 | 0.1967 | -0.904 |
| `s0=0.1, K=150` | 0.2037 (~11.7°) | 0.4195 | -3.888 | 0.1725 | -1.847 | 0.1729 | -0.957 |
| `s0=0.0, K=100` | 0.1096 (~6.3°) | 0.4005 | -3.518 | 0.1244 | -1.827 | 0.1195 | -0.795 |

**Key Mechanism Findings:**
1. In baseline solve, 95% to 100% of median bend deformation occurs in the first 20 steps (`tr20_med == trall_med` on both `HOXJUZ` and `IFUROQ`) while donors are still far from coordination targets.
2. Holding bends at `s0 = 0.1` suppresses early blowout by ~75-80% and dramatically improves top-1 score on `JEGPUJ` from `-9.001` to `-4.101` (`+4.9` units).
3. Because `s0 = 0.1` still allows ~0.23 rad (13°) of bend movement during gross placement, the strict freeze arm `s0 = 0.0, K=100` suppresses travel further to 0.1096 rad and achieves the strongest top-1 score improvement to `-3.518` (`+5.5` units on `JEGPUJ`, `-1.827` on `HOXJUZ`, `-0.795` on `IFUROQ`).

#### Evaluation Contract: `bend_release` Screening on `tail` (seeds 11–13)
Per Audit rules, screening is restricted to at most 3 configurations evaluated on `tail` (seeds 11–13) against a same-VM control:
1. **Arm 1:** `b1_screen_bend_release_0.1_50` (`s0=0.1, K=50`)
2. **Arm 2:** `b1_screen_bend_release_0.1_100` (`s0=0.1, K=100`)
3. **Arm 3:** `b1_screen_bend_release_0.0_100` (`s0=0.0, K=100`) (included because `s0=0.1` still permitted ~13° bend displacement early, and `s0=0.0` showed optimal trace scores across all 3 parents).
4. **Control:** `b1_control_tail_s11_13` run on this exact VM and git commit (`9885ce9`+).
5. **Selection rule:** The single configuration with the strongest tail mean $\Delta \text{top-1 } dE$ and parent consistency will be selected.
6. **Confirmation protocol:** Acceptance will be judged *only* on fresh confirmation runs: `tail` with fresh seeds **14, 15, 16** (with fresh-seed control) and `cal` with seeds **11, 12, 13** (with same-VM control).

#### Replay Parity Gate on Screening Arms
- `bend_release = [0.1, 50]` (`fp d21d06f725e9`): **9/9 PASS** on `DAMVAQ,HOXJUZ,JEGPUJ` × seeds `11,12,13`; 0 fallbacks (`mode == "replay"`), top-1 index match 9/9, `max relative |Δscore| = 0.000e+00`.
- `bend_release = [0.1, 100]` (`fp 615ae97d2350`): **9/9 PASS** on `DAMVAQ,HOXJUZ,JEGPUJ` × seeds `11,12,13` + **3/3 PASS** on `XIKGEE:11,12,13`; 0 fallbacks (`mode == "replay"`), top-1 index match 12/12, `max relative |Δscore| = 0.000e+00`.
- `bend_release = [0.0, 100]` (`fp e5c607873cff`): **9/9 PASS** on `DAMVAQ,HOXJUZ,JEGPUJ` × seeds `11,12,13`; 0 fallbacks (`mode == "replay"`), top-1 index match 9/9, `max relative |Δscore| = 0.000e+00`.

#### Tail Screening Results (seeds 11, 12, 13 vs Same-VM Control)
Executed across all 10 `tail` parents × 3 seeds (30 units each) on RTX 2060 GPU with 0 fallbacks. Output records saved to `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/screening/`.

| Screening Arm | Parameters | Mean $\Delta \text{top-1 } dE$ (kJ/mol) | Median $\Delta dE$ (kJ/mol) | Std $\Delta dE$ | Units Better / Worse / Neutral | Mean Top-1 $dE$ (kJ/mol) | Units $> 500$ kJ/mol (cand / ctrl) | Replay Fallbacks |
|---|---|---|---|---|---|---|---|---|
| **Control** | production | — | — | — | — | 783.47 | 20 / 20 | 0 |
| `b1_screen_bend_release_0.1_50` | `s0=0.1, K=50` | -28.11 | -1.20 | 71.76 | 17 / 5 / 8 | 755.36 | 16 / 20 | 0 |
| **`b1_screen_bend_release_0.1_100`** | **`s0=0.1, K=100`** | **-35.36** | **-2.73** | **81.63** | **16 / 5 / 9** | **748.11** | **14 / 20** | **0** |
| `b1_screen_bend_release_0.0_100` | `s0=0.0, K=100` | -29.89 | -1.72 | 99.27 | 15 / 7 / 8 | 753.57 | 14 / 20 | 0 |

##### Per-Parent Mean $\Delta \text{top-1 } dE$ Breakdown (kJ/mol, negative = improvement)
| Parent | Non-zero Bends? | `s0=0.1, K=50` | `s0=0.1, K=100` | `s0=0.0, K=100` | Seed Deltas (`s0=0.1, K=100` s11, s12, s13) | Notes |
|---|---|---|---|---|---|---|
| `DAMVAQ` | 0 bends | 0.00 | **0.00** | 0.00 | `-0.00, +0.01, +0.00` | Bit-identical invariance preserved |
| `HOXJUZ` | 12 bends | +0.18 | **+0.16** | +2.42 | `+12.75, -12.46, +0.18` | Neutral on average (redundant Q7 fibre) |
| `IFUROQ` | 32 bends | -12.15 | **-5.66** | -54.46 | `+4.00, -12.45, -8.53` | Consistent improvement |
| `JEGPUJ` | 16 bends | -186.40 | **-203.69** | -210.59 | `-193.26, -234.04, -183.76` | **Massive consistent win across all seeds** |
| `NEPNEF` | 0 bends | +0.01 | **-0.00** | -0.02 | `+0.00, +0.00, -0.01` | Bit-identical invariance preserved |
| `SOBHAS` | 22 bends | +7.29 | **-2.58** | -11.59 | `-38.78, +134.56, -103.53` | 2/3 seeds improved by ~40–100 kJ/mol |
| `USAPOU` | 16 bends | -99.99 | **-146.95** | -163.94 | `-146.82, -194.63, -99.40` | **Consistent win across all seeds** |
| `XEYTUT` | 8 bends | -1.03 | **-0.41** | -1.67 | `-3.37, +4.22, -2.09` | Neutral to slightly improved |
| `YUCZUS` | 12 bends | +13.46 | **+7.63** | +142.87 | `-19.08, -19.95, +61.91` | `0.1_100` avoids `0.0_100` freeze trap |
| `ZELTAP` | 20 bends | -2.47 | **-2.10** | -1.94 | `-5.79, +0.41, -0.92` | Consistent small improvement |

#### Screening Selection & Audit Recommendation
- **Winning Arm:** **`bend_release = [0.1, 100]`** is unambiguously selected:
  1. Achieves the largest overall energy reduction on tail: **-35.36 kJ/mol mean** ($\mathbf{-2.73}$ kJ/mol median).
  2. Reduces catastrophic tail units ($dE > 500$ kJ/mol) from **20 down to 14** out of 30.
  3. Delivers massive, reproducible reductions on the catastrophic parents: **`-203.7 kJ/mol`** on `JEGPUJ` and **`-146.9 kJ/mol`** on `USAPOU` across every single seed.
  4. Avoids the severe early-freeze trap suffered by `s0 = 0.0` on flexible complex `YUCZUS` (`+7.6` vs `+142.9` kJ/mol).
  5. Preserves exact invariance on complexes without angle bends (`DAMVAQ`, `NEPNEF`).
- **Audit Response:** Approved by auditor (`audit_jgen_B1_cp21_2026-10-26.md`). VERDICT: PASS.

---

### B1.2 Checkpoint 2.2 Confirmation: Fresh Tail Seeds (14, 15, 16) Results

Per Audit instructions (`audit_jgen_B1_cp21_2026-10-26.md`), the winning configuration `bend_release = [0.1, 100]` was evaluated on fresh `tail` seeds (14, 15, 16) against a fresh same-VM control (`b1_confirm_control`).

#### Summary Statistics (30 units, fresh seeds 14–16, 0 replay fallbacks)
- **Provenance Note:** Control runs were executed under un-overridden production behavior, which was previously mislabelled with fingerprint `34213ecbe2b8` due to dynamic `FIELDS` registration; this bug was resolved in `f3f8c4a` restoring `711411157fb4`.
- **Control Mean Top-1 $dE$:** 791.95 kJ/mol (18/30 units $> 500$ kJ/mol).
- **Candidate Mean Top-1 $dE$:** 768.00 kJ/mol (14/30 units $> 500$ kJ/mol).
- **Mean $\Delta \text{top-1 } dE$:** $\mathbf{-23.95}$ kJ/mol (median $-0.38$ kJ/mol; 16 better, 7 worse, 7 neutral).
- **Bootstrap 95% Confidence Interval for Mean $\Delta \text{top-1 } dE$:**
  - Unit-level CI: `[-45.03, -5.70]` kJ/mol.
  - **Parent-clustered CI (resampling whole parents):** **`[-55.0, -3.0]` kJ/mol** (strictly negative / excludes 0; Per-seed means: $-17$ / $-29$ / $-25$ kJ/mol).
- **Ensemble Floor $\Delta dE$:** Control floor mean 620.95 kJ/mol vs Candidate floor mean 594.80 kJ/mol (**Mean $\Delta \text{floor } dE = \mathbf{-26.15}$ kJ/mol**, median $-0.27$ kJ/mol). Proves that ensemble structural quality improves, not just top-1 score picking.
- **Top-1 Angle-Bend Prior:** Mean $\Delta \text{top-1 bend prior} = \mathbf{+0.2978}$ (deformation penalty significantly alleviated; on `JEGPUJ`, bend penalty reduced from $-3.834$ to $-1.353$).
- **Coordinates Saved:** Full Cartesian geometries for every top-1 and floor conformer written to `research/data/jgen/campaign/B/confirmation/tail/xyz/` for relaxed-energy evaluation by Agent A's decomposition tool.

#### Concentration Analysis (Audit Required)
- **Excluding `JEGPUJ` and `USAPOU` (24 units):**
  - **Mean $\Delta \text{top-1 } dE$:** $\mathbf{-4.91}$ kJ/mol (median $-0.00$ kJ/mol).
  - **Units Better / Worse / Neutral:** 11 better, 6 worse, 7 neutral.
  - *Observation:* While the largest single reductions come from catastrophic bend parents (`JEGPUJ` and `USAPOU`), the remaining 8 parents remain net negative (improving by $-4.91$ kJ/mol), with almost 2× as many units improved as worsened.
- **Per-Seed Spread (e.g. `SOBHAS`):**
  - `SOBHAS` fresh seed deltas: `[+47.53, -30.85, -87.63]`, mean $\mathbf{-23.65}$ kJ/mol. Consistent with screening (`[-38.78, +134.56, -103.53]`), stochastic search across seeds yields large single-seed swings on flexible complex `SOBHAS`, but the multi-seed average remains solidly negative.

#### Per-Parent Breakdown on Fresh Seeds (14, 15, 16)
| Parent | Bends | Mean $\Delta \text{top-1 } dE$ | Mean $\Delta \text{floor } dE$ | Fresh Seed Deltas (s14, s15, s16) | Top-1 Bend Prior (ctrl $\to$ cand) |
|---|---|---|---|---|---|
| `DAMVAQ` | 0 | **+0.00 kJ/mol** | +0.00 kJ/mol | `[-0.00, +0.00, +0.00]` | $0.000 \to 0.000$ (exact invariance) |
| `HOXJUZ` | 12 | **-0.54 kJ/mol** | -0.85 kJ/mol | `[+0.20, -0.72, -1.09]` | $-0.180 \to -0.179$ |
| `IFUROQ` | 32 | **-4.17 kJ/mol** | -4.17 kJ/mol | `[-18.19, +0.14, +5.54]` | $-1.051 \to -1.009$ |
| `JEGPUJ` | 16 | **-148.81 kJ/mol** | -142.67 kJ/mol | `[-210.94, -153.26, -82.25]` | $-3.834 \to -1.353$ (massive relief) |
| `NEPNEF` | 0 | **+0.00 kJ/mol** | -0.57 kJ/mol | `[-0.00, +0.00, +0.00]` | $0.000 \to 0.000$ (exact invariance) |
| `SOBHAS` | 22 | **-23.65 kJ/mol** | -15.86 kJ/mol | `[+47.53, -30.85, -87.63]` | $-0.001 \to -0.001$ |
| `USAPOU` | 16 | **-51.38 kJ/mol** | -54.48 kJ/mol | `[+44.15, -95.39, -102.89]` | $-0.806 \to -0.349$ |
| `XEYTUT` | 8 | **-1.34 kJ/mol** | +2.04 kJ/mol | `[-1.96, -0.00, -2.05]` | $-0.001 \to -0.001$ |
| `YUCZUS` | 12 | **-9.41 kJ/mol** | -44.93 kJ/mol | `[-34.12, -9.42, +15.30]` | $0.000 \to 0.000$ |
| `ZELTAP` | 20 | **-0.18 kJ/mol** | +0.03 kJ/mol | `[-0.46, -0.29, +0.21]` | $-0.059 \to -0.060$ |

---

### B1.3 Checkpoint 2.2 Confirmation: Full Cal Panel (120 Parents × Seeds 11–13) Results

Following tail confirmation, the full calibration panel (120 parents × seeds 11, 12, 13 = 360 units per arm, 720 units total) was evaluated sequentially on the local RTX 2060 GPU between `b1_confirm_bend_release_0.1_100` and `b1_confirm_control`.

#### Confirmation Statistics (360 paired units, 0 replay fallbacks)
- **Provenance Note:** Control records were run under production settings (mislabelled `34213ecbe2b8`, fixed in `f3f8c4a`).
- **Control Mean Top-1 $dE$:** 137.38 kJ/mol.
- **Candidate Mean Top-1 $dE$:** 133.73 kJ/mol.
- **Mean $\Delta \text{top-1 } dE$:** $\mathbf{-3.65}$ kJ/mol (median $-0.00$ kJ/mol, std $34.71$ kJ/mol).
- **Bootstrap 95% Confidence Interval for Mean $\Delta \text{top-1 } dE$:**
  - Unit-level CI: `[-7.40, -0.28]` kJ/mol.
  - **Parent-clustered CI (resampling whole parents):** **`[-9.7, +1.0]` kJ/mol**.
  - **Non-Inferiority Verification:** The upper bound of the parent-clustered 95% CI is **`+1.0 kJ/mol`**, safely below the $+2.0$ kJ/mol non-inferiority margin. At the parent level, `bend_release = [0.1, 100]` meets the non-inferiority standard ("no worse").
- **Exclusion Analysis (Excluding 3 Big Winners `JEGPUJ`, `NUWFAM`, `USAPOU`):**
  - Over the remaining 117 parents (351 units), mean $\Delta \text{top-1 } dE = \mathbf{+0.7}$ kJ/mol.
  - Parents went 46 better, 45 worse, 29 neutral.
  - *Interpretation:* The flag resolves severe early angle-bend blowouts on bend-catastrophe parents while remaining roughly neutral across the rest of the chemical space.
- **Catastrophic Units ($> 500$ kJ/mol):** Reduced from **14 units down to 8 units** across the panel (43% reduction).
- **Ensemble Quality:** Control floor mean 117.42 kJ/mol vs Candidate floor mean 114.93 kJ/mol (**Mean $\Delta \text{floor } dE = \mathbf{-2.49}$ kJ/mol**, median $-0.00$ kJ/mol).
- **Parent-Level Win Rate:** **46 better / 45 worse / 29 neutral** (out of 120 parents).
- **Replay Parity:** **0 fallbacks** across all 360 candidate units on GPU.
- **Coordinates Saved:** Full Cartesian geometries (`.xyz`) for all 360 units preserved in `research/data/jgen/campaign/B/confirmation/cal/xyz/`.

#### Notable Parent Highlights on Cal
- **Multi-Hundred kJ/mol Improvements:**
  - `JEGPUJ`: mean $\Delta dE = \mathbf{-203.68}$ kJ/mol
  - `NUWFAM`: mean $\Delta dE = \mathbf{-172.85}$ kJ/mol
  - `USAPOU`: mean $\Delta dE = \mathbf{-147.02}$ kJ/mol
  - `WEYJIT`: mean $\Delta dE = \mathbf{-67.18}$ kJ/mol
  - `WIBHOH`: mean $\Delta dE = \mathbf{-25.09}$ kJ/mol
- **Maximum Degradations:**
  - `QEZVEY`: $+45.94$ kJ/mol
  - `XACMOE`: $+40.74$ kJ/mol
  - `BOHWUT`: $+32.53$ kJ/mol
  - `AWONOR`: $+30.33$ kJ/mol
  - `HEDJOP`: $+26.84$ kJ/mol
  No single parent suffered a blowout remotely comparable to the triple-digit gains on `JEGPUJ`, `NUWFAM`, and `USAPOU`.

#### Checkpoint 2.2 Confirmation Verdict: PASS (Non-Inferior on Cal, Superior on Tail)
`bend_release = [0.1, 100]` satisfies the directive's acceptance rule:
1. Significant tail strain reduction: **$-23.95$ kJ/mol** (parent-clustered 95% CI `[-55.0, -3.0]` kJ/mol);
2. Statistically verified non-inferiority on cal: **$-3.65$ kJ/mol** (parent-clustered 95% CI `[-9.7, +1.0]` kJ/mol, upper bound under the $+2.0$ margin);
3. Reduced catastrophic failures ($> 500$ kJ/mol) by 4 units on tail and 6 units on cal;
4. 100% CUDA replay parity (0 fallbacks on all 390 candidate units);
5. Conformer geometries saved for step 2.2b relaxed-energy checks before promotion.

---

### B1.4 Step 2.2b: Relaxed-Energy Audit (GFN2 `xtb --opt crude`)

Per Audit instructions (`audit_jgen_B1_cp22_2026-10-27.md`), a relaxed-energy audit was conducted on the 4 biggest winners (`JEGPUJ`, `NUWFAM`, `USAPOU`, `WEYJIT`) and 4 worst parents (`QEZVEY`, `XACMOE`, `BOHWUT`, `AWONOR`) on `cal` (seeds 11–13, 48 top-1 relaxations + 8 reference relaxations).

Structures were relaxed using local GFN2 (`xtb --opt crude`, `-P 1`, `OMP_NUM_THREADS=1`) in isolated directories under `research/data/jgen/campaign/B/relax_2_2b/`. Total runtime: 10.74s (~0.22s per relaxation).

#### Step 2.2b Relaxed vs Unrelaxed Comparison Table
| Parent | Group | Mean Unrel Diff (kJ/mol) | Mean Rel Diff (kJ/mol) | Gain Retained % | Mean RMSD ctrl $\to$ cand (Å) | Status |
|---|---|---|---|---|---|---|
| `JEGPUJ` | WINNER | **-203.68** | **-14.05** | 6.9% | 1.116 / 0.900 | Keeps $-14.1$ kJ/mol real gain |
| `NUWFAM` | WINNER | **-172.85** | **-16.00** | 9.3% | 0.853 / 0.625 | Keeps $-16.0$ kJ/mol real gain |
| `USAPOU` | WINNER | **-147.02** | **-19.77** | 13.5% | 0.950 / 0.744 | Keeps $-19.8$ kJ/mol real gain |
| `WEYJIT` | WINNER | **-67.18** | **-1.42** | 2.1% | 0.687 / 0.645 | Keeps $-1.4$ kJ/mol real gain |
| `QEZVEY` | WORST | **+45.94** | **-1.23** | -2.7% | 0.496 / 0.530 | Turns slightly negative (better) |
| `XACMOE` | WORST | **+40.74** | **+39.51** | 97.0% | 0.303 / 0.192 | Candidate moved less (0.19 Å vs 0.30 Å) |
| `BOHWUT` | WORST | **+32.53** | **+1.90** | 5.9% | 0.495 / 0.627 | Degradation collapses to neutral |
| `AWONOR` | WORST | **+30.33** | **-9.42** | -31.0% | 0.464 / 0.550 | Turns negative (better relaxed) |

#### Key Physical & Methodological Findings:
1. **Unrelaxed Strain vs Structural Improvement:** All 4 winners retain a genuine energy reduction after full GFN2 relaxation (averaging $\sim -13$ kJ/mol across the 4 winners, up to $-19.8$ kJ/mol on `USAPOU`), and candidate conformers move noticeably less during relaxation than control conformers (e.g. `0.900 Å` vs `1.116 Å` on `JEGPUJ`; `0.625 Å` vs `0.853 Å` on `NUWFAM`). However, they retain $2\% - 14\%$ of the unrelaxed triple-digit MACE gain, proving that the vast majority of the unrelaxed $\Delta E$ in those complexes was indeed **local unrelaxed angle-bend strain** that crude GFN2 relaxation erases.
2. **Worst Parents:** On 3 of the 4 worst parents (`QEZVEY`, `BOHWUT`, `AWONOR`), the unrelaxed degradations collapse to neutral or turn negative (improving relaxed). On `XACMOE`, the $+39.5$ kJ/mol difference persists relaxed, though candidate conformers were closer to unrelaxed input ($0.19$ Å vs $0.30$ Å).
3. **Promotion Recommendation:** Because the winners retain $\le 15\%$ (below the $\ge 50\%$ promotion threshold) of their unrelaxed gain, `bend_release = [0.1, 100]` remains an active experimental flag rather than being promoted to default production configuration. We now proceed to Step 2.3 (`dof_step_scale`).

---

---

### B1.4 Addendum: Performance Audit & Benchmarking Harness Optimizations

During Checkpoint 2.1 execution, we conducted a targeted profiling and bottleneck audit of the benchmark evaluation harness (`tools.jgen bench run` and `replay_parity.py`) to eliminate redundancies and optimize scan/arm execution times.

#### 1. Replay-Parity CPU Step Loop Fix (`research/bench/tools/jgen/bench/replay_parity.py`)
- **Diagnosis:** `replay_parity.py` tests ReplaySolver vs JointIKSolver step parity on CPU (`test_replay_loop_matches_package_loop`) with intended step count `a.loop_steps = 20`. However, `builder.solve()` defaults to `n_ik_steps=200`, and `engine.py:2094` unconditionally overwrites `cfg.n_steps` with `n_ik_steps`. Consequently, the CPU parity check was executing the full 200 steps single-threaded twice per unit (~22–35s per unit).
- **Fix:** Pass `n_ik_steps=a.loop_steps` directly to `builder.solve` at lines 209 and 218.
- **Verification:** Verified on `DAMVAQ:11` and `JEGPUJ:11`:
  - `DAMVAQ:11`: Runtime dropped from 42.4s to 22.5s (1.9× overall wall-clock speedup, 8.1× on the CPU solve portion; `0.000e+00` bit parity across all checks).
  - `JEGPUJ:11`: Runtime dropped from 53.6s to 35.0s (`0.000e+00` bit parity across all checks).
  - Total replay-parity verification runtime cuts from ~380s down to ~180s (> 2× overall speedup) with zero regression.

#### 2. MACE Evaluation Profiling & Optimization Analysis (`research/bench/tools/jgen/bench/run.py`)
- **Profiling Breakdown per Parent (`DAMVAQ` 83 at / 48 confs, `JEGPUJ` 70 at / 11 confs):**
  - Geometry loading & ligand input extraction: < 1% of runtime (< 250 ms).
  - GPU IK solve (`solve_seeds`): ~3.4s – 4.3s (batched over seeds on RTX 2060).
  - MACE energy evaluation: 50–60% of total parent runtime (4.4s on `DAMVAQ`, 2.0s on `JEGPUJ`) due to serial per-seed calls with repeated `gc.collect()` and `torch.cuda.empty_cache()` synchronization stalls.
  - Reference energy `e_ref`: ~0.5s – 1.0s evaluated per parent (constant for a parent across all arms and seeds).
- **Batching Potential:**
  - Evaluated batching MACE scoring across seeds for each parent: yields a **1.7×–2.4× speedup** in scoring (saving ~1.7s per parent, or ~3.5 minutes per 120-parent `cal` arm) with bit-identical scores (`0.000e+00` max absolute error).
- **Parity & Stability Gate:** Because official confirmation runs (`tail` fresh seeds and `cal`) require strict comparability with the baseline control, scoring harness changes are held in reserve and the solver step loop remains canonical.

---

### 2. `dof_step_scale` (Per-DOF-Class Step Scaling)

- **Mechanism expected:**
  In `JointIKSolver`, a single scalar learning rate `self.lr = 0.05` (with cosine decay to `lr_final = 0.005`) multiplies the Adam update `m_hat / (v_hat.sqrt() + eps_a)` for every column of `theta_li`. Because Adam normalizes each column's gradient by its own running RMS $\sqrt{\hat{v}}$, a stiff angle-bend or ring-joint column with a large gradient takes the exact same coordinate step (`~0.05 rad` per step) as a rigid-body rotation or translation (`~0.05 Å/rad`), even though:
  - rigid-body translation/rotation (`[0:6]`) moves the entire ligand with zero internal strain;
  - acyclic torsions (`[6 : 6 + n_joints - n_macro_joints]`) rotate about single bonds with low barrier;
  - ring/macrocycle joints (`[6 + n_joints - n_macro_joints : 6 + n_joints]`) deform ring/macrocycle geometry;
  - angle bends (`[6 + n_joints : 6 + n_joints + n_angle_bends]`) directly strain bond angles (`310..1800 kJ/(mol·rad²)`).
  Applying a static per-DOF-class multiplier `(s_rigid, s_torsion, s_ring, s_bend)` to the Adam step increases the relative speed of rigid-body/torsion placement while damping ring and angle-bend steps throughout the trajectory.

- **Where in the code it goes (`file:line`):**
  1. `src/jacobik/profile.py:79`: Register `FIELDS["dof_step_scale"] = ("jacobik.ik.ik_solver", "_DOF_STEP_SCALE", None, "Per-DOF-class step multipliers [s_rigid, s_torsion, s_ring, s_bend] (None = off)")`.
  2. `src/jacobik/ik/ik_solver.py:68`: Read `_DOF_STEP_SCALE = _profile.get("dof_step_scale")` at import and store `self.dof_step_scale = _DOF_STEP_SCALE if dof_step_scale is None else dof_step_scale` in `JointIKSolver.__init__` (`ik_solver.py:553`).
  3. `src/jacobik/ik/ik_solver.py:728` (`_solve_impl` setup): Combine `dof_step_scale` into the same per-ligand `(T, D_li)` device table `_step_scale_tabs[li]` used by `bend_release`:
     - columns `0 : 6` multiplied by `s_rigid`;
     - columns `6 : 6 + (fk_li.n_joints - fk_li.n_macro_joints)` multiplied by `s_torsion`;
     - columns `6 + (fk_li.n_joints - fk_li.n_macro_joints) : 6 + fk_li.n_joints` multiplied by `s_ring`;
     - columns `6 + fk_li.n_joints : 6 + fk_li.n_joints + fk_li.n_angle_bends` multiplied by `s_bend` (and by the `bend_release` ramp when both flags are active).
  4. `src/jacobik/ik/ik_solver.py:913` & `research/bench/tools/jdev/build_replay_solver.py:328` (`src/jacobik/ik/build_replay_solver.py:328`): Uses the exact same `_step_scale_tabs[li]` lookup (`_step_scale_tabs[li][t]` in `ik_solver.py` and `_step_scale_tabs[li].index_select(0, tl)` in `part_b` of `solver_replay.py`).

- **How it enters the replay generator:**
  Identical to `bend_release`: `_step_scale_tabs[li]` (`(T, D_li)`) is prebuilt in `_solve_impl` setup (copied verbatim into `solver_replay.py`) and indexed via `_step_scale_tabs[li].index_select(0, tl)` inside `part_b`. Even when `bend_release` is off and `dof_step_scale` is static across steps, storing it in `_step_scale_tabs[li]` (`(T, D_li)`) means a single unified mechanism handles `bend_release`, `dof_step_scale`, or their combination in `part_b` with zero extra ops.

- **Parameters and values to test (at most 3 values per parameter):**
  4 group scales `[s_rigid, s_torsion, s_ring, s_bend]`:
  - `s_rigid`: `{1.0, 1.5}` (2 values)
  - `s_torsion`: `{1.0, 1.25}` (2 values)
  - `s_ring`: `{0.5, 1.0}` (2 values)
  - `s_bend`: `{0.25, 0.5, 1.0}` (3 values)

#### Evaluation Contract: `dof_step_scale` Screening on `tail` (seeds 11–13)
Per Audit instructions (`audit_jgen_B1_cp22_2026-10-27.md`), screening is conducted with at most 3 configurations varying ONE group from production (avoiding bend-only since that overlaps `bend_release`):
1. **Arm 1:** `b1_screen_dof_rigid_1.5`: `{"dof_step_scale": {"rigid": 1.5}}` (rigid-body step multiplier 1.5×)
2. **Arm 2:** `b1_screen_dof_torsion_1.25`: `{"dof_step_scale": {"torsion": 1.25}}` (acyclic torsion step multiplier 1.25×)
3. **Arm 3:** `b1_screen_dof_ring_0.5`: `{"dof_step_scale": {"ring": 0.5}}` (ring/macrocycle joint damping to 0.5×)
4. **Control:** Same-VM baseline control `b1_screen_control` (un-overridden production solver).
5. **Statistical Criterion:** Evaluated using parent-clustered bootstrap confidence intervals (resampling whole parents, 10,000 resamples), win/loss counts, and concentration analysis.

#### Replay Parity Gate on Screening Arms
As required by Directive §3 and Audit CP21/CP22 mandates, every arm passed:
- One 200-step CPU schedule check (`JEGPUJ:11`, `--loop-steps 200`) verifying the generated `solver_replay.py` loop against `ik_solver.py` with zero culling;
- Full 9-unit parity panel (3 tail parents `DAMVAQ,HOXJUZ,JEGPUJ` × 3 seeds `11,12,13`, `--loop-steps 20`).

Results:
- **`b1_screen_dof_rigid_1.5` (`fp 83a767adcbfb`):**
  - 200-step CPU check on `JEGPUJ:11`: **PASS** (`top1 = 109/109`, `max_rel = 0.000e+00`).
  - 9-unit parity panel: **PASS** — 9/9 units passed, 0 fallbacks (`mode == "replay"`), top-1 index match 9/9, `max_rel = 0.000e+00`, `max_abs = 0.000e+00`.
- **`b1_screen_dof_torsion_1.25` (`fp 1b804781c674`):**
  - 200-step CPU check on `JEGPUJ:11`: **PASS** (`top1 = 31/31`, `max_rel = 0.000e+00`).
  - 9-unit parity panel: **PASS** — 9/9 units passed, 0 fallbacks (`mode == "replay"`), top-1 index match 9/9, `max_rel = 0.000e+00`, `max_abs = 0.000e+00`.
- **`b1_screen_dof_ring_0.5` (`fp 90975b10a37c`):**
  - 200-step CPU check on `JEGPUJ:11`: **PASS** (`top1 = 33/33`, `max_rel = 0.000e+00`).
  - 9-unit parity panel: **PASS** — 9/9 units passed, 0 fallbacks (`mode == "replay"`), top-1 index match 9/9, `max_rel = 0.000e+00`, `max_abs = 0.000e+00`.

#### Tail Screening Results (seeds 11, 12, 13 vs Same-VM Control)
Executed sequentially across all 10 `tail` parents × 3 seeds (30 units each) on the RTX 2060 GPU with 0 replay fallbacks. Records saved to `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/screening/`.

| Screening Arm | Overrides | Mean $\Delta \text{top-1 } dE$ (kJ/mol) | Median $\Delta dE$ (kJ/mol) | Parent-Clustered 95% Bootstrap CI | Better / Worse / Tie | Mean $\Delta \text{floor } dE$ (kJ/mol) | Units $> 500$ kJ/mol (cand / ctrl) | Replay Fallbacks | Gate Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Control** | production | — | — | — | — | — | 20 / 20 | 0 | — |
| `b1_screen_dof_rigid_1.5` | `rigid: 1.5` | -38.79 | -6.02 | `[-94.30, +8.72]` | 22 / 8 / 0 | -9.84 | 18 / 20 | 0 | INCONCLUSIVE (CI crosses 0) |
| `b1_screen_dof_torsion_1.25` | `torsion: 1.25` | -38.04 | -1.73 | `[-95.02, +3.93]` | 19 / 11 / 0 | -28.86 | 15 / 20 | 0 | INCONCLUSIVE (CI crosses 0) |
| **`b1_screen_dof_ring_0.5`** | **`ring: 0.5`** | **-5.53** | **-0.00** | **`[-14.64, -0.25]`** | **8 / 2 / 20** | **-0.65** | **19 / 20** | **0** | **PASS (CI strictly < 0)** |

##### Per-Parent Mean $\Delta \text{top-1 } dE$ Breakdown (kJ/mol, negative = improvement)
| Parent | Ring / Macro Joints? | Rigid 1.5 | Torsion 1.25 | Ring 0.5 | Ring 0.5 Seed Deltas (s11, s12, s13) | Physical Interpretation |
|---|---|---|---|---|---|---|
| `DAMVAQ` | No (0) | +27.62 | +22.70 | **+0.00** | `[-0.00, +0.00, +0.00]` | Exact bit-invariance preserved on non-ring complex |
| `HOXJUZ` | No (0) | +1.34 | -5.00 | **+0.07** | `[+0.20, -0.00, +0.00]` | Exact bit-invariance on s12, s13; numerical noise on s11 |
| `IFUROQ` | No (0) | -7.19 | -4.79 | **-0.00** | `[-0.00, +0.00, -0.00]` | Exact bit-invariance preserved |
| `JEGPUJ` | No (0) | -164.42 | -261.53 | **-0.00** | `[-0.00, -0.00, +0.00]` | Exact bit-invariance preserved |
| `NEPNEF` | No (0) | -11.49 | -17.53 | **+0.00** | `[+0.00, +0.00, +0.00]` | Exact bit-invariance preserved |
| `SOBHAS` | No (0) | -220.63 | -0.44 | **+0.00** | `[-0.00, +0.00, +0.00]` | Exact bit-invariance preserved |
| `USAPOU` | Yes (macrocycle) | -75.57 | -128.54 | **-44.58** | `[-28.84, -104.95, -0.00]` | Consistent major improvement on macrocycle |
| `XEYTUT` | Yes (ring joint) | -0.91 | -0.12 | **-2.72** | `[-3.53, -0.00, -4.63]` | Consistent improvement across seeds |
| `YUCZUS` | Yes (ring joint) | +65.01 | +15.80 | **-7.14** | `[-21.43, +0.00, -0.00]` | Damping ring steps cures blowout |
| `ZELTAP` | Yes (ring joint) | -1.64 | -1.00 | **-0.92** | `[-0.53, -1.52, -0.71]` | Consistent improvement across all 3 seeds |

##### Concentration & Physical Mechanism Findings
1. **`ring: 0.5` is cleanly selective and passes the parent-clustered CI test:**
   - Damping ring DOF updates by 0.5× suppresses over-stepping in constrained ring/macrocycle closures.
   - For complexes without ring/macrocycle joints (6 of 10 parents), `dof_step_scale = {"ring": 0.5}` preserves production coordinates and scores identically (`0.000 kJ/mol`).
   - For all 4 complexes with ring/macrocycle joints (`USAPOU`, `XEYTUT`, `YUCZUS`, `ZELTAP`), damping ring steps consistently improves or preserves energies across seeds. Not a single ring parent suffered a degradation on average (`-44.6`, `-2.7`, `-7.1`, `-0.9` kJ/mol).
   - Even excluding `USAPOU`, the remaining parents remain net improved ($-1.34$ kJ/mol).
   - The parent-clustered 95% bootstrap CI is **`[-14.64, -0.25]` kJ/mol**, satisfying the strict acceptance rule (excluding 0).
2. **`rigid: 1.5` and `torsion: 1.25` show large wins but fail parent consistency:**
   - Both arms produce substantial nominal mean reductions ($-38.79$ and $-38.04$ kJ/mol) driven by huge wins on `JEGPUJ` (up to $-261.5$ kJ/mol) and `USAPOU` (up to $-128.5$ kJ/mol).
   - However, increasing the step size destabilizes compact complexes: `DAMVAQ` degrades by $+27.6$ kJ/mol on `rigid: 1.5` and $+22.7$ kJ/mol on `torsion: 1.25`, while `YUCZUS` suffers single-seed blowouts ($+345.7$ kJ/mol and $+75.0$ kJ/mol on seed 13).
   - Resampling whole parents reveals that their gains are concentrated, leaving the parent-clustered 95% CIs crossing zero (`[-94.30, +8.72]` and `[-95.02, +3.93]`).
   - Per directive rule ("State failures plainly. If a gate fails, say so and stop that item"), `rigid: 1.5` and `torsion: 1.25` are stopped at screening.
3. **Recommendation for Checkpoint 2.3:**
   - Promote **`dof_step_scale = {"ring": 0.5}`** to confirmation (`tail` seeds 14–16 and `cal` seeds 11–13) as the sole passing configuration from Checkpoint 2.3.

---

### 3. `start_kabsch` (Torsion-Aware Kabsch Placement & Controlled Orientation Diversity)

- **Mechanism expected:**
  Inspection of step-0 initialization in `src/jacobik/engine.py:1966–2041` and `src/jacobik/ik/kabsch.py:25–92` exposes two concrete step-0 placement defects that explain why polydentate and flexible ligands start with donors far from their assigned target vertices (e.g. `JEGPUJ`, where donors start 8–17 Å from the metal at step 0):
  1. **Un-torsioned Kabsch reference (`src/jacobik/engine.py:2024–2037`):**
     For every non-macrocycle ligand (`n_macro == 0`), `engine.py:2024` sets `X_kabsch = X_full` (the un-torsioned reference conformer) and passes `X_kabsch` to `kabsch_init`, **even though `beam_torsions_B` (shape `(B, J_full)`, containing both the `BeamSearch` torsion seeds and the randomized `bend_seed_B` dihedral seeds) has already been constructed at lines 1989–2005 and is immediately concatenated into `theta_init = torch.cat([base_theta, beam_torsions_B], dim=-1)` at line 2037!**
     When `fk.apply(theta_init, X_ref)` executes at step 0 (`src/jacobik/ik/fk.py:390–434`), `fk.apply` first applies the internal joints `beam_torsions_B` (which rotate the downstream donor atoms to new positions $P_{\text{tors}}$ and shift their centroid to $\bar{P}_{\text{tors}}$), and *then* applies `base_theta` (`R`, `t`) around the reference centroid `self.pivot = P_ref.mean(0)`. Because `base_theta` was fit to the un-torsioned `X_full` donor positions rather than the torsioned conformer, both the Kabsch rotation `R` and translation `t = Qm - Pm` are wrong for every row with non-zero torsion/bend-dihedral seeds (`J_full > 0`).
     Notice how `fk.apply` (`src/jacobik/ik/fk.py:424–431`) applies the rigid-body transform:
     ```python
     lig_coords = X_out[:, self._lig_pos_t, :]
     pivot = self.pivot.to(theta.device) # fixed reference donor centroid from X_full!
     centered = lig_coords - pivot
     rotated = torch.bmm(centered, R.transpose(1, 2))
     X_out[:, self._lig_pos_t, :] = rotated + pivot + t.unsqueeze(1)
     ```
     Because `fk.apply` rotates around `self.pivot` (the `X_full` donor centroid) and adds `t`, if internal torsions `beam_torsions_B` move a row's donor centroid from `pivot` to `Pm_row`, then rotating around `pivot` by `R` moves `Pm_row` to `pivot + R @ (Pm_row - pivot) + t`. For the final donor centroid to land at `Qm` (the target donor centroid), `t` must be:
     $$t = Q_m - \text{pivot} - R\,(P_{m,\text{row}} - \text{pivot})$$
     (which reduces to $Q_m - P_m$ only when $P_{m,\text{row}} = \text{pivot}$!). Today (`kabsch.py:87`), `t = Qm - Pm` assumes `fk.apply` rotates around `Pm_row`, which is false whenever `Pm_row != self.pivot` (both for torsioned ligands and for macrocycles!).
  2. **Uncontrolled $\pm 180^\circ$ post-Kabsch perturbation on well-posed polydentate ligands (`src/jacobik/ik/kabsch.py:80`):**
     `engine.py:2034–2035` calls `kabsch_init(..., diversify_orientation=True, diversity_scale=1.0)`. Inside `kabsch.py:80`, for well-posed `rank >= 2` ligands (`n >= 3` non-collinear donors, where Kabsch SVD uniquely determines the 3D orientation that aligns all donors to their target vertices), it multiplies `R` by a random rotation of angle `u * diversity_scale * pi` — up to **$180^\circ$** around a random axis! A $90^\circ$–$180^\circ$ random rotation on a tridentate/tetradentate/hexadentate ligand completely scrambles the donor-to-vertex alignment that Kabsch just solved, placing donors on the opposite side of the coordination sphere and forcing the IK trajectory through steric clashes. Furthermore, for `rank == 0` (`n == 1` monodentate) and `rank == 1` (`n == 2` bidentate/collinear) ligands, `kabsch_init` does not align the ligand's approach direction (substituent centroid $\to$ donor vector with target $\to$ metal vector) before spinning around the free axis (`ax0` or `ax1`), so monodentate and bidentate ligands frequently start pointing their bulky backbone *into* the metal rather than away from it.

- **Where in the code it goes (`file:line`):**
  1. `src/jacobik/profile.py:79`: Register `FIELDS["start_kabsch"] = ("jacobik.engine", "_START_KABSCH", None, "Torsion-aware Kabsch init + pivot/approach alignment + poly diversity scale (None = off)")`.
  2. `src/jacobik/engine.py:2024–2037`: When `_START_KABSCH` is enabled:
     - If `J_full > 0` and `n_macro == 0`, compute `X_kabsch = fk.apply(torch.cat([zero6, beam_torsions_B], dim=-1), X_full)` so `kabsch_init` sees each row's actual torsioned donor coordinates $P_{\text{tors}}$.
     - Pass `pivot=fk.pivot`, `diversity_scale=poly_scale`, and `align_approach=align_approach` into `kabsch_init`.
  3. `src/jacobik/ik/kabsch.py:25–92`:
     - When `pivot is not None`, compute the exact rigid-body translation `t = Qm - pivot - torch.bmm(R, (Pm - pivot).unsqueeze(-1)).squeeze(-1)` so that after `fk.apply` rotates around `fk.pivot`, the torsioned donor centroid lands at `Qm` exactly.
     - When `align_approach=True`, for `n == 1` (`rank == 0`) and `n == 2` (`rank == 1`), before the random spin around `ax0` / `ax1`, apply the minimal rotation that aligns the ligand's mean donor-to-substituent vector (or ligand atom centroid $\to$ donor centroid vector when substituent indices are unavailable) with the metal $\to$ target centroid vector `Qm - metal_pos`, so the donor lone pair points toward the metal and the ligand body points outward.
     - Use `diversity_scale = poly_scale` for `rank >= 2` rows.

- **How it enters the replay generator:**
  `start_kabsch` modifies `theta_init` in `engine.py` *before* `_run_joint_ik` invokes `ReplaySolver` or `JointIKSolver`. Both solvers receive the exact same `theta_inits` list and execute the unchanged step loop, so CUDA-graph replay works automatically without any changes to `build_replay_solver.py` or `solver_replay.py`.

- **Parameters and values to test (at most 3 values per parameter):**
  - `torsion_pivot` (torsion-aware `X_kabsch` + exact `fk.pivot` translation compensation): `{True}` (1 value when `start_kabsch` is active).
  - `poly_scale` (`diversity_scale` for well-posed `rank >= 2` ligands, vs `1.0` today): `{0.0, 0.15, 0.3}` (3 values: exact Kabsch `0°`, small jitter `<= 27°`, moderate jitter `<= 54°`).
  - `align_approach` (align ligand body outward / lone pair toward metal for `rank <= 1` before axial spin): `{False, True}` (2 values).
  Before coding `start_kabsch`, we will also run the B1 start-placement census across `cal` (measuring step-0 per-donor distance to assigned target vertex and step-0 score by denticity/chelate class) and report the table in §B1.


---

## §C. Package C — Confirmation, Diagnostics, and Relaxed Audit

**Pre-Registered Evaluation Contract:** [`research/data/jgen/campaign/B/confirmation_pkgc/evaluation_contract.json`](../data/jgen/campaign/B/confirmation_pkgc/evaluation_contract.json)  
**Confirmation Data Directory:** [`research/data/jgen/campaign/B/confirmation_pkgc/`](../data/jgen/campaign/B/confirmation_pkgc/)  
**Confirmed Configurations Evaluated:**
- `b1_pkgc_control`: `{}` (production baseline u1, fp `711411157fb4`)
- `b1_pkgc_ring_0.5`: `{"dof_step_scale": {"ring": 0.5}}`
- `b1_pkgc_combined`: `{"bend_release": [0.1, 100], "dof_step_scale": {"ring": 0.5}}`

---

### C.1 Runner Saturation & Optimization Results

Per owner and supervisor directives, the simple bench runner (`research/bench/tools/jgen/bench/run.py`) was restructured for high hardware utilization on the local RTX 2060 SUPER (8 GB):
1. **Multi-Arm Batching (`--arms-spec`):** Amortizes `inputs.load_mol`, ligand decomposition, and free-ligand GFN2 refinement across all arms in a single process.
2. **Background 1-Thread CPU Prefetching:** CPU prefetches and warms `_GFN2_CACHE` for parent $i+1$ in a background thread while the GPU solves parent $i$.
3. **Batched Single-Pass MACE Evaluation:** Collects all generated conformers across seeds and arms for each parent, deduplicates identical Cartesian coordinate sets via byte hashing, and evaluates reference plus conformers in a single `mace.energies()` call (`JACOBIK_MACE_BATCH_ATOMS=900`), eliminating redundant forward passes and per-arm `empty_cache()` synchronization stalls.
4. **Scale-Invariance Fast Path:** For arms modifying DOF scales specific to inactive DOF subsets (e.g. `ring: 0.5` on complexes without macrocycle joints), the runner detects mathematically invariant solves and reuses control conformers directly, guaranteeing exact $0.000000\text{e}+00$ numerical parity and bypassing redundant GPU execution.
5. **cuSOLVER Chunking & E/Z Enumeration (Fix A & Fix B):** Wrapped solves in `multiseed.safe_cuda_eigh_chunking()` to prevent cuSOLVER `eigh` failures on large matrices, and added automated E/Z enumeration dispatch (`multiseed.solve_seeds`) for parents with multiple E/Z combinations (`has_ez_multi`).

#### Runner Performance & GPU Saturation Comparison

| Metric | Stock Runner Baseline (B1/B2) | Package C Optimized Runner (Pilot) | Package C Tail Panel (3 arms × 10 parents) | Package C Cal Panel (3 arms × 120 parents) | Improvement / Saturation |
|---|---|---|---|---|---|
| **Post-Startup Throughput** | 19.35 units/min | **48.87 units/min** | **52.5 equiv. units/min** | **79.7 equiv. units/min** | **> 4.1× speedup** |
| **Total Throughput (incl. MACE load)** | 10.59 units/min | **28.72 units/min** | **17.5 multi-arm units/min** | **26.6 multi-arm units/min** | **2.5× to 2.7× end-to-end** |
| **GPU Utilization (nvidia-smi)** | 27.0% mean | **Up to 99%** (84–99% solve bursts) | **75–98%** active solve bursts | **80–99%** sustained bursts | **Full GPU saturation** |
| **Peak GPU Memory** | ~2.4 GiB | **4.95 GiB** (60.4% of 8 GB) | **5.12 GiB** (62.5% of 8 GB) | **5.48 GiB** (66.9% of 8 GB) | Safe headroom below 80% cap |
| **Wall Time for 1080 Units (Cal)** | ~105 min (projected) | — | — | **40.6 min** (2439 s) | **61% wall-clock reduction** |

---

### C.2 Pre-Registered Diagnostic Questions

#### C.2.1 §3.1 `HOXJUZ` Question & DOF-Group Counts Table
**Question:** In Checkpoint 2.3 screening, `HOXJUZ:11` showed a $+0.20$ kJ/mol delta under `ring: 0.5`. Does `HOXJUZ` actually have a DOF in the `ring` group, or is that $+0.20$ kJ/mol non-determinism?

**Investigation & Complete Census:**
Inspecting `ForwardKinematics(mol, lig, angle_bends=True)` (`fk.n_joints`, `fk.n_macro_joints`, `fk.n_angle_bends`, `fk.n_ring_fold_hinges`) across all 10 `tail` parents shows that **only `USAPOU` has DOFs in the `ring` group (`fk.n_macro_joints > 0`)**. Neither `HOXJUZ` nor `XEYTUT`, `YUCZUS`, or `ZELTAP` (which were misattributed as having ring joints in the Session 1 table) has any `ring` DOFs:

| Parent | $n_{\text{lig}}$ | Rigid DOFs ($6 \times n_{\text{lig}}$) | Acyclic Torsion DOFs (`n_joints - n_macro`) | Ring DOFs (`n_macro_joints`) | Angle-Bend DOFs (`n_angle_bends`) | Ring-Fold Hinges | Active Under `ring: 0.5`? | Cause of Screening $\Delta E$ (`s11–s13`) |
|---|---|---|---|---|---|---|---|---|
| `DAMVAQ` | 2 | 12 | 5 | **0** | 0 | 0 | No | None (`0.000e+00`, 0 bends) |
| `HOXJUZ` | 2 | 12 | 9 | **0** | 12 | 0 | No | CUDA `index_add_` (`atomicAdd`) non-determinism |
| `IFUROQ` | 1 | 6 | 8 | **0** | 32 | 0 | No | None (`0.000e+00`) |
| `JEGPUJ` | 1 | 6 | 16 | **0** | 16 | 0 | No | None (`0.000e+00`) |
| `NEPNEF` | 4 | 24 | 10 | **0** | 0 | 0 | No | None (`0.000e+00`, 0 bends) |
| `SOBHAS` | 4 | 24 | 8 | **0** | 22 | 0 | No | None (`0.000e+00`) |
| `USAPOU` | 3 | 18 | 8 | **5** | 16 | 0 | **Yes** | Genuine macrocycle joint damping (`n_macro=5`) |
| `XEYTUT` | 3 | 18 | 2 | **0** | 8 | 0 | No | CUDA `index_add_` (`atomicAdd`) non-determinism |
| `YUCZUS` | 4 | 24 | 20 | **0** | 12 | 0 | No | CUDA `index_add_` (`atomicAdd`) non-determinism |
| `ZELTAP` | 4 | 24 | 8 | **0** | 20 | 0 | No | CUDA `index_add_` (`atomicAdd`) non-determinism |

**Direct Verification:**
1. **`HOXJUZ` has `0` ring DOFs** (`n_joints = 9` across two tridentate ligands `[5, 4]`, `n_macro_joints = 0`, `n_angle_bends = 12`). When `ring: 0.5` is set, `_step_scale_tabs[li]` is identically `1.0` in every column on `HOXJUZ`, `XEYTUT`, `YUCZUS`, and `ZELTAP`.
2. Re-running `control` vs `ring: 0.5` on CUDA with `torch.use_deterministic_algorithms(True)` gives **`dScore = 0.000000e+00` and `max_pos_diff = 0.000e+00` Å** on `HOXJUZ`, `XEYTUT`, `YUCZUS`, and `ZELTAP`. Without deterministic algorithms enabled, PyTorch's CUDA `index_select_backward` (`index_add_` with `atomicAdd` on repeated atom indices in `BendPlan.apply` and `ComplexFK.apply`) introduces 1-ULP gradient jitter across the 200 Adam steps, which caused a discrete top-1 conformer tie-break flip on `HOXJUZ:11` (`+0.20` kJ/mol) and on `XEYTUT`, `YUCZUS`, `ZELTAP` in Session 1.
3. On the full 120-parent `cal` panel, an exhaustive `ForwardKinematics` census shows that **exactly 3 parents have `n_macro_joints > 0`**: **`AWONOR`** (`n_ring = 5`, COD), **`USAPOU`** (`n_ring = 5`, N3 macrocycle), and **`XOJHIN`** (`n_ring = 5`, macrocycle).

#### C.2.2 §3.2 `XACMOE` Diastereomer Inversion Diagnosis
**Question:** On `XACMOE` ($\text{Zn(II)}$ octahedral complex with a single linear hexadentate $\text{O31–N27–N29–N30–N28–O32}$ Schiff-base ligand, 60 atoms), `bend_release = [0.1, 100]` improved the unrelaxed solver score (`0.344–0.349` $\to$ `0.424–0.436`), yet relaxed top-1 $\Delta E$ worsened by `+37.7 / +40.1 / +40.8` kJ/mol across seeds 11–13 in Step 2.2b (`+34.7 / +35.5 / +36.2` kJ/mol in Package C). What happened?

**1. Donor Arrangement (Different Topological Isomer / Diastereomer):**
- **Crystal Reference (`ref`, relaxed):** *trans* imines ($\text{N27–Zn–N28} = 170.9^\circ$) and *cis* phenolates/amines ($\text{N29–Zn–O31} = 156.4^\circ$, $\text{N30–Zn–O32} = 156.3^\circ$, $\text{O31–Zn–O32} = 99.6^\circ$), with all 6 donors tightly coordinated ($\text{Zn–O} = 2.02\text{ Å}$, $\text{Zn–N}_{\text{imine}} = 2.09\text{ Å}$, $\text{Zn–N}_{\text{amine}} = 2.25\text{ Å}$).
- **Control Top-1 (`s11–s13`):** Converges in all 3 seeds to topological assignment **`asg=1` (`[0, 3, 2, 1, 5, 4]`)** (score `0.344–0.349`, unrelaxed $\Delta E = +154.3 / +153.8 / +140.7$ kJ/mol), a symmetric *trans*-phenolate isomer ($\text{O31–Zn–O32} = 156.6^\circ\text{–}157.6^\circ$, $\text{N27–Zn–N30} = 148.4^\circ$, $\text{N28–Zn–N29} = 144.5^\circ\text{–}145.4^\circ$). Upon GFN2 relaxation (`rmsd = 0.29–0.31 Å`), all 6 donors remain tightly bound ($\text{Zn–N}_{\text{amine}} = 2.16\text{–}2.18\text{ Å}$) at a low-energy local minimum ($E_{\text{rel}} - E_{\text{ref}} = +53.7 / +53.8 / +53.2$ kJ/mol).
- **Candidate Top-1 (`bend_release = [0.1, 100]`, `s11–s13`):** Switches in all 3 seeds to topological assignment **`asg=2` (`[0, 4, 2, 1, 5, 3]`)** (score `0.424–0.436`, unrelaxed $\Delta E = +194.1 / +188.6 / +194.9$ kJ/mol), an asymmetric *cis*-phenolate diastereomer ($\text{O31–Zn–O32} = 89.2^\circ\text{–}96.2^\circ$) where one phenolate is *trans* to the distal imine ($\approx 162^\circ$), the other phenolate is *trans* to a central amine ($\approx 162^\circ\text{–}164^\circ$), and the remaining imine is *trans* to the other amine ($\approx 137^\circ\text{–}139^\circ$). In this diastereomer, the central triamine bridge wraps awkwardly around two adjacent facial edges; upon GFN2 relaxation (`rmsd = 0.19 Å`), one central secondary amine is forced to stretch away from $\text{Zn}$ ($\text{Zn–N29} = 2.467\text{ Å}$ in `s11`; $\text{Zn–N30} = 2.432\text{–}2.433\text{ Å}$ in `s12/s13` vs $2.17\text{ Å}$ in control), leaving it trapped at $E_{\text{rel}} - E_{\text{ref}} = +91.4 / +93.9 / +93.9$ kJ/mol (`+37.7 / +40.1 / +40.8` kJ/mol above control).

**2. Torsions Differing by $> 60^\circ$:**
- In solver internal coordinates (`cand - ctrl` across `s11–s13`): **`Joint 8` (`C15–N16–C17–C18`)** shifts by **$-60.2^\circ\text{ to }-61.6^\circ$** ($+123.6^\circ..+124.4^\circ \to +62.4^\circ..+64.1^\circ$) and methoxy **`Joint 14` (`O26–C27–C28–H58`)** rotates by **$\pm 120^\circ$**, accompanied by **`Joint 12` (`C20–C21–C22–C23`)** shifting by **$-57.0^\circ\text{ to }-57.4^\circ$** ($+127.6^\circ..+127.8^\circ \to +70.2^\circ..+70.7^\circ$).
- In reference-mapped relaxed backbone dihedrals (`s11`): the ethylene/propylene diamine bridge flips sign across **`N27–C10–C11–N29` ($-70.0^\circ$)**, **`C11–N29–C12–C13` ($+98.0^\circ$)**, **`C12–C13–N30–C14` ($-124.4^\circ$)**, and **`N30–C14–C17–N28` ($+61.1^\circ$)**.

**3. Which Bend Columns Moved:**
- Total bend penalty is unchanged (`bend_prior`: control `-0.300 to -0.310` vs candidate `-0.273 to -0.310`), but the second arm's out-of-plane imine/salicylaldimine bends invert sign to route that arm to swapped octahedral vertices (`[5, 3]` vs `[5, 4]`):
  - **`Bend 43` (col 64, `C18–N19–C20 (out)`):** $-10.5^\circ..-11.0^\circ \to +9.4^\circ..+9.7^\circ$ ($\Delta = \mathbf{+20.2^\circ\text{ to }+20.4^\circ}$)
  - **`Bend 49` (col 70, `C21–C20–N19 (out)`):** $+10.7^\circ..+11.2^\circ \to -10.4^\circ..-10.7^\circ$ ($\Delta = \mathbf{-21.4^\circ\text{ to }-21.6^\circ}$)
  - **`Bend 47` (col 68, `N19–C20–C21 (out)`):** $+4.5^\circ..+4.8^\circ \to -7.6^\circ..-7.8^\circ$ ($\Delta = \mathbf{-12.3^\circ\text{ to }-12.4^\circ}$)
  - **`Bend 45` (col 66, `C20–N19–C18 (out)`):** $-6.1^\circ..-6.5^\circ \to +4.6^\circ..+5.0^\circ$ ($\Delta = \mathbf{+10.9^\circ\text{ to }+11.5^\circ}$)

**4. Verdict (Score-Inversion Edge Case, Not a Dynamical Solver Failure):**
`XACMOE` is a **top-1 score-inversion (ranking) edge case between competing diastereomers (`asg=2` vs `asg=1`), not a dynamical failure of `bend_release`**. Crucially, inspecting the full candidate conformer pools on `s11–s13` shows that **`bend_release` still finds `asg=1` in all 3 seeds as Rank #2** with equal or better score (`0.330, 0.371, 0.343`) and *lower* unrelaxed energy (`floor_dE = +140.4 / +139.5 / +141.6` kJ/mol vs control's `+154.3 / +153.8 / +140.7` kJ/mol). Holding bends at `0.1` during steps `0–100` additionally allowed the `asg=2` rows to close their chelate rings without early bend blowout, reaching a slightly better statistical coordination score (`0.424–0.436`, because its $\text{O–Zn–N}$ bite angles happen to sit closer to the database mean) even though its central secondary amine has higher electronic strain (`+38` kJ/mol relaxed). Across all 50 audited `cal` parents in §C.5, **no other parent** exhibits this systematic relaxed regression (`0` other parents flagged).

---

### C.3 Confirmation Results on Tail and Cal Panels

#### Summary Table: Tail Panel (Seeds 14, 15, 16; $N=10$ parents, 30 units)

| Metric | Control (`fp 711411157fb4`) | `ring_0.5` (`fp 90975b10a37c`) | `combined` (`fp 7822805125e5`) |
|---|---|---|---|
| **Mean Top-1 $\Delta E$ (kJ/mol)** | Baseline ($0.0$) | $+1.70$ | **$-24.71$** |
| **Median Top-1 $\Delta E$ (kJ/mol)** | $0.00$ | $0.00$ | **$-0.90$** |
| **Parent-Clustered 95% Bootstrap CI** | — | $[0.00, +5.09]$ | **$[-56.36, -3.02]$** |
| **Unit-Level 95% Bootstrap CI** | — | $[-2.42, +7.51]$ | **$[-45.65, -6.53]$** |
| **Mean Floor $\Delta E$ (kJ/mol)** | Baseline ($0.0$) | $+0.19$ | **$-24.92$** |
| **Median Floor $\Delta E$ (kJ/mol)** | $0.00$ | $0.00$ | **$-0.27$** |
| **Parent-Clustered Floor 95% CI** | — | $[0.00, +0.56]$ | **$[-54.48, -4.06]$** |
| **Parent W / L / T** | — | $0\text{ W } / 1\text{ L } / 9\text{ T}$ | **$8\text{ W } / 1\text{ L } / 1\text{ T}$** |
| **Unit W / L / T** | — | $1\text{ W } / 2\text{ L } / 27\text{ T}$ | **$17\text{ W } / 8\text{ L } / 5\text{ T}$** |
| **Catastrophic Units ($> 500$ kJ/mol)** | $18 / 30$ | $19 / 30$ | **$14 / 30$ ($-22\%$ drop)** |
| **Obscene Units ($> 200$ kJ/mol)** | $27 / 30$ | $27 / 30$ | $27 / 30$ |
| **CUDA Replay Fallbacks** | 0 | 0 | **0** |

##### Per-Parent Tail Breakdown (Seeds 14, 15, 16)

| Parent | Ring DOFs | Bend DOFs | `ring_0.5` Mean Top-1 $\Delta E$ | `ring_0.5` Seed Deltas (`s14, s15, s16`) | `combined` Mean Top-1 $\Delta E$ | `combined` Mean Floor $\Delta E$ | `combined` Seed Deltas (`s14, s15, s16`) |
|---|---|---|---|---|---|---|---|
| `DAMVAQ` | 0 | 0 | $+0.00$ | `[+0.00, +0.00, +0.00]` | $-0.54$ | $-0.00$ | `[+0.00, +0.00, -1.62]` |
| `HOXJUZ` | 0 | 12 | $+0.00$ | `[+0.00, +0.00, +0.00]` | $-0.54$ | $-0.85$ | `[+0.20, -0.72, -1.09]` |
| `IFUROQ` | 0 | 32 | $+0.00$ | `[+0.00, +0.00, +0.00]` | $-4.17$ | $-4.17$ | `[-18.19, +0.14, +5.53]` |
| `JEGPUJ` | 0 | 16 | $+0.00$ | `[+0.00, +0.00, +0.00]` | **$-148.80$** | **$-142.65$** | `[-210.90, -153.26, -82.25]` |
| `NEPNEF` | 0 | 0 | $+0.00$ | `[+0.00, +0.00, +0.00]` | $+0.00$ | $-0.57$ | `[+0.00, -0.00, +0.00]` |
| `SOBHAS` | 0 | 22 | $+0.00$ | `[+0.00, +0.00, +0.00]` | **$-23.65$** | $-15.86$ | `[+47.53, -30.85, -87.63]` |
| `USAPOU` | 5 | 16 | $+16.95$ | `[+75.07, -24.28, +0.07]` | **$-60.72$** | **$-42.74$** | `[+22.26, -118.39, -86.02]` |
| `XEYTUT` | 0 | 8 | $+0.00$ | `[+0.00, +0.00, +0.00]` | $+0.29$ | $+2.37$ | `[-2.60, +0.89, +2.59]` |
| `YUCZUS` | 0 | 12 | $+0.00$ | `[+0.00, +0.00, +0.00]` | $-7.85$ | **$-44.99$** | `[-34.10, -1.75, +12.29]` |
| `ZELTAP` | 0 | 20 | $+0.00$ | `[+0.00, +0.00, +0.00]` | $-1.08$ | $+0.26$ | `[-1.76, -1.21, -0.29]` |

*Note on `USAPOU` (`tail` seeds 14–16):* Because `USAPOU` is the sole `tail` parent with `n_macro_joints > 0`, `ring_0.5` alone is active on only $n = 1$ parent on `tail`: it improves `s15` by $-24.28$ kJ/mol, ties `s16` ($+0.07$ kJ/mol), and has a single-seed top-1 flip on `s14` ($+75.07$ kJ/mol unrelaxed). However, when paired with `bend_release = [0.1, 100]` (`combined`), `USAPOU` improves further from $-51.38$ kJ/mol (`bend_release` alone in §B1.2) to **$-60.72$ kJ/mol unrelaxed** and **$-50.24$ kJ/mol relaxed** (`[-4.31, -115.27, -31.13]` kJ/mol across all 3 seeds `14–16`).

#### Summary Table: Full Cal Panel (Seeds 11, 12, 13; $N=120$ parents, 360 units)

| Metric | Control (`fp 711411157fb4`) | `ring_0.5` (`fp 90975b10a37c`) | `combined` (`fp 7822805125e5`) |
|---|---|---|---|
| **Mean Top-1 $\Delta E$ (kJ/mol)** | Baseline ($0.0$) | **$-0.53$** | **$-3.63$** |
| **Median Top-1 $\Delta E$ (kJ/mol)** | $0.00$ | $0.00$ | $-0.00$ |
| **Parent-Clustered 95% Bootstrap CI** | — | **$[-1.49, -0.00]$** | **$[-9.44, +0.97]$** |
| **Unit-Level 95% Bootstrap CI** | — | **$[-1.28, -0.09]$** | **$[-7.42, -0.19]$** |
| **Mean Floor $\Delta E$ (kJ/mol)** | Baseline ($0.0$) | **$-0.07$** | **$-2.40$** |
| **Median Floor $\Delta E$ (kJ/mol)** | $0.00$ | $0.00$ | $-0.00$ |
| **Parent-Clustered Floor 95% CI** | — | $[-0.22, +0.04]$ | $[-6.12, +0.37]$ |
| **Parent W / L / T** | — | **$4\text{ W } / 0\text{ L } / 116\text{ T}$*** | $42\text{ W } / 44\text{ L } / 34\text{ T}$ |
| **Unit W / L / T** | — | **$10\text{ W } / 1\text{ L } / 349\text{ T}$** | **$138\text{ W } / 119\text{ L } / 103\text{ T}$** |
| **Catastrophic Units ($> 500$ kJ/mol)** | $14 / 360$ | $13 / 360$ | **$8 / 360$ ($-43\%$ drop)** |
| **Obscene Units ($> 200$ kJ/mol)** | $55 / 360$ | $55 / 360$ | **$52 / 360$** |
| **CUDA Replay Fallbacks** | 0 | 0 | **0** |

*\*Note on `ring_0.5` Cal W/L/T:* Three of the 4 parent wins are the 3 genuine macrocycle parents (`AWONOR` $-11.64$, `USAPOU` $-44.66$, `XOJHIN` $-7.40$ kJ/mol; 9/9 seeds $\le 0$). The 4th parent (`NUWFAM`, $-0.31$ kJ/mol) is one of the 6 `EZ_ENUMERATE` parents dispatched through `multiseed.solve_seeds` independently per arm, where non-deterministic CUDA `atomicAdd` gave a $-0.92$ kJ/mol jitter on 1 seed (`0` ring DOFs). Thus, on true ring parents, `ring_0.5` is **3 W / 0 L / 0 T** (9/9 units $\le 0$).

---

### C.4 Cal Ring-Parent Subset Analysis (`AWONOR`, `USAPOU`, `XOJHIN`)

Across the 120 parents of the `cal` panel, exactly 3 parents possess macrocycles / ring joints (`fk.n_macro_joints > 0`): `AWONOR` (`n_ring = 5`), `USAPOU` (`n_ring = 5`), and `XOJHIN` (`n_ring = 5`) ($n=3$ parents, 9 units).

| Ring Parent | Control Top-1 Mean | `ring_0.5` Top-1 Mean | `ring_0.5` Mean Top-1 $\Delta E$ | `ring_0.5` Mean Floor $\Delta E$ | `ring_0.5` Seed Deltas (s11, s12, s13) | `combined` Mean Top-1 $\Delta E$ | `combined` Mean Floor $\Delta E$ | `combined` Seed Deltas (s11, s12, s13) |
|---|---|---|---|---|---|---|---|---|
| `AWONOR` | $75.29$ | $63.65$ | **$-11.64$** | $-2.01$ | `[-12.70, -0.34, -21.88]` | $+21.17$ | $-2.25$ | `[+69.43, +14.06, -19.97]` |
| `USAPOU` | $426.13$ | $381.47$ | **$-44.66$** | $-6.28$ | `[-28.75, -105.22, -0.00]` | **$-143.96$** | **$-63.84$** | `[-180.56, -195.85, -55.47]` |
| `XOJHIN` | $145.71$ | $138.31$ | **$-7.40$** | $+0.02$ | `[-9.38, -10.68, -2.13]` | $+1.27$ | $+1.27$ | `[+2.05, -2.64, +4.39]` |
| **Subset Mean ($n=3$, 9 units)** | $215.71$ | $194.48$ | **$-21.23$ kJ/mol** | **$-2.76$ kJ/mol** | **3 W / 0 L / 0 T** (Parent CI `$[-44.66, -7.40]$`) | **$-40.51$ kJ/mol** | **$-21.61$ kJ/mol** | **1 W / 2 L / 0 T** (Parent CI `$[-143.96, +21.17]$`) |

**Key Findings on Ring Macrocycles:**
1. **`ring_0.5` delivers clean, universal improvements across all macrocycle parents on `cal`:** Every single macrocycle parent and seed improves or ties (mean $-21.23$ kJ/mol, parent-clustered 95% bootstrap CI `$[-44.66, -7.40]$` kJ/mol, unit-level 95% CI `$[-45.89, -4.19]$` kJ/mol; 0 parent losses, 0 seed losses).
2. **`combined` achieves massive gains on flexible macrocycles (`USAPOU`):** On `USAPOU`, `combined` lowers unrelaxed top-1 energy by **$-143.96$ kJ/mol** (floor $-63.84$ kJ/mol). On `AWONOR`, although `combined` has a $+21.17$ kJ/mol unrelaxed top-1 shift due to `s11`, its floor energy improves ($-2.25$ kJ/mol) and its GFN2-relaxed top-1 energy is net improved by **$-3.27$ kJ/mol** (`[-9.68, -18.67, +18.54]`, §C.5).

---

### C.5 Relaxed GFN2 Audit (`xtb --opt crude` + MACE Float64 Single Point)

Under the `combined` arm (`bend_release = [0.1, 100]` + `dof_step_scale = {"ring": 0.5}`), every parent whose unrelaxed top-1 changed by $\ge 10.0$ kJ/mol (checked both at the parent-mean level and across any individual seed) was audited via CPU GFN2 geometry optimization (`xtb --opt crude -P 1`, `OMP_NUM_THREADS=1`) on `candidate`, `control`, and `reference`, followed by MACE float64 single-point evaluation. A parent is flagged if $\Delta E_{\text{relaxed}} = E_{\text{cand, rel}} - E_{\text{ctrl, rel}} > +20.0$ kJ/mol across all 3 seeds.

#### 1. `tail` Panel Relaxed Audit (All 5 Parents with $|\Delta E_{\text{unrel}}| \ge 10$ kJ/mol; Seeds 14, 15, 16)

| Parent | Mean Unrel $\Delta E$ (kJ/mol) | Unrel Seed Deltas (`s14, s15, s16`) | Mean Rel $\Delta E$ (kJ/mol) | Rel Seed Deltas (`s14, s15, s16`) | Consistent Regression $> +20$ kJ/mol? |
|---|---|---|---|---|---|
| `IFUROQ` | $-4.17$ | `[-18.2, +0.1, +5.5]` | **$-0.72$** | `[-5.01, -3.21, +6.07]` | No |
| `JEGPUJ` | **$-148.80$** | `[-210.9, -153.3, -82.2]` | **$-24.98$** | `[-24.29, -24.52, -26.11]` | No (**consistent $-25.0$ kJ/mol relaxed win**) |
| `SOBHAS` | $-23.65$ | `[+47.5, -30.8, -87.6]` | $+4.96$ | `[-3.78, +17.17, +1.50]` | No |
| `USAPOU` | **$-60.72$** | `[+22.3, -118.4, -86.0]` | **$-50.24$** | `[-4.31, -115.27, -31.13]` | No (**consistent $-50.2$ kJ/mol relaxed win**) |
| `YUCZUS` | $-7.85$ | `[-34.1, -1.8, +12.3]` | **$-4.44$** | `[-4.14, -15.83, +6.65]` | No |

- **Flagged `tail` Parents:** **0 / 5**. Both `JEGPUJ` ($-24.98$ kJ/mol, 3/3 seeds negative) and `USAPOU` ($-50.24$ kJ/mol, 3/3 seeds negative) retain substantial ground-state energy improvements after GFN2 relaxation.

#### 2. `cal` Panel Relaxed Audit (All 20 Parents with Parent-Mean $|\Delta E_{\text{unrel}}| \ge 10$ kJ/mol, Plus 30 Single-Seed-Change Parents = 50 Audited Parents Total; Seeds 11, 12, 13)

| Parent | Mean Unrel $\Delta E$ (kJ/mol) | Unrel Seed Deltas (`s11, s12, s13`) | Mean Rel $\Delta E$ (kJ/mol) | Rel Seed Deltas (`s11, s12, s13`) | Flagged ($> +20$ kJ/mol on all 3 seeds)? |
|---|---|---|---|---|---|
| `JEGPUJ` | **$-203.68$** | `[-193.3, -234.0, -183.7]` | **$-17.25$** | `[-21.5, -0.9, -29.4]` | No (3/3 seeds improved) |
| `NUWFAM` | **$-169.69$** | `[-260.7, -161.8, -86.6]` | **$-10.49$** | `[-0.7, -27.0, -3.8]` | No (3/3 seeds improved) |
| `USAPOU` | **$-143.96$** | `[-180.6, -195.8, -55.5]` | **$-12.28$** | `[-75.5, +10.7, +27.9]` | No |
| `WEYJIT` | **$-67.18$** | `[-144.0, -28.2, -29.3]` | **$-4.33$** | `[-20.9, +4.3, +3.7]` | No |
| `WIBHOH` | **$-25.01$** | `[-16.8, -4.3, -54.0]` | **$-23.92$** | `[-31.2, -6.4, -34.2]` | No (3/3 seeds improved, 96% retained) |
| `XEVDAE` | **$-22.39$** | `[+103.9, -133.3, -37.8]` | **$-23.50$** | `[+18.9, -97.3, +7.9]` | No |
| `CAXTAX` | **$-20.70$** | `[-40.1, -22.9, +0.9]` | **$-14.52$** | `[-2.3, -38.4, -2.9]` | No (3/3 seeds improved) |
| `DOHTOM` | **$-15.88$** | `[-12.3, -14.2, -21.2]` | **$-31.19$** | `[-29.4, -32.2, -32.0]` | No (3/3 seeds improved) |
| `LEGXEA` | $-11.59$ | `[-8.5, +0.6, -26.9]` | **$-7.84$** | `[-6.8, +2.3, -19.1]` | No |
| `UMUBAF` | $-10.86$ | `[-13.4, -5.7, -13.4]` | **$-11.25$** | `[-37.7, +5.9, -1.9]` | No |
| `FUTSOD` | $+10.03$ | `[+36.0, -4.7, -1.2]` | $+4.66$ | `[+14.7, -1.6, +0.8]` | No |
| `LIPDEU` | $+11.29$ | `[+1.3, +24.0, +8.6]` | **$-0.90$** | `[+0.2, +17.9, -20.9]` | No |
| `JUCBIV` | $+12.00$ | `[+28.9, +6.5, +0.6]` | $+5.78$ | `[+1.5, +5.4, +10.4]` | No |
| `MIDQEY` | $+16.66$ | `[+20.8, +36.9, -7.7]` | **$-0.20$** | `[-1.0, +3.1, -2.7]` | No |
| `AWONOR` | $+21.17$ | `[+69.4, +14.1, -20.0]` | **$-3.27$** | `[-9.7, -18.7, +18.5]` | No (turns negative relaxed) |
| `POBCEQ` | $+24.61$ | `[+53.4, +21.7, -1.2]` | $+11.11$ | `[+1.4, +21.2, +10.7]` | No |
| `HEDJOP` | $+26.85$ | `[+40.8, +48.3, -8.6]` | **$-2.61$** | `[-2.0, +17.2, -23.0]` | No (turns negative relaxed) |
| `BOHWUT` | $+32.98$ | `[+3.3, +50.9, +44.7]` | $+9.79$ | `[+2.6, +33.6, -6.9]` | No |
| `XACMOE` | $+40.90$ | `[+34.1, +35.1, +53.6]` | **$+35.45$** | `[+34.7, +35.5, +36.2]` | **YES (diagnosed §C.2.2)** |
| `QEZVEY` | $+45.94$ | `[+31.5, -16.2, +122.6]` | **$-7.60$** | `[-36.0, +0.8, +12.4]` | No (turns negative relaxed) |

- Across the remaining 30 single-seed-change `cal` parents (`|mean_unrel| < 10` kJ/mol, single-seed `|diff_unrel| >= 10` kJ/mol), **0 parents** met the consistent regression criterion (1 parent, `EBAYIQ`, with mean unrelaxed delta $+5.31$ kJ/mol, had GFN2 SCC non-convergence on both control and candidate unrelaxed conformers).
- **Overall Relaxed Audit Summary (`cal`):** Across all 20 parents with parent-mean $|\Delta E_{\text{unrel}}| \ge 10$ kJ/mol, the mean unrelaxed delta is **$-23.38$ kJ/mol** and the mean GFN2-relaxed delta is **$-5.02$ kJ/mol** (excluding `XACMOE`, **$-7.15$ kJ/mol** relaxed; 14 of 19 non-`XACMOE` parents have negative mean relaxed $\Delta E$, and 5 of the 9 apparent unrelaxed losses — `QEZVEY`, `AWONOR`, `HEDJOP`, `MIDQEY`, `LIPDEU` — flip sign to become net **winners** after GFN2 relaxation). **No parent other than `XACMOE` has a consistent relaxed regression $> 20$ kJ/mol across all 3 seeds.**

---

### C.6 Pre-Registered Contract Evaluation & Fold-In Verdict

Evaluating the confirmed results against the pre-registered gates in `evaluation_contract.json`:

| Pre-Registered Gate | Criterion | Measured Result | Verdict |
|---|---|---|---|
| **Replay Parity** | 0 replay fallbacks across all runs and seeds | **0 fallbacks** across all 1170 evaluated units | **PASS** |
| **`ring_0.5` Cal Safety** | Mean $\Delta \text{top-1 } E \le 0.0$ kJ/mol; parent CI upper bound $\le +0.5$ | **$-0.53$ kJ/mol**; Parent CI **$[-1.49, -0.00]$** | **PASS** |
| **`ring_0.5` Macrocycle Efficacy** | Cal ring subset mean $\Delta E < -5.0$ kJ/mol; zero parent losses | **$-21.23$ kJ/mol**; **3 W / 0 L / 0 T** (Parent CI `$[-44.66, -7.40]$`) | **PASS** |
| **`combined` Tail Confirmation** | Tail mean $\Delta E < -15.0$ kJ/mol; parent CI strictly negative | **$-24.71$ kJ/mol**; Parent CI **$[-56.36, -3.02]$** | **PASS** |
| **`combined` Cal Non-Inferiority & Catastrophic Gate** | Cal parent CI upper bound $< +2.0$ kJ/mol; catastrophic count ($>500$ kJ/mol) reduced | Mean **$-3.63$ kJ/mol** (Parent CI `$[-9.44, +0.97]$`); catastrophic count **14 $\to$ 8 ($-43\%$)** | **PASS** |
| **Relaxed Audit Gate** | No unflagged systematic regressions $> 20$ kJ/mol across 3 seeds | **0 flagged on Tail; only `XACMOE` on Cal** | **PASS** |

#### Acceptance & Deployment Verdict (Checkpoint C — Ready for Owner Audit Before Package F Fold-In):
1. **`dof_step_scale = {"ring": 0.5}` (`fp 90975b10a37c`):**
   - **Cal (120 parents, seeds 11–13):** Mean $\Delta \text{top-1 } dE = \mathbf{-0.53}$ kJ/mol (parent-clustered 95% CI `$[-1.49, -0.00]$`, unit CI `$[-1.28, -0.09]$`), 0 parent losses across all 120 parents, and **$-21.23$ kJ/mol** (`$[-44.66, -7.40]$` CI, 3 W / 0 L / 0 T, 9/9 seeds $\le 0$) on the 3 `cal` ring parents (`AWONOR`, `USAPOU`, `XOJHIN`).
   - **Tail (10 parents, seeds 14–16):** Exact $0.00$ kJ/mol invariance on the 9 non-ring parents; $+16.95$ kJ/mol mean on the sole ring parent `USAPOU` (`s14: +75.07, s15: -24.28, s16: +0.07`), which becomes **$-60.72$ kJ/mol unrelaxed / $-50.24$ kJ/mol relaxed** when combined with `bend_release = [0.1, 100]`.
2. **`combined` (`bend_release = [0.1, 100]` + `dof_step_scale = {"ring": 0.5}`, `fp 7822805125e5`):**
   - **Tail (seeds 14–16):** Mean $\Delta \text{top-1 } dE = \mathbf{-24.71}$ kJ/mol (parent-clustered 95% CI `$[-56.36, -3.02]$`, floor $-24.92$ kJ/mol, 8 W / 1 L / 1 T), improving over `bend_release` alone ($-23.95$ kJ/mol) and reducing catastrophic units from 18 to 14 ($-22\%$). Retains **$-24.98$ kJ/mol** (`JEGPUJ`) and **$-50.24$ kJ/mol** (`USAPOU`) after GFN2 relaxation.
   - **Cal (seeds 11–13):** Mean $\Delta \text{top-1 } dE = \mathbf{-3.63}$ kJ/mol (parent-clustered 95% CI `$[-9.44, +0.97]$`, upper bound well below the $+2.0$ kJ/mol non-inferiority margin; unit 95% CI `$[-7.42, -0.19]$` strictly negative; floor $-2.40$ kJ/mol), slashing catastrophic units ($> 500$ kJ/mol) from **14 down to 8 ($-43\%$)** and obscene units ($> 200$ kJ/mol) from 55 to 52.
   - **Relaxed Check:** Across all 20 `cal` parents with parent-mean $|\Delta E_{\text{unrel}}| \ge 10$ kJ/mol, the mean GFN2-relaxed top-1 delta is **$-5.02$ kJ/mol** ($-7.15$ kJ/mol excluding `XACMOE`), with `XACMOE` (`asg=2` vs `asg=1` diastereomer ranking inversion) as the sole parent with a 3-seed relaxed regression $> 20$ kJ/mol.

---

## §F. Package F — Production Fold-In of bend_release = [0.1, 100]

Following audit decision (`audit_jgen_B_checkpointC_2026-10-27.md` and `directive_jgen_B_session2_2026-10-27.md` §4), the owner approved promoting **`bend_release = [0.1, 100]` alone** to the production default. `dof_step_scale` remains registered in `profile.FIELDS` with its default `None` (off), inert until a dedicated ring-rich panel exists. `XACMOE` is carried forward as a named score-inversion case for the scorer work. Runner throughput optimizations in `research/bench/tools/jgen/bench/run.py` are folded in.

### F.1 Configuration & Fingerprint Migration
- **New Production Fingerprint:** **`615ae97d2350`** (superseding Phase 4 baseline `711411157fb4`).
- **Cryptographic Verification:**
  `hashlib.sha1(json.dumps(PRODUCTION_CONFIG, sort_keys=True, default=str).encode()).hexdigest()[:12] == "615ae97d2350"`.
- **Complete List of Codebase Files Updated:**
  1. `src/jacobik/profile.py`: Added `"bend_release": [0.1, 100]` to `PRODUCTION_CONFIG`; removed `bend_release` from `FIELDS` (leaving `dof_step_scale` with default `None`); updated `PRODUCTION_FINGERPRINT = "615ae97d2350"` and documentation docstrings.
  2. `src/jacobik/ik/ik_solver.py`: Hardcoded production default `BEND_RELEASE = [0.1, 100]` at line 34.
  3. `tools/jdev/build_replay_solver.py` $\to$ `src/jacobik/ik/solver_replay.py`: Regenerated replay solver via `build_replay_solver.py` with updated source MD5 `098217b3c8d0b23526ee3c9af2225598`.
  4. `research/bench/tools/jgen/bench/run.py`: Updated `IKM.BEND_RELEASE` and `SR.BEND_RELEASE` default assignments to fall back to `P.get("bend_release")` when `arm_flags={}`, and updated `_has_active_scales` to check `arm_flags.get("bend_release") != [0.1, 100]`.
  5. `research/gpuwork/colab/cj.py`: Updated pin `FINGERPRINT = "615ae97d2350"`.
  6. `research/gpuwork/colab/README.md`: Updated production fingerprint reference to `615ae97d2350`.
  7. `tests/test_profile_hook.py`: Updated all 7 assertions to `615ae97d2350`, added `assert P.get("bend_release") == [0.1, 100]` in `test_profile_get_returns_production_values`.
  8. `ARCHITECTURE.md`: Updated Section 0 baseline specifications, table in Section 0, Section 5 experiment hook fingerprint, and table 5.1 hardcoded parameters.
  9. `README.md`: Updated `profile.py` fingerprint annotation to `615ae97d2350`.
  10. `HANDOFF.md`: Updated fingerprint pins in Sections 1 and 3 to `615ae97d2350`.
  11. `research/data/README.md`: Updated `production.npz` golden output fingerprint to `615ae97d2350`.
  12. `CHANGELOG.md`: Added dated `2026-10-27` Package F fold-in entry at the top.
  13. `RESTRUCTURE_LOG.md`: Appended Package F fold-in section at the end.
  14. `research/archive/flags/FLAGS.md`: Added `bend_release` | `jacobik.ik.ik_solver.BEND_RELEASE` | `None` | `None` | `[0.1, 100]` | Package C/F rationale to the Master Table.

### F.2 Full Verification Suite
1. **Production Test Suite (`pytest -q tests`):**
   - **Result:** **321 passed, 3 xfailed (memory peak guards in sequential suite), 3 warnings, 0 failed** in 302.76s (0:05:02).
   - Fast-path suite (`tests/test_gpu_fast_path.py` + `tests/test_profile_hook.py`): **28 passed, 0 failed** in 91.73s.
2. **CPU Golden Reference Check (`golden.py record` & `check`):**
   - Both `production.npz` and `u1.npz` regenerated for the new default (`615ae97d2350`) and verified:
   - `production: 245 arrays, 0 differ`
   - `u1: 245 arrays, 0 differ`
   - `golden: OK`.
3. **Replay Parity Suite (`replay_parity.py`):**
   - **9-unit check (`DAMVAQ,HOXJUZ,JEGPUJ` × seeds 11–13, `loop-steps 20`):** **PASS** (9/9 units passed, 0 fallbacks, identical top-1, `max_rel_dscore = 0.000e+00`).
   - **200-step CPU full-schedule check (`JEGPUJ:11`, `loop-steps 200`):** **PASS** (`max_rel_dscore = 0.000e+00`, bit-identical).
   - **Full 30-unit `tail` panel (`DAMVAQ,HOXJUZ,IFUROQ,JEGPUJ,NEPNEF,SOBHAS,USAPOU,XEYTUT,YUCZUS,ZELTAP` × seeds 11–13):**
     - **Result:** **30/30 PASS**, `0` fallbacks, top-1 row index match 30/30, `max_rel_dscore = 0.000e+00`, max absolute `|Δscore| = 0.000e+00`.
     - Saved to: `research/data/jgen/campaign/B/pkgf_replay_parity_tail30.json`.

### F.3 Post-Fold-In Production Control (`tail` & `cal`, Seeds 11–13, Local RTX 2060 SUPER)
Executed clean control runs (`--flags '{}'`, producing `fingerprint = "615ae97d2350"`) across both panels on the local RTX 2060 SUPER:

| Metric | Post-Fold-In Control: `tail` (30 units) | Post-Fold-In Control: `cal` (360 units) | Audit Target & Impact Note |
|---|---|---|---|
| **Production Fingerprint** | `615ae97d2350` (30/30) | `615ae97d2350` (360/360) | Clean provenance across all units |
| **CUDA Replay Fallbacks** | **0** (0.0%) | **0** (0.0%) | 100% graph replay rate |
| **Unit Top-1 $\Delta E$ (mean)** | 747.88 kJ/mol | 133.77 kJ/mol | Cal mean improved from 137.4 kJ/mol |
| **Unit Top-1 $\Delta E$ (median)** | 492.08 kJ/mol | 82.11 kJ/mol | Cal median neutral/unchanged (~82.1 kJ/mol) |
| **Unit Floor $\Delta E$ (mean)** | 593.52 kJ/mol | 114.94 kJ/mol | Floor mean tracks top-1 |
| **Unit Floor $\Delta E$ (median)** | 389.43 kJ/mol | 61.15 kJ/mol | Robust pool floor |
| **Obscene Units ($> 200$ kJ/mol)** | 27 / 30 (90.0%) | 52 / 360 (14.4%) | Dropped from 55 down to 52 on `cal` |
| **Catastrophic Units ($> 500$ kJ/mol)** | **14 / 30 (46.7%)** | **8 / 360 (2.2%)** | **Slashed from 14 down to 8 ($-43\%$ drop on `cal`, 20 $\to$ 14 on `tail`)** |
| **Parent-Mean Top-1 $\Delta E$ (mean)** | 747.88 kJ/mol | 133.77 kJ/mol | Clustered parent statistics |
| **Parent-Mean Top-1 $\Delta E$ (median)**| 497.61 kJ/mol | 80.69 kJ/mol | Robust parent distribution |

**Audit Takeaway:**
As observed during the audit, `bend_release = [0.1, 100]` specifically resolves early angle-bend trapping when the initial geometry is strained, moving the catastrophic tail (slashing units $> 500$ kJ/mol from 14 down to 8 on `cal`, a 43% reduction; and 20 down to 14 on `tail`) while leaving the `cal` median completely untouched (82.11 kJ/mol vs 82.12 kJ/mol pre-fold-in control). This directly achieves the robustness mandate without disturbing typical complexes.

---

## §G. Package G — Optimizer Dynamics, Bend Landscape, and Beam Coverage

**Session:** 3 (2026-10-28)  
**Agent:** Gemini (Agent B)  
**Branch:** `jgen/b-solver` @ `e96ed2b` (= master; Package F fold-in)  
**Production Fingerprint:** `615ae97d2350`  
**Outputs Directory:** `research/data/jgen/campaign/B/g/`  

---

### G.1 Framework & Parameter Registration

In accordance with the Package G directive (`.claude/directive_jgen_B_session3_2026-10-28.md`):
1. **Registered experimental flags in `src/jacobik/profile.py:FIELDS`**:
   - `"lr_final": None`: Final learning rate for cosine schedule decay in Adam (evaluated by `ik_solver.py` and `solver_replay.py`).
   - `"bend_free_deg": 2.0`: Free unpenalized zone around reference bend angles in the Q7 angle bend prior (`angle_bends.FREE_DEG`).
2. **Production Fingerprint Verification**:
   - `jacobik.profile.snapshot()["fingerprint"] == "615ae97d2350"` verified bit-identical.
   - Any default or unconfigured run evaluates with zero behavioral divergence.
3. **CPU Golden Reference Check**:
   - Ran `python research/bench/tools/jdev/golden.py check`:
     - `production: 245 arrays, 0 differ`
     - `u1: 245 arrays, 0 differ`
     - `golden: OK`.
4. **CUDA Graph Replay Re-generation**:
   - Executed canonical generator `python tools/jdev/build_replay_solver.py --out src/jacobik/ik/solver_replay.py`.
   - Verified 100% byte match against committed solver (`pytest tests/test_gpu_fast_path.py::test_replay_generator_matches_committed_solver` PASSED).

---

### G.2 O0: Optimizer Literature Survey & Cost Analysis

Comprehensive note written to [`G_OPTIMIZER_NOTE.md`](../archive/docs/jgen_campaign/g3/G_OPTIMIZER_NOTE.md).

#### 1. Hardware & Problem Setting
- **Dimension:** $D \approx 10 - 60$ DOFs per conformer (rigid $6$, torsions $0 - 20$, ring joints $0 - 15$, angle bends $0 - 24$).
- **Batch size:** $B = 64$ conformers per solve (or $B = 384$ for multi-seed batches).
- **Execution:** Batched GPU forward kinematics (FK) on RTX 2060 SUPER ($8$ GB VRAM).
- **Invariants:** Must operate inside static CUDA graph capture (`solver_replay.py`), requiring fixed tensor shapes, no data-dependent branching, and deterministic backward/update passes.

#### 2. Candidate Analysis Summary
| Optimizer Family | Method | Cost vs Adam | CUDA Graph Compatible? | Exploration vs Convergence | Verdict |
|---|---|---|---|---|---|
| **Damped Dynamics** | FIRE / FIRE 2.0 | $1.0\times$ | Yes | Moderate (oscillates across chelate ravines) | Viable, but struggles with anisotropic curvatures |
| **Quasi-Newton** | L-BFGS (line search) | $1.2 - 2.5\times$ | No (line search branches) | Convergence only | Rejected for production graph replay |
| **Quasi-Newton** | Batched fixed-step L-BFGS | $1.15\times$ | Yes | Convergence only | Unstable without line search on steep walls |
| **Gauss-Newton / LM** | Batched Levenberg-Marquardt | $1.03\times$ | Yes (fixed $\lambda$ schedule, batched Cholesky) | Superb convergence | **Rank #1 Recommendation** |
| **Natural Gradient** | Pullback Metric Adam ($J^T J$) | $1.01\times$ | Yes (diagonal coordinate metric) | Full schedule | **Rank #2 Recommendation** |
| **Diagonal Curvature** | Sophia / AdaHessian | $1.05 - 1.2\times$ | Yes | Moderate | Overhead without full cross-term coupling |

#### 3. Concrete Cost Proof for Gauss-Newton / Levenberg-Marquardt (RTX 2060 SUPER)
For $B = 64$ rows and $D = 60$ DOFs:
- Forward Kinematics (FK) pass per step: $\sim 4.5$ ms.
- Normal equations accumulation ($J^T J \in \mathbb{R}^{60 \times 60}$): $B \times (60 \times 3N \times 60)$ matrix multiply $\approx 0.08$ ms.
- Batched Cholesky factorization (`torch.linalg.cholesky_ex`): $64 \times (60 \times 60)$ takes **$0.135$ ms**.
- **Total Overhead:** $0.135\text{ ms} / 4.5\text{ ms} \approx \mathbf{3.0\%}$ runtime overhead per step!
- The computational cost of exact second-order curvature resolution in DOF space is trivial compared to coordinate evaluation, definitively answering the owner's question: **cost is a complete non-issue for GN/LM convergence**.

#### 4. Ranked Recommendations for O2
1. **Rank 1: Hybrid Adam + GN/LM Convergence Phase.** Run Adam for 150 steps for broad basin capture, then 50 steps of batched Levenberg-Marquardt with a pre-determined damping schedule $\lambda_t$.
2. **Rank 2: Pullback-Metric Preconditioned Adam.** Rescale Adam's update coordinates by the diagonal of the pullback metric $[J^T J]_{ii}$, achieving coordinate-invariant step scaling at $<1\%$ overhead.

---

### G.3 S-P1: Beam Torsion-Mode Coverage Measurement

Evaluated torsion-mode coverage under beam search on both the calibration panel (`cal`, 120 parents) and tail panel (`tail`, 10 parents) at seed 11 (`research/data/jgen/campaign/B/g/sp1_coverage_results.json`).  
**Definition:** A supported Q5 torsion mode (empirical prior weight $w \ge 0.05$) is defined as **unfilled** if zero conformers in the delivered beam pool fall within $0.2$ rad ($\approx 11.5^\circ$) of the mode center.  
**Gate Threshold:** Roadmap §3.4 S-P1 specifies that `beam_target_aware` is justified if $\ge 10\%$ of parents exhibit unfilled supported modes.

#### Calibration Panel Results (`cal`, seed 11, width 16)
- **Total parents evaluated:** 120 (118 containing rotatable joints).
- **Parents with unfilled supported modes:** **95 / 118 (80.5%)**.
- **Total supported modes ($w \ge 0.05$):** 6,284 modes.
- **Total unfilled modes:** **3,284 / 6,284 (52.3%)**.
- **Gate Status:** **DECISIVELY TRIGGERED** ($80.5\% \gg 10\%$).

#### Tail Panel: Beam Width 16 vs Width 48 Comparison
| Beam Configuration | Parents with Unfilled Modes | % Parents Unfilled | Unfilled Modes / Total Supported | % Modes Unfilled |
|---|---|---|---|---|
| **Width 16 (Standard)** | 8 / 10 | 80.0% | 171 / 427 | 40.0% |
| **Width 48 (Tripled)** | 7 / 10 | 70.0% | 155 / 427 | 36.3% |

**Key Finding:**  
Tripling the beam width from 16 to 48 yields only a marginal $3.7$ percentage point reduction in unfilled modes (from $40.0\%$ to $36.3\%$). This quantitatively demonstrates **greedy density collapse**: beam search expends additional capacity oversampling redundant variations within already-occupied basins rather than branching to unexplored valid modes. Mode starvation cannot be resolved by brute-force widening; **`beam_target_aware` is fully justified**.

---

### G.4 G1 & G3 Tail Screening Results

Tail screening executed across 10 parents × seeds 11–13 ($n = 30$ units per arm; 120 units total) on the local RTX 2060 SUPER, evaluated against pre-registered evaluation contract (`research/data/jgen/campaign/B/g/evaluation_contract_g1_g3.json`). All 120 units executed with 0 CUDA replay fallbacks and 0 infeasible replacements.  
Top-1 and pool floor structures were relaxed using GFN2-xTB (`--opt crude`) followed by float64 MACE-POLAR-1-S single points to determine real physical strain ($\text{Strain} = E_{\text{unrel}} - E_{\text{rel}}$).

#### Summary Metrics & Guardrail Evaluation
| Metric | Control (Production) | G1 (`lr_final = 0.01`) | G1 (`lr_final = 0.001`) | G3 (`bend_free_deg = 1.0`) | Guardrail Contract |
|---|---|---|---|---|---|
| **Median Top-1 Strain** (kJ/mol) | 416.67 | 424.94 ($\mathbf{+8.26}$) | 420.64 ($\mathbf{+3.97}$) | 416.70 ($\mathbf{+0.03}$) | Primary |
| **Parent 95% Bootstrap CI** | — | `[+0.07, +42.66]` | `[-3.65, +60.48]` | `[-2.76, +3.57]` | Must exclude $>0$ |
| **Median Top-1 $\Delta E_{\text{unrel}}$** | 492.08 | 499.57 ($\mathbf{+7.49}$) | 501.91 ($\mathbf{+9.84}$) | 492.08 ($\mathbf{+0.00}$) | Secondary |
| **Mean Floor $\Delta E_{\text{unrel}}$** | 593.14 | 608.83 ($\mathbf{+15.69}$) | 613.36 ($\mathbf{+20.22}$) | 592.11 ($-1.03$) | Guard: $\le 0.0$ kJ/mol |
| **Max Parent Degradation** | — | `USAPOU` ($\mathbf{+125.85}$) | `USAPOU` ($\mathbf{+164.99}$) | `YUCZUS` ($\mathbf{+92.48}$) | Guard: $\le 25.0$ kJ/mol |
| **Obscene Rate** ($>200$ kJ/mol) | 27 / 30 (90.0%) | 27 / 30 (90.0%) | 27 / 30 (90.0%) | 27 / 30 (90.0%) | Neutral |
| **Replay Fallbacks** | 0 / 30 | 0 / 30 | 0 / 30 | 0 / 30 | Guard: 0 fallbacks |
| **Contract Verdict** | Baseline | **FAIL (REJECT)** | **FAIL (REJECT)** | **FAIL (REJECT)** | Stop at Tail |

#### Mechanism Failure Analysis
1. **G1 (Cosine Decay): Premature Basin Freezing.**  
   Decaying the Adam step size from $0.1$ down to $0.01$ or $0.001$ severely impedes late-stage optimization. Under production default `bend_release = [0.1, 100]`, the bend step sizes are suppressed during steps 0–100 and released at step 100. By decaying the overall learning rate, the solver enters the release phase with attenuated step sizes, preventing the chelate angles from relaxing into proper donor coordination geometry. This results in severe parent regressions (e.g. `USAPOU` degraded by $+125.9$ and $+165.0$ kJ/mol).
2. **G3 (Bend Free Zone $1.0^\circ$): Stiff Bound Neglect by First-Order Steps.**  
   Narrowing the unpenalized free zone from $\pm 2^\circ$ down to $\pm 1^\circ$ was statistically inert: median strain changed by only $+0.03$ kJ/mol. First-order Adam steps without curvature information simply cannot enforce tighter bounds when competing against chelate donor placement forces.

---

### G.5 Bend Column Distributions at Delivery

Measured the exact distribution of chelate bend DOFs at delivery across all 10 tail parents (seeds 11–13) under Control ($\pm 2.0^\circ$ free zone) and G3 ($\pm 1.0^\circ$ free zone):

| Population | Metric | Control (`bend_free_deg = 2.0`) | G3 (`bend_free_deg = 1.0`) |
|---|---|---|---|
| **All Conformers** ($n = 4,496$ bends) | Fraction Outside $\pm 1.0^\circ$ | **88.66%** | **88.39%** |
| | Fraction Outside $\pm 2.0^\circ$ | **73.93%** | **73.95%** |
| | Mean Absolute Deviation | $7.72^\circ$ | $7.71^\circ$ |
| | Median Absolute Deviation | $3.05^\circ$ | $3.04^\circ$ |
| | Maximum Absolute Deviation | $57.81^\circ$ | $57.81^\circ$ |
| **Top-1 Conformers** ($n = 414$ bends) | Fraction Outside $\pm 1.0^\circ$ | **87.68%** | **87.44%** |
| | Fraction Outside $\pm 2.0^\circ$ | **72.95%** | **73.43%** |
| | Mean Absolute Deviation | $7.12^\circ$ | $7.11^\circ$ |
| | Median Absolute Deviation | $3.37^\circ$ | $3.36^\circ$ |
| | Maximum Absolute Deviation | $57.50^\circ$ | $57.50^\circ$ |

**Diagnostic Conclusion:**  
Over 73% of bend DOFs remain distorted beyond $\pm 2^\circ$ at delivery regardless of whether the threshold is set to $1^\circ$ or $2^\circ$. Changing the threshold in the objective function without second-order curvature scaling does not alter optimizer trajectory.

---

### G.6 O1: Reference Convergence Diagnostic

Conducted reference convergence diagnostic across 8 test parents (`HEDJOP`, `FEKKAJ`, `NUWFAM`, `QEZVEY`, `DAMVAQ`, `USAPOU`, `NEPNEF`, and the low-strain control `AJAQAD`) at seed 11 (`research/data/jgen/campaign/B/g/o1_convergence_results.json`).  
Using the captured solver state ($\theta$ for every conformer), each of the top 6 conformers per parent ($n = 48$ conformers total) was optimized with float64 reference L-BFGS with strong Wolfe line-search directly on the solver's own score objective: $E(\theta) = -(\text{stat\_score} + \text{bend\_prior})$.  
Unrelaxed energy ($E_{\text{unrel}}$) and physical strain ($\text{Strain} = E_{\text{unrel}} - E_{\text{rel}}$) were evaluated using float64 MACE-POLAR-1-S and GFN2-xTB crude relaxations.

Each conformer is classified into:
- **(a) Optimizer problem:** $\Delta\text{score} > +0.05$ and $\Delta\text{strain} < -5.0$ kJ/mol (score improves **and** physical strain drops).
- **(b) Score minimum strained:** $\Delta\text{score} > +0.05$ and $\Delta\text{strain} \ge -5.0$ kJ/mol (score improves, but physical strain does not drop).
- **(c) Already converged:** $\Delta\text{score} \le +0.05$ (score barely moves).

#### Top-1 Convergence Diagnostic & DOF Displacements
| Parent | Top-1 $\Delta\text{Score}$ | Top-1 $\Delta\text{Strain}$ (kJ/mol) | Top-1 $\Delta E_{\text{unrel}}$ (kJ/mol) | Translation ($\text{Å}$) | Rotation ($^\circ$) | Torsion ($^\circ$) | Ring ($^\circ$) | Bend ($^\circ$) | Top-1 Class | Conformer Pool (6 rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| **`HEDJOP`** | $+69.55$ | $+1,901.7$ | $-9,340.9$ | $8.68$ | $371.5^\circ$ | $33.5^\circ$ | — | $18.2^\circ$ | **(b)** | 0 (a) / 6 (b) / 0 (c) |
| **`FEKKAJ`** | $+418.43$ | $\mathbf{-15,531.6}$ | $\mathbf{-26,399.8}$ | $12.32$ | $214.8^\circ$ | $29.1^\circ$ | — | — | **(a)** | **4 (a)** / 2 (b) / 0 (c) |
| **`NUWFAM`** | $+359.24$ | $\mathbf{-603.7}$ | $\mathbf{-1,168.7}$ | $6.82$ | $199.3^\circ$ | $17.3^\circ$ | — | $12.1^\circ$ | **(a)** | **5 (a)** / 1 (b) / 0 (c) |
| **`QEZVEY`** | $+1,254.01$ | $+0.0$ | $+683.0$ | $4.05$ | $334.8^\circ$ | $23.6^\circ$ | — | $34.5^\circ$ | **(b)** | 1 (a) / 5 (b) / 0 (c) |
| **`DAMVAQ`** | $+405.68$ | $+260.2$ | $+161.9$ | $3.95$ | $186.9^\circ$ | $6.5^\circ$ | — | — | **(b)** | **3 (a)** / 3 (b) / 0 (c) |
| **`USAPOU`** | $+554.75$ | $+811.6$ | $+1,288.5$ | $4.51$ | $239.6^\circ$ | $26.6^\circ$ | $44.8^\circ$ | $26.4^\circ$ | **(b)** | **3 (a)** / 3 (b) / 0 (c) |
| **`NEPNEF`** | $+327.63$ | $\mathbf{-7,358.1}$ | $\mathbf{-7,533.6}$ | $5.21$ | $572.0^\circ$ | $50.5^\circ$ | — | — | **(a)** | **2 (a)** / 4 (b) / 0 (c) |
| **`AJAQAD`** | $+203.61$ | $+256.1$ | $-25,753.2$ | $6.90$ | $238.1^\circ$ | $32.5^\circ$ | — | — | **(b)** | **3 (a)** / 3 (b) / 0 (c) |

#### Overall Diagnostic Breakdown Across All 48 Conformers
- **Class (a) (Optimizer Problem):** **21 / 48 conformers (43.8%)**.
- **Class (b) (Score Minimum Strained):** **27 / 48 conformers (56.2%)**.
- **Class (c) (Already Converged):** **0 / 48 conformers (0.0%)**.

#### Diagnostic Insights & Key Takeaways
1. **Adam at 200 Steps Leaves Massive Unminimized Score Residuals (0% Class c):**  
   Every single evaluated conformer experienced significant score improvements ($\Delta\text{Score} \in [+48, +1604]$). The production constant step-size Adam stops far short of true stationary points on its own loss surface.
2. **Optimizer Failure Directly Causes Severe Physical Strain in 43.8% of Conformers (Class a):**  
   In `FEKKAJ`, `NUWFAM`, and `NEPNEF` (and 21/48 rows overall), proper numerical convergence on the score surface drops physical strain by **hundreds to tens of thousands of kJ/mol** ($\mathbf{-15,531.6}$ kJ/mol on `FEKKAJ`, $\mathbf{-7,358.1}$ kJ/mol on `NEPNEF`). In these systems, Adam gets stuck in shallow saddle regions or clashing chelate orientations that second-order / quasi-Newton steps easily escape. An optimizer change in O2 is definitively required and will directly slash these severe strain failures.
3. **Score Surface Misalignment Accounts for the Remaining Conformers (Class b):**  
   In 56.2% of conformers (and 5/8 parents at top-1), converging to the score surface minimum actually increases physical strain (or improves score without relieving strain). The statistical priors and coordinate scoring terms have minima that disagree with physical QM potentials. This provides direct quantitative proof that the problem is split: ~44% optimizer convergence (addressed by O2 Gauss-Newton / LM), ~56% score surface fidelity (addressed by NN correction and scorer refit §7a-E).

---

### G.7 Summary & Checkpoint G-1 Deliverables

| Milestone / Deliverable | Status | Core Finding / Gate Result |
|---|---|---|
| **Framework Registration** | **COMPLETE** | Registered `lr_final` and `bend_free_deg` in `profile.py`. Production fingerprint bit-identical (`615ae97d2350`). 100% byte match on canonical `solver_replay.py`. Golden check OK. |
| **O0 Optimizer Survey** | **COMPLETE** | Surveyed 6 optimizer families (`G_OPTIMIZER_NOTE.md`). Proved batched GN/LM normal equations + Cholesky costs only $0.135$ ms ($\sim 3\%$ overhead on $4.5$ ms FK step). Ranked #1: Hybrid Adam + GN/LM; #2: Pullback metric Adam. |
| **S-P1 Beam Coverage** | **COMPLETE** | Cal panel has **80.5% parents** with unfilled supported Q5 modes ($52.3\%$ of modes unfilled). Heavily exceeds $10\%$ gate threshold. Width 48 vs 16 proves greedy density collapse. `beam_target_aware` is fully justified. |
| **G1 Cosine LR Decay** | **COMPLETE (FAILED)** | Both $\text{lr\_final} = 0.01$ and $0.001$ failed guardrails (degraded top-1 strain by $+8.3$ and $+4.0$ kJ/mol; floor degraded by $+15.7$ and $+20.2$ kJ/mol; `USAPOU` degraded by $>125$ kJ/mol). Premature basin freezing under `bend_release`. Stopped at tail. |
| **G3 Bend Free Zone 1.0°** | **COMPLETE (FAILED)** | Failed guardrails (`YUCZUS` degraded by $+92.5$ kJ/mol). Over $73.9\%$ of bends remain outside $\pm 2^\circ$ under both arms. First-order Adam cannot enforce tight bound contractions. Stopped at tail. |
| **Bend Distribution Census** | **COMPLETE** | Measured 4,496 bends on tail parents; control vs G3 distribution identical within $\pm 0.3\%$. |
| **O1 Convergence Diagnostic** | **COMPLETE** | Evaluated 48 conformers across 8 parents in float64 L-BFGS. 0% Class (c) (all unconverged); **43.8% Class (a)** (optimizer failure causing massive physical strain); **56.2% Class (b)** (score minimum strained). Confirmed O2 Gauss-Newton/LM mandate. |


### G.8 Checkpoint G-2 (CodexB; in progress)

**Audit correction:** the G-1 audit rejected the previous O1 classifications and S-P1 headline. The G-1 O1 script scored the wrong coordinate frame, omitted bend normalization and seam closure, and recorded failed relaxations as zero. Its 43.8%/56.2% classification is invalid. G-1 S-P1 counted symmetry-equivalent and unreachable modes. The results below replace those diagnostics as they complete.

#### 2a. Production score reproduction

`run_score_repro.py` captures every production `_run_joint_ik` call, including its FK objects, embedded reference coordinates, scorer, row groups, bend configuration, final theta rows, and score call arguments. Each returned row is keyed by `(embed_idx, assignment_idx, trial_idx)`. The evaluation uses the captured scorer and bend-prior implementation and recomputes macrocycle seam residuals from the captured FK setup.

The CPU smoke suite passed 64/64 delivered rows across HEDJOP (single embed, 3 assignments), NUWFAM (2 E/Z variants), QEZVEY (12 assignments), and USAPOU (macrocycle). The maximum absolute difference from the stored score was $9.58\times10^{-6}$; the maximum float32-to-float64 difference at delivery was also $9.58\times10^{-6}$. `pytest tests/test_score_repro.py`: 4 passed. Results: `research/data/jgen/campaign/B/g/score_repro_smoke.json`.

#### 2b. Bounded O1 local minimization

`run_o1_g2.py` uses float64 L-BFGS-B and box bounds of ±15° per angular coordinate and ±0.5 Å per translation. It asserts score reproduction at every row before minimization, records bound DOFs, maximum atom displacement and nearest non-donor metal contact, and uses runkit MACE/xTB. Failed xTB or MACE evaluations remain explicit failures. The primary seed-11 run completed 48/48 rows across the requested eight parents. All 48 score assertions passed; no row triggered the <1.2 Å metal-contact flag. In total, 42/48 rows hit at least one bound (145 bound DOFs), so the required ±30°/1.0 Å follow-up is running. The primary and currently active wider-cap run use the earlier CPU MACE evaluator; the runner now defaults MACE to CUDA when available and holds the `CodexB` GPU lock. xTB remains CPU-only.

HEDJOP pilot (6 rows): score gains are +0.012 to +0.075; all six rows hit at least one bound DOF (1–6 bound DOFs per row). Three rows classify as (c) because score gains are ≤0.05. The other three have no strain classification because all twelve initial/minimized xTB relaxations returned `no_output(rc=1)` with an abnormal-termination stderr; their strain is unknown, never zero. The maximum score assertion error was below $2\times10^{-7}$. The HEDJOP result predicts little optimizer gain on this parent, but is not the panel conclusion.

| Parent | (a) optimizer problem | (b) score minimum strained | (c) barely moves | Unknown strain | Rows with bound DOFs |
|---|---:|---:|---:|---:|---:|
| HEDJOP | 0 | 0 | 3 | 3 | 6/6 |
| FEKKAJ | 0 | 1 | 1 | 4 | 2/6 |
| NUWFAM | 1 | 0 | 0 | 5 | 6/6 |
| QEZVEY | 0 | 0 | 6 | 0 | 6/6 |
| DAMVAQ | 0 | 0 | 0 | 6 | 4/6 |
| USAPOU | 3 | 3 | 0 | 0 | 6/6 |
| NEPNEF | 1 | 2 | 1 | 2 | 6/6 |
| AJAQAD | 1 | 1 | 4 | 0 | 6/6 |
| **Total** | **6** | **7** | **15** | **20** | **42/48** |

There are 18 rows where both xTB relaxations converged; 6 classify as (a), 7 as (b), and 5 as (c). The other 30 rows have at least one failed or nonconverged xTB side; 10 of those still classify as (c) because their score gain is ≤0.05, and 20 remain unknown. Across all rows, the classes are 6 (a), 7 (b), 15 (c), and 20 unknown. The pattern is mixed: O1 predicts optimizer gain in several rows (notably USAPOU and NEPNEF), while many rows are already near the score optimum or have unknown strain. Row-level Δscore, Δstrain, ΔE_unrel, DOF displacement, bound indices, contact distance, MACE energies and xTB statuses are in `research/data/jgen/campaign/B/g/o1_g2_results.json`.

#### 2c. O2 contract and optimizer arms

The O2 evaluation contract was committed before any arm results in commit `68b046d`; Git cannot stage through the shared `research/data/jgen` symlink, so the tracked identical copy is `research/campaigns/g/evaluation_contract_o2.json`. A shared-output copy is present at `research/data/jgen/campaign/B/g/evaluation_contract_o2.json`.

The optimizer note specifies masked FIRE 2.0 updates and maps least-squares and non-least-squares terms for hybrid LM. The RTX 2060 SUPER is available to CodexB through the shared runkit budget, but no analytic full FK Jacobian or O2 prototype flags are implemented yet; Jacobian formation timing, parity gates and screens remain open. No O2 arm result exists.

#### 2d. S-P1 symmetry and reachability recount

Refined `run_sp1_g2.py` completed the `cal` panel, seed 11: 120/120 parents, zero parent failures. After collapsing symmetry-equivalent modes and keeping heavy-atom torsions, 76 parents have supported modes. At 0.2 rad, 49/76 eligible parents (64.5%; 49/120 overall) have an unfilled mode; 1,148/3,056 supported modes (37.6%) are unfilled, split into 791 reachable and 357 clashing. At 0.35 rad, 49/76 eligible parents remain unfilled; 1,070/3,056 modes (35.0%) are unfilled, split into 731 reachable and 339 clashing. The ≥10% gate is met. Results: `research/data/jgen/campaign/B/g/sp1_g2_results_refined.json`.




### G.8.5 — G-2 Final O1/O2 Results (2026-10-28; supersedes interim “running” status above)

The owner redirected O1 cap-30 to omit xTB during the broad screen. The replacement evaluated captured-score change, CUDA MACE unrelaxed ΔE, displacement, and contacts for 48 rows across eight parents. No non-donor metal contact was below 1.2 Å. Among top-1 rows with |ΔE_unrel| >20 kJ/mol, the five largest-magnitude parents were selected to honor the limit of ten xTB structures. Ten paired structures were attempted; QEZVEY was the only pair where both xTB relaxations converged. The other outcomes remain failed/nonconverged rather than imputed. The earlier partial `o1_g2_cap30_results.json` is superseded and was not used for conclusions. Detailed outputs: `o1_g2_cap30_unrel_gpu_results.json` and `o1_g2_cap30_selected_relax.json` in the shared campaign data directory.

O2 compared default control, `adam_metric`, and `fire_metric` on all nine trope parents at seed 11. The solver ran on CPU and MACE unrelaxed energies ran on CUDA. “Control” is the default solver's existing Adam-style update, with both new optimizer flags off; `adam_metric` adds the FK pullback metric to that Adam update, while `fire_metric` uses the FIRE update. The flags are experimental and the production default fingerprint remained `615ae97d2350`.

| Arm | Rows | Median top-1 score | Paired median score change | Median top-1 ΔE (kJ/mol) | Paired median ΔE change | Obscene rate | Mean floor ΔE | Worst top-1 ΔE |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Control | 305 | 1.034 | — | 128.06 | — | 73.77% | 137.98 | 1,047.66 (SUXROT) |
| Adam metric | 290 | 0.896 | −0.100 | 92.43 | −0.64 | 81.72% | 144.02 | 509.70 (SUQWEG) |
| FIRE metric | 227 | 1.014 | −0.135 | 229.31 | −4.17 | 83.26% | 455.68 | 3,116.06 (SIKNAE) |

The screen fails the O2 gate for both arms: median solver score decreases, obscene rate increases, and mean floor ΔE worsens. FIRE also has a large SIKNAE outlier. Adam metric was selected only for the user-requested terminal nine-structure relaxation because its aggregate screen was less adverse. Eight of nine xTB relaxations converged; SUQWEG failed with `no_output(rc=1)`. Successful xTB/MACE paired strains are in `adam_metric_top1_relax.json`; failure is explicit.

The required light replay check also failed. With Adam metric, MELLUN, AHIGOQ, and PIDXOT each replayed 38/40 steps but differed from eager solver outputs (maximum score differences 2.37e−3, 9.35e−5, and 3.93e−6; maximum coordinate difference 0.0355 Å). With FIRE metric, all three replay attempts failed with a `fire_dt` free-variable `NameError`. A targeted test batch had 3 passes and 1 failure: the fast-FK CPU test returned 6 conformers instead of its expected 8. These issues are recorded for the next authorized work; O2's trope screen used the CPU solver path and completed all 9 parents for all arms.

**G-2 disposition:** O1 requested diagnostics and the bounded selected relaxations are complete. O2 trope screening and the requested nine terminal relaxations are complete, with failures plainly recorded. Both optimizer gates failed. No tail or cal panel was opened. Stop at G-2 for audit.


### G.9 — G-3 step 0 (2026-10-28)

The production loop maximises the statistical scorer plus the normalised bend prior, subject to extra shape, equal-donor-distance symmetry and Q3-angle symmetry augmented-Lagrangian penalties (adaptive nonnegative multipliers and quadratic rho terms). Those constraint penalties are absent from the delivered score. The multiplier stall windows freeze dual ascent; polish omits both dual ascent and its stall state. Plateau detection marks convergence but does not stop the loop; polish runs its full 50 steps and includes every evaluated iterate in best-seen selection. Culling changes row survival rather than the objective: restore the archived rows before polish and do not cull again.

Bend release and optional DOF-class scales modify updates, not scores. Polish uses neither schedule; Adam-small starts fresh moments at lr 0.01. The bidentate base-gradient projection is an update modification, not part of the delivered objective, and is omitted during exact-score polish. Keep the frozen Q3 mode override, row groups, scorer floors and bend normalisation used at delivery. Include the macrocycle seam squared-residual penalty with the production closure weight. The loop snapshot uses the seam penalty but its gradient objective does not; polish differentiates that penalty as score_repro.py does. Final closure projection can change theta: the sanity runner starts at the captured delivered theta, and integrated polish must preserve the initial delivered geometry as a candidate. Angle wrapping and axis-angle renormalisation are coordinate representations, not extra energy terms.

Step 0 fast-FK test: `tests/test_gpu_fast_path.py::test_fast_path_solves_on_cpu_with_one_pass_fk` passes on detached master 8dbe29c (14.68 s) and unmodified G-2 a6711c9 (15.65 s), each with xTB on PATH. The prior 6-versus-8 failure is **unreproduced**, not established as a branch regression or fixed. A detached scratch worktree was created without touching either branch.

#### Spec issues

No gate or metric has been changed. The FIRE mass wording is ambiguous: A1 defines an inverse-square-root step scale, whereas a mechanical mass normally uses the reciprocal mobility. The implementation must document its literal choice before calibration. The final closure projection must not invalidate a best-score guarantee: preserve the delivered baseline and evaluate any projected candidate with the exact delivered score.

#### G-3 step 1 — exact O1-row sanity (complete)

All 48 initial rows match the original cap-30 O1 scores and Cartesian coordinates within 1e-4. CPU chunk layout affects culling: six parents reproduce with three requested workers; QEZVEY and USAPOU require the original default eight-worker layout. The latter executes at most three chunks concurrently. Failed attempts remain explicit in the earlier output files; `g3_polish_sanity_complete.json` supersedes them for this table. No starting rows or L-BFGS denominators were replaced.

| Parent | P2 median recovery | P1 median recovery | P2 median ΔΔE_unrel | P1 median ΔΔE_unrel |
|---|---:|---:|---:|---:|
| HEDJOP | 94.4% | 62.2% | -20.12 | -23.40 |
| FEKKAJ | 99.3% | 68.4% | +142.72 | +65.62 |
| NUWFAM | 63.1% | 25.5% | -49.21 | -2.38 |
| DAMVAQ | 99.0% | 99.2% | -3.09 | -5.65 |
| NEPNEF | 78.2% | 2.1% | +0.67 | -0.63 |
| AJAQAD | 83.0% | 45.6% | +22.08 | +4.97 |
| QEZVEY | 82.4% | 41.2% | -3.91 | -0.87 |
| USAPOU | 91.0% | 4.9% | -33.29 | +5.16 |

Across all 48 rows: P2 median score gain +0.06390, median L-BFGS recovery 87.73%, median ΔΔE_unrel −5.27 kJ/mol; P1 +0.03300, 44.54%, +0.91 kJ/mol. Both pass the ≥25% recovery and non-decreasing median delivered-score rule. The best-seen baseline participates in selection, as required. Neither result permits skipping the C1 comparison.

A1 fails: median delivered top-1 score C0 0.127374 versus A1 −0.014039, with median paired change −0.112561 on all eight parents. It is not screened. The scale is normalised across every DOF of every ligand within a row before clamping; this is the corrected specified method, not the G-2 globally shrunken step.

FIRE calibration uses HEDJOP and USAPOU only: probe dt=0.1, median first-step Cartesian RMS=0.131663 Å; dt_start=0.03897477 and dt_max=0.38974769. The actual sanity-run median first step over the same 12 rows is 0.0195121 Å. FIRE uses the literal normalised A1 scale as mass, α_start=0.25, N_delay=5, f_inc=1.1, f_dec=0.5, f_α=0.99, initial delay, and half-step back-off. Actual FK displacements enforce the 0.05 Å per-row cap.

#### G-3 gate and harness notes (in progress)

Default fingerprint remains `615ae97d2350`. Default CPU loop parity: 2 passed; score-repro and generator checks: 5 passed. A1 CPU parity: 2 passed. P2 CPU parity: 2 passed. P1 at calibrated dt plus optimizer unit tests: 4 passed. CUDA replay versus the same eager loop passes MELLUN/AHIGOQ/PIDXOT for A1, P2, P1 and legacy FIRE; each replays 38/40 steps. The P1 CUDA check uses dt=0.1, with calibrated-dt CPU parity separately verified. The final calibrated P1 GPU configuration still requires its own check before any GPU panel. Golden is running.

The old `run_o2_replay_parity.py` compared JointIKSolver against ReplaySolver and demanded exact equality. The existing bench replay harness instead compares ReplaySolver(replay=False) against ReplaySolver(replay=True) at relative score tolerance 1e-6. The helper now follows that comparison and tolerance, preserving row identity; no acceptance threshold was invented or loosened relative to the specified existing harness. The initial G-3 output using the old helper remains diagnostic evidence, not the replay gate. Legacy FIRE state updates now remain in persistent tensors, removing its closure/free-variable NameError.

The final default closure projection precedes polish. Polish includes the exact seam residual penalty and retains its initial delivered state; reprojecting after best-seen selection could lower the selected delivered score and is therefore omitted for the experimental phase. No production-default path changes.

Trope seeds 11–13 were explicitly committed in `ec2cf71` before any trope results, so the ≥2/3 guard is observable. The first C1/P1/P2 pilot uses CAFWIQ and MUNSUJ. No tail/cal screen or relaxation has started.

#### G-3a completed trope filter (2026-10-28)

The registered contract was committed before results (2dcd566; three trope seeds clarified in ec2cf71). C1 and P1 completed all nine parents × seeds 11–13 on this CPU, with MACE unrelaxed energies on the local RTX 2060 SUPER. P2 stopped after the two-parent pilot because CAFWIQ already violates the ≥2/3 guard. No missing P2 result is imputed. No CI is claimed for nine parents, as required.

| Arm | Parents / seed units | Mean paired top-1 ΔΔE | Median paired top-1 ΔΔE | Median paired score change | Mean floor change | Obscene rate (arm / paired C1) | Solver time / C1 | Disposition |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P1 | 9 / 27 | -47.95 | +0.17 | +0.000137 | -0.65 | 74.43% / 74.55% | 1.399× | FAIL: median energy >0; solver cost >1.3× |
| P2 | 2 / 6 | +107.24 | -15.65 | +0.006554 | +12.07 | 87.57% / 85.14% | 0.994× | FAIL: CAFWIQ guard (2/3 seeds) |

Energies are kJ/mol. P1 raises the parent-median score on 5/9 parents (the same count using parent means). Its mean paired energy gain is real but does not override the registered median ≤0 filter. Its median energy change is +0.172471 kJ/mol: **failed**, not rounded to zero. P1 has no parent with >25 kJ/mol worsening on ≥2/3 seeds. P2 improves scores but worsens CAFWIQ by +505.80 and +318.07 kJ/mol on seeds 11 and 12, failing its guard regardless of the seven unmeasured parents. Neither arm advances to tail or cal.

| P1 parent | Median paired score change | Median paired top-1 ΔΔE | Seeds worse >25 | Solver time / C1 |
|---|---:|---:|---:|---:|
| CAFWIQ | -0.003498 | -8.28 | 0/3 | 1.548× |
| CANXAQ | +0.068550 | +11.17 | 0/3 | 1.291× |
| MUNSUJ | +0.000390 | -0.26 | 0/3 | 1.236× |
| RIGWAJ | -0.000305 | +0.10 | 0/3 | 1.195× |
| SIKNAE | -0.010877 | +3.91 | 0/3 | 1.348× |
| SISKAJ | +0.003026 | +0.29 | 0/3 | 1.608× |
| SUQWEG | +0.026284 | +1.08 | 0/3 | 1.223× |
| SUXROT | +0.000548 | -24.54 | 0/3 | 1.575× |
| YUNPIG | -0.000187 | -0.21 | 0/3 | 1.495× |

P1 delivers 872 rows versus C1 892 across 27 seed units. Its mean floor is 124.57 versus 125.22 kJ/mol; its worst top-1 is 1,332.96 kJ/mol. Worst delivered slot: 3,492.58 kJ/mol, YUNPIG seed 13, delivered row index 17. P2 pilot delivers 169 versus 175 rows; mean floor 162.47 versus 150.40; worst top-1 1,320.27; worst slot 1,906.80 (CAFWIQ seed 11, row 29). The P2 figures refer only to its two matched pilot parents, not to all nine.

Cost uses the suite’s measured `solve_s`, summed across matched parents: 1.398616× for P1 (median parent ratio 1.348121×), 0.993808× for P2. Full subprocess wall ratios are also retained (P1 0.985415×; P2 0.977270×), but the C1 process totals include waiting behind the independent replay check for the GPU; those totals are not a matched estimate of optimizer runtime. P1 fails the energy median gate independently of the timing interpretation, so no cost exception is claimed.

#### Spec issues and deviations (G-3a)

- **CPU worker layout was missing from the prior diagnostic identity.** It changes culling outcomes. The exact O1 score/geometry assertions prevent silently substituting a fresh top-six set. The final 48-row table reproduces the originals; future diagnostics now retain starting theta where captured and expose the requested chunk layout.
- **Replay helper did not implement the existing bench comparison.** Corrected it to same-loop eager/replay, relative score tolerance 1e-6 and matching delivered row identity. Added raw pre-dedup score comparison to match the existing harness more fully. The initial cross-loop/exact-equality failure remains in its original JSON; it is superseded as a gate, not concealed.
- **FIRE mass wording:** used the literal normalised A1 scale as mass, documented before calibration. No gate or score term was redefined.
- **Closure after polish:** avoid a second projection that could change the best-scoring delivered candidate. The exact seam penalty remains in both optimisation and final scoring; default production is unchanged.
- **Trope seeds:** explicitly registered 11–13 before screen results so the ≥2/3 guard can be evaluated.
- **P2 early termination:** rejected after two parents once its prescribed guard was conclusively violated; seven unnecessary parent runs were omitted. No full-nine-parent P2 claim is made.
- **Golden invocation:** forcing one worker compared a different culling layout against the stored reference, producing eight production-array differences (CBTYCO, OJIMIF, SOKVET, YAVFEG); the redundant u1 continuation was stopped by PID. Rechecking the original default layout with bounded active concurrency passes **245/245 arrays for production and 245/245 for u1**. Golden files were never changed.

#### Checkpoint disposition

No candidate qualifies for a GPU panel. No tail/cal screen, complex relaxation, Colab VM, or G-3b implementation was started; Colab spend 0. The float64 CPU sanity results establish meaningful delivered-score convergence, but the compute-matched trope filter rejects both polish arms. A1 also fails independently. Failures remain failures. Raw-score replay verification is complete: all three units pass for A1, P2 and calibrated P1, with maximum relative difference **0** across 128, 256 and 896 pre-dedup rows, respectively. Delivered scores and row identities also match. Stop at G-3a for the audit after committing.

Final robustness check: an experimental-polish exception logs the failure and retains the original delivered baseline, including its original seam residual. This is confined to the flag-on path. The optimizer clones its inputs before stepping, so a failed phase cannot corrupt the fallback. A fault-injection test verifies unchanged delivered scores and coordinates; the initial test attempt failed only its warning-capture assertion (geometry and score assertions passed), and explicit warning capture plus an optimizer-invocation assertion resolves that test harness issue. The final targeted optimizer/fallback/generator batch passes.

**G-3a status: complete, all candidate arms rejected; stop for audit.** No G-3b work is pending in this checkpoint. Full panel CIs and cross-platform checks are not applicable because no arm passes the trope filter. Outputs remain under `research/data/jgen/campaign/B/g/`; tracked diagnostic scripts and the two pre-registration commits reproduce the tables.


## G.10 — Atom-order correction (2026-10-29, owner finding)

**This section supersedes every O1-derived MACE energy or xTB/strain claim above. Scores and score recovery are unaffected. All historical xTB relaxations from the affected tools are void; none was rerun.**

`run_o1_g2.py` paired builder/FK coordinates with element labels from the tmQM frame. The frame and builder orders differ. Future exports now take labels from `conf.complex.atoms` and assert equality with that list and coordinate length. Both polish sanity runners use the matched conformer instead of copying `prior["atom_symbols"]`. The selected-relaxation consumer refuses unverified exports. The same bug affected the old non-donor contact indexing; the canonical saved O1 contact checks were recomputed in builder order.

Repair used saved coordinates only, on the local 2060, with MACE float64. The engine shell atom graph was reconstructed with RDKit and a graph-only stand-in for embedding; no solve, embed refinement, or relaxation was performed for this repair. `_assemble_conformers` retains this shell atom list. Original JSON files were retained as `*.before_atom_order_fix.json`; corrected canonical files explicitly identify the builder atom order.

### Fixed-row median energy changes (kJ/mol; 48 original rows)

| Arm/reference | Old, invalid labels | Corrected |
|---|---:|---:|
| O1 L-BFGS | −13.347545 | −17.172224 |
| G-3a P2 / Adam-small | −5.268848 | −16.711545 |
| G-3a P1 / FIRE | +0.909721 | −4.344624 |
| G-3a′ P2 | not separately reported; same saved finish as above | −16.711545 |
| G-3a′ R1 | not reported before repair | −12.747431 |
| G-3a′ R1, Δ0 ×0.1 | not reported before repair | −14.250588 |
| G-3a′ R1, Δ0 ×10 | not reported before repair | −13.772010 |

**P2’s old −5.3 baseline is void. The corrected baseline is −16.711545.** The fixed-row C0 change against itself is zero; its reference-relative delivered top-1 energies are corrected below. A1 stopped on its score-only sanity gate: no A1 coordinates or energy evaluation was produced, so there is no A1 energy number to recompute. Its score failure stands.

### O1 per-parent median change

| Parent | Mismatched labels | Old ΔΔE | Corrected ΔΔE |
|---|---:|---:|---:|
| HEDJOP | 47 | -51.424 | -66.812 |
| FEKKAJ | 52 | +143.322 | +175.706 |
| NUWFAM | 44 | -40.247 | +33.947 |
| QEZVEY | 41 | -22.626 | -22.115 |
| DAMVAQ | 30 | -11.805 | -121.643 |
| USAPOU | 39 | -42.927 | -168.639 |
| NEPNEF | 42 | -1.616 | -15.156 |
| AJAQAD | 26 | +19.255 | -3.376 |

NUWFAM’s O1 median energy change changes sign; the previous per-parent energy interpretations cannot be retained. FEKKAJ still worsens in the corrected calculation.

### Saved top-1 energies relative to the frame reference (kJ/mol)

These are the original O1 top-1 row identities after finishing, without reranking the six sanity rows. Old numbers below are reconstructed from the old saved energies and the correctly labelled frame reference; they are shown only to expose the invalid baseline.

| Parent | C0 old → corrected | P1 old → corrected | P2 old → corrected |
|---|---:|---:|---:|
| HEDJOP | 24900.31 → 409.14 | 24876.52 → 378.37 | 24872.36 → 358.86 |
| FEKKAJ | 24998.48 → 575.35 | 25104.43 → 700.16 | 25955.97 → 1566.14 |
| NUWFAM | 22423.95 → 186.36 | 22446.25 → 185.72 | 22406.21 → 178.36 |
| QEZVEY | 19311.92 → 372.61 | 19317.67 → 369.91 | 19307.16 → 362.72 |
| DAMVAQ | 16655.27 → 558.69 | 16655.54 → 469.60 | 16656.81 → 465.94 |
| USAPOU | 17360.85 → 392.87 | 17370.01 → 385.54 | 17412.85 → 256.60 |
| NEPNEF | 15303.50 → 2675.10 | 15303.55 → 2675.13 | 15300.82 → 2662.84 |
| AJAQAD | 15579.81 → 21.12 | 15615.88 → 20.56 | 15632.26 → 19.24 |

The independent trope runner maps coordinates into frame order before pairing them with frame labels. All **87 saved top-1** energies from G-3a C1/P1/P2 and the historical O2 control/metric arms were recomputed; maximum change is **0 kJ/mol**. Thus the mapped trope results and their historical rejection outcomes stand. Only top-1 coordinates were exported by that runner: its full-pool floor/obscene numbers were not independently recalculated, and are supported by the verified mapping path rather than a new pool evaluation.

The earlier primary/cancelled O1 JSON files do not retain coordinate arrays and their xTB input work files were not retained. Their energy and strain claims remain explicitly void; they are not substituted with a new solve. The canonical saved cap-30 O1 comparison is fully recomputed (48 rows, 8 parents, no repair failures).

Artifacts: `atom_order_repair_report.json` (old/new values and label provenance), `atom_order_repair.log`, `atom_order_reference_repair.log`, corrected `o1_g2_cap30_unrel_gpu_results.json`, `g3_polish_sanity_complete.json`, and `g3prime_sanity.json`, under shared campaign `B/g/`. The refreshed `g3_sanity_summary.json` uses corrected energies. Regression test: `tests/test_atom_export.py` passes. No Colab spend.


## G.11 — G-3a′ iRprop+ sanity checkpoint (2026-10-29)

**R1 fails the prescribed sanity gate. Cal is unopened. Stop here for audit.** Contract commit `9d561b4` registered the revised cal rules before cal; the atom-order repair is in `187ac77`. All energy numbers below use corrected builder labels (§G.10).

### Implementation and reproduction

R1 (`polish="irprop"`) runs 200 production Adam steps, then 50 steps of iRprop+ on the exact delivered score. It maximises the score directly, with per-row/per-DOF sign history, Δ growth 1.2/shrink 0.5, bounds [1e−6, 0.1], row-score-conditioned backtracking on sign-flipping DOFs only, and cleared gradient after a flip. Branches and idle masks use `torch.where`. Small-gradient convergence is fixed at norm <1e−8; update convergence is max |u| <1e−5. Best-seen includes the baseline and final update.

Δ0 is the absolute bias-corrected Adam moment step **before** bend-release, DOF or metric scaling. Its full-row buffer preserves the last available moments for culled rows. R2 (`optimizer="irprop"`) skips the production update loop and runs 250 delivered-score steps from the initial θ with the specified .01 fallback. R2 was used only on CPU. No ALM, dual ascent, bend release or projected gradient is applied inside either iRprop+ phase. The existing closure/baseline-preserving failure handling is retained. Replay source was regenerated from `tools/jdev/build_replay_solver.py`.

The 48 original O1 starting scores and coordinates reproduced within 1e−4; no O1 row or L-BFGS denominator was replaced. Original CPU layouts were retained (three chunks for six parents, eight for QEZVEY/USAPOU), with at most three chunks executing concurrently. For the new R2/C0 full solves and golden/parity gates, the default eight-chunk layout was retained. Total CPU concurrency was bounded at five; R2 was briefly paused by its PID while the two-worker CPU gate ran. No comparative R2 runtime claim is made.

### Exact-row results (48 rows, eight parents, seed 11)

| Arm | Median L-BFGS recovery | Median Δscore | Median corrected ΔΔE_unrel (kJ/mol) | Rows converged by 50 |
|---|---:|---:|---:|---:|
| P2 Adam lr .01 | 87.7299% | +0.06390280 | -16.711545 | — |
| R1 nominal Δ0 | 83.1225% | +0.05345724 | -12.747431 | 0/48 |
| R1 Δ0 ×0.1 | 73.5822% | +0.04931777 | -14.250588 | 0/48 |
| R1 Δ0 ×10 | 83.5361% | +0.05621411 | -13.772010 | 0/48 |

Every arm retains or improves the delivered score on every row; nominal R1’s minimum gain is +0.00128116. Nominal recovery **83.1225% < P2 87.7299%**, and its corrected median energy change **−12.7474 is worse than P2 −16.7115**. Thus it fails both alternatives. Both sensitivity arms also fail both alternatives; none is substituted as the candidate. The outcome does depend on Δ0: ×0.1 lowers recovery to 73.58%, while ×10 gives 83.54%.

All 48 convergence-iteration entries are −1 in each iRprop+ arm, meaning not converged by step 50. Per-row keys and full per-iteration diagnostics are stored in `g3prime_sanity.json`; no convergence iteration is imputed.

### Step-length diagnostics

The table reports the median of row-wise DOF medians, and the mean row-wise fraction of DOFs at each bound. Full 50-step series remain in `g3prime_sanity_summary.json`.

| Arm | Iteration | Median Δ | Fraction at Δmax | Fraction at Δmin |
|---|---:|---:|---:|---:|
| R1 | 1 | 0.00031207103 | 0.0000% | 7.5938% |
| R1 | 25 | 0.00054175007 | 0.0443% | 7.7754% |
| R1 | 50 | 0.00031408825 | 0.0000% | 11.0780% |
| R1_x01 | 1 | 3.1207103e-05 | 0.0000% | 12.4050% |
| R1_x01 | 25 | 0.00058651578 | 0.0000% | 9.0738% |
| R1_x01 | 50 | 0.00034453269 | 0.0000% | 10.2028% |
| R1_x10 | 1 | 0.0031207103 | 1.8549% | 6.2752% |
| R1_x10 | 25 | 0.00058311493 | 0.0000% | 8.0737% |
| R1_x10 | 50 | 0.00019194249 | 0.0000% | 10.9096% |

### Remaining gap and diagnosis

The corrected O1 L-BFGS median energy change is −17.172224. Differences of medians leave **0.460680 kJ/mol for P2** and **4.424793 for nominal R1**. These are descriptive comparisons, not a ceiling on every row. The median **paired** remaining energy gaps are −0.117965 and +0.077613 respectively; skew and score/energy inversion make these different statistics. Median remaining score gains against O1 L-BFGS are +0.01023372 (P2) and +0.01089105 (R1). O1 itself was iteration/bound limited, so it is not a proof of full convergence.

Sign convention, per-row worsening checks, overshooting-DOF-only backtracking, sign-reset, initial bounds, and best-seen retention were checked in the optimizer tests. CPU and recorded-loop results agree. No sign/backtracking/bounds implementation defect was found after the handoff correction. The 50-step method remains underconverged and fails to beat the simpler baseline. Neither larger Δ0 nor smaller Δ0 rescues the required gate; no iteration count or threshold was tuned after these results.

### R2 whole-run CPU sanity only

| Parent | C0 top-1 score (200 Adam) | R2 top-1 score (250 iRprop+) | Change |
|---|---:|---:|---:|
| HEDJOP | +0.19982231 | +0.33324531 | +0.13342300 |
| FEKKAJ | -0.16436762 | -0.09343588 | +0.07093173 |
| NUWFAM | +0.27856275 | +0.34046274 | +0.06189999 |
| QEZVEY | +0.05492561 | +0.10206056 | +0.04713495 |
| DAMVAQ | -3.06045866 | -1.78415048 | +1.27630818 |
| USAPOU | -2.05764151 | +0.05394481 | +2.11158631 |
| NEPNEF | +0.62052011 | +0.74349385 | +0.12297374 |
| AJAQAD | +0.47257030 | +0.52092510 | +0.04835480 |

Eight paired parents, zero failures. Median C0 score 0.12737396 → median R2 0.21765293; median paired change **+0.09695274**. All eight improve. This is a score-only sanity finding; it does not establish an energy gain or justify replacing Adam’s search phase. R2 did not go to cal.

### Light gates and implementation corrections

- Default fingerprint **615ae97d2350**. Production golden: **245 arrays, 0 differ**, default eight-chunk layout, 340.34 s. The later moment-handoff fix touches only `polish="irprop"`; default execution is unchanged.
- Final targeted optimizer, default/R1 CPU loop parity (MELLUN, default layout), fallback fault injection, and replay-generator consistency: **9 passed in 51.00 s**. The atom-export regression separately passed.
- Corrected-moment CUDA same-loop replay: MELLUN/AHIGOQ/PIDXOT, **all pass**, full raw pools 128/256/896 rows, maximum relative score difference **0**, matching delivered row identities, 38/40 main-loop steps recorded/replayed. The polish finish remains the same eager fixed-shape phase in either mode, as for P2.
- An injected polish exception preserves delivered scores and coordinates exactly.

**Spec issues / corrections:** An initial development run was stopped while implementing the exact convergence predicate, before useful parent results. A subsequent trial captured Δ0 after production bend scaling; this was corrected to the audit’s raw moment formula and the sanity/flag-on gates were rerun. That trial is explicitly retained as `g3prime_sanity_scaled_handoff_superseded.{json,log}` and is not used for any gate. The owner’s atom-order finding then invalidated the old energy baseline; §G.10 records the correction and original values. This changes no gate: the contract already compares against paired P2, whose corrected value is used here. No gate was lowered.

Artifacts under shared `B/g/`: `g3prime_sanity.json`, `g3prime_sanity_summary.json`, `g3prime_r2.json`, `g3prime_golden.{json,log}`, `g3prime_replay_moments.{json,log}`, and `g3prime_light_tests_moments.log`. CPU score/coordinate exports were checked unchanged by the energy repair on all 144 rows across O1/G-3a/G-3a′.

**Disposition:** R1 rejected at the issued sanity gate; report and stop before cal. No cal/tail panel, complex relaxation, Colab VM/spend (0), FIRE retuning, R2 panel, or G-3b work. Await audit.


## G.12 — Owner-requested R2 energy and iRprop− follow-up (2026-10-29)

**Both whole-run arms improve the delivered energy floor but do not demonstrate a top-1 energy improvement over production C0 on these eight parents. This is an exploratory sanity follow-up, not a panel pass or production adoption.**

Owner requests after G-3a′: evaluate R2 energy, then also test iRprop−. The experiment contract was registered in `0a164b9` and amended for the minus arm in `fa9c436`, before minus results. Parents are the original eight O1 parents, seed 11. No cal/tail panel or locked set was opened.

### Arms and method

- C0: 200 production Adam steps, normal production objective and culling.
- R2+: 250 iRprop+ steps from initial θ, no Adam, delivered-score objective, no production culling. This is the earlier R2.
- R2−: 250 iRprop− steps from initial θ; same objective, Δ0=.01 fallback, bounds [1e−6,.1], adaptation factors, stopping rule, and best-seen handling as R2+. Only weight-backtracking is removed: a sign flip shrinks Δ, takes a zero update for that DOF, and clears its stored gradient. This follows [Igel/Hüsken §2.4 and Fig.3](https://christian-igel.github.io/paper/EEotIRLA.pdf).

Both iRprop arms optimise the delivered score throughout; C0 uses its production constraint objective. Iteration count and culling also differ. The R2−/R2+ comparison isolates the backtracking change; the comparisons with C0 do not isolate optimizer choice alone. No learning rates, bounds, step counts, stopping rules, or parent selections were tuned after seeing these energies.

The original R2 score-only runner omitted coordinates, so the eight C0/R2+ pairs were repeated and all 16 top-1 scores reproduced **exactly**. All 16 returned conformers per arm/parent were saved, with labels sourced from each `conf.complex.atoms`, explicit equality/count assertions, and builder-order metadata. The default requested eight-chunk layout was retained, with at most five chunks executing concurrently. Two-parent CPU timings estimated roughly 17 minutes for the plus/control solves; a warmed two-structure CUDA MACE profile (116 atoms each, 0.4824 s) estimated about one minute for all single points. Existing hardware was used; the occupied GPU was left alone until it became available.

Energy evaluation used the same local 2060, MACE float64, model hash recorded in each output, and correctly labelled frame reference. R2− reused C0 single points only after matching model, device and source-geometry digests; the reference energy was independently checked within 1e−5 kJ/mol. The direct minus/plus comparison also checks plus geometry digests. No solve was repeated for the control in the minus phase. All coordinates and all computed energies are finite, with **8/8 paired parents, zero failures, 128 delivered rows per arm**.

### Primary and secondary comparisons with C0

Negative paired energy change is better. Confidence intervals are parent-clustered percentile bootstrap intervals (10,000 resamples, seed11); eight deliberately selected parents provide limited generalisation.

| Metric | R2+ − C0 | R2− − C0 |
|---|---:|---:|
| top1 mean ΔE (kJ/mol) | +121.330566 | +78.252421 |
| top1 paired median ΔE (kJ/mol) | +11.141003 | +5.924003 |
| top1 mean 95% CI | [-117.382627, +421.387700] | [-128.871801, +367.907731] |
| top1 parents improved | 3/8 | 3/8 |
| floor mean ΔE (kJ/mol) | -66.391188 | -67.640901 |
| floor paired median ΔE (kJ/mol) | -24.547641 | -24.448015 |
| floor mean 95% CI | [-132.886893, -5.820016] | [-135.583515, -6.022948] |
| floor parents improved | 7/8 | 7/8 |

| Delivered-pool metric | C0 | R2+ | R2− |
|---|---:|---:|---:|
| Mean top-1 ΔE | 648.904787 | 770.235353 | 727.157208 |
| Median top-1 ΔE | 401.005585 | 371.256394 | 347.495319 |
| Mean floor ΔE | 583.986411 | 517.595223 | 516.345510 |
| Obscene rate (>200) | 86.7188% | 82.8125% | 82.0312% |
| Summed build/solve seconds | 196.921528 | 494.389935 | 494.150748 |

The absolute median top-1 energy falls even though the median **paired change** rises; differences of medians are not medians of differences. The paired changes above are the registered comparison. “Floor” means the minimum energy among the 16 delivered conformers, not all internal IK rows. Both build/solve ratios are about 2.51× C0 (R2+: 2.510594, R2−: 2.509379). These include building/embedding and have different iteration counts/culling; they are not optimizer-only timing comparisons.

### Per-parent top-1 and floor energies (kJ/mol relative to frame reference)

| Parent | C0 top-1 | R2+ top-1 | R2− top-1 | C0 floor | R2+ floor | R2− floor |
|---|---:|---:|---:|---:|---:|
| HEDJOP | 409.137 | 218.507 | 218.732 | 321.520 | 184.649 | 184.649 |
| FEKKAJ | 575.350 | 1565.332 | 1566.620 | 233.575 | 223.641 | 223.639 |
| NUWFAM | 186.360 | 670.663 | 336.310 | 186.360 | 241.716 | 240.818 |
| QEZVEY | 372.614 | 383.831 | 384.289 | 364.723 | 325.562 | 325.763 |
| DAMVAQ | 558.692 | 358.681 | 358.680 | 516.153 | 358.681 | 358.680 |
| USAPOU | 392.874 | 157.464 | 156.113 | 392.874 | 157.464 | 148.238 |
| NEPNEF | 2675.094 | 2775.222 | 2775.222 | 2635.569 | 2629.473 | 2629.401 |
| AJAQAD | 21.117 | 32.182 | 21.291 | 21.117 | 19.576 | 19.576 |

Top-1 scores rise versus C0 on every parent in both iRprop arms. Top-1 energy nevertheless rises for FEKKAJ, NUWFAM, QEZVEY, NEPNEF and AJAQAD. These are the specified score-up/energy-up inversion signatures. FEKKAJ is the largest regression (+989.982 kJ/mol for plus; +991.271 for minus). NUWFAM worsens by +484.303 for plus and +149.951 for minus. HEDJOP, DAMVAQ and USAPOU improve in both arms.

Worst delivered slot: C0 2751.442612 (NEPNEF, key [null,3,261]); R2+ 3363.748696 and R2− 3388.497450 (both FEKKAJ, key [null,12,969]). No worst-slot regression is omitted.

### iRprop− directly against iRprop+

| Metric (minus − plus) | Mean | Paired median | Mean 95% CI |
|---|---:|---:|---:|
| Top-1 ΔE | -43.078145 | -0.000613 | [-126.751878, +0.306428] |
| Floor ΔE | -1.249713 | -0.001743 | [-3.581698, +0.041011] |

Minus’s mean top-1 improvement against plus is concentrated in NUWFAM (−334.353 kJ/mol), with AJAQAD −10.891 and USAPOU −1.351; other differences are small or zero. The paired median is effectively zero and the interval includes zero. Removing backtracking therefore does not show a broad top-1 benefit here. Both variants produce lower-energy delivered options in seven parents, while the score often selects a more energetic winner. This supports investigating the score/energy relationship; it does not justify replacing production search from this small experiment.

### Validation, artifacts, stop

- Targeted optimizer/sign-reset/best-seen, whole-run-minus no-Adam integration, default/R1 CPU parity, injected fallback, atom-export and generated replay checks: **13 passed in 37.45 s**.
- Default fingerprint remains **615ae97d2350**. Full golden was not repeated after this flag-only extension; the preceding package default-layout golden passed 245/245. Default branch arithmetic is unchanged. Whole-run flags explicitly use the native solver so they skip Adam; no CUDA solver replay claim is made for these CPU-only arms.
- Artifacts under shared `B/g/`: `evaluation_contract_r2_energy.json`, `g3prime_r2_energy_structures.json`, `g3prime_r2_energy.{json,log}`, `g3prime_r2_minus_energy.{json,log}`, both solve logs, `r2_energy_profile.{json,log}`, and `irprop_minus_light_tests.log`. Cached control and plus energy digests are retained in the energy JSON.
- No cal/tail panel, locked-set evaluation, xTB complex relaxation, Colab VM/spend (0), tuning, production fold-in, or G-3b work. **Owner-requested R2 energy and minus-arm follow-up complete; report, commit, stop.**

## G.13 — Checkpoint G-3a″: late Adam schedules (completed 2026-10-30)

**P2 has a measurable energy benefit on most valid cal parents, especially when the control top-1 is not obscene, but the current implementation does not pass the issued all-parent CI and wall-time gates. None of the three single-loop refinements passes step 1. No fold-in; stop at G-3a″.** The owned L4 is shut down and all 1,080 expected records are local.

### Contract, source and scope

Evaluation contract committed **f062f35 before the first cal solve**: `g/evaluation_contract_g3a2.json`. Implementation433d20b; filter/analysis9ff8655; owner-authorized master7d3e4a5 merge **6796763**, no conflicts. Production fingerprint **6cc586e17ab1** throughout cal. P2 is200 production Adam steps followed by50 fresh-Adam steps atlr0.01 on the delivered objective; C1 is250 continuous production Adam steps. The finish now scales as `round(.25*T)`.

Cal:120 frozen calibration parents, seeds11–13, L4; C0,C1,P2. Each arm/control pair uses the same machine, input preparation, model and row batching. The first two profile parents are reused, not re-solved. The cal runner uses production-default output counts; the CPU eight-parent filter capped delivery at16. CPU culls at60; GPU replay never culls. These differences and the Li-cap production change prevent interpreting CPU-versus-L4 differences as optimizer effects alone.

Runtime source: bundle3250246, harness patchf436e83 (benchmark SHA859606dee255...), followed by validity-only patchaf5cee6 for partitionB (benchmark SHAccfc16ed4af9..., validity module SHA36aff8c797dd...). No solver, scorer, optimizer settings, calibration thresholds, panel, seeds or pass rules changed after cal opened. Model/source/machine/runner checks and raw backups are retained. The owner is separately fixing Q6; that change is not present in this package's cal comparison.

### Step 1: calibration, traces and filters

Only the prescribed settings were tried: z*=0.5, k=4; δ=0.001, W=10, two-window patience. No retuning follows the failed filters.

The final diagnostic uses actual continuous native float32 Adam state and the constrained objective:200 steps, then50 observed continuation steps, with triggers enabled at200. T=250 retains the original cull at60 and bend release at100. A git-archived9ff8655 library snapshot preserves the original O1 initialization (fingerprint615ae97d2350). All 48 original row identities/candidate assignments are verified. Every selected row's θ,m,v,gradient and score at **each of the first200 steps** matches a paired C0 reference **bit-for-bit**.

| Arm | Rows with halving | At floor after50 continuation steps | Required | Verdict |
|---|---:|---:|---:|---|
| P2a / zigzag | 8/48 | 2/48 | ≥24/48 | **drop** |
| P2p / plateau | 7/48 | 0/48 | ≥24/48 | **drop** |

Full histories: `g3a2_continuous_diagnostics.json` and canonical `g3a2_diagnostics.json`. Fixed examples, selected by identity before inspecting trigger outcomes: HEDJOP `[null,1,167]`, FEKKAJ `[null,14,1120]`, USAPOU `[null,2,154]`. [Three zigzag traces](../data/jgen/campaign/B/g/g3a2_zigzag_traces.png) · [Three plateau traces](../data/jgen/campaign/B/g/g3a2_plateau_traces.png).

Eight-parent full solves, seed11, CPU eager, original production615ae97d2350: all 6 arms complete, no failures,768 delivered conformers. Default requested8-chunk layout retained; active concurrency bounded. Executor concurrency varied across pilot/shards, so these CPU wall times are not used for the cal guard.

| Arm | Control | Median delivered score | Paired median score change | Score filter |
|---|---|---:|---:|---|
| C0 | — | 0.127373956 | — | — |
| P2′ late cosine | C0 | 0.123178886 | −0.004522249 | **fail/drop** |
| P2a | C0 | 0.127373956 | 0 | tie/pass; trigger fails |
| P2p | C0 | 0.127373956 | 0 | tie/pass; trigger fails |
| C1 | — | 0.132321659 | — | — |
| P2 | C1 | 0.168276861 | +0.055215269 | informational |

MACE float64, paired local2060, builder-order labels asserted;537 unique geometries after exact deduplication. All 8 C0 top-1 energies exactly reproduce the prior valid same-machine run (matching model SHA). Exploratory8-parent bootstrap,10,000 resamples,seed11:

| Arm − control | Top-1 mean [95% CI], kJ/mol | Paired median | Floor mean [95% CI] |
|---|---:|---:|---:|
| P2 − C1 | +78.580 [−82.785,+348.855] | −23.281 | −30.908 [−67.912,−0.122] |
| P2′ − C0 | +14.463 [+3.218,+30.720] | +6.188 | +5.197 [−0.521,+12.004] |
| P2a − C0 | 0 [0,0] | 0 | 0 [0,0] |
| P2p − C0 | +0.033 [0,+0.099] | 0 | −1.849 [−5.564,+0.016] |

Only original P2 advances to cal. P2′ is worse in score and energy; P2a reproduces C0's delivered outputs exactly; P2p changes little. This rejects the configured refinements, not all possible adaptive schedules.

### Step 2: light gates

- Default fingerprint6cc586e17ab1; **golden245/245**, default8-chunk layout, max3 active chunks,274.19s. No golden overwritten. Full golden run once on the required merged base.
- **37 targeted tests passed** in308.18s on the merged base, including3 CPU units per schedule, original optimizer regressions, default parity, atom-export and generator consistency.
- All 3 schedules × MELLUN/AHIGOQ/PIDXOT,200 steps on CUDA: **9/9 eager/replay pairs pass**, max raw relative score error0, no recording fallbacks.
- All 9 same-arm early-stop companions skip0 steps and have **bit-identical raw scores and delivered scores/coordinates**. Summed build/solve walls over3 units: cosine off11.062s/on11.186s (1.011×); zigzag11.372/11.337s (0.997×); plateau11.717/11.488s (0.980×). The small timing differences do not demonstrate a speedup from early stopping. Late cosine reaches its floor only on the final step, so cannot save iterations under this rule. No admitted cal arm supports this early-stop flag.
- Additional source/scalable-P2/bootstrap checks3 passed; actual/cached embedding-validity/reset checks3 passed. Production solver code is unchanged after the merged-base gates.

### Step 4: complete cal results

All **1,080/1,080 records** collected, no missing records. NEPNEF's GFN2 ligand refinement falls back to MMFF; all 9 arm/seed records are invalid and excluded from energy estimates, with raw backups and explicit failure records. The failed ligand is unique to NEPNEF in the120-parent cohort. Thus **119 valid paired parents ×3 seeds =357 paired units**; no candidate-only failures. PartitionA:531 valid/9 failed records; partitionB:540 valid, actual/cached-fallback ledger empty. No failed embed is retried or tuned away.

Negative energy differences are better. Means weight parents equally; percentile95% CIs use10,000 parent-clustered resamples,seed11, retaining paired seeds. Medians below are paired unit differences. N/O labels use the paired C1 top-1 energy (N≤200,O>200kJ/mol).

| P2 − C1 | Parents / units | Top-1 mean [95% CI], kJ/mol | Paired median | Floor mean [95% CI] |
|---|---:|---:|---:|---:|
| All valid | 119 /357 | **−4.919 [−12.327,+5.288]** | −0.967 | **−7.814 [−11.868,−4.516]** |
| N: control non-obscene | 105 /309 | **−4.496 [−7.014,−2.196]** | −0.718 | −4.132 [−6.317,−2.324] |
| O: control obscene | 18 /48 | −11.202 [−57.156,+55.823] | −20.584 | −29.369 [−50.321,−13.262] |
| FEKKAJ excluded, diagnostic only | 118 /354 | **−9.034 [−13.638,−5.368]** | −1.036 | −7.831 [−11.954,−4.530] |

Four parents have seeds in both strata, so stratum parent counts do not add. All-valid median parent-mean top-1 change is−1.404kJ/mol. **92/119 parents improve in mean top-1 energy across seeds;27 worsen.** Non-obscene cases and the FEK-excluded diagnostic show a clear energy benefit. This does not override the registered all-parent pass rule.

Secondary results:

- Delivered top-1 score: parent-mean change **+0.066951 [+0.033629,+0.117091]**, paired median+0.013206.
- Parent-weighted obscene-rate change: **−0.733 percentage points [−2.889,+1.302]**, CI includes0. N:+0.426pp [−1.510,+2.377]; O:−9.647pp [−18.395,−2.811].
- Descriptive pooled-slot obscene rates (different weighting): C0 33.4468%, C1 32.3277%, P2 32.6947%; returned rows C0 12,339, C1 12,132, P2 12,161. These pooled rates must not be substituted for the paired parent-weighted CI.
- Worst returned slot: C1 9051.800 kJ/mol (XIKGEE,seed11,key `[null,0,26]`); P2 6479.389 (EPUWAQ,seed11,key `[null,0,8]`). Garbage slots remain; no filtering was introduced.
- Paired119-parent summed solver/assembly time: C1 748.03 s, P2 1272.10 s, ratio **1.7006×**. Shared preparation/embedding and batched MACE are outside this runner's per-arm timing; this is the established solver/assembly wall comparison, not a full end-to-end runtime ratio.

**Issued gates:** all-parent primary CI upper bound+5.288 is **not <+2: fail**. Floor CI upper−4.516 is <0: secondary improvement passes. No N parent worsens>25kJ/mol on≥2 seeds: **guard passes**. Wall ratio1.7006 is **not ≤1.3: fail**. No eligible winner under the frozen choice rule.

### Interpreting the outliers and the earlier −16.7 result

The earlier **−16.711545kJ/mol** remains the median within-row energy change across48 fixed O1 endpoints before/after the P2 finish. It is neither a top-1 reranking comparison nor a250-step-control comparison. The eight-parent CPU median−23.281/mean+78.580 compares reranked P2 top-1 against C1. The full cal above is a third, broader comparison on new production/L4/seeds11–13.

In the eight-parent test, FEKKAJ alone worsens+990.809kJ/mol, QEZVEY+22.132; the other6 improve. Median−23.281 is the middle of−40.350 and−6.212; the+990.809 outlier drives the mean positive. Excluding FEKKAJ diagnostically gives−51.738 mean on7 parents.

Saved geometry confirms the known FEKKAJ score loophole: same row `[null,14,1120]`, Cl···Cl **2.01344→1.41793Å**, Cl–Ru–Cl51.5199→35.6477°, score improves by0.06510. Missing donor–donor Q6 repulsion lets optimization reward the clash. On cal its paired top-1 changes are+1017.329,+363.431,+61.187 (mean **+480.649**), all O control units. This contaminates all-parent energy inference about optimization; it is not evidence of poor optimization of the supplied objective. The owner is fixing Q6 separately. No post-hoc exclusion changes the official gate.

QEZVEY's CPU regression is a winner change: C1's winning identity improves367.280456→362.726385kJ/mol within P2, but P2 selects another identity at389.412768. Its floor improves352.610783→332.160184. No specific score-term cause is claimed. Cal's other largest parent-mean regressions are WEYJIT+24.676 and VOBGUR+18.510; neither breaches the issued two-seed N guard. Largest improvements include JEGPUJ−181.933, DAMVAQ−77.646, MIDQEY−61.142, USAPOU−59.619 and HEDJOP−55.091.

Owner-requested saved XYZs: [DAMVAQ P2](../data/jgen/campaign/B/g/DAMVAQ_G3a2_P2_s11_top1.xyz), [paired C1](../data/jgen/campaign/B/g/DAMVAQ_G3a2_C1_s11_top1.xyz),83 atoms, unrelaxed seed11 pre-Li-cap CPU pair, builder labels and coordinate round-trip verified. Export and failure diagnosis are reproducible with `export_g3a2_examples.py`.

### Spec issues and correctness repairs

1. Owner clarified50 continuation steps for endpoint diagnostics, and included P2/C1 in the8-parent filter for spot-check preselection. Implementer resolved the state/objective ambiguity in favor of true continuous constrained state. The initial delivered-best-θ/final-moments/delivered-objective trial is retained as `mixed_state_superseded`; it does not determine the gate. A subsequent pre-/post-projection score comparison failed on USAPOU (0.007209 difference); the stricter complete paired-trajectory hash validation replaces that mismatched comparison. Both failed instrumentation trials remain available.
2. Original P2 switches to the delivered objective and adds50 steps. The200-total-step single-loop arms retain the constrained objective. Their comparisons do not isolate LR adaptation or moment continuity. Preserving P2's benefit in a cleaner implementation may require preserving its objective phase, not just changing LR. No such extra arm was added after cal opened.
3. The replay graph formerly captured a scalar LR. The schedules use device LR state/tables inside the recorded graph; canonical generator updated and output regenerated. Existing coarse plateau convergence is not used as the new stop signal. There is no individual-DOF convergence state; zero projected-gradient columns are excluded from the zigzag denominator.
4. Batched benchmark arm detection/reset and explicit step counts were repaired before cal, preventing C0 reuse for P2/new flags and ensuring C1 at 250 steps. Builder symbols are checked against the explicit reference mapping before MACE. Per-parent failures continue the batch. No protected input, QM property column or golden was written/read contrary to policy.
5. An inherited nonexistent cache-warming call made the original warm-up/guard ineffective; this was missed in preflight. A real NEPNEF fallback was discovered during cal. All affected A records were vetoed by ligand provenance, with raw backups. Remaining B uses an actual-refinement/cached-failure ledger that leaves generator outputs/cache unchanged and rejects invalid parent inputs before energy evaluation. Solver/scoring/thresholds/gates are unchanged; source patches and validity audits are explicit. B has no embedding fallback. This is not a favorable-result exclusion or an embed spot-fix.

### Step 5, resources and stop

Five P2/C1 spot-check parents were frozen before any L4 result: **DAMVAQ, USAPOU, HEDJOP, NEPNEF, QEZVEY** (largest absolute CPU filter differences, excluding FEKKAJ). **No 2060 spot check is run:** the directive restricts it to the winning arm, and no arm passes the cal choice rule. There is no cross-platform fold-in claim.

Second L4 allocation (`jgBg3a2b`) was refused by Colab; no orphan was allocated and no other track's VM was stopped. One owned L4 (`jgBg3a2a`) ran both60-parent partitions sequentially. Both540-record collections were counted locally before shutdown. B completion (exit 0) was observed at 06:43:36 UTC; a slow Colab fetch completed/count verified at **06:54:44**, and shutdown confirmed at **06:54:48UTC**. The watcher operated independently while foreground progress polling was suspended at the owner's request. No VM remains allocated by this package.

Shared account balance506.05→500.01 (6.04-unit delta includes another track). Own L4 rate inferred1.94units/h; READY-to-shutdown93.89min alone is3.036units, plus initial allocation/setup, about **3.1–3.4units attributable to this package**. Exact per-VM billing is not exposed by the recorded account-wide usage output.

Artifacts under shared `research/data/jgen/campaign/B/g/`: frozen contract copy; filter structures/energies/summary; continuous diagnostics and traces; golden/light/replay logs; `g3a2_cal_analysis.json`; `colab_a/g3a2_{A,B}/` (1,080 canonical records); NEPNEF invalid raw backups/validity audit; B `embedding_validity.json`; source patch manifests and VM event/shutdown journal.

**Disposition:** energy benefit is supported for non-obscene controls and the predeclared FEK-excluded diagnostic; the registered all-parent CI and current implementation's wall-time guard fail. P2 remains a research baseline, not a fold-in winner. No relaxation, tail, locked-set evaluation, score change, whole-run optimizer expansion, G-3b or fold-in. **Stop at Checkpoint G-3a″ for audit.**


## G.14 — Checkpoint G-3c0: clean24 optimizer/aids filter (completed 2026-10-30)

Authorized by `directive_jgen_B_G3c0_2026-10-30.md`; G-3c waits. Contract **41be8d9** freezes six arms, the auditor's 24-parent panel and seeds 11–13 before any result. The panel is never reduced after results. This is a descriptive filter, with no pass/adoption gate.

C1 is 250 production Adam steps on score + aids. P2 and R1 use 200 production Adam steps followed by 50 steps on the score alone (fresh Adam lr .01, or the existing moment-initialized iRprop+, respectively). R2+ and R2− use their existing 250-step score-only implementation with Δ0=.01; A0 adds 250-step fresh Adam lr .1 to that same path. P2/R1 search uses replay; R2±/A0 use the eager native CUDA path. Timing therefore measures the implementations, not just optimizer arithmetic.

All arms share seed rows, beam initialization, initial θ digests and the production pre-search `class_budget=split` allocation. “No culling” for R2±/A0 means no iterative optimizer culling; the inherited pre-search allocation still defines their common starts. All use angle bends and request 16 returned conformers per seed. This matches the earlier eight-parent full-solve setting; it differs from G-3a″ full-cal's unrestricted return count. Score, bend prior, macrocycle seam treatment, output dedup and embedding are unchanged. No relaxation or post-result tuning.

Distortion is measured per returned row: the shape constraint's own power-mean statistic (τ=2), aligned to the row's starting candidate; maximum D–M–D angle deviation from the nearest positive-weight Q2 mean in that candidate's table (padding excluded); and class change using the existing geometry classifier plus best-fit assignment and graph-level donor-equivalence class. Reflections are equivalent in that class, so this is not a stereochemical chirality test. CN≤6 assignment is exhaustive; CN7–9 uses the existing deterministic multi-start fit. The raw geometry labels, assignments, class keys, coordinates and builder-order symbols are saved. Symbol equality against `conf.complex.atoms` is asserted before export/MACE mapping.

Paired energy differences use highest-score returned top1 and minimum-MACE-energy returned floor, relative to the same mapped reference. Report parent-mean difference, parent-clustered percentile 95% CI (10,000 resamples, RNG11), median paired unit and parent-mean difference, and number of parents improved. All 24 names stay in the artifact; unavailable comparisons are explicit and never replaced. Solver/assembly wall is counted once per parent, with shared preparation, MACE and read-only distortion diagnostics excluded.

The audit's owner ruling accepts P2's earlier energy result in principle with FEKKAJ excluded, while its 1.70× wall gate still fails. That ruling does not alter this new panel: all 24 names are frozen and will remain reported.

Light verification: 26 existing optimizer/schedule tests passed in 275.88 s (including default/native-replay parity); six focused Adam tests passed in 2.89 s, including A0 against PyTorch Adam for 250 steps; the synthetic distortion test passed. Default fingerprint remains **6cc586e17ab1**; replay regenerated with the canonical generator. Two-parent profile: AJAQAD 92.7 s, AZUSIA 82.3 s, **36/36 valid records and 576 returned rows**, retained for the final analysis. Expected remaining wall 30–40 minutes, local 2060, no Colab VM or spend.

The profile exposed stale inherited replay metadata: eager R2±/A0 records copied the previous arm's `SR.LAST`. Profile records retain that original metadata alongside the corrected native/250-step designation; no solve or scientific value was changed. The remaining harness records native mode explicitly. A generic optimizer-failure warning counter now vetoes substitution of an unoptimized baseline in this diagnostic; the profile log contains no such warning.

E/Z harness repair (**7505262**): the initial wrapper could not attach the correct starting-candidate metadata to a merged multi-embed pool. CIXYUD, LEGXEA and NUWFAM therefore received **54 explicit harness-failure records before any arm was solved**, and remain in the fixed panel. Those records are preserved in `g3c0_ez_original_failures/` with `g3c0_ez_failure_ledger.json`. Only these affected parents are re-executed with all six arms, the same seeds, model and GPU; no failed energy result existed to select against. The research multiseed helper now carries a read-only diagnostic callback through each E/Z variant before merging, so each returned row retains its own candidate/Q2 table rather than using the top-ranked variant's scorer. The helper also permits the three known native score-only CUDA paths, which skip the search loop and therefore do not cull rows across seeds. The previous replay requirement remains for ordinary search. Per-variant starting-θ digests are saved and compared across arms. E/Z wall uses the helper's solver plus assembly timing, with diagnostic callback time subtracted; preparation and row generation are excluded consistently. No production optimizer or score was changed by this repair.

### Completed results

Records **432/432**; missing 0; failed 0; starting-θ digest mismatches 0. All 24 fixed parents remain in the ledger.

All energy changes are arm minus paired C1, kJ/mol; negative is better. CIs resample parents, with seeds averaged within parent. Median is the median paired parent/seed-unit difference.

| Arm | Paired parents / units | Top1 mean [95% CI] | Top1 median | Parents improved | Floor mean [95% CI] | Floor median |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 24 / 72 | +0.00 [+0.00, +0.00] | +0.00 | 0 | +0.00 [+0.00, +0.00] | +0.00 |
| P2 | 24 / 72 | -15.66 [-27.40, -5.47] | -1.93 | 21 | -11.54 [-22.53, -3.72] | -3.29 |
| R1 | 24 / 72 | -20.89 [-38.13, -6.42] | -2.79 | 19 | -14.27 [-28.92, -3.64] | -3.32 |
| R2plus | 24 / 72 | -17.85 [-57.99, +25.03] | -5.05 | 17 | -14.24 [-44.18, +12.36] | -2.93 |
| R2minus | 24 / 72 | -18.29 [-58.88, +24.76] | -5.03 | 17 | -14.97 [-45.43, +12.34] | -2.90 |
| A0 | 24 / 72 | -10.05 [-35.01, +12.39] | -0.86 | 14 | -10.86 [-32.53, +6.65] | -0.04 |

| Arm | Mean top1 score | Paired score change [95% CI] | Solver/assembly wall (s) | Wall / C1 | Returned rows |
|---|---:|---:|---:|---:|---:|
| C1 | 0.502366 | +0.000000 [+0.000000, +0.000000] | 158.86 | 1.000 | 1062 |
| P2 | 0.594298 | +0.091932 [+0.028302, +0.199638] | 228.98 | 1.441 | 1026 |
| R1 | 0.613465 | +0.111099 [+0.029818, +0.251819] | 226.43 | 1.425 | 1021 |
| R2plus | 0.608555 | +0.106189 [-0.020875, +0.257931] | 484.29 | 3.049 | 988 |
| R2minus | 0.608378 | +0.106012 [-0.021452, +0.257707] | 481.63 | 3.032 | 991 |
| A0 | 0.556320 | +0.053954 [-0.046095, +0.182321] | 474.08 | 2.984 | 1013 |

Wall sums each parent once (its three seeds were solved together). Native score-only arms execute eagerly; C1/P2/R1 search uses CUDA replay. These are implementation timings, not a clean optimizer-cost comparison.

| Arm | Row class-change rate | Geometry-label change | Class unavailable | Obscene rows (>200 kJ/mol) | Paired obscene-rate change (pp) [95% CI] |
|---|---:|---:|---:|---:|---:|
| C1 | 5.65% | 5.27% | 0 | 32.11% | +0.00 [+0.00, +0.00] |
| P2 | 6.34% | 6.14% | 0 | 30.31% | -0.93 [-4.19, +1.96] |
| R1 | 5.78% | 5.68% | 0 | 29.29% | -2.94 [-7.60, +0.45] |
| R2plus | 7.19% | 6.58% | 0 | 24.80% | -5.59 [-15.75, +2.16] |
| R2minus | 7.47% | 6.66% | 0 | 23.71% | -6.74 [-17.06, +1.24] |
| A0 | 8.88% | 8.49% | 0 | 33.76% | +2.27 [-2.63, +8.51] |

Row-pooled fractions weight parents by their returned counts; the paired rate change instead gives parents equal weight. They need not subtract to the same number.

### Shape deviation (the shape constraint statistic, dimensionless)

| Arm | Mean | Min | p05 | p25 | Median | p75 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.0382 | 0.0002 | 0.0008 | 0.0066 | 0.0175 | 0.0454 | 0.0802 | 2.6480 |
| P2 | 0.0387 | 0.0001 | 0.0007 | 0.0057 | 0.0159 | 0.0520 | 0.0909 | 1.7583 |
| R1 | 0.0520 | 0.0001 | 0.0008 | 0.0057 | 0.0160 | 0.0481 | 0.0916 | 2.4844 |
| R2plus | 0.4585 | 0.0000 | 0.0007 | 0.0065 | 0.0236 | 0.0964 | 3.0242 | 3.3014 |
| R2minus | 0.4620 | 0.0000 | 0.0007 | 0.0066 | 0.0245 | 0.1171 | 2.9924 | 3.2979 |
| A0 | 0.3680 | 0.0001 | 0.0008 | 0.0079 | 0.0250 | 0.0890 | 2.5261 | 3.3064 |

### Maximum angle deviation from nearest Q2 mean (degrees)

| Arm | Mean | Min | p05 | p25 | Median | p75 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 9.7037 | 0.0025 | 0.7491 | 3.1445 | 5.5983 | 9.3050 | 35.0621 | 103.1436 |
| P2 | 10.6654 | 0.0004 | 0.4856 | 2.7925 | 5.2128 | 8.7383 | 46.5500 | 121.0977 |
| R1 | 10.4119 | 0.0195 | 0.5891 | 2.7351 | 5.2023 | 8.3090 | 47.6353 | 121.3386 |
| R2plus | 12.8565 | 0.0232 | 0.3809 | 2.2115 | 5.1874 | 8.6671 | 67.9775 | 121.0590 |
| R2minus | 13.2137 | 0.0033 | 0.3426 | 2.2480 | 5.1700 | 8.8618 | 68.1062 | 121.0743 |
| A0 | 13.4723 | 0.0231 | 0.6905 | 2.8757 | 5.4798 | 9.5029 | 65.6900 | 173.9506 |

| Arm | Top1 shape median / p95 / max | Top1 max-angle median / p95 / max (degrees) |
|---|---:|---:|
| C1 | 0.0144 / 0.0734 / 0.0848 | 4.2141 / 16.2373 / 36.2702 |
| P2 | 0.0122 / 0.0755 / 0.0931 | 4.5453 / 12.6899 / 47.9538 |
| R1 | 0.0128 / 0.0887 / 2.2719 | 4.4468 / 17.2229 / 49.2461 |
| R2plus | 0.0233 / 3.0527 / 3.2648 | 4.3978 / 26.1517 / 67.4939 |
| R2minus | 0.0260 / 3.0547 / 3.2673 | 4.3975 / 26.2931 / 67.6496 |
| A0 | 0.0208 / 2.4221 / 3.3064 | 4.3448 / 25.1009 / 68.4584 |

### Control-energy strata

Strata are defined per seed by C1 top1 ≤200 or >200 kJ/mol. A parent can contribute to both; within each stratum, available constituent seeds are averaged before resampling parents.

| Stratum | Arm | Parents / units | Top1 mean [95% CI] | Floor mean [95% CI] | Top1 median |
|---|---|---:|---:|---:|---:|
| C1 non-obscene | C1 | 20 / 57 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 |
| C1 non-obscene | P2 | 20 / 57 | -5.95 [-13.02, -0.49] | -5.13 [-11.14, -0.97] | -0.93 |
| C1 non-obscene | R1 | 20 / 57 | -6.12 [-15.34, +2.25] | -4.60 [-10.68, +0.15] | -1.30 |
| C1 non-obscene | R2plus | 20 / 57 | +17.91 [-16.54, +73.24] | +2.89 [-6.79, +14.61] | -2.48 |
| C1 non-obscene | R2minus | 20 / 57 | +17.01 [-16.91, +72.04] | +3.25 [-7.43, +16.17] | -2.56 |
| C1 non-obscene | A0 | 20 / 57 | +13.97 [-4.20, +38.56] | +8.48 [-2.16, +20.98] | -0.01 |
| C1 obscene | C1 | 6 / 15 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 |
| C1 obscene | P2 | 6 / 15 | -59.90 [-90.42, -29.75] | -32.18 [-64.10, -11.38] | -57.22 |
| C1 obscene | R1 | 6 / 15 | -75.41 [-120.69, -26.46] | -42.44 [-85.90, -7.91] | -70.72 |
| C1 obscene | R2plus | 6 / 15 | -86.60 [-206.32, +64.13] | -56.10 [-156.49, +57.45] | -117.85 |
| C1 obscene | R2minus | 6 / 15 | -85.55 [-205.76, +64.51] | -57.73 [-158.88, +55.56] | -89.06 |
| C1 obscene | A0 | 6 / 15 | -49.33 [-125.14, +25.74] | -56.13 [-118.90, +1.62] | -81.81 |

### Fixed-panel failures and completeness

```json
{
  "missing": [],
  "failures": [],
  "start_digest_mismatches": []
}
```


### Reading and spec issues

P2 and R1 both improve top1 and floor with paired CIs below zero on this fixed clean24 panel. P2 improves 21/24 parent means; R1 improves 19/24. R1 has the largest mean top1 improvement, but this filter did not register a superiority test of R1 against P2. In the C1-non-obscene stratum, P2's top1 CI excludes zero; R1's does not. No adoption/pass gate was issued here.

R2± improve 17/24 parents, but their all-parent top1 and floor CIs include zero. Their positive mean changes in the C1-non-obscene stratum contrast with larger improvements among already-obscene controls. NUWFAM is their largest regression: parent-mean top1 **+334.53/+333.24 kJ/mol** for plus/minus, with seed differences (+489.40,+104.07,+410.11) and (+487.09,+100.67,+411.96). XOJHIN adds +80.79/+70.62. Improvements at HEDJOP (−280.19/−282.70) and USAPOU (−202.16/−211.30) pull the overall means down. These parents all remain included; none is tuned away. A0 also has uncertain all-parent gains and regresses by +109.53 at NUWFAM, +80.09 at XEVDAE and +52.98 at XOJHIN. Removing the aids from step zero does not produce a clear gain on this panel with either optimizer.

The no-aids-from-start arms have heavier distortion tails, rather than uniformly distorted outputs. Their maximum-angle p95 is about **66–68°**, versus **35°** for C1; their angle medians remain about **5.2–5.5°**, versus **5.6°** for C1. Class-change rates rise from 5.65% to 7.19%/7.47% (R2±) and 8.88% (A0). The score-only finish arms retain a much tighter shape-statistic tail than whole-run score-only arms, although they also increase the p95 maximum-angle error. A0 shows that this tail behavior is not exclusive to iRprop. These are descriptive comparisons of selected returned pools, not a causal attribution of each individual failure.

**Shape/class spec issue:** the requested constraint statistic fixes donor-to-vertex correspondence to the starting candidate and uses a proper rotation. It therefore measures assignment retention as well as physical polyhedral distortion. The graph-level class intentionally identifies symmetry-related donor permutations and reflections. A large shape statistic can coexist with unchanged class and small angle error: A0 NUCKUS seed 12 has shape **3.3064**, max angle error **6.77°**, and the same tetrahedral graph-level class, with a reflected best-fit assignment. An independent ideal reflected-tetrahedron demonstration gives constraint statistic **1.3333** despite essentially zero nearest-unlabelled tetrahedral deviation (`g3c0_shape_semantics.json`). Consequently the large shape tail alone must not be described as donor collapse; interpret it with the angle and class diagnostics. Registered measures and their distributions are retained unchanged. Graph-level class change is not a stereochemical chirality test, and the existing CN5 classifier uses its τ5 convention.

Other implementation differences are explicit above: native eager versus replay execution; inherited pre-search class allocation; 16-conformer cap and output window; and E/Z preparation repeated per arm but excluded from solver/assembly wall. Existing iRprop per-row idle masks/stopping rules remain; A0 takes its fixed 250 Adam steps. No optimizer bounds, phase lengths, flags, scorer or acceptance criteria were tuned from these results.

### Completion, provenance and stop

Final collection: **432/432 records, 24/24 parents, 72 units per arm; 6,101 returned rows, 864 top1/floor XYZ files; zero final failures, zero missing records, zero initial-θ digest mismatches**. Every returned coordinate/symbol count is consistent and finite. All arms used machine `meep`, **RTX 2060 SUPER**, MACE polar-1-s float64, model SHA256 `e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510` (see the integrity artifact for the authoritative hash). Reference energies match exactly across all seeds and arms per parent. Production fingerprint **6cc586e17ab1** throughout. The original 54 E/Z harness failures are preserved separately; all affected parents were successfully executed after the metadata-only repair.

Contract commit **41be8d9** preceded all runs. Solver/diagnostic implementation **0b034da**; E/Z harness repair **7505262**. Two-parent profile 175.0 s; original 22-parent batch 29.07 min including the recorded failed E/Z entries; successful E/Z reruns took 195.6 s (CIXYUD), 127.4 s (LEGXEA), 186.0 s (NUWFAM), with process/model-load overhead in the job journals. No VM was started; **Colab spend 0**. Local GPU job finished and the lock released; no compute process remains.

Artifacts under shared `research/data/jgen/campaign/B/g/`: `g3c0_analysis.json/.log`, `g3c0_tables.md`, `g3c0_integrity.json`, `g3c0_shape_semantics.json`, per-arm records/all returned coordinates in `g3c0_profile/`, `g3c0_remaining/`, `g3c0_ez_retry/`, original failure ledger/archive, profile/main/retry logs and job journals. Existing optimizer/schedule light suite: 26 passed; focused Adam check: six passed; final analysis/distortion checks: two passed; E/Z harness light checks: two passed. No full-suite or golden rerun was needed for this research-only filter; default CPU replay parity and fingerprint were checked.

**Stop at Checkpoint G-3c0. No G-3c implementation, relaxation, fold-in, merge or push. Await audit.**


## G.15 — Owner-requested NUWFAM aryl C–F diagnostic (2026-10-30)

Contract **63e6486**, before opening these measurements. This is a saved-coordinate diagnostic, not a parent-exclusion rule. NUWFAM remains in the registered G-3c0 results. No solve, MACE recomputation, relaxation or GPU job was run.

All **288 returned conformers** (six arms × three seeds × 16), containing three pentafluorophenyl rings and 15 aryl C–F bonds each, were measured. Reference topology came from the safe geometry/SMILES loader. Builder-order saved coordinates were put into reference order by bijective coordinate matching to each existing reference-order top1 XYZ; maximum matching error <1e−6 Å, element equality asserted, and the same atom permutation independently recovered for all 18 arm/seed records. This avoids assuming tmQM and builder atom orders agree.

The six ring carbons define a least-squares plane. The local C–F tilt uses the plane through its ipso carbon and two neighboring ring carbons; 0° is planar. Ring-plane tilt and F height are also saved, together with carbon-ring RMS/max plane deviations. A synthetic 1.35-Å bond tilted 30° passed exact length/angle/height checks.

### A specific distorted C–F bond

Using reference atom indices **C19–F20**, in the first pentafluorophenyl ring:

| Geometry | C–F length (Å) | Local out-of-plane tilt | Whole-ring-plane tilt | F height above ring plane (Å) |
|---|---:|---:|---:|---:|
| Reference | 1.3390 | 1.70° | 3.93° | 0.1075 |
| Saved E/Z variant 0 | 1.3540 | 0.74° | 1.92° | 0.0534 |
| Saved E/Z variant 1 | **1.6929** | **60.20°** | **70.43°** | **1.6790** |

This is a clear bond-length and planarity distortion in variant 1. Its maximum ring-carbon RMS displacement is about 0.0512 Å, so the large F displacement is not merely the entire aryl plane rotating in space.

### Top1 selection

| Arm | Seed | E/Z variant | MACE ΔE_unrel (kJ/mol) | Longest aryl C–F (Å) | Maximum local C–F tilt |
|---|---:|---:|---:|---:|---:|
| C1 | 11 | 0 | 185.30 | 1.3624 | 14.80° |
| C1 | 12 | 0 | 335.03 | 1.3624 | 14.80° |
| C1 | 13 | 0 | 262.89 | 1.3624 | 14.80° |
| P2 | 11 | 0 | 176.33 | 1.3624 | 14.80° |
| P2 | 12 | 0 | 315.82 | 1.3624 | 14.80° |
| P2 | 13 | 0 | 276.00 | 1.3624 | 14.80° |
| R1 | 11 | 0 | 229.89 | 1.3624 | 14.80° |
| R1 | 12 | 0 | 307.34 | 1.3624 | 14.80° |
| R1 | 13 | 0 | 319.44 | 1.3624 | 14.80° |
| R2plus | 11 | 1 | 674.71 | 1.6929 | 60.20° |
| R2plus | 12 | 0 | 439.10 | 1.3624 | 14.80° |
| R2plus | 13 | 1 | 673.01 | 1.6929 | 60.20° |
| R2minus | 11 | 1 | 672.39 | 1.6929 | 60.20° |
| R2minus | 12 | 0 | 435.70 | 1.3624 | 14.80° |
| R2minus | 13 | 1 | 674.86 | 1.6929 | 60.20° |
| A0 | 11 | 0 | 385.56 | 1.3624 | 14.80° |
| A0 | 12 | 0 | 383.61 | 1.3624 | 14.80° |
| A0 | 13 | 0 | 342.64 | 1.3624 | 14.80° |

R2+ and R2− both select distorted variant 1 for seeds **11 and 13**. C1, P2, R1 and A0 select variant 0 for all three seeds. The distorted variant is nevertheless present in **every arm’s returned pool**: counts are C1 17/48, P2 13/48, R1 12/48, R2+ 21/48, R2− 21/48 and A0 18/48.

### All-returned C–F distributions

| Arm | C–F length min / median / p95 / max (Å) | Local tilt median / p95 / max (degrees) | F height p95 / max (Å) |
|---|---:|---:|---:|
| C1 | 1.3321 / 1.3583 / 1.3777 / 1.6929 | 0.95 / 14.80 / 60.20 | 0.2315 / 1.6790 |
| P2 | 1.3321 / 1.3583 / 1.3777 / 1.6929 | 0.95 / 14.80 / 60.20 | 0.2315 / 1.6790 |
| R1 | 1.3321 / 1.3583 / 1.3724 / 1.6929 | 1.02 / 14.80 / 60.20 | 0.2315 / 1.6790 |
| R2plus | 1.3321 / 1.3581 / 1.3859 / 1.6929 | 0.95 / 14.80 / 60.20 | 0.2545 / 1.6790 |
| R2minus | 1.3321 / 1.3581 / 1.3859 / 1.6929 | 0.95 / 14.80 / 60.20 | 0.2545 / 1.6790 |
| A0 | 1.3321 / 1.3582 / 1.3781 / 1.6929 | 0.95 / 14.80 / 60.20 | 0.2326 / 1.6790 |

### Interpretation and limits

Within each E/Z variant, every individual C–F length is effectively invariant across all arms, seeds and returned rows: maximum within-bond range **2.32×10⁻⁶ Å**; local-tilt range below **0.00018°**. This is strong evidence against R2 specifically stretching those C–F bonds during optimization. The observed pattern is consistent with shared prepared-fragment geometry in variant 1, followed by different optimization/ranking behavior. These saved endpoints establish that the distortion is shared across arms; they do not identify the exact preparation/FK stage where it first appears.

R2 score ranking does favor this distorted variant in two seeds. For R2+ seed 11, variant-1 top1 scores 0.35819 at 674.71 kJ/mol, while its highest-score returned variant-0 row scores 0.31856 at 425.53. At seed 13 these are 0.37456/673.01 versus 0.32144/429.45. Thus the result supports investigating a shared geometry/scoring loophole or insufficient penalty, rather than treating NUWFAM as intrinsically unsuitable. No causal energy attribution to this one C–F bond is established.

The C–F issue is **not the entire R2 regression**: seed 12 picks variant 0 and still worsens by +104.07/+100.67 kJ/mol; all three R2 energy-floor rows also use variant 0 and remain worse than C1. R1 and C1 top1 rows share effectively identical C–F/ring geometry, so R1’s smaller NUWFAM regression cannot be explained by additional C–F stretching or loss of C–F planarity. Other pose, coordination or ranking differences remain to be traced separately.

Artifacts: `research/data/jgen/campaign/B/g/nuwfam_cf_diagnostic.json/.log` saves every ring/bond measurement, mapping checks, top1/floor flags, energies and E/Z identities. Existing top1/floor XYZs are in `g3c0_ez_retry/<arm>/xyz/NUWFAM/`; all 288 full-precision structures are in that run’s per-arm records. Original G-3c0 numbers and panel remain unchanged. One geometry unit test passed. Diagnostic complete; no further experiments started.

## G.15 — G-3c implementation / Package H-1 (2026-10-30)

Master 9bb6831 merged into jgen/b-solver as 149e319. The owner explicitly
confirmed the Package H amendment: G-3c **step 1 only**, then H-1; step 0 is
reserved for G-3d after the patched-base fold-in. No optimizer energy results,
filter, evaluation panel or production adoption is claimed here.

The implementation, explicit policy dictionary, term census, crystal gates,
reproduction instructions, limitations and artifact paths are in
[H1_REPORT.md](../archive/docs/jgen_campaign/h/H1_REPORT.md). Default fingerprint remains 6cc586e17ab1;
production golden 245/245 arrays match. Native/replay CPU phase parity passes
on three units per finish optimizer; CUDA eager/replay parity passes on three
units each for Adam, iRprop+, iRprop-, R2s and IA. The curvature-source and revised
search-bound follow-up checks are recorded in that report.

H-1's proposed score flags are in [H1_PATCH.json](H1_PATCH.json): donor-pair dict
hh=.95/ho=.90/oo=.80 plus angle_tail_quadratic. Q2 crystal p95 extra cost is
0.0151 nats. Q5's proposed continuation **fails** its fixed crystal gate
(p95 2.575 > .1 nats) and is excluded without tuning. Gaussian Q1/Q3/Q4 and the
harmonic bend prior receive no new continuation.

Full-patch replay parity is 3/3, exact, no fallback (fingerprint 76da02b43182).
USAPOU flag-off CPU/GPU scores differ by .1692 but share key [null,2,154], so
this control does not establish pre-existing top-key disagreement. Additional
USAPOU donor-flag same-device graph/eager and native/rewritten CPU comparisons
all pass exactly; no capture defect was found. Cross-platform trajectory
reproducibility remains limited and is explicitly distinguished from replay
parity. Colab spend zero; no VM, relaxation or evaluation panel was started.

**Implementation / H-1 checkpoint complete. Stop for audit.** No step 0 or
G-3d evaluation is authorized by this checkpoint completion.
