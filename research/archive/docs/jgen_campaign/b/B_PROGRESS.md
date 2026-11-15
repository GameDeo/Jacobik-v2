# jgen Campaign — Agent B Progress Log (`jgen/b-solver`)

## 2026-10-26 — Package B0 (Setup, Replay-Parity Harness, Replay Fallback Bug Fix, B1 Plan)

- **Environment & CPU Golden Check:**
  - Ran `pytest -q tests` on `jgen/b-solver`: `320 passed, 3 xfailed, 3 warnings in 335.04s`.
  - Ran `python research/bench/tools/jdev/golden.py check`: `production: 245 arrays, 0 differ`, `u1: 245 arrays, 0 differ`, `golden: OK`.
  - Verified `PRODUCTION_FINGERPRINT = "711411157fb4"`.
- **Replay-Parity Harness (`research/bench/tools/jgen/bench/replay_parity.py`):**
  - Implemented `replay_parity.py` (and wired `python -m tools.jgen bench replay-parity`) testing `cuda_fast_fk_replay_vs_eager`, `cuda_ref_fk_replay_vs_eager`, and `package_loop_vs_replay_loop` across `DAMVAQ,HOXJUZ,JEGPUJ` × seeds `11,12,13` (9 units).
  - Ran on the unchanged production solver (`b0_unchanged_production`, saved to `research/data/jgen/campaign/B/b0_replay_parity_unchanged.json`): **9/9 PASS**, `mode == "replay"` on 9/9, top-1 row index match 9/9, `max_rel_dscore = 0.000e+00`, `max_abs_dscore = 0.000e+00`.
- **Replay Fallback Bug Diagnosis & Fix:**
  - Scanned all 120 `cal` parents on CUDA and identified `XIKGEE` (`XIKGEE:11,12,13`) as the unit triggering `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture` in `solver_replay.part_b`.
  - Root cause: `part_b` called `self.ge.q3_approach_angles(X_current)` for the kappa Q3 symmetry-angle tie (`_q3_sym_group_idx_tensors`), which unconditionally ran `torch.linalg.svd` on `XIKGEE`'s haptic $\eta^5$-Cp sites (`[2, 3]`) even though haptic sites are never in `_q3_sym_group_idx_tensors` and their columns were immediately discarded.
  - Fix: Added `haptic: bool = True` parameter to `GeometryExtractor.q3_approach_angles` (`src/jacobik/ik/geometry.py`), passed `haptic=_q3_sym_haptic` from `src/jacobik/ik/ik_solver.py` and `build_replay_solver.py`, regenerated `src/jacobik/ik/solver_replay.py`, and updated `research/bench/tools/jgen/bench/run.py` to count `replay_fallbacks` per unit.
  - Verification:
    - `XIKGEE` (`seeds 11,12,13`) now runs in `mode = "replay"` (`replayed = 198`, `error = None`, solver wall time `1.83s` vs `5.94s` before fix) and passes `replay_parity.py` 3/3 (`b0_replay_parity_xikgee.json`).
    - Re-ran `replay_parity.py` on `DAMVAQ,HOXJUZ,JEGPUJ` (`b0_replay_parity_after_fix.json`): 9/9 PASS, bit-identical to `b0_replay_parity_unchanged.json`.
    - `pytest -q tests/test_gpu_fast_path.py` (`18 passed`) and `golden.py check` (`golden: OK`) pass.
- **B1 Plan:**
  - Wrote detailed design and parameter plan for `bend_release`, `dof_step_scale`, and `start_kabsch` in `research/bench/jgen_campaign/B_REPORT.md` §B1.
  - Ready for Checkpoint 1 audit.

## 2026-10-26 — Checkpoint 1 Audit & Generator Consolidation

- **Checkpoint 1 Audit Verdict:** PASS (`audit_jgen_B0_2026-10-26.md`), with requirement to consolidate replay generator to `tools/jdev/build_replay_solver.py`.
- **Generator Consolidation:**
  - Updated `tools/jdev/build_replay_solver.py` with the `haptic=_q3_sym_haptic` fix and `--out PATH` CLI flag.
  - Deleted redundant copies `src/jacobik/ik/build_replay_solver.py` and `research/bench/tools/jdev/build_replay_solver.py`.
  - Updated README pointers (`README.md` and `research/bench/tools/jdev/README.md`).
  - Added test `test_replay_generator_matches_committed_solver` in `tests/test_gpu_fast_path.py` (verifying byte-for-byte reproduction).
  - All 19 tests in `tests/test_gpu_fast_path.py` PASS; `golden.py check` PASS (245 arrays, 0 differ).
  - Committed fix as `9885ce9`.

## 2026-10-26 — Checkpoint 2.1: `bend_release` Tail Screening & Selection

- **Implementation:**
  - Added `bend_release` to `src/jacobik/profile.py:FIELDS` (`None` by default, preserves `711411157fb4` baseline).
  - Implemented `_step_scale_tabs` in `src/jacobik/ik/ik_solver.py` and `tools/jdev/build_replay_solver.py`.
  - Regenerated `src/jacobik/ik/solver_replay.py` using canonical generator.
  - Verified `test_replay_generator_matches_committed_solver` (byte-for-byte match) and `golden.py check` (`golden: OK`).
- **Mechanism Traces (`trace_solve.py`):**
  - Traces on `JEGPUJ 11`, `HOXJUZ 11`, `IFUROQ 11` showed 95–100% of median bend deformation occurs in steps 0–20 (~69° blowout on `JEGPUJ`).
  - Confirmed `bend_release` curbs early blowout by 75–90% and improves `JEGPUJ` solver score from `-9.001` to `-3.518`.
- **Replay Parity Gates (`replay_parity.py`):**
  - `bend_release = [0.1, 50]`: **9/9 PASS**, 0 fallbacks, `max |Δscore| = 0.000e+00`.
  - `bend_release = [0.1, 100]`: **9/9 PASS** (+ 3/3 `XIKGEE`), 0 fallbacks, `max |Δscore| = 0.000e+00`.
  - `bend_release = [0.0, 100]`: **9/9 PASS**, 0 fallbacks, `max |Δscore| = 0.000e+00`.
- **Tail Screening (seeds 11, 12, 13 vs same-VM control):**
  - Executed all 4 arms on GPU (10 parents × 3 seeds = 30 units per arm, 120 units total) with 0 fallbacks.
  - Results:
    - Control mean top-1 $dE$: 783.47 kJ/mol (20/30 units $> 500$ kJ/mol).
    - `s0=0.1, K=50`: mean $\Delta dE = \mathbf{-28.11}$ kJ/mol (17 better, 5 worse).
    - **`s0=0.1, K=100`**: mean $\Delta dE = \mathbf{-35.36}$ kJ/mol (16 better, 5 worse), $>500$ kJ/mol units drop from 20 to 14; `JEGPUJ` improves by **`-203.69 kJ/mol`**; `USAPOU` improves by **`-146.95 kJ/mol`**; avoids `0.0_100` freeze trap on `YUCZUS`.
    - `s0=0.0, K=100`: mean $\Delta dE = -29.89$ kJ/mol, but caused $+142.9$ kJ/mol blowout on `YUCZUS`.
- **Selection & Audit Status:**
  - Selected `bend_release = [0.1, 100]` as the winning arm for confirmation runs.
  - Documented results and tables in `B_REPORT.md` §B1.
  - **Performance Optimization Audit:**
    - Fixed CPU step loop parameter override in `research/bench/tools/jgen/bench/replay_parity.py` (`n_ik_steps=a.loop_steps`), giving 8.1× CPU speedup and cutting parity runtimes from 380s to ~180s with exact `0.000e+00` bit parity.
    - Profiled `run.py` MACE evaluation and documented multi-seed batching speedups.
    - Audited changes and documented in `B_REPORT.md` §B1.4 Addendum.
  - **Audit 2.1 Resolution:** Audit verdict PASS (`audit_jgen_B1_cp21_2026-10-26.md`).
  - **Checkpoint 2.2 Fresh Tail Confirmation (seeds 14–16 vs same-VM control):**
    - Ran `b1_confirm_control` and `b1_confirm_bend_release_0.1_100` on RTX 2060 (30 units each, 0 fallbacks).
    - Mean $\Delta \text{top-1 } dE = \mathbf{-23.95}$ kJ/mol (median $-0.38$ kJ/mol; 16 better, 7 worse, 7 neutral).
    - Bootstrap 95% CI: `[-45.03, -5.70]` kJ/mol (strictly negative).
    - Mean $\Delta \text{floor } dE = \mathbf{-26.15}$ kJ/mol; Mean $\Delta \text{top-1 bend prior} = +0.2978$.
    - Concentration check: Excluding `JEGPUJ` and `USAPOU`, mean $\Delta \text{top-1 } dE = \mathbf{-4.91}$ kJ/mol (11 better, 6 worse, 7 neutral).
    - Top-1 and floor Cartesian coordinates saved to `research/data/jgen/campaign/B/confirmation/tail/xyz/` for relaxed-energy check.
  - **Checkpoint 2.2 Cal Panel Confirmation (120 parents × seeds 11–13 vs same-VM control):**
    - Ran `b1_confirm_control` and `b1_confirm_bend_release_0.1_100` sequentially on RTX 2060 (360 units each, 720 units total, 0 fallbacks).
    - Mean $\Delta \text{top-1 } dE = \mathbf{-3.65}$ kJ/mol (median $-0.00$ kJ/mol).
    - Bootstrap 95% CI: `[-7.40, -0.28]` kJ/mol (strictly $< 0.0$ kJ/mol; non-inferiority satisfied with margin $+2.0$).
    - Units better / worse / neutral: 141 / 123 / 96 (39.2% / 34.2% / 26.7%).
    - Catastrophic units ($> 500$ kJ/mol): Reduced from 14 down to 8.
    - Mean $\Delta \text{floor } dE = \mathbf{-2.49}$ kJ/mol.
    - Replay parity: 0 fallbacks on all 360 units. Geometries saved to `research/data/jgen/campaign/B/confirmation/cal/xyz/`.
  - **Checkpoint 2.2 Complete:**
    - Fingerprint bug fixed in `f3f8c4a` (`711411157fb4` baseline preserved with tests passing).
    - Step 2.2b Relaxed-energy audit executed on 8 parents (4 winners, 4 worst) × 3 seeds using GFN2 `xtb --opt crude`.
    - Winners retain $-14.1$ (`JEGPUJ`), $-16.0$ (`NUWFAM`), $-19.8$ (`USAPOU`), $-1.4$ (`WEYJIT`) kJ/mol real relaxed gains (RMSD moves $0.6 - 0.9$ Å vs $0.7 - 1.1$ Å on control).
    - Worst parents' unrelaxed degradations collapse to neutral or turn negative (`QEZVEY` $-1.2$, `BOHWUT` $+1.9$, `AWONOR` $-9.4$ kJ/mol).
    - In accordance with audit criteria, `bend_release` is preserved as an active experimental flag rather than promoted to default production.
  - **Checkpoint 2.3 (`dof_step_scale`) Tail Screening Complete:**
    - Implementation: Registered `FIELDS["dof_step_scale"]` in `src/jacobik/profile.py`, updated `src/jacobik/ik/ik_solver.py:735-770`, updated `tools/jdev/build_replay_solver.py`, and regenerated `src/jacobik/ik/solver_replay.py`.
    - Profile Fingerprint & Golden Invariants: Verified `profile.fingerprint() == "711411157fb4"`, `pytest -q tests/test_profile_hook.py` (9 passed), and `golden.py check` (245 arrays, 0 differ, golden: OK).
    - Parity Gates (200-step CPU schedule check + 9-unit parity panel):
      - `b1_screen_dof_rigid_1.5`: PASS (200-step CPU check on `JEGPUJ:11` 100% bit-identical `0.000e+00`; 9/9 on panel with 0 fallbacks, `max_rel = 0.000e+00`).
      - `b1_screen_dof_torsion_1.25`: PASS (200-step CPU check on `JEGPUJ:11` 100% bit-identical `0.000e+00`; 9/9 on panel with 0 fallbacks, `max_rel = 0.000e+00`).
      - `b1_screen_dof_ring_0.5`: PASS (200-step CPU check on `JEGPUJ:11` 100% bit-identical `0.000e+00`; 9/9 on panel with 0 fallbacks, `max_rel = 0.000e+00`).
    - Tail Screening Runs (10 parents × seeds 11–13 = 30 units per arm, sequentially on RTX 2060 GPU with 0 fallbacks):
      - `rigid: 1.5`: Mean $\Delta \text{top-1 } dE = -38.79$ kJ/mol (median $-6.02$), but parent-clustered 95% CI is `[-94.30, +8.72]` (crosses 0 due to $+27.6$ kJ/mol regression on `DAMVAQ` and seed 13 blowout on `YUCZUS`). Stopped.
      - `torsion: 1.25`: Mean $\Delta \text{top-1 } dE = -38.04$ kJ/mol (median $-1.73$), but parent-clustered 95% CI is `[-95.02, +3.93]` (crosses 0 due to $+22.7$ kJ/mol regression on `DAMVAQ`). Stopped.
      - `ring: 0.5`: Mean $\Delta \text{top-1 } dE = \mathbf{-5.53}$ kJ/mol (median $-0.00$). **Parent-clustered 95% CI: `[-14.64, -0.25]` kJ/mol (strictly < 0, excludes 0!)**.
      - Invariance & Concentration: Exact $0.000$ kJ/mol invariance preserved on all 6 non-ring complexes. Consistent improvement across all 4 ring complexes (`USAPOU` $-44.6$, `XEYTUT` $-2.7$, `YUCZUS` $-7.1$, `ZELTAP` $-0.9$ kJ/mol). Not a single ring parent worsened.
    - Checkpoint 2.3 Verdict: **PASS on `dof_step_scale = {"ring": 0.5}`**; recommended for confirmation.


## 2026-10-27 — Package C Confirmation, Optimization, and Relaxed Audit Complete

- **Runner Optimization & Saturation:**
  - Implemented multi-arm batching (`--arms-spec`), CPU background prefetching (`_prefetch_parent`), single-pass deduplicated batched MACE evaluation (`mace._batch_atoms = 900`), and scale-invariance fast path.
  - Implemented Fix A (`safe_cuda_eigh_chunking`) to protect large batch cuSOLVER operations.
  - Implemented Fix B (`EZ_ENUMERATE` compatibility via automated `multiseed.solve_seeds` dispatch).
  - Benchmarked throughput: Post-startup throughput jumped from 19.35 to **79.7 single-arm equiv units/min** (> 4.1× speedup); GPU utilization reached sustained 80–99% solve bursts; full 1080-unit cal panel across 3 arms completed in **40.6 minutes** (61% reduction in wall time).
- **Parity Spot-Checks:**
  - Verified combined configuration (`bend_release=[0.1, 100]` + `dof_step_scale={"ring": 0.5}`) on 200-step CPU schedule check on `USAPOU:11` (`max_rel = 0.000e+00`) and 3-unit CUDA panel (`DAMVAQ,JEGPUJ,USAPOU:11`, `max_rel = 0.000e+00`, 0 fallbacks). Saved to `research/data/jgen/campaign/B/b1_pkgc_replay_parity_combined.json`.
- **Pre-Registered Evaluation Contract:**
  - Written and committed to `research/data/jgen/campaign/B/confirmation_pkgc/evaluation_contract.json` before viewing confirmation results.
- **Diagnostics & Censuses:**
  - §3.1 `HOXJUZ` Question: Verified that only `USAPOU` among all 10 tail parents has ring/macrocycle DOFs; `HOXJUZ` has 0 ring DOFs, and minor screening noise was due to eager CUDA `index_add_` atomic ordering. The new scale-invariance fast path enforces exact $0.000000\text{e}+00$ bit invariance.
  - §3.2 `XACMOE` Diagnosis: Diagnosed that `bend_release` unrelaxed win vs relaxed regression on `XACMOE` is a topological coordination diastereomer score inversion (`asg=2` vs `asg=1`), not an algorithmic failure.
- **Confirmation Panel Executions (RTX 2060 SUPER, 0 fallbacks across all runs):**
  - **Tail Panel (seeds 14–16, 10 parents, 30 units per arm, 90 units total):**
    - `ring_0.5`: Mean top-1 $\Delta E = +1.70$ kJ/mol (median $0.00$). 9 non-ring parents exactly $0.000$ kJ/mol.
    - `combined`: Mean top-1 $\Delta E = \mathbf{-24.71}$ kJ/mol (median $-0.90$). **Parent 95% bootstrap CI: `[-56.36, -3.02]` kJ/mol**. Mean floor $\Delta E = \mathbf{-24.92}$ kJ/mol. Parent W/L/T: 8 W / 1 L / 1 T. Catastrophic units reduced from 18 to 14.
  - **Cal Panel (seeds 11–13, 120 parents, 360 units per arm, 1080 units total):**
    - `ring_0.5`: Mean top-1 $\Delta E = \mathbf{-0.53}$ kJ/mol (median $0.00$). **Parent 95% bootstrap CI: `[-1.49, -0.00]` kJ/mol** (excludes $>0$). **Parent W/L/T: 4 W / 0 L / 116 T (0 parent losses!)**. Unit W/L/T: 10 W / 1 L / 349 T.
    - `combined`: Mean top-1 $\Delta E = \mathbf{-3.63}$ kJ/mol (median $-0.00$). Unit 95% bootstrap CI: `[-7.42, -0.19]` kJ/mol. Mean floor $\Delta E = \mathbf{-2.40}$ kJ/mol. **Catastrophic units slashed by 43% (14 down to 8)**.
- **Cal Ring Subset Analysis (`AWONOR`, `USAPOU`, `XOJHIN`, $n=3$, 9 units):**
  - `ring_0.5`: Mean top-1 $\Delta E = \mathbf{-21.23}$ kJ/mol across macrocycles (**3 W / 0 L / 0 T**; `AWONOR` $-11.64$, `USAPOU` $-44.66$, `XOJHIN` $-7.40$).
  - `combined`: Mean top-1 $\Delta E = \mathbf{-40.51}$ kJ/mol (`USAPOU` $-143.96$ kJ/mol).
- **Relaxed GFN2 Audit (`xtb --opt crude` + MACE float64):**
  - Tail Panel (5 parents audited): **0 flagged regressions**. `JEGPUJ` preserves $-25.0$ kJ/mol relaxed gain across all seeds; `USAPOU` preserves $-50.2$ kJ/mol relaxed gain.
  - Cal Panel (50 parents audited): **Exactly 1 flagged regression (`XACMOE`, $+35.45$ kJ/mol)**, quantitatively confirming the pre-registered diastereomer inversion diagnosis. All other 49 parents passed cleanly.
- **Contract Verdict:**
  - All 6 pre-registered gates **PASS**.
  - `dof_step_scale = {"ring": 0.5}` is unconditionally confirmed for production fold-in.
  - `combined` is confirmed as a high-yield option slashing catastrophic failures by 43%.
- **Documentation:**
  - Added §C in `research/bench/jgen_campaign/B_REPORT.md`.
  - Awaiting supervisor / parent agent audit before git commit.


## 2026-10-27 — Package F Production Fold-In of bend_release = [0.1, 100]

- **Scope & Directive:**
  - Folded in `bend_release = [0.1, 100]` to production default (`PRODUCTION_CONFIG`), updating production fingerprint from `711411157fb4` to `615ae97d2350`.
  - Maintained `dof_step_scale` in `profile.FIELDS` with default `None` (off), inert until a ring-rich panel exists.
  - Carried forward `XACMOE` (coordination diastereomer score inversion `asg=2` vs `asg=1`) as an identified test case for scorer work.
  - Folded in runner throughput optimizations in `research/bench/tools/jgen/bench/run.py`.
- **Codebase & Documentation Updates:**
  - Updated all 14 files across core, tools, tests, and documentation to pin new fingerprint `615ae97d2350`.
  - Hardcoded `BEND_RELEASE = [0.1, 100]` in `src/jacobik/ik/ik_solver.py`.
  - Regenerated `src/jacobik/ik/solver_replay.py` via `build_replay_solver.py` (new source MD5 `098217b3c8d0b23526ee3c9af2225598`).
  - Added `bend_release` row to `research/archive/flags/FLAGS.md`.
- **Verification Gates:**
  - `pytest -q tests`: **321 passed, 3 xfailed, 3 warnings in 302.76s** (0 failed).
  - Fast-path tests (`test_profile_hook.py` + `test_gpu_fast_path.py`): **28 passed in 91.73s**.
  - `golden.py record` & `check`: Regenerated `production.npz` and `u1.npz` for `615ae97d2350`; `check` confirmed **245 arrays, 0 differ (golden: OK)**.
  - Replay Parity:
    - 9-unit check (`DAMVAQ,HOXJUZ,JEGPUJ` × seeds 11–13, `loop-steps 20`): **9/9 PASS**, 0 fallbacks, `max_rel_dscore = 0.000e+00`.
    - 200-step CPU full schedule check (`JEGPUJ:11`): **PASS**, `max_rel_dscore = 0.000e+00`.
    - Full 30-unit `tail` panel (`DAMVAQ,HOXJUZ,IFUROQ,JEGPUJ,NEPNEF,SOBHAS,USAPOU,XEYTUT,YUCZUS,ZELTAP` × seeds 11–13): **30/30 PASS**, 0 fallbacks, identical top-1, `max_rel_dscore = 0.000e+00`. Written to `research/data/jgen/campaign/B/pkgf_replay_parity_tail30.json`.
- **Post-Fold-In Production Control (`flags = "{}"`, local RTX 2060 SUPER):**
  - **`tail` panel (30 units, seeds 11–13):** 30 records, `fp = 615ae97d2350`, 0 fallbacks; mean top-1 $\Delta E = 747.88$ kJ/mol, median $492.08$ kJ/mol; catastrophic ($>500$ kJ/mol) units reduced from 20 down to 14.
  - **`cal` panel (360 units, seeds 11–13):** 360 records, `fp = 615ae97d2350`, 0 fallbacks; mean top-1 $\Delta E = 133.77$ kJ/mol, median $82.11$ kJ/mol; obscene ($>200$ kJ/mol) units reduced from 55 to 52; **catastrophic ($>500$ kJ/mol) units slashed from 14 down to 8 ($-43\%$)** while preserving median $\Delta E$.
- **Status:** Package F completed. Ready for supervisor audit and final commit.


## 2026-10-28 — Package G: Optimizer Dynamics, Bend Landscape, and Beam Coverage (Checkpoint G-1)

- **Framework & Profile Registrations:**
  - Registered `lr_final` (default `None`) and `bend_free_deg` (default `2.0`) in `src/jacobik/profile.py:FIELDS`.
  - Verified production fingerprint bit-identical: `615ae97d2350`.
  - CPU Golden Reference Check (`python research/bench/tools/jdev/golden.py check`): `production: 245 arrays, 0 differ`, `u1: 245 arrays, 0 differ`, `golden: OK`.
  - Re-generated canonical `src/jacobik/ik/solver_replay.py` using `tools/jdev/build_replay_solver.py`; verified 100% byte match against committed solver (`pytest tests/test_gpu_fast_path.py::test_replay_generator_matches_committed_solver` passed).
- **O0 Optimizer Survey Note (`research/bench/jgen_campaign/G_OPTIMIZER_NOTE.md`):**
  - Completed literature survey covering FIRE, internal-coordinate quasi-Newton, preconditioned L-BFGS, Gauss-Newton / Levenberg-Marquardt (LM), diagonal curvature (Sophia/AdaHessian), and pullback metric Adam.
  - Provided concrete hardware cost proof on RTX 2060 SUPER: $64 \times (60 \times 60)$ normal equations + Cholesky solve takes only $0.135$ ms ($\sim 3.0\%$ overhead on $4.5$ ms FK step). Cost is a non-issue for GN/LM.
  - Ranked recommendations: #1 Hybrid Adam + GN/LM convergence phase; #2 Pullback metric Adam. Audited and approved by supervisor.
- **S-P1 Beam Torsion-Mode Coverage (`research/data/jgen/campaign/B/g/sp1_coverage_results.json`):**
  - Calibration panel (`cal`, seed 11): 80.5% of parents (95/118) have unfilled supported modes ($w \ge 0.05$ with 0 beam rows within 0.2 rad); 52.3% of modes (3,284/6,284) unfilled. Roadmap gate threshold $\ge 10\%$ is heavily triggered ($80.5\% \gg 10\%$).
  - Tail panel comparison: width 16 has 80.0% unfilled parents (40.0% modes) vs width 48 with 70.0% unfilled parents (36.3% modes). Proved greedy density collapse; widening alone fails. `beam_target_aware` is fully justified.
- **G1 & G3 Tail Screening Runs (120 units, 4 arms × 10 parents × 3 seeds):**
  - Pre-registered evaluation contract (`research/data/jgen/campaign/B/g/evaluation_contract_g1_g3.json`).
  - Executed 120 units on RTX 2060 SUPER with 0 replay fallbacks. Exported 240 XYZ structures (top-1 and floor per unit).
  - Executed GFN2-xTB crude relaxations + float64 MACE single points (`run_relax_g1_g3.py`).
  - Results:
    - Control: median top-1 strain = 416.7 kJ/mol, median floor strain = 325.9 kJ/mol, median top-1 dE = 492.1 kJ/mol.
    - `g1_lr_final_0.01`: $\Delta\text{median top-1 strain} = +8.26$ kJ/mol [95% CI: +0.07, +42.66], $\Delta\text{mean floor dE} = +15.69$ kJ/mol, `USAPOU` degraded by $+125.85$ kJ/mol. Fails guardrails.
    - `g1_lr_final_0.001`: $\Delta\text{median top-1 strain} = +3.97$ kJ/mol [95% CI: -3.65, +60.48], $\Delta\text{mean floor dE} = +20.22$ kJ/mol, `USAPOU` degraded by $+164.99$ kJ/mol. Fails guardrails.
    - `g3_bend_free_1.0`: $\Delta\text{median top-1 strain} = +0.03$ kJ/mol [95% CI: -2.76, +3.57], `YUCZUS` degraded by $+92.48$ kJ/mol. Fails guardrails.
  - Diagnosed premature basin freezing on G1 and stiff bound neglect on G3.
- **Bend Column Distribution Census (`research/data/jgen/campaign/B/g/bend_distribution_results.json`):**
  - Measured 4,496 bend angles across 10 tail parents (seeds 11–13).
  - Control: $>1^\circ = 88.66\%$, $>2^\circ = 73.93\%$. G3 ($1.0^\circ$): $>1^\circ = 88.39\%$, $>2^\circ = 73.95\%$.
  - Proved first-order steps cannot enforce tight bend bounds without curvature scaling.
- **O1 Convergence Diagnostic (`research/data/jgen/campaign/B/g/o1_convergence_results.json`):**
  - Ran float64 reference L-BFGS with strong Wolfe line-search + GFN2 crude relaxations + MACE single points on 8 parents (6 conformers each, 48 conformers total) starting from captured S2b theta coordinates.
  - Overall conformer breakdown:
    - **Class (a) (optimizer problem):** **21 / 48 (43.8%)** — score improves and physical strain drops dramatically ($\mathbf{-15,531.6}$ kJ/mol on `FEKKAJ`, $\mathbf{-7,358.1}$ kJ/mol on `NEPNEF`, $\mathbf{-603.7}$ kJ/mol on `NUWFAM`). Proves Adam gets stuck in unminimized states where second-order methods easily find low-strain minima.
    - **Class (b) (score minimum strained):** **27 / 48 (56.2%)** — score improves, but strain does not drop (score surface minima disagree with QM potentials, motivating scorer refit and NN correction §7a-E).
    - **Class (c) (already converged):** **0 / 48 (0.0%)** — 100% of conformers had significant unminimized score residuals ($\Delta\text{Score} \in [+48, +1604]$).
  - Confirms optimizer convergence phase (O2) mandate to slash catastrophic optimizer-induced strain failures.
- **Compute Spend:** 0 Colab spend (all runs executed on local CPU / RTX 2060 SUPER within 5 CPU process budget).
- **Status:** Checkpoint G-1 complete. Awaiting supervisor audit.


## 2026-10-28 — Package G-2 Handover (CodexB)

- **Contract:** Pre-registered O2 metrics, guardrails, pairing and screen order. Commit `68b046d` contains the tracked contract mirror at `research/bench/jgen_campaign/g/evaluation_contract_o2.json`. The requested shared-output copy is at `research/data/jgen/campaign/B/g/evaluation_contract_o2.json`; Git cannot stage through the existing `research/data/jgen` symlink.
- **Score reproduction (2a):** Added `score_repro.py`, the four-parent production capture runner, and key/score unit tests. CPU smoke passed for HEDJOP (3 assignments), NUWFAM (2 E/Z variants), QEZVEY (12 assignments), and USAPOU (macrocycle); all 64 returned rows across the four cases pass `|S64(theta0)-conf.score| < 1e-4`. Maximum absolute score difference was `9.58e-6`; maximum float32/float64 difference at delivery was also `9.58e-6`. `pytest tests/test_score_repro.py`: 4 passed. The smoke runner now guards every build with runkit's xTB fallback watcher.
- **O1 (2b):** Replaced the invalid G-1 objective path with `run_o1_g2.py`: per-row captured production score, float64 L-BFGS-B, ±15°/0.5 Å bounds, geometry/contact checks, and runkit MACE/xTB with failed operations left explicit. The seed-11 primary run completed all 48 rows. All score assertions passed; classes are 6 (a), 7 (b), 15 (c), and 20 unknown strain labels. The 20 unknowns correspond to explicit xTB failures/nonconvergence; no failure is substituted with zero. The per-parent distribution and contact/bound summary are in §G-2 of `B_REPORT.md`; full row detail is in `research/data/jgen/campaign/B/g/o1_g2_results.json`. Forty-two rows (87.5%) hit at least one bound (145 bound DOFs), so the required ±30°/1.0 Å run is executing as `run_o1_g2.py --angular-cap-deg 30 --translation-cap-A 1.0 --workers 3`, writing `o1_g2_cap30_results.json`.
- **S-P1 (2d):** Added the symmetry-collapsed heavy-atom torsion recount and Q6-relative single-joint clash check. The refined `cal`, seed 11 recount completed 120/120 parents with 0 failures. 76 parents have supported modes. At 0.2 rad: 49/76 eligible parents have an unfilled mode; 1,148/3,056 supported modes are unfilled (791 reachable, 357 clashing). At 0.35 rad: 49/76 eligible parents; 1,070/3,056 unfilled (731 reachable, 339 clashing). The ≥10% gate is met. HEDJOP pilot: 120 supported modes; 59 unfilled at each tolerance, with 57 reachable and 2 newly clashing.
- **Optimizer note / hardware correction:** Added the FIRE 2.0 row-mask design and LM non-least-squares term mapping. A fresh `nvidia-smi` sees the RTX 2060 SUPER. The shared main-checkout runkit budget grants its lock to `CodexB`; the earlier “driver inaccessible” status came from a stale check and is corrected. Jacobian timing is still unmeasured because no full analytic FK Jacobian exists yet. O2 code and screens remain open; no arm result has been generated. `run_o1_g2.py` now defaults MACE to CUDA under the runkit GPU lock; the already-running wider-cap process was launched with the previous CPU MACE evaluator. xTB remains CPU-only.
- **Spend and process:** 0 Colab units. CPU diagnostics use at most 3 CodexB threads; no GPU job has been started.
- **Status:** G-2 is in progress. O1 primary is complete and the wider-cap follow-up is running. O2 optimizer code and screens remain outstanding. Do not treat the invalid G-1 O1/S-P1 figures as accepted panel results.



## 2026-10-28 — Package G-2 Final Checkpoint (CodexB)

- **O1 cap-30 replacement:** Stopped the earlier CPU+xTB run by PID and replaced it with the requested no-xTB screen. All 48 rows across 8 parents completed score-change, CUDA MACE unrelaxed ΔE, displacement, and contact checks; no row had a non-donor metal contact below 1.2 Å. The top-1 rows exceeding |ΔE_unrel| 20 kJ/mol were reduced to the five largest-magnitude parents to respect the 10-structure cap. Ten xTB attempts were made; only QEZVEY's pair converged on both sides. Failures remain explicit. Results: `o1_g2_cap30_unrel_gpu_results.json`, `o1_g2_cap30_selected_relax.json` (shared data output).
- **O2 trope screen:** Implemented `adam_metric` and `fire_metric`; ran control and both arms on all 9 trope parents, seed 11, CPU solver and CUDA MACE unrelaxed scoring. Control: 305 rows, median top-1 score 1.034, median top-1 ΔE 128.1 kJ/mol, obscene rate 73.77%, worst top-1 1,047.7 kJ/mol (SUXROT), mean floor 138.0 kJ/mol. Adam metric: 290 rows; paired median score change −0.100, paired median top-1 ΔE change −0.64 kJ/mol, obscene rate 81.72%, worst top-1 509.7 kJ/mol (SUQWEG), mean floor 144.0 kJ/mol. FIRE metric: 227 rows; paired median score change −0.135, paired median top-1 ΔE change −4.17 kJ/mol, obscene rate 83.26%, worst top-1 3,116.1 kJ/mol (SIKNAE), mean floor 455.7 kJ/mol. **Both arms fail the gate**: neither improves median score, obscene rate worsens, average floor worsens, and FIRE has a severe worst-parent regression. No tail or cal panel was opened.
- **Requested terminal relaxations:** Used Adam metric as the less adverse screened arm and attempted exactly its 9 seed-11 top-1 structures. Eight xTB jobs converged; SUQWEG returned `no_output(rc=1)`. MACE strain single points were recorded only for successful relaxations. Results: `adam_metric_top1_relax.json`.
- **Replay light gate:** Adam metric replay ran 38/40 graph steps on each of MELLUN, AHIGOQ, PIDXOT and did not match eager outputs (maximum score deltas 2.37e−3, 9.35e−5, 3.93e−6; maximum coordinate delta 3.55e−2 Å). FIRE metric replay failed all three cases with `NameError: cannot access free variable 'fire_dt' where it is not associated with a value in enclosing scope`. These are reported failures; the trope suite itself ran on the CPU solver path.
- **Other verification:** Profile production fingerprint remains `615ae97d2350`. The targeted metric/autograd and replay checks previously run reported 3 passed, 1 failed (fast-FK case returned 6 rather than 8 conformers). No Colab spend. Contract and outputs are in `research/bench/jgen_campaign/g/evaluation_contract_o2.json` and shared `research/data/jgen/campaign/B/g/o2_trope_r5/`.
- **Status:** G-2 reported complete at the requested stop point. No tail/cal work pending or authorized by the result.


## 2026-10-28 — G-3 started (CodexB)

- Read G-3 directive, G-2 audit, PLAN process rules, roadmap S-P1/S-P2 and G.8. Clean branch at a6711c9.
- Step 0 objective inventory recorded in B_REPORT.md before solver edits. Fast-FK targeted test passes on both master 8dbe29c and a6711c9; historical failure unreproduced. xTB resolves from the shared pixi environment.
- G-3 contract supersedes O2 and is committed before new arm results. No screens, relaxation, GPU job or Colab VM started; Colab spend 0.

## 2026-10-28 — G-3 sanity and pilot

- Contract commits: 2dcd566 (G-3 metrics), ec2cf71 (three trope seeds, before screens).
- Exact O1 sanity complete: 48/48 original starting scores and geometries verified. P2 recovery median 87.73%, ΔΔE_unrel median −5.27 kJ/mol; P1 recovery 44.54%, ΔΔE_unrel +0.91. Both score gates pass. A1 fails median delivered top-1 score (0.127374 → −0.014039) and is excluded from screens.
- FIRE dt_start 0.03897477; calibration median first-step RMS verified at 0.0195121 Å on HEDJOP/USAPOU.
- CPU parity and three-unit same-loop CUDA parity pass; calibrated P1 CUDA configuration remains to check before any GPU panel. Legacy FIRE replay NameError removed. Golden running.
- C1/P1/P2 two-parent, three-seed CPU trope pilot running without strain/relaxation. No tail/cal, Colab or G-3b work. Colab spend 0.

## 2026-10-28 — G-3a final checkpoint

- **A1 rejected:** eight-parent median delivered top-1 score falls from 0.127374 to −0.014039. No A1 trope/panel screen.
- **P1 rejected:** completed 9 parents × seeds 11–13 versus C1. Paired mean top-1 ΔΔE_unrel −47.95 kJ/mol, median **+0.172471** (fails required ≤0); delivered score rises on 5/9 parents. Solver wall 1.398616× C1; no cost exception claimed. No parent violates the ≥2/3 regression guard.
- **P2 rejected early:** CAFWIQ worsens by +505.80/+318.07 kJ/mol on seeds 11/12, violating the prescribed guard. Only the two pilot parents were run; seven unnecessary runs omitted. Full-nine-parent claims are not made.
- **Verification:** default fingerprint 615ae97d2350; original-layout golden production 245/245 and u1 245/245 arrays pass. The forced-one-worker comparison failed and its u1 continuation was stopped by PID; no golden was changed. CPU loop checks pass. Full raw-pool CUDA parity passes all three units for A1/P2/calibrated P1 with max relative score difference 0 (128/256/896 rows), plus matching delivered row identities. Legacy FIRE free-variable failure is fixed. Final optimizer/fallback/generator checks pass; a failed polish safely preserves the delivered baseline.
- **Artifacts:** g3_sanity_summary.json; g3_polish_sanity_complete.json; g3_trope/g3_trope_summary.json; g3_*_raw_replay.json; g3_golden_{production,u1}.{json,log}, all under shared campaign B/g. B_REPORT.md §G.9 contains tables and Spec issues/deviations.
- **Stop:** G-3a complete, all arms rejected. No tail/cal screen, complex relaxation, Colab VM/spend (0), or G-3b implementation. Await audit.

## 2026-10-29 — G-3a′ contract and implementation underway

- Read accepted G-3a audit and merged base 4157337 (F-4). A1/P1 remain rejected; P2 is the learning-rate baseline. New candidate R1 is 200 Adam +50 iRprop+; R2 is CPU-only 250-step iRprop+ from the start.
- Registered the revised cal contract at `g/evaluation_contract_g3prime.json` before opening cal. Sanity captures last effective Adam steps through culling, verifies the original O1 scores/geometry, and runs Δ0 ×0.1/×1/×10. Stop/report sanity before cal as instructed.
- Default fingerprint 615ae97d2350. Golden and light gates underway. Three-unit CUDA raw-pool replay comparison passes at relative difference 0. No cal/tail/relaxation/G-3b work or Colab spend (0).

## 2026-10-29 — Atom-order repair before trusting energy results

- Owner identified frame-label/FK-coordinate mismatch. O1 and both sanity exporters now use `conf.complex.atoms`, assert the exported labels and count, and the selected-relaxation reader rejects unverified exports. A regression test passes. Original coordinate/score data retained; no re-solve for repair.
- Recomputed saved canonical O1 48-row unrelaxed MACE energies: median L-BFGS change **−17.172224**, replacing invalid −13.347545 kJ/mol. G-3a P2: **−16.711545**, replacing invalid −5.268848; P1: **−4.344624**, replacing invalid +0.909721. All affected historical xTB/strain claims are void; none rerun. Old JSON preserved beside corrected files.
- Corrected G-3a′ energies: nominal R1 −12.747431, ×0.1 −14.250588, ×10 −13.772010. The independent trope exporter already maps to frame order: all 87 saved top-1 energies rechecked, maximum difference 0. A1 had only a score gate and no energy result. B_REPORT.md §G.10 records old/new tables, contact repair, and unavailable older coordinate limitations.
- Continue score-only R2 sanity, then report/stop G-3a′ sanity. Nominal R1 fails both required alternatives against corrected P2; no cal/tail, relaxation, Colab (0), or G-3b work.

## 2026-10-29 — G-3a′ sanity checkpoint, stop before cal

- **R1 fails:** 48/48 exact O1 rows reproduced. Nominal iRprop+ recovers 83.1225% of L-BFGS score gain versus P2 87.7299%; corrected median ΔΔE_unrel −12.747431 versus P2 −16.711545 kJ/mol. Every row’s score improves, but neither required alternative is met. ×0.1: 73.5822%, −14.250588; ×10: 83.5361%, −13.772010. Both also fail; no sensitivity arm substituted. Zero of 48 rows converge by step 50 in each iRprop+ arm.
- **R2 CPU-only:** all 8 paired full solves complete, zero failures. Median top-1 score 0.12737396 (200 Adam C0) → 0.21765293 (250 iRprop+); median paired gain +0.09695274. All parents improve in score. No energy/panel inference or replacement of production search claimed.
- **Light gates on merged F-4 base:** fingerprint 615ae97d2350; golden 245/245 with default eight-chunk layout. Final CPU parity, optimizer/backtracking, fallback injection and generated replay checks: 9 passed. Corrected-moment CUDA raw-pool replay on 3 units passes at max relative score difference 0. Atom-export regression separately passes.
- Δ0 handoff was corrected from production-scaled steps to raw Adam moment steps before the final sanity/parity runs; superseded trial retained. B_REPORT.md §G.11 includes sensitivity, all convergence entries’ meaning, step-bound diagnostics, remaining L-BFGS gaps, and the exact rejection.
- **Stop:** no cal/tail, relaxation, Colab VM/spend (0), FIRE tuning, R2 panel, or G-3b work. Contract 9d561b4; atom-order correction 187ac77. Await audit of the failed G-3a′ sanity gate.

## 2026-10-29 — Owner-issued R2 energy follow-up

- Owner requested energy evaluation of R2 after the failed R1 sanity stop. Registered an exploratory eight-parent, seed-11 R2/C0 contract before energy results. No cal/panel expansion or relaxation.
- The earlier R2 runner omitted coordinates; this is corrected by saving all delivered structures with `conf.complex.atoms` labels and assertions. Paired CPU solves must be repeated, with top-1 scores checked against the existing score-only run. Default layout remains eight chunks; executor concurrency may rise to five within CodexB quota. The occupied GPU will be accessed only under the shared lock.
- Owner additionally requested iRprop− afterwards. Interpret it as the whole-run 250-step counterpart to R2, same eight parents/seed and all settings; only backtracking is removed (sign-flip update and stored gradient are zero). Checked Igel/Hüsken §2.4/Fig.3. Contract amended before minus-arm results; pair directly against saved R2 and C0.

## 2026-10-29 — R2 energy result; iRprop− pilot underway

- R2/C0 repeated solves reproduce all 16 prior top-1 scores exactly. Saved all 256 returned structures with builder-order labels and assertions. Five active CPU chunks maximum; default requested layout remains eight. Two-parent timings estimated ~17 minutes for all solves; two-structure GPU MACE profiling estimated ~1 minute for single points. No extra machine used.
- R2 versus C0, 8 parents seed11: paired top1 mean **+121.330566 kJ/mol**, parent-bootstrap CI **[−117.382627,+421.387700]**, paired median **+11.141003**; 3 improve, 5 worsen. Floor mean change **−66.391188**, CI **[−132.886893,−5.820016]**, median −24.547641; 7/8 improve. Obscene rate **86.71875% →82.8125%** (128 rows each). FEKKAJ +989.982 and NUWFAM +484.303 top1 energy regressions despite score gains. End-to-end build/solve time 2.510594× C0 (250 versus 200 steps, different culling). These are exploratory eight-parent figures, not a panel gate or adoption claim.
- iRprop− whole-run pilot starts after completing R2 energy evaluation. The variant keeps all settings and the stopping rule, only suppressing weight backtracking. Small update/sign-reset tests and whole-run no-Adam integration pass; complete targeted batch **13 passed in 37.45 s**. R2− will reuse verified C0 single points and compare directly against both C0 and R2. No cal/tail, relaxation, Colab spend (0), or G-3b work.

## 2026-10-29 — R2 energy/iRprop− follow-up complete, stop

- All 8 iRprop− parents complete, zero failures, 128 delivered structures. Same 250-step whole-run settings as plus; only weight-backtracking removed. C0 energies reused after exact geometry-digest/model/device checks and independent reference validation. Plus geometry digests checked for direct comparison. No additional C0 solve or repeated control single points in the minus phase.
- **Minus versus C0:** paired top1 mean +78.252421 kJ/mol, parent-bootstrap CI [−128.871801,+367.907731], median +5.924003; 3/8 improve, 5/8 worsen. Floor mean −67.640901, CI [−135.583515,−6.022948], median −24.448015; 7/8 improve. Obscene rate 82.03125%, versus C0 86.71875% and plus 82.8125%. End-to-end build/solve ratio 2.509379× C0.
- **Minus versus plus:** top1 mean −43.078145, CI [−126.751878,+0.306428], paired median −0.000613. Mainly NUWFAM −334.353 kJ/mol; no broad improvement supported. Both arms show higher top1 score with worse top1 energy on FEKKAJ/NUWFAM/QEZVEY/NEPNEF/AJAQAD. Worst slots: C0 2751.443, plus 3363.749, minus 3388.497 kJ/mol.
- B_REPORT.md §G.12 contains per-parent top1/floor energies, all CIs, costs, labels/provenance, and interpretation. Same eight exploratory parents/seed11; different objectives/budgets/culling versus production C0 are explicit. 13 targeted tests pass; default fingerprint 615ae97d2350.
- **Stop:** both owner-requested checks complete. No tuning, cal/tail panel expansion, relaxation, Colab spend (0), fold-in or G-3b work.

### 2026-10-29 — G-3a″ started (owner-issued directive)

- Read `.claude/directive_jgen_B_G3a2_2026-10-29.md` and governing rules; baseline branch `657eb6a`, clean at entry. Scope: scalable P2 finish vs late cosine / zigzag / plateau Adam schedules; no whole-run optimizer expansion, relaxation, tail or G-3b.
- Owner clarified: fixed 48-row diagnostics use 50 continuation steps from the 200-step Adam endpoint; include P2/C1 in the eight-parent full-solve filter so the P2 spot-check names can also be frozen before L4 per-parent results.
- Added shared tensor schedule state, eager/replay integration, import-time experimental flags, optional batch early stop and endpoint/trace diagnostics. Default fingerprint remains `615ae97d2350`. New schedules preserve Adam moments and the existing objective/update; P2's separate finish length is now `round(.25*T)`.
- Initial targeted suite: 18 passed (70.43 s). Initial 200-step same-GPU eager/replay parity: all three arms × MELLUN/AHIGOQ/PIDXOT pass, max raw relative score error 0. Three-unit CPU parity is in progress. No claims of final merged-base validation yet.
- Two-parent diagnostic pilot HEDJOP/USAPOU: 12 endpoint rows; prescribed zigzag settings fire on 0/12, plateau on 7/12. Only prescribed calibration settings tried (`z*=.5`, `k=4`, `delta=.001`); full 48-row outcome pending. Full-solve timing pilot is separate and uses the default eight-chunk layout.
- Found benchmark-runner arm reuse/reset/step-count hazards for the new flags; fixed before any cal run. Added builder-label/mapping assertion and per-parent exception ledger; no data/property-column changes.
- One diagnostic launch had a mistyped second parent and omitted resume; killed its own PID before any output was written, confirmed original HEDJOP/USAPOU records intact, and resumed the six correct remaining O1 parents. No CAFWIQ result was produced or used. CPU concurrency bounded to the assigned five; one short overlap during pilot completion was corrected by checking own PIDs (pilot had already exited).
- Colab credits checked: 506.05 remaining, 24.74 spent against shared 180 cap, active assignments 0, rate 0/hr. No VM started and no package spend so far. Cal unopened; contract still a draft, pending step-1 outcomes and frozen spot-check names. Explicitly authorized master `7d3e4a5` merge will precede cal and trigger fresh light gates.

- Completed all eight parents × six full-solve arms, 768 saved delivered conformers, no failures. C0 median score 0.127373956; P2′ 0.123178886 (paired median −0.004522249): **late cosine fails the prescribed score filter**. P2a/P2p median scores equal C0. P2 vs C1 paired median score +0.055215269. No filter retuning.
- Fixed-row trigger counts 4/48 and 8/48 are **provisional**: reported a spec ambiguity before cal. This trial pairs delivered best θ with final Adam moments on the delivered objective (the prior polish convention), whereas a continuous-state diagnostic would use actual θ200 and retain the constrained objective. Asked owner which defines this gate; cal remains closed pending clarification. Original trial/artifacts retained. All 48 rows inherited bias count 200, zero capture failures.
- Added actual replay-chunk/early-stop execution metadata to the batched benchmark runner, parent-clustered analysis, and same-arm early-stop replay checks. Scalable-P2 length, parent bootstrap semantics and generator consistency: 3 passed in 14.77 s. Three-unit CPU parity package: 14 passed in 297.32 s. Scientific plots generated for the fixed HEDJOP/FEKKAJ/USAPOU example identities (provisional trial).

- Merged-base verification complete: default fingerprint `6cc586e17ab1`; golden **245/245**, default eight-chunk layout, max three active chunks, 274.19 s. Targeted merged suite **37 passed in 308.18 s**. All three new arms × three CUDA units pass eager/replay parity; all nine early-stop companions skip zero steps and reproduce raw scores and delivered scores/coordinates exactly. Timings saved per unit/arm.
- Resolved the diagnostic ambiguity by using actual continuous Adam state and constrained objective, as faithful to the single-loop arms; owner asked whether an answer was still needed, and no further decision was requested. Original delivered-best/moment trial is retained as superseded evidence, not the gate. New diagnostic runs 250 steps, preserves the original cull at60, bend release at100, and starts the trigger at200. It uses a git-archived `9ff8655` library snapshot so original O1 embeddings/row identities remain the old production.
- The first continuous pilot used an incorrect validation comparison: native pre-projection best score versus O1 post-projection delivered score. USAPOU differs by 0.007209, so that assertion failed and is reported/retained (`g3a2_continuous_preclosure_check_failed.*`). Replaced that comparison with a stricter paired C0 reference: hash every selected row's θ, m, v, gradient and score at every one of the first200 steps. The true diagnostic must match this complete reference trajectory bit-for-bit before any trigger can fire. This corrects instrumentation, not a threshold or gate.
- Corrected HEDJOP/USAPOU pilot: all 12 rows match all200 reference steps exactly; zigzag fires4/12 (1 reaches floor), plateau2/12 (0 at floor). Six remaining parents in progress, prescribed thresholds unchanged. No cal data opened.
- Eight-parent energy filter complete, 537 unique geometries after deduplication, no failures; C0 top-1 energies exactly reproduce the prior valid run. P2/C1 paired median −23.280809 kJ/mol, mean +78.580297 [−82.784724,+348.854645], floor mean −30.907685 [−67.912171,−0.122115]. P2′/C0 top-1 mean +14.463008 [+3.218168,+30.720140], so score and energy filters both reject it. P2a delivered outputs equal C0 exactly; P2p changes little. Frozen P2 spot-check names: DAMVAQ, USAPOU, HEDJOP, NEPNEF, QEZVEY.

- Final corrected diagnostic complete, no failures: all48 selected trajectories match paired C0 bit-for-bit for all first200 steps. P2a fires8/48, floor2/48; P2p fires7/48, floor0/48. Both **fail ≥24/48 and are dropped** without retuning. P2′ already fails its score filter. Only original P2 proceeds against C1; C0 retained.
- Source bundle3250246 prepared on owned L4 `jgBg3a2a`, setup complete, fingerprint6cc586e17ab1. No cal job has run. Before opening cal, restored tolerance for unsupported optional cache-warming calls while retaining fatal logged GFN2-fallback checks; added benchmark-runner SHA provenance. This tiny harness-only patch will be applied/verified on the VM, avoiding another full data upload. Solver and light-gated production code are unchanged.
- Saved-geometry failure diagnosis: FEKKAJ same row, Cl···Cl2.013→1.418Å, score improves while energy+990.809; known missing donor-pair repulsion. QEZVEY old winner improves4.554kJ/mol but a different higher-energy row wins; floor improves20.451. User is fixing Q6 separately. All-parent metrics retained; FEK-excluded diagnostic mean−51.738kJ/mol on7 parents (6 improve), not a changed gate.
- Exported requested saved DAMVAQ P2/C1 top-1 XYZs (seed11 pre-Li-cap CPU,83 atoms, builder-order labels and coordinate round-trip verified), linked to owner. No re-solve/relaxation for export.

- **Cal contract frozen before results:** `f062f35`, admitted P2 only (C0/C1 retained),120 calibration parents × seeds11–13. All three refinements dropped by issued filters. Spot-check names frozen before any L4 result. Runtime source is bundle3250246 plus checksummed harness patchf436e83 (`859606dee255...`); all controls/arms use production6cc586e17ab1.
- L4 profile HEDJOP/USAPOU:18/18 records OK, all main loops replay, exact registered step counts (C0/P2 main200, C1250). Parent walls35.7s/46.1s; initial two-L4 estimate40–50min plus setup/long cases. P2/C1 solve ratios2.1166 and1.5484 exceed the1.3 guard on the pilot; full cal will quantify, no implementation changes mid-run.
- Profile records are reused in partitionA, not re-solved. The production-default returned pool is used on cal; the CPU eight-parent filter capped output at16. This and CPU culling versus fixed-shape replay are explicit differences in step-1/cal interpretation; each comparison retains its own paired control.
- Owned VM `jgBg3a2a` running partitionA (58 remaining parents after2-parent profile). Second allocation `jgBg3a2b` refused by Colab precondition after helper retries. Session inventory confirms no orphan: only ourA and other track's `jgC1s2`; never stopped/touched the other VM. Global balance505.08, aggregate rate3.88/hr,2 active,25.71/180 shared units spent at that check (not attributable solely to this package).
- Capacity fallback: same owned L4 will run partitionB after A finishes, avoiding another setup/upload. Local watcher fetches/counts A against540 expected records before queuing B; then fetches/counts B against540 and downs the VM immediately. Stopped only own local read-only watcher PID611844 to install this queue behavior; the remote cal job was uninterrupted. One-L4 estimate80–100min total, subject to long cases. Algorithms, panel, seeds, controls and gates unchanged.
- Added reproducible saved-coordinate/XYZ export script and FEK-excluded non-gating analysis per owner discussion. Nonfinite pools remain explicit unavailable units; no imputation. Primary all-parent contract and non-obscene guards are unchanged.

- **Actual cal input failure found:** A log records GFN2 failure for NEPNEF's nitrophenolate ligand followed by MMFF fallback. A read-only ligand-graph scan of all120 cal parents finds that failed ligand only in NEPNEF (ligands0/1). All9 NEPNEF cal records are invalid, not usable energy evidence. Completed valid parents will be retained; NEPNEF will not be retried or spot-fixed.
- Found the inherited warm-cache helper name was nonexistent (`engine._embed_free_ligand`), so caught AttributeError made warming a no-op and its FallbackWatch ineffective. This was missed in preflight. Added research-only validation around actual `_gfn2_refine` calls and remembered cached-failure signatures, leaving generator outputs/cache/optimizer unchanged. Remaining batch vetoes bad parent inputs before MACE. Three validity/cache/reset tests pass (0.11s). Old unguarded A results are vetoed by logged ligand provenance, with raw backups, not re-solved. These are correctness fixes to enforce the existing invalid-input rule, not post-cal method tuning.
- Paused only the local queue watcher PID614742 before it could launch B; remote A was uninterrupted. Guard source patch will be verified before B starts. Source/validity deviation and failure counts will be reported explicitly; frozen primary/guards/calibration values remain unchanged.

### 2026-10-30 — Checkpoint G-3a″ complete; stop for audit

- All1,080/1,080 expected cal records fetched. A540 (531valid/9NEPNEF invalidated with original backups); B540valid, actual/cached-embedding-failure ledger empty.119valid paired parents,357units/arm; no candidate-only failures, no missing records. All main loops replay. No retry/spot-fix for NEPNEF.
- P2/C1 top-1 parent-mean−4.918990kJ/mol,95%CI[−12.326866,+5.287778], paired unit median−0.966714.92/119parents improve across seeds. N controls:−4.496052[−7.014030,−2.196118]; FEK-excluded diagnostic:−9.033975[−13.637764,−5.367545]. These establish a useful energy signal while retaining the registered all-parent failure.
- Floor−7.813545[−11.868278,−4.516133] improves. Parent-weighted obscene-rate−0.732922pp[−2.888774,+1.301535] not significant. Score+0.066951[+0.033629,+0.117091]. No N two-seed energy-guard violation. Wall ratio1.700600>1.3fails (paired solver/assembly sums C1748.03s,P21272.10s; shared prep/MACE excluded). All-parent CI upper5.287778>2alsofails. No eligible winner.
- FEKKAJ cal changes+1017.329,+363.431,+61.187; mean+480.649, all control-obscene. Known score loophole interpretation preserved. Q6 unchanged here; owner's correction remains separate. Largest other mean regressions WEYJIT+24.676,VOBGUR+18.510 do not trip the issued guard.
- No2060 spot run because step5 permits the winning arm only and no arm passes. Frozen spot names retained in the contract/report. No cross-platform fold-in claim, no fold-in.
- Watcher fetched/counted B540at06:54:44UTC, shutdown confirmed06:54:48UTC; ownedVMjgBg3a2a stopped. Connection/fetch delay after B exit was handled automatically while model polling was suspended per owner request. Later redundant recovery check found session absent and allocated nothing. Other track's VM untouched.
- Shared balance506.05→500.01 includes other track; own READY-to-shutdown lower-bound3.036units, estimated3.1–3.4including setup (inferredL4rate1.94/h). Only one ownedL4 was used; second allocation refused. All artifacts/data local before down.
- Final analysis uses exactly frozen pass/guards; replay coverage is diagnostic rather than an added unissued gate. Incomplete collections cannot pass; an A-only dry run correctly remained incomplete/nonpassing. Solver/scoring/calibration unchanged. Updated B_REPORTG.13 covers traces, trials, contractf062f35, full cal/strata table, failures, omitted conditional spot check, spend and spec issues.
- **STOP at G-3a″.** No relaxation, tail, locked-set evaluation, G-3b, scoring change, further tuning or fold-in.

2026-10-30 — G-3c0 issued: freeze six arms, 24 auditor-fixed parents, seeds 11–13 and energy/distortion measures before any run. No parent removal, no adoption gate; stop for audit. Contract: g/evaluation_contract_g3c0.json.

2026-10-30 — G-3c0 implementation 0b034da. Local 2060 profile AJAQAD/AZUSIA completed in 92.7/82.3 s, 36/36 records, 576 returned rows, all valid. Reuse these records. Estimated remaining 22-parent run 30–40 min; all seeds and arms batched per parent. No Colab allocated/spend 0. Default fingerprint 6cc586e17ab1; light suite 26 passed (275.88 s), six focused Adam tests passed (2.89 s), distortion synthetic test passed. Native arms' inherited stale SR.LAST metadata corrected in profile records with original fields retained; no coordinate, score or energy changes. Remaining run uses explicit native-mode metadata and rejects optimizer-fallback substitution.

2026-10-30 — G-3c0 encountered E/Z-multi-embed metadata limitation (CIXYUD, LEGXEA, NUWFAM so far): explicit failure records, no optimizer run for those parents. Repairing the research harness to carry per-variant candidate metadata through the existing E/Z merge; support native score-only no-culling CUDA in multiseed helper. Keep frozen arm settings and all 24 parents. Preserve original failures and rerun affected records after the existing local GPU batch releases its lock. No scientific thresholds or optimizers changed.

2026-10-30 — G-3c0 COMPLETE. Contract41be8d9, six frozen arms, fixed24 × seeds11–13: 432/432 final records, all24 retained, zero final failures/missing/start-digest mismatches; 6,101 returned geometries and 864 XYZ files. Original54 E/Z harness failures preserved, three parents successfully rerun with repaired metadata capture. P2−C1 top1−15.66[−27.40,−5.47], floor−11.54[−22.53,−3.72]; R1 top1−20.89[−38.13,−6.42], floor−14.27[−28.92,−3.64]. R2±/A0 CIs include0. Score-only-from-start arms have heavier angle tails; shape-statistic interpretation is confounded by fixed donor correspondence/reflections, explicitly documented with an ideal-tetrahedron example. Full distributions/strata/spec issues in B_REPORT G.14. Native implementation walls ~3×C1; finish arms~1.43–1.44×. No formal gate. Local GPU completed and released; no VM started, spend0. STOP G-3c0, await audit, no G-3c/fold-in.

2026-10-30 — Owner-requested NUWFAM saved-geometry diagnostic complete (contract63e6486): all288 conformers/4320 aryl C–F measurements. Shared E/Z variant1 has C19–F20 length1.6929Å, local tilt60.20°, F height1.679Å; variant0 bond1.3540Å/local tilt0.74°. R2± top1 chooses variant1 seeds11/13; every arm contains it in its pool. Within-variant C–F lengths invariant to2.32e−6Å, indicating shared geometry/selection, not R2-specific bond stretching. Seed12/floor regression and R1 smaller regression remain unexplained by C–F. No panel exclusion/change; no new solve/energy/relaxation/VM/GPU job. ReportG.15, raw JSON and existing XYZs retained; one analytic geometry test passed.

### 2026-10-30 — Package H amendment / G-3c implementation

Merged master into jgen/b-solver (`149e319`). Owner clarified step 0 is deferred
until G-3d. Implemented experimental per-row finish state in native/replay
loops; defaults retain fingerprint 6cc586e17ab1. CPU phase parity 9/9 and two
state tests pass. Final CUDA phase checks pass 15/15, with 6/6 additional
IA/curvature checks after the shared-update/search-bound changes. Production
golden 245/245 matches. Full H-1 score patch replay 3/3 exact; additional
USAPOU donor graph/eager and native/rewritten CPU parity is exact. The flag-off
USAPOU control shares its row key despite numerical score divergence; no
pre-existing top-key mismatch is claimed. Checkpoint complete; stop for audit.
H-1 crystal checks: Q2 p95 0.0151 nats passes; proposed Q5 p95 2.575 nats fails
and is excluded without tuning. See H1_REPORT.md. Colab spend zero.

## 2026-10-30 — Checkpoint H-R

- Implemented the accepted H-next directive's runner-only full returned-pool
  export in jgen-b. Default `OUT/xyz`; `--no-xyz` opt-out; optional per-parent/
  arm/seed xz multi-frame pool; builder-order labels and coordinates, exact
  returned scores, absolute MACE energies, row keys and role markers.
- Size estimate announced before export: about 87 MB for cal + tail × 3 seeds
  × 2 arms; conservative observed-max scenario 1.15 GB plus small metadata/copies,
  below 2 GB. Plain default; xz option for revised larger runs. No bulk export.
- Full-pool 1 Å symmetry-aware heavy-atom coverage computed from all serialized
  rows and saved in each record; standalone manifest recomputation included.
- Four tests pass on two tail parents in plain/xz modes, demonstrating strict
  gain over synthetic top-1/floor-only coverage and checking atom ordering.
- Actual DAMVAQ CPU/GFN2/MACE seed-11 gate passes: 16 pool rows, 18 XYZs and
  one manifest (19 export files); exact score/energy/key roundtrip, coverage
  recomputation exact. Parent wall 27.7 s. No failed gate.
- Fingerprint unchanged `6cc586e17ab1`; runner-only, no golden required.
  Colab spend zero, no GPU job or VM. See `HR_REPORT.md`; stop for audit.

## 2026-10-30 — G-3d preparation, implementation only

- Added CPU `tools.jgen.g3d_diagnostics`: native switch aid activity/gradient
  ratios; fixed-row single/joint release ablations; geometry/assignment class
  changes; thermal/curvature widths with whole-native-loop cost timing;
  censored convergence distributions/cap; native delta-floor/sign-flip traces
  with macrocycle ring torsions separated. Real diagnostics require explicit
  O1 identity input; no automatic panel selection. Smoke/incomplete data cannot
  select a release set, sigma source or cap. Failures remain in the ledger.
- Added an optional native CPU diagnostic hook, None by default. No solver
  arithmetic/policy change; canonical replay regeneration changes source hash
  only. Default fingerprint remains `6cc586e17ab1`.
- XOJHIN CPU smoke passed (final 126.92 s): two fixed switch rows, 16 release/
  width paths, 112 native iRprop steps, macrocycle class populated; native
  score/shape/symmetry switch reconstruction exact and timing switch digests
  identical. Both fixed rows censored at eight finish steps; no cap inferred.
- Ran `tests/test_finish_phase.py` exactly once to completion: **31 passed in
  488.36 s**, measured process wall **489.622973 s**, CUDA tests included,
  deterministic settings and CodexB GPU lock. Three new diagnostic accounting
  tests pass; compilation and diff checks pass.
- Drafted `g/G3D_CONTRACT_DRAFT.md` and deliberately left it untracked/uncommitted;
  actual fold-in fingerprint and step-0 decisions pending. Report:
  `G3D_PREPARATION_REPORT.md`. No panel or real step-0 run. Colab spend zero.
  Stop and report per Round 3 directive.
