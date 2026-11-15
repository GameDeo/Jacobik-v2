# Package G: Checkpoint G-2 Interim Status & Handover Document

**Date:** 2026-10-28 18:26 Local  
**Branch:** `jgen/b-solver` at commit `ec28790` (Merged `master` `8dbe29c`)  
**Production Fingerprint:** `615ae97d2350` (Verified bit-identical, golden clean)  
**Status:** Wrapped up and stopped cleanly per owner instruction.

---

## 1. Executive Summary & Stop State

All active background tasks, subagents, and check-in schedules have been terminated.
- **Hardware Status:** RTX 2060 SUPER is idle and clean (39°C, 647 MiB system memory). All CPU workers released.
- **Git State:** Worktree `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.claude/worktrees/jgen-b` is on `jgen/b-solver`, cleanly merged with `master` (`8dbe29c`, incorporating `runkit` R2b and the trope suite).
- **Core Investigation Accomplished:** The root cause of the invalid G-1 Task O1 score calculation has been completely identified, debugged, and mathematically resolved down to $< 10^{-7}$ precision against stored conformer scores.

---

## 2. Key Mathematical & Architectural Discovery: Task O1 Exact Score Reproduction

In Checkpoint G-1, the reference diagnostic script reported `score_calc_init` of $-152.8$ against stored `conf.score` of $0.1998$ on `HEDJOP`, causing the auditor to reject O1.

### Root Cause Analysis
1. **Coordinate Frame Baseline ($X_{\text{full}}$ vs `mol.positions()`):**
   - The G-1 diagnostic script applied the conformer solver state degrees of freedom (`conf.solver_state["thetas"]`) onto the raw crystal structure coordinates (`mol.positions()`).
   - In production (`jacobik.engine.ComplexBuilder.solve`), the solver first runs `_build_shell(...)`, embedding coordination shells around the metal and establishing an embedded reference geometry $X_{\text{full}}$.
   - Kinematics DOFs ($\theta$) are referenced strictly to $X_{\text{full}}$. Applying them to the crystal reference placed the ligands 5–12 Å away from the metal, causing massive artificial clashes ($Q_6 = -119.5$) and score collapse to $-152.8$.

2. **Energy Bend Prior Normalization:**
   - The energy bend prior is computed via `jacobik.ik.angle_bends.energy_prior`.
   - Crucially, in `solver_replay.py`:
     $$\text{total} = \sum_{\text{ligands}} \text{energy\_prior}(\theta_{\text{bends}}) / \text{scorer.\_agg\_norm()}$$
     Omission of the `_agg_norm()` divisor caused a factor of $Z = 44.0$ error in the bend prior.

3. **Multi-Shell E/Z Variant Enumeration (`_solve_ez_variants`):**
   - Complexes with E/Z stereocenters (e.g., `NUWFAM`) run independent solves per E/Z variant.
   - Each variant produces its own embedded reference geometry $X_{\text{full}}$ and forward kinematics operators.
   - Conformer results store `c.embed_idx` (or `None` for single-embed complexes). Evaluating conformer $c$ requires referencing the specific $X_{\text{full}}$ of its own variant $v = \text{c.embed\_idx if c.embed\_idx is not None else } 0$.

4. **Macrocycle Seam Projection & Residual:**
   - For complexes with macrocycles (e.g., `USAPOU`), `solver_replay.py` projects dihedrals onto the ring closure manifold and applies penalty $-1000 \cdot \text{seam\_ss}$ (`_CLOSURE_W = 1000.0`).

### Exact Objective Formulation
When evaluated using:
$$S(\theta) = S_{\text{stat}}\left(\text{FK}(\theta, X_{\text{full}}); \text{row\_groups}\right) + \frac{S_{\text{bend}}(\theta)}{Z} - 1000 \cdot \text{seam\_ss}$$
the reconstructed score matches `conf.score` exactly across test parents:
- `HEDJOP`: max diff = $1.79 \times 10^{-7}$
- `FEKKAJ`: max diff = $0.00 \times 10^{0}$
- `NUWFAM`: max diff = $2.38 \times 10^{-7}$
- `QEZVEY`: max diff = $1.79 \times 10^{-7}$
- `DAMVAQ`: max diff = $4.77 \times 10^{-7}$

This completely satisfies the mandatory G-2 assertion:
$$\left|S_{\text{calc}}(\theta_0) - \text{conf.score}\right| < 10^{-4}$$

---

## 3. Work Package Status for Checkpoint G-2

| Item | Requirement | Status | Notes |
|---|---|---|---|
| **Step 1** | Merge `master` (`8dbe29c`) into `jgen/b-solver` | **COMPLETE** | Commit `ec28790`. Fingerprint verified `615ae97d2350`. |
| **Step 2** | Redo Task O1 with assertion, bounded L-BFGS, strict failure logging | **IN PROGRESS** | Objective math & assertion solved. L-BFGS wrapper with step caps ($\le 0.5\text{ Å}, \le 15^\circ$) and xTB/MACE evaluation ready to plug in. |
| **Step 3** | S-P1 Recount (collapse rotor symmetries, heavy torsions only, reachability) | **NOT STARTED** | Held pending review per directive. |
| **Step 4** | O2 Prototypes (`adam_metric`, `fire_metric`, `hybrid_lm`) | **SPECIFIED** | Algorithmic design specified in `G_OPTIMIZER_NOTE.md` and addendum; contract pre-commitment pending. |

---

## 4. Clear Handover Instructions for Next Agent / Session

To resume and complete Checkpoint G-2:
1. **Redo Task O1:**
   - Update `research/bench/tools/jgen/run_o1_diagnostic.py` using the exact score formulation verified in Section 2.
   - Run bounded L-BFGS from $\theta_0$:
     - Max displacement: $\Delta x \le 0.5\text{ Å}$ translation, $\Delta \theta \le 15^\circ$ angular per DOF.
     - Record any rows hitting the boundary.
     - Evaluate relaxed single points via GFN2-xTB and MACE float64. Record any convergence failure explicitly as `"failed"` (never $0.0\text{ kJ/mol}$).
     - Run on the 8 parents: `HEDJOP`, `FEKKAJ`, `NUWFAM`, `QEZVEY`, `DAMVAQ`, `USAPOU`, `NEPNEF`, `AJAQAD` (top-1 + 5 rows = 48 conformers).
     - Classify conformers into (a) optimizer problem, (b) score surface strained, (c) converged.
2. **Execute S-P1 Recount:**
   - Update `run_sp1_coverage.py` to collapse symmetric rotors ($\text{CH}_3, \text{CF}_3, t\text{Bu}, C_2$) and measure heavy torsions at $0.2\text{ rad}$ and $0.35\text{ rad}$.
   - Add single-joint steric clash check on unfilled modes to split into reachable vs clashing.
3. **Commit Evaluation Contract & Prototype O2:**
   - Commit `research/data/jgen/campaign/B/g/evaluation_contract_o2.json` in its own commit before running optimizer arms.
   - Prototype `adam_metric`, `fire_metric` (branch-free $P = F \cdot v$ reset with `torch.where`), and `hybrid_lm`.
   - Screen on trope suite first, then `tail` panel (seeds 11–13) against paired control.
4. **Checkpoint G-2 Delivery:**
   - Update `B_REPORT.md` (§G) and `B_PROGRESS.md`.
