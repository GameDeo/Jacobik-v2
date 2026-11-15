# Constraints Database Tables Audit: Production vs Archived Tables

**Date:** 2026-10-25  
**Context:** Phase 3 of Jacobik Consolidation and Restructure (`src/fit`)

---

## 1. Executive Summary

In accordance with Phase 3 directive ("Check which ones are read at solve and score time with the production config, and archive the tables that are not"), this audit verifies every prefix present in the historical `constraints_2026-10-18.json` (445,864 total keys) against the production solver and scorer execution paths.

| Quantity Prefix | Historical Keys | Production Role | Status in `src/fit/` | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | 118,673 | M-D bond distance | **Production** | Required by `scorer.py:q1_objective` and `StatsLookup.dist("Q1", ...)` |
| **Q2** | 298,737 | D-M-D cis / bite angle | **Production** | Required by `scorer.py:q2_objective` and `StatsLookup.dist("Q2", ...)` |
| **Q3** | 10,267 | D-M-D trans angle | **Production** | Required by `scorer.py:q3_objective` |
| **Q4** | 1,306 | L-D-M bend angle | **Production** | Required by `scorer.py:q4_objective` |
| **Q5** | 4,155 | Ligand torsion | **Production** | Required by `scorer.py:q5_objective` (folded wells) |
| **Q6** | 535 | Steric exchange repulsive wall | **Production** | Required by `scorer.py:q6_objective` (half-Gaussian onset) |
| **QO** | 10,220 | Joint donor orientation | **Production** | Required by `scorer.py:sorient_objective` and `StatsLookup.orientation_dist` |
| **Q6P** | 336 | Purely repulsive PMF core & m1-3 | **Production** | Required by `scorer.py:_q6_pmf` under `q6_model="repulsive"` (u1 production profile) |
| **NCI** | 13 | Noncovalent interaction splines | **Retired (archived)** | Dropped from production fit/build/runtime in the 2026-11 cleanup (`research/archive/code/nci/`) |
| **QG** | 766 | Hypothesis geometry priors | **Reserved (hypothesis layer)** | Not queried yet: the coordination-hypothesis layer that uses it is planned, not built. Keep in the DB and the refit |
| **QP** | 288 | Hypothesis partition priors | **Reserved (hypothesis layer)** | Not queried yet: haptic split prior for the planned hypothesis layer. Keep in the DB and the refit |
| **QB** | 0 (unfitted) | Candidate donor rows | **Reserved** | Extracted, not fitted; for a later ligands-only / donor-ambiguity mode |
| **QV** | 124 | Bond-valence sum levels | **Reserved (off)** | Only accessed if `bvs=True`. In production `bvs=False` |
| **QVR** | 444 | Bond-valence sum R0 constants | **Reserved (off)** | Only accessed if `bvs=True`. In production `bvs=False` |

---

## 2. Detailed Query Path Analysis

### 2.1 Production Quantities (Active)
1. **Q1, Q2, Q3, Q4, Q5, Q6:**
   - Looked up via `StatsLookup.dist(quantity, key)` in `jacobik.stats.stats`.
   - Directly parameterized in `jacobik.ik.scorer.Scorer`:
     - Q1: `_q1_mean`, `_q1_var`, `_q1_weights`
     - Q2: `_q2_mean`, `_q2_var`, `_q2_weights`, `_q2_nu`
     - Q3: `_q3_mean`, `_q3_var`, `_q3_weights`
     - Q4: `_q4_mean`, `_q4_var`, `_q4_weights`
     - Q5: `_q5_means`, `_q5_kappas`, `_q5_weights`
     - Q6: `_q6_mu`, `_q6_sigma`
2. **QO (Orientation):**
   - Looked up via `StatsLookup.orientation_dist(...)`.
   - Used by `Scorer.sorient_objective` when `orientation_model="sorient"`.
3. **Q6P (Purely Repulsive PMF):**
   - Looked up via `StatsLookup.q6_pmf_tables()`.
   - Activated in `Scorer.__init__`:
     ```python
     if Q6_MODEL in ("repulsive", "pmf") and ge.n_q6 > 0:
         tabs = stats.q6_pmf_tables()
         self._q6_pmf = q6pmf.build_runtime(mol, ge, tabs, self._q6_mu, self._q6_sigma, device)
     ```
   - In production profile `u1`, `q6_model="repulsive"`.

### 2.2 Reserved Quantities (in the DB, not read by the production solve)

**Owner correction (2026-10-26):** these are not retired. QG / QP / QB are the data for the planned coordination-hypothesis
layer: let the solver resolve ambiguity (coordination-geometry interpretation; how haptic arcs split, e.g. a 1,3,5-triene
as eta6 vs eta4+eta2, or a diene plus a separately bound alkene). `hypothesis_tables.py` (rows + fits) and `reference_ll.py`
(per-term score normalisation, so hypotheses with different site counts compare) are back in `src/jacobik/stats/`.
The refit must keep extracting and fitting them.
1. **QG & QP (Categorical Hypothesis Priors):**
   - Introduced in B2.5 exploration for hypothesis-layer priors.
   - Code verification: Grepping across `src/jacobik/` confirms `stats.priors.get("QG")` and `get("QP")` are never invoked.
   - Decision: kept in the DB and the refit; consumed by the hypothesis layer once built.
2. **QB (Candidate Donors):**
   - Extracted during sharded extraction as insurance for a standalone ligands-only solver mode that was superseded.
   - Never fitted into DB entries.
   - Decision: kept as extracted rows (reserved).
3. **QV & QVR (Bond Valence):**
   - `StatsLookup.bvs_tables()` returns `(pri.get("QV", {}), pri.get("QVR", {}))`.
   - In `scorer.py`:
     ```python
     BVS: bool = _profile.get("bvs")
     if BVS and hasattr(stats, "bvs_tables"):
         self._bvs = _bvs_runtime(...)
     ```
   - In `profile.py`, `bvs` defaults to `False` across all profiles (including `u1`).
   - The bond-valence coupling was an exploratory research flag (Plan 3 Step 2) that did not enter production.
   - Decision: reserved; stays in the DB (`bvs=False` in production). `research/archive/code/interactions/bvs.py` holds the runtime.
