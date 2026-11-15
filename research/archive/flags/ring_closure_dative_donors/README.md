# S7 Flag: `ring_closure_dative_donors` (Archived Flag & Diagnostic Report)

**Directive Reference:** Phase 1 Item 2a  
**Status:** ARCHIVED (Diff only; NOT merged into production)  
**Source Commits:** `2fda996` (*feat(jgen): ring_closure_dative_donors (S7)*), `bd978c2` (*jgen round 3: S7 bench*) on branch `worktree-agent-a0217bdc6f1fac2ae` (`cc1798e`).

---

## 1. Flag Summary & Motivation

In `src/jacobik/stats/mapping.py`, `_apply_smiles_path` maps SMILES donor indices to xyz atoms using a 3D distance proximity check. If two distinct SMILES donor paths remap onto the same physical xyz atom (often due to proximity ambiguity or dative arrows), `xyz_donor_indices.add()` deduplicates them into a single atom set.

The proposed S7 flag (`ring_closure_dative_donors`, legacy `False`) intervened only for SMILES containing ring-closure datives (`->\d` or `<-\d`), forcing both candidate atoms to be retained independently without set-collapse.

---

## 2. Donor-Count Check Across the 56 Changed Parents

Per Phase 1 instructions, we evaluated the 56 parents (out of 1,680 frame parents) that change donor sets under S7, measuring the physical Euclidean distance $d(M, D)$ from the metal to each donor in the **reference crystal geometry** ($d \le r_{\text{cov}}\text{ sum} + 0.5$ Å bonding threshold):

### Key Metrics
- **Total Frame Parents:** 1,680
- **Parents with Ring-Closure Datives in SMILES:** 1,185
- **Parents with Changed Donor Sets under S7:** Exactly 56
- **Legacy Donor Set across 56 Parents:**
  - Total donors: **257** (mean 4.59 / complex)
  - Donors within bonding distance: **254 / 257 (98.8%)**
  - Complexes where 100% of donors are bonded: **54 / 56 (96.4%)**
- **S7 Donor Set across 56 Parents:**
  - Total donors: **341** (+84 donors added)
  - Donors within bonding distance: **264 / 341 (77.4%)**
  - **77 of the 84 added atoms (91.7%) are NON-BONDED distal atoms** far outside the coordination sphere:
    - Mean metal-donor distance: **4.54 Å**
    - Extreme examples: **8.61 Å** on `ESIQUT`, **7.82 Å** on `NEPNEF`, **6.58 Å** on `TINNIO`, **5.71 Å** on `JAYCEV`
    - Mayer Bond Order (BO) check: 80 / 84 have Mayer BO = **0.0000**
  - Complexes where 100% of donors are bonded: **Only 5 / 56 (8.9%)** (51 complexes severely contaminated).

---

## 3. Solver Failure Diagnosis & Recommendation

### Root Cause
tmQM SMILES generation algorithms frequently place dative ring-closure arrows (`->\d`) on remote backbone atoms that do not coordinate to the metal in the crystal structure. In legacy Jacobik, proximity remapping shifts the coordinate to the genuine coordinating donor, where set deduplication cleanly absorbs it without harming coordination geometry.

When S7 prevents deduplication, the non-bonded distal atom is flagged as an active donor. The kinematic solver subsequently creates a coordination bond constraint to this distant atom, violently pulling the ligand backbone into the metal center. This results in severe steric clash, geometry destruction, and massive energy penalties:
- `NEPNEF`: $\Delta E = +423.8\ \text{kJ/mol}$
- `JAYCEV`: $\Delta E = +305.2\ \text{kJ/mol}$

### Final Owner Decision Record
The S7 behavior change is **REJECTED** from the production solver. The diff, prevalence census, and diagnostics are preserved here for research reference.
