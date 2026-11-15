# Campaign I Evaluation Contract: Isomer Space (Scope B, Factored Symmetry, Per-Class Output)

**Date:** 2026-11-06 (Amended per Audit §5)  
**Implementer:** Gemini on `jgen/i-isomer`  
**Auditor:** Claude (Opus)  
**Directive:** `directive_jgen_I_isomer_2026-11-03.md`  
**Baseline Fingerprint:** `5f74b14b57ed` (IA-k baseline, master `74b4d47`)  

---

## 1. Objectives & Scope
The isomer-space campaign addresses two primary defects identified in A4t / M0–M3:
1. **Waste from Hidden Duplicates (§4a):** On complexes with large automorphism groups (the 13 capped cal/tail parents: `XAQCAT`, `SILROY`, `NOYTUS`, `CIFZEZ`, etc.), the graph permutation search truncates, causing duplicate conformations to escape dedup and waste output conformer slots. Factored symmetry decomposes the heavy-atom graph into core coordination and canonical rank/leaf permutations to guarantee an exact upper bound on symmetric RMSD. Note: `dedup_factored` uses `MAX_AUTOMORPHISMS = 20000` (correcting earlier typo 4096).
2. **Coverage Loss from Absent Stereoisomers (§4b, Scope B):** In cal parents (`DAMVAQ`, `FETTAZ`, `FUTSOD`, `XACMOE`, `LISVOC`; note `ZAZREZ` has no stereogenic elements and was an M3 artefact), the crystal stereoisomer was absent from every returned pool row across all seeds because unspecified ligand tetrahedral centres, binding-created donor centres, or acyclic E/Z double bonds (including M–C=X bonds) were never explored.
3. **Equitable Output Allocation (§4d):** Retain up to $K=8$ distinct rows per eligible isomer class (`out_per_class=True`) across all parents, rather than letting one dominant isomer crowd out all output slots.
4. **Ceiling Sampling & Expected Ceiling Misses (`FUTSOD`, `XUKKIA`):** Under the $1.5\times$ cost ceiling cap (held unchanged per Owner decision 2026-11-07), when feasible isomer classes exceed the cap, deterministic pseudo-random sampling selects a subset (specifically, 2 of 32 classes on `FUTSOD` and 2 of 4 classes on `XUKKIA` per seed). Because of this ceiling sampling, `FUTSOD` misses the crystal class on all three development seeds (11, 12, 13) and `XUKKIA` misses it on two seeds (12, 13; selected on seed 11). Per the Owner's decision (Audit 2026-11-07 Addendum §2), `FUTSOD` and `XUKKIA` are recorded as **expected misses caused by the cost ceiling**. All other 19 stereo-active parents have $\le 2$ feasible classes and are fully explored without ceiling truncation.
5. **Expected Misses (`DAMVAQ` Cage Ring Junction):** For `DAMVAQ`, carbanion donors (`L1_D0_C, L1_D2_C`) are recovered under Scope B. The 10 cage bridgeheads (`len(small_rings) >= 2`) are excluded from enumeration as rigid polycyclic cage vertices, leaving the central norbornyl-quadricyclane ring junction in ETKDG's default anti/exo diastereomer (1.18 Å cage RMSD vs 0.099 Å syn/endo in crystal). Per Audit §3 ruling, polycyclic ring junction stereochemistry is out of Scope B (1 parent exception) and is recorded as an expected miss. No cage enumeration is performed.

---

## 2. Experimental Arms
All arms are evaluated paired with their production control in the same invocation on the same platform (local RTX 2060 or Colab L4):

| Arm | Description | Flags / Settings |
|---|---|---|
| **C** ($I_0$) | Production control | `5f74b14b57ed` defaults (all experiment flags off) |
| **A** | Factored symmetry dedup | `dedup_factored=True`, `dedup_rmsd=0.5` |
| **AS** | A + Stereo candidates | `dedup_factored=True`, `stereo_candidates=True`, `stereo_scope="B"`, `stereo_cost_ceiling=1.5` |
| **ASP** | AS + Per-class output | `dedup_factored=True`, `stereo_candidates=True`, `out_per_class=True`, `out_per_class_k=8`, `out_per_class_window=35.0` |

---

## 3. Panels & Seeds
- **Evaluation Panel:** The 120-complex cal/tail panel (`research/campaigns/c2/C2_PANEL.json`), excluding `FEKKAJ` and `NEPNEF` from primary statistics ($N = 118$).
- **Development & Tuning:** Seeds 11, 12, 13 only.
- **Final Unbiased Check:** Fresh seeds 31, 32, 33 once before fold-in. No cherry-picking or tuning on fresh seeds.

---

## 4. Metrics & Evaluation Criteria

### Primary Metrics:
1. **Duplicate Fraction of Returned Rows:** Measured by exact offline dedup (M2 metric); target is strict reduction on the 13 capped parents and 0 increase on uncapped parents.
2. **Crystal Stereo Presence:** Percentage of stereo-bearing parents (cal/tail stereo-active parents) where the crystal stereoisomer is present in the pool. Specifically tracking recovery on the absent set (`DAMVAQ`, `FETTAZ`, `FUTSOD`, `XACMOE`, `LISVOC`).
3. **Coverage:** Percentage of complexes with RMSD $\le 1.0$ Å and $\le 0.5$ Å to the relaxed reference.
4. **Energy Floor and Top-1 ΔE:** Parent-clustered bootstrap 95% confidence intervals on ΔE (kJ/mol) vs crystal reference.
5. **Distinct Rows & Classes per Parent:** Mean number of returned rows, distinct conformers, and stereo classes.
6. **Pool Quality $\rho$:** Spearman rank correlation between internal model score and unrelaxed MACE energy within the conformer pool (more negative is better; indicates lower internal score corresponds to lower MACE energy).
7. **Wall Time and Memory:** Compute cost relative to production control.

### Qualification Routes (Directive I §2):
- **Arm A (Correctness):**
  - Hard exactness gate passed (107 uncapped parents bit-identical, 13 capped parents $\le$ survivors).
  - Energy floor and top-1 CI upper bound $< +2.0$ kJ/mol vs Control.
- **Arm AS (Coverage):**
  - Stereo-present count strictly increases on the absent set.
  - $\le 1.0$ Å coverage CI lower bound $> -3.0$ percentage points vs Control.
  - Energy floor and top-1 CI upper bound $< +2.0$ kJ/mol vs Control.
  - Solve wall time $\le 1.5\times$ Control ceiling.
- **Arm ASP (Output Contract):**
  - Every eligible class (best row within 35.0 kJ/mol window of pool best) keeps up to $K=8$ distinct conformers.
  - Energy floor and top-1 CI upper bound $< +2.0$ kJ/mol vs Control.
  - Storage footprint and wall time stated.

---

## 5. C-2b Regression Guard Specification
To prevent regressions on individual complexes relative to control $I_0$:
- **Trip:** Complex is $> +25.0$ kJ/mol worse on $\ge 2/3$ seeds compared to paired control $I_0$.
- **Mirror:** Complex is $< -25.0$ kJ/mol better on $\ge 2/3$ seeds compared to paired control $I_0$.
- **Null:** Maximum number of trips observed between seed rotations of control $I_0$ evaluated against itself across seeds 11, 12, 13.
- **Pass/Fail Rule:** The arm fails the regression guard iff:
  $$\text{trips} > \text{null} + 2 \quad \text{AND} \quad \text{trips} \ge \text{mirrors}$$
- **Known Artefact List:** Results will be reported both with and without known artefact complexes:
  `NUHPEL`, `MIDQEY`, `UMUBAF`, `YUCZUS`, `NUWFAM`, `CAFWIQ`.
  The formal binding pass/fail decision uses the panel **without** these 6 artefact complexes.

---

## 6. Guardrails & Negative Constraints
1. **Hard Exactness Gate:** Arm A must be identical to production on all 107 uncapped cal/tail parents and yield $\le$ survivors on the 13 capped parents.
2. **Graph Automorphism Integrity:** Every deduplication merge must correspond to a verified mathematical automorphism (`adj[p, p] == adj`); zero false duplicates allowed.
3. **Specified Stereochemistry Immutability:** Any stereochemical descriptor specified in the input SMILES (`@`, `@@`, `/`, `\`) must never be altered or overridden by `set_chiral` / `set_ez`.
4. **Achiral Geometry Symmetry Merging:** In an achiral coordination environment, enantiomeric pairs must merge into a single canonical class representative.
5. **Deterministic Ceiling Sampling:** When total stereoisomer classes exceed the $1.43\times$ ceiling (`N_CLASS_CAP = 8`), classes must be sampled deterministically.
6. **Replay Parity & Fingerprint Identity:** Default settings must preserve fingerprint `5f74b14b57ed` exactly. Replay graph execution must remain bit-identical.

## 7. Pre-fresh-seed runtime amendment (owner/auditor, 2026-11-08)

Declared before any fresh-seed result is seen: QEZVEY is run in its own login-A100 burst after the three fresh-seed shards. If it times out, it is reported as a timeout and excluded from primary statistics for runtime only. All other contract criteria remain unchanged.
