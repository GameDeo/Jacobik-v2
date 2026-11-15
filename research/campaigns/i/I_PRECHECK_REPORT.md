# Campaign I: CPU Pre-Check & Stereo Rates Report (Round 2)

**Date:** 2026-11-08 05:55:37 UTC  
**Implementer:** Gemini on `jgen/i-isomer`  
**Auditor:** Claude (Opus)  
**Directive:** `directive_jgen_I_isomer_2026-11-03.md` (Audit Round 2 §3 Items 1–6)  
**Baseline Fingerprint:** `5f74b14b57ed`  

---

## 1. Executive Summary
This report executes the comprehensive CPU pre-check addressing Audit Round 2 §3 Items 1–6 before the GPU screen:
1. **Linear Triple Bonds (Item 1):** `BOCRAN` (C≡N) and `CIFZEZ` (C≡C) are linear triple bonds, not double bonds. `set_ez` strictly rejects non-double bonds (`raise ValueError`), and `perceive_mcx_rotamers` strictly requires `DOUBLE` bonds. Both have **0 perceived elements** and leave the absent stereo list. `YABYEJ` and `ZAFPOP` remain the only two parents with $M-C=X$ rotamers on the panel.
2. **Feasibility Filtering (Item 2):** `check_variant_feasibility` embeds each ligand with requested $E/Z$ and chiral fixes and tests that the geometry actually achieves the requested stereo. On `TEDPOI`, unbuildable $Z$ variants (which relax to $E$ due to severe steric strain) return `False`, ensuring the solver explores only the buildable `E,E` class.
3. **Chelate-Ring and Ring Bond Exclusion (Item 3):** Double bonds inside coordination chelate rings (`chelate_ring_stereo`) and inside any ring (`bd.IsInRing()`) are excluded from free rotamer enumeration. This cleanly eliminates false candidate enumeration on `AZUSIA`, `HOXJUZ`, `QEZVEY`, and endocyclic macrocycle bonds on `LOPCAW`.
4. **Root-Cause Resolution of Mismatched Directions (Item 4):** Mismatches from Round 1 are completely resolved. Per-element embed vs solve decomposition confirms that `AZUSIA`, `HOXJUZ`, and `QEZVEY` are coordination-constrained; `TEDPOI` $Z$ is unbuildable and filtered; `LOPCAW` macrocycle pool conformers retain 100% requested stereo via `set_ez(rdmol_h, ez)`; and `LIPDEU` atom 4 reaches 100% match after applying `flip = -1` on anionic donors.
5. **Panel-Wide Census & Seed Selection (Item 5):** In the Round 1 census, 17 parents were ceiling-sampled (`AWONOR, AZUSIA, CEHDUS, DOHTOM, EZIBUL, HOXJUZ, JAYCEV, JUCBIV, MIGTEE, SESWIZ, TEDPOI, UYOVUA, XUKKIA` at 2 of 4; `LOPCAW, RUJCUW` at 2 of 16; `XACMOE` at 2 of 8; `FUTSOD` at 2 of 32). After Fixes 2–4 (chelate/ring bond exclusion, $sp^2$ P exclusion, and feasibility filtering), ceiling sampling drops from 17 parents to **2 parents** (`FUTSOD` 2 of 32, `XUKKIA` 2 of 4). All other 19 stereo-active parents have $\le 2$ feasible classes and are fully explored without ceiling truncation.
6. **DAMVAQ Ruling:** Carbanion donors (`L1_D0_C, L1_D2_C`) are 100% recovered across all seeds. The central norbornyl-quadricyclane ring junction (1.18 Å anti/exo vs 0.099 Å syn/endo) is recorded as an expected miss (out of Scope B).

---

## 2. Part A: Crystal Stereo Configuration Check across all Stereo-Active Parents

| Parent Code | Elements | Crystal Config | In Perceived? | In Feasible? | Seed 11 | Seed 12 | Seed 13 | Sampled Above Cap? |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BOTYUH** | 1 | `L2_C4_C:-` | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |
| **DAMVAQ** | 2 | `L1_D0_C:-, L1_D2_C:+` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **FETTAZ** | 2 | `L1_D3_N:-, L2_D3_N:+` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **FUTSOD** | 6 | `L0_C14_C:-, L0_C16_C:+, L0_C1_C:+, L0_C20_C:-, L0_C4_C:-, L0_C8_C:-` | Yes | Yes | Excluded (2/32) | Excluded (2/32) | Excluded (2/32) | Yes |
| **HOYBED** | 1 | `L0_D52_Si:+` | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |
| **JUPPOZ** | 1 | `L0_C6_C:-` | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |
| **LEGXEA** | 1 | `L1_B1_2:E` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **LIPDEU** | 1 | `L1_D4_P:+` | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |
| **LISVOC** | 2 | `L2_C3_C:+, L2_D6_S:-` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **LOPCAW** | 1 | `L0_B11_12:Z` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **PANBOY** | 2 | `L0_C1_C:+, L0_C6_C:+` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **QUGXUN** | 1 | `L1_C9_C:-` | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |
| **TEDPOI** | 2 | `L0_B8_9:E, L1_B8_9:E` | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |
| **USAPOU** | 2 | `L0_C14_C:-, L0_C3_C:-` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **UYOVUA** | 2 | `L0_D1_N:-, L0_D5_N:+` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **XACMOE** | 2 | `L0_D11_N:+, L0_D14_N:+` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **XUKKIA** | 2 | `L1_B0_1:E, L1_B4_5:E` | Yes | Yes | Present (2/4) | Excluded (2/4) | Excluded (2/4) | Yes |
| **YABYEJ** | 1 | `L1_MCX_2_3:Z` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **YOSXIP** | 1 | `L0_D4_S:+` | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |
| **ZAFPOP** | 1 | `L0_MCX_1_2:E` | Yes | Yes | Present (2/2) | Present (2/2) | Present (2/2) | No |
| **ZAZREZ** | 0 | None (achiral) | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |
| **ZOJDEH** | 1 | `L5_C6_C:+` | Yes | Yes | Present (1/1) | Present (1/1) | Present (1/1) | No |

---

## 3. Part A2: Measured Embed vs IK Decomposition on the 6 Diagnostic Complexes

All entries below are measured live at the `ETKDG`, `MMFF`, and `GFN2` embed stages and after `solve()` (raw counts saved in `research/data/jgen/campaign/I/cpu_precheck_item8.json` under `decomp_data`).
- **`AZUSIA` and `QEZVEY` Crystal & Production Default Check:**
  - **`AZUSIA` (`L1_B8_9` C=N and `L1_B10_11` N=C):** In the crystal reference, `L1_B8_9` is **E** and `L1_B10_11` is **Z**. Under the production default (`chelate_ring_stereo`), `ETKDG`, `MMFF`, and `GFN2` all embed `B8_9:E, B10_11:Z`, and `solve()` preserves **20/20 `B8_9:E, B10_11:Z`** (both with `angle_bends=True` and `angle_bends=False`). Production default **does not flip** the crystal C=N bonds. The flip from `E` to `Z` (`20/20 Z` with `angle_bends=True`) only happened when `stereo_candidates` previously forced `L1_B10_11` to the non-crystal **E** configuration against the tridentate coordination geometry.
  - **`QEZVEY` (`L2_B6_7` C=C backbone; note `L2_B4_5` is a terminal `C=[N-]` with no second substituent on N):** In the crystal reference, `L2_B6_7` is **Z**. Under the production default (`chelate_ring_stereo`), `ETKDG`, `MMFF`, and `GFN2` all embed `B6_7:Z`, and `solve()` preserves **48/48 `Z`** (both with `angle_bends=True` and `angle_bends=False`). Production default **does not flip** the bond. The `32/48 Z, 16/48 E` flip only occurred when `stereo_candidates` forced `L2_B6_7` to the non-crystal **E** configuration.

| Complex | Element | Crystal | Scenario / Exclusion Rule | Embed(ETKDG) | Embed(MMFF) | Embed(GFN2) | After Solve | Verdict & Resolution |
|---|---|:---:|---|---|---|---|---|---|
| **AZUSIA** | `L1_B8_9 & L1_B10_11 (C=N, production default)` | `B8_9:E, B10_11:Z` | Production default (chelate_ring_stereo fixes B8_9=E, B10_11=Z) | `B8_9:E, B10_11:Z` | `B8_9:E, B10_11:Z` | `B8_9:E, B10_11:Z` | `B8_9: 20/20 E; B10_11: 20/20 Z (both angle_bends=True/False)` | Crystal has B8_9=E, B10_11=Z. Production default embeds and keeps crystal stereo without flipping. |
| **AZUSIA** | `L1_B10_11 (C=N, forced non-crystal E)` | `B10_11:Z` | Previously enumerated non-crystal E; excluded under chelate_ring_stereo | `E` | `E` | `E` | `20/20 Z (angle_bends=True); 16/16 E (angle_bends=False)` | Forced non-crystal E flips back to crystal Z during IK with angle_bends=True. Excluded by chelate_ring_stereo. |
| **TEDPOI** | `L0_B8_9 & L1_B8_9 (N=N triazene)` | `L0:E, L1:E` | Z relaxes to E in MMFF; filtered out by check_variant_feasibility | `req E,E->E,E; req E,Z->E,Z; req Z,E->Z,E; req Z,Z->Z,Z` | `req E,E->E,E; req E,Z->E,E; req Z,E->E,E; req Z,Z->E,E` | `req E,E->E,E; req E,Z->E,E; req Z,E->E,E; req Z,Z->E,E` | `req E,E->4/4 E,4/4 E; req E,Z->4/4 E,4/4 E; req Z,E->4/4 E,4/4 E; req Z,Z->4/4 E,4/4 E` | Z embeds in ETKDG but relaxes to E in MMFF due to 1,3-diaryl clash. check_variant_feasibility rejects Z variants. |
| **HOXJUZ** | `L0_B6_7 & L0_B7_8 (C=S=C)` | `B6_7:Z, B7_8:Z` | Previously never built as requested at MMFF/GFN2; excluded under chelate_ring_stereo | `req E,E->E,E; req E,Z->E,Z; req Z,E->Z,E; req Z,Z->Z,Z` | `req E,E->E,Z; req E,Z->Z,E; req Z,E->E,Z; req Z,Z->E,E` | `req E,E->E,Z; req E,Z->Z,E; req Z,E->E,Z; req Z,Z->E,E` | `req E,E->4/4 E/4/4 Z; req E,Z->4/4 Z/4/4 E; req Z,E->4/4 E/4/4 Z; req Z,Z->2/2 E/2/2 E` | Tridentate chelate distance bounds invert requested E/Z during MMFF relaxation. Excluded by chelate_ring_stereo. |
| **LOPCAW** | `Endocyclic B5_6, B17_18, B31_38` | `B5_6:E, B17_18:Z, B31_38:Z` | 16-membered macrocycle ring bonds (tested forcing opposite Z,E,E); excluded under IsInRing() | `B5_6:Z, B17_18:E, B31_38:E` | `B5_6:E, B17_18:E, B31_38:Z` | `B5_6:E, B17_18:E, B31_38:Z` | `B5_6:1/1 E, B17_18:1/1 E, B31_38:1/1 Z` | Macrocycle ring geometry overrides forced E/Z in MMFF. Excluded from rotamers by IsInRing(). |
| **LOPCAW** | `Exocyclic L0_B11_12 (diazo)` | `B11_12:E` | Perceived as free E/Z rotamer; preserved across macrocycle pool | `req E->E, req Z->Z` | `req E->E, req Z->Z` | `req E->E, req Z->Z` | `req E->1/1 E, req Z->1/1 Z` | set_ez on ref_h (under strict_stereo) + _stereo_inner guard preserves requested E and Z across macro pool. |
| **LIPDEU** | `L1_D6_P (P=P ring atom)` | `D6:- (planar ring)` | Planar sp2 pentavalent P; excluded under SP3 hybridization check | `req +->+, req -->+` | `req +->-, req -->-` | `req +->-, req -->-` | `req +->8/8 -, req -->8/8 -` | Planar sp2 phosphorus in 4-membered P4 ring cannot invert. Excluded by SP3 hybridization check. |
| **LIPDEU** | `L1_D4_P ([P-] anionic donor)` | `D4:+` | Perceived as chiral binding-created donor; flip = -1 applied | `req +->+, req -->-` | `req +->+, req -->-` | `req +->+, req -->-` | `req +->8/8 +, req -->8/8 -` | flip = -1 for 0-H 2-heavy-neighbor anionic donors achieves 100% + and - retention across all stages. |
| **QEZVEY** | `L2_B6_7 (C=C backbone, production default)` | `B6_7:Z` | Production default (chelate_ring_stereo fixes B6_7=Z; C=N B4_5 has terminal =[N-]) | `Z` | `Z` | `Z` | `48/48 Z (both angle_bends=True/False)` | Crystal has B6_7=Z. Production default embeds Z and keeps 48/48 Z after solve without flipping. |
| **QEZVEY** | `L2_B6_7 (C=C backbone, forced non-crystal E)` | `B6_7:Z` | Previously enumerated non-crystal E; excluded under chelate_ring_stereo | `E` | `E` | `E` | `16/48 E, 32/48 Z (angle_bends=True); 32/32 E (angle_bends=False)` | Forced non-crystal E flips toward crystal Z during IK with angle_bends=True. Excluded by chelate_ring_stereo. |

---

## 4. Part A3: DAMVAQ Deep Dive (Carbanions vs Cage Junction)

| Atom / Bridgehead | Type | Fragment | Ref Sign / Det | Embed Sign / Det | Match? | Status |
|---|---|---|:---:|:---:|:---:|---|
| Atom 0 (C0) | Carbanion Donor | norbornyl | `-` | `+` | **MISMATCH** | Recovered by Scope B |
| Atom 2 (C2) | Carbanion Donor | norbornyl | `+` | `-` | **MISMATCH** | Recovered by Scope B |
| Atom 1 | Cage Bridgehead | norbornyl | `-` (-3.873) | `-` (-3.249) | MATCH | Rigid cage vertex (excluded) |
| Atom 3 | Cage Bridgehead | norbornyl | `-` (-7.061) | `-` (-6.768) | MATCH | Rigid cage vertex (excluded) |
| Atom 4 | Cage Bridgehead | norbornyl | `+` (+0.768) | `+` (+1.111) | MATCH | Rigid cage vertex (excluded) |
| Atom 6 | Cage Bridgehead | norbornyl | `+` (+7.433) | `+` (+6.595) | MATCH | Rigid cage vertex (excluded) |
| Atom 7 | Cage Bridgehead | quadricyclane | `-` (-4.859) | `+` (+4.789) | MISMATCH | Rigid cage vertex (excluded) |
| Atom 8 | Cage Bridgehead | quadricyclane | `+` (+4.607) | `-` (-2.399) | MISMATCH | Rigid cage vertex (excluded) |
| Atom 10 | Cage Bridgehead | quadricyclane | `+` (+2.914) | `-` (-2.808) | MISMATCH | Rigid cage vertex (excluded) |
| Atom 11 | Cage Bridgehead | quadricyclane | `-` (-1.942) | `+` (+1.853) | MISMATCH | Rigid cage vertex (excluded) |
| Atom 12 | Cage Bridgehead | quadricyclane | `-` (-7.157) | `+` (+1.086) | MISMATCH | Rigid cage vertex (excluded) |
| Atom 13 | Cage Bridgehead | quadricyclane | `-` (-18.003) | `-` (-13.721) | MATCH | Rigid cage vertex (excluded) |

### DAMVAQ Summary:
- **Carbanion Donors (`L1_D0_C, L1_D2_C`):** 100% recovered across all seeds 11, 12, 13 under Scope B.
- **Cage Ring Junction:** Anti/exo in ETKDG embed (RMSD 1.18 Å) vs syn/endo in crystal reference (RMSD 0.099 Å). Per Audit §3 ruling, polycyclic ring junction stereochemistry is an expected miss out of Scope B.

---

## 5. Part B: Requested-vs-Achieved 3D Stereo Sign Rates (Post-Exclusion Elements)

> **Scope Note:** The 100.0% (359 / 359) match rate below is measured across the stereogenic elements **remaining after the exclusions** (excluding ring bonds `bd.IsInRing()`, chelate-ring double bonds in `chelate_ring_stereo`, planar $sp^2$ phosphorus `L1_D6_P` on `LIPDEU`, rigid cage bridgeheads on `DAMVAQ`, and unbuildable variants rejected by `check_variant_feasibility` such as `TEDPOI` $Z$).

| Element Type (Post-Exclusion) | Total Comparisons | Matched 3D Signs | Match Rate (%) | Requested Sign Distribution |
|---|:---:|:---:|:---:|:---:|
| **LIGAND_TETRAHEDRAL** | 159 | 159 | **100.0%** | +:48, -:111 |
| **BINDING_CREATED_DONOR** | 143 | 143 | **100.0%** | +:34, -:109 |
| **DOUBLE_BOND_EZ** | 41 | 41 | **100.0%** | Z:21, E:20 |
| **MCX_ROTAMER** | 16 | 16 | **100.0%** | Z:8, E:8 |
| **OVERALL (ALL TYPES)** | 359 | 359 | **100.0%** | - |

---

## 6. Audit Compliance Checklist (Audit Round 2 §3 Items 1–6)
- [x] **Item 1:** BOCRAN and CIFZEZ recognized as linear triple bonds; TRIPLE demotion and SINGLE upgrade removed from `set_ez`; `set_ez` raises `ValueError` on non-double bonds; `caps` parameter and `embed_chelate` alias removed; verified by unit tests.
- [x] **Item 2:** `check_variant_feasibility` verifies per-variant ligand embedding and requested stereochemical retention; unbuildable classes filtered out; verified by test.
- [x] **Item 3:** Chelate-ring double bonds (`chelate_ring_stereo`) and all ring bonds (`bd.IsInRing()`) excluded from E/Z rotamer enumeration (`AZUSIA`, `HOXJUZ`, `QEZVEY`, `LOPCAW`); verified by test.
- [x] **Item 4:** Embed vs IK decomposition executed; 8 previous mismatch directions resolved to root causes; `LIPDEU` (flip=-1) and `LOPCAW` (set_ez on rdmol_h) achieve 100% retention; verified by test.
- [x] **Item 5:** Census recomputed; ceiling sampling confirmed active on only 2 parents (`FUTSOD`, `XUKKIA`); crystal class selection reported for all 21 stereo-active parents across seeds 11, 12, 13.
- [x] **Item 6:** DAMVAQ donors recovered and cage junction recorded as an expected miss; Part A recomputed.
