# T1 report — haptic typing and far-field census

Date: 2026-11-02
Worktree: `jgen/t-typing`, based on master `347496f` (`273f2ccbf892`)
Typing implementation commits: `0f06960` and `4e767bd`
Fingerprint with both T1 flags off: `273f2ccbf892`
Compute: local CPU, three worker processes; no VM or GPU use; Colab spend: 0 units. No refit, constraints DB write, or solve was run.

## Provenance and extraction

All 34 tmQM X1–X3 and LnQM shards are represented in each setting: 9 X1, 10 X2, 10 X3, and 5 LNf. Flags-off processed 26 shards in 1,605 s and reused eight completed pilot/partial-pass shards. The gate-only pass processed all 34 in 2,223 s; the combined pass processed all 34 in 2,035 s. All passes recorded zero molecule/parser errors.

| Setting | Q1 rows | Q2 rows | QO rows | S_orient rows | Face rows |
|---|---:|---:|---:|---:|---:|
| Off | 615,779 | 1,343,641 | 531,478 | 702,963 | 40,671 |
| Gate | 614,017 | 1,335,546 | 529,716 | 690,831 | 36,490 |
| Both | 614,017 | 1,335,546 | 529,716 | 690,831 | 36,490 |

The flags-off archive identity check used `(csd_code, k, bit-identical raw_face)`. It matched every archived haptic row: X1 16,378/16,378, X2 16,657/16,657, X3 5,837/5,837, and LNf 1,706/1,706. The extractor also produced 46, 39, and 8 additional rows in X1, X2, and X3 respectively (0 in LNf); these are extractor drift, not unmatched archive rows.

## Face census and bond gate

The off pass had 17,057 η² faces. The gate removes a face when its largest `|M−A|/(r_cov(M)+r_cov(A))` exceeds 1.21. It removed 4,181 faces and retained 12,876. The pair categories below combine all pairs other than C=C, C=N, C=O, and C=S as `other`. Ratios are quantiles of each η² face's maximum atom ratio.

| Pair | η² faces | Removed | Removed faces with declared κ donor | Declared donor atoms on removed faces | r p1 / p5 / p25 / p50 / p75 / p95 / p99 |
|---|---:|---:|---:|---:|---|
| C=C | 9,303 | 988 | 402 | 469 | 0.8538 / 0.9105 / 0.9811 / 1.0031 / 1.0476 / 1.4435 / 1.6419 |
| C=N | 3,405 | 1,102 | 473 | 538 | 0.8798 / 0.9348 / 1.0535 / 1.1415 / 1.2712 / 1.6071 / 1.7136 |
| C=O | 2,549 | 668 | 83 | 103 | 0.8734 / 0.8972 / 1.1201 / 1.1841 / 1.2123 / 1.3296 / 1.4515 |
| C=S | 1,459 | 1,356 | 1,230 | 1,241 | 0.9570 / 1.0534 / 1.2947 / 1.3386 / 1.4256 / 1.5107 / 1.6203 |
| Other | 341 | 67 | 65 | 68 | 0.8371 / 0.8613 / 0.9350 / 1.0007 / 1.1459 / 1.4618 / 1.6043 |
| **Total** | **17,057** | **4,181** | **2,253** | **2,419** | — |

Unchanged larger-face counts were η³ 3,003; η⁴ 1,066; η⁵ 15,539; η⁶ 3,915; η⁷ 77; η⁸ 14. For η³+ faces, maximum-ratio quantiles p1/p5/p25/p50/p75/p95/p99 were 0.9727/0.9932/1.0199/1.0433/1.0741/1.2163/1.5166. `hapto_label_canon` does not alter face detection, so the Both face census equals Gate.

## Fit-table key changes

Every key crossing the directive's thresholds (appears/disappears, absolute change ≥20 rows, or relative change >10%) is in [T1_KEY_COUNT_CHANGES.csv](../../../../bench/jgen_campaign/T1_KEY_COUNT_CHANGES.csv). The file has 83,765 key-setting rows.

| Quantity | Gate keys listed | Both keys listed | Distinct keys off / gate / both |
|---|---:|---:|---:|
| Q1 | 9,090 | 11,081 | 100,366 / 99,305 / 99,375 |
| Q2 | 22,376 | 26,320 | 182,460 / 179,033 / 179,385 |
| S_orient | 6,438 | 8,460 | 41,686 / 40,309 / 39,249 |

The η² rule is applied in `key_builder._site_subclass` for Q1, Q2, and S_orient keys. When enabled, the display `CoordinationSite.label` uses the same rule in `donor_classifier.classify`; per-donor `DonorLabel`s remain unchanged.

| NUWFAM key family | Off | Gate | Both |
|---|---:|---:|---:|
| `(Zr,1,4,C_haptic,trigonal)` rows | 10 | 10 | 10 |
| `(Zr,1,4,C_haptic)` rows | 202 | 196 | 462 |
| Baseline rows in 2.66–2.92 Å | 15 | 9 survive | 9 of the 15 survive; 10 total rows in the Both family after one row enters through relabeling |

The directive refers to 16 baseline rows in that distance interval. Both the archived Q1 observations and the complete flags-off re-extraction contain 15; this count discrepancy is reported as observed. The tenth Both-setting row entering that interval is LINWUE at 2.7059 Å.

## Parent changes

Only two eligible cal/tail parents change donor set or labels. Coordination number and builder geometry label are unchanged for both.

| Parent | Changed fields | Off → Both donor indices | Result |
|---|---|---|---|
| MIDQEY | Donor set, site labels | `[2,3,4,5,8,13,28]` → `[2,3,4,5,8]` | Gate removes the false η² contact and its two face atoms from the donor set. |
| UMUBAF | Donor set, site labels | `[2,3,32,33,37]` → `[2,3,32,33]` | Gate removes the false η² contact and its added donor atom. |

Known artefacts in the cal/tail scope: MIDQEY and UMUBAF change as above; NUHPEL and NUWFAM are present and unchanged by parent classification. YUCZUS and CAFWIQ are outside the eligible parent set (cal/tail minus locked exclusions), so no parent comparison is reported for them.

## Far-field mixture census

The census read `data/db/constraints_2026-11-01.json` only. It selected components with weight <0.15, mean outside the main component's mean ±2 sd, and sd <0.5 of the main sd. The complete 46,420-component table, including key, n, weight, mean, sd, main-sd distance, source-row count, and removed-face shares, is [T1_FARFIELD_COMPONENTS.csv](../../../../bench/jgen_campaign/T1_FARFIELD_COMPONENTS.csv).

| Quantity | Listed components | Inherited (`src=adapted`) | Components with at least one archived source row |
|---|---:|---:|---:|
| Q1 | 5,333 | 5,018 | 650 |
| Q2 | 41,087 | 40,611 | 4,917 |
| **Total** | **46,420** | **45,629** | **5,567** |

The DB marks adapted entries but does not store their back-off parent key; the component table says `not stored` for those entries. Q1 source windows are in Å. Q2 mixtures are fit in cosine(angle) space, so the source-row join converts archived angles to cosine before applying each component's ±2 sd window. Across component-window joins (rows may feed multiple pooled components), Q1 has 85,511 source-row entries and 398 confirmed removed-face matches. Q2 has 166,791 entries, 2,704 confirmed matches, and 4,754 additional possible matches where repeated subclasses make the pair-site identity ambiguous. Per-component confirmed shares and possible upper bounds are in the CSV.

## Checks and limits

`pytest -q tests/test_typing_t1.py`: 5 passed. The tests confirm both flags default off, the production fingerprint, exact flags-off classification hashes for IYUKOC/PENHEY/UMUBAF, C=S κ-donor preservation, bound/contact C=C handling, and canonical C=N labels in either atom order.

The directive's §2 forbids solves, while the repository's standard `golden.py check` recomputes seven solver cases and therefore performs CPU solves. That check was not run; no substitute solve was made. The focused default fingerprint and classification-output checks above passed. This is the sole unrun requested verification at the checkpoint.


## Amendment T1b: κ²-chelate exclusion and fitted-component diversity

The new default-off `hapto_chelate_exclusion` rejects two-atom `{C,X}` candidate faces when the face carbon is bonded to at least two declared/SMILES κ donors in the same ligand; the donors remain κ. The Zn κ²-acetate and dithiocarbamate tests reject the face and preserve both donors. The η²-CO₂ test retains its face with one bound oxygen. The focused suite passed 8 tests.

The third setting used gate + label canonicalization + chelate exclusion. All 34 shards were re-extracted: the X1s000 pilot took 191.2 s and the remaining 33 took 2,047.7 s on three workers (2,238.9 s total). There were zero errors. Totals were 614,126 Q1 rows, 1,336,222 Q2 rows, 529,825 QO rows, 686,523 S_orient rows, and 34,792 face rows. No refit, database write, solve, GPU, or VM was used.

| η² face pair | Both | T1b | Excluded |
|---|---:|---:|---:|
| C=N | 2,303 | 1,160 | 1,143 |
| C=O | 1,881 | 1,365 | 516 |
| C=S | 103 | 68 | 35 |
| C=P | 59 | 55 | 4 |
| **Total** | **12,876** | **11,178** | **1,698** |

The complete pair-type, metal, and pair-by-metal tables are [T1B_FACE_DELTA_PAIR.csv](../../../../bench/jgen_campaign/T1B_FACE_DELTA_PAIR.csv), [T1B_FACE_DELTA_METAL.csv](../../../../bench/jgen_campaign/T1B_FACE_DELTA_METAL.csv), and [T1B_FACE_DELTA_PAIR_METAL.csv](../../../../bench/jgen_campaign/T1B_FACE_DELTA_PAIR_METAL.csv). Largest exclusions by metal were Zr 271, Ti 221, Cd 136, Y 132, Zn 104, Ir 97, and Ru 92.

The key-change list is restricted to `|Δ| ≥ 20` rows versus Both and contains 297 keys: 35 Q1, 120 Q2, and 142 S_orient. It is in [T1B_KEY_COUNT_CHANGES.csv](../../../../bench/jgen_campaign/T1B_KEY_COUNT_CHANGES.csv). For the required Zn key `('Zn',1,2,'O_carboxylate','octahedral','O_carboxylate')`, counts are Gate 810, Both 504, T1b 544. Thus T1b is 266 below Gate and outside both the specified 810±5 range and Gate±5. This expected restoration did not occur; the re-extraction observed 88 excluded Zn C=O faces and no cal/tail parent changed versus Both.

The far-field re-census includes only the 791 fitted, non-adapted narrow components (Q1 315; Q2 476), matching the requested counts. All 791 had archived source rows; total component-window source-row matches were 242,256 (windows may overlap). Per component, the median distinct CSD parent count was 41 (range 8–7,404), and the median share from the largest source parent was 3.48% (range 0.03–18.18%). Source metals among matched rows were led by Ni (21,370), Ru (17,550), Mo (16,600), Pd (15,036), and Co (14,609). The full component-level table, including source-row counts, distinct parents, largest-parent share, and metal counts, is [T1B_FITTED_COMPONENTS.csv](../../../../bench/jgen_campaign/T1B_FITTED_COMPONENTS.csv). No cal/tail classification changed versus Both.


## Amendment T1c: geometric bound test

Before changing the exclusion predicate, an exact gate-to-T1b Q1 join identified 109 η² C=O faces whose site moved from the Zn carboxylate key to `C_haptic`/`O_carboxylate`. The face oxygen was undeclared in all 109. Its Zn-scaled ratio had p0/p5/p25/p50/p75/p95/p100 = 1.0565/1.0722/1.0853/1.0969/1.1032/1.1258/1.1394; every face oxygen was within the 1.21 bound.

`hapto_chelate_exclusion` now counts a carbon neighbor as metal-bound if it is a declared/SMILES donor or its metal-scaled ratio is ≤1.21. When the chelate face is rejected, the qualifying neighbors remain κ donors. Tests cover an undeclared second acetate O at 2.2 Å (excluded and retained as κ) and at 3.2 Å (the κ¹ acetate face remains). `pytest -q tests/test_typing_t1.py`: 10 passed.

The corrected T1c extraction covered all 34 shards. The `X1s000` pilot took 196.9 s; the other 33 took 2,084.1 s on three workers, 2,281.0 s total. All shards had zero molecule/parser errors. The re-extraction contains 614,870 Q1 rows, 1,339,346 Q2 rows, 530,536 QO rows, 685,195 S_orient rows, and 33,788 face rows. No refit, database write, or solve was run.

| η² face pair | T1b | T1c | Excluded |
|---|---:|---:|---:|
| C=O | 1,365 | 406 | 959 |
| C=N | 1,160 | 1,130 | 30 |
| C=S | 68 | 59 | 9 |
| C=P | 55 | 51 | 4 |
| B=C | 16 | 14 | 2 |
| **Total** | **11,178** | **10,174** | **1,004** |

By metal, largest exclusions were Zn 643, Cd 114, Ag 53, Pd 28, Cu 25, and Ni 19. Full pair and metal tables are [T1C_FACE_DELTA_PAIR.csv](../../../../bench/jgen_campaign/T1C_FACE_DELTA_PAIR.csv) and [T1C_FACE_DELTA_METAL.csv](../../../../bench/jgen_campaign/T1C_FACE_DELTA_METAL.csv); pair-by-metal detail is [T1C_FACE_DELTA_PAIR_METAL.csv](../../../../bench/jgen_campaign/T1C_FACE_DELTA_PAIR_METAL.csv). The Q1/Q2/S_orient list has 265 keys with `|Δ| ≥ 20` versus T1b (Q1 63, Q2 118, S_orient 84) in [T1C_KEY_COUNT_CHANGES.csv](../../../../bench/jgen_campaign/T1C_KEY_COUNT_CHANGES.csv). No cal/tail parent changed.

The Zn key `('Zn',1,2,'O_carboxylate','octahedral','O_carboxylate')` is 1,044 under T1c, versus 544 under T1b and 810 under gate-only. It overshoots the expected 810±5 by 234 rows; the stated expectation was not met.

## Amendment T1d: separate face exclusion and heteroatom promotion

The classifier now separates `hapto_chelate_exclusion` (reject a two-atom face when its carbon has at least two declared or geometrically bound neighbors; preserve declared donor status) from default-off `donor_promote_geometric` (add undeclared **heteroatom** neighbors as κ donors only on a rejected chelate face). The promotion ratio and its scope, `rejected_chelate_face_neighbors`, are registered settings; the defaults are 1.21 and that narrow scope. Carbon neighbors can count in the face rule, but are not promoted as donors.

The reproducible extraction entry point is [src/fit/typing_reextract.py](../../../../../src/fit/typing_reextract.py); the former bench path is a thin compatibility wrapper. `src/fit/__init__.py` now lazy-loads quantity modules, and the runner sets its per-mode profile overrides before workers start. An initial T1d extraction was discarded after it exposed that eager `fit` imports loaded classifier flags before worker overrides. A second pass also exposed that carbon neighbors were included in promotion diagnostics; the promotion predicate was narrowed to heteroatoms and the complete promotion arm was rerun. The reported numbers below are from the corrected, final runs. Every numerical worker was pinned to one thread.

Both final settings covered 34/34 tmQM X1–X3 and LnQM shards, 125,810 parent records, and had zero molecule/parser errors. Face-only elapsed time was 1,974 s; face exclusion plus heteroatom promotion elapsed time was 2,107 s (three workers each, one numerical thread each). No refit, database write, or solve was run.

| Setting | Q1 rows | Q2 rows | QO rows | S_orient rows | η² faces |
|---|---:|---:|---:|---:|---:|
| Gate + canon + chelate exclusion, no promotion | 613,201 | 1,332,073 | 528,900 | 683,559 | 10,174 |
| All four flags, with heteroatom promotion | 614,847 | 1,339,193 | 530,532 | 685,191 | 10,174 |

The face census is identical to T1c in both settings: zero η²-face delta for every pair and metal. Pair, metal, and pair-by-metal tables are [T1D_FACE_DELTA_PAIR.csv](../../../../bench/jgen_campaign/T1D_FACE_DELTA_PAIR.csv), [T1D_FACE_DELTA_METAL.csv](../../../../bench/jgen_campaign/T1D_FACE_DELTA_METAL.csv), and [T1D_FACE_DELTA_PAIR_METAL.csv](../../../../bench/jgen_campaign/T1D_FACE_DELTA_PAIR_METAL.csv). The Zn carboxylate Q1 key is 394 without promotion and 1,044 with it, matching T1c. Promotion changes 324 keys by at least 20 rows: 81 Q1, 154 Q2, and 89 S_orient. The full list is [T1D_KEY_COUNT_CHANGES.csv](../../../../bench/jgen_campaign/T1D_KEY_COUNT_CHANGES.csv).

Promotion adds 1,646 heteroatom donors. Their metal-scaled ratio quantiles p0/p5/p25/p50/p75/p95/p100 are 0.8489/0.9853/1.0486/1.0860/1.1098/1.1690/1.2098; none exceeds the registered 1.21 threshold. The M–X distance quantiles are 1.787/1.982/2.043/2.089/2.240/2.575/3.490 Å. C=O is the largest pair (1,411 atoms), followed by C=S (122), C=N (90), and C=P (17); six additions are C=Si, C=B, or C=I. Full per-metal, per-pair, and per-metal-pair quantiles are in [T1D_PROMOTION_BY_METAL.csv](../../../../bench/jgen_campaign/T1D_PROMOTION_BY_METAL.csv), [T1D_PROMOTION_BY_PAIR.csv](../../../../bench/jgen_campaign/T1D_PROMOTION_BY_PAIR.csv), and [T1D_PROMOTION_BY_METAL_PAIR.csv](../../../../bench/jgen_campaign/T1D_PROMOTION_BY_METAL_PAIR.csv); the 1,646-atom ledger is [T1D_PROMOTION_ATOMS.csv](../../../../bench/jgen_campaign/T1D_PROMOTION_ATOMS.csv).

The promoted contacts are mostly within the bond gate, but their Q1 rows extend the upper tail of the key distributions. For the Zn carboxylate key, 650 rows are added: their median distance is 2.062 Å and p95 2.149 Å, versus 1.954 Å and 2.055 Å among the 394 rows present without promotion. 59.5% of the added rows exceed the no-promotion p95 and 4.2% exceed its maximum (2.157 Å; added maximum 2.373 Å). The 10 Q1 keys with the largest positive promotion additions, plus the Zn key, are tabulated against their no-promotion rows in [T1D_Q1_ADDED_ROW_DISTANCES.csv](../../../../bench/jgen_campaign/T1D_Q1_ADDED_ROW_DISTANCES.csv). Under T1d's rule, this is a longer tail, so `donor_promote_geometric` stays off for D1; the face-exclusion flag remains separate.

Across all source parents, 1,223 change under promotion (donor set, coordination number, site labels, and geometry label). Counts by metal are in [T1D_DBWIDE_PARENT_COUNTS_BY_METAL.csv](../../../../bench/jgen_campaign/T1D_DBWIDE_PARENT_COUNTS_BY_METAL.csv), with 10 examples in [T1D_PARENT_CHANGE_EXAMPLES.csv](../../../../bench/jgen_campaign/T1D_PARENT_CHANGE_EXAMPLES.csv). No eligible cal/tail parent changes. The locked exclusions were applied to that panel comparison.

`pytest -q tests/test_typing_t1.py`: 13 passed. The suite covers exclusion and promotion separately and together, including that carbon neighbors are not promoted. `compileall` and `git diff --check` passed. The no-solve rule was followed.
