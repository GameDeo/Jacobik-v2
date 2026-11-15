# D2 60-pool rescore explanations — 2026-11-09

## Reproduction and deviation

`rescore_60_per_pool.json` is the completed run at the main-checkout path `research/data/d2/db_d2/rescore_60_per_pool.json`. Its six aggregate values match the earlier `rescore_60.json`: 60 pools; top-1 unchanged 43/60; median Spearman ρ 0.99326; minimum ρ 0.06708; p95 |Δscore| 0.92794; maximum |Δscore| 4.46228. The rerun took 78.4 s.

One complex's GFN2 embed optimization failed its initial SCF; the same input was retried once with `--etemp 1000` through the logged wrapper (`xtb_retries.log`). The retry completed and the aggregate reproduction matched. Two other deterministic ETKDG retry-exhaustion warnings were logged; they did not stop the rescore. An initial independent term-capture attempt without the explicit xTB wrapper failed on this same embed and was discarded. The successful term-capture pass used the retry wrapper, took 76.3 s, and did not write/replace the official JSON.

## Attribution method and limits

Term deltas below are `D2 DB score − production DB score` from the scorer's `rank_terms` breakdown, in summed ranking-score units (nats). For each pool, the table lists Spearman ρ, row count, whether top-1 changed, and the quantity deltas at the old top row / new top row / maximum absolute row delta. These are term-level values, not a judgment about quality.

Key schemas from the scorer are: Q1 donor-distance `(metal, spin, oxidation_state, donor_subclass, coordination_geometry, trans_partner)`; Q2 pair-angle `(coordination_geometry, pair_class, graph_distance, metal, donor_A_subclass, donor_B_subclass)`; Q3 orientation `(donor_subclass, metal, coordination_geometry, anion_co_donor_bucket)`; Q5 torsion `(type_j, type_k, type_i, type_l, ring_context)`. The captured term decomposition identifies which quantity/key family moved; it did not persist every concrete per-complex key tuple or run a typing-only / filter-only factorial. D2 changes both T1d extraction typing and fit filters, so where both could affect a key family, the paired rescore cannot isolate their separate causal shares.

The rescore compares two fitted DBs using the production scorer setup. It does **not** enable the D2 runtime `q6_physical_all` flag. Q6 deltas here are changes to the fitted Q6 term, not evidence about the physical-Q6 runtime flag. This matters for attribution: for the listed pools, observed movement comes from the changed DB; physical Q6 runtime is not the cause in this rescore.

## Low-ρ pools (ρ < 0.9)

| Pool | Rows | ρ | Top-1 changed; Δ top score | Main moved term family (old top / new top / max |Δ|) | Attribution |
|---|---:|---:|---|---|---|
| AJIXUN, seed 13 | 32 | 0.0671 | yes; −0.09717 | Q5 −0.74561 / −0.00502 / 0.77633 | Q5 torsion quartet/ring-context keys. D2 DB Q5 values account for the largest score movement; Q1–Q3 changes are ≤0.011 at the two top rows and Q6 max is 0.00182. This rescore does not identify whether filtering or T1d-related donor-site changes altered the applicable fitted entries. |
| AJIXUN, seed 11 | 32 | 0.1598 | yes; +0.01808 | Q5 −0.32656 / +0.07547 / 0.76865 | Same Q5 key family; Q1–Q3 top-row movement is small and Q6 max is 0.00181. Filtering vs donor-site contribution was not isolated. |
| AJIXUN, seed 12 | 32 | 0.4142 | yes; +0.02351 | Q5 −0.32532 / +0.04259 / 0.75984 | Same Q5 key family; Q1–Q3 top-row movement is small and Q6 max is 0.00181. Filtering vs donor-site contribution was not isolated. |
| AFOWIB, seed 11 | 80 | 0.4622 | yes; +0.00717 | Q5 −0.05499 / +0.01168 / 0.74991 | Q5 torsion quartet/ring-context keys dominate the pool's reordering. Q2 moves +0.00525/+0.00508 (max 0.00785); Q1 and Q6 are smaller. This rescore does not isolate filtering vs donor-site effects on the applicable entries. |
| AFOWIB, seed 12 | 80 | 0.5225 | yes; +0.01077 | Q5 −0.05341 / +0.01352 / 0.74928 | Same Q5 key family; Q2 top-row changes are about +0.0053 and max Q6 delta is 0.00070. Cause within the D2 DB changes was not isolated. |
| AFOWIB, seed 13 | 80 | 0.5603 | yes; +0.00556 | Q5 −0.05523 / +0.01128 / 0.71904 | Same Q5 key family; Q2 top-row changes are about +0.0053 and max Q6 delta is 0.00070. Cause within the D2 DB changes was not isolated. |
| ADUJOB, seed 12 | 96 | 0.8820 | yes; +0.12628 | Q1 +0.06348 / +0.16228 / 0.49496 | Q1 donor-distance keys dominate; Q2 is +0.00980/−0.02627 (max 0.05439), Q3 is small. The keys include donor subclasses; the combined DB rescore does not determine whether T1d retyping or fit filters account for the changed fitted entries. |
| ADUJOB, seed 13 | 96 | 0.8949 | yes; +0.10852 | Q1 +0.05044 / +0.18108 / 0.36773 | Same donor-distance keys; Q2 is +0.00631/−0.02590 (max 0.05462), Q3 is small. Typing and filtering are not isolated by this comparison. |

For AJIXUN and AFOWIB, the Q5 key schema is local torsion atom types plus a `free`/`chelate` ring context. The observed movement is on Q5 entries in the D2 DB. The paired rescore does not identify whether fit filters (1,626 charge-implausible and 360 locked parents excluded) or changes in donor-site typing affected those entries. For ADUJOB, the large Q1 movement is on donor-metal distance keys; the donor-subclass field changes the key family under T1d, while fit filters also changed the DB. The combined comparison cannot assign the movement to one cause. No claim is made about which DB produces better rankings.

## Other pools whose top-1 changed

These have ρ ≥ 0.9 but are required in the explanation because the top-1 row changed. Parenthesized term values are old top / new top / max absolute row delta.

| Pool | Rows | ρ | Δ top score | Term decomposition | Key family / cause trace |
|---|---:|---:|---:|---|---|
| ADUJOB, seed 11 | 96 | 0.9136 | +0.10146 | Q1 +0.03887 / +0.16336 / 0.44848; Q2 +0.00524 / −0.02702 / 0.05309; Q3 −0.00516 / −0.00383 / 0.03362 | Same donor-distance and donor-subclass key families as ADUJOB seed 12; the paired DB comparison does not separate T1d and filtering. |
| ADINOR, seed 11 | 32 | 0.9630 | −0.03503 | Q1 −0.00995 / −0.00954 / 0.01202; Q2 −0.02252 / −0.02170 / 0.02484; Q3 −0.00416 / −0.00376 / 0.00588 | Small Q1–Q3 shifts across donor-distance, pair-angle, and orientation key families; D2 DB typing/filter effects are not separable here. |
| ADINOR, seed 13 | 32 | 0.9663 | −0.03476 | Q1 −0.00938 / −0.00301 / 0.01084; Q2 −0.02171 / −0.02188 / 0.02612; Q3 −0.00394 / −0.00569 / 0.00619 | Same key families and limitation as seed 11. |
| AGAPEG, seed 11 | 142 | 0.9758 | −0.06651 | Q3 −0.15372 / +0.06145 / 3.79300; Q1 +0.11324 / +0.11432 / 1.69189; Q2 −0.08995 / −0.09465 / 0.31112 | Orientation and donor-distance/pair-angle key families. Q3 uses donor-subclass/metal/geometry/anion-bucket keys; this comparison does not separate typing from filtering. |
| AGAPEG, seed 12 | 136 | 0.9679 | −0.01145 | Q3 −0.17055 / +0.08728 / 3.11154; Q1 +0.11711 / +0.13578 / 1.80365; Q2 −0.08301 / +0.05245 / 0.32545 | Same key families and limitation as seed 11. |
| AGAPEG, seed 13 | 152 | 0.9642 | +0.07868 | Q3 −0.17066 / +0.08092 / 4.19903; Q1 +0.11785 / +0.16922 / 1.72128; Q2 −0.09335 / −0.02607 / 0.43153 | Same key families and limitation as seed 11. |
| AFAVUB, seed 11 | 128 | 0.9969 | +0.01024 | Q1 +0.01170 / +0.01301 / 0.79042; Q2 −0.02979 / −0.00845 / 0.13536; Q3 −0.00242 / +0.00041 / 0.03139 | Donor-distance keys carry the largest row-level change, with pair-angle keys secondary. Paired DB comparison does not separate typing from filtering. |
| AFAVUB, seed 12 | 128 | 0.9978 | −0.01204 | Q1 +0.01182 / +0.01043 / 0.68419; Q2 −0.03042 / −0.03856 / 0.13062; Q3 −0.00191 / −0.00206 / 0.02660 | Same key families and limitation as seed 11. |
| AFAVUB, seed 13 | 128 | 0.9972 | −0.00762 | Q1 +0.01156 / −0.00460 / 0.78586; Q2 −0.03138 / +0.00996 / 0.15682; Q3 −0.00135 / −0.00053 / 0.03328 | Same key families and limitation as seed 11. |

Other changed-top1 score differences in the low-rho table are already listed above. Across this changed-top1 set, Q4 has zero recorded delta; Q6 fitted-term movement is negligible (maximum 0.02564 in the captured pools), and the D2 physical-Q6 runtime flag was off. No per-pool “better/worse” interpretation is made here.

## Concrete lookup capture

A separate diagnostic at commit `6fe90b5` captured the actual requested and resolved keys for both fitted DBs, using the same 60 pools and one logged 1000 K xTB retry. Its six aggregate values exactly match `rescore_60.json`. The relevant requested/resolved keys are listed in `D2_RESCORE_KEYS.md`; complete captured fit records are in main-checkout `research/data/d2/db_d2/rescore_changed_keys.json` (SHA-256 `9f4b2d2ec8fd05453d74f3f47fb7ad4fe1f57b2b8f6c5237add4d7f3bb82a99d`). A changed fit record includes changes to metadata, so these are candidate contributing keys, not per-key score deltas. The capture does not separate typing from filtering.

## Charge-filter ablation for AJIXUN and AFOWIB Q5 keys

A targeted ablation (`61c4751`) refit the six captured Q5 L0 keys from the retained `d2-merge-full` typed observations. Locked-set parents remained excluded in both arms; no held-out results were evaluated. The arms differed only in whether charge-implausible parents were excluded. Evidence weights and the Q5 fit routine are the production fit pipeline's. Every charge-filtered reconstruction exactly matched the corresponding recorded D2 model fields (`type`, `model`, `means`, `kappas`, `weights`), which verifies the reconstruction for these six keys. Including charge-implausible observations changed each fit:

| Key | Rows with charge filter | Charge-implausible rows excluded | Filter changes fitted model |
|---|---:|---:|---|
| (`C_sp2`, `C_ar`, `C_sp2`, `C_ar`, `free_o0`) | 7,940 | 88 | yes |
| (`P_sp3`, `C_ar`, `C_sp3`, `C_ar`, `free_o0`) | 42,217 | 84 | yes |
| (`P_sp3`, `C_sp3`, `C_ar`, `H`, `free`) | 1,641 | 3 | yes |
| (`N_ar`, `C_sp3`, `C_ar`, `H`, `free`) | 8,895 | 132 | yes |
| (`N_sp2`, `N_sp3`, `C_ar`, `C_ar`, `chelate`) | 126 | 25 | yes |
| (`N_sp3`, `C_ar`, `N_sp2`, `C_ar`, `free_o0`) | 1,205 | 78 | yes |

The first three keys occur in AJIXUN; the latter three occur in AFOWIB. This establishes that the charge filter changes these Q5 fit entries, which are among the changed lookups behind their term-level movement. It does not isolate each entry's score contribution, quantify the total Q5 score effect of the filter, or establish the independent effect of T1d typing. Results with fitted model fields and counts are in `research/data/d2/db_d2/q5_charge_ablation.json` (SHA-256 `fac41dc2a8680fbb5e26165847c2d6f7de46b218d963d13761c1a0f42917aa70`).

## Per-key Q5 contributions

A separate score decomposition (`0e5667b`) computes each torsion's Q5 mixture contribution at ranking scale, groups identical keys, and checks that the per-key deltas sum to the `rank_terms` Q5 delta for every pool row (absolute tolerance 1e-4). It used the same pool rows and fitted DBs. It identifies a dominant concrete key in every AJIXUN and AFOWIB pool:

| Pool | Dominant Q5 key | Δ at old top | Δ at new top | Maximum absolute row Δ |
|---|---|---:|---:|---:|
| AJIXUN 11 | (`P_sp3`,`C_ar`,`C_sp3`,`C_ar`,`free_o0`) | -0.3266 | +0.0754 | 0.7689 |
| AJIXUN 12 | same | -0.3252 | +0.0428 | 0.7597 |
| AJIXUN 13 | same | -0.7460 | -0.0051 | 0.7768 |
| AFOWIB 11 | (`N_sp3`,`C_ar`,`N_sp2`,`C_ar`,`free_o0`) | -0.0550 | +0.0116 | 0.7499 |
| AFOWIB 12 | same | -0.0534 | +0.0135 | 0.7493 |
| AFOWIB 13 | same | -0.0552 | +0.0113 | 0.7191 |

Both dominant keys' fits are changed by the charge-sanity filter in the ablation above. This establishes a filter effect on those fits, not the amount of their score delta attributable solely to the filter or the independent typing effect. Full per-key values: `research/data/d2/db_d2/q5_key_contributions.json` (sha256 `3a5128d222439c6877457c089ba9e4e0d8eaeb1bb1923737cdcaa65144658792`). The first diagnostic attempt failed its sum check because the normalisation was missing and produced no accepted output; the corrected run passed. No xTB retry or fallback was logged for these two parents.
