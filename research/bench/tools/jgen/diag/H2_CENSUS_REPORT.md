# Package H-2 embed census and light gates — checkpoint

Date: 2026-10-30. CPU only, no Colab spend. The census covers the frozen calibration panel plus the named tail panel (the tail panel's documented val members are allowed for development). It does not use reserve or locked sets.

## Execution and artifacts

- 130 parent/panel entries: 105 cal entries plus the 10 named tail cases, with five tail cases also in cal. 124 unique parents had usable GFN2 geometry; NEPNEF was invalid in both panel appearances.
- xTB was required and resolved. NEPNEF's highly charged ligand failed to produce `xtbopt.xyz`; the builder retained MMFF geometry. Those rows are marked `gfn2_fallback` and excluded from geometry counts. No other GFN2 fallback was observed.
- LOPCAW logged a SMILES alignment failure and used the atom-mapping heuristic (16 automorphisms). It is retained with an ambiguity marker; its bond-level results should not be treated as uniquely atom-labelled.
- Parent JSON, complete solver step-0 row measurements, per-parent ring torsions, and worker checkpoints are under `research/data/jgen/campaign/H/h2_census/` (ignored data output, retained in this worktree).

## Defect census

| Class | Affected valid parents | Evidence |
|---|---:|---|
| Haptic anionic donor groups skipped by the current kappa-only Li cap | 17 / 124; 26 groups | Cp/allyl-like multi-atom donor groups with net formal charge −1; parent list is in `haptic_anions` in each JSON. NUWFAM, LOPJEF, and QUTWOT are included. |
| E/Z variants on a bond whose atoms are both inside a recorded η≥3 donor group | 0 | 26 E/Z variants were enumerated; none are inside a recorded ≥3-atom donor group. NUWFAM's known localized allyl variant is presently represented by the builder as an η² donor group, and its E/Z bond is not between two recorded donor atoms. The literal skip rule would not fix that example without a preceding hapticity/perception correction. |
| Internal ligand bond length differs from crystal by >0.1 Å | 3 parents for C–C (6 bonds), 2 for C–N (2 bonds) | Coordination bonds are excluded here. Metal–donor distances were tabulated separately and are mostly a coordination-site placement difference, not a frozen internal-bond defect. |
| Candidate pyramidalization (crystal angle sum ≥345°, embedded sum <345°) | 73 parents; 137 centres | Heuristic graph-class split: amide N 3 centres / 3 parents; aniline N 23 / 11; ring sp² N 82 / 45; other N 4 / 3; carbonyl C 1 / 1; aromatic C 11 / 8; other C 13 / 11. Parent counts overlap across centre classes. This is still a geometry-only candidate screen, not independently validated hybridization or a single defect class; no general planarity patch is proposed. |
| Fixed ring torsion mismatch | 17 parents with at least one ring torsion >30° from the crystal | 465 rings of at least five atoms were measured across valid parents. XOJHIN's eight-member COD ring has median absolute torsion difference 49.3° (maximum 166.3°) in the free-ligand embed. The per-ring JSON includes both torsion vectors. |
| Coordinated π bond length | XOJHIN | COD C=C embed length 1.342 Å versus crystal 1.375/1.396 Å. Resetting both C=C bonds on the C1 top-1 and evaluating one MACE single point changes energy by +0.67 kJ/mol, so this is not a material energy defect on that row; no surrogate refinement is proposed. |

The most frequent >0.1 Å metal–donor bond classes were N–Zn (11 parents), N–Ni (10), C–Ru (8), Cl–Ru (8), and C–Rh (6). Donor-type counts are in the per-bond `donor_class` field. The dominant classes are metal–kappa (123 parents) and metal–haptic (31); “unknown” contacts include incidental short contacts such as NUWFAM Zr···F and are not treated as donor bonds.

## Step-0 placement gate

For each valid parent, seed 11, 128 solver rows were captured at step 0. This is the post-Kabsch initialization, before the first optimizer update. The recorded displacement compares donor positions with `coord_site.position`; Kabsch initializes against assigned ideal direction vectors, so the earlier displacement and vector-error summaries do not measure residual placement error after Kabsch. They are not a valid basis for recommending `start_kabsch`.

The `start_kabsch` recommendation is withdrawn pending a metric computed against the assigned ideal directions after Kabsch, or a direct coverage comparison. H-2 does not implement or evaluate `start_kabsch`.

## MACE defect-cost ranking and limits

- The retained D-1 NUWFAM counterfactual remains the largest measured defect cost: resetting the three distorted bonds in R2+ top-1 lowered MACE by 70.96 kJ/mol. That defect is on the known E/Z variant. H-2c's attempted general conjugated-anion skip rule was tested and rejected by the H-2c/H-2d gates below.
- The XOJHIN coordinated-π bond reset was +0.67 kJ/mol and is below a material-cost threshold.
- Top-1 bond-reset probes on the exported QEZVEY and RIQDEB rows found no >0.1 Å internal bonds to reset (0 kJ/mol by this counterfactual). Several other haptic-anion census parents do not have retained top-1 XYZ in the G-3c0 exports, so no MACE reset cost is claimed for the full 17-parent class.
- The full panel has only three parents with >0.1 Å internal C–C deviations and two with C–N deviations; none of those five has a retained C1 top-1 XYZ export in the campaign. The MACE rank for the generic bond-check class is therefore unresolved, not zero.

## H-2 fixes and light gates

- Added off-by-default `embed_haptic_li_cap`. During GFN2 refinement it adds one temporary Li per negatively charged multi-atom donor group, with the Li 1.9 Å from the group centroid along the fragment-plane normal for groups of at least three atoms. For two-atom groups, it uses the centroid direction away from the remaining ligand. Li is removed after refinement and the original ligand frame is restored. If donor-group inspection or cap placement fails, it warns and continues without haptic caps. The change is general across donor groups; it does not alter the ETKDG/MMFF cap path.
- H-2c temporarily introduced `ez_skip_conjugated_anion`, but the NUWFAM gate did not support retaining it: the flag-on MACE ΔE was +491.7 kJ/mol versus +184.6 kJ/mol for flag-off top-1, and the flag-on allyl C1 angle sum was 283.46°. H-2d tested the haptic Li cap with E/Z enumeration restored; its allyl C1 threshold also failed. The temporary E/Z flag and its profile entry have been removed. The H-2c census found 125 unique parents / 130 entries, one affected parent (NUWFAM, cal only), and no scan failures. Historical H-2c output remains available for audit.
- `embed_bond_check` was already present and off by default. It compares post-GFN2 lengths with pre-GFN2 embed lengths; it does not enforce reference bond lengths. The EPUWAQ/WIBHOH flag comparison recorded three ligand checks per parent, none triggered, and identical arm geometry. WIBHOH retains a 0.115 Å C–N embed/crystal deviation. A reference-length guard is **not built**; the existing safeguard does not satisfy that requirement.

The corrected haptic Li gate evaluates each solver-placed top-1 complex against its crystal geometry in the same generated atom order, with matching element sequence, atom count, H count, charge, and spin; no Li remains in either scored shell. The MACE single-point embed-minus-crystal values (eV) were QEZVEY 2.326 off / 1.844 on, RIQDEB 0.396 / 0.495, and XEVDAE 2.222 / 1.545. These full-complex geometry differences are not an isolated defect cost or a fold-in claim. Internal bond maxima (off/on) were 0.0418/0.0418, 0.0886/0.0889, and 0.1019/0.1016 Å respectively; XEVDAE has one bond above 0.1 Å in both arms. No GFN2 fallback occurred. The JSON files contain the mapping, atom counts, charges, spin, absolute energies, bond metrics, and flag profile.

Default flag-off fingerprint remains `6cc586e17ab1`. `embed_haptic_li_cap=true` yields `2ef36ca01e9c`; `embed_bond_check=true` yields `6f76e532185e`. No GFN2 fallback occurred on these light-gate parents. Gate inputs and outputs are in `research/data/jgen/campaign/H/h2_census/gate_*.json`; H-2c NUWFAM outputs are `h2c_nuwfam_off.json` and `h2c_nuwfam_on.json`.

Coordinated-π length gets no patch because the measured reset cost is small; the census and proposal above are the review item. No individual-complex spot fix is proposed.

## H-2c NUWFAM conjugated-anion gate (historical; rule removed)

The paired CPU gate used seed 11, embed seed 42, 250 IK steps, and MACE-Polar `polar-1-s`; xTB resolved, with no GFN2 fallback. Each returned full-complex row had 68 atoms (18 H), charge 0, spin 1, and no Li cap. C0/C1 below denote the two E/Z combinations (variant 0 / variant 1) in the flag-off run; flag-on leaves one unspecialized combination. Results are in `h2c_nuwfam_off.json` and `h2c_nuwfam_on.json`.

| Geometry | C–F (Å) | C–C (Å) | C–B (Å) | F height from allyl-C3 plane (Å) | F position elevation (°) | C–F angle to plane (°) | allyl C1/C2/C3 angle sums (°) | C8 angle sum (°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Crystal | 1.401 | 1.354 | 1.651 | 2.561 | 46.34 | 18.36 | 332.15 / 359.95 / 359.99 | 359.95 |
| Embedded variant 0, flag off | 1.354 | 1.383 | 1.729 | 0.427 | 3.64 | 74.95 | 359.99 / 359.95 / 359.38 | 359.99 |
| Embedded variant 1, flag off | 1.353 | 1.494 | 1.657 | 2.675 | 30.93 | 62.52 | 359.96 / 360.00 / 330.27 | 359.99 |
| Embedded, flag on | 1.351 | 1.382 | 1.747 | 0.179 | 2.21 | 28.87 | 359.78 / 359.89 / 359.74 | 360.00 |
| Flag-off variant-0 top-1 | 1.354 | 1.383 | 1.729 | 1.759 | 20.28 | 52.83 | 329.52 / 357.50 / 359.38 | 357.90 |
| Flag-off variant-1 top-1 | 1.693 | 1.494 | 1.657 | 3.582 | 45.54 | 47.35 | 340.27 / 336.63 / 330.27 | 325.57 |
| Flag-on top-1 | 1.351 | 1.382 | 1.747 | 2.430 | 31.36 | 3.71 | 283.46 / 344.34 / 359.74 | 360.00 |

The flag-off retained pool's variant-0 top-1 had score 0.3049 and MACE ΔE vs crystal +184.6 kJ/mol. Its best retained variant-1 row (rank 7) had score 0.1573 and +757.2 kJ/mol; six variant-1 rows remained, spanning +597.9 to +794.7 kJ/mol. With the flag on, the single top-1 had score 0.4686 and +491.7 kJ/mol, and no variant-1 rows remained. This is a mixed gate result: the rule removes the high-energy variant-1 rows, but the on top-1's MACE ΔE is +307.1 kJ/mol above the off top-1. The E/Z flag does not establish a MACE improvement on this seed.

For the retained flag-off variant-1 top-1, the F-only internal-coordinate counterfactual transferred the crystal C8–F bond direction into the generated allyl-C3 frame, keeping C8 and the current C–F length fixed. It increased MACE by 58.9 kJ/mol. Resetting only C–F, C–C and C–B lengths to their crystal values from the original row lowered MACE by 67.1 kJ/mol. Thus this isolated F-direction edit did not recover the expected large cost; neither requested single-factor reset explains most of the old +484.3 kJ/mol regression. The new seed-11 top variant gap was +572.6 kJ/mol (variant 1 minus variant 0). The result is reported as measured, including the mismatch with the prior expectation; no further tuning was done.

## H-2d NUWFAM haptic-cap gate

The paired CPU gate used seed 11, embed seed 42, 250 IK steps, and MACE-Polar `polar-1-s`, with `embed_haptic_li_cap=true` and E/Z enumeration restored. xTB resolved; no GFN2 fallback or census scan failures occurred. The full complex had 68 atoms (18 H), charge 0, spin 1, and no Li cap leaked into the scored shell. The raw embed and returned top-1 geometries are compared with the crystal using the same atom mapping and composition. Results are in `research/data/jgen/campaign/H/h2_census/h2d_nuwfam_haptic_cap.json`; the reproducible gate is `h2d_nuwfam_haptic_cap_gate.py`.

| Geometry | C–F (Å) | C–C (Å) | C–B (Å) | F height from allyl-C3 plane (Å) | allyl C1/C2/C3 angle sums (°) | C8 angle sum (°) |
|---|---:|---:|---:|---:|---:|---:|
| Crystal | 1.401 | 1.354 | 1.651 | 2.561 | 332.15 / 359.95 / 359.99 | 359.95 |
| Embedded variant 1 | 1.353 | 1.348 | 1.670 | 0.194 | 332.35 / 360.00 / 359.95 | 360.00 |
| Variant-1 top-1 | 1.344 | 1.348 | 1.670 | 2.430 | 327.56 / 357.25 / 359.95 | 359.99 |

The 340° allyl angle-sum cutoff was an auditor error: NUWFAM crystal allyl C1 is 332.15°, so the embedded value 332.35° matches the crystal. C–F, C–C, and C–B also match closely; C2, C3, and C8 remain near 360°. The distorted uncapped variant-1 embed (C–F 1.69 Å, C8 angle sum about 326°) is gone. Variant-1 top-1 was rank 1 and overall top-1 (score 0.44012), with MACE ΔE vs crystal +171.386 kJ/mol, versus +184.6 kJ/mol with the cap off. **H-2d passes.** The remaining F tilt toward Zr (crystal F 2.56 Å out of the allyl plane, generated geometries about 0.2–0.4 Å) reflects an omitted secondary Zr···F contact, not an embed defect; it is outside Package H and is recorded as a known input-class case.

## Honest failures / caveats

NEPNEF is excluded from geometry summaries because its GFN2 refinement fell back to MMFF. LOPCAW's heuristic mapping is ambiguous. The ring census uses graph cycle-basis cycles and a symmetry-agnostic atom map; the ring torsion vectors are provided so the reviewer can inspect those cases. Colab spend: 0.
