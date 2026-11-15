# F-2 evaluation contract

Written before generating corrected-label or alternative-arm results (2026-10-28).

## Frozen-coordinate label

Compare production embed coordinates with bound coordinates over graph automorphisms of the heavy-atom ligand, retaining only mappings that preserve atom/bond attributes and specified stereochemistry. Select the mapping with the lowest heavy-atom RMS after rigid alignment; assign equivalent hydrogens locally by minimizing their bond/angle residuals. Report the number of automorphisms considered, whether the match cap was reached, and the selected mapping. Apply the mapping to all subsequent internal-coordinate calculations.

The primary frozen set is: all covalent bond lengths; all bond angles except pivots that are solver bend DOFs (hydrogen-containing angles reported separately); out-of-plane planarity at trigonal `sp2` centers only, using absolute displacement from the plane of the three heavy neighbors; tetrahedral/E-Z stereochemistry after production variant enumeration. Thresholds remain >0.15 Å, >15°, >20° absolute planarity-angle change, and stereo/E-Z mismatch. A ligand is bad only if every production E/Z variant is bad.

Do not put donor-pair distances, ring pucker, ring torsions, or improper values at non-planar centers in the primary bad label. Donor-pair distances are dropped from F-2 scoring because a torsion fit is not available in this embed-only comparison; ring torsions/pucker can change through ring joints. Symmetry swaps of equivalent hydrogens and equivalent substituents are removed by the selected graph automorphism.

Fit detector (b) ranges from aligned `surface_fit` references under the same coordinate selection; keep the fixed 0.5th–99.5th percentile rule and graph thresholds from the F-1 contract. Score (a), (b), and (c) with parent-clustered intervals. Report component counts for bonded anomalies, nonbonded clashes, and stereo loss.

## Comparison sets and arms

- Failure set: the scored QEZVEY ligand, HEDJOP donor ligand, all four malformed-nitro parents in their written form (NEPNEF, BOHWUT, KENMAU, SUXVEN), ZOLQIC ligand 1, and all bad ligand instances in the 20 worst cal/tail parents by the corrected label. Rank parents by bad-ligand count descending, then bad-ligand fraction descending, then code ascending. Deduplicate ligands.
- Good control: 40 corrected-label-good, scoreable cal/tail ligand instances (at least one frozen internal coordinate or assigned stereo), sampled by seed `20261029` after sorting by `(code, ligand_idx)`.
- Arms: production control; fixed dummy-metal GFN2 bound-state; donor-target-restrained GFN2; ETKDG→GFN-FF preoptimization→production GFN2; Li+ replacing the temporary H cap on formal anionic donors. If a ligand has no anionic donor, arm (e) is identical to production and is recorded as such.
- Keep production seed 42, donor indices, E/Z variants, capping rule and inputs identical. Clear the in-process GFN2 geometry cache before each ligand/arm timing so every arm includes its own cold setup and refinement. CPU only, maximum 3 workers; never alter source SMILES in-place.

For each ligand/arm, score the median frozen-coordinate residual normalized by its bad-label threshold (bond /0.15 Å, angle /15°, planarity /20°); use this as the mixed-unit non-DOF error. An arm failure receives a penalty error of 10 normalized units so failed rows cannot improve the median. Record arm failure rates and wall time including setup. The failure-set improvement is paired by ligand and parent-bootstrap. A method wins only if its failure-set median error is lower with a paired 95% CI below zero, its good-control paired 95% CI includes zero (no detectable worsening beyond the paired noise), and its median time per ligand across the combined failure and good-control sets is at most 2× production. If none meets all conditions, conclude “none wins.” This operationalizes the original contract before inspecting arm results.

## Remaining design-only item

Describe normalization of unambiguous resonance mis-writes only; do not implement it. Count eligible input instances by motif and cohort, and report ambiguity/exclusion rules.
