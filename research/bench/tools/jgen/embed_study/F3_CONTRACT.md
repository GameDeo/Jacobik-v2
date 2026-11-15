# F-3 contract: embed bond guard and paired screen

Written after merging master `8dbe29c` and before inspecting F-3 footprint or paired-screen outputs (2026-10-28).

## Existing alternative-arm re-score

No new embedding runs. For each of the five arms, read the recorded F-2 per-ligand `bad` label and count transitions among the fixed 28-ligand failure set and the fixed 40-ligand good control. A failure ligand is repaired only if its production control label is bad and the alternative label is good. A control regression occurs only if production is good and the alternative is bad. Report case-level labels for QEZVEY, HEDJOP, ZOLQIC, the four malformed nitro ligand cases, and other named F-2 examples present in the fixed set.

## `embed_bond_check`

- Boolean experiment flag, default off. The off path must preserve the production fingerprint and be bit-identical for all complexes currently embedded successfully.
- Evaluate only after GFN2 refinement and before accepting refined coordinates. Trigger if any mapped bonded distance changes from the pre-refinement coordinates by more than 0.25 Å **or** the refined distance exceeds 1.25 times the sum of covalent radii.
- On trigger, use the pre-refinement (MMFF) coordinates as the fallback; the ligand must not be rejected. Retain diagnostics for trigger reason and offending bond(s).
- Footprint denominator: unique parents in cal, tail, and the existing 300-ligand surface-fit sample, excluding val, reserve, and locked parents. Report both trigger ligand counts and unique parent counts by cohort; deduplicate parents shared between cal/tail/sample when reporting overall total.
- Validation: run the golden check with the flag off and compare production baseline output bit-for-bit; run the trope suite with flag off and require no parent other than ZOLQIC to change from the pre-fix baseline. Test the ZOLQIC recovery separately with its production solve and require no exception and a defined caller result.

## Paired footprint screen

Commit this contract before producing screen outputs. Run each footprint parent as a paired production control (flag off) and experimental arm (flag on), with identical inputs, seeds, profile, hardware, and candidate budget. On the same top-1 row from each solve, report top-1 strain and `ΔE_unrel`; pair differences by parent. Keep CPU-only, maximum three concurrent processes, and no GPU. Report all failures and do not tune the threshold or parent cohort after seeing the paired results.

If the available local artifacts do not contain the exact matched control inputs or metrics needed for this paired screen, record that part as **not done** with the blocking artifact/compute limit; do not substitute an unpaired or post-hoc cohort.

## Stereo-enumeration design

Count unspecified tetrahedral stereocenters from input ligand SMILES for each ligand instance in cal, tail, and the surface-fit sample. Report the distribution of `k` and the naive `2^k` variant count by ligand and cohort. Design-only proposal: cap full enumeration at 8 centers (256 assignments); above the cap, retain explicit input assignments, enumerate assignments involving metal donor centers first, then draw a deterministic, stratified sample of at most 256 remaining assignments using a stable hash of parent code, ligand index, and assignment. Do not implement stereo enumeration in F-3.
