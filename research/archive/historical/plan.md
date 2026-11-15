# Representative, matched, data-derived NCI

## Summary and fixed choices

The adopted `u1` profile remains the incumbent during research: repulsive-only Q6 plus calibrated HB-NCI, with π–π and C–H–π disabled. Freeze Q6 and every non-NCI term while evaluating the new candidate.

Use the baselines for distinct jobs:

- **Historical N1 (`u1c+rep`)** is the project benchmark to document and compare against.
- **`u1_pure_rep`** is the control used to generate poses and learn the incremental attraction. It has no NCI or legacy Q6 attractions, so the new term cannot count an old attraction twice.
- **Adopted `u1+nci_hb`** is the incumbent candidate comparator. Report results against all three.

The recorded dev80 #1 mean MACE64 errors are 87.98 kJ/mol for N1, 87.03 for `u1_pure_rep`, and 86.34 for adopted `u1+nci_hb`. These are the existing reference points, not guarantees about new data. [HANDOFF.md](HANDOFF.md)

The CTX1 pilot does not establish a production method. Its 96-parent cohort was selected from strict positive-contact cases using a hand-chosen H···A ≤2.50 Å and D–H···A ≥150° rule. The pilot produced a favorable mean best-pool MACE change, but increased strict-contact counts while reducing the share of occurrences that were exact reference pairs; 5 of 48 validation parents worsened, with a largest recorded regression of 18.66 kJ/mol. Treat this as exploratory evidence, not a representative cohort or a passed gate. The 2.50 Å / 150° rule is not a production label or model cutoff. IUPAC’s recommendation treats geometry as evidence among several criteria, rather than defining a universal cutoff. [IUPAC hydrogen-bond recommendation](https://publications.iupac.org/pac/83/8/1637/index.html)

## 1. Define the population, sample, and matching

Use these terms consistently in the plan:

- **Target corpus:** the exact eligible records used by the official refit. For this repo, that means cleaned tmQM X1–X3 and LnQM LNf records after the same benchmark exclusions and validity rules used for the other fitted quantities. Do not call a sample “tmQM-representative” if shards are missing; the three previously unscanned charged shards must be scanned or explicitly excluded from the named target.
- **Parent-state:** one complex with a fixed molecular state. Its identity includes connectivity and bond orders, stereochemistry, protonation/tautomer, charge, multiplicity, and metal coordination/hapticity.
- **Reference geometry:** the coordinates supplied by the dataset, not a guaranteed native or lowest-energy pose. tmQM’s structures come from the Cambridge Structural Database and were optimized with GFN2-xTB, so the plan must describe them as dataset references. [tmQM paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c01041)
- **Baseline pose:** a conformer generated from the same parent-state by the frozen `u1_pure_rep` solver.
- **Anchor opportunity:** a stable graph-defined pair or group pair whose geometry can be measured in both the reference and baseline coordinates. It is a candidate for measurement, not a claim that an attraction exists.
- **Support:** feature space where both reference and baseline data provide adequate evidence. Outside support, NCI contributes zero.

Build the frame before looking at NCI contacts or MACE energies. Include records with zero candidate opportunities. Group source siblings and records sharing the same ligand scaffold into the same split. Stratify by metal family (3d, 4d, 5d, other) and coordination-number class (≤4, 5, 6, ≥7). Within each nonempty stratum, draw groups without replacement: allocate one group first, then allocate the remaining budget in proportion to target-corpus weights using largest remainders. Use a saved random seed and record inclusion probabilities. If a stratum is smaller than its allocation, census it and reallocate the unused slots.

Start with 240 independent groups: 160 for fitting, 40 for scale calibration, and 40 held out. This is a compute allocation, not a claim that 240 is always enough. Use the same sibling weights as the refit; multiply each sampled group’s target weight by the inverse of its inclusion probability. If a feature or context lacks independent support, leave it unscored. Select replacements for failed solves from a predeclared random reserve in the same stratum; never replace a failure based on its contacts or energies.

For each selected parent-state, generate baseline poses with NCI off and Q6 unchanged, using seeds 11, 12, and 13. Save all finite, deduplicated solver conformers and divide that parent’s weight across its poses and eligible anchors. Save the frame, state signatures, scaffold groups, split, inclusion probabilities, solver settings, failures, and input hashes.

Map each reference structure to its baseline poses by molecular graph isomorphism, then canonicalize symmetry-equivalent atoms into the same anchor identity. Compare the **same anchor in the same parent-state** across different conformations. Different conformations are valid comparisons; a different isomer, protonation state, stereoisomer, or metal-coordination state is not. Ambiguous mappings and state mismatches are unknown/unsupported, never negative contacts.

## 2. From opportunities to a learned surface

Keep three operations separate:

1. **Candidate generation** enumerates graph-defined anchors and may use a broad computational neighbor list for speed. That list only decides what to measure; it does not label an interaction or create attraction.
2. **Evidence estimation** asks whether a continuous geometry/context pattern is more common in reference structures than in matched `u1_pure_rep` poses.
3. **Scoring** rewards only supported, reproducible enrichment.

For the first development case, enumerate explicit donor–H bonds and chemically plausible acceptor atoms from connectivity, valence, formal charge, resonance, aromaticity, and metal-binding context. Record those properties as context features. Do not label an opportunity “positive” from a distance/angle threshold. The pilot’s 2.50 Å / 150° selector is not carried into fitting or runtime.

For every anchor, save one reference feature vector and feature vectors for the same anchor in all baseline poses. For H bonds, use continuous H···A distance, donor–H direction relative to the acceptor, and acceptor local-approach geometry when defined; encode unavailable axes as unavailable, not as a fabricated angle. Fit a weighted matched reference-versus-baseline model with a shared smooth geometric surface and partially pooled local-chemistry context. Fit a tensor-product C² spline to the matched log enrichment

\[
L(x,z)=\log\frac{p(\text{reference geometry }x\mid z)}
                    {p(\text{baseline geometry }x\mid z)},
\]

where \(z\) is local graph and coordination context. Choose smoothing using parent/scaffold-grouped cross-validation within the 160 fit groups. Bootstrap whole parent/scaffold groups, not atom pairs or poses. A region is eligible for attraction only when its simultaneous 95% lower confidence bound is above zero and both data sources support it. Shrink uncertain regions to zero and taper the surface to zero with zero first and second derivatives at the support boundary. Negative enrichment, missing controls, and extrapolation earn no attraction; Q6 owns repulsion.

At runtime, use the same anchor generator, atom mapping, features, support mask, and normalization as extraction. Canonicalize symmetry-equivalent anchors and score each physical opportunity once. For alternative acceptors of one donor–H anchor, aggregate the nonnegative supported values \(u_{ha}\) as

\[
A_h=\tau\log\!\left(1+\sum_a[\exp(u_{ha}/\tau)-1]\right).
\]

This is zero when all alternatives score zero, is unchanged by adding zero-score candidates, and grows sublinearly when several alternatives score favorably. Select one shared \(\tau\) by grouped cross-validation on fit data; do not tune it on MACE or validation results. Sum over distinct donor–H anchors, apply the existing global scorer normalization, and fit one shared nonnegative energy-scale coefficient for the whole NCI term. Do not fit separate scales by element, chemistry context, or interaction type.

The statistical enrichment is not a physical energy. Calibrate its scale with the repo’s existing full-complex MACE64 motion protocol: on the 40 calibration groups, evaluate MACE64 with the original charge and multiplicity along the same predeclared motions; compare relative MACE energies with the frozen NCI-off response and the candidate NCI response; fit one nonnegative coefficient by equal-parent least squares. Freeze it before the 40 held-out groups. If the scale is zero, changes sign under a reanalysis of the calibration split, or fails the held-out response checks, keep the candidate off. Describe the result as an energy-calibrated empirical score against MACE64, not as a measured physical pair energy. Structural reference ratios and decoy-based potentials depend on the chosen reference distribution and task; they are not automatically thermodynamic energies. [DARS](https://pmc.ncbi.nlm.nih.gov/articles/PMC2567923/), [DOPE](https://pmc.ncbi.nlm.nih.gov/articles/PMC2242414/)

## 3. Generalize by reusing the data procedure, not cutoffs

The reusable method is graph-defined anchors, continuous geometry/context features, matched reference/control observations, and an uncertainty-tested enrichment surface. Named interaction classes supply useful anchor shapes, not hand-written attraction basins.

- **C–H···π:** define a C–H bond and graph-derived ring. Measure H position relative to the ring plane and centroid, in-plane offset, and C–H direction. [Bissantz et al.](https://doi.org/10.1006/jmbi.2000.4473)
- **Aromatic ring pairs:** define ring anchors and measure centroid separation, both plane orientations, and lateral slip. Include parallel-displaced and T-shaped arrangements; do not restrict the feature space to parallel rings. [JBC aromatic-pair study](https://doi.org/10.1074/jbc.273.25.15458)
- **Unspecified recurring motifs:** enumerate nonbonded atom, bond, ring, and local-fragment anchors using local graph descriptors and relative-coordinate features. Discover clusters on fit groups only; retain a motif for testing only if it recurs and is enriched against matched baseline data in held-out independent groups. A pair-only model that fails because the pattern depends on a third atom, ring, or coordination environment stays unsupported rather than being forced into a pair curve.

The current prototype is HB only. Add C–H···π and ring-pair feature families after the shared HB pipeline passes its gates; attempt unknown-motif discovery after those anchor and ownership rules work. A single physical pair or group arrangement cannot receive duplicate rewards through overlapping feature families.

## 4. Runtime product and validation gates

The production artifact is one versioned constraints database produced by the official full-refit command, alongside Q1–Q6 and the other distributions. It contains the learned NCI coefficients, support mask, and C² spline data under versioned NCI entries; the existing metadata sidecar records the training-frame hash, weighting, splits, feature schema, code/model hashes, calibration coefficient, and validation report. The solver loads and differentiates this surface through the normal scorer path; no monkeypatch or per-solve fitting is part of production. Candidate databases remain isolated until promotion.

Use this fixed validation sequence, with no tuning on later gates:

1. **Identity and data checks:** verify exact state and anchor matching, symmetry handling, equal-parent weighting, inclusion-probability weights, scaffold isolation, zero-contact parents, and explicit unsupported states.
2. **Surface check:** test source/runtime feature parity, C² values/gradients/Hessians, finite behavior at support edges, zero score outside support, and recurrence of predicted enrichment in held-out groups.
3. **Energy/curvature check:** assess relative-energy RMSE, force-direction agreement, wrong-sign curvature, and sampled-minimum regret on the separate 40-group scale and 40-group validation sets. Require the existing project calibration gates to pass: a positive lower 95% paired parent-bootstrap interval for RMSE improvement, no worsening of mean sampled-minimum regret or wrong-sign-curvature count, and a held-out curvature ratio within the project’s predeclared 0.7–1.5 band.
4. **Fresh-solve check:** on 16 new, nonbenchmark parent-states selected by the same frame rules, compare frozen candidate and `u1_pure_rep` with identical three seeds and MACE64 scoring. Require a positive paired parent-bootstrap 95% interval and no regression greater than 10 kJ/mol. Treat this as a screen, not a corpus-wide claim.
5. **Promotion check:** only after the earlier gates pass, run one paired dev80 evaluation and then the reserved heldout28 and ring10 evaluations once, comparing the candidate with the incumbent and N1. Do not use those results to retune the surface.

If any gate fails, record the failed criterion, retain the candidate as research-only, and stop this candidate sequence. Do not add more variants to rescue a failed result.

## 5. Budget-script cleanup and document outputs

Keep the import-safe flat paths for active budget tools, as requested. Reduce clutter by archiving completed research families, not by moving every active script into new packages.

- Keep scripts used by the current CLI, official refit, tests, or documented live workflows, including shared utilities and the current NCI census/diagnostic path. Keep CTX1/CTX2, old C²/ablation and training-calibration branches, physical-probe and embedding/cap studies, and their experiment-only tests together as archived research families unless a live consumer requires them.
- Before removing a script from the active folder, check imports, tests, notebooks, command documentation, and required artifacts. Archive the script with its family’s report, selection manifest, input/code hashes, and a clear result status. Do not delete a script based only on its name or lack of reverse imports. Preserve the existing research archive; add a separate dated budget-research archive for files not already captured.
- Retain `data/refit/evidence/` and the current N1/NCI/Q6 baseline records as the small, durable evidence set. Record unavailable `/tmp` CTX outputs as unavailable rather than claiming they are archived or reproducible.
- Reconcile `tools/jdev/README.md` and `tools/INVENTORY.md` with the actual active tree. Group remaining tools by workflow there and link each archived family to its evidence. Do not reset or overwrite unrelated working-tree changes.

The document deliverable is a complete replacement of `plan.md` with these definitions, decisions, gates, and cleanup rules.
