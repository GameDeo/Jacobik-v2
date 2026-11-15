# DS step 3: extraction review — 2026-11-11

Job 61137184 (QOS regular, 1 node, 3 min 38 s, about 0.06 node-hours): E_off (reconcile off) and E_on (reconcile on),
both with stable observation IDs, 0 failures. E_off reproduces the recorded D2 Q1 row count exactly (612,336). Output:
`$SCRATCH/jacobik/runs/ds-extract-20261110-2349`. Review scripts and the evidence JSONs sit beside it (`rev_*`).

## What the review found

**1. Events overstate impact.** E_on logged 32,912 events on 21,328 parents (LnQM siblings repeat events), but the rule
changes Q1 rows for only 6,279 parents (5%): 24,695 rows lost, 30,881 gained, 523 values changed (net +6,186, +1.0%).
Q2–Q5 change for 5,900–6,300 parents each. Most events (weak declarations, preserved, collisions) change nothing.

**2. A defect in my first reconcile rule (found here, fixed).** The gained Q1 rows had a tail to 19 Å: 2,606 gained rows
above 2.6 Å against 1,021 lost, 743 above 4 Å. 285 of the 634 donors the rule added in the sampled parents sat at
distance/covalent-sum ratios 1.8–100, in 176 of 1,285 parents. Cause: in a symmetric ligand the shell alignment can put a
declared donor on a far copy (UGUGAF: a carboxylate O at 18.8 Å on the second arm of a linker). The prior rule moved the
declaration to the best-contact atom of its ligand, a real contact at ratio 1.24. My rule discarded that move because
1.24 is above its 1.21 "supported" cutoff, kept the far image, and the remote donors then became haptic faces.
The same rule dropped some real donors (collisions on symmetric ligands, binding-mode mislabels in the SMILES).

Fixes (all in `_reconcile_donor_identities`, still default-off): (a) the prior rule's contact atom wins over an
unsupported alignment image; (b) a declared donor beyond ratio 1.6 that the prior rule did not itself produce is not a
donor (event `remote_declared_donor_unrepresented`; the added-donor ratios have a gap between 1.6 and 1.8); (c) a
supported contact the prior rule represented is restored when a declaration would be left without a donor, and replaces
an unsupported alignment image one for one, never raising the donor count.

**3. After the fix, on the same 1,285 parents** (sample: 118–120 per event kind, 300 unchanged, 139 largest gained-long,
12 named; seed 11; metric: supported heavy-atom contacts at ratio <= 1.21 that are not donors, and donors beyond 1.6;
a review metric, not ground truth):

| | rule off | rule on, first version | rule on, fixed |
|---|---|---|---|
| donors beyond ratio 1.6 | 6 | 226 | 6 |
| unrepresented supported contacts | 636 | 378 | 360 |
| parents made worse (more unrepresented) | — | 16 | 0 |
| unchanged parents altered | — | 0 | 0 |

The frozen donor census is unchanged by the fix: 29/34 audited targets exact, residuals AGAPEG/ILEPOE (decided kappa1),
BADQIH, JAYCEV, NUHPEL; the 11 Pd parents keep their corrected oxygens. The 6 remaining far donors are prior-rule
donors, untouched. 24 of 300 unchanged parents carry an unrepresented contact under both rules (8%): a background rate
the rule does not touch (mostly N, O, S, P at 2.0–2.9 Å: solvent, ion pairs and bridging contacts, not reviewed).

**4. Remaining review load.** 150 `remote_declared_donor_unrepresented` events in the sample (about 12%) record
declarations the source makes at impossible distances: tmQM SMILES artefacts, a benchmark/data-quality category for the
exports, not fitted.

## Population effect (pre-fix E_on; recompute after re-extraction)

Long carboxylate triggers: 3,021 (E_off, matches the retained D2 census) vs 2,901 (E_on). Carboxylate Q1 rows: 2,882
lost, 5,388 gained, medians 2.06 vs 2.05 Å: the rule mostly restores the second oxygen of a carboxylate. These numbers
include the defect above and are not to be quoted for the final E_on.

## Consequence

E_on must be re-extracted with the fixed code (E_off is unaffected: the flag-off path is the prior rule). The pre-fix
E_on tables are retained as evidence and must not feed any fit. Policy: no exclusion was added; the remote bound is a
donor-identity rule in the extractor, not a fit-eligibility exclusion, and is versioned by the ledger's new event kinds
(`legacy_contact_identity_preserved`, `legacy_supported_donor_restored`, `remote_declared_donor_unrepresented`).

---

# Closing report (2026-11-11)

**Outcome.** The rule chain reviewed above was replaced by one injective, evidence-ranked donor assignment (`donor_assignment`, `DS_ASSIGN_PROTOTYPE.md`), with a face-evidence rule and a separate
whole-parent screen (`structure_sanity`). Everything is flag-gated and default off; production fingerprint `5b56969afb9c` is unchanged. Rules are listed in `DS_RULE_INVENTORY.md`.

**Reviews (owner, against CSD).** Pack 1 and 2 (22 disagreements): assignment correct or better in 17, tie 1, indeterminate 4, the rule chain never better; its systematic errors were zero-BO declared
donors, the wrong symmetry twin and donors dropped on collision. Face review: all 9 faces dropped in the sample were confirmed correct drops. A further round of 10 single-donor cases (NASPOO, SUYSIO, BEGHAX, BUYBUT, DEFTIS, FOJMEX, HEKHEK, LAWDUL, MAJSOF, YUSZOD) was judged case by case; HEKHEK, LAWDUL (drop), MAJSOF and YUSZOD were confirmed as assigned, and the remaining disagreements trace to tmQM-versus-CSD binding-mode drift (xTB breaks weak contacts, e.g. SUYSIO, MAJSOF, LAWDUL; adds spurious ones, e.g. WOBKIK, JABBIY): the tmQM geometry is the target, so these are not fixed (F3, `DS_FINDINGS.md`). Screen review: all 10 sampled screened parents (e.g. JENDAH) are broken.

**Decisions.** (1) Bond-order floor 0.05 and ratio <= 1.6 for a chosen donor. (2) Face dropped when no member has BO >= 0.05 and none lies within ratio 1.21. (3) The distance exception for zero-BO non-carbon
donors was tried and reverted (F6: 167 parents affected; the owner prefers one evidence rule). (4) Pi-capable seeds for the face scale were tried and rejected (census 31/34 to 25/34). (5) A broken
parent is removed whole (`structure_sanity`: >= 25% of >= 6 carbons with two neighbours at a bend below 150 degrees; 79 parents); no donor-specific exclusions were added. (6) No source was changed:
tmQM supplies only `CSD_code`, `SMILES` and geometry.

**Extraction result (Step 4, job 61150093).** D0 reproduces the recorded D2 key by key (425,635 entries). The extraction change alone moves 12,701 births and 18,093 deaths of shared keys
(D-sample to D-both), mostly sparse; sampling alone moves 0 population and 2 deaths. The final candidate `D-final` has 419,644 entries.

**Unresolved populations (explicit).** tmQM-versus-CSD drift parents; the 167 short-contact parents behind the bond-order floor; face questions of the XIWKAR type (kappa2 versus eta2);
the 8% background of unrepresented supported contacts (solvent, ion pairs, bridging); the charge filter's Zn/Ni/Cd artefact (F5); 26 reference-incompatible gate targets (23 ambiguous, 3 unrepaired).
