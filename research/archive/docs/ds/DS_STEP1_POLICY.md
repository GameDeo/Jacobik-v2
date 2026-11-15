# DS step 1: source policy, extraction ledger, eligibility — 2026-11-10

Policy `eligibility.1`, ledger `extraction_ledger.1`, review data `source_review.1`
(`src/fit/eligibility.py`, `src/fit/data/source_review.json`, extractor `ledger_rows`).

## Decisions

| Case | Decision | Fit effect |
|---|---|---|
| Clear corrections (local O/S sibling, H-aware mapping, rejected-face kappa) | Applied by the opt-in `donor_identity_reconcile`; each is a ledger event with before/after atoms | none beyond the changed observations |
| Weak declared donor retained | Review event, retained | none |
| AGAPEG, ILEPOE | **kappa1 on the near, evidence-backed O** (owner, 2026-11-10); weak distal O not added as a second donor. The earlier kappa2 audit overlay is superseded for these two | none |
| BADQIH, JAYCEV (non-injective ring mapping), NUHPEL (false eta3) | Unresolved; retained and flagged; all keys of these parents are review dependencies | none |
| Charge/protonation | No repairs invented | unchanged (os_sanity as before) |

No parent is excluded because of an event or a long distance. `EXCLUSION_FACTORS` is empty;
`eligibility_codes` raises for an unregistered factor. A new factor needs a new policy version and a separate
attribution arm; it is never folded into the donor-only flag. Locked exclusions stay fixed.

## Ledger

One row per event: parent, dataset, metal, kind, disposition (correction / preserved / review),
before and after atoms (index, element), event detail, and `dependent`: the parent's Q1–Q5 and QO row counts and
distinct keys. A donor-count or coordination change can invalidate every geometry row of the parent, so the whole
key set is the dependency, not only the one distance. Chunk ledgers merge per split with dispositions assigned at
merge; build metadata carries `eligibility` (policy, reason counts by kind and disposition, parents, dependent rows,
unresolved parents present). Default fingerprint `5b56969afb9c` unchanged (all code is behind `donor_identity_reconcile`).

Smoke on 400 real complexes (shard X3s002, 26 s): 91 parents (23%) carry events: 61 `weak_declared_identity_retained`,
33 `hydrogen_aware_donor_mapping`, 25 `local_donor_identity`, 7 preserved, 7 collisions, 2 H-alignment ambiguities.
The review set in the full corpus will be large; step 3 samples it rather than reading it all.

## Charge filter by quantity (recorded D2 build, `os_sanity` + `fit_exclude_locked`)

| Quantity | Parents in | Charge | Locked | Either | Rows in | Rows removed |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 125,044 | 1,604 | 360 | 1,957 | 612,336 | 9,812 (1.60%) |
| Q2 | 124,891 | 1,601 | 360 | 1,954 | 1,330,413 | 21,456 (1.61%) |
| Q3 | 124,773 | 1,592 | 360 | 1,945 | 524,452 | 8,642 (1.65%) |
| Q4 | 114,633 | 1,482 | 318 | 1,793 | 322,847 | 5,391 (1.67%) |
| Q5 | 119,034 | 1,449 | 344 | 1,786 | 2,353,181 | 31,866 (1.35%) |
| Q6 | 118,394 | 1,542 | 336 | 1,871 | 832,182 | 14,211 (1.71%) |
| QO | 124,700 | 1,582 | 360 | 1,935 | 528,152 | 8,504 (1.61%) |

The filter is whole-parent, so every quantity of an implausible-charge parent goes, including geometry-only rows
(Q1 distances, Q2 angles) whose measurement does not depend on the charge label. The overall rate is uniform
(about 1.6%); the effect is concentrated in sparse keys (LIPDEU `QO:(2, 4)`: 5,742 rows, N_amidinate 994 → 2,806, and the
two-to-three component change from the charge filter). Keeping geometry rows of charge-encoding-suspect parents needs
dependency reasoning per quantity (which rows use the oxidation-state key). That is **not** changed in the four-arm
build (charge policy is fixed across D0/D-data/D-sample/D-both); it is a candidate separately attributed arm if
step 4 leaves consequential changes unexplained.

## Unresolved ledger

BADQIH, JAYCEV, NUHPEL (above); LIPDEU's broad QO fallback composition; the 23 ambiguous and three unrepairable
gate-input cases (audit `investigate/gate-inputs`); all weak-declaration events pending the step 3 sample.

## Verification

`tests/test_ds_ledger.py` (4 cases: rows carry sites and dependencies; policy excludes nothing and rejects
unregistered factors; every emitted event kind has a disposition; decided/unresolved ledger). Combined with
existing donor, profile and D2 pipeline tests: 61 passed, 2 skipped.
