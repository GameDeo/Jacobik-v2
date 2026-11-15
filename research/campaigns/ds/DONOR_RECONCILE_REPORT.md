# Donor identity reconciliation — Checkpoint A, 2026-11-10

User-authorized data/sampling investigation; parent owns sampler, confidence, and combined checks. This branch implements only the opt-in `donor_identity_reconcile` source path; registered default False (profile registration is parent commit 480bd2a). Production fingerprint stays `5b56969afb9c`. No parameter DB, golden references, solver/replay arithmetic, or production default was changed.

## Rules and limits

- Try a source-heavy-graph mapping that also preserves each heavy atom's attached H count. An inconsistent H assignment produces an uncertainty event and retains the original element-only alignment.
- Reconcile terminal, H-free O/S siblings sharing one C/N/S center, using both raw M–X BO >= 0.15 and the existing covalent-radius ratio <= 1.21 as independent contact evidence. These are carboxylate, dithio, nitrate and sulfonate motifs covered by frozen source fixtures. Keep already-supported source identities. Terminal-motif κ² cardinality and distinct supported declarations are preserved. Outside those motifs, preserve an already supported same-element correspondence from the prior loader when the new source mapping would select a remote atom. This is a compatibility guard, not new unrestricted candidate selection: if multiple declarations share an occupied prior identity, record the collision and do not introduce their unsupported remotes as extra donors. No cross-element preservation is allowed. Irreducible source identities generate review events.
- When an existing hapto rule rejects a C–X face, preserve independently supported local O/N/S neighbors of that face's carbon as κ donors. Both BO and distance are required. Authoritative builder donor groups are never changed by this rule. Existing hapto flags remain separate; `DR` below means T1d typing plus reconciliation.
- These rules are source reconciliation, not an automated observation exclusion policy. A weak declared atom is not by itself bad data; valid haptic constituents, asymmetric chelates and boundary-length source waters can generate such events.

## Validation and progress

- 2026-11-10: `pytest -q tests/test_donor_identity.py tests/test_typing_t1.py tests/test_profile_hook.py`: **39 passed**. Tests cover distal O with BO 0.19 (above the legacy remap threshold), distinct κ² preservation, unrelated atoms in the same ligand, no cross-element swap, no protonated-O swap, support requiring both BO and geometry, H-aware mapping, rejected-face κ preservation, authoritative-group protection, default fingerprint, and actual source-only JAYCEV/UKUWON/BADQIH regressions.
- Source-only pilot: C, T1d (D), and T1d+reconciliation (DR) on BOPSAD, DUCPAT, LUQVOK: **9/9 completed**, about 1.68 s per isolated task. All tasks use `tools.run diagnostic`, numerical libraries pinned to one thread.
- **BOPSAD:** D selects O57/O59 at 2.639/2.633 Å, BO 0.192/0.193. DR selects local same-acetate O58/O60 at 2.088/2.092 Å, BO 0.526/0.523; both keep the square-planar N2O2 site count.
- **DUCPAT:** D leaves four donors and assigns square-planar after dropping false C–O faces. DR restores both omitted real O atoms (2.022 Å, BO 0.354), yielding the exact audited six-O octahedral target.
- **LUQVOK:** C/D/DR have the same κ-C,C,O,O sites and square-planar geometry. Reconciliation recovers the same correct acetate O112 as the legacy remap, without selecting the distal O113.
- Final frozen census: 46 sources (all 11 watched Pd training parents, 34 independently audited repair fixtures, and LUQVOK unchanged control), each C/D/DR: **138/138 completed**, no task failures, 235.5 s summed task wall (one worker; estimate 248 s). The disabled rule reproduces atom groups/hapticity/subclass/geometry for all **36** reused historical audit sources. T1d extraction exactly reproduces all **44 retained Q1 observations** for the 11 watched training parents.
- All 11 watched Pd parents replace their 22 long primary acetate O selections (2.633–2.709 Å) with local same-acetate O atoms (2.017–2.095 Å). Their N2O2 site counts remain unchanged. No fit or energy improvement is inferred from this geometry-only result.
- Exact audited targets: **17/34** under T1d alone, **29/34** with reconciliation. Twelve gains, zero losses among previously fully matched fixtures. Across all 46 sources, 25 donor-group sets change; the event ledger records 33 local identity corrections, nine local rejected-face κ preservation events, two H-aware mappings, two preserved prior supported identities, three non-injective prior-correspondence collisions, and four weak-declaration diagnostics. No H-constrained alignment failure occurs in this selected sample.
- An initial prototype intentionally removed unrestricted legacy remapping, but its source census exposed new known errors: it lost JAYCEV N7, selected remote UKUWON N30 in place of real N27, and added remote BADQIH N3/N7. Those errors were not accepted. The generic same-element supported-correspondence compatibility guard and three actual source regression fixtures correct them. A **fresh full 138-task census after the final code freeze** confirms no newly added non-audited atoms in any of the 34 fixed fixtures. The initial prototype evidence remains under `donor_reconcile/`, separate from final `donor_reconcile_final/`.
- Negative controls include distinct supported/source-declared κ² sites and the unchanged BUPMAC κ² acetate group, plus LUQVOK unchanged primary identities. No known new loss of a supported donor remains in the final selected census. The two weak secondary identities discussed below require a target-definition decision, not an assertion that they are absent.

## Remaining source eligibility decisions

The five residual fixture disagreements are explicit; the implementation does not claim a complete automatic source cleanup.

| Case | Remaining issue | Required fit/source review |
|---|---|---|
| AGAPEG, ILEPOE | Single source declaration maps to a local near O under reconciliation. The earlier audited overlay retained both that O and the weak distal originally mapped O (AGAPEG O4, ILEPOE O7). | Decide primary κ identity versus asymmetric κ² target. The near primary is evidence-backed; presence of the second contact cannot be settled merely by its original dative mapping. |
| BADQIH | Existing supported N6 and both alkene groups are preserved. Multiple ambiguous declarations share N6; the additional real N4 remains unrepresented. | Broader heteroarene mapping is outside the tested terminal motifs. All actual Q1 keys are listed for review because geometry/coordination is affected. |
| JAYCEV | Previously correct N7 is preserved; the second real pyrimidine N9 is still absent, with redundant prior mappings colliding onto N7. | Review the non-injective donor correspondence and full geometry-dependent key set before fitting this source. |
| NUHPEL | Existing added distant η³ aryl face is unchanged; an η² rejection rule cannot fix it. | Review the spurious multi-atom donor face before treating its fitted terms as valid observations. |

`weak_declared_identity_retained` is not an exclusion criterion. In the final census its four events belong to DUCPAT waters (O21/O45) and VAVBIE sulfur sites (S2/S22): declared sites with nonzero BO and somewhat longer radii ratios, not newly introduced remote identities. Haptic weak constituents and declarations removed before final Q1 sites are separately represented by the census taxonomy when present. Non-injective prior correspondence is a distinct uncertainty class. The ledger records exact Q1 key tuples requiring source review and **`quarantine_applied: false`**. No automatic quarantine or per-code extraction exclusion has been implemented.

A full-corpus source sweep is deferred. The 11 watched training parents and 34 fixtures are a causal/mechanism check, not prevalence estimates or an unbiased quality gate. Existing retained-Q1 corpus counts are owned by the parent investigation; this branch only reads the 44 watched rows. Checkpoint A stops for a reviewed source eligibility policy before any new full rebuild or structural qualification. No energies, generated pools, embeds, solves, billed jobs, or locked data were used.

## Final artifacts and reproduction

Final output is main-checkout `research/data/audit/2026-11-10_DS/donor_reconcile_final/`. `manifest.json` freezes source snapshots and independently audited expectations; `affected_v1/` is the JAYCEV/UKUWON/BADQIH 3-profile check; `run_v1/` is the final 138-task census. `correction_ledger.json` records original/corrected sites, events, lost expected atoms and source-review taxonomy. `retained_q1_watched_rows.json` is the unchanged input subset. `census_summary.json` is the aggregate. SHA256:

- `manifest.json`: `a6f93c9f93f70cc49e7309c898ec393b3d874f4897509a37a32be14fb1569cc8`
- `correction_ledger.json`: `47204e887d9b7fd4a6a6285d1e89edfff6de982120cca53049af27c4a56acada`
- `census_summary.json`: `4a678a7dbdbbbe2ae8cfd1b79eeb0e71f0ccbc1a336dd8d15259429ce526671b`

Use the normal environment and main-checkout/output paths:

```bash
python -m tools.jgen.ds.donor_census prepare --main "$MAIN" --out "$OUT"
python -m tools.run diagnostic --manifest "$OUT/manifest.json" \
  --task tools.jgen.ds.donor_census:run --out "$OUT/run_v1" --seconds-per-parent 1.8 --estimate
python -m tools.run diagnostic --manifest "$OUT/manifest.json" \
  --task tools.jgen.ds.donor_census:run --out "$OUT/run_v1" --seconds-per-parent 1.8 --timeout-s 45
python -m tools.jgen.ds.donor_census summarize --main "$MAIN" --out "$OUT"
```

The investigation lead owns the one combined workstation checkpoint check after integration. No duplicate full `ws` or production fold-in is attempted by this branch.
