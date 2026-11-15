# D3 gate input audit — 2026-11-10 checkpoint

This review used source SMILES, reference geometry/bond-order evidence, and input typing. It did not open arm scores, generated pools, energy results, or locked populations. The stage-1 C2 registration and the stage-2 A/B/C/D registrations give 120 + 110 disjoint parents. Both stage-1 primary exclusions are reviewed for input quality but retain their original exclusion. Historical examples YUCZUS and CAFWIQ are outside this audit.

| Input disposition | Stage 1 | Stage 2 | Total |
|---|---:|---:|---:|
| No discrepancy detected by the checks | 100 | 70 | 170 |
| Confirmed error; static corrected target validated | 9 | 25 | 34 |
| Ambiguous target; no replacement inferred | 8 | 15 | 23 |
| Confirmed error; no defensible repair | 3 | 0 | 3 |
| **Total** | **120** | **110** | **230** |

111 cases triggered review: 94 have atom-level case notes in [GATE_INPUT_REVIEW.tsv](GATE_INPUT_REVIEW.tsv), and 17 were checked mechanically because the only change was a label with identical atoms, hapticity, and geometry. `No discrepancy detected` is the scope of these checks; it does not establish a unique ground-state complex. In particular, distance ratios are review triggers, not truth labels. A zero stored metal BO means below the parser's 0.08 storage floor.

## What changed in the audited targets

The 34 donor/group corrections are versioned benchmark fixtures. Every one passed static mapping to the existing builder input format, without changing the ligand SMILES, elemental composition, charge, spin, or reference coordinates. Twelve corrections also change the coordination geometry because the reference directions are now assigned to the actual sites. All corrected shape margins that have a runner-up exceed the audit's boundary trigger. No fixture was applied to a solver run or to the original gate. Each fixture records the old and corrected atom groups, shape, source and evidence hashes, reason, and exact builder inputs.

- **MIDQEY and UMUBAF:** false η² faces add long undeclared carbons at the center of an S,S chelate, with no stored metal BO. Retain both actual S donors. **NUHPEL:** three added aryl carbons are 3.245–4.014 Å from Ag, none with stored metal BO; preserve three declared As donors and Br. T1d cannot correct the false η³ group.
- **DUCPAT, IBUTAD, WIJXIW and related cases:** both O atoms have source reference distances/BO consistent with real donors; the carbon has no stored metal BO. T1d removes the false C–O η² face but can also drop a real O. The corrected fixture preserves both κO sites. **MAKJAM** is left ambiguous: its short 2.067 Å Pd–O contacts are evident, but its alternate declared O atoms at 2.698–2.700 Å retain a weak BO and a unique κ versus asymmetric κ² target is not established.
- **BUXJIO, OREHAV, IYUKOC and other alignment losses:** a weak or distal first-map donor gets remapped onto an already occupied site, omitting a distinct short source-supported donor. Restore the distinct site where the source graph and contacts agree. **LUQVOK** does not suffer this error: the existing remap already chooses O112 (2.068 Å, BO 0.508) rather than distal O113 (2.823 Å, BO 0.118), on the same acetate C100. It is κ-C,C,O,O square planar in both modes.
- **NUWFAM and TEBWIH:** uncertain integer hapticity and omitted short F contacts require a separate chemical adjudication. **BODBAY:** benzyl C64 is a clear κ anchor (1.915 Å, BO 0.818), whereas C67/C68 give weaker aryl contacts. Neither source nor geometry establishes an isolated η² alkene target. These are not silently retyped.
- **CEXPOM / NEPNEF:** source Zn formal charge +6 / +14 is implausible; the coupled ligand redox/protonation encodings cannot be replaced by changing the metal charge alone. **LOPCAW:** raw graph fails source-SMILES alignment; prior case evidence identifies a weak long extra N–N edge. It stays flagged for general mapping repair.

All six requested regression cases (LUQVOK, SESWIZ, LIPDEU, YABYEJ, LOPJEF, ZAQNUE) have no η² sites in either reference typing mode and no detected target discrepancy. The source audit therefore does not explain those six as false η² benchmarks.

## Stage 2 and reference compatibility

The 110 historical stage-2 entries include 10 no-haptic collateral controls (C) and 40 cap-eligible entries (D), which need not have η². A/B were drawn from archived η² QO observations. The counts below are **raw loader input hapticity**, not the edge-on metric (which uses separately defined geometric C–C pairs) and not new gate results.

| Stratum | Parents | η² under original typing | η² under T1d | η² in no-discrepancy or corrected target |
|---|---:|---:|---:|---:|
| A | 40 | 40 | 36 | 31 |
| B | 20 | 20 | 3 | 2 |
| C | 10 | 0 | 0 | 0 |
| D | 40 | 10 | 4 | 4 |
| **Total** | **110** | **70** | **43** | **37** |

Keep historical results and populations intact. Use the dispositions to report clean comparisons and ambiguous/unrepaired cases separately. Any future corrected-target comparison must use the same fixture for both arms, new target-aware cache identities, identical reference/evaluator identity, and a frozen contract. Original reference coordinates can be retained for these donor-only repairs if the reference identity is verified; old generated pools cannot be reused as repaired-target outputs. A fixed parent’s E_ref cancels in the paired ΔE; relabeling the reference energy alone cannot undo a historical difference. No new gate is inferred from this audit.

## Reproduction and progress

- 2026-11-10: source pass on the historical `4982024` typing implementation; scope frozen before the full pass. Three pilot parents matched the actual imported off/on profile results exactly (6/6). All parent work ran through `tools.run diagnostic`, one worker with pinned numerical threads. 230/230 source records, no task failures; 468 s summed per-parent wall, 2.02 s median. 34/34 static target validations, no failures.
- 2026-11-10: `pytest -q tests/test_typing_t1.py` (13 passed), Python compilation and `git diff --check`; historical default fingerprint `6b47c3fb45a9`. No production or DB edits, no embeds, solves, billed jobs or Colab spend. Shared workstation checkpoint check is owned by the investigation lead.

The audit data is under `research/data/audit/2026-11-10_DS/gate_inputs/` in the main checkout. `manifest.json` freezes source paths/hashes and registered panel hashes. `run_v1/records` contains the 230 source evidence records. `audit_dispositions.json` and `.csv` bind each disposition to its evidence; `target_overlays_v1.json` contains all verified corrected builder inputs. Key file SHA256:

- `manifest.json`: `e71e945249df6cb80a67a9b27f119e5487310d6db59ff3e5374d360d01d4e51d`
- `audit_dispositions.json`: `d5017022d522b0512f725dd185cde784e208749d120819a15075a0260e5f4e43`
- `target_overlays_v1.json`: `9a9c820bbe9417ad2284a16c87245fe8f5caf0b856e094aac03a9f7df8e0d127`

Recreate in this worktree with the normal one-thread environment, `AUDIT` an empty output directory, and `MAIN` the main checkout. Use `--estimate` before the matching diagnostic invocations.

```bash
python -m tools.jgen.ds.gate_audit freeze --root "$MAIN" \
  --panel research/campaigns/c2/C2_PANEL.json --out "$AUDIT"
python -m tools.run diagnostic --manifest "$AUDIT/manifest.json" \
  --task tools.jgen.ds.gate_audit:audit_parent --out "$AUDIT/run_v1" --seconds-per-parent 2.1 --timeout-s 45
python -m tools.jgen.ds.gate_review freeze-review --out "$AUDIT" \
  --notes research/campaigns/ds/GATE_INPUT_REVIEW.tsv
python -m tools.run diagnostic --manifest "$AUDIT/repair_validation_manifest.json" \
  --task tools.jgen.ds.gate_review:validate_overlay --out "$AUDIT/repair_validation" --seconds-per-parent 2.5 --timeout-s 45
python -m tools.jgen.ds.gate_review publish-overlays --out "$AUDIT"
```
