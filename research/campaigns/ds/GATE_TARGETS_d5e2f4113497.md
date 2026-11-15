# Gate targets under production `d5e2f4113497` (auditor, 2026-11-12)

Question: after the DS fold-in, are the benchmark targets of the 230 gate parents (stage 1 cal + tail 120, stage 2 eta2 holdout 110)
the audited ones?

## Inputs (sha256)
- Audit manifest (frozen source blocks): `research/data/audit/2026-11-10_DS/gate_inputs/manifest.json` `e71e9452...d4e51d`
- Audited corrected targets: `target_overlays_v1.json` `9a9c820b...0d127` (34 repairs)
- Frozen eligibility: `research/campaigns/ds/gate_eligibility_v1.json` `a8d7a55b...864a1e` (primary = valid + repaired)

## Method
`python -m tools.run diagnostic --manifest <audit manifest> --task tools.jgen.ds.gate_audit:production_target` (new task: `mapping.assemble`
on the frozen source block, import-time production flags, no patched hooks = the bench's own target path, checked against
`tools.jgen.inputs.load_mol` on 8 parents). 230/230 ok, fingerprint `d5e2f4113497` in every record. Outputs (gitignored):
`research/data/audit/2026-11-12_gate_fixes/target_d5e2f4113497/`, comparison `compare_v2.txt`.

Note: the audit's own `audit_imported` / `_typing` path patches mapping internals and gives wrong targets under the DS rules
(e.g. USAPOU, XACMOE with 0 sites); do not use it on DS-era code. `production_target` replaces it for this check.

## Result
| Stratum | Parents | Under production typing |
|---|---:|---|
| valid | 170 | 169 identical to the audited (old production) target; GOTWET an equivalent-atom swap only |
| repaired | 34 | **31 reproduce the audited repair exactly**; AGAPEG, ILEPOE give kappa1 on the near O (owner decision 2026-11-10, supersedes the kappa2 overlay); BADQIH differs from the repair (production 5 sites: three N donors + two eta2 C=C of the norbornadiene; repair 4 sites). **Owner 2026-11-12: production is correct; the audit repair is superseded** |
| ambiguous | 23 | 19 unchanged, NEZGAB/YUSBOE = T1d typing, GACHAT/MAKJAM changed; outside the primary root either way |
| unrepaired | 3 | CEXPOM, LOPCAW unchanged; NEPNEF changed; outside the primary root |

## Consequences
1. `--target-overlay` is identity-only: the bench never applied the overlay (also not in the DS gate snapshot `51cf9ad`). In the DS
   gate the C arm (old typing) therefore ran the unrepaired targets on the 34 "repaired" primary parents, while D's typing repaired
   31 of them natively. Part of D's stage-1 gain sits on these parents (e.g. VOBGUR, VAVBIE, UMUBAF, MIDQEY are in the repaired set);
   that is an effect of the D typing, not a target change both arms shared. The fold-in decision is unaffected; the statement
   "the audited targets apply to production as well" (DS_GATE_RESULTS.md section 2) was not true.
2. From `d5e2f4113497` on, both arms of any comparison build targets with production typing, so the primary root is the audited
   target set, with the two owner-decided kappa1 parents and BADQIH at its production target (owner 2026-11-12). Every primary
   target is settled.
3. Future gates: eligibility `gate_eligibility_v1.json` (primary decides, all reported); targets = production typing; BADQIH reported as a named case.

## Gate-code amendments (auditor + owner, 2026-11-12; `research/bench/tools/jgen/d3/analyse.py`, all stages)
- Failures gate: fails only on `arm_failures`, units (parent, seed) that failed in the arm while the control was ok.
  `control_failures` (control failed, arm ok; e.g. EPUPEL's control timeout in DS stage 2) are reported, not held against the
  arm. Units failed in both arms are excluded and listed.
- Catastrophic counts (top1_dE > 200) and the stage-2 edge-on holdout are both taken over units ok in both arms; an ok record without a count is unknown (gate fails).
- Read-only recheck of the DS gate outputs with the amended code: stage 1 D qualified (unchanged); stage 2 D holdout and
  failures pass (no arm-only failures; control-only EPUPEL). Historical `analysis.json` files are untouched.
