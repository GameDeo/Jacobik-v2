# η² face plane, Step 4 holdout: evaluation contract (2026-11-01, written before any solve)

**Question:** does `eta2_face_plane` with the η² refit (DB v3) fix the η² definition without costing energy, and
leave everything else unchanged? Owner scope (2026-11-01): fold in the S_orient definition alone; cap off.

## Holdout
`ETA2_HOLDOUT.json` (copy of `research/data/jgen/campaign/eta2/holdout/ETA2_HOLDOUT.json`), drawn by
`research/bench/tools/jgen/diag/eta2_step4_holdout.py` (rule in its docstring, seed 31) from `surface_fit`, excluding
cal, val, tail, reserve, locked sets and every code named in an H-track or η² file. NUWFAM/QUTWOT/EPUWAQ (dev) are not
in it. The statistics DB was fitted on tmQM, so this is held out from the package's development, not from the fit.
- **A** (40): ≥ 1 η² face where each face atom has ≥ 2 substituents (the plane is used).
- **B** (20): η² faces, none qualifying (metal-normal fallback; only the refit entries differ).
- **C** (10): no haptic face (collateral).

## Arms (local RTX 2060, sequential, same machine, seeds 11–13, `--panel named`)
- `ctl`: production (fingerprint `3f75c9525a4e`, protected DB).
- `pl3`: `{"eta2_face_plane": true}`, `JACOBIK_DB` = `constraints_2026-11-01_eta2c.json` (DB v3).

## Measures (per parent and seed)
- `top1_dE`, `floor_dE` (kJ/mol vs crystal, MACE, unrelaxed), status.
- Geometry over the whole exported pool. η² faces are taken from the crystal in builder order: bonded C–C pairs
  (≤ 1.6 Å) with both M–C ≤ 1.21 (r_M + r_C) whose connected set of such carbons has size 2.
  - geminal: the smallest H–C–H angle at an η² face carbon (H within 1.25 Å);
  - tilt: |n · u_M| with n the shared-plane normal (`orientation.eta2_plane_normal`), for faces where it is defined;
    a row is **edge-on** if any face has tilt < 0.5 (crystal bonded faces: p1 0.91).

## Gates
1. **Definition (A):** in `pl3`, no pool row has a geminal angle < 30°, and the edge-on row count is ≤ `ctl`'s.
2. **Energy, non-inferiority (A, then B separately):** paired Δ = `pl3` − `ctl` of the seed-mean top-1 and of the
   seed-mean floor per parent; 95 % parent-clustered bootstrap CI (10 000 resamples, seed 0). Pass if the upper bound
   is ≤ +10 kJ/mol for both.
3. **Collateral (C):** |Δ top-1| and |Δ floor| ≤ 0.05 kJ/mol for every parent and seed (same-GPU top-1 reproduces
   within 0.011 kJ/mol).
4. **Failures:** a parent/seed that fails in one arm only is reported and counted against `pl3`.

Everything is reported whether it passes or fails. No tuning on these outcomes; a failed gate stops the package for
the owner.

## Amendment 1 (2026-11-01, before any result was opened): the haptic cap goes in with the plane
**Owner:** the fold-in is `embed_haptic_li_cap` + `eta2_face_plane` together (the original PLAN step 5); the auditor had
narrowed it to the plane alone. At this point only record counts had been read, no energy or geometry.
- **Cap eligibility** (`diag/eta2_step4_cap_eligible.py`, the engine's own gate, CPU): 186 of 1200 `surface_fit`
  parents; in the holdout A 8/40, B 9/20, C 0/10.
- **Stratum D** (`ETA2_HOLDOUT_D.json`): 40 eligible parents drawn (seed 31) from the 169 eligible outside the
  holdout, same exclusions.
- **New arm `cappl3`:** `{"embed_haptic_li_cap": true, "eta2_face_plane": true}`, DB v3, on A + B + C + D;
  `ctl` and `pl3` also run on D. Same machine, seeds, settings.
- **Gates for the fold-in candidate `cappl3`** (gates 1–4 stay as written for `pl3`):
  5. **Cap-eligible set E** (eligible parents of A and B, plus D; 57): seed-mean top-1, `cappl3` − `ctl`, 95 % parent-
     clustered bootstrap (10 000, seed 0): **upper bound < 0**. Also reported: `cappl3` − `pl3` (what the cap adds) and
     the floors.
  6. **Not eligible** (the other holdout parents): `cappl3` vs `pl3` |Δ top-1| and |Δ floor| ≤ 0.05 kJ/mol for every
     parent and seed (the cap does not act there).
  7. **Geometry** in `cappl3` on A, B and D: no pool row with a geminal angle < 30°; edge-on rows ≤ `ctl`'s.

## Amendment 2 (2026-11-01, before any `pl5` result): fold-in candidate = plane without the bonded cut
**Owner:** fold in `eta2_face_plane` with the η² keys refit on every η² face (no bonded cut; the cut goes to the
far-field / typing package). The cap stays off (gate 7: geminal collapse on TEBWIH, open defect). Context
(`ETA2_RESULT.md`): DB v3 changed only η² keys, its "collateral" was the cut on C=O / C=N / C=S η² faces; stratum C
cannot see the DB, its movement is same-arm GPU noise (up to 96 kJ/mol), so the 0.05 kJ/mol identity gates (3, 6) are
replaced by CPU checks: the DB diff (only η² keys differ; refit guard) and the no-η² test below.
- **Arm `pl5`:** `{"eta2_face_plane": true}`, `JACOBIK_DB` = `constraints_2026-11-01_eta2e.json` (DB v5: production DB
  with the 967 η² keys refit through `fit.qo` / `orientation.fit_rows`; all 9,253 other QO keys reproduce production
  within 1e-9). A + B + C + D, same machine, seeds, settings; `ctl` reused.
- **Fold-in gates:**
  - G1 (geometry, A): in `pl5`, no pool row with a geminal angle < 30°; edge-on rows ≤ `ctl`'s.
  - G2 (geometry, B + D): no pool row with a geminal angle < 30° beyond `ctl`'s count.
  - G4: parents failing in `pl5` only are reported and count against it.
  - CPU: parents without an η² face score bit-identically under the flag (score of the `ctl` pools rescored with and
    without the flag + DB v5, CPU).
- **Reported, not gating** (2060 noise): paired Δ top-1 / floor per stratum with bootstrap CIs; and the owner's one
  relaxation before fold-in: GFN2 crude relaxation of `ctl` and `pl5` top-1 and floor on A + B (MACE float64 strain and
  relaxed Δ), `run_relax_g1_g3.xtb_opt` settings.

## Amendment 3 (2026-11-01, before any `capgb` result): the haptic cap with the group bend
**Owner:** fix the cap's geminal collapse and fold the cap in on top of the eta2 plane fold-in (production
`b1fcfae58190`, DB `constraints_2026-11-01.json` = v5). Fix: `eta2_group_bend` (angle_bends): on an eta2 face that does
not use the shared plane, a face atom with >= 2 substituents bends them as one rigid group (TEBWIH CH2: two per-H specs
-> one group spec; plane faces and single-substituent faces unchanged; unit test: H-C-H fixed under any lean / swing).
- **Arm `capgb`:** `{"embed_haptic_li_cap": true, "eta2_group_bend": true}`, DB v5 + sidecar, A + B + C + D, same
  machine, seeds, settings. Comparator: `pl5` (same fingerprint and DB as the new production).
- **Gates:**
  - G7': `capgb` on A + B + D: no pool row with a geminal angle < 30°; edge-on rows <= `pl5`'s.
  - G4: parents failing in `capgb` only are reported and count against it.
  - G5 (as Amendment 1): cap-eligible set E, `capgb` − `pl5` seed-mean top-1, 95 % parent bootstrap: upper bound < 0.
    Also reported: non-inferiority (upper bound <= +10) on E and on the non-eligible parents (where only the group bend
    acts), and floors.
- Same rules: everything reported; a failed gate goes to the owner.
