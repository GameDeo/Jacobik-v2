# η² face plane + haptic cap (H-patch): result (2026-11-01)

Pre-registration: `ETA2_CONTRACT.md` (base text + amendments 1–3, each committed before its results). Holdout: 110
`surface_fit` parents (A 40 plane faces, B 20 η² without the plane, C 10 no η², D 40 cap-eligible), local RTX 2060,
seeds 11–13, `--panel named`. Data: `research/data/jgen/campaign/eta2/holdout/` (`run.sh`, `run4.sh`); analysis:
`research/bench/tools/jgen/diag/eta2_step4_analyze.py` → `ETA2_HOLDOUT_RESULT.json`.

## 1. Folded in: `eta2_face_plane` + DB `constraints_2026-11-01.json` (plane alone: `b1fcfae58190`; with §2: production `273f2ccbf892`)
η² faces whose atoms each carry ≥ 2 substituents take the face normal from the plane through the bond that best fits
the substituents (the metal only sets its sign), in S_orient and in the lean DOF (lean range 24°); η² features are
tilt, bend, slip. The 967 η² S_orient keys (k 12/112/212) are refit under the plane on every η² face through
`fit.qo.fit_from_observations` → `orientation.fit_rows`; the other 9,253 QO keys and every non-QO key equal
`constraints_2026-10-18.json` (refit guard, ≤ 1e-9).

Arm `pl5` vs `ctl` (old production), amendment 2:

| Check | Result | Numbers |
|---|---|---|
| G1 geometry, A | **fail (owner accepted)** | geminal < 30°: 0 → 0; **edge-on rows 172 → 299** |
| G2 geometry, B + D | pass | geminal < 30°: 0 → 0 |
| G4 one-arm failures | pass | none (ADINOR, LADJAB, IWOLEN fail in every arm) |
| no-η² parents identical (CPU rescore, stratum C) | pass | 0.0 over 1,136 pool rows |
| replay fallbacks | pass | 0 |

Δ top-1 (pl5 − ctl, kJ/mol, seed-mean per parent, 95 % parent bootstrap): A −9.6 [−20.5, +0.4]; B +17.2 [−6.7, +48.7];
C −0.01 [−0.03, 0.00]; D −1.8 [−7.3, +3.1]. Floors: A −6.3 [−16.3, +2.1], B +7.8 [−2.4, +21.1].

**Why edge-on rises:** the η² class contains loader artefacts. On plane faces 12 % of crystal rows are M···C contacts
(M–C / (r_M + r_C) > 1.21), and they are 90 % of the edge-on rows, so the refit learns edge-on as normal. A bonded cut
fixes that but also removes κ-donor rows typed as η² C=X (C=S 93 %, C=N / C=O ~30 % unbonded). The owner chose to fold
in without the cut and fix the typing instead (far-field / typing package, PLAN §7).

## 2. Folded in: haptic cap + group bend (amendment 3; production `273f2ccbf892`)
`eta2_group_bend`: on an η² face without the shared plane, a face atom with ≥ 2 substituents bends them as one rigid
group (fixes the geminal collapse the cap exposed: TEBWIH propargyl CH2, H–C–H 0.6°). Arm `capgb` =
`{"embed_haptic_li_cap": true, "eta2_group_bend": true}` vs `pl5`.

| Check | Result | Numbers (pl5 → capgb) |
|---|---|---|
| G7' geometry, A + B + D | pass | geminal < 30°: 0 → 0 (smallest geminal angle 51.4° → 99.8°); edge-on 299 → 294 |
| G4 one-arm failures | pass | none |
| G5 cap-eligible E (55 of 57 solved), top-1 upper bound < 0 | **fail (owner folded in)** | −7.5 [−29.7, +14.9]; floor −6.3 [−19.3, +3.2] |
| non-inferiority (upper ≤ +10), E / not eligible | fail / pass | E as above; not eligible +0.3 [−0.0, +0.8] |
| replay fallbacks | pass | 0 |

The cap's effect is heavy-tailed on parents whose top-1 is already far off: 21 eligible parents better by > 10 kJ/mol
(UZOREI −370, BIXTAF −162, VUMHIU −130), 11 worse (CUGXEJ +352, AYARIB +142, NEZGAB +137, TEBWIH +99); median −2.4.

**Fold-in checks:** full CPU suite 410 passed, 20 skipped, 3 xfailed; replay solver regenerated without change;
golden: 6/245 arrays per profile change, all solves of HUYNEU, SOKVET, YAVFEG (bis-Cp complexes: the haptic cap acts on
the Cp⁻ faces; with the cap off they reproduce the old references exactly; master reproduces them 245/245). The
relaxation before fold-in was waived by the owner.

## 3. Infrastructure found and fixed on the way
- **DB meta sidecar:** `stats.load_db` reads `<db stem>.meta.json` (q5_model, q1_variant). Experimental DBs need it;
  the bench refuses a `JACOBIK_DB` without one; `cj.py` now uploads it (Colab VMs never had the 2026-10-18 sidecar).
  The first holdout pass ran without it and was discarded.
- **QO fit pipeline:** `orientation.fit_rows` is the single training table; `fit.qo` takes its constants from
  `orientation` (they had drifted and reproduced no production key; now all 10,220 within 2e-14). Haptic QO rows carry
  `raw_face_elems`; the archived observations got them by exact face-geometry match
  (`diag/eta2_step6_reextract.py`; the current extractor adds 187 complexes the archive lacks — not used).

## 4. Carried forward
The far-field / typing package (PLAN §7): η² typing of κ donors next to C=X, consistent η² subclass labels, M···C
contacts out of the η² class (then the bonded-cut question disappears), narrow far Q1/Q2 components. NUWFAM (η³ in the
CSD, typed η²) is not a valid η² test case.
