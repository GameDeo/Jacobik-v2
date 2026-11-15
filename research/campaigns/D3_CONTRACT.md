# D3 gate contract — DRAFT, not frozen (2026-11-04; freeze when D2 lands)

Prepared ahead so D3 starts the moment D2 produces the new DB. The implementer copies it to
`research/campaigns/D3_CONTRACT.md`, fills the bracketed items from D2/C-2b/I results, commits it **before any arm
runs**, and changes nothing else.

## Identity
- Control C: production `5f74b14b57ed`, DB `constraints_2026-11-01.json` + `.meta.json`.
- D: new DB `[constraints_2026-11-0X.json]` + meta (D2 build, reproduced from scratch) + surviving D flags `[list]`,
  fingerprint `[fp]`.
- S: P: `{"start_kabsch": true, "start_kabsch_pivot": true}`, fingerprint `e2a5cdba943f` (audit Addendum 2).
- DS: D + S.
- Isomer arms join only if the I package qualified before D3 starts: `[pending I audit; no isomer arm registered here]`, else they gate separately.

## Runs (Colab L4, paired in one invocation per VM; `jgD1`, `jgD2`; cap 20 units for the real D3 (D3-H audit; launch still requires D2/freeze); estimate below)
| Stage | Panel | Seeds | Arms |
|---|---|---|---|
| 1 | cal + tail minus `locked_exclusions.json` (120 parents; primary 118) | 11–13 | C, D, S=P, DS=D+P |
| 2 | η² holdout: 110 `surface_fit` parents by `diag/eta2_step4_holdout.py` (seed 31), as `ETA2_CONTRACT.md` | 11–13 | C, D |
| 3 | **jgen promotion reserve** (80 groups, locked; `LOCKED_SETS.md`), **once**, only if stage 1 qualifies; the auditor supplies the code list at run time, agents never read the partition file; includes a C-vs-C repeat for its null | 11–13 | C, C′, D (or DS, or P under the registered fallback) |

Full-pool xz export, unrelaxed MACE Polar-1-S float64, symmetric heavy RMSD, shell metric, row-step ledgers, nvidia-smi
and load logged.

## Metrics and statistics
As C-2/C-2b: floor ΔE, top-1 ΔE, coverage ≤ 1 / ≤ 0.5 Å, shell ≤ 0.3 / 0.5 Å, distinct rows, pool ρ (size-matched where
pools differ), wall. Per-parent seed means; parent-clustered bootstrap, 10,000 resamples, seed 11; 95 % percentile CIs;
failures stay in denominators. Primary exclusions FEKKAJ, NEPNEF (or as D-A's charge census rules; state which).

## Gates (D vs C on stage 1; same for DS vs S and DS vs C if present)
1. **Energy:** top-1 ΔE CI upper < +2 kJ/mol **and** (floor CI upper < 0 **or** top-1 CI upper < 0).
2. **Guard:** C-2b definition (trip > 25 kJ/mol worse on ≥ 2/3 seeds; seed-rotation null; fail iff trips > null + 2
   and trips ≥ mirrors), artefact list NUHPEL, MIDQEY, UMUBAF, YUCZUS, NUWFAM, CAFWIQ reported both ways; decision
   without them. **Plus** the far-field guard: catastrophic count (top-1 > 200 kJ/mol) must not rise.
3. **Near-basin curvature (D3-H amendment 3):** use Q1–Q7, including Q7, for every arm on the fixed u1 directions.
   Gate the combined pool+crystal view. If the Q7-inclusive control has median 1.15–1.40 and rank ≥ 0.80,
   require the arm in that same band. If the control itself fails either bound, require arm median within ±0.10
   of control and rank ≥ control − 0.03. These relative bounds are inclusive. Both scopes (Q1–Q6 and Q1–Q7),
   and crystal/pool strata, are reported; only the Q7-inclusive combined view gates.
4. **η² holdout (stage 2):** top-1 CI upper < +2 kJ/mol; edge-on η² row count not above control.
5. **Wall:** generator end-to-end ≤ 1.3× control.
6. **Promotion reserve (stage 3):** top-1 CI upper < +2 kJ/mol and guard pass against its own C-vs-C null. Opened once;
   the opening is recorded in `LOCKED_SETS.md` "Openings" by the auditor.

**2 × 2 reading (registered):** with S present, report the interaction (DS − D) − (S − C) with its CI. The fold-in
candidate is whichever of D or DS passes all gates with the better top-1; ties go to D (fewer moving parts).
**P fallback (D3-H amendment 4):** if neither D nor DS qualifies, P may qualify alone using its existing C-2b
correctness route D qualification, provided (1) the **top-1** interaction CI upper is < +2 kJ/mol and (2) stage 1
S=P versus C top-1 CI upper is < +2 kJ/mol. The floor interaction remains reported. The D3 improvement-CI gate is
not imposed again on P. Both contrasts must exist; missing/DRYRUN/reproduction data cannot authorize fallback.
The machine candidate is `S`, labelled `P`, with a separate `p_fallback.qualified` decision for stage 3. If the chosen
candidate is DS, P is included in DS. A qualifying D/DS takes priority over P fallback. Stage 3 retains its own gates.

## Failure rule
A failed gate triggers the registered ablation, once: drop one flag group at a time, in this order: far-field form,
physical Q6, Q7 bound centre, typing/OS. Rerun stage 1 for each ablated arm only. No retuning of any setting.
Report every arm.

## Named cases
D0-a shortlist (MIDQEY, FEKKAJ, DAMVAQ, USAPOU, HEDJOP, NUCKUS, VOBGUR, QEZVEY, NUWFAM, JEGPUJ); lenient parents
(QEZVEY, VAVBIE, USAPOU, AZUSIA); CBTYCO, aqua_La3 (golden watch); FELSAU, CAXTAX; per seed, top-1 / floor.


## Harness details (draft, before any D3 result)

- Primary exclusions are FEKKAJ and NEPNEF, unchanged from C-2b.
- Stage 2 uses the existing seed-31 draw A/B/C plus amendment D (40), exactly 110 unique parents; no redraw.
- The VM starts one persistent subprocess per arm. Each loads its own DB, flags and MACE model once, builds the
  parent-specific scorer once for each parent, and completes all arms on a parent before moving to the next.
  GPU work is sequential; idle workers release transient CUDA allocations while retaining their models.
  Generation wall follows C-2b allocation; startup, process walls, PIDs and model/DB load counts are separate.
- Curvature: CPU `forces.py` ranking-term central differences on the saved u1 dev32+n48 directions; positive-direction
  median energy/kT : score ratio over pool+crystal; Spearman median within point. Both strata are also reported.
  The inclusive scope appends Q7 to `forces.py`'s term list, on the same score × aggregation-norm scale.
  The legacy Q1–Q6 scope is also recomputed on the same evaluations for direct comparison; no gate CLI switch.
  **Q7 ruling (audit 2026-11-06 §4):** scope is Q1–Q6 plus the Cartesian `Scorer` Q7 term.
  The FK bend prior is declared out of scope. Do not add bend directions or reconstruct bend coordinates.
  Affirmative coverage requires `q7_scorer_in_scope=true` plus recorded, matching effective FK prior settings
  (`bend_penalty`, scale, sigma, weight, free zone and production bend/haptic profile controls) across arms.
  Local launch refuses missing/different identities; each isolated VM worker verifies its runtime defaults and
  arm flags against that identity. Curvature summaries carry the same identity and must match the arm and control.
  The control baseline remains median **1.3315397846869683**, rank **0.8639999488508491**; its Scorer Q7 is zero.
  Existing summaries without the coverage/identity fields must be regenerated before a real launch.
- With `q7_bound_centre` enabled, the DB's sibling `<stem>.bends.json` is mandatory, in addition to `.meta.json`.
  Record its SHA256 as `bends_sha256` in arm identities/manifests and curvature summaries. Ship the exact sidecar
  beside the DB on the VM, verify its remote SHA256 before starting workers, and verify it again inside the arm
  worker before loading the DB/model. Missing or changed files refuse launch; no silent Q7 fallback is accepted.
- Stage 1 qualification is provisional for opening stage 3. Final fold-in also requires the separately reported
  stage 2 holdout and stage 3 gates. DRYRUN and reproduction records can never qualify.
- D2 must supply DB + reproducible meta (`build.code_commit`, `build.flags`, `build.input_hashes`, or those at top level),
  exact D/DS fingerprints, and per-group ablation recipes including reproducible DB paths for DB-changing groups.
- Cost uses the C-2b C/P mean 3.912327849419389 s per parent/arm/seed, with conservative L4 factor 1.0.
  Startup is charged once per arm/VM, max(20 s, observed smoke maximum); setup 10 min per VM, planning rate 1.9 units/h.
  Successful smoke process/MACE/export/enrichment overhead is added per parent/arm, explicitly based on only three
  parents. `D3_H_AMENDMENTS_REPORT.md` records the updated estimates. A second VM adds setup and per-arm startup;
  it partitions parents, never arms, and does not change contrasts. The real D3 cap is 20 units.
- Named cases additionally include JEGPUJ and SIVBUV; cases outside the registered panel are reported missing.
