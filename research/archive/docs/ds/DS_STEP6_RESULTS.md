# DS Step 6 results: generation on the corrected DB, P/S start, damaged-DB trials (2026-11-11)

Job J2 = Perlmutter 61160368 (GPU, `regular`, 1 node, 4 A100). Elapsed 3 min 37 s (estimate 12 min), about 0.06 node-h. Snapshot `ds-51cf9ad`, candidate DB `DF` (sha `be4e0571...`), seeds 11-13.
All six phases rc 0, 39 records per arm (13 parents x 3 seeds), 0 failed parents. Output `$SCRATCH/jacobik/runs/jk-ds-step6-20261111-2344`; local copy `research/data/ds_proto/step6/j2_full`, analysis `.../j2_analysis/step6_summary.json`.
Arms: C = production DB, current start; D = DF, current start; DS = DF + existing P/S start. Contract v2 C, written before these results. Negative top-1 dE difference = lower energy = better.
Analysis note: `step6_analyse.py` first returned n = 0 because the records lacked the `c2_metrics` enrichment (pool-based shell RMSD); fixed in the script (same `enrich_record` as `gate_pm assemble`), no result was read before the fix.

## 1. P/S candidate rule: FAILS (by 0.26 kJ/mol on the first clause)
11 development parents, parent-clustered bootstrap, mean over seeds per parent.

| DS minus D | mean | 95% CI |
|---|---:|---|
| top-1 dE (kJ/mol) | +0.33 | [-3.31, +5.26] |
| floor dE (kJ/mol) | -3.16 | [-5.76, -0.95] |
| shell05 | 0.0 | [0, 0] |
| distinct rows | +0.39 | [-0.49, +1.58] |

Rule: upper bound of the top-1 difference below +5: **not met (5.26)**; coordination not lower: met; no development parent worse than 25 kJ/mol on two seeds: met (no parent has two seeds above 25;
LUQVOK +23/+37/+4.5, ZAQNUE seed 11 +38, USAPOU seed 13 +40 each exceed 25 on one seed only). Contract consequence: P/S does not stay a candidate on this evidence; **J3 runs arms C and D only.**
The lower floor with P (CI excludes 0) is noted and is not used to reverse the rule (it was not the registered criterion). Supplement JEGPUJ/SIVBUV (n = 2, no weight): DS minus D top-1 -12.5 [-25.0, +0.07].
Largest single contributor to the failing bound: LUQVOK (+21.6). A failed rule stays failed; a revised P/S start would need its own registered comparison.

## 2. Corrected DB against production (D minus C, current start; read-out, no registered pass/fail rule in v2 C)
top-1 -13.9 kJ/mol, CI [-41.4, +10.7]; floor -6.1 [-35.1, +17.9]; hit1 -0.03; shell05 0.0 for every parent and arm (every pool row within 0.5 A of the reference shell in both arms, so this metric does not discriminate here).

| Parent | D-C top-1 per seed (kJ/mol) | |
|---|---|---|
| USAPOU | -139 / -72 / -108 | better (known, D2 package) |
| UMUBAF | -115 / -54 / -56 | better (repaired target) |
| SESWIZ | -194 / +9 / +2 | seed 11 is the known ranking-driven spread (identical under both DBs earlier) |
| LIPDEU | -41 / +21 / -15 | mixed |
| LUQVOK, HOXJUZ, MIGTEE, EFOXII | +12/-16/+1; +6/+9/+2; +4/+1/-2; +1/-1/0 | within noise or small |
| **YABYEJ** | **+53 / +55 / +44** | worse on all three seeds |
| **LOPJEF** | +9 / +30 / +29 | worse on two seeds above 25 |
| **ZAQNUE** | -6 / +45 / +27 | worse on two seeds above 25 |

Reading: the three regressions are the ones already recorded for the D2 package in `DS_INVESTIGATION_REPORT.md` (YABYEJ +49, LOPJEF +56, ZAQNUE +38 at the time), attributed there to a missing low-energy basin in the generated pool
(same rows selected within D; "trace the variant-0 starts and optimization before changing fits"), not to a table-induced reversal. DF does not make them worse; it also does not fix them. They are search/pool leads, not a DS-data mechanism (see `DS_STEP7_DECISION.md`).
They will count in the gate's guard if the gate reproduces them.

## 3. Damaged-DB phases: can a weight shift move the generated winner?
Each case: one trial (the worst pilot Spearman per mechanism, fixed in advance), D on the damaged DB against D on the clean DF, same parent, seeds 11/12/13. Top-1 dE change (damaged minus clean, kJ/mol):

| Trial | Change | Coordination of the winner |
|---|---|---|
| LUQVOK Q1 observed-input (2 to 1 component) | -6 / +17 / -1 | top-1 shell RMSD 0.11 A vs 0.10 clean |
| LUQVOK Q1 parent removal (2 to 3) | -9 / +3 / +2 | 0.10-0.11 A |
| **LIPDEU QO observed-input (Spearman -0.20)** | **-52 / -92 / -67** | 0.18-0.21 A vs 0.16-0.22 clean; all pool rows within 0.5 A |
| LIPDEU QO parent removal (3 to 2) | -7 / -19 / -0.3 | 0.21-0.22 A |

- **Yes, a shift of the weights of one key moved the generated winner, and by a lot, for LIPDEU QO (50-90 kJ/mol).** The fixed-pool pilot could not show this (32 rows, nearly flat key).
- **Not in the direction the worry points:** the winner moved to lower energy with intact coordination; no trial produced a higher-energy or newly broken top-1. For LUQVOK, where the pilot collapsed two components into one, the generated winner moved by at most 17 kJ/mol, with mixed sign.
- The clean DF LIPDEU winner is 125-169 kJ/mol above its reference, the damaged one 73-76. So the clean LIPDEU QO entry is not the best for that parent's generated energy: it supports the open lead "broad LIPDEU QO fallback" (all-donor QO for P donors). Not acted on: no adverse loss is demonstrated (Step 7 rule), and these are one trial per case, three seeds.
- DS minus D on the damaged DBs: +21.0, +22.6 (LUQVOK), +19.2, +0.9 (LIPDEU); LUQVOK's P/S penalty is the same as on the clean DB (+21.6), so it is a start effect, not a DB effect.
- Caveat: each damaged DB is one selected trial; the selection was the worst pilot ranking, so this is an adversarial pick, not a distribution.

## 4. What this changes
- Step 7: no mechanism-specific implementation is triggered (D-line of `DS_STEP7_DECISION.md`); the LIPDEU QO and P/S rows are updated there.
- J3 goes ahead with arms C and D (no P arms), as the contract requires.
- Not shown yet: the qualification gate (J3).
