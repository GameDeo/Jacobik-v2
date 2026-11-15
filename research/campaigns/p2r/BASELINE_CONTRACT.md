# P2R Step 2: coverage baseline on production `d5e2f4113497`, evaluation contract

Written 2026-11-12, before any Step 2 solve or relaxation result exists. Metric definition: `METRIC_REPORT.md` (A4c).
Plan: `.claude/PLAN_p2_restart_2026-11-12.md`, Step 2.

## What this is
This is a measurement, not a gate. Nothing passes or fails. It is the first measurement of the class-aware coverage metric on
current production. It serves three purposes:
- the fresh control for every Phase 2 arm;
- the A4b rerun owed since the D fold-in;
- the input to Step 3, where every miss is classified.

## Fixed inputs
| Item | Value |
|---|---|
| Production | fingerprint `d5e2f4113497`, master code; tools from branch `jgen/p2r-baseline` (research tools only; `src/` == master) |
| DB | `data/db/constraints_2026-11-12.npz` (sha256 `62c58d53…`, embedded meta) |
| Panel | stage-1 cal + tail: 120 parents (`vm_split.json` cal 120 + tail 10, 5 overlap, minus 5 locked codes) |
| Inputs | `charge_mult.json` `98fb6c42…`, `vm_split.json` `7bf3fd4c…`, `fallback_codes.json` `12ec0c93…` (copied from A4b 2026-11-01) |
| Eligibility | `research/campaigns/ds/gate_eligibility_v1.json` `a8d7a55b…`, stage 1. The primary root (valid + repaired) has 109 parents. Ambiguous (8) and unrepaired (3) are reported as their own strata. Failed or missing parents stay in every denominator |
| Seeds | 11, 12, 13 |
| Solve | `tools.run panel`, control arm, no flags, `--mps --no-pack`, on a Perlmutter login A100 (free). Per-parent cap as in the runner defaults; timeouts are recorded and never retried |
| Relax set | `a4c.relax_select`, at most 5 rows per unit: top-1, floor, up to 3 same-class rows by core RMSD ≤ 1.5 Å, then one representative per other class. Rows dropped by the cap are logged in `selection.json` |
| Relaxation | GFN2-xTB crude, 200 cycles. MACE-POLAR-1-S float64 single points on CPU. One billed CPU node (`p2r_baseline_cpu.sbatch`, debug QOS, 30 min cap; needs the owner's go) |
| Metric | A4c, core depth 2: same class AND (core RMSD ≤ 0.3 Å OR MACE offset ≤ +5 kJ/mol). Enantiomers are distinct classes |

## Reported
Every figure is a parent-clustered bootstrap (10k resamples, seed 20261112), reported for cal and tail separately and for
the primary root, plus per stratum.
1. **Coverage (headline):** the C-MISS share, i.e. no relaxed row of the unit is a hit. Also C-TOP1 and C-RANK shares.
2. **Mirror-completed miss:** units that also count as covered when a relaxed row is the reference's enantiomer at
   hit level. This is the expected gain from Step-4 mirror completion. It does not count as coverage.
3. **Top-1 energy (owner criterion):** the share of units whose relaxed top-1 is at or below the relaxed reference, in any class. Two thresholds are
   reported: ≤ 0 and ≤ +5 kJ/mol.
4. **Seed dependence:** `strict_any` (the parent is covered on some seed) and `strict_all` (covered on every seed). This is a diagnostic only.
5. **Miss split (input to Step 3):** C-MISS units are split into "not offered" (no unrelaxed pool row is the reference
   class) and "offered" (the class is in the pool, but no relaxed row hits). Both lists are given by unit.
6. **Class completeness:** classes per unit in the unrelaxed pool, and units where the cap dropped rows.
7. **Garbage:** the share of relaxed rows with a topology change or an energy more than +50 kJ/mol above the relaxed reference.
8. **A4t:** full-pool relaxations of the cal parents that are C-MISS on every seed. They show whether any pool row relaxes
   into the reference class.
9. **Run facts:** units expected vs found, failures and timeouts, wall time per stage, MACE count.

## Comparability (stated, not adjusted)
- The retrospective (C-MISS 0.319 on cal) used old production `273f2ccbf892`, the old relax set (≤ 1.5 Å full RMSD) and
  rules revised on those same rows. It is context only and is not a paired comparison. A paired claim needs both arms
  run here.
- The A4b R-buckets in `a4b/units.csv` are computed over the A4c relax set. They are not comparable to the
  2026-11-01 A4b tables.
- Solve platform: Perlmutter login A100 (shared). Any later arm compared against this control runs on the same platform
  type and is relaxed with the same pipeline.

## Stop rules
- Any unit built without GFN2 (the log line "GFN2 embed refinement failed") makes the run invalid. The run is redone, not
  filtered.
- If more than 5 % of units are missing or failed, the numbers are reported with the gap stated, and no Step 3 targets are
  set from them.
