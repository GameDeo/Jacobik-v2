# E-2 report — donor-pair score fix on cal

Branch `jgen/e-score`; evaluation contract commit `5b2b927` (`E2_CONTRACT.md`) was made before opening cal outcomes. No fold-in was performed. Production fingerprint remains `6cc586e17ab1`.

## Step 0: light gates

| Check | Result |
|---|---|
| Flags off | Production fingerprint `6cc586e17ab1`; golden 245/245 arrays identical for production and u1 (`research/data/jgen/campaign/E/e2_golden_flags_off.log`) |
| `angle_tail_quadratic` crystal check | 120 requested cal parents, 996 Q2 pairs, 0 failures; 63/996 pairs (6.325%) have nearest-component `|z|>2`; extra cost median 0.00000, p95 0.01510 nats; 267 pairs have positive extra cost |
| Donor-pair replay, XEYSUP | Replay mode active; top key equal; CPU/CUDA scores 1.41276693 / 1.41276610 (absolute difference 8.3e-7) |
| Donor-pair replay, USAPOU | Replay mode active, but top key differs (`[null,2,141]` CPU vs `[null,0,51]` CUDA); scores −0.914021 / −0.472851 (absolute difference 0.44117). **Top-row parity failed for this unit.** |
| Donor-pair replay, YOVNEC | Replay mode active; top key equal; CPU/CUDA scores 1.20854890 / 1.20853639 (absolute difference 1.25e-5) |
| Tail replay, FEKKAJ off | Replay mode active; CPU/CUDA scores −0.40901279 / −0.40993643; same top key |
| Tail replay, FEKKAJ on | Replay mode active; CPU/CUDA scores −0.41311198 / −0.41311324; same top key. The flag changes the top trial from 1120 to 1134, and CPU/CUDA agree on the flagged top key. |

The donor-pair dictionary was active when the replay graph was recorded (`3f75c9525a4e`), and all three CUDA runs reported replay mode with no replay error. The USAPOU key and score disagreement means exact CPU/CUDA top-row parity is not established across all three cases, even though graph replay itself did not fall back. Q6 capture is supported by the active flag fingerprint, the nonzero donor-pair counts (XEYSUP 7, USAPOU 3, YOVNEC 6), and successful replay mode; the numerical mismatch on USAPOU remains a spec issue.

Artifacts are under `research/data/jgen/campaign/E/`: `e2_tail_crystal_check.json`, `e2_q6_parity/e2_q6_parity.json`, `e2_fekkaj_tail_off/e2_fekkaj_tail_off.json`, and `e2_fekkaj_tail_on/e2_fekkaj_tail_on.json`.

## Cal runs

The paired panel contained 119 eligible parents and seeds 11–13 (357 units per arm), excluding only NEPNEF. The fixed candidate was `q6_donor_pairs={"hh":0.95,"ho":0.90,"oo":0.80}`; control used production defaults. Both arms ran on the same Colab L4 VM within each shard. All 357 records in each arm have status `ok`.

### Primary and secondary metrics

Parent bootstrap resamples use 10,000 draws with seed 11. Deltas are candidate minus control; negative energy is favourable.

| Metric | Result |
|---|---:|
| Top-1 ΔE, parent mean (95% CI), kJ/mol | −3.165 [−8.452, −0.232] |
| Top-1 ΔE, parent median (95% CI), kJ/mol | −0.00008 [−0.01233, +0.00027] |
| Floor ΔE, parent mean (95% CI), kJ/mol | −0.204 [−0.762, +0.356] |
| Floor ΔE, parent median (95% CI), kJ/mol | −0.012 [−0.256, −0.00004] |
| Obscene rate (`ΔE > 200 kJ/mol`) | control 14.01% (50/357); candidate 13.73% (49/357); parent-mean rate delta −0.28 pp, 95% CI [−0.84, 0.00] pp |
| Parents with lower mean top-1 ΔE | 65/119 (54 worse, 0 tied) |
| Per-parent Spearman(score×Z, −E) delta | mean +0.0058, median +0.0020; mean 95% CI [−0.0072, +0.0194] |
| Summed solver wall ratio, candidate/control | 1.0014 |

**Reference proximity at 1 Å:** exact A4 symmetry-aware heavy-atom RMSD was computed against each input crystal, after checking XYZ element order. On saved top-1 rows, 104/357 control units (29.1%) and 100/357 candidate units (28.0%) are within 1 Å; at least one seed hits for 45/119 control parents and 43/119 candidate parents. On saved floor rows, the counts are 123/357 and 126/357, respectively; at least one saved top-1 or floor hits for 59/119 and 61/119 parents.

**Coverage limitation:** the current benchmark runner writes only top-1 and floor XYZs to `xyz_dir`; it does not preserve every pool coordinate. The 1 Å counts above therefore describe these exported selections, not full-pool coverage. The full-pool Spearman calculations use all stored score and energy rows. This is recorded as a spec limitation rather than treating selected-pose hits as full-pool coverage.

### Guards and gate

- **No parent** worsened by more than 25 kJ/mol on at least two of the three seeds. No guard parent required a term breakdown.
- **FEKKAJ top-1 improved on all three seeds:** ΔE was −279.675, −322.808, and −249.160 kJ/mol for seeds 11, 12, and 13.
- **Wall ratio:** 1.0014 from summed per-unit `wall_solve_s`; below 1.1.
- **Primary:** 95% CI upper bound is −0.232 kJ/mol, below +2 kJ/mol.

Therefore the **cal non-inferiority and fixed-case gate passes**. The USAPOU replay top-row mismatch and selected-XYZ-only coverage limit remain open spec issues; no production fold-in is authorized or made here.

## Run provenance and issues

- Runtime estimate: FEKKAJ + XEYSUP, seeds 11–13, on `jgE2`: control 64.7 s; donor-pair 62.0 s. The full panel was split into 59 and 60 parent shards on `jgE2` and `jgE3`, with each pair run on one L4. Each arm has 357 `ok` records and 714 labeled XYZs (top-1 and floor).
- Fingerprints: control `6cc586e17ab1`; donor-pair `3f75c9525a4e`. The donor-pair flag did not change production defaults.
- E-2 Colab spend was approximately 3.08 units (balance 499.23 → 496.15); zero VMs remain active. No local GPU was used.
- A launch attempt failed before solving because the Colab command stripped JSON quotes in `--flags`; it was corrected, and that failed launch's output was not mixed into the panel.
- Runner logs recorded nonfatal warnings for LOPCAW's heuristic SMILES alignment fallback, infeasible baseline assignments replaced with feasible assignments, ETKDG retry success, and XEYSUP kekulization messages. All corresponding panel records were delivered with status `ok`; no batch stopped.

### Artifact paths

- Contract and code: `research/bench/jgen_campaign/E2_CONTRACT.md`; `research/bench/tools/jgen/diag/e2_analyze_cal.py`.
- Paired run records and XYZs: `research/data/jgen/campaign/E/e2_cal_{a,b}_{control,candidate}/e2_cal_{a,b}_{control,candidate}/`.
- Summary metrics: `research/data/jgen/campaign/E/e2_cal_summary.json`.
- Step 0 outputs: paths listed above under `research/data/jgen/campaign/E/`.

Assisted-by: Codex on jgen/e-score
