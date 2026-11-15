# E-2 evaluation contract — donor-pair dictionary on cal

**Committed before opening cal.** Worktree `jgen-e`, branch `jgen/e-score`. Candidate flag is the fixed E-1b dictionary `q6_donor_pairs={"hh":0.95,"ho":0.90,"oo":0.80}`. Control uses production defaults. `angle_tail_quadratic` remains off in both arms. Use no relaxation, no `val`, reserve, or locked parents. Exclude only NEPNEF as the invalid input specified by the directive.

## Pairing and units

Run control and candidate on the same Colab L4 VM for each seed 11, 12, and 13. Use the cal panel parent list and the same builder, MACE model, and resource settings within each pair. Export all conformers with `xyz_dir` and atom labels sourced from `conf.complex.atoms`. No local GPU.

## Primary

For each parent and seed, select the highest-scoring delivered conformer and evaluate unrelaxed MACE energy relative to that parent's crystal reference. Compute the paired arm-minus-control ΔE in kJ/mol for all cal parents except NEPNEF. Report the mean, median, and parent-clustered percentile 95% bootstrap CI for the mean using 10,000 resamples and seed 11. Pass only if the CI upper bound is < 2 kJ/mol.

## Secondary

Report paired floor ΔE, obscene rate (`ΔE_unrel > 200 kJ/mol`), number of parents improved, reference coverage at 1 Å under A4 alignment, per-parent Spearman between score×Z and −E over each delivered pool (report arm-minus-control differences), and wall-time ratio.

## Guards

- No parent worsens by >25 kJ/mol on at least 2 of 3 seeds. List each parent meeting this condition and include its score-term breakdown.
- FEKKAJ top-1 improves on all 3 seeds.
- Candidate/control wall-time ratio ≤1.1.

All primary and guard requirements must pass. Report failures plainly. This is a score-fix evaluation; it is not authorization to fold in.

## Evaluation workflow

1. Complete the CPU light gates and L4 replay-parity gates in the directive.
2. Estimate wall time on 1–2 cal parents, then run paired control/candidate jobs on the same L4 VM, filling remote capacity within the two-VM project limit.
3. Fetch, count, and validate every output before stopping each VM. Record Colab spend. Do not open results until the full evaluation output is fetched.
