# Stress driver pilot — LUQVOK Q1, 2026-11-11 (workstation, no billed compute)

Chain exercised end to end: registered design (`registry.register`) -> perturbed raw table (`confidence.perturb_observations`) ->
exact closure refit (`stress.refit_resolved`) -> fixed-pool re-scoring of the saved D3 `D`-arm pools, seeds 11-13, under the refitted
entry (`tools.jgen.ds.stress_trial`, `Traced.override` on every requested lookup key that resolves to the key) ->
`confidence.evaluate` severity rule. Resolved key `('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')`, 71 supporting parents.

| Trial | Refit means (Å) | Weights | Selected-energy change vs clean D2, seeds 11/12/13 (kJ/mol) | Severe |
|---|---|---|---|---|
| remove 1 random parent (x2) | 2.056, 2.657 | 0.843, 0.157 | 0, 0, 0 | no |
| remove 4 random parents | 2.057, 2.657 | 0.836, 0.164 | 0, 0, 0 | no |
| remove the 11 erroneous Pd parents | 2.010, 2.072 | 0.252, 0.748 | +17.3, -131.8, -129.0 | no (improvement on 2 seeds) |

The last row reproduces the control table (C: 2.00979, 2.07165; 0.25214, 0.74786) and the earlier finding that C scoring improves the selected
energies by about 132 and 129 kJ/mol on seeds 12 and 13. Random damage of up to 4 of 71 parents leaves the erroneous long component and the
selection unchanged: the long component is robust to random removal and disappears only when its 11 supporting parents go.

Cost: the exact refit dominates (about 115 s per trial, the legacy sampler's shared random stream forces a full 600k-row expansion each time);
fixed-pool scoring is 0.3 s. With stable ancestor sampling (D-sample, D-both) the expansion restricts to the chain's feeding rows
(`stress.feeds_closure`), expected seconds per trial. Full design for one case, 96 trials per scenario: about 3 h serial on D2 tables, minutes packed.

Provisional parts: the broken-coordination threshold (`stress_trial.BROKEN_RATIO = 1.25` times the source donor distance; the clean D2 pool already
has broken rows under it, only new breaks count) is fixed in the Step 5 contract before results are opened; the geometry scenario is refused by
the driver (needs coherent re-extraction) and therefore always recorded as a failed trial.
