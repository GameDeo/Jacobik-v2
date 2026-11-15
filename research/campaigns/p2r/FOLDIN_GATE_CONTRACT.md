# P2R fold-in gate: evaluation contract (draft for the owner, 2026-11-13)

Written before any fresh-seed result exists. Nothing here has been run. Both jobs are billed and need the owner's go
per job, with the QOS named.

## Question
Does the P2R package, as one configuration, improve coverage of the reference class and basin without costing top-1
energy, robustness or wall time?

## Arms (same job, same node, same GPU)
| Arm | Fingerprint | Flags |
|---|---|---|
| C | `d5e2f4113497` | production (all five off) |
| P | `a47283387172` | `out_mirror`, `ring_states`, `ring_states_five`, `config_states`, `config_near` on |

Code: branch `jgen/p2r-final` (flags default off; the arm switches them). `config_remote` stays off: no cal or tail
parent has an element it can derive, so it cannot be gated on this panel.

## Panel and seeds
cal + tail (120 parents), fresh seeds 31, 32, 33, used once. Targets from production typing; primary root from
`research/campaigns/ds/gate_eligibility_v1.json` (all-root reported beside it).
Unsolved units count as misses in both arms. Per-parent cap 5 minutes; a timeout is recorded, never retried.

## Jobs
| Job | Node | What | Estimate |
|---|---|---|---|
| 1 | one A100 (GPU node) | `tools.run panel`, arms C and P, seeds 31-33, no control cache (fresh control) | about 25 min: one arm and half took 5 min on a free login A100 on 2026-11-13 |
| 2 | one CPU node | per arm: `a4b.rerun post --select a4c` (crude relaxation, the registered metric), then `a4c.relax_level --level normal`, then `a4c.paired` at both levels | about 25 min: 13 min crude (job 61204770) plus about 4 min per arm-half at the normal level |

## Gates (all must hold; parent-clustered bootstrap, 10,000 resamples, 95 % interval)
1. **Coverage, registered metric (crude relaxation):** paired difference in C-MISS, P minus C, below zero with the upper
   bound below zero.
2. **Coverage at convergence (normal relaxation):** paired difference in C-MISS not above zero (upper bound at most
   +0.02).
3. **Top-1:** paired difference in C-TOP1 not below zero by more than 0.03 (lower bound), and the paired median change
   of the unrelaxed top-1 MACE energy against the reference at most +2 kJ/mol (upper bound).
4. **Failures:** no parent fails or times out in P that solved in C. Control-only failures are reported, not held
   against P.
5. **Wall:** total solve wall of P at most 1.3 times C on the parents both solved.
6. **Garbage:** the share of relaxed rows with a topology change or more than +50 kJ/mol over the relaxed reference
   does not rise (upper bound of the paired difference at most +0.02).

## Reported, not gated
- Gains and losses per unit with their source (reflected row, ring state, extra class).
- `strict_any` and `strict_all`; classes per unit.
- Settle cost per arm (`a4c.settle` on the normal-level rows): cycles, energy released in the crude step and after it,
  heavy-atom RMSD to the own minimum.
- The six contact-set parents (DAMVAQ, GOTWET, HEDJOP, HOXJUZ, JUCBIV, NUWFAM) separately: see `FINAL_PASS.md`.

## If it passes
Merge `foldin/p2r` (defaults flipped, fingerprint `a47283387172`, pins updated). The golden recorded for that
fingerprint replaces `research/data/golden/` at the merge; the current one moves to
`research/archive/data/2026-11_golden_history/golden_d5e2f4113497/`.

## If a gate fails
It is stated as failed. No retuning on these seeds.
