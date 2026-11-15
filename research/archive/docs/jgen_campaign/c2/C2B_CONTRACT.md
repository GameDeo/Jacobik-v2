# C-2b contract — FROZEN, STOP FOR AUDIT — 2026-11-03

Guardrails were registered before new development measurements. The four
finalists, routes, settings and source identities are now frozen in the commit
containing this file. This is the contract-freeze checkpoint: stop until audit
clearance. Seeds 21–23 have not been generated or evaluated.

## Population, inputs and timing context

Use exactly C2_PANEL.json: 120 cal/tail parents, locked exclusions applied
before loading inputs. Primary statistics exclude FEKKAJ and NEPNEF (118
parents); both remain in tables/failure accounting. Population/frame/exclusion
hashes are frozen in that file. Do not open val, reserve or locked inputs.
Use the unchanged production constraints_2026-11-01.json and its meta sidecar.
The final finalist manifest records their hashes and source hashes.

No relaxation: unrelaxed MACE Polar-1-S float64 energies against the crystal
reference, plus symmetric heavy-atom RMSD including the metal and the C-2
shell metric. Keep full-pool xz export and all row-step/failure ledgers.

Owner caveat: development occurred under heavy host/device contention from
concurrent refit/re-extraction, with 100% CPU utilisation. Timing observations
and extrapolations are not intrinsic speed measurements. Pairing cannot
remove time-varying load between serial arms. The numerical wall rule below
is unchanged; confirmation must record load context, report the computed
rule and any timing limitation, and must not rerun to obtain a nicer ratio.

## Declared finalists

Exact machine-readable definitions are in C2B_FINALISTS.json. All unlisted
profile settings remain at the production/default values; these flags are
default off. M=8, T=200; ordinary W=16, S1 W=48; class allocation as C-2.

| Arm | Exact overrides | Budget | Fingerprint | One declared route |
|---|---|---|---|---|
| C | `{}` | three_w16 | `5f74b14b57ed` | control |
| K | `start_kabsch=true` | three_w16 | `38dcb8218b73` | B, targeted |
| P | `start_kabsch=true, start_kabsch_pivot=true` | three_w16 | `e2a5cdba943f` | D, correctness |
| KU | `start_kabsch=true, beam_union=true` | three_w16 | `2285ca6881d1` | B, targeted |
| S1 | `{}` | single_w48 | `5f74b14b57ed` | C, equal compute |

K/P/KU retain start_kabsch_diversity=1.0. KU retains beam_union_share=0.5,
beam_union_tol=1e-6 rad and beam_union_group_batch=16. pool_window_fill=false
in every efficacy finalist. F/FU (diversity=0) are development variants only.
No untested mixture of qualified finalists may be substituted afterwards.

P's correctness proof is
`tests/test_c2_coverage.py::test_seeded_kabsch_translation_uses_original_fk_pivot`.
The before traces miss donor placement
under both the production unseeded fit and the incomplete C-2 seeded fit;
seeded fitting plus the FK-pivot compensation reaches the constructed target.
The panel-wide placement census and actual-FK named traces support its scope.
This is not a claim that better donor placement always lowers MACE energy.

## Frozen statistics and route definitions

Each finalist gets only its declared route; no alternative-route rescue after
opening confirmation. Use per-parent seed means, parent-clustered bootstrap,
10,000 resamples, seed 11, 95% percentile CIs. Undefined quantities are not
imputed; failures remain in coverage/distinct-row denominators and the failure
ledger. Report paired sample counts. Settings: C2B_SETTINGS.json.

All routes require generator end-to-end wall (prepare + seed + solve +
assembly, excluding MACE/export) <=1.3 times control and the guard below.
Ordinary arms compare corresponding seeds 21/22/23. S1 runs three independent
W48 noise replicates; each is compared with the union of C's three W16 seeds.
Charge full preparation to each independent wide replicate; average wide
cost over replicates. Assert equal IK row-steps under the C-2 tolerance.

- A (registered, no declared finalist): floor delta-E CI upper <0 OR <=1 A
  coverage CI lower >0; top-1 CI upper <+2 kJ/mol.
- B (K, KU): floor and top-1 delta-E CI upper <+2 kJ/mol; mirror count >=null+3
  AND >=2*trips. Counts use the artefact-excluded guard population.
- C (S1): floor and top-1 CI upper <+2 kJ/mol; <=1 A coverage CI lower >-3
  percentage points; AND either size-matched rho delta CI upper <0 or
  distinct-row delta CI lower >0. Require matching row-step accounting.
- D (P): the stated before/after correctness proof; floor and top-1 CI upper
  <+2 kJ/mol. No improvement required.

Rho is score vs MACE energy (negative is preferable). Preserve the C-2 union
correlation definition: concatenate C's exported rows across its three seeds,
including cross-seed duplicates; symmetric dedup is separately used for the
union distinct count. For each wide replicate, subsample the larger pool to
the smaller row count without replacement, fixed seed 11, 20 draws; average
defined paired rho deltas over draws, then over replicates within parent.
One-row/constant pools stay undefined. Do not replace undefined correlations
with zero. These details are fixed before the unseen confirmation.

Guard: trip if top-1 is >25 kJ/mol worse on >=2/3 seeds; mirror reverses the
sign. Null=max of the two within-arm seed-rotation trip counts. Report C's
rotation null too. Fail iff trips>null+2 AND trips>=mirrors. Report both with
and without NUHPEL, MIDQEY, UMUBAF, YUCZUS, NUWFAM, CAFWIQ; decision uses
without. Those parents stay in every other table. This list was registered
before C-2b data; no additions after confirmation.

## Once-only confirmation after audit clearance

Do not execute this phase at the contract-freeze checkpoint. After explicit
audit clearance, use one 2060 invocation for control plus all declared
finalists, seeds 21–23, full frozen population:

```bash
python -m tools.jgen.c2.driver --out <C2b>/confirmation \
  --arms C,K,P,KU,S1 --seeds 21,22,23 --confirmation-after-audit
python -m tools.jgen.c2.finalise --root <C2b>/confirmation
python -m tools.jgen.c2.enrich --root <C2b>/confirmation
```

For a long unattended run, prefer completion events or scheduled checks about
20 minutes apart, following the owner monitoring preference; short checks are
for errors, checkpoints and immediate dependent work.

Use c2b_analyse with C2B_SETTINGS.json and each declared route; P supplies the
correctness-proof flag only with the proof above retained. No development
subset flag. Record nvidia-smi and host load. A problem is reported, not fixed
and rerun. Every finalist gets a result, qualified or not. Named cases are the
C-2 list plus SIVBUV, MIVXAR and LOPCAW, per seed, top-1/floor and failures.

Optional L4 replicate only after local qualification: paired control on the
same VM/type/invocation, max two L4s and hard cap 12 units, log credits before
and after, fetch/count then immediately down only owned VMs. None is started
at this checkpoint.

Qualified search finalists enter D3's same-machine/same-seed 2x2:
control / D only / search only / D+search. No standalone fold-in here; the
owner decides after that joint gate.
