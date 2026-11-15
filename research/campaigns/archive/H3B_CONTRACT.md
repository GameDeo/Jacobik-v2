# H-3b final confirmation contract

Frozen before confirmation results, 2026-10-31. This replaces the donor-only
contract at f52ab5a, withdrawn when the owner requested further cap work.
Implementation: 63d2bb2; the complete contract/source commit will be recorded
by the launcher and every benchmark record. Local PC / RTX 2060 SUPER only.
No Colab, merge, promotion or production-default change.

## Final design

- Donor: unchanged E-1b crystal-derived `q6_donor_pairs={"hh":0.95,"ho":0.90,"oo":0.80}`.
- Cap: eligible anionic haptic components. Reject locally neutral zwitterions;
  skip aromatic monoanions with no face-bound H; preserve multiply charged
  components and H-bearing faces. Heteroatom-localized charge uses existing
  kappa Li placement. Aromaticity is perceived on a copy of Kekule inputs.
- Close the geminal-H orientation loophole with the **existing** fitted
  face-mean statistic, and measure the **existing** geminal lean-wall slots
  with unwrapped face-relative angles. Keep all independent DOFs and original
  wall widths/stiffnesses. Device face indices preserve graph capture; undefined
  polar geometry falls back to the original wall with finite gradients.
- Drop the original `angle_tail_quadratic` design. Its fixed empirical-scale
  cutoff penalizes valid crystal geometry (WEYJIT) and causes consistent local
  regressions. FEXWUC's tail-associated win is a known cost, not a retained win.

## Arms and execution

The exact four arms are in `H3B_ARMS.json`: control, cap alone, donor alone,
and combined donor + cap. No tail flag. Each arm runs in a fresh process so
all import-time consumers receive its flags. Benchmark baseline settings:
nci=false, deterministic_embed=true, finish_phase=None, replay enabled,
normal production CUDA math. The control fingerprint must be `6cc586e17ab1`.

Paired frozen cal + tail panels, seeds 11/12/13, same machine and GPU.
Manifest: `research/data/refit_nci_v2/scaled_v1/manifest.json`, SHA-256
`5c3e5897f099eac36eb52bf14b67ecd8fd299e3819c83c8d0d5daa0d0f0ce450`.
NEPNEF remains the pre-existing invalid-input exclusion. Retain every failed
record and all new failures; no post-hoc exclusions or tuning. Never use val,
reserve or locked panels. Expected: 1,560 records, including 24 NEPNEF failures
if the pre-existing failure reproduces (1,536 valid records).

## Frozen endpoints

Primary: combined-minus-control cal mean top-1 MACE delta (kJ/mol), 95%
parent-clustered percentile bootstrap CI, 10,000 resamples, RNG seed 11.
Resample parents with all three seeds together. **PASS only if upper CI < 0**.

Per-parent guard: no parent >+25 kJ/mol worse on at least two of three seeds,
for cal and tail, in combined and each surviving flag alone. Unexpected
missing/failed pairs fail completeness; do not silently estimate past them.

Individual evidence: report cap-alone and donor-alone cal CIs. Claim each
flag beneficial on its own only if its upper CI is <0 and its guards pass.
An uncertain individual CI is explicitly unproven even if the package primary
passes. Tail CIs are reported; their width is not a primary failure gate.

Secondary: floor delta/CIs; named FEKKAJ, NUWFAM, QUTWOT and EPUWAQ results;
>200 kJ/mol rate and transitions; full-pool 1-A coverage; score-versus-MACE
correlations; predeclared ex-FEKKAJ cal sensitivity; solve and end-to-end wall
times; failure ledger; replay fallback
counts and geminal geometry. Report the sacrificed FEXWUC tail gain plainly.
No secondary endpoint replaces the primary or excuses a failed guard.

## Verification and compute gates

Production defaults must retain fingerprint 6cc586e17ab1 and golden 245/245
for both profiles. Full CPU suite must pass, including restored logging
isolation. Replay: identical-start, 200-step checks on FEKKAJ/NUWFAM/QUTWOT,
actual FAST_FK path, under deterministic CUDA algorithms and deterministic
cuBLAS. Require <=1e-6 raw relative score error and identical returned rows.
Also report the ordinary-CUDA strict-parity failures and eager-repeat noise;
do not change the numerical tolerance or hide those failures. Confirmation
uses normal CUDA math, paired for all arms.

Local development suggests approximately 2–4 hours for four arms on cal+tail,
conservatively below the 10-hour cap. Confirm against completed development
timings before launch; stop for owner decision if estimate exceeds 10 hours.
One local GPU process at a time; at most three concurrent CPU workers.

Stop at **Checkpoint H-3b** after analysis/report/commit, whether gates pass or
fail. No confirmation-driven retuning. Colab spend: **0**.
