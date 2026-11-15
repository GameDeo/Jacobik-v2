# G-3c implementation and H-1 score patch — 2026-10-30

Branch `jgen/b-solver`; master merged as `149e319` (master `9bb6831`).
Owner clarification: follow the Package H amendment. G-3c step 0 remains deferred
until G-3d on the patched base. No evaluation panels or energy comparisons run.

## Implementation

`finish_phase` defaults to None. Its dictionary requires explicit `release`
(subset of shape/symmetry/projection), per-class `sigma`, and `cap`. `search`
and `optimizer` accept adam/irprop+/irprop-. `sigma_source` accepts thermal or
curvature (four fixed seed-11 Hutchinson probes, score Hessian converted to
nats; nonpositive/nonfinite estimates retain the class width). `switch=100`
provides IA; `score_only=true, search=irprop-, optimizer=irprop-` provides R2s
and disables CPU culling. iRprop search initializes delta from the production
learning rate (delta0=lr, delta_max=lr/0.14, delta_min=0.001*lr/0.14),
with the existing per-DOF update scales; the finish resets to thermal bounds. These policy values remain experimental, pending
step 0. Test widths/caps are correctness fixtures, not selected policy.
The curvature option evaluates its fixed probe block every step for a fixed
CUDA graph and masks width writes to switching rows. Its eventual cost gate
must charge that full overhead; no cheap-curvature or <=10% claim is made.

Both native and generated replay loops mask aids by row, restart per-row
optimizer state, use 0.14 sigma steps and [0.001 sigma, sigma] iRprop bounds,
and freeze converged rows. Total iterations are capped at 250 when the flag is on.
Optional diagnostics retain the last 20 sign-flip masks per DOF; deltas and
curvature-validity masks remain available through the native trace hook, enabling
the deferred per-DOF-class floor/flip report without host sync in replay. The stop uses score gradients in **nats**, reversing
the scorer's normalization, with the fixed 0.1 kT threshold. Host batch checks
occur every eight steps. CPU culling selects the phase state with the rows.

## H-1 term census and crystal gate

Q1: Gaussian mixtures and an existing quadratic upper wall; no heavy-tail
component to replace. Q2: Student-t mixtures; existing angle_tail_quadratic
continuation applies beyond z0=2. Q3/Q4: Gaussian mixtures (S_orient replaces
these with its Gaussian joint model when available); no Student-t continuation.
Q5: von Mises mixtures have a bounded, flattening far field. Bend prior:
flat-bottom harmonic, quadratic outside the free zone; it does not flatten in
the far field. No new rule applies to Q1/Q3/Q4/bend prior.

Crystal checks use cal geometry only and the original 2026-10-17 statistics DB,
privately extracted from the archived final.tar.zst; protected inputs and golden
references remain untouched. Failures are recorded per parent.

| Term | Features | Extra nats median | p95 | Share beyond z0 | Included |
|---|---:|---:|---:|---:|---|
| Q1 (unchanged Gaussian/wall) | — | 0 | 0 | n/a | No new rule |
| Q3/Q4/S_orient (unchanged Gaussian) | — | 0 | 0 | n/a | No new rule |
| Bend prior (unchanged harmonic) | — | 0 | 0 | n/a | No new rule |
| Q2 | 996 | 0 | 0.0151029 | 2.2088% | Yes |
| Q5 | 1160 | 0.000000358 | 2.57495 | 18.1897% | **No: fails p95 <= 0.1** |

120 cal parents requested, zero failures. Q5 exists in 118 parents. Q5 uses
periodic angular residuals, z0=2 thermal widths, matching value/slope at the
join and thermal curvature outside. Its failed proposal is retained only as
a diagnostic helper and unit test; no active score flag was added. It was not
tuned after failure. Q2's old 6.3% beyond-z0 figure used raw fitted widths;
this check fixes the denominator to the **effective floored widths actually
used by the scorer**, explaining the 2.21% share. Its costs are unchanged.

## Gates

- Default fingerprint: `6cc586e17ab1`.
- Existing default native/replay CPU parity: 2 passed.
- Phase CPU parity: 3 units each for adam/irprop+/irprop-, 9 passed;
  two phase-state tests passed (11 total). The quadratic curvature-width unit
  test and actual curvature-source native/replay CPU solve also pass. A native/
  replay Adam spot-check passed after removing redundant legacy Adam updates.
- Score tests: 13 passed (Q2, donor dict, Q5 diagnostic rule).
- CUDA phase parity: 15/15 (three units each for Adam, iRprop+, iRprop-,
  R2s, IA), bit-identical output scores/coordinates, no replay fallback,
  238 seconds. Initial unwrapped eigensolver run captured but failed equality;
  rerunning with the existing deterministic safe_cuda_eigh_chunking wrapper
  passed. Additional checks cover the curvature source and production-derived
  iRprop search initialization after the shared update was simplified: **6/6
  passed** (three IA, three curvature-source units), exact CUDA eager/replay
  scores and coordinates, no fallback, 160.2 seconds.
- Golden: production 245 arrays, zero differences (287.8 seconds). The initial
  archived-path failure was resolved using the original archived DB, not another DB.
- USAPOU flag-off: CPU top score -2.0576415, replay -2.2268531 (absolute
  difference 0.1692116); replay mode succeeds. Both select key [null,2,154].
  Thus the flag-off control shows numerical divergence, but **does not reproduce
  the flag-on top-key mismatch**. A USAPOU same-device graph/eager and native/
  rewritten CPU check with the donor flag on was added. It passes exactly:
  USAPOU donor flag fingerprint 3f75c9525a4e, seed 11; graph/eager fast FK,
  graph/eager reference FK, and native/rewritten CPU loops all have zero score
  difference and matching top rows. Therefore no donor-flag capture or rewritten-
  loop defect is found; CPU/GPU trajectory divergence remains a platform
  reproducibility limitation. The flag-off top-key mismatch was **not**
  reproduced and is not claimed as such. No graph repair was warranted by
  the exact same-device comparisons (123.5 seconds).
  Full patch replay: **3/3 passed**, fingerprint `76da02b43182`, maximum
  absolute/relative score difference zero; fast FK graph/eager, reference FK
  graph/eager and native/rewritten CPU loops all agree on DAMVAQ, HOXJUZ,
  JEGPUJ, seed 11 (200 CPU-loop steps).
- Colab spend: zero; no VM started.

## Reproduction / artifact locations

Run from jgen-b with PATH pointing at the main checkout's pixi linux bin and
PYTHONPATH=src:.:research/bench. Correctness tests: tests/test_finish_phase.py,
tests/test_gpu_fast_path.py (existing replay-loop cases), test_angle_tail.py,
test_q6_donor_pairs.py and test_torsion_tail.py. CUDA phase checks use
CUBLAS_WORKSPACE_CONFIG=:4096:8, torch deterministic algorithms, runkit
`gpu_lock('CodexB')`, and the existing safe_cuda_eigh_chunking context.

Artifacts under this worktree's research/data/jgen/campaign/B/: h1_q2_crystals.json,
h1_q5_crystals.json, h1_usapou_control.json, h1_usapou_donor_graph.json,
h1_patch_replay.json; golden JSON/log
under g/g3c_implementation_golden.*. The exact proposed score flags are tracked
in H1_PATCH.json. Neither score flag is promoted to production by this package.
No golden directory was written; no val/reserve panel or QM property was read.

## Checkpoint disposition

G-3c step 1 and H-1 implementation/correctness work is complete. Stop for the
auditor. Step 0, optimizer evaluation, H-3 fold-in and G-3d remain outside this
checkpoint. Production behavior/fingerprint are unchanged. Q5 is explicitly a
failed crystal gate, not a candidate included in H1_PATCH.json. No score patch
or optimizer has been adopted as production.
