# DS step 2: comparison machinery — 2026-11-10

Branch `data/ds-cleanup`. No billed compute, no Colab. Production fingerprint `5b56969afb9c` unchanged
(all new behaviour is flag-gated or outside the generator).

| Item | Result | Evidence |
|---|---|---|
| Stress refit, supported Q1/Q2 keys | `fit.pipeline.stress.refit_closure`: independent fits where the chain needs them, adaptation down the parent chain; mirrors `PooledFitter.entries` for one key | synthetic test; real D2 Q1: three adapted sparse keys (chains of 12–15 keys) and LUQVOK reproduce the recorded entries, max mean diff 0.0 (`stress_repro.py`, 30 s expansion + 15–45 s per key) |
| Sparse ancestor dependencies | A restricted expansion is exact only under stable sampling (the legacy sampler shares one random stream across buckets); `refit_resolved` expands the whole table in legacy mode | docstring + code |
| QO fallback populations | `stress.refit_qo`: same group assembly and fit as the build; LIPDEU `QO:(2, 4)` (5,742 rows) reproduces the recorded D2 fit | `stress_repro_qo.py`, 25 s |
| Trial registry | `fit.pipeline.registry`: deterministic trial IDs, immutable input hashes, independent-parent denominator, broken-coordination rule; missing, failed, duplicated, unregistered or wrong-denominator trials block acceptance and are never censored as a pass | 2 tests |
| Null perturbation / replay | empty damage, zero amplitude and parent_removal on no parents return the clean table exactly; stable sampling is order-independent and local (existing); legacy Q6 sampling is byte-identical with the stable flag on or off | 2 new tests |
| Target-aware control identity | `tools.run panel --target-overlay F [--gate-eligibility G]`: hashes of the overlay, the source-review policy and the gate eligibility enter the control key; absent when not declared, so existing keys and caches stay valid | test in `test_control_cache.py` |
| Corrected-target overlays | The generic reconciliation reproduces 29/34 audited targets exactly. AGAPEG/ILEPOE follow the owner's kappa1 decision (the kappa2 overlay v1 entries are superseded; the fixtures assert the near O only). BADQIH, JAYCEV, NUHPEL stay visible as unresolved. No separate execution hook was added: the corrected arm runs the generic rule (`donor_identity_reconcile`) through its arm flags, which are already in the key | fixtures + `source_review.json` |
| Frozen gate eligibility | `gate_eligibility_v1.json` from the input-only audit, written before any new performance result: stage 1 valid 100 / repaired 9 / ambiguous 8 / unrepaired 3; stage 2 valid 70 / repaired 25 / ambiguous 15. Corrected primary = valid + repaired; the other strata stay visible; failed parents stay in every denominator. sha256 `a8d7a55b0d6f68077694c62c92547e3f0ea445a2793c593a6c6ddbeb60864a1e` | `freeze_gate_eligibility.py` |
| Failed/missing-parent accounting | Stage-2 `None` edge-on counts (absent record or missing field) fail the holdout gate and the failure gate, never pass | new test in `test_d3_harness.py` |
| 14 CUDA parity failures | Owner 2026-11-10: not platform-specific, ignored. Still reported as failed; assertions untouched | — |
| Intended-GPU preflight | A100-PCIE-40GB on login30 (login21's GPU was shared and busy): LUQVOK + HOXJUZ, control arm, 2 lanes, 61 s wall, no embed fallback, `embedding_validity.json` empty; LUQVOK top-1 dE 61.92 kJ/mol, identical to the D3 L4 control record | `$SCRATCH/jacobik/runs/ds-preflight` |

## Estimates needed before the next step (stages measured separately)

| Stage | Where | Estimate |
|---|---|---|
| Fresh extraction with IDs and ledger (two extractions: reconcile off / on) | billed CPU node (regular), 2 x 64 workers concurrent | measured `--quick` (2% of complexes, 20 workers, login node): 49 s wall, 3 min 49 s CPU. Full corpus is about 50x the CPU, so about 3 min CPU-time-limited per extraction on 64 workers plus merge: estimate 8-12 min wall for both, cap 25 min, at most 0.42 node-hours |
| Four DB fits from the extracted tables | CPU node, after review | D2: two builds including extraction took 11:44; fits alone are about 4–5 min each, so about 10 min for two concurrent pairs |
| Ledger/eligibility reporting | login node | seconds |
| Stress refits (Q1 closure pilot) | login node, 32 threads | 30 s expansion + 15–45 s per key and trial; 96 trials × scenarios per key is hours serial, so the stress runs must pack trials in parallel across workers |

## Not done in step 2 (stated)

- Stress orchestration is the refit machinery and the registry; the driver that runs a registered design through the
  refit, the fixed-pool scorer and the severity rule is Checkpoint B work (needs the four DBs).
- Source-coordinate perturbation needs re-extraction of the perturbed parent; only the observation-replacement and
  parent-removal paths exist end to end.
- Reference/evaluator compatibility: 204/230 audited targets are reference-compatible; the 26 incompatible ones
  need regenerated references before they can enter a corrected comparison (listed in `gate_eligibility_v1.json`).

## Quick staged extraction through the build CLI (login node, free)

`fit.build_db --stop-after merge --quick` with reconcile and stable IDs on: ran end to end in 49 s; wrote
`ledger_<split>.parquet` and the observation tables with IDs; `stop_after_merge.json` carries the eligibility report
(771 events on 400 parents in the 2% sample: 482 review, 259 correction, 30 preserved; 282 of the review events are
`legacy_supported_identity_collision`, the largest class to sample in the review; no unclassified kind; none of the
three unresolved parents fell in the sample). Output: `$SCRATCH/jacobik/runs/ds-quick-extract`.
