# S4 solver-miss analysis

## Scope and evidence

This is a read-only synthesis of the recorded development-set diagnostics. It does not change the solver, score, NCI calibration, or baseline, and it uses no held-out cases. The main evidence is the 2026-10-16 analysis in [CHANGELOG.md](../CHANGELOG.md#L1373) and the error decomposition in [ARCHITECTURE.md](../ARCHITECTURE.md#L576).

The recorded track comparison covers 76 valid development complexes and three seeds. Its base predates the adopted `u1` + calibrated NCI setup, so the classifications below are historical evidence about failure mechanisms, not a measured estimate of current `u1` performance. The referenced raw files (`data/energy_scale_refit/tracks_report.txt`, `internal_stereo.txt`, and pool arrays) are not present in this checkout; numeric results below are transcribed from the checked-in summary, not independently recomputed.

## Classification

| Failure signature | Existing evidence | Classification and implication |
|---|---|---|
| Wrong stereocentre, E/Z, or ring configuration in the top-ranked structure | 16/80 open-pool #1 picks have a configuration absent from the reference: 6 stereocentre cases, 6 E/Z cases, and 4 ring-pucker cases. Their median error is 99 kJ/mol versus 69 kJ/mol without a mismatch. SOKVET has two wrong E/Z bonds; supplying crystal internals moves it from 525 to 70 kJ/mol. | Strong S4 candidate. These degrees of freedom are fixed by the free-ligand embed, so a later solver stage cannot repair them. The evidence supports enumerating unspecified stereo and comparing a bound-state embed; it does not establish which intervention is best. |
| Wrong torsion mode despite a supported crystal mode | TAPWUG remains at 643 kJ/mol even with crystal internals. Its two pyridyl groups are flipped by 180 degrees; the crystal-aligned Q5 mode has nearly all the weight, but the beam selects the near-zero-weight narrow mode. Across the dev set, 34 multimodal joints in 16 complexes have every beam row on a mode with under 20% of the dominant mode's mass. | S3 beam/search failure, not S4 frozen internals. The beam's mode-density preference and lack of metal context explain why a correct torsion seed can be discarded before the metal-aware solve. |
| Wrong macrocycle fold persists with crystal internals | FIGHOT (Zr with two 11-membered N,O,N macrocycles) has ring-torsion differences up to 119 degrees, including with crystal internals; metal–donor distances remain stretched. The summary attributes the discrepancy to macrocycle-pool folds and macrocycle joints. | Primarily S3 fold coverage/search. It is distinct from an embed-fixed small-ring pucker; macrocycle fold enumeration is a candidate, but the available result does not isolate enumeration from joint search. |
| Large residual after giving the solver crystal internals and crystal-positioned statistics | On the 26 complexes whose open #1 is at least 100 kJ/mol, the median error shares are 3% coordination-isomer choice, 61% internals, 10% data positioning (only measured with crystal internals), and 36% residual after those oracle changes. The residual median is 58 kJ/mol; examples include TAPWUG 479, PEWBOL 438, YAVFEG 179, and FIGHOT 148. Crystal-positioning the statistics had no detectable set-wide effect. | Not explained by S4 alone. It may combine reach/search, the ranking objective, and energy-model disagreement. The aggregate oracle comparison does not justify calling it a scoring defect without case-level score-versus-energy evidence. |

## What the measurements support

- Ligand internals are a substantial development-set error source: giving the pool crystal internals lowers the #1 median by 24.8 kJ/mol and the mean by 46.8 kJ/mol (paired interval [-72.5, -18.2], 53 better and 9 worse). This is an oracle ceiling, not an expected gain from any deployable S4 method.
- Coordination-isomer choice is not the main median lever: forcing the reference isomer changes the #1 median by +0.1 kJ/mol, while worsening some named cases. That separates internal ligand configuration from donor-to-site assignment.
- Beam and macrocycle misses remain visible even under oracle internals. Keep TAPWUG and FIGHOT in the S3 bucket when prioritizing S4 work; otherwise improvements can be credited to internals while the actual failure is still unreachable or pruned search space.
- The 36% residual is unresolved by these aggregate comparisons. Although crystal-positioned statistics showed no global gain, that alone does not distinguish ranking error from a pose the solver cannot reach or energy-model error.

## Read-only follow-up when the pool artifacts are available

Use the existing development outputs only. For each high-error case, join its top-1 and floor across seeds with the crystal-internals track, the reference-isomer track, `internal_stereo` configuration labels, distinct torsion-mode/beam signatures, and macrocycle torsion differences. Then assign a primary label using the signatures above and retain a secondary label when evidence overlaps. Mark a scoring issue only when a low-energy candidate already exists in the stored pool but ranking selects a materially worse structure; a missing low-energy candidate is search/generation evidence instead. Keep the old and adopted `u1` pool versions separate, and do not infer a post-NCI effect from the pre-NCI study.

The existing analysis entry points are `tools/jdev/budget/tracks.py`, `tools/jdev/budget/internal_stereo.py`, and `tools/jdev/budget/ensemble_stats.py`; their current loaders expect local data under `data/`, which is not available here. No commands that solve, rescore, fit, or evaluate held-out sets are needed for this classification.
