# Checkpoint H-3c — stopped at Step 0

2026-10-31. Source: `bd5494a`, `jgen/h-patch`. Local CPU only; no GPU
job, Colab use, holdout draw, tuning, or solver changes.

**Step 0 FAIL.** The directive predicts that every collapsed NUWFAM row
(H–C–H <15°) has at least one H with absolute face lean >57°.
None of the 26 collapsed rows inspected satisfies that prediction.
The directive explicitly says: “If this fails, stop and report. Do not fall
back to face-mean or a pair wall.” Work stops under that rule.

## Measurement and evidence

Each H is measured separately by the existing
`jacobik.stats.orientation.haptic_features`: `asin(n · unit(X − A))`, in
degrees. For η², the normal is the centroid-to-metal direction with its
face-bond component removed, normalized and oriented towards the metal.
The wall width comes from `angle_bends.lean_free_deg()[2]`: 57°.

A fresh CPU scorer preparation independently verified generated-order
indices: metal 0; face [1,2]; substituent pivot 1; H atoms 40 and 41.
The per-H scorer descriptors have one substituent each, η²=1 and the same
face. The diagnostic calls the fitting-coordinate implementation directly,
without rebuilding saved conformers or changing their coordinates.
Crystal mapping uses `argsort(gen_to_ref_idx)` and asserts element order.
`xtb` resolved in the prescribed environment; CPU preparation completed.

| Saved arm | Pool rows | Collapsed <15° | Collapsed with any H beyond ±57° | Largest absolute H lean among collapsed | Minimum H–C–H |
|---|---:|---:|---:|---:|---:|
| H-3x cap | 96 | 17 | **0/17** | 35.589° | 3.283° |
| H-3x control | 96 | 0 | — | — | 58.941° |
| H-3b cap_normal | 96 | 4 | **0/4** | 30.916° | 6.535° |
| H-3b cap_allyl | 96 | 5 | **0/5** | 30.724° | 11.874° |

Example counterexample: `NUWFAM_cap_s11_row00002.xyz` in the H-3x cap
pool has H–C–H **3.282943°**, with individual face leans
**−20.690111° and −23.954949°**. Both are well inside ±57°.

Control top-1/floor rows and crystals satisfy the other prediction: all nine
checked entries are within the wall. The cap and both early experimental
arms also have all nine top-1/floor/crystal entries within it. The crystal
entries are the reference repeated across seeds, not independent crystals.
Pools share a parent and are not independent statistical evidence; this is
a deterministic counterexample check, not a significance test.

## Interpretation and scope

The proposed absolute-face-lean violation does not explain these saved
collapses. Two nearly parallel H vectors can both have modest projection
onto the face normal. Also, `asin(n · u)` does not encode which side of the
in-plane angular branch a substituent occupies. These are geometric
limitations of that scalar measurement, not proof that a constant offset
wall can never help. No offset-wall optimization was performed, so its
behavior remains untested. A revised directive would need to distinguish
the signed accumulated rotation from this principal-value face lean and
state a prediction consistent with the counterexamples.

## Step status

| Step | Status |
|---|---|
| 0: root-cause verification | **FAIL; mandatory stop triggered** |
| 1: remove rejected H-3b mechanisms | Not executed: Step 0 precedes coding |
| 2: offset wall and crystal parity | Not executed |
| 3: 34-parent development check | Not executed |
| 4: pre-registered holdout | Not drawn, opened, or solved |
| 5: production/final replay comparison | Not executed |

The existing rejected H-3b implementations remain on the development
branch exactly as at `bd5494a`; they are not endorsed or promoted by this
checkpoint. No production defaults, donor flags, or generated replay code
were changed. No merge or push.

## Reproduction and artifacts

Run from this worktree with the prescribed pixi PATH and PYTHONPATH:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m tools.jgen.diag.h3c_root_cause
```

- [Read-only diagnostic](../tools/jgen/diag/h3c_root_cause.py)
- [Tracked summary](../../../../bench/jgen_campaign/archive/H3C_STEP0_SUMMARY.json)
- [All 26 collapsed-row measurements](../../../../bench/jgen_campaign/archive/H3C_COLLAPSED_ROWS.csv)
- Full 420-entry geometry table: `research/data/jgen/campaign/H/h3c/step0_rows.csv`.
- Raw JSON: `research/data/jgen/campaign/H/h3c/step0_summary.json`.

Verification: diagnostic completed successfully with all four pools present;
384 pool rows plus 24 selected conformers and 12 crystal entries measured.
Python compilation and `git diff --check` passed. A first diagnostic attempt
named `h3b/dev/cap/xyz/NUWFAM`, which has no saved NUWFAM pool; it failed
before writing results. That nonexistent source was removed, and the
complete four-pool rerun above produced the reported artifacts.
