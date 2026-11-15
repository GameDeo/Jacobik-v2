# Checkpoint H-3d — Step 0(a) fails

2026-10-31. Branch `jgen/h-patch`; starting source `9dde7b3`.
Read the directive, H-3b/H-3c audit sections and H3C_REPORT before work.

**Stopped under the explicit Step 0 rule:** “If (a) fails, stop and report.”
Prediction (a), that every collapsed NUWFAM row comes from a mismatched
start, fails under the prescribed substituent-lean sign measurement.

## Where the collapsed rows start

Reconstructed all 96 exported H-3x cap rows across seeds 11–13 by
E/Z variant, assignment and trial. Of the 17 rows with final H–C1–H <15°:

- **6 start with both H face leans negative: both lean away from the metal.**
- 11 start with mixed signs: one H away and one towards the metal.
- 0 start with both H face leans positive.
- Only 2 have positive mean starting bend (mean of the two sine projections).

The six unambiguously away-facing counterexamples are:

| Seed | Variant | Assignment | Trial | Pool row | H40 start lean | H41 start lean | Final H–C1–H |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0 | 0 | 106 | 19 | −46.044° | −7.852° | 6.731° |
| 11 | 0 | 0 | 24 | 30 | −44.032° | −9.983° | 4.033° |
| 12 | 0 | 0 | 92 | 19 | −10.060° | −43.478° | 9.870° |
| 12 | 0 | 0 | 35 | 22 | −46.493° | −7.373° | 9.983° |
| 12 | 0 | 0 | 19 | 31 | −47.701° | −6.079° | 3.547° |
| 13 | 0 | 0 | 14 | 29 | −9.162° | −44.302° | 3.744° |

Thus even the permissive interpretation “mismatch if **any** H initially
leans towards the metal” misses six collapsed rows. Reporting mixed signs
separately avoids assigning a single face from an ambiguous pair. This
does not establish whether a Li-aware placement would help some starts;
it contradicts the universal prediction required to proceed.

## Reconstruction and limits

Used the same preparation, seed-row generation, class-budget filtering and
ComplexFK application as `h3b_start_mode_check.py`. The child process turns
off the rejected face-mean and geometric-wall switches in memory; no source
is edited and no optimization is run. The scorer descriptors independently
confirm metal 0, η² face [1,2], pivot 1, per-H substituents 40 and 41.
Leans are measured by `orientation.haptic_features` separately for each H,
then converted with asin to degrees. Assignment indices are asserted
against each saved row key.

All 96 rows agree with the earlier H-3b start reconstruction to <0.001° per
H (exact maximum in the tracked summary). The saved pools have no starting
coordinate hashes, so this remains a reconstruction rather than an exact
historical-state certificate. The original cap Li coordinates were not
retained; this check uses the directive's substituent-sign proxy, and does
not directly measure the relative Li/metal face. Neither limitation is
silently converted into evidence for the hypothesis.

## Step status and compute

| Work | Result |
|---|---|
| Step 0(a), NUWFAM cap, seeds 11–13 | **FAIL: 6/17 collapsed rows start with both H away** |
| Step 0(b), majority/energy comparison | Not pursued after mandatory stop |
| Step 0(c), uncapped planarity | Not run after mandatory stop |
| Other seven cap parents | Not reconstructed after mandatory stop |
| Step 1, removal of rejected parts and Li-aware placement | Not started |
| Step 2, development solves | Not run |
| Step 3, holdout and replay comparison | Not run; holdout untouched |

Local RTX 2060 SUPER and CPU only. `nvidia-smi` confirmed no competing
compute job before starting. `xtb` resolved in the prescribed pixi
environment; embedding validity checks passed. One GPU preparation job,
three CPU threads; no IK or MACE evaluation. Colab spend 0. The process
completed successfully. No production, solver, replay, donor or default
changes; rejected H-3b mechanisms remain as at the starting commit pending
a revised directive. No merge or push.

## Artifacts and reproduction

```bash
# From jgen-h, with prescribed pixi PATH and PYTHONPATH
python -m tools.jgen.diag.h3d_start_face --arm cap
```

- [Diagnostic](../tools/jgen/diag/h3d_start_face.py)
- [All 96 matched rows](../../../../bench/jgen_campaign/archive/H3D_STEP0_ROWS.csv)
- [Summary](../../../../bench/jgen_campaign/archive/H3D_STEP0_SUMMARY.json)
- Raw: `research/data/jgen/campaign/H/h3d/step0_cap.json`.
- Process output: `research/data/jgen/campaign/H/h3d_step0_cap.out`.

Verification: successful diagnostic exit; 96 unique seed/variant/pool-row
keys, 17 collapsed rows and 6 both-away counterexamples; prior reconstruction
comparison; Python compilation; `git diff --check`.
