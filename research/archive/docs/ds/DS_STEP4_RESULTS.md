# DS Step 4 results: five DB builds (2026-11-11)

Job 61150093 (regular QOS, 1 CPU node, 22 min wall, exit 0), code snapshot `4c0857c` (`$SCRATCH/jacobik/ext/ds-4c0857c`),
output `$SCRATCH/jacobik/runs/ds-pipeline-20261111-1734`, local copy `research/data/ds_proto/pipeline_20261111/`.
Both extractions 3.5 min; fits 8-10.6 min each (peak RSS 3.4-3.9 GB). Reports: `db_compare_report.py`.
(DB meta shows `code_commit: unknown` because the snapshot has no .git; fixed in `01dd5b9`, the commit is in `job.txt`.)

## Gate: D0 reproduces the recorded D2
`--check-reproduction` against `constraints_2026-11-09.json`: 425,635 entries equal exact, 0 different, 0 missing, 0 extra, 0 model-label differences.

## Arms (entries; births/deaths vs the reference arm)
| arm | flags | entries |
|---|---|---:|
| D0 | rules off, legacy sampler | 425,635 |
| D-data | assignment, legacy sampler | 420,244 |
| D-sample | rules off, stable sampler | 425,633 |
| D-both | assignment, stable sampler | 420,241 |
| D-final | D-both + structure_sanity | 419,644 |

| comparison | births | deaths | Q1 support changed / consequential (of 107,563 shared) | reading |
|---|---:|---:|---|---|
| D0 -> D-sample | 0 | 2 | 0 / 13,348 | sampling alone: no population change; refits move because the cap-500 buckets draw different rows |
| D-sample -> D-both | 12,701 | 18,093 | 15,971 / 19,808 | extraction effect at a fixed sampler: the real data change |
| D0 -> D-both | 12,701 | 18,095 | 15,971 / 27,323 | both |
| D-both -> D-final | 0 | 597 | 740 / 486 | screen: 79 parents; 562 Q2 and 26 Q1 keys lose all support |

Q2 shows the same pattern (39,046 support changes, 58,231 consequential, D-sample -> D-both). Births and deaths are mostly sparse keys.

## Watch keys (D0 -> D-both)
- LUQVOK Pd/O_carboxylate Q1: n 142 -> 198; mixture means (2.056, 2.658) weights (0.845, 0.155) -> means (2.012, 2.075) weights (0.217, 0.783). The 2.66 A component (the erroneous primary-O observations) is gone.
- LIPDEU `QO:(2, 4)`: n 5,742 -> 6,135; means (-0.777, -0.708, -0.665) -> (-0.781, -0.708, -0.694); weights (0.659, 0.222, 0.119) -> (0.607, 0.226, 0.167). Still the broad fallback; not fixed by the data change alone.

## Screen
79 broken-structure parents of the 1,687 excluded by the filter stage (the rest are charge-implausible); fingerprint of D-final `d5e2f4113497`.

## New finding (F7): single-component Q5 fits flip on ~1% data changes
Of 1,818 shared single-component Q5 keys, 27 move their mean by more than 90 degrees between D-sample and D-both
(e.g. `('P_sp2','C_ar','S_sp3','C_ar','free_o0')` n 794 -> 806: -2.5 -> 179.8 deg). Bimodal torsion data fitted with k=1 pick whichever mode wins.
This is a pre-existing fit fragility exposed by small data changes, not an extraction defect. Not changed here; a Step 7 candidate.
