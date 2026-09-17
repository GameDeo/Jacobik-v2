# Root-caused and fixed the ligand-orientation regression

Found it, and it was small — a scoping bug in `q3_mode_commit` (2.5.0). Confirmed via a live
A/B test (the codebase's own changelog admits this feature was "not yet empirically
validated against a live solve" when it was written — this is the first time it's actually
been run).

## What was wrong

`symmetric_groups` groups donor atoms purely by (SMILES, RDKit canonical rank) — so 6
separate water ligands on `[Fe(H2O)6]3+` land in one group, exactly like EDTA's 4
carboxylate oxygens (which really are one group: four arms of the same molecule, physically
coupled through a shared backbone). `q3_mode_commit` then hard-locks the *whole group* to
whichever single Q3 (donor approach-angle) statistical mode best matches one arbitrary
initial pose — a good idea for EDTA's arms, which have a genuine "which of several
real binding modes did this molecule commit to" question. It's a bad idea for 6 chemically
independent water molecules, which have no reason to share a rotational conformation at all.

Live-traced it on `[Fe(H2O)6]3+`: the mode-commit picked the same single mode for all 6
waters, every trial, every seed — but it was a **~4%-weight minority mode** in the real
fitted data (there are 6 real modes; the dominant one carries 35%). Because the objective
then scores against *only* that committed mode (the others are switched off), nailing even
a rare mode scores better than the correct, symmetric answer scores against the full honest
mixture — so the optimizer confidently produces the wrong geometry. That's exactly the
68.5°/172.4° M–O–H split from the earlier report: chemically nonsensical, but numerically
rewarded.

This also lines up with your account: `q3_mode_commit` was added in 2.5.0, layered directly
on top of the Q2/Q3 refit that made the CO/TBP-straightness fix work (2.4.0), specifically
to patch EDTA — and was never tested against a simple repeated-monodentate case like water
or CN⁻ before now.

## The fix

Added `q3_symmetric_groups` — the same grouping, but restricted to donors that share a
single **ligand instance** — and used it only for the Q3 tie/mode-commit. The bond-length
tie (`sym_weight`) still uses the original, unrestricted grouping (tying together 6
chemically-identical Fe–O bond lengths is correct chemistry; nothing about that needed to
change). EDTA's carboxylates are already one ligand instance, so this restriction changes
nothing for EDTA — confirmed byte-identical.

Changed: `src/jacobik/engine.py` (+42 lines, builds `q3_symmetric_groups`), `src/jacobik/ik/ik_solver.py`
(new optional parameter, two loop sites redirected to it). Both files get a versioned
changelog entry in the same style as the rest of the file's history. Patch script saved at
`script_outputs/pi_lanthanide_nitrato_task/after_q3_fix/patch_q3_crossligand.py`.

## Verification

| Case | Before | After |
|---|---|---|
| Fe(H2O)6³⁺, all 6 waters | M–O–H: 68.5°/172.4° (every water) | **126.4°/126.4°** (every water) |
| La(H2O)9³⁺ (CN9), 9 waters | all 9 badly asymmetric | 7/9 perfect (126.4°/126.4°), 2/9 mildly off |
| Fe(CN)6³⁻, M–C–N straightness | 166.9° | **172.9°** |
| EDTA (Co, hexadentate) | score −4.576, 16/32 converged, O-spread 0.203 Å | **identical** — −4.576, 16/32, 0.203 Å |
| Full benchmark test suite | 6/6 pass | 6/6 pass |

Across the full 12-complex PI task batch, `water_suspicious` count (my orientation-flag
metric from the earlier report) dropped from 100% (every water in every complex) to
effectively 0% (0/8, 0/9, 0/8, 1/9, 1/7, ... — isolated outliers instead of universal
failure). Updated results saved to
`script_outputs/pi_lanthanide_nitrato_task/after_q3_fix/full_analysis.json`.

## One thing this uncovered, not fixed — flagging honestly

Fixing the water orientation removed an unintended side effect: in `[Fe(H2O)6]3+`
specifically, the *old, buggy* hard-committed Q3 term occasionally (1/16 trials, by chance)
provided enough of a different gradient push to also land in the correct, compact Fe–O bond
length (~2.06 Å, RMSD 0.058 Å vs. the DFT reference — the excellent result in the first
report). With the bug fixed, none of 16 trials (across 4 seeds tested) find that basin
anymore — every trial now lands in a separate, pre-existing local minimum with correct
octahedral shape (90°/180° angles, perfect) but an inflated bond length (2.88 Å vs. ~2.02 Å
real). I confirmed this basin already existed *before* my fix too (it was the 15/16 majority
in the original run) — my change didn't create it, it just removed the accidental,
never-intentional escape route the old bug sometimes provided. This looks like a separate,
pre-existing weakness in how the Q1 (bond-length) term competes with the other terms during
optimization, unrelated to the Q3/orientation mechanism. I haven't chased it — didn't want
to scope-creep past what you asked for — but wanted to flag it rather than let a quietly
worse bond-length number slip through. Happy to dig into it next if useful.
