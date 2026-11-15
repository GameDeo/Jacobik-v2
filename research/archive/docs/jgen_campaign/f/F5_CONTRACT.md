# F-5 contract — corrected Li donor cap

Frozen before opening any corrected-arm embed or solve results (2026-10-29).

## Arm and measurements

- Correct `embed_li_cap` so Li⁺ is placed only on a κ donor whose local group is net anionic: the donor atom's formal charge plus the formal charges of its directly bonded atoms is below zero.
- Re-run the corrected arm's frozen-coordinate embed label screen on the same 684 F-4 attempts. Report donor classes whose status changes, the corrected ligand/parent footprint, and how many of the prior 111 parents entered only through CO-like donors.
- Run unrelaxed paired solves on the corrected footprint: `cal` plus `tail`, seeds 11, 12, and 13, on one Colab L4 with a same-L4 control. Pool seeds by parent for the paired estimates; use 10,000 parent-cluster bootstrap replicates and report the three metrics and 95% CIs used in F-4. No relaxation in this screen.
- Attribute every parent worse by more than 25 kJ/mol on at least two of three seeds from the ligand assignment used by the top-1 conformer, not from candidate donor lists.

## Fold-in decision rule

A fold-in proposal passes only when all conditions hold:

1. The paired top-1 mean ΔE_unrel 95% CI upper bound is below zero, pooling all three seeds by parent.
2. Floor and obscene-rate paired 95% CIs exclude zero in the improving direction.
3. Every remaining >25 kJ/mol guard breach has a stated cause based on the assignment actually used, and no remaining breach is caused by a donor-rule defect. A breach is not a global veto by itself; this is the auditor's ruling on the F-4 contract.
4. The win reproduces on the paired L4 panel and survives a local RTX 2060 spot check on a few named parents in the same direction. For this CPU-side embed flag, compare corrected cap against control using the same named parents, seeds, and local machine; record the local GPU identity and the paired per-parent result.

Interpretation frozen before the solve screen: “same direction” means the mean paired top-1 ΔE_unrel across the named parents (each parent averaged over seeds) has the same sign on the 2060 as on that same parent subset on L4. The small 2060 sample is a descriptive sign check, not a second bootstrap gate.

If all gates pass, write a fold-in proposal for owner review; do not change the production default. If a gate fails or the spec does not permit an unambiguous decision, report that and keep the flag off.

## Scope and operating constraints

- Use only `cal` and `tail`; do not read or evaluate `val`, reserve, or locked sets.
- Stay on branch `jgen/f-embed` from master `1f19ef7`; keep production defaults and fingerprint unchanged.
- Local CPU concurrency at most three; one GPU job at a time. Check `which xtb` and `nvidia-smi` before relevant runs. Colab L4 use is limited to one `jgF*` VM; fetch and count outputs before `cj.py down`.
- Report spec issues and failures plainly. Do not relax structures or tune after viewing outcomes.

Assisted-by: Codex on jgen/f-embed
