# G-3g IA-k production fold-in checkpoint — 2026-11-02

IA-k is the production default on `jgen/b-solver`, fingerprint **5f74b14b57ed**,
exactly matching the tested G-3e/G-3f arm. Owner merge remains pending.
The preceding C1 remains reachable with `JACOBIK_PROFILE_OVERRIDES='{"finish_phase": null}'`
and retains fingerprint `273f2ccbf892`.

## Authority and scope

The owner approved this fold-in in directive_jgen_G3g_foldin_2026-11-02.md
after audit_jgen_G3f_2026-11-02.md. The registered G-3f regression guard
remains **FAILED**: 3 parents versus limit2. L4 cal top-1 -10.9 [-18.7,-4.6]
kJ/mol, floor -9.5, end-to-end wall1.22x. The audit records a degenerate
same-trajectory C1 null, seed-rotation null0–2, and16 mirror improvements.
Owner accepted the failed guard. FELSAU is a ranking case; CAXTAX remains
a post-fold-in A4b watch case. This checkpoint does not re-judge those gates.

The exact Amendment5 finish dictionary is promoted: iRprop+ search and
finish, switch100, symmetry-only release, retained shape/projection, thermal
widths. No score arithmetic, gradient fusion, width/budget tuning or DB/meta
change. Existing caller step budgets are preserved; the golden solve cases
use their historical default200 steps, while the identity smoke uses the
tested250-step budget. No merge or push was performed.

Integration changes: a fixed switch beyond a short requested budget leaves
the finish inactive and runs the existing search updates, avoiding a
validation exception in two-step scorer/fixture setup. Benchmark reuse
eligibility honors the inherited IA-k default, so explicit C1 cannot reuse
an IA-k solve. Legacy schedule/polish tests explicitly select C1; production
default, explicit-C1, short-budget and reuse tests cover the new integration.

## Golden archive and regeneration

Old references copied **before regeneration** to `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/archive/data/2026-11_golden_history/golden_273f2ccbf892`.
Existing-target refusal enforced; all three archived file hashes matched
the original directory. The legacy NPZ is also preserved unchanged. Only
the authorized production/u1 references were regenerated in the shared
`research/data/golden/` directory. No DB or sidecar changed.

| Profile | Changed cases / total | Changed arrays / total | New golden check |
|---|---:|---:|---|
| production | 7/40 | 14/245 | PASS 245/245 |
| u1 | 7/40 | 14/245 | PASS 245/245 |

A case is a fixed-pool rescore parent/seed or a solved parent/seed. All33
fixed-pool rescore cases (11 parents ×3 seeds,231 component arrays) remain
identical. All seven seed11 solve cases change coordinates and scores.
The exact and above-check-tolerance change counts coincide. Production
and u1 arrays and energy summaries agree.

### CPU golden top-1 energy regression record

Unrelaxed MACE polar-1-s float64, CPU. ΔE is relative to each case’s crystal
reference; change is new minus archived C1, kJ/mol. Builder atom order is
used for stored solver coordinates and crystal order for the reference.
These fixed CPU fixtures, worker layout and budgets are a regression record,
not another acceptance gate or a replacement for the paired L4 results.

| Case (seed11) | Old top-1 ΔE | New top-1 ΔE | Change |
|---|---:|---:|---:|
| OJIMIF | 8.916 | 10.972 | +2.055 |
| HUYNEU | 68.010 | 49.956 | -18.054 |
| CBTYCO | -5.262 | 45.873 | +51.135 |
| KIQYEQ | 328.046 | 63.096 | -264.950 |
| SOKVET | 118.450 | 99.066 | -19.384 |
| YAVFEG | 282.158 | 283.111 | +0.953 |
| aqua_La3_8H2O | -27.991 | -4.226 | +23.764 |

Seven-case mean change -32.069, median +0.953 kJ/mol; three improve and four worsen.
CBTYCO’s +51.135 and aqua_La3_8H2O’s +23.764 regressions are retained.
No tuning or rerun follows these descriptive outcomes.

## Checks

| Check | Result |
|---|---|
| New production identity | PASS: 5f74b14b57ed, equal to tested IA-k and direct config hash |
| Explicit old C1 | PASS: finish_phase=null, 273f2ccbf892 |
| Canonical replay regeneration | PASS: generated file unchanged |
| Full suite, deterministic Torch/cuBLAS | PASS:439 passed,3 xfailed;1353.87s |
| CUDA finish-phase replay parity | PASS:18/18, included in full suite |
| Golden checks | PASS:245/245 each, production and u1 |
| AWONOR seed11 CPU identity | PASS: top-1 score 0.8301905989646912 exactly |

The three xfails are existing expected failures, not skipped solver coverage;
their names/reasons are retained in G3G_RESULT.json. Full suite was run once.
The later optional CPU identity CLI path was verified by its real smoke
and focused parser/profile/reuse checks recorded in integration_tests.log.

The initial ordinary CPU benchmark used its usual culling and returned
score0.8069269061. That layout differs from the saved G-3f CPU identity run;
the original record is retained. `--cpu-identity` explicitly selects one
Torch/solver worker and no row culling, recorded in output metadata.
With flags={} and250 steps it reproduces the saved G-3f CPU score
0.8301905989646912 exactly. The baseline is the CPU identity NPZ, not an
L4/2060 score comparison. Default benchmark behavior is unchanged by this
optional layout selector.

## Documentation and checkpoint

Updated profile header/history, README, archived flag history and Colab
fingerprint pin/README. Explicitly requested shared AGENTS fingerprint/history
and PLAN status were updated in the main checkout, stating owner merge
pending; their reviewable patch is G3G_SHARED_DOCS.patch because the shared
files are not tracked in this worktree. The preceding master remains347496f
until the owner merges. No other worktree or branch was changed.

G3G_ARCHIVE.json preserves original reference/input hashes; G3G_GOLDEN_RESULT.json
contains per-case scores/energies and array changes; G3G_RESULT.json contains
check results, profile snapshot and artifact hashes. Raw logs, regenerated
references and smoke exports are retained under campaign/B/g3g and the shared
golden directory. All owned jobs have finished; local CPU/2060 only, Colab
spend0. Checkpoint: commit and stop for audit; owner merges to master.
