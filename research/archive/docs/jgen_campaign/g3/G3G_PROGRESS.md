# G-3g IA-k fold-in progress

## 2026-11-02 — owner-approved local fold-in

Read accepted G-3f audit and G-3g directive. Owner authorizes IA-k despite
the registered regression guard failure. No score/optimizer arithmetic changes.
Shared old golden directory copied first to golden_273f2ccbf892 under the
November archive, refusing an existing target, all three file hashes verified.
Archive/input hashes frozen in g/G3G_ARCHIVE.json before regeneration.

Profile default now exactly the tested Amendment5 IA-k dictionary. Production
fingerprint5f74b14b57ed equals both the tested arm identity and direct config
hash; explicit finish_phase=null restores C1 and historical273f2ccbf892 hash.
Bench control-reuse eligibility now accounts for inherited non-null default;
short fixed-switch budgets run search only without an invalid-cap exception,
with no arithmetic change to the tested250-step configuration. These are
integration fixes required by promoting the existing flag, not new tuning.

README/flag history/Colab pin updated for new production. Shared main AGENTS
fingerprint/history and PLAN status updated as explicitly directed, stating
owner merge pending; patch copy committed for review because both shared
files are outside worktree tracking. FELSAU ranking and CAXTAX watch noted.
No DB/meta writes. Local only; Colab spend0. Next: canonical replay,
golden regeneration/check, full tests once, default AWONOR CPU identity smoke.

## 2026-11-02 — full suite and CPU identity

Full suite run once under deterministic Torch/cuBLAS settings: 439 passed,
3 xfailed, in1353.87s. Includes reference/replay parity and canonical generator
checks; no unanticipated failures or skipped solver checks. Legacy Adam
schedule/polish tests explicitly select C1 so they test their intended code,
while profile/default/short-budget tests exercise new IA-k behavior.

AWONOR seed11 ordinary CPU benchmark initially used its usual row culling;
it returned top-1 score0.8069269061 rather than the no-cull G-3f CPU identity
score0.8301905990. That result is retained, not compared across layouts.
Added an explicit bench --cpu-identity option (CPU only, one Torch/solver
worker, no culling, recorded in metadata). It changes no default behavior or
profile fingerprint. With that matching layout, default profile (flags={}),
250-step AWONOR smoke through python -m tools.jgen bench run exactly matches
G-3f identity/AWONOR_11_after.npz: score0.8301905989646912. Both original and
matched-layout records retained. No comparison to GPU scores as CPU identity.

Golden record completed245 arrays for each profile. Production's read-only
check passed245/245; u1 check still running. Per-case MACE summary now runs
on CPU, descriptive only. No Colab used or started for G-3g.

## 2026-11-02 — G-3g checkpoint complete

New production5f74b14b57ed; explicit C1 null273f2ccbf892; direct hash and
fresh-process profile snapshots verified. Exact tested phase dictionary.
Canonical replay regenerated unchanged;18 CUDA phase parity cases pass
inside full439-pass/3-xfail suite. Focused integration checks14/14 pass.

Golden recording/checking finished for production andu1:245/245 each.
Per profile7/40 golden cases change,14/245 arrays: all seven solved seed11
cases change, all33 fixed-pool rescore cases/231 arrays remain identical.
CPU MACE top-1 new-minus-old changes (kJ/mol): OJIMIF+2.055, HUYNEU-18.054,
CBTYCO+51.135, KIQYEQ-264.950, SOKVET-19.384, YAVFEG+0.953,
aqua_La3_8H2O+23.764. Both profiles agree. Regression record, not a gate;
regressions retained and not tuned away. Default golden200-step CPU layout
preserved. G3G_GOLDEN_RESULT.json has per-case old/new scores and energies.

Old archive remains hash-identical at research/archive/data/
2026-11_golden_history/golden_273f2ccbf892/. Legacy reference unchanged.
Production DB and.meta hashes equal pre-task values; no writes to either.
AWONOR default-profile CPU identity smoke250 steps matches saved G-3f CPU
score0.8301905989646912 exactly, using explicit single-worker/no-cull option.
Ordinary culled-layout smoke mismatch retained and disclosed separately.

G3G_REPORT.md and G3G_RESULT.json record authority, exact fingerprints,
archive/artifact hashes, checks and expected failure reasons. Shared PLAN
status now records checkpoint complete/owner merge pending; shared AGENTS
history and all tracked docs consistent. No merge/push, local only, Colab0.
All owned jobs finished. Commit and STOP at G-3g checkpoint for audit.
