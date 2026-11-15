# D3-H Q7 ruling — 2026-11-06 checkpoint

Implemented `.claude/audit_jgen_checkpoints_2026-11-06.md` §4 after acceptance of 0c78211.
All changes are bench harness, tests or documentation on `jgen/d3-harness`. Contract remains **DRAFT, not frozen**.

Curvature includes Q1–Q6 plus the Cartesian `Scorer` Q7 term on the existing cached directions. The FK bend
prior is declared out of scope. No bend directions or reconstructed bend coordinates were added. Affirmative
coverage now records `q7_scorer_in_scope=true` and effective FK prior settings: solve defaults for penalty,
scale, sigma and weight, the arm's free zone, and production bend/haptic profile controls. Settings must match
across arms; local launch refuses mismatches before accessing VM tooling. Each isolated worker checks runtime
settings again. Curvature summaries must carry the same arm identity; analysis refuses differing arm/control
FK settings. Existing summaries without the new coverage/identity fields must be regenerated before real launch.

With `q7_bound_centre` enabled, `<db-stem>.bends.json` is mandatory. Its SHA256 is recorded as `bends_sha256`
in arm identities (and therefore plans, manifests and records), and in newly computed curvature summaries.
The launcher ships it beside the DB and meta file, checks its exact remote SHA256 before starting workers,
and the arm worker verifies it again before loading DB/model. Upload deduplication uses exact destination/hash
pairs, rejects sidecar basename collisions, and cannot skip bends because a DB/meta pair is already present.
Missing, changed or unverified required sidecars refuse launch. Local identities/files are rechecked before provisioning.

Validation: `pytest -q tests/test_d3_harness.py tests/test_c2b_confirmation_report.py` — **67 passed**.
Tests cover Scorer Q7 coverage with a zero control term, scope contribution, all five named FK settings,
registered free-zone mismatch, runtime identity mismatch, missing/changed bends, pre-VM refusal, upload/hash
verification, deduplication and remote sidecar basename collisions. Existing persistent-worker, P fallback,
curvature thresholds and reporting checks remain passing. `compileall` and `git diff --check` passed.
Real configured production identities validated locally in DRYRUN mode; fingerprint remains `5f74b14b57ed`.

No VM or GPU was used, no panel or curvature rerun occurred, and incremental Colab spend is **0 units**.
The accepted control baseline remains median **1.3315397846869683**, rank **0.8639999488508491** (Scorer Q7 zero).
The accepted estimate remains **8.479172 units**; L4 speed factor remains unmeasured and four-arm GPU residency
remains untested. D2 identities and contract freeze remain pending. No D3 arm is authorized here. Stop for audit.
