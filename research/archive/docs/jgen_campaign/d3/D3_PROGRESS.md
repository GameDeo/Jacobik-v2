# D3 harness progress

2026-11-04: Read D3-H and required contracts/audits; working only on jgen/d3-harness from 74b4d47.
No D3 arm or D3 result. D2 placeholders remain unresolved. Implemented draft harness, synthetic gates, C-2b
reproduction, CPU curvature wrapper, planning estimate, and isolated per-parent paired VM worker.
Curvature baseline running on CPU. One authorized DRYRUN L4 smoke pending, <=0.5 unit; no local GPU use.

2026-11-04 §4 checkpoint: full-precision C-2b S=P reproduction exact; 38 harness tests passed; focused C-2 reporting
compatibility checked. Guarded CPU curvature 80/80, 15,055 directions, median 1.3315397846869683, rank
0.8639999488508491; fingerprint unchanged. One L4 smoke FAILED before worker launch (plan-upload exec stall),
0/6 records, no rerun. Fetch/count attempted before down; jgD3s down 08:44:06.703771 UTC, session-not-found verified.
VM lifetime 617.39 s; estimated 0.325846 unit at planning rate, isolated billing unavailable. Upload budget now
interruptible; redundant control DB upload skipped and staging kept outside read-only input directories.
D3_CONTRACT remains DRAFT. See D3_REPORT.md and D3_COMMANDS.md. Stopping for audit; successful GPU worker path
remains unvalidated and D2 placeholders remain open. No D3 arm/result, no production edits, no local GPU.

2026-11-04 D3-H audit accepted f6070d3; implementing amendments 2–4 before the one newly owner-approved
amendment-1 smoke (L4 jgD3s, <=0.5 unit, one attempt). Persistent per-arm lockstep, Q7-inclusive curvature with
registered control-relative fallback, and P fallback candidate rule added. Control recomputation is running on CPU;
55 focused tests pass. Plan now embedded in the bundle before provisioning. D2 placeholders remain unresolved;
no real D3 arm is authorized or run. Revised startup-only planning total 6.86785 units before measured overhead.

2026-11-04 amended checkpoint: one owner-approved jgD3s L4 smoke PASSED, 6/6 successful DRYRUN records,
40/40 files hash-verified before down, six xz pools/266 finite-energy rows verified locally, zero replay fallbacks.
Persistent C PID4179/DS PID4285 each served all three parents with one DB/model load. VM down 09:19:07.262063 UTC;
651.457549 s lifetime, estimated 0.343825 unit at planning rate. No retry. Analysis ran once, no candidate.
Re-estimated stage units 4.153939 / 2.077490 / 2.247743, total 8.479172, before any real D3 launch.
56 final focused tests pass; exact C-2b reproduction retained. Q7 Cartesian recomputation 80/80 shows unchanged
1.33154 / 0.86400 in both scopes because production's active Q7 is the FK prior, absent the Cartesian scorer
breakdown. Numerical rule implemented; active-prior measurement definition needs clarification. No surrogate
substituted. Analysis and real stage1 preflight fail closed without active-prior coverage. Contract remains DRAFT.
See D3_H_AMENDMENTS_REPORT.md. Commit/report/stop; no further smoke or real D3 run.


2026-11-06 accepted checkpoint 0c78211; implemented audit_jgen_checkpoints_2026-11-06.md §4 ruling.
The earlier active-FK measurement blocker is superseded: curvature is Q1–Q6 plus Cartesian Scorer Q7;
FK prior is out of scope. No bend directions or reconstructed bend coordinates. Affirmative Scorer Q7 coverage
and matching effective FK identities (penalty/scale/sigma/weight/free and bend/haptic controls) are mandatory.
Local launch and isolated worker verify identities. q7_bound_centre now requires the DB .bends.json sidecar,
records bends_sha256, ships it beside the DB, and verifies SHA256 before worker launch and inside the worker.
67 focused tests passed; compileall and whitespace checks passed; production fingerprint 5f74b14b57ed unchanged.
No VM, no GPU, no curvature/panel rerun, no spend (0 units). Previous baseline and 8.479172-unit estimate retained.
Contract remains DRAFT; D2 placeholders/freeze remain pending. See D3_Q7_RULING_REPORT.md. Commit/report/stop.
