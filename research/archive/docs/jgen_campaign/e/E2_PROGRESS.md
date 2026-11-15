# E-2 progress — donor-pair score fix on cal

Branch: `jgen/e-score`
Date: 2026-10-30

## Dated log

- **2026-10-30 — evaluation contract:** committed `E2_CONTRACT.md` as `5b2b927` before opening the cal outcomes. Candidate dictionary is fixed at `hh=0.95`, `ho=0.90`, `oo=0.80`; `angle_tail_quadratic` remains off in both cal arms.
- **2026-10-30 — Step 0:** flags-off production fingerprint `6cc586e17ab1`; golden check 245/245 arrays matched for production and u1. Tail crystal check: 996 Q2 pairs over 120 requested cal parents, no failures; 6.325% beyond `|z|=2`, extra-cost median 0 and p95 0.01510 nats. L4 replay checks used seeds 11 and the prescribed cases.
- **2026-10-30 — runtime estimate:** FEKKAJ and XEYSUP, seeds 11–13, each arm on the same L4. Control completed in 64.7 s for both parents; donor-pair completed in 62.0 s. Candidate/control estimate ratio 0.958.
- **2026-10-30 — cal:** 119 eligible parents × seeds 11–13, excluding only NEPNEF. Shards were 59 and 60 parents on L4 VMs `jgE2` and `jgE3`; control and candidate ran sequentially on the same VM within each shard. All 357 records per arm are `ok`. Each arm has 714 mapped XYZ files (top-1 and floor for each unit). Both VMs were fetched/count-checked and shut down; active Colab assignments are zero.
- **2026-10-30 — spend:** E-2 Colab balance fell from 499.23 to 496.15 units, about 3.08 units total; no local GPU use.

## Current status

Cal primary and all stated guards pass. The Step 0 CPU/CUDA replay comparison has a top-row mismatch on USAPOU despite replay mode being active; this is reported as an unresolved replay-parity issue. The benchmark runner exported mapped top-1 and floor XYZs, not every pool coordinate; the reported 1 Å coverage therefore describes exported selections, not full-pool coverage. See `E2_REPORT.md`.

No fold-in. No production fingerprint change.
