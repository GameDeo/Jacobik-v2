# H-3b progress

## 2026-10-31 — Checkpoint H-3b; stopped for audit

- Local PC / RTX 2060 SUPER only; Colab spend **0**. Branch jgen/h-patch.
- Implementation **63d2bb2**; frozen confirmation source/contract **49814f8**.
- All eight four-arm cal/tail phases completed: **1,560 records, 1,536 valid**,
  24 preregistered NEPNEF failures; no unexpected failures, missing pairs,
  metadata violations or replay fallbacks. Confirmation wall **97.04 min**.
- Combined cal top-1 **−4.148 [−10.024, −0.402] kJ/mol: PASS**.
- Cap alone **−1.877 [−4.549, −0.128]: PASS**.
- Donor alone **−2.815 [−8.164, +0.228]: standalone benefit unproven**.
- No >+25 loss on ≥2/3 seeds for any candidate on cal or tail.
- Original quadratic-tail design dropped; FEXWUC tail win sacrificed.
  Full directive goal **not met** despite the passing package primary.
- No severe <15° NUWFAM/QUTWOT geminal collapse in final pools. Face-mean
  necessity, changed wall coordinate/calibration and empirical cap scope
  remain philosophical/generalizability questions for audit.
- Starting-mode inspection does not establish a mode-lock replacement:
  the valid reference and collapsed rows can share the same MAP component.
  No locked-component optimization ablation was performed.
- Stable final suite **410 passed / 20 skipped / 3 xfailed**; golden both
  profiles **245/245, zero differences**; canonical replay regeneration exact.
- Deterministic 200-step CUDA parity is exact on three units. Ordinary CUDA
  strict parity **fails**, including FEKKAJ graph error above its eager-repeat
  variation; both results retained and distinguished in the report.
- No merge, promotion, default change or confirmation-driven retuning.

Final audit handoff: [H3B_REPORT.md](H3B_REPORT.md).
Full development history, failed attempts and corrections: [H3B_LOG.md](H3B_LOG.md).
