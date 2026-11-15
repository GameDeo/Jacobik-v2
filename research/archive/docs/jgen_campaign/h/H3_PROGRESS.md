# Package H-3 progress

### 2026-10-31 — H-3 completed; stopped at checkpoint

- Ran production control vs the combined H-1 + H-2 arm on one L4 VM: cal then tail, seeds 11–13. Control fingerprint `6cc586e17ab1`; combined `d587e9503943`.
- Fetched and counted all outputs before shutting down `jgH3`: 780 records, 768 valid, 12 pre-registered NEPNEF invalid-input failures; 29,128 XYZ files; 109,465,299 bytes. Colab spend 37.95 units; balance 492.84.
- **Failures:** primary CI gate failed on cal and tail; catastrophic-regression guard failed for LOPJEF and NUWFAM (cal) and IFUROQ (tail); FEKKAJ improved, NUWFAM did not; tail >200 kJ/mol rate rose. Full details are in `H3_REPORT.md` and `H3_FAILURE_LEDGER.csv`.
- Final audit: CPU test suite 390 passed / 20 skipped / 3 xfailed / 1 failed (`test_actual_and_cached_fallback_rejected_without_retry_or_geometry_change`). Fresh production and u1 golden files (245 arrays each) were recorded in a new directory. Replay parity was omitted at the owner's direction.
- Stopped at Checkpoint H-3. No merge, promotion, or fold-in.
