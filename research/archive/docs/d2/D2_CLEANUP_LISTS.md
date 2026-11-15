# D2 cleanup inventory — 2026-11-09

Owner review list only. No flags or data were deleted, renamed, or moved. This is a static inventory: no flag removal was applied, so removal-specific fingerprint/golden invariance has **not** been verified. Production defaults on this branch remain the baseline (`6b47c3fb45a9`); that is not evidence that deleting a field is safe.

## Profile field classification

Classification follows current use: **production** means active in the baseline configuration; **D arm** means part of the named D2 profile; **retire** means the plan explicitly says this experiment failed or is superseded; **dormant** means a registered experiment with no current promotion. The field-to-reader mapping is the module/attribute tuple in `src/jacobik/profile.py::FIELDS`. “Reader” below means the declared module; some flags also have generated replay / explicit API paths.

| Class | Fields (`profile.FIELDS`) | Declared readers | Retirement evidence / disposition |
|---|---|---|---|
| Production | `finish_phase`, `dedup_factored` | `jacobik.ik.ik_solver`; `jacobik.ik.dedup` | Current production settings. Keep. |
| D arm | `hapto_bond_gate`, `hapto_label_canon`, `hapto_chelate_exclusion` | `jacobik.topology.donor_classifier`; `jacobik.stats.key_builder` | D2 profile enables T1d runtime typing. Keep through D3 decision. |
| D arm | `q6_physical_all`, `q6_weight_14` | `jacobik.stats.q6` | D2 profile enables physical Q6 at runtime; `q6_weight_14=0`. Keep through D3 decision. |
| D arm | `os_sanity`, `fit_exclude_locked` | `fit.sanity` | D2 fit profile filters charge-implausible and locked parents. DB build provenance/use applies; do not remove as solver cleanup. |
| Retire candidate | `q7_bound_centre`, `q7_shift_max` | `jacobik.ik.scorer` | Plan §2.1 says q7 bound-centre is closed; feature is off and requires embedded bends table (`profile.FLAG_TABLES`). Retire candidate; no removal check. |
| Retire candidate | `out_per_class`, `out_per_class_k`, `out_per_class_window` | `jacobik.engine`; `jacobik.ik.stereo` | Plan §3.1 identifies ASP as failed; profile header says it stays off. Explicit engine API arguments and stereo enumeration still exist, so audit call sites before removal. |
| Retire candidate | `beam_union`, `beam_union_share`, `beam_union_tol`, `beam_union_group_batch` | `jacobik.engine`; `jacobik.ik.beam` | Plan §3.1 says union beam U is superseded by Phase 2 allocation/halving. Search/report code still names U; candidate only. |
| Retire candidate (specific value) | `dof_step_scale` (ring override `{"ring": 0.5}`) | `jacobik.ik.ik_solver`; generated `jacobik.ik.solver_replay` | Plan §3.1 and §6 say pre-iRprop ring step tuning is superseded. Retire the historical ring value only; field supports other overrides and remains registered. Replay must be regenerated if reader code changes. |
| Dormant | `start_kabsch`, `start_kabsch_pivot`, `start_kabsch_diversity` | `jacobik.engine` | Off-by-default experiments; no retirement evidence in this directive. |
| Dormant | `pool_window_fill` | `jacobik.engine` | Off-by-default experiment. |
| Dormant | `finish_schedule`, `finish_early_stop`, `finish_zigzag`, `finish_patience`, `finish_delta` | `jacobik.ik.ik_solver` | Production finish-phase controls / optional schedules; not on the retire list. |
| Dormant | `polish`, `optimizer`, `polish_dt`, `lr_final`, `bend_free_deg`, `adam_metric`, `fire_metric` | `jacobik.ik.ik_solver`; `jacobik.ik.angle_bends` | Optional optimizer/step experiments; not on the retire list. |
| Dormant | `embed_bond_check` | `jacobik.engine` | Experiment flag, off by default. |
| Dormant | `angle_tail_quadratic` | `jacobik.ik.scorer` | Experiment flag, off by default. |
| Dormant | `q6_class4_physical` | `jacobik.stats.q6` | Separate class-4 experiment, not D2 physical-Q6 setting. |
| Dormant | `dedup_rmsd`, `dedup_max_dev` | `jacobik.ik.dedup` | Threshold experiments; factored dedup itself is production. |
| Dormant | `stereo_candidates`, `stereo_scope`, `stereo_cost_ceiling` | `jacobik.engine` | Stereo candidate flag remains off; current stereo work is separately gated and not a retirement decision. |
| Dormant | `donor_promote_geometric`, `donor_promote_geometric_ratio`, `donor_promote_geometric_scope` | `jacobik.topology.donor_classifier` | Off-by-default donor promotion experiment. |
| Dormant | `prune_below_thermal` | `fit.thermal_prune` | Off-by-default fit experiment. |

The requested names `q6_scale_14` and `beam_target_aware` do not occur as registered `FIELDS` or runtime symbols in this checkout (`rg` source search); no current field can be classified for retirement under those names. `beam_target_aware` appears in an old coverage script/doc; inspect history before removing any such references. The registered union-beam names above are the current U implementation.

**Removal check status:** no candidate was removed. Consequently no claim is made that removing any candidate preserves fingerprint or golden output. To establish this, remove an approved candidate on an isolated branch, run `python -m tools.check ws`, and report the resulting fingerprint and golden comparison. An off-by-default flag can still affect fingerprint serialization or API compatibility when removed.

## Code paths

| Path | Static citations / callers found | Proposed action |
|---|---|---|
| `refit/run_refit.py`, root `refit_lib.py` / `refit/refit_lib.py` | `run_refit.py` imports the old helper; active research tools still import `refit.refit_lib` (`research/refit/nci_refit.py`, `nci_v2_frame.py`, multiple `research/bench/tools/jdev/budget/*` scripts); tests import `refit_lib`. | Keep pending caller migration. Not safe to archive as a unit. Split old driver from shared library only after scripts/tests have supported replacements. |
| `src/fit/common/pooling.py` | `src/fit/common/__init__.py` re-exports it; `tests/test_db_container.py` imports `pool_ladder`. | Keep. D2 implementation is not proof of supersession; migrate callers and tests first. |
| `research/archive/` fit/NCI tests and pipelines | Numerous historical tests still import `refit` paths; some are archived evidence rather than production entrypoints. | Keep as evidence for now; inventory each archive group before packing or removal. |

These are `rg` source-reference results, not runtime call-graph proof.

## Local data roots

The metadata-only inventory of immediate entries is in `D2_LOCAL_DATA_INVENTORY.md`; machine-readable counts are in main-checkout `research/data/d2/db_d2/local_inventory.json`. It reports logical bytes and files without following 256 symlinks; the scan reported no errors. Exact-path tracked citations are indicative only. Root totals: `research/data` 9,888,864,383 bytes / 259,654 files; `research/archive/data` 5,843,223,688 bytes / 156,247 files; `data` 918,566,922 bytes / 1,248 files. Symlink targets are not included in these totals. Protected paths `data/db/constraints_2026-10-18.json` and `data/refit/experiments/index.json` remain never-write. No deletion or movement is proposed without owner approval.

## Perlmutter scratch inventory

Per-entry counts, exact-path tracked citations and proposed actions are in `D2_SCRATCH_INVENTORY.md`. Source: metadata-only inventory at `/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/d2/db_d2/scratch_inventory.json`, generated 2026-11-09. It sums logical bytes/file counts by immediate entry; symlinks were omitted and inaccessible files were skipped without an error count. The owner-reported filesystem total is **19.4 GiB / 176K inodes**; totals below are partial inventory, not quota totals. No file contents were opened. Keep/pack proposals are tentative; no scratch changes were made.

| Root | Inventory size | Files | Recommendation and known citation/use |
|---|---:|---:|---|
| `$SCRATCH/jacobik/cache` | 12.10 GB | 92,998 | Keep active runtime caches. `cache/rattler` alone is 11.93 GB / 86,218 files; do not delete without checking active pixi/env processes and cache provenance. |
| `$SCRATCH/jacobik/data` | 1.30 GB | 7,722 | Keep. Includes DBs/sidecars, golden copies and `MANIFEST.sha256`; D2 DB is required for D3. `data/ref` 573 MB / 70 files; `data/db_d2` 183 MB / 4 files; `data/db` 148 MB / 2 files. |
| `$SCRATCH/jacobik/ext` | 1.16 GB | 13,020 | Pack/archive snapshots after branch checkpoints and confirm each `COMMIT`/run manifest. Includes D2 build snapshot (86 MB / 1,011 files), multiple agent snapshots and a coverage tar. |
| `$SCRATCH/jacobik/runs` | 5.73 GB | 34,620 | Preserve until the wrap-up reports cite them; then pack completed multi-file runs with manifests. D2 build run 61051826 is 1.27 GB / 6,493 files plus a 1.28 GB tar; run 61064577 is 432 MB / 15 files plus a 432 MB tar; D2 smoke 240 KB / 39 files. Stereo login is 85 MB / 7,433 files. Do not remove failed runs until their failure is recorded. |

Scratch inventory sums to ~18.3 GB / 148,360 listed files, below owner-reported 19.4 GiB / 176K inodes because the scan omitted symlinks/inaccessible paths and may differ in allocated-vs-logical bytes. Exact cited consumers by individual run directory remain to be added by the reports; the D2 build/smoke and current campaign outputs are known active evidence.
