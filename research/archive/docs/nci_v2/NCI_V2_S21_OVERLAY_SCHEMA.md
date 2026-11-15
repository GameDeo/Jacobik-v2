# S2.1 overlay schema (s2_1_overlay.1) — schema only, no data

R = data/refit/nci_v2/scaled_v1/   (main repo; gitignored)

## Files
- R/repair_s2_1/repair_manifest.json            index (below)
- R/repair_s2_1/overlay/<CODE>_<SEED>_control.npz   one per repaired ORIGINAL pool
- R/repair_s2_1/retry_pools/                     orchestrator output of retries, normal pool
  format (<CODE>_<SEED>_control.{npz,json,mace64.json} under its own pools/ subdir as the
  orchestrator lays it out) plus its own manifest.json. Retried pools are produced at
  HEAD with R1/R2 fixes, so they carry NO overlay.
- R/repair_s2_1/frame_n_heavy.json               frozen frame + integer n_heavy per entry
- R/repair_s2_1/{pre_hashes,post_hashes}.json, ab_gate/, S2_1_REPORT.md   (audit only)

Every successful original pool gets an overlay file (R3_health is applied to all of them).

## Overlay npz (np.savez; load with allow_pickle=False must work: no object arrays)
Always present:
| key | dtype / shape | meaning |
|---|---|---|
| schema_version | str scalar (`<U`) | "s2_1_overlay.1" |
| source_pool_sha256 | str scalar | sha256 of ORIGINAL pools/<key>.npz file bytes |
| source_summary_sha256 | str scalar | sha256 of ORIGINAL pools/<key>.json file bytes |
| repair_commit | str scalar | 40-char git sha used to compute this overlay |
| repairs | str scalar | JSON list, subset of ["R1_mace","R2_mapping","R3_health"] |

If "R2_mapping" (pool originally had atom_mapping_error; mapping now built):
| gen_to_ref_idx | int64 [N] | X[:, g] is in REFERENCE order; symbols[g] == ref_symbols |
| ref_to_gen_idx | int64 [N] | inverse permutation |
| mapping_ambiguous | bool scalar | |
| num_automorphisms | int64 scalar | |
| coord_intact | bool [P] | same function as orchestrator (coordination_integrity) |
| coord_sigma_shift_A | float64 [P] | |
| coord_haptic_slip_A | float64 [P] | |
| coord_haptic_shift_A | float64 [P] | |

If "R1_mace" (eval pool whose stored MACE energies are invalid):
| energies_kj | float64 [P] | replaces mace64.json energies_kj |
| ref_energy_kj | float64 scalar | |
| top1_idx | int64 scalar | unchanged (score-based) |
| floor_idx | int64 scalar | argmin energies_kj |
| top1_kj, floor_kj, top1_dE_kj, floor_dE_kj | float64 scalars | dE = E - ref_energy_kj |

If "R3_health" (per conformer, reference order via stored or R2 mapping):
| max_bond_dev_A | float64 [P] | max abs(d_gen - d_ref) over reference covalent ligand bonds |
| min_heavy_nonbonded_A | float64 [P] | min over non-H non-metal pairs not bonded and >2.6 A apart in reference |
| graph_match | bool [P] | covalent graph from the conformer's own coords == reference graph |
| max_nn_dist_A | float64 [P] | max over atoms of nearest-neighbour distance |
| health_radii_json | str scalar | JSON {element: covalent radius A} and bond factor (1.25) used |
A pool still unmappable after R2 (e.g. IWOLEN) has NO R3 arrays and "R3_health" absent
from repairs; it is listed in manifest.unmappable.

P = conformers in the original pool, N = atoms. Precedence for a reader:
overlay key present -> use it; else original npz/json/mace64.json value.
Readers must verify source_pool_sha256 against the original file and fail closed.

## repair_manifest.json
```json
{
  "schema_version": "s2_1_overlay.1",
  "repair_commit": "<40-char sha>",
  "mapping_convention": "X[:, gen_to_ref_idx] is in reference atom order; symbols[gen_to_ref_idx] == ref_symbols",
  "frame": {"frozen_path": "...frame.json", "frozen_sha256": "...",
            "derived_path": "...repair_s2_1/frame_n_heavy.json", "derived_sha256": "...",
            "derivation": "added integer n_heavy only; canonical equality proven"},
  "pools": {
    "<CODE>:<SEED>:control": {
      "split": "surface_fit|calibration|motion_validation", "stratum": "...",
      "repairs": ["R3_health", ...],
      "overlay_path": "repair_s2_1/overlay/<CODE>_<SEED>_control.npz",
      "overlay_sha256": "...", "source_pool_sha256": "...", "source_summary_sha256": "..."
    }
  },
  "unmappable": {"<CODE>:<SEED>:control": "<error string>"},
  "retries": {
    "<CODE>:<SEED>:control": {"split": "...", "status": "success|oom|failed",
      "error": null, "pool_path": "repair_s2_1/retry_pools/...npz or null",
      "pool_sha256": "... or null", "code_commit": "<sha>"}
  },
  "slots": {
    "<FAILED_CODE>": {"stratum": "...", "state": "restored|still_failed",
      "replacement_code": "<reserve code from Step 2>",
      "replacement_state": "unused_after_retry|in_use"}
  },
  "unresolved": ["free text"]
}
```
Key format in "pools"/"retries" matches manifest.json tasks keys ("CODE:SEED:control").

---------------------------------------------------------------------------
# Step 2 run manifest (R/manifest.json): record layout (schema only)
The same layout applies to the manifest.json that the orchestrator writes for
R/repair_s2_1/retry_pools/.

Top level: run_id str, created_at str (ISO), total_tasks int, completed_tasks int,
succeeded_tasks int, failed_tasks int, total_timeouts int, per_stratum_timeouts dict,
total_wall_s float, peak_vram_mb float, peak_rss_mb float, base_provenance dict,
arm_fingerprints dict {arm: fingerprint}, replaced_failures dict, exhausted_strata list,
tasks dict.

tasks: keyed "CODE:SEED:ARM" (e.g. "ACAPDA:11:control").
status in {"success","oom","failed"}; the code can also emit "empty_conformers"
(none observed). Note: task.run_id is the literal "orchestrator_run", not top-level run_id.

SUCCESS record:
 code str, seed int, arm str, group_id str, stratum str, split str, run_id str,
 worker_pid int, status str, t_solve float, mace_t float, stage_embed_s, stage_seed_rows_s,
 stage_solver_s, stage_dedup_s, stage_mace_s, stage_other_s, wall_s float, n_confs int,
 top1_idx int, floor_idx int, top1_kj float, floor_kj float, ref_kj float,
 top1_dE_kj float|null, floor_dE_kj float|null, confs_sha256 str (sha256 of X.tobytes()),
 top1_coord_intact bool|null, floor_coord_intact bool|null,
 pool_coord_intact_fraction float|null, peak_rss_mb float, peak_gpu_mb float,
 solver_class str, solve_device str, batched_assembly bool, angle_bends bool,
 max_rows int|null, pool_path str, summary_path str, mace_path str,
 provenance dict {spec_sha256, frame_sha256, code_git_commit, code_dirty_diff_sha256,
   constraints_db_sha256, mace_model_sha256, profile_fingerprint, full_resolved_profile}
 NO n_heavy on success records (backfill: frame_n_heavy.json).
 TRAPS:
 - surface_fit has no MACE: top1_dE_kj/floor_dE_kj are null, and top1_kj, floor_kj and
   ref_kj are 0.0 PLACEHOLDERS, not energies. Its floor_idx is 0 by default, not
   energy-based.
 - The *_coord_intact fields are null when the original mapping failed (R2 overlay
   supplies the arrays).
 - For R1 pools, the stored *_kj, *_dE_kj and floor_idx values are invalid; the overlay
   overrides them.
 - Paths are repo-relative strings ("data/refit/nci_v2/scaled_v1/pools/<CODE>_<SEED>_control.npz").

FAILURE record (oom / failed):
 code, seed, arm, group_id, stratum, split, n_heavy int (always 0 in Step 2: invalid),
 run_id, worker_pid, status, error str, wall_s, peak_rss_mb, peak_gpu_mb,
 profile_fingerprint str (may be absent). No pool/summary/MACE paths and no conformer
 fields. A placeholder npz may still exist on disk for failed tasks: never treat file
 existence as success; use status.

replaced_failures: {FAILED_CODE: {stratum, failed_code, replacement_code, reason,
 timestamp}}. The replacement's tasks appear in tasks under split "surface_fit".

Pool npz (Step 2, allow_pickle=True needed: max_rows is an object array):
 X [P,N,3] f64 (gen order), symbols [N], score [P] f64 (descending), asg [P] int64
 (candidate index; None mapped to 0), rank [P] f32, Z f64, t_solve, ref_X [N,3],
 ref_symbols [N], charge, mult, profile (JSON str), profile_fingerprint,
 worker_pid, deterministic_embed, arm, seed, code, solver_class, solve_device,
 batched_assembly, angle_bends, max_rows (object), terms_Q1..terms_Q6 [P] f32,
 [mapping keys as in the overlay R2 list] OR atom_mapping_error str,
 spec_sha256, frame_sha256, code_git_commit, code_dirty_diff_sha256,
 constraints_db_sha256, mace_model_sha256.
mace64.json (eval splits): {code, seed, arm, run_id, energies_kj [P] (in pool conformer
 order), ref_energy_kj, top1_idx, floor_idx}. For surface_fit, an empty placeholder.
