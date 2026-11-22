"""
src/jacobik/profile.py
Experiment profile and flag hook for Jacobik.

Phase 4 consolidation (2026-10-26):
All 37 baseline profile flags have been finalized and hardcoded directly into the production
solver modules under the validated u1 configuration:
- nci = False
- deterministic_embed = True
- ring_fold_hinges = False
producing baseline production fingerprint: 711411157fb4.

Package F fold-in (2026-10-27):
Promoted `bend_release = [0.1, 100]` to production default in solver modules, producing the new
production fingerprint: 615ae97d2350 (superseding 711411157fb4).

Li-cap fold-in (2026-10-29):
Promoted `embed_li_cap = True` (temporary Li counter-ions at locally anionic kappa donors during the GFN2
embed refinement; F-5) to production, producing the production fingerprint 6cc586e17ab1 (superseding 615ae97d2350).

Package H fold-in (2026-10-31):
Promoted `q6_donor_pairs = {"hh": 0.95, "ho": 0.90, "oo": 0.80}` (kappa-kappa donor pairs on different ligands scored by
the physical Q6 repulsion with crystal-derived onset scales; FEKKAJ fix) to production, producing the production
fingerprint 3f75c9525a4e (superseding 6cc586e17ab1).

eta2 fold-in (2026-11-01):
Promoted `eta2_face_plane = True` (eta2 faces with >= 2 substituents on each atom: the face normal is the plane through
the bond that best fits the substituents; S_orient and the lean DOF) to production, with the constraints DB
constraints_2026-11-01.json (the eta2 S_orient keys refit under the plane on every eta2 face; every other key as in
constraints_2026-10-18.json), producing the production fingerprint b1fcfae58190 (superseding 3f75c9525a4e). Owner
accepted the failed edge-on gate (ETA2_RESULT.md, amendment 2) in favour of the typing fix later. In the same fold-in:
`embed_haptic_li_cap = True` (temporary Li caps on eligible anionic haptic faces during the GFN2 embed) and
`eta2_group_bend = True` (on eta2 faces without the shared plane, a face atom's substituents bend as one rigid group),
producing the production fingerprint 273f2ccbf892 (b1fcfae58190 = the plane alone, never production on master).

IA-k fold-in (2026-11-02, G-3g, owner-approved after the G-3f audit):
Production uses iRprop+ search, switch 100 to iRprop+ finish, thermal widths,
releasing symmetry only while retaining shape/projection. L4 cal top-1 -10.9
kJ/mol [-18.7, -4.6], floor -9.5, end-to-end wall 1.22x. Regression guard
failed (3 parents vs threshold 2); the audit found a degenerate C1-repeat null
and a seed-rotation null of 0-2, against 16 mirror improvements. Owner accepted
the failure. Production fingerprint 5f74b14b57ed is the tested IA-k identity.
Set finish_phase=null to reach the preceding C1 behavior (273f2ccbf892).

This module is the empty, documented hook for future experiments:
1. To add a new experimental flag, register it in `FIELDS`:
   FIELDS["flag_name"] = ("jacobik.module", "ATTR_NAME", default_val, "Description")
   and have that module read it at import: `ATTR_NAME = profile.get("flag_name")`. Nothing copies override values
   into modules for you; a flag that no code reads changes the fingerprint and nothing else.
2. Override flags via `JACOBIK_PROFILE_OVERRIDES` (JSON string, e.g. '{"flag_name": true}').
3. Adding or overriding flags changes `snapshot()["fingerprint"]`, preserving result provenance.
4. Unregistered PRODUCTION_CONFIG values are hardcoded in their modules and no longer read from here. An override may restate
   one (e.g. the bench's '{"nci": false}'), but setting one to a different value raises: it would be a silent no-op arm.
   To test a production setting again, register it as a new FIELDS flag and read it in its module (step 1).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from typing import Any, Dict, Optional, Tuple

#: DS data fold-in (owner 2026-11-12; gate J3, Perlmutter 61160804, arms C = previous production vs D = this configuration + DB `constraints_2026-11-12.npz` (entry-for-entry the gated DF json)
#: (sha256 be4e057108c0552bdbd5b923dc8a0f452331624e147eee8c9e5095986b90e4db), seeds 11-13, audited target overlay sha 9a9c820b... and gate_eligibility_v1.json on both arms):
#: stage 1 (109 primary parents) top-1 -9.27 kJ/mol [-18.31, -1.58], all seven gates true; stage 2 (95 holdout parents) top-1 -24.19 [-43.96, -7.14], but the registered
#: holdout and failures gates are False because parents did not complete (ADINOR, LADJAB, UZOREI, EPUPEL; owner accepted 2026-11-12); stage 3 not run. Promoted to
#: production: hapto_bond_gate, hapto_label_canon, hapto_chelate_exclusion, q6_physical_all (q6_weight_14 stays 0.0), os_sanity, fit_exclude_locked (the former "d2"
#: package) and donor_identity_reconcile, donor_assignment, stable_ancestor_sampling, structure_sanity (the DS package). The new fingerprint equals the fingerprint of
#: the gated D arm. The start_kabsch / start_kabsch_pivot starts stay off (their candidate rule failed in Step 6). Every flag stays registered in FIELDS: set it False
#: to reach the previous behaviour. Supersedes 5b56969afb9c (stereo fold-in), whose DB is data/db/constraints_2026-11-01.json. Records: research/archive/docs/ds/DS_GATE_RESULTS.md, research/archive/docs/ds/DS_HANDOFF.md.
#: Isomer fold-in, dedup only (owner-approved 2026-11-08): dedup_factored on; supersedes IA-k 5f74b14b57ed.
#: Stereo fold-in (owner 2026-11-10, D3 stage 1x AS vs A on fresh seeds 31-33: top-1 -2.3 kJ/mol, hit<=1 A +3.8 pp, +4.5 distinct
#: rows, guard 0 trips): stereo_candidates on as the superset of production (production E/Z classes kept, capped extras);
#: supersedes 6b47c3fb45a9. out_per_class stays off (ASP failed). Fingerprint = sha1(PRODUCTION_CONFIG)[:12].
#: P2R fold-in (branch foldin/p2r, prepared 2026-11-13; merge on the owner's go after the gate in
#: research/campaigns/p2r/FOLDIN_GATE_CONTRACT.md): out_mirror, ring_states, ring_states_five, config_states and config_near on.
#: Dev evidence (cal + tail, seeds 11-13, login A100, final code): C-MISS at the registered crude relaxation 0.322 -> 0.239,
#: d -0.083 [-0.128, -0.043]; C-TOP1 0.407 -> 0.413. The new fingerprint equals that arm's. config_remote stays off (no panel parent
#: has an element it can derive). Supersedes d5e2f4113497 (DS data fold-in); the DB is unchanged. Record: FINAL_PASS.md.
#: Merged `donor_identity_reconcile` into `donor_assignment` (2026-11-14, fingerprint `2c87e2940af2`, superseding `a47283387172`).
#: v1.0b fold-in (branch foldin/v1.0b, prepared 2026-11-20; merge on the owner's go): macro_open_adj, macro_root, macro_pool_rank,
#: li_cap_spread, embed_refine="mace", settle_steps=10, dedup_qcp and dedup_capped_fast on; the former stereo_cage flag folded into
#: stereo_candidates (cage diastereomers are stereo candidates). Evidence: v1.0b reports .claude/REPORT_v1.0b_2026-11-18b.md and
#: .claude/STATUS_v1.0b_and_NN_2026-11-19.md (FOLD2 arm 87db9ca25d49 = this configuration without dedup_capped_fast, which keeps the
#: same rows on every captured capped call). Supersedes 2c87e2940af2 (v1.0a); the DB is unchanged. Set a flag back to reach v1.0a.
PRODUCTION_FINGERPRINT = "1e4cde06304f"

IA_K_FINISH_PHASE = {
    "search": "irprop+", "optimizer": "irprop+", "switch": 100,
    "release": ["symmetry"],
    "sigma": {"translation": .0585, "rotation": .08609, "torsion": .2402,
              "ring": .2402, "bend": .08609},
    "sigma_source": "thermal",
}

#: Production values; finish_phase remains registered and explicitly overrideable.
PRODUCTION_CONFIG: Dict[str, Any] = {
    "dedup_factored": True,
    "stereo_candidates": True,
    "out_mirror": True,
    "ring_states": True,
    "ring_states_five": True,
    "config_states": True,
    "config_near": True,
    "macro_open_adj": True,
    "macro_root": True,
    "macro_pool_rank": True,
    "li_cap_spread": True,
    "embed_refine": "mace",
    "settle_steps": 10,
    "dedup_qcp": True,
    "dedup_capped_fast": True,
    "finish_phase": IA_K_FINISH_PHASE,
    "angle_bends": True,
    "beam_mode_score": "density",
    "bend_per_z": True,
    "bend_release": [0.1, 100],
    "bound_surrogate": True,
    "branch_bends": True,
    "branch_prior": False,
    "bvs": False,
    "carbene_ylide": True,
    "chelate_ez": True,
    "dedup_rmsd": 0.5,
    "donor_assignment": True,
    "fit_exclude_locked": True,
    "hapto_bond_gate": True,
    "hapto_chelate_exclusion": True,
    "hapto_label_canon": True,
    "os_sanity": True,
    "q6_physical_all": True,
    "stable_ancestor_sampling": True,
    "structure_sanity": True,
    "deterministic_embed": True,
    "donor_bends": False,
    "embed": "gfn2",
    "embed_haptic_li_cap": True,
    "embed_li_cap": True,
    "eta2_cap": False,
    "eta2_face_plane": True,
    "eta2_group_bend": True,
    "ez_enumerate": True,
    "haptic_bends": True,
    "haptic_free_lean_max": 6,
    "haptic_lean_all": True,
    "haptic_lean_wall": True,
    "haptic_per_sub": True,
    "local_plane_deg": 20.0,
    "nci": False,
    "nci_types": ("hb",),
    "out_dedup": True,
    "out_window": 200.0,
    "oxidation_model": "training",
    "q1_tail_sigma": 0.0,
    "q1_wall": True,
    "q5_fold": "auto",
    "q6_donor_pairs": {"hh": 0.95, "ho": 0.90, "oo": 0.80},
    "q6_model": "repulsive",
    "rank_agg": "sum",
    "rank_floor_fit": False,
    "ring_fold_hinges": False,
    "ring_joints": True,
    "ring_open_adj": True,
    "score_agg": "sum",
    "solve_floor": "thermal",
}

#: Experiment fields: name -> (module, attribute, default_value, description)
#: The promoted finish_phase default and the available off-by-default experiments.
FIELDS: Dict[str, Tuple[str, str, Any, str]] = {
    "start_kabsch": ("jacobik.builder.engine", "START_KABSCH", False, "Fit every ligand pose to its own torsion-seeded row"),
    "start_kabsch_pivot": ("jacobik.builder.engine", "START_KABSCH_PIVOT", False, "Compensate FK's original donor-atom pivot (requires start_kabsch)"),
    "start_kabsch_diversity": ("jacobik.builder.engine", "START_KABSCH_DIVERSITY", 1.0, "Well-posed orientation perturbation scale for per-row Kabsch"),
    "out_mirror": ("jacobik.builder.engine", "OUT_MIRROR", True, "Complete enantiomers by reflecting the output rows (only when no ligand carries a tetrahedral tag); the reflected rows keep their source score"),
    "ring_states": ("jacobik.builder.engine", "RING_STATES", True, "Per-ligand ring-pucker state library in the one batch (chairs, twist-boats of near six-rings; rows split by state energy, state energy in the score)"),
    "config_near": ("jacobik.builder.engine", "CONFIG_NEAR", True, "Stereo (needs config_states): classes = capped product of the open elements (near: binding-created or within 2 bonds of a donor; remote kept when not derivable), mirror pairs collapsed only with out_mirror and no tag, cap x2 (one batch); an extra that repeats a kept configuration stays (a second embed and search)"),
    "config_remote": ("jacobik.builder.engine", "CONFIG_REMOTE", False, "P2R (needs config_near): remote acyclic stereo elements leave the solved product; each class's best rows are inverted there locally (isometric subtree reflection / 180 deg bond turn, jacobik.builder.configs), repaired by a short joint IK, rescored and added as derived classes"),
    "config_states": ("jacobik.builder.engine", "CONFIG_STATES", True, "Every configuration class (E/Z combinations, stereo classes) solved in ONE joint IK as row blocks (per-class prep, per-block cull, per-class selection, same merge); replaces the sequential class solves"),
    "embed_gfn2_opt": ("jacobik.builder.engine", "EMBED_GFN2_OPT", "crude", "xtb --opt level of the GFN2 ligand embed refinement: crude (production), sloppy, loose, normal, tight. The frozen bond lengths/angles of every row come from it (XOJHIN carbene N-C-N 129 deg after crude, bound 119)"),
    "li_cap_spread": ("jacobik.builder.engine", "LI_CAP_SPREAD", True, "GFN2 embed refinement: Li counter-ion caps of anionic donors are spread so no two coincide (macrocycle cavities: UYOVUA's two caps 0.55 A apart made the GFN2 SCF diverge)"),
    "macro_fused": ("jacobik.topology.decompose", "MACRO_FUSED", False, "Rings of >= 8 atoms with >= 2 donors get macrocycle joints even with non-single or fused bonds (kept rigid; >= 4 free single bonds); FK joints only bonds that split the opened ring, opening on an unshared bond"),
    "macro_root": ("jacobik.topology.decompose", "MACRO_ROOT", True, "Root the FK tree of a ligand with a jointed macrocycle (>= 8 atoms) at the ring body instead of the farthest donor-free body (a pendant arm's end)"),
    "macro_pool_refine": ("jacobik.builder.engine", "MACRO_POOL_REFINE", None, "Macrocycle pool conformers after their free-ligand MMFF: None = as is; 'gfn2' = GFN2 crude refinement with the donor-donor distances held, as the ligand embed (parallel single-threaded xtb runs); 'mace' = the embed_refine MACE refinement, all conformers of the pool in one batch"),
    "macro_template": ("jacobik.builder.engine", "MACRO_TEMPLATE", False, "Macrocycle ring-conformer pool embedded around a virtual metal (each donor at its Q1 mean distance, donor substituents >= 100 deg from the M-D bond, other atoms clear of the metal; MMFF with donor-donor restraints), template-ranked and diversified; free-ligand ETKDG pool as fallback"),
    "embed_refine": ("jacobik.builder.engine", "EMBED_REFINE", "mace", "Ligand embed refinement on MACE-POLAR instead of GFN2-xTB, every embed of a call in one batched optimisation (builder.mace_embed): None = GFN2 (production); 'mace' = GFN2 semantics (free ligand, Li caps, donor-donor distances held); GFN2 fallback on any failure"),
    "macro_haptic_filter": ("jacobik.builder.engine", "MACRO_HAPTIC_FILTER", False, "Ring pool of a ring with >= 2 haptic faces (COD, CDT): keep only folds whose faces can all point at one metal site (face sites 2.05 A off each face centroid within 2.5 A); free COD: drops the chair (33 of 64 ETKDG folds), keeps the tubs; the input embed is always kept"),
    "macro_pool_rank": ("jacobik.builder.engine", "MACRO_POOL_RANK", True, "Rank the conformer pool of every jointed ring with >= 3 kappa donors, any element, by their lone-pair convergence (beam._donor_face_convergence); production ranks only rings with >= 3 sp3 ring nitrogens and takes the first free-ligand folds unranked for crown ethers, thiacrowns and mixed N/O/S rings. Rings with a haptic segment (COD, CDT) keep the production pool (face-normal ranking made COD worse)"),
    "macro_open_adj": ("jacobik.ik.fk", "MACRO_OPEN_ADJ", True, "Macrocycles (>= 8 all-single rings) also get a torsion joint on the two ring bonds next to the opening, as small donor rings do (ring_open_adj); without them two ring torsions and the frame of any arm on the opening atoms stay at their seeded values (n-8 instead of n-6 ring degrees of freedom)"),
    "ring_states_five": ("jacobik.topology.rings", "FIVE", True, "ring_states also offers saturated five-ring puckers near a donor (envelopes/twists on the pseudorotation circle); needs ring_states"),
    "finish_phase": ("jacobik.ik.ik_solver", "FINISH_PHASE", IA_K_FINISH_PHASE, "Production IA-k; null restores C1. Dictionary: release, sigma, cap, search, optimizer, switch, score_only"),
    "polish": ("jacobik.ik.ik_solver", "POLISH", None, "Delivered-score polish after the solve: irprop (iRprop+) or irprop_minus (iRprop-); None = off"),
    "optimizer": ("jacobik.ik.ik_solver", "OPTIMIZER", None, "Whole-run score-only optimizer: irprop (iRprop+) or irprop_minus (iRprop-); None = Adam"),
    "dof_step_scale": ("jacobik.ik.ik_solver", "DOF_STEP_SCALE", None, "Per-DOF-class step scale overrides (dict, e.g. {'rigid': 1.5})"),
    "bend_free_deg": ("jacobik.ik.angle_bends", "FREE_DEG", 2.0, "Angle bend prior free zone in degrees (default: 2.0)"),
    "embed_bond_check": ("jacobik.builder.engine", "EMBED_BOND_CHECK", False, "Constrain stretched bonds during GFN2 embed refinement"),
    "dedup_factored": ("jacobik.ik.dedup", "DEDUP_FACTORED", True, "Factored symmetry for output deduplication"),
    "dedup_rmsd": ("jacobik.ik.dedup", "DEDUP_RMSD", 0.5, "Heavy-atom symmetric RMSD threshold (Angstroms)"),
    "settle_steps": ("jacobik.builder.settle", "SETTLE_STEPS", 10, "Settle stage (builder.settle): k batched MACE-POLAR steps on a candidate pool settle_pool x the requested size, metal-donor distances held, then dedup and per-class output by settled energy; 0 = off (production: the solver rows as scored)"),
    "settle_pool": ("jacobik.builder.settle", "SETTLE_POOL", 2.0, "Candidate pool factor of the settle stage (n_conformers / out_per_class_k times this before settling)"),
    "settle_dedup": ("jacobik.builder.settle", "SETTLE_DEDUP", False, "Settle stage: deduplicate the settled rows (production deduper) before output (on: -12 % low basins, 2026-11-19)"),
    "relax_carry": ("jacobik.builder.settle", "SETTLE_CARRY", False, "Carry the settle stage's L-BFGS state (curvature pairs formed with no hold active, trust radius) into the final relaxation of the same row (EnsembleResult.relax); a row whose coordinates changed in between starts cold"),
    "settle_hold": ("jacobik.builder.settle", "SETTLE_HOLD", 0.25, "Flat-bottom half width (A) on every metal-donor distance during the settle stage; 0 = unrestrained"),
    "dedup_capped_fast": ("jacobik.ik.dedup", "DEDUP_CAPPED_FAST", True, "Capped automorphism groups (>= MAX_AUTOMORPHISMS): the factored symmetric match decides alone (the scan over the capped automorphism list found no duplicate it missed on 15 capped parents, 2026-11-20), with the core-start scan on the GPU when one is available, starts screened by a nearest-same-class-atom RMSD (<= 2.5 x the threshold) before the first alternating round and only starts within 1.5 x after it refined further; with CUDA, select() decides all (row, kept row) and in-block row pairs at once (core-start scan, start rotations and screen on the GPU, rounds on the CPU for the surviving starts; same rows on all 25 captured capped calls, QEZVEY 268 s -> 5 s on an A100, 2026-11-20)"),
    "dedup_qcp": ("jacobik.ik.dedup", "DEDUP_QCP", True, "Faster factored symmetric match: closed-form Kabsch (QCP: Horn key-matrix eigenvalue by Newton, rotation from its eigenvector) instead of batched 3x3 SVDs (agrees to 1e-12, SVD fallback where Newton has not converged)"),
    "dedup_max_dev": ("jacobik.ik.dedup", "DEDUP_MAX_DEV", 0.0, "Optional second test: maximum per-atom deviation threshold (0.0=off)"),
    "stereo_candidates": ("jacobik.builder.engine", "STEREO_CANDIDATES", True, "Enumerate unspecified and binding-created stereoisomer candidates"),
    "stereo_scope": ("jacobik.builder.engine", "STEREO_SCOPE", "B", "Scope of stereoisomer enumeration: 'B' (centres + E/Z + binding-created incl. M-C=X)"),
    "stereo_cost_ceiling": ("jacobik.builder.engine", "STEREO_COST_CEILING", 1.5, "Candidate class cost ceiling multiplier relative to cal wall (uniform class sampling above it)"),
    "out_per_class": ("jacobik.builder.engine", "OUT_PER_CLASS", False, "Enforce per-class output guarantee: keep up to K rows per eligible class inside the window"),
    "out_per_class_k": ("jacobik.builder.engine", "OUT_PER_CLASS_K", 8, "Maximum rows kept per class in per-class output contract"),
    "out_per_class_window": ("jacobik.builder.engine", "OUT_PER_CLASS_WINDOW", 200.0, "Per-class score window in nats"),
    "hapto_bond_gate": ("jacobik.topology.donor_classifier", "HAPTO_BOND_GATE", True,
                        "Reject eta2 haptic faces whose most distant atom exceeds the covalent-radius bond ratio"),
    "hapto_label_canon": ("jacobik.stats.keys", "HAPTO_LABEL_CANON", True,
                          "Use atom-order-independent coordination labels for eta2 faces"),
    "hapto_chelate_exclusion": ("jacobik.topology.donor_classifier", "HAPTO_CHELATE_EXCLUSION", True,
                                "Reject eta2 C-X faces when carbon is bonded to two kappa donors to the same metal"),
    "donor_assignment": ("jacobik.topology.mapping", "DONOR_ASSIGNMENT", True,
                         "Source donors as one evidence-ranked injective assignment within symmetry classes"),
    "stable_ancestor_sampling": ("fit.sanity", "STABLE_ANCESTOR_SAMPLING", True,
                                 "Select ancestor observations by persistent per-observation priority"),
    "os_sanity": ("fit.sanity", "OS_SANITY", True,
                  "Exclude parents with implausible source metal formal charge and use metal-level OS fallback"),
    "structure_sanity": ("fit.sanity", "STRUCTURE_SANITY", True,
                         "Exclude parents whose carbon framework is broken (>= 25% of carbons under-bonded); applied with the os_sanity parent filter"),
    "fit_exclude_locked": ("fit.sanity", "FIT_EXCLUDE_LOCKED", True,
                           "Exclude locked benchmark parents from fitted reference tables"),
    "q6_physical_all": ("jacobik.stats.steric", "Q6_PHYSICAL_ALL", True,
                        "Use physical repulsion for supported non-bonded Q6 pairs, including 1-4, metal and polar contacts"),
    "q6_weight_14": ("jacobik.stats.steric", "Q6_WEIGHT_14", 0.0,
                     "Physical Q6 energy multiplier for 1-4 pairs; zero omits them when q6_physical_all is enabled"),
}

#: Named profiles: "production" (default) and "u1" (alias). Legacy profile has been removed.
#: "d2" (owner 2026-11-09, directive_jgen_D2): the D arm for D3 = the D2 constraints DB + these runtime flags. The same
#: profile builds the D2 DB (python -m fit.build_db --profile d2): the three T1d typing flags act at extraction and at
#: solve time; q6_physical_all/q6_weight_14 are runtime only (the DB still fits the Q6 tables). Historical since the DS
#: fold-in (2026-11-12): these flags are now the production defaults, so "d2" no longer differs from "production" in
#: those flags (it is kept so the provenance of the D2 DB, whose meta says profile d2, still resolves).
PROFILES: Dict[str, Dict[str, Any]] = {
    "production": {},
    "u1": {},
    "d2": {
        "hapto_bond_gate": True,
        "hapto_label_canon": True,
        "hapto_chelate_exclusion": True,
        "q6_physical_all": True,
        "q6_weight_14": 0.0,
        # fit-frame filters (owner 2026-11-09): charge-implausible and locked parents are left out of the D2 fits
        "os_sanity": True,
        "fit_exclude_locked": True,
    },
}

#: Runtime flags that need a table embedded in the DB container (`_tables`). The bench and tools.run refuse to start when
#: the DB lacks it (stats.missing_tables); the library logs an error and degrades.
FLAG_TABLES: Dict[str, str] = {}
DEFAULT_PROFILE = "production"

#: Retired environment variables that used to set solver fields.
RETIRED_ENV = {
    "JACOBIK_SCORE_AGG": "score_agg", "JACOBIK_RANK_AGG": "rank_agg", "JACOBIK_GMM_FLOOR": "solve_floor",
    "JACOBIK_RANK_FLOOR": "rank_floor_fit", "JACOBIK_Q1_TAIL_SIGMA": "q1_tail_sigma", "JACOBIK_Q1_WALL": "q1_wall",
    "JACOBIK_Q5_FOLD": "q5_fold", "JACOBIK_Q6_MODEL": "q6_model", "JACOBIK_Q6_AGG": None,
    "JACOBIK_Q6_PMF_WELLS": None, "JACOBIK_HAPTIC_PER_SUB": "haptic_per_sub", "JACOBIK_EMBED": "embed",
    "JACOBIK_CHELATE_EZ": "chelate_ez", "JACOBIK_CARBENE_YLIDE": "carbene_ylide", "JACOBIK_ETA2_CAP": "eta2_cap",
    "JACOBIK_BRANCH_BENDS": "branch_bends", "JACOBIK_BRANCH_PRIOR": "branch_prior",
    "JACOBIK_RING_OPEN_ADJ": "ring_open_adj", "JACOBIK_RING_JOINTS": "ring_joints", "JACOBIK_DONOR_BENDS": "donor_bends",
    "JACOBIK_HAPTIC_BENDS": "haptic_bends", "JACOBIK_HAPTIC_FREE_LEAN_MAX": "haptic_free_lean_max",
    "JACOBIK_HAPTIC_LEAN_WALL": "haptic_lean_wall", "JACOBIK_LOCAL_PLANE_DEG": "local_plane_deg",
    "JACOBIK_OUT_DEDUP": "out_dedup", "JACOBIK_DEDUP_RMSD": "dedup_rmsd", "JACOBIK_OUT_WINDOW": "out_window",
    "JACOBIK_Q6_WCA_N": None, "JACOBIK_Q6_WCA_T": None, "JACOBIK_SHAPE_CORE": None, "JACOBIK_SHAPE_TAIL": None,
    "JACOBIK_SHAPE_TAIL_FORM": None, "JACOBIK_SHAPE_TERMS": None, "JACOBIK_Q5_TEMPER": None, "JACOBIK_RING_LEAN": None,
    "JACOBIK_RING_KEEP_FOLDS": None, "JACOBIK_CARBON_STRUCTURAL": None, "JACOBIK_RESONANCE_LABELS": None,
    "JACOBIK_HAPTIC_V2": None, "JACOBIK_SHAPE_LABELS": None, "JACOBIK_Q6_METAL_EXCL": None,
    "JACOBIK_Q6_THROUGH_METAL": None,
}


def _norm(v: Any) -> Any:
    """JSON has no tuples: compare a list override with a tuple setting as equal."""
    return list(v) if isinstance(v, tuple) else v


def _coerce(name: str, v: Any) -> Any:
    if name in FIELDS:
        ref = FIELDS[name][2]
        if isinstance(ref, bool):
            return v if isinstance(v, bool) else str(v).lower() not in ("0", "", "false", "no", "off")
        if isinstance(ref, int) and not isinstance(ref, bool):
            return int(v)
        if isinstance(ref, float):
            return float(v)
    return v


def _load() -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    stale = sorted(k for k in RETIRED_ENV if k in os.environ)
    if stale:
        hint = {k: RETIRED_ENV[k] for k in stale}
        raise RuntimeError(f"retired environment switches set: {stale}. Use JACOBIK_PROFILE_OVERRIDES {hint}")
    name = os.environ.get("JACOBIK_PROFILE", DEFAULT_PROFILE)
    if name not in PROFILES:
        raise ValueError(f"JACOBIK_PROFILE={name!r} unknown; profiles: {sorted(PROFILES)}")
    raw = os.environ.get("JACOBIK_PROFILE_OVERRIDES", "").strip()
    over = json.loads(raw) if raw else {}
    bad = sorted(k for k in over if k not in FIELDS and k not in PRODUCTION_CONFIG)
    if bad:
        raise ValueError(f"JACOBIK_PROFILE_OVERRIDES: unknown fields {bad}; registered experiment fields: {sorted(FIELDS)}")
    frozen = sorted(k for k in over if k in PRODUCTION_CONFIG and k not in FIELDS and _norm(over[k]) != _norm(PRODUCTION_CONFIG[k]))
    if frozen:
        raise ValueError(f"JACOBIK_PROFILE_OVERRIDES: {frozen} are hardcoded production settings; overriding them would "
                         f"change the fingerprint but not the solver. Register an experiment flag in FIELDS instead.")
    vals = {k: v[2] for k, v in FIELDS.items()}
    vals.update(PROFILES[name])
    vals.update({k: _coerce(k, v) for k, v in over.items()})
    return name, vals, over


ACTIVE_NAME, _VALUES, OVERRIDES = _load()


def get(name: str) -> Any:
    """The value of a field under the active profile (+ overrides), or the production default."""
    if name in _VALUES:
        return _VALUES[name]
    if name in FIELDS:
        return FIELDS[name][2]
    if name in PRODUCTION_CONFIG:
        return PRODUCTION_CONFIG[name]
    raise KeyError(f"unknown profile field {name!r}")


def snapshot() -> Dict[str, Any]:
    """Every field as the code will use it NOW, the profile name and a fingerprint."""
    live = dict(PRODUCTION_CONFIG)
    for k, (mod, attr, default_val, _) in FIELDS.items():
        try:
            live[k] = getattr(importlib.import_module(mod), attr)
        except Exception:
            live[k] = _VALUES.get(k, default_val)
    for k, v in OVERRIDES.items():
        live[k] = v

    differs = {k: live[k] for k in FIELDS if live.get(k) != FIELDS[k][2]}
    return dict(profile=ACTIVE_NAME, fingerprint=fingerprint(live), differs_from_profile=differs, fields=live)


def fingerprint(live: Optional[Dict[str, Any]] = None) -> str:
    """Identity of a solver configuration.
    
    With no experimental fields registered or modified, returns the production
    fingerprint: 1e4cde06304f (v1.0b, superseding 2c87e2940af2).
    Adding a test flag to FIELDS or setting non-baseline values returns a distinct SHA-1 fingerprint.
    """
    if live is None:
        live = snapshot()["fields"]
    # Check if any registered experiment field differs from baseline
    extra = {}
    for k in FIELDS:
        if k in live and live[k] != FIELDS[k][2]:
            extra[k] = live[k]
    # Check if any overrides differ from production config
    for k, v in live.items():
        if k in FIELDS:
            continue
        if k in PRODUCTION_CONFIG and v != PRODUCTION_CONFIG[k]:
            extra[k] = v
        elif k not in PRODUCTION_CONFIG:
            extra[k] = v
    if not extra:
        return PRODUCTION_FINGERPRINT
    combined = {**PRODUCTION_CONFIG, **extra}
    if combined.get("finish_phase") is None:
        # Before IA-k was promoted, null was the absent experiment default.
        # Preserve the historical identity of C1 and its explicit overrides.
        combined.pop("finish_phase", None)
    return hashlib.sha1(json.dumps(combined, sort_keys=True, default=str).encode()).hexdigest()[:12]
