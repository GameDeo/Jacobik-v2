  AUDIT OF STEPS 0–1 — Step 1 is NOT certified. Do not start Step 2 until items A1–A8 pass.
  Several claims in your Step 0–1 summary contradict the artifacts. Correct the record in the
  reports; do not just re-run.

  Independent audit scripts (read-only; reproduce my numbers):
    /tmp/claude-1000/-mnt-e-Dev-Projects-jacobik-2026-10-02-full/4ccc2848-98c4-4a71-bdf1-5c468392b230/scratchpad/audit_recount.py
    /tmp/claude-1000/-mnt-e-Dev-Projects-jacobik-2026-10-02-full/4ccc2848-98c4-4a71-bdf1-5c468392b230/scratchpad/audit_common.py

  == CORRECTIONS TO THE RECORD ==
  1. Repeatability was run on seed 11 only (--quick; evaluated_seeds_count=1), not seeds 11–13.
     For HESBAM, SUDMEM, KOTJIP and HAMYUP, Run B ran in the same process as Run A, with the
     _GFN2_CACHE embed cache warm, so it was not an independent repeat.
  2. The compaction numbers quoted (Rg 1.052, range 0.974–1.317, contacts 1.396) do not match
     the file (1.070 / min 0.898 ICEYAQ / 1.51), and the file's values are invalid anyway (A2).
  3. The activated_ch_acceptor tightening was not justified. Under the roadmap sp/sp² rule
     (deg ≤ 3), on the same 172 dev complexes with the authoritative enumerator: median 51.5,
     p95 196, max 552 → PASSES. The report's "previous_max 446" and "previous_p95 > 300" are
     hard-coded literals (446 is SUDMEM's ch_ring count). OMAXUV is 231 under the roadmap rule
     and 171 under the narrowed rule; the 339→171 change mixes the enumerator change with the
     rule change.
  4. "< 0.35% unsupported B-splines" is an observation fraction, not a basis fraction.
     65% of active basis functions are unsupported, and ring_ring is 18% unsupported.
     The support analysis is invalid anyway (B1).
  5. The enumerator is not declarative. It checks only class NAMES from spec.json; elements and
     activation criteria are hard-coded. Bond order is ≥1.4 in code vs 1.5 in spec.
     allow_anionic_carbon_acceptor is not implemented.
  6. The common-ligand list is not frozen. spec.json/frame.json hold only a count and
     hand-written examples, no graph keys.
  7. The pilot was replaced after solves had run. The original pilot included KETFIB
     (unmappable) plus RUWSEJ and RETMIO; their pools remain in pools/. The frame was rewritten
     at 22:37 while the "final" pool runs spanned 22:21–22:46. The KETFIB failure is not reported.

  == MUST FIX BEFORE STEP 2 ==
  A1 Repeatability: run seeds 11, 12, 13 twice each on the 6 parents, with Run A and Run B in
     SEPARATE processes and a cold embed cache. Recompute the tolerances. Record that
     deterministic_embed=True is a research-arm setting.
  A2 Atom maps: generated pools use a different atom order from the reference for ALL parents.
     compute_heavy_atom_rg and compute_nonbonded_contacts apply reference indices to generated
     coordinates. Build graph-constrained generated→reference atom maps (a Step 2 deliverable
     anyway); flag ambiguous maps. Recompute Rg and contacts for top-1 and the pool. Also fix the
     Phase-3 MACE timing call and the NCI grad-norm measurement, which pair reference symbols and
     graph with generated coordinates. Re-examine ICEYAQ (top-1 +830 kJ/mol): is it a real
     generation failure or a state/charge problem? Report it as an outcome either way.
  A3 Admission spec:
     (a) Restore the roadmap sp/sp² activated_ch_acceptor rule (it passes), or justify the
         narrowing with measured numbers, and record which spec version produced every count.
     (b) Make enumerate_opportunities read element sets, activation criteria and bond-order
         thresholds from spec.json.
     (c) Reconcile 1.4 vs 1.5; implement or remove allow_anionic_carbon_acceptor.
     (d) Context keys: the level-1 root currently uses whole-complex integer charge. Implement
         the spec's level-1 descriptors (endpoint element/charge class, bond-order class, ring
         composition, binding role, intra/inter flag, has_c_neighbors) or amend the spec
         explicitly.
     (e) Delete the stale ad-hoc counter evaluate_candidate_opportunities in nci_v2_frame.py.
     Bump the spec version and record its sha256.
  A4 Common-ligand exemption: frequency ">50 sibling families" exempts 244 ligands, of which 101
     have ≥13 heavy atoms and 47 have >20 (C44N4 porphyrin-type, C48N4, C44P2, C41P3, B1C30N6
     Tp-type, C27N2/C29N2 …). The threshold is a cliff: 52 exempt at 51–60 families vs 59
     grouped at 41–50. Replace it with an explicit frozen list of small, common ligands stored as
     exact graph keys (PPh3, Cp/Cp*, py, THF, COD, acac, bpy, dppe-type, etc.). Keep large and
     polydentate scaffolds (macrocycles, Tp, tri/tetraphosphines) in scaffold grouping unless
     that recreates the giant component; if it does, document the trade-off. Call these "common
     ligands exempt from scaffold grouping", not "spectators"; they remain fully modeled.
     Report, per split and vs benchmarks, the counts and fractions of exempt-ligand overlap
     (currently 15/40 cal and 15/40 val representatives share an exempt ligand with fit; 12
     exempt ligands in cal/val also occur in benchmarks). Add one INDEPENDENT leakage check
     using a different canonicalization (e.g. RDKit canonical SMILES/InChIKey of ligand
     fragments, or exact isomorphism), not the same WL hash used to build the groups.
  A5 Unmappable inputs: 43/288 main-allocation representatives fail ligand_inputs_from_mol
     (25/160 fit, 7/40 cal, 5/40 val, 6/48 reserve). Before any Step 2 solve, predeclare in
     frame.json: failures are recorded as outcomes; reserve replacement applies only to training
     measurements, within stratum and in predeclared order; evaluation cohorts never replace.
     Re-check that each split still meets the Gate 1 minimums (≥10 informative cal and val
     groups per activated family). Report solver-mappability coverage as a population
     narrowing.
  A6 Provenance: every report must carry the sha256 of spec, frame, code (git hash + dirty
     diff hash), constraints DB and MACE model, plus the full resolved profile. Pools must carry
     the same. Add registry records for pilot_v1 runs.
  A7 Reproducibility: move evaluate_pilot_scaling_and_support.py out of the agent scratch dir
     into the repo (refit/ or a v2-pilot subcommand). Remove every hard-coded result or verdict
     string ("previous_*", "PASS: …", "fully_reconciled…"); verdicts must be computed. Fix the
     mislabelled "max_single_class_median_opf" (it is a max per-complex total).
  A8 Re-freeze the frame and pilot, THEN regenerate the scaling, repeatability and budget
     reports from the frozen artifacts, in that order. Nothing produced before the freeze
     counts as final.

  == FIX BEFORE STEP 3 (support/fit) ==
  B1 Redo the joint-support feasibility analysis:
     - Count only rows inside the active radial window (r < r_off; also report r < r_on). Today
       45% of bond_ring rows are at r ≥ 8 Å and 80% at ≥ 6 Å, and they are clamped into the last
       radial basis function. Only 6.5% of bond_ring and 5.8% of ring_ring rows are < 4.5 Å.
     - Use the roadmap weights (w_p/O_pf) and independent-group masses, plus the ESS criterion.
     - Report per ROOT CONTEXT, not per family. Distinguish insufficient context data from
       angular fragmentation (check 1D/2D marginal group counts ≥ 20, per the §7.1 rule).
     - Fix the angular domains: (n_R·n_S)² and μ₂ live in [0,1], not [−1,1].
     - Label it explicitly as reference-only preliminary feasibility; matched support requires
       baseline poses. Choose the 3- vs 2-interval resolution from this corrected analysis,
       on fit/dev data only.
  B2 fit_density_ratio hard-codes is_H=True; heavy-atom bond_atom classes must use (1.80, 2.20,
     6.0, 8.0).
  B3 Symmetric bonds: runtime averages both directions, but features_torch (extraction) uses
     one. Make extraction match: either both directions as half-weight rows or the same
     averaging, through ONE shared feature implementation. Runtime currently re-implements the
     features. Replace the flag-only test with a real parity test.
  B4 Extract features in float64 (the scratch script used float32).

  == FIX BEFORE ANY CANDIDATE SOLVE (Step 4 motions / Step 6) ==
  C1 _safe_get_surface falls back to the FIRST surface for an unknown context key. This must
     instead return zero for an unknown root, pool to a compatible ancestor for a known root,
     and fail loudly on schema errors.
  C2 Runtime cost: per-occurrence Python loop with per-occurrence spline evaluation takes
     ~3 s/batch-16 and ~12 s for QOCKUO (864 occurrences), measured with a mock radial-only
     surface. Vectorize by surface/context. Add an exact neighbor list (permitted by §3.2 because
     the outer taper gives exact compact support), since most enumerated occurrences are beyond
     r_off. Measure first-order gradient cost per IK step with a realistic 4D-support surface,
     separately from the HVP, and project it against the Gate 5 limit (mean ≤1.20×,
     p95 ≤1.25× incumbent).

  == BUDGET REPORT ==
  D1 Replace the assumed constants (fit 0.5 h, +0.25 h, ×1.5 RAM, range offsets, ~25
     poses/parent) with measured values or explicitly labelled assumptions. Include forces at
     scan centers, wider scans, reference + baseline (+ early/mid) centers, the reserve, and the
     15% failure rate. Report the p95 of the TOTAL, not 720 × the per-solve p95. Base solve
     timings on all 3 seeds.

  == SCOPE NOTE ==
  None of this changes the roadmap architecture. These are conformance and implementation
  fixes plus three spec clarifications (active-window support counting, context root keys,
  common-ligand definition). Report each fix with before/after numbers and the spec version
  that produced them.

  If you want something lighter, A1–A8 are the Step 2 gate; B, C and D can be issued later.
