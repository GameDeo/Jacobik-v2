# D2 causal investigation — 2026-11-10 checkpoint

**LUQVOK is a training donor-selection failure amplified by scoring and search.**
The new O-trans-O table has a 2.6575 Å component absent from the control table.
All of its long-distance observations come from 11 parents where T1d removes
the false haptic face and leaves a weaker, farther oxygen as the sole donor,
omitting the stronger oxygen on the same carboxylate. The target for LUQVOK
itself has no detected mismatch. LIPDEU has a separate orientation-fit issue.
Several other regressions depend on which rows are returned rather than a
preference reversal on fixed rows.

## LUQVOK source observations and generated geometries

The active resolved Q1 key is
`('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')`. It is
assignment-specific. The broader key with a 0.24 → 0.33 weight change from
the handoff does not explain this assignment.

| Fit | Independent parents | Component means (Å) | Weights |
|---|---:|---|---|
| C | 60 | 2.00979, 2.07165 | 0.25214, 0.74786 |
| D2 | 71 | 2.05605, 2.65748 | 0.84507, 0.15493 |
| Diagnostic: use nearer oxygens for the 11 additions | 71 | 2.01094, 2.07458 | 0.23402, 0.76598 |

The archived C observations and hash-matched D2 raw observations reconstruct
both model tables exactly on this workstation. No charge or locked exclusion
removed an observation from this key. Adding the 22 new D2 rows to C
reproduces D2, and removing them from D2 reproduces C. Both keys exceed the
50-independent-parent threshold, so this fit does not inherit a sampled
ancestor. In 64 parent bootstrap refits of D2 the long component appeared
40 times. Its 11 supporting parents are close to the ten-effective-parent
component threshold.

The 11 codes are BOPSAD, CEQHIR, FIFJIO, GORTEP, MAKJAM, MAKJEQ, NATBOZ,
REYKUF, TEHWUY, VIHKEE and WESZEA. All are octahedral under the old typing
and square planar under T1d. For both carboxylates in each, T1d keeps the
2.633–2.709 Å O (BO 0.156–0.196), while the same carbon's other O at
2.017–2.095 Å has BO 0.491–0.536. The weak contact may be chemically
meaningful as a secondary contact. Retaining it as the sole primary κO
observation omits the closer O. The complete κ versus asymmetric κ² target
remains uncertain for some training parents. The near-O fit tests the
effect of substituting those 22 distances.

For the saved seed-11 top structures in a common C runtime, C's statistical
score favours the short-contact row by 4.797 normalized log-score units.
D2 shrinks that to 0.0013. Restoring just the active O-trans-O table in D2
restores 4.507 units of margin. Restoring the active C-trans-C table adds
0.289, for a total Q1 shift of 4.795. O-table mean/variance restoration has
the large effect. A component-weight swap ordered by means cannot match
the new long component to an old long component, since none exists.

The long geometry still pays a Q1 penalty under D2 (0.228 units in this
comparison), but its QO orientation, Q5 and Q6 advantages nearly cancel
that penalty. In native D with physical Q6, the D top row outscores the
C top row statistically by 0.029, 0.028 and 0.055 across seeds. Under the
common C runtime its corresponding margins are −0.001, −0.015 and +0.008.
The rigid outward translation of each carboxylate-bearing ligand shows a
second high-score region near +0.6 Å under D2 (−0.664 vs −0.596 at the
start), while C drops from −0.589 to −5.416 along the same coordinate
scan. The long component also passes
the solver's 10%-of-dominant-weight threshold for sampling start distances.
The saved records do not establish which initialization or pruning step
determined a particular row.

All 32 seed-11 D rows have long Pd–O contacts. Seeds 12 and 13 include
shorter rows. In those
seeds, C scoring on D's returned rows improves the selected energies by
132 and 129 kJ/mol, yet D's best-energy rows still trail C's by about
100 kJ/mol. DQ6 still has +208.5 kJ/mol mean top-1 and +130.7 floor
regressions even though its long-bond top rows lose to the C top rows on
the common D2 score. A better scoring preference and access to a better
pool both matter.

## Other fixed-pool cases

All 81 native arm/seed comparisons reproduced the saved top and floor
scores after subtracting the recorded bend prior (maximum absolute
residual 5.60e-6, tolerance 5e-4). This covers C, D and DQ6 for nine
parents at seeds 11–13, including both saved E/Z variants of YABYEJ.
Manifest and XYZ row identities, DB hashes, source coordinates and the
native shell/assignment contexts were checked. Recorded C/D fingerprints
can reflect S/DS flags because the bench snapshots live module attributes
before resetting the next parent's first-arm flags. Explicit flags with
fresh imports reconstruct 6b47c3fb45a9 / 302a15e37622, and native scores
match, corroborating the score contexts. The historical trajectory
would require separate replay evidence. The added ref_ll/ref_sd fields
are fit metadata here.

| Parent | D top-1 / floor Δ vs C (kJ/mol) | Reading |
|---|---:|---|
| LUQVOK | +208.3 / +133.3 | New Q1 basin changes the ranking and the available pool. |
| SESWIZ | +62.6 / +0.5 | Shared ranking inversion: assignment 4 beats lower-energy assignments by 0.07–0.10 score units while costing 189–198 kJ/mol, including in C seed 11. Q1 and Q5 gains offset the Q2 penalty. Changing DB alone does not change the selected row within D/DQ6. The regression is how often this basin wins, not a demonstrated new table-induced reversal. |
| LIPDEU | +60.1 / +29.5 | QO reverses C-vs-D top preference in all seeds. C rescoring recovers 19–32 kJ/mol within D pools, whose floor remains higher. |
| YABYEJ | +49.0 / +49.5 | Both DBs prefer the C top and select the same row within D. All top/floor rows share E/Z variant 0 and assignment 0, so the better basin is missing within that variant. Seed-11 Q5 margin shift is only about −0.0068, mainly C_ar–C_sp3–C_ar–C_sp3 free_o1. Trace the variant-0 starts and optimization before changing fits. |
| LOPJEF | +56.3 / −3.8 | D finds low-energy rows. C rescoring recovers 55/11/0 kJ/mol across seeds. Ti N_pyridine distances and C_eta5/O_alkoxide angles contribute. Both DBs prefer the C top in the matched comparison. |
| ZAQNUE | +38.0 / +33.7 | Both DBs prefer the C top and select the same row within D. The D pool misses the low-energy C basin. |
| HOXJUZ | −81.4 / −88.2 | D2 reverses preference toward the better-energy row. The Ni S_thiocarbonyl trans-N Q1 lookup dominates the seed-11 change. |
| USAPOU | −51.7 / −33.5 | Ru haptic Q1 trans-N_pyridine and trans-Cl keys dominate D2's preference shift towards the improved structure in seed 11. |
| UMUBAF | −82.5 / −68.7 | Input audit confirms false η². C/D shell signatures differ, so no same-target cross-arm attribution is made. |

The complete row and lookup evidence is retained, including marginal
single-key substitutions. A missing good row does not by itself locate the
failure in initialization, optimization or pruning. In particular, SESWIZ
is not explained by the LUQVOK Q1 mechanism.

For **LIPDEU**, the requested P_donor/Pt square-planar orientation with two
substituents and a four-membered ring resolves to the all-donor `QO:(2, 4)`
fallback. Restoring this one fit changes the seed-11 top-row margin by
−0.538, accounting for essentially all of the +0.538 QO margin shift.
Refitting the archived and hash-matched D2 groups reproduces both fits.
The group grows from 3,655 to 5,742 rows: N_amidinate goes from 994 to
2,806, while P_donor remains 46. The differently typed population is
already large with locked exclusion alone (5,869 rows). Applying D2's
charge filter to those typed rows changes the mixture from two to three
components. Applying both filters to the old typed rows changes its
three-component fit to four. Thus both population and fit selection are
sensitive. This does not prove which new N observations are invalid,
nor does it justify removing rare P geometries.

## Process and gate input evidence

The archived job 61051826 production reproduction has zero differing,
missing or extra keys for Q1–Q4, Q6 and QO. Its documented Q5, Q6P and NCI
exceptions remain. There is no reason to rerun a broad reproduction job
before repairing the now-identified typing and fitting mechanisms.

The actual Q1 ancestor stream was replayed with D2 typing and fixed locked
exclusions, changing only the charge filter. Of 476 watched ancestor
buckets (fallback ladders for the LUQVOK O/C keys), 250 have identical
eligible rows, and **73 of these change sampled membership**, with a
maximum of 482/500 replaced. This confirms collateral resampling on the
real training data. It does not cause the independent O-trans-O fit above,
and its contribution to other changed fits is still to be measured.

The independent audit on `investigate/gate-inputs` (f18ec0c) completed 230
dispositions and 34/34 static corrected-target validations. It found
170 with no discrepancy detected, 23 ambiguous and three confirmed
errors without a defensible repair. All six named losses have no η²
sites under either typing and no detected target mismatch. The audit
includes the exact input changes and atom-level reasoning. It also finds true short O donors
lost by some false-face corrections, and a false η³ case that T1d misses.

Descriptively, on the 99 primary stage-1 cases with no input discrepancy
detected, D is +3.06 [−2.22, +9.32] and DS +3.05 [−1.10, +7.50] kJ/mol
top-1, with 10,000 parent bootstrap draws (seed 11). Original results
remain +2.28 and +2.11 respectively, and the historical verdict remains
failed. The audit does not rescue D/DS. No modified-target stage-2 gate
or promotion-reserve calculation was done.

## Next proposed package

1. **Fix donor identity before refitting.** Reconcile mapped SMILES donors
   against the geometry within the same chemically equivalent donor group
   before false-face removal, and retain distinct source-supported donors.
   Differentiate clear primary O selection, real κ² and unresolved
   secondary coordination. Put the shared rule in the extraction/input
   pipeline, with positive and negative fixtures (11 Pd cases, true
   κ² controls, DUCPAT-type lost O and asymmetric ambiguous cases).
   Preserve raw source and log observation-level corrections. Test whether
   the corrected key and small same-machine paired runs recover LUQVOK
   without sacrificing the Q1-driven improvements.
2. **Decouple sampling and audit fit stability.** Give each ancestor bucket
   a stable selection keyed by observation identity so unrelated filtering
   cannot change its membership. Treat sampler, typing and charge policy as
   separate comparisons with locked exclusions fixed. Audit what the
   formal-charge rule removes for each quantity. Preserve valid local
   measurements where the suspect encoding is unrelated.
3. **Address sparse orientation separately.** Review the newly contributing
   ring-4 observations, then test chemistry-conditioned partial pooling
   for the P orientation rather than trusting the all-donor fallback.
   Check candidate score margins, local sensitivity and parent-level
   regression tails before a full build. Instrument start/beam/prune
   traces for unresolved SESWIZ, YABYEJ and ZAQNUE before changing search.
4. **Version the fair gate.** Use the audited overlays for both arms,
   hash targets into cache identities, and report original, validated and
   unresolved strata separately. Any changed target that invalidates its
   reference requires a new versioned reference. Freeze the next contract
   before evaluating it and retain the same-machine control and guard rules.

Changes are limited to diagnostics and benchmark fixtures. Billed compute
and Colab spend were zero.

## Reproduction and progress

- 2026-11-10: isolated branch from the historical D3 runtime (4982024).
  Implemented CPU diagnostic dispatch and frozen pool/typing contracts.
  All parent diagnostics used `tools.run` with one worker per task batch.
  Initial training audit pilot was stopped to keep within the process
  cap and superseded by the complete 22/22 training-source records.
- 2026-11-10: retrieved small retained Q1/QO observation parquet files,
  charge census and reproduction report from Perlmutter scratch. The
  Q1/QO files match the D2 build metadata hashes.
- 2026-11-10: completed 27/27 fixed-pool tasks, 22/22 training-source
  tasks, direct/near-O and orientation refits, reservoir replay and audit
  strata. Detailed outputs are under `research/data/audit/2026-11-10_DS/`
  in the main checkout. Important files: `cross_score_summary.json`,
  `luqvok_q1_direct_fit.json`, `luqvok_near_oxygen_counterfactual.json`,
  `lipdeu_orientation_fit.json`, `q1_ancestor_resampling.json`,
  and `stage1_audit_strata.json`. The saved manifest and per-parent records
  preserve exact input identities and lookup parameters. `MANIFEST.sha256`
  covers the transferred fit inputs, frozen manifests and main summaries.
- 2026-11-10: workstation fingerprint 6b47c3fb45a9 passed. All 490
  golden arrays match the archived reference for this historical checkout.
  The server restart interrupted report serialization. All 16 golden
  workers completed and their saved arrays passed the standard comparison
  in `ws_golden/recovered_comparison.json`.
- 2026-11-10: the first pytest worker exceeded the 4 GB RSS cap and
  was interrupted after 494 tests. Eight sequential batches completed
  the remaining 158 tests. The combined result covers all 652 cases
  with 633 passed, five skipped and 14 failed. The 14 failure IDs
  exactly match the prior D2 workstation run's CUDA replay failures.
  The workstation pytest stage remains failed. See
  `ws_pytest_resume/combined_result.json`.

For the fixed-pool run, generate the manifest with
`tools.jgen.ds.cross_score.manifest(main_checkout)` and pass it to
`python -m tools.run diagnostic` with
`--task tools.jgen.ds.cross_score:run --seconds-per-parent 35`.
Use `tools.jgen.ds.summarize.assemble(manifest_path, records_dir)` to
aggregate the records. The training-source manifest runs the same
diagnostic entry point with `--task tools.jgen.ds.fit_trace:run`.
The `direct_fit`, `orientation_fit` and `replay_samples` functions
in `fit_trace.py` take the main checkout and evidence directory and
reproduce the table-level diagnostics from the saved fit inputs.
