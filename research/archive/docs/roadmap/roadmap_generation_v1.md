# Jacobik Roadmap: Conformer Generation Workstreams (v1)

> **Date:** 2026-10-24 (revision 2, 2026-10-25: input audit, fold findings, never-fail reframing)
> **Status:** Proposed implementation and evaluation specification. Nothing in this document changes `src/` behaviour, the frozen NCI v2 control, or any artifact under `data/`. Diagnostic scripts are committed under `tools/jgen/diag/`.
> **Companion:** [roadmapv2.md](roadmapv2.md) (domain 1, transferable NCI, is the main pipeline and is not restated here). This document covers the other generation workstreams so that implementer agents can code them behind flags and run them later as one batch.
> **Objective:** Reduce the catastrophic energy tails, the seed dependence, and the failure modes of fresh conformer generation, with every change defined well enough to implement and test without interpretation.

## 0. Scope, evidence, and how to read this document

### 0.1 Status terms

- **Measured:** computed from stored artifacts by a script in `tools/jgen/diag/`. Each measured number carries a tag `[Dnn]` that maps to a command in Appendix A.
- **Cited:** taken from an existing report (`S2_1_REPORT.md`, `roadmapv2.md`, `research/archive/docs/HANDOFF.md`) and **not re-run** here. Cited items say so.
- **Hypothesis:** a mechanism consistent with the measured facts, with the code location where it would act, and the test that would refute it.
- **Proposal:** a code change with a flag, default, algorithm, tests, metric, and acceptance rule.
- **Conditional:** an extension that is justified only if a named measurement fails.

Numerical thresholds below are proposed defaults. They must be frozen in the evaluation contract before candidate results are opened (Section 1.4).

### 0.2 Frozen data used

All measurements use the Step 2 control pools, `data/refit/nci_v2/scaled_v1/`, with S2.1 repair overlays under `repair_s2_1/`. Pools are MACE-POLAR-1-S, float64, control arm, seeds 11, 12, 13, splits `calibration` (117 parents) and `motion_validation` (115 parents). That is 351 + 345 = **696 pools over 232 parents**. The shared `data/` tree is read-only for this work. The worktree has no `data/` directory, so every script takes `--pools` / `--manifest` as absolute paths into the main checkout.

Per-pool files used:

| File | Fields used |
|---|---|
| `pools/<CODE>_<seed>_control.npz` | `X` (rows × atoms × 3, generated order), `gen_to_ref_idx`, `ref_X`, `ref_symbols`, `score` (argmax = native top-1; the solve objective: equals `rank` plus the bend prior, see M-5), `rank` (the ranking total; `rank × Z` = sum of `terms_Q1..Q6`, see M-6), `asg` (assignment index), `coord_intact`, `terms_Q1..Q6` (already multiplied by Z), `num_automorphisms`, `gen_to_ref_idx`, `profile` |
| `pools/<CODE>_<seed>_control.mace64.json` | `energies_kj` (per row), `ref_energy_kj`, `top1_idx`, `floor_idx` |
| `pools/<CODE>_<seed>_control.json` | `stage_*_s`, `wall_s`, `n_confs`, `top1_dE_kj`, `floor_dE_kj`, `top1_coord_intact`, `floor_coord_intact` |
| `s2b_eval_state_v1/pools/*.npz` | `thetas_flat` (per pose), `bend_columns`, `bend_k`, `bend_prior_solver` (S2b, per-pose solver state; calibration and validation units only) |
| `repair_s2_1/repair_manifest.json` | `pools[key].overlay_path` (relative to `scaled_v1/`); overlays with `floor_dE_kj` are the R1-repaired pools |

**Per-row energy.** Per-row dE is `energies_kj − ref_energy_kj` from the mace64 sidecar. The pool `.npz` does not store per-row energies. Top-1 is `argmax(score)`; in every pool examined, that is row 0 of the stored array (the pools are stored in score order). The floor is `argmin(dE)`.

### 0.3 Definitions used throughout

| Term | Definition |
|---|---|
| `top1_dE` | MACE energy of the native top-1 row minus the reference energy, kJ/mol (`[D01]` aggregates these from the stored summaries). |
| `floor_dE` | Minimum dE over the stored pool rows (finite energies only). |
| `regret` | `top1_dE − floor_dE` ≥ 0. Any selection-only change is bounded above by the mean regret. |
| `seed range` | max − min of `top1_dE` over the three seeds of one parent. |
| `intact_ref` | The stored `coord_intact` flag from `coordination_integrity` (`refit/nci_v2_pilot.py:267`, tolerance `COORD_SHIFT_TOL_A = 0.30` Å, bound scale 1.15). **Uses the reference geometry. Evaluation only.** |
| `intact_site` | The reference-free production analogue defined in Section 1.5. It is what a generator may use. |
| `assignment class` | The integer `asg` stored per row (the solver's candidate/assignment index). Section 6.1 gives the operational definition. |
| R1 overlay | An S2.1 repair overlay carrying corrected `floor_dE_kj`/`top1_dE_kj`. Headline scoreboards use overlays where present. The lottery analysis excludes parents with R1 overlays (12 parents), giving 220 parents. Both conventions are used and labelled. |

### 0.4 Effective pools and reproduction

All analyses read **effective** pools: the raw pool with the S2.1 overlay applied where one exists (`tools/jgen/diag/poolio.py`). R1 (36 pools, 12 parents) carries recomputed MACE energies; R2 (237 pools) carries `gen_to_ref_idx` and the coordination arrays; R3 (all pools) carries health fields only. Earlier passes read raw arrays for energies and coordination; those numbers are superseded. Scripts that touch only the tail and failure parents, which carry R3 only, read raw arrays, identical to effective ones (asserted by the repair manifest).

Every `[Dnn]` tag is produced by a committed script. Appendix A gives the exact command and the reported values. All scripts run CPU-only with `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, and the pixi Python at `.pixi/envs/linux/bin/python`.

## 1. Workstream 0: separate module, profile gating, and bench

### 1.1 Package and layout

All new experiment code lives in `tools/jgen/`, parallel to `tools/jdev/` (read [tools/jdev/README.md](tools/jdev/README.md) and [tools/INVENTORY.md](tools/INVENTORY.md) first). The package name is `jgen` (generation). Layout:

```text
tools/jgen/
  __init__.py
  __main__.py           CLI: python -m tools.jgen {diag, bench, fingerprint-check}
  bench/
    panels.py           panel definitions (Section 1.4.2), reads manifest + repair manifest
    metrics.py          e, f, r, paired change, bootstrap (roadmapv2 section 5.2 exactly)
    simple.py           simple reference-baseline mode (Section 1.4.3)
    versioned.py        versioned mode with control-vs-control repeat (Section 1.4.4); not built yet
    outputs.py          JSON + Markdown writers, failure ledger
  diag/                 committed diagnostic scripts; each prints JSON or plain text; no side effects
    d01_pool_scoreboard.py  ... d09_intact_filter_oracle.py   (this document's measurements)
  site.py               reference-free site test (Section 1.5)
tests/jgen/             unit tests (Section 1.2.4 and per-proposal tests)
```

The solver is not forked. Generation changes are profile fields (Section 1.2). The bench calls the existing orchestrator `refit/nci_v2_orchestrator.py` with `JACOBIK_PROFILE_OVERRIDES` set, so arms differ only by the profile JSON.

### 1.2 Profile gating

#### 1.2.1 Mechanism (measured from source)

`src/jacobik/profile.py` declares `FIELDS: field -> (module, attribute, legacy_value, meaning)` (`profile.py:25`). `PROFILES["u1"]` (`profile.py:85-97`) is the incumbent. `JACOBIK_PROFILE_OVERRIDES` (JSON) overrides single fields (`profile.py:138-158`). Each module reads its value at import through `get(name)`.

**Fingerprint hazard (measured).** `snapshot()` hashes **every** field of `FIELDS` (`profile.py:166-177`, `sha1(json.dumps(live, sort_keys=True))[:12]`). Adding any new field, even at its legacy value, changes the fingerprint. The Step 2 pools record `fingerprint 485e943af036` for `u1` (`profile` string in every NPZ). Adding flags without a fix would make the incumbent's identity look different while its results are unchanged.

#### 1.2.2 Required change (proposal P0)

In `snapshot()`, hash only fields whose live value differs from the legacy value **or** that belong to a frozen list recorded in the pool, and always record the complete live dict in `fields` for provenance:

```text
live = {k: current module attribute}
core = {k: v for k, v in live.items() if k in LEGACY_HASHED_FIELDS}      # the 2026-10 set, frozen
extra = {k: v for k, v in live.items() if k not in LEGACY_HASHED_FIELDS and v != FIELDS[k][2]}
fp = sha1(json.dumps({**core, **extra}, sort_keys=True))[:12]
```

`LEGACY_HASHED_FIELDS` is the exact set of fields present in `FIELDS` before this change. Under `u1`, every new field sits at its legacy value, so `extra == {}` and the fingerprint stays `485e943af036`.

**Test T-P0 (required before any flag is merged).** `snapshot()["fingerprint"] == "485e943af036"` under `JACOBIK_PROFILE` unset, and under `JACOBIK_PROFILE=u1` with no overrides. Also: setting one new flag to a non-legacy value changes the fingerprint.

#### 1.2.3 Flags proposed in this document

Every new flag is default OFF at its legacy value, so `u1` results are bit-unchanged. Existing fields used as experimental arms need no code change and are listed after the table.

| Flag (profile field) | Legacy value (default) | Module attribute | Workstream | Meaning |
|---|---|---|---|---|
| `rank_site_intact` | `False` | `jacobik.ik.ik_solver` `_RANK_SITE_INTACT` (new) | 1 (T2) | Choose the native top-1 by `score` among rows with `intact_site`, if any exist. |
| `ring_joint_prior` | `0.0` (weight; `0` = off) | `jacobik.ik.angle_bends` `RING_JOINT_PRIOR_WEIGHT` (new) | 1 (T3) | Q7-form prior on ring-internal angles of ring-joint rings, reference = embed value. Conditional on D-T3 step 3. |
| `beam_target_aware` | `False` | `jacobik.engine` `BEAM_TARGET_AWARE` (new) | 2 (S-P2) | Seed each candidate's torsion beam from its own target-placed geometry (Section 3.4). |
| `dedup_exact` | `True` | `jacobik.ik.dedup` (`select(..., exact=True)`, `dedup.py:335`) | 4 (P-D2) | Keep exact dedup. The non-exact branch (`dedup.py:346`) is a separate arm, never a silent change. |
| `dedup_aut_prefilter` | `0` (off) | `jacobik.ik.dedup` `AUT_PREFILTER` (new) | 4 (P-D2) | Exact lower-bound prefilter over automorphisms; output must be identical (Section 5.4). |
| `dedup_class_aware` | `False` | `jacobik.ik.dedup` `CLASS_AWARE` (new) | 5 (6.2) | Never merge rows whose `asg` differs. Only if measurement D1 is nonzero. |
| `embed_bond_check` | `False` | `jacobik.engine` `EMBED_BOND_CHECK` (new) | 3 (E-P1) | Flag and penalise (never reject) a free-ligand embed whose covalent bond lengths disagree with the bond-order table. The output is still produced. |
| `eta2_cap` | `False` (exists) | `jacobik.chelate_embed` `ETA2_CAP` | 3 (E-P3) | Existing field, tested here, not re-implemented. |

Existing fields used as arms, with no code change: `branch_prior` (`fk.py:65-67`; `u1` = `False`; the arm sets `True`, see T3), `out_window` (`engine.py:88`), `embed` and `deterministic_embed`.

The earlier proposal `bend_pivot_scope` is withdrawn (see T3). Execution-only switches do **not** become profile fields. Timing instrumentation (Section 5.2) is `JACOBIK_TIMING_TRACE=1`, which writes extra keys to the pool JSON and changes no result. The seed budget (Section 3.3) is a runner argument, not a solver field, but the bench records it in the experiment identity.

#### 1.2.4 Tests for gating (required)

- **T-G1** For every flag: default value equals legacy value; `snapshot()` fingerprint under `u1` is unchanged (T-P0).
- **T-G2** Matched-output test (Section 4.4 defines it once): with the flag at legacy value, `python tools/jdev/golden.py check` prints `golden: OK`, and the fixed CPU sample S20 (Section 4.4) reproduces its recorded outputs at the golden tolerances. The stored pools are **not** the reference, because the GPU path is not bit-reproducible (roadmapv2 section 9.8).
- **T-G3** Unknown field in `JACOBIK_PROFILE_OVERRIDES` raises (existing behaviour, `profile.py:150-151`); test it does not silently accept a misspelled new flag.

### 1.3 Command-line interface

```text
python -m tools.jgen diag <script-name> [script args]     # runs tools/jgen/diag/<script-name>.py
python -m tools.jgen fingerprint-check                   # asserts T-P0 under u1; exit code 0/1
python -m tools.jgen bench simple  --candidate DIR --reference DIR \
        --manifest FILE --repair FILE --panel {tail,cal,val,failures,all} --out DIR
python -m tools.jgen bench versioned --plan FILE.json            # later (Section 1.4.4)
```

All commands print their outputs under `--out` and never write under `data/`. `--reference` for the simple mode is the frozen Step 2 `scaled_v1/pools` directory.

### 1.4 Bench: panels, metrics, and modes

#### 1.4.1 Metrics (exactly roadmapv2 section 5.2)

For method m, parent p, seed s, reference energy `E_ref(R_p)` and native top-1 `X^top1_mps`:

```text
e_mps = E_ref(X^top1_mps) − E_ref(R_p)          (signed; stored as top1_dE)
f_mps = min over valid rows of [E_ref(X) − E_ref(R_p)]   (stored as floor_dE)
r_mps = e_mps − f_mps                           (regret ≥ 0)
d_p^(x) = (1/3) Σ_s (x_cand,ps − x_base,ps),  x ∈ {e, f, r}     (negative is favourable)
```

Confidence intervals resample **parents** (seeds kept together), 4,000 resamples, seed 0 (roadmapv2 section 5.2). Also reported: valid native top-1 share; top-1 within 4.184 kJ/mol of the valid floor; parent mean changes; number of parent mean losses > 10 kJ/mol; worst single-seed loss; seed range `max(e) − min(e)` (labelled distinctly from the historical `max − median`); tail counts at 100, 200, 500, 1000 kJ/mol; wall and peak memory by stage.

#### 1.4.2 Panels (frozen by manifest hash)

| Panel | Definition | Size |
|---|---|---|
| `tail` | The 10 parents with any `top1_dE > 500` in the stored pools: DAMVAQ, HOXJUZ, IFUROQ, JEGPUJ, NEPNEF, SOBHAS, USAPOU, XEYTUT, YUCZUS, ZELTAP. [D01] | 10 parents, 30 pools |
| `cal` | All `calibration` successes with pools present | 117 parents, 351 pools |
| `val` | All `motion_validation` successes with pools present | 115 parents, 345 pools |
| `failures` | Named failure cases from Section 2 that have pools, plus any named case the loader cannot find (reported as `not_in_cohort`, never silently dropped): IRAQUO, RELCOA, ACUYUV, NALJOZ, IFAPUD, KADCUN, LAWRAC, ROJMEJ, SUQWEG, NEPNEF, IWOLEN, UQUNUQ, VENGUR, ZELTAP, BODCAY, LIYTAQ, OBABAX, ADINOR | ≤ 18 parents |
| `all` | `cal` ∪ `val` | 232 parents, 696 pools |

Panels are defined by the manifest, not by a list of outcomes. A panel never removes a failed pool; failures stay in the denominator with a reason.

#### 1.4.3 Simple reference-baseline mode (decided: first)

**Baseline.** A candidate pool set is compared with the stored Step 2 control pools (effective, `poolio.py`). No new control run is needed for this mode.

**Noise band (measured, not assumed).** The S2b units (registry record 22, code commit f71f196) regenerated the calibration and motion-validation units under the same configuration, differing only by per-pose state capture. They therefore give a run-to-run repeat of the control arm on the same cohort. `[D12]` (`tools/jgen/diag/d12_s2b_repeatability.py`): 696 units compared; the 15 S2b-only pools (5 units) are excluded from pairing and reported; reference energies agree exactly (max difference 0.0).

| Quantity | median | p95 | > 1 kJ/mol | > 5 kJ/mol | > 10 kJ/mol | max |
|---|---:|---:|---:|---:|---:|---:|
| per-unit abs(Δ top-1), kJ/mol | 0.0002 | 0.263 | 25 | 14 | 7 | 164.4 (LEKXIK, seed 12) |
| per-unit abs(Δ floor), kJ/mol | 0.0002 | 0.301 | 27 | 15 | 7 | 60.2 |
| per-parent abs(d_p^(e)), kJ/mol (232 parents) | 0.000 | 0.648 | | | | |

**Rule.** The noise band is the per-parent 95th percentile of abs(d_p): **0.65 kJ/mol** for cal and val. A parent change above the band is a candidate effect. The unit-level tail is heavy: about 1% of units move by more than 10 kJ/mol between two identical-configuration runs. The simple mode therefore reports every unit above 10 kJ/mol next to its parent, and never reports a parent change as significant from one candidate run alone.

**Seed range is a different quantity.** The median seed range (9.4 kJ/mol over 232 parents, `[D01]`) is spread between seeds 11, 12 and 13, which are different stochastic draws by design. It measures search variability (Section 3), not run-to-run repeatability, and is not used as a noise band.

**What the simple mode can conclude.**
1. Descriptive paired e, f, r differences per parent, with the failure ledger (roadmapv2 section 5.2).
2. Whether a parent change exceeds the 0.65 kJ/mol band, and whether any unit exceeds 10 kJ/mol.
3. A failure screen: new invalid or empty outputs, or new parent mean losses above 10 kJ/mol.

**What it cannot conclude.** It cannot establish improvement. The band covers cal and val under the code of f71f196 and this profile only. A change to the generator changes the band, so every new generator identity needs its own band (Section 8.2). The simple mode prints `status = descriptive`.

**Pairing.** Pair by (parent code, seed). A pair is complete only if both pools exist with finite energies, and the candidate's `code_git_commit` and `frame_sha256` match the frozen arm values. A missing or invalid pool enters the failure ledger and is never dropped.

#### 1.4.4 Versioned mode (later; not built)

Required before any promotion claim, following roadmapv2 Gates 4-5:

1. An `evaluation_contract.json` written before any candidate result is opened.
2. The candidate arm run on the identical cohort with identical input hashes.
3. The noise band of Section 1.4.3 applied to every parent-level change. A **new** panel (for example the B4 reserve panel) needs its own control-vs-control repeat on that panel, because the repeat `[D12]` covers only cal and val.
4. The Gate rules of roadmapv2 section 5.3, with the thresholds written in the contract.
5. GPU timing and memory sweeps (Sections 5.1 and 5.5).

The mode is versioned by its contract. Changes after a result create a new contract.

### 1.5 Reference-free site test `intact_site` (proposal P1)

`intact_ref` (`refit/nci_v2_pilot.py:267`, tolerance `COORD_SHIFT_TOL_A = 0.30` Å) compares generated donor distances with the **reference** geometry. A generator cannot use the reference, so the production test needs its own definition.

**Groups.** Metal-donor groups follow `coordination_integrity`: a single bonded non-metal atom is a sigma donor; a connected set of bonded atoms is a haptic group.

**Target distance μ_i (exact rule).** Let `key` be the Q1 key the solver builds for the donor site (`engine.py`, `_beam_search_all_candidates`, tuple `(metal, spin, OS or OS_UNKNOWN, subclass, geometry, trans_p)`, where `trans_p` comes from the row's assignment). Let `dist = stats.dist("Q1", key)` (`StatsLookup.dist`, `src/jacobik/stats/stats.py`, around line 415; it dispatches to `_fallback_q1` at `stats.py:581`, which resolves the ladder `ladders.q1_candidates`). If `dist` has means and weights, **μ_i is the mean of the highest-weight component** (ties broken by the lowest index). If it has none, μ_i is the solver's table value `_BOND_LENGTH_TABLE[family]` (module table used by `_q1_sample_distances`, `engine.py:286-330`), where `family` is the donor label family. The solver's 0.05 Å sigma fallback is not used by `intact_site`.

**Two keys exist.** The scorer builds its key through `key_builder.q1_site_keys(mol)` (`src/jacobik/stats/key_builder.py:402`, used by `scorer.py:441`), which has no per-row `trans_p`. The solver's per-row key differs in `trans_p`. `intact_site` uses the **solver's per-row key**, because a row belongs to one assignment and `trans_p` is then defined. Whether the two keys differ on the stored pools must be checked before P1 is coded (a required step).

**Rule.**

```text
intact_site(row) = AND over sigma donors i:  |d(M,i) − μ_i| ≤ 0.30 Å
                   AND over haptic groups g: slip_g ≤ 0.30 Å and shift_g ≤ 0.30 Å
```

where `slip_g` and `shift_g` follow `coordination_integrity`, with μ_i replacing the reference distances.

**Validation (on the stored effective pools, once implemented).** Agreement with the stored `coord_intact` over all 696 pools (23,050 stored rows, `[D01]`). Acceptance: agreement ≥ 95% on rows where `intact_ref` is defined, with the disagreement pattern listed by parent. Below 95%, P1 is not used for ranking.

## 2. Workstream 1: catastrophic tails and robustness

### 2.0 Input audit and the never-fail principle (revision 2)

**Principle (binding).** The generator accepts impossible or nonsensical inputs and never fails. It never rejects an input, raises, or returns an empty result. A bad input is flagged, its score is penalised so that it is clearly terrible, and the best available structure is still output. Every proposal below that rejects, raises or drops is reframed this way.

**Input labels (d14, `tools/jgen/diag/d14_input_consistency.py`).** Each parent is labelled `consistent`, `inconsistent` or `ambiguous` from its SMILES and its reference geometry. The labels are analysis only and are never a generator rule. **Disclosure:** the charge rule was revised once (v3) after looking at named parents. All numbers here use v3. The v2 rule is not reported. Named labels: HOXJUZ, NEPNEF and XEYTUT inconsistent; ZELTAP and IFUROQ ambiguous; DAMVAQ, JEGPUJ, SOBHAS, USAPOU and YUCZUS consistent. LIYTAQ is ambiguous (`partial_alternation`). BODCAY and OBABAX are inconsistent.

**Prevalence (v3 labels).**

| Set | n | consistent | ambiguous | inconsistent |
|---|---|---|---|---|
| Parents with a successful pool (d14_v3_pools) | 232 | 174 (75.0%) | 43 (18.5%) | 15 (6.5%) |
| surface_fit split (d14_v3_surface_fit) | 1200 | 934 (77.8%) | 209 (17.4%) | 57 (4.8%) |

Surface-fit flag counts (parents carrying the flag): partial_alternation 147, bond_soft_mismatch 87, quinoid_written_aromatic 35, absurd_metal_charge 17, donor_typing 17, charge_imbalance_N 12, hypervalent_S 3, absurd_light_charge 1, bond_contradicts_order 1.

**Scoreboards split by label (d17_v3; three seeds per parent).**

| Label | parents | pools | top-1 mean (kJ/mol) | top-1 median | top-1 > 200 | top-1 > 500 | regret mean | seed range > 50 kJ/mol |
|---|---|---|---|---|---|---|---|---|
| consistent | 174 | 522 | 111.0 | 65.7 | 12.6% | 2.9% | 22.1 | 16 |
| ambiguous | 43 | 129 | 152.6 | 83.0 | 24.0% | 4.7% | 42.8 | 5 |
| inconsistent | 15 | 45 | 454.7 | 157.6 | 42.2% | 20.0% | 54.9 | 2 |

Quality claims in this document are made on the consistent label. The other two are reported separately, and the aggregate is not used for ranking.

**Separation of bad from good inputs (d19_v3, reference-free).** The SMILES-only plausibility status has AUC 0.55 for pool failure (implausible-status AUC 0.56). Failure share: plausible 15%, suspect 14%, implausible 53% (30 pools). The current score has AUC 0.49 for consistent versus inconsistent parents, so it does not separate bad inputs. The status therefore needs generated-geometry checks (Section 2.5, P-IN1).

**Named failure repros (d18_v3, CPU, read-only).**
- **BODCAY** (inconsistent; SMILES contains `S=S=S=S`). Raises `ValueError: Bad Conformer Id` at `src/jacobik/engine.py:588` (`MMFFOptimizeMolecule` after a failed embed).
- **LIYTAQ** (ambiguous). Raises the same error at the same line.
- **OBABAX** (inconsistent; absurd_metal_charge). CPU setup completes. Its earlier failure was a GPU out-of-memory error.
- **ZAZREZ** (ambiguous; found while measuring folds). Raises `Bad Conformer Id` in the shell build.

**Required behaviour (never-fail fix).** When the embed fails, fall back to a non-embedded geometry (for example the SMILES coordinates with the metal at the donor centroid), set `embed_failed`, and score the row terribly. The exception must not propagate. This is an embed change and falls under Section 4.

---

### 2.1 Tail census (measured)

[D01] over all 696 pools:

| Quantity | Mean | Median | p90 |
|---|---:|---:|---:|
| Native top-1 `top1_dE`, kJ/mol | 140.9 | 77.5 | 272.8 |
| Floor `floor_dE`, kJ/mol | 112.8 | 60.1 | 239.5 |
| Regret `top1 − floor`, kJ/mol | 28.1 | 4.8 | 55.6 |

Other counts: top-1 within 4.184 kJ/mol of the floor in 48.7% of pools; pools with top-1 > 100 / 200 / 500 / 1000 kJ/mol: 273 / 116 / 30 / 12. Capping top-1 at 500 kJ/mol lowers the mean to 114.1 (this is an accounting statement, not a fix). Among the 232 three-seed parents, the seed range exceeds 20 kJ/mol for 72 and exceeds 50 kJ/mol for 23.

**Ceilings implied by these numbers.** A selection-only change (ranking, output window, dedup) can recover at most the mean regret, 28.1 kJ/mol (median 4.8). Only a generation change can move the floor, which averages 112.8 kJ/mol. Every tail item below is tagged with which of these ceilings applies.

### 2.2 Tail census per parent (measured)

Each row is one parent with its seed values. `top1 int` and `floor int` are `intact_ref`. [D01], [D02], [D04].

| Parent | Stratum | n rows | top-1 dE (3 seeds) | floor dE (3 seeds) | top-1 intact | floor intact | Angle RMS top1 / floor (deg) | ρ(angle, dE) range |
|---|---|---:|---|---|---|---|---|---|
| DAMVAQ | 4d_pos_med | 16 | 623, 662, 634 | 556, 550, 554 | no | no | 3.1 / 2.8–3.0 | 0.91–0.98 |
| HOXJUZ | 3d_zero_low | 16 | 1800, 1643, 1629 | 1225, 1181, 1206 | yes | yes | 10.3–10.8 / 7.7–8.2 | 0.84–0.93 |
| IFUROQ | 3d_zero_low | 1–2 | 756, 755, 754 | 756, 755, 732 | yes | yes | 3.5–3.9 / same | undefined (n<3) |
| JEGPUJ | 5d_zero_low | 3–4 | 685, 702, 677 | 685, 621, 621 | yes | yes | 2.9–3.0 / 2.7–2.9 | 0.5–1.0 |
| NEPNEF | 3d_zero_med | 48 | 2726, 2655, 2673 | 2620, 2597, 2614 | no | no | 5.6 / 5.6 (all rows) | −0.21–0.29 |
| SOBHAS | 3d_zero_high | 224 | 1064, 1583, 1662 | 373, 456, 489 | no | yes (73–77% of rows) | 1.6–1.8 / 1.7–2.1 | −0.09–−0.03 |
| USAPOU | 4d_pos_high | 80 | 557, 568, 505 | 472, 448, 505 | no | no (all rows) | 2.9–4.1 / 2.6–3.1 | 0.10–0.25 |
| XEYTUT | 4d_pos_med | 13 | 772, 775, 759 | 618, 713, 702 | yes | no (31–39% of rows) | 7.7–7.8 / 5.0–6.5 | −0.30–−0.04 |
| YUCZUS | 5d_zero_low | 48 | 632, 706, 611 | 350, 403, 434 | yes | yes | 2.8–3.3 / 2.4–2.7 | 0.24–0.62 |
| ZELTAP | 5d_zero_high | 96 | 1191, 1481, 1401 | 383, 343, 316 | yes | yes (85–90% of rows) | 5.9–7.5 / 1.7–2.3 | 0.89–0.94 |

Angle RMS is the RMS over bonded non-metal angles of `|θ_gen − θ_ref|`, with the same mapping as the orchestrator (`X[:, gen_to_ref_idx]`). ρ is Spearman over pool rows (`[D04]`). Values from `[D01]` and `[D04]`; the stratum names are from the pool JSON.

### 2.3 Cross-cutting measurements

**M-1. Selection versus generation.** `[D09]` on the effective pools (696). Ranking the native top-1 only among `intact_ref` rows (an evaluation-label oracle, not production-valid) moves the mean top-1 from 140.9 to 140.3 kJ/mol. 108 of 696 pools have a non-intact native top-1; 29 of them contain an intact row, and on those the mean change is −15.1 kJ/mol (median −6.9). **Consequence:** a coordination filter on ranking is a weak lever. Most of the tail is generation or intra-ligand geometry. (An earlier 660-pool count came from raw coordination arrays of R2-repaired pools; it is superseded.)

**M-2. Score and rank correlations with energy.** `[D02]` on the effective pools. The native top-1 uses `score`, so the score correlation is the one that matters; the rank correlation is shown for comparison. Spearman with −dE over rows, seeds 11 / 12 / 13:

| Parent | ρ(score, −dE) | ρ(rank, −dE) |
|---|---|---|
| HOXJUZ | −0.64 / −0.64 / −0.69 | −0.77 / −0.59 / −0.65 |
| ZELTAP | −0.03 / −0.17 / −0.06 | +0.05 / −0.14 / −0.01 |
| SOBHAS | −0.11 / −0.22 / −0.07 | −0.13 / −0.24 / −0.12 |
| YUCZUS | +0.03 / −0.13 / −0.03 | +0.03 / −0.13 / −0.03 |
| NEPNEF | +0.01 / +0.18 / +0.38 | same |
| USAPOU | +0.74 / +0.78 / +0.64 | +0.67 / +0.59 / +0.56 |
| XEYTUT | +0.68 / +0.89 / +0.70 | +0.68 / +0.90 / +0.70 |
| DAMVAQ | +0.41 / +0.51 / +0.50 | same |
| JEGPUJ (3–4 rows) | +0.50 / +0.60 / +0.60 | −0.50 / +0.40 / +0.40 (unreliable) |
| IFUROQ (1–2 rows) | undefined | undefined |

Over all 696 pools, argmax(score) equals argmax(rank) in 89.5% of pools. For pools with at least 3 rows the median ρ(score, −dE) is 0.66 and ρ(rank, −dE) is 0.64. Within HOXJUZ the reward ranks the worst structures first under either array. The choice between score and rank changes the top-1 in about one pool in ten; its effect on the mean top-1 is small (142.5 kJ/mol by rank versus 140.9 by score, `[D10]`).

**M-3. Bond lengths are not the driver.** `[D03]` Covalent bond RMS deviation from reference is 0.011–0.023 Å in every tail parent except NEPNEF (0.131 Å) and XEYTUT (0.054 Å). Its rank correlation with dE has no consistent sign across seeds.

**M-4. Moving angles track energy in the high-energy intact tails.** `[D04]` In ZELTAP and HOXJUZ the native top-1 has larger bonded-angle RMS than the floor in every seed (ZELTAP 5.9–7.5° vs 1.7–2.3°; HOXJUZ 10.3–10.8° vs 7.7–8.2°), and ρ(angle RMS, dE) is 0.84–0.94 within pool. The per-angle decomposition of T3 (`[D13]`) shows that the excess sits in angles that move within the pool. No intra-ligand clash exists in any tail pool (`n_clash = 0`, threshold 0.8 × covalent radius sum).

**M-5. The bend prior is part of the native top-1 choice, and is small.** `[D10]` `score − rank` equals `bend_prior_solver` (stored in the S2b state) to within 1e-3 in 697 of the 711 S2b units (median unit difference 1.6e-6). The bend prior therefore enters the native top-1 directly, in score units (the units of `score` and `rank`). The other 14 units differ by up to 0.04 and carry a further term that is not stored; the solver comment at `engine.py:2266` names a ring-closure penalty as kept in the score. That attribution is **not verified**. Values of the bend prior are small: for the native top-1 vs the floor, ZELTAP −0.045/−0.053/−0.057 vs −0.070/−0.040/−0.048; SOBHAS −0.0001/−0.0004/−0.0004 vs −0.010/−0.016/−0.003; HOXJUZ −0.146/−0.118/−0.115 vs −0.173/−0.199/−0.202. In HOXJUZ the **floor** carries the larger penalty, so the bend prior does not explain the top-1 vs floor difference in these pools (`[D07]`).

**M-6. The stored terms are the ranking terms, times Z.** `[D10]` `rank × Z` equals `Σ terms_Q1..Q6` in every pool, with a maximum absolute difference of 2.2e-5 (float32 rounding). Z ranges from 9 to 52 (median 26). The earlier statement that the score-to-terms difference was an unexplained residual of 9–205 is **withdrawn**: it compared `score`, which is on the per-Z scale, with the unnormalised sum of terms. The ranking identity has no unexplained residual.

**M-7. Non-intact tails and coordination.** `[D01]` The native top-1 is intact (`intact_ref`) in 84.5% of all pools, in 60.0% of pools with top-1 > 500 kJ/mol, and in 93.4% of pools with top-1 ≤ 100 kJ/mol.

### 2.4 Per-case entries

Each entry gives: observation, hypothesis with code location, the diagnostic that confirms or refutes it, the proposal, flag, test, metric and acceptance rule, expected effect with its evidence basis, cost, and risks.

---

**T1. Non-intact tails with no intact row (DAMVAQ, NEPNEF, USAPOU).**

- **Observation** `[D01]`, `[D02]`: all rows non-intact in each seed; floor is also non-intact. DAMVAQ 623–662 vs 550–556; USAPOU 505–568 vs 448–505.
- **Hypothesis.** The IK solve does not reach a first-shell coordination state for these ligands, so no ranking can fix it. Ceiling: the entire mean of the top-1 (not the regret), so this is a **generation** item.
- **Diagnostic.** The tail table in Section 2.2 (`[D01]`) gives the all-rows-non-intact status. For each non-intact parent, compute `d_site` per donor (distance to `μ_i` from Section 1.5) for the best-scored and the lowest-energy row. If one donor is far off for all rows, the cause is a fixed-target or donor-mapping problem, not the solver's search. Pass: a systematic donor offset > 0.3 Å on every row. Fail: scattered offsets (then search).
- **Proposal.** Conditional on the diagnostic: (a) if systematic, trace the target geometry from `ik/targets.py` (the Hungarian site assignment at `targets.py:378-431`) for that parent; (b) if scattered, the search extension of Section 3.
- **Flag.** None until the diagnostic decides. **Test.** A fixture from one parent with an injected donor mapping error must be caught by the diagnostic.
- **Metric and acceptance.** Fraction of pools with any `intact_site` row; the tail parent's floor. Accept if the count of non-intact-only pools falls and no healthy (`cal` non-tail) pool loses its intact share by more than 2 percentage points.
- **Expected effect.** Unknown. Ceiling is the floor (112.8 mean). Guess only.
- **Cost.** CPU diagnostic, one day; GPU re-solves for the affected parents. **Risk.** The fix can increase non-intact share elsewhere if it changes the target.

---

**T2. Ranking selects non-intact rows when intact rows exist (SOBHAS, XEYTUT).**

- **Observation.** SOBHAS: top-1 non-intact in all seeds, floor intact in 73–77% of rows, top-1 1064–1662 vs 373–489. XEYTUT: top-1 intact in all seeds, floor non-intact.
- **Hypothesis.** Ranking is `argmax(score)` (`refit/nci_v2_pilot.py:589`; orchestrator `:746`) and the reward has no coordination-integrity term. Q1 wall (`q1_wall`, `scorer.py`) acts only beyond table support. This is a **selection** item, so the ceiling is the mean regret.
- **Measured bound.** `[D09]` 29 of the 108 pools with a non-intact top-1 contain an intact row. The mean change of an intact-restricted top-1 on those is −15.1 kJ/mol (median −6.9), and across all 696 pools the mean changes by −0.6 kJ/mol. Selection alone is a small lever.
- **Proposal P2 (flag `rank_site_intact`).** Among rows with `intact_site`, take the argmax score; if no row is intact, keep the legacy argmax and set `top1_status = "no_intact_row"`. Ties keep legacy order. Never drop non-intact rows from the output window, only from the top-1 choice.
- **Test.** On the stored pools, the reproduction equals `d09` to 0.1 kJ/mol using `intact_site` (P1) in place of `intact_ref`; on a pool with no intact row, the legacy top-1 is returned exactly.
- **Metric and acceptance.** Paired parent change in `e` over the `tail` panel and `cal`; reject if any parent mean worsens by > 10 kJ/mol or the valid-output share decreases. Accept the flag only if `cal` mean `e` falls with paired 95% upper bound < 0 (versioned mode).
- **Expected effect.** Measured ceiling −0.6 kJ/mol on the set; the tail gain is limited to the 29 pools that contain an intact row. Low expected value; keep as a cheap guard.
- **Cost.** Negligible compute. **Risk.** `intact_site` must agree with `intact_ref` (P1 acceptance) or the flag will demote good rows.

---

**T3. Intact but high-energy tails: moving ring and non-donor angles (ZELTAP, HOXJUZ; YUCZUS and JEGPUJ with smaller effects).**

- **Observation.** M-4. The per-angle decomposition `[D13]` uses S2b units, where `embed_idx = −1` on every row of HOXJUZ, ZELTAP, YUCZUS and SOBHAS: one embed per pool, so all row-to-row differences come from the solver's joint DOFs. Top-1 minus floor sum of squared angle error (deg²) by role, over moving angles (spread across rows > 0.1°):

| Parent | moving angles per pool (donor-centred / ring / other) | top-1 minus floor, donor-centred | ring | other |
|---|---|---|---|---|
| HOXJUZ (seeds 11, 12, 13) | 1 / 8 / 20 | −20, −13, −23 | +5157, +2737, +3970 | +442, +1150, +612 |
| ZELTAP | 2 / 8 / 24 | +64, +14, +2 | +1888, +3655, +2813 | +1291, +2297, +2094 |
| YUCZUS | 0 / 0 / 6 | 0 | 0 | +217, +558, +29 |
| SOBHAS | 19 / 0 / 0 | −20, −130, −117 | 0 | 0 |

  Non-moving bonded angles already differ from the reference by 0.9–3.2° RMS in these pools (`fixed_angles_rms_top1`). This is an embed-level error present in every row, not solver motion.
- **What the data say about the brief's primary hypothesis.** Donor-centred angles (the angles at kappa donors that the legacy branch-bend joints of `fk.py:262` can move) do not carry the excess: in HOXJUZ the top-1 has *less* donor-centred error than the floor, and in ZELTAP they contribute at most 2% of the excess. **Hypothesis (b) is therefore not the primary driver in these pools.**
- **Hypothesis H-T3c (primary, data-driven).** Ring-internal angles move under ring-related DOFs (type c: ring-opening joints, `ring_open_adj`, and small-ring joints, `ring_joints`; `topology.decompose` RING_JOINTS) with no prior. These carry 57–61% of the top-1 excess in ZELTAP and 70–91% in HOXJUZ (computed from the sums of squares above, per seed). Refutation: the moving ring angles are instead moved by torsion joints of acyclic bonds (type d), which the mapping step decides.
- **Hypothesis H-T3d (non-donor, non-ring moving angles; about 40% of the excess in ZELTAP and all of it in YUCZUS).** Such angles are moved by pivot joints, either priced (type a: the Q7 chelate-backbone bridges of `ik/angle_bends.py`) or unpriced (branch joints at non-donor atoms, or macrocycle joints). Their type cannot be decided without the mapping step.
- **Hypothesis H-T3b (kept as the alternative).** Q7 is too weak relative to the score differences. It applies only if the type-(a) moving angles carry the top-1 excess after mapping.
- **Revision (d15, d16, d22).** Moving ring angles in HOXJUZ are moved by ring-fold hinges (type e, Section 2.5), not by ring-joint torsions. The type split must include type e. P3 does not address type e.
- **Withdrawn.** The earlier proposal `bend_pivot_scope = "all_covalent"` (pricing every covalent angle) is withdrawn. A bonded angle can change between rows only if some joint moves it (the `angle_bends.py` header: rigid rotations; bond lengths and ring internals do not change). Angles that no joint moves are constant within a pool, so pricing them against the embed value would add a constant and change nothing.
- **Diagnostic D-T3.**
  1. **Done:** per-angle spread and role split (`tools/jgen/diag/d13_angle_motion_attribution.py`).
  2. **Required:** map each moving angle to the joints that move it. Build the joint list per ligand from the FK construction (`ik/fk.py`: joints and their downstream atom sets, including the branch-bend joints at `fk.py:262` and the ring joints), the bend specifications of `ik/angle_bends.py`, and the ring openings. A joint moves angle i-j-k if its downstream set contains exactly one of i, j, k, or contains j with i and k on the same side. Every moving angle (spread > 0.1°) must map to at least one joint. **Any unmapped moving angle is a bug to be reported, not attributed.**
  3. **Required:** re-split the top-1 minus floor sum of squared error by joint type a, b, c, d.
  - Decision: H-T3c is confirmed if type c carries at least 50% of the top-1 excess; H-T3d if types a and d together carry it.
- **Arm (existing flag, no new code).** `branch_prior = True` (`fk.py:65-67`: with angle bends on, donor branch points get prior-penalised bend DOFs instead of the legacy unpenalised joints; `ARCHITECTURE.md:490` records it as "off in both profiles, no benefit shown"). What differs now: (i) the test set is the tail panel of ten parents, not the development panel; (ii) the metric is paired per-parent top-1 and floor on effective pools, with the noise band of Section 1.4.3; (iii) the per-angle attribution compares the arm's top-1 with control's top-1, which shows whether the arm changed the donor-centred angles it was meant to change. The arm is a control for hypothesis (b) and is not expected to change the ring-driven excess.
- **Prerequisite for P3 (auditor note).** "With no prior" is not yet established. Ring-joint rings are closed by a penalty that is not stored in the pools: `engine.py:2266` names a ring-closure term, and 14 S2b units have a score residual of up to 0.04 that this term may explain. Before P3 is coded, read that term and state its form and weight, and whether it acts on ring angles or only on the closure distance. If it already constrains ring angles, P3 becomes a change to its weight or form under the same flag, not a second penalty.
- **Proposal P3 (conditional on D-T3 step 3; flag `ring_joint_prior`, default weight 0).** A Q7-form prior on ring-internal angles of ring-joint rings: `ring_prior = −w · mean_i ½ (max(0, |Δθ_i| − free)/σ)²`, reference = the embed value, free zone 2°. The stiffness σ is an **extrapolation**: the table `BEND_K` in `angle_bends.py` was measured for pivot DOFs, so a ring stiffness must be fitted on a calibration set before the arm is used.
- **Test.** (i) Weight 0 is bit-identical to legacy (T-G2). (ii) The analytic gradient matches a 5-point finite difference to 1e-6 relative at 20 random poses. (iii) On ZELTAP, with weight > 0, the ring-angle RMS of the top-1 in a single pool solve moves toward the floor's, within 2°. (iv) With `branch_prior = True`, the donor-centred moving-angle count changes on at least one tail parent (otherwise the arm is inert there).
- **Metric and acceptance.** Tail and `cal` panels, versioned mode: paired change in `e`, and the joint-type split of the top-1 excess. Reject if any `cal` parent mean worsens by more than 10 kJ/mol or the valid-output share falls. Accept only if the tail panel's mean `e` falls with the paired upper bound below 0.
- **Expected effect.** Unknown. The ceiling for these parents is their regret (ZELTAP 808–1138 and HOXJUZ 423–575 kJ/mol, from the tail table). Guess only.
- **Cost.** One penalty evaluation per IK step; the solver is 30–50% of wall time (Section 5.1). **Risk.** A ring prior can stiffen legitimate ring puckers and macrocycles, so the sweep must include `cal`, not only the tail panel.

---

**T4. Single-class intact tail with very few rows (HOXJUZ 16 rows, one class; IFUROQ 1–2 rows; JEGPUJ 3–4 rows).**

- **Observation** `[D08]` and `[D01]`: IFUROQ returns 1–2 rows, so top-1 equals floor and no selection is possible. JEGPUJ returns 3–4 rows. HOXJUZ has one assignment class and 16 rows.
- **Hypothesis H-T4.** The output window (`engine.py:93-96`, `OUT_WINDOW = 200` nats, applied in score space as `OUT_WINDOW / z_norm`) and the dedup at `0.5` Å (`ik/dedup.py:335`, `DEDUP_RMSD`) remove most rows. The pre-window count is not stored.
- **Diagnostic.** `JACOBIK_TIMING_TRACE=1` (Section 5.2) records pre-window, post-window, and post-dedup counts. Pass: a post-dedup count that is much lower than the pre-window count for these parents. Fail: the solver itself produces few valid rows (then search, Section 3).
- **Proposal.** No change until the trace exists. If the window is the cause, measure the output-window setting on the `cal` panel as a separate arm (not a new flag; it is `out_window`, an existing field).
- **Test.** Trace counts add to the pool JSON; the stored pool arrays are unchanged.
- **Expected effect.** Unknown. **Risk.** Widening the window increases output size and dedup cost (Section 5).

---

**T5. Lanthanides losing donor contact (KADCUN, LAWRAC, ROJMEJ, SUQWEG).**

- **Observation (cited, S2.1 report, not re-run here).** All conformers `coord_intact` = False. Reference C donors 2.47–2.78 Å and N 3.04 Å; generated 4.3–5.1 Å (1.4–2.0 Å off). The donors are real first-shell atoms: the bound-donor definition includes long La–C contacts, but the deviation is not a definition artifact.
- **Hypothesis H-T5.** The target donor distances for 4f metals come from the Q1 table or covalent radii, which do not match the lanthanide donor distances. The Hungarian site assignment (`targets.py:378-431`) then places donors on sites that cannot reach the true contact distance. Alternative H-T5b: the Q1 wall (`q1_wall`) pushes donors away from a table value that is too short.
- **Diagnostic D-T5 (CPU).** For each of the four parents: (i) the solver's target donor distance for each donor (the value the site geometry asks for), (ii) the reference donor distance, (iii) the Q1 mode for (La, donor element). **H-T5 is supported** if target ≠ reference by more than 0.3 Å. **H-T5b is supported** if the Q1 mode differs from the reference by more than 0.3 Å.
- **Proposal P5.** If H-T5 holds: a metal-family site-radius table for 4f donors (declarative, versioned in `spec`), tested only on those parents plus the `cal` 4f stratum. If H-T5b holds: the Q1 support for 4f is the defect; route to the scorer, not to generation.
- **Test.** For the four parents: the post-change `intact_site` fraction. **Acceptance.** Intact share rises for the four; no `cal` 4f parent loses intact share.
- **Expected effect.** Unknown; ceiling is their floors, currently 1.4–2.0 Å donor error. **Risk.** Changing 4f targets changes every 4f parent, so the stratum must be in the same arm.

---

**T6. NEPNEF: O–N bond lengths of a nitro-type group (template error).**

- **Observation** `[D03]`, `[D04]` (reproduced): every row (48 of 48, all seeds) has bond RMS 0.131 Å and at least one bond deviating > 0.2 Å. The worst bonds are O–N: reference 1.206 Å (double-bond character), generated 1.48–1.52 Å (single-bond). Floor is 2597–2620 kJ/mol, so the whole pool is shifted and not just the tail.
- **Hypothesis H-T6.** The free-ligand embed or the bond-order table gives these N–O bonds single-bond character (bond order or formal-charge representation of the nitro/N-oxide-type group), and the IK solver preserves bond lengths (rigid sides, `angle_bends.py` header), so the error is inherited unchanged from the embed.
- **Diagnostic D-T6 (CPU, RDKit).** From the builder's SMILES for NEPNEF: list each N–O bond's order, formal charges, and whether the conformer built by the free-ligand embed (`engine.py:583-585`, the `ETKDGv3` path, with the ETKDG fallback) gives the same bond length as the reference. **H-T6 is supported** if the embedded bond length is already ≈ 1.5 Å before any IK solve. **H-T6 is refuted** if the embedded length is ≈ 1.2 Å and the solver lengthens it.
- **Proposal P6.** If supported: a fix in the ligand graph or SMILES for NEPNEF (a data change, not a solver change). Generalise: a bond-length check at embed time against the bond-order table (flag `embed_bond_check`, default False) that rejects an embed with any bond deviating > 0.15 Å from the table; rejected embeds retry with the next embed seed.
- **Test.** NEPNEF's embedded N–O length within 0.05 Å of 1.21 Å; the flag reproduces the diagnostic on the stored ligand; no other `cal` parent's embedded bond set changes unless it violates the table.
- **Acceptance.** NEPNEF floor drops below 1000 kJ/mol in the re-solve; no `cal` parent's floor worsens by > 10 kJ/mol.
- **Expected effect.** Large for NEPNEF (≈ 2600 → unknown); a guess for the rest. **Cost.** Low. **Risk.** The bond-order table can be wrong for other charged groups; the check must be validated on the `cal` panel before the flag is used.

---

**T7. IRAQUO: hydrogen mapping error (the S2.1 placement reading is corrected).**

- **Observation** `[D11]` (all rows of all three seeds; IRAQUO carries an R3 overlay only, so raw and effective arrays are identical). In each seed exactly two of the 26 reference carbon–hydrogen pairs are bad, both on the carbons C16 and C39:

| Seed | rows | bad pair (ref H on carbon) | mapped distance from carbon, every row | raw nearest-H distance from carbon |
|---|---:|---|---|---|
| 11 | 11 | H17 on C16 | 29.75 Å (11/11 rows) | 1.08 Å |
| 11 | 11 | H41 on C39 | 28.45 Å (11/11 rows) | 1.11 Å |
| 12 | 11 | H18 on C16 | 28.45 Å (11/11 rows) | 1.11 Å |
| 12 | 11 | H41 on C39 | 29.75 Å (11/11 rows) | 1.08 Å |
| 13 | 12 | H18 on C16 | 28.45 Å (12/12 rows) | 1.11 Å |
| 13 | 12 | H41 on C39 | 29.75 Å (12/12 rows) | 1.08 Å |

  The other 24 reference carbon–hydrogen pairs in each seed map within 1.08–1.11 Å in every row. In the raw generated frame, each of the two carbons has a hydrogen at bonding distance in every row, so the generated structure is chemically intact at these carbons. The mapping sends one reference hydrogen per carbon to a generated hydrogen 28–30 Å away. `mapping_ambiguous = True` and `num_automorphisms = 2` in all three seeds, and `gen_to_ref_idx` differs between seeds, as expected for a pool-level mapping.
- **Correction of the S2.1 report.** `S2_1_REPORT.md` (line 164) reads the stray hydrogen as a deterministic placement error with the stage unattributed. The stored data support a **mapping** error: the coordinates are chemically correct, and the reference-to-generated hydrogen index is assigned to a hydrogen on a different carbon.
- **Where the pairing happens.** The orchestrator calls `build_atom_mapping(mol, gen_mol=confs[0].complex, X_ref=ref_X, X_gen=X[0])` once per pool (`refit/nci_v2_orchestrator.py:709`) and applies `gen_to_ref_idx` to every row (`mapped_confs = X[:, gen_to_ref, :]`). The function is `build_atom_mapping` in `src/jacobik/builder/atom_map.py:191`. Hydrogen pairing happens in three places in it:
  1. the bonded-hydrogen lists `h_neighbors` of each heavy atom, built from the bond graph of each molecule (`atom_map.py:136-149`);
  2. the attached-hydrogen matching of each matched heavy parent: one-to-one when each side has one hydrogen, and a permutation search minimising |P_ref − P_gen| using `X_gen` when a parent has several (`atom_map.py:320-335` and `:386-400`);
  3. the residual pass for symmetry-equivalent hydrogens, which assigns the remaining ones by minimum distance among the parent's hydrogens (`atom_map.py:450-475`).
- **Hypotheses.** H-T7a: the bonded-hydrogen list of the generated image of C16 or C39 contains the far hydrogen, so the bond graph of `confs[0].complex` is inconsistent with the coordinates. H-T7b: the bond graph is consistent, but the permutation of step 2, or the residual pass of step 3, selects the far hydrogen.
- **Discriminating test (required; CPU; needs the builder).** For IRAQUO seed 11, print (i) the bonded-hydrogen indices of the generated images of C16 and C39 in `confs[0].complex`, and (ii) the `h_neighbors` lists used by the mapping. **H-T7a is supported** if the far hydrogen is in the bonded list of the carbon. **H-T7b is supported** if the bonded list is correct and the permutation or residual pass chooses the far hydrogen. The `d11` table gives the expected pairs.
- **Proposal P7 (a mapping fix, not a generation change).** After pairing, every hydrogen pairing must satisfy a bonding distance of at most 1.25 Å in the generated coordinates of the conformer used. If a pairing violates it, re-run the permutation over that parent's hydrogens with the bonding-distance constraint, falling back to the lowest-cost pairing. Mapping is per pool, so the fix can use the first conformer or the best intact row. The fix changes health labels (`graph_match`) and possibly the NCI extraction set, not the MACE energies.
- **Test.** T-P7: every reference carbon–hydrogen pair in IRAQUO maps within 1.25 Å in all rows of all seeds. Healthy controls (ACAPDA, QUTWOT, AFUQAT, per the S2.1 report) keep identical `gen_to_ref_idx`.
- **Acceptance.** IRAQUO becomes healthy; the count of `graph_match` failures does not increase on any pool; no healthy pool's mapping changes.
- **Expected effect.** A health-label correction; its energy effect is unknown. **Risk.** Changing the mapping changes which poses the health policy keeps (roadmapv2 section 9.8). Coordinate with the NCI v2 health policy before merging.

---

**T8. Lanthanide-free and other failures carried from Step 2 (cited).**

- **OBABAX, TATXIW, FEKKAJ, QEZVEY (OOM).** Cited (`S2_1_REPORT.md` lines 77, 82, 105): largely caused by the orchestrator retry bug, fixed in `21d5446`. Not retried. **Proposal:** a GPU re-run of OBABAX under the fixed orchestrator (batch, Section 8). No solver change. **Acceptance:** the pool completes within the memory guard and passes the health check.
- **BIKQIY (OOM after the fix), COJSIG/IYAMUR, OMAYEJ/LELQID, UZOREI/SIXQUL, VALJEB/ZORLAV, KOVLUC, RALVAD, XOLXAX, ZOLQIC, MIVLAF (RDKit `Bad Conformer Id` or CUDA assert).** Cited, not retried. **Proposal:** CPU reproduction of the embed step (`chelate_embed.py:405-422`, the bounded embed with `EmbedMultipleConfs(N_TRY)`), catching the exception and recording the failing call and its arguments. Fixing `Bad Conformer Id` is an embed change and falls under Section 4.
- **BODCAY, LIYTAQ (RDKit `Bad Conformer Id`).** Named in the task brief. **Not verified in this session** (they are not in the `failures` panel's pool set in this audit). **Proposal:** the same CPU reproduction as above.
- **ADINOR (chelate-aware embed fallback).** Named in the brief; the fallback path is `chelate_embed.py:405-422` (violation-based retry) and `engine.py:583-585` (unbounded ETKDG retry). **Proposal:** record which path produced each embed; Section 4 covers the fix.
- **Baseline assignment repaired (topologically infeasible).** `engine.py:1530-1572` (`_repair_infeasible_baseline`). The warning text at `:1556` ("replaced by lowest-cost feasible assignment") and `:1565` ("keeping it") is logged per solve. **Proposal:** count these events per pool in the trace (Section 5.2); no change to the repair until the count is known. An infeasible baseline that is kept (`:1565`) is a pool-level flag in the output.

---

**T9. ZELTAP η²-alkyne float32 sensitivity (cited, not re-run here).**

- **Observation (cited, roadmapv2 section 4.2 and the auditor's scratch scripts `zeltap.py`/`zeltap64.py`).** Coordinates differ by up to 0.05 Å between float32 and float64 forward kinematics on the η² alkyne donor. ZELTAP is also the parent with the largest top-1 seed spread among intact parents (1191–1481 kJ/mol).
- **Hypothesis H-T9.** Float32 forward kinematics at the alkyne donor perturbs the geometry enough to change the reward ranking and the angle distribution. **Note:** this mechanism is distinct from the angle distortion of T3, which is present in float64 pools too (the solver precision of the stored pools is not stated in the pool JSON; see Section 9).
- **Diagnostic D-T9 (CPU, now).** Recompute coordinates from the S2b `thetas_flat` in float64 with the same FK path; compare with the stored `X` at the alkyne carbons. Pass for H-T9: differences > 0.01 Å at the alkyne. Then, for the top-1 and floor rows, rerun the angle RMS of `d04` on the float64 coordinates. **H-T9 is supported** if the float64 angle RMS of the top-1 drops toward the floor's.
- **Proposal P9.** Run the FK in float64 for alkyne donors (dtype policy in `ik/fk.py`), only as an arm; dtype changes are a new baseline identity (roadmapv2 section 4.1).
- **Test.** The float64 FK reproduces the float64 reference to 1e-8 on a healthy parent.
- **Acceptance.** ZELTAP tail reduction without a `cal` parent loss > 10 kJ/mol.
- **Expected effect.** Unknown. Guess only. **Cost.** Float64 FK is slower; measure in Section 5.

---

**T10. Output scarcity and worst-case timing (cited and measured).**

- Covered in T4 (window and dedup) and Section 5 (dedup outliers). The solver time is not the limiting factor for the tail parents; output scarcity is for IFUROQ and JEGPUJ.

---

**Summary of tail priorities (the ordering is fixed):**

| Order | Item | Ceiling | Evidence now | Blocking measurement |
|---|---|---|---|---|
| 1 | T6 NEPNEF template | floor (2600) | strong, [D03] [D04] | D-T6 RDKit check |
| 2 | T3 moving ring and non-donor angles | regret of ZELTAP/HOXJUZ | per-angle split, [D04] [D13]; donor-centred angles refuted as primary | D-T3 step 2 (joint mapping) |
| 3 | T7 H mapping (IRAQUO) | health label | strong, [D05] | P7 mapping test |
| 4 | T1 non-intact generation | floor | descriptive, [D01] | D1 systematic vs scattered |
| 5 | T5 lanthanide targets | floor | cited | D-T5 target distances |
| 6 | T2 selection filter | regret (measured bound −0.6) | measured, [D09] | P1 agreement with `intact_ref` |
| 7 | T4 output scarcity | unknown | measured counts only | timing trace |
| 8 | T9 float32 FK | unknown | cited | D-T9 |

### 2.5 HOXJUZ crush and ring folds (revision 2; d15, d16, d20, d22)

**Where the crush is born.** HOXJUZ's ring deviation is 1.74° in the embed and 8.7–33° in the final rows. Rebuilding the rows from the S2b thetas is exact. Zeroing the bend columns restores 1.74° (d16). The crush is born in the solve, through the bend plan's ring-fold hinges. This answers the puzzle in the brief.

**Mechanism (code).**
- `src/jacobik/ik/angle_bends.py:284-336` (ring folds): a non-aromatic ring of at least 6 atoms, with donors on both sides of a line through two non-adjacent ring atoms, gets a fold hinge. "Non-aromatic" means no bond of order 1.4–1.6 touching a ring atom (`mol_arom`, `:263-266`, checked at `:296`). HOXJUZ's SMILES writes the phenyls as quinoid (orders 1 and 2), so they are non-aromatic here, although the reference rings are aromatic.
- `angle_bends.py:79-80`: `stiffness()` gives every hinge k = 0 on purpose. The haptic rationale ("paid for by the metal bond", priced by S_orient, `scorer.py:831-850`) covers rings with a haptic face atom. It does not cover fold hinges on rings with no haptic face atom, such as HOXJUZ's κ-S rings. Nothing prices those folds.
- The header claim at `angle_bends.py:10` ("ring internals never change") is false for folds and must be corrected.
- Ring joints (`topology/decompose.py:116-121`) drop folds whose moving set or axis touches a joint ring (`angle_bends.py:353-359`).

**Corrected fold counts (d22, on the solver's own shell).** The earlier count of 4 consistent parents came from the reference molecule and is unreliable. On the reference, HOXJUZ has zero fold hinges, but the solver shell has six. Reason: `load_mol` returns the reference complex with its own atom order and donor sets (HOXJUZ ligand 0 donors `[2, 3, 4]`). The solver re-derives donors in `_build_shell` (`src/jacobik/engine.py:1318`) through `classify(mol, smiles_path=True)` (`engine.py:1337`), giving `[16, 9, 2]`. `bend_specs` reads `donor_atoms` and the bond graph, so it sees a different graph on the reference. d22 rebuilds the shell with `builder._build_shell` and runs `decompose_ligand` as the solver does (`topology/decompose.py:392`).

| Label | parents | parents with a fold splitting a 5/6-ring | all-sp2 ring (haptic) | sp3 or non-conjugated ring (non-haptic) |
|---|---|---|---|---|
| consistent | 174 | 5 (DAMVAQ, NUHPEL, POJHIH, REGDIR, XEVDAE) | 3 (NUHPEL, REGDIR, XEVDAE) | 2 (DAMVAQ, POJHIH) |
| ambiguous | 43 (42 plus ZAZREZ shell error) | 6 | 3 (BAKLEF, JUPDON, QUTWOT) | 3 (IFUROQ, LUQWOK, ZELTAP) |
| inconsistent | 15 | 2 | 0 | 2 (HOXJUZ, RATVAK) |

The earlier per-label counts (170/4/1) were from the reference and are superseded.

**Reference fold angles** (d22 `fold_deg_ref`; 0 = planar):
- Consistent, non-haptic: DAMVAQ 11.2° and 24.2°; POJHIH 31.8–52.7°. POJHIH is a consistent input whose reference ring is genuinely folded, with no multiple bonds in the ring.
- Ambiguous, non-haptic: LUQWOK 28–59° (reference folded).
- Flat references: HOXJUZ 0.4–2.4°, ZELTAP 0.3–3.0°, IFUROQ 0.2–3.7°, RATVAK 0.1–0.2°. Folds there are not reference geometry.
- Haptic rings report exactly 180.0° (BAKLEF, JUPDON, QUTWOT, REGDIR) or 1.7–2.6° (NUHPEL, XEVDAE). The 180.0° values are suspicious and unverified. They are not used.

**Bond-order gate (labelled comparison arm only, not a fix).** `fold_gate_multi`: no fold on a ring with any bond of order ≥ 1.4 in the SMILES. The user has rejected keying folds on bond orders tuned on one complex, because TMCs need ring folds in real cases. On the d22 data the gate would:
- remove the folds of HOXJUZ, ZELTAP, IFUROQ and RATVAK (2 multiple bonds each), fixing the HOXJUZ crush;
- keep DAMVAQ, POJHIH and LUQWOK (no multiple bonds), so the genuinely folded references keep their folds;
- remove the haptic folds of NUHPEL, XEVDAE and REGDIR, whose folds are priced by the metal bond. This is a cost the gate would impose on haptic chemistry.

**Proposals (flags default off).**
- **F-P1 `fold_prior_nonhaptic`.** Fold hinges on rings with no haptic face atom get a prior, using the data-fitted `BEND_K` by pivot hybridisation (`angle_bends.py:90-91`) on the fold-angle change from the embed value. The free zone must cover the reference folds (at least 10°; POJHIH needs about 53° reachable). Measure the reference fold each ring needs against the free zone first. **Tests:** POJHIH and DAMVAQ reach their reference fold within the free zone; HOXJUZ's ring deviation after the prior is below 5°; weight 0 is bit-identical to legacy.
- **F-P1 alternative `fold_haptic_only`.** Restrict folds to ligands with a haptic group. Not preferred on current evidence: it removes the non-haptic folds of POJHIH, DAMVAQ and LUQWOK, which the references show are folded. Only if F-P1 fails its test.
- **Comparison arm `fold_gate_multi`.** As above.
- **Still to measure.** Whether each COD-type or haptic ring got ring joints (`decompose.py:116-121`), which drops its folds at `angle_bends.py:353-359`. d22 counts folds after that drop but does not record the drop per ring.

**T3 revision.** The d13 attribution labels moving ring angles as "ring" (type c). In HOXJUZ those angles are moved by fold hinges, a separate joint type (type e, `angle_bends.py:284-336`, k = 0). The T3 decision rule must split type c and type e.

**P-IN1 (input status, never fails).** An `input_status` in {plausible, suspect, implausible} built from the d14 flags plus generated-geometry checks: ring-angle deviation from the embed, bond-graph change against the SMILES, and coordination loss. Implausible rows get a score penalty and the output is still produced. **Test:** pool-failure AUC above 0.7 on the consistent-plus-inconsistent set, reported per label. The SMILES-only AUC is 0.56, so the geometry checks are required.

### 2.6 Fold-hinge removal test and the solver issues it exposed (2026-10-25)

**Decision (owner): remove ring-fold hinges** (`("hinge", 0)`, `angle_bends.py:284-336`).
- **Why they existed.** They were added 2026-09-27 to close COD from the staggered embed into the tub.
- **Why COD no longer needs them.** COD now gets ring joints, which drop all folds on the ring (`:353-359`). Keeping COD's folds alongside the joints was tested 09-30 and was worse, and that option was removed 10-16.
- **Verified.** AWONOR, MOWRUN, USAPOU and XOJHIN have COD ring joints and 0 fold hinges. Since 10-16 the hinges fire only on rings they were not designed for, mostly κ-chelates, where k = 0 and nothing prices the fold.

**Test.**
- **Script:** scratch, `fold_test.py`. Arms: control (u1, nci off) and fold hinges dropped by a runtime patch. GPU, seeds 11–13, MACE.
- **Outputs:** xyz files in `inspect/fold_test/<CODE>/`.

| Complex | Tests | Fold hinges | top-1 dE control s11/12/13 | top-1 dE no-fold |
|---|---|---|---|---|
| HOXJUZ | crush case | 6 | 1800 / 1643 / 1617 | 269 / 281 / 269 |
| XOJHIN Rh(COD)Cl(NHC) | COD | 0 (ring joints) | 200 / 194 / 177 | 201 / 194 / 177 |
| CEHPIO CpTiCl₃ | plain Cp⁻ | 0 | 4.6 / 4.4 / 4.8 | identical |
| BEBQEE Zn(Bn₃-TAC)Et | κ³ 1,3,5-triazacyclohexane | 0 (ring joints) | 276 / 172 / 256 | identical within 0.1 |

**HOXJUZ result.**
- The phenyls stay intact: max ring-angle deviation 62–66° → 2.5°, minimum C–C–C angle 55–59° → 112°.
- Seed spread drops from 183 to 12 kJ/mol. Top-1 score drops from +0.4 to −1.85.

**Batch 2 (done 2026-10-25).** Run on the remaining fold parents from d22; seed-mean top-1 change, no-fold minus control:

| Parent | Fold hinges | top-1 control s11/12/13 | top-1 no-fold | Δ mean |
|---|---|---|---|---|
| ZELTAP | 6 | 1190 / 1483 / 1401 | 140 / 141 / 137 | −1219 |
| IFUROQ | 9 | 756 / 755 / 754 | 260 / 265 / 258 | −494 |
| DAMVAQ (ref folded) | 2 | 623 / 562 / 634 | 541 / 485 / 509 | −95 |
| RATVAK | 3 | 128 / 113 / 90 | 48 / 47 / 44 | −64 |
| LUQWOK (ref folded) | 7 | 236 / 191 / 212 | 169 / 159 / 166 | −48 |
| POJHIH (ref folded) | 3 | 125 / 111 / 164 | 97 / 111 / 99 | −31 |
| NUHPEL, REGDIR, XEVDAE, BAKLEF, JUPDON, QUTWOT | 0 | | identical | 0 |

- **Outcome.** 7 of 7 fold parents improve, counting HOXJUZ. There are no regressions, and the floors improve too.
- **Folded references.** The three parents whose reference rings are folded also improve. The embed already gives a usable ring shape.
- **Haptic six.** These have no `("hinge", 0)` specs. d22 counted their ring-segment hinges, which are unaffected.
- **Outputs:** `inspect/fold_test/summary_batch2.json`.

**Implementation (assigned to the implementing agent, together with the other flags in this document; not yet done).** A plain deletion now would change `u1` behaviour. It would also break NCI v2 Step 4: the A1 replay rebuilds `u1` bend columns from S2b state, which include fold hinges (ZELTAP, motion_validation, has 6). Sequence:

1. **P0 fingerprint fix first** (Section 1.2.2, test T-P0).
2. **New profile field `ring_fold_hinges`.** Legacy value `True`. Module attribute `jacobik.ik.angle_bends.RING_FOLD_HINGES`. `False` skips the ring-fold block (`angle_bends.py:284-336`) only. Ring-segment hinges (`("hinge", n>0)`) and the joint-ring drop (`:353-359`) are unchanged.
3. **Tests:**
   - T-P0;
   - T-G2 with the flag at `True` (`golden: OK`);
   - with `False`, no `("hinge", 0)` spec on HOXJUZ, ZELTAP and POJHIH, and identical specs on XOJHIN, CEHPIO, BEBQEE and the haptic six.
4. **The next generation profile sets `ring_fold_hinges = False`.** Bench it on the full eval panel against the S2b noise band.
5. **Delete the code path** once NCI v2 no longer replays `u1`. Correct the header claim at `angle_bends.py:10`, and fix the comment at `:285-287`, which still says the folds close COD.

**Solver issues exposed (owner visual inspection; numbers measured on the no-fold rows).**

- **S1. HOXJUZ: the central S is inverted in every row of every seed** (Ni inside the C–S–C wedge).
  - Measurements: C–S–Ni 52–63°, angle sum at S 229°, against the reference 102–107° and 315°. The ring–ring dihedral is 85° against the reference 55°.
  - **Mechanism (measured).**
    - The quinoid SMILES gives an embed whose S···S bites are 3.45 Å, against 3.01–3.08 Å in the reference.
    - With the rings rigid, the three S donors span a circle of radius 2.40 Å, so no Ni position lies 2.15–2.25 Å from all three.
    - Inverting the central S is a rigid-body move. No bend pays for it, only S_orient (about 22 nats on that donor, +3.9 → −17.7).
    - The alternative is bending the thione C–S bonds inward (two `kind=None` bends with `BEND_K` sp2 in-plane 1300 kJ/mol/rad² and a 2° free zone) or the central-S bends.
  - **Not established:**
    - whether the solver rejects the non-inverted basin on cost or never visits it;
    - whether the solve objective (variance-floored, B2.7 option B) weighs S_orient lower than the ranking score does.
  - **Diagnostic (CPU):** build the best non-inverted pose by penalising inversion (e.g. ×10 S_orient on that donor). Compare its solve objective, ranking score and MACE energy with the inverted row's.
    - If the non-inverted pose scores better but is never found, the problem is search.
    - If it scores worse, the balance between the bend stiffness and S_orient is wrong.
  - This is an inconsistent input (wrong bond orders), so the terrible score is the intended outcome. The open question is whether a less absurd structure was available.
- **S2. XOJHIN: carbene isopropyl methyl H at 2.09–2.16 Å from Rh in every seed** (reference 4.1–5.5 Å; 5 bonds from Rh, so inside Q6).
  - The Q6 (H, Rh) wall is a half-Gaussian with μ = 3.04 Å and σ = 0.357 Å (n = 2,994). At 2.09 Å that is z ≈ 2.7, about 3.5 raw nats per contact before normalisation. It is soft because the fit pools agostic and steric M···H contacts under one element-pair key.
  - **NCI v2 does not address this.** It adds attraction, while Q6 keeps doing repulsion.
  - **Candidates:**
    - a contact-class key for M···H (distinguishing agostic geminal/β C–H from remote methyl H by bond distance from the donor);
    - or a hard core at the 1st percentile of the M···H distribution.
  - Measure first: how often top-1 rows have M···H below 2.2 Å where the reference has more than 3 Å, per metal.
- **S3. BEBQEE: κ³ TAC half-detached.** Zn–N is 2.45–3.07 Å in top-1 rows against 2.24–2.36 Å in the reference; Zn–C is 2.02–2.34 Å.
  - The engine warns: "Baseline assignment [1, 2, 0, 3] is topologically infeasible and no feasible alternative was found; keeping it" (`engine.py:1565`), and "Symmetry enumeration found no reachable assignment".
  - So the donor→site assignment for this fac-κ³ small-ring donor set is judged unreachable. The solve then runs on a fixed assignment that the ring-joint ligand cannot close onto.
  - This is unrelated to folds: both arms are identical. Check the ring-closure term (`engine.py:2266`) on this case first (see the note before P3).
  - **Diagnostic:** why the feasibility test rejects every assignment (pass/fail per assignment; geometry of the fac N₃ set in the embed), and how often the warning fires across the 232 parents. The warning text is a cheap census key.
- **S4. DAMVAQ: collapsed rhodacyclobutane and an unencoded σ-C–C interaction.**
  - Top-1 is 485–634 kJ/mol in both arms. The input is labelled consistent: SMILES bond orders agree with the geometry.
  - **Reference.** Rh(III) has:
    - P at 2.28 Å;
    - two Rh–C σ bonds, C10 and C15 at 2.04 Å, which form a rhodacyclobutane Rh–C10–C11–C15 (C10–Rh–C15 64°, Rh–C–C 96°, C10···C15 2.15 Å);
    - an η²-interaction with a cyclopropane C2–C8 edge of the same cage, Rh···C 2.36 / 2.36 Å.
  - **(a) The σ-C–C interaction is not in the SMILES**, which has three donors only. The generator has no reason to fold the cage onto Rh: the cyclopropane ends up 5.6–5.8 Å away in every row. This is a representation limit, not a solver fault.
    - Not addressed directly.
    - A 3-coordinate Rh(III) is a candidate unsaturation flag for P-IN1.
  - **(b) Solver/embed: the metallacycle bite is wrong.**
    - The metal-free embed gives C10···C15 = 2.53 Å, a normal sp3 angle at C11. The rigid cage cannot compress it to 2.15 Å.
    - The solver closes the ring by dropping Rh onto the bridgehead: Rh–C11 1.96–2.12 Å against 2.68 Å, Rh–C–C 53–65° against 96°.
    - In some seeds P also goes trans to an alkyl (165°; reference cis/cis 99°).
    - Rh···C11 is 2 bonds from Rh, so Q6 excludes it. Only S_orient at C10/C15 prices the collapse.
    - **Same class as S1** (HOXJUZ bite 3.45 against 3.05 Å): a free-ligand embed sets a small-metallacycle bite that the metal must change, and the solver distorts the metal side instead.
  - **Candidate fix (embed, Section 4), withdrawn 2026-10-25 (see S5).** Bound the donor-pair distance of 3- and 4-membered metallacycles in the chelate-aware embed (`chelate_embed.embed`) from the DB's own Q1 distances and the Q2 D–M–D angle, d = 2r·sin(θ/2) (here 2 × 2.04 × sin 32° = 2.16 Å). No new fit is needed.
  - **Prevalence.** A rough SMILES count gives about 733 tmQM complexes (0.7%) with an sp3-C-bridged 4-membered metallacycle. This is an undercount: it misses DAMVAQ. In the pools: BIBFUN (platinacyclobutane) top-1 86–91, UKEPIJ (dppm) 65. Flexible rings suffer far less than a rigid cage.
  - **Test.** The bite-bounded embed on DAMVAQ, HOXJUZ, BIBFUN and UKEPIJ against legacy. Report Rh/Pt···bridgehead distance, M–C–C angles and top-1.
- **S5. JEGPUJ: the planar κ⁴ C^N\*N^C Pt core comes out twisted** (found by owner visual inspection, 2026-10-25).
  - **Input.** Labelled consistent. Pt(II) with two aryl C⁻ and two triazole N donors; the aryls are bridged by an ether O. The chelate rings are 5-6-5, with a Pt–C–C–O–C–C six-ring in the middle.
  - **Measured** (`scratch jeg_planar.py`; reference vs the top-1 of seeds 11–13, which is the same in the control and no-fold arms):
    - planarity of the 24 core heavy atoms (Pt, rings and O; hexyl chains excluded): RMS 0.006 Å in the reference against 1.07–1.09 Å generated;
    - angles between ring planes: 0–1° in the reference against 41–90°;
    - C–O–C: 124° against 106–110°;
    - donors out of the PtD₄ plane: 0.003 Å against 0.28–0.33 Å, so the coordination is tetrahedrally twisted;
    - top-1 dE: 677–702 kJ/mol in both arms, and the same in the stored GFN2 pools, so the cause is not the fallback embed.
  - **Earlier coverage was partial.** JEGPUJ was in the tail panel, in T3 (smaller angle effects) and in T4 (only 3–4 rows). The loss of planarity itself was not identified.
  - **Hypothesis (same class as S1 and S4).**
    - The free-ligand embed sets the chelate geometry without the metal: a propeller-twisted diaryl ether with C–O–C near the sp3 value.
    - Flattening it needs the C–O–C opened to about 124° and four coupled torsions turned together while three chelate rings close.
    - The solver keeps the embed's ring-to-ring frame and distorts the metal side instead.
    - Not established: whether the planar pose scores better under the current score and is never found (search), or scores worse (scoring).
  - **Diagnostic (CPU).**
    1. Measure the planarity and C–O–C of the free-ligand and chelate embeds, MMFF and GFN2.
    2. List the joints that can move the aryl–O and aryl–triazole torsions and the C–O–C angle.
    3. Score the reference pose with the u1 scorer, through the same mapping, against the top-1.
       - If the reference scores better, the problem is search or embed.
       - If it scores worse, name the terms that penalise it.
  - **Embed spot-fixes are rejected (owner, 2026-10-25).**
    - A dummy-metal or templated embed has been tried before. It helped some cases and damaged others, and its quality depends on how close the embed already is.
    - Not every structure embeds well under every algorithm.
    - The solver is meant to find the structure, and here it has the DOFs to do so: 16 rotatable bonds and 8 chelate-backbone bends. The S4 bite bound is withdrawn for the same reason.
  - **Replacement: the reachability triage (S6) applied to JEGPUJ, HOXJUZ and DAMVAQ.**

- **S6. Reachability triage: why the solver misses a structure it can represent** (owner principle: the IK solver finds the structure; embeds and spot fixes do not). For each tail parent and each parent whose regret is above 50 kJ/mol, run three steps on CPU:
  1. **Representability.** Fit the solver's DOFs (rigid body, torsions, bends, ring joints) to the reference by least squares from the actual embed, through the same atom mapping. Report the residual RMSD and the per-fragment residual. A large residual means the DOF set cannot express the structure: a missing DOF, or a frozen ring internal from the embed.
  2. **Scoring.** Score the fitted pose X_fit with the u1 solve objective and with the ranking score, against the top-1. Give the per-term breakdown.
  3. **Basin.** Run a local solve started at X_fit. Does it stay near X_fit, or drift to the top-1 basin?

  **Classification:**

  | Case | Meaning | Route |
  |---|---|---|
  | **R** | not representable | DOF set |
  | **S** | representable, but X_fit scores worse than the top-1 | score or prior balance (e.g. bend priors anchored to the embed's angles) |
  | **F** | X_fit scores better and stays, but the search never reaches it | search, workstream 2 (S-P1, S-P2, seed allocation) |
  | **D** | X_fit scores better but drifts away under the local solve | objective vs ranking mismatch (the solve objective is variance-floored, B2.7 option B) |

  The fixes are general solver changes chosen by class counts, never per-structure patches. S1 (HOXJUZ), S4(b) (DAMVAQ) and S5 (JEGPUJ) are the first cases.

---

## 3. Workstream 2: search coverage and seed dependence

### 3.1 Measured facts

**S-M1. Seeds are one embedding and many beam/initialisation draws.** `tools/jdev/multiseed.py:67-75` calls `prepare(builder, seeds[0])`, which embeds once. `seed_rows` (`multiseed.py:145-160`) re-seeds numpy and torch per seed and re-runs `_beam_search_all_candidates`. So the seed lottery is **not** an embedding lottery. Any ceiling for seed reallocation is therefore a ceiling for the beam and initialisation stage only.

**S-M2. Seed union gain.** `[D05]` (lottery, with the auditor's rule, 220 parents): mean single-seed floor 113.9 → union-of-three floor 107.7 (median 60.2 → 53.6). **40 parents gain > 10 kJ/mol and 2 gain > 50.** `[D01]` over the 232 three-seed parents: median seed range 9.4 kJ/mol; 23 parents have range > 50 kJ/mol.

**Ceiling.** The union of three independent seeds gives −6.2 kJ/mol on the floor mean. An equal-compute single-seed method cannot in general beat the union, so **−6 kJ/mol is an approximate ceiling for seed-budget reallocation at fixed compute**. (The bound assumes seeds are exchangeable; the union is not free.)

**S-M3. Assignment classes.** `[D08]`: 61.8% of pools (430 of 696) contain more than one assignment class; median 2 classes, maximum 57. The floor row shares the native top-1's class in 72.0% of pools and in 54.7% of multi-class pools. So in roughly 45% of multi-class pools, the best-energy row lies in a different class from the native top-1. The median number of classes within 20 kJ/mol of the floor in multi-class pools is 2.

**S-M4. Partial-geometry repulsion already exists.** The beam uses partial Q6 scoring, `ik/beam.py:132-137` (enabled when Q6 tables exist), `beam.py:198-201` (`_partial_q6_score` per expansion), and the engine passes the tables at `engine.py:2110-2112`. Roadmapv2 section 4.1 item 3 ("add cheap partial-geometry repulsive scoring") is therefore partly present. The proposal must be stated as a measured change to an existing mechanism, not a new term.

**S-M5. Beam ranking of modes.** `beam.py:189`: `BEAM_MODE_SCORE` is `"density"` (legacy) or `"mass"`. Under `"density"` the mode's log density at its mean sets the expansion score (`vmm_log_prob`, `beam.py:192`). Pruning keeps the top W unique prefixes by score (`beam.py:212-226`); duplicate prefixes keep the higher score (`beam.py:212-217`).

### 3.1b Additional measured facts (added in revision)

**S-M6. The torsion beam is target-blind.** `engine.py:2103-2105` runs `beam.search` once per ligand with `X_ref = X_full`, the free-ligand geometry (`engine.py:2012`), which does not depend on the candidate. The comment there reads "Torsion seeds are shared by all candidates". The seeds are tiled over candidates, coordination samples and beam slots (`engine.py:2127-2131`). Coordination targets enter only later, through `kabsch_init` (`engine.py:2167`), which builds the base pose of each row. Non-macrocycle ligands are the case described here (macrocycles use `X_rows`, `engine.py:2152`).

### 3.2 Target definition

The target is **equal-compute floor**, defined from the real row formula.

```text
rows_per_seed(arm) = n_candidates × M × W              (engine.py:2020-2021, B = B_per_cand × n_candidates)
                     with M = n_coord_samples = 8      (engine.py:1034, default)
                          W = beam_width = 16          (engine.py:1030, default; registry settings, refit/nci_v2_registry.py:349)
ik_steps(arm)      = n_ik_steps = 200                 (engine.py:1031, default; registry adam_steps 200)
compute(arm)       = Σ over seeds of rows_per_seed × ik_steps          (IK work)
                   + Σ over seeds of beam work, proportional to W per ligand   (beam work)
floor_arm(p)       = mean over seeds of the minimum valid dE (f, Section 1.4.1)
target: mean over parents of floor_arm(p), at compute(arm) within ±10% of compute(legacy)
```

The legacy arm is three seeds at `W = 16`. For a parent with `n_candidates` assignments, the IK rows per seed are `8 × 16 × n_candidates`, so `compute(legacy) = 3 × 128 × n_candidates × 200` in row-steps. Equal compute is matched by IK row-steps and by measured wall time within ±10% (Section 5.4).

### 3.3 Arms

| Arm | Definition | Purpose |
|---|---|---|
| `L` | Legacy: seeds 11–13, `W = 16`, flags off | Incumbent |
| `S1` | One seed (11), `W = 48` | Equal IK rows: `1 × 8 × 48 × n_cand = 3 × 8 × 16 × n_cand`. Tests one deeper beam against three seeds. Beam work is three times one beam run per seed; measure it. |
| `S2` | Seeds 11–13, `W = 16`, `beam_target_aware = True` | Torsion seeds that respond to each candidate's coordination geometry (S-P2) |
| `S3` | Seeds 11–13, `W = 16`, partial Q6 at full depth | Conditional on S-M4 measurements (S-P4) |

`S1` decides whether the lottery is worth keeping: if `S1` fails against `L`, seed diversity is worth its cost. `S2` tests the coverage hypothesis that the torsion seeds are target-blind (S-M6).

### 3.4 Proposals

**S-P1. Mode coverage measurement (measure before changing).** Count, for each joint of each ligand, how many beam rows fall within 0.2 rad of a mode mean of the stored Q5 table, and report supported modes with weight above 0.05 that receive no initial row. This is a CPU diagnostic on the S2b state once beam rows are stored; if they are not stored, the storage is added under `JACOBIK_TIMING_TRACE`. **Proceed to S-P2 only if** at least 10% of parents have an unfilled supported mode.

**S-P2. Target-aware torsion beams (flag `beam_target_aware`, default False).** This replaces the earlier per-class budget proposal, which is withdrawn: the code already gives every candidate its own `W` rows per coordination sample (`B = M × W × n_candidates`), and already has a per-class row budget for topologically equivalent assignments (`_class_budget_rows`, `engine.py:1780-1800`). What is not target-aware is the **beam** (S-M6).

Algorithm when the flag is on: for each candidate `c` and coordination sample `m`, compute the target-placed base pose `θ0(c,m)` by the existing `kabsch_init` on the free ligand with that candidate's donor targets, and set `X_ref(c,m) = fk.apply(θ0 with zero torsions, X_free)`. Run the beam once per `(c, m)`, in batches of `n_candidates × M`, with the same `W`. The seed tiling over `(c, m, slot)` is unchanged, and the row count per seed is unchanged (`B = M × W × n_candidates`). Beam work grows by the factor `n_candidates × M`, which must be measured.

**Test.** (i) Flag off: bit-identical to legacy (T-G2). (ii) Row count per seed identical to legacy (asserted). (iii) On a parent with two or more candidates whose targets differ, the torsion seeds of at least one candidate differ from legacy. **Acceptance.** Versioned mode: paired `cal` change in the floor with upper 95% bound below 0; no tail parent loss above 10 kJ/mol; beam wall time reported with its share of total time.

**S-P3. Seed reallocation (arm S1).** A runner argument `--seed-budget single_w48`, not a solver field. The orchestrator writes `seed_budget` and `W` into the experiment identity so arms are never confused. **Test.** IK row-steps of S1 equal L within ±10%. **Acceptance.** As in 3.2. **Expected effect.** Bounded by the ceiling of S-M2 (about 6 kJ/mol). Guess otherwise.

**S-P4. Partial-geometry depth (conditional on S-M4).** Partial Q6 already runs in the beam whenever the Q6 tables exist (`ik/beam.py:132-137`, `:198-201`; tables passed at `engine.py:2110-2112`). A full-depth variant is implemented only if S-P1 finds unfilled modes **and** the partial Q6 is shown to be shallow, measured as the fraction of expansions in which the ligand has no Q6 pair and the mean fraction of placed atoms at evaluation. Otherwise it is not implemented.

### 3.5 Tests common to all search arms

- **Row-count identity** per seed (asserted at run time, recorded in the pool).
- **Pool schema identity:** the NPZ keys and dtypes are unchanged; new keys are additive.
- **Golden:** `tools/jdev/golden.py check` must print `golden: OK` with all flags at legacy values.

### 3.6 Risks and interactions

Any change to the beam or seed rows changes the generator identity. Roadmapv2 section 4.1 requires a new pose-library identity and an NCI transfer check before the fitted denominator is reused. Section 8 lists this as a gate.

## 4. Workstream 3: embedding

### 4.1 Measured and cited facts

- The free-ligand embed is `engine.py:575-600` (`ETKDGv3` with fixed `randomSeed = 42` by default, `embed_seed` when given; a second `ETKDG()` retry on failure at `:583-585`). The chelate-aware bounded embed is `chelate_embed.py:364` (`embed`), with violation-based retries at `:405-422`.
- `bound_surrogate` (`chelate_embed.py:163`) and `cap_anions` (`:272`) are the existing group-13 and anion surrogates. `eta2_cap` (`chelate_embed.py:53`) is off in `u1`.
- **Cited (roadmapv2 section 4.2).** A broad cap-retention screen failed generality (18 training ligands: bond-RMS better/worse 11/7; donor-angle RMS better/worse in 4 of 12 cases). Any new cap proposal must name a chemical mechanism and differ from that screen.
- **Measured (this document).** NEPNEF's embedded or solved N–O bond length is 1.48–1.52 Å against 1.21 Å (T6); `[D03]`.

### 4.2 Proposals

**E-P1. Bond-order and charge fidelity check (flag `embed_bond_check`).** Algorithm: after embed, for each covalent bond with a bond-order-table entry (table keyed by element pair, order, and formal charges), compute `|d − d_table|`; reject the embed if any exceeds 0.15 Å, retrying with the next seed up to `N` times (`N = 4`). **Test.** NEPNEF embed passes; a synthetic molecule with a deliberately wrong order is rejected. **Acceptance.** `cal` parents: no increase in embed failures; NEPNEF floor drops (T6). **Cost.** One distance check per embed.

**E-P2. Chelate ring-fold diversity (conditional).** Only if the T1/T3 diagnostics show that solver search cannot reach a fold that the reference has. Measured first: the reference ring-fold class (Section 6.1) versus the set of embedded folds for each parent in the `failures` panel. **Acceptance.** An arm that adds folds must keep the row-count budget and report added-fold coverage separately.

**E-P3. ETA2 cap (existing flag, tested here).** `eta2_cap = True` on ZELTAP and the `cal` parents with η² donors, versus legacy. The mechanism is the existing valence cap plus shared CH2 bridge (`chelate_embed.py:50-53`). **Acceptance.** Paired on `cal` η² parents; no parent loss > 10 kJ/mol; ZELTAP intact share unchanged or better.

**E-P4. Lanthanide site distances (T5).** A metal-family table for donor distances, versioned in `spec`, applied only to 4f metals. Its validity is the D-T5 diagnostic; no change to the embed itself.

### 4.3 Metrics for every embedding arm

Validity (independent graph and state checks, `benchmarks/EVALUATION_DESIGN.md` style), bond RMS to reference, donor-angle RMS, embed failure count and path (bounded, unbounded fallback, rejected), pool floor and top-1, and cost. Geometry improvements alone do not establish generation benefit (roadmapv2 section 4.2). An embedding change requires new baseline poses and an NCI transfer check.

### 4.4 Tests and the matched-output test

**Matched-output test (defined once; used by every flag and every performance change).** The stored Step 2 pools cannot serve as the reference, because the GPU path is not bit-reproducible (roadmapv2 section 9.8). The reference is the CPU golden set plus a fixed CPU sample:

- **(a) Golden check.** `python tools/jdev/golden.py check` must print `golden: OK`. It compares CPU solves with `data/golden/<profile>.npz` for the profiles `legacy-2026-10` and `u1`: 11 rescored pools (`RESCORE`, arm `etkdg_u1_rep` of the budget tree) and 7 solves at seed 11 with `n_ik_steps = 2`, `n_conformers = 1`, `device = "cpu"` (`SOLVE`). Tolerances: scores and terms 1e-6 nats, coordinates 1e-5 Å (`tools/jdev/golden.py:32`, `:91-95`).
- **(b) Fixed CPU sample S20.** The first 20 calibration parents of the frame's `calibration` split, in order of `group_id`. The list is written to the bench output before any run. Seed 11, CPU, `n_ik_steps = 200`, default `n_coord_samples` and `beam_width`. Recorded: the row count per candidate, the assignment indices, and the per-row energies of the final pool. Compared at the same tolerances (1e-6 nats, 1e-5 Å, identical row count and assignment indices). The recording is made once per commit. Its run time on two parents is measured before S20 is adopted.

A flag at its legacy value must pass (a) and (b). A performance change must reproduce (a) and (b); a change that alters any output of (b) is not a performance change (Section 5.4).

- **Golden:** flags at legacy values pass (a) and (b).
- **Embed determinism:** `deterministic_embed` (`profile.py`; `u1` sets it true) must be on for every embed arm (`engine.py`, `DETERMINISTIC_EMBED`).

## 5. Workstream 4: solve performance

### 5.1 Measured stage profile

`[D06]`, from the stored `stage_*_s` fields (means over pools; shares are of summed stage time):

| Stage | Eval calibration mean (s) | share | Eval validation mean (s) | share | Fit mean (s) | share |
|---|---:|---:|---:|---:|---:|---:|
| `stage_solver_s` | 4.63 | 0.296 | 5.14 | 0.314 | 5.81 | 0.497 |
| `stage_dedup_s` | 1.13 | 0.072 | 2.02 | 0.123 | 1.98 | 0.170 |
| `stage_embed_s` | 1.18 | 0.076 | 0.76 | 0.046 | 1.15 | 0.098 |
| `stage_seed_rows_s` | 0.96 | 0.062 | 0.88 | 0.054 | 0.94 | 0.081 |
| `stage_mace_s` | 2.12 | 0.135 | 2.16 | 0.132 | 0.00 | 0.000 |
| `stage_other_s` | 5.61 | **0.359** | 5.43 | **0.331** | 1.81 | 0.155 |
| `wall_s` | 15.64 | 1.00 | 16.39 | 1.00 | 11.70 | 1.00 |

Fit pools have no MACE call, so their `other` is 1.81 s. Evaluation pools have `other` of 5.4–5.6 s, which is **the largest single stage on evaluation**, and it is unattributed.

**`stage_other_s` definition (measured from source).** It is the residual `wall_total − (embed + seed_rows + solver + dedup + mace)` (`refit/nci_v2_orchestrator.py:815-823`). `wall_total` is measured from `t_start` (`refit/nci_v2_orchestrator.py:516`, set before the solve) to just before the pool hash (`:795-796`). So `other` contains everything not in the named stages: the atom mapping (`build_atom_mapping`, `:709`), `coordination_integrity` (`:~717`), the NPZ compression (`save_pool_npz`, `refit/nci_v2_pilot.py`), the sha256 of the conformer array (`:796`), per-stage JSON writing, and any MACE-related post-processing on evaluation pools. **The evaluation-only excess of about 3.7 s is not attributed.** Candidates are the MACE energy mapping and the coordination integrity run, which execute only on the evaluation split. Attribution is the first task of this workstream.

### 5.2 Instrumentation (proposal P-I, execution-only)

With `JACOBIK_TIMING_TRACE=1`, add timers around: atom mapping, coordination integrity, MACE result handling (not the MACE call itself), NPZ save, JSON write, sha256, each dedup call, and the candidate counts (pre-window, post-window, post-dedup, rows per class, count of `_repair_infeasible_baseline` events, count of embed path types). Keys are additive to the pool JSON. **Test.** The matched-output test of Section 4.4 passes with the trace on and off. **Acceptance.** The residual `other` after the new timers is < 5% of wall time on a 20-pool sample.

### 5.3 Dedup (`src/jacobik/ik/dedup.py`)

**What the code does (measured from source).** `StructureDeduper.select` (`dedup.py:335`) sorts rows by score, then keeps a row unless it is an automorphism-equivalent duplicate (RMSD ≤ `DEDUP_RMSD` = 0.5 Å over heavy atoms and metal) of a kept row. The exact path has two branches: a block branch when the automorphism count `M ≤ BLOCK_MAX_MAPS` (`dedup.py:51`, value 48), and a row-by-row branch (`_full_any`) when `M > BLOCK_MAX_MAPS`. A non-exact variant (`exact=False`, `dedup.py:346`) changes the selection and is not a silent option. The device defaults to CPU (`DEDUP_DEVICE`, `dedup.py:46`).

**Measured cost.** `[D06]` `stage_dedup_s` correlates with the stored automorphism count: Spearman 0.50–0.58 across pool sets. The largest outliers are not explained by `M` alone:

| Pool | Stage | Rows (n_confs) | Automorphisms `M` | dedup s |
|---|---|---:|---:|---:|
| PAJCAE (fit, seeds 11–13) | fit | 192 | 28,800 | 539.0 |
| NADYUO (fit) | fit | 192 | 82,944 | 147.9 |
| SIZYOP (validation) | validation | 96 | 9,216 | 66.2 |
| SOBHAS (validation) | validation | 224 | 3,840 | 64.6 |
| XAQCAT (calibration) | calibration | 48 | 3,158,028 | 14.8 |

XAQCAT has the largest `M` and is not slow, so the cost is not monotone in `M`. A plausible driver is the number of candidate pairs that pass the radial prefilter (`_radial` at `dedup.py:90`) and then need the full automorphism scan, multiplied by kept rows. **Not verified.**

**P-D1. Measure before changing.** Instrument the number of pairs passing the radial prefilter and the number of automorphisms scanned per pair, on PAJCAE, NADYUO, SOBHAS, XAQCAT. Pass: scanned pairs × `M` accounts for > 80% of dedup time on at least two outliers.

**P-D2. Exact prefilter (flag `dedup_aut_prefilter`).** A lower bound on the minimum RMSD over automorphisms, computed from sorted per-atom radial distances, skipping pairs whose bound already exceeds `DEDUP_RMSD`. The bound must be a true lower bound (proof in code comments and a unit test that it never exceeds the exact minimum over random pairs). **Output identity:** the matched-output test of Section 4.4 (a) and (b). Any difference in selected row indices is a failure.

**P-D3. Non-exact variant.** Not proposed. If P-D2 is insufficient, a non-exact variant is a separate arm with its own output identity report, because it changes the pool.

### 5.4 Acceptance for performance changes (matched outputs required)

A performance change is accepted only when, on the inputs of the matched-output test of Section 4.4:

1. the golden check (a) passes, and the sample S20 (b) reproduces its row counts, assignment indices and per-row energies at the golden tolerances; and
2. the per-stage wall time drops by the claimed amount over at least 10 repeats, with the repeats recorded.

Any change that alters outputs is a generation change and follows Section 3.5 and Section 8.

### 5.5 GPU utilisation and batching

Not measured in this document (no GPU use, per the task rules). **Proposal P-G1.** During the batch (Section 8), sample `nvidia-smi` utilisation at 1 Hz across one control parent, record mean and p95, and compare with the stage shares of 5.1 to decide whether batching the dedup or embed stages can help. Batching changes numerical order of operations and must pass the matched-output test of 5.4.

## 6. Workstream 5: deduplication, isomer classes, and thermal ensembles

### 6.1 Operational definitions (each from data the code already has)

| Class | Definition | Source in code or data |
|---|---|---|
| **Assignment class** | The integer `asg` stored per row. Two rows are in the same assignment class iff their `asg` values are equal. | `pool.npz asg`; generated by `site_enumeration.enumerate_assignments` (`ik/site_enumeration.py:173`) and `enumerate_unique_assignments` (`:444`), whose `topological_class_keys` (`:405`) define symmetry classes. |
| **E/Z class** | The vector of E/Z labels set by `set_ez` (`chelate_embed.py:214`) for the embedded variant. Rows from different `embed_idx` values belong to different E/Z classes. **Data availability (verified):** the Step 2 control pools store no `embed_idx` and no `trial_idx`, so the E/Z class is **not computable on Step 2 pools**. The S2b pools store `embed_idx`, but it is `−1` on every row of HOXJUZ, ZELTAP, YUCZUS and SOBHAS, so those pools have one embed each and the class does not partition their rows. No stored pool has several embed variants per pool, so an E/Z class is computable only on pools regenerated with multi-variant embeds. | `ez_variants` (`chelate_embed.py:137`); profile `ez_enumerate` (`u1`: on). |
| **Haptic mode** | For each metal, the sorted tuple of the sizes of its haptic donor groups (`coordination_integrity` groups, `nci_v2_pilot.py:267`, singles are sigma). | Reference-derived in evaluation; production uses the declared sites. |
| **Stereocentre set** | The set of ligand atoms with a specified stereo label after `AssignStereochemistry` on the builder molecule. Not stored in the pool; **proposal**: store the sorted atom-index tuple in the pool JSON under `JACOBIK_TIMING_TRACE`. | RDKit on the builder mol (not verified here). |
| **Ring-fold class** | For each ring in the ligand graph, the sign pattern of its ring torsions (a dihedral sign vector), taken from the coordinates; two rows share a fold class iff their sign vectors are equal for all rings. | Coordinates; the ring list from `topology.decompose` (`RING_JOINTS`). |

These are operational definitions. A class is defined by the data it is computed from, not by an external chemical label. Where a class is not stored, the proposal states exactly what to store.

### 6.2 Class-aware dedup (flag `dedup_class_aware`)

**Measured context.** `[D08]` 61.8% of pools have more than one assignment class, and the floor is in another class from the native top-1 in about 45% of multi-class pools. The current dedup merges rows across classes if their RMSD ≤ 0.5 Å. Two rows in different classes should not be within 0.5 Å (different donor placements), but this is not measured. **Measurement first (D1):** count pairs of stored rows with different `asg` and heavy-atom RMSD ≤ 0.5 Å, per pool, on the `cal` panel. If that count is zero for all pools, class-aware dedup is a no-op and the flag is not needed; the document says so and the flag is dropped.

**Algorithm if D1 is nonzero.** Partition rows by `asg`; dedup inside each partition; then take the union and re-sort by score. Output size may grow.

### 6.3 Ensembles

**What a thermal ensemble can mean here.** The stored energies are MACE electronic energies of rows (no vibrational, entropic, solvent, or free-energy terms; the score is an ordinal surrogate, roadmapv2 section 2.4). A thermal ensemble is therefore a **Boltzmann distribution over electronic energies of the stored rows at kT = 2.479 kJ/mol** (298.15 K). It is not a free-energy ensemble and not a population of the complex.

**Measured** `[D08]`: the electronic-energy effective number of rows, `N_eff = exp(−Σ p ln p)` with `p_i ∝ exp(−dE_i/kT)`, has a median of **1.50** over 696 pools (1.59 over pools with at least 3 rows). The Boltzmann distribution is therefore dominated by one row in most pools. **Consequence:** an ensemble metric over stored rows is nearly a single-state metric. It can say that a pool is near-degenerate only when `N_eff` is large, and it cannot say anything about basins the solver did not reach.

**Metrics allowed:** `N_eff`; class populations (Boltzmann sums per `asg`); the fraction of probability in the floor's class; the number of classes within 10 and 20 kJ/mol of the floor. **Not allowed:** any statement about free energy, entropy, or temperature-dependent populations.

### 6.4 Tests

- **D-Dup-1.** Dedup with `dedup_class_aware = False` is bit-identical to legacy.
- **D-Dup-2.** Two synthetic rows with different `asg` and RMSD 0.2 Å are both kept under class-aware dedup and one is kept under legacy.

## 7. Workstream 6: non-tmQM tests

### 7.1 What exists (measured from the repository)

- **Challenge panel** (`benchmarks/CHALLENGE.md`, `benchmarks/data/challenge/cases.json`, `validation_report.json`): 29 targets, 39 ligand definitions, 12 papers. Adapter status: 25 `direct`, 3 `requires_adapter` (H04–H06, needing richer haptic adapters), 1 `assembly_only`. Origins: 14 designed, 8 literature inputs, 7 literature motifs. Reference coordinates: 1 experimental (D03, Zr-HOPO, graph-mapped, 55 heavy atoms) and 12 published computed geometries (B03 and B05, three arrangement families each). The challenge panel states that Linux Architector execution has **not** yet been run, so there are no comparator numbers.
- **Baseline complexes** (`benchmarks/CATALOG.md`): 40 cases labelled S (control), A (literature anchor), D (designed target), with `--suite baseline`.
- **Evaluation design** (`benchmarks/EVALUATION_DESIGN.md`): separate questions for build validity, structure recovery, self-ranking, and cost, with no combined score. This is the protocol for this workstream.
- **Sources.** `benchmarks/sources/` holds CIF files (A01–A08 are listed at the head of the directory) and `ARCHITECTOR_LICENSE.txt`. Use of these CIFs is subject to the license file; this document does not change their status.

### 7.2 Feasibility

| Test set | Reference available | Feasible now | Gap |
|---|---|---|---|
| Challenge `direct` (25) | Chemical checks (RDKit and Open Babel); no energies | Build, validity, own ranking | No reference energies; no comparator |
| Challenge D03 | Experimental, graph-mapped | Structure recovery (RMSD to crystal over mapped heavy atoms) | One case only |
| Challenge B03, B05 (12 computed geometries) | Published computed; method not re-checked here | Recovery against each family | Energy level of the published geometries unknown; compute with MACE (GPU batch) before using as energies |
| Baseline 40 (S/A/D) | Crystal references for A-cases in `sources/` | Recovery and validity | Some cases may be outside the element set of MACE-POLAR-1-S; check per case |
| H04–H06 (haptic) | Needs adapter | No | Adapter is a prerequisite |

### 7.3 Protocol

1. **Freeze inputs.** Record the SHA-256 of `cases.json`, `ligands.json`, and the chosen reference files. The challenge panel's `validation_report.json` must match.
2. **Run with the control profile** (`u1`, NCI off as the research control; `u1` as the incumbent). Two arms, identical seeds 11–13.
3. **Report separately** (EVALUATION_DESIGN): build validity (graph, charge, electron parity, donor identity, site bookkeeping); recovery where a reference exists (per-case RMSD over mapped heavy atoms and donor distance errors); self-ranking (top-1 vs floor within the pool, without a reference energy); cost (wall, memory).
4. **No combined score.** Reference energies for the 12 computed geometries are needed before any energy statement.

### 7.4 Acceptance

Structure recovery and build validity are reported with uncertainty by case. A non-tmQM result is inconclusive unless each claim names its reference and its level of theory. This panel is a **generalisation check**, not a promotion gate, until the 12 computed-geometry energies are computed with MACE-POLAR-1-S float64 in the batch.

## 8. Dependencies, interaction with NCI v2, and batch plan

### 8.1 Dependency order

| Order | Item | Needs GPU? | Depends on |
|---|---|---|---|
| 0 | P0 fingerprint invariance (T-P0) | No | none |
| 1 | P-I timing instrumentation (5.2) | No (code); yes to measure | P0 |
| 2 | D-T3, D-T5, D-T6, D-T9 CPU diagnostics | No | none |
| 3 | P1 `intact_site` and its validation on stored pools | No | Q1 lookup name confirmed |
| 4 | P2 `rank_site_intact` | No | P1 |
| 5 | P7 mapping test (T7) | No | none; coordinate with NCI health policy |
| 6 | P3 `ring_joint_prior` (conditional) and the `branch_prior` control arm | Yes (validation) | D-T3 steps 2-3 |
| 7 | P6 / E-P1 `embed_bond_check` | Yes (validation) | D-T6 |
| 8 | P-D1 dedup measurement, then P-D2 | No | P-I |
| 9 | S-P1 mode coverage measurement | Partly | S2b beam-row storage |
| 10 | S-P2 target-aware beams (`beam_target_aware`) | Yes | S-P1 |
| 11 | S-P3 seed reallocation | Yes | none (runner) |
| 12 | D1 class-aware dedup measurement | No | none |
| 13 | Non-tmQM challenge (7) | Yes (computed energies) | protocol frozen |

### 8.2 Interaction with NCI v2 (roadmapv2)

- **Any generator change creates a new pose-library identity.** Changes to the beam (S-P2), bend penalty (P3), embed (E-P1, P6), mapping (P7), dedup (P-D2, class-aware), float precision (P9), and targets (P5) all change the pool. Per roadmapv2 section 4.1, the fitted NCI denominator must be re-checked for transfer before reuse.
- **The S2b state changes if the bend penalty changes** (P3 adds penalties at new angles). The S2b `bend_columns` schema is then a new version.
- **The health policy (roadmapv2 section 9.8)** depends on coordination integrity and mapping. P7 and P1 change health labels, so they change which poses the NCI fit uses. These are **not** energy changes but they change the NCI training set, and must be reported separately.
- **Control-vs-control repeat (roadmapv2 section 9.8).** Every arm comparison in this document requires it (Section 1.4.4).

### 8.3 Batch plan

Resource rules: one GPU job at a time; RAM within limits; no environments under `/tmp`; pixi Python; kill only by PID.

- **B0 (CPU; done in this document).** `d01`-`d13` and `poolio.py` are committed under `tools/jgen/diag/`. Done: the repeatability reference `[D12]`, which replaces the control-vs-control repeat for cal and val.
- **B1 (CPU).** Implement the bench (Section 1.4) with the noise band of 1.4.3 and self-check it: the frozen control compared with itself must give zero change, and compared with S2b must reproduce the band of `[D12]`. Implement the T-P0 fingerprint fix. Validate P1 (`intact_site`) on the stored effective pools after resolving the key discrepancy of Section 1.5. Run the CPU diagnostics still pending: D-T3 steps 2 and 3, D-T5, D-T6, D-T9, and the T7 builder test.
- **B2 (GPU; only for new panels).** A control-vs-control repeat is required for any panel not covered by `[D12]`. The reserve panel of B4 is the first such panel.
- **B3 (GPU).** Arms for the flags that passed B1, in this order: `rank_site_intact` (a re-solve is needed only where the top-1 changes), `embed_bond_check` (NEPNEF first), `branch_prior = True` (control for hypothesis b), `ring_joint_prior` (only if D-T3 step 3 supports H-T3c), `beam_target_aware` (S-P2), and S1 (S-P3). Each arm runs on the tail panel, then on `cal`. The `val` split is not used here (see the split rule).
- **B4 (GPU; reserved; once).** The **reserve panel** is the frame's reserve split (`frame.json`, `main_allocation.splits.reserve`, 240 groups), minus all 20 replacement codes (`repair_manifest.json`, `slots[*].replacement_code`), minus any group with a member in a manifest task. That leaves **220 groups**. The panel is drawn stratified by `stratum`, with the seed written into the contract before B3. MACE runs as part of B4 (float64, exact charge and multiplicity, roadmapv2 MACE policy). B4 runs once, after the `cal` gate of B3 passes.

**Split rule.** `val` is NCI's `motion_validation` split. Every diagnostic in this document has already read it (the tail and failure parents are drawn from `cal` and `val`), so it is no longer unseen. Generation work may evaluate on `val` **only once**, as a confirmation after `cal` passes, and must never tune on it. The reserve panel is the only panel for promotion.

**Stop rules.** A failing gate at any stage closes that flag as inconclusive or fail (Section 9). A failing diagnostic that refutes its hypothesis closes its proposal without a flag.

### 8.4 Statistics DB, NCI v2 and input labels (revision 2)

- **Surface-fit prevalence.** 934 consistent, 209 ambiguous and 57 inconsistent parents in 1200 (Section 2.0). About 22% of surface-fit parents carry a non-consistent label. Their statistics enter the NCI v2 fit.
- **Q1 key contamination (estimate, not a measurement).** The constraints DB (`data/refit/final/constraints_2026-10-18.json`) has no per-key provenance, so contamination cannot be read from it. d21 estimates, per donor subclass, how many surface_fit parents carrying that subclass are inconsistent or ambiguous. It says how many training parents have a defective SMILES. It does not say which DB key each parent fed or with what weight. Result (d21, surface_fit, 1200 parents, 57 inconsistent): the highest inconsistent shares are N_donor:N_amine 17.2% (17 of 99 parents), S_donor:S_thiocarbonyl 15.2% (7 of 46), N_donor:N_amido 10.7% (8 of 75), O_donor:O_alkoxide 8.0% (4 of 50), Hal_donor:Hal_I 7.0% (3 of 43). The largest absolute counts are P_donor:P_phosphine 15 of 303 (5.0%) and Hal_donor:Hal_Cl 11 of 382 (2.9%). The estimate is per subclass, not per DB key (see Section 8.4 caveat).
- **Consequence for NCI v2.** Q1 keys with a high non-consistent share should be refit with those parents excluded or flagged, under roadmapv2 section 4.1. Changing the labels changes the training set and is reported separately from energy changes (Section 8.2).

## 9. Evidence limits and undetermined items

**Limits of the measurements.**

1. The control arm is not reproducible run to run. The repeat `[D12]` measures this on cal and val: per-parent noise p95 0.65 kJ/mol, and 7 of 696 units move by more than 10 kJ/mol between runs. Changes below the band cannot be interpreted; changes above it need the heavy-tail check of Section 1.4.3.
2. The score identity is resolved (M-5, M-6): the bend prior is in score units. Fourteen S2b units carry a further term of up to 0.04 score units that is not stored; its identity is not verified.
3. `intact_ref` uses the reference geometry and is an evaluation label. Section 2 uses it only for description and for the selection oracle M-1.
4. Scripts `d03`, `d04`, `d05`, `d11` and `d13` read raw arrays on tail and failure parents. Those parents carry R3 overlays only, so this equals the effective pool (the repair manifest is the check).
5. The pools carry no `embed_idx` (Step 2) or `embed_idx = −1` on every row (S2b tail parents), so E/Z classes are not computable (Section 6.1).

**Items not determined in this document, and why.**

- **Joint-ring drop per fold ring** (2.5): d22 counts folds after `angle_bends.py:353-359` but does not record the drop per ring.
- **Haptic reference fold convention** (2.5): the 180.0° values are unverified.
- **Fold prior F-P1** (2.5): free zone and stiffness are untested; reference fold angles are measured for only three parents.
- **Cause of score AUC 0.49 on consistent versus inconsistent parents** (2.0): mostly by design. The score judges structures against the SMILES as written (a faithful structure from a wrong SMILES scores well); fold hinges let the solver hide input conflicts at no cost (HOXJUZ score +0.4 -> -1.85 without them); the absolute score is size-dependent (Spearman -0.63 with atom count), and size adjustment only moves the AUC to 0.53. Measured 2026-10-25 (scratch `score_auc.py`).
- **NEPNEF mechanism** (T6): D-T6 (RDKit) not run. The bond-length observation is measured; its cause is a hypothesis.
- **Angle attribution beyond roles** (T3): D-T3 step 2 (the joint-to-angle mapping) is required and not done. The role split is measured; the joint-type split is not.
- **IRAQUO cause** (T7): the mapping error is measured in all rows; whether the pairing or the bond graph is at fault needs the builder test.
- **Lanthanide donor cause** (T5): cited from the S2.1 report; D-T5 not run.
- **ZELTAP float32 sensitivity** (T9): cited; the auditor's scratch scripts (`scratchpad/zeltap.py`, `zeltap64.py`) are outside the repository and must be copied into `tools/jgen/diag/` before reuse.
- **BODCAY, LIYTAQ, ADINOR, OBABAX** (T8): reproduced in revision 2 (Section 2.0). BODCAY and LIYTAQ raise `Bad Conformer Id` at `engine.py:588`; OBABAX completes CPU setup. ADINOR not reproduced.
- **`stage_other_s` attribution** (5.1): not attributed; instrumentation is proposed.
- **Dedup cost driver** (5.3): not identified; P-D1 measures it. XAQCAT (3.2 million automorphisms) is not slow, so the automorphism count is not the driver.
- **Thermal ensembles** (6.3): electronic only. No free-energy statement is supported by these data.
- **Non-tmQM energies** (7.2): no reference energies for the challenge panel; the 12 computed geometries need a MACE recomputation.
- **Equal-compute approximation** (3.2): assumes the candidate count per parent is fixed across arms and that beam cost is proportional to W; both are to be measured.
- **Q1 key discrepancy** (1.5): the solver key (with `trans_p`) and the scorer key (`q1_site_keys`) differ; which one governs each row's ranking must be checked before P1.
- **Solver precision of the stored pools** (T9): not stated in the pool JSON.
- **Score-minus-rank term in 14 S2b units** (M-5): up to 0.04 score units, not attributed.

## Appendix A. Reproduction of every measured number

All commands run from the repository root or this worktree, with:

```text
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
PY=/mnt/e/Dev/Projects/jacobik_2026-10-02_full/.pixi/envs/linux/bin/python
ROOT=/mnt/e/Dev/Projects/jacobik_2026-10-02_full/data/refit/nci_v2/scaled_v1
```

Scripts read effective pools (`poolio.py`) unless stated. Tail parents: DAMVAQ, HOXJUZ, IFUROQ, JEGPUJ, NEPNEF, SOBHAS, USAPOU, XEYTUT, YUCZUS, ZELTAP. `OUT` is any writable directory outside `data/`.

| Tag | Number(s) | Command |
|---|---|---|
| D01 | 696 pools; top-1 140.9 / 77.5 / 272.8; floor 112.8 / 60.1 / 239.5; regret 28.1 / 4.8 / 55.6; 0.487 within 4.184; pools > 100/200/500/1000 = 273/116/30/12; capped mean 114.1; 232 parents; seed range > 50: 23, > 20: 72; median seed range 9.4; 23,050 stored rows; effective top-1 intact 0.845 (all), 0.60 (> 500), 0.934 (≤ 100) | `$PY tools/jgen/diag/d01_pool_scoreboard.py --data $ROOT --out-json OUT/d01.json` |
| D02 | Per-tail top-1 and floor rows, dE, assignment, intact flags; ρ(score, −dE) and ρ(rank, −dE) | `$PY tools/jgen/diag/d02_tail_terms.py --pools $ROOT/pools --root $ROOT --codes DAMVAQ,HOXJUZ,IFUROQ,JEGPUJ,NEPNEF,SOBHAS,USAPOU,XEYTUT,YUCZUS,ZELTAP` |
| D03 | Covalent bond RMS per tail parent; NEPNEF 0.131 Å on all rows | `$PY tools/jgen/diag/d03_bond_distortion.py --pools $ROOT/pools --codes DAMVAQ,HOXJUZ,IFUROQ,JEGPUJ,NEPNEF,SOBHAS,USAPOU,XEYTUT,YUCZUS,ZELTAP` |
| D04 | Bonded-angle RMS of top-1 and floor; ρ(angle, dE); clash count 0; NEPNEF O–N 1.518 vs 1.206 Å | `$PY tools/jgen/diag/d04_angles_clashes.py --pools $ROOT/pools --codes <as D03> --bond-report NEPNEF` |
| D05 | Lottery on 220 parents (R1 excluded): 113.9 → 107.7; median 60.2 → 53.6; 40 parents gain > 10, 2 gain > 50 | `$PY tools/jgen/diag/d05_iraquo_mapping_and_lottery.py --pools $ROOT/pools --manifest $ROOT/manifest.json --repair $ROOT/repair_s2_1/repair_manifest.json` |
| D06 | Stage means and shares (5.1); dedup ρ with automorphisms 0.50–0.58; outliers PAJCAE 539.0 s, NADYUO 147.9 s, SIZYOP 66.2 s, SOBHAS 64.6 s, XAQCAT 14.8 s | `$PY tools/jgen/diag/d06_stage_timing.py --pools $ROOT/pools --manifest $ROOT/manifest.json` |
| D07 | Bend-prior values, top-1 vs floor, ZELTAP, SOBHAS, HOXJUZ (score units) | `$PY tools/jgen/diag/d07_bend_prior_state.py --root $ROOT --codes ZELTAP,SOBHAS,HOXJUZ` |
| D08 | 430/696 multi-class pools; median 2 classes, max 57; floor in the top-1's class 0.720 (0.547 for multi-class); N_eff median 1.62 (1.71 for ≥ 3 rows); 0.397 rows in another class | `$PY tools/jgen/diag/d08_classes_and_ensemble.py --root $ROOT` |
| D09 | Oracle on effective pools: 140.9 → 140.3; 108 non-intact top-1; 29 with an intact row; mean −15.1, median −6.9 | `$PY tools/jgen/diag/d09_intact_filter_oracle.py --root $ROOT` |
| D10 | rank × Z = Σ terms (max 2.2e-5); argmax(score) = argmax(rank) in 0.895 of pools; top-1 mean 140.9 by score, 142.5 by rank; S2b: score − rank = bend_prior_solver in 697/711 units | `$PY tools/jgen/diag/d10_score_rank_check.py --root $ROOT --s2b $ROOT/s2b_eval_state_v1` |
| D11 | IRAQUO mapping table, all rows and seeds (two bad pairs per seed, on C16 and C39); gen_to_ref_idx differs between seeds | `$PY tools/jgen/diag/d11_iraquo_table.py --pools $ROOT/pools` |
| D12 | S2b vs Step 2 repeat on 696 units: abs Δtop-1 median 0.0002, p95 0.263, > 1: 25, > 5: 14, > 10: 7, max 164.4 (LEKXIK s12); abs Δfloor p95 0.301, > 10: 7; per-parent p95 0.648; 15 S2b-only pools | `$PY tools/jgen/diag/d12_s2b_repeatability.py --root $ROOT --out-json OUT/d12.json` |
| D13 | Moving angles per tail pool by role; top-1 minus floor sum of squared error by role (T3) | `$PY tools/jgen/diag/d13_angle_motion_attribution.py --s2b $ROOT/s2b_eval_state_v1/pools --codes HOXJUZ,ZELTAP,YUCZUS,SOBHAS` |

`d13` requires `networkx`, present in the pixi environment.

## Appendix B. Tests named in this document

| ID | Test | Section |
|---|---|---|
| T-P0 | Fingerprint of `u1` unchanged; one flag change alters it | 1.2.2 |
| T-G1..G3 | Flag defaults, golden identity, unknown-field rejection | 1.2.4 |
| P1-V | `intact_site` vs `intact_ref` agreement ≥ 95% on rows where `intact_ref` is defined | 1.5 |
| P2-R | Reproduces `d09` to 0.1 kJ/mol; no-intact-row fallback exact | 2.4 T2 |
| P3-A | `ring_joint_prior` weight 0 bit-identical; finite-difference gradient 1e-6 relative at 20 poses | 2.4 T3 |
| P3-B | ZELTAP top-1 ring-angle RMS within 2° of floor's, weight > 0 (single solve) | 2.4 T3 |
| P3-C | `branch_prior = True` changes the donor-centred moving-angle count on at least one tail parent | 2.4 T3 |
| P6-A | NEPNEF embedded N–O within 0.05 Å of 1.21 Å | 2.4 T6 |
| P7-A | IRAQUO mapped H–C ≤ 1.25 Å both carbons, all seeds | 2.4 T7 |
| S-P2-A | Row count identical per seed; flag off bit-identical; seeds differ for candidates with different targets | 3.4 |
| MO-1 | Matched-output test: golden check (a) and sample S20 (b) at golden tolerances | 4.4 |
| P-D2-A | Selected row indices identical to legacy | 5.3 |
| D-Dup-1, D-Dup-2 | Class-aware dedup identity; synthetic split | 6.4 |
| E-P1-A | Synthetic wrong-order bond flagged, not rejected; NEPNEF passes | 4.2 |
| D14 | Input labels (v3), prevalence, flag counts | 2.0 |
| D15, D16 | HOXJUZ stage trace; zeroing bend columns restores 1.74° ring deviation | 2.5 |
| D17 | Label scoreboard | 2.0 |
| D18 | Failed-parent repro (BODCAY, LIYTAQ raise; OBABAX setup completes; ZAZREZ shell error) | 2.0 |
| D19 | SMILES-only status, AUC 0.55 | 2.0 |
| D20 | Reference-based ring deviation by label | 2.0 |
| D21 | Q1 subclass contamination estimate (surface_fit) | 8.4 |
| D22 | Solver-shell fold prevalence and reference fold angles | 2.5 |
