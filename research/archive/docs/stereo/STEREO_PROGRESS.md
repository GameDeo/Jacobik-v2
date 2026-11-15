# STEREO_PROGRESS (jgen/i-stereo, directive 2026-11-09)

## 2026-11-09 checkpoint (a)
- `ik/stereo.py::superset_variants`: class list = every production E/Z combination (`Engine._ez_combinations`, same order,
  solved with ez_fix only, non-strict embed, same seed) + capped extras from the new stereo elements.
- Extras: ligand tetrahedral centres, binding-created donors, E/Z bonds production does not enumerate. M-C-X rotamers and
  production-covered E/Z bonds are dropped. Cap: ceil(n_prod*(mult-1)), at least 1. Order: permutation seeded by crc32 of
  the ligand SMILES + metal (no cheap prior exists); first feasible are taken; k-th extra pairs with production combo k mod n_prod.
- Engine: no extras -> exactly the production path. Cross-class dedup protects production rows (visited first).
- Row check: `research/bench/tools/jdev/stereo_rowcheck.py`; results in research/data/jgen/stereo/.
- Tests: tests/test_stereo_superset.py.

## Checkpoint (a) results (2026-11-09, CPU, seed 11, golden call; research/data/jgen/stereo/*.json|log)
All 7 golden SOLVE parents: no production row lost, top-1 identical. Only CBTYCO gets an extra class (3 classes in pool,
30/30 production rows found + 16 extra rows). SOKVET: 4 production E/Z classes kept, new-element list empty after dropping
production-covered E/Z and M-C-X rotamers -> identical to production (128/128 rows).
tools.check ws: fingerprint PASS (6b47c3fb45a9), golden 0/490 PASS, pytest 589 pass / 14 FAIL: all 14 are
test_finish_phase.py::test_phase_cuda_capture (CUDA replay vs eager np.array_equal, last-digit differences). Reproduced
on pristine master d0395ca (master-ro, adam-MELLUN) -> pre-existing on this 2060 / environment, unrelated to this branch.

## Checkpoint (b) findings
LIPDEU: tmQM SMILES types the cyclo-P4 ring with cumulated P=P bonds (each ring P valence 5, SP2, degree 3). RDKit sees no
stereocentre (ring P not in perceive_*), and ETKDG does not honour chiral tags on these atoms (16 tag combinations x 4 seeds:
identical side patterns). Free-ligand embed side pattern over 24 seeds: +-+- (all-trans, = reference) 19, ++-- 3, +++- 2;
the complex embed gives the wrong 3:1 pattern in 95/96 rows, so it is donor-constraint/seed dependent. A fix needs a
mechanism other than chiral tags (rejection sampling on ring-substituent signed volume, or ring-side constraint). Not done.
DUHHIX: the terminal azo N0 (donor, one implicit H) in N=NC(...)=C([O-])..: chelate_ring_stereo fixes only C2=C11;
undecided_ez drops N0=N1 because N0 has no heavy substituent (the stereo atom would be its H). Free embed: H-N0=N1-C2
dihedral ~0 deg (H in ring side, lone pair outward, wrong for chelation) in 17 of 20 seeds, ~180 deg in 3. Generic fix:
chelate_ring_stereo extended to a donor =N-H on the chelate path (H trans to the chain continuing to the chelate ring).
Production embed change, behind a flag; not implemented.

## Cap change (owner-approved): n_extra <= max(min(4, n_new_configs), ceil(n_prod*(mult-1))). Rowcheck 7 golden: all ok; ws: fingerprint+golden PASS, pytest = known cuda_capture set.

## 2026-11-09 — login panel analysis
Analysed A-relative contract metrics on the PM login node (snapshot e90c1aa). AS fails to qualify on the available evidence although observed numeric thresholds pass: NEPNEF and UYOVUA have failed GFN2 records in every arm/seed and QEZVEY was excluded, so the hard check is incomplete. ASP fails the per-class K check. See STEREO_PANEL_REPORT.md and research/data/jgen/stereo/panel_login/summary.json. No Colab spend.

Cache dev check: workstation 2060, DAMVAQ and FUTSOD seed 11, A/AS. Estimate 97 s; initial cache fill 193 s, canary 2/2, then restored 2 A records and ran AS alone, canary 2/2, 157 s. `cache_dev2` / `cache_dev3` under research/data/jgen/stereo/. AS-only records carry null control_union_distinct, since no A conformers are in memory; restored A records retain the value and are the inputs to downstream control_union. Minimum available memory 3.9 GB during second run (7.5 GB before start). No Colab spend.
- 2026-11-09 workstation checkpoint: fingerprint `6b47c3fb45a9` passed; golden 0/490 differed; pytest 589 passed, 14 failed, 3 skipped. All 14 are the known `test_finish_phase.py::test_phase_cuda_capture` failures on the 2060. Overall `tools.check ws` reports FAIL due to those cases. Report: main checkout `research/data/jgen/stereo/wrapup_ws/ws_20261109-215612.json`. No billed job or Colab spend.
- QEZVEY separate A/AS seed-11 attempt with one-start experimental dedup (`jgen/q-stereo-check` `709956f`) timed out at 480 s; both records are timeout, MPS stopped cleanly. Observed wall would put three seeds beyond the directive's 20-minute limit, so seeds 12–13 were not run. QEZVEY remains excluded; see panel report.
- Runner follow-up: the screen now applies registered arm flags to their live modules and rejects unregistered names, allowing runner-provided flags to take effect. Focused isolated live-flag test passed; all built-in arm flags were checked against `profile.FIELDS` (none unregistered). The workstation check above preceded this bench-only change; a final full checkpoint is still required after integration with the corresponding runner change.
