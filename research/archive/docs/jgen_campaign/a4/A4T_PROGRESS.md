# A4t progress — 2026-11-02

## Step 1: implementation

Built `research/bench/tools/jgen/a4t/` on `jgen/a4-baseline`. It imports the A4b pool/coordinate interfaces and runkit geometry, xTB and MACE helpers. It implements full-pool row relaxation with A4b-result reuse, element-constrained donor-shell RMSD, ideal-geometry shape measures, RDKit geometry stereo/ring measurements, the ordered A4t decision list, sensitivity classes, and parent/seed/row tables. The 35-parent list reproduces exactly from A4b `units.csv`: each is cal and `R-MISS` at seeds 11, 12 and 13.

Step 1 code commit: `621a21ac256fef7df1cc0fa9299284dbe620f193`. The initial unit tests covered reference shell RMSD, within-element donor permutation, and the then-current seven-class decision list; amended tests now also cover lazy 40,320 permutation capping and the `G*` branch (3 passed). `py_compile` passed.

## Step 2: local CPU smoke and run estimate

Ran the VM driver locally on BABNAV seed 11 with local xTB and CPU MACE, for plumbing only. It processed all 16 pool rows: 15 new xTB relaxations, one reused A4b relaxation, one reference relaxation, no manifest failures. xTB summed wall was 37.6 s; CPU MACE evaluated 17 distinct geometries in 17.4 s. These local values are not used in A4t analysis. During the first smoke invocation, serialization raised on duplicate `reused_from`; the writer was fixed and the rerun passed.

The 105 frozen pool manifests contain 3,516 rows. A4b already relaxed 328 of them; those are reused, leaving **3,188 new pool-row relaxations**, plus 35 reference relaxations required for VM reference parity. The parents are split by expected new-relaxation count plus one reference, longest-first in snake order: VM A has 1,641 jobs and VM B 1,582. At the directive estimate of 5 s per relaxation and three workers per VM, elapsed estimate is **45.6 minutes** on the slower VM (44.8 minutes ideal aggregate). This is below the 2 h gate. `split.json` and `credits_start.txt` are in the gitignored A4t data directory.

## Step 3: full-pool run and reference parity

Both L4 jobs ran from commit `57358ee1713c801ac7454054942464c4601511c3`, fingerprint `273f2ccbf892`. VM A processed 1,797 rows (1,624 new, 173 reused), 17 references; VM B processed 1,719 rows (1,564 new, 155 reused), 18 references. The fetched output had 105 unit JSONs, 3,516 row results and all 35 references; every unit row count matched its A4b pool manifest. All 3,516 row results were `ok`; there were no manifest failures. The VM reference MACE energies matched A4b `E_refrel` exactly for all 35 parents (maximum absolute difference 0.0 kJ/mol). Both VMs were fetched and counted before shutdown; active assignments are 0.

Colab balance at A4t start was 483.18 and after both VMs were down was 478.58, for **4.60 units**. Two launch/setup attempts failed and were shut down, and one early input-extraction launch failed because the asynchronous extract and run were started separately; those failures are included in the spend. The final run used one sequential extraction+driver job per VM. `credits_end.txt`, VM metrics, fetched outputs and split inputs are under the gitignored A4t data directory.

## Step 4: amended local analyses and §7 checkpoint

Amendment 1 defines geometry as unavailable outside CN 2–9, keeps T ahead of G, assigns `G*` only if undefined geometry reaches G, and requests pad-0.3 reference donors for donor-mismatch T parents. The analyzer handles those cases; high-symmetry permutations are generated lazily and capped at 40,320. Its local analysis covered all 35 parents and 3,516 stored relaxation rows. Amendment 1 counts were Z 13, R 6, T 5, G 1, F 5, S 2, C 3, G* 0.

Amendment 2 corrects the Z rule: it excludes metal CIP labels, maps row stereo to the reference using the optimal RMSD automorphism, and accepts globally inverted compared-centre labels while requiring double-bond labels to match. Reclassification used only the stored A4t relaxations; no solves, relaxations, or VM work were run. Seven former Z parents leave Z: BABNAV, CAQLEP, FEKKAJ, FEXWUC, SILROY, UYOVUA, ZAQNUE. New counts: R 9, C 7, Z 6, T 5, F 5, S 2, G 1, G* 0. The report lists the old-rule differing elements, the Amendment 2 change that resolves each, and any remaining mismatch for all 13 old-Z parents.

Validation: `pytest -q tests/test_a4t.py`: 5 passed; `py_compile` passed. Both VMs remain down (active assignments 0); Colab balance remains 478.58, A4t spend 4.60 units. Amendment 2 report and progress are ready for the §7 commit/checkpoint.

## Amendment 3: stored-data correction

Applied the two Amendment 3 fixes: double-bond stereo is compared only when both mapped labels are definite (absent and `STEREOANY` labels are ignored), and a remaining centre cause is “donor” only when a differing centre is in the donor set. Added one unit test for each fix. Re-ran the full decision analysis locally from the stored 3,516 relaxation rows; there were no solves, relaxations, or VM actions. All 3,516 rows remain present, with zero capped or failed rows. Tests: `pytest -q tests/test_a4t.py` (7 passed), `py_compile`, and `git diff --check`.

Amendment 2 → Amendment 3 counts: Z 6 → 4, C 7 → 9; F 5, T 5, G 1, G* 0, S 2 and R 9 are unchanged. YOVNEC and ZAZREZ leave Z because their only remaining double-bond differences include `STEREOANY` on one side. The report gives the per-atom causes for the four remaining Z parents, with zero-based indices, elements, direct metal-bound status, and the selected best-shell row. No new compute was used; Colab balance remains 478.58 with zero active assignments.
