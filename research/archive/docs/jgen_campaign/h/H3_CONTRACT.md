# Package H-3 evaluation contract

Status: pre-registered after merging master `ab7dbea` (includes `jgen/b-solver` at `42e3716`); committed before any H-3 solve or result inspection.

## Question and arms

Compare the production control with one combined candidate on the same L4 GPU, software environment, data, and seeds:

- **Control:** production defaults, with the runner's standard `nci=false` and `deterministic_embed=true` settings applied identically to both arms.
- **Combined:** control plus `q6_donor_pairs={"hh": 0.95, "ho": 0.90, "oo": 0.80}`, `angle_tail_quadratic=true`, and `embed_haptic_li_cap=true`.

No E/Z skip flag is included. Record both fingerprints from the run metadata; the control must be `6cc586e17ab1` and the combined fingerprint must be `d587e9503943`. Confirm `finish_phase=None` (off) for both. The score flags are specified by `research/bench/jgen_campaign/H1_PATCH.json`.

## Panels, pairing, and execution

Run `cal` first and then `tail`, all on the same L4 VM. Within each panel run control and combined as separate single-arm `bench run` invocations, in that order, with identical seeds 11, 12, and 13. This ensures each process loads its own profile flags before module initialization. Freeze panel membership from the shared manifest before launching. The source manifest contains three task rows per parent, while the runner deduplicates to unique codes. The frozen selection is 120 cal parents (360 manifest task rows) and 10 tail parents (30 task rows), with five parents overlapping. Treat cal and tail as separate analyses; do not deduplicate their overlap across panel summaries.

Use these per-panel invocation settings (replace `PANEL` with `cal` or `tail`):

```bash
python -m tools.jgen bench run --panel PANEL --seeds 11,12,13 --arm control \
  --flags '{}' --out /content/out/h3/PANEL/control
python -m tools.jgen bench run --panel PANEL --seeds 11,12,13 --arm combined \
  --flags '{"q6_donor_pairs":{"hh":0.95,"ho":0.90,"oo":0.80},"angle_tail_quadratic":true,"embed_haptic_li_cap":true}' \
  --out /content/out/h3/PANEL/combined
```

Manifest: `research/data/refit_nci_v2/scaled_v1/manifest.json`, SHA-256 `5c3e5897f099eac36eb52bf14b67ecd8fd299e3819c83c8d0d5daa0d0f0ce450`.

Frozen sorted parent selections:

- **cal (120):** AJAQAD, AWONOR, AYOGUR, AZUSIA, BABNAV, BIBFUN, BOCRAN, BOHWUT, BOLGIU, BOTYUH, BUXJIO, CAQLEP, CAXTAX, CEHDUS, CEXPOM, CIFZEZ, CIXYUD, DAMVAQ, DOHTOM, DUHHIX, EBAYIQ, EDOBUV, EFOXII, EPUWAQ, EQODOF, EZIBUL, FASPOH, FEKKAJ, FELSAU, FETTAZ, FEXWUC, FUTSOD, GOTWET, HEDJOP, HOCCUA, HOHZAG, HOXJUZ, HOYBED, HUNGOP, IYUKOC, JAYCEV, JEGPUJ, JUCBIV, JUPDON, JUPPOZ, KIXVUJ, LEGXEA, LEPFUJ, LIPDEU, LISVOC, LOPCAW, LOPJEF, LUQVOK, MEBHEK, MEJYIN, MIDQEY, MIGTEE, MIVXAR, MUBREJ, MUFFEA, NAGHUC, NEPNEF, NEQCIV, NOYTUS, NUCKUS, NUHPEL, NUWFAM, OMAGAK, OREHAV, OWAWIU, PANBOY, PENHEY, PIWHAF, POBCEQ, QAPFAS, QERSUB, QEZVEY, QUGXUN, QUTWOT, RIQDEB, RUJCUW, RUVKIE, SEJKOK, SESWIZ, SILROY, SIVBUV, TEDPOI, TEJQAD, TIKZOF, TOBGOH, UBOBUI, UKEPIJ, UMUBAF, USAPOU, UYOVUA, VAVBIE, VOBGUR, WECHOE, WEYJIT, WIBHOH, XACMOE, XAQCAT, XEVDAE, XEYSUP, XIKGEE, XOJHIN, XUHMIY, XUKKIA, YABYEJ, YEQNUD, YISQUM, YOSXIP, YOVNEC, YUYGOP, ZAFPOP, ZAQNUE, ZAZREZ, ZEFXAM, ZOJDEH, ZOQFOA.
- **tail (10):** DAMVAQ, HOXJUZ, IFUROQ, JEGPUJ, NEPNEF, SOBHAS, USAPOU, XEYTUT, YUCZUS, ZELTAP.

Keep every selected panel entry in the ledger. NEPNEF is reported in both panels and is excluded from inferential denominators only if the H-2 record still classifies its input as invalid due to GFN2 fallback. No other parent is excluded after outcomes are observed. Any arm/unit failure is recorded with its reason; a panel with a missing paired arm result fails the primary gate rather than imputing a numeric ΔE.

Use full-pool XYZ export for every returned conformer (runner default; do not pass `--no-xyz`), with builder-order atom labels, row keys, returned scores, MACE energies, top-1/floor markers, and serialized full-pool 1 Å coverage. The historical H-R estimate is about 87 MB expected and 1.15 GB under its observed-max scenario for 130 parent-panel entries × 3 seeds × 2 arms; output remains below the 2 GB compression trigger, so use uncompressed export unless the pre-run estimate on the merged tree exceeds 2 GB. Record the actual bytes and file counts before shutting down the VM.

Save outputs under `research/data/jgen/campaign/H/h3/{cal,tail}/{control,combined}/` with separate records/XYZ. Keep the same GPU type and VM for all four invocations; report GPU model, VM identifier, software commit, wall times, and Colab spend. Fetch outputs, verify expected record and export counts locally, then shut down the VM.

## Primary endpoint and gates

For each parent and seed, use the runner's returned-score top-1 and its unrelaxed MACE energy relative to that parent's crystal reference, in kJ/mol. The paired outcome is `combined ΔE − control ΔE` for the same parent and seed. On each panel separately, report the mean and median paired difference and a two-sided 95% percentile parent-clustered bootstrap CI for the mean: 10,000 resamples, RNG seed 11, resampling parents and retaining all three seed pairs for each sampled parent. **Pass the primary gate only if the CI upper bound is below +2 kJ/mol on both cal and tail.**

Predeclared guards:

- No parent may have combined top-1 ΔE more than 25 kJ/mol worse than control on at least two of its three seeds. List every parent crossing this rule and provide its per-seed Q1–Q6 score-term breakdown for the control and combined top-1 rows.
- FEKKAJ and NUWFAM must each improve: their three-seed mean paired top-1 ΔE must be below zero. Report all three paired seed differences for each.
- The top-1 obscene rate must not rise, using `ΔE_unrel > 200 kJ/mol`; report counts, denominators, and paired parent/seed transitions.
- The ratio of summed per-unit `wall_solve_s` (combined/control) must be ≤1.10. Report panel-specific and pooled ratios.
- Any arm-level failure beyond the predeclared NEPNEF exclusion fails the affected panel's primary gate; report it without replacement or retuning.

## Secondary reporting

Report paired floor ΔE, top-1 and floor results per parent, complete failure ledgers, full-pool coverage at heavy-atom symmetry-aware Kabsch RMSD ≤1 Å (including the metal), top-1/floor versus full-pool coverage, and score–energy Spearman correlation within each returned pool. Report panel and parent summaries; keep FEKKAJ, NUWFAM, and XOJHIN separate. Report NEPNEF either way, including why it is excluded if the H-2 invalid-input finding remains.

Attribute the combined effect per fix from these two clearly defined categories: H-1 score terms (`q6_donor_pairs` plus `angle_tail_quadratic`) and H-2 haptic cap (`embed_haptic_li_cap`). Do not run extra ablations unless the combined results leave an attribution materially unclear. If an ablation is needed, state the ambiguity and commit the one-flag-off arm specification before launching it; use the same paired seeds, panel, and hardware, and label it exploratory rather than replacing the primary comparison.

The frozen-record summary is produced by `python -m tools.jgen.analyze_h3`; it verifies all expected parent/seed pairs from the frozen manifest, keeps excluded NEPNEF records visible, and writes the paired analysis JSON. Run with `--exclude NEPNEF` if its H-2 invalid-input finding remains. The report and raw records remain the audit source if any value must be inspected or explained.

## Decision and final audit

Report every gate as pass or fail. A failed gate is not tuned away. After the paired run and analysis, perform the directive's one heavy final audit: full test suite, golden generated into a new directory under auditor review, and replay-parity panel. Record commands, runtimes, artifacts, fingerprints, and any failures. Stop at Checkpoint H-3 for audit and owner fold-in; do not merge or promote the patch.
