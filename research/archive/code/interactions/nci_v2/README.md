# NCI v2 Research & Refit Archive

**Repository Origin:** Branch `codex/nci-v2-steps3-4` (HEAD `9c25173`)  
**Report Reference:** [`TASK3_STEP4_REPORT.md`](../../../docs/nci_v2/TASK3_STEP4_REPORT.md) and [`task3_step4_report.json`](task3_step4_report.json)  
**Outcome:** **$\gamma_{\text{shared}} = 0$** (All family scales zero; candidate DB byte-identical to base production DB).

---

## 1. Overview & Objectives

NCI v2 was an extensive effort to build a second-generation non-covalent interaction scoring model based on reference quantum mechanical interaction calculations, surface coordinate binning, directional contact geometries, and regularized likelihood scaling.

The full pipeline encompassed Steps 1 through 4:
- Coordinate extraction and columnar storage (`nci_v2_extract.py`, `nci_v2_columnar.py`)
- Surface and frame parameterization (`nci_v2_frame.py`, `nci_v2_surface.py`)
- MACE-POLAR-1 GPU labeling and timing benchmarks (`nci_v2_step4_label_gpu.py`)
- Regularized likelihood response modeling (`nci_v2_response.py`, `nci_v2_step4_fit_candidate.py`)
- Ranking benchmark on fixed evaluation pools (`nci_v2_step4_rank_benchmark.py`)

---

## 2. Outcome: $\gamma = 0$

Upon completing Step 4 fitting and validation:
1. **Zero Weight / Inactive:** The regularized bootstrap optimizer converged to $\gamma_{\text{shared}} = 0.0$ and scale factors of $0.0$ across all interaction families.
2. **Byte-Identical Candidate Database:** Because all fitted scaling parameters collapsed to zero, the resulting candidate database was byte-for-byte identical (`48ac82b56d71fd6e...`) to the base production database `constraints_2026-10-18.json`.
3. **Zero Ranking Effect:** On the 60-pool ranking benchmark, $\Delta \text{score} = 0.0000000000\ \text{nats}$, top-1 agreement remained 100%, and no improvement in conformer ordering was observed.

---

## 3. Scientific Reasons for the $\gamma = 0$ Result

1. **Placebo Sensitivity Test:** The objective function incorporated a placebo sensitivity safeguard: if adding interaction features failed to exceed statistical noise on shuffled conformer pairs, regularization penalized the term to zero.
2. **Lack of Discriminating Variance:** MACE-POLAR-1 energy gradients across near-native conformers showed that intra-ligand steric repulsion and coordination shell constraints overwhelmingly dominated the variance ($>99.5\%$), leaving the non-covalent terms unable to improve ranking fidelity without introducing unphysical distortions.
3. **Robust Model Selection:** Rather than forcing a non-zero hyperparameter that would risk overfitting or destabilizing downstream kinematics, the pipeline cleanly chose $\gamma = 0$.

---

## 4. Preservation & Future Refits

All code, estimators, transforms, and test suites are preserved here under `research/archive/code/interactions/nci_v2/` for future reference, re-analysis with alternative quantum chemistries, or exploration in post-restructure refits.
