# Preserved Non-Covalent Interactions Research (`research/archive/code/interactions/`)

This directory preserves non-covalent interaction research modules, exploratory scoring formulations, and calibration pipelines developed prior to the October 2026 consolidation. These files are preserved for scientific provenance and documentation; they are not maintained as part of the production package.

---

## 1. Modules and Outcomes

### A. NCI v2 Pipeline (`nci_v2/`)
- **Outcome:** Full surface response calibration completed across 120 groups and 6,130 label rows with 4,000 bootstrap replicates. The calibration converged to $\gamma_{\text{shared}} = 0.0$ with all family scales zero. This established that the proposed NCI surface did not provide a statistically significant signal over the baseline repulsive potential on real complex geometries.
- **Documentation:** See `research/archive/code/interactions/nci_v2/README.md` and `research/archive/docs/nci_v2/roadmapv2.md`.
- **Original Run Commands:**
  ```bash
  # Step 3b all-root research fit
  python -m refit.nci_v2_step3b_fit --frame data/refit/nci_v2/scaled_v1/frame.json --out ./step3b_out
  # Step 4 calibration
  python -m refit.nci_v2_step4_calibration --labels ./labels.parquet --out ./calibration_out
  ```

### B. Legacy NCI (`nci.py`)
- **Outcome:** Initial regularized C2 spline formulation covering hydrogen bonding (`hb`), pi-pi stacking (`pp`), and CH-pi interactions (`chp`). While `hb` produced modest dev80 score gains (+0.69 kJ/mol), broader evaluation revealed severe ligand-clash artifacts on larger complexes. Replaced by the NCI v2 study and disabled in production (`nci = False`).
- **Original Run Commands:**
  ```bash
  python refit/run_refit.py nci --workers 4
  ```

### C. Q6 Potential of Mean Force (`q6pmf.py`)
- **Outcome:** Explored statistical pair potentials across distance histograms for non-bonded contacts (modes `all`, `hbond`, `contact`). Attractive potential wells created artificial conformational collapse in unconstrained ligands. Retired in favor of clean, purely repulsive exchange sterics implemented in `src/jacobik/stats/q6.py`.
- **Original Run Commands:**
  ```bash
  python -m jacobik.stats.q6pmf fit --tmqm-dir data/ref/tmQM --out data/refit/final/q6pmf.json
  ```

### D. Bond-Valence Sum Coupling (`bvs.py`)
- **Outcome:** Investigated bond-valence coupling $V = \sum \exp((R_0 - R)/b)$ to model multi-bond coordination tightness. Benchmarking showed neutral to negative rank correlation effects across transition metal complexes. Retired and removed from the active solver.
- **Original Run Commands:**
  ```bash
  python -m jacobik.stats.bvs fit --tmqm-dir data/ref/tmQM --out data/refit/final/bvs.json
  ```
