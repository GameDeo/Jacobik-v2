# Jacobik technical notes

These notes describe the implemented architecture and scientific models in the repository. They document mechanisms and interfaces, not experimental outcomes. Model parameters learned from data or offline calculations are identified as fitted or calibrated; result values and benchmark comparisons are intentionally omitted.

## Documents

- [Glossary and notation](glossary.md): canonical terms, symbols, units, and naming rules used throughout this set.
- [Solver architecture](architecture.md): inputs, molecular representation, conformer construction, assignment, kinematics, optimization, and output.
- [Scoring and chemical models](scoring-and-chemistry.md): statistical likelihood terms, steric chemistry, coordination chemistry, and mathematical conventions.
- [Training data and refitting](data-and-refit.md): structure processing, feature extraction, statistical fitting, database artifacts, and provenance.
- [Experimental modules](experiments/README.md): current research implementations, their intended interfaces, and their status.

All paths in the prose are relative to the repository root. The `ARCHITECTURE.md` document remains the current operational reference; this directory expands its technical descriptions and separates experimental design from adopted behavior.
