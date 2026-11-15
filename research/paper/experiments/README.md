# Experimental modules

This directory documents research implementations and their intended model contracts. “Implemented” means code or data interfaces exist; it does not mean the module is enabled in the production profile or validated for general use. Experimental outcomes and benchmark measurements are intentionally excluded.

## Notes

- [NCI v2 surface model](nci-v2.md): transferable contact opportunities, smooth geometric features, density-ratio surfaces, and offline energy/force calibration.
- [Other experimental components](other-components.md): legacy NCI, bond-valence coupling, metal 1–3 contacts, search coverage, and ligand-internal geometry.

The adopted model and common notation are defined in [scoring and chemical models](../scoring-and-chemistry.md) and [the glossary](../glossary.md). Experimental modules must use those terms consistently. A runtime implementation is a separate code path unless explicitly stated otherwise in its module design.
