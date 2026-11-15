# Branch Archive: `gemini/registry-and-db-prune`

**HEAD:** `6626211f62174c30c451da7247a82b49dfef0ea4`  
**Status:** Fully merged into master at commit `dfd5f20` (*feat(refit): lossless DB container and mass-weighted pruning study*).  
**Worktree:** `.claude/worktrees/gemini-registry-and-db-prune`

### Contents
- Lossless float64 binary database container (`refit/db_prune_study.py`), achieving 25.43 MB (81.5% reduction) with bit-identical fidelity ($0.0000000000\ \text{nats}$).
- Comprehensive study and report in `data/refit/db_prune_study/REPORT.md`.
- Formally adopted as the production storage format in Phase 3 (`src/fit/build_db.py`).
