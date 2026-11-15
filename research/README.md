# research/

Everything around the generator that is not the generator: benchmarks, the evidence behind production settings, paper drafts and
the archive. The package itself is in `src/`; developer tests and tools are in `dev/` (see `ARCHITECTURE.md`, section 8).

| Path | What |
|---|---|
| `bench/benchmarks/` | Panels, catalogues, protocols and the **locked sets** (`LOCKED_SETS.md`, `locked_exclusions.json`; never moved, never used for tuning) |
| `bench/tools/` | Benchmark tooling: `jgen` (runner, panels, diagnostics), `jdev`, `architector`, `q6_ablation`, `runkit` |
| `campaigns/` | Contracts, reports and inputs of the campaigns behind the current production settings (`p2r/` P2R fold-in, `ds/` data fold-in, D3, eta2, stereo, IA-k) |
| `paper/` | Paper and SI drafts (describe the final product only) |
| `gpuwork/colab/` | Colab VM helper `cj.py` (fallback compute) |
| `archive/` | Closed work: `docs/` (index in its README), `code/` (`interactions/`, `refit/`, `trope_suite/`, `fit_legacy/`, `nci/`, `refit_top_legacy/`, `unused_src/`), flags ledger, old tests, branch notes, notebooks, historical zips |
| `data/` | Untracked outputs (see its README) |

Where to look for the evidence of a production setting: `CHANGELOG.md` (one entry per fold-in) and `PERFORMANCE.md` (`v1.0a`) point to contracts and reports
in `research/campaigns/`; older campaigns are under `archive/docs/jgen_campaign/<group>/`.
