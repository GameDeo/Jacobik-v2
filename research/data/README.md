# research/data: live research data (gitignored except this file)

Outputs of runs, golden references and evidence for the current production state. Finished work lives in
`research/archive/data/` (dated folders, tar + sha256); read it, do not write into it. Nothing here is a source of truth for
code: tracked reports and contracts are in `research/campaigns/`, `PERFORMANCE.md` and `CHANGELOG.md`.

| Path | What |
|---|---|
| `golden/` | Golden references for the production fingerprint (`production.npz`, `u1.npz`, `legacy-2026-10.npz`; checked by `tools.check` and `research/bench/tools/jdev/golden.py`). `golden_<fp>/` next to it holds a prepared or previous fingerprint; older ones are in `research/archive/data/2026-11_golden_history/` (`JACOBIK_GOLDEN_DIR` selects one). Never overwrite without the auditor. |
| `p2r/` | P2R fold-in evidence: `final/` (current evidence, protected), `proto/`, `baseline/`, `step3/`, `metric/` |
| `architector/` | Architector comparison runs (protected) |
| `runs/` | `tools.run` spool and control cache (`control_cache/`) |
| `audit/` | Dated audit outputs (DS investigation, gate fixes) |
| `jgen/campaign/` | Campaign outputs (A4, C, D2, D3, eta2, I); candidates for archiving once their reports are final |
| `d2/`, `nn_correction/`, `infra/`, `infra_fuse/`, `runkit/`, `lit_isomers*/` | Track outputs of D2, the parked NN correction, run infrastructure and the literature isomer audit |
| `refit_nci_v2/` | Symlink to the frozen fit frame `data/refit/frame_v1` (read by `tools.jgen.inputs` and the fit pipeline) |

Production fingerprint history is kept in `CHANGELOG.md` (one entry per fold-in).
