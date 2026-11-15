# research/gpuwork/colab: GPU jobs on Google Colab

`cj.py` runs GPU work on one Colab VM, so local GPU work (NCI v2 Step 4) does not have to share the RTX 2060.

The Colab CLI is installed with `uv tool install google-colab-cli`. Authentication uses ADC with the colaboratory scope, which the owner has already done. Run `cj.py` from the main checkout with the pixi python.

```bash
P=.pixi/envs/linux/bin/python
$P research/gpuwork/colab/cj.py credits
$P research/gpuwork/colab/cj.py up   --name jg --gpu L4 --repo .claude/worktrees/<wt> --ref HEAD   # ~10 min: bundle, upload, pixi env, fingerprint
$P research/gpuwork/colab/cj.py run  --name jg --tag ctl_tail -- python -m tools.jgen bench run --arm control --panel tail --out /content/out/ctl_tail
$P research/gpuwork/colab/cj.py wait --name jg --tag ctl_tail --max-min 120                           # exit 0 only if the job exited 0
$P research/gpuwork/colab/cj.py get  --name jg /content/out/ctl_tail research/data/jgen/colab/   # copy the results back
$P research/gpuwork/colab/cj.py down --name jg                                                         # always stop the VM when done
```

## What `up` ships
- `git archive` of `--ref`. Only committed code is shipped, so commit before `up`.
- Read-only inputs:
  - the scaled_v1 and pilot_v1 spec, frame and manifest, and `ref_mace_cache`;
  - the constraints DB `48ac82b5…`;
  - all clean tmQM/lnQM xyz and BO shards.
- The tmQM/lnQM CSV, **sanitized**: only CSD_code, CSD_years and SMILES. The QM property columns never leave the machine. A loader that needs them (`TMQMParser.csv_index` / `_attach_qm_properties`) then gets None for them, as intended.
- The env is built from `pixi.lock` (`--frozen`), plus `pip_extras.txt`: the pip-installed packages of the local env, pinned, with mace at git `726ecec`.

`up` fails unless the production fingerprint is `5f74b14b57ed` (IA-k: iRprop+ search/finish, fixed switch 100, symmetry-only release, thermal widths; old C1 via `finish_phase=null`; on top of the eta2 / haptic-cap fold-in `eta2_face_plane`, `eta2_group_bend`, `embed_haptic_li_cap` true with the DB `constraints_2026-11-01.json` and its `.meta.json` sidecar, on top of the Package H fold-in with `q6_donor_pairs={"hh":0.95,"ho":0.90,"oo":0.80}`, on top of the Li-cap fold-in `embed_li_cap=true`, Package F `bend_release=[0.1, 100]` and restructured code with `nci=false`, `deterministic_embed=true`, `ring_fold_hinges=false` hardcoded; the bench restates the first two as overrides, which leaves it unchanged).

Jobs run in `/content/jk` with:
- `PYTHONPATH` (src, repo, refit, budget);
- `JACOBIK_XTB` pinned to the env's xtb;
- `JACOBIK_DATA_ROOT=/content/jk/data`;
- the env's bin first on PATH.

## Rules
- **Compare only within one platform.** Pilot (2026-10-25, ABATUT and HOXJUZ, seeds 11–13):
  - L4 vs L4 repeat: top-1 identical, floors within 0.3 kJ/mol.
  - L4 vs the stored 2060 pools: HOXJUZ top-1 dE differs by up to 17 kJ/mol, and seed 13 picked a different top-1.
  - MACE reference energies are identical across platforms.

  So every arm on Colab is paired with a **control run on Colab, on the same GPU type**, never with the stored Step 2 pools. The noise band is re-measured there.
- **One VM at a time.** `up` refuses if a VM is active.
- **Credit cap.** `up` and `run` refuse once 180 units are spent since 530.79 (owner budget: 200 units per month). An L4 costs ~1.5 units/h, an A100 more; check with `credits`.
- **Stop the VM** (`down`) when a batch is done or a job fails. An idle VM still bills.
- **Keep `wait` running for long jobs.** Its polls keep the session active.
- **The CLI connection drops now and then.** `cj.py` retries every call with a hard timeout. Do not call `colab exec -f` directly: it can return no output.
- **Bundles are cached per commit** in `gpuwork/out/colab/bundles/`. Delete the cached `.tgz` if the data list in `cj.py` changes.
