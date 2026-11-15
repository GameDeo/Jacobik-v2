# A4b progress — 2026-11-01

## Implementation and smoke

Resumed under Amendment 1 on `jgen/a4-baseline`. `charge_mult.py` resolved all 125 panel parents through `inputs.load_mol`; no charge/multiplicity lookup failed. The split was 120 cal + 10 tail parents with five shared parents counted in both panels, giving 125 unique parents and 375 unique parent-seed units.

The setup-only Pixi mirror/retry patch is committed as `61fe4eb`; relaxed-row serialization fix as `92a184e`; completed measurement/table implementation and smoke gate as `fd79db9`. The smoke ran on L4 `jgA4b2` at `92a184e`, fingerprint `273f2ccbf892`, for QEZVEY and BOTYUH, seeds 11–13. All six records and manifests passed the smoke criteria: status/fingerprint, row counts, top-1 and floor reproduction, finite relaxed reference MACE energies, and unrelaxed reference MACE parity. The fallback check exited 0. Smoke spend was 0.63 units.

Six CPU tests, py_compile, strict smoke verification, and smoke-only table/report generation passed. Smoke solve wall was 48.09 s per QEZVEY seed and 1.29 s per BOTYUH seed; 11 selected row relaxations summed to 44.71 s.

## Full run and analysis

The first full-run attempt was discarded because the local controller treated the integer from `Session.wait()` as a result object and shut both VMs down before fetch. It consumed 2.12 units; no outputs from that attempt are used.

The successful retry used the same `fd79db9` bundle and fingerprint on both L4 VMs. It fetched 375 records, 372 pool manifests, 372 per-unit relaxation files, and 124 parent references. The three expected failures are NEPNEF seeds 11–13, covered by Amendment 2; no other parent showed the embed fallback. All 372 successful records matched the fingerprint and manifest row counts. Top-1/floor energy reproduction errors were 0.0 kJ/mol; reference unrelaxed MACE parity errors were 0.0 kJ/mol; there were zero U-bucket disagreements. All 125 charge/multiplicity entries resolved. Relaxation had 0 failures and 9 capped row relaxations; all 124 reference relaxations succeeded.

The full run used 2.52 units; total A4b task spend from the recorded 488.45 start balance is 5.27 units, below the 8-unit cap. Successful full-run VM time was 1.34 aggregate VM-hours (80.4 VM-minutes), with 43.5 minutes elapsed across both VMs. Summed solve wall was 1,765.1 s; xTB relaxation wall-sum was 3,899.4 s; CUDA MACE wall was 45.4 s. Both VMs were fetched, counted, and brought down; current active assignment count is 0.

The run followed the directive's data selection and chemistry requirements, but the CPU relax batch did not overlap the GPU solve batch on each VM: the xTB/MACE stage started after that VM's full bench batch. This scheduling deviation is stated in the report. The final tables contain 13,697 panel-row entries, 390 panel-unit entries, 125 parent rows, per-panel clustered bootstrap statistics, seed consistency, and all six sensitivity settings.

`A4B_REPORT.md` and the data outputs under `research/data/jgen/campaign/A4b/` are complete. The checkpoint commit is still pending; no work beyond the directive checkpoint has started.
