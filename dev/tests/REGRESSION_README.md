# Tests of the generator and the fit pipeline

Run from the repository root with the pixi environment: `pytest` (the pyproject sets `testpaths = dev/tests` and the
import path `src:dev:research/bench`). `conftest.py` asserts that `jacobik` is imported from `src`, never from an
installed copy. The golden reference check is not a test: `python -m tools.check ws` (fingerprint + golden + sharded pytest).

| File | Guards |
|---|---|
| `test_profile.py` | fingerprint identity, flag registration (every FIELDS entry resolves to its module attribute), override rules |
| `test_typing.py` | donor typing, hapto gates, evidence-weighted assignment, identity reconcile |
| `test_keys_and_ladders.py` | Q-quantity keys, ladders, runtime vs extractor parity, fit-pipeline sanity |
| `test_scorer.py` | physical Q6 donor pairs, torsion tail |
| `test_stereo_configs.py` | isomer / configuration-class enumeration, mirror completion, ring states, site enumeration, output by class |
| `test_solver.py` | IA-k finish phase, iRprop toggles, polish, CUDA-graph replay equivalence and generator, macrocycles, start Kabsch |
| `test_embed.py` | embed fixes (haptic Li cap, GFN2 cache) |
| `test_never_raise.py` | hang and never-raise guards |
| `test_db_container.py` | lossless DB container round trip |
| `test_fit_determinism.py` | fit pipeline stages, fit rules, per-quantity fit modules, determinism |
| `test_chem_elements.py`, `test_inputs_paths.py` | shared element tables and numpy helpers; package paths and import hygiene |
| `test_dev_tools.py` | `tools.run` (control cache, lanes, runner, notify) and `tools.check` helpers |
| `test_src.py`, `test_v2_features.py` | older broad suites (typing, OS deduction, orientation, embed), kept whole |

`test_phase_cuda_capture` is marked `xfail(strict=False, reason="known, owner 2026-11-11")` (CUDA graph capture; ignored by
decision). Tests of research tooling live in `research/bench/tests` (run with `pytest research/bench/tests`).
`regression/` holds the 24-case accuracy panel (`run_panel.py`, MACE; a tool, not a test) with its fixtures.
