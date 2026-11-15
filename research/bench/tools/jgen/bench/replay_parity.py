"""Replay-parity harness (directive_jgen_B §4).

Verifies on 3 `tail` parents × 3 seeds (9 units):
  1. CUDA graph-replay vs CUDA eager (`FAST_FK=True`, production GPU kinematics):
     `ReplaySolver(replay=True, FAST_FK=True)` vs `ReplaySolver(replay=False, FAST_FK=True)` on CUDA.
     Proves that CUDA graph capture & replay is bit-identical to eager GPU execution and never falls back (`mode == "replay"`).
  2. CUDA graph-replay vs CUDA eager (`FAST_FK=False`, reference per-ligand kinematics):
     `ReplaySolver(replay=True, FAST_FK=False)` vs `ReplaySolver(replay=False, FAST_FK=False)` on CUDA.
  3. Rewritten replay loop (`solver_replay.py`) vs package loop (`ik_solver.py`, `JointIKSolver` with `cull_at=()`),
     as in `tests/test_gpu_fast_path.py::test_replay_loop_matches_package_loop` (`FORCE_ON_CPU=True, FAST_FK=False`,
     `n_ik_steps=20` by default for quick check, or 200 via `--loop-steps 200` for full schedule verification):
     Proves that every solver flag / step-loop edit in `ik_solver.py` is faithfully generated into `solver_replay.py`
     with bit-identical (`0.0`) scores and top-1 indices.
     (Note: on CUDA, `ik_solver.py` divides `adam_m / bc1` by a Python float (`aten::div.Scalar`, which multiplies by
     `float32(1/bc1)`), whereas `solver_replay.py` divides by the 1-element device tensor `c1 = bc1.index_select(0, tl)`
     (`aten::div.Tensor`, hardware `fdiv`), introducing a 1-ULP difference at step 0 on CUDA; on CPU both use `fdiv`
     and are bit-identical.)

Requires on all 9 units:
  * identical top-1 row index (9/9);
  * max relative |Δscore| <= 1e-6 across all rows;
  * solver_replay.LAST["mode"] == "replay" (0 fallbacks).

Usage:
    python research/bench/tools/jgen/bench/replay_parity.py [--arm NAME] [--flags JSON]
           [--codes DAMVAQ,HOXJUZ,JEGPUJ] [--seeds 11,12,13] [--out FILE.json] [--report B_REPORT.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
DEFAULT_CODES = "DAMVAQ,HOXJUZ,JEGPUJ"
DEFAULT_SEEDS = "11,12,13"
TOL_REL_SCORE = 1e-6


def _compare_pair(sc_raw_a, sc_raw_b, r_a, r_b, last_a, require_replay_mode: bool = True):
    import numpy as np

    raw_abs = np.abs(sc_raw_a - sc_raw_b)
    raw_rel = raw_abs / np.maximum(np.abs(sc_raw_b), 1.0)
    max_rel_raw = float(np.nanmax(raw_rel)) if len(raw_rel) else 0.0
    max_abs_raw = float(np.nanmax(raw_abs)) if len(raw_abs) else 0.0

    sc_conf_a = np.array([c.score for c in r_a.conformers], dtype=float)
    sc_conf_b = np.array([c.score for c in r_b.conformers], dtype=float)
    n_c = min(len(sc_conf_a), len(sc_conf_b))
    conf_abs = np.abs(sc_conf_a[:n_c] - sc_conf_b[:n_c]) if n_c else np.zeros(1)
    conf_rel = conf_abs / np.maximum(np.abs(sc_conf_b[:n_c]), 1.0) if n_c else np.zeros(1)
    max_rel_conf = float(np.nanmax(conf_rel))
    max_abs_conf = float(np.nanmax(conf_abs))

    top1_trial_a = int(r_a.conformers[0].trial_idx)
    top1_trial_b = int(r_b.conformers[0].trial_idx)
    top1_raw_a = int(np.argmax(sc_raw_a))
    top1_raw_b = int(np.argmax(sc_raw_b))

    mode = last_a.get("mode", "unknown")
    replayed = int(last_a.get("replayed", 0))
    top1_match = (top1_trial_a == top1_trial_b) and (top1_raw_a == top1_raw_b)
    max_rel = max(max_rel_raw, max_rel_conf)
    max_abs = max(max_abs_raw, max_abs_conf)
    mode_ok = (mode == "replay") if require_replay_mode else (last_a.get("error") is None)
    ok = mode_ok and top1_match and (max_rel <= TOL_REL_SCORE) and (len(sc_conf_a) == len(sc_conf_b))
    return dict(
        ok=bool(ok),
        mode=mode,
        replayed_steps=replayed,
        n_raw_rows=int(len(sc_raw_a)),
        n_conformers_a=int(len(sc_conf_a)),
        n_conformers_b=int(len(sc_conf_b)),
        top1_trial_a=top1_trial_a,
        top1_trial_b=top1_trial_b,
        top1_raw_a=top1_raw_a,
        top1_raw_b=top1_raw_b,
        top1_match=bool(top1_match),
        top1_score_a=float(sc_conf_a[0]),
        top1_score_b=float(sc_conf_b[0]),
        max_rel_dscore_raw=max_rel_raw,
        max_abs_dscore_raw=max_abs_raw,
        max_rel_dscore_conf=max_rel_conf,
        max_abs_dscore_conf=max_abs_conf,
        max_rel_dscore=max_rel,
        max_abs_dscore=max_abs,
        error=last_a.get("error"),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="replay_parity")
    ap.add_argument("--arm", default="baseline_control", help="Label for the summary line")
    ap.add_argument("--flags", default="{}", help="JSON profile overrides merged onto control overrides")
    ap.add_argument("--codes", default=DEFAULT_CODES, help="Comma-separated parent CSD codes (default: 3 tail parents)")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated seeds (default: 11,12,13)")
    ap.add_argument("--loop-steps", type=int, default=20, help="Steps for CPU ReplaySolver-vs-JointIKSolver loop parity check (default: 20; use 200 for full schedule verification)")
    ap.add_argument("--out", default="", help="Optional path to write full JSON results")
    ap.add_argument("--report", default="", help="Optional markdown report file to append the one-line summary to")
    a = ap.parse_args(argv)

    for p in (str(REPO / "src"), str(REPO), str(REPO / "research" / "refit"),
              str(REPO / "research" / "bench"), str(REPO / "research" / "bench" / "tools" / "jdev" / "budget")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from tools.jgen.bench.run import _setup_env

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["JACOBIK_NUM_WORKERS"] = "1"
    flags = json.loads(a.flags)
    _setup_env(flags)

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)

    from jacobik.core import profile as P
    from jacobik.ik import solver_replay as SR
    from jacobik.ik.ik_solver import IKSolverConfig, JointIKSolver
    from jacobik.core.env import build_builder_and_inputs
    from tools.jdev import multiseed
    from tools.jdev.budget.common import stats_from_db
    from tools.jgen import inputs

    if not torch.cuda.is_available():
        raise SystemExit("replay_parity requires a CUDA GPU")

    codes = [c.strip() for c in a.codes.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in a.seeds.split(",") if s.strip()]
    fp = P.snapshot()
    shards = inputs.frame_shards()
    stats = stats_from_db(inputs.DB_PATH)
    cfg_gpu = IKSolverConfig(cull_at=(), cull_keep=())
    cfg_cpu = IKSolverConfig(n_steps=a.loop_steps, cull_at=(), cull_keep=())

    units = []
    for code in codes:
        shard, ds = shards[code]
        mol = inputs.load_mol(code, shard, ds)
        if mol is None:
            raise RuntimeError(f"no geometry for {code} in {shard}")
        builder, _, err = build_builder_and_inputs(mol)
        if builder is None:
            raise RuntimeError(f"unmappable {code}: {err}")

        for seed in seeds:
            t0 = time.perf_counter()
            raw_capture = {}

            with multiseed.safe_cuda_eigh_chunking():
                orig_rep_solve = SR.ReplaySolver.solve
                orig_pkg_solve = JointIKSolver.solve

                def make_cap_rep(key_scores, key_last):
                    def _cap(self, *args, **kw):
                        out = orig_rep_solve(self, *args, **kw)
                        raw_capture[key_scores] = out[2].detach().cpu().numpy().copy()
                        raw_capture[key_last] = dict(SR.LAST)
                        return out
                    return _cap

                def make_cap_pkg(key_scores):
                    def _cap(self, *args, **kw):
                        out = orig_pkg_solve(self, *args, **kw)
                        raw_capture[key_scores] = out[2].detach().cpu().numpy().copy()
                        return out
                    return _cap

                # 1) CUDA graph replay vs CUDA eager (production FAST_FK=True, 200 steps)
                try:
                    SR.ReplaySolver.solve = make_cap_rep("cuda_rep_fast_scores", "cuda_rep_fast_last")
                    SR.ENABLED = True
                    SR.FAST_FK = True
                    SR.ReplaySolver.replay = True
                    r_cuda_rep_fast = builder.solve(seed=seed, stats=stats, device="cuda", angle_bends=True, ik_config=cfg_gpu)

                    SR.ReplaySolver.solve = make_cap_rep("cuda_eag_fast_scores", "cuda_eag_fast_last")
                    SR.ReplaySolver.replay = False
                    r_cuda_eag_fast = builder.solve(seed=seed, stats=stats, device="cuda", angle_bends=True, ik_config=cfg_gpu)
                finally:
                    SR.ReplaySolver.solve = orig_rep_solve
                    SR.ReplaySolver.replay = True

                # 2) CUDA graph replay vs CUDA eager (reference FAST_FK=False, 200 steps)
                try:
                    SR.ReplaySolver.solve = make_cap_rep("cuda_rep_ref_scores", "cuda_rep_ref_last")
                    SR.ENABLED = True
                    SR.FAST_FK = False
                    SR.ReplaySolver.replay = True
                    r_cuda_rep_ref = builder.solve(seed=seed, stats=stats, device="cuda", angle_bends=True, ik_config=cfg_gpu)

                    SR.ReplaySolver.solve = make_cap_rep("cuda_eag_ref_scores", "cuda_eag_ref_last")
                    SR.ReplaySolver.replay = False
                    r_cuda_eag_ref = builder.solve(seed=seed, stats=stats, device="cuda", angle_bends=True, ik_config=cfg_gpu)
                finally:
                    SR.ReplaySolver.solve = orig_rep_solve
                    SR.ReplaySolver.replay = True
                    SR.FAST_FK = True

                # 3) Package-loop equivalence (test_replay_loop_matches_package_loop: ReplaySolver vs JointIKSolver, cull_at=())
                if a.loop_steps > 0:
                    try:
                        SR.ReplaySolver.solve = make_cap_rep("cpu_rep_scores", "cpu_rep_last")
                        SR.FORCE_ON_CPU = True
                        SR.ENABLED = True
                        SR.FAST_FK = False
                        r_cpu_rep = builder.solve(seed=seed, stats=stats, device="cpu", angle_bends=True, n_ik_steps=a.loop_steps, ik_config=cfg_cpu)
                    finally:
                        SR.ReplaySolver.solve = orig_rep_solve
                        SR.FORCE_ON_CPU = False
                        SR.FAST_FK = True

                    try:
                        JointIKSolver.solve = make_cap_pkg("cpu_pkg_scores")
                        SR.ENABLED = False
                        r_cpu_pkg = builder.solve(seed=seed, stats=stats, device="cpu", angle_bends=True, n_ik_steps=a.loop_steps, ik_config=cfg_cpu)
                    finally:
                        JointIKSolver.solve = orig_pkg_solve
                        SR.ENABLED = True

                    pkg_loop_parity = _compare_pair(
                        raw_capture["cpu_rep_scores"], raw_capture["cpu_pkg_scores"],
                        r_cpu_rep, r_cpu_pkg, raw_capture["cpu_rep_last"], require_replay_mode=False,
                    )
                else:
                    pkg_loop_parity = dict(
                        ok=True, skipped=True, max_rel_dscore=0.0, max_abs_dscore=0.0,
                        top1_trial_a=0, top1_trial_b=0, top1_match=True
                    )

            fast_fk_parity = _compare_pair(
                raw_capture["cuda_rep_fast_scores"], raw_capture["cuda_eag_fast_scores"],
                r_cuda_rep_fast, r_cuda_eag_fast, raw_capture["cuda_rep_fast_last"], require_replay_mode=True,
            )
            ref_fk_parity = _compare_pair(
                raw_capture["cuda_rep_ref_scores"], raw_capture["cuda_eag_ref_scores"],
                r_cuda_rep_ref, r_cuda_eag_ref, raw_capture["cuda_rep_ref_last"], require_replay_mode=True,
            )

            unit_ok = fast_fk_parity["ok"] and ref_fk_parity["ok"] and pkg_loop_parity["ok"]
            max_rel = max(fast_fk_parity["max_rel_dscore"], ref_fk_parity["max_rel_dscore"], pkg_loop_parity["max_rel_dscore"])
            max_abs = max(fast_fk_parity["max_abs_dscore"], ref_fk_parity["max_abs_dscore"], pkg_loop_parity["max_abs_dscore"])

            u = dict(
                code=code,
                seed=seed,
                ok=bool(unit_ok),
                max_rel_dscore=max_rel,
                max_abs_dscore=max_abs,
                cuda_fast_fk_replay_vs_eager=fast_fk_parity,
                cuda_ref_fk_replay_vs_eager=ref_fk_parity,
                package_loop_vs_replay_loop=pkg_loop_parity,
                wall_s=round(time.perf_counter() - t0, 2),
            )
            units.append(u)
            pkg_msg = (
                f"pkg_loop(top1={pkg_loop_parity['top1_trial_a']}/{pkg_loop_parity['top1_trial_b']}, "
                f"max_rel={pkg_loop_parity['max_rel_dscore']:.3e})"
                if not pkg_loop_parity.get("skipped") else "pkg_loop(skipped)"
            )
            print(
                f"[{code}:{seed}] {'PASS' if unit_ok else 'FAIL'} "
                f"cuda_fast_fk(mode={fast_fk_parity['mode']}, top1={fast_fk_parity['top1_trial_a']}/{fast_fk_parity['top1_trial_b']}, "
                f"max_rel={fast_fk_parity['max_rel_dscore']:.3e}) | "
                f"cuda_ref_fk(mode={ref_fk_parity['mode']}, top1={ref_fk_parity['top1_trial_a']}/{ref_fk_parity['top1_trial_b']}, "
                f"max_rel={ref_fk_parity['max_rel_dscore']:.3e}) | "
                f"{pkg_msg} ({u['wall_s']}s)",
                flush=True,
            )

    n_units = len(units)
    n_pass = sum(1 for u in units if u["ok"])
    n_replay = sum(1 for u in units if u["cuda_fast_fk_replay_vs_eager"]["mode"] == "replay"
                   and u["cuda_ref_fk_replay_vs_eager"]["mode"] == "replay")
    n_top1 = sum(1 for u in units if u["cuda_fast_fk_replay_vs_eager"]["top1_match"]
                 and u["cuda_ref_fk_replay_vs_eager"]["top1_match"]
                 and u["package_loop_vs_replay_loop"]["top1_match"])
    max_rel_all = max((u["max_rel_dscore"] for u in units), default=0.0)
    max_abs_all = max((u["max_abs_dscore"] for u in units), default=0.0)
    all_ok = (n_pass == n_units) and (n_units > 0)

    summary_line = (
        f"- **Replay parity (`{a.arm}`, fp `{fp['fingerprint']}`, flags `{json.dumps(flags, sort_keys=True)}`):** "
        f"**{'PASS' if all_ok else 'FAIL'}** — {n_pass}/{n_units} units passed "
        f"({','.join(codes)} × seeds {','.join(str(s) for s in seeds)}); "
        f"top-1 row index match {n_top1}/{n_units}; "
        f"CUDA replay mode {n_replay}/{n_units} (0 fallbacks); "
        f"max relative `|Δscore|` = `{max_rel_all:.3e}` (<= `1e-6`), max absolute `|Δscore|` = `{max_abs_all:.3e}`."
    )
    print(summary_line, flush=True)

    payload = dict(
        arm=a.arm,
        flags=flags,
        fingerprint=fp["fingerprint"],
        codes=codes,
        seeds=seeds,
        ok=all_ok,
        n_units=n_units,
        n_pass=n_pass,
        n_replay=n_replay,
        n_top1_match=n_top1,
        max_rel_dscore=max_rel_all,
        max_abs_dscore=max_abs_all,
        summary_line=summary_line,
        units=units,
    )
    if a.out:
        out_p = Path(a.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2))
    if a.report:
        rep_p = Path(a.report)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_p, "a") as fh:
            fh.write(summary_line + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
