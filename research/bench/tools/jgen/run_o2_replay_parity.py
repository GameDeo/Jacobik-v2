"""Compare replay against the same eager replay loop (existing 1e-6 score tolerance)."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIX = REPO / "tests" / "regression" / "fixtures"
CASES = {"MELLUN": ("tmqm_sample40", "S"), "AHIGOQ": ("tmqm_sample40", "S"),
         "PIDXOT": ("tmqm_sample40", "S")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--flag", choices=("adam_metric", "fire_metric", "adam_small", "fire", "irprop", "late_cosine", "zigzag", "plateau"), default="adam_metric")
    ap.add_argument("--polish-dt",type=float,default=0.1)
    ap.add_argument("--check-early-stop", action="store_true")
    args = ap.parse_args()
    import os
    os.environ["JACOBIK_PROFILE"] = "u1"
    os.environ["JACOBIK_PROFILE_OVERRIDES"] = json.dumps({"finish_schedule": args.flag} if args.flag in ("late_cosine", "zigzag", "plateau") else {"polish": args.flag, "polish_dt": args.polish_dt} if args.flag in ("adam_small", "fire", "irprop") else {args.flag: True})
    os.environ["JACOBIK_NUM_WORKERS"] = "1"
    import numpy as np
    import torch
    from tools.runkit import resources
    from tools.jdev.cases import get_case
    from tools.jdev.budget.common import stats_from_db
    from jacobik.ik import solver_replay as SR
    from jacobik.ik.ik_solver import IKSolverConfig
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    db = REPO / "data/db/constraints_2026-10-18.json"
    if not db.exists():
        db = REPO / "research/archive/data/refit/final/constraints_2026-10-17.json"
    stats = stats_from_db(db)
    result = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "flag": json.loads(os.environ["JACOBIK_PROFILE_OVERRIDES"]), "device": "cuda", "steps": args.steps, "cases": {}}
    with resources.gpu_lock("CodexB", timeout=900):
        for code, (folder, split) in CASES.items():
            row = {"status": "failed"}
            try:
                builder = get_case(f"tmqm:{FIX / folder}:{split}:{code}").make_builder()

                raw_scores = []
                original_joint = builder._run_joint_ik
                def observed_joint(*a, **kw):
                    result = original_joint(*a, **kw)
                    raw_scores.append(result[2].detach().cpu().numpy().copy())
                    return result
                builder._run_joint_ik = observed_joint

                def solve(replay):
                    raw_scores.clear()
                    SR.ENABLED = True
                    SR.ReplaySolver.replay = replay
                    SR.FAST_FK = False
                    return builder.solve(stats=stats, seed=11, angle_bends=True, n_conformers=8,
                                         n_ik_steps=args.steps, device="cuda",
                                         ik_config=IKSolverConfig(cull_at=(), cull_keep=()))

                wall0=time.monotonic()
                eager = solve(False)
                eager_wall=time.monotonic()-wall0
                eager_raw = np.concatenate(raw_scores)
                eager_scores = np.asarray([c.score for c in eager.conformers], float)
                eager_xyz = np.asarray([[a.position for a in c.complex.atoms] for c in eager.conformers], float)
                wall0=time.monotonic()
                replay = solve(True)
                replay_wall=time.monotonic()-wall0
                replay_raw = np.concatenate(raw_scores)
                raw_rel = (float(np.max(np.abs(eager_raw-replay_raw)/np.maximum(np.abs(eager_raw),1.0)))
                           if eager_raw.shape == replay_raw.shape else float("inf"))
                replay_scores = np.asarray([c.score for c in replay.conformers], float)
                replay_xyz = np.asarray([[a.position for a in c.complex.atoms] for c in replay.conformers], float)
                row.update(status="ok", raw_rows=len(eager_raw), max_raw_relative_score=raw_rel, replay_mode=SR.LAST.get("mode"), replayed_steps=SR.LAST.get("replayed"),
                           eager_rows=len(eager_scores), replay_rows=len(replay_scores),
                           max_score_abs=float(np.max(np.abs(eager_scores - replay_scores))) if len(eager_scores) == len(replay_scores) else None,
                           max_xyz_abs=float(np.max(np.abs(eager_xyz - replay_xyz))) if eager_xyz.shape == replay_xyz.shape else None,
                           parity=(raw_rel <= 1e-6 and len(eager_scores) == len(replay_scores) and eager_xyz.shape == replay_xyz.shape
                                   and np.max(np.abs(eager_scores - replay_scores) / np.maximum(np.abs(eager_scores), 1.0)) <= 1e-6
                                   and [c.trial_idx for c in eager.conformers] == [c.trial_idx for c in replay.conformers]),
                           replay_error=SR.LAST.get("error"))
                row.update(eager_wall_s=eager_wall,replay_wall_s=replay_wall)
                if args.check_early_stop:
                    from jacobik.ik import ik_solver as IKM
                    IKM.FINISH_EARLY_STOP=SR.FINISH_EARLY_STOP=True
                    wall0=time.monotonic()
                    early=solve(True)
                    early_wall=time.monotonic()-wall0
                    early_raw=np.concatenate(raw_scores)
                    same_raw=np.array_equal(replay_raw,early_raw)
                    early_scores=np.asarray([c.score for c in early.conformers],float)
                    early_xyz=np.asarray([[a.position for a in c.complex.atoms] for c in early.conformers],float)
                    same_delivered=np.array_equal(replay_scores,early_scores) and np.array_equal(replay_xyz,early_xyz)
                    saved_steps=args.steps-int(SR.LAST.get("steps_executed",args.steps))
                    row.update(early_stop_wall_s=early_wall,early_stop_saved_steps=saved_steps,
                               early_stop_same_raw=same_raw,early_stop_same_delivered=same_delivered,
                               early_stop_mode=SR.LAST.get("mode"),early_stop_error=SR.LAST.get("error"))
                    row['early_stop_check_pass']=SR.LAST.get("mode")=="replay" and (saved_steps>0 or (same_raw and same_delivered))
                    IKM.FINISH_EARLY_STOP=SR.FINISH_EARLY_STOP=False
            except BaseException as exc:
                row.update(reason=f"{type(exc).__name__}: {str(exc)[:250]}", traceback=traceback.format_exc()[-1500:])
            result["cases"][code] = row
            print(code, row.get("status"), "parity=", row.get("parity"), "replayed=", row.get("replayed_steps"), flush=True)
    result["all_pass"] = len(result["cases"]) == 3 and all(r.get("status") == "ok" and r.get("replay_mode") == "replay" and r.get("parity") and r.get("early_stop_check_pass",True) for r in result["cases"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("all_pass:", result["all_pass"])


if __name__ == "__main__":
    main()
