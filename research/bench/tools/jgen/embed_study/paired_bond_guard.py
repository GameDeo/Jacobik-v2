"""Paired top-1 strain and unrelaxed-energy screen for bond-guard footprint parents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]

from tools.runkit import resources  # noqa: E402
from tools.runkit.policy import assert_allowed  # noqa: E402


def _record(path: Path, code: str):
    return json.loads((path / f"{code}.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--footprint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    scan = json.loads(args.footprint.read_text())
    codes = scan["parents"]["overall_unique"]["trigger_parents"]
    assert_allowed(codes)
    args.output.mkdir(parents=True, exist_ok=True)
    logdir = args.output / "logs"
    budget = resources.Budget("Codex")
    results = []
    for code in codes:
        for arm, flags in (("control", {}), ("bond_guard", {"embed_bond_check": True})):
            out = args.output / arm
            target = out / f"{code}.json"
            if not target.exists():
                env = resources.env(threads=1, cuda=False,
                                   extra={"JACOBIK_PROFILE_OVERRIDES": json.dumps(flags)})
                cmd = [sys.executable, "-m", "tools.runkit.trope.suite", "parent", "--code", code,
                       "--arm", arm, "--flags", json.dumps(flags), "--seeds", "11", "--out", str(out)]
                proc = resources.spawn(cmd, budget, f"f3-{arm}-{code}", log=logdir / f"{arm}-{code}.log",
                                       env=env, cpus=1)
                rc = proc.wait()
                if rc != 0 and not target.exists():
                    continue
        control_path = args.output / "control"
        arm_path = args.output / "bond_guard"
        if not (control_path / f"{code}.json").exists() or not (arm_path / f"{code}.json").exists():
            results.append({"code": code, "status": "missing_pair"})
            continue
        c, a = _record(control_path, code), _record(arm_path, code)
        cr, ar = c.get("per_seed", {}).get("11", {}), a.get("per_seed", {}).get("11", {})
        if cr.get("status") != "ok" or ar.get("status") != "ok":
            results.append({"code": code, "status": "solve_failed", "control": c.get("status"),
                            "arm": a.get("status"), "control_reason": c.get("reason"), "arm_reason": a.get("reason")})
            continue
        results.append({"code": code, "status": "ok", "control_strain": cr.get("strain"),
                        "guard_strain": ar.get("strain"),
                        "delta_strain": None if cr.get("strain") is None or ar.get("strain") is None else ar["strain"] - cr["strain"],
                        "control_dE_unrel": cr.get("dE_unrel"), "guard_dE_unrel": ar.get("dE_unrel"),
                        "delta_dE_unrel": None if cr.get("dE_unrel") is None or ar.get("dE_unrel") is None else ar["dE_unrel"] - cr["dE_unrel"],
                        "control_host": c.get("host"), "guard_host": a.get("host"),
                        "control_fingerprint": c.get("fingerprint"), "guard_fingerprint": a.get("fingerprint")})
    good = [r for r in results if r["status"] == "ok"]
    import numpy as np
    summary = {k: (float(np.median([r[k] for r in good if r[k] is not None]))
                   if any(r[k] is not None for r in good) else None)
               for k in ("delta_strain", "delta_dE_unrel")}
    report = {"codes": codes, "results": results, "median_paired_delta": summary,
              "n_ok": len(good), "n_failed": len(results) - len(good), "seed": 11,
              "workers": 1, "note": "paired by parent; CPU-only; top-1 metrics from runkit parent records"}
    (args.output / "paired_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"n_parents": len(codes), "n_ok": len(good), "n_failed": len(results)-len(good),
                      "median_paired_delta": summary, "output": str(args.output / 'paired_summary.json')}))


if __name__ == "__main__":
    main()
