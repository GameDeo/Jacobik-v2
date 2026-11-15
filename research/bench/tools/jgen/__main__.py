"""CLI (research/archive/docs/roadmap/roadmap_generation_v1.md 1.3).

    python -m tools.jgen diag <script-name> [args]            run tools/jgen/diag/<script-name>.py
    python -m tools.jgen fingerprint-check                    T-P0 under the control arm and bare u1; exit 0 / 1
    python -m tools.jgen bench selfcheck --root DIR --manifest F --repair F [--panel all] --out DIR
    python -m tools.jgen bench simple --candidate DIR --reference DIR --manifest F --repair F --panel P --out DIR
    python -m tools.jgen bench run ...                        GPU arm runner (tools/jgen/bench/run.py)

The bench never writes under data/. --reference is the frozen frame_v1 root or its pools/ directory.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DIAG = Path(__file__).resolve().parent / "diag"
CONTROL_OVERRIDES = {"nci": False, "deterministic_embed": True}


def _root_of(p: str) -> str:
    return os.path.dirname(os.path.abspath(p)) if os.path.basename(os.path.normpath(p)) == "pools" else p


def cmd_fingerprint_check() -> int:
    code = ("import json,os,sys;"
            "import jacobik.core.profile as P;"
            "print(json.dumps(P.snapshot()['fingerprint']))")
    results = {}
    for name, env_over in (("control", {"JACOBIK_PROFILE": "u1",
                                        "JACOBIK_PROFILE_OVERRIDES": json.dumps(CONTROL_OVERRIDES)}),
                           ("u1_bare", {"JACOBIK_PROFILE": "u1"})):
        env = {k: v for k, v in os.environ.items() if k not in ("JACOBIK_PROFILE", "JACOBIK_PROFILE_OVERRIDES")}
        env.update(env_over)
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), str(REPO / "research" / "bench")])
        out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=str(REPO), text=True)
        results[name] = json.loads(out.strip().splitlines()[-1])
    want = {"control": "485e943af036", "u1_bare": "18dec82067b1"}
    ok = all(results[k] == want[k] for k in want)
    print(json.dumps(dict(results=results, expected=want, ok=ok)))
    return 0 if ok else 1


def cmd_diag(name: str, args) -> int:
    script = DIAG / f"{name}.py"
    if not script.exists():
        print(f"no diagnostic {name!r} in {DIAG}")
        return 2
    sys.argv = [str(script)] + args
    runpy.run_path(str(script), run_name="__main__")
    return 0


def cmd_bench(a) -> int:
    from tools.jgen.bench import outputs, selfcheck, simple
    if a.bench_cmd == "selfcheck":
        root = _root_of(a.root)
        res = {"control_vs_self": selfcheck.control_vs_self(root, a.manifest, a.repair, a.panel),
               "control_vs_s2b": selfcheck.control_vs_s2b(root, a.manifest, a.repair, a.panel)}
        outputs.write_json(os.path.join(a.out, "selfcheck.json"), res)
        print(json.dumps({k: {kk: v[kk] for kk in ("units_paired", "parents_paired", "parent_band_exceed_count",
                                                  "units_gt10") if kk in v} for k, v in res.items()}, default=len))
        return 0
    if a.bench_cmd == "simple":
        if not a.control_dir and not a.reference:
            raise SystemExit("bench simple: give --control-dir or --reference")
        cand = simple.read_candidate_dir(a.candidate)
        if a.control_dir:
            s = simple.run(a.panel, cand, None, a.manifest, a.repair,
                           control=simple.read_candidate_dir(a.control_dir))
        else:
            s = simple.run(a.panel, cand, _root_of(a.reference), a.manifest, a.repair)
        os.makedirs(a.out, exist_ok=True)
        outputs.write_json(os.path.join(a.out, f"simple_{a.panel}.json"), s)
        with open(os.path.join(a.out, f"simple_{a.panel}.md"), "w") as fh:
            fh.write(outputs.arm_block(os.path.basename(os.path.normpath(a.candidate)), s))
        print(json.dumps({k: s[k] for k in ("panel", "units_paired", "parents_paired", "parent_band_exceed_count")}))
        return 0
    if a.bench_cmd == "report":
        from tools.jgen.bench import report
        return report.main(a)
    if a.bench_cmd == "run":
        from tools.jgen.bench import run as runner
        return runner.main(a.rest)
    return 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:2] == ["bench", "run"]:  # the GPU runner has its own options
        from tools.jgen.bench import run as runner
        return runner.main(argv[2:])
    if argv[:2] in (["bench", "replay-parity"], ["bench", "replay_parity"]):
        from tools.jgen.bench import replay_parity
        return replay_parity.main(argv[2:])
    p = argparse.ArgumentParser(prog="python -m tools.jgen")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fingerprint-check")
    d = sub.add_parser("diag")
    d.add_argument("name")
    d.add_argument("rest", nargs=argparse.REMAINDER)
    b = sub.add_parser("bench")
    bs = b.add_subparsers(dest="bench_cmd", required=True)
    sc = bs.add_parser("selfcheck")
    sc.add_argument("--root", required=True)
    sc.add_argument("--manifest", required=True)
    sc.add_argument("--repair", required=True)
    sc.add_argument("--panel", default="all")
    sc.add_argument("--out", required=True)
    sm = bs.add_parser("simple")
    sm.add_argument("--candidate", required=True)
    sm.add_argument("--reference", default="", help="frozen frame_v1 root (stored Step 2 pools)")
    sm.add_argument("--control-dir", default="", help="records dir of a control bench run (paired baseline; "
                                                     "replaces --reference)")
    sm.add_argument("--manifest", required=True)
    sm.add_argument("--repair", required=True)
    sm.add_argument("--panel", required=True, choices=["tail", "cal", "val", "failures", "all"])
    sm.add_argument("--out", required=True)
    rp = bs.add_parser("report")
    rp.add_argument("--labels", required=True)
    rp.add_argument("--reference", required=True)
    rp.add_argument("--manifest", required=True)
    rp.add_argument("--repair", required=True)
    rp.add_argument("--arm", action="append", required=True, help="NAME=RECORDS_DIR")
    rp.add_argument("--panel", required=True, choices=["tail", "cal", "val", "failures", "all"])
    rp.add_argument("--out", required=True)
    rp.add_argument("--control-dir", default="", help="records dir of a control bench run (paired baseline)")
    rn = bs.add_parser("run")
    rn.add_argument("rest", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.cmd == "fingerprint-check":
        return cmd_fingerprint_check()
    if a.cmd == "diag":
        return cmd_diag(a.name, a.rest)
    return cmd_bench(a)


if __name__ == "__main__":
    sys.exit(main())
