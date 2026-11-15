"""jdev -- routine Jacobik development tasks as single commands. Run from the repo root:

  python -m tools.jdev fingerprint                  code / DB / model hashes (paste into any report)
  python -m tools.jdev guards                       fast regression guards (pytest tests, ~1 min)
  python -m tools.jdev panel [--jobs 2] [...]       accuracy/speed panel vs baseline (~5-10 min)
  python -m tools.jdev cases                        list case names
  python -m tools.jdev case VUCVUN [--start etkdg|full_oracle|junction_oracle] [--seed 11] [--energy]
  python -m tools.jdev diagnose YARHIM [--solve JSON]   where does the excess sit? (donors, per-atom energy
                                                    map vs crystal, worst angles, close contacts)
  python -m tools.jdev audit VUCVUN [--xyz file.xyz] [--relax 500]
                                                    is the energy difference trustworthy? (composition,
                                                    charge/mult, protonation, single-point vs basin energy)

Every `case` / `audit` result is written to runs/<date>/... together with the fingerprint, so any
number can be traced to the exact code + DB that produced it.
Options shared by case/audit: --db PATH (default data/refit/final/constraints_2026-10-17.json or $JACOBIK_DB),
--mace PATH (default $JACOBIK_MACE_MODEL_PATH; dev/tools/jdev/MACE-POLAR-1-S.model is the budget tools' model).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from . import env


def _print_json(d):
    print(json.dumps(d, indent=1, default=float))


def cmd_fingerprint(a):
    _print_json(env.fingerprint(env.db_path(a.db), env.mace_path(a.mace)))


def cmd_guards(a):
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", str(env.CODE_REPO / "dev" / "tests"), "-q", *a.rest],
                             cwd=str(env.REPO)))


def cmd_panel(a):
    args = [sys.executable, str(env.CODE_REPO / "dev" / "tests" / "regression" / "run_panel.py"), "--db", str(env.db_path(a.db))]
    m = env.mace_path(a.mace)
    if m:
        args += ["--mace-model", str(m)]
    sys.exit(subprocess.call(args + a.rest, cwd=str(env.REPO)))


def cmd_cases(a):
    from .cases import panel_case_names

    env.use_src()
    print("\n".join(panel_case_names()))
    print("\n(any tmQM split: tmqm:<dir>:<split>:<CODE>)")


def cmd_case(a):
    from . import measure
    from .cases import get_case

    db, mace = env.db_path(a.db), env.mace_path(a.mace) if a.energy else None
    case = get_case(a.name)
    stats = env.load_stats(db, a.q1_variant, a.orientation_variant)
    skw = json.loads(a.solve) if a.solve else {}
    res, best, sy, X = measure.solve(case, stats, start=a.start, seed=a.seed, embed_seed=a.embed_seed, **skw)
    res["checks"] = measure.chem_checks(case, sy, X)
    if a.energy:
        res.update(measure.energies(case, env.mace_evaluator(mace), sy, X))
    res["fingerprint"] = env.fingerprint(db, mace)
    res["fingerprint"].update(q1_variant=stats.q1_variant, orientation_variant=stats.orientation_variant, solve_kw=a.solve)
    path = env.save_run("case", case.name.replace(":", "_"), res,
                        (sy, X, f"{case.name} start={a.start} score={res['score']:.4f} charge={case.charge} mult={case.mult}"))
    brief = {k: res.get(k) for k in ("case", "start", "score", "converged", "solve_s", "n_donors", "n_donors_crystal",
                                      "max_md_err", "dE_sp")}
    _print_json(brief)
    print(f"saved: {path}")


def cmd_audit(a):
    import numpy as np

    from . import measure
    from .cases import get_case

    db, mace = env.db_path(a.db), env.mace_path(a.mace)
    case = get_case(a.name)
    if a.xyz:
        L = open(a.xyz).read().splitlines()
        sy = [l.split()[0] for l in L[2:] if l.strip()]
        X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:] if l.strip()])
        res = {"case": case.name, "xyz": a.xyz}
    else:
        res, best, sy, X = measure.solve(case, env.load_stats(db, a.q1_variant, a.orientation_variant), start=a.start,
                                         seed=a.seed, **(json.loads(a.solve) if a.solve else {}))
    res["checks"] = measure.chem_checks(case, sy, X)
    res.update(measure.energies(case, env.mace_evaluator(mace), sy, X, relax_steps=a.relax))
    res["fingerprint"] = env.fingerprint(db, mace)
    path = env.save_run("audit", case.name.replace(":", "_"), res)
    _print_json({k: v for k, v in res.items() if k not in ("donors", "fingerprint")})
    print(f"saved: {path}")


def cmd_diagnose(a):
    from . import measure
    from .cases import get_case
    from .diagnose import diagnose

    db, mace = env.db_path(a.db), env.mace_path(a.mace)
    case = get_case(a.name)
    stats = env.load_stats(db, a.q1_variant, a.orientation_variant)
    skw = json.loads(a.solve) if a.solve else {}
    res, best, sy, X = measure.solve(case, stats, start=a.start, seed=a.seed, **skw)
    ev = env.mace_evaluator(mace)
    res.update(measure.energies(case, ev, sy, X))
    res["diagnosis"] = diagnose(case, best, sy, X, ev)
    res["fingerprint"] = env.fingerprint(db, mace)
    res["fingerprint"].update(q1_variant=stats.q1_variant, orientation_variant=stats.orientation_variant, solve_kw=a.solve)
    path = env.save_run("diagnose", case.name.replace(":", "_"), res,
                        (sy, X, f"{case.name} start={a.start} score={res['score']:.4f} charge={case.charge} mult={case.mult}"))
    dg = res["diagnosis"]
    print(f"== {case.name}: dE_sp {res.get('dE_sp', float('nan')):.0f} kJ/mol  score {res['score']:.2f}  "
          f"converged {res['converged']}/{res['n_conformers']}  local excess {dg['local_excess_total']:.0f}")
    print("   donors (crystal -> solved): " + "  ".join(f"{s}:{r:.2f}->{g:.2f}" for s, r, g in res.get("donors", [])))
    print("   ligands: " + " | ".join(f"{i}: {s}" for i, s in enumerate(dg["ligand_smiles"])))
    print("   excess by ligand: " + str(dg["excess_by_ligand"]))
    for t in dg["top_atoms"]:
        print(f"   {t['atom']:>5s} lig {t['ligand']} {'DONOR' if t['donor'] else '     '} heavy-nbrs {t['n_heavy_nbrs']} "
              f"excess {t['excess_kJ']:7.1f}  | " + "; ".join(t["worst_angles"]))
    print(f"   close contacts ({dg['n_close_contacts']}): " + ", ".join(dg["close_contacts"][:8]))
    print(f"saved: {path}")


def main():
    ap = argparse.ArgumentParser(prog="python -m tools.jdev", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=None)
    common.add_argument("--mace", default=None)
    sub.add_parser("fingerprint", parents=[common]).set_defaults(fn=cmd_fingerprint)
    p = sub.add_parser("guards", parents=[common]); p.add_argument("rest", nargs=argparse.REMAINDER); p.set_defaults(fn=cmd_guards)
    p = sub.add_parser("panel", parents=[common]); p.add_argument("rest", nargs=argparse.REMAINDER); p.set_defaults(fn=cmd_panel)
    sub.add_parser("cases", parents=[common]).set_defaults(fn=cmd_cases)
    for name, fn in (("case", cmd_case), ("audit", cmd_audit), ("diagnose", cmd_diagnose)):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("name")
        p.add_argument("--start", default="etkdg", choices=("etkdg", "full_oracle", "junction_oracle", "bodies_oracle"))
        p.add_argument("--q1-variant", default=None, help="override the DB sidecar Q1 ladder (e.g. mo_state)")
        p.add_argument("--orientation-variant", default=None, help="override the DB sidecar Q3/Q4 ladder")
        p.add_argument("--seed", type=int, default=11)
        p.add_argument("--solve", default=None,
                       help='extra solve() keyword arguments as JSON, e.g. \'{"angle_bends": true}\' or \'{"orientation_model": "q3q4"}\'')
        if name == "diagnose":
            pass
        elif name == "case":
            p.add_argument("--embed-seed", type=int, default=None)
            p.add_argument("--energy", action="store_true", help="also compute MACE dE_sp")
        if name == "audit":
            p.add_argument("--xyz", default=None, help="audit this structure instead of solving")
            p.add_argument("--relax", type=int, default=0, help="local MACE relaxation steps for dE_basin/strain (0 = off)")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
