"""DOE input parity: jacobik's own case construction vs the exported Architector inputs (root jacobik env).

The DOE constructor is run exactly as tests/regression/run_panel.py does it for `aqua_<name>`:
    ComplexBuilder(metal, GEOM[cn], charge=q, spin=SPIN[...], ligands=[LigandInput("O", donors=[[0]])] * cn)
SPIN and GEOM are read from that file by AST (no import, so the runner's module code does not execute).
Asserts equality of metal, oxidation state, charge, multiplicity, coordination number and donor indices with the
exported Architector spec (both native and matched records, if present). Writes parity_doe.json into the run dir.

Usage: python -m tools.architector.doe_parity --run-dir <run dir> --cases aqua_Er3_8H2O,aqua_Fe2_6H2O
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from tools.architector.export_inputs import REPO

RUN_PANEL = REPO / "tests/regression/run_panel.py"


def _literal(name):
    tree = ast.parse(RUN_PANEL.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cases", default="aqua_Er3_8H2O,aqua_Fe2_6H2O")
    a = ap.parse_args(argv)
    from jacobik import ComplexBuilder, LigandInput  # the production constructor, run for real
    SPIN, GEOM = _literal("SPIN"), _literal("GEOM")
    out = {}
    ok = True
    for case in a.cases.split(","):
        name = case[len("aqua_"):]
        metal, q, cn = re.match(r"([A-Z][a-z]?)(\d)_(\d)H2O", name).groups()
        q, cn = int(q), int(cn)
        spin = SPIN[metal if metal != "Fe" else f"Fe{q}"]
        builder = ComplexBuilder(metal, GEOM[cn], charge=q, spin=spin,
                                 ligands=[LigandInput("O", donors=[[0]]) for _ in range(cn)])
        jac = dict(metal=builder.metal, oxidation_state=int(builder.oxidation_state), charge=int(builder.charge),
                   multiplicity=int(builder.spin), coordination_number=len(builder.ligands),
                   donors=[[list(map(int, s)) for s in li.donors] for li in builder.ligands],
                   geometry=builder.geometry)
        rec = json.loads((Path(a.run_dir) / "inputs" / "doe_aqua" / f"{case}.json").read_text())
        checks = {}
        for mode, body in rec["modes"].items():
            if body["status"] != "direct":
                checks[mode] = "capability (not compared)"
                continue
            spec = body["spec"]
            arch = dict(metal=spec["core"]["metal"],
                        oxidation_state=int(spec["parameters"]["metal_ox"]),
                        charge=int(spec["parameters"]["full_charge"]),
                        multiplicity=int(spec["parameters"]["full_spin"]) + 1,
                        coordination_number=sum(len(li["coordList"]) for li in spec["ligands"]),
                        donors=[[[d] for d in li["coordList"]] for li in spec["ligands"]])
            diffs = {k: (jac[k], arch.get(k)) for k in ("metal", "oxidation_state", "charge", "multiplicity",
                                                         "coordination_number")
                     if jac[k] != arch.get(k)}
            # donor index of each ligand instance: jacobik [[0]] per site vs Architector coordList [0]
            if [li[0] for li in jac["donors"]] != [li[0] for li in arch["donors"]]:
                diffs["donors"] = (jac["donors"], arch["donors"])
            checks[mode] = "equal" if not diffs else {"differs": diffs}
            ok = ok and not diffs
        out[case] = dict(jacobik=jac, checks=checks)
        print(case, checks, flush=True)
    out["all_equal"] = ok
    (Path(a.run_dir) / "parity_doe.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("all_equal", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
