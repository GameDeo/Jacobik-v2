"""
Batch solver for the PI's lanthanide/Fe aqua & nitrato comparison task.
Builds all 12 requested complexes, solves each, saves the best conformer
as .xyz, and prints a compact summary table.
"""
import sys, os, json, time
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pi_task_results"
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, "/tmp/jscripts")

from pi_task_ligands import build_aqua, build_nitrato_bidentate

# spin multiplicities (2S+1), ground state, for reference/metadata only
SPIN = {"La": 1, "Nd": 4, "Dy": 6, "Fe": 6}

CASES = []
for m in ["La", "Nd", "Dy"]:
    CASES.append((f"{m}_aqua_y8", lambda m=m: build_aqua(m, 8, charge=3, spin=SPIN[m])))
    CASES.append((f"{m}_aqua_y9", lambda m=m: build_aqua(m, 9, charge=3, spin=SPIN[m])))
for m in ["La", "Nd", "Dy"]:
    CASES.append((f"{m}_nitrato_y7", lambda m=m: build_nitrato_bidentate(m, 7, charge=2, spin=SPIN[m])))
CASES.append(("Fe_aqua_y6", lambda: build_aqua("Fe", 6, charge=3, spin=SPIN["Fe"])))
CASES.append(("Fe_nitrato_y4", lambda: build_nitrato_bidentate("Fe", 4, charge=2, spin=SPIN["Fe"])))
CASES.append(("Fe_nitrato_y5", lambda: build_nitrato_bidentate("Fe", 5, charge=2, spin=SPIN["Fe"])))

summary = []
for name, build_fn in CASES:
    t0 = time.time()
    try:
        builder = build_fn()
        result = builder.solve(seed=0)
        best = result.best
        dt = time.time() - t0
        xyz_path = os.path.join(OUT, f"{name}.xyz")
        best.write_xyz(xyz_path)
        n_conv = len(result.converged)
        n_tot = len(result.conformers)
        scores = sorted({round(c.score, 3) for c in result.conformers}, reverse=True)
        row = dict(
            name=name, ok=True, dt=round(dt, 1),
            best_score=round(best.score, 4), converged=best.converged,
            n_converged=n_conv, n_total=n_tot,
            n_distinct_scores=len(scores), top_scores=scores[:3],
            geometry=builder.geometry, cn=len(best.complex.coord_sites),
            xyz=xyz_path,
        )
    except Exception as e:
        dt = time.time() - t0
        row = dict(name=name, ok=False, dt=round(dt, 1), error=str(e))
        import traceback; traceback.print_exc()
    summary.append(row)
    print(json.dumps(row))

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone. {sum(r['ok'] for r in summary)}/{len(summary)} solved OK. Results in {OUT}")
