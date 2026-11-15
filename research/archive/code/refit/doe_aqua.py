"""doe_aqua.py -- DOE aqua benchmark (Ln(H2O)n 3+ and Fe(H2O)6 2+/3+) as a stand-alone process, so the SAME script can be
run against the production `src` (baseline) and `src_next` (refit) without the two importing each other.

    python doe_aqua.py --src <dir containing jacobik/> --db constraints.json --out result.json [--shannon] [--seed 11]
                       [--doe-dir <folder with Ln-xyz/ and misc-metal-xyz/>] [--fe-seeds 11,12,23,34,53]

Reports per system: reference mean M-O, best-scoring conformer's mean M-O, and the water H-orientation error (angle
between the O->M vector and the H-O-H bisector, conformer vs reference).  DOE is a VALIDATION set: nothing is fitted on it.
"""
import argparse, glob, json, os, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--db", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--doe-dir", default=None); ap.add_argument("--shannon", action="store_true")
ap.add_argument("--seed", type=int, default=11); ap.add_argument("--fe-seeds", default="11,12,23,34,53")
ap.add_argument("--no-oxidation-state-arg", action="store_true", help="baseline src has no oxidation_state kwarg (never passed anyway)")
ap.add_argument("--mace-model-path", default=None, help="local MACE-POLAR checkpoint for matched energy deltas")
ap.add_argument(
    "--score-all-conformers",
    action="store_true",
    help="with --mace-model-path, score the full returned ensemble and report ranking regret",
)
a = ap.parse_args()
sys.path.insert(0, a.src)
import logging; logging.getLogger("jacobik").setLevel(logging.ERROR)
from jacobik import ComplexBuilder, LigandInput
from jacobik.stats.stats import StatsLookup, load_db

SPIN = {'La': 1, 'Ce': 2, 'Pr': 3, 'Nd': 4, 'Pm': 5, 'Sm': 6, 'Eu': 7, 'Gd': 8, 'Tb': 7, 'Dy': 6, 'Ho': 5, 'Er': 4, 'Tm': 3, 'Yb': 2, 'Lu': 1, 'Fe2': 5, 'Fe3': 6}
GEOM = {8: 'square_anti', 9: 'tricap_pris', 6: 'octahedral'}
if a.doe_dir:
    base = a.doe_dir.rstrip("/") + "/"
else:
    root = os.getcwd()
    base = glob.glob(os.path.join(root, "data", "10.25584 - 3019314", "*", "opt-xyz"))[0] + "/"
FILES = sorted(glob.glob(base + "Ln-xyz/Ln-H2O_x/*.xyz")) + [base + "misc-metal-xyz/metal-H2O_x/Fe2_6H2O.xyz", base + "misc-metal-xyz/metal-H2O_x/Fe3_6H2O.xyz"]

try:
    ST = StatsLookup(load_db(a.db), shannon=a.shannon)
except TypeError:            # production StatsLookup has no shannon switch
    ST = StatsLookup(load_db(a.db))

ENERGY = None
if a.mace_model_path:
    from jacobik.benchmark.energy import EV_TO_KJ_MOL, MACEPolarEvaluator
    ENERGY = MACEPolarEvaluator(model_path=a.mace_model_path)


def read(f):
    L = [x.split() for x in open(f).read().splitlines()[2:] if x.strip()]
    return [x[0] for x in L], np.array([[float(v) for v in x[1:4]] for x in L])


def tilt(sy, X, mi):
    """mean angle (deg) between O->M and the H-O-H bisector over waters bound to the metal at index mi"""
    out = []
    for i, s in enumerate(sy):
        if s != 'O' or np.linalg.norm(X[i] - X[mi]) > 3.2:
            continue
        hs = [j for j, t in enumerate(sy) if t == 'H' and np.linalg.norm(X[j] - X[i]) < 1.25]
        if len(hs) != 2:
            continue
        b = (X[hs[0]] + X[hs[1]]) / 2 - X[i]; m = X[mi] - X[i]
        c = np.dot(b, m) / (np.linalg.norm(b) * np.linalg.norm(m))
        out.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return float(np.mean(out)) if out else float('nan')


def _rankdata(values):
    """Average ranks for a short vector, without adding a scipy dependency."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _spearman(x, y):
    if len(x) < 2:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def score_ensemble(result, charge, spin, reference_ev):
    """Separate search coverage from Jacobik-to-energy ranking quality."""
    ensemble = []
    for rank, conformer in enumerate(
        sorted(result.conformers, key=lambda c: c.score, reverse=True), 1
    ):
        coords = np.asarray([atom.position for atom in conformer.complex.atoms])
        symbols = [atom.symbol for atom in conformer.complex.atoms]
        energy_ev = ENERGY.energy(symbols, coords, charge, spin)
        ensemble.append(
            {
                "jacobik_rank": rank,
                "jacobik_score": float(conformer.score),
                "mace_energy_ev": energy_ev,
                "mace_delta_kj_mol": (energy_ev - reference_ev) * EV_TO_KJ_MOL,
            }
        )
    energies = np.asarray([r["mace_energy_ev"] for r in ensemble])
    scores = np.asarray([r["jacobik_score"] for r in ensemble])
    oracle_i = int(np.argmin(energies))
    return {
        "n_conformers_scored": len(ensemble),
        "oracle_jacobik_rank": oracle_i + 1,
        "oracle_delta_kj_mol": float((energies[oracle_i] - reference_ev) * EV_TO_KJ_MOL),
        "ranking_regret_kj_mol": float((energies[0] - energies[oracle_i]) * EV_TO_KJ_MOL),
        # A useful scorer has a negative correlation: higher Jacobik score,
        # lower MACE energy.  NaN is emitted as null by the final sanitiser.
        "score_energy_spearman": _spearman(scores, energies),
        "conformers": ensemble,
    }


def solve(metal, q, cn, seed):
    return ComplexBuilder(metal, GEOM[cn], charge=q, spin=SPIN[metal if metal != 'Fe' else f'Fe{q}'],
                          ligands=[LigandInput('O', donors=[[0]]) for _ in range(cn)]).solve(stats=ST, seed=seed)


rows = []
for i, f in enumerate(FILES, 1):
    m = re.match(r'([A-Z][a-z]?)(\d)_(\d)H2O', os.path.basename(f)); metal, q, cn = m.group(1), int(m.group(2)), int(m.group(3))
    sy, X = read(f); mi = sy.index(metal)
    dr = [np.linalg.norm(X[i] - X[mi]) for i, s in enumerate(sy) if s == 'O' and np.linalg.norm(X[i] - X[mi]) < 3.2]
    if len(dr) != cn:
        continue
    res = solve(metal, q, cn, a.seed)
    best = max(res.conformers, key=lambda c: c.score)        # the engine MAXIMISES score
    Y = np.array([at.position for at in best.complex.atoms]); ys = [at.symbol for at in best.complex.atoms]
    row = dict(system=f"{metal}{q}", cn=cn, ref=float(np.mean(dr)),
               got=float(np.mean([np.linalg.norm(Y[i] - Y[0]) for i, s in enumerate(ys) if s == 'O'])),
               tilt_ref=tilt(sy, X, mi), tilt_got=tilt(ys, Y, 0))
    if ENERGY is not None:
        e_ref = ENERGY.energy(sy, X, q, SPIN[metal if metal != 'Fe' else f'Fe{q}'])
        e_got = ENERGY.energy(ys, Y, q, SPIN[metal if metal != 'Fe' else f'Fe{q}'])
        row.update(mace_reference_ev=e_ref, mace_generated_ev=e_got,
                   mace_delta_ev=e_got - e_ref,
                   mace_delta_kj_mol=(e_got - e_ref) * EV_TO_KJ_MOL)
        if a.score_all_conformers:
            row["ensemble_energy"] = score_ensemble(
                res, q, SPIN[metal if metal != 'Fe' else f'Fe{q}'], e_ref
            )
    rows.append(row)
    print(f"scored {i}/{len(FILES)} {row['system']}/{cn}", flush=True)

fe = []
sy, X = read(base + "misc-metal-xyz/metal-H2O_x/Fe3_6H2O.xyz")
for sd in [int(s) for s in a.fe_seeds.split(",")]:
    r = solve('Fe', 3, 6, sd); c = max(r.conformers, key=lambda c: c.score)
    Y = np.array([at.position for at in c.complex.atoms]); ys = [at.symbol for at in c.complex.atoms]
    d = np.array([np.linalg.norm(Y[i] - Y[0]) for i, s in enumerate(ys) if s == 'O'])
    fe.append(dict(seed=sd, score=float(c.score), FeO_mean=float(d.mean()), FeO_max=float(d.max())))
def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


json.dump(_json_safe(dict(rows=rows, fe_seeds=fe)), open(a.out, "w"), indent=1)
print("wrote", a.out, len(rows), "systems")
