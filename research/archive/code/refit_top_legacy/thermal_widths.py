"""Thermal widths of the scorer's features: the physical minimum width of a table component (plan U1, 2026-10-16).

Why: the solve scored Q1 with a 0.2 A variance floor that has no justification beyond "it worked", while the fitted
tables contain components a few thousandths of an A wide (spreads of minima across molecules, not physics). The
justified floor is the thermal width: a component cannot be narrower than the room-temperature fluctuation of the
coordinate it describes.

How (no per-key measurement): in a harmonic well the classical configurational covariance of any feature f is
    var_T(f) = kT * g^T H^+ g,      g = df/dX,   H^+ = pseudo-inverse of the Cartesian Hessian (rigid-body modes out)
(compliance constants, Decius 1963; "relaxed force constants", Grunenberg 2004). tmQM geometries are GFN2-xTB
minima, so the GFN2 Hessian (`xtb --hess`) at the reference is taken at a stationary point. g is computed by autograd
through the scorer's OWN geometry code (GeometryExtractor: Q1 distances, Q2 cos D-M-D, Q5 torsions; S_orient
features), so sigma_T is in exactly the units and definitions the tables use. Training complexes only (tmQM X1, the
188 benchmark / dev codes excluded).

    python refit/thermal_widths.py sample --n 300            -> data/refit/thermal/sample.json
    python refit/thermal_widths.py hess --workers 10          -> data/refit/thermal/hess/<code>.npy (GFN2, CPU)
    python refit/thermal_widths.py sigma                      -> data/refit/thermal/features.csv
    python refit/thermal_widths.py fit                        -> data/refit/final/thermal_widths.json + report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO, REPO / "tools" / "jdev" / "budget"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
TMQM = REPO / "data" / "tmQM"
OUT = REPO / "data" / "refit" / "thermal"
FINAL = REPO / "data" / "refit" / "final" / "thermal_widths.json"
DB = Path(os.environ.get("JACOBIK_BUDGET_DB", REPO / "data/refit/final/constraints_refit_q6p5.json"))
XTB = os.environ.get("JACOBIK_XTB") or shutil.which("xtb") or str(REPO / ".pixi/envs/default/Library/bin/xtb.exe")
KT = 2.479  # kJ/mol, 298 K
EH_BOHR2 = 2625.4996 / 0.529177210903 ** 2  # Eh/bohr^2 -> kJ/mol/A^2
ROW = {**{e: "3d" for e in "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()},
       **{e: "4d" for e in "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd".split()},
       **{e: "5d" for e in "La Hf Ta W Re Os Ir Pt Au Hg".split()}}


def _quiet():
    import logging
    import warnings
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("jacobik").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _case(m):
    """tools.jdev.cases.Case for a parsed tmQM molecule (as cases._tmqm, without re-streaming the split)."""
    from jacobik import ComplexBuilder
    from jacobik.vis.benchmark import _build_ligand_groups, ligand_inputs_from_mol
    from tools.jdev.cases import Case

    groups = _build_ligand_groups(m, m.smiles)
    lig_in = ligand_inputs_from_mol(m, groups)
    if lig_in is None:
        return None
    rows = {a.idx: i for i, a in enumerate(m.atoms)}
    return Case(name=m.csd_code,
                make_builder=lambda: ComplexBuilder(metal=m.metal_symbol, geometry=m.coord_geom, charge=m.charge,
                                                    spin=m.spin_state, ligands=lig_in),
                charge=m.charge, mult=m.spin_state, ref_symbols=[a.symbol for a in m.atoms],
                ref_X=np.array([a.position for a in m.atoms]), ref_metal_row=rows[m.metal_idx], tmqm_mol=m,
                ligand_groups=groups, ligand_inputs=lig_in)


def _headers():
    """(csd_code, n_atoms, metal) per X1 block from the raw xyz file: seconds, no molecule assembly."""
    from jacobik.stats.parser import _iter_xyz_blocks
    for comment, lines in _iter_xyz_blocks(TMQM / "xyz" / "tmQM_X1.xyz"):
        code = comment.split("|")[0].split("=")[1].strip()
        metals = {ln.split()[0] for ln in lines} & set(ROW)
        yield code, len(lines), (metals.pop() if len(metals) == 1 else None)


def cmd_sample(a):
    """Stratified sample of tmQM X1 training complexes: metal rows 3d / 4d / 5d balanced, at most `per_metal` per metal,
    <= max_atoms atoms, SMILES mappable, no benchmark / dev code. Chosen from the raw headers; only the candidates are
    assembled (to drop the unmappable ones)."""
    _quiet()
    hold = json.loads((REPO / "refit/benchmark_holdout_codes.json").read_text())
    excl = {c for k, v in hold.items() if isinstance(v, list) for c in v}
    rng = np.random.default_rng(a.seed)
    by_metal, seen = {}, 0
    for code, n, metal in _headers():
        seen += 1
        if code not in excl and n <= a.max_atoms and metal is not None:
            by_metal.setdefault(metal, []).append(code)
    per_row = a.n // 3
    rows = {}
    for metal, codes in sorted(by_metal.items()):
        rng.shuffle(codes)
        rows.setdefault(ROW[metal], []).extend(codes[: a.per_metal])
    cand = []
    for r, codes in sorted(rows.items()):
        rng.shuffle(codes)
        cand += codes[: int(per_row * 1.8)]  # headroom for unmappable SMILES
    ok = {m.csd_code: ROW[m.metal_symbol] for m in _molecules(cand) if _case(m) is not None}
    pick, taken = [], {}
    for c in cand:
        r = ok.get(c)
        if r is not None and taken.get(r, 0) < per_row:
            pick.append(c)
            taken[r] = taken.get(r, 0) + 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps({"split": "X1", "codes": pick, "seed": a.seed, "streamed": seen,
                                                  "per_row": taken}, indent=1))
    print(f"sampled {len(pick)} of {seen} ({taken}; {len(cand) - len(ok)} of {len(cand)} candidates unmappable)")


def _molecules(codes):
    """Assemble only the wanted X1 blocks (raw blocks are cheap, assembly is not); stops once all are found."""
    from jacobik.stats.mapping import assemble
    from jacobik.stats.parser import TMQMParser, _assemble_raw, _attach_qm_properties, _iter_bo_blocks, _iter_xyz_blocks
    want = set(codes)
    csv_index = TMQMParser(TMQM).csv_index
    for (comment, xl), (code, bl) in zip(_iter_xyz_blocks(TMQM / "xyz" / "tmQM_X1.xyz"),
                                         _iter_bo_blocks(TMQM / "bo" / "tmQM_X1.BO")):
        if code not in want:
            continue
        want.discard(code)
        try:
            row = csv_index.get(code)
            m = assemble(_assemble_raw(comment, xl, code, bl, row))
            if row is not None:
                _attach_qm_properties(m, row)
        except Exception:
            continue
        yield m
        if not want:
            return


def _xtb_hess(sy, X, charge, mult):
    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/m.xyz", "w") as f:
            f.write(f"{len(sy)}\n\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(sy, X)))
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="500M")
        subprocess.run([XTB, "m.xyz", "--gfn", "2", "--hess", "--chrg", str(charge), "--uhf", str(mult - 1)],
                       cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600, env=env)
        txt = Path(td, "hessian").read_text()
    vals = np.array([float(v) for v in re.findall(r"-?\d+\.\d+(?:[EeDd][-+]?\d+)?", txt.replace("D", "E"))])
    n = 3 * len(sy)
    return vals[: n * n].reshape(n, n) * EH_BOHR2  # kJ/mol/A^2, crystal atom order


def cmd_hess(a):
    codes = json.loads((OUT / "sample.json").read_text())["codes"]
    hd = OUT / "hess"
    hd.mkdir(parents=True, exist_ok=True)
    jobs = [(m.csd_code, [x.symbol for x in m.atoms], np.array([x.position for x in m.atoms]), m.charge, m.spin_state)
            for m in _molecules(codes) if not (hd / f"{m.csd_code}.npy").exists()]

    def work(j):
        code, sy, X, q, mult = j
        try:
            np.save(hd / f"{code}.npy", _xtb_hess(sy, X, q, mult))
            return code, "ok"
        except Exception as exc:
            return code, f"ERROR {type(exc).__name__}: {str(exc)[:80]}"

    with ThreadPoolExecutor(a.workers) as ex:
        for k, (code, msg) in enumerate(ex.map(work, jobs), 1):
            print(f"hess {k}/{len(jobs)} {code}: {msg}", flush=True)


def _pinv_hessian(H, n_rigid=6, rel_cut=1e-6):
    """Pseudo-inverse without the rigid-body and negative (non-minimum) modes; (H^+, n negative modes dropped)."""
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    order = np.argsort(np.abs(w))
    keep = np.ones(len(w), bool)
    keep[order[:n_rigid]] = False
    neg = keep & (w < 0)
    keep &= w > rel_cut * np.abs(w).max()
    Vk = V[:, keep]
    return (Vk / w[keep]) @ Vk.T, int(neg.sum())


def cmd_sigma(a):
    _quiet()
    import pandas as pd
    import torch
    from jacobik.stats.stats import StatsLookup, load_db
    from oracle_targets import _so_features, crystal_in_engine_order

    meta = DB.with_name(DB.stem + ".meta.json")
    st = StatsLookup(load_db(json.loads(DB.read_text()), meta=json.loads(meta.read_text()) if meta.exists() else None))
    codes = a.codes.split(",") if a.codes else json.loads((OUT / "sample.json").read_text())["codes"]
    hd = OUT / "hess"
    rows = []
    for m in _molecules(codes):
        hf = hd / f"{m.csd_code}.npy"
        if not hf.exists():
            continue
        try:
            case = _case(m)
            if case is None:
                print(f"{m.csd_code}: skipped (no case)", flush=True)
                continue
            r = case.make_builder().solve(seed=11, stats=st, angle_bends=True, n_conformers=1, n_ik_steps=2,
                                          device="cpu")
            sc, mol = r.scorer, r.conformers[0].complex
            ge = sc._ge
            Xp = np.array([at.position for at in mol.atoms])
            # crystal coordinates in engine order; atoms that cannot be mapped (H whose per-parent count differs
            # between the SMILES and the crystal) keep conformer positions, and any feature that depends on one is
            # dropped below (its gradient there is non-zero) instead of dropping the whole complex
            Xc, ok = crystal_in_engine_order(case, mol, Xp)
            ref = np.asarray(case.ref_X, float)
            cmap = {}
            for e in np.flatnonzero(ok):
                c = int(np.argmin(((ref - Xc[e]) ** 2).sum(-1)))
                if ((ref[c] - Xc[e]) ** 2).sum() < 1e-8:
                    cmap[int(e)] = c
            if len(set(cmap.values())) != len(cmap):
                print(f"{m.csd_code}: skipped (atom map not 1:1)", flush=True)
                continue
            Hp, n_neg = _pinv_hessian(np.load(hf))  # crystal order
            sy = [at.symbol for at in mol.atoms]
            Xt = torch.tensor(Xc, dtype=torch.float64)

            def feats(x):
                x = x[None].float()
                # angles in angle space (rad): the cos the tables use has zero slope at 0 / 180 deg; the floor is
                # converted to cos at each component's mean when applied (sigma_cos^2 = sin^2 s^2 + cos^2 s^4 / 2)
                out = [("Q1", ge.q1_distances(x)[0]), ("Q2", ge.q2_dmd_angles(x)[0])]
                if ge.n_q5:
                    out.append(("Q5", ge.q5_torsions(x)[0]))
                M = x[:, ge.metal_pos_idx, :]
                for g in getattr(sc, "_so", []):
                    c = _so_features(x, M, g).reshape(-1).clamp(-1 + 1e-9, 1 - 1e-9)
                    out.append((f"SO{g['k']}", torch.acos(c)))
                return out

            f0 = feats(Xt)
            names = [(q, len(v)) for q, v in f0]
            vals = torch.cat([v.double() for _, v in f0]).detach().numpy()
            J = torch.autograd.functional.jacobian(lambda x: torch.cat([v.double() for _, v in feats(x)]), Xt)
            G = J.reshape(J.shape[0], len(Xc), 3).numpy()
            unm = np.array([e not in cmap for e in range(len(Xc))])
            touches = np.abs(G[:, unm]).reshape(len(G), -1).max(-1, initial=0.0) > 0 if unm.any() else np.zeros(len(G), bool)
            Gc = np.zeros((len(G), len(ref), 3))
            for e, c in cmap.items():
                Gc[:, c] = G[:, e]
            Gc = Gc.reshape(len(G), -1)
            var = KT * np.einsum("fi,ij,fj->f", Gc, Hp, Gc)
            sig = np.where(touches | (var <= 0), np.nan, np.sqrt(np.clip(var, 0, None)))  # <= 0: clamped linear angle
            # descriptors per feature
            don = []
            for site in mol.coord_sites:
                el = sorted({mol.atoms[[a_.idx for a_ in mol.atoms].index(d)].symbol for d in site.donor_indices}) \
                    if hasattr(site, "donor_indices") else ["?"]
                don.append(("eta" if getattr(site, "is_haptic", False) else "") + "".join(el))
            k = 0
            pairs = [(i, j) for i in range(len(don)) for j in range(i + 1, len(don))]
            for q, n in names:
                for t in range(n):
                    d = dict(code=m.csd_code, metal=m.metal_symbol, row=ROW.get(m.metal_symbol, "?"),
                             geom=m.coord_geom, cn=len(don), quantity=q, i=t, value=float(vals[k]), sigma_T=float(sig[k]),
                             n_neg_modes=n_neg, n_atoms=len(sy))
                    if q == "Q1":
                        d["donor"] = don[t] if t < len(don) else "?"
                    elif q == "Q2" and t < len(pairs):
                        d["donor"] = "-".join(sorted((don[pairs[t][0]], don[pairs[t][1]])))
                    elif q == "Q5":
                        a_, b_ = int(ge._q5_j[t]), int(ge._q5_k[t])
                        d["donor"] = "-".join(sorted((sy[a_], sy[b_])))
                    rows.append(d)
                    k += 1
            print(f"{m.csd_code}: {k} features ({int(np.isnan(sig).sum())} touch unmapped atoms), negative modes {n_neg}", flush=True)
        except Exception as exc:
            print(f"{m.csd_code}: skipped ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "features.csv", index=False)
    print(f"{len(df)} features from {df.code.nunique() if len(df) else 0} complexes -> {OUT / 'features.csv'}")


def cos_floor_sd(mu_cos, s_ang):
    """Thermal sd in cos units for an angle whose mean cos is mu_cos and whose thermal sd in angle is s_ang (rad):
    Var(cos t) for t ~ N(t0, s^2) to second order = sin^2 t0 s^2 + cos^2 t0 s^4 / 2 (non-zero at 0 / 180 deg)."""
    mu = np.clip(np.asarray(mu_cos, float), -1.0, 1.0)
    return np.sqrt((1.0 - mu ** 2) * s_ang ** 2 + 0.5 * mu ** 2 * s_ang ** 4)


def cmd_fit(a):
    """Median thermal sigma per coarse class (n >= MIN_N, else the quantity's median), and the share of table
    components in the DB narrower than the floor they would get."""
    import pandas as pd
    df = pd.read_csv(OUT / "features.csv")
    n_all = len(df)
    df = df[np.isfinite(df.sigma_T)]
    # complexes with small negative modes (loose tmQM optimisations) are kept: _pinv_hessian drops those modes, and
    # the medians with / without them agree within 2 % (2026-10-16 d)
    neg = df.groupby("code").n_neg_modes.first()
    df["base"] = df.quantity.str.replace(r"SO\d", "SO", regex=True)
    out = {"_doc": "thermal sd of each scorer feature at 298 K: sigma_T^2 = kT g^T H^+ g, H = GFN2 Hessian at the tmQM "
                   "minimum (refit/thermal_widths.py, CHANGELOG 2026-10-16 d). Units: Q1 A; Q2, SO (S_orient "
                   "angles) and Q5 in rad of the ANGLE (convert to cos at a component's mean with cos_floor_sd). "
                   "'quantity': median per quantity; 'by_class': median per 'quantity|metal row|donor' with n >= 20.",
           "units": {"Q1": "A", "Q2": "rad", "SO": "rad", "Q5": "rad"},
           "quantity": {}, "p10": {}, "p90": {}, "by_class": {}, "n_complexes": int(df.code.nunique())}
    lines = [f"features {n_all} ({n_all - len(df)} dropped: touch an unmapped atom / clamped linear angle; complexes "
             f"with negative modes: kept, {int((neg > 0).sum())}); {df.code.nunique()} complexes",
             f"complexes per row: {df.groupby('row').code.nunique().to_dict()}"]
    for q, g in df.groupby("base"):
        sc = 1.0 if q == "Q1" else 180 / np.pi
        u = "A" if q == "Q1" else "deg"
        out["quantity"][q] = float(g.sigma_T.median())
        out["p10"][q], out["p90"][q] = float(g.sigma_T.quantile(.1)), float(g.sigma_T.quantile(.9))
        lines.append(f"{q:3s} n {len(g):5d}  sigma_T median {g.sigma_T.median() * sc:.3f} {u}  p10 "
                     f"{g.sigma_T.quantile(.1) * sc:.3f}  p90 {g.sigma_T.quantile(.9) * sc:.3f}")
        for row, gg in g.groupby("row"):
            lines.append(f"      {row}: n {len(gg):5d}  median {gg.sigma_T.median() * sc:.3f}")
        if "donor" in g:
            for (row, don), gg in g.groupby(["row", "donor"]):
                if len(gg) >= 20:
                    out["by_class"][f"{q}|{row}|{don}"] = float(gg.sigma_T.median())
    # Q1 by donor element (kappa) / "eta" (haptic centroid): the one descriptor that moves sigma_T by more than the
    # plan's factor sqrt(2) (C 0.04 vs N 0.07 A); metal row and CN do not (report.txt)
    g1 = df[df.base == "Q1"].assign(el=lambda x: np.where(x.donor.str.startswith("eta"), "eta", x.donor))
    out["Q1_by_donor"] = {el: float(gg.sigma_T.median()) for el, gg in g1.groupby("el") if len(gg) >= 15}
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    FINAL.write_text(json.dumps(out, indent=1))
    # share of DB components narrower than the floor (quantity-level medians)
    db = json.loads(DB.read_text())
    fq = out["quantity"]
    for pre, q in (("Q1:", "Q1"), ("Q2:", "Q2"), ("QO:", "SO")):
        if q not in fq:
            continue
        sd, fl, w = [], [], []
        for key, ent in db.items():
            if not key.startswith(pre) or not isinstance(ent, dict):
                continue
            if pre == "QO:":
                for wt, mu, cov in zip(ent.get("weights", []), ent.get("means", []), ent.get("covs", [])):
                    d = np.sqrt(np.clip(np.diag(np.asarray(cov, float)), 0, None))
                    sd += list(d); fl += list(cos_floor_sd(mu, fq[q])); w += [wt] * len(d)
            else:
                for wt, mu, v in zip(ent.get("weights", []), ent.get("means", []), ent.get("vars", [])):
                    sd.append(np.sqrt(max(float(v), 0.0)))
                    fl.append(fq[q] if q == "Q1" else float(cos_floor_sd(mu, fq[q])))
                    w.append(wt)
        sd, fl, w = map(np.asarray, (sd, fl, w))
        nar = sd < fl
        lines.append(f"{q} DB components {len(sd)}: narrower than the thermal floor {nar.mean():.2f} "
                     f"(weight-averaged {np.average(nar, weights=w):.2f}); narrower than half of it "
                     f"{(sd < 0.5 * fl).mean():.2f}")
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt)
    print(txt)
    print(f"-> {FINAL}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sample", "hess", "sigma", "fit"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--per-metal", dest="per_metal", type=int, default=40)
    p.add_argument("--max-atoms", dest="max_atoms", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--codes", default="", help="comma-separated subset (sigma stage; testing)")
    a = p.parse_args()
    {"sample": cmd_sample, "hess": cmd_hess, "sigma": cmd_sigma, "fit": cmd_fit}[a.stage](a)


if __name__ == "__main__":
    main()
