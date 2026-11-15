"""export_lnqm.py -- LnQM (lnqm.h5) -> the tmQM file layout the refit reads (xyz/tmQM_LN.xyz, bo/tmQM_LN.BO, tmQM_y.csv).

    python refit/export_lnqm.py                                  # data/lnQM/lnqm.h5 -> data/refit/source/lnQM
    python refit/export_lnqm.py --h5 X.h5 --out DIR [--limit N] [--workers 12]

Per entry (h5 'data' arrays, ragged by 'slices'):
  * geometry: 'coord' (the final optimised geometry; = the last 'trajectory' frame to ~0.01 Bohr), Bohr -> Angstrom,
    so the refit runs with ln_xyz_in_bohr = False;
  * header: CSD_code = uid ("<32 hex>_<Ln>": one ligand set re-optimised with up to 15 lanthanides; the refit treats the
    siblings as one independent structure, jacobik.stats.mixture.structure_id), q = 'charge' (stored as uint64:
    read as int64), S = 'unpaired_e' (tmQM's S is the unpaired-electron count; the parser stores S + 1), Hill
    stoichiometry, MND = metal neighbours with bond order >= 0.08 (parser.BO_FLOOR);
  * .BO, written for both atoms of a pair and sorted by bond order as tmQM does (pairs below BO_WRITE dropped; the
    parser ignores < 0.08):
      - metal-ligand pairs: a contact is kept when GFN2 sees it (WBO >= 0.08) or when the DFT Mayer order sees it
        (>= 0.08) at bonding distance (d <= CLOSE_RCOV x covalent-radius sum); written with the Mayer order when it
        has one, else the GFN2 order. Full run 2, 2026-10-15 (1/4 of LnQM): contacts both methods see 29 315, median
        d / rcov 0.94; Mayer only 3 187, median 1.32, 56 % beyond 1.3 (artefacts: Ln...C 3.6-3.8 A at Mayer ~0.12
        while the O donor they hang on has Mayer 0); GFN2 only 1 182, median 0.98 (donors Mayer missed). Also dropped,
        1,3 artefacts: X covalently bonded (GFN2 >= 0.5) to an atom that binds the metal (either method) and
        sits > 0.3 A closer, with GFN2 M-X < WBO_13 (full refit 2026-10-15: 1 289 LnQM C "donors" at 3.2-4.2 A,
        mostly Ln-N/O-C carbons; 1,3 for 79 %, GFN2 < 0.08 for 78 %). Dropped pairs -> dropped_13_contacts.csv;
      - ligand-internal pairs: GFN2-xTB Wiberg bond orders on the same DFT geometry. LnQM's Mayer orders are not
        usable inside the ligands (diffuse-basis artefact; 2026-10-15): aromatic C-H at 1.09 A have Mayer BO -1.0
        to +0.38 or none, aromatic C-C at 1.40 A alternate 2.06 / 0.21. Without SMILES the mapping reads these raw
        orders (mapping._apply_fallback_path), so C-H bonds vanished (their H became Q6 "contacts": a mini refit
        put the C...H wall at 1.09 A) and rings were not perceived as aromatic. GFN2 (f-in-core for Ln, 0.1-0.2 s
        per entry): C-H 0.91-0.97, aromatic C-C 1.38-1.45. An entry whose xtb run fails is skipped (counted).
    The atom's third field is the sum of its written bond orders (unused by the refit; tmQM: bonded valence).
  * tmQM_y.csv: CSD_code, Electronic_E = 'energy' (Hartree), empty SMILES (the oxidation state comes from the header:
    oxidation.os_from_header, Ln(III) when S equals the Hund f^n count).
No coordinate is altered beyond the unit change; the contact cleaning (refit stage clean) runs on the export as on tmQM.
GFN2 bond orders are cached in <out>/gfn2_wbo.npz (re-exports reuse them).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOHR = 0.529177210903
BO_WRITE = 0.05
BO_FLOOR = 0.08
WBO_13 = 0.05
CLOSE_RCOV = 1.15
SPLIT = "LN"
XTB = os.environ.get("JACOBIK_XTB") or shutil.which("xtb") or str(HERE.parent / ".pixi/envs/default/Library/bin/xtb.exe")


def _hill(sym):
    c = Counter(sym)
    order = (["C", "H"] if "C" in c else []) + sorted(k for k in c if not ("C" in c and k in ("C", "H")))
    return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in order)


def _wbo_one(sym, X, q):
    """GFN2-xTB Wiberg bond orders [(i, j, wbo)] (0-based) of one geometry, or None. uhf 0, then 1 (odd electron
    counts; GFN2 keeps the Ln f shell in the core, so the DFT spin does not apply)."""
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "m.xyz"), "w") as fh:
            fh.write(f"{len(sym)}\n\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(sym, X)))
        for uhf in (0, 1):
            wf = os.path.join(td, "wbo")
            if os.path.exists(wf):
                os.remove(wf)
            r = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(q), "--uhf", str(uhf), "--sp"], cwd=td,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
            if r.returncode == 0 and os.path.exists(wf):
                W = np.loadtxt(wf, ndmin=2)
                return np.column_stack([W[:, 0] - 1, W[:, 1] - 1, W[:, 2]]) if len(W) else np.zeros((0, 3))
    return None


def _wbo_chunk(items):
    return [(i, _wbo_one(sym, X, q)) for i, sym, X, q in items]


def gfn2_wbo(entries, n_workers: int, cache: Path) -> dict:
    """{entry index: (n, 3) array or None} for entries [(i, sym, X, q)], cached in `cache` (npz)."""
    done = {}
    if cache.exists():
        z = np.load(cache)
        done = {int(k[1:]): z[k] for k in z.files if k.startswith("w")}
        done.update({int(k[1:]): None for k in z.files if k.startswith("f")})
    todo = [e for e in entries if e[0] not in done]
    if todo:
        from joblib import Parallel, delayed

        chunks = [todo[k:k + 100] for k in range(0, len(todo), 100)]
        n = 0
        for res in Parallel(n_jobs=n_workers, return_as="generator_unordered")(delayed(_wbo_chunk)(c) for c in chunks):
            for i, w in res:
                done[i] = w
            n += 1
            if n % 20 == 0 or n == len(chunks):
                print(f"  GFN2 bond orders: {n}/{len(chunks)} chunks", flush=True)
        arrs = {f"w{i}": w for i, w in done.items() if w is not None}
        arrs.update({f"f{i}": np.zeros(0) for i, w in done.items() if w is None})
        np.savez_compressed(cache, **arrs)
    return done


def export(h5: Path, out: Path, limit: int | None = None, n_workers: int | None = None) -> dict:
    import h5py
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    out = Path(out)
    (out / "xyz").mkdir(parents=True, exist_ok=True)
    (out / "bo").mkdir(parents=True, exist_ok=True)
    with h5py.File(h5, "r") as f:
        D, S = f["data"], f["slices"]
        n = len(S["uid"]) - 1 if limit is None else min(limit, len(S["uid"]) - 1)
        uid = [u.decode() for u in D["uid"][:n]]
        Z = D["numbers"][:, 0].astype(int)
        X = D["coord"][:] * BOHR
        bo = D["mayer_bo"][:]
        charge = D["charge"][:, 0].view(np.int64)
        unp = D["unpaired_e"][:, 0].astype(np.int64)
        energy = D["energy"][:, 0]
        sa, sb = S["coord"][:], S["mayer_bo"][:]
    sym_of = {z: pt.GetElementSymbol(int(z)) for z in np.unique(Z)}
    entries = [(i, [sym_of[z] for z in Z[sa[i]:sa[i + 1]]], X[sa[i]:sa[i + 1]], int(charge[i])) for i in range(n)]
    wbo = gfn2_wbo(entries, n_workers or max(1, (os.cpu_count() or 2)), out / "gfn2_wbo.npz")
    stats = Counter()
    dropped = []
    with open(out / "xyz" / f"tmQM_{SPLIT}.xyz", "w") as fx, open(out / "bo" / f"tmQM_{SPLIT}.BO", "w") as fb, \
            open(out / "tmQM_y.csv", "w", newline="") as fy:
        wy = csv.writer(fy, delimiter=";")
        wy.writerow(["CSD_code", "Electronic_E", "Dispersion_E", "Dipole_M", "Metal_q", "HL_Gap", "HOMO_Energy",
                     "LUMO_Energy", "Polarizability", "CSD_years", "SMILES"])
        for i, sym, Xi, q in entries:
            nat = len(sym)
            metals = [k for k, s in enumerate(sym) if 57 <= pt.GetAtomicNumber(s) <= 71]
            if len(metals) != 1:
                stats["skipped_metal_count"] += 1
                continue
            if wbo.get(i) is None:
                stats["skipped_xtb_failed"] += 1
                continue
            m = metals[0]
            nb = [[] for _ in range(nat)]

            def add(p, r, v):
                if v >= BO_WRITE and 0 <= p < nat and 0 <= r < nat and p != r:
                    nb[p].append((v, r))
                    nb[r].append((v, p))

            may, wm, cov = {}, {}, {}
            for p, r, v in bo[int(sb[i]):int(sb[i + 1])]:          # DFT Mayer: metal-ligand pairs only
                p, r = int(p), int(r)
                if m in (p, r):
                    may[r if p == m else p] = v
            for p, r, v in wbo[i]:                                  # GFN2 Wiberg: ligand-internal pairs
                p, r = int(p), int(r)
                if m in (p, r):
                    wm[r if p == m else p] = v
                else:
                    add(p, r, v)
                    if v >= 0.5:
                        cov.setdefault(p, []).append(r)
                        cov.setdefault(r, []).append(p)
            dM = np.linalg.norm(Xi - Xi[m], axis=1)
            rm = pt.GetRcovalent(pt.GetAtomicNumber(sym[m]))
            binds = lambda k: max(may.get(k, 0.0), wm.get(k, 0.0)) >= BO_FLOOR
            for j in sorted(set(may) | set(wm)):
                v, w = may.get(j, 0.0), wm.get(j, 0.0)
                close = dM[j] <= CLOSE_RCOV * (rm + pt.GetRcovalent(pt.GetAtomicNumber(sym[j])))
                # 1,3 artefact: j covalently bonded to an atom k that binds the metal (either method) and sits
                # > 0.3 A closer, and GFN2 sees no M-j bond (k2-carboxylate C / nitrate N at 2.8-2.9 A, Ln-O-C carbons)
                k13 = [k for k in cov.get(j, ()) if binds(k) and dM[k] < dM[j] - 0.3]
                why = ("13" if (k13 and w < WBO_13) else
                       None if (w >= BO_FLOOR or (v >= BO_FLOOR and close)) else "mayer_only_far" if v >= BO_WRITE else "weak")
                if why is not None:
                    if why != "weak":
                        dropped.append((uid[i], sym[m], j + 1, sym[j], round(float(dM[j]), 3), round(float(v), 3),
                                        round(float(w), 3), (k13[0] + 1) if k13 else "", sym[k13[0]] if k13 else "", why))
                    continue
                add(m, j, v if v >= BO_FLOOR else w)  # DFT Mayer order when it has one, else the GFN2 order
            mnd = sum(v >= BO_FLOOR for v, _ in nb[m])
            fx.write(f"{nat}\nCSD_code = {uid[i]} | q = {q} | S = {int(unp[i])} | "
                     f"Stoichiometry = {_hill(sym)} | MND = {mnd} | LnQM\n")
            fx.write("".join(f"{s:<2} {x:20.14f} {y:20.14f} {z:20.14f}\n" for s, (x, y, z) in zip(sym, Xi)))
            fx.write("\n")
            fb.write(f"CSD_code = {uid[i]} | LnQM\n")
            for k in range(nat):
                parts = "".join(f"    {sym[r]:<2} {r + 1:4d} {v:.3f}" for v, r in sorted(nb[k], reverse=True))
                fb.write(f"{k + 1:6d}  {sym[k]:<2} {sum(v for v, _ in nb[k]):6.3f}    {parts}\n")
            fb.write("\n")
            wy.writerow([uid[i], f"{energy[i]:.6f}", "", "", "", "", "", "", "", "", ""])
            stats["written"] += 1
    with open(out / "dropped_13_contacts.csv", "w", newline="") as fd:
        w = csv.writer(fd)
        w.writerow(["code", "metal", "x_idx", "x_el", "dist", "mayer_bo", "gfn2_wbo", "via_idx", "via_el", "reason"])
        w.writerows(dropped)
    stats["dropped_13_contacts"] = len(dropped)
    return dict(stats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=str(HERE.parent / "data" / "lnQM" / "lnqm.h5"))
    ap.add_argument("--out", default=str(HERE.parent / "data" / "refit" / "source" / "lnQM"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    st = export(Path(a.h5), Path(a.out), a.limit, a.workers)
    print(f"exported to {a.out}: {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
