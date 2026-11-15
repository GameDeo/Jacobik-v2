"""Architector post-processing (root jacobik env): common relaxation, reference geometry metrics, tables. No MACE here.

Per returned structure k of a job, four geometries exist on disk: arch_init_k.xyz (Architector's unrelaxed assembly),
arch_final_k.xyz (Architector's own xTB-relaxed output), and this stage writes relax_init_k.xyz / relax_final_k.xyz, the
jgen common relaxation (tools.runkit.xtb.opt, crude, 200 cycles, 1 thread; the protocol of tools.jgen.a4t.relax_run).
MACE energies are not computed here (Amendment 1): mace_pass.py fills those columns in a later GPU pass.

Per job the stage writes post.json and skips existing ones, so it can run alongside the Architector jobs (resumable).
Tables are rebuilt from all post.json files by `--tables-only` or at the end of a run.

Geometry vs reference (only where a reference exists: DOE fixtures / a case with a reference record): atom correspondence by
an element-labelled covalent graph isomorphism (networkx VF2, bond cutoff 1.2 x sum of covalent radii, then 1.4), then
symmetric heavy-atom RMSD (tools.runkit.geom.sym_rmsd_to_ref), shell_rmsd (tools.jgen.a4t.metrics), metal-donor mean.
A structure with no isomorphism is recorded as map_ok=false (kept, without reference metrics).

Usage: python -m tools.architector.post --run-dir <run dir> [--job-glob 'doe_aqua/*/*/seed_*'] [--tables-only]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from tools.runkit import geom, xtb as rk_xtb
from tools.jgen.a4t.metrics import shell_rmsd, _kabsch_rmsd


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    syms, X = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0])
        X.append([float(v) for v in p[1:4]])
    return syms, np.asarray(X, float)


def write_xyz(path, symbols, X, comment):
    lines = [str(len(symbols)), comment]
    lines += [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, X)]
    Path(path).write_text("\n".join(lines) + "\n")


def covalent_graph(symbols, X, factor):
    import networkx as nx
    G = nx.Graph()
    for i, s in enumerate(symbols):
        G.add_node(i, el=s)
    r = np.array([geom.rcov(str(s)) for s in symbols])
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < factor * (r[i] + r[j]):
                G.add_edge(i, j)
    return G


def map_to_reference(sym_gen, X_gen, sym_ref, X_ref):
    """perm[r] = g (generated index for each reference index), and the bond factor used; (None, None) if none found."""
    from networkx.algorithms import isomorphism as iso
    if sorted(sym_gen) != sorted(sym_ref):
        return None, None
    for factor in (1.2, 1.4):
        Gg = covalent_graph(sym_gen, X_gen, factor)
        Gr = covalent_graph(sym_ref, X_ref, factor)
        gm = iso.GraphMatcher(Gg, Gr, node_match=lambda a, b: a["el"] == b["el"])
        if gm.is_isomorphic():
            perm = np.empty(len(sym_ref), int)
            for g, r in gm.mapping.items():
                perm[r] = g
            return perm, factor
    return None, None


def map_ligand_only(sym_g, X_g, sym_r, X_r, mg, mr):
    """Mapping of all non-metal atoms with metal edges removed: perm[r] = g (metal mapped to metal), or (None, None)."""
    from networkx.algorithms import isomorphism as iso
    keep_g = [i for i in range(len(sym_g)) if i != mg]
    keep_r = [i for i in range(len(sym_r)) if i != mr]
    if sorted(sym_g[i] for i in keep_g) != sorted(sym_r[i] for i in keep_r):
        return None, None
    for factor in (1.2, 1.4):
        Gg = covalent_graph([sym_g[i] for i in keep_g], X_g[keep_g], factor)
        Gr = covalent_graph([sym_r[i] for i in keep_r], X_r[keep_r], factor)
        gm = iso.GraphMatcher(Gg, Gr, node_match=lambda a, b: a["el"] == b["el"])
        if gm.is_isomorphic():
            perm = np.full(len(sym_r), -1, int)
            perm[mr] = mg
            for lr, lg in gm.mapping.items():
                perm[keep_r[lr]] = keep_g[lg]
            return perm, factor
    return None, None


BOND_GATE = 1.21   # jgen bond gate: M-D <= 1.21 x (rcov M + rcov D)
CLASH_FRAC = 0.75  # non-bonded pair closer than 0.75 x (rcov i + rcov j) is a clash
CLOSE_FRAC = 1.2   # non-bonded pair closer than 1.2 x (rcov i + rcov j) is in the covalent-graph range (close contact)
MATCH_FACTOR = 1.4  # covalent cutoff for matching ligand graphs (lenient: extra edges are allowed in a monomorphism)
MAX_MONO = 300      # cap on candidate ligand mappings per ligand instance


def _gate(sym, X, m, i):
    return np.linalg.norm(X[i] - X[m]) <= BOND_GATE * (geom.rcov(str(sym[m])) + geom.rcov(str(sym[i])))


def declared_donor_map(sym, X, metal_idx, ligands):
    """Map each declared ligand instance (spec SMILES graph, coordList donors) into the output atom order.

    Each instance is a monomorphism of its SMILES graph into the heavy-atom covalent graph of the not-yet-used output
    atoms (extra contacts are allowed, so a clashing assembly still maps). Among up to MAX_MONO candidates the mapping
    with the most declared donors inside the bond gate is kept, so automorphic donor choices are resolved toward the
    metal. Returns a list (one per instance) of dict(map=template->output or None, donors_out=[...] or None)."""
    from networkx.algorithms import isomorphism as iso
    from rdkit import Chem
    import networkx as nx
    heavy = [i for i, s_ in enumerate(sym) if s_ != "H" and i != metal_idx]
    G = nx.Graph()
    for i in heavy:
        G.add_node(i, el=sym[i])
    r = {i: geom.rcov(str(sym[i])) for i in heavy}
    for a in range(len(heavy)):
        for b in range(a + 1, len(heavy)):
            i, j = heavy[a], heavy[b]
            if np.linalg.norm(X[i] - X[j]) < MATCH_FACTOR * (r[i] + r[j]):
                G.add_edge(i, j)
    used, out = set(), []
    for li in ligands:
        mol = Chem.MolFromSmiles(li["smiles"])
        if mol is None:
            out.append(dict(map=None, donors_out=None, reason="smiles_parse_failed"))
            continue
        T = nx.Graph()
        for a in mol.GetAtoms():
            T.add_node(a.GetIdx(), el=a.GetSymbol())
        for b in mol.GetBonds():
            T.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        free = [n for n in heavy if n not in used]
        Gf = G.subgraph(free).copy()
        best, best_score = None, -1
        gm = iso.GraphMatcher(Gf, T, node_match=lambda a_, b_: a_["el"] == b_["el"])
        for k, m_ in enumerate(gm.subgraph_monomorphisms_iter()):
            inv = {t: o for o, t in m_.items()}  # template -> output
            donors = [inv[d] for d in li["coordList"]]
            score = sum(_gate(sym, X, metal_idx, d) for d in donors)
            if score > best_score:
                best, best_score = inv, score
                if score == len(li["coordList"]):
                    break
            if k + 1 >= MAX_MONO:
                break
        if best is None:
            out.append(dict(map=None, donors_out=None, reason="no_monomorphism"))
            continue
        used.update(best.values())
        out.append(dict(map=best, donors_out=[best[d] for d in li["coordList"]], reason=None))
    return out


def geometry_metrics(row, sym, X, ref, metal_sym, ligands, init_only):
    """Fills one structure-geometry row.

    Declared donors (spec) are mapped into the output order (declared_donor_map). n_donors_bound counts declared donors
    inside the bond gate; n_extra_contacts counts other non-H, non-metal atoms inside the gate; clash fields use the
    non-bonded pairs of the input graph (mapped ligands and metal). Reference metrics:
    map_mode = full        : all atoms, metal edges (sym_heavy_rmsd_A; shell_rmsd_A if all declared donors bound)
    map_mode = ligand_only : all non-metal atoms, no metal edges (ligand_kabsch_rmsd_A; shell if all bound)
    map_mode = none        : neither mapping; no reference metrics (row kept).
    """
    expected = sum(len(li["coordList"]) for li in ligands)
    row["n_donors_expected"] = int(expected)
    row["arch_init_only"] = bool(init_only)
    metal_idx = next((i for i, s_ in enumerate(sym) if s_ == metal_sym), None)
    if metal_idx is None:
        row["map_mode"] = "none"
        return
    maps = declared_donor_map(sym, X, metal_idx, ligands)
    donors_out = [d for m_ in maps if m_["donors_out"] for d in m_["donors_out"]]
    bound = [d for d in donors_out if _gate(sym, X, metal_idx, d)]
    row["n_donors_mapped"] = len(donors_out)
    row["n_donors_unmapped"] = expected - len(donors_out)
    row["n_donors_bound"] = len(bound)
    row["dissociated"] = len(bound) < expected
    row["md_mean_A"] = (float(np.mean([np.linalg.norm(X[metal_idx] - X[d]) for d in bound])) if bound else None)
    declared = set(donors_out)
    row["n_extra_contacts"] = sum(1 for i, s_ in enumerate(sym)
                                  if i != metal_idx and s_ != "H" and i not in declared and _gate(sym, X, metal_idx, i))
    # clash fields: non-bonded pairs of the input graph among mapped heavy atoms and the metal
    bonded = set()
    atoms = {metal_idx}
    from rdkit import Chem
    for li, m_ in zip(ligands, maps):
        if not m_["map"]:
            continue
        mol = Chem.MolFromSmiles(li["smiles"])
        atoms.update(m_["map"].values())
        for b in mol.GetBonds():
            bonded.add(frozenset((m_["map"][b.GetBeginAtomIdx()], m_["map"][b.GetEndAtomIdx()])))
    for d in donors_out:
        bonded.add(frozenset((metal_idx, d)))
    atoms = sorted(atoms)
    dmin, nclash, nclose = None, 0, 0
    for a_i in range(len(atoms)):
        for b_i in range(a_i + 1, len(atoms)):
            i, j = atoms[a_i], atoms[b_i]
            if frozenset((i, j)) in bonded:
                continue
            d = float(np.linalg.norm(X[i] - X[j]))
            dmin = d if dmin is None else min(dmin, d)
            rs = geom.rcov(str(sym[i])) + geom.rcov(str(sym[j]))
            if d < CLASH_FRAC * rs:
                nclash += 1
            if d < CLOSE_FRAC * rs:
                nclose += 1
    row["min_nonbonded_dist_A"] = dmin
    row["n_clash_pairs"] = nclash          # non-bonded pairs < 0.75 x (rcov i + rcov j)
    row["n_close_pairs"] = nclose          # non-bonded pairs < 1.2 x (rcov i + rcov j): the graph-edge range
    if ref is None:
        row["map_mode"] = "no_reference"
        return
    if ref.get("heavy_only"):  # the reference omits hydrogens: compare heavy atoms only
        keep = [i for i, s_ in enumerate(sym) if s_ != "H"]
        sym_m, X_m = [sym[i] for i in keep], X[keep]
        remap = {old: new for new, old in enumerate(keep)}
        metal_m = remap[metal_idx]
        bound_m_all = len(bound) == expected
    else:
        sym_m, X_m, metal_m = sym, X, metal_idx
        bound_m_all = len(bound) == expected
    row["heavy_only_reference"] = bool(ref.get("heavy_only"))
    perm, factor = map_to_reference(sym_m, X_m, ref["symbols"], ref["X"])
    if perm is not None:
        row["map_mode"] = "full"
        row["map_bond_factor"] = factor
        row.update(metrics_vs_reference(X_m[perm], ref))
        row["shell_rmsd_A"] = row["shell_rmsd_A"] if bound_m_all else None
        return
    perm, factor = map_ligand_only(sym_m, X_m, ref["symbols"], ref["X"], metal_m, ref["metal"])
    if perm is None:
        row["map_mode"] = "none"
        return
    row["map_mode"] = "ligand_only"
    row["map_bond_factor"] = factor
    X_ord = np.full((len(ref["symbols"]), 3), np.nan)
    for r_, g in enumerate(perm):
        X_ord[r_] = X_m[g]
    lig = [i for i, s_ in enumerate(ref["symbols"]) if s_ != "H" and i != ref["metal"]]
    row["ligand_kabsch_rmsd_A"] = float(_kabsch_rmsd(X_ord[lig], ref["X"][lig]))
    row["shell_rmsd_A"] = (float(shell_rmsd(X_ord, ref["X"], ref["symbols"], ref["metal"], ref["donors"]))
                           if bound_m_all else None)


def metrics_vs_reference(X_ref_order, ref):
    m, donors = ref["metal"], ref["donors"]
    heavy = np.array([s != "H" for s in ref["symbols"]])
    sym = float(geom.sym_rmsd_to_ref(X_ref_order[heavy][None], ref["X"][heavy], ref["P"])[0])
    sh = float(shell_rmsd(X_ref_order, ref["X"], ref["symbols"], m, donors))
    return dict(sym_heavy_rmsd_A=sym, shell_rmsd_A=sh)


def common_relax(symbols, X, q, mult, workroot):
    Path(workroot).mkdir(parents=True, exist_ok=True)
    r = rk_xtb.opt(list(symbols), np.asarray(X, float), int(q), int(mult), str(workroot), cycles=200, level="crude")
    return dict(relax_status=r.get("status"), relax_converged=bool(r.get("converged")), relax_cycles=r.get("cycles"),
                relax_E_h=r.get("E_h"), relax_wall_s=r.get("wall_s"), xtb_version=r.get("xtb_version"),
                xtb_sha256=r.get("xtb_sha256"),
                X=None if r.get("X") is None else np.asarray(r["X"], float))


BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def catalog_reference(panel, case):
    """Baseline (A01-A08) and challenge experimental heavy-atom references, from the curated catalog files.

    Returns the reference dict (heavy_only=True: hydrogens are omitted in the source) or None if no reference exists.
    Donors come from the reference record's ligand_instances (donor_reference_indices); charge/spin from cases.json."""
    import ast
    ref_dir = BENCH / "data" / ("challenge/references" if panel == "challenge" else "references")
    xyz, js = ref_dir / f"{case}.xyz", ref_dir / f"{case}.json"
    if not (xyz.exists() and js.exists()):
        return None
    rec = json.loads(js.read_text())
    cases = json.loads((BENCH / "data" / ("challenge/cases.json" if panel == "challenge" else "cases.json")).read_text())["cases"]
    c = next(x for x in cases if x["id"] == case)
    symbols, X = read_xyz(xyz)
    metal = int(rec["metal_reference_index"])
    inst = rec["ligand_instances"]
    inst = ast.literal_eval(inst) if isinstance(inst, str) else inst
    donors = sorted({int(d) for li in inst for d in li["donor_reference_indices"]})
    return dict(symbols=symbols, X=X, metal=metal, donors=donors, charge=int(c["charge"]),
                mult=int(c["unpaired_electrons"]) + 1, heavy_only=True, case=case, panel=panel,
                source=str(js.relative_to(BENCH)))


def reference_for(run, panel, case, cache, scratch):
    """Reference geometry for a case from its exported record (DOE fixtures). None when the case has no reference record."""
    key = (panel, case)
    if key in cache:
        return cache[key]
    p = run / "inputs" / panel / f"{case}.json"
    if p.exists():
        rec = json.loads(p.read_text())
        r = rec["reference"]
        base = dict(symbols=list(r["symbols"]), X=np.asarray(r["X"], float), metal=int(r["metal_index"]),
                    donors=[int(d) for d in r["donor_indices"]], charge=int(rec["charge"]), mult=int(rec["mult"]),
                    heavy_only=False)
    else:
        base = catalog_reference(panel, case)
    if base is None:
        cache[key] = None
        return None
    symbols, X = base["symbols"], base["X"]
    metal, donors = base["metal"], base["donors"]
    P, capped = geom.automorphisms(X, symbols, metal)
    ref = dict(base, P=P, capped=bool(capped), case=case, panel=panel)
    if base["heavy_only"]:
        # A hydrogen-less reference cannot be relaxed meaningfully with xTB: no reference relaxation is run.
        ref["relax"] = dict(relax_status="not_applicable_heavy_atom_reference")
    else:
        rel = common_relax(symbols, X, ref["charge"], ref["mult"], scratch / "xtb_ref")
        ref["relax"] = {k: v for k, v in rel.items() if k != "X"}
        if rel["X"] is not None:
            ref["X_relaxed"] = rel["X"]
    cache[key] = ref
    return ref


def process_job(run, jdir, job, scratch_root, ref_cache):
    res = json.loads((jdir / "result.json").read_text())
    inp = json.loads((jdir / "input.json").read_text())
    q = int(inp["parameters"]["full_charge"])
    mult = int(inp["parameters"]["full_spin"]) + 1
    ref = reference_for(run, job["panel"], job["case"], ref_cache, scratch_root / "refs")
    rows = []
    for st in res.get("structures", []):
        k = st["rank"]
        s = st.get("scalars", {})
        arch = {}
        for name in ("arch_init", "arch_final"):
            fn = st.get("files", {}).get(name)
            if fn:
                arch[name] = read_xyz(jdir / fn)
        for name in ("arch_init", "arch_final"):
            if name not in arch:
                continue
            sym, X = arch[name]
            rel = common_relax(sym, X, q, mult, scratch_root / "xtb" / job["panel"] / job["case"] / f"s{job['seed']}")
            tag = "relax_" + name.split("_")[1]
            if rel["X"] is not None:
                write_xyz(jdir / f"{tag}_{k:03d}.xyz", sym, rel["X"], f"{tag} from {name} rank {k}")
            arch[tag] = (sym, rel["X"])
            rows.append(dict(job_row(job), rank=k, upstream_key=st.get("upstream_key"), geometry=tag,
                             n_atoms=len(sym), charge=q, mult=mult, source_geometry=name,
                             xtb_common_status=rel["relax_status"], converged=rel["relax_converged"],
                             cycles=rel["relax_cycles"], xtb_E_h=rel["relax_E_h"], relax_wall_s=rel["relax_wall_s"],
                             xtb_version=rel["xtb_version"], xtb_sha256=rel["xtb_sha256"]))
        # Architector's own output rows (no common relaxation of its own)
        for name in ("arch_init", "arch_final"):
            if name not in arch:
                continue
            sym, X = arch[name]
            rows.append(dict(job_row(job), rank=k, upstream_key=st.get("upstream_key"), geometry=name,
                             n_atoms=len(sym), charge=q, mult=mult, source_geometry=name,
                             arch_energy_ev=s.get("energy") if name == "arch_final" else None,
                             arch_init_energy_ev=s.get("init_energy"),
                             arch_total_charge=s.get("total_charge"),
                             arch_unpaired_xtb=s.get("xtb_n_unpaired_electrons")))
        for name in ("arch_init", "arch_final", "relax_init", "relax_final"):
            if name not in arch or arch[name][1] is None:
                continue
            sym_g, X = arch[name][0], arch[name][1]
            row = next(r for r in rows if r.get("rank") == k and r["geometry"] == name)
            geometry_metrics(row, sym_g, X, ref, inp["core"]["metal"], inp["ligands"],
                             str(st.get("upstream_key", "")).endswith("_init_only"))
            row["mace_kjmol"] = None  # filled by mace_pass.py
    return dict(structures=len(res.get("structures", [])), opt_steps=res.get("opt_steps", []), rows=rows,
                reference=(None if ref is None else dict(ref.get("relax", {}), n_atoms=len(ref["symbols"]),
                                                       automorphisms_capped=ref["capped"])))


def job_row(job):
    return dict(panel=job["panel"], case=job["case"], mode=job["mode"], seed=job["seed"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--job-glob", default="*/*/*/seed_*")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    os.environ.update({v: "1" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    scratch_root = run / "scratch" / "post"
    scratch_root.mkdir(parents=True, exist_ok=True)
    os.environ["RUNKIT_SELFCHECK_DIR"] = str(scratch_root)
    os.environ["TMPDIR"] = str(scratch_root)
    if not a.tables_only:
        rk_xtb.require()
        ref_cache = {}
        for jf in sorted((run / "structures").glob(a.job_glob + "/job.json")):
            job = json.loads(jf.read_text())
            if job["status"] != "returned_structures":
                continue
            jdir = jf.parent
            out = jdir / "post.json"
            if out.exists():
                continue
            try:
                body = process_job(run, jdir, job, scratch_root, ref_cache)
            except Exception as exc:  # record and continue; never fail the batch
                body = dict(error=f"{type(exc).__name__}: {exc}", rows=[])
            out.write_text(json.dumps(body, indent=1, allow_nan=False, default=str) + "\n")
            print("post", job["panel"], job["case"], job["mode"], job["seed"], "rows", len(body["rows"]), flush=True)
    write_tables(run)
    return 0


def write_tables(run):
    rows, jobs_post, refs = [], [], {}
    for pj in sorted((run / "structures").glob("*/*/*/seed_*/post.json")):
        body = json.loads(pj.read_text())
        mj = pj.parent / "mace.json"  # written by mace_pass.py; keys "<geometry>_<rank:03d>", kJ/mol
        mace = json.loads(mj.read_text()) if mj.exists() else {}
        for r in body.get("rows", []):
            if r.get("geometry") and r.get("rank") is not None:
                r["mace_kjmol"] = mace.get(f"{r['geometry']}_{int(r['rank']):03d}")
            if "error" in mace:
                r["mace_error"] = mace["error"]
        rows += body.get("rows", [])
        if body.get("reference"):
            refs[str(pj.parent.relative_to(run / "structures"))] = body["reference"]
    tab = run / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with (tab / "structures_post.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    jobs = [json.loads(jf.read_text()) for jf in sorted((run / "structures").glob("*/*/*/seed_*/job.json"))]
    if jobs:
        keys = ["panel", "case", "mode", "seed", "n_conformers", "status", "n_structures", "sentinel_energy_count",
                "wall_s", "cpu_s", "peak_rss_mb", "exit_code", "timed_out", "timeout_s", "error", "directory", "log",
                "finished_utc"]
        with (tab / "jobs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(jobs)
    (tab / "references_post.json").write_text(json.dumps(refs, indent=1, allow_nan=False, default=str) + "\n")
    print("tables rows", len(rows), "references", len(refs))


if __name__ == "__main__":
    raise SystemExit(main())
