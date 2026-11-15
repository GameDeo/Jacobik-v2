"""Unrelaxed ring census; reads A4b inputs, never the A4t relaxed geometries."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from tools.jgen import inputs
from tools.jgen.a4b.relax_run import to_ref_order
from tools.jgen.a4t.analyse import _load_parent
from tools.jgen.a4t.metrics import ring_share, shell_rmsd
from tools.jgen.bench.pool_export import _read_frames
from tools.jgen.bench.panels import TAIL
from tools.runkit import geom

MAIN = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
R_PARENTS = 'BABNAV CIXYUD DOHTOM DUHHIX EFOXII FEKKAJ HOHZAG UYOVUA ZAFPOP'.split()


def population():
    path = MAIN / 'research/bench/benchmarks/locked_exclusions.json'
    if not path.exists():
        # Colab receives the already-frozen eligible population, not auditor
        # custody files or an opportunity to select parents from results.
        frozen = Path(inputs.REPO) / 'research/campaigns/c2/C2_PANEL.json'
        panel = json.loads(frozen.read_text())
        frame_path = inputs.FRAME_V1 / 'frame.json'
        assert hashlib.sha256(frame_path.read_bytes()).hexdigest() == panel['frame_sha256']
        frame = json.loads(frame_path.read_text())
        cal = {g['representative_code'] for g in frame['main_allocation']['splits']['calibration']}
        codes = panel['codes']
        assert set(codes) <= cal | set(TAIL)
        assert not set(codes) & set(panel['excluded_requested'])
        return codes, {k: panel[k] for k in ['exclusions_sha256', 'excluded_requested']}
    exclusion = json.loads(path.read_text())
    locked = {c for v in exclusion.values() if isinstance(v, list) for c in v}
    frame = json.loads((inputs.FRAME_V1 / 'frame.json').read_text())
    cal = {g['representative_code'] for g in frame['main_allocation']['splits']['calibration']}
    codes = sorted((cal | set(TAIL)) - locked)
    assert not set(codes) & locked
    return codes, dict(exclusions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       excluded_requested=sorted((cal | set(TAIL)) & locked))


def ligand_rings(mol):
    """Use the same ligand graph and macrocycle detection as production FK.

    Chelate rings are a minimum cycle basis after adding authoritative M–D
    edges. Retain the ligand-only basis separately, since metal edges can change
    the basis. Ring atoms below use zero-based input/crystal indices.
    """
    from jacobik.topology.decompose import _build_ligand_graph, decompose_ligand

    records = []
    for li, lig in enumerate(mol.ligands):
        decompose_ligand(mol, lig)
        graph, orders = _build_ligand_graph(mol, lig)
        macros = [set(r) for r in lig.macrocycles]
        pivots = {frozenset((b.i, b.j)) for b in lig.rotatable_bonds}
        rings = nx.minimum_cycle_basis(graph)
        augmented = graph.copy()
        augmented.add_edges_from((mol.metal_idx, d) for d in lig.donor_atoms)
        rings += [r for r in nx.minimum_cycle_basis(augmented) if mol.metal_idx in r]
        for r in rings:
            nodes = set(r)
            edges = list(graph.subgraph(nodes).edges())
            bos = [orders.get(tuple(sorted(e)), 1.) for e in edges]
            # Production graph quantises aromatic orders into the 1.5 band.
            aromatic = bool(bos) and all(1.3 <= b <= 1.7 for b in bos)
            if mol.metal_idx in nodes:
                kind = 'chelate'
            elif any(nodes == m for m in macros):
                kind = 'macrocycle'
            elif 5 <= len(nodes) <= 8 and not aromatic:
                kind = 'internal_5_8'
            else:
                kind = 'aromatic_or_other'
            moving = [sorted(e) for e in edges if frozenset(e) in pivots]
            records.append(dict(ligand=li, atoms=sorted(int(a)-1 for a in nodes),
                                kind=kind, aromatic=aromatic,
                                endocyclic_fk_pivots=moving,
                                pucker_fixed=kind != 'macrocycle' and not moving))
    return records


def aligned_displacements(X, ref, symbols, perms):
    heavy = np.flatnonzero(np.asarray(symbols) != 'H')
    _, mapping = ring_share(X, ref, symbols, perms, set(), return_atom_map=True)
    A = X[heavy]; B = ref[[mapping[int(i)] for i in heavy]]
    A = A - A.mean(0); B = B - B.mean(0)
    u, _, vt = np.linalg.svd(A.T @ B)
    fix = np.eye(3); fix[-1, -1] = np.linalg.det(u @ vt)
    ss = ((A @ (u @ fix @ vt) - B)**2).sum(1)
    return {int(i): float(s) for i, s in zip(heavy, ss)}, mapping


def measure(code, root):
    mol, _, donors = _load_parent(code)
    rings = ligand_rings(mol)
    if code not in R_PARENTS:
        return dict(code=code, rings=rings)
    symbols = [a.symbol for a in mol.atoms]
    metal = int(mol.metal_idx)-1
    best = None
    sources = []
    for seed in (11, 12, 13):
        mp = root / 'xyz' / code / f'{code}_control_s{seed}_pool.json'
        manifest = json.loads(mp.read_text())
        ref = np.asarray(manifest['ref_X'], float)
        sources.append(dict(seed=seed, sha256=hashlib.sha256(mp.read_bytes()).hexdigest()))
        frames = list(_read_frames(mp.parent / manifest['rows'][0]['file']))
        for i, (_, _, X) in enumerate(frames):
            X = to_ref_order(X, manifest['gen_to_ref_idx'])
            shell = shell_rmsd(X, ref, symbols, metal, donors)
            if best is None or (shell, seed, i) < best[0]:
                best = ((shell, seed, i), X, ref)
    (shell, seed, index), X, ref = best
    perms, capped = geom.automorphisms(ref, symbols, metal)
    ss, mapping = aligned_displacements(X, ref, symbols, perms)
    total = sum(ss.values())
    for ring in rings:
        ring['share'] = sum(ss.get(a, 0.) for a in ring['atoms']) / total if total else 0.
    types = {}
    for kind in sorted({r['kind'] for r in rings}):
        atoms = {a for r in rings if r['kind'] == kind for a in r['atoms']}
        types[kind] = sum(ss.get(a, 0.) for a in atoms) / total if total else 0.
    return dict(code=code, rings=rings, type_shares=types, best_seed=seed,
                best_row=index, shell_rmsd=shell, symmetry_capped=capped,
                atom_map=mapping, total_heavy_ss=total, sources=sources)


def verify_fk(code):
    """Check actual production embed/FK without taking an IK step."""
    from tools.jgen.bench.run import _setup_env
    _setup_env({})
    from jacobik.builder import engine
    from tools.jdev import multiseed
    from tools.jdev.budget.common import stats_from_db
    from tools.jgen.embedding_validity import EmbeddingValidity

    _, builder, _ = _load_parent(code)
    validity = EmbeddingValidity(engine).install()
    validity.parent = code
    try:
        ctx, assignments, W, T = multiseed.prepare(
            builder, 11, device='cpu', angle_bends=True,
            stats=stats_from_db(inputs.DB_PATH))
        validity.check(code)
        fks, _, _, _, _, _, _, cand = builder._beam_search_all_candidates(ctx, assignments, W)
        return dict(rings=ligand_rings(ctx.mol),
                    fk=[dict(base=f.n_joints_base, bend=f.n_bend_joints,
                             macro=f.n_macro_joints, angle=f.n_angle_bends) for f in fks],
                    W=W, T=T, B=len(cand), embed_valid=True)
    finally:
        validity.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--verify-fk', action='store_true')
    a = ap.parse_args()
    codes, meta = population()
    if a.only:
        requested = a.only.split(',')
        assert set(requested) <= set(codes), 'ineligible parent requested'
        codes = requested
    a.out.mkdir(parents=True, exist_ok=True)
    for code in codes:
        start = time.monotonic()
        try:
            result = verify_fk(code) if a.verify_fk else measure(code, MAIN / 'research/data/jgen/campaign/A4b')
            data = dict(code=code, status='ok', **{k: v for k, v in result.items() if k != 'code'})
        except Exception as exc:
            data = dict(code=code, status='failed', error=repr(exc))
        data.update(wall_s=time.monotonic()-start, population=meta)
        (a.out / f'{code}.json').write_text(json.dumps(data, indent=2))
        print(code, data['status'], data.get('type_shares', {}), round(data['wall_s'], 2), flush=True)


if __name__ == '__main__':
    main()
