"""Phase 0 initial-placement, aromatic attachment and tiny-pool measurements.

Instrumentation lives outside production. Each parent has an independent
failure record. No relaxation or MACE evaluation is performed here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import time
from pathlib import Path

import numpy as np

from .rings import MAIN, population, ligand_rings

TINY = 'QAPFAS UYOVUA MIGTEE MIVXAR ZAZREZ SIVBUV HOXJUZ'.split()
AROMATIC = 'BABNAV CIXYUD DUHHIX EFOXII ZAFPOP'.split()
QUANTILES = [0, .25, .5, .75, .9, .95, 1]


def quantiles(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(zip(map(str, QUANTILES), map(float, np.quantile(values, QUANTILES)))) if len(values) else {}


def prepare_parent(code, device, flags=None, ez_variant=None):
    from tools.jgen.bench.run import _setup_env
    _setup_env(flags or {})
    import torch
    torch.set_num_threads(1)
    from tools.jgen.a4t.analyse import _load_parent
    from tools.jdev import multiseed
    from tools.jdev.budget.common import stats_from_db
    from tools.jgen import inputs
    from tools.jgen.embedding_validity import EmbeddingValidity
    from jacobik.builder import engine
    original, builder, _ = _load_parent(code)
    if ez_variant is not None:
        fixes=builder._ez_combinations()[ez_variant]
        builder._ez_inner=True
        for lig,fix in zip(builder.ligands,fixes):lig.ez_fix=fix
    validity = EmbeddingValidity(engine).install()
    validity.parent = code
    try:
        ctx, assignments, W, T = multiseed.prepare(
            builder, 11, device=device, angle_bends=True, stats=stats_from_db(inputs.DB_PATH))
        validity.check(code)
    finally:
        validity.close()
    return original, builder, ctx, assignments, W, T


def seeded_rows(builder, ctx, assignments, W, seed):
    """Capture actual Kabsch inputs and RNG, leaving production outputs intact."""
    import torch
    from jacobik.ik import ik_solver
    real = ik_solver.kabsch_init
    captures = []
    def capture(*args, **kwargs):
        state = torch.get_rng_state()
        result = real(*args, **kwargs)
        captures.append((kwargs, state))
        return result
    ctx.np_rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    ik_solver.kabsch_init = capture
    try:
        rows = builder._beam_search_all_candidates(ctx, assignments, W)
    finally:
        ik_solver.kabsch_init = real
    return rows, captures


def placement(rows, captures, ctx):
    import torch
    from jacobik.ik.ik_solver import kabsch_init
    from jacobik.ik.fk_compose import ComplexFK
    fks, theta, dpi, ge, _, X, _, _ = rows
    adjusted = []
    end_rng = torch.get_rng_state()
    try:
        for fk, th, (kw, rng) in zip(fks, theta, captures):
            seed = th.clone(); seed[:, :6] = 0
            with torch.no_grad():
                own = fk.apply(seed, X)
                torch.set_rng_state(rng)
                base = kabsch_init(**dict(kw, X_ref=own))
                adjusted.append(torch.cat([base, th[:, 6:]], -1))
    finally:
        torch.set_rng_state(end_rng)
    with torch.no_grad():
        old = ComplexFK(fks).apply(theta, X)
        new = ComplexFK(fks).apply(adjusted, X)
    results = []
    for label, coords in [('production', old), ('per_row_fit', new)]:
        pos = ge.donor_positions(coords)
        metal = coords[:, ge.metal_pos_idx]
        per_ligand = []
        all_offsets = []
        for li, (indices, (kw, _)) in enumerate(zip(dpi, captures)):
            Q = kw['donor_targets']
            P = pos[:, indices]
            offsets = (P-Q).norm(dim=-1).cpu().numpy()
            md = (P-metal[:, None]).norm(dim=-1).cpu().numpy()
            target_md = (Q-metal[:, None]).norm(dim=-1).cpu().numpy()
            ligand = ctx.mol.ligands[li]
            n = len(indices)
            kind = ('macrocycle' if fks[li].n_macro_joints else 'haptic' if ligand.haptic_groups
                    else 'mono' if n == 1 else 'bi' if n == 2 else 'poly')
            per_ligand.append(dict(ligand=li, kind=kind, offsets=offsets.tolist(),
                                   row_max=offsets.max(1).tolist(), md=md.tolist(),
                                   md_error=np.abs(md-target_md).tolist()))
            all_offsets.append(offsets)
        maxima = np.concatenate(all_offsets, axis=1).max(1)
        results.append(dict(kind=label, row_max=maxima.tolist(), quantiles=quantiles(maxima), ligands=per_ligand))
    return results


def large_rings(mol):
    from jacobik.topology.decompose import (_build_ligand_graph, _MACRO_MIN_SIZE,
        _MACRO_MIN_DONORS, _BO_SINGLE, _BO_SINGLE_TOL)
    import networkx as nx
    ligand_rings(mol)  # populate macrocycle decomposition
    out = []
    for li, lig in enumerate(mol.ligands):
        graph, orders = _build_ligand_graph(mol, lig)
        for ring in nx.minimum_cycle_basis(graph):
            if len(ring) < 9:
                continue
            nodes = set(ring); sub = graph.subgraph(nodes)
            bos = [orders.get(tuple(sorted(e)), 1.) for e in sub.edges()]
            aromatic = all(1.3 <= b <= 1.7 for b in bos)
            if aromatic:
                continue
            all_single = all(abs(b-_BO_SINGLE) <= _BO_SINGLE_TOL for b in bos)
            donors = len(nodes & set(lig.donor_atoms))
            on_path = any(nodes == set(r) for r in lig.macrocycles)
            reasons = []
            if len(ring) < _MACRO_MIN_SIZE: reasons.append('size')
            if not all_single: reasons.append('not_all_single')
            if donors < _MACRO_MIN_DONORS: reasons.append('fewer_than_2_donors')
            if not on_path and not reasons: reasons.append('other_cycle_path_condition')
            out.append(dict(ligand=li, atoms=sorted(nodes), size=len(ring), all_single=all_single,
                            donors=donors, on_macro_path=on_path, failed_conditions=reasons))
    return out


def aromatic_attachments(code, original, ctx, rows, pool_bundles=None):
    """Angles use production FK quads mapped to the stored reference order."""
    import torch
    from jacobik.ik.kernels import dihedral_axis_from_quad
    from jacobik.ik.fk_compose import ComplexFK
    from tools.jgen.bench.pool_export import _read_frames
    root = MAIN / 'research/data/jgen/campaign'
    census = json.loads((root / 'C2/rings' / f'{code}.json').read_text())
    seed = census['best_seed']
    mp = root / 'A4b/xyz' / code / f'{code}_control_s{seed}_pool.json'
    manifest = json.loads(mp.read_text())
    frames = list(_read_frames(mp.parent / manifest['rows'][0]['file']))
    ref = np.asarray(manifest['ref_X'], float)
    row = frames[census['best_row']][2][manifest['gen_to_ref_idx']]
    # A4b's manifest mapping is shared by the canonical builder atom order
    # across E/Z variants. Do not choose a fresh symmetry mapping from the
    # free embed, which can exchange equivalent rings relative to pool rows.
    g2r=np.asarray(manifest['gen_to_ref_idx'],int)
    assert [ctx.mol.atoms[i].symbol for i in g2r]==manifest['ref_symbols']
    gen_to_ref = np.argsort(g2r)
    fks, theta = rows[:2]
    conformers=[]
    for variant_ctx,variant_rows in (pool_bundles or [(ctx,rows)]):
        vf,vt,_,_,_,vX,_,_=variant_rows
        assert [variant_ctx.mol.atoms[i].symbol for i in g2r]==manifest['ref_symbols']
        with torch.no_grad():conformers.append(ComplexFK(vf).apply(vt,vX)[:,g2r].cpu())
    seeded=torch.cat(conformers,0)
    out = []
    def angle(coords, quad):
        return float(dihedral_axis_from_quad(torch.tensor(coords[quad][None], dtype=torch.float32))[0].item())
    def wrap(x): return np.arctan2(np.sin(x), np.cos(x))
    for ring in census['rings']:
        if not ring['aromatic'] or ring['share'] <= .2:
            continue
        atoms = set(ring['atoms']); attachments = []
        for li, fk in enumerate(fks):
            for j in range(fk.n_joints_base):
                a, b = int(gen_to_ref[fk._pivot_j[j]]), int(gen_to_ref[fk._pivot_k[j]])
                if (a in atoms) == (b in atoms):
                    continue
                quad = gen_to_ref[fk._quad_idx[j].cpu().numpy()]
                optimal_map = {int(k): int(v) for k, v in census['atom_map'].items()}
                crystal_quad = np.asarray([optimal_map.get(int(i), int(i)) for i in quad])
                target = angle(ref, crystal_quad); actual = angle(row, quad)
                seeds=dihedral_axis_from_quad(seeded[:,quad])[0].numpy()
                distance = float(np.abs(wrap(seeds-target)).min())
                attachments.append(dict(ligand=li, joint=j, reference_quad=quad.tolist(),
                    symmetry_mapped_crystal_quad=crystal_quad.tolist(),
                    row_rad=actual, crystal_rad=target, difference_rad=float(wrap(actual-target)),
                    closest_seed_rad=distance, within_02=distance <= .2))
        out.append(dict(atoms=ring['atoms'], share=ring['share'], attachments=attachments,
                        status='ok' if attachments else 'no_attachment_torsion_joint'))
    return dict(seed=seed, row=census['best_row'], rings=out,
                n_pool_variants=len(pool_bundles or [(ctx,rows)]),n_initial_rows=len(seeded),
                seed_source='current-production reconstruction; A4b selected row retained')


def census_parent(code, out, flags=None):
    start = time.monotonic()
    try:
        original, builder, ctx, assignments, W, T = prepare_parent(code, 'cpu', flags=flags)
        values = []
        aromatic = None
        for seed in (11, 12, 13):
            rows, capture = seeded_rows(builder, ctx, assignments, W, seed)
            values.append(dict(seed=seed, placement=placement(rows, capture, ctx)))
            if code in AROMATIC:
                saved = json.loads((MAIN / f'research/data/jgen/campaign/C2/rings/{code}.json').read_text())
                if seed == saved['best_seed']:
                    aromatic = aromatic_attachments(code, original, ctx, rows)
        data = dict(code=code, status='ok', seeds=values, large_rings=large_rings(original),
                    aromatic=aromatic, W=W, T=T, flags=flags or {})
    except Exception as exc:
        import traceback
        data = dict(code=code, status='failed', error=repr(exc), traceback=traceback.format_exc())
    data['wall_s'] = time.monotonic()-start
    Path(out, f'{code}.json').write_text(json.dumps(data))
    return dict(code=code, status=data['status'], wall_s=data['wall_s'], error=data.get('error'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--flags', default='{}')
    a = ap.parse_args()
    assert 1 <= a.workers <= 4
    codes, _ = population()
    if a.only:
        requested = a.only.split(','); assert set(requested) <= set(codes)
        codes = requested
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(census_parent, c, str(a.out), json.loads(a.flags)) for c in codes]
        for job in as_completed(jobs):
            print(json.dumps(job.result()), flush=True)


if __name__ == '__main__':
    main()
