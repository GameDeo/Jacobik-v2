"""Assignment-matched physical occurrences and parent-balanced source rows.

Legacy development pools lack assignment equivalence certificates. Strict extraction
records them as missing. An explicit geometry-only development policy permits the
frozen donor-angle fingerprint diagnostic; it is not a production state certificate.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import re
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2 import FAMILIES, enumerate_opportunities, features_torch
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN, Run, digest, load_json, write_json
from research.archive.code.interactions.nci_v2.nci_v2_poolio import mapped_geometry, certified_assignment_mask


def freeze(x):
    return tuple(freeze(v) for v in x) if isinstance(x, (list, tuple)) else x


def graph_record(mol):
    return dict(csd_code=mol.csd_code, metal_idx=mol.metal_idx,
                metal_symbol=mol.metal_symbol, coord_geom=mol.coord_geom,
                charge=int(mol.charge), spin_state=int(mol.spin_state),
                atoms=[dict(idx=a.idx, symbol=a.symbol, position=np.asarray(a.position).tolist(),
                            formal_charge=int(getattr(a, 'formal_charge', getattr(a, 'charge', 0)) or 0)) for a in mol.atoms],
                bonds=[dict(i=b.i, j=b.j, order=float(b.order)) for b in mol.bonds],
                # Declared sites must survive loading (bound roles are graph descriptors).
                coord_sites=[dict(donor_indices=list(s.donor_indices)) for s in getattr(mol, 'coord_sites', [])])


def graph_from_record(raw):
    d = copy.deepcopy(raw)
    for k in ('atoms', 'bonds', 'coord_sites'):
        d[k] = [SimpleNamespace(**a) for a in d.get(k, [])]
    return SimpleNamespace(**d)


def unique_opportunities(opps):
    """Deduplicate representations, never distinct symmetry-related physical anchors."""
    out = copy.deepcopy(opps)
    for fam, occs in out.opportunities.items():
        seen, keep = set(), []
        for o in occs:
            if fam == 'ring_ring':
                key = tuple(sorted((tuple(sorted(o['ring_indices_R'])), tuple(sorted(o['ring_indices_S'])))))
            else:
                bond = (o['idx_A'], o['idx_B'])
                if o.get('symmetric_bond'):
                    bond = tuple(sorted(bond))
                key = (bond, o['idx_C'] if fam == 'bond_atom' else tuple(sorted(o['ring_indices'])))
            if key not in seen:
                keep.append(o)
                seen.add(key)
        out.opportunities[fam] = keep
    out.total_count = sum(map(len, out.opportunities.values()))
    return out


def same_pose(a, b, tolerance=1e-5, mol=None):
    """Proper-rotation, graph-symmetry-aware duplicate test; never merge mirrors.

    Complete geometric edge labels prune graph isomorphism, avoiding enumeration
    of the entire chemical automorphism group (including methyl H permutations).
    """
    a, b = a-a.mean(0), b-b.mean(0)
    if np.sqrt(np.mean((np.sort(np.linalg.norm(a,axis=1))-np.sort(np.linalg.norm(b,axis=1)))**2))>tolerance:
        return False
    def aligned(x,y):
        u, _, vt = np.linalg.svd(x.T @ y)
        correction = np.diag([1., 1., np.linalg.det(u @ vt)])
        return np.sqrt(np.mean(np.sum((x @ (u @ correction @ vt)-y)**2, axis=1))) <= tolerance
    if aligned(a,b): return True
    if mol is None: return False
    import networkx as nx
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors
    bound=set(donor_indices_of(mol)); pos={v.idx:i for i,v in enumerate(mol.atoms)}
    nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
    edges={tuple(sorted((pos[i],pos[j]))) for i,ns in nb.items() for j in ns}
    orders={tuple(sorted((pos[b.i],pos[b.j]))): ('single' if b.order<1.4 else 'double' if b.order<=2.4 else 'triple') for b in mol.bonds if tuple(sorted((pos[b.i],pos[b.j]))) in edges}
    def graph(X):
        G=nx.Graph()
        for i,atom in enumerate(mol.atoms):
            G.add_node(i,chem=(atom.symbol,getattr(atom,'formal_charge',0),atom.idx in bound,atom.idx==mol.metal_idx),
                       radius=float(np.linalg.norm(X[i])))
        for i in range(len(X)):
            for j in range(i+1,len(X)):
                G.add_edge(i,j,bond=orders.get((i,j)),distance=float(np.linalg.norm(X[i]-X[j])))
        return G
    slack=2*np.sqrt(len(a))*tolerance+1e-12
    matcher=nx.algorithms.isomorphism.GraphMatcher(graph(a),graph(b),
        node_match=lambda x,y:x['chem']==y['chem'] and abs(x['radius']-y['radius'])<=slack,
        edge_match=lambda x,y:x['bond']==y['bond'] and abs(x['distance']-y['distance'])<=slack)
    for mapping in matcher.isomorphisms_iter():
        if aligned(a,b[[mapping[i] for i in range(len(a))]]): return True
    return False


def deduplicate_poses(poses, tolerance=1e-5, mol=None):
    keep = []
    for x in np.asarray(poses, dtype=float):
        if np.isfinite(x).all() and not any(same_pose(x, y, tolerance, mol) for y in keep):
            keep.append(x)
    return np.asarray(keep).reshape(-1, *poses.shape[1:])


def balanced_rows(rows):
    """Equal priors within each family; preserve separate unnormalized parent masses."""
    totals = Counter()
    for r in rows:
        totals[(r['family'], r['y'])] += r['weight']
    return [dict(r, fit_weight=r['weight']/(2*totals[(r['family'], r['y'])])) for r in rows]


def parent_rows(parent, mol, ref_X, seed_poses, schema=None):
    opps = unique_opportunities(enumerate_opportunities(mol, schema))
    poses = {str(s): deduplicate_poses(np.asarray(x),mol=mol) for s, x in seed_poses.items() if len(x)}
    poses = {s: x for s, x in poses.items() if len(x)}
    counts = {f: len(opps.opportunities[f]) for f in FAMILIES}
    rows = []
    # If no matched baseline exists, neither source is included in the contrast.
    if not poses:
        return rows, dict(code=parent['code'], counts=counts, matched_seeds=0, status='missing_matched_assignment')
    all_X = np.concatenate([np.asarray(ref_X)[None]] + list(poses.values()))
    fb = features_torch(torch.tensor(all_X, dtype=torch.float64), opps, schema)
    reversed_opps = copy.deepcopy(opps)
    has_symmetric = False
    for candidates in reversed_opps.opportunities.values():
        for o in candidates:
            if o.get('symmetric_bond'):
                o['pos_A'], o['pos_B'] = o['pos_B'], o['pos_A']
                has_symmetric = True
    reverse = features_torch(torch.tensor(all_X, dtype=torch.float64), reversed_opps, schema) if has_symmetric else fb
    wp = float(parent.get('parent_weight', 1.0))
    for fam in FAMILIES:
        O = counts[fam]
        if not O:
            continue
        dim = [2 if fam == 'bond_atom' and not o['has_c_neighbors'] else 4 for o in opps.opportunities[fam]]
        def emit(k, y, weight, seed=None, pose=None):
            for i, o in enumerate(opps.opportunities[fam]):
                rows.append(dict(parent_id=parent['code'], group_id=parent['group_id'], family=fam,
                                 occurrence_id=o['occ_id'], context=o['z_o'],
                                 features=fb.features[fam][k, i, :dim[i]].tolist(),
                                 full_features=fb.features[fam][k, i].tolist(),
                                 feature_views=sorted([fb.features[fam][k,i,:dim[i]].tolist(),reverse.features[fam][k,i,:dim[i]].tolist()]) if o.get('symmetric_bond') else [fb.features[fam][k,i,:dim[i]].tolist()],
                                 has_c_neighbors=o.get('has_c_neighbors'), y=y, weight=weight,
                                 seed=seed, pose=pose, units=fb.units[fam][:dim[i]],
                                 admission_class=o['admission_class']))
        emit(0, 1, wp/O)
        k = 1
        for seed, X in poses.items():
            for j in range(len(X)):
                emit(k, 0, wp/(O*len(poses)*len(X)), seed, j)
                k += 1
    return rows, dict(code=parent['code'], counts=counts, parent_weight=wp,
                      matched_seeds=len(poses), deduplicated_poses={s: len(x) for s, x in poses.items()}, status='matched')


def development_assignment_mask(mol, ref, X, tolerance_deg=15.):
    """Conservative geometry diagnostic, deliberately separate from strict certificates.

    Compare sorted donor-pair angles within graph WL classes. Reference donor identities
    and roles are frozen. This allows symmetry-equivalent labels but not arbitrary asg=0.
    """
    from jacobik.topology.graph import get_ligand_neighbors, wl_classes
    from jacobik.benchmark.coordination import coordination_integrity
    from jacobik.stats.mapping import _covalent_radius
    symbols = [a.symbol for a in mol.atoms]
    m = next(i for i, a in enumerate(mol.atoms) if a.idx == mol.metal_idx)
    dr = np.linalg.norm(ref-ref[m], axis=1)
    donors = [i for i,s in enumerate(symbols) if i != m and s != 'H' and
              dr[i] <= 1.15*(_covalent_radius(symbols[m])+_covalent_radius(s))]
    nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
    cls = wl_classes([a.idx for a in mol.atoms], nb, {a.idx:a.symbol for a in mol.atoms})
    pairs = defaultdict(list)
    for j,i in enumerate(donors):
        for k in donors[j+1:]:
            key = tuple(sorted((str(cls[mol.atoms[i].idx]), str(cls[mol.atoms[k].idx]))))
            pairs[key].append((i,k))
    def angles(x, pp):
        v = x-x[m]; v /= np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
        return np.sort([np.degrees(np.arccos(np.clip(v[i] @ v[j],-1,1))) for i,j in pp])
    expected = {k:angles(ref, pp) for k,pp in pairs.items()}
    mask = np.array([all(np.max(np.abs(angles(x, pp)-expected[k]), initial=0) <= tolerance_deg
                         for k,pp in pairs.items()) for x in X])
    return mask & coordination_integrity(symbols, ref, X)['intact']


def extract_stage(spec_path, frame_path, pools, clean_root, out=None, development_geometry=False, codes=None):
    spec, frame = load_json(spec_path), load_json(frame_path)
    parents = frame['pilot_cohort']['parents']
    if codes:
        parents = [p for p in parents if p['code'] in set(codes)]
    if not parents:
        raise ValueError('no development parents selected')
    pool_paths = []
    allowed = [MAIN/'data/refit/nci_v2'/p for p in ('pilot_v1/pools', 'recert_run_a', 'recert_arms')]
    for d in map(Path, pools):
        if not any(d.resolve().is_relative_to(a.resolve()) for a in allowed):
            raise ValueError('only allowlisted development pools may be read')
        for p in sorted(d.rglob('*.npz')):
            if any(p.name.startswith(q['code']+'_') for q in parents) and ('control' in p.name or re.fullmatch(r'.+_s[0-9]+\.npz', p.name)):
                pool_paths.append(p)
    # Explicit development repairs only; S2 repair access remains unauthorized.
    overlays = spec.get('pool_mapping_overlays', {})
    certificates = spec.get('assignment_certificate_masks', {})
    for repair in [*overlays.values(), *certificates.values()]:
        if not Path(repair).resolve().is_relative_to((Path(__file__).resolve().parents[1]/'data_dev').resolve()):
            raise ValueError('repair inputs currently restricted to data_dev; S2 access not authorized')
    inputs = [frame_path, *pool_paths, *overlays.values(), *certificates.values()]
    # Full reference inputs, including state CSV, are covered by hashes.
    for p in parents:
        base = Path(clean_root)/('lnQM' if p['dataset']=='LNf' else 'tmQM')
        inputs += [base/'xyz'/f"tmQM_{p['shard']}.xyz", base/'bo'/f"tmQM_{p['shard']}.BO", base/'tmQM_y.csv']
    run = Run('extract', spec_path, inputs, dict(development_geometry=development_geometry, codes=codes), out)
    if run.resumed:
        return run.path
    from jacobik.inputs import load_mol
    rows, coverage, diagnostics, graphs = [], [], [], {}
    # The pilot is explicitly development-only, with no population inference.
    group_counts = Counter(p['group_id'] for p in parents)
    for parent in parents:
        code = parent['code']
        print('extract', code, flush=True)
        mol = load_mol(code, parent['shard'], parent['dataset'], Path(clean_root))
        if mol is None:
            coverage.append(dict(code=code,status='missing_reference')); continue
        graphs[code] = graph_record(mol)
        ref = np.array([a.position for a in mol.atoms]); symbols = np.array([a.symbol for a in mol.atoms])
        by_seed = defaultdict(list)
        for path in pool_paths:
            if not path.name.startswith(code+'_'): continue
            with np.load(path, allow_pickle=False) as z:
                if 'error' in z.files:
                    diagnostics.append(dict(code=code,path=str(path),status='pool_error')); continue
                try:
                    if int(z['charge']) != int(mol.charge) or int(z['mult']) != int(mol.spin_state):
                        raise ValueError('parent_state_charge_or_multiplicity_mismatch')
                    if not np.allclose(z['ref_X'], ref, atol=1e-7) or not np.array_equal(z['ref_symbols'], symbols):
                        raise ValueError('reference_identity_mismatch')
                    X, mapping_info = mapped_geometry(z, pool_path=path, overlay_path=overlays.get(path.name))
                    finite = np.isfinite(X).all(axis=(1,2))
                    if path.name in certificates:
                        matched = certified_assignment_mask(certificates[path.name],path,len(X)) & finite
                        policy = 'versioned_assignment_certificate'
                    elif development_geometry:
                        matched = development_assignment_mask(mol,ref,X) & finite
                        policy = 'development_donor_angle_diagnostic'
                    else:
                        matched = np.zeros(len(X),dtype=bool); policy='missing_assignment_certificate'
                    seed = int(z['seed']) if 'seed' in z.files else int(path.stem.split('_')[1].lstrip('s'))
                    by_seed[seed].extend(X[matched])
                    # Keep other classes as geometric diagnostics, never classifier negatives.
                    missing=np.flatnonzero(~matched)
                    if len(missing):
                        opp=unique_opportunities(enumerate_opportunities(mol,spec))
                        ff=features_torch(torch.tensor(X[missing],dtype=torch.float64),opp,spec)
                        dd=run.path/'opportunities/diagnostics';dd.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(dd/(path.stem+'.npz'),pose_indices=missing,asg=z['asg'][missing],
                            **{'features_'+f:ff.features[f].numpy() for f in FAMILIES},
                            **{'occurrence_ids_'+f:np.array(ff.occurrence_ids[f],dtype=str) for f in FAMILIES})
                    diagnostics.append(dict(code=code,path=str(path),policy=policy,poses=len(X),
                        **mapping_info, matched=int(matched.sum()),missing=int((~matched).sum()),
                        assignment_classes={str(a):int(np.sum(z['asg']==a)) for a in np.unique(z['asg'])},
                        mapping_ambiguous=bool(z['mapping_ambiguous']) if 'mapping_ambiguous' in z.files else None,
                        num_automorphisms=int(z['num_automorphisms']) if 'num_automorphisms' in z.files else None,
                        stored_spec_sha256=str(z['spec_sha256']) if 'spec_sha256' in z else None,
                        requested_spec_sha256=digest(spec_path)))
                except (ValueError, KeyError) as exc:
                    diagnostics.append(dict(code=code,path=str(path),status='excluded',reason=str(exc),excluded_poses=len(z['X']) if 'X' in z.files else 0))
        pp = dict(parent, parent_weight=1/group_counts[parent['group_id']])
        rr, cc = parent_rows(pp,mol,ref,{s:np.asarray(x) for s,x in by_seed.items()},spec)
        cc['stratum'] = '_'.join(str(parent.get(k,'unknown')) for k in ('metal_row','charge_class','cn_class'))
        cc['n_heavy'] = sum(s!='H' for s in symbols)
        rows.extend(rr); coverage.append(cc)
    rows = balanced_rows(rows)
    target = run.path/'opportunities'/'rows.jsonl'; target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',',':'))+'\n')
    write_json(run.path/'opportunities/graphs.json', graphs)
    write_json(run.path/'frame.json', frame)
    strata = {}
    for stratum in sorted({c.get('stratum','unknown') for c in coverage}):
        cs = [c for c in coverage if c.get('stratum','unknown')==stratum]
        strata[stratum] = dict(submitted=len(cs),matched=sum(c['status']=='matched' for c in cs))
    report = dict(parents=coverage, strata=strata, pools=diagnostics,
                  excluded_pools=sum(p.get('status')=='excluded' for p in diagnostics),
                  excluded_poses=sum(p.get('excluded_poses',0) for p in diagnostics),
                  rows=len(rows), class_totals={f:{str(y):sum(r['fit_weight'] for r in rows if r['family']==f and r['y']==y) for y in (0,1)} for f in FAMILIES},
                  development_only=True, missing_assignments_are_negatives=False)
    write_json(run.path/'reports/coverage.json',report)
    run.finish(dict(parents=len(parents),pool_files=len(pool_paths),rows=len(rows)),
               ['Legacy development assignments require a new production certificate; diagnostic geometry matching is not such a certificate.',
                'Pilot pools may predate frozen spec; stored and requested hashes are reported, never silently equated.'])
    return run.path
