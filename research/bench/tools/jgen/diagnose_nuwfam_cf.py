"""NUWFAM aryl C-F geometry from saved G-3c0 coordinates; no solves/energy calls."""
import argparse
import json
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


def geometry(X, rings, cf_pairs):
    ring_values = []; bonds = []
    for ri, ring in enumerate(rings):
        points = X[ring]; center = points.mean(0)
        _, _, vt = np.linalg.svd(points-center, full_matrices=False)
        normal = vt[-1]; heights = (points-center) @ normal
        ring_values.append(dict(ring=ri+1, rms_height_A=float(np.sqrt(np.mean(heights**2))),
                                max_height_A=float(np.max(np.abs(heights)))))
        for c, f in cf_pairs:
            if c not in ring:
                continue
            k = ring.index(c); prev, nxt = ring[(k-1)%len(ring)], ring[(k+1)%len(ring)]
            local = np.cross(X[prev]-X[c], X[nxt]-X[c]); local /= np.linalg.norm(local)
            bond = X[f]-X[c]; length = np.linalg.norm(bond)
            def tilt(n):
                return float(np.degrees(np.arcsin(np.clip(abs(bond @ n)/length, 0, 1))))
            bonds.append(dict(ring=ri+1, C_reference_position=c, F_reference_position=f,
                              length_A=float(length), ring_plane_tilt_deg=tilt(normal),
                              local_plane_tilt_deg=tilt(local),
                              normal_bond_component_A=float(abs(bond @ normal)),
                              F_height_from_ring_plane_A=float(abs((X[f]-center) @ normal))))
    return dict(rings=ring_values, bonds=bonds,
                cf_min_A=min(b['length_A'] for b in bonds), cf_max_A=max(b['length_A'] for b in bonds),
                max_local_tilt_deg=max(b['local_plane_tilt_deg'] for b in bonds),
                max_ring_tilt_deg=max(b['ring_plane_tilt_deg'] for b in bonds),
                max_F_height_A=max(b['F_height_from_ring_plane_A'] for b in bonds),
                max_ring_rms_A=max(r['rms_height_A'] for r in ring_values))


def stats(values):
    a=np.asarray(values)
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a,.95)), max=float(a.max()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    from tools.jgen import inputs
    shard,dataset=inputs.frame_shards()['NUWFAM']; mol=inputs.load_mol('NUWFAM',shard,dataset)
    idx={atom.idx:i for i,atom in enumerate(mol.atoms)};symbols=[atom.symbol for atom in mol.atoms]
    graph=nx.Graph()
    graph.add_nodes_from(range(len(symbols)))
    for bond in mol.bonds:
        if mol.metal_idx not in (bond.i,bond.j) and bond.order>=.5:
            graph.add_edge(idx[bond.i],idx[bond.j])
    rings=[ring for ring in nx.cycle_basis(graph) if len(ring)==6 and all(symbols[i]=='C' for i in ring)
           and any(symbols[j]=='F' for i in ring for j in graph[i])]
    rings.sort(key=min)
    cf_pairs=sorted((i,j) for ring in rings for i in ring for j in graph[i] if symbols[j]=='F')
    assert len(rings)==3 and len(set(cf_pairs))==15
    assert {f for _,f in cf_pairs}=={i for i,s in enumerate(symbols) if s=='F'}
    result=dict(parent='NUWFAM',reference_atom_indices_1based=[atom.idx for atom in mol.atoms],
                reference_symbols=symbols,rings_reference_positions=rings,cf_pairs_reference_positions=cf_pairs,
                reference=geometry(np.asarray([atom.position for atom in mol.atoms]),rings,cf_pairs),
                rows=[],arms={},mapping_checks=[])
    permutations=[]
    for arm in ['C1','P2','R1','R2plus','R2minus','A0']:
        arm_rows=[]
        for seed in [11,12,13]:
            record=root/'g3c0_ez_retry'/arm/'records'/f'NUWFAM_{seed}.json'
            r=json.loads(record.read_text());top=int(np.argmax(r['scores']));floor=int(np.argmin(r['energies_kj']))
            xyz=root/'g3c0_ez_retry'/arm/'xyz/NUWFAM'/f'NUWFAM_{arm}_s{seed}_top1.xyz'
            text=xyz.read_text().splitlines();fields=[line.split() for line in text[2:]]
            assert [f[0] for f in fields]==symbols
            target=np.asarray([[float(v) for v in f[1:4]] for f in fields])
            raw=np.asarray(r['distortion'][top]['X_A']);raw_symbols=r['distortion'][top]['symbols']
            distance=np.linalg.norm(target[:,None,:]-raw[None,:,:],axis=-1)
            mask=np.asarray(symbols)[:,None]!=np.asarray(raw_symbols)[None,:]
            row,col=linear_sum_assignment(distance+mask*1e6)
            assert np.array_equal(row,np.arange(len(symbols))) and float(distance[row,col].max())<1e-6
            assert [raw_symbols[i] for i in col]==symbols
            # Rounding cannot make the atom correspondence ambiguous.
            separations=np.linalg.norm(raw[:,None,:]-raw[None,:,:],axis=-1)+np.eye(len(raw))*1e9
            assert separations.min()>2e-6
            permutations.append(col.tolist())
            result['mapping_checks'].append(dict(arm=arm,seed=seed,max_coordinate_error_A=float(distance[row,col].max()),
                                                  reference_to_builder_positions=col.tolist()))
            for i,d in enumerate(r['distortion']):
                assert [d['symbols'][j] for j in col]==symbols
                X=np.asarray(d['X_A'])[col];g=geometry(X,rings,cf_pairs)
                entry=dict(arm=arm,seed=seed,row=i,key=r['row_keys'][i],top1=i==top,floor=i==floor,
                           score=r['scores'][i],dE_kj=r['energies_kj'][i]-r['ref_energy_kj'],**g)
                result['rows'].append(entry);arm_rows.append(entry)
        bonds=[b for r in arm_rows for b in r['bonds']]
        result['arms'][arm]=dict(rows=len(arm_rows),top1=[r for r in arm_rows if r['top1']],
            floor=[r for r in arm_rows if r['floor']],
            bond_length_A=stats([b['length_A'] for b in bonds]),
            local_tilt_deg=stats([b['local_plane_tilt_deg'] for b in bonds]),
            ring_tilt_deg=stats([b['ring_plane_tilt_deg'] for b in bonds]),
            F_height_A=stats([b['F_height_from_ring_plane_A'] for b in bonds]),
            ring_rms_A=stats([g['rms_height_A'] for r in arm_rows for g in r['rings']]))
    result['same_atom_permutation_all_arms_and_seeds']=all(p==permutations[0] for p in permutations)
    result['by_embed']={}
    for embed in sorted({r['key'][0] for r in result['rows']}):
        group=[r for r in result['rows'] if r['key'][0]==embed]
        result['by_embed'][str(embed)]=dict(rows=len(group),
            cf_min_A=min(r['cf_min_A'] for r in group), cf_max_A=max(r['cf_max_A'] for r in group),
            max_local_tilt_deg=max(r['max_local_tilt_deg'] for r in group),
            max_F_height_A=max(r['max_F_height_A'] for r in group),
            max_within_bond_length_range_A=float(np.ptp([[b['length_A'] for b in r['bonds']] for r in group],axis=0).max()),
            max_within_bond_local_tilt_range_deg=float(np.ptp([[b['local_plane_tilt_deg'] for b in r['bonds']] for r in group],axis=0).max()),
            top1_selections=[[r['arm'],r['seed']] for r in group if r['top1']])
    dest=root/'nuwfam_cf_diagnostic.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    print('Reference',json.dumps(result['reference']))
    print('Mapping consistent',result['same_atom_permutation_all_arms_and_seeds'])
    for arm,v in result['arms'].items():
        print(arm,json.dumps({k:x for k,x in v.items() if k not in ['top1','floor']}))
        for r in v['top1']:
            print(' TOP1',r['seed'],json.dumps({k:r[k] for k in ['dE_kj','cf_min_A','cf_max_A','max_local_tilt_deg','max_ring_tilt_deg','max_F_height_A','max_ring_rms_A']}))


if __name__=='__main__':main()
