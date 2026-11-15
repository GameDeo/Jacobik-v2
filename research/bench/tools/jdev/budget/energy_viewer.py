"""Build an offline 3D force/energy/motion inspector from cached diagnostics.

No MACE evaluations or optimizations. Atom identities and ligand/coordination
graphs come from engine topology; rigid alignment rotates vector fields too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, stats_from_db
from pools import setup, set_cases
from nci_targeted_rescore import CANDIDATE


def align_transform(X, reference, rows):
    """Row-vector proper Kabsch rotation; translation and vectors kept distinct."""
    a, b = np.asarray(X)[rows], np.asarray(reference)[rows]
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ca).T@(b-cb))
    rotation = u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    return rotation, cb-ca@rotation


def rotate_mode(mode, rotation, translation):
    out = {k:v for k,v in mode.items() if k not in ('v','axis','origin')}
    out['v'] = np.asarray(mode['v'])@rotation
    if 'axis' in mode:
        out['axis'] = np.asarray(mode['axis'])@rotation
        out['origin'] = np.asarray(mode['origin'])@rotation+translation
    return out


def serial(value):
    if isinstance(value,np.ndarray):return np.round(value,7).tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def build_data(roots, codes, db=CANDIDATE, case_overrides=None):
    import torch
    from jacobik.core import profile
    from research.archive.code.interactions import nci
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.topology.graph import get_ligand_neighbors
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    from energy_diagnosis import motion_modes,add_contact_modes,contact_scores

    assert profile.ACTIVE_NAME=='u1'
    torch.set_num_threads(1)
    names={c:n for split in ('dev32','n48') for c,n in set_cases(split)}
    stats=stats_from_db(db)
    db_hash=hashlib.sha256(db.read_bytes()).hexdigest()
    model_hashes=set()
    result={}
    for code in codes:
        cached={}
        for root in roots:
            folder=root/code
            refinement_file=root/'mode_refinements.json'
            refinements=json.loads(refinement_file.read_text())['poses'] if refinement_file.exists() else {}
            for f in sorted(folder.glob('*.json')):
                if not f.with_suffix('.npz').exists():continue
                obj=json.loads(f.read_text())
                if 'summary' not in obj:continue
                config=obj['provenance']['config']
                assert config['db_sha256']==db_hash, f'wrong diagnostic DB: {f}'
                model_hashes.add(config['model_sha256'])
                assert len(model_hashes)==1, 'cannot mix MACE checkpoints'
                with np.load(f.with_suffix('.npz')) as z:
                    arrays={k:z[k] for k in z.files}
                refinement=refinements.get(f'{code}/{f.stem}')
                if refinement:
                    assert hashlib.sha256(f.read_bytes()).hexdigest()==refinement['source_json_sha256']
                    assert hashlib.sha256(f.with_suffix('.npz').read_bytes()).hexdigest()==refinement['source_npz_sha256']
                    for row in obj['summary']['modes']:
                        row.update(refinement['modes'].get(row['label'],{}))
                if f.stem in cached:
                    previous=cached[f.stem]
                    assert np.allclose(previous['arrays']['X'],arrays['X'],atol=1e-8)
                    assert abs(previous['summary']['energy']['total']-obj['summary']['energy']['total'])<.002
                    continue
                cached[f.stem]=dict(summary=obj['summary'],arrays=arrays,config=obj['provenance']['config'])
        if not cached:continue
        case=case_overrides[code] if case_overrides and code in case_overrides else get_case(names[code])
        with starting_geometry(case,'full_oracle'):
            mol,_,_=case.make_builder()._build_shell(False,False)
        for lig in mol.ligands:decompose_ligand(mol,lig)
        pos={a.idx:i for i,a in enumerate(mol.atoms)}
        nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
        orders={tuple(sorted((b.i,b.j))):b.order for b in mol.bonds}
        bonds=[]
        for a in nb:
            for b in nb[a]:
                if a>=b:continue
                order=float(orders.get((a,b),1))
                aromatic=abs(order-1.5)<.1
                # Aromatic edges use one stick, never round every aromatic edge
                # to a double bond. Retain the original order in the payload.
                bonds.append(dict(i=pos[a],j=pos[b],order=1 if aromatic else max(1,min(3,round(order))),
                                  topology_order=order,aromatic=aromatic))
        sites=[dict(rows=[pos[u] for u in s.donor_indices],kind=s.kind,label=s.label) for s in mol.coord_sites]
        symbols=[a.symbol for a in mol.atoms]
        ligand=[next((l.ligand_id for l in mol.ligands if a.idx in l.atom_indices),-1) for a in mol.atoms]
        configurations=[p['config'] for p in cached.values()]
        contact_model=configurations[0].get('contact_model','legacy')
        active_types=tuple(configurations[0].get('nci_types',nci.TYPES))
        assert all(c.get('contact_model','legacy')==contact_model and
                   tuple(c.get('nci_types',nci.TYPES))==active_types for c in configurations)
        runtime=nci.build_runtime(mol,stats.nci_tables(),'cpu',types=active_types) or {}
        candidates=nci.candidates(mol);tables=stats.nci_tables()
        candidates={t:[c for c in cs if any(k in tables for k in c[-1])] for t,cs in candidates.items()}
        reference=cached['reference']['arrays']['X']
        metal=pos[mol.metal_idx]
        seeds=sorted({int(k.split('_')[0][1:]) for k in cached if k!='reference'})
        assert all(f's{s}_{kind}' in cached for s in seeds for kind in ('before','after'))
        output=dict(symbols=symbols,ligand=ligand,bonds=bonds,sites=sites,metal=metal,seeds=seeds,poses={})
        output['angle_pairs']=[[i,j] for i in range(len(sites)) for j in range(i+1,len(sites))]
        for label,p in cached.items():
            a=p['arrays'];X=a['X'];s=p['summary'];config=p['config']
            modes=motion_modes(mol,X)
            add_contact_modes(modes,mol,X,runtime,config['max_contact_modes'],contact_model)
            assert [m['label'] for m in modes]==a['mode_labels'].tolist(),(code,label)
            assert np.allclose(np.asarray([m['v'] for m in modes]),a['mode_vectors'],atol=1e-7)
            contacts=[]
            xt=torch.tensor(X[None],dtype=torch.float32)
            for t,e in runtime.items():
                r,ang=nci.features(t,xt,e['idx'])
                score=contact_scores(t,e,xt,contact_model)[0].detach().numpy()
                for j in np.argsort(-np.abs(score)):
                    if abs(score[j])<.05:continue
                    c=candidates[t][j]
                    groups=([c[0]],[c[1]],[c[2]]) if t=='hb' else (([c[0]],c[1]) if t=='chp' else (c[0],c[1]))
                    contacts.append(dict(type=t,candidate=int(j),score=float(score[j]),r=float(r[0,j]),angle=float(ang[0,j]),
                                         groups=[[pos[u] for u in group] for group in groups]))
            frames={}
            for alignment in ('core','heavy','raw'):
                if alignment=='raw':R,t=np.eye(3),np.zeros(3)
                elif alignment=='heavy':R,t=align_transform(X,reference,[i for i,sym in enumerate(symbols) if sym!='H'])
                else:
                    xp=np.stack([X[metal]]+[X[site['rows']].mean(0) for site in sites])
                    rp=np.stack([reference[metal]]+[reference[site['rows']].mean(0) for site in sites])
                    R,t=align_transform(xp,rp,np.arange(len(xp)))
                frames[alignment]=dict(X=X@R+t,forces=a['forces']@R,
                                        modes=[rotate_mode(m,R,t) for m in modes])
            bylabel={m['label']:m for m in s['modes']}
            n=len(modes);h=config['step'];scan=a['scan_energy']
            for j,m in enumerate(modes):
                row=dict(bylabel[m['label']])
                row['curve']=[float(scan[k]-scan[0]) for k in (1+n+j,1+3*n+j,0,1+2*n+j,1+j)]
                row['h']=[-h,-h/2,0,h/2,h]
                for frame in frames.values():frame['modes'][j].update(row)
            lengths=[float(np.linalg.norm(X[b['i']]-X[b['j']])) for b in bonds]
            distances=[float(np.linalg.norm(X[site['rows']].mean(0)-X[metal])) for site in sites]
            directions=np.array([X[site['rows']].mean(0)-X[metal] for site in sites])
            directions/=np.linalg.norm(directions,axis=1)[:,None]
            angles=[float(np.degrees(np.arccos(np.clip(directions[i]@directions[j],-1,1)))) for i,j in output['angle_pairs']]
            output['poses'][label]=dict(frames=frames,local=a['model_local_energy'],energy=s['energy'],
                                        contacts=contacts,bond_lengths=lengths,site_distances=distances,site_angles=angles)
        result[code]=output
        print(f'{code}: {len(cached)} poses, {len(bonds)} ligand bonds, {len(sites)} coordination sites',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--cases',nargs='+',default=['XEPKAF','FETSUU','PEWBOL','JEFJUZ','BULGAT','CBTYCO','YIYXEJ'])
    p.add_argument('--out',type=Path,default=REPO/'data/inspect/nci_force_viewer.html')
    p.add_argument('--db',type=Path,default=CANDIDATE)
    a=p.parse_args();setup()
    data=build_data(a.roots,a.cases,a.db)
    if not data:raise ValueError('no measured poses')
    lib=(REPO/'dev/tools/jdev/vendor/3Dmol-min.js').read_text(encoding='utf-8').replace('</script','<\\/script')
    payload=json.dumps(data,default=serial,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    template=Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    html=template.replace('/*__3DMOL__*/',lib).replace('/*__DATA__*/',payload)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(html,encoding='utf-8')
    a.out.with_suffix('.json').write_text(json.dumps(data,default=serial,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(a.out,flush=True)


if __name__=='__main__':main()
