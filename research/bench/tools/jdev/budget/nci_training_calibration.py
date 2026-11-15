"""Freeze independent, scaffold-separated full-complex NCI motion checks.

Reference geometries are measurement inputs only. No solving, fitting on dev
cases, complex relaxation or geometry-dependent cohort selection by energy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zlib
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO,stats_from_db
from pools import setup
from nci_targeted_rescore import CANDIDATE
from energy_diagnosis import motion_modes,moved

OUT=REPO/'data/refit/nci/2026-10-18_training_calibration_v2'
STEPS=np.array([-.16,-.08,-.04,-.02,0,.02,.04,.08,.16])
N_PER_SPLIT=16


def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rank(s):return zlib.crc32(('nci-training-calibration-2026-10-18:'+s).encode())


def source_candidates():
    hist=REPO/'data/refit/nci/2026-10-18_bound_fix'
    shards=json.loads((REPO/'data/refit/reports/contacts/shards.json').read_text())
    sources={r['shard']:r for r in shards};allrows={}
    for path in sorted(hist.glob('ncihist_*.json')):
        obj=json.loads(path.read_text())
        for row in obj['census']:
            key=(obj['shard'],row['code']);allrows[key]=dict(**row,**sources[obj['shard']])
    chosen={}
    for field in ('hbond','pipi','chpi'):
        eligible=sorted((r for r in allrows.values() if r[field]>0),key=lambda r:rank(r['code']))
        for row in eligible[:160]:chosen[(row['shard'],row['code'])]=row
    return list(chosen.values()),{p.name:digest(p) for p in sorted(hist.glob('ncihist_*.json'))}


def stream(chosen):
    from fit.parser import TMQMParser,_iter_xyz_blocks,_iter_bo_blocks,_assemble_raw,_attach_qm_properties
    from jacobik.topology.mapping import assemble
    parsers={};byshard=defaultdict(dict)
    for r in chosen:byshard[(r['dir'],r['shard'])][r['code']]=r
    for (directory,shard),codes in byshard.items():
        root=Path(directory)
        if directory not in parsers:parsers[directory]=TMQMParser(root)
        parser=parsers[directory]
        for (comment,atoms),(code,bonds) in zip(_iter_xyz_blocks(root/'xyz'/f'tmQM_{shard}.xyz'),_iter_bo_blocks(root/'bo'/f'tmQM_{shard}.BO')):
            if code not in codes:continue
            if len(atoms)>160:
                yield codes[code],None,'more than 160 atoms';continue
            try:
                csv=parser.csv_index.get(code);mol=assemble(_assemble_raw(comment,atoms,code,bonds,csv))
                if csv is not None:_attach_qm_properties(mol,csv)
                yield codes[code],mol,None
            except Exception as exc:yield codes[code],None,f'{type(exc).__name__}: {exc}'


def contact_indices(t,c):
    if t=='hb':return [c[0],c[1],c[2]]
    if t=='pp':return list(c[0])+list(c[1])
    return [c[0]]+list(c[1])


def scaffold(smi):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    rd=Chem.MolFromSmiles(smi)
    if rd is None:raise ValueError('unreadable ligand scaffold')
    core=MurckoScaffold.GetScaffoldForMol(rd)
    # Acyclic chemistry has no Murcko core: retain its full canonical graph.
    return Chem.MolToSmiles(core if core.GetNumAtoms() else rd,isomericSmiles=False)


def select():
    import torch
    import networkx as nx
    from research.archive.code.interactions import nci
    from jacobik.builder.ligands import _build_ligand_groups
    from jacobik.topology.graph import donor_indices_of,get_ligand_neighbors
    from fit.pipeline.extract import benchmark_codes
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    chosen,hashes=source_candidates();excluded=benchmark_codes();assert not any(r['code'] in excluded for r in chosen)
    config=dict(source_census=hashes,exclusions=sorted(excluded),candidate_order='stable hash; up to 160 reference-contact cases per type from existing census',
                limits=dict(max_atoms=160,per_type_per_split=N_PER_SPLIT),steps=STEPS.tolist(),
                split='Parent complexes and EACH participating ligand Murcko scaffold are connected before splitting, across all contact types; no energy selection.')
    write(OUT/'selection_contract.json',config)
    stats=stats_from_db(CANDIDATE);tables=stats.nci_tables();records=[];failures=[];contexts=defaultdict(lambda:defaultdict(int))
    for k,(row,mol,error) in enumerate(stream(chosen),1):
        code=row['code']
        if error:failures.append(dict(code=code,error=error));continue
        try:
            if mol.spin_state is None or mol.spin_state<1:raise ValueError('unknown spin')
            groups=_build_ligand_groups(mol,mol.smiles)
            pos={a.idx:i for i,a in enumerate(mol.atoms)};nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
            owner={};scaffolds={}
            for j,g in enumerate(groups):
                smi=g.get('canonical_smiles')
                if not smi:raise ValueError('missing ligand SMILES')
                scaffolds[j]=scaffold(smi)
                for idx in g['xyz_indices_0based']:owner[mol.atoms[idx].idx]=j
            bound=set(donor_indices_of(mol));X=np.array([a.position for a in mol.atoms]);xt=torch.tensor(X[None],dtype=torch.float32)
            rt=nci.build_runtime(mol,tables,'cpu',types=nci.TYPES) or {};cands=nci.candidates(mol)
            for t,e in rt.items():
                cs=[c for c in cands[t] if any(key in tables for key in c[-1])]
                r,angle=nci.features(t,xt,e['idx']);r=r[0].numpy();angle=angle[0].numpy()
                threshold=(r<=2.5)&(angle>=120) if t=='hb' else ((r<=4.4)&(angle<=30) if t=='pp' else (r<=3.)&(angle<=30))
                eligible=[]
                for index in np.flatnonzero(threshold):
                    c=cs[index];atoms=contact_indices(t,c)
                    if any(a not in owner for a in atoms):continue
                    ligand_ids=sorted(set(owner[a] for a in atoms))
                    key=t+':'+repr(sorted(scaffolds[j] for j in ligand_ids))
                    if t=='pp':context='both_bound_faces' if set(c[0])<=bound and set(c[1])<=bound else 'one_bound_face' if set(c[0])<=bound or set(c[1])<=bound else 'unbound_faces'
                    elif t=='chp':
                        parent=nb[c[0]][0];context=('bound_parent' if parent in bound else 'unbound_parent')+'_'+('bound_ring' if set(c[1])<=bound else 'unbound_ring')
                    else:context='one_bound_partner' if c[0] in bound or c[2] in bound else 'outer_sphere'
                    contexts[t][context]+=1
                    eligible.append(dict(type=t,index=int(index),r=float(r[index]),angle=float(angle[index]),atoms=atoms,
                                         scaffold_key=key,scaffolds=sorted(set(scaffolds[j] for j in ligand_ids)),context=context,ligand_ids=ligand_ids))
                if not eligible:continue
                pick=min(eligible,key=lambda c:rank(code+repr(c['atoms'])))
                records.append(dict(code=code,shard=row['shard'],directory=row['dir'],dataset=row['dataset'],
                    charge=mol.charge,mult=mol.spin_state,metal=mol.metal_symbol,**pick))
        except Exception as exc:failures.append(dict(code=code,error=f'{type(exc).__name__}: {exc}'))
        if k%25==0:print(k,'candidates',len(records),'contacts',len(failures),'failures',flush=True)
    graph=nx.Graph()
    for r in records:
        for core in r['scaffolds']:graph.add_edge('parent:'+r['code'],'scaffold:'+core)
    components=list(nx.connected_components(graph));components.sort(key=lambda c:rank(repr(sorted(c))))
    counts={s:{t:0 for t in nci.TYPES} for s in ('calibration','validation')};selected=[];used=set()
    for nodes in components:
        candidates=[r for r in records if 'parent:'+r['code'] in nodes]
        # Assign each entire connected group to the currently less-filled split.
        scores={s:sum(min(N_PER_SPLIT-counts[s][t],sum(r['type']==t for r in candidates)) for t in nci.TYPES) for s in counts}
        split=max(('calibration','validation'),key=lambda s:(scores[s],-sum(counts[s].values())))
        for r in sorted(candidates,key=lambda r:rank(r['code']+r['type'])):
            if counts[split][r['type']]>=N_PER_SPLIT:continue
            selected.append(dict(**r,split=split));counts[split][r['type']]+=1;used.add(r['code'])
    assert not ({r['code'] for r in selected if r['split']=='calibration'}&{r['code'] for r in selected if r['split']=='validation'})
    assert not ({r['scaffold_key'] for r in selected if r['split']=='calibration'}&{r['scaffold_key'] for r in selected if r['split']=='validation'})
    assert not ({core for r in selected if r['split']=='calibration' for core in r['scaffolds']}&{core for r in selected if r['split']=='validation' for core in r['scaffolds']})
    write(OUT/'selection.json',dict(config=config,counts=counts,selected=selected,failures=failures,
        candidate_contact_contexts={t:dict(v) for t,v in contexts.items()},available_contacts=len(records),scaffold_components=len(components),
        source_hash=__import__('tools.jdev.env',fromlist=['tree_hash']).tree_hash(),driver_sha256=digest(Path(__file__)),
        coverage_pass=all(n==N_PER_SPLIT for s in counts.values() for n in s.values())))
    print(json.dumps(dict(counts=counts,contexts={t:dict(v) for t,v in contexts.items()},groups=len(components),failures=len(failures))),flush=True)


def prepare():
    import torch
    from research.archive.code.interactions import nci
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from jacobik.topology.decompose import decompose_ligand
    torch.set_num_threads(1);selection=json.loads((OUT/'selection.json').read_text());stats=stats_from_db(CANDIDATE)
    assert selection['coverage_pass'],'independent scaffold-disjoint cohort coverage failed; do not measure or fit'
    chosen={}
    for r in selection['selected']:chosen[(r['shard'],r['code'])]=dict(code=r['code'],shard=r['shard'],dir=r['directory'])
    failures=[];completed=[]
    for row,mol,error in stream(chosen.values()):
        code=row['code'];requests=[r for r in selection['selected'] if r['code']==code]
        if all((OUT/f"{r['type']}_{code}.npz").exists() for r in requests):continue
        try:
            if error:raise ValueError(error)
            for lig in mol.ligands:decompose_ligand(mol,lig)
            X=np.array([a.position for a in mol.atoms]);modes=motion_modes(mol,X)
            ge=GeometryExtractor(mol,torch.device('cpu'));sc=Scorer(mol,ge,stats,torch.device('cpu'))
            rt=nci.build_runtime(mol,stats.nci_tables(),'cpu',types=nci.TYPES)
            for request in requests:
                t=request['type'];j=request['index'];e=rt[t]
                test=np.array([moved(X,m,h) for m in modes for h in (-.02,.02)])
                rr,aa=nci.features(t,torch.tensor(test,dtype=torch.float32),e['idx'])
                dr=(rr[1::2,j]-rr[::2,j]).numpy()/.05;da=(aa[1::2,j]-aa[::2,j]).numpy()/5.
                magnitude=dr**2+da**2
                chosen_modes=[modes[i] for i in np.argsort(-magnitude)[:6] if magnitude[i]>1e-10]
                assert chosen_modes,'no reachable contact-changing motions'
                coordinates=np.array([X]+[moved(X,m,h) for m in chosen_modes for h in STEPS if h!=0])
                x=torch.tensor(coordinates,dtype=torch.float32);values={}
                flag,types=nci.NCI_ON,nci.ACTIVE_TYPES
                try:
                    for enabled in (False,True):
                        nci.NCI_ON=enabled;nci.ACTIVE_TYPES=nci.TYPES;sc._nci=rt if enabled else None
                        with torch.no_grad():total,bd=sc.score(x)
                        for key,v in bd.items():values[f'{"on" if enabled else "off"}_{key}']=v.numpy()*float(sc._agg_norm())
                        values['on_total' if enabled else 'off_total']=total.numpy()*float(sc._agg_norm())
                    for typ,entry in rt.items():values['NCI_'+typ]=nci.evaluate({typ:entry},x).numpy()
                finally:nci.NCI_ON=flag;nci.ACTIVE_TYPES=types
                stem=OUT/f'{t}_{code}'
                np.savez_compressed(stem.with_suffix('.npz'),X=coordinates,**values)
                write(stem.with_suffix('.json'),dict(request=request,symbols=[a.symbol for a in mol.atoms],charge=mol.charge,mult=mol.spin_state,
                    modes=[dict(label=m['label'],kind=m['kind']) for m in chosen_modes],steps=[float(h) for h in STEPS if h!=0],
                    geometry_source='unchanged training reference; native topology, fixed reference-site statistical keys; no solve/bend-prior reconstruction',
                    selection_sha256=digest(OUT/'selection.json'),db_sha256=digest(CANDIDATE)))
                completed.append(stem.name);print(stem.name,len(coordinates),'prepared',flush=True)
        except Exception as exc:failures.append(dict(code=code,error=f'{type(exc).__name__}: {exc}'));print('FAILED',failures[-1],flush=True)
        write(OUT/'prepare_report.json',dict(completed=completed,failures=failures))


def reuse():
    """Reuse identical physical inputs, never old fits or split assignments."""
    import shutil
    old=OUT.with_name('2026-10-18_training_calibration')
    selection=json.loads((OUT/'selection.json').read_text());prior=json.loads((old/'selection.json').read_text())
    assert selection['coverage_pass'] and selection['source_hash']==prior['source_hash']
    copied=[]
    for r in selection['selected']:
        src=old/f"{r['type']}_{r['code']}.npz";dst=OUT/src.name
        if not src.exists() or dst.exists():continue
        meta=json.loads(src.with_suffix('.json').read_text());before=meta['request']
        for key in ('code','type','index','atoms','r','angle','charge','mult'):
            assert before[key]==r[key],(src,key)
        assert meta['db_sha256']==digest(CANDIDATE)
        shutil.copyfile(src,dst)
        energy=src.with_suffix('.mace64.json')
        if energy.exists():shutil.copyfile(energy,dst.with_suffix('.mace64.json'))
        meta.update(request=r,selection_sha256=digest(OUT/'selection.json'),
                    reused_measurement=dict(path=str(src.relative_to(REPO)),npz_sha256=digest(src),
                        note='Identical unchanged reference/motion input. Old split and fits discarded; new grouping uses each scaffold, without energy selection.'))
        write(dst.with_suffix('.json'),meta);copied.append(dst.stem)
    write(OUT/'reuse_report.json',dict(copied=copied,scope='Only raw identical coordinates/scores/energies reused, never old fitted coefficients or validation decisions.'))
    print('Reused',len(copied),flush=True)


def measure():
    import torch
    from tools.jdev.env import tree_hash
    from jacobik.core.energy import MACEPolarEvaluator
    torch.set_num_threads(1)
    selection=json.loads((OUT/'selection.json').read_text())
    assert selection['coverage_pass'] and selection['source_hash']==tree_hash()
    files=sorted(OUT.glob('hb_*.npz'))+sorted(OUT.glob('pp_*.npz'))+sorted(OUT.glob('chp_*.npz'))
    assert len(files)==96,'incomplete prepared cohort'
    manifest=dict(source_hash=tree_hash(),selection_sha256=digest(OUT/'selection.json'),db_sha256=digest(CANDIDATE),
        model_sha256=digest(REPO/'tools/jdev/MACE-POLAR-1-S.model'),driver_sha256=digest(Path(__file__)),
        coordinates_and_scores={p.name:digest(p) for p in files},evaluator='MACE-POLAR-1-S cuda float64, original complex charge/multiplicity',
        reused_measurements=json.loads((OUT/'reuse_report.json').read_text()) if (OUT/'reuse_report.json').exists() else None)
    mp=OUT/'measurement_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'measurement provenance changed'
    else:write(mp,manifest)
    ev=MACEPolarEvaluator(model='polar-1-s',device='cuda',dtype='float64',threads=1,model_path=REPO/'tools/jdev/MACE-POLAR-1-S.model')
    for f in files:
        target=f.with_suffix('.mace64.json')
        if target.exists():continue
        meta=json.loads(f.with_suffix('.json').read_text())
        with np.load(f) as z:coords=z['X']
        energy=ev.energies([(meta['symbols'],x,meta['charge'],meta['mult']) for x in coords])*96.4853321233
        assert np.isfinite(energy).all()
        write(target,dict(relative=(energy-energy[0]).tolist(),reference=float(energy[0])))
        print(f.stem,len(energy),'MACE64',flush=True)
    assert tree_hash()==manifest['source_hash'] and digest(CANDIDATE)==manifest['db_sha256']
    write(OUT/'measurement_complete.json',dict(cases=len(files),manifest_sha256=digest(mp),energies={p.with_suffix('.mace64.json').name:digest(p.with_suffix('.mace64.json')) for p in files}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','prepare','measure','reuse']);a=p.parse_args()
    setup();{'select':select,'prepare':prepare,'measure':measure,'reuse':reuse}[a.stage]()
