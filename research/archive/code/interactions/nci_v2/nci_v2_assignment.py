"""Versioned P2 assignment-equivalence certificates; CPU build, never solve.

Native asg labels are used when their ordered table is reproducible. Reference
classification uses the same ideal template, modulo proper polyhedron rotations
and *whole-ligand* graph automorphisms. Explicit fallback cases use complete
angle graphs, not histograms that discard donor correlations.
"""
from __future__ import annotations
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 JACOBIK_PROFILE='u1',JACOBIK_PROFILE_OVERRIDES='{"nci":false,"deterministic_embed":true}')
import argparse,copy,hashlib,itertools,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import networkx as nx
import torch
from jacobik.topology.graph import get_ligand_neighbors
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,Run,load_json,write_json,canonical
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import reference_stream,BudgetPause
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget
from research.archive.code.interactions.nci_v2.nci_v2_extract import development_assignment_mask,graph_record

torch.set_num_threads(1)


def proper_rotation(A,B):
    U,_,Vt=np.linalg.svd(A.T@B)
    return U@np.diag([1.,1.,np.linalg.det(U@Vt)])@Vt


def polyhedron_symmetries(ideal,tolerance=1e-8):
    Q=np.asarray(ideal,float);Q=Q/np.linalg.norm(Q,axis=1,keepdims=True);n=len(Q)
    G=nx.Graph()
    for i in range(n):G.add_node(i)
    for i in range(n):
        for j in range(i+1,n):G.add_edge(i,j,dot=float(Q[i]@Q[j]))
    matcher=nx.algorithms.isomorphism.GraphMatcher(G,G,edge_match=lambda a,b:abs(a['dot']-b['dot'])<1e-9)
    result=[]
    for mapping in matcher.isomorphisms_iter():
        p=np.array([mapping[i] for i in range(n)]);R=proper_rotation(Q,Q[p])
        if np.sqrt(np.mean(np.sum((Q@R-Q[p])**2,axis=1)))<=tolerance:result.append(p)
    return result


def directions(X,groups,metal):
    D=np.stack([np.mean(X[group],axis=0)-X[metal] for group in groups]);norm=np.linalg.norm(D,axis=1)
    if np.any(norm<1e-8):raise ValueError('degenerate metal-to-site direction')
    return D/norm[:,None]


def best_template_permutation(D,ideal):
    """Bounded-memory exhaustive assignment search for CN<=9."""
    Q=np.asarray(ideal,float);Q=Q/np.linalg.norm(Q,axis=1,keepdims=True)
    n=len(D);best=np.inf;assignment=None;iterator=itertools.permutations(range(n))
    while True:
        batch=list(itertools.islice(iterator,512))
        if not batch:break
        budget();perms=np.asarray(batch,dtype=int);P=Q[perms];cross=np.einsum('bni,nj->bij',P,D)
        U,s,Vt=np.linalg.svd(cross);det=np.linalg.det(U@Vt)
        cost=np.maximum(0.,2.-2*(s[:,0]+s[:,1]+det*s[:,2])/n)
        i=int(np.argmin(cost))
        if cost[i]<best-1e-12:best=float(cost[i]);assignment=perms[i]
    R=proper_rotation(Q[assignment],D);cos=np.clip(np.sum((Q[assignment]@R)*D,axis=1),-1,1)
    rms=float(np.sqrt(np.mean(np.degrees(np.arccos(cos))**2)))
    return assignment,rms


class AssignmentGeometry:
    def __init__(self,mol,groups,ideal=None,atom_stereo=None,symmetry_tolerance=1e-8,chirality_tolerance=1e-3):
        self.mol=mol;self.X=np.asarray([a.position for a in mol.atoms],float)
        self.pos={a.idx:i for i,a in enumerate(mol.atoms)};self.metal=self.pos[mol.metal_idx]
        self.atom_groups=[list(g) for g in groups];self.groups=[[self.pos[i] for i in g] for g in groups]
        self.n=len(groups);self.ideal=None if ideal is None else np.asarray(ideal,float)
        if len(set(i for g in groups for i in g))!=sum(map(len,groups)):raise ValueError('overlapping declared coordination sites')
        self.D=directions(self.X,self.groups,self.metal)
        nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx);donors={i for g in groups for i in g};atom_stereo=atom_stereo or {}
        G=nx.Graph()
        for a in mol.atoms:
            if a.idx!=mol.metal_idx and (a.symbol!='H' or a.idx in donors):
                G.add_node(a.idx,chem=repr(('atom',a.symbol,getattr(a,'formal_charge',0),atom_stereo.get(a.idx,''))),pin='')
        for b in mol.bonds:
            if b.i in G and b.j in G and b.order>=.5:G.add_edge(b.i,b.j,order=repr(round(float(b.order),3)))
        for i,g in enumerate(groups):
            node=('site',i);G.add_node(node,chem=repr(('site','haptic' if len(g)>1 else 'kappa',len(g))),pin='')
            for a in g:G.add_edge(node,a,order='anchor')
        self.graph=G
        hashes=nx.weisfeiler_lehman_subgraph_hashes(G,edge_attr='order',node_attr='chem',iterations=3)
        self.colors=[hashes[('site',i)][-1] for i in range(self.n)]
        self.chirality_tolerance=chirality_tolerance;self.aut_cache={};self.symmetries=polyhedron_symmetries(self.ideal,symmetry_tolerance) if ideal is not None and len(ideal)==self.n and np.all(np.linalg.norm(self.ideal,axis=1)>1e-8) else []
        self.fixed_chiral=[]
        for atom in atom_stereo:
            ns=sorted(nb.get(atom,[]))
            if len(ns) not in (3,4):continue
            self.fixed_chiral.append((self.pos[atom],[self.pos[i] for i in ns]))
    def preserves_chemistry(self,p):
        key=tuple(map(int,p))
        if key in self.aut_cache:return self.aut_cache[key]
        if any(self.colors[i]!=self.colors[j] for i,j in enumerate(p)):return False
        if key==tuple(range(self.n)):return True
        A=self.graph.copy();B=self.graph.copy()
        for i,j in enumerate(p):A.nodes[('site',i)]['pin']=str(i);B.nodes[('site',int(j))]['pin']=str(i)
        match=nx.algorithms.isomorphism.GraphMatcher(A,B,node_match=lambda a,b:a['chem']==b['chem'] and a['pin']==b['pin'],edge_match=lambda a,b:a['order']==b['order'])
        result=bool(match.is_isomorphic());self.aut_cache[key]=result;return result
    def equivalent(self,a,b):
        if self.ideal is None or len(self.ideal)!=self.n:raise ValueError('template equivalence requires saturated known geometry')
        inv=np.argsort(b)
        return any(self.preserves_chemistry(inv[sym[np.asarray(a,dtype=int)]]) for sym in self.symmetries)
    def fixed_state(self,X,tolerance=None):
        tolerance=self.chirality_tolerance if tolerance is None else tolerance
        def volume(Y,center,ns):
            v=Y[ns]-Y[center]
            if len(ns)==4:v=v[:3]-v[3]
            scale=np.prod(np.maximum(np.linalg.norm(v,axis=1),1e-12))
            return float(np.linalg.det(v)/scale)
        for c,ns in self.fixed_chiral:
            a,b=volume(self.X,c,ns),volume(X,c,ns)
            if abs(a)<tolerance or abs(b)<tolerance or a*b<0:return False
        return True
    def geometric_match(self,X,tolerance=15.):
        if not self.fixed_state(X):return False
        try:D=directions(X,self.groups,self.metal)
        except ValueError:return False
        angles=lambda d:np.degrees(np.arccos(np.clip(d@d.T,-1,1)))
        left,right=angles(self.D),angles(D);A=nx.Graph();B=nx.Graph()
        for i in range(self.n):A.add_node(i,color=self.colors[i]);B.add_node(i,color=self.colors[i])
        for i in range(self.n):
            for j in range(i+1,self.n):A.add_edge(i,j,angle=float(left[i,j]));B.add_edge(i,j,angle=float(right[i,j]))
        gm=nx.algorithms.isomorphism.GraphMatcher(A,B,node_match=lambda a,b:a['color']==b['color'],edge_match=lambda a,b:abs(a['angle']-b['angle'])<=tolerance)
        for mapping in gm.isomorphisms_iter():
            p=[mapping[i] for i in range(self.n)]
            if not self.preserves_chemistry(p):continue
            # Angles alone cannot distinguish a reflected chiral metal center.
            target=D[p];R=proper_rotation(self.D,target)
            rms=np.sqrt(np.mean(np.degrees(np.arccos(np.clip(np.sum((self.D@R)*target,axis=1),-1,1)))**2))
            if rms<=tolerance+1e-8:return True
        return False


def native_build(ref,seed):
    """Rebuild topology/assignment metadata using ETKDG only; never call solve."""
    from jacobik.inputs import build_builder_and_inputs
    from jacobik.builder.atom_map import build_atom_mapping
    from rdkit import Chem
    builder,_,error=build_builder_and_inputs(ref)
    if error:raise ValueError(error)
    shell,rdmols,full=builder._build_shell(fast_embed=True,debug=False,embed_seed=seed,chelate_embed=True,embed='etkdg')
    ideal,groups,n=builder._collect_donor_groups(shell,full)
    pairs,heavy=builder._detect_chelate_pairs(shell,groups,n)
    symmetric=builder._symmetric_donor_groups(shell,rdmols,groups)
    ctx=SimpleNamespace(mol=shell,ideal=ideal,all_donor_groups=groups,n_donors=n,heavy_graph=heavy,symmetric_donor_groups=symmetric)
    base=builder._compute_baseline_assignment(shell,ideal,groups,n,pairs,heavy_graph=heavy)
    fully_explicit=all(l.sites is not None for l in builder.ligands)
    assignments=[base] if fully_explicit else builder._run_symmetry_enumeration(ctx,base,False,embed_aware=True)
    mapping=build_atom_mapping(ref,gen_mol=shell)
    from research.archive.code.interactions.nci_v2.nci_v2_poolio import checked_mapping
    checked_mapping([a.symbol for a in shell.atoms],[a.symbol for a in ref.atoms],mapping.gen_to_ref_idx)
    positions=[[int(mapping.ref_to_gen_idx[i-1]) for i in group] for _,_,group in groups]
    atom_groups=[[ref.atoms[i].idx for i in g] for g in positions]
    stereo={}
    for li,rdmol in enumerate(rdmols):
        Chem.AssignStereochemistry(rdmol,cleanIt=True,force=True)
        for atom in rdmol.GetAtoms():
            if atom.HasProp('_CIPCode'):
                generated=shell.ligands[li].atom_indices[atom.GetIdx()]-1
                ridx=int(mapping.ref_to_gen_idx[generated]);stereo[ref.atoms[ridx].idx]=atom.GetProp('_CIPCode')
    return dict(builder=builder,shell=shell,rdmols=rdmols,donor_full=full,groups=groups,atom_groups=atom_groups,
        ideal=ideal,assignments=[np.asarray(a,dtype=int) for a in assignments],mapping=mapping,stereo=stereo,
        ez_variants=len(builder._ez_combinations()),geometry=builder.geometry,
        table_fingerprint=hashlib.sha256(canonical(dict(geometry=builder.geometry,ideal=ideal,assignments=assignments,
             ligands=[dict(smiles=l.smiles,donors=l.donors) for l in builder.ligands]))).hexdigest())


def make_certificate(ref,policy):
    budget();a=native_build(ref,11);budget();b=native_build(ref,12)
    n=len(a['atom_groups'])
    if not policy['min_coordination_sites']<=n<=policy['max_coordination_sites']:
        return None,dict(status='uncertifiable',reason='coordination_site_count_outside_2_9',coordination_sites=n)
    model=AssignmentGeometry(ref,a['atom_groups'],a['ideal'],a['stereo'],
        policy['template_symmetry_rms_tolerance'],policy['chirality_volume_tolerance'])
    stable=a['table_fingerprint']==b['table_fingerprint'];known=len(a['ideal'])==n and bool(model.symmetries)
    reasons=[]
    if not stable:reasons.append('native_index_order_depends_on_build')
    if not known:reasons.append('native_geometry_is_unsaturated_or_has_no_fixed_template')
    if a['ez_variants']!=1:reasons.append('E_Z_branch_identity_not_serialized_with_asg')
    payload=dict(schema='nci_v2_assignment_certificate.1',code=ref.csd_code,coordination_sites=n,geometry=a['geometry'],
        groups_reference_atom_ids=a['atom_groups'],ideal=a['ideal'],ordered_assignments=a['assignments'],
        native_index_fingerprint=a['table_fingerprint'],native_rebuild_identical=stable,ez_variants=a['ez_variants'],
        polyhedron_proper_symmetries=len(model.symmetries),fixed_atom_stereo=a['stereo'],no_solve=True,
        build_embed='etkdg topology replay; no GFN2 energy/refinement',uncertifiable_cases_are_missing=True)
    if reasons:
        payload.update(status='certified_geometric_fallback',mode='geometric',fallback_reasons=reasons,reference_assignment_indices=None)
    else:
        best,rms=best_template_permutation(model.D,a['ideal'])
        matched=[i for i,p in enumerate(a['assignments']) if model.equivalent(best,p)]
        payload.update(reference_best_permutation=best,reference_template_rms_angle=rms,reference_assignment_indices=matched)
        if rms>policy['reference_template_max_rms_angle_degrees']:
            payload.update(status='uncertifiable',mode='native',reason='reference_too_distorted_for_template')
        else:payload.update(status='certified_native',mode='native')
    if a['ez_variants']!=1:
        payload.update(status='uncertifiable',reason='E_Z_branch_identity_unavailable_and_full_state_not_certified')
    payload['reference_state_sha256']=hashlib.sha256(canonical(dict(graph=graph_record(ref),stereo=a['stereo'],charge=ref.charge,multiplicity=ref.spin_state))).hexdigest()
    return model,payload


def certify_poses(model,payload,X,asg,policy,threshold=None):
    finite=np.isfinite(X).all((1,2));matched=np.zeros(len(X),dtype=bool)
    if model is None or payload['status']=='uncertifiable':return matched
    if payload['mode']=='native':
        limit=policy['reference_template_max_rms_angle_degrees'] if threshold is None else threshold
        if payload['reference_template_rms_angle']>limit:return matched
        valid_index=(asg>=0)&(asg<len(payload['ordered_assignments']))
        for i in np.flatnonzero(finite&valid_index):
            matched[i]=int(asg[i]) in payload['reference_assignment_indices'] and model.fixed_state(X[i])
    else:
        tol=policy['geometric_fallback_max_pair_angle_degrees'] if threshold is None else threshold
        for i in np.flatnonzero(finite):
            budget();matched[i]=model.geometric_match(X[i],tol)
    return matched


def run(spec_path,frame_path,out=None):
    if os.nice(0)<19:raise ValueError('nice -n 19 required')
    budget(start=True);spec=load_json(spec_path);policy=spec['assignment_certificate'];frame=load_json(frame_path)
    parents=frame['pilot_cohort']['parents'];clean=MAIN/'data/refit/clean'
    dirs=dict(pilot=MAIN/'data/refit/nci_v2/pilot_v1/pools',recert=MAIN/'data/refit/nci_v2/recert_run_a/pools')
    import re
    files={panel:[p for p in sorted(d.glob('*.npz')) if p.name.split('_')[0] in {q['code'] for q in parents}
        and (re.fullmatch(r'.+_s[0-9]+\.npz',p.name) if panel=='pilot' else p.name.endswith('_control.npz'))] for panel,d in dirs.items()}
    inputs=[frame_path,*sum(files.values(),[])]
    for p in parents:
        base=clean/('lnQM' if p['dataset']=='LNf' else 'tmQM');inputs.extend([base/'xyz'/f"tmQM_{p['shard']}.xyz",base/'bo'/f"tmQM_{p['shard']}.BO",base/'tmQM_y.csv'])
    run=Run('phase2-assignment-certificate',spec_path,inputs,dict(reference_only_build=True,energy_calls=0),out)
    if run.resumed:return run.path
    rows=[];certificates={}
    refparents=[dict(p,representative_code=p['code'],representative_shard=p['shard']) for p in parents]
    from jacobik.benchmark.coordination import coordination_integrity
    for parent,ref,error in reference_stream(refparents,clean):
        budget();code=parent['code'];print('assignment certificate',code,flush=True)
        if error:
            certificates[code]=dict(status='uncertifiable',reason=error);continue
        try:model,payload=make_certificate(ref,policy)
        except BudgetPause:raise
        except Exception as exc:
            certificates[code]=dict(status='uncertifiable',reason=f'{type(exc).__name__}: {exc}');continue
        certificates[code]=payload;write_json(run.path/f'certificates/{code}.json',payload)
        if model is not None:
            Xref=np.asarray([a.position for a in ref.atoms]);R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
            payload['validation']=dict(reference_self=model.geometric_match(Xref,1e-5),rigid_rotation=model.geometric_match(Xref@R+7,1e-5))
        else:Xref=np.asarray([a.position for a in ref.atoms])
        for panel,paths in files.items():
            for path in paths:
                if not path.name.startswith(code+'_'):continue
                budget()
                with np.load(path,allow_pickle=False) as z:
                    if 'error' in z.files:rows.append(dict(panel=panel,code=code,status='pool_error',path=str(path)));continue
                    from research.archive.code.interactions.nci_v2.nci_v2_poolio import mapped_geometry
                    X,_=mapped_geometry(z,pool_path=path);asg=z['asg'].astype(int)
                    symbols=np.array([a.symbol for a in ref.atoms])
                    if not np.array_equal(z['ref_symbols'],symbols) or not np.allclose(z['ref_X'],Xref,atol=1e-7):raise ValueError('reference identity mismatch')
                    matched=certify_poses(model,payload,X,asg,policy)
                    ci=coordination_integrity(symbols,Xref,X)['intact'];heuristic=development_assignment_mask(ref,Xref,X)
                    sensitivity={str(t):int((certify_poses(model,payload,X,asg,policy,t)&ci).sum()) for t in policy['sensitivity_degrees']}
                    record=dict(panel=panel,code=code,path=str(path),status=payload['status'],mode=payload.get('mode'),poses=len(X),
                        matched_assignment=int(matched.sum()),matched_and_coord_intact=int((matched&ci).sum()),heuristic_matched=int(heuristic.sum()),
                        agreement_after_integrity=float(np.mean((matched&ci)==heuristic)),threshold_sensitivity=sensitivity,
                        invalid_asg_indices=int(((asg<0)|(asg>=len(payload.get('ordered_assignments',[])))).sum()))
                    rows.append(record)
                    dest=run.path/f'masks/{panel}/{path.stem}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                    from research.archive.code.interactions.nci_v2.nci_v2_poolio import file_sha256
                    np.savez(dest,assignment_matched=matched,coord_intact=ci,heuristic_matched=heuristic,
                        source_pool_sha256=file_sha256(path),schema='nci_v2_assignment_certificate.1',
                        certificate_status=payload['status'],reference_state_sha256=payload.get('reference_state_sha256','uncertifiable'))
        write_json(run.path/f'certificates/{code}.json',payload)
    write_json(run.path/'reports/assignment.json',dict(policy=policy,certificates=certificates,pools=rows,
        energy_calls=0,solve_calls=0,production_ready=False))
    run.finish(dict(parents=len(parents),pool_files=len(rows),certified_native=sum(c.get('status')=='certified_native' for c in certificates.values())),
        ['ETKDG-only CPU topology replay; no solve or energy evaluation.',
         'E/Z branch IDs are absent from pools; such cases explicitly use a whole-graph geometric fallback.',
         'Assignment identity and coordination health are reported separately.'])
    return run.path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--frame',type=Path,required=True);p.add_argument('--out-dir',type=Path);a=p.parse_args()
    try:print(run(a.spec,a.frame,a.out_dir),flush=True)
    except BudgetPause as e:print(f'PAUSED {e}',flush=True);raise SystemExit(75)

if __name__=='__main__':main()
