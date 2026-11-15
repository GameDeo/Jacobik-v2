"""Read-only reconstruction of the old NUWFAM per-H mixture components."""
import os,json,collections
os.environ['JACOBIK_PROFILE']='u1';os.environ['JACOBIK_PROFILE_OVERRIDES']=json.dumps({'nci':False,'deterministic_embed':True,'embed_haptic_li_cap':True});os.environ['OMP_NUM_THREADS']='3'
import numpy as np,torch
from pathlib import Path
from tools.jgen.diag import ctxio
from tools.jdev import multiseed
from jacobik.core.env import build_builder_and_inputs
from jacobik.builder import engine as E
from jacobik.ik import scorer as S,angle_bends as AB
from jacobik.ik.fk_compose import ComplexFK
from jacobik.ik.kabsch import _ref_rows
from jacobik.stats.stats import StatsLookup,load_db
from tools.jgen.embedding_validity import EmbeddingValidity
S.CAP_GEMINAL_FACE=False;AB.GEOMETRIC_LEAN_WALL=False
stats=StatsLookup(load_db(str(ctxio.CONSTRAINTS)));guard=EmbeddingValidity(E).install();guard.parent='NUWFAM'
b,_,_=build_builder_and_inputs(ctxio.load_parent('NUWFAM'));variants=b._ez_combinations();b._ez_inner=True
root=Path('research/data/jgen/campaign/H/h3x/nuwfam_2060/cap/xyz/NUWFAM');out=[];models={}

def observe(X,sc,ge):
 F=torch.nn.functional;M=X[:,ge.metal_pos_idx];found={}
 for h in sc._so_hapt:
  P=X[:,h['face']];fm=h['fmask'][None,:,:,None];c=(P*fm).sum(2)/fm.sum(2).clamp(min=1.)
  Q=P-c[:,:,None];Qn=torch.gather(Q,2,h['nxt'][None,:,:,None].expand(Q.shape[0],-1,-1,3))
  n=(torch.cross(Q,Qn,dim=-1)*h['emask'][None,:,:,None]).sum(2);uM=F.normalize(M[:,None]-c,dim=-1)
  bv=F.normalize(P[:,:,1]-P[:,:,0],dim=-1);n2=uM-(uM*bv).sum(-1,keepdim=True)*bv;e2=h['eta2'][None,:,None]
  n=F.normalize(e2*n2+(1.-e2)*n,dim=-1);tilt=(n*uM).sum(-1);sign=torch.where(tilt<0,-1.,1.);n=n*sign[:,:,None];tilt=tilt*sign
  u=F.normalize(X[:,h['sx']]-X[:,h['sa']],dim=-1);sm=h['smask'][None]
  bend=((u*n[:,:,None]).sum(-1)*sm).sum(-1)/sm.sum(-1).clamp(min=1.)
  fm2=h['fmask'][None];nf=fm2.sum(-1).clamp(min=1.);d=torch.linalg.norm(P-M[:,None,None],dim=-1);dm=(d*fm2).sum(-1)/nf
  slip=torch.sqrt(((d-dm[:,:,None]).square()*fm2).sum(-1)/nf+1e-12)/dm
  features=torch.stack([tilt,bend,slip],dim=-1)[...,h['sel']];diff=features[:,:,None]-h['mu'][None]
  maha=torch.einsum('bnki,nkij,bnkj->bnk',diff,h['prec'],diff)
  logp=-.5*(maha+h['logdet'][None]+h['dim']*np.log(2*np.pi))+h['logw'][None];prob=logp.softmax(-1)
  for j in range(h['sa'].shape[0]):
   atom=int(h['sx'][j,0]);pivot=int(h['sa'][j,0])
   if pivot==1 and atom in (40,41) and int(h['smask'][j].sum())==1:
    found[atom]={'mode':prob[:,j].argmax(-1).cpu().numpy(),'probabilities':prob[:,j].cpu().numpy(),'lean':bend[:,j].cpu().numpy()}
    valid=h['logw'][j]>-1e8
    models[str(atom)]={'features':h['sel'].tolist(),'means':h['mu'][j,valid].tolist(),'weights':h['logw'][j,valid].exp().tolist(),
                       'covariances':torch.linalg.inv(h['prec'][j,valid]).tolist()}
 assert set(found)=={40,41},found.keys()
 return found

with torch.no_grad(),multiseed.safe_cuda_eigh_chunking():
 for vi,fixes in enumerate(variants):
  for ligand,fix in zip(b.ligands,fixes):ligand.ez_fix=fix
  ctx,assign,width,_=multiseed.prepare(b,11,device='cuda',angle_bends=True,stats=stats)
  (fks,_,ge,sc),rows=multiseed.seed_rows(b,ctx,assign,width,(11,12,13));guard.check('NUWFAM')
  keep=b._class_budget_rows(ctx,assign,width) if E.CLASS_BUDGET=='split' else None
  cfk=ComplexFK(fks)
  for si,seed in enumerate((11,12,13)):
   theta,Xref,_,cand=rows[si]
   if keep is not None:
    kt=torch.as_tensor(keep,device='cuda');theta=[t[kt] for t in theta];Xref=Xref[kt] if Xref.dim()==3 else Xref;cand=cand[keep]
   initial=cfk.apply(theta,Xref) if cfk.supported else _ref_rows(Xref,theta[0].shape[0])
   if not cfk.supported:
    for fk,t in zip(fks,theta):initial=fk.apply(t,initial)
   start=observe(initial,sc,ge)
   files=[];headers=[];xyz=[]
   for p in sorted(root.glob(f'*s{seed}_row*.xyz')):
    lines=p.read_text().splitlines();header=json.loads(lines[1])
    if header['row_key'][0]!=vi:continue
    files.append(p);headers.append(header);xyz.append([[float(v) for v in line.split()[1:4]] for line in lines[2:]])
   final=observe(torch.tensor(xyz,dtype=torch.float32,device='cuda'),sc,ge)
   for i,(p,h,X) in enumerate(zip(files,headers,np.asarray(xyz))):
    trial=int(h['row_key'][2]);assert int(cand[trial])==int(h['row_key'][1])
    u=X[40]-X[1];v=X[41]-X[1];angle=float(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
    row={'seed':seed,'variant':vi,'row':h['row_index'],'trial':trial,'geminal_angle_deg':angle,'collapsed_lt15':angle<15,
         'initial_modes':[int(start[a]['mode'][trial]) for a in (40,41)],'final_modes':[int(final[a]['mode'][i]) for a in (40,41)],
         'initial_probabilities':[start[a]['probabilities'][trial].tolist() for a in (40,41)],
         'final_probabilities':[final[a]['probabilities'][i].tolist() for a in (40,41)],
         'initial_leans':[float(start[a]['lean'][trial]) for a in (40,41)],'final_leans':[float(final[a]['lean'][i]) for a in (40,41)]}
    out.append(row)
summary=[]
for seed in (11,12,13):
 rows=[r for r in out if r['seed']==seed];bad=[r for r in rows if r['collapsed_lt15']]
 summary.append({'seed':seed,'pool_rows':len(rows),'collapsed_rows':len(bad),
                 'collapsed_initial_same_mode':sum(r['initial_modes'][0]==r['initial_modes'][1] for r in bad),
                 'collapsed_initial_distinct_modes':sum(r['initial_modes'][0]!=r['initial_modes'][1] for r in bad),
                 'collapsed_no_MAP_mode_change':sum(r['initial_modes']==r['final_modes'] for r in bad),
                 'all_rows_initial_same_mode':sum(r['initial_modes'][0]==r['initial_modes'][1] for r in rows)})
m=json.loads((root/'NUWFAM_cap_s11_pool.json').read_text());order=np.argsort(m['gen_to_ref_idx']);reference_X=np.asarray(m['ref_X'])[order]
generated_symbols=[line.split()[0] for line in (root/'NUWFAM_cap_s11_top1.xyz').read_text().splitlines()[2:]]
assert np.asarray(m['ref_symbols'])[order].tolist()==generated_symbols
with torch.no_grad():reference_obs=observe(torch.as_tensor(reference_X,dtype=torch.float32,device='cuda')[None],sc,ge)
u=reference_X[40]-reference_X[1];v=reference_X[41]-reference_X[1]
reference={'geminal_angle_deg':float(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))),
           'assignments':[{'atom':atom,'lean':float(reference_obs[atom]['lean'][0]),'mode':int(reference_obs[atom]['mode'][0]),
                           'probabilities':reference_obs[atom]['probabilities'][0].tolist()} for atom in (40,41)]}
result={'note':'GPU reconstructed starts matched by E/Z, assignment and trial. Old records lack initial-state hashes; MAP components are not a locked-optimization experiment.',
        'models':models,'summary':summary,'rows':out,'reference':reference}
Path('research/data/jgen/campaign/H/h3b/start_mode_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'models':models,'summary':summary},indent=2));guard.close()
