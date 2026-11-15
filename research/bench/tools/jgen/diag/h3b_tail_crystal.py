import os,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['JACOBIK_PROFILE']='u1';os.environ['JACOBIK_PROFILE_OVERRIDES']=json.dumps({'nci':False,'deterministic_embed':True})
import torch,numpy as np
from tools.jgen.diag import ctxio
from tools.jdev import multiseed
from jacobik.core.env import build_builder_and_inputs
from jacobik.stats.stats import StatsLookup,load_db
from jacobik.ik import scorer as S
stats=StatsLookup(load_db(str(ctxio.CONSTRAINTS)))
for code in ['KIXVUJ','WEYJIT','FEXWUC']:
 try:
  mol=ctxio.load_parent(code);b,_,err=build_builder_and_inputs(mol)
  ctx,assign,w,_=multiseed.prepare(b,11,device='cpu',angle_bends=True,stats=stats)
  (fks,_,ge,sc),rows=multiseed.seed_rows(b,ctx,assign,w,(11,))
  # Crystal coordinates mapped from the exported pool manifest into builder order.
  from pathlib import Path
  p=Path('research/data/jgen/campaign/H/h3x/abl_2060/control_nocap/xyz')/code/f'{code}_control_s11_pool.json'
  manifest=json.loads(p.read_text());X=np.asarray(manifest['ref_X'])[np.argsort(manifest['gen_to_ref_idx'])]
  X=torch.as_tensor(X,dtype=torch.float32).unsqueeze(0)
  terms=[]
  for flag in [False,True]:
   S.ANGLE_TAIL_QUADRATIC=flag
   n=sc._q2_cand[0].shape[0] if sc._q2_cand is not None else 1
   _,breakdown=sc.score(X.expand(n,-1,-1),row_groups=torch.arange(n));terms.append(float(breakdown['Q2'].max()))
  print(code,'crystal_Q2',terms,'delta',terms[1]-terms[0],flush=True)
 except Exception as e: print(code,type(e).__name__,str(e),flush=True)
