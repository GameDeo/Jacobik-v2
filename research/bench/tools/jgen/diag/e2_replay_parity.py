"""E-2 step 0.3: CPU vs CUDA graph replay with fixed Q6 donor dict; optionally tail flag too."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--codes',default='XEYSUP,USAPOU,YOVNEC');ap.add_argument('--seeds',default='11');ap.add_argument('--tail',action='store_true');ap.add_argument('--control',action='store_true');a=ap.parse_args()
 flags={'nci':False,'deterministic_embed':True,'q6_donor_pairs':{'hh':0.95,'ho':0.90,'oo':0.80},'angle_tail_quadratic':bool(a.tail)}
 if a.control:flags['q6_donor_pairs']=None
 from tools.jgen.bench.run import _setup_env
 _setup_env(flags)
 import numpy as np,torch
 from jacobik.core import profile as P
 from jacobik.ik import solver_replay as SR
 from jacobik.ik.geometry import GeometryExtractor
 from jacobik.ik.ik_solver import IKSolverConfig
 from jacobik.builder import engine
 from jacobik.core.env import build_builder_and_inputs
 from tools.jdev.budget.common import stats_from_db
 from tools.jdev import multiseed
 from tools.jgen import inputs
 if not torch.cuda.is_available():raise SystemExit('CUDA required for replay parity')
 torch.set_num_threads(1);os.environ['JACOBIK_NUM_WORKERS']='1'
 stats=stats_from_db(inputs.DB_PATH);results=[];cfg=IKSolverConfig(cull_at=(),cull_keep=())
 SR.ENABLED=True;SR.FAST_FK=True;SR.ReplaySolver.replay=True;SR.FORCE_ON_CPU=False
 for code in [x.strip() for x in a.codes.split(',') if x.strip()]:
  try:
   shard,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,shard,ds);builder,_,err=build_builder_and_inputs(mol)
   if builder is None:raise RuntimeError(f'{code}: {err}')
   shell,_,_=builder._build_shell(False,False,chelate_embed=True);ge=GeometryExtractor(shell,torch.device('cpu'))
   donor_count=int(ge.n_q6_donor)
   for seed in [int(x) for x in a.seeds.split(',') if x.strip()]:
    t=time.monotonic();SR.FORCE_ON_CPU=True;SR.FAST_FK=False
    rcpu=builder.solve(seed=seed,stats=stats,device='cpu',angle_bends=True,ik_config=cfg)
    cpu_wall=time.monotonic()-t;SR.FORCE_ON_CPU=False;SR.FAST_FK=True;SR.ReplaySolver.replay=True
    with multiseed.safe_cuda_eigh_chunking():
     t=time.monotonic();rgpu=builder.solve(seed=seed,stats=stats,device='cuda',angle_bends=True,ik_config=cfg);gpu_wall=time.monotonic()-t
    last=dict(SR.LAST);cp=rcpu.conformers[0];gp=rgpu.conformers[0]
    row={'code':code,'seed':seed,'flag_fingerprint':P.snapshot()['fingerprint'],'flags':flags,'q6_donor_pair_count':donor_count,
         'cpu_top1_score':float(cp.score),'cuda_replay_top1_score':float(gp.score),'cpu_top1_key':[cp.embed_idx,int(cp.assignment_idx),int(cp.trial_idx)],
         'cuda_replay_top1_key':[gp.embed_idx,int(gp.assignment_idx),int(gp.trial_idx)],'top1_key_equal':[cp.embed_idx,int(cp.assignment_idx),int(cp.trial_idx)]==[gp.embed_idx,int(gp.assignment_idx),int(gp.trial_idx)],
         'top1_score_abs_diff':abs(float(cp.score)-float(gp.score)),'cpu_wall_s':cpu_wall,'cuda_replay_wall_s':gpu_wall,'solver_replay_last':last}
    row['replay_mode_ok']=last.get('mode')=='replay' and last.get('error') is None
    results.append(row);print(code,seed,row['cpu_top1_score'],row['cuda_replay_top1_score'],row['replay_mode_ok'],flush=True)
  except Exception as e:
   results.append({'code':code,'error':f'{type(e).__name__}: {e}','replay_mode_ok':False})

 out={'flags':flags,'profile':P.snapshot(),'n_units':len(results),'all_replay_mode_ok':all(x['replay_mode_ok'] for x in results),'units':results}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'n_units':len(results),'all_replay_mode_ok':out['all_replay_mode_ok'],'out':a.out},indent=2))
if __name__=='__main__':main()
