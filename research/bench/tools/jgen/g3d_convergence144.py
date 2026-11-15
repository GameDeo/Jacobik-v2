"""Amendment-3 convergence-only instrument; no energy or policy re-selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from tools.jgen import inputs
from tools.jgen.g3d_diagnostics import capture_switch,finish,censored_cap
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb
from jacobik.core.env import build_builder_and_inputs
from jacobik.core import profile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parents',nargs='+',required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    os.environ.pop('JACOBIK_NUM_WORKERS',None);torch.set_num_threads(1);xtb.require()
    assert profile.snapshot()['fingerprint']=='273f2ccbf892'
    policy=json.loads(args.policy.read_text())
    manifest=json.loads(Path('research/campaigns/g/G3D_ROWS_273f2ccbf892.json').read_text())
    measurements=json.loads(Path(policy['measurements']).read_text())
    stats=stats_from_db(inputs.DB_PATH)
    result=dict(horizon=144,switch_step=200,native_total_steps=250,seed=11,
                policy=policy,policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),parents={})
    for code in args.parents:
        started=time.monotonic()
        try:
            keys=[r['key'] for r in manifest['parents'][code]['rows']]
            shard,dataset=inputs.frame_shards()[code];mol=inputs.load_mol(code,shard,dataset)
            builder,_,err=build_builder_and_inputs(mol)
            if builder is None:raise ValueError(err)
            snapshots,activities,search_wall,ledger=capture_switch(builder,stats,11,200,250,keys,6,manifest['parents'][code]['switch_chunks'])
            name='all' if len(policy['release'])==3 else '+'.join(policy['release']) if policy['release'] else 'keep_all'
            if policy['sigma_source']=='curvature':name='curvature:'+name
            previous={tuple(k):step for v in measurements['parents'][code]['variants']
                      for k,step in zip(v['keys'],v['arms'][name]['diagnostics']['first_converged_step'])}
            rows=[];traces=[]
            for snap in snapshots:
                arm=finish(snap,policy['release'],policy['sigma_source'],144)
                steps=arm['diagnostics']['first_converged_step']
                for key,step in zip(snap.keys,steps):
                    prior=previous[tuple(key)]
                    if prior is not None and step!=prior:
                        raise ValueError(f'first-50 convergence mismatch for {key}: {prior} vs {step}')
                    if prior is None and step is not None and step<=50:
                        raise ValueError(f'previously censored row now converges before 50: {key}')
                    rows.append(dict(key=key,first_converged_step=step,censored=step is None))
                traces.append(dict(keys=snap.keys,switch_digest=hashlib.sha256(b''.join(
                    v.contiguous().numpy().tobytes() for v in snap.theta)).hexdigest(),
                    steps_executed=arm['steps_executed'],wall_s=arm['wall_s']))
            result['parents'][code]=dict(status='ok',rows=rows,row_failures=ledger,traces=traces,
                                         search_wall_s=search_wall,wall_s=time.monotonic()-started)
        except Exception as exc:
            import traceback
            result['parents'][code]=dict(status='failed',reason=str(exc),traceback=traceback.format_exc())
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
        print(code,result['parents'][code]['status'],flush=True)
        if 'GFN2 embed refinement failed' in result['parents'][code].get('reason',''):
            result['stop_reason']='nonexempt_embed_fallback';args.out.write_text(json.dumps(result,indent=2));break

if __name__=='__main__':main()
