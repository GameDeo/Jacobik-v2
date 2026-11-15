"""H-1 crystal-only tail census; no solves, properties or locked panels."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tools.jgen import inputs
from tools.jdev.budget.common import stats_from_db
from jacobik.ik.geometry import GeometryExtractor
from jacobik.ik.scorer import Scorer, THERMAL_SD, _torsion_tail_logp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
    codes=(root/'research/data/jgen/campaign/A/a4/codes_cal.txt').read_text().split()
    from tools.runkit import policy
    policy.assert_panel('cal',codes)
    torch.set_num_threads(1);stats=stats_from_db(Path(a.db));rows=[];failures=[]
    for code in codes:
        try:
            mol=inputs.load_mol(code,*inputs.frame_shards()[code]);device=torch.device('cpu')
            ge=GeometryExtractor(mol,device);sc=Scorer(mol,ge,stats,device)
            x=torch.tensor(np.asarray([[at.position for at in mol.atoms]]),dtype=torch.float32)
            # GeometryExtractor centers only vector-dependent features internally.
            if ge.n_q5==0:continue
            obs=ge.q5_torsions(x)
            mu,k,ln,lw=sc._q5_means,sc._q5_kappas,sc._q5_log_norm,sc._q5_log_w
            delta=obs.unsqueeze(-1)-mu.unsqueeze(0)
            radius=torch.atan2(delta.sin(),delta.cos()).abs()
            off=torch.logsumexp(k[None]*delta.cos()-ln[None]+lw[None],-1)
            on=torch.logsumexp(_torsion_tail_logp(obs,mu,k,ln)+lw[None],-1)
            valid=sc._q5_weights>0
            nearest=(radius/THERMAL_SD['Q5']).masked_fill(~valid[None],float('inf')).min(-1).values
            for i,cost in enumerate((off-on)[0]):
                rows.append(dict(code=code,term='Q5',index=i,extra_cost_nats=float(cost),beyond_z0=bool(nearest[0,i]>2)))
        except Exception as e:
            failures.append(dict(code=code,error=f'{type(e).__name__}: {e}'))
    costs=np.array([r['extra_cost_nats'] for r in rows]);beyond=[r['beyond_z0'] for r in rows]
    summary=dict(parents_requested=len(codes),parents_measured=len({r['code'] for r in rows}),terms=len(rows),failures=failures,
                 median=float(np.median(costs)) if len(costs) else None,p95=float(np.percentile(costs,95)) if len(costs) else None,
                 share_beyond_z0=float(np.mean(beyond)) if len(beyond) else None)
    summary['included']=bool(len(costs) and summary['p95']<=.1 and not failures)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
