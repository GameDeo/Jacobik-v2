"""CPU rescoring of the frozen u1 directions; cached MACE curvatures are read-only."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr


def summarize(rows, expected):
    energy=np.concatenate([r[1] for r in rows]) if rows else np.array([])
    score=np.concatenate([r[2] for r in rows]) if rows else np.array([])
    pos=(energy>0)&(score>0)
    ranks=[]
    strata={}
    for where in ('crystal','pool'):
        ratios=[];rr=[]
        for name,e,s,points in rows:
            for point in np.unique(points):
                mask=points==point
                if (point<0)!=(where=='crystal'):continue
                good=mask&(e>0)&(s>0);ratios.extend((e[good]/s[good]).tolist())
                if mask.sum()>5 and np.ptp(e[mask])>0 and np.ptp(s[mask])>0:
                    rr.append(float(spearmanr(e[mask],s[mask]).statistic))
        strata[where]=dict(median_ratio=float(np.median(ratios)) if ratios else None,
                           rank_agreement=float(np.nanmedian(rr)) if rr else None)
        ranks.extend(rr)
    return dict(complete=len(rows)==expected and len(rows)>0,n_cases=len(rows),expected_cases=expected,
        n_directions=len(energy),median_ratio=float(np.median(energy[pos]/score[pos])) if pos.any() else None,
        rank_agreement=float(np.nanmedian(ranks)) if ranks else None,strata=strata,
        definition='energy/kT divided by score curvature, positive directions; rank median within point; '
                   'same frozen dev32+n48 u1 points/directions as forces.py; combined pool+crystal scalar gate')


def scope_curvatures(terms, n, step):
    values=[]
    for score in (terms.sum(1),terms[:,:-1].sum(1)):
        values.append(-(score[1:1+n]+score[1+n:]-2*score[0])/step**2)
    return tuple(values)


def compute(db, flags, out):
    os.environ['JACOBIK_PROFILE']='u1';os.environ['JACOBIK_PROFILE_OVERRIDES']=json.dumps(flags)
    budget=Path(__file__).resolve().parents[2]/'jdev/budget'
    sys.path.insert(0,str(budget))
    import forces
    from pools import setup,set_cases
    setup()
    import torch
    from tools.jdev.cases import get_case
    from tools.jdev.budget.common import stats_from_db
    from .panels import MAIN
    from .arms import db_identity, fk_prior_settings
    from jacobik.core import profile as P
    identity=db_identity(db,False,flags)
    torch.set_num_threads(1)
    import shutil,logging
    from jacobik.builder import engine
    from tools.jgen.embedding_validity import EmbeddingValidity
    if not shutil.which('xtb'):raise RuntimeError('xtb is required for curvature scorer reconstruction')
    logging.getLogger().setLevel(logging.WARNING)
    validity=EmbeddingValidity(engine).install()
    cache=MAIN/'research/archive/data/energy_scale_refit/curvature2/geom'
    files=sorted(cache.glob('etkdg_*.npz'))
    files=[f for f in files if f.stem.split('_')[1] in ('dev32','n48')]
    names={s:dict(set_cases(s)) for s in ('dev32','n48')}
    forces.TERMS=[q for q in forces.TERMS if q!='Q7']+['Q7']
    stats=stats_from_db(db);rows=[];legacy_rows=[];failures=[];out.mkdir(parents=True,exist_ok=False)
    for f in files:
        try:
            _,s,code=f.stem.split('_',2)
            g=np.load(f,allow_pickle=False)
            validity.parent=code
            result=get_case(names[s][code]).make_builder().solve(seed=11,stats=stats,angle_bends=True,n_conformers=1,n_ik_steps=2,device='cpu')
            validity.check(code)
            sc=result.scorer;P0,V,point,asg=g['P0'],g['V'],g['point'],g['asg']
            pidx=np.where(point<0,len(P0)-1,point);parts=[];legacy_parts=[]
            for i,x in enumerate(P0):
                v=V[pidx==i].astype(np.float64);h=forces.H_STEP
                xs=np.concatenate([x[None],x[None]+h*v,x[None]-h*v])
                rg=torch.full((len(xs),),int(asg[i]),dtype=torch.long)
                terms=forces.term_values(sc,xs,rg);n=len(v)
                inclusive,legacy=scope_curvatures(terms,n,h)
                parts.append(inclusive);legacy_parts.append(legacy)
            kS=np.concatenate(parts);legacy_kS=np.concatenate(legacy_parts);kE=g['kE']/forces.KT
            np.savez_compressed(out/f.name,kE=kE,kS=kS,legacy_kS=legacy_kS,point=point)
            rows.append((code,kE,kS,point));legacy_rows.append((code,kE,legacy_kS,point));print(code,'ok',flush=True)
        except Exception as exc:
            failures.append(dict(file=f.name,error=repr(exc)));print(f.name,repr(exc),flush=True)
    validity.close()
    res=summarize(rows,len(files));res.update(scope='Q1-Q7',q7_scorer_in_scope=True,
        fk_prior_scope='out of scope',fk_prior_settings=fk_prior_settings(flags),
        q7_basis='Q1-Q6 plus Cartesian Scorer Q7 on the unchanged cached directions',
        scopes={'Q1-Q6':summarize(legacy_rows,len(files)),'Q1-Q7':summarize(rows,len(files))});res.update(identity,flags=flags,fingerprint=P.snapshot()['fingerprint'],failures=failures)
    (out/'summary.json').write_text(json.dumps(res,indent=2))
    return res


def main():
    p=argparse.ArgumentParser();p.add_argument('--db',type=Path,required=True);p.add_argument('--flags',default='{}')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(compute(a.db,json.loads(a.flags),a.out))

if __name__=='__main__':main()
