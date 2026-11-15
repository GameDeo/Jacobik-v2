"""One MACE pass for all saved filter arms, deduplicating identical geometries."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
from tools.runkit import mace,resources

PAIRS={'P2':'C1','P2prime':'C0','P2a':'C0','P2p':'C0'}


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--structures',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    src=json.loads(args.structures.read_text())
    result=json.loads(args.out.read_text()) if args.resume and args.out.exists() else {'parents':{},'failures':{},'device':'cuda','dtype':'float64','source':str(args.structures)}
    with resources.gpu_lock('CodexB',timeout=900):
        ev=mace.Evaluator('cuda',dtype='float64',threads=1)
        assert result.get('model_sha256',ev.model_sha256)==ev.model_sha256
        result['model_sha256']=ev.model_sha256
        for code,parent in src['parents'].items():
            if code in result['parents']:
                assert result['parents'][code]['source_digest']==digest(parent),'cached parent differs'
                continue
            start=time.monotonic()
            try:
                reference=parent['C0']['reference'];q,spin=reference['charge'],reference['spin']
                assert all(p['reference']==reference for p in parent.values())
                structures=[];seen={}
                def add(symbols,X):
                    a=np.asarray(X,dtype=np.float64);assert a.shape==(len(symbols),3) and np.isfinite(a).all()
                    key=(tuple(symbols),a.tobytes())
                    if key not in seen:seen[key]=len(structures);structures.append((symbols,a,q,spin))
                    return seen[key]
                refidx=add(reference['symbols'],reference['X_A']);indices={}
                for arm,p in parent.items():
                    indices[arm]=[]
                    for row in p['conformers']:
                        assert row['symbols_order']=='builder/conf.complex.atoms'
                        indices[arm].append(add(row['symbols'],row['X_A']))
                energies=ev.energies_kj(structures);assert np.isfinite(energies).all()
                rec={'source_digest':digest(parent),'reference_energy_kj':float(energies[refidx]),'unique_structures':len(structures),'arms':{}}
                for arm,ids in indices.items():
                    p=parent[arm];dE=energies[ids]-energies[refidx];scores=np.asarray([r['score'] for r in p['conformers']]);top=int(scores.argmax());worst=int(dE.argmax())
                    rec['arms'][arm]={'top1_score':float(scores[top]),'top1_delta_E_unrel_kj':float(dE[top]),'floor_delta_E_unrel_kj':float(dE.min()),'all_delta_E_unrel_kj':dE.tolist(),
                        'energies_kj':energies[ids].tolist(),'scores':scores.tolist(),'n_rows':len(ids),'obscene_count':int((dE>200).sum()),'obscene_rate':float((dE>200).mean()),
                        'worst_delta_E_unrel_kj':float(dE[worst]),'worst_row_key':p['conformers'][worst]['key'],'solve_s':p['wall_s']}
                rec['energy_wall_s']=time.monotonic()-start;result['parents'][code]=rec
            except Exception:result['failures'][code]=traceback.format_exc()
            args.out.write_text(json.dumps(result,indent=2));print(code,result['parents'].get(code,{}).get('unique_structures'),round(time.monotonic()-start,2),result['failures'].get(code,''),flush=True)
    for arm,control in PAIRS.items():
        paired={};rng=np.random.default_rng(11)
        for code,p in result['parents'].items():
            c,a=p['arms'][control],p['arms'][arm]
            paired[code]={control:c,arm:a,'paired_top1_change_kj':a['top1_delta_E_unrel_kj']-c['top1_delta_E_unrel_kj'],'paired_floor_change_kj':a['floor_delta_E_unrel_kj']-c['floor_delta_E_unrel_kj'],'paired_score_change':a['top1_score']-c['top1_score']}
        summary={'paired_parents':len(paired),'complete':len(paired)==8 and not result['failures']}
        if paired:
            for metric in ('top1','floor'):
                d=np.array([p[f'paired_{metric}_change_kj'] for p in paired.values()]);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
                summary[metric]={'mean_paired_change_kj':float(d.mean()),'median_paired_change_kj':float(np.median(d)),'mean_ci95_kj':np.percentile(boot,[2.5,97.5]).tolist(),'parents_improved':int((d<0).sum())}
        dest=args.out.parent/f'g3a2_energy_{arm}.json';dest.write_text(json.dumps({'parents':paired,'failures':result['failures'],'summary':summary,'source':str(args.out),'model_sha256':result['model_sha256']},indent=2))

if __name__=='__main__':main()
