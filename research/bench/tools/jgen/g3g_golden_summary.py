"""Compare archived/new CPU goldens; per-case unrelaxed top-1 MACE regression record."""
import hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
OUT=ROOT/'research/data/jgen/campaign/B/g3g'


def main():
    from tools.jdev import golden
    from tools.jdev.cases import get_case
    sys.path.insert(0,str(Path('research/bench/tools/jdev/budget').resolve()))
    from pools import set_cases,setup
    setup()
    from jacobik.core.energy import MACEPolarEvaluator,EV_TO_KJ_MOL
    from jacobik.core.env import MACE_MODEL_PATH
    from tools.runkit import xtb
    evaluator=MACEPolarEvaluator(model='polar-1-s',device='cpu',dtype='float64',mode='energy-only',threads=1,model_path=str(MACE_MODEL_PATH)).load()
    evaluator._batch_atoms=300
    names={c:n for s in ['dev32','n48'] for c,n in set_cases(s)};result=dict(profiles={},energies={},model_sha256=hashlib.sha256(Path(MACE_MODEL_PATH).read_bytes()).hexdigest());started=time.monotonic()
    def checkpoint():
        (OUT/'golden_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        Path('research/campaigns/g/G3G_GOLDEN_RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for profile in ['production','u1']:
        old=np.load(ROOT/f'research/archive/data/2026-11_golden_history/golden_273f2ccbf892/{profile}.npz')
        new=np.load(ROOT/f'research/data/golden/{profile}.npz');assert sorted(old.files)==sorted(new.files)
        exact=[];beyond=[]
        for key in old.files:
            a,b=old[key],new[key]
            if a.shape!=b.shape or not np.array_equal(a,b,equal_nan=True):exact.append(key)
            tol=golden.TOL_X if key.endswith('__X') else golden.TOL_SCORE
            if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=tol,equal_nan=True):beyond.append(key)
        unit=lambda key:'__'.join(key.split('__')[:-1])
        parents=lambda keys:sorted({key.split('__')[1] for key in keys})
        result['profiles'][profile]=dict(arrays=len(old.files),arrays_changed_exact=len(exact),arrays_changed_above_check_tolerance=len(beyond),
            changed_arrays=exact,changed_above_tolerance=beyond,total_cases=len({unit(k) for k in old.files}),
            cases_changed_exact=len({unit(k) for k in exact}),cases_changed_above_check_tolerance=len({unit(k) for k in beyond}),
            parents_changed=parents(exact),solve_cases_changed=sorted({k.split('__')[1] for k in exact if k.startswith('solve__')}))
        checkpoint()
        result['energies'][profile]=[]
        for code in golden.SOLVE:
            row=dict(code=code,seed=11)
            try:
                case=get_case(names.get(code,code));builder=case.make_builder()
                with xtb.FallbackWatch(raise_on_exit=True):mol,_,_=builder._build_shell(False,False)
                symbols=[a.symbol for a in mol.atoms]
                assert sorted(symbols)==sorted(case.ref_symbols)
                oldx=old[f'solve__{code}__X'];newx=new[f'solve__{code}__X'];olds=old[f'solve__{code}__score'];news=new[f'solve__{code}__score']
                io,inew=int(np.argmax(olds)),int(np.argmax(news))
                assert oldx[io].shape==newx[inew].shape==(len(symbols),3)
                energies=evaluator.energies([(symbols,oldx[io],case.charge,case.mult),(symbols,newx[inew],case.charge,case.mult),(case.ref_symbols,case.ref_X,case.charge,case.mult)])
                eo,en,er=map(lambda e:float(e)*EV_TO_KJ_MOL,energies)
                row.update(status='ok',old_top1_score=float(olds[io]),new_top1_score=float(news[inew]),old_dE_kj=eo-er,new_dE_kj=en-er,delta_dE_kj=en-eo,
                           old_pool_rows=len(olds),new_pool_rows=len(news),atom_order='builder shell order for stored solver coordinates; crystal order for reference',charge=case.charge,multiplicity=case.mult)
            except Exception as exc:row.update(status='failed',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
            result['energies'][profile].append(row);checkpoint();print(profile,code,row['status'],row.get('delta_dE_kj'),flush=True)
    result['wall_s']=time.monotonic()-started;checkpoint()
if __name__=='__main__':main()
