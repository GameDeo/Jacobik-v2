"""Before/after IA-k CPU output and exact CPU rescore identity on nine units."""
import ast,hashlib,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from tools.jgen.bench.run import _setup_env,_set_finish_flags
_setup_env({})
from tools.jgen.g3e_native_twist import prepare
from tools.jgen.score_repro import CaptureSession
from tools.jdev.budget.common import stats_from_db
from tools.jgen import inputs
from jacobik.ik.finish_phase import FinishPhase
ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3f/identity')


def main():
    torch.set_num_threads(1);ROOT.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','show','5dc37dc:src/jacobik/ik/finish_phase.py'],text=True)
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='FinishPhase')
    fun=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    namespace={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[])),'<pre-G3f-update>','exec'),namespace)
    original=namespace['update'];candidate=FinishPhase.update;stats=stats_from_db(inputs.DB_PATH)
    arms=json.loads(Path('research/campaigns/g/G3F_MANIFEST.json').read_text())['arms'];flags=arms[1]['flags'];units=[];start=time.monotonic()
    for code in ['AJAQAD','AWONOR','CAXTAX']:
        for seed in [11,12,13]:
            result=dict(code=code,seed=seed);payload=[]
            try:
                for label,method in [('before',original),('after',candidate)]:
                    FinishPhase.update=method;_set_finish_flags(flags)
                    builder,ctx,assignments,W,T=prepare(code,seed,stats)
                    with CaptureSession(builder) as cap:
                        solved=builder._solve_for_assignments(ctx,assignments,W,T)
                    cap.attach_conformers(solved.conformers)
                    assert len(cap.captures)==1
                    capture=cap.captures[0];rescored=[]
                    for conf in solved.conformers:
                        row=int(conf.trial_idx)
                        rescored.append(float(capture.score([t[row] for t in capture.thetas_final],row).detach()))
                    arrays=dict(scores=np.asarray([c.score for c in solved.conformers]),rescores=np.asarray(rescored),
                                positions=np.asarray([[a.position for a in c.complex.atoms] for c in solved.conformers]),
                                keys=np.asarray([[int(c.assignment_idx),int(c.trial_idx)] for c in solved.conformers]))
                    for i,t in enumerate(capture.thetas_final):arrays['theta_'+str(i)]=t.numpy()
                    np.savez_compressed(ROOT/f'{code}_{seed}_{label}.npz',**arrays);payload.append(arrays)
                same={k:bool(np.array_equal(payload[0][k],payload[1][k],equal_nan=True)) for k in payload[0]}
                result.update(status='pass' if all(same.values()) else 'failed',identity=same,n_rows=len(payload[0]['scores']))
            except Exception as exc:result.update(status='failed',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
            units.append(result);print(code,seed,result['status'],flush=True)
            (ROOT/'result.json').write_text(json.dumps(dict(units=units,wall_s=time.monotonic()-start,all_pass=len(units)==9 and all(u['status']=='pass' for u in units)),indent=2))
    FinishPhase.update=candidate
    if not all(u['status']=='pass' for u in units):raise SystemExit(1)
if __name__=='__main__':main()
