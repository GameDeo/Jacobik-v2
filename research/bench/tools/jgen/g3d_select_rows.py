"""Amendment 1: select new-base rows by returned score, without step-0 measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from tools.jgen import inputs
from tools.jgen.g3d_diagnostics import O1_PARENTS, chunk_offsets
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb
from jacobik.core import profile
from jacobik.ik import ik_solver as I
from jacobik.core.env import build_builder_and_inputs


def select_parent(code, stats):
    shard, dataset = inputs.frame_shards()[code]
    mol = inputs.load_mol(code, shard, dataset)
    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        raise ValueError(err)
    state = {}; chunks = []; calls = 0
    original = builder._run_joint_ik
    old_hook = I.JointIKSolver.diagnostic_hook
    def joint(*args, **kwargs):
        nonlocal calls
        state['variant'] = calls
        calls += 1
        return original(*args, **kwargs)
    def hook(solver, t, data):
        if t == 200:
            chunks.append(dict(variant=state['variant'], offset=offsets.offset,
                               digest=hashlib.sha256(b''.join(x.detach().contiguous().numpy().tobytes()
                                                             for x in data['thetas'])).hexdigest()))
    builder._run_joint_ik = joint
    I.JointIKSolver.diagnostic_hook = staticmethod(hook)
    started = time.monotonic()
    try:
        with chunk_offsets() as offsets, xtb.FallbackWatch(raise_on_exit=True):
            result = builder.solve(seed=11, stats=stats, angle_bends=True, device='cpu',
                                   n_conformers=16, n_ik_steps=250)
    finally:
        builder._run_joint_ik = original
        I.JointIKSolver.diagnostic_hook = old_hook
    rows = [dict(rank=i+1, key=[c.embed_idx, int(c.assignment_idx), int(c.trial_idx)], score=float(c.score))
            for i, c in enumerate(sorted(result.conformers, key=lambda c:c.score, reverse=True)[:6])]
    chunks.sort(key=lambda x:(x['variant'],x['offset']))
    return dict(status='ok', rows=rows, shortfall=6-len(rows), switch_chunks=chunks,
                switch_state_digest=hashlib.sha256(json.dumps(chunks,sort_keys=True).encode()).hexdigest(),
                wall_s=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    os.environ.pop('JACOBIK_NUM_WORKERS',None);torch.set_num_threads(1);xtb.require()
    assert profile.snapshot()['fingerprint']=='273f2ccbf892'
    stats=stats_from_db(inputs.DB_PATH)
    result=dict(rule='top 6 delivered rows by returned score; seed 11; CPU default worker layout; 250 steps',
                fingerprint=profile.snapshot()['fingerprint'],seed=11,parents={})
    for code in O1_PARENTS:
        try:
            result['parents'][code]=select_parent(code,stats)
        except Exception as exc:
            import traceback
            result['parents'][code]=dict(status='failed',rows=[],shortfall=6,error=str(exc),traceback=traceback.format_exc())
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(code,result['parents'][code]['status'],len(result['parents'][code]['rows']),flush=True)

if __name__=='__main__': main()
