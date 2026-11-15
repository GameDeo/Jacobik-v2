"""Watch one owned cal job; fetch/count outputs before immediate VM shutdown."""
import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter
from types import SimpleNamespace


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=('jgBg3a2a','jgBg3a2b'),required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--remote',required=True)
    ap.add_argument('--local',type=Path,required=True);ap.add_argument('--expected',type=int,required=True)
    ap.add_argument('--journal',type=Path,required=True)
    ap.add_argument('--queue-contract',type=Path,help='After A fetch/count, run pending B on this same VM when second allocation is refused')
    args=ap.parse_args()
    spec=importlib.util.spec_from_file_location('cj',Path('research/gpuwork/colab/cj.py'));cj=importlib.util.module_from_spec(spec);spec.loader.exec_module(cj)
    args.local.mkdir(parents=True,exist_ok=True)
    def note(event,**kw):
        item={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'name':args.name,'event':event,**kw}
        with args.journal.open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    note('watch_started',tag=args.tag,expected=args.expected)
    marker=None
    while marker is None:
        try:
            log=cj.colab(['exec','-s',args.name],timeout=45,retries=1,check=False,
                stdin=f"from pathlib import Path\np=Path('/content/jobs/{args.tag}.log')\nprint(p.read_text()[-6000:] if p.exists() else 'LOG_PENDING')\n") or ''
            (args.local/f'{args.tag}_latest.log').write_text(log)
            found=re.search(r'CJ_EXIT (\d+)',log)
            note('poll',done_in_tail=len(re.findall(r'\] done in ',log)),exit_code=int(found[1]) if found else None,last_lines=log.splitlines()[-3:])
            if found:marker=int(found[1]);break
        except Exception as exc:note('poll_error',reason=str(exc)[:500])
        time.sleep(45)
    # Remote computation has ended. Preserve all available output, including failures.
    fetched=False
    while not fetched:
        try:
            cj.cmd_get(SimpleNamespace(name=args.name,remote=f'/content/jobs/{args.tag}.log',local_dir=str(args.local)))
            cj.cmd_get(SimpleNamespace(name=args.name,remote=args.remote,local_dir=str(args.local)))
            fetched=True
        except Exception as exc:
            note('fetch_error_retrying',reason=str(exc)[:500]);time.sleep(20)
    root=args.local/Path(args.remote).name;files=list(root.rglob('records/*.json'))
    statuses=Counter();arms=Counter()
    for p in files:
        r=json.loads(p.read_text());statuses[r['status']]+=1;arms[r['arm']]+=1
    note('fetched_and_counted',files=len(files),expected=args.expected,count_matches=len(files)==args.expected,statuses=dict(statuses),arms=dict(arms),exit_code=marker)
    if args.queue_contract is not None and marker==0 and len(files)==args.expected:
        assert args.name=='jgBg3a2a' and args.tag=='g3a2_cal_A'
        contract=json.loads(args.queue_contract.read_text())
        parents=contract['cal']['partitions']['B'];assert len(parents)==60
        cj.cmd_run(SimpleNamespace(name=args.name,tag='g3a2_cal_B',
            env=['OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1'],
            cmd=['python','-m','tools.jgen.run_g3a2_cal','--parents',','.join(parents),
                 '--arms','C0,C1,P2','--out','/content/out/g3a2_B']))
        note('queued_B_on_same_vm',reason='second L4 allocation refused',parents=len(parents))
        subprocess.run([sys.executable,'-m','tools.jgen.watch_g3a2_colab','--name',args.name,
            '--tag','g3a2_cal_B','--remote','/content/out/g3a2_B','--local',str(args.local),
            '--expected','540','--journal',str(args.journal)],check=True)
        note('A_and_B_complete_child_confirmed_down')
        return
    # Count mismatches are failures, not a reason to leave a completed/failed VM billing.
    stopped=False
    while not stopped:
        try:cj.cmd_down(SimpleNamespace(name=args.name));stopped=True
        except Exception as exc:note('down_error_retrying',reason=str(exc)[:500]);time.sleep(20)
    note('down_confirmed',files=len(files),expected=args.expected,exit_code=marker)

if __name__=='__main__':main()
