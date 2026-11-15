"""Once-only C-2b launcher and long-interval load journal; never retries a run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def snapshot(root, label, start):
    import psutil
    processes=[]
    for p in psutil.process_iter():
        try:p.cpu_percent(None);processes.append(p)
        except psutil.Error:pass
    cpu=psutil.cpu_percent(interval=1)
    rows=[]
    for p in processes:
        try:
            rows.append(dict(pid=p.pid,name=p.name(),cpu_percent=p.cpu_percent(None),
                rss_bytes=p.memory_info().rss,threads=p.num_threads(),command=' '.join(p.cmdline())[:1000]))
        except psutil.Error:pass
    now=datetime.now(timezone.utc)
    data=dict(utc=now.isoformat(),elapsed_s=time.monotonic()-start,label=label,
        load_average=os.getloadavg(),cpu_percent=cpu,ram=psutil.virtual_memory()._asdict(),
        swap=psutil.swap_memory()._asdict(),top_cpu=sorted(rows,key=lambda r:r['cpu_percent'],reverse=True)[:15])
    tag=now.strftime('%Y%m%dT%H%M%SZ')+'_'+label
    data['nvidia_smi_file']=str(root/'load'/f'{tag}_nvidia-smi.txt')
    result=subprocess.run(['nvidia-smi'],text=True,capture_output=True)
    Path(data['nvidia_smi_file']).write_text(result.stdout+result.stderr)
    data['nvidia_smi_returncode']=result.returncode
    (root/'load'/f'{tag}.json').write_text(json.dumps(data,indent=2)+'\n')
    with (root/'load_context.jsonl').open('a') as out:out.write(json.dumps(data)+'\n')
    print(f"{label} {data['utc']} load={data['load_average']} available_GiB={data['ram']['available']/2**30:.2f}",flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--interval-seconds',type=float,default=1200.)
    a=ap.parse_args()
    if a.interval_seconds<=0:ap.error('interval must be positive')
    # Atomic nonexistence check: there is deliberately no resume/retry mode.
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'load').mkdir()
    start=time.monotonic()
    command=[sys.executable,'-m','tools.jgen.c2.driver','--out',str(a.out),
        '--arms','C,K,P,KU,S1','--seeds','21,22,23','--confirmation-after-audit']
    launch=dict(command=command,monitor_pid=os.getpid(),interval_seconds=a.interval_seconds,
        launch_utc=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        authorization='Owner cleared confirmation; audit C-2b addendum accepted freeze 5a19ef8',
        retries=0)
    (a.out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    snapshot(a.out,'start',start)
    with (a.out/'driver.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        launch['driver_pid']=proc.pid
        (a.out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        while True:
            try:
                code=proc.wait(timeout=a.interval_seconds);break
            except subprocess.TimeoutExpired:snapshot(a.out,'periodic',start)
    snapshot(a.out,'end',start)
    result=dict(returncode=code,driver_pid=proc.pid,elapsed_s=time.monotonic()-start,
        finished_utc=datetime.now(timezone.utc).isoformat(),retries=0)
    (a.out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Driver completed:',result,flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
