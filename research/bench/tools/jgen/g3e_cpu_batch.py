"""Run fixed CPU unit groups; native per-unit checkpoints are reused."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['generate','paths'],required=True);args=parser.parse_args()
    root=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3e/native_twist')
    groups=['1,4,7','2,5,8','3,6,9'];jobs=[];start=time.monotonic()
    for i,indices in enumerate(groups):
        log=open(root/f'{args.stage}_{i}.log','a')
        proc=subprocess.Popen([sys.executable,'-u','-m','tools.jgen.g3e_native_twist','--stage',args.stage,'--indices',indices],stdout=log,stderr=subprocess.STDOUT)
        jobs.append(proc);print(args.stage,i,proc.pid,flush=True)
    codes=[p.wait() for p in jobs]
    (root/f'{args.stage}_batch.json').write_text(json.dumps(dict(exit_codes=codes,wall_s=time.monotonic()-start),indent=2))

if __name__=='__main__':main()
