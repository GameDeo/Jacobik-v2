"""Run isolated ligand-only hypotheses sequentially, one GPU workload at a time."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
if __name__=='__main__':
    if '--finish' in sys.argv:
        with (ROOT/'data/refit/reports/n2_internal_energy_2026-10-18.log').open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('nci_internal_experiment.py')),'energy'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    trials=[(['--repair-cp','--control'],'cp_off'),(['--repair-cp'],'cp_on'),(['--repair-caps','--control'],'caps_off')]
    if '--isomer-only' in sys.argv or '--finish' in sys.argv:
        trials=[(['--isomer-ref','--control'],'pewbol_isoref_off'),(['--isomer-ref'],'pewbol_isoref_on')]
    if '--both-only' in sys.argv:
        trials=[(['--repair-both','--control'],'both_off'),(['--repair-both'],'both_on')]
    for flags,name in trials:
        log=ROOT/f'data/refit/reports/n2_repair_{name}_2026-10-18.log'
        print('Starting',name,flush=True)
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('nci_fresh_solve.py')),*flags],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('Completed',name,flush=True)
