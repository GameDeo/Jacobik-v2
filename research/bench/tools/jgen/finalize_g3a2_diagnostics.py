"""Canonicalize validated continuous-state diagnostics, retaining prior trials."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    src=json.loads((root/'g3a2_continuous_diagnostics.json').read_text())
    assert not src['failures'] and len(src['parents'])==8
    assert all(s['rows']==48 and s['all_first200_trajectories_identical'] for s in src['summary'].values())
    for suffix in ('json','log'):
        old=root/f'g3a2_diagnostics.{suffix}';backup=root/f'g3a2_mixed_state_superseded.{suffix}'
        if old.exists() and not backup.exists():shutil.copy2(old,backup)
    for mode in ('zigzag','plateau'):
        old=root/f'g3a2_{mode}_traces.png';backup=root/f'g3a2_{mode}_mixed_state_superseded.png'
        if old.exists() and not backup.exists():shutil.copy2(old,backup)
    result={'parents':{},'failures':{},'summary':src['summary'],'calibration_trials':src['calibration_trials'],
            'source':'g3a2_continuous_diagnostics.json','definition':src['diagnostic'],'dtype':src['dtype']}
    for code,p in src['parents'].items():
        rows=[]
        for control in p['arms']['control']['rows']:
            rec={'key':control['key'],'old_delivered_score':control['old_delivered_score'],'first200_control_sha256':control['first200_trajectory_sha256'],'arms':{}}
            for mode in ('zigzag','plateau'):
                row=next(r for r in p['arms'][mode]['rows'] if r['key']==control['key'])
                assert row['first200_trajectory_sha256']==rec['first200_control_sha256']
                rec['arms'][mode]=row
            rows.append(rec)
        result['parents'][code]={'rows':rows,'wall_s':sum(a['wall_s'] for a in p['arms'].values())}
    (root/'g3a2_diagnostics.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
