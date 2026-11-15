"""Freeze G-3a-double-prime CPU filters and spot-check names before cal."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--plots-only',action='store_true');args=ap.parse_args();root=args.root
    diag=json.loads((root/'g3a2_diagnostics.json').read_text())
    if args.plots_only:
        plot_traces(diag,root);return
    src=json.loads((root/'g3a2_filter_structures.json').read_text())
    result={'diagnostic_summary':diag['summary'],'arms':{},'failures':{'diagnostics':diag['failures'],'solves':src['failures']}}
    for arm,control,mode in [('P2','C1',None),('P2prime','C0',None),('P2a','C0','zigzag'),('P2p','C0','plateau')]:
        energy=json.loads((root/f'g3a2_energy_{arm}.json').read_text())
        pairs=[p for p in src['parents'].values() if arm in p and control in p]
        changes=np.array([p[arm]['top1_score']-p[control]['top1_score'] for p in pairs])
        cmed=float(np.median([p[control]['top1_score'] for p in pairs]));amed=float(np.median([p[arm]['top1_score'] for p in pairs]))
        criterion=True if mode is None else diag['summary'][mode]['criterion_fired']>=24
        complete=(len(pairs)==8 and len(energy['parents'])==8 and not energy['failures'] and len([r for p in diag['parents'].values() for r in p['rows']])==48)
        gate=(amed>=cmed and criterion and complete) if arm!='P2' else None
        names=sorted((code for code in energy['parents'] if code!='FEKKAJ'),key=lambda code:(-abs(energy['parents'][code]['paired_top1_change_kj']),code))[:5]
        result['arms'][arm]={'control':control,'paired_parents':len(pairs),'median_control_score':cmed,'median_arm_score':amed,'median_paired_score_change':float(np.median(changes)),
            'score_filter_pass':amed>=cmed,'trigger_filter_pass':criterion,'gate':gate,'complete':complete,'energy_summary':energy['summary'],'preselected_2060_parents':names}
    (root/'g3a2_filter_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    plot_traces(diag,root)


def plot_traces(diag,root):
    # Fixed examples, selected by identity rather than trigger outcome.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=[(code,diag['parents'][code]['rows'][0]) for code in ('HEDJOP','FEKKAJ','USAPOU')]
    for mode in ('zigzag','plateau'):
        fig,axes=plt.subplots(3,3 if mode=='zigzag' else 2,figsize=(12,8),squeeze=False)
        for i,(code,row) in enumerate(examples):
            trace=row['arms'][mode]['trace'];ts=[r['t'] for r in trace]
            fields=['score','z','lr'] if mode=='zigzag' else ['score','lr']
            for ax,field in zip(axes[i],fields):
                ax.plot(ts,[r[field] for r in trace]);ax.set_xlabel('Continuation step t');ax.set_ylabel(field);ax.set_title(f'{code}, key {row["key"]}')
                if field=='score':
                    ax.plot(ts,[r['best'] for r in trace],ls='--',label='best');ax.legend(fontsize=7)
                if field=='z':ax.axhline(.5,color='grey',ls='--')
        fig.tight_layout();fig.savefig(root/f'g3a2_{mode}_traces.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
