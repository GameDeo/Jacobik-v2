"""Summarize verified native CPU gradients and prescribed score paths."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3e/native_twist')
    protocol=json.loads(Path('research/campaigns/g/G3E_CPU_TWIST_PROTOCOL.json').read_text())
    summary=dict(units=[],artifacts={},original_GPU_inversion_failures_retained=True,owner_CPU_regeneration=True)
    for u in protocol['units']:
        d=root/f"{u['code']}_{u['seed']}";g=json.loads((d/'gradients.json').read_text());p=json.loads((d/'path.json').read_text())
        assert g['status']=='ok'
        for r in g['observations'].values():assert r['reconstruction_rmsd_A']<=1e-5 and abs(r['score_error'])<=1e-4
        row=dict(unit=u,gradients=g['observations'],path_status=p['status'])
        if p['status']=='ok':
            row.update(barrier_nats=p['barrier_nats'],barrier_terms_nats=p['barrier_terms_nats'],
                       path_minimum_index=p['path_minimum_index'],interior_minimum=p['interior_minimum'],
                       endpoint_score_change_nats=p['endpoint_score_change_nats'],mace_energy_kj=p['mace_energy_kj'])
        else:
            row.update(path_failure=p['reason'],reference_diagnostic=json.loads((d/'reference_diagnostic.json').read_text()))
            row['mace_endpoint_energy_kj']=json.loads((d/'endpoint_mace.json').read_text())['energy_kj']
        summary['units'].append(row)
        for f in d.glob('*'):
            if f.suffix in ['.json','.npz']:summary['artifacts'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    successful=[r for r in summary['units'] if r['path_status']=='ok'];summary['paths_complete']=len(successful);summary['gradients_complete']=len(summary['units'])
    summary['converged_by_remaining_gain_criterion']=sum(.5*r['gradients']['twisted']['thermal_gradient_norm_nats']**2<.1 for r in summary['units'])
    out=Path('research/campaigns/g');(out/'G3E_TWIST_RESULT.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(4,2,figsize=(10,11),constrained_layout=True)
    for ax,row in zip(axes.flat,successful):
        u=row['unit'];p=json.loads((root/f"{u['code']}_{u['seed']}"/'path.json').read_text());t=[v['t'] for v in p['path']];s=np.array([v['score_nats'] for v in p['path']]);s-=s[0]
        ax.plot(t,s,'o-',markersize=3,color='#224f83');j=p['path_minimum_index'];ax.axvline(t[j],ls='--',color='#b05c3c',alpha=.6)
        ax.set_title(f"{u['code']} seed {u['seed']} — barrier {p['barrier_nats']:.1f} nats",fontsize=10)
        ax.set_xlabel('Linear DOF path fraction');ax.set_ylabel('Score − twisted score (nats)');ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(successful):]:
        ax.axis('off');ax.text(.05,.75,'USAPOU seeds 11–13:\nDifferent sampled FK references.\nCommon-reference endpoint check failed.\nNo path or barrier substituted.',va='top',fontsize=10)
    fig.suptitle('Prescribed fixed-key CPU score paths\nStraight DOF interpolation; higher score is better',fontsize=13)
    fig.savefig(out/'G3E_TWIST_SCORE_PATHS.png',dpi=160);plt.close(fig)
    lines=['# R2s twist diagnostic — regenerated CPU states, 2026-11-02','',
           '**Result:** native gradients verified for 10/10 frozen units; complete',
           '21-point paths for 7/10. Six twisted states meet the solver’s predicted',
           'remaining-gain threshold; USAPOU’s three and OMAGAK do not. Every valid',
           'path has an interior score minimum, but these straight-path barriers',
           'do not establish a full-dimensional local maximum. The diagnostic does',
           'not support calling all twisted rows spurious score maxima.','',
           'The three USAPOU path failures are retained: native endpoints use',
           'different sampled macrocycle references. Direct interpolation of their',
           'captured DOFs under one reference misses the endpoint. No alternative',
           'partner, optimized path, score change or new arm was substituted.','',
           '## Data and verification','',
           'Original GPU exports omitted native DOFs. Coordinate-only recovery failed',
           'the score/state checks and remains recorded in shared g3e/twist/result.json;',
           'none of its gradients or barriers is accepted. The owner explicitly',
           'approved CPU regeneration. These CPU rows are separately labelled and',
           'do not replace either GPU clean24 result. Same ten parent/seed units;',
           'native C1/R2s reference pools, one CPU worker, no culling, 250 steps.',
           'All selection rules were frozen before native results; selected native',
           'IDs/state hashes were committed before remaining paths. No case replaced.','',
           'Captured FK references, returned DOFs, assignments and Q3 commitments are',
           'persisted. Every reported native role reproduces geometry <=1e-5 Å and',
           'returned score <=1e-4 native units. All paths keep the twisted row’s',
           'starting-shell Q2 pair keys and native Q3 commitment fixed. No current-',
           'shape re-keying was implemented. Native score units and summed nats are',
           'both preserved: nats = native score × scorer._agg_norm(). Gradients use',
           'the same conversion; thermal norms use the resolved per-DOF widths.','',
           '## Native DOF gradients','',
           '| Unit | Z | Raw native ∥g∥, twisted / top-1 | Summed-nat ∥g∥, twisted / top-1 | Thermal ∥gσ∥, twisted / top-1 | ½∥gσ∥² <0.1 |',
           '|---|---:|---|---|---|---|']
    for r in summary['units']:
        u=r['unit'];g=r['gradients'];a=g['twisted'];b=g['top1']
        lines.append(f"| {u['code']} s{u['seed']} | {a['normalization']:.0f} | {a['gradient_norm_native']:.4f} / {b['gradient_norm_native']:.4f} | {a['gradient_norm_nats']:.4f} / {b['gradient_norm_nats']:.4f} | {a['thermal_gradient_norm_nats']:.4f} / {b['thermal_gradient_norm_nats']:.4f} | {'yes' if .5*a['thermal_gradient_norm_nats']**2<.1 else 'no'} |")
    lines += ['', 'HEDJOP’s selected twisted row is also its regenerated R2s unit top-1 in',
              'all three seeds; the equal norms are the same point, not independent',
              'samples. The remaining-gain flag is descriptive, using the existing',
              'criterion; finite gradients and one-dimensional paths are not Hessian',
              'certificates of local maximality.','',
              '## Prescribed score barriers and CPU MACE','',
              'Barrier = twisted score minus the minimum of 21 equally spaced linear',
              'DOF path points. Equivalent angular branches and rigid global alignment',
              'are fixed before interpolation; bend DOFs keep native values. All valid',
              'minima are interior. MACE polar-1-s float64, CPU only, no relaxation.',
              'Energy differences below use the same CPU model for all points.','',
              '| Unit | Barrier (nats) | Untwisted − twisted score (nats) | E(twisted) − E(untwisted), kJ/mol | E(path minimum) − E(twisted), kJ/mol | Largest score-drop terms (nats) |',
              '|---|---:|---:|---:|---:|---|']
    for r in summary['units']:
        u=r['unit'];label=f"{u['code']} s{u['seed']}"
        if r['path_status']=='ok':
            e=r['mace_energy_kj'];terms=sorted(r['barrier_terms_nats'].items(),key=lambda x:-x[1])[:3]
            lines.append(f"| {label} | {r['barrier_nats']:.3f} | {r['endpoint_score_change_nats']:.3f} | {e[0]-e[1]:.3f} | {e[2]-e[0]:.3f} | {', '.join(f'{k}: {v:.2f}' for k,v in terms)} |")
        else:
            e=r['mace_endpoint_energy_kj'];lines.append(f"| {label} | unavailable | unavailable | {e[0]-e[1]:.3f} | unavailable | common-reference endpoint failure |")
    favored=sum((r['mace_energy_kj'][0]-r['mace_energy_kj'][1] if r['path_status']=='ok' else r['mace_endpoint_energy_kj'][0]-r['mace_endpoint_energy_kj'][1])<0 for r in summary['units'])
    lines += ['',f'CPU MACE favors the selected twisted endpoint over its proximity-selected untwisted partner in {favored}/10 pairs.',
              'Partners were selected by geometry/assignment, not energy. These pairs',
              'do not establish that the untwisted endpoint is the better physical',
              'basin; they also do not establish global quality of the twisted pool.',
              'Large MACE increases and Q1/Q6 drops at some linear-path minima show',
              'that the prescribed interpolation passes through strained/clashing',
              'structures. The path is not a minimum-barrier transition path.','',
              'Q1/Q6 dominate several barriers, Q5 dominates HEDJOP, and Q3 dominates',
              'AZUSIA seed 13. Q2 is not the sole source. Full Q1–Q7, bend-prior and',
              'seam terms at all 21 points are in each raw path.json. Q7 is zero when',
              'the Cartesian Q7 term is inactive; the native DOF bend prior is listed',
              'separately and included in totals.','',
              '![Fixed-key score paths](g/G3E_TWIST_SCORE_PATHS.png)','',
              '## USAPOU reference limitation','',
              '| Seed | Own-reference aligned RMSD (Å) | Twisted-reference endpoint RMSD (Å) | Macrocycle DOFs |',
              '|---|---:|---:|---|']
    for r in summary['units']:
        if r['path_status']=='ok':continue
        x=r['reference_diagnostic'];lines.append(f"| {r['unit']['seed']} | {x['own_reference_aligned_RMSD_A']:.3g} | {x['common_reference_RMSD_A']:.6f} | {x['macro_joints']} |")
    lines += ['', 'The own-reference rotation checks are at numerical precision, while the',
              'common-reference endpoint misses by about 2 Å. USAPOU has row-dependent',
              'sampled references and five macrocycle torsions. This instrument did not',
              'construct a common-reference path for the frozen nearest partners. It',
              'does not prove that no such path exists. Gradients and both endpoint',
              'MACE energies are still reported for all three units.','',
              '## Interpretation and scope','',
              'Six rows show small predicted remaining gain plus score barriers along',
              'the prescribed fixed-key line: evidence of nearly settled score basins',
              'under the starting-shell model. A full-dimensional local-maximum claim',
              'would require more than this path. Four rows have larger gradients and',
              'cannot be classified as settled maxima by this criterion. The selected',
              'endpoint MACE comparisons often favor the twisted state, so “spurious”',
              'is not established by these pairs. No score change follows from this',
              'descriptive diagnostic. R2s remains built and off, reserved for the',
              'hypothesis layer; current-shape pair-key adaptation remains future work.','',
              'G3E_TWIST_RESULT.json contains all native gradient observations, path',
              'summaries, exact term drops and raw-state SHA256s. Full native pools,',
              'DOFs/references, 21-point series, MACE values and verification failures',
              'remain in shared research/data/jgen/campaign/B/g3e/. No GPU was used by',
              'this CPU diagnostic; Colab spend zero.']
    Path('research/campaigns/G3E_R2S_TWIST_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('gradients',summary['gradients_complete'],'paths',summary['paths_complete'],'criterion',summary['converged_by_remaining_gain_criterion'],'MACE favors twist',favored)

if __name__=='__main__':main()
