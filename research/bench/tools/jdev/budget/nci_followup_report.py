"""Validate, export, and summarize the targeted NCI diagnosis experiments."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO
from pools import setup

REPORTS=REPO/'data/refit/reports'
DIRS=dict(off='n2_fresh_control_2026-10-18',on='n2_fresh_solve_2026-10-18',
    cp_off='n2_repair_cp_off_2026-10-18_v2',cp_on='n2_repair_cp_on_2026-10-18_v2',
    caps_off='n2_repair_caps_off_2026-10-18_v2',both_off='n2_repair_both_off_2026-10-18_v2',
    both_on='n2_repair_both_on_2026-10-18_v2',isoref_off='n2_pewbol_isoref_off_2026-10-18',
    isoref_on='n2_pewbol_isoref_on_2026-10-18')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])


def run():
    import pandas as pd
    from scipy.stats import spearmanr
    from tools.jdev.env import tree_hash
    from research.archive.code.refit.nci_refit import digest
    from nci_targeted_rescore import CANDIDATE
    rows=[];manifests={};exports=[]
    for arm,name in DIRS.items():
        folder=REPORTS/name;manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['status']=='complete',(arm,manifest['status'])
        assert manifest['config']['fingerprint']['src']==tree_hash(),arm
        assert manifest['config']['db_sha256']==digest(CANDIDATE),arm
        manifests[arm]=manifest
        files=sorted(folder.glob('*_s*.npz'));assert len(files)==manifest['pools'],arm
        for f in files:
            code,seed=f.stem.split('_s');seed=int(seed)
            with np.load(f) as z:
                e=np.array(json.loads(f.with_suffix('.mace64.json').read_text())['dE'])
                assert np.isfinite(e).all() and np.isfinite(z['X']).all()
                assert len(e)==len(z['X'])==len(z['score'])
                p=json.loads(str(z['profile']))['fields'];assert p['nci']==arm.endswith('on')
                assert p['oxidation_model']=='training' and not p['bvs']
                i=int(z['score'].argmax());lo=e<=e.min()+50
                rho=float(spearmanr(z['score'][lo],-e[lo]).statistic) if lo.sum()>=5 else None
                row=dict(arm=arm,case=code,seed=seed,pick=float(e[i]),floor=float(e.min()),regret=float(e[i]-e.min()),
                    top1=bool(e[i]<=e.min()+4.184),rho_lo=rho,invalid_share=float(np.mean(e>1000)),n=len(e),
                    assignment_ids=len(np.unique(z['asg'])),score_error=float(z['score_error']))
                rows.append(row)
                path=REPO/'data/inspect'/code/f'followup_{arm}_s{seed}.xyz'
                path.parent.mkdir(exist_ok=True)
                path.write_text(f"{len(z['symbols'])}\n{code} {arm} seed {seed}; MACE64 dE={e[i]:.6f} kJ/mol; no complex relaxation\n"+
                    ''.join(f'{s} {a:.8f} {b:.8f} {c:.8f}\n' for s,(a,b,c) in zip(z['symbols'],z['X'][i])),encoding='utf-8')
                exports.append(str(path.relative_to(REPO)))
    df=pd.DataFrame(rows)
    df.to_csv(REPORTS/'n2_followup_metrics_2026-10-18.csv',index=False)
    by_case={}
    for code in df.case.unique():
        by_case[code]={arm:{k:float(g[k].mean()) for k in ('pick','floor','regret','top1','invalid_share','n')} | dict(rho_lo=float(g.rho_lo.mean()) if g.rho_lo.notna().any() else None)
                       for arm,g in df[df.case==code].groupby('arm')}
    fit=json.loads((CANDIDATE.parent/'fit_summary.json').read_text())
    base=REPO/'data/refit/final/constraints_2026-10-17.json'
    assert digest(base)==fit['base_sha256'] and digest(CANDIDATE)==fit['candidate_sha256']
    pose=json.loads((REPORTS/'n2_pose_audit_2026-10-18.json').read_text())
    curvature=json.loads((REPORTS/'n2_balance_audit_2026-10-18.json').read_text())
    result=dict(scope='Selected diagnostic cohort, no dev80 estimate or adoption. One fresh control per seed, not an estimate of solver-noise variance.',
                checks=dict(completed_pools=len(rows),finite_pools=len(rows),score_max_error=max(r['score_error'] for r in rows),
                    curvature_central_term_max_error=max(v for c in curvature['checks'] for v in c['central_term_errors'].values()),
                    base_db_unchanged=True,candidate_db_unchanged=True,source_hash=tree_hash()),
                cases=by_case,manifests=manifests,exports=exports,
                basin_share='not measured; requires separate relaxed-basin measurements')
    (REPORTS/'n2_followup_summary_2026-10-18.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    comparison=[]
    for code in ('XEPKAF','PEWBOL','JEFJUZ','FETSUU','BULGAT','CBTYCO','YIYXEJ'):
        a,b=by_case[code]['off'],by_case[code]['on']
        comparison.append([code,*[f'{v:.2f}' for v in (a['pick'],b['pick'],b['pick']-a['pick'],a['floor'],b['floor'],a['regret'],b['regret'])]])
    repair=[]
    for code in ('JEFJUZ','XEPKAF'):
        for arm in ('off','on','cp_off','cp_on','caps_off','both_off','both_on'):
            if arm not in by_case[code]:continue
            g=df[(df.case==code)&(df.arm==arm)].sort_values('seed')
            repair.append([code,arm,*[f'{x:.2f}' for x in g.pick],f'{g.pick.mean():.2f}',f'{g.floor.mean():.2f}'])
    curve_rows=[]
    for code,c in curvature['summary'].items():
        curve_rows.append([code,f"{c['off']['median_ratio']:.3f}",f"{c['on']['median_ratio']:.3f}",
            f"{c['off']['stable_median_ratio']:.3f}",f"{c['on']['stable_median_ratio']:.3f}",
            f"{c['off']['energy_up_score_unstable']}/{c['off']['directions']}",f"{c['on']['energy_up_score_unstable']}/{c['on']['directions']}"])
    isoref=[]
    for r in pose['PEWBOL']['pools']:
        if r['arm'].startswith('isoref'):
            p=r['picks']['pick'];isoref.append([r['arm'],r['seed'],f"{p['energy']:.2f}",f"{p['N_M_N']:.1f}",f"{p['N_S_vector_angle']:.1f}",p['hbond']])
    text=f'''# NCI follow-up: structures, curvature, and full solves (2026-10-18)

NCI remains off. The targeted full solves confirm adverse effects in PEWBOL, FETSUU and YIYXEJ and a modest loss
in XEPKAF. JEFJUZ has a separately identifiable ligand-embedding problem. A ligand-only repair is tested below;
it is not a production default or a correction to NCI weights.

## Fresh NCI generation, matched controls

Seven viewer cases, seeds 11/12/13, complete production-budget solves with NCI on, plus matched NCI-off solves.
All comparisons use u1 with training OS, candidate DB whose non-NCI entries exactly match the adopted base,
angle bends enabled, CUDA and MACE float64. Multi-seed batching is used except CBTYCO, whose native two-variant
E/Z dispatch is preserved with serial seeds. Solver and MACE workloads run sequentially on the GPU.
All entries are mean kJ/mol over three seeds; negative change is favorable.

{table(['Case','Off pick','On pick','Change','Off floor','On floor','Off regret','On regret'],comparison)}

These are selected-case diagnostics, not a dev80 effect estimate. Complete pool metrics (top-1 within 4.184,
Spearman rho within floor+50, invalid share >1000, assignment IDs and pool size) are in
`n2_followup_metrics_2026-10-18.csv`. Basin share was not measured. All scored energy arrays are finite.

The fresh off controls mostly reproduce the stored run closely, but PEWBOL seed 11 changes from 153.73 to
242.86 kJ/mol (+89.13); seed 13 changes by -4.76. Its original and fresh results must not be mixed to claim a
precise treatment effect. The matched on/off changes are still adverse in all three PEWBOL seeds. One control
run per seed does not estimate the full distribution of numerical solver noise.

## XEPKAF: recovered stacking, direct Cp coupling, and frozen Cp strain

The recovered phenyl stack is real. A hotspot on a P-bound carbon does not establish a separate interaction
energy: atomic MACE energy is a model decomposition and the global residual remains separate. Covalent bonds
change by at most 2.25e-6 A between stored before/after selections, so the displayed carbon signal is not NCI
stretching that bond. Coordination geometry is recorded in `n2_pose_audit_2026-10-18.json`.

The Cp ring actually participates directly in NCI. The topology generator admits five-membered sp2 rings,
including metal-bound Cp, in both pi-pi and CH-pi candidates. For fresh seed 11, Cp-involving CH-pi contacts
contribute 2.448 of 2.449 CH-pi nats, and Cp-involving pi-pi contributes 0.209 of 3.496 pi-pi nats. Seed 12
has 2.399 Cp-involving CH-pi nats. Thus an assumption that NCI leaves Cp untouched is false; rigid coordination
couples the ligands too. This is not evidence that such contacts should be excluded indiscriminately.

Independent of NCI, the free Cp embedding has uneven C-C bonds: one is 1.35046 A vs 1.41950 A in the XEPKAF
reference. GFN2's donor-distance restraints include every pair of atoms within a haptic face, preserving the
MMFF distortion. An unconstrained ligand-only Cp-anion refinement lowers stored-pose full-complex energies
by 5.16-11.05 kJ/mol; it also improves the full-solve results below. This test never relaxes a complex.

## JEFJUZ: embedding mismatch, not damaged target coordinates

Exact topology and atom mapping give ordinary reference bonds: N-N 1.43727 A and aryl C-N 1.36358/1.37472 A.
The production free-ligand refinement gives N-N 1.37833 A and C-N about 1.31059 A. Cp bonds reach 1.35046 A vs
1.41598 A in one reference edge. The production embeddings reproduce the saved bad lengths; all three seeds
and NCI selections retain them within approximately 2e-6 A. This localizes the mismatch to ligand embedding,
not NCI-induced stretching or a malformed viewer bond graph.

Two distinct mechanisms were tested: release intra-face Cp restraints; retain H binding caps during GFN2
refinement of the dianionic N-N ligand. The current path removes those caps before refining a free dianion.
The capped trial reduces its bond-length RMS error from 0.02652 to 0.00689 A. Inserting its entire geometry
into a fixed complex raises energy, but letting the statistical solver reorient it improves all three seeds.
Fixed-pose insertion alone was therefore insufficient to judge this embedding candidate.

A stricter bond-only experiment translates bridge-connected fragments along C-N/N-N axes: other bond lengths,
all bond angles and torsions remain unchanged. Using cap-predicted lengths lowers the three original picks
by 27.03, 23.52 and 10.95 kJ/mol; reference lengths give decreases of 22.60, 23.34 and 14.67. This confirms a
real bond-strain contribution while also showing it cannot explain the entire 250-327 kJ/mol error.

Full solves with ligand-only changes, unaltered scoring (energy relative to the same original reference):

{table(['Case','Embedding/NCI','Seed 11','Seed 12','Seed 13','Mean pick','Mean floor'],repair)}

The combined hypothesis was declared after the individual trials and before its own run. These candidate
embeddings use no target coordinates. They remain in development tooling; their chemical admission and
broader regression behavior have not been validated for a production default.

## PEWBOL: coordination isomer and sulfonate conformation are different constraints

The true reference has N-M-N 179.995 degrees, opposing N-to-S vectors 179.992 degrees, and eight eligible
H-bonds under the shared co-bound exclusion. An ideal trans donor assignment does not force the sulfonates
to point oppositely. The stored pools contain only 2/0/2 structures with N-M-N >140 degrees AND N-to-S-vector
angle >150 degrees (seeds 11/12/13). Their energies remain high: floors 252.90/absent/270.16 kJ/mol. Fresh NCI
pools have 1/1/2 such structures, floors 466.92/499.13/375.89. Some anti-like structures are sampled, but the
reference-quality basin is not recovered simply by changing a pick.

The original seed-11 pick is N-cis but sulfonates largely opposite (89.68/160.61 degrees), with seven H-bonds.
The NCI re-ranked pick is N-trans but sulfonates oriented differently (176.31/66.50 degrees), with six H-bonds.
Fresh NCI picks have only 5/4/3 H-bonds vs reference eight. More counted bonds is not a sufficient energy
criterion, but the replacement demonstrably does not preserve the reference H-bond network.

Reference-isomer diagnostic solves pin only the donor/site assignment (reference use explicitly limited to
this oracle test). Embedding, sulfonate orientations and scoring remain free:

{table(['NCI','Seed','Pick kJ/mol','N-M-N','N-to-S angle','H-bonds'],isoref)}

Prior same-pool ablations show removing old Q6 polar-H wells is the dominant PEWBOL ranking change, including
co-bound contacts outside NCI's candidate set; restricting removal only to covered contacts already failed.
The evidence supports both an objective/coverage problem and an insufficiently recovered ligand/H-bond basin,
not a simple inability to enumerate any trans isomer. No coefficient was fitted to these selected cases.

## Score balance and curvature

Architecture sections 4.10/8.4 define ratio = energy curvature / (RT * negative score curvature); 1 is energy-like.
Reused the exact MACE energy scans and topology motions at six cases x three seeds x stored before/after.
Rebuilt the original assignment-specific score and reproduced every central stored term within
{result['checks']['curvature_central_term_max_error']:.3g} raw nats. Identical displacements compare NCI off/on;
H-bond, stacking and CH-pi derivatives are retained separately. No new curvature energies or weight fits.

{table(['Case','Off ratio','On ratio','Stable off','Stable on','Wrong sign off','Wrong sign on'],curve_rows)}

Ratios use positive energy and score curvature. Stable rows require both physical and score curvature to agree
within 25% across 0.04 and 0.02 A displacement steps (with a unit floor in the tolerance). Wrong sign counts
positive physical curvature with zero/negative score curvature. Counts describe this selected motion set,
not independent samples. The median is not the whole curvature distribution.

There is no common multiplicative balance collapse. XEPKAF barely changes; PEWBOL's median moves closer to 1,
yet its wrong-sign directions increase from 3 to 28/300. JEFJUZ gains ten such directions. Bilinear NCI tables
also create step-sensitive curvature near grid-cell boundaries; full per-mode values and step sensitivity
are in `n2_balance_audit_2026-10-18.csv`. Cartesian scans omit the unreconstructable bend/closure prior in both
arms, following `forces.py`; they do not claim the complete joint-coordinate Hessian. CBTYCO's E/Z-dependent
assignment reconstruction was excluded from this curvature audit, but its full solves were run correctly.

## Artifacts and validation

- `n2_followup_summary_2026-10-18.json`: checks, all arm manifests, metrics and exported XYZ paths.
- `n2_structure_audit_2026-10-18.json`: every covalent bond traced through ETKDG, MMFF, GFN2 and stored selections.
- `n2_pose_audit_2026-10-18.json`: measured isomer/conformer categories and direct Cp NCI coupling.
- `n2_internal_experiment_2026-10-18/`: ligand trials, bond-only/whole-ligand counterfactuals and energy results.
- `n2_fresh_{{solve,control}}_2026-10-18/`: complete 21-pool on/off experiments and MACE64 arrays.
- `n2_repair_*_2026-10-18_v2/`: validated experimental embeddings. Earlier non-v2 Cp trials were invalidated
  because post-MMFF aromatic flags skipped the intended intervention; never use them as repair evidence.
- `data/inspect/<CODE>/followup_<arm>_s<seed>.xyz`: fresh and repaired selected structures; prior viewer/XYZ retained.
- 23 focused numerical/topology tests passed, including bond-only angle/torsion/ring invariants. Every saved
  statistical ranking re-evaluation agrees; {len(rows)} successful finite pools. DB hashes and source hashes verified.

Initial sandboxed embedding checks silently fell back to MMFF after temporary-directory access denial. They
were rerun with working xTB access and replaced before interpretation. A recursive prototype wrapper was stopped
before usable output; the validated runner verifies an actual change in internal distances. No production
default, runtime scorer, historical DB, benchmark input or reference coordinate was changed by this follow-up.
'''
    (REPORTS/'n2_followup_2026-10-18.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    codes=[r[0] for r in comparison]
    for j,c in enumerate(codes):
        off=df[(df.case==c)&(df.arm=='off')].set_index('seed')
        on=df[(df.case==c)&(df.arm=='on')].set_index('seed')
        delta=on.pick-off.pick
        axes[0].scatter(delta,np.full(3,j),color='#cc603c',alpha=.8)
        axes[0].plot([delta.mean()],[j],marker='|',markersize=18,color='black')
    axes[0].set_yticks(range(len(codes)),codes);axes[0].axvline(0,color='gray',lw=1)
    axes[0].set_xlabel('NCI on - matched off pick energy (kJ/mol)');axes[0].set_title('Fresh full solves; three seeds per case')
    jef=df[(df.case=='JEFJUZ') & df.arm.isin(['off','on','cp_off','caps_off','both_off','both_on'])]
    arms=['off','on','cp_off','caps_off','both_off','both_on']
    for s,g in jef.groupby('seed'):
        axes[1].plot(range(len(arms)),g.set_index('arm').loc[arms].pick,marker='o',label=f'seed {s}')
    axes[1].set_xticks(range(len(arms)),arms,rotation=25);axes[1].axhline(200,color='gray',ls='--',lw=1)
    axes[1].set_ylabel('Pick energy relative to reference (kJ/mol)');axes[1].set_title('JEFJUZ: ligand-only embedding hypotheses');axes[1].legend()
    fig.savefig(REPORTS/'n2_followup_2026-10-18.png',dpi=170);plt.close(fig)
    print(json.dumps(result['checks']),flush=True)


if __name__=='__main__':setup();run()
