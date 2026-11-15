"""Reference-only P0 feasibility; no pools, model, or energy-property reads.

Run with nice -n 19, one CPU thread. Budget exits leave hash-checked parent
reductions for resume. Every independent count uses the frame group_id.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import resource
from types import SimpleNamespace
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2 import FAMILIES,enumerate_opportunities,features_torch
from research.archive.code.interactions.nci_v2.nci_v2_extract import unique_opportunities,freeze
from research.archive.code.interactions.nci_v2.nci_v2_surface import domains
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,Run,load_json,write_json,digest

torch.set_num_threads(1)

class BudgetPause(RuntimeError):
    pass


def host_budget(min_available_gib=5.,max_rss_mib=2300.):
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(mem['MemAvailable'].split()[0])/2**20
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if available<min_available_gib:
        raise BudgetPause(f'available host RAM {available:.3f} GiB below {min_available_gib} GiB')
    if peak>=max_rss_mib:
        raise BudgetPause(f'peak RSS {peak:.1f} MiB exceeds conservative {max_rss_mib} MiB guard')
    return dict(available_gib=available,peak_rss_mib=peak)


def parent_reduction(parent,mol,spec):
    opportunities=unique_opportunities(enumerate_opportunities(mol,spec))
    X=torch.tensor(np.asarray([a.position for a in mol.atoms]),dtype=torch.float64)
    features=features_torch(X,opportunities,spec).features
    reductions={}
    for family in FAMILIES:
        grouped=defaultdict(lambda:dict(rows=0,in_radial_domain=0,nonzero_taper=0))
        for i,o in enumerate(opportunities.opportunities[family]):
            root=freeze(o['z_o'])[0];key=str(root)
            dim=2 if family=='bond_atom' and not o['has_c_neighbors'] else 4
            domain,window=domains(family,root,dim,spec)
            r=float(features[family][0,i,0])
            entry=grouped[key];entry['rows']+=1
            entry['in_radial_domain']+=int(domain[0][0]<=r<=domain[0][1])
            entry['nonzero_taper']+=int(window[0]<r<window[-1])
        reductions[family]=dict(grouped)
    return dict(code=parent['representative_code'],group_id=parent['group_id'],
                stratum=parent.get('stratum'),roots=reductions,n_atoms=len(mol.atoms),
                energy_properties_attached=False,status='ok')


def aggregate(parents,expected_parents):
    roots={f:{} for f in FAMILIES};groups={f:defaultdict(set) for f in FAMILIES}
    inside_groups={f:defaultdict(set) for f in FAMILIES}
    for p in parents:
        if p['status']!='ok':continue
        for family,entries in p['roots'].items():
            for root,values in entries.items():
                e=roots[family].setdefault(root,dict(root=root,rows=0,in_radial_domain=0,nonzero_taper=0,parent_counts=[],parent_ids=[]))
                for k in ('rows','in_radial_domain','nonzero_taper'):e[k]+=values[k]
                e['parent_counts'].append(values['rows']);e['parent_ids'].append(p['code'])
                groups[family][root].add(p['group_id'])
                if values['in_radial_domain']:inside_groups[family][root].add(p['group_id'])
    family_summary={}
    for f,entries in roots.items():
        for root,e in entries.items():
            counts=np.array(e.pop('parent_counts'));n=len(groups[f][root]);ni=len(inside_groups[f][root])
            e.update(independent_groups=n,in_domain_independent_groups=ni,can_reach_20_groups=n>=20,
                     can_reach_20_in_domain_groups=ni>=20,parents=len(counts),
                     rows_per_nonzero_parent=dict(min=int(counts.min()),median=float(np.median(counts)),
                        p95=float(np.quantile(counts,.95)),max=int(counts.max())),
                     fraction_in_radial_domain=e['in_radial_domain']/e['rows'],
                     fraction_nonzero_taper=e['nonzero_taper']/e['rows'])
            e['group_ids']=sorted(groups[f][root]);e['in_domain_group_ids']=sorted(inside_groups[f][root])
        rr=list(entries.values());total=sum(e['rows'] for e in rr);inside=sum(e['in_radial_domain'] for e in rr)
        qualifying=[e for e in rr if e['can_reach_20_groups']];active=[e for e in rr if e['can_reach_20_in_domain_groups']]
        family_summary[f]=dict(roots=len(rr),reference_rows=total,in_radial_domain_rows=inside,
            fraction_in_radial_domain=inside/total if total else None,
            roots_with_20_groups=len(qualifying),roots_with_20_in_domain_groups=len(active),
            fraction_rows_in_qualifying_roots=sum(e['rows'] for e in qualifying)/total if total else None,
            fraction_in_domain_rows_in_qualifying_roots=sum(e['in_radial_domain'] for e in active)/inside if inside else None)
    total_inside=sum(x['in_radial_domain_rows'] for x in family_summary.values())
    eligible_inside=sum(e['in_radial_domain'] for entries in roots.values() for e in entries.values() if e['can_reach_20_in_domain_groups'])
    share=eligible_inside/total_inside if total_inside else 0.
    # Predeclared decision uses eligible in-domain row coverage, not a long-tail
    # count of rare roots. Fewer than 5% means almost no reference coverage can activate.
    pause=share<.05
    return dict(submitted_parents=len(expected_parents),submitted_groups=len({p['group_id'] for p in expected_parents}),
        processed_parents=len(parents),successful_parents=sum(p['status']=='ok' for p in parents),
        missing=[p for p in parents if p['status']!='ok'],families=family_summary,roots=roots,
        eligible_in_domain_row_fraction=share,pause_further_work=pause,
        pause_rule='eligible in-domain reference rows <5% (predeclared reporting decision, not a model policy)',
        interpretation='Reference-side upper bound only: baseline group support, ESS, local joint support and calibration remain unknown.',
        production_ready=False)


def selected_smiles(clean_root,dataset,wanted):
    """Read only identifier/SMILES fields; never parse or attach energy properties."""
    path=Path(clean_root)/('lnQM' if dataset=='LNf' else 'tmQM')/'tmQM_y.csv'
    result={}
    with path.open() as stream:
        reader=csv.DictReader(stream,delimiter=';')
        for row in reader:
            if row['CSD_code'] in wanted:result[row['CSD_code']]=row.get('SMILES') or None
    return result


def reference_stream(parents,clean_root):
    """Scan each selected shard once; assemble only selected reference molecules."""
    from jacobik.fit.parser import _iter_xyz_blocks,_iter_bo_blocks,_assemble_raw
    from jacobik.stats.mapping import assemble
    by_shard=defaultdict(dict)
    for p in parents:by_shard[(p['dataset'],p['representative_shard'])][p['representative_code']]=p
    smiles={ds:selected_smiles(clean_root,ds,{p['representative_code'] for p in parents if p['dataset']==ds})
            for ds in {p['dataset'] for p in parents}}
    for (dataset,shard),wanted in sorted(by_shard.items()):
        host_budget()
        base=Path(clean_root)/('lnQM' if dataset=='LNf' else 'tmQM')
        found=set()
        for (comment,atoms),(code,bonds) in zip(_iter_xyz_blocks(base/'xyz'/f'tmQM_{shard}.xyz'),
                                               _iter_bo_blocks(base/'bo'/f'tmQM_{shard}.BO')):
            if code not in wanted:continue
            host_budget();found.add(code)
            try:
                raw=_assemble_raw(comment,atoms,code,bonds,SimpleNamespace(smiles=smiles[dataset].get(code)))
                mol=assemble(raw);mol.smiles=raw.smiles
                yield wanted[code],mol,None
            except Exception as exc:yield wanted[code],None,f'{type(exc).__name__}: {exc}'
            if len(found)==len(wanted):break
        for code in sorted(set(wanted)-found):yield wanted[code],None,'reference_not_found'


def report_markdown(report):
    lines=['# P0 scaled reference-root feasibility','',
        f"{report['submitted_parents']} parents / {report['submitted_groups']} independent group_id values.",
        f"Eligible in-domain row coverage: {report['eligible_in_domain_row_fraction']:.2%}.",
        f"Pause further work: {report['pause_further_work']}. Reference support is an upper bound only.",'',
        '| Family | Roots | Roots ≥20 groups | Roots ≥20 in-domain groups | Reference rows | Fraction in domain |',
        '|---|---:|---:|---:|---:|---:|']
    for f,e in report['families'].items():
        fraction=e['fraction_in_radial_domain']
        lines.append(f"| {f} | {e['roots']} | {e['roots_with_20_groups']} | {e['roots_with_20_in_domain_groups']} | {e['reference_rows']} | {fraction:.2%} |" if fraction is not None else f'| {f} | 0 | 0 | 0 | 0 | n/a |')
    lines+=['','## Root table','',
        '| Family | Root | Groups | In-domain groups | Rows | In-domain fraction | Parent median / p95 / max |',
        '|---|---|---:|---:|---:|---:|---|']
    for f,entries in report['roots'].items():
        for e in sorted(entries.values(),key=lambda x:(-x['independent_groups'],x['root'])):
            q=e['rows_per_nonzero_parent']
            lines.append(f"| {f} | `{e['root']}` | {e['independent_groups']} | {e['in_domain_independent_groups']} | {e['rows']} | {e['fraction_in_radial_domain']:.2%} | {q['median']} / {q['p95']} / {q['max']} |")
    return '\n'.join(lines)+'\n'


def run_p0(spec_path,frame_path,clean_root,out=None):
    if os.nice(0)<19:raise ValueError('P0 must run under nice -n 19')
    host_budget();frame=load_json(frame_path);spec=load_json(spec_path)
    parents=frame['main_allocation']['splits']['surface_fit']
    if len({p['representative_code'] for p in parents})!=len(parents):raise ValueError('duplicate parent identity')
    inputs=[frame_path]
    for p in parents:
        b=Path(clean_root)/('lnQM' if p['dataset']=='LNf' else 'tmQM')
        inputs += [b/'xyz'/f"tmQM_{p['representative_shard']}.xyz",b/'bo'/f"tmQM_{p['representative_shard']}.BO",b/'tmQM_y.csv']
    run=Run('phase2-root-feasibility',spec_path,inputs,dict(split='surface_fit',min_groups=20,min_available_gib=5,max_rss_mib=2300),out)
    if run.resumed:return run.path
    checkpoint=run.path/'checkpoint.json';saved=load_json(checkpoint) if checkpoint.exists() else {}
    reductions=[]
    for code,item in saved.items():
        if digest(run.path/item['path'])!=item['sha256']:raise ValueError('checkpoint checksum mismatch')
        reductions.append(load_json(run.path/item['path']))
    remaining=[p for p in parents if p['representative_code'] not in saved]
    try:
        for parent,mol,error in reference_stream(remaining,clean_root):
            host_budget();code=parent['representative_code']
            reduction=parent_reduction(parent,mol,spec) if error is None else dict(code=code,group_id=parent['group_id'],status='missing',error=error)
            rel=f'parents/{code}.json';write_json(run.path/rel,reduction)
            saved[code]=dict(path=rel,sha256=digest(run.path/rel));write_json(checkpoint,saved);reductions.append(reduction)
            if len(saved)%25==0:print(f'P0 {len(saved)}/{len(parents)}: {host_budget()}',flush=True)
    except BudgetPause as exc:
        write_json(run.path/'reports/pause.json',dict(reason=str(exc),completed=len(saved),status='paused_low_resources'))
        print(f'PAUSED {exc}; resume directory {run.path}',flush=True)
        raise
    report=aggregate(reductions,parents);write_json(run.path/'reports/root_feasibility.json',report)
    md=run.path/'reports/root_feasibility.md';md.write_text(report_markdown(report))
    run.finish(dict(parents=len(parents),independent_groups=report['submitted_groups'],
        reference_rows=sum(e['reference_rows'] for e in report['families'].values())),
        ['Reference-only upper bounds; no pools, energies or production outputs read.',
         'P0 threshold requires independent group_id, never parent count.'])
    return run.path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--frame',type=Path,required=True)
    p.add_argument('--clean-root',type=Path,default=MAIN/'data/refit/clean');p.add_argument('--out-dir',type=Path)
    a=p.parse_args()
    try:print(run_p0(a.spec,a.frame,a.clean_root,a.out_dir),flush=True)
    except BudgetPause:raise SystemExit(75)

if __name__=='__main__':main()
