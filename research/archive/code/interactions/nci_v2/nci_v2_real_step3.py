"""Audited real-data Step 3 extraction and real-subset CV check.

Inputs are read-only. All run products are restricted to data_dev/.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, os, resource, time
from pathlib import Path
import numpy as np
import torch

from research.archive.code.interactions.nci_v2.nci_v2_devio import DEV, MAIN, load_json, write_json, digest
from research.archive.code.interactions.nci_v2.nci_v2_s21_overlay import S21Adapter, ExcludedPool, IntegrityError
from research.archive.code.interactions.nci_v2.nci_v2_controls import health_metrics, placebo_opportunities
from research.archive.code.interactions.nci_v2.nci_v2_extract import graph_record, unique_opportunities
from research.archive.code.interactions.nci_v2.nci_v2_extract_columnar import stream_parent_rows
from research.archive.code.interactions.nci_v2.nci_v2_columnar import write_columns, RootRows, support_mass, grouped_cv_root, host_budget
from research.archive.code.interactions.nci_v2.nci_v2_binning import bin_root, supported_error
from research.archive.code.interactions.nci_v2.nci_v2 import enumerate_opportunities, features_torch, FAMILIES

DATA = MAIN / 'research/data/refit_nci_v2/scaled_v1'
CLEAN = MAIN / 'data/refit/clean'
TMQM = CLEAN / 'tmQM'
POLICY_PATH = Path(__file__).with_name('nci_v2_controls_policy.json')
SEEDS = (11, 12, 13)


def set_cpu():
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)


def max_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 if os.name != 'darwin' else 1024 * 1024))


def code_id(parent):
    return parent.get('representative_code', parent.get('code'))


def resolve_slot_parent(slot_code, original, reserve, slot=None, *, sensitivity=False):
    """Resolve one predeclared slot from its exact frozen frame entry."""
    slot = slot or {}
    if slot and slot.get('state') in ('still_failed','not_retried'):
        occupant=slot['replacement_code']; origin='step2_replacement'
        parent=reserve.get(occupant)
        if parent is None: raise IntegrityError(f'replacement {occupant} absent from frozen reserve frame')
    elif slot and slot.get('state')=='restored' and sensitivity:
        occupant=slot_code; origin='step2_original_sensitivity'; parent=original
    else:
        occupant=slot_code; origin='retry' if slot.get('state')=='restored' else 'step2_original'
        parent=original
    resolved=dict(parent,code=occupant,shard=parent['representative_shard'],origin=origin,
                  slot_code=slot_code,slot_stratum=original['stratum'])
    if resolved['shard']!=parent['representative_shard'] or resolved['dataset']!=parent['dataset']:
        raise IntegrityError(f'frame shard/dataset identity changed for {occupant}')
    return occupant,resolved,origin


def resolve_cohort(frame, manifest, repair, *, sensitivity=False):
    """Resolve the 1,200 frozen slots against S2.1 state and Step 2 occupants."""
    splits = frame['main_allocation']['splits']
    fit = {code_id(p): p for p in splits['surface_fit']}
    reserve = {code_id(p): p for p in splits['reserve']}
    slots = repair['slots']; result=[]; decisions=[]
    for slot_code, original in fit.items():
        slot=slots.get(slot_code);occupant,parent,origin=resolve_slot_parent(slot_code,original,reserve,slot,sensitivity=sensitivity)
        if occupant in {code_id(x) for x in result}: raise IntegrityError(f'duplicate cohort occupant {occupant}')
        task=manifest['tasks'].get(f'{occupant}:11:control')
        if task is None: raise IntegrityError(f'missing Step 2 task for resolved occupant {occupant}')
        if task.get('group_id') and task.get('group_id') != parent['group_id']:
            raise IntegrityError(f'frame/task group mismatch: {occupant}')
        result.append(parent)
        decisions.append(dict(slot_code=slot_code,occupant=occupant,origin=origin,
                              slot_state=slot.get('state') if slot else 'unreplaced',
                              replacement_state=slot.get('replacement_state') if slot else None))
    if len(result)!=1200: raise IntegrityError(f'cohort resolves to {len(result)}, expected 1200 slots')
    return result,decisions


class Step2OccupantAdapter:
    """Use Step 2 pools for restored codes in the no-restoration sensitivity run."""
    def __init__(self,adapter,restored_codes,unavailable_codes=()):
        self.adapter=adapter;self.restored_codes=set(restored_codes)
        self.unavailable_codes=set(unavailable_codes)
    def open(self,key):
        code=key.split(':',1)[0]
        return self.adapter.open(key,step2_original=code in self.restored_codes)
    def skip_parent(self,parent):return parent['code'] in self.unavailable_codes
    def may_skip_unavailable_pool(self,code):return code in self.restored_codes
    def reference_identity(self,key):return self.adapter.reference_identity(key)


def shard_provenance(parents):
    files={}
    for shard in sorted({p['shard'] for p in parents}):
        for kind,rel in [('xyz',f'xyz/tmQM_{shard}.xyz'),('bo',f'bo/tmQM_{shard}.BO')]:
            p=TMQM/rel
            files[f'{shard}_{kind}']={'path':str(p),'sha256':digest(p)}
    return files


def prepare_parent(parent, spec, loader_commit, cache_dir, cache_records, exclusions):
    from jacobik.inputs import load_mol, build_builder_and_inputs
    code=parent['code']; shard=parent['shard']; dataset=parent['dataset']
    if shard!=parent['representative_shard'] or dataset!=parent['dataset']:
        raise IntegrityError(f'frame shard/dataset mismatch for {code}')
    try: mol=load_mol(code,shard,dataset,CLEAN)
    except Exception as exc:
        exclusions.append(dict(code=code,stage='load_mol',error=f'{type(exc).__name__}: {exc}'));return None
    if mol is None:
        exclusions.append(dict(code=code,stage='load_mol',error='load_mol returned None'));return None
    # Remove attached QM fields by deletion only. Never read or serialize them.
    for field in ('electronic_energy','dispersion_energy','dipole_moment','metal_charge',
                  'hl_gap','homo_energy','lumo_energy','polarizability'):
        try: delattr(mol,field)
        except AttributeError: pass
    builder, ligand_inputs, error=build_builder_and_inputs(mol)
    if error or builder is None:
        exclusions.append(dict(code=code,stage='build_builder_and_inputs',error=error or 'builder is None'));return None
    record=graph_record(mol)
    cache_records.setdefault(shard,{})[code]=record
    return mol


def pool_health(parent,mol,view,policy,overlay_agreement):
    # The adapter has already checked file/index/overlay SHA256. Accessing this
    # property rechecks those hashes, applies gen_to_ref_idx, and validates the
    # mapping convention before returning reference-ordered coordinates.
    assert np.array_equal(view.symbols[view.g],view.ref_symbols)
    X=view.coordinates
    assert np.array_equal(X,view.X[:,view.g,:])
    ref=np.asarray([a.position for a in mol.atoms],dtype=np.float64)
    symbols=np.asarray([a.symbol for a in mol.atoms])
    if (not np.array_equal(symbols,view.ref_symbols) or ref.shape!=view.ref_X.shape or
        not np.isfinite(ref).all() or np.max(np.abs(ref-view.ref_X),initial=0.)>1e-6):
        raise IntegrityError(f'reference identity mismatch for {view.key}')
    if int(mol.charge)!=view.charge or int(mol.spin_state)!=view.multiplicity:
        raise IntegrityError(f'charge/multiplicity mismatch for {view.key}')
    finite=np.isfinite(X).all((1,2)); full={}
    for k in ('healthy','severity','max_bond_relative','max_shell_shift','angle_rms_shift_degrees',
              'pucker_rms_shift_angstrom','gross_overlap','heavy_overlap','stray_atom','min_nonbonded_heavy_distance',
              'bad_bond','coord_intact','min_nonbonded_ratio'):
        full[k]=np.zeros(len(X),dtype=bool if k in ('healthy','gross_overlap','heavy_overlap','stray_atom','bad_bond','coord_intact') else float)
    fi=np.flatnonzero(finite)
    if len(fi):
        m, catalog, refv, vals=health_metrics(mol,ref,X[fi],policy,view.coordination['coord_intact'][fi])
        for k,v in m.items(): full[k][fi]=v
        # Independent recomputation of the overlay R3 fields, same definitions
        # as S2.1; this is diagnostic only and never changes admission.
        from research.archive.code.refit.nci_v2_s2_1_repair import health_metrics as r3_health
        direct=r3_health(ref,symbols,X[fi])
        if view.health_source=='R3_overlay':
            checks={
                'graph_match': np.asarray(view.health['graph_match'])[fi] == direct['graph_match'],
                'max_bond_dev_A': np.isclose(np.asarray(view.health['max_bond_dev_A'])[fi],direct['max_bond_dev_A'],rtol=0,atol=1e-10),
                'min_heavy_nonbonded_A': np.isclose(np.asarray(view.health['min_heavy_nonbonded_A'])[fi],direct['min_heavy_nonbonded_A'],rtol=0,atol=1e-10,equal_nan=True),
            }
            r3_bond_ok=m['max_bond_relative']<=policy['health']['max_covalent_bond_relative_deviation']
            r3_heavy_ok=np.asarray(view.health['min_heavy_nonbonded_A'])[fi]>=policy['health']['min_nonbonded_heavy_distance_angstrom']
            r3_core=np.asarray(view.health['graph_match'])[fi]&r3_bond_ok&r3_heavy_ok
            for k,v in checks.items():
                overlay_agreement['field_checks'][k][0]+=int(np.count_nonzero(v))
                overlay_agreement['field_checks'][k][1]+=len(v)
            policy_ok=m['healthy']
            overlay_agreement['pass_fail']['n']+=len(fi)
            overlay_agreement['pass_fail']['agree']+=int(np.count_nonzero(policy_ok==r3_core))
            overlay_agreement['pass_fail']['policy_pass_r3_fail']+=int(np.count_nonzero(policy_ok&~r3_core))
            overlay_agreement['pass_fail']['policy_fail_r3_pass']+=int(np.count_nonzero(~policy_ok&r3_core))
            overlay_agreement['pass_fail']['both_fail']+=int(np.count_nonzero(~policy_ok&~r3_core))
            graph_fail=~np.asarray(view.health['graph_match'])[fi]
            bond_fail=~r3_bond_ok
            heavy_fail=~r3_heavy_ok
            ppr3f=policy_ok&~r3_core;pf_r3p=~policy_ok&r3_core
            component_masks={'graph_match':graph_fail,'bond_deviation':bond_fail,'min_heavy_nonbonded':heavy_fail}
            for name,mask in component_masks.items():
                overlay_agreement['r3_disagreement']['policy_pass_r3_fail_component_'+name]+=int(np.count_nonzero(ppr3f&mask))
            for i in np.flatnonzero(ppr3f):
                pattern='+'.join(name for name,mask in component_masks.items() if mask[i])
                overlay_agreement['r3_disagreement']['policy_pass_r3_fail_pattern:'+pattern]+=1
                overlay_agreement['policy_pass_r3_fail_cases'].append(dict(
                    code=view.code,seed=int(view.seed),pose_index=int(fi[i]),
                    failed_components=[name for name,mask in component_masks.items() if mask[i]]))
            coord_fail=~m['coord_intact']
            overlay_agreement['r3_disagreement']['policy_fail_r3_pass_coordination_fail']+=int(np.count_nonzero(pf_r3p&coord_fail))
            policy_reason_masks={'coordination':coord_fail,'gross_overlap':m['gross_overlap'],
                                 'bond_deviation':m['bad_bond'],'heavy_overlap':m['heavy_overlap'],
                                 'stray_atom':m['stray_atom']}
            for i in np.flatnonzero(pf_r3p):
                pattern='+'.join(name for name,mask in policy_reason_masks.items() if mask[i])
                overlay_agreement['r3_disagreement']['policy_fail_r3_pass_policy_pattern:'+pattern]+=1
        overlay_agreement['recomputed_r3']['graph_match']+=len(fi) if view.health_source!='R3_overlay' else int(np.count_nonzero(checks['graph_match']))
    full['healthy'] &= finite
    full['healthy'][~finite]=False
    full['nonfinite']=~finite
    return X,full


HEALTH_FAILURE_REASONS=('nonfinite','bad_bond','coord_intact','gross_overlap','heavy_overlap','stray_atom')


def health_failure_counts(metrics):
    """Count frozen-policy failures; coordination integrity is True when it passes."""
    return {reason:int(np.count_nonzero(~metrics[reason] if reason=='coord_intact' else metrics[reason]))
            for reason in HEALTH_FAILURE_REASONS}


def extract_columns(parents, adapter, spec, out, *, group_subset=None, focus_one_root=False, include_placebo=True):
    out=Path(out)
    if not out.resolve().is_relative_to(DEV.resolve()): raise ValueError('output must be under data_dev')
    out.mkdir(parents=True,exist_ok=True)
    parent_groups=collections.Counter(p['group_id'] for p in parents)
    exclusions=[];cache_records={};pool_records=[];health_by_code={};root_health=collections.defaultdict(collections.Counter)
    overlay_agreement={'field_checks':collections.defaultdict(lambda:[0,0]),
                       'pass_fail':collections.Counter(n=0,agree=0,policy_pass_r3_fail=0,policy_fail_r3_pass=0,both_fail=0),
                       'r3_disagreement':collections.Counter(),
                       'policy_pass_r3_fail_cases':[],
                       'recomputed_r3':collections.Counter()}
    prepared=[];identity_counts=collections.Counter()
    for i,parent in enumerate(parents,1):
        if group_subset is not None and parent['group_id'] not in group_subset: continue
        host_budget()
        code=parent['code'];print('real Step 3 parent',i,'/',len(parents),code,flush=True)
        if getattr(adapter,'skip_parent',lambda _:False)(parent):
            reason='Step 2 occupant has no successful pool for any cohort seed; restoration excluded by sensitivity design'
            exclusions.append(dict(code=code,stage='step2_sensitivity_pool_unavailable',error=reason))
            health_by_code[code]=dict(group_id=parent['group_id'],origin=parent['origin'],
                excluded_pool=len(SEEDS),healthy=0,unavailable_step2_pool=len(SEEDS))
            for seed in SEEDS:
                pool_records.append(dict(code=code,seed=seed,origin='step2_original_sensitivity',
                    status='excluded',reason='task_not_successful',shard=parent['shard'],dataset=parent['dataset']))
            identity_counts['parents_without_successful_step2_pool_excluded']+=1
            continue
        mol=prepare_parent(parent,spec,None,out/'graph_cache',cache_records,exclusions)
        if mol is None: continue
        identity_counts['loaded_parents_frame_shard_dataset_checked']+=1
        ref=np.asarray([a.position for a in mol.atoms],dtype=np.float64)
        opp=unique_opportunities(enumerate_opportunities(mol,spec))
        poses=collections.defaultdict(list);health_parent=collections.Counter()
        for seed in SEEDS:
            key=f'{code}:{seed}:control'
            try:view=adapter.open(key)
            except ExcludedPool as exc:
                if str(exc)=='unmappable':
                    pool_symbols,pool_ref=adapter.reference_identity(key)
                    mol_symbols=np.asarray([a.symbol for a in mol.atoms]);mol_ref=np.asarray([a.position for a in mol.atoms])
                    if (not np.array_equal(mol_symbols,pool_symbols) or mol_ref.shape!=pool_ref.shape or
                        not np.isfinite(mol_ref).all() or np.max(np.abs(mol_ref-pool_ref),initial=0.)>1e-6):
                        raise IntegrityError(f'reference identity mismatch for excluded unmappable {key}')
                    identity_counts['unmappable_pool_reference_identity_checked']+=1
                elif str(exc)=='task_not_successful' and getattr(adapter,'may_skip_unavailable_pool',lambda _:False)(code):
                    pool_records.append(dict(code=code,seed=seed,origin='step2_original_sensitivity',
                        status='excluded',reason='task_not_successful',shard=parent['shard'],dataset=parent['dataset']))
                    health_parent['unavailable_step2_pool']+=1
                    continue
                else:
                    raise IntegrityError(f'unexpected cohort pool exclusion for {key}: {exc}') from exc
                pool_records.append(dict(code=code,seed=seed,origin=parent['origin'],status='excluded',reason=str(exc),
                    shard=parent['shard'],dataset=parent['dataset']))
                health_parent['excluded_pool']+=1;continue
            if view.group_id and view.group_id != parent['group_id']:
                raise IntegrityError(f'pool group identity mismatch for {key}: {view.group_id} != {parent["group_id"]}')
            if view.stratum != parent['stratum']:
                raise IntegrityError(f'frame/pool stratum mismatch for {key}')
            X,metrics=pool_health(parent,mol,view,spec['solver_artifact_controls'],overlay_agreement)
            identity_counts['mapped_pool_reference_identity_checked']+=1
            keep=metrics['healthy']
            fail_counts=health_failure_counts(metrics)
            for reason,count in fail_counts.items():health_parent[f'fail_{reason}']+=count
            health_parent['poses']+=len(X);health_parent['healthy']+=int(keep.sum())
            for fam,occurrences in opp.opportunities.items():
                roots={str(o['z_o'][0]) for o in occurrences}
                for root in roots:
                    rec=root_health[(fam,root)]
                    rec['seen_conformers']+=len(X);rec['healthy_conformers']+=int(keep.sum())
                    rec['excluded_conformers']+=int((~keep).sum())
                    for reason,count in fail_counts.items():rec[f'fail_{reason}']+=count
            poses[seed].extend(X[keep])
            pool_records.append(dict(code=code,seed=seed,origin=view.origin,status='mapped',n_confs=len(X),
                healthy=int(keep.sum()),excluded=int((~keep).sum()),health_source=view.health_source,
                shard=parent['shard'],dataset=parent['dataset'],
                mapping_sha256=hashlib.sha256(np.asarray(view.g,dtype=np.int64).tobytes()).hexdigest()))
        health_by_code[code]=dict(group_id=parent['group_id'],origin=parent['origin'],**health_parent)
        prepared.append((parent,mol,{s:np.asarray(x) for s,x in poses.items()}))
    focus=None
    if focus_one_root:
        support=collections.defaultdict(lambda:{0:set(),1:set(), 'opportunities':0})
        for parent,mol,poses in prepared:
            group=parent['group_id']; has_base=any(len(x) for x in poses.values())
            opp=unique_opportunities(enumerate_opportunities(mol,spec))
            for fam,entries in opp.opportunities.items():
                for o in entries:
                    root=str(o['z_o'][0]);entry=support[(fam,root)]
                    entry[1].add(group);entry['opportunities']+=1
                    if has_base:entry[0].add(group)
        choices=[(min(len(v[0]),len(v[1])),len(v[0])+len(v[1]),v['opportunities'],len(v[0]),len(v[1]),fam,root)
                 for (fam,root),v in support.items() if min(len(v[0]),len(v[1]))>=25]
        if not choices:raise RuntimeError('subset has no root with >=25 groups in both sources')
        best=max(choices);focus=dict(family=best[5],root=best[6],group_support=best[0],
                                      groups_baseline=best[3],groups_reference=best[4],opportunities=best[2])
    # Cache only graph identity/topology and source shard hashes. No CSV field
    # other than loader-projected CSD_code/SMILES enters or is stored.
    provenance=shard_provenance(parents)
    from subprocess import check_output
    loader_commit=check_output(['git','log','-1','--format=%H','--','research/archive/code/refit/nci_v2_pilot.py','src/jacobik/fit/parser.py'],
                               cwd=Path(__file__).resolve().parents[1],text=True).strip()
    for shard,graphs in cache_records.items():
        shard_hashes={k:v for k,v in provenance.items() if k.startswith(shard+'_')}
        stamp=hashlib.sha256(json.dumps(shard_hashes,sort_keys=True).encode()).hexdigest()
        path=out/'graph_cache'/f'graphs_{shard}_{stamp}_{loader_commit[:12]}.json'
        write_json(path,dict(shard=shard,source_shard_hashes=shard_hashes,loader_commit=loader_commit,graphs=graphs))
    def row_stream():
        for parent,mol,poses in prepared:
            parent=dict(parent,parent_weight=1/parent_groups[parent['group_id']])
            for row in stream_parent_rows(parent,mol,np.asarray([a.position for a in mol.atoms]),poses,spec):
                if focus is None or (row['family']==focus['family'] and str(row['context'][0])==focus['root']):
                    yield row
    main_meta=write_columns(row_stream(),out/'columns',spec,chunk_size=8192)
    # Frozen placebo features: heavy-atom graph-distance 1-3/1-4 and interligand
    # reference-band controls, excluding admitted anchor cross-pairs.
    def placebo_stream():
        for parent,mol,poses in prepared:
            po,counts=placebo_opportunities(mol,np.asarray([a.position for a in mol.atoms]),spec)
            counts={k:v for k,v in counts.items() if v}
            if not counts: continue
            wp=1/parent_groups[parent['group_id']]
            for source,seed,coords in [(1,None,np.asarray([a.position for a in mol.atoms])[None])]+[(0,s,x) for s,x in poses.items()]:
                for x in coords:
                    feat=features_torch(torch.as_tensor(x,dtype=torch.float64),po,spec)
                    for fam,entries in po.opportunities.items():
                        for i,o in enumerate(entries):
                            n=counts[o['control_family']]
                            weight=wp/n if source else wp/(n*len(poses)*len(coords))
                            yield dict(parent_id=parent['code'],group_id=parent['group_id'],family=fam,
                                context=o['z_o'],features=feat.features[fam][0,i,:1].tolist(),y=source,weight=weight)
    placebo_meta=write_columns(placebo_stream(),out/'placebo_columns',spec,chunk_size=8192) if include_placebo else dict(physical_input_rows=0,roots=[])
    report=dict(parents_submitted=len(parents),parents_loaded=len(prepared),groups_submitted=len({p['group_id'] for p in parents}),
        groups_loaded=len({p['group_id'] for p,_,_ in prepared}),pool_records=pool_records,parent_health=health_by_code,
        exclusions=exclusions,health_exclusions_by_root=[dict(family=f,root=r,**v) for (f,r),v in sorted(root_health.items())],
        overlay_r3_agreement={'field_checks':{k:dict(agree=v[0],n=v[1]) for k,v in overlay_agreement['field_checks'].items()},
            'policy_vs_r3_subset_gate':dict(overlay_agreement['pass_fail']),
            'disagreement_breakdown':dict(sorted(overlay_agreement['r3_disagreement'].items())),
            'policy_pass_r3_fail_cases':overlay_agreement['policy_pass_r3_fail_cases'],
            'definition':'R3 subset gate = overlay graph_match AND policy relative-bond gate AND overlay min-heavy distance gate; coordination, stray-H/F and other frozen policy terms remain policy-only.'},
        unmappable_surface_fit_codes=sorted({p['code'] for p in pool_records
                                             if p['status']=='excluded' and p.get('reason')=='unmappable'}),
        columnar=dict(physical_rows=main_meta['physical_input_rows'],stored_rows=sum(r['rows'] for r in main_meta['roots']),roots=main_meta['roots']),
        placebo_columnar=dict(physical_rows=placebo_meta['physical_input_rows'],stored_rows=sum(r['rows'] for r in placebo_meta['roots']),roots=placebo_meta['roots']),
        focused_root=focus,
        policy_version=spec['solver_artifact_controls']['version'],identity_checks=dict(identity_counts),qm_property_values_accessed=False,
        qm_csv_fields_used=['CSD_code','SMILES'],source_shard_hashes=provenance,loader_commit=loader_commit,
        peak_rss_mib=max_rss_mib())
    write_json(out/'reports/extraction.json',report)
    return report


def run_cv_check(columns_dir, spec, out):
    columns_dir=Path(columns_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metadata=load_json(columns_dir/'dataset.json');books=[]
    for rid,root in enumerate(metadata['roots']):
        rows=RootRows(columns_dir,rid)
        group_counts={y:len({rows.dataset['groups'][int(g)] for g in np.unique(rows.g[rows.y==y])}) for y in (0,1)}
        if min(group_counts.values())>=25: books.append((min(group_counts.values()),rid,root))
    if not books: raise RuntimeError('subset has no root with >=25 independent groups in both sources; increase subset size')
    _,rid,root=max(books,key=lambda x:x[0]);rows=RootRows(columns_dir,rid);mass=support_mass(rows,spec)
    start=time.perf_counter();exact_surface,exact_cv=grouped_cv_root(rows,spec,mass_cache=mass,workers=4);exact_s=time.perf_counter()-start
    summary_dir=out/'binned';bin_meta=bin_root(rows,summary_dir,spec,8,4);summary=RootRows(summary_dir)
    start=time.perf_counter();binned_surface,binned_cv=grouped_cv_root(summary,spec,mass_cache=mass,workers=4);binned_s=time.perf_counter()-start
    losses_a=np.asarray([v['mean'] for v in exact_cv['results']]);losses_b=np.asarray([v['mean'] for v in binned_cv['results']])
    from scipy.stats import spearmanr
    error=supported_error(rows,exact_surface,binned_surface,seed=23,count=4096)
    result=dict(subset_root_id=rid,root=root['root'],family=root['family'],dimension=root['dimension'],
        physical_rows=rows.n,summary_rows=summary.n,independent_groups=len(rows.group_names),
        groups_by_source={str(y):len({rows.dataset['groups'][int(g)] for g in np.unique(rows.g[rows.y==y])}) for y in (0,1)},
        configurations=96,folds=5,exact_wall_s=exact_s,binned_wall_s=binned_s,
        exact_one_se_choice=exact_cv['selected']['config'],binned_one_se_choice=binned_cv['selected']['config'],
        same_one_se_choice=exact_cv['selected']['config']==binned_cv['selected']['config'],
        spearman_cv_loss=float(spearmanr(losses_a,losses_b).statistic),
        max_abs_cv_loss_delta=float(np.max(np.abs(losses_a-losses_b))),supported_probe_error=error,
        binned_policy=bin_meta['approximation'],peak_rss_mib=max_rss_mib(),cv_workers=4,
        conservative_cv_rss_upper_bound_mib=max_rss_mib()+sum(sorted(
            {w['pid']:w['peak_rss_mib'] for w in exact_cv['worker_peak_rss_mib']+binned_cv['worker_peak_rss_mib']}.values(),
            reverse=True)[:4]))
    write_json(out/'reports/cv_comparison.json',result)
    write_json(out/'reports/exact_cv.json',exact_cv);write_json(out/'reports/binned_cv.json',binned_cv)
    write_json(out/'surfaces/exact.json',exact_surface);write_json(out/'surfaces/binned.json',binned_surface)
    return result


def main():
    set_cpu();p=argparse.ArgumentParser();p.add_argument('--groups',type=int,default=240);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cv-only',action='store_true',help='fit from an already extracted subset without reloading parents')
    a=p.parse_args();out=a.out.resolve()
    if not out.is_relative_to(DEV.resolve()):raise ValueError('output must stay under data_dev')
    frame=load_json(DATA/'frame.json');manifest=load_json(DATA/'manifest.json');repair=load_json(DATA/'repair_s2_1/repair_manifest.json')
    spec=load_json(DATA/'spec.json');policy=load_json(POLICY_PATH)
    if policy['version']!='solver-artifact-controls.2':raise IntegrityError('frozen health/placebo policy version mismatch')
    spec['solver_artifact_controls']=policy
    if a.cv_only:
        start=time.perf_counter();cv=run_cv_check(out/'subset_columns/columns',spec,out/'cv_check')
        summary=dict(cv=cv,cv_wall_s=time.perf_counter()-start,peak_rss_mib=max_rss_mib())
        write_json(out/'reports/step1_checkpoint.json',summary)
        print(json.dumps(summary,indent=2),flush=True);return
    adapter=S21Adapter(DATA,MAIN,allow_audited_production=True)
    parents,decisions=resolve_cohort(frame,manifest,repair)
    groups=sorted({p['group_id'] for p in parents},key=lambda g:hashlib.sha256(str(g).encode()).hexdigest())[:a.groups]
    start=time.perf_counter()
    report=extract_columns(parents,adapter,spec,out/'subset_columns',group_subset=set(groups),focus_one_root=True,include_placebo=False)
    cv=run_cv_check(out/'subset_columns/columns',spec,out/'cv_check')
    summary=dict(cohort_resolved=len(parents),parents_in_subset=report['parents_loaded'],groups_in_subset=len(groups),
        occupant_counts=dict(collections.Counter(d['origin'] for d in decisions)),unmappable_surface_fit_codes=report['unmappable_surface_fit_codes'],
        extraction_seconds=time.perf_counter()-start,cv=cv,peak_rss_mib=max_rss_mib(),
        provenance=dict(policy_sha256=digest(POLICY_PATH),spec_sha256=digest(DATA/'spec.json'),
            frame_sha256=digest(DATA/'frame.json'),manifest_sha256=digest(DATA/'manifest.json'),repair_manifest_sha256=digest(DATA/'repair_s2_1/repair_manifest.json')))
    write_json(out/'reports/step1_checkpoint.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
