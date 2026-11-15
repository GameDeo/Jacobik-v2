"""Read-only S2.1 adapter; synthetic/dev inputs only unless audited explicitly.

Schema authority: S2_1_OVERLAY_SCHEMA.md supplied by the user. This module never
writes to a source pool. Geometry access never opens MACE JSON or energy arrays.
"""
from __future__ import annotations
import hashlib,json,re,zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from research.archive.code.interactions.nci_v2.nci_v2_devio import DEV
from research.archive.code.interactions.nci_v2.nci_v2_poolio import checked_mapping,file_sha256

SCHEMA='s2_1_overlay.1'
CONVENTION='X[:, gen_to_ref_idx] is in reference atom order; symbols[gen_to_ref_idx] == ref_symbols'
REPAIRS={'R1_mace','R2_mapping','R3_health'}
META={'schema_version','source_pool_sha256','source_summary_sha256','repair_commit','repairs'}
R2={'gen_to_ref_idx','ref_to_gen_idx','mapping_ambiguous','num_automorphisms','coord_intact','coord_sigma_shift_A','coord_haptic_slip_A','coord_haptic_shift_A'}
R1={'energies_kj','ref_energy_kj','top1_idx','floor_idx','top1_kj','floor_kj','top1_dE_kj','floor_dE_kj'}
R3={'max_bond_dev_A','min_heavy_nonbonded_A','graph_match','max_nn_dist_A','health_radii_json'}

class IntegrityError(ValueError):pass
class ExcludedPool(ValueError):pass
class EnergyUnavailable(ValueError):pass


def sha(path,expected):
    try:actual=file_sha256(path)
    except OSError as exc:raise IntegrityError(f'missing or unreadable hashed input: {path.name}') from exc
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected) or actual!=expected:
        raise IntegrityError(f'sha256 mismatch: {path.name}')


def commit(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{40}',value):raise IntegrityError('invalid repair/code commit')


def headers(path):
    """Read NPZ headers only: reject object overlays without reading energy values."""
    result={}
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.endswith('.npy') or '/' in member:raise IntegrityError('invalid overlay member')
            with z.open(member) as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                else:raise IntegrityError('unsupported NPY overlay header version')
                if dtype.hasobject:raise IntegrityError('object array in overlay')
                name=member[:-4]
                if name in result:raise IntegrityError('duplicate overlay member')
                result[name]=(shape,dtype)
    return result


def typed(h,key,shape,kind):
    if key not in h or h[key][0]!=shape:raise IntegrityError(f'wrong/missing shape: {key}')
    dtype=h[key][1]
    if kind=='str':ok=dtype.kind=='U'
    elif kind=='bool':ok=dtype.kind=='b'
    elif kind=='int64':ok=dtype==np.dtype('int64')
    else:ok=dtype==np.dtype('float64')
    if not ok:raise IntegrityError(f'wrong dtype: {key}')


def finite(a,name,positive=False,allow_inf=False):
    a=np.asarray(a)
    if np.isnan(a).any() or (not allow_inf and not np.isfinite(a).all()) or (positive and np.any(a<0)):
        raise IntegrityError(f'invalid values: {name}')
    return a


def strip_heavy(value):
    if isinstance(value,dict):
        if 'n_heavy' in value and (type(value['n_heavy']) is not int or value['n_heavy']<0):raise IntegrityError('invalid derived n_heavy')
        return {k:strip_heavy(v) for k,v in value.items() if k!='n_heavy'}
    if isinstance(value,list):return [strip_heavy(v) for v in value]
    return value


class S21Adapter:
    def __init__(self,run_root,repo_root,*,allow_audited_production=False):
        self.root=Path(run_root).resolve();self.repo=Path(repo_root).resolve()
        if not allow_audited_production and not self.root.is_relative_to(DEV.resolve()):
            raise PermissionError('real S2.1 inputs remain disabled until audited authorization')
        self.index_path=self.root/'repair_s2_1/repair_manifest.json'
        self.index=self._json(self.index_path)
        if self.index.get('schema_version')!=SCHEMA or self.index.get('mapping_convention')!=CONVENTION:raise IntegrityError('index schema/convention mismatch')
        commit(self.index.get('repair_commit'))
        for key in ('pools','retries','slots','unmappable'):
            if not isinstance(self.index.get(key),dict):raise IntegrityError(f'invalid index {key}')
        frame=self.index['frame'];frozen=self.path(frame['frozen_path']);derived=self.path(frame['derived_path'])
        sha(frozen,frame['frozen_sha256']);sha(derived,frame['derived_sha256'])
        a=self._json(frozen);b=self._json(derived)
        if strip_heavy(a)!=strip_heavy(b):raise IntegrityError('derived frame changes more than n_heavy')
        self.heavy={}
        def collect(x):
            if isinstance(x,dict):
                code=x.get('code',x.get('representative_code'))
                if code is not None and 'n_heavy' in x:
                    if code in self.heavy and self.heavy[code]!=x['n_heavy']:raise IntegrityError('conflicting n_heavy')
                    self.heavy[code]=x['n_heavy']
                for v in x.values():collect(v)
            elif isinstance(x,list):
                for v in x:collect(v)
        collect(b)
        self._tasks=self._json(self.root/'manifest.json')['tasks']
        # S2.1 retries were re-emitted after the OOM fix as retry_pools_v2.
        # Older synthetic fixtures and early repair layouts use retry_pools.
        # Resolve the exact retry root from the index when supplied, and keep
        # all subsequent path checks bounded to that root.
        retry_dirs={v.get('retry_dir') for v in self.index['retries'].values() if v.get('retry_dir')}
        retry_paths={Path(v['pool_path']).parts[1] for v in self.index['retries'].values()
                     if isinstance(v.get('pool_path'),str) and len(Path(v['pool_path']).parts)>1
                     and Path(v['pool_path']).parts[0]=='repair_s2_1'}
        if len(retry_dirs)>1 or len(retry_paths)>1:raise IntegrityError('inconsistent retry directory')
        retry_dir=next(iter(retry_dirs or retry_paths), 'retry_pools')
        if retry_paths and retry_dir not in retry_paths:raise IntegrityError('retry directory disagrees with pool paths')
        if not re.fullmatch(r'retry_pools(?:_v2)?',retry_dir):raise IntegrityError('invalid retry directory')
        self.retry_root=(self.root/'repair_s2_1'/retry_dir).resolve()
        if not self.retry_root.is_relative_to(self.root):raise IntegrityError('retry root escapes run root')
        self.unused=set();self.replaced=set()
        self.restored=set()
        for code,slot in self.index['slots'].items():
            if slot.get('state') not in ('restored','still_failed','not_retried') or slot.get('replacement_state') not in ('unused_after_retry','in_use'):raise IntegrityError('invalid slot state')
            if slot['replacement_state']=='unused_after_retry':
                if slot['state']!='restored':raise IntegrityError('unused replacement without restored slot')
                self.unused.add(slot['replacement_code'])
                self.restored.add(code)
            elif slot['state']=='still_failed':self.replaced.add(code)
            elif slot['state']=='not_retried':self.replaced.add(code)
            else:raise IntegrityError('restored slot still uses reserve replacement')

    def _json(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root):raise IntegrityError('JSON path escapes run root')
        with path.open() as f:return json.load(f)

    def path(self,value):
        p=Path(value)
        if '..' in p.parts:raise IntegrityError('path traversal')
        if not p.is_absolute():p=(self.repo/p) if p.parts[:2]==('data','refit') else self.root/p
        p=p.resolve()
        if not p.is_relative_to(self.root):raise IntegrityError('path escapes run root')
        return p

    def keys(self):return sorted(set(self._tasks)|set(self.index['retries']))

    def reference_identity(self,key):
        """Return only the stored reference identity for an explicitly excluded original pool."""
        if key not in self.index['unmappable']:raise ExcludedPool('reference_identity_only_for_unmappable')
        task=self._tasks.get(key);entry=self.index['pools'].get(key)
        if not task or task.get('status')!='success' or entry is None:raise IntegrityError('unmappable pool lacks successful indexed source')
        path=self.path(task['pool_path']);sha(path,entry['source_pool_sha256'])
        sha(self.path(task['summary_path']),entry['source_summary_sha256'])
        with np.load(path,allow_pickle=False) as z:
            if 'error' in z.files:raise IntegrityError('success points to failed placeholder')
            sy=np.asarray(z['ref_symbols']);ref=np.asarray(z['ref_X'])
            if sy.ndim!=1 or ref.shape!=(len(sy),3):raise IntegrityError('invalid stored reference identity')
            return sy,ref

    def open(self,key,*,step2_original=False):
        match=re.fullmatch(r'([^:]+):([0-9]+):control',key)
        if not match:raise IntegrityError('unsupported task key/arm')
        code,seed=match[1],int(match[2])
        if code in self.unused:raise ExcludedPool('unused_after_retry')
        if code in self.replaced:raise ExcludedPool('replaced_failed_slot')
        original=self._tasks.get(key);retry=self.index['retries'].get(key)
        overlay=None
        if step2_original:
            if code not in self.restored:raise IntegrityError('Step 2 original override is only valid for restored slots')
            if not original or original.get('status')!='success':raise ExcludedPool('task_not_successful')
            if key in self.index['unmappable']:raise ExcludedPool('unmappable')
            task=original;entry=self.index['pools'].get(key)
            if entry is None:raise IntegrityError('successful original missing overlay index')
            if task.get('split')!=entry['split'] or task.get('stratum')!=entry['stratum']:raise IntegrityError('overlay split/stratum mismatch')
            path=self.path(task['pool_path']);sha(path,entry['source_pool_sha256'])
            sha(self.path(task['summary_path']),entry['source_summary_sha256'])
            overlay=self.path(entry['overlay_path']);sha(overlay,entry['overlay_sha256']);origin='step2_original_sensitivity'
        else:
            if code in self.restored and (retry is None or retry.get('status')!='success'):
                raise IntegrityError('restored slot missing successful retry')
            if retry is not None and retry.get('status') not in ('success','oom','failed'):raise IntegrityError('invalid retry status')
        if not step2_original and retry is not None and retry['status']=='success':
            commit(retry['code_commit']);path=self.path(retry['pool_path']);sha(path,retry['pool_sha256'])
            retry_manifest=self.retry_root/'manifest.json'
            if retry_manifest.exists():
                retry_tasks=self._json(retry_manifest)['tasks'];task=retry_tasks.get(key)
                if not task or task.get('status')!='success' or self.path(task['pool_path'])!=path:raise IntegrityError('retry record/status mismatch')
                if task.get('split')!=retry['split']:raise IntegrityError('retry split mismatch')
                for item in ('pool_path','summary_path','mace_path'):
                    if not self.path(task[item]).is_relative_to(self.retry_root):raise IntegrityError('retry artifact points outside retry output')
            else:
                # retry_pools_v2 is indexed directly by repair_manifest.json;
                # these retries intentionally have no independent sidecar manifest.
                task=dict(original or {})
                task.update(code=code,seed=seed,arm='control',status='success',split=retry['split'],
                            stratum=retry.get('stratum',task.get('stratum')),
                            pool_path=retry['pool_path'],provenance=dict(task.get('provenance',{}),code_git_commit=retry['code_commit']))
                task.pop('n_confs',None);task.pop('confs_sha256',None)
            if original and task.get('group_id')!=original.get('group_id'):raise IntegrityError('retry group changed')
            if task.get('provenance',{}).get('code_git_commit')!=retry['code_commit']:raise IntegrityError('retry commit mismatch')
            origin='retry'
        elif not step2_original:
            if not original or original.get('status')!='success':raise ExcludedPool('task_not_successful')
            if key in self.index['unmappable']:raise ExcludedPool('unmappable')
            task=original;entry=self.index['pools'].get(key)
            if entry is None:raise IntegrityError('successful original missing overlay index')
            if task.get('split')!=entry['split'] or task.get('stratum')!=entry['stratum']:raise IntegrityError('overlay split/stratum mismatch')
            path=self.path(task['pool_path']);sha(path,entry['source_pool_sha256'])
            sha(self.path(task['summary_path']),entry['source_summary_sha256'])
            overlay=self.path(entry['overlay_path']);sha(overlay,entry['overlay_sha256']);origin='original'
        if task.get('code')!=code or task.get('seed')!=seed or task.get('arm')!='control':raise IntegrityError('task identity mismatch')
        if task.get('split') not in ('surface_fit','calibration','motion_validation'):raise IntegrityError('unknown split')
        return PoolView(self,key,task,path,origin,overlay)


class PoolView:
    def __init__(self,adapter,key,task,path,origin,overlay):
        self.adapter=adapter;self.key=key;self._task=task;self.path=path;self.origin=origin;self.overlay=overlay;self.repairs=set();self.overlay_keys=set()
        self.pool_sha=adapter.index['retries'][key]['pool_sha256'] if origin=='retry' else adapter.index['pools'][key]['source_pool_sha256']
        with np.load(path,allow_pickle=False) as z:
            if 'error' in z.files:raise IntegrityError('success points to failed placeholder')
            self.X=np.asarray(z['X']);self.symbols=np.asarray(z['symbols']);self.ref_symbols=np.asarray(z['ref_symbols']);self.ref_X=np.asarray(z['ref_X'])
            if self.X.ndim!=3 or self.X.shape[2]!=3 or len(self.symbols)!=self.X.shape[1] or self.ref_X.shape!=(len(self.ref_symbols),3):raise IntegrityError('pool coordinate shape mismatch')
            self.P,self.N=self.X.shape[:2];self.original_unmapped='atom_mapping_error' in z.files or 'gen_to_ref_idx' not in z.files
            if task.get('n_confs') is not None and self.P!=task['n_confs']:raise IntegrityError('conformer count mismatch')
            if task.get('confs_sha256') is not None and task['confs_sha256']!=hashlib.sha256(self.X.tobytes()).hexdigest():raise IntegrityError('confs_sha256 mismatch')
            self.charge=int(z['charge']);self.multiplicity=int(z['mult']);self.health={};self.coordination={}
            if origin=='retry' and str(z['code_git_commit'])!=adapter.index['retries'][key]['code_commit']:raise IntegrityError('retry pool commit mismatch')
            if overlay is not None:self._validate_overlay()
            def value(k):
                if k in self.overlay_keys:
                    with np.load(self.overlay,allow_pickle=False) as o:return np.asarray(o[k])
                if k not in z.files:raise IntegrityError(f'missing pool field {k}')
                return np.asarray(z[k])
            if self.original_unmapped and 'R2_mapping' not in self.repairs:raise ExcludedPool('unmapped_without_R2')
            self.g=checked_mapping(self.symbols,self.ref_symbols,value('gen_to_ref_idx'))
            inverse=value('ref_to_gen_idx')
            if inverse.dtype.kind not in 'iu' or not np.array_equal(inverse,np.argsort(self.g)):raise IntegrityError('inverse mapping mismatch')
            for k in ('coord_intact','coord_sigma_shift_A','coord_haptic_slip_A','coord_haptic_shift_A'):
                a=value(k)
                if a.shape!=(self.P,):raise IntegrityError(f'coordination shape mismatch {k}')
                if k=='coord_intact' and a.dtype.kind!='b':raise IntegrityError('coord_intact dtype mismatch')
                if k!='coord_intact':finite(a,k,positive=True)
                self.coordination[k]=a
            if 'R3_health' in self.repairs:
                for k in R3:self.health[k]=value(k)
            self.group_id=task['group_id'];self.split=task['split'];self.stratum=task['stratum'];self.code=task['code'];self.seed=task['seed'];self.n_heavy=adapter.heavy.get(task['code'])
            self.health_source='R3_overlay' if 'R3_health' in self.repairs else 'recompute_from_retry_geometry'
            if self.n_heavy is None:raise IntegrityError('missing frame-derived n_heavy')

    def _validate_overlay(self):
        a=self.adapter;entry=a.index['pools'][self.key];h=headers(self.overlay)
        for k in META:typed(h,k,(),'str')
        with np.load(self.overlay,allow_pickle=False) as z:
            if str(z['schema_version'])!=SCHEMA or str(z['repair_commit'])!=a.index['repair_commit']:raise IntegrityError('overlay version/commit mismatch')
            for k in ('source_pool_sha256','source_summary_sha256'):
                if str(z[k])!=entry[k]:raise IntegrityError('overlay/index source sha mismatch')
            repairs=json.loads(str(z['repairs']))
            if not isinstance(repairs,list) or len(set(repairs))!=len(repairs) or not set(repairs)<=REPAIRS or not isinstance(entry['repairs'],list) or len(set(entry['repairs']))!=len(entry['repairs']) or set(repairs)!=set(entry['repairs']):raise IntegrityError('repairs mismatch')
            self.repairs=set(repairs);expected=META|set().union(*(fields for r,fields in [('R1_mace',R1),('R2_mapping',R2),('R3_health',R3)] if r in self.repairs))
            if set(h)!=expected:raise IntegrityError('overlay fields do not match repairs')
            if 'R3_health' not in self.repairs:raise IntegrityError('mappable original requires R3_health')
            if 'R2_mapping' in self.repairs:
                for k in ('gen_to_ref_idx','ref_to_gen_idx'):typed(h,k,(self.N,),'int64')
                typed(h,'mapping_ambiguous',(),'bool');typed(h,'num_automorphisms',(),'int64')
                typed(h,'coord_intact',(self.P,),'bool')
                for k in ('coord_sigma_shift_A','coord_haptic_slip_A','coord_haptic_shift_A'):typed(h,k,(self.P,),'float64');finite(z[k],k,positive=True)
            if 'R3_health' in self.repairs:
                typed(h,'graph_match',(self.P,),'bool');typed(h,'health_radii_json',(),'str')
                if not isinstance(json.loads(str(z['health_radii_json'])),dict):raise IntegrityError('invalid radii JSON')
                for k in ('max_bond_dev_A','min_heavy_nonbonded_A','max_nn_dist_A'):
                    typed(h,k,(self.P,),'float64');finite(z[k],k,positive=True,allow_inf=k=='min_heavy_nonbonded_A')
            if 'R1_mace' in self.repairs:
                if self._task['split']=='surface_fit':raise IntegrityError('surface_fit cannot carry R1 energy labels')
                typed(h,'energies_kj',(self.P,),'float64')
                for k in R1-{'energies_kj'}:typed(h,k,(),'int64' if k.endswith('_idx') else 'float64')
        self.overlay_keys=set(h)

    def _verify_sources(self):
        sha(self.path,self.pool_sha)
        if self.overlay is not None:
            entry=self.adapter.index['pools'][self.key]
            sha(self.overlay,entry['overlay_sha256'])
            sha(self.adapter.path(self._task['summary_path']),entry['source_summary_sha256'])

    @property
    def coordinates(self):
        self._verify_sources();return self.X[:,self.g,:]

    def labels(self,keys):
        allowed={'score','rank','Z','asg',*[f'terms_Q{i}' for i in range(1,7)]}
        if not set(keys)<=allowed:raise EnergyUnavailable('energy labels require explicit energies() access')
        self._verify_sources()
        with np.load(self.path,allow_pickle=False) as z:result={k:np.asarray(z[k]) for k in keys}
        for k,v in result.items():
            if v.shape!=(() if k=='Z' else (self.P,)):raise IntegrityError(f'label shape mismatch {k}')
            finite(v,k)
            if k=='Z' and float(v)<=0:raise IntegrityError('nonpositive normalizer')
        return result

    def energies(self,*,authorized=False):
        if not authorized:raise EnergyUnavailable('energy access not authorized')
        self._verify_sources()
        if self.split=='surface_fit':raise EnergyUnavailable('surface_fit MACE values are placeholders, not energies')
        if self.origin=='original' and self.original_unmapped and 'R1_mace' not in self.repairs:raise EnergyUnavailable('R1 required for originally unmapped energies')
        if 'R1_mace' in self.repairs:
            with np.load(self.overlay,allow_pickle=False) as z:d={k:np.asarray(z[k]) for k in R1}
        else:
            d=self.adapter._json(self.adapter.path(self._task['mace_path']))
            for k in ('code','seed','arm'):
                if d.get(k)!=self._task[k]:raise IntegrityError('MACE identity mismatch')
            d={k:np.asarray(d[k]) for k in ('energies_kj','ref_energy_kj','top1_idx','floor_idx')}
        e=finite(d['energies_kj'],'energies_kj');ref=float(d['ref_energy_kj']);finite(ref,'ref_energy_kj')
        if e.shape!=(self.P,):raise IntegrityError('energy shape mismatch')
        for key in ('top1_idx','floor_idx'):
            if np.asarray(d[key]).shape!=() or np.asarray(d[key]).dtype.kind not in 'iu':raise IntegrityError('energy index dtype mismatch')
        top=int(d['top1_idx']);floor=int(d['floor_idx'])
        if not 0<=top<self.P or floor!=int(np.argmin(e)):raise IntegrityError('energy indices mismatch')
        score=self.labels(['score'])['score']
        if top!=self._task['top1_idx'] or top!=int(np.argmax(score)):raise IntegrityError('score-based top1 index mismatch')
        expected=dict(top1_kj=e[top],floor_kj=e[floor],top1_dE_kj=e[top]-ref,floor_dE_kj=e[floor]-ref)
        for k,v in expected.items():
            observed=d.get(k,self._task.get(k))
            if observed is None or not np.isclose(float(observed),v,rtol=1e-12,atol=1e-8):raise IntegrityError(f'energy summary mismatch {k}')
        return dict(energies_kj=e,ref_energy_kj=ref,top1_idx=top,floor_idx=floor,**expected)
