"""Bounded Step 4 GPU response labeling of frozen calibration motions."""
from __future__ import annotations

import hashlib, json, os, time
from collections import defaultdict
from pathlib import Path
import csv
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("JACOBIK_PROFILE", "u1")
os.environ.setdefault("JACOBIK_PROFILE_OVERRIDES", '{"nci":false,"deterministic_embed":true}')

import numpy as np
import torch

from jacobik.benchmark.energy import MACEPolarEvaluator, EV_TO_KJ_MOL
from research.archive.code.interactions.nci_v2.nci_v2 import enumerate_opportunities
from jacobik.stats.mapping import assemble
from jacobik.fit.parser import _assemble_raw, _iter_bo_blocks, _iter_xyz_blocks
from research.archive.code.interactions.nci_v2.nci_v2_extract import unique_opportunities
from research.archive.code.interactions.nci_v2.nci_v2_response import measure_motion, RawScorerBaseline, jsonable
from research.archive.code.refit.nci_v2_s2b_verify import load_npz


def safe_smiles_index(path):
    """Read only the code and SMILES CSV cells; never inspect QM properties."""
    out={}
    with Path(path).open(newline='') as fh:
        reader=csv.reader(fh,delimiter=';'); header=next(reader)
        ci,si=header.index('CSD_code'),header.index('SMILES')
        for row in reader: out[row[ci].strip()]=row[si].strip() or None
    return out


def labeled_motion_count(rows):
    """Count labeled paths by center identity, with legacy rows grouped once."""
    return len({(r['group_id'],r.get('motion_id'),r.get('center'),
                 r.get('center_id') if r.get('center_id') is not None else '__legacy_center__')
                for r in rows})


def safe_load_mol(code,shard,dataset,clean_root,smiles_index):
    base=Path(clean_root)/('lnQM' if dataset=='LNf' else 'tmQM')
    xyz=_iter_xyz_blocks(base/'xyz'/f'tmQM_{shard}.xyz')
    bos=_iter_bo_blocks(base/'bo'/f'tmQM_{shard}.BO')
    for (comment,atoms),(bo_code,bonds) in zip(xyz,bos):
        if bo_code==code:
            raw=_assemble_raw(comment,atoms,bo_code,bonds,
                SimpleNamespace(smiles=smiles_index.get(code)))
            mol=assemble(raw);mol.smiles=smiles_index.get(code)
            return mol
    return None


def safe_load_mol_shard(codes, shard, dataset, clean_root, smiles_index):
    """Read one shard once and retain only requested molecules.

    Calibration parents commonly share a shard. Reopening and reparsing that
    shard for every parent turns molecule setup into repeated O(shard-size)
    work. This bounded cache stores only the requested calibration structures.
    """
    wanted=set(codes); found={}
    base=Path(clean_root)/('lnQM' if dataset=='LNf' else 'tmQM')
    xyz=_iter_xyz_blocks(base/'xyz'/f'tmQM_{shard}.xyz')
    bos=_iter_bo_blocks(base/'bo'/f'tmQM_{shard}.BO')
    for (comment,atoms),(bo_code,bonds) in zip(xyz,bos):
        if bo_code not in wanted: continue
        raw=_assemble_raw(comment,atoms,bo_code,bonds,
            SimpleNamespace(smiles=smiles_index.get(bo_code)))
        mol=assemble(raw);mol.smiles=smiles_index.get(bo_code)
        found[bo_code]=mol
        if len(found)==len(wanted): break
    return found


def replay_group_rows(builder,seeds,solve_kw,MS):
    """Build the frozen control scorer without running seed-specific beam search.

    The labeler needs only the scorer; replaying ``MS.seed_rows`` generated and
    discarded beam-search starting poses. Mirror the scorer construction in
    ``ComplexBuilder._beam_search_all_candidates`` directly after preparation.
    """
    ctx,assignments,_W,_T=MS.prepare(builder,seeds[0],**solve_kw)
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    ge=GeometryExtractor(ctx.mol,ctx.device)
    scorer=Scorer(ctx.mol,ge,ctx.stats,device=ctx.device,
        quantity_weights=ctx.quantity_weights,assignments=assignments,
        donor_groups=[grp for (_,_,grp) in ctx.all_donor_groups],ideal=ctx.ideal,
        orientation_model=ctx.orientation_model,angle_bend_penalty=ctx.angle_bend_penalty)
    ctx.score_norm=float(scorer._agg_norm())
    return ctx,assignments,scorer


class CachedGPUBackend:
    """MACE-POLAR float64 GPU energies/forces, deduplicated by exact geometry."""
    def __init__(self, model_path, mem_fraction=.45):
        if not torch.cuda.is_available():
            raise RuntimeError("Step 4 labeling requires CUDA")
        torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)
        self.evaluator=MACEPolarEvaluator(model="polar-1-s",device="cuda",dtype="float64",
            model_path=model_path,mode="energy-only",threads=1)
        self.energies={}; self.forces={}; self.model_path=Path(model_path)
        self.calls=0; self.force_calls=0
        self.energy_seconds=0.; self.force_seconds=0.
    @staticmethod
    def key(symbols, x, charge, multiplicity):
        h=hashlib.sha256(np.asarray(x,dtype="<f8").tobytes()).hexdigest()
        return (tuple(symbols),int(charge),int(multiplicity),h)
    def evaluate(self,symbols,positions,charge,multiplicity,*,forces=False):
        xs=np.asarray(positions,dtype=np.float64); es=[]; fs=[]
        keys=[self.key(symbols,x,charge,multiplicity) for x in xs]
        missing={k:x for k,x in zip(keys,xs) if k not in self.energies}
        if missing:
            started=time.perf_counter()
            structures=[(symbols,x,charge,multiplicity) for x in missing.values()]
            values=self.evaluator.energies(structures)
            self.energies.update({k:float(v)*EV_TO_KJ_MOL for k,v in zip(missing,values)})
            self.calls+=len(missing)
            self.energy_seconds+=time.perf_counter()-started
        if forces:
            started=time.perf_counter()
            for k,x in zip(keys,xs):
                if k in self.forces: continue
                props=self.evaluator.properties(symbols,x,charge,multiplicity)
                self.energies[k]=float(props['energy'])*EV_TO_KJ_MOL
                self.forces[k]=np.asarray(props['forces'],dtype=np.float64)*EV_TO_KJ_MOL
                self.force_calls+=1
            self.force_seconds+=time.perf_counter()-started
        es=[self.energies[k] for k in keys]
        if forces: fs=[self.forces[k] for k in keys]
        return np.asarray(es),np.asarray(fs) if forces else None


def _bend_callback(motion, center_npz, pool, norm):
    from jacobik.ik.angle_bends import energy_prior
    cols=np.asarray(pool.get('bend_columns',np.empty((0,2))),dtype=np.int64).reshape(-1,2)
    if not len(cols) or float(np.asarray(pool.get('bend_scale',0.)).item())==0:
        return None
    offsets=np.asarray(center_npz.get('thetas_lig_offsets',pool.get('thetas_lig_offsets')),dtype=np.int64)
    if 'theta' in center_npz:
        theta=np.asarray(center_npz['theta'],dtype=np.float64)
        beta0=np.asarray([theta[offsets[li]+col] for li,col in cols],dtype=np.float64)
    else:
        beta0=np.zeros(len(cols),dtype=np.float64)
    free=np.asarray(pool['bend_free_deg'],dtype=np.float64)
    ks=torch.as_tensor(pool['bend_k'],dtype=torch.float64)
    scale=float(np.asarray(pool['bend_scale']).item())
    per_z=bool(np.asarray(pool['bend_per_z']).item())
    path=motion.get('bend_beta_path_rad',{})
    tgrid=np.asarray(motion['t'],dtype=float)
    fd=float(motion['fd_step']); ts=list(tgrid)+[-fd,fd]
    if motion.get('half_fd_positions') is not None: ts+=list(np.asarray([-fd/2,fd/2]))
    beta=np.tile(beta0,(len(ts),1))
    for j,(li,col) in enumerate(cols):
        vals=path.get(f'{int(li)}:{int(col)}')
        if vals is not None: beta[:,j]=np.interp(ts,tgrid,np.asarray(vals,dtype=float))
    beta=torch.as_tensor(beta,dtype=torch.float64)
    def callback(X):
        if len(X)!=len(beta): raise ValueError('bend path rows do not align with frozen motion')
        out=energy_prior(beta,ks,torch.as_tensor(free,dtype=torch.float64),scale)
        return out/float(norm) if per_z else out
    return callback


def run(frame_path,spec_path,clean_root,pools_dir,centers_dir,paths_dir,frozen_surface_path,
        db_path,model_path,out_path,*,mem_fraction=.45,max_motions=None):
    from jacobik.stats.stats import StatsLookup,load_db
    from research.archive.code.refit.nci_v2_orchestrator import build_builder_and_inputs,apply_arm_profile_live
    from tools.jdev import multiseed as MS
    from research.archive.code.interactions.nci_v2.nci_v2 import SurfaceArtifact

    frame=json.loads(Path(frame_path).read_text());spec=json.loads(Path(spec_path).read_text())
    parents=frame['main_allocation']['splits']['calibration']
    by_code={p['representative_code']:p for p in parents}
    smiles=safe_smiles_index(Path(clean_root)/'tmQM'/'tmQM_y.csv')
    db=json.loads(Path(db_path).read_text()); meta_p=Path(db_path).with_name(Path(db_path).stem+'.meta.json')
    stats=StatsLookup(load_db(db,meta=json.loads(meta_p.read_text()) if meta_p.exists() else None))
    frozen=json.loads(Path(frozen_surface_path).read_text())
    artifacts={f:SurfaceArtifact(**a) for f,a in frozen.get('surface_artifacts',frozen).items()}
    evaluator=CachedGPUBackend(model_path,mem_fraction)
    evaluator.evaluator.load();torch.cuda.synchronize();t0=time.perf_counter()
    apply_arm_profile_live('control')
    paths=sorted(Path(paths_dir).glob('*.json'))
    progress=Path(out_path).with_suffix('.progress.json')
    rows=json.loads(progress.read_text()).get('rows',[]) if progress.exists() else []
    done={(r['group_id'],r.get('motion_id'),r.get('center'),r['center_id'])
          for r in rows if r.get('center_id') is not None}
    # Older checkpoints omitted the per-pool center identity. Their label was
    # produced by the first eligible matching path in sorted path order; skip
    # that first occurrence once, then allow the other seed-specific centers.
    legacy_rows=defaultdict(list)
    for r in rows:
        if r.get('center_id') is None:
            legacy_rows[(r['group_id'],r.get('motion_id'),r.get('center'))].append(r)
    legacy_done=set(legacy_rows)
    scorers={}; opps_by_code={}; pool_cache={}; center_cache={}; mol_cache={}
    mol_codes_by_shard=defaultdict(set)
    for parent_code,parent in by_code.items():
        mol_codes_by_shard[(parent['dataset'],parent['representative_shard'])].add(parent_code)
    setup_seconds=measure_seconds=0.; response_profile={}
    n_done=0
    for path in paths:
        if max_motions is not None and n_done>=max_motions: break
        doc=json.loads(path.read_text()); center=doc.get('center',{}); code=center.get('code')
        if code not in by_code: continue
        pool_name=center.get('pool')
        reference_center=center.get('center')=='reference'
        if not pool_name and reference_center: pool_name=f'{code}_11_control.npz'
        if not pool_name: continue
        center_id=f"{center.get('center','unknown')}:{pool_name}"
        # Baseline centers identify their saved pose as ``*_baseline.npz``;
        # the aligned S2b state is stored under the corresponding control name.
        # It carries the exact pose/bend state needed to reconstruct the frozen
        # raw scorer. Reference centers use the original control-pool payload.
        pool_root=Path(pools_dir)
        if not reference_center:
            pool_root=pool_root.parent/'s2b_eval_state_v1'/'pools'
            pool_name=pool_name.replace('_baseline.npz','_control.npz')
        pool_path=pool_root/pool_name
        if not pool_path.exists(): continue
        seed=int(pool_name.split('_')[-2])
        center_path=Path(centers_dir)/(f'{code}_reference.npz' if reference_center else f'{code}_{seed}_baseline.npz')
        if not center_path.exists(): continue
        pool_key=(str(pool_path.parent.resolve()),pool_name)
        if pool_key not in pool_cache:
            pool_cache[pool_key]=load_npz(pool_path)
        pool=pool_cache[pool_key]
        center_key=(code,seed,reference_center)
        if center_key not in center_cache:
            with np.load(center_path,allow_pickle=False) as cz:
                center_cache[center_key]={k:np.array(cz[k]) for k in cz.files}
        center_npz=center_cache[center_key]
        if code not in scorers:
            parent=by_code[code]
            shard_key=(parent['dataset'],parent['representative_shard'])
            if shard_key not in mol_cache:
                setup_t0=time.perf_counter()
                mol_cache[shard_key]=safe_load_mol_shard(
                    mol_codes_by_shard[shard_key],parent['representative_shard'],
                    parent['dataset'],Path(clean_root),smiles)
                setup_seconds+=time.perf_counter()-setup_t0
            mol=mol_cache[shard_key].get(code)
            if mol is None: continue
            builder,_,err=build_builder_and_inputs(mol)
            if builder is None: raise RuntimeError(f'{code}: builder failed: {err}')
            seed_ids=[s for s in (11,12,13) if (Path(pools_dir)/f'{code}_{s}_control.npz').exists()]
            if not seed_ids: continue
            _ctx,_assignments,scorer=replay_group_rows(builder,seed_ids,
                dict(device='cpu',angle_bends=True,stats=stats),MS)
            scorers[code]=(mol,scorer)
            opps_by_code[code]=unique_opportunities(enumerate_opportunities(mol,spec))
        mol,scorer=scorers[code]
        if 'ref_symbols' in pool and 'charge' in pool and 'mult' in pool:
            symbols=[str(x) for x in pool['ref_symbols']]
            charge=int(np.asarray(pool['charge']).item()); mult=int(np.asarray(pool['mult']).item())
        elif reference_center:
            # A reference trajectory can be labeled even when this parent's
            # first saved pool is an error record with no chemistry payload.
            ref_mol=scorers[code][0]
            symbols=[str(a.symbol) for a in ref_mol.atoms]
            charge=int(ref_mol.charge); mult=int(ref_mol.spin_state)
            pool={}
        else:
            continue
        row=int(np.asarray(center_npz['pool_row_index']).item()) if 'pool_row_index' in center_npz else 0
        row_group=int(np.asarray(pool['asg'])[row]) if ('asg' in pool and not reference_center) else 0
        norm=float(scorer._agg_norm())
        pending=[]; queued=set(done)
        for motion in doc.get('motions',[]):
            if not (motion.get('selected') and motion.get('valid') and 'positions' in motion): continue
            if not motion.get('informative_roots_floored'): continue
            key3=(str(motion['group_id']),motion.get('motion_id'),motion.get('center'))
            dk=(*key3,center_id)
            if dk in queued: continue
            if key3 in legacy_done:
                legacy_done.remove(key3)
                for old_row in legacy_rows.pop(key3,[]):old_row['center_id']=center_id
                queued.add(dk)
                done.add(dk)
                continue
            if max_motions is not None and n_done+len(pending)>=max_motions: break
            pending.append(motion)
            queued.add(dk)
        # Queue all energy geometries for this center together. MACE inference
        # is fast once batches are full; per-motion calls left the GPU idle
        # between small 7–9-geometry batches and repeated Python dispatch.
        if pending:
            energy_positions=[]
            for motion in pending:
                X=np.asarray(motion['positions'],dtype=np.float64)
                fd=np.asarray(motion['fd_positions'],dtype=np.float64)
                pieces=[X,fd]
                if motion.get('half_fd_positions') is not None:
                    pieces.append(np.asarray(motion['half_fd_positions'],dtype=np.float64))
                energy_positions.append(np.concatenate(pieces,axis=0))
            evaluator.evaluate(symbols,np.concatenate(energy_positions,axis=0),charge,mult)
        for motion in pending:
            measure_t0=time.perf_counter()
            baseline=RawScorerBaseline(scorer,row_group=row_group,
                bend_reward=_bend_callback(motion,center_npz,pool,norm))
            records=measure_motion(motion,symbols,charge,mult,baseline,opps_by_code[code],
                artifacts,evaluator,schema=spec,include_cell_columns=False,profile=response_profile)
            for record in records:
                record['source_code']=code; record['center']=motion.get('center')
                record['center_id']=center_id
                if reference_center: record['baseline_class_assignment']='class_0_reference_proxy'
                rows.append(jsonable(record))
            done.add(dk);n_done+=1
            measure_seconds+=time.perf_counter()-measure_t0
            if n_done%5==0:
                progress.write_text(json.dumps(dict(rows=rows,completed_motions=labeled_motion_count(rows),new_this_run=n_done,
                    setup_seconds=setup_seconds,measure_seconds=measure_seconds,
                    energy_seconds=evaluator.energy_seconds,force_seconds=evaluator.force_seconds,
                    response_profile=response_profile),separators=(',',':'))+'\n')
                print(f'completed {n_done} motions; unique energies={evaluator.calls}; forces={evaluator.force_calls}; '
                      f'setup={setup_seconds:.1f}s measure={measure_seconds:.1f}s '
                      f'energy={evaluator.energy_seconds:.1f}s force={evaluator.force_seconds:.1f}s '
                      f'response={response_profile}',flush=True)
        if max_motions is not None and n_done>=max_motions: break
    progress.write_text(json.dumps(dict(rows=rows,completed_motions=labeled_motion_count(rows),new_this_run=n_done,
        setup_seconds=setup_seconds,measure_seconds=measure_seconds,
        energy_seconds=evaluator.energy_seconds,force_seconds=evaluator.force_seconds,
        response_profile=response_profile),separators=(',',':'))+'\n')
    elapsed=time.perf_counter()-t0
    result=dict(schema='nci_v2_step4_gpu_labels_v1',rows=rows,completed_motions=labeled_motion_count(rows),label_rows=len(rows),new_this_run=n_done,
        unique_energy_calls=evaluator.calls,unique_force_calls=evaluator.force_calls,
        setup_seconds=setup_seconds,measure_seconds=measure_seconds,
        energy_seconds=evaluator.energy_seconds,force_seconds=evaluator.force_seconds,
        response_profile=response_profile,
        cache_energy_entries=len(evaluator.energies),cache_force_entries=len(evaluator.forces),
        elapsed_s=elapsed,gpu=evaluator.evaluator.describe(),mem_fraction=mem_fraction,
        model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        frozen_surface_sha256=hashlib.sha256(Path(frozen_surface_path).read_bytes()).hexdigest())
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    Path(out_path).write_text(json.dumps(result,separators=(',',':'))+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for x in ('frame','spec','clean-root','pools','centers','paths','frozen-surface','db','model','out'):p.add_argument('--'+x,required=True)
    p.add_argument('--max-motions',type=int);p.add_argument('--mem-fraction',type=float,default=.45)
    a=p.parse_args();r=run(a.frame,a.spec,a.clean_root,a.pools,a.centers,a.paths,a.frozen_surface,a.db,a.model,a.out,mem_fraction=a.mem_fraction,max_motions=a.max_motions)
    print(json.dumps({k:r[k] for k in ('completed_motions','new_this_run','unique_energy_calls','unique_force_calls','elapsed_s')}))
