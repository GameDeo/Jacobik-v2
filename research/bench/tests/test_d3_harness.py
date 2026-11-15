"""Registered gates, identity checks and once-only locked-run safeguards."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from tools.jgen.d3.analyse import energy_gate, curvature_gate, holdout_gate, pick, interaction
from tools.jgen.d3.arms import db_identity, validate, PRODUCTION, SEARCH
from tools.jgen.d3.panels import stage3, locked_path
from tools.jgen.d3.run import verify_fetch, ablations
from tools.jgen.c2.analyse import guard


def metrics(top, floor):return dict(top1={'ci':[top-1,top]},floor={'ci':[floor-1,floor]})


@pytest.mark.parametrize('top,floor,passed',[(1.999,-.001,True),(2,-1,False),(0,0,False),(-.001,1,True),(1,0,False)])
def test_energy(top,floor,passed):assert energy_gate(metrics(top,floor))==passed


def curvature(median=1.3,rank=.85):return dict(complete=True,scope='Q1-Q7',q7_scorer_in_scope=True,fk_prior_settings={"bend_penalty":"energy"},median_ratio=median,rank_agreement=rank)


@pytest.mark.parametrize('median,rank,passed',[(1.15,.8,True),(1.4,.8,True),(1.149,.9,False),(1.401,.9,False),(1.3,.799,False)])
def test_curvature(median,rank,passed):assert curvature_gate(curvature(median,rank),curvature())==passed



def test_missing_curvature():assert not curvature_gate({})


@pytest.mark.parametrize('top,a,c,passed',[(1.99,2,2,True),(2,1,2,False),(0,3,2,False),(0,None,2,False)])
def test_holdout(top,a,c,passed):assert holdout_gate(metrics(top,0),a,c)==passed


def test_pick():
    x={n:dict(qualified=True,metrics=dict(top1=dict(mean=-2))) for n in ('D','DS')}
    assert pick(x)=='D';x['DS']['metrics']['top1']['mean']=-3;assert pick(x)=='DS'
    x['DS']['qualified']=False;assert pick(x)=='D'
    x['D']['qualified']=False;assert pick(x) is None


def test_interaction_ci():
    c={n:dict(per_parent={p:{k:dict(delta=v) for k in ('floor','top1')} for p in ('A','B','C')})
       for n,v in [('D',-2),('DS',-5),('S',-1)]}
    r=interaction(c,dict(primary_exclusions=[],bootstrap_seed=11,bootstrap_resamples=10000))
    assert r['top1']==dict(n=3,mean=-2.,ci=[-2.,-2.])


def test_guard_edges():
    s=dict(regression_threshold_kj=25,regression_min_seeds=2,regression_null_margin=2)
    assert guard([[25]*3]*3,[[0]*3]*3,s)['passed']
    assert guard([[26]*3]*2,[[0]*3]*2,s)['passed']
    assert not guard([[26]*3]*3,[[0]*3]*3,s)['passed']


def provenance(tmp_path):
    db=tmp_path/'db.json';db.write_text('{}')
    db.with_suffix('.meta.json').write_text(json.dumps(dict(code_commit='a'*40,flags={},input_hashes={'raw':'b'*64})))
    return db


def test_sidecar(tmp_path):
    db=tmp_path/'db.json';db.write_text('{}')
    with pytest.raises(FileNotFoundError):db_identity(db)
    db.with_suffix('.meta.json').write_text('{}')
    with pytest.raises(ValueError):db_identity(db)
    provenance(tmp_path);assert len(db_identity(db)['db_sha256'])==64


def config(tmp_path):
    db=provenance(tmp_path);model=tmp_path/'model';model.write_text('model')
    return dict(mace_path=str(model),arms={n:dict(db=str(db),flags=f,fingerprint=fp) for n,f,fp in
        [('C',{},PRODUCTION),('Cprime',{},PRODUCTION),('S',SEARCH,'ddca4fedde5b'),('D',{},PRODUCTION),('DS',SEARCH,'ddca4fedde5b')]})


def test_identity(tmp_path):
    c=config(tmp_path);assert validate(c)['S']['fingerprint']=='ddca4fedde5b'
    other=tmp_path/'other.json';other.write_text('{"changed":true}');other.with_suffix('.meta.json').write_text('{}')
    c['arms']['S']['db']=str(other)
    with pytest.raises(ValueError,match='identity mismatch'):validate(c)


def test_placeholder(tmp_path):
    c=config(tmp_path);c['arms']['D']['flags']=None
    with pytest.raises(ValueError,match='placeholders'):validate(c)


def locked_fixture(tmp_path):
    codes=tmp_path/'codes';codes.write_text('\n'.join(f'C{i:05}' for i in range(80)))
    analysis=tmp_path/'analysis.json';analysis.write_text(json.dumps(dict(stage='stage1',label='D3',candidate='D',comparisons={'D':{'qualified':True}})))
    run=tmp_path/'run';run.mkdir()
    return codes,hashlib.sha256(codes.read_bytes()).hexdigest(),analysis,run


def test_locked_refusals(tmp_path):
    args=locked_fixture(tmp_path)
    with pytest.raises(ValueError,match='SHA256'):stage3(args[0],'0'*64,*args[2:])
    with pytest.raises(ValueError,match='OWNER_GO'):stage3(*args)
    (args[3]/'OWNER_GO').touch();assert len(stage3(*args)[0])==80
    args[2].write_text(json.dumps(dict(stage='stage1',label='D3',candidate=None)))
    with pytest.raises(ValueError,match='qualified'):stage3(*args)
    with pytest.raises(ValueError,match='forbidden'):locked_path(tmp_path/'.claude/locked/codes')


def test_fetch_count_and_hash(tmp_path):
    p=tmp_path/'C/records/a.json';p.parent.mkdir(parents=True);p.write_text('{}')
    (tmp_path/'completion.json').write_text(json.dumps(dict(expected_records=1,actual_records=1,files={'C/records/a.json':hashlib.sha256(p.read_bytes()).hexdigest()})))
    assert verify_fetch(tmp_path)['passed'];p.write_text('bad')
    with pytest.raises(ValueError):verify_fetch(tmp_path)


def test_ablation_refusal():
    with pytest.raises(ValueError):ablations({},dict(stage='stage1',label='DRYRUN'))


def test_lifecycle_fetch_before_down_after_error(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.jgen.d3 import run
    events=[]
    def colab(args,**kwargs):
        if args[0]=='sessions':return ''
        if args[0]=='new':events.append('new');return ''
    def fail(*args,**kwargs):raise RuntimeError('upload failed')
    cj=SimpleNamespace(colab=colab,guard_credits=lambda:None,build_bundle=lambda *a:('bundle','commit'),
        push=fail,cmd_get=lambda a:events.append('fetch'),cmd_down=lambda a:events.append('down'))
    monkeypatch.setattr(run,'load_cj',lambda:cj)
    monkeypatch.setattr(run,'bundle_plan',lambda *a:'bundle')
    with pytest.raises(RuntimeError,match='upload'):
        run.launch(dict(codes=['A'],arms=['C'],seeds=[11]),tmp_path/'run','own',1)
    assert events==['new','fetch','down']


def test_existing_vm_is_not_stopped(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools.jgen.d3 import run
    events=[]
    monkeypatch.setattr(run,'load_cj',lambda:SimpleNamespace(colab=lambda *a,**k:'own',cmd_down=lambda a:events.append('down')))
    with pytest.raises(ValueError,match='already exists'):run.launch({},tmp_path/'run','own',1)
    assert not events


@pytest.mark.parametrize('ratio,passed',[(None,False),(1.3,True),(1.30001,False)])
def test_wall(ratio,passed):
    from tools.jgen.d3.analyse import wall_gate
    assert wall_gate(ratio)==passed


@pytest.mark.parametrize('trips,mirrors,null,passed',[(3,0,1,True),(4,0,1,False),(4,5,1,True),(None,0,1,False)])
def test_stage3_own_null(trips,mirrors,null,passed):
    from tools.jgen.d3.analyse import own_null_gate
    assert own_null_gate(dict(trips=trips,mirror=mirrors),dict(trips=null))==passed


def test_catastrophic_threshold_and_failures(tmp_path):
    from tools.jgen.d3.analyse import extra_counts
    for name,top in [('D',200.001),('C',200)]:
        p=tmp_path/name/'records/A_11.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(status='ok',top1_dE=top,d3_geometry=dict(edge_on_rows=2))))
    r=extra_counts(tmp_path,'D','C',['A'],[11])
    assert not r['catastrophic']['passed'];assert r['catastrophic']['arm']==1
    (tmp_path/'D/records/A_11.json').unlink()
    assert extra_counts(tmp_path,'D','C',['A'],[11])['asymmetric_failures']==[['A',11]]


def test_failed_parent_remains_in_coverage_denominator(tmp_path):
    from tools.jgen.d3.comparison import compare
    s=dict(seeds=[11,12,13],rho_draws=20,rho_seed=11,bootstrap_seed=11,bootstrap_resamples=10000,
        primary_exclusions=[],guard_artifacts=[],regression_threshold_kj=25,regression_min_seeds=2,regression_null_margin=2)
    r=compare(tmp_path,'D','D',s,codes=['MISSING'],proof=True)
    assert r['metrics']['hit1']['n']==1 and r['metrics']['hit1']['mean']==0
    assert r['failures']==['MISSING']


def make_records(root,names,codes,values):
    for name in names:
        for code in codes:
            for seed in (11,12,13):
                p=root/name/'records'/f'{code}_{seed}.json';p.parent.mkdir(parents=True,exist_ok=True)
                top=values[name]
                p.write_text(json.dumps(dict(status='ok',top1_dE=top,floor_dE=top-1,n_rows=2,
                    scores=[2.,1.],energies_kj=[top,top+1],coverage={'rmsd_A':[.1,.2]},
                    c2_timing={'end_to_end_s':2},c2_metrics={'minimum_shell_rmsd_A':.1},
                    d3_geometry={'edge_on_rows':0})))


def test_stage1_end_to_end_and_dryrun_never_qualifies(tmp_path):
    from tools.jgen.d3.analyse import analyse
    names=['C','D','S','DS'];codes=['A','B']
    make_records(tmp_path,names,codes,dict(C=100,D=90,S=100,DS=90))
    manifest=dict(codes=codes,seeds=[11,12,13],arms=names,label='D3',curvature={n:curvature() for n in names})
    r=analyse(tmp_path,'stage1',manifest)
    assert r['candidate']=='D'
    assert r['contrasts']['interaction']['top1']['ci']==[0.,0.]
    assert all(r['comparisons']['DS']['gates'].values())
    manifest['label']='DRYRUN';assert analyse(tmp_path,'stage1',manifest)['candidate'] is None


def test_stage3_uses_measured_control_repeat(tmp_path):
    from tools.jgen.d3.analyse import analyse
    names=['C','Cprime','D'];codes=['A','B','C']
    make_records(tmp_path,names,codes,dict(C=100,Cprime=130,D=130))
    r=analyse(tmp_path,'stage3',dict(codes=codes,seeds=[11,12,13],arms=names,label='D3'))
    assert r['comparisons']['D']['own_control_null']['trips']==3
    assert r['comparisons']['D']['gates']['guard']
    assert not r['comparisons']['D']['gates']['top1']


def test_ablation_order_and_complete_recipes():
    from tools.jgen.d3.run import ABLATION_ORDER
    c=dict(ablation_groups={g:dict(flags={},db='db',fingerprint='fp') for g in reversed(ABLATION_ORDER)})
    analysis=dict(stage='stage1',label='D3',candidate=None,comparisons={'D':{'qualified':False}})
    assert [r['group'] for r in ablations(c,analysis)]==ABLATION_ORDER
    c['ablation_groups']['typing_os']=None
    with pytest.raises(ValueError,match='typing_os'):ablations(c,analysis)


@pytest.mark.parametrize('median,rank,passed',[(1.5,.82,True),(1.7,.82,True),(1.49999,.85,False),(1.70001,.85,False),(1.6,.81999,False)])
def test_relative_curvature_gate(median,rank,passed):
    assert curvature_gate(curvature(median,rank),curvature(1.6,.85))==passed


def test_relative_rule_when_control_rank_outside_band():
    from tools.jgen.d3.analyse import curvature_rule
    control=curvature(1.3,.77)
    assert curvature_rule(control)=='control_relative'
    assert curvature_gate(curvature(1.4,.74),control)
    assert not curvature_gate(curvature(1.40001,.74),control)
    assert not curvature_gate(dict(complete=True,median_ratio=1.3,rank_agreement=.9),control)
    assert not curvature_gate(curvature(),{})


def test_q7_changes_inclusive_scope_only():
    import numpy as np
    from tools.jgen.d3.curvature import scope_curvatures
    terms=np.zeros((3,6));terms[1:,0]=-.5;terms[1:,-1]=-1
    inclusive,legacy=scope_curvatures(terms,1,1.)
    assert inclusive.tolist()==[3.] and legacy.tolist()==[1.]


def fallback_results():
    return {n:dict(qualified=False,metrics=metrics(.5,1)) for n in ('D','DS','S')}


@pytest.mark.parametrize('interaction_upper,p_upper,chosen',[(1.999,1.999,'S'),(2,0,None),(0,2,None),(None,0,None)])
def test_p_fallback_edges(interaction_upper,p_upper,chosen):
    results=fallback_results();results['S']['metrics']=metrics(p_upper,1)
    ci=None if interaction_upper is None else [interaction_upper-1,interaction_upper]
    assert pick(results,dict(interaction={'top1':{'ci':ci}}))==chosen


def test_d_or_ds_takes_priority_over_fallback():
    r=fallback_results();c=dict(interaction={'top1':{'ci':[-1,1]}})
    r['D']['qualified']=True;r['D']['metrics']['top1']['mean']=-1
    assert pick(r,c)=='D'
    r['D']['qualified']=False;r['DS']['qualified']=True;r['DS']['metrics']['top1']['mean']=-2
    assert pick(r,c)=='DS'
    del r['DS'];r['D']['qualified']=False;assert pick(r,c) is None


def test_fallback_end_to_end_and_locked_opening(tmp_path):
    from tools.jgen.d3.analyse import analyse
    names=['C','D','S','DS'];codes=['A','B']
    make_records(tmp_path,names,codes,dict(C=100,D=100,S=100,DS=100))
    manifest=dict(codes=codes,seeds=[11,12,13],arms=names,label='D3',curvature={n:curvature() for n in names})
    r=analyse(tmp_path,'stage1',manifest)
    assert r['candidate']=='S' and r['candidate_label']=='P' and r['p_fallback']['qualified']
    assert not r['comparisons']['S']['qualified']  # own C-2b route, not D3 improvement gate
    manifest['label']='DRYRUN';assert analyse(tmp_path,'stage1',manifest)['candidate'] is None
    assert analyse(tmp_path,'stage1',manifest)['p_fallback']['qualified'] is False
    path,digest,analysis,run=locked_fixture(tmp_path)
    analysis.write_text(json.dumps(r));(run/'OWNER_GO').touch()
    assert stage3(path,digest,analysis,run)[1]=='S'


def test_lockstep_persistent_workers_continue_after_failure():
    from tools.jgen.d3.worker import drive_lockstep
    calls=[];failures=[]
    class Fake:
        def __init__(self,name):self.name=name
        def run_parent(self,code):
            calls.append((code,self.name))
            if (code,self.name)==('A','D'):raise RuntimeError('failed parent')
            return dict(code=code,pid=self.name)
    workers={n:Fake(n) for n in ('C','D','S','DS')}
    drive_lockstep(workers,['A','B'],lambda *x:failures.append(x),lambda *a,**k:None)
    assert calls==[(c,n) for c in ('A','B') for n in workers]
    assert len(failures)==1


def test_arm_protocol_persists_one_pid(tmp_path):
    import sys
    from tools.jgen.d3.worker import ArmProcess
    source="import os,json,sys;print(json.dumps({'ready':True,'pid':os.getpid()}),flush=True)\nfor line in sys.stdin:\n r=json.loads(line)\n if r.get('stop'):break\n print(json.dumps(dict(code=r['code'],pid=os.getpid())),flush=True)\n"
    arm=ArmProcess([sys.executable,'-u','-c',source],tmp_path/'worker.log')
    try:
        ready=arm.start();a=arm.run_parent('A');b=arm.run_parent('B')
        assert a['pid']==b['pid']==ready['pid']
    finally:arm.close()
    assert arm.process.poll()==0


def test_plan_is_embedded_and_remote_paths_are_normalized(tmp_path):
    import io,tarfile
    from tools.jgen.d3.run import bundle_plan
    source=tmp_path/'source.tgz'
    with tarfile.open(source,'w:gz') as tf:
        member=tarfile.TarInfo('code.txt');member.size=4;tf.addfile(member,io.BytesIO(b'code'))
    plan=dict(identities={'C':dict(db='/readonly/db.json',db_sha256='abc',meta_sha256='def')},label='DRYRUN')
    target=bundle_plan(source,plan,tmp_path/'target.tgz')
    with tarfile.open(target) as tf:
        embedded=json.load(tf.extractfile('./D3_PLAN.json'))
        assert embedded['identities']['C']['db']=='/content/jk/data/db/db.json'
        assert tf.extractfile('code.txt').read()==b'code'
    assert plan['identities']['C']['db']=='/readonly/db.json'


def test_scorer_q7_coverage_and_fk_identity():
    from tools.jgen.d3.analyse import curvature_valid
    summary=curvature()
    assert curvature_valid(summary)  # control Scorer Q7 may be identically zero
    summary['q7_scorer_in_scope']=False
    assert not curvature_valid(summary)
    summary=curvature();summary.pop('fk_prior_settings')
    assert not curvature_valid(summary)
    summary=curvature();summary['fk_prior_settings']={'bend_penalty':'angle'}
    assert not curvature_gate(summary,curvature())


@pytest.mark.parametrize('key,value', [('bend_penalty','angle'),('bend_scale',2.),
    ('bend_sigma_deg',6.),('bend_weight',2.),('bend_free_deg',3.)])
def test_fk_prior_mismatch_refuses_before_vm(tmp_path,monkeypatch,key,value):
    from tools.jgen.d3 import run
    from tools.jgen.d3.arms import fk_prior_settings
    settings=fk_prior_settings({})
    altered=copy.deepcopy(settings);altered[key]=value
    plan={'identities':{'C':{'fk_prior_settings':settings},'D':{'fk_prior_settings':altered}}}
    monkeypatch.setattr(run,'load_cj',lambda:pytest.fail('VM tooling must not be reached'))
    with pytest.raises(ValueError,match='FK bend prior'):run.launch(plan,tmp_path/'run','unused',1)


def test_effective_free_zone_and_runtime_identity(tmp_path):
    from tools.jgen.d3.arms import fk_prior_settings,verify_runtime_identity
    assert fk_prior_settings({'bend_free_deg':3.})['bend_free_deg']==3.
    arm=validate(config(tmp_path))['C']
    verify_runtime_identity(arm)
    arm['fk_prior_settings']['bend_scale']=2.
    with pytest.raises(ValueError,match='runtime FK'):verify_runtime_identity(arm)


def test_bends_required_hashed_and_verified(tmp_path):
    from tools.jgen.d3.arms import sha,verify_runtime_identity,fk_prior_settings
    db=provenance(tmp_path);flags={'q7_bound_centre':True}
    assert 'bends_sha256' not in db_identity(db,False)
    with pytest.raises(FileNotFoundError):db_identity(db,False,flags)
    bends=db.with_suffix('.bends.json');bends.write_text('{"table":1}')
    identity=db_identity(db,False,flags)
    assert identity['bends_sha256']==sha(bends)
    arm=dict(identity,db=str(db),flags=flags,fk_prior_settings=fk_prior_settings(flags))
    verify_runtime_identity(arm)
    bends.write_text('{"table":2}')
    with pytest.raises(ValueError,match='bends_sha256'):verify_runtime_identity(arm)
    bends.unlink()
    with pytest.raises(FileNotFoundError):verify_runtime_identity(arm)


def test_ship_bends_and_verify_remote(tmp_path):
    from types import SimpleNamespace
    from tools.jgen.d3.run import ship_inputs,remote_plan
    from tools.jgen.d3.arms import sha
    db=provenance(tmp_path);db.with_suffix('.bends.json').write_text('{}')
    arm=dict(db=str(db),flags={'q7_bound_centre':True},**db_identity(db,False,{'q7_bound_centre':True}))
    remote={};uploads=[]
    def push(vm,source,target):remote[target]=sha(source);uploads.append(target)
    def remote_py(vm,code,**kwargs):
        # Execute only the test's generated hash probe with a fake remote filesystem.
        import ast
        target=next(ast.literal_eval(line[7:-1]) for line in code.splitlines() if line.startswith('p=Path('))
        return remote.get(target,'')+'\n'
    cj=SimpleNamespace(push=push,remote_py=remote_py)
    ship_inputs(cj,'fake',{'C':arm,'D':arm},tmp_path)
    assert len(uploads)==3 and uploads[-1].endswith('.bends.json')
    normalized=remote_plan({'identities':{'D':arm}})['identities']['D']
    assert normalized['db']=='/content/jk/data/db/db.json'
    assert normalized['bends_sha256']==arm['bends_sha256']
    ship_inputs(cj,'fake',{'D':arm},tmp_path);assert len(uploads)==3
    remote.clear();cj.push=lambda *a:None
    with pytest.raises(ValueError,match='remote input SHA256'):ship_inputs(cj,'fake',{'D':arm},tmp_path)
    arm.pop('bends_sha256')
    with pytest.raises(ValueError,match='requires bends_sha256'):remote_plan({'identities':{'D':arm}})


def test_registered_arm_free_zone_difference_is_rejected(tmp_path):
    from jacobik.core import profile as P
    c=config(tmp_path)
    for name in ('D','DS'):
        flags=dict(c['arms'][name]['flags'],bend_free_deg=3.)
        c['arms'][name].update(flags=flags,fingerprint=P.fingerprint(
            {**P.PRODUCTION_CONFIG,**{k:v[2] for k,v in P.FIELDS.items()},**flags}))
    with pytest.raises(ValueError,match='FK bend prior settings'):validate(c)


def test_required_bends_refused_before_vm(tmp_path,monkeypatch):
    from tools.jgen.d3 import run
    from tools.jgen.d3.arms import fk_prior_settings
    db=provenance(tmp_path)
    arm=dict(db=str(db),flags={'q7_bound_centre':True},**db_identity(db,False),
             bends_sha256='a'*64,fk_prior_settings=fk_prior_settings({}))
    monkeypatch.setattr(run,'load_cj',lambda:pytest.fail('VM tooling must not be reached'))
    with pytest.raises(FileNotFoundError):run.launch({'identities':{'D':arm}},tmp_path/'run','unused',1)


def test_remote_sidecar_basename_collision(tmp_path):
    from tools.jgen.d3.run import remote_plan
    arm=dict(db='/one/db.json',db_sha256='a',meta_sha256='b',
             flags={'q7_bound_centre':True},bends_sha256='c')
    other=dict(arm,db='/two/db.json',bends_sha256='d')
    with pytest.raises(ValueError,match='same remote basename'):
        remote_plan({'identities':{'D':arm,'DS':other}})


def test_stage2_failures_are_judged_by_symmetry(tmp_path):
    """Amendments 2026-11-12: a unit failed in BOTH arms is excluded from the edge-on count (listed) and does not fail
    the stage-2 gates; a unit failed in the control only is reported, not held against the arm; a unit failed in the arm
    only fails the failures gate; an ok record without an edge-on count is unknown."""
    from tools.jgen.d3.analyse import analyse, extra_counts
    names=['C','D'];codes=['A','B','E']   # E stays complete: compare() drops a parent with any failed seed from energy CIs
    make_records(tmp_path,names,codes,dict(C=100,D=90))
    manifest=dict(codes=codes,seeds=[11,12,13],arms=names,label='D3',curvature={n:curvature() for n in names})
    ok=analyse(tmp_path,'stage2',manifest)['comparisons']['D']
    assert ok['gates']['holdout'] and ok['gates']['failures'] and ok['qualified']
    # failed in both arms (e.g. a pre-existing generator defect or a capped parent): excluded, gates still evaluable
    for n in names:
        (tmp_path/n/'records/A_12.json').unlink()
    ec=extra_counts(tmp_path,'D','C',codes,[11,12,13])
    assert ec['edge_on']['arm']==0 and ec['edge_on']['excluded']==[['A',12]] and ec['asymmetric_failures']==[] and ec['arm_failures']==[]
    sym=analyse(tmp_path,'stage2',manifest)['comparisons']['D']
    assert sym['gates']['holdout'] and sym['gates']['failures']
    # failed in the control only (EPUPEL-like): reported, not held against the arm (owner 2026-11-12)
    (tmp_path/'C/records/B_12.json').unlink()
    p=tmp_path/'D/records/B_12.json';rec=json.loads(p.read_text());rec['top1_dE']=500;p.write_text(json.dumps(rec))
    ec=extra_counts(tmp_path,'D','C',codes,[11,12,13])
    assert ec['control_failures']==[['B',12]] and ec['arm_failures']==[] and ['B',12] in ec['edge_on']['excluded']
    assert ec['catastrophic']['arm']==0 and ec['catastrophic']['passed']   # the arm's unit has no control pair
    ctl=analyse(tmp_path,'stage2',manifest)['comparisons']['D']
    assert ctl['gates']['failures'] and ctl['gates']['holdout'] and ctl['qualified']
    # failed in the arm only: the failures gate fails and the arm cannot qualify
    (tmp_path/'D/records/B_13.json').unlink()
    ec=extra_counts(tmp_path,'D','C',codes,[11,12,13])
    assert ec['arm_failures']==[['B',13]] and ['B',13] in ec['edge_on']['excluded']
    bad=analyse(tmp_path,'stage2',manifest)['comparisons']['D']
    assert not bad['gates']['failures'] and not bad['qualified']
    # edge-on count missing from an ok record is unknown, not zero: the holdout gate fails
    p=tmp_path/'D/records/B_11.json';rec=json.loads(p.read_text());rec['d3_geometry']={};p.write_text(json.dumps(rec))
    assert extra_counts(tmp_path,'D','C',['B'],[11])['edge_on']['arm'] is None
    assert not analyse(tmp_path,'stage2',manifest)['comparisons']['D']['gates']['holdout']
