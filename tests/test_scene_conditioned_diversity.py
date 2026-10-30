import copy
import json
import sqlite3

import numpy as np
import pytest

from driversi.rsi.decisions.environment_speed import policy_from_facts
from driversi.rsi.decisions.adaptive_diversity import compare, validate, policy
from driversi.rsi.decisions.scene_diversity_skill import compile_policy, saturated, source_context


def env(speed=10., gap=None, curve=0.):
    return policy_from_facts(dict(token='train',anchor_speed_mps=speed,
        observed_curvature_inv_m=curve,forward_box_gap_m=gap,objects_observed=True))


def context(tags=(), speed=36., feedback=None):
    return dict(source_token='train',tags=list(tags),cluster=dict(leaf='leaf',tags=[],
        speed_mean_kmh=speed,out_of_train_radius=False),feedback=feedback or {})


def settings(e=None,c=None,revision=0,action='initial'):
    return compile_policy(e or env(),c or context(),'mixed',revision,action)


def desc(speed=10.,side=0.,e=None):
    t=np.arange(71)*.1
    return dict(candidate_id=str((speed,side)),comparison_key={'source':'same'},
        environment_speed_policy=e or env(),xy=np.column_stack([t*speed,side*np.minimum(t/2,1)]).tolist())


def test_same_cluster_different_measured_speeds_and_geometry():
    a,b=settings(env(8.)),settings(env(11.))
    assert a['minimum_ade_m']!=b['minimum_ade_m']
    assert a['minimum_speed_mae_mps']!=b['minimum_speed_mae_mps']
    assert settings(env(10.,gap=5.))['minimum_speed_mae_mps']<settings()['minimum_speed_mae_mps']
    assert settings(env(curve=.03))['minimum_path_ade_m']<settings()['minimum_path_ade_m']


def test_cluster_hint_changes_strategy_without_claiming_congestion():
    c=context();c['cluster']['tags']=[dict(text='弯道',count=4)]
    p=settings(c=c)
    assert p['skill_id']=='constrained_source_speed' and not p['reasons']['measured_slow']
    assert p['reasons']['cluster_hint']
    c['cluster']['out_of_train_radius']=True
    assert not settings(c=c)['reasons']['cluster_hint']


def test_duplicate_feedback_does_not_force_all_sources_to_speed(monkeypatch):
    from driversi.rsi.decisions import environment_speed,generation_advisor,scene_diversity_skill
    from driversi.rsi.experiments.reactive_cohort import expand_plan
    from driversi.rsi.experiments.generation_policy import spatial_grid
    monkeypatch.setattr(environment_speed,'source_policy',lambda *a:env())
    monkeypatch.setattr(generation_advisor,'advise',lambda *a:dict(preference='spatial'))
    monkeypatch.setattr(scene_diversity_skill,'source_context',lambda *a:context())
    original=dict(assignments=[dict(id='a',token='train',source_log='log',skill='perturbed_history_tts')])
    family=dict(attempts_per_anchor=10,target_per_anchor=10,seed_start=42,method='history_endpoint_grid_tts_v1',proposal_grid=spatial_grid(),
        environment_speed_context=dict(version=environment_speed.VERSION,logs_root='source'),
        adaptive_context=dict(action='adapt_spatial_or_speed_diversity',revision=1),
        scene_diversity=dict(caption_database='unused'))
    p=expand_plan(original,family)['assignments'][0]['adaptive_diversity_policy']
    assert p['preference']=='spatial' and p['skill_id']=='mixed_source'


def test_bounded_feedback_replays_and_does_not_change_safety():
    c=context(feedback=dict(trajectory_duplicate_before_render=20,teacher_rejected=5,training_sample_ready=3))
    p=settings(c=c,revision=1,action='adapt_spatial_or_speed_diversity')
    assert p['minimum_speed_mae_mps']<settings()['minimum_speed_mae_mps']
    assert validate(p,env())==p and validate(policy(env()),env())==policy(env())
    for field,value in [('minimum_speed_mae_mps',0.),('pdms_strict_gt',.94)]:
        changed=copy.deepcopy(p);changed[field]=value
        with pytest.raises(ValueError):validate(changed,env())
    with pytest.raises(ValueError):settings(c=context(speed=float('nan')))
    with pytest.raises(ValueError):settings(c=context(feedback={'teacher_rejected':-1}))


def test_small_speed_variant_in_slow_scene_not_spatial_or_identical():
    e=env(.8,gap=3.);p=settings(e,context(speed=3.))
    a,b=desc(.8,e=e),desc(.95,e=e)
    result=compare(b,[a],p)
    assert result['distinct'] and result['comparisons'][0]['novelty_basis']=='speed'
    assert not compare(a,[a],p)['distinct']
    assert not compare(desc(.81,e=e),[a],p)['distinct']
    b=copy.deepcopy(a);b['xy'][:15]=(np.asarray(b['xy'][:15])+1.).tolist()
    assert not compare(b,[a],p)['distinct']


def test_duplicate_tail_yields_but_novel_pending_does_not():
    rows=[dict(status='trajectory_duplicate_before_render')]*6+[dict(status='teacher_rejected')]*2
    assert saturated(rows)
    assert not saturated(rows+[dict(status='trajectory_pending_render')])
    assert not saturated([dict(status='teacher_rejected')]*8)


def test_training_caption_and_real_cluster_membership_are_bound(tmp_path):
    db=tmp_path/'navtrain/captions.sqlite3';db.parent.mkdir()
    cap=dict(token='train',split='navtrain',future_images_included=False,
        observation_scope='current_anchor_four_cameras',retrieval_tags=['弯道'])
    with sqlite3.connect(db) as conn:
        conn.execute('create table captions (token text, caption text)')
        conn.execute('insert into captions values (?,?)',('train',json.dumps(cap)))
    cluster=tmp_path/'clusters';(cluster/'navtrain').mkdir(parents=True)
    (cluster/'navtrain/members.jsonl').write_text(json.dumps(dict(token='train',split='navtrain',leaf='a',out_of_train_radius=False))+'\n')
    (cluster/'clusters.json').write_text(json.dumps(dict(nodes=[dict(id='a',label='bend',speed_mean_kmh=36.,speed_range_kmh=[20.,40.])])) )
    value=source_context(db,env())
    assert value['cluster']['leaf']=='a' and value['caption_sha256'] and value['cluster']['member_sha256']
    with sqlite3.connect(db) as conn:
        cap['split']='navtest';conn.execute('update captions set caption=?',(json.dumps(cap),))
    with pytest.raises(ValueError):source_context(db,env())


@pytest.mark.parametrize('attempts,expected',[ (32,50), (10,20) ])
@pytest.mark.parametrize('fail_last_account',[False,True])
def test_collects_before_grouped_render_and_replays_without_new_work(tmp_path,monkeypatch,attempts,expected,fail_last_account):
    from driversi.rsi.experiments import reactive_cohort as mod, reactive_generation, resident_production
    from driversi.visualization import curriculum_operations
    from driversi.runtime.budget import write_json
    c=dict(student_model='chainflow_stage1',training_population='synthetic_only',
        train_catalog='unused',max_samples=50,max_rendered_branches=100,max_candidates=2,
        trajectory_family=dict(target_per_anchor=50,attempts_per_anchor=attempts,minimum_ade_m=.5,
            seed_start=42,cohort_mode='global_50_min1_v1',anchor_budget_mode='explore_until_global_limit'),
        resident_postprocessing=dict(batch_size=16,gpus=[4,5],collect_before_render=50))
    config=tmp_path/'config.json';write_json(config,c)
    plan=tmp_path/'plan.json';write_json(plan,dict(plan_id='p',context={},assignments=[
        dict(id=a,token=a,skill='driverl_full_window') for a in ('a','b')]))
    root=tmp_path/'wave';root.mkdir();events=[];batches=[]
    def generate(plan,cid,asset,db,out,**kw):
        assert kw['render'] is False
        events.append(('trajectory',cid));out.mkdir(parents=True,exist_ok=True)
        write_json(out/'report.json',dict(status='trajectory_ready'))
        return dict(status='trajectory_ready')
    def describe(generated,cid):
        n=int(cid.rsplit('v',1)[-1]);a=cid.split('-v')[0]
        t=np.arange(71)*.1
        return dict(candidate_id=cid,comparison_key={'source':a},
            xy=np.column_stack([t*10.,np.ones(71)*n*3.]).tolist(),behavior_types=[],files={})
    def dispatch(cfg,where,jobs,deadline):
        events.append(('render',len(jobs)));batches.append(jobs)
        assert sum(e[0]=='trajectory' for e in events)>=expected
        assert len({a['anchor_assignment_id'] for a,_,_ in jobs})==1
        result=[]
        for a,d,_ in jobs:
            write_json(d/'generation/render-reservation.json',{})
            result.append(dict(prepared_generation=str(d/'generation/report.json'),outcome=dict(
                status='training_sample_ready',trainable=True,sample_report=str(d/'sample.json'),
                dataset_reference={'package_id':a['id']})))
        return result
    monkeypatch.setattr(reactive_generation,'generate',generate)
    monkeypatch.setattr(mod,'describe',describe)
    monkeypatch.setattr(resident_production,'dispatch',dispatch)
    failed=[]
    def pulse(where):
        value=json.loads((where/'trajectory-family-summary.json').read_text())
        if fail_last_account and not failed and value['qualified']==expected:
            failed.append(True)
            raise OSError('simulated last-account publication failure')
    monkeypatch.setattr(curriculum_operations,'pulse',pulse)
    selected=dict(plan=str(plan),assets=[{},{}]);status=lambda *a,**k:None
    if fail_last_account:
        from driversi.agents.backends.native import digest
        with pytest.raises(OSError,match='last-account'):mod.produce(selected,config,root,1e20,status)
        assert failed
        seals={p: p.read_bytes() for p in root.glob('candidates/*/outcome-seal.json')}
        before={p.parent/'outcome.json':(p.parent/'outcome.json').read_bytes() for p in seals}
        assert len(seals)==expected
        for p in seals:
            assert json.loads(p.read_text())['sha256']==digest(p.parent/'outcome.json')
            assert json.loads((p.parent/'outcome.json').read_text())['trainable'] is True
        events.clear();batches.clear()
    ready=mod.produce(selected,config,root,1e20,status)
    assert len(ready)==expected
    if fail_last_account:
        assert not events and not batches
        assert all(p.read_bytes()==b for p,b in {**seals,**before}.items())
    else:assert any(len(j)>2 for j in batches)
    saved=json.loads((root/'trajectory-family-summary.json').read_text())
    assert saved['target_met']==(expected==50)
    events.clear();batches.clear()
    assert mod.produce(selected,config,root,1e20,status)==ready
    assert not events and not batches
