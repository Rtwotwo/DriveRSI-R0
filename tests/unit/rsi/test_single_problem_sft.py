"""Contract fixtures only; these tests do not establish real training benefit."""
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from driversi.rsi.decisions.problem_learning import source_quota, sft_updates, target_success, select_problem_sources
from driversi.rsi.decisions.continual_learning import decide, SAFETY
from driversi.rsi.experiments import continual_curriculum as cc
from driversi_agents.chainflow.sampling import GlobalEpochCursor, distributed_settings


def rows():
    return ([dict(sample_id=f'r{i}',kind='original_navtrain') for i in range(100)]
            + [dict(sample_id=f's{i}',kind='synthetic_navtrain') for i in range(50)])


def test_five_sources_not_fifty_from_one_source():
    selected=[str(i) for i in range(5)]
    assert source_quota(selected, [s for s in selected for _ in range(10)], 10)['passed']
    assert not source_quota(selected, ['0']*50, 10)['passed']
    assert not source_quota(selected, [s for s in selected for _ in range(10)]+['other'],10)['passed']


def test_actual_similarity_determines_count_and_later_round_expands_pool():
    candidates=[dict(token=str(i),max_similarity=s) for i,s in enumerate([.9,.89,.865,.855,.8])]
    policy=dict(max_rounds=3,source_caps=[5,5,6],similarity_margins=[.03,.03,.06])
    assert len(select_problem_sources(candidates,0,policy))==2
    assert len(select_problem_sources(candidates,1,policy))==2
    assert len(select_problem_sources(candidates,2,policy))==4
    assert source_quota(['a'],['a']*10,10)['passed']


def test_new_only_four_epochs_every_sample_drawn_four_times_and_no_original():
    data=rows(); cursor=GlobalEpochCursor(data,42,2,1,1); counts=Counter()
    for _ in range(sft_updates(50,4,2)):
        ids,_=cursor.next_batch();counts.update(data[i]['sample_id'] for i in ids)
    assert counts==Counter({f's{i}':4 for i in range(50)})
    assert cursor.original_draws==0 and cursor.synthetic_draws==200 and cursor.epoch==4
    assert sft_updates(100,4,2)==200


def test_new_only_resume_replays_cursor_exactly():
    a=GlobalEpochCursor(rows(),42,2,1,1)
    for _ in range(37):a.next_batch()
    b=GlobalEpochCursor(rows(),42,2,1,1);b.load_state_dict(a.state_dict())
    assert [a.next_batch() for _ in range(63)]==[b.next_batch() for _ in range(63)]
    assert a.state_dict()==b.state_dict()


def test_original_control_and_existing_mixed_recipe_unchanged():
    control=GlobalEpochCursor(rows()[:100],42,2,1,1)
    for _ in range(100):control.next_batch()
    assert control.original_draws==200 and control.synthetic_draws==0
    old=GlobalEpochCursor(rows(),42,2,1,8)
    for _ in range(40):old.next_batch()
    assert old.original_draws==70 and old.synthetic_draws==10
    assert distributed_settings({'distributed_training':dict(devices=[6],checkpoint_interval=2,
                                                            synthetic_every=1)})['synthetic_every']==1


def make_report(values):
    return dict(model='chainflow_stage1',complete=True,score_unit='0_to_1',native_score_unmodified=True,
        rows=[dict(token=t,metrics=dict(PDMS=v,**{k:1. for k in SAFETY})) for t,v in values.items()])


@pytest.mark.parametrize('score,success',[(0,False),(.001,True),(.5,True),(1,True)])
def test_zero_escape_is_strict_positive_not_95(score,success):
    p=make_report(dict(q=0,old=1));c=make_report(dict(q=score,old=1))
    d=decide(p,c,p,expected=['q','old'],target='q',retained=['old'],native_paired_benefit=True,
             mastery=0,strict_target=True)
    assert d['target_success']==success and d['promoted']==success


def test_local_success_does_not_promote_safety_regression():
    p=make_report(dict(q=0,old=1));c=make_report(dict(q=.5,old=1))
    c['rows'][1]['metrics'][SAFETY[0]]=0
    d=decide(p,c,p,expected=['q','old'],target='q',retained=[],native_paired_benefit=True,
             mastery=0,strict_target=True)
    assert d['target_success'] and not d['promoted']
    s=dict(policy=dict(max_rounds=2,replay_limit=0,strict_target=True),incumbent='parent',
        active='p',retained=[],replay=[],rounds=[])
    case=dict(query='q',id='p',round=0)
    cc.apply_verdict(s,case,d,'candidate',['sample'])
    assert case['status']=='target_improved_release_rejected' and s['incumbent']=='parent'
    assert s['active'] is None and not s['replay']


def test_explicit_target_population_is_not_full_or_random_pilot():
    from driversi_agents.chainflow.benchmark import evaluation_population
    idx=dict(records=[dict(token='q'),dict(token='other')])
    assert evaluation_population(dict(target_token='q'),idx)==[dict(token='q')]
    for config in [dict(target_token='missing'),dict(target_token='q',split='navtrain'),
                   dict(target_token='q',pilot_subset='random'),dict(target_token='q',shards=2)]:
        with pytest.raises(ValueError):evaluation_population(config,idx)


def test_failed_target_stops_before_full_evaluation_and_cost_has_two_proposals(tmp_path):
    from driversi.rsi.experiments.chainflow_campaign import target_stop_guard
    from driversi.rsi.plugins.chainflow_pre_eval import prepare
    assert prepare(dict(side='treatment',target_token='q'),None,None)[1]==80
    p=tmp_path/'target-feedback.json'
    target_stop_guard(p)
    p.write_text('{"target_success":true}');target_stop_guard(p)
    p.write_text('{"target_success":false}')
    with pytest.raises(RuntimeError,match='local_target_still_zero'):target_stop_guard(p)


def test_new_27db_policy_is_strict_and_legacy_profiles_are_preserved():
    from driversi.rendering.diagnostics.numeric_admission import numeric_settings
    from driversi.rendering.diagnostics.worldengine_admission import visual_policy
    p=numeric_settings('experimental_numeric27_strict_ade05_v1')
    assert p==dict(psnr_db=27.,strict_psnr=True,minimum_ade_m=.5)
    assert numeric_settings('experimental_numeric25_v1')['psnr_db']==25
    assert visual_policy('experimental_numeric27_strict_ade05_v1')['source_psnr_min_db']==27


def test_new_training_rejects_sealed_legacy25_sample(tmp_path,monkeypatch):
    from driversi.rsi.experiments.chainflow_campaign import train_pair
    from driversi_agents.chainflow import synthetic
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    config=save('config.json',dict(training_population='synthetic_only',
        visual_quality_profile='experimental_numeric27_strict_ade05_v1',strict_source_psnr_gt=27.))
    visual=save('visual.json',dict(visual_quality_profile='experimental_numeric25_v1'))
    sample=save('sample.json',dict(row={},files={}))
    monkeypatch.setattr(synthetic,'validate_admission',lambda *a:dict(visual_admission=dict(path=visual)))
    with pytest.raises(ValueError,match='Legacy visual admission'):
        train_pair(config,[sample],tmp_path/'pair')
    assert not (tmp_path/'pair').exists()


def test_strict27_postprocessing_uses_numeric_and_geometry_routes(tmp_path,monkeypatch):
    from driversi.rsi.experiments import reactive_postprocess as post, readiness
    from driversi.rendering.diagnostics import numeric_admission
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return str(path)
    generation=save(tmp_path/'generation.json',dict(status='rendered',files={},
        render_report=save(tmp_path/'render.json',{}),
        source_report=save(tmp_path/'source.json',{}),
        branch_report='branch.json',render_payload='render-payload.json'))
    config=save(tmp_path/'config.json',dict(
        visual_quality_profile='experimental_numeric27_strict_ade05_v1',
        strict_source_psnr_gt=27.,train_catalog='train',profile='feature',generated_catalog='new'))
    monkeypatch.setattr(readiness,'analyze_render',lambda *a:dict(blockers=[],source_psnr_mean_db=28.))
    seen=[]
    def admit(generation,diversity,output,*,profile):
        seen.append(profile)
        save(output/'result.json',dict(admitted=True));save(output/'admission.json',{})
    def dispatch(action,arguments,directory,**kwargs):
        seen.append(action)
        assert action not in ('review','labels')
        out=Path(arguments['output'])
        if action=='geometry_labels':save(out/'label-review.json',{})
        elif action=='publish':save(out/'report.json',{})
        elif action=='features':
            save(out/'report.json',dict(trainable=True,parent_vectors_copied=False))
            save(Path(directory)/'result.json',{})
    monkeypatch.setattr(numeric_admission,'admit',admit)
    monkeypatch.setattr(post,'run_native',dispatch)
    result=post.process(generation,config,tmp_path/'post')
    assert result['trainable']
    assert seen==['experimental_numeric27_strict_ade05_v1','preview','geometry_labels','publish','features']


@pytest.mark.parametrize('psnr',[25.04,26.999,27.0])
def test_strict27_preflight_rejects_legacy_and_exact_boundary(tmp_path,monkeypatch,psnr):
    from driversi.rsi.experiments import reactive_postprocess as post, readiness
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    generation=save('generation.json',dict(status='rendered',files={},
        render_report=save('render.json',{}),source_report=save('source.json',{})))
    config=save('config.json',dict(visual_quality_profile='experimental_numeric27_strict_ade05_v1',
                                 strict_source_psnr_gt=27.))
    monkeypatch.setattr(readiness,'analyze_render',lambda *a:dict(blockers=[],source_psnr_mean_db=psnr))
    monkeypatch.setattr(post,'run_native',lambda *a,**kw:pytest.fail('Rejected source dispatched'))
    result=post.process(generation,config,tmp_path/'post')
    assert result['status']=='visual_preflight_rejected' and not result['trainable']


def test_compiler_declares_new_only_actual_exposure_and_target_stage(tmp_path,monkeypatch):
    from driversi.rsi.plugins import chainflow_synthetic as compiler
    from driversi_agents.chainflow import benchmark,data,synthetic
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    base=dict(schema_version=2,benchmark=dict(protocol='official_navtest'),files={},splits=dict(
        train=[dict(sample_id=f'r{i}',kind='original_navtrain',source_log='train') for i in range(100)],
        development=[dict(sample_id='q',kind='original_navtrain',source_log='test',metric_cache='/cache')]))
    model=write('model.json',dict(model='chainflow_stage1',settings=dict(
        distributed_training=dict(devices=[6],checkpoint_interval=2,synthetic_every=1),
        training_recipe=dict(scheduler_dataset_size=50,scheduler_num_epochs=4))))
    samples=[write(f's{i}.json',dict(kind='chainflow_admitted_synthetic_sample',files={},
        row=dict(sample_id=f's{i}',kind='synthetic_navtrain',source_log='train'))) for i in range(50)]
    monkeypatch.setattr(compiler,'project_root',lambda:tmp_path)
    monkeypatch.setattr(data,'validate_manifest',lambda p:json.loads(Path(p).read_text()))
    monkeypatch.setattr(benchmark,'evaluator_files',lambda _: {})
    monkeypatch.setattr(synthetic,'validate_admission',lambda *a:None)
    task=compiler.compile_paired_task(write('base.json',base),samples,model,tmp_path/'exp/pair',
        epoch_training=True,batch_size=2,updates_per_arm=100,pre_evaluation_gate=True,
        training_population='synthetic_only',target_token='q')
    d=json.loads((task.parent/'intervention.json').read_text())
    assert d['expected_sample_exposures']['treatment']=={f's{i}':4 for i in range(50)}
    assert not set(d['expected_sample_exposures']['control']) & {f's{i}' for i in range(50)}
    stages=json.loads(task.read_text())['steps']
    gate=next(s for s in stages if s['id']=='treatment_pre_evaluation')
    assert gate['config']['target_token']=='q' and gate['max_wall_seconds']==660
    with pytest.raises(ValueError,match='no replay'):
        compiler.compile_paired_task(tmp_path/'base.json',samples,model,tmp_path/'exp/bad',
            epoch_training=True,training_population='synthetic_only',replay_reports=[samples[0]])
