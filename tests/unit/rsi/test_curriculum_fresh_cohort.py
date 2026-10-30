"""Fresh-data continuation checks, with synthetic fixtures only."""
import json
from pathlib import Path
import pytest
from driversi.agents.backends.native import digest
from driversi.rsi.experiments import problem_cohort as pc, generation_policy as gp
from driversi.rsi.experiments.reactive_cohort import validate_budget


def row(source,n):
    return dict(source_token=source,trainable=True,rendered=True,status='training_sample_ready',
                sample_report=f'report{n}',dataset_reference=dict(package_id=f'p{n}'),descriptor=dict(id=n))


def test_previous_fifty_cannot_trigger_same_trial_and_freshness_keeps_coverage():
    old=[row('a',i) for i in range(49)]+[row('b',49)]
    consumed=[r['sample_report'] for r in old]
    assert not pc.training_selection(['a','b'],old,consumed)['passed']
    assert pc.training_selection(['a','b'],old,consumed)['fresh_shortage']==10
    new=[row('a',i) for i in range(50,60)]
    assert not pc.training_selection(['a','b'],old+new[:9],consumed)['passed']
    result=pc.training_selection(['a','b'],old+new,consumed)
    assert result['passed'] and result['fresh_selected']==10 and len(result['selected_rows'])==50
    assert {'a','b'}=={r['source_token'] for r in result['selected_rows']}
    assert pc.training_selection(['a','b'],old)['passed']


def test_seed_reserves_fresh_slots_but_keeps_full_prior_pool(tmp_path):
    outcome=tmp_path/'out.json';outcome.write_text('{}')
    rows=[row('a',i) for i in range(49)]+[row('b',49)]
    for r in rows:r.update(original_outcome=str(outcome),original_outcome_sha256=digest(outcome))
    pool=tmp_path/'pool.json'
    data=dict(rows=rows,previously_consumed_reports=[r['sample_report'] for r in rows])
    pool.write_text(json.dumps(data))
    cfg=dict(prior_cohort=dict(path=str(pool),sha256=digest(pool)),
             trajectory_family=dict(target_per_anchor=50,cohort_mode='global_50_min1_v1'))
    result=pc.seed(cfg,dict(assignments=[dict(token=t,anchor_assignment_id=t) for t in 'ab']))
    assert len(result)==40 and {r['source_token'] for r in result}=={'a','b'}
    assert len(json.loads(pool.read_text())['rows'])==50


def test_history_counts_all_completed_failed_targets_not_inflight(tmp_path):
    for i in range(3):
        folder=tmp_path/f'rounds/p-r{i+1}';folder.mkdir(parents=True)
        (folder/'training-input.json').write_text(json.dumps(dict(samples=[f's{i}'])))
        if i<2:(folder/'local-learning-decision.json').write_text(json.dumps(dict(status='target_still_zero',target_success=False)))
    used,evidence=pc.failed_training_history(tmp_path,dict(id='p'))
    assert used=={'s0','s1'} and len(evidence)==4


def test_one_remaining_attempt_passes_real_primary_and_family_validation(tmp_path):
    policy=dict(kind='symmetric_history_native_tts_resident_v1',physical_prefetch_workers=2,
                resident_postprocessing=dict(batch_size=16,gpus=[4,5]))
    (tmp_path/'primary-generation-policy.json').write_text(json.dumps(policy))
    config=dict(student_model='chainflow_stage1',training_population='synthetic_only',
        trajectory_family=dict(target_per_anchor=50,attempts_per_anchor=1,minimum_ade_m=.5,
            seed_start=42,cohort_mode='global_50_min1_v1',anchor_budget_mode='explore_until_global_limit',
            stop_after_consecutive_duplicates=6,stop_after_visual_failures=2),
        max_samples=50,max_rendered_branches=100,max_candidates=3)
    (tmp_path/'round').mkdir()
    gp.apply(tmp_path,config,tmp_path/'round')
    assert validate_budget(config)['attempts_per_anchor']==1
    assert config['trajectory_family']['stop_after_consecutive_duplicates']==1
    config['trajectory_family'].pop('cohort_mode')
    with pytest.raises(ValueError):validate_budget(config)


def test_false_ready_is_demoted_before_train(tmp_path,monkeypatch):
    from driversi.rsi.experiments import continual_curriculum as cc
    folder=tmp_path/'round';folder.mkdir()
    (folder/'generation-config.json').write_text(json.dumps(dict(visual_quality_profile='experimental_numeric25_adaptive_v1')))
    case=dict(id='p',status='data_ready',current_round=str(folder))
    state=dict(active='p',cases=[case]);path=tmp_path/'state.json';path.write_text(json.dumps(state))
    monkeypatch.setattr(cc,'state_path',lambda r:path)
    monkeypatch.setattr(cc,'save',lambda r,s:path.write_text(json.dumps(s)))
    monkeypatch.setattr(pc,'collect',lambda *a:dict(qualified=50,unique_per_source={'a':50},ready=False))
    pc.update(tmp_path)
    assert json.loads(path.read_text())['cases'][0]['status']=='generation_stopped'
