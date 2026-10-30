import json

import pytest

from driversi.rsi.decisions.model_services import diagnose_augustus_branch, student_contract
from driversi.rsi.decisions.capability_skills import SkillMemory, make_plan, training_outcome
from driversi.rsi.plugins import capability_skills as selection
from driversi.rsi.plugins import model_services as services


def candidates():
    return {'snapshot_id':'fixture','candidates':[{'token':'train-token','source_log':'train-log',
        'source_group':'train-log','capability':'route_progress','shared_conditions':['history_turning'],
        'ranks':{'visual':1},'status':'condition_match_requires_diagnosis'}],
        'capability_feedback':{'schema_version':1,'benchmark':'navtest','checkpoint':'/other/chainflow.ckpt',
            'evaluator_version':'native-v1','slices':[{'capability':'route_progress','count':10,'failures':2}]}}


def feedback(p):
    return {'kind':'chainflow_paired_feedback','status':'completed','effective_recipe_verified':True,
        'exposed_added_sample_ids':['synthetic'],'unconsumed_added_sample_ids':[],
        'total_experiment_wall_seconds':10,'treatment_minus_continued_training':{'score':1},
        'capability_skill_binding':{'plan_id':p['plan_id'],'assignment_ids':[p['assignments'][0]['id']],
            'evaluation_split':'official_navtest_locked_subset_pilot','model':'chainflow_stage1',
            'parent_checkpoint':p['context']['parent_checkpoint'],'verified_benchmark':p['context']['benchmark'],
            'verified_evaluator_version':p['context']['evaluator_version']}}


def branch():
    return {"kind": "selected_native_branch", "branch": "teacher", "teacher_qualified": True, "future_frames": 10,
        "trajectory_evidence": {"arrival_contact_records": 0, "future_contact_records": 0,
            "history_changed": True, "restore_exact": True, "score": {"score": 1.,
                "native_score_threshold": .95,
                "protocol": "navsim-v1-native-math-executed-counterfactual-v1"}}}


def test_high_score_cannot_override_world_contacts_or_prove_student_failure():
    b=branch();b['trajectory_evidence']['future_contact_records']=300
    b['teacher_qualified']=False
    r=diagnose_augustus_branch(b)
    assert r['teacher_score100']==100
    assert r['next_action']=='audit_world_contacts_before_synthesis'
    assert 'future_contacts_absent' in r['failed_physical_checks']
    assert not r['student_failure_established'] and not r['student_evaluated']
    assert not r['trainable'] and not r['promoted']


@pytest.mark.parametrize('mutation,failed',[
    (lambda b:b['trajectory_evidence'].pop('arrival_contact_records'),'arrival_contacts_absent'),
    (lambda b:b['trajectory_evidence']['score'].update(score=float('nan')),'native_teacher_pdms_ge_95'),
    (lambda b:b['trajectory_evidence']['score'].update(native_score_threshold=.9),'native_teacher_pdms_ge_95'),
    (lambda b:b['trajectory_evidence']['score'].update(protocol='original_log'),'native_teacher_pdms_ge_95'),
    (lambda b:b.update(future_frames=7),'future_support'),
    (lambda b:b['trajectory_evidence'].update(restore_exact=False),'restore_exact'),
])
def test_missing_or_incompatible_evidence_fails_closed(mutation,failed):
    b=branch();mutation(b)
    assert failed in diagnose_augustus_branch(b)['failed_physical_checks']


def test_clean_physics_still_needs_native_joint_supervision_and_feedback():
    r=diagnose_augustus_branch(branch())
    assert not r['failed_physical_checks']
    assert r['next_action']=='build_and_verify_branch_consistent_augustus_supervision'
    assert len(r['learning_blockers'])==2 and not r['trainable']
    assert r['student_contract']['future_points']==8
    assert not r['teacher_samples_are_on_policy']
    c=student_contract('augustus');c['future_points']=100
    assert student_contract('augustus')['future_points']==8
    with pytest.raises(ValueError): student_contract('unknown')


@pytest.mark.parametrize('name',['baseline',None])
def test_baseline_with_global_teacher_metadata_is_not_a_teacher_score(name):
    b=branch();b['branch']=name
    b['trajectory_evidence']['teacher']='DriveRL_native_argmax'
    with pytest.raises(ValueError,match='selected teacher branch'):
        diagnose_augustus_branch(b)


def test_augustus_plan_and_feedback_cannot_relabel_chainflow_gain(tmp_path):
    m=SkillMemory(tmp_path/'memory.sqlite')
    p=make_plan(candidates(),model='augustus',memory=m,
                budget={'max_tasks':1,'max_wall_seconds_per_task':100})
    m.close()
    assert p['student_contract']['future_points']==8
    assert not p['student_contract']['paired_skill_feedback']
    assert all(o['skill']!='same_anchor_relabel' for o in p['assignments'][0]['options'])
    f=feedback(p);f['capability_skill_binding']['model']='augustus'
    r=training_outcome(p,f,'fixture')
    assert not r['credited']
    assert 'student_skill_paired_feedback_not_connected' in r['reasons']


def test_augustus_selection_requires_actual_checkpoint_feedback(tmp_path,monkeypatch):
    monkeypatch.setattr(selection,'project_root',lambda:tmp_path)
    cfg={'model':'augustus','memory':'exp/m.sqlite','budget':{},'advisor':None}
    with pytest.raises(ValueError,match='actual native'):
        selection.prepare(cfg,tmp_path,tmp_path/'r.sqlite')
    native=tmp_path/'model.json'
    native.write_text(json.dumps({'model':'augustus','weights':{'parent':'weights/augustus/SFT'}}))
    prepared,_=selection.prepare({**cfg,'model_config':str(native)},tmp_path,tmp_path/'r.sqlite')
    c=tmp_path/'c.json';c.write_text(json.dumps(candidates()))
    with pytest.raises(ValueError,match='another student checkpoint'):
        selection.execute(prepared,{'candidates':{'artifacts':{'report':str(c)}}},tmp_path/'out',tmp_path/'r.sqlite')


def test_branch_teacher_must_match_declared_teacher(tmp_path):
    b=branch();b['trajectory_evidence']['teacher']='other'
    p=tmp_path/'b.json';p.write_text(json.dumps(b))
    config={'files':{},'student_parent_files':{},'roles':{'teacher_checkpoint':'expected'}}
    with pytest.raises(ValueError,match='Actual branch teacher'):
        services.execute(config,{'branch':{'artifacts':{'report':str(p)}}},tmp_path/'out',tmp_path/'r.sqlite')


def test_sealed_reference_preserves_rejection_and_source_groups(tmp_path,monkeypatch):
    from driversi.rsi.plugins import chainflow_synthetic
    b=branch();b['teacher_qualified']=False
    p=tmp_path/'original.json';p.write_text(json.dumps(b))
    window=tmp_path/'w.json';window.write_text('{}')
    original={'artifacts':{'report':str(p),'trajectory_window':str(window)}}
    state={'task':{'source_groups':['train-log']}}
    monkeypatch.setattr(chainflow_synthetic,'_stored_stage',lambda *a:(state,original,{'identity':'sealed'}))
    c,_=services.prepare_reference({'source_task':'a','source_stage':'b'},tmp_path,tmp_path/'r')
    assert c['source_groups']==['train-log']
    r=services.execute_reference(c,{},tmp_path/'out',tmp_path/'r')
    assert not json.loads((tmp_path/'out/report.json').read_text())['teacher_qualified']
    assert not json.loads((tmp_path/'out/origin.json').read_text())['physics_reexecuted']
    c['identity']='changed'
    with pytest.raises(ValueError,match='branch changed'):
        services.execute_reference(c,{},tmp_path/'out2',tmp_path/'r')
