import json
from pathlib import Path
import pytest
from driversi.agents.backends.native import digest
from driversi.rsi.experiments.augustus_pair import _evaluation_attempt, compare


def attempt(tmp_path):
    root=tmp_path/'arm';(root/'resume').mkdir(parents=True)
    checkpoint=root/'checkpoint';checkpoint.write_bytes(b'unchanged learned state')
    ancestor=root/'resume/training-result.json'
    ancestor.write_text(json.dumps({'checkpoint_files':{str(checkpoint):digest(checkpoint)}}))
    out=tmp_path/'retry';out.mkdir()
    plan={'phase':'evaluate','output':str(out),'previous':str(ancestor),
          'files':{str(ancestor):digest(ancestor),str(checkpoint):digest(checkpoint)}}
    (out/'training-plan.json').write_text(json.dumps(plan))
    value={'status':'completed','metrics':{'parent':1},'metric_scope':'pilot',
           'paired_samples':[{'x':1}],'evaluated_checkpoint':str(checkpoint)}
    for name in ('worker-result.json','training-result.json'):(out/name).write_text(json.dumps(value))
    return root,out,checkpoint


def test_retry_binds_existing_checkpoint_without_changing_ancestor(tmp_path):
    root,out,checkpoint=attempt(tmp_path);before=digest(root/'resume/training-result.json')
    assert _evaluation_attempt(root,out/'training-result.json')==out/'training-result.json'
    assert digest(root/'resume/training-result.json')==before


def test_retry_rejects_another_arm_ancestor(tmp_path):
    root,out,_=attempt(tmp_path)
    with pytest.raises(ValueError,match='same learned ancestor'):
        _evaluation_attempt(tmp_path/'different-arm',out/'training-result.json')


def test_retry_rejects_checkpoint_mutation(tmp_path):
    root,out,checkpoint=attempt(tmp_path);checkpoint.write_bytes(b'different')
    with pytest.raises(ValueError,match='input changed'):
        _evaluation_attempt(root,out/'training-result.json')


def test_retry_cannot_replace_actual_worker_metrics(tmp_path):
    root,out,_=attempt(tmp_path);value=json.loads((out/'training-result.json').read_text())
    value['metrics']['parent']=2;(out/'training-result.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='actual worker'):
        _evaluation_attempt(root,out/'training-result.json')


def test_both_arms_must_be_repeated(tmp_path):
    with pytest.raises(ValueError,match='Both arms'):
        compare(tmp_path,tmp_path,evaluation_reports={'control':'new'})


def test_service_settlement_keeps_old_failure_and_is_idempotent(tmp_path, monkeypatch):
    from driversi.rsi.experiments import augustus_settle as s
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    old=tmp_path/'exp/old'
    plan={'id':'run','program_id':'old-id','context':{'model':'augustus'},
          'source_logs':['train-log'],'conditions':['ego_moving'],
          'memory':str(tmp_path/'memory.sqlite'),'memory_mode':'evolve','generation_costs':{}}
    write(old/'program.json',plan)
    fail=write(old/'review.json',{'node':'independent_evaluation','training_executed':True,'status':'blocked'})
    before=fail.read_bytes()
    reports={}
    for arm in ('control','treatment'):
        root=old/'tasks'/('run-'+arm)
        write(root/'feedback/feedback.json',{'cross_process_resume':{'different_pids':True,'same_compared_state':True}})
        write(root/'report.json',{'elapsed_seconds':10,'sim_agent_steps':0})
        reports[arm]=str(write(tmp_path/'exp/retry'/arm/'training-result.json',{'status':'completed'}))
    execution=write(tmp_path/'execution.json',{'parent_program_id':'old-id','optimizer_updates':0,'wall_seconds':3,
        'rows':{a:{'report':p,'sha256':digest(p),'status':'completed'} for a,p in reports.items()}})
    monkeypatch.setattr(s,'project_root',lambda:tmp_path)
    monkeypatch.setattr(s,'verify_program',lambda _:plan)
    calls=[]
    def paired(*a,**kw):
        calls.append(kw)
        return {'kind':'augustus_paired_feedback','status':'completed','effective_recipe_verified':True,
          'positive_lr_added_sample_ids':['s'],'exposed_added_sample_ids':['s'],'gain_vs_control_pdms_points':.1,
          'gain_vs_parent_pdms_points':-.1,'safety_regressions':[],'evidence_files':{},'synthetic_rows':1,
          'evaluation_scope':'official_navtest_pilot','evaluation_samples':4,'training_evaluation_costs':{}}
    monkeypatch.setattr(s,'compare',paired)
    out=tmp_path/'exp/settled'
    result=s.settle(old,reports,out,execution_receipt=execution)
    assert result['status']=='no_benefit' and result['new_optimizer_updates']==0
    assert fail.read_bytes()==before
    assert s.settle(old,reports,out,execution_receipt=execution)==result
    assert len(calls)==1
    assert json.loads((out/'next-iteration.json').read_text())['next_action']=='revise_data_intervention_then_repeat_matched_trial'


def test_settlement_cannot_replace_old_program(tmp_path,monkeypatch):
    from driversi.rsi.experiments import augustus_settle as s
    monkeypatch.setattr(s,'project_root',lambda:tmp_path)
    with pytest.raises(ValueError,match='separate exp'):
        s.settle(tmp_path/'exp/old',{},tmp_path/'exp/old',execution_receipt='unused')


def test_identical_scores_cannot_hide_different_parent_trajectories(tmp_path):
    import numpy as np
    from driversi.rsi.experiments.augustus_pair import _same_parent_predictions
    a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    np.savez(a,tokens=np.array(['a','b']),trajectories=np.array([[[1.,2.]],[[3.,4.]]]))
    np.savez(b,tokens=np.array(['b','a']),trajectories=np.array([[[3.,4.]],[[1.,2.]]]))
    _same_parent_predictions(a,b)
    np.savez(b,tokens=np.array(['b','a']),trajectories=np.array([[[3.,4.]],[[1.01,2.]]]))
    with pytest.raises(ValueError,match='arrays differ'):_same_parent_predictions(a,b)


def test_memory_retains_failure_but_does_not_reschedule_recovered_evaluation(tmp_path):
    from driversi.rsi.decisions.service_memory import ServiceMemory
    evidence=tmp_path/'evidence';evidence.write_text('sealed')
    common=dict(context={'model':'a'},source_logs=['train'],conditions=['moving'],
                promoted=False,skill_gain_credited=False,training_executed=True,
                evidence={str(evidence):digest(evidence)})
    store=ServiceMemory(tmp_path/'memory.sqlite','evolve')
    old=dict(common,program_id='old',node='independent_evaluation',status='blocked')
    new=dict(common,program_id='new',node='learning_review',status='no_benefit',recovered_from='old')
    store.record(old);store.record(new)
    advice=store.advise(common['context'],['train'],conditions=['moving'])
    assert advice['resolved_episodes']==['old'] and len(store.rows())==2
    assert advice['preflight_actions']==['revise_data_intervention_then_repeat_matched_trial']
    store.close()
