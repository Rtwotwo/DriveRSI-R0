"""CPU-only state-machine tests; no API, GPU or driving-result claims."""
import copy
import json
import sys
import types
from pathlib import Path

import pytest

from driversi.rsi.experiments import hard_case_repair as engine
from driversi.rsi.decisions.skill_evolution import memory


NODE = engine.NODE
TEXTS = [
    'Inspect the lane ahead and smoothly preserve progress unless visible history shows an approaching conflict.',
    'Compare crossing traffic across observed frames. Delay committing into occupied space while preserving the route.',
    'Assess the visible corridor boundaries and retain lateral continuity. Avoid unnecessary evasive motion toward oncoming lanes.',
    'Resolve uncertain occlusion conservatively from observed motion. Allow an adequate response interval before entering a crossing.',
]


def skill(text):
    return dict(version='v-' + str(TEXTS.index(text)), low_speed_mps=2,
                skills={NODE:text, 'turn_conflict_check':'turn', 'low_speed_progress_check':'slow'})


def member(token, role='development', group=None):
    return dict(token=token, role=role, validation_group=group, case_id=token,
                source_log='drive-' + token + '_0_1', repair_family=token=='target',
                skill=dict(id=NODE, evidence=dict(navigation_command='GO STRAIGHT', speed_mps=4)))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


def setup_run(tmp_path, monkeypatch, transfer_pass=True, budget=6):
    members=[member('target'), member('guard'), member('family','validation','family'),
             member('control','validation','control')]
    source=dump(tmp_path/'source.json',dict(members=members))
    evidence=dump(tmp_path/'evidence.json',dict(target_tokens=['target'],evidence=[],cases=[],images=[]))
    plan=dict(task_id='cpu-state-machine',skills=skill(TEXTS[0]),members=members,
              target_tokens=['target'],seeds=[42,43],source_prepared=source,evidence=evidence,
              memory=str(tmp_path/'memory.sqlite'),prior_history=[dict(candidate=skill(TEXTS[1]))],
              max_doctor_requests=budget,scope='synthetic_fixture',validation_scope='synthetic_fixture',
              checkpoint_sha256='fixture')
    plan_path=dump(tmp_path/'plan.json',plan)
    monkeypatch.setattr(engine.pilot,'validate_plan',lambda path: copy.deepcopy(plan))
    calls=dict(request=[],stages=[])

    def request(p,ev,state,history,out):
        i=len(calls['request']); calls['request'].append(dict(index=i,history=len(history)))
        text=TEXTS[i+1]
        return dict(candidate=skill(text),skill_id=NODE,target_tokens=['target'],valid=True,error=None,
                    proposal=dict(diagnosis=dict(hypothesis='fixture')),usage=dict(total_tokens=5),doctor_seconds=0.1)
    monkeypatch.setattr(engine,'request_proposal',request)

    def subset(source,p,selected,incumbent,candidate,out):
        return Path(dump(Path(out)/'prepared.json',dict(members=selected,candidate=candidate,incumbent=incumbent)))
    monkeypatch.setattr(engine,'subset',subset)

    def stage(prepared,out):
        p=json.loads(Path(prepared).read_text()); calls['stages'].append([m['token'] for m in p['members']])
        good=p['candidate']['skills'][NODE]==TEXTS[3]
        scored=[]
        for m in p['members']:
            for seed in (42,43):
                for arm in ('native','incumbent','candidate'):
                    difficult=m['token'] in ('target','family')
                    improvement=good and arm=='candidate' and difficult and (m['token']=='target' or transfer_pass)
                    ttc=1 if not difficult or improvement else 0
                    pdms=0.9 if ttc else 0.5
                    metric=dict(PDMS=pdms,TTC=ttc,NC=1,DAC=1,DDC=1,EP=1,C=1)
                    scored.append(dict(token=m['token'],seed=seed,arm=arm,metrics=metric,
                        raw_trajectory=[[float(i)+(0.1 if arm=='candidate' else 0),0.,0.] for i in range(8)]))
        path=dump(Path(out)/'scoring/scores.json',scored)
        return dict(scores=path,scores_sha256=engine.digest(path),sim_agent_steps=len(scored)*80)
    monkeypatch.setattr(engine,'evaluate_stage',stage)
    return plan,plan_path,calls


@pytest.mark.parametrize('transfer_pass',[True,False])
def test_duplicate_skip_failure_continue_single_transfer_and_replay(tmp_path,monkeypatch,transfer_pass):
    plan,path,calls=setup_run(tmp_path,monkeypatch,transfer_pass)
    out=tmp_path/'run'
    result=engine.run(path,out)
    assert result['doctor_calls']==3
    assert result['attempts'][0]['sim_agent_steps']==0
    assert 'repeated_intervention' in result['attempts'][0]['precheck']['reasons']
    assert not result['attempts'][1]['screen']['accepted']
    assert result['attempts'][2]['screen']['accepted']
    assert calls['stages']==[['target','guard'],['target','guard'],['family','control']]
    assert result['skill_promoted'] is transfer_pass
    assert result['stop_reason']==('skill_promoted' if transfer_pass else 'transfer_rejected_holdout_consumed')
    assert calls['request'][2]['history']>calls['request'][1]['history']
    state=memory(plan['memory'])
    assert state['revision']==3
    assert state['active']==skill(TEXTS[3] if transfer_pass else TEXTS[0])
    before=copy.deepcopy(calls)
    replay=engine.run(path,out)
    assert calls==before
    assert replay['usage']==result['usage']
    assert replay['scored_predictions']==result['scored_predictions']


def test_reconcile_committed_memory_before_receipt_does_not_repeat_request(tmp_path,monkeypatch):
    plan,path,calls=setup_run(tmp_path,monkeypatch,budget=1)
    real=engine.write_json
    failed=[]
    def interrupt(path,value):
        if Path(path).name=='decision.json' and not failed:
            failed.append(True)
            raise OSError('synthetic interrupted after memory commit')
        return real(path,value)
    monkeypatch.setattr(engine,'write_json',interrupt)
    with pytest.raises(OSError):engine.run(path,tmp_path/'run')
    assert memory(plan['memory'])['revision']==1
    assert len(calls['request'])==1
    replay=engine.run(path,tmp_path/'run')
    assert len(calls['request'])==1 and calls['stages']==[]
    assert replay['doctor_calls']==1 and replay['stop_reason']=='proposal_budget_exhausted'


def test_evaluation_unknown_inference_is_not_rerun(tmp_path,monkeypatch):
    called=[]
    monkeypatch.setitem(sys.modules,'driversi_agents.augustus.skill_inference',types.SimpleNamespace(run=lambda *a:called.append(a)))
    prep=Path(dump(tmp_path/'prepared.json',{}));out=tmp_path/'stage'
    dump(out/'evaluation-intent.json',dict(prepared_sha256=engine.digest(prep)))
    with pytest.raises(RuntimeError,match='inference_outcome_unknown'):
        engine.evaluate_stage(prep,out)
    assert called==[]


def test_completed_score_reuse_rejects_changed_score(tmp_path,monkeypatch):
    called=[]
    monkeypatch.setitem(sys.modules,'driversi_agents.augustus.skill_inference',types.SimpleNamespace(run=lambda *a:called.append(a)))
    prep=Path(dump(tmp_path/'prepared.json',{}));out=tmp_path/'stage'
    scored=Path(dump(out/'scoring/scores.json',[]))
    report=dict(prepared_sha256=engine.digest(prep),scores=str(scored),scores_sha256=engine.digest(scored))
    dump(out/'scoring/report.json',report)
    assert engine.evaluate_stage(prep,out)==report
    scored.write_text('[{}]')
    with pytest.raises(ValueError,match='Scored stage identity changed'):engine.evaluate_stage(prep,out)
    assert called==[]


@pytest.mark.parametrize('malformed',[['not','an','object'],'text',17])
def test_malformed_doctor_reply_is_recorded_and_next_attempt_runs(tmp_path,monkeypatch,malformed):
    plan,path,calls=setup_run(tmp_path,monkeypatch,budget=2)
    def request(p,ev,state,history,out):
        calls['request'].append(state['revision'])
        return dict(candidate=state['active'],skill_id=NODE,target_tokens=['target'],valid=False,
                    error='schema',proposal=malformed,usage={},doctor_seconds=0)
    monkeypatch.setattr(engine,'request_proposal',request)
    result=engine.run(path,tmp_path/'run')
    assert calls['request']==[0,1] and calls['stages']==[]
    assert result['doctor_calls']==2 and not result['skill_promoted']
    assert all(e['diagnosis'] is None for e in memory(plan['memory'])['events'])


def test_visible_history_excludes_validation_results():
    e=dict(candidate=skill(TEXTS[1]),decision=dict(reasons=['development:no_improvement',
           'validation:safety_regression','no_cross_log_TTC_transfer','subgroup_family:mean_decline']))
    assert engine.history_context([e])[0]['reasons']==['development:no_improvement']
