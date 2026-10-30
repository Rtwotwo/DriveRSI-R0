import copy
import json

import pytest

from driversi.rsi.decisions import grounded_skills as g
from driversi_agents.augustus.skill_inference import evolution_prompts


def example():
    incumbent=dict(version='base',skills={'following':'Keep the visible route and preserve safe progress. Check forward separation before changing speed.'})
    value=dict(skill_id='following',skill_text=incumbent['skills']['following'],
        rationale='Apply only where the available observation supports this check.',expected_effect='Avoid unsupported intervention.',
        diagnosis=dict(hypothesis='The changed response may interfere with the native stopping behavior.',
            supporting_evidence=['case-01.candidate.NC'],counter_evidence=[],
            uncertainty='A collision metric does not establish intent; inspect independent temporal evidence.'),
        eligibility=dict(commands=['GO STRAIGHT'],speed_min_mps=1,speed_max_mps=15,exclude_stop=True))
    evidence=[dict(id='case-01.candidate.NC',kind='measured_metric',value=0),dict(id='case-01.caption.0',kind='unverified_annotation',value='road')]
    return incumbent,value,evidence


def test_gate_only_edit_preserves_other_fields_and_native_fallback():
    incumbent,value,evidence=example();candidate=g.validate(value,'following',incumbent,evidence)
    skill=dict(id='following',text=value['skill_text'],evidence=dict(navigation_command='STOP',speed_mps=2))
    gated=g.apply_gate(skill,candidate)
    assert not gated['enabled'] and 'stop_excluded' in gated['eligibility_reason']
    base='Native prompt with trailing space \n'
    prompts=evolution_prompts(base,{'incumbent':skill,'candidate':gated})
    assert prompts['candidate']==prompts['native']==base
    assert incumbent['version']=='base' and 'eligibility' not in incumbent
    for command,speed,enabled in [('GO STRAIGHT',5,True),('GO STRAIGHT',0,False),('GO STRAIGHT',20,False),('UNKNOWN',5,False)]:
        x={**skill,'evidence':dict(navigation_command=command,speed_mps=speed)}
        assert g.apply_gate(x,candidate)['enabled'] is enabled


@pytest.mark.parametrize('change',[lambda v:v['diagnosis'].update(supporting_evidence=['made-up']),
    lambda v:v['diagnosis'].update(supporting_evidence=['case-01.caption.0']),
    lambda v:v['eligibility'].update(speed_max_mps=float('nan')),
    lambda v:v['eligibility'].update(commands=['A PRIVATE SCENE']),
    lambda v:v.update(skill_text='Drive at 5 m/s. '+v['skill_text'])])
def test_invalid_grounding_or_gate_is_rejected(change):
    initial,value,evidence=example();change(value)
    with pytest.raises(ValueError):g.validate(value,'following',initial,evidence)


def test_feedback_includes_paired_counterevidence_excludes_validation():
    members=[dict(token=t,role=role,skill=dict(id='following',evidence=dict(navigation_command='STOP',speed_mps=2)))
        for t,role in [('private-a','development'),('private-b','development'),('private-validation','validation')]]
    rows=[]
    for m in members:
        for arm in ('native','incumbent','candidate'):
            for seed in (42,43):
                metrics={k:1.0 for k in g.METRICS}
                if m['token']=='private-a' and arm=='candidate':metrics.update(NC=0,PDMS=0)
                rows.append(dict(token=m['token'],arm=arm,seed=seed,metrics=metrics))
    out=g.feedback(rows,members,'candidate',{'private-validation':{'secret':'VALIDATION_SECRET'}})['following']
    assert out['scene_count']==2 and len(out['difficult_observations'])==2
    first=out['difficult_observations'][0]
    assert first['safety_regression'] and first['comparisons']['native']['NC']==1
    assert first['comparisons']['candidate']['NC']==0
    assert 'private-' not in json.dumps(out) and 'VALIDATION_SECRET' not in json.dumps(out)


def test_reset_hides_history_but_preserves_current_skill_and_revision():
    state={'events':[{'id':'e0'}],'revision':1,'active':{'version':'promoted'}}
    reset=g.visible_memory(state,'reset')
    assert reset['events']==[] and reset['active']==state['active'] and reset['revision']==1
    assert state['events'] and g.visible_memory(state,'retain')==state
    with pytest.raises(ValueError):g.visible_memory(state,'unknown')


@pytest.mark.parametrize('active_arm',['incumbent','candidate'])
def test_reset_feedback_cannot_reveal_obsolete_candidate_or_incumbent(active_arm):
    rows=[dict(arm=a,private_value=a) for a in ('native','incumbent','candidate')]
    filtered,arm=g.feedback_rows(rows,active_arm,'reset',1)
    assert arm=='incumbent' and {r['arm'] for r in filtered}=={'native','incumbent'}
    assert {r['private_value'] for r in filtered}=={'native',active_arm}


def test_serialized_grounded_request_has_only_actual_diagnostic_ids():
    evidence=[dict(id='case-01.native.NC',kind='measured_metric',value=1)]
    job=dict(prompt='diagnose',evidence=[],diagnosis_evidence_ids=['case-01.native.NC'],facts=dict(development={'evidence':evidence}))
    body=g.skill_doctor.request_payload(job,dict(max_tokens=1600,enable_thinking=False))
    text=body['messages'][0]['content'][0]['text']
    assert 'Allowed evidence_ids' not in text and 'Allowed diagnosis evidence references: ["case-01.native.NC"]' in text
    assert body['max_tokens']==1600 and body['temperature']==0


def test_same_initial_request_then_experience_only_difference(tmp_path,monkeypatch):
    initial,value,evidence=example();requests=[]
    def fake(job,*args,**kwargs):
        requests.append(job);return dict(content=json.dumps(value),finish_reason='stop',usage={},seconds=0)
    monkeypatch.setattr(g.skill_doctor,'cached_request',fake)
    dev={'following':dict(mean_metrics={'PDMS':.4},evidence=evidence)}
    state=dict(active=initial,revision=0,events=[])
    for mode in ['retain','reset']:assert g.propose(dev,state,tmp_path/mode,'fixture','fixture',mode)['valid']
    assert requests[0]==requests[1]
    candidate=g.validate(value,'following',initial,evidence)
    state['events']=[dict(skill_id='following',candidate=candidate,decision=dict(accepted=False,reasons=['safety']),development=dev)]
    state['revision']=1
    keep=g.propose(dev,state,tmp_path/'k','fixture','fixture','retain')
    reset=g.propose(dev,state,tmp_path/'r','fixture','fixture','reset')
    assert keep['experience_count']==1 and reset['experience_count']==0
    assert requests[-1]['facts']['persisted_experience']==[]
