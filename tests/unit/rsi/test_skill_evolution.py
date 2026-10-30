import copy
import json

import pytest

from driversi.rsi.decisions.skill_evolution import acquisition_group, evaluate, identity, memory, validate_proposal
from driversi.rsi.decisions.skill_doctor import feedback, cached_request, response_fields, MODEL


def fixture():
    members=[dict(token=t,role=role,skill=dict(id='following',evidence={'speed_mps':3,'navigation_command':'GO STRAIGHT'}))
             for t,role in [('a','development'),('b','validation')]]
    rows=[]
    for m in members:
        for seed in [42,43]:
            for arm in ['native','incumbent','candidate']:
                metrics={k:1.0 for k in ['NC','DAC','DDC','TTC','C','EP']}
                metrics['PDMS']=0.6 if arm=='candidate' else 0.5
                rows.append(dict(token=m['token'],seed=seed,arm=arm,metrics=metrics))
    return members,rows


def test_cross_scene_gate_and_zero_regression():
    members,rows=fixture()
    assert evaluate(rows,members,[42,43])['accepted']
    rows[-1]['metrics']['PDMS']=0
    report=evaluate(rows,members,[42,43])
    assert not report['accepted'] and 'validation:new_zero_vs_native' in report['reasons']


def test_safety_veto_even_if_mean_improves():
    members,rows=fixture();rows[-1]['metrics']['NC']=0
    assert not evaluate(rows,members,[42,43])['accepted']


def test_no_transfer_is_not_promotion():
    members,rows=fixture()
    for r in rows:
        if r['token']=='b':r['metrics']['PDMS']=0.5
    assert 'validation:no_positive_transfer_over_incumbent' in evaluate(rows,members,[42,43])['reasons']


def test_incomplete_nonfinite_and_duplicate_evidence_rejected():
    members,rows=fixture()
    for broken in [rows[:-1],rows+[rows[-1]], [{**r,'metrics':{**r['metrics'],'PDMS':float('nan')}} for r in rows]]:
        with pytest.raises(ValueError):evaluate(broken,members,[42,43])


def test_feedback_excludes_validation_and_identifiers():
    members,rows=fixture();rows[-1]['metrics']['PDMS']=0.123456789
    facts=feedback(rows,members,'candidate')
    assert facts['following']['scene_count']==1
    assert '0.123456789' not in json.dumps(facts) and 'token' not in json.dumps(facts)


def test_atomic_memory_promotion_rejection_and_replay(tmp_path):
    initial={'version':'base','skills':{'x':'base'}};candidate={'version':'next','skills':{'x':'next'}}
    path=tmp_path/'state.sqlite';memory(path,initial)
    event=dict(id='e0',incumbent_hash=identity(initial),candidate=candidate,decision={'accepted':True})
    accepted=memory(path,event=event,expected_revision=0)
    assert accepted['active']==candidate and accepted['revision']==1
    assert memory(path,event=event,expected_revision=0)==accepted
    rejected=dict(id='e1',incumbent_hash=identity(candidate),candidate=initial,decision={'accepted':False})
    state=memory(path,event=rejected,expected_revision=1)
    assert state['active']==candidate and len(state['events'])==2
    with pytest.raises(ValueError):memory(path,event={**rejected,'id':'stale'},expected_revision=1)
    assert memory(path)==state
    with pytest.raises(ValueError):memory(path,event={**event,'candidate':initial},expected_revision=0)


def test_doctor_cannot_edit_other_nodes_or_fixed_numeric_action():
    initial={'version':'base','skills':{'x':'base','y':'untouched'}}
    value=dict(skill_id='x',skill_text='Inspect the visible road boundaries and follow the navigation command. Maintain smooth progress when the observed route is clear.',rationale='feedback',expected_effect='boundary compliance')
    candidate=validate_proposal(value,'x',initial)
    assert candidate['skills']['y']=='untouched' and initial['skills']['x']=='base'
    for changed in [{**value,'skill_id':'y'},{**value,'skill_text':value['skill_text']+' Brake at 5 m/s.'}]:
        with pytest.raises(ValueError):validate_proposal(changed,'x',initial)


def test_unknown_request_is_not_repeated_and_saved_response_is_reused(tmp_path):
    job={'prompt':'fixture','facts':{},'evidence':[]}
    (tmp_path/'request-intent.json').write_text(json.dumps({'job_hash':identity(job)}))
    with pytest.raises(RuntimeError,match='unknown'):
        cached_request(job,tmp_path,'must-not-read','must-not-read')
    saved={'job_hash':identity(job),'content':'cached'}
    (tmp_path/'response.json').write_text(json.dumps(saved))
    assert cached_request(job,tmp_path,'must-not-read','must-not-read')==saved


def test_drive_groups_exclude_adjacent_segments():
    assert acquisition_group('2021.05.25.14.16.10_veh-35_00083_00485') == acquisition_group('2021.05.25.14.16.10_veh-35_01100_01664')


@pytest.mark.parametrize('reply',[{},None,{'choices':[]},{'choices':[{'message':{}}]},
    {'choices':[{'message':{'content':None},'finish_reason':'stop'}]}])
def test_known_malformed_reply_is_not_unknown(reply):
    assert response_fields(reply)['response_error']=='invalid_response_shape'


def test_malformed_reply_cached_and_not_reissued(tmp_path,monkeypatch):
    from driversi.data.catalog import bailian_caption as api
    monkeypatch.setattr(api,'private_config',lambda _:dict(models=[MODEL],base_url='private'))
    monkeypatch.setattr(api,'credential',lambda _:'fixture-only-not-a-key')
    monkeypatch.setattr(api,'private_endpoint',lambda _:None)
    monkeypatch.setattr(api,'payload',lambda *args:{})
    calls=[]
    def transport(*args):calls.append(1);return {'choices':[]}
    job={'prompt':'fixture','facts':{},'evidence':[]}
    a=cached_request(job,tmp_path,'fixture','fixture',transport)
    b=cached_request(job,tmp_path,'fixture','fixture',transport)
    assert a==b and len(calls)==1 and a['response_error']=='invalid_response_shape'


def test_native_skill_metadata_is_explicitly_absent():
    from driversi_agents.augustus.skill_inference import skill_metadata
    member={'skill':{'text':'candidate'},'arm_skills':{'incumbent':{'text':'old'},'candidate':{'text':'candidate'}}}
    assert skill_metadata(member,'native',True) is None
    assert skill_metadata(member,'candidate',True)==member['arm_skills']['candidate']
    assert skill_metadata(member,'native',False)==member['skill']


def test_arm_prompt_identity():
    from driversi_agents.augustus.skill_inference import evolution_prompts
    base='unchanged native input\n'
    prompts=evolution_prompts(base,{'incumbent':{'text':'old skill'},'candidate':{'text':'new skill'}})
    assert prompts['native']==base
    assert prompts['incumbent']==base.rstrip()+'\n\nPlanning checklist for this observation:\nold skill'
    assert prompts['candidate']==base.rstrip()+'\n\nPlanning checklist for this observation:\nnew skill'


@pytest.mark.parametrize('accepted',[False,True])
def test_next_doctor_uses_committed_incumbent_and_failed_feedback(tmp_path,monkeypatch,accepted):
    from driversi.rsi.decisions import skill_doctor
    initial={'version':'base','skills':{'following':'initial'}}
    candidate={'version':'candidate','skills':{'following':'previous candidate'}}
    memory(tmp_path/'memory.sqlite',initial)
    development={'following':{'mean_metrics':{'PDMS':0.4},'zero_pairs':2}}
    event=dict(id='episode-0',skill_id='following',incumbent_hash=identity(initial),candidate=candidate,
               decision={'accepted':accepted,'reasons':[] if accepted else ['development:new_zero_vs_incumbent']},
               development=development)
    state=memory(tmp_path/'memory.sqlite',event=event,expected_revision=0)
    captured=[]
    reply=dict(skill_id='following',skill_text='Use the observed history to check forward separation and visible road boundaries. Preserve useful progress when the visible corridor is clear.',rationale='Bounded revision from previous feedback.',expected_effect='Fewer observed conflicts.')
    def fake(job,*args):
        captured.append(job)
        return dict(content=json.dumps(reply),finish_reason='stop',usage={},seconds=0)
    monkeypatch.setattr(skill_doctor,'cached_request',fake)
    result=skill_doctor.propose(development,state,tmp_path/'doctor','fixture','fixture')
    facts=captured[0]['facts']
    assert result['valid'] and result['experience_count']==1
    assert facts['incumbent_text']==('previous candidate' if accepted else 'initial')
    assert facts['persisted_experience'][0]['development']==development
    assert facts['persisted_experience'][0]['reasons']==event['decision']['reasons']
    assert facts['validation_examples_available'] is False


def test_transport_unknown_stays_unresolved_without_resend(tmp_path,monkeypatch):
    from driversi.data.catalog import bailian_caption as api
    monkeypatch.setattr(api,'private_config',lambda _:dict(models=[MODEL],base_url='private'))
    monkeypatch.setattr(api,'credential',lambda _:'fixture-only-not-a-key')
    monkeypatch.setattr(api,'private_endpoint',lambda _:None)
    monkeypatch.setattr(api,'payload',lambda *args:{})
    calls=[]
    def fail(*args):calls.append(1);raise api.APIError('network_outcome_unknown_no_automatic_retry')
    job={'prompt':'fixture','facts':{},'evidence':[]}
    with pytest.raises(api.APIError):cached_request(job,tmp_path,'fixture','fixture',fail)
    with pytest.raises(RuntimeError,match='unknown'):cached_request(job,tmp_path,'fixture','fixture',fail)
    assert len(calls)==1 and not (tmp_path/'response.json').exists()
