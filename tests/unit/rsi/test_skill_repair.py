import copy
import pytest
from driversi.rsi.decisions.skill_repair import ARMS,NODE,variants,validate_contract,decide

def preset():
    return dict(version='base',low_speed_mps=2,skills={NODE:'old text','turn_conflict_check':'turn','low_speed_progress_check':'slow'})

def test_factorial_separates_text_and_gate_without_mutating_incumbent():
    old=preset();new=copy.deepcopy(old);new['skills'][NODE]='new text'
    new['eligibility']={NODE:dict(commands=['GO STRAIGHT'],speed_min_mps=4,speed_max_mps=8,exclude_stop=True)}
    v=variants(old,new)
    assert v['text_only']['skills'][NODE]=='new text' and 'eligibility' not in v['text_only']
    assert v['gate_only']['skills'][NODE]=='old text' and v['gate_only']['eligibility']==new['eligibility']
    assert v['combined']['skills'][NODE]=='new text' and old==preset()

def test_claimed_speed_fix_must_actually_change_failed_case_activation():
    old=preset();new=copy.deepcopy(old);new['skills'][NODE]='new text'
    new['eligibility']={NODE:dict(commands=['GO STRAIGHT'],speed_min_mps=3,speed_max_mps=8,exclude_stop=True)}
    cases={'case-01':dict(observed_state=dict(navigation_command='GO STRAIGHT',speed_mps=3.59))}
    contract=dict(target_metric='TTC',target_cases=['case-01'],gate_changes=[dict(case_id='case-01',expected_enabled=False)])
    checked=validate_contract(contract,old,new,cases,['case-01'])
    assert not checked['gate_claims_passed'] and checked['gate_checks'][0]['after']
    new['eligibility'][NODE]['speed_min_mps']=4
    assert validate_contract(contract,old,new,cases,['case-01'])['gate_claims_passed']
    contract['target_cases']=['validation-case']
    with pytest.raises(ValueError):validate_contract(contract,old,new,cases,['case-01'])

def fixture():
    members=[dict(token='target',role='development',repair_family=True,case_id='case-01'),
             dict(token='protected',role='development',repair_family=False,case_id='case-02'),
             dict(token='family',role='validation',validation_group='family'),
             dict(token='control',role='validation',validation_group='control')]
    rows=[]
    for m in members:
        for a in ARMS:
            metric={k:1.0 for k in ('NC','DAC','DDC','C','EP')}
            target=m['token'] in ('target','family');metric.update(PDMS=.5 if target else 1.,TTC=0. if target else 1.)
            if target and a in ('text_only','combined'):metric.update(PDMS=.8,TTC=1.)
            rows.append(dict(token=m['token'],seed=42,arm=a,metrics=metric,raw_trajectory=[[1 if a in ('text_only','combined') else 0]]))
    proposal=dict(valid=True,modification_check=dict(target_cases=['case-01'],text_changed=True,gate_claims_passed=True))
    obligations=[dict(token='protected',seed=42,origin='old_regression')]
    return rows,members,proposal,obligations

def test_promotion_reuses_original_gates_and_general_controls_need_no_gain():
    rows,ms,p,reg=fixture();d=decide(rows,ms,[42],p,reg)
    assert d['selected_variant']=='text_only' and d['decision']['accepted']
    assert d['variants']['text_only']['regression']['passed'] and not d['selection_uses_validation']

def test_no_validation_cherry_pick_after_development_selection():
    rows,ms,p,reg=fixture()
    for r in rows:
        if r['token']=='family' and r['arm']=='text_only':r['metrics']['PDMS']=.4
    d=decide(rows,ms,[42],p,reg)
    assert d['selected_variant']=='text_only' and not d['decision']['accepted']
    assert d['variants']['combined']['full_decision']['accepted']

def test_remembered_regression_and_effect_failures_cannot_promote():
    rows,ms,p,reg=fixture()
    for r in rows:
        if r['token']=='protected' and r['arm'] in ('text_only','combined'):r['metrics']['DDC']=0
    d=decide(rows,ms,[42],p,reg)
    assert d['selected_variant'] is None and not d['decision']['accepted']
    assert 'remembered_regression_failed' in d['variants']['text_only']['development_reasons']
    rows,ms,p,reg=fixture()
    for r in rows:
        if r['token']=='target' and r['arm']!='native':r['metrics']['TTC']=0
    assert decide(rows,ms,[42],p,reg)['selected_variant'] is None

def test_missing_rows_fail_closed():
    rows,ms,p,reg=fixture()
    with pytest.raises(ValueError):decide(rows[:-1],ms,[42],p,reg)


def test_native_prompts_preserve_fallback_and_factorial_text():
    from driversi_agents.augustus.skill_inference import evolution_prompts
    prompts=evolution_prompts('native exact\n',{
        'incumbent':dict(text='old',enabled=True),'text_only':dict(text='new',enabled=True),
        'gate_only':dict(text='old',enabled=False),'combined':dict(text='new',enabled=False)})
    assert tuple(prompts)==ARMS
    assert prompts['gate_only']==prompts['combined']==prompts['native']=='native exact\n'
    assert prompts['incumbent'].endswith('old') and prompts['text_only'].endswith('new')


def test_real_decision_stage_persists_rollback_and_reuses_regression(tmp_path,monkeypatch):
    from driversi.rsi.experiments import skill_repair as stage
    from driversi.rsi.decisions.skill_evolution import memory,identity
    from driversi.agents.backends.native import digest
    from driversi.runtime.budget import write_json
    rows,ms,p,reg=fixture()
    for r in rows:
        if r['token']=='protected' and r['arm']=='text_only':r['metrics']['DDC']=0
    db=str(tmp_path/'state.sqlite');memory(db,preset())
    plan=dict(memory=db,seeds=[42],task_id='test',initial_regressions=reg)
    prep=tmp_path/'prepared.json';write_json(prep,dict(plan=plan,members=ms))
    scores=tmp_path/'scores.json';write_json(scores,rows)
    prop=tmp_path/'proposal.json';p.update(variants={a:preset() for a in ARMS[2:]},incumbent=preset(),candidate=preset(),
        prepared=str(prep),prepared_sha256=digest(prep),episode=0,memory_revision=0,regression_obligations=reg,diagnosis=None)
    write_json(prop,p)
    scoring=tmp_path/'scoring.json';write_json(scoring,dict(prepared_sha256=digest(prep),scores=str(scores),scores_sha256=digest(scores)))
    monkeypatch.setattr(stage.grounded,'feedback',lambda *a,**k:{})
    result=stage.decide(prop,scoring,tmp_path/'decision')
    state=memory(db)
    assert state['revision']==1 and identity(state)==result['memory_after_hash']
    assert state['events'][0]['variant_feedback']['text_only']['development_eligible'] is False
    assert set(state['events'][0]['variant_definitions'])==set(ARMS[2:])
    assert state['events'][0]['proposed_preset']==p.get('candidate',preset())
    assert any(r['token']=='protected' for r in stage.obligations_for(plan,state))
    # Exact stage replay must not append experience twice.
    assert stage.decide(prop,scoring,tmp_path/'decision')['memory_revision']==1


def test_prior_regression_import_never_uses_validation(tmp_path):
    from driversi.rsi.experiments.skill_repair import imported_regressions
    from driversi.runtime.budget import write_json
    rows,_,_,_=fixture();old=[]
    for r in rows:
        if r['arm'] not in ('incumbent','text_only'):continue
        v=copy.deepcopy(r);v['arm']='candidate' if v['arm']=='text_only' else v['arm']
        if v['arm']=='candidate':v['metrics']['DDC']=0
        old.append(v)
    p=tmp_path/'old.json';write_json(p,old)
    obligations=imported_regressions([p],{'target','protected'})
    assert {r['token'] for r in obligations}=={'target','protected'}


def test_doctor_receives_historical_interventions_and_named_variant_definitions(tmp_path,monkeypatch):
    from driversi.rsi.decisions import skill_repair as repair
    full=preset();full['skills'][NODE]='full proposed text'
    event=dict(candidate=preset(),proposed_preset=full,diagnostic_variant='gate_only',selected_variant=None,
        variant_definitions={'text_only':dict(text='full proposed text',eligibility=None)},
        variant_feedback={'text_only':dict(target_ttc_delta=0)},decision=dict(accepted=False,reasons=['no_gain']))
    captured={}
    def request(job,*args):
        captured.update(job);return dict(finish_reason='stop',content='{}',usage={},seconds=0)
    monkeypatch.setattr(repair.skill_doctor,'cached_request',request)
    group=dict(difficult_observations=[dict(case_id='case-01')],evidence=[])
    result=repair.propose({NODE:group},dict(active=preset(),revision=0,events=[]),tmp_path,'config','credential',['case-01'],[],[event])
    h=captured['facts']['persisted_experience'][0]
    assert h['proposed_text']=='full proposed text' and h['diagnostic_variant']=='gate_only'
    assert h['variant_definitions']['text_only']['text']=='full proposed text'
    assert result['experience_count']==1 and not result['valid']
