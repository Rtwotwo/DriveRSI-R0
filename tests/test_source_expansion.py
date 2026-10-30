"""Bounded breadth decisions and terminal reconciliation, no GPU execution."""
import copy
import json
from pathlib import Path
import pytest
from driversi.rsi.experiments import source_expansion as se
from driversi.rsi.experiments import cohort_recovery as cr
from driversi.rsi.experiments import problem_cohort


def fixture():
    case=dict(id='p',round=0,selected_sources=3,generation_revisions={'0':1},
        candidates=[dict(token=str(i),source_log=f'log{i}',max_similarity=1-i*.04) for i in range(16)])
    policy=dict(attempts_per_source=32,max_rounds=3,targets=[10,20,20])
    rows=[dict(status='trajectory_duplicate_before_render',trainable=False) for _ in range(16)]
    return case,policy,dict(rows=rows)


def decision(case=None,summary=None,spent=None,asset_failure=False):
    c,p,s=fixture()
    return se.propose(case or c,p,spent or {},['0','2','4'],s if summary is None else summary,
        se.DEFAULT_POLICY,asset_failure=asset_failure)


def test_expands_preserving_existing_and_actual_ranks_not_coordinates():
    c,p,s=fixture()
    for r in c['candidates']:r.update(world_xy=[999,999],heading=3.14)
    d=decision(c,s)
    assert d['tokens']==['0','2','4','1','3']
    assert d['original_visual_ranks']==[1,3,5,2,4]
    assert d['limit']==5 and d['location_used'] is False
    assert d['pdms_strict_gt']==.95 and d['psnr_strict_gt']==25.


@pytest.mark.parametrize('reason',['execution_failed','visual_rejected','visual_review_incomplete','native_language_review_incomplete'])
def test_unknown_or_quality_failure_never_relabels_as_source_shortage(reason):
    c,p,s=fixture();s['rows'].append(dict(status=reason))
    assert decision(c,s) is None


def test_no_expansion_for_short_batch_good_yield_or_before_strategy_revision():
    c,p,s=fixture();s['rows']=s['rows'][:15]
    assert decision(c,s) is None
    c,p,s=fixture();c['generation_revisions']['0']=0
    assert decision(c,s) is None
    c,p,s=fixture();s['rows']=[dict(status='training_sample_ready',trainable=True)]*10+[
        dict(status='teacher_rejected',trainable=False)]*10
    assert decision(c,s) is None


def test_carried_samples_do_not_inflate_fresh_yield():
    c,p,s=fixture();s['rows'] += [dict(status='training_sample_ready',trainable=True,reused_prior_cohort=True)]*50
    d=decision(c,s)
    assert d['fresh_qualified']==0 and d['fresh_attempts']==16


def test_progression_caps_and_no_budget_reset():
    c,p,s=fixture();d=decision(c,s,{'0':90,'2':80,'4':60})
    assert '0' not in d['tokens'] and d['charged_attempts']['0']==90
    se.commit(c,d);d2=decision(c,s);assert d2['limit']==8
    se.commit(c,d2);d3=decision(c,s);assert d3['limit']==10
    se.commit(c,d3);assert decision(c,s) is None
    assert c['generation_revisions']['0']==1


def test_known_failed_assets_are_excluded_and_ready_asset_cannot_wake_partial_pool():
    c,p,s=fixture();c['unavailable_source_tokens']=['1','3']
    d=decision(c,{},asset_failure=True);assert d['tokens']==['0','2','4','5','6']
    se.commit(c,d)
    assert not se.readiness(c,{'0':{},'2':{},'4':{}})
    assert se.readiness(c,{t:{} for t in d['tokens']})


def test_short_or_exhausted_pool_is_not_invented():
    c,p,s=fixture();c['candidates']=c['candidates'][:3]
    assert se.propose(c,p,{},['0','1','2'],s,se.DEFAULT_POLICY) is None


def test_duplicate_logs_not_allowed():
    c,p,s=fixture();c['candidates'][3]['source_log']='log0'
    with pytest.raises(ValueError,match='Distinct'):decision(c,s)


def test_rebuild_preserves_carried_qualified_and_separates_new_rejects(monkeypatch):
    prior=[dict(candidate='old',anchor_assignment_id='a',trainable=True,rendered=True,
        status='training_sample_ready',sample_report='sample',dataset_reference=dict(package_id='package'),
        reused_prior_cohort=True)]
    monkeypatch.setattr(problem_cohort,'seed',lambda *args:copy.deepcopy(prior))
    cfg=dict(trajectory_family=dict(target_per_anchor=10))
    plan=dict(assignments=[dict(id='new',anchor_assignment_id='a')])
    outcomes=[dict(candidate='new',anchor_assignment_id='a',status='teacher_rejected',trainable=False)]
    old=dict(status='completed_with_shortages',qualified=1,attempted=1,rows=outcomes)
    fixed,gate=cr.rebuild(cfg,plan,old,outcomes)
    assert fixed['qualified']==1 and len(fixed['rows'])==2 and fixed['reused_prior_cohort']==1
    assert not gate['passed'] and gate['shortages']['a']==9
    assert old['rows']==outcomes
    with pytest.raises(ValueError,match='witnesses'):cr.rebuild(cfg,plan,dict(old,qualified=2),outcomes)
    with pytest.raises(ValueError,match='source'):cr.rebuild(cfg,plan,old,[dict(outcomes[0],anchor_assignment_id='wrong')])


def test_accounting_recognition_is_exact_and_never_training(tmp_path):
    case=dict(status='unknown_execution',current_round=str(tmp_path/'round'))
    p=Path(case['current_round'])/'waves/wave-0001/production-failure.txt';p.parent.mkdir(parents=True)
    p.write_text('ValueError: Unrelated error\n');assert not cr.eligible(tmp_path,case)
    p.write_text(cr.ERROR+'\n');assert cr.eligible(tmp_path,case)
    case['status']='training';assert not cr.eligible(tmp_path,case)


def test_policy_is_opt_in_and_bounded(tmp_path):
    assert se.policy(tmp_path) is None
    p=tmp_path/'source-expansion-policy.json';p.write_text(json.dumps(se.DEFAULT_POLICY))
    assert se.policy(tmp_path)['levels']==[3,5,8,10]
    p.write_text(json.dumps(dict(se.DEFAULT_POLICY,levels=[3,100])))
    with pytest.raises(ValueError):se.policy(tmp_path)


def test_global_quota_can_use_one_remaining_attempt_and_replace_exhausted_source():
    c,p,s=fixture();p['cohort_mode']='global_50_min1_v1'
    c['source_search']=dict(limit=10,tokens=[str(i) for i in range(10)])
    spent={'0':96,'1':95}
    d=se.propose(c,p,spent,c['source_search']['tokens'],dict(rows=[dict(status='teacher_rejected')]),se.DEFAULT_POLICY)
    assert d['limit']==10 and '0' not in d['tokens'] and '1' in d['tokens'] and '10' in d['tokens']
    assert d['per_source_target']==1 and d['reason']=='settled_source_attempt_budget_exhausted'
    assert se.propose(c,p,spent,c['source_search']['tokens'],dict(rows=[dict(status='execution_failed')]),se.DEFAULT_POLICY) is None


def test_unknown_failure_does_not_become_a_storage_retry(tmp_path):
    from driversi.rsi.experiments.continual_execution import registered_recovery,next_operation
    (tmp_path/'storage-recovery-policy.json').write_text(json.dumps(dict(rule='png_cleanup_terminal_v1')))
    case=dict(id='p',status='unknown_execution',current_round=str(tmp_path/'round'))
    assert registered_recovery(tmp_path,case) is None
    memory=tmp_path/'recovery-memory';memory.mkdir()
    (memory/'event-001.json').write_text(json.dumps(dict(status='recovered',wave=str(tmp_path/'round/waves/wave-0001'))))
    assert registered_recovery(tmp_path,case) is None
    case['status']='asset_recovery_pending'
    assert next_operation(dict(active='p',cases=[case]))==(None,'asset_recovery_in_progress')


@pytest.mark.parametrize('k',[5,8,10])
@pytest.mark.parametrize('target',[10,20])
def test_expanded_selection_fits_real_production_and_sft_budgets(k,target):
    from driversi.rsi.decisions.augmentation_budget import build_budget
    from driversi.rsi.decisions.training_recipe import choose
    c,p,s=fixture();c['source_search']=dict(tokens=[str(i) for i in range(k)],limit=k)
    rows=se.select(c,c['candidates'],target)
    budget=build_budget(rows,[],[],target=target,wave_size=len(rows))
    assert budget['target_qualified_samples']<=128
    recipe=choose(len(rows)*target,0 if target==10 else 1,[6],256)
    assert recipe['optimizer_updates_per_arm']<=256 and recipe['epochs']>=4
    assert budget['target_per_anchor']==target
    if k==10 and target==10:assert len(rows)==10


def test_final_asset_tier_uses_verified_subset_without_lowering_quota(tmp_path,monkeypatch):
    from driversi.rsi.experiments import continual_curriculum as cc,asset_recovery,generation_policy
    (tmp_path/'source-expansion-policy.json').write_text(json.dumps(se.DEFAULT_POLICY))
    c,p,s=fixture();c.update(status='assets_unresolved',current_round=str(tmp_path/'round'),
        source_search=dict(limit=10,tokens=['0','2','4']),asset_recovery=[dict(token='2',status='source_preparation_incomplete')])
    state=dict(active='p',cases=[c],policy=p)
    (tmp_path/'round').mkdir();(tmp_path/'round/round.json').write_text(json.dumps(dict(selected_training_tokens=['0','2','4'])))
    monkeypatch.setattr(cc,'state_path',lambda root:tmp_path/'state.json')
    (tmp_path/'state.json').write_text(json.dumps(state))
    saved=[];monkeypatch.setattr(cc,'save',lambda root,value:saved.append(copy.deepcopy(value)))
    monkeypatch.setattr(cc,'prepare',lambda root:'prepared')
    monkeypatch.setattr(asset_recovery,'ready_sources',lambda root:{'0':{},'4':{}})
    monkeypatch.setattr(generation_policy,'started_attempts',lambda *a:{'0':71,'4':44})
    assert se.after_assets(tmp_path)=='prepared'
    case=saved[-1]['cases'][0]
    assert case['source_search']['tokens']==['0','4']
    assert case['source_search']['limit']==10 and case['source_search']['original_visual_ranks']==[1,5]
    assert not case.get('source_expansions')
    assert saved[-1]['policy']['targets']==[10,20,20]
