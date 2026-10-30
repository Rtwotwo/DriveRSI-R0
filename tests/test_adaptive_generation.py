import numpy as np
import pytest

from driversi.rsi.decisions.environment_speed import policy_from_facts
from driversi.rsi.decisions.adaptive_diversity import policy,compare,proposal_grid
from driversi.rsi.decisions.generation_advisor import validate_advice
from driversi.rsi.experiments.adaptive_generation import diagnose


def env(speed=4.):
    return policy_from_facts(dict(anchor_speed_mps=speed,observed_curvature_inv_m=0.,
        forward_box_gap_m=None,objects_observed=True))


def trajectory(speed=4.,side=0.):
    t=np.arange(71)*.1
    return dict(candidate_id=str((speed,side)),comparison_key={'source':'same'},
        xy=np.column_stack([t*speed,side*np.minimum(t/2,1)]).tolist(),
        environment_speed_policy=env())


def test_exact_duplicate_rejected():
    a=trajectory();assert not compare(a,[a],policy(env()))['distinct']


def test_speed_only_accepted_as_speed_not_spatial():
    a,b=trajectory(),trajectory(4.4)
    result=compare(b,[a],policy(env(),'speed'))
    assert result['distinct'] and result['comparisons'][0]['novelty_basis']=='speed'


def test_small_speed_jitter_and_history_only_change_rejected():
    a,b=trajectory(),trajectory(4.02)
    assert not compare(b,[a],policy(env()))['distinct']
    b=trajectory();b['xy'][:15]=trajectory(4.5)['xy'][:15]
    assert not compare(b,[a],policy(env()))['distinct']


def test_spatial_difference_accepted_without_speed_change():
    result=compare(trajectory(side=.45),[trajectory()],policy(env(),'spatial'))
    assert result['distinct'] and result['comparisons'][0]['spatial_distinct']


def test_first_member_and_fixed_pdms():
    assert compare(trajectory(),[],policy(env()))['distinct']
    p=policy(env());p['pdms_strict_gt']=.94
    with pytest.raises(ValueError):compare(trajectory(),[],p)


def test_context_bounded_grid_excludes_used():
    e=env();p=policy(e,'speed',1,'reduce_history_perturbation');grid=proposal_grid(e,p)
    assert len(grid)==64
    assert max(abs(r['lateral_m']) for r in grid)<=.26
    assert len({tuple(r.values()) for r in grid})==len(grid)
    assert not any(r in grid[:10] for r in proposal_grid(e,p,used=grid[:10]))


def test_recovery_responds_to_actual_failure_and_caps():
    summary=dict(rows=[dict(status='teacher_rejected')]*8,qualified=0,rendered=0)
    assert diagnose(summary,0)['action']=='reduce_history_perturbation'
    assert diagnose(summary,2)['action']=='generation_budget_exhausted'
    summary['rows']=[dict(status='trajectory_duplicate_before_render')]*2
    assert diagnose(summary,0)['action']=='adapt_spatial_or_speed_diversity'
    summary['rows'].append(dict(status='execution_failed'))
    assert diagnose(summary,0)['action']=='needs_execution_reconciliation'


def test_advisor_cannot_return_thresholds_or_arbitrary_camera():
    row=dict(preference='speed',observations=[dict(camera='CAM_F0',observation_zh='road')],uncertainty_zh='unknown')
    assert validate_advice(row)==row
    with pytest.raises(ValueError):validate_advice(dict(row,pdms=.90))
    row['observations'][0]['camera']='invented'
    with pytest.raises(ValueError):validate_advice(row)


def test_two_source_expand_plan(monkeypatch):
    from driversi.rsi.experiments.reactive_cohort import expand_plan
    from driversi.rsi.decisions import environment_speed,generation_advisor
    from driversi.rsi.experiments.generation_policy import spatial_grid
    monkeypatch.setattr(environment_speed,'source_policy',lambda *a:env())
    monkeypatch.setattr(generation_advisor,'advise',lambda *a:dict(preference='conservative'))
    original=dict(assignments=[dict(id=str(i),token=str(i),source_log='source'+str(i),skill='perturbed_history_tts') for i in range(2)])
    family=dict(attempts_per_anchor=10,target_per_anchor=10,seed_start=42,method='history_endpoint_grid_tts_v1',proposal_grid=spatial_grid(),
        environment_speed_context=dict(version=environment_speed.VERSION,logs_root='source'),
        adaptive_context=dict(action='adapt_spatial_or_speed_diversity',revision=1))
    result=expand_plan(original,family)
    assert len(result['assignments'])==20
    assert all(a['adaptive_diversity_policy']['preference']=='speed' for a in result['assignments'])
    family['adaptive_context']['source_actions']={'0':'reduce_history_perturbation'}
    result=expand_plan(original,family)
    assert {a['adaptive_diversity_policy']['action'] for a in result['assignments'] if a['token']=='1'}=={'initial'}


def test_distinct_recovery_actions_and_per_learning_round_context(tmp_path):
    from driversi.rsi.experiments.adaptive_generation import context
    e=env()
    reduced=proposal_grid(e,policy(e,'speed',1,'reduce_history_perturbation'))
    alternate=proposal_grid(e,policy(e,'speed',1,'adapt_spatial_or_speed_diversity'))
    assert reduced!=alternate
    case=dict(id='problem-1',round=1,generation_adaptations={'0':dict(revision=2,action='reduce_history_perturbation')})
    assert context(tmp_path,case)['revision']==0


def test_known_arrival_rejection_is_not_unknown_failure():
    from driversi.rsi.experiments.processing_resume import known_arrival_rejection
    g=dict(status='teacher_rejected');t=dict(status='execution_or_admission_rejected',harness_status='failed')
    n=dict(status='rejected',reason='arrival_ego_terminated',arrival_events=dict(offroad=[True]),input_files={'scene':'hash'})
    assert known_arrival_rejection(g,t,n)
    assert not known_arrival_rejection(g,t,dict(n,error_type='RuntimeError'))
    assert not known_arrival_rejection(g,t,dict(n,arrival_events=dict(offroad=[False])))
    assert not known_arrival_rejection(g,t,dict(n,reason='arbitrary_failure'))


def test_future_physical_termination_is_known_but_never_admitted():
    from driversi.rsi.decisions.generation_outcomes import incomplete_physical_branch
    row=dict(completed_steps=1,failure='ego_terminated_before_complete_score_window',
        trajectory_admitted=False,score=None,events=dict(offroad=1))
    native=dict(status='completed',restore_exact=True,source_files={'source':'hash'},branches=dict(teacher=row))
    assert incomplete_physical_branch(native)
    for changed in [dict(row,score=.99),dict(row,completed_steps=50),dict(row,events={}),dict(row,failure='unknown')]:
        assert not incomplete_physical_branch(dict(native,branches=dict(teacher=changed)))
    assert not incomplete_physical_branch(dict(native,restore_exact=False))


def test_budget_prefix_covers_endpoint_and_speed_axes():
    e=env();rows=proposal_grid(e,policy(e,'conservative'))[:10]
    assert min(r['longitudinal_m'] for r in rows)<0<max(r['longitudinal_m'] for r in rows)
    assert min(r['lateral_m'] for r in rows)<0<max(r['lateral_m'] for r in rows)
    assert min(r['speed_delta_mps'] for r in rows)<0<max(r['speed_delta_mps'] for r in rows)
    assert rows==proposal_grid(e,policy(e,'conservative'))[:10]


def test_sources_choose_different_recoveries():
    from driversi.rsi.experiments.adaptive_generation import source_actions
    rows=[dict(candidate='a',status='teacher_rejected')]*3+[dict(candidate='b',status='trajectory_duplicate_before_render')]*3
    assignments=[dict(id='a',token='first'),dict(id='b',token='second')]
    value=source_actions(dict(rows=rows),assignments,0)
    assert value=={'first':'reduce_history_perturbation','second':'adapt_spatial_or_speed_diversity'}
