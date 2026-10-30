import numpy as np
import pytest

from driversi.rsi.decisions.environment_speed import policy_from_facts, adapt_grid, pair_speed
from driversi.rsi.experiments.generation_policy import spatial_grid
from driversi.rsi.decisions.trajectory_diversity import compare


def context(speed=8., curvature=.001, gap=None, observed=True):
    return policy_from_facts(dict(anchor_speed_mps=speed,observed_curvature_inv_m=curvature,
        forward_box_gap_m=gap,objects_observed=observed))


def test_curvature_and_near_object_tighten_not_relax_speed():
    normal=context()
    assert context(curvature=.1)['max_positive_delta_mps']==0
    assert context(gap=6.)['max_positive_delta_mps']==0
    assert context(curvature=None)['max_positive_delta_mps']<normal['max_positive_delta_mps']
    assert context(observed=False)['max_positive_delta_mps']<normal['max_positive_delta_mps']
    assert context(speed=.1)['max_negative_delta_mps']<=.1
    assert context(speed=.1)['minimum_speed_mae_mps']==.3
    assert context(speed=30.)['minimum_speed_mae_mps']==1.


def test_clipped_grid_is_unique_and_never_inflates_budget():
    for policy in (context(),context(gap=6.),context(speed=0.,curvature=None)):
        rows=adapt_grid(spatial_grid(),policy,32)
        assert 10<=len(rows)<=32
        assert len({tuple(sorted(row.items())) for row in rows})==len(rows)
        assert all(-policy['max_negative_delta_mps']-1e-7<=r['speed_delta_mps']<=policy['max_positive_delta_mps']+1e-7 for r in rows)
    assert {r['speed_delta_mps'] for r in adapt_grid(spatial_grid(),context(),32)}=={-2.,0.,1.,2.}


def test_mean_and_sustained_speed_difference_required():
    t=np.linspace(0,7,71)
    a=np.column_stack([8*t,np.zeros(71)])
    b=np.column_stack([9*t,np.zeros(71)])
    assert pair_speed(a,b,context())['speed_distinct']
    jitter=a.copy();jitter[30,0]+=1.
    assert not pair_speed(a,jitter,context())['speed_distinct']
    tiny=np.column_stack([8.1*t,np.zeros(71)])
    assert not pair_speed(a,tiny,context())['speed_distinct']


def test_speed_distinction_does_not_replace_spatial_gate():
    t=np.linspace(0,7,71);env=context()
    def row(v,name):return dict(candidate_id=name,comparison_key={'source':1},
        xy=np.column_stack([v*t,np.zeros(71)]).tolist(),environment_speed_policy=env)
    result=compare(row(9,'b'),[row(8,'a')],minimum_path_ade_m=.5)
    assert result['comparisons'][0]['speed_distinct']
    assert not result['distinct']


def test_unsubstantiated_policy_changes_are_rejected():
    policy=context();policy['max_positive_delta_mps']=999
    with pytest.raises(ValueError):adapt_grid(spatial_grid(),policy,32)


def test_source_history_serializes_and_does_not_use_future(tmp_path):
    import pickle,json
    from driversi.rsi.decisions.environment_speed import source_policy
    frames=[dict(token=str(i),timestamp=i*500000,ego2global_translation=[i*4.,0,0],
        ego2global_rotation=[1,0,0,0],ego_dynamic_state=[8.,0,0,0],
        anns={'gt_boxes':np.zeros((0,7))}) for i in range(6)]
    path=tmp_path/'train.pkl';path.write_bytes(pickle.dumps(frames))
    first=source_policy(tmp_path,'train','4');json.dumps(first)
    frames[5]['ego_dynamic_state']=[1000,0,0,0];path.write_bytes(pickle.dumps(frames))
    second=source_policy(tmp_path,'train','4')
    first['facts'].pop('source_sha256');second['facts'].pop('source_sha256')
    assert first==second


def test_twenty_sample_quota_reports_sixteen_feasible_references(monkeypatch):
    from driversi.rsi.experiments.reactive_cohort import expand_plan
    monkeypatch.setattr('driversi.rsi.decisions.environment_speed.source_policy',
        lambda *args:context(speed=0.,curvature=None))
    policy=dict(method='history_endpoint_grid_tts_v1',proposal_grid=spatial_grid(),
        attempts_per_anchor=32,target_per_anchor=20,seed_start=42,
        environment_speed_context=dict(version='source_history_speed_envelope_v1',logs_root='/train'))
    original=dict(assignments=[dict(id='a',skill='perturbed_history_tts',source_log='train',token='one')])
    plan=expand_plan(original,policy)
    assert plan['proposal_capacity_shortages']=={'a':16}
    assert len(plan['assignments'])==16
