import copy
import json
import pytest
from driversi.evaluation.benchmarks.chainflow_pair_import import shared_identity, exact_changes, verify_shard_rows, METRICS
from driversi.evaluation.benchmarks.chainflow_pair import validate_updates
from driversi.evaluation.benchmarks.chainflow_pair_import import recovery_overhead


def test_superseded_cost_and_replay_are_visible_without_relabeling_reserved_usage(tmp_path):
    p=tmp_path/'tasks/paired';(p/'treatment_replay').mkdir(parents=True)
    (p/'treatment_replay/training-result.json').write_text(json.dumps(dict(status='completed',consumed=dict(wall_seconds=7))))
    (p/'report.json').write_text(json.dumps(dict(status='cancelled',stages=[dict(id='control_evaluation',status='outcome_unknown',elapsed_seconds=9,charged_agent_steps=1000,cost_basis='reserved_cap_unknown_usage')])) )
    value=recovery_overhead(tmp_path)
    assert value['resume_verification']['consumed']['wall_seconds']==7
    assert value['resume_verification']['included_in_unique_training_cost'] is False
    old=value['superseded_serial_evaluation']
    assert old['wall_seconds']==9 and old['stages'][0]['cost_basis']=='reserved_cap_unknown_usage'
    assert old['scoring_cost_may_be_reserved_upper_bound']


@pytest.mark.parametrize('updates',[8,40])
def test_declared_pair_updates_are_verified(updates):
    receipts={a:dict(status='completed',total_updates=updates) for a in ('control','treatment')}
    assert validate_updates(updates,receipts)==updates
    receipts['treatment']['total_updates']-=1
    with pytest.raises(ValueError,match='Actual training updates'):
        validate_updates(updates,receipts)


@pytest.mark.parametrize('updates',[True,0,257,40.0])
def test_invalid_pair_budget_rejected(updates):
    with pytest.raises(ValueError):validate_updates(updates,{})


def identities():
    return {(a,s):dict(checkpoint_sha256=a,shard=s,seed=42,shards=2,
            precision='32-true',tto=False,backbone_sha256='backbone',model_config_sha256='yaml',index_sha256='index')
            for a in ('parent','control','treatment') for s in (0,1)}


@pytest.mark.parametrize('field,value',[('seed',0),('shards',4),('precision','16-mixed'),('tto',True),
    ('backbone_sha256','different'),('model_config_sha256','different'),('index_sha256','different'),('checkpoint_sha256','parent'),('shard',1)])
def test_reject_cross_arm_identity_drift(field,value):
    rows=identities();rows['control',0][field]=value
    with pytest.raises(ValueError):shared_identity(rows,{a:a for a in ('parent','control','treatment')})


def test_complete_matched_identity():
    rows=identities();assert shared_identity(rows,{a:a for a in ('parent','control','treatment')})['seed']==42
    rows.pop(('control',0))
    with pytest.raises(ValueError):shared_identity(rows,{a:a for a in ('parent','control','treatment')})


def test_safety_regression_is_not_erased_by_roundoff_tolerance():
    old={'case':{k:1. for k in METRICS}};new=copy.deepcopy(old)
    new['case']['time_to_collision_within_bound']=1.-1e-12
    assert exact_changes(new,old)['time_to_collision_within_bound']['regressed']==1


def test_merged_row_swap_and_foreign_shard_rejected():
    values={k:1. for k in METRICS}
    rows=[dict(token='a',log_name='log',valid=True,**values)]
    verify_shard_rows(rows,{'a':'log'},{'a':values})
    swapped=dict(values,score=.9)
    with pytest.raises(ValueError,match='Merged table differs'):
        verify_shard_rows(rows,{'a':'log'},{'a':swapped})
    with pytest.raises(ValueError,match='membership'):
        verify_shard_rows(rows,{'b':'log'},{'a':values})
