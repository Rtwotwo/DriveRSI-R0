import json
from pathlib import Path
import pytest
from driversi.rsi.experiments.augustus_pair import _recipe, _arm, record

def test_recipe_preserves_loss_and_schedule_but_ignores_output_location(tmp_path):
    a=tmp_path/'a.yaml';b=tmp_path/'b.yaml'
    a.write_text('training: {output_dir: a, learning_rate: 0.00001}\ndata: {train_data_path: a, val_data_path: fixed}\nloss: {fm: 10}\n')
    b.write_text(a.read_text().replace('output_dir: a','output_dir: b').replace('train_data_path: a','train_data_path: b'))
    assert _recipe(a)==_recipe(b)
    b.write_text(b.read_text().replace('fm: 10','fm: 1'))
    assert _recipe(a)!=_recipe(b)

def test_other_checkpoint_or_train_source_evaluation_cannot_claim_gain(tmp_path):
    for phase in ['train','resume','evaluate']:(tmp_path/phase).mkdir()
    (tmp_path/'train/training-plan.json').write_text('{}')
    a=dict(model='augustus',status='completed',context='x',checkpoint='real')
    b=dict(model='augustus',status='completed',context='x',evaluated_checkpoint='other',metric_scope='native_navsim_pdm_on_locked_official_navtest_pilot')
    (tmp_path/'resume/training-result.json').write_text(json.dumps(a))
    (tmp_path/'evaluate/training-result.json').write_text(json.dumps(b))
    with pytest.raises(ValueError,match='checkpoint'):_arm(tmp_path)

def test_unbound_measured_learning_remains_uncredited(tmp_path):
    report=dict(parent_assets={'parent':'sha'},evaluation_scope='locked',gain_vs_control_pdms_points=.2,safety_regressions=[],training_evaluation_costs={})
    result=record(report,tmp_path/'report.json',tmp_path/'memory.sqlite')
    assert result['recorded'] and not result['credited']
    with pytest.raises(ValueError,match='Preserve'):record(report,tmp_path/'report.json',tmp_path/'memory.sqlite')


def test_zero_lr_forward_is_not_positive_update_exposure():
    from driversi.rsi.experiments.augustus_pair import _positive_exposure
    events = [dict(sample_ids=['synthetic'], learning_rates=[0., 0.]),
              dict(sample_ids=['real'], learning_rates=[1e-7, 1e-7])]
    assert _positive_exposure(events) == {'real'}
    events.append(dict(sample_ids=['synthetic'], learning_rates=[2e-7, 2e-7]))
    assert _positive_exposure(events) == {'real', 'synthetic'}


@pytest.mark.parametrize('rates', [[], [float('nan')], [-1.], [True]])
def test_invalid_optimizer_rates_are_not_evidence(rates):
    from driversi.rsi.experiments.augustus_pair import _positive_exposure
    with pytest.raises(ValueError, match='learning rates'):
        _positive_exposure([dict(sample_ids=['synthetic'], learning_rates=rates)])


def test_paired_metrics_require_unique_complete_finite_population():
    import copy
    from driversi.rsi.experiments.augustus_pair import _paired_population
    metrics={k:1. for k in ['PDMS','NC','DAC','DDC','EP','TTC','C']}
    a=dict(sample_id='a',source_log='log-a',parent=metrics.copy(),candidate=metrics.copy())
    b=dict(sample_id='b',source_log='log-b',parent=metrics.copy(),candidate=metrics.copy())
    subset={'records':[dict(token='a',source_log='log-a'),dict(token='b',source_log='log-b')]}
    value=dict(paired_samples=[a,b],metrics=dict(parent=metrics.copy(),candidate=metrics.copy()))
    assert set(_paired_population(value,subset))=={'a','b'}
    for bad_rows in [[a,a],[a],[dict(a,source_log='wrong'),b]]:
        with pytest.raises(ValueError,match='population'):
            _paired_population(dict(value,paired_samples=bad_rows),subset)
    bad=copy.deepcopy(value);bad['metrics']['candidate']['PDMS']=.9
    with pytest.raises(ValueError,match='Aggregate'):_paired_population(bad,subset)
    bad=copy.deepcopy(value);bad['paired_samples'][0]['parent']['PDMS']=float('nan')
    with pytest.raises(ValueError,match='Finite'):_paired_population(bad,subset)


def test_fixed_pool_requires_historical_byte_equality():
    from driversi.rsi.experiments.augustus_pair import _fixed_cache_evidence
    a={'files':{'cache':'old'}};b={'files':{'cache':'new'}}
    with pytest.raises(ValueError,match='cache bytes'):
        _fixed_cache_evidence([a,b],{'cache':'new'})
    _fixed_cache_evidence([a,a],{'cache':'old'})


def test_joint_pool_uses_sealed_manifest_closure_without_rewriting_plan(tmp_path):
    from driversi.rsi.experiments.augustus_pair import _fixed_original_pool
    from driversi.agents.backends.native import digest
    manifest=tmp_path/'joint.json';manifest.write_text('{}')
    control={'files':{'original.arrow':'a'}}
    treatment={'data':{'manifest':str(manifest)},'files':{str(manifest):digest(manifest),'mixed.arrow':'m'}}
    joint={'base_cache_files':{'original.arrow':'a'},'train_cache_files':{'mixed.arrow':'m'}}
    _fixed_original_pool(control,treatment,joint)
    changed={'base_cache_files':{'original.arrow':'new'},'train_cache_files':{'mixed.arrow':'m'}}
    with pytest.raises(ValueError,match='cache bytes'):_fixed_original_pool(control,treatment,changed)
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='manifest changed'):_fixed_original_pool(control,treatment,joint)
