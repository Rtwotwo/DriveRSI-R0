"""Synthetic contract fixtures, never evidence of model improvement."""
import copy
import json

import pytest

from driversi.rsi.decisions.continual_learning import SAFETY, decide, scores
from driversi.rsi.experiments import continual_curriculum as cc
from driversi.visualization.current import current_config


def report(values):
    return dict(model='chainflow_stage1', complete=True, score_unit='0_to_1',
                native_score_unmodified=True, rows=[dict(token=t, metrics=dict(
                    PDMS=s, **{k: 1.0 for k in SAFETY})) for t, s in values.items()])


def verdict(candidate=None, retained=()):
    p = report(dict(target=0, old=1, other=1))
    c = candidate or report(dict(target=0.96, old=1, other=1))
    return decide(p, c, p, expected=['target', 'old', 'other'], target='target', retained=retained,
                  native_paired_benefit=True)


def test_full_release_observed_gain_not_generalization():
    d = verdict(retained=['old'])
    assert d['promoted'] and all(d['checks'].values())
    assert d['adaptive_evaluation'] and not d['blind_generalization_established']


@pytest.mark.parametrize('kind', ['global', 'safety', 'retention', 'mastery', 'paired'])
def test_local_gain_does_not_override_release_constraints(kind):
    c = report(dict(target=0.96, old=1, other=1))
    if kind == 'global':
        c['rows'][2]['metrics']['PDMS'] = 0
    elif kind == 'safety':
        c['rows'][2]['metrics'][SAFETY[0]] = 0
    elif kind == 'retention':
        c['rows'][1]['metrics']['PDMS'] = 0.94
    elif kind == 'mastery':
        c['rows'][0]['metrics']['PDMS'] = 0.9
    if kind == 'paired':
        p = report(dict(target=0, old=1, other=1))
        d = decide(p, c, c, expected=['target', 'old', 'other'], target='target', retained=['old'],
                   native_paired_benefit=True)
    else:
        d = verdict(c, ['old'])
    assert not d['promoted'] and not d['checks'][dict(global_='global_pdms_non_decreasing').get(
        kind + '_', {'safety':'safety_non_decreasing', 'retention':'retained_cases_preserved',
                    'mastery':'target_mastered', 'paired':'paired_benefit'}.get(kind))]


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'nan', 'wrong_model', 'partial'])
def test_bad_evaluation_cannot_release(kind):
    r = report(dict(target=0.96, old=1, other=1))
    if kind == 'missing': r['rows'].pop()
    elif kind == 'duplicate': r['rows'].append(copy.deepcopy(r['rows'][0]))
    elif kind == 'nan': r['rows'][0]['metrics']['PDMS'] = float('nan')
    elif kind == 'wrong_model': r['model'] = 'augustus'
    else: r['complete'] = False
    with pytest.raises(ValueError): verdict(r)


def test_comfort_and_native_paired_rejection_cannot_be_overridden():
    c=report(dict(target=.96,old=1,other=1))
    c['rows'][1]['metrics']['comfort']=0
    assert not verdict(c)['promoted']
    c['rows'][1]['metrics']['comfort']=1
    p=report(dict(target=0,old=1,other=1))
    for benefit in (False,None):
        d=decide(p,c,p,expected=['target','old','other'],target='target',retained=[],
                 native_paired_benefit=benefit)
        assert not d['promoted'] and not d['checks']['native_paired_benefit']


def state():
    return dict(model='chainflow_stage1', incumbent=dict(revision=0, pdms=94.04,
                checkpoint='/private/parent'), active='problem-001',
                policy=dict(max_candidates=3, targets=[10,20], max_rounds=2,
                            mastery_pdms=.95, attempts_per_source=32, replay_limit=32),
                cases=[dict(id='problem-001', query='secret-test-token', status='data_ready',
                round=0, target=10, candidates=[dict(token='train-token',source_log='train-log')],
                failure_classes=[SAFETY[0]])], retained=[], replay=[], rounds=[])


def test_promotion_keeps_only_accepted_replay_and_updates_parent():
    s = state(); c = s['cases'][0]
    child = dict(checkpoint='/private/child',revision=1,pdms=95)
    cc.apply_verdict(s, c, verdict(), child, ['sample1'])
    assert s['incumbent'] == child and s['active'] is None
    assert s['retained'] == ['secret-test-token'] and s['replay'] == ['sample1']
    assert c['status'] == 'solved'


def test_rejected_candidate_never_replaces_parent_and_retry_is_bounded():
    s = state(); c = s['cases'][0]; parent = copy.deepcopy(s['incumbent'])
    d = dict(promoted=False, status='rejected')
    cc.apply_verdict(s,c,d,{},['bad-sample'])
    assert c['status'] == 'needs_revision' and s['incumbent'] == parent
    cc.apply_verdict(s,c,d,{},['bad-sample'])
    assert c['status'] == 'budget_exhausted' and s['active'] is None
    assert s['replay'] == [] and s['incumbent'] == parent


def test_native_metric_alias_normalization_allows_parent_parity():
    p = report(dict(target=0.9))
    metrics = dict(p['rows'][0]['metrics']); metrics['score'] = metrics.pop('PDMS')
    native = dict(model='chainflow_stage1',status='completed',phase='evaluate',
        evaluation_context=dict(value=dict(protocol='chainflow_native_navsim_pdm_v1')),
        paired_samples=[dict(sample_id='target',source_log='test-log',parent=metrics,candidate=metrics)])
    result = cc.evaluation_from_native(native,['target'],'parent')
    assert scores(result,['target']) == scores(p,['target'])


def test_current_projection_hides_test_details_and_old_experiments():
    s = state(); payload = cc.project(s)
    rendered = json.dumps(payload)
    assert all(word not in rendered for word in ('secret-test-token','train-token','/private/'))
    cfg = dict(display_mode='current_only', experiments=[dict(old='old-run')],
        training=['old-training'],clustering=dict(root='pool',failure_mapping='old-report',overlap_summary='old-overlap'))
    clean = current_config(cfg)
    assert clean['experiments'] == clean['training'] == []
    assert clean['clustering'] == dict(root='pool') and cfg['experiments']


def test_dispatch_reuses_native_receipt_without_starting_job(tmp_path,monkeypatch):
    s = state(); c = s['cases'][0]; c.update(status='blocked_assets',current_round=str(tmp_path/'round'))
    out = tmp_path/'round/waves/wave-0001'; out.mkdir(parents=True)
    (out/'production-status.json').write_text('{"status":"production_budget_blocked"}')
    (tmp_path/'private').mkdir(); (tmp_path/'private/state.json').write_text(json.dumps(s))
    from driversi.rsi.experiments import reactive_autoloop
    monkeypatch.setattr(reactive_autoloop,'run',lambda *a:pytest.fail('Repeated native job'))
    assert cc.dispatch(tmp_path)['active']['status'] == 'blocked_assets'


def test_training_cannot_start_from_asset_block(tmp_path):
    s = state();s['cases'][0]['status']='blocked_assets'
    (tmp_path/'private').mkdir();(tmp_path/'private/state.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='admitted generation'):
        cc.train_current(tmp_path)


def test_shared_admitted_replay_is_exposed_in_both_native_arms(tmp_path, monkeypatch):
    from pathlib import Path
    from driversi.rsi.plugins import chainflow_synthetic as compiler
    from driversi_agents.chainflow import benchmark, data, synthetic
    base = dict(schema_version=2, benchmark=dict(protocol='official_navtest'), files={},
        splits=dict(train=[dict(sample_id=f'original-{i}',kind='original_navtrain',source_log='train') for i in range(16)],
                    development=[dict(sample_id='test',kind='original_navtrain',source_log='test',
                                      metric_cache=str(tmp_path/'cache.pkl'))]))
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return p
    base_path=write('base.json',base)
    model=write('model.json',dict(model='chainflow_stage1',settings=dict(
        distributed_training=dict(devices=[6,7],checkpoint_interval=2,synthetic_every=8),
        training_recipe=dict(scheduler_dataset_size=256,scheduler_num_epochs=1))))
    receipt=lambda token:dict(kind='chainflow_admitted_synthetic_sample', files={},
        row=dict(sample_id=token,source_log='train-'+token,kind='synthetic_navtrain'))
    new,old=write('new.json',receipt('new')),write('old.json',receipt('old'))
    monkeypatch.setattr(compiler,'project_root',lambda:tmp_path)
    monkeypatch.setattr(data,'validate_manifest',lambda p:json.loads(Path(p).read_text()))
    monkeypatch.setattr(benchmark,'evaluator_files',lambda _: {})
    checked=[]
    monkeypatch.setattr(synthetic,'validate_admission',lambda row,files:checked.append(row['sample_id']))
    task=compiler.compile_paired_task(base_path,[new],model,tmp_path/'exp/pair',
        epoch_training=True,batch_size=2,updates_per_arm=8,replay_reports=[old])
    out=task.parent
    manifests={a:json.loads((out/a/'manifest.json').read_text()) for a in ('control','treatment')}
    assert [r['sample_id'] for r in manifests['control']['splits']['train']]==[f'original-{i}' for i in range(16)]+['old']
    assert [r['sample_id'] for r in manifests['treatment']['splits']['train']]==[f'original-{i}' for i in range(16)]+['old','new']
    receipt=json.loads((out/'intervention.json').read_text())
    assert receipt['added_sample_ids']==['new'] and receipt['shared_replay_sample_ids']==['old']
    for arm in ('control','treatment'):
        assert receipt['expected_sample_exposures'][arm]['old']>0
    assert receipt['expected_sample_exposures']['treatment']['new']>0
    assert checked==['new','old']
    with pytest.raises(ValueError,match='duplicates'):
        compiler.compile_paired_task(base_path,[new],model,tmp_path/'exp/duplicate',
            epoch_training=True,batch_size=2,updates_per_arm=8,replay_reports=[new])
    def rejected(row,files):
        if row['sample_id']=='old':raise ValueError('Replay admission invalid')
    monkeypatch.setattr(synthetic,'validate_admission',rejected)
    with pytest.raises(ValueError,match='Replay admission invalid'):
        compiler.compile_paired_task(base_path,[new],model,tmp_path/'exp/rejected',
            epoch_training=True,batch_size=2,updates_per_arm=8,replay_reports=[old])
    assert not (tmp_path/'exp/rejected').exists()


def test_settle_full_native_contract_and_idempotent_promotion(tmp_path):
    """All metrics here are test fixtures; no inference or training is run."""
    from pathlib import Path
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    s=state();case=s['cases'][0];case.update(query='t0',status='awaiting_release')
    folder=tmp_path/'rounds/problem-001-r1';case['current_round']=str(folder)
    parent=tmp_path/'parent.ckpt';parent.write_bytes(b'fixture-parent')
    tokens=[f't{i}' for i in range(12146)]
    baseline=report({t:0 if t=='t0' else 1 for t in tokens})
    evalfile=write(tmp_path/'baseline.json',baseline)
    s['incumbent'].update(checkpoint=str(parent),sha256=cc.digest(parent),evaluation=str(evalfile))
    pair=folder/'waves/wave-0001/chainflow-pair';task=pair/'tasks/trial'
    write(pair/'training-status.json',dict(status='completed',directory=str(task)))
    write(folder/'training-input.json',dict(parent=s['incumbent'],samples=['admitted-fixture'],replay=[]))
    write(task/'paired_feedback/report.json',dict(kind='chainflow_paired_feedback',status='completed',
        effective_recipe_verified=True,added_sample_ids=['synthetic'],exposed_added_sample_ids=['synthetic'],
        unconsumed_added_sample_ids=[],observed_pilot_benefit=True,regressions_against_control=[],unique_training_cost={}))
    write(task/'treatment_feedback/feedback.json',dict(cross_process_resume=dict(different_pids=True,same_compared_state=True)))
    for arm in ('control','treatment'):
        checkpoint=task/(arm+'_resume/model.ckpt');checkpoint.parent.mkdir(parents=True,exist_ok=True)
        checkpoint.write_bytes(('fixture-'+arm).encode())
        write(task/(arm+'_train/training-plan.json'),dict(model_config=dict(weights=dict(checkpoint=str(parent)))))
        write(task/(arm+'_resume/training-result.json'),dict(checkpoint=str(checkpoint),
            checkpoint_files={str(checkpoint):cc.digest(checkpoint)},native_loss_unchanged=True))
        rows=[]
        for row in baseline['rows']:
            metrics=dict(row['metrics']);metrics['score']=metrics.pop('PDMS')
            candidate=dict(metrics)
            if arm=='treatment' and row['token']=='t0':candidate['score']=.96
            rows.append(dict(sample_id=row['token'],source_log='test-log',parent=metrics,candidate=candidate))
        write(task/(arm+'_evaluation/training-result.json'),dict(model='chainflow_stage1',status='completed',
            phase='evaluate',evaluated_checkpoint=str(checkpoint),paired_samples=rows,
            evaluation_context=dict(value=dict(protocol='chainflow_native_navsim_pdm_v1'))))
    (tmp_path/"private").mkdir()
    cc.save(tmp_path,s)
    result=cc.settle(tmp_path)
    assert result['revision']==1 and result['active'] is None
    actual=json.loads((tmp_path/'private/state.json').read_text())
    assert actual['incumbent']['checkpoint'].endswith('treatment_resume/model.ckpt')
    assert actual['replay']==['admitted-fixture'] and actual['retained']==['t0']
    from driversi.rsi.decisions.service_memory import ServiceMemory
    mem=ServiceMemory(tmp_path/'service-memory.sqlite','frozen')
    try: assert len(mem.rows())==1 and mem.rows()[0]['training_executed']
    finally: mem.close()
    before=(tmp_path/'private/state.json').read_bytes()
    with pytest.raises((ValueError,StopIteration)):cc.settle(tmp_path)
    assert (tmp_path/'private/state.json').read_bytes()==before
