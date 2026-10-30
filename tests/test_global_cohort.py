import pytest
from driversi.rsi.decisions.global_cohort import select


def row(source, n):
    return dict(source_token=source,trainable=True,status='training_sample_ready',rendered=True,
        sample_report=f'report-{n}',dataset_reference=dict(package_id=f'pkg-{n}'))


def test_uneven_fifty_and_coverage():
    rows=[row('a',i) for i in range(49)]+[row('b',49)]
    result=select(['a','b'],rows)
    assert result['passed'] and len(result['selected_rows'])==50
    assert {x['source_token'] for x in result['selected_rows'][:2]}=={'a','b'}
    assert result['available_per_source']=={'a':49,'b':1}


def test_no_train_if_a_participating_source_absent():
    result=select(['a','b'],[row('a',i) for i in range(60)])
    assert not result['passed'] and result['missing_sources']==['b']


def test_deduplicate_and_detect_ambiguous_source():
    rows=[row('a',i) for i in range(49)]+[row('b',49)]
    assert select(['a','b'],rows+[rows[0]])['available_unique']==50
    altered=dict(rows[0],source_token='b')
    with pytest.raises(ValueError,match='ambiguous'):select(['a','b'],rows+[altered])


def test_rejects_unadmitted_and_missing_package():
    with pytest.raises(ValueError,match='admitted'):select(['a'],[dict(row('a',0),trainable=False)])
    with pytest.raises(ValueError,match='identity'):select(['a'],[dict(row('a',0),dataset_reference={})])
    with pytest.raises(ValueError,match='Unexpected'):select(['a'],[row('b',1)])


def test_insufficient_total_and_invalid_source_contract():
    result=select(['a','b'],[row('a',0),row('b',1)])
    assert not result['passed'] and result['shortage']==48
    with pytest.raises(ValueError):select(['a','a'],[])
    with pytest.raises(ValueError):select([str(i) for i in range(51)],[])


def test_reused_report_is_not_two_examples():
    rows=[row('a',i) for i in range(49)]+[row('b',49)]
    rows[-1]['sample_report']=rows[0]['sample_report']
    with pytest.raises(ValueError,match='report'):select(['a','b'],rows)


def test_family_gate_enforces_global_total_and_coverage_without_changing_legacy():
    from driversi.rsi.experiments.reactive_cohort import family_training_gate
    rows=[dict(row('a',i),anchor_assignment_id='a') for i in range(49)]+[
        dict(row('b',49),anchor_assignment_id='b')]
    summary=dict(rows=rows,qualified=50)
    assert family_training_gate(summary,['a','b'],50,global_target=50)['passed']
    assert not family_training_gate(summary,['a','b'],10)['passed']
    missing=dict(rows=rows[:49],qualified=49)
    result=family_training_gate(missing,['a','b'],50,global_target=50)
    assert not result['passed'] and result['global_shortage']==1 and result['shortages']['b']==1
    rows[-1]['dataset_reference']=rows[0]['dataset_reference']
    with pytest.raises(ValueError,match='multiple sources'):
        family_training_gate(summary,['a','b'],50,global_target=50)


def test_global_budget_is_finite_and_does_not_weaken_legacy():
    from driversi.rsi.decisions.augmentation_budget import build_budget
    rows=[dict(token=str(i),source_log=str(i)) for i in range(3)]
    budget=build_budget(rows,[],[],target=50,attempts=32,wave_size=3,global_target=50)
    assert budget['target_qualified_samples']==50 and budget['global_attempt_cap']==96
    with pytest.raises(ValueError):build_budget(rows,[],[],target=50,attempts=32,wave_size=3)


def test_global_family_budget_requires_the_explicit_new_contract():
    from driversi.rsi.experiments.reactive_cohort import validate_budget
    c=dict(student_model='chainflow_stage1',training_population='synthetic_only',
        trajectory_family=dict(target_per_anchor=50,attempts_per_anchor=32,minimum_ade_m=.5,
            seed_start=42,cohort_mode='global_50_min1_v1',anchor_budget_mode='explore_until_global_limit'),
        max_samples=50,max_rendered_branches=100,max_candidates=3)
    assert validate_budget(c)['target_per_anchor']==50
    bad=dict(c,training_population='mixed')
    with pytest.raises(ValueError,match='explicit'):validate_budget(bad)
    bad=dict(c,max_samples=49)
    with pytest.raises(ValueError,match='explicit'):validate_budget(bad)


def test_prior_seed_reserves_capacity_for_missing_sources(tmp_path):
    import json,hashlib
    from driversi.rsi.experiments.problem_cohort import seed
    outcome=tmp_path/'outcome.json';outcome.write_text('{}')
    sha=hashlib.sha256(outcome.read_bytes()).hexdigest()
    rows=[dict(source_token='a',original_outcome=str(outcome),original_outcome_sha256=sha,id=i) for i in range(49)]
    rows += [dict(source_token='b',original_outcome=str(outcome),original_outcome_sha256=sha,id=50)]
    pool=tmp_path/'pool.json';pool.write_text(json.dumps(dict(rows=rows)))
    cfg=dict(prior_cohort=dict(path=str(pool),sha256=hashlib.sha256(pool.read_bytes()).hexdigest()),
        trajectory_family=dict(target_per_anchor=50,cohort_mode='global_50_min1_v1'))
    plan=dict(assignments=[dict(token=t,anchor_assignment_id=t) for t in 'abc'])
    selected=seed(cfg,plan)
    assert len(selected)==49 and {r['source_token'] for r in selected}=={'a','b'}


def test_prefetched_surplus_cannot_take_the_last_uncovered_source_slot():
    from driversi.rsi.experiments.reactive_cohort import coverage_slot_reserved
    from collections import Counter
    counts=Counter(a=24,b=25,c=0)
    assert coverage_slot_reserved(True,counts,['a','b','c'],50,49,'b')
    assert not coverage_slot_reserved(True,counts,['a','b','c'],50,49,'c')
    assert not coverage_slot_reserved(False,counts,['a','b','c'],50,49,'b')
    assert not coverage_slot_reserved(True,counts,['a','b','c'],50,48,'b')
