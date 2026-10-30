import numpy as np
import pytest

from driversi.rsi.experiments.cluster_failure_mapping import (
    focus_counts,
    pool_choices,
    ranked_candidates,
    select_scores,
)


def score_fixture():
    values = [0, .1, .59999, .6, 1]
    test = [dict(token=str(i), source_log='log' + str(i)) for i in range(len(values))]
    rows = [dict(token=r['token'], log_name=r['source_log'], valid='True', error='', score=v)
            for r, v in zip(test, values)]
    summary = dict(complete=True, evaluation_split='navtest', metric='NAVSIM_v1_PDMS',
                   failed=0, duplicate_rows=0, expected=5, succeeded=5,
                   identity={'checkpoint_sha256': 'parent'}, pdms=np.mean(values) * 100)
    return rows, summary, test


def test_strict_threshold_and_disjoint_groups():
    rows, summary, test = score_fixture()
    q = select_scores(rows, summary, test, 'parent')
    assert [r['token'] for r in q] == ['0', '1', '2']
    assert [r['group'] for r in q] == ['zero', 'low', 'low']


@pytest.mark.parametrize('field,value', [('score', float('nan')), ('score', -1),
    ('score', 60), ('valid', 'False'), ('log_name', 'wrong'), ('error', 'failed')])
def test_invalid_score_rejected(field, value):
    rows, summary, test = score_fixture()
    rows[1][field] = value
    with pytest.raises(ValueError):
        select_scores(rows, summary, test, 'parent')


def test_duplicate_missing_checkpoint_and_summary_rejected():
    rows, summary, test = score_fixture()
    for bad in [rows[:-1], rows[:-1] + [rows[0]]]:
        with pytest.raises(ValueError):
            select_scores(bad, summary, test, 'parent')
    with pytest.raises(ValueError):
        select_scores(rows, summary, test, 'another_parent')
    with pytest.raises(ValueError):
        select_scores(rows, {**summary, 'pdms': 94.05}, test, 'parent')


def test_expansion_keeps_primary_without_duplicate_leaf():
    assert pool_choices([2, 5], [[2, 3, 4], [1, 2, 3]]) == [[2, 3, 4], [5, 1, 2, 3]]


def test_focus_deduplicates_ancestors_per_query():
    nodes = [{'parent': -1}, {'parent': 0}, {'parent': 0}]
    c = focus_counts([{'group': 'zero'}, {'group': 'low'}], [1, 2], [[1, 2], [2]], nodes)
    assert c[0]['expanded_total'] == c[0]['primary_total'] == 2
    assert c[1]['primary_zero'] == 1
    assert c[2]['primary_low'] == 1
    assert c[2]['expanded_total'] == 2


def test_ranking_source_cap_distance_and_determinism():
    x = np.asarray([[0., 0], [.1, 0], [.2, 0], [.3, 0]])
    rows = [dict(source_log=log) for log in ['a', 'a', 'b', 'c']]
    ranked = ranked_candidates(x[0], [3, 2, 1, 0], x, rows, limit=3, per_source=1)
    assert [i for i, _ in ranked] == [0, 2, 3]
    assert [d for _, d in ranked] == [0, .2, .3]
