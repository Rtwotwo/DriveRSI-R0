import pytest

from driversi.rsi.experiments.cluster_retrieval_overlap import (
    per_query_overlap,
    query_dict,
    set_overlap,
)


def test_overlap_definitions():
    r = set_overlap(['a', 'b', 'b'], ['b', 'c'])
    assert r['shared'] == r['cluster_only'] == r['comparison_only'] == 1
    assert r['jaccard'] == 1 / 3
    assert r['fraction_of_comparison'] == .5


def test_global_overlap_is_not_same_query_agreement():
    a, b = {'q1': ['a'], 'q2': ['b']}, {'q1': ['b'], 'q2': ['a']}
    assert set_overlap(['a', 'b'], ['a', 'b'])['shared'] == 2
    assert per_query_overlap(a, b)['queries_with_shared_candidate'] == 0
    with pytest.raises(ValueError):
        per_query_overlap(a, {'other': ['a']})


def test_equal_budget_limit_and_duplicate_rejection():
    a, b = {'q': ['a', 'b']}, {'q': ['b', 'a']}
    assert per_query_overlap(a, b)['mean_shared_candidates'] == 2
    assert per_query_overlap(a, b, limit=1)['mean_shared_candidates'] == 0
    with pytest.raises(ValueError):
        query_dict([{'q': 'q1', 'c': ['a', 'a']}], 'q', 'c')
