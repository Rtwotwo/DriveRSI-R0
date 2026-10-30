from driversi.rsi.experiments.compact_cluster_retrieval import choose, compact_cover, gate


def pair(token='train', distance=5, heading=4, same=True, score=.9):
    return dict(token=token, same_map=same, distance_m=distance,
                heading_difference_deg=heading, similarity=score)


def test_geography_keeps_maps_and_direction_separate():
    assert gate(pair())
    assert not gate(pair(distance=26))
    assert not gate(pair(heading=31))
    assert not gate(pair(same=False, distance=None, heading=None))


def test_unknown_history_and_outlier_do_not_gain_training_eligibility():
    assert not choose([pair()], radius=25, continuous={'train': None})
    assert not choose([pair()], radius=25, continuous={'train': True}, uncertain=True)
    assert choose([pair()], radius=25, continuous={'train': True}) == [pair()]


def test_asset_filter_cannot_rescue_unrelated_candidate():
    rows = [pair('a', distance=50), pair('b')]
    assert not choose(rows, radius=25, continuous={'a': True, 'b': True}, assets={'a'})


def test_compact_cover_reuses_seed_without_duplicate_source_logs():
    sources = {t: dict(source_log=log, source_group=log) for t, log in
               [('a', 'L1'), ('b', 'L1'), ('c', 'L2')]}
    rows = {'q1': [pair('a'), pair('c')], 'q2': [pair('a')], 'q3': [pair('b')]}
    selected, covered = compact_cover(rows, sources, 24)
    assert [r['token'] for r in selected] == ['a']
    assert covered == 2
    assert selected[0]['trainable'] is False
    assert 'q1' not in str(selected)


def test_empty_queries_and_budget():
    assert compact_cover({'q1': []}, {}, 24) == ([], 0)
    assert compact_cover({'q1': [pair()]}, {'train': dict(source_log='l', source_group='s')}, 0) == ([], 0)
