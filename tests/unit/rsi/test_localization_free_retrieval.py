import copy

import pytest

from driversi.rsi.experiments.localization_free_retrieval import aggregate, select_ranked
from driversi.rsi.decisions.augmentation_budget import build_budget, require_localization_free, wave_gate
from driversi.rsi.experiments.reactive_cohort import family_stop_reason


SOURCES = {t: dict(source_log='log'+t, source_group='drive'+t, history_continuous=True)
           for t in ['a', 'b', 'c']}


def test_location_and_future_fields_cannot_change_selection_or_queue():
    clean = [dict(token='a', similarity=.8), dict(token='b', similarity=.9)]
    contaminated = copy.deepcopy(clean)
    contaminated[0].update(distance_m=0, heading_difference_deg=0, same_map=True,
                           future_target=[0, 0], pose=[1, 2])
    contaminated[1].update(distance_m=1e9, heading_difference_deg=180, same_map=False,
                           future_target=[999, 999], pose=[float('nan'), 2])
    a, b = select_ranked(clean, SOURCES, 2), select_ranked(contaminated, SOURCES, 2)
    assert a == b
    assert aggregate({'private-query': a}, SOURCES) == aggregate({'private-query': b}, SOURCES)
    assert 'private-query' not in str(aggregate({'private-query': a}, SOURCES))
    assert all(set(x) == {'token', 'similarity', 'original_rank'} for x in a)


def test_history_and_source_diversity_are_independent_of_location():
    sources = copy.deepcopy(SOURCES)
    sources['b']['source_group'] = sources['a']['source_group']
    sources['c']['history_continuous'] = None
    rows = [dict(token=t, similarity=s) for t, s in [('b', .9), ('a', .8), ('c', .7)]]
    assert [r['token'] for r in select_ranked(rows, sources, 3)] == ['b']


def test_unknown_token_and_nonfinite_score_rejected():
    with pytest.raises(ValueError):
        select_ranked([dict(token='test-token', similarity=.9)], SOURCES, 3)
    with pytest.raises(ValueError):
        select_ranked([dict(token='a', similarity=float('nan'))], SOURCES, 3)


def test_budget_distinguishes_ten_targets_from_attempts_and_assets():
    rows = [dict(token=t, source_log='log'+t) for t in SOURCES]
    budget = build_budget(rows, {'a', 'b'}, {'b'})
    assert budget['target_qualified_samples'] == 30
    assert budget['global_attempt_cap'] == 96
    assert budget['state_counts'] == {'asset_registered_unadmitted': 1,
                                      'needs_asset_repair': 1, 'needs_asset_build': 1}
    assert wave_gate(budget)['selected_tokens'] == ['a']
    assert not wave_gate(budget, unresolved=True)['allowed']
    assert not wave_gate(budget, used_attempts=95)['allowed']
    assert not wave_gate(budget, used_renders=59)['allowed']
    budget = build_budget(rows, {'a', 'b'}, {'a', 'b'})
    assert wave_gate(budget)['reason'] == 'blocked_assets_or_repair'


def test_geographic_production_evidence_is_rejected():
    ok = dict(localization_used_for_selection=False, contains_test_tokens=False)
    require_localization_free(ok, ok, ok)
    for invalid in [{}, dict(ok, geography_used_for_selection=True), dict(ok, localization_used_for_selection=True)]:
        with pytest.raises(ValueError):
            require_localization_free(invalid, ok, ok)


def test_existing_executor_stops_redundant_sampling_without_lowering_admission():
    duplicates = [dict(status='trajectory_duplicate_before_render')] * 3
    policy = dict(stop_after_consecutive_duplicates=3, stop_after_visual_failures=1,
                  stop_after_execution_failures=2)
    assert family_stop_reason(duplicates, policy) == 'consecutive_non_novel_native_rollouts'
    assert family_stop_reason([dict(status='visual_preflight_rejected')], policy) == 'visual_repair_required'
    assert family_stop_reason([dict(status='teacher_rejected')], policy) is None
