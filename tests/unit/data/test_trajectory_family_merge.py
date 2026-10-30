import copy
import json
from pathlib import Path

import pytest

from driversi.rsi.experiments import trajectory_family_merge as mod


def candidate(name, offset, package=None, source='source'):
    return dict(candidate=name, assignment=dict(source_log=source, token='token'),
                dataset_reference=dict(package_id=package or name),
                descriptor=dict(candidate_id=name, comparison_key={'source_log': source},
                                xy=[[i*.1, offset] for i in range(71)]))


def test_cross_strategy_duplicates_not_counted_twice():
    rows = [candidate('native', 0), candidate('composite-duplicate', .49),
            candidate('composite-distinct', .6)]
    before = copy.deepcopy(rows)
    selected, excluded = mod.distinct_rows(rows, .5)
    assert [r['candidate'] for r in selected] == ['native', 'composite-distinct']
    assert excluded[0]['reason'] == 'cross_strategy_duplicate'
    assert rows == before


def test_same_package_cannot_be_counted_as_distinct():
    selected, excluded = mod.distinct_rows([candidate('a', 0, 'same'),
                                           candidate('b', 10, 'same')], .5)
    assert len(selected) == 1
    assert excluded[0]['reason'] == 'package_repeated'


def test_separate_sources_are_not_deduplicated_together():
    selected, excluded = mod.distinct_rows([candidate('a', 0, source='one'),
                                           candidate('b', 0, source='two')], .5)
    assert len(selected) == 2 and not excluded


def test_incompatible_source_state_is_rejected():
    a, b = candidate('a', 0), candidate('b', 2)
    b['descriptor']['comparison_key']['timestamp_us'] = 1
    with pytest.raises(ValueError, match='Different source states'):
        mod.distinct_rows([a, b], .5)


def test_future_only_differences_do_not_satisfy_full_window_gate():
    a, b = candidate('a', 0), candidate('b', 0)
    b['descriptor']['xy'] = [[i*.1, .6 if i >= 20 else 0] for i in range(71)]
    selected, excluded = mod.distinct_rows([a, b], .5)
    assert len(selected) == 1 and excluded[0]['reason'] == 'cross_strategy_duplicate'


def test_extra_qualified_sample_is_preserved_but_not_over_budget():
    rows = [candidate(f'c{i}', i) for i in range(11)]
    kept, excluded = mod.bounded_selection(rows, {('source', 'token'): 'anchor'}, 10, 10)
    assert len(kept) == 10
    assert excluded == [dict(candidate='c10', package_id='c10',
                             reason='qualified_but_not_selected_budget', trainable=True)]


def test_each_source_has_reserved_family_budget():
    with pytest.raises(ValueError, match='every declared source'):
        mod.bounded_selection([], {('a', 't'): 'a', ('b', 't'): 'b'}, 10, 10)


@pytest.mark.parametrize('trajectory,visual', [
    ('strict_world95_v1', 'experimental_numeric25_v1'),
    ('pdms_gt95_v1', 'experimental_numeric24_ade03_v1')])
def test_weaker_previous_admission_cannot_enter_current_campaign(tmp_path, trajectory, visual):
    branch, review, proof = (tmp_path/n for n in ['branch.json','visual.json','proof.json'])
    branch.write_text(json.dumps(dict(trajectory_evidence=dict(admission_profile=trajectory))))
    review.write_text(json.dumps(dict(visual_quality_profile=visual)))
    proof.write_text(json.dumps(dict(branch=dict(report=str(branch)),
                                    visual_admission=dict(path=str(review)))))
    with pytest.raises(ValueError, match='profile differs'):
        mod.require_profiles(dict(row=dict(admission=dict(path=str(proof)))),
                             dict(trajectory_admission_profile='pdms_gt95_v1',
                                  visual_quality_profile='experimental_numeric25_v1'))


def config(tmp_path, fraction=.2, devices=(6,)):
    model = tmp_path/'model.json'
    model.write_text(json.dumps(dict(settings=dict(gpu_memory_fraction=fraction,
                                                  distributed_training=dict(devices=list(devices))))))
    path = tmp_path/'config.json'
    path.write_text(json.dumps(dict(training_model_config=str(model))))
    return path


def test_explicit_shared_gpu_has_fixed_memory_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *a, **kw: '6, 3000, 97000\n')
    result = mod.training_preflight(config(tmp_path), 6)
    assert result['pytorch_memory_fraction'] == .2


def test_sharing_does_not_authorize_other_occupied_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *a, **kw: '6, 3000, 97000\n7, 3000, 97000\n')
    with pytest.raises(ValueError, match='GPU 7 occupied'):
        mod.training_preflight(config(tmp_path, devices=(6, 7)), 6)


def test_evaluation_incompatible_fraction_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='compatible'):
        mod.training_preflight(config(tmp_path, fraction=.85), 6)


def test_no_training_if_shared_gpu_lacks_headroom(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *a, **kw: '6, 80000, 97000\n')
    with pytest.raises(ValueError, match='headroom'):
        mod.training_preflight(config(tmp_path), 6)


def test_one_shot_wait_times_out_without_outputs(tmp_path):
    with pytest.raises(TimeoutError, match='no training launched'):
        mod.wait_for_families([tmp_path], 0)


def test_running_producer_not_mistaken_for_completed(tmp_path):
    (tmp_path/'trajectory-family-summary.json').write_text('{"status":"running"}')
    with pytest.raises(TimeoutError):
        mod.wait_for_families([tmp_path], 0)
