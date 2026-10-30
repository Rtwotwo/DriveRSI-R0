import copy
import json
import math
import pickle

import numpy as np
import pytest
import yaml

from driversi.rsi.decisions.scene_index import (
    build_index,
    describe_records,
    extract_history_descriptors,
    retrieve,
)


def frame(token, source, tick, *, speed=2, yaw=0, names=("vehicle", "barrier")):
    return {
        "token": token,
        "log_name": source,
        "timestamp": tick * 500000,
        "ego_dynamic_state": [speed, 0, 0, 0],
        "ego2global_rotation": [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
        "driving_command": [0, 1, 0, 0],
        "anns": {
            "gt_boxes": np.asarray([[5, 0, 0, 1, 1, 1, 0] for _ in names]).reshape(-1, 7),
            "gt_names": np.asarray(names),
        },
    }


@pytest.fixture
def pool(tmp_path):
    source = "2021.01.01.00.00.00_veh-01_00010_00020"
    frames = [frame(str(i), source, i, speed=i, yaw=0.05 * i) for i in range(7)]
    log = tmp_path / (source + ".pkl")
    log.write_bytes(pickle.dumps(frames))
    split = tmp_path / "navtrain.yaml"
    split.write_text(yaml.safe_dump({"log_names": [source], "tokens": ["3", "4"]}))
    records = [
        {"token": "3", "source_log": source, "log_path": str(log), "anchor_index": 3},
        {"token": "4", "source_log": source, "log_path": str(log), "anchor_index": 4},
    ]
    return records, frames, split


def feedback(capability="collision"):
    return {
        "schema_version": 1,
        "benchmark": "navtest",
        "checkpoint": "student-a",
        "evaluator_version": "native-v1",
        "slices": [{"capability": capability, "count": 5, "failures": 3}],
    }


def index(pool, records=None, **kwargs):
    rows, _, split = pool
    return build_index(
        rows if records is None else records,
        official_split_path=split,
        development_groups=kwargs.pop("development_groups", []),
        protected_groups=kwargs.pop("protected_groups", []),
        **kwargs,
    )


def test_history_features_are_observable_and_future_invariant(pool):
    rows, frames, _ = pool
    first = index(pool, rows[:1])
    assert first["records"][0]["features"]["ego_speed_mps"] == 3
    assert {
        "nearby_vehicle",
        "nearby_static",
        "history_turning",
        "history_speed_change",
        "nav_command_1",
    } <= set(first["records"][0]["conditions"])
    # Changing all future frames cannot change an anchor descriptor or ranking.
    for f in frames[4:]:
        f.clear()
        f.update({"future_failure": True})
    with open(rows[0]["log_path"], "wb") as stream:
        pickle.dump(frames, stream)
    second = index(pool, rows[:1])
    assert first["records"] == second["records"]
    assert (
        retrieve(first, feedback(), 1)["candidates"]
        == retrieve(second, feedback(), 1)["candidates"]
    )


def test_missing_observations_are_unknown_not_sparse_or_stationary():
    result = extract_history_descriptors([{}, {}])
    assert not result["conditions"]
    assert result["features"]["ego_speed_mps"] is None
    assert "current_actor_annotations" in result["unknown"]
    assert "navigation_command" in result["unknown"]


def test_yaw_wrap_does_not_invent_turn():
    history = [frame("a", "log", 0, yaw=math.pi - 0.01), frame("b", "log", 1, yaw=-math.pi + 0.01)]
    result = extract_history_descriptors(history)
    assert result["features"]["history_yaw_range_rad"] == pytest.approx(0.02)
    assert "history_turning" not in result["conditions"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("token", "not-official"),
        ("source_log", "unlisted_00010_00020"),
        ("source_split", "navtest"),
        ("kind", "synthetic_navtrain"),
    ],
)
def test_declared_split_cannot_admit_nonofficial_sources(pool, field, value):
    rows = copy.deepcopy(pool[0][:1])
    rows[0]["source_split"] = "navtrain"
    rows[0][field] = value
    with pytest.raises(ValueError, match="official"):
        index(pool, rows)


@pytest.mark.parametrize("role", ["development_groups", "protected_groups"])
def test_all_shards_of_excluded_original_drive_are_rejected(pool, role):
    source = pool[0][0]["source_log"]
    sibling = source.rsplit("_", 2)[0] + "_90000_99999.pkl"
    with pytest.raises(ValueError, match="official"):
        index(pool, **{role: [sibling]})


def test_official_navtest_log_group_is_automatically_excluded(pool):
    source = pool[0][0]["source_log"]
    pool[2].with_name("navtest.yaml").write_text(
        yaml.safe_dump(
            {
                "log_names": [source.rsplit("_", 2)[0] + "_90000_99999"],
                "tokens": ["test"],
            }
        )
    )
    with pytest.raises(ValueError, match="official"):
        index(pool)


def test_official_tokens_without_log_allowlist_is_rejected(pool):
    pool[2].write_text(yaml.safe_dump({"tokens": ["3"], "log_names": []}))
    with pytest.raises(ValueError, match="allowlists"):
        index(pool)


def test_raw_identity_is_verified_not_only_declared_membership(pool):
    rows = copy.deepcopy(pool[0][:1])
    rows[0]["anchor_index"] = 4
    with pytest.raises(ValueError, match="identity"):
        index(pool, rows)


def test_history_log_and_time_mismatch_rejected(pool):
    rows, frames, _ = pool
    for field, value, match in [
        ("log_name", "other", "identity"),
        ("timestamp", -1, "Continuous"),
    ]:
        changed = copy.deepcopy(frames)
        changed[1][field] = value
        with open(rows[0]["log_path"], "wb") as stream:
            pickle.dump(changed, stream)
        with pytest.raises(ValueError, match=match):
            index(pool, rows[:1])


def test_duplicate_token_rejected_at_build_and_retrieval(pool):
    with pytest.raises(ValueError, match="Duplicate"):
        index(pool, pool[0][:1] * 2)
    value = index(pool)
    value["records"].append(value["records"][0])
    with pytest.raises(ValueError, match="duplicate"):
        retrieve(value, feedback(), 5)


def test_retrieval_rechecks_excluded_source_groups(pool):
    value = index(pool)
    value["policy"]["excluded_source_groups"] = [value["records"][0]["source_group"]]
    with pytest.raises(ValueError, match="official"):
        retrieve(value, feedback(), 5)


def test_retrieval_rechecks_official_split_hash(pool):
    value = index(pool)
    pool[2].write_text(pool[2].read_text() + "\n# policy changed\n")
    with pytest.raises(ValueError, match="changed"):
        retrieve(value, feedback(), 5)


def test_cohort_match_and_explicit_fallback_do_not_claim_failure(pool):
    value = index(pool)
    exact = retrieve(
        value,
        feedback(),
        1,
        cohorts=[
            {
                "capability": "collision",
                "conditions": ["nearby_vehicle"],
                "count": 5,
                "failures": 3,
            }
        ],
    )
    candidate = exact["candidates"][0]
    assert candidate["status"] == "proposed"
    assert candidate["requires_model_verification"]
    assert candidate["match"]["mode"] == "exact_observable_cohort"
    assert "failed" not in candidate
    fallback = retrieve(
        value,
        feedback(),
        1,
        cohorts=[
            {
                "capability": "collision",
                "conditions": ["traffic_dense"],
                "count": 5,
                "failures": 3,
            }
        ],
    )
    assert fallback["candidates"][0]["match"]["mode"] == "capability_heuristic_fallback"
    assert fallback["coverage"]["unmatched_cohorts"]


def test_unsupported_capability_requires_observed_cohort(pool):
    result = retrieve(index(pool), feedback("occlusion"), 2)
    assert not result["candidates"]
    assert result["coverage"]["uncovered_capabilities"] == ["occlusion"]
    assert result["coverage"]["unsupported_heuristic_capabilities"] == ["occlusion"]
    same_conditions = retrieve(
        index(pool),
        feedback("occlusion"),
        2,
        cohorts=[
            {
                "capability": "occlusion",
                "conditions": ["nearby_static"],
                "count": 5,
                "failures": 3,
            }
        ],
    )
    assert same_conditions["candidates"][0]["match"]["mode"] == "exact_observable_cohort"


def test_cohorts_cannot_smuggle_test_tokens_or_unknown_semantic_tags(pool):
    value = index(pool)
    for cohort in [
        {"capability": "collision", "conditions": ["test-token"], "count": 5, "failures": 3},
        {
            "capability": "collision",
            "conditions": ["nearby_static"],
            "count": 5,
            "failures": 3,
            "token": "navtest-token",
        },
    ]:
        with pytest.raises(ValueError, match="Sanitized"):
            retrieve(value, feedback(), 2, cohorts=[cohort])


def test_streamed_log_mode_is_bounded_and_same_as_record_mode(pool):
    rows, _, split = pool
    value = build_index(
        log_root=split.parent,
        official_split_path=split,
        development_groups=[],
        protected_groups=[],
        max_scenes=1,
    )
    assert value["records"] == index(pool, rows[:1])["records"]
    assert value["coverage"]["scope"] == "streamed_official_logs"
    assert value["coverage"]["indexed_scenes"] == 1


def test_describe_development_does_not_admit_it_to_training(pool):
    source = pool[0][0]["source_log"]
    described = describe_records(pool[0])
    assert len(described) == 2
    with pytest.raises(ValueError, match="official"):
        index(pool, development_groups=[source])


def test_output_is_json_serializable_and_deterministic(pool):
    a = index(pool)
    b = index(pool, reversed(pool[0]))
    assert a["records"] == b["records"]
    json.dumps(retrieve(a, feedback(), 2), allow_nan=False)


def test_population_keeps_time_gaps_but_synthesis_retrieval_excludes_them(pool):
    rows, frames, split = pool
    for f in frames[3:]:
        f['timestamp'] += 500000
    with open(rows[0]['log_path'], 'wb') as stream:
        pickle.dump(frames, stream)
    with pytest.raises(ValueError, match='Continuous'):
        describe_records(rows)
    described = describe_records(rows, allow_irregular_history=True)
    assert len(described) == 2
    assert all('history_sampling_gap' in row['unknown'] for row in described)
    assert all('history_turning' not in row['conditions'] for row in described)
    value = index(pool, allow_irregular_history=True)
    result = retrieve(value, feedback(), 2, require_continuous_history=True)
    assert result['coverage']['pool_scenes'] == 2
    assert result['coverage']['excluded_irregular_history'] == 2
    assert result['candidates'] == []


def test_confirmation_source_diversity_does_not_repeat_adjacent_anchors(pool):
    value = index(pool)
    assert len(retrieve(value, feedback(), 2)['candidates']) == 2
    selected = retrieve(value, feedback(), 2, per_source_limit=1)
    assert len(selected['candidates']) == 1
    assert selected['coverage']['selected_source_groups'] == 1


def test_conditional_failure_rates_affect_rank_not_only_alphabetic_order(pool):
    value = index(pool)
    value['records'][0]['conditions'] = ['traffic_sparse', 'nearby_vehicle']
    value['records'][1]['conditions'] = ['traffic_dense', 'nearby_vehicle']
    rates = dict(feedback(), slices=[{'capability':'collision','count':100,'failures':10}])
    cohorts = [
        {'capability':'collision','conditions':['traffic_sparse'],'count':80,'failures':2},
        {'capability':'collision','conditions':['traffic_dense'],'count':20,'failures':8},
    ]
    result = retrieve(value,rates,2,cohorts=cohorts)
    assert [r['token'] for r in result['candidates']] == ['4','3']
    assert result['candidates'][0]['match']['score'] > result['candidates'][1]['match']['score']
    assert result['candidates'][0]['requires_model_verification'] is True


def test_capability_budgets_prevent_progress_from_consuming_safety_slots(pool):
    value = index(pool)
    value['records'][0]['conditions'] = ['nearby_vehicle']
    value['records'][1]['conditions'] = ['ego_low_speed']
    rates = dict(feedback(), slices=[
        {'capability': 'collision', 'count': 1000, 'failures': 1},
        {'capability': 'route_progress', 'count': 1000, 'failures': 900},
    ])
    budget = {'collision': 1, 'route_progress': 1}
    result = retrieve(value, rates, 2, capability_budget=budget)
    assert result['coverage']['assigned_by_capability'] == budget
    assert result['coverage']['unfilled_capability_budget'] == {}
    assert {r['token'] for r in result['candidates']} == {'3', '4'}
    assert all(r['requires_model_verification'] for r in result['candidates'])
    # Input iteration order must not change deterministic tied selection.
    value['records'].reverse()
    assert retrieve(value, rates, 2, capability_budget=budget) == result


def test_capability_budget_shortage_is_reported_without_duplicate_log(pool):
    value = index(pool)
    rates = dict(feedback(), slices=[
        {'capability': 'collision', 'count': 10, 'failures': 1},
        {'capability': 'interaction', 'count': 10, 'failures': 2},
    ])
    result = retrieve(value, rates, 2, capability_budget={'collision': 1, 'interaction': 1},
                      per_source_limit=1)
    assert len(result['candidates']) == 1
    assert result['coverage']['unfilled_capability_budget'] == {'interaction': 1}


@pytest.mark.parametrize('budget', [
    {'collision': 1}, {'collision': 1, 'test-token': 1}, {'collision': 2.0},
    {'collision': True}, {}, {'collision': 0},
])
def test_invalid_capability_budgets_are_rejected(pool, budget):
    with pytest.raises(ValueError, match='Capability budget'):
        retrieve(index(pool), feedback(), 2, capability_budget=budget)
