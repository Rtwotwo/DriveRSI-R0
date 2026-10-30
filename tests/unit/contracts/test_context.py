import copy
import json

import pytest

from driversi.contracts.context import SceneContext, scene_fingerprint


def context_for(scene):
    return SceneContext(
        {"schema_version": 1, "scene_fingerprint": scene_fingerprint(scene),
         **{key: scene.metadata.get(key) for key in ("scenario_id", "source_log", "map_name",
                                                    "map_version", "origin_xyz")},
         "scene_times": scene.times.tolist(), "reference_policy": "previous_sample",
         "max_sample_age_seconds": .75},
        [{"time_seconds": t, "timestamp_us": 1_000_000 + round(t * 1e6),
          "objects": [{"id": "source-only", "kind": "unknown-category", "pose": [1, 2, 0, .4],
                       "dimensions": [1, .5, 1]}],
          "signals": {"available": True, "records": [{"connector_id": "17", "is_red": False}]}}
         for t in [0, .5, float(scene.times[-1])]],
        [{"id": "crosswalks:1", "layer": "crosswalks", "properties": {},
          "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 1], [0, 0]]]}}],
        {"crosswalks": {"availability": "available", "feature_count": 1},
         "traffic_signs": {"availability": "unavailable", "feature_count": 0},
         "walkways": {"availability": "loaded_empty", "feature_count": 0}},
    )


def test_context_roundtrip_keeps_unknown_category_and_missing_vs_empty(straight_scene, tmp_path):
    context = context_for(straight_scene)
    path = tmp_path / "context.json"
    context.save(path)
    restored = SceneContext.load(path)
    restored.validate_scene(straight_scene)
    assert restored.to_dict() == context.to_dict()
    assert restored.map_layers["traffic_signs"] != restored.map_layers["walkways"]
    assert restored.frames[0]["objects"][0]["kind"] == "unknown-category"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        context.save(path)
    assert path.read_bytes() == before
    isolated = context.to_dict()
    isolated["frames"][0]["objects"].clear()
    assert context.frames[0]["objects"]


def test_context_binding_rejects_different_scene_content_and_coordinates(straight_scene):
    context = context_for(straight_scene)
    original = copy.deepcopy(straight_scene)
    straight_scene.poses[-1, 0, 0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        context.validate_scene(straight_scene)
    context.metadata["origin_xyz"][0] += 10
    with pytest.raises(ValueError, match="origin"):
        context.validate_scene(original)


@pytest.mark.parametrize("mutation", [
    lambda c: c.frames[0]["objects"].append(copy.deepcopy(c.frames[0]["objects"][0])),
    lambda c: c.frames[0]["objects"][0]["dimensions"].__setitem__(0, -1),
    lambda c: c.frames[1].__setitem__("timestamp_us", 1_600_000),
    lambda c: c.frames[0]["signals"].__setitem__("available", False),
    lambda c: c.frames[0]["signals"]["records"][0].__setitem__("is_red", "false"),
    lambda c: c.map_features[0]["geometry"]["coordinates"][0][-1].__setitem__(0, 1),
    lambda c: c.map_layers["crosswalks"].__setitem__("availability", "unavailable"),
    lambda c: c.map_layers["crosswalks"].__setitem__("feature_count", 2),
    lambda c: c.metadata.__setitem__("reference_policy", "future_nearest"),
])
def test_context_rejects_ambiguous_or_invalid_records(straight_scene, mutation):
    context = context_for(straight_scene)
    mutation(context)
    with pytest.raises(ValueError):
        context.validate()


def test_context_rejects_nonfinite_json(straight_scene, tmp_path):
    path = tmp_path / "bad.json"
    data = context_for(straight_scene).to_dict()
    data["map_features"][0]["properties"]["bad"] = float("nan")
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        SceneContext.load(path)


def test_task_only_context_rebinding_preserves_source_and_rejects_other_changes(straight_scene):
    from driversi.contracts.driving_task import bind_driving_tasks
    from driversi.contracts.road import Route

    context = context_for(straight_scene)
    original = context.to_dict()
    tasked = bind_driving_tasks(straight_scene, {
        "a": Route([[0, 0], [50, 0]]), "b": Route([[0, 8], [50, 8]])
    })
    rebound = context.with_driving_tasks(straight_scene, tasked)
    rebound.validate_scene(tasked)
    assert rebound.frames == context.frames and rebound.map_features == context.map_features
    assert context.to_dict() == original
    rebound.frames[0]["objects"].clear()
    assert context.frames[0]["objects"]
    for attribute in ("poses", "dimensions"):
        changed = copy.deepcopy(tasked)
        getattr(changed, attribute).flat[0] += 0.01
        with pytest.raises(ValueError, match="only driving task"):
            context.with_driving_tasks(straight_scene, changed)
    changed = copy.deepcopy(tasked)
    changed.metadata["source_log"] += "-different"
    with pytest.raises(ValueError, match="only driving task"):
        context.with_driving_tasks(straight_scene, changed)
    with pytest.raises(ValueError, match="fingerprint"):
        context.with_driving_tasks(tasked, tasked)
