import copy
import pickle

import numpy as np
import pytest

from driversi.data.scenes.context import _read_map_context, _source_frames, compile_context
from driversi.data.sources.navsim import compile_frames


def source_frames():
    frames = []
    for index in range(5):
        tokens = ["vehicle-a", "cone"] if index != 2 else ["new-pedestrian"]
        kinds = ["vehicle", "traffic_cone"] if index != 2 else ["pedestrian"]
        frames.append({
            "timestamp": 1_000_000 + index * 500_000, "token": f"{index:016d}",
            "log_name": "source", "map_location": "test-map",
            "ego2global_translation": [100 + index * 2, 200, 4],
            "ego2global_rotation": [1, 0, 0, 0],
            "traffic_lights": [(123, False)],
            "anns": {"track_tokens": tokens, "gt_names": kinds,
                     "gt_boxes": np.tile([10, 0, 1, 4, 2, 1.5, 0], (len(tokens), 1))},
        })
    return frames


def test_context_preserves_arrivals_disappearances_and_unsupported_categories():
    source = source_frames()
    del source[1]["traffic_lights"]
    source[2]["traffic_lights"] = []
    scene = compile_frames(source, start=0, duration=2, split="mini")
    assert set(scene.actor_ids) == {"ego", "cone", "new-pedestrian", "vehicle-a"}
    context = _source_frames(scene, source)
    assert [obj["id"] for obj in context[0]["objects"]] == ["vehicle-a", "cone"]
    assert [obj["id"] for obj in context[2]["objects"]] == ["new-pedestrian"]
    assert context[0]["objects"][1]["kind"] == "traffic_cone"
    assert context[0]["objects"][0]["pose"] == [10, 0, 1, 0]
    assert context[0]["signals"]["records"] == [{"connector_id": "123", "is_red": False}]
    assert context[1]["signals"] == {"available": False, "records": []}
    assert context[2]["signals"] == {"available": True, "records": []}


def test_context_includes_the_endpoint_bracket_but_no_unrelated_later_frames():
    source = source_frames()
    scene = compile_frames(source, start=1, duration=.6, split="mini")
    context = _source_frames(scene, source)
    assert [frame["time_seconds"] for frame in context] == [0, .5, 1]
    assert [frame["timestamp_us"] for frame in context] == [1_500_000, 2_000_000, 2_500_000]


@pytest.mark.parametrize("change", ["identity", "origin", "actor_reference", "gap", "bad_box",
                                    "nonboolean_signal"])
def test_context_rejects_mismatched_or_invalid_source(change):
    source = source_frames()
    scene = compile_frames(source, start=0, duration=2, split="mini")
    altered = copy.deepcopy(source)
    if change == "identity":
        altered[0]["token"] = "other"
    elif change == "origin":
        altered[0]["ego2global_translation"][0] += 1
    elif change == "actor_reference":
        scene.poses[0, 0, 0] += 1
    elif change == "gap":
        altered[2]["timestamp"] += 300_000
    elif change == "bad_box":
        altered[0]["anns"]["gt_boxes"][0, 3] = 0
    elif change == "nonboolean_signal":
        altered[0]["traffic_lights"] = [(123, 1)]
    with pytest.raises(ValueError):
        _source_frames(scene, altered)


def test_complete_context_does_not_modify_scene_or_limit_display_population(tmp_path, monkeypatch):
    source = source_frames()
    scene = compile_frames(source, start=0, duration=2, split="mini", population_mode="persistent")
    scene.metadata["map_version"] = "version-one"
    # The source contains 40 display-only objects; simulation keeps its own population.
    for frame in source:
        frame["anns"] = {"track_tokens": [f"cone-{i}" for i in range(40)],
                         "gt_names": ["traffic_cone"] * 40,
                         "gt_boxes": np.asarray([[i * 2, 0, 1, .5, .5, 1, 0] for i in range(40)])}
    log = tmp_path / "source.pkl"
    log.write_bytes(pickle.dumps(source))
    original_poses = scene.poses.copy()
    original_metadata = copy.deepcopy(scene.metadata)
    observed = {}

    def map_fixture(received_scene, root, bounds):
        observed["bounds"] = bounds
        return [], {"traffic_signs": {"availability": "unavailable", "feature_count": 0}}, {
            "path": "fixture", "version": "version-one"}

    monkeypatch.setattr("driversi.data.scenes.context._read_map_context", map_fixture)
    context = compile_context(scene, log, tmp_path, trusted=True)
    assert len(context.frames[0]["objects"]) == 40
    assert context.metadata["reference_policy"] == "previous_sample"
    assert context.metadata["map_role"] == "diagnostic_only"
    assert len(context.metadata["source_sha256"]) == 64
    assert observed["bounds"][2] >= 118
    assert scene.metadata == original_metadata
    np.testing.assert_array_equal(scene.poses, original_poses)


def test_context_requires_trusted_source_and_bounded_margin(tmp_path):
    source = source_frames()
    scene = compile_frames(source, start=0, duration=2, split="mini")
    with pytest.raises(ValueError, match="trusted"):
        compile_context(scene, tmp_path / "missing.pkl", tmp_path)
    with pytest.raises(ValueError, match="margin"):
        compile_context(scene, tmp_path / "missing.pkl", tmp_path, margin=0)


def test_context_binding_checks_source_actor_type_even_when_pose_matches():
    source = source_frames()
    source[1]["ego2global_translation"][0] += 1  # Keep a controlled vehicle in this short window.
    scene = compile_frames(source, start=0, duration=.5, split="mini")
    altered = copy.deepcopy(source)
    altered[0]["anns"]["gt_names"][0] = "pedestrian"
    with pytest.raises(ValueError, match="kind"):
        _source_frames(scene, altered)


@pytest.mark.parametrize("pedestrian_layers_present", [False, True])
def test_missing_map_categories_are_distinct_from_loaded_empty(
    tmp_path, monkeypatch, pedestrian_layers_present,
):
    pyogrio = pytest.importorskip("pyogrio")
    pytest.importorskip("shapely")
    pytest.importorskip("pyproj")
    raw = pytest.importorskip("pyogrio.raw")

    scene = compile_frames(source_frames(), start=0, duration=2, split="mini")
    scene.metadata["map_version"] = "version-one"
    map_path = tmp_path / "test-map" / "version-one" / "map.gpkg"
    map_path.parent.mkdir(parents=True)
    map_path.touch()
    layer_inventory = [("baseline_paths", "LineString"), ("custom_source_layer", "Point")]
    if pedestrian_layers_present:
        layer_inventory.extend([("crosswalks", "Polygon"), ("walkways", "Polygon")])

    def read_fixture(path, *, layer, **kwargs):
        if layer == "meta":
            return {}, None, None, [["projectedCoordSystem"], ["EPSG:3857"]]
        return {"fields": []}, np.array([], dtype=int), np.array([], dtype=object), []

    monkeypatch.setattr(pyogrio, "list_layers", lambda path: layer_inventory)
    monkeypatch.setattr(pyogrio, "read_info", lambda *a, **kw: {"crs": "EPSG:3857"})
    monkeypatch.setattr(raw, "read", read_fixture)
    features, layers, _ = _read_map_context(scene, tmp_path, np.array([-20, -20, 20, 20]))
    assert features == []
    expected = "loaded_empty" if pedestrian_layers_present else "unavailable"
    for category in ("crosswalks", "walkways"):
        assert layers[category] == {"availability": expected, "feature_count": 0}
    assert layers["traffic_signs"]["availability"] == "unavailable"
    assert layers["custom_source_layer"] == {"availability": "loaded_empty", "feature_count": 0}
