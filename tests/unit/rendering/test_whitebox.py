from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.rendering.whitebox.exporter import interaction_frames, sequence, source_frames
from driversi.rendering.whitebox.schema import validate_pair, validate_sequence


@pytest.fixture
def scene():
    return SimpleNamespace(
        metadata=dict(
            origin_xyz=[100.0, 200.0, 5.0],
            source_start_timestamp_us=1000000,
            scenario_id="anchor",
            source_log="log",
            map_name="map",
            split="train",
            source_start_frame=0,
            navsim_source=dict(log_path="trusted.pkl"),
        ),
        times=np.array([0.0, 0.5, 1.0]),
        actor_ids=["ego", "late"],
        kinds=np.array([1, 1]),
        dimensions=np.array([[5.0, 2.0, 1.5], [4.0, 2.0, 1.5]]),
    )


def raw_frames():
    return [
        dict(
            timestamp=1000000 + i * 500000,
            token="anchor" if i == 0 else str(i),
            log_name="log",
            map_location="map",
            ego2global_translation=[100.0 + i, 200.0, 5.0],
            ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
            traffic_lights=[],
            anns=dict(
                track_tokens=["source_only"],
                gt_names=["pedestrian"],
                gt_boxes=[[2.0, 0.0, 1.0, 0.5, 0.6, 1.8, 0.0]],
            ),
        )
        for i in range(3)
    ]


def test_source_keeps_uncontrolled_objects_and_measured_elevation(scene, monkeypatch):
    monkeypatch.setattr(
        "driversi.rendering.whitebox.exporter.load_log", lambda *a, **k: raw_frames()
    )
    frames, provenance = source_frames(scene, trusted=True)
    assert len(frames) == 3
    assert frames[0]["objects"][1]["actor_id"] == "source_only"
    assert frames[0]["objects"][1]["pose_xyz_yaw"] == [2.0, 0.0, 1.0, 0.0]
    validate_sequence(sequence(scene, "source_log", frames, provenance))


def test_source_origin_rejected(scene, monkeypatch):
    data = raw_frames()
    data[0]["ego2global_translation"][0] += 1
    monkeypatch.setattr("driversi.rendering.whitebox.exporter.load_log", lambda *a, **k: data)
    with pytest.raises(ValueError, match="origin"):
        source_frames(scene, trusted=True)


def test_branch_presence_not_source_presence(scene):
    run = dict(
        states=np.zeros((3, 2, 13)),
        presence=[[True, False], [True, True], [True, False]],
        frames=[dict(control_sources={"ego": "policy"}) for _ in range(3)],
        lifecycle=[{}, {}, {}],
    )
    frames = interaction_frames(scene, run)
    assert [len(f["objects"]) for f in frames] == [1, 2, 1]
    assert frames[1]["objects"][0]["control_source"] == "policy"
    assert all(o["actor_id"] != "source_only" for f in frames for o in f["objects"])


def test_pair_identity_and_clock_enforced(scene, monkeypatch):
    monkeypatch.setattr(
        "driversi.rendering.whitebox.exporter.load_log", lambda *a, **k: raw_frames()
    )
    frames, p = source_frames(scene, trusted=True)
    source = sequence(scene, "source_log", frames, p)
    branch = sequence(scene, "executed_branch", deepcopy(frames), deepcopy(p))
    validate_pair(source, branch)
    branch["source_log"] = "other"
    with pytest.raises(ValueError, match="source_log"):
        validate_pair(source, branch)
    branch["source_log"] = "log"
    branch["frames"][1]["timestamp_us"] += 100
    with pytest.raises(ValueError, match="clock"):
        validate_pair(source, branch)


@pytest.mark.parametrize("mutation", ["duplicate", "nan", "size", "time", "trainable"])
def test_bad_sequence_rejected(scene, monkeypatch, mutation):
    monkeypatch.setattr(
        "driversi.rendering.whitebox.exporter.load_log", lambda *a, **k: raw_frames()
    )
    frames, p = source_frames(scene, trusted=True)
    value = sequence(scene, "source_log", frames, p)
    obj = frames[0]["objects"][0]
    if mutation == "duplicate":
        frames[0]["objects"].append(deepcopy(obj))
    if mutation == "nan":
        obj["pose_xyz_yaw"][0] = float("nan")
    if mutation == "size":
        obj["dimensions_lwh"][0] = 0
    if mutation == "time":
        frames[1]["time_seconds"] = 0
    if mutation == "trainable":
        value["trainable"] = True
    with pytest.raises(ValueError):
        validate_sequence(value)


def test_absent_lifecycle_is_unknown_not_zero_contacts(scene):
    run = dict(
        states=np.zeros((3, 2, 13)),
        presence=[[True, False]] * 3,
        frames=[{} for _ in range(3)],
        lifecycle=[],
    )
    frames = interaction_frames(scene, run)
    assert all(f["lifecycle"]["contacts"] is None for f in frames)
    assert all(f["lifecycle"]["available"] is False for f in frames)
    assert all(len(f["objects"]) == 1 for f in frames)
