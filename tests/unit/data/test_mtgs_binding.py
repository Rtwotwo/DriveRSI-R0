"""Source identity and coordinate provenance must fail closed before rendering."""

import copy

import numpy as np
import pytest

from driversi.contracts.scene import Scene
from driversi.data.assets.mtgs_binding import audit_source_files, match_navsim_traversal


def pair():
    original = np.eye(4)
    original[:3, 3] = [2, 3, 4]
    global_pose = original.copy()
    global_pose[:3, 3] += [100, 200, 5]
    objects = {"track_tokens": ["car", "person"], "gt_names": ["vehicle", "pedestrian"],
               "gt_boxes": np.asarray([[1, 2, 0, 4, 2, 1, 0], [3, 4, 0, 1, 1, 2, 0]])}
    frames, source = [], []
    for i in range(2):
        base = {"token": f"frame{i}", "timestamp": i*500000, "log_name": "log"}
        camera = {"data_path": f"log/CAM_F0/camera{i}.jpg"}
        frames.append({**base, "map_location": "map", "ego2global": global_pose.copy(),
                       "cams": {"CAM_F0": camera}, "anns": copy.deepcopy(objects)})
        refined = original.copy()
        refined[0, 3] += 0.25*i
        source.append({**base, "ego2global_original": original.copy(), "ego2global": refined,
                       "cams": {"CAM_F0": {**camera, "token": f"camera{i}",
                                            "timestamp": i*500000-12000}}, **copy.deepcopy(objects)})
    video = {"log_name": "log", "map_location": "map", "video_token": "block-7",
             "global2world_translation": [100, 200, 5], "frame_infos": source}
    return frames, video


def test_source_file_entry_refuses_untrusted_deserialization_before_read(tmp_path):
    missing = tmp_path / "not-read.pkl"
    with pytest.raises(ValueError, match="trusted-pickle"):
        audit_source_files(log_path=missing, metadata_path=missing, scene_path=missing, travel_id=7)


def test_exact_identity_keeps_refinement_and_exposure_offsets_explicit():
    frames, video = pair()
    result = match_navsim_traversal(frames, video)
    assert result["match_count"] == 2
    assert result["matched_span_seconds"] == 0.5
    assert result["refinement_residual"]["max_translation_norm_m"] == 0.25
    first = result["frames"][0]
    assert first["camera_timestamp_us"] == -12000
    assert first["timestamp_us"] == 0
    assert all(obj["physics_bound"] is None for obj in first["source_objects"])
    assert result["render_binding_ready"] is False
    assert result["trainable"] is False


def test_missing_reconstruction_annotation_does_not_erase_source_person():
    frames, video = pair()
    source = video["frame_infos"][0]
    for key in ("track_tokens", "gt_names", "gt_boxes"):
        source[key] = source[key][:1]
    result = match_navsim_traversal(frames, video)
    person = result["frames"][0]["source_objects"][1]
    assert person["track_token"] == "person"
    assert person["present_in_mtgs_annotations"] is False


@pytest.mark.parametrize("mutation,reason", [
    (lambda f, v: v["frame_infos"][0].update(timestamp=1), "different timestamps"),
    (lambda f, v: v["frame_infos"][0]["cams"]["CAM_F0"].update(data_path="wrong.jpg"), "paths differ"),
    (lambda f, v: v["frame_infos"][0]["cams"]["CAM_F0"].update(token="wrong"), "token differs"),
    (lambda f, v: v.update(global2world_translation=[100, 200, 6]), "pose disagree"),
    (lambda f, v: f[0].update(map_location="wrong"), "different source"),
    (lambda f, v: f.append(copy.deepcopy(f[0])), "Duplicate"),
    (lambda f, v: f[0].update(timestamp=0.0), "integer microseconds"),
])
def test_identity_mismatches_are_rejected(mutation, reason):
    frames, video = pair()
    mutation(frames, video)
    with pytest.raises(ValueError, match=reason):
        match_navsim_traversal(frames, video)


def test_wrong_box_and_nonrigid_pose_are_rejected():
    frames, video = pair()
    video["frame_infos"][0]["gt_boxes"][0, 0] += 1
    with pytest.raises(ValueError, match="annotations differ"):
        match_navsim_traversal(frames, video)
    frames, video = pair()
    video["frame_infos"][0]["ego2global"][0, 0] = 2
    with pytest.raises(ValueError, match="rigid 4x4"):
        match_navsim_traversal(frames, video)


def test_skipped_or_unmatched_source_is_not_claimed_as_support():
    frames, video = pair()
    for frame in video["frame_infos"]:
        frame["skipped"] = "out_of_region"
    with pytest.raises(ValueError, match="No exact usable"):
        match_navsim_traversal(frames, video)


def test_scene_binding_preserves_unbound_objects_and_checks_full_window():
    frames, video = pair()
    scene = Scene(
        {"schema_version": 1, "scenario_id": "frame0", "source_log": "log", "map_name": "map",
         "split": "development", "origin_xyz": [100, 200, 5], "source_start_frame": 0,
         "source_frames": 2, "source_start_timestamp_us": 0},
        ("car",), np.ones(1, dtype=np.int32), np.ones((1, 3)), np.array([0., .5]),
        np.zeros((2, 1, 4)), np.zeros((2, 1, 3)), np.ones(1, dtype=np.bool_),
        np.empty((0, 3)), np.array([0], dtype=np.int64), np.empty(0, dtype=np.int32),
    )
    result = match_navsim_traversal(frames, video, scene=scene)
    objects = result["frames"][0]["source_objects"]
    assert objects[0]["physics_bound"] is True
    assert objects[1]["physics_bound"] is False
    video["frame_infos"][-1]["skipped"] = "out_of_region"
    with pytest.raises(ValueError, match="every compiled scene"):
        match_navsim_traversal(frames, video, scene=scene)
    scene.metadata["source_start_timestamp_us"] = 1
    with pytest.raises(ValueError, match="start identity"):
        match_navsim_traversal(frames, video, scene=scene)

