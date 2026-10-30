"""Quality metadata cannot turn lens coverage into training qualification."""
import copy

import numpy as np
import pytest

from driversi.rendering.diagnostics.quality import (
    assess_frame_quality,
    summarize_quality,
    validate_frame_quality,
)


def frame():
    return {"generation": 0, "tick": 2, "render_tick": 2, "simulation_timestamp_us": 1200000,
            "cameras": [{"camera_id": "front", "owner_actor_id": "ego", "virtual": False,
                         "source_timestamp_us": 1200000, "source_seed_error_us": 0,
                         "source_lens_valid_fraction": 1., "camera_to_world": np.eye(4).tolist(),
                         "camera_model": "opencv_distorted_source_lens", "width": 960, "height": 540,
                         "K": [[772., 0., 480.], [0., 746., 280.], [0., 0., 1.]],
                         "distortion": [-.33, .10, -.005, .002, 0.],
                         "camera_to_body": np.eye(4).tolist(),
                         "height_binding": {"supported": True}, "scene_height_supported": True,
                         "support": {"distance_from_seed_camera_m": 1.,
                                     "rotation_from_seed_camera_degrees": 0.}}],
            "coverage": {"source_inventories": [{"camera_id": "front",
                "source_without_reconstruction_node": [], "source_without_physics_actor": [],
                "reconstruction_only_current_nodes": []}], "ego_appearance": "camera_car_not_reconstructed"}}


def test_full_lens_and_height_proxy_do_not_qualify_reconstruction():
    result = assess_frame_quality(frame())
    assert result["diagnostic_usable"]
    assert not result["high_quality_supervision_eligible"]
    camera = result["cameras"][0]
    assert camera["checks"]["participant_inventory"] == "passed"
    assert camera["checks"]["height_proxy"] == "passed"
    assert camera["checks"]["reconstruction_quality"] == "unknown"
    assert camera["seed_camera_distance_m"] == 1.
    assert "novel_view_domain_unknown" in camera["reasons"]


@pytest.mark.parametrize("mutation,check,status", [
    (lambda f: f["cameras"][0].update(source_lens_valid_fraction=0.), "lens_sampling_support", "failed"),
    (lambda f: f["cameras"][0].pop("camera_to_world"), "finite_rigid_camera_pose", "unknown"),
    (lambda f: f["cameras"][0].update(camera_to_world=[[0.]*4]*3), "finite_rigid_camera_pose", "failed"),
    (lambda f: f.update(render_tick=1), "frame_identity_and_seed_time", "failed"),
    (lambda f: f["cameras"][0].update(source_seed_error_us=1), "frame_identity_and_seed_time", "failed"),
])
def test_invalid_or_unknown_input_is_not_diagnostic_usable(mutation, check, status):
    value = frame()
    mutation(value)
    result = assess_frame_quality(value)
    assert not result["diagnostic_usable"]
    assert result["cameras"][0]["checks"][check] == status


def test_missing_participants_and_proxy_failure_are_preserved_without_faking_visibility():
    value = frame()
    value["coverage"]["source_inventories"][0]["source_without_reconstruction_node"] = [{"kind": "pedestrian"}]
    value["cameras"][0]["height_binding"]["supported"] = False
    value["cameras"][0]["virtual"] = True
    result = assess_frame_quality(value)
    assert result["diagnostic_usable"]
    camera = result["cameras"][0]
    assert camera["checks"]["participant_inventory"] == "failed"
    assert "ego_appearance_missing_for_virtual_view" in camera["reasons"]
    assert "not_pixel_quality_or_visibility" in result["scope"]


def test_unknown_inventory_is_not_complete_and_booleans_cannot_approve_quality():
    value = frame()
    value.pop("coverage")
    value["cameras"][0]["support"].update(quality_verified=True)
    result = assess_frame_quality(value)
    assert result["cameras"][0]["checks"]["participant_inventory"] == "unknown"
    assert not result["high_quality_supervision_eligible"]


def test_changed_report_or_metadata_cannot_pass_validation():
    value = frame()
    value["render_quality"] = assess_frame_quality(value)
    value["cameras"][0]["quality"] = copy.deepcopy(value["render_quality"]["cameras"][0])
    assert validate_frame_quality(value) == value["render_quality"]
    value["cameras"][0]["source_lens_valid_fraction"] = .5
    with pytest.raises(ValueError, match="quality contradicts"):
        validate_frame_quality(value)


def test_summary_counts_reasons_and_retains_unqualified_frames():
    a, b = frame(), frame()
    b["cameras"][0].pop("camera_to_world")
    summary = summarize_quality([assess_frame_quality(a), assess_frame_quality(b)])
    assert summary["camera_frames"] == 2
    assert summary["diagnostic_usable_camera_frames"] == 1
    assert summary["high_quality_supervision_camera_frames"] == 0
    assert summary["reason_counts"]["finite_rigid_camera_pose_unknown"] == 1


def test_readiness_reports_missing_scene_even_with_checkpoint(tmp_path):
    import json

    from driversi.rendering.diagnostics.quality import audit_session_inputs

    ready = tmp_path/"checkpoint"
    ready.write_bytes(b"present")
    config = {key: str(ready) for key in ("config_path", "checkpoint_path", "road_block_config",
                                         "world_python", "native_path")}
    config.update(scene=str(tmp_path/"missing-scene"))
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    report = audit_session_inputs(path)
    assert not report["input_files_present"]
    assert {f["field"] for f in report["files"] if not f["present"]} == {"scene", "road_graph", "binding"}


def test_offline_manifest_audit_does_not_claim_new_pixel_measurement(tmp_path):
    import json

    from driversi.rendering.diagnostics.quality import audit_manifest

    path = tmp_path/"manifest.json"
    path.write_text(json.dumps({"kind": "mtgs_world_session", "status": "completed", "frames": [frame()]}))
    report = audit_manifest(path)
    assert report["quality"]["camera_frames"] == 1
    assert report["latency"]["p95_seconds"] is None
    assert report["qualification"] == "not_measured_no_new_render_or_pixel_inspection"


@pytest.mark.parametrize("field,check", [
    ("camera_model", "camera_model"), ("K", "camera_intrinsics"),
    ("distortion", "camera_distortion"), ("camera_to_body", "camera_body_transform"),
    ("width", "camera_dimensions"), ("height", "camera_dimensions"),
])
def test_missing_calibration_is_unknown_and_not_diagnostic(field, check):
    value = frame()
    value["cameras"][0].pop(field)
    report = assess_frame_quality(value)
    assert not report["diagnostic_usable"]
    assert report["cameras"][0]["checks"][check] == "unknown"


@pytest.mark.parametrize("field,value,check", [
    ("camera_model", "pinhole", "camera_model"),
    ("K", [[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]], "camera_intrinsics"),
    ("K", [[1., 0., 0.], [0., 1., 0.], [0., 1., 1.]], "camera_intrinsics"),
    ("K", [[float("nan"), 0., 0.], [0., 1., 0.], [0., 0., 1.]], "camera_intrinsics"),
    ("K", [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]], "camera_intrinsics"),
    ("K", [[1., 0.], [0., 1.]], "camera_intrinsics"),
    ("distortion", [0.] * 6, "camera_distortion"),
    ("distortion", [[0.] * 5], "camera_distortion"),
    ("distortion", [0., 0., float("inf"), 0., 0.], "camera_distortion"),
    ("camera_to_body", np.eye(4)[:3].tolist(), "camera_body_transform"),
    ("camera_to_body", np.diag([-1., 1., 1., 1.]).tolist(), "camera_body_transform"),
    ("camera_to_body", [[True] * 4] * 4, "camera_body_transform"),
    ("width", True, "camera_dimensions"), ("height", 0, "camera_dimensions"),
])
def test_invalid_calibration_cannot_be_diagnostic(field, value, check):
    sample = frame()
    sample["cameras"][0][field] = value
    report = assess_frame_quality(sample)
    assert not report["diagnostic_usable"]
    assert report["cameras"][0]["checks"][check] == "failed"


@pytest.mark.parametrize("count", [4, 5, 8, 12, 14])
def test_distortion_lengths_match_actual_opencv_runtime(count):
    sample = frame()
    sample["cameras"][0]["distortion"] = [0.] * count
    assert assess_frame_quality(sample)["diagnostic_usable"]


def test_previous_quality_version_is_not_silently_relabelled():
    sample = frame()
    sample["render_quality"] = assess_frame_quality(sample)
    sample["render_quality"]["version"] = "mtgs-frame-metadata-quality-v1"
    with pytest.raises(ValueError, match="quality contradicts"):
        validate_frame_quality(sample)
