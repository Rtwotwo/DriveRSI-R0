import copy
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.rendering.diagnostics.dynamic_binding import source_registration, validate_history_binding
from driversi.rendering.diagnostics.worldengine_admission import _rows_by_frame


def fixture():
    poses = np.zeros((71, 2, 4)); poses[:, 1, :3] = [5, 0, 10]
    states = poses.copy(); states[:, 1, 0] += 1
    scene = SimpleNamespace(actor_ids=("ego", "car"), poses=poses,
                            metadata={"source_actor_kinds": {"car": "vehicle"}})
    source = dict(token="capture", track_tokens=["car"], gt_boxes=[[5, 0, 10, 4, 2, 2, 0]], ego2global=np.eye(4), ego2global_original=np.eye(4))
    source["cams"] = {"CAM_F0": {"colmap_param": {"sensor2global_rotation": np.eye(3), "sensor2global_translation": np.zeros(3)}}}
    binding = source_registration(source, "car", [[5, 0, 10], [5, 0, 10]])
    binding.update(source_pivot=[5, 0, 10], applied_planar_delta=[1, 0, 0],
                   applied_translation=[1, 0, 0], decoded_target_center=[6, 0, 10])
    camera = dict(active_object_tokens=["car"], physical_object_transport={"car": binding},
                  terrain_evidence={"car": {"status": "unsupported"}, "ego": {"status": "unsupported"}},
                  camera_to_rear_axle=np.eye(4),
                  camera_to_world=np.eye(4)[:3], output_ray_bounds=[-1, 1, -1, 1])
    frames = [dict(tick=tick, source_token="capture", physical_missing_appearance=[],
                   motions={"ego": [0, 0, 0], "car": [1, 0, 0]}, cameras={"CAM_F0": copy.deepcopy(camera)})
              for tick in [5, 10, 15, 20]]
    return dict(image_contract="history_rgb_future_states_v1", frames=frames), scene, states, np.ones((71, 2), bool), {"capture": source}


def test_actual_registered_cloud_and_same_branch_motion():
    audit = validate_history_binding(*fixture())
    assert not audit["rejection_reasons"]
    assert audit["independent_pixel_review"] is False


def test_tampered_object_motion_cannot_pass():
    args = fixture(); args[0]["frames"][0]["motions"]["car"][0] = 3
    with pytest.raises(ValueError, match="executed object motion"):
        validate_history_binding(*args)


def test_unregistered_actual_cloud_is_rejected():
    args = fixture(); item = args[0]["frames"][0]["cameras"]["CAM_F0"]["physical_object_transport"]["car"]
    item.update(source_registration(args[4]["capture"], "car", [[100, 0, 10]]))
    item["decoded_target_center"] = [101, 0, 10]
    assert "source_object_cloud_not_registered_to_observed_box" in validate_history_binding(*args)["rejection_reasons"]


def test_visible_missing_vehicle_not_excused_by_psnr():
    args = fixture(); frame = args[0]["frames"][0]
    frame["physical_missing_appearance"] = ["car"]
    frame["motions"].pop("car")
    frame["unresolved_physical_boxes"] = {"car": [[x, y, z] for x in [4, 8] for y in [-1, 1] for z in [9, 11]]}
    frame["cameras"]["CAM_F0"]["active_object_tokens"] = []
    frame["cameras"]["CAM_F0"]["physical_object_transport"] = {}
    assert "visible_physical_actor_without_bound_appearance" in validate_history_binding(*args)["rejection_reasons"]


def test_history_comparison_requires_exact_frames_and_forbids_future():
    rows = [dict(tick=t, camera=c) for t in [5, 10, 15, 20] for c in ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]]
    assert len(_rows_by_frame(rows, "source", image_contract="history_rgb_future_states_v1")) == 16
    with pytest.raises(ValueError, match="Unexpected"):
        _rows_by_frame(rows + [dict(tick=25, camera="CAM_F0")], "source", image_contract="history_rgb_future_states_v1")


def test_forged_source_registration_flag_rejected():
    args = fixture(); args[0]["frames"][0]["cameras"]["CAM_F0"]["physical_object_transport"]["car"]["observed_box"][0] = 99
    with pytest.raises(ValueError, match="source observation"):
        validate_history_binding(*args)


def test_rasterization_camera_must_follow_branch_not_just_declared_delta():
    args = fixture(); args[0]["frames"][0]["cameras"]["CAM_F0"]["camera_to_world"][0, 3] += 1
    with pytest.raises(ValueError, match="Rasterization camera"):
        validate_history_binding(*args)


def test_arbitrary_vertical_object_shift_rejected():
    args = fixture(); binding = args[0]["frames"][0]["cameras"]["CAM_F0"]["physical_object_transport"]["car"]
    binding["applied_translation"][2] += 2; binding["decoded_target_center"][2] += 2
    with pytest.raises(ValueError, match="vertical transport"):
        validate_history_binding(*args)


def test_anchor_binding_requires_only_actual_anchor_and_rejects_other_ticks():
    args = fixture()
    args[0]["image_contract"] = "anchor_rgb_history_future_states_v1"
    args[0]["frames"] = args[0]["frames"][-1:]
    assert not validate_history_binding(*args)["rejection_reasons"]
    args[0]["frames"][0]["tick"] = 25
    with pytest.raises(ValueError, match="image ticks"):
        validate_history_binding(*args)


def test_anchor_psnr_coverage_is_four_and_threshold_remains_strict():
    from driversi.rendering.diagnostics.worldengine_admission import photometric_rejection, visual_policy
    from driversi.rendering.diagnostics.numeric_admission import numeric_settings
    rows = [dict(tick=20, camera=c) for c in ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]]
    assert len(_rows_by_frame(rows, "source", image_contract="anchor_rgb_history_future_states_v1")) == 4
    with pytest.raises(ValueError, match="Unexpected"):
        _rows_by_frame(rows + [dict(tick=15, camera="CAM_F0")], "source", image_contract="anchor_rgb_history_future_states_v1")
    assert photometric_rejection(25, "dynamic_anchor25_v1")
    assert photometric_rejection(25.01, "dynamic_anchor25_v1") is None
    assert numeric_settings("dynamic_anchor25_v1")["strict_psnr"]
    assert visual_policy("dynamic_anchor25_v1")["require_complete_object_binding"]


def test_native_storage_retry_requires_settled_nonzero_exit(tmp_path, monkeypatch):
    import json
    import os
    from driversi.agents.backends.native import digest
    from driversi.rendering.diagnostics.dynamic_history import settled_storage_failure
    assert not settled_storage_failure(tmp_path)
    job=tmp_path/'job.json';job.write_text('{}')
    (tmp_path/'launch.json').write_text(json.dumps({'pid':999999999}))
    (tmp_path/'worker.log').write_text('OSError: [Errno 39] Directory not empty')
    (tmp_path/'exit.json').write_text(json.dumps({'returncode':1,'job_sha256':digest(job)}))
    def absent(*args):raise ProcessLookupError()
    monkeypatch.setattr(os,'killpg',absent)
    assert settled_storage_failure(tmp_path)
    monkeypatch.setattr(os,'killpg',lambda *args:None)
    assert not settled_storage_failure(tmp_path)
    monkeypatch.setattr(os,'killpg',absent)
    (tmp_path/'worker.log').write_text('CUDA out of memory')
    assert not settled_storage_failure(tmp_path)
