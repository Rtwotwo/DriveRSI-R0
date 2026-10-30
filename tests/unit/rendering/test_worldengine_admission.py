import numpy as np
import pytest

from driversi.agents.backends.native import digest
from driversi.rendering.diagnostics.worldengine_admission import (
    CAMERAS,
    TICKS,
    _rows_by_frame,
    _sealed,
    _validate_source_frame,
)


def fixture_source(tmp_path):
    rendered = tmp_path / "0020-CAM_F0-source-render.png"
    real = tmp_path / "canonical.jpg"
    rendered.write_bytes(b"sealed producer render")
    real.write_bytes(b"real photograph")
    pose = np.c_[np.eye(3), [1.0, 2.0, 3.0]]
    canonical = {
        "token": "source",
        "timestamp": 12345,
        "cams": {
            "CAM_F0": {
                "colmap_param": {
                    "sensor2global_rotation": pose[:, :3],
                    "sensor2global_translation": pose[:, 3],
                }
            }
        },
    }
    source = {
        "tick": 20,
        "source_token": "source",
        "source_appearance_timestamp_us": 12345,
        "source_camera_to_world": pose.tolist(),
        "source_motions": {},
        "rendered_source_image": str(rendered),
        "real_source_image": str(real),
    }
    return source, canonical, {str(p): digest(p) for p in (rendered, real)}


def test_source_photograph_cannot_replace_canonical_source_render(tmp_path):
    source, canonical, files = fixture_source(tmp_path)
    _validate_source_frame(source, canonical, "CAM_F0", files)
    source["rendered_source_image"] = source["real_source_image"]
    with pytest.raises(ValueError, match="photos"):
        _validate_source_frame(source, canonical, "CAM_F0", files)


def test_source_render_substitution_after_production_is_rejected(tmp_path):
    source, canonical, files = fixture_source(tmp_path)
    from pathlib import Path

    Path(source["rendered_source_image"]).write_bytes(
        Path(source["real_source_image"]).read_bytes()
    )
    with pytest.raises(ValueError, match="Unsealed or changed"):
        _validate_source_frame(source, canonical, "CAM_F0", files)


@pytest.mark.parametrize("change", ["token", "appearance_time", "novel_pose", "motion"])
def test_source_and_counterfactual_or_other_source_cannot_mix(tmp_path, change):
    source, canonical, files = fixture_source(tmp_path)
    if change == "token":
        source["source_token"] = "another"
    if change == "appearance_time":
        source["source_appearance_timestamp_us"] += 1
    if change == "novel_pose":
        source["source_camera_to_world"][0][3] += 0.1
    if change == "motion":
        source["source_motions"] = {"ego": [1, 0, 0]}
    with pytest.raises(ValueError):
        _validate_source_frame(source, canonical, "CAM_F0", files)


def test_all_fifteen_source_times_and_four_cameras_are_required():
    rows = [{"tick": t, "camera": c} for t in TICKS for c in CAMERAS]
    assert len(_rows_by_frame(rows, "source")) == 60
    with pytest.raises(ValueError, match="Incomplete"):
        _rows_by_frame(rows[:-1], "source")
    with pytest.raises(ValueError, match="repeated"):
        _rows_by_frame(rows + rows[:1], "source")
    with pytest.raises(ValueError, match="Unexpected"):
        _rows_by_frame(rows + [{"tick": 75, "camera": "CAM_F0"}], "source")


def test_visual_seal_rejects_foreign_unregistered_file(tmp_path):
    path = tmp_path / "foreign.png"
    path.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="Unsealed"):
        _sealed(path, {})


def test_physical_motion_and_missing_actor_evidence_cannot_be_omitted():
    from types import SimpleNamespace

    from driversi.rendering.diagnostics.worldengine_admission import _validate_physical_frame

    scene = SimpleNamespace(actor_ids=("ego", "moving", "missing"), poses=np.zeros((1, 3, 4)))
    states = np.zeros((1, 3, 4))
    states[0, 1, 0] = 2
    presence = np.ones((1, 3), dtype=bool)
    frame = {
        "tick": 0,
        "physical_missing_appearance": ["missing"],
        "motions": {"ego": [0, 0, 0], "moving": [2, 0, 0]},
    }
    camera = {"active_object_tokens": ["moving"]}
    assert _validate_physical_frame(frame, camera, scene, states, presence) == []
    frame["motions"]["moving"] = [0, 0, 0]
    with pytest.raises(ValueError, match="executed object motion"):
        _validate_physical_frame(frame, camera, scene, states, presence)
    frame["motions"]["moving"] = [2, 0, 0]
    frame["physical_missing_appearance"] = []
    with pytest.raises(ValueError, match="appearance deficits"):
        _validate_physical_frame(frame, camera, scene, states, presence)


def test_extra_active_reconstruction_objects_need_per_camera_exclusion():
    from itertools import product
    from types import SimpleNamespace

    from driversi.rendering.diagnostics.worldengine_admission import _validate_physical_frame

    scene = SimpleNamespace(actor_ids=("ego",), poses=np.zeros((1, 1, 4)))
    states, presence = np.zeros((1, 1, 4)), np.ones((1, 1), dtype=bool)
    frame = {"tick": 0, "physical_missing_appearance": [], "motions": {"ego": [0, 0, 0]}}
    camera = {
        "active_object_tokens": ["ghost"],
        "camera_to_world": np.c_[np.eye(3), np.zeros(3)],
        "output_ray_bounds": [-1, 1, -1, 1],
    }
    assert _validate_physical_frame(frame, camera, scene, states, presence) == ["ghost"]
    cube = np.asarray(list(product((-1, 1), repeat=3)))
    frame["source_only_object_boxes"] = {"ghost": cube + [100, 0, 5]}
    assert _validate_physical_frame(frame, camera, scene, states, presence) == []
    frame["source_only_object_boxes"]["ghost"] = cube + [0, 0, 5]
    assert _validate_physical_frame(frame, camera, scene, states, presence) == ["ghost"]


@pytest.mark.parametrize("name, threshold", [("source27_v1", 27), ("experimental_source24_v1", 24)])
def test_explicit_quality_profiles_retain_geometry_and_do_not_claim_nvs(name, threshold):
    from driversi.rendering.diagnostics.worldengine_admission import (
        photometric_rejection,
        visual_policy,
    )

    policy = visual_policy(name)
    assert policy["source_psnr_min_db"] == threshold
    assert policy["geometry_and_temporal_checks"] == "unchanged"
    assert "not_held_out" in policy["quality_claim"]
    assert photometric_rejection(threshold, name) is None
    assert photometric_rejection(threshold - 0.01, name)


def test_low_fidelity_must_be_explicit_and_cannot_change_its_threshold():
    from driversi.rendering.diagnostics.worldengine_admission import (
        photometric_rejection,
        visual_policy,
    )

    assert visual_policy()["source_psnr_min_db"] == 27
    assert photometric_rejection(24.5, "source27_v1")
    assert photometric_rejection(24.5, "experimental_source24_v1") is None
    for value in ("source0", 24, {"source_psnr_min_db": 0}, None):
        with pytest.raises(ValueError, match="Unknown"):
            visual_policy(value)
    with pytest.raises(ValueError, match="Non-finite"):
        photometric_rejection(float("nan"), "experimental_source24_v1")


def test_static_background_frame_cannot_change_anchored_box_or_identity():
    import copy
    from itertools import product
    from types import SimpleNamespace

    from driversi.rendering.diagnostics.worldengine_admission import _validate_background_frame
    from driversi.rendering.gaussian.worldengine_static_bindings import (
        static_background_box_at_tick,
    )

    scene = SimpleNamespace(actor_ids=("ego", "fixed"), metadata={"source_start_timestamp_us": 1000000})
    states = np.zeros((2, 2, 4))
    binding = {
        "policy": "immutable_static_background_observation_v1", "actor_id": "fixed",
        "reference_tick": 0, "reference_timestamp_us": 1000000,
        "scene_start_timestamp_us": 1000000,
        "temporal_policy": "past_observed_static_background_v1",
        "source_timestamp_us": 1000000, "source_token": "observed",
        "reference_physical_pose": [0, 0, 0, 0],
        "whole_branch_immobility": {"last_tick": 1},
        "reconstruction_corners": np.asarray(list(product((-1, 1), repeat=3))).tolist(),
    }
    corners, evidence = static_background_box_at_tick(binding, states[0, 1], 0, 1000000)
    frame = {"tick": 0, "physical_missing_appearance": ["fixed"],
             "unresolved_physical_boxes": {"fixed": corners.tolist()},
             "static_background_candidates": [{"actor_id": "fixed", "immutable_source_binding": evidence}]}
    _validate_background_frame(frame, {"fixed": binding}, scene, states)
    forged = copy.deepcopy(frame)
    forged["unresolved_physical_boxes"]["fixed"][0][2] += 1
    with pytest.raises(ValueError, match="immutable source registration"):
        _validate_background_frame(forged, {"fixed": binding}, scene, states)
    forged = copy.deepcopy(frame)
    forged["static_background_candidates"][0]["immutable_source_binding"]["target_tick"] = 1
    with pytest.raises(ValueError, match="immutable source registration"):
        _validate_background_frame(forged, {"fixed": binding}, scene, states)
    states[0, 1, 0] = 0.01
    with pytest.raises(ValueError, match="physical pose changed"):
        _validate_background_frame(frame, {"fixed": binding}, scene, states)


def test_augustus_source_and_review_require_all_thirteen_times():
    rows = [{'tick': t, 'camera': c} for t in range(0, 61, 5) for c in CAMERAS]
    assert len(_rows_by_frame(rows, 'augustus', future_points=8)) == 52
    with pytest.raises(ValueError, match='Incomplete'):
        _rows_by_frame(rows[:-1], 'augustus', future_points=8)
    with pytest.raises(ValueError, match='Unexpected'):
        _rows_by_frame(rows + [{'tick': 65, 'camera': 'CAM_F0'}], 'augustus', future_points=8)
    with pytest.raises(ValueError, match='Incomplete'):
        _rows_by_frame(rows, 'chainflow-default')
    with pytest.raises(ValueError, match='eight or ten'):
        _rows_by_frame(rows, 'invalid', future_points=7)
