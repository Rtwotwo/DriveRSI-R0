"""Static GS anchors must follow source 3D, not planar simulator elevation."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.data.scenes.pufferdrive import native_observed_pose
from driversi.rendering.gaussian.worldengine_static_bindings import (
    build_static_background_binding,
    static_background_box_at_tick,
    validate_static_background_binding,
)


def fixture(kind="vehicle"):
    roll = 0.08
    rotation = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    refined = np.eye(4)
    refined[:3, :3] = rotation
    box = np.array([10.0, 40.0, 0.5, 4.0, 2.0, 1.8, 0.0])
    frames, sources = [], {}
    for i in range(3):
        frame = dict(
            token=f"frame{i}",
            timestamp=i * 500000,
            log_name="train-log",
            ego2global_translation=[0, 0, 0],
            ego2global_rotation=[1, 0, 0, 0],
            anns=dict(
                track_tokens=["object"],
                gt_names=[kind],
                gt_boxes=[box.copy()],
                gt_velocity_3d=[[0.0, 0.0, 0.0]],
            ),
        )
        frames.append(frame)
        # Deliberately omitted source track list, just like native static filtering.
        sources[frame["token"]] = dict(
            token=frame["token"],
            timestamp=frame["timestamp"],
            ego2global=refined.copy(),
            ego2global_original=np.eye(4),
        )
    poses = np.zeros((11, 2, 4))
    poses[:, 1] = [10, 40, 0.5, 0]
    scene = SimpleNamespace(
        actor_ids=("ego", "object"),
        controlled=np.array([True, False]),
        times=np.arange(11) * 0.1,
        poses=poses,
        dimensions=np.array([[5, 2, 2], box[3:6]]),
        road_points=np.array([[0, 0, 0], [0.7, 3.4, 0], [50, 1, 0]]),
        metadata=dict(
            source_log="train-log",
            source_start_timestamp_us=0,
            origin_xyz=[0, 0, 0],
            source_actor_kinds={"ego": "vehicle", "object": kind},
        ),
    )
    scene.presence_at = lambda tick: np.ones(2, dtype=bool)
    states = np.zeros((11, 2, 13))
    states[:, 1, :4] = native_observed_pose(scene, poses[0, 1])
    presence = np.ones((11, 2), dtype=bool)
    return scene, states, presence, frames, sources


def build(values):
    scene, states, presence, frames, sources = values
    return build_static_background_binding(
        scene, states, presence, frames, sources.__getitem__, "object"
    )


def test_full_source_roll_is_fixed_and_never_uses_planar_physical_z():
    data = fixture()
    binding = build(data)
    scene, states, presence, frames, sources = data
    expected = sources["frame0"]["ego2global"][:3, :3] @ frames[0]["anns"]["gt_boxes"][0][:3]
    assert expected[2] - states[0, 1, 2] > 3.0
    np.testing.assert_allclose(np.mean(binding["reconstruction_corners"], axis=0), expected)
    # Later reconstructed local registration drift must not drag a parked asset.
    sources["frame2"]["ego2global"][2, 3] = 5.0
    other = build(data)
    assert other == binding
    a, _ = static_background_box_at_tick(binding, states[0, 1], 0, 0)
    b, _ = static_background_box_at_tick(binding, states[10, 1], 10, 1000000)
    np.testing.assert_array_equal(a, b)
    assert (
        validate_static_background_binding(
            binding, scene, states, presence, frames, sources.__getitem__
        )
        == binding
    )
    assert binding["visual_admitted"] is False


def test_exact_native_pose_identity_rejects_motion_and_disappearance():
    data = fixture()
    data[1][6, 1, 0] = np.nextafter(np.float32(data[1][6, 1, 0]), np.float32(np.inf))
    with pytest.raises(ValueError, match="moved"):
        build(data)
    data = fixture()
    data[2][-1, 1] = False
    with pytest.raises(ValueError, match="continue"):
        build(data)


@pytest.mark.parametrize(
    "failure", ["velocity", "missing_velocity", "single", "unknown", "position"]
)
def test_vehicle_requires_real_parking_evidence(failure):
    data = fixture()
    if failure == "velocity":
        data[3][-1]["anns"]["gt_velocity_3d"] = [[1, 0, 0]]
    elif failure == "missing_velocity":
        del data[3][-1]["anns"]["gt_velocity_3d"]
    elif failure == "single":
        data[3][:] = data[3][:1]
    elif failure == "unknown":
        for frame in data[3]:
            frame["anns"]["gt_names"] = ["unknown"]
        data[0].metadata["source_actor_kinds"]["object"] = "unknown"
    else:
        data[3][-1]["anns"]["gt_boxes"][0][0] += 2
    with pytest.raises(ValueError):
        build(data)


def test_native_static_category_is_not_unknown_or_vehicle_velocity_guess():
    data = fixture("generic_object")
    for f in data[3]:
        f["anns"].pop("gt_velocity_3d")
    binding = build(data)
    assert binding["kind_or_parking_provenance"]["rule"] == "native_nuplan_static_object_type_v1"
    assert binding["source_kind"] == "generic_object"


def test_late_static_annotation_cannot_be_backdated_without_explicit_survey_prior():
    data = fixture("traffic_cone")
    data[3][:] = data[3][1:]
    with pytest.raises(ValueError, match="future"):
        build(data)


def test_recompute_rejects_modified_anchor_and_target_requires_identity():
    data = fixture()
    binding = build(data)
    bad = copy.deepcopy(binding)
    bad["reconstruction_corners"][0][2] += 0.01
    with pytest.raises(ValueError, match="differs"):
        validate_static_background_binding(bad, *data[:4], data[4].__getitem__)
    changed = data[1][1, 1].copy()
    changed[0] += 0.01
    with pytest.raises(ValueError, match="changed"):
        static_background_box_at_tick(binding, changed, 1, 100000)
    with pytest.raises(ValueError, match="before"):
        static_background_box_at_tick(binding, data[1][1, 1], -1, -100000)


def test_reviewed_static_survey_prior_keeps_later_annotation_time(straight_scene, tmp_path):
    import hashlib

    from driversi.contracts.scene import OTHER_OBJECT
    from driversi.data.participants.traffic_continuity import configure_traffic_continuity
    from driversi.data.scenes.static_context import apply_static_scene_context

    source = copy.deepcopy(straight_scene)
    source.kinds[2] = OTHER_OBJECT
    source.poses[:, 2] = [10, -8, 0, 0]
    source.velocities[:, 2] = 0
    source.presence = np.ones(source.poses.shape[:2], bool)
    source.presence[:5, 2] = False
    source.metadata.update(
        source_start_frame=0,
        source_start_timestamp_us=0,
        source_actor_kinds={"replay": "generic_object"},
        origin_xyz=[0, 0, 0],
        map_bounds_global=[-100, -100, 100, 100],
    )

    class Graph:
        def validate_scene(self, scene):
            scene.validate()

    source = configure_traffic_continuity(source, Graph())
    image = tmp_path / "initial.jpg"
    image.write_bytes(b"explicit-reviewed-source-image-test-fixture")
    frames = [
        dict(
            token="initial",
            timestamp=0,
            log_name=source.metadata["source_log"],
            anns=dict(track_tokens=[]),
            cams={"CAM_B0": {"data_path": "initial.jpg"}},
        ),
        dict(
            token="later",
            timestamp=500000,
            log_name=source.metadata["source_log"],
            ego2global_translation=[0, 0, 0],
            ego2global_rotation=[1, 0, 0, 0],
            anns=dict(
                track_tokens=["replay"],
                gt_names=["generic_object"],
                gt_boxes=[[10, -8, 0, *source.dimensions[2], 0]],
                gt_velocity_3d=[[0, 0, 0]],
            ),
        ),
    ]
    manifest = dict(
        version=1,
        policy="survey_prior_static_native_v1",
        source_log=source.metadata["source_log"],
        actors={
            "replay": dict(
                immutable_visible_at_scene_start_reviewed=True,
                reviewer="test-fixture-review",
                evidence_refs=[
                    dict(
                        raw_index=0,
                        camera="CAM_B0",
                        path=str(image),
                        sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    )
                ],
            )
        },
    )
    scene = apply_static_scene_context(source, frames, manifest)
    states = np.zeros((*scene.poses.shape[:2], 13))
    states[:, 2, :4] = native_observed_pose(scene, scene.poses[0, 2])
    canonical = dict(
        token="later", timestamp=500000, ego2global=np.eye(4), ego2global_original=np.eye(4)
    )

    def resolver(token):
        return canonical

    binding = build_static_background_binding(
        scene, states, scene.presence, frames, resolver, "replay"
    )
    assert binding["source_timestamp_us"] == 500000
    assert binding["reference_timestamp_us"] == 0
    assert binding["temporal_policy"] == "offline_survey_prior_static_native_v1"
    corners, evidence = static_background_box_at_tick(binding, states[0, 2], 0, 0)
    np.testing.assert_allclose(corners.mean(0), [10, -8, 0])
    assert evidence["source_timestamp_us"] == 500000
    assert (
        validate_static_background_binding(binding, scene, states, scene.presence, frames, resolver)
        == binding
    )
    image.write_bytes(b"changed-test-fixture")
    with pytest.raises(ValueError, match="evidence changed"):
        validate_static_background_binding(binding, scene, states, scene.presence, frames, resolver)


def test_target_timestamp_must_match_sealed_tick():
    data = fixture()
    binding = build(data)
    with pytest.raises(ValueError):
        static_background_box_at_tick(binding, data[1][1, 1], 1, 200000)


def test_self_consistent_shifted_scene_and_state_cannot_rebind_normal_source_pixels():
    data = fixture()
    scene, states, presence, frames, sources = data
    scene.poses[:, 1, 0] += 0.1
    states[:, 1, :4] = native_observed_pose(scene, scene.poses[0, 1])
    with pytest.raises(ValueError, match="original raw source geometry"):
        build(data)


def test_stationary_bicycle_keeps_category_and_requires_unchanged_physical_pose():
    data = fixture("bicycle")
    binding = build(data)
    assert binding["source_kind"] == "bicycle"
    assert binding["kind_or_parking_provenance"]["source_category"] == "bicycle"
    assert binding["visual_admitted"] is False
    data[1][5, 1, 0] += 0.001
    with pytest.raises(ValueError, match="moved"):
        build(data)


@pytest.mark.parametrize("failure", ["moving", "single", "missing_velocity", "drift"])
def test_bicycle_label_alone_never_proves_background_support(failure):
    data = fixture("bicycle")
    if failure == "moving":
        data[3][-1]["anns"]["gt_velocity_3d"] = [[0.21, 0, 0]]
    elif failure == "single":
        data[3][:] = data[3][:1]
    elif failure == "missing_velocity":
        del data[3][-1]["anns"]["gt_velocity_3d"]
    else:
        data[3][-1]["anns"]["gt_boxes"][0][0] += 1.51
    with pytest.raises(ValueError):
        build(data)
