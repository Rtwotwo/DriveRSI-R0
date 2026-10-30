"""CPU verification of observed native-asset anchors in physical coordinates."""

import numpy as np


def compiled_observed_pose(source, token, scene, *, native_centering=False):
    """Use exact compiler semantics; refined 3D GT boxes are not planar physics."""
    if list(source["track_tokens"]).count(token) != 1 or token not in scene.actor_ids:
        raise ValueError("Unique observed same-actor reference required")
    tick = int(
        round((int(source["timestamp"]) - scene.metadata["source_start_timestamp_us"]) / 100000)
    )
    expected = scene.metadata["source_start_timestamp_us"] + tick * 100000
    index = scene.actor_ids.index(token)
    if (
        abs(int(source["timestamp"]) - expected) > 50000
        or not 0 <= tick < len(scene.times)
        or tick % 5 != 0
        or not scene.presence_at(tick)[index]
    ):
        raise ValueError("Asset reference lacks actual compiled source observation")
    pose = np.asarray(scene.poses[tick, index], dtype=float)
    if native_centering:
        from driversi.data.scenes.pufferdrive import native_observed_pose

        pose = native_observed_pose(scene, pose)
    return pose, tick


def validate_binding(binding, source, token, state, tick, timestamp, scene):
    from driversi.rendering.gaussian.worldengine_replay import physical_delta

    if (
        binding.get("actor_id") != token
        or binding.get("source_token") != source["token"]
        or binding.get("source_timestamp_us") != int(source["timestamp"])
        or int(source["timestamp"]) > timestamp
        or binding.get("target_timestamp_us") != timestamp
        or binding.get("target_tick") != tick
        or binding.get("checkpoint_presence_unchanged") is not True
        or type(binding.get("native_frame_index")) is not int
        or binding["native_frame_index"] < 0
        or binding.get("policy")
        not in {"immutable_static_observation_v1", "latest_observed_asset_v1"}
    ):
        raise ValueError("Native asset anchor identity or chronological order differs")
    contract = binding.get("reference_contract")
    if contract not in {
        "sealed_scene_planar_observation_v1",
        "native_centered_scene_observation_v2",
    }:
        raise ValueError("Unknown native source observation representation")
    expected, reference_tick = compiled_observed_pose(
        source, token, scene, native_centering=contract == "native_centered_scene_observation_v2"
    )
    if binding.get("reference_tick") != reference_tick:
        raise ValueError("Native object binding source observation differs")
    for key, value in [
        ("reference_physical_pose", expected),
        ("target_physical_pose", state[:4]),
        ("planar_delta", physical_delta(state, expected)),
    ]:
        actual = np.asarray(binding.get(key), dtype=float)
        if actual.shape != np.asarray(value).shape or not np.allclose(
            actual, value, atol=1e-8, rtol=0
        ):
            raise ValueError("Native asset binding does not match executed physical state: " + key)


def exact_float32_identity(reference, target):
    """No free motion tolerance: both poses must round to identical float32 bits."""
    reference, target = np.asarray(reference), np.asarray(target)
    if (
        reference.shape != (4,)
        or target.shape != (4,)
        or not np.isfinite([*reference, *target]).all()
    ):
        raise ValueError("Finite identity reference and target required")
    equal = np.array_equal(reference.astype(np.float32), target.astype(np.float32))
    return dict(
        equal=bool(equal),
        method="identical_float32_pose_v1",
        max_roundtrip_error=float(np.max(np.abs(reference - target))),
        reference_float32=reference.astype(np.float32).tolist(),
        target_float32=target.astype(np.float32).tolist(),
    )


def observed_3d_box_corners(source, token, dimensions):
    """Source render OBB uses full observed ego pose, not planar simulator Z.

    The simulator's source state intentionally flattens ego roll/pitch. A
    distant actor can therefore have a metre-scale planar-Z difference while
    its original 3D annotation and native asset agree. This function only
    constructs the source registration check; it moves no physics or Gaussian.
    """
    import itertools

    tokens = list(source["track_tokens"])
    if tokens.count(token) != 1:
        raise ValueError("Unique same-actor 3D observation required")
    box = np.asarray(source["gt_boxes"][tokens.index(token)], dtype=float)
    dims = np.asarray(dimensions, dtype=float)
    pose = np.asarray(source["ego2global"], dtype=float)
    if (
        box.shape != (7,)
        or dims.shape != (3,)
        or pose.shape != (4, 4)
        or not np.isfinite([*box, *dims, *pose.ravel()]).all()
        or np.any(dims <= 0)
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
    ):
        raise ValueError("Finite rigid source 3D annotation required")
    c, s = np.cos(box[6]), np.sin(box[6])
    yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = np.asarray(list(itertools.product((-1, 1), repeat=3))) * dims / 2
    local = local @ yaw.T + box[:3]
    corners = local @ pose[:3, :3].T + pose[:3, 3]
    return corners, dict(
        contract="same_actor_observed_3d_obb_v1",
        source_token=source["token"],
        source_timestamp_us=int(source["timestamp"]),
        actor_id=token,
        observed_box=box.tolist(),
        physical_dimensions=dims.tolist(),
        center=corners.mean(0).tolist(),
        scope="source_annotation_registration_only_planar_physics_unchanged",
    )
