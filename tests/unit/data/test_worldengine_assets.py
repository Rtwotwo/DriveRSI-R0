import io
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.data.assets.worldengine import bundle_files, read_config_index
from driversi.data.participants.fleet import configure_vehicle_fleet
from driversi.data.participants.traffic_continuity import configure_traffic_continuity


def archive(tmp_path, entries):
    path = tmp_path / "configs.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        for name, text in entries:
            raw = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            stream.addfile(member, io.BytesIO(raw))
    return path


def config(log):
    return (
        "!!python/object:untrusted.MustNotExecute\ncentral_log: "
        + log
        + "\ncity: us-ma-boston\nroad_block_name: "
        + log
        + "-abc\nsplit: trainval\n"
    )


def test_external_yaml_tags_are_not_executed_and_sources_stay_separate(tmp_path):
    p = archive(tmp_path, [("a.yaml", config("train-log")), ("b.yaml", config("test-log"))])
    rows = read_config_index(p, ["train-log"], ["test-log"])["rows"]
    assert {x["source_log"]: x["source_eligible"] for x in rows} == {
        "train-log": True,
        "test-log": False,
    }


def test_duplicate_asset_or_overlapping_source_is_rejected(tmp_path):
    p = archive(tmp_path, [("a.yaml", config("train")), ("b.yaml", config("train"))])
    with pytest.raises(ValueError, match="Duplicate"):
        read_config_index(p, ["train"])
    with pytest.raises(ValueError, match="Disjoint"):
        read_config_index(p, ["train"], ["train"])


def test_bundle_cannot_escape_managed_root(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        bundle_files(tmp_path, "../escape")
    files = bundle_files(tmp_path, "normal")
    assert not any(x["present"] for x in files.values())


@pytest.mark.parametrize("kind", [2, 3, 10])
def test_continuity_handles_nonvehicle_only_background(straight_scene, kind):
    import copy

    scene = copy.deepcopy(straight_scene)
    scene.kinds[2] = kind
    scene.metadata.update(origin_xyz=[0, 0, 0], map_bounds_global=[-100, -100, 100, 100])
    scene = configure_vehicle_fleet(scene)
    assert "reactive_background" not in scene.metadata

    def unsupported(*args, **kwargs):
        raise ValueError("No supported route in fixture")

    graph = SimpleNamespace(validate_scene=lambda scene: None, route_from_pose=unsupported)
    result = configure_traffic_continuity(scene, graph)
    assert result.actor_ids == scene.actor_ids
    assert "replay" in result.metadata["reactive_background"]["assignments"]
    assert "replay" in result.metadata["traffic_continuity"]["navigation"]
    np.testing.assert_array_equal(result.poses, scene.poses)
    np.testing.assert_array_equal(result.controlled, scene.controlled)


def test_source_binding_rejects_cross_bundle_and_modified_source():
    import copy

    from driversi.data.assets.worldengine import SourceBinding

    f = dict(
        token="token",
        timestamp=123,
        skipped=False,
        ego2global=np.eye(4),
        ego2global_original=np.eye(4),
        cams={},
    )
    video = dict(log_name="source", global2world_translation=[1.0, 2.0, 3.0], frame_infos=[f])
    binding = SourceBinding("source-abc", {"source-abc-0": video}, [1.0, 2.0, 3.0])
    assert binding.resolve("token") is f
    with pytest.raises(ValueError, match="identities"):
        SourceBinding("another-abc", {"source-abc-0": video}, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="origins"):
        SourceBinding("source-abc", {"source-abc-0": video}, [1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="belong"):
        binding.resolve("another-token")
    changed = copy.deepcopy(f)
    changed["ego2global"][0, 3] = 1.0
    with pytest.raises(ValueError, match="pose"):
        binding.resolve(changed)
    changed = copy.deepcopy(f)
    changed["timestamp"] += 1
    with pytest.raises(ValueError, match="timestamp"):
        binding.resolve(changed)


def test_worldengine_camera_rotates_about_vehicle_center():
    from driversi.rendering.gaussian.mtgs_registration import move_camera
    from driversi.rendering.gaussian.worldengine_portable import ego_center_pivot

    source = dict(ego2global=np.eye(4), ego2global_original=np.eye(4))
    camera = np.c_[np.eye(3), [2.461, 0.0, 1.5]]
    pivot = ego_center_pivot(source)
    moved = move_camera(source, camera, pivot, [0.0, 0.0], np.pi / 2)
    np.testing.assert_allclose(moved[:, 3], [1.461, 1.0, 1.5], atol=1e-12)
    np.testing.assert_array_equal(move_camera(source, camera, pivot, [0.0, 0.0], 0), camera)


def test_render_only_physical_key_matches_native_world(straight_scene):
    from driversi.sim.backends.pufferdrive import PufferDriveWorld, physical_scene_key

    with PufferDriveWorld(straight_scene) as world:
        assert physical_scene_key(straight_scene) == world.snapshot().scene_key


def test_branch_motion_does_not_treat_elevation_as_yaw():
    from driversi.rendering.gaussian.worldengine_replay import physical_delta

    state = np.zeros(13)
    state[:4] = [20.0, 30.0, 0.0, -2.49779344]
    source = np.array([19.7, 30.1, -0.52707186, -2.48402519])
    np.testing.assert_allclose(physical_delta(state, source), [0.3, -0.1, -0.01376825], atol=1e-8)
    state[:4] = source
    np.testing.assert_allclose(physical_delta(state, source), [0, 0, 0], atol=1e-12)
    state[2] += 100
    np.testing.assert_allclose(physical_delta(state, source), [0, 0, 0], atol=1e-12)


def test_camera_optics_use_refined_bundle_and_reject_partial_calibration():
    import numpy as np
    import pytest

    from driversi.rendering.gaussian.worldengine_portable import camera_calibration

    camera = dict(
        cam_intrinsic=np.eye(3) * 100,
        distortion=np.ones(5),
        colmap_param=dict(
            cam_intrinsic=np.array([[1200.0, 0, 960], [0, 1400, 540], [0, 0, 1]]),
            distortion=np.arange(5) / 100,
        ),
    )
    k, d = camera_calibration(camera, 960)
    np.testing.assert_array_equal(k, [[600, 0, 480], [0, 700, 270], [0, 0, 1]])
    np.testing.assert_array_equal(d, np.arange(5) / 100)
    assert camera["colmap_param"]["cam_intrinsic"][0, 0] == 1200
    del camera["colmap_param"]["distortion"]
    with pytest.raises(ValueError, match="refined"):
        camera_calibration(camera, 960)
