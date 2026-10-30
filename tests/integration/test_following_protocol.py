import json

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from driversi.contracts.scene import Scene
from driversi.contracts.transition import Snapshot
from driversi.evaluation.benchmarks.following import _metrics, run_case
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.rendering.bev.replay import export_replay
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout

pytestmark = pytest.mark.native


def test_stopped_actor_does_not_mean_geometric_contact():
    states = np.zeros((2, 3, 13))
    states[:, 1, 0] = 30
    states[1, 0, 9] = 1
    assert not _metrics({"states": states})["geometric_contact_any"]


def test_paired_braking_replay_and_far_negative_controls(tmp_path):
    report = run_case(8, 30, output=tmp_path)
    assert report["passed"], {name: passed for name, passed in report["checks"].items() if not passed}
    assert report["modes"]["reactive"]["desired_response_delay_s"] == pytest.approx(.1)


def test_paired_rollouts_preserve_real_prefix_and_pass_viewer_validation(tmp_path):
    run_case(8, 30, output=tmp_path)
    for mode in ("reactive", "replay"):
        scene = Scene.load(tmp_path / f"{mode}-scene.npz")
        runs = {}
        for branch in ("keep", "brake", "far"):
            path = tmp_path / f"{mode}-{branch}-rollout.npz"
            with np.load(path, allow_pickle=False) as saved:
                data = {key: saved[key].copy() for key in saved.files}
            runs[branch] = data
            metadata = json.loads(str(data["metadata"]))
            diagnostics = json.loads(str(data["controller_diagnostics"]))
            assert data["states"].shape == (91, 3, 13)
            assert data["motion"].shape == (91, 3, 3)
            assert data["actions"].shape == (90, int(scene.controlled.sum()))
            assert metadata["steps"] == 90
            assert metadata["protocol"]["common_prefix_steps"] == 10
            assert metadata["protocol"]["intervention_seconds"] == 1
            assert metadata["protocol"]["metric_interval_start_seconds"] == 1
            assert set(metadata["controller_configuration"]["routes"]) == ({"follower"} if mode == "reactive" else set())
            assert len(diagnostics) == 90
            for step, commands in enumerate(diagnostics):
                assert [command["actor"] for command in commands] == list(data["controlled_ids"])
                assert [command["action"] for command in commands] == list(data["actions"][step])
            with PufferDriveWorld(scene) as world:
                np.testing.assert_array_equal(data["states"][0], world.state())
                np.testing.assert_array_equal(data["motion"][0], world.motion())
            with np.load(tmp_path / f"{mode}-{branch}.npz", allow_pickle=False) as interval:
                # Existing protocol metrics and raw records retain their 1–9 s domain.
                for name in ("states", "motion", "actions", "valid"):
                    np.testing.assert_array_equal(data[name][10:], interval[name])
            result = export_replay(scene, path, tmp_path / f"{mode}-{branch}.html")
            assert result["frames"] == 91
        for branch in ("brake", "far"):
            for name in ("states", "motion"):
                np.testing.assert_array_equal(runs["keep"][name][:11], runs[branch][name][:11])
            for name in ("actions", "rewards", "valid", "terminated", "truncated", "collision", "offroad", "goal"):
                np.testing.assert_array_equal(runs["keep"][name][:10], runs[branch][name][:10])
        compared = export_replay(scene, tmp_path / f"{mode}-keep-rollout.npz",
                                 tmp_path / f"{mode}-comparison.html",
                                 compare_path=tmp_path / f"{mode}-brake-rollout.npz")
        assert compared["frames"] == 91 and compared["comparison"]
        # Restore the saved 1 s state and execute recorded joint actions: exact state,
        # actuator memory, events and masks must match the full-prefix trajectory.
        with PufferDriveWorld(scene) as restored:
            restored.restore(Snapshot.load(tmp_path / f"{mode}-snapshot.npz"))
            brake = runs["brake"]
            np.testing.assert_array_equal(restored.state(), brake["states"][10])
            for step in range(10, 90):
                transition = restored.step(brake["actions"][step])
                np.testing.assert_array_equal(restored.state(), brake["states"][step+1])
                np.testing.assert_array_equal(restored.motion(), brake["motion"][step+1])
                for name in ("valid", "terminated", "truncated", "rewards"):
                    np.testing.assert_array_equal(getattr(transition, name), brake[name][step])
                for name in ("collision", "offroad", "goal"):
                    np.testing.assert_array_equal(transition.events[name], brake[name][step])


def test_prefix_join_rejects_mismatched_world_state():
    with PufferDriveWorld(following_scene(8, 30)) as world:
        prefix = RolloutRecord.start(world)
        world.step({actor: 7 for actor in world.actor_ids})
        later = RolloutRecord.start(world)
        with pytest.raises(ValueError, match="boundary state"):
            prefix.followed_by(later)


def test_engineering_writer_cannot_relabel_a_branch_as_starting_at_zero(tmp_path):
    scene = following_scene(8, 30)
    with PufferDriveWorld(scene) as world:
        snapshot = world.snapshot()
        joint = {actor: 7 for actor in world.actor_ids}
        world.step(joint)
        branch = RolloutRecord.start(world)
        branch.append(world, joint, world.step(joint))
        with pytest.raises(ValueError, match="contiguous prefix"):
            write_rollout(scene, tmp_path / "misleading.npz", record=branch, snapshot=snapshot,
                          controlled_ids=world.actor_ids, policy="coast", controller_configuration={})
    assert not (tmp_path / "misleading.npz").exists()


def test_current_observation_does_not_read_log_future():
    first, second = following_scene(8, 30), following_scene(8, 30)
    second.poses[1:, :, :2] += 1000
    second.velocities[1:] = -500
    with PufferDriveWorld(first) as a, PufferDriveWorld(second) as b:
        np.testing.assert_array_equal(a.observe().metric.ego, b.observe().metric.ego)
        np.testing.assert_array_equal(a.observe().metric.neighbors, b.observe().metric.neighbors)
        np.testing.assert_array_equal(a.observe().features, b.observe().features)
        a.step({actor: 7 for actor in a.actor_ids})
        b.step({actor: 7 for actor in b.actor_ids})
        np.testing.assert_array_equal(a.state(), b.state())
