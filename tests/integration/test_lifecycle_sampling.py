# ruff: noqa: E402
import argparse
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from tests.support.diagnostic_agents import ActorCritic
from driversi.data.datasets.world_model import export_sequences, iter_joint_sequences, load_sequence
from driversi.training.sampling.collector import Collector, rollout_nbytes
from driversi.training.ppo import train


def expanded(scene, count=40):
    scene = copy.deepcopy(scene)
    extra = count - 3
    scene.actor_ids += tuple(f"late-{i}" for i in range(extra))
    poses = np.zeros((31, extra, 4))
    poses[:, :, 0] = 100 + (np.arange(extra) % 80) * 2
    poses[:, :, 1] = 30 + (np.arange(extra) // 80) * 2
    scene.poses = np.concatenate((scene.poses, poses), axis=1)
    scene.velocities = np.concatenate((scene.velocities, np.zeros((31, extra, 3))), axis=1)
    scene.dimensions = np.concatenate((scene.dimensions, np.tile([0.6, 0.6, 1.7], (extra, 1))))
    scene.kinds = np.concatenate((scene.kinds, np.full(extra, 2)))
    scene.controlled = np.concatenate((scene.controlled, np.zeros(extra, bool)))
    scene.presence = np.ones((31, count), bool)
    scene.presence[1:3, -1] = False
    scene.poses[1:3, -1] = 0
    return scene


@pytest.mark.parametrize("count", [40, 576, 1024])
def test_full_roster_process_sampling_and_restore(straight_scene, tmp_path, count):
    path = tmp_path / "scene.npz"
    expanded(straight_scene, count).save(path)
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = ActorCritic()
    local = Collector([path], num_worlds=2)
    parallel = Collector([path], num_worlds=2, workers=1)
    try:
        checkpoint = parallel.snapshot()
        torch.manual_seed(13)
        a = local.collect(model, 5)
        torch.manual_seed(13)
        b = parallel.collect(model, 5)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        assert a["world_before"].shape == (5, 2, count, 13)
        assert not a["world_presence_after"][0, :, -1].any()
        assert a["world_presence_after"][2, :, -1].all()
        assert sum(x.nbytes for x in a.values()) == rollout_nbytes(5, 2, world_capacity=count)
        # Export actual sampled transitions without invoking a training optimiser.
        source = local.scenes[0]
        run = tmp_path / "sample-only"
        (run / "rollouts").mkdir(parents=True)
        metadata = {
            "joint_state_version": "puffer-state13-motion3-v1",
            "dt_seconds": .1,
            "policy_version": "frozen-diagnostic",
            "scene_paths": [str(path)],
            "actor_ids": [list(source.actor_ids)],
            "controlled_ids": [[a for a, flag in zip(source.actor_ids, source.controlled) if flag]],
        }
        np.savez_compressed(run / "rollouts/batch-00000.npz", metadata=json.dumps(metadata), **a)
        output = tmp_path / "world-model"
        manifest = export_sequences(run, output, context_steps=1, prediction_steps=2, stride=1)
        assert manifest["status"] == "complete"
        sequence = load_sequence(output, 0)
        assert sequence.states.shape == (4, count, 16)
        assert sequence.presence[:, -1].tolist() == [True, False, False, True]
        assert sequence.model_inputs()["presence_history"].shape == (2, count)
        assert "future_presence" not in sequence.model_inputs()
        parallel.restore(checkpoint)
        torch.manual_seed(13)
        c = parallel.collect(model, 5)
        for key in b:
            np.testing.assert_array_equal(b[key], c[key])
    finally:
        local.close()
        parallel.close()


def test_lifecycle_world_model_masks_survive_disk_and_do_not_leak_future(straight_scene, tmp_path):
    scene = expanded(straight_scene)
    scene.save(tmp_path / "train.npz")
    evaluation = copy.deepcopy(scene)
    evaluation.metadata["source_log"] = "other-log"
    evaluation.save(tmp_path / "eval.npz")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "train": [{"path": str(tmp_path / "train.npz")}],
                "evaluation": [{"path": str(tmp_path / "eval.npz")}],
                "limits": [],
            }
        )
    )
    train(
        argparse.Namespace(
            suite=suite,
            output_dir=tmp_path / "run",
            budget=24,
            rollout_steps=5,
            worlds=1,
            seed=5,
            device="cpu",
            resume=None,
        )
    )
    sequences = list(
        iter_joint_sequences(tmp_path / "run", context_steps=1, prediction_steps=2, stride=1)
    )
    first = sequences[0]
    assert first.presence[:, -1].tolist() == [True, False, False, True]
    assert first.model_inputs()["presence_history"].shape == (2, 40)
    assert "future_presence" not in first.model_inputs()
    assert first.training_targets()["future_presence"][:, -1].tolist() == [False, True]
    export_sequences(
        tmp_path / "run", tmp_path / "sequences", context_steps=1, prediction_steps=2, stride=1
    )
    disk = load_sequence(tmp_path / "sequences", 0)
    np.testing.assert_array_equal(disk.presence, first.presence)
