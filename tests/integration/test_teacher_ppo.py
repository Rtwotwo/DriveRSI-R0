# ruff: noqa: E402
# Optional Torch/native imports must follow dependency skips.
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from driversi.training.legacy.frozen import NpzPolicy as PrivilegedTeacher
from tests.support.diagnostic_agents import ActorCritic
from driversi.training.ppo import (
    Collector,
    restore_checkpoint,
    save_checkpoint,
    update,
)


def fixture_collector(straight_scene, tmp_path):
    path = tmp_path / "scene.npz"
    straight_scene.save(path)
    return Collector([path], num_worlds=2, seed=12)


def test_numpy_frozen_policy_matches_torch(straight_scene, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = ActorCritic()
    collector = fixture_collector(straight_scene, tmp_path)
    try:
        model.export(tmp_path / "model.npz", "test-v1")
        frozen = PrivilegedTeacher(tmp_path / "model.npz")
        x = collector.envs[0].observe()
        with torch.no_grad():
            logits, value = model(torch.from_numpy(x))
        a, b = frozen.predict(x)
        np.testing.assert_allclose(a, logits.numpy(), atol=1e-7)
        np.testing.assert_allclose(b, value.numpy(), atol=1e-6)
        np.testing.assert_array_equal(frozen.actions(x), logits.argmax(-1).numpy())
    finally:
        collector.close()


def test_real_optimizer_update_and_exact_checkpoint_continuation(straight_scene, tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(31)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    collector = fixture_collector(straight_scene, tmp_path)
    try:
        first = collector.collect(model, 4)
        weights = copy.deepcopy(model.state_dict())
        result = update(model, optimizer, first, epochs=1, minibatch=16)
        assert np.isfinite(list(result.values())).all()
        assert any(not torch.equal(weights[k], v) for k, v in model.state_dict().items())
        save_checkpoint(tmp_path / "state.pt", model, optimizer, collector, 1, int(first["valid"].sum()), {})
        expected = collector.collect(model, 35)  # includes finite horizon reset
        update(model, optimizer, expected, epochs=1, minibatch=16)
        expected_weights = copy.deepcopy(model.state_dict())
        assert restore_checkpoint(tmp_path / "state.pt", model, optimizer, collector, {})[0] == 1
        actual = collector.collect(model, 35)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        update(model, optimizer, actual, epochs=1, minibatch=16)
        for key, value in model.state_dict().items():
            assert torch.equal(value, expected_weights[key]), key
        assert actual["valid"].sum() < actual["valid"].size
        assert (actual["episode"] > 0).any()
        assert actual["world_after"].shape == (35, 2, 32, 13)
    finally:
        collector.close()


def test_checkpoint_mismatch_rejected(straight_scene, tmp_path):
    torch.set_num_threads(1)
    collector = fixture_collector(straight_scene, tmp_path)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters())
    try:
        save_checkpoint(tmp_path / "state.pt", model, optimizer, collector, 0, 0, {"seed": 1})
        with pytest.raises(ValueError, match="configuration"):
            restore_checkpoint(tmp_path / "state.pt", model, optimizer, collector, {"seed": 2})
    finally:
        collector.close()


def test_resumed_report_keeps_the_original_initial_network(straight_scene, tmp_path):
    import argparse
    import json

    from driversi.training.ppo import train

    train_scene = copy.deepcopy(straight_scene)
    eval_scene = copy.deepcopy(straight_scene)
    train_scene.metadata["source_log"] = "train-log"
    eval_scene.metadata["source_log"] = "eval-log"
    train_scene.save(tmp_path / "train.npz")
    eval_scene.save(tmp_path / "eval.npz")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"train": [{"path": str(tmp_path / "train.npz")}],
                                 "evaluation": [{"path": str(tmp_path / "eval.npz")}], "limits": []}))
    args = argparse.Namespace(suite=suite, output_dir=tmp_path / "run1", budget=12,
                              rollout_steps=4, worlds=2, seed=2, device="cpu", resume=None, agent_config=__import__("pathlib").Path(__file__).resolve().parents[2] / "tests/fixtures/legacy/agents/mlp.yaml")
    train(args)
    first = json.loads((args.output_dir / "report.json").read_text())
    args.resume = args.output_dir / "checkpoint.pt"
    args.output_dir = tmp_path / "run2"
    args.budget = 40
    train(args)
    second = json.loads((args.output_dir / "report.json").read_text())
    assert first["baseline"]["initial"] == second["baseline"]["initial"]
    assert first["evaluation"] == second["baseline"]["resume_start"]
    assert second["valid_transitions"] > first["valid_transitions"]


def test_audit_replays_real_behavior_and_rejects_corruption(straight_scene, tmp_path):
    import json

    from driversi.training.sampling.audit import audit_run

    torch.set_num_threads(1)
    torch.manual_seed(17)
    collector = fixture_collector(straight_scene, tmp_path)
    try:
        model = ActorCritic()
        batch = collector.collect(model, 4)
        count = int(batch["valid"].sum())
        root = tmp_path / "run"
        (root / "rollouts").mkdir(parents=True)
        model.export(root / "rollouts/policy-00000.npz", "audit-v1")
        path = root / "rollouts/batch-00000.npz"
        metadata = json.dumps({"policy_version": "audit-v1", "valid_transitions": count})
        np.savez_compressed(path, metadata=metadata, **batch)
        (root / "report.json").write_text(json.dumps({"resume_from": None, "valid_transitions": count}))
        result = audit_run(root)
        assert result["valid_transitions_in_this_run"] == count
        assert result["frozen_policy_logprob_reproduced"]
        index = tuple(np.argwhere(batch["valid"])[0])
        for field, value, reason in [
            ("log_probs", np.nan, "Nonfinite behavior"),
            ("log_probs", float(batch["log_probs"][index]) + .1, "cannot be reproduced"),
            ("actions", -1, "Invalid behavior action"),
        ]:
            corrupt = {k: v.copy() for k, v in batch.items()}
            corrupt[field][index] = value
            np.savez_compressed(path, metadata=metadata, **corrupt)
            with pytest.raises(ValueError, match=reason):
                audit_run(root)
    finally:
        collector.close()


def test_direct_evaluation_reports_unknown_training_sources(straight_scene, tmp_path):
    from driversi.evaluation.search.teacher import evaluate

    train = copy.deepcopy(straight_scene)
    train.metadata["source_log"] = "training-log"
    train.save(tmp_path / "train.npz")
    evaluation = copy.deepcopy(straight_scene)
    evaluation.metadata["source_log"] = "evaluation-log"
    evaluation.save(tmp_path / "eval.npz")
    unchecked = evaluate(None, [tmp_path / "eval.npz"])
    checked = evaluate(None, [tmp_path / "eval.npz"], training_paths=[tmp_path / "train.npz"])
    assert unchecked.pop("source_separation_checked") is False
    assert checked.pop("source_separation_checked") is True
    assert unchecked == checked


def test_budget_callback_preserves_collected_samples(straight_scene, tmp_path):
    torch.manual_seed(901)
    model = ActorCritic()
    first = fixture_collector(straight_scene, tmp_path)
    second = fixture_collector(straight_scene, tmp_path)
    charges = []
    try:
        rng = torch.get_rng_state()
        expected = first.collect(model, 8)
        torch.set_rng_state(rng)
        actual = second.collect(model, 8, before_step=charges.append)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        assert len(charges) == 8 and sum(charges) == int(expected["valid"].sum())
    finally:
        first.close()
        second.close()
