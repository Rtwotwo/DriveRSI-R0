# ruff: noqa: E402
import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from tests.support.diagnostic_agents import ActorCritic
from tests.support.diagnostic_config import build_agent, load_inference, normalize_agent, read_agent_config
from driversi.training.legacy.frozen import NpzPolicy as PrivilegedTeacher
from driversi.cli import main
from driversi.training.sampling.audit import audit_run

ROOT = Path(__file__).resolve().parents[2]


def test_default_factory_preserves_weights_and_rng():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    old = ActorCritic()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(37)
    new = build_agent()
    for name, weights in old.state_dict().items():
        assert torch.equal(weights, new.state_dict()[name])
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_custom_agent_needs_only_module_and_config(monkeypatch, tmp_path):
    (tmp_path / "example_agent.py").write_text(
        "from tests.support.diagnostic_agents import ActorCritic\n"
        "class CustomAgent(ActorCritic):\n"
        "    def __init__(self):\n"
        "        super().__init__(hidden_sizes=(16, 8))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    config = normalize_agent()
    config.update(model_target="example_agent:CustomAgent", parameters={})
    model = build_agent(config)
    model.export(tmp_path / "frozen.npz", "custom-v1")
    frozen = load_inference(tmp_path / "frozen.npz", config)
    x = np.random.default_rng(3).normal(size=(3, 1125)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x))
    actual = frozen.predict(x)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b.numpy(), atol=2e-6)


@pytest.mark.parametrize("agent_name", ["privileged_mlp", "linear_actor_critic", "mlp", "linear", "custom"])
def test_cli_config_drives_real_update_export_audit_and_resume(straight_scene, tmp_path, monkeypatch, agent_name):
    rows = []
    for name in ("train", "evaluation"):
        scene = copy.deepcopy(straight_scene)
        scene.metadata["source_log"] = name + "-unique-log"
        path = tmp_path / (name + ".npz")
        scene.save(path)
        rows.append({"path": str(path)})
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"train": rows[:1], "evaluation": rows[1:], "limits": []}))
    config = json.loads((ROOT / "tests/fixtures/legacy/training/ppo.json").read_text())
    config["agent"] = str(ROOT / "tests/fixtures/legacy/agents/privileged_mlp.json")
    config["optimization"] = {"learning_rate": .001, "epochs": 1, "minibatch_size": 8}
    run_config = tmp_path / "training.json"
    run_config.write_text(json.dumps(config))
    if agent_name in ("privileged_mlp", "linear_actor_critic"):
        agent_path = ROOT / f"tests/fixtures/legacy/agents/{agent_name}.json"
        selector = ["--agent-config", str(agent_path)]
    else:
        import yaml

        # A config-selected model package requires no trainer or registry edits.
        config["agent"] = "replaced-missing-config.yaml"
        run_config = tmp_path / "training.yaml"
        run_config.write_text(yaml.safe_dump(config))
        selector = ["--agent", agent_name]
        if agent_name != "custom":
            selector += ["--agent-dir", str(ROOT / "tests/fixtures/legacy/agents")]
        agent_path = ROOT / f"tests/fixtures/legacy/agents/{agent_name}.yaml"
        if agent_name == "custom":
            package = tmp_path / "custom_model"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "agent.py").write_text(
                "from tests.support.diagnostic_agents import ActorCritic\n"
                "class CustomAgent(ActorCritic):\n"
                "    def __init__(self, width):\n"
                "        super().__init__(hidden_sizes=(width,))\n")
            directory = tmp_path / "agent_configs"
            directory.mkdir()
            agent_path = directory / "custom.yaml"
            custom = read_agent_config(ROOT / "tests/fixtures/legacy/agents/mlp.yaml")
            custom["_target_"] = "custom_model.agent.CustomAgent"
            custom.pop("model_target")
            custom["parameters"] = {"width": 16}
            agent_path.write_text(yaml.safe_dump(custom))
            selector += ["--agent-dir", str(directory)]
            monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    common = ["legacy-train", "--config", str(run_config), *selector,
              "--suite", str(suite), "--rollout-steps", "4", "--worlds", "2", "--seed", "9"]
    if agent_name == "linear":
        common += ["--workers", "2", "--scene-cache-size", "1"]
    assert main([*common, "--output-dir", str(out), "--budget", "32"]) == 0
    effective = json.loads((out / "config.json").read_text())
    assert effective["agent"] == read_agent_config(agent_path)
    assert effective["sampling"]["workers"] == (2 if agent_name == "linear" else 0)
    inventory = json.loads((out / "scene-inventory.json").read_text())
    assert inventory["scene_paths"] == [rows[0]["path"]]
    with np.load(next((out / "rollouts").glob("batch-*.npz"))) as batch:
        metadata = json.loads(str(batch["metadata"]))
        assert metadata["source_manifest"] == "scene-inventory.json"
        assert "scene_paths" not in metadata
    assert effective["lr"] == .001 and effective["epochs"] == 1 and effective["minibatch_size"] == 8
    with np.load(out / "initial.npz") as a, np.load(out / "candidate.npz") as b:
        assert any(not np.array_equal(a[key], b[key]) for key in a.files if key != "metadata")
        architecture = json.loads(str(b["metadata"]))["architecture"]
        assert architecture["hidden_sizes"] == (
            [16] if agent_name == "custom" else [128, 128] if agent_name in ("privileged_mlp", "mlp") else [])
    assert audit_run(out)["frozen_policy_logprob_reproduced"]
    assert main([*common, "--output-dir", str(tmp_path / "resume"), "--budget", "48",
                 "--resume", str(out / "checkpoint.pt")]) == 0
    report = json.loads((tmp_path / "resume/report.json").read_text())
    assert report["valid_transitions"] >= 48


def test_legacy_frozen_model_loads_without_architecture(tmp_path):
    model = ActorCritic()
    path = tmp_path / "model.npz"
    model.export(path, "legacy-v1")
    with np.load(path) as data:
        values = {k: data[k] for k in data.files}
    metadata = json.loads(str(values["metadata"]))
    metadata.pop("architecture")
    values["metadata"] = json.dumps(metadata)
    np.savez(path, **values)
    assert PrivilegedTeacher(path).weights["trunk.0.weight"].shape == (128, 1125)
    metadata["architecture"] = {"version": 1, "hidden_sizes": [64], "activation": "tanh"}
    values["metadata"] = json.dumps(metadata)
    np.savez(path, **values)
    with pytest.raises(ValueError, match="weights"):
        PrivilegedTeacher(path)


def test_metrics_recording_keeps_training_identical_without_claiming_rollout_dataset(straight_scene, tmp_path):
    paths = []
    for name in ("train-metrics", "eval-metrics"):
        scene = copy.deepcopy(straight_scene)
        scene.metadata["source_log"] = name
        path = tmp_path / (name + ".npz")
        scene.save(path)
        paths.append(path)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"train": [{"path": str(paths[0])}],
                                 "evaluation": [{"path": str(paths[1])}], "limits": []}))
    common = ["legacy-train", "--agent-dir", str(ROOT / "tests/fixtures/legacy/agents"), "--agent", "linear", "--suite", str(suite), "--worlds", "2",
              "--workers", "1", "--rollout-steps", "4", "--budget", "16", "--seed", "73"]
    for mode in ("full", "metrics"):
        out = tmp_path / mode
        assert main([*common, "--output-dir", str(out), "--recording", mode]) == 0
        report = json.loads((out / "report.json").read_text())
        assert report["rollout_data_available"] == (mode == "full")
        assert bool(list((out / "rollouts").glob("*.npz"))) == (mode == "full")
        assert (out / "checkpoint.pt").is_file()
    with np.load(tmp_path / "full/candidate.npz") as a, np.load(tmp_path / "metrics/candidate.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    from driversi.data.datasets.world_model import iter_joint_sequences
    with pytest.raises(ValueError, match="No actual rollout batches"):
        next(iter_joint_sequences(tmp_path / "metrics"))
