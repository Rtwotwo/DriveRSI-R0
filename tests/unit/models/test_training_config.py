import argparse
import json
from pathlib import Path

import pytest

from driversi.agents.interfaces.catalog import list_agents, resolve_agent
from tests.support.diagnostic_config import normalize_agent, read_agent_config
from driversi.configuration import read_mapping
from driversi.training.legacy.config import resolve_training

ROOT = Path(__file__).resolve().parents[3]


def args(**changes):
    data = dict(suite=Path("suite.json"), output_dir=Path("result"), resume=None)
    data.update(changes)
    return argparse.Namespace(**data)


def test_grouped_config_resolves_agent_relative_to_config_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    value = resolve_training(args(config=ROOT / "tests/fixtures/legacy/training/ppo.json", budget=31))
    assert value.agent == read_agent_config(ROOT / "tests/fixtures/legacy/agents/privileged_mlp.json")
    assert value.budget == 31 and value.rollout_steps == 32
    assert value.optimization == {"learning_rate": .0003, "epochs": 4, "minibatch_size": 512}


def test_agent_override_replaces_reference_without_loading_it(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"schema_version": 1, "trainer": "ppo", "agent": "missing.json"}))
    value = resolve_training(args(config=path, agent_config=ROOT / "tests/fixtures/legacy/agents/linear_actor_critic.json"))
    assert value.agent["model_target"].endswith(":LinearActorCritic")


@pytest.mark.parametrize("change", [{"unknown": 3}, {"trainer": "world_model"},
    {"optimization": {"gamma": .8}}, {"worlds": 0}, {"optimization": {"epochs": True}}])
def test_unused_or_invalid_training_options_are_rejected(tmp_path, change):
    value = {"schema_version": 1, "trainer": "ppo", "agent": str(ROOT / "tests/fixtures/legacy/agents/privileged_mlp.json")}
    value.update(change)
    path = tmp_path / "train.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        resolve_training(args(config=path))


@pytest.mark.parametrize("key,value", [("action_space", "continuous"), ("input_mode", "vision"),
                                       ("observation_version", "other"), ("task", "world_model")])
def test_incompatible_agent_contract_is_rejected(key, value):
    config = normalize_agent()
    config[key] = value
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_agent(config)


def test_yaml_and_dotted_target_normalize_to_existing_contract():
    value = read_agent_config(ROOT / "tests/fixtures/legacy/agents/mlp.yaml")
    assert value["model_target"] == "tests.support.diagnostic_agents:ActorCritic"
    assert value["parameters"] == {"hidden_sizes": [128, 128]}
    assert "_target_" not in value
    run = resolve_training(args(config=ROOT / "tests/fixtures/legacy/training/ppo.yaml"))
    assert run.agent == value


def test_named_selection_is_independent_of_cwd_and_overrides_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "training.yaml"
    path.write_text("schema_version: 1\ntrainer: ppo\nagent: missing.yaml\n")
    value = resolve_training(args(config=path, agent="linear", agent_dir=ROOT / "tests/fixtures/legacy/agents"))
    assert value.agent == read_agent_config(ROOT / "tests/fixtures/legacy/agents/linear.yaml")
    assert resolve_agent("mlp", ROOT / "tests/fixtures/legacy/agents") == ROOT / "tests/fixtures/legacy/agents/mlp.yaml"


def test_custom_directory_listing_does_not_import_agent(tmp_path):
    config = normalize_agent()
    config["model_target"] = "not_installed_but_configurable:Example"
    (tmp_path / "custom.json").write_text(json.dumps(config))
    result = list_agents(tmp_path)
    assert result == [{"name": "custom", "path": str(tmp_path / "custom.json"), "config": config}]
    run = resolve_training(args(agent="custom", agent_dir=tmp_path))
    assert run.agent == config


@pytest.mark.parametrize("name", ["../mlp", "mlp.yaml", "/mlp", "unknown"])
def test_bad_or_unknown_agent_name_is_rejected(name):
    with pytest.raises(ValueError):
        resolve_agent(name)


def test_duplicate_named_configs_are_rejected(tmp_path):
    for ext in ("json", "yaml"):
        (tmp_path / ("same." + ext)).write_text("{}")
    with pytest.raises(ValueError, match="Ambiguous"):
        list_agents(tmp_path)


@pytest.mark.parametrize("changes", [dict(agent="mlp", agent_config=Path("some.json")),
                                      dict(agent_dir=Path("somewhere"))])
def test_contradictory_selection_is_rejected(changes):
    with pytest.raises(ValueError):
        resolve_training(args(**changes))


@pytest.mark.parametrize("text,extension", [
    ('{"a": 1, "a": 2}', 'json'),
    ('a: 1\na: 2\n', 'yaml'),
    ('a: .nan\n', 'yaml'),
    ('a: ${oc.env:VALUE}\n', 'yaml'),
    ('a: !!python/object/apply:os.system ["echo invalid"]\n', 'yaml'),
    ('a: 2026-10-14\n', 'yaml'),
    ('- no mapping\n', 'yaml'),
])
def test_invalid_or_unsupported_config_syntax_is_rejected(tmp_path, text, extension):
    path = tmp_path / ("bad." + extension)
    path.write_text(text)
    with pytest.raises(ValueError):
        read_mapping(path)


def test_nested_construction_and_double_model_target_are_rejected():
    config = normalize_agent()
    config["parameters"] = {"nested": {"_target_": "module.Class"}}
    with pytest.raises(ValueError, match="Nested"):
        normalize_agent(config)
    config = normalize_agent()
    config["_target_"] = config["model_target"]
    with pytest.raises(ValueError, match="only one"):
        normalize_agent(config)


def test_agent_list_cli_does_not_import_torch():
    import os
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; from driversi.cli import main; main(['legacy-agents']); "
        "assert 'torch' not in sys.modules"], env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["agents"] == []


@pytest.mark.parametrize("sampling", [{"workers": -1}, {"workers": 5},
    {"scene_cache_size": 0}, {"max_rollout_mib": False}, {"worker_timeout_seconds": 0},
    {"unsupported": 1}])
def test_invalid_sampling_config_rejected(tmp_path, sampling):
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"schema_version": 1, "trainer": "ppo", "sampling": sampling,
                               "agent": str(ROOT / "tests/fixtures/legacy/agents/mlp.yaml")}))
    with pytest.raises(ValueError):
        resolve_training(args(config=path))


def test_sampling_cli_overrides_yaml():
    value = resolve_training(args(config=ROOT / "tests/fixtures/legacy/training/ppo.yaml", workers=2,
                                  scene_cache_size=1, max_rollout_mib=16, worker_timeout_seconds=5.))
    assert value.sampling == dict(workers=2, scene_cache_size=1,
                                  max_rollout_mib=16, worker_timeout_seconds=5.)


def test_rollout_inventory_rejects_outside_paths_and_conflicting_inline_sources(tmp_path):
    from driversi.data.scenes.scene_store import rollout_sources

    with pytest.raises(ValueError, match="outside"):
        rollout_sources({"source_manifest": "../outside.json"}, tmp_path, {})
    value = {"schema_version": 1, "kind": "rollout-scene-inventory", "scene_paths": ["a"],
             "actor_ids": [["a"]], "controlled_ids": [["a"]], "environment_specs": [{}]}
    (tmp_path / "inventory.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="disagree"):
        rollout_sources({"source_manifest": "inventory.json", "scene_paths": ["b"]}, tmp_path, {})
    assert rollout_sources({"scene_paths": ["legacy"]}, tmp_path, {})["scene_paths"] == ["legacy"]


def test_opponent_manifest_path_resolution(tmp_path):
    agent = str(ROOT / "tests/fixtures/legacy/agents/mlp.yaml")
    path=tmp_path/"training.json"
    path.write_text(json.dumps(dict(schema_version=1,trainer="ppo",agent=agent,
                                   opponent_pool="history.json",learner_count=2)))
    result=resolve_training(args(config=path))
    assert result.opponent_pool == tmp_path/"history.json" and result.learner_count == 2
    assert resolve_training(args(agent_config=ROOT / "tests/fixtures/legacy/agents/mlp.yaml")).opponent_pool is None


@pytest.mark.parametrize("options", [dict(learner_count=0),dict(learner_count=True),
    dict(learner_count=32),dict(learner_count=2)])
def test_invalid_or_unbound_role_count_rejected(options):
    with pytest.raises(ValueError):
        resolve_training(args(**options))


@pytest.mark.parametrize("adaptive", [False, {"unknown": 1}, {"interval_updates": 0},
    {"min_actor_episodes": True}, {"max_world_steps": 0}, {"seed": -1},
    {"uniform_mix": 0}, {"power": 9}])
def test_invalid_adaptive_configuration(tmp_path, adaptive):
    path = tmp_path / "training.json"
    path.write_text(json.dumps(dict(schema_version=1, trainer="ppo", agent=str(ROOT / "tests/fixtures/legacy/agents/mlp.yaml"),
                                   opponent_pool="pool.json", adaptive_opponents=adaptive)))
    with pytest.raises(ValueError):
        resolve_training(args(config=path))


def test_adaptive_requires_pool_rejects_ddp_and_defaults_off(tmp_path):
    path = tmp_path / "training.json"
    raw = dict(schema_version=1, trainer="ppo", agent=str(ROOT / "tests/fixtures/legacy/agents/mlp.yaml"), adaptive_opponents={})
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="require a pool"):
        resolve_training(args(config=path))
    raw["opponent_pool"] = "pool.json"
    path.write_text(json.dumps(raw))
    assert resolve_training(args(config=path)).adaptive_opponents["interval_updates"] == 10
    with pytest.raises(ValueError, match="DDP"):
        resolve_training(args(config=path, distributed=True))
    assert resolve_training(args(agent_config=ROOT / "tests/fixtures/legacy/agents/mlp.yaml")).adaptive_opponents is None
