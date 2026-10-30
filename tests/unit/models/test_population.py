"""Population contracts: frozen identities and same-time grouped dispatch."""

import copy
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
from driversi.agents.traffic_control.population import PopulationController
from driversi.contracts.population import EpisodePopulation
from driversi.contracts.road import Route
from driversi.contracts.transition import MetricObservation, Observation


def specification(external=False):
    return {"schema_version": 1,
            "policies": [{"policy_id": "fast", "version": "v1", "controller": "route-idm-v2",
                          "parameters": {"desired_speed": 10}},
                         {"policy_id": "slow", "version": "v2",
                          "controller": "external-joint-v1" if external else "route-idm-v2",
                          "parameters": {} if external else {"desired_speed": 2}}],
            "assignments": [{"actor_id": "a", "policy_id": "fast", "role": "candidate"},
                            {"actor_id": "b", "policy_id": "slow", "role": "reference_opponent"}]}


def routes():
    return {actor: Route([[0, y], [100, y]]) for actor, y in (("a", 0), ("b", 8))}


def observation():
    return Observation(("a", "b"), np.zeros((2, 1124)), np.ones(2, bool), 11,
                       MetricObservation(np.asarray([[0, y, 0, 4, 0, 4, 2, 0, 0, 0]
                                                     for y in (0, 8)], dtype=float),
                                         np.zeros((2, 0, 7)), np.zeros((2, 0), bool), ((), ())))


def test_contract_owns_values_and_canonicalizes_order():
    source = specification()
    population = EpisodePopulation.from_dict(source)
    original = population.to_dict()
    source["policies"][0]["parameters"]["desired_speed"] = 99
    source["assignments"][0]["role"] = "participant"
    detached = population.to_dict()
    detached["policies"][0]["parameters"]["desired_speed"] = 77
    assert population.to_dict() == original
    reversed_source = copy.deepcopy(original)
    reversed_source["policies"].reverse()
    reversed_source["assignments"].reverse()
    assert EpisodePopulation.from_dict(reversed_source) == population
    with pytest.raises(FrozenInstanceError):
        population.policies = ()
    assert population.behavior_policies["a"]["role"] == "candidate"


@pytest.mark.parametrize("kind", ["duplicate_policy", "duplicate_actor", "unknown_policy",
                                   "unknown_controller", "unknown_role", "bad_version",
                                   "bool_parameter", "nonfinite_parameter", "extra_field",
                                   "trainable", "wrong_observation", "bad_schema"])
def test_rejects_unsupported_population_declarations(kind):
    payload = specification()
    if kind == "duplicate_policy":
        payload["policies"].append(payload["policies"][0])
    elif kind == "duplicate_actor":
        payload["assignments"].append(payload["assignments"][0])
    elif kind == "unknown_policy":
        payload["assignments"][0]["policy_id"] = "missing"
    elif kind == "unknown_controller":
        payload["policies"][0]["controller"] = "python:load_anything"
    elif kind == "unknown_role":
        payload["assignments"][0]["role"] = "trained"
    elif kind == "bad_version":
        payload["policies"][0]["version"] = ""
    elif kind == "bool_parameter":
        payload["policies"][0]["parameters"]["desired_speed"] = True
    elif kind == "nonfinite_parameter":
        payload["policies"][0]["parameters"]["desired_speed"] = float("nan")
    elif kind == "extra_field":
        payload["policies"][0]["code"] = "anything"
    elif kind == "trainable":
        payload["policies"][0]["trainable"] = True
    elif kind == "wrong_observation":
        payload["policies"][0]["observation_version"] = "rgb"
    else:
        payload["schema_version"] = True
    with pytest.raises(ValueError):
        EpisodePopulation.from_dict(payload)


@pytest.mark.parametrize("actors", [("a",), ("a", "b", "replay"), ("a", "a"), ("b", "wrong")])
def test_assignment_covers_exact_controlled_population(actors):
    with pytest.raises(ValueError, match="exactly every controlled"):
        EpisodePopulation.from_dict(specification()).validate_actors(actors)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "population.json"
    path.write_text('{"schema_version":1,"schema_version":1,"policies":[],"assignments":[]}')
    with pytest.raises(ValueError, match="Duplicate"):
        EpisodePopulation.load(path)


def test_default_controller_preserves_rule_diagnostics_and_actions():
    controller = PopulationController(routes(), desired_speed=9)
    legacy = RouteFollower(routes(), FollowingConfig(desired_speed=9))
    assert controller.commands(observation()) == legacy.commands(observation())
    assert controller.population.policies[0].to_dict()["parameters"] == asdict(legacy.config)


def test_distinct_policies_use_own_parameters_in_same_world():
    controller = PopulationController(routes(), EpisodePopulation.from_dict(specification()))
    commands = {item["actor"]: item for item in controller.commands(observation())}
    assert commands["a"]["desired_acceleration"] > 0
    assert commands["b"]["desired_acceleration"] < 0
    assert commands["a"]["action"] != commands["b"]["action"]
    assert controller.behavior_policies["a"]["version"] == "v1"
    assert controller.behavior_policies["b"]["version"] == "v2"


def test_group_calls_receive_copies_of_one_common_observation(monkeypatch):
    controller = PopulationController(routes(), EpisodePopulation.from_dict(specification()))
    seen = []
    def commands(part):
        seen.append((part.step, part.actor_ids, part.features.copy(), part.metric.ego.copy()))
        part.features[:] = 100
        part.metric.ego[:] = 200
        return [{"actor": actor, "action": 7} for actor in part.actor_ids]
    for rule in controller._controllers.values():
        monkeypatch.setattr(rule, "commands", commands)
    current = observation()
    before = copy.deepcopy(current)
    assert controller.commands(current) == [{"actor": "a", "action": 7}, {"actor": "b", "action": 7}]
    assert [item[0] for item in seen] == [11, 11]
    assert all(not item[2].any() for item in seen)
    np.testing.assert_array_equal(current.features, before.features)
    np.testing.assert_array_equal(current.metric.ego, before.metric.ego)


def test_external_actions_do_not_claim_verified_observation_or_learning():
    controller = PopulationController(routes(), EpisodePopulation.from_dict(specification(external=True)))
    commands = controller.commands(observation(), {"b": 4})
    assert commands[1] == {"actor": "b", "action": 4, "external_observation_verified": False}
    assert controller.external_actor_ids == ("b",)
    assert controller.population.external_actor_ids == ("b",)
    assert controller.behavior_policies["b"]["observation_version"] == "external-unverified"
    assert not controller.behavior_policies["b"]["trainable"]


@pytest.mark.parametrize("external", [None, {}, {"a": 7}, {"b": 7, "a": 1}, {"b": True},
                                      {"b": 2.0}, {"b": 12}, {"b": -1}, []])
def test_invalid_external_actions_rejected_before_rule_calls(external, monkeypatch):
    controller = PopulationController(routes(), EpisodePopulation.from_dict(specification(external=True)))
    def must_not_run(_):
        pytest.fail("Invalid external joint action reached rule inference")
    monkeypatch.setattr(controller._controllers["fast"], "commands", must_not_run)
    with pytest.raises(ValueError):
        controller.commands(observation(), external)


def test_external_policy_rejects_implicit_parameters():
    payload = specification(external=True)
    payload["policies"][1]["parameters"] = {"desired_speed": 3}
    with pytest.raises(ValueError, match="External policies"):
        EpisodePopulation.from_dict(payload)


def test_unknown_rule_parameter_fails_before_world_creation():
    payload = specification()
    payload["policies"][0]["parameters"] = {"hidden_future": 3}
    with pytest.raises(ValueError, match="Unknown route"):
        PopulationController(routes(), EpisodePopulation.from_dict(payload))


def test_shared_policy_is_called_once_for_its_batch(monkeypatch):
    controller = PopulationController(routes())
    original = controller._controllers["default"].commands
    batches = []
    def recording(part):
        batches.append((part.actor_ids, part.step))
        return original(part)
    monkeypatch.setattr(controller._controllers["default"], "commands", recording)
    assert len(controller.commands(observation())) == 2
    assert batches == [(("a", "b"), 11)]
