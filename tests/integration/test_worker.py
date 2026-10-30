"""An isolated process retains one world and rejects stale or invalid actions."""

import sys

import pytest

from driversi.contracts.road import RoadGraph
from driversi.sim.workers.worker import WorldWorker as WorkerService
from driversi.sim.workers.worker_client import WorldWorker

native = pytest.importorskip("driversi.sim._native")
pytestmark = pytest.mark.native


@pytest.fixture
def graph():
    return RoadGraph(
        {
            "schema_version": 1,
            "map_name": "straight",
            "map_version": None,
            "origin_xyz": [0, 0, 0],
            "bounds_local": [-40, -20, 60, 20],
        },
        {
            f"lane:{actor}": {
                "kind": "lane",
                "points": [[-40, y], [60, y]],
                "outgoing": [],
                "speed_limit_mps": None,
            }
            for actor, y in (("a", 0), ("b", 8))
        },
    )


@pytest.fixture
def worker(straight_scene, graph, tmp_path):
    scene_file, graph_file = tmp_path / "scene.npz", tmp_path / "graph.json"
    straight_scene.save(scene_file)
    graph.save(graph_file)
    with WorldWorker(
        python=sys.executable,
        scene=scene_file,
        road_graph=graph_file,
        native_path=native.__file__,
        timeout=15,
    ) as client:
        yield client


def assert_world_equal(left, right):
    for key in ("tick", "state", "motion", "alive", "done", "build_id"):
        assert left[key] == right[key], key


def test_subprocess_reset_replays_the_same_joint_transition(worker):
    initial = worker.current
    process_id = worker.process.pid
    assert initial["status"] == "ready"
    assert initial["all_actor_ids"] == ["a", "b", "replay"]
    assert initial["controlled_ids"] == ["a", "b"]
    assert initial["trainable"] is False
    first = worker.request("step", expected_tick=0)
    assert first["tick"] == 1
    assert first["valid"] == [True, True]
    assert set(first["events"]) == {"collision", "offroad", "goal"}
    assert_world_equal(initial, worker.request("reset"))
    repeated = worker.request("step", expected_tick=0)
    assert_world_equal(first, repeated)
    assert first["diagnostics"] == repeated["diagnostics"]
    assert first["policy_actions"] == first["executed_actions"] == repeated["executed_actions"]
    assert worker.process.pid == process_id
    response = worker.request("close")
    assert response["status"] == "ok"
    assert worker.process.wait(timeout=5) == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"expected_tick": 1},
        {"expected_tick": True},
        {"expected_tick": 0, "overrides": {"replay": 1}},
        {"expected_tick": 0, "overrides": {"a": 12}},
        {"expected_tick": 0, "overrides": {"a": True}},
        {"expected_tick": 0, "overrides": {"a": 1.0}},
    ],
)
def test_invalid_requests_do_not_advance_the_world(worker, arguments):
    initial = worker.current
    with pytest.raises(RuntimeError, match="Physics request failed"):
        worker.request("step", **arguments)
    assert_world_equal(initial, worker.request("observe"))
    assert worker.request("step", expected_tick=0)["tick"] == 1


def test_override_does_not_change_other_actors_same_tick_decisions(worker):
    baseline = worker.request("step", expected_tick=0)
    worker.request("reset")
    alternate = (baseline["policy_actions"]["a"] + 3) % 12
    overridden = worker.request("step", expected_tick=0, overrides={"a": alternate})
    assert overridden["policy_actions"] == baseline["policy_actions"]
    assert overridden["diagnostics"] == baseline["diagnostics"]
    assert overridden["executed_actions"] == {**baseline["policy_actions"], "a": alternate}
    assert overridden["state"][1:] == baseline["state"][1:]
    assert overridden["state"][0] != baseline["state"][0]


def test_failed_transition_restores_complete_world(straight_scene, graph, monkeypatch):
    service = WorkerService(straight_scene, graph)
    try:
        initial = service.world.snapshot()
        original_step = service.world.step

        def advance_then_fail(actions):
            original_step(actions)
            raise RuntimeError("simulated record failure after world advance")

        monkeypatch.setattr(service.world, "step", advance_then_fail)
        with pytest.raises(RuntimeError, match="simulated record failure"):
            service.handle({"request_id": 1, "operation": "step", "expected_tick": 0})
        assert service.world.snapshot() == initial
    finally:
        service.close()


def population_spec(external=False):
    return {
        "schema_version": 1,
        "policies": [
            {
                "policy_id": "rule",
                "version": "fixed-v1",
                "controller": "route-idm-v2",
                "parameters": {"desired_speed": 9},
            },
            {
                "policy_id": "other",
                "version": "candidate-v2",
                "controller": "external-joint-v1" if external else "route-idm-v2",
                "parameters": {} if external else {"desired_speed": 2},
            },
        ],
        "assignments": [
            {"actor_id": "a", "policy_id": "rule", "role": "reference_opponent"},
            {"actor_id": "b", "policy_id": "other", "role": "candidate"},
        ],
    }


def test_default_population_matches_prior_rule_action_path(straight_scene, graph):
    from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
    from driversi.agents.traffic_control.routing import resolve_routes

    service = WorkerService(straight_scene, graph, desired_speed=9)
    try:
        routes, _ = resolve_routes(straight_scene, graph)
        legacy = RouteFollower(routes, FollowingConfig(desired_speed=9))
        expected = legacy.commands(service.world.observe())
        reply = service.handle({"request_id": 1, "operation": "step", "expected_tick": 0})
        assert reply["diagnostics"] == expected
        assert reply["policy_actions"] == {item["actor"]: item["action"] for item in expected}
        assert reply["behavior_policies"] == service.ready()["behavior_policies"]
        assert reply["action_origins"] == {"a": "route-idm-v2", "b": "route-idm-v2"}
    finally:
        service.close()


def test_population_config_reaches_subprocess_and_reset(straight_scene, graph, tmp_path):
    import json

    scene_file, graph_file, config_file = (
        tmp_path / name for name in ("scene.npz", "graph.json", "population.json")
    )
    straight_scene.save(scene_file)
    graph.save(graph_file)
    config_file.write_text(json.dumps(population_spec()))
    with WorldWorker(
        python=sys.executable,
        scene=scene_file,
        road_graph=graph_file,
        native_path=native.__file__,
        population_config=config_file,
        timeout=15,
    ) as client:
        ready = client.current
        assert ready["population_configuration"]["schema_version"] == 1
        assert ready["behavior_policies"]["b"]["version"] == "candidate-v2"
        first = client.request("step", expected_tick=0)
        assert first["policy_actions"]["a"] != first["policy_actions"]["b"]
        # The source config is not reread within an episode, including reset.
        config_file.write_text("invalid after startup")
        reset = client.request("reset")
        assert reset["behavior_policies"] == ready["behavior_policies"]
        repeated = client.request("step", expected_tick=0)
        assert_world_equal(first, repeated)
        assert first["behavior_policies"] == repeated["behavior_policies"]
        assert first["policy_actions"] == repeated["policy_actions"]


@pytest.mark.parametrize(
    "external", [None, {}, {"a": 1}, {"b": True}, {"b": 1.0}, {"b": 12}, {"b": 7, "a": 7}]
)
def test_invalid_external_step_cannot_mutate_world(straight_scene, graph, external):
    from driversi.contracts.population import EpisodePopulation

    service = WorkerService(
        straight_scene, graph, population=EpisodePopulation.from_dict(population_spec(True))
    )
    try:
        initial = service.world.snapshot()
        request = {"request_id": 1, "operation": "step", "expected_tick": 0}
        if external is not None:
            request["external_actions"] = external
        with pytest.raises(ValueError):
            service.handle(request)
        assert service.world.snapshot() == initial
    finally:
        service.close()


def test_external_actions_overrides_and_identity_have_distinct_origins(straight_scene, graph):
    from driversi.contracts.population import EpisodePopulation

    service = WorkerService(
        straight_scene, graph, population=EpisodePopulation.from_dict(population_spec(True))
    )
    try:
        first = service.handle(
            {"request_id": 1, "operation": "step", "expected_tick": 0, "external_actions": {"b": 7}}
        )
        service.handle({"request_id": 2, "operation": "reset"})
        second = service.handle(
            {
                "request_id": 3,
                "operation": "step",
                "expected_tick": 0,
                "external_actions": {"b": 7},
                "overrides": {"b": 1},
            }
        )
        assert first["policy_actions"] == second["policy_actions"]
        assert first["executed_actions"]["b"] == 7
        assert second["executed_actions"]["b"] == 1
        assert first["action_origins"]["b"] == "external-joint-v1"
        assert second["action_origins"]["b"] == "override"
        assert second["override_actor_ids"] == ["b"]
        assert first["behavior_policies"] == second["behavior_policies"]
        assert second["behavior_policies"]["b"]["observation_version"] == "external-unverified"
        assert first["state"][0] == second["state"][0]
        assert first["state"][1] != second["state"][1]
        current = service.world.snapshot()
        with pytest.raises(ValueError, match="expected_tick"):
            service.handle(
                {
                    "request_id": 4,
                    "operation": "step",
                    "expected_tick": 0,
                    "external_actions": {"b": 7},
                }
            )
        assert service.world.snapshot() == current
    finally:
        service.close()


def test_snapshot_moves_between_independent_workers(straight_scene, graph, worker):
    for tick in range(3):
        worker.request("step", expected_tick=tick)
    checkpoint = worker.request("snapshot")["snapshot"]
    original = worker.request("step", expected_tick=3)
    service = WorkerService(straight_scene, graph)
    try:
        restored = service.handle(
            {"request_id": 1, "operation": "restore", "expected_tick": 0, "snapshot": checkpoint}
        )
        assert restored["tick"] == 3
        replayed = service.handle({"request_id": 2, "operation": "step", "expected_tick": 3})
        assert_world_equal(original, replayed)
        for key in (
            "policy_actions",
            "executed_actions",
            "diagnostics",
            "valid",
            "terminated",
            "truncated",
            "events",
        ):
            assert original[key] == replayed[key]
        assert worker.current["tick"] == 4
    finally:
        service.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tick", 90),
        ("build_id", "wrong"),
        ("native_state", "broken"),
        ("alive", [1, True]),
        ("actor_ids", ["b", "a", "replay"]),
        ("configuration", {}),
        ("episode_done", 1),
        ("version", "wrong"),
    ],
)
def test_bad_checkpoint_cannot_mutate_world(worker, field, value):
    checkpoint = worker.request("snapshot")["snapshot"]
    checkpoint[field] = value
    before = worker.request("observe")
    with pytest.raises(RuntimeError):
        worker.request("restore", expected_tick=0, snapshot=checkpoint)
    assert_world_equal(before, worker.request("observe"))


def test_restore_rejects_stale_expected_tick(worker):
    checkpoint = worker.request("snapshot")["snapshot"]
    worker.request("step", expected_tick=0)
    with pytest.raises(RuntimeError, match="expected_tick"):
        worker.request("restore", expected_tick=0, snapshot=checkpoint)
    assert worker.request("observe")["tick"] == 1


def native_population():
    from driversi.contracts.population import (
        EpisodePopulation,
        PopulationPolicy,
        ActorPolicyBinding,
    )

    return EpisodePopulation(
        (PopulationPolicy("native", "checkpoint-sha", "external-native-state-v1"),),
        tuple(ActorPolicyBinding(a, "native") for a in ("a", "b")),
    )


def native_request(service, tick=0):
    state = service.world.state()
    targets = {
        a: [
            float(state[i, 0] + 0.1),
            float(state[i, 1]),
            float(state[i, 2]),
            float(state[i, 3]),
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        for i, a in enumerate(service.world.actor_ids)
    }
    return dict(
        request_id=1,
        operation="native_step",
        expected_tick=tick,
        native_targets=targets,
        native_actions={a: [0.0, 0.0] for a in service.world.actor_ids},
    )


def test_native_port_requires_native_ownership(straight_scene, graph):
    s = WorkerService(straight_scene, graph)
    try:
        before = s.world.snapshot()
        with pytest.raises(ValueError, match="ownership"):
            s.handle(native_request(s))
        assert s.world.snapshot() == before
    finally:
        s.close()


def test_native_port_atomic_replay_preserves_state_and_events(straight_scene, graph, monkeypatch):
    s = WorkerService(straight_scene, graph, population=native_population())
    try:
        before = s.world.snapshot()
        request = native_request(s)
        first = s.handle(request)
        assert first["tick"] == 1 and first["action_space"] == "native_continuous_state"
        s.handle(dict(request_id=2, operation="reset"))
        again = s.handle(request)
        assert_world_equal(first, again)
        assert first["events"] == again["events"]
        s.handle(dict(request_id=2, operation="reset"))
        original = s.world.step_native

        def fail(*a, **kw):
            original(*a, **kw)
            raise RuntimeError("after native commit")

        monkeypatch.setattr(s.world, "step_native", fail)
        with pytest.raises(RuntimeError, match="after native"):
            s.handle(request)
        assert s.world.snapshot() == before
    finally:
        s.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda q: q.update(expected_tick=1),
        lambda q: q["native_targets"].pop("b"),
        lambda q: q["native_targets"]["a"].__setitem__(0, float("nan")),
        lambda q: q["native_actions"]["a"].__setitem__(0, True),
        lambda q: q.update(overrides={"a": 1}),
    ],
)
def test_bad_native_payload_leaves_world_unchanged(straight_scene, graph, mutation):
    s = WorkerService(straight_scene, graph, population=native_population())
    try:
        before = s.world.snapshot()
        request = native_request(s)
        mutation(request)
        with pytest.raises(ValueError):
            s.handle(request)
        assert s.world.snapshot() == before
    finally:
        s.close()
