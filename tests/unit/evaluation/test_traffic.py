import numpy as np
import pytest

from driversi.agents.traffic_control.following import RouteFollower
from driversi.agents.traffic_control.population import PopulationController
from driversi.agents.traffic_control.traffic import TrafficConfig, TrafficController, route_crossing, swept_contact
from driversi.contracts.population import ActorPolicyBinding, EpisodePopulation, PopulationPolicy
from driversi.contracts.road import Route
from driversi.contracts.transition import MetricObservation, Observation


def crossing_observation():
    ids = ("a", "b", "external")
    states = np.array(
        [
            [-18, 0, 0, 6, 0, 4, 2, 0, 0, 0],
            [0, -18, np.pi / 2, 0, 6, 4, 2, 0, 0, 0],
            [-18, 45, 0, 6, 0, 4, 2, 0, 0, 0],
        ],
        float,
    )
    others = [[j for j in range(3) if j != i] for i in range(3)]
    return Observation(
        ids,
        np.zeros((3, 1124)),
        np.ones(3, bool),
        0,
        MetricObservation(
            states,
            np.array([states[js, :7] for js in others]),
            np.ones((3, 2), bool),
            tuple(tuple(ids[j] for j in js) for js in others),
        ),
    )


def routes():
    return {
        "a": Route([[-40, 0], [100, 0]]),
        "b": Route([[0, -40], [0, 100]]),
        "external": Route([[-40, 45], [100, 45]]),
    }


def test_swept_contact_catches_between_ticks_and_rejects_separated():
    a = np.array([0, 0, 0, 100, 0, 0.2, 0.2], float)
    b = np.array([5, 0, 0, 0, 0, 0.2, 0.2], float)
    assert swept_contact(a, b, a[3:5], 0.1) == pytest.approx(0.048)
    b[1] = 1
    assert swept_contact(a, b, a[3:5], 0.1) is None


def test_crossing_priority_and_external_no_yield_promise():
    obs = crossing_observation()
    policy = TrafficController(routes(), cooperative_ids=("a", "b"))
    commands = {c["actor"]: c for c in policy.commands(obs)}
    assert commands["b"]["traffic_priority"] == "yield"
    assert commands["b"]["desired_acceleration"] < commands["a"]["desired_acceleration"]
    uncooperative = TrafficController(routes()).commands(obs)
    assert uncooperative[0]["traffic_hazards"][0]["reason"] == "swept_box_conflict"
    assert uncooperative[0]["traffic_priority"] == "none"


def test_actor_order_same_snapshot_and_absent_neighbor():
    obs = crossing_observation()
    policy = TrafficController(routes(), cooperative_ids=("a", "b"))
    before = obs.metric.ego.copy()
    first = {c["actor"]: c for c in policy.commands(obs)}
    indices = [2, 1, 0]
    reverse = Observation(
        tuple(obs.actor_ids[i] for i in indices),
        obs.features[indices],
        obs.alive[indices],
        obs.step,
        obs.metric.select(indices),
    )
    assert {c["actor"]: c for c in policy.commands(reverse)} == first
    np.testing.assert_array_equal(before, obs.metric.ego)
    obs.metric.visible[1, 0] = False
    assert policy.commands(obs)[1]["traffic_priority"] == "none"


def test_grouped_population_preserves_external_and_old_default():
    rs, obs = routes(), crossing_observation()
    pop = EpisodePopulation(
        (
            PopulationPolicy("x", "x1", "route-conflict-v1"),
            PopulationPolicy("y", "y1", "route-conflict-v1", (("desired_speed", 7.0),)),
            PopulationPolicy("e", "e1", "external-joint-v1"),
        ),
        (
            ActorPolicyBinding("a", "x"),
            ActorPolicyBinding("b", "y"),
            ActorPolicyBinding("external", "e"),
        ),
    )
    controller = PopulationController(rs, pop)
    result = controller.commands(obs, {"external": 9})
    assert result[1]["traffic_priority"] == "yield"
    assert result[2]["action"] == 9 and "traffic_hazards" not in result[2]
    assert PopulationController(rs).commands(obs) == RouteFollower(rs).commands(obs)
    restored = PopulationController(rs, EpisodePopulation.from_dict(pop.to_dict()))
    assert restored.commands(obs, {"external": 9}) == result


def test_occupied_crossing_priority_and_parallel_paths():
    obs = crossing_observation()
    a, b = obs.metric.ego[:2]
    b[1] = 0
    assert route_crossing(routes()["a"], a, routes()["b"], b, 40)[1] == pytest.approx(0)
    assert route_crossing(routes()["a"], a, routes()["external"], obs.metric.ego[2], 40) is None
    obs.metric.neighbors[0, 0, 1] = 0
    commands = TrafficController(routes(), cooperative_ids=("a", "b")).commands(obs)
    assert commands[0]["traffic_priority"] == "yield"


def test_limits_and_dead_actors():
    with pytest.raises(ValueError):
        TrafficConfig(horizon_seconds=100)
    obs = crossing_observation()
    obs.alive[1] = False
    assert TrafficController(routes(), cooperative_ids=("a", "b")).commands(obs)[1]["action"] == 7


def test_full_crossing_avoids_contact_and_both_cars_progress():
    from driversi.evaluation.search.traffic import advance, crossing_scene, metrics, population
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    scene, rs = crossing_scene()
    with PufferDriveWorld(scene) as world:
        old = advance(world, RouteFollower(rs), 90)
        assert metrics(scene, old, rs, "b")["collision_events"] > 0
        world.reset()
        root = world.snapshot()
        new = advance(world, population(rs), 90)
        summary = metrics(scene, new, rs, "b")
        assert (
            summary["collision_events"]
            == summary["offroad_events"]
            == summary["contact_ticks"]
            == 0
        )
        assert new.states[-1][0, 0] > 10 and new.states[-1][1, 1] > 10
        world.restore(root)
        repeated = advance(world, population(rs), 90)
        np.testing.assert_array_equal(new.states, repeated.states)
        np.testing.assert_array_equal(new.actions, repeated.actions)


def test_search_forks_reactions_budget_trust_and_repeat(tmp_path):
    from driversi.evaluation.search.traffic import (
        AnchorSearchConfig,
        advance,
        crossing_scene,
        population,
        search,
    )
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    scene, rs = crossing_scene()
    cfg = AnchorSearchConfig(
        steps=40, anchor_speeds=(4.0, 8.0), refinements=(0.5,), max_attempts=4, max_agent_steps=480
    )
    with PufferDriveWorld(scene) as world:
        parent = world.snapshot()
        report, records = search(world, rs, "a", tmp_path / "search", cfg)
        assert report["status"] == "completed" and report["agent_steps"] == 480
        assert world.snapshot() == parent and len(records) == 4
        assert report["selected"] is not None
        states = [np.asarray(r.states) for r in records.values()]
        assert np.any(abs(states[0][:, 1, :2] - states[2][:, 1, :2]) > 0.001)
        repeated, _ = search(world, rs, "a", tmp_path / "repeat", cfg)
        assert report == repeated
        short, _ = search(
            world, rs, "a", tmp_path / "budget", AnchorSearchConfig(steps=40, max_agent_steps=119)
        )
        assert short["status"] == "budget_exhausted" and short["agent_steps"] == 0
        strict, _ = search(
            world,
            rs,
            "a",
            tmp_path / "strict",
            AnchorSearchConfig(
                steps=40,
                anchor_speeds=(8.0,),
                refinements=(0.5,),
                max_deviation_m=0.00001,
                max_attempts=2,
                max_agent_steps=240,
            ),
        )
        assert strict["attempts"][1]["trust_region_pass"] is False
        assert strict["attempts"][1]["accepted"] is False
        advance(world, population(rs), 80)
        with pytest.raises(ValueError, match="remaining"):
            search(world, rs, "a", tmp_path / "late", cfg)


def test_nonzero_search_requires_and_preserves_actual_prefix(tmp_path):
    from driversi.evaluation.search.traffic import (
        AnchorSearchConfig,
        advance,
        crossing_scene,
        population,
        search,
    )
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    scene, rs = crossing_scene()
    with PufferDriveWorld(scene) as world:
        prefix = advance(world, population(rs), 10)
        cfg = AnchorSearchConfig(steps=40, anchor_speeds=(8.0,), refinements=(), max_attempts=1)
        with pytest.raises(ValueError, match="actual contiguous prefix"):
            search(world, rs, "a", tmp_path / "missing", cfg)
        assert not (tmp_path / "missing").exists()
        report, records = search(world, rs, "a", tmp_path / "good", cfg, prefix=prefix)
        assert report["anchor_tick"] == 10 and report["agent_steps"] == 120
        path = tmp_path / "good" / f"{report['attempts'][0]['id']}.npz"
        with np.load(path) as data:
            assert data["states"].shape[0] == 51
            np.testing.assert_array_equal(data["states"][:11], prefix.states)
            np.testing.assert_array_equal(data["states"][10:], next(iter(records.values())).states)
        prefix.states[-1][0, 0] += 1
        with pytest.raises(ValueError, match="boundary state"):
            search(world, rs, "a", tmp_path / "bad", cfg, prefix=prefix)
        assert not (tmp_path / "bad").exists()


def test_crossing_paired_causality_unrelated_and_action_replay():
    from driversi.evaluation.benchmarks.interaction import _advance
    from driversi.evaluation.search.traffic import advance, crossing_scene, population
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    scene, rs = crossing_scene()
    with PufferDriveWorld(scene) as world:
        advance(world, population(rs), 10)
        anchor = world.snapshot()
        baseline, _ = _advance(world, population(rs), steps=80)
        world.restore(anchor)
        changed, _ = _advance(world, population(rs), steps=80, lead="a")
        world.restore(anchor)
        irrelevant, _ = _advance(world, population(rs), steps=80, lead="far")
        world.restore(anchor)
        replay, _ = _advance(
            world, population(rs), steps=80, lead="a", observer="b", replay=baseline
        )
    first = np.flatnonzero(np.asarray(baseline.actions)[:, 1] != np.asarray(changed.actions)[:, 1])
    assert len(first) and first[0] > 0
    assert (
        baseline.diagnostics[0][1]["desired_acceleration"]
        == changed.diagnostics[0][1]["desired_acceleration"]
    )
    np.testing.assert_array_equal(
        np.asarray(baseline.states)[:, 1], np.asarray(irrelevant.states)[:, 1]
    )
    np.testing.assert_array_equal(
        np.asarray(baseline.actions)[:, 1], np.asarray(replay.actions)[:, 1]
    )
    assert np.any(np.asarray(changed.states)[:, 1] != np.asarray(replay.states)[:, 1])


def test_mid_candidate_wall_budget_keeps_partial_evidence(tmp_path, monkeypatch):
    import driversi.evaluation.search.traffic as traffic
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    scene, rs = traffic.crossing_scene()
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 5 else 2.0

    monkeypatch.setattr(traffic.time, "monotonic", clock)
    with PufferDriveWorld(scene) as world:
        before = world.snapshot()
        report, records = traffic.search(
            world,
            rs,
            "a",
            tmp_path / "partial",
            traffic.AnchorSearchConfig(max_wall_seconds=1.0, anchor_speeds=(8.0,), refinements=()),
        )
        assert world.snapshot() == before
    attempt = report["attempts"][0]
    assert report["status"] == "budget_exhausted" and report["selected"] is None
    assert attempt["accepted"] is False and attempt["metrics"]["executed_steps"] == 3
    assert report["agent_steps"] == 9
    assert (tmp_path / "partial" / attempt["partial_rollout"]).exists()
    with np.load(tmp_path / "partial" / attempt["partial_rollout"]) as data:
        np.testing.assert_array_equal(data["actions"], records[attempt["id"]].actions)


def test_actuator_diagnostics_describe_final_conflict_action():
    from driversi.agents.traffic_control.following import _actuator_choice

    obs = crossing_observation()
    commands = TrafficController(routes(), cooperative_ids=("a", "b")).commands(obs)
    changed = 0
    for i, command in enumerate(commands):
        if not command["traffic_modified"]:
            continue
        changed += 1
        e = obs.metric.ego[i]
        action, diagnostics = _actuator_choice(
            command["desired_acceleration"],
            command["desired_lateral_acceleration"],
            e[7:9],
            command["actual_speed_mps"],
            steering=e[9],
            wheelbase=0.6 * e[5],
        )
        assert command["action"] == action
        assert {key: command[key] for key in diagnostics} == diagnostics
    assert changed > 0


def test_other_policy_collisions_are_not_selected_owner_failures():
    from driversi.evaluation.search.traffic import advance, crossing_scene, metrics, population
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    scene, rs = crossing_scene()
    scene.poses[:, 1, :2] = [0, 40]
    scene.poses[:, 2, :2] = [0, 40]
    scene.velocities[:, 1:] = 0
    with PufferDriveWorld(scene) as world:
        record = advance(world, population(rs), 3)
    m = metrics(scene, record, rs, "a")
    assert m["contact_ticks"] > 0  # Initial overlaps are not necessarily new native events.
    assert m["owner_collision_events"] == m["owner_contact_ticks"] == 0
    assert m["other_contact_ticks"] > 0 and m["official_score"] is None
    assert not m["trainable"]
    assert m["contact_evidence"]["onset_counts"] == {"initial_overlap": 1}


def test_contact_onset_reports_anchor_entry_and_execution_separately():
    from driversi.evaluation.metrics.contacts import contact_report
    from driversi.evaluation.search.traffic import crossing_scene

    scene, _ = crossing_scene()
    states = np.zeros((3, 3, 13))
    present = np.array([[True, True, False], [True, True, True], [True, True, True]])
    frames = [
        {"tick": 20, "contacts": [["a", "b"]]},
        {"tick": 21, "contacts": [["a", "b"], ["b", "far"]]},
        {"tick": 22, "contacts": [["a", "far"]]},
    ]
    evidence = contact_report(scene, states, present, frames, "a")
    assert evidence["onset_counts"] == {
        "existing_at_anchor": 1,
        "entry_overlap": 1,
        "execution_contact": 1,
    }
    assert evidence["owner_contact_ticks"] == 3 and evidence["other_contact_ticks"] == 1
