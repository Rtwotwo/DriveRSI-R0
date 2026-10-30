"""Finite producer tests check causal execution, budget cost and honest rejection."""

import copy
import json

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from driversi.contracts.road import RoadGraph, Route
from driversi.evaluation.search import challenge
from driversi.evaluation.search.challenge import BrakeCandidate, run_challenge_search
from driversi.evaluation.benchmarks.real_scenes import verify_replay
from driversi.evaluation.benchmarks.scenarios import following_scene

pytestmark = pytest.mark.native


def graph_for(scene):
    return RoadGraph(
        {
            "schema_version": 1,
            "map_name": "straight",
            "map_version": None,
            "origin_xyz": [0, 0, 0],
            "bounds_local": [-30, -15, 250, 100],
        },
        {
            f"lane:{y}": {
                "kind": "lane",
                "points": [[-30, y], [250, y]],
                "outgoing": [],
                "speed_limit_mps": None,
            }
            for y in (0, 80)
        },
    )


def run(tmp_path, *, scene=None, name="run", **changes):
    scene = scene or following_scene(8, 25, duration=3)
    settings = dict(
        follower_id="follower",
        challenger_id="lead",
        fork_tick=5,
        candidates=[BrakeCandidate(0, 10, -3)],
        max_world_steps=100,
        wall_seconds=60,
    )
    settings.update(changes)
    return run_challenge_search(scene, graph_for(scene), tmp_path / name, **settings)


def test_complete_causal_search_and_replay(tmp_path):
    scene = following_scene(8, 25, duration=6)
    original = copy.deepcopy(scene)
    report = run(
        tmp_path,
        scene=scene,
        candidates=[BrakeCandidate(0, 20, -3), BrakeCandidate(5, 20, -4)],
        max_world_steps=200,
    )
    assert report["status"] == "completed"
    assert report["selected_candidate"] in {"candidate-000", "candidate-001"}
    assert report["budget"]["world_steps_completed"] == 5 + 3 * 55
    assert report["budget"]["physical_agent_steps"] == 3 * (5 + 3 * 55)
    assert report["trainable"] is False
    assert report["parent_unchanged"]
    np.testing.assert_array_equal(scene.poses, original.poses)
    for item in report["attempts"]:
        assert item["parent_unchanged"]
        compare = item["comparison_metrics"]
        assert compare["unmodified_prefix_exact"]
        assert compare["follower_same_onset_action"] is True
        assert compare["first_follower_action_response_seconds_after_onset"] >= 0.1
        assert item["action_override_steps"] > 0
        assert all(verify_replay(scene, tmp_path / "run" / item["rollout"]).values())
        with np.load(tmp_path / "run" / item["rollout"], allow_pickle=False) as data:
            assert len(data["actions"]) == 60
            commands = json.loads(str(data["controller_diagnostics"]))
            assert all(
                c["action"] == c["policy_action"]
                for row in commands
                for c in row
                if c["actor"] != "lead"
            )
            assert all(
                c["predicted_next_lateral_acceleration"] is None
                for row in commands
                for c in row
                if c.get("intervention")
            )
    assert (tmp_path / "run/index.html").exists()


def test_repeated_search_preserves_every_state_action_and_choice(tmp_path):
    first, second = run(tmp_path, name="first"), run(tmp_path, name="second")
    assert first["selected_candidate"] == second["selected_candidate"]
    assert first["attempts"][0]["metrics"] == second["attempts"][0]["metrics"]
    for name in ("baseline.npz", "candidate-000.npz"):
        with np.load(tmp_path / "first" / name) as a, np.load(tmp_path / "second" / name) as b:
            for key in ("states", "motion", "actions", "valid", "collision", "offroad", "goal"):
                np.testing.assert_array_equal(a[key], b[key])


def test_step_budget_counts_prefix_baseline_and_attempts(tmp_path):
    report = run(
        tmp_path, max_world_steps=60, candidates=[BrakeCandidate(i, 5, -3) for i in (0, 1, 2)]
    )
    assert report["status"] == "partial"
    assert report["budget"]["world_steps_attempted"] == 55
    assert report["budget"]["world_steps_completed"] == 55
    assert [r["status"] for r in report["attempts"]] == ["completed", "not_run", "not_run"]
    assert all(
        r["reason"] == "insufficient_complete_candidate_step_budget" for r in report["attempts"][1:]
    )


def test_insufficient_baseline_budget_preserves_prefix(tmp_path):
    report = run(tmp_path, max_world_steps=15)
    assert report["reason"] == "insufficient_complete_baseline_step_budget"
    assert report["budget"]["world_steps_completed"] == 5
    assert (tmp_path / "run/prefix.npz").exists()
    assert report["attempt_counts"]["not_run"] == 1


def test_no_effect_candidate_is_retained_and_never_selected(tmp_path):
    report = run(tmp_path, candidates=[BrakeCandidate(0, 5, -0.0001)])
    assert report["attempts"][0]["status"] == "no_effect"
    assert report["attempts"][0]["action_override_steps"] == 0
    assert report["selected_candidate"] is None
    assert (tmp_path / "run/candidate-000.npz").exists()


def test_out_of_horizon_and_duration_candidates_are_declared_rejections(tmp_path):
    report = run(tmp_path, candidates=[BrakeCandidate(0, 50, -3), BrakeCandidate(25, 1, -3)])
    assert report["budget"]["world_steps_completed"] == 30
    assert [row["reason"] for row in report["attempts"]] == [
        "pulse_duration_outside_bound",
        "pulse_exceeds_remaining_horizon",
    ]
    assert report["selected_candidate"] is None


@pytest.mark.parametrize(
    "changes",
    [{"follower_id": "far"}, {"scene": following_scene(8, 1, duration=3), "fork_tick": 0}],
)
def test_unsupported_pairs_do_not_become_generated_challenges(tmp_path, changes):
    report = run(tmp_path, **changes)
    assert report["status"] == "rejected"
    assert not report["pair_support"]["supported"]
    assert report["selected_candidate"] is None
    assert all(r["status"] == "not_run" for r in report["attempts"])


def test_wall_budget_stops_before_next_physics_step(tmp_path, monkeypatch):
    clock = iter(np.arange(1000, dtype=float))
    monkeypatch.setattr(challenge.time, "monotonic", lambda: next(clock))
    report = run(tmp_path, wall_seconds=0.5)
    assert report["budget"]["world_steps_attempted"] == 0
    assert report["prefix"]["stop_reason"] == "wall_budget_exhausted"
    assert report["selected_candidate"] is None


def test_native_exception_is_charged_and_retained(tmp_path, monkeypatch):
    def fail(self, actions):
        raise RuntimeError("injected native failure")

    monkeypatch.setattr(challenge.PufferDriveWorld, "step", fail)
    report = run(tmp_path, fork_tick=0)
    assert report["status"] == "incomplete"
    assert report["budget"]["world_steps_attempted"] == 1
    assert report["budget"]["world_steps_completed"] == 0
    assert "injected native failure" in report["baseline"]["error"]
    assert report["attempt_counts"]["not_run"] == 1


def test_other_current_lead_is_not_relabelled_as_selected_challenger(tmp_path):
    scene = following_scene(8, 25, duration=3)
    # Initial pair remains valid; a fixed crossing obstacle later contacts it.
    scene.controlled[2] = False
    scene.metadata["task_goals_xy"].pop("far")
    scene.poses[:, 2, 0] = 12
    scene.poses[:, 2, 1] = 0
    scene.velocities[:, 2] = 0
    report = run(tmp_path, scene=scene, fork_tick=0)
    # A third object becomes the current lead, so this case must stay rejected.
    assert report["status"] == "rejected"
    assert "challenger_not_current_route_lead" in report["pair_support"]["reasons"]


def test_report_and_output_are_not_overwritten(tmp_path):
    run(tmp_path)
    before = (tmp_path / "run/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert before == (tmp_path / "run/report.json").read_bytes()


@pytest.mark.parametrize(
    "args", [(True, 2, -3), (0, 0, -3), (0, 2, float("nan")), (0, 2, -6), (0, 2, 0)]
)
def test_invalid_candidate_cannot_enter_cohort(args):
    with pytest.raises(ValueError):
        BrakeCandidate(*args)


def test_full_events_survive_contact_and_actor_termination(tmp_path):
    scene = following_scene(8, 25, duration=3)
    scene.controlled[2] = False
    scene.metadata["task_goals_xy"].pop("far")
    scene.poses[:, 2, 0] = 6
    scene.poses[:, 2, 1] = np.interp(scene.times, [0, 0.5, 1.0, 3.0], [80, 80, 0, 0])
    scene.velocities[:, 2] = 0
    report = run(tmp_path, scene=scene, fork_tick=0)
    assert report["pair_support"]["supported"]
    assert report["baseline"]["metrics"]["events"]["collision"]["follower"] == 1
    for row in [report["baseline"], *report["attempts"]]:
        with np.load(tmp_path / "run" / row["rollout"]) as record:
            assert (
                record["collision"][:, 0].sum() == row["metrics"]["events"]["collision"]["follower"]
            )
            assert len(record["actions"]) == 30
    assert report["attempts"][0]["comparison_metrics"]["common_precontact_action_steps"] < 30


def test_unequal_endings_keep_individual_full_views_without_padding(tmp_path):
    scene = following_scene(8, 25, duration=3)
    scene.metadata["task_goals_xy"] = {"follower": [12, 0], "lead": [41, 0], "far": [12, 80]}
    report = run(tmp_path, scene=scene, fork_tick=0)
    base, row = report["baseline"], report["attempts"][0]
    assert base["actual_seconds"] != row["actual_seconds"]
    assert row["comparison"] == "single_branch"
    assert (tmp_path / "run" / row["viewer"]).exists()
    for item in (base, row):
        with np.load(tmp_path / "run" / item["rollout"]) as data:
            assert len(data["actions"]) == round(item["actual_seconds"] * 10)


@pytest.mark.parametrize("split", ["heldout", "test", "navtest", "navhard"])
def test_protected_evaluation_inputs_cannot_be_adaptively_mined(tmp_path, split):
    scene = following_scene(8, 25, duration=3)
    scene.metadata["split"] = split
    with pytest.raises(ValueError, match="protected"):
        run(tmp_path, scene=scene)
    assert not (tmp_path / "run").exists()


def test_failed_candidate_persistence_is_not_reported_as_unattempted(tmp_path, monkeypatch):
    export = challenge.export_replay

    def fail_on_candidate(scene, rollout, output, **kwargs):
        if "candidate-" in str(output):
            raise OSError("injected artifact error")
        return export(scene, rollout, output, **kwargs)

    monkeypatch.setattr(challenge, "export_replay", fail_on_candidate)
    report = run(tmp_path)
    assert report["status"] == "failed"
    row = report["attempts"][0]
    assert row["status"] == "failed"
    assert row["world_steps_attempted"] == 25
    assert "injected artifact error" in row["reason"]
    assert report["attempt_counts"]["failed"] == 1


def test_executed_braking_metric_includes_last_successor_motion():
    scene = following_scene(8, 25, duration=3)
    routes, _ = challenge.resolve_routes(scene, graph_for(scene))
    controller = challenge.RouteFollower(routes)
    with challenge.PufferDriveWorld(scene) as world:
        record = challenge.RolloutRecord.start(world)
        commands = controller.commands(world.observe())
        actions = {actor: 1 for actor in world.actor_ids}
        transition = world.step(actions)
        record.append(world, actions, transition, commands)
        metrics = challenge._metrics(
            scene, record, controller, "follower", "lead", challenge.ChallengeBounds()
        )
    index = scene.actor_ids.index("follower")
    assert record.motion[0][index, 0] == 0
    assert record.motion[1][index, 0] < 0
    assert (
        metrics["actors"]["follower"]["minimum_actual_acceleration_mps2"]
        == record.motion[1][index, 0]
    )


@pytest.mark.parametrize("terminates", [False, True])
@pytest.mark.parametrize(
    "boundary,expected",
    [
        ("corridor", "follower:route_tracking_outside_supported_corridor"),
        ("endpoint", "follower:route_exhausted"),
    ],
)
def test_last_valid_successor_geometry_is_checked_even_when_terminated(
    boundary, expected, terminates
):
    from types import SimpleNamespace

    scene = following_scene(8, 25, duration=3)
    routes, _ = challenge.resolve_routes(scene, graph_for(scene))
    states = np.zeros((2, 3, 13))
    states[:, :, :4] = scene.poses[:2]
    states[:, :, 4:7] = scene.velocities[:2]
    if boundary == "corridor":
        states[:, 0, 1] = [1.49, 1.51]
        distance = 1.49
    else:
        routes["follower"] = Route([[-30, 0], [0.8, 0]])
        states[:, 0, 0] = [0.79, 0.81]
        distance = 0.0
    states[-1, 0, 9] = int(terminates)
    transition = SimpleNamespace(
        valid=np.ones(3, bool),
        terminated=np.array([terminates, False, False]),
        events={name: np.zeros(3, bool) for name in ("collision", "offroad", "goal")},
    )
    record = challenge.RolloutRecord(
        list(states),
        [np.zeros((3, 3))] * 2,
        [transition],
        [[7] * 3],
        [
            [
                {
                    "route_distance_m": distance if actor == "follower" else 0.0,
                    "route_exhausted": False,
                }
                for actor in scene.actor_ids
            ]
        ],
    )
    result = challenge._metrics(
        scene,
        record,
        challenge.RouteFollower(routes),
        "follower",
        "lead",
        challenge.ChallengeBounds(),
    )
    assert result["events"]["offroad"]["follower"] == 0
    assert result["constraint_violations"] == [expected]


def test_challenge_carries_source_context_to_every_view(tmp_path):
    from driversi.contracts.context import SceneContext, scene_fingerprint

    scene = following_scene(8, 25, duration=3)
    meta = {
        "schema_version": 1,
        "scene_fingerprint": scene_fingerprint(scene),
        **{k: scene.metadata[k] for k in ("scenario_id", "source_log", "map_name", "origin_xyz")},
        "scene_times": scene.times.tolist(),
        "reference_policy": "previous_sample",
        "max_sample_age_seconds": 0.75,
    }
    frames = [
        {
            "time_seconds": float(t),
            "timestamp_us": int(round(t * 1e6)),
            "objects": [
                {
                    "id": "context-person",
                    "kind": "pedestrian",
                    "pose": [10, 4, 0, 0],
                    "dimensions": [0.5, 0.5, 1.7],
                }
            ],
            "signals": {"available": False, "records": []},
        }
        for t in np.arange(0, 3.1, 0.5)
    ]
    context = SceneContext(meta, frames, [], {})
    report = run(tmp_path, scene=scene, scene_context=context)
    assert report["source_context"]["status"] == "provided"
    assert "context.json" in report["input_artifacts"]
    for name in ["baseline.html", "candidate-000.html"]:
        text = (tmp_path / "run" / name).read_text()
        assert "context-person" in text and "source_context_only" in text
    bad = copy.deepcopy(context)
    bad.metadata["scene_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        run(tmp_path, name="bad", scene_context=bad)
    assert not (tmp_path / "bad").exists()
