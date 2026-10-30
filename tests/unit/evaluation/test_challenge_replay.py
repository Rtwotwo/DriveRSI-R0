"""Pure contract checks; no native simulator, Torch, or GPU dependency."""

import copy
import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from driversi.contracts.road import RoadGraph
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.rendering.execution import challenge_replay as bridge


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path):
    scene = following_scene(8, 25, duration=2)
    scene.metadata["source_start_timestamp_us"] = 123
    scene.save(tmp_path / "scene.npz")
    graph = RoadGraph(
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
    graph.save(tmp_path / "road_graph.json")
    states = np.zeros((21, 3, 13), dtype=np.float32)
    states[:, :, :4], states[:, :, 4:7] = scene.poses, scene.velocities
    states[:, :, 7:9] = [[220, 0], [220, 0], [220, 80]]
    arrays = {
        "states": states,
        "motion": np.zeros((21, 3, 3), np.float32),
        "actions": np.full((20, 3), 7, np.int32),
        "valid": np.ones((20, 3), bool),
    }
    arrays.update(
        {
            key: np.zeros((20, 3), bool)
            for key in ("terminated", "truncated", "collision", "offroad", "goal")
        }
    )
    arrays["truncated"][-1] = True
    key = bridge._scene_key(scene)
    metadata = {
        "schema_version": 1,
        "kind": "engineering_rollout",
        "scene_key": key,
        "scenario_id": scene.metadata["scenario_id"],
        "scene_metadata": scene.metadata,
        "steps": 20,
        "dt_seconds": 0.1,
        "action_space": "jerk_discrete_12",
        "backend_build": "test-native-build",
        "protocol": {
            "kind": bridge.SOURCE_VERSION,
            "branch": "candidate-000",
            "trainable": False,
            "details": {"stop_reason": "episode_boundary"},
        },
    }
    np.savez(
        tmp_path / "candidate.npz",
        metadata=json.dumps(metadata),
        actor_ids=scene.actor_ids,
        controlled_ids=scene.actor_ids,
        controller_diagnostics=json.dumps([]),
        **arrays,
    )
    report = {
        "schema_version": 1,
        "kind": bridge.SOURCE_VERSION,
        "trainable": False,
        "status": "completed",
        "scenario_id": scene.metadata["scenario_id"],
        "scene_key": key,
        "backend_build": metadata["backend_build"],
        "selected_candidate": "candidate-000",
        "input_artifacts": {
            name: {"path": name, "sha256": bridge._sha(tmp_path / name)}
            for name in ("scene.npz", "road_graph.json")
        },
        "attempts": [
            {
                "id": "candidate-000",
                "status": "completed",
                "metrics": {"constraint_violations": []},
                "rollout": "candidate.npz",
                "rollout_sha256": bridge._sha(tmp_path / "candidate.npz"),
            }
        ],
    }
    write_json(tmp_path / "report.json", report)
    binding = {
        "log_name": scene.metadata["source_log"],
        "map_name": "straight",
        "source_start_timestamp_us": 123,
        "inputs": {"scene": {"sha256": bridge._sha(tmp_path / "scene.npz")}},
    }
    write_json(tmp_path / "binding.json", binding)
    fake = tmp_path / "asset"
    fake.write_text("test asset")
    config = {
        "scene": str(tmp_path / "scene.npz"),
        "road_graph": str(tmp_path / "road_graph.json"),
        "binding": str(tmp_path / "binding.json"),
        "world_python": sys.executable,
        "cameras": ["CAM_F0"],
        "virtual_camera_actors": [],
        **{
            name: str(fake)
            for name in ("native_path", "config_path", "checkpoint_path", "road_block_config")
        },
    }
    write_json(tmp_path / "config.json", config)
    return tmp_path


def load(source):
    return bridge.load_replay_input(source / "config.json", source / "report.json")


def current(bundle, tick):
    a = bundle.arrays
    result = {
        "tick": tick,
        "build_id": bundle.metadata["backend_build"],
        "dt_seconds": 0.1,
        "all_actor_ids": list(bundle.scene.actor_ids),
        "controlled_ids": list(bundle.controlled),
        "state": a["states"][tick].tolist(),
        "motion": a["motion"][tick].tolist(),
        "alive": [True] * 3,
        "done": tick == 20,
    }
    if tick:
        actions = dict(zip(bundle.controlled, a["actions"][tick - 1].tolist()))
        result.update(
            policy_actions=actions,
            executed_actions=actions.copy(),
            **{name: a[name][tick - 1].tolist() for name in ("valid", "terminated", "truncated")},
            events={name: a[name][tick - 1].tolist() for name in ("collision", "offroad", "goal")},
        )
    return result


class FakeSession:
    def __init__(self, bundle, mutate=None):
        self.bundle, self.mutate = bundle, mutate
        self.current, self.calls = current(bundle, 0), []

    def decision_identity(self):
        return {"session_id": "fake", "generation": 0, "tick": self.current["tick"]}

    def advance(self, **values):
        self.calls.append(copy.deepcopy(values))
        tick = self.current["tick"]
        assert values["expected_tick"] == tick
        assert values["external_actions"] == dict(
            zip(self.bundle.controlled, self.bundle.arrays["actions"][tick].tolist())
        )
        self.current = current(self.bundle, tick + 1)
        if self.mutate:
            self.mutate(self.current)


def test_completed_bundle_and_full_joint_replay(source):
    bundle = load(source)
    session = FakeSession(bundle)
    verification = {"verified_steps": 0}
    bridge.replay_into_session(session, bundle, verification)
    assert verification == {"verified_steps": 20, "initial_state_exact": True}
    assert len(session.calls) == 20
    assert all(
        call["policy_decision"]["rendered_rgb_used_for_replay_action"] is False
        for call in session.calls
    )
    assert all(call["policy_decision"]["rgb_is_policy_input"] is None for call in session.calls)
    assert not bundle.arrays["states"].flags.writeable


@pytest.mark.parametrize(
    "field,value",
    [
        ("build_id", "other-native"),
        ("controlled_ids", ["lead", "follower", "far"]),
        ("all_actor_ids", ["lead", "follower", "far"]),
        ("tick", 1),
        ("dt_seconds", 0.2),
    ],
)
def test_initial_identity_rejected(source, field, value):
    bundle = load(source)
    row = current(bundle, 0)
    row[field] = value
    with pytest.raises(ValueError, match="identity/build/order"):
        bridge.verify_world_record(bundle, row, 0)


@pytest.mark.parametrize(
    "name",
    [
        "state",
        "motion",
        "policy_actions",
        "executed_actions",
        "valid",
        "terminated",
        "truncated",
        "events",
    ],
)
def test_successor_mismatch_stops_immediately(source, name):
    bundle = load(source)

    def corrupt(row):
        if name in ("state", "motion"):
            row[name][0][0] += 1
        elif name in ("policy_actions", "executed_actions"):
            row[name]["lead"] = 2
        elif name == "events":
            row[name]["collision"][0] = True
        else:
            row[name][0] = not row[name][0]

    session = FakeSession(bundle, corrupt)
    evidence = {"verified_steps": 0}
    with pytest.raises(ValueError):
        bridge.replay_into_session(session, bundle, evidence)
    assert evidence["verified_steps"] == 0
    assert len(session.calls) == 1


def test_source_hash_mismatch_rejected(source):
    with (source / "candidate.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        load(source)


def test_report_build_mismatch_rejected(source):
    path = source / "report.json"
    report = json.loads(path.read_text())
    report["backend_build"] = "different"
    write_json(path, report)
    with pytest.raises(ValueError, match="provenance/build/branch"):
        load(source)


def test_action_actor_order_cannot_be_changed(source):
    path = source / "candidate.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["controlled_ids"] = ["lead", "follower", "far"]
    np.savez(path, **arrays)
    report = json.loads((source / "report.json").read_text())
    report["attempts"][0]["rollout_sha256"] = bridge._sha(path)
    write_json(source / "report.json", report)
    with pytest.raises(ValueError, match="actor identity/order"):
        load(source)


@pytest.mark.parametrize("status", ["failed", "no_effect", "not_run", "rejected"])
def test_only_completed_candidates_are_rendered(source, status):
    path = source / "report.json"
    report = json.loads(path.read_text())
    report["attempts"][0]["status"] = status
    write_json(path, report)
    with pytest.raises(ValueError, match="completed challenge"):
        load(source)


def test_explicit_candidate_and_missing_selection(source):
    path = source / "report.json"
    report = json.loads(path.read_text())
    report["selected_candidate"] = None
    write_json(path, report)
    with pytest.raises(ValueError):
        load(source)
    assert (
        bridge.load_replay_input(source / "config.json", path, "candidate-000").candidate["id"]
        == "candidate-000"
    )


def test_binding_must_match_actual_scene(source):
    path = source / "binding.json"
    binding = json.loads(path.read_text())
    binding["inputs"]["scene"]["sha256"] = "wrong"
    write_json(path, binding)
    with pytest.raises(ValueError, match="binding does not match"):
        load(source)


def test_nonempty_route_override_is_never_discarded(source):
    path = source / "route_tasks.json"
    write_json(path, {"initial_nodes": {"lead": "lane:0"}})
    report = json.loads((source / "report.json").read_text())
    report["input_artifacts"][path.name] = {"path": path.name, "sha256": bridge._sha(path)}
    write_json(source / "report.json", report)
    with pytest.raises(ValueError, match="route overrides"):
        load(source)


def test_load_failure_written_without_loading_renderer(source, monkeypatch):
    (source / "candidate.npz").write_bytes(b"broken")
    monkeypatch.setattr(bridge, "_runtime", lambda _: pytest.fail("GPU model must not load"))
    output = source / "rejected-render"
    result = bridge.run(source / "config.json", source / "report.json", output)
    assert result["status"] == "failed" and result["verified_steps"] == 0
    assert result["failure"]["type"] == "ValueError"
    assert json.loads((output / "verification.json").read_text()) == result
    assert not (output / "session").exists()


def test_state_change_between_hash_load_and_replay_is_detected(source):
    original = load(source)
    arrays = {name: value.copy() for name, value in original.arrays.items()}
    arrays["states"][1, 0, 0] += 0.1
    changed = replace(original, arrays=arrays)
    with pytest.raises(ValueError, match="state differs"):
        bridge.replay_into_session(FakeSession(original), changed, {"verified_steps": 0})


@pytest.mark.parametrize("workers", [None, 4])
def test_replay_failure_survives_session_recording_and_cleanup_errors(source, monkeypatch, workers):
    if workers is not None:
        config = json.loads((source / "config.json").read_text())
        config["image_write_workers"] = workers
        write_json(source / "config.json", config)
    bundle = load(source)

    def mutate(row):
        row["state"][0][0] += 1

    session = FakeSession(bundle, mutate)
    session.manifest = {}

    def failure(_):
        raise OSError("manifest unavailable")

    def close():
        raise OSError("worker cleanup unavailable")

    session._failed, session.close = failure, close
    calls = []
    monkeypatch.setattr(bridge, "_preflight", lambda *_: None)
    monkeypatch.setattr(bridge, "_runtime", lambda config: calls.append(config))
    def create_session(_, settings):
        assert settings["image_write_workers"] == (workers or 1)
        return session

    monkeypatch.setattr(bridge, "_session", create_session)
    output = source / "failed-step-render"
    result = bridge.run(source / "config.json", source / "report.json", output)
    assert result["status"] == "failed" and result["verified_steps"] == 0
    assert result["failure"]["reason"] == "state differs at tick 1"
    assert result["failure_recording_error"] == "manifest unavailable"
    assert result["close_error"] == "worker cleanup unavailable"
    assert len(calls) == 1
    assert json.loads((output / "verification.json").read_text()) == result
    assert not (output / "sequence").exists()


def test_full_cpu_preflight_rejects_late_drift_before_gpu_load(source, monkeypatch):
    from driversi.sim.workers import worker_client
    bundle = load(source)
    class Worker:
        def __init__(self, **settings):
            self.current = current(bundle, 0)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def request(self, command, **values):
            assert command == "step"
            tick = self.current["tick"]
            assert values["expected_tick"] == tick
            self.current = current(bundle, tick + 1)
            if tick == 17:
                self.current["motion"][0][0] += 0.01
            return self.current
    monkeypatch.setattr(worker_client, "WorldWorker", Worker)
    monkeypatch.setattr(bridge, "_runtime", lambda _: pytest.fail("GPU must not load after drift"))
    result = bridge.run(source / "config.json", source / "report.json", source / "late-drift")
    assert result["status"] == "failed"
    assert result["failure"]["reason"] == "motion differs at tick 18"
    assert result["verified_steps"] == 0
