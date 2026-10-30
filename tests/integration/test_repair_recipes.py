"""Executed-prefix, budget and memory contracts; no teacher score stand-ins."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from driversi.agents.traffic_control.following import FollowingConfig
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import Route
from driversi.evaluation.search.augmentation import generate_pair
from driversi.rsi.correction.recipes import VERIFIER, prepare, run
from driversi.rsi.correction.store import RepairStore
from driversi.sim.persistence.recording import write_rollout

pytest.importorskip("driversi.sim._native")


def make_request(straight_scene, root, group="log-a"):
    root.mkdir(parents=True, exist_ok=True)
    scene = copy.deepcopy(straight_scene)
    scene.metadata["source_log"] = group
    scene.poses[:, 0, 0] += 16
    scene.poses[:, 1, 1] = 0
    routes = {a: Route([[-40, 0], [140, 0]]) for a in ("a", "b")}
    scene = bind_driving_tasks(scene, routes)
    report, records, initial, _ = generate_pair(
        scene, routes, actor="a", explore_steps=10, explore_speed=2
    )
    scene.save(root / "scene.npz")
    from driversi.agents.traffic_control.following import RouteFollower

    write_rollout(
        scene,
        root / "source.npz",
        record=records[1],
        snapshot=initial,
        controlled_ids=("a", "b"),
        policy="test-source",
        controller_configuration=RouteFollower(routes, FollowingConfig()).describe(),
    )
    base = {
        "id": "ordinary-v1",
        "backend": "route_following_v1",
        "desired_speed": 8,
        "headway": 1.5,
    }
    return {
        "schema_version": 1,
        "id": group + "-run",
        "case": {
            "scene": str(root / "scene.npz"),
            "rollout": str(root / "source.npz"),
            "actor": "a",
            "anchor_tick": 10,
            "future_steps": 20,
            "failure_type": "diagnostic",
        },
        "baseline": base,
        "recipes": [{**base, "id": "same-v1"}, {**base, "id": "cautious-v1", "desired_speed": 3}],
        "verifier": {"version": VERIFIER, "min_progress_m": 0},
        "budget": {"max_attempts": 3, "max_sim_agent_steps": 200, "max_wall_seconds": 30},
    }


def test_real_prefix_same_recipe_and_causal_opponents(straight_scene, tmp_path):
    raw = make_request(straight_scene, tmp_path / "data")
    store = RepairStore(tmp_path / "ledger.sqlite")
    result = run(store, raw, tmp_path / "run")
    assert result["status"] == "completed", result
    assert result["anchor_verified"] and result["admitted_agent_steps"] == 140
    assert len(result["attempts"]) == 3 and not result["trainable"]
    a = np.load(tmp_path / "run/ordinary-v1.npz")
    b = np.load(tmp_path / "run/same-v1.npz")
    c = np.load(tmp_path / "run/cautious-v1.npz")
    source = np.load(raw["case"]["rollout"])
    for k in ("states", "motion", "actions", "valid", "terminated", "truncated", "presence"):
        np.testing.assert_array_equal(a[k], b[k])
    for k in ("states", "motion", "presence"):
        np.testing.assert_array_equal(a[k][:11], c[k][:11])
        np.testing.assert_array_equal(a[k][:11], source[k][:11])
    assert not np.array_equal(a["states"][11:], c["states"][11:])
    assert not np.array_equal(a["actions"][10:, 1], c["actions"][10:, 1])
    assert all(r["metrics"]["official_score"] is None for r in result["attempts"])
    assert store.get(raw["id"])["status"] == "completed"


@pytest.mark.parametrize(
    "cap,status,attempts,steps", [(59, "budget_exhausted", 0, 20), (60, "budget_exhausted", 1, 60)]
)
def test_reserves_whole_suffix(straight_scene, tmp_path, cap, status, attempts, steps):
    raw = make_request(straight_scene, tmp_path / "data")
    raw["budget"]["max_sim_agent_steps"] = cap
    result = run(RepairStore(tmp_path / "ledger.sqlite"), raw, tmp_path / "run")
    assert result["status"] == status
    assert len(result["attempts"]) == attempts
    assert result["admitted_agent_steps"] == steps


def test_corrupt_prefix_rejected_without_using_future(straight_scene, tmp_path):
    raw = make_request(straight_scene, tmp_path / "data")
    with np.load(raw["case"]["rollout"]) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["states"][4, 0, 0] += 1
    np.savez_compressed(raw["case"]["rollout"], **arrays)
    result = run(RepairStore(tmp_path / "ledger.sqlite"), raw, tmp_path / "run")
    assert result["status"] == "failed" and "prefix differs" in result["failure_reason"]
    assert not result["attempts"]


@pytest.mark.parametrize("change", ["teacher", "score", "horizon", "test"])
def test_strict_contract(straight_scene, tmp_path, change):
    raw = make_request(straight_scene, tmp_path / "data")
    if change == "teacher":
        raw["recipes"][0]["backend"] = "driverl"
    elif change == "score":
        raw["recipes"][0]["min_progress_m"] = 0
    elif change == "horizon":
        raw["case"]["future_steps"] = 1000
    else:
        from driversi.contracts.scene import Scene

        scene = Scene.load(Path(raw["case"]["scene"]))
        scene.metadata["split"] = "navtest"
        scene.save(Path(raw["case"]["scene"]))
    with pytest.raises(ValueError):
        prepare(raw, tmp_path)


def test_memory_excludes_current_log_and_changed_verifier(straight_scene, tmp_path):
    raw = make_request(straight_scene, tmp_path / "data")
    store = RepairStore(tmp_path / "ledger.sqlite")
    run(store, raw, tmp_path / "run")
    _, evidence = store.order(prepare(raw, tmp_path))
    assert evidence == []
    other = make_request(straight_scene, tmp_path / "other", "log-b")
    _, evidence = store.order(prepare(other, tmp_path))
    assert len(evidence) == 2
    other["verifier"]["min_progress_m"] = 1
    assert store.order(prepare(other, tmp_path))[1] == []


def gate_request(store, straight_scene, root, incumbent, candidate, name="gate-one"):
    cases = []
    for group in ("heldout-b", "heldout-c"):
        r = make_request(straight_scene, root / group, group)
        cases.append({k: r[k] for k in ("id", "case", "budget")})
    return {
        "schema_version": 1,
        "id": name,
        "library": "test-recipes",
        "incumbent": incumbent,
        "candidate": candidate,
        "verifier": {"version": VERIFIER, "min_progress_m": 0},
        "policy": {
            "min_cases": 2,
            "min_groups": 2,
            "min_pass_gain": 0,
            "min_progress_gain_m": 0.01,
            "max_progress_regression_m": 0,
        },
        "cases": cases,
    }


def test_recipe_promotion_separate_from_models_and_memory(straight_scene, tmp_path):
    from driversi.rsi.correction.gate import run_gate

    store = RepairStore(tmp_path / "ledger.sqlite")
    raw = make_request(straight_scene, tmp_path / "dev")
    run(store, raw, tmp_path / "development")
    gate = gate_request(store, straight_scene, tmp_path, raw["recipes"][1], raw["baseline"])
    result = run_gate(store, gate, tmp_path / "gate")
    assert result["status"] == "completed", result
    assert result["decision"]["outcome"] == "promoted", result["decision"]
    assert not result["decision"]["model_promoted"] and not result["decision"]["teacher_qualified"]
    with store.registry._connect() as db:
        assert (
            db.execute("SELECT recipe_id FROM repair_libraries").fetchone()[0]
            == raw["baseline"]["id"]
        )
    assert store.registry.protected_groups() == {"heldout-b", "heldout-c"}
    new = make_request(straight_scene, tmp_path / "later", "later-log")
    _, evidence = store.order(prepare(new, tmp_path))
    assert {row["source_log"] for row in evidence} == {"log-a"}
    gate["id"] = "duplicate-gate"
    with pytest.raises(ValueError, match="overlap"):
        run_gate(store, gate, tmp_path / "gate-reused")
    forbidden = make_request(straight_scene, tmp_path / "bad", "heldout-b")
    with pytest.raises(ValueError, match="Protected"):
        run(store, forbidden, tmp_path / "bad-run")


def test_same_actions_do_not_promote_recipe(straight_scene, tmp_path):
    from driversi.rsi.correction.gate import run_gate

    store = RepairStore(tmp_path / "ledger.sqlite")
    raw = make_request(straight_scene, tmp_path / "dev")
    run(store, raw, tmp_path / "development")
    gate = gate_request(store, straight_scene, tmp_path, raw["baseline"], raw["recipes"][0])
    result = run_gate(store, gate, tmp_path / "gate")
    assert result["decision"]["outcome"] == "rejected"
    assert result["decision"]["mean_route_progress_gain_m"] == 0
    with store.registry._connect() as db:
        assert (
            db.execute("SELECT recipe_id FROM repair_libraries").fetchone()[0]
            == raw["baseline"]["id"]
        )


def test_gate_budget_failure_preserves_incumbent(straight_scene, tmp_path):
    from driversi.rsi.correction.gate import run_gate

    store = RepairStore(tmp_path / "ledger.sqlite")
    raw = make_request(straight_scene, tmp_path / "dev")
    run(store, raw, tmp_path / "development")
    gate = gate_request(store, straight_scene, tmp_path, raw["baseline"], raw["recipes"][0])
    gate["cases"][0]["budget"]["max_sim_agent_steps"] = 1
    result = run_gate(store, gate, tmp_path / "gate")
    assert result["decision"]["outcome"] == "inconclusive"
    assert len(result["runs"]) == 2


def test_memory_fails_closed_on_errors_and_changed_recipe(straight_scene, tmp_path):
    raw = make_request(straight_scene, tmp_path / "data")
    store = RepairStore(tmp_path / "ledger.sqlite")
    run(store, raw, tmp_path / "run")
    raw["id"] = "altered-version"
    raw["recipes"][0]["headway"] = 2
    with pytest.raises(ValueError, match="different implementation"):
        run(store, raw, tmp_path / "changed")


def test_dead_owner_recovery_is_charged(straight_scene, tmp_path):
    raw = make_request(straight_scene, tmp_path / "data")
    spec = prepare(raw, tmp_path)
    store = RepairStore(tmp_path / "ledger.sqlite")
    directory = tmp_path / "interrupted"
    directory.mkdir()
    state = store.register_run(spec, directory, role="development", gate_id=None)
    with pytest.raises(ValueError, match="active"):
        store.recover(raw["id"])
    with store.registry._connect() as db:
        state["owner"]["start"] = "already-ended"
        db.execute("UPDATE repair_runs SET payload=? WHERE id=?", (json.dumps(state), raw["id"]))
    (directory / "budget.json").write_text(json.dumps({"admitted_agent_steps": 17}))
    result = store.recover(raw["id"])
    assert result["status"] == "interrupted" and result["admitted_agent_steps"] == 17


def test_non_target_challenger_route_and_prefix_diagnostics_preserved(straight_scene, tmp_path):
    from driversi.agents.traffic_control.following import RouteFollower
    from driversi.contracts.scene import Scene
    from driversi.sim.backends.pufferdrive import PufferDriveWorld
    from driversi.sim.persistence.recording import RolloutRecord

    raw = make_request(straight_scene, tmp_path / "data")
    scene = Scene.load(Path(raw["case"]["scene"]))
    original = {a: Route([[-40, 0], [140, 0]]) for a in ("a", "b")}
    changed = {**original, "b": Route([[-40, 0], [10, 0], [20, 2], [140, 2]])}
    normal, challenger = RouteFollower(original), RouteFollower(changed)
    with PufferDriveWorld(scene) as world:
        initial = world.snapshot()
        record = RolloutRecord.start(world)
        for tick in range(30):
            commands = (normal if tick < 5 else challenger).commands(world.observe())
            actions = {c["actor"]: c["action"] for c in commands}
            record.append(world, actions, world.step(actions), commands)
    configuration = challenger.describe()
    configuration["lateral_intervention"] = {"applied": True, "actor": "b", "tick": 5}
    write_rollout(
        scene,
        Path(raw["case"]["rollout"]),
        record=record,
        snapshot=initial,
        controlled_ids=("a", "b"),
        policy="route-change",
        controller_configuration=configuration,
    )
    result = run(RepairStore(tmp_path / "ledger.sqlite"), raw, tmp_path / "run")
    assert result["status"] == "completed", result
    with np.load(tmp_path / "run/ordinary-v1.npz") as data:
        # Identical speed/headway means continuation must reproduce the challenger.
        np.testing.assert_array_equal(data["states"], np.asarray(record.states))
        np.testing.assert_array_equal(data["actions"], np.asarray(record.actions))
        diagnostics = json.loads(str(data["controller_diagnostics"]))
        assert diagnostics[:10] == record.diagnostics[:10]
        assert json.loads(str(data["metadata"]))["controller_configuration"] == json.loads(
            json.dumps(configuration)
        )
        assert diagnostics[10][0]["configured_desired_speed_mps"] == 8
    with np.load(tmp_path / "run/cautious-v1.npz") as data:
        diag = json.loads(str(data["controller_diagnostics"]))
        assert diag[10][0]["configured_desired_speed_mps"] == 3
        assert diag[10][1]["configured_desired_speed_mps"] == 8


def test_dead_gate_recovery_preserves_incumbent_and_cohort(straight_scene, tmp_path, monkeypatch):
    import driversi.rsi.correction.gate as gate_module

    store = RepairStore(tmp_path / "ledger.sqlite")
    raw = make_request(straight_scene, tmp_path / "dev")
    run(store, raw, tmp_path / "development")
    gate = gate_request(store, straight_scene, tmp_path, raw["baseline"], raw["recipes"][0])

    def interrupted(store, spec, output, *, role, gate_id, **kw):
        output.mkdir(parents=True)
        store.register_run(spec, output, role=role, gate_id=gate_id)
        (output / "budget.json").write_text(json.dumps({"admitted_agent_steps": 17}))
        raise KeyboardInterrupt("Simulated abrupt gate exit")

    monkeypatch.setattr(gate_module, "run_prepared", interrupted)
    with pytest.raises(KeyboardInterrupt):
        gate_module.run_gate(store, gate, tmp_path / "gate")
    with pytest.raises(ValueError, match="active"):
        store.recover_gate(gate["id"])
    with store.registry._connect() as db:
        for table in ("repair_gates", "repair_runs"):
            for row in db.execute("SELECT id,payload FROM " + table).fetchall():
                state = json.loads(row[1])
                if state["status"] in ("registered", "running"):
                    state["owner"]["start"] = "already-ended"
                    db.execute(
                        "UPDATE " + table + " SET payload=? WHERE id=?", (json.dumps(state), row[0])
                    )
    result = store.recover_gate(gate["id"])
    assert result["status"] == "interrupted" and result["decision"]["outcome"] == "inconclusive"
    assert result["runs"][0]["admitted_agent_steps"] == 17
    assert result["runs"][0]["status"] == "interrupted"
    assert result["unstarted_cases"] == [gate["cases"][1]["id"]]
    assert store.registry.protected_groups() == {"heldout-b", "heldout-c"}
    assert store.recover_gate(gate["id"]) == result
    with store.registry._connect() as db:
        assert (
            db.execute("SELECT recipe_id FROM repair_libraries").fetchone()[0]
            == raw["baseline"]["id"]
        )


@pytest.mark.parametrize("kind", ["same-state-repair-v1", "two-stage-reactive-augmentation-v1"])
def test_unsupported_source_parameter_schedule_rejected(straight_scene, tmp_path, kind):
    raw = make_request(straight_scene, tmp_path / "data")
    with np.load(raw["case"]["rollout"]) as source:
        arrays = {k: source[k].copy() for k in source.files}
    meta = json.loads(str(arrays["metadata"]))
    meta["protocol"] = {
        "kind": kind,
        "actor": "b",
        "recipe": raw["recipes"][1],
        "phases": {
            "exploration_end_tick": 15,
            "continuation_start_tick": 15,
            "continuation_policy": "stateless-idm-route-rule-diagnostic-v1",
        },
        "continuation_configuration": meta["controller_configuration"]["parameters"],
    }
    arrays["metadata"] = json.dumps(meta)
    np.savez_compressed(raw["case"]["rollout"], **arrays)
    result = run(RepairStore(tmp_path / "ledger.sqlite"), raw, tmp_path / "run")
    assert result["status"] == "failed" and "Source protocol" in result["failure_reason"]
    assert not result["attempts"] and result["admitted_agent_steps"] == 0


def test_gate_registration_gap_recovery_creates_report(straight_scene, tmp_path, monkeypatch):
    from driversi.rsi.correction.gate import run_gate

    store = RepairStore(tmp_path / "ledger.sqlite")
    raw = make_request(straight_scene, tmp_path / "dev")
    run(store, raw, tmp_path / "development")
    gate = gate_request(store, straight_scene, tmp_path, raw["baseline"], raw["recipes"][0])
    output = tmp_path / "uncreated-gate"
    original_mkdir = Path.mkdir

    def crash_before_mkdir(path, *args, **kwargs):
        if path == output:
            raise KeyboardInterrupt("Crash after registration before mkdir")
        return original_mkdir(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", crash_before_mkdir)
        with pytest.raises(KeyboardInterrupt):
            run_gate(store, gate, output)
    assert not output.exists()
    state = store.get_gate(gate["id"])
    state["owner"]["start"] = "already-ended"
    with store.registry._connect() as db:
        db.execute("UPDATE repair_gates SET payload=? WHERE id=?", (json.dumps(state), gate["id"]))
    recovered = store.recover_gate(gate["id"])
    assert recovered["status"] == "interrupted" and recovered["runs"] == []
    assert len(recovered["unstarted_cases"]) == 2
    assert json.loads((output / "gate.json").read_text()) == recovered
    assert store.recover_gate(gate["id"]) == recovered
