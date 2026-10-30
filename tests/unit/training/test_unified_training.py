"""Test-only native backend fixtures; no model quality or learning claim."""

import json
from pathlib import Path

import pytest
import yaml

from driversi.training.contracts.job import resolve_request, validate_request
from driversi.training.orchestration import engine

ROOT = Path(__file__).resolve().parents[3]


def request(path, model="augustus", mode="supervised"):
    config = path / "model.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": model,
                "family": "test_family",
                "settings": {},
                "paths": {"dataset_manifest": "missing", "training_manifest": "missing"},
            }
        )
    )
    data = dict(
        schema_version=1,
        id="fixture",
        model_config="model.yaml",
        mode=mode,
        initialization="model_weights",
        budget=dict(
            max_wall_seconds=10,
            max_optimizer_steps=3,
            max_environment_steps=2 if mode == "online_rl" else 0,
            max_scoring_agent_steps=0,
        ),
    )
    p = path / "request.json"
    p.write_text(json.dumps(data))
    return p


def test_modes_and_resume_do_not_silently_convert(tmp_path):
    p = request(tmp_path, "driverl", "online_rl")
    assert resolve_request(p)["reason"] == "native_learning_backend_not_connected"
    value = json.loads(p.read_text())
    value["initialization"] = "exact_resume"
    with pytest.raises(ValueError, match="resume"):
        validate_request(value)
    value["initialization"] = "model_weights"
    value["mode"] = "supervised"
    with pytest.raises(ValueError, match="environment"):
        validate_request(value)






@pytest.mark.parametrize("model", ["augustus"])
@pytest.mark.parametrize("steps", [2, 4])
def test_common_native_receipt_and_no_promotion(tmp_path, monkeypatch, model, steps):
    p = request(tmp_path, model)
    monkeypatch.setattr(engine.native, "project_root", lambda: tmp_path)
    output = tmp_path / "exp/job"
    checkpoint = tmp_path / "exp/checkpoints" / model
    protocol = (
        "augustus_native_sft"
        if model == "augustus"
        else "wa_jepa_native_navsim_stage2_original_logs"
    )

    def prepare(cfg, operation, out):
        value = yaml.safe_load(Path(cfg).read_text())
        assert value["settings"]["max_optimizer_steps"] == 3
        return dict(
            protocol=protocol, missing=[], ready_to_attempt=True, checkpoint_output=str(checkpoint)
        )

    def execute(plan, **kw):
        assert kw["timeout_seconds"] == 10
        folder = output / "native"
        folder.mkdir()
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights.fixture").write_text("TEST_ONLY")
        info = {"checkpoint": str(checkpoint), "updates": steps, "native_global_steps": steps}
        (
            folder / ("candidate.json" if model == "augustus" else "training-receipt.json")
        ).write_text(json.dumps(info))
        return dict(status="completed", reserved_gpu_seconds=1)

    monkeypatch.setattr(engine.native, "prepare", prepare)
    monkeypatch.setattr(engine.native, "execute", execute)
    engine.prepare(p, output)
    result = engine.run(output / "training-plan.json")
    assert result["status"] == ("completed" if steps <= 3 else "failed")
    assert bool(result["checkpoint"]) is (steps <= 3)
    assert result["consumed"]["successful_optimizer_steps"] is None
    assert "skipped_batches" in result["counter_semantics"]
    assert not result["promoted"] and not result["policy_activated"]
    with pytest.raises(ValueError, match="already exists"):
        engine.run(output / "training-plan.json")


def test_unsupported_native_learner_never_starts_process(tmp_path, monkeypatch):
    p = request(tmp_path, "driverl", "online_rl")
    monkeypatch.setattr(engine.native, "project_root", lambda: tmp_path)
    monkeypatch.setattr(engine.native, "execute", lambda *a, **k: pytest.fail("must not launch"))
    output = tmp_path / "exp/no-learner"
    plan = engine.prepare(p, output)
    assert not plan["ready_to_attempt"]
    result = engine.run(output / "training-plan.json")
    assert result["status"] == "rejected" and result["checkpoint"] is None


@pytest.mark.parametrize("intervene", [False, True])
def test_online_round_keeps_world_and_exports_native_candidate(tmp_path, monkeypatch, intervene):
    from driversi.training.online_rl.online import RolloutSession
    from tests.unit.training.test_online_learning import Actor, Env, Learner

    class NativeActor(Actor):
        model_id = "driverl"

    class NativeLearner(Learner):
        model_id = "driverl"
        implementation_version = "TEST_ONLY_NATIVE_LEARNER"

        def update(self, batch, *, max_optimizer_steps):
            self.calls += 1
            assert len(batch.transitions) == 2 and batch.boundary_memory == 2
            path = tmp_path / "exp/checkpoints/fixture"
            path.mkdir(parents=True)
            (path / "fixture.pt").write_text("TEST_ONLY")
            return {"candidate": str(path), "optimizer_steps": max_optimizer_steps}

    p = request(tmp_path, "driverl", "online_rl")
    monkeypatch.setattr(engine.native, "project_root", lambda: tmp_path)
    env = Env(intervene=intervene)
    actor = NativeActor()
    learner = NativeLearner()
    session = RolloutSession(env, actor, memory=0, protected_logs=[])
    out = engine.run_online(
        p, tmp_path / "exp/online", session=session, learner=learner, chunk_steps=1
    )
    assert out["status"] == ("rejected" if intervene else "completed")
    assert learner.calls == (0 if intervene else 1)
    assert out["consumed"]["environment_steps"] == (1 if intervene else 2)
    assert env.state[0] == out["consumed"]["environment_steps"]
    assert actor.policy_version == "parent-v1" and not out["policy_activated"]
    assert bool(out["checkpoint"]) is (not intervene)
    assert out["initialization"] == "caller_supplied_live_actor_state"
    assert out["effective_parent"]["policy_version"] == actor.policy_version
    assert out["consumed"]["gpu_seconds"] is None


def test_native_world_sequence_uses_executed_actions_and_hides_future():
    import numpy as np

    from driversi.training.contracts.samples import sequence_from_batch
    from tests.unit.training.test_online_learning import Env, session

    batch = session().collect(max_environment_steps=4, max_wall_seconds=10)
    seq = sequence_from_batch(
        batch,
        timestamps_ns=np.arange(5, dtype=np.int64) * 100000000,
        branch_id="actual-branch",
        action_contract="native_continuous_fixture",
        context_frames=2,
    )
    assert len(seq.policy_inputs()["observations"]) == 2
    assert "future_executed_actions" not in seq.policy_inputs()
    assert len(seq.world_inputs()["future_executed_actions"]) == 3
    assert len(seq.targets()["observations"]) == 3
    interrupted = session(Env(intervene=True)).collect(max_environment_steps=2, max_wall_seconds=10)
    world = sequence_from_batch(
        interrupted,
        timestamps_ns=np.array([0, 100000000]),
        branch_id="overridden-branch",
        action_contract="native_continuous_fixture",
        context_frames=1,
    )
    assert np.array_equal(world.world_inputs()["future_executed_actions"][0], [0.0, 0.0])
    assert world.targets()["interventions"] == ("safety_override",)
    with pytest.raises(ValueError, match="timestamps"):
        sequence_from_batch(
            batch,
            timestamps_ns=np.zeros(5, dtype=np.int64),
            branch_id="b",
            action_contract="native",
            context_frames=2,
        )


def test_harness_unified_provider_keeps_native_supervision_and_rejects_missing_learner(
    tmp_path, monkeypatch
):
    from driversi.rsi.workflow.providers import builtin_providers

    p = request(tmp_path, "driverl", "online_rl")
    monkeypatch.setattr(engine.native, "project_root", lambda: tmp_path)
    provider = builtin_providers().get("native_model_unified_training")
    assert provider.name.startswith("native_model_")
    config, cap = provider.prepare({"request": str(p)}, tmp_path, tmp_path / "registry")
    assert cap == 2
    result = provider.execute(config, {}, tmp_path / "exp/harness", tmp_path / "registry")
    assert result["status"] == "rejected" and result["sim_agent_steps"] == 0
    assert Path(result["artifacts"]["report"]).is_file()
    value = json.loads(p.read_text())
    value["budget"]["max_optimizer_steps"] += 1
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed since registration"):
        provider.execute(config, {}, tmp_path / "exp/changed", tmp_path / "registry")


def test_inventory_lists_retained_training_config():
    from driversi.training.contracts.protocols import inventory

    rows = {r["model"]: r for r in inventory()["models"]}
    assert rows["augustus"]["training_modes"]["supervised"]
    assert rows["augustus"]["training_configs"]
    assert not rows["driverl"]["training_modes"]["online_rl"]


@pytest.mark.parametrize("where", ["inference", "environment", "learner"])
def test_online_partial_progress_and_unknown_update_are_preserved(tmp_path, monkeypatch, where):
    from driversi.training.online_rl.online import RolloutSession
    from tests.unit.training.test_online_learning import Actor, Env, Learner

    class NativeActor(Actor):
        model_id = "driverl"

        def act(self, observation, memory):
            if where == "inference" and memory == 1:
                raise RuntimeError("inference failed")
            return super().act(observation, memory)

    class NativeEnv(Env):
        def step(self, action):
            result = super().step(action)
            if where == "environment" and self.state[0] == 2:
                raise RuntimeError("response lost after advance")
            return result

    class NativeLearner(Learner):
        model_id = "driverl"
        implementation_version = "TEST_ONLY"

        def update(self, *a, **kw):
            self.calls += 1
            raise RuntimeError("native update result lost")

    p = request(tmp_path, "driverl", "online_rl")
    monkeypatch.setattr(engine.native, "project_root", lambda: tmp_path)
    learner = NativeLearner()
    session = RolloutSession(NativeEnv(), NativeActor(), memory=0, protected_logs=[])
    result = engine.run_online(
        p, tmp_path / "exp/partial", session=session, learner=learner, chunk_steps=2
    )
    counts = result["consumed"]
    assert counts["environment_steps"] == (1 if where == "inference" else 2)
    assert counts["confirmed_transitions"] == (2 if where == "learner" else 1)
    assert counts["unknown_environment_outcomes"] == (1 if where == "environment" else 0)
    assert result["status"] == ("failed" if where == "inference" else "outcome_unknown")
    assert learner.calls == (1 if where == "learner" else 0)
    assert counts["successful_optimizer_steps"] == (None if where == "learner" else 0)
    if where == "environment":
        with pytest.raises(ValueError, match="outcome unknown"):
            session.collect(max_environment_steps=1, max_wall_seconds=10)


@pytest.mark.parametrize("slow_stage", ["revalidation", "execution"])
def test_total_native_wall_budget_cannot_report_completed(tmp_path, monkeypatch, slow_stage):
    p = request(tmp_path)
    monkeypatch.setattr(engine.native, "project_root", lambda: tmp_path)
    output = tmp_path / "exp/wall"
    checkpoint = tmp_path / "exp/checkpoints/native"
    monkeypatch.setattr(
        engine.native,
        "prepare",
        lambda *a, **k: dict(
            protocol="augustus_native_sft",
            missing=[],
            ready_to_attempt=True,
            checkpoint_output=str(checkpoint),
        ),
    )
    now = [0.0]
    monkeypatch.setattr(engine.time, "monotonic", lambda: now[0])

    def execute(plan, **kw):
        assert kw["deadline_monotonic"] == 10
        now[0] = 11.0  # verification and artifact work exceeded the declared common deadline
        folder = output / "native"
        folder.mkdir()
        checkpoint.mkdir(parents=True)
        (checkpoint / "fixture.pt").write_text("TEST_ONLY")
        (folder / "candidate.json").write_text(
            json.dumps({"updates": 2, "checkpoint": str(checkpoint)})
        )
        return dict(status="completed", reserved_gpu_seconds=1)

    monkeypatch.setattr(engine.native, "execute", execute)
    engine.prepare(p, output)
    original = engine.resolve_request

    def resolve(path):
        if slow_stage == "revalidation":
            now[0] = 11.0
        return original(path)

    monkeypatch.setattr(engine, "resolve_request", resolve)
    result = engine.run(output / "training-plan.json")
    assert result["status"] == "budget_exhausted" and not result["promoted"]
    assert result["checkpoint"]  # retain artifacts even though this attempt exceeded budget
