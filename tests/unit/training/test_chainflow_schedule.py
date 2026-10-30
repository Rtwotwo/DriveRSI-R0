"""Native schedule cadence and sealed incremental recovery contracts."""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from driversi_agents.chainflow.training import _training_recipe, _validate_recovery


def config():
    return SimpleNamespace(
        scheduler_args=SimpleNamespace(dataset_size=85000, num_epochs=25),
        lr_args={"name": "AdamW", "base_lr": 2e-4, "base_batch_size": 64},
    )


def recipe(**overrides):
    return _training_recipe(config(), {"training_recipe": overrides}, 1)


def state(value, updates=2):
    return {
        "recovery_schema_version": 2,
        "context": "same-plan",
        "effective_training_recipe": value,
        "total_updates": updates,
        "cursor": updates,
        "scheduler": {"last_epoch": updates},
        "sample_trace": [
            {"optimizer_step": i + 1, "sample_ids": ["sample"], "native_scheduler_step_before": i}
            for i in range(updates)
        ],
    }


def test_default_preserves_reference_horizon_and_exposes_fp32_difference():
    value = recipe()
    assert value["scheduler"]["total_updates"] == 2125000
    assert value["scheduler"]["warmup_updates"] == 212500
    assert value["optimizer"]["scaled_base_lr"] == 2.5e-5
    assert value["incremental_schedule_override"] is False
    assert value["precision"] == "32-true"
    assert value["precision_matches_native_default"] is False
    assert value["native_full_recipe_reproduction"] is False


def test_explicit_schedule_is_plan_hyperparameter_not_claimed_pool_size():
    cfg = config()
    value = _training_recipe(cfg, {"training_recipe": {"scheduler_dataset_size": 2, "scheduler_num_epochs": 10}}, 1)
    assert cfg.scheduler_args.dataset_size == 2
    assert cfg.scheduler_args.num_epochs == 10
    assert value["scheduler"]["total_updates"] == 20
    assert value["scheduler"]["warmup_updates"] == 2
    assert "not_measured_sample_count" in value["scheduler"]["size_semantics"]
    assert value["incremental_schedule_override"] is True


@pytest.mark.parametrize("bad", [
    {"scheduler_dataset_size": True}, {"scheduler_dataset_size": 0},
    {"scheduler_num_epochs": 1.2}, {"scheduler_dataset_size": "manifest"},
    {"loss_weight": 0}, {"precision": "16-mixed"},
    {"scheduler_dataset_size": 1, "scheduler_num_epochs": 1},
])
def test_no_silent_recipe_approximation(bad):
    with pytest.raises(ValueError):
        recipe(**bad)


def test_recovery_requires_exact_runtime_recipe_and_cursor():
    value = recipe(scheduler_dataset_size=2, scheduler_num_epochs=10)
    checkpoint = state(value)
    rows = [{"sample_id": "sample"}]
    _validate_recovery(checkpoint, "same-plan", value, rows)
    for key, replacement in (("recovery_schema_version", None), ("cursor", 1), ("context", "other"), ("scheduler", {"last_epoch": 1})):
        bad = copy.deepcopy(checkpoint)
        bad[key] = replacement
        with pytest.raises(ValueError):
            _validate_recovery(bad, "same-plan", value, rows)
    other_recipe = recipe(scheduler_dataset_size=2, scheduler_num_epochs=11)
    with pytest.raises(ValueError, match="recipe"):
        _validate_recovery(checkpoint, "same-plan", other_recipe, rows)
    with pytest.raises(ValueError, match="sample order"):
        _validate_recovery(checkpoint, "same-plan", value, [{"sample_id": "other"}])


def test_native_step_schedule_values_and_resume():
    torch = pytest.importorskip("torch")
    # Execute the owned released method itself without importing heavyweight
    # model dependencies. No copied optimizer or scheduler implementation.
    root = Path(__file__).resolve().parents[3]
    source = root / "agents/chainflow/native/navsim/agents/chainflow_vla/chainflow_vla_agent.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_optimizers")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {"torch": torch, "math": math, "_get_scheduler_arg": lambda a, k: a[k]}
    exec(compile(module, str(source), "exec"), scope)

    def create():
        model = torch.nn.Linear(1, 1)
        native = SimpleNamespace(
            batch_size=1, num_gpus=1,
            _lr_args=config().lr_args,
            scheduler_args={"dataset_size": 2, "num_epochs": 10},
            _chainflow_vla_model=model,
        )
        optimizers, schedules = scope["get_optimizers"](native)
        assert schedules[0]["interval"] == "step"
        return model, optimizers[0], schedules[0]["scheduler"]

    def advance(model, opt, sched, n):
        lrs = []
        for _ in range(n):
            lrs.append(opt.param_groups[0]["lr"])
            opt.zero_grad()
            (model(torch.ones(1, 1)) ** 2).mean().backward()
            opt.step()
            sched.step()
        return lrs

    model, opt, sched = create()
    first = advance(model, opt, sched, 4)
    saved = copy.deepcopy((model.state_dict(), opt.state_dict(), sched.state_dict()))
    expected_lrs = advance(model, opt, sched, 4)
    expected = copy.deepcopy(model.state_dict())
    replay_model, replay_opt, replay_sched = create()
    replay_model.load_state_dict(saved[0])
    replay_opt.load_state_dict(saved[1])
    replay_sched.load_state_dict(saved[2])
    assert advance(replay_model, replay_opt, replay_sched, 4) == expected_lrs
    assert all(torch.equal(expected[k], replay_model.state_dict()[k]) for k in expected)
    assert first[:3] == pytest.approx([2.5e-11, 1.25000125e-5, 2.5e-5])
    assert replay_sched.last_epoch == 8


def test_cache_preflight_cost_reserved_before_verification(tmp_path, monkeypatch):
    import json

    from driversi_agents.chainflow import training

    budget = tmp_path / "budget.json"
    rows = [{"kind": "synthetic_navtrain"}, {"kind": "original_navtrain"}]
    observed = []
    def verify(value):
        observed.append(json.loads(budget.read_text()))
        assert value == rows
        return 40
    monkeypatch.setattr(training, "verify_records", verify)
    assert training._verify_inputs(rows, budget, 40) == 40
    assert observed[0]["reserved_agent_steps"] == 40
    assert observed[0]["cache_validation_agent_steps"] == 40
    with pytest.raises(ValueError, match="exceeds"):
        training._verify_inputs(rows, budget, 39)
    assert len(observed) == 1
