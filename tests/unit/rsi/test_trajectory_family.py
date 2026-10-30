import copy
import json

import numpy as np
import pytest

from driversi.rsi.decisions.trajectory_diversity import compare, distances
from driversi.rsi.experiments.reactive_cohort import expand_plan, validate_budget


def configuration():
    return dict(
        trajectory_family=dict(
            target_per_anchor=10, attempts_per_anchor=24, minimum_ade_m=0.5, seed_start=42
        ),
        max_samples=20,
        max_rendered_branches=24,
        max_candidates=2,
    )


def descriptor(name, offset=0):
    xy = np.stack([np.arange(71) * 0.1, np.linspace(0, offset, 71)], axis=1)
    return dict(candidate_id=name, comparison_key={"source": "same"}, xy=xy.tolist())


def test_full_and_future_diversity_require_same_physical_coordinates():
    a, b = descriptor("a"), descriptor("b", 2)
    assert compare(b, [a])["distinct"]
    assert not compare(a, [a])["distinct"]
    assert not compare(descriptor("tiny", 0.2), [a])["distinct"]
    b["comparison_key"] = {"source": "other"}
    with pytest.raises(ValueError):
        compare(b, [a])
    with pytest.raises(ValueError):
        distances(np.zeros((70, 2)), a["xy"])
    b = descriptor("future_same")
    b["xy"][:20] = (np.asarray(b["xy"][:20]) + 100).tolist()
    assert not compare(b, [a])["distinct"]


def test_family_is_deterministic_but_does_not_certify_distinctness():
    a = dict(id="a", skill="driverl_full_window", token="train", recipe={"version": 1})
    original = dict(assignments=[a], plan_id="old", context={})
    plan = expand_plan(original, configuration()["trajectory_family"])
    assert len(plan["assignments"]) == 24
    assert len({x["id"] for x in plan["assignments"]}) == 24
    assert plan["assignments"][1]["sampling_seed"] == 43
    assert all("distinct" not in x for x in plan["assignments"])
    assert original["assignments"][0] == a
    assert plan == expand_plan(original, configuration()["trajectory_family"])
    a["skill"] = "reactive_new_history"
    with pytest.raises(ValueError):
        expand_plan(original, configuration()["trajectory_family"])


def test_native_search_seed_declaration():
    from driversi_agents.driverl.search import search_window

    assert search_window(8, 0, 70, seed=77)["seed"] == 77
    for seed in [True, -1, 2**31, 0.5]:
        with pytest.raises(ValueError):
            search_window(8, 0, 70, seed=seed)


def test_invalid_budgets_do_not_lower_acceptance():
    c = configuration()
    assert validate_budget(c)["target_per_anchor"] == 10
    for key, value in [
        ("attempts_per_anchor", 9),
        ("minimum_ade_m", 0),
        ("target_per_anchor", True),
    ]:
        bad = copy.deepcopy(c)
        bad["trajectory_family"][key] = value
        with pytest.raises(ValueError):
            validate_budget(bad)


@pytest.mark.parametrize("diagnostic_failure", [False, True])
def test_duplicate_suppression_and_cross_process_accounting(
    tmp_path, monkeypatch, diagnostic_failure
):
    from driversi.rsi.experiments import reactive_cohort as mod
    from driversi.rsi.experiments import reactive_generation, reactive_postprocess
    from driversi.runtime.budget import write_json

    c = configuration()
    c["trajectory_family"]["attempts_per_anchor"] = 10
    c["trajectory_family"]["target_per_anchor"] = 10
    c["train_catalog"] = "training-catalog"
    config = tmp_path / "config.json"
    write_json(config, c)
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        dict(
            assignments=[dict(id="a", skill="driverl_full_window", token="train")],
            plan_id="old",
            context={},
        ),
    )
    root = tmp_path / "campaign"
    root.mkdir()
    calls = []

    def generate(plan_path, cid, asset, database, out, **kw):
        calls.append((cid, kw.get("render", True)))
        out.mkdir(parents=True, exist_ok=True)
        if kw.get("render", True):
            write_json(out / "render-reservation.json", {})
            return dict(status="rendered")
        return dict(status="trajectory_ready", candidate_id=cid)

    def describe(generated, cid):
        d = descriptor(cid)
        d.update(behavior_types=["unclassified"], files={})
        return d

    monkeypatch.setattr(reactive_generation, "generate", generate)
    monkeypatch.setattr(
        reactive_postprocess, "process", lambda *a: dict(status="visual_rejected", trainable=False)
    )
    monkeypatch.setattr(mod, "describe", describe)
    if diagnostic_failure:

        def failed_coverage(*args):
            raise ValueError("Invalid diagnostic reference frame")

        monkeypatch.setattr(mod, "family_coverage", failed_coverage)
    selection = dict(plan=str(plan), assets=[{}])
    assert mod.produce(selection, config, root, 1e20, lambda *a, **kw: None) == []
    assert len(calls) == 11 and sum(render for _, render in calls) == 1
    summary = json.loads((root / "trajectory-family-summary.json").read_text())
    assert summary["counts"]["trajectory_duplicate_before_render"] == 9
    assert summary["anchors"][0]["qualified"] == 0
    assert summary["anchors"][0]["shortage"] == 10
    if diagnostic_failure:
        assert summary["anchors"][0]["coverage_diagnostics"]["status"] == "diagnostic_failed"
    calls.clear()
    mod.produce(selection, config, root, 1e20, lambda *a, **kw: None)
    assert calls == []
    resumed = json.loads((root / "trajectory-family-summary.json").read_text())
    assert resumed == summary


def test_teacher_only_generation_never_reserves_or_renders(tmp_path, monkeypatch):
    from driversi.rsi.experiments import reactive_generation as mod
    from driversi.rsi.experiments import skill_execution
    from driversi.runtime.budget import write_json

    monkeypatch.setattr(mod, "project_root", lambda: tmp_path)
    plan = tmp_path / "plan.json"
    write_json(
        plan, dict(assignments=[dict(id="a", token="t", source_log="log")], selection_role="train")
    )
    monkeypatch.setattr(
        skill_execution, "run_assignment", lambda *a, **k: dict(status="trajectory_admitted")
    )
    monkeypatch.setattr(
        mod, "task_state", lambda *a, **k: pytest.fail("Teacher-only phase must not render")
    )
    out = tmp_path / "exp/attempt"
    result = mod.generate(
        plan,
        "a",
        dict(token="t", source_log="log", bundle_root="asset"),
        tmp_path / "catalog",
        out,
        render=False,
    )
    assert result["status"] == "trajectory_ready"
    assert not (out / "render-reservation.json").exists()


def test_pair_budget_covers_two_passes_without_dropping_tail():
    from driversi.rsi.experiments.reactive_pair import sampling_budget

    for originals, synthetic in [(1, 10), (1, 32), (2, 20), (2, 21)]:
        batch, steps = sampling_budget(originals, synthetic, family=True, requested_steps=8)
        assert originals % batch == (originals + synthetic) % batch == 0
        assert steps * batch > originals + synthetic
        assert steps <= 64


def test_positive_lr_exposure_checked_before_evaluation(tmp_path):
    from driversi.rsi.experiments.reactive_pair import verify_training_exposure
    from driversi.runtime.budget import write_json

    samples = []
    for name in ("a", "b"):
        p = tmp_path / (name + ".json")
        write_json(p, dict(sample_id=name))
        samples.append(str(p))
    root = tmp_path / "train"
    for phase, rate, sample in [("train", 0.0, "a"), ("resume", 0.001, "b")]:
        d = root / phase
        d.mkdir(parents=True)
        write_json(d / "training-result.json", dict(updates=1))
        (d / "consumed-training-samples.jsonl").write_text(
            json.dumps(dict(sample_ids=[sample], learning_rates=[rate])) + "\n"
        )
    value = verify_training_exposure(
        dict(samples=samples), dict(treatment=dict(directory=str(root)))
    )
    assert not value["passed"] and value["missing"] == ["a"]


def test_deadline_preserved_without_training_fallthrough(tmp_path):
    from driversi.rsi.experiments.reactive_cohort import produce
    from driversi.runtime.budget import write_json

    config = tmp_path / "config.json"
    write_json(config, configuration())
    plan = tmp_path / "plan.json"
    write_json(
        plan, dict(assignments=[dict(id="a", skill="driverl_full_window", token="t")], context={})
    )
    root = tmp_path / "out"
    root.mkdir()
    with pytest.raises(TimeoutError):
        produce(dict(plan=str(plan), assets=[{}]), config, root, 0, lambda *a, **k: None)
    assert (
        json.loads((root / "trajectory-family-summary.json").read_text())["status"]
        == "wall_budget_exhausted"
    )


def test_unsettled_teacher_is_reconciled_without_terminal_failure(tmp_path, monkeypatch):
    from driversi.rsi.experiments import reactive_cohort as mod
    from driversi.rsi.experiments import reactive_generation
    from driversi.runtime.budget import write_json

    c = configuration()
    c["trajectory_family"]["attempts_per_anchor"] = 10
    c["train_catalog"] = "train"
    config = tmp_path / "config.json"
    write_json(config, c)
    plan = tmp_path / "plan.json"
    write_json(
        plan, dict(assignments=[dict(id="a", skill="driverl_full_window", token="t")], context={})
    )
    root = tmp_path / "out"
    root.mkdir()
    selection = dict(plan=str(plan), assets=[{}])
    monkeypatch.setattr(
        reactive_generation, "generate", lambda *a, **k: dict(status="teacher_outcome_unknown")
    )
    with pytest.raises(mod.PendingGeneration):
        mod.produce(selection, config, root, 1e20, lambda *a, **k: None)
    assert (root / "candidates/a-v000/pending.json").exists()
    assert not (root / "candidates/a-v000/outcome.json").exists()
    calls = []

    def settled(*a, **k):
        calls.append(a[1])
        return dict(status="teacher_rejected")

    monkeypatch.setattr(reactive_generation, "generate", settled)
    mod.produce(selection, config, root, 1e20, lambda *a, **k: None)
    assert calls[0] == "a-v000" and len(calls) == 10
    assert (root / "candidates/a-v000/outcome-seal.json").exists()


def test_pending_precoverage_receipt_resumes_without_rewriting(tmp_path, monkeypatch):
    from driversi.rsi.experiments import reactive_cohort as mod
    from driversi.rsi.experiments import reactive_generation, reactive_postprocess
    from driversi.runtime.budget import write_json

    c = configuration()
    c["trajectory_family"]["attempts_per_anchor"] = 10
    c["train_catalog"] = "train"
    config = tmp_path / "config.json"
    write_json(config, c)
    plan = tmp_path / "plan.json"
    write_json(
        plan, dict(assignments=[dict(id="a", skill="driverl_full_window", token="t")], context={})
    )
    root = tmp_path / "out"
    root.mkdir()
    selection = dict(plan=str(plan), assets=[{}])
    updated = [False]

    def generate(plan_path, cid, asset, database, out, **kw):
        if not kw.get("render", True):
            return dict(status="trajectory_ready")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "render-reservation.json", {})
        return dict(status="rendered" if updated[0] else "render_running")

    def describe(generated, cid):
        d = descriptor(cid)
        d.update(behavior_types=["unclassified"], files={})
        if updated[0]:
            d["initial_heading_rad"] = 0.0
        return d

    monkeypatch.setattr(reactive_generation, "generate", generate)
    monkeypatch.setattr(mod, "describe", describe)
    monkeypatch.setattr(
        reactive_postprocess, "process", lambda *a: dict(status="visual_rejected", trainable=False)
    )
    with pytest.raises(mod.PendingGeneration):
        mod.produce(selection, config, root, 1e20, lambda *a, **kw: None)
    evidence = root / "candidates/a-v000/diversity.json"
    old = evidence.read_bytes()
    updated[0] = True
    mod.produce(selection, config, root, 1e20, lambda *a, **kw: None)
    assert evidence.read_bytes() == old
    assert (
        json.loads((root / "candidates/a-v000/outcome.json").read_text())["status"]
        == "visual_rejected"
    )
    summary = json.loads((root / "trajectory-family-summary.json").read_text())
    assert summary["counts"] == {"visual_rejected": 1, "trajectory_duplicate_before_render": 9}
    assert (
        summary["anchors"][0]["coverage_diagnostics"]["status"]
        == "unavailable_missing_initial_heading"
    )


def test_exploration_budget_does_not_claim_every_anchor_target_is_reserved():
    c = configuration()
    c["max_candidates"] = 16
    with pytest.raises(ValueError):
        validate_budget(c)
    c["trajectory_family"]["anchor_budget_mode"] = "explore_until_global_limit"
    c["trajectory_family"]["stop_after_consecutive_duplicates"] = 3
    assert validate_budget(c)["target_per_anchor"] == 10
    c["trajectory_family"]["stop_after_consecutive_duplicates"] = True
    with pytest.raises(ValueError):
        validate_budget(c)


def test_stop_policy_routes_budget_without_certifying_samples():
    from driversi.rsi.experiments.reactive_cohort import family_stop_reason

    d = dict(status="trajectory_duplicate_before_render", trainable=False)
    policy = dict(stop_after_consecutive_duplicates=3, stop_after_visual_failures=1)
    assert family_stop_reason([d, d], policy) is None
    assert family_stop_reason([d, d, d], policy) == "consecutive_non_novel_native_rollouts"
    assert family_stop_reason([d, dict(status="teacher_rejected"), d], policy) is None
    assert (
        family_stop_reason([dict(status="visual_preflight_rejected")], policy)
        == "visual_repair_required"
    )
    assert family_stop_reason([d] * 20, {}) is None


def test_stopped_family_settles_pending_and_rotates_other_anchor(tmp_path, monkeypatch):
    from driversi.rsi.experiments import reactive_cohort as mod
    from driversi.rsi.experiments import reactive_generation, reactive_postprocess
    from driversi.runtime.budget import write_json

    c = configuration()
    c["trajectory_family"].update(attempts_per_anchor=10, stop_after_visual_failures=1)
    c.update(train_catalog="train", production_gpu=7)
    config = tmp_path / "config.json"
    write_json(config, c)
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        dict(
            assignments=[dict(id=k, skill="driverl_full_window", token=k) for k in ("a", "b")],
            context={},
        ),
    )
    root = tmp_path / "out"
    root.mkdir()
    selection = dict(plan=str(plan), assets=[{}, {}])
    settle = [False]
    calls = []

    def generate(plan_path, cid, asset, database, out, **kw):
        assert kw["gpu"] == 7
        calls.append((cid, kw.get("render", True)))
        if cid == "b-v000" and not settle[0]:
            return dict(status="teacher_outcome_unknown")
        if cid != "a-v000":
            return dict(status="teacher_rejected")
        if not kw.get("render", True):
            return dict(status="trajectory_ready")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "render-reservation.json", {})
        return dict(status="rendered")

    def describe(generated, cid):
        d = descriptor(cid)
        d.update(behavior_types=["unclassified"], files={})
        return d

    monkeypatch.setattr(reactive_generation, "generate", generate)
    monkeypatch.setattr(mod, "describe", describe)
    monkeypatch.setattr(
        reactive_postprocess, "process", lambda *a: dict(status="visual_rejected", trainable=False)
    )
    with pytest.raises(mod.PendingGeneration):
        mod.produce(selection, config, root, 1e20, lambda *a, **kw: None)
    # An already-owned task must settle even if its family has since hit its stop budget.
    pending = root / "candidates/a-v001/pending.json"
    pending.parent.mkdir(parents=True)
    write_json(pending, dict(status="teacher_outcome_unknown"))
    calls.clear()
    settle[0] = True
    assert mod.produce(selection, config, root, 1e20, lambda *a, **kw: None) == []
    assert ("a-v001", False) in calls
    assert not any(cid in {"a-v000", "a-v002"} for cid, _ in calls)
    assert sum(cid.startswith("b-") for cid, _ in calls) == 10
    summary = json.loads((root / "trajectory-family-summary.json").read_text())
    assert summary["attempted"] == 12 and summary["rendered"] == 1
    assert summary["anchors"][0]["search_stop_reason"] == "visual_repair_required"
    assert summary["qualified"] == 0 and not summary["target_met"]
    calls.clear()
    mod.produce(selection, config, root, 1e20, lambda *a, **kw: None)
    assert calls == []
    assert json.loads((root / "trajectory-family-summary.json").read_text()) == summary


def test_started_family_settles_when_only_language_request_budget_remains(tmp_path, monkeypatch):
    from driversi.rendering.diagnostics import gs_sequence_critic
    from driversi.rsi.experiments import reactive_cohort as mod
    from driversi.rsi.experiments import reactive_generation
    from driversi.runtime.budget import write_json

    c = configuration()
    c.update(
        train_catalog="train",
        review_budget_preflight=True,
        critic_config="private",
        review_ledger="ledger",
    )
    c["trajectory_family"]["attempts_per_anchor"] = 10
    config = tmp_path / "config.json"
    write_json(config, c)
    plan = tmp_path / "plan.json"
    write_json(
        plan, dict(assignments=[dict(id="a", skill="driverl_full_window", token="t")], context={})
    )
    root = tmp_path / "campaign"
    pending = root / "candidates/a-v000/pending.json"
    pending.parent.mkdir(parents=True)
    write_json(pending, dict(status="teacher_outcome_unknown"))
    calls = []

    def generate(*args, **kwargs):
        calls.append(args[1])
        return dict(status="teacher_rejected")

    monkeypatch.setattr(reactive_generation, "generate", generate)
    monkeypatch.setattr(
        gs_sequence_critic, "review_capacity", lambda *a: dict(sufficient=False, remaining=1)
    )
    assert (
        mod.produce(dict(plan=str(plan), assets=[{}]), config, root, 1e20, lambda *a, **k: None)
        == []
    )
    assert calls == ["a-v000"]
    assert (pending.parent / "outcome-seal.json").exists()
    assert (
        json.loads((root / "trajectory-family-summary.json").read_text())["status"]
        == "review_budget_exhausted"
    )


def test_expansion_policy_reserves_all_seven_anchor_targets():
    from driversi.rsi.experiments.reactive_cohort import validate_budget

    c = dict(
        student_model="chainflow_stage1",
        visual_quality_profile="experimental_numeric24_ade03_v1",
        max_candidates=7,
        max_samples=70,
        max_rendered_branches=140,
        trajectory_family=dict(
            target_per_anchor=10,
            attempts_per_anchor=64,
            minimum_ade_m=0.3,
            seed_start=42,
            anchor_budget_mode="reserve_all_targets",
        ),
    )
    assert validate_budget(c)["minimum_ade_m"] == 0.3
    c["max_samples"] = 20
    with pytest.raises(ValueError, match="Global sample quota"):
        validate_budget(c)
    c["max_samples"] = 70
    c["visual_quality_profile"] = "experimental_numeric25_v1"
    with pytest.raises(ValueError, match="threshold differs"):
        validate_budget(c)
