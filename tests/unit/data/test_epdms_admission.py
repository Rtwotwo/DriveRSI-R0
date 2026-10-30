import copy
import json
from pathlib import Path

import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.correction import epdms_admission as gate


def score(value=1.0):
    return dict(kind="executed_counterfactual_epdms", protocol=gate.PROTOCOL,
        score_unit="0_to_1", native_metric_math_unmodified=True, logged_benchmark=False,
        actual_world_seconds=5.0, scored_seconds=4.0, history_origin="actual_generated_world",
        observation_origin="actual_executed_counterfactual_world", original_human_filter_applied=False,
        two_frame_extended_comfort=1.0,
        two_frame_comfort_policy="required_native_adjacent_plans",
        score=value, strict_threshold=0.95, passed=value > 0.95,
        components=dict(no_at_fault_collisions=1.0, drivable_area_compliance=1.0,
            driving_direction_compliance=1.0, traffic_light_compliance=1.0,
            ego_progress=(value * 16 - 11) / 5, time_to_collision_within_bound=1.0,
            lane_keeping=1.0, history_comfort=1.0, two_frame_extended_comfort=1.0))


def test_strict_per_trajectory_not_mean_or_pdms():
    assert gate.validate_score(score(0.96))["score"] == 0.96
    with pytest.raises(ValueError, match="strictly"):
        gate.validate_score(score(0.95))
    with pytest.raises(ValueError, match="strictly"):
        gate.validate_score(score(0.94))
    for old in [None, {}, dict(score=1, protocol="navsim-v1-native-math-executed-counterfactual-v1"),
                dict(epdms=99, scope="navtest_average")]:
        with pytest.raises(ValueError):
            gate.validate_score(old)


@pytest.mark.parametrize("bad", [True, "0.99", 99.0, float("nan"), float("inf"), -0.1])
def test_invalid_scale_and_nonfinite_fail(bad):
    item = score()
    item["score"] = bad
    with pytest.raises(ValueError):
        gate.validate_score(item)


@pytest.mark.parametrize("key", list(gate.MULTIPLIERS) + list(gate.WEIGHTS))
def test_missing_component_never_defaults_to_full_score(key):
    item = score()
    del item["components"][key]
    with pytest.raises(ValueError):
        gate.validate_score(item)


def test_component_failure_cannot_be_overridden_by_good_scalar():
    item = score()
    item["components"]["traffic_light_compliance"] = 0.0
    with pytest.raises(ValueError, match="aggregation"):
        gate.validate_score(item)
    item.update(score=0.0, passed=False)
    assert gate.validate_score(item, require_pass=False)["score"] == 0.0


def test_extended_comfort_cannot_be_missing_or_inconsistent():
    item = score()
    item.pop("two_frame_extended_comfort")
    with pytest.raises(ValueError, match="extended comfort"):
        gate.validate_score(item)
    item["two_frame_extended_comfort"] = 0.0
    with pytest.raises(ValueError, match="extended comfort"):
        gate.validate_score(item)
    item["components"]["two_frame_extended_comfort"] = 0.0
    item.update(score=0.875, passed=False)
    assert gate.validate_score(item, require_pass=False)["score"] == 0.875
    with pytest.raises(ValueError, match="strictly"):
        gate.validate_score(item)


def test_same_branch_and_code_revalidated(tmp_path, monkeypatch):
    scene, rollout, context, source = [tmp_path / n for n in ["scene.npz", "teacher.npz", "ctx.pkl", "scorer.py"]]
    for p in [scene, rollout, context, source]:
        p.write_bytes(p.name.encode())
    files = {str(p): digest(p) for p in [scene, rollout]}
    value = score()
    value.update(branch_files=files, source_files={str(source): digest(source)},
                 context=dict(path=str(context), sha256=digest(context)),
                 planning_capture=dict(path=str(context), sha256=digest(context)))
    report = tmp_path / "report.json"
    report.write_text(json.dumps(dict(trajectory_evidence=dict(score=dict(epdms=value)))))
    proof = tmp_path / "native.json"
    proof.write_text(json.dumps(dict(branch=dict(scene=str(scene), rollout=str(rollout), report=str(report)), files=files)))
    sample = dict(kind="chainflow_admitted_synthetic_sample",
                  row=dict(admission=dict(path=str(proof), sha256=digest(proof))))
    monkeypatch.setattr(gate, "scorer_sources", lambda: {str(source): digest(source)})
    assert gate.validate_sample(sample)["score"] == 1.0
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="complete owned"):
        gate.validate_sample(sample)
    source.write_bytes(source.name.encode())
    context.write_bytes(b"another branch context")
    with pytest.raises(ValueError, match="changed"):
        gate.validate_sample(sample)
    context.write_bytes(context.name.encode())
    value["branch_files"] = {str(scene): digest(scene), str(tmp_path / "other.npz"): digest(rollout)}
    report.write_text(json.dumps(dict(trajectory_evidence=dict(score=dict(epdms=value)))))
    with pytest.raises(ValueError, match="different physical branch"):
        gate.validate_sample(sample)


def test_current_campaign_only(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "project_root", lambda: tmp_path)
    root = tmp_path / "dataset"
    root.mkdir()
    campaign = tmp_path / "exp/new"
    (root / "collection.json").write_text(json.dumps(dict(campaign=str(campaign), trajectory_quality=gate.POLICY)))
    assert gate.required_for(campaign / "teacher")
    assert not gate.required_for(tmp_path / "exp/historical")


@pytest.mark.parametrize("mutation", [None, "original", "copy", "swap"])
def test_relocated_branch_requires_identical_originals_and_role_copies(tmp_path, monkeypatch, mutation):
    original, copied = tmp_path / "original", tmp_path / "copied"
    original.mkdir(); copied.mkdir()
    for name in ("scene.npz", "teacher.npz"):
        (original / name).write_bytes(name.encode())
        (copied / name).write_bytes(name.encode())
    context = tmp_path / "context.pkl"
    context.write_bytes(b"context")
    value = score()
    value.update(branch_files={str(p): digest(p) for p in original.iterdir()}, source_files={},
        context=dict(path=str(context), sha256=digest(context)),
        planning_capture=dict(path=str(context), sha256=digest(context)))
    files = {str(p): digest(p) for p in copied.iterdir()}
    if mutation == "original": (original / "scene.npz").write_bytes(b"changed")
    if mutation == "copy": (copied / "scene.npz").write_bytes(b"changed")
    if mutation == "swap":
        value["branch_files"] = {str(original / "scene.npz"): digest(original / "teacher.npz"),
            str(original / "teacher.npz"): digest(original / "scene.npz")}
    report = tmp_path / "report.json"
    report.write_text(json.dumps(dict(trajectory_evidence=dict(score=dict(epdms=value)))))
    proof = tmp_path / "admission.json"
    proof.write_text(json.dumps(dict(branch=dict(scene=str(copied / "scene.npz"),
        rollout=str(copied / "teacher.npz"), report=str(report)), files=files)))
    sample = dict(kind="chainflow_admitted_synthetic_sample",
        row=dict(admission=dict(path=str(proof), sha256=digest(proof))))
    monkeypatch.setattr(gate, "scorer_sources", lambda: {})
    if mutation:
        with pytest.raises(ValueError): gate.validate_sample(sample)
    else:
        assert gate.validate_sample(sample)["score"] == 1.0
