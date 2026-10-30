# ruff: noqa: E402
"""Synthetic scorer artifacts test admission mechanics, never official scores."""

import copy
import hashlib
import json

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")

from driversi.data.datasets.teacher_candidates import build_teacher_overlay, validate_candidate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def candidate_case(tmp_path):
    trajectory = tmp_path / "trajectory.npz"
    times = np.arange(1, 9) * 0.5
    np.savez(trajectory, times=times, poses=np.column_stack((times * 4, times * 0, times * 0)))
    photo = tmp_path / "front.png"
    Image.new("RGB", (32, 16), "grey").save(photo)
    policy = {
        "id": "TEST_ONLY",
        "metric": "pdms",
        "protocol_version": "fixture-v1",
        "scorer_version": "fixture-v1",
        "min_score": 0.95,
        "min_gain": 0.01,
        "min_horizon_seconds": 4,
        "required_views": ["front"],
    }
    record = {
        "schema_version": 2,
        "id": "candidate",
        "source_token": "token",
        "source_log": "training-log",
        "source_split": "navtrain",
        "episode_id": "episode",
        "branch_id": "branch",
        "observation_state_id": "state",
        "observation_tick": 10,
        "state_origin": "logged",
        "teacher": {
            "backend": "drivetto",
            "version": "fixture",
            "training_groups": ["training-log"],
            "information_set": "current_state_history",
        },
        "trajectory": {
            "path": str(trajectory),
            "sha256": digest(trajectory),
            "coordinate_frame": "ego_at_observation",
            "kind": "executed",
        },
        "execution": dict.fromkeys(
            ("collision_free", "road_supported", "finite_dynamics", "trajectory_executed"), True
        ),
        "images": [
            {
                "path": str(photo),
                "sha256": digest(photo),
                "view": "front",
                "observation_state_id": "state",
                "episode_id": "episode",
                "branch_id": "branch",
                "observation_tick": 10,
                "source": "original_log",
                "asset_version": None,
                "alignment_valid": True,
                "support_valid": True,
            }
        ],
    }
    score = {
        "metric": "pdms",
        "value": 0.99,
        "reference_value": 0.80,
        "scale": "unit_interval",
        "protocol_version": "fixture-v1",
        "scorer_version": "fixture-v1",
        "context_state_id": "state",
        "trajectory_sha256": digest(trajectory),
    }
    context = tmp_path / "context.json"
    context.write_text(json.dumps({k: record[k] for k in
        ("source_token", "episode_id", "branch_id", "observation_state_id", "observation_tick", "state_origin")}))
    score["verification"] = {
        "role": "final_verifier", "mode": "independent_fixed_reference_pair",
        "world_context": {"path": str(context), "sha256": digest(context)},
        "reference_trajectory": {"path": str(trajectory), "sha256": digest(trajectory)},
        "baseline_trajectory": {"path": str(trajectory), "sha256": digest(trajectory)},
    }
    evidence = tmp_path / "score.json"
    evidence.write_text(
        json.dumps(
            {
                **score,
                "valid": True,
                "candidate_id": "candidate",
                "source_token": "token",
                "branch_id": "branch",
            }
        )
    )
    record["score"] = {**score, "evidence_path": str(evidence), "evidence_sha256": digest(evidence)}
    return record, policy, {"token": "training-log"}, ["eval-log"]


@pytest.mark.parametrize("backend", ["drivetto", "privileged_teacher"])
def test_both_teacher_backends_use_same_admission(candidate_case, backend):
    record, policy, membership, protected = candidate_case
    record["teacher"]["backend"] = backend
    before = copy.deepcopy(record)
    assert validate_candidate(record, policy, membership, protected) == before
    assert record == before


@pytest.mark.parametrize(
    "defect",
    [
        "navtest",
        "navhard",
        "not_member",
        "lineage",
        "counterfactual_original",
        "future_image",
        "branch_image",
        "image_support",
        "score_units",
        "score_protocol",
        "score_identity",
        "unexecuted",
        "collision",
        "too_short",
        "image_contents",
    ],
)
def test_bad_supervision_is_not_admitted(candidate_case, defect):
    record, policy, membership, protected = candidate_case
    if defect in ("navtest", "navhard"):
        record["source_split"] = defect
    elif defect == "not_member":
        membership.clear()
    elif defect == "lineage":
        record["teacher"]["training_groups"] = ["eval-log"]
    elif defect == "counterfactual_original":
        record["state_origin"] = "counterfactual"
    elif defect == "future_image":
        record["images"][0]["observation_tick"] += 1
    elif defect == "branch_image":
        record["images"][0]["branch_id"] = "other"
    elif defect == "image_support":
        record["images"][0]["support_valid"] = False
    elif defect == "score_units":
        record["score"]["value"] = 99
    elif defect == "score_protocol":
        record["score"]["metric"] = "epdms"
    elif defect == "score_identity":
        record["score"]["context_state_id"] = "other"
    elif defect == "unexecuted":
        record["trajectory"]["kind"] = "proposal"
    elif defect == "collision":
        record["execution"]["collision_free"] = False
    elif defect == "too_short":
        policy["min_horizon_seconds"] = 6
    else:
        from pathlib import Path

        path = Path(record["images"][0]["path"])
        path.write_bytes(b"not an image")
        record["images"][0]["sha256"] = digest(path)
    with pytest.raises((ValueError, OSError)):
        validate_candidate(record, policy, membership, protected)


def test_threshold_is_strict_and_same_metric_gain_required(candidate_case):
    from pathlib import Path

    record, policy, membership, protected = candidate_case
    score = record["score"]
    path = Path(score["evidence_path"])
    evidence = json.loads(path.read_text())
    score["value"] = evidence["value"] = 0.95
    path.write_text(json.dumps(evidence))
    score["evidence_sha256"] = digest(path)
    with pytest.raises(ValueError, match="threshold"):
        validate_candidate(record, policy, membership, protected)


def test_counterfactual_requires_rendered_aligned_assets(candidate_case):
    record, policy, membership, protected = candidate_case
    record["state_origin"] = "counterfactual"
    from pathlib import Path

    context = Path(record["score"]["verification"]["world_context"]["path"])
    value = json.loads(context.read_text())
    value["state_origin"] = "counterfactual"
    context.write_text(json.dumps(value))
    record["score"]["verification"]["world_context"]["sha256"] = digest(context)
    evidence = Path(record["score"]["evidence_path"])
    value = json.loads(evidence.read_text())
    value["verification"] = record["score"]["verification"]
    evidence.write_text(json.dumps(value))
    record["score"]["evidence_sha256"] = digest(evidence)
    record["images"][0].update(source="rendered", asset_version="fixture-render-v1")
    assert validate_candidate(record, policy, membership, protected)


def test_overlay_retains_source_and_rejection_without_overwrite(candidate_case, tmp_path):
    record, policy, membership, protected = candidate_case
    bad = copy.deepcopy(record)
    bad["id"] = "bad"
    bad["source_split"] = "navtest"
    before = copy.deepcopy(record)
    target = tmp_path / "dataset-v1.json"
    result = build_teacher_overlay([record, bad], policy, membership, protected, target)
    assert result["attempted"] == 2 and result["accepted"] == 1
    assert result["source_data_modified"] is False
    assert result["admitted"][0]["replacement_key"]["source_token"] == "token"
    assert result["rejected"][0]["id"] == "bad"
    assert record == before and json.loads(target.read_text()) == result
    with pytest.raises(ValueError, match="exists"):
        build_teacher_overlay([record], policy, membership, protected, target)
    with pytest.raises(ValueError, match="unique"):
        build_teacher_overlay(
            [record, record], policy, membership, protected, tmp_path / "duplicate.json"
        )


@pytest.mark.parametrize("defect", ["legacy", "oracle", "pool", "wrong_context", "changed_reference"])
def test_search_evidence_is_not_final_verification(candidate_case, defect):
    from pathlib import Path

    record, policy, membership, protected = candidate_case
    if defect == "legacy":
        record["schema_version"] = 1
    elif defect == "oracle":
        record["teacher"]["information_set"] = "logged_future"
    elif defect == "pool":
        record["score"]["verification"]["mode"] = "search_pool"
    else:
        v = record["score"]["verification"]
        if defect == "wrong_context":
            p = Path(v["world_context"]["path"])
            data = json.loads(p.read_text())
            data["branch_id"] = "another-branch"
            p.write_text(json.dumps(data))
            v["world_context"]["sha256"] = digest(p)
        else:
            Path(v["reference_trajectory"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_candidate(record, policy, membership, protected)


def search_request(candidate_case):
    record, policy, membership, protected = candidate_case
    keys = ("source_token", "source_log", "observation_state_id", "branch_id", "episode_id",
            "observation_tick", "state_origin")
    cases = [{"id": "case-0", **{k: record[k] for k in keys},
              "verification": copy.deepcopy(record["score"]["verification"]),
              "baseline_score": record["score"]["reference_value"]}]
    for i in (1, 2):
        from pathlib import Path

        case = copy.deepcopy(cases[0])
        case.update(id=f"case-{i}", observation_state_id=f"state-{i}")
        context = Path(case["verification"]["world_context"]["path"])
        value = json.loads(context.read_text())
        value["observation_state_id"] = case["observation_state_id"]
        context = context.with_name(f"case-{i}-context.json")
        context.write_text(json.dumps(value))
        case["verification"]["world_context"] = {"path": str(context), "sha256": digest(context)}
        cases.append(case)
    cost = dict(gpu_seconds=1.0, cpu_seconds=2.0, llm_tokens=0, sim_agent_steps=10,
                rendered_frames=0, trials=1)
    spec = {"schema_version": 1, "id": "TEST_ONLY_SEARCH",
            "versions": dict(student="student-v1", teacher="fixture", improver="fixed-search-v1",
                             source_dataset="fixture-data", evaluator="fixture-v1"),
            "cases": cases, "budget_per_case": {**cost, "trials": 2, "gpu_seconds": 10., "cpu_seconds": 10.}}
    attempts = [{"id": "a0", "case_id": "case-0", "status": "completed", "cost": cost,
                 "candidates": [record], "failure_reason": None},
                {"id": "a1", "case_id": "case-1", "status": "timeout", "cost": cost,
                 "candidates": [], "failure_reason": "declared timeout"}]
    return dict(search_spec=spec, attempts=attempts, policy=policy,
                navtrain_membership=membership, protected_groups=protected)


def publish_request(request, target):
    from driversi.rsi.correction.dataset import build_search_dataset

    return build_search_dataset(request["search_spec"], request["attempts"], request["policy"],
                                request["navtrain_membership"], request["protected_groups"], target)


def test_failed_and_unattempted_cases_keep_denominator(candidate_case, tmp_path):
    request = search_request(candidate_case)
    value = publish_request(request, tmp_path / "search.json")
    stats = value["summary"]
    assert stats["registered_cases"] == 3 and stats["attempted_cases"] == 2
    assert stats["trajectory_solve_rate"] == stats["supervision_solve_rate"] == 1/3
    assert stats["total_cost"]["trials"] == 2 and stats["total_cost"]["cpu_seconds"] == 4
    assert value["cases"][2]["status"] == "not_attempted"
    assert value["experience_pool"]["imitation_eligible"] is False
    assert value["experience_pool"]["attempts"] == request["attempts"]
    assert len(value["qualified_supervision"]["admitted"]) == 1


def test_render_failure_does_not_become_trajectory_failure(candidate_case, tmp_path):
    request = search_request(candidate_case)
    request["attempts"][0]["candidates"][0]["images"][0]["support_valid"] = False
    value = publish_request(request, tmp_path / "search.json")
    assert value["summary"]["trajectory_solved"] == 1
    assert value["summary"]["supervision_solved"] == 0
    assert value["qualified_supervision"]["rejected"][0]["reason"] == "Image alignment/support not qualified"


def test_over_budget_case_retains_cost_but_no_labels(candidate_case, tmp_path):
    request = search_request(candidate_case)
    request["attempts"][0]["cost"]["cpu_seconds"] = 11
    value = publish_request(request, tmp_path / "search.json")
    assert value["summary"]["trajectory_solved"] == value["summary"]["supervision_solved"] == 0
    assert value["summary"]["total_cost"]["cpu_seconds"] == 22  # both fixture attempts share the cost
    assert value["cases"][0]["over_budget"] == ["cpu_seconds"]
    assert value["experience_pool"]["attempts"][0]["candidates"]


@pytest.mark.parametrize("defect", ["duplicate_case", "protected", "wrong_version", "wrong_case", "duplicate_attempt", "free_failure", "unknown_cost", "pending"])
def test_search_publication_rejects_invalid_cohort(candidate_case, tmp_path, defect):
    request = search_request(candidate_case)
    if defect == "duplicate_case":
        request["search_spec"]["cases"].append(request["search_spec"]["cases"][0])
    elif defect == "protected":
        request["protected_groups"].append("training-log")
    elif defect == "wrong_version":
        request["search_spec"]["versions"]["teacher"] = "another-teacher"
    elif defect == "wrong_case":
        request["attempts"][0]["case_id"] = "case-1"
    elif defect == "duplicate_attempt":
        request["attempts"].append(request["attempts"][0])
    elif defect == "free_failure":
        request["attempts"][1]["cost"]["trials"] = 0
    elif defect == "unknown_cost":
        request["attempts"][1]["cost"]["gpu_seconds"] = float("nan")
    else:
        request["attempts"][1]["status"] = "running"
    target = tmp_path / "no-publication.json"
    with pytest.raises(ValueError):
        publish_request(request, target)
    assert not target.exists()


def test_teacher_search_cli_uses_existing_protected_ledger(candidate_case, tmp_path, capsys):
    from driversi.rsi.__main__ import main
    from driversi.rsi.execution.registry import ExperimentRegistry

    request = search_request(candidate_case)
    file = tmp_path / "request.json"
    file.write_text(json.dumps(request))
    ledger = tmp_path / "ledger.sqlite"
    ExperimentRegistry(ledger)
    output = tmp_path / "dataset.json"
    assert main(["--registry", str(ledger), "admit-teacher-search", str(file), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["registered_cases"] == 3
    assert json.loads(capsys.readouterr().out)["supervision_solved"] == 1


def test_overlay_publication_failure_leaves_no_partial_file(candidate_case, tmp_path, monkeypatch):
    import os

    record, policy, membership, protected = candidate_case
    target = tmp_path / "unpublished.json"
    def fail_link(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="publication"):
        build_teacher_overlay([record], policy, membership, protected, target)
    assert not target.exists() and not list(tmp_path.glob(".teacher-*.tmp"))


def test_search_case_freezes_independent_reference(candidate_case, tmp_path):
    request = search_request(candidate_case)
    request["attempts"][0]["candidates"][0]["score"]["verification"]["role"] = "different-verifier"
    result = publish_request(request, tmp_path / "changed-case.json")
    assert result["summary"]["supervision_solved"] == 0
    assert result["cases"][0]["trajectory_rejections"][0]["reason"] == "candidate_changed_frozen_scoring_context"


def test_context_tick_must_match_observation(candidate_case):
    record, policy, membership, protected = candidate_case
    record["observation_tick"] += 1
    with pytest.raises(ValueError, match="context differs"):
        validate_candidate(record, policy, membership, protected)


def add_negative(request, tmp_path, *, supported=True, same_score=False):
    from pathlib import Path

    original = request["attempts"][0]["candidates"][0]
    negative = copy.deepcopy(original)
    negative["id"] = "bad-candidate"
    negative["score"]["value"] = original["score"]["value"] if same_score else .30
    negative["execution"]["collision_free"] = False
    negative["images"][0]["support_valid"] = supported
    evidence = json.loads(Path(original["score"]["evidence_path"]).read_text())
    evidence.update(candidate_id=negative["id"], value=negative["score"]["value"])
    path = tmp_path / "bad-score.json"
    path.write_text(json.dumps(evidence))
    negative["score"].update(evidence_path=str(path), evidence_sha256=digest(path))
    request["attempts"][0]["candidates"].append(negative)


def test_bad_trajectory_retained_as_preference_not_imitation(candidate_case, tmp_path):
    request = search_request(candidate_case)
    add_negative(request, tmp_path)
    result = publish_request(request, tmp_path / "with-negatives.json")
    assert len(result["qualified_supervision"]["admitted"]) == 1
    pair = result["preference_pairs"][0]
    assert pair["chosen_id"] == "candidate" and pair["rejected_id"] == "bad-candidate"
    assert pair["negative_is_imitation_target"] is False
    assert pair["score_gap"] == pytest.approx(.69)
    assert len(result["experience_pool"]["attempts"][0]["candidates"]) == 2


@pytest.mark.parametrize("bad_pixels,tie", [(True, False), (False, True)])
def test_invalid_images_or_tied_scores_do_not_create_preferences(candidate_case, tmp_path, bad_pixels, tie):
    request = search_request(candidate_case)
    add_negative(request, tmp_path, supported=not bad_pixels, same_score=tie)
    result = publish_request(request, tmp_path / "no-pair.json")
    assert result["preference_pairs"] == []


def test_search_case_freezes_baseline_score(candidate_case, tmp_path):
    request = search_request(candidate_case)
    request["attempts"][0]["candidates"][0]["score"]["reference_value"] = .5
    result = publish_request(request, tmp_path / "changed-baseline.json")
    assert result["summary"]["supervision_solved"] == 0
    assert result["cases"][0]["trajectory_rejections"][0]["reason"] == "candidate_changed_frozen_baseline_score"


def test_legacy_search_candidate_keeps_attempt_cost_and_failed_denominator(candidate_case, tmp_path):
    request = search_request(candidate_case)
    record = request["attempts"][0]["candidates"][0]
    record["schema_version"] = 1
    record["score"].pop("verification")
    record["teacher"].pop("information_set")
    result = publish_request(request, tmp_path / "legacy-diagnostic.json")
    assert result["summary"]["registered_cases"] == 3
    assert result["summary"]["total_cost"]["trials"] == 2
    assert result["summary"]["supervision_solved"] == 0
    assert result["experience_pool"]["attempts"][0]["candidates"][0] == record
    assert result["cases"][0]["trajectory_rejections"][0]["reason"] == "legacy_or_unknown_schema_diagnostic_only"
