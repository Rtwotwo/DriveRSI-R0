import json

import pytest

from driversi.rsi.decisions.evaluation_aggregate import METRICS, aggregate, from_weakness_review
from driversi.rsi.plugins.mining import _records, prepare_aggregate


def manifest():
    return {
        "schema_version": 1,
        "benchmark": "navtest",
        "checkpoint": "model",
        "evaluator_version": "native-v1",
        "scope": "bounded_probe",
        "expected_tokens": ["private-a", "private-b"],
        "allowed_logs": ["test-log"],
    }


def rows():
    values = {k: 1.0 for k in ("score", *METRICS)}
    return [
        {
            "token": token,
            "source_log": "test-log",
            "status": "scored",
            "metrics": {**values, "ego_progress": 0.5 if token == "private-a" else 1},
            "conditions": ["nearby_static"],
        }
        for token in manifest()["expected_tokens"]
    ]


def test_counts_transfer_without_test_identity():
    report = aggregate(manifest(), rows())
    text = json.dumps(report)
    assert all(s not in text for s in ("private-a", "private-b", "test-log"))
    assert report["coverage"]["all_scored"]
    assert next(
        s for s in report["capability_feedback"]["slices"] if s["capability"] == "route_progress"
    ) == {"capability": "route_progress", "count": 2, "failures": 1}
    assert report["evaluation_role"] == "adaptive_evaluation"


def test_runtime_failures_are_not_driving_failures():
    value = rows()
    value[1].update(status="failed", metrics=None)
    result = aggregate(manifest(), value)
    assert result["coverage"]["complete"] and not result["coverage"]["all_scored"]
    assert result["coverage"]["runtime_failed_or_unknown"] == 1
    assert all(s["count"] == 1 for s in result["capability_feedback"]["slices"])


def test_missing_duplicate_and_foreign_results():
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(manifest(), rows()[:1])
    assert aggregate(manifest(), rows()[:1], allow_partial=True)["coverage"]["missing"] == 1
    for value in (rows() + rows()[:1], [{**rows()[0], "source_log": "foreign"}, rows()[1]]):
        with pytest.raises(ValueError):
            aggregate(manifest(), value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 1.1, True])
def test_invalid_native_metrics(value):
    data = rows()
    data[0]["metrics"]["ego_progress"] = value
    with pytest.raises(ValueError):
        aggregate(manifest(), data)


def test_arbitrary_tags_cannot_export_test_content():
    data = rows()
    data[0]["conditions"] = ["test-token-or-coordinate"]
    with pytest.raises(ValueError, match="condition"):
        aggregate(manifest(), data)


def test_alias_mapping_and_subresolution_changes():
    weakness = {
        "checkpoint": "model",
        "support": {"paired_samples": 4},
        "signals": [
            {
                "capability_hypothesis": "collision_avoidance",
                "candidate_compliance_shortfall": True,
                "deficient_sample_ids": ["x", "y"],
            },
            {
                "capability_hypothesis": "anticipatory_interaction",
                "actionable_from_this_change": True,
                "regressed_sample_ids": ["z"],
            },
            {
                "capability_hypothesis": "route_progress",
                "actionable_from_this_change": False,
                "regressed_sample_ids": ["noise"],
            },
        ],
    }
    result = from_weakness_review(weakness, evaluator_version="native")
    assert result["slices"] == [
        {"capability": "collision", "count": 4, "failures": 2},
        {"capability": "interaction", "count": 4, "failures": 1},
    ]


def test_formal_coverage_must_match_full_filter(tmp_path):
    m = manifest()
    m["scope"] = "formal_benchmark"
    for name, data in (
        ("manifest.json", m),
        ("results.json", {"rows": rows()}),
        (
            "filter.json",
            {"tokens": ["private-a", "private-b", "missing"], "log_names": ["test-log"]},
        ),
    ):
        (tmp_path / name).write_text(json.dumps(data))
    config = {
        "manifest": "manifest.json",
        "results": "results.json",
        "official_filter": "filter.json",
        "descriptors": None,
        "allow_partial": False,
    }
    with pytest.raises(ValueError, match="canonical official"):
        prepare_aggregate(config, tmp_path, tmp_path / "registry")


def test_native_csv_ignores_average_and_keeps_invalid(tmp_path):
    import csv

    fields = ["token", "valid", "score", *METRICS]
    path = tmp_path / "scores.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"token": "private-a", "valid": True, **rows()[0]["metrics"]})
        w.writerow({"token": "private-b", "valid": False})
        w.writerow({"token": "average", "valid": False, "score": 999})
    desc = tmp_path / "desc.json"
    desc.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "token": r["token"],
                        "source_log": r["source_log"],
                        "conditions": r["conditions"],
                    }
                    for r in rows()
                ]
            }
        )
    )
    imported = _records({"results": str(path), "descriptors": str(desc)})
    assert len(imported) == 2 and imported[1]["metrics"] is None
    result = aggregate(manifest(), imported)
    assert result["coverage"]["runtime_failed_or_unknown"] == 1


def test_confirmation_requires_exact_bound_cohort(tmp_path):
    from driversi.rsi.plugins.mining import run_decision

    retrieve = {
        "capability_feedback": {
            "schema_version": 1,
            "benchmark": "development",
            "checkpoint": "current",
            "evaluator_version": "native",
            "slices": [{"capability": "route_progress", "count": 2, "failures": 1}],
        },
        "candidates": [{"token": "train", "source_log": "train-log"}],
        "feedback_scope": "bounded_probe",
    }
    observed = {
        "checkpoint": "current",
        "checkpoint_unchanged": True,
        "rows": [
            {
                "token": "train",
                "source_log": "train-log",
                "status": "scored",
                "metrics": {k: 1.0 for k in ("score", *METRICS)},
            }
        ],
    }
    rp, cp = tmp_path / "retrieval.json", tmp_path / "confirmation.json"
    rp.write_text(json.dumps(retrieve))
    inputs = {
        "retrieval": {
            "artifacts": {"report": str(rp)},
            "summary": {"artifact_reference": {"identity": "bound"}},
        },
        "confirmation": {
            "artifacts": {"report": str(cp)},
            "summary": {"input_references": {"retrieval": "bound"}},
        },
    }
    cp.write_text(json.dumps(observed))
    result = run_decision({}, inputs, tmp_path / "ok", tmp_path / "registry")
    assert result["summary"]["decisions"][0]["action"] == "retain_or_expand_search"
    variants = [
        {**observed, "checkpoint": "other"},
        {**observed, "rows": []},
        {**observed, "rows": observed["rows"] * 2},
        {**observed, "rows": [{**observed["rows"][0], "source_log": "other"}]},
    ]
    for i, value in enumerate(variants):
        cp.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="linkage differs"):
            run_decision({}, inputs, tmp_path / str(i), tmp_path / "registry")
