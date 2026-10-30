"""Publish frozen-cohort search experience and qualified supervision separately.

This ingests completed producer records; it neither runs teachers nor computes PDMS.
Failed attempts and untouched cases stay visible. Costs are producer-reported and
checked against frozen caps, not reconstructed hardware utilization.
"""

import json
import math

from driversi.data.datasets.teacher_candidates import (
    _artifact,
    _publish_new,
    _teacher_overlay,
    validate_admission_policy,
    validate_comparison_candidate,
    validate_trajectory_candidate,
)
from driversi.rsi.contracts import BUDGET_FIELDS, _identity, _keys
from driversi.training.sampling.environment import source_log_group


def _resources(value):
    _keys(value, BUDGET_FIELDS, "search resources")
    for key, count in value.items():
        if type(count) not in (int, float) or not math.isfinite(count) or count < 0:
            raise ValueError("Search resources must be finite and nonnegative")
        if key not in ("cpu_seconds", "gpu_seconds") and type(count) is not int:
            raise ValueError("Discrete search resources require integer counts")
    return dict(value)


def build_search_dataset(spec, attempts, policy, navtrain_membership, protected_groups, output):
    """Create an immutable dataset manifest from all attempts in a frozen case cohort."""
    validate_admission_policy(policy)
    _keys(spec, ("schema_version", "id", "versions", "cases", "budget_per_case"), "search cohort")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("Unknown search cohort schema")
    _identity(spec["id"], "dataset version")
    versions = spec["versions"]
    _keys(versions, ("student", "teacher", "improver", "source_dataset", "evaluator"), "versions")
    for key, value in versions.items():
        _identity(value, key)
    if versions["evaluator"] != policy["scorer_version"]:
        raise ValueError("Frozen evaluator differs from admission policy")
    caps = _resources(spec["budget_per_case"])
    if caps["trials"] < 1:
        raise ValueError("Search cohort requires a positive attempt budget")
    if not isinstance(spec["cases"], list) or not spec["cases"]:
        raise ValueError("A nonempty cohort is required before reading search outcomes")
    protected = {source_log_group(x) for x in protected_groups}
    cases, keys = {}, set()
    for case in spec["cases"]:
        _keys(case, ("id", "source_token", "source_log", "observation_state_id", "branch_id",
                     "episode_id", "observation_tick", "state_origin", "verification", "baseline_score"), "search case")
        for key in ("id", "source_token", "observation_state_id", "branch_id", "episode_id"):
            _identity(case[key], key)
        if (type(case["observation_tick"]) is not int or case["observation_tick"] < 0
                or case["state_origin"] not in ("logged", "counterfactual")):
            raise ValueError("Invalid case observation")
        baseline = case["baseline_score"]
        if type(baseline) not in (float, int) or not math.isfinite(baseline) or not 0 <= baseline <= 1:
            raise ValueError("Frozen baseline score must use finite [0,1]")
        verification = case["verification"]
        _keys(verification, ("role", "mode", "world_context", "reference_trajectory", "baseline_trajectory"),
              "frozen case verification")
        if (verification["role"] != "final_verifier"
                or verification["mode"] != "independent_fixed_reference_pair"):
            raise ValueError("Frozen cases require independent fixed-reference verification")
        for name in ("world_context", "reference_trajectory", "baseline_trajectory"):
            entry = verification[name]
            _keys(entry, ("path", "sha256"), name)
            _artifact(entry["path"], entry["sha256"])
        context = json.loads(_artifact(verification["world_context"]["path"],
                                       verification["world_context"]["sha256"]).read_text())
        if any(context.get(k) != case[k] for k in ("source_token", "episode_id", "branch_id",
               "observation_state_id", "observation_tick", "state_origin")):
            raise ValueError("Frozen case context differs from declared observation")
        token, group = case["source_token"], source_log_group(case["source_log"])
        if (token not in navtrain_membership or source_log_group(navtrain_membership[token]) != group
                or group in protected):
            raise ValueError("Search case is outside unprotected navtrain membership")
        identity = (token, case["observation_state_id"], case["branch_id"])
        if case["id"] in cases or identity in keys:
            raise ValueError("Duplicate case identity would inflate the solve-rate denominator")
        keys.add(identity)
        cases[case["id"]] = case
    if not isinstance(attempts, list):
        raise ValueError("Search attempts must be a complete list")
    by_case = {cid: [] for cid in cases}
    seen, candidate_ids = set(), set()
    candidate_case, candidate_blockers = {}, {}
    for attempt in attempts:
        _keys(attempt, ("id", "case_id", "status", "cost", "candidates", "failure_reason"), "attempt")
        _identity(attempt["id"], "attempt id")
        if attempt["id"] in seen or attempt["case_id"] not in cases:
            raise ValueError("Duplicate attempt or unknown case")
        seen.add(attempt["id"])
        if attempt["status"] not in ("completed", "no_solution", "error", "timeout", "interrupted"):
            raise ValueError("Only terminal search attempts may be published")
        _resources(attempt["cost"])
        if attempt["cost"]["trials"] != 1:
            raise ValueError("Every attempt, including failures, consumes exactly one trial")
        if not isinstance(attempt["candidates"], list):
            raise ValueError("Candidates must be a list")
        if attempt["status"] != "completed" and not attempt["failure_reason"]:
            raise ValueError("Unsuccessful attempts require a reason")
        if attempt["failure_reason"] is not None and not isinstance(attempt["failure_reason"], str):
            raise ValueError("Failure reason must be text or null")
        if attempt["status"] == "completed" and not attempt["candidates"]:
            raise ValueError("Completed searches need candidates; use no_solution otherwise")
        case = cases[attempt["case_id"]]
        for candidate in attempt["candidates"]:
            cid = candidate.get("id")
            _identity(cid, "candidate id")
            if cid in candidate_ids:
                raise ValueError("Candidate identities must be unique across attempts")
            candidate_ids.add(cid)
            if any(candidate.get(k) != case[k] for k in case if k not in ("id", "source_log", "verification", "baseline_score")):
                raise ValueError("Candidate does not belong to its declared search case")
            if source_log_group(candidate.get("source_log")) != source_log_group(case["source_log"]):
                raise ValueError("Candidate source log differs from case")
            if candidate.get("teacher", {}).get("version") != versions["teacher"]:
                raise ValueError("Candidate teacher differs from frozen producer version")
            candidate_case[cid] = attempt["case_id"]
            if candidate.get("schema_version") != 2:
                candidate_blockers[cid] = "legacy_or_unknown_schema_diagnostic_only"
            elif candidate.get("score", {}).get("verification") != case["verification"]:
                candidate_blockers[cid] = "candidate_changed_frozen_scoring_context"
            elif candidate["score"].get("reference_value") != case["baseline_score"]:
                candidate_blockers[cid] = "candidate_changed_frozen_baseline_score"
        by_case[attempt["case_id"]].append(attempt)
    eligible, case_rows, trajectory_solved = [], [], set()
    total_cost = dict.fromkeys(BUDGET_FIELDS, 0)
    for cid, case in cases.items():
        rows = by_case[cid]
        cost = {k: sum(row["cost"][k] for row in rows) for k in BUDGET_FIELDS}
        for key in cost:
            total_cost[key] += cost[key]
        exceeded = [k for k in cost if cost[k] > caps[k]]
        reasons = []
        for row in rows:
            for candidate in row["candidates"]:
                blocker = candidate_blockers.get(candidate["id"])
                if exceeded or row["status"] != "completed" or blocker:
                    reasons.append({"candidate_id": candidate["id"], "reason":
                                    "case_budget_exceeded" if exceeded else
                                    "attempt_not_completed" if row["status"] != "completed" else blocker})
                    continue
                try:
                    validate_trajectory_candidate(candidate, policy, navtrain_membership, protected)
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    reasons.append({"candidate_id": candidate["id"], "reason": str(exc)})
                else:
                    trajectory_solved.add(cid)
                eligible.append(candidate)
        case_rows.append({"case": case, "attempt_count": len(rows), "cost": cost,
                          "over_budget": exceeded, "trajectory_qualified": cid in trajectory_solved,
                          "supervision_qualified": False,
                          "status": "over_budget" if exceeded else "attempted" if rows else "not_attempted",
                          "trajectory_rejections": reasons})
    overlay = _teacher_overlay(eligible, policy, navtrain_membership, protected)
    supervised = {candidate_case[row["candidate"]["id"]] for row in overlay["admitted"]}
    winners = {candidate_case[row["candidate"]["id"]]: row["candidate"] for row in overlay["admitted"]}
    pairs = []
    for candidate in eligible:
        chosen = winners.get(candidate_case[candidate["id"]])
        if chosen is None or chosen["id"] == candidate["id"]:
            continue
        try:
            validate_comparison_candidate(candidate, policy, navtrain_membership, protected)
        except (ValueError, TypeError, KeyError, OSError):
            continue
        gap = chosen["score"]["value"] - candidate["score"]["value"]
        if gap > 0 and gap >= policy["min_gain"]:
            pairs.append({"case_id": candidate_case[candidate["id"]], "chosen_id": chosen["id"],
                          "rejected_id": candidate["id"], "score_gap": gap,
                          "metric": policy["metric"], "negative_is_imitation_target": False})
    for row in case_rows:
        row["supervision_qualified"] = row["case"]["id"] in supervised
    count = len(cases)
    manifest = {
        "schema_version": 1, "kind": "navtrain_search_dataset", "search_spec": spec,
        "admission_policy": policy, "source_data_modified": False,
        "qualified_supervision": overlay,
        "experience_pool": {"imitation_eligible": False, "attempts": attempts},
        "cases": case_rows, "preference_pairs": pairs,
        "summary": {"registered_cases": count, "attempted_cases": sum(bool(v) for v in by_case.values()),
                    "attempts": len(attempts), "preference_pairs": len(pairs), "trajectory_solved": len(trajectory_solved),
                    "supervision_solved": len(supervised),
                    "trajectory_solve_rate": len(trajectory_solved)/count,
                    "supervision_solve_rate": len(supervised)/count, "total_cost": total_cost,
                    "denominator": "all_declared_frozen_cases_including_unattempted",
                    "cost_source": "producer_reported_all_terminal_attempts"},
        "qualification": {"official_scores_computed_here": False, "approved_teacher": False,
                          "recursive_improver_learning_proven": False},
    }
    _publish_new(output, manifest)
    return manifest
