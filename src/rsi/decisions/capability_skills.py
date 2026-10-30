"""Model-service skills: bounded interventions and scoped experimental memory.

A skill is an experimental procedure, NOT a driving policy or a claimed cause.
Only linked paired training can earn effectiveness credit. Search/admission
success, unlinked historical trials and incomplete cost evidence cannot.
"""

import json
import math
import sqlite3
from pathlib import Path

from driversi.data.catalog.store import identity
from driversi.rsi.decisions.capability_feedback import validate_feedback

CONDITIONS = {
    "collision": (
        ["nearby_vehicle", "nearby_static", "traffic_dense"],
        "vehicle barrier pedestrian obstacle",
    ),
    "road_compliance": (
        ["history_turning", "nearby_static"],
        "curve road edge barrier intersection",
    ),
    "direction_compliance": (
        ["history_turning", "nav_command_0", "nav_command_2"],
        "curve junction lane markings turn",
    ),
    "interaction": (
        ["nearby_vehicle", "traffic_dense", "history_speed_change"],
        "vehicle merging junction queue pedestrian",
    ),
    "route_progress": (
        ["ego_stopped", "ego_low_speed", "nearby_static"],
        "stopped vehicle obstacle narrow road queue",
    ),
    "comfort": (["history_speed_change", "history_turning"], "curve junction braking traffic"),
    "occlusion": (
        ["nearby_vehicle", "nearby_static"],
        "occluded parked vehicle pedestrian barrier",
    ),
    "recovery": (["history_turning", "nearby_static"], "road edge curve barrier"),
    "distribution_shift": (["history_turning", "traffic_dense"], "junction dense traffic curve"),
}

SKILLS = {
    "same_anchor_relabel": {
        "version": 1,
        "mode": "relabel",
        "provider": "drivetto_search_skill",
        "preconditions": ["same_anchor", "valid_map_and_world", "complete_history"],
        "postconditions": ["counterfactual_pdms_ge_95", "physical_integrity", "label_alignment"],
        "cost_class": "trajectory_search",
        "changes_history": False,
    },
    "reactive_new_history": {
        "version": 1,
        "mode": "new_state",
        "provider": "navsim_native_teacher_probe",
        "preconditions": ["valid_map_and_world", "complete_history", "matching_render_assets"],
        "postconditions": [
            "continuous_new_history",
            "counterfactual_pdms_ge_95",
            "physical_integrity",
            "visual_alignment",
            "future_label_alignment",
        ],
        "cost_class": "rollout_search_render",
        "changes_history": True,
        "arrival_backend": "driverl_native",
        "arrival_action": [0.0, 0.0],
    },
    "bounded_perturbation_history": {
        "version": 1,
        "mode": "new_state",
        "provider": "navsim_native_teacher_probe",
        "preconditions": ["valid_map_and_world", "complete_history", "matching_render_assets"],
        "postconditions": [
            "continuous_new_history",
            "counterfactual_pdms_ge_95",
            "physical_integrity",
            "visual_alignment",
            "future_label_alignment",
        ],
        "cost_class": "rollout_search_render",
        "changes_history": True,
        "arrival_backend": "bounded_control",
        "arrival_action": [-0.1, 0.0],
    },
}


# Versioned executable variants preserve native teacher/scorer implementations.
# The arrival still comes from DriveRL; only the same-anchor future procedure differs.
for _name, _backend, _search in (("reactive_new_history_tts", "driverl_native", 8),):
    SKILLS[_name] = {
        **SKILLS["reactive_new_history"],
        "teacher_backend": _backend,
        "search_candidates": _search,
    }
SKILLS["driverl_full_window"] = {
    **SKILLS["reactive_new_history"],
    "arrival_backend": "driverl_native_causal_history",
    "teacher_backend": "driverl_native",
    "search_candidates": 8,
    "history_bootstrap": "causal_resampled_10hz_v1",
    "control_scope": "native_DriveRL_all_70_actions",
}
SKILLS["driverl_sample_full_window"] = {
    **SKILLS["driverl_full_window"],
    "search_candidates": 0,
    "sampling_method": "sample",
}
SKILLS["perturbed_history_tts"] = {
    **SKILLS["driverl_full_window"],
    "arrival_backend": "source_history_perturbation_v1",
    "control_scope": "C2_reference_native_LQR_ticks_0_20_then_native_TTS_ticks_20_70",
}
NEW_HISTORY_SKILLS = (
    "perturbed_history_tts",
    "reactive_new_history",
    "bounded_perturbation_history",
    "reactive_new_history_tts",
    "driverl_full_window",
    "driverl_sample_full_window",
)


def _number(value, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("Finite evidence value required")
    return value


def validate_plan(plan):
    if plan.get("kind") != "capability_skill_plan" or plan.get("plan_id") != identity(
        {k: v for k, v in plan.items() if k != "plan_id"}
    ):
        raise ValueError("Capability skill plan changed")
    return plan


def validate_sample_lineage(plan, rows, files):
    """Recheck executed samples at BOTH compilation and feedback boundaries."""
    from driversi.agents.backends.native import digest
    from driversi.configuration import read_mapping
    from driversi.contracts.scene import Scene

    validate_plan(plan)
    selected = {a["id"]: a for a in plan["assignments"]}
    if len(selected) != len(plan["assignments"]):
        raise ValueError("Duplicate skill assignment IDs")
    assignment_ids = set()
    if not rows:
        raise ValueError("Skill attribution requires actual added samples")
    for row in rows:
        if row.get("kind") != "synthetic_navtrain":
            raise ValueError("Skill attribution requires synthetic samples")
        ref = row["admission"]
        if files.get(ref["path"]) != ref["sha256"] or digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Skill sample admission is not sealed by executed dataset")
        proof = read_mapping(ref["path"])
        origin = proof.get("source_train_membership", proof["parent"])
        scene = proof["branch"]["scene"]
        sha = proof["files"].get(scene)
        if not sha or files.get(scene) != sha or digest(scene) != sha:
            raise ValueError("Skill physical branch is not sealed by executed dataset")
        actual = Scene.load(Path(scene)).metadata.get("capability_skill")
        assignment = None
        if actual and actual.get("plan_id") == plan["plan_id"]:
            assignment = selected.get(actual.get("assignment_id"))
        else:
            matches = []
            for new_id, reference in plan.get("reused_executions", {}).items():
                if new_id not in selected:
                    raise ValueError("Reused reference has no selected assignment")
                if digest(reference["path"]) != reference["sha256"]:
                    raise ValueError("Reused generation receipt changed")
                generation = read_mapping(reference["path"])
                teacher = generation["teacher"]
                if digest(teacher["plan_path"]) != teacher["plan_sha256"]:
                    raise ValueError("Reused executed plan changed")
                executed_plan = validate_plan(read_mapping(teacher["plan_path"]))
                if not actual or (actual.get("plan_id"), actual.get("assignment_id")) != (
                    executed_plan["plan_id"],
                    teacher["assignment_id"],
                ):
                    continue
                original = next(
                    (
                        a
                        for a in executed_plan["assignments"]
                        if a["id"] == teacher["assignment_id"]
                    ),
                    None,
                )
                candidate = selected[new_id]
                ignored = {"id", "anchor_assignment_id"}
                if (
                    generation.get("status") != "rendered"
                    or original is None
                    or {k: v for k, v in original.items() if k not in ignored}
                    != {k: v for k, v in candidate.items() if k not in ignored}
                    or executed_plan["context"] != plan["context"]
                ):
                    raise ValueError("Reused execution changes source, controls, or parent context")
                report_path = generation["branch_report"]
                if generation["files"].get(report_path) != digest(report_path):
                    raise ValueError("Reused physical branch report changed")
                original_branch = read_mapping(report_path)
                if not any(
                    Path(p).name == "scene.npz" and h == sha
                    for p, h in original_branch["files"].items()
                ):
                    raise ValueError("Reused execution does not bind this physical scene")
                expected_old = dict(
                    plan_id=executed_plan["plan_id"],
                    assignment_id=original["id"],
                    skill=original["skill"],
                    skill_version=original["skill_version"],
                )
                if actual != expected_old:
                    raise ValueError("Synthetic physical branch lacks executed skill lineage")
                matches.append(candidate)
            if len(matches) != 1:
                raise ValueError("Executed scene requires one unambiguous reused assignment")
            assignment = matches[0]
        if (
            assignment is None
            or assignment["skill"] not in NEW_HISTORY_SKILLS
            or assignment["source_log"] != origin["source_log"]
            or assignment["token"] != origin["token"]
        ):
            raise ValueError("Synthetic branch does not implement selected skill/source")
        if actual["plan_id"] == plan["plan_id"] and actual != dict(
            plan_id=plan["plan_id"],
            assignment_id=assignment["id"],
            skill=assignment["skill"],
            skill_version=assignment["skill_version"],
        ):
            raise ValueError("Synthetic physical branch lacks executed skill lineage")
        assignment_ids.add(assignment["id"])
    return sorted(assignment_ids)


class SkillMemory:
    """Append-only experiment memory, idempotent across worker restarts."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outcomes (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.db.commit()

    def close(self):
        self.db.close()

    def record(self, outcome):
        key = outcome["evidence_id"]
        encoded = json.dumps(outcome, sort_keys=True, allow_nan=False)
        with self.db:
            old = self.db.execute("SELECT payload FROM outcomes WHERE id=?", (key,)).fetchone()
            if old and old[0] != encoded:
                raise ValueError("Conflicting outcome for the same experiment evidence")
            if outcome.get("supersedes") and not old:
                previous = self.db.execute(
                    "SELECT payload FROM outcomes WHERE id=?", (outcome["supersedes"],)
                ).fetchone()
                if previous is None:
                    raise ValueError("Execution review requires existing evidence")
                previous = json.loads(previous[0])
                fixed = (
                    "outcome_type",
                    "context",
                    "capability",
                    "skill",
                    "skill_version",
                    "token",
                    "cost_wall_seconds",
                    "execution_status",
                    "credited",
                )
                if (
                    outcome["outcome_type"] != "execution"
                    or outcome["credited"]
                    or any(outcome.get(k) != previous.get(k) for k in fixed)
                ):
                    raise ValueError("Execution review cannot rewrite identity, cost or gain")
                if any(
                    json.loads(raw).get("supersedes") == outcome["supersedes"]
                    for (raw,) in self.db.execute("SELECT payload FROM outcomes")
                ):
                    raise ValueError("Execution evidence already has a review")
            self.db.execute("INSERT OR IGNORE INTO outcomes VALUES (?,?)", (key, encoded))
        return {"recorded": old is None, "evidence_id": key, "credited": outcome["credited"]}

    def _current_rows(self):
        rows = [
            json.loads(r[0]) for r in self.db.execute("SELECT payload FROM outcomes ORDER BY id")
        ]
        superseded = {r["supersedes"] for r in rows if r.get("supersedes")}
        return [r for r in rows if r["evidence_id"] not in superseded]

    def statistics(self, context, capability, skill):
        rows = self._current_rows()
        family_keys = ("model", "evaluator_version", "benchmark")
        matched = [
            r
            for r in rows
            if all(r.get("context", {}).get(k) == context[k] for k in family_keys)
            and r.get("capability") == capability
            and r.get("skill") == skill
            and r.get("skill_version") == SKILLS[skill]["version"]
        ]
        credited = [r for r in matched if r["credited"] and r["context"] == context]
        prior = [r for r in matched if r["credited"] and r["context"] != context]
        attempts = [r for r in matched if r.get("outcome_type") == "execution"]
        executed = [
            r
            for r in attempts
            if r.get("failure_domain") not in {"platform", "world_validity_pending"}
            and r["execution_status"] not in {"preparation_or_execution_failed", "interrupted"}
        ]
        complete_cost = bool(credited) and all(r["cost_wall_seconds"] is not None for r in credited)
        cost = sum(r["cost_wall_seconds"] or 0 for r in credited)
        gain = sum(
            r["gain_vs_control"] if r["safety_passed"] else -abs(r["gain_vs_control"]) - 1
            for r in credited
        )
        return {
            "observations": len(matched),
            "credited_trials": len(credited),
            "cross_parent_prior_trials": len(prior),
            "cross_parent_prior_mean_gain": sum(
                r["gain_vs_control"] if r["safety_passed"] else -abs(r["gain_vs_control"]) - 1
                for r in prior
            )
            / len(prior)
            if prior
            else None,
            "cross_parent_prior_safety_failures": sum(not r["safety_passed"] for r in prior),
            "execution_attempts": len(attempts),
            "execution_failures": sum(
                r["execution_status"] != "trajectory_admitted" for r in executed
            ),
            "platform_failures": sum(
                r.get("failure_domain") == "platform"
                or r["execution_status"] in {"preparation_or_execution_failed", "interrupted"}
                for r in attempts
            ),
            "world_validity_pending": sum(
                r.get("failure_domain") == "world_validity_pending" for r in attempts
            ),
            "execution_wall_seconds": sum(r["cost_wall_seconds"] for r in attempts),
            "mean_execution_wall_seconds": sum(r["cost_wall_seconds"] for r in executed)
            / len(executed)
            if executed
            else None,
            "safety_failures": sum(not r["safety_passed"] for r in credited),
            "mean_gain_vs_control": gain / len(credited) if credited else None,
            "net_gain_per_wall_hour": gain / (cost / 3600) if complete_cost and cost > 0 else None,
            "cost_wall_seconds": cost,
            "complete_cost_evidence": complete_cost,
            "reliable_skill_improvement": False,
        }

    def attempted_tokens(self, context):
        return {
            r["token"]
            for r in self._current_rows()
            if r.get("outcome_type") == "execution"
            and r.get("context") == context
            and r.get("failure_domain") != "platform"
            and r["execution_status"] not in {"preparation_or_execution_failed", "interrupted"}
        }


def make_plan(candidates, *, model, memory, budget, advisor=None, evaluation_scope="locked_subset"):
    """No score query, evaluator invocation, synthesis admission or promotion."""
    feedback = validate_feedback(candidates["capability_feedback"])
    if not isinstance(model, str) or not model:
        raise ValueError("Explicit model family required")
    from driversi.rsi.decisions.model_services import student_contract

    contract = student_contract(model)
    if set(budget) != {"max_tasks", "max_wall_seconds_per_task"} or any(
        type(budget[k]) is not int or budget[k] <= 0 for k in budget
    ):
        raise ValueError("Positive bounded skill budget required")
    if budget["max_tasks"] > 64 or budget["max_wall_seconds_per_task"] > 3600:
        raise ValueError("Skill pilot budget too large")
    if evaluation_scope not in ("locked_subset", "full_coverage"):
        raise ValueError("Explicit supported feedback evaluation scope required")
    context = {
        "model": model,
        "parent_checkpoint": feedback["checkpoint"],
        "evaluator_version": "native_navsim_v1_pdm_40x0.1_" + evaluation_scope,
        "benchmark": feedback["benchmark"],
    }
    assignments, used = [], set()
    attempted = memory.attempted_tokens(context)
    groups = {}
    for row in candidates["candidates"]:
        groups.setdefault(row["capability"], []).append(row)
    # Round robin stops common progress deficits from consuming every slot.
    ordered = [
        group[i]
        for i in range(max(map(len, groups.values()), default=0))
        for group in groups.values()
        if i < len(group)
    ]
    for row in ordered:
        if (
            row["source_group"] in used
            or row["token"] in attempted
            or row["status"] != "condition_match_requires_diagnosis"
        ):
            continue
        options = []
        for skill in SKILLS:
            # A registered legacy relabel service is not yet a native ChainFlow
            # training bridge. Offer only executable new-history recipes here.
            if model in ("chainflow_stage1", "augustus") and skill == "same_anchor_relabel":
                continue
            stats = memory.statistics(context, row["capability"], skill)
            options.append({"skill": skill, "statistics": stats})
        # Untried procedures retain an exploration slot. With matched paired
        # evidence, compare measured gain/cost; this is a heuristic, not a theorem.
        options.sort(
            key=lambda x: (
                x["statistics"]["safety_failures"] > 0
                or x["statistics"]["cross_parent_prior_safety_failures"] > 0,
                x["statistics"]["execution_failures"] > 0,
                x["statistics"]["credited_trials"] > 0,
                -(
                    x["statistics"]["net_gain_per_wall_hour"]
                    if x["statistics"]["net_gain_per_wall_hour"] is not None
                    else x["statistics"]["mean_gain_vs_control"]
                    if x["statistics"]["mean_gain_vs_control"] is not None
                    else x["statistics"]["cross_parent_prior_mean_gain"] or 0
                ),
                x["statistics"]["mean_execution_wall_seconds"] or 0,
                x["skill"] not in ("same_anchor_relabel", "reactive_new_history"),
            )
        )
        chosen = options[0]["skill"]
        assignments.append(
            {
                "id": "assignment-" + str(len(assignments)),
                "token": row["token"],
                "source_log": row["source_log"],
                "source_group": row["source_group"],
                "capability": row["capability"],
                "skill": chosen,
                "skill_version": SKILLS[chosen]["version"],
                "recipe": SKILLS[chosen],
                "options": options,
                "condition_evidence": row["shared_conditions"],
                "retrieval_ranks": row["ranks"],
                "max_wall_seconds": budget["max_wall_seconds_per_task"],
                "requires_precondition_validation": True,
                "trainable": False,
            }
        )
        used.add(row["source_group"])
        if len(assignments) == budget["max_tasks"]:
            break
    advisory = {"status": "disabled", "requests": 0}
    if advisor is not None:
        from driversi.rsi.decisions.llm_advisor import advise_skills

        choices, advisory = advise_skills(advisor, assignments, feedback)
        for row in assignments:
            if row["id"] in choices:
                row["skill"] = choices[row["id"]]
                row["skill_version"] = SKILLS[row["skill"]]["version"]
                row["recipe"] = SKILLS[row["skill"]]
    plan = {
        "schema_version": 1,
        "kind": "capability_skill_plan",
        "context": context,
        "student_contract": contract,
        "capability_feedback": feedback,
        "assignments": assignments,
        "advisor": advisory,
        "budget": budget,
        "candidate_snapshot": candidates["snapshot_id"],
        "student_scores_required": False,
        "automatic_training": False,
        "trainable": False,
        "promoted": False,
        "model_cause_established": False,
        "policy": "source_diverse_capability_round_robin_then_explore_or_gain_per_cost",
        "evaluation_role": "adaptive_evaluation"
        if feedback["benchmark"] != "development"
        else "development",
    }
    return {**plan, "plan_id": identity(plan)}


def training_outcome(plan, feedback, evidence_id):
    """Consume native paired feedback; never infer attribution from a filename."""
    validate_plan(plan)
    binding = feedback.get("capability_skill_binding", {})
    assignments = {r["id"]: r for r in plan["assignments"]}
    ids = binding.get("assignment_ids", [])
    linked = (
        binding.get("plan_id") == plan["plan_id"]
        and bool(ids)
        and len(ids) == len(set(ids))
        and set(ids) <= assignments.keys()
    )
    selected = [assignments[x] for x in ids] if linked else []
    keys = {(x["capability"], x["skill"], x["skill_version"]) for x in selected}
    cost = feedback.get("total_experiment_wall_seconds")
    if cost is not None:
        _number(cost, nonnegative=True)
    gain = feedback.get("treatment_minus_continued_training", {}).get("score")
    if gain is not None:
        _number(gain)
    safety = not feedback.get("regressions_against_control", []) and not feedback.get(
        "weakness_reviews", {}
    ).get("treatment", {}).get("local_safety_regressions", [])
    reasons = []
    from driversi.rsi.decisions.model_services import student_contract

    if not student_contract(plan["context"]["model"])["paired_skill_feedback"]:
        reasons.append("student_skill_paired_feedback_not_connected")
    if not linked:
        reasons.append("unlinked_intervention")
    if len(keys) != 1:
        reasons.append("mixed_or_unknown_skill_attribution")
    if binding.get("evaluation_split") not in {
        "official_navtest",
        "official_navtest_locked_subset_pilot",
    }:
        reasons.append("evaluation_protocol_not_navtest")
    if (
        binding.get("model") != plan["context"]["model"]
        or binding.get("parent_checkpoint") != plan["context"]["parent_checkpoint"]
    ):
        reasons.append("model_or_parent_mismatch")
    if (
        binding.get("verified_benchmark") != plan["context"]["benchmark"]
        or binding.get("verified_evaluator_version") != plan["context"]["evaluator_version"]
    ):
        reasons.append("executed_evaluator_mismatch")
    if (
        feedback.get("kind") != "chainflow_paired_feedback"
        or feedback.get("status") != "completed"
        or feedback.get("effective_recipe_verified") is not True
        or not feedback.get("exposed_added_sample_ids")
        or feedback.get("unconsumed_added_sample_ids")
    ):
        reasons.append("paired_intervention_not_verified")
    if gain is None:
        reasons.append("missing_measured_gain")
    capability, skill, version = next(iter(keys)) if len(keys) == 1 else (None, None, None)
    return {
        "evidence_id": evidence_id,
        "outcome_type": "paired_training",
        "plan_id": plan["plan_id"],
        "context": plan["context"],
        "capability": capability,
        "skill": skill,
        "skill_version": version,
        "credited": not reasons,
        "reasons": reasons,
        "safety_passed": safety,
        "gain_vs_control": gain,
        "cost_wall_seconds": cost,
        "paired_training_evaluation_wall_seconds": feedback.get(
            "paired_training_evaluation_wall_seconds"
        ),
        "cost_complete": cost is not None and cost > 0,
        "training_was_executed_this_call": False,
        "stable_improvement_demonstrated": False,
    }
