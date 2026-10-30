"""Paired ChainFlow evidence over the shared native training lifecycle.

This plugin consumes sealed training/evaluation outputs. It cannot authorize
synthetic data or change an evaluator, model recipe or training loss.
"""

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.decisions.paired_evidence import assess
from driversi.rsi.decisions.weaknesses import review
from driversi.rsi.workflow.results import result
from driversi.training.orchestration import lifecycle
from driversi_agents.chainflow.data import validate_manifest

SAFETY = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "comfort",
)


def prepare_feedback(config, base, registry):
    if not config:
        return {}, 0
    if set(config) != {"skill_binding"}:
        raise ValueError("Only an explicit capability skill binding is allowed")
    binding = config["skill_binding"]
    from driversi.rsi.decisions.capability_skills import validate_plan

    if digest(binding["plan"]) != binding["plan_sha256"]:
        raise ValueError("Capability plan changed")
    plan = validate_plan(read_mapping(binding["plan"]))
    if binding["plan_id"] != plan["plan_id"]:
        raise ValueError("Foreign capability plan")
    return config, 0


def compare(control, treatment):
    """Equal-budget control; diagnostic data and identical arms earn no benefit."""
    a, b = control, treatment
    for arm in (a, b):
        student, evaluation, plan = arm["student"], arm["evaluation"], arm["plan"]
        if student["model"] != "chainflow_stage1":
            raise ValueError("ChainFlow paired feedback cannot mix student families")
        arm["feedback"] = lifecycle.feedback(student, evaluation)
        arm["weaknesses"] = review(student, evaluation, arm["feedback"])
        if student["total_updates"] <= 0 or plan["phase"] not in ("train", "resume"):
            raise ValueError("Actual training updates required")
    if a["plan"]["model_config"] != b["plan"]["model_config"]:
        raise ValueError("Paired native recipe or parent weights differ")
    for key in ("pilot", "budget"):
        if a["plan"]["resolved"]["request"][key] != b["plan"]["resolved"]["request"][key]:
            raise ValueError("Paired training budget/control differs")
    recipes = [arm["student"].get("effective_training_recipe") for arm in (a, b)]
    if recipes[0] != recipes[1]:
        raise ValueError("Paired effective training recipes differ")
    if a["student"]["total_updates"] != b["student"]["total_updates"]:
        raise ValueError("Paired completed optimizer updates differ")
    da, db = a["dataset"], b["dataset"]
    if da["splits"]["development"] != db["splits"]["development"]:
        raise ValueError("Paired independent development coverage differs")
    original_order = [r["sample_id"] for r in da["splits"]["train"]]
    treatment_order = [r["sample_id"] for r in db["splits"]["train"]]
    original_members = set(original_order)
    if [k for k in treatment_order if k in original_members] != original_order:
        raise ValueError("Paired original sample order changed")
    original = {r["sample_id"]: r for r in da["splits"]["train"]}
    augmented = {r["sample_id"]: r for r in db["splits"]["train"]}
    if not original.keys() <= augmented.keys() or any(
        augmented[k] != v for k, v in original.items()
    ):
        raise ValueError("Augmentation must retain its original-data control pool")
    for row in original.values():
        if row['kind'] == 'synthetic_navtrain':
            from driversi_agents.chainflow.synthetic import validate_admission
            validate_admission(row, da['files'])
        elif row['kind'] != 'original_navtrain':
            raise ValueError('Control pool must contain original data or admitted shared replay')
    common_files = set(da["files"]) & set(db["files"])
    if any(da["files"][k] != db["files"][k] for k in common_files):
        raise ValueError("Paired common input contents differ")
    ea, eb = a["evaluation"], b["evaluation"]
    for field in ("directions", "primary_metric", "safety_higher", "metric_scope"):
        if ea.get(field) != eb.get(field):
            raise ValueError("Paired evaluation protocol differs")
    if ea["metrics"]["parent"] != eb["metrics"]["parent"]:
        raise ValueError("Parent evaluation parity not established")
    ar = {r["sample_id"]: r for r in ea["paired_samples"]}
    br = {r["sample_id"]: r for r in eb["paired_samples"]}
    if ar.keys() != br.keys() or any(
        ar[k]["parent"] != br[k]["parent"] or ar[k]["source_log"] != br[k]["source_log"] for k in ar
    ):
        raise ValueError("Paired sample parents or evaluation identity differ")
    local = [
        {
            "sample_id": k,
            "metric": metric,
            "control": ar[k]["candidate"][metric],
            "treatment": br[k]["candidate"][metric],
        }
        for k in ar
        for metric in SAFETY
        if br[k]["candidate"][metric] < ar[k]["candidate"][metric]
    ]
    deltas = {
        k: eb["metrics"]["candidate"][k] - ea["metrics"]["candidate"][k]
        for k in ea["metrics"]["candidate"]
    }
    added = sorted(augmented.keys() - original.keys())
    exposures = {}
    for name, arm, pool in (("control", a, original), ("treatment", b, augmented)):
        trace = arm["student"].get("sample_trace", [])
        batch_size = arm["plan"]["resolved"]["request"]["pilot"]["batch_size"]
        if arm["plan"]["resolved"]["request"].get("schema_version") == 4:
            world = arm["student"]["effective_training_recipe"]["world_size"]
            batch_size *= world
            if any(
                len(x.get("rank_sample_ids", [])) != world
                or [s for rank in x["rank_sample_ids"] for s in rank] != x["sample_ids"]
                for x in trace
            ):
                raise ValueError("DDP rank exposure does not match global sample trace")
        if (
            len(trace) != arm["student"]["total_updates"]
            or [x["optimizer_step"] for x in trace] != list(range(1, len(trace) + 1))
            or any(
                len(x["sample_ids"]) != batch_size or any(k not in pool for k in x["sample_ids"])
                for x in trace
            )
        ):
            raise ValueError("Confirmed native optimizer sample exposure is missing or mismatched")
        exposures[name] = [k for x in trace for k in x["sample_ids"]]
    exposed_added = sorted(set(added) & set(exposures["treatment"]))
    evidence = assess(
        [
            {
                "sample_id": k,
                "source_log": ar[k]["source_log"],
                "control": ar[k]["candidate"],
                "treatment": br[k]["candidate"],
            }
            for k in ar
        ],
        primary_metric="score",
        safety_metrics=SAFETY,
    )
    unconsumed_added = sorted(set(added) - set(exposed_added))
    benefit = bool(
        exposed_added
        and not unconsumed_added
        and not local
        and not b["weaknesses"]["local_safety_regressions"]
        and deltas["score"] > evidence["policy"]["score_effect_floor"]
        and b["feedback"]["pilot_metric_improved"]
    )
    return {
        "kind": "chainflow_paired_feedback",
        "status": "completed",
        "treatment_minus_continued_training": deltas,
        "regressions_against_control": local,
        "added_sample_ids": added,
        "exposed_added_sample_ids": exposed_added,
        "unconsumed_added_sample_ids": unconsumed_added,
        "sample_exposures": exposures,
        "observed_pilot_benefit": benefit,
        "paired_evidence": evidence,
        "effective_recipe_verified": recipes[0] is not None,
        "stable_improvement_demonstrated": False,
        "scope": "augmentation_pilot" if added else "identical_data_plumbing_only",
        "synthetic_gain_demonstrated": benefit
        and any(augmented[k]["kind"] == "synthetic_navtrain" for k in exposed_added),
        "weakness_reviews": {"control": a["weaknesses"], "treatment": b["weaknesses"]},
        "feedbacks": {"control": a["feedback"], "treatment": b["feedback"]},
        "promoted": False,
        "policy_activated": False,
        "improver_learning_proven": False,
        "next_action": (
            "diagnose_safety_regression"
            if local or b["weaknesses"]["local_safety_regressions"]
            else "verify_real_data_intervention"
            if not exposed_added or unconsumed_added
            else "expand_independent_log_evaluation"
            if evidence["status"] == "insufficient_independent_logs"
            else "replicate_on_more_independent_logs"
            if benefit
            else "diagnose_before_more_training"
        ),
    }


def run_feedback(config, inputs, output, registry):
    arms = {}
    for side in ("control", "treatment"):
        student = inputs[side + "_student"]
        plan = read_mapping(student["artifacts"]["plan"])
        dataset = validate_manifest(plan["data"]["manifest"], groups=plan["source_groups"])
        for path, sha in plan["files"].items():
            if digest(path) != sha:
                raise ValueError("Paired training inputs changed")
        arms[side] = {
            "student": read_mapping(student["artifacts"]["report"]),
            "evaluation": read_mapping(inputs[side + "_evaluation"]["artifacts"]["report"]),
            "plan": plan,
            "dataset": dataset,
        }
    report = compare(arms["control"], arms["treatment"])
    report["training_cost_with_ancestors"] = {
        side: lifecycle.training_lineage(inputs[side + "_student"]["artifacts"]["report"])
        for side in arms
    }
    receipts = {}
    for lineage in report["training_cost_with_ancestors"].values():
        for receipt in lineage["receipts"]:
            receipts.setdefault(receipt["sha256"], receipt["path"])
    report["unique_training_cost"] = {
        key: sum(read_mapping(path)["consumed"].get(key, 0) for path in receipts.values())
        for key in ("wall_seconds", "scoring_agent_steps", "environment_steps")
    }
    report["unique_training_receipts"] = list(receipts.values())
    report["cost_scope"] = "shared_training_ancestors_counted_once_evaluation_cost_separate"
    report["paired_training_evaluation_wall_seconds"] = report["unique_training_cost"][
        "wall_seconds"
    ] + sum(arm["evaluation"].get("consumed", {}).get("wall_seconds", 0) for arm in arms.values())
    # Rollout/render costs are not in these receipts. Never present partial
    # training cost as the end-to-end price of a synthesis skill.
    report["total_experiment_wall_seconds"] = None
    if config.get("skill_binding"):
        binding = dict(config["skill_binding"])
        if digest(binding["plan"]) != binding["plan_sha256"]:
            raise ValueError("Capability skill plan changed during training")
        if sorted(binding["added_sample_ids"]) != sorted(report["added_sample_ids"]):
            raise ValueError("Feedback does not match selected skill's training intervention")
        from driversi.rsi.decisions.capability_skills import validate_sample_lineage

        actual_rows = [
            row
            for row in arms["treatment"]["dataset"]["splits"]["train"]
            if row["sample_id"] in report["added_sample_ids"]
        ]
        actual_ids = validate_sample_lineage(
            read_mapping(binding["plan"]), actual_rows, arms["treatment"]["dataset"]["files"]
        )
        if actual_ids != sorted(binding["assignment_ids"]):
            raise ValueError("Executed sample skill assignments differ from feedback binding")
        # Parent identity comes from the executed recipe, not the proposal alone.
        model_config = arms["treatment"]["plan"]["model_config"]
        if isinstance(model_config, str):
            model_config = read_mapping(model_config)
        from pathlib import Path

        from driversi.agents.backends.native import project_root

        checkpoint = model_config.get("weights", {}).get("checkpoint")
        if (
            checkpoint is None
            or (project_root() / checkpoint).resolve()
            != Path(binding["parent_checkpoint"]).resolve()
        ):
            raise ValueError("Executed parent checkpoint differs from skill plan")
        actual_scope = arms["treatment"]["evaluation"]["metric_scope"]
        expected_scopes = {
            "official_navtest": (
                "official_navtest_v1_full_coverage_adaptive_evaluation",
                "full_coverage",
            ),
            "official_navtest_locked_subset_pilot": (
                "official_navtest_v1_locked_subset_adaptive_pilot_not_full_benchmark",
                "locked_subset",
            ),
        }
        expected = expected_scopes.get(binding["evaluation_split"])
        if expected is None or actual_scope != expected[0]:
            raise ValueError("Executed evaluation scope differs from skill binding")
        context = (arms["treatment"]["evaluation"].get("evaluation_context") or {}).get("value", {})
        if context.get("protocol") != "chainflow_native_navsim_pdm_v1":
            raise ValueError("Executed evaluator is not the declared native PDM protocol")
        binding["verified_benchmark"] = "navtest"
        binding["verified_evaluator_version"] = "native_navsim_v1_pdm_40x0.1_" + expected[1]
        report["capability_skill_binding"] = binding
    output.mkdir(parents=True)
    return result(output, report)


def register(registry, Provider):
    from driversi.rsi.plugins.chainflow_pre_eval import register as register_gate

    register_gate(registry, Provider)
    registry.register(
        Provider(
            "chainflow_paired_feedback",
            "1",
            (
                ("control_student", "unified_training_result"),
                ("treatment_student", "unified_training_result"),
                ("control_evaluation", "unified_learning_evaluation"),
                ("treatment_evaluation", "unified_learning_evaluation"),
            ),
            "chainflow_paired_feedback",
            prepare_feedback,
            run_feedback,
            ("report",),
        )
    )
