"""Dataset-bound native learning and matched interventions, under DriveHarness.

The shared lifecycle owns optimizer/resume/evaluation. This adapter only binds a
typed dataset to its supported model mode; it never substitutes a loss function.
"""

import copy
from pathlib import Path

from driversi.agents.backends.native import asset_files, digest, path_at, project_root, signature
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.synthesis import check_files
from driversi.rsi.plugins.training import prepare_job
from driversi.runtime.budget import write_json
from driversi.training.contracts.job import resolve_request
from driversi.training.orchestration import lifecycle


def prepare_training(config, base, registry):
    prepared, cap = prepare_job(config, base, registry)
    resolved = resolve_request(prepared["request"])
    if (
        resolved["request"]["schema_version"] != 2
        or resolved["model"] != "augustus"
        or resolved["request"]["mode"] != "supervised"
    ):
        raise ValueError(
            "Offline AUGUstus labels require its native supervised backend; not RL transitions or world-model targets"
        )
    model = read_mapping(prepared["model_config"])
    parent = path_at(project_root(), model["weights"]["parent"])
    prepared["parent_files"] = {p: digest(p) for p in asset_files(parent)}
    if not prepared["parent_files"]:
        raise ValueError("Fixed parent assets are missing")
    return prepared, cap


def finish(output, report, binding):
    write_json(output / "binding.json", binding)
    return result(
        output,
        {
            "model": report["model"],
            "status": report["status"],
            "dataset": binding["dataset_identity"],
        },
        status=report["status"],
        reason=report.get("reason"),
        steps=report["consumed"].get(
            "sim_agent_steps",
            report["consumed"].get("scoring_agent_steps", 0)
            + report["consumed"].get("environment_steps", 0),
        ),
        artifacts={
            "report": str(output / "native/training-result.json"),
            "plan": str(output / "native/training-plan.json"),
            "binding": str(output / "binding.json"),
        },
    )


def run_training(config, inputs, output, registry):
    check_files(
        {
            config["request"]: config["request_sha256"],
            config["model_config"]: config["model_config_sha256"],
        }
    )
    check_files(config["parent_files"])
    ds = inputs["dataset"]
    receipt = read_mapping(ds["artifacts"]["report"])
    check_files(receipt["cache_files"])
    if digest(ds["artifacts"]["manifest"]) != receipt["manifest_sha256"]:
        raise ValueError("Generated dataset manifest changed")
    resolved = resolve_request(config["request"])
    request = copy.deepcopy(resolved["request"])
    source_data = read_mapping(Path(config["request"]).parent / request["data"])
    source_manifest = read_mapping(source_data["manifest"])
    manifest = read_mapping(ds["artifacts"]["manifest"])
    if (
        receipt["source_manifest_sha256"] != digest(source_data["manifest"])
        or any(
            manifest["splits"][s] != source_manifest["splits"][s]
            for s in ("validation", "development")
        )
        or manifest["splits"]["train"]["records"] != source_manifest["splits"]["train"]["records"]
    ):
        raise ValueError("Label intervention changed input pool or heldout set")
    output.mkdir(parents=True)
    source_data["manifest"] = ds["artifacts"]["manifest"]
    write_json(output / "data.json", source_data)
    request.update(model_config=config["model_config"], data=str(output / "data.json"))
    write_json(output / "request.json", request)
    binding = {
        "request": str(output / "request.json"),
        "dataset_identity": ds["summary"]["artifact_reference"]["identity"],
        "dataset_receipt": ds["artifacts"]["report"],
        "dataset_receipt_sha256": digest(ds["artifacts"]["report"]),
        "source_manifest": read_mapping(
            Path(config["request"]).parent / resolved["request"]["data"]
        )["manifest"],
        "comparison_identity": signature(
            {
                "parent_recipe": config["model_config_sha256"],
                "parent_assets": config["parent_files"],
                "pilot": request["pilot"],
                "budget": request["budget"],
                "source_manifest": receipt["source_manifest_sha256"],
                "source_files": receipt["source_cache_files"],
            }
        ),
        "label_source": receipt["label_source"],
        "sensor_identity": receipt["sensor_identity"],
        "search_identity": receipt["search_identity"],
        "files": {
            str(output / "request.json"): digest(output / "request.json"),
            str(output / "data.json"): digest(output / "data.json"),
            ds["artifacts"]["manifest"]: receipt["manifest_sha256"],
            config["model_config"]: config["model_config_sha256"],
            **receipt["cache_files"],
            **config["parent_files"],
        },
    }
    plan = lifecycle.prepare(resolve_request(output / "request.json"), output / "native")
    report = lifecycle.execute(plan, own_process_group=False)
    check_files(binding["files"])
    return finish(output, report, binding)


def prepare_resume(config, base, registry):
    _keys(config, ("max_sim_agent_steps",), "bound resume budget")
    if (
        type(config["max_sim_agent_steps"]) is not int
        or not 0 <= config["max_sim_agent_steps"] <= 100000
    ):
        raise ValueError("Explicit continuation reservation required")
    return config, config["max_sim_agent_steps"]


def prepare_evaluation(config, base, registry):
    _keys(config, ("evaluator", "max_sim_agent_steps"), "bound independent evaluation")
    prepared, cap = prepare_resume(
        {"max_sim_agent_steps": config["max_sim_agent_steps"]}, base, registry
    )
    path = (base / config["evaluator"]).resolve(strict=True)
    return {**prepared, "evaluator": str(path), "sha256": digest(path)}, cap


def phase(config, inputs, output, registry, name):
    student = inputs["student"]
    binding = read_mapping(student["artifacts"]["binding"])
    check_files(binding["files"])
    check_files({binding["dataset_receipt"]: binding["dataset_receipt_sha256"]})
    dataset_receipt = read_mapping(binding["dataset_receipt"])
    check_files({dataset_receipt["manifest"]: dataset_receipt["manifest_sha256"]})
    resolved = resolve_request(binding["request"])
    cap = (
        resolved["request"]["budget"]["max_scoring_agent_steps"]
        + resolved["request"]["budget"]["max_environment_steps"]
    )
    if config["max_sim_agent_steps"] < cap:
        raise ValueError("Bound phase reservation is smaller than the native request")
    output.mkdir(parents=True)
    evaluator = None
    if name == "evaluate":
        check_files({config["evaluator"]: config["sha256"]})
        original = read_mapping(config["evaluator"])
        data = read_mapping(Path(binding["request"]).parent / resolved["request"]["data"])
        receipt = read_mapping(binding["dataset_receipt"])
        if original["dataset_manifest_sha256"] != receipt["source_manifest_sha256"]:
            raise ValueError("Independent evaluator does not bind the original heldout dataset")
        # Only the training cache path/labels changed. Heldout records and cache
        # content were sealed before the intervention, never selected by it.
        original["dataset_manifest_sha256"] = digest(data["manifest"])
        evaluator = output / "evaluator.json"
        write_json(evaluator, original)
        binding = {**binding, "independent_evaluator_sha256": config["sha256"]}
    plan = lifecycle.prepare(
        resolved,
        output / "native",
        phase=name,
        previous=student["artifacts"]["report"],
        evaluator=evaluator,
    )
    report = lifecycle.execute(plan, own_process_group=False)
    check_files(binding["files"])
    return finish(output, report, binding)


def run_resume(config, inputs, output, registry):
    return phase(config, inputs, output, registry, "resume")


def run_evaluation(config, inputs, output, registry):
    return phase(config, inputs, output, registry, "evaluate")


def prepare_feedback(config, base, registry):
    if config:
        raise ValueError("Paired feedback cannot change scoring thresholds")
    return {}, 0


def compare(control, treatment):
    for key in (
        "comparison_identity",
        "sensor_identity",
        "independent_evaluator_sha256",
        "search_identity",
    ):
        if control["binding"][key] != treatment["binding"][key]:
            raise ValueError("Unmatched paired experiment: " + key)
    if (control["binding"]["label_source"], treatment["binding"]["label_source"]) != (
        "original",
        "drivetto",
    ):
        raise ValueError("Explicit ordinary-training control required")
    if control["student"]["total_updates"] != treatment["student"]["total_updates"]:
        raise ValueError("Paired optimizer budgets differ")
    a, b = control["evaluation"], treatment["evaluation"]
    if (
        a["metrics"]["parent"] != b["metrics"]["parent"]
        or a["evaluated_sample_ids"] != b["evaluated_sample_ids"]
    ):
        raise ValueError("Paired evaluation starts or coverage differ")
    control_rows = {x["sample_id"]: x for x in a["paired_samples"]}
    treatment_rows = {x["sample_id"]: x for x in b["paired_samples"]}
    if (
        len(control_rows) != len(a["paired_samples"])
        or len(treatment_rows) != len(b["paired_samples"])
        or set(control_rows) != set(a["evaluated_sample_ids"])
        or control_rows.keys() != treatment_rows.keys()
    ):
        raise ValueError("Paired sample coverage differs")
    local = []
    for token, ar in control_rows.items():
        br = treatment_rows[token]
        if ar["source_log"] != br["source_log"] or ar["parent"] != br["parent"]:
            raise ValueError("Paired sample baseline differs")
        for metric in ("NC", "DAC", "DDC", "TTC", "C"):
            if br["candidate"][metric] < ar["candidate"][metric]:
                local.append(
                    {
                        "sample_id": token,
                        "source_log": ar["source_log"],
                        "metric": metric,
                        "control": ar["candidate"][metric],
                        "treatment": br["candidate"][metric],
                    }
                )
    from driversi.rsi.decisions.weaknesses import review

    feedbacks, reviews = {}, {}
    for name, arm in (("control", control), ("treatment", treatment)):
        feedbacks[name] = lifecycle.feedback(arm["student"], arm["evaluation"])
        reviews[name] = review(arm["student"], arm["evaluation"], feedbacks[name])
    deltas = {
        k: b["metrics"]["candidate"][k] - a["metrics"]["candidate"][k]
        for k in ("PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C")
    }
    unsafe = (
        bool(local)
        or bool(reviews["treatment"]["local_safety_regressions"])
        or any(deltas[k] < 0 for k in ("NC", "DAC", "DDC", "TTC", "C"))
    )
    benefit = (
        not unsafe
        and deltas["PDMS"] > 0
        and b["metrics"]["candidate"]["PDMS"] > b["metrics"]["parent"]["PDMS"]
    )
    return {
        "kind": "paired_synthesis_feedback",
        "treatment_minus_continued_training": deltas,
        "feedbacks": feedbacks,
        "weakness_reviews": reviews,
        "regressions_against_control": local,
        "observed_pilot_benefit": benefit,
        "promoted": False,
        "policy_activated": False,
        "scope": "current_run_log_heldout_parent_provenance_unknown",
        "next_round": {
            "action": "replicate_on_more_training_logs"
            if benefit
            else "diagnose_before_spending_more",
            "automatic_training": False,
            "score_threshold_changed": False,
        },
        "improver_learning_proven": False,
        "cost": {
            name: arm["student"]["consumed"]
            for name, arm in (("control", control), ("treatment", treatment))
        },
    }


def run_feedback(config, inputs, output, registry):
    arms = {}
    for name in ("control", "treatment"):
        s, e = inputs[name + "_student"], inputs[name + "_evaluation"]
        arms[name] = {
            "student": read_mapping(s["artifacts"]["report"]),
            "evaluation": read_mapping(e["artifacts"]["report"]),
            "binding": read_mapping(e["artifacts"]["binding"]),
        }
    report = compare(arms["control"], arms["treatment"])
    if (
        arms["treatment"]["binding"]["search_identity"]
        != inputs["search"]["summary"]["artifact_reference"]["identity"]
    ):
        raise ValueError("Feedback search did not produce the trained labels")
    replay = read_mapping(inputs["replay"]["artifacts"]["report"])
    trained = arms["treatment"]["student"]
    if replay["context"] != trained["context"] or replay["resume_from"] != trained["resume_from"]:
        raise ValueError("Recovery comparison uses a different checkpoint")
    report["resume_check"] = {
        "different_processes": replay["worker_pid"] != trained["worker_pid"],
        "same_state": replay["state_fingerprint"] == trained["state_fingerprint"],
    }
    if not all(report["resume_check"].values()):
        report["observed_pilot_benefit"] = False
        report["next_round"]["action"] = "repair_recovery_before_learning"
    report["search"] = read_mapping(inputs["search"]["artifacts"]["diagnostics"])
    report["training_cost_with_ancestors"] = {
        name: lifecycle.training_lineage(inputs[name + "_student"]["artifacts"]["report"])["cost"]
        for name in ("control", "treatment")
    }
    report["replay_cost"] = replay["consumed"]
    output.mkdir(parents=True)
    return result(output, report)


def register(registry, Provider):
    from driversi.rsi.plugins import chainflow

    chainflow.register(registry, Provider)
    for p in (
        Provider(
            "native_dataset_training",
            "1",
            (("dataset", "native_training_dataset"),),
            "bound_learning_result",
            prepare_training,
            run_training,
            ("report", "plan", "binding"),
        ),
        Provider(
            "native_dataset_resume",
            "1",
            (("student", "bound_learning_result"),),
            "bound_learning_result",
            prepare_resume,
            run_resume,
            ("report", "plan", "binding"),
        ),
        Provider(
            "native_dataset_evaluation",
            "1",
            (("student", "bound_learning_result"),),
            "bound_learning_evaluation",
            prepare_evaluation,
            run_evaluation,
            ("report", "plan", "binding"),
        ),
        Provider(
            "paired_synthesis_feedback",
            "1",
            (
                ("control_student", "bound_learning_result"),
                ("treatment_student", "bound_learning_result"),
                ("control_evaluation", "bound_learning_evaluation"),
                ("treatment_evaluation", "bound_learning_evaluation"),
                ("replay", "bound_learning_result"),
                ("search", "search_evidence"),
            ),
            "paired_synthesis_feedback",
            prepare_feedback,
            run_feedback,
            ("report",),
        ),
    ):
        registry.register(p)
