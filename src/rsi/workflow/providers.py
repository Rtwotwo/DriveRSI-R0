"""DriveHarness service providers. Trusted platform code, never model-supplied scripts.

Design reference: deepseek-ai/deepseek-harness 86a4022 workflow service/provider
separation. This is an independent Python implementation; no DSH runtime dependency.
"""

from dataclasses import dataclass
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys


@dataclass(frozen=True)
class Provider:
    name: str
    version: str
    inputs: tuple
    output: str
    prepare: object
    execute: object
    artifacts: tuple = ()

    def contract(self):
        return {
            "schema_version": 1,
            "inputs": dict(self.inputs),
            "output": self.output,
            "required_artifacts": list(self.artifacts),
        }

    def validate_output(self, result):
        if result["status"] == "completed" and any(
            not isinstance(result["artifacts"].get(name), str) or not result["artifacts"][name]
            for name in self.artifacts
        ):
            raise ValueError("Provider output is missing required artifacts: " + self.name)


class Providers:
    def __init__(self):
        self._items = {}

    def register(self, provider):
        if provider.name in self._items:
            raise ValueError("Duplicate provider: " + provider.name)
        self._items[provider.name] = provider

    def get(self, name):
        if name not in self._items:
            raise ValueError("Unsupported provider: " + str(name))
        return self._items[name]

    def describe(self):
        return [
            {
                "name": p.name,
                "version": p.version,
                "inputs": dict(p.inputs),
                "output": p.output,
                "contract": p.contract(),
            }
            for p in self._items.values()
        ]


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit input path required")
    return str((base / value).resolve(strict=True))


def prepare_repair(config, base, registry):
    from driversi.rsi.correction.recipes import prepare

    _keys(config, ("spec",), "repair provider config")
    path = Path(_path(config["spec"], base))
    spec = prepare(read_mapping(path), path.parent)
    return {"spec": spec}, spec["budget"]["max_sim_agent_steps"]


def run_repair(config, inputs, output, registry):
    from driversi.rsi.correction.recipes import run_prepared
    from driversi.rsi.correction.store import RepairStore

    state = run_prepared(RepairStore(registry), config["spec"], output)
    artifacts = {"report": str(output / "report.json")}
    if state["status"] == "completed":
        artifacts.update(scene=str(output / "scene.npz"), anchor=str(output / "anchor.npz"))
        artifacts.update(
            {
                "rollout:" + a["recipe"]["id"]: str(output / a["rollout"])
                for a in state["attempts"]
                if a.get("rollout")
            }
        )
    return {
        "status": state["status"],
        "reason": state["failure_reason"],
        "sim_agent_steps": state.get("admitted_agent_steps", 0),
        "artifacts": artifacts,
        "summary": {
            "selected_recipe": state.get("selected_recipe"),
            "attempts": len(state["attempts"]),
            "attempt_outcomes": [
                {
                    "recipe": a["recipe"]["id"],
                    "status": a["status"],
                    "metrics": a.get("metrics"),
                    "cost": a.get("charged_agent_steps", 0),
                }
                for a in state["attempts"]
            ],
            "anchor_verified": state.get("anchor_verified", False),
        },
        "admission": {
            "trajectory": "engineering_only",
            "visual": "not_checked",
            "trainable": False,
        },
    }


def prepare_admission(config, base, registry):
    _keys(config, ("requirement",), "admission config")
    if config["requirement"] not in ("engineering_diagnostic", "qualified_supervision"):
        raise ValueError("Unknown data admission requirement")
    return config, 0


def run_admission(config, inputs, output, registry):
    import json

    from driversi.runtime.budget import write_json

    report = json.loads(Path(inputs["evidence"]["artifacts"]["report"]).read_text())
    selected = report.get("selected_recipe")
    accepted = any(
        row["recipe"]["id"] == selected
        and row["status"] == "completed"
        and row["metrics"]["engineering_pass"]
        for row in report["attempts"]
    )
    # This service only understands the existing engineering verifier. A number
    # named official_score in an imported report is NOT a trusted scoring adapter.
    qualified = config["requirement"] == "qualified_supervision"
    ok = accepted and not qualified
    reason = (
        None
        if ok
        else (
            "independent_teacher_admission_not_connected"
            if accepted
            else "no_engineering_candidate"
        )
    )
    result = {
        "status": "completed" if ok else "rejected",
        "reason": reason,
        "sim_agent_steps": 0,
        "artifacts": {"report": str(output / "admission.json")},
        "summary": {
            "engineering_pass": accepted,
            "requirement": config["requirement"],
            "teacher_qualified": False,
            "selected_recipe": selected,
        },
        "admission": {
            "trajectory": "engineering_only",
            "visual": "not_checked",
            "trainable": False,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "admission.json", result)
    return result


def prepare_export(config, base, registry):
    from driversi.contracts.scene import Scene
    from driversi.rsi.execution.registry import ExperimentRegistry
    from driversi.training.sampling.environment import source_log_group, validate_development_split

    required = {"session", "scene", "anchor_tick", "future_images"}
    optional = {"cameras", "navigation"}
    if not isinstance(config, dict) or required - set(config) or set(config) - required - optional:
        raise ValueError("Invalid NAVSIM export config")
    if type(config["future_images"]) is not bool or type(config["anchor_tick"]) is not int:
        raise ValueError("Explicit image purpose and integer anchor required")
    config = dict(config)
    for key in ("session", "scene"):
        config[key] = _path(config[key], base)
    scene = Scene.load(Path(config["scene"]))
    validate_development_split(scene.metadata["split"])
    if (
        source_log_group(scene.metadata["source_log"])
        in ExperimentRegistry(registry).protected_groups()
    ):
        raise ValueError("Protected log cannot enter synthesis")
    from driversi.data.assets.mtgs_assets import file_sha256

    config["source_hashes"] = {k: file_sha256(config[k]) for k in ("session", "scene")}
    config["source_group"] = source_log_group(scene.metadata["source_log"])
    return config, 0


def run_export(config, inputs, output, registry):
    from driversi.data.assets.mtgs_assets import file_sha256
    from driversi.rsi.execution.registry import ExperimentRegistry

    if any(file_sha256(config[k]) != v for k, v in config["source_hashes"].items()):
        raise ValueError("Export source changed after task registration")
    if config["source_group"] in ExperimentRegistry(registry).protected_groups():
        raise ValueError("Source became protected after registration")
    from driversi.data.datasets.navsim_synthetic import CAMERAS, export_navsim_sample

    result = export_navsim_sample(
        config["session"],
        config["scene"],
        output,
        anchor_tick=config["anchor_tick"],
        cameras=config.get("cameras", CAMERAS),
        navigation=config.get("navigation"),
        include_future_images=config["future_images"],
    )
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": 0,
        "artifacts": {"report": str(output / "manifest.json")},
        "summary": {
            "clip_id": result["clip_id"],
            "image_counts": result["image_counts"],
            "rejection_reasons": result["rejection_reasons"],
        },
        "admission": {"trajectory": "unqualified", "visual": "unqualified", "trainable": False},
    }


def prepare_experiment(config, base, registry):
    from driversi.rsi.execution.execution_spec import resolve_execution
    from driversi.rsi.execution.jobs import JobStore

    _keys(config, ("spec",), "structured experiment config")
    path = Path(_path(config["spec"], base))
    raw = read_mapping(path)
    job = resolve_execution(raw, path.parent, JobStore(registry).plan(raw["plan_id"]))
    # Freeze source declarations now; execute rechecks them at the normal boundary.
    return {"spec": str(path), "resolved": job}, (
        job["max_train_agent_steps"] + job["evaluation_max_agent_steps"]
    )


def run_experiment(config, inputs, output, registry):
    from driversi.rsi.execution.execution_spec import resolve_execution
    from driversi.rsi.execution.executor import execute
    from driversi.rsi.execution.jobs import JobStore

    path = Path(config["spec"])
    raw = read_mapping(path)
    actual = resolve_execution(raw, path.parent, JobStore(registry).plan(raw["plan_id"]))
    if actual != config["resolved"]:
        raise ValueError("Experiment changed since task preparation")
    state = execute(registry, path, output)
    return {
        "status": state["status"],
        "reason": state["failure_reason"],
        "sim_agent_steps": state["cost"]["sim_agent_steps"],
        "artifacts": {"report": str(Path(state["directory"]) / "result.json")},
        "summary": {
            "job_id": state["id"],
            "evaluation": state["evaluation"],
            "cost": state["cost"],
            "input_kind": "structured_scenes",
            "teacher_labels_consumed": False,
        },
        "admission": {"model": "candidate_only", "promoted": False},
    }


def builtin_providers():
    from driversi.rsi.plugins import augustus as aug
    from driversi.rsi.plugins import learning
    from driversi.rsi.plugins import training as unified
    from driversi.rsi.plugins.dataset import prepare_dataset, run_dataset
    from driversi.rsi.plugins.feedback import (
        prepare_evaluation,
        prepare_feedback,
        run_evaluation,
        run_feedback,
    )
    from driversi.rsi.plugins.native import (
        prepare_inventory,
        prepare_preflight,
        run_inventory,
        run_preflight,
    )
    from driversi.rsi.plugins.student import prepare_student, run_student
    from driversi.rsi.plugins.tto import prepare_candidates, run_candidates

    registry = Providers()
    for provider in (
        Provider(
            "native_model_training_reference",
            "1",
            (),
            "unified_training_result",
            unified.prepare_reference,
            unified.run_reference,
            ("report", "plan", "origin"),
        ),
        Provider(
            "native_model_unified_resume",
            "1",
            (("student", "unified_training_result"),),
            "unified_training_result",
            unified.prepare_job,
            unified.run_resume,
            ("report", "plan"),
        ),
        Provider(
            "native_model_unified_evaluation",
            "1",
            (("student", "unified_training_result"),),
            "unified_learning_evaluation",
            unified.prepare_evaluation_job,
            unified.run_evaluation,
            ("report", "plan"),
        ),
        Provider(
            "native_model_unified_feedback",
            "1",
            (
                ("student", "unified_training_result"),
                ("replay", "unified_training_result"),
                ("evaluation", "unified_learning_evaluation"),
            ),
            "unified_learning_feedback",
            unified.prepare_feedback,
            unified.run_feedback,
            ("report",),
        ),
        Provider(
            "native_model_unified_training",
            "1",
            (),
            "unified_training_result",
            unified.prepare_job,
            unified.run_job,
            ("report", "plan"),
        ),
        Provider(
            "learning_decision",
            "1",
            (),
            "learning_decision",
            learning.prepare_decision,
            learning.run_decision,
            ("report",),
        ),
        Provider(
            "native_model_selected_training",
            "1",
            (("decision", "learning_decision"), ("dataset", "native_training_dataset")),
            "native_rsi_student",
            learning.prepare_selected,
            learning.run_selected,
            ("report", "candidate", "plan"),
        ),
        Provider(
            "drivetto_candidates",
            "2",
            (),
            "trajectory_proposals",
            prepare_candidates,
            run_candidates,
            ("report",),
        ),
        Provider(
            "native_model_dataset",
            "2",
            (("proposals", "trajectory_proposals"),),
            "native_training_dataset",
            aug.prepare_dataset,
            aug.run_dataset,
            ("report", "manifest"),
        ),
        Provider(
            "native_model_rsi_training",
            "2",
            (("dataset", "native_training_dataset"),),
            "native_rsi_student",
            aug.prepare_job,
            aug.run_training,
            ("report", "candidate", "plan"),
        ),
        Provider(
            "native_model_rsi_evaluation",
            "2",
            (("dataset", "native_training_dataset"), ("student", "native_rsi_student")),
            "native_rsi_evaluation",
            aug.prepare_job,
            aug.run_evaluation,
            ("report", "parent_plan", "candidate_plan", "parent_scores", "candidate_scores"),
        ),
        Provider(
            "native_model_learning_feedback",
            "2",
            (
                ("dataset", "native_training_dataset"),
                ("student", "native_rsi_student"),
                ("evaluation", "native_rsi_evaluation"),
            ),
            "native_learning_feedback",
            aug.prepare_feedback,
            aug.run_feedback,
            ("report",),
        ),
        Provider(
            "native_model_inventory",
            "1",
            (),
            "native_model_inventory",
            prepare_inventory,
            run_inventory,
            ("report",),
        ),
        Provider(
            "native_model_preflight",
            "1",
            (("inventory", "native_model_inventory"),),
            "native_model_preflight",
            prepare_preflight,
            run_preflight,
            ("report",),
        ),
        Provider(
            "student_imitation",
            "1",
            (("dataset", "correction_dataset"),),
            "correction_student",
            prepare_student,
            run_student,
            ("report", "parent", "candidate", "agent"),
        ),
        Provider(
            "student_evaluation",
            "1",
            (("dataset", "correction_dataset"), ("student", "correction_student")),
            "student_evaluation",
            prepare_evaluation,
            run_evaluation,
            ("report", "parent_metrics", "candidate_metrics"),
        ),
        Provider(
            "learning_feedback",
            "1",
            (
                ("dataset", "correction_dataset"),
                ("student", "correction_student"),
                ("evaluation", "student_evaluation"),
            ),
            "learning_feedback",
            prepare_feedback,
            run_feedback,
            ("report",),
        ),
        Provider(
            "correction_dataset",
            "1",
            (("repair", "repair_evidence"), ("admission", "diagnostic_evidence")),
            "correction_dataset",
            prepare_dataset,
            run_dataset,
            ("report", "samples", "scene"),
        ),
        Provider(
            "same_state_repair",
            "1",
            (),
            "repair_evidence",
            prepare_repair,
            run_repair,
            ("report", "scene", "anchor"),
        ),
        Provider(
            "repair_admission",
            "1",
            (("evidence", "repair_evidence"),),
            "diagnostic_evidence",
            prepare_admission,
            run_admission,
            ("report",),
        ),
        Provider(
            "navsim_export",
            "1",
            (),
            "navsim_inspection_dataset",
            prepare_export,
            run_export,
            ("report",),
        ),
        Provider(
            "structured_experiment", "1", (), "evaluated_model", prepare_experiment, run_experiment
        ),
    ):
        registry.register(provider)
    from driversi.rsi.plugins import (
        collection,
        navsim,
        navsim_teacher,
        paired,
        reactive,
        recovery,
        synthesis,
    )

    synthesis.register(registry, Provider)
    paired.register(registry, Provider)
    reactive.register(registry, Provider)
    navsim.register(registry, Provider)
    navsim_teacher.register(registry, Provider)
    from driversi.rsi.plugins import chainflow_synthetic, native_branch, worldengine

    native_branch.register(registry, Provider)
    worldengine.register(registry, Provider)
    from driversi.rsi.plugins import mtgs_window

    mtgs_window.register(registry, Provider)
    from driversi.rsi.plugins import conditioned_video

    conditioned_video.register(registry, Provider)
    chainflow_synthetic.register(registry, Provider)
    collection.register(registry, Provider)
    recovery.register(registry, Provider)
    from driversi.rsi.plugins import mining

    mining.register(registry, Provider)
    from driversi.rsi.plugins import scene_catalog

    scene_catalog.register(registry, Provider)
    from driversi.rsi.plugins import synthetic_classification
    synthetic_classification.register(registry, Provider)
    from driversi.rsi.plugins import capability_skills
    capability_skills.register(registry, Provider)
    from driversi.rsi.plugins import model_services
    model_services.register(registry, Provider)
    from driversi.rsi.plugins import vla_memory
    vla_memory.register(registry, Provider)
    from driversi.rsi.plugins import continuation
    continuation.register(registry, Provider)
    from driversi.rsi.plugins import frozen_skills
    frozen_skills.register(registry, Provider)
    from driversi.rsi.plugins import skill_evolution
    skill_evolution.register(registry, Provider)
    from driversi.rsi.plugins import dynamic_history
    dynamic_history.register(registry, Provider)
    return registry


def settle_interrupted_child(row, output, registry):
    """Recover only this stage's dead child, retaining unknown-cost reservations.

    A reused child ID is not ownership. Its output path must match this stage.
    No provider is executed or retried by this function.
    """
    if row["provider"] == "same_state_repair":
        from driversi.rsi.correction.store import RepairStore

        store = RepairStore(registry)
        child_id = row["config"]["spec"]["id"]
        expected = output
        missing = "Unknown repair run"
        recover_child = store.recover
        active = {"registered", "running"}
    elif row["provider"] == "structured_experiment":
        from driversi.rsi.execution.executor import recover
        from driversi.rsi.execution.jobs import JobStore

        store = JobStore(registry)
        child_id = row["config"]["resolved"]["id"]
        expected = output / child_id
        missing = "Unknown job"

        def recover_child(identity):
            return recover(store, identity)

        active = {"queued", "running"}
    else:
        return {"status": "no_child_ledger"}
    try:
        child = store.get(child_id)
    except ValueError as exc:
        if str(exc) != missing:
            raise
        return {"status": "not_started", "id": child_id}
    if Path(child["directory"]).resolve() != expected.resolve():
        return {"status": "foreign_child_identity", "id": child_id}
    if child["status"] in active:
        child = recover_child(child_id)
    return {"status": "settled", "id": child_id, "child_status": child["status"]}
