"""Frozen development evaluation and evidence-only recipe benefit feedback."""
from pathlib import Path

from driversi.agents.interfaces.config import load_inference, read_agent_config
from driversi.contracts.scene import Scene
from driversi.data.assets.mtgs_assets import file_sha256
from driversi.runtime.budget import StepBudget, write_json
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.artifacts import fingerprint
from driversi.rsi.plugins.dataset import result
from driversi.training.sampling.environment import validate_development_paths, validate_splits

PROTOCOL = "structured-paired-diagnostic-v1"


def evaluator_sources():
    from driversi.agents.traffic_control import background
    from driversi.contracts import driving_task
    from driversi.evaluation.search import teacher
    from driversi.sim.backends import pufferdrive
    from driversi.sim.backends.native_loader import load_native
    from driversi.training.sampling import environment
    return {str(Path(m.__file__).resolve()): file_sha256(m.__file__)
            for m in (teacher, environment, background, driving_task, pufferdrive, load_native())}


def prepare_evaluation(config, base, registry):
    _keys(config, ("protocol", "scenes", "max_sim_agent_steps"), "independent evaluation config")
    if config["protocol"] != PROTOCOL or not isinstance(config["scenes"], list):
        raise ValueError("Unsupported frozen evaluation protocol")
    scenes = [str((base / p).resolve(strict=True)) for p in config["scenes"]]
    groups = sorted(validate_development_paths(scenes))
    cap = config["max_sim_agent_steps"]
    needed = sum(int(s.controlled.sum()) * (len(s.times) - 1) * 2 for s in map(lambda p: Scene.load(Path(p)), scenes))
    if type(cap) is not int or cap < needed:
        raise ValueError("Reserve both parent and candidate full evaluation horizons")
    frozen = {**config, "scenes": scenes, "source_groups": groups,
        "source_hashes": {p: file_sha256(p) for p in scenes}, "evaluator_sources": evaluator_sources(),
        "rules": {"progress_weight": .02, "safety": "no_per_scene_collision_or_offroad_increase",
                  "benefit": "positive_mean_return_delta", "promotion": "never_automatic", "policy_scope": "all_controlled_actors",
                  "training_scope": "one_target_actor_correction"}}
    frozen["identity"] = fingerprint(frozen)
    return frozen, cap


def run_evaluation(config, inputs, output, registry):
    from driversi.evaluation.search.teacher import evaluate
    from driversi.rsi.execution.registry import ExperimentRegistry
    if config["identity"] != fingerprint({k: v for k, v in config.items() if k != "identity"}):
        raise ValueError("Evaluation protocol changed")
    if evaluator_sources() != config["evaluator_sources"] or any(file_sha256(p) != h for p, h in config["source_hashes"].items()):
        raise ValueError("Evaluation code or scenes changed since registration")
    if set(config["source_groups"]) & ExperimentRegistry(registry).protected_groups():
        raise ValueError("Protected logs cannot be used for iterative feedback")
    dataset, student = inputs["dataset"], inputs["student"]
    if any(file_sha256(p) != h for p, h in student["summary"]["implementation_files"].items()):
        raise ValueError("Inference implementation changed since training declaration")
    identity = dataset["summary"]["artifact_reference"]["identity"]
    if student["summary"]["dataset_identity"] != identity:
        raise ValueError("Student was trained on a different dataset")
    if set(student["summary"]["source_groups"]) & set(config["source_groups"]):
        raise ValueError("Parent or correction training log overlaps evaluation")
    train_paths = [Path(dataset["artifacts"]["scene"])]
    validate_splits(train_paths, config["scenes"])
    output.mkdir(parents=True, exist_ok=False)
    budget = StepBudget(output / "budget.json", config["max_sim_agent_steps"], "independent_evaluation")
    agent = read_agent_config(Path(student["artifacts"]["agent"]))
    measurements = {}
    for arm in ("parent", "candidate"):
        policy = load_inference(Path(student["artifacts"][arm]), agent)
        measurements[arm] = evaluate(policy, config["scenes"], training_paths=train_paths,
            progress_weight=config["rules"]["progress_weight"], before_step=budget.before_step)
        write_json(output / (arm + ".json"), measurements[arm])
    pairs = []
    for before, after in zip(measurements["parent"]["episodes"], measurements["candidate"]["episodes"]):
        assert before["scene"] == after["scene"]
        pairs.append({"scene": before["scene"], "collision_delta": after["collision"] - before["collision"],
            "offroad_delta": after["offroad"] - before["offroad"],
            "return_delta": after["return_sum"] - before["return_sum"]})
    safety = all(p["collision_delta"] <= 0 and p["offroad_delta"] <= 0 for p in pairs)
    delta = measurements["candidate"]["return_sum_per_actor"] - measurements["parent"]["return_sum_per_actor"]
    report = {"protocol": config, "measurements": measurements, "pairs": pairs,
        "dataset_identity": identity, "student_identity": student["summary"]["artifact_reference"]["identity"],
        "source_separation_checked": True, "safety_nonregression": safety, "return_delta_per_actor": delta,
        "observed_benefit": safety and delta > 1e-8, "promoted": False,
        "interpretation": "paired_development_change_not_causal_or_teacher_qualification"}
    write_json(output / "report.json", report)
    return result(output, report, steps=budget.state["admitted_agent_steps"], artifacts={
        "report": str(output / "report.json"), "parent_metrics": str(output / "parent.json"),
        "candidate_metrics": str(output / "candidate.json")})


def prepare_feedback(config, base, registry):
    _keys(config, (), "feedback config")
    return config, 0


def run_feedback(config, inputs, output, registry):
    dataset, student, evaluation = (inputs[k] for k in ("dataset", "student", "evaluation"))
    data_id = dataset["summary"]["artifact_reference"]["identity"]
    student_id = student["summary"]["artifact_reference"]["identity"]
    ev = evaluation["summary"]
    if ev["dataset_identity"] != data_id or ev["student_identity"] != student_id:
        raise ValueError("Feedback lineage mismatch")
    st = student["summary"]
    report = {"recipe": dataset["summary"]["recipe"], "dataset_identity": data_id,
        "student_identity": student_id, "parent_sha256": st["parent_sha256"],
        "candidate_sha256": st["candidate_sha256"], "protocol_identity": ev["protocol"]["identity"],
        "training_sources": st["source_groups"], "evaluation_sources": ev["protocol"]["source_groups"],
        "samples_consumed": st["samples_consumed"], "updates": st["updates"],
        "return_delta_per_actor": ev["return_delta_per_actor"], "safety_nonregression": ev["safety_nonregression"],
        "observed_benefit": ev["observed_benefit"], "promoted": False,
        "outcome": "observed_benefit_requires_replication" if ev["observed_benefit"] else "no_benefit_or_regression",
        "cost": {"repair_agent_steps": dataset["summary"]["repair_agent_steps"],
                 "training_wall_seconds": st["wall_seconds"], "sample_exposures": st["sample_exposures"],
                 "evaluation_agent_steps": evaluation["sim_agent_steps"], "dataset_replay_agent_steps": dataset["sim_agent_steps"]},
        "use": "cross_log_feedback_for_review_not_automatic_recipe_promotion"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    return result(output, report)
