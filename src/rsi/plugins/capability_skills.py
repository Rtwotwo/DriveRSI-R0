"""Execute model-service skill selection and consume paired feedback in Harness."""
from pathlib import Path

from driversi.agents.backends.native import project_root, digest
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys
from driversi.rsi.decisions.capability_skills import SkillMemory, make_plan, training_outcome, validate_plan
from driversi.rsi.workflow.results import result


def memory_path(value, base):
    path = (base / value).resolve()
    if not path.is_relative_to(project_root().resolve() / "exp"):
        raise ValueError("Skill memory must live under exp")
    return path


def prepare(config, base, registry):
    _keys(config, ("model", "memory", "budget", "advisor",
                   *(k for k in ("evaluation_scope", "model_config") if k in config)), "capability skill selection")
    from driversi.rsi.decisions.model_services import student_contract
    student_contract(config["model"])
    if config["model"] == "augustus" and not config.get("model_config"):
        raise ValueError("AUGUstus selection requires its actual native model configuration")
    if config.get("model_config"):
        path = (base / config["model_config"]).resolve(strict=True)
        native = read_mapping(path)
        if native.get("model") != config["model"]:
            raise ValueError("Student model configuration mismatch")
        weight = native["weights"].get("parent", native["weights"].get("checkpoint"))
        if not weight:raise ValueError("Explicit parent checkpoint required")
        config = {**config, "model_config": str(path), "model_config_sha256": digest(path),
                  "expected_parent": str((project_root()/weight).resolve())}
    advisor = config["advisor"]
    if advisor is not None:
        from driversi.rsi.decisions.llm_advisor import validate_config
        advisor = validate_config(read_mapping((base / advisor).resolve(strict=True)))
    return {**config, "advisor": advisor, "memory": str(memory_path(config["memory"], base))}, 0


def execute(config, inputs, output, registry):
    candidates = read_mapping(inputs["candidates"]["artifacts"]["report"])
    if config.get("model_config"):
        if digest(config["model_config"]) != config["model_config_sha256"]:
            raise ValueError("Student configuration changed")
        actual = Path(candidates["capability_feedback"]["checkpoint"])
        if (project_root()/actual).resolve() != Path(config["expected_parent"]):
            raise ValueError("Capability feedback belongs to another student checkpoint")
    memory = SkillMemory(config["memory"])
    try:
        report = make_plan(candidates, model=config["model"], memory=memory,
                           budget=config["budget"], advisor=config["advisor"],
                           evaluation_scope=config.get("evaluation_scope", "locked_subset"))
    finally:
        memory.close()
    output.mkdir(parents=True)
    return result(output, report)


def prepare_outcome(config, base, registry):
    _keys(config, ("memory", "plan"), "capability skill feedback")
    plan = (base / config["plan"]).resolve(strict=True)
    validate_plan(read_mapping(plan))
    return {"memory": str(memory_path(config["memory"], base)),
            "plan": str(plan), "plan_sha256": digest(plan)}, 0


def execute_outcome(config, inputs, output, registry):
    if digest(config["plan"]) != config["plan_sha256"]:
        raise ValueError("Skill plan changed after registration")
    feedback_path = Path(inputs["feedback"]["artifacts"]["report"])
    evidence = training_outcome(read_mapping(config["plan"]), read_mapping(feedback_path), digest(feedback_path))
    memory = SkillMemory(config["memory"])
    try:
        written = memory.record(evidence)
    finally:
        memory.close()
    output.mkdir(parents=True)
    return result(output, {"kind": "capability_skill_memory_update", "outcome": evidence,
                           "write": written, "promoted": False, "trainable": False})


def register(registry, Provider):
    registry.register(Provider("navsim_capability_skill_selection", "1",
        (("candidates", "navsim_scene_expansion_candidates"),), "capability_skill_plan", prepare, execute, ("report",)))
    registry.register(Provider("capability_skill_feedback", "1",
        (("feedback", "chainflow_paired_feedback"),), "capability_skill_memory_update",
        prepare_outcome, execute_outcome, ("report",)))
