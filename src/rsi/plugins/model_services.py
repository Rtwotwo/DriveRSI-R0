"""Executable AUGUstus hard-case audits over sealed DriveRL branches."""
from pathlib import Path
import shutil

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys
from driversi.rsi.decisions.model_services import diagnose_augustus_branch
from driversi.rsi.workflow.results import result


def prepare(config, base, registry):
    _keys(config, ("teacher_config", "student_config"), "privileged teacher/student service")
    paths = {key: (base / value).resolve(strict=True) for key, value in config.items()}
    teacher, student = (read_mapping(paths[key]) for key in ("teacher_config", "student_config"))
    if teacher.get("model") != "driverl" or student.get("model") != "augustus":
        raise ValueError("DriveRL privileged teacher and AUGUstus student required")
    if student.get("family") != "vision_language_action" or student["capabilities"].get("train") is not True:
        raise ValueError("Declared native AUGUstus SFT adapter required")
    root = project_root()
    checkpoint = path_at(root, teacher["weights"]["checkpoint"])
    parent = path_at(root, student["weights"]["parent"])
    from driversi_agents.augustus.adapter import complete_assets
    assets = complete_assets(parent)
    missing = [p for p in assets if not Path(p).is_file()]
    required = ["config.yaml", "action_norm_stats.json", "config.json", "model.safetensors",
                "tokenizer.json", "tokenizer_config.json", "processor_config.json"]
    missing += [str(parent / name) for name in required if not (parent / name).is_file()]
    from driversi.data.catalog.store import identity
    parent_assets = {str(p): digest(p) for p in assets if Path(p).is_file()}
    return {"roles": {"teacher": "driverl_privileged_frozen", "student": "augustus",
                      "student_parent": str(parent), "teacher_checkpoint": str(checkpoint)},
            "student_parent_assets_sha256": identity(parent_assets),
            "student_parent_files": parent_assets,
            "files": {str(p): digest(p) for p in [*paths.values(), checkpoint]},
            "student_assets": {"structural_complete": not missing, "missing": missing,
                               "strict_tensor_load_tested_this_run": False},
            "native_student_runtime_available": path_at(root, student["python"]).is_file(),
            "source_groups": []}, 0


def execute(config, inputs, output, registry):
    for path, sha in {**config["files"], **config["student_parent_files"]}.items():
        if digest(path) != sha:
            raise ValueError("Teacher/student service configuration changed")
    branch = read_mapping(inputs["branch"]["artifacts"]["report"])
    checkpoint = config["roles"]["teacher_checkpoint"]
    if (branch.get("trajectory_evidence", {}).get("teacher") != "DriveRL_native_argmax"
            or branch.get("upstream_source_files", {}).get(checkpoint) != config["files"][checkpoint]):
        raise ValueError("Actual branch teacher differs from declared privileged teacher")
    report = diagnose_augustus_branch(branch)
    report.update(roles=config["roles"], student_assets=config["student_assets"],
                  native_student_runtime_available=config["native_student_runtime_available"],
                  source_branch=inputs["branch"]["summary"]["artifact_reference"]["identity"])
    output.mkdir(parents=True)
    return result(output, report)


def register(registry, Provider):
    registry.register(Provider("augustus_difficult_case_diagnosis", "2",
        (("branch", "selected_native_branch"),), "augustus_model_service_diagnosis",
        prepare, execute, ("report",)))
    registry.register(Provider("native_branch_reference", "1", (), "selected_native_branch",
        prepare_reference, execute_reference, ("report", "trajectory_window", "origin")))


def prepare_reference(config, base, registry):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage
    state, _, ref = _stored_stage(config, "selected_native_branch", registry)
    return {"source_task": config["source_task"], "source_stage": config["source_stage"],
            "identity": ref["identity"], "source_groups": state["task"]["source_groups"]}, 0


def execute_reference(config, inputs, output, registry):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage
    from driversi.runtime.budget import write_json
    state, original, ref = _stored_stage({k:config[k] for k in ("source_task","source_stage")},
                                     "selected_native_branch", registry)
    if ref["identity"] != config["identity"] or state["task"]["source_groups"] != config["source_groups"]:
        raise ValueError("Referenced physical branch changed")
    output.mkdir(parents=True)
    shutil.copy2(original["artifacts"]["trajectory_window"], output/"trajectory-window.json")
    write_json(output/"origin.json", {"source": ref, "physics_reexecuted": False})
    return result(output, read_mapping(original["artifacts"]["report"]), artifacts={
        "trajectory_window": str(output/"trajectory-window.json"), "origin":str(output/"origin.json")})
