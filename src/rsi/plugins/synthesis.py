"""Executable synthesis skills in the existing typed DriveHarness graph.

Search, independent admission, sensor purpose and native supervision are distinct
services. Logged-world scores cannot admit a reactive simulation or GS image.
"""

import copy
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.agents.backends.native_process import OwnedTree
from driversi.configuration import read_mapping
from driversi.data.datasets.tto_audit import COMPONENTS
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.tto import collect, prepare_candidates
from driversi.runtime.budget import write_json
from driversi.runtime.processes import subreaper
from driversi.training.sampling.environment import source_log_group


def check_files(files):
    for name, sha in files.items():
        if digest(name) != sha:
            raise ValueError("Sealed skill evidence changed: " + name)


def run_process(argv, output, seconds, *, environment=None):
    # The Harness parent is a subreaper, but prctl is not inherited by its
    # exec worker. Adopt native orphan helpers here so OwnedTree can reap them.
    previous_subreaper = subreaper(True)
    tree = OwnedTree()
    env = dict(os.environ)
    env.update(environment or {})
    env["PATH"] = str(Path(argv[0]).parent) + ":/usr/local/cuda/bin:" + env.get("PATH", "")
    try:
        with (output / "worker.log").open("wb") as stream:
            process = subprocess.Popen(
                argv,
                cwd=project_root(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            tree.attach(process)
            code = tree.wait(timeout=seconds)
        if code:
            raise RuntimeError("Skill worker failed; inspect " + str(output / "worker.log"))
    finally:
        try:
            tree.stop()
        finally:
            subreaper(previous_subreaper)


def protect(logs, registry, extra=()):
    from driversi.rsi.execution.registry import ExperimentRegistry

    protected = ExperimentRegistry(registry).protected_groups() | set(extra)
    official = (
        project_root()
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    protected |= set(read_mapping(official)["log_names"])
    if {source_log_group(x) for x in logs} & {source_log_group(x) for x in protected}:
        raise ValueError("Protected source cannot enter a synthesis skill")


def prepare_search(config, base, registry):
    _keys(
        config,
        (
            "backend",
            "python",
            "cases",
            "protected_logs",
            "initial_keys",
            "refinement_keys",
            "max_queries",
            "max_wall_seconds",
        ),
        "DriveTTO search skill",
    )
    config = copy.deepcopy(config)
    for key in ("backend", "cases", "protected_logs"):
        config[key] = str((base / config[key]).resolve(strict=True))
    config["python"] = str(path_at(base, config["python"]))
    if (
        type(config["max_wall_seconds"]) not in (int, float)
        or not math.isfinite(config["max_wall_seconds"])
        or not 0 < config["max_wall_seconds"] <= 3600
    ):
        raise ValueError("Search skill needs a bounded wall time")
    for key in ("initial_keys", "refinement_keys"):
        values = config[key]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 64
            or any(
                not isinstance(v, list)
                or len(v) != 2
                or any(type(x) is not int or abs(x) > 3000 for x in v)
                for v in values
            )
            or len({tuple(x) for x in values}) != len(values)
            or [0, 0] in values
        ):
            raise ValueError("Distinct bounded integer endpoint offsets required")
    cases_path = config["cases"]
    cases = read_mapping_list(cases_path)
    if not 1 <= len(cases) <= 16:
        raise ValueError("Pilot search supports 1..16 explicit anchors")
    cap = len(cases) * (3 + len(config["initial_keys"]) + len(config["refinement_keys"]))
    if type(config["max_queries"]) is not int or config["max_queries"] != cap:
        raise ValueError("Query cap must cover controls and both search phases exactly")
    files = {
        cases_path: digest(cases_path),
        config["protected_logs"]: digest(config["protected_logs"]),
    }
    logs, tokens, resolved = [], set(), []
    for case in cases:
        _keys(case, ("contract",), "search anchor")
        path = (Path(cases_path).parent / case["contract"]).resolve(strict=True)
        value = read_mapping(path)
        identity, protocol = value["identity"], value["protocol"]
        if (
            identity["split"] != "train"
            or len(identity["token"]) != 16
            or any(c not in "0123456789abcdef" for c in identity["token"])
            or identity["token"] in tokens
        ):
            raise ValueError("Unique explicit training anchors required")
        if (
            protocol["constructor"]["num_poses"],
            protocol["constructor"]["interval_s"],
            protocol["constructor"]["coordinate_frame"],
            protocol["sampling"]["integer_lattice_m"],
        ) != (8, 0.5, "current_ego_x_forward_y_left", 0.001):
            raise ValueError("Unregistered trajectory convention")
        labeling = protocol["labeling"]
        if (labeling["protocol"], labeling["evaluation_group"], labeling["score_unit"]) != (
            "navsim_v1_official_single_candidate",
            "reference_plus_one_candidate",
            "raw_0_to_1",
        ) or tuple(identity["component_names"]) != COMPONENTS:
            raise ValueError("Independent official singleton scorer required")
        gt = np.asarray(identity["expert_gt"])
        if gt.shape != (8, 3) or not np.isfinite(gt).all():
            raise ValueError("Malformed expert trajectory")
        files[str(path)] = digest(path)
        files[identity["official_cache_path"]] = digest(identity["official_cache_path"])
        tokens.add(identity["token"])
        logs.append(identity["log_name"])
        resolved.append({"contract": str(path)})
    protect(logs, registry, read_mapping_list(config["protected_logs"]))
    backend = Path(config["backend"])
    tracked = subprocess.check_output(
        ["git", "ls-files", "feedback/src", "navsim"], cwd=backend, text=True
    ).splitlines()
    for name in tracked:
        if name.endswith(".py"):
            files[str(backend / name)] = digest(backend / name)
    if not (backend / "feedback/src/drivetto_feedback/batch/source.py").is_file():
        raise ValueError("Original DriveTTO backend missing")
    config.update(
        cases=resolved,
        files=files,
        source_groups=sorted(set(map(source_log_group, logs))),
        skill="drivetto_endpoint_diagnosis_v1",
        scope="original_anchor_offline",
        backend_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=backend, text=True
        ).strip(),
    )
    return config, cap * 80


def read_mapping_list(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list):
        raise ValueError("Explicit list required")
    return value


def _reference(config, registry):
    from driversi.rsi.workflow.artifacts import verify
    from driversi.rsi.workflow.store import HarnessStore

    state = HarnessStore(registry).get(config["task"])
    declaration = next(x for x in state["task"]["steps"] if x["id"] == config["stage"])
    stage = next(x for x in state["stages"] if x["id"] == config["stage"])
    if declaration["output_kind"] != "search_evidence" or stage["status"] != "completed":
        raise ValueError("Only a completed sealed search may be reused")
    verify(stage["result"], state["id"], declaration)
    protect(state["task"]["source_groups"], registry)
    return stage["result"]


def prepare_reference(config, base, registry):
    _keys(config, ("task", "stage"), "completed search reference")
    ref = _reference(config, registry)
    return {**config, "identity": ref["summary"]["artifact_reference"]["identity"]}, 0


def run_reference(config, inputs, output, registry):
    import shutil

    ref = _reference(config, registry)
    if ref["summary"]["artifact_reference"]["identity"] != config["identity"]:
        raise ValueError("Referenced search identity changed")
    output.mkdir(parents=True)
    artifacts = {}
    for i, (name, path) in enumerate(ref["artifacts"].items()):
        dest = output / (str(i) + "-" + Path(path).name)
        shutil.copyfile(path, dest)
        artifacts[name] = str(dest)
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": 0,
        "artifacts": artifacts,
        "summary": {"reused": True, "origin_identity": config["identity"], "new_queries": 0},
        "admission": {"trainable": False},
    }


def _worker(config, output, mode):
    write_json(output / "worker-config.json", config)
    run_process(
        [
            config["python"],
            "-B",
            str(project_root() / "src/evaluation/search/tto_skill.py"),
            str(output / "worker-config.json"),
            str(output / "execution"),
            mode,
        ],
        output,
        config["max_wall_seconds"],
    )
    return read_mapping(output / "execution/report.json")


def run_search(config, inputs, output, registry):
    check_files(config["files"])
    protect(config["source_groups"], registry, read_mapping_list(config["protected_logs"]))
    output.mkdir(parents=True)
    evidence = _worker(config, output, "search")
    cases = [{"split": "train", "token": row["token"]} for row in evidence["cases"]]
    write_json(output / "cases.json", cases)
    imported, _ = prepare_candidates(
        dict(
            bank=str(output / "execution"),
            cases=str(output / "cases.json"),
            protected_logs=config["protected_logs"],
            max_scenes=len(cases),
        ),
        output,
        registry,
    )
    proposals = collect(imported)
    proposals.update(
        new_score_queries=evidence["cost"]["completed_queries"],
        score_scope="new_official_v1_original_anchor_not_reactive",
        diagnostics=evidence["cases"],
        execution_config=str(output / "worker-config.json"),
        execution_config_sha256=digest(output / "worker-config.json"),
    )
    artifacts = {
        "config": str(output / "worker-config.json"),
        "diagnostics": str(output / "execution/report.json"),
    }
    artifacts.update(
        {
            "evidence:" + str(p.relative_to(output)): str(p)
            for p in (output / "execution").rglob("*")
            if p.is_file()
        }
    )
    return result(
        output, proposals, steps=evidence["cost"]["attempted_queries"] * 80, artifacts=artifacts
    )


def prepare_admission(config, base, registry):
    _keys(config, ("max_scenes", "max_wall_seconds"), "independent search admission")
    if (
        type(config["max_scenes"]) is not int
        or not 1 <= config["max_scenes"] <= 16
        or not 0 < config["max_wall_seconds"] <= 3600
    ):
        raise ValueError("Bounded admission required")
    return config, config["max_scenes"] * 80


def run_admission(config, inputs, output, registry):
    proposals = read_mapping(inputs["search"]["artifacts"]["report"])
    execution = inputs["search"]["artifacts"]["config"]
    if digest(execution) != proposals["execution_config_sha256"]:
        raise ValueError("Search implementation changed")
    worker = read_mapping(execution)
    check_files(worker["files"])
    protect(proposals["source_groups"], registry, read_mapping_list(worker["protected_logs"]))
    output.mkdir(parents=True)
    if not proposals["accepted"]:
        return result(output, proposals, status="rejected", reason="no_qualified_new_candidate")
    if len(proposals["accepted"]) > config["max_scenes"]:
        raise ValueError("Admission budget exceeded")
    selected = []
    for row in proposals["accepted"]:
        for field in ("table", "contract", "publication"):
            worker["files"][row[field]] = row[field + "_sha256"]
        selected.append({"contract": row["contract"], "selected": row})
    worker.update(
        cases=selected, max_queries=len(selected), max_wall_seconds=config["max_wall_seconds"]
    )
    evidence = _worker(worker, output, "verify")
    accepted = {row["token"] for row in evidence["cases"] if row["accepted"]}
    if len(accepted) != len(selected):
        return result(
            output,
            evidence,
            status="rejected",
            reason="independent_native_rescore_failed",
            steps=evidence["cost"]["attempted_queries"] * 80,
        )
    report = {
        **proposals,
        "independent_rescore": evidence,
        "search_identity": inputs["search"]["summary"]["artifact_reference"]["identity"],
        "admitted_scope": "original_anchor_offline_trajectory_only",
        "trainable": False,
    }
    return result(
        output,
        report,
        steps=len(selected) * 80,
        artifacts={"verification": str(output / "execution/report.json")},
    )


def prepare_history(config, base, registry):
    _keys(config, ("manifest",), "original history purpose")
    path = (base / config["manifest"]).resolve(strict=True)
    return {"manifest": str(path), "sha256": digest(path)}, 0


def run_history(config, inputs, output, registry):
    from driversi_agents.augustus.data import cache_files, manifest

    if digest(config["manifest"]) != config["sha256"]:
        raise ValueError("History manifest changed")
    value = manifest(config["manifest"], allow_smoke=True)
    proposals = read_mapping(inputs["admission"]["artifacts"]["report"])
    if proposals.get("admitted_scope") != "original_anchor_offline_trajectory_only":
        raise ValueError("Original images cannot represent a counterfactual branch")
    records = {x["frame_id"]: x for x in value["splits"]["train"]["records"]}
    for row in proposals["accepted"]:
        rec = records.get(row["token"])
        if (
            not rec
            or rec["log_name"] != row["log_name"]
            or rec["timestamp_us"] != row["timestamp_us"]
            or not np.allclose(rec["ego_pose"], row["ego_pose"], atol=1e-7, rtol=0)
        ):
            raise ValueError("Selected branch and historical observation anchor differ")
    protect(
        [r["log_name"] for b in value["splits"].values() for r in b["records"]],
        registry,
        value["protected_logs"],
    )
    output.mkdir(parents=True)
    write_json(output / "proposals.json", proposals)
    return result(
        output,
        {
            "kind": "synthesis_sensor_receipt",
            "mode": "original_history_reuse",
            "render_required": False,
            "new_images": 0,
            "future_rgb_required": False,
            "source_manifest": config["manifest"],
            "source_manifest_sha256": config["sha256"],
            "cache_files": {p: digest(p) for p in cache_files(value)},
            "admission_identity": inputs["admission"]["summary"]["artifact_reference"]["identity"],
            "search_identity": proposals["search_identity"],
            "visual_counterfactual_admitted": False,
        },
        artifacts={"proposals": str(output / "proposals.json")},
    )


def prepare_dataset(config, base, registry):
    _keys(config, ("python", "max_wall_seconds", "label_source"), "admitted native dataset")
    if (
        config["label_source"] not in ("original", "drivetto")
        or not 0 < config["max_wall_seconds"] <= 1800
    ):
        raise ValueError("Invalid dataset skill bounds")
    return {**config, "python": str(path_at(base, config["python"]))}, 0


def run_dataset(config, inputs, output, registry):
    receipt = read_mapping(inputs["sensors"]["artifacts"]["report"])
    if receipt["mode"] != "original_history_reuse":
        raise ValueError("This native adapter only supports logged-world relabeling")
    check_files(
        {receipt["source_manifest"]: receipt["source_manifest_sha256"], **receipt["cache_files"]}
    )
    output.mkdir(parents=True)
    run_process(
        [
            config["python"],
            "-B",
            "-m",
            "driversi_agents.augustus.data",
            "--source",
            receipt["source_manifest"],
            "--proposals",
            inputs["sensors"]["artifacts"]["proposals"],
            "--output",
            str(output / "dataset"),
            "--label-source",
            config["label_source"],
            "--allow-smoke",
        ],
        output,
        config["max_wall_seconds"],
    )
    report = read_mapping(output / "dataset/report.json")
    report.update(
        sensor_identity=inputs["sensors"]["summary"]["artifact_reference"]["identity"],
        search_identity=receipt["search_identity"],
        evaluation_scope="current_run_log_heldout_parent_provenance_unknown",
    )
    return result(output, report, artifacts={"manifest": str(output / "dataset/dataset.json")})


def register(registry, Provider):
    for p in (
        Provider(
            "completed_search_reference",
            "1",
            (),
            "search_evidence",
            prepare_reference,
            run_reference,
            ("report", "config", "diagnostics"),
        ),
        Provider(
            "drivetto_search_skill",
            "1",
            (),
            "search_evidence",
            prepare_search,
            run_search,
            ("report", "config", "diagnostics"),
        ),
        Provider(
            "drivetto_independent_admission",
            "1",
            (("search", "search_evidence"),),
            "admitted_original_branch",
            prepare_admission,
            run_admission,
            ("report", "verification"),
        ),
        Provider(
            "synthesis_history",
            "1",
            (("admission", "admitted_original_branch"),),
            "synthesis_sensors",
            prepare_history,
            run_history,
            ("report", "proposals"),
        ),
        Provider(
            "synthesis_native_dataset",
            "1",
            (("sensors", "synthesis_sensors"),),
            "native_training_dataset",
            prepare_dataset,
            run_dataset,
            ("report", "manifest"),
        ),
    ):
        registry.register(p)
