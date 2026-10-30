"""Bounded learning lifecycle used by the unified engine and DriveHarness.

Model workers own algorithms. This module owns process and artifact boundaries.
"""

import math
import os
import subprocess
import time
from pathlib import Path

from driversi.agents.backends.native import digest, path_at, project_root, signature
from driversi.agents.backends.native_process import OwnedTree
from driversi.agents.backends.sources import require_source
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi.training.sampling.environment import source_log_group

WORKERS = {
    "chainflow_stage1": "driversi_agents.chainflow.training",
    "driverl": "driversi_agents.driverl.training",
    "augustus": "driversi_agents.augustus.training",
}


def sources(data):
    groups = {
        k: sorted({source_log_group(x) for x in data[k]["source_logs"]})
        for k in ("train", "development")
    }
    if not all(groups.values()) or set(groups["train"]) & set(groups["development"]):
        raise ValueError("Training and development must use separate original logs")
    official = (
        project_root()
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    protected = set(map(source_log_group, read_mapping(official)["log_names"]))
    protocol = data.get("evaluation_protocol")
    benchmark = protocol in ("official_navtest_v1", "official_navtest_v1_locked_pilot")
    if benchmark:
        from driversi_agents.chainflow.benchmark import official_filter

        _, test = official_filter("navtest")
        expected = set(map(source_log_group, test["log_names"]))
        if protocol == "official_navtest_v1_locked_pilot":
            from driversi_agents.chainflow.benchmark import read_pilot_subset

            pilot = read_pilot_subset(data["evaluation_subset"])
            expected = {source_log_group(r["source_log"]) for r in pilot["records"]}
        if set(groups["development"]) != expected:
            raise ValueError("Official campaign requires the complete navtest source population")
    if protected & set(groups["train"] if benchmark else groups["train"] + groups["development"]):
        raise ValueError("Official test logs cannot enter iterative learning pilots")
    if data.get("parent_provenance") != "unknown":
        raise ValueError("Pilot does not certify released-parent training provenance")
    return groups


def prepare(resolved, output, *, phase="train", previous=None, evaluator=None):
    root = project_root()
    output = Path(output).resolve()
    if not output.is_relative_to(root / "exp") or output.exists():
        raise ValueError("A new learning attempt must live under exp")
    if phase not in ("train", "resume", "evaluate"):
        raise ValueError("Unsupported learning phase")
    cfg = read_mapping(resolved["model_config"])
    request = resolved["request"]
    data_path = path_at(Path(resolved["request_path"]).parent, request["data"])
    data = read_mapping(data_path)
    groups = sources(data)
    benchmark = data.get("evaluation_protocol") in (
        "official_navtest_v1",
        "official_navtest_v1_locked_pilot",
    )
    if benchmark != (request["schema_version"] in (3, 4)):
        raise ValueError("Official test access requires an explicit benchmark request")
    repository = path_at(root, cfg["repository"])
    if (
        repository
        != root
        / "agents"
        / ({"chainflow_stage1": "chainflow"}.get(cfg["model"], cfg["model"]))
        / "native"
    ):
        raise ValueError("Native models must be owned under agents")
    require_source(repository, cfg["revision"])
    python = path_at(root, cfg["python"])
    previous_path = Path(previous).resolve(strict=True) if previous else None
    if (phase != "train") != bool(previous_path):
        raise ValueError("Resume/evaluation require the previous training receipt")
    previous_report = read_mapping(previous_path) if previous_path else None
    implementation = {}
    if cfg["model"] == "chainflow_stage1":
        for name in (
            "agents/chainflow/data.py",
            "agents/chainflow/benchmark.py",
            "agents/chainflow/training.py",
            "agents/chainflow/distributed.py",
            "agents/chainflow/sampling.py",
            "agents/chainflow/stage1.py",
            "agents/chainflow/synthetic.py",
            "src/rsi/correction/trajectory_admission.py",
            "src/data/datasets/navsim_synthetic.py",
            "src/rendering/diagnostics/worldengine_admission.py",
            "agents/chainflow/native/source-manifest.json",
            "agents/chainflow/native/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
            "agents/chainflow/native/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
            "src/training/orchestration/worker.py",
            "src/training/orchestration/state.py",
        ):
            implementation[str(root / name)] = digest(root / name)
    consumed_inputs = {}
    if cfg["model"] == "chainflow_stage1":
        from driversi_agents.chainflow.data import validate_manifest

        manifest = validate_manifest(data["manifest"], groups=groups)
        if benchmark != (manifest["schema_version"] in (2, 3)):
            raise ValueError("Benchmark request and data protocol differ")
        if benchmark and data["evaluation_protocol"] != manifest["benchmark"]["protocol"]:
            raise ValueError("Benchmark request and named data protocol differ")
        if (
            manifest["schema_version"] == 3
            and data.get("evaluation_subset") != manifest["benchmark"]["test_subset"]
        ):
            raise ValueError("Pilot request differs from sealed evaluation subset")
        if request["pilot"]["evaluation_steps"] != len(manifest["splits"]["development"]):
            raise ValueError("Evaluation budget differs from exact scene coverage")
        consumed_inputs[data["manifest"]] = digest(data["manifest"])
        consumed_inputs.update(manifest["files"])
        for weight in cfg["weights"].values():
            asset = path_at(root, weight)
            consumed_inputs[str(asset)] = digest(asset)
    if cfg["model"] == "augustus":
        from driversi.agents.backends.native import asset_files
        from driversi_agents.augustus.data import cache_files
        from driversi_agents.augustus.data import manifest as read_augustus
        from driversi_agents.augustus.metric_cache import scoring_files

        manifest = read_augustus(data["manifest"], allow_smoke=True)
        named_protocol = manifest.get("benchmark", {}).get("protocol")
        if benchmark != bool(named_protocol) or (
            benchmark and named_protocol != data.get("evaluation_protocol")
        ):
            raise ValueError("AUGUstus request and supervision evaluation protocol differ")
        if benchmark:
            from driversi_agents.augustus.benchmark import sealed_files

            if data.get("evaluation_subset") != manifest["benchmark"]["test_subset"]:
                raise ValueError("AUGUstus request differs from locked evaluation subset")
            consumed_inputs.update(sealed_files(manifest))
            if data.get("full_evaluation_plan"):
                full_path = Path(data["full_evaluation_plan"])
                full = read_mapping(full_path)
                consumed_inputs[str(full_path)] = digest(full_path)
                report_path = Path(full["output"]) / "report.json"
                consumed_inputs[str(report_path)] = digest(report_path)
                consumed_inputs.update(full["inputs"])
                for name in (
                    "agents/augustus/full_pair_evaluation.py",
                    "agents/augustus/candidate_dispatch.py",
                    "agents/augustus/full_evaluation.py",
                    "agents/augustus/evaluation.py",
                ):
                    consumed_inputs[str(root / name)] = digest(root / name)
        for side in ("train", "development"):
            actual = {source_log_group(r["log_name"]) for r in manifest["splits"][side]["records"]}
            if actual != set(groups[side]):
                raise ValueError("AUGUstus source groups differ from actual native cache")
        if benchmark and request["pilot"]["evaluation_steps"] != len(
            manifest["splits"]["development"]["records"]
        ):
            raise ValueError("AUGUstus evaluation budget differs from exact sample coverage")
        consumed_inputs[data["manifest"]] = digest(data["manifest"])
        for name in cache_files(manifest):
            consumed_inputs[name] = digest(name)
        for name in scoring_files(
            data["metric_cache"], manifest["splits"]["development"]["records"]
        ):
            consumed_inputs[name] = digest(name)
        for weight in cfg["weights"].values():
            for name in asset_files(path_at(root, weight)):
                consumed_inputs[name] = digest(name)
        for name in (
            "agents/augustus/data.py",
            "agents/augustus/benchmark.py",
            "agents/augustus/branch_supervision.py",
            "agents/augustus/projection.py",
            "agents/augustus/training.py",
            "agents/augustus/runner.py",
            "agents/augustus/normalization.py",
            "agents/augustus/budget.py",
            "src/training/orchestration/state.py",
        ):
            implementation[str(root / name)] = digest(root / name)
    # The recipe identity excludes run/output names, but binds the actual data,
    # model declaration, update cap and all engineering controls.
    context = signature(
        {
            "model": cfg,
            **(
                {"implementation": implementation, "consumed_inputs": consumed_inputs}
                if implementation
                else {}
            ),
            "data": data,
            "data_sha256": digest(data_path),
            "pilot": request["pilot"],
            "budget": request["budget"],
        }
    )
    if previous_report and (
        (
            previous_report["status"] != "completed"
            and not (
                request["schema_version"] == 4
                and phase == "resume"
                and previous_report.get("recoverable_checkpoint") is True
                and previous_report["status"] in ("failed", "budget_exhausted")
            )
        )
        or previous_report["context"] != context
    ):
        raise ValueError("Continuation context differs from the producing run")
    files = {
        resolved["request_path"]: digest(resolved["request_path"]),
        resolved["model_config"]: digest(resolved["model_config"]),
        str(data_path): digest(data_path),
    }
    files.update(implementation)
    files.update(consumed_inputs)
    if data["train"].get("curriculum"):
        if cfg["model"] != "driverl":
            raise ValueError("Online recovery curriculum requires its native RL adapter")
        from driversi.rsi.correction.curriculum import validate_curriculum

        manifest_path = data["train"]["curriculum"]
        manifest, _ = validate_curriculum(
            manifest_path,
            scene=data["train"]["scene"],
            source_log=data["train"]["source_logs"][0],
            checkpoint=path_at(root, cfg["weights"]["checkpoint"]),
            arm=data["train"].get("curriculum_arm"),
        )
        files[manifest_path] = digest(manifest_path)
        files.update(manifest["files"])
    if previous_path:
        files[str(previous_path)] = digest(previous_path)
        files.update(previous_report["checkpoint_files"])
    if cfg["model"] in WORKERS and resolved["backend"] is None:
        raise ValueError("Model does not support the requested learning mode")
    ready = cfg["model"] in WORKERS and python.is_file()
    evaluation_context = None
    if evaluator:
        if phase != "evaluate":
            raise ValueError("Independent evaluator context is only valid for evaluation")
        from driversi.training.contracts.evaluation import resolve_context

        evaluation_context = resolve_context(evaluator, cfg["model"], groups["development"])
        files[evaluation_context["manifest"]] = evaluation_context["manifest_sha256"]
        files.update(evaluation_context["value"]["files"])
    output.mkdir(parents=True)
    plan = {
        "schema_version": 2,
        "resolved": resolved,
        "output": str(output),
        "phase": phase,
        "evaluation_context": evaluation_context,
        "context": context,
        "model_config": cfg,
        "data": data,
        "source_groups": groups,
        "files": files,
        "previous": str(previous_path) if previous_path else None,
        "python": str(python),
        "worker": WORKERS.get(cfg["model"]),
        "ready_to_attempt": ready,
        "missing": (
            []
            if ready
            else [
                "native_learning_backend_not_released"
                if cfg["model"] not in WORKERS
                else "python_environment_missing"
            ]
        ),
    }
    write_json(output / "training-plan.json", plan)
    return plan


def execute(plan, *, own_process_group=True):
    output = Path(plan["output"])
    report_path = output / "training-result.json"
    if report_path.exists():
        raise ValueError("Attempt already settled; resume into a new attempt")
    for path, expected in plan["files"].items():
        if digest(path) != expected:
            raise ValueError("Learning input changed: " + path)
    cfg = plan["model_config"]
    require_source(path_at(project_root(), cfg["repository"]), cfg["revision"])
    base = {
        "schema_version": 2,
        "model": cfg["model"],
        "phase": plan["phase"],
        "evaluation_context": plan.get("evaluation_context"),
        "context": plan["context"],
        "source_groups": plan["source_groups"],
        "promoted": False,
        "policy_activated": False,
        "evaluation_scope": (
            "official_navtest_locked_subset_adaptive_pilot_not_full_benchmark"
            if plan["data"].get("evaluation_protocol") == "official_navtest_v1_locked_pilot"
            else "official_navtest_adaptive_evaluation"
            if plan["data"].get("evaluation_protocol") == "official_navtest_v1"
            else "current_run_log_heldout_parent_provenance_unknown"
        ),
    }
    if not plan["ready_to_attempt"]:
        result = {
            **base,
            "status": "rejected",
            "reason": ";".join(plan["missing"]),
            "consumed": {"wall_seconds": 0.0, "environment_steps": 0, "scoring_agent_steps": 0},
        }
        write_json(report_path, result)
        return result
    # A saved plan is not an idempotency key for repeating optimizer updates.
    # Keep an execution claim even if a worker dies before publishing a receipt.
    with (output / "execution-claim.json").open("x") as claim:
        claim.write(str(os.getpid()))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")}
    env.update(
        CUDA_VISIBLE_DEVICES=str(cfg["settings"]["gpu"]),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled",
        SWANLAB_MODE="disabled",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED=str(plan["resolved"]["request"]["pilot"]["seed"]),
    )
    distributed = plan["resolved"]["request"]["schema_version"] == 4 and plan["phase"] != "evaluate"
    if plan["resolved"]["request"]["schema_version"] == 4:
        from driversi_agents.chainflow.sampling import distributed_settings

        devices = distributed_settings(cfg["settings"])["devices"]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices if distributed else devices[:1]))
    command = [plan["python"]]
    if distributed:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=" + str(len(devices)),
        ]
    command += (
        [str(project_root() / "src/training/orchestration/worker.py")]
        if cfg["model"] == "chainflow_stage1"
        else ["-m", "driversi.training.orchestration.worker"]
    )
    command += [str(output / "training-plan.json")]
    start = time.monotonic()
    tree = OwnedTree()
    process = None
    result = {**base, "status": "failed", "reason": "worker_did_not_publish"}
    try:
        with (output / "native.log").open("wb") as stream:
            process = subprocess.Popen(
                command,
                cwd=project_root(),
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=own_process_group,
            )
            tree.attach(process)
            code = tree.wait(timeout=plan["resolved"]["request"]["budget"]["max_wall_seconds"])
        if code == 0 and (output / "worker-result.json").is_file():
            result = {**base, **read_mapping(output / "worker-result.json")}
            if plan["phase"] != "evaluate":
                if (
                    not 0
                    < result["updates"]
                    <= plan["resolved"]["request"]["budget"]["max_optimizer_steps"]
                ):
                    raise ValueError("No confirmed bounded optimizer updates")
                paths = result["checkpoint_files"]
                if not paths or any(not Path(p).resolve().is_relative_to(output) for p in paths):
                    raise ValueError("Checkpoint must belong to this exp attempt")
                if any(digest(p) != sha for p, sha in paths.items()):
                    raise ValueError("Checkpoint publication mismatch")
        else:
            result.update(reason="native_process_failed", returncode=code)
    except subprocess.TimeoutExpired:
        result.update(status="budget_exhausted", reason="native_wall_budget_exhausted")
    except Exception as exc:
        result.update(status="failed", reason=repr(exc))
    finally:
        tree.stop()
    if result["status"] != "completed":
        budget = plan["resolved"]["request"]["budget"]
        result["consumed"] = {
            "environment_steps": (
                2 * plan["resolved"]["request"]["pilot"]["evaluation_steps"]
                if plan["phase"] == "evaluate" and cfg["model"] == "driverl"
                else budget["max_environment_steps"]
            ),
            "scoring_agent_steps": budget["max_scoring_agent_steps"],
            "cost_basis": "reserved_cap_worker_outcome_unconfirmed",
        }
        if cfg["model"] == "driverl":
            result["consumed"]["sim_agent_steps"] = (
                2 * plan["resolved"]["request"]["pilot"]["evaluation_steps"]
                if plan["phase"] == "evaluate"
                else 20 + 2 * budget["max_environment_steps"]
            )
            if plan["data"]["train"].get("curriculum"):
                from driversi.contracts.scene import Scene

                side = "development" if plan["phase"] == "evaluate" else "train"
                result["consumed"]["sim_agent_steps"] *= len(
                    Scene.load(Path(plan["data"][side]["scene"])).actor_ids
                )
                result["consumed"]["cost_basis"] = (
                    "reserved_cap_all_world_slots_worker_outcome_unconfirmed"
                )
    result["consumed"] = {**result.get("consumed", {}), "wall_seconds": time.monotonic() - start}
    result["wall_budget_scope"] = (
        "worker_execution_and_output_verification_excluding_input_preflight"
    )
    if (
        result["status"] == "completed"
        and result["consumed"]["wall_seconds"]
        > plan["resolved"]["request"]["budget"]["max_wall_seconds"]
    ):
        result.update(status="budget_exhausted", reason="output_verification_exceeded_wall_budget")
    if distributed and result["status"] in ("failed", "budget_exhausted"):
        latest = output / "latest-checkpoint.json"
        if latest.is_file():
            try:
                checkpoint = read_mapping(latest)
                verify_interval_checkpoint(checkpoint, plan)
                result.update(checkpoint)
                result["recoverable_checkpoint"] = True
                result["recovery_note"] = (
                    "Incomplete tail discarded; reserved failed-attempt cost retained"
                )
            except (ValueError, KeyError, OSError) as exc:
                result["checkpoint_recovery_error"] = str(exc)
    if plan["phase"] != "evaluate" and (
        result["status"] == "completed" or result.get("recoverable_checkpoint") is True
    ):
        result["training_lineage_cost"] = accumulated_training_cost(plan, result)
    result["worker_pid"] = process.pid if process else None
    result["runtime_reproducibility"] = {
        "python_hash_seed": env["PYTHONHASHSEED"],
        "cublas_workspace_config": env["CUBLAS_WORKSPACE_CONFIG"],
        "scope": "fixed_before_worker_start_not_retroactive_to_old_results",
    }
    write_json(report_path, result)
    return result


def accumulated_training_cost(plan, result):
    # Reconstruct ancestors, including old recoverable failures without a cached sum.
    earlier = training_lineage(plan["previous"])["cost"] if plan["previous"] else {}
    return {
        k: earlier.get(k, 0) + result["consumed"].get(k, 0)
        for k in ("wall_seconds", "environment_steps", "scoring_agent_steps", "sim_agent_steps")
    }


def verify_interval_checkpoint(checkpoint, plan):
    """Accept only fully published files from this attempt, after its tree stopped."""
    output = Path(plan["output"]).resolve()
    if checkpoint["context"] != plan["context"] or checkpoint.get("recovery_schema_version") != 3:
        raise ValueError("Interval checkpoint context or format differs")
    if (
        type(checkpoint.get("updates")) is not int
        or not 0
        < checkpoint["updates"]
        <= plan["resolved"]["request"]["budget"]["max_optimizer_steps"]
    ):
        raise ValueError("Interval update count outside attempt budget")
    paths = checkpoint["checkpoint_files"]
    if set(paths) != {checkpoint["checkpoint"], checkpoint["recovery_checkpoint"]}:
        raise ValueError("Incomplete interval checkpoint publication")
    if any(
        not Path(p).resolve().is_relative_to(output / "checkpoints") or digest(p) != sha
        for p, sha in paths.items()
    ):
        raise ValueError("Interval checkpoint publication mismatch")


def training_lineage(report_path):
    """Reconstruct costs from immutable ancestor receipts, including old schemas."""
    costs = dict(wall_seconds=0.0, environment_steps=0, scoring_agent_steps=0, sim_agent_steps=0)
    receipts, seen, context = [], set(), None
    path = Path(report_path).resolve()
    while path:
        if path in seen:
            raise ValueError("Cyclic training lineage")
        seen.add(path)
        report = read_mapping(path)
        plan = read_mapping(path.parent / "training-plan.json")
        context = report["context"] if context is None else context
        if (
            (
                report["status"] != "completed"
                and not (
                    report.get("recoverable_checkpoint") is True
                    and report["status"] in ("failed", "budget_exhausted")
                )
            )
            or report["context"] != context
            or plan["phase"] == "evaluate"
        ):
            raise ValueError("Invalid training ancestor")
        for key in costs:
            value = report["consumed"].get(key, 0)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid ancestor cost")
            costs[key] += value
        receipts.append({"path": str(path), "sha256": digest(path)})
        path = Path(plan["previous"]).resolve() if plan["previous"] else None
    return {
        "cost": costs,
        "receipts": receipts,
        "basis": "sum_of_training_ancestors_including_recoverable_failure_reserved_cost_excludes_replay_and_evaluation",
    }


def feedback(student, evaluation):
    if (
        student["status"] != "completed"
        or evaluation["status"] != "completed"
        or student["context"] != evaluation["context"]
    ):
        raise ValueError("Only matching completed training/evaluation can produce feedback")
    if evaluation["evaluated_checkpoint"] != student["checkpoint"]:
        raise ValueError("Evaluation belongs to another student")
    metrics = evaluation["metrics"]
    deltas = {}
    for key, direction in evaluation["directions"].items():
        a, b = metrics["parent"][key], metrics["candidate"][key]
        if not all(type(x) in (float, int) and math.isfinite(x) for x in (a, b)):
            raise ValueError("Nonfinite paired evaluation")
        deltas[key] = {
            "parent": a,
            "candidate": b,
            "delta": b - a,
            "improved": (b - a) * direction > 0,
        }
    safety = all(
        metrics["candidate"][k] >= metrics["parent"][k] for k in evaluation.get("safety_higher", [])
    )
    primary = evaluation["primary_metric"]
    return {
        "status": "completed",
        "model": student["model"],
        "context": student["context"],
        "deltas": deltas,
        "pilot_metric_improved": deltas[primary]["improved"] and safety,
        "observed_generalization_benefit": False,
        "promoted": False,
        "reason": "small_run_holdout_unknown_parent_provenance_no_safety_promotion",
        "checkpoint": student["checkpoint"],
        "resume_from": student.get("resume_from"),
        "training_cost": student.get("training_lineage_cost", student["consumed"]),
        "evaluation_cost": evaluation["consumed"],
        "next_action": "expand_independent_development_if_pilot_improves_else_diagnose",
    }
