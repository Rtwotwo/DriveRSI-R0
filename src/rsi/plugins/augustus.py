"""Native AUGUstus stages in the existing typed, bounded Harness graph."""

import json
import subprocess
from pathlib import Path

import yaml

from driversi.agents.backends.native import asset_files, digest, execute, path_at, prepare, project_root
from driversi.agents.backends.native_process import OwnedTree
from driversi.runtime.budget import write_json
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.training.sampling.environment import source_log_group
from driversi_agents.augustus.data import manifest




def protect_dataset(path, registry):
    from driversi.rsi.execution.registry import ExperimentRegistry

    value = manifest(path)
    protected = set(value["protected_logs"]) | ExperimentRegistry(registry).protected_groups()
    official = (
        project_root()
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    protected |= set(yaml.safe_load(official.read_text())["log_names"])
    groups = {source_log_group(r["log_name"]) for b in value["splits"].values() for r in b["records"]}
    if groups & {source_log_group(x) for x in protected}:
        raise ValueError("Protected source in native dataset at execution")
    return value


def bounds(config):
    import math

    v = config["max_wall_seconds"]
    if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 86400:
        raise ValueError("Explicit bounded wall time required")


def prepare_dataset(config, base, registry):
    config = {"label_source": "drivetto", **config}
    _keys(config, ("manifest", "python", "max_wall_seconds", "label_source"), "native dataset")
    if config["label_source"] not in ("original", "drivetto"):
        raise ValueError("Unknown native label source")
    bounds(config)
    path = (base / config["manifest"]).resolve()
    # Keep the venv executable path: resolving its symlink switches sys.prefix
    # to the base interpreter and silently loses isolated dependencies.
    python = path_at(base, config["python"])
    value = manifest(path)
    from driversi.rsi.execution.registry import ExperimentRegistry

    protected = set(value["protected_logs"]) | ExperimentRegistry(registry).protected_groups()
    official = (
        project_root()
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    protected |= set(yaml.safe_load(official.read_text())["log_names"])
    groups = {source_log_group(r["log_name"]) for block in value["splits"].values() for r in block["records"]}
    if groups & {source_log_group(x) for x in protected}:
        raise ValueError("Protected source in native dataset")
    return {
        **config,
        "manifest": str(path),
        "python": str(python),
        "sha256": digest(path),
        "source_groups": sorted(groups),
    }, 0


def run_dataset(config, inputs, output, registry):
    output.mkdir(parents=True, exist_ok=False)
    if digest(config["manifest"]) != config["sha256"]:
        raise ValueError("Source dataset declaration changed")
    value = protect_dataset(config["manifest"], registry)
    missing = [b["cache"] for b in value["splits"].values() if not Path(b["cache"]).is_dir()]
    if not Path(config["python"]).is_file():
        missing.append(config["python"])
    if missing:
        return result(
            output, {"missing": missing}, status="failed", reason="native_cache_or_runtime_missing"
        )
    proposal = inputs["proposals"]["artifacts"]["report"]
    tree = OwnedTree()
    with (output / "data.log").open("wb") as log:
        process = subprocess.Popen(
            [
                config["python"],
                "-B",
                "-m",
                "driversi_agents.augustus.data",
                "--source",
                config["manifest"],
                "--proposals",
                proposal,
                "--output",
                str(output / "dataset"),
                "--label-source",
                config.get("label_source", "drivetto"),
            ],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=False,
        )
        tree.attach(process)
        try:
            code = tree.wait(timeout=config["max_wall_seconds"])
        except subprocess.TimeoutExpired:
            return result(
                output,
                {"reason": "dataset_timeout"},
                status="budget_exhausted",
                reason="dataset_timeout",
            )
        finally:
            tree.stop()
    if code:
        return result(
            output, {"returncode": code}, status="failed", reason="native_dataset_validation_failed"
        )
    report = json.loads((output / "dataset/report.json").read_text())
    report["proposal_identity"] = inputs["proposals"]["summary"]["artifact_reference"]["identity"]
    return result(output, report, artifacts={"manifest": str(output / "dataset/dataset.json")})


def prepare_job(config, base, registry):
    _keys(config, ("model_config", "max_wall_seconds", "max_sim_agent_steps"), "native RSI job")
    bounds(config)
    cap = config["max_sim_agent_steps"]
    if type(cap) is not int or cap < 0:
        raise ValueError("Native scoring budget required")
    path = (base / config["model_config"]).resolve(strict=True)
    cfg = yaml.safe_load(path.read_text())
    if cfg["model"] != "augustus":
        raise ValueError("Native RSI job requires a registered AUGUstus adapter")
    return {**config, "model_config": str(path), "config_sha256": digest(path)}, cap


def resolved(config, dataset, output, operation, parent=None):
    # A path-only manifest is not the immutable dataset published by the data stage.
    evidence = json.loads(Path(dataset["artifacts"]["report"]).read_text())
    from driversi_agents.augustus.data import cache_files

    value = manifest(dataset["artifacts"]["manifest"])
    actual = {p: digest(p) for p in cache_files(value)}
    if not actual or actual != evidence.get("cache_files"):
        raise ValueError("Native dataset content changed since publication")
    if digest(config["model_config"]) != config["config_sha256"]:
        raise ValueError("Native model declaration changed")
    cfg = yaml.safe_load(Path(config["model_config"]).read_text())
    cfg["paths"]["dataset_manifest"] = dataset["artifacts"]["manifest"]
    cfg["settings"]["max_sim_agent_steps"] = config["max_sim_agent_steps"]
    if "max_optimizer_steps" in config:
        cfg["settings"]["max_optimizer_steps"] = config["max_optimizer_steps"]
    if parent:
        cfg["weights"]["parent"] = parent
    if operation == "evaluate":
        cfg["settings"]["gpus"] = [cfg["settings"]["gpu"]]
    path = output / "model.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return prepare(path, operation, output / "native")


def consumed(native):
    path = native / "scoring-budget.json"
    return json.loads(path.read_text())["reserved_agent_steps"] if path.exists() else 0


def bundle(path):
    return {p: digest(p) for p in asset_files(path) if Path(p).is_file()}


def verify_bundle(card):
    if not card["files"] or bundle(card["checkpoint"]) != card["files"]:
        raise ValueError("Model bundle changed")


def run_training(config, inputs, output, registry):
    output.mkdir(parents=True, exist_ok=False)
    protect_dataset(inputs["dataset"]["artifacts"]["manifest"], registry)
    plan = resolved(config, inputs["dataset"], output, "train")
    if not plan["ready_to_attempt"]:
        return result(
            output,
            {"missing": plan["missing"]},
            status="failed",
            reason="native_prerequisites_missing",
        )
    outcome = execute(plan, timeout_seconds=config["max_wall_seconds"], own_process_group=False)
    steps = consumed(output / "native")
    if outcome["status"] != "completed":
        status = "budget_exhausted" if outcome["status"] == "timeout" else "failed"
        return result(
            output,
            outcome,
            status=status,
            reason=outcome.get("reason", "native_training_failed"),
            steps=steps,
        )
    info = json.loads((output / "native/candidate.json").read_text())
    cfg = yaml.safe_load(Path(config["model_config"]).read_text())
    parent = str((project_root() / cfg["weights"]["parent"]).resolve())
    if info["parent"] != parent:
        raise ValueError("Wrong parent model")
    data = manifest(inputs["dataset"]["artifacts"]["manifest"])
    card = {
        "checkpoint": info["checkpoint"],
        "files": bundle(info["checkpoint"]),
        "parent": {"checkpoint": parent, "files": bundle(parent)},
        "dataset_identity": inputs["dataset"]["summary"]["artifact_reference"]["identity"],
        "training_sources": sorted(
            set(data["parent_training_logs"])
            | {r["log_name"] for r in data["splits"]["train"]["records"]}
        ),
        "model_config_sha256": config["config_sha256"],
        "native_plan_sha256": plan["plan_sha256"],
        "wall_seconds": outcome["wall_seconds"],
        "reserved_gpu_seconds": outcome["reserved_gpu_seconds"],
        "scoring_agent_steps": steps,
        "updates": info["updates"],
        "performance_parity": "not_measured",
        "promoted": False,
    }
    verify_bundle(card)
    write_json(output / "candidate.json", card)
    return result(
        output, card, steps=steps, artifacts={"candidate": str(output / "candidate.json"),
            "plan": str(output / "native/plan.json")}
    )


def compare(parent, candidate, records):
    tokens = {r["frame_id"] for r in records}

    def parse(report):
        rows = report["samples"]
        if len(rows) != len(tokens) or {r["frame_id"] for r in rows} != tokens:
            raise ValueError("Independent evaluation coverage mismatch")
        indexed = {}
        for row in rows:
            if row["frame_id"] in indexed:
                raise ValueError("Duplicate evaluation token")
            for key in ("PDMS", "NC", "DAC", "TTC", "EP", "C", "DDC"):
                import math

                if (
                    type(row.get(key)) not in (float, int)
                    or not math.isfinite(row[key])
                    or not 0 <= row[key] <= 1
                ):
                    raise ValueError("Missing/invalid native score: " + key)
            indexed[row["frame_id"]] = row
        return indexed

    a, b = parse(parent), parse(candidate)
    pairs = [
        {
            "token": t,
            "pdms_delta": b[t]["PDMS"] - a[t]["PDMS"],
            "collision_regression": b[t]["NC"] < a[t]["NC"],
            "road_regression": b[t]["DAC"] < a[t]["DAC"],
        }
        for t in sorted(tokens)
    ]
    safety = not any(p["collision_regression"] or p["road_regression"] for p in pairs)
    gain = sum(p["pdms_delta"] for p in pairs) / len(pairs)
    return {
        "pairs": pairs,
        "mean_pdms_delta": gain,
        "safety_nonregression": safety,
        "observed_benefit": safety and gain > 1e-8,
        "promoted": False,
        "interpretation": "paired_development_evidence_not_causal_or_public_benchmark",
    }


def run_evaluation(config, inputs, output, registry):
    output.mkdir(parents=True, exist_ok=False)
    dataset, student = inputs["dataset"], inputs["student"]
    card = json.loads(Path(student["artifacts"]["candidate"]).read_text())
    if (
        card["dataset_identity"] != dataset["summary"]["artifact_reference"]["identity"]
        or card["model_config_sha256"] != config["config_sha256"]
    ):
        raise ValueError("Student/data/model lineage mismatch")
    verify_bundle(card)
    verify_bundle(card["parent"])
    data = protect_dataset(dataset["artifacts"]["manifest"], registry)
    records = data["splits"]["development"]["records"]
    if {source_log_group(r["log_name"]) for r in records} & {source_log_group(x) for x in card["training_sources"]}:
        raise ValueError("Independent development overlaps parent/candidate training")
    if config["max_sim_agent_steps"] < len(records) * 160:
        raise ValueError("Both full scoring arms must fit budget")
    reports = {}
    total = 0
    wall = 0.0
    reserved_gpu = 0.0
    protocols = []
    for arm, checkpoint in (
        ("parent", card["parent"]["checkpoint"]),
        ("candidate", card["checkpoint"]),
    ):
        folder = output / arm
        folder.mkdir()
        # Each arm may use at most its exact one-pass reference+proposal allocation.
        arm_config = {**config, "max_sim_agent_steps": len(records) * 80}
        plan = resolved(arm_config, dataset, folder, "evaluate", checkpoint)
        protocols.append(
            {"source": plan["source"], "files": plan["files"], "plan": plan["plan_sha256"]}
        )
        if not plan["ready_to_attempt"]:
            return result(
                output,
                {"missing": plan["missing"], "arm": arm},
                status="failed",
                reason="native_evaluation_prerequisites_missing",
                steps=total,
            )
        outcome = execute(
            plan,
            timeout_seconds=max(0.1, config["max_wall_seconds"] - wall),
            own_process_group=False,
        )
        total += consumed(folder / "native")
        wall += outcome["wall_seconds"]
        reserved_gpu += outcome["reserved_gpu_seconds"]
        if outcome["status"] != "completed":
            return result(
                output, outcome, status="failed", reason="native_evaluation_failed", steps=total
            )
        reports[arm] = json.loads((folder / "native/eval_detailed.json").read_text())
    report = compare(reports["parent"], reports["candidate"], records)
    report.update(
        dataset_identity=card["dataset_identity"],
        student_identity=student["summary"]["artifact_reference"]["identity"],
        evaluation_sources=sorted({r["log_name"] for r in records}),
        wall_seconds=wall,
        reserved_gpu_seconds=reserved_gpu,
        scoring_agent_steps=total,
        protocols=protocols,
    )
    return result(output, report, steps=total, artifacts={
        f"{arm}_{kind}": str(output / arm / "native" / filename)
        for arm in ("parent", "candidate")
        for kind, filename in (("plan", "plan.json"), ("scores", "eval_detailed.json"))
    })


def prepare_feedback(config, base, registry):
    _keys(config, (), "native feedback")
    return config, 0


def run_feedback(config, inputs, output, registry):
    output.mkdir(parents=True, exist_ok=False)
    d, s, e = (inputs[k] for k in ("dataset", "student", "evaluation"))
    report = dict(e["summary"])
    if (
        report["dataset_identity"] != d["summary"]["artifact_reference"]["identity"]
        or report["student_identity"] != s["summary"]["artifact_reference"]["identity"]
    ):
        raise ValueError("Native feedback lineage mismatch")
    report.update(
        kind="native_student_feedback",
        evidence_scope="paired_development_change_only",
        recursive_improvement_validated=False,
        policy_activated_in_reactive_world=False,
        improver_activated=False,
        student=s["summary"],
        proposal_identity=d["summary"]["proposal_identity"],
        replaced_rows=d["summary"]["replaced_rows"],
        next_round={
            "candidate": s["summary"]["checkpoint"],
            "eligible_for_review": report["observed_benefit"],
            "automatic_promotion": False,
        },
        outcome="benefit_requires_replication"
        if report["observed_benefit"]
        else "no_benefit_or_regression",
    )
    return result(output, report)
