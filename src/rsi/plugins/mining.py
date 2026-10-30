"""Artifact-linked evaluation aggregates -> training retrieval -> native verification."""

import csv
import subprocess
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.synthesis import check_files
from driversi.runtime.budget import write_json
from driversi.training.sampling.environment import source_log_group


def _file(base, path):
    return str((base / path).resolve(strict=True))


def prepare_aggregate(config, base, registry):
    _keys(
        config,
        ("manifest", "results", "allow_partial", "official_filter", "descriptors"),
        "evaluation aggregation",
    )
    if type(config["allow_partial"]) is not bool:
        raise ValueError("Explicit partial-coverage policy required")
    out = dict(config)
    files = {}
    for name in ("manifest", "results", "official_filter", "descriptors"):
        if config[name] is not None:
            out[name] = _file(base, config[name])
            files[out[name]] = digest(out[name])
    manifest = read_mapping(out["manifest"])
    if manifest["scope"] == "formal_benchmark":
        if out["official_filter"] is None:
            raise ValueError("Formal feedback must bind the official full scene filter")
        if manifest["benchmark"] != "navtest":
            raise ValueError("Formal importer currently supports native NAVSIM v1 navtest only")
        canonical = project_root() / (
            "agents/chainflow/native/navsim/planning/script/config/common/"
            "train_test_split/scene_filter/navtest.yaml"
        )
        if Path(out["official_filter"]) != canonical.resolve():
            raise ValueError("Formal benchmark requires the repository canonical official filter")
        official = read_mapping(canonical)
        if set(manifest["expected_tokens"]) != set(official["tokens"]) or set(
            manifest["allowed_logs"]
        ) != set(official["log_names"]):
            raise ValueError("Formal cohort differs from the official full filter")
    out["files"] = files
    return out, 0


def _records(config):
    path = Path(config["results"])
    if path.suffix != ".csv":
        return read_mapping(path)["rows"]
    if config["descriptors"] is None:
        raise ValueError("Native CSV needs evaluator-side token/source/history descriptors")
    descriptors = read_mapping(config["descriptors"])["rows"]
    by_token = {r["token"]: r for r in descriptors}
    if len(by_token) != len(descriptors):
        raise ValueError("Duplicate evaluation descriptors")
    from driversi.rsi.decisions.evaluation_aggregate import METRICS

    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["token"] == "average":
                continue  # Never count or trust upstream's aggregate row.
            meta = by_token[row["token"]]
            flag = row["valid"].lower()
            if flag not in ("true", "false"):
                raise ValueError("Native CSV valid flag must be boolean")
            valid = flag == "true"
            rows.append(
                {
                    "token": row["token"],
                    "source_log": meta["source_log"],
                    "conditions": meta["conditions"],
                    "status": "scored" if valid else "failed",
                    "metrics": {k: float(row[k]) for k in ("score", *METRICS)} if valid else None,
                }
            )
    return rows


def run_aggregate(config, inputs, output, registry):
    from driversi.rsi.decisions.evaluation_aggregate import aggregate

    check_files(config["files"])
    report = aggregate(
        read_mapping(config["manifest"]), _records(config), allow_partial=config["allow_partial"]
    )
    check_files(config["files"])
    report["identity_evidence"] = "trusted_operator_import_declarations"
    report["checkpoint_output_parity_verified_by_importer"] = False
    report["coverage_evidence"] = "canonical_token_filter" if (
        report["scope"] == "formal_benchmark") else "declared_bounded_cohort"
    output.mkdir(parents=True)
    return result(output, report)


def prepare_retrieval(config, base, registry):
    _keys(config, ("manifest", "limit", *(k for k in ("scene_index", "capability_budget") if k in config)), "training retrieval")
    if type(config["limit"]) is not int or not 1 <= config["limit"] <= 64:
        raise ValueError("Bounded confirmation pilot requires one to sixty-four candidates")
    path = _file(base, config["manifest"])
    data = read_mapping(path)
    from driversi.rsi.plugins.navsim import _protected

    official = project_root() / (
        "agents/chainflow/native/navsim/planning/script/config/common/"
        "train_test_split/scene_filter/navtrain.yaml"
    )
    logs = sorted({r["source_log"] for r in data["splits"]["train"]})
    protected = sorted(set(_protected(registry)))
    dev = sorted({r["source_log"] for r in data["splits"]["development"]})
    if {source_log_group(x) for x in logs} & {source_log_group(x) for x in protected + dev}:
        raise ValueError("Retrieval training sources overlap heldout/protected sources")
    files = {path: digest(path), str(official): digest(official)}
    # One hash per log, not once per anchor in the full 103k population.
    for log in {row["log_path"] for row in data["splits"]["train"]}:
        files[log] = digest(log)
    scene_index = None
    if "scene_index" in config:
        scene_index = _file(base, config["scene_index"])
        files[scene_index] = digest(scene_index)
        if data.get("schema_version") != 2:
            raise ValueError("Cached full-population retrieval requires benchmark dataset schema 2")
    return {
        "manifest": path,
        **({"scene_index": scene_index} if scene_index else {}),
        "limit": config["limit"],
        **({"capability_budget": config["capability_budget"]} if "capability_budget" in config else {}),
        "files": files,
        "official_filter": str(official),
        "development_logs": dev,
        "protected_logs": protected,
        "source_groups": sorted({source_log_group(x) for x in logs}),
    }, 0


def run_retrieval(config, inputs, output, registry):
    from driversi.rsi.decisions.scene_index import build_index, retrieve

    check_files(config["files"])
    aggregate = read_mapping(inputs["feedback"]["artifacts"]["report"])
    if not aggregate["coverage"]["scored"]:
        raise ValueError("No scored evidence available for retrieval")
    data = read_mapping(config["manifest"])
    if config.get("scene_index"):
        from driversi_agents.chainflow.data import validate_manifest
        validate_manifest(config["manifest"])
        index = read_mapping(config["scene_index"])
        expected = {r["token"]: r for r in data["splits"]["train"]}
        if len(index["records"]) != len(expected) or {r["token"] for r in index["records"]} != set(expected):
            raise ValueError("Full training retrieval index coverage differs")
        for row in index["records"]:
            if any(row[k] != expected[row["token"]][k] for k in ("source_log", "log_path", "anchor_index")):
                raise ValueError("Indexed training anchor differs")
        check_files(index["source_files"])
    else:
        index = build_index(
            data["splits"]["train"],
            official_split_path=config["official_filter"],
            development_groups=config["development_logs"],
            protected_groups=config["protected_logs"],
        )
    report = retrieve(
        index, aggregate["capability_feedback"], config["limit"], cohorts=aggregate["cohorts"],
        per_source_limit=1 if data.get("schema_version") == 2 else None,
        require_continuous_history=data.get("schema_version") == 2,
        capability_budget=config.get("capability_budget"),
    )
    report["capability_feedback"] = aggregate["capability_feedback"]
    report["feedback_coverage"] = aggregate["coverage"]
    report["feedback_scope"] = aggregate["scope"]
    report["development_logs"] = config["development_logs"]
    report["protected_logs"] = config["protected_logs"]
    report["manifest"] = config["manifest"]
    check_files(config["files"])
    output.mkdir(parents=True)
    write_json(output / "index.json", index)
    return result(output, report, artifacts={"index": str(output / "index.json")})


def prepare_confirmation(config, base, registry):
    _keys(config, ("config", "python"), "native candidate confirmation")
    path = _file(base, config["config"])
    # Resolving a venv interpreter symlink discards its environment prefix.
    python = str((base / config["python"]).absolute())
    if not Path(python).is_file():
        raise ValueError("Native Python executable unavailable")
    doc = read_mapping(path)
    cap = doc["max_scoring_steps"]
    if type(cap) is not int or not 0 < cap <= 5120:
        raise ValueError("Small native confirmation scoring budget required")
    files = {path: digest(path)}
    for key in ("manifest", "stage1_config"):
        p = _file(project_root(), doc[key])
        files[p] = digest(p)
    model = read_mapping(_file(project_root(), doc["stage1_config"]))
    for key in ("checkpoint", "backbone"):
        p = _file(project_root(), model[key])
        files[p] = digest(p)
    return {"config": path, "python": python, "files": files}, cap


def run_confirmation(config, inputs, output, registry):
    check_files(config["files"])
    selected = read_mapping(inputs["retrieval"]["artifacts"]["report"])
    doc = read_mapping(config["config"])
    candidates = selected["candidates"]
    output.mkdir(parents=True)
    if not candidates:
        return result(
            output,
            {
                "rows": [],
                "reason": "no_matching_training_candidates",
                "checkpoint": selected["capability_feedback"]["checkpoint"],
                "trainable": False,
            },
        )
    tokens = [r["token"] for r in candidates]
    doc.update(selected_tokens=tokens, output_dir=str(output / "native"))
    model = read_mapping(_file(project_root(), doc["stage1_config"]))
    checkpoint = _file(project_root(), model["checkpoint"])
    if selected["capability_feedback"]["checkpoint"] != checkpoint:
        raise ValueError("Feedback and verification checkpoint differ")
    if Path(_file(project_root(), doc["manifest"])) != Path(selected["manifest"]):
        raise ValueError("Retrieval and verifier pools differ")
    doc["protected_source_logs"] = selected["protected_logs"]
    doc["development_source_logs"] = selected["development_logs"]
    job = output / "request.json"
    write_json(job, doc)
    with (output / "worker.log").open("w") as log:
        proc = subprocess.run(
            [
                config["python"],
                str(project_root() / "agents/chainflow/mining.py"),
                "--config",
                str(job),
            ],
            cwd=project_root(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600 + 10 * max(0, doc["max_samples"] - 16),
        )
    check_files(config["files"])
    receipt = output / "native" / "report.json"
    if proc.returncode or not receipt.is_file():
        return result(
            output,
            {"execution_returncode": proc.returncode, "reason": "native_confirmation_failed"},
            status="failed",
            reason="native_confirmation_failed",
            steps=doc["max_scoring_steps"],
        )
    report = read_mapping(receipt)
    report["retrieval_checkpoint"] = selected["capability_feedback"]["checkpoint"]
    return result(
        output,
        report,
        steps=report["consumed"]["scoring_agent_steps"],
        artifacts={
            "native_report": str(receipt),
            "request": str(job),
            "log": str(output / "worker.log"),
        },
    )


def prepare_decision(config, base, registry):
    _keys(config, (), "confirmed weakness decisions")
    return {}, 0


def run_decision(config, inputs, output, registry):
    from driversi.rsi.decisions.capability_feedback import priorities
    from driversi.rsi.decisions.evaluation_aggregate import METRICS

    retrieved = read_mapping(inputs["retrieval"]["artifacts"]["report"])
    observed = read_mapping(inputs["confirmation"]["artifacts"]["report"])
    wanted = priorities(retrieved["capability_feedback"])
    selected = {r["token"]: r for r in retrieved["candidates"]}
    linkage = inputs["confirmation"]["summary"].get("input_references", {}).get("retrieval")
    expected_link = inputs["retrieval"]["summary"]["artifact_reference"]["identity"]
    tokens = [row["token"] for row in observed["rows"]]
    if (
        linkage != expected_link
        or observed.get("checkpoint") != retrieved["capability_feedback"]["checkpoint"]
        or len(tokens) != len(set(tokens))
        or set(tokens) != set(selected)
        or any(
            row["source_log"] != selected[row["token"]]["source_log"] for row in observed["rows"]
        )
        or (tokens and observed.get("checkpoint_unchanged") is not True)
    ):
        raise ValueError("Confirmation checkpoint, source, coverage or retrieval linkage differs")
    decisions = []
    for row in observed["rows"]:
        if row["token"] not in selected:
            raise ValueError("Confirmation does not belong to selected training cohort")
        if row["status"] != "scored":
            action, shortfalls = "repair_evaluation", []
        else:
            shortfalls = [
                cap
                for metric, cap in METRICS.items()
                if cap in wanted and row["metrics"][metric] < 1
            ]
            action = "same_anchor_attribution" if shortfalls else "retain_or_expand_search"
        decisions.append(
            {
                "token": row["token"],
                "source_log": row["source_log"],
                "retrieval_target_capability": selected[row["token"]].get("match", {}).get("target_capability"),
                "confirmed_capability_shortfalls": shortfalls,
                "action": action,
                "score": row.get("metrics", {}).get("score"),
                "model_cause_established": False,
                "trainable": False,
                "synthesis_authorized": False,
            }
        )
    report = {
        "schema_version": 1,
        "kind": "confirmed_navtrain_weakness_plan",
        "checkpoint": retrieved["capability_feedback"]["checkpoint"],
        "decisions": decisions,
        "trainable": False,
        "promoted": False,
        "next_step": "same_anchor_platform_reference_and_student_controls_before_generation",
        "feedback_scope": retrieved["feedback_scope"],
    }
    output.mkdir(parents=True)
    return result(output, report)


def register(registry, Provider):
    for row in (
        Provider(
            "navsim_evaluation_aggregate",
            "1",
            (),
            "navsim_capability_aggregate",
            prepare_aggregate,
            run_aggregate,
            ("report",),
        ),
        Provider(
            "navsim_train_retrieval",
            "2",
            (("feedback", "navsim_capability_aggregate"),),
            "navsim_training_candidates",
            prepare_retrieval,
            run_retrieval,
            ("report", "index"),
        ),
        Provider(
            "navsim_candidate_confirmation",
            "2",
            (("retrieval", "navsim_training_candidates"),),
            "navsim_candidate_observations",
            prepare_confirmation,
            run_confirmation,
            ("report",),
        ),
        Provider(
            "navsim_confirmed_weakness_decision",
            "1",
            (
                ("retrieval", "navsim_training_candidates"),
                ("confirmation", "navsim_candidate_observations"),
            ),
            "navsim_confirmed_weakness_plan",
            prepare_decision,
            run_decision,
            ("report",),
        ),
    ):
        registry.register(row)
