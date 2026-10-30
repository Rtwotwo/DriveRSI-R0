"""Admitted GS samples -> native SFT pair -> same-host full benchmark feedback."""

import copy
import json
import socket
from pathlib import Path

import yaml

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import task_state
from driversi.runtime.budget import write_json


def sampling_budget(original_rows, synthetic_rows, *, family=False, requested_steps=4):
    if min(original_rows, synthetic_rows, requested_steps) <= 0:
        raise ValueError("Nonempty pair and positive budget required")
    total = original_rows + synthetic_rows
    batch = 2 if original_rows % 2 == 0 and total % 2 == 0 else 1
    steps = max(requested_steps, total // batch + 1) if family else requested_steps
    if steps > 64:
        raise ValueError("Sample coverage exceeds bounded native training budget")
    return batch, steps


def compile_pair(
    config_path, sample_reports, output, *, max_samples=4, optimizer_steps=4, worker_gpu=6
):
    from driversi.data.catalog.store import Catalog
    from driversi.rsi.experiments.augustus_round import _task
    from driversi.rsi.experiments.navval_loop import prepare_evaluation
    from driversi.rsi.experiments.navval_retrieval import training_members
    from driversi.training.sampling.environment import source_log_group
    from driversi_agents.augustus.branch_supervision import build_training_pair, validate_sample
    from driversi_agents.augustus.preparation import rows_cache
    from driversi_agents.chainflow.benchmark import read_index

    if type(max_samples) is not int or not 1 <= max_samples <= 32:
        raise ValueError("Bounded qualified sample budget required")
    if type(optimizer_steps) is not int or not 1 <= optimizer_steps <= 64:
        raise ValueError("Small native update budget required")
    if type(worker_gpu) is not int or not 0 <= worker_gpu <= 7:
        raise ValueError("Invalid pair training GPU")
    settings = dict(max_samples=max_samples, optimizer_steps=optimizer_steps)
    if worker_gpu != 6:
        settings["worker_gpu"] = worker_gpu
    c = read_mapping(config_path)
    root = project_root()
    out = Path(output).resolve()
    inputs = {str(Path(p).resolve()): digest(p) for p in [config_path, *sample_reports]}
    if (out / "compiled.json").exists():
        old = read_mapping(out / "compiled.json")
        if (
            old["inputs"] != inputs
            or old.get("pair_settings", dict(max_samples=4, optimizer_steps=4)) != settings
        ):
            raise ValueError("Compiled pair inputs changed")
        for p, sha in old["files"].items():
            if digest(p) != sha:
                raise ValueError("Compiled training declaration changed")
        return old
    samples = [validate_sample(p) for p in sample_reports]
    if not 1 <= len(samples) <= max_samples:
        raise ValueError("Qualified GS count exceeds declared pair budget")
    prepare_evaluation(config_path)
    out.mkdir(parents=True, exist_ok=True)
    index = read_index(c["train_index"], "navtrain")
    known = {x["token"]: x for x in index["records"]}
    cat = Catalog(c["train_catalog"], "navtrain", readonly=True)
    try:
        allowed, _ = training_members(cat, "navtrain")
    finally:
        cat.close()
    tokens = []
    for sample in samples:
        branch = read_mapping(sample["branch_report"])
        from driversi.contracts.scene import Scene

        scene = Scene.load(Path(branch["execution"]) / "scene.npz")
        origin = scene.metadata["source_train_membership"]
        if allowed.get(origin["token"]) != sample["source_log"]:
            raise ValueError("Synthetic source is outside official training membership")
        tokens.append(origin["token"])
    originals = [known[t] for t in sorted(set(tokens))]
    batch_size, effective_steps = sampling_budget(
        len(originals), len(samples), family=max_samples > 4, requested_steps=optimizer_steps
    )
    train = rows_cache(originals, c["train_annotations"], out / "original-train")
    groups = {source_log_group(x["source_log"]) for x in originals}
    validation = next(
        x
        for x in index["records"]
        if x["token"] in allowed and source_log_group(x["source_log"]) not in groups
    )
    val = rows_cache([validation], c["train_annotations"], out / "validation")
    parent_out = Path(c["output"])
    dev = read_mapping(parent_out / "evaluation-data/receipt.json")["block"]
    base = dict(
        schema_version=1,
        kind="augustus_full_navtest_cache",
        dataset_version="reactive-navtrain-gs-v2",
        splits=dict(train=train, validation=val, development=dev),
        protected_logs=[],
        parent_training_logs=[],
        parent_provenance="unknown",
        evaluation_scope="official_navtest_v1",
        benchmark=dict(
            protocol="official_navtest_v1",
            test_subset=c["test_index"],
            test_subset_sha256=digest(c["test_index"]),
        ),
    )
    write_json(out / "original.json", base)
    pair = build_training_pair(
        out / "original.json", sample_reports, out / "joint", allow_smoke=True
    )
    context = read_mapping(parent_out / "evaluation-context/receipt.json")
    tasks = {}
    declared = []
    for arm, key in (("control", "control"), ("treatment", "intervention")):
        from driversi_agents.augustus.data import manifest

        d = out / arm
        d.mkdir(exist_ok=True)
        mp = Path(pair[key])
        value = manifest(mp, allow_smoke=True)
        data = dict(
            schema_version=1,
            parent_provenance="unknown",
            manifest=str(mp),
            metric_cache=context["metric_cache"],
            evaluation_protocol="official_navtest_v1",
            evaluation_subset=c["test_index"],
            full_evaluation_plan=str(parent_out / "baseline/plan.json"),
            full_evaluation_execution=dict(hostname="jory-h20-4-0", gpus=list(range(8))),
            train=dict(
                source_logs=sorted({r["log_name"] for r in value["splits"]["train"]["records"]})
            ),
            development=dict(source_logs=sorted({r["log_name"] for r in dev["records"]})),
        )
        model = copy.deepcopy(read_mapping(root / "src/config/agents/augustus/native.yaml"))
        model["settings"].update(gpu=worker_gpu, gpus=[worker_gpu])
        model["weights"]["parent"] = "weights/augustus/SFT"
        model["paths"].update(
            dataset_manifest=str(mp), development_metric_cache=context["metric_cache"]
        )
        write_json(d / "data.json", data)
        (d / "model.yaml").write_text(yaml.safe_dump(model))
        request = dict(
            schema_version=3,
            id="reactive-navval-" + arm,
            model_config=str(d / "model.yaml"),
            mode="supervised",
            initialization="model_weights",
            data=str(d / "data.json"),
            pilot=dict(seed=c["seed"], batch_size=batch_size, evaluation_steps=12146),
            budget=dict(
                max_wall_seconds=86400,
                max_optimizer_steps=effective_steps,
                max_environment_steps=0,
                max_scoring_agent_steps=2 * 12146 * 40,
            ),
        )
        write_json(d / "request.json", request)
        evaluator = dict(
            schema_version=1,
            model="augustus",
            protocol="bundled_navsim_pdm_v1",
            development_source_logs=data["development"]["source_logs"],
            dataset_manifest_sha256=digest(mp),
            metric_cache=context["metric_cache"],
            files=context["files"],
        )
        write_json(d / "evaluator.json", evaluator)
        task = _task(request["id"], d / "request.json", d / "evaluator.json")
        task["steps"] = task["steps"][
            :3
        ]  # Typed optimizer/resume/replay receipts; test worker is on32000.
        write_json(d / "task.json", task)
        tasks[arm] = str(d / "task.json")
        declared.extend(
            d / p
            for p in ("request.json", "model.yaml", "data.json", "evaluator.json", "task.json")
        )
    value = dict(
        kind="native_reactive_pair",
        pair_settings=settings,
        effective_pair_budget=dict(
            batch_size=batch_size,
            optimizer_steps_per_phase=effective_steps,
            train_and_resume_phases=2,
            exposure_checked_before_full_evaluation=True,
        ),
        tasks=tasks,
        inputs=inputs,
        samples=sample_reports,
        registry=str(out / "harness.sqlite"),
        output=str(out),
        files={
            str(p): digest(p)
            for p in [
                *declared,
                out / "original.json",
                Path(pair["control"]),
                Path(pair["intervention"]),
            ]
        },
        train_host="gene-i-0jl0w4i4r9fl5us6cstn-0",
        evaluation_host="jory-h20-4-0",
    )
    write_json(out / "compiled.json", value)
    return value


def verify_training_exposure(plan, rows):
    """Fail before 12146-sample evaluation if generated rows did not update weights."""
    from driversi.rsi.experiments.augustus_pair import _positive_exposure

    expected = {read_mapping(p)["sample_id"] for p in plan["samples"]}
    events = []
    root = Path(rows["treatment"]["directory"])
    for phase in ("train", "resume"):
        report = read_mapping(root / phase / "training-result.json")
        phase_events = [
            json.loads(line)
            for line in (root / phase / "consumed-training-samples.jsonl").read_text().splitlines()
        ]
        if len(phase_events) != report["updates"] or any(not e["sample_ids"] for e in phase_events):
            raise ValueError("Actual optimizer exposure journal incomplete")
        events.extend(phase_events)
    positive = _positive_exposure(events)
    return dict(
        passed=bool(expected) and expected <= positive,
        required=sorted(expected),
        positive_lr_samples=sorted(positive),
        missing=sorted(expected - positive),
    )


def train(compiled):
    plan = read_mapping(compiled)
    if socket.gethostname() != plan["train_host"]:
        raise ValueError("Wrong training host")
    out = Path(plan["output"])
    registry = Path(plan["registry"])
    rows = {}
    for arm, path in plan["tasks"].items():
        state = task_state(registry, read_mapping(path), Path(path), out / "tasks", execute=True)
        rows[arm] = dict(status=state["status"], directory=state["directory"])
        if state["status"] != "completed":
            write_json(out / "training-status.json", dict(status="failed", arms=rows))
            return False
    exposure = verify_training_exposure(plan, rows)
    write_json(out / "training-exposure.json", exposure)
    if not exposure["passed"]:
        write_json(
            out / "training-status.json",
            dict(
                status="failed",
                arms=rows,
                reason="generated_samples_missing_positive_lr_updates",
                exposure=exposure,
            ),
        )
        return False
    write_json(
        out / "training-status.json",
        dict(
            status="completed",
            arms=rows,
            compiled_sha256=digest(compiled),
            awaiting_full_navtest=True,
        ),
    )
    return True


def evaluate_and_feedback(compiled):
    from driversi.rsi.experiments.augustus_pair import compare
    from driversi.rsi.plugins.training import run_feedback
    from driversi.training.contracts.job import resolve_request
    from driversi.training.orchestration import lifecycle

    plan = read_mapping(compiled)
    if socket.gethostname() != plan["evaluation_host"]:
        raise ValueError("Candidate evaluation requires baseline host")
    out = Path(plan["output"])
    status = read_mapping(out / "training-status.json")
    if status["status"] != "completed" or status["compiled_sha256"] != digest(compiled):
        raise ValueError("Unsettled training")
    roots = {}
    for arm, row in status["arms"].items():
        root = Path(row["directory"])
        roots[arm] = root
        target = root / "evaluate"
        if not (target / "training-result.json").exists():
            if target.exists():
                raise ValueError("Unsettled evaluation attempt requires reconciliation")
            evaluation = lifecycle.prepare(
                resolve_request(out / arm / "request.json"),
                target,
                phase="evaluate",
                previous=root / "resume/training-result.json",
                evaluator=out / arm / "evaluator.json",
            )
            result = lifecycle.execute(evaluation)
        else:
            result = read_mapping(target / "training-result.json")
        if result["status"] != "completed":
            write_json(
                out / "evaluation-status.json",
                dict(status="failed", arm=arm, reason=result.get("reason")),
            )
            return False
        if not (root / "feedback/feedback.json").exists():
            inputs = {
                key: dict(artifacts=dict(report=str(root / stage / "training-result.json")))
                for key, stage in (
                    ("student", "resume"),
                    ("replay", "replay"),
                    ("evaluation", "evaluate"),
                )
            }
            run_feedback({}, inputs, root / "feedback", Path(plan["registry"]))
    result = compare(roots["control"], roots["treatment"])
    write_json(out / "paired-feedback.json", result)
    from driversi.rsi.decisions.service_memory import ServiceMemory, review_learning

    resume_checks = [
        read_mapping(roots[arm] / "feedback/feedback.json")["cross_process_resume"]
        for arm in ("control", "treatment")
    ]
    verdict = review_learning(result, resume_checks)
    verdict["evaluation_scope"] = "full_navtest_12146"
    verdict["promotion_reason"] = "single_adaptive_test_trial_requires_predeclared_replication"
    write_json(out / "learning-decision.json", verdict)
    memory = ServiceMemory(out.parent / "service-memory.sqlite", "evolve")
    try:
        context = dict(
            model="augustus",
            parent=digest(project_root() / "weights/augustus/SFT/model.safetensors"),
            protocol="full_navtest_12146",
            intervention="actual_driverl_history_gs_native_sft",
        )
        memory.record(
            dict(
                **verdict,
                program_id=digest(compiled),
                context=context,
                source_logs=sorted({read_mapping(p)["source_log"] for p in plan["samples"]}),
                evidence={
                    str(p): digest(p)
                    for p in (out / "paired-feedback.json", out / "learning-decision.json")
                },
                training_executed=True,
            )
        )
    finally:
        memory.close()
    write_json(
        out / "evaluation-status.json",
        dict(
            status="completed",
            full_navtest=True,
            expected=12146,
            paired_feedback=str(out / "paired-feedback.json"),
            decision=str(out / "learning-decision.json"),
            promoted=False,
            reason="bounded_first_paired_trial_not_stable_gain_evidence",
        ),
    )
    return True
