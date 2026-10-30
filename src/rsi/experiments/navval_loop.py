"""Bounded AUGUstus SFT evaluation, navval retrieval and native learning loop."""

import argparse
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore
from driversi.rsi.experiments.navval_retrieval import navval_members
from driversi.runtime.budget import write_json


def require_relabel_route(config):
    c = read_mapping(config)
    policy = Path(c["output"]) / "continuation-policy.json"
    if (
        policy.exists()
        and read_mapping(policy).get("production_route") != "original_anchor_relabel"
    ):
        raise ValueError(
            "Campaign production now requires DriveRL new history and GS; relabel fallback is disabled"
        )


def preflight(config):
    c = read_mapping(config)
    catalog = Catalog(c["train_catalog"], "navtrain", readonly=True)
    try:
        members, files = navval_members(catalog)
        vectors = VectorStore(catalog)
        vectors.profile_data(c["profile"])
        sample = next(iter(members))
        probe = catalog.db.execute(
            "SELECT status FROM vectors WHERE profile=? AND token=?", (c["profile"], sample)
        ).fetchone()
        result = dict(
            status="prepared",
            navval_members=len(members),
            ready_navval_vectors=None,
            vector_probe_status=probe[0] if probe else "missing",
            coverage="measured during bounded retrieval",
            membership_files=files,
            training_started=False,
        )
        out = Path(c["output"])
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "preflight.json", result)
        return result
    finally:
        catalog.close()


def prepare_evaluation(config):
    from driversi_agents.augustus.preparation import metric_context, rows_cache
    from driversi_agents.chainflow.benchmark import make_pilot_subset, read_pilot_subset

    c = read_mapping(config)
    out = Path(c["output"])
    if c["evaluation_count"] == 12146:
        from driversi_agents.augustus.benchmark_bundle import materialize

        return materialize(out / "baseline/plan.json", out)
    subset = out / "locked-navtest.json"
    if subset.exists():
        pilot = read_pilot_subset(subset)
    else:
        pilot = make_pilot_subset(
            c["test_index"], subset, count=c["evaluation_count"], seed=c["selection_seed"]
        )
    block = rows_cache(pilot["records"], c["test_annotations"], out / "evaluation-data")
    context = metric_context(pilot["records"], c["maps"], out / "evaluation-context")
    return dict(
        status="prepared", count=len(block["records"]), metric_cache=context["metric_cache"]
    )


def retrieve_and_relabel(config):
    import numpy as np

    from driversi.agents.backends.native import digest
    from driversi.rsi.experiments.navval_retrieval import retrieve
    from driversi.rsi.plugins.tto import collect
    from driversi_agents.augustus.data import manifest, relabel
    from driversi_agents.augustus.preparation import metric_context, rows_cache
    from driversi_agents.chainflow.benchmark import read_index

    require_relabel_route(config)
    c = read_mapping(config)
    out = Path(c["output"])
    baseline = read_mapping(out / "baseline/report.json")
    if (
        c["evaluation_count"] != 12146
        or baseline.get("completed") != 12146
        or not baseline.get("complete")
    ):
        raise ValueError("This campaign requires completed full 12146 navtest before mapping")
    poor = sorted(
        [x for x in baseline["rows"] if x["metrics"]["PDMS"] < 0.95],
        key=lambda x: (x["metrics"]["PDMS"], x["token"]),
    )
    if not poor:
        raise ValueError("No below-95 sample in full benchmark; do not fabricate a failure")
    q = Catalog(c["test_catalog"], "navtest", readonly=True)
    t = Catalog(c["train_catalog"], "navtrain", readonly=True)
    try:
        allowed, membership = navval_members(t)
        retrieval = out / "retrieval.json"
        if retrieval.exists():
            result = read_mapping(retrieval)
        else:
            queue = []
            ready = []
            for row in poor:
                available = q.db.execute(
                    "SELECT status FROM vectors WHERE profile=? AND token=?",
                    (c["profile"], row["token"]),
                ).fetchone()
                ok = available is not None and available[0] == "ready"
                queue.append(dict(token=row["token"], pdms=row["metrics"]["PDMS"], vector_ready=ok))
                if ok and len(ready) < 4:
                    ready.append(row["token"])
            write_json(
                out / "failure-query-queue.json",
                dict(
                    rows=queue,
                    zero_count=sum(x["pdms"] == 0 for x in queue),
                    missing_vectors=sum(not x["vector_ready"] for x in queue),
                    selection="zero_then_ascending_pdms_first4_ready",
                    private=True,
                ),
            )
            if not ready:
                raise ValueError("No ready failure vectors")
            result = retrieve(q, t, c["profile"], ready, limit=8)
            result["configuration"] = dict(
                profile=c["profile"],
                max_training_scenes=c["max_training_scenes"],
                queries=4,
                neighbors=8,
            )
            result["baseline_sha256"] = digest(out / "baseline/report.json")
            write_json(retrieval, result)
        if result.get("configuration") != dict(
            profile=c["profile"],
            max_training_scenes=c["max_training_scenes"],
            queries=4,
            neighbors=8,
        ):
            raise ValueError("Retrieval configuration changed")
        if result["baseline_sha256"] != digest(out / "baseline/report.json"):
            raise ValueError("Retrieval parent evidence changed")
    finally:
        q.close()
        t.close()
    # Round-robin query allocation avoids one failure dominating the candidate list.
    selected = []
    groups = set()
    for rank in range(8):
        for row in result["rows"]:
            if rank >= len(row["candidates"]):
                continue
            x = row["candidates"][rank]
            if x["token"] not in allowed or allowed[x["token"]] != x["source_log"]:
                raise ValueError("Not a navval member")
            if x["source_group"] not in groups:
                selected.append(x)
                groups.add(x["source_group"])
            if len(selected) == c["max_training_scenes"]:
                break
        if len(selected) == c["max_training_scenes"]:
            break
    if len(selected) != c["max_training_scenes"]:
        raise ValueError("Insufficient diverse candidates")
    train_index = read_index(c["train_index"], "navtrain")
    known = {x["token"]: x for x in train_index["records"]}
    originals = [known[x["token"]] for x in selected]
    train = rows_cache(originals, c["train_annotations"], out / "training-data")
    from driversi.training.sampling.environment import source_log_group

    validation = next(
        x
        for x in train_index["records"]
        if x["token"] in allowed and source_log_group(x["source_log"]) not in groups
    )
    val = rows_cache([validation], c["train_annotations"], out / "validation-data")
    dev = read_mapping(out / "evaluation-data/receipt.json")["block"]
    subset = Path(c["test_index"])
    base = dict(
        schema_version=1,
        kind="augustus_full_navtest_cache",
        dataset_version="navval-vector-original-v1",
        splits=dict(train=train, validation=val, development=dev),
        protected_logs=[],
        parent_training_logs=[],
        parent_provenance="unknown",
        evaluation_scope="official_navtest_v1",
        benchmark=dict(
            protocol="official_navtest_v1",
            test_subset=str(subset),
            test_subset_sha256=digest(subset),
        ),
    )
    write_json(out / "original.json", base)
    manifest(out / "original.json", allow_smoke=True)
    write_json(out / "cases.json", [dict(split="val", token=x["token"]) for x in selected])
    write_json(
        out / "protected.json", [x["source_log"] for x in read_index(subset, "navtest")["records"]]
    )
    cfg = dict(
        bank=c["bank"],
        cases=str(out / "cases.json"),
        protected_logs=str(out / "protected.json"),
        cases_sha256=digest(out / "cases.json"),
        protected_sha256=digest(out / "protected.json"),
    )
    proposals = collect(cfg, allowed_navval=allowed)
    proposals["source_label_precision"] = "official_p3_two_decimals"
    context = metric_context(originals, c["maps"], out / "teacher-context")
    import sys

    from driversi.agents.backends.native import project_root

    sft = project_root() / "agents/augustus/native/AUGUstus_SFT"
    sys.path[:0] = [
        str(sft),
        str(sft / "third_party/navsim"),
        str(sft / "third_party/nuplan-devkit"),
    ]
    import model.utils as utils

    if not utils._init_pdm(context["metric_cache"]):
        raise RuntimeError("Native teacher scoring unavailable")
    accepted = []
    for row in proposals["accepted"]:
        score = utils.compute_pdm_scores_batch(
            np.asarray([row["trajectory"]], dtype=np.float32),
            [row["token"]],
            metric_cache_path=context["metric_cache"],
            return_per_sample=True,
        )
        row["native_recheck"] = score["per_sample"][0]
        if row["native_recheck"] is None:
            raise ValueError("Missing teacher recheck")
        if row["native_recheck"]["PDMS"] >= 0.95:
            accepted.append(row)
        else:
            row["reason"] = "native_pdms_below_95"
            proposals["rejected"].append(row)
    proposals["accepted"] = accepted
    proposals.update(
        native_context_files=context["files"], new_score_queries=len(selected), new_rollouts=0
    )
    write_json(out / "proposals.json", proposals)
    if not accepted:
        raise ValueError("No candidate passes original and native teacher scoring")
    if not (out / "relabel/report.json").exists():
        relabel(out / "original.json", out / "proposals.json", out / "relabel", allow_smoke=True)
    return dict(
        status="prepared",
        poor_test_samples=len(poor),
        queries=len(result["rows"]),
        navval_selected=len(selected),
        relabeled=len(accepted),
    )


def compile_training(config):
    import copy

    import yaml

    from driversi.agents.backends.native import digest, project_root
    from driversi.rsi.experiments.augustus_round import _task

    require_relabel_route(config)
    c = read_mapping(config)
    out = Path(c["output"])
    root = project_root()
    native = read_mapping(root / "src/config/agents/augustus/native.yaml")
    context = read_mapping(out / "evaluation-context/receipt.json")
    tasks = {}
    for arm, mp in (
        ("control", out / "original.json"),
        ("treatment", out / "relabel/dataset.json"),
    ):
        d = out / "pair" / arm
        d.mkdir(parents=True, exist_ok=True)
        value = read_mapping(mp)
        data = dict(
            schema_version=1,
            parent_provenance="unknown",
            manifest=str(mp),
            metric_cache=context["metric_cache"],
            evaluation_protocol="official_navtest_v1",
            evaluation_subset=c["test_index"],
            full_evaluation_plan=str(out / "baseline/plan.json"),
            train=dict(
                source_logs=sorted({r["log_name"] for r in value["splits"]["train"]["records"]})
            ),
            development=dict(
                source_logs=sorted(
                    {r["log_name"] for r in value["splits"]["development"]["records"]}
                )
            ),
        )
        model = copy.deepcopy(native)
        model["settings"]["gpu"] = c["gpu"]
        model["settings"]["gpus"] = [c["gpu"]]
        model["weights"]["parent"] = "weights/augustus/SFT"
        model["paths"]["dataset_manifest"] = str(mp)
        model["paths"]["development_metric_cache"] = context["metric_cache"]
        write_json(d / "data.json", data)
        (d / "model.yaml").write_text(yaml.safe_dump(model))
        request = dict(
            schema_version=3,
            id="navval-vector-" + arm,
            model_config=str(d / "model.yaml"),
            mode="supervised",
            initialization="model_weights",
            data=str(d / "data.json"),
            pilot=dict(seed=c["seed"], batch_size=2, evaluation_steps=c["evaluation_count"]),
            budget=dict(
                max_wall_seconds=86400,
                max_optimizer_steps=4,
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
        write_json(d / "task.json", _task(request["id"], d / "request.json", d / "evaluator.json"))
        tasks[arm] = str(d / "task.json")
    write_json(out / "pair/tasks.json", tasks)
    return tasks


def train_and_feedback(config):
    from driversi.agents.backends.native import digest
    from driversi.rsi.experiments.augustus_pair import compare, record
    from driversi.rsi.workflow import engine
    from driversi.rsi.workflow.store import HarnessStore

    require_relabel_route(config)
    c = read_mapping(config)
    out = Path(c["output"])
    registry = out / "harness.sqlite"
    tasks = (
        read_mapping(out / "pair/tasks.json")
        if (out / "pair/tasks.json").exists()
        else compile_training(config)
    )
    roots = {}
    for arm, path in tasks.items():
        task = read_mapping(path)
        store = HarnessStore(registry)
        try:
            state = store.get(task["id"])
        except ValueError as exc:
            if str(exc) != "Unknown Harness task":
                raise
            engine.register(registry, path, out / "tasks")
            state = store.get(task["id"])
        if state["status"] == "registered":
            state = engine.run(registry, task["id"])
        if state["status"] == "running":
            state = engine.recover(registry, task["id"])
        if state["status"] != "completed":
            raise ValueError("Native task not completed: " + arm + " " + state["status"])
        roots[arm] = out / "tasks" / task["id"]
    report = compare(
        roots["control"],
        roots["treatment"],
        relabel_report=out / "relabel/report.json",
        generation_costs=dict(
            retrieval=str(out / "retrieval.json"), teacher=str(out / "proposals.json"), new_images=0
        ),
    )
    from driversi.rsi.decisions.service_memory import review_learning

    checks = [
        read_mapping(root / "feedback/feedback.json")["cross_process_resume"]
        for root in roots.values()
    ]
    review = review_learning(report, checks)
    report["resume_evidence"] = {arm: check for arm, check in zip(roots, checks, strict=True)}
    report["source_partition"] = "navval"
    report["retrieval_sha256"] = digest(out / "retrieval.json")
    write_json(out / "learning-review.json", review)
    if review["node"] == "resume_verification":
        write_json(out / "rejected-paired-feedback.json", report)
        raise ValueError("Cross-process native state recovery mismatch")
    receipt = record(
        report, out / "paired-feedback.json", out / "learning-memory.sqlite", resume=True
    )
    write_json(out / "memory-receipt.json", receipt)
    write_json(
        out / "next-iteration.json",
        dict(
            action=review["reason"],
            review_status=review["status"],
            promoted=False,
            why="One bounded adaptive-test experiment cannot establish stable multi-round improvement",
            gain_vs_control=report["gain_vs_control_pdms_points"],
            gain_vs_parent=report["gain_vs_parent_pdms_points"],
            safety_regression_count=len(report["safety_regressions"]),
            feedback_sha256=digest(out / "paired-feedback.json"),
        ),
    )
    return report


def run(config):
    import fcntl

    c = read_mapping(config)
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "loop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight(config)
        if c["evaluation_count"] != 12146 or (out / "scope-superseded.json").exists():
            raise ValueError("Superseded pilot cannot trigger training")
        from driversi.rsi.experiments.augustus_campaign import evaluate_full

        evaluate_full(config)
        prepare_evaluation(config)
        retrieval = retrieve_and_relabel(config)
        write_json(out / "augmentation-summary.json", retrieval)
        from driversi.data.catalog.augmentations import register

        features = register(
            c["train_catalog"],
            c["profile"],
            out / "original.json",
            out / "relabel/report.json",
            out / "generated-vectors.sqlite",
        )
        write_json(out / "feature-ingestion.json", features)
        result = train_and_feedback(config)
        summary = dict(
            status="completed",
            evaluation_samples=c["evaluation_count"],
            **{k: v for k, v in retrieval.items() if k != "status"},
            metrics=result["metrics"],
            gain_vs_control=result["gain_vs_control_pdms_points"],
            gain_vs_parent=result["gain_vs_parent_pdms_points"],
            safety_regressions=len(result["safety_regressions"]),
            promoted=False,
            new_images=0,
            augmentation_kind="original_anchor_relabel",
            source_partition="navval",
            adaptive_test=True,
            full_benchmark=True,
        )
        write_json(out / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = {
        "preflight": preflight,
        "prepare-evaluation": prepare_evaluation,
        "augment": retrieve_and_relabel,
        "train": train_and_feedback,
        "run": run,
    }
    parser.add_argument("command", choices=list(actions))
    parser.add_argument("config")
    args = parser.parse_args()
    try:
        result = actions[args.command](args.config)
    except Exception as exc:
        c = read_mapping(args.config)
        write_json(
            Path(c["output"]) / "last-failure.json",
            dict(stage=args.command, error=type(exc).__name__ + ": " + str(exc)),
        )
        raise
    print(result)
