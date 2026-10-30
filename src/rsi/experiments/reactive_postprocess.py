"""Same-branch pixels -> independent admission -> native sample -> fresh vectors."""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

# Native model environments retain their released dependencies. Direct file
# execution binds only this repository's namespaces, as the training worker does.
if __package__ in (None, ""):
    import types

    root = Path(__file__).resolve().parents[3]
    for name, folder in (("driversi", "src"), ("driversi_agents", "agents")):
        module = types.ModuleType(name)
        module.__path__ = [str(root / folder)]
        sys.modules[name] = module

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def run_native(action, arguments, directory, *, gpu=6, timeout=3600, python=None):
    """Isolate model environments; an unknown exit never becomes a passed stage."""
    root = project_root()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    job = directory / "job.json"
    frozen_json(job, dict(action=action, arguments=arguments))
    result = directory / "exit.json"
    if result.exists():
        saved = read_mapping(result)
        if saved["job_sha256"] != digest(job):
            raise ValueError("Changed subprocess job")
        if saved["returncode"] != 0:
            raise ValueError("Prior native stage failed; preserve attempt")
        return
    if (directory / "launch.json").exists():
        raise ValueError("Unsettled native stage; reconcile its output before restart")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    with (directory / "worker.log").open("w") as log:
        p = subprocess.Popen(
            [
                sys.executable,
                str(root / "src/runtime/parent_exec.py"),
                str(os.getpid()),
                str(
                    Path(python)
                    if python
                    else root
                    / (
                        ".venvs/mtgs/bin/python"
                        if action == "repair_terrain"
                        else ".venvs/augustus/bin/python"
                    )
                ),
                str(root / "src/rsi/experiments/reactive_postprocess.py"),
                "--job",
                str(job),
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(directory / "launch.json", dict(pid=p.pid, job_sha256=digest(job)))
        interrupted = False
        try:
            code = p.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            interrupted = isinstance(error, KeyboardInterrupt)
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            code = 130 if interrupted else 124
    write_json(result, dict(returncode=code, job_sha256=digest(job)))
    if interrupted:
        raise KeyboardInterrupt("Native stage cancelled after child cleanup")
    if code:
        raise ValueError(f"Native {action} stage failed with code {code}; see sealed worker log")


def process(generation, config, output):
    from driversi.rsi.experiments.readiness import analyze_render

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    g = read_mapping(generation)
    c = read_mapping(config)
    from functools import partial

    worker_gpu = c.get("production_gpu", 6)
    if type(worker_gpu) is not int or not 0 <= worker_gpu <= 7:
        raise ValueError("Invalid production GPU")
    dispatch = partial(run_native, gpu=worker_gpu)
    frozen_json(
        output / "inputs.json",
        dict(generation_sha256=digest(generation), config_sha256=digest(config)),
    )
    if g["status"] != "rendered":
        raise ValueError("Completed physical and GS branch required")
    for p, sha in g["files"].items():
        if digest(p) != sha:
            raise ValueError("Generated branch artifacts changed")
    receipt = read_mapping(g["render_report"])
    source = read_mapping(g["source_report"])
    visual_profile = c.get("visual_quality_profile", "experimental_scene25_v1")
    readiness = analyze_render(receipt, source, visual_profile)
    if c.get("strict_source_psnr_gt") is not None:
        score = readiness.get("source_psnr_mean_db")
        if score is None or not score > float(c["strict_source_psnr_gt"]):
            readiness["blockers"].append("source_psnr_not_strictly_above_required_value")
    write_json(output / "readiness.json", readiness)
    if readiness["blockers"]:
        value = dict(
            status="visual_preflight_rejected", trainable=False, blockers=readiness["blockers"]
        )
        payload = read_mapping(g["render_payload"]) if g.get("render_payload") else {}
        if (
            "finite_terrain_support_missing" in readiness["blockers"]
            and payload.get("renderer_backend") == "worldengine_portable"
        ):
            # Derive measured-support repairs, but keep this original render rejected.
            # A derived grid must be consumed by a new render before visual admission.
            repair_output = output / "terrain-repair"
            try:
                dispatch(
                    "repair_terrain",
                    dict(render_report=g["render_payload"], output=str(repair_output)),
                    output / "calls/terrain-repair",
                    timeout=600,
                )
                repaired = read_mapping(repair_output / "report.json")
                value["terrain_repair"] = dict(
                    status=repaired["status"],
                    report=str(repair_output / "report.json"),
                    failures_before=repaired["failures_before"],
                    failures_after=repaired["failures_after"],
                    rerender_required=True,
                    admission_granted=False,
                )
            except Exception as error:
                value["terrain_repair"] = dict(
                    status="repair_execution_failed",
                    error_type=type(error).__name__,
                    evidence=str(output / "calls/terrain-repair"),
                    admission_granted=False,
                )
        write_json(output / "report.json", value)
        return value
    review = output / "visual"
    from driversi.rendering.diagnostics.worldengine_admission import NUMERIC_PROFILES

    numeric = visual_profile in NUMERIC_PROFILES
    if numeric:
        from driversi.rendering.diagnostics.numeric_admission import admit

        admit(
            generation,
            Path(generation).parent.parent / "diversity.json",
            review,
            profile=visual_profile,
        )
    else:
        dispatch(
            "review",
            dict(
                branch_path=g["branch_report"],
                render_path=g["render_payload"],
                source_path=g["source_report"],
                output=str(review),
                branch_identity=receipt["branch_identity"],
                visual_quality_profile=visual_profile,
                private_config=c["critic_config"],
                credential_file=c["credential_file"],
                request_budget=c["review_ledger"],
            ),
            output / "calls/visual",
        )
    visual = read_mapping(review / "result.json")
    if not visual["admitted"]:
        technical = visual.get("technical_review_failures", [])
        state = (
            "review_budget_exhausted"
            if any(f["reason"] == "request_budget_exhausted" for f in technical)
            else "visual_review_incomplete"
            if technical
            else "visual_rejected"
        )
        value = dict(status=state, trainable=False, visual=str(review / "result.json"))
        write_json(output / "report.json", value)
        return value
    native = output / "native"
    dispatch(
        "preview",
        dict(
            branch_report=g["branch_report"], render_report=g["render_payload"], output=str(native)
        ),
        output / "calls/preview",
    )
    labels = output / "labels"
    if numeric:
        dispatch(
            "geometry_labels",
            dict(preview=str(native), output=str(labels)),
            output / "calls/labels",
        )
    else:
        dispatch(
            "labels",
            dict(
                preview=str(native),
                render_path=g["render_payload"],
                output=str(labels),
                private_config=c["critic_config"],
                credential_file=c["credential_file"],
                request_budget=c["review_ledger"],
            ),
            output / "calls/labels",
        )
        label = read_mapping(labels / "label-review.json")
        if not label["approved"]:
            value = dict(
                status=label.get("technical_review_failure", "native_language_rejected"),
                trainable=False,
                labels=str(labels / "label-review.json"),
            )
            write_json(output / "report.json", value)
            return value
        if digest(label["raw_response"]) != label["raw_response_sha256"]:
            raise ValueError("Language reviewer raw evidence changed")
    dispatch(
        "publish",
        dict(
            branch_report=g["branch_report"],
            render_report=g["render_payload"],
            visual_admission=str(review / "admission.json"),
            label_review=str(labels / "label-review.json"),
            output=str(native),
        ),
        output / "calls/publish",
    )
    export = native / "native/export.json"
    dispatch(
        "features",
        dict(
            export_path=str(export),
            source_catalog=c["train_catalog"],
            profile_id=c["profile"],
            database=c["generated_catalog"],
            output=str(output / "features"),
            sample_report=str(native / "report.json"),
        ),
        output / "calls/features",
    )
    feature = read_mapping(output / "features/report.json")
    if feature.get("trainable") is not True or feature.get("parent_vectors_copied") is not False:
        raise ValueError("Fresh admitted generated-image feature receipt required")
    feature_result = read_mapping(output / "calls/features/result.json")
    dataset_reference = feature_result.get("dataset_reference")
    value = dict(
        dataset_reference=dataset_reference,
        scene_classification=feature_result.get("scene_classification"),
        status="training_sample_ready",
        trainable=True,
        sample_report=str(native / "report.json"),
        feature_report=str(output / "features/report.json"),
        generation=str(Path(generation).resolve()),
        teacher_supervision_not_on_policy=True,
        images_changed=True,
        original_anchor_relabel=False,
        files={
            str(p): digest(p)
            for p in (
                native / "report.json",
                output / "features/report.json",
                review / "admission.json",
                labels / "label-review.json",
            )
        },
    )
    write_json(output / "report.json", value)
    return value


def execute_job(job, *, feature_session=None):
    action = job["action"]
    if action == "dynamic_history_render":
        from driversi.rendering.diagnostics.dynamic_history import render_and_check as function
    elif action == "repair_terrain":
        from driversi.rendering.diagnostics.terrain_repair import run as function
    elif action == "chainflow_export":
        from driversi_agents.chainflow.synthetic_export import run as function
    elif action == "features":
        from driversi.data.catalog.derived_features import encode as function
    elif action in ("review", "labels"):
        from driversi.rendering.diagnostics import gs_sequence_critic

        function = (
            gs_sequence_critic.review if action == "review" else gs_sequence_critic.review_labels
        )
    elif action == "geometry_labels":
        from driversi_agents.augustus.branch_supervision import geometry_label_contract as function
    elif action in ("preview", "publish"):
        from driversi_agents.augustus import branch_supervision

        function = (
            branch_supervision.diagnostic_preview
            if action == "preview"
            else branch_supervision.publish
        )
    else:
        raise ValueError("Unknown fixed pipeline operation")
    arguments = dict(job["arguments"])
    if action == "features" and feature_session is not None:
        arguments["_session"] = feature_session
    result = function(**arguments)
    if action == "features" and result.get("trainable") is True:
        from driversi.data.datasets.synthetic_publish import index_admitted_package

        export = read_mapping(job["arguments"]["export_path"])
        if export.get("package_id"):
            result = dict(
                result,
                dataset_reference=index_admitted_package(
                    job["arguments"]["sample_report"],
                    Path(job["arguments"]["output"]) / "report.json",
                ),
            )
            from driversi.rsi.services.synthetic_classification import complete

            result = dict(result, scene_classification=complete(
                job["arguments"]["export_path"], job["arguments"]["sample_report"]
            ))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    job = read_mapping(args.job)
    if job["action"] == "features":
        from driversi.data.catalog.feature_worker import descriptor, submit

        config = descriptor(job["arguments"]["database"])
        if config.exists():
            from driversi.agents.backends.native import signature

            service = read_mapping(config)
            key = signature(dict(path=str(args.job.resolve()), sha256=digest(args.job)))
            cancel = Path(service["directory"]) / "jobs" / (key + ".cancel.json")

            def cancelled(signum, _frame):
                write_json(cancel, dict(reason="native_proxy_cancelled", signal=signum))
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, cancelled)
            signal.signal(signal.SIGINT, cancelled)
            result = submit(args.job, config)
        else:
            result = execute_job(job)
    else:
        result = execute_job(job)
    write_json(args.job.parent / "result.json", result)


if __name__ == "__main__":
    main()
