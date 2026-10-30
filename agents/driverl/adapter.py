"""DriveRL official preflight/evaluation. Public release is not a trainer."""


def build(cfg, root, repo, python, weights, paths, output, operation):
    s = cfg["settings"]
    if set(s) != {"release_id", "gpu", "task"} or s["release_id"] not in (
        "driverl-teacher-u2400",
        "driverl-selfplay-u3800",
    ):
        raise ValueError("Unsupported official DriveRL release")
    if (
        operation not in ("preflight", "evaluate")
        or set(weights) != {"checkpoint"}
        or set(paths) != {"data", "maps"}
    ):
        raise ValueError("DriveRL native inference/evaluation only")
    if s["task"] not in (
        "val14_nr",
        "val14_r",
        "test14hard_nr",
        "test14hard_r",
        "test14random_nr",
        "test14random_r",
    ):
        raise ValueError("Unsupported official nuPlan task")
    native = repo / "DriveRL"
    kind = "teacher" if s["release_id"] == "driverl-teacher-u2400" else "selfplay"
    config = native / f"release/configs/driverl_{kind}.yaml"
    script = native / "scripts/driverl_release_preflight.py"
    env = dict(
        PYTHONPATH=str(native / "src") + ":" + str(native / "nuplan-devkit"),
        NUPLAN_DATA_ROOT=paths["data"],
        NUPLAN_MAPS_ROOT=paths["maps"],
        CUDA_VISIBLE_DEVICES="" if operation == "preflight" else str(s["gpu"]),
    )
    argv = [
        python,
        str(script),
        "--manifest",
        str(native / "release/driverl_checkpoints.yaml"),
        "--config",
        str(config),
        "--checkpoint",
        weights["checkpoint"],
        "--release-id",
        s["release_id"],
        "--driverl-root",
        str(native),
        "--nuplan-root",
        str(native / "nuplan-devkit"),
        "--device",
        "cpu",
        "--check-simulation-import",
    ]
    files = [
        weights["checkpoint"],
        str(script),
        str(config),
        str(native / "release/driverl_checkpoints.yaml"),
    ]
    directories = [str(native / "nuplan-devkit/nuplan")]
    if operation == "evaluate":
        script = native / "scripts/run_driverl_nuplan_eval.sh"
        argv = ["/bin/bash", str(script), "eval"]
        env.update(
            DRIVERL_EVAL_ROOT=str(native),
            DRIVERL_EVAL_NUPLAN_ROOT=str(native / "nuplan-devkit"),
            DRIVERL_EVAL_PYTHON=python,
            DRIVERL_EVAL_RELEASE_ID=s["release_id"],
            DRIVERL_EVAL_CONFIG_PATH=str(config),
            DRIVERL_EVAL_CHECKPOINT_PATH=weights["checkpoint"],
            DRIVERL_EVAL_OUTPUT_ROOT=str(output),
            DRIVERL_EVAL_TASKS=s["task"],
            DRIVERL_EVAL_LOCAL_GPUS="1",
            DRIVERL_EVAL_WORKER_MODE="sequential",
            DRIVERL_EVAL_TTS_ENABLED="0",
            DRIVERL_EVAL_DEVICE="cuda",
        )
        files.append(str(script))
        directories += list(paths.values())
    return dict(
        argv=argv,
        env=env,
        files=files,
        directories=directories,
        protocol="driverl_nuplan_routefix_six_split",
    )
