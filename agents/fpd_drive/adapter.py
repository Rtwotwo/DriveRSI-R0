"""Preserve FPD presets, BF16 ranking, DDIM seed/steps and Top1/Top8 selection."""


def build(cfg, root, repo, python, weights, paths, output, operation):
    s = cfg["settings"]
    if (
        set(s) != {"variant", "benchmark", "gpu"}
        or s["variant"] not in ("base", "pro")
        or s["benchmark"] not in ("v1", "v2")
    ):
        raise ValueError("Unsupported FPD native settings")
    if operation not in ("inference", "evaluate"):
        raise ValueError("FPD release provides inference/evaluation only")
    if set(weights) != {"checkpoint", "backbone"}:
        raise ValueError("FPD checkpoint and backbone required")
    if set(paths) != {
        "navsim_v1",
        "navsim_v2",
        "data",
        "maps",
        "metric_cache",
        "synthetic_sensors",
        "synthetic_scenes",
        "nuplan",
    }:
        raise ValueError("FPD explicit devkit/data/cache paths required")
    argv = [
        python,
        str(repo / "run_inference.py"),
        "--variant",
        s["variant"],
        "--benchmark",
        s["benchmark"],
        "--checkpoint-path",
        weights["checkpoint"],
        "--gpus",
        str(s["gpu"]),
    ]
    if operation == "inference":
        argv.append("--inference-only")
    env = dict(
        CUDA_VISIBLE_DEVICES=str(s["gpu"]),
        NAVSIM_V1_ROOT=paths["navsim_v1"],
        NAVSIM_V2_ROOT=paths["navsim_v2"],
        OPENSCENE_DATA_ROOT=paths["data"],
        NUPLAN_MAPS_ROOT=paths["maps"],
        METRIC_CACHE_PATH=paths["metric_cache"],
        BACKBONE_WEIGHTS=weights["backbone"],
        SYNTHETIC_SENSOR_PATH=paths["synthetic_sensors"],
        SYNTHETIC_SCENES_PATH=paths["synthetic_scenes"],
        OUTPUT_DIR=str(output),
        NAVSIM_EXP_ROOT=str(output),
    )
    filter_root = paths["navsim_v1"] if s["benchmark"] == "v1" else paths["navsim_v2"]
    filter_name = "navtest" if s["benchmark"] == "v1" else "navhard_two_stage"
    env["SCENE_FILTER_PATH"] = (
        filter_root
        + "/navsim/planning/script/config/common/train_test_split/scene_filter/"
        + filter_name
        + ".yaml"
    )
    env["PYTHONPATH"] = paths["nuplan"] + ":" + filter_root
    files = list(weights.values()) + [
        env["SCENE_FILTER_PATH"],
        str(repo / "run_inference.py"),
        str(repo / f"configs/inference/{s['variant']}_{s['benchmark']}.yaml"),
        paths["navsim_v1"] + "/navsim/agents/drivoR/drivor_features.py",
    ]
    directories = [
        paths["nuplan"],
        paths["data"] + "/navsim_logs/test",
        paths["data"] + "/sensor_blobs/test",
    ]
    if operation == "evaluate":
        directories += [paths["maps"], paths["metric_cache"]]
    if s["benchmark"] == "v2":
        directories += [paths["synthetic_sensors"], paths["synthetic_scenes"], paths["navsim_v2"]]
    return dict(
        argv=argv,
        env=env,
        files=files,
        directories=directories,
        protocol="navsim_v1_navtest_pdms"
        if s["benchmark"] == "v1"
        else "navsim_v2_navhard_two_stage",
    )
