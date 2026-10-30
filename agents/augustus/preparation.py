"""Original-log native SFT rows and native metric contexts for bounded campaigns."""

import copy
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi_agents.augustus.cache import build_video, original_window
from driversi_agents.augustus.data import read_cache, validate_row


def rows_cache(records, annotations, output):
    """Preserve official supervision; normalize escaped formatting, never invent labels."""
    output = Path(output)
    identity = dict(
        annotation_sha256=digest(annotations), records=records, builder_sha256=digest(__file__)
    )
    receipt = output / "receipt.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        expected = {k: v for k, v in identity.items() if k != "builder_sha256"}
        if {k: v for k, v in prior["inputs"].items() if k != "builder_sha256"} != expected:
            raise ValueError("Cache resume inputs changed")
        for name, sha in prior["files"].items():
            if digest(name) != sha:
                raise ValueError("Cache resume output changed")
        read_cache(prior["block"])
        return prior["block"]
    output.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(Path(annotations).read_text())
    source = (
        project_root() / "agents/augustus/native/AUGUstus_SFT/data_process/4_4_build_hf_cache.py"
    )
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("original_cache_builder", source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    rows = []
    metadata = []
    evidence = []
    for item in records:
        token, log = item["token"], item["source_log"]
        key = log + "_" + token
        if key not in annotations:
            raise ValueError("Official annotation absent: " + key)
        entry = copy.deepcopy(annotations[key])
        entry["historical_trajectory"] = (
            entry["historical_trajectory"].replace("\\n", "\n").replace("\\t", "\t")
        )
        for field, value in entry["solution"].items():
            entry["solution"][field] = value.replace("\\n", "\n").replace("\\t", "\t")
        with Path(item["log_path"]).open("rb") as f:
            frames = pickle.load(f)
        official_test = item.get("kind") == "original_navtest"
        if official_test:
            from driversi_agents.chainflow.benchmark import official_filter

            _, official = official_filter("navtest")
            if token not in set(official["tokens"]) or log not in set(official["log_names"]):
                raise ValueError("Invalid official test timing exception")
        window, history, future, pose = original_window(
            frames, entry, require_uniform_timing=not official_test
        )
        video = output / "videos" / (token + ".mp4")
        images = build_video(window[:4], item["sensor_root"], video)
        row = dict(
            id=key,
            video=video.read_bytes(),
            scene_token=log,
            frame_id=token,
            driving_command=entry["driving_command"],
            ego_dynamic_state=entry["ego_dynamic_state"],
            historical_x_y_angle=json.dumps(history.tolist()),
            future_x_y_angle=json.dumps(future.tolist()),
            historical_vx_vy=json.dumps(
                [[round(float(x), 2) for x in f["ego_dynamic_state"][:2]] for f in window[:4]]
            ),
            reasoning_trace=None,
            solution_perception=entry["solution"]["perception"],
            solution_prediction=entry["solution"]["prediction"],
            solution_planning=entry["solution"]["planning"],
        )
        validate_row(row)
        rows.append(row)
        metadata.append(
            dict(
                frame_id=token,
                log_name=log,
                timestamp_us=int(window[3]["timestamp"]),
                ego_pose=pose,
            )
        )
        evidence.append(
            dict(
                token=token,
                images=images,
                source_log_sha256=digest(item["log_path"]),
                video_sha256=digest(video),
                official_annotation_key=key,
                prediction=entry["solution"]["prediction"],
                timestamp_us=[int(frame["timestamp"]) for frame in window],
                timing_policy="official_test_indexed_frames_no_interpolation"
                if official_test
                else "continuous_2hz_training",
            )
        )
    cache = output / "cache"
    target = cache / "batch_0"
    if target.exists():
        raise ValueError("Incomplete cache requires a separate attempt")
    native.create_dataset(rows).save_to_disk(str(target))
    block = dict(cache=str(cache), records=metadata)
    write_json(
        receipt,
        dict(
            inputs=identity,
            block=block,
            evidence=evidence,
            files={str(p): digest(p) for p in cache.rglob("*") if p.is_file()},
        ),
    )
    return block


def metric_context(records, maps, output):
    """Generate unchanged native NAVSIM caches for exactly the requested sources."""
    output = Path(output)
    receipt = output / "receipt.json"
    inputs = dict(records=records, maps=str(Path(maps).resolve()), builder_sha256=digest(__file__))
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if {k: v for k, v in prior["inputs"].items() if k != "builder_sha256"} != {
            k: v for k, v in inputs.items() if k != "builder_sha256"
        }:
            raise ValueError("Metric context inputs changed")
        for p, h in prior["files"].items():
            if digest(p) != h:
                raise ValueError("Metric context changed")
        return prior
    sft = project_root() / "agents/augustus/native/AUGUstus_SFT"
    sys.path[:0] = [
        str(sft),
        str(sft / "third_party/navsim"),
        str(sft / "third_party/nuplan-devkit"),
    ]
    os.environ["NUPLAN_MAPS_ROOT"] = inputs["maps"]
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from nuplan.planning.training.experiments.cache_metadata_entry import save_cache_metadata

    tokens = [r["token"] for r in records]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Unique metric tokens required")
    roots = {str(Path(r["log_path"]).parent) for r in records}
    if len(roots) != 1:
        raise ValueError("Metric request must have one log split root")
    loader = SceneLoader(
        Path(next(iter(roots))),
        None,
        SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            log_names=sorted({r["source_log"] for r in records}),
            tokens=tokens,
        ),
        SensorConfig.build_no_sensors(),
    )
    if set(loader.tokens) != set(tokens):
        raise ValueError("Missing requested native scoring window")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "metric_cache"
    processor = MetricCacheProcessor(str(cache), False)
    entries = []
    for token in loader.tokens:
        scenario = NavSimScenario(
            loader.get_scene_from_token(token),
            map_root=inputs["maps"],
            map_version="nuplan-maps-v1.0",
        )
        entry = processor.compute_metric_cache(scenario)
        if entry is None:
            raise ValueError("Metric generation failed")
        entries.append(entry)
    save_cache_metadata(entries, cache, 0)
    checker = MetricCacheLoader(cache)
    for token in tokens:
        item = checker.get_from_token(token)
        if not hasattr(item.centerline, "project"):
            raise ValueError("Wrong native context")
    from driversi_agents.augustus.metric_cache import scoring_files

    files = scoring_files(
        cache, [dict(frame_id=r["token"], log_name=r["source_log"]) for r in records]
    )
    result = dict(inputs=inputs, metric_cache=str(cache), files={str(p): digest(p) for p in files})
    write_json(receipt, result)
    return result
