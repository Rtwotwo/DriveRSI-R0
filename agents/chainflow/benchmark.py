"""Official NAVSIM population transport and resumable ChainFlow evaluation.

Ported engineering semantics from DriveScorer's feedback-v2 evaluator: exact
test membership, strict checkpoint loading, inference-only inputs, per-token
scores and explicit incomplete results. No FeedbackV2 model/loss is imported.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import sys
from numbers import Real
from pathlib import Path

if __package__ in (None, ""):
    import types
    _root = Path(__file__).resolve().parents[2]
    for _name, _folder in (("driversi", "src"), ("driversi_agents", "agents")):
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_root / _folder)]
        sys.modules[_name] = _module

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi_agents.chainflow.data import source_group

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance",
           "driving_direction_compliance", "ego_progress",
           "time_to_collision_within_bound", "comfort")
PILOT_PROTOCOL = "official_navtest_v1_locked_pilot"


def select_pilot_rows(records, count, seed):
    """Score-blind deterministic sampling with one anchor per original log."""
    if type(count) is not int or not 1 <= count <= 256 or not isinstance(seed, str) or not seed:
        raise ValueError("Locked pilot requires 1..256 samples and a nonempty selection seed")
    ordered = sorted(records, key=lambda row: (
        hashlib.sha256((seed + ":" + row["token"]).encode()).hexdigest(), row["token"]))
    rows, groups = [], set()
    for row in ordered:
        group = source_group(row["source_log"])
        if group not in groups:
            rows.append(row)
            groups.add(group)
        if len(rows) == count:
            return rows
    raise ValueError("Insufficient independent original logs for locked pilot")


def make_pilot_subset(test_index, output, *, count=64, seed="gs-multiview-training-v2"):
    index = read_index(test_index, "navtest")
    rows = select_pilot_rows(index["records"], count, seed)
    value = dict(schema_version=1, kind="chainflow_locked_navtest_pilot",
                 protocol=PILOT_PROTOCOL, test_index=str(Path(test_index).resolve()),
                 test_index_sha256=digest(test_index), selection_seed=seed, count=count,
                 selection_method="sha256_seed_token_one_per_original_log",
                 tokens=[r["token"] for r in rows], records=rows,
                 full_benchmark=False, adaptive_evaluation=True)
    write_json(output, value)
    return value


def read_pilot_subset(path):
    value = json.loads(Path(path).read_text())
    required = {"schema_version", "kind", "protocol", "test_index", "test_index_sha256",
                "selection_seed", "count", "selection_method", "tokens", "records",
                "full_benchmark", "adaptive_evaluation"}
    if (set(value) != required or value["schema_version"] != 1 or
            value["kind"] != "chainflow_locked_navtest_pilot" or
            value["protocol"] != PILOT_PROTOCOL or value["full_benchmark"] is not False or
            value["adaptive_evaluation"] is not True or
            value["selection_method"] != "sha256_seed_token_one_per_original_log" or
            digest(value["test_index"]) != value["test_index_sha256"]):
        raise ValueError("Invalid locked navtest pilot receipt")
    index = read_index(value["test_index"], "navtest")
    expected = select_pilot_rows(index["records"], value["count"], value["selection_seed"])
    if value["records"] != expected or value["tokens"] != [r["token"] for r in expected]:
        raise ValueError("Locked pilot differs from score-blind official membership selection")
    return value


def evaluator_files(dataset):
    """Bind complete official index plus the exact subset when applicable."""
    spec = dataset["benchmark"]
    keys = ("test_index", "test_subset") if dataset["schema_version"] == 3 else ("test_index",)
    return {spec[k]: dataset["files"][spec[k]] for k in keys}


def make_pilot_manifest(train_index, test_index, subset, output):
    result = make_manifest(train_index, test_index, None)
    pilot = read_pilot_subset(subset)
    if pilot["test_index"] != str(Path(test_index).resolve()):
        raise ValueError("Pilot population differs from manifest test index")
    result["schema_version"] = 3
    result["benchmark"].update(protocol=PILOT_PROTOCOL, test_subset=str(Path(subset).resolve()))
    result["splits"]["development"] = pilot["records"]
    result["files"][str(Path(subset).resolve())] = digest(subset)
    validate_manifest(result)
    write_json(output, result)
    return result


def official_filter(split):
    import yaml
    if split not in ("navtrain", "navtest"):
        raise ValueError("Only official navtrain learning and navtest evaluation are supported")
    path = project_root() / ("agents/chainflow/native/navsim/planning/script/config/common/"
                             "train_test_split/scene_filter/" + split + ".yaml")
    value = yaml.safe_load(path.read_text())
    if not value.get("tokens") or not value.get("log_names"):
        raise ValueError("Exact official split identities required")
    if len(value["tokens"]) != len(set(value["tokens"])):
        raise ValueError("Duplicate official token")
    return path, value


def metric_paths(root):
    result = {}
    for path in sorted((Path(root) / "metadata").glob("*.csv")):
        with path.open() as stream:
            for item in csv.DictReader(stream):
                cache = Path(item["file_name"])
                token = cache.parent.name
                if token in result and result[token] != str(cache):
                    raise ValueError("Ambiguous scoring cache: " + token)
                result[token] = str(cache)
    return result


def inventory(split, log_root, sensor_root, cache_root, output):
    """Stream one source log at a time; never filter away missing official tokens."""
    filter_path, spec = official_filter(split)
    expected = set(spec["tokens"])
    caches = metric_paths(cache_root)
    records, failures, seen = [], [], set()
    log_root, sensor_root, output = Path(log_root), Path(sensor_root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_stats = {}
    for number, name in enumerate(sorted(spec["log_names"])):
        path = log_root / (name + ".pkl")
        if not path.is_file():
            failures.append({"source_log": name, "reason": "missing_log"})
            continue
        stat = path.stat()
        source_stats[str(path)] = [stat.st_size, stat.st_mtime_ns]
        with path.open("rb") as stream:
            frames = pickle.load(stream)
        for anchor, frame in enumerate(frames):
            token = frame["token"]
            if token not in expected:
                continue
            if token in seen:
                raise ValueError("Duplicate source anchor: " + token)
            seen.add(token)
            reasons = []
            window = frames[max(0, anchor - 3):anchor + 11]
            if anchor < 3 or len(window) != 14:
                reasons.append("incomplete_four_history_ten_future")
            if spec.get("has_route") and not frame.get("roadblock_ids"):
                reasons.append("missing_route")
            if any(w.get("log_name") != name for w in window):
                reasons.append("source_identity_mismatch")
            images = [str(sensor_root / frame["cams"][camera]["data_path"]) for camera in CAMERAS]
            if any(not Path(p).is_file() for p in images):
                reasons.append("missing_current_camera")
            cache = caches.get(token)
            if not cache or not Path(cache).is_file():
                reasons.append("missing_metric_cache")
            if reasons:
                failures.append({"token": token, "source_log": name, "reasons": reasons})
                continue
            records.append(dict(sample_id=token, token=token, source_log=name,
                                kind="original_" + split, log_path=str(path),
                                sensor_root=str(sensor_root), anchor_index=anchor,
                                metric_cache=cache, training_metric_cache=cache,
                                image_files=images, admission=None))
        if number % 25 == 0:
            write_json(output / "progress.json", dict(split=split, logs=number + 1,
                       expected_logs=len(spec["log_names"]), available=len(records), failures=len(failures)))
            print(split, number + 1, "logs", len(records), "scenes", flush=True)
    failures.extend({"token": t, "reason": "official_token_not_found"} for t in sorted(expected - seen))
    index = dict(schema_version=1, split=split, official_filter=str(filter_path),
                 official_filter_sha256=digest(filter_path), expected_tokens=sorted(expected),
                 source_stats=source_stats, records=records, failures=failures,
                 complete=not failures and {r["token"] for r in records} == expected)
    write_json(output / "index.json", index)
    write_json(output / "coverage.json", dict(split=split, expected=len(expected),
               available=len(records), failed=len(failures), complete=index["complete"],
               source_logs=len(spec["log_names"]), index=str(output / "index.json"),
               files_checked="log identity, full native window, four current images and cache presence",
               content_validation="native consumer validates each actual input when consumed"))
    return index


def read_index(path, split):
    value = json.loads(Path(path).read_text())
    source, spec = official_filter(split)
    if (value.get("split") != split or value.get("complete") is not True
            or value.get("official_filter_sha256") != digest(source)
            or set(value.get("expected_tokens", [])) != set(spec["tokens"])):
        raise ValueError("Incomplete or changed official split inventory")
    records = value["records"]
    allowed_logs = set(spec["log_names"])
    if (len(records) != len(spec["tokens"]) or
            {r["token"] for r in records} != set(spec["tokens"]) or
            any(r["source_log"] not in allowed_logs or
                r["kind"] != "original_" + split for r in records)):
        raise ValueError("Inventory is not exact official membership")
    for name, expected in value["source_stats"].items():
        stat = Path(name).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError("Indexed source log changed: " + name)
    return value


def make_manifest(train_index, test_index, output):
    train, test = read_index(train_index, "navtrain"), read_index(test_index, "navtest")
    a = {source_group(r["source_log"]) for r in train["records"]}
    b = {source_group(r["source_log"]) for r in test["records"]}
    if a & b:
        raise ValueError("Train/test original log overlap")
    result = dict(schema_version=2, kind="chainflow_native_scene_dataset",
                  benchmark=dict(protocol="official_navtest_v1", train_index=str(train_index),
                                 test_index=str(test_index)),
                  splits=dict(train=train["records"], development=test["records"]),
                  files={str(p): digest(p) for p in (train_index, test_index)})
    if output is not None:
        write_json(output, result)
    return result


def validate_manifest(value, groups=None):
    required = {"schema_version", "kind", "benchmark", "splits", "files"}
    if (set(value) != required or value["schema_version"] not in (2, 3) or
            value["kind"] != "chainflow_native_scene_dataset" or
            set(value["splits"]) != {"train", "development"}):
        raise ValueError("Malformed official benchmark dataset")
    spec = value["benchmark"]
    pilot = value["schema_version"] == 3
    keys = {"protocol", "train_index", "test_index"} | ({"test_subset"} if pilot else set())
    if set(spec) != keys or spec["protocol"] != (PILOT_PROTOCOL if pilot else "official_navtest_v1"):
        raise ValueError("Only official navtest benchmark protocol allowed")
    for name, sha in value["files"].items():
        if digest(name) != sha:
            raise ValueError("Benchmark input changed: " + name)
    for name in (("train_index", "test_index", "test_subset") if pilot else ("train_index", "test_index")):
        if spec[name] not in value["files"]:
            raise ValueError("Unsealed benchmark population index")
    train, test = read_index(spec["train_index"], "navtrain"), read_index(spec["test_index"], "navtest")
    expected_test = test["records"]
    if pilot:
        locked = read_pilot_subset(spec["test_subset"])
        if locked["test_index"] != spec["test_index"]:
            raise ValueError("Pilot index differs from sealed official population")
        expected_test = locked["records"]
    real = [r for r in value["splits"]["train"] if r["kind"] == "original_navtrain"]
    if real != train["records"] or value["splits"]["development"] != expected_test:
        raise ValueError("Full navtrain pool and full navtest coverage required")
    allowed_tokens = set(train["expected_tokens"])
    ids = {r["sample_id"] for r in real + test["records"]}
    for row in value["splits"]["train"]:
        if row["kind"] == "original_navtrain":
            continue
        if row["kind"] != "synthetic_navtrain" or row["sample_id"] in ids:
            raise ValueError("Only unique admitted navtrain synthetic additions allowed")
        from driversi_agents.chainflow.synthetic import validate_admission
        proof = validate_admission(row, value["files"])
        member = proof.get("source_train_membership") or proof["parent"]
        if member["token"] not in allowed_tokens:
            raise ValueError("Synthetic origin is outside official navtrain")
        ids.add(row["sample_id"])
    actual = {side: {source_group(r["source_log"]) for r in rows}
              for side, rows in value["splits"].items()}
    if actual["train"] & actual["development"]:
        raise ValueError("Original-log split overlap")
    if groups is not None and any(actual[k] != set(groups[k]) for k in actual):
        raise ValueError("Declared source groups differ")
    return value


def sample_indices(rows, count, seed, synthetic_every=8):
    """All real samples eligible; deterministic shuffled cycles, bounded mixing.

    Synthetic slots replace real draws within an equal update budget. They do
    not change the stored original pool or loss. This is not a full-epoch claim.
    """
    if not rows or count < 0 or synthetic_every < 2:
        raise ValueError("Invalid sampling budget")
    real = [i for i, row in enumerate(rows) if row["kind"] == "original_navtrain"]
    synth = [i for i, row in enumerate(rows) if row["kind"] == "synthetic_navtrain"]
    if not real:
        raise ValueError("Original training population required")
    rng = random.Random(seed)
    order, cursor, synthetic_cursor, result = [], 0, 0, []
    for step in range(count):
        if synth and (step + 1) % synthetic_every == 0:
            result.append(synth[synthetic_cursor % len(synth)])
            synthetic_cursor += 1
        else:
            if cursor == len(order):
                order = list(real)
                rng.shuffle(order)
                cursor = 0
            result.append(order[cursor])
            cursor += 1
    return result


def summarize(rows, expected):
    expected = set(expected)
    tokens = [r["token"] for r in rows]
    valid = [r for r in rows if r.get("valid") is True and
             all(isinstance(r.get(k), Real) and not isinstance(r[k], bool) and
                 math.isfinite(r[k]) and 0 <= r[k] <= 1 for k in METRICS)]
    missing, unexpected = sorted(expected - set(tokens)), sorted(set(tokens) - expected)
    duplicate = len(tokens) - len(set(tokens))
    complete = bool(expected) and not (missing or unexpected or duplicate or len(valid) != len(rows))
    means = {k: float(sum(r[k] for r in valid) / len(valid)) for k in METRICS} if valid else {}
    return dict(expected=len(expected), succeeded=len(valid), failed=len(rows) - len(valid),
                missing_tokens=missing, unexpected_tokens=unexpected, duplicate_rows=duplicate,
                complete=complete, metrics=means, pdms=means.get("score", 0) * 100 if complete else None,
                partial_metrics_not_benchmark=not complete)


def evaluation_population(config, index):
    """Select the pre-registered pilot only through its independently sealed receipt."""
    if config.get('target_token') is not None:
        if config.get('pilot_subset') or config.get('split', 'navtest') != 'navtest':
            raise ValueError('Adaptive target cannot mix with another population selector')
        rows = [r for r in index['records'] if r['token'] == config['target_token']]
        if len(rows) != 1 or config.get('shards', 1) != 1:
            raise ValueError('Exactly one official adaptive target and one shard required')
        return rows
    if not config.get("pilot_subset"):
        return index["records"]
    if config.get("split", "navtest") != "navtest":
        raise ValueError("Locked test pilot cannot select training rows")
    pilot = read_pilot_subset(config["pilot_subset"])
    if (pilot["test_index"] != str(Path(config["index"]).resolve()) or
            pilot["test_index_sha256"] != digest(config["index"])):
        raise ValueError("Pilot source index differs from evaluator")
    known = {row["token"]: row for row in index["records"]}
    if any(known.get(row["token"]) != row for row in pilot["records"]):
        raise ValueError("Pilot rows differ from exact official inventory")
    return pilot["records"]


def evaluation_identity(config):
    """Seal native inference identity for both worker and coordinator resume."""
    root = project_root()
    identity = dict(checkpoint_sha256=digest(config['checkpoint']), index_sha256=digest(config['index']),
                    seed=config.get('seed', 0), shard=config.get('shard', 0), shards=config.get('shards', 1),
                    model_config_sha256=digest(root / 'agents/chainflow/native/navsim/planning/script/config/common/agent/chainflow_vla_stage1.yaml'),
                    backbone_sha256=digest(config['backbone']), precision='32-true', tto=False)
    if config.get('split', 'navtest') == 'navtrain':
        identity.update(split='navtrain', purpose='training_pool_mining',
                        cache_protocol='native_MetricCacheProcessor_if_reference_missing')
    if config.get('pilot_subset'):
        identity.update(pilot_protocol=PILOT_PROTOCOL,
                        pilot_subset_sha256=digest(config['pilot_subset']),
                        full_benchmark=False, adaptive_evaluation=True)
    if config.get('target_token') is not None:
        identity.update(target_token=config['target_token'], full_benchmark=False,
                        adaptive_evaluation=True, selection_protocol='explicit_failure_target')
    return identity


def sample_boundary_stop():
    """Finish the current result before a requested evaluator shutdown."""
    import contextlib
    import signal

    @contextlib.contextmanager
    def boundary():
        requested = {'value': False}
        def request_stop(signum, frame):
            requested['value'] = True
        previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield requested
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return boundary()


def evaluate(config):
    """Shard by exact official token, retaining failures and incremental scores."""
    import gc
    import lzma
    import time
    from dataclasses import asdict

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    root = project_root()
    sys.path[:0] = [str(root / "agents/chainflow/native"), str(root / "src/vendor/driving/nuplan-devkit")]
    os.environ.update(CHAINFLOW_ENABLE_RAY="0", CHAINFLOW_USE_RAY_SCORE="0",
                      NUPLAN_MAPS_ROOT=config["maps_root"])
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat
    install_numpy_2_pickle_compat()
    from navsim.common.dataclasses import AgentInput
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from driversi.training.orchestration.state import deterministic
    from driversi_agents.chainflow.data import _LogFrameCache
    from driversi_agents.chainflow.stage1 import trajectory_from_input
    out = Path(config["output"]).resolve()
    if not out.is_relative_to(root / "exp"):
        raise ValueError("Evaluation outputs belong under exp")
    out.mkdir(parents=True, exist_ok=True)
    split = config.get("split", "navtest")
    index = read_index(config["index"], split)
    shard, shards = config.get("shard", 0), config.get("shards", 1)
    if not 0 <= shard < shards:
        raise ValueError("Invalid evaluation shard")
    rows = evaluation_population(config, index)[shard::shards]
    identity = evaluation_identity(config)
    ident = out / "identity.json"
    if ident.exists() and json.loads(ident.read_text()) != identity:
        raise ValueError("Resume identity differs")
    write_json(ident, identity)
    path = out / "scores.jsonl"
    results = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    prior = summarize(results, [r["token"] for r in rows])
    if prior["unexpected_tokens"] or prior["duplicate_rows"]:
        raise ValueError("Corrupt evaluation resume coverage")
    done = {r["token"] for r in results}
    deterministic(identity["seed"])
    fraction = config.get("gpu_memory_fraction", 0.12)
    if not 0 < fraction <= 0.25:
        raise ValueError("Bounded GPU memory allocation required")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    cfg = OmegaConf.load(root / "agents/chainflow/native/navsim/planning/script/config/common/agent/chainflow_vla_stage1.yaml")
    cfg.batch_size, cfg.num_gpus, cfg.enable_ray, cfg.progress_bar = 1, 1, False, False
    cfg.config.image_backbone.model_weights = config["backbone"]
    cfg.config.lidar_backbone.model_weights = config["backbone"]
    cfg.checkpoint_path, cfg.checkpoint_strict_load = config["checkpoint"], True
    OmegaConf.save(cfg, out / "native-config.yaml")
    agent = instantiate(cfg)
    agent.initialize()
    agent.to("cuda").eval()
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    frames = _LogFrameCache(1)
    scenes = None
    if split == "navtrain":
        from driversi_agents.chainflow.data import NativeScenes
        from driversi_agents.chainflow.mining import evaluation_cache
        scenes = NativeScenes(rows, agent, native_original_windows=True)
        scenes.evaluation_cache_dir = out / "evaluation-cache"
    maximum = config.get("max_new_samples")
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        raise ValueError("Positive optional per-invocation sample cap required")
    newly_recorded = 0
    started = time.monotonic()
    profiling = config.get('profile_phases', False)
    with sample_boundary_stop() as stop, path.open("a", buffering=1) as stream:
        for row_index, row in enumerate(rows):
            if stop['value']:
                break
            if row["token"] in done:
                continue
            result = dict(token=row["token"], log_name=row["source_log"], valid=False)
            timing = {}
            phase = time.monotonic()
            try:
                window = frames.read(row["log_path"])[row["anchor_index"] - 3:row["anchor_index"] + 1]
                if len(window) != 4 or window[-1]["token"] != row["token"]:
                    raise ValueError("Evaluation anchor mismatch")
                agent_input = AgentInput.from_scene_dict_list(window, Path(row["sensor_root"]), 4, agent.get_sensor_config())
                timing['input_read'] = time.monotonic() - phase
                phase = time.monotonic()
                trajectory = trajectory_from_input(agent, agent_input)
                # This adapter returns host numpy poses, so inference timing
                # already includes completion of the device-to-host transfer.
                timing['native_inference'] = time.monotonic() - phase
                phase = time.monotonic()
                if scenes is None:
                    with lzma.open(row["metric_cache"], "rb") as source:
                        cache = pickle.load(source)
                else:
                    cache, evidence = evaluation_cache(scenes, row_index, row)
                    result['evaluation_cache'] = evidence
                timing['evaluation_cache'] = time.monotonic() - phase
                phase = time.monotonic()
                if int(cache.ego_state.time_point.time_us) != int(window[-1]["timestamp"]):
                    raise ValueError("Evaluation cache timestamp differs")
                score = pdm_score(cache, trajectory, sampling, PDMSimulator(sampling), PDMScorer(sampling))
                result.update(asdict(score), valid=True, poses=trajectory.poses.tolist())
                timing['native_scoring'] = time.monotonic() - phase
                if any(not math.isfinite(result[k]) or not 0 <= result[k] <= 1 for k in METRICS):
                    raise ValueError("Invalid PDM metric")
            except Exception as exc:
                result.update(valid=False, error=type(exc).__name__ + ": " + str(exc))
                result = {k: v for k, v in result.items() if k not in METRICS}
            if profiling:
                result['timing_seconds'] = timing
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            results.append(result)
            newly_recorded += 1
            if len(results) % 10 == 0:
                write_json(out / "progress.json", dict(expected=len(rows), recorded=len(results),
                           failed=sum(not r["valid"] for r in results), wall_seconds=time.monotonic() - started))
                print(split, shard, len(results), "/", len(rows), flush=True)
            if maximum is not None and newly_recorded >= maximum:
                break
    write_json(out / "progress.json", dict(expected=len(rows), recorded=len(results),
               failed=sum(not r["valid"] for r in results), wall_seconds=time.monotonic() - started))
    summary = summarize(results, [r["token"] for r in rows])
    summary.update(identity=identity, metric="NAVSIM_v1_PDMS", scope=(
        "adaptive_single_failure_target_not_full_benchmark" if config.get('target_token') else
        "official_navtest_locked_subset_pilot_not_full_benchmark" if config.get("pilot_subset") else
        "official_navtest_shard" if split == "navtest" else "full_navtrain_mining_shard"),
                   inference_inputs="four_real_history_frames_no_future_targets", promoted=False)
    if config.get("pilot_subset") or config.get('target_token'):
        summary.update(full_benchmark=False, official_population=len(index["records"]),
                       subset_complete=summary["complete"], adaptive_evaluation=True)
    write_json(out / "summary.json", summary)
    del agent
    gc.collect()
    return summary


def merge(config):
    split = config.get("split", "navtest")
    index = read_index(config["index"], split)
    rows, identities = [], []
    for folder in config["shard_outputs"]:
        folder = Path(folder)
        identity = json.loads((folder / "identity.json").read_text())
        identities.append(identity)
        rows.extend(json.loads(line) for line in (folder / "scores.jsonl").read_text().splitlines())
    if any("pilot_protocol" in identity or 'target_token' in identity for identity in identities):
        raise ValueError("Locked pilot results cannot enter formal full-benchmark merge")
    common = [{k: v for k, v in x.items() if k != "shard"} for x in identities]
    if not common or any(x != common[0] for x in common):
        raise ValueError("Evaluation shard identities differ")
    if (len(identities) != common[0]["shards"] or
            {x["shard"] for x in identities} != set(range(len(identities))) or
            common[0]["index_sha256"] != digest(config["index"])):
        raise ValueError("Incomplete shard ownership")
    result = summarize(rows, index["expected_tokens"])
    result.update(metric="NAVSIM_v1_PDMS", evaluation_split=split,
                  identity=common[0],
                  scope="official_navtest" if split == "navtest" else "training_pool_mining_not_benchmark",
                  adaptive_evaluation=True, training_data_source="navtrain_only", promoted=False)
    if split == "navtest":
        result.update(official_reference_pdms=94.05,
                      difference_from_released_pdms=(result["pdms"] - 94.05 if result["complete"] else None))
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per-scene.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["token", "log_name", "valid", *METRICS, "error"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "evaluate", "merge"))
    parser.add_argument("config")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = inventory(**config) if args.command == "inventory" else evaluate(config) if args.command == "evaluate" else merge(config)
    if not result.get("complete"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
