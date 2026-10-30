"""Finite native Scene inputs with provenance and strict synthetic admission.

The adapter changes transport only. Native builders own images and targets.
"""

import json
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from driversi.agents.backends.native import digest, project_root


def source_group(name):
    name = Path(name).name
    if name.endswith(".pkl"):
        name = name[:-4]
    return re.sub(r"_\d+_\d+$", "", name)


@lru_cache(maxsize=2)
def _parse_official_train_split(content):
    import yaml

    # Keyed by exact current bytes, never by path or potentially stale timestamps.
    return yaml.safe_load(content)


def _official_train_split():
    source = (
        project_root()
        / "agents/chainflow/native/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    )
    return _parse_official_train_split(source.read_bytes())


def official_train_logs():
    names = _official_train_split().get("log_names")
    if not isinstance(names, list) or not names:
        raise ValueError("Official navtrain source allowlist unavailable")
    return set(names)


def official_train_tokens():
    """The official split filters anchors as well as whole source logs."""
    tokens = _official_train_split().get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("Exact official navtrain token allowlist unavailable")
    return set(tokens)


def validate_manifest(path, groups=None):
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") in (2, 3):
        from driversi_agents.chainflow.benchmark import validate_manifest as validate_benchmark

        return validate_benchmark(value, groups)
    if (
        set(value) != {"schema_version", "kind", "splits", "files"}
        or value["schema_version"] != 1
        or value["kind"] != "chainflow_native_scene_dataset"
        or set(value["splits"]) != {"train", "development"}
    ):
        raise ValueError("Malformed ChainFlow native scene manifest")
    files, seen, seen_tokens, actual = value["files"], set(), set(), {}
    allowed = official_train_logs()
    allowed_tokens = official_train_tokens()
    if not files:
        raise ValueError("Native dataset needs sealed input files")
    for name, sha in files.items():
        if not Path(name).is_absolute() or digest(name) != sha:
            raise ValueError("Native input changed: " + name)
    for side, rows in value["splits"].items():
        if not isinstance(rows, list) or not rows:
            raise ValueError("Both train and heldout scenes are required")
        actual[side] = set()
        for row in rows:
            if set(row) != {
                "sample_id",
                "source_log",
                "kind",
                "log_path",
                "sensor_root",
                "anchor_index",
                "token",
                "metric_cache",
                "training_metric_cache",
                "image_files",
                "admission",
            }:
                raise ValueError("Incomplete native scene record")
            if row["source_log"] not in allowed:
                raise ValueError("Source is not in the fixed official navtrain allowlist")
            if row["kind"] == "original_navtrain" and row["token"] not in allowed_tokens:
                raise ValueError(
                    "Token is not in the exact official navtrain allowlist: " + str(row["token"])
                )
            if row["sample_id"] in seen or not row["sample_id"] or not row["source_log"]:
                raise ValueError("Duplicate or empty scene identity")
            if row["token"] in seen_tokens:
                raise ValueError("Duplicate native token across scene identities or splits")
            seen.add(row["sample_id"])
            seen_tokens.add(row["token"])
            if type(row["anchor_index"]) is not int or row["anchor_index"] < 3:
                raise ValueError("Four continuous history frames are required")
            required = [row[k] for k in ("log_path", "metric_cache", "training_metric_cache")]
            if len(row["image_files"]) != 4 or len(set(row["image_files"])) != 4:
                raise ValueError("Official stage1 consumes four current cameras")
            required += row["image_files"]
            if any(p not in files for p in required):
                raise ValueError("Scene source/cache/images not sealed")
            if row["kind"] == "original_navtrain":
                if row["admission"] is not None:
                    raise ValueError("Original sources cannot impersonate synthetic admission")
            elif row["kind"] == "synthetic_navtrain":
                if side != "train":
                    raise ValueError("Synthetic scenes cannot enter independent development")
                from driversi_agents.chainflow.synthetic import validate_admission

                proof = validate_admission(row, files)
                membership = proof.get("source_train_membership")
                member_token = (
                    membership["token"] if membership is not None else proof["parent"]["token"]
                )
                if member_token not in allowed_tokens:
                    raise ValueError(
                        "Synthetic source membership is not an exact official navtrain token"
                    )
            else:
                raise ValueError("Unsupported source kind")
            actual[side].add(source_group(row["source_log"]))
    if actual["train"] & actual["development"]:
        raise ValueError("Source-log overlap across learning and development")
    if groups is not None and any(actual[k] != set(groups[k]) for k in actual):
        raise ValueError("Native records differ from declared source logs")
    return value


def scoring_cache_paths(rows):
    """Keep new branch tokens in the upstream model's synthetic cache namespace."""
    original, synthetic = {}, {}
    seen = set()
    for row in rows:
        token = row["token"]
        if token in seen:
            raise ValueError("Duplicate scoring tokens")
        seen.add(token)
        if row["kind"] == "original_navtrain":
            original[token] = Path(row["training_metric_cache"])
        elif row["kind"] == "synthetic_navtrain":
            if not token.startswith("synthetic_"):
                raise ValueError("Synthetic cache requires distinct native token")
            synthetic[token] = Path(row["training_metric_cache"])
        else:
            raise ValueError("Unsupported native scoring source")
    return original, synthetic


class _LogFrameCache(OrderedDict):
    """LRU of whole native pickles; evict before opening the next source log.

    The bound is per dataset/worker and counts logs, not bytes. Native pickle
    loading still requires enough memory for one complete source log.
    """

    def __init__(self, max_cached_logs=1):
        super().__init__()
        if type(max_cached_logs) is not int or not 1 <= max_cached_logs <= 16:
            raise ValueError("max_cached_logs must be an integer in [1,16]")
        self.max_cached_logs = max_cached_logs

    def read(self, path):
        import pickle

        path = str(Path(path).resolve())
        if path in self:
            self.move_to_end(path)
            return self[path]
        while len(self) >= self.max_cached_logs:
            self.popitem(last=False)
        with Path(path).open("rb") as stream:
            self[path] = pickle.load(stream)
        return self[path]


class NativeScenes:
    """Native features/targets with a bounded source-log LRU (one log by default).

    Four history plus ten actual future frames are required for the released
    stage1 dual-horizon targets. Eight main-trajectory future poses alone are
    insufficient; history/future frames are never padded or fabricated.
    """

    def __init__(self, rows, agent, *, max_cached_logs=1, native_original_windows=False):
        self.rows, self.agent = rows, agent
        self._frames = _LogFrameCache(max_cached_logs)
        self.native_original_windows = native_original_windows
        self.original_time_gaps = {}

    def __len__(self):
        return len(self.rows)

    def scene(self, index, *, sensors=True):
        from navsim.common.dataclasses import Scene, SensorConfig

        row = self.rows[index]
        path = row["log_path"]
        frames, anchor = self._frames.read(path), row["anchor_index"]
        window = frames[anchor - 3 : anchor + 11]
        if len(window) != 14 or window[3]["token"] != row["token"]:
            raise ValueError("Missing 4 history + 10 future physical context or wrong anchor")
        validate_consumption(row, window)
        times = [f["timestamp"] for f in window]
        continuous = validate_window_times(row, times, self.native_original_windows)
        if not continuous:
            self.original_time_gaps[row["token"]] = [int(b - a) for a, b in zip(times, times[1:])]
        cameras = self.agent.get_sensor_config() if sensors else SensorConfig.build_no_sensors()
        return Scene.from_scene_dict_list(window, Path(row["sensor_root"]), 4, 10, cameras)

    def __getitem__(self, index):
        scene = self.scene(index)
        features, targets = {}, {}
        for builder in self.agent.get_feature_builders():
            features.update(builder.compute_features(scene.get_agent_input()))
        targets.update(validated_native_targets(scene, self.agent))
        if targets["token"] != self.rows[index]["token"]:
            raise ValueError("Native target token differs from its scoring cache")
        return features, targets


def validate_consumption(row, window, *, verify_reference=False):
    """Verify actual raw frames/cameras/caches, not just declared manifest paths."""
    import lzma
    import pickle

    synthetic = row.get("kind") == "synthetic_navtrain"
    proof = physical = None
    if synthetic:
        from driversi_agents.chainflow.synthetic import validate_admission, validate_window

        proof = validate_admission(row)
        physical = validate_window(row, window, proof)
        native_log = Path(row["log_path"]).stem
    else:
        native_log = row["source_log"]
    if any(frame["log_name"] != native_log for frame in window):
        raise ValueError("Actual source log differs from declaration")
    if Path(row["log_path"]).name != native_log + ".pkl":
        raise ValueError("Original source filename differs from raw log identity")
    anchor = window[3]
    root = Path(row["sensor_root"]).resolve(strict=True)
    actual_images = []
    for camera in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
        image = (root / anchor["cams"][camera]["data_path"]).resolve(strict=True)
        if not image.is_relative_to(root):
            raise ValueError("Native camera path escapes sensor root")
        actual_images.append(str(image))
    if actual_images != row["image_files"]:
        raise ValueError("Actual consumed images differ from sealed camera inputs")
    caches = {}
    for key in ("metric_cache", "training_metric_cache"):
        with lzma.open(row[key], "rb") as stream:
            cache = pickle.load(stream)
        caches[key] = cache
        if (
            int(cache.ego_state.time_point.time_us) != int(anchor["timestamp"])
            or Path(cache.file_path).parents[2].name != native_log
            or Path(cache.file_path).parent.name != row["token"]
            or row["token"] != anchor["token"]
        ):
            raise ValueError("Scoring cache belongs to another anchor")
        if (
            key == "training_metric_cache"
            and row.get("kind") != "original_navtest"
            and not hasattr(cache, "pdm_progress")
        ):
            raise ValueError("Missing native proposal supervision context")
        if synthetic:
            from driversi_agents.chainflow.synthetic import validate_cache_context

            validate_cache_context(
                row, cache, proof, *physical, training=key == "training_metric_cache"
            )
    if synthetic and verify_reference:
        from driversi_agents.chainflow.synthetic import validate_cache_pair

        validate_cache_pair(caches["metric_cache"], caches["training_metric_cache"], proof)


def validate_window_times(row, times, native_original_windows=False):
    gaps = [int(b - a) for a, b in zip(times, times[1:])]
    if any(gap <= 0 for gap in gaps):
        raise ValueError("Discontinuous/non-increasing NAVSIM timestamps")
    continuous = all(450000 <= gap <= 550000 for gap in gaps)
    original = row.get("kind") in ("original_navtrain", "original_navtest")
    if not continuous and not (native_original_windows and original):
        raise ValueError("Discontinuous NAVSIM history/future")
    return continuous


def verify_records(rows, *, max_cached_logs=1, native_original_windows=False):
    """Verify sealed sources before GPU use, retaining at most a bounded log LRU.

    Preserve row order and require four history plus ten real future frames for
    both native target horizons; an eight-future-frame source is never padded.
    """
    loaded = _LogFrameCache(max_cached_logs)
    scoring_steps = 0
    for row in rows:
        anchor = row["anchor_index"]
        window = loaded.read(row["log_path"])[anchor - 3 : anchor + 11]
        if len(window) != 14:
            raise ValueError("Complete native 4+10 source window required")
        if row.get("kind") == "synthetic_navtrain":
            validate_consumption(row, window, verify_reference=True)
            scoring_steps += 40
        else:
            validate_consumption(row, window)
        times = [int(f["timestamp"]) for f in window]
        validate_window_times(row, times, native_original_windows)
    return scoring_steps


def validated_native_targets(scene, agent):
    """Reject native target-builder fallback; never silently shorten supervision."""
    import numpy as np
    from scipy.interpolate import CubicSpline

    raw = np.asarray(scene.get_future_trajectory(num_trajectory_frames=10).poses)
    short = np.asarray(scene.get_future_trajectory(num_trajectory_frames=8).poses)
    if (
        raw.shape != (10, 3)
        or short.shape != (8, 3)
        or not np.isfinite(raw).all()
        or not np.isfinite(short).all()
    ):
        raise ValueError("Finite complete ten-pose native future supervision required")
    # Validate the released builder's 10->8 stretched target, not a new target.
    x = np.arange(10, dtype=np.float32)
    query = np.arange(8, dtype=np.float32)
    query += np.cumsum((query + 1) * (2 * 2 / (8 * 9)))
    expected = np.stack([CubicSpline(x, raw[:, i])(query) for i in range(3)], axis=1)
    targets = {}
    for builder in agent.get_target_builders():
        targets.update(builder.compute_targets(scene))
    actual = targets.get("trajectory_long")
    if hasattr(actual, "detach"):
        actual = actual.detach().cpu().numpy()
    if (
        actual is None
        or np.asarray(actual).shape != (8, 3)
        or not np.array_equal(expected, np.asarray(actual))
    ):
        raise ValueError("Native long trajectory target fell back or changed")
    return targets
