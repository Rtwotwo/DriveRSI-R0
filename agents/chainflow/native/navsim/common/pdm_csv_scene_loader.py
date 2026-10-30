"""Opt-in, read-only PDM scene loading from existing metric-cache metadata.

The CSV contains real anchor tokens and log names, even when its absolute paths
refer to an old mount. Only the CSV is read at construction time (plus the split
manifest and directory names); raw pickle contents are read on sample access.
This is deliberately separate from SceneLoader: real datasets and validation
keep their existing behavior, and no feature/target cache is created here.
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
import re
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import yaml

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat


install_numpy_2_pickle_compat()
logger = logging.getLogger(__name__)


class PdmCsvSceneLoader:
    """Read existing PDM 0--4 CSV mappings without retaining raw scene frames.

    The existing metric CSVs certify 4-history/8-future, routed PDM windows.
    They do not certify camera availability: the existing invalid manifest is
    applied separately and the selected window is checked again when read.
    There is intentionally no fallback to scanning/retaining every pickle.
    """

    def __init__(
        self,
        data_path: Path,
        sensor_blobs_path: Path,
        scene_filter: SceneFilter,
        sensor_config: Optional[SensorConfig] = None,
        metric_cache_path: Optional[Path] = None,
        metadata_csv: Optional[Path] = None,
        load_image_path: bool = False,
    ):
        self._data_path = Path(data_path)
        self._sensor_blobs_path = Path(sensor_blobs_path)
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config if sensor_config is not None else SensorConfig.build_no_sensors()
        self.load_image_path = load_image_path
        match = re.fullmatch(r"synthetic_reaction_pdm_v1\.0-([0-4])", self._data_path.name)
        if match is None:
            raise ValueError(f"metric_csv only supports PDM v1.0 shards 0--4: {self._data_path}")
        self._shard_suffix = f"{int(match.group(1)):03d}"
        if self._sensor_blobs_path.name != self._data_path.name:
            raise ValueError("PDM log and sensor directories must name the same shard")
        if not self._data_path.is_dir() or not self._sensor_blobs_path.is_dir():
            raise FileNotFoundError(f"Missing PDM log/sensor directory: {self._data_path}, {self._sensor_blobs_path}")
        if (
            scene_filter.num_history_frames != 4
            or scene_filter.num_future_frames != 8
            or scene_filter.frame_interval != 1
            or not scene_filter.has_route
            or scene_filter.max_scenes is not None
        ):
            raise ValueError("metric_csv requires the certified PDM filter: history=4, future=8, interval=1, has_route=true, max_scenes=null")

        cache_name = "train_metric_" + self._data_path.name
        if metric_cache_path is None:
            raise ValueError("metric_cache_path is required for PDM CSV loading")
        metric_cache_path = Path(metric_cache_path)
        if metric_cache_path.name != cache_name:
            raise ValueError(f"PDM metric cache directory must be named {cache_name}")
        self.metadata_csv = (
            Path(metadata_csv) if metadata_csv is not None
            else metric_cache_path / "metadata" / f"{cache_name}_metadata_node_0.csv"
        )

        invalid_path = self._sensor_blobs_path / "invalid_tokens.yaml"
        self._invalid_tokens = set()
        if invalid_path.is_file():
            with invalid_path.open() as stream:
                invalid_document = yaml.safe_load(stream)
            invalid_tokens = invalid_document.get("invalid_tokens") if isinstance(invalid_document, dict) else None
            if not isinstance(invalid_tokens, list) or not all(isinstance(item, str) for item in invalid_tokens):
                raise ValueError(f"Invalid PDM invalid-token manifest: {invalid_path}")
            self._invalid_tokens = set(invalid_tokens)

        allowed_logs = set(scene_filter.log_names) if scene_filter.log_names is not None else None
        allowed_tokens = set(scene_filter.tokens) if scene_filter.tokens is not None else None
        self.token_to_log_file: Dict[str, str] = {}
        self._sensor_scene_ids: Dict[str, str] = {}
        excluded_invalid = 0
        with self.metadata_csv.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["file_name"]:
                raise ValueError(f"Expected a single file_name column in {self.metadata_csv}")
            for line_number, row in enumerate(reader, start=2):
                if None in row or not row.get("file_name"):
                    raise ValueError(f"Malformed CSV row: {self.metadata_csv}:{line_number}")
                raw_parts = row["file_name"].split("/")
                if "\\" in row["file_name"] or any(part in {".", ".."} for part in raw_parts):
                    raise ValueError(f"Unsafe metadata path: {self.metadata_csv}:{line_number}")
                parts = PurePosixPath(row["file_name"]).parts
                if parts.count(cache_name) != 1:
                    raise ValueError(f"Wrong PDM cache source: {self.metadata_csv}:{line_number}")
                relative = parts[parts.index(cache_name) + 1:]
                if len(relative) != 4 or relative[-1] != "metric_cache.pkl":
                    raise ValueError(f"Unexpected metric path layout: {self.metadata_csv}:{line_number}")
                log_name, _scenario_type, token, _filename = relative
                if allowed_logs is not None and log_name not in allowed_logs:
                    continue
                if allowed_tokens is not None and token not in allowed_tokens:
                    continue
                sensor_match = re.fullmatch(r".+-([0-9a-f]{16}-" + self._shard_suffix + r")", log_name)
                if sensor_match is None or not token.endswith("-" + self._shard_suffix):
                    raise ValueError(f"Invalid PDM shard/log identity: {self.metadata_csv}:{line_number}")
                sensor_scene_id = sensor_match.group(1)
                if sensor_scene_id in self._invalid_tokens:
                    excluded_invalid += 1
                    continue
                if token in self.token_to_log_file:
                    raise ValueError(f"Duplicate PDM token {token} in {self.metadata_csv}")
                self.token_to_log_file[token] = log_name
                self._sensor_scene_ids[token] = sensor_scene_id

        # One directory-name pass, not hundreds of thousands of NAS stat calls.
        # Do not inspect pickle contents here. The actual open checks readability.
        with os.scandir(self._data_path) as entries:
            filenames = {entry.name for entry in entries if entry.name.endswith(".pkl")}
        missing = sorted({name + ".pkl" for name in self.token_to_log_file.values()} - filenames)
        if missing:
            raise FileNotFoundError(f"{len(missing)} PDM log files referenced by {self.metadata_csv} are missing; first: {missing[0]}")
        # Fixed order on every rank, O(1) indexing, no per-sample list rebuilding.
        self._tokens: Tuple[str, ...] = tuple(sorted(self.token_to_log_file))
        logger.info(
            "PDM CSV index: source=%s samples=%d excluded_invalid=%d metadata=%s; raw pickles loaded=0",
            self._data_path.name, len(self._tokens), excluded_invalid, self.metadata_csv,
        )

    @property
    def tokens(self) -> Tuple[str, ...]:
        return self._tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> str:
        return self._tokens[idx]

    def _get_scene_frames(self, token: str) -> List[dict]:
        log_name = self.token_to_log_file[token]
        path = self._data_path / (log_name + ".pkl")
        try:
            with path.open("rb") as stream:
                frames = pickle.load(stream)
            if not isinstance(frames, list):
                raise ValueError("PDM pickle must contain a frame list")
            history = self._scene_filter.num_history_frames
            num_frames = history + self._scene_filter.num_future_frames
            anchors = [i for i, frame in enumerate(frames) if frame.get("token") == token]
            if len(anchors) != 1:
                raise ValueError(f"Expected one anchor for token, found {len(anchors)}")
            start = anchors[0] - history + 1
            if start < 0 or start + num_frames > len(frames):
                raise ValueError("PDM token has no complete history/future window")
            window = frames[start:start + num_frames]
            anchor = window[history - 1]
            if self._scene_filter.has_route and not len(anchor["roadblock_ids"]):
                raise ValueError("PDM token no longer passes the route filter")
            if window[0]["log_name"] != log_name:
                raise ValueError("PDM raw log_name does not match CSV mapping")
            camera_path = Path(anchor["cams"]["CAM_R0"]["data_path"])
            if camera_path.is_absolute() or ".." in camera_path.parts:
                raise ValueError("PDM camera path must be relative to the configured sensor root")
            image_path = self._sensor_blobs_path / camera_path
            sensor_scene_id = image_path.parent.parent.name
            if sensor_scene_id != self._sensor_scene_ids[token] or sensor_scene_id in self._invalid_tokens:
                raise ValueError("PDM camera identity differs from the CSV/invalid-token selection")
            return window
        except Exception as exc:
            raise RuntimeError(f"PDM sample read failed: source={self._data_path.name} token={token} path={path}: {exc}") from exc

    def get_scene_from_token(self, token: str) -> Scene:
        return Scene.from_scene_dict_list(
            self._get_scene_frames(token), self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config, load_image_path=self.load_image_path,
        )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        return AgentInput.from_scene_dict_list(
            self._get_scene_frames(token), self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            sensor_config=self._sensor_config, load_image_path=self.load_image_path,
        )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for token in self._tokens:
            result.setdefault(self.token_to_log_file[token], []).append(token)
        return result
