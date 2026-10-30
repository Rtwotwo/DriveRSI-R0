"""World-driven MTGS display sessions, explicitly separate from visual training.

The authoritative world runs in its supported Python/native process. Images
apply reference-relative planar motion to the matched reconstruction, whose
refined geometry is not identical to raw NAVSIM collision geometry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from driversi.contracts.scene import VEHICLE, Scene
from driversi.rendering.gaussian.mtgs import _array, _sha256, _write_manifest
from driversi.rendering.gaussian.mtgs_artifacts import (
    CaptureArtifactWriter,
    validate_image_write_workers,
)
from driversi.rendering.gaussian.mtgs_height import ReferenceTrackHeight
from driversi.rendering.gaussian.mtgs_multiview import (
    novel_view_support,
    roof_front_camera,
    validate_ego_camera_owner,
    virtual_camera_owners,
)
from driversi.sim.workers.worker_client import WorldWorker


def planar_delta(current, reference) -> tuple[np.ndarray, float]:
    current, reference = np.asarray(current), np.asarray(reference)
    if (
        current.shape != (4,)
        or reference.shape != (4,)
        or not np.isfinite([current, reference]).all()
    ):
        raise ValueError("World/reference poses must be finite XYZ/yaw vectors")
    yaw = float(np.arctan2(np.sin(current[3] - reference[3]), np.cos(current[3] - reference[3])))
    return current[:2] - reference[:2], yaw


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def camera_metadata_equal(left, right) -> bool:
    """Metadata includes tensor/array values whose dict equality is ambiguous."""
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(camera_metadata_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(camera_metadata_equal(a, b) for a, b in zip(left, right))
        )
    return bool(np.array_equal(_array(left), _array(right)))


def apply_rigid_delta(xyz, wxyz, delta_xy, delta_yaw) -> tuple[np.ndarray, np.ndarray]:
    """Left-compose map-axis yaw; retain the reconstruction's height and tilt."""
    xyz, q = np.asarray(xyz, dtype=float), np.asarray(wxyz, dtype=float)
    delta_xy = np.asarray(delta_xy, dtype=float)
    if (
        xyz.shape != (3,)
        or q.shape != (4,)
        or delta_xy.shape != (2,)
        or not np.isfinite([*xyz, *q, *delta_xy, delta_yaw]).all()
        or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)
    ):
        raise ValueError("Invalid rigid delta")
    c, s = np.cos(delta_yaw / 2), np.sin(delta_yaw / 2)
    w, x, y, z = q
    return xyz + [*delta_xy, 0], np.asarray(
        [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
    )


def apply_camera_delta(pose, pivot, delta_xy, delta_yaw) -> np.ndarray:
    pose = np.array(pose, dtype=float, copy=True)
    pivot, delta_xy = np.asarray(pivot), np.asarray(delta_xy)
    if (
        pose.shape != (3, 4)
        or pivot.shape != (3,)
        or delta_xy.shape != (2,)
        or not np.isfinite([*pose.ravel(), *pivot, *delta_xy, delta_yaw]).all()
    ):
        raise ValueError("Invalid camera delta")
    rotation = yaw_matrix(delta_yaw)
    pose[:, :3] = rotation @ pose[:, :3]
    pose[:, 3] = pivot + rotation @ (pose[:, 3] - pivot) + [*delta_xy, 0]
    return pose


def render_moved_source_camera(
    runtime,
    camera,
    pivot,
    delta_xy,
    delta_yaw,
    height_correction,
    rigid_poses,
    *,
    exclude_tokens=(),
    rigid_source_frames=None,
    source_registration=None,
):
    """Move the effective calibrated camera in reconstruction/world axes.

    Nerfstudio's correction is SO3xR3: R_out=A@R_in, t_out=t_in+b.
    A world yaw and this correction do not commute. The runtime's target
    inversion compensates the frozen correction without mutating its weights.
    Pure translation does commute, so retain that exact source rendering path.
    The returned support poses use the same coordinate semantics as each other.
    """
    if not np.isfinite(height_correction):
        raise ValueError("Camera height correction must be finite")
    binding_args = {"rigid_source_frames": rigid_source_frames} if rigid_source_frames else {}
    source_pose = _array(camera.camera_to_worlds)[0].copy()
    if source_registration is not None:
        from driversi.rendering.gaussian.mtgs_registration import move_camera

        def move(pose):
            return move_camera(source_registration, pose, pivot, delta_xy, delta_yaw)
    else:

        def move(pose):
            return apply_camera_delta(pose, pivot, delta_xy, delta_yaw)

    moved = move(source_pose)
    moved[2, 3] += height_correction
    if delta_yaw == 0:
        translated = copy.deepcopy(camera)
        translated.camera_to_worlds = camera.camera_to_worlds.new_tensor(moved[None])
        result = runtime.render_camera(
            translated,
            rigid_poses=rigid_poses,
            **binding_args,
            **({"exclude_tokens": exclude_tokens} if exclude_tokens else {}),
        )
        return result, moved, source_pose
    # Do not move an uncorrected pose and then rotate its correction in the
    # wrong order. This pure query leaves the source camera and model untouched.
    optimized_source = _array(runtime.model.camera_optimizer.apply_to_camera(camera))[0]
    target = move(optimized_source)
    target[2, 3] += height_correction
    result = runtime.render_camera(
        camera,
        rigid_poses=rigid_poses,
        target_camera_to_world=target,
        **binding_args,
        **({"exclude_tokens": exclude_tokens} if exclude_tokens else {}),
    )
    return result, target, optimized_source


class MTGSWorldSession:
    """Serial session, with one authoritative tick and immutable saved frames."""

    def __init__(
        self,
        *,
        runtime,
        scene: Path,
        road_graph: Path,
        binding: Path,
        world_python: Path,
        native_path: Path,
        output: Path,
        cameras: tuple[str, ...] = ("CAM_F0", "CAM_R0"),
        sensor_stride: int = 2,
        desired_speed: float = 13.0,
        virtual_camera_actors: tuple[str, ...] = (),
        height_mode: str = "source-time",
        population_config: Path | None = None,
        visual_policy: bool = False,
        visual_actors: tuple[str, ...] = (),
        visual_history: int = 2,
        visual_adapter=None,
        registration_mode="legacy-map-axes",
        require_complete_assets=False,
        image_write_workers: int = 1,
        recorded_manifest: Path | None = None,
    ):
        if recorded_manifest is not None and (
            visual_policy or visual_actors or visual_adapter is not None
        ):
            raise ValueError("Recorded worlds cannot run new visual policies")
        self.recorded_mode = recorded_manifest is not None
        self.image_write_workers = validate_image_write_workers(image_write_workers)
        if type(visual_policy) is not bool or (visual_actors and not visual_policy):
            raise ValueError("Visual actors require an explicit visual_policy switch")
        if type(visual_history) is not int or not 1 <= visual_history <= 16:
            raise ValueError("visual_history must be an integer in [1,16]")
        if visual_adapter is not None and not visual_policy:
            raise ValueError("visual_adapter requires visual_policy")
        self.visual_mode, self.visual_sampler = visual_policy, None
        self._sensor_buffers = {}
        self.session_id = uuid.uuid4().hex
        if height_mode not in ("source-time", "reference-track"):
            raise ValueError("Unknown reconstruction height mode")
        if registration_mode not in ("legacy-map-axes", "source-frame"):
            raise ValueError("Unknown registration mode")
        if type(require_complete_assets) is not bool:
            raise ValueError("Asset requirement must be explicit boolean")
        self.registration_mode = registration_mode
        self.height_mode = "source-frame" if registration_mode == "source-frame" else height_mode
        if type(sensor_stride) is not int or not 1 <= sensor_stride <= 10:
            raise ValueError("sensor_stride must be an integer in [1,10]")
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("Camera IDs must be nonempty and unique")
        if (
            any(not isinstance(name, str) or not name for name in cameras)
            or not np.isfinite(desired_speed)
            or desired_speed <= 0
        ):
            raise ValueError("Camera IDs and desired speed must be valid")
        self.runtime, self.output = runtime, Path(output).resolve()
        if self.output.exists():
            raise ValueError("Session output must be new")
        self.binding = json.loads(Path(binding).read_text())
        source_scene = Scene.load(scene)
        self.metadata = source_scene.metadata
        if (
            self.metadata.get("coordinate_frame") != "map_axes_translated_from_initial_ego"
            or self.metadata.get("pose_reference") != "vehicle_center"
        ):
            raise ValueError("Matched rendering requires map-axis vehicle-centre scene poses")
        self.lifecycle_presence = (
            np.asarray(source_scene.metadata["reactive_background"]["source_presence"], bool)
            if "reactive_background" in source_scene.metadata
            else source_scene.presence
        )
        self.ids = source_scene.actor_ids
        self.reference, self.times = source_scene.poses, source_scene.times
        self.controlled = source_scene.controlled
        self.dimensions, self.kinds = source_scene.dimensions, source_scene.kinds
        self.road_points, self.road_offsets = source_scene.road_points, source_scene.road_offsets
        if not np.allclose(np.diff(self.times), 0.1, rtol=0, atol=1e-8):
            raise ValueError("The matched renderer requires the world's 0.1 second grid")
        if (
            self.binding.get("schema_version") != 1
            or self.binding["log_name"] != self.metadata["source_log"]
            or self.binding["map_name"] != self.metadata["map_name"]
            or self.binding["source_start_timestamp_us"]
            != self.metadata["source_start_timestamp_us"]
            or "ego" not in self.ids
        ):
            raise ValueError("A source-matched NAVSIM scene with ego is required")
        validate_ego_camera_owner(self.ids, self.controlled)
        self.travel_id = int(self.binding["travel_id"])
        self.capture_end_tick = self.binding.get("render_end_tick")
        if self.capture_end_tick is None:
            self.capture_end_tick = len(self.times) - 1
        if type(self.capture_end_tick) is not int or not 0 < self.capture_end_tick < len(
            self.times
        ):
            raise ValueError("Bounded rendering interval outside world support")
        self._validate_binding()
        assets = {
            name: {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))}
            for name, path in zip(("config", "checkpoint", "road_block_config"), runtime.paths)
        }
        assets["source_metadata"] = {
            "path": str(runtime.metadata_path.resolve()),
            "sha256": _sha256(runtime.metadata_path),
        }
        scene_digest = _sha256(Path(scene))
        for key, digest in (
            ("scene", scene_digest),
            ("mtgs_metadata", assets["source_metadata"]["sha256"]),
        ):
            identity = self.binding.get("inputs", {}).get(key)
            if identity is not None and (
                not isinstance(identity, dict) or identity.get("sha256") != digest
            ):
                raise ValueError(f"Binding {key} content hash differs from loaded input")
        self.camera_names, self.sensor_stride = cameras, sensor_stride
        from driversi.data.assets.mtgs_coverage import runtime_nodes, verified_asset_preflight

        self.asset_preflight = verified_asset_preflight(
            self.binding, runtime_nodes(runtime), source_scene
        )
        if require_complete_assets and not self.asset_preflight["complete"]:
            raise ValueError("Source reconstruction assets incomplete; see preflight report")
        self.node_tokens = {
            name.removeprefix("rigid_object_")
            for name, node in runtime.model.gaussian_models.items()
            if name.startswith("rigid_object_") and int(node.travel_id) == self.travel_id
        }
        # Lifetime masks are immutable source metadata; copy once instead of
        # synchronizing a GPU scalar for every node on every captured frame.
        self._node_presence = {
            token: _array(runtime.model.gaussian_models["rigid_object_" + token].in_frame_mask)
            .astype(bool)
            .copy()
            for token in self.node_tokens
        }
        missing = {
            actor
            for actor, active in zip(self.ids, self.controlled)
            if active and actor != "ego" and actor not in self.node_tokens
        }
        if missing:
            raise ValueError(f"Controlled vehicles lack reconstruction nodes: {sorted(missing)}")
        self.virtual_camera_actors = virtual_camera_owners(
            virtual_camera_actors, self.ids, self.controlled, self.node_tokens
        )
        for actor in self.virtual_camera_actors:
            if runtime.model.gaussian_models["rigid_object_" + actor].is_static:
                raise ValueError("Virtual cameras require a dynamic reconstructed owner node")
        source_cameras = tuple(
            dict.fromkeys((*cameras, *(("CAM_F0",) if self.virtual_camera_actors else ())))
        )
        self._indices = {}
        for camera_name in source_cameras:
            entries = []
            for i, path in enumerate(runtime.sources.image_filenames):
                if (
                    int(runtime.sources.travel_ids[i]) == self.travel_id
                    and Path(path).parent.name == camera_name
                ):
                    frame = runtime.source_frames[
                        (self.travel_id, str(runtime.sources.frame_tokens[i]))
                    ]
                    if (
                        frame.get("log_name", self.metadata["source_log"])
                        != self.metadata["source_log"]
                    ):
                        raise ValueError("Reconstruction camera belongs to another log")
                    timestamp = int(frame["timestamp"])
                    if not self._reference_seed_supported(timestamp):
                        continue
                    if not (
                        self.binding["source_start_timestamp_us"]
                        <= timestamp
                        <= self.binding["source_end_timestamp_us"]
                    ):
                        continue
                    calibration = frame["cams"].get(camera_name)
                    if calibration is None or Path(calibration["data_path"]).stem != str(
                        runtime.sources.cam_tokens[i]
                    ):
                        raise ValueError("Reconstruction camera token differs from source metadata")
                    entries.append((timestamp, i))
            if not entries:
                raise ValueError(f"Matched traversal has no source camera {camera_name}")
            self._indices[camera_name] = sorted(entries)
            if len({stamp for stamp, _ in entries}) != len(entries):
                raise ValueError("Duplicate camera seeds at a source timestamp")
            # All sensor ticks and any early-terminal tick must be renderable.
            for value in self.times[: self.capture_end_tick + 1]:
                stamp = self.metadata["source_start_timestamp_us"] + round(float(value) * 1e6)
                if min(abs(seed - stamp) for seed, _ in entries) > 60000:
                    raise ValueError("Reconstruction does not cover every possible terminal tick")
        self._height_profiles, self._height_profile_reasons = {}, {}
        if self.height_mode == "reference-track":
            self._build_height_profiles()
        self._samples = {}
        self._source_images = {}
        self.generation = 0
        self.output.mkdir(parents=True)
        self.manifest = {
            "schema_version": 1,
            "kind": "mtgs_world_session",
            "registration_mode": self.registration_mode,
            "asset_preflight": self.asset_preflight,
            "status": "initializing",
            "session_id": self.session_id,
            "sampling": {"decision_mode": "structured", "visual_actor_ids": [], "enabled": False},
            "trainable": False,
            "display_only": True,
            "source_log": self.metadata["source_log"],
            "scenario_id": self.metadata["scenario_id"],
            "dt_seconds": 0.1,
            "sensor_stride": sensor_stride,
            "camera_ids": [
                *cameras,
                *(owner + "__virtual_front" for owner in self.virtual_camera_actors),
            ],
            "virtual_camera_actors": list(self.virtual_camera_actors),
            "initial_world_by_generation": {},
            "binding_mode": (
                "source_basis_transport_without_terrain_certificate"
                if self.registration_mode == "source-frame"
                else "reference_relative_planar_motion_with_refined_height_and_tilt"
                if height_mode == "source-time"
                else "reference_relative_planar_motion_with_bounded_track_height_correction"
            ),
            "height_mode": self.height_mode,
            "requested_height_mode": height_mode,
            "height_configuration": (
                next(iter(self._height_profiles.values())).configuration
                if self._height_profiles
                else None
            ),
            "height_profile_unavailable": dict(self._height_profile_reasons),
            "replay_binding": "uncontrolled_nodes_keep_source_reconstruction_pose",
            "time_alignment": {
                "reference": "interpolate_reference_at_each_source_seed_then_apply_world_delta",
                "max_seed_offset_us": 60000,
                "endpoint_reference_clamp_limit_us": 1000,
                "unbound_objects": "source_seed_time_not_interpolated_to_world_tick",
                "exposure": "original_camera_exposure_offset_retained_not_time_warped",
            },
            "limits": [
                "rule_driven_interaction_not_learned_visual_self_play",
                "reference_relative_display_not_registered_collision_geometry",
                "reconstruction_only_nodes_are_not_physics_actors",
                "pedestrian_and_cyclist_appearance_unverified",
                "novel_view_quality_unverified",
                "source_training_camera_appearance",
            ],
            "inputs": {
                name: {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))}
                for name, path in (
                    ("scene", scene),
                    ("road_graph", road_graph),
                    ("binding", binding),
                )
            },
            "assets": assets,
            "backend": {
                "image_write_workers": self.image_write_workers,
                "artifact_writer_source_sha256": _sha256(
                    Path(__file__).with_name("mtgs_artifacts.py")
                ),
                "revision": runtime.revision,
                "camera_motion_version": "optimized_source_then_world_delta-v1",
                "session_source_sha256": _sha256(Path(__file__)),
                "runtime_source_sha256": _sha256(Path(__file__).with_name("mtgs_runtime.py")),
                "surface_quality_source_sha256": _sha256(
                    Path(__file__).with_name("surface_quality.py")
                ),
                "multiview_source_sha256": _sha256(Path(__file__).with_name("mtgs_multiview.py")),
                "height_source_sha256": _sha256(Path(__file__).with_name("mtgs_height.py")),
            },
            "frames": [],
            "steps": [],
            "resets": [],
            "performance": {},
        }
        if population_config is not None:
            self.manifest["inputs"]["population"] = {
                "path": str(Path(population_config).resolve()),
                "sha256": _sha256(Path(population_config)),
            }
        selected_visual_actors = ()
        if visual_policy:
            from driversi.contracts.population import EpisodePopulation
            from driversi.sim.sensors.visual_sampling import visual_population

            original = (
                EpisodePopulation.load(population_config) if population_config is not None else None
            )
            effective, selected_visual_actors = visual_population(
                tuple(a for a, active in zip(self.ids, self.controlled) if active),
                (*(("ego",) if "CAM_F0" in self.camera_names else ()), *self.virtual_camera_actors),
                visual_actors,
                original,
                desired_speed,
                visual_adapter,
            )
            population_config = self.output / "effective-population.json"
            population_config.write_text(json.dumps(effective.to_dict(), allow_nan=False))
            self.manifest["inputs"]["effective_population"] = {
                "path": str(population_config),
                "sha256": _sha256(population_config),
            }
        _write_manifest(self.output, self.manifest, compact=True)
        self.worker = None
        try:
            self._worker_settings = dict(
                python=world_python,
                scene=scene,
                road_graph=road_graph,
                native_path=native_path,
                desired_speed=desired_speed,
                population_config=population_config,
            )
            if recorded_manifest is None:
                self.worker = WorldWorker(**self._worker_settings)
            else:
                from driversi.rendering.gaussian.recorded_world import RecordedWorldWorker

                self.worker = RecordedWorldWorker(
                    manifest=recorded_manifest, **self._worker_settings
                )
                self.manifest["world_source"] = "recorded_states_no_new_simulation"
                self.manifest["recorded_manifest_sha256"] = _sha256(recorded_manifest)

            self.current = self.worker.current
            self.manifest["initial_world_by_generation"][str(self.generation)] = copy.deepcopy(
                self.current
            )
            if tuple(self.current["all_actor_ids"]) != self.ids:
                raise ValueError("Physics and renderer actor ordering differ")
            if self.current.get("dt_seconds") != 0.1 or tuple(
                self.current["controlled_ids"]
            ) != tuple(actor for actor, active in zip(self.ids, self.controlled) if active):
                raise ValueError("Physics timing or controlled population differs from scene")
            self._record_population(self.current)
            self.manifest.update(
                controlled_ids=self.current["controlled_ids"],
                physics_build_id=self.current["build_id"],
                controller_configuration=self.current["controller_configuration"],
                status="running",
            )
            self.map = {
                "roads": [
                    self.road_points[a:b, :2].tolist()
                    for a, b in zip(self.road_offsets[:-1], self.road_offsets[1:])
                ],
                "routes": [
                    {"actor_id": actor, "points": value["points"]}
                    for actor, value in self.current["controller_configuration"]["routes"].items()
                ],
            }
            self.manifest["map"] = self.map
            self.latest = self.capture()
            if visual_policy:
                from driversi.sim.sensors.visual_sampling import ManagedVisualSampler

                self.visual_sampler = ManagedVisualSampler(
                    self, selected_visual_actors, history=visual_history, policy=visual_adapter
                )
                self.manifest["sampling"]["enabled"] = True
            self._publish()
        except BaseException as error:
            self._failed(error)
            self.close()
            raise

    def _record_population(self, ready: dict) -> None:
        population = ready.get("population_configuration")
        policies = ready.get("behavior_policies")
        if population is not None:
            from driversi.contracts.population import EpisodePopulation

            parsed = EpisodePopulation.from_dict(population)
            parsed.validate_actors(tuple(ready["controlled_ids"]))
            if parsed.behavior_policies != policies:
                raise ValueError("Worker population and behavior identity differ")
            population = parsed.to_dict()
        elif policies is not None:
            raise ValueError("Worker behavior identity requires population configuration")
        external = any(
            binding.get("controller") in {"external-joint-v1", "external-native-state-v1"}
            for binding in (policies or {}).values()
        )
        if external:
            self.manifest["limits"] = [
                "external_policy_inputs_unrecorded"
                if value == "rule_driven_interaction_not_learned_visual_self_play"
                else value
                for value in self.manifest.get("limits", [])
            ]
        self.manifest.update(
            population_configuration=copy.deepcopy(population),
            behavior_policies=copy.deepcopy(policies),
            protocol={
                "kind": "external_population_interaction"
                if external
                else "rule_population_interaction",
                "rgb_is_policy_input": None if external else False,
            },
        )

    def _failed(self, error):
        self.manifest.update(
            status="failed", failure={"type": type(error).__name__, "reason": str(error)}
        )
        _write_manifest(self.output, self.manifest, compact=True)

    def _validate_binding(self):
        """Recheck the audit against this loaded traversal, before starting physics."""
        binding = self.binding
        frames = binding.get("frames", [])
        count = self.metadata.get("source_frames")
        start = self.metadata.get("source_start_frame")
        if (
            binding.get("kind") != "navsim_mtgs_exact_source_binding_audit"
            or type(count) is not int
            or type(start) is not int
            or count < 2
            or not 2 <= len(frames) <= count
            or (binding.get("render_end_tick") is None and len(frames) != count)
            or binding.get("match_count") != len(frames)
            or not str(binding.get("video_token", "")).endswith("-" + str(self.travel_id))
        ):
            raise ValueError("Binding must cover the complete declared source window")
        timestamps, tokens = [], set()
        for offset, frame in enumerate(frames):
            token, timestamp = frame.get("token"), frame.get("timestamp_us")
            source = self.runtime.source_frames.get((self.travel_id, token))
            if (
                frame.get("navsim_index") != start + offset
                or token in tokens
                or type(timestamp) is not int
                or source is None
                or source.get("skipped")
                or int(source["timestamp"]) != timestamp
                or source.get("log_name") != self.metadata["source_log"]
            ):
                raise ValueError("Binding frame token, traversal, time, or source identity differs")
            camera = source["cams"].get(binding.get("camera_name"))
            if (
                camera is None
                or str(camera["data_path"]) != frame.get("camera_image_path")
                or Path(camera["data_path"]).stem != frame.get("camera_token")
                or int(camera["timestamp"]) != frame.get("camera_timestamp_us")
            ):
                raise ValueError("Binding camera identity or exposure differs")
            for field, original in (
                ("original_pose_local", "ego2global_original"),
                ("refined_pose_local", "ego2global"),
            ):
                if not np.array_equal(np.asarray(frame.get(field)), np.asarray(source[original])):
                    raise ValueError("Binding was made from different reconstruction poses")
            tokens.add(token)
            timestamps.append(timestamp)
        if (
            frames[0]["token"] != self.metadata["scenario_id"]
            or timestamps[0] != binding["source_start_timestamp_us"]
            or timestamps[-1] != binding.get("source_end_timestamp_us")
            or any(a >= b for a, b in zip(timestamps, timestamps[1:]))
            or timestamps[-1]
            < timestamps[0]
            + round(float(self.times[getattr(self, "capture_end_tick", len(self.times) - 1)]) * 1e6)
        ):
            raise ValueError("Binding source window does not cover the declared capture horizon")

    def _reference_seed_supported(self, timestamp: int) -> bool:
        # Source binding includes the extra bracketing frame needed to compile
        # a 10 Hz horizon. That frame is not a valid simulator pose or height
        # sample. Keep the exact provenance audit, but never extrapolate it.
        elapsed = timestamp - self.metadata["source_start_timestamp_us"]
        return (
            round(float(self.times[0]) * 1e6) - 1000
            <= elapsed
            <= round(float(self.times[-1]) * 1e6) + 1000
        )

    def _reference_sample(self, timestamp: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample the physical reference without interpolating absent placeholders."""
        seconds = (timestamp - self.metadata["source_start_timestamp_us"]) / 1e6
        if seconds < self.times[0] - 0.001 or seconds > self.times[-1] + 0.001:
            raise ValueError("Source seed is outside bounded reference support")
        seconds = float(np.clip(seconds, self.times[0], self.times[-1]))
        left = int(
            np.clip(
                np.searchsorted(self.times, seconds + 1e-10, side="right") - 1,
                0,
                len(self.times) - 1,
            )
        )
        right = min(left + 1, len(self.times) - 1)
        ratio = (
            0.0
            if left == right
            else (seconds - self.times[left]) / (self.times[right] - self.times[left])
        )
        before, after = self.reference[left], self.reference[right]
        presence = getattr(self, "lifecycle_presence", None)
        present = np.ones(len(before), bool) if presence is None else presence[left].copy()
        interpolate = present if presence is None else present & presence[right]
        result = before.copy()
        result[interpolate] += ratio * (after[interpolate] - before[interpolate])
        yaw = np.arctan2(np.sin(after[:, 3] - before[:, 3]), np.cos(after[:, 3] - before[:, 3]))
        result[interpolate, 3] = before[interpolate, 3] + ratio * yaw[interpolate]
        result[~present] = 0
        return result, present

    def _reference_at(self, timestamp: int) -> np.ndarray:
        return self._reference_sample(timestamp)[0]

    def _build_height_profiles(self):
        """Cache same-actor raw XY -> refined centre Z; this is not terrain."""
        # Use the complete matched source frame set, independent of requested
        # camera order or camera-specific sampling density.
        by_timestamp = {}
        for index, token in enumerate(self.runtime.sources.frame_tokens):
            if int(self.runtime.sources.travel_ids[index]) != self.travel_id:
                continue
            source = self.runtime.source_frames[(self.travel_id, str(token))]
            timestamp = int(source["timestamp"])
            if not self._reference_seed_supported(timestamp):
                continue
            if (
                source.get("log_name", self.metadata["source_log"]) != self.metadata["source_log"]
                or not self.binding["source_start_timestamp_us"]
                <= timestamp
                <= self.binding["source_end_timestamp_us"]
            ):
                continue
            identity = str(token), int(self.runtime.sources.frame_ids[index])
            if timestamp in by_timestamp and by_timestamp[timestamp][0] != identity:
                raise ValueError("Ambiguous reconstruction frame identity for height sampling")
            by_timestamp.setdefault(timestamp, (identity, index))
        entries = [(timestamp, item[1]) for timestamp, item in sorted(by_timestamp.items())]
        if len(entries) < 2:
            raise ValueError("Height profiles require at least two distinct source frames")
        reference = np.stack([self._reference_at(timestamp) for timestamp, _ in entries])
        for actor, controlled in zip(self.ids, self.controlled, strict=True):
            if not controlled:
                continue
            node = None
            if actor != "ego":
                node = (
                    self.runtime.model.gaussian_models["rigid_object_" + actor]
                    if "rigid_object_" + actor in self.runtime.model.gaussian_models
                    else None
                )
                if node is None or node.is_static:
                    self._height_profile_reasons[actor] = "no_dynamic_height_reference"
                    continue
            heights = []
            for _, index in entries:
                source = self.runtime.source_frames[
                    (self.travel_id, str(self.runtime.sources.frame_tokens[index]))
                ]
                if actor == "ego":
                    matrix = np.asarray(source["ego2global"], dtype=float)
                    heights.append(float((matrix[:3, 3] + matrix[:3, 0] * 1.461)[2]))
                else:
                    frame_index = int(self.runtime.sources.frame_ids[index])
                    quat, trans = node.get_object_pose(frame_index, None)
                    if quat is None:
                        self._height_profile_reasons[actor] = "incomplete_node_height_reference"
                        break
                    heights.append(float(_array(trans)[2]))
            if len(heights) == len(entries):
                self._height_profiles[actor] = ReferenceTrackHeight(
                    reference[:, self.ids.index(actor), :2], np.asarray(heights)
                )

    def _height_corrections(self, current_poses, reference):
        mode = getattr(self, "height_mode", "source-time")
        if getattr(self, "registration_mode", "legacy-map-axes") == "source-frame":
            return {
                actor: {
                    "mode": "source-frame",
                    "actor_id": actor,
                    "supported": False,
                    "correction_m": 0.0,
                    "reason": "source_frame_transport_not_terrain",
                    "geometry_role": "source_basis_transport_only",
                    "fallback": "no_additive_track_height_to_avoid_double_counting",
                }
                for i, actor in enumerate(self.ids)
                if self.controlled[i]
            }
        values = {}
        for i, actor in enumerate(self.ids):
            if not self.controlled[i]:
                continue
            value = {
                "supported": False,
                "correction_m": 0.0,
                "reason": "source_time_height_not_ground_qualified",
            }
            if mode == "reference-track":
                profile = self._height_profiles.get(actor)
                if profile is None:
                    value["reason"] = self._height_profile_reasons.get(
                        actor, "missing_height_reference"
                    )
                else:
                    value = profile.correction(current_poses[i, :2], reference[i, :2])
            values[actor] = {
                "mode": mode,
                "actor_id": actor,
                **value,
                "geometry_role": "display_height_proxy_not_physics_or_terrain",
                "fallback": (None if value["supported"] else "retain_source_time_height"),
            }
        return values

    def _earlier_asset_anchor(self, actor, timestamp, frame_index):
        """Same actor/log/window only; never borrow a future pose or new identity."""
        i = self.ids.index(actor)
        owned = self.controlled[i] or actor in self.metadata.get("reactive_background", {}).get(
            "assignments", {}
        )
        if self.kinds[i] != VEHICLE or not owned:
            return None
        cache = getattr(self, "_asset_anchor_cache", None)
        if cache is None:
            cache = self._asset_anchor_cache = {}
        if actor not in cache:
            i = self.ids.index(actor)
            candidates = []
            entries = sorted({entry for values in self._indices.values() for entry in values})
            seen = set()
            for stamp, index in entries:
                frame = int(self.runtime.sources.frame_ids[index])
                if frame in seen:
                    continue
                seen.add(frame)
                mask = self._node_presence[actor]
                if frame >= len(mask) or not mask[frame]:
                    continue
                reference, present = self._reference_sample(stamp)
                if present[i]:
                    candidates.append((stamp, frame, reference))
            cache[actor] = candidates
        eligible = [v for v in cache[actor] if v[0] <= timestamp and v[1] <= frame_index]
        return max(eligible, key=lambda v: (v[0], v[1])) if eligible else None

    def _source_inventory(self, source, frame_index: int):
        tokens, kinds = source["track_tokens"], source["gt_names"]
        if len(tokens) != len(kinds) or len(set(tokens)) != len(tokens):
            raise ValueError("Source inventory has inconsistent track identities")
        present_nodes = set()
        for token, presence in self._node_presence.items():
            if frame_index < len(presence) and bool(presence[frame_index]):
                present_nodes.add(token)
        current = getattr(self, "current", {})
        present_world = {
            a for a, p in zip(self.ids, current.get("presence", [True] * len(self.ids))) if p
        }
        counts, missing, unbound = {}, [], []
        for token, kind in zip(tokens, kinds, strict=True):
            token, kind = str(token), str(kind)
            counts[kind] = counts.get(kind, 0) + 1
            item = {"track_token": token, "kind": kind}
            if token not in present_nodes:
                missing.append(item)
            if token not in present_world:
                unbound.append(item)
        return {
            "source_frame_token": str(source["token"]),
            "source_timestamp_us": int(source["timestamp"]),
            "source_object_count": len(tokens),
            "source_category_counts": counts,
            "source_without_reconstruction_node": missing,
            "source_without_physics_actor": unbound,
            "reconstruction_only_current_nodes": sorted(present_nodes - set(self.ids)),
            "scope": "source_annotations_and_active_nodes_not_pixel_visibility",
        }

    def _transition(self):
        if "policy_actions" not in self.current:
            return None
        return {
            "generation": self.generation,
            "observation_tick": self.current["tick"] - 1,
            "successor_tick": self.current["tick"],
            **{
                key: copy.deepcopy(self.current[key])
                for key in (
                    "policy_actions",
                    "executed_actions",
                    "diagnostics",
                    "valid",
                    "terminated",
                    "truncated",
                    "events",
                )
            },
            **{
                key: copy.deepcopy(self.current[key])
                for key in (
                    "behavior_policies",
                    "action_origins",
                    "override_actor_ids",
                    "policy_decision",
                )
                if key in self.current
            },
        }

    def _sample(self, camera_name: str, timestamp: int):
        entries = self._indices[camera_name]
        source_time, index = min(entries, key=lambda item: (abs(item[0] - timestamp), item[0]))
        if abs(source_time - timestamp) > 60000:
            raise ValueError("No source reconstruction seed within 60 ms of simulation time")
        if index not in self._samples:
            camera = self.runtime.source_camera(index)
            projection = self.runtime.source_lens_projection(camera)
            self._samples[index] = {}, camera, projection
        data, camera, projection = self._samples[index]
        return source_time, index, data, copy.deepcopy(camera), projection

    def _actors(self):
        state = np.asarray(self.current["state"])
        alive = dict(zip(self.current["controlled_ids"], self.current["alive"]))
        return [
            {
                "id": actor,
                "controlled": bool(self.controlled[i]),
                "position": state[i, :2].tolist(),
                "yaw": float(state[i, 3]),
                "speed": float(np.linalg.norm(state[i, 4:6])),
                "alive": alive.get(actor),
                "collision": bool(state[i, 10]),
                "offroad": bool(state[i, 11]),
                "dimensions": self.dimensions[i].tolist(),
                "kind": int(self.kinds[i]),
                "reconstructed": actor in self.node_tokens,
                "present": self.current.get("presence", [True] * len(self.ids))[i],
                "motion_origin": self.metadata.get("observed_motion_inference", {}).get(actor),
                "role": "controlled"
                if self.controlled[i]
                else "reactive_background"
                if actor in self.metadata.get("reactive_background", {}).get("assignments", {})
                else "world_replay",
            }
            for i, actor in enumerate(self.ids)
        ]

    def capture(self):
        # The writer joins on every exit, including errors during later rendering.
        # _capture publishes only after finish has resolved all artifact hashes.
        with CaptureArtifactWriter(getattr(self, "image_write_workers", 1)) as writer:
            return self._capture(writer)

    def _capture(self, writer):
        from driversi.rendering.gaussian.mtgs_runtime import project_source_lens
        from driversi.rendering.gaussian.surface_quality import (
            summarize_surface,
            surface_previews,
        )

        started = time.perf_counter()
        tick = int(self.current["tick"])
        if tick > getattr(self, "capture_end_tick", len(self.times) - 1):
            raise ValueError("No GS capture outside the explicitly bound render interval")
        timestamp = self.metadata["source_start_timestamp_us"] + round(tick * 0.1 * 1e6)
        state = np.asarray(self.current["state"])
        current_poses = state[:, :4]
        if tick >= len(self.times):
            raise ValueError("World tick exceeds compiled reference support")
        world_deltas = [
            planar_delta(pose, reference)
            for pose, reference in zip(current_poses, self.reference[tick])
        ]
        inventories, height_reports = [], []
        frame = {
            "generation": self.generation,
            "tick": tick,
            "render_tick": tick,
            "time_seconds": tick * 0.1,
            "simulation_timestamp_us": timestamp,
            "actors": self._actors(),
            "cameras": [],
            "done": self.current["done"],
            "lifecycle": copy.deepcopy(self.current.get("lifecycle")),
            "source_log": self.metadata["source_log"],
            "controlled_ids": self.current["controlled_ids"],
        }
        if "behavior_policies" in self.current:
            frame["behavior_policies"] = copy.deepcopy(self.current["behavior_policies"])
        transition = self._transition()
        frame["transition"] = transition
        if transition is not None:
            frame.update(
                {
                    key: copy.deepcopy(transition[key])
                    for key in ("policy_actions", "executed_actions", "diagnostics")
                }
            )
        output = self.output / f"generation-{self.generation:03d}"
        output.mkdir(exist_ok=True)
        streams = [("ego", name, name, False) for name in self.camera_names]
        streams += [
            (actor, "CAM_F0", actor + "__virtual_front", True)
            for actor in getattr(self, "virtual_camera_actors", ())
        ]
        buffers = {}
        source_images = dict(self._source_images)
        artifact_records = []
        for owner, camera_name, stream_id, virtual in streams:
            sample_started = time.perf_counter()
            source_time, index, _, camera, projection = self._sample(camera_name, timestamp)
            sample_seconds = time.perf_counter() - sample_started
            seed_reference, seed_present = self._reference_sample(source_time)
            deltas = [
                planar_delta(pose, reference)
                for pose, reference in zip(current_poses, seed_reference)
            ]
            height = self._height_corrections(current_poses, seed_reference)
            owner_height = height.get(
                owner,
                {
                    "mode": getattr(self, "height_mode", "source-time"),
                    "actor_id": owner,
                    "supported": False,
                    "correction_m": 0.0,
                    "reason": "uncontrolled_source_camera",
                    "geometry_role": "display_height_proxy_not_physics_or_terrain",
                    "fallback": "retain_source_time_height",
                },
            )
            scene_height_supported = all(item["supported"] for item in height.values())
            height_reports.append(
                {
                    "camera_id": stream_id,
                    "source_timestamp_us": source_time,
                    "all_controlled_supported": scene_height_supported,
                    "actors": list(height.values()),
                }
            )
            source = self.runtime.source_frames[
                (self.travel_id, str(camera.metadata["frame_token"]))
            ]
            frame_index = self.runtime.validate_camera(camera)[1]
            inventories.append(
                {
                    "camera_id": stream_id,
                    "owner_actor_id": owner,
                    **self._source_inventory(source, frame_index),
                }
            )
            poses, asset_frames, asset_evidence = {}, {}, []
            logged = getattr(self, "lifecycle_presence", None) is not None
            present = np.asarray(self.current.get("presence", [True] * len(self.ids)), bool)
            present_ids = {actor for actor, flag in zip(self.ids, present) if flag}
            excluded = set(self.node_tokens) - present_ids if logged else set()
            unsupported = []
            for i, actor in enumerate(self.ids):
                # Only controlled objects override their reconstructed source.
                # Static MTGS nodes can retain averaged non-unit quaternions;
                # normalizing those would change even the untouched baseline.
                if present[i] and (self.controlled[i] or logged) and actor in self.node_tokens:
                    node = self.runtime.model.gaussian_models["rigid_object_" + actor]
                    asset_frame, delta, correction = (
                        frame_index,
                        deltas[i],
                        height.get(actor, {}).get("correction_m", 0.0),
                    )
                    reused = logged and (
                        not self._node_presence[actor][frame_index] or not seed_present[i]
                    )
                    if reused:
                        anchor = self._earlier_asset_anchor(
                            actor, min(source_time, timestamp), frame_index
                        )
                        if anchor is None:
                            excluded.add(actor)
                            unsupported.append(actor)
                            continue
                        stamp, asset_frame, anchor_reference = anchor
                        delta = planar_delta(current_poses[i], anchor_reference[i])
                        correction = (
                            self._height_corrections(current_poses, anchor_reference)
                            .get(actor, {})
                            .get("correction_m", 0.0)
                        )
                        asset_frames[actor] = asset_frame
                        asset_evidence.append(
                            {
                                "actor_id": actor,
                                "source_frame": asset_frame,
                                "timestamp_us": stamp,
                                "age_seconds": (timestamp - stamp) / 1e6,
                                "mode": "retained_rigid_asset_world_pose",
                                "quality": "appearance_and_ground_support_unverified",
                            }
                        )
                    quat, trans = node.get_object_pose(asset_frame, None)
                    if quat is None:
                        raise ValueError(f"World actor {actor} absent from source reconstruction")
                    if (
                        not reused
                        and not self.controlled[i]
                        and np.linalg.norm(delta[0]) < 0.002
                        and abs(delta[1]) < 1e-5
                    ):
                        continue  # Preserve unmodified averaged static-node quaternions.
                    quaternion = _array(quat)
                    if not self.controlled[i]:
                        quaternion = quaternion / np.linalg.norm(quaternion)
                    if getattr(self, "registration_mode", "legacy-map-axes") == "source-frame":
                        from driversi.rendering.gaussian.mtgs_registration import move_rigid

                        asset_source = source
                        if reused:
                            token = self.runtime.sources.frame_tokens[
                                next(
                                    j
                                    for j, fid in enumerate(self.runtime.sources.frame_ids)
                                    if int(fid) == asset_frame
                                    and int(self.runtime.sources.travel_ids[j]) == self.travel_id
                                )
                            ]
                            asset_source = self.runtime.source_frames[(self.travel_id, str(token))]
                        xyz, quaternion = move_rigid(
                            asset_source, _array(trans), quaternion, *delta
                        )
                    else:
                        xyz, quaternion = apply_rigid_delta(_array(trans), quaternion, *delta)
                    xyz[2] += correction
                    poses[actor] = xyz, quaternion
            binding_args = {"rigid_source_frames": asset_frames} if asset_frames else {}
            virtual_details = {"retained_asset_bindings": asset_evidence}
            if virtual:
                if owner not in poses:
                    raise ValueError(
                        "Virtual camera owner has no current world-bound reconstruction pose"
                    )
                target, rig = roof_front_camera(
                    *poses[owner], self.dimensions[self.ids.index(owner)]
                )
                virtual_details.update(
                    {
                        "rig": rig,
                        "support": novel_view_support(target, _array(camera.camera_to_worlds)[0]),
                        "source_comparison": "unavailable_no_real_image_for_virtual_owner",
                    }
                )
                if getattr(self, "height_mode", "source-time") == "reference-track":
                    virtual_details["support"]["height_support"] = (
                        "bounded_same_actor_reference_track_proxy"
                    )
                    virtual_details["support"]["counterfactual_road_clearance"] = (
                        "unverified_proxy_not_terrain"
                    )
                result = self.runtime.render_camera(
                    camera,
                    rigid_poses=poses,
                    target_camera_to_world=target,
                    **binding_args,
                    exclude_tokens=tuple(sorted(excluded | {owner})),
                )
            else:
                # MTGS stores the refined ego pose directly in reconstruction axes.
                ego_matrix = np.asarray(source["ego2global"], dtype=float)
                pivot = ego_matrix[:3, 3] + ego_matrix[:3, 0] * 1.461
                ego_delta, ego_yaw = deltas[self.ids.index("ego")]
                result, moved, support_seed = render_moved_source_camera(
                    self.runtime,
                    camera,
                    pivot,
                    ego_delta,
                    ego_yaw,
                    owner_height["correction_m"],
                    poses,
                    source_registration=source
                    if getattr(self, "registration_mode", "legacy-map-axes") == "source-frame"
                    else None,
                    **binding_args,
                    exclude_tokens=tuple(sorted(excluded)),
                )
                virtual_details["support"] = novel_view_support(moved, support_seed)
                virtual_details["support"]["pose_comparison"] = (
                    "pre_optimizer_moved_vs_source_seed"
                    if ego_yaw == 0
                    else "effective_world_target_vs_optimized_source_seed"
                )
                virtual_details["support"]["limitations"] = [
                    "source_calibration_counterfactual_pose",
                    "novel_view_quality_unverified",
                    "ego_vehicle_has_no_reconstruction_asset",
                    "pedestrian_cyclist_assets_unverified",
                ]
                virtual_details["support"]["height_support"] = (
                    "source_basis_transport_not_terrain"
                    if getattr(self, "registration_mode", "legacy-map-axes") == "source-frame"
                    else "bounded_same_actor_reference_track_proxy"
                    if getattr(self, "height_mode", "source-time") == "reference-track"
                    else "source_time_node_altitude_not_terrain_following"
                )
            if virtual:
                camera_to_body = rig["camera_to_body_optical"]
            else:
                from driversi.sim.sensors.visual_sampling import body_camera_transform

                body = ego_matrix.copy()
                body[:3, 3] = pivot
                if getattr(self, "registration_mode", "legacy-map-axes") == "source-frame":
                    from driversi.rendering.gaussian.mtgs_registration import move_camera

                    body[:3] = move_camera(source, body[:3], pivot, *deltas[self.ids.index("ego")])
                else:
                    body[:3] = apply_camera_delta(body[:3], pivot, *deltas[self.ids.index("ego")])
                body[2, 3] += owner_height["correction_m"]
                camera_to_body = body_camera_transform(result["camera_to_world"], body).tolist()
            projected = project_source_lens(result, projection)
            image, valid = projected["rgb"], projected["valid"]
            surface_alpha, surface_depth = projected["accumulation"], projected["depth"]
            surface_summary = summarize_surface(image, surface_alpha, surface_depth, valid)
            if getattr(self, "visual_mode", False):
                buffers[stream_id] = {"rgb": image.copy(), "valid": valid.copy()}
                buffers[stream_id]["rgb"].setflags(write=False)
                buffers[stream_id]["valid"].setflags(write=False)
            write_started = time.perf_counter()
            names = {
                key: f"{tick:04d}-{stream_id}-{key}.png"
                for key in ("image", "pinhole", "source", "valid")
            }
            for key, array in (
                ("image", image),
                ("pinhole", result["rgb"]),
                ("valid", valid.astype(np.uint8) * 255),
            ):
                writer.png(output / names[key], array)
            if not virtual and index not in source_images:
                writer.source(
                    output / names["source"],
                    self.runtime.sources.image_filenames[index],
                    (image.shape[1], image.shape[0]),
                )
                source_images[index] = output / names["source"]
            paths = {
                key: str((output / name).relative_to(self.output)).replace("\\", "/")
                for key, name in names.items()
            }
            paths["source"] = (
                None if virtual else source_images[index].relative_to(self.output).as_posix()
            )
            surface_assets = {}
            if surface_alpha is not None:
                alpha_preview, depth_preview = surface_previews(surface_alpha, surface_depth, valid)
                surface_path = output / f"{tick:04d}-{stream_id}-surface.npz"
                artifact_futures = {
                    "surface": writer.surface(surface_path, surface_alpha, surface_depth, valid)
                }
                artifact_paths = {"surface": surface_path}
                for key, preview in (("alpha", alpha_preview), ("depth", depth_preview)):
                    artifact_paths[key] = output / f"{tick:04d}-{stream_id}-{key}.png"
                    artifact_futures[key] = writer.png(artifact_paths[key], preview, with_hash=True)
                surface_assets = {
                    key: path.relative_to(self.output).as_posix()
                    for key, path in artifact_paths.items()
                }
                artifact_records.append((surface_summary, surface_assets, artifact_futures))
            frame["cameras"].append(
                {
                    "camera_id": stream_id,
                    "owner_actor_id": owner,
                    "sensor_id": "virtual_front" if virtual else camera_name,
                    "virtual": virtual,
                    "trainable": False,
                    "calibration_source_camera_id": camera_name,
                    "source_seed": {
                        "index": index,
                        "frame_token": str(camera.metadata["frame_token"]),
                        "camera_token": str(camera.metadata.get("cam_token", "")),
                        "timestamp_us": source_time,
                    },
                    **virtual_details,
                    **paths,
                    **surface_assets,
                    "surface_diagnostics": surface_summary,
                    "photometric": result.get(
                        "photometric",
                        {
                            "render_domain": "unknown",
                            "source_reference_domain": "source_rgb",
                            "quality_certificate": False,
                        },
                    ),
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "image_pixel_sha256": (
                        hashlib.sha256(image.tobytes()).hexdigest()
                        if getattr(self, "visual_mode", False)
                        else None
                    ),
                    "valid_pixel_sha256": (
                        hashlib.sha256(valid.tobytes()).hexdigest()
                        if getattr(self, "visual_mode", False)
                        else None
                    ),
                    "camera_to_body": camera_to_body,
                    "camera_to_body_semantics": "optical_RDF_to_display_body_FLU_not_collision_registration",
                    "camera_to_world": result["camera_to_world"],
                    "camera_to_world_semantics": result["camera_to_world_semantics"],
                    "camera_coordinate_frame": result["camera_coordinate_frame"],
                    "camera_pose_axes": result["camera_pose_axes"],
                    "source_timestamp_us": source_time,
                    "source_seed_error_us": source_time - timestamp,
                    "reference_endpoint_clamp_us": source_time
                    - int(
                        np.clip(
                            source_time,
                            self.metadata["source_start_timestamp_us"],
                            self.metadata["source_start_timestamp_us"]
                            + round(float(self.times[-1]) * 1e6),
                        )
                    ),
                    "camera_timestamp_us": (
                        timestamp if virtual else int(source["cams"][camera_name]["timestamp"])
                    ),
                    "source_camera_timestamp_us": int(source["cams"][camera_name]["timestamp"]),
                    "exposure_model": (
                        "virtual_instantaneous_world_tick_with_source_appearance_seed"
                        if virtual
                        else "source_camera_exposure_seed"
                    ),
                    "camera_model": projection["camera_model"],
                    "K": projection["K"],
                    "distortion": projection["distortion"],
                    "source_lens_valid_fraction": float(valid.mean()),
                    "render_seconds": result["render_seconds"],
                    "sample_seconds": sample_seconds,
                    "image_write_seconds": time.perf_counter() - write_started,
                    "image_write_seconds_scope": (
                        "artifact_submission_and_backpressure_excludes_capture_flush"
                        if writer.workers > 1
                        else "serial_artifact_write_and_surface_hash"
                    ),
                    "override_tokens": result["override_tokens"],
                    "excluded_tokens": result.get("excluded_tokens", []),
                    "present_world_without_source_asset_support": unsupported,
                    "camera_pose_semantics": result.get("camera_pose_semantics"),
                    "height_binding": owner_height,
                    "scene_height_supported": scene_height_supported,
                }
            )
        flush_started = time.perf_counter()
        writer.finish()
        for summary, assets, futures in artifact_records:
            summary["artifacts"] = {
                key: {"path": assets[key], "sha256": future.result()["sha256"]}
                for key, future in futures.items()
            }
        frame["artifact_write"] = {
            "workers": writer.workers,
            "max_pending_jobs": writer.max_pending,
            "peak_pending_jobs": writer.peak_pending,
            "flush_seconds": time.perf_counter() - flush_started,
            "completion": "all_files_closed_and_surface_hashes_resolved_before_frame_publish",
            "durability": "file_close_not_fsync",
        }
        frame["height_binding"] = {
            "mode": getattr(self, "height_mode", "source-time"),
            "all_controlled_supported": all(
                report["all_controlled_supported"] for report in height_reports
            ),
            "per_camera": height_reports,
        }
        frame["coverage"] = {
            "world_actor_count": len(present_ids),
            "world_roster_count": len(self.ids),
            "lifecycle_binding": "world_presence_authoritative" if logged else "legacy_persistent",
            "bound_rigid_count": len(set(self.ids) & self.node_tokens),
            "world_driven_rigid_tokens": sorted(poses),
            "source_reconstruction_replay_tokens": sorted(self.node_tokens - set(poses) - excluded),
            "world_without_rigid_nodes": sorted(present_ids - self.node_tokens - {"ego"}),
            "reconstruction_only_nodes": sorted(self.node_tokens - set(self.ids)),
            "source_inventories": inventories,
            "ego_appearance": "camera_car_not_reconstructed",
        }
        frame["max_reference_xy_deviation_m"] = max(
            float(np.linalg.norm(delta[0])) for delta in world_deltas
        )
        from driversi.rendering.diagnostics.quality import assess_frame_quality

        frame["render_quality"] = assess_frame_quality(frame)
        for camera, quality in zip(
            frame["cameras"], frame["render_quality"]["cameras"], strict=True
        ):
            camera["quality"] = quality
        frame["capture_seconds"] = time.perf_counter() - started
        self._source_images = source_images
        self._sensor_buffers = buffers
        self.manifest["frames"].append(frame)
        self.latest = frame
        return copy.deepcopy(frame)

    def _journal_transition(self, record):
        """Durably append the committed step before optional image production."""
        with (self.output / "transitions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _publish(self):
        values = [entry["advance_seconds"] for entry in self.manifest["steps"]]
        self.manifest["performance"] = {
            "steps": len(values),
            "physics_seconds": len(values) * 0.1,
            "advance_wall_seconds": sum(values),
            "initial_capture_seconds": self.manifest["frames"][0]["capture_seconds"],
            "scope": "step_decision_IPC_render_and_image_write_excludes_manifest_publish_state_copy_and_model_load",
            "step_latency_p50_seconds": float(np.percentile(values, 50)) if values else None,
            "step_latency_p95_seconds": float(np.percentile(values, 95)) if values else None,
            "advance_excluding_manifest_real_time_factor": len(values) * 0.1 / sum(values)
            if values and sum(values)
            else None,
        }
        if getattr(self, "visual_mode", False):
            self.manifest["performance"]["scope"] = (
                "physical_worker_IPC_render_write_excludes_managed_policy_and_manifest_publish"
            )
            self.manifest["performance"]["managed_policy_and_input_journal_wall_seconds"] = sum(
                entry.get("policy_decision", {}).get("collector_seconds", 0.0)
                for entry in self.manifest["steps"]
            )
        if self.manifest.get("branch_lineage"):
            start = self.manifest["branch_lineage"][-1]["inherited_steps"]
            own = self.manifest["steps"][start:]
            self.manifest["performance"]["branch_suffix"] = {
                "steps": len(own),
                "physics_seconds": len(own) * 0.1,
                "advance_wall_seconds": sum(row["advance_seconds"] for row in own),
                "inherited_prefix_excluded": True,
            }
        _write_manifest(self.output, self.manifest, compact=True)

    def decision_identity(self):
        return {
            "session_id": self.session_id,
            "generation": self.generation,
            "tick": self.current["tick"],
            "policy_versions": {
                a: {"policy_id": p["policy_id"], "version": p["version"]}
                for a, p in self.current.get("behavior_policies", {}).items()
            },
        }

    def advance(
        self,
        overrides: dict[str, int] | None = None,
        *,
        external_actions: dict[str, int] | None = None,
        expected_tick: int | None = None,
        decision_identity: dict | None = None,
        policy_decision: dict | None = None,
        native_targets: dict | None = None,
    ):
        if decision_identity is not None and decision_identity != self.decision_identity():
            raise ValueError("Decision identity is stale or belongs to another session/policy")
        if self.manifest["status"] != "running":
            raise RuntimeError("Reset the ended or failed session before advancing")
        if self.current["done"]:
            raise RuntimeError("Episode complete; reset before advancing")
        sampler = getattr(self, "visual_sampler", None)
        native = native_targets is not None
        native_ids = {
            a
            for a, p in self.current.get("behavior_policies", {}).items()
            if p["controller"] == "external-native-state-v1"
        }
        if native or native_ids:
            if (
                not native
                or sampler is not None
                or overrides
                or set(native_ids) != set(self.current["controlled_ids"])
                or decision_identity is None
                or expected_tick != self.current["tick"]
            ):
                raise ValueError(
                    "Native execution requires exact identity and exclusive native policy ownership"
                )
        if sampler is not None and external_actions is None:
            return sampler.advance(overrides)
        if sampler is not None and decision_identity is None:
            raise ValueError("Managed visual actions require a complete decision identity")
        if sampler is not None and (
            sampler.held_record is None
            or policy_decision is None
            or policy_decision.get("decision_id") != sampler.held_record["decision_id"]
            or policy_decision.get("observation_tick") != sampler.observed_tick
            or external_actions != sampler.held_actions
        ):
            raise ValueError("Managed visual actions must reference the recorded sampler decision")
        external_ids = {
            actor
            for actor, policy in self.current.get("behavior_policies", {}).items()
            if policy["controller"] == "external-joint-v1"
        }
        if not native and (external_actions is not None or external_ids):
            if decision_identity is None:
                raise ValueError("External actions require a complete decision identity")
            if type(expected_tick) is not int or expected_tick != self.current["tick"]:
                raise ValueError("External actions require the exact expected_tick")
            if (
                not isinstance(external_actions, dict)
                or set(external_actions) != external_ids
                or any(
                    type(value) is not int or not 0 <= value < 12
                    for value in external_actions.values()
                )
            ):
                raise ValueError(
                    "External actions must cover exactly the external actors with discrete actions"
                )
        elif expected_tick is not None and (
            type(expected_tick) is not int or expected_tick != self.current["tick"]
        ):
            raise ValueError("expected_tick does not match current world")
        request = {"expected_tick": self.current["tick"], "overrides": overrides or {}}
        if external_actions is not None:
            request["external_actions"] = external_actions
        if native:
            request = {
                "expected_tick": self.current["tick"],
                "native_targets": native_targets,
                "native_actions": external_actions,
            }
        started = time.perf_counter()
        record = None
        try:
            self.current = self.worker.request("native_step" if native else "step", **request)
            if policy_decision is not None:
                self.current["policy_decision"] = copy.deepcopy(policy_decision)
            due = self.current["tick"] % self.sensor_stride == 0 or self.current["done"]
            record = {
                "generation": self.generation,
                **copy.deepcopy(self.current),
                "capture_status": "pending" if due else "not_due",
                "advance_seconds": time.perf_counter() - started,
            }
            # The physics transition has committed. Persist it before optional
            # GPU/image I/O so an image failure cannot erase actions and events.
            self.manifest["steps"].append(record)
            self._journal_transition(record)
            if due:
                self.capture()
                record["capture_status"] = "captured"
            record["advance_seconds"] = time.perf_counter() - started
            if self.current["done"]:
                self.manifest["status"] = "completed"
            self._publish()
            return self.state()
        except BaseException as error:
            if record is not None:
                record["advance_seconds"] = time.perf_counter() - started
                if record["capture_status"] == "pending":
                    record["capture_status"] = "failed"
            self._failed(error)
            raise

    def reset(self):
        if getattr(self, "recorded_mode", False):
            raise ValueError("Recorded rendering cannot reset or simulate")
        try:
            self.current = self.worker.request("reset")
            self.generation += 1
            if getattr(self, "visual_sampler", None) is not None:
                self.visual_sampler.reset()
            self._sensor_buffers = {}
            self.manifest.setdefault("initial_world_by_generation", {})[str(self.generation)] = (
                copy.deepcopy(self.current)
            )
            self.manifest["status"] = "resetting"
            record = {
                "generation": self.generation,
                "tick": self.current["tick"],
                "capture_status": "pending",
            }
            self.manifest["resets"].append(record)
            _write_manifest(self.output, self.manifest, compact=True)
            self.capture()
            record["capture_status"] = "captured"
            self.manifest["status"] = "running"
            self.manifest.pop("failure", None)
            self._publish()
            return self.state()
        except BaseException as error:
            if "record" in locals():
                record["capture_status"] = "failed"
            self._failed(error)
            raise

    def state(self):
        frame = copy.deepcopy(self.latest)
        for camera in frame["cameras"]:
            camera.update(
                url="/artifacts/" + camera["image"],
                source_url="/artifacts/" + camera["source"] if camera.get("source") else None,
                pinhole_url="/artifacts/" + camera["pinhole"],
            )
            for key in ("surface", "alpha", "depth"):
                if camera.get(key):
                    camera[key + "_url"] = "/artifacts/" + camera[key]
        return {
            **frame,
            "world_tick": self.current["tick"],
            "dt": 0.1,
            "world_generation": self.generation,
            "latest_transition": self._transition(),
            "failure": copy.deepcopy(self.manifest.get("failure")),
            "sampling": copy.deepcopy(self.manifest.get("sampling")),
            "sensor_stride": self.sensor_stride,
            "status": self.manifest["status"],
            "sequence": len(self.manifest["frames"]),
            "scenario_id": self.metadata["scenario_id"],
            "map": self.map,
            "limits": self.manifest["limits"],
            "performance": self.manifest["performance"],
        }

    def snapshot(self):
        """Capture complete collection state; image dependencies remain on server."""
        if getattr(self, "recorded_mode", False):
            raise ValueError("Recorded rendering cannot create simulation checkpoints")
        from driversi.sim.persistence.session_checkpoint import snapshot_session

        return snapshot_session(self)

    def fork(self, output: Path, checkpoint=None):
        """Independent worker and recording; shared renderer is used sequentially."""
        if getattr(self, "recorded_mode", False):
            raise ValueError("Recorded rendering cannot create simulation forks")
        from driversi.sim.persistence.session_checkpoint import fork_session

        return fork_session(self, output, checkpoint)

    def close(self):
        if self.worker:
            self.worker.close()


def run_matched_mtgs(
    *,
    config_path: Path,
    checkpoint_path: Path,
    road_block_config: Path,
    scene: Path,
    road_graph: Path,
    binding: Path,
    world_python: Path,
    native_path: Path,
    output: Path,
    scale: float = 0.5,
    sensor_stride: int = 2,
    cameras: tuple[str, ...] = ("CAM_F0", "CAM_R0"),
    desired_speed: float = 13.0,
    paired_brake_actor: str | None = None,
    serve_port: int | None = None,
    trusted_assets: bool = False,
    virtual_camera_actors: tuple[str, ...] = (),
    height_mode: str = "source-time",
    population_config: Path | None = None,
    visual_policy: bool = False,
    visual_actors: tuple[str, ...] = (),
    visual_history: int = 2,
    visual_adapter=None,
    image_write_workers: int = 1,
) -> dict:
    image_write_workers = validate_image_write_workers(image_write_workers)
    if type(visual_policy) is not bool or (
        (visual_actors or visual_adapter is not None) and not visual_policy
    ):
        raise ValueError("visual_actors/adapter requires visual_policy")
    from PIL import Image

    from driversi.rendering.gaussian.mtgs_live_viewer import write_session_viewer
    from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime

    if population_config is not None:
        document = json.loads(Path(population_config).read_text())
        if any(
            policy.get("controller") == "external-joint-v1"
            for policy in document.get("policies", [])
        ):
            raise ValueError(
                "Automatic CLI rollout requires rule policies; use MTGSWorldSession.advance for external actions"
            )
    if visual_policy:
        from driversi.contracts.population import EpisodePopulation
        from driversi.sim.sensors.visual_sampling import visual_population

        preflight_scene = Scene.load(scene)
        original = (
            EpisodePopulation.load(population_config) if population_config is not None else None
        )
        visual_population(
            tuple(
                a
                for a, active in zip(preflight_scene.actor_ids, preflight_scene.controlled)
                if active
            ),
            (*(("ego",) if "CAM_F0" in cameras else ()), *virtual_camera_actors),
            visual_actors,
            original,
            desired_speed,
            visual_adapter,
        )
    started = time.perf_counter()
    runtime = MTGSRuntime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        road_block_config=road_block_config,
        scale=scale,
        trusted_assets=trusted_assets,
    )
    session = MTGSWorldSession(
        runtime=runtime,
        scene=scene,
        road_graph=road_graph,
        binding=binding,
        world_python=world_python,
        native_path=native_path,
        output=output,
        cameras=cameras,
        sensor_stride=sensor_stride,
        desired_speed=desired_speed,
        virtual_camera_actors=virtual_camera_actors,
        height_mode=height_mode,
        population_config=population_config,
        visual_policy=visual_policy,
        visual_actors=visual_actors,
        visual_history=visual_history,
        visual_adapter=visual_adapter,
        image_write_workers=image_write_workers,
    )
    try:
        calibration_checks = []
        for name, seeds in session._indices.items():
            for index in dict.fromkeys((seeds[0][1], seeds[-1][1])):
                _, official = runtime.source_sample(index)
                fast = runtime.source_camera(index)
                fields = ("camera_to_worlds", "fx", "fy", "cx", "cy", "width", "height", "times")
                exact = {
                    field: bool(
                        np.array_equal(
                            _array(getattr(official, field)), _array(getattr(fast, field))
                        )
                    )
                    for field in fields
                }
                if not all(exact.values()) or not camera_metadata_equal(
                    official.metadata, fast.metadata
                ):
                    raise ValueError(
                        f"Camera-only setup differs from official source loader: {name}"
                    )
                calibration_checks.append(
                    {
                        "camera_id": name,
                        "dataset_index": index,
                        "exact_fields": exact,
                        "metadata_equal": True,
                    }
                )
        session.manifest["camera_calibration_checks"] = calibration_checks
        session.manifest["image_encoding"] = {
            "format": "lossless_png",
            "compress_level": 1,
            "source_images": "one_copy_per_source_seed_shared_across_generations",
            "transition_journal": "append_fsync_before_render_transitions.jsonl",
        }
        session.manifest["startup_wall_seconds"] = time.perf_counter() - started
        if paired_brake_actor and paired_brake_actor not in session.current["controlled_ids"]:
            raise ValueError("Paired intervention actor is not controlled")
        session.manifest["protocol"] = {
            "kind": (
                "managed_visual_sampling"
                if visual_policy
                else "paired_rule_interaction"
                if paired_brake_actor
                else "rule_interaction"
            ),
            "brake_actor": paired_brake_actor,
            "intervention_start_tick": 10,
            "intervention_end_tick_exclusive": 20,
            "override_action": 1,
            "action_semantics": "longitudinal_jerk_minus15_lateral_jerk_zero",
            "rgb_is_policy_input": visual_policy,
            "policy_inputs_recorded": visual_policy,
        }
        branch_seconds = []
        for branch in range(2 if paired_brake_actor else 1):
            if branch:
                session.reset()
            branch_start = time.perf_counter()
            while not session.current["done"]:
                tick = session.current["tick"]
                overrides = {paired_brake_actor: 1} if branch and 10 <= tick < 20 else {}
                session.advance(overrides)
            branch_seconds.append(time.perf_counter() - branch_start)
        session.manifest["performance"]["branch_end_to_end_wall_seconds"] = branch_seconds
        session.manifest["performance"]["branch_timing_scope"] = (
            "all_advance_calls_including_manifest_writes_no_sleep"
        )
        session.manifest["performance"]["end_to_end_real_time_factor"] = session.manifest[
            "performance"
        ]["physics_seconds"] / sum(branch_seconds)
        if paired_brake_actor:
            first = [f for f in session.manifest["frames"] if f["generation"] == 0]
            second = [f for f in session.manifest["frames"] if f["generation"] == 1]
            image_checks = []
            for a, b in zip(first[0]["cameras"], second[0]["cameras"]):
                with (
                    Image.open(session.output / a["image"]) as left,
                    Image.open(session.output / b["image"]) as right,
                ):
                    equal = np.array_equal(np.asarray(left), np.asarray(right))
                image_checks.append(
                    {"camera_id": a["camera_id"], "initial_rgb_equal_after_reset": equal}
                )
            steps = session.manifest["steps"]
            baseline = {s["tick"]: s for s in steps if s["generation"] == 0}
            intervention = {s["tick"]: s for s in steps if s["generation"] == 1}
            first_changed = {}
            for tick in sorted(baseline.keys() & intervention.keys()):
                for actor in session.current["controlled_ids"]:
                    if (
                        baseline[tick]["policy_actions"][actor]
                        != intervention[tick]["policy_actions"][actor]
                    ):
                        first_changed.setdefault(
                            actor, tick - 1
                        )  # policy observes the preceding state
            session.manifest["paired_checks"] = {
                "initial_actors_equal": first[0]["actors"] == second[0]["actors"],
                "images": image_checks,
                "first_changed_policy_observation_tick": first_changed,
                "scope": "one_real_source_paired_intervention_not_general_safety_or_learning_validation",
            }
        _write_manifest(session.output, session.manifest, compact=True)
        # The immutable paired gallery stays available while a live generation runs.
        snapshot = session.output / "paired-manifest.json"
        snapshot.write_text(
            json.dumps(session.manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        write_session_viewer(snapshot)
        result = {
            "output": str(session.output),
            "source_log": session.metadata["source_log"],
            "controlled_actors": len(session.current["controlled_ids"]),
            "frames": len(session.manifest["frames"]),
            "trainable": False,
            "performance": session.manifest["performance"],
            "paired_checks": session.manifest.get("paired_checks"),
        }
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        if serve_port is not None:
            from driversi.rendering.gaussian.mtgs_live import serve_session

            session.reset()
            serve_session(session, port=serve_port)
        return result
    except BaseException as error:
        session._failed(error)
        raise
    finally:
        session.close()
