"""Full native MTGS rendering of sealed executed windows.

Preserves original checkpoint camera correction, appearance model and Gaussian
parameters. Independent admission remains outside this rendering worker.
"""

import argparse
import json
import pickle
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def asset_inputs(config, checkpoint, road_block, travel_id):
    import yaml

    from driversi.agents.backends.native import digest

    if type(travel_id) is not int or travel_id < 0:
        raise ValueError("Explicit nonnegative traversal required")
    road = yaml.load(Path(road_block).read_text(), Loader=yaml.BaseLoader)
    block = road["road_block_name"]
    folder = Path(road["data_root"]) / block
    prefix = folder / "rgb_point_cloud" / f"{block}-{travel_id}"
    paths = [
        Path(config),
        Path(checkpoint),
        Path(road_block),
        folder / "video_scene_dict.pkl",
        prefix.with_suffix(".pcd"),
        prefix.with_name(prefix.name + "_sem_labels.npy"),
    ]
    support_file = Path(road_block).parent / "terrain-support.json"
    if support_file.exists():
        support = json.loads(support_file.read_text())
        references = support.get("reference_travel_ids", [])
        if (
            support.get("source_travel_id") != travel_id
            or not references
            or len(references) != len(set(references))
            or travel_id in references
            or any(type(t) is not int or t < 0 for t in references)
        ):
            raise ValueError("Independent terrain traversal manifest differs")
        paths.append(support_file)
        if support.get("registration_certificate"):
            certificate = Path(support["registration_certificate"]).resolve(strict=True)
            registration = json.loads(certificate.read_text())
            if registration.get("source_fit_traversal_unused") != travel_id or {
                registration.get("reference_traversal"),
                registration.get("aligned_traversal"),
            } != set(references):
                raise ValueError("Surface registration traversals differ from bound assets")
            paths.append(certificate)
            for key in ("calibration_evidence", "experiment_exclusion_evidence"):
                evidence = Path(registration[key]).resolve(strict=True)
                if digest(evidence) != registration[key + "_sha256"]:
                    raise ValueError("Surface registration evidence changed")
                paths.append(evidence)
        for reference in references:
            prefix = folder / "rgb_point_cloud" / f"{block}-{reference}"
            paths.extend(
                (prefix.with_suffix(".pcd"), prefix.with_name(prefix.name + "_sem_labels.npy"))
            )
    return {str(p.resolve(strict=True)): digest(p) for p in paths}


def cv_pose(pose):
    pose = np.array(pose, dtype=float, copy=True)
    if pose.shape != (3, 4) or not np.isfinite(pose).all():
        raise ValueError("Finite native camera pose required")
    pose[:, :3] = pose[:, :3] @ np.diag([1.0, -1.0, -1.0])
    return pose


def pixel_ray_bounds(intrinsic, distortion, width=960, height=540):
    import cv2

    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx.ravel() + 0.5, yy.ravel() + 0.5), axis=1)
    rays = cv2.undistortPointsIter(
        pixels[:, None],
        np.asarray(intrinsic),
        np.asarray(distortion),
        None,
        None,
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 50, 1e-10),
    ).reshape(-1, 2)
    if not np.isfinite(rays).all():
        raise ValueError("Invalid output source-lens rays")
    lo, hi = rays.min(0) - 0.01, rays.max(0) + 0.01
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]


def quaternion_product(left, right):
    """Broadcast Hamilton product; preserve original quaternion magnitudes."""
    import torch

    w1, xyz1 = left[..., :1], left[..., 1:]
    w2, xyz2 = right[..., :1], right[..., 1:]
    return torch.cat(
        (
            w1 * w2 - (xyz1 * xyz2).sum(-1, keepdim=True),
            w1 * xyz2 + w2 * xyz1 + torch.cross(xyz1.expand_as(xyz2), xyz2, dim=-1),
        ),
        -1,
    )


@contextmanager
def decoded_cloud_motion(models, transforms):
    """Rigidly transport native decoded clouds, including nonunit static poses.

    No checkpoint parameter is normalized or mutated. Original source rendering
    remains untouched when no transport is requested. All method/cache edits
    are restored on exceptions as well as successful captures.
    """
    import types

    import torch

    from driversi.rendering.gaussian.mtgs import clear_rigid_cache

    saved = []
    try:
        for token, (
            translation,
            rotation,
            delta_q,
            original_q,
            center,
            frame_index,
        ) in transforms.items():
            node = models["rigid_object_" + token].node
            original = node.get_gaussians
            local = node.__dict__.get("get_gaussians")
            saved.append((node, local))

            def wrap(original, translation, rotation, delta_q, original_q, center, frame_index):
                def transformed(self, camera_to_worlds, *args, **kwargs):
                    # Decode the explicitly bound observed frame. Retained actors may
                    # be absent from the current source frame; checkpoint presence
                    # and learned pose parameters remain untouched.
                    kwargs = dict(kwargs)
                    kwargs["frame_idx"] = frame_index
                    if hasattr(self, "dataframe_dict"):
                        kwargs["timestamp"] = self.dataframe_dict["frame_timestamps"][frame_index]
                    values = original(camera_to_worlds, *args, **kwargs)
                    if values is None:
                        return None
                    values = dict(values)
                    means = values["means"]
                    xyz = means.new_tensor(center)
                    shift = means.new_tensor(translation)
                    turn = means.new_tensor(rotation)
                    dq = means.new_tensor(delta_q)
                    means = (means - xyz) @ turn.T + xyz + shift
                    values["means"] = means
                    values["quats"] = quaternion_product(dq[None], values["quats"])
                    if "rgbs" in values:
                        q = quaternion_product(dq, means.new_tensor(original_q))
                        stamp = kwargs.get("timestamp")
                        if stamp is None:
                            stamp = self.dataframe_dict["frame_timestamps"][frame_index]
                        values["rgbs"] = self.get_rgbs(
                            camera_to_worlds, q, xyz + shift, stamp, global_current_means=means
                        )
                    if "velocities" in values:
                        values["velocities"] = values["velocities"] @ turn.T
                    return values

                return transformed

            node.get_gaussians = types.MethodType(
                wrap(original, translation, rotation, delta_q, original_q, center, frame_index),
                node,
            )
            clear_rigid_cache(node)
        with torch.no_grad():
            yield
    finally:
        for node, previous in reversed(saved):
            if previous is None:
                delattr(node, "get_gaussians")
            else:
                node.get_gaussians = previous
            clear_rigid_cache(node)


class NativeModelView:
    def __init__(self, node, runtime, travel_id, rigid=False):
        self.node, self.runtime, self.travel_id = node, runtime, travel_id
        if rigid:
            self.static_in_log = bool(node.is_static)

    def get_global_gaussians(self):
        return {"means": self.node.get_means()}

    def get_opacity(self):
        return self.node.get_opacity()


class NativeTrainingRenderer:
    backend = "mtgs_native"

    def __init__(self, config, checkpoint, road_block, travel_id, *, terrain=True):
        import yaml

        from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime

        self.backend_config = {
            "config": str(Path(config).resolve()),
            "checkpoint": str(Path(checkpoint).resolve()),
            "road_block": str(Path(road_block).resolve()),
            "travel_id": travel_id,
        }
        self.asset_files = asset_inputs(config, checkpoint, road_block, travel_id)
        self.runtime = MTGSRuntime(
            config_path=config,
            checkpoint_path=checkpoint,
            road_block_config=road_block,
            source_travel_id=travel_id,
            scale=0.5,
            trusted_assets=True,
        )
        self.travel_id = travel_id
        road = yaml.load(Path(road_block).read_text(), Loader=yaml.BaseLoader)
        with self.runtime.metadata_path.open("rb") as stream:
            videos = pickle.load(stream)
        self.video_id = road["road_block_name"] + "-" + str(travel_id)
        video = videos[self.video_id]
        self.origin = np.asarray(video["global2world_translation"])
        self.frames = {f["token"]: f for f in video["frame_infos"] if not f.get("skipped")}
        for source in self.frames.values():
            if any(
                "colmap_param" not in source["cams"][cam] or "v_adjust" not in source["cams"][cam]
                for cam in CAMERAS
            ):
                raise ValueError("Final optimized source metadata required; no raw fallback")
        self.binding = SimpleNamespace(
            asset_id=road["road_block_name"], source_log=video["log_name"], resolve=self.resolve
        )
        self.indices = {
            (str(t), Path(p).parent.name): i
            for i, (t, p) in enumerate(
                zip(self.runtime.sources.frame_tokens, self.runtime.sources.image_filenames)
            )
            if int(self.runtime.sources.travel_ids[i]) == travel_id
        }
        self.by_timestamp = {int(f["timestamp"]): f for f in self.frames.values()}
        self.models = {
            name: NativeModelView(
                node, self.runtime, travel_id, rigid=name.startswith("rigid_object_")
            )
            for name, node in self.runtime.model.gaussian_models.items()
            if name == "background"
            or (name.startswith("rigid_object_") and int(node.travel_id) == travel_id)
        }
        self.terrain = None
        if terrain:
            from driversi.rendering.gaussian.mtgs_road_surface import (
                IndependentlyValidatedRoadSurface,
                MeasuredRoadSurface,
                TraversalConsensusRoadSurface,
            )

            pcd_paths = [Path(p) for p in self.asset_files if p.endswith(".pcd")]
            source_pcd = next(p for p in pcd_paths if p.stem.endswith("-" + str(travel_id)))

            def read_surface(path):
                return MeasuredRoadSurface.from_pcd(
                    path, path.with_name(path.stem + "_sem_labels.npy")
                )

            measured = read_surface(source_pcd)
            support_file = Path(road_block).parent / "terrain-support.json"
            support = json.loads(support_file.read_text()) if support_file.exists() else {}
            if support.get("registration_certificate"):
                from driversi.rendering.gaussian.mtgs_surface_registration import (
                    load_registered_road_surface,
                )

                certificate = support["registration_certificate"]
                registration = json.loads(Path(certificate).read_text())
                reference_pcd = next(
                    p
                    for p in pcd_paths
                    if p.stem.endswith("-" + str(registration["reference_traversal"]))
                )
                moving_pcd = next(
                    p
                    for p in pcd_paths
                    if p.stem.endswith("-" + str(registration["aligned_traversal"]))
                )

                def labels(p):
                    return p.with_name(p.stem + "_sem_labels.npy")

                self.terrain = load_registered_road_surface(
                    source_pcd,
                    labels(source_pcd),
                    reference_pcd,
                    labels(reference_pcd),
                    moving_pcd,
                    labels(moving_pcd),
                    certificate,
                )
            elif support.get("method") == "traversal_consensus_measured_road_v1":
                self.terrain = TraversalConsensusRoadSurface(
                    measured, [read_surface(p) for p in pcd_paths if p != source_pcd]
                )
            elif support_file.exists():
                independent = [read_surface(p) for p in pcd_paths if p != source_pcd]
                if not independent:
                    raise ValueError("Independent measured terrain clouds absent")
                combined = MeasuredRoadSurface(
                    np.concatenate([x.points for x in independent]),
                    {"disjoint_traversals": [x.provenance for x in independent]},
                )
                self.terrain = IndependentlyValidatedRoadSurface(measured, combined)
            else:
                self.terrain = measured
        self._rays = {}
        self._physics = None
        self._static_anchors = {}
        self._source_geometry = {}

    def resolve(self, token):
        return self.frames[token] if isinstance(token, str) else self.frames[token["token"]]

    def bind_physics(self, scene, states, presence):
        """Bind an immutable executed branch before selecting native anchors."""
        self._physics = (scene, states, presence)
        self._static_anchors = {}
        self._source_geometry = {}

    def asset_anchor(self, model, timestamp):
        source = self.by_timestamp.get(int(timestamp))
        if source is None:
            return None
        token = next(
            (
                name.removeprefix("rigid_object_")
                for name, view in self.models.items()
                if view is model
            ),
            None,
        )
        if token is None:
            return None
        available = []
        for item in self.frames.values():
            if int(item["timestamp"]) > int(timestamp) or token not in item["track_tokens"]:
                continue
            if self._physics is not None:
                scene = self._physics[0]
                if token in scene.actor_ids:
                    source_tick = int(
                        round(
                            (int(item["timestamp"]) - scene.metadata["source_start_timestamp_us"])
                            / 100000
                        )
                    )
                    if (
                        not 0 <= source_tick < len(scene.times)
                        or source_tick % 5 != 0
                        or not scene.presence_at(source_tick)[scene.actor_ids.index(token)]
                    ):
                        continue
            index = self.indices[(item["token"], "CAM_F0")]
            frame_index = int(self.runtime.sources.frame_ids[index])
            if frame_index < model.node.num_frames and bool(model.node.in_frame_mask[frame_index]):
                available.append((int(item["timestamp"]), item, frame_index))
        if not available:
            return None
        available.sort(key=lambda value: value[0])
        if self._physics is None:
            # Canonical source reconstruction must retain unmodified native time.
            return available[-1] if available[-1][0] == int(timestamp) else None
        scene, _, presence = self._physics
        if token not in scene.actor_ids:
            return available[-1] if available[-1][0] == int(timestamp) else None
        if model.static_in_log:
            first = np.flatnonzero(presence[:, scene.actor_ids.index(token)])
            if not len(first):
                return None
            start = scene.metadata["source_start_timestamp_us"] + int(first[0]) * 100000
            if int(timestamp) < start - 50000:
                return available[-1] if available[-1][0] == int(timestamp) else None
            if token not in self._static_anchors:
                # First actual observation at/after physical entry, never a future
                # frame relative to this capture. Earlier history is used only if
                # the actor was already retained at physical entry.
                observed = [item for item in available if item[0] >= start - 50000]
                self._static_anchors[token] = observed[0] if observed else available[-1]
            return self._static_anchors[token]
        return available[-1]

    def supported_log_pose(self, model, timestamp):
        anchor = self.asset_anchor(model, timestamp)
        if anchor is None:
            return None
        q, t = model.node.get_object_pose(anchor[2], None)
        return None if q is None else (q, t)

    def physical_binding(self, token, model, source):
        from driversi.rendering.gaussian.mtgs_bindings import compiled_observed_pose
        from driversi.rendering.gaussian.worldengine_replay import physical_delta

        anchor = self.asset_anchor(model, source["timestamp"])
        if anchor is None or self._physics is None:
            return None
        scene, states, presence = self._physics
        if token not in scene.actor_ids:
            return None
        tick = int(
            round((int(source["timestamp"]) - scene.metadata["source_start_timestamp_us"]) / 100000)
        )
        index = scene.actor_ids.index(token)
        if not 0 <= tick < len(states) or not presence[tick, index]:
            return None
        reference, reference_tick = compiled_observed_pose(
            anchor[1], token, scene, native_centering=True
        )
        return dict(
            actor_id=token,
            source_token=anchor[1]["token"],
            source_timestamp_us=anchor[0],
            native_frame_index=anchor[2],
            reference_tick=reference_tick,
            reference_contract="native_centered_scene_observation_v2",
            target_timestamp_us=int(source["timestamp"]),
            target_tick=tick,
            reference_physical_pose=reference.tolist(),
            target_physical_pose=states[tick, index, :4].tolist(),
            planar_delta=physical_delta(states[tick, index], reference),
            policy="immutable_static_observation_v1"
            if model.static_in_log
            else "latest_observed_asset_v1",
            checkpoint_presence_unchanged=True,
        )

    def render(self, source, camera_id, *, motions=None, absent_physical=(), width=960):
        from driversi.rendering.gaussian.mtgs import _array
        from driversi.rendering.gaussian.mtgs_registration import (
            motion_in_source,
            move_camera,
        )
        from driversi.rendering.gaussian.mtgs_runtime import project_source_lens
        from driversi.rendering.gaussian.worldengine_geometry import UnsupportedTerrain
        from driversi.rendering.gaussian.worldengine_portable import ego_center_pivot

        if width != 960:
            raise ValueError("Native training pilot preserves fixed 960x540 camera sampling")
        source = self.resolve(source)
        motions = motions or {}
        index = self.indices[(source["token"], camera_id)]
        camera = self.runtime.source_camera(index)
        pose = _array(self.runtime.model.camera_optimizer.apply_to_camera(camera))[0]
        source_pose = pose.copy()
        terrain_evidence = {}

        def lift(token, center, delta):
            if self.terrain is None:
                terrain_evidence[token] = {"status": "unverified", "reason": "terrain_not_bound"}
                return delta
            try:
                adjusted, evidence = self.terrain.lift(center, delta)
            except UnsupportedTerrain as ex:
                from driversi.rendering.gaussian.worldengine_geometry import terrain_failure_evidence

                terrain_evidence[token] = terrain_failure_evidence(ex)
                return delta
            terrain_evidence[token] = evidence
            return adjusted

        active, absent, transforms, bindings = [], [], {}, {}
        for name, model in self.models.items():
            if not hasattr(model, "static_in_log"):
                continue
            token = name.removeprefix("rigid_object_")
            if token in absent_physical:
                continue
            original = self.supported_log_pose(model, source["timestamp"])
            if original is None:
                absent.append(token)
                continue
            active.append(token)
            if token in motions:
                binding = self.physical_binding(token, model, source)
                anchor = self.asset_anchor(model, source["timestamp"])
                transport_source = source if binding is None else anchor[1]
                delta = motions[token] if binding is None else binding["planar_delta"]
                if binding is not None:
                    bindings[token] = binding
                q, t = map(_array, original)
                if binding is not None:
                    from driversi.rendering.gaussian.worldengine_geometry import (
                        background_physical_box_support,
                    )

                    geometry_key = (token, binding["source_token"])
                    if geometry_key not in self._source_geometry:
                        physical_scene = self._physics[0]
                        from driversi.rendering.gaussian.mtgs_bindings import (
                            observed_3d_box_corners,
                        )

                        corners, source_obb = observed_3d_box_corners(
                            transport_source,
                            token,
                            physical_scene.dimensions[physical_scene.actor_ids.index(token)],
                        )
                        native_q, native_t = original
                        points = _array(model.node.get_means(native_q, native_t))
                        support = background_physical_box_support(
                            points, _array(model.node.get_opacity()), corners
                        )
                        axis = np.stack(
                            (
                                corners[4] - corners[0],
                                corners[2] - corners[0],
                                corners[1] - corners[0],
                            ),
                            axis=1,
                        )
                        lengths = np.linalg.norm(axis, axis=0)
                        center_local = (t - corners.mean(0)) @ (axis / lengths)
                        support["source_3d_observation"] = source_obb
                        support["native_center_inside_physical_source_box"] = bool(
                            np.all(np.abs(center_local) <= lengths / 2)
                        )
                        support["native_center_local_m"] = center_local.tolist()
                        support["scope"] = "source_obb_geometry_not_semantic_visual_admission"
                        self._source_geometry[geometry_key] = support
                    binding["source_obb_registration"] = self._source_geometry[geometry_key]
                translation, _, _ = motion_in_source(transport_source, delta[:2], delta[2])
                identity = None
                if binding is not None:
                    from driversi.rendering.gaussian.mtgs_bindings import exact_float32_identity

                    identity = exact_float32_identity(
                        binding["reference_physical_pose"], binding["target_physical_pose"]
                    )
                if identity is not None and identity["equal"]:
                    # The decoded source asset remains exactly where it was;
                    # no new surface height is inferred from a road-only map.
                    translation = np.zeros(3)
                    delta = [0.0, 0.0, 0.0]
                    adjusted = translation
                    terrain_evidence[token] = dict(
                        status="source_identity",
                        method="source_identity_transport_v1",
                        identity=identity,
                        source_token=binding["source_token"],
                        reference_tick=binding["reference_tick"],
                        scope="unchanged_source_geometry_requires_separate_obb_visual_review",
                    )
                else:
                    adjusted = lift(token, t, translation)
                current_frame = int(self.runtime.sources.frame_ids[index])
                frame_index = current_frame if anchor is None else anchor[2]
                if (
                    np.max(np.abs([*translation, delta[2]])) <= 1e-10
                    and frame_index == current_frame
                ):
                    continue
                _, rotation, dq = motion_in_source(transport_source, delta[:2], delta[2])
                transforms[token] = (adjusted, rotation, dq, q, t, frame_index)
        if set(motions) - set(active) - {"ego"}:
            raise ValueError("World motion has no active matching source asset")
        if "ego" in motions:
            delta = motions["ego"]
            pivot = ego_center_pivot(source)
            translation, _, _ = motion_in_source(source, delta[:2], delta[2])
            adjusted = lift("ego", pivot, translation)
            pose = move_camera(source, pose, pivot, delta[:2], delta[2])
            pose[:, 3] += adjusted - translation
        if transforms:
            # Let the native rasterizer test moved assets; stale source culling
            # cannot hide a moved actor that enters the new camera frustum.
            camera.metadata["visible_tokens"] = list(self.runtime.model.gaussian_models)
        with decoded_cloud_motion(self.models, transforms):
            native = self.runtime.render_camera(
                camera,
                target_camera_to_world=pose if motions else None,
                exclude_tokens=tuple(absent_physical),
            )
        projection = self.runtime.source_lens_projection(camera)
        converted = project_source_lens(native, projection)
        effective = cv_pose(native["camera_to_world"])
        source_cv = cv_pose(source_pose)
        homogeneous = np.eye(4)
        homogeneous[:3] = source_cv
        extrinsic = np.linalg.solve(np.asarray(source["ego2global"]), homogeneous)
        key = (tuple(np.asarray(projection["K"]).ravel()), tuple(projection["distortion"]))
        if key not in self._rays:
            self._rays[key] = pixel_ray_bounds(projection["K"], projection["distortion"])
        return converted["rgb"], dict(
            asset_id=self.binding.asset_id,
            source_log=self.binding.source_log,
            source_token=source["token"],
            timestamp_us=int(source["timestamp"]),
            camera_id=camera_id,
            active_object_tokens=active,
            absent_at_source_tokens=absent,
            excluded_by_physical_presence=list(absent_physical),
            world_moved_tokens=list(transforms),
            object_transport="decoded_native_cloud_rigid_v2",
            object_bindings=bindings,
            low_alpha_fraction=float(np.mean(converted["accumulation"] < 0.5)),
            lens_support_fraction=float(converted["valid"].mean()),
            camera_to_world=effective.tolist(),
            canonical_source_camera_to_world=source_cv.tolist(),
            camera_to_rear_axle=extrinsic.tolist(),
            camera_to_rear_axle_reference="reconstruction_refined_nuplan_rear_axle",
            pose_reference="vehicle_center",
            camera_time_policy="frame_timestamp_not_shutter_timestamp",
            native_calibration_size=list(projection["raw_size"]),
            output_calibration_size=list(projection["output_size"]),
            calibration_source="native_checkpoint_optimized_camera_and_colmap_optics",
            distorted_intrinsic=projection["K"],
            distortion=projection["distortion"],
            output_ray_bounds=self._rays[key],
            source_image_relative_path=source["cams"][camera_id]["data_path"],
            source_exposure_factor=float(source["cams"][camera_id]["v_adjust"]),
            native_photometric=native["photometric"],
            terrain_evidence=terrain_evidence,
            terrain_support_semantics="measured_lift_or_proven_unchanged_native_geometry",
            terrain_fully_supported=bool(terrain_evidence)
            and all(
                v["status"] in {"supported", "source_identity"} for v in terrain_evidence.values()
            ),
            quality_admitted=False,
            scope="rendered_pixels_not_quality_certificate",
        )


def run(scene, rollout, raw_log, config, checkpoint, road_block, travel_id, output, *, future_points=10):
    from driversi.rendering.gaussian.worldengine_replay import run as render_window

    if type(future_points) is not int or future_points not in (8, 10):
        raise ValueError("Native future points must be eight or ten")
    renderer = NativeTrainingRenderer(config, checkpoint, road_block, travel_id)
    return render_window(
        scene,
        rollout,
        Path(road_block).parent,
        raw_log,
        output,
        future_points=future_points,
        renderer=renderer,
        asset_files=list(renderer.asset_files),
    )


def source_run(render_report, output):
    from PIL import Image

    from driversi.agents.backends.native import digest
    from driversi.contracts.scene import Scene
    from driversi.rendering.diagnostics.worldengine_admission import source_view_psnr

    report = json.loads(Path(render_report).read_text())
    if report.get("renderer_backend") != "mtgs_native":
        raise ValueError("Native MTGS branch required")
    for path, sha in report["inputs"].items():
        if digest(path) != sha:
            raise ValueError("Native branch evidence changed")
    renderer = NativeTrainingRenderer(**report["backend_config"], terrain=False)
    scene = Scene.load(next(Path(p) for p in report["inputs"] if Path(p).name == "scene.npz"))
    sensor = Path(scene.metadata["navsim_source"]["sensor_root"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frames, files = [], {}
    for frame in report["frames"]:
        source = renderer.resolve(frame["source_token"])
        for cam in CAMERAS:
            pixels, evidence = renderer.render(source, cam)
            path = output / f"{frame['tick']:04d}-{cam}-source-render.png"
            Image.fromarray(pixels).save(path)
            real = sensor / source["cams"][cam]["data_path"]
            factor = float(source["cams"][cam]["v_adjust"])
            files[str(path)] = digest(path)
            files[str(real)] = digest(real)
            frames.append(
                dict(
                    tick=frame["tick"],
                    camera=cam,
                    source_token=source["token"],
                    rendered_source_image=str(path),
                    real_source_image=str(real),
                    exposure_factor=factor,
                    measured_psnr_db=source_view_psnr(path, real, factor),
                    source_camera_to_world=evidence["camera_to_world"],
                    source_motions={},
                    source_appearance_timestamp_us=int(source["timestamp"]),
                    native_photometric=evidence["native_photometric"],
                )
            )
        print("native source rendered", frame["tick"], flush=True)
    code = Path(__file__).resolve()
    code_paths = [code, code.with_name("mtgs_runtime.py"), code.with_name("mtgs_registration.py")]
    value = dict(
        schema_version=1,
        kind="worldengine_source_comparison",
        renderer_backend="mtgs_native",
        render_report_sha256=digest(render_report),
        inputs={str(Path(render_report)): digest(render_report), **renderer.asset_files},
        files=files,
        producer_code={str(p): digest(p) for p in code_paths},
        source_mode="canonical_source_pose_without_motion_v1",
        frames=frames,
    )
    (output / "report.json").write_text(json.dumps(value, indent=2))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "scene",
        "rollout",
        "raw-log",
        "config",
        "checkpoint",
        "road-block",
        "source-render-report",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--future-points", type=int, choices=(8, 10), default=10)
    parser.add_argument("--travel-id", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch

    torch.cuda.set_per_process_memory_fraction(0.25, 0)
    if args.source_render_report:
        source_run(args.source_render_report, args.output)
    else:
        run(
            args.scene,
            args.rollout,
            args.raw_log,
            args.config,
            args.checkpoint,
            args.road_block,
            args.travel_id,
            args.output,
            future_points=args.future_points,
        )
