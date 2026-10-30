"""Opt-in multi-camera rendering of one physical instant; scalar is unchanged."""

import numpy as np

from driversi.rendering.gaussian.mtgs_registration import motion_in_source, move_camera
from driversi.rendering.gaussian.worldengine_portable import (
    PortableRenderer,
    camera_calibration,
    camera_pose,
    ego_center_pivot,
    supported_log_pose,
)


class BatchedPortableRenderer(PortableRenderer):
    """Reuse decoded geometry and optics while preserving per-camera SH colors.

    No admission is granted here. Each call decodes actor poses afresh, so a
    previous branch's transported global_means cannot become its new origin.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optics = {}

    def _camera_optics(self, camera, width):
        import cv2

        intrinsic, distortion = camera_calibration(camera, width)
        key = (width, tuple(intrinsic.ravel()), tuple(distortion))
        if key not in self._optics:
            height = round(width * 1080 / 1920)
            new_k, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (width, height), 1)
            mx, my = cv2.initInverseRectificationMap(
                intrinsic, distortion, None, new_k, (width, height), cv2.CV_32FC1
            )
            yy, xx = np.mgrid[0:height, 0:width]
            pixels = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
            rays = cv2.undistortPoints(pixels[:, None], intrinsic, distortion).reshape(-1, 2)
            if not np.isfinite(rays).all():
                raise ValueError("Nonfinite inverse-distortion output rays")
            low, high = rays.min(0) - 0.01, rays.max(0) + 0.01
            bounds = [float(low[0]), float(high[0]), float(low[1]), float(high[1])]
            self._optics[key] = (intrinsic, distortion, new_k, mx, my, bounds)
        return self._optics[key]

    def render_many(self, source, camera_ids, *, motions=None, absent_physical=(), width=960):
        import cv2
        import torch
        from gsplat.rendering import rasterization

        source = self.binding.resolve(source)
        camera_ids = tuple(camera_ids)
        if not camera_ids or len(camera_ids) > 4 or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("One to four distinct cameras required")
        if type(width) is not int or width <= 0:
            raise ValueError("Positive integer output width required")
        motions = motions or {}
        absent_physical = set(absent_physical)
        objects = {
            k.removeprefix("rigid_object_")
            for k, model in self.models.items()
            if hasattr(model, "static_in_log")
        }
        if not absent_physical <= objects or absent_physical & set(motions):
            raise ValueError("Physical absence must reference existing unmoved object assets")
        if set(motions) - objects - {"ego"}:
            raise ValueError("Motion references missing reconstructed assets")
        source_rear_axle = np.asarray(source["ego2global"], dtype=float)
        if source_rear_axle.shape != (4, 4) or not np.isfinite(source_rear_axle).all():
            raise ValueError("Finite reconstruction rear-axle pose required")
        terrain_evidence = {}

        def lift(token, center, translation):
            if self.terrain is None:
                terrain_evidence[token] = {"status": "unverified", "reason": "terrain_not_bound"}
                return translation
            from driversi.rendering.gaussian.worldengine_geometry import (
                UnsupportedTerrain,
                terrain_failure_evidence,
            )

            try:
                corrected, evidence = self.terrain.lift(center, translation)
            except UnsupportedTerrain as error:
                terrain_evidence[token] = terrain_failure_evidence(error)
                return translation
            terrain_evidence[token] = evidence
            return corrected

        cameras, poses, extrinsics, optics = [], [], [], []
        for camera_id in camera_ids:
            camera = source["cams"][camera_id]
            if camera.get("valid") is not True:
                raise ValueError("Source camera is invalid")
            pose = camera_pose(source, camera)
            homogeneous = np.eye(4)
            homogeneous[:3] = pose
            extrinsics.append(np.linalg.solve(source_rear_axle, homogeneous))
            if "ego" in motions:
                delta = motions["ego"]
                pivot = ego_center_pivot(source)
                translation, _, _ = motion_in_source(source, delta[:2], delta[2])
                corrected = lift("ego", pivot, translation)
                pose = move_camera(source, pose, pivot, delta[:2], delta[2])
                pose[:, 3] += corrected - translation
            cameras.append(camera)
            poses.append(pose)
            optics.append(self._camera_optics(camera, width))

        stamp = int(source["timestamp"])
        chunks = {k: [] for k in ("means", "quats", "scales", "opacities")}
        color_chunks = [[] for _ in camera_ids]
        active, absent, moved = [], [], []
        poses_t = [torch.as_tensor(p, device=self.device, dtype=torch.float32) for p in poses]
        with torch.no_grad():
            for name, model in self.models.items():
                token = name.rsplit("_", 1)[-1]
                if hasattr(model, "static_in_log"):
                    if token in absent_physical:
                        continue
                    pose = supported_log_pose(model, stamp)
                    if pose is None:
                        if token in motions:
                            raise ValueError(
                                "Controlled object has no source-time appearance: " + token
                            )
                        absent.append(token)
                        continue
                    q, t = pose
                    gs = {
                        "means": model.get_means(q, t),
                        "quats": model.get_quats(q, t),
                        "scales": model.get_scales(),
                        "opacities": model.get_opacity(),
                    }
                    if token in motions:
                        delta = motions[token]
                        translation, rotation, dq = motion_in_source(source, delta[:2], delta[2])
                        translation = lift(token, t.detach().cpu().numpy(), translation)
                        rotation = torch.as_tensor(
                            rotation, device=self.device, dtype=gs["means"].dtype
                        )
                        translation = torch.as_tensor(
                            translation, device=self.device, dtype=gs["means"].dtype
                        )
                        gs["means"] = (gs["means"] - t) @ rotation.T + t + translation
                        from driversi.rendering.gaussian.worldengine_native.utils.gaussian_utils import (
                            quat_mult,
                        )

                        gs["quats"] = quat_mult(
                            torch.as_tensor(dq, device=self.device, dtype=q.dtype)[None],
                            gs["quats"],
                        )
                        model.global_means = gs["means"]
                        moved.append(token)
                    active.append(token)
                else:
                    gs = model.get_global_gaussians()
                for key, value in gs.items():
                    chunks[key].append(value)
                # The original native color functions use a single-camera SH
                # layout. Preserve it, batching only the rasterization call.
                for i, pose_t in enumerate(poses_t):
                    color_chunks[i].append(
                        model.get_gaussian_rgbs(pose_t[None], timestamp=stamp, device=self.device)
                    )
            tensors = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
            tensors["colors"] = torch.stack([torch.cat(v, dim=0) for v in color_chunks])
            full = torch.stack([torch.eye(4, device=self.device) for _ in poses_t])
            for i, pose_t in enumerate(poses_t):
                full[i, :3] = pose_t
            height = round(width * 1080 / 1920)
            # Invert each 4x4 exactly as the scalar path before stacking.
            views = torch.stack([torch.linalg.inv(matrix) for matrix in full])
            rgb, alpha, _ = rasterization(
                **tensors,
                viewmats=views,
                Ks=torch.stack(
                    [torch.as_tensor(o[2], device=self.device, dtype=torch.float32) for o in optics]
                ),
                width=width,
                height=height,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                rasterize_mode="antialiased",
                radius_clip=0.0,
            )
            pixels = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            alphas = alpha.cpu().numpy()

        result = {}
        for i, camera_id in enumerate(camera_ids):
            camera = cameras[i]
            intrinsic, distortion, _, mx, my, bounds = optics[i]
            image = cv2.remap(pixels[i], mx, my, cv2.INTER_LINEAR)
            remapped_alpha = cv2.remap(alphas[i], mx, my, cv2.INTER_LINEAR)
            result[camera_id] = (
                image,
                {
                    "asset_id": self.binding.asset_id,
                    "source_log": self.binding.source_log,
                    "source_token": source["token"],
                    "timestamp_us": stamp,
                    "camera_id": camera_id,
                    "active_object_tokens": list(active),
                    "absent_at_source_tokens": list(absent),
                    "excluded_by_physical_presence": sorted(absent_physical),
                    "world_moved_tokens": list(moved),
                    "low_alpha_fraction": float(np.mean(remapped_alpha < 0.5)),
                    "camera_to_world": poses[i].tolist(),
                    "camera_to_rear_axle": extrinsics[i].tolist(),
                    "camera_to_rear_axle_reference": "reconstruction_refined_nuplan_rear_axle",
                    "pose_reference": "vehicle_center",
                    "camera_time_policy": "frame_timestamp_not_shutter_timestamp",
                    "native_calibration_size": [1920, 1080],
                    "output_image_size": [width, height],
                    "calibration_source": "reconstruction_colmap_pose_intrinsics_distortion",
                    "distorted_intrinsic": intrinsic.tolist(),
                    "distortion": distortion.tolist(),
                    "output_ray_bounds": bounds,
                    "source_image_relative_path": camera["data_path"],
                    "source_exposure_factor": float(camera.get("v_adjust", 1.0)),
                    "terrain_evidence": dict(terrain_evidence),
                    "terrain_fully_supported": bool(terrain_evidence)
                    and all(v["status"] == "supported" for v in terrain_evidence.values()),
                    "quality_admitted": False,
                    "scope": "rendered_pixels_not_quality_certificate",
                },
            )
        return result
