"""Portable WorldEngine asset renderer with explicit source-frame pose transport.

Uses owned copies of upstream Gaussian parameter/color functions. It does not
invent missing objects, extrapolate object lifetime, or certify pixel quality.
"""

import pickle
from pathlib import Path

import numpy as np

from driversi.data.assets.worldengine import SourceBinding
from driversi.rendering.gaussian.mtgs_registration import motion_in_source, move_camera


def ego_center_pivot(source):
    """nuPlan physical ego origin is 1.461 m ahead of the rear axle."""
    pose = np.asarray(source["ego2global"], dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Finite refined ego transform required")
    return pose[:3, 3] + pose[:3, 0] * 1.461


def camera_pose(source, camera):
    colmap = camera.get("colmap_param", {})
    if "sensor2global_rotation" not in colmap or "sensor2global_translation" not in colmap:
        raise ValueError("Reconstruction-refined camera pose is required")
    rotation = colmap["sensor2global_rotation"]
    rotation = (
        rotation.rotation_matrix if hasattr(rotation, "rotation_matrix") else np.asarray(rotation)
    )
    pose = np.c_[rotation, np.asarray(colmap["sensor2global_translation"])]
    if pose.shape != (3, 4) or not np.isfinite(pose).all():
        raise ValueError("Malformed refined camera pose")
    return pose


def camera_calibration(camera, width):
    """Refined pose and optics must belong to the same reconstruction bundle."""
    colmap = camera.get("colmap_param", {})
    if "cam_intrinsic" not in colmap or "distortion" not in colmap:
        raise ValueError("Reconstruction-refined camera optics are required")
    intrinsic = np.array(colmap["cam_intrinsic"], dtype=float)
    distortion = np.array(colmap["distortion"], dtype=float)
    if (
        intrinsic.shape != (3, 3)
        or distortion.shape != (5,)
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(distortion).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.array_equal(intrinsic[2], [0, 0, 1])
    ):
        raise ValueError("Malformed refined camera optics")
    intrinsic[:2] *= width / 1920
    return intrinsic, distortion


def supported_log_pose(model, timestamp):
    """Do not let upstream nearest-index fallback extrapolate a missing actor."""
    import torch

    if model.static_in_log:
        return model.log_quats, model.log_trans
    stamps = model.config.log_timestamps.reshape(-1)
    if timestamp < stamps[0].item() or timestamp > stamps[-1].item():
        return None
    upper = min(
        int(torch.searchsorted(stamps, stamps.new_tensor(timestamp)).item()), len(stamps) - 1
    )
    lower = upper if stamps[upper].item() == timestamp else max(0, upper - 1)
    if any(model.log_trans[i, 2].item() >= 1000 for i in {lower, upper}):
        return None
    q, t, _ = model._get_log_pose_from_timestamp(timestamp)
    return q, t


class PortableRenderer:
    def __init__(self, checkpoint, *, metadata, terrain=None, device="cuda:0"):
        import torch

        from driversi.rendering.gaussian.worldengine_geometry import RoadHeightMap
        from driversi.rendering.gaussian.worldengine_native.gaussian_model.rigid_object import (
            RigidPortableSubModel,
        )
        from driversi.rendering.gaussian.worldengine_native.gaussian_model.rigid_object_mirrored import (
            MirroredRigidPortableSubModel,
        )
        from driversi.rendering.gaussian.worldengine_native.gaussian_model.vanilla_gaussian_splatting import (
            VanillaPortableGaussianModel,
        )
        from driversi.rendering.gaussian.worldengine_native.utils.portable_utils import (
            convert_to_attribute_dict,
        )

        self.terrain = RoadHeightMap.load(terrain) if terrain is not None else None
        self.device = torch.device(device)
        raw = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        mapping = {
            "VanillaGaussianSplattingModel": VanillaPortableGaussianModel,
            "SkyboxGaussianSplattingModel": VanillaPortableGaussianModel,
            "RigidSubModel": RigidPortableSubModel,
            "MirroredRigidSubModel": MirroredRigidPortableSubModel,
        }
        self.models = {}
        self._ray_bounds = {}
        self.origin = np.asarray(raw["background"]["config"]["recon2world_translation"])
        with Path(metadata).open("rb") as stream:
            self.binding = SourceBinding(Path(checkpoint).stem, pickle.load(stream), self.origin)
        for key, asset in raw.items():
            kind = asset["config"]["type"]
            if kind not in mapping:
                raise ValueError("Unsupported asset model: " + kind)
            self.models[key] = mapping[kind](
                asset=convert_to_attribute_dict(asset), model_name=key
            ).to(self.device)

    def render(self, source, camera_id, *, motions=None, absent_physical=(), width=960):
        """motions contains measured physical world dx,dy,dyaw for ego/object IDs.

        Empty motions reproduce the reconstructed source. A caller must bind
        motions to a replayed physical branch; this class grants no admission.
        """
        import cv2
        import torch
        from gsplat.rendering import rasterization

        source = self.binding.resolve(source)
        if not isinstance(width, int) or width <= 0:
            raise ValueError("Positive integer output width required")
        motions = motions or {}
        absent_physical = set(absent_physical)
        object_tokens = {
            k.removeprefix("rigid_object_")
            for k, model in self.models.items()
            if hasattr(model, "static_in_log")
        }
        if not absent_physical <= object_tokens or absent_physical & set(motions):
            raise ValueError("Physical absence must reference existing unmoved object assets")
        unknown = (
            set(motions)
            - {"ego"}
            - {
                k.removeprefix("rigid_object_")
                for k, model in self.models.items()
                if hasattr(model, "static_in_log")
            }
        )
        if unknown:
            raise ValueError(
                "Motion references missing reconstructed assets: " + str(sorted(unknown))
            )
        camera = source["cams"][camera_id]
        if camera.get("valid") is not True:
            raise ValueError("Source camera is invalid")
        c2w = camera_pose(source, camera)
        # Reconstruction metadata retains the nuPlan rear-axle ego origin.
        # Derive rigid extrinsics before counterfactual camera/terrain transport.
        source_rear_axle = np.asarray(source["ego2global"], dtype=float)
        if source_rear_axle.shape != (4, 4) or not np.isfinite(source_rear_axle).all():
            raise ValueError("Finite reconstruction rear-axle pose required")
        camera_homogeneous = np.eye(4)
        camera_homogeneous[:3] = c2w
        camera_to_rear_axle = np.linalg.solve(source_rear_axle, camera_homogeneous)
        terrain_evidence = {}

        def lift(token, center, translation):
            if self.terrain is None:
                terrain_evidence[token] = {"status": "unverified", "reason": "terrain_not_bound"}
                return translation
            from driversi.rendering.gaussian.worldengine_geometry import UnsupportedTerrain

            try:
                corrected, evidence = self.terrain.lift(center, translation)
            except UnsupportedTerrain as ex:
                # Keep diagnostic output possible, but never mislabel this as
                # finite terrain support or admit it downstream.
                from driversi.rendering.gaussian.worldengine_geometry import terrain_failure_evidence

                terrain_evidence[token] = terrain_failure_evidence(ex)
                return translation
            terrain_evidence[token] = evidence
            return corrected

        if "ego" in motions:
            delta = motions["ego"]
            pivot = ego_center_pivot(source)
            translation, _, _ = motion_in_source(source, delta[:2], delta[2])
            corrected = lift("ego", pivot, translation)
            c2w = move_camera(source, c2w, pivot, delta[:2], delta[2])
            c2w[:, 3] += corrected - translation
        stamp = int(source["timestamp"])
        chunks = {k: [] for k in ("means", "quats", "scales", "opacities", "colors")}
        active, absent, moved = [], [], []
        transports = {}
        c2w_t = torch.as_tensor(c2w, device=self.device, dtype=torch.float32)
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
                        from driversi.rendering.diagnostics.dynamic_binding import source_registration

                        binding = source_registration(source, token, gs["means"].detach().cpu().numpy())
                        binding["source_pivot"] = t.detach().cpu().numpy().tolist()
                        # Learned object quaternions are not exactly unit length.
                        # Transport the native decoded cloud, preserving its shape
                        # instead of normalizing/reinterpreting those parameters.
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
                        binding.update(applied_planar_delta=list(delta),
                                       applied_translation=translation.detach().cpu().numpy().tolist(),
                                       decoded_target_center=gs["means"].mean(0).detach().cpu().numpy().tolist())
                        transports[token] = binding
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
                colors = model.get_gaussian_rgbs(c2w_t[None], timestamp=stamp, device=self.device)
                for key, value in gs.items():
                    chunks[key].append(value)
                chunks["colors"].append(colors)
            tensors = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
            h = round(width * 1080 / 1920)
            intrinsic, distortion = camera_calibration(camera, width)
            new_k, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (width, h), 1)
            mx, my = cv2.initInverseRectificationMap(
                intrinsic, distortion, None, new_k, (width, h), cv2.CV_32FC1
            )
            full = torch.eye(4, device=self.device)
            full[:3] = c2w_t
            rgb, alpha, _ = rasterization(
                **tensors,
                viewmats=torch.linalg.inv(full)[None],
                Ks=torch.as_tensor(new_k, device=self.device, dtype=torch.float32)[None],
                width=width,
                height=h,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                rasterize_mode="antialiased",
                radius_clip=0.0,
            )
            pixels = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            alpha = alpha[0].cpu().numpy()
        # Bound all discrete distorted output-pixel rays, including the image
        # perimeter. The margin enlarges the frustum and can only retain more
        # unresolved objects, never exclude a marginal one.
        cache_key = (width, tuple(intrinsic.ravel()), tuple(distortion))
        if cache_key not in self._ray_bounds:
            yy, xx = np.mgrid[0:h, 0:width]
            pixels_xy = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
            rays = cv2.undistortPoints(pixels_xy[:, None], intrinsic, distortion).reshape(-1, 2)
            if not np.isfinite(rays).all():
                raise ValueError("Nonfinite inverse-distortion output rays")
            low, high = rays.min(0) - 0.01, rays.max(0) + 0.01
            self._ray_bounds[cache_key] = [
                float(low[0]),
                float(high[0]),
                float(low[1]),
                float(high[1]),
            ]
        image = cv2.remap(pixels, mx, my, cv2.INTER_LINEAR)
        alpha = cv2.remap(alpha, mx, my, cv2.INTER_LINEAR)
        return image, {
            "asset_id": self.binding.asset_id,
            "source_log": self.binding.source_log,
            "source_token": source["token"],
            "timestamp_us": stamp,
            "camera_id": camera_id,
            "active_object_tokens": active,
            "absent_at_source_tokens": absent,
            "excluded_by_physical_presence": sorted(absent_physical),
            "world_moved_tokens": moved,
            "physical_object_transport": transports,
            "low_alpha_fraction": float(np.mean(alpha < 0.5)),
            "camera_to_world": c2w.tolist(),
            "camera_to_rear_axle": camera_to_rear_axle.tolist(),
            "camera_to_rear_axle_reference": "reconstruction_refined_nuplan_rear_axle",
            "pose_reference": "vehicle_center",
            "camera_time_policy": "frame_timestamp_not_shutter_timestamp",
            "native_calibration_size": [1920, 1080],
            "output_image_size": [width, h],
            "calibration_source": "reconstruction_colmap_pose_intrinsics_distortion",
            "distorted_intrinsic": intrinsic.tolist(),
            "distortion": distortion.tolist(),
            "output_ray_bounds": self._ray_bounds[cache_key],
            "source_image_relative_path": camera["data_path"],
            "source_exposure_factor": float(camera.get("v_adjust", 1.0)),
            "terrain_evidence": terrain_evidence,
            "terrain_fully_supported": bool(terrain_evidence)
            and all(value["status"] == "supported" for value in terrain_evidence.values()),
            "quality_admitted": False,
            "scope": "rendered_pixels_not_quality_certificate",
        }
