"""Bounded, read-only inspection of audited candidates and original NAVSIM sensors."""
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np

from driversi.data.sources.navsim import load_log, transform_box
from driversi.data.scenes.source_gallery import select_source_window
from driversi.data.datasets.tto_audit import json_read, scene_path, sha256, verified_rows

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")


def contained(root, relative):
    root = Path(root).resolve()
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Expected relative asset")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Asset escapes configured root")
    return path


def navsim_yaw(q):
    """Match NAVSIM's pyquaternion intrinsic yaw, including nonzero roll/pitch."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("Invalid source quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return float(np.arctan2(2*(w*z-x*y), 1-2*(y*y+z*z)))


def local_path_to_map(trajectory, yaw):
    path = np.asarray(trajectory, dtype=float).copy()
    c, s = np.cos(yaw), np.sin(yaw)
    path[:, :2] = path[:, :2] @ np.array([[c, s], [-s, c]])
    path[:, 2] += yaw
    return path.tolist()


class TTOInspector:
    def __init__(self, audit, dataset, *, trusted=False):
        if not trusted:
            raise ValueError("Explicit trusted NAVSIM pickle permission required")
        self.audit = json_read(audit)
        if (self.audit.get("kind") != "tto_shortfall_audit" or self.audit.get("schema_version") != 1):
            raise ValueError("Unknown candidate audit")
        self.records = {r["token"]: r for r in self.audit["scenes"]}
        if len(self.records) != len(self.audit["scenes"]):
            raise ValueError("Duplicate scene identity")
        self.bank = Path(self.audit["bank_root"])
        self.context_root = Path(audit).resolve().parent / "map-context"
        self.dataset = Path(dataset).resolve()
        self.cache = OrderedDict()
        self.lock = RLock()
        self.diagnostics = {}
        diagnostic_path = Path(audit).resolve().parent / "controlled-scores.json"
        if diagnostic_path.is_file():
            diagnostics = json_read(diagnostic_path)
            if (diagnostics.get("kind") != "tto_controlled_score_diagnostics"
                    or diagnostics.get("audit_sha256") != sha256(audit)):
                raise ValueError("Score contrasts belong to another audit")
            self.diagnostics = {case["token"]: case for case in diagnostics["cases"]}


    def index(self, protected=()):
        fields = ("token", "split", "log_name", "category", "queries", "qualified_unique", "best_pdms",
                  "expert_labels", "gt_high_but_candidates_low", "component_zero_counts", "best_labels")
        scenes = [{k: r[k] for k in fields} for r in self.records.values() if r["log_name"] not in protected]
        for row in scenes:
            case = self.diagnostics.get(row["token"])
            row["diagnosis"] = None
            if case and case["status"] == "verified":
                values = case["results"]
                row["diagnosis"] = ("missing_zero" if values["zero_offset_warp"]["labels"][-1] >= .95
                    else "heading" if values["zero_offset_gt_heading"]["labels"][-1] >= .95
                    else "unresolved")
        return {"scenes": scenes, "components": self.audit["components"],
                "summary": self.audit["summary"], "hidden_reserved": len(self.records)-len(scenes),
                "generated_at": self.audit["generated_at"], "read_only": True}

    def _get(self, token, protected=()):
        if token not in self.records or self.records[token]["log_name"] in protected:
            raise ValueError("Unknown or reserved scene")
        with self.lock:
            if token not in self.cache:
                self.cache[token] = self._load(self.records[token])
                while len(self.cache) > 4:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(token)
            return self.cache[token]

    def _image_path(self, camera):
        relative = camera.get("data_path")
        if not isinstance(relative, str):
            return None
        # Hist and future stores are separate in some NAVSIM installations.
        for name in ("trainval", "trainval_all", "trainval_fut"):
            path = contained(self.dataset / "sensor_blobs" / name, relative)
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return path
        return None

    def _load(self, record):
        directory = scene_path(self.bank, record["split"], record["token"])
        rows, digest = verified_rows(directory)
        if (digest != record["candidate_sha256"] or
                sha256(directory / "contract.json") != record["contract_sha256"] or
                sha256(directory / "expert.json") != record["expert_sha256"]):
            raise ValueError("Source bank changed since this audit; publish a new audit")
        log = record["log_name"]
        if Path(log).name != log:
            raise ValueError("Unsafe log identity")
        source = None
        for name in ("trainval", "trainval_all", "trainval_fut"):
            candidate = contained(self.dataset / "navsim_logs" / name, log + ".pkl")
            if candidate.is_file():
                source = candidate
                break
        if source is None:
            raise ValueError("Source log not available")
        frames = load_log(source, trusted=True)
        indices = [i for i, f in enumerate(frames) if str(f["token"]) == record["token"]]
        if len(indices) != 1:
            raise ValueError("Source anchor is not unique")
        anchor = indices[0]
        window = select_source_window(frames, anchor-3, 12)
        origin = np.asarray(window[3]["ego2global_translation"], dtype=float)
        yaw = navsim_yaw(window[3]["ego2global_rotation"])
        error = np.r_[origin[:2], yaw] - np.asarray(record["ego_pose"], dtype=float)
        error[-1] = np.arctan2(np.sin(error[-1]), np.cos(error[-1]))
        if int(window[3]["timestamp"]) != record["timestamp_us"] or np.max(np.abs(error)) > 1e-5:
            raise ValueError("Candidate and raw source anchor differ")
        images, output = {}, []
        for index, frame in enumerate(window):
            translation = np.asarray(frame["ego2global_translation"], dtype=float)
            angle = navsim_yaw(frame["ego2global_rotation"])
            objects = []
            for track, kind, box in zip(frame["anns"]["track_tokens"], frame["anns"]["gt_names"],
                                        frame["anns"]["gt_boxes"], strict=True):
                pose = transform_box(box, translation, angle)
                pose[:3] -= origin
                objects.append({"id": str(track), "kind": str(kind), "pose": pose.tolist(),
                                "dimensions": np.asarray(box[3:6]).tolist()})
            available = []
            for camera_id in CAMERAS:
                camera = frame.get("cams", {}).get(camera_id, {})
                image = self._image_path(camera)
                if image is not None:
                    images[index, camera_id] = image
                    available.append(camera_id)
            output.append({"source_index": anchor-3+index, "token": str(frame["token"]),
                           "time_seconds": (int(frame["timestamp"])-record["timestamp_us"])/1e6,
                           "ego_pose": [*map(float, translation[:2]-origin[:2]), angle],
                           "objects": objects, "cameras": available,
                           "traffic_lights": [[str(i), bool(red)] for i, red in frame.get("traffic_lights", [])]})
        from driversi.data.scenes.context import read_map_context
        # Bounded diagnostic ROI; no objects are deleted from the response.
        gt_map = local_path_to_map(record["expert_gt"], yaw)
        points = np.asarray([f["ego_pose"][:2] for f in output] + [p[:2] for p in gt_map])
        bounds = np.r_[points.min(axis=0)-35, points.max(axis=0)+35]
        map_error = None
        gt_map = local_path_to_map(record["expert_gt"], yaw)
        points = np.asarray([f["ego_pose"][:2] for f in output] + [p[:2] for p in gt_map])
        bounds = np.r_[points.min(axis=0)-35, points.max(axis=0)+35]
        source_digest = sha256(source)
        context_path = self.context_root / (record["token"] + ".json")
        cached = json_read(context_path) if context_path.is_file() else None
        if cached is not None:
            if (cached["source_sha256"] != source_digest or cached["map_name"] != window[3]["map_location"]
                    or cached["bounds"] != bounds.tolist() or cached["origin_xyz"] != origin.tolist()):
                raise ValueError("Cached map context belongs to another source")
            features, map_error = cached["features"], cached["error"]
        else:
            try:
                features, _, _ = read_map_context(window[3]["map_location"], self.dataset / "maps", origin, bounds)
            except (ValueError, OSError, ImportError) as error:
                features, map_error = [], str(error)
        angles = np.asarray([f["ego_pose"][2] for f in output[4:]])
        raw_gt = np.c_[np.asarray([f["ego_pose"][:2] for f in output[4:]]), angles]
        gt_error = raw_gt-np.asarray(gt_map)
        gt_error[:, 2] = np.arctan2(np.sin(gt_error[:, 2]), np.cos(gt_error[:, 2]))
        data = {"record": record, "frames": output, "map_name": window[3]["map_location"],
                "map": features, "map_error": map_error, "bounds": bounds.tolist(), "yaw": yaw,
                "gt_map": gt_map, "source_sha256": source_digest, "origin_xyz": origin.tolist(),
                "source_future_vs_gt_max_xy_m": float(np.max(np.linalg.norm(gt_error[:, :2], axis=1))),
                "source_future_vs_gt_max_heading_rad": float(np.max(np.abs(gt_error[:, 2]))),
                "cameras": list(CAMERAS), "source_photos_not_candidate_render": True,
                "candidate_is_plan_not_closed_loop_execution": True,
                "diagnostic_exposure_only": True, "trainable": False}
        return data, rows, images

    def scene(self, token, protected=()):
        data, rows, _ = self._get(token, protected)
        diagnostic = self.diagnostics.get(token)
        contrasts = {}
        if diagnostic and diagnostic["status"] == "verified":
            contrasts = {name: {**value, "row": None, "diagnostic": name,
                         "map_trajectory": local_path_to_map(value["trajectory"], data["yaw"])}
                         for name, value in diagnostic["results"].items()}
        return {**data, "controlled_contrasts": contrasts, "candidates": {"scores": rows["labels"][:, -1].tolist(),
                                      "offsets": (rows["keys"]*.001).tolist()},
                "candidate": self.candidate(token, data["record"]["best_row"], protected)}

    def candidate(self, token, index, protected=()):
        data, rows, _ = self._get(token, protected)
        if type(index) is not int or not 0 <= index < len(rows):
            raise ValueError("Unknown candidate row")
        row = rows[index]
        return {"row": index, "labels": row["labels"].tolist(), "trajectory": row["trajectories"].tolist(),
                "map_trajectory": local_path_to_map(row["trajectories"], data["yaw"]),
                "offset_m": (row["keys"]*.001).tolist()}

    def image(self, token, frame, camera, protected=()):
        _, _, images = self._get(token, protected)
        if type(frame) is not int or not 0 <= frame < 12 or camera not in CAMERAS:
            raise ValueError("Invalid camera request")
        path = images.get((frame, camera))
        if path is None or not path.is_file() or path.stat().st_size > 20*1024*1024:
            raise ValueError("Image unavailable")
        return path.read_bytes(), "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
